//! A wasmtime host for the engine as a core module, with run-time translation: for an
//! embedder that keeps the personality native and runs only the engine as Wasm. (It is not a
//! host for the node's `translate:jit` import; nothing in this repository serves that.)
//!
//! **Embedder surface, unstable**: no program in this repository uses [`EngineInstance`]; the
//! node runs its engine in-process and `wasmshim-run` uses only [`blob`] from this crate.
//!
//! The host side of the engine core ABI ([`wasmshim_cpu_remote::abi`]): an [`EngineInstance`]
//! instantiates a guest-supplied engine module (no imports allowed) in its **own** [`Store`],
//! forwards calls to it, and serves its compile requests: it compiles the translated module,
//! checks that it imports nothing but the engine's own exports, instantiates it against the
//! engine instance (same memory, same table) and appends its functions to the engine's table.
//! The host never learns the guest ISA or what the calls mean. Nothing here depends on the
//! embedder.
//!
//! The store runs with its own [`wasmtime::Config`] ([`engine_config`]): no fuel (the
//! guest-visible preemption is the engine's retired-instruction budget, never Wasm fuel) and
//! NaN canonicalisation on.

#![deny(unsafe_code)]

pub mod blob;

use std::collections::HashMap;
use std::time::{Duration, Instant};

use wasmshim_cpu_remote::abi;
use wasmshim_cpu_remote::{Link, Reply};
use wasmtime::{
    Config, Engine, ExternType, Instance, Linker, Memory, Module, Ref, Store, StoreLimits,
    StoreLimitsBuilder, Table, TypedFunc,
};

/// Instances one engine store may hold: the engine plus one per translated module (a region,
/// or a page's regions). Far above any run so far (a large JavaScript build translates about
/// 13 000 regions).
const MAX_INSTANCES: usize = 1 << 22;

/// Why an engine could not be created or a call failed.
#[derive(Debug, thiserror::Error)]
pub enum JitError {
    /// The engine module (or a module it asked to compile) breaks the engine core ABI.
    #[error("invalid engine module: {0}")]
    Invalid(String),
    /// The engine trapped or misbehaved during a call.
    #[error("engine failure: {0}")]
    Trap(String),
}

fn invalid(e: impl std::fmt::Display) -> JitError {
    JitError::Invalid(e.to_string())
}

fn trap(e: impl std::fmt::Display) -> JitError {
    JitError::Trap(format!("{e:#}"))
}

/// The wasmtime configuration for engine stores: Cranelift, no fuel, NaN canonicalisation.
pub fn engine_config() -> Config {
    let mut c = Config::new();
    c.cranelift_nan_canonicalization(true);
    c.wasm_multi_memory(false);
    // Translated code marks its slow paths cold (`metadata.code.branch_hint`): code
    // layout only, never semantics.
    c.wasm_branch_hinting(true);
    c
}

/// Host-side counters (never visible to the guest).
#[derive(Clone, Copy, Debug, Default)]
pub struct HostStats {
    /// Modules compiled by Cranelift.
    pub compiled: u64,
    /// Compile requests served from the in-memory module cache.
    pub cache_hits: u64,
    /// Wall time spent compiling and instantiating translated modules.
    pub compile_time: Duration,
    /// Bytes of translated Wasm compiled.
    pub wasm_bytes: u64,
    /// Calls into the engine.
    pub calls: u64,
}

/// An instantiated engine module and the translated code it asked for.
pub struct EngineInstance {
    store: Store<StoreLimits>,
    memory: Memory,
    table: Table,
    io: usize,
    call: TypedFunc<(), i32>,
    input: TypedFunc<i32, i32>,
    linker: Linker<StoreLimits>,
    /// Translated modules by content (same bytes, same code): FNV-1a hash → (bytes, module).
    cache: HashMap<u64, Vec<(Vec<u8>, Module)>>,
    /// Exports a translated module may import.
    importable: Vec<String>,
    stats: HostStats,
}

impl EngineInstance {
    /// Compiles and instantiates `module` (an engine core module) on `engine` (normally
    /// created from [`engine_config`]).
    pub fn new(engine: &Engine, module: &[u8]) -> Result<Self, JitError> {
        let module = Module::new(engine, module).map_err(invalid)?;
        Self::from_module(&module)
    }

    /// Instantiates an already compiled engine module (compile once, instantiate per guest).
    pub fn from_module(module: &Module) -> Result<Self, JitError> {
        let engine = module.engine();
        if let Some(i) = module.imports().next() {
            return Err(invalid(format!(
                "an engine module may not import anything (imports {}.{})",
                i.module(),
                i.name()
            )));
        }
        let importable = module
            .exports()
            .filter(|e| {
                matches!(
                    e.ty(),
                    ExternType::Func(_) | ExternType::Memory(_) | ExternType::Table(_)
                )
            })
            .map(|e| e.name().to_owned())
            .collect();
        // Every translated region is an instance in this store, so a long run passes wasmtime's default of 10 000 instances
        // per store; translated modules define no memories or tables of their own.
        let limits = StoreLimitsBuilder::new()
            .instances(MAX_INSTANCES)
            .tables(MAX_INSTANCES)
            .memories(MAX_INSTANCES)
            .build();
        let mut store = Store::new(engine, limits);
        store.limiter(|l| l);
        // No interruption unless sampling asks for it (engines without epochs ignore this).
        store.set_epoch_deadline(1 << 62);
        let instance = Instance::new(&mut store, module, &[]).map_err(trap)?;
        let memory = instance
            .get_memory(&mut store, abi::EXPORT_MEMORY)
            .ok_or_else(|| invalid("no memory export"))?;
        let table = instance
            .get_table(&mut store, abi::EXPORT_TABLE)
            .ok_or_else(|| invalid("no function table export"))?;
        let io_fn = instance
            .get_typed_func::<(), i32>(&mut store, abi::EXPORT_IO)
            .map_err(invalid)?;
        let call = instance
            .get_typed_func::<(), i32>(&mut store, abi::EXPORT_CALL)
            .map_err(invalid)?;
        let input = instance
            .get_typed_func::<i32, i32>(&mut store, abi::EXPORT_INPUT)
            .map_err(invalid)?;
        let io = io_fn.call(&mut store, ()).map_err(trap)? as u32 as usize;
        let mut linker = Linker::new(engine);
        linker
            .instance(&mut store, abi::IMPORT_MODULE, instance)
            .map_err(invalid)?;
        let mut me = Self {
            store,
            memory,
            table,
            io,
            call,
            input,
            linker,
            cache: HashMap::new(),
            importable,
            stats: HostStats::default(),
        };
        me.check_io()?;
        Ok(me)
    }

    fn check_io(&mut self) -> Result<(), JitError> {
        let len = self.memory.data_size(&self.store);
        if self
            .io
            .checked_add(abi::IO_SIZE)
            .is_none_or(|end| end > len)
        {
            return Err(invalid("the io block lies outside memory"));
        }
        Ok(())
    }

    /// Host-side counters.
    pub fn stats(&self) -> HostStats {
        self.stats
    }

    fn put_u32(&mut self, off: usize, v: u32) {
        let at = self.io + off;
        self.memory.data_mut(&mut self.store)[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn get_u32(&self, off: usize) -> u32 {
        let at = self.io + off;
        let d = self.memory.data(&self.store);
        u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]])
    }

    /// Performs one call (see [`wasmshim_cpu_remote::abi`]): `service(args, input)`, serving
    /// compile requests until the engine is done, then fills `reply`.
    pub fn call(
        &mut self,
        service: u32,
        args: &[u64],
        input: &[u8],
        reply: &mut Reply,
    ) -> Result<(), JitError> {
        self.stats.calls += 1;
        if args.len() > abi::MAX_WORDS {
            return Err(invalid("too many arguments"));
        }
        if !input.is_empty() {
            let ptr = self
                .input
                .call(&mut self.store, input.len() as i32)
                .map_err(trap)? as u32 as usize;
            let mem = self.memory.data_mut(&mut self.store);
            let dst = mem
                .get_mut(ptr..ptr + input.len())
                .ok_or_else(|| trap("input buffer outside memory"))?;
            dst.copy_from_slice(input);
        }
        {
            let io = self.io;
            let mem = self.memory.data_mut(&mut self.store);
            let block = &mut mem[io..io + abi::IO_SIZE];
            block[abi::IO_SERVICE..abi::IO_SERVICE + 4].copy_from_slice(&service.to_le_bytes());
            block[abi::IO_NARGS..abi::IO_NARGS + 4]
                .copy_from_slice(&(args.len() as u32).to_le_bytes());
            block[abi::IO_INPUT_LEN..abi::IO_INPUT_LEN + 4]
                .copy_from_slice(&(input.len() as u32).to_le_bytes());
            block[abi::IO_REQ_LEN..abi::IO_REQ_LEN + 4].copy_from_slice(&0u32.to_le_bytes());
            for (i, a) in args.iter().enumerate() {
                let o = abi::IO_ARGS + 8 * i;
                block[o..o + 8].copy_from_slice(&a.to_le_bytes());
            }
        }
        loop {
            match self.call.call(&mut self.store, ()).map_err(trap)? {
                abi::STATUS_DONE => break,
                abi::STATUS_COMPILE => {
                    let slot = self.compile_request()?;
                    self.put_u32(abi::IO_REQ_SLOT, slot);
                }
                s => return Err(trap(format!("unknown call status {s}"))),
            }
        }
        let nvalues = (self.get_u32(abi::IO_NVALUES) as usize).min(abi::MAX_WORDS);
        let out_ptr = self.get_u32(abi::IO_OUTPUT_PTR) as usize;
        let out_len = self.get_u32(abi::IO_OUTPUT_LEN) as usize;
        let mem = self.memory.data(&self.store);
        reply.values.clear();
        for i in 0..nvalues {
            let o = self.io + abi::IO_VALUES + 8 * i;
            reply.values.push(u64::from_le_bytes(
                mem[o..o + 8].try_into().expect("8 bytes"),
            ));
        }
        reply.output.clear();
        reply.output.extend_from_slice(
            mem.get(out_ptr..out_ptr + out_len)
                .ok_or_else(|| trap("output outside memory"))?,
        );
        Ok(())
    }

    /// Serves a compile request; returns the table slot of `f0` (or [`abi::SLOT_DECLINED`]).
    fn compile_request(&mut self) -> Result<u32, JitError> {
        let ptr = self.get_u32(abi::IO_REQ_PTR) as usize;
        let len = self.get_u32(abi::IO_REQ_LEN) as usize;
        let funcs = self.get_u32(abi::IO_REQ_FUNCS);
        let bytes = self
            .memory
            .data(&self.store)
            .get(ptr..ptr + len)
            .ok_or_else(|| trap("compile request outside memory"))?
            .to_vec();
        let started = Instant::now();
        let module = self.module_for(&bytes)?;
        let instance = self
            .linker
            .instantiate(&mut self.store, &module)
            .map_err(invalid)?;
        let base = self
            .table
            .grow(&mut self.store, u64::from(funcs), Ref::Func(None))
            .map_err(trap)?;
        for i in 0..funcs {
            let f = instance
                .get_func(&mut self.store, &format!("f{i}"))
                .ok_or_else(|| invalid(format!("translated module lacks export f{i}")))?;
            self.table
                .set(&mut self.store, base + u64::from(i), Ref::Func(Some(f)))
                .map_err(trap)?;
        }
        self.stats.compile_time += started.elapsed();
        u32::try_from(base).map_err(|_| trap("function table too large"))
    }

    /// The compiled module for `bytes`, from the cache or Cranelift.
    fn module_for(&mut self, bytes: &[u8]) -> Result<Module, JitError> {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        if let Some(m) = self
            .cache
            .get(&h)
            .and_then(|v| v.iter().find(|(b, _)| b == bytes))
        {
            self.stats.cache_hits += 1;
            return Ok(m.1.clone());
        }
        if let Some(dir) = std::env::var_os("LINUX_JIT_DUMP") {
            // Debugging aid: keep every translated module (named by content hash).
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(
                std::path::Path::new(&dir).join(format!("{h:016x}.wasm")),
                bytes,
            );
        }
        let module = Module::new(self.store.engine(), bytes).map_err(invalid)?;
        for i in module.imports() {
            let ok =
                i.module() == abi::IMPORT_MODULE && self.importable.iter().any(|n| n == i.name());
            if !ok {
                return Err(invalid(format!(
                    "a translated module may only import the engine's exports (imports {}.{})",
                    i.module(),
                    i.name()
                )));
            }
        }
        self.stats.compiled += 1;
        self.stats.wasm_bytes += bytes.len() as u64;
        self.cache
            .entry(h)
            .or_default()
            .push((bytes.to_vec(), module.clone()));
        Ok(module)
    }
}

impl Link for EngineInstance {
    fn call(&mut self, service: u32, args: &[u64], input: &[u8], reply: &mut Reply) {
        if let Err(e) = EngineInstance::call(self, service, args, input, reply) {
            panic!("jit engine: {e}");
        }
    }
}

/// [`engine_config`] plus epoch interruption, for [`EngineInstance::start_sampling`].
pub fn engine_config_sampling() -> Config {
    let mut c = engine_config();
    c.epoch_interruption(true);
    c
}

/// A wall-clock sampling profile of the engine store: innermost Wasm function per sample.
pub struct Sampler {
    counts: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, u64>>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Sampler {
    /// Stops sampling; returns (function, samples) by decreasing count.
    pub fn finish(mut self) -> Vec<(String, u64)> {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let mut v: Vec<(String, u64)> = self
            .counts
            .lock()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .unwrap_or_default();
        v.sort_by_key(|x| std::cmp::Reverse(x.1));
        v
    }
}

impl EngineInstance {
    /// Samples the innermost Wasm frame every `interval` (the store's engine must come from
    /// [`engine_config_sampling`]). Diagnostics only: wall-clock driven, never guest-visible.
    pub fn start_sampling(&mut self, interval: Duration) -> Sampler {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};
        let counts: Arc<Mutex<std::collections::BTreeMap<String, u64>>> = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        let c = counts.clone();
        self.store.set_epoch_deadline(1);
        self.store.epoch_deadline_callback(move |ctx| {
            let bt = wasmtime::WasmBacktrace::capture(&ctx);
            let name = bt
                .frames()
                .first()
                .map(|f| {
                    let n = f.func_name().unwrap_or("?");
                    if n.starts_with("linux_aot_region") {
                        "<translated region>".to_owned()
                    } else {
                        n.to_owned()
                    }
                })
                .unwrap_or_else(|| "<no frame>".to_owned());
            if let Ok(mut m) = c.lock() {
                *m.entry(name).or_default() += 1;
            }
            Ok(wasmtime::UpdateDeadline::Continue(1))
        });
        let engine = self.store.engine().clone();
        let s = stop.clone();
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                std::thread::sleep(interval);
                engine.increment_epoch();
            }
        });
        Sampler {
            counts,
            stop,
            thread: Some(thread),
        }
    }
}
