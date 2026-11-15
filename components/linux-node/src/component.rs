//! WASI glue: the [`Host`] over WASI 0.3, the run loop ([`run`]), its extension points
//! ([`NodeExt`]) and, with the default feature `export`, `wasi:cli/run@0.3.0`.
//!
//! An embedder that needs more than the plain node (another socket provider's extensions,
//! devices, a debugger, a paused start) builds its own component on this crate with
//! `default-features = false`, implements [`NodeExt`] and exports `wasi:cli/run` itself, calling
//! [`run`].

#![allow(unsafe_code)] // generated bindings

use std::cell::RefCell;
use std::io::{Read as _, Write as _};
use std::rc::Rc;

use wasip3::clocks::{monotonic_clock, system_clock};
use wasmshim_linux::{Arch, BlobError, Config, Frontend, Host, Kernel, RunStatus, Spawn, vfs::Vfs};

use crate::{CachedBlob, Options, parse_args};

#[cfg(not(feature = "blob-words"))]
wit_bindgen::generate!({ world: "node-host", path: "../../wit/linux-host" });
#[cfg(feature = "blob-words")]
wit_bindgen::generate!({ world: "node-host-words", path: "../../wit/linux-host" });

use linux::host::blob;

/// Reads `len` bytes of blob `name` at `offset`. With `blob-words` (a node composed with its
/// image provider) the bytes cross the component boundary as words: the adapter between two
/// components copies a list element by element, so bytes would cost one loop turn each.
pub fn blob_read(name: &str, offset: u64, len: u32) -> Result<Vec<u8>, BlobError> {
    #[cfg(feature = "blob-words")]
    let read = {
        linux::host::blob_words::read_words(name, offset, len).map(|words| {
            let mut bytes = Vec::with_capacity(words.len() * 8);
            for w in words {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            bytes.truncate(len as usize);
            bytes
        })
    };
    #[cfg(not(feature = "blob-words"))]
    let read = blob::read(name, offset, len);
    read.map_err(|e| match e {
        blob::Error::OutOfRange => BlobError::OutOfRange,
        _ => BlobError::Host,
    })
}

/// The size of blob `name`.
pub fn blob_size(name: &str) -> Result<u64, String> {
    blob::size(name).map_err(|e| format!("{e:?}"))
}

// Only a Wasm build exports (a native build of the cdylib cannot carry WIT export names).
#[cfg(all(feature = "export", target_family = "wasm"))]
wasip3::cli::command::export!(Node);

/// Exit status when the node itself fails (bad options, no image, guest deadlock).
const NODE_FAILURE: u8 = 125;

thread_local! {
    /// The node's clock minus the host's: zero, unless the node was restored from a state,
    /// whose instant its clock then reads at the moment of the restore.
    pub static SHIFT: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

fn shifted(t: u64, by: i64) -> u64 {
    (i128::from(t) + i128::from(by)).clamp(0, i128::from(u64::MAX)) as u64
}

/// The node's monotonic clock: the host's, plus [`SHIFT`].
pub fn device_now() -> u64 {
    shifted(monotonic_clock::now(), SHIFT.with(std::cell::Cell::get))
}

/// A node instant on the host's clock.
fn host_time(t: u64) -> u64 {
    shifted(t, -SHIFT.with(std::cell::Cell::get))
}

/// What a restore needs from [`run`]: the machine to replace, the loop to wake, and how to make
/// a fresh image mount, engine and configuration.
pub struct Restorer {
    pub kernel: Rc<RefCell<Kernel>>,
    pub shared: Rc<Loop>,
    #[allow(clippy::type_complexity)]
    pub make: Box<dyn Fn() -> Result<(Vfs, Box<dyn Frontend>, Config), String>>,
}

/// The run loop's extension points. Every method's default is the plain node's behaviour.
///
/// **Embedder surface, unstable**: the plain node overrides nothing here, and the methods follow
/// what embedders need (a paused start, a restore, devices), so they may change between
/// versions.
pub trait NodeExt {
    /// The personality's configuration, from the command line, before the kernel is built (and
    /// again for a restored one): what the embedder sets beyond the options.
    fn config(&self, cfg: &mut Config) {
        let _ = cfg;
    }

    /// The kernel exists; nothing has run yet.
    fn attach(&self, kernel: &Rc<RefCell<Kernel>>, opts: &Options) {
        let _ = (kernel, opts);
    }

    /// The socket provider's extensions (TCP abort and no-delay, ICMP echo sockets, socket
    /// checkpoints).
    #[cfg(feature = "net")]
    fn socket_ext(&self) -> Option<Rc<dyn crate::net::SocketExt>> {
        None
    }

    /// After the console, before the first program: attaches devices. Returns whether device
    /// input stays open (something may still wake a guest that waits without a deadline).
    fn start_devices(
        &self,
        kernel: &Rc<RefCell<Kernel>>,
        shared: &Rc<Loop>,
    ) -> Result<bool, String> {
        let _ = (kernel, shared);
        Ok(false)
    }

    /// Before each kernel run (device input).
    fn before_run(&self, kernel: &mut Kernel) {
        let _ = kernel;
    }

    /// After each kernel run (device output).
    fn after_run(&self, kernel: &mut Kernel) {
        let _ = kernel;
    }

    /// The node starts paused: it does not boot, it waits for a restore ([`NodeExt::park`]).
    fn paused(&self) -> bool {
        false
    }

    /// A paused node hands over what a restore needs.
    fn park(&self, restorer: Restorer) {
        let _ = restorer;
    }

    /// The paused node was restored: [`run`] goes on with the restored machine.
    fn restored(&self) -> bool {
        true
    }

    /// The personality drew `words` words of host randomness (8 bytes each, rounded up per
    /// call), just before the draw.
    fn drew_random(&self, words: u64) {
        let _ = words;
    }

    /// `run` is one app of a composed component: a failure is returned, never an `exit`
    /// (which would end the whole component).
    fn composed(&self) -> bool {
        false
    }

    /// An observer stop ([`Host::stop`], [`wasmshim_kernel::debug`]); the default resumes at once.
    fn stop(
        &self,
        stop: &wasmshim_kernel::debug::Stop,
        at: &mut dyn wasmshim_kernel::debug::StopTarget,
    ) -> wasmshim_kernel::debug::Resume {
        let _ = (stop, at);
        wasmshim_kernel::debug::Resume::default()
    }
}

/// The plain node: no extensions.
pub struct Plain;

impl NodeExt for Plain {}

/// The personality's view of the outside world: WASI 0.3 clocks and random, std stdio.
struct WasiHost {
    ext: Rc<dyn NodeExt>,
}

impl Host for WasiHost {
    fn monotonic_ns(&mut self) -> u64 {
        device_now()
    }
    fn realtime_ns(&mut self) -> u64 {
        let t = system_clock::now();
        let ns = (t.seconds as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(u64::from(t.nanoseconds));
        shifted(ns, SHIFT.with(std::cell::Cell::get))
    }
    fn random(&mut self, buf: &mut [u8]) {
        // The host serves 8 bytes per word, a call rounding up (a restore replays the count).
        self.ext.drew_random((buf.len() as u64).div_ceil(8));
        let bytes = wasip3::random::random::get_random_bytes(buf.len() as u64);
        let n = bytes.len().min(buf.len());
        buf[..n].copy_from_slice(&bytes[..n]);
    }
    fn stdout(&mut self, data: &[u8]) {
        // std's stdout is line-buffered: flush, or a prompt without a newline ("# ") would wait
        // for the next newline (a console host sees nothing until a key is typed).
        let mut out = std::io::stdout();
        let _ = out.write_all(data);
        let _ = out.flush();
    }
    fn stderr(&mut self, data: &[u8]) {
        let _ = std::io::stderr().write_all(data);
    }
    fn stdin(&mut self, buf: &mut [u8]) -> usize {
        std::io::stdin().read(buf).unwrap_or(0)
    }
    fn diag(&mut self, line: &str) {
        let _ = writeln!(std::io::stderr(), "[linux] {line}");
    }

    fn stop(
        &mut self,
        stop: &wasmshim_kernel::debug::Stop,
        at: &mut dyn wasmshim_kernel::debug::StopTarget,
    ) -> wasmshim_kernel::debug::Resume {
        self.ext.stop(stop, at)
    }
}

fn fail(msg: &str) -> ! {
    let _ = writeln!(std::io::stderr(), "linux-node: {msg}");
    let _ = std::io::stderr().flush();
    wasip3::cli::exit::exit_with_code(NODE_FAILURE);
    unreachable!("exit_with_code returned")
}

const DEADLOCK: &str = "guest deadlock: every thread is blocked with no deadline";

/// What wakes the run loop besides its own deadlines: the network pumps (`net` builds) and the
/// console pump.
#[cfg(feature = "net")]
pub type Loop = crate::net::Shared;

/// What wakes the run loop besides its own deadlines: the console pump.
#[cfg(not(feature = "net"))]
#[derive(Default)]
pub struct Loop {
    pub wake: crate::Notify,
    pub timers: crate::Timers,
}

#[cfg(not(feature = "net"))]
impl Loop {
    fn busy(&self) -> bool {
        false
    }
}

/// Console input read from `wasi:cli/stdin` by the console pump, for the run loop to hand to
/// the kernel.
#[derive(Default)]
struct ConsoleIn {
    data: std::cell::RefCell<Vec<u8>>,
    /// The stream ended.
    eof: std::cell::Cell<bool>,
    /// The kernel was told.
    eof_delivered: std::cell::Cell<bool>,
}

/// Reads the console's input (`wasi:cli/stdin@0.3.0`, raw keystrokes) until it ends.
async fn console_pump(console: std::rc::Rc<ConsoleIn>, shared: std::rc::Rc<Loop>) {
    use wasip3::wit_bindgen::StreamResult;
    let (mut rx, _done) = wasip3::cli::stdin::read_via_stream();
    loop {
        let (result, data) = rx.read(Vec::with_capacity(4096)).await;
        if !data.is_empty() {
            console.data.borrow_mut().extend_from_slice(&data);
            shared.wake.notify();
        }
        if !matches!(result, StreamResult::Complete(_)) {
            break;
        }
    }
    console.eof.set(true);
    shared.wake.notify();
}

/// Waits until the kernel has something to do: its deadline passed, a socket changed (`net`
/// builds), or console input arrived. Timers are spawned and never cancelled: an event that
/// ends the wait early leaves its timer running, and it later causes one spurious wake.
async fn wait(shared: &std::rc::Rc<Loop>, until: Option<u64>, inputs_open: bool) {
    let now = device_now();
    let seen = shared.wake.generation();
    match until {
        // A pure yield (the quantum's charge is already paid under a real clock): let the
        // pumps and the host run.
        Some(t) if t <= now => wit_bindgen::yield_async().await,
        Some(t) => {
            if shared.timers.arm(now, t) {
                let s = shared.clone();
                wit_bindgen::spawn_local(async move {
                    monotonic_clock::wait_until(host_time(t)).await;
                    s.timers.fired(t);
                    s.wake.notify();
                });
            }
            shared.wake.changed(seen).await;
        }
        // Blocked on sockets, the console or devices only: wait for them.
        None if shared.busy() || inputs_open => shared.wake.changed(seen).await,
        None => fail(DEADLOCK),
    }
}

/// The plain node's `wasi:cli/run`.
#[cfg(all(feature = "export", target_family = "wasm"))]
struct Node;

#[cfg(all(feature = "export", target_family = "wasm"))]
impl wasip3::exports::cli::run::Guest for Node {
    async fn run() -> Result<(), ()> {
        run(Rc::new(Plain)).await
    }
}

/// The node: parses the command line, mounts the image, boots the program (or waits for a
/// restore, [`NodeExt::paused`]) and runs the machine until it exits. The result is what
/// `wasi:cli/run` returns.
#[allow(clippy::result_unit_err)]
pub async fn run(ext: Rc<dyn NodeExt>) -> Result<(), ()> {
    let args: Vec<String> = wasip3::cli::environment::get_arguments()
        .into_iter()
        .skip(1)
        .collect();
    let opts = parse_args(&args).unwrap_or_else(|e| fail(&e));
    // The run-time tier before the engine is chosen: `auto` takes the two-tier engine when it
    // is on.
    #[cfg(feature = "jit")]
    crate::jit::enable(opts.jit_threshold);

    let image = opts.image.clone();
    let size = blob_size(&image).unwrap_or_else(|e| fail(&format!("image {image:?}: {e}")));
    let name = image.clone();
    // Host reads of the image (count, bytes), for `--stats`.
    let traffic = std::rc::Rc::new(std::cell::Cell::new((0u64, 0u64)));
    let counted = traffic.clone();
    let cached = CachedBlob::new(size, move |offset, len| {
        let (n, bytes) = counted.get();
        counted.set((n + 1, bytes + u64::from(len)));
        blob_read(&name, offset, len)
    });
    let mut vfs = Vfs::new();
    if let Err(e) = vfs.mount_image(Box::new(cached)) {
        fail(&format!("image {image:?}: {e}"));
    }

    let mut host = WasiHost { ext: ext.clone() };
    let (frontend, tier) = crate::frontend(opts.engine);
    // Shared so control calls can read it while `run` waits; never borrowed across an await.
    let mut cfg = opts.config.to_config();
    ext.config(&mut cfg);
    let kernel = std::rc::Rc::new(std::cell::RefCell::new(Kernel::new(
        cfg,
        Arch::Aarch64,
        frontend,
        vfs,
    )));
    ext.attach(&kernel, &opts);
    let spawn = Spawn {
        path: opts.argv[0].clone().into_bytes(),
        argv: opts.argv.iter().map(|a| a.clone().into_bytes()).collect(),
        envp: opts.env.iter().map(|e| e.clone().into_bytes()).collect(),
    };
    let shared = std::rc::Rc::new(Loop::default());
    #[cfg(feature = "net")]
    {
        shared.set_socket_ext(ext.socket_ext());
        kernel
            .borrow_mut()
            .set_network(Box::new(crate::net::WasiNet::new(shared.clone())));
    }
    // With `--console`, stdio is a terminal; its input is a raw byte stream read from
    // wasi:cli/stdin by a pump task.
    let console = opts.console.map(|(cols, rows)| {
        kernel.borrow_mut().attach_console(rows, cols);
        kernel.borrow_mut().set_console_hints(opts.console_hints);
        let c = std::rc::Rc::new(ConsoleIn::default());
        wit_bindgen::spawn_local(console_pump(c.clone(), shared.clone()));
        c
    });
    // Devices, before the first program starts.
    let hid_open = ext
        .start_devices(&kernel, &shared)
        .unwrap_or_else(|e| fail(&e));
    // A node built paused does not boot; a restore replaces its machine, and the loop below
    // runs the restored machine.
    if ext.paused() {
        let (image, engine, cfg) = (opts.image.clone(), opts.engine, opts.config.clone());
        let ext2 = ext.clone();
        ext.park(Restorer {
            kernel: kernel.clone(),
            shared: shared.clone(),
            make: Box::new(move || {
                let size = blob_size(&image).map_err(|e| format!("image {image:?}: {e}"))?;
                let name = image.clone();
                let cached =
                    CachedBlob::new(size, move |offset, len| blob_read(&name, offset, len));
                let mut vfs = Vfs::new();
                vfs.mount_image(Box::new(cached))
                    .map_err(|e| format!("image {image:?}: {e}"))?;
                let mut cfg = cfg.to_config();
                ext2.config(&mut cfg);
                Ok((vfs, crate::frontend(engine).0, cfg))
            }),
        });
        while !ext.restored() {
            let seen = shared.wake.generation();
            shared.wake.changed(seen).await;
        }
    } else {
        let spawned = kernel.borrow_mut().spawn(&spawn, &mut host);
        if let Err(e) = spawned {
            fail(&format!("{}: {e}", opts.argv[0]));
        }
    }
    let started = monotonic_clock::now();
    let status = loop {
        if let Some(c) = &console {
            let typed = std::mem::take(&mut *c.data.borrow_mut());
            if !typed.is_empty() {
                kernel.borrow_mut().console_input(&typed, &mut host);
            }
            if c.eof.get() && !c.eof_delivered.get() {
                c.eof_delivered.set(true);
                kernel.borrow_mut().console_eof();
            }
        }
        ext.before_run(&mut kernel.borrow_mut());
        let console_open = console.as_ref().is_some_and(|c| !c.eof_delivered.get());
        let ran = kernel.borrow_mut().run(&mut host);
        ext.after_run(&mut kernel.borrow_mut());
        match ran {
            RunStatus::Exited(status) => break status,
            // Without a network or a console nothing but time can wake the guest.
            #[cfg(not(feature = "net"))]
            // `t` is on the node's clock: after a restore it differs from the host's.
            RunStatus::Wait { until: Some(t) } if console.is_none() && !hid_open => {
                monotonic_clock::wait_until(host_time(t)).await
            }
            RunStatus::Wait { until } => wait(&shared, until, console_open || hid_open).await,
        }
    };
    if opts.stats {
        // The monotonic clock is real under stock wasmtime (MIPS is meaningful there) and
        // virtual under a deterministic host (where it equals retired x ns-per-insn plus idle
        // time).
        let elapsed = monotonic_clock::now().saturating_sub(started);
        let retired = kernel.borrow().retired();
        host.diag(&format!(
            "stats: retired={retired} elapsed={elapsed}ns mips={:.1} blob-reads={} blob-bytes={}",
            retired as f64 * 1e3 / elapsed.max(1) as f64,
            traffic.get().0,
            traffic.get().1
        ));
        if let Some(tier) = tier {
            // Engine counters (wall-clock only; never guest-visible).
            let s = tier.borrow().stats;
            host.diag(&format!(
                "engine: translated={} interpreted={} calls={} steps={} blocks={}",
                s.insns_jit, s.insns_interp, s.jit_calls, s.steps, s.interp_blocks
            ));
        }
        if let Some(j) = wasmshim_aarch64_engine::jit::stats() {
            host.diag(&format!(
                "jit: hot={} compiled={} cached={} declined={} refused={} funcs={} insns={} wasm={}B",
                j.hot, j.compiled, j.cache_hits, j.declined, j.refused, j.funcs, j.insns,
                j.wasm_bytes
            ));
        }
    }
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    let code = status.shell_code();
    if code != 0 {
        if ext.composed() {
            // One app of a composed component returns its failure instead of calling `exit`,
            // which would trap the whole component.
            host.diag(&format!("exit status {code}"));
            return Err(());
        }
        wasip3::cli::exit::exit_with_code(code);
    }
    Ok(())
}
