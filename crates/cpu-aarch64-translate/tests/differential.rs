//! Translated code against the interpreter, instruction class by instruction class.
//!
//! Each test builds straight-line A64 code (random encodings within a class, random register
//! values), runs it in the `wasmshim-aarch64` interpreter and as a translated region inside
//! wasmtime (register file, TLB and one data page in a Wasm memory laid out like the engine's),
//! and requires identical registers, flags, PC, memory and retired-instruction budget.
//! `jit_step` is a stub that records the PC it was called for and executes nothing, so a test
//! also proves which paths stay inline.

#![cfg(not(target_arch = "wasm32"))]

use wasmshim_aarch64::{Cpu, decode, layout};
use wasmshim_aarch64_translate::{Block, Options, Region, abi, translate_with};
use wasmshim_cpu::{
    A64Context, CpuContext, Event, GuestCpu, GuestMemory, PAGE_SIZE, PagedMemory, Perms,
};
use wasmtime::{Engine, Instance, Linker, Memory, MemoryType, Module, Store};

const CODE: u64 = 0x40_0000;
const DATA: u64 = 0x7f00_1000;
/// Layout inside the Wasm memory.
const W_CPU: usize = 0x1000;
const W_CTX: usize = 0x3000;
const W_TLB: usize = 0x1_0000;
const W_DATA: usize = 0x2_0000;

/// splitmix64.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Register values with edge cases mixed in.
    fn value(&mut self) -> u64 {
        match self.below(8) {
            0 => 0,
            1 => u64::MAX,
            2 => 1 << 63,
            3 => 0x8000_0000,
            4 => 0xffff_ffff,
            5 => self.below(64),
            _ => self.next(),
        }
    }
}

struct Outcome {
    ctx: A64Context,
    data: Vec<u8>,
    left: i64,
    exit: i32,
    steps: Vec<u64>,
}

fn initial(rng: &mut Rng) -> A64Context {
    let mut ctx = A64Context::default();
    for x in ctx.x.iter_mut() {
        *x = rng.value();
    }
    // Some registers point into the data page, so memory classes hit it.
    for r in [1, 2, 3] {
        ctx.x[r] = (DATA + rng.below(PAGE_SIZE / 2)) & !7;
    }
    ctx.sp = DATA + 0x800;
    ctx.pc = CODE;
    ctx.flags = rng.below(16) << 28;
    ctx.tp = rng.next();
    for v in ctx.v.iter_mut() {
        *v = match rng.below(4) {
            0 => 0,
            1 => u128::MAX,
            // Bytes from a small alphabet, so equal lanes and zero bytes are common.
            2 => (0..16).fold(0u128, |acc, i| {
                acc | (u128::from([0u8, 1, 0x80, 0xff][rng.below(4) as usize]) << (8 * i))
            }),
            _ => u128::from(rng.next()) << 64 | u128::from(rng.next()),
        };
    }
    ctx
}

fn interp(words: &[u32], ctx: &A64Context, data: &[u8], budget: u64) -> Outcome {
    let mut mem = PagedMemory::new();
    mem.map(CODE, PAGE_SIZE, Perms::RX).unwrap();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    mem.poke(CODE, &bytes).unwrap();
    mem.map(DATA, PAGE_SIZE, Perms::RW).unwrap();
    mem.poke(DATA, data).unwrap();
    let mut cpu = Cpu::new();
    cpu.set_context(&CpuContext::A64(ctx.clone()));
    let event = cpu.run_raw(&mut mem, budget);
    let retired = cpu.retired();
    let mut out = vec![0; PAGE_SIZE as usize];
    mem.read(DATA, &mut out).unwrap();
    Outcome {
        ctx: a64(cpu.context()),
        data: out,
        left: budget as i64 - retired as i64,
        exit: match event {
            Event::Yield => 0,
            Event::Syscall(_) => abi::EXIT_SYSCALL,
            Event::Fault(_) => -1,
        },
        steps: Vec::new(),
    }
}

struct Jit {
    store: Store<Vec<u64>>,
    memory: Memory,
    instance: Instance,
}

fn jit(words: &[u32], ctx: &A64Context, data: &[u8], budget: u64, tlb: bool) -> Outcome {
    let insts = words.iter().map(|&w| decode(w)).collect();
    let region = Region {
        blocks: vec![Block { pc: CODE, insts }],
        nest: None,
    };
    let opts = JIT_OPTS.with(|o| o.get()).unwrap_or_default();
    jit_region(&region, opts, ctx, data, budget, tlb)
}

/// Runs `region` (translated with `opts`) from block 0.
fn jit_region(
    region: &Region,
    opts: Options,
    ctx: &A64Context,
    data: &[u8],
    budget: u64,
    tlb: bool,
) -> Outcome {
    let (engine, module) = compile(region, opts);
    jit_run(&engine, &module, ctx, data, budget, tlb)
}

/// Translates and compiles `region` once (to run it many times).
fn compile(region: &Region, opts: Options) -> (Engine, Module) {
    let wasm = translate_with(region, opts);
    wasmparser::Validator::new()
        .validate_all(&wasm)
        .expect("translated module validates");
    let engine = Engine::default();
    let module = Module::new(&engine, &wasm).unwrap();
    (engine, module)
}

/// Runs a compiled region from block 0 (`ctx.pc` is its first block).
fn jit_run(
    engine: &Engine,
    module: &Module,
    ctx: &A64Context,
    data: &[u8],
    budget: u64,
    tlb: bool,
) -> Outcome {
    let engine = engine.clone();
    let mut store = Store::new(&engine, Vec::new());
    let memory = Memory::new(&mut store, MemoryType::new(4, None)).unwrap();
    let mut linker = Linker::new(&engine);
    linker
        .define(&store, abi::IMPORT_MODULE, abi::IMPORT_MEMORY, memory)
        .unwrap();
    linker
        .func_wrap(
            abi::IMPORT_MODULE,
            abi::IMPORT_FILL,
            move |mut caller: wasmtime::Caller<'_, Vec<u64>>,
                  addr: i64,
                  write: i32,
                  size: i32|
                  -> i32 {
                // With FILLS on, the data page fills like the engine's `jit_fill` (TLB entries,
                // the host address); otherwise every fill fails.
                let addr = addr as u64;
                if !FILLS.with(|f| f.get())
                    || addr & !0xfff != DATA
                    || (addr & 0xfff) + size as u64 > 0x1000
                {
                    return 0;
                }
                let m = memory.data_mut(&mut caller);
                let idx = ((DATA >> 12) & (abi::TLB_ENTRIES - 1)) as usize;
                let addend = (W_DATA as u32).wrapping_sub(DATA as u32);
                let tables: &[u64] = if write != 0 {
                    &[abi::TLB_READ, abi::TLB_WRITE]
                } else {
                    &[abi::TLB_READ]
                };
                for &t in tables {
                    let e = W_TLB + t as usize + idx * 16;
                    m[e..e + 8].copy_from_slice(&DATA.to_le_bytes());
                    m[e + 8..e + 12].copy_from_slice(&addend.to_le_bytes());
                }
                (W_DATA as u64 + (addr - DATA)) as i32
            },
        )
        .unwrap();
    let table = wasmtime::Table::new(
        &mut store,
        wasmtime::TableType::new(wasmtime::RefType::FUNCREF, 0, None),
        wasmtime::Ref::Func(None),
    )
    .unwrap();
    linker
        .define(&store, abi::IMPORT_MODULE, abi::IMPORT_TABLE, table)
        .unwrap();
    linker
        .func_wrap(
            abi::IMPORT_MODULE,
            abi::IMPORT_STEP,
            move |mut caller: wasmtime::Caller<'_, Vec<u64>>, pc: i64| -> i32 {
                // Record the PC the slow path was taken for, and report it (test failure
                // unless the test expects it). Like the engine's `jit_step`, it sets the CPU's
                // PC: a status means the instruction did not retire and runs from there.
                caller.data_mut().push(pc as u64);
                if let Some(words) = EXEC_STEPS.with(|e| e.borrow().clone()) {
                    // Executing mode: the interpreter runs the one instruction on the Wasm
                    // memory's registers and data page, as the engine's `jit_step` does.
                    return exec_step(memory.data_mut(&mut caller), &words, pc as u64);
                }
                let at = W_CPU + layout::PC;
                memory.data_mut(&mut caller)[at..at + 8].copy_from_slice(&pc.to_le_bytes());
                99
            },
        )
        .unwrap();
    let instance = linker.instantiate(&mut store, module).unwrap();
    let mut j = Jit {
        store,
        memory,
        instance,
    };
    j.write_cpu(ctx);
    let mem = j.memory.data_mut(&mut j.store);
    mem[W_DATA..W_DATA + data.len()].copy_from_slice(data);
    // TLB: every entry empty, then the data page in both tables (and the code page for reads).
    for i in 0..(2 * abi::TLB_ENTRIES) as usize {
        let e = W_TLB + i * abi::TLB_ENTRY_SIZE as usize;
        mem[e..e + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    }
    if tlb {
        for table in [abi::TLB_READ, abi::TLB_WRITE] {
            let idx = ((DATA >> 12) & (abi::TLB_ENTRIES - 1)) as usize;
            let e = W_TLB + table as usize + idx * 16;
            mem[e..e + 8].copy_from_slice(&DATA.to_le_bytes());
            let addend = (W_DATA as u32).wrapping_sub(DATA as u32);
            mem[e + 8..e + 12].copy_from_slice(&addend.to_le_bytes());
        }
    }
    mem[W_CTX..W_CTX + 8].copy_from_slice(&(budget as i64).to_le_bytes());
    mem[W_CTX + 8..W_CTX + 12].copy_from_slice(&0u32.to_le_bytes());
    mem[W_CTX + 16..W_CTX + 24].copy_from_slice(&CODE.to_le_bytes());
    // An empty chain table (no entry matches epoch MAX).
    mem[W_CTX + 24..W_CTX + 28].copy_from_slice(&(0x3_0000u32).to_le_bytes());
    mem[W_CTX + 32..W_CTX + 40].copy_from_slice(&u64::MAX.to_le_bytes());
    let f = j
        .instance
        .get_typed_func::<(i32, i32, i32), i32>(&mut j.store, abi::EXPORT_ENTRY)
        .unwrap();
    let exit = f
        .call(&mut j.store, (W_CPU as i32, W_TLB as i32, W_CTX as i32))
        .unwrap();
    let mem = j.memory.data(&j.store);
    let left = i64::from_le_bytes(mem[W_CTX..W_CTX + 8].try_into().unwrap());
    let data = mem[W_DATA..W_DATA + PAGE_SIZE as usize].to_vec();
    Outcome {
        ctx: j.read_cpu(),
        data,
        left,
        exit,
        steps: j.store.data().clone(),
    }
}

impl Jit {
    fn write_cpu(&mut self, ctx: &A64Context) {
        cpu_write(self.memory.data_mut(&mut self.store), ctx);
    }

    fn read_cpu(&self) -> A64Context {
        cpu_read(self.memory.data(&self.store))
    }
}

thread_local! {
    /// When set, the `jit_fill` stub fills the data page's TLB entries (as the engine does).
    static FILLS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Code-shape options for [`jit`] (default: what ships).
    static JIT_OPTS: std::cell::Cell<Option<Options>> = const { std::cell::Cell::new(None) };
    /// When set (the code at `CODE`), the `jit_step` stub executes the instruction.
    static EXEC_STEPS: std::cell::RefCell<Option<Vec<u32>>> = const { std::cell::RefCell::new(None) };
}

/// `jit_step` in executing mode: one interpreted instruction at `pc`, 0 when it retired.
fn exec_step(m: &mut [u8], words: &[u32], pc: u64) -> i32 {
    let mut ctx = cpu_read(m);
    ctx.pc = pc;
    let data = m[W_DATA..W_DATA + PAGE_SIZE as usize].to_vec();
    let mut mem = PagedMemory::new();
    mem.map(CODE, PAGE_SIZE, Perms::RX).unwrap();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    mem.poke(CODE, &bytes).unwrap();
    mem.map(DATA, PAGE_SIZE, Perms::RW).unwrap();
    mem.poke(DATA, &data).unwrap();
    let mut cpu = Cpu::new();
    cpu.set_context(&CpuContext::A64(ctx));
    let (event, _) = cpu.step_one(&mut mem);
    if !matches!(event, Event::Yield) {
        let at = W_CPU + layout::PC;
        m[at..at + 8].copy_from_slice(&pc.to_le_bytes());
        return 99;
    }
    let monitor = W_CPU + layout::MONITOR;
    let keep: [u8; 8] = m[monitor..monitor + 8].try_into().unwrap();
    let after = a64(cpu.context());
    cpu_write(m, &after);
    m[monitor..monitor + 8].copy_from_slice(&keep);
    let kind = W_CPU + layout::FLAGS_KIND;
    m[kind..kind + 4].copy_from_slice(&0u32.to_le_bytes());
    mem.read(DATA, &mut m[W_DATA..W_DATA + PAGE_SIZE as usize])
        .unwrap();
    0
}

/// The register file at `W_CPU` (the engine's `Cpu` layout); the monitor is cleared.
fn cpu_write(m: &mut [u8], ctx: &A64Context) {
    {
        let at = |off: usize| W_CPU + off;
        for i in 0..33 {
            let v = match i {
                31 => 0,
                32 => ctx.sp,
                _ => ctx.x[i],
            };
            m[at(layout::X + 8 * i)..at(layout::X + 8 * i) + 8].copy_from_slice(&v.to_le_bytes());
        }
        m[at(layout::PC)..at(layout::PC) + 8].copy_from_slice(&ctx.pc.to_le_bytes());
        for (i, b) in [31, 30, 29, 28].iter().enumerate() {
            m[at(layout::NZCV) + i] = ((ctx.flags >> b) & 1) as u8;
        }
        m[at(layout::TPIDR)..at(layout::TPIDR) + 8].copy_from_slice(&ctx.tp.to_le_bytes());
        m[at(layout::MONITOR)..at(layout::MONITOR) + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        m[at(layout::FPSR)..at(layout::FPSR) + 8].copy_from_slice(&ctx.fpsr.to_le_bytes());
        for i in 0..32 {
            let o = at(layout::V + 16 * i);
            m[o..o + 16].copy_from_slice(&ctx.v[i].to_le_bytes());
        }
    }
}

fn cpu_read(m: &[u8]) -> A64Context {
    {
        let at = |off: usize| W_CPU + off;
        let u64_at = |o: usize| u64::from_le_bytes(m[at(o)..at(o) + 8].try_into().unwrap());
        let mut ctx = A64Context::default();
        for i in 0..31 {
            ctx.x[i] = u64_at(layout::X + 8 * i);
        }
        ctx.sp = u64_at(layout::X + 8 * 32);
        ctx.pc = u64_at(layout::PC);
        let kind = u32::from_le_bytes(
            m[at(layout::FLAGS_KIND)..at(layout::FLAGS_KIND) + 4]
                .try_into()
                .unwrap(),
        );
        if kind == 0 {
            for (i, b) in [31, 30, 29, 28].iter().enumerate() {
                ctx.flags |= u64::from(m[at(layout::NZCV) + i]) << b;
            }
        } else {
            let (n, z, c, v) = wasmshim_aarch64::lazy_flags::nzcv(
                kind,
                u64_at(layout::FLAGS_X),
                u64_at(layout::FLAGS_Y),
                u64_at(layout::FLAGS_R),
            );
            ctx.flags =
                u64::from(n) << 31 | u64::from(z) << 30 | u64::from(c) << 29 | u64::from(v) << 28;
        }
        ctx.tp = u64_at(layout::TPIDR);
        ctx.fpsr = u64_at(layout::FPSR);
        for i in 0..32 {
            let o = at(layout::V + 16 * i);
            ctx.v[i] = u128::from_le_bytes(m[o..o + 16].try_into().unwrap());
        }
        assert_eq!(
            u64_at(layout::X + 8 * 31),
            0,
            "the zero register stays zero"
        );
        ctx
    }
}

fn check(name: &str, words: &[u32], rng: &mut Rng, tlb: bool) {
    let ctx = initial(rng);
    let data: Vec<u8> = (0..PAGE_SIZE).map(|_| rng.next() as u8).collect();
    let budget = words.len() as u64 + rng.below(3);
    let a = interp(words, &ctx, &data, budget);
    let b = jit(words, &ctx, &data, budget, tlb);
    let show = || {
        words
            .iter()
            .map(|w| format!("{w:08x} {:?}", decode(*w)))
            .collect::<Vec<_>>()
            .join("\n  ")
    };
    assert!(
        b.steps.is_empty(),
        "{name}: unexpected slow path in\n  {}",
        show()
    );
    assert_eq!(b.exit, abi::EXIT_JUMP, "{name}: exit\n  {}", show());
    // The interpreter ran the whole budget; the region stopped at its end: compare there.
    let _ = a.exit;
    assert_eq!(a.ctx.x, b.ctx.x, "{name}: registers\n  {}", show());
    assert_eq!(a.ctx.sp, b.ctx.sp, "{name}: sp\n  {}", show());
    assert_eq!(a.ctx.flags, b.ctx.flags, "{name}: flags\n  {}", show());
    assert_eq!(a.ctx.v, b.ctx.v, "{name}: vector registers\n  {}", show());
    assert_eq!(a.ctx.tp, b.ctx.tp, "{name}: tpidr\n  {}", show());
    assert_eq!(a.data, b.data, "{name}: memory\n  {}", show());
    let _ = (a.left, b.left);
}

/// Runs `n` random straight-line sequences whose words come from `gen`. The code is followed
/// by `B .+4`-free padding: the region's block simply ends and exits with `EXIT_JUMP`; the
/// interpreter gets a budget of exactly the code length.
fn class(name: &str, seed: u64, make: impl Fn(&mut Rng) -> u32) {
    let mut rng = Rng(seed);
    let mut ran = 0;
    for _ in 0..300 {
        let len = 1 + rng.below(6) as usize;
        let mut words = Vec::new();
        while words.len() < len {
            let w = make(&mut rng);
            if matches!(decode(w), wasmshim_aarch64::Inst::Undefined) {
                continue;
            }
            words.push(w);
        }
        let ctx = initial(&mut rng);
        let data: Vec<u8> = (0..PAGE_SIZE).map(|_| rng.next() as u8).collect();
        let budget = words.len() as u64;
        let a = interp(&words, &ctx, &data, budget);
        let b = jit(&words, &ctx, &data, budget, !FILLS.with(|f| f.get()));
        if b.exit == abi::EXIT_MEM {
            // A side exit: the engine runs that instruction and the rest of the
            // block in the interpreter; so does the test, and it must end where the
            // interpreter alone ends.
            assert!(b.steps.is_empty(), "{name}: a side exit after a step");
            let c = interp(&words, &b.ctx, &b.data, b.left as u64);
            assert_eq!(a.ctx, c.ctx, "{name}: state after a side exit");
            assert_eq!(a.data, c.data, "{name}: memory after a side exit");
            assert_eq!((a.left, a.exit), (c.left, c.exit), "{name}: budget, exit");
            continue;
        }
        if a.exit != 0 {
            // The interpreter faulted (an undefined combination, SP alignment, ...): the
            // translation must have handed that instruction to it, not computed anything.
            assert_eq!(
                b.exit, 99,
                "{name}: translated what the interpreter rejects"
            );
            continue;
        }
        let show = || {
            words
                .iter()
                .map(|w| format!("{w:08x} {:?}", decode(*w)))
                .collect::<Vec<_>>()
                .join("\n  ")
        };
        assert!(
            b.steps.is_empty(),
            "{name}: unexpected slow path at {:x?}
  {}",
            b.steps,
            show()
        );
        assert_eq!(b.exit, abi::EXIT_JUMP, "{name}: exit\n  {}", show());
        assert_eq!(a.ctx, b.ctx, "{name}: state\n  {}", show());
        assert_eq!(a.data, b.data, "{name}: memory\n  {}", show());
        assert_eq!(a.left, b.left, "{name}: budget\n  {}", show());
        ran += 1;
    }
    assert!(ran >= 30, "{name}: only {ran} sequences ran");
}

fn rd(r: &mut Rng) -> u32 {
    r.below(32) as u32
}

#[test]
fn add_sub_immediate() {
    // sf op S 100010 sh imm12 Rn Rd
    class("add/sub imm", 1, |r| {
        (r.below(2) as u32) << 31
            | (r.below(2) as u32) << 30
            | (r.below(2) as u32) << 29
            | 0b100010 << 23
            | (r.below(2) as u32) << 22
            | (r.below(4096) as u32) << 10
            | rd(r) << 5
            | rd(r)
    });
}

#[test]
fn add_sub_shifted_and_extended() {
    class("add/sub shifted", 2, |r| {
        let sf = r.below(2) as u32;
        let amount = r.below(if sf == 1 { 64 } else { 32 }) as u32;
        sf << 31
            | (r.below(2) as u32) << 30
            | (r.below(2) as u32) << 29
            | 0b01011 << 24
            | (r.below(3) as u32) << 22
            | rd(r) << 16
            | amount << 10
            | rd(r) << 5
            | rd(r)
    });
    class("add/sub extended", 3, |r| {
        (r.below(2) as u32) << 31
            | (r.below(2) as u32) << 30
            | (r.below(2) as u32) << 29
            | 0b01011001 << 21
            | rd(r) << 16
            | (r.below(8) as u32) << 13
            | (r.below(5) as u32) << 10
            | rd(r) << 5
            | rd(r)
    });
}

#[test]
fn logical() {
    class("logical shifted", 4, |r| {
        let sf = r.below(2) as u32;
        let amount = r.below(if sf == 1 { 64 } else { 32 }) as u32;
        sf << 31
            | (r.below(4) as u32) << 29
            | 0b01010 << 24
            | (r.below(4) as u32) << 22
            | (r.below(2) as u32) << 21
            | rd(r) << 16
            | amount << 10
            | rd(r) << 5
            | rd(r)
    });
    class("logical immediate", 5, |r| {
        let sf = r.below(2) as u32;
        let n = if sf == 1 { r.below(2) as u32 } else { 0 };
        sf << 31
            | (r.below(4) as u32) << 29
            | 0b100100 << 23
            | n << 22
            | (r.below(64) as u32) << 16
            | (r.below(64) as u32) << 10
            | rd(r) << 5
            | rd(r)
    });
}

#[test]
fn moves_bitfields_extract() {
    class("move wide", 6, |r| {
        let sf = r.below(2) as u32;
        let opc = [0, 2, 3][r.below(3) as usize];
        let hw = r.below(if sf == 1 { 4 } else { 2 }) as u32;
        sf << 31 | opc << 29 | 0b100101 << 23 | hw << 21 | (r.below(65536) as u32) << 5 | rd(r)
    });
    class("bitfield", 7, |r| {
        let sf = r.below(2) as u32;
        let lim = if sf == 1 { 64 } else { 32 };
        sf << 31
            | (r.below(3) as u32) << 29
            | 0b100110 << 23
            | sf << 22
            | (r.below(lim) as u32) << 16
            | (r.below(lim) as u32) << 10
            | rd(r) << 5
            | rd(r)
    });
    class("extract", 8, |r| {
        let sf = r.below(2) as u32;
        let lim = if sf == 1 { 64 } else { 32 };
        sf << 31
            | 0b00100111 << 23
            | sf << 22
            | rd(r) << 16
            | (r.below(lim) as u32) << 10
            | rd(r) << 5
            | rd(r)
    });
}

#[test]
fn conditional_select_and_compare() {
    class("csel", 9, |r| {
        (r.below(2) as u32) << 31
            | (r.below(2) as u32) << 30
            | 0b011010100 << 21
            | rd(r) << 16
            | (r.below(16) as u32) << 12
            | (r.below(2) as u32) << 10
            | rd(r) << 5
            | rd(r)
    });
    class("ccmp/ccmn", 10, |r| {
        (r.below(2) as u32) << 31
            | (r.below(2) as u32) << 30
            | 1 << 29
            | 0b11010010 << 21
            | rd(r) << 16
            | (r.below(16) as u32) << 12
            | (r.below(2) as u32) << 11
            | rd(r) << 5
            | (r.below(16) as u32)
    });
    class("adc/sbc", 11, |r| {
        (r.below(2) as u32) << 31
            | (r.below(2) as u32) << 30
            | 0b11010000 << 21
            | rd(r) << 16
            | rd(r) << 5
            | rd(r)
    });
}

#[test]
fn data_processing() {
    class("dp2", 12, |r| {
        let opcode = [2, 3, 8, 9, 10, 11][r.below(6) as usize];
        (r.below(2) as u32) << 31
            | 0b11010110 << 21
            | rd(r) << 16
            | opcode << 10
            | rd(r) << 5
            | rd(r)
    });
    class("dp3", 13, |r| {
        // MADD/MSUB (64 and 32), SMADDL/SMSUBL/UMADDL/UMSUBL.
        let (sf, op31) = [(0, 0), (1, 0), (1, 1), (1, 5)][r.below(4) as usize];
        sf << 31
            | 0b11011 << 24
            | op31 << 21
            | rd(r) << 16
            | (r.below(2) as u32) << 15
            | rd(r) << 10
            | rd(r) << 5
            | rd(r)
    });
    class("clz/cls", 14, |r| {
        (r.below(2) as u32) << 31
            | 0b1011010110 << 21
            | (4 + r.below(2) as u32) << 10
            | rd(r) << 5
            | rd(r)
    });
    class("adr/adrp", 15, |r| {
        (r.below(2) as u32) << 31
            | (r.below(4) as u32) << 29
            | 0b10000 << 24
            | (r.below(1 << 19) as u32) << 5
            | rd(r)
    });
}

#[test]
fn loads_and_stores_hit_the_tlb() {
    // LDR/STR (unsigned offset, 8 bytes and 4 bytes) and LDP/STP (signed offset) off x1..x3.
    class("ldr/str uimm", 16, |r| {
        let size = 2 + r.below(2) as u32;
        let opc = r.below(3) as u32; // store, load, (load signed for size 2 only)
        let opc = if size == 3 && opc == 2 { 1 } else { opc };
        size << 30
            | 0b111001 << 24
            | opc << 22
            | (r.below(64) as u32) << 10
            | (1 + r.below(3) as u32) << 5
            | rd(r)
    });
    class("ldp/stp", 17, |r| {
        let opc = r.below(2) as u32 * 2; // 32-bit or 64-bit
        opc << 30
            | 0b101_0010 << 23
            | (r.below(2) as u32) << 22
            | (r.below(32) as u32) << 15
            | rd(r) << 10
            | (1 + r.below(3) as u32) << 5
            | rd(r)
    });
    class("ldrb/strb/ldrh post-index", 18, |r| {
        let size = r.below(2) as u32;
        let opc = r.below(4) as u32;
        size << 30
            | 0b111000 << 24
            | opc << 22
            | (r.below(32) as u32) << 12
            | 0b01 << 10
            | (1 + r.below(3) as u32) << 5
            | (4 + r.below(20) as u32)
    });
}

/// The same accesses with an empty TLB and a `jit_fill` that succeeds: the compact miss path
/// fills, joins the one copy of the access and goes on (no step, no exit).
#[test]
fn loads_and_stores_after_a_fill() {
    FILLS.with(|f| f.set(true));
    loads_and_stores_hit_the_tlb();
    // And with compact miss paths (one copy of the access, joined after the fill).
    JIT_OPTS.with(|o| {
        o.set(Some(Options {
            compact_miss: true,
            ..Options::default()
        }))
    });
    loads_and_stores_hit_the_tlb();
    FILLS.with(|f| f.set(false));
    // Fills that fail: side exits.
    loads_and_stores_hit_the_tlb();
    JIT_OPTS.with(|o| o.set(None));
}

#[test]
fn budget_shorter_than_the_block_exits_before_it() {
    let words = [0x9100_0400u32; 4]; // add x0, x0, #1
    let mut rng = Rng(42);
    let ctx = initial(&mut rng);
    let data = vec![0; PAGE_SIZE as usize];
    let b = jit(&words, &ctx, &data, 3, true);
    assert_eq!(b.exit, abi::EXIT_BUDGET);
    assert_eq!(b.left, 3);
    assert_eq!(b.ctx, ctx, "nothing ran");
}

#[test]
fn tlb_miss_steps_or_leaves_with_the_registers_in_memory() {
    // add x0, x0, #1; str x0, [x1]; ...: with an empty TLB (and a fill that fails, as the
    // stub's does), the store goes to `jit_step` in straight-line code (the stub returns 99),
    // or with compact miss paths leaves the region (EXIT_MEM), as it does in a block that
    // loops on itself; always with the add retired and stored, and the store and
    // the rest of the block given back.
    let mut rng = Rng(7);
    let mut ctx = initial(&mut rng);
    ctx.x[1] = DATA;
    let data = vec![0; PAGE_SIZE as usize];
    let straight = [0x9100_0400u32, 0xf900_0020, 0x9100_0400];
    let b = jit(&straight, &ctx, &data, 10, false);
    assert_eq!(b.steps, vec![CODE + 4], "the store is stepped");
    assert_eq!(b.exit, 99, "the step status is the exit code");
    assert_eq!(
        b.left, 9,
        "the store and the rest of the block are given back"
    );
    assert_eq!(
        b.ctx.x[0],
        ctx.x[0].wrapping_add(1),
        "the add retired, x0 is in memory"
    );
    JIT_OPTS.with(|o| {
        o.set(Some(Options {
            compact_miss: true,
            ..Options::default()
        }))
    });
    let b = jit(&straight, &ctx, &data, 10, false);
    JIT_OPTS.with(|o| o.set(None));
    assert!(b.steps.is_empty(), "no call to jit_step");
    assert_eq!(b.exit, abi::EXIT_MEM);
    assert_eq!(b.ctx.pc, CODE + 4, "pc = the store");
    assert_eq!(
        b.left, 9,
        "the store and the rest of the block are given back"
    );
    assert_eq!(
        b.ctx.x[0],
        ctx.x[0].wrapping_add(1),
        "the add retired, x0 is in memory"
    );
    let looping = [0x9100_0400u32, 0xf900_0020, 0x17ff_fffe]; // ...; b .-8
    let region = Region {
        blocks: vec![Block {
            pc: CODE,
            insts: looping.iter().map(|&w| decode(w)).collect(),
        }],
        nest: None,
    };
    let b = jit_region(&region, Options::default(), &ctx, &data, 10, false);
    assert!(b.steps.is_empty(), "no call to jit_step");
    assert_eq!(b.exit, abi::EXIT_MEM);
    assert_eq!(
        b.left, 9,
        "the store and the rest of the block are given back"
    );
    assert_eq!(b.ctx.pc, CODE + 4, "pc = the store");
    assert_eq!(
        b.ctx.x[0],
        ctx.x[0].wrapping_add(1),
        "the add retired, x0 is in memory"
    );
    assert_eq!(b.data, data, "the store did not run");
    check("no-op", &[0xd503_201f], &mut rng, true);
}

#[test]
fn exclusives_and_acquire_release() {
    class("ldxr/stxr/ldar/stlr", 19, |r| {
        let rn = 1 + r.below(3) as u32;
        let rt = rd(r);
        let rs = rd(r);
        let x = (r.below(2) as u32) << 30; // 64-bit (size 11) or 32-bit (size 10)
        match r.below(6) {
            0 => 0x885F_7C00 | x | rn << 5 | rt,            // LDXR
            1 => 0x885F_FC00 | x | rn << 5 | rt,            // LDAXR
            2 => 0x8800_7C00 | x | rs << 16 | rn << 5 | rt, // STXR
            3 => 0x8800_FC00 | x | rs << 16 | rn << 5 | rt, // STLXR
            4 => 0x88DF_FC00 | x | rn << 5 | rt,            // LDAR
            _ => 0x889F_FC00 | x | rn << 5 | rt,            // STLR
        }
    });
}

fn vr(r: &mut Rng) -> u32 {
    r.below(32) as u32
}

#[test]
fn fp_general_moves_reversals_and_high_multiplies() {
    class("fmov general", 20, |r| {
        let base = [
            0x9E66_0000u32,
            0x9E67_0000,
            0x1E26_0000,
            0x1E27_0000,
            0x9EAE_0000,
            0x9EAF_0000,
        ];
        base[r.below(6) as usize] | vr(r) << 5 | vr(r)
    });
    class("rev/rbit", 21, |r| {
        let sf = r.below(2) as u32;
        let opcode = r.below(4) as u32;
        0x5AC0_0000 | sf << 31 | opcode << 10 | rd(r) << 5 | rd(r)
    });
    class("umulh/smulh", 22, |r| {
        let base = if r.below(2) == 0 {
            0x9BC0_7C00u32
        } else {
            0x9B40_7C00
        };
        base | rd(r) << 16 | rd(r) << 5 | rd(r)
    });
}

#[test]
fn simd_three_same_two_misc() {
    class("three same", 23, |r| {
        let opcode = [3u32, 17, 6, 7, 20, 21, 23][r.below(7) as usize];
        let u = r.below(2) as u32;
        // 64-bit lanes have no unsigned compare or max/min in Wasm SIMD: those step.
        let size = if matches!(opcode, 6 | 7 | 20 | 21) {
            r.below(3)
        } else {
            r.below(4)
        } as u32;
        0x0E20_0400
            | (r.below(2) as u32) << 30
            | u << 29
            | size << 22
            | vr(r) << 16
            | opcode << 11
            | vr(r) << 5
            | vr(r)
    });
    class("two misc", 24, |r| {
        let opcode = [5u32, 8, 9, 10][r.below(4) as usize];
        // CNT/NOT are byte operations (RBIT vector, size 1, steps).
        let size = if opcode == 5 { 0 } else { r.below(4) as u32 };
        0x0E20_0800
            | (r.below(2) as u32) << 30
            | (r.below(2) as u32) << 29
            | size << 22
            | opcode << 12
            | vr(r) << 5
            | vr(r)
    });
}

#[test]
fn simd_copy_permute_ext_tbl() {
    class("copy", 25, |r| {
        let imm5 = 1 + r.below(31) as u32;
        let imm4 = [0u32, 1, 3, 5, 7, r.below(16) as u32][r.below(6) as usize];
        0x0E00_0400
            | (r.below(2) as u32) << 30
            | (r.below(2) as u32) << 29
            | imm5 << 16
            | imm4 << 11
            | rd(r) << 5
            | rd(r)
    });
    class("permute", 26, |r| {
        let q = r.below(2) as u32;
        0x0E00_0800
            | q << 30
            | (r.below(if q == 1 { 4 } else { 3 }) as u32) << 22
            | vr(r) << 16
            | (r.below(8) as u32) << 12
            | vr(r) << 5
            | vr(r)
    });
    class("ext", 27, |r| {
        let q = r.below(2) as u32;
        let imm4 = r.below(if q == 1 { 16 } else { 8 }) as u32;
        0x2E00_0000 | q << 30 | vr(r) << 16 | imm4 << 11 | vr(r) << 5 | vr(r)
    });
    class("tbl", 28, |r| {
        0x0E00_0000 | (r.below(2) as u32) << 30 | vr(r) << 16 | vr(r) << 5 | vr(r)
    });
}

/// A counted loop whose block branches to itself (`add; subs; b.ne`), then an `SVC`, run for
/// every budget from 0 up: translated with and without `Options::self_loops`, finished by the
/// interpreter after a budget exit (as the engine does), it must stop where the interpreter
/// alone stops, with the same registers, flags and budget.
#[test]
fn self_loops_exit_exactly_like_the_interpreter() {
    let words = [
        0x8b02_0021u32, // add x1, x1, x2
        0xf100_0400,    // subs x0, x0, #1
        0x54ff_ffc1,    // b.ne .-8
        0xd400_0001,    // svc #0
    ];
    let region = Region {
        blocks: vec![
            Block {
                pc: CODE,
                insts: words[..3].iter().map(|&w| decode(w)).collect(),
            },
            Block {
                pc: CODE + 12,
                insts: vec![decode(words[3])],
            },
        ],
        nest: None,
    };
    let mut rng = Rng(7);
    let data = vec![0; PAGE_SIZE as usize];
    for n in [0u64, 1, 2, 3, 7, 20] {
        let mut ctx = initial(&mut rng);
        ctx.pc = CODE;
        ctx.x[0] = n;
        for budget in 0..80 {
            let a = interp(&words, &ctx, &data, budget);
            let mut outcomes = Vec::new();
            for self_loops in [false, true] {
                let b = jit_region(
                    &region,
                    Options {
                        self_loops,
                        ..Options::default()
                    },
                    &ctx,
                    &data,
                    budget,
                    true,
                );
                assert!(b.steps.is_empty(), "no slow path");
                let end = match b.exit {
                    abi::EXIT_BUDGET => interp(&words, &b.ctx, &b.data, b.left as u64),
                    abi::EXIT_SYSCALL => b,
                    e => panic!("n={n} budget={budget}: exit {e}"),
                };
                let what = format!("n={n} budget={budget} self_loops={self_loops}");
                assert_eq!(end.ctx, a.ctx, "{what}: state");
                assert_eq!(end.left, a.left, "{what}: budget");
                assert_eq!(end.exit, a.exit, "{what}: exit");
                outcomes.push((end.ctx, end.left));
            }
            assert_eq!(outcomes[0], outcomes[1]);
        }
    }
}

/// A second data page the translated code never has in its TLB.
const DATA2: u64 = 0x7f00_3000;

/// The interpreter with both data pages: (outcome with `data`, the second page).
fn interp2(
    words: &[u32],
    ctx: &A64Context,
    data: &[u8],
    data2: &[u8],
    budget: u64,
) -> (Outcome, Vec<u8>) {
    let mut mem = PagedMemory::new();
    mem.map(CODE, PAGE_SIZE, Perms::RX).unwrap();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    mem.poke(CODE, &bytes).unwrap();
    for (at, page) in [(DATA, data), (DATA2, data2)] {
        mem.map(at, PAGE_SIZE, Perms::RW).unwrap();
        mem.poke(at, page).unwrap();
    }
    let mut cpu = Cpu::new();
    cpu.set_context(&CpuContext::A64(ctx.clone()));
    let event = cpu.run_raw(&mut mem, budget);
    let retired = cpu.retired();
    let mut out = vec![0; PAGE_SIZE as usize];
    mem.read(DATA, &mut out).unwrap();
    let mut out2 = vec![0; PAGE_SIZE as usize];
    mem.read(DATA2, &mut out2).unwrap();
    let outcome = Outcome {
        ctx: a64(cpu.context()),
        data: out,
        left: budget as i64 - retired as i64,
        exit: match event {
            Event::Yield => 0,
            Event::Syscall(_) => abi::EXIT_SYSCALL,
            Event::Fault(_) => -1,
        },
        steps: Vec::new(),
    };
    (outcome, out2)
}

/// Side exits at every access position: self-looping blocks of loads and stores (single, pair,
/// post-index; 1, 2, 4 and 8 bytes) mixed with arithmetic, where exactly the `j`-th access goes
/// through a page the translated code's TLB does not map, for every `j`. The region must leave
/// with `EXIT_MEM` at that access with everything before it retired and nothing of it done; the
/// interpreter then finishes the block (as the engine does) and the result must equal the
/// interpreter's alone: registers, flags, both pages, budget.
#[test]
fn memory_side_exits_at_every_access_position() {
    let access = |r: &mut Rng, base: u32| -> u32 {
        match r.below(3) {
            0 => {
                let size = r.below(4) as u32;
                let opc = r.below(2) as u32; // store, load
                size << 30
                    | 0b111001 << 24
                    | opc << 22
                    | (r.below(64) as u32) << 10
                    | base << 5
                    | (5 + r.below(15) as u32)
            }
            1 => {
                let opc = r.below(2) as u32 * 2;
                opc << 30
                    | 0b101_0010 << 23
                    | (r.below(2) as u32) << 22
                    | (r.below(32) as u32) << 15
                    | (5 + r.below(15) as u32) << 10
                    | base << 5
                    | (5 + r.below(15) as u32)
            }
            _ => {
                let size = r.below(2) as u32;
                let opc = r.below(2) as u32;
                size << 30
                    | 0b111000 << 24
                    | opc << 22
                    | (r.below(32) as u32) << 12
                    | 0b01 << 10
                    | base << 5
                    | (5 + r.below(15) as u32)
            }
        }
    };
    let mut rng = Rng(102);
    let mut checked = 0;
    for _ in 0..200 {
        let m = 1 + rng.below(5) as usize;
        // The shape: accesses with arithmetic between them (add x20..x24, #imm).
        let mut shape = Vec::new();
        for _ in 0..m {
            for _ in 0..rng.below(3) {
                shape.push(None);
            }
            shape.push(Some(rng.next()));
        }
        let mut ctx = initial(&mut rng);
        ctx.x[4] = (DATA2 + rng.below(PAGE_SIZE / 2)) & !7;
        let data: Vec<u8> = (0..PAGE_SIZE).map(|_| rng.next() as u8).collect();
        let data2: Vec<u8> = (0..PAGE_SIZE).map(|_| rng.next() as u8).collect();
        for j in 0..m {
            let mut words = Vec::new();
            let mut nth = 0;
            let mut at = 0;
            for s in &shape {
                match s {
                    None => {
                        let rd = 20 + rng.below(5) as u32;
                        words.push(0x9100_0000 | (rng.below(4096) as u32) << 10 | rd << 5 | rd);
                    }
                    Some(seed) => {
                        let mut r = Rng(*seed);
                        let base = if nth == j { 4 } else { 1 + r.below(3) as u32 };
                        if nth == j {
                            at = words.len();
                        }
                        words.push(access(&mut r, base));
                        nth += 1;
                    }
                }
            }
            if words
                .iter()
                .any(|&w| matches!(decode(w), wasmshim_aarch64::Inst::Undefined))
            {
                continue;
            }
            // The block loops on itself (side exits are taken in loops): `b` back to its start.
            words.push(0x1400_0000 | ((-(words.len() as i32)) as u32 & 0x03ff_ffff));
            let budget = words.len() as u64;
            let (a, a2) = interp2(&words, &ctx, &data, &data2, budget);
            let region = Region {
                blocks: vec![Block {
                    pc: CODE,
                    insts: words.iter().map(|&w| decode(w)).collect(),
                }],
                nest: None,
            };
            let b = jit_region(&region, Options::default(), &ctx, &data, budget, true);
            let show = || {
                words
                    .iter()
                    .map(|w| format!("{w:08x} {:?}", decode(*w)))
                    .collect::<Vec<_>>()
                    .join("\n  ")
            };
            assert!(b.steps.is_empty(), "no jit_step:\n  {}", show());
            if a.exit != 0 && b.exit != abi::EXIT_MEM {
                // An earlier access faulted in the interpreter (a register it loaded): the
                // translation left before or at it, compared below only when it side-exited.
                continue;
            }
            assert_eq!(b.exit, abi::EXIT_MEM, "j={j}: exit\n  {}", show());
            assert!(
                b.ctx.pc <= CODE + 4 * at as u64,
                "j={j}: left after the access\n  {}",
                show()
            );
            let (c, c2) = interp2(&words, &b.ctx, &b.data, &data2, b.left as u64);
            assert_eq!(a.ctx, c.ctx, "j={j}: state\n  {}", show());
            assert_eq!(a.data, c.data, "j={j}: page 1\n  {}", show());
            assert_eq!(a2, c2, "j={j}: page 2\n  {}", show());
            assert_eq!(
                (a.left, a.exit),
                (c.left, c.exit),
                "j={j}: budget, exit\n  {}",
                show()
            );
            checked += 1;
        }
    }
    assert!(checked >= 300, "only {checked} cases ran");
}

/// Loop regions: random loop nests (an outer counted loop around an inner one
/// of one or two blocks, forward branches over parts of both, flags across blocks, loads and
/// stores that sometimes go through a page the translated code's TLB does not map) formed by
/// `aot::translate_page_with` as the profile would see them. The loop region runs from its
/// header with every budget from 0 to past the end; whatever it exits with (budget, side exit,
/// loop exit, syscall), the interpreter finishes the budget, as the engine does, and the result
/// must equal the interpreter's alone: registers, flags, both pages, budget, exit.
#[test]
fn loop_regions_exit_exactly_like_the_interpreter() {
    use wasmshim_aarch64_translate::aot::{PageProfile, translate_page_with};
    let mut rng = Rng(104);
    let (mut formed, mut runs, mut mem_exits) = (0, 0, 0);
    for _ in 0..60 {
        let mut words: Vec<u32> = Vec::new();
        let body = |r: &mut Rng, words: &mut Vec<u32>, min: u64, spread: u64| {
            let n = min + r.below(spread);
            for _ in 0..n {
                let rd = 21 + r.below(6) as u32;
                let rm = 21 + r.below(6) as u32;
                words.push(match r.below(6) {
                    0 => 0x9100_0000 | (r.below(4096) as u32) << 10 | rd << 5 | rd,
                    1 => 0xca00_0000 | rm << 16 | rd << 5 | rd,
                    2 => 0xeb00_001f | rm << 16 | rd << 5, // cmp
                    _ => {
                        // ldr/str (unsigned offset); base x4 is the unmapped page (1 in 6).
                        let base = if r.below(6) == 0 {
                            4
                        } else {
                            1 + r.below(3) as u32
                        };
                        let size = r.below(4) as u32;
                        (size << 30)
                            | 0b111001 << 24
                            | (r.below(2) as u32) << 22
                            | (r.below(64) as u32) << 10
                            | base << 5
                            | rd
                    }
                });
            }
        };
        let cond_branch = |r: &mut Rng, from: usize, to: usize| -> u32 {
            let off = ((to as i64 - from as i64) as u32) & 0x3fff;
            let bit = r.below(64) as u32;
            let op = if r.below(2) == 0 {
                0x3600_0000
            } else {
                0x3700_0000
            }; // tbz / tbnz
            op | (bit >> 5) << 31 | (bit & 31) << 19 | off << 5 | (21 + r.below(6) as u32)
        };
        let b_ne = |from: usize, to: usize| -> u32 {
            0x5400_0001 | (((to as i64 - from as i64) as u32) & 0x7_ffff) << 5
        };
        // H: body A, maybe a forward branch to K, `movz x20, #m`
        body(&mut rng, &mut words, 1, 3);
        let skip_outer = rng.below(3) == 0;
        let skip_at = words.len();
        if skip_outer {
            words.push(0); // patched below
        }
        words.push(0xd280_0014 | ((1 + rng.below(4)) as u32) << 5);
        // I: body B, maybe a second inner block, `subs x20, x20, #1; b.ne I`
        let inner = words.len();
        body(&mut rng, &mut words, 1, 3);
        if rng.below(2) == 0 {
            let at = words.len();
            words.push(0);
            body(&mut rng, &mut words, 1, 2);
            let j = words.len();
            words[at] = cond_branch(&mut rng, at, j);
            body(&mut rng, &mut words, 0, 2);
        }
        words.push(0xf100_0694); // subs x20, x20, #1
        let at = words.len();
        words.push(b_ne(at, inner));
        // K: body D, `subs x19, x19, #1; b.ne H`, then `svc #0`.
        let k = words.len();
        if skip_outer {
            words[skip_at] = cond_branch(&mut rng, skip_at, k);
        }
        body(&mut rng, &mut words, 0, 3);
        words.push(0xf100_0673); // subs x19, x19, #1
        let at = words.len();
        words.push(b_ne(at, 0));
        words.push(0xd400_0001);
        if words
            .iter()
            .any(|&w| matches!(decode(w), wasmshim_aarch64::Inst::Undefined))
        {
            continue;
        }
        // The profile's view: every block start is a hot entry.
        let mut entries = vec![0u16];
        for (i, &w) in words.iter().enumerate() {
            let inst = decode(w);
            if wasmshim_aarch64_translate::ends_block(&inst) {
                let b = Block {
                    pc: 4 * i as u64,
                    insts: vec![inst],
                };
                for s in wasmshim_aarch64_translate::successors(&b) {
                    if (s as usize) < 4 * words.len() {
                        entries.push(s as u16);
                    }
                }
            }
        }
        let mut bytes = vec![0u8; PAGE_SIZE as usize];
        for (i, w) in words.iter().enumerate() {
            bytes[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        let t = translate_page_with(&PageProfile { bytes, entries }, Options::default());
        let Some(region) = t
            .regions
            .iter()
            .find(|r| r.nest.is_some() && r.blocks[0].pc == 0)
        else {
            continue;
        };
        formed += 1;
        let (engine, module) = compile(region, Options::default());
        let show = || {
            words
                .iter()
                .enumerate()
                .map(|(i, w)| format!("{:3x}: {w:08x} {:?}", 4 * i, decode(*w)))
                .collect::<Vec<_>>()
                .join("\n  ")
        };
        for _ in 0..3 {
            let mut ctx = initial(&mut rng);
            ctx.pc = CODE;
            ctx.x[19] = 1 + rng.below(4);
            ctx.x[4] = (DATA2 + rng.below(PAGE_SIZE / 2)) & !7;
            let data: Vec<u8> = (0..PAGE_SIZE).map(|_| rng.next() as u8).collect();
            let data2: Vec<u8> = (0..PAGE_SIZE).map(|_| rng.next() as u8).collect();
            let (all, _) = interp2(&words, &ctx, &data, &data2, 100_000);
            let total = (100_000 - all.left) as u64;
            for budget in 0..=total + 2 {
                let (a, a2) = interp2(&words, &ctx, &data, &data2, budget);
                let b = jit_run(&engine, &module, &ctx, &data, budget, true);
                assert!(b.steps.is_empty(), "no jit_step:\n  {}", show());
                if b.exit == abi::EXIT_MEM {
                    mem_exits += 1;
                }
                let (c, c2) = interp2(&words, &b.ctx, &b.data, &data2, b.left as u64);
                let what = format!("budget {budget} (exit {}):\n  {}", b.exit, show());
                assert_eq!(a.ctx, c.ctx, "state, {what}");
                assert_eq!(a.data, c.data, "page 1, {what}");
                assert_eq!(a2, c2, "page 2, {what}");
                assert_eq!((a.left, a.exit), (c.left, c.exit), "budget, exit, {what}");
                runs += 1;
            }
        }
    }
    println!("{formed} loop regions, {runs} runs, {mem_exits} side exits");
    assert!(formed >= 55, "only {formed} loop regions formed");
    assert!(mem_exits >= 100, "only {mem_exits} side exits");
}

/// The A64 registers of a context.
fn a64(c: CpuContext) -> A64Context {
    let CpuContext::A64(c) = c;
    c
}

/// Page regions: random acyclic code of several blocks (forward conditional
/// and unconditional branches, flags set in one block and read in the next, loads and stores of
/// which one in six goes through a page the TLB does not map, so `jit_step` is called and
/// leaves), translated as a page region in every code shape: registers in locals for the whole
/// call (the old shape), block-scoped registers, and block-scoped registers carried along single
/// forward edges. Run from its first block with every budget, finished by the interpreter as
/// the engine does, it must equal the interpreter alone: registers, flags, both pages, budget.
#[test]
fn page_regions_exit_exactly_like_the_interpreter() {
    use wasmshim_aarch64_translate::aot::{PageProfile, translate_page_with};
    let mut rng = Rng(107);
    let shapes = [
        Options {
            block_regs: false,
            ..Options::default()
        },
        Options {
            block_regs: true,
            carry: false,
            ..Options::default()
        },
        Options {
            block_regs: true,
            carry: true,
            ..Options::default()
        },
    ];
    let (mut regions, mut runs, mut steps) = (0, 0, 0);
    for _ in 0..40 {
        let mut words: Vec<u32> = Vec::new();
        let chunks = 3 + rng.below(4) as usize;
        let mut starts = Vec::new();
        // (index of a branch to patch, target chunk, branch kind)
        let mut patches: Vec<(usize, usize, u32)> = Vec::new();
        for c in 0..chunks {
            starts.push(words.len());
            let n = 1 + rng.below(4);
            for _ in 0..n {
                let rd = 21 + rng.below(6) as u32;
                let rm = 21 + rng.below(6) as u32;
                words.push(match rng.below(7) {
                    0 => 0x9100_0000 | (rng.below(4096) as u32) << 10 | rd << 5 | rd,
                    1 => 0xca00_0000 | rm << 16 | rd << 5 | rd,
                    2 => 0xeb00_001f | rm << 16 | rd << 5, // cmp
                    // csel rd, rd, rm, cond (the flags perhaps from an earlier block)
                    3 => 0x9a80_0000 | rm << 16 | (rng.below(14) as u32) << 12 | rd << 5 | rd,
                    _ => {
                        let base = if rng.below(6) == 0 {
                            4
                        } else {
                            1 + rng.below(3) as u32
                        };
                        let size = rng.below(4) as u32;
                        (size << 30)
                            | 0b111001 << 24
                            | (rng.below(2) as u32) << 22
                            | (rng.below(64) as u32) << 10
                            | base << 5
                            | rd
                    }
                });
            }
            if c + 1 == chunks {
                words.push(0xd400_0001); // svc #0
                continue;
            }
            let to = c + 1 + rng.below((chunks - c - 1) as u64) as usize;
            let kind = rng.below(3) as u32;
            patches.push((words.len(), to, kind));
            words.push(0);
        }
        for (at, to, kind) in patches {
            let off = (starts[to] as i64 - at as i64) as u32;
            words[at] = match kind {
                0 => 0x1400_0000 | (off & 0x3ff_ffff), // b
                1 => 0x5400_0000 | (off & 0x7_ffff) << 5 | rng.below(14) as u32, // b.cond
                _ => {
                    let bit = rng.below(64) as u32;
                    let op = if rng.below(2) == 0 {
                        0x3600_0000
                    } else {
                        0x3700_0000
                    };
                    op | (bit >> 5) << 31
                        | (bit & 31) << 19
                        | (off & 0x3fff) << 5
                        | (21 + rng.below(6) as u32)
                }
            };
        }
        if words
            .iter()
            .any(|&w| matches!(decode(w), wasmshim_aarch64::Inst::Undefined))
        {
            continue;
        }
        let mut entries = vec![0u16];
        for (i, &w) in words.iter().enumerate() {
            let inst = decode(w);
            if wasmshim_aarch64_translate::ends_block(&inst) {
                let b = Block {
                    pc: 4 * i as u64,
                    insts: vec![inst],
                };
                for s in wasmshim_aarch64_translate::successors(&b) {
                    if (s as usize) < 4 * words.len() {
                        entries.push(s as u16);
                    }
                }
            }
        }
        let mut bytes = vec![0u8; PAGE_SIZE as usize];
        for (i, w) in words.iter().enumerate() {
            bytes[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        let show = || {
            words
                .iter()
                .enumerate()
                .map(|(i, w)| format!("{:3x}: {w:08x} {:?}", 4 * i, decode(*w)))
                .collect::<Vec<_>>()
                .join("\n  ")
        };
        for (s, &opts) in shapes.iter().enumerate() {
            let t = translate_page_with(
                &PageProfile {
                    bytes: bytes.clone(),
                    entries: entries.clone(),
                },
                opts,
            );
            let Some(region) = t
                .regions
                .iter()
                .find(|r| r.nest.is_none() && r.blocks[0].pc == 0)
            else {
                continue;
            };
            regions += 1;
            let (engine, module) = compile(region, opts);
            for _ in 0..2 {
                let mut ctx = initial(&mut rng);
                ctx.pc = CODE;
                ctx.x[4] = (DATA2 + rng.below(PAGE_SIZE / 2)) & !7;
                let data: Vec<u8> = (0..PAGE_SIZE).map(|_| rng.next() as u8).collect();
                let data2: Vec<u8> = (0..PAGE_SIZE).map(|_| rng.next() as u8).collect();
                let (all, _) = interp2(&words, &ctx, &data, &data2, 100_000);
                let total = (100_000 - all.left) as u64;
                for budget in 0..=total + 2 {
                    let (a, a2) = interp2(&words, &ctx, &data, &data2, budget);
                    let b = jit_run(&engine, &module, &ctx, &data, budget, true);
                    // Slow paths: steps, and memory side exits (a failed fill outside loops).
                    steps += b.steps.len() + usize::from(b.exit == abi::EXIT_MEM);
                    // A syscall ends the run in both; anything else the interpreter finishes.
                    let (c, c2) = if b.exit == abi::EXIT_SYSCALL {
                        let mut c = interp2(&words, &b.ctx, &b.data, &data2, 0).0;
                        c.exit = abi::EXIT_SYSCALL;
                        c.left = b.left;
                        (c, a2.clone())
                    } else {
                        interp2(&words, &b.ctx, &b.data, &data2, b.left as u64)
                    };
                    let what = format!(
                        "shape {s}, budget {budget} (exit {}):\n  {}",
                        b.exit,
                        show()
                    );
                    assert_eq!(a.ctx, c.ctx, "state, {what}");
                    assert_eq!(a.data, c.data, "page 1, {what}");
                    assert_eq!(a2, c2, "page 2, {what}");
                    assert_eq!((a.left, a.exit), (c.left, c.exit), "budget, exit, {what}");
                    runs += 1;
                }
            }
        }
    }
    println!("{regions} page regions, {runs} runs, {steps} steps");
    assert!(regions >= 90, "only {regions} page regions");
    assert!(steps >= 100, "only {steps} steps");
}

// ---- scalar floating point ----

/// A float operand that is never a NaN: ±0, ±inf, denormals, small integers, ordinary values.
fn fp_value(r: &mut Rng, d: bool) -> u64 {
    let sign = if d {
        r.below(2) << 63
    } else {
        r.below(2) << 31
    };
    let mag = match r.below(8) {
        0 => 0,
        1 => {
            if d {
                0x7ff0_0000_0000_0000
            } else {
                0x7f80_0000
            }
        }
        2 => r.below(if d { 1 << 52 } else { 1 << 23 }) | 1, // denormal
        3 => {
            let int = 1 + r.below(100);
            if d {
                (int as f64).to_bits()
            } else {
                u64::from((int as f32).to_bits())
            }
        }
        4 => {
            // around 2^31 and 2^63 (conversions saturate there)
            let e = [31.0f64, 32.0, 63.0, 64.0][r.below(4) as usize];
            let v = 2f64.powf(e) * (0.75 + (r.below(4) as f64) / 8.0);
            if d {
                v.to_bits()
            } else {
                u64::from((v as f32).to_bits())
            }
        }
        5 => {
            // halves and their neighbours (rounding ties), magnitude only
            let v = r.below(100) as f64 + 0.5;
            let nudge = r.below(3) as i64 - 1;
            if d {
                (v.to_bits() as i64 + nudge) as u64
            } else {
                u64::from(((v as f32).to_bits() as i64 + nudge) as u32)
            }
        }
        _ => {
            // finite, any exponent below the maximum
            if d {
                (r.next() & 0x000f_ffff_ffff_ffff) | (r.below(0x7fe) + 1) << 52
            } else {
                (r.next() & 0x007f_ffff) | (r.below(0xfe) + 1) << 23
            }
        }
    };
    sign | mag
}

/// Like [`class`] for scalar FP: the vector registers hold singles and doubles (one in eight a
/// NaN). The `jit_step` stub executes what steps (a NaN result, Arm's NaN rules), so every
/// sequence must end in the interpreter's state, whether it stepped or not.
fn class_fp(name: &str, seed: u64, make: impl Fn(&mut Rng) -> u32) {
    class_with(
        name,
        seed,
        |rng| {
            let d = rng.below(2) == 0;
            u128::from(fp_lane(rng, d)) | u128::from(rng.next()) << 64
        },
        make,
    );
}

/// A float lane (`d`: double) for the FP tests: one in eight a NaN (quiet or signalling, any
/// payload, either sign), else [`fp_value`].
fn fp_lane(rng: &mut Rng, d: bool) -> u64 {
    if rng.below(8) == 0 {
        if d {
            0x7ff0_0000_0000_0000 | (rng.next() & 0x800f_ffff_ffff_ffff) | 1
        } else {
            0x7f80_0000 | (rng.next() & 0x807f_ffff) | 1
        }
    } else {
        fp_value(rng, d)
    }
}

/// A vector register of float lanes: four singles or two doubles.
fn fp_vector(rng: &mut Rng) -> u128 {
    if rng.below(2) == 0 {
        u128::from(fp_lane(rng, true)) | u128::from(fp_lane(rng, true)) << 64
    } else {
        (0..4).fold(0u128, |acc, i| {
            acc | u128::from(fp_lane(rng, false)) << (32 * i)
        })
    }
}

/// A vector register for integer lanes: the extremes saturating arithmetic meets, small
/// values, or random bits.
fn int_vector(rng: &mut Rng) -> u128 {
    let lane = |rng: &mut Rng, esize: u32| -> u64 {
        let m = if esize == 64 {
            u64::MAX
        } else {
            (1 << esize) - 1
        };
        (match rng.below(6) {
            0 => 0,
            1 => m,
            2 => m >> 1,
            3 => (m >> 1) + 1,
            4 => rng.below(4),
            _ => rng.next(),
        }) & m
    };
    let esize = 8u32 << rng.below(4);
    (0..128 / esize).fold(0u128, |acc, i| {
        acc | u128::from(lane(rng, esize)) << (esize * i)
    })
}

/// The differential loop of the FP and vector tests: random sequences of up to four
/// instructions from `make`, vector registers from `vinit`. The `jit_step` stub executes what
/// steps, so every sequence must end in the interpreter's state, stepped or not.
fn class_with(
    name: &str,
    seed: u64,
    vinit: impl Fn(&mut Rng) -> u128,
    make: impl Fn(&mut Rng) -> u32,
) {
    let mut rng = Rng(seed);
    let (mut ran, mut stepped) = (0, 0);
    let mut blank = A64Context {
        pc: CODE,
        sp: DATA + 0x800,
        ..A64Context::default()
    };
    // Bases into the data page, as `initial` sets them (for the memory classes).
    for r in [1, 2, 3] {
        blank.x[r] = DATA;
    }
    let zeros = vec![0u8; PAGE_SIZE as usize];
    for _ in 0..400 {
        let len = 1 + rng.below(4) as usize;
        let mut words = Vec::new();
        while words.len() < len {
            let w = make(&mut rng);
            // Only encodings the interpreter accepts (SIMD undefinedness is state-free).
            if matches!(decode(w), wasmshim_aarch64::Inst::Undefined)
                || interp(&[w], &blank, &zeros, 1).exit != 0
            {
                continue;
            }
            words.push(w);
        }
        let mut ctx = initial(&mut rng);
        for v in ctx.v.iter_mut() {
            *v = vinit(&mut rng);
        }
        if X5_SMALL.with(|f| f.get()) {
            ctx.x[5] = rng.below(64);
        }
        let data: Vec<u8> = (0..PAGE_SIZE).map(|_| rng.next() as u8).collect();
        let budget = words.len() as u64;
        let a = interp(&words, &ctx, &data, budget);
        EXEC_STEPS.with(|e| *e.borrow_mut() = Some(words.clone()));
        let b = jit(&words, &ctx, &data, budget, true);
        EXEC_STEPS.with(|e| *e.borrow_mut() = None);
        let show = || {
            words
                .iter()
                .map(|w| format!("{w:08x} {:?}", decode(*w)))
                .collect::<Vec<_>>()
                .join("\n  ")
        };
        if b.exit == abi::EXIT_MEM {
            // A memory side exit (a fill that failed: a fault, a page crossing): nothing of
            // that instruction ran; the interpreter takes over from there.
            let c = interp(&words, &b.ctx, &b.data, b.left as u64);
            assert_eq!(
                a.ctx,
                c.ctx,
                "{name}: state after a side exit\n  {}",
                show()
            );
            assert_eq!(a.data, c.data, "{name}: memory after a side exit");
            assert_eq!((a.left, a.exit), (c.left, c.exit), "{name}: budget, exit");
            ran += 1;
            continue;
        }
        if a.exit != 0 {
            assert_eq!(
                b.exit,
                99,
                "{name}: translated what the interpreter rejects\n  {}",
                show()
            );
            continue;
        }
        if !b.steps.is_empty() {
            // A NaN result (or an unlowered form) stepped: the interpreter ran it, and the
            // translated code went on from there.
            stepped += 1;
        }
        assert_eq!(b.exit, abi::EXIT_JUMP, "{name}: exit\n  {}", show());
        assert_eq!(a.ctx, b.ctx, "{name}: state\n  {}", show());
        assert_eq!(a.data, b.data, "{name}: memory\n  {}", show());
        assert_eq!(a.left, b.left, "{name}: budget\n  {}", show());
        ran += 1;
    }
    println!("{name}: {ran} runs, {stepped} of them stepped");
    assert!(
        ran >= 100 && stepped < ran,
        "{name}: {ran} runs, {stepped} stepped"
    );
}

#[test]
fn scalar_fp_arithmetic_moves_and_rounding() {
    class_fp("fp2", 40, |r| {
        let opcode = r.below(9) as u32;
        0x1E20_0800 | (r.below(2) as u32) << 22 | vr(r) << 16 | opcode << 12 | vr(r) << 5 | vr(r)
    });
    class_fp("fp1", 41, |r| {
        let ftype = r.below(2) as u32;
        let opcode = match r.below(13) {
            n @ 0..=3 => n as u32,
            n @ 4..=7 => 4 + n as u32, // FRINTN, P, M, Z
            8 => 14,
            9 => 15,
            10 => 12, // FRINTA
            // FCVT to the other precision
            _ => 4 | (1 - ftype),
        };
        0x1E20_4000 | ftype << 22 | opcode << 15 | vr(r) << 5 | vr(r)
    });
    class_fp("fmov imm, fcsel", 42, |r| {
        let ftype = r.below(2) as u32;
        if r.below(2) == 0 {
            0x1E20_1000 | ftype << 22 | (r.below(256) as u32) << 13 | vr(r)
        } else {
            0x1E20_0C00
                | ftype << 22
                | vr(r) << 16
                | (r.below(16) as u32) << 12
                | vr(r) << 5
                | vr(r)
        }
    });
}

#[test]
fn scalar_fp_compares_set_the_flags() {
    class_fp("fcmp", 43, |r| {
        let opc = (r.below(4) as u32) << 3;
        0x1E20_2000 | (r.below(2) as u32) << 22 | vr(r) << 16 | vr(r) << 5 | opc
    });
    // FCMP then a conditional select reading the flags it set, and FCCMP.
    class_fp("fcmp + csel, fccmp", 44, |r| match r.below(3) {
        0 => 0x1E20_2000 | (r.below(2) as u32) << 22 | vr(r) << 16 | vr(r) << 5,
        1 => 0x9A80_0000 | rd(r) << 16 | (r.below(16) as u32) << 12 | rd(r) << 5 | rd(r),
        _ => {
            0x1E20_0400
                | (r.below(2) as u32) << 22
                | vr(r) << 16
                | (r.below(16) as u32) << 12
                | vr(r) << 5
                | (r.below(2) as u32) << 4
                | r.below(16) as u32
        }
    });
}

#[test]
fn scalar_fp_integer_conversions() {
    class_fp("scvtf/ucvtf, fcvt[npmz][su]", 45, |r| {
        let sf = r.below(2) as u32;
        let ftype = r.below(2) as u32;
        let (rmode, opcode) = if r.below(2) == 0 {
            (0u32, [2u32, 3, 4, 5][r.below(4) as usize])
        } else {
            (r.below(4) as u32, r.below(2) as u32)
        };
        sf << 31 | 0x1E20_0000 | ftype << 22 | rmode << 19 | opcode << 16 | rd(r) << 5 | rd(r)
    });
    // Fixed point: SCVTF/UCVTF from, FCVTZS/FCVTZU to a general register, 2^fbits scaling.
    class_fp("scvtf/ucvtf/fcvtz[su], fixed point", 47, |r| {
        let sf = r.below(2) as u32;
        let (rmode, opcode) = if r.below(2) == 0 {
            (0u32, 2 + r.below(2) as u32)
        } else {
            (3, r.below(2) as u32)
        };
        let scale = if sf == 1 {
            r.below(64)
        } else {
            32 + r.below(32)
        } as u32;
        sf << 31
            | 0x1E00_0000
            | (r.below(2) as u32) << 22
            | rmode << 19
            | opcode << 16
            | scale << 10
            | rd(r) << 5
            | rd(r)
    });
}

/// Scalar FP inside loop regions (block registers held across blocks and iterations, flags
/// from FCMP read by branches in later blocks, NaN results stepped and executed mid-region),
/// at every budget, against the interpreter.
#[test]
fn scalar_fp_in_loop_regions() {
    use wasmshim_aarch64_translate::aot::{PageProfile, translate_page_with};
    let mut rng = Rng(46);
    let (mut formed, mut runs, mut stepped) = (0, 0, 0);
    let fp = |r: &mut Rng| -> u32 {
        let v = |r: &mut Rng| r.below(8) as u32;
        let ftype = (r.below(2) as u32) << 22;
        match r.below(9) {
            0 | 1 => {
                0x1E20_0800 | ftype | v(r) << 16 | (r.below(9) as u32) << 12 | v(r) << 5 | v(r)
            }
            2 => {
                let op = [0u32, 1, 2, 3, 8, 9, 10, 11][r.below(8) as usize];
                0x1E20_4000 | ftype | op << 15 | v(r) << 5 | v(r)
            }
            3 => 0x1E20_2000 | ftype | v(r) << 16 | v(r) << 5 | (r.below(4) as u32) << 3,
            4 => 0x1E20_0C00 | ftype | v(r) << 16 | (r.below(16) as u32) << 12 | v(r) << 5 | v(r),
            5 => {
                0x1E20_0400
                    | ftype
                    | v(r) << 16
                    | (r.below(16) as u32) << 12
                    | v(r) << 5
                    | r.below(16) as u32
            }
            // scvtf d/s, x21..x26
            6 => 0x9E22_0000 | ftype | (21 + r.below(6) as u32) << 5 | v(r),
            // fcvtzs x21..x26, d/s
            7 => 0x9E38_0000 | ftype | v(r) << 5 | (21 + r.below(6) as u32),
            // fmov d/s, #imm
            _ => 0x1E20_1000 | ftype | (r.below(256) as u32) << 13 | v(r),
        }
    };
    for _ in 0..60 {
        let mut words: Vec<u32> = Vec::new();
        let body = |r: &mut Rng, words: &mut Vec<u32>, min: u64, spread: u64| {
            for _ in 0..min + r.below(spread) {
                words.push(fp(r));
            }
        };
        let b_cond = |r: &mut Rng, from: usize, to: usize| -> u32 {
            0x5400_0000 | (((to as i64 - from as i64) as u32) & 0x7_ffff) << 5 | r.below(14) as u32
        };
        let b_ne = |from: usize, to: usize| -> u32 {
            0x5400_0001 | (((to as i64 - from as i64) as u32) & 0x7_ffff) << 5
        };
        body(&mut rng, &mut words, 1, 3);
        words.push(0xd280_0014 | ((1 + rng.below(4)) as u32) << 5); // movz x20, #n
        let inner = words.len();
        body(&mut rng, &mut words, 1, 4);
        if rng.below(2) == 0 {
            // an FCMP, then a branch on its flags over part of the body
            words.push(
                0x1E20_2000
                    | (rng.below(2) as u32) << 22
                    | (rng.below(8) as u32) << 16
                    | (rng.below(8) as u32) << 5,
            );
            // FP work (maybe a stepped NaN) between the compare and the branch reading it
            for _ in 0..rng.below(3) {
                words.push(fp(&mut rng));
            }
            let at = words.len();
            words.push(0);
            body(&mut rng, &mut words, 1, 2);
            let j = words.len();
            words[at] = b_cond(&mut rng, at, j);
            body(&mut rng, &mut words, 0, 2);
        }
        words.push(0xf100_0694); // subs x20, x20, #1
        let at = words.len();
        words.push(b_ne(at, inner));
        body(&mut rng, &mut words, 0, 3);
        words.push(0xf100_0673); // subs x19, x19, #1
        let at = words.len();
        words.push(b_ne(at, 0));
        words.push(0xd400_0001);
        if words
            .iter()
            .any(|&w| matches!(decode(w), wasmshim_aarch64::Inst::Undefined))
        {
            continue;
        }
        let mut entries = vec![0u16];
        for (i, &w) in words.iter().enumerate() {
            let inst = decode(w);
            if wasmshim_aarch64_translate::ends_block(&inst) {
                let b = Block {
                    pc: 4 * i as u64,
                    insts: vec![inst],
                };
                for s in wasmshim_aarch64_translate::successors(&b) {
                    if (s as usize) < 4 * words.len() {
                        entries.push(s as u16);
                    }
                }
            }
        }
        let mut bytes = vec![0u8; PAGE_SIZE as usize];
        for (i, w) in words.iter().enumerate() {
            bytes[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        let t = translate_page_with(&PageProfile { bytes, entries }, Options::default());
        let Some(region) = t
            .regions
            .iter()
            .find(|r| r.nest.is_some() && r.blocks[0].pc == 0)
        else {
            continue;
        };
        formed += 1;
        let (engine, module) = compile(region, Options::default());
        let show = || {
            words
                .iter()
                .enumerate()
                .map(|(i, w)| format!("{:3x}: {w:08x} {:?}", 4 * i, decode(*w)))
                .collect::<Vec<_>>()
                .join("\n  ")
        };
        for _ in 0..3 {
            let mut ctx = initial(&mut rng);
            ctx.pc = CODE;
            ctx.x[19] = 1 + rng.below(4);
            for v in ctx.v.iter_mut() {
                let d = rng.below(2) == 0;
                let mut x = fp_value(&mut rng, d);
                if rng.below(8) == 0 {
                    x = if d {
                        0x7ff0_0000_0000_0000 | (rng.next() & 0x800f_ffff_ffff_ffff) | 1
                    } else {
                        0x7f80_0000 | (rng.next() & 0x807f_ffff) | 1
                    };
                }
                *v = u128::from(x) | u128::from(rng.next()) << 64;
            }
            let data: Vec<u8> = (0..PAGE_SIZE).map(|_| rng.next() as u8).collect();
            let all = interp(&words, &ctx, &data, 100_000);
            let total = (100_000 - all.left) as u64;
            for budget in 0..=total + 2 {
                let a = interp(&words, &ctx, &data, budget);
                EXEC_STEPS.with(|e| *e.borrow_mut() = Some(words.clone()));
                let b = jit_run(&engine, &module, &ctx, &data, budget, true);
                EXEC_STEPS.with(|e| *e.borrow_mut() = None);
                if !b.steps.is_empty() {
                    stepped += 1;
                }
                let c = if b.exit == abi::EXIT_SYSCALL {
                    let mut c = interp(&words, &b.ctx, &b.data, 0);
                    c.exit = abi::EXIT_SYSCALL;
                    c.left = b.left;
                    c
                } else {
                    interp(&words, &b.ctx, &b.data, b.left as u64)
                };
                let what = format!(
                    "budget {budget} (exit {}, steps {:x?}):\n  {}",
                    b.exit,
                    b.steps,
                    show()
                );
                assert_eq!(a.ctx, c.ctx, "state, {what}");
                assert_eq!((a.left, a.exit), (c.left, c.exit), "budget, exit, {what}");
                runs += 1;
            }
        }
    }
    println!("{formed} FP loop regions, {runs} runs, {stepped} with steps");
    assert!(formed >= 40, "only {formed} loop regions formed");
    assert!(stepped >= 50, "only {stepped} runs stepped");
}

// ---- Advanced SIMD lane arithmetic (a64/neon.rs) ----

fn bit(r: &mut Rng) -> u32 {
    r.below(2) as u32
}

/// Advanced SIMD three same (vector or scalar) with an opcode from `opcodes`.
fn three_same(r: &mut Rng, opcodes: &[u32], scalar: bool) -> u32 {
    let base = if scalar { 0x5E20_0400 } else { 0x0E20_0400 };
    let opcode = opcodes[r.below(opcodes.len() as u64) as usize];
    base | if scalar { 0 } else { bit(r) << 30 }
        | bit(r) << 29
        | (r.below(4) as u32) << 22
        | vr(r) << 16
        | opcode << 11
        | vr(r) << 5
        | vr(r)
}

/// Advanced SIMD two-register miscellaneous (vector or scalar).
fn two_misc(r: &mut Rng, opcodes: &[u32], scalar: bool) -> u32 {
    let base = if scalar { 0x5E20_0800 } else { 0x0E20_0800 };
    let opcode = opcodes[r.below(opcodes.len() as u64) as usize];
    base | if scalar { 0 } else { bit(r) << 30 }
        | bit(r) << 29
        | (r.below(4) as u32) << 22
        | opcode << 12
        | vr(r) << 5
        | vr(r)
}

#[test]
fn vector_fp_three_same() {
    let fp_ops: Vec<u32> = (0b11000..=0b11111).collect();
    class_with("fp three same", 50, fp_vector, |r| {
        three_same(r, &fp_ops, false)
    });
    class_with("fp three same, scalar", 51, fp_vector, |r| {
        three_same(r, &fp_ops, true)
    });
}

#[test]
fn vector_fp_two_misc() {
    let fp_ops: Vec<u32> = (0b01100..=0b11111)
        .filter(|o| !(0b10010..=0b10100).contains(o))
        .collect();
    class_with("fp two misc", 52, fp_vector, |r| {
        two_misc(r, &fp_ops, false)
    });
    class_with("fp two misc, scalar", 53, fp_vector, |r| {
        two_misc(r, &fp_ops, true)
    });
    // Conversions from integer lanes.
    class_with("scvtf/ucvtf vector", 54, int_vector, |r| {
        let scalar = r.below(4) == 0;
        two_misc(r, &[0b11101], scalar)
    });
}

#[test]
fn vector_integer_three_same() {
    let ops: Vec<u32> = (0..=0b10111).collect();
    class_with("int three same", 55, int_vector, |r| {
        three_same(r, &ops, false)
    });
    // The ones neon.rs lowers, more often.
    class_with("int three same, lowered", 56, int_vector, |r| {
        three_same(
            r,
            &[
                0b00001, 0b00101, 0b01100, 0b01101, 0b01110, 0b01111, 0b10000, 0b10010, 0b10011,
            ],
            false,
        )
    });
}

#[test]
fn vector_integer_two_misc() {
    let ops: Vec<u32> = (0..=0b01011).chain(0b10010..=0b10100).collect();
    class_with("int two misc", 57, int_vector, |r| two_misc(r, &ops, false));
    class_with("int two misc, scalar", 58, int_vector, |r| {
        two_misc(r, &[0b01011, 0b01000, 0b01001, 0b01010], true)
    });
}

#[test]
fn vector_shifts_by_immediate() {
    let shift = |r: &mut Rng, scalar: bool| -> u32 {
        let base = if scalar { 0x5F00_0400 } else { 0x0F00_0400 };
        let opcode = [
            0b00000, 0b00010, 0b00100, 0b01000, 0b01010, 0b10000, 0b10001, 0b10100, 0b01110,
        ][r.below(9) as usize];
        base | if scalar { 0 } else { bit(r) << 30 }
            | bit(r) << 29
            | (1 + r.below(15) as u32) << 19
            | (r.below(8) as u32) << 16
            | opcode << 11
            | vr(r) << 5
            | vr(r)
    };
    class_with("shift imm", 59, int_vector, |r| shift(r, false));
    class_with("shift imm, scalar", 60, int_vector, |r| shift(r, true));
    // Fixed-point conversions (single lanes: immh 01xx).
    let fixed = |r: &mut Rng| -> u32 {
        let scalar = r.below(4) == 0;
        let base = if scalar { 0x5F00_0400 } else { 0x0F00_0400 };
        base | if scalar { 0 } else { bit(r) << 30 }
            | bit(r) << 29
            | (4 + r.below(4) as u32) << 19
            | (r.below(8) as u32) << 16
            | [0b11100u32, 0b11111][r.below(2) as usize] << 11
            | vr(r) << 5
            | vr(r)
    };
    class_with(
        "fixed point, vector",
        72,
        |r| {
            if r.below(2) == 0 {
                fp_vector(r)
            } else {
                int_vector(r)
            }
        },
        fixed,
    );
}

#[test]
fn vector_across_lanes() {
    let across = |r: &mut Rng| -> u32 {
        let opcode = [0b00011, 0b01010, 0b11010, 0b11011, 0b01100, 0b01111][r.below(6) as usize];
        0x0E30_0800
            | bit(r) << 30
            | bit(r) << 29
            | (r.below(4) as u32) << 22
            | opcode << 12
            | vr(r) << 5
            | vr(r)
    };
    class_with("across lanes", 61, int_vector, across);
    class_with("across lanes, fp", 62, fp_vector, across);
}

#[test]
fn vector_by_element() {
    let indexed = |r: &mut Rng| -> u32 {
        let scalar = r.below(4) == 0;
        let base = if scalar { 0x5F00_0000 } else { 0x0F00_0000 };
        let opcode = [0b1001, 0b1000, 0b0000, 0b0100, 0b0001][r.below(5) as usize];
        base | if scalar { 0 } else { bit(r) << 30 }
            | bit(r) << 29
            | (r.below(4) as u32) << 22
            | bit(r) << 21
            | bit(r) << 20
            | (r.below(16) as u32) << 16
            | opcode << 12
            | bit(r) << 11
            | vr(r) << 5
            | vr(r)
    };
    class_with("by element, fp", 63, fp_vector, indexed);
    class_with("by element, int", 64, int_vector, indexed);
}

#[test]
fn vector_three_different_and_scalar_forms() {
    let three_diff = |r: &mut Rng| -> u32 {
        let opcode = [
            0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111, 0b1000, 0b1010, 0b1100,
            0b1110,
        ][r.below(12) as usize];
        0x0E20_0000
            | bit(r) << 30
            | bit(r) << 29
            | (r.below(4) as u32) << 22
            | vr(r) << 16
            | opcode << 12
            | vr(r) << 5
            | vr(r)
    };
    class_with("three different", 65, int_vector, three_diff);
    class_with("scalar dup", 66, int_vector, |r| {
        0x5E00_0400 | (1 + r.below(31) as u32) << 16 | vr(r) << 5 | vr(r)
    });
    let pairwise = |r: &mut Rng| -> u32 {
        let opcode = [0b11011, 0b01100, 0b01101, 0b01111][r.below(4) as usize];
        0x5E30_0800 | bit(r) << 29 | (r.below(4) as u32) << 22 | opcode << 12 | vr(r) << 5 | vr(r)
    };
    class_with("scalar pairwise, fp", 67, fp_vector, pairwise);
    class_with("scalar pairwise, int", 68, int_vector, pairwise);
    class_with("sshl/ushl", 69, int_vector, |r| {
        three_same(r, &[0b01000], false)
    });
}

#[test]
fn vector_structure_loads_and_stores() {
    // Bases x1-x3 (into the data page) or SP; post-index by the size (rm = 31) or by x5
    // (set small here), or none.
    let base = |r: &mut Rng| -> u32 { [1u32, 2, 3, 31][r.below(4) as usize] };
    let post = |r: &mut Rng| -> (u32, u32) {
        match r.below(3) {
            0 => (0, 0),
            1 => (1, 31),
            _ => (1, 5),
        }
    };
    let multi = |r: &mut Rng| -> u32 {
        let opcode =
            [0b0111u32, 0b1010, 0b0110, 0b0010, 0b1000, 0b0100, 0b0000][r.below(7) as usize];
        let (p, rm) = post(r);
        0x0C00_0000
            | bit(r) << 30
            | p << 23
            | bit(r) << 22
            | rm << 16
            | opcode << 12
            | (r.below(4) as u32) << 10
            | base(r) << 5
            | vr(r)
    };
    let single = |r: &mut Rng| -> u32 {
        let opcode = [0b000u32, 0b010, 0b100, 0b110, 0b001, 0b101][r.below(6) as usize];
        let (p, rm) = post(r);
        0x0D00_0000
            | bit(r) << 30
            | p << 23
            | bit(r) << 22
            | bit(r) << 21
            | rm << 16
            | opcode << 13
            | bit(r) << 12
            | (r.below(4) as u32) << 10
            | base(r) << 5
            | vr(r)
    };
    let init = |r: &mut Rng| int_vector(r);
    class_with_x5("ld1/st1 multiple", 70, init, multi);
    class_with_x5("ld1-4/st1-4 single", 71, init, single);
}

/// [`class_with`] with x5 small (a post-index register that stays inside the data page).
fn class_with_x5(
    name: &str,
    seed: u64,
    vinit: impl Fn(&mut Rng) -> u128,
    make: impl Fn(&mut Rng) -> u32,
) {
    X5_SMALL.with(|f| f.set(true));
    class_with(name, seed, vinit, make);
    X5_SMALL.with(|f| f.set(false));
}

thread_local! {
    static X5_SMALL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Fused multiply-add, emulated exactly (rounding to odd): random and adversarial operands
/// (cancellation, ties, extreme exponents, zeros, infinities, NaNs), all four forms, single and
/// double, against the interpreter's correctly rounded `fma`.
#[test]
fn fused_multiply_add_is_exact() {
    let mut rng = Rng(73);
    let data = vec![0u8; PAGE_SIZE as usize];
    let (mut ran, mut stepped) = (0u64, 0u64);
    for ftype in [0u32, 1] {
        for o in 0..4u32 {
            // fmadd-family d0/s0, d1, d2, d3
            let w = 0x1F00_0000
                | ftype << 22
                | (o >> 1) << 21
                | 2 << 16
                | (o & 1) << 15
                | 3 << 10
                | 1 << 5;
            let insts = vec![decode(w)];
            let region = Region {
                blocks: vec![Block { pc: CODE, insts }],
                nest: None,
            };
            let (engine, module) = compile(&region, Options::default());
            let d = ftype == 1;
            // FMA_RUNS=N for a longer search.
            let runs: u64 = std::env::var("FMA_RUNS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4000);
            for i in 0..runs {
                let r = &mut rng;
                let mant = |r: &mut Rng, d: bool| -> u64 {
                    if d {
                        r.next() & 0x000f_ffff_ffff_ffff
                    } else {
                        r.next() & 0x007f_ffff
                    }
                };
                let make = |r: &mut Rng, e: i64, d: bool| -> u64 {
                    let s = r.below(2);
                    if d {
                        s << 63 | ((1023 + e) as u64) << 52 | mant(r, d)
                    } else {
                        s << 31 | ((127 + e.clamp(-126, 127)) as u64) << 23 | mant(r, d)
                    }
                };
                let span = if d { 60 } else { 30 };
                let (x, y) = (
                    {
                        let e = r.below(2 * span) as i64 - span as i64;
                        make(r, e, d)
                    },
                    {
                        let e = r.below(2 * span) as i64 - span as i64;
                        make(r, e, d)
                    },
                );
                let z = match i % 6 {
                    // near -x*y: cancellation
                    0 | 1 => {
                        let p = if d {
                            (f64::from_bits(x) * f64::from_bits(y)).to_bits()
                        } else {
                            u64::from(
                                (f32::from_bits(x as u32) * f32::from_bits(y as u32)).to_bits(),
                            )
                        };
                        let sign = if d { 1u64 << 63 } else { 1 << 31 };
                        (p ^ sign).wrapping_add(r.below(5)).wrapping_sub(2)
                    }
                    2 => 0,
                    3 => fp_lane(r, d),
                    _ => {
                        let e = r.below(2 * span) as i64 - span as i64;
                        make(r, e, d)
                    }
                };
                let (x, y) = if i % 17 == 0 {
                    (fp_lane(r, d), fp_lane(r, d))
                } else {
                    (x, y)
                };
                let mut ctx = A64Context {
                    pc: CODE,
                    ..A64Context::default()
                };
                ctx.v[1] = u128::from(x);
                ctx.v[2] = u128::from(y);
                ctx.v[3] = u128::from(z);
                let a = interp(&[w], &ctx, &data, 1);
                EXEC_STEPS.with(|e| *e.borrow_mut() = Some(vec![w]));
                let b = jit_run(&engine, &module, &ctx, &data, 1, true);
                EXEC_STEPS.with(|e| *e.borrow_mut() = None);
                if !b.steps.is_empty() {
                    stepped += 1;
                }
                assert_eq!(
                    a.ctx.v[0],
                    b.ctx.v[0],
                    "{w:08x} {:?}: x {x:x} y {y:x} z {z:x}",
                    decode(w)
                );
                assert_eq!(a.ctx, b.ctx);
                ran += 1;
            }
        }
    }
    println!("fma: {ran} runs, {stepped} stepped");
    assert!(stepped * 5 < ran, "{stepped} of {ran} stepped");
}

/// ADCS/SBCS after flags of every kind, and flag readers after them.
#[test]
fn add_and_subtract_with_carry_set_the_flags() {
    class("adcs/sbcs, cmp, csel", 76, |r| {
        let sf = bit(r) << 31;
        match r.below(5) {
            // ADC, ADCS, SBC, SBCS
            0 | 1 => {
                sf | [0x1A00_0000u32, 0x3A00_0000, 0x5A00_0000, 0x7A00_0000][r.below(4) as usize]
                    | rd(r) << 16
                    | rd(r) << 5
                    | rd(r)
            }
            // CMP (sets C every way), ADDS
            2 => sf | 0x6B00_001F | rd(r) << 16 | rd(r) << 5,
            3 => sf | 0x2B00_0000 | rd(r) << 16 | rd(r) << 5 | rd(r),
            // CSEL on any condition
            _ => sf | 0x1A80_0000 | rd(r) << 16 | (r.below(16) as u32) << 12 | rd(r) << 5 | rd(r),
        }
    });
}

/// TBL/TBX with one to four table registers, indices in and past the table.
#[test]
fn table_lookups_of_every_length() {
    let tbl = |r: &mut Rng| -> u32 {
        0x0E00_0000
            | bit(r) << 30
            | vr(r) << 16
            | (r.below(4) as u32) << 13
            | bit(r) << 12
            | vr(r) << 5
            | vr(r)
    };
    // Bytes below 80: every table length has indices on both sides of its end.
    let indices = |r: &mut Rng| -> u128 {
        (0..16).fold(0u128, |acc, i| {
            acc | u128::from(r.below(80) as u8) << (8 * i)
        })
    };
    class_with("tbl/tbx", 74, indices, tbl);
    class_with("tbl/tbx, any bytes", 75, int_vector, tbl);
}

/// Straight-line code longer than a block: the page's regions continue it in more
/// blocks, and a run through the cuts, at every budget, ends where the interpreter ends.
#[test]
fn long_blocks_continue_across_the_cut() {
    use wasmshim_aarch64_translate::aot::{PageProfile, translate_page_with};
    let mut rng = Rng(81);
    for _ in 0..3 {
        // 600 ALU instructions on x21..x26, then `svc #0`.
        let mut words: Vec<u32> = (0..600)
            .map(|_| {
                let rd = 21 + rng.below(6) as u32;
                let rm = 21 + rng.below(6) as u32;
                match rng.below(3) {
                    0 => 0x9100_0000 | (rng.below(4096) as u32) << 10 | rd << 5 | rd,
                    1 => 0xca00_0000 | rm << 16 | rd << 5 | rd,
                    _ => 0x8b00_0000 | rm << 16 | rd << 5 | rd,
                }
            })
            .collect();
        words.push(0xd400_0001);
        let mut bytes = vec![0u8; PAGE_SIZE as usize];
        for (i, w) in words.iter().enumerate() {
            bytes[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        let t = translate_page_with(
            &PageProfile {
                bytes,
                entries: vec![0],
            },
            Options::default(),
        );
        let region = t
            .regions
            .iter()
            .find(|r| r.blocks.iter().any(|b| b.pc == 0))
            .expect("the entry's region");
        assert!(
            t.regions.iter().map(|r| r.blocks.len()).sum::<usize>() >= 3,
            "the cut blocks continue"
        );
        let (engine, module) = compile(region, Options::default());
        let ctx = A64Context {
            pc: CODE,
            ..initial(&mut rng)
        };
        let data: Vec<u8> = (0..PAGE_SIZE).map(|_| rng.next() as u8).collect();
        let data2 = vec![0u8; PAGE_SIZE as usize];
        for budget in (0..=605).step_by(7) {
            let (a, _) = interp2(&words, &ctx, &data, &data2, budget);
            let b = jit_run(&engine, &module, &ctx, &data, budget, true);
            assert!(b.steps.is_empty());
            let c = if b.exit == abi::EXIT_SYSCALL {
                let mut c = interp2(&words, &b.ctx, &b.data, &data2, 0).0;
                c.exit = abi::EXIT_SYSCALL;
                c.left = b.left;
                c
            } else {
                interp2(&words, &b.ctx, &b.data, &data2, b.left as u64).0
            };
            assert_eq!(a.ctx, c.ctx, "budget {budget} (exit {})", b.exit);
            assert_eq!((a.left, a.exit), (c.left, c.exit), "budget {budget}");
        }
    }
}

/// Ties away from zero (`FRINTA`, `FCVTAS`/`FCVTAU`, scalar and vector) on its edge cases.
#[test]
fn round_away_edges() {
    let values: Vec<f64> = vec![
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        0.49999997,
        -0.49999997,
        0.3,
        -0.3,
        0.7,
        -0.7,
        8388607.5,
        8388609.0,
        4503599627370495.5,
        1e300,
        -1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2147483647.5,
        -2147483648.5,
        4294967295.5,
        1e-310,
        -1e-40,
    ];
    let shapes: &[(&str, u32)] = &[
        ("frinta s0, s1", 0x1E26_4020),
        ("frinta d0, d1", 0x1E66_4020),
        ("fcvtas w0, s1", 0x1E24_0020),
        ("fcvtas x0, d1", 0x9E64_0020),
        ("fcvtau w0, d1", 0x1E65_0020),
        ("frinta v0.4s, v1.4s", 0x6E21_8820),
        ("frinta v0.2d, v1.2d", 0x6E61_8820),
        ("fcvtas v0.4s, v1.4s", 0x4E21_C820),
        ("fcvtau v0.4s, v1.4s", 0x6E21_C820),
    ];
    let data = vec![0u8; PAGE_SIZE as usize];
    for &(name, w) in shapes {
        for &v in &values {
            let mut ctx = A64Context {
                pc: CODE,
                ..A64Context::default()
            };
            let s = u128::from((v as f32).to_bits());
            ctx.v[1] = if name.contains("d1") || name.contains("2d") {
                u128::from(v.to_bits()) | u128::from((-v).to_bits()) << 64
            } else {
                s | s << 32 | (u128::from((-v as f32).to_bits())) << 64 | s << 96
            };
            let a = interp(&[w], &ctx, &data, 1);
            assert_eq!(a.exit, 0, "{name}");
            let b = jit(&[w], &ctx, &data, 1, true);
            assert!(b.steps.is_empty(), "{name} of {v} stepped");
            assert_eq!(a.ctx, b.ctx, "{name} of {v}");
        }
    }
}

/// Common vector shapes (float lane arithmetic and compares, integer min/max, widening
/// reductions) run inline instead of stepping.
#[test]
fn common_vector_shapes_run_inline() {
    let shapes: &[(&str, u32)] = &[
        ("uaddlv h0, v1.8b", 0x2E30_3820),
        ("fcmge v0.2s, v1.2s, v2.2s", 0x2E22_E420),
        ("fadd v0.2s, v1.2s, v2.2s", 0x0E22_D420),
        ("smax v0.2s, v1.2s, v2.2s", 0x0EA2_6420),
        ("fsub v0.2s, v1.2s, v2.2s", 0x0EA2_D420),
        ("fcmgt v0.2s, v1.2s, v2.2s", 0x2EA2_E420),
        ("fmul v0.2s, v1.2s, v2.2s", 0x2E22_DC20),
        ("sqadd v0.2s, v1.2s, v2.2s", 0x0EA2_0C20),
        ("fmaxnm v0.2s, v1.2s, v2.2s", 0x0E22_C420),
        ("fadd v0.4s, v1.4s, v2.4s", 0x4E22_D420),
        ("sqsub v0.2s, v1.2s, v2.2s", 0x0EA2_2C20),
        ("fabd s0, s1, s2", 0x7EA2_D420),
        ("fmul v0.4s, v1.4s, v2.4s", 0x6E22_DC20),
        ("add v0.2d, v1.2d, v2.2d", 0x4EE2_8420),
        ("mul v0.8h, v1.8h, v2.8h", 0x4E62_9C20),
        ("fdiv v0.2s, v1.2s, v2.2s", 0x2E22_FC20),
        ("scvtf s0, s1", 0x5E21_D820),
        ("scvtf v0.2s, v1.2s", 0x0E21_D820),
        ("fcvtzs v0.2s, v1.2s", 0x0EA1_B820),
        ("ucvtf v0.2s, v1.2s", 0x2E21_D820),
        ("xtn v0.2s, v1.2d", 0x0EA1_2820),
        ("xtn v0.4h, v1.4s", 0x0E61_2820),
        ("rev64 v0.2s, v1.2s", 0x0EA0_0820),
        ("shll v0.2d, v1.2s, #32", 0x2EA1_3820),
        ("frintm v0.2s, v1.2s", 0x0E21_9820),
        ("ushll v0.8h, v1.8b, #0", 0x2F08_A420),
        ("sshll v0.8h, v1.8b, #0", 0x0F08_A420),
        ("shrn v0.8b, v1.8h, #8", 0x0F08_8420),
        ("ushr v0.4s, v1.4s, #1", 0x6F3F_0420),
        ("umaxv b0, v1.16b", 0x6E30_A820),
        ("fmul v0.2s, v1.2s, v2.s[1]", 0x0FA2_9020),
        ("fmul v0.4s, v1.4s, v2.s[0]", 0x4F82_9020),
        ("sub v0.4s, v1.4s, v2.4s", 0x6EA2_8420),
    ];
    let mut ctx = A64Context {
        pc: CODE,
        ..A64Context::default()
    };
    // 1.0f in every single lane: no NaN, no saturation, no division by zero.
    for v in ctx.v.iter_mut() {
        *v = (0..4).fold(0u128, |acc, i| {
            acc | u128::from(1.0f32.to_bits()) << (32 * i)
        });
    }
    let data = vec![0u8; PAGE_SIZE as usize];
    for &(name, w) in shapes {
        let a = interp(&[w], &ctx, &data, 1);
        assert_eq!(
            a.exit,
            0,
            "{name}: the interpreter accepts it ({:?})",
            decode(w)
        );
        let b = jit(&[w], &ctx, &data, 1, true);
        assert!(
            b.steps.is_empty(),
            "{name} ({w:08x}) still steps: {:?}",
            decode(w)
        );
        assert_eq!(a.ctx, b.ctx, "{name}");
    }
}
