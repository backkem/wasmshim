//! The two-tier dispatcher: translated regions where they exist, the interpreter one basic
//! block at a time elsewhere, until the budget is used up or an event happens. Shared by the
//! in-process engine ([`crate::local`]) and the host-instantiated core module.
//!
//! `unsafe` is confined here: translated code reaches the CPU and the address space through
//! raw pointers (it shares this instance's memory), and `linux_jit_step` re-enters the engine
//! while the dispatcher waits in a translated call, so neither may be borrowed across that call.

#![allow(unsafe_code)]

extern crate alloc;

use wasmshim_aarch64::Cpu;
use wasmshim_aarch64_translate::abi as jit;
use wasmshim_aarch64_translate::abi::CTX_SIZE;
use wasmshim_cpu::{Event, GuestCpu, SyscallFrame};

use crate::mem::{ChainState, EngineMem};
use crate::state::instance_state;

/// `jit_step` statuses (see [`wasmshim_aarch64_translate::abi::STEP_STATUS_MIN`]).
pub const STATUS_FAULT: i32 = jit::STEP_STATUS_MIN;
/// The stepped instruction retired and wrote into code: the rest of the region is stale.
pub const STATUS_CODE: i32 = jit::STEP_STATUS_MIN + 1;
/// The stepped instruction retired and raised an event (a system call): the dispatcher stops
/// and hands the event to its caller.
pub const STATUS_EVENT: i32 = jit::STEP_STATUS_MIN + 2;

/// Engine counters (never guest-visible).
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub jit_calls: u64,
    pub interp_blocks: u64,
    pub compiles: u64,
    pub wasm_bytes: u64,
    pub budget_exits: u64,
    /// Translated code that returned to the dispatcher to continue at another PC (`EXIT_JUMP`:
    /// its chain-table lookup failed), by the chain entry for that PC: none, from an older code
    /// epoch, or current (statistics only; chained transitions themselves are not counted).
    pub exit_jumps: u64,
    pub jumps_absent: u64,
    pub jumps_stale: u64,
    pub jumps_current: u64,
    /// Of `jumps_absent`, those whose target has a translation (the dispatcher enters it): its
    /// chain slot holds another PC, or it was never dispatched before. The rest run interpreted.
    pub jumps_absent_translated: u64,
    /// Code-epoch moves the dispatcher's lookups saw (each makes every chain entry stale).
    pub epoch_moves: u64,
    pub steps: u64,
    pub insns_jit: u64,
    pub insns_interp: u64,
    pub insns_translated: u64,
    /// 1 if the last dispatch ran out of budget inside a translated region (`EXIT_BUDGET`), 0
    /// otherwise: at an instruction-exact cut (a debugger's or profiler's `insn` stop) it
    /// says which tier the thread was executing. Read-only bookkeeping.
    pub last_cut_translated: u64,
}

/// Tiering policy and counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tier {
    /// Interpreted executions of a block before the dispatcher reports it hot ([`Stop::Hot`])
    /// for the run-time tier (`0`: never; translated code then comes only from the AOT
    /// directory).
    pub threshold: u32,
    pub stats: Stats,
}

/// Why the dispatcher stopped.
pub enum Stop {
    /// The run ended (the budget, a syscall, a fault).
    Event(Event),
    /// The block at this PC reached the run-time tier's threshold; nothing of it ran yet. The
    /// caller may translate its page ([`crate::jit`]) and dispatch again.
    Hot(u64),
}

/// What `linux_jit_step` works on while translated code runs.
struct Current {
    cpu: *mut Cpu,
    mem: *mut EngineMem,
    event: Option<Event>,
    steps: u64,
}

instance_state! {
    fn current_cell() -> *mut Current = Current {
        cpu: core::ptr::null_mut(),
        mem: core::ptr::null_mut(),
        event: None,
        steps: 0,
    };
}

fn current() -> &'static mut Current {
    // SAFETY: one thread reaches the cell (`state`); only the dispatcher, `step` and `fill`
    // touch it, never at the same time through a live reference (the dispatcher re-derives
    // after each translated call).
    unsafe { &mut *current_cell() }
}

instance_state! {
    /// The context block of translated calls (`left`, `entry`, `base`).
    fn ctx_block() -> *mut [u64; CTX_SIZE / 8] = [0; CTX_SIZE / 8];
}

const _: () = assert!(CTX_SIZE == 40);

/// Calls translated code at table index `slot`.
#[cfg(target_arch = "wasm32")]
fn exec(slot: u32, cpu: usize, tlb: usize, ctx: usize) -> i32 {
    // SAFETY: `slot` holds a function of exactly this type (the translator's entry type), put
    // there by the linker (AOT) or the host (run-time translation); a wrong type traps in
    // `call_indirect`, it never misbehaves. Slot 0 (no function) becomes `None`, never a null
    // function value.
    let f: Option<extern "C" fn(i32, i32, i32) -> i32> =
        unsafe { core::mem::transmute(slot as usize) };
    let f = f.expect("translated code at table slot 0");
    f(cpu as i32, tlb as i32, ctx as i32)
}

/// Natively there is no translated code (it is Wasm): no translation is ever installed.
#[cfg(not(target_arch = "wasm32"))]
fn exec(_slot: u32, _cpu: usize, _tlb: usize, _ctx: usize) -> i32 {
    unreachable!("translated code only exists in a Wasm engine")
}

/// Runs `cpu` on `mem` for at most `*left` instructions (decremented as they retire).
///
/// # Safety
///
/// `cpu` and `mem` must be valid and not otherwise borrowed for the duration of the call.
pub unsafe fn dispatch(
    cpu: *mut Cpu,
    mem: *mut EngineMem,
    left: &mut u64,
    tier: &mut Tier,
) -> Stop {
    let cur = current();
    cur.cpu = cpu;
    cur.mem = mem;
    let ctx = ctx_block() as *mut u64;
    tier.stats.last_cut_translated = 0;
    // The last region left with `EXIT_JUMP` to a target with no chain entry (statistics).
    let mut absent_jump = false;
    // SAFETY: see the function contract; no reference to `*cpu`/`*mem` lives across `exec`.
    let stop = unsafe {
        loop {
            if *left == 0 {
                break Stop::Event(Event::Yield);
            }
            let pc = (*cpu).pc();
            let generation = (*mem).generation();
            let found = (*mem).lookup(pc);
            if (*mem).generation() != generation {
                tier.stats.epoch_moves += 1;
            }
            if absent_jump && found.is_some() {
                tier.stats.jumps_absent_translated += 1;
            }
            absent_jump = false;
            if let Some((slot, idx)) = found {
                tier.stats.jit_calls += 1;
                *ctx = *left;
                *ctx.add(1) = u64::from(idx);
                *ctx.add(2) = pc & !0xfff;
                let (chain, epoch) = (*mem).chain();
                *ctx.add(3) = chain as u64;
                *ctx.add(4) = epoch;
                let tlb = (*mem).tlb_ptr();
                let steps = current().steps;
                let code = exec(slot, cpu as usize, tlb, ctx as usize);
                tier.stats.steps += current().steps - steps;
                let after = *ctx;
                tier.stats.insns_jit += *left - after;
                let retired = (*cpu).retired() + (*left - after);
                (*cpu).set_retired(retired);
                *left = after;
                match code {
                    jit::EXIT_JUMP => {
                        tier.stats.exit_jumps += 1;
                        match (*mem).chain_state((*cpu).pc()) {
                            ChainState::Absent => {
                                tier.stats.jumps_absent += 1;
                                absent_jump = true;
                            }
                            ChainState::Stale => tier.stats.jumps_stale += 1,
                            ChainState::Current => tier.stats.jumps_current += 1,
                        }
                        continue;
                    }
                    jit::EXIT_BUDGET => {
                        // Less than a block's worth left: finish exactly in the interpreter.
                        tier.stats.budget_exits += 1;
                        tier.stats.last_cut_translated = 1;
                        let before = (*cpu).retired();
                        let ev = (*cpu).run_raw(&mut *mem, *left);
                        let ran = (*cpu).retired() - before;
                        tier.stats.insns_interp += ran;
                        *left -= ran;
                        break Stop::Event(ev);
                    }
                    jit::EXIT_SYSCALL => break Stop::Event(Event::Syscall(frame(&*cpu))),
                    jit::EXIT_MEM => {
                        // A memory access translated code left to us: run that one
                        // instruction here, filling the TLB as `jit_step` does, and go on (the
                        // rest of its block runs interpreted, the next block translated).
                        tier.stats.steps += 1;
                        let before = (*cpu).retired();
                        (*mem).filling.set(true);
                        let ev = (*cpu).run_raw(&mut *mem, 1);
                        (*mem).filling.set(false);
                        let ran = (*cpu).retired() - before;
                        tier.stats.insns_interp += ran;
                        *left -= ran;
                        match ev {
                            Event::Yield => continue,
                            ev => break Stop::Event(ev),
                        }
                    }
                    STATUS_FAULT => {
                        break Stop::Event(current().event.take().unwrap_or(Event::Yield));
                    }
                    STATUS_CODE => {
                        // The stepped store retired; the translation may be stale.
                        *left -= 1;
                        (*cpu).set_retired(retired + 1);
                        continue;
                    }
                    _ => {
                        // STATUS_EVENT: the stepped instruction retired with an event.
                        *left -= 1;
                        (*cpu).set_retired(retired + 1);
                        break Stop::Event(current().event.take().unwrap_or(Event::Yield));
                    }
                }
            }
            if tier.threshold > 0 && (*mem).bump(pc) == tier.threshold {
                break Stop::Hot(pc);
            }
            tier.stats.interp_blocks += 1;
            let before = (*cpu).retired();
            let ev = (*cpu).run_block(&mut *mem, *left);
            let ran = (*cpu).retired() - before;
            tier.stats.insns_interp += ran;
            #[cfg(feature = "step-stats")]
            interp_page(&*mem, pc, ran);
            *left -= ran;
            if let Some(ev) = ev {
                break Stop::Event(ev);
            }
        }
    };
    let cur = current();
    cur.cpu = core::ptr::null_mut();
    cur.mem = core::ptr::null_mut();
    stop
}

/// `jit_step`: executes the instruction at `pc` in the interpreter, for translated code.
pub fn step(pc: u64) -> i32 {
    let cur = current();
    cur.steps += 1;
    let (cpu, mem) = (cur.cpu, cur.mem);
    if cpu.is_null() || mem.is_null() {
        return STATUS_FAULT;
    }
    // SAFETY: set by `dispatch` for the duration of the translated call that is calling us;
    // the dispatcher holds no reference to either meanwhile.
    let (cpu, mem) = unsafe { (&mut *cpu, &mut *mem) };
    cpu.set_pc(pc);
    #[cfg(feature = "step-stats")]
    classify(mem, pc);
    let retired = cpu.retired();
    mem.filling.set(true);
    let (event, code_changed) = cpu.step_one(mem);
    mem.filling.set(false);
    // The dispatcher counts retired instructions of translated code from the budget.
    cpu.set_retired(retired);
    match event {
        Event::Yield => {
            if code_changed {
                STATUS_CODE
            } else {
                0
            }
        }
        Event::Fault(_) => {
            current().event = Some(event);
            STATUS_FAULT
        }
        Event::Syscall(_) => {
            current().event = Some(event);
            STATUS_EVENT
        }
    }
}

instance_state! {
    fn step_classes_cell() -> *mut [u64; 6] = [0; 6];
}

/// Stepped instructions by class (counted with feature `step-stats`, zeros without): memory ops
/// translated code handles inline (so: TLB misses, page crossings, unaligned SP), SIMD/FP, SIMD
/// memory, exclusives and atomics, system, other.
pub fn step_classes() -> [u64; 6] {
    // SAFETY: one thread reaches the cell (`state`), and no reference into it is live.
    unsafe { *step_classes_cell() }
}

#[cfg(feature = "step-stats")]
type Shapes = alloc::collections::BTreeMap<alloc::string::String, u64>;

#[cfg(feature = "step-stats")]
instance_state! {
    /// Stepped instructions by shape: the instruction's debug form without its register and
    /// immediate fields.
    fn step_shapes_cell() -> *mut Option<Shapes> = None;
}

#[cfg(feature = "step-stats")]
fn classify(mem: &mut EngineMem, pc: u64) {
    use wasmshim_aarch64::{Inst, simd::SimdInst};
    use wasmshim_cpu::GuestMemory;
    let Ok(word) = mem.mem.fetch(pc) else { return };
    let inst = wasmshim_aarch64::decode(word);
    let class = match inst {
        Inst::LoadStore { .. } | Inst::LoadStorePair { .. } | Inst::LoadLiteral { .. } => 0,
        Inst::Simd(SimdInst::LdStMulti { .. } | SimdInst::LdStSingle { .. }) => 2,
        Inst::Simd(_) => 1,
        Inst::Exclusive { .. } | Inst::Cas { .. } | Inst::Casp { .. } | Inst::Atomic { .. } => 3,
        Inst::Mrs { .. } | Inst::Msr { .. } | Inst::DcZva { .. } | Inst::Clrex | Inst::Nop => 4,
        _ => 5,
    };
    let text = alloc::format!("{inst:?}");
    let shape: alloc::string::String = text
        .split(", ")
        .filter(|f| {
            let f = f.trim_start_matches(|c: char| {
                c != ' ' && c != '{' && c != '(' && !c.is_lowercase()
            });
            ![
                "rd:", "rn:", "rm:", "rt:", "ra:", "rs:", "rt2:", "imm", "index", "bits", "offset",
                "amount", "lsb", "immr", "imms", "wmask", "tmask", "nzcv", "bit:",
            ]
            .iter()
            .any(|p| f.trim_start().starts_with(p) || f.contains(&alloc::format!("{{ {p}")))
        })
        .collect::<alloc::vec::Vec<_>>()
        .join(", ");
    // SAFETY: one thread reaches the cells (`state`); nothing here re-enters the engine.
    unsafe {
        (*step_classes_cell())[class] += 1;
        *(*step_shapes_cell())
            .get_or_insert_with(Default::default)
            .entry(shape)
            .or_default() += 1;
    }
}

#[cfg(feature = "step-stats")]
type Pages = alloc::collections::BTreeMap<u64, (u64, u64, u64, u64, u64)>;

#[cfg(feature = "step-stats")]
instance_state! {
    /// Interpreted instructions by code page: (instructions, blocks, the code version last seen,
    /// versions seen, translated entries).
    fn interp_pages_cell() -> *mut Option<Pages> = None;
}

#[cfg(feature = "step-stats")]
instance_state! {
    /// Interpreted blocks by start PC: (instructions, executions).
    fn interp_blocks_cell() -> *mut Option<alloc::collections::BTreeMap<u64, (u64, u64)>> = None;
}

#[cfg(feature = "step-stats")]
instance_state! {
    /// The content key of each interpreted page's latest version.
    fn interp_keys_cell() -> *mut Option<alloc::collections::BTreeMap<u64, [u8; 16]>> = None;
}

#[cfg(feature = "step-stats")]
fn interp_page(mem: &EngineMem, pc: u64, ran: u64) {
    use wasmshim_cpu::GuestMemory;
    let version = mem.mem.code_version(pc);
    // SAFETY (here and below): one thread reaches the cells (`state`); nothing in this function
    // re-enters the engine.
    let pages = unsafe { (*interp_pages_cell()).get_or_insert_with(Default::default) };
    let e = pages.entry(pc >> 12).or_insert((0, 0, u64::MAX, 0, 0));
    e.4 = mem.translated_entries(pc) as u64;
    e.0 += ran;
    e.1 += 1;
    if e.2 != version {
        e.2 = version;
        e.3 += 1;
        // The page's content key (as profiles and the AOT directory name it), per version.
        if let Some(bytes) = mem.mem.page_bytes(pc) {
            let key = wasmshim_aarch64_translate::aot::page_key(bytes);
            unsafe { (*interp_keys_cell()).get_or_insert_with(Default::default) }
                .insert(pc >> 12, key);
        }
    }
    let blocks = unsafe { (*interp_blocks_cell()).get_or_insert_with(Default::default) };
    let b = blocks.entry(pc).or_insert((0, 0));
    b.0 += ran;
    b.1 += 1;
}

/// Feature `step-stats`: the blocks most instructions were interpreted in: (PC, instructions,
/// executions), and the content keys of interpreted pages.
#[cfg(feature = "step-stats")]
#[allow(clippy::type_complexity)]
pub fn interp_blocks() -> (
    alloc::vec::Vec<(u64, u64, u64)>,
    alloc::collections::BTreeMap<u64, [u8; 16]>,
) {
    // SAFETY: one thread reaches the cells (`state`), and no reference into them is live.
    let (blocks, keys) = unsafe {
        (
            (*interp_blocks_cell()).clone().unwrap_or_default(),
            (*interp_keys_cell()).clone().unwrap_or_default(),
        )
    };
    let mut v: alloc::vec::Vec<_> = blocks.into_iter().map(|(pc, (i, n))| (pc, i, n)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    (v, keys)
}

/// Feature `step-stats`: the pages most instructions were interpreted in, most first:
/// (page number, instructions, blocks, versions).
#[cfg(feature = "step-stats")]
pub fn interp_pages() -> alloc::vec::Vec<(u64, u64, u64, u64, u64)> {
    // SAFETY: one thread reaches the cell (`state`), and no reference into it is live.
    let pages = unsafe { (*interp_pages_cell()).clone() }.unwrap_or_default();
    let mut v: alloc::vec::Vec<_> = pages
        .into_iter()
        .map(|(vpn, (i, b, _, n, t))| (vpn, i, b, n, t))
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

/// Feature `step-stats`: the steps by class ([`step_classes`]) and by shape, most frequent
/// first.
#[cfg(feature = "step-stats")]
pub fn step_stats() -> ([u64; 6], alloc::vec::Vec<(alloc::string::String, u64)>) {
    let classes = step_classes();
    // SAFETY: one thread reaches the cell (`state`), and no reference into it is live.
    let shapes = unsafe { (*step_shapes_cell()).clone() };
    let mut v: alloc::vec::Vec<_> = shapes.unwrap_or_default().into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    (classes, v)
}

instance_state! {
    fn fills_cell() -> *mut u64 = 0;
}

/// Calls of `jit_fill` (translated code's TLB misses; never guest-visible).
pub fn fills() -> u64 {
    // SAFETY: one thread reaches the cell (`state`), and no reference into it is live.
    unsafe { *fills_cell() }
}

/// `jit_fill`: makes `addr`'s page accessible through the TLB and returns the host address of
/// `addr` (see `wasmshim_aarch64_translate::abi::IMPORT_FILL`), or 0.
pub fn fill(addr: u64, write: bool, size: u64) -> u32 {
    // SAFETY: one thread reaches the cell (`state`), and no reference into it is live.
    unsafe { *fills_cell() = (*fills_cell()).wrapping_add(1) };
    let mem = current().mem;
    if mem.is_null() || (addr & 0xfff) + size > 0x1000 {
        return 0;
    }
    // SAFETY: as in `step`.
    let mem = unsafe { &mut *mem };
    mem.fill_for_jit(addr, write)
        .map_or(0, |page| (page + (addr & 0xfff) as usize) as u32)
}

/// The fill symbol translated code calls.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn linux_jit_fill(addr: i64, write: i32, size: i32) -> i32 {
    fill(addr as u64, write != 0, size as u64) as i32
}

/// The symbol translated code calls (AOT objects and run-time modules alike).
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn linux_jit_step(pc: i64) -> i32 {
    step(pc as u64)
}

/// The AAPCS64 syscall frame of the current registers.
fn frame(cpu: &Cpu) -> SyscallFrame {
    SyscallFrame {
        number: cpu.x(8),
        pc: cpu.pc(),
        args: [cpu.x(0), cpu.x(1), cpu.x(2), cpu.x(3), cpu.x(4), cpu.x(5)],
    }
}
