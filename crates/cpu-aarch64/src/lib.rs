//! An AArch64 (A64) CPU backend: decoder plus interpreter behind [`wasmshim_cpu::GuestCpu`].
//!
//! * [`decode`] turns a 32-bit word into an [`Inst`]; it is pure and table-driven per
//!   encoding group (see `decode.rs`).
//! * [`Cpu`] holds the user-visible register state (X0–X30, SP, PC, NZCV, V0–V31, FPCR/FPSR,
//!   TPIDR_EL0) and interprets instructions against a [`GuestMemory`].
//!
//! The model is EL0 only: no exception levels, no MMU (the [`GuestMemory`] is the address
//! space), a single-copy-atomic view of memory (one host thread), and an exclusive monitor
//! that is cleared whenever `run_until_event` returns. It knows nothing about any OS: `SVC`
//! just becomes an [`Event::Syscall`] in the AAPCS64 convention.

#![no_std]
#![deny(unsafe_code)]

extern crate alloc;

use alloc::boxed::Box;

use code_cache::CodeCache;
use wasmshim_cpu::{
    A64Context, CpuContext, DebugCpu, Event, Fault, GuestCpu, GuestMemory, SyscallFrame,
};

pub mod asm;
mod code_cache;
mod decode;
mod exec;
mod fp;
mod inst;
pub mod simd;
mod simd_exec;

pub use decode::decode;
pub use exec::{CTR_EL0, DC_ZVA_BYTES, DCZID_EL0, MIDR_EL1};
pub use inst::*;

/// `CNTFRQ_EL0`: the virtual counter ticks in nanoseconds.
pub const COUNTER_FREQUENCY: u64 = 1_000_000_000;

/// The exclusive monitor value meaning "no reservation" (no aligned exclusive access can
/// start at this address).
pub const NO_MONITOR: u64 = u64::MAX;

/// One A64 hardware thread. `#[repr(C)]`: see [`layout`].
#[derive(Clone)]
#[repr(C)]
pub struct Cpu {
    /// X0–X30, then index 31 (always zero: writes are discarded), then index 32 = SP.
    x: [u64; 33],
    pc: u64,
    n: bool,
    z: bool,
    c: bool,
    v: bool,
    /// Lazily evaluated flags (see [`lazy_flags`]): when not [`lazy_flags::NONE`], NZCV is a
    /// function of `flags_kind` and the three operands below, and `n`..`v` are stale. Only
    /// translated code sets it; the interpreter normalises it before it runs.
    flags_kind: u32,
    flags_x: u64,
    flags_y: u64,
    flags_r: u64,
    /// V0–V31 (little-endian lanes: lane 0 is the low bits).
    vreg: [u128; 32],
    fpcr: u64,
    fpsr: u64,
    tpidr: u64,
    tpidrro: u64,
    counter: u64,
    /// Address reserved by the last load-exclusive, or [`NO_MONITOR`].
    monitor: u64,
    retired: u64,
    cache: CodeCache,
    /// Per-PC execution counts, shared by every copy of this CPU (feature `profile`).
    #[cfg(feature = "profile")]
    pub profile: Option<alloc::rc::Rc<core::cell::RefCell<alloc::collections::BTreeMap<u64, u64>>>>,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu {
    /// A CPU with every register zero.
    pub fn new() -> Self {
        Self {
            x: [0; 33],
            pc: 0,
            n: false,
            z: false,
            c: false,
            v: false,
            flags_kind: lazy_flags::NONE,
            flags_x: 0,
            flags_y: 0,
            flags_r: 0,
            vreg: [0; 32],
            fpcr: 0,
            fpsr: 0,
            tpidr: 0,
            tpidrro: 0,
            counter: 0,
            monitor: NO_MONITOR,
            retired: 0,
            cache: CodeCache::new(),
            #[cfg(feature = "profile")]
            profile: None,
        }
    }

    /// General register `n` (0–30); 31 reads as zero.
    pub fn x(&self, n: u8) -> u64 {
        self.x[usize::from(n.min(31))]
    }

    /// Sets general register `n` (0–30); 31 is ignored.
    pub fn set_x(&mut self, n: u8, value: u64) {
        if n < 31 {
            self.x[usize::from(n)] = value;
        }
    }

    /// SIMD&FP register `n` as a 128-bit value.
    pub fn vreg(&self, n: u8) -> u128 {
        self.vreg[usize::from(n & 31)]
    }

    /// Sets SIMD&FP register `n`.
    pub fn set_vreg(&mut self, n: u8, value: u128) {
        self.vreg[usize::from(n & 31)] = value;
    }

    /// The flags as `NZCV` in bits 31:28.
    pub fn nzcv(&self) -> u64 {
        let (n, z, c, v) = self.flags();
        (u64::from(n) << 31) | (u64::from(z) << 30) | (u64::from(c) << 29) | (u64::from(v) << 28)
    }

    /// N, Z, C, V, evaluating lazy flags if needed.
    fn flags(&self) -> (bool, bool, bool, bool) {
        if self.flags_kind == lazy_flags::NONE {
            (self.n, self.z, self.c, self.v)
        } else {
            lazy_flags::nzcv(self.flags_kind, self.flags_x, self.flags_y, self.flags_r)
        }
    }

    /// Evaluates lazy flags into `n`..`v` (the interpreter reads only those).
    #[inline]
    pub(crate) fn normalize_flags(&mut self) {
        if self.flags_kind != lazy_flags::NONE {
            (self.n, self.z, self.c, self.v) = self.flags();
            self.flags_kind = lazy_flags::NONE;
        }
    }

    /// Sets the flags from bits 31:28.
    pub fn set_nzcv(&mut self, value: u64) {
        self.flags_kind = lazy_flags::NONE;
        self.n = value & (1 << 31) != 0;
        self.z = value & (1 << 30) != 0;
        self.c = value & (1 << 29) != 0;
        self.v = value & (1 << 28) != 0;
    }

    /// `FPCR`.
    pub fn fpcr(&self) -> u64 {
        self.fpcr
    }

    /// `FPSR`.
    pub fn fpsr(&self) -> u64 {
        self.fpsr
    }
    /// Runs like [`GuestCpu::run_until_event`] but keeps the exclusive monitor: for engines
    /// that split one `run_until_event` into many steps (the DBT's interpreter tier), which
    /// must then call [`Cpu::clear_monitor`] once at the end.
    pub fn run_raw(&mut self, mem: &mut dyn GuestMemory, budget: u64) -> Event {
        // (The interpreter's hot loop: kept a plain, single-instantiation loop. A generic
        // version shared with `run_block` cost a third of its speed in inlining.)
        let mut left = budget;
        self.normalize_flags();
        self.cache.sync(mem);
        loop {
            if left == 0 {
                break Event::Yield;
            }
            let pc = self.pc;
            if pc & 3 != 0 {
                break Event::Fault(Fault::Alignment { pc, addr: pc });
            }
            let inst = match self.cache.get(mem, pc) {
                Ok(inst) => inst,
                Err(fault) => break Event::Fault(fault),
            };
            #[cfg(feature = "profile")]
            if let Some(p) = &self.profile {
                *p.borrow_mut().entry(pc).or_default() += 1;
            }
            match self.execute(mem, inst, pc) {
                Ok(None) => {
                    if inst.may_write_memory() {
                        // A store into code is visible to the very next instruction.
                        self.cache.sync(mem);
                    }
                }
                Ok(Some(event)) => {
                    self.retired += 1;
                    break event;
                }
                Err(fault) => {
                    self.pc = pc;
                    break Event::Fault(fault);
                }
            }
            self.retired += 1;
            left -= 1;
        }
    }

    /// Executes the one instruction at `pc` for translated code's slow path (`jit_step`):
    /// `run_raw(mem, 1)`, plus whether the memory's code epoch moved while it ran (a store into
    /// code: the caller's translation of the rest may be stale). Reads the epoch twice, where a
    /// caller wrapping `run_raw` read it four times.
    pub fn step_one(&mut self, mem: &mut dyn GuestMemory) -> (Event, bool) {
        self.normalize_flags();
        self.cache.sync(mem);
        let before = self.cache.epoch();
        let pc = self.pc;
        if pc & 3 != 0 {
            return (Event::Fault(Fault::Alignment { pc, addr: pc }), false);
        }
        let inst = match self.cache.get(mem, pc) {
            Ok(inst) => inst,
            Err(fault) => return (Event::Fault(fault), false),
        };
        let event = match self.execute(mem, inst, pc) {
            Ok(None) => {
                self.retired += 1;
                Event::Yield
            }
            Ok(Some(event)) => {
                self.retired += 1;
                event
            }
            Err(fault) => {
                self.pc = pc;
                Event::Fault(fault)
            }
        };
        self.cache.sync(mem);
        (event, self.cache.epoch() != before)
    }

    /// Runs at most `budget` instructions, stopping after the first control-flow
    /// instruction (taken or not): one basic block. `None` means the block ended (or the
    /// budget ran out exactly there) with no event. Keeps the exclusive monitor.
    pub fn run_block(&mut self, mem: &mut dyn GuestMemory, budget: u64) -> Option<Event> {
        let mut left = budget;
        self.normalize_flags();
        self.cache.sync(mem);
        while left > 0 {
            let pc = self.pc;
            if pc & 3 != 0 {
                return Some(Event::Fault(Fault::Alignment { pc, addr: pc }));
            }
            let inst = match self.cache.get(mem, pc) {
                Ok(inst) => inst,
                Err(fault) => return Some(Event::Fault(fault)),
            };
            match self.execute(mem, inst, pc) {
                Ok(None) => {
                    if inst.may_write_memory() {
                        self.cache.sync(mem);
                    }
                }
                Ok(Some(event)) => {
                    self.retired += 1;
                    return Some(event);
                }
                Err(fault) => {
                    self.pc = pc;
                    return Some(Event::Fault(fault));
                }
            }
            self.retired += 1;
            left -= 1;
            if inst.is_control_flow() {
                break;
            }
        }
        None
    }

    /// Clears the exclusive monitor (see [`Cpu::run_raw`]).
    pub fn clear_monitor(&mut self) {
        self.monitor = NO_MONITOR;
    }

    /// [`DebugCpu::run_debug`] in the interpreter: [`Cpu::run_raw`] (which keeps the monitor),
    /// one instruction at a time while `breaks` is not empty. Stepping is slower than the hot
    /// loop and never different: `run_raw` is exact per instruction.
    pub fn run_debug_raw(
        &mut self,
        mem: &mut dyn GuestMemory,
        budget: u64,
        breaks: &[u64],
    ) -> Event {
        if breaks.is_empty() {
            return self.run_raw(mem, budget);
        }
        let mut left = budget;
        let mut first = true;
        loop {
            if left == 0 {
                break Event::Yield;
            }
            if !first && breaks.binary_search(&self.pc).is_ok() {
                break Event::Yield;
            }
            first = false;
            let before = self.retired;
            let event = self.run_raw(mem, 1);
            left -= (self.retired - before).min(left);
            if !matches!(event, Event::Yield) {
                break event;
            }
        }
    }

    /// Sets the retired-instruction counter (engines that count retired instructions
    /// themselves, e.g. for translated code, keep it exact).
    pub fn set_retired(&mut self, retired: u64) {
        self.retired = retired;
    }
}
impl DebugCpu for Cpu {
    fn run_debug(&mut self, mem: &mut dyn GuestMemory, budget: u64, breaks: &[u64]) -> Event {
        self.run_debug_raw(mem, budget, breaks)
    }

    fn clear_monitor(&mut self) {
        self.monitor = NO_MONITOR;
    }
}

impl GuestCpu for Cpu {
    fn run_until_event(&mut self, mem: &mut dyn GuestMemory, budget: u64) -> Event {
        let event = self.run_raw(mem, budget);
        self.monitor = NO_MONITOR;
        event
    }

    fn debug(&mut self) -> Option<&mut dyn DebugCpu> {
        Some(self)
    }

    fn set_syscall_result(&mut self, value: u64) {
        self.x[0] = value;
    }

    fn pc(&self) -> u64 {
        self.pc
    }

    fn set_pc(&mut self, pc: u64) {
        self.pc = pc;
    }

    fn sp(&self) -> u64 {
        self.x[usize::from(SP)]
    }

    fn set_sp(&mut self, sp: u64) {
        self.x[usize::from(SP)] = sp;
    }

    fn thread_pointer(&self) -> u64 {
        self.tpidr
    }

    fn set_thread_pointer(&mut self, tp: u64) {
        self.tpidr = tp;
    }

    fn retired(&self) -> u64 {
        self.retired
    }

    fn set_virtual_time(&mut self, ns: u64) {
        self.counter = ns;
    }

    fn context(&self) -> CpuContext {
        let mut x = [0u64; 31];
        x.copy_from_slice(&self.x[..31]);
        CpuContext::A64(A64Context {
            x,
            sp: self.x[usize::from(SP)],
            pc: self.pc,
            flags: self.nzcv(),
            v: self.vreg,
            fpsr: self.fpsr,
            fpcr: self.fpcr,
            tp: self.tpidr,
        })
    }

    fn set_context(&mut self, ctx: &CpuContext) {
        let CpuContext::A64(ctx) = ctx;
        self.x[..31].copy_from_slice(&ctx.x);
        self.x[usize::from(ZR)] = 0;
        self.x[usize::from(SP)] = ctx.sp;
        self.pc = ctx.pc;
        self.set_nzcv(ctx.flags);
        self.vreg = ctx.v;
        self.fpsr = ctx.fpsr & 0xf800_009f;
        self.fpcr = ctx.fpcr & 0x07ff_9f00;
        self.tpidr = ctx.tp;
        self.monitor = NO_MONITOR;
    }

    fn duplicate(&self) -> Box<dyn GuestCpu> {
        Box::new(self.clone())
    }
}

/// The syscall frame for the current register state (AAPCS64: number in X8, args in X0–X5).
pub(crate) fn syscall_frame(x: &[u64; 33], pc: u64) -> SyscallFrame {
    SyscallFrame {
        number: x[8],
        pc,
        args: [x[0], x[1], x[2], x[3], x[4], x[5]],
    }
}

/// Byte offsets of the register file inside [`Cpu`] (which is `#[repr(C)]`), for engines whose
/// generated code reads and writes the registers in place (the DBT). Only these fields have a
/// stable layout.
pub mod layout {
    use core::mem::offset_of;

    use super::Cpu;

    /// `x: [u64; 33]`: X0–X30, then index 31 (always zero), then index 32 = SP.
    pub const X: usize = offset_of!(Cpu, x);
    /// `pc: u64`.
    pub const PC: usize = offset_of!(Cpu, pc);
    /// N, Z, C and V as four consecutive bytes, each 0 or 1 (little-endian: one `u32` holds
    /// `n | z << 8 | c << 16 | v << 24`).
    pub const NZCV: usize = offset_of!(Cpu, n);
    /// `flags_kind: u32` ([`super::lazy_flags`]), then the operands `flags_x`, `flags_y`,
    /// `flags_r: u64`.
    pub const FLAGS_KIND: usize = offset_of!(Cpu, flags_kind);
    pub const FLAGS_X: usize = offset_of!(Cpu, flags_x);
    pub const FLAGS_Y: usize = offset_of!(Cpu, flags_y);
    pub const FLAGS_R: usize = offset_of!(Cpu, flags_r);
    /// `counter: u64`: the virtual time `CNTVCT_EL0` reads.
    pub const COUNTER: usize = offset_of!(Cpu, counter);
    /// `TPIDR_EL0: u64`.
    pub const TPIDR: usize = offset_of!(Cpu, tpidr);
    /// `vreg: [u128; 32]`.
    pub const V: usize = offset_of!(Cpu, vreg);
    /// `monitor: u64`: the exclusive monitor ([`super::NO_MONITOR`]: none).
    pub const MONITOR: usize = offset_of!(Cpu, monitor);
    /// `fpsr: u64`: of its bits only QC (bit 27, cumulative saturation) is ever set.
    pub const FPSR: usize = offset_of!(Cpu, fpsr);

    const _: () = {
        assert!(offset_of!(Cpu, z) == NZCV + 1);
        assert!(offset_of!(Cpu, c) == NZCV + 2);
        assert!(offset_of!(Cpu, v) == NZCV + 3);
    };
}

/// Lazily evaluated flags: what translated code leaves in the CPU instead of NZCV (the
/// operands and result of the last flag-setting instruction, and its kind).
pub mod lazy_flags {
    /// The flags are `n`, `z`, `c`, `v` themselves.
    pub const NONE: u32 = 0;
    /// `x - y = r` (64-bit).
    pub const SUB64: u32 = 1;
    /// `x + y = r` (64-bit).
    pub const ADD64: u32 = 2;
    /// `x - y = r` on the low 32 bits (operands zero-extended).
    pub const SUB32: u32 = 3;
    /// `x + y = r` on the low 32 bits.
    pub const ADD32: u32 = 4;
    /// A logical result `r` (64-bit): C = V = 0.
    pub const LOGIC64: u32 = 5;
    /// A logical result `r` on the low 32 bits.
    pub const LOGIC32: u32 = 6;

    /// NZCV of a lazy description (`NONE` gives all false).
    pub fn nzcv(kind: u32, x: u64, y: u64, r: u64) -> (bool, bool, bool, bool) {
        match kind {
            SUB64 => (
                (r as i64) < 0,
                r == 0,
                x >= y,
                (((x ^ y) & (x ^ r)) as i64) < 0,
            ),
            ADD64 => (
                (r as i64) < 0,
                r == 0,
                r < x,
                ((!(x ^ y) & (x ^ r)) as i64) < 0,
            ),
            SUB32 | ADD32 => {
                let (x, y, r) = (x as u32, y as u32, r as u32);
                let c = if kind == SUB32 { x >= y } else { r < x };
                let v = if kind == SUB32 {
                    (((x ^ y) & (x ^ r)) as i32) < 0
                } else {
                    ((!(x ^ y) & (x ^ r)) as i32) < 0
                };
                ((r as i32) < 0, r == 0, c, v)
            }
            LOGIC64 => ((r as i64) < 0, r == 0, false, false),
            LOGIC32 => ((r as i32) < 0, r as u32 == 0, false, false),
            _ => (false, false, false, false),
        }
    }
}
