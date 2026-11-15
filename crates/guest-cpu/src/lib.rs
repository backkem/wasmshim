//! The execution-frontend contract shared by every CPU backend and every consumer.
//!
//! * [`GuestCpu`]: one guest hardware thread. A backend (interpreter, DBT, ...) runs guest code
//!   until something the outside world must handle happens, and reports it as an [`Event`].
//! * [`GuestMemory`]: a guest virtual address space with per-page R/W/X permissions. Accesses
//!   that violate them are [`MemFault`]s, never host panics.
//! * [`PagedMemory`]: the simple sparse implementation used by the interpreter.
//!
//! This crate knows nothing about Linux (or any OS): a "syscall" here is only "the guest
//! executed its ISA's supervisor-call instruction", described in the backend's own calling
//! convention by a [`SyscallFrame`].

#![no_std]
#![deny(unsafe_code)]

extern crate alloc;

use alloc::boxed::Box;

mod memory;
mod paged;
mod shared;

pub use memory::{
    Access, GuestMemory, MapError, MemFault, MemFaultKind, PAGE_SIZE, PageRef, PageSource,
    PageSourceError, PageView, Perms, SharedMemory,
};
pub use paged::PagedMemory;
pub use shared::SharedPages;

/// A supervisor call, decoded with the backend's calling convention.
///
/// For AArch64 this is the AAPCS64 `SVC #0` convention (number in `x8`, arguments in `x0`–`x5`,
/// result in `x0`). The personality decides what the number means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyscallFrame {
    /// The syscall number register.
    pub number: u64,
    /// Address of the instruction after the supervisor call (where execution resumes).
    pub pc: u64,
    /// The six argument registers, in order.
    pub args: [u64; 6],
}

/// Why guest execution stopped abnormally. The program counter is the faulting instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// A load, store or instruction fetch hit an unmapped or protected page.
    Memory { pc: u64, fault: MemFault },
    /// The instruction is undefined, or not implemented by this backend.
    Undefined { pc: u64, insn: u32 },
    /// A breakpoint instruction (A64 `BRK #imm`).
    Breakpoint { pc: u64, imm: u16 },
    /// A misaligned access where the ISA requires alignment (exclusives, atomics, SP checks).
    Alignment { pc: u64, addr: u64 },
}

impl Fault {
    /// The program counter of the faulting instruction.
    pub fn pc(&self) -> u64 {
        match *self {
            Self::Memory { pc, .. }
            | Self::Undefined { pc, .. }
            | Self::Breakpoint { pc, .. }
            | Self::Alignment { pc, .. } => pc,
        }
    }
}

impl core::fmt::Display for Fault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::Memory { pc, fault } => write!(f, "{fault} at pc {pc:#x}"),
            Self::Undefined { pc, insn } => {
                write!(f, "undefined instruction {insn:#010x} at pc {pc:#x}")
            }
            Self::Breakpoint { pc, imm } => write!(f, "breakpoint #{imm:#x} at pc {pc:#x}"),
            Self::Alignment { pc, addr } => {
                write!(f, "misaligned access to {addr:#x} at pc {pc:#x}")
            }
        }
    }
}

impl core::error::Error for Fault {}

/// What [`GuestCpu::run_until_event`] stopped for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// The guest made a supervisor call. The program counter already points past the
    /// instruction. The handler sets the result with [`GuestCpu::set_syscall_result`].
    Syscall(SyscallFrame),
    /// The guest faulted. The program counter still points at the faulting instruction.
    Fault(Fault),
    /// The instruction budget is used up. Calling `run_until_event` again resumes seamlessly.
    Yield,
}

/// A full register snapshot, for signal frames, `clone` and debuggers (rare, bulk paths only;
/// syscalls never need it, and personalities read syscall arguments only through the
/// [`SyscallFrame`]). One variant per instruction set: code that needs registers
/// matches on the ISA, so a second ISA is a new variant, not a reinterpretation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuContext {
    /// AArch64.
    A64(A64Context),
}

impl CpuContext {
    /// The program counter.
    pub fn pc(&self) -> u64 {
        match self {
            Self::A64(c) => c.pc,
        }
    }

    /// The stack pointer.
    pub fn sp(&self) -> u64 {
        match self {
            Self::A64(c) => c.sp,
        }
    }

    /// The user thread pointer.
    pub fn thread_pointer(&self) -> u64 {
        match self {
            Self::A64(c) => c.tp,
        }
    }
}

/// The AArch64 register state: `x` is X0–X30, `flags` holds NZCV in bits 31:28, `v` is
/// V0–V31, `fpsr`/`fpcr` are the FP status and control registers and `tp` is `TPIDR_EL0`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct A64Context {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub flags: u64,
    pub v: [u128; 32],
    pub fpsr: u64,
    pub fpcr: u64,
    pub tp: u64,
}

/// One guest hardware thread (register state) driven by some execution backend.
///
/// Backends are dumb: they know about their ISA and [`GuestMemory`], nothing else. Every
/// implementation must be deterministic: the same state, memory and budget always produce
/// the same event, the same state afterwards and the same [`GuestCpu::retired`] count.
pub trait GuestCpu {
    /// Runs guest code until an [`Event`] happens or `budget` instructions have retired.
    ///
    /// A supervisor call counts as retired; a faulting instruction does not. Any exclusive
    /// monitor (load-linked/store-conditional reservation) is cleared on return, so a
    /// conditional store that straddles two calls fails and the guest retries.
    fn run_until_event(&mut self, mem: &mut dyn GuestMemory, budget: u64) -> Event;

    /// Writes a syscall's result into the backend's result register.
    fn set_syscall_result(&mut self, value: u64);

    /// The program counter.
    fn pc(&self) -> u64;
    /// Sets the program counter (e.g. to an entry point or a signal handler).
    fn set_pc(&mut self, pc: u64);
    /// The stack pointer.
    fn sp(&self) -> u64;
    /// Sets the stack pointer.
    fn set_sp(&mut self, sp: u64);
    /// The user thread pointer (A64 `TPIDR_EL0`).
    fn thread_pointer(&self) -> u64;
    /// Sets the user thread pointer.
    fn set_thread_pointer(&mut self, tp: u64);

    /// Instructions retired since this CPU was created.
    fn retired(&self) -> u64;

    /// Sets the virtual time in nanoseconds that the ISA's user-readable counter reports
    /// (A64 `CNTVCT_EL0` at a 1 GHz `CNTFRQ_EL0`). The backend never reads a real clock: the
    /// embedder sets this before each `run_until_event`, so the counter is frozen within a run.
    fn set_virtual_time(&mut self, ns: u64);

    /// The full register state.
    fn context(&self) -> CpuContext;

    /// Replaces the full register state (flags and FP control are masked to what the ISA
    /// lets user code set).
    fn set_context(&mut self, ctx: &CpuContext);

    /// A copy of this CPU's full register state (for `fork`/`clone`).
    fn duplicate(&self) -> Box<dyn GuestCpu>;

    /// Debugger support (instruction-level stops), if this backend has it. The
    /// default has none: a personality then refuses PC breakpoints and instruction stops.
    fn debug(&mut self) -> Option<&mut dyn DebugCpu> {
        None
    }
}

/// Instruction-level stops for a debugger. A stop made through this interface is
/// invisible to the guest: the thread resumes with the same registers, the same retired count
/// and the same exclusive monitor, so its run is the run it would have had without the stop.
pub trait DebugCpu {
    /// Runs like [`GuestCpu::run_until_event`], with two differences:
    ///
    /// * it returns [`Event::Yield`] *before* executing an instruction whose address is in
    ///   `breaks` (sorted, deduplicated), except the first instruction this call executes (so
    ///   resuming from a breakpoint steps over it);
    /// * it **keeps** the exclusive monitor on return. The caller ends a debugger stop by
    ///   resuming this CPU with nothing else in between (no other thread ran, no guest memory
    ///   was written), or calls [`DebugCpu::clear_monitor`] first.
    ///
    /// With `breaks` empty, the instructions retired, their order and every event are those of
    /// `run_until_event` with the same budget: a budget is exact in every tier.
    fn run_debug(&mut self, mem: &mut dyn GuestMemory, budget: u64, breaks: &[u64]) -> Event;

    /// Clears the exclusive monitor, as `run_until_event` does on return.
    fn clear_monitor(&mut self);
}
