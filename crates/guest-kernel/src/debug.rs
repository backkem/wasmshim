//! The substrate's debug hooks, typed: what an embedder's debugger sees at a stop and how it
//! answers.
//!
//! **Embedder surface, unstable**: no debugger lives in this repository; the hooks follow what
//! embedders need and may change between versions.
//!
//! Probes ([`ProbeSpec`]) are armed on the personality's [`crate::observe::Observer`]. When one
//! fires, the run loop calls [`crate::Host::stop`] with a [`Stop`] and a [`StopTarget`] (the
//! stopped thread and the personality's state, as `dyn Any`, for the embedder's own typed reads
//! and writes). The answer ([`Resume`]) says whether the debugger changed the machine (the
//! syscall frame is then read again from the registers) and whether to stop again after a step.
//! Nothing here runs unless a probe or a step is armed, and a stop changes nothing a guest sees:
//! the same instructions retire and the same preemption points are taken with or without it.

use alloc::string::String;
use alloc::vec::Vec;
use core::any::Any;

use wasmshim_cpu::Access;

pub use crate::observe::{Phase, Point};
use crate::task::Tid;

/// One probe to arm: an id (`0` is reserved for steps), the point it fires at, how many more
/// hits it takes (`None`: unlimited; it disarms after its last), and what a hit does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeSpec {
    pub id: u32,
    pub point: Point,
    pub hits: Option<u32>,
    /// A hit stops the machine for the debugger (`true`), or only reports the hit, as a stop
    /// the debugger answers at once (`false`: an action such as a write).
    pub hold: bool,
    /// The stop carries the substrate's own [`StopInfo`] (otherwise the debugger reads what it
    /// wants itself).
    pub info: bool,
}

/// Where a stop is, on the instance's clocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Position {
    /// The `n`th syscall entry (instance-wide, from 1), at its entry or after its result.
    Sys { n: u64, exit: bool },
    /// The instance has retired exactly this many instructions.
    Insn(u64),
}

/// Why an instruction stop happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// An instruction-count point.
    Insn,
    /// A PC breakpoint.
    Pc,
    /// The end of a step.
    Step,
}

impl Why {
    /// The point's name (`insn`, `pc`, `step`).
    pub fn name(self) -> &'static str {
        match self {
            Why::Insn => "insn",
            Why::Pc => "pc",
            Why::Step => "step",
        }
    }
}

/// What changed a watched range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WatchBy {
    /// An instruction of the thread.
    Cpu,
    /// The kernel, in this syscall (`syscall <name>`, or `syscall <number>`).
    Syscall(String),
}

/// The substrate's facts about a stop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopInfo {
    Syscall {
        /// The syscall's ordinal (the `sys` clock).
        sys: u64,
        number: u64,
        tid: Tid,
        args: [u64; 6],
        /// The personality's name for the number.
        name: Option<&'static str>,
        /// The raw result, at exit.
        ret: Option<u64>,
    },
    Insn {
        /// The instance's retired count.
        insn: u64,
        tid: Tid,
        pc: u64,
        /// The thread's retired count.
        retired: u64,
        why: Why,
        pid: Option<u32>,
        symbol: Option<String>,
    },
    Watch {
        insn: u64,
        tid: Tid,
        pc: u64,
        retired: u64,
        start: u64,
        len: u64,
        old: Vec<u8>,
        new: Vec<u8>,
        by: WatchBy,
        /// The faulting access, for a change by the CPU.
        fault: Option<(u64, Access)>,
        pid: Option<u32>,
        symbol: Option<String>,
    },
}

/// A stop: the probe that fired (`0`: the end of a step), whether it holds the machine, where
/// it is, and the substrate's facts when the probe asked for them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stop {
    pub probe: u32,
    pub hold: bool,
    pub position: Position,
    pub info: Option<StopInfo>,
}

/// What steps count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepBy {
    /// Syscall entries: stop again at the entry of the nth syscall after now.
    Sys,
    /// Instructions of the stopped thread.
    Insn,
    /// Source lines of the stopped thread (the personality's line tables).
    Line,
}

/// The debugger's answer to a stop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resume {
    /// The debugger changed the machine (registers, memory) during the stop.
    pub wrote: bool,
    /// Stop again after `n` steps of this kind (only for a holding stop).
    pub step: Option<(StepBy, u64)>,
}

/// The machine at a stop, for the debugger: the stopped thread and the personality.
pub trait StopTarget {
    /// The thread that stopped.
    fn tid(&self) -> Tid;
    /// The personality's own state ([`crate::Personality::as_any`]), for the debugger's typed
    /// reads and writes.
    fn personality(&mut self) -> Option<&mut dyn Any>;
}
