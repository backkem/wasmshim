//! The engine in-process: a [`GuestCpu`] that runs translated code where it has some (linked
//! ahead of time) and the interpreter elsewhere, on an [`EngineMem`] address space. This is
//! what a node without the run-time tier runs; it needs no host capability.

#![allow(unsafe_code)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::rc::Rc;
use core::cell::RefCell;

use wasmshim_aarch64::Cpu;
use wasmshim_cpu::{CpuContext, DebugCpu, Event, GuestCpu, GuestMemory, PagedMemory};

use crate::dispatch::{Stop, Tier, dispatch};
use crate::mem::EngineMem;

/// A fresh address space for [`DbtCpu`]s.
pub fn new_memory() -> Box<dyn GuestMemory> {
    Box::new(EngineMem::new(PagedMemory::new()))
}

/// `true` if translated code is linked into this module (else [`DbtCpu`] would only add
/// dispatch overhead to the interpreter).
pub fn has_translations() -> bool {
    crate::aot::directory().is_some()
}

/// Engine counters shared by the CPUs of one machine (never guest-visible).
pub type SharedTier = Rc<RefCell<Tier>>;

/// One A64 hardware thread on the two-tier engine.
pub struct DbtCpu {
    cpu: Box<Cpu>,
    tier: SharedTier,
}

impl DbtCpu {
    /// A CPU with every register zero.
    pub fn new(tier: SharedTier) -> Self {
        Self {
            cpu: Box::new(Cpu::new()),
            tier,
        }
    }
}

impl DbtCpu {
    /// Translated code where there is some, the interpreter elsewhere; keeps the exclusive
    /// monitor (the callers decide).
    fn run_engine(&mut self, mem: &mut dyn GuestMemory, budget: u64) -> Event {
        let Some(m) = mem.as_any_mut().and_then(|a| a.downcast_mut::<EngineMem>()) else {
            // Not an engine address space: plain interpretation.
            return self.cpu.run_raw(mem, budget);
        };
        let mut left = budget;
        let mut tier = *self.tier.borrow();
        // Run-time translation only with a host that instantiates it (`crate::jit`).
        tier.threshold = crate::jit::threshold();
        let cpu: *mut Cpu = &mut *self.cpu;
        let event = loop {
            // SAFETY: `cpu` and `m` are exclusively ours for the call.
            match unsafe { dispatch(cpu, m, &mut left, &mut tier) } {
                Stop::Event(e) => break e,
                Stop::Hot(pc) => crate::jit::hot(m, pc),
            }
        };
        self.tier.borrow_mut().stats = tier.stats;
        event
    }
}

/// Debugger stops. Without breakpoints the run is the engine's own (a budget is
/// exact in every tier, the monitor is kept). With breakpoints the thread is interpreted, one
/// instruction at a time: translated regions never stop mid-region at an arbitrary address, so
/// they are not entered while a breakpoint is armed. Only wall time differs.
impl DebugCpu for DbtCpu {
    fn run_debug(&mut self, mem: &mut dyn GuestMemory, budget: u64, breaks: &[u64]) -> Event {
        if breaks.is_empty() {
            self.run_engine(mem, budget)
        } else {
            self.cpu.run_debug_raw(mem, budget, breaks)
        }
    }

    fn clear_monitor(&mut self) {
        self.cpu.clear_monitor();
    }
}

impl GuestCpu for DbtCpu {
    fn run_until_event(&mut self, mem: &mut dyn GuestMemory, budget: u64) -> Event {
        let event = self.run_engine(mem, budget);
        self.cpu.clear_monitor();
        event
    }

    fn debug(&mut self) -> Option<&mut dyn DebugCpu> {
        Some(self)
    }

    fn set_syscall_result(&mut self, value: u64) {
        self.cpu.set_syscall_result(value);
    }
    fn pc(&self) -> u64 {
        self.cpu.pc()
    }
    fn set_pc(&mut self, pc: u64) {
        self.cpu.set_pc(pc);
    }
    fn sp(&self) -> u64 {
        self.cpu.sp()
    }
    fn set_sp(&mut self, sp: u64) {
        self.cpu.set_sp(sp);
    }
    fn thread_pointer(&self) -> u64 {
        self.cpu.thread_pointer()
    }
    fn set_thread_pointer(&mut self, tp: u64) {
        self.cpu.set_thread_pointer(tp);
    }
    fn retired(&self) -> u64 {
        self.cpu.retired()
    }
    fn set_virtual_time(&mut self, ns: u64) {
        self.cpu.set_virtual_time(ns);
    }
    fn context(&self) -> CpuContext {
        self.cpu.context()
    }
    fn set_context(&mut self, ctx: &CpuContext) {
        self.cpu.set_context(ctx);
    }
    fn duplicate(&self) -> Box<dyn GuestCpu> {
        Box::new(Self {
            cpu: Box::new((*self.cpu).clone()),
            tier: self.tier.clone(),
        })
    }
}
