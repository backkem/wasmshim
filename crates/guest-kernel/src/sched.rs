//! The deterministic scheduler and syscall entry.
//!
//! * Tasks are green threads in one instance. The scheduler runs the first runnable task of a
//!   FIFO run queue for up to one quantum of retired instructions, then rotates.
//! * Every quantum of guest work is charged as virtual time (`ns_per_insn`) and yields to the
//!   host ([`Status::Wait`]), so compute advances the clock and other machines, timers and
//!   sockets progress.
//! * A syscall is entered here (time taken at entry), dispatched to the
//!   [`Personality`], traced, and its [`Flow`] applied.

use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use wasmshim_cpu::{Event, Fault, GuestMemory, SyscallFrame};

use crate::host::Host;
use crate::observe::{self, Phase};
use crate::task::{TaskState, Tid};
use crate::wait::Blocker;
use crate::{Core, FlowOf, Types};

/// The instruction set of the tasks' CPUs (what syscall entry must know about it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// AArch64 (`SVC #0`).
    Aarch64,
}

impl Isa {
    /// Length of the syscall instruction (to restart a syscall by rewinding the PC).
    pub fn syscall_insn_len(self) -> u64 {
        match self {
            Self::Aarch64 => 4,
        }
    }
}

/// Scheduler tunables. All of them affect guest-visible behaviour.
#[derive(Clone, Copy, Debug)]
pub struct SchedConfig {
    pub isa: Isa,
    /// Instructions a task may run before the scheduler switches tasks (and yields to the host).
    /// At least 1: a [`Core`] made with 0 uses 1, since a quantum of nothing
    /// would yield before every instruction and never run one.
    pub quantum: u64,
    /// Virtual nanoseconds charged per retired instruction (`0` disables).
    pub ns_per_insn: u64,
    /// Emit a syscall trace through [`Host::diag`].
    pub trace: bool,
}

impl SchedConfig {
    /// This configuration as a core runs it: a `quantum` of 0 becomes 1.
    pub(crate) fn checked(self) -> Self {
        Self {
            quantum: self.quantum.max(1),
            ..self
        }
    }
}

/// What [`run`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The personality reports the run finished ([`Personality::finished`]).
    Exited,
    /// Let virtual time pass until `until` (monotonic ns), then call [`run`] again. `None`
    /// means every task is blocked with no deadline: nothing can ever happen.
    Wait { until: Option<u64> },
}

/// What happens to the calling task after a syscall.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow<K, P> {
    /// Return this raw value (already `-errno` encoded for errors).
    Ret(u64),
    /// The task got a fresh register state (`execve`); write nothing.
    Exec,
    /// Suspend the calling task (only it; the machine keeps running).
    Block(Blocker<K, P>),
    /// End the calling task.
    ExitThread(i32),
    /// End the calling task's group (the personality decides what a group is).
    ExitGroup(i32),
}

/// An address space as the scheduler needs it: the memory the task's CPU runs against.
pub trait Space {
    fn memory(&mut self) -> &mut dyn GuestMemory;
}

/// A personality: a syscall table plus its own objects over a [`Core`].
///
/// Hooks are called at fixed points of [`run`]; their order is part of determinism.
pub trait Personality {
    type T: Types;
    type Space: Space;

    fn core(&self) -> &Core<Self::T>;
    fn core_mut(&mut self) -> &mut Core<Self::T>;
    /// The run is over (Linux: the initial process ended).
    fn finished(&self) -> bool;
    /// The address space task `tid` runs in, if it still has one.
    fn space(&self, tid: Tid) -> Option<Rc<RefCell<Self::Space>>>;
    /// Handles one syscall (entry bookkeeping is done: `current`, `current_pc`, `slice_now`).
    fn syscall(&mut self, tid: Tid, frame: SyscallFrame, host: &mut dyn Host) -> FlowOf<Self::T>;
    /// A synchronous fault of task `tid` (the PC still points at the faulting instruction).
    fn fault(&mut self, tid: Tid, fault: Fault, host: &mut dyn Host);
    /// `Flow::ExitThread`.
    fn exit_thread(&mut self, tid: Tid, code: i32);
    /// `Flow::ExitGroup`.
    fn exit_group(&mut self, tid: Tid, code: i32);
    /// The raw result a timed-out futex wait returns (Linux: `-ETIMEDOUT`).
    fn futex_timeout(&self) -> u64;

    /// At the start of every [`run`] (Linux: collect the network provider's changes).
    fn on_run(&mut self) {}
    /// Before any task runs at `now` (Linux: itimers and POSIX timers).
    fn expire(&mut self, _now: u64) {}
    /// The personality's own next deadline (its timers).
    fn next_deadline(&self) -> Option<u64> {
        None
    }
    /// Before each stretch of `tid` within a slice (Linux: signal delivery). `false`: do not
    /// run it now.
    fn before_run(&mut self, _tid: Tid, _host: &mut dyn Host) -> bool {
        true
    }
    /// After every syscall, before it is traced (Linux: inotify events, the console).
    fn after_syscall(&mut self, _host: &mut dyn Host) {}

    /// Trace: the syscall's name and how many arguments to show (the ABI table).
    fn describe(&self, _number: u64) -> Option<(&'static str, usize)> {
        None
    }
    /// Trace: how a task is labelled (`[label] name(args) = result`).
    fn label(&self, tid: Tid) -> String {
        format!("{tid}")
    }
    /// Trace: how an error result reads (Linux: `-1 ENOENT`); `None` for a plain value.
    fn show_error(&self, _ret: u64) -> Option<String> {
        None
    }

    /// The machine's observer (armed syscall probes, the `sys`
    /// clock). `None`, the default: never observed.
    fn observer(&mut self) -> Option<&mut crate::observe::Observer> {
        None
    }
    /// The process (address space) of task `tid`, for PC breakpoints
    /// scoped to one. `None`, the default: such breakpoints never match.
    fn process_of(&self, _tid: Tid) -> Option<u32> {
        None
    }
    /// A name for `pc` in task `tid`'s program (`function+0x14`), for stop info.
    fn symbolize(&mut self, _tid: Tid, _pc: u64) -> Option<String> {
        None
    }
    /// The source line at `pc` in task `tid`'s program (file, line) and whether `pc`
    /// starts a statement of it, for stepping by line. `None`, the default: no line tables.
    fn source_line(&mut self, _tid: Tid, _pc: u64) -> Option<(String, u32, bool)> {
        None
    }
    /// The personality's own state, for a debugger at a stop ([`crate::debug::StopTarget`]).
    /// `None`, the default: a debugger sees only the substrate's facts.
    fn as_any(&mut self) -> Option<&mut dyn core::any::Any> {
        None
    }
}

/// Runs tasks until the personality reports the run finished or every task waits.
pub fn run<P: Personality>(p: &mut P, host: &mut dyn Host) -> Status {
    let status = run_inner(p, host);
    p.core_mut().flush_diags(host);
    status
}

fn run_inner<P: Personality>(p: &mut P, host: &mut dyn Host) -> Status {
    p.on_run();
    {
        let core = p.core();
        if !p.finished() && core.busy_until > host.monotonic_ns() {
            // Still paying for the last quantum: events that arrived meanwhile wait for the CPU.
            return Status::Wait {
                until: Some(core.busy_until),
            };
        }
    }
    loop {
        p.core_mut().flush_diags(host);
        if p.finished() {
            return Status::Exited;
        }
        // Yield to the host at every quantum of guest work, charging it as virtual time.
        {
            let core = p.core_mut();
            if core.since_yield >= core.cfg.quantum {
                // The deadline counts from when the work started: identical under virtual
                // time, and usually already past under a real clock (then a pure yield).
                let charge = core.since_yield.saturating_mul(core.cfg.ns_per_insn);
                core.since_yield = 0;
                core.busy_until = core.accrual_start.saturating_add(charge);
                return Status::Wait {
                    until: Some(core.busy_until),
                };
            }
        }
        let now = host.monotonic_ns();
        p.core_mut().stats.now = now;
        p.expire(now);
        let futex_timeout = p.futex_timeout();
        p.core_mut().expire_deadlines(now, futex_timeout);
        let Some(tid) = p.core_mut().next_runnable() else {
            let until = p
                .core()
                .tasks
                .values()
                .filter_map(|t| match t.state {
                    TaskState::Blocked(b) => b.deadline(),
                    TaskState::Runnable => None,
                })
                .chain(p.next_deadline())
                .min();
            let core = p.core_mut();
            // Compute not yet charged is charged before the tasks sleep.
            let charge = core.since_yield.saturating_mul(core.cfg.ns_per_insn);
            core.since_yield = 0;
            let floor = core.accrual_start.saturating_add(charge);
            if charge > 0 {
                // Even if nothing has a deadline (tasks wait for I/O), the CPU time just spent
                // is paid before anything else runs.
                core.busy_until = floor;
                return Status::Wait {
                    until: Some(until.map_or(floor, |u| u.max(floor))),
                };
            }
            return Status::Wait {
                until: until.map(|u| u.max(floor)),
            };
        };
        run_slice(p, tid, now, host);
    }
}

impl<T: Types> Core<T> {
    fn next_runnable(&mut self) -> Option<Tid> {
        while let Some(tid) = self.run_queue.pop_front() {
            if self
                .tasks
                .get(&tid)
                .is_some_and(|t| t.state == TaskState::Runnable)
            {
                return Some(tid);
            }
        }
        None
    }
}

/// Runs one scheduling slice of `tid`.
fn run_slice<P: Personality>(p: &mut P, tid: Tid, now: u64, host: &mut dyn Host) {
    p.core_mut().slice_now = now;
    let quantum = p.core().cfg.quantum;
    let mut used = 0;
    while used < quantum {
        if !p.before_run(tid, host) {
            return;
        }
        match p.core().tasks.get(&tid) {
            Some(t) if t.state == TaskState::Runnable => {}
            _ => return,
        }
        let Some(space) = p.space(tid) else {
            return;
        };
        let armed = p.observer().is_some_and(|o| o.cpu_armed());
        let event = if armed {
            // Instruction stops cut the run and the quantum goes on.
            match run_observed(p, tid, now, quantum, &mut used, &space, host) {
                Some(event) => event,
                None => return,
            }
        } else {
            let core = p.core_mut();
            let Some(task) = core.tasks.get_mut(&tid) else {
                return;
            };
            let before = task.cpu.retired();
            task.cpu.set_virtual_time(now);
            let event = task
                .cpu
                .run_until_event(space.borrow_mut().memory(), quantum - used);
            let delta = task.cpu.retired() - before;
            used += delta.max(1);
            if core.since_yield == 0 {
                core.accrual_start = now;
            }
            core.since_yield += delta;
            event
        };
        match event {
            Event::Yield => break,
            Event::Syscall(frame) => {
                if !handle_syscall(p, tid, frame, host) {
                    return;
                }
            }
            Event::Fault(fault) => p.fault(tid, fault, host),
        }
    }
    let core = p.core_mut();
    if core
        .tasks
        .get(&tid)
        .is_some_and(|t| t.state == TaskState::Runnable)
    {
        core.run_queue.push_back(tid);
    }
}

/// One run of `tid` while instruction stops are armed: the stops due before it fire
/// (holds), then the CPU runs cut at the next instruction target and before its breakpoints.
/// A cut that is not the end of the quantum, a syscall or a fault goes on in the same quantum
/// with the exclusive monitor kept, so the thread's run is the one it would have had without
/// stops; every real boundary clears the monitor, as `run_until_event` does. `None`: the task
/// is gone or no longer runnable.
fn run_observed<P: Personality>(
    p: &mut P,
    tid: Tid,
    now: u64,
    quantum: u64,
    used: &mut u64,
    space: &Rc<RefCell<P::Space>>,
    host: &mut dyn Host,
) -> Option<Event> {
    loop {
        if observe::before_insn(p, tid, host) {
            // A debugger's store breaks the reservation, like another agent's.
            if let Some(d) = p.core_mut().tasks.get_mut(&tid)?.cpu.debug() {
                d.clear_monitor();
            }
        }
        match p.core().tasks.get(&tid) {
            Some(t) if t.state == TaskState::Runnable => {}
            _ => return None,
        }
        let rest = quantum - *used;
        let cut = if p.observer().is_some_and(|o| o.cpu_armed()) {
            observe::plan(p, tid, rest)
        } else {
            observe::Cut {
                budget: rest,
                breaks: Vec::new(),
            }
        };
        // Watchpoints: the view of this thread's process while its CPU runs.
        let pid = p.process_of(tid);
        let watches = p.observer().map(|o| o.watches(pid)).unwrap_or_default();
        let view = observe::watch_view(&watches);
        if !watches.is_empty() {
            let mut s = space.borrow_mut();
            observe::watch_snapshot(p, pid, &watches, s.memory());
        }
        let core = p.core_mut();
        let task = core.tasks.get_mut(&tid)?;
        let before = task.cpu.retired();
        task.cpu.set_virtual_time(now);
        let mut trap = None;
        let (event, debugged) = {
            let mut space = space.borrow_mut();
            let mem = space.memory();
            match task.cpu.debug() {
                Some(d) => {
                    let viewing = !view.is_empty() && mem.set_watch(&view);
                    let e = d.run_debug(mem, cut.budget, &cut.breaks);
                    if viewing {
                        mem.set_watch(&[]);
                    }
                    match e {
                        // A fault the view made: the instruction runs again without it, exactly
                        // as it would have, and the watches compare before and after.
                        Event::Fault(Fault::Memory { pc, fault })
                            if viewing && observe::is_watch_fault(&view, mem, &fault) =>
                        {
                            let old = observe::watch_read(&watches, mem);
                            let again = d.run_debug(mem, 1, &[]);
                            let new = observe::watch_read(&watches, mem);
                            if !matches!(again, Event::Fault(_)) {
                                trap = Some((pc, fault, old, new));
                            }
                            (again, true)
                        }
                        e => (e, true),
                    }
                }
                // No debugger in this backend: it runs as if nothing were armed.
                None => (task.cpu.run_until_event(mem, rest), false),
            }
        };
        let delta = task.cpu.retired() - before;
        *used += delta.max(1);
        if core.since_yield == 0 {
            core.accrual_start = now;
        }
        core.since_yield += delta;
        if let Some((pc, fault, old, new)) = trap
            && observe::watch_hit(p, tid, pc, &fault, &watches, &old, &new, host)
            && let Some(d) = p.core_mut().tasks.get_mut(&tid)?.cpu.debug()
        {
            d.clear_monitor();
        }
        if debugged && matches!(event, Event::Yield) && *used < quantum {
            continue;
        }
        if debugged && let Some(d) = p.core_mut().tasks.get_mut(&tid)?.cpu.debug() {
            d.clear_monitor();
        }
        return Some(event);
    }
}

/// Enters, dispatches, traces and completes one syscall; returns `false` if the task stopped
/// running (exited, blocked).
fn handle_syscall<P: Personality>(
    p: &mut P,
    tid: Tid,
    mut frame: SyscallFrame,
    host: &mut dyn Host,
) -> bool {
    {
        let core = p.core_mut();
        if let Some(t) = core.tasks.get_mut(&tid) {
            t.last_syscall = Some(frame.number);
        }
        core.current = tid;
        core.current_pc = frame.pc;
    }
    // The time of this syscall (under a real clock, later than the slice's start).
    p.core_mut().slice_now = host.monotonic_ns();
    {
        let core = p.core_mut();
        core.stats.now = core.slice_now;
        core.stats.enter(tid, frame.number);
    }
    // Observer stops at entry (and below at exit); nothing unless armed.
    observe::on_syscall(p, tid, &mut frame, Phase::Entry, host);
    let flow = p.syscall(tid, frame, host);
    p.after_syscall(host);
    if p.core().cfg.trace {
        trace(p, tid, &frame, &flow, host);
    }
    match flow {
        Flow::Ret(v) => {
            if let Some(t) = p.core_mut().tasks.get_mut(&tid) {
                t.timed = None;
                t.cpu.set_syscall_result(v);
            }
            observe::on_syscall(p, tid, &mut frame, Phase::Exit(v), host);
            true
        }
        Flow::Exec => true,
        Flow::Block(blocker) => {
            let core = p.core_mut();
            let rewind = core.cfg.isa.syscall_insn_len();
            if let Some(t) = core.tasks.get_mut(&tid) {
                if let Blocker::Restart { resume_pc, .. } = blocker {
                    t.cpu.set_pc(resume_pc - rewind);
                }
                t.state = TaskState::Blocked(blocker);
            }
            let now = core.slice_now;
            core.stats.block(tid, frame.number, now);
            false
        }
        Flow::ExitThread(code) => {
            p.exit_thread(tid, code);
            false
        }
        Flow::ExitGroup(code) => {
            p.exit_group(tid, code);
            false
        }
    }
}

/// Writes one `strace`-like line: `[label] name(args) = result`.
fn trace<P: Personality>(
    p: &P,
    tid: Tid,
    frame: &SyscallFrame,
    flow: &FlowOf<P::T>,
    host: &mut dyn Host,
) {
    let (name, n) = match p.describe(frame.number) {
        Some((name, n)) => (String::from(name), n.min(6)),
        None => (format!("syscall_{}", frame.number), 6),
    };
    let args: Vec<String> = frame.args[..n].iter().map(|a| format!("{a:#x}")).collect();
    let result = match flow {
        Flow::Ret(v) => p.show_error(*v).unwrap_or_else(|| format!("{v:#x}")),
        Flow::Exec => "0".into(),
        Flow::Block(b) => format!("? <blocked: {b:?}>"),
        Flow::ExitThread(c) | Flow::ExitGroup(c) => format!("? <exit {c}>"),
    };
    host.diag(&format!(
        "[{}] {name}({}) = {result}",
        p.label(tid),
        args.join(", ")
    ));
}
