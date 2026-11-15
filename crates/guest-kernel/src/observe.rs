//! Observer stops: in-device holds at syscall entry and exit and at exact instructions, and
//! actions that replay, for every personality. The typed contract is [`crate::debug`].
//!
//! - **Clocks.** `sys` counts syscall entries instance-wide (restarts of a blocked call included),
//!   from 1; `insn` is the instance's retired instructions ([`crate::Core::retired`]). Both are
//!   host-side: nothing the guest sees depends on them.
//! - **Exit** is after a plain result (including a restarted call's last attempt); a call
//!   completed by a wake (`wait.rs`: sleeps, futex, deadlines) has no exit stop yet.
//! - **Points** ([`Point`]): a syscall by number with conditions (entry or exit, thread, equal
//!   arguments, a minimum result), the Nth syscall, the instance's Nth retired instruction (periodic
//!   with `every`: a sampler whose samples are exact positions), a PC in a thread or process,
//!   and accesses to a range of memory.
//! - **Stops** go to [`crate::Host::stop`] (the embedder's debugger; the default resumes at
//!   once), with the substrate's [`crate::debug::StopInfo`] when the probe asked for it. Its
//!   [`crate::debug::Resume`] may step: by `sys` (stop again at the entry of the nth syscall
//!   after now), `insn` (after the stopped thread retires n more instructions) or `line` (when
//!   it reaches the start of a statement of another source line n times,
//!   [`crate::Personality::source_line`]; it steps by instructions meanwhile).
//! - **Writes** at an entry stop that change the argument registers are seen by the syscall: the
//!   frame is re-read from the registers after the stop.
//!
//! **Instruction stops are invisible**: the slice is cut with
//! [`wasmshim_cpu::DebugCpu::run_debug`] (an exact budget, PC breakpoints, the exclusive monitor
//! kept), the stop runs, and the *same* quantum continues: the thread retires the same
//! instructions and takes the same preemption points as without the stop. A write at a stop
//! clears the monitor (a debugger's store, like another agent's, breaks a reservation).
//!
//! Nothing here runs unless a probe or a step is armed: per syscall one counter increment and
//! an emptiness check, per CPU run one flag test, and retired counts never change.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use wasmshim_cpu::{Access, CpuContext, GuestMemory, MemFault, MemFaultKind, Perms, SyscallFrame};

use crate::debug::{Position, ProbeSpec, Resume, StepBy, Stop, StopInfo, StopTarget, WatchBy, Why};
use crate::host::Host;
use crate::sched::{Personality, Space};
use crate::task::Tid;

/// Where in a syscall a stop is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Entry,
    /// After the result (raw) was written back.
    Exit(u64),
}

/// Where a probe fires.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Point {
    /// A syscall by number (any, without one), with conditions.
    Syscall {
        number: Option<u64>,
        exit: bool,
        tid: Option<Tid>,
        args: [Option<u64>; 6],
        /// At exit: the raw result, as a signed value, at least this.
        ret_ge: Option<i64>,
    },
    /// The Nth syscall.
    Sys { n: u64, exit: bool },
    /// The instance has retired exactly `n` instructions (then every `every` more, if set).
    Insn { n: u64, every: Option<u64> },
    /// A thread is about to execute the instruction at `addr`.
    Pc {
        addr: u64,
        pid: Option<u32>,
        tid: Option<Tid>,
    },
    /// Accesses (`deny`: write and/or read) to `[start, start + len)` in process `pid` (or any).
    Watch {
        start: u64,
        len: u64,
        pid: Option<u32>,
        deny: Perms,
    },
}

impl Point {
    fn is_cpu(&self) -> bool {
        matches!(
            self,
            Point::Insn { .. } | Point::Pc { .. } | Point::Watch { .. }
        )
    }
}

/// One armed probe.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Probe {
    point: Point,
    hits: Option<u32>,
    hold: bool,
    info: bool,
}

/// The armed probes by id.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Probes {
    map: BTreeMap<u32, Probe>,
}

impl Probes {
    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn iter(&self) -> impl Iterator<Item = (u32, &Probe)> {
        self.map.iter().map(|(&id, p)| (id, p))
    }

    fn point_mut(&mut self, id: u32) -> Option<&mut Point> {
        self.map.get_mut(&id).map(|p| &mut p.point)
    }

    /// Takes one hit of probe `id`: the probe, which is disarmed if its hits ran out.
    fn hit(&mut self, id: u32) -> Option<Probe> {
        let p = self.map.get_mut(&id)?;
        if let Some(h) = &mut p.hits {
            *h = h.saturating_sub(1);
        }
        if p.hits == Some(0) {
            self.map.remove(&id)
        } else {
            Some(p.clone())
        }
    }
}
/// The page size watch views work in.
const PAGE: u64 = 0x1000;

/// A machine's armed probes and its `sys` clock. A personality owns one and hands it to the
/// scheduler through [`Personality::observer`].
#[derive(Clone, Debug, Default)]
pub struct Observer {
    probes: Probes,
    /// Syscall entries so far.
    syscalls: u64,
    /// A step's stop: the entry of this syscall.
    step_to: Option<u64>,
    /// A step's stop: when this thread's CPU has retired this many instructions.
    step_insn: Option<(Tid, u64)>,
    /// A step by source line in progress (driven by single instruction steps).
    step_line: Option<StepLine>,
    /// Instruction probes are armed (or an instruction step): the scheduler cuts slices.
    cpu_armed: bool,
    /// The bytes of each watched range as last seen, by (probe, process): what a syscall's
    /// write is compared with.
    snapshots: BTreeMap<(u32, u32), Vec<u8>>,
    /// The instruction position of the last instruction stop (it fires once there).
    stopped_at: Option<u64>,
    /// The thread and its retired count where a PC stop last fired (resuming steps over it).
    resumed_from: Option<(Tid, u64)>,
}

/// A step by source line: the thread, the line it is leaving, and how many lines are left.
#[derive(Clone, Debug, PartialEq, Eq)]
struct StepLine {
    tid: Tid,
    from: Option<(String, u32)>,
    left: u64,
}

/// A cut of one CPU run: at most `budget` instructions, stopping before `breaks`.
pub(crate) struct Cut {
    pub budget: u64,
    pub breaks: Vec<u64>,
}

impl Observer {
    /// Syscall entries so far (the `sys` clock).
    pub fn syscalls(&self) -> u64 {
        self.syscalls
    }

    /// Whether anything can stop or write.
    pub fn armed(&self) -> bool {
        !self.probes.is_empty() || self.step_to.is_some() || self.step_insn.is_some()
    }

    /// Whether CPU runs must be cut (instruction probes or an instruction step).
    pub fn cpu_armed(&self) -> bool {
        self.cpu_armed
    }

    fn recompute(&mut self) {
        self.cpu_armed =
            self.step_insn.is_some() || self.probes.iter().any(|(_, p)| p.point.is_cpu());
    }

    /// Arms probes: each replaces the one with its id, and `hits: Some(0)` disarms it. The
    /// caller has checked them (id `0` is the step's and never armed).
    pub fn arm(&mut self, probes: &[ProbeSpec]) {
        for p in probes {
            if p.hits == Some(0) {
                self.probes.map.remove(&p.id);
            } else {
                self.probes.map.insert(
                    p.id,
                    Probe {
                        point: p.point.clone(),
                        hits: p.hits,
                        hold: p.hold,
                        info: p.info,
                    },
                );
            }
        }
        self.recompute();
    }

    /// Disarms probes by id.
    pub fn disarm(&mut self, ids: &[u32]) {
        for id in ids {
            self.probes.map.remove(id);
        }
        self.recompute();
    }
    fn fired(&self, tid: Tid, frame: &SyscallFrame, phase: Phase) -> Vec<u32> {
        let exit = matches!(phase, Phase::Exit(_));
        self.probes
            .iter()
            .filter(|(_, p)| match &p.point {
                Point::Sys { n, exit: e } => *n == self.syscalls && *e == exit,
                Point::Syscall {
                    number,
                    exit: e,
                    tid: t,
                    args,
                    ret_ge,
                } => {
                    *e == exit
                        && number.is_none_or(|n| n == frame.number)
                        && t.is_none_or(|t| t == tid)
                        && args
                            .iter()
                            .zip(frame.args)
                            .all(|(want, got)| want.is_none_or(|w| w == got))
                        && match (ret_ge, phase) {
                            (None, _) => true,
                            (Some(min), Phase::Exit(ret)) => ret as i64 >= *min,
                            (Some(_), Phase::Entry) => false,
                        }
                }
                Point::Insn { .. } | Point::Pc { .. } | Point::Watch { .. } => false,
            })
            .map(|(id, _)| id)
            .collect()
    }

    fn pc_matches(point: &Point, tid: Tid, pid: Option<u32>) -> Option<u64> {
        match point {
            Point::Pc {
                addr,
                pid: want_pid,
                tid: want_tid,
            } if want_tid.is_none_or(|t| t == tid) && want_pid.is_none_or(|w| pid == Some(w)) => {
                Some(*addr)
            }
            _ => None,
        }
    }

    /// The instruction stops due before thread `tid` runs on: `(probe id, why)`, probe 0 for a
    /// step.
    fn due(&self, tid: Tid, pid: Option<u32>, node: u64, thread: u64, pc: u64) -> Vec<(u32, Why)> {
        let mut out = Vec::new();
        if self.step_insn == Some((tid, thread)) {
            out.push((0, Why::Step));
        }
        let resumed = self.resumed_from == Some((tid, thread));
        for (id, p) in self.probes.iter() {
            match &p.point {
                Point::Insn { n, .. } if *n == node && self.stopped_at != Some(node) => {
                    out.push((id, Why::Insn));
                }
                point if !resumed && Self::pc_matches(point, tid, pid) == Some(pc) => {
                    out.push((id, Why::Pc));
                }
                _ => {}
            }
        }
        out
    }

    /// How to run thread `tid` next: at most `budget` instructions (the slice's rest), cut at
    /// the next instruction target and before its breakpoints.
    pub(crate) fn cut(
        &self,
        tid: Tid,
        pid: Option<u32>,
        node: u64,
        thread: u64,
        budget: u64,
    ) -> Cut {
        let mut budget = budget;
        for (_, p) in self.probes.iter() {
            if let Point::Insn { n, .. } = p.point
                && n > node
            {
                budget = budget.min(n - node);
            }
        }
        if let Some((t, target)) = self.step_insn
            && t == tid
            && target > thread
        {
            budget = budget.min(target - thread);
        }
        let mut breaks: Vec<u64> = self
            .probes
            .iter()
            .filter_map(|(_, p)| Self::pc_matches(&p.point, tid, pid))
            .collect();
        breaks.sort_unstable();
        breaks.dedup();
        Cut { budget, breaks }
    }
}

/// The frame's registers as AArch64 passes them (number in `x8`, arguments in `x0`–`x5`), read
/// back after a stop that may have written them.
fn refresh_frame(ctx: &CpuContext, frame: &mut SyscallFrame) {
    match ctx {
        CpuContext::A64(c) => {
            frame.number = c.x[8];
            frame.args.copy_from_slice(&c.x[..6]);
        }
    }
}

/// One armed watch as a run of a process sees it.
#[derive(Clone, Debug)]
pub(crate) struct Watch {
    id: u32,
    start: u64,
    len: u64,
    deny: Perms,
}

impl Observer {
    /// The watches that apply to process `pid`.
    pub(crate) fn watches(&self, pid: Option<u32>) -> Vec<Watch> {
        self.probes
            .iter()
            .filter_map(|(id, p)| match p.point {
                Point::Watch {
                    start,
                    len,
                    pid: want,
                    deny,
                } if want.is_none_or(|w| pid == Some(w)) => Some(Watch {
                    id,
                    start,
                    len,
                    deny,
                }),
                _ => None,
            })
            .collect()
    }
}

/// The watch view of `watches`: every page they touch, with the accesses to deny there.
pub(crate) fn watch_view(watches: &[Watch]) -> Vec<(u64, Perms)> {
    let mut view: Vec<(u64, Perms)> = Vec::new();
    for w in watches {
        let mut page = w.start & !(PAGE - 1);
        while page < w.start.saturating_add(w.len) {
            match view.iter_mut().find(|(p, _)| *p == page) {
                Some((_, d)) => *d = *d | w.deny,
                None => view.push((page, w.deny)),
            }
            page += PAGE;
        }
    }
    view
}

/// The watched bytes now (empty where unreadable).
pub(crate) fn watch_read(watches: &[Watch], mem: &dyn GuestMemory) -> Vec<Vec<u8>> {
    watches
        .iter()
        .map(|w| {
            let mut b = alloc::vec![0; w.len as usize];
            match mem.read(w.start, &mut b) {
                Ok(()) => b,
                Err(_) => Vec::new(),
            }
        })
        .collect()
}

/// Whether `fault` (with the view cleared) is the view's: a protection fault on a watched page
/// that the guest's own permissions allow.
pub(crate) fn is_watch_fault(
    view: &[(u64, Perms)],
    mem: &dyn GuestMemory,
    fault: &MemFault,
) -> bool {
    fault.kind == MemFaultKind::Protection
        && view.iter().any(|(p, _)| *p == fault.addr & !(PAGE - 1))
        && mem
            .perms(fault.addr)
            .is_some_and(|g| g.allows(fault.access))
}

/// Takes the first snapshots of `watches` in process `pid` (a syscall's write is compared with
/// them).
pub(crate) fn watch_snapshot<P: Personality>(
    p: &mut P,
    pid: Option<u32>,
    watches: &[Watch],
    mem: &dyn GuestMemory,
) {
    let Some(obs) = p.observer() else { return };
    for w in watches {
        let key = (w.id, pid.unwrap_or(0));
        if let alloc::collections::btree_map::Entry::Vacant(slot) = obs.snapshots.entry(key) {
            let mut b = alloc::vec![0; w.len as usize];
            if mem.read(w.start, &mut b).is_ok() {
                slot.insert(b);
            }
        }
    }
}

/// After the instruction at `pc` that hit the view was run again: the watches it touched stop
/// (a write that changed the range or wrote into it; a read from it). Returns whether a write
/// was applied at a stop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn watch_hit<P: Personality>(
    p: &mut P,
    tid: Tid,
    pc: u64,
    fault: &MemFault,
    watches: &[Watch],
    old: &[Vec<u8>],
    new: &[Vec<u8>],
    host: &mut dyn Host,
) -> bool {
    let pid = p.process_of(tid);
    let mut hits = Vec::new();
    for (i, w) in watches.iter().enumerate() {
        if let Some(obs) = p.observer() {
            obs.snapshots
                .insert((w.id, pid.unwrap_or(0)), new[i].clone());
        }
        let inside = fault.addr >= w.start && fault.addr < w.start.saturating_add(w.len);
        let hit = match fault.access {
            Access::Write => w.deny.write() && (inside || old[i] != new[i]),
            Access::Read => w.deny.read() && inside,
            Access::Execute => false,
        };
        if hit {
            hits.push(i);
        }
    }
    let mut wrote = false;
    for i in hits {
        wrote |= watch_stop(
            p,
            tid,
            pid,
            pc,
            Some(fault),
            &watches[i],
            &old[i],
            &new[i],
            &WatchBy::Cpu,
            host,
        );
    }
    wrote
}

/// After a syscall of `tid`: watched ranges of its process that the kernel changed stop.
pub(crate) fn watch_after_syscall<P: Personality>(
    p: &mut P,
    tid: Tid,
    number: u64,
    host: &mut dyn Host,
) {
    let pid = p.process_of(tid);
    let Some(obs) = p.observer() else { return };
    let watches = obs.watches(pid);
    if watches.is_empty() {
        return;
    }
    let Some(space) = p.space(tid) else { return };
    let now = {
        let mut s = space.borrow_mut();
        watch_read(&watches, s.memory())
    };
    let pc = p.core().tasks.get(&tid).map_or(0, |t| t.cpu.pc());
    let name = p.describe(number).map_or_else(
        || alloc::format!("syscall {number}"),
        |(n, _)| alloc::format!("syscall {n}"),
    );
    let by = WatchBy::Syscall(name);
    for (i, w) in watches.iter().enumerate() {
        let key = (w.id, pid.unwrap_or(0));
        let Some(obs) = p.observer() else { return };
        let before = obs.snapshots.insert(key, now[i].clone());
        if let Some(before) = before
            && before != now[i]
            && w.deny.write()
        {
            watch_stop(p, tid, pid, pc, None, w, &before, &now[i], &by, host);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn watch_stop<P: Personality>(
    p: &mut P,
    tid: Tid,
    pid: Option<u32>,
    pc: u64,
    fault: Option<&MemFault>,
    w: &Watch,
    old: &[u8],
    new: &[u8],
    by: &WatchBy,
    host: &mut dyn Host,
) -> bool {
    let Some(probe) = take_hit(p, w.id) else {
        return false;
    };
    let node = p.core().retired();
    let thread = p.core().tasks.get(&tid).map_or(0, |t| t.cpu.retired());
    let info = (probe.hold && probe.info).then(|| StopInfo::Watch {
        insn: node,
        tid,
        pc,
        retired: thread,
        start: w.start,
        len: w.len,
        old: old.to_vec(),
        new: new.to_vec(),
        by: by.clone(),
        fault: fault.map(|f| (f.addr, f.access)),
        pid,
        symbol: p.symbolize(tid, pc),
    });
    let stop = Stop {
        probe: w.id,
        hold: probe.hold,
        position: Position::Insn(node),
        info,
    };
    stop_at(p, tid, &stop, host)
}

/// Takes one hit of probe `id`; the probe if it is still (or was just) armed.
fn take_hit<P: Personality>(p: &mut P, id: u32) -> Option<Probe> {
    let obs = p.observer()?;
    let probe = obs.probes.hit(id)?;
    if probe.hits == Some(0) {
        obs.recompute();
    } else if let Some(Point::Insn { n, every: Some(e) }) = obs.probes.point_mut(id) {
        // A periodic instruction probe (a sampler): its next instant.
        *n = n.saturating_add(*e);
    }
    Some(probe)
}

/// The scheduler's hook at a syscall's entry and (after a plain result) its exit.
pub(crate) fn on_syscall<P: Personality>(
    p: &mut P,
    tid: Tid,
    frame: &mut SyscallFrame,
    phase: Phase,
    host: &mut dyn Host,
) {
    let Some(obs) = p.observer() else {
        return;
    };
    if phase == Phase::Entry {
        obs.syscalls += 1;
    }
    if !obs.armed() {
        return;
    }
    let n = obs.syscalls;
    let fired = obs.fired(tid, frame, phase);
    let stepped = phase == Phase::Entry && obs.step_to == Some(n);
    if stepped {
        obs.step_to = None;
    }
    let mut held = false;
    let mut wrote = false;
    for id in fired {
        let Some(probe) = take_hit(p, id) else {
            continue;
        };
        held |= probe.hold;
        let info = (probe.hold && probe.info).then(|| syscall_info(p, tid, frame, phase, n));
        let stop = Stop {
            probe: id,
            hold: probe.hold,
            position: Position::Sys {
                n,
                exit: matches!(phase, Phase::Exit(_)),
            },
            info,
        };
        // An action (not a hold) may find the machine changed since it was armed (a thread
        // exited); the debugger then skips it, and a replay diverges only if the session did.
        wrote |= stop_at(p, tid, &stop, host);
    }
    if stepped && !held {
        let stop = Stop {
            probe: 0,
            hold: true,
            position: Position::Sys {
                n,
                exit: matches!(phase, Phase::Exit(_)),
            },
            info: Some(syscall_info(p, tid, frame, phase, n)),
        };
        wrote |= stop_at(p, tid, &stop, host);
    }
    if wrote
        && phase == Phase::Entry
        && let Some(t) = p.core().tasks.get(&tid)
    {
        refresh_frame(&t.cpu.context(), frame);
    }
    // Watched ranges the kernel wrote during the call (the view only catches the CPU's).
    if matches!(phase, Phase::Exit(_)) {
        watch_after_syscall(p, tid, frame.number, host);
    }
}

/// The scheduler's hook before thread `tid` runs on (instruction probes armed): fires the
/// stops due now. Returns whether a write was applied (the caller then clears the monitor).
pub(crate) fn before_insn<P: Personality>(p: &mut P, tid: Tid, host: &mut dyn Host) -> bool {
    let pid = p.process_of(tid);
    let node = p.core().retired();
    let Some(task) = p.core().tasks.get(&tid) else {
        return false;
    };
    let thread = task.cpu.retired();
    let pc = task.cpu.pc();
    let Some(obs) = p.observer() else {
        return false;
    };
    let mut due = obs.due(tid, pid, node, thread, pc);
    if due.is_empty() {
        return false;
    }
    // A step by line goes on one instruction at a time, without stopping, until the thread is
    // at the start of a statement of another line (`left` times).
    let line_step = obs.step_line.clone().filter(|s| s.tid == tid);
    if let Some(mut sl) = line_step
        && due.iter().any(|(_, w)| *w == Why::Step)
    {
        let here = p.source_line(tid, pc);
        let arrived = matches!(&here, Some((f, l, true))
            if sl.from.as_ref().is_none_or(|(ff, ll)| ff != f || ll != l));
        if arrived {
            sl.left -= 1;
        }
        let Some(obs) = p.observer() else {
            return false;
        };
        if arrived && sl.left == 0 {
            obs.step_line = None;
        } else {
            if arrived {
                sl.from = here.map(|(f, l, _)| (f, l));
            }
            obs.step_insn = Some((tid, thread + 1));
            obs.step_line = Some(sl);
            due.retain(|(_, w)| *w != Why::Step);
            if due.is_empty() {
                return false;
            }
        }
    }
    let Some(obs) = p.observer() else {
        return false;
    };
    if due.iter().any(|(_, w)| *w == Why::Insn) {
        obs.stopped_at = Some(node);
    }
    if due.iter().any(|(_, w)| *w == Why::Step) {
        obs.step_insn = None;
        obs.recompute();
    }
    if due.iter().any(|(_, w)| *w == Why::Pc) {
        obs.resumed_from = Some((tid, thread));
    }
    let mut wrote = false;
    let mut held = false;
    let step = due.iter().any(|(_, w)| *w == Why::Step);
    for (id, why) in due.into_iter().filter(|(id, _)| *id != 0) {
        let Some(probe) = take_hit(p, id) else {
            continue;
        };
        held |= probe.hold;
        let info =
            (probe.hold && probe.info).then(|| insn_info(p, tid, pid, node, thread, pc, why));
        let stop = Stop {
            probe: id,
            hold: probe.hold,
            position: Position::Insn(node),
            info,
        };
        wrote |= stop_at(p, tid, &stop, host);
    }
    if step && !held {
        let stop = Stop {
            probe: 0,
            hold: true,
            position: Position::Insn(node),
            info: Some(insn_info(p, tid, pid, node, thread, pc, Why::Step)),
        };
        wrote |= stop_at(p, tid, &stop, host);
    }
    // The thread resumes by executing its next instruction, wherever a register write put the
    // PC (as the CPU's own breakpoints exempt the first instruction of a run).
    if let Some(obs) = p.observer() {
        obs.resumed_from = Some((tid, thread));
    }
    wrote
}

/// Where the next run of `tid` is cut (instruction probes armed).
pub(crate) fn plan<P: Personality>(p: &mut P, tid: Tid, budget: u64) -> Cut {
    let pid = p.process_of(tid);
    let node = p.core().retired();
    let thread = p.core().tasks.get(&tid).map_or(0, |t| t.cpu.retired());
    match p.observer() {
        Some(obs) => obs.cut(tid, pid, node, thread, budget),
        None => Cut {
            budget,
            breaks: Vec::new(),
        },
    }
}

fn syscall_info<P: Personality>(
    p: &mut P,
    tid: Tid,
    frame: &SyscallFrame,
    phase: Phase,
    n: u64,
) -> StopInfo {
    StopInfo::Syscall {
        sys: n,
        number: frame.number,
        tid,
        args: frame.args,
        name: p.describe(frame.number).map(|(name, _)| name),
        ret: match phase {
            Phase::Exit(ret) => Some(ret),
            Phase::Entry => None,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn insn_info<P: Personality>(
    p: &mut P,
    tid: Tid,
    pid: Option<u32>,
    node: u64,
    thread: u64,
    pc: u64,
    why: Why,
) -> StopInfo {
    StopInfo::Insn {
        insn: node,
        tid,
        pc,
        retired: thread,
        why,
        pid,
        symbol: p.symbolize(tid, pc),
    }
}

/// One stop, answered by the host's debugger; applies its step and returns whether it wrote.
fn stop_at<P: Personality>(p: &mut P, tid: Tid, stop: &Stop, host: &mut dyn Host) -> bool {
    let Resume { wrote, step } = host.stop(stop, &mut AtStop { p: &mut *p, tid });
    let Some((by, count)) = step.filter(|_| stop.hold) else {
        return wrote;
    };
    match by {
        StepBy::Sys => {
            if let Some(obs) = p.observer() {
                obs.step_to = Some(obs.syscalls.saturating_add(count));
            }
        }
        StepBy::Insn => {
            let Some(thread) = p.core().tasks.get(&tid).map(|t| t.cpu.retired()) else {
                return wrote;
            };
            if let Some(obs) = p.observer() {
                obs.step_insn = Some((tid, thread.saturating_add(count)));
                obs.recompute();
            }
        }
        StepBy::Line => {
            let Some((thread, pc)) = p
                .core()
                .tasks
                .get(&tid)
                .map(|t| (t.cpu.retired(), t.cpu.pc()))
            else {
                return wrote;
            };
            let from = p.source_line(tid, pc).map(|(f, l, _)| (f, l));
            if let Some(obs) = p.observer() {
                obs.step_line = Some(StepLine {
                    tid,
                    from,
                    left: count,
                });
                obs.step_insn = Some((tid, thread.saturating_add(1)));
                obs.recompute();
            }
        }
    }
    wrote
}

/// A personality at a stop in task `tid`, as the host's debugger sees it.
struct AtStop<'a, P> {
    p: &'a mut P,
    tid: Tid,
}

impl<P: Personality> StopTarget for AtStop<'_, P> {
    fn tid(&self) -> Tid {
        self.tid
    }
    fn personality(&mut self) -> Option<&mut dyn core::any::Any> {
        self.p.as_any()
    }
}
