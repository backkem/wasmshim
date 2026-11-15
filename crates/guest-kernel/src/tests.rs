//! Unit tests: wake order, the trace format of blockers, deadlines and a scheduler run over a
//! scripted CPU.

use alloc::boxed::Box;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;

use wasmshim_cpu::{CpuContext, Event, Fault, GuestCpu, GuestMemory, PagedMemory, SyscallFrame};

use crate::*;

/// A CPU that plays a script: each step retires `n` instructions, then yields an event.
#[derive(Clone)]
struct Script {
    steps: Vec<(u64, Option<u64>)>,
    retired: u64,
    pc: u64,
    result: u64,
}

impl Script {
    fn new(steps: Vec<(u64, Option<u64>)>) -> Self {
        Self {
            steps,
            retired: 0,
            pc: 0x1000,
            result: 0,
        }
    }
}

impl GuestCpu for Script {
    fn run_until_event(&mut self, _mem: &mut dyn GuestMemory, budget: u64) -> Event {
        let Some(&(n, sys)) = self.steps.first() else {
            // Out of script: spin.
            self.retired += budget;
            return Event::Yield;
        };
        if n > budget {
            self.steps[0].0 -= budget;
            self.retired += budget;
            return Event::Yield;
        }
        self.steps.remove(0);
        self.retired += n;
        match sys {
            Some(number) => {
                self.pc += 4;
                Event::Syscall(SyscallFrame {
                    number,
                    pc: self.pc,
                    args: [self.pc, 0, 0, 0, 0, 0],
                })
            }
            None => Event::Yield,
        }
    }
    fn set_syscall_result(&mut self, value: u64) {
        self.result = value;
    }
    fn pc(&self) -> u64 {
        self.pc
    }
    fn set_pc(&mut self, pc: u64) {
        self.pc = pc;
    }
    fn sp(&self) -> u64 {
        0
    }
    fn set_sp(&mut self, _sp: u64) {}
    fn thread_pointer(&self) -> u64 {
        0
    }
    fn set_thread_pointer(&mut self, _tp: u64) {}
    fn retired(&self) -> u64 {
        self.retired
    }
    fn set_virtual_time(&mut self, _ns: u64) {}
    fn context(&self) -> CpuContext {
        CpuContext::A64(Default::default())
    }
    fn set_context(&mut self, _ctx: &CpuContext) {}
    fn duplicate(&self) -> Box<dyn GuestCpu> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Key {
    Children(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Park {
    Vfork { child: u32 },
}

struct T;
impl Types for T {
    type Task = u32; // a "pid"
    type Key = Key;
    type Park = Park;
}

fn cfg() -> SchedConfig {
    SchedConfig {
        isa: Isa::Aarch64,
        quantum: 100,
        ns_per_insn: 1,
        trace: true,
    }
}

fn core_with(n: u32) -> Core<T> {
    let mut c = Core::<T>::new(cfg());
    for tid in 1..=n {
        c.add_task(Task::new(tid, Box::new(Script::new(vec![])), 0, tid));
    }
    c
}

fn block(c: &mut Core<T>, tid: Tid, b: BlockerOf<T>) {
    c.tasks.get_mut(&tid).unwrap().state = TaskState::Blocked(b);
}

#[test]
fn restart_waiters_wake_in_tid_order_and_anyio_matches_objects() {
    let mut c = core_with(4);
    c.run_queue.clear();
    let restart = |key| Blocker::Restart {
        key,
        deadline: None,
        resume_pc: 0,
        timeout_result: None,
    };
    block(&mut c, 3, restart(WaitKey::Object(7)));
    block(&mut c, 1, restart(WaitKey::AnyIo));
    block(&mut c, 2, restart(WaitKey::Object(8)));
    block(&mut c, 4, restart(WaitKey::Object(7)));
    assert_eq!(c.wake(WaitKey::Object(7), usize::MAX, u32::MAX), 0);
    assert_eq!(c.run_queue.iter().copied().collect::<Vec<_>>(), [1, 3, 4]);
    assert!(matches!(c.tasks[&2].state, TaskState::Blocked(_)));
}

#[test]
fn futex_waiters_wake_in_arrival_order_up_to_the_limit() {
    let mut c = core_with(3);
    c.run_queue.clear();
    let key = WaitKey::Futex(1, 0x40);
    for (tid, seq) in [(1, 5), (2, 3), (3, 4)] {
        block(
            &mut c,
            tid,
            Blocker::Futex {
                key,
                bitset: u32::MAX,
                deadline: None,
                seq,
            },
        );
    }
    assert_eq!(c.wake(key, 2, u32::MAX), 2);
    assert_eq!(c.run_queue.iter().copied().collect::<Vec<_>>(), [2, 3]);
    assert_eq!(c.requeue(key, WaitKey::Futex(1, 0x80), 9), 1);
    assert_eq!(c.wake(WaitKey::Futex(1, 0x80), 1, u32::MAX), 1);
    assert_eq!(c.run_queue.back(), Some(&1));
}

#[test]
fn personal_keys_and_blockers_print_as_themselves() {
    // The derived formats of the enums this replaced (trace lines must not change).
    let b: BlockerOf<T> = Blocker::Restart {
        key: WaitKey::Personal(Key::Children(3)),
        deadline: Some(5),
        resume_pc: 4100,
        timeout_result: None,
    };
    assert_eq!(
        format!("{b:?}"),
        "Restart { key: Children(3), deadline: Some(5), resume_pc: 4100, timeout_result: None }"
    );
    let b: BlockerOf<T> = Blocker::Personal(Park::Vfork { child: 9 });
    assert_eq!(format!("{b:?}"), "Vfork { child: 9 }");
    let b: BlockerOf<T> = Blocker::Futex {
        key: WaitKey::Futex(1, 64),
        bitset: 1,
        deadline: None,
        seq: 2,
    };
    assert_eq!(
        format!("{b:?}"),
        "Futex { key: Futex(1, 64), bitset: 1, deadline: None, seq: 2 }"
    );
    let b: BlockerOf<T> = Blocker::Sleep { until: 1, rem: 0 };
    assert_eq!(format!("{b:?}"), "Sleep { until: 1, rem: 0 }");
    let k: KeyOf<T> = WaitKey::AnyIo;
    assert_eq!(format!("{k:?}"), "AnyIo");
    let k: KeyOf<T> = WaitKey::Object(4);
    assert_eq!(format!("{k:?}"), "Object(4)");
}

#[test]
fn deadlines_expire_with_their_results() {
    let mut c = core_with(3);
    c.run_queue.clear();
    block(&mut c, 1, Blocker::Sleep { until: 10, rem: 0 });
    block(
        &mut c,
        2,
        Blocker::Restart {
            key: WaitKey::AnyIo,
            deadline: Some(10),
            resume_pc: 0x2000,
            timeout_result: Some(0),
        },
    );
    block(&mut c, 3, Blocker::Personal(Park::Vfork { child: 1 }));
    c.expire_deadlines(9, 99);
    assert!(c.run_queue.is_empty());
    c.expire_deadlines(10, 99);
    assert_eq!(c.run_queue.iter().copied().collect::<Vec<_>>(), [1, 2]);
    assert_eq!(c.tasks[&2].cpu.pc(), 0x2000);
}

#[test]
fn restart_deadlines_survive_restarts_of_one_call() {
    let mut c = core_with(1);
    c.current_pc = 0x44;
    assert_eq!(c.restart_deadline(1, Some(100), 5), Some(105));
    assert_eq!(c.restart_deadline(1, Some(100), 50), Some(105));
    c.current_pc = 0x48;
    assert_eq!(c.restart_deadline(1, Some(100), 50), Some(150));
    assert_eq!(c.restart_deadline(1, None, 50), None);
}

/// A minimal personality: syscall 1 sleeps until `now + 50`, syscall 0 exits.
struct Mini {
    core: Core<T>,
    space: Rc<RefCell<Mem>>,
    exited: bool,
    /// Observed when present.
    obs: Option<observe::Observer>,
    /// What a debugger sees of the personality's own state.
    state: u32,
}

struct Mem(PagedMemory);
impl Space for Mem {
    fn memory(&mut self) -> &mut dyn GuestMemory {
        &mut self.0
    }
}

struct NullHost {
    now: u64,
    diags: Vec<String>,
}
impl Host for NullHost {
    fn monotonic_ns(&mut self) -> u64 {
        self.now
    }
    fn realtime_ns(&mut self) -> u64 {
        self.now
    }
    fn random(&mut self, buf: &mut [u8]) {
        buf.fill(0);
    }
    fn stdout(&mut self, _: &[u8]) {}
    fn stderr(&mut self, _: &[u8]) {}
    fn stdin(&mut self, _: &mut [u8]) -> usize {
        0
    }
    fn diag(&mut self, line: &str) {
        self.diags.push(line.into());
    }
}

impl Personality for Mini {
    type T = T;
    type Space = Mem;
    fn core(&self) -> &Core<T> {
        &self.core
    }
    fn core_mut(&mut self) -> &mut Core<T> {
        &mut self.core
    }
    fn finished(&self) -> bool {
        self.exited
    }
    fn space(&self, _tid: Tid) -> Option<Rc<RefCell<Mem>>> {
        Some(self.space.clone())
    }
    fn syscall(&mut self, _tid: Tid, frame: SyscallFrame, _host: &mut dyn Host) -> FlowOf<T> {
        match frame.number {
            0 => Flow::ExitGroup(0),
            1 => Flow::Block(Blocker::Sleep {
                until: self.core.slice_now + 50,
                rem: 0,
            }),
            _ => Flow::Ret(u64::MAX - 37),
        }
    }
    fn fault(&mut self, _tid: Tid, _fault: Fault, _host: &mut dyn Host) {}
    fn exit_thread(&mut self, tid: Tid, _code: i32) {
        self.core.remove_task(tid);
    }
    fn exit_group(&mut self, tid: Tid, _code: i32) {
        self.core.remove_task(tid);
        self.exited = true;
    }
    fn futex_timeout(&self) -> u64 {
        u64::MAX - 109
    }
    fn describe(&self, number: u64) -> Option<(&'static str, usize)> {
        match number {
            0 => Some(("exit", 1)),
            1 => Some(("sleep", 0)),
            _ => None,
        }
    }
    fn show_error(&self, ret: u64) -> Option<String> {
        (ret > u64::MAX - 4096).then(|| format!("-1 E{}", ret.wrapping_neg()))
    }
    fn observer(&mut self) -> Option<&mut observe::Observer> {
        self.obs.as_mut()
    }
    fn as_any(&mut self) -> Option<&mut dyn core::any::Any> {
        Some(&mut self.state)
    }
}

/// A host whose debugger plays a script of answers per stop and records the stops.
struct Monitor {
    host: NullHost,
    stops: Vec<debug::Stop>,
    /// Per stop, the answer (`Resume::default()` past the end).
    plan: Vec<debug::Resume>,
}

impl Host for Monitor {
    fn monotonic_ns(&mut self) -> u64 {
        self.host.now
    }
    fn realtime_ns(&mut self) -> u64 {
        self.host.now
    }
    fn random(&mut self, buf: &mut [u8]) {
        buf.fill(0);
    }
    fn stdout(&mut self, _: &[u8]) {}
    fn stderr(&mut self, _: &[u8]) {}
    fn stdin(&mut self, _: &mut [u8]) -> usize {
        0
    }
    fn diag(&mut self, line: &str) {
        self.host.diags.push(line.into());
    }
    fn stop(&mut self, stop: &debug::Stop, at: &mut dyn debug::StopTarget) -> debug::Resume {
        // The debugger reaches the personality itself.
        assert!(
            at.personality()
                .is_some_and(|p| p.downcast_mut::<u32>() == Some(&mut 7))
        );
        self.stops.push(stop.clone());
        self.plan
            .get(self.stops.len() - 1)
            .copied()
            .unwrap_or_default()
    }
}

fn observed_run(probes: &[debug::ProbeSpec], plan: Vec<debug::Resume>) -> (Mini, Monitor) {
    let cpu = Script::new(vec![(30, Some(1)), (250, Some(7)), (10, Some(0))]);
    observed_cpu(Box::new(cpu), probes, plan)
}

fn observed_cpu(
    cpu: Box<dyn GuestCpu>,
    probes: &[debug::ProbeSpec],
    plan: Vec<debug::Resume>,
) -> (Mini, Monitor) {
    let mut p = Mini {
        core: Core::new(cfg()),
        space: Rc::new(RefCell::new(Mem(PagedMemory::new()))),
        exited: false,
        obs: Some(observe::Observer::default()),
        state: 7,
    };
    p.obs.as_mut().unwrap().arm(probes);
    p.core.add_task(Task::new(1, cpu, 0, 1));
    let mut host = Monitor {
        host: NullHost {
            now: 0,
            diags: Vec::new(),
        },
        stops: Vec::new(),
        plan,
    };
    loop {
        match run(&mut p, &mut host) {
            Status::Exited => break,
            Status::Wait { until } => host.host.now = until.expect("never stuck"),
        }
    }
    (p, host)
}

fn probe(id: u32, point: observe::Point) -> debug::ProbeSpec {
    debug::ProbeSpec {
        id,
        point,
        hits: None,
        hold: true,
        info: true,
    }
}

fn syscall(number: u64, exit: bool) -> observe::Point {
    observe::Point::Syscall {
        number: Some(number),
        exit,
        tid: None,
        args: [None; 6],
        ret_ge: None,
    }
}

fn step(by: debug::StepBy) -> debug::Resume {
    debug::Resume {
        wrote: false,
        step: Some((by, 1)),
    }
}

#[test]
fn syscall_stops_at_entry_and_exit_step_and_change_nothing() {
    use debug::Position::Sys;
    let (plain, plain_host) = observed_run(&[], Vec::new());
    let (p, host) = observed_run(
        &[probe(1, syscall(1, false)), probe(2, syscall(7, true))],
        // At sleep's entry: step one syscall (the entry of syscall 7).
        vec![step(debug::StepBy::Sys)],
    );
    let at: Vec<(u32, debug::Position)> =
        host.stops.iter().map(|s| (s.probe, s.position)).collect();
    assert_eq!(
        at,
        [
            (1, Sys { n: 1, exit: false }),
            (0, Sys { n: 2, exit: false }),
            (2, Sys { n: 2, exit: true })
        ]
    );
    // The substrate's facts: the syscall's ordinal, number, thread, arguments and (at exit)
    // result.
    let Some(debug::StopInfo::Syscall {
        sys, number, ret, ..
    }) = &host.stops[2].info
    else {
        panic!("{:?}", host.stops[2]);
    };
    assert_eq!((*sys, *number, *ret), (2, 7, Some(u64::MAX - 37)));
    // Stops change nothing: the same trace, retired count and syscall count.
    assert_eq!(host.host.diags, plain_host.host.diags);
    assert_eq!(p.core.retired(), plain.core.retired());
    assert_eq!(p.obs.as_ref().unwrap().syscalls(), 3);
    // A probe that does not hold reports its hit and does not step.
    let mut action = probe(1, observe::Point::Sys { n: 3, exit: false });
    action.hold = false;
    action.info = false;
    let (_, host) = observed_run(&[action], vec![step(debug::StepBy::Sys)]);
    assert_eq!(host.stops.len(), 1);
    assert!(!host.stops[0].hold && host.stops[0].info.is_none());
}
#[test]
fn a_run_charges_compute_sleeps_and_traces() {
    let mut p = Mini {
        core: Core::new(cfg()),
        space: Rc::new(RefCell::new(Mem(PagedMemory::new()))),
        exited: false,
        obs: None,
        state: 7,
    };
    // 30 instructions then sleep; 250 then an unknown syscall; 10 then exit.
    let cpu = Script::new(vec![(30, Some(1)), (250, Some(7)), (10, Some(0))]);
    p.core.add_task(Task::new(1, Box::new(cpu), 0, 1));
    let mut host = NullHost {
        now: 0,
        diags: Vec::new(),
    };
    let mut waits = Vec::new();
    loop {
        match run(&mut p, &mut host) {
            Status::Exited => break,
            Status::Wait { until } => {
                let until = until.expect("never stuck");
                waits.push(until);
                host.now = until;
            }
        }
    }
    // Sleep at 0 until 50 (30 ns of compute charged first), then two 100-instruction quanta
    // and the rest of the 250, then exit.
    assert_eq!(waits, [50, 150, 250]);
    assert_eq!(p.core.retired(), 290);
    assert_eq!(
        host.diags,
        [
            "[1] sleep() = ? <blocked: Sleep { until: 50, rem: 0 }>",
            "[1] syscall_7(0x1008, 0x0, 0x0, 0x0, 0x0, 0x0) = -1 E38",
            "[1] exit(0x100c) = ? <exit 0>",
        ]
    );
}

/// A straight-line CPU for instruction stops: instruction `i` is at `0x1000 + 4i`.
/// Instructions `LL..=LL+3` are a load-exclusive/store-exclusive sequence: the store at `LL+3`
/// fails if the monitor was cleared since the load at `LL`, and the sequence repeats (the
/// guest's retry loop), so a visible stop shows up as extra retired instructions.
#[derive(Clone)]
struct Linear {
    retired: u64,
    pc: u64,
    monitor: bool,
    retries: Rc<core::cell::Cell<u64>>,
}

const LL: u64 = 50;
const SYSCALL_AT: u64 = 120;
const EXIT_AT: u64 = 199;

impl Linear {
    fn new() -> (Self, Rc<core::cell::Cell<u64>>) {
        let retries = Rc::new(core::cell::Cell::new(0));
        (
            Self {
                retired: 0,
                pc: 0x1000,
                monitor: false,
                retries: retries.clone(),
            },
            retries,
        )
    }

    fn index(&self) -> u64 {
        (self.pc - 0x1000) / 4
    }

    /// Executes one instruction; an event if it was a syscall.
    fn step(&mut self) -> Option<Event> {
        let i = self.index();
        self.retired += 1;
        self.pc += 4;
        if i == LL {
            self.monitor = true;
        } else if i == LL + 3 {
            if !self.monitor {
                self.retries.set(self.retries.get() + 1);
                self.pc = 0x1000 + 4 * LL;
                return None;
            }
            self.monitor = false;
        }
        let number = match i {
            SYSCALL_AT => 7,
            EXIT_AT => 0,
            _ => return None,
        };
        Some(Event::Syscall(SyscallFrame {
            number,
            pc: self.pc,
            args: [0; 6],
        }))
    }
}

impl wasmshim_cpu::DebugCpu for Linear {
    fn run_debug(&mut self, _mem: &mut dyn GuestMemory, budget: u64, breaks: &[u64]) -> Event {
        let mut left = budget;
        let mut first = true;
        loop {
            if left == 0 || (!first && breaks.binary_search(&self.pc).is_ok()) {
                return Event::Yield;
            }
            first = false;
            left -= 1;
            if let Some(e) = self.step() {
                return e;
            }
        }
    }

    fn clear_monitor(&mut self) {
        self.monitor = false;
    }
}

impl GuestCpu for Linear {
    fn run_until_event(&mut self, mem: &mut dyn GuestMemory, budget: u64) -> Event {
        let e = wasmshim_cpu::DebugCpu::run_debug(self, mem, budget, &[]);
        self.monitor = false;
        e
    }
    fn set_syscall_result(&mut self, _value: u64) {}
    fn pc(&self) -> u64 {
        self.pc
    }
    fn set_pc(&mut self, pc: u64) {
        self.pc = pc;
    }
    fn sp(&self) -> u64 {
        0
    }
    fn set_sp(&mut self, _sp: u64) {}
    fn thread_pointer(&self) -> u64 {
        0
    }
    fn set_thread_pointer(&mut self, _tp: u64) {}
    fn retired(&self) -> u64 {
        self.retired
    }
    fn set_virtual_time(&mut self, _ns: u64) {}
    fn context(&self) -> CpuContext {
        CpuContext::A64(wasmshim_cpu::A64Context {
            pc: self.pc,
            ..Default::default()
        })
    }
    fn set_context(&mut self, _ctx: &CpuContext) {}
    fn duplicate(&self) -> Box<dyn GuestCpu> {
        Box::new(self.clone())
    }
    fn debug(&mut self) -> Option<&mut dyn wasmshim_cpu::DebugCpu> {
        Some(self)
    }
}

#[test]
fn instruction_stops_are_exact_and_invisible() {
    use debug::Position::Insn;
    let (cpu, plain_retries) = Linear::new();
    let (plain, plain_host) = observed_cpu(Box::new(cpu), &[], Vec::new());
    assert_eq!(plain_retries.get(), 0);

    let (cpu, retries) = Linear::new();
    let (p, host) = observed_cpu(
        Box::new(cpu),
        &[
            // Between the load- and the store-exclusive.
            probe(
                1,
                observe::Point::Insn {
                    n: LL + 2,
                    every: None,
                },
            ),
            // At the end of the first quantum (100 instructions).
            probe(
                2,
                observe::Point::Pc {
                    addr: 0x1000 + 4 * 100,
                    pid: None,
                    tid: None,
                },
            ),
        ],
        vec![step(debug::StepBy::Insn)],
    );
    let at: Vec<(u32, debug::Position)> =
        host.stops.iter().map(|s| (s.probe, s.position)).collect();
    assert_eq!(at, [(1, Insn(52)), (0, Insn(53)), (2, Insn(100))]);
    let why = |i: usize| match &host.stops[i].info {
        Some(debug::StopInfo::Insn { pc, why, .. }) => (*pc, *why),
        other => panic!("{other:?}"),
    };
    assert_eq!(why(2), (0x1000 + 4 * 100, debug::Why::Pc));
    assert_eq!(why(1).1, debug::Why::Step);
    // Invisible: no retry of the exclusive sequence, the same instructions, trace and time.
    assert_eq!(retries.get(), 0);
    assert_eq!(p.core.retired(), plain.core.retired());
    assert_eq!(p.core.retired(), EXIT_AT + 1);
    assert_eq!(host.host.diags, plain_host.host.diags);
    assert_eq!(host.host.now, plain_host.host.now);

    // The control: the same cut without the debugger interface clears the monitor, and the
    // guest retries.
    let (mut cpu, retries) = Linear::new();
    let mut mem = PagedMemory::new();
    cpu.run_until_event(&mut mem, LL + 2);
    cpu.run_until_event(&mut mem, 10);
    assert_eq!(retries.get(), 1);
}
mod checkpoints {
    use super::*;
    use crate::checkpoint::{Ckpt, CkptError, Reader, Writer, read_header, write_header};

    impl Ckpt for Key {
        fn save(&self, w: &mut Writer) {
            let Key::Children(p) = self;
            p.save(w);
        }
        fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
            Ok(Key::Children(u32::load(r)?))
        }
    }

    impl Ckpt for Park {
        fn save(&self, w: &mut Writer) {
            let Park::Vfork { child } = self;
            child.save(w);
        }
        fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
            Ok(Park::Vfork {
                child: u32::load(r)?,
            })
        }
    }

    fn save(c: &Core<T>) -> Vec<u8> {
        let mut w = Writer::new();
        write_header(&mut w);
        c.save(&mut w);
        w.into_bytes()
    }

    fn load(bytes: &[u8], cpu_retired: u64) -> Result<Core<T>, CkptError> {
        let mut r = Reader::new(bytes);
        read_header(&mut r)?;
        let mut new_cpu = |_: &CpuContext| -> Box<dyn GuestCpu> {
            let mut s = Script::new(vec![]);
            s.retired = cpu_retired;
            Box::new(s)
        };
        let c = Core::<T>::load(cfg(), &mut r, &mut new_cpu)?;
        if !r.done() {
            return Err(CkptError::new("trailing bytes"));
        }
        Ok(c)
    }

    fn sample() -> Core<T> {
        let mut c = core_with(3);
        c.run_queue.clear();
        c.run_queue.push_back(3);
        block(&mut c, 1, Blocker::Personal(Park::Vfork { child: 3 }));
        block(
            &mut c,
            2,
            Blocker::Futex {
                key: WaitKey::Personal(Key::Children(9)),
                bitset: 5,
                deadline: Some(77),
                seq: 2,
            },
        );
        c.tasks.get_mut(&2).unwrap().timed = Some((0x1004, 500));
        let t3 = c.tasks.get_mut(&3).unwrap();
        t3.last_syscall = Some(63);
        let mut cpu = Script::new(vec![]);
        cpu.retired = 12_345;
        t3.cpu = Box::new(cpu);
        t3.retired_base = 345;
        c.retired_dead = 99;
        c.next_wait_seq = 3;
        let _ = c.new_object_id();
        let _ = c.new_space_id();
        c.diag(String::from("before the checkpoint"));
        c.slice_now = 1_000;
        c.current = 3;
        c.current_pc = 0x1008;
        c
    }

    #[test]
    fn the_substrate_round_trips_byte_identically() {
        let c = sample();
        let bytes = save(&c);
        // A new CPU's own count is irrelevant: the tasks keep theirs (CPU-time clocks).
        for cpu_retired in [0, 7, 1 << 40] {
            let d = load(&bytes, cpu_retired).unwrap();
            assert_eq!(save(&d), bytes);
            assert_eq!(d.retired(), c.retired());
            assert_eq!(d.tasks[&3].own_retired(), 12_000);
            assert_eq!(d.tasks[&2].state, c.tasks[&2].state);
            assert_eq!(d.run_queue, c.run_queue);
        }
    }

    #[test]
    fn a_damaged_checkpoint_is_an_error() {
        let bytes = save(&sample());
        for n in 0..bytes.len() {
            assert!(load(&bytes[..n], 0).is_err(), "prefix of {n} bytes");
        }
        let mut other = cfg();
        other.quantum += 1;
        let mut r = Reader::new(&bytes);
        read_header(&mut r).unwrap();
        let mut new_cpu = |_: &CpuContext| -> Box<dyn GuestCpu> { Box::new(Script::new(vec![])) };
        assert!(Core::<T>::load(other, &mut r, &mut new_cpu).is_err());
    }

    #[test]
    fn maps_and_sets_load_only_in_their_saved_order() {
        use alloc::collections::{BTreeMap, BTreeSet};
        let bytes_of = |v: &dyn Fn(&mut Writer)| {
            let mut w = Writer::new();
            v(&mut w);
            w.into_bytes()
        };
        let map = BTreeMap::from([(1u32, 10u8), (2, 20), (7, 70)]);
        let set = BTreeSet::from([3u64, 4, 9]);
        // What `save` writes loads back to the same value.
        let b = bytes_of(&|w| map.save(w));
        assert_eq!(BTreeMap::<u32, u8>::load(&mut Reader::new(&b)), Ok(map));
        let b = bytes_of(&|w| set.save(w));
        assert_eq!(BTreeSet::<u64>::load(&mut Reader::new(&b)), Ok(set));
        // The same entries unsorted or repeated (written as a vector, the same layout) do not.
        for pairs in [vec![(2u32, 20u8), (1, 10)], vec![(1, 10), (1, 11)]] {
            let b = bytes_of(&|w| pairs.save(w));
            assert!(
                BTreeMap::<u32, u8>::load(&mut Reader::new(&b)).is_err(),
                "{pairs:?}"
            );
        }
        for keys in [vec![9u64, 3], vec![3, 3]] {
            let b = bytes_of(&|w| keys.save(w));
            assert!(
                BTreeSet::<u64>::load(&mut Reader::new(&b)).is_err(),
                "{keys:?}"
            );
        }
        // Tasks are saved in tid order too. Cut each task's record out of a checkpoint of it
        // alone, then splice two records into one with no tasks.
        let only = |tids: &[u32]| {
            let mut c = core_with(2);
            c.run_queue.clear();
            c.tasks.retain(|t, _| tids.contains(t));
            save(&c)
        };
        let none = only(&[]);
        // The task count is where the checkpoints first differ (1 against 0).
        let at = (0..none.len()).find(|&i| only(&[1])[i] != none[i]).unwrap();
        let record = |tid: u32| {
            let one = only(&[tid]);
            one[at + 8..one.len() - (none.len() - at - 8)].to_vec()
        };
        let splice = |a: u32, b: u32| {
            let mut bytes = none[..at].to_vec();
            bytes.extend_from_slice(&2u64.to_le_bytes());
            bytes.extend(record(a));
            bytes.extend(record(b));
            bytes.extend_from_slice(&none[at + 8..]);
            bytes
        };
        assert_eq!(
            splice(1, 2),
            only(&[1, 2]),
            "the splice is what save writes"
        );
        assert!(load(&splice(1, 2), 0).is_ok());
        assert!(load(&splice(2, 1), 0).is_err(), "out of order");
        assert!(load(&splice(1, 1), 0).is_err(), "repeated");
    }

    mod spaces {
        use alloc::collections::BTreeMap;

        use wasmshim_cpu::{PAGE_SIZE, PageRef, PageSource, PageSourceError, PageView, Perms};

        use super::*;
        use crate::checkpoint::Baseline;
        use crate::space::{AddressSpace, MapKind, Vma};

        const P: u64 = PAGE_SIZE;

        /// A mapping kind with a file-like offset (in pages).
        #[derive(Clone, Debug, PartialEq, Eq)]
        enum Kind {
            Anon,
            File(u64),
        }

        impl MapKind for Kind {
            fn advance(&mut self, by: u64) {
                if let Kind::File(i) = self {
                    *i += by / P;
                }
            }
            fn continues(&self, len: u64, next: &Self) -> bool {
                match (self, next) {
                    (Kind::File(a), Kind::File(b)) => a + len / P == *b,
                    (a, b) => a == b,
                }
            }
        }

        impl Ckpt for Kind {
            fn save(&self, w: &mut Writer) {
                match self {
                    Kind::Anon => 0u8.save(w),
                    Kind::File(i) => {
                        1u8.save(w);
                        i.save(w);
                    }
                }
            }
            fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
                match u8::load(r)? {
                    0 => Ok(Kind::Anon),
                    1 => Ok(Kind::File(u64::load(r)?)),
                    _ => Err(CkptError::new("bad kind")),
                }
            }
        }

        /// A "file": page `i` is all `i | 1`.
        struct File;
        impl PageSource for File {
            fn page(&self, i: u64) -> Result<PageRef, PageSourceError> {
                Ok(Rc::new([(i as u8) | 1; PAGE_SIZE as usize]))
            }
        }

        fn file() -> Rc<dyn PageSource> {
            Rc::new(File)
        }

        /// Every mapped page: its permissions and bytes (`None`: zeros; shared: not here).
        fn contents(s: &AddressSpace<Kind>) -> BTreeMap<u64, (Perms, Option<Vec<u8>>)> {
            let mut out = BTreeMap::new();
            s.mem
                .for_each_mapped(&mut |addr, n, perms, view| {
                    for k in 0..n {
                        let bytes = match view {
                            PageView::Zero => None,
                            PageView::Data(d) => Some(d.to_vec()),
                            PageView::Source(src, i) => Some(src.page(i).unwrap().to_vec()),
                            PageView::Shared(_) => unreachable!(),
                        };
                        let bytes = bytes.filter(|b| b.iter().any(|&x| x != 0));
                        out.insert(addr + k * P, (perms, bytes));
                    }
                })
                .unwrap();
            out
        }

        /// The record, then (in this test's own framing) the pages it gave out.
        fn save_space(s: &AddressSpace<Kind>, src: &Rc<dyn PageSource>) -> Vec<u8> {
            let mut w = Writer::new();
            let mut pages = Vec::new();
            s.save(
                &mut w,
                &mut |v: &Vma<Kind>| match v.kind {
                    Kind::Anon => Baseline::Zero,
                    Kind::File(i) => Baseline::Source(src.clone(), i),
                },
                &mut |addr, page| pages.push((addr, page.to_vec())),
            )
            .unwrap();
            let record = w.into_bytes();
            let mut w = Writer::new();
            w.bytes(&record);
            pages.save(&mut w);
            w.into_bytes()
        }

        fn load_space(
            bytes: &[u8],
            src: &Rc<dyn PageSource>,
        ) -> Result<AddressSpace<Kind>, CkptError> {
            let mut outer = Reader::new(bytes);
            let record = outer.bytes()?;
            let pages: Vec<(u64, Vec<u8>)> = Ckpt::load(&mut outer)?;
            if !outer.done() {
                return Err(CkptError::new("trailing bytes"));
            }
            let pages: Vec<(u64, &[u8])> = pages.iter().map(|(a, p)| (*a, &p[..])).collect();
            let mut r = Reader::new(record);
            let s = AddressSpace::<Kind>::load(
                &mut r,
                Box::new(PagedMemory::new()),
                &mut |mem: &mut dyn GuestMemory, v: &Vma<Kind>| {
                    let len = v.end - v.start;
                    match v.kind {
                        Kind::Anon => mem.map(v.start, len, v.perms),
                        Kind::File(i) => mem.map_source(v.start, len, v.perms, src.clone(), i),
                    }
                    .map_err(|_| CkptError::new("replay"))
                },
                &pages,
            )?;
            if !r.done() {
                return Err(CkptError::new("trailing bytes"));
            }
            Ok(s)
        }

        #[test]
        fn address_spaces_round_trip_with_only_the_pages_a_replay_cannot_give() {
            let src = file();
            let mut s = AddressSpace::<Kind>::new(7, Box::new(PagedMemory::new()));
            s.map(0x10000, 0x10000 + 4 * P, Perms::RW, Kind::Anon)
                .unwrap();
            s.mem.write_u8(0x10000 + P + 3, 0xaa).unwrap();
            // A page written back to zeros, and one only read: neither is stored.
            s.mem.write_u8(0x10000 + 2 * P, 1).unwrap();
            s.mem.write_u8(0x10000 + 2 * P, 0).unwrap();
            s.mem.read_u8(0x10000 + 3 * P).unwrap();
            // A private file mapping: page 0 read (equal to the file), page 2 written.
            s.map(0x20000, 0x20000 + 3 * P, Perms::RX, Kind::File(5))
                .unwrap();
            s.mem
                .map_source(0x20000, 3 * P, Perms::RX, src.clone(), 5)
                .unwrap();
            s.mem.read_u8(0x20000).unwrap();
            s.mem.poke(0x20000 + 2 * P, &[0x55]).unwrap();
            // mprotect splits the file mapping (and does not merge it back).
            s.protect(0x20000 + P, 0x20000 + 2 * P, Perms::R).unwrap();
            // Memory permissions that differ from the mapping's (pages past an object).
            s.mem.protect(0x10000 + 3 * P, P, Perms::NONE).unwrap();
            // A lazy reservation with one written page.
            s.map(0x1000_0000, 0x1000_0000 + 256 * P, Perms::RW, Kind::Anon)
                .unwrap();
            s.mem.write_u8(0x1000_0000 + 200 * P, 9).unwrap();

            let bytes = save_space(&s, &src);
            // Three pages stored: the two written anonymous pages and the written file page.
            let header = 8 + 8 + s.vmas().count() * (8 + 8 + 1 + 9);
            assert!(bytes.len() < header + 64 + 3 * (8 + PAGE_SIZE as usize) + 32);
            assert!(bytes.len() > 3 * PAGE_SIZE as usize);

            let d = load_space(&bytes, &src).unwrap();
            assert_eq!(d.id, 7);
            assert_eq!(d.vmas().collect::<Vec<_>>(), s.vmas().collect::<Vec<_>>());
            assert_eq!(d.vmas().count(), 5, "the split file mapping stays split");
            assert_eq!(contents(&d), contents(&s));
            assert_eq!(save_space(&d, &src), bytes);
        }

        #[test]
        fn a_damaged_address_space_is_an_error() {
            let src = file();
            let mut s = AddressSpace::<Kind>::new(1, Box::new(PagedMemory::new()));
            s.map(0x10000, 0x10000 + 2 * P, Perms::RW, Kind::Anon)
                .unwrap();
            s.mem.write_u8(0x10000, 1).unwrap();
            let bytes = save_space(&s, &src);
            for n in 0..bytes.len() {
                assert!(
                    load_space(&bytes[..n], &src).is_err(),
                    "prefix of {n} bytes"
                );
            }
        }
    }
}
