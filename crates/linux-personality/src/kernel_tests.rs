//! Linux's waits on the substrate: the wake order and the trace's spelling of blockers are
//! pinned here, because replays and trace comparisons depend on them. Nothing orders wait keys (no map
//! or sort is keyed by them); determinism lives in the wake order: restartable waiters in tid
//! order, futex waiters in arrival order.

use super::*;
use wasmshim_cpu::{A64Context, CpuContext, Event, GuestMemory};

#[derive(Clone)]
struct Idle(u64);

impl wasmshim_cpu::GuestCpu for Idle {
    fn run_until_event(&mut self, _mem: &mut dyn GuestMemory, _budget: u64) -> Event {
        Event::Yield
    }
    fn set_syscall_result(&mut self, value: u64) {
        self.0 = value;
    }
    fn pc(&self) -> u64 {
        0
    }
    fn set_pc(&mut self, _pc: u64) {}
    fn sp(&self) -> u64 {
        0
    }
    fn set_sp(&mut self, _sp: u64) {}
    fn thread_pointer(&self) -> u64 {
        0
    }
    fn set_thread_pointer(&mut self, _tp: u64) {}
    fn retired(&self) -> u64 {
        0
    }
    fn set_virtual_time(&mut self, _ns: u64) {}
    fn context(&self) -> CpuContext {
        CpuContext::A64(A64Context::default())
    }
    fn set_context(&mut self, _ctx: &CpuContext) {}
    fn duplicate(&self) -> Box<dyn wasmshim_cpu::GuestCpu> {
        Box::new(self.clone())
    }
}

fn core_with(blockers: &[(Tid, Blocker)]) -> wasmshim_kernel::Core<Linux> {
    let mut core = wasmshim_kernel::Core::<Linux>::new(wasmshim_kernel::SchedConfig {
        isa: wasmshim_kernel::Isa::Aarch64,
        quantum: 100_000,
        ns_per_insn: 1,
        trace: false,
    });
    for &(tid, b) in blockers {
        let mut t = Thread::new(tid, Box::new(Idle(0)), 0, LinuxThread::new(1, 0));
        t.state = ThreadState::Blocked(b);
        core.tasks.insert(tid, t);
    }
    core
}

fn restart(key: WaitKey) -> Blocker {
    Blocker::Restart {
        key,
        deadline: None,
        resume_pc: 0x40_0000,
        timeout_result: None,
    }
}

fn runnable(core: &wasmshim_kernel::Core<Linux>) -> Vec<Tid> {
    core.tasks
        .values()
        .filter(|t| t.state == ThreadState::Runnable)
        .map(|t| t.tid)
        .collect()
}

#[test]
fn wakes_follow_keys_and_tid_order() {
    let children = WaitKey::Personal(LinuxKey::Children(1));
    let signals = WaitKey::Personal(LinuxKey::Signals(1));
    let mut core = core_with(&[
        (5, restart(WaitKey::Object(7))),
        (2, restart(WaitKey::AnyIo)),
        (3, restart(children)),
        (4, restart(signals)),
        (6, Blocker::Personal(Park::Vfork { child: 9 })),
        (7, Blocker::Personal(Park::SigSuspend)),
    ]);
    // An object change wakes its waiters and every poller; nothing else.
    core.wake(WaitKey::Object(7), usize::MAX, u32::MAX);
    assert_eq!(runnable(&core), [2, 5]);
    // Children and signals are distinct keys of the same process.
    core.wake(children, usize::MAX, u32::MAX);
    assert_eq!(runnable(&core), [2, 3, 5]);
    core.wake(signals, usize::MAX, u32::MAX);
    assert_eq!(runnable(&core), [2, 3, 4, 5]);
    // Parked threads are woken only explicitly (vfork's child, a handled signal).
    core.wake(WaitKey::AnyIo, usize::MAX, u32::MAX);
    assert_eq!(runnable(&core), [2, 3, 4, 5]);
}

#[test]
fn futex_waiters_wake_in_arrival_order() {
    let key = WaitKey::Futex(1, 0x1000);
    let futex = |seq| Blocker::Futex {
        key,
        bitset: u32::MAX,
        deadline: None,
        seq,
    };
    let mut core = core_with(&[(2, futex(7)), (3, futex(5)), (4, futex(6))]);
    assert_eq!(core.wake(key, 2, u32::MAX), 2);
    assert_eq!(runnable(&core), [3, 4]);
}

#[test]
fn blockers_trace_spelling() {
    // The trace format of blockers: the spellings of the derived `Debug` of `Blocker` and
    // `WaitKey`.
    let cases: [(Blocker, &str); 6] = [
        (
            restart(WaitKey::Personal(LinuxKey::Children(3))),
            "Restart { key: Children(3), deadline: None, resume_pc: 4194304, timeout_result: None }",
        ),
        (
            restart(WaitKey::Personal(LinuxKey::Signals(3))),
            "Restart { key: Signals(3), deadline: None, resume_pc: 4194304, timeout_result: None }",
        ),
        (
            restart(WaitKey::AnyIo),
            "Restart { key: AnyIo, deadline: None, resume_pc: 4194304, timeout_result: None }",
        ),
        (
            Blocker::Personal(Park::Vfork { child: 9 }),
            "Vfork { child: 9 }",
        ),
        (Blocker::Personal(Park::SigSuspend), "SigSuspend"),
        (
            Blocker::Sleep { until: 5, rem: 0 },
            "Sleep { until: 5, rem: 0 }",
        ),
    ];
    for (b, want) in cases {
        assert_eq!(format!("{b:?}"), want);
    }
}
