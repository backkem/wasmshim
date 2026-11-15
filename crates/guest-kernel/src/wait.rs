//! The one wait primitive: a task blocks on a key until a writer wakes the key or a deadline
//! passes. Every blocking call of every personality goes through it.
//!
//! * Most waits *restart*: the task's PC is rewound to its `SVC`, so after a wake-up the
//!   syscall simply runs again against the new state ([`Blocker::Restart`]).
//! * Futex-style waits complete with a value and wake in arrival order ([`Blocker::Futex`]).
//! * A personality may park a task on its own terms ([`Blocker::Personal`]): woken only
//!   explicitly, never by a deadline.
//!
//! Wake order is part of determinism: restartable waiters wake in task-id order, futex waiters
//! in arrival (sequence) order.

use alloc::vec::Vec;
use core::fmt;

use crate::task::{TaskState, Tid};
use crate::{BlockerOf, Core, PendingWait, Types};

/// What a task waits on.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WaitKey<K> {
    /// An object (pipe, eventfd, socket, ...) changed state; the id is the object's.
    Object(u64),
    /// Any object's readiness changed (`poll`/`select`).
    AnyIo,
    /// A futex word: (address-space id, address).
    Futex(u64, u64),
    /// The personality's own keys (Linux: children, signals).
    Personal(K),
}

/// `Personal` keys print as themselves, so traces show the personality's names.
impl<K: fmt::Debug> fmt::Debug for WaitKey<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Object(id) => f.debug_tuple("Object").field(id).finish(),
            Self::AnyIo => f.write_str("AnyIo"),
            Self::Futex(space, addr) => f.debug_tuple("Futex").field(space).field(addr).finish(),
            Self::Personal(k) => k.fmt(f),
        }
    }
}

/// Why a task is not runnable.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Blocker<K, P> {
    /// A timed sleep: completes with 0 at `until`; if the personality interrupts it, the time
    /// left goes to `rem` (unless 0).
    Sleep { until: u64, rem: u64 },
    /// Waiting for `key`; the PC points at the `SVC`, so a wake re-runs the syscall. At
    /// `deadline` the syscall instead returns `timeout_result` and resumes at `resume_pc`, or,
    /// if `timeout_result` is `None`, simply runs again (a timer the syscall waits for is due).
    Restart {
        key: WaitKey<K>,
        deadline: Option<u64>,
        resume_pc: u64,
        timeout_result: Option<u64>,
    },
    /// A futex wait: woken (0) by a wake on `key` with a matching bitset, or times out with
    /// the personality's timeout value.
    Futex {
        key: WaitKey<K>,
        bitset: u32,
        deadline: Option<u64>,
        seq: u64,
    },
    /// The personality's own parking (Linux: `vfork`, `rt_sigsuspend`).
    Personal(P),
}

/// `Personal` blockers print as themselves (trace lines are unchanged by the extraction).
impl<K: fmt::Debug, P: fmt::Debug> fmt::Debug for Blocker<K, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sleep { until, rem } => f
                .debug_struct("Sleep")
                .field("until", until)
                .field("rem", rem)
                .finish(),
            Self::Restart {
                key,
                deadline,
                resume_pc,
                timeout_result,
            } => f
                .debug_struct("Restart")
                .field("key", key)
                .field("deadline", deadline)
                .field("resume_pc", resume_pc)
                .field("timeout_result", timeout_result)
                .finish(),
            Self::Futex {
                key,
                bitset,
                deadline,
                seq,
            } => f
                .debug_struct("Futex")
                .field("key", key)
                .field("bitset", bitset)
                .field("deadline", deadline)
                .field("seq", seq)
                .finish(),
            Self::Personal(p) => p.fmt(f),
        }
    }
}

/// The deadlines of the objects a readiness-polling call watches ([`Core::wait_objects`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObjectDeadlines {
    wake_at: Option<u64>,
}

impl ObjectDeadlines {
    /// Adds an object's own deadline (when it becomes ready by itself), scanned at `now`.
    /// Deadlines at or before `now` are ignored: such an object is ready already (and
    /// reported, or not asked for), so waking for it would spin.
    pub fn watch(&mut self, deadline: Option<u64>, now: u64) {
        if let Some(d) = deadline.filter(|&d| d > now) {
            self.wake_at = Some(self.wake_at.map_or(d, |t| t.min(d)));
        }
    }

    /// The earliest watched deadline.
    pub fn earliest(&self) -> Option<u64> {
        self.wake_at
    }
}

impl<K, P> Blocker<K, P> {
    /// The monotonic instant at which the wait ends by itself, if any.
    pub fn deadline(&self) -> Option<u64> {
        match self {
            Self::Sleep { until, .. } => Some(*until),
            Self::Restart { deadline, .. } | Self::Futex { deadline, .. } => *deadline,
            Self::Personal(_) => None,
        }
    }
}

impl<T: Types> Core<T> {
    /// Makes a blocked task runnable.
    pub fn make_runnable(&mut self, tid: Tid) {
        if let Some(t) = self.tasks.get_mut(&tid)
            && t.state != TaskState::Runnable
        {
            t.state = TaskState::Runnable;
            self.run_queue.push_back(tid);
            let now = self.stats.now;
            self.stats.wake(tid, now);
        }
    }

    /// Wakes tasks waiting on `key`. Restartable waiters all wake (in tid order; an `AnyIo`
    /// waiter also wakes for any `Object` key); futex waiters wake in arrival order, at most
    /// `limit`, if their bitset intersects `bitset`. Returns the number of futex waiters woken.
    pub fn wake(&mut self, key: WaitKey<T::Key>, limit: usize, bitset: u32) -> usize {
        let mut restart = Vec::new();
        let mut futex: Vec<(u64, Tid)> = Vec::new();
        for t in self.tasks.values() {
            match t.state {
                TaskState::Blocked(Blocker::Restart { key: k, .. })
                    if k == key || (k == WaitKey::AnyIo && matches!(key, WaitKey::Object(_))) =>
                {
                    restart.push(t.tid);
                }
                TaskState::Blocked(Blocker::Futex {
                    key: k,
                    bitset: b,
                    seq,
                    ..
                }) if k == key && b & bitset != 0 => {
                    futex.push((seq, t.tid));
                }
                _ => {}
            }
        }
        for tid in restart {
            self.make_runnable(tid);
        }
        futex.sort_unstable();
        let woken = futex.len().min(limit);
        for &(_, tid) in &futex[..woken] {
            if let Some(t) = self.tasks.get_mut(&tid) {
                t.cpu.set_syscall_result(0);
            }
            self.make_runnable(tid);
        }
        woken
    }

    /// Moves up to `limit` futex waiters from `from` to `to` (`FUTEX_REQUEUE`).
    pub fn requeue(&mut self, from: WaitKey<T::Key>, to: WaitKey<T::Key>, limit: usize) -> usize {
        let mut waiters: Vec<(u64, Tid)> = self
            .tasks
            .values()
            .filter_map(|t| match t.state {
                TaskState::Blocked(Blocker::Futex { key, seq, .. }) if key == from => {
                    Some((seq, t.tid))
                }
                _ => None,
            })
            .collect();
        waiters.sort_unstable();
        waiters.truncate(limit);
        for &(_, tid) in &waiters {
            if let Some(t) = self.tasks.get_mut(&tid)
                && let TaskState::Blocked(Blocker::Futex { key, .. }) = &mut t.state
            {
                *key = to;
            }
        }
        waiters.len()
    }

    /// Ends every wait whose deadline has passed. `futex_timeout` is the raw value a timed-out
    /// futex wait returns (Linux: `-ETIMEDOUT`).
    pub fn expire_deadlines(&mut self, now: u64, futex_timeout: u64) {
        let due: Vec<(Tid, BlockerOf<T>)> = self
            .tasks
            .values()
            .filter_map(|t| match t.state {
                TaskState::Blocked(b) if b.deadline().is_some_and(|d| d <= now) => Some((t.tid, b)),
                _ => None,
            })
            .collect();
        for (tid, b) in due {
            if let Some(t) = self.tasks.get_mut(&tid) {
                match b {
                    Blocker::Sleep { .. } => t.cpu.set_syscall_result(0),
                    Blocker::Restart {
                        resume_pc,
                        timeout_result,
                        ..
                    } => {
                        if let Some(result) = timeout_result {
                            t.cpu.set_pc(resume_pc);
                            t.cpu.set_syscall_result(result);
                            t.timed = None;
                        }
                    }
                    Blocker::Futex { .. } => t.cpu.set_syscall_result(futex_timeout),
                    Blocker::Personal(_) => {}
                }
            }
            self.make_runnable(tid);
        }
    }

    /// Ends every syscall blocked (restartably) on `key` with the raw result `ret` instead of
    /// re-running it (an object removed under its waiters).
    pub fn fail_waiters(&mut self, key: WaitKey<T::Key>, ret: u64) {
        let hit: Vec<(Tid, u64)> = self
            .tasks
            .values()
            .filter_map(|t| match t.state {
                TaskState::Blocked(Blocker::Restart {
                    key: k, resume_pc, ..
                }) if k == key => Some((t.tid, resume_pc)),
                _ => None,
            })
            .collect();
        for (tid, resume_pc) in hit {
            if let Some(t) = self.tasks.get_mut(&tid) {
                t.cpu.set_pc(resume_pc);
                t.cpu.set_syscall_result(ret);
                t.timed = None;
            }
            self.make_runnable(tid);
        }
    }

    /// The absolute deadline of the current blocking syscall with relative `timeout` (ns):
    /// computed at its first entry and kept while it restarts after wake-ups, so spurious
    /// wake-ups never extend it. `None` without a timeout.
    pub fn restart_deadline(&mut self, tid: Tid, timeout: Option<u64>, now: u64) -> Option<u64> {
        let pc = self.current_pc;
        let t = self.tasks.get_mut(&tid)?;
        let Some(timeout) = timeout else {
            t.timed = None;
            return None;
        };
        match t.timed {
            Some((at, deadline)) if at == pc => Some(deadline),
            _ => {
                let deadline = now.saturating_add(timeout);
                t.timed = Some((pc, deadline));
                Some(deadline)
            }
        }
    }

    /// Arranges for the current syscall to block on `key` (and to return `timeout_result` at
    /// `deadline`). The personality's handler then reports "blocked" and turns the arrangement
    /// into a [`Blocker::Restart`] with [`Core::take_block`].
    pub fn wait_for(
        &mut self,
        key: WaitKey<T::Key>,
        deadline: Option<u64>,
        timeout_result: Option<u64>,
    ) {
        self.block_on = Some(PendingWait {
            key,
            deadline,
            timeout_result,
        });
    }

    /// Blocks the current readiness-polling call that found nothing ready: until any object
    /// changes, the earliest deadline of the watched objects, or the caller's `deadline`
    /// (`None`: forever), whichever comes first. At an object's deadline the call runs again
    /// (and finds it ready); at the caller's it returns 0. This is the one multi-object wait:
    /// every personality's `poll`/`select`/`epoll_wait` goes through it, so an object with a
    /// deadline (a timer) wakes all of them alike. The handler then reports "blocked".
    pub fn wait_objects(&mut self, objects: ObjectDeadlines, deadline: Option<u64>) {
        match (deadline, objects.wake_at) {
            (_, Some(t)) if deadline.is_none_or(|d| t < d) => {
                self.wait_for(WaitKey::AnyIo, Some(t), None);
            }
            (d, _) => self.wait_for(WaitKey::AnyIo, d, Some(0)),
        }
    }

    /// Drops the wait the current syscall arranged with [`Core::wait_for`] (a later part of
    /// the call made progress after all).
    pub fn cancel_wait(&mut self) {
        self.block_on = None;
    }

    /// The restartable blocker the current syscall arranged with [`Core::wait_for`], if any.
    pub fn take_block(&mut self, resume_pc: u64) -> Option<BlockerOf<T>> {
        self.block_on.take().map(|w| Blocker::Restart {
            key: w.key,
            deadline: w.deadline,
            resume_pc,
            timeout_result: w.timeout_result,
        })
    }
}
