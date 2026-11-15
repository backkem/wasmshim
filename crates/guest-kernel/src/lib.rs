//! The syscall substrate: what every OS personality stands on.
//!
//! * [`Host`], [`Blob`], [`Frontend`]: the host contract and how execution engines plug in.
//! * [`Core`]: tasks ([`Task`]) and the deterministic scheduler state, object and address-space
//!   ids, the diagnostics queue.
//! * [`wait`]: the one wait primitive ([`WaitKey`], [`Blocker`], [`Core::wake`]).
//! * [`sched`]: the run loop ([`run`]) over a [`Personality`]: green tasks preempted on
//!   retired-instruction quanta, compute charged as virtual time, syscall entry, the trace.
//!
//! A personality is a syscall table plus its own objects: it owns a [`Core`] and implements
//! [`Personality`]. It never copies the scheduler or the waits. This crate has no syscall table
//! and no OS objects; it depends on `wasmshim-cpu` only. Its vocabulary is shaped by Linux-style
//! kernels (futex waits and `-errno` return values, AArch64 syscall entry), and the
//! personality supplies everything else.

#![no_std]
#![deny(unsafe_code)]

extern crate alloc;

pub mod checkpoint;
pub mod debug;
mod host;
pub mod observe;
pub mod sched;
pub mod space;
pub mod stats;
mod task;
pub mod wait;

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Debug;

pub use host::{Blob, BlobError, Frontend, Host};
pub use sched::{Flow, Isa, Personality, SchedConfig, Space, Status, run};
pub use task::{Task, TaskState, Tid};
pub use wait::{Blocker, ObjectDeadlines, WaitKey};

/// The types a personality plugs into the substrate.
pub trait Types {
    /// Per-task state the personality keeps (Linux: signal masks and queues); reached through
    /// [`Task`]'s `Deref`.
    type Task;
    /// The personality's own wait keys (Linux: children, signals).
    type Key: Copy + Eq + Debug;
    /// The personality's own parking reasons (Linux: `vfork`, `rt_sigsuspend`).
    type Park: Copy + Eq + Debug;
}

/// The wait keys of personality `T`.
pub type KeyOf<T> = WaitKey<<T as Types>::Key>;
/// The blockers of personality `T`.
pub type BlockerOf<T> = Blocker<<T as Types>::Key, <T as Types>::Park>;
/// The syscall flows of personality `T`.
pub type FlowOf<T> = Flow<<T as Types>::Key, <T as Types>::Park>;

/// A wait the current syscall arranged ([`Core::wait_for`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PendingWait<K> {
    key: WaitKey<K>,
    deadline: Option<u64>,
    timeout_result: Option<u64>,
}

/// The substrate's state: tasks, the run queue, time accounting, ids, diagnostics.
pub struct Core<T: Types> {
    pub cfg: SchedConfig,
    /// Every live task, in tid order (the order wake-ups and reports follow).
    pub tasks: BTreeMap<Tid, Task<T>>,
    pub(crate) run_queue: VecDeque<Tid>,
    next_space_id: u64,
    /// Arrival sequence of futex waits.
    pub next_wait_seq: u64,
    next_object_id: u64,
    /// Instructions retired since the last yield to the host.
    pub(crate) since_yield: u64,
    /// Monotonic time when the instructions in `since_yield` started to accrue.
    pub(crate) accrual_start: u64,
    /// The virtual CPU is busy (charged compute) until this instant: [`run`] runs nothing
    /// before it, even when the embedder calls early (e.g. on a network event).
    pub(crate) busy_until: u64,
    /// Instructions retired by tasks that no longer exist.
    pub retired_dead: u64,
    /// Set by a handler that blocks: what to wait for ([`Core::wait_for`]).
    pub(crate) block_on: Option<PendingWait<T::Key>>,
    /// Diagnostics produced where no [`Host`] is at hand; flushed by [`run`].
    diags: Vec<String>,
    /// Monotonic time of the syscall being handled (at the start of a slice until then; the
    /// same under virtual time, later under a real clock).
    pub slice_now: u64,
    /// The task whose syscall is being handled.
    pub current: Tid,
    /// The resume pc of the syscall being handled ([`Core::restart_deadline`]).
    pub current_pc: u64,
    /// Profile counters (`stat:` paths; never saved, never guest-visible).
    pub stats: stats::Stats,
}

impl<T: Types> Core<T> {
    /// An empty machine. A `quantum` of 0 is taken as 1 ([`SchedConfig::quantum`]).
    pub fn new(cfg: SchedConfig) -> Self {
        Self {
            cfg: cfg.checked(),
            tasks: BTreeMap::new(),
            run_queue: VecDeque::new(),
            next_space_id: 1,
            next_wait_seq: 0,
            next_object_id: 1,
            since_yield: 0,
            accrual_start: 0,
            busy_until: 0,
            retired_dead: 0,
            block_on: None,
            diags: Vec::new(),
            slice_now: 0,
            current: 0,
            current_pc: 0,
            stats: stats::Stats::default(),
        }
    }

    /// Registers a runnable task and queues it.
    pub fn add_task(&mut self, t: Task<T>) {
        let tid = t.tid;
        self.tasks.insert(tid, t);
        self.run_queue.push_back(tid);
    }

    /// Removes a task, keeping its retired instructions in the total.
    pub fn remove_task(&mut self, tid: Tid) -> Option<Task<T>> {
        let t = self.tasks.remove(&tid)?;
        self.retired_dead += t.own_retired();
        Some(t)
    }

    /// Total instructions retired by every task so far.
    pub fn retired(&self) -> u64 {
        self.retired_dead + self.tasks.values().map(Task::own_retired).sum::<u64>()
    }

    /// A unique object id (wait keys use it).
    pub fn new_object_id(&mut self) -> u64 {
        let id = self.next_object_id;
        self.next_object_id += 1;
        id
    }

    /// A unique address-space id (futex keys use it).
    pub fn new_space_id(&mut self) -> u64 {
        let id = self.next_space_id;
        self.next_space_id += 1;
        id
    }

    /// Queues a diagnostic line for the host.
    pub fn diag(&mut self, line: String) {
        self.diags.push(line);
    }

    /// Hands the queued diagnostics to `host`.
    pub fn flush_diags(&mut self, host: &mut dyn Host) {
        for line in self.diags.drain(..) {
            host.diag(&line);
        }
    }
}

#[cfg(test)]
mod tests;
