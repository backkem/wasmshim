//! Tasks: one guest hardware thread each, with the substrate's scheduling state and the
//! personality's own per-task state (`ext`, reached through `Deref`).

use alloc::boxed::Box;
use core::ops::{Deref, DerefMut};

use wasmshim_cpu::GuestCpu;

use crate::Types;
use crate::wait::Blocker;

/// A task id (personalities may share the number space with their own ids: Linux's tids and
/// pids).
pub type Tid = u32;

/// Whether a task can run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskState<K, P> {
    Runnable,
    Blocked(Blocker<K, P>),
}

/// One task.
pub struct Task<T: Types> {
    pub tid: Tid,
    pub cpu: Box<dyn GuestCpu>,
    pub state: TaskState<T::Key, T::Park>,
    /// A timed blocking syscall in progress: the `SVC` resume pc and the absolute deadline
    /// fixed at its first entry (kept across the restarts of one call; see
    /// [`crate::Core::restart_deadline`]).
    pub timed: Option<(u64, u64)>,
    /// The CPU's retired count when this task got it (a cloned CPU starts with its parent's;
    /// after a restore, the new CPU's count minus the task's own, modulo 2^64).
    pub retired_base: u64,
    /// The number of the last syscall this task entered (diagnostics).
    pub last_syscall: Option<u64>,
    /// The personality's per-task state.
    pub ext: T::Task,
}

impl<T: Types> Task<T> {
    /// A runnable task.
    pub fn new(tid: Tid, cpu: Box<dyn GuestCpu>, retired_base: u64, ext: T::Task) -> Self {
        Self {
            tid,
            cpu,
            state: TaskState::Runnable,
            timed: None,
            retired_base,
            last_syscall: None,
            ext,
        }
    }

    /// Instructions this task retired itself.
    pub fn own_retired(&self) -> u64 {
        self.cpu.retired().wrapping_sub(self.retired_base)
    }
}

impl<T: Types> Deref for Task<T> {
    type Target = T::Task;
    fn deref(&self) -> &T::Task {
        &self.ext
    }
}

impl<T: Types> DerefMut for Task<T> {
    fn deref_mut(&mut self) -> &mut T::Task {
        &mut self.ext
    }
}
