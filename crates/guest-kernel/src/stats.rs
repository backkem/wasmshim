//! Profile counters (the guest profiler): syscalls by number and the virtual
//! time threads spend blocked, for `stat:` inspect paths. Host-side bookkeeping only: nothing
//! the guest sees depends on them, they are not saved in checkpoints (a restored instance counts
//! from zero).
//!
//! They are updated on every syscall, so they are dense tables indexed by syscall number and by
//! thread id rather than maps (simpler per syscall than two map updates).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::task::Tid;

/// One syscall number's counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SysStat {
    /// Entries (a restarted blocking call counts once per attempt, like the `sys` clock).
    pub entries: u64,
    /// Entries that blocked.
    pub blocks: u64,
    /// Virtual nanoseconds its blocked waits lasted (block to wake).
    pub blocked_ns: u64,
}

/// One thread's counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThreadStat {
    pub syscalls: u64,
    /// Virtual nanoseconds spent blocked (completed waits).
    pub blocked_ns: u64,
    /// The wait in progress: since when, in which syscall.
    pub blocked_since: Option<(u64, u64)>,
}

/// Syscall numbers below this are counted in a dense table, the rest in a map.
const DENSE_SYS: u64 = 1024;
/// Thread ids below this are counted in a dense table, the rest in a map.
const DENSE_TIDS: usize = 1 << 16;

/// Per syscall number: a dense table and a map for rare large numbers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SysTable {
    dense: Vec<SysStat>,
    sparse: BTreeMap<u64, SysStat>,
}

impl SysTable {
    fn get_mut(&mut self, number: u64) -> &mut SysStat {
        if number < DENSE_SYS {
            let i = number as usize;
            if self.dense.len() <= i {
                self.dense.resize(i + 1, SysStat::default());
            }
            &mut self.dense[i]
        } else {
            self.sparse.entry(number).or_default()
        }
    }

    /// The numbers seen, ascending, with their counts.
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &SysStat)> + '_ {
        static NUMBERS: [u64; DENSE_SYS as usize] = {
            let mut n = [0u64; DENSE_SYS as usize];
            let mut i = 0;
            while i < n.len() {
                n[i] = i as u64;
                i += 1;
            }
            n
        };
        self.dense
            .iter()
            .enumerate()
            .filter(|(_, s)| **s != SysStat::default())
            .map(|(i, s)| (&NUMBERS[i], s))
            .chain(self.sparse.iter())
    }

    /// The counts of `number`, if it was seen.
    pub fn get(&self, number: &u64) -> Option<&SysStat> {
        if *number < DENSE_SYS {
            self.dense
                .get(*number as usize)
                .filter(|s| **s != SysStat::default())
        } else {
            self.sparse.get(number)
        }
    }
}

/// Per thread: a dense table and a map for rare large ids.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThreadTable {
    dense: Vec<Option<ThreadStat>>,
    sparse: BTreeMap<Tid, ThreadStat>,
}

impl ThreadTable {
    fn entry(&mut self, tid: Tid) -> &mut ThreadStat {
        let i = tid as usize;
        if i < DENSE_TIDS {
            if self.dense.len() <= i {
                self.dense.resize(i + 1, None);
            }
            self.dense[i].get_or_insert_with(ThreadStat::default)
        } else {
            self.sparse.entry(tid).or_default()
        }
    }

    fn get_mut(&mut self, tid: Tid) -> Option<&mut ThreadStat> {
        let i = tid as usize;
        if i < DENSE_TIDS {
            self.dense.get_mut(i).and_then(Option::as_mut)
        } else {
            self.sparse.get_mut(&tid)
        }
    }

    /// The counts of thread `tid`, if it made a syscall.
    pub fn get(&self, tid: &Tid) -> Option<&ThreadStat> {
        let i = *tid as usize;
        if i < DENSE_TIDS {
            self.dense.get(i).and_then(Option::as_ref)
        } else {
            self.sparse.get(tid)
        }
    }
}

/// The instance's profile counters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub sys: SysTable,
    pub threads: ThreadTable,
    /// The virtual time wakes are stamped with: the scheduler's latest look at the clock.
    pub now: u64,
}

impl Stats {
    /// A syscall entry.
    pub(crate) fn enter(&mut self, tid: Tid, number: u64) {
        self.sys.get_mut(number).entries += 1;
        self.threads.entry(tid).syscalls += 1;
    }

    /// The syscall being handled blocked at `now`.
    pub(crate) fn block(&mut self, tid: Tid, number: u64, now: u64) {
        self.sys.get_mut(number).blocks += 1;
        self.threads.entry(tid).blocked_since = Some((now, number));
    }

    /// The thread woke at `now`.
    pub(crate) fn wake(&mut self, tid: Tid, now: u64) {
        let Some(t) = self.threads.get_mut(tid) else {
            return;
        };
        if let Some((since, number)) = t.blocked_since.take() {
            let d = now.saturating_sub(since);
            t.blocked_ns += d;
            self.sys.get_mut(number).blocked_ns += d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_read_like_the_maps_they_replace() {
        let mut s = Stats::default();
        for (tid, n) in [(7, 98), (7, 63), (70_000, 2000), (3, 98)] {
            s.enter(tid, n);
        }
        s.block(7, 98, 10);
        s.wake(7, 25);
        s.wake(9, 30); // never made a syscall: ignored
        let sys: Vec<(u64, SysStat)> = s.sys.iter().map(|(&n, &x)| (n, x)).collect();
        let stat = |entries, blocks, blocked_ns| SysStat {
            entries,
            blocks,
            blocked_ns,
        };
        assert_eq!(
            sys,
            [
                (63, stat(1, 0, 0)),
                (98, stat(2, 1, 15)),
                (2000, stat(1, 0, 0))
            ]
        );
        assert_eq!(s.sys.get(&1), None);
        assert_eq!(
            s.threads.get(&7).map(|t| (t.syscalls, t.blocked_ns)),
            Some((2, 15))
        );
        assert_eq!(s.threads.get(&70_000).map(|t| t.syscalls), Some(1));
        assert!(s.threads.get(&9).is_none());
    }
}
