//! File locks: POSIX record locks (`fcntl` `F_SETLK`/`F_SETLKW`/`F_GETLK`), open-file-description
//! locks (`F_OFD_*`) and `flock(2)`.
//!
//! One table for the whole kernel, keyed by inode. POSIX locks belong to a process and vanish
//! when it closes *any* descriptor for the file or exits. OFD locks and `flock` locks belong to
//! an open file description and vanish when the description closes. `flock` locks never
//! conflict with `fcntl` locks (as on Linux). A blocked `F_SETLKW` waits on [`LOCK_WAKE`] and
//! retries after every release, so the order in which waiters win is the scheduler's, which is
//! deterministic. There is no deadlock detection (`EDEADLK`).

use crate::vfs::Ino;

/// Wait object that threads blocked on a lock sleep on.
pub(crate) const LOCK_WAKE: u64 = u64::MAX - 1;

pub(crate) const F_RDLCK: u16 = 0;
pub(crate) const F_WRLCK: u16 = 1;
pub(crate) const F_UNLCK: u16 = 2;

/// Who holds a lock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Owner {
    /// A process (`fcntl` POSIX locks), by pid.
    Posix(u32),
    /// An open file description (`F_OFD_*`), by identity.
    Ofd(usize),
    /// An open file description's `flock`.
    Flock(usize),
}

impl Owner {
    fn is_flock(self) -> bool {
        matches!(self, Self::Flock(_))
    }
}

/// One held lock: the byte range `[start, end)`, `end == u64::MAX` meaning "to end of file".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Lock {
    pub ino: Ino,
    pub owner: Owner,
    pub start: u64,
    pub end: u64,
    pub write: bool,
}

impl Lock {
    fn overlaps(&self, start: u64, end: u64) -> bool {
        self.start < end && start < self.end
    }
}

/// All locks held in the system.
#[derive(Debug, Default)]
pub(crate) struct LockTable {
    pub(crate) locks: Vec<Lock>,
}

impl LockTable {
    /// The first lock held by someone else that stops `owner` taking `[start, end)`.
    pub fn conflict(
        &self,
        ino: Ino,
        owner: Owner,
        start: u64,
        end: u64,
        write: bool,
    ) -> Option<Lock> {
        self.locks
            .iter()
            .find(|l| {
                l.ino == ino
                    && l.owner != owner
                    && l.owner.is_flock() == owner.is_flock()
                    && l.overlaps(start, end)
                    && (l.write || write)
            })
            .copied()
    }

    /// Removes `owner`'s locks over `[start, end)` (splitting those that stick out).
    fn carve(&mut self, ino: Ino, owner: Owner, start: u64, end: u64) {
        let mut out = Vec::with_capacity(self.locks.len() + 1);
        for l in self.locks.drain(..) {
            if l.ino != ino || l.owner != owner || !l.overlaps(start, end) {
                out.push(l);
                continue;
            }
            if l.start < start {
                out.push(Lock { end: start, ..l });
            }
            if l.end > end {
                out.push(Lock { start: end, ..l });
            }
        }
        self.locks = out;
    }

    /// Takes (or converts) `owner`'s lock on `[start, end)`; the caller checked for conflicts.
    pub fn set(&mut self, ino: Ino, owner: Owner, start: u64, end: u64, write: bool) {
        self.carve(ino, owner, start, end);
        let mut new = Lock {
            ino,
            owner,
            start,
            end,
            write,
        };
        // Merge with adjacent locks of the same kind, as the kernel does.
        self.locks.retain(|l| {
            let adjacent = l.ino == ino && l.owner == owner && l.write == write;
            if adjacent && l.end == new.start {
                new.start = l.start;
                false
            } else if adjacent && l.start == new.end {
                new.end = l.end;
                false
            } else {
                true
            }
        });
        self.locks.push(new);
    }

    /// Drops `owner`'s locks over `[start, end)`.
    pub fn unlock(&mut self, ino: Ino, owner: Owner, start: u64, end: u64) {
        self.carve(ino, owner, start, end);
    }

    /// Drops every POSIX lock process `pid` holds on `ino` (it closed a descriptor for it).
    pub fn drop_posix_on(&mut self, pid: u32, ino: Ino) {
        self.locks
            .retain(|l| !(l.ino == ino && l.owner == Owner::Posix(pid)));
    }

    /// Drops every POSIX lock of a process that exited.
    pub fn drop_posix(&mut self, pid: u32) {
        self.locks.retain(|l| l.owner != Owner::Posix(pid));
    }

    /// Drops the OFD and `flock` locks of a closed open file description.
    pub fn drop_description(&mut self, id: usize) {
        self.locks
            .retain(|l| l.owner != Owner::Ofd(id) && l.owner != Owner::Flock(id));
    }

    /// Whether any lock is held (cheap guard for the close path).
    pub fn is_empty(&self) -> bool {
        self.locks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Owner = Owner::Posix(1);
    const B: Owner = Owner::Posix(2);

    #[test]
    fn readers_share_writers_conflict() {
        let mut t = LockTable::default();
        t.set(7, A, 0, 10, false);
        assert!(t.conflict(7, B, 0, 10, false).is_none());
        assert!(t.conflict(7, B, 5, 6, true).is_some());
        assert!(t.conflict(7, B, 10, 20, true).is_none());
        assert!(t.conflict(8, B, 0, 10, true).is_none());
        assert!(t.conflict(7, A, 0, 10, true).is_none());
    }

    #[test]
    fn unlock_splits_and_merge_joins() {
        let mut t = LockTable::default();
        t.set(7, A, 0, 100, true);
        t.unlock(7, A, 40, 60);
        assert!(t.conflict(7, B, 45, 55, true).is_none());
        assert!(t.conflict(7, B, 30, 45, true).is_some());
        t.set(7, A, 40, 60, true);
        assert_eq!(t.locks.len(), 1);
        assert_eq!((t.locks[0].start, t.locks[0].end), (0, 100));
    }

    #[test]
    fn flock_is_separate_and_ofd_conflicts_with_posix() {
        let mut t = LockTable::default();
        t.set(7, A, 0, u64::MAX, true);
        assert!(t.conflict(7, Owner::Flock(9), 0, u64::MAX, true).is_none());
        assert!(t.conflict(7, Owner::Ofd(9), 0, 1, false).is_some());
        t.drop_posix_on(1, 7);
        assert!(t.is_empty());
        t.set(7, Owner::Ofd(9), 0, 1, true);
        t.drop_description(9);
        assert!(t.is_empty());
    }
}
