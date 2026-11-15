//! `fcntl` record locks (POSIX and open-file-description) and `flock(2)`: see [`crate::locks`].

use std::rc::Rc;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{open, seek};

use super::SysResult;
use crate::fd::{FileKind, FileRef};
use crate::kernel::{Kernel, Tid, WaitKey};
use crate::locks::{F_RDLCK, F_UNLCK, F_WRLCK, LOCK_WAKE, Owner};
use crate::uaccess::{read_array, write_bytes};

/// `fcntl` commands for OFD locks.
pub(super) const F_OFD_GETLK: u64 = 36;
pub(super) const F_OFD_SETLK: u64 = 37;
pub(super) const F_OFD_SETLKW: u64 = 38;

const LOCK_SH: u64 = 1;
const LOCK_EX: u64 = 2;
const LOCK_NB: u64 = 4;
const LOCK_UN: u64 = 8;

/// The identity of an open file description (only compared, never dereferenced).
pub(crate) fn description_id(file: &FileRef) -> usize {
    Rc::as_ptr(file) as *const u8 as usize
}

impl Kernel {
    fn lockable_ino(file: &FileRef) -> Option<u64> {
        match file.borrow().kind {
            FileKind::Inode { ino, .. } => Some(ino),
            _ => None,
        }
    }

    fn wait_for_lock(&mut self) -> Errno {
        self.wait_for(WaitKey::Object(LOCK_WAKE), None, None)
    }

    fn lock_released(&mut self) {
        self.wake(WaitKey::Object(LOCK_WAKE), 0, 0);
    }

    /// `F_GETLK`, `F_SETLK`, `F_SETLKW` and their `F_OFD_*` forms; `arg` is a `struct flock *`.
    pub(super) fn sys_fcntl_lock(
        &mut self,
        tid: Tid,
        file: &FileRef,
        cmd: u64,
        arg: u64,
    ) -> SysResult {
        use wasmshim_linux_abi::consts::fcntl::{F_GETLK, F_SETLK, F_SETLKW};
        let ofd = matches!(cmd, F_OFD_GETLK | F_OFD_SETLK | F_OFD_SETLKW);
        let raw = read_array::<32>(self.mm(tid).borrow_mut().mem.as_mut(), arg)?;
        let ltype = u16::from_le_bytes([raw[0], raw[1]]);
        let whence = i16::from_le_bytes([raw[2], raw[3]]);
        let l_start = i64::from_le_bytes(raw[8..16].try_into().unwrap_or_default());
        let l_len = i64::from_le_bytes(raw[16..24].try_into().unwrap_or_default());
        let l_pid = i32::from_le_bytes(raw[24..28].try_into().unwrap_or_default());
        if ltype > F_UNLCK || (ofd && l_pid != 0) {
            return Err(Errno::EINVAL);
        }
        let Some(ino) = Self::lockable_ino(file) else {
            return Err(Errno::EBADF);
        };
        let pid = self.pid_of(tid);
        let owner = if ofd {
            Owner::Ofd(description_id(file))
        } else {
            Owner::Posix(pid)
        };
        let base: i64 = match whence as u64 {
            seek::SEEK_SET => 0,
            seek::SEEK_CUR => file.borrow().offset as i64,
            seek::SEEK_END => self.vfs.size(ino) as i64,
            _ => return Err(Errno::EINVAL),
        };
        let abs = base.checked_add(l_start).ok_or(Errno::EINVAL)?;
        let (start, end) = if l_len > 0 {
            (abs, abs.checked_add(l_len).ok_or(Errno::EINVAL)?)
        } else if l_len == 0 {
            (abs, i64::MAX)
        } else {
            (abs.checked_add(l_len).ok_or(Errno::EINVAL)?, abs)
        };
        if start < 0 {
            return Err(Errno::EINVAL);
        }
        let (start, end) = (
            start as u64,
            if end == i64::MAX {
                u64::MAX
            } else {
                end as u64
            },
        );
        let write = ltype == F_WRLCK;
        match cmd {
            F_GETLK | F_OFD_GETLK => {
                let mut out = raw;
                match self.locks.conflict(ino, owner, start, end, write) {
                    None => out[0..2].copy_from_slice(&F_UNLCK.to_le_bytes()),
                    Some(l) => {
                        let t = if l.write { F_WRLCK } else { F_RDLCK };
                        out[0..2].copy_from_slice(&t.to_le_bytes());
                        out[2..4].copy_from_slice(&0i16.to_le_bytes());
                        out[8..16].copy_from_slice(&(l.start as i64).to_le_bytes());
                        let len = if l.end == u64::MAX {
                            0
                        } else {
                            (l.end - l.start) as i64
                        };
                        out[16..24].copy_from_slice(&len.to_le_bytes());
                        let holder = match l.owner {
                            Owner::Posix(p) => p as i32,
                            _ => -1,
                        };
                        out[24..28].copy_from_slice(&holder.to_le_bytes());
                    }
                }
                write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), arg, &out)?;
                Ok(0)
            }
            F_SETLK | F_SETLKW | F_OFD_SETLK | F_OFD_SETLKW => {
                if ltype == F_UNLCK {
                    self.locks.unlock(ino, owner, start, end);
                    self.lock_released();
                    return Ok(0);
                }
                let acc = file.borrow().flags & open::O_ACCMODE;
                if (write && acc == open::O_RDONLY) || (!write && acc == open::O_WRONLY) {
                    return Err(Errno::EBADF);
                }
                if self.locks.conflict(ino, owner, start, end, write).is_some() {
                    return if matches!(cmd, F_SETLKW | F_OFD_SETLKW) {
                        Err(self.wait_for_lock())
                    } else {
                        Err(Errno::EAGAIN)
                    };
                }
                self.locks.set(ino, owner, start, end, write);
                // A conversion may have released a conflict for someone else.
                self.lock_released();
                Ok(0)
            }
            _ => Err(Errno::EINVAL),
        }
    }

    /// `flock(fd, operation)`.
    pub(super) fn sys_flock(&mut self, tid: Tid, fd: u64, op: u64) -> SysResult {
        let file = self.file(tid, fd)?;
        let Some(ino) = Self::lockable_ino(&file) else {
            return Ok(0);
        };
        let owner = Owner::Flock(description_id(&file));
        match op & !LOCK_NB {
            LOCK_UN => {
                self.locks.unlock(ino, owner, 0, u64::MAX);
                self.lock_released();
                Ok(0)
            }
            LOCK_SH | LOCK_EX => {
                let write = op & LOCK_EX != 0;
                if self
                    .locks
                    .conflict(ino, owner, 0, u64::MAX, write)
                    .is_some()
                {
                    return if op & LOCK_NB != 0 {
                        Err(Errno::EAGAIN)
                    } else {
                        Err(self.wait_for_lock())
                    };
                }
                self.locks.set(ino, owner, 0, u64::MAX, write);
                self.lock_released();
                Ok(0)
            }
            _ => Err(Errno::EINVAL),
        }
    }

    /// A descriptor for `file` was closed by `pid`: POSIX locks of that process on the file go.
    pub(crate) fn locks_fd_closed(&mut self, pid: u32, file: &FileRef) {
        if self.locks.is_empty() {
            return;
        }
        if let Some(ino) = Self::lockable_ino(file) {
            self.locks.drop_posix_on(pid, ino);
            self.lock_released();
        }
    }

    /// An open file description closed: its OFD and `flock` locks go.
    pub(crate) fn locks_description_closed(&mut self, file: &FileRef) {
        if self.locks.is_empty() {
            return;
        }
        self.locks.drop_description(description_id(file));
        self.lock_released();
    }
}
