//! Process file descriptors: `pidfd_open`, `pidfd_send_signal`, `pidfd_getfd`.
//!
//! A pidfd names a process by its pid (pids are never reused within a kernel's lifetime here,
//! so the number is a stable identity). It is readable once the process has exited.

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{open, signal};

use super::{EPOLLIN, EPOLLRDNORM};
use crate::fd::{FileKind, FileRef, open_file};
use crate::kernel::{Kernel, Pid, Tid};
use crate::siginfo::SigInfo;
use crate::syscall::SysResult;

/// `PIDFD_THREAD` (`O_EXCL`): the pidfd names a thread.
const PIDFD_THREAD: u64 = open::O_EXCL;
/// `pidfd_send_signal` scopes.
const PIDFD_SIGNAL_THREAD: u64 = 1;
const PIDFD_SIGNAL_THREAD_GROUP: u64 = 2;
const PIDFD_SIGNAL_PROCESS_GROUP: u64 = 4;

impl Kernel {
    /// Readiness of a pidfd: readable (and hung up) once the process has exited.
    pub(crate) fn pidfd_events(&self, pid: Pid) -> (u32, u64) {
        if self.procs.contains_key(&pid) || self.core.tasks.contains_key(&pid) {
            (0, 0)
        } else {
            (EPOLLIN | EPOLLRDNORM, 1)
        }
    }

    /// The pid behind pidfd `fd`, and the description.
    pub(crate) fn pidfd_target(&self, tid: Tid, fd: u64) -> Result<(Pid, FileRef), Errno> {
        let file = self.file(tid, fd)?;
        let pid = match file.borrow().kind {
            FileKind::PidFd { pid } => pid,
            _ => return Err(Errno::EBADF),
        };
        Ok((pid, file))
    }

    /// `pidfd_open(pid, flags)`.
    pub(crate) fn sys_pidfd_open(&mut self, tid: Tid, pid: u64, flags: u64) -> SysResult {
        if flags & !(open::O_NONBLOCK | PIDFD_THREAD) != 0 || pid as i32 <= 0 {
            return Err(Errno::EINVAL);
        }
        let pid = pid as u32;
        let leader = self.procs.contains_key(&pid) || self.zombies.contains_key(&pid);
        if !leader {
            if self.core.tasks.contains_key(&pid) {
                if flags & PIDFD_THREAD == 0 {
                    return Err(Errno::EINVAL);
                }
            } else {
                return Err(Errno::ESRCH);
            }
        }
        let file = open_file(
            FileKind::PidFd { pid },
            open::O_RDWR | (flags & (open::O_NONBLOCK | PIDFD_THREAD)),
            b"anon_inode:[pidfd]",
        );
        // Always close-on-exec, like Linux.
        Ok(self.fds(tid).borrow_mut().insert(file, true, 0)? as u64)
    }

    /// `pidfd_send_signal(pidfd, sig, info, flags)`.
    pub(crate) fn sys_pidfd_send_signal(
        &mut self,
        tid: Tid,
        fd: u64,
        s: u64,
        info_ptr: u64,
        flags: u64,
    ) -> SysResult {
        let scope =
            flags & (PIDFD_SIGNAL_THREAD | PIDFD_SIGNAL_THREAD_GROUP | PIDFD_SIGNAL_PROCESS_GROUP);
        if flags != scope || scope.count_ones() > 1 {
            return Err(Errno::EINVAL);
        }
        let s = u32::try_from(s).map_err(|_| Errno::EINVAL)?;
        if s > signal::NSIG {
            return Err(Errno::EINVAL);
        }
        let (pid, file) = self.pidfd_target(tid, fd)?;
        let thread_fd = file.borrow().flags & PIDFD_THREAD != 0;
        let me = self.pid_of(tid);
        let info = if info_ptr != 0 {
            let tgid = self.core.tasks.get(&pid).map_or(pid, |t| t.pid);
            let raw = crate::uaccess::read_bytes(
                self.mm(tid).borrow_mut().mem.as_mut(),
                info_ptr,
                crate::siginfo::SIGINFO_SIZE,
            )?;
            if u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) != s {
                return Err(Errno::EINVAL);
            }
            let info = SigInfo::from_raw(s, &raw);
            if (info.code() >= 0 || info.code() == signal::SI_TKILL) && tgid != me {
                return Err(Errno::EPERM);
            }
            info
        } else {
            SigInfo::sender(s, signal::SI_USER, me, self.creds(tid).ruid)
        };
        let alive = self.procs.contains_key(&pid) || self.core.tasks.contains_key(&pid);
        if !alive {
            // A zombie still exists (the signal goes nowhere); a reaped process does not.
            return if self.zombies.contains_key(&pid) {
                Ok(0)
            } else {
                Err(Errno::ESRCH)
            };
        }
        let to_thread = scope == PIDFD_SIGNAL_THREAD || (scope == 0 && thread_fd);
        if to_thread {
            return self.queue_to_thread(pid, info);
        }
        if scope == PIDFD_SIGNAL_PROCESS_GROUP {
            let pgid = self.procs.get(&pid).map_or(pid, |p| p.pgid);
            let members: Vec<Pid> = self
                .procs
                .iter()
                .filter(|(_, p)| p.pgid == pgid)
                .map(|(&q, _)| q)
                .collect();
            for q in members {
                let _ = self.queue_to_process(q, info);
            }
            return Ok(0);
        }
        let leader = self.core.tasks.get(&pid).map_or(pid, |t| t.pid);
        self.queue_to_process(leader, info)
    }

    /// `pidfd_getfd(pidfd, targetfd, flags)`: a copy of another process's descriptor
    /// (the same open file description), close-on-exec.
    pub(crate) fn sys_pidfd_getfd(
        &mut self,
        tid: Tid,
        fd: u64,
        target: u64,
        flags: u64,
    ) -> SysResult {
        if flags != 0 {
            return Err(Errno::EINVAL);
        }
        let (pid, _) = self.pidfd_target(tid, fd)?;
        let leader = self.core.tasks.get(&pid).map_or(pid, |t| t.pid);
        let Some(p) = self.procs.get(&leader) else {
            return Err(Errno::ESRCH);
        };
        let file = p.fds.borrow().get(target as i32)?;
        Ok(self.fds(tid).borrow_mut().insert(file, true, 0)? as u64)
    }
}
