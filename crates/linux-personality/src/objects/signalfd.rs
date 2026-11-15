//! `signalfd4`: reads pending signals as `struct signalfd_siginfo` records.
//!
//! A signalfd holds only a mask. Reading takes pending signals of the mask from the *reading*
//! thread (its own first, then its process's), like `rt_sigtimedwait`; readiness is computed
//! for the thread that asks (`Kernel::current`), as Linux does for `current`.

use std::cell::RefCell;
use std::rc::Rc;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{open, signal};

use super::{EPOLLIN, EPOLLRDNORM};
use crate::fd::{FileKind, open_file};
use crate::kernel::{Kernel, LinuxKey, Tid, WaitKey};
use crate::siginfo::SIGINFO_SIZE;
use crate::syscall::SysResult;

/// One signalfd.
#[derive(Debug)]
pub(crate) struct SignalFd {
    /// The signals it reads (never `SIGKILL` or `SIGSTOP`).
    pub mask: u64,
}

/// `SIGKILL` and `SIGSTOP`, which a signalfd never takes.
const UNBLOCKABLE: u64 = (1 << (signal::SIGKILL - 1)) | (1 << (signal::SIGSTOP - 1));

impl Kernel {
    /// `signalfd4(fd, mask, sizemask, flags)`: a new signalfd (`fd == -1`) or a new mask for
    /// an existing one.
    pub(crate) fn sys_signalfd4(
        &mut self,
        tid: Tid,
        fd: u64,
        mask_ptr: u64,
        size: u64,
        flags: u64,
    ) -> SysResult {
        if size != 8 || flags & !(open::O_CLOEXEC | open::O_NONBLOCK) != 0 {
            return Err(Errno::EINVAL);
        }
        let mask = crate::uaccess::read_u64(self.mm(tid).borrow_mut().mem.as_mut(), mask_ptr)?
            & !UNBLOCKABLE;
        if fd as i32 != -1 {
            let file = self.file(tid, fd)?;
            let FileKind::SignalFd(s) = &file.borrow().kind else {
                return Err(Errno::EINVAL);
            };
            s.borrow_mut().mask = mask;
            // Signals of the new mask may already be pending.
            self.wake(WaitKey::AnyIo, usize::MAX, u32::MAX);
            return Ok(fd);
        }
        let file = open_file(
            FileKind::SignalFd(Rc::new(RefCell::new(SignalFd { mask }))),
            open::O_RDWR | (flags & open::O_NONBLOCK),
            b"anon_inode:[signalfd]",
        );
        Ok(self
            .fds(tid)
            .borrow_mut()
            .insert(file, flags & open::O_CLOEXEC != 0, 0)? as u64)
    }

    /// Readiness of a signalfd for the calling thread, and the change generation.
    pub(crate) fn signalfd_events(&self, s: &SignalFd) -> (u32, u64) {
        let ready = self.pending_set(self.core.current) & s.mask != 0;
        (
            if ready { EPOLLIN | EPOLLRDNORM } else { 0 },
            self.sig_generation,
        )
    }

    /// Reads as many pending signals of the mask as fit, at least one (or blocks).
    pub(crate) fn signalfd_read(
        &mut self,
        s: &Rc<RefCell<SignalFd>>,
        buf: &mut [u8],
        nonblock: bool,
    ) -> Result<usize, Errno> {
        if buf.len() < SIGINFO_SIZE {
            return Err(Errno::EINVAL);
        }
        let tid = self.core.current;
        let mask = s.borrow().mask;
        let mut n = 0;
        while n + SIGINFO_SIZE <= buf.len() {
            let Some(info) = self.dequeue_signal(tid, mask) else {
                break;
            };
            buf[n..n + SIGINFO_SIZE].copy_from_slice(&info.to_signalfd());
            n += SIGINFO_SIZE;
        }
        if n > 0 {
            return Ok(n);
        }
        if nonblock {
            return Err(Errno::EAGAIN);
        }
        let pid = self.pid_of(tid);
        Err(self.wait_for(WaitKey::Personal(LinuxKey::Signals(pid)), None, None))
    }
}
