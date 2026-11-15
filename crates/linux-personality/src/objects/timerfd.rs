//! `timerfd_create`/`timerfd_settime`/`timerfd_gettime`, on the virtual clocks.
//!
//! Expirations are computed lazily from the current time (no timer events exist): a blocked
//! reader waits with a deadline at the next expiry and re-runs its `read` then.

use std::cell::RefCell;
use std::rc::Rc;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{clock, open};
use wasmshim_linux_abi::structs::Timespec;

use super::{EPOLLIN, EPOLLRDNORM};
use crate::fd::{FileKind, open_file};
use crate::host::Host;
use crate::kernel::{Kernel, Tid, WaitKey};
use crate::syscall::SysResult;
use crate::uaccess::{read_array, write_bytes};

const TFD_TIMER_ABSTIME: u64 = 1;

/// One timerfd. Times are monotonic nanoseconds.
#[derive(Debug)]
pub(crate) struct TimerFd {
    pub id: u64,
    pub realtime: bool,
    /// The next expiry (monotonic ns), if armed.
    pub next: Option<u64>,
    pub interval: u64,
    /// Expirations counted but not yet read.
    pub ticks: u64,
    /// Total expirations ever (the edge-trigger generation).
    pub total: u64,
}

impl TimerFd {
    /// Expirations that have happened by `now` beyond what is recorded.
    fn due(&self, now: u64) -> u64 {
        match self.next {
            Some(next) if next <= now => 1 + (now - next).checked_div(self.interval).unwrap_or(0),
            _ => 0,
        }
    }

    pub(crate) fn events(&self, now: u64) -> (u32, u64) {
        let due = self.due(now);
        let ev = if self.ticks + due > 0 {
            EPOLLIN | EPOLLRDNORM
        } else {
            0
        };
        (ev, self.total + due)
    }

    /// The next expiry, if the timer is armed.
    pub(crate) fn next_expiry(&self) -> Option<u64> {
        self.next
    }

    /// Folds expirations up to `now` into `ticks`.
    fn advance(&mut self, now: u64) {
        let due = self.due(now);
        if due > 0 {
            self.ticks += due;
            self.total += due;
            self.next = match (self.next, self.interval) {
                (Some(next), i) if i > 0 => Some(next + due * i),
                _ => None,
            };
        }
    }
}

impl Kernel {
    /// `timerfd_create(clockid, flags)`.
    pub(crate) fn sys_timerfd_create(&mut self, tid: Tid, clockid: u64, flags: u64) -> SysResult {
        let realtime = match clockid {
            clock::CLOCK_REALTIME => true,
            clock::CLOCK_MONOTONIC | clock::CLOCK_BOOTTIME => false,
            _ => return Err(Errno::EINVAL),
        };
        if flags & !(open::O_NONBLOCK | open::O_CLOEXEC) != 0 {
            return Err(Errno::EINVAL);
        }
        let id = self.new_object_id();
        let t = Rc::new(RefCell::new(TimerFd {
            id,
            realtime,
            next: None,
            interval: 0,
            ticks: 0,
            total: 0,
        }));
        let file = open_file(
            FileKind::TimerFd(t),
            open::O_RDWR | (flags & open::O_NONBLOCK),
            b"anon_inode:[timerfd]",
        );
        Ok(self
            .fds(tid)
            .borrow_mut()
            .insert(file, flags & open::O_CLOEXEC != 0, 0)? as u64)
    }

    fn timerfd(&self, tid: Tid, fd: u64) -> Result<Rc<RefCell<TimerFd>>, Errno> {
        match &self.file(tid, fd)?.borrow().kind {
            FileKind::TimerFd(t) => Ok(t.clone()),
            _ => Err(Errno::EINVAL),
        }
    }

    fn itimerspec(t: &TimerFd, now: u64) -> [u8; 32] {
        let remaining = t.next.map_or(0, |n| n.saturating_sub(now).max(1));
        let mut out = [0u8; 32];
        out[..16].copy_from_slice(&Timespec::from_nanos(t.interval).to_bytes());
        out[16..].copy_from_slice(&Timespec::from_nanos(remaining).to_bytes());
        out
    }

    /// `timerfd_settime(fd, flags, new, old)`.
    pub(crate) fn sys_timerfd_settime(
        &mut self,
        tid: Tid,
        fd: u64,
        flags: u64,
        new: u64,
        old: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let t = self.timerfd(tid, fd)?;
        let raw: [u8; 32] = read_array(self.mm(tid).borrow_mut().mem.as_mut(), new)?;
        let interval = Timespec::from_bytes(raw[..16].try_into().unwrap_or(&[0; 16]))
            .to_nanos_saturating()
            .ok_or(Errno::EINVAL)?;
        let value = Timespec::from_bytes(raw[16..].try_into().unwrap_or(&[0; 16]))
            .to_nanos_saturating()
            .ok_or(Errno::EINVAL)?;
        let now = host.monotonic_ns();
        let id = {
            let mut tm = t.borrow_mut();
            tm.advance(now);
            if old != 0 {
                let prev = Self::itimerspec(&tm, now);
                write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), old, &prev)?;
            }
            tm.interval = interval;
            tm.ticks = 0;
            tm.next = if value == 0 {
                None
            } else if flags & TFD_TIMER_ABSTIME != 0 {
                // Absolute times are on the timer's own clock; convert realtime to monotonic.
                Some(if tm.realtime {
                    now.saturating_add(value.saturating_sub(host.realtime_ns()))
                } else {
                    value
                })
            } else {
                Some(now.saturating_add(value))
            };
            tm.id
        };
        self.object_changed(id);
        Ok(0)
    }

    /// `timerfd_gettime(fd, cur)`.
    pub(crate) fn sys_timerfd_gettime(
        &mut self,
        tid: Tid,
        fd: u64,
        cur: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let t = self.timerfd(tid, fd)?;
        let now = host.monotonic_ns();
        t.borrow_mut().advance(now);
        let spec = Self::itimerspec(&t.borrow(), now);
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), cur, &spec)?;
        Ok(0)
    }

    pub(crate) fn timerfd_read(
        &mut self,
        t: &Rc<RefCell<TimerFd>>,
        buf: &mut [u8],
        nonblock: bool,
        now: u64,
    ) -> Result<usize, Errno> {
        if buf.len() < 8 {
            return Err(Errno::EINVAL);
        }
        let mut tm = t.borrow_mut();
        tm.advance(now);
        if tm.ticks == 0 {
            if nonblock {
                return Err(Errno::EAGAIN);
            }
            let (id, next) = (tm.id, tm.next);
            drop(tm);
            // Re-run the read at the next expiry (or when re-armed).
            return Err(self.wait_for(WaitKey::Object(id), next, None));
        }
        buf[..8].copy_from_slice(&tm.ticks.to_le_bytes());
        tm.ticks = 0;
        Ok(8)
    }
}
