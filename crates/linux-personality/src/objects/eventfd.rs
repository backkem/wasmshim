//! `eventfd2`: a 64-bit counter.

use std::cell::RefCell;
use std::rc::Rc;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::open;

use super::{EPOLLIN, EPOLLOUT, EPOLLRDNORM, EPOLLWRNORM};
use crate::fd::{FileKind, open_file};
use crate::kernel::{Kernel, Tid, WaitKey};
use crate::syscall::SysResult;

const EFD_SEMAPHORE: u64 = 1;
const MAX: u64 = u64::MAX - 1;

/// One eventfd.
#[derive(Debug)]
pub(crate) struct EventFd {
    pub id: u64,
    pub count: u64,
    pub semaphore: bool,
    pub generation: u64,
}

impl EventFd {
    pub(crate) fn events(&self) -> (u32, u64) {
        let mut ev = 0;
        if self.count > 0 {
            ev |= EPOLLIN | EPOLLRDNORM;
        }
        if self.count < MAX {
            ev |= EPOLLOUT | EPOLLWRNORM;
        }
        (ev, self.generation)
    }
}

impl Kernel {
    /// `eventfd2(initval, flags)`.
    pub(crate) fn sys_eventfd2(&mut self, tid: Tid, init: u64, flags: u64) -> SysResult {
        if flags & !(EFD_SEMAPHORE | open::O_NONBLOCK | open::O_CLOEXEC) != 0 {
            return Err(Errno::EINVAL);
        }
        let id = self.new_object_id();
        let e = Rc::new(RefCell::new(EventFd {
            id,
            count: init & 0xffff_ffff,
            semaphore: flags & EFD_SEMAPHORE != 0,
            generation: 0,
        }));
        let file = open_file(
            FileKind::EventFd(e),
            open::O_RDWR | (flags & open::O_NONBLOCK),
            b"anon_inode:[eventfd]",
        );
        Ok(self
            .fds(tid)
            .borrow_mut()
            .insert(file, flags & open::O_CLOEXEC != 0, 0)? as u64)
    }

    pub(crate) fn eventfd_read(
        &mut self,
        e: &Rc<RefCell<EventFd>>,
        buf: &mut [u8],
        nonblock: bool,
    ) -> Result<usize, Errno> {
        if buf.len() < 8 {
            return Err(Errno::EINVAL);
        }
        let id = {
            let mut ev = e.borrow_mut();
            if ev.count == 0 {
                if nonblock {
                    return Err(Errno::EAGAIN);
                }
                let id = ev.id;
                drop(ev);
                return Err(self.wait_for(WaitKey::Object(id), None, None));
            }
            let value = if ev.semaphore { 1 } else { ev.count };
            ev.count -= value;
            ev.generation += 1;
            buf[..8].copy_from_slice(&value.to_le_bytes());
            ev.id
        };
        self.object_changed(id);
        Ok(8)
    }

    pub(crate) fn eventfd_write(
        &mut self,
        e: &Rc<RefCell<EventFd>>,
        data: &[u8],
        nonblock: bool,
    ) -> Result<usize, Errno> {
        let Some(bytes) = data.get(..8) else {
            return Err(Errno::EINVAL);
        };
        let mut raw = [0u8; 8];
        raw.copy_from_slice(bytes);
        let value = u64::from_le_bytes(raw);
        if value == u64::MAX {
            return Err(Errno::EINVAL);
        }
        let id = {
            let mut ev = e.borrow_mut();
            if MAX - ev.count < value {
                if nonblock {
                    return Err(Errno::EAGAIN);
                }
                let id = ev.id;
                drop(ev);
                return Err(self.wait_for(WaitKey::Object(id), None, None));
            }
            ev.count += value;
            ev.generation += 1;
            ev.id
        };
        self.object_changed(id);
        Ok(8)
    }
}
