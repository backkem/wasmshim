//! Linux native AIO: `io_setup`, `io_destroy`, `io_submit`, `io_getevents`, `io_pgetevents`,
//! `io_cancel`.
//!
//! Every request completes **synchronously at submission** (the VFS is in memory, so there is
//! nothing to overlap), and its completion is queued on the context and, with
//! `IOCB_FLAG_RESFD`, signalled through the eventfd, exactly as a fast device would. nginx
//! (`--with-file-aio`) creates a context in every worker and uses it with `aio on;`.
//!
//! The context id is, as on Linux, the address of a ring mapping in the caller's address
//! space. Its header's magic is left 0, so libaio-style readers that peek at the ring fall back
//! to `io_getevents`, which is the only completion path here.

use std::collections::VecDeque;

use wasmshim_linux_abi::Errno;

use super::SysResult;
use crate::fd::FileKind;
use crate::host::Host;
use crate::kernel::{Kernel, Tid, WaitKey};
use crate::mm::{VmaKind, page_up};
use crate::uaccess::{read_array, write_bytes, write_u64};

const IOCB_CMD_PREAD: u16 = 0;
const IOCB_CMD_PWRITE: u16 = 1;
const IOCB_CMD_FSYNC: u16 = 2;
const IOCB_CMD_FDSYNC: u16 = 3;
const IOCB_CMD_PREADV: u16 = 7;
const IOCB_CMD_PWRITEV: u16 = 8;
const IOCB_FLAG_RESFD: u32 = 1;
/// The kernel's per-context ceiling (`aio-max-nr` is much larger; this bounds memory).
const MAX_EVENTS: u64 = 65536;

/// One AIO context.
#[derive(Debug)]
pub(crate) struct AioCtx {
    /// Wait-queue id for `io_getevents`.
    pub id: u64,
    pub nr: u64,
    /// Completed `struct io_event`s (32 bytes each), oldest first.
    pub done: VecDeque<[u8; 32]>,
    pub ring_len: u64,
}

impl Kernel {
    fn aio_key(&self, tid: Tid, ctx: u64) -> (u64, u64) {
        (self.mm(tid).borrow().id, ctx)
    }

    /// `io_setup(nr_events, ctxp)`.
    pub(super) fn sys_io_setup(&mut self, tid: Tid, nr: u64, ctxp: u64) -> SysResult {
        let mm = self.mm(tid);
        if mm.borrow().mem.read_u64(ctxp).map_err(|_| Errno::EFAULT)? != 0 {
            return Err(Errno::EINVAL);
        }
        if nr == 0 || nr > MAX_EVENTS {
            return Err(Errno::EINVAL);
        }
        // The ring: a 32-byte header plus one io_event per slot.
        let ring_len = page_up(32 + 32 * nr);
        let ring = {
            let mut mm = mm.borrow_mut();
            let at = mm.find_free(ring_len).ok_or(Errno::EAGAIN)?;
            mm.map(at, at + ring_len, wasmshim_cpu::Perms::RW, VmaKind::Anon)?;
            // id, nr, head, tail, magic (0: no user-space ring), compat, incompat, header len.
            let mut hdr = Vec::with_capacity(32);
            for v in [0u32, nr as u32, 0, 0, 0, 1, 0, 32] {
                hdr.extend_from_slice(&v.to_le_bytes());
            }
            mm.mem.poke(at, &hdr).map_err(|_| Errno::EFAULT)?;
            at
        };
        let id = self.new_object_id();
        let key = self.aio_key(tid, ring);
        self.aio.insert(
            key,
            AioCtx {
                id,
                nr,
                done: VecDeque::new(),
                ring_len,
            },
        );
        write_u64(mm.borrow_mut().mem.as_mut(), ctxp, ring)?;
        Ok(0)
    }

    /// `io_destroy(ctx)`.
    pub(super) fn sys_io_destroy(&mut self, tid: Tid, ctx: u64) -> SysResult {
        let key = self.aio_key(tid, ctx);
        let c = self.aio.remove(&key).ok_or(Errno::EINVAL)?;
        let _ = self.mm(tid).borrow_mut().unmap(ctx, ctx + c.ring_len);
        self.object_changed(c.id);
        Ok(0)
    }

    /// `io_submit(ctx, nr, iocbpp)`: runs each request now and queues its completion.
    pub(super) fn sys_io_submit(
        &mut self,
        tid: Tid,
        ctx: u64,
        nr: u64,
        iocbpp: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let key = self.aio_key(tid, ctx);
        if !self.aio.contains_key(&key) {
            return Err(Errno::EINVAL);
        }
        if nr as i64 <= 0 {
            return if nr == 0 { Ok(0) } else { Err(Errno::EINVAL) };
        }
        let mut submitted = 0;
        for i in 0..nr {
            match self.aio_one(tid, key, iocbpp + 8 * i, host) {
                Ok(()) => submitted += 1,
                Err(e) if submitted == 0 => return Err(e),
                Err(_) => break,
            }
        }
        if let Some(c) = self.aio.get(&key) {
            let id = c.id;
            self.object_changed(id);
        }
        Ok(submitted)
    }

    fn aio_one(
        &mut self,
        tid: Tid,
        key: (u64, u64),
        slot: u64,
        host: &mut dyn Host,
    ) -> Result<(), Errno> {
        let iocb_ptr = crate::uaccess::read_u64(self.mm(tid).borrow_mut().mem.as_mut(), slot)?;
        let cb: [u8; 64] = read_array(self.mm(tid).borrow_mut().mem.as_mut(), iocb_ptr)?;
        let u64_at = |o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().expect("8 bytes"));
        let u32_at = |o: usize| u32::from_le_bytes(cb[o..o + 4].try_into().expect("4 bytes"));
        let data = u64_at(0);
        let opcode = u16::from_le_bytes([cb[16], cb[17]]);
        let fd = u64::from(u32_at(20));
        let (buf, nbytes, offset) = (u64_at(24), u64_at(32), u64_at(40));
        let (flags, resfd) = (u32_at(56), u32_at(60));
        if u32_at(12) != 0 || u64_at(48) != 0 {
            // aio_rw_flags and aio_reserved2: not supported / must be zero.
            return Err(Errno::EINVAL);
        }
        if (offset as i64) < 0 {
            return Err(Errno::EINVAL);
        }
        self.file(tid, fd)?;
        let resfd_file = if flags & IOCB_FLAG_RESFD != 0 {
            let f = self.file(tid, u64::from(resfd))?;
            if !matches!(f.borrow().kind, FileKind::EventFd(_)) {
                return Err(Errno::EINVAL);
            }
            Some(f)
        } else {
            None
        };
        let res: i64 = {
            let r = match opcode {
                IOCB_CMD_PREAD => self.sys_read(tid, fd, buf, nbytes, Some(offset), host),
                IOCB_CMD_PWRITE => self.sys_write(tid, fd, buf, nbytes, Some(offset), host),
                IOCB_CMD_PREADV => self.sys_readv(tid, fd, buf, nbytes, Some(offset), host),
                IOCB_CMD_PWRITEV => self.sys_writev(tid, fd, buf, nbytes, Some(offset), 0, host),
                IOCB_CMD_FSYNC | IOCB_CMD_FDSYNC => Ok(0),
                _ => return Err(Errno::EINVAL),
            };
            match r {
                Ok(n) => n as i64,
                // A request that would block (a pipe or socket) completes with EAGAIN: the
                // completion is synchronous here.
                Err(super::BLOCK) => {
                    self.cancel_wait();
                    Errno::EAGAIN.ret() as i64
                }
                Err(e) => e.ret() as i64,
            }
        };
        let mut ev = [0u8; 32];
        ev[..8].copy_from_slice(&data.to_le_bytes());
        ev[8..16].copy_from_slice(&iocb_ptr.to_le_bytes());
        ev[16..24].copy_from_slice(&res.to_le_bytes());
        let c = self.aio.get_mut(&key).ok_or(Errno::EINVAL)?;
        if c.done.len() as u64 >= c.nr {
            return Err(Errno::EAGAIN);
        }
        c.done.push_back(ev);
        if let Some(f) = resfd_file {
            let kind = f.borrow().kind.clone();
            if let FileKind::EventFd(e) = kind {
                let _ = self.eventfd_write(&e, &1u64.to_le_bytes(), true);
            }
        }
        Ok(())
    }

    /// `io_getevents(ctx, min_nr, nr, events, timeout)` (and `io_pgetevents`, whose signal
    /// mask is ignored).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_io_getevents(
        &mut self,
        tid: Tid,
        ctx: u64,
        min_nr: u64,
        nr: u64,
        events: u64,
        timeout: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let key = self.aio_key(tid, ctx);
        let deadline = self.read_timeout(tid, timeout, host)?;
        let c = self.aio.get_mut(&key).ok_or(Errno::EINVAL)?;
        if (min_nr as i64) < 0 || (nr as i64) < 0 || min_nr > nr {
            return Err(Errno::EINVAL);
        }
        let ready = c.done.len() as u64;
        if ready < min_nr {
            let now = host.monotonic_ns();
            if deadline.is_none_or(|d| d > now) {
                let id = c.id;
                return Err(self.wait_for(WaitKey::Object(id), deadline, Some(0)));
            }
        }
        let n = ready.min(nr) as usize;
        let out: Vec<u8> = c.done.drain(..n).flatten().collect();
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), events, &out)?;
        Ok(n as u64)
    }

    /// `io_cancel`: every request has completed already.
    pub(super) fn sys_io_cancel(&mut self, tid: Tid, ctx: u64) -> SysResult {
        let key = self.aio_key(tid, ctx);
        let _known = self.aio.contains_key(&key);
        Err(Errno::EINVAL)
    }
}
