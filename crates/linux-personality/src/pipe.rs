//! Pipes: a bounded byte queue with reader and writer counts.
//!
//! Blocking follows the restart model: a reader of an empty pipe (or a writer of a full one)
//! waits on `WaitKey::Object(id)` and re-runs its syscall when the pipe changes.
//! Writes of up to `PIPE_BUF` bytes are atomic. Larger blocking writes return the partial
//! count instead of blocking mid-way (libc and busybox loop on short writes).

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{open, signal};

use crate::fd::{FileKind, open_file};
use crate::kernel::{Kernel, Tid, WaitKey};
use crate::syscall::SysResult;
use crate::uaccess::write_bytes;

/// Pipe capacity (Linux default: 16 pages).
pub(crate) const PIPE_CAPACITY: usize = 65536;
/// Writes up to this size are atomic.
pub(crate) const PIPE_BUF: usize = 4096;

/// One pipe.
#[derive(Debug)]
pub(crate) struct Pipe {
    pub id: u64,
    /// Change generation (edge-triggered epoll).
    pub generation: u64,
    pub buf: VecDeque<u8>,
    /// Open read-end descriptions.
    pub readers: u32,
    /// Open write-end descriptions.
    pub writers: u32,
}

pub(crate) type PipeRef = Rc<RefCell<Pipe>>;

impl Kernel {
    /// `pipe2(fds, flags)`.
    pub(crate) fn sys_pipe2(&mut self, tid: Tid, fds_ptr: u64, flags: u64) -> SysResult {
        if flags & !(open::O_CLOEXEC | open::O_NONBLOCK | open::O_DIRECT) != 0 {
            return Err(Errno::EINVAL);
        }
        let id = self.new_object_id();
        let pipe = Rc::new(RefCell::new(Pipe {
            id,
            generation: 0,
            buf: VecDeque::new(),
            readers: 1,
            writers: 1,
        }));
        let nb = flags & open::O_NONBLOCK;
        let cloexec = flags & open::O_CLOEXEC != 0;
        let r = open_file(
            FileKind::Pipe {
                pipe: pipe.clone(),
                write: false,
            },
            open::O_RDONLY | nb,
            b"pipe:",
        );
        let w = open_file(
            FileKind::Pipe { pipe, write: true },
            open::O_WRONLY | nb,
            b"pipe:",
        );
        let fds = self.fds(tid);
        let rfd = fds.borrow_mut().insert(r, cloexec, 0)?;
        let wfd = match fds.borrow_mut().insert(w, cloexec, 0) {
            Ok(fd) => fd,
            Err(e) => {
                let _ = fds.borrow_mut().remove(rfd);
                return Err(e);
            }
        };
        let mut out = [0u8; 8];
        out[..4].copy_from_slice(&rfd.to_le_bytes());
        out[4..].copy_from_slice(&wfd.to_le_bytes());
        if let Err(e) = write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), fds_ptr, &out) {
            for fd in [rfd, wfd] {
                if let Ok(slot) = fds.borrow_mut().remove(fd) {
                    drop(slot);
                }
            }
            return Err(e);
        }
        Ok(0)
    }

    /// Reads from a pipe's read end.
    pub(crate) fn pipe_read(
        &mut self,
        pipe: &PipeRef,
        buf: &mut [u8],
        nonblock: bool,
    ) -> Result<usize, Errno> {
        let (id, n) = {
            let mut p = pipe.borrow_mut();
            if p.buf.is_empty() {
                if p.writers == 0 || buf.is_empty() {
                    return Ok(0);
                }
                if nonblock {
                    return Err(Errno::EAGAIN);
                }
                let id = p.id;
                drop(p);
                return Err(self.wait_for(WaitKey::Object(id), None, None));
            }
            let n = buf.len().min(p.buf.len());
            for (dst, src) in buf.iter_mut().zip(p.buf.drain(..n)) {
                *dst = src;
            }
            p.generation += 1;
            (p.id, n)
        };
        self.wake(WaitKey::Object(id), usize::MAX, u32::MAX);
        Ok(n)
    }

    /// Writes to a pipe's write end. A pipe without readers raises `SIGPIPE` and fails with
    /// `EPIPE`.
    pub(crate) fn pipe_write(
        &mut self,
        tid: Tid,
        pipe: &PipeRef,
        data: &[u8],
        nonblock: bool,
    ) -> Result<usize, Errno> {
        let (id, n) = {
            let mut p = pipe.borrow_mut();
            if p.readers == 0 {
                drop(p);
                let _ = self.send_to_thread(tid, signal::SIGPIPE);
                return Err(Errno::EPIPE);
            }
            let space = PIPE_CAPACITY - p.buf.len();
            let fits = if data.len() <= PIPE_BUF {
                space >= data.len()
            } else {
                space > 0
            };
            if !fits {
                if nonblock {
                    return Err(Errno::EAGAIN);
                }
                let id = p.id;
                drop(p);
                return Err(self.wait_for(WaitKey::Object(id), None, None));
            }
            let n = data.len().min(space);
            p.buf.extend(&data[..n]);
            p.generation += 1;
            (p.id, n)
        };
        self.wake(WaitKey::Object(id), usize::MAX, u32::MAX);
        Ok(n)
    }

    /// The last description of one end closed.
    pub(crate) fn pipe_end_closed(&mut self, pipe: &PipeRef, write: bool) {
        let id = {
            let mut p = pipe.borrow_mut();
            if write {
                p.writers = p.writers.saturating_sub(1);
            } else {
                p.readers = p.readers.saturating_sub(1);
            }
            p.generation += 1;
            p.id
        };
        self.wake(WaitKey::Object(id), usize::MAX, u32::MAX);
    }
}
