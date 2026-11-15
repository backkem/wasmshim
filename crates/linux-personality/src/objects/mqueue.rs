//! POSIX message queues (`mq_open`, `mq_unlink`, `mq_timedsend`, `mq_timedreceive`,
//! `mq_notify`, `mq_getsetattr`).
//!
//! Queues live in a kernel-wide name table (the `mqueue` filesystem without a mount point).
//! Messages are kept highest priority first, FIFO within a priority. Absolute timeouts are on
//! `CLOCK_REALTIME`, converted to the monotonic clock once (the clock is never stepped).

use std::cell::RefCell;
use std::rc::Rc;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{open, signal};
use wasmshim_linux_abi::structs::Timespec;

use super::{EPOLLIN, EPOLLOUT, EPOLLRDNORM, EPOLLWRNORM};
use crate::fd::{FileKind, FileRef, open_file};
use crate::host::Host;
use crate::kernel::{Blocker, Kernel, Pid, ThreadState, Tid, WaitKey};
use crate::siginfo::SigInfo;
use crate::syscall::SysResult;
use crate::uaccess::{read_array, read_bytes, read_cstr_max, write_bytes};

/// `MQ_PRIO_MAX`.
const MQ_PRIO_MAX: u64 = 32768;
/// `/proc/sys/fs/mqueue` defaults: the limits for unprivileged callers, and the defaults.
const MSG_MAX: i64 = 10;
const MSGSIZE_MAX: i64 = 8192;
/// Hard limits (root).
const HARD_MSGMAX: i64 = 65536;
const HARD_MSGSIZEMAX: i64 = 16 << 20;
const SIGEV_SIGNAL: i32 = 0;
const SIGEV_NONE: i32 = 1;

/// Who `mq_notify` asked to hear about a message arriving in an empty queue.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MqNotify {
    pid: Pid,
    how: i32,
    signo: u32,
    value: u64,
}

/// One queue.
#[derive(Debug)]
pub(crate) struct MQueue {
    /// Wait key of receivers (`id`) and of senders (`id + 1`).
    id: u64,
    maxmsg: i64,
    msgsize: i64,
    /// (priority, bytes), highest priority first.
    msgs: Vec<(u32, Vec<u8>)>,
    notify: Option<MqNotify>,
    generation: u64,
}

/// A shared queue.
pub(crate) type MqRef = Rc<RefCell<MQueue>>;

impl MQueue {
    /// Readiness and change generation.
    pub(crate) fn events(&self) -> (u32, u64) {
        let mut ev = 0;
        if !self.msgs.is_empty() {
            ev |= EPOLLIN | EPOLLRDNORM;
        }
        if (self.msgs.len() as i64) < self.maxmsg {
            ev |= EPOLLOUT | EPOLLWRNORM;
        }
        (ev, self.generation)
    }

    /// What `read` on a queue descriptor returns.
    pub(crate) fn status(&self) -> Vec<u8> {
        let qsize: usize = self.msgs.iter().map(|(_, m)| m.len()).sum();
        let (how, signo, pid) = self.notify.map_or((0, 0, 0), |n| (n.how, n.signo, n.pid));
        format!("QSIZE:{qsize:<10} NOTIFY:{how:<5} SIGNO:{signo:<5} NOTIFY_PID:{pid:<6}\n")
            .into_bytes()
    }
}

/// `struct mq_attr` (four longs).
fn encode_attr(flags: u64, q: &MQueue) -> [u8; 64] {
    let mut b = [0u8; 64];
    for (i, v) in [flags as i64, q.maxmsg, q.msgsize, q.msgs.len() as i64]
        .into_iter()
        .enumerate()
    {
        b[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
    b
}

impl Kernel {
    /// Reads and checks a queue name (without the leading `/` the libc strips).
    fn mq_name(&self, tid: Tid, ptr: u64) -> Result<Vec<u8>, Errno> {
        let mm = self.mm(tid);
        let mm = mm.borrow();
        let name = read_cstr_max(mm.mem.as_ref(), ptr, 256)?;
        if name.is_empty() {
            return Err(Errno::ENOENT);
        }
        if name.contains(&b'/') {
            return Err(Errno::EACCES);
        }
        Ok(name)
    }

    /// The queue behind `fd`.
    fn mq_of(&self, tid: Tid, fd: u64) -> Result<(MqRef, FileRef), Errno> {
        let file = self.file(tid, fd)?;
        let q = match &file.borrow().kind {
            FileKind::MQueue(q) => q.clone(),
            _ => return Err(Errno::EBADF),
        };
        Ok((q, file))
    }

    pub(crate) fn sys_mq_open(
        &mut self,
        tid: Tid,
        name: u64,
        oflag: u64,
        _mode: u64,
        attr: u64,
    ) -> SysResult {
        let name = self.mq_name(tid, name)?;
        if oflag & open::O_ACCMODE == open::O_ACCMODE {
            return Err(Errno::EINVAL);
        }
        let q = match self.mqueues.get(&name) {
            Some(q) => {
                if oflag & open::O_CREAT != 0 && oflag & open::O_EXCL != 0 {
                    return Err(Errno::EEXIST);
                }
                q.clone()
            }
            None => {
                if oflag & open::O_CREAT == 0 {
                    return Err(Errno::ENOENT);
                }
                let (maxmsg, msgsize) = if attr == 0 {
                    (MSG_MAX, MSGSIZE_MAX)
                } else {
                    let b: [u8; 32] = read_array(self.mm(tid).borrow_mut().mem.as_mut(), attr)?;
                    let word = |i: usize| {
                        i64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap_or_default())
                    };
                    let (maxmsg, msgsize) = (word(1), word(2));
                    let root = self.creds(tid).euid == 0;
                    let (lm, ls) = if root {
                        (HARD_MSGMAX, HARD_MSGSIZEMAX)
                    } else {
                        (MSG_MAX, MSGSIZE_MAX)
                    };
                    if maxmsg <= 0 || msgsize <= 0 || maxmsg > lm || msgsize > ls {
                        return Err(Errno::EINVAL);
                    }
                    (maxmsg, msgsize)
                };
                let id = self.new_object_id();
                // The senders' wait key.
                let _ = self.new_object_id();
                let q = Rc::new(RefCell::new(MQueue {
                    id,
                    maxmsg,
                    msgsize,
                    msgs: Vec::new(),
                    notify: None,
                    generation: 0,
                }));
                self.mqueues.insert(name.clone(), q.clone());
                q
            }
        };
        let mut path = b"/".to_vec();
        path.extend_from_slice(&name);
        let file = open_file(
            FileKind::MQueue(q),
            oflag & (open::O_ACCMODE | open::O_NONBLOCK),
            &path,
        );
        // Queue descriptors are always close-on-exec.
        Ok(self.fds(tid).borrow_mut().insert(file, true, 0)? as u64)
    }

    pub(crate) fn sys_mq_unlink(&mut self, tid: Tid, name: u64) -> SysResult {
        let name = self.mq_name(tid, name)?;
        self.mqueues.remove(&name).map(|_| 0).ok_or(Errno::ENOENT)
    }

    /// The monotonic deadline of an absolute `CLOCK_REALTIME` timeout, checked like Linux.
    fn mq_deadline(
        &mut self,
        tid: Tid,
        ptr: u64,
        host: &mut dyn Host,
    ) -> Result<Option<u64>, Errno> {
        if ptr == 0 {
            return Ok(None);
        }
        let ts = Timespec::from_bytes(&read_array(self.mm(tid).borrow_mut().mem.as_mut(), ptr)?);
        let abs = ts.to_nanos_saturating().ok_or(Errno::EINVAL)?;
        let now = host.monotonic_ns();
        Ok(Some(
            now.saturating_add(abs.saturating_sub(host.realtime_ns())),
        ))
    }

    /// Blocks on `key` until `deadline` (then `ETIMEDOUT`), or fails at once if it has passed.
    fn mq_wait(&mut self, key: u64, deadline: Option<u64>, now: u64) -> Errno {
        if deadline.is_some_and(|d| d <= now) {
            return Errno::ETIMEDOUT;
        }
        self.wait_for(WaitKey::Object(key), deadline, Some(Errno::ETIMEDOUT.ret()))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sys_mq_timedsend(
        &mut self,
        tid: Tid,
        fd: u64,
        msg: u64,
        len: u64,
        prio: u64,
        timeout: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if prio >= MQ_PRIO_MAX {
            return Err(Errno::EINVAL);
        }
        let deadline = self.mq_deadline(tid, timeout, host)?;
        let (q, file) = self.mq_of(tid, fd)?;
        let flags = file.borrow().flags;
        if flags & open::O_ACCMODE == open::O_RDONLY {
            return Err(Errno::EBADF);
        }
        // `msg_len` is a `size_t`: compared unsigned, 2^63 and above are just too large.
        if len > u64::try_from(q.borrow().msgsize).unwrap_or(0) {
            return Err(Errno::EMSGSIZE);
        }
        let data = read_bytes(self.mm(tid).borrow_mut().mem.as_mut(), msg, len as usize)?;
        let (full, id) = {
            let b = q.borrow();
            (b.msgs.len() as i64 >= b.maxmsg, b.id)
        };
        if full {
            if flags & open::O_NONBLOCK != 0 {
                return Err(Errno::EAGAIN);
            }
            let now = host.monotonic_ns();
            return Err(self.mq_wait(id + 1, deadline, now));
        }
        let was_empty = q.borrow().msgs.is_empty();
        {
            let mut b = q.borrow_mut();
            let prio = prio as u32;
            let at = b
                .msgs
                .iter()
                .position(|(p, _)| *p < prio)
                .unwrap_or(b.msgs.len());
            b.msgs.insert(at, (prio, data));
            b.generation += 1;
        }
        // A message into an empty queue notifies its registered process, unless a receiver
        // is already waiting for it.
        let receiver_waiting = self.core.tasks.values().any(|t| {
            matches!(t.state, ThreadState::Blocked(Blocker::Restart { key, .. })
                if key == WaitKey::Object(id))
        });
        if was_empty && !receiver_waiting {
            let notify = q.borrow_mut().notify.take();
            if let Some(n) = notify
                && n.how == SIGEV_SIGNAL
            {
                let me = self.pid_of(tid);
                let uid = self.creds(tid).ruid;
                let mut info = SigInfo::sender(n.signo, signal::SI_MESGQ, me, uid);
                info.raw[24..32].copy_from_slice(&n.value.to_le_bytes());
                let _ = self.queue_to_process(n.pid, info);
            }
        }
        self.object_changed(id);
        Ok(0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sys_mq_timedreceive(
        &mut self,
        tid: Tid,
        fd: u64,
        buf: u64,
        len: u64,
        prio_ptr: u64,
        timeout: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let deadline = self.mq_deadline(tid, timeout, host)?;
        let (q, file) = self.mq_of(tid, fd)?;
        let flags = file.borrow().flags;
        if flags & open::O_ACCMODE == open::O_WRONLY {
            return Err(Errno::EBADF);
        }
        if len < u64::try_from(q.borrow().msgsize).unwrap_or(0) {
            return Err(Errno::EMSGSIZE);
        }
        let id = q.borrow().id;
        if q.borrow().msgs.is_empty() {
            if flags & open::O_NONBLOCK != 0 {
                return Err(Errno::EAGAIN);
            }
            let now = host.monotonic_ns();
            return Err(self.mq_wait(id, deadline, now));
        }
        let (prio, data) = {
            let mut b = q.borrow_mut();
            b.generation += 1;
            b.msgs.remove(0)
        };
        let mm = self.mm(tid);
        write_bytes(mm.borrow_mut().mem.as_mut(), buf, &data)?;
        if prio_ptr != 0 {
            write_bytes(mm.borrow_mut().mem.as_mut(), prio_ptr, &prio.to_le_bytes())?;
        }
        // Senders waiting for room.
        self.object_changed(id + 1);
        self.wake(WaitKey::AnyIo, usize::MAX, u32::MAX);
        Ok(data.len() as u64)
    }

    pub(crate) fn sys_mq_notify(&mut self, tid: Tid, fd: u64, sevp: u64) -> SysResult {
        let (q, _) = self.mq_of(tid, fd)?;
        let pid = self.pid_of(tid);
        if sevp == 0 {
            let mut b = q.borrow_mut();
            if b.notify.is_some_and(|n| n.pid == pid) {
                b.notify = None;
            }
            return Ok(0);
        }
        let b: [u8; 64] = read_array(self.mm(tid).borrow_mut().mem.as_mut(), sevp)?;
        let word = |o: usize| i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let value = u64::from_le_bytes(b[0..8].try_into().unwrap_or_default());
        let (signo, how) = (word(8), word(12));
        match how {
            SIGEV_SIGNAL if signo > 0 && signo as u32 <= signal::NSIG => {}
            SIGEV_NONE => {}
            // SIGEV_THREAD needs a netlink notification socket (not modelled).
            _ => return Err(Errno::EINVAL),
        }
        let mut qb = q.borrow_mut();
        if qb.notify.is_some_and(|n| self.procs.contains_key(&n.pid)) {
            return Err(Errno::EBUSY);
        }
        qb.notify = Some(MqNotify {
            pid,
            how,
            signo: if how == SIGEV_SIGNAL { signo as u32 } else { 0 },
            value,
        });
        Ok(0)
    }

    pub(crate) fn sys_mq_getsetattr(&mut self, tid: Tid, fd: u64, new: u64, old: u64) -> SysResult {
        let (q, file) = self.mq_of(tid, fd)?;
        let new_flags = if new != 0 {
            let b: [u8; 32] = read_array(self.mm(tid).borrow_mut().mem.as_mut(), new)?;
            let flags = u64::from_le_bytes(b[0..8].try_into().unwrap_or_default());
            if flags & !open::O_NONBLOCK != 0 {
                return Err(Errno::EINVAL);
            }
            Some(flags)
        } else {
            None
        };
        if old != 0 {
            let flags = file.borrow().flags & open::O_NONBLOCK;
            let attr = encode_attr(flags, &q.borrow());
            write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), old, &attr[..32])?;
        }
        if let Some(flags) = new_flags {
            let mut f = file.borrow_mut();
            f.flags = (f.flags & !open::O_NONBLOCK) | flags;
        }
        Ok(0)
    }
}
