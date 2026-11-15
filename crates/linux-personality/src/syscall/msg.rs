//! SysV message queues (`msgget`, `msgsnd`, `msgrcv`, `msgctl`).

use std::collections::VecDeque;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::ipc;

use super::SysResult;
use crate::host::Host;
use crate::ipc::{IPC_PRIVATE, IpcObject, IpcPerm, IpcTable};
use crate::kernel::{Kernel, Tid, WaitKey};
use crate::uaccess::{read_bytes, write_bytes};

/// Largest message.
const MSGMAX: u64 = 8192;
/// Default bytes per queue.
const MSGMNB: u64 = 16384;
/// Most queues.
const MSGMNI: u32 = 32000;
const MSG_NOERROR: u64 = 0o10000;
const MSG_EXCEPT: u64 = 0o20000;
const MSG_COPY: u64 = 0o40000;
const MSG_STAT: u64 = 11;
const MSG_INFO: u64 = 12;
const MSG_STAT_ANY: u64 = 13;
/// `sizeof(struct msqid64_ds)` on arm64.
const MSQID_DS_SIZE: usize = 120;

/// One queue.
#[derive(Clone, Debug)]
pub(crate) struct MsgQueue {
    perm: IpcPerm,
    /// Object id (wait key).
    id: u64,
    msgs: VecDeque<(i64, Vec<u8>)>,
    cbytes: u64,
    qbytes: u64,
    stime: i64,
    rtime: i64,
    ctime: i64,
    lspid: u32,
    lrpid: u32,
}

impl IpcObject for MsgQueue {
    fn perm(&self) -> &IpcPerm {
        &self.perm
    }
    fn perm_mut(&mut self) -> &mut IpcPerm {
        &mut self.perm
    }
}

/// The message queues.
pub(crate) type MsgTable = IpcTable<MsgQueue>;

impl Default for MsgTable {
    fn default() -> Self {
        Self::new(MSGMNI)
    }
}

fn now_s(host: &mut dyn Host) -> i64 {
    (host.realtime_ns() / 1_000_000_000) as i64
}

/// The index of the message `msgrcv` selects with `msgtyp` and `flags`.
fn select(msgs: &VecDeque<(i64, Vec<u8>)>, msgtyp: i64, flags: u64) -> Option<usize> {
    if flags & MSG_COPY != 0 {
        let i = usize::try_from(msgtyp).ok()?;
        return (i < msgs.len()).then_some(i);
    }
    match msgtyp {
        0 => (!msgs.is_empty()).then_some(0),
        t if t > 0 && flags & MSG_EXCEPT != 0 => msgs.iter().position(|(m, _)| *m != t),
        t if t > 0 => msgs.iter().position(|(m, _)| *m == t),
        t => {
            let limit = t.unsigned_abs();
            let mut best: Option<(usize, i64)> = None;
            for (i, (m, _)) in msgs.iter().enumerate() {
                if (*m as u64) <= limit && best.is_none_or(|(_, b)| *m < b) {
                    best = Some((i, *m));
                }
            }
            best.map(|(i, _)| i)
        }
    }
}

impl Kernel {
    pub(super) fn sys_msgget(
        &mut self,
        tid: Tid,
        key: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let key = key as i32;
        if key != IPC_PRIVATE
            && let Some(index) = self.msg.find_key(key)
        {
            if flags & ipc::IPC_CREAT != 0 && flags & ipc::IPC_EXCL != 0 {
                return Err(Errno::EEXIST);
            }
            return self.msg.id_of(index).map(|i| i as u64).ok_or(Errno::EINVAL);
        }
        if key != IPC_PRIVATE && flags & ipc::IPC_CREAT == 0 {
            return Err(Errno::ENOENT);
        }
        let (uid, gid) = {
            let c = self.creds(tid);
            (c.euid, c.egid)
        };
        let id = self.new_object_id();
        self.msg
            .insert(MsgQueue {
                perm: IpcPerm::new(key, uid, gid, flags),
                id,
                msgs: VecDeque::new(),
                cbytes: 0,
                qbytes: MSGMNB,
                stime: 0,
                rtime: 0,
                ctime: now_s(host),
                lspid: 0,
                lrpid: 0,
            })
            .map(|i| i as u64)
            .ok_or(Errno::ENOSPC)
    }

    pub(super) fn sys_msgsnd(
        &mut self,
        tid: Tid,
        id: u64,
        msgp: u64,
        size: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if size > MSGMAX {
            return Err(Errno::EINVAL);
        }
        let mm = self.mm(tid);
        let raw = read_bytes(mm.borrow_mut().mem.as_mut(), msgp, 8 + size as usize)?;
        let mtype = i64::from_le_bytes(raw[..8].try_into().unwrap_or_default());
        if mtype < 1 {
            return Err(Errno::EINVAL);
        }
        let index = self.msg.index_of(id as i32).ok_or(Errno::EINVAL)?;
        let q = &self.msg.objs[&index];
        let obj = q.id;
        if q.cbytes + size > q.qbytes || q.msgs.len() as u64 + 1 > q.qbytes {
            if flags & ipc::IPC_NOWAIT != 0 {
                return Err(Errno::EAGAIN);
            }
            return Err(self.wait_for(WaitKey::Object(obj), None, None));
        }
        let pid = self.pid_of(tid);
        let now = now_s(host);
        if let Some(q) = self.msg.objs.get_mut(&index) {
            q.msgs.push_back((mtype, raw[8..].to_vec()));
            q.cbytes += size;
            q.lspid = pid;
            q.stime = now;
        }
        self.object_changed(obj);
        Ok(0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_msgrcv(
        &mut self,
        tid: Tid,
        id: u64,
        msgp: u64,
        size: u64,
        msgtyp: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if (size as i64) < 0 {
            return Err(Errno::EINVAL);
        }
        if flags & MSG_COPY != 0 && (flags & ipc::IPC_NOWAIT == 0 || flags & MSG_EXCEPT != 0) {
            return Err(Errno::EINVAL);
        }
        let index = self.msg.index_of(id as i32).ok_or(Errno::EINVAL)?;
        let q = &self.msg.objs[&index];
        let obj = q.id;
        let Some(at) = select(&q.msgs, msgtyp as i64, flags) else {
            if flags & (ipc::IPC_NOWAIT | MSG_COPY) != 0 {
                return Err(Errno::ENOMSG);
            }
            return Err(self.wait_for(WaitKey::Object(obj), None, None));
        };
        let (mtype, text) = q.msgs[at].clone();
        if text.len() as u64 > size && flags & MSG_NOERROR == 0 {
            return Err(Errno::E2BIG);
        }
        let n = text.len().min(size as usize);
        let mut out = mtype.to_le_bytes().to_vec();
        out.extend_from_slice(&text[..n]);
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), msgp, &out)?;
        if flags & MSG_COPY == 0 {
            let pid = self.pid_of(tid);
            let now = now_s(host);
            if let Some(q) = self.msg.objs.get_mut(&index) {
                q.msgs.remove(at);
                q.cbytes -= text.len() as u64;
                q.lrpid = pid;
                q.rtime = now;
            }
            self.object_changed(obj);
        }
        Ok(n as u64)
    }

    fn msqid_ds(&self, index: u32) -> Vec<u8> {
        let q = &self.msg.objs[&index];
        let mut b = q.perm.encode();
        for v in [
            q.stime as u64,
            q.rtime as u64,
            q.ctime as u64,
            q.cbytes,
            q.msgs.len() as u64,
            q.qbytes,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&q.lspid.to_le_bytes());
        b.extend_from_slice(&q.lrpid.to_le_bytes());
        b.resize(MSQID_DS_SIZE, 0);
        b
    }

    pub(super) fn sys_msgctl(
        &mut self,
        tid: Tid,
        id: u64,
        cmd: u64,
        buf: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let cmd = cmd & !ipc::IPC_64;
        let mm = self.mm(tid);
        match cmd {
            ipc::IPC_INFO | MSG_INFO => {
                let (pool, map, tql) = if cmd == MSG_INFO {
                    let msgs: usize = self.msg.objs.values().map(|q| q.msgs.len()).sum();
                    let bytes: u64 = self.msg.objs.values().map(|q| q.cbytes).sum();
                    (self.msg.objs.len() as i32, msgs as i32, bytes as i32)
                } else {
                    let pool = (u64::from(MSGMNI) * MSGMNB / 1024) as i32;
                    (pool, MSGMNB as i32, MSGMNB as i32)
                };
                let mut b = Vec::new();
                for v in [
                    pool,
                    map,
                    MSGMAX as i32,
                    MSGMNB as i32,
                    MSGMNI as i32,
                    16,
                    tql,
                ] {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                b.extend_from_slice(&0xffffu16.to_le_bytes());
                b.resize(32, 0);
                write_bytes(mm.borrow_mut().mem.as_mut(), buf, &b)?;
                return Ok(self.msg.max_index());
            }
            MSG_STAT | MSG_STAT_ANY => {
                let index = u32::try_from(id).map_err(|_| Errno::EINVAL)?;
                if !self.msg.objs.contains_key(&index) {
                    return Err(Errno::EINVAL);
                }
                write_bytes(mm.borrow_mut().mem.as_mut(), buf, &self.msqid_ds(index))?;
                return self.msg.id_of(index).map(|i| i as u64).ok_or(Errno::EINVAL);
            }
            _ => {}
        }
        let index = self.msg.index_of(id as i32).ok_or(Errno::EINVAL)?;
        let obj = self.msg.objs[&index].id;
        match cmd {
            ipc::IPC_STAT => {
                write_bytes(mm.borrow_mut().mem.as_mut(), buf, &self.msqid_ds(index))?;
                Ok(0)
            }
            ipc::IPC_SET => {
                let b = read_bytes(mm.borrow_mut().mem.as_mut(), buf, MSQID_DS_SIZE)?;
                let qbytes = u64::from_le_bytes(b[88..96].try_into().unwrap_or_default());
                if qbytes > MSGMNB && self.creds(tid).euid != 0 {
                    return Err(Errno::EPERM);
                }
                let now = now_s(host);
                if let Some(q) = self.msg.objs.get_mut(&index) {
                    q.perm.set_from(&b);
                    q.qbytes = qbytes;
                    q.ctime = now;
                }
                // More room may let blocked senders go.
                self.object_changed(obj);
                Ok(0)
            }
            ipc::IPC_RMID => {
                self.msg.objs.remove(&index);
                self.fail_waiters(WaitKey::Object(obj), Errno::EIDRM);
                Ok(0)
            }
            _ => Err(Errno::EINVAL),
        }
    }
}
