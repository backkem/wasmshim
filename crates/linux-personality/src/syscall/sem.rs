//! SysV semaphores (`semget`, `semop`, `semtimedop`, `semctl`).
//!
//! A `semop` applies all its operations or none. One that cannot proceed blocks (restartably)
//! on the set; any change to the set wakes its waiters, which re-run their `semop` in run-queue
//! order (deterministic). `SEM_UNDO` adjustments are kept per process and applied at exit.

use std::collections::BTreeMap;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::ipc;
use wasmshim_linux_abi::structs::Timespec;

use super::SysResult;
use crate::host::Host;
use crate::ipc::{IPC_PRIVATE, IpcObject, IpcPerm, IpcTable};
use crate::kernel::{Blocker, Kernel, Pid, ThreadState, Tid, WaitKey};
use crate::uaccess::{read_array, read_bytes, write_bytes};

/// Largest semaphore value.
const SEMVMX: i32 = 32767;
/// Largest `SEM_UNDO` adjustment.
const SEMAEM: i32 = SEMVMX;
/// Most semaphores per set.
const SEMMSL: u64 = 32000;
/// Most operations per `semop`.
const SEMOPM: u64 = 500;
/// Most sets.
const SEMMNI: u32 = 32000;
/// Most semaphores in the system.
const SEMMNS: u32 = SEMMNI * SEMMSL as u32;
const SEM_UNDO: i16 = 0x1000;
const IPC_NOWAIT: i16 = 0o4000;
const GETPID: u64 = 11;
const GETVAL: u64 = 12;
const GETALL: u64 = 13;
const GETNCNT: u64 = 14;
const GETZCNT: u64 = 15;
const SETVAL: u64 = 16;
const SETALL: u64 = 17;
const SEM_STAT: u64 = 18;
const SEM_INFO: u64 = 19;
const SEM_STAT_ANY: u64 = 20;
/// `sizeof(struct semid64_ds)` on arm64.
const SEMID_DS_SIZE: usize = 88;

/// One semaphore.
#[derive(Clone, Copy, Debug, Default)]
struct Sem {
    val: i32,
    pid: u32,
}

/// One set.
#[derive(Clone, Debug)]
pub(crate) struct SemSet {
    perm: IpcPerm,
    /// Object id: the wait key of the set's waiters, and the `SEM_UNDO` key.
    id: u64,
    sems: Vec<Sem>,
    otime: i64,
    ctime: i64,
    /// What each blocked thread waits for: (semaphore, waits for zero).
    waiters: BTreeMap<Tid, (u16, bool)>,
}

impl IpcObject for SemSet {
    fn perm(&self) -> &IpcPerm {
        &self.perm
    }
    fn perm_mut(&mut self) -> &mut IpcPerm {
        &mut self.perm
    }
}

/// The semaphore sets.
pub(crate) type SemTable = IpcTable<SemSet>;

impl Default for SemTable {
    fn default() -> Self {
        Self::new(SEMMNI)
    }
}

/// `SEM_UNDO` adjustments of one process: set object id -> semaphore -> adjustment.
pub(crate) type SemUndo = BTreeMap<u64, BTreeMap<u16, i32>>;

/// One `struct sembuf`.
#[derive(Clone, Copy, Debug)]
struct SemBuf {
    num: u16,
    op: i16,
    flg: i16,
}

/// The outcome of trying a `semop`.
enum Try {
    Done(Vec<i32>),
    Block(u16, bool),
}

fn now_s(host: &mut dyn Host) -> i64 {
    (host.realtime_ns() / 1_000_000_000) as i64
}

impl Kernel {
    pub(super) fn sys_semget(
        &mut self,
        tid: Tid,
        key: u64,
        nsems: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let key = key as i32;
        if nsems > SEMMSL {
            return Err(Errno::EINVAL);
        }
        if key != IPC_PRIVATE
            && let Some(index) = self.sem.find_key(key)
        {
            if flags & ipc::IPC_CREAT != 0 && flags & ipc::IPC_EXCL != 0 {
                return Err(Errno::EEXIST);
            }
            if nsems > self.sem.objs[&index].sems.len() as u64 {
                return Err(Errno::EINVAL);
            }
            return self.sem.id_of(index).map(|i| i as u64).ok_or(Errno::EINVAL);
        }
        if key != IPC_PRIVATE && flags & ipc::IPC_CREAT == 0 {
            return Err(Errno::ENOENT);
        }
        if nsems == 0 {
            return Err(Errno::EINVAL);
        }
        let total: u64 = self.sem.objs.values().map(|s| s.sems.len() as u64).sum();
        if total + nsems > u64::from(SEMMNS) {
            return Err(Errno::ENOSPC);
        }
        let (uid, gid) = {
            let c = self.creds(tid);
            (c.euid, c.egid)
        };
        let id = self.new_object_id();
        self.sem
            .insert(SemSet {
                perm: IpcPerm::new(key, uid, gid, flags),
                id,
                sems: vec![Sem::default(); nsems as usize],
                otime: 0,
                ctime: now_s(host),
                waiters: BTreeMap::new(),
            })
            .map(|i| i as u64)
            .ok_or(Errno::ENOSPC)
    }

    /// Tries `ops` on `set` for process `pid` (all or nothing).
    fn sem_try(set: &SemSet, ops: &[SemBuf], undo: &BTreeMap<u16, i32>) -> Result<Try, Errno> {
        let mut vals: Vec<i32> = set.sems.iter().map(|s| s.val).collect();
        let mut adj = undo.clone();
        for o in ops {
            let n = usize::from(o.num);
            let v = vals[n];
            let op = i32::from(o.op);
            if op > 0 {
                if v + op > SEMVMX {
                    return Err(Errno::ERANGE);
                }
                vals[n] = v + op;
            } else if op == 0 {
                if v != 0 {
                    if o.flg & IPC_NOWAIT != 0 {
                        return Err(Errno::EAGAIN);
                    }
                    return Ok(Try::Block(o.num, true));
                }
            } else if v < -op {
                if o.flg & IPC_NOWAIT != 0 {
                    return Err(Errno::EAGAIN);
                }
                return Ok(Try::Block(o.num, false));
            } else {
                vals[n] = v + op;
            }
            if o.flg & SEM_UNDO != 0 {
                let a = adj.entry(o.num).or_insert(0);
                let next = *a - op;
                if !(-SEMAEM - 1..=SEMAEM).contains(&next) {
                    return Err(Errno::ERANGE);
                }
                *a = next;
            }
        }
        Ok(Try::Done(vals))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_semtimedop(
        &mut self,
        tid: Tid,
        id: u64,
        sops: u64,
        nsops: u64,
        timeout: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if nsops == 0 {
            return Err(Errno::EINVAL);
        }
        if nsops > SEMOPM {
            return Err(Errno::E2BIG);
        }
        let mm = self.mm(tid);
        let raw = read_bytes(mm.borrow_mut().mem.as_mut(), sops, 6 * nsops as usize)?;
        let ops: Vec<SemBuf> = raw
            .as_chunks::<6>()
            .0
            .iter()
            .map(|c| SemBuf {
                num: u16::from_le_bytes([c[0], c[1]]),
                op: i16::from_le_bytes([c[2], c[3]]),
                flg: i16::from_le_bytes([c[4], c[5]]),
            })
            .collect();
        let timeout_ns = if timeout == 0 {
            None
        } else {
            let ts = Timespec::from_bytes(&read_array(mm.borrow_mut().mem.as_mut(), timeout)?);
            Some(ts.to_nanos_saturating().ok_or(Errno::EINVAL)?)
        };
        let index = self.sem.index_of(id as i32).ok_or(Errno::EINVAL)?;
        let set = &self.sem.objs[&index];
        if ops.iter().any(|o| usize::from(o.num) >= set.sems.len()) {
            return Err(Errno::EFBIG);
        }
        let pid = self.pid_of(tid);
        let obj = set.id;
        let undo = self
            .procs
            .get(&pid)
            .and_then(|p| p.semundo.get(&obj))
            .cloned()
            .unwrap_or_default();
        let outcome = Self::sem_try(set, &ops, &undo);
        if let Some(s) = self.sem.objs.get_mut(&index) {
            s.waiters.remove(&tid);
        }
        match outcome? {
            Try::Done(vals) => {
                let now = now_s(host);
                if let Some(s) = self.sem.objs.get_mut(&index) {
                    for (i, v) in vals.into_iter().enumerate() {
                        s.sems[i].val = v;
                    }
                    for o in &ops {
                        s.sems[usize::from(o.num)].pid = pid;
                    }
                    s.otime = now;
                }
                if ops.iter().any(|o| o.flg & SEM_UNDO != 0)
                    && let Some(p) = self.procs.get_mut(&pid)
                {
                    let adj = p.semundo.entry(obj).or_default();
                    for o in ops.iter().filter(|o| o.flg & SEM_UNDO != 0) {
                        *adj.entry(o.num).or_insert(0) -= i32::from(o.op);
                    }
                }
                if let Some(t) = self.core.tasks.get_mut(&tid) {
                    t.timed = None;
                }
                self.object_changed(obj);
                Ok(0)
            }
            Try::Block(num, zero) => {
                let now = host.monotonic_ns();
                let deadline = self.restart_deadline(tid, timeout_ns, now);
                if deadline.is_some_and(|d| d <= now) {
                    if let Some(t) = self.core.tasks.get_mut(&tid) {
                        t.timed = None;
                    }
                    return Err(Errno::EAGAIN);
                }
                if let Some(s) = self.sem.objs.get_mut(&index) {
                    s.waiters.insert(tid, (num, zero));
                }
                Err(self.wait_for(WaitKey::Object(obj), deadline, Some(Errno::EAGAIN.ret())))
            }
        }
    }

    /// Threads blocked in `semop` on semaphore `num` of the set in slot `index`, waiting for
    /// zero (`GETZCNT`) or for an increase (`GETNCNT`).
    fn sem_count(&self, index: u32, num: u16, zero: bool) -> u64 {
        let Some(set) = self.sem.objs.get(&index) else {
            return 0;
        };
        set.waiters
            .iter()
            .filter(|(_, w)| **w == (num, zero))
            .filter(|(tid, _)| {
                self.core.tasks.get(tid).is_some_and(|t| {
                    matches!(t.state, ThreadState::Blocked(Blocker::Restart { key, .. })
                        if key == WaitKey::Object(set.id))
                })
            })
            .count() as u64
    }

    fn semid_ds(&self, index: u32) -> Vec<u8> {
        let s = &self.sem.objs[&index];
        let mut b = s.perm.encode();
        for v in [s.otime as u64, s.ctime as u64, s.sems.len() as u64] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.resize(SEMID_DS_SIZE, 0);
        b
    }

    /// Clears the `SEM_UNDO` adjustments of every process for the set `obj` (all semaphores,
    /// or one).
    fn sem_clear_undo(&mut self, obj: u64, num: Option<u16>) {
        for p in self.procs.values_mut() {
            match num {
                None => {
                    p.semundo.remove(&obj);
                }
                Some(n) => {
                    if let Some(a) = p.semundo.get_mut(&obj) {
                        a.remove(&n);
                    }
                }
            }
        }
    }

    pub(super) fn sys_semctl(
        &mut self,
        tid: Tid,
        id: u64,
        num: u64,
        cmd: u64,
        arg: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let cmd = cmd & !ipc::IPC_64;
        let mm = self.mm(tid);
        match cmd {
            ipc::IPC_INFO | SEM_INFO => {
                let used = self.sem.objs.len() as i32;
                let total: i32 = self.sem.objs.values().map(|s| s.sems.len() as i32).sum();
                let semmns = SEMMNS as i32;
                let info: [i32; 10] = [
                    semmns,
                    SEMMNI as i32,
                    semmns,
                    semmns,
                    SEMMSL as i32,
                    SEMOPM as i32,
                    SEMOPM as i32,
                    if cmd == SEM_INFO { used } else { 20 },
                    SEMVMX,
                    if cmd == SEM_INFO { total } else { SEMAEM },
                ];
                let b: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
                write_bytes(mm.borrow_mut().mem.as_mut(), arg, &b)?;
                return Ok(self.sem.max_index());
            }
            SEM_STAT | SEM_STAT_ANY => {
                let index = u32::try_from(id).map_err(|_| Errno::EINVAL)?;
                if !self.sem.objs.contains_key(&index) {
                    return Err(Errno::EINVAL);
                }
                write_bytes(mm.borrow_mut().mem.as_mut(), arg, &self.semid_ds(index))?;
                return self.sem.id_of(index).map(|i| i as u64).ok_or(Errno::EINVAL);
            }
            _ => {}
        }
        let index = self.sem.index_of(id as i32).ok_or(Errno::EINVAL)?;
        let nsems = self.sem.objs[&index].sems.len();
        let obj = self.sem.objs[&index].id;
        let one = || -> Result<usize, Errno> {
            let n = num as usize;
            if n >= nsems {
                return Err(Errno::EINVAL);
            }
            Ok(n)
        };
        let now = now_s(host);
        match cmd {
            ipc::IPC_STAT => {
                write_bytes(mm.borrow_mut().mem.as_mut(), arg, &self.semid_ds(index))?;
                Ok(0)
            }
            ipc::IPC_SET => {
                let b = read_bytes(mm.borrow_mut().mem.as_mut(), arg, 24)?;
                if let Some(s) = self.sem.objs.get_mut(&index) {
                    s.perm.set_from(&b);
                    s.ctime = now;
                }
                Ok(0)
            }
            ipc::IPC_RMID => {
                self.sem.objs.remove(&index);
                self.sem_clear_undo(obj, None);
                self.fail_waiters(WaitKey::Object(obj), Errno::EIDRM);
                Ok(0)
            }
            GETVAL => Ok(self.sem.objs[&index].sems[one()?].val as u64),
            GETPID => Ok(u64::from(self.sem.objs[&index].sems[one()?].pid)),
            GETNCNT => Ok(self.sem_count(index, one()? as u16, false)),
            GETZCNT => Ok(self.sem_count(index, one()? as u16, true)),
            SETVAL => {
                let n = one()?;
                let v = arg as i32;
                if !(0..=SEMVMX).contains(&v) {
                    return Err(Errno::ERANGE);
                }
                let pid = self.pid_of(tid);
                if let Some(s) = self.sem.objs.get_mut(&index) {
                    s.sems[n] = Sem { val: v, pid };
                    s.ctime = now;
                }
                self.sem_clear_undo(obj, Some(n as u16));
                self.object_changed(obj);
                Ok(0)
            }
            GETALL => {
                let b: Vec<u8> = self.sem.objs[&index]
                    .sems
                    .iter()
                    .flat_map(|s| (s.val as u16).to_le_bytes())
                    .collect();
                write_bytes(mm.borrow_mut().mem.as_mut(), arg, &b)?;
                Ok(0)
            }
            SETALL => {
                let b = read_bytes(mm.borrow_mut().mem.as_mut(), arg, 2 * nsems)?;
                let vals: Vec<i32> = b
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i32::from(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                if vals.iter().any(|&v| v > SEMVMX) {
                    return Err(Errno::ERANGE);
                }
                let pid = self.pid_of(tid);
                if let Some(s) = self.sem.objs.get_mut(&index) {
                    for (sem, v) in s.sems.iter_mut().zip(vals) {
                        *sem = Sem { val: v, pid };
                    }
                    s.ctime = now;
                }
                self.sem_clear_undo(obj, None);
                self.object_changed(obj);
                Ok(0)
            }
            _ => Err(Errno::EINVAL),
        }
    }

    /// Applies an exiting process's `SEM_UNDO` adjustments.
    pub(crate) fn sem_exit(&mut self, pid: Pid, undo: SemUndo) {
        for (obj, adj) in undo {
            let Some(set) = self.sem.objs.values_mut().find(|s| s.id == obj) else {
                continue;
            };
            for (n, a) in adj {
                if let Some(sem) = set.sems.get_mut(usize::from(n)) {
                    sem.val = (sem.val + a).clamp(0, SEMVMX);
                    sem.pid = pid;
                }
            }
            self.object_changed(obj);
        }
    }
}
