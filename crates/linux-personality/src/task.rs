//! Task lifecycle: `clone` (threads, `fork`, `vfork`), exit, zombies and `wait4`.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{clone, open, signal, wait};

use crate::host::Host;
use crate::kernel::{
    Blocker, ExitStatus, Kernel, LinuxKey, LinuxThread, Park, Pid, Process, Thread, ThreadState,
    Tid, WaitKey, Zombie,
};
use crate::syscall::Flow;
use crate::uaccess::{write_bytes, write_u32};

/// `struct rusage` is 144 bytes on LP64.
const RUSAGE_SIZE: usize = 144;

impl Kernel {
    /// `clone(flags, stack, parent_tid, tls, child_tid)` (AArch64 argument order).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sys_clone(
        &mut self,
        tid: Tid,
        flags: u64,
        stack: u64,
        ptid: u64,
        tls: u64,
        ctid: u64,
        _host: &mut dyn Host,
    ) -> Flow {
        let thread = flags & clone::CLONE_THREAD != 0;
        let sighand = flags & clone::CLONE_SIGHAND != 0;
        let vm = flags & clone::CLONE_VM != 0;
        if (thread && !sighand) || (sighand && !vm) {
            return Flow::Ret(Errno::EINVAL.ret());
        }
        let unsupported = clone::CLONE_NEWNS | clone::CLONE_PIDFD | 0x7e00_0000 /* other namespaces */;
        if flags & unsupported != 0 {
            return Flow::Ret(Errno::EINVAL.ret());
        }
        // RLIMIT_NPROC: fork(2), "EAGAIN: ... the RLIMIT_NPROC soft resource limit ... was
        // reached". Linux counts per user; this machine has one.
        if self.core.tasks.len() as u64 >= self.cfg.limits.tasks {
            return Flow::Ret(Errno::EAGAIN.ret());
        }
        let Some(parent_thread) = self.core.tasks.get(&tid) else {
            return Flow::Ret(Errno::ESRCH.ret());
        };
        let parent_pid = parent_thread.pid;
        let sigmask = parent_thread.sigmask;
        let mut cpu = parent_thread.cpu.duplicate();
        // The copy carries the parent's counter: this thread's own work starts from here.
        let retired_base = cpu.retired();
        cpu.set_syscall_result(0);
        if stack != 0 {
            cpu.set_sp(stack);
        }
        if flags & clone::CLONE_SETTLS != 0 {
            cpu.set_thread_pointer(tls);
        }
        let child = self.alloc_pid();

        if !thread {
            let mm_id = self.alloc_mm_id();
            let parent = &self.procs[&parent_pid];
            let mm = if vm {
                parent.mm.clone()
            } else {
                Rc::new(RefCell::new(parent.mm.borrow().duplicate(mm_id)))
            };
            let fds = if flags & clone::CLONE_FILES != 0 {
                parent.fds.clone()
            } else {
                Rc::new(RefCell::new(parent.fds.borrow().clone()))
            };
            let sigactions = if sighand {
                parent.sigactions.clone()
            } else {
                Rc::new(RefCell::new(*parent.sigactions.borrow()))
            };
            let ppid = if flags & clone::CLONE_PARENT != 0 {
                parent.ppid
            } else {
                parent_pid
            };
            let proc_ = Process {
                ppid,
                pgid: parent.pgid,
                sid: parent.sid,
                mm,
                fds,
                cwd: parent.cwd,
                umask: parent.umask,
                sigactions,
                exe: parent.exe.clone(),
                comm: parent.comm.clone(),
                cmdline: parent.cmdline.clone(),
                environ: parent.environ.clone(),
                auxv: parent.auxv.clone(),
                threads: BTreeSet::new(),
                children: BTreeSet::new(),
                exit_signal: (flags & clone::CSIGNAL) as u32,
                pending: Default::default(),
                vfork_parent: (flags & clone::CLONE_VFORK != 0).then_some(tid),
                creds: parent.creds.clone(),
                itimers: Default::default(),
                timers: Default::default(),
                semundo: Default::default(),
            };
            self.procs.insert(child, proc_);
            if let Some(p) = self.procs.get_mut(&ppid) {
                p.children.insert(child);
            }
        }
        let child_pid = if thread { parent_pid } else { child };
        let mm = self.procs[&child_pid].mm.clone();
        if flags & clone::CLONE_CHILD_SETTID != 0 {
            let _ = write_u32(mm.borrow_mut().mem.as_mut(), ctid, child);
        }
        if flags & clone::CLONE_PARENT_SETTID != 0 {
            let pmm = self.procs[&parent_pid].mm.clone();
            let _ = write_u32(pmm.borrow_mut().mem.as_mut(), ptid, child);
        }
        let mut ext = LinuxThread::new(child_pid, sigmask);
        if flags & clone::CLONE_CHILD_CLEARTID != 0 {
            ext.clear_child_tid = ctid;
        }
        self.add_thread(Thread::new(child, cpu, retired_base, ext));
        if flags & clone::CLONE_VFORK != 0 && !thread {
            return Flow::Block(Blocker::Personal(Park::Vfork { child }));
        }
        Flow::Ret(u64::from(child))
    }

    /// Releases a thread blocked in `vfork` with the child's pid.
    pub(crate) fn finish_vfork(&mut self, parent: Tid, child: Pid) {
        if let Some(t) = self.core.tasks.get_mut(&parent)
            && t.state == ThreadState::Blocked(Blocker::Personal(Park::Vfork { child }))
        {
            t.cpu.set_syscall_result(u64::from(child));
            self.make_runnable(parent);
        }
    }

    /// Removes a thread (no exit-status bookkeeping): `clear_child_tid` is cleared and woken.
    pub(crate) fn drop_thread(&mut self, tid: Tid) {
        let Some(t) = self.core.tasks.remove(&tid) else {
            return;
        };
        self.core.retired_dead += t.own_retired();
        if let Some(p) = self.procs.get_mut(&t.pid) {
            p.threads.remove(&tid);
        }
        if t.clear_child_tid != 0
            && let Some(p) = self.procs.get(&t.pid)
        {
            let mm = p.mm.clone();
            let id = mm.borrow().id;
            if write_u32(mm.borrow_mut().mem.as_mut(), t.clear_child_tid, 0).is_ok() {
                self.wake(WaitKey::Futex(id, t.clear_child_tid), 1, u32::MAX);
            }
        }
    }

    /// `exit`: ends one thread; the last one ends the process.
    pub(crate) fn exit_thread(&mut self, tid: Tid, code: i32) {
        let Some(pid) = self.core.tasks.get(&tid).map(|t| t.pid) else {
            return;
        };
        let last = self.procs.get(&pid).is_none_or(|p| p.threads.len() <= 1);
        if last {
            self.exit_process(pid, ExitStatus::Code(code as u8));
        } else {
            self.drop_thread(tid);
        }
    }

    /// Ends a whole process: threads, descriptors, children are reparented to pid 1, and the
    /// parent gets a zombie to reap (unless it ignores `SIGCHLD`) and its exit signal.
    pub(crate) fn exit_process(&mut self, pid: Pid, status: ExitStatus) {
        let Some(threads) = self.procs.get(&pid).map(|p| p.threads.clone()) else {
            return;
        };
        for tid in threads {
            self.drop_thread(tid);
        }
        self.locks.drop_posix(pid);
        let Some(p) = self.procs.remove(&pid) else {
            return;
        };
        let fds: Vec<_> = p.fds.borrow().iter().map(|(_, f)| f.file.clone()).collect();
        let semundo = p.semundo.clone();
        let uid = p.creds.ruid;
        let (ppid, pgid, exit_signal, children, vfork_parent) = (
            p.ppid,
            p.pgid,
            p.exit_signal,
            p.children.clone(),
            p.vfork_parent,
        );
        let leader = p.sid == pid;
        drop(p);
        self.sem_exit(pid, semundo);
        for f in fds {
            self.release_file(&f);
        }
        // Its working directory may have been removed.
        self.reap_dirs();
        if leader {
            self.session_leader_exited(pid);
        }
        if let Some(parent) = vfork_parent {
            self.finish_vfork(parent, pid);
        }
        // Orphans go to init.
        for c in children {
            if let Some(cp) = self.procs.get_mut(&c) {
                cp.ppid = 1;
            }
            if let Some(z) = self.zombies.get_mut(&c) {
                z.ppid = 1;
            }
            if let Some(init) = self.procs.get_mut(&1) {
                init.children.insert(c);
            }
        }
        // pidfds of this process became readable.
        self.wake(WaitKey::AnyIo, usize::MAX, u32::MAX);
        if pid == 1 {
            self.set_init_exit(status);
            return;
        }
        let reap_now = self.procs.get(&ppid).is_none_or(|parent| {
            let chld = parent.sigactions.borrow()[signal::SIGCHLD as usize - 1];
            chld.handler == signal::SIG_IGN || chld.flags & signal::SA_NOCLDWAIT != 0
        });
        if reap_now {
            if let Some(parent) = self.procs.get_mut(&ppid) {
                parent.children.remove(&pid);
            }
        } else {
            self.zombies.insert(
                pid,
                Zombie {
                    ppid,
                    pgid,
                    uid,
                    status,
                },
            );
            self.wake(
                WaitKey::Personal(LinuxKey::Children(ppid)),
                usize::MAX,
                u32::MAX,
            );
        }
        if exit_signal != 0 {
            let (code, value) = status.child_code();
            let mut info = crate::siginfo::SigInfo::child(pid, uid, code, value);
            info.raw[0..4].copy_from_slice(&exit_signal.to_le_bytes());
            let _ = self.queue_to_process(ppid, info);
        }
    }

    /// `wait4(pid, wstatus, options, rusage)`.
    pub(crate) fn sys_wait4(
        &mut self,
        tid: Tid,
        pid: u64,
        wstatus: u64,
        options: u64,
        rusage: u64,
        resume_pc: u64,
    ) -> Flow {
        let me = self.pid_of(tid);
        let Some(proc_) = self.procs.get(&me) else {
            return Flow::Ret(Errno::ESRCH.ret());
        };
        let my_pgid = proc_.pgid;
        let target = pid as i64 as i32;
        let pgid_of = |k: &Kernel, c: Pid| {
            k.procs
                .get(&c)
                .map(|p| p.pgid)
                .or_else(|| k.zombies.get(&c).map(|z| z.pgid))
        };
        let matches = |k: &Kernel, c: Pid| match target {
            -1 => true,
            0 => pgid_of(k, c) == Some(my_pgid),
            t if t < -1 => pgid_of(k, c) == Some(t.unsigned_abs()),
            t => c == t as u32,
        };
        let children: Vec<Pid> = proc_.children.iter().copied().collect();
        if !children.iter().any(|&c| matches(self, c)) {
            return Flow::Ret(Errno::ECHILD.ret());
        }
        let done = children
            .iter()
            .copied()
            .find(|c| self.zombies.contains_key(c) && matches(self, *c));
        if let Some(c) = done {
            let status = self.zombies[&c].status;
            let mm = self.mm(tid);
            let mut mm = mm.borrow_mut();
            if wstatus != 0 && write_u32(mm.mem.as_mut(), wstatus, status.wait_status()).is_err() {
                return Flow::Ret(Errno::EFAULT.ret());
            }
            if rusage != 0 && write_bytes(mm.mem.as_mut(), rusage, &[0u8; RUSAGE_SIZE]).is_err() {
                return Flow::Ret(Errno::EFAULT.ret());
            }
            drop(mm);
            if options & wait::WNOWAIT == 0 {
                self.zombies.remove(&c);
                if let Some(p) = self.procs.get_mut(&me) {
                    p.children.remove(&c);
                }
            }
            return Flow::Ret(u64::from(c));
        }
        if options & wait::WNOHANG != 0 {
            return Flow::Ret(0);
        }
        Flow::Block(Blocker::Restart {
            key: WaitKey::Personal(LinuxKey::Children(me)),
            deadline: None,
            resume_pc,
            timeout_result: None,
        })
    }
}

/// Which children a `waitid` selects.
type ChildFilter = Box<dyn Fn(&Kernel, Pid) -> bool>;

impl Kernel {
    /// `waitid(idtype, id, infop, options, rusage)`. Only exits are reported (there is no
    /// job control, so `WSTOPPED`/`WCONTINUED` alone wait for nothing).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sys_waitid(
        &mut self,
        tid: Tid,
        idtype: u64,
        id: u64,
        infop: u64,
        options: u64,
        rusage: u64,
        resume_pc: u64,
    ) -> Flow {
        match self.waitid(tid, idtype, id, infop, options, rusage) {
            Ok(Some(v)) => Flow::Ret(v),
            Ok(None) => Flow::Block(Blocker::Restart {
                key: WaitKey::Personal(LinuxKey::Children(self.pid_of(tid))),
                deadline: None,
                resume_pc,
                timeout_result: None,
            }),
            Err(e) => Flow::Ret(e.ret()),
        }
    }

    /// `waitid`'s work: `Ok(None)` means "block".
    fn waitid(
        &mut self,
        tid: Tid,
        idtype: u64,
        id: u64,
        infop: u64,
        options: u64,
        rusage: u64,
    ) -> Result<Option<u64>, Errno> {
        let known = wait::WNOHANG
            | wait::WSTOPPED
            | wait::WEXITED
            | wait::WCONTINUED
            | wait::WNOWAIT
            | wait::WNOTHREAD
            | wait::WALL
            | wait::WCLONE;
        if options & !known != 0
            || options & (wait::WEXITED | wait::WSTOPPED | wait::WCONTINUED) == 0
        {
            return Err(Errno::EINVAL);
        }
        let me = self.pid_of(tid);
        let my_pgid = self.procs.get(&me).map_or(me, |p| p.pgid);
        let id = id as i32;
        let mut nohang = options & wait::WNOHANG != 0;
        let mut eagain = false;
        let wanted: ChildFilter = match idtype {
            wait::P_ALL => Box::new(|_, _| true),
            wait::P_PID if id > 0 => Box::new(move |_, c| c == id as u32),
            wait::P_PGID if id >= 0 => {
                let pg = if id == 0 { my_pgid } else { id as u32 };
                Box::new(move |k: &Kernel, c| {
                    k.procs
                        .get(&c)
                        .map(|p| p.pgid)
                        .or_else(|| k.zombies.get(&c).map(|z| z.pgid))
                        == Some(pg)
                })
            }
            wait::P_PIDFD if id >= 0 => {
                let (pid, file) = self.pidfd_target(tid, id as u64)?;
                if file.borrow().flags & open::O_NONBLOCK != 0 {
                    nohang = true;
                    eagain = true;
                }
                Box::new(move |_, c| c == pid)
            }
            _ => return Err(Errno::EINVAL),
        };
        let children: Vec<Pid> = self
            .procs
            .get(&me)
            .map(|p| p.children.iter().copied().collect())
            .unwrap_or_default();
        if !children.iter().any(|&c| wanted(self, c)) {
            return Err(Errno::ECHILD);
        }
        let mm = self.mm(tid);
        let done = (options & wait::WEXITED != 0)
            .then(|| {
                children
                    .iter()
                    .copied()
                    .find(|c| self.zombies.contains_key(c) && wanted(self, *c))
            })
            .flatten();
        let Some(c) = done else {
            if !nohang {
                return Ok(None);
            }
            if eagain {
                return Err(Errno::EAGAIN);
            }
            if infop != 0 {
                // Nothing to report: si_signo, si_errno, si_code, si_pid, si_uid, si_status
                // read as zero.
                write_bytes(mm.borrow_mut().mem.as_mut(), infop, &[0u8; 32])?;
            }
            return Ok(Some(0));
        };
        let z = self.zombies[&c];
        let (code, status) = z.status.child_code();
        if infop != 0 {
            let info = crate::siginfo::SigInfo::child(c, z.uid, code, status);
            write_bytes(mm.borrow_mut().mem.as_mut(), infop, &info.raw)?;
        }
        if rusage != 0 {
            write_bytes(mm.borrow_mut().mem.as_mut(), rusage, &[0u8; RUSAGE_SIZE])?;
        }
        if options & wait::WNOWAIT == 0 {
            self.zombies.remove(&c);
            if let Some(p) = self.procs.get_mut(&me) {
                p.children.remove(&c);
            }
        }
        Ok(Some(0))
    }
}
