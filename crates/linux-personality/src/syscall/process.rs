//! Process, identity, resource and signal syscalls.

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{prctl, rlimit, signal};
use wasmshim_linux_abi::structs::{KernelSigaction, Rlimit, Timeval};

use super::{Flow, SysResult};
use crate::host::Host;
use crate::kernel::{Kernel, Pid, Spawn, Tid};
use crate::mm::STACK_SIZE;
use crate::uaccess::{read_array, read_path, read_str_array, write_bytes, write_u32, write_u64};

/// `struct rusage` is 144 bytes on LP64.
const RUSAGE_SIZE: usize = 144;

impl Kernel {
    pub(super) fn sys_execve(
        &mut self,
        tid: Tid,
        path: u64,
        argv: u64,
        envp: u64,
        host: &mut dyn Host,
    ) -> Flow {
        let path = match read_path(self.mm(tid).borrow_mut().mem.as_mut(), path) {
            Ok(p) => p,
            Err(e) => return Flow::Ret(e.ret()),
        };
        self.exec_with(tid, path, argv, envp, None, host)
    }

    /// `execve` of `path` (relative to `start`, or the file `start` names).
    pub(super) fn exec_with(
        &mut self,
        tid: Tid,
        path: Vec<u8>,
        argv: u64,
        envp: u64,
        start: Option<crate::exec::ExecStart>,
        host: &mut dyn Host,
    ) -> Flow {
        let args = (|| {
            let mm = self.mm(tid);
            let mut mm = mm.borrow_mut();
            let argv = read_str_array(mm.mem.as_mut(), argv)?;
            let envp = read_str_array(mm.mem.as_mut(), envp)?;
            Ok::<_, Errno>((argv, envp))
        })();
        let (argv, envp) = match args {
            Ok(a) => a,
            Err(e) => return Flow::Ret(e.ret()),
        };
        let pid = self.pid_of(tid);
        let Some(thread) = self.core.tasks.get_mut(&tid) else {
            return Flow::Ret(Errno::ESRCH.ret());
        };
        let placeholder = self.frontend.new_cpu();
        let before = thread.own_retired();
        let mut cpu = std::mem::replace(&mut thread.cpu, placeholder);
        let result = self.exec_image(
            pid,
            Some(tid),
            &mut cpu,
            Spawn { path, argv, envp },
            start,
            host,
        );
        if let Some(thread) = self.core.tasks.get_mut(&tid) {
            thread.cpu = cpu;
            if result.is_ok() {
                // A fresh CPU (counter 0): keep the work done before the exec.
                thread.retired_base = 0;
                self.core.retired_dead += before;
                thread.clear_child_tid = 0;
                thread.altstack = [0, 0, 2];
            }
        }
        match result {
            Ok(()) => Flow::Exec,
            Err(e) => Flow::Ret(e.ret()),
        }
    }

    pub(super) fn sys_set_tid_address(&mut self, tid: Tid, ptr: u64) -> SysResult {
        if let Some(t) = self.core.tasks.get_mut(&tid) {
            t.clear_child_tid = ptr;
        }
        Ok(u64::from(tid))
    }

    fn target_pid(&self, tid: Tid, pid: u64) -> Result<Pid, Errno> {
        let pid = pid as i32;
        if pid == 0 {
            return Ok(self.pid_of(tid));
        }
        let pid = u32::try_from(pid).map_err(|_| Errno::EINVAL)?;
        if self.procs.contains_key(&pid) {
            Ok(pid)
        } else {
            Err(Errno::ESRCH)
        }
    }

    pub(super) fn sys_getpgid(&mut self, tid: Tid, pid: u64) -> SysResult {
        let pid = self.target_pid(tid, pid)?;
        Ok(u64::from(self.procs[&pid].pgid))
    }

    pub(super) fn sys_setpgid(&mut self, tid: Tid, pid: u64, pgid: u64) -> SysResult {
        let pid = self.target_pid(tid, pid)?;
        let pgid = if pgid == 0 {
            pid
        } else {
            u32::try_from(pgid).map_err(|_| Errno::EINVAL)?
        };
        if let Some(p) = self.procs.get_mut(&pid) {
            p.pgid = pgid;
        }
        Ok(0)
    }

    pub(super) fn sys_setsid(&mut self, tid: Tid) -> SysResult {
        let pid = self.pid_of(tid);
        let p = self.procs.get_mut(&pid).ok_or(Errno::ESRCH)?;
        if p.pgid == pid {
            return Err(Errno::EPERM);
        }
        p.sid = pid;
        p.pgid = pid;
        Ok(u64::from(pid))
    }

    pub(super) fn sys_prctl(&mut self, tid: Tid, option: u64, arg: u64) -> SysResult {
        let pid = self.pid_of(tid);
        match option {
            prctl::PR_SET_NAME => {
                let name = crate::uaccess::read_cstr_max(
                    self.mm(tid).borrow_mut().mem.as_mut(),
                    arg,
                    4096,
                )?;
                let name = name[..name.len().min(15)].to_vec();
                if let Some(t) = self.core.tasks.get_mut(&tid) {
                    t.name.clone_from(&name);
                }
                if let Some(p) = self.procs.get_mut(&pid) {
                    p.comm = name;
                }
                Ok(0)
            }
            prctl::PR_GET_NAME => {
                let mut name = self.procs[&pid].comm.clone();
                name.resize(16, 0);
                write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), arg, &name)?;
                Ok(0)
            }
            prctl::PR_SET_PDEATHSIG | prctl::PR_SET_DUMPABLE => Ok(0),
            prctl::PR_GET_DUMPABLE => Ok(1),
            _ => Err(Errno::EINVAL),
        }
    }

    pub(super) fn sys_prlimit64(
        &mut self,
        tid: Tid,
        pid: u64,
        resource: u64,
        _new: u64,
        old: u64,
    ) -> SysResult {
        self.target_pid(tid, pid)?;
        let lim = match resource {
            rlimit::RLIMIT_STACK => Rlimit {
                cur: STACK_SIZE,
                max: rlimit::RLIM_INFINITY,
            },
            rlimit::RLIMIT_NOFILE => Rlimit {
                cur: crate::fd::MAX_FDS as u64,
                max: crate::fd::MAX_FDS as u64,
            },
            rlimit::RLIMIT_CORE => Rlimit {
                cur: 0,
                max: rlimit::RLIM_INFINITY,
            },
            rlimit::RLIMIT_NPROC => Rlimit {
                cur: self.cfg.limits.tasks,
                max: self.cfg.limits.tasks,
            },
            r if r < 16 => Rlimit {
                cur: rlimit::RLIM_INFINITY,
                max: rlimit::RLIM_INFINITY,
            },
            _ => return Err(Errno::EINVAL),
        };
        if old != 0 {
            write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), old, &lim.to_bytes())?;
        }
        Ok(0)
    }

    pub(super) fn sys_getrusage(&mut self, tid: Tid, buf: u64) -> SysResult {
        let mut b = [0u8; RUSAGE_SIZE];
        let cpu_ns = self
            .core
            .tasks
            .get(&tid)
            .map_or(0, |t| t.own_retired())
            .saturating_mul(self.cfg.ns_per_insn.max(1));
        b[..16].copy_from_slice(&Timeval::from_nanos(cpu_ns).to_bytes());
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &b)?;
        Ok(0)
    }

    pub(super) fn sys_sched_getaffinity(&mut self, tid: Tid, len: u64, mask: u64) -> SysResult {
        if len < 8 || !len.is_multiple_of(8) {
            return Err(Errno::EINVAL);
        }
        // One virtual CPU.
        write_u64(self.mm(tid).borrow_mut().mem.as_mut(), mask, 1)?;
        Ok(8)
    }

    pub(super) fn sys_getcpu(&mut self, tid: Tid, cpu: u64, node: u64) -> SysResult {
        let mm = self.mm(tid);
        let mut mm = mm.borrow_mut();
        for p in [cpu, node] {
            if p != 0 {
                write_u32(mm.mem.as_mut(), p, 0)?;
            }
        }
        Ok(0)
    }

    pub(super) fn sys_rt_sigaction(
        &mut self,
        tid: Tid,
        sig: u64,
        act: u64,
        oact: u64,
        setsize: u64,
    ) -> SysResult {
        if setsize != 8 || sig == 0 || sig > u64::from(signal::NSIG) {
            return Err(Errno::EINVAL);
        }
        let sig = sig as u32;
        let pid = self.pid_of(tid);
        let actions = self.procs[&pid].sigactions.clone();
        let new = if act != 0 {
            if sig == signal::SIGKILL || sig == signal::SIGSTOP {
                return Err(Errno::EINVAL);
            }
            Some(KernelSigaction::from_bytes(&read_array(
                self.mm(tid).borrow_mut().mem.as_mut(),
                act,
            )?))
        } else {
            None
        };
        let slot = sig as usize - 1;
        let old = actions.borrow()[slot];
        if oact != 0 {
            write_bytes(
                self.mm(tid).borrow_mut().mem.as_mut(),
                oact,
                &old.to_bytes(),
            )?;
        }
        if let Some(mut new) = new {
            new.mask &= !((1 << (signal::SIGKILL - 1)) | (1 << (signal::SIGSTOP - 1)));
            actions.borrow_mut()[slot] = new;
            if new.handler == signal::SIG_IGN
                || (new.handler == signal::SIG_DFL && crate::signal::default_ignored(sig))
            {
                // Pending instances of a signal that is now ignored are dropped.
                self.discard_pending(pid, 1 << (sig - 1));
            }
        }
        Ok(0)
    }

    pub(super) fn sys_rt_sigprocmask(
        &mut self,
        tid: Tid,
        how: u64,
        set: u64,
        oset: u64,
        setsize: u64,
    ) -> SysResult {
        if setsize != 8 {
            return Err(Errno::EINVAL);
        }
        let old = self.core.tasks.get(&tid).map_or(0, |t| t.sigmask);
        let new = if set != 0 {
            let s = crate::uaccess::read_u64(self.mm(tid).borrow_mut().mem.as_mut(), set)?;
            let unblockable = (1 << (signal::SIGKILL - 1)) | (1 << (signal::SIGSTOP - 1));
            Some(
                match how {
                    signal::SIG_BLOCK => old | s,
                    signal::SIG_UNBLOCK => old & !s,
                    signal::SIG_SETMASK => s,
                    _ => return Err(Errno::EINVAL),
                } & !unblockable,
            )
        } else {
            None
        };
        if oset != 0 {
            write_u64(self.mm(tid).borrow_mut().mem.as_mut(), oset, old)?;
        }
        if let (Some(new), Some(t)) = (new, self.core.tasks.get_mut(&tid)) {
            t.sigmask = new;
        }
        Ok(0)
    }

    pub(super) fn sys_sigaltstack(&mut self, tid: Tid, ss: u64, old: u64) -> SysResult {
        let current = self.core.tasks.get(&tid).map_or([0, 0, 2], |t| t.altstack);
        if old != 0 {
            let mut b = [0u8; 24];
            b[..8].copy_from_slice(&current[0].to_le_bytes());
            b[8..12].copy_from_slice(&(current[2] as u32).to_le_bytes());
            b[16..].copy_from_slice(&current[1].to_le_bytes());
            write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), old, &b)?;
        }
        if ss != 0 {
            let b: [u8; 24] = read_array(self.mm(tid).borrow_mut().mem.as_mut(), ss)?;
            let sp = u64::from_le_bytes(b[..8].try_into().unwrap_or_default());
            let flags = u32::from_le_bytes(b[8..12].try_into().unwrap_or_default());
            let size = u64::from_le_bytes(b[16..].try_into().unwrap_or_default());
            // kernel/signal.c, `do_sigaltstack`: SS_ONSTACK is taken as 0 (older programs pass
            // it); SS_AUTODISARM is not supported here.
            let flags = match flags {
                signal::SS_ONSTACK => 0,
                f if f & !signal::SS_DISABLE != 0 => return Err(Errno::EINVAL),
                f => f,
            };
            if flags & signal::SS_DISABLE == 0 && size < signal::MINSIGSTKSZ {
                return Err(Errno::ENOMEM);
            }
            if let Some(t) = self.core.tasks.get_mut(&tid) {
                t.altstack = [sp, size, u64::from(flags)];
            }
        }
        Ok(0)
    }
}

impl Kernel {
    /// `capget(hdr, data)`: the guest runs as root with every capability (no user namespaces
    /// or capability checks exist here).
    pub(super) fn sys_capget(&mut self, tid: Tid, hdr: u64, data: u64) -> SysResult {
        const V1: u32 = 0x1998_0330;
        const V2: u32 = 0x2007_1026;
        const V3: u32 = 0x2008_0522;
        // Root's capabilities: effective while the effective uid is 0, permitted while any of
        // the real, effective or saved uids is 0 (Linux's rules when dropping root).
        let c = self.creds(tid);
        let effective = c.euid == 0;
        let permitted = [c.ruid, c.euid, c.suid].contains(&0);
        let mm = self.mm(tid);
        let mut mm = mm.borrow_mut();
        let version = mm.mem.read_u32(hdr).map_err(|_| Errno::EFAULT)?;
        let words = match version {
            V1 => 1,
            V2 | V3 => 2,
            _ => {
                write_u32(mm.mem.as_mut(), hdr, V3)?;
                return if data == 0 { Ok(0) } else { Err(Errno::EINVAL) };
            }
        };
        if data != 0 {
            // { effective, permitted, inheritable } per 32-bit word; CAP_LAST_CAP is 40.
            let full = [u32::MAX, (1 << 9) - 1];
            let mut out = Vec::new();
            for &word in &full[..words] {
                out.extend_from_slice(&(if effective { word } else { 0 }).to_le_bytes());
                out.extend_from_slice(&(if permitted { word } else { 0 }).to_le_bytes());
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            write_bytes(mm.mem.as_mut(), data, &out)?;
        }
        Ok(0)
    }
}
