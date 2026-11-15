//! Signals: queueing, interrupting blocked threads, delivery on the AArch64 `rt_sigframe`
//! layout, and `rt_sigreturn`.
//!
//! * Sending follows Linux: an ignored, unblocked signal is discarded at once; a fatal one
//!   (default action "terminate", unblocked) kills the process at once; one with a handler is
//!   queued and interrupts its target thread if it is blocked in a syscall (`EINTR`, or a
//!   restart with `SA_RESTART`).
//! * Delivery happens whenever a thread is about to run guest code: the frame is built from
//!   the thread's [`CpuContext`] exactly like `arch/arm64/kernel/signal.c` lays it out
//!   (`siginfo`, `ucontext` with `sigcontext` and an `fpsimd_context` record).

use wasmshim_cpu::{A64Context, CpuContext, Fault, MemFaultKind};
use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::signal::{self as sig};
use wasmshim_linux_abi::structs::KernelSigaction;

use crate::host::Host;
use crate::kernel::{
    Blocker, ExitStatus, Kernel, LinuxKey, Park, Pid, Thread, ThreadState, Tid, WaitKey,
};
use crate::siginfo::SigInfo;
use crate::syscall::{Flow, SysResult};
use crate::uaccess::{read_bytes, write_bytes};

const SIGINFO_SIZE: u64 = 128;
/// Offset of `uc_mcontext` inside `struct ucontext`.
const MCONTEXT: u64 = 176;
/// `sizeof(struct ucontext)`: 176 + 8 (fault_address) + 31*8 + 3*8, then `__reserved[4096]`
/// aligned to 16 at offset 464.
const UCONTEXT_SIZE: u64 = 464 + 4096;
/// `struct rt_sigframe` (siginfo + ucontext) plus the 16-byte frame record above it.
pub const SIGFRAME_SIZE: u64 = SIGINFO_SIZE + UCONTEXT_SIZE + 16;
const FPSIMD_MAGIC: u32 = 0x4650_8001;
const FPSIMD_SIZE: u32 = 528;

/// What a signal does to a process with a given disposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Ignore,
    Terminate,
    Handle(KernelSigaction),
}

fn bit(s: u32) -> u64 {
    1 << (s - 1)
}

/// Signals that cannot be blocked or caught.
const UNBLOCKABLE: u64 = (1 << (sig::SIGKILL - 1)) | (1 << (sig::SIGSTOP - 1));

pub(crate) fn default_ignored(s: u32) -> bool {
    // No job control: stop signals are ignored too.
    matches!(
        s,
        sig::SIGCHLD
            | sig::SIGCONT
            | sig::SIGURG
            | sig::SIGWINCH
            | sig::SIGSTOP
            | sig::SIGTSTP
            | sig::SIGTTIN
            | sig::SIGTTOU
    )
}

impl Kernel {
    fn action(&self, pid: Pid, s: u32) -> Action {
        let Some(p) = self.procs.get(&pid) else {
            return Action::Ignore;
        };
        let a = p.sigactions.borrow()[s as usize - 1];
        if s == sig::SIGKILL {
            return Action::Terminate;
        }
        match a.handler {
            sig::SIG_IGN => Action::Ignore,
            sig::SIG_DFL if default_ignored(s) => Action::Ignore,
            sig::SIG_DFL => Action::Terminate,
            _ => Action::Handle(a),
        }
    }

    /// Sends `s` to process `pid` from the kernel itself (`SI_KERNEL`).
    pub(crate) fn send_to_process(&mut self, pid: Pid, s: u32) -> SysResult {
        self.queue_to_process(pid, SigInfo::kernel(s))
    }

    /// Sends `s` to thread `tid` from the kernel itself (`SI_KERNEL`).
    pub(crate) fn send_to_thread(&mut self, tid: Tid, s: u32) -> SysResult {
        self.queue_to_thread(tid, SigInfo::kernel(s))
    }

    /// `true` if thread `t` would take signal `s` now: it does not block it, or it waits for it
    /// in `rt_sigtimedwait` (Linux's `wants_signal`).
    fn wants(t: &Thread, s: u32) -> bool {
        t.sigmask & bit(s) == 0 || t.sigwait.is_some_and(|(set, _)| set & bit(s) != 0)
    }

    /// Sends `info` to process `pid`: handled by the first thread that takes it, else pending
    /// for the process.
    pub(crate) fn queue_to_process(&mut self, pid: Pid, info: SigInfo) -> SysResult {
        let s = info.signo();
        if s > sig::NSIG {
            return Err(Errno::EINVAL);
        }
        let Some(p) = self.procs.get(&pid) else {
            return Err(Errno::ESRCH);
        };
        if s == 0 {
            return Ok(0);
        }
        let target = p
            .threads
            .iter()
            .copied()
            .find(|t| self.core.tasks.get(t).is_some_and(|t| Self::wants(t, s)));
        match target {
            Some(tid) => self.queue_to_thread(tid, info),
            None => {
                // Everyone blocks it: it stays pending for the process (unless ignored and
                // nobody could ever want it, which `deliver_signals` sorts out on unblock).
                if let Some(p) = self.procs.get_mut(&pid) {
                    p.pending.push(info);
                }
                self.signal_pending(pid);
                Ok(0)
            }
        }
    }

    /// Sends `info` to thread `tid`.
    pub(crate) fn queue_to_thread(&mut self, tid: Tid, info: SigInfo) -> SysResult {
        let s = info.signo();
        if s > sig::NSIG {
            return Err(Errno::EINVAL);
        }
        let Some(t) = self.core.tasks.get(&tid) else {
            return Err(Errno::ESRCH);
        };
        if s == 0 {
            return Ok(0);
        }
        let pid = t.pid;
        // A signal that `rt_sigtimedwait` waits for is taken by it, never delivered.
        let waited = t.sigwait.is_some_and(|(set, _)| set & bit(s) != 0);
        let blocked = (t.sigmask & bit(s) != 0 || waited) && s != sig::SIGKILL;
        match (self.action(pid, s), blocked) {
            (Action::Ignore, false) => Ok(0),
            (Action::Terminate, false) => {
                if self.cfg.trace {
                    self.diag(format!("[{pid}] killed by {}", sig::name(s)));
                }
                self.exit_process(pid, ExitStatus::Signal(s));
                Ok(0)
            }
            (Action::Handle(a), false) => {
                if let Some(t) = self.core.tasks.get_mut(&tid) {
                    t.pending.push(info);
                }
                self.interrupt(tid, a.flags & sig::SA_RESTART != 0);
                Ok(0)
            }
            (_, true) => {
                if let Some(t) = self.core.tasks.get_mut(&tid) {
                    t.pending.push(info);
                }
                self.signal_pending(pid);
                Ok(0)
            }
        }
    }

    /// A blocked signal became pending for `pid` or one of its threads: wake `rt_sigtimedwait`
    /// and `signalfd` readers, and pollers.
    fn signal_pending(&mut self, pid: Pid) {
        self.sig_generation += 1;
        self.wake(
            WaitKey::Personal(LinuxKey::Signals(pid)),
            usize::MAX,
            u32::MAX,
        );
        self.wake(WaitKey::AnyIo, usize::MAX, u32::MAX);
    }

    /// Takes the lowest pending signal of `mask` for thread `tid`: its own first, then its
    /// process's (Linux's `dequeue_signal`).
    pub(crate) fn dequeue_signal(&mut self, tid: Tid, mask: u64) -> Option<SigInfo> {
        let t = self.core.tasks.get_mut(&tid)?;
        let pid = t.pid;
        let mut info = match t.pending.first(mask) {
            Some(s) => t.pending.take(s)?,
            None => {
                let p = self.procs.get_mut(&pid)?;
                let s = p.pending.first(mask)?;
                p.pending.take(s)?
            }
        };
        if info.code() == sig::SI_TIMER {
            self.timer_signal_taken(pid, &mut info);
        }
        Some(info)
    }

    /// The signals pending for thread `tid` (its own and its process's).
    pub(crate) fn pending_set(&self, tid: Tid) -> u64 {
        let Some(t) = self.core.tasks.get(&tid) else {
            return 0;
        };
        t.pending.set | self.procs.get(&t.pid).map_or(0, |p| p.pending.set)
    }

    /// Ends a blocking syscall early because a signal will be handled.
    fn interrupt(&mut self, tid: Tid, restart: bool) {
        let Some(t) = self.core.tasks.get_mut(&tid) else {
            return;
        };
        let ThreadState::Blocked(b) = t.state else {
            return;
        };
        t.timed = None;
        // An interrupted `rt_sigtimedwait` is over (it is never restarted).
        let sigwait = t.sigwait.take().is_some();
        match b {
            Blocker::Restart {
                resume_pc,
                key,
                timeout_result,
                ..
            } => {
                // signal(7), "Interruption of system calls": never restarted, whatever
                // SA_RESTART says, are the readiness waits (poll, ppoll, select, pselect6,
                // epoll_wait: the only calls that wait on `AnyIo`) and the calls that end in
                // EAGAIN at a deadline (socket calls with SO_RCVTIMEO/SO_SNDTIMEO, semop).
                let never = key == WaitKey::AnyIo || timeout_result == Some(Errno::EAGAIN.ret());
                if !restart || sigwait || never {
                    t.cpu.set_pc(resume_pc);
                    t.cpu.set_syscall_result(Errno::EINTR.ret());
                }
            }
            Blocker::Sleep { until, rem } => {
                t.cpu.set_syscall_result(Errno::EINTR.ret());
                if rem != 0 {
                    let left = until.saturating_sub(self.core.slice_now);
                    let pid = t.pid;
                    if let Some(p) = self.procs.get(&pid) {
                        let ts = wasmshim_linux_abi::structs::Timespec::from_nanos(left).to_bytes();
                        let _ = write_bytes(p.mm.borrow_mut().mem.as_mut(), rem, &ts);
                    }
                }
            }
            Blocker::Futex { .. } | Blocker::Personal(Park::SigSuspend) => {
                t.cpu.set_syscall_result(Errno::EINTR.ret());
            }
            Blocker::Personal(Park::Vfork { .. }) => return,
        }
        self.make_runnable(tid);
    }

    /// Delivers pending, unblocked signals to `tid` before it runs guest code. Returns `false`
    /// if the thread no longer runs (its process was killed).
    pub(crate) fn deliver_signals(&mut self, tid: Tid, host: &mut dyn Host) -> bool {
        loop {
            let Some(t) = self.core.tasks.get(&tid) else {
                return false;
            };
            let pid = t.pid;
            let unblocked = !(t.sigmask & !UNBLOCKABLE);
            let Some(info) = self.dequeue_signal(tid, unblocked) else {
                // A call that waited with a temporary mask (ppoll, pselect6, epoll_pwait) and
                // returns without running a handler gets its old mask back, which may unblock
                // more (Linux: `restore_saved_sigmask`). A handler's frame already took it.
                if let Some(t) = self.core.tasks.get_mut(&tid)
                    && t.state == ThreadState::Runnable
                    && let Some(old) = t.saved_mask.take()
                {
                    t.sigmask = old;
                    continue;
                }
                return true;
            };
            let s = info.signo();
            match self.action(pid, s) {
                Action::Ignore => continue,
                Action::Terminate => {
                    if self.cfg.trace {
                        host.diag(&format!("[{pid}] killed by {}", sig::name(s)));
                    }
                    self.exit_process(pid, ExitStatus::Signal(s));
                    return false;
                }
                Action::Handle(a) => {
                    if self.setup_frame(tid, s, a, info).is_err() {
                        // An unwritable signal stack is fatal, like Linux's force_sigsegv.
                        host.diag(&format!(
                            "[{pid}] cannot deliver {}: bad signal stack",
                            sig::name(s)
                        ));
                        self.exit_process(pid, ExitStatus::Signal(sig::SIGSEGV));
                        return false;
                    }
                }
            }
        }
    }

    /// A synchronous fault: deliver its signal to a handler, or kill the process.
    pub(crate) fn fault_signal(&mut self, tid: Tid, fault: Fault, host: &mut dyn Host) {
        let Some(pid) = self.core.tasks.get(&tid).map(|t| t.pid) else {
            return;
        };
        let (s, code, addr, what) = match fault {
            Fault::Memory { fault, .. }
                if self
                    .procs
                    .get(&pid)
                    .is_some_and(|p| p.mm.borrow().beyond_object(fault.addr, fault.access)) =>
            {
                (
                    sig::SIGBUS,
                    sig::BUS_ADRERR,
                    fault.addr,
                    format!(
                        "{:?} of {:#x} (past the end of a shared object)",
                        fault.access, fault.addr
                    ),
                )
            }
            Fault::Memory { fault, .. } => (
                sig::SIGSEGV,
                if fault.kind == MemFaultKind::Unmapped {
                    sig::SEGV_MAPERR
                } else {
                    sig::SEGV_ACCERR
                },
                fault.addr,
                format!(
                    "{:?} of {:#x} ({})",
                    fault.access,
                    fault.addr,
                    if fault.kind == MemFaultKind::Unmapped {
                        "unmapped"
                    } else {
                        "protection"
                    }
                ),
            ),
            Fault::Undefined { insn, pc } => (
                sig::SIGILL,
                sig::ILL_ILLOPC,
                pc,
                format!("undefined instruction {insn:#010x}"),
            ),
            Fault::Breakpoint { imm, pc } => {
                (sig::SIGTRAP, sig::TRAP_BRKPT, pc, format!("brk #{imm:#x}"))
            }
            Fault::Alignment { addr, .. } => (
                sig::SIGBUS,
                sig::BUS_ADRALN,
                addr,
                format!("misaligned access at {addr:#x}"),
            ),
        };
        let blocked = self
            .core
            .tasks
            .get(&tid)
            .is_some_and(|t| t.sigmask & bit(s) != 0);
        if let (Action::Handle(a), false) = (self.action(pid, s), blocked)
            && self
                .setup_frame(tid, s, a, SigInfo::fault(s, code, addr))
                .is_ok()
        {
            return;
        }
        host.diag(&format!(
            "[{pid}] killed by {}: {what} at pc {:#x}",
            sig::name(s),
            fault.pc()
        ));
        self.exit_process(pid, ExitStatus::Signal(s));
    }

    /// Builds an `rt_sigframe` on the thread's (alternate) stack and points it at the handler.
    fn setup_frame(
        &mut self,
        tid: Tid,
        s: u32,
        a: KernelSigaction,
        info: SigInfo,
    ) -> Result<(), Errno> {
        let t = self.core.tasks.get(&tid).ok_or(Errno::ESRCH)?;
        let CpuContext::A64(mut ctx) = t.cpu.context();
        let [ss_sp, ss_size, ss_flags] = t.altstack;
        // As Linux's `on_sig_stack` and `sigsp`: the guest's `ss_sp + ss_size` may wrap (the
        // frame write then faults).
        let on_alt = ss_flags & u64::from(sig::SS_DISABLE) == 0
            && ctx.sp > ss_sp
            && ctx.sp - ss_sp <= ss_size;
        let use_alt =
            a.flags & sig::SA_ONSTACK != 0 && ss_flags & u64::from(sig::SS_DISABLE) == 0 && !on_alt;
        let top = if use_alt {
            ss_sp.wrapping_add(ss_size)
        } else {
            ctx.sp
        };
        let frame = top.wrapping_sub(SIGFRAME_SIZE) & !15;
        let saved_mask = match t.state {
            ThreadState::Runnable => t.saved_mask.unwrap_or(t.sigmask),
            ThreadState::Blocked(_) => t.sigmask,
        };
        let mut buf = vec![0u8; SIGFRAME_SIZE as usize];
        buf[..SIGINFO_SIZE as usize].copy_from_slice(&info.raw);
        let uc = SIGINFO_SIZE as usize;
        let put = |b: &mut [u8], at: usize, v: u64| b[at..at + 8].copy_from_slice(&v.to_le_bytes());
        // uc_stack
        put(&mut buf, uc + 16, ss_sp);
        buf[uc + 24..uc + 28].copy_from_slice(
            &((if on_alt { sig::SS_ONSTACK } else { 0 } | ss_flags as u32).to_le_bytes()),
        );
        put(&mut buf, uc + 32, ss_size);
        put(&mut buf, uc + 40, saved_mask);
        let mc = uc + MCONTEXT as usize;
        put(&mut buf, mc, info.fault_addr());
        for (i, x) in ctx.x.iter().enumerate() {
            put(&mut buf, mc + 8 + 8 * i, *x);
        }
        put(&mut buf, mc + 256, ctx.sp);
        put(&mut buf, mc + 264, ctx.pc);
        put(&mut buf, mc + 272, ctx.flags);
        let fp = mc + 288;
        buf[fp..fp + 4].copy_from_slice(&FPSIMD_MAGIC.to_le_bytes());
        buf[fp + 4..fp + 8].copy_from_slice(&FPSIMD_SIZE.to_le_bytes());
        buf[fp + 8..fp + 12].copy_from_slice(&(ctx.fpsr as u32).to_le_bytes());
        buf[fp + 12..fp + 16].copy_from_slice(&(ctx.fpcr as u32).to_le_bytes());
        for (i, v) in ctx.v.iter().enumerate() {
            buf[fp + 16 + 16 * i..fp + 32 + 16 * i].copy_from_slice(&v.to_le_bytes());
        }
        // (The zero terminator record follows the fpsimd record.)
        let record = SIGINFO_SIZE as usize + UCONTEXT_SIZE as usize;
        put(&mut buf, record, ctx.x[29]);
        put(&mut buf, record + 8, ctx.x[30]);

        let pid = t.pid;
        let mm = self.procs.get(&pid).ok_or(Errno::ESRCH)?.mm.clone();
        write_bytes(mm.borrow_mut().mem.as_mut(), frame, &buf)?;

        ctx.x[0] = u64::from(s);
        ctx.x[1] = frame;
        ctx.x[2] = frame + SIGINFO_SIZE;
        ctx.x[29] = frame + record as u64;
        ctx.x[30] = if a.flags & sig::SA_RESTORER != 0 {
            a.restorer
        } else {
            crate::mm::SIGTRAMP
        };
        ctx.sp = frame;
        ctx.pc = a.handler;
        let t = self.core.tasks.get_mut(&tid).ok_or(Errno::ESRCH)?;
        t.cpu.set_context(&CpuContext::A64(ctx));
        t.saved_mask = None;
        let mut mask = t.sigmask | a.mask;
        if a.flags & sig::SA_NODEFER == 0 {
            mask |= bit(s);
        }
        t.sigmask = mask & !UNBLOCKABLE;
        if a.flags & sig::SA_RESETHAND != 0
            && let Some(p) = self.procs.get(&pid)
        {
            p.sigactions.borrow_mut()[s as usize - 1] = KernelSigaction::default();
        }
        Ok(())
    }

    /// `rt_sigreturn`: restores the context saved by [`Kernel::setup_frame`].
    pub(crate) fn sys_rt_sigreturn(&mut self, tid: Tid, host: &mut dyn Host) -> Flow {
        let Some(t) = self.core.tasks.get(&tid) else {
            return Flow::Ret(Errno::ESRCH.ret());
        };
        let pid = t.pid;
        let frame = t.cpu.sp();
        let Some(mm) = self.procs.get(&pid).map(|p| p.mm.clone()) else {
            return Flow::Ret(Errno::ESRCH.ret());
        };
        let buf = match read_bytes(
            mm.borrow_mut().mem.as_mut(),
            frame,
            (SIGINFO_SIZE + UCONTEXT_SIZE) as usize,
        ) {
            Ok(b) => b,
            Err(_) => {
                host.diag(&format!(
                    "[{pid}] rt_sigreturn with a bad frame at {frame:#x}"
                ));
                self.exit_process(pid, ExitStatus::Signal(sig::SIGSEGV));
                return Flow::Exec;
            }
        };
        let get = |at: usize| u64::from_le_bytes(buf[at..at + 8].try_into().unwrap_or_default());
        let uc = SIGINFO_SIZE as usize;
        let mc = uc + MCONTEXT as usize;
        let mut ctx = A64Context::default();
        for i in 0..31 {
            ctx.x[i] = get(mc + 8 + 8 * i);
        }
        ctx.sp = get(mc + 256);
        ctx.pc = get(mc + 264);
        ctx.flags = get(mc + 272) & 0xf000_0000;
        let fp = mc + 288;
        if u32::from_le_bytes(buf[fp..fp + 4].try_into().unwrap_or_default()) == FPSIMD_MAGIC {
            ctx.fpsr = u64::from(u32::from_le_bytes(
                buf[fp + 8..fp + 12].try_into().unwrap_or_default(),
            ));
            ctx.fpcr = u64::from(u32::from_le_bytes(
                buf[fp + 12..fp + 16].try_into().unwrap_or_default(),
            ));
            for i in 0..32 {
                ctx.v[i] = u128::from_le_bytes(
                    buf[fp + 16 + 16 * i..fp + 32 + 16 * i]
                        .try_into()
                        .unwrap_or_default(),
                );
            }
        }
        let mask = get(uc + 40) & !UNBLOCKABLE;
        if let Some(t) = self.core.tasks.get_mut(&tid) {
            ctx.tp = t.cpu.thread_pointer();
            t.cpu.set_context(&CpuContext::A64(ctx));
            t.sigmask = mask;
        }
        Flow::Exec
    }

    /// `rt_sigsuspend(mask)`: waits for a handled signal with `mask` in place.
    pub(crate) fn sys_rt_sigsuspend(&mut self, tid: Tid, mask_ptr: u64, size: u64) -> Flow {
        if size != 8 {
            return Flow::Ret(Errno::EINVAL.ret());
        }
        let mm = self.mm(tid);
        let mask = match mm.borrow().mem.read_u64(mask_ptr) {
            Ok(m) => m,
            Err(_) => return Flow::Ret(Errno::EFAULT.ret()),
        };
        let Some(t) = self.core.tasks.get_mut(&tid) else {
            return Flow::Ret(Errno::ESRCH.ret());
        };
        t.saved_mask = Some(t.sigmask);
        t.sigmask = mask & !UNBLOCKABLE;
        Flow::Block(Blocker::Personal(Park::SigSuspend))
    }

    /// Runs `call`, a readiness wait (`ppoll`, `pselect6`, `epoll_pwait`), with the signal mask
    /// at `mask_ptr` (`size` bytes; a null pointer: none) in place of the thread's for the
    /// call's duration, as those calls do atomically. A signal the mask lets through ends a
    /// wait that found nothing with `EINTR`; its handler's frame records the old mask, which
    /// `rt_sigreturn` puts back. Otherwise the old mask is back when the call returns.
    pub(crate) fn with_call_mask(
        &mut self,
        tid: Tid,
        mask_ptr: u64,
        size: u64,
        call: impl FnOnce(&mut Self) -> SysResult,
    ) -> SysResult {
        if mask_ptr == 0 {
            return call(self);
        }
        if size != 8 {
            return Err(Errno::EINVAL);
        }
        let mask = crate::uaccess::read_u64(self.mm(tid).borrow_mut().mem.as_mut(), mask_ptr)?;
        let Some(t) = self.core.tasks.get_mut(&tid) else {
            return Err(Errno::ESRCH);
        };
        // A restarted wait (woken to look again) keeps the mask it saved at its first entry.
        let current = t.sigmask;
        let old = *t.saved_mask.get_or_insert(current);
        t.sigmask = mask & !UNBLOCKABLE;
        let r = call(self);
        if matches!(r, Ok(0) | Err(crate::syscall::BLOCK)) && self.signal_deliverable(tid) {
            self.cancel_wait();
            return Err(Errno::EINTR);
        }
        if r != Err(crate::syscall::BLOCK)
            && let Some(t) = self.core.tasks.get_mut(&tid)
        {
            t.sigmask = old;
            t.saved_mask = None;
        }
        r
    }

    /// `true` if a signal pending for `tid` is unblocked and would do something (run a
    /// handler or end the process).
    fn signal_deliverable(&self, tid: Tid) -> bool {
        let Some(t) = self.core.tasks.get(&tid) else {
            return false;
        };
        let ready = self.pending_set(tid) & !t.sigmask;
        (1..=sig::NSIG).any(|s| ready & bit(s) != 0 && self.action(t.pid, s) != Action::Ignore)
    }

    /// `kill(pid, sig)`.
    pub(crate) fn sys_kill(&mut self, tid: Tid, pid: u64, s: u64) -> SysResult {
        let target = pid as i32;
        let s = u32::try_from(s).map_err(|_| Errno::EINVAL)?;
        let me = self.pid_of(tid);
        let info = SigInfo::sender(s, sig::SI_USER, me, self.creds(tid).ruid);
        let group = |k: &Kernel, pgid: Pid| {
            k.procs
                .iter()
                .filter(|(_, p)| p.pgid == pgid)
                .map(|(&q, _)| q)
                .collect::<Vec<_>>()
        };
        let targets: Vec<Pid> = match target {
            0 => {
                let pg = self.procs.get(&me).map_or(me, |p| p.pgid);
                group(self, pg)
            }
            // kill(2): every process the caller may signal except init; Linux also skips the
            // caller itself (kernel/signal.c, `kill_something_info`), and with nobody else
            // the call fails with ESRCH.
            -1 => self
                .procs
                .keys()
                .copied()
                .filter(|&q| q != 1 && q != me)
                .collect(),
            t if t < 0 => group(self, t.unsigned_abs()),
            t => vec![t as u32],
        };
        if targets.is_empty() {
            return Err(Errno::ESRCH);
        }
        let mut result = Err(Errno::ESRCH);
        for p in targets {
            let r = if self.procs.contains_key(&p) {
                self.queue_to_process(p, info)
            } else if self.zombies.contains_key(&p) {
                // A zombie is a process until it is reaped: the signal is discarded.
                if s > sig::NSIG {
                    Err(Errno::EINVAL)
                } else {
                    Ok(0)
                }
            } else {
                Err(Errno::ESRCH)
            };
            match r {
                Ok(_) => result = Ok(0),
                // kill(2): EINVAL "An invalid signal was specified" (to a process that exists).
                Err(Errno::EINVAL) if result.is_err() => result = Err(Errno::EINVAL),
                Err(_) => {}
            }
        }
        result
    }

    /// `tgkill(tgid, tid, sig)` / `tkill(tid, sig)` (`tgid` < 0).
    pub(crate) fn sys_tgkill(&mut self, tid: Tid, tgid: i64, target: u64, s: u64) -> SysResult {
        let target = u32::try_from(target).map_err(|_| Errno::EINVAL)?;
        let s = u32::try_from(s).map_err(|_| Errno::EINVAL)?;
        let pid = self
            .core
            .tasks
            .get(&target)
            .map(|t| t.pid)
            .ok_or(Errno::ESRCH)?;
        if tgid > 0 && tgid as u32 != pid {
            return Err(Errno::ESRCH);
        }
        let info = SigInfo::sender(s, sig::SI_TKILL, self.pid_of(tid), self.creds(tid).ruid);
        self.queue_to_thread(target, info)
    }
}

impl Kernel {
    /// Drops the pending instances of the signals in `mask` for process `pid` and its threads
    /// (a signal that became ignored).
    pub(crate) fn discard_pending(&mut self, pid: Pid, mask: u64) {
        let Some(p) = self.procs.get_mut(&pid) else {
            return;
        };
        p.pending.discard(mask);
        let threads: Vec<Tid> = p.threads.iter().copied().collect();
        for tid in threads {
            if let Some(t) = self.core.tasks.get_mut(&tid) {
                t.pending.discard(mask);
            }
        }
    }

    /// `rt_sigpending(set, sigsetsize)`: pending signals that are blocked.
    pub(crate) fn sys_rt_sigpending(&mut self, tid: Tid, set: u64, size: u64) -> SysResult {
        if size > 8 {
            return Err(Errno::EINVAL);
        }
        let mask = self.core.tasks.get(&tid).map_or(0, |t| t.sigmask);
        let pending = self.pending_set(tid) & mask;
        let bytes = pending.to_le_bytes();
        write_bytes(
            self.mm(tid).borrow_mut().mem.as_mut(),
            set,
            &bytes[..size as usize],
        )?;
        Ok(0)
    }

    /// `rt_sigtimedwait(set, info, timeout, sigsetsize)`: takes a pending signal of `set`, or
    /// waits for one until the (relative) timeout. The deadline is fixed at the first entry:
    /// wake-ups that find nothing (another thread took the signal) wait for the rest of it.
    pub(crate) fn sys_rt_sigtimedwait(
        &mut self,
        tid: Tid,
        set_ptr: u64,
        info_ptr: u64,
        ts_ptr: u64,
        size: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if size != 8 {
            return Err(Errno::EINVAL);
        }
        let mm = self.mm(tid);
        let set = crate::uaccess::read_u64(mm.borrow_mut().mem.as_mut(), set_ptr)? & !UNBLOCKABLE;
        let timeout = if ts_ptr == 0 {
            None
        } else {
            let ts = wasmshim_linux_abi::structs::Timespec::from_bytes(
                &crate::uaccess::read_array(mm.borrow_mut().mem.as_mut(), ts_ptr)?,
            );
            Some(ts.to_nanos_saturating().ok_or(Errno::EINVAL)?)
        };
        let now = host.monotonic_ns();
        let previous = self.core.tasks.get(&tid).and_then(|t| t.sigwait);
        let deadline = match previous {
            Some((s, d)) if s == set => d,
            _ => timeout.map(|t| now.saturating_add(t)),
        };
        if let Some(info) = self.dequeue_signal(tid, set) {
            if let Some(t) = self.core.tasks.get_mut(&tid) {
                t.sigwait = None;
            }
            if info_ptr != 0 {
                write_bytes(mm.borrow_mut().mem.as_mut(), info_ptr, &info.raw)?;
            }
            return Ok(u64::from(info.signo()));
        }
        if deadline.is_some_and(|d| d <= now) {
            if let Some(t) = self.core.tasks.get_mut(&tid) {
                t.sigwait = None;
            }
            return Err(Errno::EAGAIN);
        }
        let pid = self.pid_of(tid);
        if let Some(t) = self.core.tasks.get_mut(&tid) {
            t.sigwait = Some((set, deadline));
        }
        // At the deadline the call runs again and times out.
        Err(self.wait_for(WaitKey::Personal(LinuxKey::Signals(pid)), deadline, None))
    }

    /// The `siginfo_t` a `*sigqueueinfo` caller passed, checked like Linux: only the kernel may
    /// claim to be `kill` or `tgkill` (`si_code >= 0`, `SI_TKILL`) towards another process.
    fn user_siginfo(&self, tid: Tid, tgid: u32, s: u32, ptr: u64) -> Result<SigInfo, Errno> {
        let raw = read_bytes(
            self.mm(tid).borrow_mut().mem.as_mut(),
            ptr,
            crate::siginfo::SIGINFO_SIZE,
        )?;
        let info = SigInfo::from_raw(s, &raw);
        if (info.code() >= 0 || info.code() == sig::SI_TKILL) && tgid != self.pid_of(tid) {
            return Err(Errno::EPERM);
        }
        Ok(info)
    }

    /// `rt_sigqueueinfo(tgid, sig, info)`.
    pub(crate) fn sys_rt_sigqueueinfo(
        &mut self,
        tid: Tid,
        tgid: u64,
        s: u64,
        info: u64,
    ) -> SysResult {
        let s = u32::try_from(s).map_err(|_| Errno::EINVAL)?;
        let tgid = u32::try_from(tgid).map_err(|_| Errno::ESRCH)?;
        if s > sig::NSIG {
            return Err(Errno::EINVAL);
        }
        let info = self.user_siginfo(tid, tgid, s, info)?;
        self.queue_to_process(tgid, info)
    }

    /// `rt_tgsigqueueinfo(tgid, tid, sig, info)`.
    pub(crate) fn sys_rt_tgsigqueueinfo(
        &mut self,
        tid: Tid,
        tgid: u64,
        target: u64,
        s: u64,
        info: u64,
    ) -> SysResult {
        let s = u32::try_from(s).map_err(|_| Errno::EINVAL)?;
        let (Ok(tgid), Ok(target)) = (u32::try_from(tgid), u32::try_from(target)) else {
            return Err(Errno::EINVAL);
        };
        if s > sig::NSIG || tgid == 0 || target == 0 {
            return Err(Errno::EINVAL);
        }
        let info = self.user_siginfo(tid, tgid, s, info)?;
        if self.core.tasks.get(&target).map(|t| t.pid) != Some(tgid) {
            return Err(Errno::ESRCH);
        }
        self.queue_to_thread(target, info)
    }
}
