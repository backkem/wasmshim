//! Pending signals with their `siginfo_t`: what `kill`, `sigqueue`, a child's exit or a fault
//! tell the receiver (`SA_SIGINFO` handlers, `sigtimedwait`, `signalfd`).
//!
//! Like Linux, a standard signal (1–31) is pending at most once (a second one while it is
//! pending is dropped, keeping the first one's information), and real-time signals (32–64)
//! queue every instance in order. The lowest-numbered pending signal is taken first.

use std::collections::VecDeque;

use wasmshim_kernel::checkpoint::{Ckpt, CkptError, Reader, Writer};

use wasmshim_linux_abi::consts::signal::{self as sig};

/// Size of `siginfo_t` and of `struct signalfd_siginfo`.
pub(crate) const SIGINFO_SIZE: usize = 128;

/// A `siginfo_t` as the AArch64 kernel lays it out (`si_signo`, `si_errno`, `si_code`, then
/// the union at offset 16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SigInfo {
    pub raw: [u8; SIGINFO_SIZE],
}

impl SigInfo {
    /// Signal `signo` with `si_code` `code` and an empty union.
    pub fn new(signo: u32, code: i32) -> Self {
        let mut raw = [0; SIGINFO_SIZE];
        raw[0..4].copy_from_slice(&signo.to_le_bytes());
        raw[8..12].copy_from_slice(&code.to_le_bytes());
        Self { raw }
    }

    /// From a guest's `siginfo_t` (`rt_sigqueueinfo`), as signal `signo`.
    pub fn from_raw(signo: u32, raw: &[u8]) -> Self {
        let mut info = Self {
            raw: [0; SIGINFO_SIZE],
        };
        let n = raw.len().min(SIGINFO_SIZE);
        info.raw[..n].copy_from_slice(&raw[..n]);
        info.raw[0..4].copy_from_slice(&signo.to_le_bytes());
        info
    }

    fn put32(mut self, at: usize, v: u32) -> Self {
        self.raw[at..at + 4].copy_from_slice(&v.to_le_bytes());
        self
    }

    fn put64(mut self, at: usize, v: u64) -> Self {
        self.raw[at..at + 8].copy_from_slice(&v.to_le_bytes());
        self
    }

    fn get32(&self, at: usize) -> u32 {
        u32::from_le_bytes([
            self.raw[at],
            self.raw[at + 1],
            self.raw[at + 2],
            self.raw[at + 3],
        ])
    }

    fn get64(&self, at: usize) -> u64 {
        u64::from(self.get32(at)) | u64::from(self.get32(at + 4)) << 32
    }

    /// `si_signo`.
    pub fn signo(&self) -> u32 {
        self.get32(0)
    }

    /// The fault address of a synchronous fault signal (`uc_mcontext.fault_address`), else 0.
    pub fn fault_addr(&self) -> u64 {
        let fault = matches!(
            self.signo(),
            sig::SIGILL | sig::SIGFPE | sig::SIGSEGV | sig::SIGBUS | sig::SIGTRAP
        );
        if fault && self.code() > 0 && self.code() != sig::SI_KERNEL {
            self.get64(16)
        } else {
            0
        }
    }

    /// `si_code`.
    pub fn code(&self) -> i32 {
        self.get32(8) as i32
    }

    /// A signal sent by a process (`kill`: `SI_USER`, `tgkill`: `SI_TKILL`).
    pub fn sender(signo: u32, code: i32, pid: u32, uid: u32) -> Self {
        Self::new(signo, code).put32(16, pid).put32(20, uid)
    }

    /// A signal the kernel sends on its own (`SI_KERNEL`: timers, terminals).
    pub fn kernel(signo: u32) -> Self {
        Self::new(signo, sig::SI_KERNEL)
    }

    /// A synchronous fault at `addr`.
    pub fn fault(signo: u32, code: i32, addr: u64) -> Self {
        Self::new(signo, code).put64(16, addr)
    }

    /// `SIGCHLD` for a child that changed state.
    pub fn child(pid: u32, uid: u32, code: i32, status: i32) -> Self {
        Self::new(sig::SIGCHLD, code)
            .put32(16, pid)
            .put32(20, uid)
            .put32(24, status as u32)
    }

    /// A POSIX timer expiry (`SI_TIMER`): timer id, overrun count and the timer's value.
    pub fn timer(signo: u32, timer_id: u32, overrun: u32, value: u64) -> Self {
        Self::new(signo, sig::SI_TIMER)
            .put32(16, timer_id)
            .put32(20, overrun)
            .put64(24, value)
    }

    /// `si_timerid` of an `SI_TIMER` record.
    pub fn timer_id(&self) -> u32 {
        self.get32(16)
    }

    /// Sets `si_overrun` of an `SI_TIMER` record.
    pub fn set_overrun(&mut self, overrun: u32) {
        self.raw[20..24].copy_from_slice(&overrun.to_le_bytes());
    }

    /// The `struct signalfd_siginfo` a `signalfd` read returns for this signal (Linux's
    /// `signalfd_copyinfo`: the fields of the union member the signal and code select).
    pub fn to_signalfd(self) -> [u8; SIGINFO_SIZE] {
        let mut out = [0u8; SIGINFO_SIZE];
        let put32 = |out: &mut [u8; SIGINFO_SIZE], at: usize, v: u32| {
            out[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        let put64 = |out: &mut [u8; SIGINFO_SIZE], at: usize, v: u64| {
            out[at..at + 8].copy_from_slice(&v.to_le_bytes());
        };
        let signo = self.signo();
        let code = self.code();
        put32(&mut out, 0, signo); // ssi_signo
        put32(&mut out, 4, self.get32(4)); // ssi_errno
        put32(&mut out, 8, code as u32); // ssi_code
        let fault = matches!(
            signo,
            sig::SIGILL | sig::SIGFPE | sig::SIGSEGV | sig::SIGBUS | sig::SIGTRAP
        );
        if code == sig::SI_TIMER {
            let value = self.get64(24);
            put32(&mut out, 24, self.get32(16)); // ssi_tid
            put32(&mut out, 32, self.get32(20)); // ssi_overrun
            put32(&mut out, 44, value as u32); // ssi_int
            put64(&mut out, 48, value); // ssi_ptr
        } else if code > 0 && signo == sig::SIGCHLD {
            put32(&mut out, 12, self.get32(16)); // ssi_pid
            put32(&mut out, 16, self.get32(20)); // ssi_uid
            put32(&mut out, 40, self.get32(24)); // ssi_status
            put64(&mut out, 56, self.get64(32)); // ssi_utime
            put64(&mut out, 64, self.get64(40)); // ssi_stime
        } else if code > 0 && code != sig::SI_KERNEL && fault {
            put64(&mut out, 72, self.get64(16)); // ssi_addr
        } else {
            // kill, tgkill, sigqueue and the like: sender, uid and value.
            let value = self.get64(24);
            put32(&mut out, 12, self.get32(16));
            put32(&mut out, 16, self.get32(20));
            put32(&mut out, 44, value as u32);
            put64(&mut out, 48, value);
        }
        out
    }
}

fn bit(s: u32) -> u64 {
    1 << (s - 1)
}

/// The signals pending for a thread or a process.
#[derive(Clone, Debug, Default)]
pub(crate) struct SigQueue {
    /// Which signals are pending.
    pub set: u64,
    /// Their information, in arrival order.
    queue: VecDeque<SigInfo>,
}

impl SigQueue {
    /// Queues `info`; `false` if it was a standard signal that is already pending (dropped).
    pub fn push(&mut self, info: SigInfo) -> bool {
        let s = info.signo();
        if s == 0 || s > sig::NSIG {
            return false;
        }
        if s < sig::SIGRTMIN && self.set & bit(s) != 0 {
            return false;
        }
        self.set |= bit(s);
        self.queue.push_back(info);
        true
    }

    /// The lowest-numbered pending signal in `mask`.
    pub fn first(&self, mask: u64) -> Option<u32> {
        let ready = self.set & mask;
        (ready != 0).then(|| ready.trailing_zeros() + 1)
    }

    /// Takes the oldest instance of signal `s`, if pending.
    pub fn take(&mut self, s: u32) -> Option<SigInfo> {
        if s == 0 || s > sig::NSIG || self.set & bit(s) == 0 {
            return None;
        }
        let info = match self.queue.iter().position(|i| i.signo() == s) {
            Some(at) => self.queue.remove(at),
            None => None,
        };
        if !self.queue.iter().any(|i| i.signo() == s) {
            self.set &= !bit(s);
        }
        Some(info.unwrap_or_else(|| SigInfo::new(s, sig::SI_KERNEL)))
    }

    /// Drops every pending instance of the signals in `mask` (a signal becoming ignored).
    pub fn discard(&mut self, mask: u64) {
        self.set &= !mask;
        self.queue.retain(|i| mask & bit(i.signo()) == 0);
    }
}

impl Ckpt for SigInfo {
    fn save(&self, w: &mut Writer) {
        self.raw.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(Self {
            raw: Ckpt::load(r)?,
        })
    }
}

impl Ckpt for SigQueue {
    fn save(&self, w: &mut Writer) {
        self.set.save(w);
        self.queue.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(Self {
            set: Ckpt::load(r)?,
            queue: Ckpt::load(r)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_signals_coalesce_and_realtime_ones_queue() {
        let mut q = SigQueue::default();
        assert!(q.push(SigInfo::sender(sig::SIGUSR1, 0, 7, 0)));
        assert!(!q.push(SigInfo::sender(sig::SIGUSR1, 0, 8, 0)));
        assert!(q.push(SigInfo::new(33, sig::SI_QUEUE)));
        assert!(q.push(SigInfo::new(33, sig::SI_QUEUE)));
        assert!(q.push(SigInfo::new(32, sig::SI_QUEUE)));
        assert_eq!(q.first(u64::MAX), Some(sig::SIGUSR1));
        assert_eq!(q.take(sig::SIGUSR1).map(|i| i.get32(16)), Some(7));
        assert_eq!(q.take(sig::SIGUSR1), None);
        assert_eq!(q.first(u64::MAX), Some(32));
        assert!(q.take(33).is_some());
        assert_eq!(q.first(1 << 32), Some(33), "one instance of 33 is left");
        assert!(q.take(33).is_some());
        assert_eq!(q.set, 1 << 31);
    }

    #[test]
    fn signalfd_layouts() {
        let s = SigInfo::sender(sig::SIGUSR1, sig::SI_QUEUE, 5, 6).put64(24, 42);
        let f = s.to_signalfd();
        assert_eq!(&f[12..16], &5u32.to_le_bytes());
        assert_eq!(&f[44..48], &42u32.to_le_bytes());
        assert_eq!(&f[48..56], &42u64.to_le_bytes());
        let c = SigInfo::child(9, 0, sig::CLD_EXITED, 3).to_signalfd();
        assert_eq!(&c[12..16], &9u32.to_le_bytes());
        assert_eq!(&c[40..44], &3u32.to_le_bytes());
    }
}
