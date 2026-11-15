//! POSIX per-process timers (`timer_create`, `timer_settime`, `timer_gettime`,
//! `timer_getoverrun`, `timer_delete`), on virtual time.
//!
//! Wall-clock timers (`CLOCK_REALTIME`, `CLOCK_MONOTONIC`, `CLOCK_BOOTTIME`, `CLOCK_TAI`) expire
//! on the monotonic clock from the scheduler, like `ITIMER_REAL`; an absolute `CLOCK_REALTIME`
//! (or TAI) expiry is converted once, when it is set (the clock is never stepped here). CPU-time
//! timers (`CLOCK_PROCESS_CPUTIME_ID`, `CLOCK_THREAD_CPUTIME_ID`) expire when the process's (or
//! thread's) retired instructions, at the configured rate, pass the expiry; they are checked
//! between scheduling slices, so they fire within a quantum.
//!
//! As on Linux, one signal per timer is pending at a time: expiries while it is pending, and
//! periods missed between two checks, count as overruns, reported by the signal's `si_overrun`
//! and by `timer_getoverrun` once the signal has been taken.

use std::collections::BTreeMap;

use wasmshim_kernel::checkpoint::{Ckpt, CkptError, Reader, Writer};

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{clock, signal};
use wasmshim_linux_abi::structs::Timespec;

use super::SysResult;
use crate::host::Host;
use crate::kernel::{Kernel, Pid, Tid};
use crate::siginfo::SigInfo;
use crate::uaccess::{read_array, write_bytes};

const SIGEV_SIGNAL: i32 = 0;
const SIGEV_NONE: i32 = 1;
const SIGEV_THREAD: i32 = 2;
const SIGEV_THREAD_ID: i32 = 4;
/// `DELAYTIMER_MAX`: overrun counts saturate here.
const DELAYTIMER_MAX: u32 = i32::MAX as u32;
/// Most timers per process (Linux bounds them by `RLIMIT_SIGPENDING`).
const MAX_TIMERS: usize = 4096;

/// Which clock drives a timer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimerClock {
    /// Monotonic virtual time (wall-clock timers).
    Wall,
    /// CPU time of the process (`None`) or of one thread.
    Cpu(Option<Tid>),
}

/// What an expiry does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Notify {
    None,
    Process(u32),
    Thread(Tid, u32),
}

/// One timer.
#[derive(Clone, Debug)]
pub(crate) struct PosixTimer {
    clock: TimerClock,
    /// Absolute times are on the realtime clock (`CLOCK_REALTIME`, `CLOCK_TAI`).
    realtime: bool,
    notify: Notify,
    /// `sigev_value`, passed in the signal's `si_value`.
    value: u64,
    /// Next expiry on the timer's clock; 0 when disarmed.
    expiry: u64,
    interval: u64,
    /// Expiries since its signal was queued (while it is pending).
    overrun: u32,
    /// `timer_getoverrun`: the overrun count of the last signal taken.
    overrun_last: u32,
    /// Its signal is queued and not yet taken.
    pending: bool,
}

/// A process's timers (not inherited by `fork`, deleted by `execve`).
#[derive(Clone, Debug, Default)]
pub(crate) struct PosixTimers {
    timers: BTreeMap<i32, PosixTimer>,
    next_id: i32,
}

/// `struct itimerspec` (interval, then value) to nanoseconds.
fn parse_itimerspec(b: &[u8; 32]) -> Result<(u64, u64), Errno> {
    let ts = |o: usize| {
        let mut half = [0u8; 16];
        half.copy_from_slice(&b[o..o + 16]);
        Timespec::from_bytes(&half)
            .to_nanos_saturating()
            .ok_or(Errno::EINVAL)
    };
    Ok((ts(0)?, ts(16)?))
}

impl Ckpt for PosixTimer {
    fn save(&self, w: &mut Writer) {
        match self.clock {
            TimerClock::Wall => (0u8, 0u32).save(w),
            TimerClock::Cpu(None) => (1u8, 0u32).save(w),
            TimerClock::Cpu(Some(t)) => (2u8, t).save(w),
        }
        self.realtime.save(w);
        match self.notify {
            Notify::None => (0u8, 0u32, 0u32).save(w),
            Notify::Process(s) => (1u8, 0u32, s).save(w),
            Notify::Thread(t, s) => (2u8, t, s).save(w),
        }
        self.value.save(w);
        self.expiry.save(w);
        self.interval.save(w);
        self.overrun.save(w);
        self.overrun_last.save(w);
        self.pending.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        let clock = match <(u8, u32)>::load(r)? {
            (0, _) => TimerClock::Wall,
            (1, _) => TimerClock::Cpu(None),
            (2, t) => TimerClock::Cpu(Some(t)),
            _ => return Err(CkptError::new("bad timer clock")),
        };
        let realtime = Ckpt::load(r)?;
        let notify = match <(u8, u32, u32)>::load(r)? {
            (0, _, _) => Notify::None,
            (1, _, s) => Notify::Process(s),
            (2, t, s) => Notify::Thread(t, s),
            _ => return Err(CkptError::new("bad timer notification")),
        };
        Ok(Self {
            clock,
            realtime,
            notify,
            value: Ckpt::load(r)?,
            expiry: Ckpt::load(r)?,
            interval: Ckpt::load(r)?,
            overrun: Ckpt::load(r)?,
            overrun_last: Ckpt::load(r)?,
            pending: Ckpt::load(r)?,
        })
    }
}

impl Ckpt for PosixTimers {
    fn save(&self, w: &mut Writer) {
        self.timers.save(w);
        self.next_id.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(Self {
            timers: Ckpt::load(r)?,
            next_id: Ckpt::load(r)?,
        })
    }
}

fn format_itimerspec(interval: u64, value: u64) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[..16].copy_from_slice(&Timespec::from_nanos(interval).to_bytes());
    b[16..].copy_from_slice(&Timespec::from_nanos(value).to_bytes());
    b
}

impl Kernel {
    /// CPU time in ns of thread `tid` (`Some`) or of process `pid`.
    pub(crate) fn cpu_time(&self, pid: Pid, tid: Option<Tid>) -> u64 {
        let rate = self.cfg.ns_per_insn.max(1);
        let insns: u64 = match tid {
            Some(t) => self.core.tasks.get(&t).map_or(0, |t| t.own_retired()),
            None => self.procs.get(&pid).map_or(0, |p| {
                p.threads
                    .iter()
                    .filter_map(|t| self.core.tasks.get(t))
                    .map(|t| t.own_retired())
                    .sum()
            }),
        };
        insns.saturating_mul(rate)
    }

    /// The current value of a timer's clock.
    fn timer_now(&self, pid: Pid, clock: TimerClock, now: u64) -> u64 {
        match clock {
            TimerClock::Wall => now,
            TimerClock::Cpu(tid) => self.cpu_time(pid, tid),
        }
    }

    fn timers_of(&mut self, tid: Tid) -> Result<&mut PosixTimers, Errno> {
        let pid = self.pid_of(tid);
        Ok(&mut self.procs.get_mut(&pid).ok_or(Errno::ESRCH)?.timers)
    }

    /// `timer_create(clockid, sevp, timerid)`.
    pub(super) fn sys_timer_create(
        &mut self,
        tid: Tid,
        clk: u64,
        sevp: u64,
        id_ptr: u64,
    ) -> SysResult {
        let pid = self.pid_of(tid);
        let clock = match clk {
            clock::CLOCK_REALTIME
            | clock::CLOCK_MONOTONIC
            | clock::CLOCK_BOOTTIME
            | clock::CLOCK_TAI => TimerClock::Wall,
            clock::CLOCK_PROCESS_CPUTIME_ID => TimerClock::Cpu(None),
            clock::CLOCK_THREAD_CPUTIME_ID => TimerClock::Cpu(Some(tid)),
            _ => return Err(Errno::EINVAL),
        };
        let mm = self.mm(tid);
        let (notify, value, explicit_value) = if sevp == 0 {
            (Notify::Process(signal::SIGALRM), 0, false)
        } else {
            let b: [u8; 64] = read_array(mm.borrow_mut().mem.as_mut(), sevp)?;
            let word = |o: usize| i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            let value = u64::from_le_bytes(b[0..8].try_into().unwrap_or_default());
            let (signo, how) = (word(8), word(12));
            let signo_ok = signo > 0 && signo as u32 <= signal::NSIG;
            let notify = match how {
                SIGEV_NONE => Notify::None,
                SIGEV_SIGNAL | SIGEV_THREAD if signo_ok => Notify::Process(signo as u32),
                SIGEV_THREAD_ID if signo_ok => {
                    let target = word(16) as u32;
                    if self.core.tasks.get(&target).map(|t| t.pid) != Some(pid) {
                        return Err(Errno::EINVAL);
                    }
                    Notify::Thread(target, signo as u32)
                }
                _ => return Err(Errno::EINVAL),
            };
            (notify, value, true)
        };
        let timers = self.timers_of(tid)?;
        if timers.timers.len() >= MAX_TIMERS {
            return Err(Errno::EAGAIN);
        }
        let mut id = timers.next_id;
        while timers.timers.contains_key(&id) {
            id = id.checked_add(1).unwrap_or(0);
        }
        timers.next_id = id.checked_add(1).unwrap_or(0);
        timers.timers.insert(
            id,
            PosixTimer {
                clock,
                realtime: matches!(clk, clock::CLOCK_REALTIME | clock::CLOCK_TAI),
                notify,
                // Without a `sigevent`, `si_value` is the timer id.
                value: if explicit_value { value } else { id as u64 },
                expiry: 0,
                interval: 0,
                overrun: 0,
                overrun_last: 0,
                pending: false,
            },
        );
        let id_bytes = id.to_le_bytes();
        if let Err(e) = write_bytes(mm.borrow_mut().mem.as_mut(), id_ptr, &id_bytes) {
            self.timers_of(tid)?.timers.remove(&id);
            return Err(e);
        }
        Ok(0)
    }

    /// The `(interval, remaining)` of timer `id` now.
    fn timer_value(&self, pid: Pid, t: &PosixTimer, now: u64) -> (u64, u64) {
        if t.expiry == 0 {
            return (t.interval, 0);
        }
        let clock_now = self.timer_now(pid, t.clock, now);
        (t.interval, t.expiry.saturating_sub(clock_now).max(1))
    }

    /// `timer_settime(id, flags, new, old)`.
    pub(super) fn sys_timer_settime(
        &mut self,
        tid: Tid,
        id: u64,
        flags: u64,
        new: u64,
        old: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let pid = self.pid_of(tid);
        let id = id as i32;
        let now = host.monotonic_ns();
        let mm = self.mm(tid);
        let (interval, value) = parse_itimerspec(&read_array(mm.borrow_mut().mem.as_mut(), new)?)?;
        let t = self
            .procs
            .get(&pid)
            .and_then(|p| p.timers.timers.get(&id))
            .cloned()
            .ok_or(Errno::EINVAL)?;
        let previous = self.timer_value(pid, &t, now);
        let clock_now = self.timer_now(pid, t.clock, now);
        let expiry = if value == 0 {
            0
        } else if flags & clock::TIMER_ABSTIME != 0 {
            match t.clock {
                // An absolute wall-clock time: the same distance from now on the monotonic clock.
                TimerClock::Wall if t.realtime => {
                    let real = host.realtime_ns();
                    now.saturating_add(value.saturating_sub(real)).max(1)
                }
                _ => value.max(1),
            }
        } else {
            clock_now.saturating_add(value)
        };
        if let Some(t) = self.timers_of(tid)?.timers.get_mut(&id) {
            t.expiry = expiry;
            t.interval = if expiry == 0 { 0 } else { interval };
            t.overrun = 0;
        }
        if old != 0 {
            write_bytes(
                mm.borrow_mut().mem.as_mut(),
                old,
                &format_itimerspec(previous.0, previous.1),
            )?;
        }
        // An expiry already due fires before the caller runs again.
        self.expire_posix_timers(now);
        Ok(0)
    }

    /// `timer_gettime(id, curr)`.
    pub(super) fn sys_timer_gettime(
        &mut self,
        tid: Tid,
        id: u64,
        out: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let pid = self.pid_of(tid);
        let now = host.monotonic_ns();
        let t = self
            .procs
            .get(&pid)
            .and_then(|p| p.timers.timers.get(&(id as i32)))
            .ok_or(Errno::EINVAL)?;
        let (i, v) = self.timer_value(pid, t, now);
        write_bytes(
            self.mm(tid).borrow_mut().mem.as_mut(),
            out,
            &format_itimerspec(i, v),
        )?;
        Ok(0)
    }

    /// `timer_getoverrun(id)`.
    pub(super) fn sys_timer_getoverrun(&mut self, tid: Tid, id: u64) -> SysResult {
        let t = self
            .timers_of(tid)?
            .timers
            .get(&(id as i32))
            .ok_or(Errno::EINVAL)?;
        Ok(u64::from(t.overrun_last))
    }

    /// `timer_delete(id)`.
    pub(super) fn sys_timer_delete(&mut self, tid: Tid, id: u64) -> SysResult {
        self.timers_of(tid)?
            .timers
            .remove(&(id as i32))
            .map(|_| 0)
            .ok_or(Errno::EINVAL)
    }

    /// Fires every timer due at `now` (and CPU-time timers whose clock passed their expiry).
    pub(crate) fn expire_posix_timers(&mut self, now: u64) {
        let mut due = Vec::new();
        for (&pid, p) in &self.procs {
            for (&id, t) in &p.timers.timers {
                if t.expiry != 0 && t.expiry <= self.timer_now(pid, t.clock, now) {
                    due.push((pid, id));
                }
            }
        }
        for (pid, id) in due {
            let Some(t) = self.procs.get(&pid).and_then(|p| p.timers.timers.get(&id)) else {
                continue;
            };
            let clock_now = self.timer_now(pid, t.clock, now);
            let missed = clock_now
                .saturating_sub(t.expiry)
                .checked_div(t.interval)
                .unwrap_or(0);
            let notify = t.notify;
            let value = t.value;
            let pending = t.pending;
            let mut queue = None;
            if let Some(t) = self
                .procs
                .get_mut(&pid)
                .and_then(|p| p.timers.timers.get_mut(&id))
            {
                // Reload (periodic) or disarm.
                if t.interval == 0 {
                    t.expiry = 0;
                } else {
                    t.expiry = t
                        .expiry
                        .saturating_add((missed + 1).saturating_mul(t.interval));
                }
                let extra = u32::try_from(missed).unwrap_or(DELAYTIMER_MAX);
                if notify == Notify::None {
                    continue;
                }
                if pending {
                    t.overrun = t.overrun.saturating_add(extra.saturating_add(1));
                } else {
                    t.overrun = t.overrun.saturating_add(extra);
                    t.pending = true;
                    queue = Some(notify);
                }
                t.overrun = t.overrun.min(DELAYTIMER_MAX);
            }
            let info = |signo| SigInfo::timer(signo, id as u32, 0, value);
            let sent = match queue {
                Some(Notify::Process(s)) => self.queue_to_process(pid, info(s)),
                Some(Notify::Thread(target, s)) => {
                    if self.core.tasks.get(&target).map(|t| t.pid) == Some(pid) {
                        self.queue_to_thread(target, info(s))
                    } else {
                        self.queue_to_process(pid, info(s))
                    }
                }
                _ => Ok(0),
            };
            if sent.is_err()
                && let Some(t) = self
                    .procs
                    .get_mut(&pid)
                    .and_then(|p| p.timers.timers.get_mut(&id))
            {
                t.pending = false;
            }
        }
    }

    /// The earliest wall-clock timer expiry, if any (CPU-time timers advance only while their
    /// process runs, and are checked between slices).
    pub(crate) fn next_posix_timer(&self) -> Option<u64> {
        self.procs
            .values()
            .flat_map(|p| p.timers.timers.values())
            .filter(|t| t.clock == TimerClock::Wall && t.expiry != 0)
            .map(|t| t.expiry)
            .min()
    }

    /// A timer signal of process `pid` is being taken: it carries the overruns counted since it
    /// was queued, which `timer_getoverrun` reports from now on.
    pub(crate) fn timer_signal_taken(&mut self, pid: Pid, info: &mut SigInfo) {
        let id = info.timer_id() as i32;
        if let Some(t) = self
            .procs
            .get_mut(&pid)
            .and_then(|p| p.timers.timers.get_mut(&id))
            && t.pending
        {
            t.pending = false;
            t.overrun_last = t.overrun;
            t.overrun = 0;
            info.set_overrun(t.overrun_last);
        }
    }
}
