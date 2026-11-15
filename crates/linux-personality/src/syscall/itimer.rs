//! `setitimer`/`getitimer` (and so `alarm`, which libc builds on them): per-process interval
//! timers. `ITIMER_REAL` runs on the monotonic clock (virtual under a deterministic host) and
//! delivers `SIGALRM`
//! from the scheduler like any deadline. `ITIMER_VIRTUAL` and `ITIMER_PROF` are stored and read
//! back but never fire (no profiler we run needs them yet).

use wasmshim_linux_abi::Errno;

use super::SysResult;
use crate::host::Host;
use crate::kernel::{Kernel, Tid};
use crate::uaccess::{read_array, write_bytes};

const ITIMER_REAL: u64 = 0;
const ITIMER_VIRTUAL: u64 = 1;
const ITIMER_PROF: u64 = 2;

/// One interval timer: the absolute expiry (monotonic ns; for the CPU-time timers, the value
/// as set) and the reload interval, both 0 when disarmed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ITimer {
    pub expiry: u64,
    pub interval: u64,
}

/// `struct itimerval` (two 64-bit `struct timeval`s: interval, then value) to ns.
fn parse(b: &[u8; 32]) -> Result<(u64, u64), Errno> {
    let f = |o: usize| i64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes"));
    let tv = |s: i64, us: i64| {
        if s < 0 || !(0..1_000_000).contains(&us) {
            return Err(Errno::EINVAL);
        }
        Ok((s as u64).saturating_mul(1_000_000_000) + us as u64 * 1000)
    };
    Ok((tv(f(0), f(8))?, tv(f(16), f(24))?))
}

fn format(interval: u64, value: u64) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, ns) in [interval, value].into_iter().enumerate() {
        // Linux rounds a remaining time up to a whole microsecond.
        let us = ns.div_ceil(1000);
        b[i * 16..i * 16 + 8].copy_from_slice(&(us / 1_000_000).to_le_bytes());
        b[i * 16 + 8..i * 16 + 16].copy_from_slice(&(us % 1_000_000).to_le_bytes());
    }
    b
}

impl Kernel {
    fn itimer_slot(&mut self, tid: Tid, which: u64) -> Result<&mut ITimer, Errno> {
        let pid = self.pid_of(tid);
        let p = self.procs.get_mut(&pid).ok_or(Errno::ESRCH)?;
        match which {
            ITIMER_REAL => Ok(&mut p.itimers[0]),
            ITIMER_VIRTUAL => Ok(&mut p.itimers[1]),
            ITIMER_PROF => Ok(&mut p.itimers[2]),
            _ => Err(Errno::EINVAL),
        }
    }

    fn itimer_value(t: ITimer, which: u64, now: u64) -> (u64, u64) {
        let left = if which == ITIMER_REAL {
            t.expiry.saturating_sub(now)
        } else {
            t.expiry
        };
        (t.interval, if t.expiry == 0 { 0 } else { left.max(1) })
    }

    pub(super) fn sys_getitimer(
        &mut self,
        tid: Tid,
        which: u64,
        out: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let now = host.monotonic_ns();
        let t = *self.itimer_slot(tid, which)?;
        let (i, v) = Self::itimer_value(t, which, now);
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), out, &format(i, v))?;
        Ok(0)
    }

    pub(super) fn sys_setitimer(
        &mut self,
        tid: Tid,
        which: u64,
        new: u64,
        old: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let now = host.monotonic_ns();
        let (interval, value) = if new == 0 {
            (0, 0)
        } else {
            parse(&read_array(self.mm(tid).borrow_mut().mem.as_mut(), new)?)?
        };
        let prev = *self.itimer_slot(tid, which)?;
        let next = if value == 0 {
            ITimer::default()
        } else {
            ITimer {
                expiry: if which == ITIMER_REAL {
                    now.saturating_add(value)
                } else {
                    value
                },
                interval,
            }
        };
        *self.itimer_slot(tid, which)? = next;
        if old != 0 {
            let (i, v) = Self::itimer_value(prev, which, now);
            write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), old, &format(i, v))?;
        }
        Ok(0)
    }

    /// Fires every `ITIMER_REAL` due at `now` (`SIGALRM`, then reload or disarm).
    pub(crate) fn expire_itimers(&mut self, now: u64) {
        let due: Vec<crate::kernel::Pid> = self
            .procs
            .iter()
            .filter(|(_, p)| p.itimers[0].expiry != 0 && p.itimers[0].expiry <= now)
            .map(|(&pid, _)| pid)
            .collect();
        for pid in due {
            if let Some(p) = self.procs.get_mut(&pid) {
                let t = &mut p.itimers[0];
                // Overruns collapse into one signal, as on Linux.
                match (now - t.expiry).checked_div(t.interval) {
                    None => *t = ITimer::default(),
                    Some(behind) => t.expiry += (behind + 1) * t.interval,
                }
            }
            let _ = self.send_to_process(pid, wasmshim_linux_abi::consts::signal::SIGALRM);
        }
    }

    /// The earliest armed `ITIMER_REAL`, if any.
    pub(crate) fn next_itimer(&self) -> Option<u64> {
        self.procs
            .values()
            .map(|p| p.itimers[0].expiry)
            .filter(|&e| e != 0)
            .min()
    }
}
