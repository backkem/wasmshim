//! Time, randomness and system information.

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{clock, random};
use wasmshim_linux_abi::structs::{Timespec, Timeval, Utsname};

use super::{Flow, SysResult};
use crate::host::Host;
use crate::kernel::Blocker;
use crate::kernel::{Kernel, Tid};
use crate::uaccess::{read_array, write_bytes};

impl Kernel {
    /// The value of `clk` in nanoseconds, or `EINVAL` for unknown clocks. Negative ids are the
    /// CPU clocks of other processes and threads (`clock_getcpuclockid`,
    /// `pthread_getcpuclockid`: `~pid << 3 | type`, bit 2 for a thread).
    fn clock_ns(&self, tid: Tid, clk: u64, host: &mut dyn Host) -> Result<u64, Errno> {
        if (clk as i64) < 0 {
            let (pid, thread) = self.cpu_clock_target(tid, clk)?;
            return Ok(self.cpu_time(pid, thread));
        }
        match clk {
            clock::CLOCK_REALTIME
            | clock::CLOCK_REALTIME_COARSE
            | clock::CLOCK_REALTIME_ALARM
            // The TAI offset is 0 until something sets it (nothing here does).
            | clock::CLOCK_TAI => Ok(host.realtime_ns()),
            clock::CLOCK_MONOTONIC
            | clock::CLOCK_MONOTONIC_RAW
            | clock::CLOCK_MONOTONIC_COARSE
            | clock::CLOCK_BOOTTIME
            | clock::CLOCK_BOOTTIME_ALARM => Ok(host.monotonic_ns()),
            // CPU time is retired instructions at the configured virtual rate.
            clock::CLOCK_PROCESS_CPUTIME_ID => Ok(self.cpu_time(self.pid_of(tid), None)),
            clock::CLOCK_THREAD_CPUTIME_ID => Ok(self.cpu_time(self.pid_of(tid), Some(tid))),
            _ => Err(Errno::EINVAL),
        }
    }

    /// The process (and thread) a negative CPU clock id names.
    fn cpu_clock_target(&self, tid: Tid, clk: u64) -> Result<(u32, Option<Tid>), Errno> {
        const CPUCLOCK_PERTHREAD: u64 = 4;
        const CPUCLOCK_MAX: u64 = 3;
        if clk & 3 == CPUCLOCK_MAX {
            return Err(Errno::EINVAL);
        }
        let id = (!(clk as i64 >> 3)) as u32;
        if clk & CPUCLOCK_PERTHREAD != 0 {
            let t = if id == 0 { tid } else { id };
            let pid = self
                .core
                .tasks
                .get(&t)
                .map(|t| t.pid)
                .ok_or(Errno::EINVAL)?;
            Ok((pid, Some(t)))
        } else {
            let pid = if id == 0 { self.pid_of(tid) } else { id };
            if !self.procs.contains_key(&pid) {
                return Err(Errno::EINVAL);
            }
            Ok((pid, None))
        }
    }

    pub(super) fn sys_clock_gettime(
        &mut self,
        tid: Tid,
        clk: u64,
        tp: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let ns = self.clock_ns(tid, clk, host)?;
        write_bytes(
            self.mm(tid).borrow_mut().mem.as_mut(),
            tp,
            &Timespec::from_nanos(ns).to_bytes(),
        )?;
        Ok(0)
    }

    pub(super) fn sys_clock_getres(&mut self, tid: Tid, clk: u64, tp: u64) -> SysResult {
        if (clk as i64) < 0 {
            self.cpu_clock_target(tid, clk)?;
        } else if !matches!(clk, 0..=clock::CLOCK_BOOTTIME_ALARM | clock::CLOCK_TAI) {
            return Err(Errno::EINVAL);
        }
        if tp != 0 {
            write_bytes(
                self.mm(tid).borrow_mut().mem.as_mut(),
                tp,
                &Timespec { sec: 0, nsec: 1 }.to_bytes(),
            )?;
        }
        Ok(0)
    }

    /// `clock_settime`: the clocks are virtual and owned by the host, so setting one is refused
    /// like in a container without `CAP_SYS_TIME` (`EPERM`); clocks that cannot be set at all
    /// are `EINVAL`, as on Linux.
    pub(super) fn sys_clock_settime(&mut self, tid: Tid, clk: u64, tp: u64) -> SysResult {
        let ts = Timespec::from_bytes(&read_array(self.mm(tid).borrow_mut().mem.as_mut(), tp)?);
        match clk {
            clock::CLOCK_REALTIME => {
                ts.to_nanos().ok_or(Errno::EINVAL)?;
                Err(Errno::EPERM)
            }
            _ => Err(Errno::EINVAL),
        }
    }

    /// `settimeofday`: refused like `clock_settime` (`EPERM`).
    pub(super) fn sys_settimeofday(&mut self, tid: Tid, tv: u64, tz: u64) -> SysResult {
        if tv != 0 {
            let b: [u8; 16] = read_array(self.mm(tid).borrow_mut().mem.as_mut(), tv)?;
            let us = i64::from_le_bytes(b[8..16].try_into().unwrap_or_default());
            let s = i64::from_le_bytes(b[0..8].try_into().unwrap_or_default());
            if s < 0 || !(0..1_000_000).contains(&us) {
                return Err(Errno::EINVAL);
            }
        } else if tz == 0 {
            return Ok(0);
        }
        Err(Errno::EPERM)
    }

    pub(super) fn sys_gettimeofday(&mut self, tid: Tid, tv: u64, host: &mut dyn Host) -> SysResult {
        if tv != 0 {
            let now = host.realtime_ns();
            write_bytes(
                self.mm(tid).borrow_mut().mem.as_mut(),
                tv,
                &Timeval::from_nanos(now).to_bytes(),
            )?;
        }
        Ok(0)
    }

    pub(super) fn sys_times(&mut self, tid: Tid, buf: u64) -> SysResult {
        // clock_t ticks at AT_CLKTCK = 100 Hz.
        let ticks = self.retired().saturating_mul(self.cfg.ns_per_insn.max(1)) / 10_000_000;
        if buf != 0 {
            let mut b = [0u8; 32];
            b[..8].copy_from_slice(&ticks.to_le_bytes());
            write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &b)?;
        }
        Ok(ticks)
    }

    fn sleep_request(&self, tid: Tid, req: u64) -> Result<u64, Errno> {
        let ts = Timespec::from_bytes(&read_array(self.mm(tid).borrow_mut().mem.as_mut(), req)?);
        ts.to_nanos_saturating().ok_or(Errno::EINVAL)
    }

    pub(super) fn sys_nanosleep(
        &mut self,
        tid: Tid,
        req: u64,
        rem: u64,
        host: &mut dyn Host,
    ) -> Flow {
        match self.sleep_request(tid, req) {
            Ok(ns) => Flow::Block(Blocker::Sleep {
                until: host.monotonic_ns().saturating_add(ns),
                rem,
            }),
            Err(e) => Flow::Ret(e.ret()),
        }
    }

    /// `clock_nanosleep(clk, flags, req, rem)`. An absolute time on a realtime clock is the
    /// same distance away on the monotonic clock (the clocks are never stepped); `rem` is
    /// written only for an interrupted relative sleep, as on Linux.
    pub(super) fn sys_clock_nanosleep(
        &mut self,
        tid: Tid,
        clk: u64,
        flags: u64,
        req: u64,
        rem: u64,
        host: &mut dyn Host,
    ) -> Flow {
        let realtime = match clk {
            clock::CLOCK_REALTIME | clock::CLOCK_REALTIME_ALARM | clock::CLOCK_TAI => true,
            clock::CLOCK_MONOTONIC | clock::CLOCK_BOOTTIME | clock::CLOCK_BOOTTIME_ALARM => false,
            // Sleeping on a CPU clock is not modelled (the thread's own would never advance).
            clock::CLOCK_PROCESS_CPUTIME_ID => return Flow::Ret(Errno::EOPNOTSUPP.ret()),
            _ => return Flow::Ret(Errno::EINVAL.ret()),
        };
        let ns = match self.sleep_request(tid, req) {
            Ok(ns) => ns,
            Err(e) => return Flow::Ret(e.ret()),
        };
        let now_mono = host.monotonic_ns();
        if flags & clock::TIMER_ABSTIME != 0 {
            let until = if realtime {
                now_mono.saturating_add(ns.saturating_sub(host.realtime_ns()))
            } else {
                ns
            };
            return Flow::Block(Blocker::Sleep { until, rem: 0 });
        }
        Flow::Block(Blocker::Sleep {
            until: now_mono.saturating_add(ns),
            rem,
        })
    }

    pub(super) fn sys_getrandom(
        &mut self,
        tid: Tid,
        buf: u64,
        len: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if flags & !(random::GRND_NONBLOCK | random::GRND_RANDOM | random::GRND_INSECURE) != 0 {
            return Err(Errno::EINVAL);
        }
        let len = crate::uaccess::io_len(len);
        let mut data = vec![0u8; len];
        host.random(&mut data);
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &data)?;
        Ok(len as u64)
    }

    pub(super) fn sys_uname(&mut self, tid: Tid, buf: u64) -> SysResult {
        let u = Utsname {
            sysname: "Linux",
            nodename: &self.cfg.hostname,
            release: &self.cfg.kernel_release,
            version: &self.cfg.kernel_version,
            machine: self.arch.machine(),
            domainname: "(none)",
        };
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &u.to_bytes())?;
        Ok(0)
    }

    pub(super) fn sys_sysinfo(&mut self, tid: Tid, buf: u64, host: &mut dyn Host) -> SysResult {
        let mut b = [0u8; 112];
        let uptime = host.monotonic_ns() / 1_000_000_000;
        b[..8].copy_from_slice(&uptime.to_le_bytes());
        b[32..40].copy_from_slice(&(4u64 << 30).to_le_bytes()); // totalram
        b[40..48].copy_from_slice(&(3u64 << 30).to_le_bytes()); // freeram
        b[80..82].copy_from_slice(&(self.procs.len() as u16).to_le_bytes());
        b[104..108].copy_from_slice(&1u32.to_le_bytes()); // mem_unit
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &b)?;
        Ok(0)
    }
}
