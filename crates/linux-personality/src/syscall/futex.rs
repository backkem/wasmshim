//! `futex`: wait queues keyed by (address space, address).
//!
//! Waiters are woken in arrival order. `FUTEX_WAIT` completes with 0 when woken and with
//! `ETIMEDOUT` at its deadline; it does not restart. Private and shared futexes use the same
//! key. Words in shared memory are keyed by (memory object, offset), so processes sharing an
//! object wake each other; any other word by (address space, address).

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::futex;
use wasmshim_linux_abi::structs::Timespec;

use super::Flow;
use crate::host::Host;
use crate::kernel::{Blocker, Kernel, Tid, WaitKey};
use crate::uaccess::read_array;

/// Marks futex keys of memory objects (address-space ids never reach this bit).
const SHARED_FUTEX: u64 = 1 << 63;

impl Kernel {
    /// A futex word in a shared mapping is keyed by its memory object (so processes sharing it
    /// wake each other); any other word by its address space.
    fn futex_key(&self, tid: Tid, addr: u64) -> WaitKey {
        let mm = self.mm(tid);
        let mm = mm.borrow();
        match mm.shared_at(addr) {
            Some((obj, off)) => WaitKey::Futex(SHARED_FUTEX | obj.id, off),
            None => WaitKey::Futex(mm.id, addr),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_futex(
        &mut self,
        tid: Tid,
        uaddr: u64,
        op: u64,
        val: u64,
        timeout: u64,
        uaddr2: u64,
        val3: u64,
        host: &mut dyn Host,
    ) -> Flow {
        let cmd = op & futex::FUTEX_CMD_MASK;
        if !uaddr.is_multiple_of(4) {
            return Flow::Ret(Errno::EINVAL.ret());
        }
        let key = self.futex_key(tid, uaddr);
        match cmd {
            futex::FUTEX_WAIT | futex::FUTEX_WAIT_BITSET => {
                let bitset = if cmd == futex::FUTEX_WAIT {
                    u32::MAX
                } else {
                    val3 as u32
                };
                if bitset == 0 {
                    return Flow::Ret(Errno::EINVAL.ret());
                }
                let current = match self.mm(tid).borrow().mem.read_u32(uaddr) {
                    Ok(v) => v,
                    Err(_) => return Flow::Ret(Errno::EFAULT.ret()),
                };
                if current != val as u32 {
                    return Flow::Ret(Errno::EAGAIN.ret());
                }
                let deadline = if timeout == 0 {
                    None
                } else {
                    let ts = match read_array::<16>(self.mm(tid).borrow_mut().mem.as_mut(), timeout)
                    {
                        Ok(b) => Timespec::from_bytes(&b),
                        Err(e) => return Flow::Ret(e.ret()),
                    };
                    let Some(ns) = ts.to_nanos_saturating() else {
                        return Flow::Ret(Errno::EINVAL.ret());
                    };
                    let now = host.monotonic_ns();
                    Some(if cmd == futex::FUTEX_WAIT {
                        now.saturating_add(ns)
                    } else if op & futex::FUTEX_CLOCK_REALTIME != 0 {
                        // Absolute CLOCK_REALTIME: convert to the monotonic clock.
                        now.saturating_add(ns.saturating_sub(host.realtime_ns()))
                    } else {
                        ns // absolute CLOCK_MONOTONIC
                    })
                };
                let seq = self.core.next_wait_seq;
                self.core.next_wait_seq += 1;
                Flow::Block(Blocker::Futex {
                    key,
                    bitset,
                    deadline,
                    seq,
                })
            }
            futex::FUTEX_WAKE | futex::FUTEX_WAKE_BITSET => {
                let bitset = if cmd == futex::FUTEX_WAKE {
                    u32::MAX
                } else {
                    val3 as u32
                };
                if bitset == 0 {
                    return Flow::Ret(Errno::EINVAL.ret());
                }
                let n = self.wake(key, (val as u32).min(i32::MAX as u32) as usize, bitset);
                Flow::Ret(n as u64)
            }
            futex::FUTEX_REQUEUE | futex::FUTEX_CMP_REQUEUE => {
                if cmd == futex::FUTEX_CMP_REQUEUE {
                    match self.mm(tid).borrow().mem.read_u32(uaddr) {
                        Ok(v) if v == val3 as u32 => {}
                        Ok(_) => return Flow::Ret(Errno::EAGAIN.ret()),
                        Err(_) => return Flow::Ret(Errno::EFAULT.ret()),
                    }
                }
                let woken = self.wake(key, (val as u32) as usize, u32::MAX);
                // For the requeue ops, the timeout argument is the requeue limit (`val2`).
                let key2 = self.futex_key(tid, uaddr2);
                let moved = self.requeue(key, key2, (timeout as u32) as usize);
                Flow::Ret(
                    (woken
                        + if cmd == futex::FUTEX_CMP_REQUEUE {
                            moved
                        } else {
                            0
                        }) as u64,
                )
            }
            _ => Flow::Ret(Errno::ENOSYS.ret()),
        }
    }
}
