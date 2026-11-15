//! Syscall dispatch. Handlers live in topic modules and return `Result<u64, Errno>`; the few
//! that change control flow (exit, sleep, execve) return a [`Flow`] directly.

mod aio;
mod creds;
mod fileops;
mod flock;
pub(crate) use flock::description_id;
mod fs;
mod futex;
mod itimer;
mod mem;
mod misc;
mod msg;
mod net;
mod netlink;
mod process;
mod ptimer;
mod sem;
mod shm;
mod tty;

pub(crate) use aio::AioCtx;
pub(crate) use creds::Creds;
pub(crate) use itimer::ITimer;
pub(crate) use msg::MsgTable;
pub(crate) use ptimer::PosixTimers;
pub(crate) use sem::{SemTable, SemUndo};

use std::cell::RefCell;
use std::rc::Rc;

use wasmshim_cpu::SyscallFrame;
use wasmshim_linux_abi::{Errno, Sysno};

use crate::fd::{FdTable, FileRef};
use crate::host::Host;
use crate::kernel::{Kernel, Linux, Pid, Tid, WaitKey};
use crate::mm::AddressSpace;

/// What happens to the calling thread after a syscall: the substrate's flow (`Ret`, `Exec`
/// after `execve`, `Block`, `ExitThread` for `exit`, `ExitGroup` for `exit_group`).
pub(crate) type Flow = wasmshim_kernel::FlowOf<Linux>;

pub(crate) type SysResult = Result<u64, Errno>;

/// Internal pseudo-errno (Linux's `ERESTARTSYS`): the handler called [`Kernel::wait_for`] and
/// the thread must block and re-run the syscall when woken. Never reaches the guest.
pub(crate) const BLOCK: Errno = Errno(512);

impl Kernel {
    /// Arranges for the current syscall to block on `key` (and to return `timeout_result` at
    /// `deadline`); returns [`BLOCK`] for the handler to propagate.
    pub(crate) fn wait_for(
        &mut self,
        key: WaitKey,
        deadline: Option<u64>,
        timeout_result: Option<u64>,
    ) -> Errno {
        self.core.wait_for(key, deadline, timeout_result);
        BLOCK
    }

    /// Drops a wait the current syscall arranged but no longer needs (a later part of the
    /// call made progress).
    pub(crate) fn cancel_wait(&mut self) {
        self.core.cancel_wait();
    }

    /// Blocks the current syscall on socket object `id`, for at most `timeout` ns
    /// (`SO_RCVTIMEO`/`SO_SNDTIMEO`: the call then fails with `EAGAIN`). The deadline counts
    /// from the call's first entry and survives its restarts ([`Kernel::restart_deadline`]).
    pub(crate) fn wait_sock(&mut self, id: u64, timeout: Option<u64>) -> Errno {
        let deadline = self.restart_deadline(self.core.current, timeout, self.core.slice_now);
        self.wait_for(
            WaitKey::Object(id),
            deadline,
            timeout.map(|_| Errno::EAGAIN.ret()),
        )
    }

    /// Turns a handler result into a [`Flow`], honouring [`BLOCK`].
    fn finish(&mut self, r: SysResult, frame: &SyscallFrame) -> Flow {
        match r {
            Ok(v) => Flow::Ret(v),
            Err(BLOCK) => match self.core.take_block(frame.pc) {
                Some(blocker) => Flow::Block(blocker),
                None => Flow::Ret(Errno::EINTR.ret()),
            },
            Err(e) => Flow::Ret(e.ret()),
        }
    }
}

impl Kernel {
    pub(crate) fn pid_of(&self, tid: Tid) -> Pid {
        self.core.tasks.get(&tid).map_or(0, |t| t.pid)
    }

    pub(crate) fn mm(&self, tid: Tid) -> Rc<RefCell<AddressSpace>> {
        let pid = self.pid_of(tid);
        self.procs[&pid].mm.clone()
    }

    pub(crate) fn fds(&self, tid: Tid) -> Rc<RefCell<FdTable>> {
        let pid = self.pid_of(tid);
        self.procs[&pid].fds.clone()
    }

    pub(crate) fn file(&self, tid: Tid, fd: u64) -> Result<FileRef, Errno> {
        self.fds(tid).borrow().get(fd as i32)
    }

    /// Handles one syscall. The substrate has done the entry bookkeeping (`current`,
    /// `current_pc`, `slice_now` taken at entry) and traces the result.
    pub(crate) fn dispatch(&mut self, tid: Tid, frame: SyscallFrame, host: &mut dyn Host) -> Flow {
        let Some(sys) = self.arch.sysno(frame.number) else {
            return Flow::Ret(Errno::ENOSYS.ret());
        };
        let a = frame.args;
        let r: SysResult = match sys {
            // ---- process lifetime ----
            Sysno::exit => return Flow::ExitThread(a[0] as i32),
            Sysno::exit_group => return Flow::ExitGroup(a[0] as i32),
            Sysno::execve => return self.sys_execve(tid, a[0], a[1], a[2], host),
            Sysno::execveat => {
                return self.sys_execveat(tid, a[0], a[1], a[2], a[3], a[4], host);
            }
            Sysno::clone => return self.sys_clone(tid, a[0], a[1], a[2], a[3], a[4], host),
            Sysno::clone3 => Err(Errno::ENOSYS),
            Sysno::wait4 => return self.sys_wait4(tid, a[0], a[1], a[2], a[3], frame.pc),
            Sysno::rt_sigreturn => return self.sys_rt_sigreturn(tid, host),
            Sysno::rt_sigsuspend => return self.sys_rt_sigsuspend(tid, a[0], a[1]),
            Sysno::waitid => {
                return self.sys_waitid(tid, a[0], a[1], a[2], a[3], a[4], frame.pc);
            }
            Sysno::rt_sigpending => self.sys_rt_sigpending(tid, a[0], a[1]),
            Sysno::rt_sigtimedwait => self.sys_rt_sigtimedwait(tid, a[0], a[1], a[2], a[3], host),
            Sysno::rt_sigqueueinfo => self.sys_rt_sigqueueinfo(tid, a[0], a[1], a[2]),
            Sysno::rt_tgsigqueueinfo => self.sys_rt_tgsigqueueinfo(tid, a[0], a[1], a[2], a[3]),
            Sysno::signalfd4 => self.sys_signalfd4(tid, a[0], a[1], a[2], a[3]),
            Sysno::pidfd_open => self.sys_pidfd_open(tid, a[0], a[1]),
            Sysno::pidfd_send_signal => self.sys_pidfd_send_signal(tid, a[0], a[1], a[2], a[3]),
            Sysno::pidfd_getfd => self.sys_pidfd_getfd(tid, a[0], a[1], a[2]),
            Sysno::futex => return self.sys_futex(tid, a[0], a[1], a[2], a[3], a[4], a[5], host),
            Sysno::pipe2 => self.sys_pipe2(tid, a[0], a[1]),
            Sysno::eventfd2 => self.sys_eventfd2(tid, a[0], a[1]),
            Sysno::timerfd_create => self.sys_timerfd_create(tid, a[0], a[1]),
            Sysno::timerfd_settime => self.sys_timerfd_settime(tid, a[0], a[1], a[2], a[3], host),
            Sysno::timerfd_gettime => self.sys_timerfd_gettime(tid, a[0], a[1], host),
            Sysno::inotify_init1 => self.sys_inotify_init1(tid, a[0]),
            Sysno::inotify_add_watch => self.sys_inotify_add_watch(tid, a[0], a[1], a[2]),
            Sysno::inotify_rm_watch => self.sys_inotify_rm_watch(tid, a[0], a[1]),
            Sysno::epoll_create1 => self.sys_epoll_create1(tid, a[0]),
            Sysno::epoll_ctl => self.sys_epoll_ctl(tid, a[0], a[1], a[2], a[3]),
            Sysno::epoll_pwait => {
                let now = host.monotonic_ns();
                let timeout = match a[3] as i32 {
                    t if t < 0 => None,
                    t => Some(u64::from(t as u32) * 1_000_000),
                };
                let deadline = self.restart_deadline(tid, timeout, now);
                self.with_call_mask(tid, a[4], a[5], |k| {
                    k.epoll_wait(tid, a[0], a[1], a[2], deadline, host)
                })
            }
            Sysno::epoll_pwait2 => match self.read_timeout(tid, a[3], host) {
                Ok(deadline) => self.with_call_mask(tid, a[4], a[5], |k| {
                    k.epoll_wait(tid, a[0], a[1], a[2], deadline, host)
                }),
                Err(e) => Err(e),
            },
            Sysno::socket => self.sys_socket(tid, a[0], a[1], a[2]),
            Sysno::socketpair => self.sys_socketpair(tid, a[0], a[1], a[2], a[3]),
            Sysno::bind => self.sys_bind(tid, a[0], a[1], a[2], host),
            Sysno::listen => self.sys_listen(tid, a[0], a[1]),
            Sysno::accept => self.sys_accept4(tid, a[0], a[1], a[2], 0),
            Sysno::accept4 => self.sys_accept4(tid, a[0], a[1], a[2], a[3]),
            Sysno::connect => self.sys_connect(tid, a[0], a[1], a[2]),
            Sysno::getsockname => self.sys_getsockname(tid, a[0], a[1], a[2], false),
            Sysno::getpeername => self.sys_getsockname(tid, a[0], a[1], a[2], true),
            Sysno::sendto => self.sys_sendto(tid, a[0], a[1], a[2], a[3], a[4], a[5]),
            Sysno::recvfrom => self.sys_recvfrom(tid, a[0], a[1], a[2], a[3], a[4], a[5]),
            Sysno::sendmsg => self.sys_sendmsg(tid, a[0], a[1], a[2]),
            Sysno::recvmsg => self.sys_recvmsg(tid, a[0], a[1], a[2]),
            Sysno::sendmmsg => self.sys_sendmmsg(tid, a[0], a[1], a[2], a[3]),
            Sysno::recvmmsg => self.sys_recvmmsg(tid, a[0], a[1], a[2], a[3], a[4]),
            Sysno::shutdown => self.sys_shutdown(tid, a[0], a[1]),
            Sysno::getsockopt => self.sys_getsockopt(tid, a[0], a[1], a[2], a[3], a[4]),
            Sysno::setsockopt => self.sys_setsockopt(tid, a[0], a[1], a[2], a[3], a[4]),
            Sysno::timer_create => self.sys_timer_create(tid, a[0], a[1], a[2]),
            Sysno::timer_settime => self.sys_timer_settime(tid, a[0], a[1], a[2], a[3], host),
            Sysno::timer_gettime => self.sys_timer_gettime(tid, a[0], a[1], host),
            Sysno::timer_getoverrun => self.sys_timer_getoverrun(tid, a[0]),
            Sysno::timer_delete => self.sys_timer_delete(tid, a[0]),
            Sysno::clock_settime => self.sys_clock_settime(tid, a[0], a[1]),
            Sysno::settimeofday => self.sys_settimeofday(tid, a[0], a[1]),
            Sysno::getitimer => self.sys_getitimer(tid, a[0], a[1], host),
            Sysno::setitimer => self.sys_setitimer(tid, a[0], a[1], a[2], host),
            Sysno::nanosleep => return self.sys_nanosleep(tid, a[0], a[1], host),
            Sysno::clock_nanosleep => {
                return self.sys_clock_nanosleep(tid, a[0], a[1], a[2], a[3], host);
            }

            // ---- files ----
            Sysno::read => self.sys_read(tid, a[0], a[1], a[2], None, host),
            Sysno::pread64 => self.sys_read(tid, a[0], a[1], a[2], Some(a[3]), host),
            Sysno::write => self.sys_write(tid, a[0], a[1], a[2], None, host),
            Sysno::pwrite64 => self.sys_write(tid, a[0], a[1], a[2], Some(a[3]), host),
            Sysno::readv => self.sys_readv(tid, a[0], a[1], a[2], None, host),
            Sysno::preadv => self.sys_readv(tid, a[0], a[1], a[2], Some(a[3]), host),
            Sysno::writev => self.sys_writev(tid, a[0], a[1], a[2], None, 0, host),
            Sysno::pwritev => self.sys_writev(tid, a[0], a[1], a[2], Some(a[3]), 0, host),
            Sysno::io_setup => self.sys_io_setup(tid, a[0], a[1]),
            Sysno::io_destroy => self.sys_io_destroy(tid, a[0]),
            Sysno::io_submit => self.sys_io_submit(tid, a[0], a[1], a[2], host),
            Sysno::io_cancel => self.sys_io_cancel(tid, a[0]),
            Sysno::io_getevents | Sysno::io_pgetevents => {
                self.sys_io_getevents(tid, a[0], a[1], a[2], a[3], a[4], host)
            }
            Sysno::preadv2 => self.sys_rwv2(tid, &a, false, host),
            Sysno::pwritev2 => self.sys_rwv2(tid, &a, true, host),
            Sysno::lseek => self.sys_lseek(tid, a[0], a[1], a[2]),
            Sysno::openat => self.sys_openat(tid, a[0], a[1], a[2], a[3], host),
            Sysno::close => self.sys_close(tid, a[0]),
            Sysno::close_range => self.sys_close_range(tid, a[0], a[1], a[2]),
            Sysno::dup => self.sys_dup(tid, a[0]),
            Sysno::dup3 => self.sys_dup3(tid, a[0], a[1], a[2]),
            Sysno::fcntl => self.sys_fcntl(tid, a[0], a[1], a[2]),
            Sysno::ioctl => self.sys_ioctl(tid, a[0], a[1], a[2]),
            Sysno::getdents64 => self.sys_getdents64(tid, a[0], a[1], a[2]),
            Sysno::newfstatat => self.sys_newfstatat(tid, a[0], a[1], a[2], a[3]),
            Sysno::fstat => self.sys_fstat(tid, a[0], a[1]),
            Sysno::statx => self.sys_statx(tid, a[0], a[1], a[2], a[3], a[4]),
            Sysno::statfs => self.sys_statfs(tid, a[0], a[1]),
            Sysno::fstatfs => self.sys_fstatfs(tid, a[0], a[1]),
            Sysno::readlinkat => self.sys_readlinkat(tid, a[0], a[1], a[2], a[3]),
            Sysno::faccessat => self.sys_faccessat(tid, a[0], a[1], a[2], 0),
            Sysno::faccessat2 => self.sys_faccessat(tid, a[0], a[1], a[2], a[3]),
            Sysno::mkdirat => self.sys_mkdirat(tid, a[0], a[1], a[2], host),
            Sysno::mknodat => self.sys_mknodat(tid, a[0], a[1], a[2], host),
            Sysno::unlinkat => self.sys_unlinkat(tid, a[0], a[1], a[2], host),
            Sysno::renameat => self.sys_renameat2(tid, a[0], a[1], a[2], a[3], 0, host),
            Sysno::renameat2 => self.sys_renameat2(tid, a[0], a[1], a[2], a[3], a[4], host),
            Sysno::symlinkat => self.sys_symlinkat(tid, a[0], a[1], a[2], host),
            Sysno::linkat => self.sys_linkat(tid, a[0], a[1], a[2], a[3], a[4], host),
            Sysno::fchmod => self.sys_fchmod(tid, a[0], a[1], host),
            Sysno::fchmodat => self.sys_fchmodat(tid, a[0], a[1], a[2], host),
            Sysno::fchown => self.sys_fchown(tid, a[0], a[1], a[2], host),
            Sysno::fchownat => self.sys_fchownat(tid, a[0], a[1], a[2], a[3], a[4], host),
            Sysno::utimensat => self.sys_utimensat(tid, a[0], a[1], a[2], a[3], host),
            Sysno::truncate => self.sys_truncate(tid, a[0], a[1], host),
            Sysno::ftruncate => self.sys_ftruncate(tid, a[0], a[1], host),
            Sysno::fallocate => self.sys_fallocate(tid, a[0], a[1], a[2], a[3], host),
            Sysno::setxattr => self.xattr_call(tid, fileops::XattrOp::Set, 0, &a, host),
            Sysno::lsetxattr => self.xattr_call(tid, fileops::XattrOp::Set, 1, &a, host),
            Sysno::fsetxattr => self.xattr_call(tid, fileops::XattrOp::Set, 2, &a, host),
            Sysno::getxattr => self.xattr_call(tid, fileops::XattrOp::Get, 0, &a, host),
            Sysno::lgetxattr => self.xattr_call(tid, fileops::XattrOp::Get, 1, &a, host),
            Sysno::fgetxattr => self.xattr_call(tid, fileops::XattrOp::Get, 2, &a, host),
            Sysno::listxattr => self.xattr_call(tid, fileops::XattrOp::List, 0, &a, host),
            Sysno::llistxattr => self.xattr_call(tid, fileops::XattrOp::List, 1, &a, host),
            Sysno::flistxattr => self.xattr_call(tid, fileops::XattrOp::List, 2, &a, host),
            Sysno::removexattr => self.xattr_call(tid, fileops::XattrOp::Remove, 0, &a, host),
            Sysno::lremovexattr => self.xattr_call(tid, fileops::XattrOp::Remove, 1, &a, host),
            Sysno::fremovexattr => self.xattr_call(tid, fileops::XattrOp::Remove, 2, &a, host),
            Sysno::copy_file_range => {
                self.sys_copy_file_range(tid, a[0], a[1], a[2], a[3], a[4], a[5], host)
            }
            Sysno::splice => self.sys_splice(tid, a[0], a[1], a[2], a[3], a[4], a[5], host),
            Sysno::tee => self.sys_tee(tid, a[0], a[1], a[2], a[3]),
            Sysno::vmsplice => self.sys_vmsplice(tid, a[0], a[1], a[2], a[3]),
            Sysno::openat2 => self.sys_openat2(tid, a[0], a[1], a[2], a[3], host),
            Sysno::readahead => self.sys_readahead(tid, a[0]),
            Sysno::getcwd => self.sys_getcwd(tid, a[0], a[1]),
            Sysno::chdir => self.sys_chdir(tid, a[0]),
            Sysno::fchdir => self.sys_fchdir(tid, a[0]),
            Sysno::chroot => Err(Errno::EPERM),
            Sysno::umask => self.sys_umask(tid, a[0]),
            Sysno::ppoll => self.with_call_mask(tid, a[3], a[4], |k| {
                k.sys_ppoll(tid, a[0], a[1], a[2], host)
            }),
            Sysno::pselect6 => match self.pselect6_mask(tid, a[5]) {
                Ok((mask, size)) => self.with_call_mask(tid, mask, size, |k| {
                    k.sys_pselect6(tid, a[0], a[1], a[2], a[3], a[4], host)
                }),
                Err(e) => Err(e),
            },
            Sysno::sync | Sysno::syncfs => Ok(0),
            Sysno::sync_file_range => self.sys_sync_file_range(tid, a[0], a[1], a[2], a[3]),
            Sysno::fsync | Sysno::fdatasync | Sysno::fadvise64 => self.file(tid, a[0]).map(|_| 0),
            Sysno::flock => self.sys_flock(tid, a[0], a[1]),
            Sysno::sendfile => self.sys_sendfile(tid, a[0], a[1], a[2], a[3], host),

            // ---- memory ----
            Sysno::brk => self.sys_brk(tid, a[0]),
            Sysno::mmap => self.sys_mmap(tid, a[0], a[1], a[2], a[3], a[4], a[5], host),
            Sysno::munmap => self.sys_munmap(tid, a[0], a[1], host),
            Sysno::msync => self.sys_msync(tid, a[0], a[1], a[2], host),
            Sysno::mprotect => self.sys_mprotect(tid, a[0], a[1], a[2]),
            Sysno::mremap => self.sys_mremap(tid, a[0], a[1], a[2], a[3], a[4], host),
            Sysno::madvise => self.sys_madvise(tid, a[0], a[1], a[2]),
            Sysno::mlock | Sysno::munlock | Sysno::mlockall | Sysno::munlockall | Sysno::mlock2 => {
                Ok(0)
            }
            Sysno::membarrier => Ok(0),
            Sysno::memfd_create => self.sys_memfd_create(tid, a[0], a[1], host),
            Sysno::shmget => self.sys_shmget(tid, a[0], a[1], a[2], host),
            Sysno::shmat => self.sys_shmat(tid, a[0], a[1], a[2], host),
            Sysno::shmdt => self.sys_shmdt(tid, a[0], host),
            Sysno::shmctl => self.sys_shmctl(tid, a[0], a[1], a[2], host),
            Sysno::semget => self.sys_semget(tid, a[0], a[1], a[2], host),
            Sysno::semop => self.sys_semtimedop(tid, a[0], a[1], a[2], 0, host),
            Sysno::semtimedop => self.sys_semtimedop(tid, a[0], a[1], a[2], a[3], host),
            Sysno::semctl => self.sys_semctl(tid, a[0], a[1], a[2], a[3], host),
            Sysno::msgget => self.sys_msgget(tid, a[0], a[1], host),
            Sysno::msgsnd => self.sys_msgsnd(tid, a[0], a[1], a[2], a[3], host),
            Sysno::msgrcv => self.sys_msgrcv(tid, a[0], a[1], a[2], a[3], a[4], host),
            Sysno::msgctl => self.sys_msgctl(tid, a[0], a[1], a[2], host),
            Sysno::mq_open => self.sys_mq_open(tid, a[0], a[1], a[2], a[3]),
            Sysno::mq_unlink => self.sys_mq_unlink(tid, a[0]),
            Sysno::mq_timedsend => self.sys_mq_timedsend(tid, a[0], a[1], a[2], a[3], a[4], host),
            Sysno::mq_timedreceive => {
                self.sys_mq_timedreceive(tid, a[0], a[1], a[2], a[3], a[4], host)
            }
            Sysno::mq_notify => self.sys_mq_notify(tid, a[0], a[1]),
            Sysno::mq_getsetattr => self.sys_mq_getsetattr(tid, a[0], a[1], a[2]),

            // ---- process, ids, signals ----
            Sysno::set_tid_address => self.sys_set_tid_address(tid, a[0]),
            Sysno::set_robust_list => Ok(0),
            Sysno::getpid => Ok(u64::from(self.pid_of(tid))),
            Sysno::gettid => Ok(u64::from(tid)),
            Sysno::getppid => Ok(u64::from(self.procs[&self.pid_of(tid)].ppid)),
            Sysno::getuid | Sysno::geteuid | Sysno::getgid | Sysno::getegid => {
                self.sys_getid(tid, sys)
            }
            Sysno::getresuid => self.sys_getresid(tid, a[0], a[1], a[2], false),
            Sysno::getresgid => self.sys_getresid(tid, a[0], a[1], a[2], true),
            Sysno::setuid => self.sys_setid(tid, a[0], false),
            Sysno::setgid => self.sys_setid(tid, a[0], true),
            Sysno::setreuid => self.sys_setreid(tid, a[0], a[1], false),
            Sysno::setregid => self.sys_setreid(tid, a[0], a[1], true),
            Sysno::setresuid => self.sys_setresid(tid, a[0], a[1], a[2], false),
            Sysno::setresgid => self.sys_setresid(tid, a[0], a[1], a[2], true),
            Sysno::setfsuid => self.sys_setfsid(tid, a[0], false),
            Sysno::setfsgid => self.sys_setfsid(tid, a[0], true),
            Sysno::getgroups => self.sys_getgroups(tid, a[0], a[1]),
            Sysno::setgroups => self.sys_setgroups(tid, a[0], a[1]),
            Sysno::getpgid => self.sys_getpgid(tid, a[0]),
            Sysno::setpgid => self.sys_setpgid(tid, a[0], a[1]),
            Sysno::getsid => Ok(u64::from(self.procs[&self.pid_of(tid)].sid)),
            Sysno::setsid => self.sys_setsid(tid),
            Sysno::prctl => self.sys_prctl(tid, a[0], a[1]),
            Sysno::prlimit64 => self.sys_prlimit64(tid, a[0], a[1], a[2], a[3]),
            Sysno::getrlimit => self.sys_prlimit64(tid, 0, a[0], 0, a[1]),
            Sysno::setrlimit => Ok(0),
            Sysno::getrusage => self.sys_getrusage(tid, a[1]),
            Sysno::sched_yield => Ok(0),
            Sysno::sched_getaffinity => self.sys_sched_getaffinity(tid, a[1], a[2]),
            Sysno::sched_setaffinity => Ok(0),
            // One scheduling class (SCHED_OTHER, priority 0): green threads are scheduled by
            // retired-instruction quanta, so policies and priorities are accepted and ignored.
            Sysno::capget => self.sys_capget(tid, a[0], a[1]),
            Sysno::capset => Ok(0),
            Sysno::sched_getscheduler => Ok(0),
            Sysno::sched_getparam => {
                crate::uaccess::write_u32(self.mm(tid).borrow_mut().mem.as_mut(), a[1], 0)
                    .map(|()| 0)
            }
            Sysno::sched_setscheduler | Sysno::sched_setparam => Ok(0),
            Sysno::sched_get_priority_max | Sysno::sched_get_priority_min => match a[0] {
                0 | 3 | 5 => Ok(0), // SCHED_OTHER, SCHED_BATCH, SCHED_IDLE
                1 | 2 => Ok(if sys == Sysno::sched_get_priority_max {
                    99
                } else {
                    1
                }),
                _ => Err(Errno::EINVAL),
            },
            Sysno::getcpu => self.sys_getcpu(tid, a[0], a[1]),
            Sysno::rt_sigaction => self.sys_rt_sigaction(tid, a[0], a[1], a[2], a[3]),
            Sysno::rt_sigprocmask => self.sys_rt_sigprocmask(tid, a[0], a[1], a[2], a[3]),
            Sysno::sigaltstack => self.sys_sigaltstack(tid, a[0], a[1]),
            Sysno::kill => self.sys_kill(tid, a[0], a[1]),
            Sysno::tkill => self.sys_tgkill(tid, -1, a[0], a[1]),
            Sysno::tgkill => self.sys_tgkill(tid, a[0] as i64, a[1], a[2]),

            // ---- time, randomness, system info ----
            Sysno::clock_gettime => self.sys_clock_gettime(tid, a[0], a[1], host),
            Sysno::clock_getres => self.sys_clock_getres(tid, a[0], a[1]),
            Sysno::gettimeofday => self.sys_gettimeofday(tid, a[0], host),
            Sysno::times => self.sys_times(tid, a[0]),
            Sysno::getrandom => self.sys_getrandom(tid, a[0], a[1], a[2], host),
            Sysno::uname => self.sys_uname(tid, a[0]),
            Sysno::sysinfo => self.sys_sysinfo(tid, a[0], host),
            Sysno::getpriority => Ok(20),
            Sysno::setpriority => Ok(0),

            _ => Err(Errno::ENOSYS),
        };
        self.finish(r, &frame)
    }
}

/// How many arguments to show in a trace line (the substrate writes the line:
/// `[pid] name(args) = result`, with `[pid:tid]` for threads other than a process's first).
pub(crate) fn nargs(s: Sysno) -> usize {
    use Sysno::*;
    match s {
        getpid | gettid | getppid | getuid | geteuid | getgid | getegid | sched_yield | setsid
        | sync => 0,
        exit | exit_group | close | dup | brk | set_tid_address | umask | chdir | fchdir
        | uname | sysinfo | times => 1,
        munmap | fstat | getcwd | statfs | fstatfs | clock_gettime | clock_getres
        | gettimeofday | nanosleep | sigaltstack | kill | tkill | ftruncate | truncate | fchmod
        | set_robust_list | getrusage => 2,
        read | write | readv | writev | lseek | ioctl | fcntl | getdents64 | mprotect | madvise
        | getrandom | dup3 | execve | mkdirat | unlinkat | symlinkat | tgkill | faccessat
        | fchown | ppoll => 3,
        openat | newfstatat | readlinkat | rt_sigaction | rt_sigprocmask | prlimit64 | pread64
        | pwrite64 | renameat | fchmodat | utimensat | clock_nanosleep | sendfile => 4,
        statx | mremap | renameat2 | linkat | fchownat => 5,
        _ => 6,
    }
}
