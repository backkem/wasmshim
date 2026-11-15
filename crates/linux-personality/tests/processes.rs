//! Processes, threads and signals: `fork` (as `clone(SIGCHLD)`), `wait4` and its status words,
//! `execve` (close-on-exec, a fresh address space, its errors), threads and `exit` versus
//! `exit_group`, and signal handlers interrupting or restarting a blocked call.

mod support;

use support::*;
use wasmshim_linux::ExitStatus;
use wasmshim_linux_abi::consts::open::*;
use wasmshim_linux_abi::consts::signal::*;

const WNOHANG: u64 = 1;

/// `kill(pid, sig)`.
fn kill(g: &mut Guest, pid: i64, sig: u32) -> i64 {
    g.syscall(Sysno::kill, &[pid as u64, u64::from(sig)])
}

/// `rt_sigprocmask(how, &set, NULL, 8)`.
fn sigprocmask(g: &mut Guest, how: u64, set: u64) -> i64 {
    let p = g.put(&set.to_le_bytes());
    g.syscall(Sysno::rt_sigprocmask, &[how, p, 0, 8])
}

#[test]
fn fork_and_wait_for_the_exit_status() {
    run(|g| {
        let child = g.fork(|c| {
            assert_eq!(c.syscall(Sysno::getppid, &[]), 1);
            c.exit(3);
        });
        assert_eq!(child, 2);
        // wait(2): the status word holds the exit code in bits 8..16.
        let (r, status) = g.wait4(child, 0);
        assert_eq!((r, status), (2, 3 << 8));
        // No children left: ECHILD.
        assert_eq!(g.wait4(-1, 0).0, -ECHILD);
    });
}

#[test]
fn the_child_sees_its_own_pid() {
    run(|g| {
        let (_, rd, wr) = g.pipe2(0);
        let child = g.fork(move |c| {
            let pid = c.syscall(Sysno::getpid, &[]);
            c.write_all(wr, &pid.to_le_bytes());
        });
        let got = g.read_vec(rd, 8);
        assert_eq!(i64::from_le_bytes(got.try_into().unwrap()), child);
        g.wait4(child, 0);
    });
}

#[test]
fn wnohang_and_a_child_killed_by_a_signal() {
    run(|g| {
        let (_, rd, _wr) = g.pipe2(0);
        let child = g.fork(move |c| {
            // Blocks forever: nothing writes.
            c.read_vec(rd, 1);
        });
        // Let the child run and block.
        g.nanosleep(1_000);
        assert_eq!(g.wait4(child, WNOHANG).0, 0, "still running");
        assert_eq!(kill(g, child, SIGKILL), 0);
        let (r, status) = g.wait4(child, 0);
        // The low 7 bits hold the terminating signal (wait(2)).
        assert_eq!((r, status), (child, SIGKILL));
        // kill(2): ESRCH once it is reaped.
        assert_eq!(kill(g, child, SIGKILL), -ESRCH);
    });
}

#[test]
fn kill_with_an_invalid_signal_is_einval() {
    // kill(2): EINVAL "An invalid signal was specified."
    run(|g| {
        assert_eq!(kill(g, 1, 99), -EINVAL);
        assert_eq!(kill(g, 1, 65), -EINVAL);
    });
}

#[test]
fn a_zombie_can_still_be_signalled() {
    // A child that exited stays a process (a zombie) until it is waited for; kill(2) with
    // signal 0 checks that the pid exists, so it succeeds (as does any signal, which is
    // discarded).
    run(|g| {
        let child = g.fork(|_| {});
        g.nanosleep(1_000);
        assert_eq!(kill(g, child, 0), 0);
        assert_eq!(g.wait4(child, 0).0, child);
        assert_eq!(kill(g, child, 0), -ESRCH);
    });
}

#[test]
fn execve_replaces_the_image_and_closes_cloexec_descriptors() {
    let out = Machine::default().run(|g| {
        let kept = g.open("/etc/hostname", O_RDONLY);
        let closed = g.open("/etc/hostname", O_RDONLY | O_CLOEXEC);
        let old = g.put(b"old image");
        let r = g.execve(PROGRAM, &["prog", "again"], move |n| {
            assert_eq!(n.syscall(Sysno::getpid, &[]), 1, "the same process");
            assert_eq!(n.read_vec(kept, 64), b"testhost\n");
            let buf = n.alloc(8);
            assert_eq!(n.read(closed, buf, 8), -EBADF);
            assert!(
                n.try_bytes(old, 1).is_none_or(|b| b != b"o"),
                "the old memory is gone"
            );
            n.exit(9);
        });
        unreachable!("execve returned {r}");
    });
    assert_eq!(out.status, ExitStatus::Code(9));
}

#[test]
fn execve_errors_leave_the_caller_running() {
    run(|g| {
        let never = |_: &mut Guest| unreachable!("the exec failed");
        // execve(2): ENOENT, EACCES (not executable), ENOEXEC (an executable that is not a
        // known format).
        assert_eq!(g.execve("/bin/missing", &["x"], never), -ENOENT);
        assert_eq!(g.execve("/etc/hostname", &["x"], never), -EACCES);
        let fd = g.openat(AT_FDCWD, "/tmp/junk", O_CREAT | O_WRONLY, 0o755);
        g.write_all(fd, b"not an executable at all");
        g.close(fd);
        assert_eq!(g.execve("/tmp/junk", &["x"], never), -ENOEXEC);
        let p = g.cstr(PROGRAM);
        assert_eq!(g.syscall(Sysno::execve, &[p, 8, 0]), -EFAULT);
        assert_eq!(g.syscall(Sysno::getpid, &[]), 1);
    });
}

#[test]
fn execve_of_a_directory_is_eacces() {
    // execve(2): EACCES "The file or a script interpreter is not a regular file" (may_open
    // refuses MAY_EXEC on a directory with EACCES).
    run(|g| {
        assert_eq!(g.execve("/bin", &["x"], |_| unreachable!()), -EACCES);
    });
}

#[test]
fn a_thread_exit_leaves_the_process_running() {
    run(|g| {
        let (_, rd, wr) = g.pipe2(0);
        let tid = g.spawn_thread(move |t| {
            assert_eq!(t.syscall(Sysno::getpid, &[]), 1, "a thread of pid 1");
            let me = t.syscall(Sysno::gettid, &[]);
            t.write_all(wr, &me.to_le_bytes());
            // Returning ends only this thread (exit, not exit_group).
        });
        let got = g.read_vec(rd, 8);
        assert_eq!(i64::from_le_bytes(got.try_into().unwrap()), tid);
        g.nanosleep(1_000);
        assert_eq!(g.syscall(Sysno::getpid, &[]), 1);
    });
}

#[test]
fn exit_group_from_a_thread_ends_the_process() {
    let out = Machine::default().run(|g| {
        let (_, rd, _wr) = g.pipe2(0);
        g.spawn_thread(|t| {
            t.syscall(Sysno::exit_group, &[5]);
            unreachable!();
        });
        // Blocks until the whole process ends.
        g.read_vec(rd, 1);
        unreachable!("the process ended");
    });
    assert_eq!(out.status, ExitStatus::Code(5));
}

#[test]
fn a_signal_to_oneself_is_handled_before_kill_returns() {
    run(|g| {
        assert_eq!(g.handle(SIGUSR1, 0), 0);
        assert_eq!(kill(g, 1, SIGUSR1), 0);
        assert_eq!(g.signals(), [SIGUSR1]);
        // A blocked signal waits until it is unblocked (sigprocmask(2)).
        assert_eq!(sigprocmask(g, SIG_BLOCK, 1 << (SIGUSR1 - 1)), 0);
        assert_eq!(kill(g, 1, SIGUSR1), 0);
        assert_eq!(g.signals(), []);
        assert_eq!(sigprocmask(g, SIG_UNBLOCK, 1 << (SIGUSR1 - 1)), 0);
        assert_eq!(g.signals(), [SIGUSR1]);
        // Ignored signals are discarded.
        let act = g.put(
            &[SIG_IGN, 0, 0, 0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        assert_eq!(
            g.syscall(Sysno::rt_sigaction, &[u64::from(SIGUSR2), act, 0, 8]),
            0
        );
        assert_eq!(kill(g, 1, SIGUSR2), 0);
        assert_eq!(g.signals(), []);
        // rt_sigaction(2): SIGKILL's action cannot change.
        assert_eq!(g.handle(SIGKILL, 0), -EINVAL);
    });
}

#[test]
fn a_handled_signal_interrupts_a_blocking_read() {
    // signal(7): without SA_RESTART, a read on a pipe interrupted by a handler fails with EINTR.
    run(|g| {
        assert_eq!(g.handle(SIGUSR1, 0), 0);
        let (_, rd, _wr) = g.pipe2(0);
        let child = g.fork(|c| {
            c.nanosleep(1_000_000);
            assert_eq!(c.syscall(Sysno::kill, &[1, u64::from(SIGUSR1)]), 0);
        });
        let buf = g.alloc(8);
        assert_eq!(g.read(rd, buf, 8), -EINTR);
        assert_eq!(g.signals(), [SIGUSR1]);
        g.wait4(child, 0);
    });
}

#[test]
fn sa_restart_restarts_a_blocking_read() {
    // signal(7): with SA_RESTART, read(2) on a pipe is restarted after the handler.
    run(|g| {
        assert_eq!(g.handle(SIGUSR1, SA_RESTART), 0);
        let (_, rd, wr) = g.pipe2(0);
        let child = g.fork(move |c| {
            c.nanosleep(1_000_000);
            c.syscall(Sysno::kill, &[1, u64::from(SIGUSR1)]);
            c.nanosleep(1_000_000);
            c.write_all(wr, b"late");
        });
        assert_eq!(g.read_vec(rd, 8), b"late");
        assert_eq!(g.signals(), [SIGUSR1]);
        g.wait4(child, 0);
    });
}

#[test]
fn a_handled_signal_cuts_a_sleep_short() {
    run(|g| {
        assert_eq!(g.handle(SIGUSR1, SA_RESTART), 0);
        let child = g.fork(|c| {
            c.nanosleep(2_000_000);
            c.syscall(Sysno::kill, &[1, u64::from(SIGUSR1)]);
        });
        // nanosleep(2): EINTR, with the time left in *rem; never restarted (signal(7)).
        let req = g.put(&timespec(10_000_000));
        let rem = g.alloc(16);
        assert_eq!(g.syscall(Sysno::nanosleep, &[req, rem]), -EINTR);
        assert_eq!(g.signals(), [SIGUSR1]);
        let left = g.u64_at(rem) * 1_000_000_000 + g.u64_at(rem + 8);
        assert_eq!(left, 8_000_000);
        g.wait4(child, 0);
    });
}
