//! Pipes and polling: `pipe2`, end of file and `EPIPE`, non-blocking reads, a blocking read
//! woken by another thread's or process's write, `ppoll`, `epoll_create1`/`epoll_ctl`/
//! `epoll_pwait` (level- and edge-triggered, timeouts, errors) and `eventfd2`.

mod support;

use support::*;
use wasmshim_linux_abi::consts::clock::CLOCK_MONOTONIC;
use wasmshim_linux_abi::consts::open::*;
use wasmshim_linux_abi::consts::signal::SIGPIPE;

const POLLIN: u16 = 0x1;
const POLLOUT: u16 = 0x4;
const POLLERR: u16 = 0x8;
const POLLHUP: u16 = 0x10;
const POLLNVAL: u16 = 0x20;

const EPOLLIN: u32 = 0x1;
const EPOLLOUT: u32 = 0x4;
const EPOLLET: u32 = 1 << 31;
const EPOLL_CTL_ADD: u64 = 1;
const EPOLL_CTL_DEL: u64 = 2;
const EPOLL_CTL_MOD: u64 = 3;

const EFD_SEMAPHORE: u64 = 1;

/// `ppoll` over `(fd, events)` with a timeout in ns (`None`: forever): the result and each
/// entry's `revents`.
fn ppoll(g: &mut Guest, fds: &[(i64, u16)], timeout: Option<u64>) -> (i64, Vec<u16>) {
    let raw: Vec<u8> = fds
        .iter()
        .flat_map(|&(fd, ev)| {
            let mut e = [0u8; 8];
            e[..4].copy_from_slice(&(fd as i32).to_le_bytes());
            e[4..6].copy_from_slice(&ev.to_le_bytes());
            e
        })
        .collect();
    let p = g.put(&raw);
    let tmo = match timeout {
        Some(ns) => g.put(&timespec(ns)),
        None => 0,
    };
    let r = g.syscall(Sysno::ppoll, &[p, fds.len() as u64, tmo, 0, 8]);
    let back = g.bytes(p, raw.len());
    let revents = back
        .chunks(8)
        .map(|e| u16::from_le_bytes([e[6], e[7]]))
        .collect();
    (r, revents)
}

/// `epoll_ctl(epfd, op, fd, {events, data})`.
fn epoll_ctl(g: &mut Guest, epfd: i64, op: u64, fd: i64, events: u32, data: u64) -> i64 {
    let mut ev = [0u8; 16];
    ev[..4].copy_from_slice(&events.to_le_bytes());
    ev[8..].copy_from_slice(&data.to_le_bytes());
    let p = g.put(&ev);
    g.syscall(Sysno::epoll_ctl, &[epfd as u64, op, fd as u64, p])
}

/// `epoll_pwait(epfd, events, 8, timeout_ms, NULL)`: the result and `(events, data)` pairs.
fn epoll_wait(g: &mut Guest, epfd: i64, timeout_ms: i32) -> (i64, Vec<(u32, u64)>) {
    let buf = g.alloc(8 * 16);
    let r = g.syscall(
        Sysno::epoll_pwait,
        &[epfd as u64, buf, 8, timeout_ms as i64 as u64, 0, 8],
    );
    let raw = g.bytes(buf, 8 * 16usize);
    let got = (0..r.max(0) as usize)
        .map(|k| {
            let e = &raw[k * 16..k * 16 + 16];
            (
                u32::from_le_bytes(e[..4].try_into().unwrap()),
                u64::from_le_bytes(e[8..].try_into().unwrap()),
            )
        })
        .collect();
    (r, got)
}

#[test]
fn a_pipe_carries_bytes_in_order() {
    run(|g| {
        let (r, rd, wr) = g.pipe2(0);
        assert_eq!(r, 0);
        assert_eq!((rd, wr), (3, 4));
        assert_eq!(g.write_all(wr, b"hello "), 6);
        assert_eq!(g.write_all(wr, b"world"), 5);
        // A read takes what is there, up to its size.
        assert_eq!(g.read_vec(rd, 4), b"hell");
        assert_eq!(g.read_vec(rd, 64), b"o world");
        // The ends are one-way.
        let buf = g.alloc(4);
        assert_eq!(g.write(rd, buf, 1), -EBADF);
        assert_eq!(g.read(wr, buf, 1), -EBADF);
        // Pipes cannot seek (lseek(2), pread(2)).
        assert_eq!(g.lseek(rd, 0, 0), -ESPIPE);
        assert_eq!(g.syscall(Sysno::pread64, &[rd as u64, buf, 1, 0]), -ESPIPE);
    });
}

#[test]
fn pipe2_flags_and_errors() {
    run(|g| {
        let (r, rd, wr) = g.pipe2(O_CLOEXEC | O_NONBLOCK);
        assert_eq!(r, 0);
        assert_eq!(g.syscall(Sysno::fcntl, &[rd as u64, 1 /* F_GETFD */]), 1);
        let fl = g.syscall(Sysno::fcntl, &[wr as u64, 3 /* F_GETFL */]);
        assert_eq!(fl as u64 & O_NONBLOCK, O_NONBLOCK);
        // An empty non-blocking pipe: EAGAIN (pipe(7)).
        let buf = g.alloc(8);
        assert_eq!(g.read(rd, buf, 8), -EAGAIN);
        // pipe(2): EINVAL for unknown flags, EFAULT for a bad array.
        assert_eq!(g.syscall(Sysno::pipe2, &[buf, 0x1]), -EINVAL);
        assert_eq!(g.syscall(Sysno::pipe2, &[8, 0]), -EFAULT);
    });
}

#[test]
fn end_of_file_and_broken_pipes() {
    run(|g| {
        let (_, rd, wr) = g.pipe2(0);
        g.write_all(wr, b"last");
        assert_eq!(g.close(wr), 0);
        // Data written before the close is still read; then end of file (pipe(7)).
        assert_eq!(g.read_vec(rd, 64), b"last");
        assert_eq!(g.read_vec(rd, 64), b"");

        // Writing with no reader: SIGPIPE, and EPIPE once the signal is handled (pipe(7)).
        assert_eq!(g.handle(SIGPIPE, 0), 0);
        let (_, rd, wr) = g.pipe2(0);
        assert_eq!(g.close(rd), 0);
        assert_eq!(g.write_all(wr, b"x"), -EPIPE);
        assert_eq!(g.signals(), [SIGPIPE]);
    });
}

#[test]
fn sigpipe_kills_a_writer_by_default() {
    let out = Machine::default().run(|g| {
        let (_, rd, wr) = g.pipe2(0);
        g.close(rd);
        g.write_all(wr, b"x");
        unreachable!("the write is fatal");
    });
    assert_eq!(out.status, wasmshim_linux::ExitStatus::Signal(SIGPIPE));
}

#[test]
fn a_blocking_read_waits_for_another_thread() {
    let out = run(|g| {
        let (_, rd, wr) = g.pipe2(0);
        let start = g.clock_ns(CLOCK_MONOTONIC);
        let tid = g.spawn_thread(move |t| {
            assert_eq!(t.nanosleep(1_000_000), 0);
            assert_eq!(t.write_all(wr, b"ping"), 4);
        });
        assert!(tid > 1, "{tid}");
        // Blocks until the thread writes, 1 ms later.
        assert_eq!(g.read_vec(rd, 64), b"ping");
        assert_eq!(g.clock_ns(CLOCK_MONOTONIC), start + 1_000_000);
    });
    assert_eq!(out.now, START_NS + 1_000_000);
}

#[test]
fn a_blocking_read_waits_for_a_child_process() {
    run(|g| {
        let (_, rd, wr) = g.pipe2(0);
        let child = g.fork(move |c| {
            c.close(rd);
            c.nanosleep(2_000_000);
            c.write_all(wr, b"from the child");
        });
        assert!(child > 1, "{child}");
        g.close(wr);
        assert_eq!(g.read_vec(rd, 64), b"from the child");
        // The child exited, closing the last write end: end of file.
        assert_eq!(g.read_vec(rd, 64), b"");
        let (r, status) = g.wait4(child, 0);
        assert_eq!((r, status), (child, 0));
    });
}

#[test]
fn ppoll_reports_readiness() {
    run(|g| {
        let (_, rd, wr) = g.pipe2(0);
        // Empty: only the write end is ready; a zero timeout returns at once.
        let (r, rev) = ppoll(g, &[(rd, POLLIN), (wr, POLLOUT)], Some(0));
        assert_eq!((r, rev), (1, vec![0, POLLOUT]));
        g.write_all(wr, b"x");
        let (r, rev) = ppoll(g, &[(rd, POLLIN), (wr, POLLOUT)], Some(0));
        assert_eq!((r, rev), (2, vec![POLLIN, POLLOUT]));
        // poll(2): a negative fd is skipped (revents 0); a closed one is POLLNVAL, and counts.
        let (r, rev) = ppoll(g, &[(-1, POLLIN), (99, POLLIN)], Some(0));
        assert_eq!((r, rev), (1, vec![0, POLLNVAL]));
        // The writer gone: POLLHUP (with POLLIN while data is left).
        g.close(wr);
        let (r, rev) = ppoll(g, &[(rd, POLLIN)], Some(0));
        assert_eq!((r, rev), (1, vec![POLLIN | POLLHUP]));
        g.read_vec(rd, 8);
        let (_, rev) = ppoll(g, &[(rd, POLLIN)], Some(0));
        assert_eq!(rev, [POLLHUP]);
        // The reader gone: POLLERR on the write end.
        let (_, rd2, wr2) = g.pipe2(0);
        g.close(rd2);
        let (_, rev) = ppoll(g, &[(wr2, POLLOUT)], Some(0));
        assert_eq!(rev[0] & POLLERR, POLLERR);
    });
}

#[test]
fn ppoll_times_out_after_exactly_its_timeout() {
    let out = run(|g| {
        let (_, rd, _wr) = g.pipe2(0);
        let start = g.clock_ns(CLOCK_MONOTONIC);
        let (r, rev) = ppoll(g, &[(rd, POLLIN)], Some(5_000_000));
        assert_eq!((r, rev), (0, vec![0]));
        assert_eq!(g.clock_ns(CLOCK_MONOTONIC), start + 5_000_000);
        // A bad timeout is EINVAL; a bad array EFAULT (ppoll(2)).
        let bad = g.put(&{
            let mut b = [0u8; 16];
            b[8..].copy_from_slice(&2_000_000_000u64.to_le_bytes());
            b
        });
        let fds = g.put(&[0u8; 8]);
        assert_eq!(g.syscall(Sysno::ppoll, &[fds, 1, bad, 0, 8]), -EINVAL);
        assert_eq!(g.syscall(Sysno::ppoll, &[8, 1, 0, 0, 8]), -EFAULT);
    });
    assert_eq!(out.now, START_NS + 5_000_000);
}

#[test]
fn ppoll_wakes_when_a_thread_writes() {
    run(|g| {
        let (_, rd, wr) = g.pipe2(0);
        g.spawn_thread(move |t| {
            t.nanosleep(3_000_000);
            t.write_all(wr, b"!");
        });
        let start = g.clock_ns(CLOCK_MONOTONIC);
        let (r, rev) = ppoll(g, &[(rd, POLLIN)], None);
        assert_eq!((r, rev), (1, vec![POLLIN]));
        assert_eq!(g.clock_ns(CLOCK_MONOTONIC), start + 3_000_000);
    });
}

#[test]
fn epoll_level_triggered() {
    run(|g| {
        let ep = g.syscall(Sysno::epoll_create1, &[0]);
        assert!(ep >= 0, "{ep}");
        let (_, rd, wr) = g.pipe2(0);
        assert_eq!(epoll_ctl(g, ep, EPOLL_CTL_ADD, rd, EPOLLIN, 42), 0);
        assert_eq!(epoll_wait(g, ep, 0), (0, vec![]));
        g.write_all(wr, b"abc");
        assert_eq!(epoll_wait(g, ep, 0), (1, vec![(EPOLLIN, 42)]));
        // Level-triggered: still ready until drained.
        assert_eq!(epoll_wait(g, ep, 0), (1, vec![(EPOLLIN, 42)]));
        g.read_vec(rd, 8);
        assert_eq!(epoll_wait(g, ep, 0), (0, vec![]));
        // MOD changes the interest and data; DEL removes it.
        assert_eq!(epoll_ctl(g, ep, EPOLL_CTL_ADD, wr, EPOLLOUT, 7), 0);
        assert_eq!(epoll_wait(g, ep, 0), (1, vec![(EPOLLOUT, 7)]));
        assert_eq!(epoll_ctl(g, ep, EPOLL_CTL_MOD, wr, EPOLLOUT, 8), 0);
        assert_eq!(epoll_wait(g, ep, 0), (1, vec![(EPOLLOUT, 8)]));
        assert_eq!(epoll_ctl(g, ep, EPOLL_CTL_DEL, wr, 0, 0), 0);
        assert_eq!(epoll_wait(g, ep, 0), (0, vec![]));
    });
}

#[test]
fn epoll_edge_triggered_reports_each_change_once() {
    run(|g| {
        let ep = g.syscall(Sysno::epoll_create1, &[0]);
        let (_, rd, wr) = g.pipe2(0);
        assert_eq!(epoll_ctl(g, ep, EPOLL_CTL_ADD, rd, EPOLLIN | EPOLLET, 1), 0);
        g.write_all(wr, b"a");
        assert_eq!(epoll_wait(g, ep, 0), (1, vec![(EPOLLIN, 1)]));
        // Nothing new arrived: no event, although the pipe is still readable (epoll(7)).
        assert_eq!(epoll_wait(g, ep, 0), (0, vec![]));
        g.write_all(wr, b"b");
        assert_eq!(epoll_wait(g, ep, 0), (1, vec![(EPOLLIN, 1)]));
    });
}

#[test]
fn epoll_ctl_errors() {
    run(|g| {
        let ep = g.syscall(Sysno::epoll_create1, &[0]);
        let (_, rd, _wr) = g.pipe2(0);
        assert_eq!(epoll_ctl(g, ep, EPOLL_CTL_ADD, rd, EPOLLIN, 0), 0);
        // epoll_ctl(2):
        assert_eq!(epoll_ctl(g, ep, EPOLL_CTL_ADD, rd, EPOLLIN, 0), -EEXIST);
        let (_, other, _) = g.pipe2(0);
        assert_eq!(epoll_ctl(g, ep, EPOLL_CTL_MOD, other, EPOLLIN, 0), -ENOENT);
        assert_eq!(epoll_ctl(g, ep, EPOLL_CTL_DEL, other, 0, 0), -ENOENT);
        assert_eq!(
            epoll_ctl(g, ep, 99, rd, EPOLLIN, 0),
            -EINVAL,
            "an unknown op"
        );
        assert_eq!(
            epoll_ctl(g, ep, EPOLL_CTL_ADD, ep, EPOLLIN, 0),
            -EINVAL,
            "itself"
        );
        assert_eq!(
            epoll_ctl(g, rd, EPOLL_CTL_ADD, other, EPOLLIN, 0),
            -EINVAL,
            "not an epoll"
        );
        assert_eq!(epoll_ctl(g, ep, EPOLL_CTL_ADD, 99, EPOLLIN, 0), -EBADF);
        assert_eq!(epoll_ctl(g, 99, EPOLL_CTL_ADD, rd, EPOLLIN, 0), -EBADF);
        // Regular files and directories do not support epoll: EPERM.
        let file = g.open("/etc/hostname", O_RDONLY);
        assert_eq!(epoll_ctl(g, ep, EPOLL_CTL_ADD, file, EPOLLIN, 0), -EPERM);
        assert_eq!(
            g.syscall(
                Sysno::epoll_ctl,
                &[ep as u64, EPOLL_CTL_ADD, other as u64, 8]
            ),
            -EFAULT
        );
        // epoll_create1(2): EINVAL for unknown flags.
        assert_eq!(g.syscall(Sysno::epoll_create1, &[0x1]), -EINVAL);
        // epoll_wait(2): EINVAL when maxevents is not positive.
        let buf = g.alloc(16);
        assert_eq!(
            g.syscall(Sysno::epoll_pwait, &[ep as u64, buf, 0, 0, 0, 8]),
            -EINVAL
        );
    });
}

#[test]
fn epoll_pwait_times_out_and_wakes() {
    let out = run(|g| {
        let ep = g.syscall(Sysno::epoll_create1, &[0]);
        let (_, rd, wr) = g.pipe2(0);
        epoll_ctl(g, ep, EPOLL_CTL_ADD, rd, EPOLLIN, 5);
        let start = g.clock_ns(CLOCK_MONOTONIC);
        assert_eq!(epoll_wait(g, ep, 10), (0, vec![]));
        assert_eq!(g.clock_ns(CLOCK_MONOTONIC), start + 10_000_000);
        g.spawn_thread(move |t| {
            t.nanosleep(4_000_000);
            t.write_all(wr, b"x");
        });
        assert_eq!(epoll_wait(g, ep, -1), (1, vec![(EPOLLIN, 5)]));
        assert_eq!(g.clock_ns(CLOCK_MONOTONIC), start + 14_000_000);
    });
    assert_eq!(out.now, START_NS + 14_000_000);
}

#[test]
fn closing_the_last_descriptor_leaves_the_epoll_set() {
    // epoll(7): a file description is removed from every set once all of its descriptors are
    // closed.
    run(|g| {
        let ep = g.syscall(Sysno::epoll_create1, &[0]);
        let (_, rd, wr) = g.pipe2(0);
        epoll_ctl(g, ep, EPOLL_CTL_ADD, rd, EPOLLIN, 1);
        g.write_all(wr, b"x");
        g.close(rd);
        assert_eq!(epoll_wait(g, ep, 0), (0, vec![]));
    });
}

#[test]
fn eventfd_counts() {
    run(|g| {
        let efd = g.syscall(Sysno::eventfd2, &[3, O_NONBLOCK]);
        assert!(efd >= 0, "{efd}");
        let buf = g.alloc(8);
        // eventfd(2): a read returns the counter and resets it.
        assert_eq!(g.read(efd, buf, 8), 8);
        assert_eq!(g.u64_at(buf), 3);
        assert_eq!(g.read(efd, buf, 8), -EAGAIN);
        // Writes add to it.
        let five = g.put(&5u64.to_le_bytes());
        let two = g.put(&2u64.to_le_bytes());
        assert_eq!(g.write(efd, five, 8), 8);
        assert_eq!(g.write(efd, two, 8), 8);
        assert_eq!(g.read(efd, buf, 8), 8);
        assert_eq!(g.u64_at(buf), 7);
        // A buffer under 8 bytes, or a write of 2^64-1: EINVAL.
        assert_eq!(g.read(efd, buf, 4), -EINVAL);
        assert_eq!(g.write(efd, five, 4), -EINVAL);
        let max = g.put(&u64::MAX.to_le_bytes());
        assert_eq!(g.write(efd, max, 8), -EINVAL);
        // Readable while non-zero; always writable below the maximum.
        let (_, rev) = ppoll(g, &[(efd, POLLIN | POLLOUT)], Some(0));
        assert_eq!(rev, [POLLOUT]);
        g.write(efd, two, 8);
        let (_, rev) = ppoll(g, &[(efd, POLLIN | POLLOUT)], Some(0));
        assert_eq!(rev, [POLLIN | POLLOUT]);
        assert_eq!(
            g.syscall(Sysno::eventfd2, &[0, 0x10]),
            -EINVAL,
            "unknown flags"
        );
    });
}

#[test]
fn eventfd_semaphore_mode_reads_one_at_a_time() {
    run(|g| {
        let efd = g.syscall(Sysno::eventfd2, &[2, EFD_SEMAPHORE | O_NONBLOCK]);
        let buf = g.alloc(8);
        for _ in 0..2 {
            assert_eq!(g.read(efd, buf, 8), 8);
            assert_eq!(g.u64_at(buf), 1);
        }
        assert_eq!(g.read(efd, buf, 8), -EAGAIN);
    });
}

#[test]
fn a_blocking_eventfd_read_waits_for_a_write() {
    run(|g| {
        let efd = g.syscall(Sysno::eventfd2, &[0, 0]);
        g.spawn_thread(move |t| {
            t.nanosleep(1_000);
            let v = t.put(&9u64.to_le_bytes());
            assert_eq!(t.write(efd, v, 8), 8);
        });
        let buf = g.alloc(8);
        assert_eq!(g.read(efd, buf, 8), 8);
        assert_eq!(g.u64_at(buf), 9);
    });
}
