//! Regression tests for defects found by reading the code: guest-reachable panics of the
//! personality (a directory removed while in use, two epoll instances watching each other, a
//! huge `mq_timedsend` length) and smaller fidelity gaps (`sendfile` into a short write, signal
//! masks and `SA_RESTART` around `ppoll`, `F_GETPIPE_SZ`, FIFOs, the alternate signal stack's
//! minimum size, a directory listed while it changes, `kill(-1)`, timeouts too large for
//! nanoseconds, option lengths and ranges). Each test states what Linux does.

mod support;

use support::*;
use wasmshim_linux_abi::consts::at::AT_REMOVEDIR;
use wasmshim_linux_abi::consts::mode::S_IFIFO;
use wasmshim_linux_abi::consts::open::*;
use wasmshim_linux_abi::consts::signal::*;

const EPOLLIN: u32 = 0x1;
const EPOLL_CTL_ADD: u64 = 1;
const POLLIN: u16 = 0x1;

// ---- A directory removed while it is in use ----

#[test]
fn rmdir_of_the_cwd_then_getcwd_is_enoent() {
    // rmdir(2) removes a directory that is some process's cwd (EBUSY is only for mount points
    // and the root); getcwd(2) then fails with ENOENT: "The current working directory has been
    // unlinked."
    run(|g| {
        assert_eq!(g.mkdir("/tmp/d", 0o755), 0);
        assert_eq!(g.chdir("/tmp/d"), 0);
        assert_eq!(g.unlinkat(AT_FDCWD, "../d", AT_REMOVEDIR), 0);
        assert_eq!(g.getcwd().0, -ENOENT);
        // Nothing can be created in it.
        assert_eq!(g.open("new", O_CREAT | O_WRONLY), -ENOENT);
        // The process can leave it.
        assert_eq!(g.chdir("/"), 0);
        assert_eq!(g.getcwd().1, b"/");
    });
}

#[test]
fn a_removed_directory_still_open_stats_and_lists_as_deleted() {
    // A descriptor keeps a removed directory's inode alive: fstat works and shows no links;
    // getdents64 on a dead directory is ENOENT (fs/readdir.c checks IS_DEADDIR).
    run(|g| {
        assert_eq!(g.mkdir("/tmp/e", 0o755), 0);
        let fd = g.open("/tmp/e", O_RDONLY | O_DIRECTORY);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/e", AT_REMOVEDIR), 0);
        let (r, st) = g.fstat(fd);
        assert_eq!(r, 0);
        assert_eq!(st.nlink, 0);
        let buf = g.alloc(256);
        assert_eq!(
            g.syscall(Sysno::getdents64, &[fd as u64, buf, 256]),
            -ENOENT
        );
        assert_eq!(g.openat(fd as i32, "x", O_CREAT | O_WRONLY, 0o644), -ENOENT);
        assert_eq!(g.close(fd), 0);
    });
}

#[test]
fn renaming_over_the_cwd_leaves_it_deleted() {
    // rename(2) may replace an empty directory; a process whose cwd it was is left in a deleted
    // directory, as after rmdir.
    run(|g| {
        assert_eq!(g.mkdir("/tmp/a", 0o755), 0);
        assert_eq!(g.mkdir("/tmp/b", 0o755), 0);
        assert_eq!(g.chdir("/tmp/b"), 0);
        assert_eq!(g.rename("/tmp/a", "/tmp/b"), 0);
        assert_eq!(g.getcwd().0, -ENOENT);
        assert_eq!(
            g.chdir("/tmp/b"),
            0,
            "the name now holds the moved directory"
        );
        assert_eq!(g.getcwd().1, b"/tmp/b");
    });
}

#[test]
fn removed_directories_stay_walkable_while_in_use() {
    // A removed cwd's `..` still leads to its old parent, even once that is removed too (the
    // kernel keeps the parent dentry); an fd to a removed directory can become the cwd.
    run(|g| {
        assert_eq!(g.mkdir("/tmp/a", 0o755), 0);
        assert_eq!(g.mkdir("/tmp/a/b", 0o755), 0);
        let fd = g.open("/tmp/a/b", O_RDONLY | O_DIRECTORY);
        assert_eq!(g.chdir("/tmp/a/b"), 0);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/a/b", AT_REMOVEDIR), 0);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/a", AT_REMOVEDIR), 0);
        assert_eq!(g.getcwd().0, -ENOENT);
        assert_eq!(g.mkdir("x", 0o755), -ENOENT);
        assert_eq!(g.symlink("/", "l"), -ENOENT);
        // /proc/self/cwd of a removed directory reads without failing.
        assert!(g.readlink("/proc/self/cwd").0 > 0);
        assert_eq!(g.chdir(".."), 0);
        assert_eq!(g.getcwd().0, -ENOENT);
        assert_eq!(g.syscall(Sysno::fchdir, &[fd as u64]), 0);
        assert_eq!(g.close(fd), 0);
        assert_eq!(g.fstat(fd).0, -EBADF);
        assert_eq!(g.getcwd().0, -ENOENT, "the cwd keeps it alive");
        assert_eq!(g.chdir("/tmp"), 0);
        assert_eq!(g.getcwd().1, b"/tmp");
        assert_eq!(g.stat("/tmp/a").0, -ENOENT);
        // A name can be reused at once.
        assert_eq!(g.mkdir("/tmp/a", 0o755), 0);
    });
}

#[test]
fn a_child_inherits_a_removed_cwd() {
    // fork(2) copies the cwd; the directory lives until the last process leaves it.
    run(|g| {
        assert_eq!(g.mkdir("/tmp/c", 0o755), 0);
        assert_eq!(g.chdir("/tmp/c"), 0);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/c", AT_REMOVEDIR), 0);
        let child = g.fork(|c| {
            assert_eq!(c.getcwd().0, -ENOENT);
            assert_eq!(c.open("f", O_CREAT | O_WRONLY), -ENOENT);
        });
        assert_eq!(g.chdir("/"), 0);
        let (r, status) = g.wait4(child, 0);
        assert_eq!((r, status), (child, 0));
    });
}

// ---- Epoll instances watching each other ----

fn epoll_add(g: &mut Guest, epfd: i64, fd: i64) -> i64 {
    let mut ev = [0u8; 16];
    ev[..4].copy_from_slice(&EPOLLIN.to_le_bytes());
    let p = g.put(&ev);
    g.syscall(
        Sysno::epoll_ctl,
        &[epfd as u64, EPOLL_CTL_ADD, fd as u64, p],
    )
}

#[test]
fn epoll_instances_cannot_watch_each_other() {
    // epoll_ctl(2): ELOOP when the add would make a loop of epoll instances monitoring one
    // another. Waiting on either then works as usual.
    run(|g| {
        let a = g.syscall(Sysno::epoll_create1, &[0]);
        let b = g.syscall(Sysno::epoll_create1, &[0]);
        assert_eq!(epoll_add(g, a, b), 0, "nesting is allowed");
        assert_eq!(epoll_add(g, b, a), -ELOOP);
        let events = g.alloc(64);
        assert_eq!(
            g.syscall(Sysno::epoll_pwait, &[a as u64, events, 4, 0, 0, 8]),
            0
        );
        assert_eq!(
            g.syscall(Sysno::epoll_pwait, &[b as u64, events, 4, 0, 0, 8]),
            0
        );
        let fds = g.put(&{
            let mut e = [0u8; 8];
            e[..4].copy_from_slice(&(a as i32).to_le_bytes());
            e[4..6].copy_from_slice(&POLLIN.to_le_bytes());
            e
        });
        let zero = g.put(&timespec(0));
        assert_eq!(g.syscall(Sysno::ppoll, &[fds, 1, zero, 0, 8]), 0);
    });
}

#[test]
fn a_longer_epoll_cycle_is_refused_too() {
    // The loop check covers any path back to the instance, not only a direct pair.
    run(|g| {
        let a = g.syscall(Sysno::epoll_create1, &[0]);
        let b = g.syscall(Sysno::epoll_create1, &[0]);
        let c = g.syscall(Sysno::epoll_create1, &[0]);
        assert_eq!(epoll_add(g, a, b), 0);
        assert_eq!(epoll_add(g, b, c), 0);
        assert_eq!(epoll_add(g, c, a), -ELOOP);
        let events = g.alloc(64);
        assert_eq!(
            g.syscall(Sysno::epoll_pwait, &[a as u64, events, 4, 0, 0, 8]),
            0
        );
    });
}

#[test]
fn epoll_instances_nest_at_most_five_deep() {
    // epoll_ctl(2), ELOOP: "a nesting depth of epoll instances greater than 5" (fs/eventpoll.c
    // EP_MAX_NESTS = 4 levels below the top one), whichever end the chain is built from.
    run(|g| {
        let e: Vec<i64> = (0..6)
            .map(|_| g.syscall(Sysno::epoll_create1, &[0]))
            .collect();
        for i in 0..4 {
            assert_eq!(epoll_add(g, e[i], e[i + 1]), 0, "level {i}");
        }
        assert_eq!(epoll_add(g, e[4], e[5]), -ELOOP, "a sixth level below");
        let top = g.syscall(Sysno::epoll_create1, &[0]);
        assert_eq!(epoll_add(g, top, e[0]), -ELOOP, "a sixth level above");
        // Built from the bottom up: the depth below the added instance counts.
        let f: Vec<i64> = (0..6)
            .map(|_| g.syscall(Sysno::epoll_create1, &[0]))
            .collect();
        for i in (1..5).rev() {
            assert_eq!(epoll_add(g, f[i], f[i + 1]), 0, "level {i}");
        }
        assert_eq!(epoll_add(g, f[0], f[1]), -ELOOP);
        // Waiting on the deepest legal chain works.
        let events = g.alloc(64);
        assert_eq!(
            g.syscall(Sysno::epoll_pwait, &[e[0] as u64, events, 4, 0, 0, 8]),
            0
        );
    });
}

// ---- POSIX message queues ----

#[test]
fn mq_timedsend_with_a_huge_length_is_emsgsize() {
    // mq_send(3): EMSGSIZE when msg_len is greater than the queue's mq_msgsize. The length is
    // a size_t: 2^63 and above are just very large sizes.
    run(|g| {
        let name = g.cstr("q");
        let mq = g.syscall(Sysno::mq_open, &[name, O_CREAT | O_RDWR, 0o600, 0]);
        assert!(mq >= 0, "{mq}");
        let msg = g.put(b"hi");
        for len in [8193u64, 1 << 63, u64::MAX] {
            assert_eq!(
                g.syscall(Sysno::mq_timedsend, &[mq as u64, msg, len, 0, 0]),
                -EMSGSIZE,
                "length {len:#x}"
            );
        }
        // A fitting message still goes through.
        assert_eq!(
            g.syscall(Sysno::mq_timedsend, &[mq as u64, msg, 2, 0, 0]),
            0
        );
    });
}

// ---- sendfile ----

#[test]
fn sendfile_advances_the_input_offset_by_what_it_wrote() {
    // sendfile(2): with a NULL offset, the input file's offset is updated by the bytes
    // transferred; when the output takes fewer bytes than asked (a full non-blocking pipe),
    // the rest is still in the file for the next call.
    run(|g| {
        let data: Vec<u8> = (0..100_000u32).map(|i| (i % 253) as u8).collect();
        let f = g.open("/tmp/src", O_CREAT | O_RDWR);
        assert_eq!(g.write_all(f, &data), data.len() as i64);
        g.lseek(f, 0, 0);
        let (_, rd, wr) = g.pipe2(O_NONBLOCK);
        let n = g.syscall(
            Sysno::sendfile,
            &[wr as u64, f as u64, 0, data.len() as u64],
        );
        assert!(n > 0 && n < data.len() as i64, "a short transfer: {n}");
        assert_eq!(
            g.lseek(f, 0, 1 /* SEEK_CUR */),
            n,
            "the offset follows the bytes sent"
        );
        // Drain the pipe and send the rest: every byte arrives once, in order.
        let mut got = Vec::new();
        while (got.len() as i64) < n {
            got.extend(g.read_vec(rd, 65_536));
        }
        let m = g.syscall(Sysno::sendfile, &[wr as u64, f as u64, 0, 4096]);
        assert_eq!(m, 4096);
        got.extend(g.read_vec(rd, 4096));
        assert_eq!(got, data[..got.len()]);
    });
}

// ---- Signals around ppoll ----

/// `ppoll` on `fd` for POLLIN with a timeout and an optional signal mask: the result.
fn ppoll_one(g: &mut Guest, fd: i64, timeout_ns: u64, mask: Option<u64>) -> i64 {
    let fds = g.put(&{
        let mut e = [0u8; 8];
        e[..4].copy_from_slice(&(fd as i32).to_le_bytes());
        e[4..6].copy_from_slice(&POLLIN.to_le_bytes());
        e
    });
    let tmo = g.put(&timespec(timeout_ns));
    let sigmask = match mask {
        Some(m) => g.put(&m.to_le_bytes()),
        None => 0,
    };
    g.syscall(Sysno::ppoll, &[fds, 1, tmo, sigmask, 8])
}

/// A child that sends `SIGUSR1` to pid 1 after 1 ms.
fn signal_parent_soon(g: &mut Guest) -> i64 {
    g.fork(|c| {
        c.nanosleep(1_000_000);
        c.syscall(Sysno::kill, &[1, u64::from(SIGUSR1)]);
    })
}

#[test]
fn sa_restart_does_not_restart_ppoll() {
    // signal(7): poll(2) and ppoll(2) are never restarted after a handler, whatever SA_RESTART
    // says; they fail with EINTR.
    run(|g| {
        assert_eq!(g.handle(SIGUSR1, SA_RESTART), 0);
        let (_, rd, _wr) = g.pipe2(0);
        let child = signal_parent_soon(g);
        let start = g.clock_ns(1);
        assert_eq!(ppoll_one(g, rd, 50_000_000, None), -EINTR);
        assert_eq!(g.signals(), [SIGUSR1]);
        assert_eq!(g.clock_ns(1), start + 1_000_000);
        g.wait4(child, 0);
    });
}

#[test]
fn ppoll_installs_its_signal_mask_while_it_waits() {
    // ppoll(2): the mask argument replaces the thread's signal mask for the duration of the
    // call, so a signal blocked outside may interrupt the wait (EINTR, handler run).
    run(|g| {
        assert_eq!(g.handle(SIGUSR1, 0), 0);
        let usr1 = 1u64 << (SIGUSR1 - 1);
        let set = g.put(&usr1.to_le_bytes());
        assert_eq!(g.syscall(Sysno::rt_sigprocmask, &[SIG_BLOCK, set, 0, 8]), 0);
        let (_, rd, _wr) = g.pipe2(0);
        let child = signal_parent_soon(g);
        assert_eq!(ppoll_one(g, rd, 50_000_000, Some(0)), -EINTR);
        assert_eq!(g.signals(), [SIGUSR1]);
        // The old mask is back: SIGUSR1 is blocked again.
        let old = g.alloc(8);
        assert_eq!(g.syscall(Sysno::rt_sigprocmask, &[0, 0, old, 8]), 0);
        assert_eq!(g.u64_at(old), usr1);
        g.wait4(child, 0);
    });
}

/// `pselect6` waiting to read `fd`, with a timeout and an optional signal mask.
fn pselect_one(g: &mut Guest, fd: i64, timeout_ns: u64, mask: Option<u64>) -> i64 {
    let set = g.put(&(1u64 << fd).to_le_bytes());
    let tmo = g.put(&timespec(timeout_ns));
    // The sixth argument: `{ const sigset_t *ss; size_t ss_len; }`.
    let pack = match mask {
        Some(m) => {
            let ss = g.put(&m.to_le_bytes());
            let mut b = [0u8; 16];
            b[..8].copy_from_slice(&ss.to_le_bytes());
            b[8..].copy_from_slice(&8u64.to_le_bytes());
            g.put(&b)
        }
        None => 0,
    };
    g.syscall(Sysno::pselect6, &[fd as u64 + 1, set, 0, 0, tmo, pack])
}

/// `epoll_pwait` on a new epoll instance watching `fd` for input, with an optional mask.
fn epoll_pwait_one(g: &mut Guest, fd: i64, timeout_ms: u64, mask: Option<u64>) -> i64 {
    let ep = g.syscall(Sysno::epoll_create1, &[0]);
    assert_eq!(epoll_add(g, ep, fd), 0);
    let events = g.alloc(64);
    let sigmask = match mask {
        Some(m) => g.put(&m.to_le_bytes()),
        None => 0,
    };
    let r = g.syscall(
        Sysno::epoll_pwait,
        &[ep as u64, events, 4, timeout_ms, sigmask, 8],
    );
    g.close(ep);
    r
}

/// The calling thread's signal mask.
fn sigmask_now(g: &mut Guest) -> u64 {
    let old = g.alloc(8);
    assert_eq!(g.syscall(Sysno::rt_sigprocmask, &[0, 0, old, 8]), 0);
    g.u64_at(old)
}

/// A readiness wait on one descriptor: (fd, timeout in ns, signal mask) to its result.
type Wait = fn(&mut Guest, i64, u64, Option<u64>) -> i64;

#[test]
fn sa_restart_does_not_restart_pselect6_or_epoll_pwait() {
    // signal(7): select(2), pselect(2) and epoll_wait(2) are never restarted either.
    run(|g| {
        assert_eq!(g.handle(SIGUSR1, SA_RESTART), 0);
        let (_, rd, _wr) = g.pipe2(0);
        let waits: [Wait; 2] = [pselect_one, |g, fd, ns, m| {
            epoll_pwait_one(g, fd, ns / 1_000_000, m)
        }];
        for wait in waits {
            let child = signal_parent_soon(g);
            let start = g.clock_ns(1);
            assert_eq!(wait(g, rd, 50_000_000, None), -EINTR);
            assert_eq!(g.signals(), [SIGUSR1]);
            assert_eq!(g.clock_ns(1), start + 1_000_000);
            g.wait4(child, 0);
        }
    });
}

#[test]
fn pselect6_and_epoll_pwait_install_their_signal_masks() {
    // pselect(2), epoll_pwait(2): the mask replaces the thread's for the call, as in ppoll.
    run(|g| {
        assert_eq!(g.handle(SIGUSR1, 0), 0);
        let usr1 = 1u64 << (SIGUSR1 - 1);
        let set = g.put(&usr1.to_le_bytes());
        assert_eq!(g.syscall(Sysno::rt_sigprocmask, &[SIG_BLOCK, set, 0, 8]), 0);
        let (_, rd, _wr) = g.pipe2(0);
        let child = signal_parent_soon(g);
        assert_eq!(pselect_one(g, rd, 50_000_000, Some(0)), -EINTR);
        assert_eq!(g.signals(), [SIGUSR1]);
        assert_eq!(sigmask_now(g), usr1);
        g.wait4(child, 0);
        let child = signal_parent_soon(g);
        assert_eq!(epoll_pwait_one(g, rd, 50, Some(0)), -EINTR);
        assert_eq!(g.signals(), [SIGUSR1]);
        assert_eq!(sigmask_now(g), usr1);
        g.wait4(child, 0);
        // A mask of the wrong size is EINVAL (the kernel's sigset_t is 8 bytes).
        let fds = g.alloc(8);
        let zero = g.put(&timespec(0));
        let m = g.put(&0u64.to_le_bytes());
        assert_eq!(g.syscall(Sysno::ppoll, &[fds, 0, zero, m, 4]), -EINVAL);
    });
}

#[test]
fn a_pending_signal_the_call_mask_unblocks_ends_the_wait_at_once() {
    // ppoll(2): the mask is installed atomically with the wait, so a signal already pending
    // and blocked outside the call interrupts it before it waits.
    run(|g| {
        assert_eq!(g.handle(SIGUSR1, 0), 0);
        let usr1 = 1u64 << (SIGUSR1 - 1);
        let set = g.put(&usr1.to_le_bytes());
        assert_eq!(g.syscall(Sysno::rt_sigprocmask, &[SIG_BLOCK, set, 0, 8]), 0);
        assert_eq!(g.syscall(Sysno::kill, &[1, u64::from(SIGUSR1)]), 0);
        assert!(g.signals().is_empty(), "blocked: still pending");
        let (_, rd, _wr) = g.pipe2(0);
        let start = g.clock_ns(1);
        assert_eq!(ppoll_one(g, rd, 50_000_000, Some(0)), -EINTR);
        assert_eq!(g.signals(), [SIGUSR1]);
        assert_eq!(g.clock_ns(1), start, "no waiting");
        assert_eq!(sigmask_now(g), usr1);
    });
}

#[test]
fn the_old_mask_is_back_after_a_wait_that_times_out() {
    // The call's mask lasts only for the call: a signal it blocked is delivered once the call
    // returns (after its timeout), under the restored mask.
    run(|g| {
        assert_eq!(g.handle(SIGUSR1, 0), 0);
        let usr1 = 1u64 << (SIGUSR1 - 1);
        let (_, rd, _wr) = g.pipe2(0);
        let child = signal_parent_soon(g);
        let start = g.clock_ns(1);
        assert_eq!(ppoll_one(g, rd, 5_000_000, Some(usr1)), 0);
        assert_eq!(g.clock_ns(1), start + 5_000_000);
        assert_eq!(g.signals(), [SIGUSR1], "delivered after the call");
        assert_eq!(sigmask_now(g), 0);
        g.wait4(child, 0);
    });
}

// ---- Small fidelity gaps ----

#[test]
fn f_getpipe_sz_reports_the_pipe_capacity() {
    // fcntl(2), F_GETPIPE_SZ: the pipe's capacity; 65536 bytes by default (pipe(7)).
    run(|g| {
        let (_, rd, wr) = g.pipe2(0);
        assert_eq!(g.syscall(Sysno::fcntl, &[rd as u64, 1032]), 65_536);
        assert_eq!(g.syscall(Sysno::fcntl, &[wr as u64, 1032]), 65_536);
        // Not a pipe: EBADF.
        let f = g.open("/etc/hostname", O_RDONLY);
        assert_eq!(g.syscall(Sysno::fcntl, &[f as u64, 1032]), -EBADF);
    });
}

#[test]
fn mknodat_makes_a_fifo() {
    // mknod(2): a FIFO needs no privilege (EPERM is for device nodes as an unprivileged user).
    run(|g| {
        let p = g.cstr("/tmp/fifo");
        assert_eq!(
            g.syscall(
                Sysno::mknodat,
                &[AT_FDCWD as u64, p, u64::from(S_IFIFO | 0o644), 0]
            ),
            0
        );
        assert_eq!(g.stat("/tmp/fifo").1.kind(), S_IFIFO);
    });
}

#[test]
fn sigaltstack_refuses_a_stack_below_minsigstksz() {
    // sigaltstack(2): ENOMEM when ss_size is less than MINSIGSTKSZ, which is 5120 on arm64
    // (arch/arm64/include/uapi/asm/signal.h).
    run(|g| {
        let stack = g.alloc(16384);
        let ss = |sp: u64, size: u64| {
            let mut b = [0u8; 24];
            b[..8].copy_from_slice(&sp.to_le_bytes());
            b[16..].copy_from_slice(&size.to_le_bytes());
            b
        };
        let small = g.put(&ss(stack, 4096));
        assert_eq!(g.syscall(Sysno::sigaltstack, &[small, 0]), -ENOMEM);
        let fine = g.put(&ss(stack, 5120));
        assert_eq!(g.syscall(Sysno::sigaltstack, &[fine, 0]), 0);
    });
}

// ---- A directory listed while it changes ----

/// One `getdents64` call into a buffer of `len` bytes: `(name, d_off)` of each record.
fn getdents(g: &mut Guest, fd: i64, len: u64) -> Vec<(String, u64)> {
    let buf = g.alloc(len as usize);
    let n = g.syscall(Sysno::getdents64, &[fd as u64, buf, len]);
    assert!(n >= 0, "getdents64({fd}) failed: {n}");
    let raw = g.bytes(buf, n);
    let mut out = Vec::new();
    let mut at = 0;
    while at < raw.len() {
        let off = u64::from_le_bytes(raw[at + 8..at + 16].try_into().unwrap());
        let reclen = usize::from(u16::from_le_bytes([raw[at + 16], raw[at + 17]]));
        let name = &raw[at + 19..at + reclen];
        let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
        out.push((String::from_utf8_lossy(&name[..end]).into_owned(), off));
        at += reclen;
    }
    out
}

/// Creates `n` empty files `<dir>/<prefix><i>`.
fn create_files(g: &mut Guest, dir: &str, prefix: &str, n: usize) {
    for i in 0..n {
        let f = g.open(&format!("{dir}/{prefix}{i:03}"), O_CREAT | O_WRONLY);
        assert!(f >= 0, "create: {f}");
        g.close(f);
    }
}

#[test]
fn unlinking_what_was_listed_skips_nothing() {
    // readdir(3p): an entry neither removed nor added since the listing began is returned
    // exactly once. `rm -rf` reads a buffer, unlinks what it read and reads on: it must find
    // every name, however many buffers the directory takes.
    run(|g| {
        assert_eq!(g.mkdir("/tmp/d", 0o755), 0);
        create_files(g, "/tmp/d", "f", 100);
        let fd = g.open("/tmp/d", O_RDONLY | O_DIRECTORY);
        let mut removed = 0;
        loop {
            let batch = getdents(g, fd, 256);
            if batch.is_empty() {
                break;
            }
            for (name, _) in batch {
                if name != "." && name != ".." {
                    assert_eq!(g.unlinkat(fd as i32, &name, 0), 0, "{name}");
                    removed += 1;
                }
            }
        }
        assert_eq!(removed, 100);
        assert_eq!(g.close(fd), 0);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/d", AT_REMOVEDIR), 0);
    });
}

#[test]
fn names_created_while_listing_repeat_nothing() {
    // Names that sort before the position, created after the first buffer, must not push
    // entries already returned back into the next one.
    run(|g| {
        assert_eq!(g.mkdir("/tmp/e", 0o755), 0);
        create_files(g, "/tmp/e", "b", 40);
        let fd = g.open("/tmp/e", O_RDONLY | O_DIRECTORY);
        let mut names: Vec<String> = getdents(g, fd, 256).into_iter().map(|e| e.0).collect();
        create_files(g, "/tmp/e", "a", 20);
        loop {
            let batch = getdents(g, fd, 256);
            if batch.is_empty() {
                break;
            }
            names.extend(batch.into_iter().map(|e| e.0));
        }
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "a name came twice: {names:?}");
        for i in 0..40 {
            assert!(names.contains(&format!("b{i:03}")), "b{i:03} missing");
        }
    });
}

#[test]
fn d_off_counts_entries_and_seekdir_returns_there() {
    // An unchanged directory: each record's d_off is the position after it (1, 2, 3, ...);
    // lseek to one of them (seekdir) continues after that entry, and to 0 (rewinddir)
    // starts over.
    run(|g| {
        let fd = g.open("/data", O_RDONLY | O_DIRECTORY);
        let all = getdents(g, fd, 4096);
        let names: Vec<&str> = all.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(names, [".", "..", "big", "dangling", "link", "sub"]);
        let offs: Vec<u64> = all.iter().map(|e| e.1).collect();
        assert_eq!(offs, [1, 2, 3, 4, 5, 6]);
        // One record at a time gives the same records.
        assert_eq!(g.lseek(fd, 0, 0), 0);
        let mut one_by_one = Vec::new();
        loop {
            let batch = getdents(g, fd, 32);
            if batch.is_empty() {
                break;
            }
            one_by_one.extend(batch);
        }
        assert_eq!(one_by_one, all);
        assert_eq!(g.lseek(fd, 3, 0), 3);
        assert_eq!(getdents(g, fd, 4096), all[3..]);
        assert_eq!(g.lseek(fd, 0, 0), 0);
        assert_eq!(getdents(g, fd, 4096), all);
    });
}

// ---- kill(-1) ----

#[test]
fn kill_minus_one_spares_the_caller() {
    // kill(2): pid -1 signals every process the caller may signal except init; Linux skips
    // the caller too (kernel/signal.c, `kill_something_info`), and with nobody left the call
    // fails with ESRCH.
    run(|g| {
        assert_eq!(g.handle(SIGUSR1, 0), 0);
        let minus_one = -1i64 as u64;
        let lone = g.fork(move |c| {
            let r = c.syscall(Sysno::kill, &[minus_one, u64::from(SIGUSR1)]);
            let ok = r == -ESRCH && c.signals().is_empty();
            c.exit(if ok { 0 } else { 1 });
        });
        assert_eq!(
            g.wait4(lone, 0),
            (lone, 0),
            "alone, the caller signalled itself"
        );
        let sibling = g.fork(|c| {
            c.nanosleep(10_000_000_000);
            let got = c.signals();
            c.exit(if got == [SIGUSR1] { 7 } else { 1 });
        });
        let sender = g.fork(move |c| {
            let r = c.syscall(Sysno::kill, &[minus_one, u64::from(SIGUSR1)]);
            let ok = r == 0 && c.signals().is_empty();
            c.exit(if ok { 0 } else { 1 });
        });
        assert_eq!(
            g.wait4(sender, 0),
            (sender, 0),
            "the sender got its own signal"
        );
        assert_eq!(
            g.wait4(sibling, 0),
            (sibling, 7 << 8),
            "the other process got it"
        );
        assert_eq!(g.signals(), [], "init is spared");
    });
}

// ---- Timeouts too large for nanoseconds ----

/// A `struct timespec` of `sec` seconds and `nsec` nanoseconds.
fn timespec_raw(sec: i64, nsec: i64) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&sec.to_le_bytes());
    b[8..].copy_from_slice(&nsec.to_le_bytes());
    b
}

#[test]
fn a_timeout_past_the_largest_time_waits_for_ever() {
    // A valid timespec whose nanoseconds do not fit is KTIME_MAX (include/linux/ktime.h,
    // `timespec64_to_ktime`): `i64::MAX` seconds is accepted and never expires. Negative or
    // malformed values stay EINVAL (`timespec64_valid`).
    const FUTEX_WAIT_PRIVATE: u64 = 128;
    const FUTEX_WAKE_PRIVATE: u64 = 129;
    const CLOCK_MONOTONIC: u64 = 1;
    run(|g| {
        let forever = g.put(&timespec_raw(i64::MAX, 999_999_999));
        assert_eq!(g.handle(SIGUSR1, 0), 0);

        // ppoll, woken by a write a second later.
        let (_, rd, wr) = g.pipe2(0);
        let writer = g.fork(move |c| {
            c.nanosleep(1_000_000_000);
            c.write_all(wr, b"x");
        });
        let mut pfd = (rd as i32).to_le_bytes().to_vec();
        pfd.extend_from_slice(&POLLIN.to_le_bytes());
        pfd.extend_from_slice(&[0, 0]);
        let pfd = g.put(&pfd);
        assert_eq!(g.syscall(Sysno::ppoll, &[pfd, 1, forever, 0, 8]), 1);

        // pselect6 on the same pipe, still readable.
        let set = g.put(&(1u64 << rd).to_le_bytes());
        assert_eq!(
            g.syscall(Sysno::pselect6, &[rd as u64 + 1, set, 0, 0, forever, 0]),
            1
        );

        // A futex wait, woken by a thread a second later.
        let word = g.put(&0u32.to_le_bytes());
        g.spawn_thread(move |t| {
            t.nanosleep(1_000_000_000);
            t.syscall(Sysno::futex, &[word, FUTEX_WAKE_PRIVATE, 1]);
        });
        assert_eq!(
            g.syscall(Sysno::futex, &[word, FUTEX_WAIT_PRIVATE, 0, forever]),
            0
        );

        // nanosleep and clock_nanosleep, each ended by a signal a second later.
        for clock in [None, Some(CLOCK_MONOTONIC)] {
            let killer = g.fork(|c| {
                c.nanosleep(1_000_000_000);
                c.syscall(Sysno::kill, &[1, u64::from(SIGUSR1)]);
            });
            let r = match clock {
                None => g.syscall(Sysno::nanosleep, &[forever, 0]),
                Some(clk) => g.syscall(Sysno::clock_nanosleep, &[clk, 0, forever, 0]),
            };
            assert_eq!(r, -EINTR, "{clock:?}");
            assert_eq!(g.signals(), [SIGUSR1]);
            assert_eq!(g.wait4(killer, 0), (killer, 0));
        }
        assert_eq!(g.wait4(writer, 0), (writer, 0));

        // Malformed values keep their errors.
        for (sec, nsec) in [
            (-1, 0),
            (0, -1),
            (0, 1_000_000_000),
            (i64::MAX, 1_000_000_000),
        ] {
            let ts = g.put(&timespec_raw(sec, nsec));
            assert_eq!(
                g.syscall(Sysno::nanosleep, &[ts, 0]),
                -EINVAL,
                "{sec} {nsec}"
            );
            assert_eq!(g.syscall(Sysno::ppoll, &[pfd, 1, ts, 0, 8]), -EINVAL);
        }
    });
}

// ---- A scheduling quantum of zero ----

#[test]
fn a_zero_quantum_runs_as_one() {
    // A quantum of nothing would yield before every instruction: the machine runs with 1.
    let mut m = Machine::default();
    m.config.quantum = 0;
    let out = m.run(|g| {
        let child = g.fork(|c| {
            c.write_all(1, b"child\n");
        });
        assert_eq!(g.wait4(child, 0), (child, 0));
        g.write_all(1, b"parent\n");
    });
    assert_eq!(out.status, wasmshim_linux::ExitStatus::Code(0));
    assert_eq!(out.stdout, b"child\nparent\n");
}

// ---- sendfile's range ----

#[test]
fn sendfile_refuses_a_range_past_the_largest_offset() {
    // sendfile(2) checks its range like read (fs/read_write.c, `rw_verify_area`): a start
    // that is negative as a loff_t, a range whose end passes it, or a count that is negative
    // as a ssize_t is EINVAL, and the offset is left alone.
    run(|g| {
        let f = g.open("/data/big", O_RDONLY) as u64;
        let out = g.open("/tmp/out", O_CREAT | O_WRONLY) as u64;
        for start in [1u64 << 63, u64::MAX, i64::MAX as u64] {
            let off = g.put(&start.to_le_bytes());
            assert_eq!(
                g.syscall(Sysno::sendfile, &[out, f, off, 100]),
                -EINVAL,
                "{start:#x}"
            );
            assert_eq!(g.u64_at(off), start);
        }
        let off = g.put(&0u64.to_le_bytes());
        assert_eq!(g.syscall(Sysno::sendfile, &[out, f, off, 1 << 63]), -EINVAL);
        assert_eq!(g.syscall(Sysno::sendfile, &[out, f, 0, 1 << 63]), -EINVAL);
        // In range, it sends.
        assert_eq!(g.syscall(Sysno::sendfile, &[out, f, off, 100]), 100);
        assert_eq!(g.u64_at(off), 100);
    });
}

// ---- Socket option lengths ----

#[test]
fn integer_socket_options_need_a_whole_int() {
    // setsockopt(2): an integer option with optlen below sizeof(int) is EINVAL at the socket
    // and TCP levels (net/core/sock.c `sk_setsockopt`, net/ipv4/tcp.c), and a negative optlen
    // is EINVAL (net/socket.c). getsockopt(2) writes back how many bytes it copied
    // (`sk_getsockopt`: `if (len > lv) len = lv`).
    const AF_INET: u64 = 2;
    const SOCK_STREAM: u64 = 1;
    const SOL_SOCKET: u64 = 1;
    const SO_KEEPALIVE: u64 = 9;
    const IPPROTO_TCP: u64 = 6;
    const TCP_NODELAY: u64 = 1;
    run(|g| {
        let s = g.syscall(Sysno::socket, &[AF_INET, SOCK_STREAM, 0]) as u64;
        let one = g.put(&1u32.to_le_bytes());
        for len in [0, 1, 3, u64::MAX, 0xffff_ffff] {
            for (level, opt) in [(SOL_SOCKET, SO_KEEPALIVE), (IPPROTO_TCP, TCP_NODELAY)] {
                assert_eq!(
                    g.syscall(Sysno::setsockopt, &[s, level, opt, one, len]),
                    -EINVAL,
                    "level {level} option {opt} length {len:#x}"
                );
            }
        }
        // Nothing was set by those.
        let val = g.put(&[0xaa; 8]);
        let len = g.put(&4u32.to_le_bytes());
        let get = [s, IPPROTO_TCP, TCP_NODELAY, val, len];
        assert_eq!(g.syscall(Sysno::getsockopt, &get), 0);
        assert_eq!(g.u32_at(val), 0);
        assert_eq!(
            g.syscall(Sysno::setsockopt, &[s, SOL_SOCKET, SO_KEEPALIVE, one, 4]),
            0
        );
        // Two bytes asked for: two copied, and the length says two.
        g.write_mem(val, &[0xaa; 8]);
        g.write_mem(len, &2u32.to_le_bytes());
        let get = [s, SOL_SOCKET, SO_KEEPALIVE, val, len];
        assert_eq!(g.syscall(Sysno::getsockopt, &get), 0);
        assert_eq!(g.u32_at(len), 2);
        assert_eq!(g.bytes(val, 4usize), [1, 0, 0xaa, 0xaa]);
        // A negative length is EINVAL.
        g.write_mem(len, &(-1i32).to_le_bytes());
        assert_eq!(g.syscall(Sysno::getsockopt, &get), -EINVAL);
    });
}

// ---- xattr sizes ----

#[test]
fn xattr_sizes_are_compared_before_they_are_narrowed() {
    // setxattr(2): E2BIG over XATTR_SIZE_MAX; getxattr(2) and listxattr(2): ERANGE only for
    // a buffer smaller than the value. Sizes are 64-bit: 2^32 + 5 is not 5 (on a 32-bit host,
    // where the node runs, narrowing first made it so).
    run(|g| {
        let f = g.open("/tmp/x", O_CREAT | O_WRONLY);
        g.close(f);
        let path = g.cstr("/tmp/x");
        let name = g.cstr("user.k");
        let value = g.put(b"value");
        let big = (1u64 << 32) + 5;
        assert_eq!(
            g.syscall(Sysno::setxattr, &[path, name, value, big, 0]),
            -E2BIG
        );
        assert_eq!(g.syscall(Sysno::setxattr, &[path, name, value, 5, 0]), 0);
        let buf = g.alloc(64);
        assert_eq!(
            g.syscall(Sysno::getxattr, &[path, name, buf, (1 << 32) + 1]),
            5
        );
        assert_eq!(g.bytes(buf, 5usize), b"value");
        assert_eq!(g.syscall(Sysno::listxattr, &[path, buf, (1 << 32) + 1]), 7);
        assert_eq!(g.bytes(buf, 7usize), b"user.k\0");
    });
}
