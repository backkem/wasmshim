//! Guest-supplied sizes, offsets and options at their extremes: the personality answers as
//! Linux does and never panics, overflows or allocates without bound. Each test states the
//! Linux behaviour.

mod support;

use support::*;
use wasmshim_linux_abi::consts::at::AT_REMOVEDIR;
use wasmshim_linux_abi::consts::map::{MAP_ANONYMOUS, MAP_PRIVATE};
use wasmshim_linux_abi::consts::mode::{S_IFCHR, S_IFMT, S_IFREG};
use wasmshim_linux_abi::consts::open::*;
use wasmshim_linux_abi::consts::prot::{PROT_READ, PROT_WRITE};
use wasmshim_linux_abi::consts::signal::*;

const AF_UNIX: u64 = 1;
const AF_INET: u64 = 2;
const AF_NETLINK: u64 = 16;
const SOCK_STREAM: u64 = 1;
const SOCK_DGRAM: u64 = 2;
const SOCK_RAW: u64 = 3;
const SOL_SOCKET: u64 = 1;
const SO_REUSEADDR: u64 = 2;
const SO_RCVBUF: u64 = 8;
const IPPROTO_TCP: u64 = 6;
const TCP_NODELAY: u64 = 1;
const MSG_TRUNC: u64 = 0x20;
const MADV_DONTNEED: u64 = 4;

/// `socketpair(AF_UNIX, ty, 0)`: the two ends.
fn socketpair(g: &mut Guest, ty: u64) -> (i64, i64) {
    let sv = g.alloc(8);
    assert_eq!(g.syscall(Sysno::socketpair, &[AF_UNIX, ty, 0, sv]), 0);
    (i64::from(g.i32_at(sv)), i64::from(g.i32_at(sv + 4)))
}

/// A `struct iovec` array.
fn iovecs(g: &mut Guest, v: &[(u64, u64)]) -> u64 {
    let raw: Vec<u8> = v
        .iter()
        .flat_map(|(base, len)| base.to_le_bytes().into_iter().chain(len.to_le_bytes()))
        .collect();
    g.put(&raw)
}

/// A `struct msghdr` (LP64) with no name and no control buffer.
fn msghdr(g: &mut Guest, iov: u64, iovlen: u64) -> u64 {
    let mut h = [0u8; 56];
    h[16..24].copy_from_slice(&iov.to_le_bytes());
    h[24..32].copy_from_slice(&iovlen.to_le_bytes());
    g.put(&h)
}

/// `setsockopt(fd, level, opt, &1, 4)`.
fn setsockopt_int(g: &mut Guest, fd: i64, level: u64, opt: u64) -> i64 {
    let one = g.put(&1u32.to_le_bytes());
    g.syscall(Sysno::setsockopt, &[fd as u64, level, opt, one, 4])
}

// ---- Sockets ----

#[test]
fn sendmsg_gathers_at_most_as_much_as_writev() {
    // sendmsg(2) of 1024 iovecs of 16 MiB each: like writev, one call moves at most 16 MiB
    // (a short count on a stream), never 16 GiB of host memory.
    run(|g| {
        let len = 16u64 << 20;
        let buf = g.mmap(
            0,
            len,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        );
        assert!(buf > 0, "{buf}");
        let iov = iovecs(g, &vec![(buf as u64, len); 1024]);
        let msg = msghdr(g, iov, 1024);
        let (a, _b) = socketpair(g, SOCK_STREAM | O_NONBLOCK);
        let n = g.syscall(Sysno::sendmsg, &[a as u64, msg, 0]);
        assert!(n > 0 && n <= len as i64, "{n}");
    });
}

#[test]
fn recvmsg_with_iovec_lengths_past_2_64_receives() {
    // The lengths are the guest's: their sum may overflow; the receive fills the first iovec.
    run(|g| {
        let (a, b) = socketpair(g, SOCK_DGRAM);
        assert_eq!(g.write_all(a, b"hello"), 5);
        let (x, y) = (g.alloc(16), g.alloc(16));
        let iov = iovecs(g, &[(x, u64::MAX), (y, u64::MAX)]);
        let msg = msghdr(g, iov, 2);
        assert_eq!(g.syscall(Sysno::recvmsg, &[b as u64, msg, 0]), 5);
        assert_eq!(g.bytes(x, 5usize), b"hello");
    });
}

#[test]
fn vmsplice_with_iovec_lengths_past_2_64_reads_the_pipe() {
    // As for recvmsg: the iovec lengths are the guest's and may overflow when added.
    run(|g| {
        let (_, rd, wr) = g.pipe2(0);
        assert_eq!(g.write_all(wr, b"hello"), 5);
        let (x, y) = (g.alloc(16), g.alloc(16));
        let iov = iovecs(g, &[(x, u64::MAX), (y, u64::MAX)]);
        assert_eq!(g.syscall(Sysno::vmsplice, &[rd as u64, iov, 2, 0]), 5);
        assert_eq!(g.bytes(x, 5usize), b"hello");
    });
}

#[test]
fn msg_trunc_returns_the_real_datagram_length() {
    // recv(2), MSG_TRUNC: "return the real length of the packet or datagram, even when it was
    // longer than the passed buffer"; without it, the bytes copied (and msg_flags says
    // MSG_TRUNC).
    run(|g| {
        let (a, b) = socketpair(g, SOCK_DGRAM);
        let buf = g.alloc(16);
        let datagram = b"0123456789";
        g.write_all(a, datagram);
        let r = g.syscall(Sysno::recvfrom, &[b as u64, buf, 4, MSG_TRUNC, 0, 0]);
        assert_eq!(r, 10);
        assert_eq!(g.bytes(buf, 4usize), b"0123");
        g.write_all(a, datagram);
        assert_eq!(g.syscall(Sysno::recvfrom, &[b as u64, buf, 4, 0, 0, 0]), 4);
        for (flags, want) in [(0, 4), (MSG_TRUNC, 10)] {
            g.write_all(a, datagram);
            let iov = iovecs(g, &[(buf, 4)]);
            let msg = msghdr(g, iov, 1);
            assert_eq!(g.syscall(Sysno::recvmsg, &[b as u64, msg, flags]), want);
            assert_eq!(g.u32_at(msg + 48) as u64 & MSG_TRUNC, MSG_TRUNC);
        }
        // A datagram that fits: its length either way.
        g.write_all(a, b"ab");
        let r = g.syscall(Sysno::recvfrom, &[b as u64, buf, 16, MSG_TRUNC, 0, 0]);
        assert_eq!(r, 2);
    });
}

#[test]
fn setsockopt_at_a_level_the_socket_does_not_have_fails() {
    // AF_UNIX sockets have only SOL_SOCKET options (no protocol setsockopt: EOPNOTSUPP); an IP
    // socket refuses a level it has no options at (ENOPROTOOPT), as does netlink outside
    // SOL_SOCKET and SOL_NETLINK.
    run(|g| {
        let (u, _) = socketpair(g, SOCK_STREAM);
        assert_eq!(setsockopt_int(g, u, IPPROTO_TCP, TCP_NODELAY), -EOPNOTSUPP);
        assert_eq!(setsockopt_int(g, u, SOL_SOCKET, SO_REUSEADDR), 0);
        let s = g.syscall(Sysno::socket, &[AF_INET, SOCK_STREAM, 0]);
        assert!(s >= 0, "{s}");
        assert_eq!(setsockopt_int(g, s, 999, 1), -ENOPROTOOPT);
        assert_eq!(setsockopt_int(g, s, IPPROTO_TCP, TCP_NODELAY), 0);
        assert_eq!(setsockopt_int(g, s, SOL_SOCKET, SO_REUSEADDR), 0);
        let n = g.syscall(Sysno::socket, &[AF_NETLINK, SOCK_RAW, 0]);
        assert!(n >= 0, "{n}");
        assert_eq!(setsockopt_int(g, n, IPPROTO_TCP, TCP_NODELAY), -ENOPROTOOPT);
        assert_eq!(setsockopt_int(g, n, SOL_SOCKET, SO_RCVBUF), 0);
    });
}

// ---- Memory ----

#[test]
fn madvise_refuses_unknown_advice_and_keeps_the_vdso() {
    // madvise(2), EINVAL: "advice is not valid". MADV_DONTNEED on the vDSO page refaults the
    // same code (it is a special mapping, not anonymous memory).
    run(|g| {
        let page = g.mmap(
            0,
            4096,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        );
        let page = page as u64;
        for bad in [5, 6, 7, 26, 99, 104, u64::MAX] {
            assert_eq!(
                g.syscall(Sysno::madvise, &[page, 4096, bad]),
                -EINVAL,
                "advice {bad}"
            );
        }
        for good in [0, 3, 14, 15, 16, 18] {
            assert_eq!(g.syscall(Sysno::madvise, &[page, 4096, good]), 0, "{good}");
        }
        let tramp = wasmshim_linux::SIGTRAMP;
        let before = g.bytes(tramp, 8usize);
        assert_eq!(
            g.syscall(Sysno::madvise, &[tramp & !0xfff, 4096, MADV_DONTNEED]),
            0
        );
        assert_eq!(g.bytes(tramp, 8usize), before);
    });
}

#[test]
fn a_file_mapping_must_end_within_the_largest_offset() {
    // mmap(2), EOVERFLOW (mm/mmap.c `file_mmap_ok`): offset plus length past MAX_LFS_FILESIZE.
    run(|g| {
        let fd = g.open("/data/big", O_RDONLY);
        for off in [
            1u64 << 63,
            0xffff_ffff_ffff_f000,
            (1 << 63) - 4096,
            (1 << 63) - 8192,
        ] {
            assert_eq!(
                g.mmap(0, 8192, PROT_READ, MAP_PRIVATE, fd, off),
                -EOVERFLOW,
                "offset {off:#x}"
            );
        }
        // The last offsets that fit (the end at most 2^63 - 4096): a mapping past the end of
        // the file, split and merged.
        let at = g.mmap(0, 8192, PROT_READ, MAP_PRIVATE, fd, (1 << 63) - 12288);
        assert!(at > 0, "{at}");
        let at = at as u64;
        assert_eq!(g.syscall(Sysno::mprotect, &[at + 4096, 4096, 0]), 0);
        assert_eq!(g.syscall(Sysno::mprotect, &[at + 4096, 4096, PROT_READ]), 0);
        assert_eq!(g.syscall(Sysno::munmap, &[at, 8192]), 0);
    });
}

// ---- Signals ----

#[test]
fn an_alternate_stack_that_wraps_ends_in_sigsegv() {
    // sigaltstack(2) takes any ss_sp (only ss_size has a minimum); a frame that does not fit
    // there cannot be written, and the process dies of SIGSEGV (Linux's force_sigsegv).
    run(|g| {
        let child = g.fork(|c| {
            let mut ss = [0u8; 24];
            ss[..8].copy_from_slice(&(u64::MAX - 0xfff).to_le_bytes());
            ss[16..].copy_from_slice(&0x10000u64.to_le_bytes());
            let p = c.put(&ss);
            assert_eq!(c.syscall(Sysno::sigaltstack, &[p, 0]), 0);
            assert_eq!(c.handle(SIGUSR1, SA_ONSTACK), 0);
            let me = c.syscall(Sysno::getpid, &[]);
            c.syscall(Sysno::kill, &[me as u64, u64::from(SIGUSR1)]);
            unreachable!("the signal frame cannot be written");
        });
        let (r, status) = g.wait4(child, 0);
        assert_eq!((r, status & 0x7f), (child, SIGSEGV));
    });
}

#[test]
fn sigaltstack_takes_ss_onstack_as_zero() {
    // kernel/signal.c, do_sigaltstack: ss_flags SS_ONSTACK is accepted (as 0); SS_ONSTACK with
    // SS_DISABLE is not a mode.
    run(|g| {
        let stack = g.alloc(8192);
        let ss = |flags: u32| {
            let mut b = [0u8; 24];
            b[..8].copy_from_slice(&stack.to_le_bytes());
            b[8..12].copy_from_slice(&flags.to_le_bytes());
            b[16..].copy_from_slice(&8192u64.to_le_bytes());
            b
        };
        let bad = g.put(&ss(SS_ONSTACK | SS_DISABLE));
        assert_eq!(g.syscall(Sysno::sigaltstack, &[bad, 0]), -EINVAL);
        let on = g.put(&ss(SS_ONSTACK));
        assert_eq!(g.syscall(Sysno::sigaltstack, &[on, 0]), 0);
        let old = g.alloc(24);
        assert_eq!(g.syscall(Sysno::sigaltstack, &[0, old]), 0);
        assert_eq!(g.u64_at(old), stack);
        assert_eq!(g.u32_at(old + 8), 0, "not on it, enabled");
    });
}

#[test]
fn kill_checks_the_signal_against_every_target() {
    // kill(2): EINVAL for an invalid signal sent to processes that exist; ESRCH when none does.
    run(|g| {
        assert_eq!(g.syscall(Sysno::kill, &[0, 99]), -EINVAL);
        assert_eq!(g.syscall(Sysno::kill, &[4242, 99]), -ESRCH);
        assert_eq!(g.syscall(Sysno::kill, &[4242, 0]), -ESRCH);
    });
}

// ---- Files ----

#[test]
fn statfs_needs_a_path_that_resolves() {
    // statfs(2): ENOENT, ENOTDIR... like any lookup; one filesystem answers for the rest.
    run(|g| {
        let buf = g.alloc(120);
        for (path, want) in [
            ("/tmp", 0),
            ("/data/big", 0),
            ("/proc", 0),
            ("/proc/self/status", 0),
            ("/nope", -ENOENT),
            ("/data/big/x", -ENOTDIR),
            ("/loop/a", -ELOOP),
        ] {
            let p = g.cstr(path);
            assert_eq!(g.syscall(Sysno::statfs, &[p, buf]), want, "{path}");
        }
    });
}

#[test]
fn every_at_call_ignores_dirfd_for_an_absolute_path() {
    // openat(2) and the other *at calls: "If pathname is absolute, then dirfd is ignored."
    run(|g| {
        assert_eq!(g.stat_at(99, "/etc/hostname", 0).0, 0);
        let p = g.cstr("/tmp/m");
        assert_eq!(g.syscall(Sysno::mkdirat, &[99, p, 0o755]), 0);
        assert_eq!(g.unlinkat(99, "/tmp/m", AT_REMOVEDIR), 0);
        // A relative path still needs a valid dirfd.
        assert_eq!(g.stat_at(99, "etc/hostname", 0).0, -EBADF);
    });
}

#[test]
fn preadv_and_pwritev_refuse_a_negative_offset() {
    // preadv(2): EINVAL for a negative offset (as pread).
    run(|g| {
        let fd = g.open("/tmp/v", O_CREAT | O_RDWR);
        let buf = g.alloc(4);
        let iov = iovecs(g, &[(buf, 4)]);
        let neg = (-4096i64) as u64;
        assert_eq!(g.syscall(Sysno::preadv, &[fd as u64, iov, 1, neg]), -EINVAL);
        assert_eq!(
            g.syscall(Sysno::pwritev, &[fd as u64, iov, 1, neg]),
            -EINVAL
        );
        assert_eq!(g.syscall(Sysno::pwritev, &[fd as u64, iov, 1, 8]), 4);
    });
}

#[test]
fn sendfile_into_a_full_pipe_leaves_the_input_offset() {
    // sendfile(2) moved nothing (EAGAIN): the input offset stays where it was.
    run(|g| {
        let f = g.open("/data/big", O_RDONLY);
        let (_, _rd, wr) = g.pipe2(O_NONBLOCK);
        let fill = vec![7u8; 65_536];
        assert_eq!(g.write_all(wr, &fill), 65_536);
        assert_eq!(
            g.syscall(Sysno::sendfile, &[wr as u64, f as u64, 0, 100]),
            -EAGAIN
        );
        assert_eq!(g.lseek(f, 0, 1 /* SEEK_CUR */), 0);
    });
}

#[test]
fn o_creat_follows_a_chain_of_dangling_symlinks() {
    // open(2) follows every trailing symlink to create the name the chain ends at; O_NOFOLLOW
    // refuses the link (ELOOP) and a loop stays ELOOP.
    run(|g| {
        assert_eq!(g.symlink("/tmp/l2", "/tmp/l1"), 0);
        assert_eq!(g.symlink("t", "/tmp/l2"), 0);
        assert_eq!(g.open("/tmp/l1", O_CREAT | O_WRONLY | O_NOFOLLOW), -ELOOP);
        let fd = g.open("/tmp/l1", O_CREAT | O_WRONLY);
        assert!(fd >= 0, "{fd}");
        assert_eq!(g.stat("/tmp/t").1.kind(), S_IFREG);
        assert_eq!(g.open("/loop/a", O_CREAT | O_WRONLY), -ELOOP);
    });
}

#[test]
fn mknodat_checks_the_file_type() {
    // mknod(2): EINVAL for a type that is no file type; EPERM for device nodes here (none can
    // be served); type 0 is a regular file.
    run(|g| {
        let mk = |g: &mut Guest, path: &str, mode: u32| {
            let p = g.cstr(path);
            g.syscall(Sysno::mknodat, &[AT_FDCWD as u64, p, u64::from(mode), 0])
        };
        assert_eq!(mk(g, "/tmp/bad", S_IFMT | 0o644), -EINVAL);
        assert_eq!(mk(g, "/tmp/dev", S_IFCHR | 0o644), -EPERM);
        assert_eq!(mk(g, "/tmp/plain", 0o644), 0);
        assert_eq!(g.stat("/tmp/plain").1.kind(), S_IFREG);
        assert_eq!(
            mk(
                g,
                "/tmp/fifo",
                wasmshim_linux_abi::consts::mode::S_IFIFO | 0o644
            ),
            0
        );
        assert_eq!(
            mk(
                g,
                "/tmp/fifo",
                wasmshim_linux_abi::consts::mode::S_IFIFO | 0o644
            ),
            -EEXIST
        );
    });
}

#[test]
fn pselect6_checks_descriptors_in_exceptfds() {
    // select(2): EBADF for a closed descriptor in any set; no exceptional conditions are
    // reported, so a valid one comes back cleared.
    run(|g| {
        let zero = g.put(&timespec(0));
        let eset = g.put(&(1u64 << 50).to_le_bytes());
        assert_eq!(
            g.syscall(Sysno::pselect6, &[51, 0, 0, eset, zero, 0]),
            -EBADF
        );
        let (_, rd, _wr) = g.pipe2(0);
        let eset = g.put(&(1u64 << rd).to_le_bytes());
        assert_eq!(
            g.syscall(Sysno::pselect6, &[rd as u64 + 1, 0, 0, eset, zero, 0]),
            0
        );
        assert_eq!(g.u64_at(eset), 0);
    });
}

#[test]
fn mq_timedreceive_with_a_huge_length_receives() {
    // mq_receive(3): EMSGSIZE only when msg_len is less than mq_msgsize (a size_t: 2^63 and
    // above are large, not negative).
    run(|g| {
        let name = g.cstr("big");
        let mq = g.syscall(Sysno::mq_open, &[name, O_CREAT | O_RDWR, 0o600, 0]);
        let msg = g.put(b"hi");
        assert_eq!(
            g.syscall(Sysno::mq_timedsend, &[mq as u64, msg, 2, 0, 0]),
            0
        );
        let buf = g.alloc(8192);
        assert_eq!(
            g.syscall(Sysno::mq_timedreceive, &[mq as u64, buf, u64::MAX, 0, 0]),
            2
        );
        assert_eq!(g.bytes(buf, 2usize), b"hi");
    });
}
