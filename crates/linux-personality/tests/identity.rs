//! Time and identity: `clock_gettime` reads the host's virtual clock, `nanosleep` advances it
//! by exactly what was asked, `getpid`/`gettid`/`getppid`, `uname`, seeded `getrandom`, an
//! unimplemented syscall number (`ENOSYS`), and how stdout reaches the host.

mod support;

use support::*;
use wasmshim_linux::ExitStatus;
use wasmshim_linux_abi::consts::clock::{CLOCK_MONOTONIC, CLOCK_REALTIME};

#[test]
fn the_first_program_is_pid_1_with_no_parent() {
    run(|g| {
        assert_eq!(g.syscall(Sysno::getpid, &[]), 1);
        assert_eq!(g.syscall(Sysno::gettid, &[]), 1);
        assert_eq!(g.syscall(Sysno::getppid, &[]), 0);
    });
}

#[test]
fn exit_status_reaches_the_embedder() {
    let out = Machine::default().run(|g| g.exit(42));
    assert_eq!(out.status, ExitStatus::Code(42));
    // Only the low 8 bits survive (exit(3)).
    let out = Machine::default().run(|g| g.exit(0x1_07));
    assert_eq!(out.status, ExitStatus::Code(7));
}

#[test]
fn stdout_and_stderr_go_to_the_host() {
    let out = run(|g| {
        assert_eq!(g.write_all(1, b"hello\n"), 6);
        assert_eq!(g.write_all(2, b"oops\n"), 5);
    });
    assert_eq!(out.stdout, b"hello\n");
    assert_eq!(out.stderr, b"oops\n");
}

#[test]
fn stdin_comes_from_the_host() {
    let m = Machine {
        stdin: b"typed".to_vec(),
        ..Machine::default()
    };
    let out = m.run(|g| {
        assert_eq!(g.read_vec(0, 64), b"typed");
        // End of input reads as 0 bytes.
        assert_eq!(g.read_vec(0, 64), b"");
    });
    assert_eq!(out.status, ExitStatus::Code(0));
}

#[test]
fn clock_gettime_reads_the_virtual_clock() {
    let out = run(|g| {
        assert_eq!(g.clock_ns(CLOCK_MONOTONIC), START_NS);
        assert_eq!(g.clock_ns(CLOCK_REALTIME), EPOCH_NS);
        // A bad clock id is EINVAL (clock_gettime(2)); a bad pointer is EFAULT.
        let ts = g.alloc(16);
        assert_eq!(g.syscall(Sysno::clock_gettime, &[99, ts]), -EINVAL);
        assert_eq!(
            g.syscall(Sysno::clock_gettime, &[CLOCK_MONOTONIC, 8]),
            -EFAULT
        );
    });
    // Nothing slept, and instructions cost no time on this machine.
    assert_eq!(out.now, START_NS);
}

#[test]
fn nanosleep_advances_time_by_exactly_the_request() {
    let out = run(|g| {
        let before = g.clock_ns(CLOCK_MONOTONIC);
        assert_eq!(g.nanosleep(1_500_000_000), 0);
        assert_eq!(g.clock_ns(CLOCK_MONOTONIC), before + 1_500_000_000);
        assert_eq!(g.nanosleep(7), 0);
        assert_eq!(g.clock_ns(CLOCK_MONOTONIC), before + 1_500_000_007);
        // The wall clock moves with it.
        assert_eq!(g.clock_ns(CLOCK_REALTIME), EPOCH_NS + 1_500_000_007);
        // tv_nsec out of range is EINVAL (nanosleep(2)).
        let bad = g.put(&{
            let mut b = [0u8; 16];
            b[8..].copy_from_slice(&1_000_000_000u64.to_le_bytes());
            b
        });
        assert_eq!(g.syscall(Sysno::nanosleep, &[bad, 0]), -EINVAL);
        assert_eq!(g.syscall(Sysno::nanosleep, &[8, 0]), -EFAULT);
    });
    assert_eq!(out.now, START_NS + 1_500_000_007);
}

#[test]
fn uname_reports_the_configured_machine() {
    run(|g| {
        let buf = g.alloc(390);
        assert_eq!(g.syscall(Sysno::uname, &[buf]), 0);
        let field = |raw: &[u8], i: usize| {
            let f = &raw[i * 65..(i + 1) * 65];
            let end = f.iter().position(|&b| b == 0).unwrap();
            String::from_utf8(f[..end].to_vec()).unwrap()
        };
        let raw = g.bytes(buf, 390usize);
        assert_eq!(field(&raw, 0), "Linux");
        assert_eq!(field(&raw, 1), "testhost");
        assert_eq!(field(&raw, 2), "6.6.0");
        assert_eq!(field(&raw, 4), "aarch64");
        assert_eq!(g.syscall(Sysno::uname, &[0]), -EFAULT);
    });
}

/// 32 bytes of `getrandom` from a machine seeded with `seed`.
fn random_bytes(seed: u64) -> Vec<u8> {
    let out = Machine {
        seed,
        ..Machine::default()
    }
    .run(|g| {
        let buf = g.alloc(32);
        assert_eq!(g.syscall(Sysno::getrandom, &[buf, 32, 0]), 32);
        let bytes = g.bytes(buf, 32usize);
        g.write_all(1, &bytes);
    });
    out.stdout
}

#[test]
fn getrandom_is_seeded() {
    let a = random_bytes(7);
    assert_eq!(a.len(), 32);
    assert_eq!(a, random_bytes(7), "the same seed gives the same bytes");
    assert_ne!(a, random_bytes(8), "another seed gives other bytes");
    run(|g| {
        // Unknown flags are EINVAL (getrandom(2)); a bad buffer is EFAULT.
        let buf = g.alloc(8);
        assert_eq!(g.syscall(Sysno::getrandom, &[buf, 8, 0x100]), -EINVAL);
        assert_eq!(g.syscall(Sysno::getrandom, &[8, 8, 0]), -EFAULT);
    });
}

#[test]
fn an_unimplemented_syscall_is_enosys() {
    run(|g| {
        // 244 is unassigned on arm64; 1000 is past the table; clone3 is not implemented.
        assert_eq!(g.syscall_nr(244, &[]), -ENOSYS);
        assert_eq!(g.syscall_nr(1000, &[1, 2, 3]), -ENOSYS);
        assert_eq!(g.syscall(Sysno::clone3, &[0, 0]), -ENOSYS);
        // The process carries on afterwards.
        assert_eq!(g.syscall(Sysno::getpid, &[]), 1);
    });
}
