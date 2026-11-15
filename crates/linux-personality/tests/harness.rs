//! The scripted guest itself: a failed assertion inside a guest program fails the test, a
//! guest deadlock is reported instead of hanging, and a run with threads, processes, sleeps
//! and entropy repeats byte for byte.

mod support;

use support::*;
use wasmshim_linux_abi::consts::clock::CLOCK_MONOTONIC;

#[test]
#[should_panic(expected = "the guest says no")]
fn a_guest_assertion_fails_the_test() {
    run(|g| {
        g.syscall(Sysno::getpid, &[]);
        panic!("the guest says no");
    });
}

#[test]
#[should_panic(expected = "a child's assertion")]
fn a_child_assertion_fails_the_test() {
    run(|g| {
        let child = g.fork(|_| panic!("a child's assertion"));
        g.wait4(child, 0);
    });
}

#[test]
#[should_panic(expected = "guest deadlock")]
fn a_guest_deadlock_is_reported() {
    run(|g| {
        // Nobody else holds the write end, and this thread waits on the read end forever.
        let (_, rd, _wr) = g.pipe2(0);
        g.read_vec(rd, 1);
    });
}

/// A program that exercises the scheduler: two processes and a thread, pipes, sleeps, entropy;
/// everything it observes goes to stdout.
fn busy(g: &mut Guest) {
    let (_, rd, wr) = g.pipe2(0);
    let child = g.fork(move |c| {
        for i in 0..3u8 {
            c.nanosleep(1_000 * u64::from(i + 1));
            c.write_all(wr, &[b'c', b'0' + i]);
        }
    });
    g.spawn_thread(move |t| {
        for i in 0..3u8 {
            t.nanosleep(1_500 * u64::from(i + 1));
            t.write_all(wr, &[b't', b'0' + i]);
        }
    });
    // (The thread shares this descriptor table, so `wr` stays open here.)
    let mut log = Vec::new();
    for _ in 0..6 {
        let got = g.read_vec(rd, 2);
        let now = g.clock_ns(CLOCK_MONOTONIC);
        log.extend_from_slice(&got);
        log.extend_from_slice(format!("@{now} ").as_bytes());
    }
    let buf = g.alloc(16);
    g.syscall(Sysno::getrandom, &[buf, 16, 0]);
    log.extend(g.bytes(buf, 16usize));
    g.write_all(1, &log);
    g.wait4(child, 0);
}

#[test]
fn a_run_repeats_byte_for_byte() {
    let a = run(busy);
    let b = run(busy);
    assert_eq!(a.stdout, b.stdout);
    assert_eq!(a.now, b.now);
    assert_eq!(a.kernel.retired(), b.kernel.retired());
    // The interleaving is the one virtual time dictates.
    let text = String::from_utf8_lossy(&a.stdout);
    assert!(
        text.starts_with("c0@1000001000 t0@1000001500 c1@1000003000 t1@1000004500 "),
        "{text}"
    );
}
