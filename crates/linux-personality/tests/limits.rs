//! `Config::limits`: at a limit the guest gets the error Linux gives there (`EAGAIN` for one
//! task too many, `EFBIG` for a file too large, `ENOSPC` for file data that does not fit), and
//! the machine goes on.

mod support;

use support::*;
use wasmshim_linux::{Config, ExitStatus, Limits};
use wasmshim_linux_abi::consts::open::*;
use wasmshim_linux_abi::consts::rlimit::RLIMIT_NPROC;
use wasmshim_linux_abi::consts::seek::SEEK_SET;

fn machine(limits: Limits) -> Machine {
    let base = Machine::default();
    Machine {
        config: Config {
            limits,
            ..base.config
        },
        ..base
    }
}

#[test]
fn written_files_stop_at_the_total_and_freeing_gives_room_back() {
    let limits = Limits {
        total_file_bytes: 8192,
        ..Limits::default()
    };
    let out = machine(limits).run(|g| {
        let a = g.open("/tmp/a", O_CREAT | O_RDWR);
        assert_eq!(g.write_all(a, &[1; 4096]), 4096);
        let b = g.open("/tmp/b", O_CREAT | O_RDWR);
        assert_eq!(g.write_all(b, &[2; 4096]), 4096);
        // Full. Growing either file is refused, by a write or by a truncate.
        assert_eq!(g.write_all(b, &[3]), -ENOSPC);
        assert_eq!(g.syscall(Sysno::ftruncate, &[a as u64, 4097]), -ENOSPC);
        // A refused write changed nothing.
        assert_eq!(g.fstat(b).1.size, 4096);
        // Writing over what is there is not growth.
        assert_eq!(g.lseek(a, 0, SEEK_SET), 0);
        assert_eq!(g.write_all(a, &[4; 4096]), 4096);
        // Shrinking a file gives room to another.
        assert_eq!(g.syscall(Sysno::ftruncate, &[a as u64, 1000]), 0);
        assert_eq!(g.write_all(b, &[5; 3096]), 3096);
        assert_eq!(g.write_all(b, &[6]), -ENOSPC);
        // So does removing one, once nothing has it open.
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/a", 0), 0);
        assert_eq!(g.write_all(b, &[7]), -ENOSPC, "still open: still held");
        assert_eq!(g.close(a), 0);
        assert_eq!(g.write_all(b, &[8; 1000]), 1000);
    });
    assert_eq!(out.status, ExitStatus::Code(0), "{:?}", out.diag);
    assert_eq!(out.kernel.vfs.written_bytes(), 8192);
}

#[test]
fn changing_an_image_file_counts_its_copy() {
    // `/etc/hostname` is 9 bytes in the image. Reading it holds nothing; the first write
    // copies it into memory.
    let limits = |total_file_bytes| Limits {
        total_file_bytes,
        ..Limits::default()
    };
    let out = machine(limits(8)).run(|g| {
        let fd = g.open("/etc/hostname", O_RDWR);
        assert_eq!(g.read_vec(fd, 64), b"testhost\n");
        assert_eq!(g.lseek(fd, 0, SEEK_SET), 0);
        assert_eq!(g.write_all(fd, b"T"), -ENOSPC);
        assert_eq!(g.lseek(fd, 0, SEEK_SET), 0);
        assert_eq!(g.read_vec(fd, 64), b"testhost\n", "the file is as it was");
    });
    assert_eq!(out.status, ExitStatus::Code(0), "{:?}", out.diag);
    assert_eq!(out.kernel.vfs.written_bytes(), 0);

    let out = machine(limits(9)).run(|g| {
        let fd = g.open("/etc/hostname", O_RDWR);
        assert_eq!(g.write_all(fd, b"T"), 1);
        assert_eq!(g.lseek(fd, 0, SEEK_SET), 0);
        assert_eq!(g.read_vec(fd, 64), b"Testhost\n");
    });
    assert_eq!(out.status, ExitStatus::Code(0), "{:?}", out.diag);
    assert_eq!(out.kernel.vfs.written_bytes(), 9);
}

#[test]
fn a_file_stops_at_the_file_limit() {
    let limits = Limits {
        file_bytes: 4096,
        ..Limits::default()
    };
    let out = machine(limits).run(|g| {
        let fd = g.open("/tmp/f", O_CREAT | O_RDWR);
        assert_eq!(g.write_all(fd, &[1; 4096]), 4096);
        assert_eq!(g.write_all(fd, &[2]), -EFBIG);
        assert_eq!(g.syscall(Sysno::ftruncate, &[fd as u64, 4097]), -EFBIG);
        let byte = g.put(&[3]);
        assert_eq!(
            g.syscall(Sysno::pwrite64, &[fd as u64, byte, 1, 4096]),
            -EFBIG
        );
        assert_eq!(g.syscall(Sysno::pwrite64, &[fd as u64, byte, 1, 4095]), 1);
        assert_eq!(g.fstat(fd).1.size, 4096);
    });
    assert_eq!(out.status, ExitStatus::Code(0), "{:?}", out.diag);
}

#[test]
fn clone_stops_at_the_task_limit() {
    let limits = Limits {
        tasks: 3,
        ..Limits::default()
    };
    let out = machine(limits).run(|g| {
        // This thread is the first task; two more fit.
        for _ in 0..2 {
            let tid = g.spawn_thread(|g| {
                g.nanosleep(1_000_000_000);
            });
            assert!(tid > 0, "clone: {tid}");
        }
        let refused = g.spawn_thread(|_| {});
        assert_eq!(refused, -EAGAIN);
        assert_eq!(g.fork(|_| {}), -EAGAIN, "a process is a task too");
        // getrlimit reports the limit that is enforced.
        let old = g.alloc(16);
        assert_eq!(g.syscall(Sysno::prlimit64, &[0, RLIMIT_NPROC, 0, old]), 0);
        assert_eq!(g.u64_at(old), 3);
        assert_eq!(g.u64_at(old + 8), 3);
    });
    assert_eq!(out.status, ExitStatus::Code(0), "{:?}", out.diag);
}
