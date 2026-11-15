//! Memory: anonymous `mmap`, `munmap`, `mprotect` (checked by touching the pages as user code
//! would), `MAP_FIXED`/`MAP_FIXED_NOREPLACE`, `brk`, file mappings (private and shared) read
//! back, and what `fork` copies or shares.

mod support;

use support::*;
use wasmshim_linux_abi::consts::map::*;
use wasmshim_linux_abi::consts::open::*;
use wasmshim_linux_abi::consts::prot::*;

const PAGE: u64 = 4096;

#[test]
fn anonymous_mappings_are_zeroed_and_private() {
    run(|g| {
        let a = g.mmap(
            0,
            3 * PAGE,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        );
        assert!(a > 0, "{a}");
        let a = a as u64;
        assert_eq!(a % PAGE, 0, "page-aligned");
        assert_eq!(g.bytes(a, 3 * PAGE as usize), vec![0; 3 * PAGE as usize]);
        g.write_mem(a + PAGE + 5, b"written");
        assert_eq!(g.bytes(a + PAGE + 5, 7usize), b"written");
        // A length that is not a page multiple is rounded up.
        let b = g.mmap(
            0,
            10,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        ) as u64;
        assert!(g.try_write_mem(b + PAGE - 1, b"x"));
        assert_ne!(a, b);
    });
}

#[test]
fn munmap_removes_pages() {
    run(|g| {
        let a = g.mmap(
            0,
            4 * PAGE,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        ) as u64;
        // Unmap the middle two pages: the ends stay.
        assert_eq!(g.syscall(Sysno::munmap, &[a + PAGE, 2 * PAGE]), 0);
        assert!(g.try_bytes(a, 1).is_some());
        assert!(g.try_bytes(a + PAGE, 1).is_none());
        assert!(g.try_bytes(a + 2 * PAGE, 1).is_none());
        assert!(g.try_bytes(a + 3 * PAGE, 1).is_some());
        // munmap(2): unmapping a range with nothing in it is not an error.
        assert_eq!(g.syscall(Sysno::munmap, &[a + PAGE, PAGE]), 0);
        // EINVAL for an unaligned address or a zero length.
        assert_eq!(g.syscall(Sysno::munmap, &[a + 1, PAGE]), -EINVAL);
        assert_eq!(g.syscall(Sysno::munmap, &[a, 0]), -EINVAL);
    });
}

#[test]
fn mmap_argument_errors() {
    run(|g| {
        let rw = PROT_READ | PROT_WRITE;
        // mmap(2), EINVAL: length 0; neither MAP_PRIVATE nor MAP_SHARED; an unaligned offset
        // or MAP_FIXED address.
        assert_eq!(
            g.mmap(0, 0, rw, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0),
            -EINVAL
        );
        assert_eq!(g.mmap(0, PAGE, rw, MAP_ANONYMOUS, -1, 0), -EINVAL);
        let fd = g.open("/data/big", O_RDONLY);
        assert_eq!(g.mmap(0, PAGE, PROT_READ, MAP_PRIVATE, fd, 100), -EINVAL);
        assert_eq!(
            g.mmap(
                0x1000_0001,
                PAGE,
                rw,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                -1,
                0
            ),
            -EINVAL
        );
        // EBADF: a file mapping without a valid descriptor.
        assert_eq!(g.mmap(0, PAGE, PROT_READ, MAP_PRIVATE, 99, 0), -EBADF);
        // EACCES: a write-only descriptor, or a shared writable mapping of a read-only one.
        let wo = g.open("/tmp/w", O_CREAT | O_WRONLY);
        assert_eq!(g.mmap(0, PAGE, PROT_READ, MAP_PRIVATE, wo, 0), -EACCES);
        assert_eq!(g.mmap(0, PAGE, rw, MAP_SHARED, fd, 0), -EACCES);
        // A private writable mapping of a read-only descriptor is fine (copy on write).
        assert!(g.mmap(0, PAGE, rw, MAP_PRIVATE, fd, 0) > 0);
    });
}

#[test]
fn map_fixed_replaces_and_noreplace_refuses() {
    run(|g| {
        let rw = PROT_READ | PROT_WRITE;
        let a = g.mmap(0, 2 * PAGE, rw, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) as u64;
        g.write_mem(a, b"old");
        // MAP_FIXED discards whatever was there (mmap(2)).
        let r = g.mmap(a, PAGE, rw, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        assert_eq!(r, a as i64);
        assert_eq!(g.bytes(a, 3usize), [0, 0, 0]);
        // MAP_FIXED_NOREPLACE: EEXIST over an existing mapping.
        let r = g.mmap(
            a,
            PAGE,
            rw,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
            -1,
            0,
        );
        assert_eq!(r, -EEXIST);
        // A free hint is taken as given.
        g.syscall(Sysno::munmap, &[a, 2 * PAGE]);
        let r = g.mmap(a + PAGE, PAGE, rw, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert_eq!(r, (a + PAGE) as i64);
    });
}

#[test]
fn mprotect_changes_what_user_code_may_do() {
    run(|g| {
        let a = g.mmap(
            0,
            2 * PAGE,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        ) as u64;
        g.write_mem(a, b"keep");
        assert_eq!(g.syscall(Sysno::mprotect, &[a, PAGE, PROT_READ]), 0);
        assert!(!g.try_write_mem(a, b"x"), "read-only now");
        assert_eq!(g.bytes(a, 4usize), b"keep");
        assert!(
            g.try_write_mem(a + PAGE, b"x"),
            "the next page is untouched"
        );
        assert_eq!(g.syscall(Sysno::mprotect, &[a, PAGE, PROT_NONE]), 0);
        assert!(g.try_bytes(a, 1).is_none(), "no access at all");
        assert_eq!(
            g.syscall(Sysno::mprotect, &[a, PAGE, PROT_READ | PROT_WRITE]),
            0
        );
        assert_eq!(
            g.bytes(a, 4usize),
            b"keep",
            "contents survive protection changes"
        );
        // A kernel copy into a read-only page is EFAULT, like a user store.
        g.syscall(Sysno::mprotect, &[a, PAGE, PROT_READ]);
        let fd = g.open("/etc/hostname", O_RDONLY);
        assert_eq!(g.read(fd, a, 4), -EFAULT);
        // mprotect(2): EINVAL for an unaligned address; ENOMEM for unmapped pages.
        assert_eq!(
            g.syscall(Sysno::mprotect, &[a + 1, PAGE, PROT_READ]),
            -EINVAL
        );
        g.syscall(Sysno::munmap, &[a, 2 * PAGE]);
        assert_eq!(g.syscall(Sysno::mprotect, &[a, PAGE, PROT_READ]), -ENOMEM);
    });
}

#[test]
fn brk_grows_and_shrinks_the_heap() {
    run(|g| {
        let start = g.syscall(Sysno::brk, &[0]) as u64;
        assert!(
            start > 0x40_0000,
            "the break starts after the program: {start:#x}"
        );
        assert!(
            g.try_bytes(start, 1).is_none(),
            "nothing is mapped at the break yet"
        );
        let top = start + 3 * PAGE;
        assert_eq!(g.syscall(Sysno::brk, &[top]) as u64, top);
        g.write_mem(start, b"heap");
        g.write_mem(top - 1, b"!");
        assert_eq!(g.bytes(start, 4usize), b"heap");
        // Shrinking unmaps the pages above the new break.
        assert_eq!(g.syscall(Sysno::brk, &[start + PAGE]) as u64, start + PAGE);
        assert!(g.try_bytes(start + 2 * PAGE, 1).is_none());
        assert_eq!(g.bytes(start, 4usize), b"heap");
        // A request below the start fails by returning the current break (brk(2), the raw
        // syscall).
        assert_eq!(g.syscall(Sysno::brk, &[0x1000]) as u64, start + PAGE);
    });
}

#[test]
fn a_private_file_mapping_reads_the_file() {
    run(|g| {
        let fd = g.open("/data/big", O_RDONLY);
        // /data/big is 10,000 bytes of i % 251. Map from the second page on.
        let m = g.mmap(0, 2 * PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, PAGE);
        assert!(m > 0, "{m}");
        let m = m as u64;
        let want: Vec<u8> = (4096..10_000u32).map(|i| (i % 251) as u8).collect();
        assert_eq!(g.bytes(m, want.len()), want);
        // Past the end of the file, to the end of its last page: zeros (mmap(2)).
        assert_eq!(
            g.bytes(m + want.len() as u64, 2 * PAGE as usize - want.len()),
            vec![0; 2 * PAGE as usize - want.len()]
        );
        // Private: writes stay in this mapping.
        g.write_mem(m, b"XYZ");
        let buf = g.alloc(3);
        assert_eq!(g.syscall(Sysno::pread64, &[fd as u64, buf, 3, PAGE]), 3);
        assert_eq!(
            g.bytes(buf, 3usize),
            [4096 % 251, 4097 % 251, 4098 % 251].map(|v| v as u8)
        );
        // The descriptor may be closed: the mapping stays.
        g.close(fd);
        assert_eq!(g.bytes(m, 3usize), b"XYZ");
    });
}

#[test]
fn a_shared_file_mapping_and_the_file_are_the_same_bytes() {
    run(|g| {
        let fd = g.open("/tmp/shared", O_CREAT | O_RDWR);
        g.write_all(fd, &[b'.'; 100]);
        let m = g.mmap(0, PAGE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        assert!(m > 0, "{m}");
        let m = m as u64;
        // A store through the mapping is visible to read(2)...
        g.write_mem(m + 10, b"mapped");
        let buf = g.alloc(6);
        assert_eq!(g.syscall(Sysno::pread64, &[fd as u64, buf, 6, 10]), 6);
        assert_eq!(g.bytes(buf, 6usize), b"mapped");
        // ...and a write(2) is visible through the mapping.
        let data = g.put(b"written");
        assert_eq!(g.syscall(Sysno::pwrite64, &[fd as u64, data, 7, 50]), 7);
        assert_eq!(g.bytes(m + 50, 7usize), b"written");
    });
}

#[test]
fn fork_copies_private_memory_and_shares_shared_memory() {
    run(|g| {
        let rw = PROT_READ | PROT_WRITE;
        let private = g.mmap(0, PAGE, rw, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) as u64;
        let shared = g.mmap(0, PAGE, rw, MAP_SHARED | MAP_ANONYMOUS, -1, 0) as u64;
        g.write_mem(private, b"parent");
        let child = g.fork(move |c| {
            assert_eq!(
                c.bytes(private, 6usize),
                b"parent",
                "the child starts with a copy"
            );
            c.write_mem(private, b"child!");
            c.write_mem(shared, b"hello");
        });
        let (r, status) = g.wait4(child, 0);
        assert_eq!((r, status), (child, 0));
        assert_eq!(
            g.bytes(private, 6usize),
            b"parent",
            "the copy was the child's own"
        );
        assert_eq!(g.bytes(shared, 5usize), b"hello", "MAP_SHARED is shared");
    });
}

#[test]
fn threads_share_the_address_space() {
    run(|g| {
        let cell = g.alloc(8);
        g.spawn_thread(move |t| {
            t.write_mem(cell, b"thread");
        });
        // Wait for the thread to have run: it finishes within its first slice.
        g.nanosleep(1);
        assert_eq!(g.bytes(cell, 6usize), b"thread");
    });
}
