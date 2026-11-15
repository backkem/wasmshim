//! Files: `openat` and its flags (`O_CREAT`, `O_EXCL`, `O_TRUNC`, `O_APPEND`, `O_DIRECTORY`,
//! `O_NOFOLLOW`), `read`/`write`/`pread64`/`lseek`, `fstat`/`newfstatat`, `ftruncate`,
//! `unlinkat`, `renameat`, `symlinkat`/`readlinkat`, `getdents64`, `dup`/`dup3`/`fcntl`, and the
//! errors: bad descriptors (`EBADF`), bad pointers (`EFAULT`), `ENOENT`, `ENOTDIR`, `EEXIST`,
//! `EISDIR`.

mod support;

use support::*;
use wasmshim_linux_abi::consts::at::{AT_EMPTY_PATH, AT_REMOVEDIR, AT_SYMLINK_NOFOLLOW};
use wasmshim_linux_abi::consts::dtype::{DT_DIR, DT_LNK, DT_REG};
use wasmshim_linux_abi::consts::fcntl::*;
use wasmshim_linux_abi::consts::mode::{S_IFDIR, S_IFLNK, S_IFREG};
use wasmshim_linux_abi::consts::open::*;
use wasmshim_linux_abi::consts::seek::{SEEK_CUR, SEEK_END, SEEK_SET};

#[test]
fn open_read_and_close_a_file_from_the_image() {
    run(|g| {
        let fd = g.open("/etc/hostname", O_RDONLY);
        assert_eq!(fd, 3, "the lowest free descriptor");
        assert_eq!(g.read_vec(fd, 64), b"testhost\n");
        assert_eq!(g.read_vec(fd, 64), b"", "end of file reads 0 bytes");
        assert_eq!(g.close(fd), 0);
        assert_eq!(g.close(fd), -EBADF, "a closed descriptor");
    });
}

#[test]
fn open_errors() {
    run(|g| {
        assert_eq!(g.open("/etc/missing", O_RDONLY), -ENOENT);
        assert_eq!(g.open("/missing/file", O_CREAT | O_WRONLY), -ENOENT);
        // A path component that is not a directory (open(2): ENOTDIR).
        assert_eq!(g.open("/etc/hostname/x", O_RDONLY), -ENOTDIR);
        // O_DIRECTORY on something else.
        assert_eq!(g.open("/etc/hostname", O_RDONLY | O_DIRECTORY), -ENOTDIR);
        // A directory may be opened only for reading (open(2): EISDIR).
        assert_eq!(g.open("/etc", O_WRONLY), -EISDIR);
        assert_eq!(g.open("/etc", O_RDWR), -EISDIR);
        let dir = g.open("/etc", O_RDONLY | O_DIRECTORY);
        assert!(dir >= 0);
        let buf = g.alloc(8);
        assert_eq!(g.read(dir, buf, 8), -EISDIR, "read(2) of a directory");
        assert_eq!(g.open("", O_RDONLY), -ENOENT, "an empty path");
        // A path pointer the process cannot read.
        assert_eq!(
            g.syscall(Sysno::openat, &[AT_FDCWD as u64, 8, O_RDONLY, 0]),
            -EFAULT
        );
        let long = "x".repeat(300);
        assert_eq!(g.open(&format!("/tmp/{long}"), O_RDONLY), -ENAMETOOLONG);
    });
}

#[test]
fn create_exclusive_and_truncate() {
    run(|g| {
        let fd = g.openat(AT_FDCWD, "/tmp/f", O_CREAT | O_EXCL | O_WRONLY, 0o600);
        assert!(fd >= 0, "{fd}");
        assert_eq!(g.write_all(fd, b"hello world"), 11);
        g.close(fd);
        // The file exists now.
        assert_eq!(g.open("/tmp/f", O_CREAT | O_EXCL | O_WRONLY), -EEXIST);
        // Plain O_CREAT opens the existing file and keeps its contents.
        let fd = g.open("/tmp/f", O_CREAT | O_RDWR);
        assert_eq!(g.read_vec(fd, 64), b"hello world");
        g.close(fd);
        // Mode bits: the requested mode minus the umask (022).
        let (r, st) = g.stat("/tmp/f");
        assert_eq!(r, 0);
        assert_eq!(st.mode, S_IFREG | 0o600);
        // O_TRUNC empties it.
        let fd = g.open("/tmp/f", O_TRUNC | O_WRONLY);
        assert_eq!(g.fstat(fd).1.size, 0);
        g.close(fd);
        let fd = g.openat(AT_FDCWD, "/tmp/g", O_CREAT | O_WRONLY, 0o777);
        assert_eq!(g.fstat(fd).1.mode, S_IFREG | 0o755);
    });
}

#[test]
fn o_creat_through_a_dangling_symlink_creates_its_target() {
    // Linux follows a final symlink for O_CREAT (without O_EXCL or O_NOFOLLOW): a dangling link
    // gets its target created (path_openat's trailing-symlink walk; open(2), O_CREAT).
    run(|g| {
        let fd = g.open("/data/dangling", O_CREAT | O_WRONLY);
        assert!(fd >= 0, "{fd}");
        assert_eq!(g.write_all(fd, b"made"), 4);
        g.close(fd);
        let fd = g.open("/data/nowhere", O_RDONLY);
        assert_eq!(g.read_vec(fd, 16), b"made");
    });
}

#[test]
fn o_excl_does_not_follow_a_symlink() {
    // open(2), O_EXCL: with O_CREAT, symbolic links are not followed; if pathname is a symlink,
    // open fails with EEXIST wherever the link points.
    run(|g| {
        assert_eq!(g.symlink("/tmp/elsewhere", "/tmp/l"), 0);
        assert_eq!(g.open("/tmp/l", O_CREAT | O_EXCL | O_WRONLY), -EEXIST);
        assert_eq!(g.stat("/tmp/elsewhere").0, -ENOENT, "nothing was created");
    });
}

#[test]
fn o_nofollow_refuses_a_final_symlink() {
    run(|g| {
        // open(2), ELOOP: "O_NOFOLLOW was specified but pathname refers to a symbolic link".
        assert_eq!(g.open("/data/link", O_RDONLY | O_NOFOLLOW), -ELOOP);
        // Symlinks in the middle of the path are still followed.
        assert_eq!(g.symlink("/data", "/tmp/d"), 0);
        let fd = g.open("/tmp/d/big", O_RDONLY | O_NOFOLLOW);
        assert!(fd >= 0, "{fd}");
    });
}

#[test]
fn append_writes_at_the_end_whatever_the_offset() {
    run(|g| {
        let fd = g.open("/tmp/log", O_CREAT | O_RDWR);
        g.write_all(fd, b"abc");
        g.close(fd);
        let fd = g.open("/tmp/log", O_WRONLY | O_APPEND);
        assert_eq!(g.lseek(fd, 0, SEEK_SET), 0);
        assert_eq!(g.write_all(fd, b"def"), 3);
        assert_eq!(g.lseek(fd, 0, SEEK_CUR), 6, "the offset follows the append");
        g.close(fd);
        let fd = g.open("/tmp/log", O_RDONLY);
        assert_eq!(g.read_vec(fd, 64), b"abcdef");
    });
}

#[test]
fn read_write_and_seek() {
    run(|g| {
        let fd = g.open("/tmp/rw", O_CREAT | O_RDWR);
        assert_eq!(g.write_all(fd, b"0123456789"), 10);
        assert_eq!(g.lseek(fd, 2, SEEK_SET), 2);
        assert_eq!(g.read_vec(fd, 3), b"234");
        assert_eq!(g.lseek(fd, -1, SEEK_CUR), 4);
        assert_eq!(g.lseek(fd, -2, SEEK_END), 8);
        assert_eq!(g.read_vec(fd, 10), b"89");
        // A negative resulting offset is EINVAL; a bad whence too (lseek(2)).
        assert_eq!(g.lseek(fd, -100, SEEK_SET), -EINVAL);
        assert_eq!(g.lseek(fd, 0, 17), -EINVAL);
        // Seeking past the end and writing leaves a hole that reads as zeros.
        assert_eq!(g.lseek(fd, 12, SEEK_SET), 12);
        g.write_all(fd, b"!");
        assert_eq!(g.lseek(fd, 8, SEEK_SET), 8);
        assert_eq!(g.read_vec(fd, 10), b"89\0\0!");
        // Zero-length transfers succeed and do nothing.
        let buf = g.alloc(1);
        assert_eq!(g.read(fd, buf, 0), 0);
        assert_eq!(g.write(fd, buf, 0), 0);
    });
}

#[test]
fn pread_does_not_move_the_offset() {
    run(|g| {
        let fd = g.open("/data/big", O_RDONLY);
        let buf = g.alloc(16);
        assert_eq!(g.syscall(Sysno::pread64, &[fd as u64, buf, 4, 300]), 4);
        // /data/big holds i % 251 at offset i.
        assert_eq!(g.bytes(buf, 4usize), [49, 50, 51, 52]);
        assert_eq!(g.lseek(fd, 0, SEEK_CUR), 0);
        // At or past the end: 0 bytes.
        assert_eq!(g.syscall(Sysno::pread64, &[fd as u64, buf, 4, 10_000]), 0);
        let fd = g.open("/tmp/pw", O_CREAT | O_RDWR);
        let data = g.put(b"xy");
        assert_eq!(g.syscall(Sysno::pwrite64, &[fd as u64, data, 2, 3]), 2);
        assert_eq!(g.lseek(fd, 0, SEEK_CUR), 0);
        assert_eq!(g.read_vec(fd, 16), b"\0\0\0xy");
    });
}

#[test]
fn pread_with_a_negative_offset_is_einval() {
    // pread(2): EINVAL when offset is negative (ksys_pread64 checks `pos < 0`).
    run(|g| {
        let fd = g.open("/data/big", O_RDONLY);
        let buf = g.alloc(16);
        let neg = (-1i64) as u64;
        assert_eq!(
            g.syscall(Sysno::pread64, &[fd as u64, buf, 4, neg]),
            -EINVAL
        );
        let w = g.open("/tmp/pw", O_CREAT | O_RDWR);
        assert_eq!(
            g.syscall(Sysno::pwrite64, &[w as u64, buf, 4, neg]),
            -EINVAL
        );
    });
}

#[test]
fn access_modes_are_enforced_per_descriptor() {
    run(|g| {
        let buf = g.put(b"data");
        let ro = g.open("/tmp/f", O_CREAT | O_RDONLY);
        assert_eq!(
            g.write(ro, buf, 4),
            -EBADF,
            "write(2) on a read-only descriptor"
        );
        let wo = g.open("/tmp/f", O_WRONLY);
        assert_eq!(
            g.read(wo, buf, 4),
            -EBADF,
            "read(2) on a write-only descriptor"
        );
        assert_eq!(g.write(wo, buf, 4), 4);
    });
}

#[test]
fn bad_descriptors_are_ebadf() {
    run(|g| {
        let buf = g.alloc(16);
        for fd in [-1i64, 99, 1 << 20] {
            assert_eq!(g.read(fd, buf, 1), -EBADF);
            assert_eq!(g.write(fd, buf, 1), -EBADF);
            assert_eq!(g.close(fd), -EBADF);
            assert_eq!(g.lseek(fd, 0, SEEK_SET), -EBADF);
            assert_eq!(g.fstat(fd).0, -EBADF);
            assert_eq!(g.syscall(Sysno::dup, &[fd as u64]), -EBADF);
            assert_eq!(g.syscall(Sysno::fcntl, &[fd as u64, F_GETFD]), -EBADF);
            assert_eq!(g.syscall(Sysno::getdents64, &[fd as u64, buf, 16]), -EBADF);
        }
        // A relative path with a bad dirfd is EBADF.
        assert_eq!(g.openat(99, "hostname", O_RDONLY, 0), -EBADF);
        // A dirfd that is not a directory: ENOTDIR.
        let file = g.open("/etc/hostname", O_RDONLY);
        assert_eq!(g.openat(file as i32, "x", O_RDONLY, 0), -ENOTDIR);
    });
}

#[test]
fn an_absolute_path_ignores_dirfd() {
    // openat(2): "If pathname is absolute, then dirfd is ignored."
    run(|g| {
        assert!(g.openat(99, "/etc/hostname", O_RDONLY, 0) >= 0);
        let file = g.open("/etc/hostname", O_RDONLY);
        assert!(g.openat(file as i32, "/etc/hostname", O_RDONLY, 0) >= 0);
    });
}

#[test]
fn bad_buffers_are_efault() {
    run(|g| {
        let fd = g.open("/etc/hostname", O_RDONLY);
        // Address 8 is never mapped.
        assert_eq!(g.read(fd, 8, 4), -EFAULT);
        let out = g.open("/tmp/f", O_CREAT | O_WRONLY);
        assert_eq!(g.write(out, 8, 4), -EFAULT);
        assert_eq!(g.syscall(Sysno::fstat, &[fd as u64, 8]), -EFAULT);
        let p = g.cstr("/etc/hostname");
        assert_eq!(
            g.syscall(Sysno::newfstatat, &[AT_FDCWD as u64, p, 8, 0]),
            -EFAULT
        );
    });
}

#[test]
fn fstat_and_newfstatat() {
    run(|g| {
        let fd = g.open("/data/big", O_RDONLY);
        let (r, st) = g.fstat(fd);
        assert_eq!(r, 0);
        assert_eq!(st.kind(), S_IFREG);
        assert_eq!(st.mode & 0o7777, 0o644);
        assert_eq!(st.size, 10_000);
        assert_eq!(st.nlink, 1);
        // newfstatat follows a final symlink unless AT_SYMLINK_NOFOLLOW.
        let (r, via_link) = g.stat("/data/link");
        assert_eq!(r, 0);
        assert_eq!(via_link.ino, st.ino);
        let (r, link) = g.stat_at(AT_FDCWD, "/data/link", AT_SYMLINK_NOFOLLOW);
        assert_eq!(r, 0);
        assert_eq!(link.kind(), S_IFLNK);
        assert_eq!(link.size, 3, "a symlink's size is its target's length");
        // AT_EMPTY_PATH with "" stats the descriptor itself.
        let (r, same) = g.stat_at(fd as i32, "", AT_EMPTY_PATH);
        assert_eq!((r, same.ino), (0, st.ino));
        // Without AT_EMPTY_PATH an empty path is ENOENT (stat(2)).
        assert_eq!(g.stat_at(fd as i32, "", 0).0, -ENOENT);
        // A path relative to a directory descriptor.
        let dir = g.open("/data", O_RDONLY | O_DIRECTORY);
        let (r, rel) = g.stat_at(dir as i32, "big", 0);
        assert_eq!((r, rel.ino), (0, st.ino));
        let (r, d) = g.stat("/data/sub");
        assert_eq!(r, 0);
        assert_eq!(d.kind(), S_IFDIR);
        assert_eq!(g.stat("/data/missing").0, -ENOENT);
        assert_eq!(g.stat("/data/big/x").0, -ENOTDIR);
    });
}

#[test]
fn ftruncate_grows_with_zeros_and_shrinks() {
    run(|g| {
        let fd = g.open("/tmp/t", O_CREAT | O_RDWR);
        g.write_all(fd, b"abcdef");
        assert_eq!(g.syscall(Sysno::ftruncate, &[fd as u64, 3]), 0);
        assert_eq!(g.fstat(fd).1.size, 3);
        assert_eq!(g.syscall(Sysno::ftruncate, &[fd as u64, 5]), 0);
        // The offset is unchanged (6), past the new end of 5: reads give nothing.
        assert_eq!(g.read_vec(fd, 8), b"");
        g.lseek(fd, 0, SEEK_SET);
        assert_eq!(g.read_vec(fd, 8), b"abc\0\0");
        // ftruncate(2): EINVAL for a negative length, or a descriptor not open for writing.
        assert_eq!(
            g.syscall(Sysno::ftruncate, &[fd as u64, (-1i64) as u64]),
            -EINVAL
        );
        let ro = g.open("/tmp/t", O_RDONLY);
        assert_eq!(g.syscall(Sysno::ftruncate, &[ro as u64, 0]), -EINVAL);
        assert_eq!(g.syscall(Sysno::ftruncate, &[99, 0]), -EBADF);
    });
}

#[test]
fn unlink_files_and_remove_directories() {
    run(|g| {
        let fd = g.open("/tmp/u", O_CREAT | O_RDWR);
        g.write_all(fd, b"still here");
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/u", 0), 0);
        assert_eq!(g.open("/tmp/u", O_RDONLY), -ENOENT);
        // An open descriptor keeps an unlinked file alive (unlink(2)).
        let (r, st) = g.fstat(fd);
        assert_eq!((r, st.nlink), (0, 0));
        g.lseek(fd, 0, SEEK_SET);
        assert_eq!(g.read_vec(fd, 32), b"still here");
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/u", 0), -ENOENT);

        assert_eq!(g.mkdir("/tmp/d", 0o755), 0);
        // unlink(2) of a directory is EISDIR on Linux.
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/d", 0), -EISDIR);
        let f = g.open("/tmp/d/f", O_CREAT | O_WRONLY);
        g.close(f);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/d", AT_REMOVEDIR), -ENOTEMPTY);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/d/f", AT_REMOVEDIR), -ENOTDIR);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/d/f", 0), 0);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/d", AT_REMOVEDIR), 0);
        assert_eq!(g.stat("/tmp/d").0, -ENOENT);
        // unlink of a symlink removes the link, not its target.
        assert_eq!(g.symlink("/etc/hostname", "/tmp/s"), 0);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/s", 0), 0);
        assert_eq!(g.stat("/etc/hostname").0, 0);
    });
}

#[test]
fn unlinkat_rejects_unknown_flags() {
    // unlinkat(2): EINVAL when flags has bits other than AT_REMOVEDIR.
    run(|g| {
        let f = g.open("/tmp/x", O_CREAT | O_WRONLY);
        g.close(f);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/x", 0x1), -EINVAL);
        assert_eq!(g.stat("/tmp/x").0, 0, "nothing was removed");
    });
}

#[test]
fn rename_files_and_directories() {
    run(|g| {
        let fd = g.open("/tmp/a", O_CREAT | O_WRONLY);
        g.write_all(fd, b"A");
        g.close(fd);
        let fd = g.open("/tmp/b", O_CREAT | O_WRONLY);
        g.write_all(fd, b"B");
        g.close(fd);
        // Renaming over an existing file replaces it atomically.
        assert_eq!(g.rename("/tmp/a", "/tmp/b"), 0);
        assert_eq!(g.open("/tmp/a", O_RDONLY), -ENOENT);
        let fd = g.open("/tmp/b", O_RDONLY);
        assert_eq!(g.read_vec(fd, 8), b"A");
        assert_eq!(g.rename("/tmp/missing", "/tmp/c"), -ENOENT);
        // Directories (rename(2)).
        assert_eq!(g.mkdir("/tmp/d1", 0o755), 0);
        assert_eq!(g.mkdir("/tmp/d2", 0o755), 0);
        let f = g.open("/tmp/d2/inside", O_CREAT | O_WRONLY);
        g.close(f);
        assert_eq!(g.rename("/tmp/d1", "/tmp/d2"), -ENOTEMPTY);
        assert_eq!(g.rename("/tmp/b", "/tmp/d1"), -EISDIR);
        assert_eq!(g.rename("/tmp/d1", "/tmp/b"), -ENOTDIR);
        // A directory cannot move into itself.
        assert_eq!(g.rename("/tmp/d2", "/tmp/d2/sub"), -EINVAL);
        assert_eq!(g.rename("/tmp/d2", "/tmp/d3"), 0);
        assert_eq!(g.stat("/tmp/d3/inside").0, 0);
        // Renaming a file onto itself does nothing and succeeds.
        assert_eq!(g.rename("/tmp/b", "/tmp/b"), 0);
        assert_eq!(g.stat("/tmp/b").0, 0);
    });
}

#[test]
fn symlinks_and_readlink() {
    run(|g| {
        assert_eq!(g.symlink("hostname", "/etc/alias"), 0);
        let (r, target) = g.readlink("/etc/alias");
        assert_eq!(r, 8, "the length, with no terminating NUL (readlink(2))");
        assert_eq!(target, b"hostname");
        let fd = g.open("/etc/alias", O_RDONLY);
        assert_eq!(g.read_vec(fd, 64), b"testhost\n");
        assert_eq!(g.symlink("x", "/etc/alias"), -EEXIST);
        // readlink of something that is not a link: EINVAL; of nothing: ENOENT.
        assert_eq!(g.readlink("/etc/hostname").0, -EINVAL);
        assert_eq!(g.readlink("/etc/nothing").0, -ENOENT);
        // A short buffer truncates silently.
        let p = g.cstr("/etc/alias");
        let buf = g.alloc(4);
        assert_eq!(
            g.syscall(Sysno::readlinkat, &[AT_FDCWD as u64, p, buf, 4]),
            4
        );
        assert_eq!(g.bytes(buf, 4usize), b"host");
        // bufsiz 0 is EINVAL.
        assert_eq!(
            g.syscall(Sysno::readlinkat, &[AT_FDCWD as u64, p, buf, 0]),
            -EINVAL
        );
        // A dangling link reads back, but opening it is ENOENT.
        assert_eq!(g.readlink("/data/dangling").1, b"nowhere");
        assert_eq!(g.open("/data/dangling", O_RDONLY), -ENOENT);
    });
}

#[test]
fn getdents64_lists_a_directory() {
    run(|g| {
        let fd = g.open("/data", O_RDONLY | O_DIRECTORY);
        let mut entries = g.list_dir(fd);
        entries.sort();
        assert_eq!(
            entries,
            [
                (".".to_owned(), DT_DIR),
                ("..".to_owned(), DT_DIR),
                ("big".to_owned(), DT_REG),
                ("dangling".to_owned(), DT_LNK),
                ("link".to_owned(), DT_LNK),
                ("sub".to_owned(), DT_DIR),
            ]
        );
        // Listing again after the end gives nothing until the offset is rewound.
        assert_eq!(g.list_dir(fd), []);
        assert_eq!(g.lseek(fd, 0, SEEK_SET), 0);
        assert_eq!(g.list_dir(fd).len(), 6);
        // A buffer too small for one entry is EINVAL (getdents(2)).
        g.lseek(fd, 0, SEEK_SET);
        let buf = g.alloc(64);
        assert_eq!(g.syscall(Sysno::getdents64, &[fd as u64, buf, 4]), -EINVAL);
        // A file is not a directory.
        let file = g.open("/etc/hostname", O_RDONLY);
        assert_eq!(
            g.syscall(Sysno::getdents64, &[file as u64, buf, 64]),
            -ENOTDIR
        );
        assert_eq!(g.syscall(Sysno::getdents64, &[fd as u64, 8, 64]), -EFAULT);
    });
}

#[test]
fn getdents64_in_small_steps_sees_every_entry_once() {
    run(|g| {
        let dir = g.open("/tmp", O_RDONLY | O_DIRECTORY);
        for i in 0..20 {
            let f = g.open(&format!("/tmp/file{i:02}"), O_CREAT | O_WRONLY);
            g.close(f);
        }
        // A 64-byte buffer holds one or two records at a time.
        let buf = g.alloc(64);
        let mut names = Vec::new();
        loop {
            let n = g.syscall(Sysno::getdents64, &[dir as u64, buf, 64]);
            assert!(n >= 0, "{n}");
            if n == 0 {
                break;
            }
            names.extend(parse_dirents(&g.bytes(buf, n)).into_iter().map(|e| e.0));
        }
        names.sort();
        let mut want: Vec<String> = (0..20).map(|i| format!("file{i:02}")).collect();
        want.extend([".".to_owned(), "..".to_owned()]);
        want.sort();
        assert_eq!(names, want);
    });
}

#[test]
fn dup_dup3_and_fcntl() {
    run(|g| {
        let fd = g.open("/data/big", O_RDONLY);
        let d = g.syscall(Sysno::dup, &[fd as u64]);
        assert_eq!(d, fd + 1, "the lowest free descriptor");
        // Duplicates share the file offset (dup(2)).
        g.read_vec(fd, 10);
        assert_eq!(g.lseek(d, 0, SEEK_CUR), 10);
        // dup3 onto a chosen number, with O_CLOEXEC.
        assert_eq!(g.syscall(Sysno::dup3, &[fd as u64, 10, O_CLOEXEC]), 10);
        assert_eq!(g.syscall(Sysno::fcntl, &[10, F_GETFD]), FD_CLOEXEC as i64);
        assert_eq!(g.syscall(Sysno::fcntl, &[d as u64, F_GETFD]), 0);
        // dup3 with old == new, or unknown flags, is EINVAL (dup(2)).
        assert_eq!(g.syscall(Sysno::dup3, &[fd as u64, fd as u64, 0]), -EINVAL);
        assert_eq!(g.syscall(Sysno::dup3, &[fd as u64, 11, 0x1]), -EINVAL);
        // dup3 onto an open descriptor closes it first.
        let other = g.open("/etc/hostname", O_RDONLY);
        assert_eq!(g.syscall(Sysno::dup3, &[fd as u64, other as u64, 0]), other);
        assert_eq!(g.lseek(other, 0, SEEK_CUR), 10);
        // F_SETFD / F_GETFD.
        assert_eq!(g.syscall(Sysno::fcntl, &[d as u64, F_SETFD, FD_CLOEXEC]), 0);
        assert_eq!(
            g.syscall(Sysno::fcntl, &[d as u64, F_GETFD]),
            FD_CLOEXEC as i64
        );
        // F_DUPFD takes the lowest free number at or above the argument.
        assert_eq!(g.syscall(Sysno::fcntl, &[fd as u64, F_DUPFD, 20]), 20);
        assert_eq!(g.syscall(Sysno::fcntl, &[fd as u64, F_DUPFD, 20]), 21);
        assert_eq!(
            g.syscall(Sysno::fcntl, &[fd as u64, F_DUPFD_CLOEXEC, 30]),
            30
        );
        assert_eq!(g.syscall(Sysno::fcntl, &[30, F_GETFD]), FD_CLOEXEC as i64);
        // F_GETFL reports the access mode and status flags; F_SETFL changes O_APPEND.
        let w = g.open("/tmp/fl", O_CREAT | O_WRONLY);
        let fl = g.syscall(Sysno::fcntl, &[w as u64, F_GETFL]);
        assert_eq!(fl as u64 & O_ACCMODE, O_WRONLY);
        assert_eq!(fl as u64 & O_APPEND, 0);
        assert_eq!(g.syscall(Sysno::fcntl, &[w as u64, F_SETFL, O_APPEND]), 0);
        let fl = g.syscall(Sysno::fcntl, &[w as u64, F_GETFL]);
        assert_eq!(fl as u64 & O_APPEND, O_APPEND);
        assert_eq!(
            fl as u64 & O_ACCMODE,
            O_WRONLY,
            "F_SETFL keeps the access mode"
        );
        // An unknown command is EINVAL (fcntl(2)).
        assert_eq!(g.syscall(Sysno::fcntl, &[fd as u64, 0xdead]), -EINVAL);
    });
}
