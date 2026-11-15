//! Directories and paths: `mkdirat`, `chdir`/`fchdir`, `getcwd`, relative lookups (from the
//! cwd and from a directory descriptor), `.` and `..` (also at the root), trailing slashes,
//! symlinks in the middle of a path, and symlink loops (`ELOOP`).

mod support;

use support::*;
use wasmshim_linux_abi::consts::at::{AT_REMOVEDIR, AT_SYMLINK_NOFOLLOW};
use wasmshim_linux_abi::consts::mode::{S_IFDIR, S_IFLNK};
use wasmshim_linux_abi::consts::open::*;

#[test]
fn mkdir_creates_with_the_umask_applied() {
    run(|g| {
        assert_eq!(g.mkdir("/tmp/a", 0o777), 0);
        let (r, st) = g.stat("/tmp/a");
        assert_eq!(r, 0);
        assert_eq!(st.mode, S_IFDIR | 0o755, "0777 minus the umask 022");
        assert_eq!(st.nlink, 2, "its own `.` and its name in /tmp");
        // A subdirectory's `..` adds a link to its parent.
        assert_eq!(g.mkdir("/tmp/a/b", 0o700), 0);
        assert_eq!(g.stat("/tmp/a").1.nlink, 3);
        // A trailing slash is fine for a directory.
        assert_eq!(g.mkdir("/tmp/c/", 0o755), 0);
        // umask(2) returns the previous mask.
        assert_eq!(g.syscall(Sysno::umask, &[0]), 0o022);
        assert_eq!(g.mkdir("/tmp/open", 0o777), 0);
        assert_eq!(g.stat("/tmp/open").1.mode, S_IFDIR | 0o777);
    });
}

#[test]
fn mkdir_errors() {
    run(|g| {
        // mkdir(2): EEXIST for any existing name, even a dangling symlink.
        assert_eq!(g.mkdir("/etc", 0o755), -EEXIST);
        assert_eq!(g.mkdir("/etc/hostname", 0o755), -EEXIST);
        assert_eq!(g.mkdir("/data/dangling", 0o755), -EEXIST);
        assert_eq!(g.mkdir("/", 0o755), -EEXIST);
        assert_eq!(g.mkdir("/tmp/x/y", 0o755), -ENOENT, "a missing parent");
        assert_eq!(g.mkdir("/etc/hostname/y", 0o755), -ENOTDIR);
        assert_eq!(
            g.syscall(Sysno::mkdirat, &[AT_FDCWD as u64, 8, 0o755]),
            -EFAULT
        );
    });
}

#[test]
fn mkdirat_and_openat_relative_to_a_directory_descriptor() {
    run(|g| {
        let dir = g.open("/tmp", O_RDONLY | O_DIRECTORY);
        let name = g.cstr("made");
        assert_eq!(g.syscall(Sysno::mkdirat, &[dir as u64, name, 0o755]), 0);
        assert_eq!(g.stat("/tmp/made").1.kind(), S_IFDIR);
        let fd = g.openat(dir as i32, "made/f", O_CREAT | O_WRONLY, 0o644);
        assert!(fd >= 0, "{fd}");
        assert_eq!(g.stat("/tmp/made/f").0, 0);
        // The descriptor keeps naming the directory after a chdir elsewhere.
        assert_eq!(g.chdir("/etc"), 0);
        assert!(g.openat(dir as i32, "made/f", O_RDONLY, 0) >= 0);
        assert_eq!(g.unlinkat(dir as i32, "made/f", 0), 0);
        assert_eq!(g.unlinkat(dir as i32, "made", AT_REMOVEDIR), 0);
    });
}

#[test]
fn getcwd_starts_at_the_root_and_follows_chdir() {
    run(|g| {
        let (r, cwd) = g.getcwd();
        assert_eq!(
            (r, cwd.as_slice()),
            (2, &b"/"[..]),
            "the length includes the NUL"
        );
        assert_eq!(g.chdir("/data/sub"), 0);
        let (r, cwd) = g.getcwd();
        assert_eq!((r, cwd.as_slice()), (10, &b"/data/sub"[..]));
        // Relative lookups start at the cwd.
        let fd = g.open("two", O_RDONLY);
        assert_eq!(g.read_vec(fd, 8), b"22");
        let fd = g.open("../big", O_RDONLY);
        assert_eq!(g.fstat(fd).1.size, 10_000);
        let fd = g.open("./one", O_RDONLY);
        assert_eq!(g.read_vec(fd, 8), b"1");
        // A relative chdir.
        assert_eq!(g.chdir(".."), 0);
        assert_eq!(g.getcwd().1, b"/data");
        // getcwd(2): ERANGE when the buffer is too small for the path and its NUL.
        let buf = g.alloc(8);
        assert_eq!(g.syscall(Sysno::getcwd, &[buf, 5]), -ERANGE);
        assert_eq!(g.syscall(Sysno::getcwd, &[buf, 6]), 6);
        assert_eq!(g.syscall(Sysno::getcwd, &[8, 64]), -EFAULT);
    });
}

#[test]
fn chdir_errors() {
    run(|g| {
        assert_eq!(g.chdir("/etc/hostname"), -ENOTDIR);
        assert_eq!(g.chdir("/nowhere"), -ENOENT);
        assert_eq!(g.chdir(""), -ENOENT);
        assert_eq!(g.syscall(Sysno::chdir, &[8]), -EFAULT);
        // A failed chdir leaves the cwd where it was.
        assert_eq!(g.getcwd().1, b"/");
        // fchdir(2): EBADF for a bad descriptor, ENOTDIR for a file.
        assert_eq!(g.syscall(Sysno::fchdir, &[99]), -EBADF);
        let file = g.open("/etc/hostname", O_RDONLY);
        assert_eq!(g.syscall(Sysno::fchdir, &[file as u64]), -ENOTDIR);
        let dir = g.open("/data", O_RDONLY | O_DIRECTORY);
        assert_eq!(g.syscall(Sysno::fchdir, &[dir as u64]), 0);
        assert_eq!(g.getcwd().1, b"/data");
    });
}

#[test]
fn chdir_through_a_symlink_lands_in_its_target() {
    run(|g| {
        assert_eq!(g.symlink("/data/sub", "/tmp/s"), 0);
        assert_eq!(g.chdir("/tmp/s"), 0);
        // The kernel's cwd is the directory itself, not the path used to reach it.
        assert_eq!(g.getcwd().1, b"/data/sub");
        assert_eq!(g.chdir(".."), 0);
        assert_eq!(g.getcwd().1, b"/data");
    });
}

#[test]
fn dot_dot_at_the_root_is_the_root() {
    // path_resolution(7): in the root directory, `..` is the root itself.
    run(|g| {
        let root = g.stat("/").1;
        assert_eq!(g.stat("/..").1.ino, root.ino);
        assert_eq!(g.stat("/../../..").1.ino, root.ino);
        let fd = g.open("/../../etc/hostname", O_RDONLY);
        assert_eq!(g.read_vec(fd, 64), b"testhost\n");
        assert_eq!(g.chdir("/"), 0);
        assert_eq!(g.chdir("../.."), 0);
        assert_eq!(g.getcwd().1, b"/");
        assert!(g.open("../etc/hostname", O_RDONLY) >= 0);
    });
}

#[test]
fn dots_and_slashes_in_paths() {
    run(|g| {
        let big = g.stat("/data/big").1;
        for path in [
            "/data/./big",
            "/data/sub/../big",
            "//data///big",
            "/data/sub/./../../data/big",
        ] {
            let (r, st) = g.stat(path);
            assert_eq!((r, st.ino), (0, big.ino), "{path}");
        }
        // A trailing slash demands a directory (path_resolution(7)).
        assert_eq!(g.stat("/data/sub/").1.kind(), S_IFDIR);
        assert_eq!(g.stat("/data/big/").0, -ENOTDIR);
        assert_eq!(g.open("/etc/hostname/", O_RDONLY), -ENOTDIR);
        // `..` of a file is ENOTDIR: the walk needs a directory there.
        assert_eq!(g.stat("/data/big/..").0, -ENOTDIR);
    });
}

#[test]
fn symlinks_in_the_middle_of_a_path_are_followed() {
    run(|g| {
        // Relative targets resolve from the link's own directory.
        assert_eq!(g.symlink("../data", "/tmp/up"), 0);
        assert_eq!(g.stat("/tmp/up/big").1.size, 10_000);
        // Absolute targets restart at the root.
        assert_eq!(g.symlink("/data/sub", "/tmp/abs"), 0);
        assert_eq!(g.stat("/tmp/abs/one").1.size, 1);
        // A symlink to a directory with a trailing slash is the directory.
        assert_eq!(g.stat("/tmp/abs/").1.kind(), S_IFDIR);
        let (r, link) = g.stat_at(AT_FDCWD, "/tmp/abs", AT_SYMLINK_NOFOLLOW);
        assert_eq!((r, link.kind()), (0, S_IFLNK));
    });
}

#[test]
fn symlink_loops_are_eloop() {
    run(|g| {
        // /loop/a -> b, /loop/b -> a (path_resolution(7), open(2): ELOOP).
        assert_eq!(g.open("/loop/a", O_RDONLY), -ELOOP);
        assert_eq!(g.stat("/loop/a").0, -ELOOP);
        assert_eq!(g.chdir("/loop/a"), -ELOOP);
        assert_eq!(g.stat("/loop/a/x").0, -ELOOP);
        // The link itself is fine to look at.
        let (r, st) = g.stat_at(AT_FDCWD, "/loop/a", AT_SYMLINK_NOFOLLOW);
        assert_eq!((r, st.kind()), (0, S_IFLNK));
        assert_eq!(g.readlink("/loop/a").1, b"b");
        // A self-loop.
        assert_eq!(g.symlink("self", "/tmp/self"), 0);
        assert_eq!(g.open("/tmp/self", O_RDONLY), -ELOOP);
    });
}

#[test]
fn forty_links_resolve_and_forty_one_do_not() {
    // path_resolution(7): Linux follows at most 40 symbolic links in one lookup.
    run(|g| {
        assert_eq!(g.symlink("/etc/hostname", "/tmp/l0"), 0);
        for i in 1..=40 {
            assert_eq!(g.symlink(&format!("l{}", i - 1), &format!("/tmp/l{i}")), 0);
        }
        // l39 takes 40 links to reach the file; l40 takes 41.
        assert!(g.open("/tmp/l39", O_RDONLY) >= 0);
        assert_eq!(g.open("/tmp/l40", O_RDONLY), -ELOOP);
    });
}

#[test]
fn rmdir_of_dot_and_dot_dot() {
    run(|g| {
        assert_eq!(g.mkdir("/tmp/d", 0o755), 0);
        // rmdir(2): EINVAL when the last component is `.`; ENOTEMPTY for `..`.
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/d/.", AT_REMOVEDIR), -EINVAL);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/d/..", AT_REMOVEDIR), -ENOTEMPTY);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/missing", AT_REMOVEDIR), -ENOENT);
        assert_eq!(g.unlinkat(AT_FDCWD, "/tmp/d", AT_REMOVEDIR), 0);
    });
}
