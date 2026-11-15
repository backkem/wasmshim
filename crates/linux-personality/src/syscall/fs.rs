//! File and filesystem syscalls.

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{access, at, fcntl, ioctl, mode, open, rwf, seek};
use wasmshim_linux_abi::structs::{Dirent64, Iovec, Stat, Statx, Timespec};

use super::SysResult;
use crate::fd::{FileKind, FileRef, OpenFile, Stdio, open_file};
use crate::host::Host;
use crate::kernel::{Kernel, Tid};
use crate::objects::ObjectWait;
use crate::uaccess::{MAX_IO, io_len, read_array, read_bytes, read_path, write_bytes, write_u64};
use crate::vfs::{Device, Ino, Node, ROOT};

const UTIME_NOW: i64 = (1 << 30) - 1;
const UTIME_OMIT: i64 = (1 << 30) - 2;
const RENAME_NOREPLACE: u64 = 1;
const TMPFS_MAGIC: u64 = 0x0102_1994;
const POLLIN: u16 = 0x1;
const POLLOUT: u16 = 0x4;
const POLLERR: u16 = 0x8;
const POLLHUP: u16 = 0x10;
const POLLNVAL: u16 = 0x20;
const MAX_IOV: u64 = 1024;

pub(super) fn writable(f: &OpenFile) -> bool {
    matches!(f.flags & open::O_ACCMODE, open::O_WRONLY | open::O_RDWR)
        && f.flags & open::O_PATH == 0
}

pub(super) fn readable(f: &OpenFile) -> bool {
    matches!(f.flags & open::O_ACCMODE, open::O_RDONLY | open::O_RDWR)
        && f.flags & open::O_PATH == 0
}

/// pread(2), preadv(2): `EINVAL` for a negative position (checked before the descriptor).
fn check_offset(off: Option<u64>) -> Result<(), Errno> {
    match off {
        Some(o) if (o as i64) < 0 => Err(Errno::EINVAL),
        _ => Ok(()),
    }
}

/// Where a name sorts in a directory listing: `.`, then `..`, then every other name in byte
/// order.
fn dirent_key(name: &[u8]) -> (u8, &[u8]) {
    let rank = match name {
        b"." => 0,
        b".." => 1,
        _ => 2,
    };
    (rank, name)
}

/// A `stat` for one of the host stdio streams: a pipe, which is what a captured stream is.
fn stdio_stat(s: Stdio) -> Stat {
    Stat {
        dev: 0xc,
        ino: 1 + s as u64,
        mode: mode::S_IFIFO | 0o600,
        nlink: 1,
        blksize: 4096,
        ..Stat::default()
    }
}

impl Kernel {
    fn cwd(&self, tid: Tid) -> Ino {
        self.procs[&self.pid_of(tid)].cwd
    }

    /// Frees the removed directories that no descriptor and no working directory refers to
    /// any more (a no-op unless a directory was removed while in use).
    pub(crate) fn reap_dirs(&mut self) {
        let procs = &self.procs;
        self.vfs
            .reap_dead_dirs(|ino| procs.values().any(|p| p.cwd == ino));
    }

    fn umask_of(&self, tid: Tid) -> u32 {
        self.procs[&self.pid_of(tid)].umask
    }

    /// The directory a `*at` call starts from.
    pub(super) fn dir_ino(&self, tid: Tid, dirfd: u64) -> Result<Ino, Errno> {
        if dirfd as i32 == at::AT_FDCWD {
            return Ok(self.cwd(tid));
        }
        let file = self.file(tid, dirfd)?;
        let f = file.borrow();
        match f.kind {
            FileKind::Inode { ino, .. } if self.vfs.exists(ino) && self.vfs.inode(ino).is_dir() => {
                Ok(ino)
            }
            _ => Err(Errno::ENOTDIR),
        }
    }

    /// The directory a `*at` call resolves `path` from: openat(2), "If pathname is absolute,
    /// then dirfd is ignored" (it is not even checked).
    fn start_dir(&self, tid: Tid, dirfd: u64, path: &[u8]) -> Result<Ino, Errno> {
        if path.starts_with(b"/") {
            Ok(ROOT)
        } else {
            self.dir_ino(tid, dirfd)
        }
    }

    pub(super) fn read_path_arg(&self, tid: Tid, ptr: u64) -> Result<Vec<u8>, Errno> {
        read_path(self.mm(tid).borrow_mut().mem.as_mut(), ptr)
    }

    /// Resolves `(dirfd, path)`; an empty path with `AT_EMPTY_PATH` names `dirfd` itself
    /// (`Ok(None)` when that is not an inode, e.g. stdio).
    pub(crate) fn lookup_at(
        &self,
        tid: Tid,
        dirfd: u64,
        path_ptr: u64,
        flags: u64,
    ) -> Result<Option<Ino>, Errno> {
        let path = self.read_path_arg(tid, path_ptr)?;
        if path.is_empty() {
            if flags & at::AT_EMPTY_PATH == 0 {
                return Err(Errno::ENOENT);
            }
            if dirfd as i32 == at::AT_FDCWD {
                return Ok(Some(self.cwd(tid)));
            }
            let file = self.file(tid, dirfd)?;
            return Ok(match file.borrow().kind {
                FileKind::Inode { ino, .. } => Some(ino),
                _ => None,
            });
        }
        let dir = self.start_dir(tid, dirfd, &path)?;
        self.vfs
            .resolve(dir, &path, flags & at::AT_SYMLINK_NOFOLLOW == 0)
            .map(Some)
    }

    fn parent_at(&self, tid: Tid, dirfd: u64, path_ptr: u64) -> Result<(Ino, Vec<u8>), Errno> {
        let path = self.read_path_arg(tid, path_ptr)?;
        let dir = self.start_dir(tid, dirfd, &path)?;
        self.vfs.resolve_parent(dir, &path)
    }

    fn file_stat(&self, file: &FileRef) -> Stat {
        match file.borrow().kind {
            FileKind::Unix(_) | FileKind::Inet(_) | FileKind::Netlink(_) => Stat {
                dev: 0x8,
                ino: 1,
                mode: mode::S_IFSOCK | 0o777,
                nlink: 1,
                blksize: 4096,
                ..Stat::default()
            },
            FileKind::EventFd(_)
            | FileKind::TimerFd(_)
            | FileKind::Epoll(_)
            | FileKind::SignalFd(_)
            | FileKind::PidFd { .. }
            | FileKind::MQueue(_)
            | FileKind::Inotify(_) => Stat {
                dev: 0xe,
                ino: 1,
                mode: 0o600,
                nlink: 1,
                blksize: 4096,
                ..Stat::default()
            },
            FileKind::Stdio(s) => stdio_stat(s),
            FileKind::HidRaw(ref h) => self.vfs.stat(h.borrow().ino),
            FileKind::Tty { ino, ref tty, .. } => {
                if self.vfs.exists(ino) {
                    self.vfs.stat(ino)
                } else {
                    Stat {
                        dev: 0x16,
                        ino,
                        mode: mode::S_IFCHR | 0o620,
                        nlink: 1,
                        rdev: tty.borrow().rdev(),
                        blksize: 1024,
                        ..Stat::default()
                    }
                }
            }
            FileKind::Proc { ref data } => {
                crate::procfs::ProcNode::File(data.to_vec()).stat(&file.borrow().path)
            }
            FileKind::ProcDir { ref path } => crate::procfs::ProcNode::Dir(Vec::new()).stat(path),
            FileKind::Inode { ino, .. } => self.vfs.stat(ino),
            FileKind::Pipe { ref pipe, .. } => Stat {
                dev: 0xd,
                ino: pipe.borrow().id,
                mode: mode::S_IFIFO | 0o600,
                nlink: 1,
                blksize: 4096,
                ..Stat::default()
            },
        }
    }

    // ---- reading and writing ----

    pub(super) fn read_file(
        &mut self,
        file: &FileRef,
        buf: &mut [u8],
        at_offset: Option<u64>,
        host: &mut dyn Host,
    ) -> Result<usize, Errno> {
        let mut f = file.borrow_mut();
        if !readable(&f) {
            return Err(Errno::EBADF);
        }
        match f.kind.clone() {
            FileKind::Proc { data } => {
                let pos = at_offset.unwrap_or(f.offset).min(data.len() as u64) as usize;
                let n = buf.len().min(data.len() - pos);
                buf[..n].copy_from_slice(&data[pos..pos + n]);
                if at_offset.is_none() {
                    f.offset += n as u64;
                }
                Ok(n)
            }
            FileKind::ProcDir { .. } => Err(Errno::EISDIR),
            FileKind::EventFd(e) => {
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.eventfd_read(&e, buf, nonblock)
            }
            FileKind::TimerFd(t) => {
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                let now = host.monotonic_ns();
                self.timerfd_read(&t, buf, nonblock, now)
            }
            FileKind::Inotify(i) => {
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.inotify_read(&i, buf, nonblock)
            }
            FileKind::SignalFd(s) => {
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.signalfd_read(&s, buf, nonblock)
            }
            FileKind::PidFd { .. } => Err(Errno::EINVAL),
            FileKind::MQueue(q) => {
                let data = q.borrow().status();
                let pos = at_offset.unwrap_or(f.offset).min(data.len() as u64) as usize;
                let n = buf.len().min(data.len() - pos);
                buf[..n].copy_from_slice(&data[pos..pos + n]);
                if at_offset.is_none() {
                    f.offset += n as u64;
                }
                Ok(n)
            }
            FileKind::Epoll(_) => Err(Errno::EINVAL),
            FileKind::Unix(s) => {
                if at_offset.is_some() {
                    return Err(Errno::ESPIPE);
                }
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.unix_recv(&s, buf, false, nonblock).map(|(n, _, _)| n)
            }
            FileKind::Inet(s) => {
                if at_offset.is_some() {
                    return Err(Errno::ESPIPE);
                }
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.inet_recv(&s, buf, false, nonblock)
            }
            FileKind::Netlink(n) => {
                if at_offset.is_some() {
                    return Err(Errno::ESPIPE);
                }
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.netlink_recv(&n, buf, false, nonblock)
                    .map(|(len, _)| len)
            }
            FileKind::Stdio(Stdio::In) => {
                if at_offset.is_some() {
                    return Err(Errno::ESPIPE);
                }
                Ok(host.stdin(buf))
            }
            FileKind::Stdio(_) => Err(Errno::EBADF),
            FileKind::Tty { tty, master, .. } => {
                if at_offset.is_some() {
                    return Err(Errno::ESPIPE);
                }
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.tty_read(&tty, master, buf, nonblock, host.monotonic_ns())
            }
            FileKind::Pipe { pipe, write: false } => {
                if at_offset.is_some() {
                    return Err(Errno::ESPIPE);
                }
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.pipe_read(&pipe, buf, nonblock)
            }
            FileKind::Pipe { write: true, .. } => Err(Errno::EBADF),
            FileKind::HidRaw(h) => {
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.hidraw_read(&h, buf, nonblock)
            }
            FileKind::Inode {
                device: Some(Device::Fb(n)),
                ..
            } => {
                let pos = at_offset.unwrap_or(f.offset);
                let got = self.fb_read(n, pos, buf)?;
                if at_offset.is_none() {
                    f.offset = pos + got as u64;
                }
                Ok(got)
            }
            FileKind::Inode {
                device: Some(d), ..
            } => Ok(match d {
                Device::Null => 0,
                Device::Zero | Device::Full => {
                    buf.fill(0);
                    buf.len()
                }
                Device::Random => {
                    host.random(buf);
                    buf.len()
                }
                // `/dev/console` without an attached console: the host's plain stdin.
                Device::Tty | Device::Console => host.stdin(buf),
                Device::Ptmx | Device::Pts(_) => return Err(Errno::EIO),
                // Opened by an `O_PATH` description only (`read` is EBADF before this).
                Device::HidRaw(_) | Device::Fb(_) => return Err(Errno::EBADF),
            }),
            FileKind::Inode { ino, device: None } => {
                if self.vfs.inode(ino).is_dir() {
                    return Err(Errno::EISDIR);
                }
                let pos = at_offset.unwrap_or(f.offset);
                let n = self.vfs.read(ino, pos, buf)?;
                if n > 0 {
                    self.vfs.note(ino, crate::vfs::fsmask::IN_ACCESS);
                }
                if at_offset.is_none() {
                    f.offset += n as u64;
                }
                Ok(n)
            }
        }
    }

    /// Writes to a description. `at_offset` is `pwrite`'s position (`None`: the file
    /// offset); `rw_flags` holds `pwritev2`'s `RWF_APPEND`/`RWF_NOAPPEND`. As on Linux, a
    /// positioned write to an `O_APPEND` file appends, unless `RWF_NOAPPEND`.
    pub(super) fn write_file(
        &mut self,
        tid: Tid,
        file: &FileRef,
        data: &[u8],
        at_offset: Option<u64>,
        rw_flags: u64,
        host: &mut dyn Host,
    ) -> Result<usize, Errno> {
        let mut f = file.borrow_mut();
        if !writable(&f) {
            return Err(Errno::EBADF);
        }
        match f.kind.clone() {
            FileKind::Proc { .. } | FileKind::ProcDir { .. } => Err(Errno::EACCES),
            FileKind::EventFd(e) => {
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.eventfd_write(&e, data, nonblock)
            }
            FileKind::TimerFd(_)
            | FileKind::Epoll(_)
            | FileKind::Inotify(_)
            | FileKind::SignalFd(_)
            | FileKind::MQueue(_)
            | FileKind::PidFd { .. } => Err(Errno::EINVAL),
            FileKind::Unix(s) => {
                if at_offset.is_some() {
                    return Err(Errno::ESPIPE);
                }
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                match self.unix_send(&s, data, None, nonblock) {
                    Err(Errno::EPIPE) => {
                        let _ =
                            self.send_to_thread(tid, wasmshim_linux_abi::consts::signal::SIGPIPE);
                        Err(Errno::EPIPE)
                    }
                    r => r,
                }
            }
            FileKind::Inet(s) => {
                if at_offset.is_some() {
                    return Err(Errno::ESPIPE);
                }
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                match self.inet_send(&s, data, nonblock) {
                    Err(Errno::EPIPE) => {
                        let _ =
                            self.send_to_thread(tid, wasmshim_linux_abi::consts::signal::SIGPIPE);
                        Err(Errno::EPIPE)
                    }
                    r => r,
                }
            }
            FileKind::Netlink(n) => {
                if at_offset.is_some() {
                    return Err(Errno::ESPIPE);
                }
                drop(f);
                self.netlink_send(tid, &n, data)
            }
            FileKind::Stdio(Stdio::Out) => {
                host.stdout(data);
                Ok(data.len())
            }
            FileKind::Stdio(Stdio::Err) => {
                host.stderr(data);
                Ok(data.len())
            }
            FileKind::Stdio(Stdio::In) => Err(Errno::EBADF),
            FileKind::Pipe { pipe, write: true } => {
                if at_offset.is_some() {
                    return Err(Errno::ESPIPE);
                }
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.pipe_write(tid, &pipe, data, nonblock)
            }
            FileKind::Pipe { write: false, .. } => Err(Errno::EBADF),
            // Output reports (keyboard LEDs) have nowhere to go: accepted and dropped.
            FileKind::HidRaw(_) => Ok(data.len()),
            FileKind::Inode {
                device: Some(Device::Fb(n)),
                ..
            } => {
                let pos = at_offset.unwrap_or(f.offset);
                let n = self.fb_write(n, pos, data)?;
                if at_offset.is_none() {
                    f.offset = pos + n as u64;
                }
                Ok(n)
            }
            FileKind::Inode {
                device: Some(d), ..
            } => match d {
                Device::Full => Err(Errno::ENOSPC),
                Device::Tty | Device::Console => {
                    host.stdout(data);
                    Ok(data.len())
                }
                Device::Ptmx | Device::Pts(_) => Err(Errno::EIO),
                _ => Ok(data.len()),
            },
            FileKind::Tty { tty, master, .. } => {
                if at_offset.is_some() {
                    return Err(Errno::ESPIPE);
                }
                let nonblock = f.flags & open::O_NONBLOCK != 0;
                drop(f);
                self.tty_write(&tty, master, data, nonblock, host.monotonic_ns())
            }
            FileKind::Inode { ino, device: None } => {
                let now = Timespec::from_nanos(host.realtime_ns());
                let append = rw_flags & rwf::RWF_APPEND != 0
                    || (f.flags & open::O_APPEND != 0 && rw_flags & rwf::RWF_NOAPPEND == 0);
                let pos = match at_offset {
                    _ if append => self.vfs.size(ino),
                    Some(p) => p,
                    None => f.offset,
                };
                let n = self.vfs.write(ino, pos, data, now)?;
                if at_offset.is_none() {
                    f.offset = pos + n as u64;
                }
                Ok(n)
            }
        }
    }

    pub(super) fn sys_read(
        &mut self,
        tid: Tid,
        fd: u64,
        buf: u64,
        len: u64,
        off: Option<u64>,
        host: &mut dyn Host,
    ) -> SysResult {
        check_offset(off)?;
        let file = self.file(tid, fd)?;
        let len = io_len(len);
        let mut data = vec![0u8; len];
        let n = self.read_file(&file, &mut data, off, host)?;
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &data[..n])?;
        Ok(n as u64)
    }

    pub(super) fn sys_write(
        &mut self,
        tid: Tid,
        fd: u64,
        buf: u64,
        len: u64,
        off: Option<u64>,
        host: &mut dyn Host,
    ) -> SysResult {
        check_offset(off)?;
        let file = self.file(tid, fd)?;
        let len = io_len(len);
        let data = read_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, len)?;
        self.write_file(tid, &file, &data, off, 0, host)
            .map(|n| n as u64)
    }

    pub(super) fn iovecs(&self, tid: Tid, iov: u64, cnt: u64) -> Result<Vec<Iovec>, Errno> {
        if cnt > MAX_IOV {
            return Err(Errno::EINVAL);
        }
        let mm = self.mm(tid);
        let mut mm = mm.borrow_mut();
        (0..cnt)
            .map(|i| {
                Ok(Iovec::from_bytes(&read_array::<16>(
                    mm.mem.as_mut(),
                    iov + 16 * i,
                )?))
            })
            .collect()
    }

    pub(super) fn sys_readv(
        &mut self,
        tid: Tid,
        fd: u64,
        iov: u64,
        cnt: u64,
        off: Option<u64>,
        host: &mut dyn Host,
    ) -> SysResult {
        check_offset(off)?;
        let file = self.file(tid, fd)?;
        let mut total = 0u64;
        for v in self.iovecs(tid, iov, cnt)? {
            let len = io_len(v.len);
            let mut data = vec![0u8; len];
            let n = match self.read_file(&file, &mut data, off.map(|o| o + total), host) {
                Ok(n) => n,
                // Data already read is returned rather than lost to a blocking restart.
                Err(e) if total > 0 && matches!(e, super::BLOCK | Errno::EAGAIN) => {
                    self.cancel_wait();
                    break;
                }
                Err(e) => return Err(e),
            };
            write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), v.base, &data[..n])?;
            total += n as u64;
            if n < len {
                break;
            }
        }
        Ok(total)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_writev(
        &mut self,
        tid: Tid,
        fd: u64,
        iov: u64,
        cnt: u64,
        off: Option<u64>,
        rw_flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        check_offset(off)?;
        let file = self.file(tid, fd)?;
        // Gather first so one writev is one write (atomic for pipes and the host's streams).
        let mut data = Vec::new();
        for v in self.iovecs(tid, iov, cnt)? {
            let len = io_len(v.len).min(MAX_IO.saturating_sub(data.len()));
            data.extend(read_bytes(
                self.mm(tid).borrow_mut().mem.as_mut(),
                v.base,
                len,
            )?);
        }
        self.write_file(tid, &file, &data, off, rw_flags, host)
            .map(|n| n as u64)
    }

    /// `preadv2`/`pwritev2` (`fd, iov, iovcnt, pos_l, pos_h, flags`; 64-bit, so `pos_l` is the
    /// position): -1 means the file offset. `RWF_HIPRI`, `RWF_DSYNC` and `RWF_SYNC` are no-ops
    /// here; `RWF_NOWAIT` is refused with `EOPNOTSUPP` (callers fall back).
    pub(super) fn sys_rwv2(
        &mut self,
        tid: Tid,
        a: &[u64; 6],
        write: bool,
        host: &mut dyn Host,
    ) -> SysResult {
        let flags = a[5];
        let known = rwf::RWF_HIPRI | rwf::RWF_DSYNC | rwf::RWF_SYNC;
        let known = if write {
            known | rwf::RWF_APPEND | rwf::RWF_NOAPPEND
        } else {
            known
        };
        if flags & !known != 0 {
            return Err(Errno::EOPNOTSUPP);
        }
        let off = match a[3] as i64 {
            -1 => None,
            o if o < 0 => return Err(Errno::EINVAL),
            o => Some(o as u64),
        };
        if write {
            self.sys_writev(tid, a[0], a[1], a[2], off, flags, host)
        } else {
            self.sys_readv(tid, a[0], a[1], a[2], off, host)
        }
    }

    pub(super) fn sys_sendfile(
        &mut self,
        tid: Tid,
        out_fd: u64,
        in_fd: u64,
        off_ptr: u64,
        count: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let input = self.file(tid, in_fd)?;
        let output = self.file(tid, out_fd)?;
        let explicit = if off_ptr == 0 {
            None
        } else {
            Some(crate::uaccess::read_u64(
                self.mm(tid).borrow_mut().mem.as_mut(),
                off_ptr,
            )?)
        };
        // sendfile(2): the input offset moves by the bytes transferred. A file is read at its
        // offset without moving it, so bytes the output did not take (a short write, or a
        // write that blocks) stay for the next call.
        let seekable = matches!(input.borrow().kind, FileKind::Inode { device: None, .. });
        let at = match explicit {
            None if seekable => Some(input.borrow().offset),
            at => at,
        };
        // rw_verify_area (fs/read_write.c): a count or a start position that is negative as a
        // signed value, or a range whose end passes the largest offset, is EINVAL. So
        // `pos + written` below cannot overflow.
        let end = at.unwrap_or(0).checked_add(count);
        if (count as i64) < 0 || end.is_none_or(|end| end > i64::MAX as u64) {
            return Err(Errno::EINVAL);
        }
        let mut data = vec![0u8; io_len(count)];
        let n = self.read_file(&input, &mut data, at, host)?;
        let written = self.write_file(tid, &output, &data[..n], None, 0, host)?;
        if explicit.is_none()
            && seekable
            && let Some(pos) = at
        {
            input.borrow_mut().offset = pos + written as u64;
        }
        if let Some(o) = explicit {
            write_u64(
                self.mm(tid).borrow_mut().mem.as_mut(),
                off_ptr,
                o + written as u64,
            )?;
        }
        Ok(written as u64)
    }

    pub(super) fn sys_lseek(&mut self, tid: Tid, fd: u64, offset: u64, whence: u64) -> SysResult {
        let file = self.file(tid, fd)?;
        let mut f = file.borrow_mut();
        let size = match f.kind {
            FileKind::Stdio(_)
            | FileKind::Pipe { .. }
            | FileKind::Unix(_)
            | FileKind::Inet(_)
            | FileKind::Netlink(_)
            | FileKind::EventFd(_)
            | FileKind::SignalFd(_)
            | FileKind::PidFd { .. }
            | FileKind::MQueue(_)
            | FileKind::TimerFd(_)
            | FileKind::Epoll(_)
            | FileKind::Inotify(_)
            | FileKind::HidRaw(_)
            | FileKind::Tty { .. } => return Err(Errno::ESPIPE),
            FileKind::Proc { ref data } => data.len() as u64,
            FileKind::ProcDir { .. } => 0,
            FileKind::Inode {
                device: Some(Device::Fb(n)),
                ..
            } => self.fb_size(n),
            FileKind::Inode {
                device: Some(_), ..
            } => return Ok(0),
            FileKind::Inode { ino, .. } => self.vfs.size(ino),
        };
        let base = match whence {
            seek::SEEK_SET => 0,
            seek::SEEK_CUR => f.offset,
            seek::SEEK_END => size,
            _ => return Err(Errno::EINVAL),
        };
        let new = (base as i64)
            .checked_add(offset as i64)
            .filter(|&n| n >= 0)
            .ok_or(Errno::EINVAL)?;
        f.offset = new as u64;
        Ok(new as u64)
    }

    // ---- opening and closing ----

    pub(super) fn sys_openat(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path_ptr: u64,
        flags: u64,
        perm: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let path = self.read_path_arg(tid, path_ptr)?;
        self.open_path(tid, dirfd, path, flags, perm, host, 0)
    }

    /// `openat` on an already-copied path (`depth` counts /proc symlink hops).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn open_path(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path: Vec<u8>,
        flags: u64,
        perm: u64,
        host: &mut dyn Host,
        depth: u32,
    ) -> SysResult {
        if path.is_empty() {
            return Err(Errno::ENOENT);
        }
        if let Some(alias) = Self::dev_alias(&path, flags & open::O_NOFOLLOW == 0)
            && let Some(proc_path) = self.proc_target(tid, at::AT_FDCWD as u64, &alias)
        {
            return self.proc_open(tid, proc_path, flags, host, depth);
        }
        if let Some(proc_path) = self.proc_target(tid, dirfd, &path) {
            return self.proc_open(tid, proc_path, flags, host, depth);
        }
        if flags & open::O_TMPFILE == open::O_TMPFILE {
            return Err(Errno::EOPNOTSUPP);
        }
        let dir = self.start_dir(tid, dirfd, &path)?;
        let follow = flags & open::O_NOFOLLOW == 0;
        // A symlink chain that leaves the VFS for /proc (`access.log -> /dev/stdout`).
        if let Ok(crate::vfs::Resolved::Escape(rest)) = self.vfs.resolve_escape(dir, &path, follow)
        {
            if depth > 8 {
                return Err(Errno::ELOOP);
            }
            return self.open_path(tid, at::AT_FDCWD as u64, rest, flags, perm, host, depth + 1);
        }
        let acc = flags & open::O_ACCMODE;
        let now = Timespec::from_nanos(host.realtime_ns());
        let mut created = false;
        let ino = if flags & open::O_CREAT != 0 {
            if path.ends_with(b"/") {
                return Err(Errno::EISDIR);
            }
            match self.vfs.resolve(dir, &path, follow) {
                Ok(_) if flags & open::O_EXCL != 0 => return Err(Errno::EEXIST),
                Ok(ino) => ino,
                Err(Errno::ENOENT) => {
                    let (parent, name) = self.vfs.resolve_parent(dir, &path)?;
                    // open(2): with O_CREAT|O_EXCL a trailing symlink is never followed, so a
                    // dangling one is EEXIST; with O_CREAT alone its target is created.
                    if flags & open::O_EXCL != 0 && self.vfs.lookup(parent, &name).is_ok() {
                        return Err(Errno::EEXIST);
                    }
                    let (parent, name) = self.vfs.creation_target(parent, name)?;
                    let perm = (perm as u32) & !self.umask_of(tid) & 0o7777;
                    created = true;
                    self.vfs.create(parent, &name, perm, now)?
                }
                Err(e) => return Err(e),
            }
        } else {
            self.vfs.resolve(dir, &path, follow)?
        };
        let inode = self.vfs.inode(ino);
        let is_dir = inode.is_dir();
        let device = match inode.node {
            Node::Device(d) => Some(d),
            Node::Symlink(_) if flags & open::O_PATH == 0 => return Err(Errno::ELOOP),
            Node::Special | Node::Socket if flags & open::O_PATH == 0 => return Err(Errno::ENXIO),
            _ => None,
        };
        if flags & open::O_DIRECTORY != 0 && !is_dir {
            return Err(Errno::ENOTDIR);
        }
        if is_dir
            && (acc != open::O_RDONLY || flags & open::O_CREAT != 0)
            && flags & open::O_PATH == 0
        {
            return Err(Errno::EISDIR);
        }
        // A file this call created is empty already (Linux skips the truncate, and its event).
        if flags & open::O_TRUNC != 0
            && !created
            && acc != open::O_RDONLY
            && matches!(self.vfs.inode(ino).node, Node::File(_))
        {
            self.vfs.truncate(ino, 0, now)?;
        }
        let canonical = self.vfs.realpath(dir, &path).unwrap_or(path);
        let kind = match device {
            Some(Device::HidRaw(n)) if flags & open::O_PATH == 0 => Some(self.hidraw_open(n, ino)?),
            Some(d) if flags & open::O_PATH == 0 => self.tty_open(tid, d, ino, flags, now)?,
            _ => None,
        }
        .unwrap_or(FileKind::Inode { ino, device });
        let file = open_file(
            kind,
            flags & !(open::O_CREAT | open::O_EXCL | open::O_TRUNC | open::O_CLOEXEC),
            &canonical,
        );
        self.vfs.retain(ino);
        let fd =
            match self
                .fds(tid)
                .borrow_mut()
                .insert(file.clone(), flags & open::O_CLOEXEC != 0, 0)
            {
                Ok(fd) => fd,
                Err(e) => {
                    self.release_file(&file);
                    return Err(e);
                }
            };
        if flags & open::O_PATH == 0 {
            self.vfs.note(ino, crate::vfs::fsmask::IN_OPEN);
        }
        Ok(fd as u64)
    }

    pub(super) fn sys_close(&mut self, tid: Tid, fd: u64) -> SysResult {
        let slot = self.fds(tid).borrow_mut().remove(fd as i32)?;
        let pid = self.pid_of(tid);
        self.locks_fd_closed(pid, &slot.file);
        self.release_file(&slot.file);
        Ok(0)
    }

    pub(super) fn sys_close_range(
        &mut self,
        tid: Tid,
        first: u64,
        last: u64,
        flags: u64,
    ) -> SysResult {
        const CLOSE_RANGE_CLOEXEC: u64 = 4;
        if first > last {
            return Err(Errno::EINVAL);
        }
        let fds = self.fds(tid);
        let targets: Vec<i32> = fds
            .borrow()
            .iter()
            .map(|(fd, _)| fd)
            .filter(|&fd| (fd as u64) >= first && (fd as u64) <= last)
            .collect();
        for fd in targets {
            if flags & CLOSE_RANGE_CLOEXEC != 0 {
                if let Ok(slot) = fds.borrow_mut().slot_mut(fd) {
                    slot.cloexec = true;
                }
            } else {
                let slot = fds.borrow_mut().remove(fd);
                if let Ok(slot) = slot {
                    let pid = self.pid_of(tid);
                    self.locks_fd_closed(pid, &slot.file);
                    self.release_file(&slot.file);
                }
            }
        }
        Ok(0)
    }

    pub(super) fn sys_dup(&mut self, tid: Tid, fd: u64) -> SysResult {
        let fds = self.fds(tid);
        let file = fds.borrow().get(fd as i32)?;
        Ok(fds.borrow_mut().insert(file, false, 0)? as u64)
    }

    pub(super) fn sys_dup3(&mut self, tid: Tid, old: u64, new: u64, flags: u64) -> SysResult {
        if old == new || flags & !open::O_CLOEXEC != 0 {
            return Err(Errno::EINVAL);
        }
        let fds = self.fds(tid);
        let file = fds.borrow().get(old as i32)?;
        let replaced =
            fds.borrow_mut()
                .insert_at(new as i32, file, flags & open::O_CLOEXEC != 0)?;
        if let Some(r) = replaced {
            let pid = self.pid_of(tid);
            self.locks_fd_closed(pid, &r.file);
            self.release_file(&r.file);
        }
        Ok(new)
    }

    pub(super) fn sys_fcntl(&mut self, tid: Tid, fd: u64, cmd: u64, arg: u64) -> SysResult {
        let fds = self.fds(tid);
        let file = fds.borrow().get(fd as i32)?;
        match cmd {
            fcntl::F_DUPFD | fcntl::F_DUPFD_CLOEXEC => {
                Ok(fds
                    .borrow_mut()
                    .insert(file, cmd == fcntl::F_DUPFD_CLOEXEC, arg as i32)?
                    as u64)
            }
            fcntl::F_GETFD => Ok(u64::from(fds.borrow_mut().slot_mut(fd as i32)?.cloexec)),
            fcntl::F_SETFD => {
                fds.borrow_mut().slot_mut(fd as i32)?.cloexec = arg & fcntl::FD_CLOEXEC != 0;
                Ok(0)
            }
            fcntl::F_GETFL => Ok(file.borrow().flags),
            fcntl::F_SETFL => {
                let changeable = open::O_APPEND
                    | open::O_NONBLOCK
                    | open::O_DIRECT
                    | open::O_NOATIME
                    | open::O_ASYNC;
                let mut f = file.borrow_mut();
                f.flags = (f.flags & !changeable) | (arg & changeable);
                Ok(0)
            }
            fcntl::F_GETLK
            | fcntl::F_SETLK
            | fcntl::F_SETLKW
            | super::flock::F_OFD_GETLK
            | super::flock::F_OFD_SETLK
            | super::flock::F_OFD_SETLKW => self.sys_fcntl_lock(tid, &file, cmd, arg),
            // Recorded, but SIGIO is never sent (signal-driven I/O is not modelled; nginx sets
            // it on its master/worker channels and does not depend on it).
            fcntl::F_SETOWN => {
                file.borrow_mut().owner = arg as i32;
                Ok(0)
            }
            fcntl::F_GETOWN => Ok(file.borrow().owner as i64 as u64),
            // fcntl(2): a pipe's capacity (either end); EBADF for anything else.
            fcntl::F_GETPIPE_SZ => match file.borrow().kind {
                FileKind::Pipe { .. } => Ok(crate::pipe::PIPE_CAPACITY as u64),
                _ => Err(Errno::EBADF),
            },
            fcntl::F_ADD_SEALS => self.add_seals(&file, arg),
            fcntl::F_GET_SEALS => self.get_seals(&file),
            _ => Err(Errno::EINVAL),
        }
    }

    pub(super) fn sys_ioctl(&mut self, tid: Tid, fd: u64, req: u64, arg: u64) -> SysResult {
        // The kernel takes `unsigned int cmd`: musl passes an `int`, sign-extended, so
        // `_IOR` requests (bit 31 set) arrive with the upper half set.
        let req = req & 0xffff_ffff;
        let file = self.file(tid, fd)?;
        // The display and input devices take every request but the generic
        // descriptor ones.
        if !matches!(
            req,
            ioctl::FIOCLEX | ioctl::FIONCLEX | ioctl::FIOASYNC | ioctl::FIONBIO
        ) {
            let kind = file.borrow().kind.clone();
            match kind {
                FileKind::HidRaw(h) => return self.hidraw_ioctl(tid, &h, req, arg),
                FileKind::Inode {
                    device: Some(Device::Fb(n)),
                    ..
                } => return self.fb_ioctl(tid, n, req, arg),
                _ => {}
            }
        }
        match req {
            ioctl::FIOCLEX | ioctl::FIONCLEX => {
                self.fds(tid).borrow_mut().slot_mut(fd as i32)?.cloexec = req == ioctl::FIOCLEX;
                Ok(0)
            }
            ioctl::FIOASYNC => {
                let on = self
                    .mm(tid)
                    .borrow()
                    .mem
                    .read_u32(arg)
                    .map_err(|_| Errno::EFAULT)?
                    != 0;
                let mut f = file.borrow_mut();
                if on {
                    f.flags |= open::O_ASYNC;
                } else {
                    f.flags &= !open::O_ASYNC;
                }
                Ok(0)
            }
            ioctl::FIONBIO => {
                let on = self
                    .mm(tid)
                    .borrow()
                    .mem
                    .read_u32(arg)
                    .map_err(|_| Errno::EFAULT)?
                    != 0;
                let mut f = file.borrow_mut();
                if on {
                    f.flags |= open::O_NONBLOCK;
                } else {
                    f.flags &= !open::O_NONBLOCK;
                }
                Ok(0)
            }
            _ if matches!(file.borrow().kind, FileKind::Tty { .. }) => {
                self.tty_ioctl(tid, &file, req, arg)
            }
            // The host's streams are captured, not terminals: every TTY request is ENOTTY.
            ioctl::TCGETS
            | ioctl::TCSETS
            | ioctl::TCSETSW
            | ioctl::TCSETSF
            | ioctl::TIOCGWINSZ
            | ioctl::TIOCSWINSZ
            | ioctl::TIOCGPGRP
            | ioctl::TIOCSPGRP
            | ioctl::TIOCSCTTY => Err(Errno::ENOTTY),
            ioctl::FIONREAD => match file.borrow().kind {
                FileKind::Inode { ino, device: None } => {
                    let left = self.vfs.size(ino).saturating_sub(file.borrow().offset);
                    crate::uaccess::write_u32(
                        self.mm(tid).borrow_mut().mem.as_mut(),
                        arg,
                        left.min(i32::MAX as u64) as u32,
                    )?;
                    Ok(0)
                }
                // Bytes (stream) or the next datagram's size waiting to be read.
                ref kind @ (FileKind::Unix(_)
                | FileKind::Inet(_)
                | FileKind::Pipe { .. }
                | FileKind::Inotify(_)) => {
                    let n = match kind {
                        FileKind::Unix(s) => s.borrow().readable_bytes(),
                        FileKind::Inet(s) => s.borrow().rx.len(),
                        FileKind::Pipe { pipe, .. } => pipe.borrow().buf.len(),
                        FileKind::Inotify(i) => i.borrow().pending_bytes(),
                        _ => 0,
                    };
                    crate::uaccess::write_u32(
                        self.mm(tid).borrow_mut().mem.as_mut(),
                        arg,
                        n.min(i32::MAX as usize) as u32,
                    )?;
                    Ok(0)
                }
                _ => Err(Errno::ENOTTY),
            },
            _ => Err(Errno::ENOTTY),
        }
    }

    // ---- directories and metadata ----

    pub(super) fn sys_getdents64(&mut self, tid: Tid, fd: u64, buf: u64, count: u64) -> SysResult {
        let file = self.file(tid, fd)?;
        let mut f = file.borrow_mut();
        // A directory lists `.`, `..`, then its names in byte order, so the last name returned
        // marks the position and the next call resumes after it. Names removed or created
        // meanwhile do not shift the rest: every name present throughout is returned exactly
        // once (readdir(3p); tmpfs keeps stable offsets for this, fs/libfs.c
        // `offset_readdir`), so `rm -rf` of a large directory removes everything. `/proc`
        // listings keep plain indices.
        let mut sorted = false;
        let entries = match &f.kind {
            FileKind::Inode { ino, .. } => {
                sorted = true;
                self.vfs.readdir(*ino)?
            }
            FileKind::ProcDir { path } => {
                let crate::procfs::ProcNode::Dir(list) = self.proc_lookup(path, 0, 0)? else {
                    return Err(Errno::ENOTDIR);
                };
                let mut all = vec![
                    crate::vfs::DirEntry {
                        name: b".".to_vec(),
                        ino: 1,
                        dtype: wasmshim_linux_abi::consts::dtype::DT_DIR,
                    },
                    crate::vfs::DirEntry {
                        name: b"..".to_vec(),
                        ino: 1,
                        dtype: wasmshim_linux_abi::consts::dtype::DT_DIR,
                    },
                ];
                for (i, (name, dtype)) in list.into_iter().enumerate() {
                    all.push(crate::vfs::DirEntry {
                        name,
                        ino: 2 + i as u64,
                        dtype,
                    });
                }
                all
            }
            _ => return Err(Errno::ENOTDIR),
        };
        let mut index = match &f.dir_cursor {
            Some((at, last)) if sorted && *at == f.offset => {
                let last = dirent_key(last);
                entries.partition_point(|e| dirent_key(&e.name) <= last)
            }
            // A position set by `lseek` (`rewinddir`, `seekdir`) is an entry index.
            _ => usize::try_from(f.offset).unwrap_or(usize::MAX),
        };
        let mut out = Vec::new();
        let mut last = None;
        while let Some(e) = entries.get(index) {
            let len = Dirent64::reclen(e.name.len());
            if (out.len() + len) as u64 > count {
                if out.is_empty() {
                    return Err(Errno::EINVAL);
                }
                break;
            }
            let start = out.len();
            out.resize(start + len, 0);
            Dirent64::encode(&mut out[start..], e.ino, index as u64 + 1, e.dtype, &e.name);
            last = Some(&e.name);
            index += 1;
        }
        // At the end nothing moves: names created later still come after the cursor.
        if let Some(name) = last {
            f.offset = index as u64;
            f.dir_cursor = sorted.then(|| (f.offset, name.clone()));
        }
        drop(f);
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &out)?;
        Ok(out.len() as u64)
    }

    fn put_stat(&self, tid: Tid, buf: u64, st: &Stat) -> SysResult {
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &st.to_bytes())?;
        Ok(0)
    }

    pub(super) fn sys_fstat(&mut self, tid: Tid, fd: u64, buf: u64) -> SysResult {
        let file = self.file(tid, fd)?;
        let st = self.file_stat(&file);
        self.put_stat(tid, buf, &st)
    }

    pub(super) fn sys_newfstatat(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path: u64,
        buf: u64,
        flags: u64,
    ) -> SysResult {
        let follow = flags & at::AT_SYMLINK_NOFOLLOW == 0;
        if let Some(p) = self.proc_path_follow(tid, dirfd, path, follow)? {
            let st = self.proc_stat(tid, &p, flags & at::AT_SYMLINK_NOFOLLOW == 0)?;
            return self.put_stat(tid, buf, &st);
        }
        let st = match self.lookup_at(tid, dirfd, path, flags)? {
            Some(ino) => self.vfs.stat(ino),
            None => self.file_stat(&self.file(tid, dirfd)?),
        };
        self.put_stat(tid, buf, &st)
    }

    pub(super) fn sys_statx(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path: u64,
        flags: u64,
        _mask: u64,
        buf: u64,
    ) -> SysResult {
        let follow = flags & at::AT_SYMLINK_NOFOLLOW == 0;
        let st = if let Some(p) = self.proc_path_follow(tid, dirfd, path, follow)? {
            self.proc_stat(tid, &p, flags & at::AT_SYMLINK_NOFOLLOW == 0)?
        } else {
            match self.lookup_at(tid, dirfd, path, flags)? {
                Some(ino) => self.vfs.stat(ino),
                None => self.file_stat(&self.file(tid, dirfd)?),
            }
        };
        let x = Statx {
            mask: Statx::STATX_BASIC_STATS,
            blksize: st.blksize as u32,
            nlink: st.nlink,
            uid: st.uid,
            gid: st.gid,
            mode: st.mode as u16,
            ino: st.ino,
            size: st.size as u64,
            blocks: st.blocks as u64,
            atime: st.atime,
            ctime: st.ctime,
            mtime: st.mtime,
            rdev_major: (st.rdev >> 8) as u32,
            rdev_minor: (st.rdev & 0xff) as u32,
            dev_major: (st.dev >> 8) as u32,
            dev_minor: (st.dev & 0xff) as u32,
            ..Statx::default()
        };
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &x.to_bytes())?;
        Ok(0)
    }

    fn put_statfs(&self, tid: Tid, buf: u64) -> SysResult {
        let mut b = [0u8; 120];
        let fields: [u64; 7] = [
            TMPFS_MAGIC,
            4096,
            1 << 20,
            1 << 19,
            1 << 19,
            1 << 16,
            1 << 15,
        ];
        for (i, v) in fields.iter().enumerate() {
            b[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        b[64..72].copy_from_slice(&255u64.to_le_bytes()); // f_namelen
        b[72..80].copy_from_slice(&4096u64.to_le_bytes()); // f_frsize
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &b)?;
        Ok(0)
    }

    /// `statfs(path, buf)`: one filesystem answers for every file, but the path must resolve
    /// (statfs(2): ENOENT, ENOTDIR, ELOOP, EFAULT...). `/proc` and `/sys` paths are generated
    /// outside the VFS and taken as they are.
    pub(super) fn sys_statfs(&mut self, tid: Tid, path: u64, buf: u64) -> SysResult {
        if self
            .proc_path_follow(tid, at::AT_FDCWD as u64, path, true)?
            .is_none()
        {
            self.lookup_at(tid, at::AT_FDCWD as u64, path, 0)?;
        }
        self.put_statfs(tid, buf)
    }

    pub(super) fn sys_fstatfs(&mut self, tid: Tid, fd: u64, buf: u64) -> SysResult {
        self.file(tid, fd)?;
        self.put_statfs(tid, buf)
    }

    pub(super) fn sys_readlinkat(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path: u64,
        buf: u64,
        size: u64,
    ) -> SysResult {
        if (size as i64) <= 0 {
            return Err(Errno::EINVAL);
        }
        if let Some(p) = self.proc_path_arg(tid, dirfd, path)? {
            let crate::procfs::ProcNode::Link(target) = self.proc_lookup(&p, 0, 0)? else {
                return Err(Errno::EINVAL);
            };
            let n = target.len().min(size as usize);
            write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &target[..n])?;
            return Ok(n as u64);
        }
        let ino = self
            .lookup_at(
                tid,
                dirfd,
                path,
                at::AT_SYMLINK_NOFOLLOW | at::AT_EMPTY_PATH,
            )?
            .ok_or(Errno::EINVAL)?;
        let target = self.vfs.readlink(ino)?;
        let n = target.len().min(size as usize);
        let data = target[..n].to_vec();
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &data)?;
        Ok(n as u64)
    }

    pub(super) fn sys_faccessat(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path: u64,
        amode: u64,
        flags: u64,
    ) -> SysResult {
        if let Some(p) = self.proc_path_follow(tid, dirfd, path, true)? {
            self.proc_stat(tid, &p, true)?;
            return Ok(0);
        }
        let ino = self
            .lookup_at(tid, dirfd, path, flags)?
            .ok_or(Errno::EBADF)?;
        if amode & access::X_OK != 0 {
            let i = self.vfs.inode(ino);
            if !i.is_dir() && i.mode & 0o111 == 0 {
                return Err(Errno::EACCES);
            }
        }
        Ok(0)
    }

    pub(super) fn sys_mkdirat(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path: u64,
        perm: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        // An existing path (including "/", which has no parent entry) is EEXIST.
        let full = self.read_path_arg(tid, path)?;
        let base = self.start_dir(tid, dirfd, &full)?;
        if !full.is_empty() && self.vfs.resolve(base, &full, false).is_ok() {
            return Err(Errno::EEXIST);
        }
        let (dir, name) = self.parent_at(tid, dirfd, path)?;
        let perm = (perm as u32) & !self.umask_of(tid) & 0o7777;
        let now = Timespec::from_nanos(host.realtime_ns());
        self.vfs.mkdir(dir, &name, perm, now).map(|_| 0)
    }

    pub(super) fn sys_mknodat(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path: u64,
        kind: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        // mknod(2): a regular file (type 0 too) or a FIFO, which needs no privilege; device
        // nodes and sockets are not made here (EPERM); any other type is EINVAL.
        let ty = (kind as u32) & mode::S_IFMT;
        match ty {
            0 | mode::S_IFREG | mode::S_IFIFO => {}
            mode::S_IFCHR | mode::S_IFBLK | mode::S_IFSOCK => return Err(Errno::EPERM),
            _ => return Err(Errno::EINVAL),
        }
        let (dir, name) = self.parent_at(tid, dirfd, path)?;
        let perm = (kind as u32) & !self.umask_of(tid) & 0o7777;
        let now = Timespec::from_nanos(host.realtime_ns());
        if ty == mode::S_IFIFO {
            return self.vfs.mkfifo(dir, &name, perm, now).map(|_| 0);
        }
        self.vfs.create(dir, &name, perm, now).map(|_| 0)
    }

    pub(super) fn sys_unlinkat(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        // unlinkat(2): EINVAL for any flag but AT_REMOVEDIR.
        if flags & !at::AT_REMOVEDIR != 0 {
            return Err(Errno::EINVAL);
        }
        let (dir, name) = self.parent_at(tid, dirfd, path)?;
        let now = Timespec::from_nanos(host.realtime_ns());
        self.vfs
            .remove(dir, &name, flags & at::AT_REMOVEDIR != 0, now)?;
        self.reap_dirs();
        Ok(0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_renameat2(
        &mut self,
        tid: Tid,
        odirfd: u64,
        opath: u64,
        ndirfd: u64,
        npath: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if flags & !RENAME_NOREPLACE != 0 {
            return Err(Errno::EINVAL);
        }
        let (odir, oname) = self.parent_at(tid, odirfd, opath)?;
        let (ndir, nname) = self.parent_at(tid, ndirfd, npath)?;
        if flags & RENAME_NOREPLACE != 0 && self.vfs.lookup(ndir, &nname).is_ok() {
            return Err(Errno::EEXIST);
        }
        let now = Timespec::from_nanos(host.realtime_ns());
        self.vfs.rename(odir, &oname, ndir, &nname, now)?;
        self.reap_dirs();
        Ok(0)
    }

    pub(super) fn sys_symlinkat(
        &mut self,
        tid: Tid,
        target: u64,
        dirfd: u64,
        path: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let target = self.read_path_arg(tid, target)?;
        if target.is_empty() {
            return Err(Errno::ENOENT);
        }
        let (dir, name) = self.parent_at(tid, dirfd, path)?;
        let now = Timespec::from_nanos(host.realtime_ns());
        self.vfs.symlink(dir, &name, &target, now).map(|_| 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_linkat(
        &mut self,
        tid: Tid,
        odirfd: u64,
        opath: u64,
        ndirfd: u64,
        npath: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let follow = if flags & at::AT_SYMLINK_FOLLOW != 0 {
            0
        } else {
            at::AT_SYMLINK_NOFOLLOW
        };
        let ino = self
            .lookup_at(tid, odirfd, opath, follow | (flags & at::AT_EMPTY_PATH))?
            .ok_or(Errno::EPERM)?;
        let (dir, name) = self.parent_at(tid, ndirfd, npath)?;
        let now = Timespec::from_nanos(host.realtime_ns());
        self.vfs.link(ino, dir, &name, now).map(|_| 0)
    }

    fn fd_ino(&self, tid: Tid, fd: u64) -> Result<Option<Ino>, Errno> {
        Ok(match self.file(tid, fd)?.borrow().kind {
            FileKind::Inode { ino, .. } => Some(ino),
            _ => None,
        })
    }

    pub(super) fn sys_fchmod(
        &mut self,
        tid: Tid,
        fd: u64,
        perm: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if let Some(ino) = self.fd_ino(tid, fd)? {
            let now = Timespec::from_nanos(host.realtime_ns());
            self.vfs.chmod(ino, perm as u32, now);
        }
        Ok(0)
    }

    pub(super) fn sys_fchmodat(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path: u64,
        perm: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let ino = self.lookup_at(tid, dirfd, path, 0)?.ok_or(Errno::EBADF)?;
        let now = Timespec::from_nanos(host.realtime_ns());
        self.vfs.chmod(ino, perm as u32, now);
        Ok(0)
    }

    pub(super) fn sys_fchown(
        &mut self,
        tid: Tid,
        fd: u64,
        uid: u64,
        gid: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if let Some(ino) = self.fd_ino(tid, fd)? {
            let now = Timespec::from_nanos(host.realtime_ns());
            self.vfs.chown(ino, uid as u32, gid as u32, now);
        }
        Ok(0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_fchownat(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path: u64,
        uid: u64,
        gid: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let ino = self
            .lookup_at(tid, dirfd, path, flags)?
            .ok_or(Errno::EBADF)?;
        let now = Timespec::from_nanos(host.realtime_ns());
        self.vfs.chown(ino, uid as u32, gid as u32, now);
        Ok(0)
    }

    pub(super) fn sys_utimensat(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path: u64,
        times: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let ino = if path == 0 {
            self.fd_ino(tid, dirfd)?
        } else {
            self.lookup_at(tid, dirfd, path, flags)?
        };
        let Some(ino) = ino else { return Ok(0) };
        let now = Timespec::from_nanos(host.realtime_ns());
        let (atime, mtime) = if times == 0 {
            (Some(now), Some(now))
        } else {
            let raw: [u8; 32] = read_array(self.mm(tid).borrow_mut().mem.as_mut(), times)?;
            let pick = |b: &[u8]| {
                let t = Timespec::from_bytes(b.try_into().unwrap_or(&[0; 16]));
                match t.nsec {
                    UTIME_NOW => Ok(Some(now)),
                    UTIME_OMIT => Ok(None),
                    n if (0..1_000_000_000).contains(&n) => Ok(Some(t)),
                    _ => Err(Errno::EINVAL),
                }
            };
            (pick(&raw[..16])?, pick(&raw[16..])?)
        };
        self.vfs.set_times(ino, atime, mtime, now);
        Ok(0)
    }

    pub(super) fn sys_truncate(
        &mut self,
        tid: Tid,
        path: u64,
        len: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let ino = self
            .lookup_at(tid, at::AT_FDCWD as u64, path, 0)?
            .ok_or(Errno::EINVAL)?;
        if (len as i64) < 0 {
            return Err(Errno::EINVAL);
        }
        let now = Timespec::from_nanos(host.realtime_ns());
        self.vfs.truncate(ino, len, now).map(|_| 0)
    }

    pub(super) fn sys_ftruncate(
        &mut self,
        tid: Tid,
        fd: u64,
        len: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let file = self.file(tid, fd)?;
        let f = file.borrow();
        let FileKind::Inode { ino, device: None } = f.kind else {
            return Err(Errno::EINVAL);
        };
        if !writable(&f) || (len as i64) < 0 {
            return Err(Errno::EINVAL);
        }
        let now = Timespec::from_nanos(host.realtime_ns());
        self.vfs.truncate(ino, len, now).map(|_| 0)
    }

    pub(super) fn sys_getcwd(&mut self, tid: Tid, buf: u64, size: u64) -> SysResult {
        // getcwd(2): ENOENT "The current working directory has been unlinked."
        if self.vfs.is_dead_dir(self.cwd(tid)) {
            return Err(Errno::ENOENT);
        }
        let mut path = self.vfs.path_of(self.cwd(tid));
        path.push(0);
        if (size as usize) < path.len() {
            return Err(Errno::ERANGE);
        }
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &path)?;
        Ok(path.len() as u64)
    }

    pub(super) fn sys_chdir(&mut self, tid: Tid, path: u64) -> SysResult {
        let ino = self
            .lookup_at(tid, at::AT_FDCWD as u64, path, 0)?
            .unwrap_or(ROOT);
        if !self.vfs.inode(ino).is_dir() {
            return Err(Errno::ENOTDIR);
        }
        self.set_cwd(tid, ino);
        Ok(0)
    }

    pub(super) fn sys_fchdir(&mut self, tid: Tid, fd: u64) -> SysResult {
        let ino = self.dir_ino(tid, fd)?;
        self.set_cwd(tid, ino);
        Ok(0)
    }

    /// Moves the caller's process to directory `ino`; the old one may have been removed.
    fn set_cwd(&mut self, tid: Tid, ino: Ino) {
        let pid = self.pid_of(tid);
        if let Some(p) = self.procs.get_mut(&pid) {
            p.cwd = ino;
        }
        self.reap_dirs();
    }

    pub(super) fn sys_umask(&mut self, tid: Tid, mask: u64) -> SysResult {
        let pid = self.pid_of(tid);
        let p = self.procs.get_mut(&pid).ok_or(Errno::ESRCH)?;
        let old = p.umask;
        p.umask = (mask as u32) & 0o777;
        Ok(u64::from(old))
    }

    /// Readiness of a description at `now`: (readable, writable, hang-up, error).
    fn readiness(&self, file: &FileRef, now: u64) -> (bool, bool, bool, bool) {
        use crate::objects::{EPOLLERR, EPOLLHUP, EPOLLIN, EPOLLOUT};
        let (ev, _) = self.poll_events(file, now);
        (
            ev & EPOLLIN != 0,
            ev & EPOLLOUT != 0,
            ev & EPOLLHUP != 0,
            ev & EPOLLERR != 0,
        )
    }

    /// Reads an optional relative timeout; `Ok(None)` means "wait forever".
    pub(crate) fn read_timeout(
        &mut self,
        tid: Tid,
        ptr: u64,
        host: &mut dyn Host,
    ) -> Result<Option<u64>, Errno> {
        if ptr == 0 {
            return Ok(None);
        }
        let ts = Timespec::from_bytes(&read_array(self.mm(tid).borrow_mut().mem.as_mut(), ptr)?);
        let ns = ts.to_nanos_saturating().ok_or(Errno::EINVAL)?;
        // Fixed at the call's first entry: wake-ups that find nothing never extend it.
        let now = host.monotonic_ns();
        Ok(self.restart_deadline(tid, Some(ns), now))
    }

    /// `ppoll(fds, nfds, timeout, sigmask)`: blocks (restartably) until a description changes or
    /// a watched timer expires ([`Kernel::wait_objects`]).
    pub(super) fn sys_ppoll(
        &mut self,
        tid: Tid,
        fds_ptr: u64,
        nfds: u64,
        tmo: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if nfds > 1024 {
            return Err(Errno::EINVAL);
        }
        let mm = self.mm(tid);
        let raw = read_bytes(mm.borrow_mut().mem.as_mut(), fds_ptr, 8 * nfds as usize)?;
        let mut out = raw.clone();
        let mut ready = 0;
        let mut wait = ObjectWait::default();
        let now = host.monotonic_ns();
        for i in 0..nfds as usize {
            let fd =
                i32::from_le_bytes([raw[8 * i], raw[8 * i + 1], raw[8 * i + 2], raw[8 * i + 3]]);
            let events = u16::from_le_bytes([raw[8 * i + 4], raw[8 * i + 5]]);
            let revents = if fd < 0 {
                0
            } else {
                match self.fds(tid).borrow().get(fd) {
                    Ok(file) => {
                        let (r, w, hup, err) = self.readiness(&file, now);
                        wait.watch(self, &file, now, !r && events & POLLIN != 0);
                        (if r { events & POLLIN } else { 0 })
                            | (if w { events & POLLOUT } else { 0 })
                            | if hup { POLLHUP } else { 0 }
                            | if err { POLLERR } else { 0 }
                    }
                    Err(_) => POLLNVAL,
                }
            };
            if revents != 0 {
                ready += 1;
            }
            out[8 * i + 6..8 * i + 8].copy_from_slice(&revents.to_le_bytes());
        }
        let deadline = self.read_timeout(tid, tmo, host)?;
        if ready == 0 && deadline.is_none_or(|d| d > now) {
            return Err(self.wait_objects(wait, deadline));
        }
        write_bytes(mm.borrow_mut().mem.as_mut(), fds_ptr, &out)?;
        Ok(ready)
    }

    /// `pselect6`'s last argument: a pointer to `{ const sigset_t *ss; size_t ss_len; }`
    /// (null: no mask). Returns the mask pointer and its size.
    pub(super) fn pselect6_mask(&self, tid: Tid, arg: u64) -> Result<(u64, u64), Errno> {
        if arg == 0 {
            return Ok((0, 0));
        }
        let raw: [u8; 16] = read_array(self.mm(tid).borrow_mut().mem.as_mut(), arg)?;
        let word = |at: usize| u64::from_le_bytes(raw[at..at + 8].try_into().unwrap_or_default());
        Ok((word(0), word(8)))
    }

    /// `pselect6(nfds, readfds, writefds, exceptfds, timeout, sigmask)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_pselect6(
        &mut self,
        tid: Tid,
        nfds: u64,
        rset: u64,
        wset: u64,
        eset: u64,
        tmo: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if nfds > 1024 {
            return Err(Errno::EINVAL);
        }
        let words = nfds.div_ceil(64) as usize;
        let mm = self.mm(tid);
        let read_set = |mm: &mut crate::mm::AddressSpace, p: u64| -> Result<Vec<u64>, Errno> {
            if p == 0 {
                return Ok(vec![0; words]);
            }
            (0..words)
                .map(|w| crate::uaccess::read_u64(mm.mem.as_mut(), p + 8 * w as u64))
                .collect()
        };
        let (r_in, w_in, e_in) = {
            let mut m = mm.borrow_mut();
            (
                read_set(&mut m, rset)?,
                read_set(&mut m, wset)?,
                read_set(&mut m, eset)?,
            )
        };
        let (mut r_out, mut w_out) = (vec![0u64; words], vec![0u64; words]);
        let mut ready = 0;
        let mut wait = ObjectWait::default();
        let now = host.monotonic_ns();
        for fd in 0..nfds as usize {
            let (w, b) = (fd / 64, 1u64 << (fd % 64));
            if (r_in[w] | w_in[w]) & b == 0 {
                // select(2): EBADF for a closed descriptor in any set. Exceptional
                // conditions (POLLPRI: out-of-band data) are never reported, so `exceptfds`
                // comes back empty.
                if e_in[w] & b != 0 {
                    self.fds(tid).borrow().get(fd as i32)?;
                }
                continue;
            }
            let file = self.fds(tid).borrow().get(fd as i32)?;
            let (readable, writable, hup, _) = self.readiness(&file, now);
            wait.watch(self, &file, now, !readable && r_in[w] & b != 0);
            if r_in[w] & b != 0 && (readable || hup) {
                r_out[w] |= b;
                ready += 1;
            }
            if w_in[w] & b != 0 && writable {
                w_out[w] |= b;
                ready += 1;
            }
        }
        let deadline = self.read_timeout(tid, tmo, host)?;
        if ready == 0 && deadline.is_none_or(|d| d > now) {
            return Err(self.wait_objects(wait, deadline));
        }
        let mut m = mm.borrow_mut();
        for (p, set) in [(rset, &r_out), (wset, &w_out), (eset, &vec![0u64; words])] {
            if p != 0 {
                for (w, v) in set.iter().enumerate() {
                    write_u64(m.mem.as_mut(), p + 8 * w as u64, *v)?;
                }
            }
        }
        Ok(ready)
    }
}

impl Kernel {
    /// Called whenever a descriptor reference to `file` is dropped. When it was the last
    /// reference (the description closes), inodes lose an open count and pipe ends close.
    pub(crate) fn release_file(&mut self, file: &FileRef) {
        if std::rc::Rc::strong_count(file) != 1 {
            return;
        }
        let kind = file.borrow().kind.clone();
        match kind {
            FileKind::Inode { ino, .. } => {
                self.locks_description_closed(file);
                let wrote = file.borrow().flags & open::O_ACCMODE != open::O_RDONLY;
                let path_only = file.borrow().flags & open::O_PATH != 0;
                if !path_only {
                    self.vfs.note(
                        ino,
                        if wrote {
                            crate::vfs::fsmask::IN_CLOSE_WRITE
                        } else {
                            crate::vfs::fsmask::IN_CLOSE_NOWRITE
                        },
                    );
                }
                self.vfs.release(ino);
                self.reap_dirs();
            }
            FileKind::Inotify(i) => self.inotify_closed(&i),
            FileKind::Tty { tty, master, ino } => self.tty_closed(&tty, master, ino),
            FileKind::Pipe { pipe, write } => self.pipe_end_closed(&pipe, write),
            FileKind::Unix(s) => self.unix_closed(&s),
            FileKind::Inet(s) => self.inet_closed(&s),
            _ => {}
        }
    }
}

// ---- /proc glue ----

impl Kernel {
    /// Opens a `/proc` node; symlinks are followed (into the VFS or back into /proc).
    fn proc_open(
        &mut self,
        tid: Tid,
        path: Vec<u8>,
        flags: u64,
        host: &mut dyn Host,
        depth: u32,
    ) -> SysResult {
        use crate::procfs::ProcNode;
        if depth > 8 {
            return Err(Errno::ELOOP);
        }
        let node = self.proc_lookup(&path, host.monotonic_ns(), host.realtime_ns())?;
        let acc = flags & open::O_ACCMODE;
        let kind = match node {
            ProcNode::Link(target) if flags & (open::O_NOFOLLOW | open::O_PATH) == 0 => {
                if let Some(file) = self.proc_fd_description(&path) {
                    // `/proc/<pid>/fd/<n>` reopens that file.
                    let kind = self.reopen_kind(&file.borrow().kind);
                    let p = file.borrow().path.clone();
                    let new = open_file(
                        kind,
                        flags & !(open::O_CREAT | open::O_EXCL | open::O_TRUNC | open::O_CLOEXEC),
                        &p,
                    );
                    return Ok(self.fds(tid).borrow_mut().insert(
                        new,
                        flags & open::O_CLOEXEC != 0,
                        0,
                    )? as u64);
                }
                let target = if target.starts_with(b"/") {
                    target
                } else {
                    let mut t = path[..path.iter().rposition(|&b| b == b'/').unwrap_or(0)].to_vec();
                    t.push(b'/');
                    t.extend_from_slice(&target);
                    t
                };
                return self.open_path(tid, at::AT_FDCWD as u64, target, flags, 0, host, depth + 1);
            }
            ProcNode::Link(_) => return Err(Errno::ELOOP),
            ProcNode::File(data) => {
                if acc != open::O_RDONLY {
                    return Err(Errno::EACCES);
                }
                if flags & open::O_DIRECTORY != 0 {
                    return Err(Errno::ENOTDIR);
                }
                FileKind::Proc {
                    data: std::rc::Rc::new(data),
                }
            }
            ProcNode::Dir(_) => {
                if acc != open::O_RDONLY {
                    return Err(Errno::EISDIR);
                }
                FileKind::ProcDir { path: path.clone() }
            }
        };
        let file = open_file(
            kind,
            flags & !(open::O_CREAT | open::O_EXCL | open::O_TRUNC | open::O_CLOEXEC),
            &path,
        );
        Ok(self
            .fds(tid)
            .borrow_mut()
            .insert(file, flags & open::O_CLOEXEC != 0, 0)? as u64)
    }

    /// The open file behind `/proc/<pid>/fd/<n>`, if `path` is one.
    fn proc_fd_description(&self, path: &[u8]) -> Option<FileRef> {
        let text = std::str::from_utf8(path).ok()?;
        let parts: Vec<&str> = text.split('/').filter(|c| !c.is_empty()).collect();
        let ["proc", pid, "fd", n] = parts.as_slice() else {
            return None;
        };
        let pid: crate::kernel::Pid = pid.parse().ok()?;
        let fd: i32 = n.parse().ok()?;
        self.procs.get(&pid)?.fds.borrow().get(fd).ok()
    }

    /// A fresh description of the same object (Linux "reopen" through /proc/<pid>/fd).
    fn reopen_kind(&mut self, kind: &FileKind) -> FileKind {
        match kind {
            FileKind::Inode { ino, .. } => self.vfs.retain(*ino),
            FileKind::Tty { tty, master, ino } => self.tty_reopened(tty, *master, *ino),
            FileKind::Pipe { pipe, write } => {
                let mut p = pipe.borrow_mut();
                if *write {
                    p.writers += 1;
                } else {
                    p.readers += 1;
                }
            }
            _ => {}
        }
        kind.clone()
    }

    /// The normalised `/proc` path named by `(dirfd, path_ptr)`, if it is one.
    /// [`Self::proc_path_arg`], also following `/dev/std*` and `/dev/fd` into `/proc`.
    fn proc_path_follow(
        &self,
        tid: Tid,
        dirfd: u64,
        path_ptr: u64,
        follow: bool,
    ) -> Result<Option<Vec<u8>>, Errno> {
        let path = self.read_path_arg(tid, path_ptr)?;
        if let Some(alias) = Self::dev_alias(&path, follow) {
            return Ok(self.proc_target(tid, at::AT_FDCWD as u64, &alias));
        }
        Ok(self.proc_target(tid, dirfd, &path))
    }

    fn proc_path_arg(&self, tid: Tid, dirfd: u64, path_ptr: u64) -> Result<Option<Vec<u8>>, Errno> {
        let path = self.read_path_arg(tid, path_ptr)?;
        Ok(self.proc_target(tid, dirfd, &path))
    }

    /// `stat` of a `/proc` path (following symlinks if `follow`).
    fn proc_stat(&mut self, tid: Tid, path: &[u8], follow: bool) -> Result<Stat, Errno> {
        use crate::procfs::ProcNode;
        let mut path = path.to_vec();
        for _ in 0..8 {
            let node = self.proc_lookup(&path, 0, 0)?;
            let ProcNode::Link(target) = &node else {
                return Ok(node.stat(&path));
            };
            if !follow {
                return Ok(node.stat(&path));
            }
            if let Some(file) = self.proc_fd_description(&path) {
                return Ok(self.file_stat(&file));
            }
            if target.starts_with(b"/") {
                match self.proc_target(tid, at::AT_FDCWD as u64, target) {
                    Some(p) => path = p,
                    None => {
                        let ino = self.vfs.resolve(crate::vfs::ROOT, target, true)?;
                        return Ok(self.vfs.stat(ino));
                    }
                }
            } else {
                let mut t = path[..path.iter().rposition(|&b| b == b'/').unwrap_or(0)].to_vec();
                t.push(b'/');
                t.extend_from_slice(target);
                path = t;
            }
        }
        Err(Errno::ELOOP)
    }
}
