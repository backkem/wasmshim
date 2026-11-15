//! More file operations: extended attributes, `copy_file_range`, `splice`/`tee`/
//! `vmsplice`, `openat2`, `execveat`, `readahead`, `sync_file_range` and `fallocate`.
//!
//! The VFS behaves like tmpfs (what `statfs` reports): `user.`, `trusted.` and `security.`
//! attributes are kept per inode, `fallocate` supports allocating and punching holes only.

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{at, open};
use wasmshim_linux_abi::structs::Timespec;

use super::fs::{readable, writable};
use super::{Flow, SysResult};
use crate::exec::ExecStart;
use crate::fd::{FileKind, FileRef};
use crate::host::Host;
use crate::kernel::{Kernel, Tid, WaitKey};
use crate::pipe::{PIPE_CAPACITY, PipeRef};
use crate::uaccess::{MAX_IO, read_bytes, read_cstr_max, write_bytes};
use crate::vfs::{Ino, Node, Restrict, Walked};

const XATTR_CREATE: u64 = 1;
const XATTR_REPLACE: u64 = 2;
const XATTR_NAME_MAX: usize = 255;
const XATTR_SIZE_MAX: usize = 65536;
const XATTR_LIST_MAX: usize = 65536;

const SPLICE_F_NONBLOCK: u64 = 2;
const SPLICE_F_ALL: u64 = 0xf;

const FALLOC_FL_KEEP_SIZE: u64 = 1;
const FALLOC_FL_PUNCH_HOLE: u64 = 2;
const FALLOC_FL_COLLAPSE_RANGE: u64 = 8;
const FALLOC_FL_ZERO_RANGE: u64 = 0x10;
const FALLOC_FL_INSERT_RANGE: u64 = 0x20;
const FALLOC_FL_UNSHARE_RANGE: u64 = 0x40;

const RESOLVE_NO_XDEV: u64 = 0x01;
const RESOLVE_NO_MAGICLINKS: u64 = 0x02;
const RESOLVE_NO_SYMLINKS: u64 = 0x04;
const RESOLVE_BENEATH: u64 = 0x08;
const RESOLVE_IN_ROOT: u64 = 0x10;
const RESOLVE_CACHED: u64 = 0x20;
/// `sizeof(struct open_how)` of the first version.
const OPEN_HOW_SIZE_VER0: u64 = 24;

/// Where an xattr call points.
enum XattrTarget {
    Inode(Ino),
    /// A `/proc` file: a regular file without xattr support.
    None,
    /// A pipe, socket or other non-VFS object: a special inode without xattr support.
    Special,
}

/// What a pipe-side argument of `splice`/`tee` is.
fn pipe_of(file: &FileRef) -> Option<(PipeRef, bool)> {
    match &file.borrow().kind {
        FileKind::Pipe { pipe, write } => Some((pipe.clone(), *write)),
        _ => None,
    }
}

impl Kernel {
    // ---- extended attributes ----

    /// The inode a path-based xattr call names.
    fn xattr_path(&self, tid: Tid, path: u64, follow: bool) -> Result<XattrTarget, Errno> {
        let p = self.read_path_arg(tid, path)?;
        if self.proc_target(tid, at::AT_FDCWD as u64, &p).is_some() {
            return Ok(XattrTarget::None);
        }
        let flags = if follow { 0 } else { at::AT_SYMLINK_NOFOLLOW };
        Ok(self
            .lookup_at(tid, at::AT_FDCWD as u64, path, flags)?
            .map_or(XattrTarget::None, XattrTarget::Inode))
    }

    /// The inode an fd-based xattr call names.
    fn xattr_fd(&self, tid: Tid, fd: u64) -> Result<XattrTarget, Errno> {
        let file = self.file(tid, fd)?;
        let f = file.borrow();
        if f.flags & open::O_PATH != 0 {
            return Err(Errno::EBADF);
        }
        Ok(match f.kind {
            FileKind::Inode { ino, .. } if self.vfs.exists(ino) => XattrTarget::Inode(ino),
            FileKind::Tty { ino, .. } if self.vfs.exists(ino) => XattrTarget::Inode(ino),
            _ => XattrTarget::Special,
        })
    }

    /// Checks an attribute name and the caller's right to it on `ino`, like Linux's
    /// `xattr_permission` with tmpfs's handlers.
    fn xattr_check(&self, tid: Tid, ino: Ino, name: &[u8], write: bool) -> Result<(), Errno> {
        if name.is_empty() || name.len() > XATTR_NAME_MAX {
            return Err(Errno::ERANGE);
        }
        let inode = self.vfs.inode(ino);
        if name.starts_with(b"user.") {
            let plain = matches!(inode.node, Node::File(_) | Node::Dir(_));
            if !plain {
                return Err(if write { Errno::EPERM } else { Errno::ENODATA });
            }
            Ok(())
        } else if name.starts_with(b"trusted.") {
            if self.creds(tid).euid != 0 {
                return Err(if write { Errno::EPERM } else { Errno::ENODATA });
            }
            Ok(())
        } else if name.starts_with(b"security.") {
            Ok(())
        } else {
            // `system.` (POSIX ACLs) and unknown namespaces.
            Err(Errno::EOPNOTSUPP)
        }
    }

    fn xattr_name(&self, tid: Tid, ptr: u64) -> Result<Vec<u8>, Errno> {
        let mm = self.mm(tid);
        let mm = mm.borrow();
        match read_cstr_max(mm.mem.as_ref(), ptr, XATTR_NAME_MAX + 1) {
            Err(Errno::ENAMETOOLONG) => Err(Errno::ERANGE),
            r => r,
        }
    }

    /// `setxattr`, `lsetxattr`, `fsetxattr`.
    #[allow(clippy::too_many_arguments)]
    fn sys_setxattr(
        &mut self,
        tid: Tid,
        target: Result<XattrTarget, Errno>,
        name: u64,
        value: u64,
        size: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if flags & !(XATTR_CREATE | XATTR_REPLACE) != 0 {
            return Err(Errno::EINVAL);
        }
        let name = self.xattr_name(tid, name)?;
        // fs/xattr.c: a value over XATTR_SIZE_MAX is E2BIG. Sizes are compared as the guest's
        // 64-bit values, before they are narrowed.
        if size > XATTR_SIZE_MAX as u64 {
            return Err(Errno::E2BIG);
        }
        let data = read_bytes(self.mm(tid).borrow_mut().mem.as_mut(), value, size as usize)?;
        let ino = match target? {
            XattrTarget::Inode(ino) => ino,
            XattrTarget::Special => return Err(special_xattr(&name, true)),
            XattrTarget::None => return Err(Errno::EOPNOTSUPP),
        };
        self.xattr_check(tid, ino, &name, true)?;
        let exists = self.vfs.get_xattr(ino, &name).is_some();
        if flags & XATTR_CREATE != 0 && exists {
            return Err(Errno::EEXIST);
        }
        if flags & XATTR_REPLACE != 0 && !exists {
            return Err(Errno::ENODATA);
        }
        let now = Timespec::from_nanos(host.realtime_ns());
        self.vfs.set_xattr(ino, &name, Some(data), now);
        Ok(0)
    }

    /// `getxattr`, `lgetxattr`, `fgetxattr`.
    fn sys_getxattr(
        &mut self,
        tid: Tid,
        target: Result<XattrTarget, Errno>,
        name: u64,
        value: u64,
        size: u64,
    ) -> SysResult {
        let name = self.xattr_name(tid, name)?;
        let ino = match target? {
            XattrTarget::Inode(ino) => ino,
            XattrTarget::Special => return Err(special_xattr(&name, false)),
            XattrTarget::None => return Err(Errno::EOPNOTSUPP),
        };
        self.xattr_check(tid, ino, &name, false)?;
        let data = self
            .vfs
            .get_xattr(ino, &name)
            .ok_or(Errno::ENODATA)?
            .to_vec();
        if size == 0 {
            return Ok(data.len() as u64);
        }
        if size < data.len() as u64 {
            return Err(Errno::ERANGE);
        }
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), value, &data)?;
        Ok(data.len() as u64)
    }

    /// `listxattr`, `llistxattr`, `flistxattr`: the names the caller may see, NUL-terminated.
    fn sys_listxattr(
        &mut self,
        tid: Tid,
        target: Result<XattrTarget, Errno>,
        list: u64,
        size: u64,
    ) -> SysResult {
        let XattrTarget::Inode(ino) = target? else {
            return Ok(0);
        };
        let root = self.creds(tid).euid == 0;
        let mut out = Vec::new();
        // tmpfs lists in descending name order (what Linux 6.18 does).
        for name in self.vfs.inode(ino).xattrs.keys().rev() {
            if name.starts_with(b"trusted.") && !root {
                continue;
            }
            out.extend_from_slice(name);
            out.push(0);
        }
        if size == 0 {
            return Ok(out.len() as u64);
        }
        if size < out.len() as u64 || out.len() > XATTR_LIST_MAX {
            return Err(Errno::ERANGE);
        }
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), list, &out)?;
        Ok(out.len() as u64)
    }

    /// `removexattr`, `lremovexattr`, `fremovexattr`.
    fn sys_removexattr(
        &mut self,
        tid: Tid,
        target: Result<XattrTarget, Errno>,
        name: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        let name = self.xattr_name(tid, name)?;
        let ino = match target? {
            XattrTarget::Inode(ino) => ino,
            XattrTarget::Special => return Err(special_xattr(&name, true)),
            XattrTarget::None => return Err(Errno::EOPNOTSUPP),
        };
        self.xattr_check(tid, ino, &name, true)?;
        if self.vfs.get_xattr(ino, &name).is_none() {
            return Err(Errno::ENODATA);
        }
        let now = Timespec::from_nanos(host.realtime_ns());
        self.vfs.set_xattr(ino, &name, None, now);
        Ok(0)
    }

    /// Dispatches the twelve xattr syscalls (`kind`: 0 path, 1 `l` (no follow), 2 `f`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn xattr_call(
        &mut self,
        tid: Tid,
        op: XattrOp,
        kind: u8,
        a: &[u64; 6],
        host: &mut dyn Host,
    ) -> SysResult {
        let target = match kind {
            2 => self.xattr_fd(tid, a[0]),
            k => self.xattr_path(tid, a[0], k == 0),
        };
        match op {
            XattrOp::Set => self.sys_setxattr(tid, target, a[1], a[2], a[3], a[4], host),
            XattrOp::Get => self.sys_getxattr(tid, target, a[1], a[2], a[3]),
            XattrOp::List => self.sys_listxattr(tid, target, a[1], a[2]),
            XattrOp::Remove => self.sys_removexattr(tid, target, a[1], host),
        }
    }

    // ---- copy_file_range ----

    /// Reads a `loff_t` argument (a pointer, or `None` for the file offset).
    fn offset_arg(&self, tid: Tid, ptr: u64) -> Result<Option<u64>, Errno> {
        if ptr == 0 {
            return Ok(None);
        }
        let v = crate::uaccess::read_u64(self.mm(tid).borrow_mut().mem.as_mut(), ptr)?;
        if (v as i64) < 0 {
            return Err(Errno::EINVAL);
        }
        Ok(Some(v))
    }

    /// The regular file behind a `copy_file_range` argument.
    fn regular_file(&self, file: &FileRef) -> Result<Ino, Errno> {
        match file.borrow().kind {
            FileKind::Inode { ino, device: None } => {
                if self.vfs.inode(ino).is_dir() {
                    Err(Errno::EISDIR)
                } else if matches!(self.vfs.inode(ino).node, Node::File(_)) {
                    Ok(ino)
                } else {
                    Err(Errno::EINVAL)
                }
            }
            _ => Err(Errno::EINVAL),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_copy_file_range(
        &mut self,
        tid: Tid,
        fd_in: u64,
        off_in: u64,
        fd_out: u64,
        off_out: u64,
        len: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if flags != 0 {
            return Err(Errno::EINVAL);
        }
        let fin = self.file(tid, fd_in)?;
        let fout = self.file(tid, fd_out)?;
        if !readable(&fin.borrow())
            || !writable(&fout.borrow())
            || fout.borrow().flags & open::O_APPEND != 0
        {
            return Err(Errno::EBADF);
        }
        let ino_in = self.regular_file(&fin)?;
        let ino_out = self.regular_file(&fout)?;
        let pos_in = match self.offset_arg(tid, off_in)? {
            Some(p) => p,
            None => fin.borrow().offset,
        };
        let pos_out = match self.offset_arg(tid, off_out)? {
            Some(p) => p,
            None => fout.borrow().offset,
        };
        let size = self.vfs.size(ino_in);
        let n = len.min(size.saturating_sub(pos_in)).min(MAX_IO as u64);
        if ino_in == ino_out
            && pos_in < pos_out.saturating_add(len)
            && pos_out < pos_in.saturating_add(len)
        {
            return Err(Errno::EINVAL);
        }
        if n == 0 {
            return Ok(0);
        }
        let mut buf = vec![0u8; n as usize];
        let got = self.vfs.read(ino_in, pos_in, &mut buf)?;
        let now = Timespec::from_nanos(host.realtime_ns());
        let put = self.vfs.write(ino_out, pos_out, &buf[..got], now)?;
        let mm = self.mm(tid);
        if off_in != 0 {
            crate::uaccess::write_u64(mm.borrow_mut().mem.as_mut(), off_in, pos_in + put as u64)?;
        } else {
            fin.borrow_mut().offset = pos_in + put as u64;
        }
        if off_out != 0 {
            crate::uaccess::write_u64(mm.borrow_mut().mem.as_mut(), off_out, pos_out + put as u64)?;
        } else {
            fout.borrow_mut().offset = pos_out + put as u64;
        }
        Ok(put as u64)
    }

    // ---- splice, tee, vmsplice ----

    /// Waits for pipe `p` to change (or fails with `EAGAIN` when non-blocking).
    fn pipe_wait(&mut self, p: &PipeRef, nonblock: bool) -> Errno {
        if nonblock {
            return Errno::EAGAIN;
        }
        let id = p.borrow().id;
        self.wait_for(WaitKey::Object(id), None, None)
    }

    /// Bytes now in pipe `p`, or what an empty pipe means: 0 (no writers left) or a wait.
    fn pipe_ready(&mut self, p: &PipeRef, nonblock: bool) -> Result<usize, Errno> {
        let (len, writers) = {
            let b = p.borrow();
            (b.buf.len(), b.writers)
        };
        if len > 0 {
            return Ok(len);
        }
        if writers == 0 {
            return Ok(0);
        }
        Err(self.pipe_wait(p, nonblock))
    }

    /// Free space in pipe `p` for a writer, or `EPIPE` (with `SIGPIPE`) without readers, or a
    /// wait when full.
    fn pipe_space(&mut self, tid: Tid, p: &PipeRef, nonblock: bool) -> Result<usize, Errno> {
        let (len, readers) = {
            let b = p.borrow();
            (b.buf.len(), b.readers)
        };
        if readers == 0 {
            let _ = self.send_to_thread(tid, wasmshim_linux_abi::consts::signal::SIGPIPE);
            return Err(Errno::EPIPE);
        }
        if len < PIPE_CAPACITY {
            return Ok(PIPE_CAPACITY - len);
        }
        Err(self.pipe_wait(p, nonblock))
    }

    fn pipe_changed(&mut self, p: &PipeRef) {
        let id = {
            let mut b = p.borrow_mut();
            b.generation += 1;
            b.id
        };
        self.wake(WaitKey::Object(id), usize::MAX, u32::MAX);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_splice(
        &mut self,
        tid: Tid,
        fd_in: u64,
        off_in: u64,
        fd_out: u64,
        off_out: u64,
        len: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if flags & !SPLICE_F_ALL != 0 {
            return Err(Errno::EINVAL);
        }
        let fin = self.file(tid, fd_in)?;
        let fout = self.file(tid, fd_out)?;
        if !readable(&fin.borrow()) || !writable(&fout.borrow()) {
            return Err(Errno::EBADF);
        }
        if fout.borrow().flags & open::O_APPEND != 0 {
            return Err(Errno::EINVAL);
        }
        let pin = pipe_of(&fin);
        let pout = pipe_of(&fout);
        if pin.is_none() && pout.is_none() {
            return Err(Errno::EINVAL);
        }
        if (pin.is_some() && off_in != 0) || (pout.is_some() && off_out != 0) {
            return Err(Errno::ESPIPE);
        }
        if let (Some((a, _)), Some((b, _))) = (&pin, &pout)
            && std::rc::Rc::ptr_eq(a, b)
        {
            return Err(Errno::EINVAL);
        }
        if len == 0 {
            return Ok(0);
        }
        let len = crate::uaccess::io_len(len);
        let nb = |f: &FileRef| {
            flags & SPLICE_F_NONBLOCK != 0 || f.borrow().flags & open::O_NONBLOCK != 0
        };
        match (pin, pout) {
            (Some((src, _)), Some((dst, _))) => {
                let avail = self.pipe_ready(&src, nb(&fin))?;
                if avail == 0 {
                    return Ok(0);
                }
                let space = self.pipe_space(tid, &dst, nb(&fout))?;
                let n = len.min(avail).min(space);
                let data: Vec<u8> = src.borrow_mut().buf.drain(..n).collect();
                dst.borrow_mut().buf.extend(data);
                self.pipe_changed(&src);
                self.pipe_changed(&dst);
                Ok(n as u64)
            }
            (Some((src, _)), None) => {
                let avail = self.pipe_ready(&src, nb(&fin))?;
                if avail == 0 {
                    return Ok(0);
                }
                let n = len.min(avail);
                let data: Vec<u8> = src.borrow().buf.iter().take(n).copied().collect();
                let at = self.offset_arg(tid, off_out)?;
                let put = self.write_file(tid, &fout, &data, at, 0, host)?;
                if let Some(at) = at {
                    crate::uaccess::write_u64(
                        self.mm(tid).borrow_mut().mem.as_mut(),
                        off_out,
                        at + put as u64,
                    )?;
                }
                src.borrow_mut().buf.drain(..put);
                self.pipe_changed(&src);
                Ok(put as u64)
            }
            (None, Some((dst, _))) => {
                let space = self.pipe_space(tid, &dst, nb(&fout))?;
                let n = len.min(space);
                let mut buf = vec![0u8; n];
                let at = self.offset_arg(tid, off_in)?;
                let got = self.read_file(&fin, &mut buf, at, host)?;
                if let Some(at) = at {
                    crate::uaccess::write_u64(
                        self.mm(tid).borrow_mut().mem.as_mut(),
                        off_in,
                        at + got as u64,
                    )?;
                }
                if got > 0 {
                    dst.borrow_mut().buf.extend(&buf[..got]);
                    self.pipe_changed(&dst);
                }
                Ok(got as u64)
            }
            (None, None) => Err(Errno::EINVAL),
        }
    }

    pub(super) fn sys_tee(
        &mut self,
        tid: Tid,
        fd_in: u64,
        fd_out: u64,
        len: u64,
        flags: u64,
    ) -> SysResult {
        if flags & !SPLICE_F_ALL != 0 {
            return Err(Errno::EINVAL);
        }
        let fin = self.file(tid, fd_in)?;
        let fout = self.file(tid, fd_out)?;
        let (Some((src, false)), Some((dst, true))) = (pipe_of(&fin), pipe_of(&fout)) else {
            return Err(if readable(&fin.borrow()) && writable(&fout.borrow()) {
                Errno::EINVAL
            } else {
                Errno::EBADF
            });
        };
        if std::rc::Rc::ptr_eq(&src, &dst) {
            return Err(Errno::EINVAL);
        }
        if len == 0 {
            return Ok(0);
        }
        let nb = |f: &FileRef| {
            flags & SPLICE_F_NONBLOCK != 0 || f.borrow().flags & open::O_NONBLOCK != 0
        };
        let avail = self.pipe_ready(&src, nb(&fin))?;
        if avail == 0 {
            return Ok(0);
        }
        let space = self.pipe_space(tid, &dst, nb(&fout))?;
        let n = crate::uaccess::host_len(len).min(avail).min(space);
        let data: Vec<u8> = src.borrow().buf.iter().take(n).copied().collect();
        dst.borrow_mut().buf.extend(data);
        self.pipe_changed(&dst);
        Ok(n as u64)
    }

    pub(super) fn sys_vmsplice(
        &mut self,
        tid: Tid,
        fd: u64,
        iov: u64,
        cnt: u64,
        flags: u64,
    ) -> SysResult {
        if flags & !SPLICE_F_ALL != 0 {
            return Err(Errno::EINVAL);
        }
        let file = self.file(tid, fd)?;
        let Some((pipe, write)) = pipe_of(&file) else {
            return Err(Errno::EBADF);
        };
        let nonblock =
            flags & SPLICE_F_NONBLOCK != 0 || file.borrow().flags & open::O_NONBLOCK != 0;
        let iovs = self.iovecs(tid, iov, cnt)?;
        // The guest's lengths may add up past 2^64: the sum saturates (it is capped anyway).
        let total = crate::uaccess::io_len(iovs.iter().fold(0u64, |t, v| t.saturating_add(v.len)));
        if total == 0 {
            return Ok(0);
        }
        let mm = self.mm(tid);
        if write {
            let space = self.pipe_space(tid, &pipe, nonblock)?;
            let mut data = Vec::new();
            for v in &iovs {
                if data.len() >= space {
                    break;
                }
                let n = crate::uaccess::host_len(v.len).min(space - data.len());
                data.extend(read_bytes(mm.borrow_mut().mem.as_mut(), v.base, n)?);
            }
            pipe.borrow_mut().buf.extend(&data);
            self.pipe_changed(&pipe);
            Ok(data.len() as u64)
        } else {
            let avail = self.pipe_ready(&pipe, nonblock)?;
            let n = total.min(avail);
            let data: Vec<u8> = pipe.borrow_mut().buf.drain(..n).collect();
            let mut at = 0;
            for v in &iovs {
                if at >= data.len() {
                    break;
                }
                let k = crate::uaccess::host_len(v.len).min(data.len() - at);
                write_bytes(mm.borrow_mut().mem.as_mut(), v.base, &data[at..at + k])?;
                at += k;
            }
            if n > 0 {
                self.pipe_changed(&pipe);
            }
            Ok(n as u64)
        }
    }

    // ---- openat2, execveat ----

    pub(super) fn sys_openat2(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path_ptr: u64,
        how: u64,
        size: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if size < OPEN_HOW_SIZE_VER0 {
            return Err(Errno::EINVAL);
        }
        if size > wasmshim_cpu::PAGE_SIZE {
            return Err(Errno::E2BIG);
        }
        let raw = read_bytes(self.mm(tid).borrow_mut().mem.as_mut(), how, size as usize)?;
        if raw[OPEN_HOW_SIZE_VER0 as usize..].iter().any(|&b| b != 0) {
            return Err(Errno::E2BIG);
        }
        let word =
            |i: usize| u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap_or_default());
        let (flags, mode, resolve) = (word(0), word(1), word(2));
        let valid = open::O_ACCMODE
            | open::O_CREAT
            | open::O_EXCL
            | open::O_NOCTTY
            | open::O_TRUNC
            | open::O_APPEND
            | open::O_NONBLOCK
            | open::O_DSYNC
            | open::O_ASYNC
            | open::O_DIRECT
            | open::O_LARGEFILE
            | open::O_DIRECTORY
            | open::O_NOFOLLOW
            | open::O_NOATIME
            | open::O_CLOEXEC
            | open::O_SYNC
            | open::O_PATH
            | open::O_TMPFILE;
        if flags & !valid != 0 {
            return Err(Errno::EINVAL);
        }
        let creating = flags & open::O_CREAT != 0 || flags & open::O_TMPFILE == open::O_TMPFILE;
        if (creating && mode & !0o7777 != 0) || (!creating && mode != 0) {
            return Err(Errno::EINVAL);
        }
        if flags & open::O_PATH != 0
            && flags & !(open::O_PATH | open::O_DIRECTORY | open::O_NOFOLLOW | open::O_CLOEXEC) != 0
        {
            return Err(Errno::EINVAL);
        }
        let known = RESOLVE_NO_XDEV
            | RESOLVE_NO_MAGICLINKS
            | RESOLVE_NO_SYMLINKS
            | RESOLVE_BENEATH
            | RESOLVE_IN_ROOT
            | RESOLVE_CACHED;
        if resolve & !known != 0
            || resolve & (RESOLVE_BENEATH | RESOLVE_IN_ROOT) == RESOLVE_BENEATH | RESOLVE_IN_ROOT
        {
            return Err(Errno::EINVAL);
        }
        if resolve & RESOLVE_CACHED != 0 && (creating || flags & open::O_TRUNC != 0) {
            return Err(Errno::EAGAIN);
        }
        let path = self.read_path_arg(tid, path_ptr)?;
        let rules = Restrict {
            beneath: resolve & RESOLVE_BENEATH != 0,
            in_root: resolve & RESOLVE_IN_ROOT != 0,
            no_symlinks: resolve & RESOLVE_NO_SYMLINKS != 0,
            no_xdev: resolve & RESOLVE_NO_XDEV != 0,
        };
        let restricted = rules.beneath || rules.in_root || rules.no_symlinks || rules.no_xdev;
        if !restricted {
            if resolve & RESOLVE_NO_MAGICLINKS != 0 {
                self.check_magic_link(tid, dirfd, &path, flags)?;
            }
            return self.open_path(tid, dirfd, path, flags, mode, host, 0);
        }
        let start = self.dir_ino(tid, dirfd)?;
        let follow = flags & open::O_NOFOLLOW == 0;
        let target = match self.vfs.resolve_restricted(start, &path, follow, rules)? {
            Walked::Proc(rest) => {
                if resolve & RESOLVE_NO_MAGICLINKS != 0 {
                    self.check_magic_link(tid, at::AT_FDCWD as u64, &rest, flags)?;
                }
                rest
            }
            Walked::Dir(ino) => self.vfs.path_of(ino),
            Walked::Entry { parent, name } => {
                let mut p = self.vfs.path_of(parent);
                if p != b"/" {
                    p.push(b'/');
                }
                p.extend_from_slice(&name);
                if path.ends_with(b"/") {
                    p.push(b'/');
                }
                p
            }
        };
        self.open_path(tid, at::AT_FDCWD as u64, target, flags, mode, host, 0)
    }

    /// `RESOLVE_NO_MAGICLINKS`: a `/proc` "magic link" (`/proc/<pid>/fd/<n>`, `exe`, `cwd`,
    /// `root`) that the open would follow is `ELOOP`.
    fn check_magic_link(&self, tid: Tid, dirfd: u64, path: &[u8], flags: u64) -> Result<(), Errno> {
        let Some(proc_path) = self.proc_target(tid, dirfd, path) else {
            return Ok(());
        };
        let text = String::from_utf8_lossy(&proc_path).into_owned();
        let parts: Vec<&str> = text.split('/').filter(|c| !c.is_empty()).collect();
        let magic = |i: usize| match parts.get(i..) {
            Some([_, "fd", n, ..]) => n.parse::<u32>().is_ok() && i + 3 <= parts.len(),
            Some([_, "exe" | "cwd" | "root", ..]) => true,
            _ => false,
        };
        // parts[0] is "proc"; a magic link is at parts[1..].
        if parts.first() == Some(&"proc") && magic(1) {
            let link_len = if parts.get(2) == Some(&"fd") { 4 } else { 3 };
            let trailing = parts.len() == link_len;
            let followed = !trailing || flags & (open::O_NOFOLLOW | open::O_PATH) == 0;
            if followed {
                return Err(Errno::ELOOP);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_execveat(
        &mut self,
        tid: Tid,
        dirfd: u64,
        path_ptr: u64,
        argv: u64,
        envp: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> Flow {
        let r = (|| {
            if flags & !(at::AT_EMPTY_PATH | at::AT_SYMLINK_NOFOLLOW) != 0 {
                return Err(Errno::EINVAL);
            }
            let path = {
                let mm = self.mm(tid);
                let mm = mm.borrow();
                crate::uaccess::read_cstr_max(mm.mem.as_ref(), path_ptr, crate::vfs::PATH_MAX)?
            };
            if path.is_empty() {
                if flags & at::AT_EMPTY_PATH == 0 {
                    return Err(Errno::ENOENT);
                }
                let file = self.file(tid, dirfd)?;
                let ino = match file.borrow().kind {
                    FileKind::Inode { ino, .. } => ino,
                    _ => return Err(Errno::EACCES),
                };
                let name = file.borrow().path.clone();
                let shown = format!("/dev/fd/{}", dirfd as i32).into_bytes();
                return Ok((
                    shown,
                    ExecStart {
                        dir: ino,
                        follow: true,
                        file: Some((ino, name)),
                    },
                ));
            }
            let dir = if path.starts_with(b"/") {
                crate::vfs::ROOT
            } else {
                self.dir_ino(tid, dirfd)?
            };
            Ok((
                path,
                ExecStart {
                    dir,
                    follow: flags & at::AT_SYMLINK_NOFOLLOW == 0,
                    file: None,
                },
            ))
        })();
        match r {
            Ok((path, start)) => self.exec_with(tid, path, argv, envp, Some(start), host),
            Err(e) => Flow::Ret(e.ret()),
        }
    }

    // ---- readahead, sync_file_range, fallocate ----

    pub(super) fn sys_readahead(&mut self, tid: Tid, fd: u64) -> SysResult {
        let file = self.file(tid, fd)?;
        if !readable(&file.borrow()) {
            return Err(Errno::EBADF);
        }
        match file.borrow().kind {
            FileKind::Inode { ino, device: None }
                if matches!(self.vfs.inode(ino).node, Node::File(_)) =>
            {
                Ok(0)
            }
            _ => Err(Errno::EINVAL),
        }
    }

    pub(super) fn sys_sync_file_range(
        &mut self,
        tid: Tid,
        fd: u64,
        off: u64,
        nbytes: u64,
        flags: u64,
    ) -> SysResult {
        if flags & !7 != 0 || (off as i64) < 0 || (nbytes as i64) < 0 {
            return Err(Errno::EINVAL);
        }
        if off.checked_add(nbytes).is_none_or(|e| (e as i64) < 0) {
            return Err(Errno::EINVAL);
        }
        let file = self.file(tid, fd)?;
        match file.borrow().kind {
            FileKind::Inode { ino, device: None }
                if matches!(
                    self.vfs.inode(ino).node,
                    Node::File(_) | Node::Dir(_) | Node::Symlink(_)
                ) =>
            {
                Ok(0)
            }
            _ => Err(Errno::ESPIPE),
        }
    }

    /// `fallocate` as tmpfs implements it: allocation (mode 0, `KEEP_SIZE`) and punching
    /// holes; other modes are `EOPNOTSUPP`.
    pub(super) fn sys_fallocate(
        &mut self,
        tid: Tid,
        fd: u64,
        fmode: u64,
        off: u64,
        len: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if (off as i64) < 0 || (len as i64) <= 0 {
            return Err(Errno::EINVAL);
        }
        let all = FALLOC_FL_KEEP_SIZE
            | FALLOC_FL_PUNCH_HOLE
            | FALLOC_FL_COLLAPSE_RANGE
            | FALLOC_FL_ZERO_RANGE
            | FALLOC_FL_INSERT_RANGE
            | FALLOC_FL_UNSHARE_RANGE;
        if fmode & !all != 0 {
            return Err(Errno::EOPNOTSUPP);
        }
        let punch = fmode & FALLOC_FL_PUNCH_HOLE != 0;
        if punch && fmode & FALLOC_FL_ZERO_RANGE != 0 {
            return Err(Errno::EOPNOTSUPP);
        }
        if punch && fmode & FALLOC_FL_KEEP_SIZE == 0 {
            return Err(Errno::EOPNOTSUPP);
        }
        let exclusive = FALLOC_FL_COLLAPSE_RANGE | FALLOC_FL_INSERT_RANGE;
        if fmode & exclusive != 0
            && fmode != FALLOC_FL_COLLAPSE_RANGE
            && fmode != FALLOC_FL_INSERT_RANGE
        {
            return Err(Errno::EINVAL);
        }
        let file = self.file(tid, fd)?;
        if !writable(&file.borrow()) {
            return Err(Errno::EBADF);
        }
        let ino = match file.borrow().kind {
            FileKind::Pipe { .. } => return Err(Errno::ESPIPE),
            FileKind::Inode { ino, device: None } => ino,
            _ => return Err(Errno::ENODEV),
        };
        match self.vfs.inode(ino).node {
            Node::Dir(_) => return Err(Errno::EISDIR),
            Node::File(_) => {}
            _ => return Err(Errno::ENODEV),
        }
        let end = off.checked_add(len).ok_or(Errno::EFBIG)?;
        if (end as i64) < 0 {
            return Err(Errno::EFBIG);
        }
        if fmode & !(FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE) != 0 {
            return Err(Errno::EOPNOTSUPP);
        }
        let now = Timespec::from_nanos(host.realtime_ns());
        if punch {
            let seals = self.vfs.inode(ino).seals.unwrap_or(0);
            if seals
                & (wasmshim_linux_abi::consts::seal::F_SEAL_WRITE
                    | wasmshim_linux_abi::consts::seal::F_SEAL_FUTURE_WRITE)
                != 0
            {
                return Err(Errno::EPERM);
            }
            self.vfs.zero_range(ino, off, len, now)?;
            return Ok(0);
        }
        let grow_sealed = self.vfs.inode(ino).seals.unwrap_or(0)
            & wasmshim_linux_abi::consts::seal::F_SEAL_GROW
            != 0;
        if grow_sealed && end > self.vfs.size(ino) {
            return Err(Errno::EPERM);
        }
        if fmode & FALLOC_FL_KEEP_SIZE == 0 && end > self.vfs.size(ino) {
            self.vfs.truncate(ino, end, now)?;
        }
        Ok(0)
    }
}

/// The error for an xattr call on a special inode (pipe, socket, anonymous inode): Linux's
/// `xattr_permission` refuses `user.*` there first (`EPERM`/`ENODATA`); other namespaces
/// find no handler (`EOPNOTSUPP`).
fn special_xattr(name: &[u8], write: bool) -> Errno {
    if name.starts_with(b"user.") {
        if write { Errno::EPERM } else { Errno::ENODATA }
    } else {
        Errno::EOPNOTSUPP
    }
}

/// The four xattr operations.
#[derive(Clone, Copy, Debug)]
pub(super) enum XattrOp {
    Set,
    Get,
    List,
    Remove,
}
