//! Memory-management syscalls: `brk`, `mmap`, `munmap`, `mprotect`, `mremap`, `madvise`,
//! `msync`.
//!
//! `MAP_SHARED` mappings map memory objects ([`crate::shm`]): shared anonymous
//! memory gets a fresh object, a file its page cache. Private file mappings are copies of the
//! file's bytes at the time of the call, filled page by page on first touch.

use std::rc::Rc;

use wasmshim_cpu::{PAGE_SIZE, PageSource, Perms};
use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{madvise, map, mremap, msync, open, prot, seal};

use super::SysResult;
use crate::fd::FileKind;
use crate::host::Host;
use crate::kernel::{Kernel, Tid};
use crate::mm::{AddressSpace, MMAP_MIN, Vma, VmaKind, page_up};
use crate::shm::{MemObject, MemRef};
use crate::vfs::{Device, Ino};

const PAGE_MASK: u64 = PAGE_SIZE - 1;

fn perms(p: u64) -> Result<Perms, Errno> {
    if p & !(prot::PROT_READ | prot::PROT_WRITE | prot::PROT_EXEC) != 0 {
        return Err(Errno::EINVAL);
    }
    // PROT_WRITE implies readable on AArch64 (and so does PROT_EXEC for user pages).
    let w = p & prot::PROT_WRITE != 0;
    let x = p & prot::PROT_EXEC != 0;
    Ok(Perms::new(p & prot::PROT_READ != 0 || w, w, x))
}

/// Copies `len` bytes of page contents from `src` to `dst` (both mapped), ignoring permissions.
fn copy_pages(mm: &mut AddressSpace, src: u64, dst: u64, len: u64) -> Result<(), Errno> {
    let mut buf = vec![0u8; PAGE_SIZE as usize];
    let mut off = 0;
    while off < len {
        let n = (len - off).min(PAGE_SIZE) as usize;
        let page_perms = mm.mem.perms(src + off).ok_or(Errno::EFAULT)?;
        if page_perms.read() {
            mm.mem
                .read(src + off, &mut buf[..n])
                .map_err(|_| Errno::EFAULT)?;
        } else {
            // Unreadable source pages (PROT_NONE) are temporarily opened up to copy them.
            let page = (src + off) & !PAGE_MASK;
            mm.mem
                .protect(page, PAGE_SIZE, Perms::R)
                .map_err(|_| Errno::EFAULT)?;
            let r = mm.mem.read(src + off, &mut buf[..n]);
            let _ = mm.mem.protect(page, PAGE_SIZE, page_perms);
            r.map_err(|_| Errno::EFAULT)?;
        }
        mm.mem
            .poke(dst + off, &buf[..n])
            .map_err(|_| Errno::EFAULT)?;
        off += n as u64;
    }
    Ok(())
}

/// What an `mmap` maps.
enum Backing {
    /// Anonymous memory (also `/dev/zero`).
    Anon,
    /// A framebuffer (`/dev/fb<n>`): its screen memory, private or shared alike
    /// (as `fb_mmap` maps the device memory whatever the flags), and its size.
    Fb { n: u32, path: Vec<u8> },
    /// A regular file: its inode, the path `/proc/<pid>/maps` shows, and whether a shared
    /// mapping of it may ever be writable (`VM_MAYWRITE`).
    File {
        ino: Ino,
        path: Vec<u8>,
        may_write: bool,
    },
}

impl Kernel {
    /// A new memory object from the frontend (its pages start as `source`'s, if any).
    pub(crate) fn new_mem_object(
        &mut self,
        source: Option<(Rc<dyn PageSource>, u64)>,
        size: Option<u64>,
    ) -> Result<MemRef, Errno> {
        let pages = self
            .frontend
            .new_shared_memory(source)
            .map_err(|_| Errno::EIO)?;
        Ok(Rc::new(MemObject {
            id: self.new_object_id(),
            pages,
            size,
        }))
    }

    /// The memory object of regular file `ino`: its page cache, made now if it has none (the
    /// file's current bytes become the object's, read page by page as they are touched).
    pub(crate) fn file_object(&mut self, ino: Ino) -> Result<MemRef, Errno> {
        if let Some(obj) = self.vfs.file_pages(ino) {
            return Ok(obj);
        }
        let contents = self.vfs.file_contents(ino)?;
        let size = contents.len();
        let obj = self.new_mem_object(
            Some((contents.pages(0, size), page_up(size) / PAGE_SIZE)),
            None,
        )?;
        self.vfs.set_file_pages(ino, obj.clone())?;
        Ok(obj)
    }

    /// `true` if some address space maps `obj` shared with `may_write` (Linux's
    /// `mapping_writably_mapped`, for `F_SEAL_WRITE`).
    pub(crate) fn writably_mapped(&self, obj: &MemRef) -> bool {
        let mut seen = std::collections::BTreeSet::new();
        self.procs.values().any(|p| {
            let mm = p.mm.borrow();
            seen.insert(mm.id)
                && mm.vmas().any(|v| {
                    matches!(&v.kind, VmaKind::Shared { obj: o, may_write: true, .. } if o == obj)
                })
        })
    }

    pub(super) fn sys_brk(&mut self, tid: Tid, addr: u64) -> SysResult {
        Ok(self.mm(tid).borrow_mut().set_brk(addr))
    }

    /// What `fd` maps, after Linux's access checks.
    fn mmap_backing(&self, tid: Tid, fd: u64, shared: bool, prot_: u64) -> Result<Backing, Errno> {
        let file = self.file(tid, fd)?;
        let f = file.borrow();
        let ino = match f.kind {
            FileKind::Inode {
                device: Some(Device::Zero),
                ..
            } => return Ok(Backing::Anon),
            FileKind::Inode {
                device: Some(Device::Fb(n)),
                ..
            } => {
                if prot_ & prot::PROT_WRITE != 0 && f.flags & open::O_ACCMODE != open::O_RDWR {
                    return Err(Errno::EACCES);
                }
                return Ok(Backing::Fb {
                    n,
                    path: f.path.clone(),
                });
            }
            FileKind::Inode { ino, device: None } if !self.vfs.inode(ino).is_dir() => ino,
            _ => return Err(Errno::ENODEV),
        };
        let acc = f.flags & open::O_ACCMODE;
        if acc == open::O_WRONLY {
            return Err(Errno::EACCES);
        }
        let seals = self.vfs.inode(ino).seals.unwrap_or(0);
        let write_sealed = seals & (seal::F_SEAL_WRITE | seal::F_SEAL_FUTURE_WRITE) != 0;
        let writable = acc == open::O_RDWR && f.flags & open::O_APPEND == 0;
        if shared && prot_ & prot::PROT_WRITE != 0 {
            if !writable {
                return Err(Errno::EACCES);
            }
            if write_sealed {
                return Err(Errno::EPERM);
            }
        }
        Ok(Backing::File {
            ino,
            path: f.path.clone(),
            may_write: writable && !write_sealed,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_mmap(
        &mut self,
        tid: Tid,
        addr: u64,
        len: u64,
        prot_: u64,
        flags: u64,
        fd: u64,
        off: u64,
        _host: &mut dyn Host,
    ) -> SysResult {
        if len == 0 || off & PAGE_MASK != 0 {
            return Err(Errno::EINVAL);
        }
        let len = page_up(len);
        if len == 0 || len > 1 << 47 {
            return Err(Errno::ENOMEM);
        }
        let ty = flags & map::MAP_TYPE;
        if !matches!(
            ty,
            map::MAP_PRIVATE | map::MAP_SHARED | map::MAP_SHARED_VALIDATE
        ) {
            return Err(Errno::EINVAL);
        }
        let shared = ty != map::MAP_PRIVATE;
        let perms = perms(prot_)?;
        let backing = if flags & map::MAP_ANONYMOUS != 0 {
            Backing::Anon
        } else {
            self.mmap_backing(tid, fd, shared, prot_)?
        };
        // mmap(2), EOVERFLOW: a file mapping must end within the largest file offset
        // (mm/mmap.c, `file_mmap_ok`: `MAX_LFS_FILESIZE`).
        if matches!(backing, Backing::File { .. })
            && off.checked_add(len).is_none_or(|end| end > i64::MAX as u64)
        {
            return Err(Errno::EOVERFLOW);
        }
        // Objects first (they need `self`), then the address space.
        let object = match &backing {
            Backing::Anon if shared => Some(self.new_mem_object(None, Some(len))?),
            Backing::File { ino, .. } if shared => Some(self.file_object(*ino)?),
            Backing::Fb { n, .. } => {
                let (obj, size) = self.fb_object(*n)?;
                if off.checked_add(len).is_none_or(|end| end > size) {
                    return Err(Errno::EINVAL);
                }
                Some(obj)
            }
            _ => None,
        };
        let private_file = match &backing {
            Backing::File { ino, .. } if !shared => Some(self.vfs.file_contents(*ino)?),
            _ => None,
        };

        let mm_rc = self.mm(tid);
        let mut mm = mm_rc.borrow_mut();
        let fixed = flags & (map::MAP_FIXED | map::MAP_FIXED_NOREPLACE) != 0;
        let start = if fixed {
            if addr & PAGE_MASK != 0 || addr < MMAP_MIN {
                return Err(if addr & PAGE_MASK != 0 {
                    Errno::EINVAL
                } else {
                    Errno::EPERM
                });
            }
            let end = addr.checked_add(len).ok_or(Errno::ENOMEM)?;
            if end > mm.mem.address_limit() {
                return Err(Errno::ENOMEM);
            }
            if flags & map::MAP_FIXED_NOREPLACE != 0 && !mm.is_free(addr, end) {
                return Err(Errno::EEXIST);
            }
            addr
        } else {
            let hint = addr & !PAGE_MASK;
            if hint >= MMAP_MIN
                && hint
                    .checked_add(len)
                    .is_some_and(|e| e <= mm.mem.address_limit())
                && mm.is_free(hint, hint + len)
            {
                hint
            } else {
                mm.find_free(len).ok_or(Errno::ENOMEM)?
            }
        };
        let end = start + len;
        match (backing, object) {
            (Backing::Anon, Some(obj)) => {
                mm.map_shared(
                    start,
                    end,
                    perms,
                    &obj,
                    0,
                    b"/dev/zero (deleted)".to_vec(),
                    true,
                )?;
            }
            (Backing::Anon, None) => mm.map(start, end, perms, VmaKind::Anon)?,
            (Backing::Fb { path, .. }, Some(obj)) => {
                mm.map_shared(start, end, perms, &obj, off, path, true)?
            }
            (Backing::Fb { .. }, None) => return Err(Errno::ENODEV),
            (
                Backing::File {
                    path, may_write, ..
                },
                Some(obj),
            ) => mm.map_shared(start, end, perms, &obj, off, path, may_write)?,
            (Backing::File { path, .. }, None) => {
                mm.map(start, end, perms, VmaKind::File { path, offset: off })?;
                if let Some(contents) = private_file {
                    let size = contents.len();
                    if off < size {
                        // Pages holding file bytes come from the file; the rest stay zero.
                        let pages = page_up(size - off).min(len);
                        mm.mem
                            .map_source(start, pages, perms, contents.pages(off, size), 0)
                            .map_err(|_| Errno::ENOMEM)?;
                    }
                }
            }
        }
        Ok(start)
    }

    pub(super) fn sys_munmap(
        &mut self,
        tid: Tid,
        addr: u64,
        len: u64,
        _host: &mut dyn Host,
    ) -> SysResult {
        if addr & PAGE_MASK != 0 || len == 0 {
            return Err(Errno::EINVAL);
        }
        let end = addr.checked_add(page_up(len)).ok_or(Errno::EINVAL)?;
        self.mm(tid).borrow_mut().unmap(addr, end)?;
        Ok(0)
    }

    pub(super) fn sys_mprotect(&mut self, tid: Tid, addr: u64, len: u64, prot_: u64) -> SysResult {
        if addr & PAGE_MASK != 0 {
            return Err(Errno::EINVAL);
        }
        // PROT_GROWSDOWN/PROT_GROWSUP are accepted and ignored.
        let perms = perms(prot_ & 0x7)?;
        if len == 0 {
            return Ok(0);
        }
        let end = addr.checked_add(page_up(len)).ok_or(Errno::ENOMEM)?;
        let mm = self.mm(tid);
        let mut mm = mm.borrow_mut();
        if perms.write()
            && mm.vmas().any(|v| {
                v.start < end
                    && addr < v.end
                    && matches!(
                        v.kind,
                        VmaKind::Shared {
                            may_write: false,
                            ..
                        }
                    )
            })
        {
            // A shared mapping of a file opened read-only (or write-sealed).
            return Err(Errno::EACCES);
        }
        mm.protect(addr, end, perms)?;
        Ok(0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_mremap(
        &mut self,
        tid: Tid,
        old: u64,
        old_len: u64,
        new_len: u64,
        flags: u64,
        new_addr: u64,
        _host: &mut dyn Host,
    ) -> SysResult {
        if old & PAGE_MASK != 0
            || new_len == 0
            || flags & !(mremap::MREMAP_MAYMOVE | mremap::MREMAP_FIXED) != 0
        {
            return Err(Errno::EINVAL);
        }
        if flags & mremap::MREMAP_FIXED != 0 {
            // MREMAP_FIXED is not implemented.
            return Err(Errno::EINVAL);
        }
        let _ = new_addr;
        let old_len = page_up(old_len);
        let new_len = page_up(new_len);
        let mm = self.mm(tid);
        let mut mm = mm.borrow_mut();
        let first = mm.vma_at(old).cloned().ok_or(Errno::EFAULT)?;
        if old_len == 0 {
            // A second mapping of the same shared pages.
            let VmaKind::Shared {
                obj,
                offset,
                name,
                may_write,
            } = &first.kind
            else {
                return Err(Errno::EINVAL);
            };
            if flags & mremap::MREMAP_MAYMOVE == 0 {
                return Err(Errno::EINVAL);
            }
            let dst = mm.find_free(new_len).ok_or(Errno::ENOMEM)?;
            let at = offset + (old - first.start);
            mm.map_shared(
                dst,
                dst + new_len,
                first.perms,
                obj,
                at,
                name.clone(),
                *may_write,
            )?;
            return Ok(dst);
        }
        let old_end = old.checked_add(old_len).ok_or(Errno::EFAULT)?;
        if !mm.covered(old, old_end) {
            return Err(Errno::EFAULT);
        }
        let shared = matches!(first.kind, VmaKind::Shared { .. });
        if shared && first.end < old_end {
            // Linux moves one mapping at a time.
            return Err(Errno::EFAULT);
        }
        if new_len <= old_len {
            if new_len < old_len {
                mm.unmap(old + new_len, old_end)?;
            }
            return Ok(old);
        }
        let grow_end = old.checked_add(new_len).ok_or(Errno::ENOMEM)?;
        if grow_end <= mm.mem.address_limit() && mm.is_free(old_end, grow_end) {
            Self::map_like(&mut mm, &first, old_end, grow_end, old_end - first.start)?;
            return Ok(old);
        }
        if flags & mremap::MREMAP_MAYMOVE == 0 {
            return Err(Errno::ENOMEM);
        }
        let dst = mm.find_free(new_len).ok_or(Errno::ENOMEM)?;
        if shared {
            // The same pages at a new place.
            Self::map_like(&mut mm, &first, dst, dst + new_len, old - first.start)?;
            mm.unmap(old, old_end)?;
            return Ok(dst);
        }
        mm.map(dst, dst + new_len, first.perms, first.kind.clone())?;
        copy_pages(&mut mm, old, dst, old_len)?;
        // Keep each copied page's own permissions.
        let pieces: Vec<_> = mm
            .vmas()
            .filter(|v| v.end > old && v.start < old_end)
            .cloned()
            .collect();
        for v in pieces {
            let s = v.start.max(old);
            let e = v.end.min(old_end);
            if v.perms != first.perms {
                mm.protect(dst + (s - old), dst + (e - old), v.perms)?;
            }
        }
        mm.unmap(old, old_end)?;
        Ok(dst)
    }

    /// Maps `[start, end)` like `like` continued: `skip` bytes past `like.start`'s offset for
    /// file and shared mappings (an `mremap` growing or moving it).
    fn map_like(
        mm: &mut AddressSpace,
        like: &Vma,
        start: u64,
        end: u64,
        skip: u64,
    ) -> Result<(), Errno> {
        match &like.kind {
            VmaKind::Shared {
                obj,
                offset,
                name,
                may_write,
            } => mm.map_shared(
                start,
                end,
                like.perms,
                obj,
                offset + skip,
                name.clone(),
                *may_write,
            ),
            VmaKind::Heap => mm.map(start, end, like.perms, VmaKind::Anon),
            VmaKind::File { path, offset } => mm.map(
                start,
                end,
                like.perms,
                VmaKind::File {
                    path: path.clone(),
                    offset: offset + skip,
                },
            ),
            kind => mm.map(start, end, like.perms, kind.clone()),
        }
    }

    pub(super) fn sys_madvise(&mut self, tid: Tid, addr: u64, len: u64, advice: u64) -> SysResult {
        // madvise(2), EINVAL "advice is not valid" (mm/madvise.c, `madvise_behavior_valid`,
        // checked first): MADV_NORMAL to MADV_DONTNEED (0-4), MADV_FREE to MADV_COLLAPSE
        // (8-25) and MADV_HWPOISON to MADV_GUARD_REMOVE (100-103). The ones not handled below
        // are hints with no effect here.
        if !matches!(advice, 0..=4 | 8..=25 | 100..=103) {
            return Err(Errno::EINVAL);
        }
        if addr & PAGE_MASK != 0 {
            return Err(Errno::EINVAL);
        }
        let end = addr.checked_add(page_up(len)).ok_or(Errno::EINVAL)?;
        let mm = self.mm(tid);
        let mut mm = mm.borrow_mut();
        if !mm.covered(addr, end) {
            return Err(Errno::ENOMEM);
        }
        let pieces: Vec<_> = mm
            .vmas()
            .filter(|v| v.end > addr && v.start < end)
            .cloned()
            .collect();
        match advice {
            madvise::MADV_DONTNEED | madvise::MADV_FREE => {
                // Anonymous private memory reads back as zeros afterwards; shared memory keeps
                // its contents (they live in the object).
                for v in pieces {
                    let s = v.start.max(addr);
                    let e = v.end.min(end);
                    // The vDSO page faults back in from its special mapping, unchanged.
                    if matches!(v.kind, VmaKind::Shared { .. } | VmaKind::Vdso) {
                        continue;
                    }
                    mm.mem.map(s, e - s, v.perms).map_err(|_| Errno::EINVAL)?;
                }
            }
            madvise::MADV_REMOVE => {
                // Punches a hole in the objects of shared, writable mappings.
                if pieces
                    .iter()
                    .any(|v| !matches!(v.kind, VmaKind::Shared { .. }))
                {
                    return Err(Errno::EINVAL);
                }
                if pieces.iter().any(|v| !v.perms.write()) {
                    return Err(Errno::EACCES);
                }
                let zeros = vec![0u8; PAGE_SIZE as usize];
                for v in pieces {
                    let VmaKind::Shared { obj, offset, .. } = &v.kind else {
                        continue;
                    };
                    let mut at = v.start.max(addr);
                    while at < v.end.min(end) {
                        obj.pages
                            .write(offset + (at - v.start), &zeros)
                            .map_err(|_| Errno::EIO)?;
                        at += PAGE_SIZE;
                    }
                }
            }
            _ => {}
        }
        Ok(0)
    }

    /// `msync`: shared mappings are their objects, so there is nothing to write back; only
    /// the arguments are checked.
    pub(super) fn sys_msync(
        &mut self,
        tid: Tid,
        addr: u64,
        len: u64,
        flags: u64,
        _host: &mut dyn Host,
    ) -> SysResult {
        if addr & PAGE_MASK != 0
            || flags & !(msync::MS_ASYNC | msync::MS_INVALIDATE | msync::MS_SYNC) != 0
            || flags & (msync::MS_ASYNC | msync::MS_SYNC) == msync::MS_ASYNC | msync::MS_SYNC
        {
            return Err(Errno::EINVAL);
        }
        let end = addr.checked_add(page_up(len)).ok_or(Errno::ENOMEM)?;
        if !self.mm(tid).borrow().covered(addr, end) {
            return Err(Errno::ENOMEM);
        }
        Ok(0)
    }
}
