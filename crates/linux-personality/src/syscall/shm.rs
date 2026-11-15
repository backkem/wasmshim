//! Shared memory objects with names and handles: `memfd_create` (and its seals) and SysV
//! shared memory (`shmget`, `shmat`, `shmdt`, `shmctl`). Both are memory objects
//! ([`crate::shm`]); `/dev/shm` needs nothing of its own (its files get a page
//! cache when first mapped shared, like any file).

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{ipc, memfd, open, seal};
use wasmshim_linux_abi::structs::Timespec;

use super::SysResult;
use crate::fd::{FileKind, FileRef, open_file};
use crate::host::Host;
use crate::kernel::{Kernel, Tid};
use crate::mm::{VmaKind, page_up};
use crate::shm::{IPC_PRIVATE, IpcPerm, SHM_DEST, SHMMAX, SHMMNI, ShmSegment};
use crate::uaccess::{read_bytes, read_cstr_max, write_bytes};
use crate::vfs::FileData;

/// `SHM_LOCKED` in `shm_perm.mode`.
const SHM_LOCKED: u32 = 0o2000;
/// Size of `struct shmid64_ds` on arm64.
const SHMID_DS_SIZE: usize = 112;

impl Kernel {
    /// `memfd_create`: an anonymous file in a memory object.
    pub(super) fn sys_memfd_create(
        &mut self,
        tid: Tid,
        name_ptr: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        const HUGE_SIZE_BITS: u64 = 0x3f << 26;
        let known = memfd::MFD_CLOEXEC
            | memfd::MFD_ALLOW_SEALING
            | memfd::MFD_HUGETLB
            | memfd::MFD_NOEXEC_SEAL
            | memfd::MFD_EXEC;
        let huge_size = flags & HUGE_SIZE_BITS;
        if flags & !(known | HUGE_SIZE_BITS) != 0
            || (huge_size != 0 && flags & memfd::MFD_HUGETLB == 0)
            || flags & (memfd::MFD_NOEXEC_SEAL | memfd::MFD_EXEC)
                == memfd::MFD_NOEXEC_SEAL | memfd::MFD_EXEC
        {
            return Err(Errno::EINVAL);
        }
        let name = {
            let mm = self.mm(tid);
            let mm = mm.borrow();
            match read_cstr_max(mm.mem.as_ref(), name_ptr, memfd::NAME_MAX + 1) {
                Ok(n) => n,
                Err(Errno::ENAMETOOLONG) => return Err(Errno::EINVAL),
                Err(e) => return Err(e),
            }
        };
        let noexec = flags & memfd::MFD_NOEXEC_SEAL != 0;
        let seals = if noexec {
            seal::F_SEAL_EXEC
        } else if flags & memfd::MFD_ALLOW_SEALING != 0 {
            0
        } else {
            seal::F_SEAL_SEAL
        };
        let mem = self.new_mem_object(None, None)?;
        let now = Timespec::from_nanos(host.realtime_ns());
        let perm = if noexec { 0o666 } else { 0o777 };
        let ino = self
            .vfs
            .create_unlinked(perm, FileData::Pages { mem, len: 0 }, now);
        self.vfs.set_seals(ino, Some(seals));
        let (uid, gid) = {
            let c = self.creds(tid);
            (c.fsuid, c.fsgid)
        };
        self.vfs.chown(ino, uid, gid, now);
        let mut path = b"/memfd:".to_vec();
        path.extend_from_slice(&name);
        path.extend_from_slice(b" (deleted)");
        let file = open_file(
            FileKind::Inode { ino, device: None },
            open::O_RDWR | open::O_LARGEFILE,
            &path,
        );
        self.vfs.retain(ino);
        let fd =
            self.fds(tid)
                .borrow_mut()
                .insert(file.clone(), flags & memfd::MFD_CLOEXEC != 0, 0);
        match fd {
            Ok(fd) => Ok(fd as u64),
            Err(e) => {
                self.release_file(&file);
                Err(e)
            }
        }
    }

    /// `fcntl(F_ADD_SEALS)`.
    pub(super) fn add_seals(&mut self, file: &FileRef, add: u64) -> SysResult {
        let (ino, writable) = {
            let f = file.borrow();
            let FileKind::Inode { ino, device: None } = f.kind else {
                return Err(Errno::EINVAL);
            };
            (ino, f.flags & open::O_ACCMODE != open::O_RDONLY)
        };
        let Some(seals) = self.vfs.inode(ino).seals else {
            return Err(Errno::EINVAL);
        };
        let add = u32::try_from(add).map_err(|_| Errno::EINVAL)?;
        if add & !seal::ALL != 0 {
            return Err(Errno::EINVAL);
        }
        if !writable || seals & seal::F_SEAL_SEAL != 0 {
            return Err(Errno::EPERM);
        }
        if add & seal::F_SEAL_WRITE != 0
            && seals & seal::F_SEAL_WRITE == 0
            && let Some(obj) = self.vfs.file_pages(ino)
            && self.writably_mapped(&obj)
        {
            return Err(Errno::EBUSY);
        }
        self.vfs.set_seals(ino, Some(seals | add));
        Ok(0)
    }

    /// `fcntl(F_GET_SEALS)`.
    pub(super) fn get_seals(&self, file: &FileRef) -> SysResult {
        match file.borrow().kind {
            FileKind::Inode { ino, device: None } => self
                .vfs
                .inode(ino)
                .seals
                .map(u64::from)
                .ok_or(Errno::EINVAL),
            _ => Err(Errno::EINVAL),
        }
    }

    // ---- SysV shared memory ----

    /// Attachments of the segment in slot `index` (one per mapping, over every address space,
    /// as Linux counts `vm_ops->open`).
    fn shm_nattch(&self, index: u32) -> u64 {
        let Some(seg) = self.shm.objs.get(&index) else {
            return 0;
        };
        let mut seen = std::collections::BTreeSet::new();
        let mut n = 0;
        for p in self.procs.values() {
            let mm = p.mm.borrow();
            if !seen.insert(mm.id) {
                continue;
            }
            n += mm
                .vmas()
                .filter(|v| matches!(&v.kind, VmaKind::Shared { obj, .. } if *obj == seg.obj))
                .count() as u64;
        }
        n
    }

    /// Frees removed segments nobody has attached any more.
    fn shm_collect(&mut self) {
        let doomed: Vec<u32> = self
            .shm
            .objs
            .iter()
            .filter(|(_, s)| s.perm.mode & SHM_DEST != 0)
            .map(|(&i, _)| i)
            .filter(|&i| self.shm_nattch(i) == 0)
            .collect();
        for i in doomed {
            self.shm.objs.remove(&i);
        }
    }

    pub(super) fn sys_shmget(
        &mut self,
        tid: Tid,
        key: u64,
        size: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        self.shm_collect();
        let key = key as i32;
        if key != IPC_PRIVATE
            && let Some(index) = self.shm.find_key(key)
        {
            if flags & ipc::IPC_CREAT != 0 && flags & ipc::IPC_EXCL != 0 {
                return Err(Errno::EEXIST);
            }
            if size > self.shm.objs[&index].size {
                return Err(Errno::EINVAL);
            }
            return self
                .shm
                .id_of(index)
                .map(|id| id as u64)
                .ok_or(Errno::EINVAL);
        }
        if key != IPC_PRIVATE && flags & ipc::IPC_CREAT == 0 {
            return Err(Errno::ENOENT);
        }
        if size == 0 || size > SHMMAX {
            return Err(Errno::EINVAL);
        }
        if self.shm.objs.len() as u64 >= SHMMNI {
            return Err(Errno::ENOSPC);
        }
        let obj = self.new_mem_object(None, Some(size))?;
        let (uid, gid) = {
            let c = self.creds(tid);
            (c.euid, c.egid)
        };
        let now = (host.realtime_ns() / 1_000_000_000) as i64;
        let id = self.shm.insert(ShmSegment {
            perm: IpcPerm::new(key, uid, gid, flags),
            size,
            obj,
            atime: 0,
            dtime: 0,
            ctime: now,
            cpid: self.pid_of(tid),
            lpid: 0,
        });
        id.map(|id| id as u64).ok_or(Errno::ENOSPC)
    }

    pub(super) fn sys_shmat(
        &mut self,
        tid: Tid,
        id: u64,
        addr: u64,
        flags: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        self.shm_collect();
        let index = self.shm.index_of(id as i32).ok_or(Errno::EINVAL)?;
        let seg = self.shm.objs[&index].clone();
        let len = page_up(seg.size);
        let mut perms = if flags & ipc::SHM_RDONLY != 0 {
            wasmshim_cpu::Perms::R
        } else {
            wasmshim_cpu::Perms::RW
        };
        if flags & ipc::SHM_EXEC != 0 {
            perms = perms | wasmshim_cpu::Perms::X;
        }
        let mm = self.mm(tid);
        let mut mm = mm.borrow_mut();
        let start = if addr == 0 {
            if flags & ipc::SHM_REMAP != 0 {
                return Err(Errno::EINVAL);
            }
            mm.find_free(len).ok_or(Errno::ENOMEM)?
        } else {
            let a = if flags & ipc::SHM_RND != 0 {
                addr & !(ipc::SHMLBA - 1)
            } else if addr & (ipc::SHMLBA - 1) != 0 {
                return Err(Errno::EINVAL);
            } else {
                addr
            };
            let end = a.checked_add(len).ok_or(Errno::EINVAL)?;
            if a == 0 || end > mm.mem.address_limit() {
                return Err(Errno::EINVAL);
            }
            if flags & ipc::SHM_REMAP == 0 && !mm.is_free(a, end) {
                return Err(Errno::EINVAL);
            }
            a
        };
        let name = format!("/SYSV{:08x} (deleted)", seg.perm.key as u32).into_bytes();
        mm.map_shared(
            start,
            start + len,
            perms,
            &seg.obj,
            0,
            name,
            flags & ipc::SHM_RDONLY == 0,
        )?;
        drop(mm);
        let pid = self.pid_of(tid);
        if let Some(s) = self.shm.objs.get_mut(&index) {
            s.atime = (host.realtime_ns() / 1_000_000_000) as i64;
            s.lpid = pid;
        }
        Ok(start)
    }

    pub(super) fn sys_shmdt(&mut self, tid: Tid, addr: u64, host: &mut dyn Host) -> SysResult {
        let mm = self.mm(tid);
        let index = {
            let mm = mm.borrow();
            let v = mm.vma_at(addr).ok_or(Errno::EINVAL)?;
            match &v.kind {
                VmaKind::Shared { obj, offset, .. } if v.start == addr && *offset == 0 => {
                    self.shm.by_object(obj.id).ok_or(Errno::EINVAL)?
                }
                _ => return Err(Errno::EINVAL),
            }
        };
        let seg = self.shm.objs[&index].clone();
        let end = addr.saturating_add(page_up(seg.size));
        {
            let mut mm = mm.borrow_mut();
            let pieces: Vec<(u64, u64)> = mm
                .vmas()
                .filter(|v| v.start >= addr && v.start < end)
                .filter(|v| matches!(&v.kind, VmaKind::Shared { obj, .. } if *obj == seg.obj))
                .map(|v| (v.start, v.end))
                .collect();
            for (s, e) in pieces {
                mm.unmap(s, e)?;
            }
        }
        let pid = self.pid_of(tid);
        if let Some(s) = self.shm.objs.get_mut(&index) {
            s.dtime = (host.realtime_ns() / 1_000_000_000) as i64;
            s.lpid = pid;
        }
        self.shm_collect();
        Ok(0)
    }

    /// `struct shmid64_ds` of the segment in slot `index`.
    fn shmid_ds(&self, index: u32) -> Vec<u8> {
        let s = &self.shm.objs[&index];
        let mut b = s.perm.encode();
        for v in [s.size, s.atime as u64, s.dtime as u64, s.ctime as u64] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&s.cpid.to_le_bytes());
        b.extend_from_slice(&s.lpid.to_le_bytes());
        b.extend_from_slice(&self.shm_nattch(index).to_le_bytes());
        b.resize(SHMID_DS_SIZE, 0);
        b
    }

    /// Highest slot in use (`IPC_INFO`, `SHM_INFO`), 0 with none.
    fn shm_max_index(&self) -> u64 {
        self.shm
            .objs
            .keys()
            .next_back()
            .map_or(0, |&i| u64::from(i))
    }

    pub(super) fn sys_shmctl(
        &mut self,
        tid: Tid,
        id: u64,
        cmd: u64,
        buf: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        self.shm_collect();
        let cmd = cmd & !ipc::IPC_64;
        let mm = self.mm(tid);
        let put = |data: &[u8]| write_bytes(mm.borrow_mut().mem.as_mut(), buf, data);
        match cmd {
            ipc::IPC_INFO => {
                let mut b = Vec::new();
                for v in [SHMMAX, 1, SHMMNI, SHMMNI, SHMMAX, 0, 0, 0, 0] {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                put(&b)?;
                return Ok(self.shm_max_index());
            }
            ipc::SHM_INFO => {
                let pages: u64 = self.shm.objs.values().map(|s| page_up(s.size) / 4096).sum();
                let mut b = Vec::new();
                b.extend_from_slice(&(self.shm.objs.len() as u32).to_le_bytes());
                b.resize(8, 0);
                for v in [pages, pages, 0, 0, 0] {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                put(&b)?;
                return Ok(self.shm_max_index());
            }
            ipc::SHM_STAT | ipc::SHM_STAT_ANY => {
                let index = u32::try_from(id).map_err(|_| Errno::EINVAL)?;
                if !self.shm.objs.contains_key(&index) {
                    return Err(Errno::EINVAL);
                }
                put(&self.shmid_ds(index))?;
                return self.shm.id_of(index).map(|i| i as u64).ok_or(Errno::EINVAL);
            }
            _ => {}
        }
        let index = self.shm.index_of(id as i32).ok_or(Errno::EINVAL)?;
        let now = (host.realtime_ns() / 1_000_000_000) as i64;
        match cmd {
            ipc::IPC_STAT => {
                put(&self.shmid_ds(index))?;
                Ok(0)
            }
            ipc::IPC_SET => {
                let b = read_bytes(mm.borrow_mut().mem.as_mut(), buf, 24)?;
                if let Some(s) = self.shm.objs.get_mut(&index) {
                    s.perm.set_from(&b);
                    s.ctime = now;
                }
                Ok(0)
            }
            ipc::IPC_RMID => {
                if let Some(s) = self.shm.objs.get_mut(&index) {
                    s.perm.mode |= SHM_DEST;
                    s.perm.key = IPC_PRIVATE;
                }
                self.shm_collect();
                Ok(0)
            }
            ipc::SHM_LOCK | ipc::SHM_UNLOCK => {
                if let Some(s) = self.shm.objs.get_mut(&index) {
                    if cmd == ipc::SHM_LOCK {
                        s.perm.mode |= SHM_LOCKED;
                    } else {
                        s.perm.mode &= !SHM_LOCKED;
                    }
                }
                Ok(0)
            }
            _ => Err(Errno::EINVAL),
        }
    }
}
