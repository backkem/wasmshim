//! inotify, synthesised from the VFS (for file watchers such as Vite's and chokidar's).
//!
//! The VFS records [`FsEvent`]s while anyone watches (`Vfs::watch_events`); the kernel turns
//! them into `struct inotify_event`s after every syscall ([`Kernel::dispatch_fs_events`]) and
//! adds the open/close/access events it sees itself. Everything happens inside one personality,
//! so events are exact and deterministic: a write by any process reaches every watcher before
//! the writing syscall returns.

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::{Rc, Weak};

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::open;

use super::{EPOLLIN, EPOLLRDNORM};
use crate::fd::{FileKind, open_file};
use crate::kernel::{Kernel, Tid, WaitKey};
use crate::syscall::SysResult;
use crate::vfs::{FsEvent, Ino, fsmask::*};

/// `/proc/sys/fs/inotify/max_queued_events`.
const MAX_QUEUED: usize = 16384;
/// `/proc/sys/fs/inotify/max_user_watches` (per instance here).
const MAX_WATCHES: usize = 65536;
const IN_CLOEXEC: u64 = open::O_CLOEXEC;
const IN_NONBLOCK: u64 = open::O_NONBLOCK;

/// One inotify instance.
#[derive(Debug)]
pub(crate) struct Inotify {
    pub id: u64,
    pub generation: u64,
    pub(crate) next_wd: i32,
    /// wd -> (inode, mask).
    pub(crate) watches: BTreeMap<i32, (Ino, u32)>,
    /// Encoded `struct inotify_event`s (name padded to 16 bytes), oldest first.
    pub(crate) queue: VecDeque<Vec<u8>>,
    pub(crate) overflowed: bool,
}

/// Shared handle.
pub(crate) type InotifyRef = Rc<RefCell<Inotify>>;

fn encode(wd: i32, mask: u32, cookie: u32, name: &[u8]) -> Vec<u8> {
    let len = if name.is_empty() {
        0
    } else {
        (name.len() + 1).div_ceil(16) * 16
    };
    let mut e = Vec::with_capacity(16 + len);
    e.extend_from_slice(&wd.to_le_bytes());
    e.extend_from_slice(&mask.to_le_bytes());
    e.extend_from_slice(&cookie.to_le_bytes());
    e.extend_from_slice(&(len as u32).to_le_bytes());
    e.extend_from_slice(name);
    e.resize(16 + len, 0);
    e
}

impl Inotify {
    pub(crate) fn events(&self) -> (u32, u64) {
        let ev = if self.queue.is_empty() {
            0
        } else {
            EPOLLIN | EPOLLRDNORM
        };
        (ev, self.generation)
    }

    /// Bytes waiting (`FIONREAD`).
    pub(crate) fn pending_bytes(&self) -> usize {
        self.queue.iter().map(Vec::len).sum()
    }

    /// Queues one event, merging it with an identical last one (as Linux does).
    fn push(&mut self, wd: i32, mask: u32, cookie: u32, name: &[u8]) {
        let e = encode(wd, mask, cookie, name);
        if self.queue.back() == Some(&e) {
            return;
        }
        if self.queue.len() >= MAX_QUEUED {
            if !self.overflowed {
                self.overflowed = true;
                self.queue.push_back(encode(-1, IN_Q_OVERFLOW, 0, b""));
                self.generation += 1;
            }
            return;
        }
        self.queue.push_back(e);
        self.generation += 1;
    }
}

impl Kernel {
    fn inotify_of(&self, tid: Tid, fd: u64) -> Result<InotifyRef, Errno> {
        let file = self.file(tid, fd)?;
        let f = file.borrow();
        match &f.kind {
            FileKind::Inotify(i) => Ok(i.clone()),
            _ => Err(Errno::EINVAL),
        }
    }

    /// `inotify_init1(flags)`.
    pub(crate) fn sys_inotify_init1(&mut self, tid: Tid, flags: u64) -> SysResult {
        if flags & !(IN_CLOEXEC | IN_NONBLOCK) != 0 {
            return Err(Errno::EINVAL);
        }
        let id = self.new_object_id();
        let i = Rc::new(RefCell::new(Inotify {
            id,
            generation: 0,
            next_wd: 1,
            watches: BTreeMap::new(),
            queue: VecDeque::new(),
            overflowed: false,
        }));
        let file = open_file(
            FileKind::Inotify(i),
            open::O_RDONLY | (flags & IN_NONBLOCK),
            b"anon_inode:inotify",
        );
        Ok(self
            .fds(tid)
            .borrow_mut()
            .insert(file, flags & IN_CLOEXEC != 0, 0)? as u64)
    }

    /// `inotify_add_watch(fd, path, mask)`.
    pub(crate) fn sys_inotify_add_watch(
        &mut self,
        tid: Tid,
        fd: u64,
        path: u64,
        mask: u64,
    ) -> SysResult {
        let mask = mask as u32;
        let inotify = self.inotify_of(tid, fd)?;
        if mask & IN_ALL_EVENTS == 0 {
            return Err(Errno::EINVAL);
        }
        if mask & IN_MASK_ADD != 0 && mask & IN_MASK_CREATE != 0 {
            return Err(Errno::EINVAL);
        }
        let flags = if mask & IN_DONT_FOLLOW != 0 {
            wasmshim_linux_abi::consts::at::AT_SYMLINK_NOFOLLOW
        } else {
            0
        };
        let ino = self
            .lookup_at(
                tid,
                wasmshim_linux_abi::consts::at::AT_FDCWD as u64,
                path,
                flags,
            )?
            .ok_or(Errno::ENOENT)?;
        if mask & IN_ONLYDIR != 0 && !self.vfs.inode(ino).is_dir() {
            return Err(Errno::ENOTDIR);
        }
        let wanted = mask & (IN_ALL_EVENTS | IN_ONESHOT | IN_EXCL_UNLINK);
        let mut i = inotify.borrow_mut();
        let existing = i
            .watches
            .iter()
            .find(|(_, (w, _))| *w == ino)
            .map(|(&wd, _)| wd);
        let wd = match existing {
            Some(_) if mask & IN_MASK_CREATE != 0 => return Err(Errno::EEXIST),
            Some(wd) => {
                let entry = i.watches.get_mut(&wd).expect("watch");
                entry.1 = if mask & IN_MASK_ADD != 0 {
                    entry.1 | wanted
                } else {
                    wanted
                };
                wd
            }
            None => {
                if i.watches.len() >= MAX_WATCHES {
                    return Err(Errno::ENOSPC);
                }
                let wd = i.next_wd;
                i.next_wd += 1;
                i.watches.insert(wd, (ino, wanted));
                self.inotify_watches
                    .entry(ino)
                    .or_default()
                    .push((Rc::downgrade(&inotify), wd));
                wd
            }
        };
        drop(i);
        self.vfs.watch_events();
        Ok(wd as u64)
    }

    /// `inotify_rm_watch(fd, wd)`.
    pub(crate) fn sys_inotify_rm_watch(&mut self, tid: Tid, fd: u64, wd: u64) -> SysResult {
        let inotify = self.inotify_of(tid, fd)?;
        let wd = wd as i32;
        let id = {
            let mut i = inotify.borrow_mut();
            let (ino, _) = i.watches.remove(&wd).ok_or(Errno::EINVAL)?;
            self.forget_watch(ino, &inotify, wd);
            i.push(wd, IN_IGNORED, 0, b"");
            i.id
        };
        self.object_changed(id);
        Ok(0)
    }

    fn forget_watch(&mut self, ino: Ino, inotify: &InotifyRef, wd: i32) {
        if let Some(list) = self.inotify_watches.get_mut(&ino) {
            list.retain(|(w, d)| {
                !(*d == wd && w.upgrade().is_some_and(|i| Rc::ptr_eq(&i, inotify)))
            });
            if list.is_empty() {
                self.inotify_watches.remove(&ino);
            }
        }
    }

    /// `read` on an inotify descriptor: whole events only.
    pub(crate) fn inotify_read(
        &mut self,
        inotify: &InotifyRef,
        buf: &mut [u8],
        nonblock: bool,
    ) -> Result<usize, Errno> {
        let mut i = inotify.borrow_mut();
        let Some(first) = i.queue.front() else {
            if nonblock {
                return Err(Errno::EAGAIN);
            }
            let id = i.id;
            drop(i);
            return Err(self.wait_for(WaitKey::Object(id), None, None));
        };
        if first.len() > buf.len() {
            return Err(Errno::EINVAL);
        }
        let mut n = 0;
        while let Some(e) = i.queue.front() {
            if n + e.len() > buf.len() {
                break;
            }
            buf[n..n + e.len()].copy_from_slice(e);
            n += e.len();
            i.queue.pop_front();
        }
        if i.queue.is_empty() {
            i.overflowed = false;
        }
        i.generation += 1;
        Ok(n)
    }

    /// Delivers the VFS's recorded events (plus `extra`) to every interested watch.
    pub(crate) fn dispatch_fs_events(&mut self) {
        if !self.vfs.has_events() {
            return;
        }
        let events = self.vfs.take_events();
        if self.inotify_watches.is_empty() {
            return;
        }
        let mut changed = Vec::new();
        for ev in events {
            // (watched inode, event mask, cookie, name for the watcher).
            let mut targets: Vec<(Ino, u32, u32, Vec<u8>)> = Vec::new();
            match ev {
                FsEvent::Child {
                    dir,
                    name,
                    ino,
                    mask,
                    cookie,
                } => {
                    targets.push((dir, mask, cookie, name));
                    // The child's own watchers see self-type events (modify, attrib, ...)
                    // without a name; directory-entry events (create, delete, moves) only
                    // concern the parent.
                    let own = mask & !(IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO);
                    if own & IN_ALL_EVENTS != 0 {
                        targets.push((ino, own, cookie, Vec::new()));
                    }
                }
                FsEvent::Inode {
                    ino,
                    mask,
                    isdir,
                    parent,
                } => {
                    let isdir = if isdir { IN_ISDIR } else { 0 };
                    let selfish = mask & (IN_DELETE_SELF | IN_MOVE_SELF) != 0;
                    // Linux tells the parent's watchers (by name) first, then the inode's own.
                    if let Some((dir, name)) = parent {
                        targets.push((dir, mask | isdir, 0, name));
                    }
                    targets.push((
                        ino,
                        if selfish { mask } else { mask | isdir },
                        0,
                        Vec::new(),
                    ));
                }
            }
            for (watched, mask, cookie, name) in targets {
                let Some(list) = self.inotify_watches.get(&watched).cloned() else {
                    continue;
                };
                for (weak, wd) in list {
                    let Some(inotify) = weak.upgrade() else {
                        continue;
                    };
                    let mut i = inotify.borrow_mut();
                    let Some(&(_, wmask)) = i.watches.get(&wd) else {
                        continue;
                    };
                    let bits = mask & IN_ALL_EVENTS;
                    let deleted = mask & IN_DELETE_SELF != 0;
                    if bits & wmask == 0 && !deleted {
                        continue;
                    }
                    if bits & wmask != 0 {
                        i.push(wd, mask & (wmask | IN_ISDIR), cookie, &name);
                    }
                    // A deleted inode, or a one-shot watch that fired, ends the watch.
                    if deleted || wmask & IN_ONESHOT != 0 {
                        i.watches.remove(&wd);
                        i.push(wd, IN_IGNORED, 0, b"");
                        drop(i);
                        self.forget_watch(watched, &inotify, wd);
                    } else {
                        drop(i);
                    }
                    changed.push(inotify.borrow().id);
                }
            }
        }
        changed.sort_unstable();
        changed.dedup();
        for id in changed {
            self.object_changed(id);
        }
    }

    /// The last description of an inotify instance closed: its watches go.
    pub(crate) fn inotify_closed(&mut self, inotify: &InotifyRef) {
        let watches: Vec<(i32, Ino)> = inotify
            .borrow()
            .watches
            .iter()
            .map(|(&wd, &(ino, _))| (wd, ino))
            .collect();
        for (wd, ino) in watches {
            self.forget_watch(ino, inotify, wd);
        }
        inotify.borrow_mut().watches.clear();
    }
}

/// Weak handles to the instances watching one inode.
pub(crate) type Watchers = Vec<(Weak<RefCell<Inotify>>, i32)>;
