//! `epoll`: interest lists over any file description, level- or edge-triggered.
//!
//! An interest holds a *weak* reference to its description, so closing the last descriptor
//! of a file removes it from every epoll set, as on Linux. Edge-triggered interests report a
//! ready description only when its change generation moved since the last report.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::{Rc, Weak};

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::open;

use super::{ALWAYS, EP_MAX_NESTS, EPOLLET, EPOLLIN, EPOLLONESHOT, ObjectWait};
use crate::fd::{FileKind, OpenFile, open_file};
use crate::host::Host;
use crate::kernel::{Kernel, Tid};
use crate::syscall::SysResult;
use crate::uaccess::{read_array, write_bytes};

const EPOLL_CTL_ADD: u64 = 1;
const EPOLL_CTL_DEL: u64 = 2;
const EPOLL_CTL_MOD: u64 = 3;
/// `struct epoll_event` on AArch64: `u32 events`, padding, `u64 data` (not packed).
const EVENT_SIZE: u64 = 16;

/// One registered interest.
#[derive(Debug)]
pub(crate) struct Interest {
    pub(crate) file: Weak<RefCell<OpenFile>>,
    pub(crate) events: u32,
    pub(crate) data: u64,
    /// Generation last reported (edge-triggered).
    pub(crate) last_gen: Option<u64>,
    /// A one-shot interest that already fired (re-armed by `EPOLL_CTL_MOD`).
    pub(crate) disabled: bool,
}

/// One epoll instance.
#[derive(Debug)]
pub(crate) struct Epoll {
    pub id: u64,
    pub interests: BTreeMap<i32, Interest>,
}

impl Kernel {
    /// `epoll_create1(flags)`.
    pub(crate) fn sys_epoll_create1(&mut self, tid: Tid, flags: u64) -> SysResult {
        if flags & !open::O_CLOEXEC != 0 {
            return Err(Errno::EINVAL);
        }
        let id = self.new_object_id();
        let e = Rc::new(RefCell::new(Epoll {
            id,
            interests: BTreeMap::new(),
        }));
        let file = open_file(FileKind::Epoll(e), open::O_RDWR, b"anon_inode:[eventpoll]");
        Ok(self
            .fds(tid)
            .borrow_mut()
            .insert(file, flags & open::O_CLOEXEC != 0, 0)? as u64)
    }

    fn epoll_of(&self, tid: Tid, epfd: u64) -> Result<Rc<RefCell<Epoll>>, Errno> {
        match &self.file(tid, epfd)?.borrow().kind {
            FileKind::Epoll(e) => Ok(e.clone()),
            _ => Err(Errno::EINVAL),
        }
    }

    /// `epoll_ctl(epfd, op, fd, event)`.
    pub(crate) fn sys_epoll_ctl(
        &mut self,
        tid: Tid,
        epfd: u64,
        op: u64,
        fd: u64,
        event: u64,
    ) -> SysResult {
        let ep = self.epoll_of(tid, epfd)?;
        let file = self.file(tid, fd)?;
        if Rc::ptr_eq(&file, &self.file(tid, epfd)?) {
            return Err(Errno::EINVAL);
        }
        if matches!(
            file.borrow().kind,
            FileKind::Inode { .. } | FileKind::Proc { .. } | FileKind::ProcDir { .. }
        ) {
            // Regular files and directories cannot be polled by epoll.
            return Err(Errno::EPERM);
        }
        // epoll_ctl(2), ELOOP: adding an epoll instance must not close a loop of instances
        // watching one another, nor nest them more than `EP_MAX_NESTS` deep
        // (fs/eventpoll.c, `ep_loop_check`).
        if op == EPOLL_CTL_ADD
            && let FileKind::Epoll(to) = &file.borrow().kind
        {
            let id = ep.borrow().id;
            let below = to
                .try_borrow()
                .ok()
                .and_then(|to| self.epoll_nesting(&to, id, 0));
            match below {
                Some(depth) if depth + 1 + self.epoll_nesting_above(id) <= EP_MAX_NESTS => {}
                _ => return Err(Errno::ELOOP),
            }
        }
        let fd = fd as i32;
        let (events, data) = if op == EPOLL_CTL_DEL {
            (0, 0)
        } else {
            let raw: [u8; 16] = read_array(self.mm(tid).borrow_mut().mem.as_mut(), event)?;
            (
                u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                u64::from_le_bytes(raw[8..16].try_into().unwrap_or_default()),
            )
        };
        let mut e = ep.borrow_mut();
        let live = e
            .interests
            .get(&fd)
            .is_some_and(|i| i.file.upgrade().is_some_and(|f| Rc::ptr_eq(&f, &file)));
        match op {
            EPOLL_CTL_ADD => {
                if live {
                    return Err(Errno::EEXIST);
                }
                e.interests.insert(
                    fd,
                    Interest {
                        file: Rc::downgrade(&file),
                        events,
                        data,
                        last_gen: None,
                        disabled: false,
                    },
                );
            }
            EPOLL_CTL_MOD => {
                let Some(i) = e.interests.get_mut(&fd).filter(|_| live) else {
                    return Err(Errno::ENOENT);
                };
                i.events = events;
                i.data = data;
                i.last_gen = None;
                i.disabled = false;
            }
            EPOLL_CTL_DEL => {
                if !live {
                    return Err(Errno::ENOENT);
                }
                e.interests.remove(&fd);
            }
            _ => return Err(Errno::EINVAL),
        }
        let id = e.id;
        drop(e);
        self.object_changed(id);
        Ok(0)
    }

    /// How deep epoll instances nest below `e` (0: it watches none), counted from `level`;
    /// `None` if instance `target` is among them (adding it would close a loop). Stops
    /// looking once deeper than [`EP_MAX_NESTS`], which is refused anyway.
    fn epoll_nesting(&self, e: &Epoll, target: u64, level: u32) -> Option<u32> {
        if e.id == target {
            return None;
        }
        if level > EP_MAX_NESTS {
            return Some(level);
        }
        let mut deepest = level;
        for i in e.interests.values() {
            let Some(f) = i.file.upgrade() else {
                continue;
            };
            let f = f.try_borrow().ok()?;
            if let FileKind::Epoll(child) = &f.kind {
                let child = child.try_borrow().ok()?;
                deepest = deepest.max(self.epoll_nesting(&child, target, level + 1)?);
            }
        }
        Some(deepest)
    }

    /// How deep epoll instance `id` sits below the instances that watch it (0: none does), as
    /// far as descriptor tables show; up to [`EP_MAX_NESTS`] + 1.
    fn epoll_nesting_above(&self, id: u64) -> u32 {
        // Every epoll instance a process holds, and the instances each one watches.
        let mut parents: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();
        for p in self.procs.values() {
            for (_, slot) in p.fds.borrow().iter() {
                let f = slot.file.borrow();
                let FileKind::Epoll(e) = &f.kind else {
                    continue;
                };
                let Ok(e) = e.try_borrow() else {
                    continue;
                };
                if !seen.insert(e.id) {
                    continue;
                }
                for i in e.interests.values() {
                    let Some(c) = i.file.upgrade() else {
                        continue;
                    };
                    if let Ok(c) = c.try_borrow()
                        && let FileKind::Epoll(c) = &c.kind
                        && let Ok(c) = c.try_borrow()
                    {
                        parents.entry(c.id).or_default().push(e.id);
                    }
                }
            }
        }
        let mut level = 0;
        let mut frontier = vec![id];
        while level <= EP_MAX_NESTS {
            let up: Vec<u64> = frontier
                .iter()
                .filter_map(|c| parents.get(c))
                .flatten()
                .copied()
                .collect();
            if up.is_empty() {
                break;
            }
            level += 1;
            frontier = up;
            frontier.sort_unstable();
            frontier.dedup();
        }
        level
    }

    /// The earliest deadline among the enabled interests (see [`Kernel::object_deadline`]).
    pub(crate) fn epoll_deadline(&self, e: &Epoll, depth: u32) -> Option<u64> {
        e.interests
            .values()
            .filter(|i| !i.disabled)
            .filter_map(|i| i.file.upgrade())
            .filter_map(|f| self.object_deadline_nested(&f, depth))
            .min()
    }

    /// Level-triggered "is anything ready" (for nesting an epoll in poll/epoll); `depth`
    /// bounds how many more nested instances are looked into.
    pub(crate) fn epoll_has_ready(&self, e: &Epoll, now: u64, depth: u32) -> bool {
        e.interests.values().any(|i| {
            !i.disabled
                && i.file.upgrade().is_some_and(|f| {
                    self.poll_events_nested(&f, now, depth).0 & (i.events | ALWAYS) != 0
                })
        })
    }

    /// `epoll_pwait(epfd, events, maxevents, timeout_ms, sigmask)`; `timeout` is a deadline
    /// already computed by the caller (`None` = forever, `Some(now)` = don't block).
    pub(crate) fn epoll_wait(
        &mut self,
        tid: Tid,
        epfd: u64,
        events: u64,
        max: u64,
        deadline: Option<u64>,
        host: &mut dyn Host,
    ) -> SysResult {
        if max == 0 || max > 1 << 20 {
            return Err(Errno::EINVAL);
        }
        let ep = self.epoll_of(tid, epfd)?;
        let now = host.monotonic_ns();
        let mut out = Vec::new();
        let mut wait = ObjectWait::default();
        {
            let mut e = ep.borrow_mut();
            e.interests.retain(|_, i| i.file.strong_count() > 0);
            let fds: Vec<i32> = e.interests.keys().copied().collect();
            for fd in fds {
                if out.len() as u64 >= max {
                    break;
                }
                let Some(i) = e.interests.get_mut(&fd) else {
                    continue;
                };
                let Some(file) = i.file.upgrade() else {
                    continue;
                };
                if i.disabled {
                    continue;
                }
                let (ready, generation) = self.poll_events(&file, now);
                let mask = ready & (i.events | ALWAYS);
                wait.watch(
                    self,
                    &file,
                    now,
                    ready & EPOLLIN == 0 && i.events & EPOLLIN != 0,
                );
                if mask == 0 {
                    continue;
                }
                if i.events & EPOLLET != 0 {
                    if i.last_gen == Some(generation) {
                        continue;
                    }
                    i.last_gen = Some(generation);
                }
                if i.events & EPOLLONESHOT != 0 {
                    i.disabled = true;
                }
                out.push((mask, i.data));
            }
        }
        if out.is_empty() {
            if deadline.is_some_and(|d| d <= now) {
                return Ok(0);
            }
            return Err(self.wait_objects(wait, deadline));
        }
        let mut buf = vec![0u8; out.len() * EVENT_SIZE as usize];
        for (k, (mask, data)) in out.iter().enumerate() {
            let at = k * EVENT_SIZE as usize;
            buf[at..at + 4].copy_from_slice(&mask.to_le_bytes());
            buf[at + 8..at + 16].copy_from_slice(&data.to_le_bytes());
        }
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), events, &buf)?;
        Ok(out.len() as u64)
    }
}
