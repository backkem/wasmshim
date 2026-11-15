//! Kernel objects behind file descriptors that are neither VFS files nor pipes: eventfd,
//! timerfd, epoll and sockets (in-kernel and through the network provider).
//!
//! Every object has an id (its [`WaitKey::Object`]) and a change *generation*: readiness is
//! computed from object state ([`Kernel::poll_events`]), the generation lets edge-triggered
//! epoll report "something new happened", and a change wakes the object's waiters plus every
//! `poll`/`select`/`epoll_wait` sleeper ([`WaitKey::AnyIo`]).

pub(crate) mod epoll;
pub(crate) mod eventfd;
pub(crate) mod inet;
pub(crate) mod inotify;
pub(crate) mod mqueue;
pub mod netlink;
pub(crate) mod pidfd;
pub(crate) mod signalfd;
pub(crate) mod timerfd;
pub(crate) mod tty;
pub(crate) mod unix;

use wasmshim_linux_abi::Errno;

use crate::fd::{FileKind, FileRef};
use crate::kernel::{Kernel, WaitKey};
use crate::pipe::PIPE_CAPACITY;
use crate::syscall::BLOCK;
use wasmshim_kernel::ObjectDeadlines;

pub(crate) const EPOLLIN: u32 = 0x001;
pub(crate) const EPOLLPRI: u32 = 0x002;
pub(crate) const EPOLLOUT: u32 = 0x004;
pub(crate) const EPOLLERR: u32 = 0x008;
pub(crate) const EPOLLHUP: u32 = 0x010;
pub(crate) const EPOLLRDNORM: u32 = 0x040;
pub(crate) const EPOLLWRNORM: u32 = 0x100;
pub(crate) const EPOLLRDHUP: u32 = 0x2000;
pub(crate) const EPOLLEXCLUSIVE: u32 = 1 << 28;
pub(crate) const EPOLLWAKEUP: u32 = 1 << 29;
pub(crate) const EPOLLONESHOT: u32 = 1 << 30;
pub(crate) const EPOLLET: u32 = 1 << 31;
/// Linux's `EP_MAX_NESTS`: how deep epoll instances may nest (fs/eventpoll.c).
pub(crate) const EP_MAX_NESTS: u32 = 4;

impl Kernel {
    /// Allocates an object id.
    pub(crate) fn new_object_id(&mut self) -> u64 {
        self.core.new_object_id()
    }

    /// Wakes the waiters of object `id` (and every poller).
    pub(crate) fn object_changed(&mut self, id: u64) {
        self.wake(WaitKey::Object(id), usize::MAX, u32::MAX);
    }

    /// Readiness of a description as `EPOLL*` bits, and its change generation.
    pub(crate) fn poll_events(&self, file: &FileRef, now: u64) -> (u32, u64) {
        self.poll_events_nested(file, now, EP_MAX_NESTS + 1)
    }

    /// [`Kernel::poll_events`], looking into at most `depth` nested epoll instances (an
    /// `epoll_ctl` keeps every nesting within that, so this only guards the recursion).
    pub(crate) fn poll_events_nested(&self, file: &FileRef, now: u64, depth: u32) -> (u32, u64) {
        let f = file.borrow();
        match &f.kind {
            FileKind::Pipe { pipe, write: false } => {
                let p = pipe.borrow();
                let mut ev = 0;
                if !p.buf.is_empty() {
                    ev |= EPOLLIN | EPOLLRDNORM;
                }
                if p.writers == 0 {
                    ev |= EPOLLHUP;
                }
                (ev, p.generation)
            }
            FileKind::Pipe { pipe, write: true } => {
                let p = pipe.borrow();
                let mut ev = 0;
                if p.buf.len() < PIPE_CAPACITY {
                    ev |= EPOLLOUT | EPOLLWRNORM;
                }
                if p.readers == 0 {
                    ev |= EPOLLERR;
                }
                (ev, p.generation)
            }
            FileKind::EventFd(e) => e.borrow().events(),
            FileKind::Inotify(i) => i.borrow().events(),
            FileKind::TimerFd(t) => t.borrow().events(now),
            FileKind::SignalFd(s) => self.signalfd_events(&s.borrow()),
            FileKind::PidFd { pid } => self.pidfd_events(*pid),
            FileKind::MQueue(q) => q.borrow().events(),
            FileKind::Unix(s) => s.borrow().events(),
            FileKind::Inet(s) => inet::events(&s.borrow()),
            FileKind::Netlink(n) => n.borrow().events(),
            FileKind::HidRaw(h) => h.borrow().events(),
            FileKind::Tty { tty, master, .. } => Kernel::tty_events(&tty.borrow(), *master),
            FileKind::Epoll(e) => (
                if depth > 0
                    && e.try_borrow()
                        .is_ok_and(|e| self.epoll_has_ready(&e, now, depth - 1))
                {
                    EPOLLIN
                } else {
                    0
                },
                0,
            ),
            // Regular files, devices, /proc and the host's streams never block.
            _ => (EPOLLIN | EPOLLOUT | EPOLLRDNORM | EPOLLWRNORM, 0),
        }
    }

    /// The earliest time this description could become ready by itself, with no writer: a
    /// timerfd's next expiry, or the earliest such deadline in an epoll set (nested up to a
    /// few levels, as Linux limits nesting).
    pub(crate) fn object_deadline(&self, file: &FileRef) -> Option<u64> {
        self.object_deadline_nested(file, 4)
    }

    pub(crate) fn object_deadline_nested(&self, file: &FileRef, depth: u32) -> Option<u64> {
        match &file.borrow().kind {
            FileKind::TimerFd(t) => t.borrow().next_expiry(),
            FileKind::Epoll(e) if depth > 0 => self.epoll_deadline(&e.borrow(), depth - 1),
            _ => None,
        }
    }

    /// Blocks the current readiness-polling call (`ppoll`, `pselect6`, `epoll_wait`) that found
    /// nothing ready: until any object changes, the earliest deadline of the watched objects,
    /// or the caller's `deadline` (`None`: forever), whichever comes first. At an object's
    /// deadline the call runs again (and finds it ready); at the caller's it returns 0. This is
    /// the one wait every such call goes through (the substrate's
    /// [`wasmshim_kernel::Core::wait_objects`]), so a timerfd wakes `ppoll` exactly as it
    /// wakes `epoll_wait`. Linux adds only the console: input is requested for the readers.
    pub(crate) fn wait_objects(&mut self, w: ObjectWait, deadline: Option<u64>) -> Errno {
        for f in &w.readers {
            self.note_input_wait(f);
        }
        self.core.wait_objects(w.deadlines, deadline);
        BLOCK
    }
}

/// What a readiness-polling call watches while it scans its descriptions: the ones it waits
/// to read from (console input is requested for them) and the objects' own deadlines (the
/// substrate's [`ObjectDeadlines`]). Handed to [`Kernel::wait_objects`].
#[derive(Default)]
pub(crate) struct ObjectWait {
    readers: Vec<FileRef>,
    deadlines: ObjectDeadlines,
}

impl ObjectWait {
    /// Adds `file`, scanned at `now`; `waits_to_read`: the caller asked to read it and it is
    /// not readable.
    pub(crate) fn watch(&mut self, k: &Kernel, file: &FileRef, now: u64, waits_to_read: bool) {
        if waits_to_read {
            self.readers.push(file.clone());
        }
        self.deadlines.watch(k.object_deadline(file), now);
    }
}

/// Bits that are always reported whether or not they were asked for.
pub(crate) const ALWAYS: u32 = EPOLLERR | EPOLLHUP;

#[allow(dead_code)]
const _UNUSED: u32 = EPOLLPRI | EPOLLEXCLUSIVE | EPOLLWAKEUP;
