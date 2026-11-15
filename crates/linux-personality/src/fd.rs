//! File descriptors and open file descriptions.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use wasmshim_linux_abi::Errno;

use crate::pipe::PipeRef;
use crate::vfs::{Device, Ino};

/// Highest number of descriptors per process (`RLIMIT_NOFILE` default soft limit).
pub const MAX_FDS: i32 = 1024;

/// Which host stream a stdio file talks to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stdio {
    In,
    Out,
    Err,
}

/// What an open file description refers to.
#[derive(Clone, Debug)]
pub enum FileKind {
    /// One of the host's standard streams (the initial fds 0–2).
    Stdio(Stdio),
    /// A VFS inode: regular file, directory, symlink (`O_PATH`) or device.
    Inode { ino: Ino, device: Option<Device> },
    /// One end of a pipe.
    Pipe { pipe: PipeRef, write: bool },
    /// A `/proc` file: contents generated at open.
    Proc { data: std::rc::Rc<Vec<u8>> },
    /// A `/proc` directory (by normalised path).
    ProcDir { path: Vec<u8> },
    /// An eventfd.
    EventFd(Rc<RefCell<crate::objects::eventfd::EventFd>>),
    /// A timerfd.
    TimerFd(Rc<RefCell<crate::objects::timerfd::TimerFd>>),
    /// An epoll instance.
    Epoll(Rc<RefCell<crate::objects::epoll::Epoll>>),
    /// An inotify instance.
    Inotify(crate::objects::inotify::InotifyRef),
    /// A signalfd.
    SignalFd(Rc<RefCell<crate::objects::signalfd::SignalFd>>),
    /// A POSIX message queue.
    MQueue(crate::objects::mqueue::MqRef),
    /// A pidfd: names process (or, with `PIDFD_THREAD`, thread) `pid`.
    PidFd { pid: crate::kernel::Pid },
    /// A local socket: AF_UNIX, or AF_INET/AF_INET6 on the in-kernel loopback (a listener or
    /// UDP socket may also have a network half attached).
    Unix(crate::objects::unix::UnixRef),
    /// An `AF_NETLINK` socket (`NETLINK_ROUTE`: the node's interfaces and addresses).
    Netlink(Rc<RefCell<crate::objects::netlink::Netlink>>),
    /// A TCP connection through the network provider (`crate::net`).
    Inet(crate::net::NetRef),
    /// `/dev/hidraw<n>`: an open description with its own report queue.
    HidRaw(crate::devices::HidRef),
    /// A terminal: the slave side (`/dev/pts/<n>`, `/dev/console`, `/dev/tty`) or a pty
    /// master (`/dev/ptmx`). `ino` is the inode it was opened through (`fstat`, `ttyname`).
    Tty {
        tty: crate::objects::tty::TtyRef,
        master: bool,
        ino: Ino,
    },
}

/// An open file description (shared by `dup`ed descriptors and across `fork`).
#[derive(Clone, Debug)]
pub struct OpenFile {
    pub kind: FileKind,
    /// `O_*` status flags (access mode, `O_APPEND`, `O_NONBLOCK`, ...).
    pub flags: u64,
    pub offset: u64,
    /// The path it was opened by (for `/proc/self/fd` and diagnostics).
    pub path: Vec<u8>,
    /// `F_SETOWN`: the process (or, negative, process group) that `O_ASYNC` would signal.
    pub owner: i32,
    /// A directory listing's position: the offset the last `getdents64` left and the name of
    /// the last entry it returned. While `offset` is still that value, the next call resumes
    /// after that name, so names removed or created meanwhile do not shift the listing.
    pub dir_cursor: Option<(u64, Vec<u8>)>,
}

/// A shared handle to an open file description.
pub type FileRef = Rc<RefCell<OpenFile>>;

/// One descriptor slot.
#[derive(Clone, Debug)]
pub struct Fd {
    pub file: FileRef,
    pub cloexec: bool,
}

/// A process's descriptor table.
#[derive(Clone, Debug, Default)]
pub struct FdTable {
    fds: BTreeMap<i32, Fd>,
}

impl FdTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The description behind `fd`.
    pub fn get(&self, fd: i32) -> Result<FileRef, Errno> {
        self.fds
            .get(&fd)
            .map(|f| f.file.clone())
            .ok_or(Errno::EBADF)
    }

    /// The slot for `fd`.
    pub fn slot_mut(&mut self, fd: i32) -> Result<&mut Fd, Errno> {
        self.fds.get_mut(&fd).ok_or(Errno::EBADF)
    }

    /// Installs `file` at the lowest free descriptor `>= min`.
    pub fn insert(&mut self, file: FileRef, cloexec: bool, min: i32) -> Result<i32, Errno> {
        let mut fd = min.max(0);
        for &used in self.fds.range(fd..).map(|(k, _)| k) {
            if used != fd {
                break;
            }
            fd += 1;
        }
        if fd >= MAX_FDS {
            return Err(Errno::EMFILE);
        }
        self.fds.insert(fd, Fd { file, cloexec });
        Ok(fd)
    }

    /// Installs `file` exactly at `fd`, returning what was there.
    pub fn insert_at(
        &mut self,
        fd: i32,
        file: FileRef,
        cloexec: bool,
    ) -> Result<Option<Fd>, Errno> {
        if !(0..MAX_FDS).contains(&fd) {
            return Err(Errno::EBADF);
        }
        Ok(self.fds.insert(fd, Fd { file, cloexec }))
    }

    /// Removes `fd`.
    pub fn remove(&mut self, fd: i32) -> Result<Fd, Errno> {
        self.fds.remove(&fd).ok_or(Errno::EBADF)
    }

    /// Removes every close-on-exec descriptor (for `execve`), returning them.
    pub fn take_cloexec(&mut self) -> Vec<Fd> {
        let doomed: Vec<i32> = self
            .fds
            .iter()
            .filter(|(_, f)| f.cloexec)
            .map(|(&k, _)| k)
            .collect();
        doomed
            .into_iter()
            .filter_map(|fd| self.fds.remove(&fd))
            .collect()
    }

    /// Every descriptor, in order.
    pub fn iter(&self) -> impl Iterator<Item = (i32, &Fd)> {
        self.fds.iter().map(|(&k, v)| (k, v))
    }
}

/// A new shared description.
pub fn open_file(kind: FileKind, flags: u64, path: &[u8]) -> FileRef {
    Rc::new(RefCell::new(OpenFile {
        kind,
        flags,
        offset: 0,
        path: path.to_vec(),
        owner: 0,
        dir_cursor: None,
    }))
}
