//! The personality's typed debug API: what a debugger or inspector reads and writes in a Linux
//! machine from outside, as plain Rust values (no paths, no wire format). An embedder's own
//! inspection protocol is built on top of it.
//!
//! **Embedder surface, unstable**: no debugger lives in this repository; the interface follows
//! what embedders need and may change between versions.
//!
//! - Processes ([`Kernel::process`]), their descriptors ([`Kernel::descriptor`]: what each one
//!   names, sockets with their peers) and mappings ([`Kernel::mappings`]).
//! - Threads are the substrate's tasks ([`Kernel::core`]); their registers through
//!   [`Kernel::thread_cpu_mut`].
//! - Memory ([`Kernel::with_memory`], [`Kernel::poke`]: a permission-ignoring store, like
//!   `ptrace`).
//! - Files: [`Kernel::vfs`] directly, `/proc` and `/sys` through [`Kernel::procfs`], and
//!   [`Kernel::replace_file`].
//! - Symbols and source lines ([`Kernel::where_is`], [`Kernel::symbol`],
//!   [`Kernel::line_addresses`], [`Kernel::resolve_addresses`], [`Kernel::stack`]).
//! - The clocks ([`Kernel::retired`], [`Kernel::syscalls`]), syscall names and numbers, and the
//!   profile counters ([`Kernel::engine_counters`], [`Kernel::syscall_stats`],
//!   [`Kernel::thread_stats`]).
//!
//! Reading changes nothing the guest can see: no task runs and no guest state moves (reading an
//! image file may fill the host-side page cache, whose contents are what the guest would read).

use std::net::SocketAddr;

use wasmshim_cpu::{GuestCpu, GuestMemory, MemFault, PAGE_SIZE, Perms};
use wasmshim_linux_abi::Errno;

use crate::fd::FileKind;
use crate::kernel::{Kernel, Pid, Tid};
use crate::mm::VmaKind;
use crate::net::{NetState, Proto};
use crate::procfs::ProcNode;

pub use crate::symbols::{Frame, SymbolInfo, Where};

/// A process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessInfo {
    pub pid: Pid,
    pub ppid: Pid,
    pub comm: Vec<u8>,
    pub exe: Vec<u8>,
    /// `argv` joined with NULs (`/proc/<pid>/cmdline`).
    pub cmdline: Vec<u8>,
    /// Its threads, by tid.
    pub threads: Vec<Tid>,
    /// Its open descriptor numbers, in order.
    pub fds: Vec<i32>,
}

/// One open descriptor of a process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Descriptor {
    pub fd: i32,
    /// What `/proc/<pid>/fd/<n>` links to (the path it was opened by, when procfs has none).
    pub path: Vec<u8>,
    /// `O_*` status flags.
    pub flags: u64,
    pub offset: u64,
    pub kind: DescriptorKind,
}

/// What a descriptor names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DescriptorKind {
    Stdio,
    /// A VFS inode.
    File,
    /// One end of a pipe (`write`: the write end).
    Pipe {
        write: bool,
    },
    /// A `/proc` file or directory.
    Procfs,
    EventFd,
    TimerFd,
    Epoll,
    Inotify,
    SignalFd,
    MQueue,
    PidFd,
    Netlink,
    /// A local socket (AF_UNIX, or AF_INET/AF_INET6 on the in-kernel loopback).
    Unix(UnixFacts),
    /// A socket through the network provider.
    Inet(InetFacts),
    /// Anything else (terminals, HID devices).
    Other,
}

/// A local socket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnixFacts {
    pub state: UnixSocketState,
    /// The address family (`AF_UNIX`, `AF_INET`, `AF_INET6`).
    pub family: u16,
    pub local: Option<SocketName>,
    pub remote: Option<SocketName>,
    /// Where the other end is open in this machine: `(pid, fd)`.
    pub peer: Option<(Pid, i32)>,
    /// Bytes queued for reading.
    pub rx_queued: usize,
}

/// A local socket's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnixSocketState {
    Unconnected,
    /// Listening, with this many connections waiting to be accepted.
    Listening {
        accept_queued: usize,
    },
    Connected,
}

/// A socket name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SocketName {
    /// `sun_path` bytes (abstract names start with NUL).
    Unix(Vec<u8>),
    Inet(SocketAddr),
}

/// A socket through the network provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InetFacts {
    pub proto: Proto,
    pub state: NetState,
    pub local: Option<SocketAddr>,
    pub remote: Option<SocketAddr>,
    pub rx_queued: usize,
    pub tx_queued: usize,
}

/// One mapping of an address space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mapping {
    pub start: u64,
    pub end: u64,
    pub perms: Perms,
    pub kind: MappingKind,
}

/// What a mapping is for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MappingKind {
    Anon,
    Heap,
    Stack,
    Vdso,
    /// A private file mapping from byte `offset` of `path`.
    File {
        path: Vec<u8>,
        offset: u64,
    },
    /// A `MAP_SHARED` mapping: `name` is what `/proc/<pid>/maps` shows.
    Shared {
        name: Vec<u8>,
        offset: u64,
    },
}

/// A `/proc` or `/sys` node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProcEntry {
    /// A directory and its entries (name, `d_type`).
    Dir(Vec<(Vec<u8>, u8)>),
    /// A file and its contents.
    File(Vec<u8>),
    /// A symlink and its target.
    Link(Vec<u8>),
}

/// One syscall number's profile counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyscallStat {
    pub number: u64,
    /// Its name (`None`: not a syscall of this architecture).
    pub name: Option<&'static str>,
    pub entries: u64,
    pub blocks: u64,
    pub blocked_ns: u64,
}

/// One live thread's profile counters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadStat {
    pub tid: Tid,
    /// Its process and the process's `comm` (lossy UTF-8).
    pub process: Option<(Pid, String)>,
    pub retired: u64,
    pub syscalls: u64,
    /// Virtual nanoseconds blocked (completed waits).
    pub blocked_ns: u64,
}

/// `len` bytes at `addr`, page by page; `None` at the first unreadable byte.
pub(crate) fn read_bytes(mem: &dyn GuestMemory, addr: u64, len: u64) -> Option<Vec<u8>> {
    let mut out = vec![0u8; usize::try_from(len).ok()?];
    let mut done = 0u64;
    while done < len {
        let at = addr.checked_add(done)?;
        let chunk = (PAGE_SIZE - at % PAGE_SIZE).min(len - done);
        mem.read(at, &mut out[done as usize..(done + chunk) as usize])
            .ok()?;
        done += chunk;
    }
    Some(out)
}

impl Kernel {
    /// The substrate: its tasks (this machine's threads), clocks and profile counters.
    pub fn core(&self) -> &wasmshim_kernel::Core<impl wasmshim_kernel::Types> {
        &self.core
    }

    /// The observer (probes, steps, the `sys` clock), to arm and disarm probes.
    pub fn observer_mut(&mut self) -> &mut wasmshim_kernel::observe::Observer {
        &mut self.observer
    }

    /// Syscall entries so far, node-wide (the `sys` clock).
    pub fn syscalls(&self) -> u64 {
        self.observer.syscalls()
    }

    /// The number of the syscall called `name` on this architecture.
    pub fn syscall_number(&self, name: &str) -> Option<u64> {
        (0..1024u64).find(|&n| self.arch.sysno(n).map(|s| s.name()) == Some(name))
    }

    /// The name of syscall `number` on this architecture.
    pub fn syscall_name(&self, number: u64) -> Option<&'static str> {
        self.arch.sysno(number).map(|s| s.name())
    }

    /// Every live process's pid, in order.
    pub fn pids(&self) -> Vec<Pid> {
        self.procs.keys().copied().collect()
    }

    /// Process `pid`.
    pub fn process(&self, pid: Pid) -> Option<ProcessInfo> {
        let p = self.procs.get(&pid)?;
        Some(ProcessInfo {
            pid,
            ppid: p.ppid,
            comm: p.comm.clone(),
            exe: p.exe.clone(),
            cmdline: p.cmdline.clone(),
            threads: p.threads.iter().copied().collect(),
            fds: p.fds.borrow().iter().map(|(fd, _)| fd).collect(),
        })
    }

    /// Whether `tid` is a live thread of process `pid`.
    pub fn has_thread(&self, pid: Pid, tid: Tid) -> bool {
        self.procs
            .get(&pid)
            .is_some_and(|p| p.threads.contains(&tid))
            && self.core.tasks.contains_key(&tid)
    }

    /// The process of live thread `tid`.
    pub fn process_of_thread(&self, tid: Tid) -> Option<Pid> {
        self.core.tasks.get(&tid).map(|t| t.pid)
    }

    /// Thread `tid`'s CPU, for register writes (`context`, `set_context`).
    pub fn thread_cpu_mut(&mut self, tid: Tid) -> Option<&mut (dyn GuestCpu + 'static)> {
        self.core.tasks.get_mut(&tid).map(|t| t.cpu.as_mut())
    }

    /// Descriptor `fd` of process `pid`.
    pub fn descriptor(&self, pid: Pid, fd: i32) -> Option<Descriptor> {
        let file = self.procs.get(&pid)?.fds.borrow().get(fd).ok()?;
        let f = file.borrow();
        let path = match self.proc_pid(pid, None, &["fd", &fd.to_string()], self.core.slice_now) {
            Ok(ProcNode::Link(l)) => l,
            _ => f.path.clone(),
        };
        let kind = match &f.kind {
            FileKind::Stdio(_) => DescriptorKind::Stdio,
            FileKind::Inode { .. } => DescriptorKind::File,
            FileKind::Pipe { write, .. } => DescriptorKind::Pipe { write: *write },
            FileKind::Proc { .. } | FileKind::ProcDir { .. } => DescriptorKind::Procfs,
            FileKind::EventFd(_) => DescriptorKind::EventFd,
            FileKind::TimerFd(_) => DescriptorKind::TimerFd,
            FileKind::Epoll(_) => DescriptorKind::Epoll,
            FileKind::Inotify(_) => DescriptorKind::Inotify,
            FileKind::SignalFd(_) => DescriptorKind::SignalFd,
            FileKind::MQueue(_) => DescriptorKind::MQueue,
            FileKind::PidFd { .. } => DescriptorKind::PidFd,
            FileKind::Netlink(_) => DescriptorKind::Netlink,
            FileKind::Unix(u) => {
                use crate::objects::unix::{SockName, UnixState};
                let s = u.borrow();
                let name = |n: &SockName| match n {
                    SockName::Inet(a) => SocketName::Inet(*a),
                    SockName::Unix(p) => SocketName::Unix(p.clone()),
                };
                DescriptorKind::Unix(UnixFacts {
                    state: match &s.state {
                        UnixState::Unconnected => UnixSocketState::Unconnected,
                        UnixState::Listening { queue, .. } => UnixSocketState::Listening {
                            accept_queued: queue.len(),
                        },
                        UnixState::Connected => UnixSocketState::Connected,
                    },
                    family: s.family,
                    local: s.name.as_ref().map(name),
                    remote: s.peer_name.as_ref().map(name),
                    peer: s.peer.upgrade().and_then(|peer| self.unix_peer(&peer)),
                    rx_queued: s.rx.len(),
                })
            }
            FileKind::Inet(n) => {
                let s = n.borrow();
                DescriptorKind::Inet(InetFacts {
                    proto: s.proto,
                    state: s.state,
                    local: s.local,
                    remote: s.remote,
                    rx_queued: s.rx.len(),
                    tx_queued: s.tx.len(),
                })
            }
            _ => DescriptorKind::Other,
        };
        Some(Descriptor {
            fd,
            path,
            flags: f.flags,
            offset: f.offset,
            kind,
        })
    }

    /// Where the other end of a local socket is open in this machine.
    fn unix_peer(
        &self,
        peer: &std::rc::Rc<std::cell::RefCell<crate::objects::unix::UnixSocket>>,
    ) -> Option<(Pid, i32)> {
        for (pid, p) in &self.procs {
            for (fd, slot) in p.fds.borrow().iter() {
                if let FileKind::Unix(u) = &slot.file.borrow().kind
                    && std::rc::Rc::ptr_eq(u, peer)
                {
                    return Some((*pid, fd));
                }
            }
        }
        None
    }

    /// The mappings of process `pid`, in address order (empty without the process).
    pub fn mappings(&self, pid: Pid) -> Vec<Mapping> {
        let Some(p) = self.procs.get(&pid) else {
            return Vec::new();
        };
        p.mm.borrow()
            .vmas()
            .map(|v| Mapping {
                start: v.start,
                end: v.end,
                perms: v.perms,
                kind: match &v.kind {
                    VmaKind::Anon => MappingKind::Anon,
                    VmaKind::Heap => MappingKind::Heap,
                    VmaKind::Stack => MappingKind::Stack,
                    VmaKind::Vdso => MappingKind::Vdso,
                    VmaKind::File { path, offset } => MappingKind::File {
                        path: path.clone(),
                        offset: *offset,
                    },
                    VmaKind::Shared { name, offset, .. } => MappingKind::Shared {
                        name: name.clone(),
                        offset: *offset,
                    },
                },
            })
            .collect()
    }

    /// Runs `f` on process `pid`'s memory (reads with [`GuestMemory::read`] have no
    /// guest-visible effect).
    pub fn with_memory<R>(&self, pid: Pid, f: impl FnOnce(&dyn GuestMemory) -> R) -> Option<R> {
        let mm = self.procs.get(&pid)?.mm.borrow();
        Some(f(&*mm.mem))
    }

    /// Stores `bytes` at `addr` in process `pid`'s memory, ignoring page permissions (like
    /// `ptrace`; a store to code moves the page's code version, so no engine runs stale code).
    /// `None` without the process.
    pub fn poke(&mut self, pid: Pid, addr: u64, bytes: &[u8]) -> Option<Result<(), MemFault>> {
        let mm = self.procs.get(&pid)?.mm.clone();
        Some(mm.borrow_mut().mem.poke(addr, bytes))
    }

    /// A `/proc` or `/sys` node by absolute path, as the guest would see it now.
    pub fn procfs(&self, path: &[u8]) -> Result<ProcEntry, Errno> {
        let now = self.core.slice_now;
        Ok(match self.proc_lookup(path, now, now)? {
            ProcNode::Dir(e) => ProcEntry::Dir(e),
            ProcNode::File(b) => ProcEntry::File(b),
            ProcNode::Link(l) => ProcEntry::Link(l),
        })
    }

    /// Replaces the contents of the file at absolute `path` (links followed) with `bytes`.
    pub fn replace_file(&mut self, path: &[u8], bytes: &[u8]) -> Result<(), Errno> {
        let now = wasmshim_linux_abi::structs::Timespec::from_nanos(self.core.slice_now);
        let ino = self.vfs.resolve(0, path, true)?;
        self.vfs
            .truncate(ino, 0, now)
            .and_then(|()| self.vfs.write(ino, 0, bytes, now).map(|_| ()))
    }

    /// The CPU engine's counters, by name (empty for the interpreter-only frontend).
    pub fn engine_counters(&self) -> Vec<(&'static str, u64)> {
        self.frontend.counters()
    }

    /// Per syscall number: entries, blocks and time blocked, in number order.
    pub fn syscall_stats(&self) -> Vec<SyscallStat> {
        self.core
            .stats
            .sys
            .iter()
            .map(|(&n, s)| SyscallStat {
                number: n,
                name: self.syscall_name(n),
                entries: s.entries,
                blocks: s.blocks,
                blocked_ns: s.blocked_ns,
            })
            .collect()
    }

    /// Per live thread, in tid order: its process, retired instructions, syscalls and time
    /// blocked.
    pub fn thread_stats(&self) -> Vec<ThreadStat> {
        let stats = &self.core.stats;
        self.core
            .tasks
            .iter()
            .map(|(&tid, t)| {
                let st = stats.threads.get(&tid).copied().unwrap_or_default();
                ThreadStat {
                    tid,
                    process: self
                        .procs
                        .iter()
                        .find(|(_, p)| p.threads.contains(&tid))
                        .map(|(&pid, p)| (pid, String::from_utf8_lossy(&p.comm).into_owned())),
                    retired: t.cpu.retired(),
                    syscalls: st.syscalls,
                    blocked_ns: st.blocked_ns,
                }
            })
            .collect()
    }
}
