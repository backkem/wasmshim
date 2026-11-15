//! The kernel object: processes and their threads on the syscall substrate.
//!
//! The substrate (`wasmshim-kernel`) owns the threads as tasks, the deterministic
//! scheduler and the one wait primitive; Linux plugs in through [`wasmshim_kernel::Personality`]
//! (signal delivery before a thread runs, its timers, syscall dispatch, the trace's names).
//!
//! * Threads are green threads in one instance. The scheduler runs the first runnable thread
//!   of a FIFO run queue for up to one quantum of retired instructions, then rotates.
//! * A blocking syscall suspends only its own thread with a [`Blocker`]. Blockers that wait
//!   on an object name a [`WaitKey`]; a state change on that object wakes exactly the threads
//!   waiting on its key (per-object wait queues), in a deterministic order.
//! * Most blockers *restart*: the thread's PC is rewound to its `SVC`, so after a wake-up the
//!   syscall simply runs again against the new state. Futex waits, sleeps and `vfork` complete
//!   with a value instead.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use wasmshim_cpu::{Fault, SyscallFrame};
use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::structs::KernelSigaction;

use crate::arch::Arch;
use crate::fd::{FdTable, FileKind, Stdio, open_file};
use crate::host::{Frontend, Host};
use crate::mm::AddressSpace;
use crate::syscall::Flow;
use crate::vfs::{Ino, ROOT, Vfs};

/// A process id.
pub type Pid = u32;
/// A thread id (same number space as pids).
pub type Tid = u32;

/// How a process ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitStatus {
    /// `exit(code)`; only the low 8 bits are kept, like Linux.
    Code(u8),
    /// Killed by a signal.
    Signal(u32),
}

impl ExitStatus {
    /// The conventional shell exit code (`128 + signal` for signals).
    pub fn shell_code(self) -> u8 {
        match self {
            Self::Code(c) => c,
            Self::Signal(s) => 128u32.saturating_add(s).min(255) as u8,
        }
    }

    /// `si_code` and `si_status` of a `SIGCHLD` or `waitid` report (core dumps are never
    /// written, so a signal death is `CLD_KILLED`).
    pub fn child_code(self) -> (i32, i32) {
        match self {
            Self::Code(c) => (wasmshim_linux_abi::consts::signal::CLD_EXITED, i32::from(c)),
            Self::Signal(s) => (wasmshim_linux_abi::consts::signal::CLD_KILLED, s as i32),
        }
    }

    /// The `wait4` status word.
    pub fn wait_status(self) -> u32 {
        match self {
            Self::Code(c) => u32::from(c) << 8,
            Self::Signal(s) => s & 0x7f,
        }
    }
}

/// What [`Kernel::run`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunStatus {
    /// The initial process ended.
    Exited(ExitStatus),
    /// Let virtual time pass until `until` (monotonic ns), then call `run` again. `None` means
    /// every thread is blocked with no deadline: nothing can ever happen (a guest deadlock).
    Wait { until: Option<u64> },
}

/// Tunables. All of them affect guest-visible behaviour, so they are part of a run's identity.
#[derive(Clone, Debug)]
pub struct Config {
    /// `uname -n`.
    pub hostname: String,
    /// Instructions a thread may run before the scheduler switches threads (and the kernel
    /// yields to the host). At least 1: a machine made with 0 runs with 1 (a quantum of nothing
    /// would never run an instruction). It sets scheduling granularity only, so 1 is a valid,
    /// if slow, choice; `Kernel::new` cannot fail, so 0 is not an error.
    pub quantum: u64,
    /// Virtual nanoseconds charged per retired instruction (`0` disables): guest compute
    /// advances the virtual clock deterministically, like QEMU's `icount`.
    pub ns_per_insn: u64,
    /// Emit a syscall trace through [`Host::diag`].
    pub trace: bool,
    /// Share the host's loopback (stock wasmtime development): a loopback `connect`/`sendto`
    /// that finds no in-node socket goes to the network provider instead of failing, like a
    /// container on the host network. When off, a node's loopback is its own.
    pub host_loopback: bool,
    /// The network interfaces besides `lo` that netlink reports (`getifaddrs`): the interface
    /// the embedder gives the node. They are descriptive only: traffic goes through the network
    /// provider whatever they say.
    pub interfaces: Vec<crate::objects::netlink::NetInterface>,
    /// `uname -r` and `/proc/sys/kernel/osrelease`.
    pub kernel_release: String,
    /// `uname -v` and `/proc/sys/kernel/version`.
    pub kernel_version: String,
    /// Who built the kernel, as `/proc/version` says (`Linux version R (BUILDER) V`).
    pub kernel_builder: String,
    /// The id a framebuffer reports (`FBIOGET_FSCREENINFO`): this prefix and its index.
    pub fb_id: String,
    /// The snapshot container format checkpoints are written in ([`wasmshim_snapshot::Snapshot::new_in`]).
    pub snapshot_format: String,
    /// What the guest may make the node hold.
    pub limits: Limits,
}

/// Limits on what a guest can make the node hold. At a limit the guest gets the error Linux
/// gives there, and the node goes on. The defaults are far above what the programs run so far
/// need; an embedder lowers them to bound a node it does not trust.
///
/// They bound counts and file data, not the node's memory as a whole: guest memory (`mmap`,
/// `brk`), page-cache objects (files mapped shared, memfds) and kernel objects are bounded by
/// the address space the embedder's runtime gives the node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Live tasks: the threads of every process, zombies included. `clone` at the limit fails
    /// with `EAGAIN`, and `RLIMIT_NPROC` reports it.
    pub tasks: u64,
    /// The largest regular file. A write or a truncate past it fails with `EFBIG`.
    pub file_bytes: u64,
    /// Bytes of written files held in memory: files created or changed since the image was
    /// mounted ([`crate::vfs::Vfs::written_bytes`]). Growing a file past it fails with
    /// `ENOSPC`. Files only read stay in the image and do not count.
    pub total_file_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            tasks: 4096,
            file_bytes: 1 << 30,
            total_file_bytes: 2 << 30,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hostname: "linux".into(),
            quantum: 100_000,
            ns_per_insn: 1,
            trace: false,
            host_loopback: false,
            interfaces: Vec::new(),
            kernel_release: "6.6.0".into(),
            kernel_version: "#1 SMP PREEMPT".into(),
            kernel_builder: "linux".into(),
            fb_id: "fb".into(),
            snapshot_format: wasmshim_snapshot::DEFAULT_FORMAT.into(),
            limits: Limits::default(),
        }
    }
}

/// The Linux personality's types on the syscall substrate (`wasmshim-kernel`).
pub(crate) struct Linux;

impl wasmshim_kernel::Types for Linux {
    type Task = LinuxThread;
    type Key = LinuxKey;
    type Park = Park;
}

/// Linux's own wait keys (the substrate has objects, `AnyIo` and futex words).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinuxKey {
    /// A child of this process changed state (`wait4`).
    Children(Pid),
    /// A signal became pending (blocked) for this process or one of its threads
    /// (`rt_sigtimedwait`, `signalfd`).
    Signals(Pid),
}

/// Linux's own parking reasons: woken only explicitly, never by a deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Park {
    /// A `vfork` parent: resumes with the child's pid when the child execs or exits.
    Vfork { child: Pid },
    /// `rt_sigsuspend`: ends (with `EINTR`) only when a handled signal arrives.
    SigSuspend,
}

/// An object threads can wait on.
pub(crate) type WaitKey = wasmshim_kernel::KeyOf<Linux>;
/// Why a thread is not runnable.
pub(crate) type Blocker = wasmshim_kernel::BlockerOf<Linux>;
pub(crate) type ThreadState = wasmshim_kernel::TaskState<LinuxKey, Park>;
/// A thread: the substrate's task with [`LinuxThread`] as its extension (fields of both are
/// reached directly: `t.cpu`, `t.state`, `t.pid`, `t.sigmask`).
pub(crate) type Thread = wasmshim_kernel::Task<Linux>;

/// What Linux keeps per thread on top of the substrate's task.
pub(crate) struct LinuxThread {
    pub pid: Pid,
    pub clear_child_tid: u64,
    pub sigmask: u64,
    /// Signals pending for this thread specifically.
    pub pending: crate::siginfo::SigQueue,
    /// `rt_sigtimedwait` in progress: the signals waited for and the absolute deadline (kept
    /// across the restarts of one call).
    pub sigwait: Option<(u64, Option<u64>)>,
    /// The mask to restore after a signal handler interrupts `rt_sigsuspend`.
    pub saved_mask: Option<u64>,
    pub altstack: [u64; 3],
    /// The name this thread gave itself (`PR_SET_NAME`; diagnostics only, the process's
    /// `comm` keeps its current semantics).
    pub name: Vec<u8>,
}

impl LinuxThread {
    /// A fresh thread of process `pid` with signal mask `sigmask`.
    pub fn new(pid: Pid, sigmask: u64) -> Self {
        Self {
            pid,
            clear_child_tid: 0,
            sigmask,
            pending: Default::default(),
            sigwait: None,
            saved_mask: None,
            altstack: [0, 0, 2],
            name: Vec::new(),
        }
    }
}

pub(crate) struct Process {
    pub ppid: Pid,
    pub pgid: Pid,
    pub sid: Pid,
    pub mm: Rc<RefCell<AddressSpace>>,
    pub fds: Rc<RefCell<FdTable>>,
    pub cwd: Ino,
    pub umask: u32,
    pub sigactions: Rc<RefCell<[KernelSigaction; 64]>>,
    pub exe: Vec<u8>,
    pub comm: Vec<u8>,
    /// `argv` joined with NULs (`/proc/<pid>/cmdline`).
    pub cmdline: Vec<u8>,
    /// `envp` joined with NULs (`/proc/<pid>/environ`).
    pub environ: Vec<u8>,
    /// The auxiliary vector (pairs of u64, ended by `AT_NULL`) as `/proc/<pid>/auxv`: the entries
    /// the loader passed except the three that point into the stack (`AT_RANDOM`, `AT_EXECFN`,
    /// `AT_PLATFORM`). Go reads `AT_HWCAP` from here.
    pub auxv: Vec<u8>,
    pub threads: BTreeSet<Tid>,
    /// Live and zombie children.
    pub children: BTreeSet<Pid>,
    /// Signal sent to the parent on exit (`clone`'s low byte, normally `SIGCHLD`).
    pub exit_signal: u32,
    /// Signals pending for the process as a whole.
    pub pending: crate::siginfo::SigQueue,
    /// A thread blocked in `vfork` until this process execs or exits.
    pub vfork_parent: Option<Tid>,
    /// User and group ids (reported, not enforced).
    pub creds: crate::syscall::Creds,
    /// `ITIMER_REAL`, `ITIMER_VIRTUAL`, `ITIMER_PROF` (kept across `execve`, not `fork`).
    pub itimers: [crate::syscall::ITimer; 3],
    /// POSIX timers (`timer_create`; not inherited by `fork`, deleted by `execve`).
    pub timers: crate::syscall::PosixTimers,
    /// `SEM_UNDO` adjustments (applied at exit; not inherited by `fork`).
    pub semundo: crate::syscall::SemUndo,
}

/// A process that exited and has not been waited for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Zombie {
    pub ppid: Pid,
    pub pgid: Pid,
    /// Real uid (`si_uid` of `waitid`).
    pub uid: u32,
    pub status: ExitStatus,
}

/// Arguments for starting a program.
#[derive(Clone, Debug, Default)]
pub struct Spawn {
    /// Path in the VFS.
    pub path: Vec<u8>,
    pub argv: Vec<Vec<u8>>,
    pub envp: Vec<Vec<u8>>,
}

/// Why the initial program could not be started.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("cannot execute the initial program: {0:?}")]
pub struct SpawnError(pub Errno);

/// The Linux personality: one "machine" with its processes, VFS and objects, over the syscall
/// substrate's tasks, waits and scheduler (`core`).
pub struct Kernel {
    pub(crate) cfg: Config,
    pub(crate) arch: Arch,
    pub(crate) frontend: Box<dyn Frontend>,
    /// The filesystem.
    pub vfs: Vfs,
    /// The substrate: threads (as tasks), the run queue, time accounting, waits, object and
    /// address-space ids, syscall entry state.
    pub(crate) core: wasmshim_kernel::Core<Linux>,
    pub(crate) procs: BTreeMap<Pid, Process>,
    pub(crate) zombies: BTreeMap<Pid, Zombie>,
    pub(crate) next_pid: Pid,
    pub(crate) init_exit: Option<ExitStatus>,
    /// Next ephemeral port to try.
    pub(crate) next_ephemeral: u16,
    /// Bound socket names (paths, abstract names, loopback ports).
    pub(crate) unix_names: BTreeMap<
        crate::objects::unix::UnixName,
        std::rc::Weak<RefCell<crate::objects::unix::UnixSocket>>,
    >,
    /// The network provider for non-local AF_INET/AF_INET6 traffic (none: `ENETUNREACH`).
    pub(crate) net: Option<Box<dyn crate::net::Network>>,
    /// Sockets a restore brought back with their provider records, and the provider's own
    /// record: re-attached by [`Kernel::reattach_network`].
    pub(crate) net_restored: crate::net::Restored,
    /// inotify watches by watched inode.
    pub(crate) inotify_watches: BTreeMap<crate::vfs::Ino, crate::objects::inotify::Watchers>,
    /// Held file locks (`fcntl`, OFD, `flock`).
    pub(crate) locks: crate::locks::LockTable,
    /// AIO contexts by (address-space id, ring address).
    pub(crate) aio: BTreeMap<(u64, u64), crate::syscall::AioCtx>,
    /// Pseudo-terminals by pts number, alive while either side is open.
    pub(crate) ptys: BTreeMap<u32, crate::objects::tty::TtyRef>,
    /// The console terminal, when the embedder attached one ([`Kernel::attach_console`]).
    pub(crate) console: Option<crate::objects::tty::TtyRef>,
    /// HID devices (`/dev/hidraw<n>`) the embedder attached.
    pub(crate) hid: Vec<crate::devices::HidDevice>,
    /// Framebuffers (`/dev/fb<n>`) the embedder attached.
    pub(crate) fbs: Vec<crate::devices::Framebuffer>,
    /// Frames taken at vblanks, for the embedder ([`Kernel::take_frames`]).
    pub(crate) fb_frames: Vec<crate::devices::FbFrame>,
    /// SysV shared memory segments.
    pub(crate) shm: crate::shm::ShmTable,
    /// SysV semaphore sets.
    pub(crate) sem: crate::syscall::SemTable,
    /// SysV message queues.
    pub(crate) msg: crate::syscall::MsgTable,
    /// POSIX message queues by name.
    pub(crate) mqueues: BTreeMap<Vec<u8>, crate::objects::mqueue::MqRef>,
    /// Bumped whenever a signal becomes pending (`signalfd` change generation).
    pub(crate) sig_generation: u64,
    /// Armed syscall probes and the `sys` clock. Host-side only.
    pub(crate) observer: wasmshim_kernel::observe::Observer,
    /// Parsed symbols and line tables of mapped programs. Host-side only.
    pub(crate) programs: crate::symbols::Programs,
}

impl Kernel {
    /// A machine with an empty process table.
    pub fn new(cfg: Config, arch: Arch, frontend: Box<dyn Frontend>, mut vfs: Vfs) -> Self {
        vfs.set_limits(crate::vfs::DataLimits {
            file_bytes: cfg.limits.file_bytes,
            total_bytes: cfg.limits.total_file_bytes,
        });
        let sched = wasmshim_kernel::SchedConfig {
            isa: arch.isa(),
            quantum: cfg.quantum,
            ns_per_insn: cfg.ns_per_insn,
            trace: cfg.trace,
        };
        Self {
            cfg,
            arch,
            frontend,
            vfs,
            core: wasmshim_kernel::Core::new(sched),
            procs: BTreeMap::new(),
            zombies: BTreeMap::new(),
            next_pid: 1,
            init_exit: None,
            next_ephemeral: 49152,
            unix_names: BTreeMap::new(),
            net: None,
            net_restored: (None, Vec::new()),
            aio: BTreeMap::new(),
            inotify_watches: BTreeMap::new(),
            locks: crate::locks::LockTable::default(),
            ptys: BTreeMap::new(),
            console: None,
            hid: Vec::new(),
            fbs: Vec::new(),
            fb_frames: Vec::new(),
            shm: crate::shm::ShmTable::default(),
            sem: Default::default(),
            msg: Default::default(),
            mqueues: BTreeMap::new(),
            sig_generation: 0,
            observer: wasmshim_kernel::observe::Observer::default(),
            programs: crate::symbols::Programs::default(),
        }
    }

    /// Attaches a network provider: AF_INET/AF_INET6 traffic to non-local addresses goes to
    /// it from now on (loopback traffic stays in the kernel).
    pub fn set_network(&mut self, net: Box<dyn crate::net::Network>) {
        self.net = Some(net);
    }

    /// Wakes the threads waiting on sockets the provider changed.
    fn collect_network_changes(&mut self) {
        let changed = match self.net.as_mut() {
            Some(net) => net.take_changed(),
            None => return,
        };
        for id in changed {
            self.object_changed(id);
        }
    }

    /// Total instructions retired by every thread so far.
    pub fn retired(&self) -> u64 {
        self.core.retired()
    }

    /// One line per live thread: what it waits for (or that it is runnable), the last syscall
    /// it entered and the instructions it retired. For a run that stalls or hits its time
    /// limit: which thread waits on what.
    pub fn thread_report(&self) -> Vec<String> {
        self.core
            .tasks
            .values()
            .map(|t| {
                let state = match t.state {
                    ThreadState::Runnable => "runnable".to_owned(),
                    ThreadState::Blocked(b) => format!("{b:?}"),
                };
                let last = t.last_syscall.map_or_else(
                    || "-".to_owned(),
                    |n| {
                        self.arch
                            .sysno(n)
                            .map_or_else(|| format!("syscall_{n}"), |s| s.name().to_owned())
                    },
                );
                format!(
                    "thread {} (pid {}, {:?}): last={last} retired={} {state}",
                    t.tid,
                    t.pid,
                    String::from_utf8_lossy(&t.name),
                    t.own_retired()
                )
            })
            .collect()
    }

    /// Number of live processes.
    pub fn process_count(&self) -> usize {
        self.procs.len()
    }

    pub(crate) fn alloc_pid(&mut self) -> Pid {
        let pid = self.next_pid;
        self.next_pid += 1;
        pid
    }

    /// A unique address-space id (futex keys use it).
    pub(crate) fn alloc_mm_id(&mut self) -> u64 {
        self.core.new_space_id()
    }

    /// A fresh address space with a unique id.
    pub(crate) fn new_address_space(
        &mut self,
        mem: Box<dyn wasmshim_cpu::GuestMemory>,
    ) -> AddressSpace {
        let id = self.alloc_mm_id();
        AddressSpace::new(id, mem)
    }

    /// Starts the initial process (pid 1) with the host's stdio on fds 0–2.
    pub fn spawn(&mut self, spawn: &Spawn, host: &mut dyn Host) -> Result<Pid, SpawnError> {
        let pid = self.alloc_pid();
        let mut fds = FdTable::new();
        if let Some(console) = self.console.clone() {
            // One description of the console on 0, 1 and 2, and the console is the session's
            // controlling terminal with pid 1 in the foreground (like `docker run -t`).
            let ino = console.borrow().ino;
            let file = open_file(
                FileKind::Tty {
                    tty: console.clone(),
                    master: false,
                    ino,
                },
                wasmshim_linux_abi::consts::open::O_RDWR,
                b"/dev/console",
            );
            for fd in 0..3 {
                let _ = fds.insert_at(fd, file.clone(), false);
            }
            self.vfs.retain(ino);
            let mut c = console.borrow_mut();
            c.slaves += 1;
            c.session = Some(pid);
            c.pgrp = Some(pid);
        } else {
            for (fd, s) in [(0, Stdio::In), (1, Stdio::Out), (2, Stdio::Err)] {
                let flags = if fd == 0 { 0 } else { 1 };
                let _ = fds.insert_at(
                    fd,
                    open_file(FileKind::Stdio(s), flags, b"/dev/console"),
                    false,
                );
            }
        }
        let mem = self.frontend.new_memory();
        let mm = self.new_address_space(mem);
        self.procs.insert(
            pid,
            Process {
                ppid: 0,
                pgid: pid,
                sid: pid,
                mm: Rc::new(RefCell::new(mm)),
                fds: Rc::new(RefCell::new(fds)),
                cwd: ROOT,
                umask: 0o022,
                sigactions: Rc::new(RefCell::new([KernelSigaction::default(); 64])),
                exe: Vec::new(),
                comm: Vec::new(),
                cmdline: Vec::new(),
                environ: Vec::new(),
                auxv: Vec::new(),
                threads: BTreeSet::new(),
                children: BTreeSet::new(),
                exit_signal: wasmshim_linux_abi::consts::signal::SIGCHLD,
                pending: Default::default(),
                vfork_parent: None,
                creds: crate::syscall::Creds::default(),
                itimers: Default::default(),
                timers: Default::default(),
                semundo: Default::default(),
            },
        );
        let mut cpu = self.frontend.new_cpu();
        self.exec_image(pid, None, &mut cpu, spawn.clone(), None, host)
            .map_err(SpawnError)?;
        self.add_thread(Thread::new(pid, cpu, 0, LinuxThread::new(pid, 0)));
        Ok(pid)
    }

    /// Registers a runnable thread with its process and the run queue.
    pub(crate) fn add_thread(&mut self, t: Thread) {
        let (tid, pid) = (t.tid, t.pid);
        if let Some(p) = self.procs.get_mut(&pid) {
            p.threads.insert(tid);
        }
        self.core.add_task(t);
    }

    /// Makes a blocked thread runnable.
    pub(crate) fn make_runnable(&mut self, tid: Tid) {
        self.core.make_runnable(tid);
    }

    /// Wakes threads waiting on `key` ([`wasmshim_kernel::Core::wake`]). Returns the number of
    /// futex waiters woken.
    pub(crate) fn wake(&mut self, key: WaitKey, limit: usize, bitset: u32) -> usize {
        self.core.wake(key, limit, bitset)
    }

    /// Moves up to `limit` futex waiters from `from` to `to` (for `FUTEX_REQUEUE`).
    pub(crate) fn requeue(&mut self, from: WaitKey, to: WaitKey, limit: usize) -> usize {
        self.core.requeue(from, to, limit)
    }

    /// Queues a diagnostic line for the host.
    pub(crate) fn diag(&mut self, line: String) {
        self.core.diag(line);
    }

    /// Runs guest threads until the initial process exits or every thread waits.
    pub fn run(&mut self, host: &mut dyn Host) -> RunStatus {
        match wasmshim_kernel::run(&mut Machine(self), host) {
            wasmshim_kernel::Status::Exited => RunStatus::Exited(
                self.init_exit
                    .expect("the substrate reports an exit only when the personality has one"),
            ),
            wasmshim_kernel::Status::Wait { until } => RunStatus::Wait { until },
        }
    }

    /// Records that the initial process ended (the run is over).
    pub(crate) fn set_init_exit(&mut self, status: ExitStatus) {
        self.init_exit = Some(status);
    }

    /// The absolute deadline of the current blocking syscall with relative `timeout` (ns):
    /// computed at its first entry and kept while it restarts after wake-ups, so spurious
    /// wake-ups never extend it. `None` without a timeout.
    pub(crate) fn restart_deadline(
        &mut self,
        tid: Tid,
        timeout: Option<u64>,
        now: u64,
    ) -> Option<u64> {
        self.core.restart_deadline(tid, timeout, now)
    }

    /// Ends every syscall blocked (restartably) on `key` with `errno` instead of re-running it
    /// (an IPC object removed under its waiters: `EIDRM`).
    pub(crate) fn fail_waiters(&mut self, key: WaitKey, errno: Errno) {
        self.core.fail_waiters(key, errno.ret());
    }
}

impl wasmshim_kernel::Space for AddressSpace {
    fn memory(&mut self) -> &mut dyn wasmshim_cpu::GuestMemory {
        self.mem.as_mut()
    }
}

/// The kernel as the substrate's run loop sees it (a private view, so the personality's types
/// stay crate-private).
struct Machine<'a>(&'a mut Kernel);

/// The hooks the substrate's run loop calls, in its order.
impl wasmshim_kernel::Personality for Machine<'_> {
    type T = Linux;
    type Space = AddressSpace;

    fn core(&self) -> &wasmshim_kernel::Core<Linux> {
        &self.0.core
    }

    fn core_mut(&mut self) -> &mut wasmshim_kernel::Core<Linux> {
        &mut self.0.core
    }

    fn finished(&self) -> bool {
        self.0.init_exit.is_some()
    }

    fn observer(&mut self) -> Option<&mut wasmshim_kernel::observe::Observer> {
        Some(&mut self.0.observer)
    }

    fn process_of(&self, tid: Tid) -> Option<u32> {
        self.0.core.tasks.get(&tid).map(|_| self.0.pid_of(tid))
    }

    fn symbolize(&mut self, tid: Tid, pc: u64) -> Option<String> {
        let pid = self.0.pid_of(tid);
        self.0.symbolize_pc(pid, pc)
    }

    fn source_line(&mut self, tid: Tid, pc: u64) -> Option<(String, u32, bool)> {
        let pid = self.0.pid_of(tid);
        self.0.source_line_at(pid, pc)
    }

    /// The kernel itself, for a debugger at a stop (it downcasts to [`Kernel`] and uses
    /// [`crate::debug`]).
    fn as_any(&mut self) -> Option<&mut dyn core::any::Any> {
        Some(&mut *self.0)
    }

    fn space(&self, tid: Tid) -> Option<Rc<RefCell<AddressSpace>>> {
        let pid = self.0.core.tasks.get(&tid)?.pid;
        Some(self.0.procs.get(&pid)?.mm.clone())
    }

    fn syscall(&mut self, tid: Tid, frame: SyscallFrame, host: &mut dyn Host) -> Flow {
        self.0.dispatch(tid, frame, host)
    }

    fn fault(&mut self, tid: Tid, fault: Fault, host: &mut dyn Host) {
        self.0.fault_signal(tid, fault, host);
    }

    fn exit_thread(&mut self, tid: Tid, code: i32) {
        self.0.exit_thread(tid, code);
    }

    fn exit_group(&mut self, tid: Tid, code: i32) {
        if let Some(pid) = self.0.core.tasks.get(&tid).map(|t| t.pid) {
            self.0.exit_process(pid, ExitStatus::Code(code as u8));
        }
    }

    fn futex_timeout(&self) -> u64 {
        Errno::ETIMEDOUT.ret()
    }

    fn on_run(&mut self) {
        self.0.collect_network_changes();
    }

    fn expire(&mut self, now: u64) {
        self.0.expire_itimers(now);
        self.0.expire_posix_timers(now);
        self.0.expire_vblanks(now);
    }

    fn next_deadline(&self) -> Option<u64> {
        self.0
            .next_itimer()
            .into_iter()
            .chain(self.0.next_posix_timer())
            .chain(self.0.next_vblank())
            .min()
    }

    fn before_run(&mut self, tid: Tid, host: &mut dyn Host) -> bool {
        self.0.deliver_signals(tid, host)
    }

    fn after_syscall(&mut self, host: &mut dyn Host) {
        self.0.dispatch_fs_events();
        self.0.flush_console(host);
    }

    fn describe(&self, number: u64) -> Option<(&'static str, usize)> {
        self.0
            .arch
            .sysno(number)
            .map(|s| (s.name(), crate::syscall::nargs(s)))
    }

    /// `[pid]`, or `[pid:tid]` for threads other than a process's first.
    fn label(&self, tid: Tid) -> String {
        let pid = self.0.pid_of(tid);
        if tid == pid {
            format!("{pid}")
        } else {
            format!("{pid}:{tid}")
        }
    }

    fn show_error(&self, ret: u64) -> Option<String> {
        Errno::from_ret(ret).map(|e| format!("-1 {e:?}"))
    }
}

#[cfg(test)]
#[path = "kernel_tests.rs"]
mod tests;
