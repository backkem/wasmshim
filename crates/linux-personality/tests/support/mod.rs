//! A scripted guest: runs the personality's system calls without a CPU emulator.
//!
//! This is also the smallest complete example of embedding the personality:
//!
//! 1. a [`Frontend`] that makes CPUs and address spaces ([`ScriptedFrontend`]),
//! 2. a [`Host`] that serves time, entropy and stdio ([`TestHost`]: a virtual clock that only
//!    moves when the kernel asks it to, a seeded generator, captured output),
//! 3. a [`Vfs`] with a root image mounted (a tar built here, with the personality's own index in
//!    front),
//! 4. [`Kernel::spawn`] of the first program, then [`Kernel::run`] in a loop that advances the
//!    virtual clock to whatever deadline [`RunStatus::Wait`] names.
//!
//! The CPU ([`ScriptedCpu`]) never executes an instruction. Its "program" is a Rust closure that
//! runs on a thread of its own and talks to the CPU over two channels in strict alternation:
//! while the closure runs, the kernel thread waits inside [`GuestCpu::run_until_event`]; while the
//! kernel runs, the closure waits for its syscall's result. Only one side runs at a time, so a
//! run is as deterministic as the personality itself.
//!
//! Each system call the closure makes becomes an [`Event::Syscall`] at a fixed address
//! ([`SVC_PC`]). The kernel's answers come back the way they would on hardware:
//!
//! * a result: `set_syscall_result` writes `x0`, and the PC is past the call;
//! * a blocked call that restarts: the kernel rewinds the PC onto the call, and the CPU issues
//!   the same call again when it is next scheduled;
//! * a signal handler: the kernel points the PC at the handler. Handlers here are all
//!   [`HANDLER`]: the CPU records the signal number ([`Guest::signals`]) and returns from the
//!   handler at once with `rt_sigreturn`, which restores the interrupted state.
//!
//! `clone` gives the child its own closure ([`Guest::fork`], [`Guest::spawn_thread`]):
//! `duplicate()` hands it to the new CPU. `execve` can do the same ([`Guest::execve`]): the
//! fresh CPU the kernel makes for the new image runs the next closure.

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::panic::{self, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::mpsc::{Receiver, Sender, channel};

use wasmshim_cpu::{
    A64Context, CpuContext, Event, GuestCpu, GuestMemory, PagedMemory, SyscallFrame,
};
use wasmshim_linux::vfs::{Vfs, index};
use wasmshim_linux::{
    Arch, Blob, BlobError, Config, ExitStatus, Frontend, Host, Kernel, RunStatus, Spawn,
};
pub use wasmshim_linux_abi::consts::at::AT_FDCWD;
pub use wasmshim_linux_abi::{Errno, Sysno};

/// Where the program's (only) supervisor call sits: the ELF entry point.
pub const SVC_PC: u64 = 0x40_0078;
/// The address every signal handler is installed at (never mapped, never executed).
pub const HANDLER: u64 = 0x7700_0000;
/// The program every run starts (a static AArch64 ELF in the root image).
pub const PROGRAM: &str = "/bin/prog";
/// Virtual time when the machine starts (`CLOCK_MONOTONIC`), and the wall clock then.
pub const START_NS: u64 = 1_000_000_000;
pub const EPOCH_NS: u64 = 1_700_000_000_000_000_000;

/// The size of the scratch mapping each program allocates its strings and buffers from.
const ARENA: u64 = 16 << 20;

/// A guest program: an ordinary closure. Returning ends it with status 0 (`exit_group(0)` for a
/// process, `exit(0)` for a thread); [`Guest::exit`] ends it with another status.
pub type Body = Box<dyn FnOnce(&mut Guest) + Send>;

// ---------------------------------------------------------------------------------------------
// Running a machine

/// What a run left behind.
pub struct Outcome {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// The personality's diagnostic lines (trace, fatal signals).
    pub diag: Vec<String>,
    /// Virtual time at the end.
    pub now: u64,
    /// The machine, for looking at its state afterwards (the VFS is public).
    pub kernel: Kernel,
}

/// A machine to run one program on.
pub struct Machine {
    pub config: Config,
    /// Seeds the host's entropy (`getrandom`, `AT_RANDOM`, `/dev/urandom`).
    pub seed: u64,
    pub stdin: Vec<u8>,
}

impl Default for Machine {
    fn default() -> Self {
        Self {
            config: Config {
                hostname: "testhost".into(),
                // Instructions cost no time here: the clock moves only by sleeps and timeouts.
                ns_per_insn: 0,
                ..Config::default()
            },
            seed: 1,
            stdin: Vec::new(),
        }
    }
}

impl Machine {
    /// Boots the machine with [`root_image`], runs `body` as pid 1 and returns when pid 1 ends.
    ///
    /// Panics (failing the test) if the guest panics, if every thread waits with nothing that
    /// could wake it, or if virtual time passes an hour.
    pub fn run(self, body: impl FnOnce(&mut Guest) + Send + 'static) -> Outcome {
        let mut vfs = Vfs::new();
        vfs.mount_image(Box::new(MemBlob(root_image())))
            .expect("the root image mounts");
        let next = Rc::new(RefCell::new(None));
        let frontend = ScriptedFrontend { next: next.clone() };
        let mut kernel = Kernel::new(self.config, Arch::Aarch64, Box::new(frontend), vfs);
        let mut host = TestHost::new(self.seed, self.stdin);
        // The CPU the kernel makes for the loaded image runs this.
        *next.borrow_mut() = Some(Start::process(Box::new(body)));
        let spawn = Spawn {
            path: PROGRAM.into(),
            argv: vec![PROGRAM.into()],
            envp: vec![b"HOME=/".to_vec()],
        };
        kernel.spawn(&spawn, &mut host).expect("the program loads");
        let status = loop {
            let step = kernel.run(&mut host);
            // The count of written file bytes is kept up to date as files change: after every
            // step of every test it must equal the count taken from scratch.
            assert_eq!(
                kernel.vfs.written_bytes(),
                kernel.vfs.recount_written(),
                "the count of written file bytes drifted"
            );
            match step {
                RunStatus::Exited(status) => break status,
                RunStatus::Wait { until: Some(t) } => {
                    host.now = host.now.max(t);
                    assert!(
                        host.now < START_NS + 3600 * 1_000_000_000,
                        "virtual time ran past an hour:\n{}",
                        kernel.thread_report().join("\n")
                    );
                }
                RunStatus::Wait { until: None } => panic!(
                    "guest deadlock: every thread waits and nothing can wake it\n{}",
                    kernel.thread_report().join("\n")
                ),
            }
        };
        Outcome {
            status,
            stdout: host.stdout,
            stderr: host.stderr,
            diag: host.diag,
            now: host.now,
            kernel,
        }
    }
}

/// Runs `body` on a default machine and checks that it exited with status 0.
pub fn run(body: impl FnOnce(&mut Guest) + Send + 'static) -> Outcome {
    let out = Machine::default().run(body);
    assert_eq!(
        out.status,
        ExitStatus::Code(0),
        "the program failed; diagnostics: {:?}",
        out.diag
    );
    out
}

// ---------------------------------------------------------------------------------------------
// The root image

/// The root filesystem every run starts from: the program, a few files, directories and links.
pub fn root_image() -> Vec<u8> {
    let big: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
    let entries: Vec<(&str, u8, u32, &[u8])> = vec![
        ("bin/", b'5', 0o755, b""),
        ("bin/prog", b'0', 0o755, &ELF),
        ("etc/", b'5', 0o755, b""),
        ("etc/hostname", b'0', 0o644, b"testhost\n"),
        ("etc/readonly", b'0', 0o444, b"read me\n"),
        ("data/", b'5', 0o755, b""),
        ("data/big", b'0', 0o644, &big),
        ("data/link", b'2', 0o777, b"big"),
        ("data/dangling", b'2', 0o777, b"nowhere"),
        ("data/sub/", b'5', 0o755, b""),
        ("data/sub/one", b'0', 0o644, b"1"),
        ("data/sub/two", b'0', 0o644, b"22"),
        ("loop/", b'5', 0o755, b""),
        ("loop/a", b'2', 0o777, b"b"),
        ("loop/b", b'2', 0o777, b"a"),
        ("empty/", b'5', 0o755, b""),
    ];
    let mut tar = Vec::new();
    for (path, kind, mode, data) in entries {
        tar_member(&mut tar, path, kind, mode, data);
    }
    tar.extend_from_slice(&[0; 1024]);
    // The personality's own index writer: mounting reads the index, not every header.
    index::with_index(&tar).expect("a well-formed archive")
}

/// Appends one ustar member: `kind` is the typeflag (`0` file, `2` symlink, `5` directory);
/// `data` is a file's contents or a link's target.
fn tar_member(out: &mut Vec<u8>, path: &str, kind: u8, mode: u32, data: &[u8]) {
    let mut h = [0u8; 512];
    h[..path.len()].copy_from_slice(path.as_bytes());
    let size = if kind == b'0' { data.len() } else { 0 };
    for (at, len, v) in [
        (100, 8, u64::from(mode)),
        (108, 8, 0),
        (116, 8, 0),
        (124, 12, size as u64),
        (136, 12, 1_700_000_000),
    ] {
        h[at..at + len].copy_from_slice(format!("{v:0w$o}\0", w = len - 1).as_bytes());
    }
    h[156] = kind;
    if kind == b'2' {
        h[157..157 + data.len()].copy_from_slice(data);
    }
    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");
    h[148..156].fill(b' ');
    let sum: u64 = h.iter().map(|&b| u64::from(b)).sum();
    h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
    out.extend_from_slice(&h);
    if kind == b'0' {
        out.extend_from_slice(data);
        out.resize(out.len().div_ceil(512) * 512, 0);
    }
}

/// A minimal static AArch64 executable: an ELF header and one `PT_LOAD` segment (read and
/// execute) at 0x400000 holding the headers and a single `svc #0` at the entry point. It is
/// loaded, never executed.
const ELF: [u8; 128] = {
    let e = [0u8; 128];
    let e = put(e, 0, b"\x7fELF\x02\x01\x01"); // 64-bit, little-endian, version 1
    let e = put(e, 16, &2u16.to_le_bytes()); // ET_EXEC
    let e = put(e, 18, &183u16.to_le_bytes()); // EM_AARCH64
    let e = put(e, 20, &1u32.to_le_bytes()); // e_version
    let e = put(e, 24, &SVC_PC.to_le_bytes()); // e_entry
    let e = put(e, 32, &64u64.to_le_bytes()); // e_phoff
    let e = put(e, 52, &64u16.to_le_bytes()); // e_ehsize
    let e = put(e, 54, &56u16.to_le_bytes()); // e_phentsize
    let e = put(e, 56, &1u16.to_le_bytes()); // e_phnum
    let e = put(e, 64, &1u32.to_le_bytes()); // PT_LOAD (p_offset stays 0)
    let e = put(e, 68, &5u32.to_le_bytes()); // PF_R | PF_X
    let e = put(e, 80, &0x40_0000u64.to_le_bytes()); // p_vaddr
    let e = put(e, 88, &0x40_0000u64.to_le_bytes()); // p_paddr
    let e = put(e, 96, &128u64.to_le_bytes()); // p_filesz
    let e = put(e, 104, &128u64.to_le_bytes()); // p_memsz
    let e = put(e, 112, &0x1000u64.to_le_bytes()); // p_align
    put(e, 120, &0xd400_0001u32.to_le_bytes()) // svc #0
};

/// `e` with `bytes` written at `at` (builds [`ELF`] at compile time).
const fn put(mut e: [u8; 128], at: usize, bytes: &[u8]) -> [u8; 128] {
    let mut i = 0;
    while i < bytes.len() {
        e[at + i] = bytes[i];
        i += 1;
    }
    e
}

struct MemBlob(Vec<u8>);

impl Blob for MemBlob {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), BlobError> {
        let start = usize::try_from(offset).map_err(|_| BlobError::OutOfRange)?;
        let src = start
            .checked_add(buf.len())
            .and_then(|end| self.0.get(start..end))
            .ok_or(BlobError::OutOfRange)?;
        buf.copy_from_slice(src);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// The host

/// Virtual time, seeded entropy and captured stdio.
pub struct TestHost {
    /// `CLOCK_MONOTONIC` in ns; the run loop moves it to each deadline the kernel names.
    pub now: u64,
    rng: u64,
    stdin: Vec<u8>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub diag: Vec<String>,
}

impl TestHost {
    fn new(seed: u64, stdin: Vec<u8>) -> Self {
        Self {
            now: START_NS,
            rng: seed,
            stdin,
            stdout: Vec::new(),
            stderr: Vec::new(),
            diag: Vec::new(),
        }
    }
}

impl Host for TestHost {
    fn monotonic_ns(&mut self) -> u64 {
        self.now
    }
    fn realtime_ns(&mut self) -> u64 {
        EPOCH_NS + (self.now - START_NS)
    }
    fn random(&mut self, buf: &mut [u8]) {
        // splitmix64
        for chunk in buf.chunks_mut(8) {
            self.rng = self.rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            chunk.copy_from_slice(&z.to_le_bytes()[..chunk.len()]);
        }
    }
    fn stdout(&mut self, data: &[u8]) {
        self.stdout.extend_from_slice(data);
    }
    fn stderr(&mut self, data: &[u8]) {
        self.stderr.extend_from_slice(data);
    }
    fn stdin(&mut self, buf: &mut [u8]) -> usize {
        let n = buf.len().min(self.stdin.len());
        buf[..n].copy_from_slice(&self.stdin[..n]);
        self.stdin.drain(..n);
        n
    }
    fn diag(&mut self, line: &str) {
        self.diag.push(line.to_owned());
    }
}

// ---------------------------------------------------------------------------------------------
// The frontend and the scripted CPU

/// Makes scripted CPUs and plain paged address spaces. `next` is the program the next CPU that
/// starts without one runs: the first program, or the image of a successful `execve`.
struct ScriptedFrontend {
    next: Rc<RefCell<Option<Start>>>,
}

impl Frontend for ScriptedFrontend {
    fn new_cpu(&self) -> Box<dyn GuestCpu> {
        Box::new(ScriptedCpu::new(self.next.clone()))
    }
    fn new_memory(&self) -> Box<dyn GuestMemory> {
        Box::new(PagedMemory::new())
    }
}

/// How a program starts: its closure, the scratch arena it inherits (a fork or a thread) or
/// none (it maps its own), and whether it is a thread (it ends with `exit`, not `exit_group`).
struct Start {
    body: Body,
    arena: Option<Arena>,
    thread: bool,
}

impl Start {
    fn process(body: Body) -> Self {
        Self {
            body,
            arena: None,
            thread: false,
        }
    }
}

/// A bump allocator over guest memory.
#[derive(Clone, Copy, Debug)]
struct Arena {
    next: u64,
    end: u64,
}

/// What the program asks of its CPU.
enum Request {
    Syscall {
        number: u64,
        args: [u64; 6],
        /// The child's program, for a `clone`.
        child: Option<Start>,
        /// The new image's program, for an `execve`.
        exec: Option<Start>,
    },
    Read {
        addr: u64,
        len: usize,
    },
    Write {
        addr: u64,
        data: Vec<u8>,
    },
    Exit {
        code: i32,
        thread: bool,
    },
    Panicked(Box<dyn Any + Send>),
}

/// What the CPU answers.
enum Reply {
    Ret { value: u64, signals: Vec<u32> },
    Read(Option<Vec<u8>>),
    Wrote(bool),
}

/// The unwinding payload of a program whose CPU is gone (its process was killed or exited).
struct Killed;

/// A [`GuestCpu`] whose instructions are a Rust closure's system calls.
pub struct ScriptedCpu {
    ctx: A64Context,
    retired: u64,
    /// The program to run when first scheduled (a clone's child).
    start: Option<Start>,
    /// The frontend's slot for a CPU that starts without a program.
    next: Rc<RefCell<Option<Start>>>,
    /// The running program's channels, once started.
    link: Option<(Sender<Reply>, Receiver<Request>)>,
    /// The program's syscall in progress (its frame, for a restart).
    inflight: Option<SyscallFrame>,
    /// The child program of a `clone` in progress, for `duplicate`.
    child: RefCell<Option<Start>>,
    /// This CPU's `execve` put a program in `next` (taken back if the call fails).
    exec_pending: bool,
    /// Signals whose handler ran since the program last heard from the CPU.
    signals: Vec<u32>,
}

impl ScriptedCpu {
    fn new(next: Rc<RefCell<Option<Start>>>) -> Self {
        Self {
            ctx: A64Context::default(),
            retired: 0,
            start: None,
            next,
            link: None,
            inflight: None,
            child: RefCell::new(None),
            exec_pending: false,
            signals: Vec::new(),
        }
    }

    /// Starts the program on its own thread.
    fn begin(&mut self) {
        let start = self
            .start
            .take()
            .or_else(|| self.next.borrow_mut().take())
            .expect("a CPU started with no program to run");
        let (reply_tx, reply_rx) = channel();
        let (req_tx, req_rx) = channel();
        std::thread::spawn(move || program_thread(start, req_tx, reply_rx));
        self.link = Some((reply_tx, req_rx));
    }

    /// Issues a syscall at [`SVC_PC`].
    fn issue(&mut self, number: u64, args: [u64; 6]) -> Event {
        let frame = SyscallFrame {
            number,
            pc: SVC_PC + 4,
            args,
        };
        self.ctx.x[..6].copy_from_slice(&args);
        self.ctx.x[8] = number;
        self.ctx.pc = frame.pc;
        self.inflight = Some(frame);
        self.retired += 1;
        Event::Syscall(frame)
    }

    /// Serves the program's memory requests until it makes its next syscall.
    fn serve(&mut self, mem: &mut dyn GuestMemory) -> Event {
        loop {
            let (tx, rx) = self.link.as_ref().expect("a started program");
            let request = rx
                .recv()
                .expect("the program's thread ended without a word");
            match request {
                Request::Read { addr, len } => {
                    let mut buf = vec![0; len];
                    let ok = mem.read(addr, &mut buf).is_ok();
                    let _ = tx.send(Reply::Read(ok.then_some(buf)));
                }
                Request::Write { addr, data } => {
                    let _ = tx.send(Reply::Wrote(mem.write(addr, &data).is_ok()));
                }
                Request::Syscall {
                    number,
                    args,
                    child,
                    exec,
                } => {
                    *self.child.borrow_mut() = child;
                    if exec.is_some() {
                        *self.next.borrow_mut() = exec;
                        self.exec_pending = true;
                    }
                    return self.issue(number, args);
                }
                Request::Exit { code, thread } => {
                    let number = if thread {
                        Sysno::exit
                    } else {
                        Sysno::exit_group
                    };
                    return self.issue(number.generic(), [code as u64, 0, 0, 0, 0, 0]);
                }
                // The program failed an assertion: fail the test with its message.
                Request::Panicked(payload) => panic::resume_unwind(payload),
            }
        }
    }
}

impl GuestCpu for ScriptedCpu {
    fn run_until_event(&mut self, mem: &mut dyn GuestMemory, _budget: u64) -> Event {
        if self.ctx.pc == HANDLER {
            // A signal handler "runs": note the signal and return from it.
            self.signals.push(self.ctx.x[0] as u32);
            self.ctx.pc = HANDLER + 4;
            self.retired += 1;
            return Event::Syscall(SyscallFrame {
                number: Sysno::rt_sigreturn.generic(),
                pc: HANDLER + 4,
                args: [0; 6],
            });
        }
        match self.inflight {
            // A blocked call the kernel rewound: issue it again.
            Some(frame) if self.ctx.pc == frame.pc - 4 => {
                self.ctx.pc = frame.pc;
                self.retired += 1;
                return Event::Syscall(frame);
            }
            Some(frame) => {
                assert_eq!(
                    self.ctx.pc, frame.pc,
                    "the scripted CPU is at an address it never ran"
                );
                self.inflight = None;
                // A failed `execve` or `clone` leaves its program behind.
                if std::mem::take(&mut self.exec_pending) {
                    self.next.borrow_mut().take();
                }
                self.child.borrow_mut().take();
                match &self.link {
                    Some((tx, _)) => {
                        let reply = Reply::Ret {
                            value: self.ctx.x[0],
                            signals: std::mem::take(&mut self.signals),
                        };
                        if tx.send(reply).is_err() {
                            panic!("the program's thread is gone");
                        }
                    }
                    // A clone's child: its program starts now (its result, 0, is implied).
                    None => self.begin(),
                }
            }
            None => self.begin(),
        }
        self.serve(mem)
    }

    fn set_syscall_result(&mut self, value: u64) {
        self.ctx.x[0] = value;
    }
    fn pc(&self) -> u64 {
        self.ctx.pc
    }
    fn set_pc(&mut self, pc: u64) {
        self.ctx.pc = pc;
    }
    fn sp(&self) -> u64 {
        self.ctx.sp
    }
    fn set_sp(&mut self, sp: u64) {
        self.ctx.sp = sp;
    }
    fn thread_pointer(&self) -> u64 {
        self.ctx.tp
    }
    fn set_thread_pointer(&mut self, tp: u64) {
        self.ctx.tp = tp;
    }
    fn retired(&self) -> u64 {
        self.retired
    }
    fn set_virtual_time(&mut self, _ns: u64) {}
    fn context(&self) -> CpuContext {
        CpuContext::A64(self.ctx.clone())
    }
    fn set_context(&mut self, ctx: &CpuContext) {
        let CpuContext::A64(c) = ctx;
        self.ctx = c.clone();
    }
    fn duplicate(&self) -> Box<dyn GuestCpu> {
        let start = self
            .child
            .borrow_mut()
            .take()
            .expect("clone without a child program: use Guest::fork or Guest::spawn_thread");
        Box::new(Self {
            ctx: self.ctx.clone(),
            retired: self.retired,
            start: Some(start),
            next: self.next.clone(),
            link: None,
            inflight: self.inflight,
            child: RefCell::new(None),
            exec_pending: false,
            signals: Vec::new(),
        })
    }
}

/// The body of a program's thread: run the closure, then report how it ended.
fn program_thread(start: Start, tx: Sender<Request>, rx: Receiver<Reply>) {
    let Start {
        body,
        arena,
        thread,
    } = start;
    let mut g = Guest {
        tx,
        rx,
        arena: arena.unwrap_or(Arena { next: 0, end: 0 }),
        thread,
        signals: Vec::new(),
    };
    let ended = panic::catch_unwind(AssertUnwindSafe(|| {
        if arena.is_none() {
            g.map_arena();
        }
        body(&mut g);
    }));
    match ended {
        Ok(()) => {
            let _ = g.tx.send(Request::Exit { code: 0, thread });
        }
        Err(payload) if payload.is::<Killed>() => {}
        Err(payload) => {
            let _ = g.tx.send(Request::Panicked(payload));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The program's side

/// A running guest program: system calls and access to its own memory.
pub struct Guest {
    tx: Sender<Request>,
    rx: Receiver<Reply>,
    arena: Arena,
    /// A thread of a process (it ends with `exit`) rather than a process.
    thread: bool,
    signals: Vec<u32>,
}

/// The CPU is gone (the process was killed or ended): unwind this thread quietly.
fn killed() -> ! {
    panic::resume_unwind(Box::new(Killed))
}

impl Guest {
    fn ask(&mut self, request: Request) -> Reply {
        if self.tx.send(request).is_err() {
            killed();
        }
        match self.rx.recv() {
            Ok(reply) => reply,
            Err(_) => killed(),
        }
    }

    fn call(
        &mut self,
        number: u64,
        args: &[u64],
        child: Option<Start>,
        exec: Option<Start>,
    ) -> i64 {
        let mut a = [0u64; 6];
        a[..args.len()].copy_from_slice(args);
        match self.ask(Request::Syscall {
            number,
            args: a,
            child,
            exec,
        }) {
            Reply::Ret { value, signals } => {
                self.signals.extend(signals);
                value as i64
            }
            _ => unreachable!("a syscall's reply is its result"),
        }
    }

    /// A raw system call: up to six arguments, the raw result (`-errno` on failure).
    pub fn syscall(&mut self, sys: Sysno, args: &[u64]) -> i64 {
        self.call(sys.generic(), args, None, None)
    }

    /// A system call by number (for numbers no [`Sysno`] names).
    pub fn syscall_nr(&mut self, number: u64, args: &[u64]) -> i64 {
        self.call(number, args, None, None)
    }

    /// Ends this program (process: `exit_group`; thread: `exit`) with `code`.
    pub fn exit(&mut self, code: i32) -> ! {
        let thread = self.thread;
        let _ = self.tx.send(Request::Exit { code, thread });
        loop {
            if self.rx.recv().is_err() {
                killed();
            }
        }
    }

    /// Signals whose handler ([`HANDLER`]) ran since the last call, in delivery order.
    pub fn signals(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.signals)
    }

    // ---- memory ----

    fn map_arena(&mut self) {
        use wasmshim_linux_abi::consts::{map, prot};
        let at = self.syscall(
            Sysno::mmap,
            &[
                0,
                ARENA,
                prot::PROT_READ | prot::PROT_WRITE,
                map::MAP_PRIVATE | map::MAP_ANONYMOUS,
                u64::MAX,
                0,
            ],
        );
        assert!(at > 0, "the scratch mapping failed: {at}");
        self.arena = Arena {
            next: at as u64,
            end: at as u64 + ARENA,
        };
    }

    /// `len` zeroed bytes of guest memory (16-byte aligned).
    pub fn alloc(&mut self, len: usize) -> u64 {
        let at = self.arena.next.next_multiple_of(16);
        assert!(
            at + len as u64 <= self.arena.end,
            "the scratch arena is full"
        );
        self.arena.next = at + len as u64;
        at
    }

    /// A copy of `data` in guest memory.
    pub fn put(&mut self, data: &[u8]) -> u64 {
        let at = self.alloc(data.len().max(1));
        self.write_mem(at, data);
        at
    }

    /// A NUL-terminated copy of `s` in guest memory.
    pub fn cstr(&mut self, s: &str) -> u64 {
        let mut b = s.as_bytes().to_vec();
        b.push(0);
        self.put(&b)
    }

    /// Reads guest memory as user code would; `None` if any byte is unreadable.
    pub fn try_bytes(&mut self, addr: u64, len: usize) -> Option<Vec<u8>> {
        match self.ask(Request::Read { addr, len }) {
            Reply::Read(r) => r,
            _ => unreachable!(),
        }
    }

    /// Reads `len` bytes of guest memory (a negative or huge `len`, as a failed call returns,
    /// is a test bug).
    pub fn bytes(&mut self, addr: u64, len: impl TryInto<usize>) -> Vec<u8> {
        let len = len.try_into().ok().expect("a valid length");
        self.try_bytes(addr, len)
            .unwrap_or_else(|| panic!("guest memory at {addr:#x} is not readable"))
    }

    /// Writes guest memory as user code would; `false` if it faults.
    pub fn try_write_mem(&mut self, addr: u64, data: &[u8]) -> bool {
        match self.ask(Request::Write {
            addr,
            data: data.to_vec(),
        }) {
            Reply::Wrote(ok) => ok,
            _ => unreachable!(),
        }
    }

    pub fn write_mem(&mut self, addr: u64, data: &[u8]) {
        assert!(
            self.try_write_mem(addr, data),
            "guest memory at {addr:#x} is not writable"
        );
    }

    pub fn u32_at(&mut self, addr: u64) -> u32 {
        u32::from_le_bytes(self.bytes(addr, 4usize).try_into().unwrap())
    }

    pub fn u64_at(&mut self, addr: u64) -> u64 {
        u64::from_le_bytes(self.bytes(addr, 8usize).try_into().unwrap())
    }

    pub fn i32_at(&mut self, addr: u64) -> i32 {
        self.u32_at(addr) as i32
    }

    // ---- processes ----

    /// `fork()` as `clone(SIGCHLD)`: the child runs `child` (with a copy of this program's
    /// memory, so pointers into the arena stay valid) and exits with 0 when it returns. Returns
    /// the child's pid to the parent.
    pub fn fork(&mut self, child: impl FnOnce(&mut Guest) + Send + 'static) -> i64 {
        let start = Start {
            body: Box::new(child),
            arena: Some(self.arena),
            thread: false,
        };
        let flags = u64::from(wasmshim_linux_abi::consts::signal::SIGCHLD);
        self.call(Sysno::clone.generic(), &[flags], Some(start), None)
    }

    /// A thread of this process (`CLONE_VM | CLONE_THREAD | ...`) running `body`, with a stack
    /// and an arena of its own carved from this program's. Returns its tid.
    pub fn spawn_thread(&mut self, body: impl FnOnce(&mut Guest) + Send + 'static) -> i64 {
        use wasmshim_linux_abi::consts::clone::*;
        let stack = self.alloc(64 << 10) + (64 << 10);
        let base = self.alloc(1 << 20);
        let start = Start {
            body: Box::new(body),
            arena: Some(Arena {
                next: base,
                end: base + (1 << 20),
            }),
            thread: true,
        };
        let flags =
            CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD | CLONE_SYSVSEM;
        self.call(Sysno::clone.generic(), &[flags, stack], Some(start), None)
    }

    /// `execve(path, argv, envp)`; on success the new image runs `next` and this call never
    /// returns.
    pub fn execve(
        &mut self,
        path: &str,
        argv: &[&str],
        next: impl FnOnce(&mut Guest) + Send + 'static,
    ) -> i64 {
        let path = self.cstr(path);
        let argv = self.ptr_array(argv);
        let envp = self.ptr_array(&[]);
        self.call(
            Sysno::execve.generic(),
            &[path, argv, envp],
            None,
            Some(Start::process(Box::new(next))),
        )
    }

    /// A NULL-terminated array of C strings.
    pub fn ptr_array(&mut self, items: &[&str]) -> u64 {
        let ptrs: Vec<u64> = items.iter().map(|s| self.cstr(s)).collect();
        let mut raw: Vec<u8> = ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
        raw.extend_from_slice(&0u64.to_le_bytes());
        self.put(&raw)
    }

    /// `wait4(pid, &status, options, NULL)`: the result and the status word.
    pub fn wait4(&mut self, pid: i64, options: u64) -> (i64, u32) {
        let status = self.alloc(4);
        let r = self.syscall(Sysno::wait4, &[pid as u64, status, options, 0]);
        (r, self.u32_at(status))
    }

    /// Installs [`HANDLER`] for `sig` with `flags` (`SA_RESTART`, ...).
    pub fn handle(&mut self, sig: u32, flags: u64) -> i64 {
        let act = self.put(
            &[HANDLER, flags, 0, 0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        self.syscall(Sysno::rt_sigaction, &[u64::from(sig), act, 0, 8])
    }

    // ---- common calls, spelled like C ----

    pub fn openat(&mut self, dirfd: i32, path: &str, flags: u64, mode: u64) -> i64 {
        let p = self.cstr(path);
        self.syscall(Sysno::openat, &[dirfd as u64, p, flags, mode])
    }

    pub fn open(&mut self, path: &str, flags: u64) -> i64 {
        self.openat(AT_FDCWD, path, flags, 0o644)
    }

    pub fn close(&mut self, fd: i64) -> i64 {
        self.syscall(Sysno::close, &[fd as u64])
    }

    pub fn read(&mut self, fd: i64, buf: u64, len: u64) -> i64 {
        self.syscall(Sysno::read, &[fd as u64, buf, len])
    }

    /// `read` into a fresh buffer; the bytes read (panics on an error).
    pub fn read_vec(&mut self, fd: i64, len: usize) -> Vec<u8> {
        let buf = self.alloc(len);
        let n = self.read(fd, buf, len as u64);
        assert!(n >= 0, "read({fd}) failed: {n}");
        self.bytes(buf, n)
    }

    pub fn write(&mut self, fd: i64, buf: u64, len: u64) -> i64 {
        self.syscall(Sysno::write, &[fd as u64, buf, len])
    }

    /// `write` of `data` (copied into guest memory first).
    pub fn write_all(&mut self, fd: i64, data: &[u8]) -> i64 {
        let buf = self.put(data);
        self.write(fd, buf, data.len() as u64)
    }

    pub fn lseek(&mut self, fd: i64, off: i64, whence: u64) -> i64 {
        self.syscall(Sysno::lseek, &[fd as u64, off as u64, whence])
    }

    /// `fstat` into a fresh buffer: the result and the parsed `struct stat`.
    pub fn fstat(&mut self, fd: i64) -> (i64, Stat) {
        let buf = self.alloc(128);
        let r = self.syscall(Sysno::fstat, &[fd as u64, buf]);
        (r, Stat::parse(&self.bytes(buf, 128usize)))
    }

    /// `newfstatat(dirfd, path, &st, flags)`.
    pub fn stat_at(&mut self, dirfd: i32, path: &str, flags: u64) -> (i64, Stat) {
        let p = self.cstr(path);
        let buf = self.alloc(128);
        let r = self.syscall(Sysno::newfstatat, &[dirfd as u64, p, buf, flags]);
        (r, Stat::parse(&self.bytes(buf, 128usize)))
    }

    pub fn stat(&mut self, path: &str) -> (i64, Stat) {
        self.stat_at(AT_FDCWD, path, 0)
    }

    pub fn mkdir(&mut self, path: &str, mode: u64) -> i64 {
        let p = self.cstr(path);
        self.syscall(Sysno::mkdirat, &[AT_FDCWD as u64, p, mode])
    }

    pub fn unlinkat(&mut self, dirfd: i32, path: &str, flags: u64) -> i64 {
        let p = self.cstr(path);
        self.syscall(Sysno::unlinkat, &[dirfd as u64, p, flags])
    }

    pub fn rename(&mut self, from: &str, to: &str) -> i64 {
        let (a, b) = (self.cstr(from), self.cstr(to));
        self.syscall(Sysno::renameat, &[AT_FDCWD as u64, a, AT_FDCWD as u64, b])
    }

    pub fn symlink(&mut self, target: &str, path: &str) -> i64 {
        let (t, p) = (self.cstr(target), self.cstr(path));
        self.syscall(Sysno::symlinkat, &[t, AT_FDCWD as u64, p])
    }

    /// `readlinkat(AT_FDCWD, path, buf, 256)`: the result and the bytes it wrote.
    pub fn readlink(&mut self, path: &str) -> (i64, Vec<u8>) {
        let p = self.cstr(path);
        let buf = self.alloc(256);
        let r = self.syscall(Sysno::readlinkat, &[AT_FDCWD as u64, p, buf, 256]);
        (r, self.bytes(buf, r.max(0)))
    }

    pub fn chdir(&mut self, path: &str) -> i64 {
        let p = self.cstr(path);
        self.syscall(Sysno::chdir, &[p])
    }

    /// `getcwd(buf, 4096)`: the result and the path (without its NUL) on success.
    pub fn getcwd(&mut self) -> (i64, Vec<u8>) {
        let buf = self.alloc(4096);
        let r = self.syscall(Sysno::getcwd, &[buf, 4096]);
        let mut path = if r > 0 {
            self.bytes(buf, r)
        } else {
            Vec::new()
        };
        if path.last() == Some(&0) {
            path.pop();
        }
        (r, path)
    }

    /// Every entry of directory `fd` as `(name, d_type)`, in the order `getdents64` gives them.
    pub fn list_dir(&mut self, fd: i64) -> Vec<(String, u8)> {
        let buf = self.alloc(8192);
        let mut out = Vec::new();
        loop {
            let n = self.syscall(Sysno::getdents64, &[fd as u64, buf, 8192]);
            assert!(n >= 0, "getdents64({fd}) failed: {n}");
            if n == 0 {
                return out;
            }
            out.extend(parse_dirents(&self.bytes(buf, n)));
        }
    }

    /// `pipe2(fds, flags)`: the result and the two ends.
    pub fn pipe2(&mut self, flags: u64) -> (i64, i64, i64) {
        let fds = self.alloc(8);
        let r = self.syscall(Sysno::pipe2, &[fds, flags]);
        let (a, b) = (self.i32_at(fds), self.i32_at(fds + 4));
        (r, i64::from(a), i64::from(b))
    }

    pub fn mmap(&mut self, addr: u64, len: u64, prot: u64, flags: u64, fd: i64, off: u64) -> i64 {
        self.syscall(Sysno::mmap, &[addr, len, prot, flags, fd as u64, off])
    }

    /// `clock_gettime(clock)` in nanoseconds (panics on an error).
    pub fn clock_ns(&mut self, clock: u64) -> u64 {
        let ts = self.alloc(16);
        assert_eq!(self.syscall(Sysno::clock_gettime, &[clock, ts]), 0);
        self.u64_at(ts) * 1_000_000_000 + self.u64_at(ts + 8)
    }

    /// `nanosleep` for `ns` nanoseconds.
    pub fn nanosleep(&mut self, ns: u64) -> i64 {
        let ts = self.put(&timespec(ns));
        self.syscall(Sysno::nanosleep, &[ts, 0])
    }
}

/// A `struct timespec` for `ns` nanoseconds.
pub fn timespec(ns: u64) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&(ns / 1_000_000_000).to_le_bytes());
    b[8..].copy_from_slice(&(ns % 1_000_000_000).to_le_bytes());
    b
}

/// The fields of AArch64's `struct stat` (128 bytes) that tests look at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stat {
    pub dev: u64,
    pub ino: u64,
    pub mode: u32,
    pub nlink: u32,
    pub size: i64,
}

impl Stat {
    pub fn parse(b: &[u8]) -> Self {
        let u64_at = |at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
        let u32_at = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        Self {
            dev: u64_at(0),
            ino: u64_at(8),
            mode: u32_at(16),
            nlink: u32_at(20),
            size: u64_at(48) as i64,
        }
    }

    /// The file type bits (`S_IFMT`).
    pub fn kind(&self) -> u32 {
        self.mode & wasmshim_linux_abi::consts::mode::S_IFMT
    }
}

/// `(name, d_type)` of each `struct linux_dirent64` in `buf`.
pub fn parse_dirents(buf: &[u8]) -> Vec<(String, u8)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 19 <= buf.len() {
        let reclen = u16::from_le_bytes([buf[at + 16], buf[at + 17]]) as usize;
        assert!(reclen >= 19, "a dirent record of length {reclen}");
        let dtype = buf[at + 18];
        let name = &buf[at + 19..at + reclen];
        let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
        out.push((String::from_utf8_lossy(&name[..end]).into_owned(), dtype));
        at += reclen;
    }
    out
}

/// Error numbers as `i64`, so a failed call reads like C: `assert_eq!(g.close(99), -EBADF)`.
macro_rules! errnos {
    ($($name:ident),* $(,)?) => {
        $(pub const $name: i64 = Errno::$name.0 as i64;)*
    };
}

errnos!(
    EPERM,
    E2BIG,
    ENOENT,
    ESRCH,
    EINTR,
    EBADF,
    ECHILD,
    EAGAIN,
    ENOMEM,
    EACCES,
    EFAULT,
    EBUSY,
    EEXIST,
    EXDEV,
    ENOTDIR,
    EISDIR,
    EINVAL,
    EMFILE,
    ENOTTY,
    EFBIG,
    ENOSPC,
    ESPIPE,
    EPIPE,
    ENAMETOOLONG,
    ENOSYS,
    ENOTEMPTY,
    ELOOP,
    EMSGSIZE,
    ETIMEDOUT,
    EOVERFLOW,
    ERANGE,
    ENOEXEC,
    ENOPROTOOPT,
    EOPNOTSUPP,
);
