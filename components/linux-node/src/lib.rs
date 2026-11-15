//! `wasmshim-linux-node`: the Linux personality with the A64 interpreter, as a WASI 0.3 command.
//!
//! The pure parts (option parsing, the chunk cache in front of `linux:host/blob`, the
//! interpreter frontend) live here and are unit-tested in wasmtime; the WASI glue is in
//! `component.rs`.

#![deny(unsafe_code)]

#[cfg(not(test))]
mod component;
#[cfg(all(not(test), feature = "jit"))]
mod jit;
#[cfg(all(not(test), feature = "net"))]
pub mod net;

#[cfg(not(test))]
pub use component::{Loop, NodeExt, Plain, Restorer, SHIFT, blob_read, blob_size, device_now, run};

use std::collections::BTreeMap;

use wasmshim_linux::{Blob, BlobError, Config, Frontend};

/// The run-time tier's default threshold (`--jit-threshold`).
pub const DEFAULT_JIT_THRESHOLD: u32 = 128;

/// Default guest environment.
pub const DEFAULT_ENV: &[&str] = &[
    "PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin",
    "HOME=/root",
    "TERM=dumb",
    "LANG=C",
];

/// Parsed command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    pub config: ConfigOpts,
    /// Name of the root filesystem blob.
    pub image: String,
    /// Print retired instructions, virtual time and MIPS to stderr at exit.
    pub stats: bool,
    /// The execution engine (`--engine`).
    pub engine: EngineChoice,
    /// `--jit-threshold N`: interpreted executions of a block before the run-time tier
    /// translates its page (`0`: never). Only a build with the `jit` feature has the tier;
    /// others ignore it.
    pub jit_threshold: u32,
    /// `--console`: stdio is a terminal of this size (columns, rows).
    pub console: Option<(u16, u16)>,
    /// `--console-hints`: the console signals a waiting reader in band.
    pub console_hints: bool,
    pub env: Vec<String>,
    /// The guest program and its arguments (`argv[0]` is the path to execute).
    pub argv: Vec<String>,
}

/// The [`Config`] fields settable from the command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigOpts {
    pub trace: bool,
    pub hostname: String,
    pub ns_per_insn: u64,
    pub quantum: u64,
    pub host_loopback: bool,
    /// `--iface`: interfaces besides `lo` that netlink reports.
    pub interfaces: Vec<wasmshim_linux::NetInterface>,
    /// `--kernel-release`, `--kernel-version`, `--kernel-builder`: what `uname` and
    /// `/proc/version` report.
    pub kernel_release: String,
    pub kernel_version: String,
    pub kernel_builder: String,
}

impl ConfigOpts {
    /// The personality configuration.
    pub fn to_config(&self) -> Config {
        Config {
            hostname: self.hostname.clone(),
            quantum: self.quantum,
            ns_per_insn: self.ns_per_insn,
            trace: self.trace,
            host_loopback: self.host_loopback,
            interfaces: self.interfaces.clone(),
            kernel_release: self.kernel_release.clone(),
            kernel_version: self.kernel_version.clone(),
            kernel_builder: self.kernel_builder.clone(),
            ..Config::default()
        }
    }
}

/// The console size when `--console` names none: columns, rows.
pub const DEFAULT_CONSOLE: (u16, u16) = (80, 24);

/// `COLSxROWS` (terminal geometry order, like `80x24`).
pub fn parse_size(s: &str) -> Result<(u16, u16), String> {
    let (c, r) = s
        .split_once('x')
        .ok_or(format!("--console-size: expected COLSxROWS, got {s}"))?;
    let num = |v: &str| {
        v.parse::<u16>()
            .ok()
            .filter(|&n| n > 0)
            .ok_or(format!("--console-size: bad number {v}"))
    };
    Ok((num(c)?, num(r)?))
}

/// Parses `args` (without the component's own `argv[0]`).
pub fn parse_args(args: &[String]) -> Result<Options, String> {
    let d = Config::default();
    let mut o = Options {
        config: ConfigOpts {
            trace: false,
            hostname: d.hostname,
            ns_per_insn: d.ns_per_insn,
            quantum: d.quantum,
            host_loopback: d.host_loopback,
            interfaces: Vec::new(),
            kernel_release: d.kernel_release,
            kernel_version: d.kernel_version,
            kernel_builder: d.kernel_builder,
        },
        image: "rootfs".into(),
        stats: false,
        engine: EngineChoice::Auto,
        jit_threshold: DEFAULT_JIT_THRESHOLD,
        console: None,
        console_hints: false,
        env: Vec::new(),
        argv: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--trace" => o.config.trace = true,
            "--stats" => o.stats = true,
            "--console" => {
                o.console.get_or_insert(DEFAULT_CONSOLE);
            }
            "--console-hints" => o.console_hints = true,
            "--console-size" => {
                o.console = Some(parse_size(&value("--console-size")?)?);
            }
            "--host-loopback" => o.config.host_loopback = true,
            "--hostname" => o.config.hostname = value("--hostname")?,
            "--kernel-release" => o.config.kernel_release = value("--kernel-release")?,
            "--kernel-version" => o.config.kernel_version = value("--kernel-version")?,
            "--kernel-builder" => o.config.kernel_builder = value("--kernel-builder")?,
            "--iface" => o
                .config
                .interfaces
                .push(wasmshim_linux::NetInterface::parse(&value("--iface")?)?),
            "--image" => o.image = value("--image")?,
            "--engine" => {
                o.engine = match value("--engine")?.as_str() {
                    "auto" => EngineChoice::Auto,
                    "interp" => EngineChoice::Interp,
                    "dbt" => EngineChoice::Dbt,
                    "profile" => EngineChoice::Profile,
                    e => {
                        return Err(format!(
                            "--engine: unknown engine {e} (auto, interp, dbt, profile)"
                        ));
                    }
                }
            }
            "--env" => o.env.push(value("--env")?),
            "--jit-threshold" => {
                o.jit_threshold = value("--jit-threshold")?
                    .parse()
                    .map_err(|e| format!("--jit-threshold: {e}"))?
            }
            "--ns-per-insn" => {
                o.config.ns_per_insn = value("--ns-per-insn")?
                    .parse()
                    .map_err(|e| format!("--ns-per-insn: {e}"))?
            }
            "--quantum" => {
                o.config.quantum = value("--quantum")?
                    .parse()
                    .map_err(|e| format!("--quantum: {e}"))?;
                if o.config.quantum == 0 {
                    return Err("--quantum must be positive".into());
                }
            }
            "--" => {
                o.argv.extend(it.by_ref().cloned());
                break;
            }
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            _ => {
                o.argv.push(a.clone());
                o.argv.extend(it.by_ref().cloned());
                break;
            }
        }
    }
    if o.argv.is_empty() {
        return Err("usage: linux-node [options] [--] PROGRAM [ARGS...]".into());
    }
    if o.env.is_empty() {
        o.env = DEFAULT_ENV.iter().map(|s| (*s).to_owned()).collect();
    }
    Ok(o)
}

/// Which execution engine runs guest code. Every engine retires exactly the same
/// instructions, so the choice changes only wall-clock time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineChoice {
    /// `dbt` when translated code is linked into this build, else `interp`.
    Auto,
    /// The `wasmshim-aarch64` interpreter alone.
    Interp,
    /// The two-tier engine: ahead-of-time translated code where it has some, the interpreter
    /// elsewhere.
    Dbt,
    /// The interpreter, counting block entries: hot code pages are recorded for ahead-of-time
    /// translation and read out through inspection (`engine:profile`).
    Profile,
}

thread_local! {
    /// The profiler of an `--engine profile` run.
    static PROFILER: std::cell::RefCell<Option<wasmshim_aarch64_engine::profile::Profiler>> =
        const { std::cell::RefCell::new(None) };
}

/// The profile so far of an `--engine profile` run (`None` with another engine).
pub fn profile() -> Option<wasmshim_aarch64_engine::profile::Profile> {
    PROFILER.with(|p| p.borrow().as_ref().map(|p| p.profile()))
}

/// The frontend for `choice`, and the engine counters when it is the two-tier engine.
pub fn frontend(
    choice: EngineChoice,
) -> (
    Box<dyn Frontend>,
    Option<wasmshim_aarch64_engine::local::SharedTier>,
) {
    let dbt = match choice {
        EngineChoice::Auto => {
            wasmshim_aarch64_engine::local::has_translations()
                || wasmshim_aarch64_engine::jit::stats().is_some()
        }
        EngineChoice::Interp => false,
        EngineChoice::Dbt => true,
        EngineChoice::Profile => {
            let p = wasmshim_aarch64_engine::profile::Profiler::new(
                wasmshim_aarch64_engine::profile::DEFAULT_THRESHOLD,
            );
            PROFILER.with(|slot| *slot.borrow_mut() = Some(p.clone()));
            return (Box::new(ProfFrontend(p)), None);
        }
    };
    if dbt {
        let tier = wasmshim_aarch64_engine::local::SharedTier::default();
        (Box::new(DbtFrontend { tier: tier.clone() }), Some(tier))
    } else {
        (Box::new(InterpFrontend), None)
    }
}

/// The two-tier engine in-process (`wasmshim-aarch64-engine`).
pub struct DbtFrontend {
    tier: wasmshim_aarch64_engine::local::SharedTier,
}

impl Frontend for DbtFrontend {
    fn new_cpu(&self) -> Box<dyn wasmshim_cpu::GuestCpu> {
        Box::new(wasmshim_aarch64_engine::local::DbtCpu::new(
            self.tier.clone(),
        ))
    }
    fn new_memory(&self) -> Box<dyn wasmshim_cpu::GuestMemory> {
        wasmshim_aarch64_engine::local::new_memory()
    }
    /// `stat:engine`: the engine's counters, read as they are (no engine change).
    fn counters(&self) -> Vec<(&'static str, u64)> {
        let s = self.tier.borrow().stats;
        #[allow(unused_mut)]
        let mut v = vec![
            // Instructions retired inside translated regions.
            ("insns-translated", s.insns_jit),
            // Instructions the interpreter ran: untranslated blocks, the one instruction of a
            // slow-path memory access (EXIT_MEM), the tail of a budget.
            ("insns-interpreted", s.insns_interp),
            // Region entries from the dispatcher (transitions into translated code).
            ("region-calls", s.jit_calls),
            // Slow-path steps (`jit_step` from translated code, and EXIT_MEM).
            ("steps", s.steps),
            ("interpreted-blocks", s.interp_blocks),
            ("budget-exits", s.budget_exits),
            // Returns to the dispatcher to continue elsewhere (`EXIT_JUMP`: translated code's
            // chain-table lookup failed), by the target's chain entry: none (never dispatched,
            // untranslated, or its slot taken), from an older code epoch, or current.
            ("exit-jumps", s.exit_jumps),
            ("exit-jumps-absent", s.jumps_absent),
            // Of those, the ones whose target has a translation (a chain slot holding another
            // PC, or a first dispatch); the rest run interpreted.
            ("exit-jumps-absent-translated", s.jumps_absent_translated),
            ("exit-jumps-stale", s.jumps_stale),
            ("exit-jumps-current", s.jumps_current),
            // Code-epoch moves seen by the dispatcher (each makes every chain entry stale).
            ("epoch-moves", s.epoch_moves),
            // At an instruction-exact stop: 1 if the thread was inside a translated region.
            ("last-cut-translated", s.last_cut_translated),
        ];
        #[cfg(feature = "step-stats")]
        v.extend(step_stats());
        v
    }
}

/// Feature `step-stats` (dev diagnostics): the slow-path steps by class, and the
/// 24 most frequent stepped shapes as `shape:<shape>` counters.
#[cfg(feature = "step-stats")]
fn step_stats() -> Vec<(&'static str, u64)> {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    // Counter names must be 'static: each shape's name is leaked once.
    static NAMES: Mutex<BTreeMap<String, &'static str>> = Mutex::new(BTreeMap::new());
    let (classes, v) = wasmshim_aarch64_engine::dispatch::step_stats();
    let mut out = vec![
        ("steps-memory", classes[0]),
        ("steps-simd-fp", classes[1]),
        ("steps-simd-memory", classes[2]),
        ("steps-atomic", classes[3]),
        ("steps-system", classes[4]),
        ("steps-other", classes[5]),
    ];
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    // The interpreted blocks (`iblock:<pc> n<executions> key:<page key>`), the 16 top ones.
    let (blocks, keys) = wasmshim_aarch64_engine::dispatch::interp_blocks();
    for (pc, insns, n) in blocks.into_iter().take(16) {
        let key: String = keys
            .get(&(pc >> 12))
            .map(|k| k.iter().map(|b| format!("{b:02x}")).collect())
            .unwrap_or_default();
        let s = format!("iblock:{pc:x} n{n} key:{key}");
        let name = *names
            .entry(s.clone())
            .or_insert_with(|| Box::leak(s.into_boxed_str()));
        out.push((name, insns));
    }
    // Where the interpreter ran (`ipage:<page> v<versions> b<blocks>`), the 16 top pages.
    for (vpn, insns, blocks, versions, translated) in
        wasmshim_aarch64_engine::dispatch::interp_pages()
            .into_iter()
            .take(16)
    {
        let s = format!("ipage:{:x} v{versions} b{blocks} t{translated}", vpn << 12);
        let name = *names
            .entry(s.clone())
            .or_insert_with(|| Box::leak(s.into_boxed_str()));
        out.push((name, insns));
    }
    for (s, n) in v.into_iter().take(24) {
        let name = *names
            .entry(s.clone())
            .or_insert_with(|| Box::leak(format!("shape:{s}").into_boxed_str()));
        out.push((name, n));
    }
    out
}

/// The profiling interpreter on engine address spaces (`--engine profile`).
pub struct ProfFrontend(wasmshim_aarch64_engine::profile::Profiler);

impl Frontend for ProfFrontend {
    fn new_cpu(&self) -> Box<dyn wasmshim_cpu::GuestCpu> {
        Box::new(self.0.cpu())
    }
    fn new_memory(&self) -> Box<dyn wasmshim_cpu::GuestMemory> {
        wasmshim_aarch64_engine::local::new_memory()
    }
}

/// The interpreter frontend: `wasmshim-aarch64` on a `PagedMemory`.
pub struct InterpFrontend;

impl Frontend for InterpFrontend {
    fn new_cpu(&self) -> Box<dyn wasmshim_cpu::GuestCpu> {
        Box::new(wasmshim_aarch64::Cpu::new())
    }
    fn new_memory(&self) -> Box<dyn wasmshim_cpu::GuestMemory> {
        Box::new(wasmshim_cpu::PagedMemory::new())
    }
}

/// Bytes per cached chunk: one host crossing serves many guest reads. Small enough that the
/// pages a program touches here and there (file mappings are filled page by page)
/// do not drag in much more of the file than they use.
pub const CHUNK: u64 = 32 << 10;
/// Chunks kept (an 8 MiB window).
const MAX_CHUNKS: usize = 256;
/// Reads at least this long go to the host in one call, past the cache.
const DIRECT: usize = 4 * CHUNK as usize;

/// A read-only blob behind a chunk cache. `fetch(offset, len)` is the (expensive) host read.
pub struct CachedBlob<F> {
    len: u64,
    fetch: F,
    /// Chunks by index, with the tick of their last use.
    chunks: BTreeMap<u64, (Vec<u8>, u64)>,
    tick: u64,
}

impl<F: FnMut(u64, u32) -> Result<Vec<u8>, BlobError>> CachedBlob<F> {
    pub fn new(len: u64, fetch: F) -> Self {
        Self {
            len,
            fetch,
            chunks: BTreeMap::new(),
            tick: 0,
        }
    }

    fn chunk(&mut self, index: u64) -> Result<&[u8], BlobError> {
        self.tick += 1;
        if !self.chunks.contains_key(&index) {
            let start = index * CHUNK;
            let len = CHUNK.min(self.len - start) as u32;
            let data = (self.fetch)(start, len)?;
            if data.len() != len as usize {
                return Err(BlobError::Host);
            }
            if self.chunks.len() >= MAX_CHUNKS {
                // Evict the least recently used chunk (a scan, only when full).
                if let Some(old) = self.chunks.iter().min_by_key(|(_, c)| c.1).map(|(&i, _)| i) {
                    self.chunks.remove(&old);
                }
            }
            self.chunks.insert(index, (data, self.tick));
        }
        let tick = self.tick;
        let entry = self.chunks.get_mut(&index).ok_or(BlobError::Host)?;
        entry.1 = tick;
        Ok(&entry.0)
    }
}

impl<F: FnMut(u64, u32) -> Result<Vec<u8>, BlobError>> Blob for CachedBlob<F> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), BlobError> {
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(BlobError::OutOfRange)?;
        if end > self.len {
            return Err(BlobError::OutOfRange);
        }
        if buf.len() >= DIRECT {
            let len = u32::try_from(buf.len()).map_err(|_| BlobError::OutOfRange)?;
            let data = (self.fetch)(offset, len)?;
            if data.len() != buf.len() {
                return Err(BlobError::Host);
            }
            buf.copy_from_slice(&data);
            return Ok(());
        }
        let mut done = 0usize;
        while done < buf.len() {
            let at = offset + done as u64;
            let (index, within) = (at / CHUNK, (at % CHUNK) as usize);
            let chunk = self.chunk(index)?;
            let n = (chunk.len() - within).min(buf.len() - done);
            buf[done..done + n].copy_from_slice(&chunk[within..within + n]);
            done += n;
        }
        Ok(())
    }
}

/// A level-triggered broadcast notifier for async tasks inside one instance (the network
/// pumps and the run loop), following these driver rules:
///
/// * level-triggered: a waiter that saw generation `g` completes as soon as the generation is
///   not `g`, however many notifies happened and whoever else waited;
/// * the waker is registered again at every poll (never a stale waker);
/// * a waiter that is dropped before completing removes itself (a cancelled wait never leaves
///   a waker behind, which could wake a finished task).
#[derive(Default)]
pub struct Notify {
    generation: std::cell::Cell<u64>,
    next_key: std::cell::Cell<u64>,
    waiters: std::cell::RefCell<BTreeMap<u64, std::task::Waker>>,
}

impl Notify {
    /// The current generation.
    pub fn generation(&self) -> u64 {
        self.generation.get()
    }

    /// Bumps the generation and wakes every waiter.
    pub fn notify(&self) {
        self.generation.set(self.generation.get().wrapping_add(1));
        let waiters = std::mem::take(&mut *self.waiters.borrow_mut());
        for w in waiters.into_values() {
            w.wake();
        }
    }

    /// Completes once the generation differs from `seen`.
    pub fn changed(&self, seen: u64) -> Changed<'_> {
        Changed {
            notify: self,
            seen,
            key: None,
        }
    }

    /// Waiters currently registered (for tests and diagnostics).
    pub fn waiters(&self) -> usize {
        self.waiters.borrow().len()
    }
}

/// The future of [`Notify::changed`].
pub struct Changed<'a> {
    notify: &'a Notify,
    seen: u64,
    key: Option<u64>,
}

impl std::future::Future for Changed<'_> {
    type Output = ();

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        let n = self.notify;
        if n.generation.get() != self.seen {
            if let Some(k) = self.key.take() {
                n.waiters.borrow_mut().remove(&k);
            }
            return std::task::Poll::Ready(());
        }
        let key = match self.key {
            Some(k) => k,
            None => {
                let k = n.next_key.get();
                n.next_key.set(k.wrapping_add(1));
                self.key = Some(k);
                k
            }
        };
        n.waiters.borrow_mut().insert(key, cx.waker().clone());
        std::task::Poll::Pending
    }
}

impl Drop for Changed<'_> {
    fn drop(&mut self) {
        if let Some(k) = self.key.take() {
            self.notify.waiters.borrow_mut().remove(&k);
        }
    }
}

/// Deadlines with a timer in flight. Timers are never cancelled: a new
/// timer is armed only when none is due at or before the requested instant; a timer that
/// fires for a deadline nobody waits for any more causes one harmless spurious wake-up.
#[derive(Default)]
pub struct Timers {
    armed: std::cell::RefCell<std::collections::BTreeSet<u64>>,
}

impl Timers {
    /// Whether a timer must be armed for `deadline` (given the current time `now`); if so,
    /// records it as armed.
    pub fn arm(&self, now: u64, deadline: u64) -> bool {
        let mut armed = self.armed.borrow_mut();
        if armed.range(now..=deadline).next().is_some() {
            return false;
        }
        armed.insert(deadline);
        true
    }

    /// The timer for `deadline` fired.
    pub fn fired(&self, deadline: u64) {
        self.armed.borrow_mut().remove(&deadline);
    }

    /// Timers in flight.
    pub fn len(&self) -> usize {
        self.armed.borrow().len()
    }

    /// No timer in flight.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poll_once<F: std::future::Future>(f: std::pin::Pin<&mut F>) -> std::task::Poll<F::Output> {
        f.poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    }

    #[test]
    fn notify_is_level_triggered_and_cancel_safe() {
        let n = Notify::default();
        let g = n.generation();
        let mut a = Box::pin(n.changed(g));
        assert!(poll_once(a.as_mut()).is_pending());
        assert!(
            poll_once(a.as_mut()).is_pending(),
            "re-registers, still one waiter"
        );
        assert_eq!(n.waiters(), 1);
        {
            let mut b = Box::pin(n.changed(g));
            assert!(poll_once(b.as_mut()).is_pending());
            assert_eq!(n.waiters(), 2);
        }
        assert_eq!(n.waiters(), 1, "a dropped waiter removes itself");
        n.notify();
        n.notify();
        assert!(poll_once(a.as_mut()).is_ready());
        // A waiter created after the change completes at once (level, not edge).
        assert!(poll_once(Box::pin(n.changed(g)).as_mut()).is_ready());
        assert_eq!(n.waiters(), 0);
    }

    #[test]
    fn timers_are_armed_once_and_never_cancelled() {
        let t = Timers::default();
        assert!(t.arm(0, 100));
        assert!(!t.arm(0, 100), "already armed");
        assert!(!t.arm(0, 150), "an earlier timer wakes us first");
        assert!(t.arm(0, 50), "an earlier deadline needs its own timer");
        t.fired(50);
        assert!(!t.arm(60, 120));
        t.fired(100);
        assert!(t.arm(110, 120));
        assert_eq!(t.len(), 1);
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn options() {
        let o = parse_args(&s(&["--trace", "--env", "A=1", "/bin/sh", "-c", "--trace"])).unwrap();
        assert!(o.config.trace);
        assert_eq!(o.env, ["A=1"]);
        assert_eq!(o.argv, ["/bin/sh", "-c", "--trace"]);
        let o = parse_args(&s(&["--", "--weird"])).unwrap();
        assert_eq!(o.argv, ["--weird"]);
        assert!(o.env.iter().any(|e| e.starts_with("PATH=")));
        assert!(parse_args(&s(&["--bogus", "x"])).is_err());
        assert!(parse_args(&s(&[])).is_err());
        assert_eq!(parse_args(&s(&["/bin/sh"])).unwrap().console, None);
        let o = parse_args(&s(&["--console", "/bin/sh"])).unwrap();
        assert_eq!(o.console, Some((80, 24)));
        let o = parse_args(&s(&["--console-size", "132x43", "/bin/sh"])).unwrap();
        assert_eq!(o.console, Some((132, 43)));
        assert!(parse_args(&s(&["--console-size", "80", "/bin/sh"])).is_err());
        let o = parse_args(&s(&["--engine", "profile", "/bin/sh"])).unwrap();
        assert_eq!(o.engine, EngineChoice::Profile);
    }

    #[test]
    fn cache_serves_reads_across_chunks() {
        let data: Vec<u8> = (0..(3 * CHUNK + 10)).map(|i| (i % 251) as u8).collect();
        let mut fetches = 0;
        let mut blob = CachedBlob::new(data.len() as u64, |off, len| {
            fetches += 1;
            Ok(data[off as usize..off as usize + len as usize].to_vec())
        });
        let mut buf = vec![0u8; 100];
        blob.read_at(CHUNK - 50, &mut buf).unwrap();
        assert_eq!(
            &buf[..],
            &data[(CHUNK - 50) as usize..(CHUNK + 50) as usize]
        );
        blob.read_at(CHUNK + 5, &mut buf).unwrap();
        assert!(blob.read_at(3 * CHUNK, &mut buf).is_err());
        drop(blob);
        assert_eq!(fetches, 2);
    }
}
