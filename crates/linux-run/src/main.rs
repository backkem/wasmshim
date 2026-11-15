//! `wasmshim-run`: runs a `wasmshim-linux-node` component deterministically under wasmtime.
//!
//! ```text
//! wasmshim-run [--image TAR] [--seed N] [--no-cache] [--quiet] [--fuel]
//!           [--sample US [--sample-out FILE]] NODE.wasm [NODE ARGS...]
//! ```
//!
//! * **Virtual time:** `wasi:clocks` (p2 and p3) read a virtual clock that starts at 0 (wall
//!   clock: 2026-01-01T00:00:00Z plus virtual time) and only moves when every task waits: the
//!   p3 `wait-until`/`wait-for` the node uses between quanta jump to their deadline. So a run
//!   is reproducible, and guest-visible timing is a function of retired instructions only.
//! * **Randomness:** every `wasi:random` stream is a seeded generator (`--seed`, default 1).
//! * **Images:** with `--image`, `linux:host/blob` serves the tar as blob `rootfs` from the
//!   file (positional reads), so an uncomposed `wasmshim_linux_node.wasm` runs with no image
//!   compiled in.
//! * **Compile cache:** compiled components are kept in the user's cache directory
//!   ([`cache_dir`]; `$LINUX_RUN_CACHE` moves it) keyed by the component bytes and the engine
//!   configuration (`--no-cache` disables).
//! * **Standard streams:** the guest's stdout and stderr are written through to this
//!   process's as they are produced. Its stdin is this process's, handed over one line at a
//!   time and only when the node can do nothing else, so piped input replays exactly
//!   ([`stdio`]).
//! * **Profiling**: `--sample US` samples the Wasm stack every `US` µs and prints
//!   the innermost functions and the share of time per component; `--sample-out FILE` also
//!   writes the folded stacks (flamegraph format). With `$LINUX_RUN_PERF_CTL=FIFO`, `perf stat
//!   -D -1 --control fifo:FIFO` counts only instantiation and run.
//!
//! A timing line goes to stderr when the run ends:
//! `wasmshim-run: load=… (cache hit|compiled) instantiate=… run=… virtual=…ns exit=…`.

#![deny(unsafe_code)]

mod stdio;

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::convert::Infallible;
use std::future::Future;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::pin::{Pin, pin};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use wasmshim_wasmtime::blob::FileBlobs;
use wasmtime::component::{Accessor, Component, HasData, Linker, ResourceTable};
use wasmtime::{Config, Engine, Result, Store, bail, format_err};
use wasmtime_wasi::clocks::{HostMonotonicClock, HostWallClock};
use wasmtime_wasi::p3::bindings::clocks::monotonic_clock as p3_mono;
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

use stdio::{HostInput, HostOutput};

/// Wall-clock reading at virtual time 0: 2026-01-01T00:00:00Z (a fixed, deterministic
/// host clock).
const WALL_EPOCH_NS: u64 = 1_767_225_600 * 1_000_000_000;

// ---- virtual time ----

/// Pending wake-ups: deadlines (with a sequence number), their wakers, the next sequence.
type Timers = (BinaryHeap<Reverse<(u64, u64)>>, BTreeMap<u64, Waker>, u64);

#[derive(Default)]
struct VirtualClock {
    now: AtomicU64,
    timers: Mutex<Timers>,
}

impl VirtualClock {
    fn now(&self) -> u64 {
        self.now.load(Ordering::SeqCst)
    }

    /// Registers a wake-up at `at`; returns its id.
    fn arm(&self, at: u64, waker: Waker) -> u64 {
        let mut t = self.timers.lock().expect("clock");
        let seq = t.2;
        t.2 += 1;
        t.0.push(Reverse((at, seq)));
        t.1.insert(seq, waker);
        seq
    }

    /// Replaces the waker of a pending wake-up (a future must wake its latest waker).
    fn rearm(&self, seq: u64, waker: &Waker) {
        let mut t = self.timers.lock().expect("clock");
        if let Some(w) = t.1.get_mut(&seq) {
            w.clone_from(waker);
        }
    }

    /// Advances to the earliest timer and wakes it; `false` if there is none.
    fn fire_next(&self) -> bool {
        let mut t = self.timers.lock().expect("clock");
        let Some(Reverse((at, seq))) = t.0.pop() else {
            return false;
        };
        self.now.fetch_max(at, Ordering::SeqCst);
        if let Some(w) = t.1.remove(&seq) {
            drop(t);
            w.wake();
        }
        true
    }
}

struct Sleep {
    clock: Arc<VirtualClock>,
    until: u64,
    armed: Option<u64>,
}

impl Future for Sleep {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.clock.now() >= self.until {
            return Poll::Ready(());
        }
        match self.armed {
            Some(seq) => self.clock.rearm(seq, cx.waker()),
            None => {
                let seq = self.clock.arm(self.until, cx.waker().clone());
                self.armed = Some(seq);
            }
        }
        Poll::Pending
    }
}

struct Woken(AtomicBool);

impl Wake for Woken {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Drives `fut`; when it is idle, fires the next virtual timer; with no timer left, delivers
/// the input a guest read waits for. `Err` on a deadlock.
fn block_on<F: Future>(clock: &VirtualClock, stdin: &HostInput, fut: F) -> Result<F::Output> {
    let flag = Arc::new(Woken(AtomicBool::new(false)));
    let waker = Waker::from(flag.clone());
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(fut);
    loop {
        flag.0.store(false, Ordering::SeqCst);
        if let Poll::Ready(out) = fut.as_mut().poll(&mut cx) {
            return Ok(out);
        }
        if flag.0.load(Ordering::SeqCst) {
            continue;
        }
        if clock.fire_next() {
            continue;
        }
        if stdin.wanted() {
            // Nothing else can move: wait for the next line of input (or its end).
            stdin.deliver();
        } else {
            bail!(
                "deadlock at virtual {} ns: the node waits with no timer",
                clock.now()
            );
        }
    }
}

struct MonoClock(Arc<VirtualClock>);

impl HostMonotonicClock for MonoClock {
    fn resolution(&self) -> u64 {
        1
    }
    fn now(&self) -> u64 {
        self.0.now()
    }
}

struct WallClock(Arc<VirtualClock>);

impl HostWallClock for WallClock {
    fn resolution(&self) -> Duration {
        Duration::from_nanos(1)
    }
    fn now(&self) -> Duration {
        Duration::from_nanos(WALL_EPOCH_NS + self.0.now())
    }
}

/// splitmix64 per stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

impl rand_core::TryRng for Rng {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok((self.next() >> 32) as u32)
    }
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.next())
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        for chunk in dst.chunks_mut(8) {
            let b = self.next().to_le_bytes();
            chunk.copy_from_slice(&b[..chunk.len()]);
        }
        Ok(())
    }
}

// ---- the store ----

struct State {
    ctx: WasiCtx,
    table: ResourceTable,
    clock: Arc<VirtualClock>,
    blobs: FileBlobs,
}

impl WasiView for State {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}

/// Our `wasi:clocks/monotonic-clock@0.3.0`: waits on the virtual clock.
struct Mono;

impl HasData for Mono {
    type Data<'a> = ClockView<'a>;
}

struct ClockView<'a>(&'a Arc<VirtualClock>);

impl p3_mono::Host for ClockView<'_> {
    fn now(&mut self) -> wasmtime::Result<p3_mono::Mark> {
        Ok(self.0.now())
    }
    fn get_resolution(&mut self) -> wasmtime::Result<p3_mono::Duration> {
        Ok(1)
    }
}

impl<U> p3_mono::HostWithStore<U> for Mono {
    async fn wait_until(store: &Accessor<U, Self>, when: p3_mono::Mark) -> wasmtime::Result<()> {
        let clock = store.with(|mut v| v.get().0.clone());
        Sleep {
            clock,
            until: when,
            armed: None,
        }
        .await;
        Ok(())
    }
    async fn wait_for(
        store: &Accessor<U, Self>,
        how_long: p3_mono::Duration,
    ) -> wasmtime::Result<()> {
        let clock = store.with(|mut v| v.get().0.clone());
        let until = clock.now().saturating_add(how_long);
        Sleep {
            clock,
            until,
            armed: None,
        }
        .await;
        Ok(())
    }
}

fn engine(sampling: bool, fuel: bool) -> Result<Engine> {
    let mut config = Config::new();
    config.epoch_interruption(sampling);
    // `--fuel`: fuel metering, as an embedder may run nodes (measuring its cost).
    config.consume_fuel(fuel);
    config.wasm_component_model(true);
    config.wasm_component_model_async(true);
    // Translated code marks its slow paths cold (`metadata.code.branch_hint`): code layout
    // only, never semantics. `LINUX_RUN_BRANCH_HINTS=0` ignores them (A/B measurements).
    config.wasm_branch_hinting(std::env::var_os("LINUX_RUN_BRANCH_HINTS").is_none_or(|v| v != "0"));
    Engine::new(&config)
}

/// The compile cache: `$LINUX_RUN_CACHE`, or `wasmshim/compiled` in the user's cache directory
/// (`%LOCALAPPDATA%` on Windows, `~/Library/Caches` on macOS, `$XDG_CACHE_HOME` or `~/.cache`
/// elsewhere). `None` when no such directory is known: the run then compiles without a cache.
///
/// Nothing prunes it: an entry is a few tens of megabytes per node build and engine
/// configuration, and the directory is safe to delete.
fn cache_dir() -> Option<PathBuf> {
    let var = |name: &str| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if let Some(dir) = var("LINUX_RUN_CACHE") {
        return Some(dir);
    }
    let base = if cfg!(windows) {
        var("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|home| home.join("Library/Caches"))
    } else {
        var("XDG_CACHE_HOME").or_else(|| var("HOME").map(|home| home.join(".cache")))
    };
    base.map(|base| base.join("wasmshim").join("compiled"))
}

/// `perf stat --control fifo:CTL` (with `-D -1`, counters start disabled): with
/// `$LINUX_RUN_PERF_CTL=CTL`, the counters run only from instantiation to exit, so the
/// counts exclude loading and compiling the component.
fn perf_control(command: &str) {
    if let Some(path) = std::env::var_os("LINUX_RUN_PERF_CTL")
        && let Ok(mut f) = std::fs::OpenOptions::new().write(true).open(path)
    {
        let _ = f.write_all(command.as_bytes());
    }
}

/// Loads (from the cache when possible) and reports how.
#[allow(unsafe_code)]
fn load(engine: &Engine, path: &Path, cache: bool) -> Result<(Component, &'static str)> {
    let bytes = std::fs::read(path).map_err(|e| format_err!("reading {}: {e}", path.display()))?;
    let Some(dir) = cache_dir().filter(|_| cache) else {
        return Ok((
            Component::from_binary(engine, &bytes)?,
            "compiled, no cache",
        ));
    };
    let mut h = Sha256::new();
    h.update(&bytes);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&engine.precompile_compatibility_hash(), &mut hasher);
    let key = format!(
        "{}-{:016x}.cwasm",
        h.finalize()[..16]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        std::hash::Hasher::finish(&hasher)
    );
    let artifact = dir.join(key);
    if artifact.is_file() {
        // SAFETY: the cache directory is the user's own (as private as the binaries they run),
        // and only this function writes it, always with `Component::serialize` output for an
        // engine with the same compatibility hash (in the file name); wasmtime checks the
        // artifact's header and configuration before using it, and a bad file is recompiled.
        // Whoever can write there can already run code as this user.
        if let Ok(c) = unsafe { Component::deserialize_file(engine, &artifact) } {
            return Ok((c, "cache hit"));
        }
    }
    let c = Component::from_binary(engine, &bytes)?;
    if let Ok(serialized) = c.serialize() {
        let _ = std::fs::create_dir_all(&dir);
        let tmp = artifact.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::write(&tmp, serialized).is_ok() {
            let _ = std::fs::rename(&tmp, &artifact);
        }
    }
    Ok((c, "compiled"))
}

struct Args {
    image: Option<PathBuf>,
    seed: u64,
    cache: bool,
    quiet: bool,
    /// Meter fuel, with an unlimited supply.
    fuel: bool,
    /// Sample the innermost Wasm function every this many microseconds (diagnostics).
    sample: Option<u64>,
    /// Write the sampled stacks here (folded: `outer;...;inner count`).
    sample_out: Option<PathBuf>,
    node: PathBuf,
    rest: Vec<String>,
}

fn parse() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        image: None,
        seed: 1,
        cache: true,
        quiet: false,
        fuel: false,
        sample: None,
        sample_out: None,
        node: PathBuf::new(),
        rest: Vec::new(),
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--image" => {
                a.image = Some(
                    it.next()
                        .ok_or_else(|| format_err!("--image needs a path"))?
                        .into(),
                )
            }
            "--seed" => {
                a.seed = it
                    .next()
                    .ok_or_else(|| format_err!("--seed needs a value"))?
                    .parse()
                    .map_err(|e| format_err!("--seed: {e}"))?
            }
            "--no-cache" => a.cache = false,
            "--quiet" => a.quiet = true,
            "--fuel" => a.fuel = true,
            "--sample-out" => {
                a.sample_out = Some(
                    it.next()
                        .ok_or_else(|| format_err!("--sample-out needs a path"))?
                        .into(),
                )
            }
            "--sample" => {
                a.sample = Some(
                    it.next()
                        .ok_or_else(|| format_err!("--sample needs microseconds"))?
                        .parse()
                        .map_err(|e| format_err!("--sample: {e}"))?,
                )
            }
            s if s.starts_with("--") => bail!("unknown option {s}"),
            _ => {
                a.node = arg.into();
                a.rest.extend(it.by_ref());
            }
        }
    }
    if a.node.as_os_str().is_empty() {
        bail!(
            "usage: wasmshim-run [--image TAR] [--seed N] [--no-cache] [--quiet] [--fuel] \
             [--sample US [--sample-out FILE]] NODE.wasm [ARGS...]"
        );
    }
    Ok(a)
}

fn main() -> std::process::ExitCode {
    match real_main() {
        Ok(code) => std::process::ExitCode::from(code),
        Err(e) => {
            eprintln!("wasmshim-run: {e:#}");
            std::process::ExitCode::from(125)
        }
    }
}

fn real_main() -> Result<u8> {
    let a = parse()?;
    let engine = engine(a.sample.is_some(), a.fuel)?;
    let t0 = Instant::now();
    let (component, how) = load(&engine, &a.node, a.cache)?;
    let load_time = t0.elapsed();

    let mut linker: Linker<State> = Linker::new(&engine);
    wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
    wasmtime_wasi::p3::add_to_linker(&mut linker)?;
    linker.allow_shadowing(true);
    p3_mono::add_to_linker::<_, Mono>(&mut linker, |s: &mut State| ClockView(&s.clock))?;
    linker.allow_shadowing(false);
    wasmshim_wasmtime::blob::add_to_linker(&mut linker, |s: &mut State| &mut s.blobs)?;

    let clock = Arc::new(VirtualClock::default());
    let stdin = HostInput::default();
    let rng = |stream: u64| Rng(a.seed.wrapping_mul(0x9e37_79b9).wrapping_add(stream));
    let mut b = WasiCtxBuilder::new();
    b.arg("linux-node")
        .args(&a.rest)
        .stdin(stdin.clone())
        .stdout(HostOutput::Stdout)
        .stderr(HostOutput::Stderr)
        .wall_clock(WallClock(clock.clone()))
        .monotonic_clock(MonoClock(clock.clone()))
        .secure_random(rng(1))
        .insecure_random(rng(2))
        .insecure_random_seed(u128::from(a.seed));
    let mut blobs = FileBlobs::default();
    if let Some(image) = &a.image {
        blobs
            .add("rootfs", image)
            .map_err(|e| format_err!("image {}: {e}", image.display()))?;
    }
    let state = State {
        ctx: b.build(),
        table: ResourceTable::new(),
        clock: clock.clone(),
        blobs,
    };
    let mut store = Store::new(&engine, state);
    store.set_epoch_deadline(1 << 62);
    if a.fuel {
        store.set_fuel(u64::MAX)?;
    }
    let samples = a.sample.map(|us| sample(&mut store, &engine, us));

    perf_control("enable\n");
    let t1 = Instant::now();
    let mut instantiate_time = Duration::ZERO;
    let run = block_on(&clock, &stdin, async {
        use wasmtime_wasi::p3::bindings::Command;
        let cmd = Command::instantiate_async(&mut store, &component, &linker).await?;
        instantiate_time = t1.elapsed();
        store
            .run_concurrent(async move |s| cmd.wasi_cli_run().call_run(s).await)
            .await?
    })
    .and_then(|r| r);
    let run_time = t1.elapsed().saturating_sub(instantiate_time);
    perf_control("disable\n");
    let code: Result<u8> = match run {
        Ok(Ok(())) => Ok(0),
        Ok(Err(())) => Ok(1),
        Err(e) => match e.downcast_ref::<wasmtime_wasi::I32Exit>() {
            Some(exit) => Ok(u8::try_from(exit.0).unwrap_or(255)),
            None => Err(e),
        },
    };
    if let Some((counts, stop)) = samples {
        stop.store(true, Ordering::SeqCst);
        let folded = counts.lock().map(|m| m.clone()).unwrap_or_default();
        report_samples(&folded, a.sample_out.as_deref());
    }
    if !a.quiet {
        let exit = match &code {
            Ok(c) => c.to_string(),
            Err(_) => "trap".into(),
        };
        eprintln!(
            "wasmshim-run: load={:.6}s ({how}) instantiate={:.6}s run={:.6}s virtual={}ns exit={exit}",
            load_time.as_secs_f64(),
            instantiate_time.as_secs_f64(),
            run_time.as_secs_f64(),
            clock.now(),
        );
    }
    // `{:?}` keeps the Wasm backtrace; `main` then prints the message chain and exits 125.
    code.inspect_err(|e| eprintln!("wasmshim-run: {e:?}"))
}

type Counts = Arc<Mutex<BTreeMap<String, u64>>>;

/// Wall-clock sampling of the Wasm stack (epoch interruption): diagnostics only. Counts
/// folded stacks (`outermost;...;innermost`, the flamegraph format); translated regions are
/// one frame, `<translated region>`.
fn sample(store: &mut Store<State>, engine: &Engine, us: u64) -> (Counts, Arc<AtomicBool>) {
    let counts: Counts = Arc::default();
    let stop = Arc::new(AtomicBool::new(false));
    let c = counts.clone();
    store.set_epoch_deadline(1);
    store.epoch_deadline_callback(move |ctx| {
        let bt = wasmtime::WasmBacktrace::capture(&ctx);
        let mut frames: Vec<String> = Vec::new();
        for f in bt.frames().iter().rev() {
            let raw = f.func_name().unwrap_or("?");
            let n = if raw.starts_with("linux_aot_region") {
                "<translated region>".to_owned()
            } else {
                // `{:#}`: without the hash suffix.
                format!("{:#}", rustc_demangle::demangle(raw))
            };
            if frames.last() != Some(&n) || n != "<translated region>" {
                frames.push(n);
            }
        }
        let folded = if frames.is_empty() {
            "<host>".to_owned()
        } else {
            frames.join(";")
        };
        if let Ok(mut m) = c.lock() {
            *m.entry(folded).or_default() += 1;
        }
        Ok(wasmtime::UpdateDeadline::Continue(1))
    });
    let engine = engine.clone();
    let s = stop.clone();
    std::thread::spawn(move || {
        while !s.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_micros(us));
            engine.increment_epoch();
        }
    });
    (counts, stop)
}

/// Prints the innermost functions and the component shares of `folded` stacks, and writes
/// them to `out` (one `stack count` line each) if given.
fn report_samples(folded: &BTreeMap<String, u64>, out: Option<&Path>) {
    let total: u64 = folded.values().sum();
    let pct = |n: u64| n as f64 * 100.0 / total.max(1) as f64;
    let mut inner: BTreeMap<&str, u64> = BTreeMap::new();
    let mut parts: BTreeMap<&str, u64> = BTreeMap::new();
    for (stack, n) in folded {
        let leaf = stack.rsplit(';').next().unwrap_or(stack);
        *inner.entry(leaf).or_default() += n;
        *parts.entry(component(stack)).or_default() += n;
    }
    let mut inner: Vec<_> = inner.into_iter().collect();
    inner.sort_by_key(|x| std::cmp::Reverse(x.1));
    eprintln!("wasmshim-run: {total} samples; innermost functions:");
    for (name, n) in inner.iter().take(40) {
        eprintln!("  {:>6.2}%  {name}", pct(*n));
    }
    let mut parts: Vec<_> = parts.into_iter().collect();
    parts.sort_by_key(|x| std::cmp::Reverse(x.1));
    eprintln!("wasmshim-run: share of time by component:");
    for (name, n) in &parts {
        eprintln!("  {:>6.2}%  {name}", pct(*n));
    }
    if let Some(path) = out {
        let text: String = folded.iter().map(|(s, n)| format!("{s} {n}\n")).collect();
        match std::fs::write(path, text) {
            Ok(()) => eprintln!("wasmshim-run: folded stacks in {}", path.display()),
            Err(e) => eprintln!("wasmshim-run: {}: {e}", path.display()),
        }
    }
}

/// The component a folded stack's time belongs to. A [`CONTEXTS`] rule that
/// matches any frame wins (in list order: e.g. a `PagedMemory::store` under the ELF loader is
/// image loading); otherwise the innermost frame a [`LEAVES`] rule recognises decides (a
/// `memcpy` is charged to its caller).
fn component(stack: &str) -> &'static str {
    let frames: Vec<&str> = stack.split(';').collect();
    let hit = |pats: &[&str], f: &str| pats.iter().any(|p| f.contains(p));
    if let Some((_, c)) = CONTEXTS
        .iter()
        .find(|(pats, _)| frames.iter().any(|f| hit(pats, f)))
    {
        return c;
    }
    frames
        .iter()
        .rev()
        .find_map(|f| LEAVES.iter().find(|(pats, _)| hit(pats, f)).map(|r| r.1))
        .unwrap_or("other")
}

/// Whole-stack rules, in priority order.
const CONTEXTS: &[(&[&str], &str)] = &[
    (
        &[
            "wasmshim_linux_elf::",
            "Kernel>::exec_image",
            "Vfs>::mount_image",
            "as wasmshim_linux::host::Blob>::read_at",
        ],
        "image loading (ELF, tar index, blob reads)",
    ),
    (
        &["EngineMem>::try_aot", "aot::page_key", "sha2::"],
        "translation lookup (SHA-256 key, directory, install)",
    ),
    (
        &["linux_jit_step", "linux_jit_fill"],
        "translated code: slow path (jit_step, TLB fill)",
    ),
];

/// Innermost-frame rules.
const LEAVES: &[(&[&str], &str)] = &[
    (
        &["<translated region>", "linux_aot_pack"],
        "translated code",
    ),
    (&["<host>"], "host (no Wasm frame)"),
    (
        &["wasmshim_aarch64_engine::mem"],
        "address translation (engine TLB)",
    ),
    (&["wasmshim_aarch64::"], "interpreter"),
    (&["wasmshim_cpu::paged"], "guest memory (PagedMemory)"),
    (&["wasmshim_aarch64_engine::dispatch"], "dispatcher"),
    (&["wasmshim_linux::"], "personality"),
    (
        &[
            "wit_bindgen",
            "wasip3",
            "wasmshim_linux_node",
            "wasi:cli/run",
        ],
        "node glue (async runtime, WASI)",
    ),
];
