# The node (`components/linux-node`)

`wasmshim-linux-node` packages the personality, the substrate and the A64 engines as one WASI 0.3
command component (`wasi:cli/run@0.3.0`). It reads its root filesystem through the small
`linux:host` WIT, takes time and entropy from WASI, and writes guest output to WASI stdio.

**Guarantee.** The node has no other way out: no host files, sockets or clocks beyond its
imports. Under a host with a virtual clock and seeded entropy (`wasmshim-run`), a run is
reproducible down to the retired-instruction count.

## Imports

- `wasi:clocks`, `wasi:random`, `wasi:cli` 0.3 (and the 0.2 stdio set Rust's `std` pulls in);
- `linux:host/blob` ([`host.wit`](../../wit/linux-host/host.wit)): `size(name)` and
  `read(name, offset, len)` over immutable blobs; with the `blob-words` feature also
  `linux:host/blob-words.read-words`, the same bytes as little-endian `u64`s, which cross
  between two composed components about eight times faster than a byte list;
- with `net`: `wasi:sockets@0.3` (`types`) for traffic that leaves the node;
- with `jit`: `translate:jit` ([engines-and-translation.md](engines-and-translation.md)).

## Command line

`wasmshim-linux-node [options] [--] PROGRAM [ARGS...]` ([`lib.rs`](../../components/linux-node/src/lib.rs)).
The first argument that is not an option starts the guest's `argv`.

| Option | Default | Meaning |
|---|---|---|
| `--image NAME` | `rootfs` | blob holding the root filesystem |
| `--env K=V` (repeatable) | `PATH=...`, `HOME=/root`, `TERM=dumb`, `LANG=C` | guest environment (any `--env` replaces the defaults) |
| `--trace` | off | one line per syscall on stderr, prefixed `[linux] ` |
| `--stats` | off | retired instructions, elapsed time, MIPS and blob traffic at exit |
| `--engine auto\|interp\|dbt\|profile` | `auto` | `auto` is `dbt` when translations are linked or the run-time tier is on |
| `--jit-threshold N` | 128 | interpreted executions before the run-time tier translates (0: never; `jit` builds only) |
| `--quantum N` | 100000 | instructions per scheduling slice (positive) |
| `--ns-per-insn N` | 1 | virtual nanoseconds per retired instruction |
| `--console`, `--console-size COLSxROWS` | off, 80x24 | stdio is a terminal; stdin is raw keystrokes |
| `--console-hints` | off | an in-band marker on output when a reader waits for input |
| `--hostname H` | `linux` | `uname -n` |
| `--kernel-release`, `--kernel-version`, `--kernel-builder` | `6.6.0`, `#1 SMP PREEMPT`, `linux` | what `uname` and `/proc/version` report |
| `--iface NAME,mac=..[,mtu=N][,addr=A/P]...` | none | an interface besides `lo` that netlink reports (descriptive only) |
| `--host-loopback` | off | loopback traffic with no local listener goes to the network |

The exit status is the guest's (128 + signal number for a signal death). The node itself exits
with 125 for bad options, a missing image or a guest deadlock (every thread blocked, nothing
that could wake one).

## Features

| Feature | Effect |
|---|---|
| `export` (default) | exports `wasi:cli/run` for the plain node; off when an embedder exports its own |
| `net` | the `wasi:sockets` network provider; without it, non-loopback `AF_INET`/`AF_INET6` traffic is `ENETUNREACH` |
| `blob-words` | reads the image through `blob-words`; for a node composed with its image provider |
| `jit` | the run-time translation tier over `translate:jit`; the core module exports its function table and `linux_jit_step`/`linux_jit_fill` |
| `step-stats` | diagnostics: engine slow-path steps by class and shape in the engine counters (slower) |

`build.rs` links the ahead-of-time translations from `LINUX_AOT_OBJ`, or an empty object when
the variable is unset or the crate is built as an embedder's dependency.

## The run loop

[`component.rs`](../../components/linux-node/src/component.rs), `run(ext: Rc<dyn NodeExt>)`:

```text
parse options; mount the image through the chunk cache; choose the frontend
Kernel::new(config after NodeExt::config); NodeExt::attach
net: install SocketExt, attach the WASI provider; console: attach, spawn the stdin pump
NodeExt::start_devices
paused? NodeExt::park(Restorer), wait until NodeExt::restored()   else Kernel::spawn
loop: console input -> NodeExt::before_run -> Kernel::run -> NodeExt::after_run
      Exited(status) -> stop;  Wait { until } -> wait for the deadline, a socket, input
--stats; exit with the guest's status
```

Waiting arms a WASI `wait-until` timer and never cancels it (a timer that fires late causes one
spurious wake-up). Socket pumps and the console pump are async tasks inside `run` that wake the
loop through a level-triggered notifier.

## Extension points

An embedder that needs more than the plain node depends on `wasmshim-linux-node` with
`default-features = false`, implements `NodeExt`, exports `wasi:cli/run` itself and calls
`wasmshim_linux_node::run`. Every hook defaults to the plain node's behaviour.

| `NodeExt` hook | For |
|---|---|
| `config(&mut Config)` | personality settings beyond the command line (again for a restored kernel) |
| `attach(&kernel, &opts)` | keep a handle to the kernel (a debugger, control calls) before anything runs |
| `socket_ext()` (`net`) | the provider's extras, as a `net::SocketExt` |
| `start_devices(&kernel, &loop) -> bool` | attach HID devices or framebuffers; `true` keeps the node waiting on device input when every thread blocks without a deadline |
| `before_run`, `after_run` | feed device input, take frames, between kernel runs |
| `paused()`, `park(Restorer)`, `restored()` | start without booting and wait for a restore |
| `drew_random(words)` | entropy accounting: called before each draw, 8-byte words rounded up per call, so a restore can position a seeded stream |
| `composed()` | `run` returns `Err(())` on failure instead of calling `exit`, which would end a whole composed component |
| `stop(&Stop, &mut StopTarget)` | the debugger ([debug-hooks.md](debug-hooks.md)) |

`net::SocketExt` adds what `wasi:sockets@0.3` lacks: `abort` (RST) and `set_no_delay`,
`icmp_create` (unprivileged ping sockets), and provider ids plus `reclaim_*` for checkpoints of
idle and listening sockets. Without it an abort is a FIN, `TCP_NODELAY` is accepted and
ignored, a ping socket is `EACCES` and socket checkpoints are refused.

**Paused start.** `park` receives a `Restorer`: the `Rc<RefCell<Kernel>>` to replace, the loop
to wake, and `make()`, which mounts a fresh image and builds a frontend and configuration. To
resume, the embedder builds the machine with `Kernel::restore` ([checkpoints.md](checkpoints.md)),
replaces the kernel inside the `Rc`, attaches the network again (`set_network` with
`net::WasiNet::new(shared)`, then `reattach_network`), sets the thread-local `SHIFT` to the
saved instant minus the host's current monotonic time, makes `restored()` return true and
calls `shared.wake.notify()`.

## `wasmshim-run`, the reference embedder

[`wasmshim-run`](../../crates/linux-run/src/main.rs) runs a node under wasmtime:
`wasmshim-run [--image TAR] [--seed N] [--no-cache] [--quiet] [--fuel] [--sample US
[--sample-out FILE]] NODE.wasm [NODE ARGS...]`. Its clocks are virtual (monotonic from 0, wall
clock from 2026-01-01T00:00:00Z) and jump to the next timer whenever the node waits; every
`wasi:random` stream is a splitmix64 generator from `--seed` (default 1); `--image` serves the
tar as blob `rootfs`. It prints a timing line on stderr unless `--quiet`.

The guest's stdout and stderr are written through to the process's as they are produced. Its
stdin is the process's, for a console (`--console`) and for a plain program alike, delivered one
line at a time (at most 64 KiB) and only when the node can do nothing else: every timer has
fired and every task waits. So piped input replays exactly (the same bytes at the same virtual
times, however fast they arrive), and a guest that waits for input with a timeout always sees
the timeout first ([`stdio.rs`](../../crates/linux-run/src/stdio.rs)).

Compiled components are cached under `wasmshim/compiled` in the user's cache directory (or
`$LINUX_RUN_CACHE`) and loaded back as trusted native code (wasmtime checks only the header and
configuration): keep that directory private. Nothing prunes it; it is safe to delete.
`wasmshim-run` does not serve `translate:jit`.

## Limits

- `--engine profile` collects a profile that only an embedder can read (`wasmshim_linux_node::profile`).
