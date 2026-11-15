<h1 align="center">wasmshim</h1>

<p align="center">
  <b>Linux programs, no Linux.</b><br>
  Unmodified AArch64 binaries, inside a WebAssembly component.<br>
  <a href="https://backkem.github.io/wasmshim/">Try it in your browser</a>
</p>

---

Take a stock arm64 Linux binary and run it where there is no Linux. wasmshim emulates the CPU
underneath and answers the system calls on top, with no kernel in between. All of it is one
WebAssembly component that speaks WASI 0.3, so it runs wherever components do: a server, a
test, a browser tab.

- 📦 **Stock binaries.** ELF files from a tar image, static or dynamically linked, nothing
  recompiled. toybox and busybox out of the box; it has run nginx, Node.js and Chromium.
- 🎯 **Deterministic.** Time, entropy and scheduling come from the embedder. Run it twice and
  you get the same bytes, down to the number of instructions retired.
- ⚡ **Fast without telling.** An AArch64 → Wasm translator runs programs several times faster
  than the interpreter, and a program cannot tell which one it is on.
- 🧱 **Sealed.** The only way out is WASI. No host files, sockets or clocks behind the
  component's back.
- 💾 **Checkpoints.** A running machine can be saved and restored later.

## Quick start

```sh
cargo install --locked wasm-tools@1.261.0     # `curl` must be on PATH too

cargo xtask image toybox            # a static toybox, pinned by SHA-256, packed into one tar
cargo xtask build                   # the node: target/wasm32-wasip2/release/wasmshim_linux_node.wasm
cargo run --release -p wasmshim-run -- --image target/images/toybox.tar \
    target/wasm32-wasip2/release/wasmshim_linux_node.wasm /bin/toybox uname -a
```

```text
Linux linux 6.6.0 #1 SMP PREEMPT aarch64 Toybox
wasmshim-run: load=0.014s (cache hit) instantiate=0.000s run=0.001s virtual=0ns exit=0
```

`wasmshim-run` is a small wasmtime embedder with a virtual clock and seeded entropy. That
node interprets. For translated code, translate the image's programs ahead of time (about five
times faster on a shell loop and a SHA-256, same output at the same virtual time):

```sh
cargo xtask aot toybox
cargo run --release -p wasmshim-run -- --image target/images/toybox.tar \
    target/aot/toybox/wasmshim_linux_node.wasm /bin/toybox sha256sum /bin/toybox
```

## How it fits together

```text
   toybox, busybox, nginx ...     stock arm64 ELF binaries from a tar image
  ─────────────────────────────   Linux system calls
   wasmshim-linux                 files, processes, signals, ttys, sockets, /proc
   wasmshim-kernel                tasks, a deterministic scheduler, waits on virtual time
   wasmshim-aarch64 (+ engine)    an interpreter, and an AArch64 → Wasm translator
  ─────────────────────────────   WASI 0.3 (+ a small `linux:host` for the image)
   any component runtime
```

Each layer knows only the one below it: a CPU knows no OS, the substrate knows no CPU model,
and a personality sees system-call frames and guest memory, never registers.
`cargo xtask check-deps` keeps it that way.

## What's in the box

| Crate | What it does |
|---|---|
| `wasmshim-linux-node` | Everything below as one WASI 0.3 command component. Start here. |
| `wasmshim-run` | Runs the node on wasmtime, from the command line. |
| `wasmshim-linux` | Linux as a library: the system calls, with `wasmshim-linux-abi` and `wasmshim-linux-elf`. |
| `wasmshim-kernel` | The substrate a personality stands on: tasks, scheduler, one wait primitive, checkpoints, debug hooks. |
| `wasmshim-aarch64` | The A64 decoder and interpreter, behind the CPU contract of `wasmshim-cpu`. |
| `wasmshim-translate`, `wasmshim-aarch64-translate`, `wasmshim-aarch64-engine` | The guest-code → Wasm translator, and the engine that runs both tiers behind one dispatcher. |
| `wasmshim-linux-image`, `wasmshim-image-pack` | Root filesystem images: one indexed tar, built from pinned, hash-checked downloads. |
| [`web/`](web) | The node in a browser tab, on the [codelib-web](https://github.com/backkem/codelib/tree/main/web) kit: the page behind the link above. |

Crates live in `crates/` and `components/`, in directories named after what they hold
(`crates/guest-kernel` is `wasmshim-kernel`, `crates/linux-personality` is `wasmshim-linux`).

## Going deeper

- [docs/](docs/README.md): how each layer works, with a reading order.
- [docs/syscalls.md](docs/syscalls.md): every system call that is implemented.
- [docs/known-issues.md](docs/known-issues.md): what we know is wrong today.
- [docs/stability.md](docs/stability.md): which parts of the API are the core, and which move.
- [CONTRIBUTING.md](CONTRIBUTING.md): building, testing and the rules a change keeps.

```sh
cargo test               # every crate, natively
cargo xtask check-deps   # the layers still point one way
cargo xtask web --test   # the browser page, in headless Chromium (needs Node 24 and a Chromium)
```

## Status

Experimental, version 0.0.0. The system-call surface is what the programs we have run needed,
not all of Linux, and APIs still move. Built on Rust 1.98.1 and wasmtime 50.0.1, both pinned.

## License

MIT or Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).
Image recipes download third-party software (Alpine Linux, busybox, toybox) under its own
licenses; none of it is stored here. The browser page ships a toybox image and generates its
notices with it.
