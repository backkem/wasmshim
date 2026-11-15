# wasmshim

**Linux programs, no Linux.**

**Try it in your browser:** <https://backkem.github.io/wasmshim/> (toybox, an unmodified AArch64
binary, in a tab; Chromium-based browsers; [`web/`](web)).

Unmodified AArch64 Linux binaries run inside a WebAssembly component: an emulated CPU
underneath, the Linux system-call ABI on top, and no kernel in between. The component speaks
WASI 0.3 to the outside, and every run is deterministic, down to the number of retired
instructions.

```text
   busybox, nginx, node ...      stock arm64 ELF binaries from a tar image
  ─────────────────────────────  Linux system calls
   wasmshim-linux             files, processes, signals, ttys, sockets, /proc
   wasmshim-kernel                  tasks, a deterministic scheduler, waits on virtual time
   wasmshim-aarch64 (+ engine)        an interpreter, and an AArch64 → Wasm translator
  ─────────────────────────────  WASI 0.3 (+ a small `linux:host` for the image)
   any component runtime
```

## Quick start

```sh
cargo install --locked wasm-tools@1.261.0     # `curl` must be on PATH too

cargo xtask image toybox            # a static toybox, pinned by SHA-256, packed into one tar
cargo xtask build                   # the node: target/wasm32-wasip2/release/wasmshim_linux_node.wasm
cargo run --release -p wasmshim-run -- --image target/images/toybox.tar \
    target/wasm32-wasip2/release/wasmshim_linux_node.wasm /bin/toybox uname -a
```

`wasmshim-run` is a small wasmtime embedder with a virtual clock and seeded entropy. Run the
same command twice and you get the same bytes at the same virtual time:

```text
Linux linux 6.6.0 #1 SMP PREEMPT aarch64 Toybox
wasmshim-run: load=0.014s (cache hit) instantiate=0.000s run=0.001s virtual=0ns exit=0
```

That node interprets. To run an image's programs as translated code, translate them ahead of
time and link the result into the node (about five times faster on a shell loop and a
SHA-256, with the same output at the same virtual time):

```sh
cargo xtask aot toybox              # every program in the image; or name files: ... /bin/toybox
cargo run --release -p wasmshim-run -- --image target/images/toybox.tar \
    target/aot/toybox/wasmshim_linux_node.wasm /bin/toybox sha256sum /bin/toybox
```

`images/alpine-busybox.toml` is a second recipe, for a dynamically linked busybox on Alpine's
root filesystem. It pins packages in Alpine's repository, which drops old package versions, so
it can stop building; the toybox recipe names a release file that stays.

## What's in the box

| Crate | What |
|---|---|
| `wasmshim-cpu` | The contract between CPUs and everything above them: `GuestCpu`, `GuestMemory`, paged memory. Knows no OS. |
| `wasmshim-aarch64` | An A64 decoder and interpreter. |
| `wasmshim-translate`, `wasmshim-aarch64-translate` | A guest-code → Wasm translator: an ISA-neutral IR and back half, and the A64 front half. |
| `wasmshim-aarch64-engine` | Interpreter and translated code behind one dispatcher, with a soft TLB. |
| `wasmshim-kernel` | The system-call substrate a personality stands on: tasks, scheduler, one wait primitive, checkpoints, debug hooks. |
| `wasmshim-linux-abi`, `wasmshim-linux-elf`, `wasmshim-linux` | Linux as a library: the ABI as data, the ELF loader, and the system calls. |
| `wasmshim-linux-node` | All of the above as one WASI 0.3 command component, with extension points for embedders. |
| `wasmshim-linux-image`, `wasmshim-image-pack` | Root filesystem images: one indexed tar, built from pinned, hash-checked downloads. |
| `wasmshim-wasmtime`, `wasmshim-run` | The embedder's side, for wasmtime. |
| `wasmshim-cbor`, `wasmshim-snapshot` | Deterministic CBOR, and the snapshot container checkpoints are written in. |
| `web/` | The node in a browser tab: a small TypeScript binding on the [codelib-web](https://github.com/backkem/codelib/tree/main/web) kit, and the page behind the "try it" link ([how it works](docs/design/browser.md)). |
| `docs/` | Design notes, and the [system calls](docs/syscalls.md) that are implemented. |

Crates live in `crates/` and the two components in `components/`, in directories named without
the `wasmshim-` prefix and after what they hold (`crates/guest-kernel` is `wasmshim-kernel`,
`crates/cpu-aarch64` is `wasmshim-aarch64`, `crates/linux-personality` is `wasmshim-linux`).

## House rules

1. **Determinism.** Time, entropy and scheduling come from the embedder. Threads switch after a
   fixed number of retired instructions, never on a timer.
2. **Speed is invisible.** Interpreted or translated, a program retires the same instructions
   and produces the same output.
3. **Layers do not look down or sideways.** A CPU knows no OS. The substrate knows no CPU
   model. A personality sees system-call frames and guest memory, never registers.
4. **The only way out is WASI.** No host files, sockets or clocks behind the component's back.

## Testing

```sh
cargo test               # every crate, natively
cargo xtask build        # the node builds and validates as a component
cargo xtask check-deps   # the layers still point one way
cargo xtask web --test   # the browser page, tested in headless Chromium (needs Node 24 and a Chromium)
```

## Status

Experimental. Version 0.0.0: APIs still move, and the [system-call surface](docs/syscalls.md)
is what the programs we have run needed, not all of Linux. What we know is wrong today is in
[docs/known-issues.md](docs/known-issues.md).

`cargo xtask build` gives a node that **interprets**; `cargo xtask aot` gives one that runs
translated code, from a static sweep of the image's programs. The translator is tested against
the interpreter instruction class by instruction class. Not here yet: translating from a
recorded profile (the library calls exist), and a host for the run-time tier
([details](docs/design/engines-and-translation.md)).

### Two kinds of public API

Everything is at version 0.0.0, but not everything moves at the same speed.

- **The core** is what this repository builds, runs and tests end to end: the node and its
  command line, `wasmshim-run`, images and recipes, `cargo xtask aot`, the personality's
  `Kernel`, `Config` and `Host`, the CPU contract and the checkpoint container.
- **Embedder surface, unstable**: public because a program that embeds the node needs it, with
  unit tests but no user in this tree, and the first to change. It is the `/dev/fb<n>` and
  `/dev/hidraw<n>` devices, the debug and inspection hooks (`wasmshim_linux::debug`, the
  substrate's probes and stops), `NodeExt` and `net::SocketExt` (the node's extension points,
  including its paused start for a restore), and the node's `jit`, `blob-words` and
  `step-stats` features. Each says so in its documentation.

Built on Rust 1.98.1 and wasmtime 50.0.1, both pinned.

## License

MIT or Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).
Image recipes download third-party software (Alpine Linux, busybox, toybox) under its own
licenses; none of it is stored in this repository. The browser page ships the `toybox` image
(toybox: 0BSD; musl: MIT); its notices are generated with it.
