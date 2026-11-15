# The node in a browser (`web/`)

[`web/`](../../web) runs the plain node ([node.md](node.md)) in a browser tab, on the
[codelib-web](https://github.com/backkem/codelib/tree/main/web) host kit: the page plays the part
of [`wasmshim-run`](../../crates/linux-run/src/main.rs). Nothing in the node changes for it: the
same `wasmshim_linux_node.wasm` that `cargo xtask build` makes, transpiled to JavaScript and core modules
with the kit's pinned jco.

**Guarantee.** A browser run repeats itself: one command and one seed give the same output bytes
and the same virtual time. It is not guaranteed to match a native run byte for byte (the order
of work inside one instant follows jco's runtime, and entropy comes from another generator), but
for the page's examples stdout, stderr, the exit status and the virtual time all equal
`wasmshim-run`'s, and the test checks that.

## What the page serves

| The node imports | `wasmshim-run` | The page ([`src/node.ts`](../../web/src/node.ts)) |
|---|---|---|
| `wasi:clocks` (0.2 and 0.3) | a virtual clock from 0 that jumps to the next timer when every task waits; wall clock 2026-01-01T00:00:00Z plus that | the kit's `virtualWorld`: the same, driven by the kit's event loop, which reports the component idle and lets the clock jump |
| `wasi:random` | splitmix64 per stream from `--seed` | the kit's seeded generator (another algorithm: other bytes for the same seed) |
| `wasi:cli` arguments, stdio, exit | `linux-node` plus the arguments; stdout and stderr written through as they are produced; `I32Exit` | the same arguments; stdout and stderr streamed to the page as they are written; `exit-with-code` recorded and the guest unwound |
| `linux:host/blob` | the tar file, read at offsets | the image in memory: `size` and `read` slice an `Uint8Array` (`blob-words` too, for a node built with it) |
| stdin | the process's stdin, one line at a time when nothing else can move | keystrokes posted by the page (`--console` only) |

The exit status is the guest's, or 0 when `run` returns, as natively; 125 when the node itself
fails (a missing program, bad options).

## The run

```text
page    fetch assets.json; compile the core modules; fetch the image (gzip) and inflate it
        per run: runInWorker(worker.js, { node, image, args, seed }), deadline 5 min
worker  virtualWorld({ seed, args: ['wasmshim-linux-node', ...args] }); wasiImports(world) + blob imports
        instantiate(wasmshim-linux-node.js, imports, { modules })  ->  call wasi:cli/run  ->  exit status
        world.now() is the virtual time; performance.now() around run is the wall time
```

One run per Worker: jco's runtime and the kit's loop are global to a realm, and terminating the
Worker is the only way to stop a component that computes forever. The image (about 0.9 MB
uncompressed) is copied into each Worker.

The node waits between scheduling quanta and in `nanosleep` through WASI 0.3's
`monotonic-clock.wait-until`: the kit's world arms a timer, the loop finds the component idle and
moves the clock to it. `sleep 10` therefore takes ten virtual seconds and a few milliseconds of
wall time.

**The shell.** With `--console`, the node reads raw keystrokes from `wasi:cli/stdin@0.3` with a
pump task that always has a read pending. Under a virtual world, a read that waits for the user
would leave the component idle with nothing to wake it, and the loop would report a deadlock. The
kit waits for a pending stdin read `patient`ly (its loop's `patient`, since kit 0.1.1): while
the guest has timers, virtual time runs; when it waits for keystrokes alone, the loop waits for
them in wall time. The binding only hands the page's keystrokes to the world as its `stdin`.

## Size

The release node is 19.4 MB, 16.5 MB of it DWARF line tables: the build drops the core modules'
`.debug_*` custom sections (the name section stays, for stack traces). The page downloads about
3.3 MB of node (core module and jco's glue) and the 0.6 MB gzipped image, plus 60 KB of the kit.

## Limits

- The node interprets; the translator ([engines-and-translation.md](engines-and-translation.md))
  is not wired into this build. `sha256sum` of the 0.9 MB toybox binary takes about 0.8 s of
  wall time in the tab.
- No memory cap and no instruction budget: a run ends at its exit or its wall-clock deadline.
- No network: the plain node is built without `net`.
- The shell gets a line at a time from an input field; its output has terminal control
  sequences removed.
