# wasmshim in the browser

`wasmshim-linux-node` in a browser tab: the page does what [`wasmshim-run`](../crates/linux-run) does
natively (a virtual clock, seeded entropy, the root filesystem as a blob, stdio, the exit
status) and runs [toybox](https://landley.net/toybox/), an unmodified static AArch64 binary,
from a small image ([`images/toybox.toml`](../images/toybox.toml)). Live at
<https://backkem.github.io/wasmshim/>.

**Chromium-based browsers only** (Chrome or Edge 137 or newer): the component runtime needs
WebAssembly JSPI. The page says so where it is missing.

Each **Run** is one command in a fresh Worker: `/bin/toybox uname -a`, or a shell script with
`/bin/toybox sh -c '...'`. **Shell** starts toybox's shell on a console (`--console`) and sends
what you type, a line at a time. The status line shows the exit status, the virtual time at the
end and the wall time of the run.

## What is here

| Path | What |
|---|---|
| `src/node.ts` | The binding: `runNode({ node, blobs, args, seed, stdout, stderr, stdin })` instantiates the transpiled node with the [codelib-web](https://github.com/backkem/codelib/tree/main/web) kit, gives it the kit's WASI over `virtualWorld({ seed })`, serves `linux:host/blob` (and `blob-words`) from memory, calls `wasi:cli/run` and returns the exit status and the times. |
| `src/worker.ts`, `src/main.ts`, `page/` | The page: the node and the image are downloaded and compiled once, each run gets its own Worker (the kit's `runInWorker`/`serveRun`) with a wall-clock deadline. |
| `scripts/build.mjs` | Assembles `dist/`: tsc, the kit's browser modules, `wasmshim_linux_node.wasm` transpiled with the kit (its DWARF left out: 16.5 of its 19.4 MB), the image gzipped, the notices. |
| `scripts/notices.mjs`, `licenses/` | `NOTICES.html`: every artifact of the image recipe, checked against its SHA-256, with its licence, version and sources, and upstream's licence texts. |
| `scripts/third-party.mjs` | The rest of `NOTICES.html`: the licence files of the Rust crates linked into the node (from `cargo metadata` and the crates' sources) and of the npm packages whose code runs in the tab. |
| `test/run.mjs` | The page end to end in headless Chromium, every run compared with `wasmshim-run`. |
| `vendor/` | The kit, as a tarball with its source and hash ([vendor/README.md](vendor/README.md)). |

```js
import { runNode } from './node.js';

const { exitCode, virtualNs, wallMs } = await runNode({
  node: { url: 'node/wasmshim-linux-node.js', modules },   // compiled core modules by file name
  blobs: { rootfs: imageBytes },                  // an uncompressed tar
  args: ['/bin/toybox', 'uname', '-a'],
  seed: 1n,
  stdout: (bytes) => term.write(bytes),
});
```

How it works: [docs/design/browser.md](../docs/design/browser.md).

## Build and test

Needs Node 24 and a Chromium, besides the Rust toolchain and `wasm-tools` the workspace needs
anyway. From the workspace root:

```sh
cargo xtask web            # the toybox image, the node, `npm ci`, `npm run build` -> web/dist
cargo xtask web --test     # also wasmshim-run and `npm test` (headless Chromium)
```

Or step by step, in `web/`, once the image and the node are built:

```sh
npm ci
npm run build              # dist/
npm test                   # CHROMIUM=/path/to/chrome to pick the browser
npm run serve              # http://127.0.0.1:8080/wasmshim/
```

`npm test` serves `dist/` under `/wasmshim/` (as GitHub Pages does) and runs `uname -a`, two
shell scripts, `cat`, `sha256sum` and `sleep 10`, each compared with `wasmshim-run` (from
`target/release`, or `LINUX_RUN`): the same stdout, stderr, exit status and virtual time. It
checks that `sleep 10` takes ten virtual seconds in far less wall time, that one seed repeats
the same bytes and virtual time and another seed does not, the exit status of a missing program
(125) and of a failing command, a session typed into the shell, the notices page, the message a
browser without JSPI gets, phone width, and that the page makes no external requests.
`.github/workflows/pages.yml` does all of this and deploys `dist/`.

## Limits

- The node interprets: the translator is not wired into this build.
- The shell's input is a line at a time from an input field, not a full terminal: no cursor
  keys or tab completion. Its output is shown with terminal control sequences removed.
- A run's random bytes differ from `wasmshim-run`'s for the same seed (the two hosts use different
  generators), so programs that print random data differ, and their virtual time may too.
