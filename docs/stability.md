# Stability

Everything is at version 0.0.0, but not everything moves at the same speed.

## The core

What this repository builds, runs and tests end to end: the node and its command line,
`wasmshim-run`, images and recipes, `cargo xtask aot`, the personality's `Kernel`, `Config` and
`Host`, the CPU contract and the checkpoint container.

`cargo xtask build` gives a node that interprets; `cargo xtask aot` gives one that runs
translated code, from a static sweep of the image's programs. The translator is tested against
the interpreter instruction class by instruction class. Not here yet: translating from a
recorded profile (the library calls exist), and a host for the run-time tier
([details](design/engines-and-translation.md)).

`images/alpine-busybox.toml` is a second recipe, for a dynamically linked busybox on Alpine's
root filesystem. It pins packages in Alpine's repository, which drops old package versions, so
it can stop building; the toybox recipe names a release file that stays.

## Embedder surface, unstable

Public because a program that embeds the node needs it, with unit tests but no user in this
tree, and the first to change:

- the `/dev/fb<n>` and `/dev/hidraw<n>` devices;
- the debug and inspection hooks (`wasmshim_linux::debug`, the substrate's probes and stops);
- `NodeExt` and `net::SocketExt`, the node's extension points, including its paused start for
  a restore;
- the node's `jit`, `blob-words` and `step-stats` features;
- `EngineInstance` and the core-module tier of `wasmshim-wasmtime`.

Each says so in its documentation.
