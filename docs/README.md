# Documentation

Design notes for wasmshim: how each layer works, what it guarantees and what it does not do.
Every note links the source it describes; where a note and the code disagree, the code is
right and the note is a bug.

| Document | What it covers |
|---|---|
| [design/substrate.md](design/substrate.md) | `wasmshim-kernel`: tasks, the deterministic scheduler, the one wait primitive, syscall entry, the trace, what a personality and a host supply. |
| [design/engines-and-translation.md](design/engines-and-translation.md) | The `wasmshim-cpu` contract, the interpreter, `wasmshim-translate` and the A64 translator, the two-tier engine, exactness, and what this tree can build today. |
| [design/cpu.md](design/cpu.md) | Which A64 instructions decode and which are `Undefined`, the floating-point model, what the CPU reports. |
| [design/personality.md](design/personality.md) | `wasmshim-linux`: the kernel object, VFS, processes, signals, terminals, sockets, `/proc`, and what is not modelled. |
| [design/checkpoints.md](design/checkpoints.md) | The snapshot container, what a checkpoint saves, what blocks one, restoring, and the trust model. |
| [design/debug-hooks.md](design/debug-hooks.md) | Probes, stops and steps in the substrate, and the personality's typed debug API. |
| [design/image-format.md](design/image-format.md) | The indexed tar image, recipes, the packer, and how a node reads its image. |
| [design/node.md](design/node.md) | `components/linux-node`: imports, command line, features, the run loop, `NodeExt`, and `wasmshim-run`. |
| [design/browser.md](design/browser.md) | `web/`: the node in a browser tab, what the page serves in `wasmshim-run`'s place, the shell's input, and the limits. |
| [syscalls.md](syscalls.md) | Every implemented system call, with notes where support is partial. |
| [known-issues.md](known-issues.md) | What is known to be wrong or missing today: exactness holes in translated code, Linux semantics, images, tests. |

## Reading order

1. [node.md](design/node.md), to see what runs and how it is driven.
2. [substrate.md](design/substrate.md), then [personality.md](design/personality.md): the
   machine underneath a syscall.
3. [syscalls.md](syscalls.md) and [cpu.md](design/cpu.md), when a program does not run: the two
   places a guest meets the edge of what is implemented.
4. [engines-and-translation.md](design/engines-and-translation.md), for speed and exactness.
5. [image-format.md](design/image-format.md), [checkpoints.md](design/checkpoints.md),
   [debug-hooks.md](design/debug-hooks.md) and [browser.md](design/browser.md), as needed when
   embedding.
