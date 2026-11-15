# Engines and translation

Guest code runs behind one contract, `wasmshim_cpu::GuestCpu`, in one of two ways: the
`wasmshim-aarch64` interpreter alone, or the two-tier engine (`wasmshim-aarch64-engine`) that runs
translated WebAssembly where it has some and the interpreter elsewhere.

**Guarantee: speed is invisible.** For the same state, memory and budget, every engine retires
the same instructions, stops at the same points, and leaves the same registers, memory and
retired count. Which tier ran changes only wall-clock time.

## The `wasmshim-cpu` contract

[`wasmshim-cpu`](../../crates/guest-cpu/src/lib.rs) knows no OS:

- `GuestCpu::run_until_event(mem, budget)` runs until an `Event` or until `budget` instructions
  retired: `Syscall(SyscallFrame)` (PC already past the `SVC`), `Fault(Fault)` (PC still on the
  faulting instruction: `Memory`, `Undefined`, `Breakpoint`, `Alignment`) or `Yield` (budget
  used up; calling again resumes seamlessly). An `SVC` counts as retired, a faulting
  instruction does not. The exclusive monitor is cleared on return.
- `set_virtual_time(ns)` sets what the user-readable counter reports for the next run; a
  backend never reads a clock. `context`/`set_context` move the full register state
  (`CpuContext::A64`), `duplicate` copies a CPU for `fork`/`clone`.
- `DebugCpu::run_debug(mem, budget, breaks)` stops before listed PCs and keeps the monitor,
  so a debugger stop does not change the thread's run.
- `GuestMemory` is an address space with per-page R/W/X permissions; violations are
  `MemFault`s, never panics. `code_epoch` moves whenever code may have changed and
  `code_version(addr)` per page, so code caches revalidate one page instead of all.
  [`PagedMemory`](../../crates/guest-cpu/src/paged.rs) is the sparse implementation: 4 KiB
  pages, 48-bit addresses, pages allocated on first write, copy-on-write `duplicate`, lazy
  file-backed ranges (`map_source`) and shared ranges (`map_shared` over `SharedPages`).

## The interpreter

[`wasmshim-aarch64`](../../crates/cpu-aarch64/src/lib.rs): `decode` turns a word into an `Inst`;
`Cpu` holds the user register state and executes against a `GuestMemory`. Decoded instructions
are cached per code page and revalidated by page version
([`code_cache.rs`](../../crates/cpu-aarch64/src/code_cache.rs)). Engines use the building
blocks `run_raw` (keeps the monitor), `run_block` (one basic block) and `step_one` (one
instruction, reporting whether it wrote into code). Coverage is in [cpu.md](cpu.md).

## `wasmshim-translate`: the ISA-neutral half

[`wasmshim-translate`](../../crates/translate-kit/src/lib.rs) knows no guest ISA, OS or host:

- [`ir`](../../crates/translate-kit/src/ir.rs): pure `i32`/`i64`/`v128` operations, guest-state
  slots the back half may keep in locals, guest memory accesses with a TLB fast path and an
  interpreter slow path (`Op::Mem`), one reference-interpreter step (`Op::Step`), front-half
  helpers (`Op::Helper`), and exits that give back the budget of what did not run (`Op::Exit`).
  [`opt`](../../crates/translate-kit/src/opt.rs) merges TLB checks of accesses off one base.
- [`region`](../../crates/translate-kit/src/region.rs): a region is blocks of one 4 KiB code
  page compiled into one function. [`nest`](../../crates/translate-kit/src/nest.rs) turns
  single-entry loop nests into loop regions laid out as structured Wasm loops.
- [`abi`](../../crates/translate-kit/src/abi.rs): the entry `f(cpu, tlb, ctx) -> exit`, the
  40-byte context block (budget left, entry block, page base, chain table, code epoch), exit
  codes, the two direct-mapped TLBs (2048 entries each, read and write) and the chain table
  (8192 two-way sets) through which regions tail-call each other.
- [`aot`](../../crates/translate-kit/src/aot.rs): a translation is a function of a page's bytes
  and its hot entries, and is position-independent, so it is keyed by content (the first 16
  bytes of the page's SHA-256). The directory is sorted by a fast hash and confirmed by a
  whole-page compare. `object` emits a relocatable Wasm object that the normal linker links.
- [`module`](../../crates/translate-kit/src/module.rs) and
  [`tier`](../../crates/translate-kit/src/tier.rs): the same translations as standalone modules,
  and the run-time tier that produces them.

## `wasmshim-aarch64-translate`: the A64 half

[`wasmshim-aarch64-translate`](../../crates/cpu-aarch64-translate/src/lib.rs) decodes pages into
blocks (a block ends at a branch or an `SVC`) and lowers A64 to the IR (`a64.rs`): registers
as slots, NZCV as lazy flags, addressing modes, exclusives, the system registers it knows.
Scalar FP (`a64/fp.rs`), Advanced SIMD lanes (`a64/neon.rs`), fused multiply-add emulated
exactly (`a64/fma.rs`) and common SIMD integer and move instructions (`a64/simd.rs`) run
inline. A float result that is a NaN, a saturating lane that saturated, or an FMA outside its
exact range hands the instruction to the interpreter. Still stepped: half precision, LSE
atomics and compare-and-swap, pair exclusives and the rest of what is not listed. The back
half ([`emit.rs`](../../crates/cpu-aarch64-translate/src/emit.rs)) writes the A64 code shape;
`emit::Options` holds its switches, and the defaults are what ships. `TRANSLATOR_VERSION`
names the format; translations of another version are ignored at run time.

## The engine

[`wasmshim-aarch64-engine`](../../crates/cpu-aarch64-engine/src/lib.rs): `local::DbtCpu` is a
`GuestCpu` over an `EngineMem` address space ([`mem.rs`](../../crates/cpu-aarch64-engine/src/mem.rs):
guest pages, the TLBs, installed translations, execution counters). The
[dispatcher](../../crates/cpu-aarch64-engine/src/dispatch.rs) loops:

```text
pc has a translation?  yes -> call the region; handle its exit code
                       no  -> interpreter, one basic block (count its executions)
until the budget is used up or an event happens
```

Exit codes: `EXIT_JUMP` (continue at the new PC), `EXIT_SYSCALL`, `EXIT_BUDGET` (finish the
budget in the interpreter), `EXIT_MEM` (run that one memory instruction in the interpreter,
filling the TLB, and go on), and `jit_step` statuses. Translated code calls `linux_jit_step`
to have the interpreter execute one instruction and `linux_jit_fill` to fill a TLB entry. The
write TLB never holds an executable page, so a store into code always reaches the interpreter,
which moves the page's code version; translations of the old version are not run again.

**Ahead-of-time directory.** The object linked into the module defines
`linux_aot_directory`. After two distinct untranslated entries into a page, the engine hashes
the page and installs the directory's entries for it, whatever address the page is mapped at.

**Run-time tier.** With a host that can instantiate modules, a block whose interpreted
executions reach `Policy::threshold` gets its page translated (with the page's other warm
blocks) into
the same code an ahead-of-time build would link, and installed. The capability is the WIT
interface `translate:jit` ([`jit.wit`](../../crates/translate-kit/wit/jit.wit)): "instantiate
this module against my memory and table". A host that answers `unsupported` turns the tier off
and the engine keeps interpreting. Translations are cached by page content.

**Profiling.** `profile::ProfCpu` is the interpreter one block at a time, counting block
entries; hot blocks are recorded as `PageProfile`s keyed by content, the input of
`wasmshim_aarch64_translate::aot::translate_page` and `object`.

A development-only variant, `wasmshim-aarch64-engine-module`, builds the engine as an import-free
core module driven through the `wasmshim-cpu-remote` ABI; `wasmshim-wasmtime::EngineInstance`
hosts it in its own wasmtime store and compiles what it asks for.

## Exactness

Translated code retires exactly what the interpreter would:

- every block first checks that the budget left covers the whole block; if not it exits with
  `EXIT_BUDGET` before running anything, and the dispatcher finishes the budget in the
  interpreter;
- a block that leaves early (a syscall, `EXIT_MEM`, a `jit_step` that faults) gives back the
  budget of the instructions it did not retire; a stepped instruction counts as not retired
  unless the engine's status says it did (`STATUS_CODE`), and the engine takes it off then;
- loops unrolled for speed check the budget once for all their copies; the checked loop stays
  the exact slow path;
- the dispatcher derives the retired count from the budget consumed, so `retired()` agrees;
- with debugger breakpoints armed, `DbtCpu::run_debug` interprets one instruction at a time,
  because a region is never stopped mid-way at an arbitrary address.

[`differential.rs`](../../crates/cpu-aarch64-translate/tests/differential.rs) runs random
encodings per instruction class through the interpreter and as translated regions in wasmtime
and requires identical registers, flags, PC, memory and budget.

## What this tree can build today

- `cargo xtask build` links an **empty** ahead-of-time object into `wasmshim_linux_node.wasm`
  ([`build.rs`](../../components/linux-node/build.rs)). That node interprets everything;
  `--engine auto` picks the interpreter because no translation is linked.
- `cargo xtask aot <recipe> [PATH...]` builds a node that runs translated code
  ([`xtask/src/aot.rs`](../../xtask/src/aot.rs)). It needs no run: it takes every code page of
  the image's AArch64 ELF files (or of the files named), finds block entries with the kit's
  linear sweep (`static_entries`: the page start, the instruction after each block end, direct
  branch targets inside the page), translates them (`aot::translate_page`, `aot::object`) and
  builds the node with `LINUX_AOT_OBJ` set to the object, into `target/aot/<recipe>/`. A page
  is found at run time by its content, so the node also runs any other image; pages and entries
  the sweep did not cover run in the interpreter. For toybox: 195 pages, 41,834 blocks, 17 MiB
  of Wasm; a shell loop plus a SHA-256 (1.38 billion instructions) runs 99.6% translated, in a
  fifth of the interpreter's time, with the same output, retired count and virtual time.
- Translating from a **profile** instead (only the blocks a run showed hot, which gives a
  smaller object for a large image) has no command here: `--engine profile` records one,
  `wasmshim_linux_node::profile()` reads it in an embedder, and the same two `aot` calls take it.
- `wasmshim-linux-node`'s `jit` feature imports `translate:jit`, which no host in this tree serves;
  `wasmshim-run` does not link it, so such a build does not instantiate there.

## Limits

- Engine state is global to the engine instance (translated code calls back through symbols that
  carry no state pointer): one cell on `wasm32`, one per thread natively
  ([`state.rs`](../../crates/cpu-aarch64-engine/src/state.rs)). Engines on different native
  threads are independent; engines on one thread share the run-time tier and the counters.
- The run-time tier's per-page records are never pruned.
- The remote engine's host-facing services assume a trusted host and panic on bad arguments.
