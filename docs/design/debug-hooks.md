# Debug hooks

The substrate and the Linux personality expose typed hooks that a debugger, a profiler or an
inspector builds on. They are plain Rust values, with no wire format: an embedder puts its own
protocol on top.

**Guarantee: stops are exact and change nothing.** With nothing armed, the hooks cost one
counter increment and one emptiness check per syscall and one flag test per CPU run. When a
probe fires, the machine stands still until the embedder answers. A stop that writes nothing
leaves every guest-visible fact as it was: the same instructions retire, the same preemption
points are taken, and the exclusive monitor survives. Writes made at a stop are the
debugger's, applied at that exact point.

## Probes and stops

[`wasmshim_kernel::debug`](../../crates/guest-kernel/src/debug.rs) and
[`wasmshim_kernel::observe`](../../crates/guest-kernel/src/observe.rs). A personality owns an
`observe::Observer` and hands it to the scheduler through `Personality::observer`. Probes are
armed on it with `Observer::arm(&[ProbeSpec])` and removed with `disarm`:

```rust
ProbeSpec {
    id: 7,                 // 0 is reserved for steps
    point: Point::Syscall { number: Some(63), exit: true, tid: None,
                            args: [None; 6], ret_ge: Some(0) },
    hits: Some(1),         // None: unlimited; disarmed after the last hit
    hold: true,            // stop for the debugger (false: report only, answered at once)
    info: true,            // attach the substrate's StopInfo
}
```

`Point`s:

| Point | Fires |
|---|---|
| `Syscall { number, exit, tid, args, ret_ge }` | at entry or exit of matching syscalls |
| `Sys { n, exit }` | at the `n`th syscall entry (node-wide, from 1) |
| `Insn { n, every }` | when the node has retired exactly `n` instructions (then every `every`) |
| `Pc { addr, pid, tid }` | before a thread executes the instruction at `addr` |
| `Watch { start, len, pid, deny }` | on reads and/or writes of a memory range, by the CPU or a syscall |

The two clocks are host-side: `sys` counts syscall entries (a blocked call that restarts counts
again) and `insn` is `Core::retired`. A syscall completed by a wake (a sleep, a futex, a
deadline) has no exit stop.

When a probe fires, the run loop calls `Host::stop(&Stop, &mut dyn StopTarget)`. `Stop` carries
the probe id, whether it holds, the `Position` (`Sys { n, exit }` or `Insn(n)`) and, if asked,
a `StopInfo`:

- `Syscall { sys, number, tid, args, name, ret }`;
- `Insn { insn, tid, pc, retired, why, pid, symbol }`, with `why` one of `Insn`, `Pc`, `Step`;
- `Watch { insn, tid, pc, retired, start, len, old, new, by, fault, pid, symbol }`, with `by`
  either the CPU or the syscall that changed the range.

`StopTarget` gives the stopped `tid` and the personality's own state as `dyn Any`
(`Personality::as_any`); the Linux personality downcasts to `Kernel`. The answer is a `Resume`:
`wrote` says the debugger changed registers or memory (at a syscall entry the frame is then
read again from the registers, so changed arguments take effect), and `step` asks for another
stop after `n` steps `StepBy::Sys` (syscall entries), `StepBy::Insn` (instructions of the
stopped thread) or `StepBy::Line` (source lines, through `Personality::source_line`). The
default `Host::stop` resumes at once and changes nothing.

## How instruction stops stay exact

While instruction probes or an instruction step are armed, the scheduler cuts each CPU run at
the next target with `DebugCpu::run_debug(mem, budget, breaks)`: an exact budget, breakpoints
checked before execution, and the exclusive monitor kept. After the stop the **same** quantum
continues, so the thread runs exactly as it would have. A write at a stop clears the monitor,
as another agent's store would. Watchpoints work by denying access to the watched pages for
the duration of the thread's run: a fault the view caused re-executes the instruction without
it and compares the bytes before and after; syscalls are compared against a snapshot of the
range. The two-tier engine interprets one instruction at a time while breakpoints are armed
(a region is never stopped mid-way); only wall time differs.

## The Linux view

[`wasmshim_linux::debug`](../../crates/linux-personality/src/debug.rs) reads and writes a
machine from outside, as typed values:

- processes (`Kernel::pids`, `process`), threads (`core`, `has_thread`, `process_of_thread`),
  registers (`thread_cpu_mut`, then `context`/`set_context`);
- descriptors (`descriptor`: what each names, socket peers inside the machine) and mappings
  (`mappings`);
- memory: `with_memory` reads, `poke` stores ignoring page permissions (like `ptrace`; a
  store to code moves the page's code version, so no engine runs stale code);
- files: `vfs`, `/proc` and `/sys` through `procfs`, and `replace_file`;
- symbols and lines (`where_is`, `symbol`, `line_addresses`, `resolve_addresses`, `stack`) from
  ELF symbol tables and DWARF line tables read by
  [`wasmshim-debuginfo`](../../crates/guest-debuginfo/src/lib.rs), at the load bias of each mapping;
- the clocks (`retired`, `syscalls`), syscall names and numbers (`syscall_name`,
  `syscall_number`), and the profile counters (`engine_counters`, `syscall_stats`,
  `thread_stats`); the observer itself through `observer_mut`.

Reading changes nothing the guest can see. Reading an image file may fill the host-side page
cache with exactly the bytes the guest would read.

## Limits

- A backend without `GuestCpu::debug` runs as if nothing were armed: PC breakpoints and
  instruction stops never fire there. Both shipped backends have it.
- An action probe (`hold: false`) may find the machine changed since it was armed (a thread
  exited); the embedder decides whether to skip it.
- `StepBy::Line` needs line tables (`Personality::source_line`). In code without them the step
  goes on instruction by instruction and does not complete; step by instructions there.
