# The syscall substrate (`wasmshim-kernel`)

`wasmshim-kernel` is what every OS personality stands on: guest threads as tasks, a deterministic
scheduler, one wait primitive with virtual-time deadlines, address-space bookkeeping, syscall
entry and the trace. It is `no_std`, depends only on `wasmshim-cpu`, and knows no operating system
and no CPU model.

**Guarantee.** The substrate never reads a clock, an entropy source or a file. Everything
external comes through [`Host`](../../crates/guest-kernel/src/host.rs). Thread switches happen
after a fixed number of retired instructions, and wake-ups follow a fixed order. The same
`Host` answers therefore always produce the same guest behaviour.

A personality is a syscall table plus its own objects. It owns a `wasmshim_kernel::Core` and
implements `wasmshim_kernel::Personality`. It does not copy the scheduler or the waits.

## Tasks

A [`Task<T>`](../../crates/guest-kernel/src/task.rs) is one guest hardware thread: a `tid`, a
`Box<dyn GuestCpu>`, a `TaskState` (`Runnable` or `Blocked(Blocker)`), the deadline of a timed
syscall in progress (`timed`), the last syscall number it entered, and `ext`, the personality's
per-task state (reached through `Deref`). `Task::own_retired` is the task's own retired count;
`Core::retired` adds the counts of tasks that have exited.

The personality picks its types through `wasmshim_kernel::Types`: `Task` (per-task state), `Key`
(its own wait keys) and `Park` (its own parking reasons).

## The scheduler

[`sched.rs`](../../crates/guest-kernel/src/sched.rs), entered through `wasmshim_kernel::run`:

```text
run(p, host)
  on_run()                                  personality hook
  busy_until > now?  -> Wait { until: busy_until }
  loop
    since_yield >= quantum?  -> Wait { until: accrual_start + since_yield * ns_per_insn }
    now = host.monotonic_ns(); expire(now); expire_deadlines(now)
    tid = first Runnable task of the FIFO run queue, or
          -> Wait { until: earliest task or personality deadline }   (None: nothing can happen)
    run_slice(tid): up to `quantum` instructions; syscalls and faults inside the slice are
                    handled and the slice goes on while the task is runnable; it ends at the
                    quantum, a block or an exit; a runnable task goes to the back of the queue
```

`SchedConfig` holds the tunables: `isa`, `quantum` (instructions per slice, and per yield to
the host), `ns_per_insn` (virtual nanoseconds charged per retired instruction, 0 disables) and
`trace`. Every one of them changes guest-visible behaviour.

Compute is charged as virtual time. After each quantum of retired instructions `run` returns
`Status::Wait { until }` with the instant at which that work is paid for; until then the CPU is
busy (`busy_until`) and a call that arrives early returns at once. Under a virtual clock the
host jumps to `until`. Under a real clock `until` is usually already past and the return is a
plain yield, so other work on the host can run.

`Status::Exited` means `Personality::finished` returned true. `Status::Wait { until: None }`
means every task is blocked with no deadline: unless the embedder has a source of wake-ups
(sockets, a console, devices), the guest is deadlocked.

## Syscall entry

When a CPU returns `Event::Syscall(frame)`, `handle_syscall` records `current`, `current_pc`
and the task's `last_syscall`, takes `slice_now = host.monotonic_ns()`, fires observer stops at
entry ([debug hooks](debug-hooks.md)), calls `Personality::syscall`, then `after_syscall`,
writes the trace line, and applies the returned `Flow`:

| `Flow` | Effect |
|---|---|
| `Ret(v)` | `v` goes to the result register (errors already encoded as `-errno`). |
| `Exec` | The task has a new register state; nothing is written. |
| `Block(b)` | Only this task stops; for `Blocker::Restart` its PC is rewound to the `SVC`. |
| `ExitThread(c)`, `ExitGroup(c)` | The personality's `exit_thread` / `exit_group`. |

A fault (`Event::Fault`) goes to `Personality::fault` with the PC still at the faulting
instruction.

## The wait primitive

[`wait.rs`](../../crates/guest-kernel/src/wait.rs). A task waits on a `WaitKey`:
`Object(id)` (one object changed), `AnyIo` (any object changed: `poll`, `select`, epoll),
`Futex(space, addr)`, or `Personal(K)`. Why it is not runnable is a `Blocker`:

- `Restart { key, deadline, resume_pc, timeout_result }`: most waits. A wake makes the task
  runnable with its PC on the `SVC`, so the syscall simply runs again against the new state.
  At `deadline` it instead returns `timeout_result` at `resume_pc`, or, when that is `None`,
  runs again (a timer it waits for is due).
- `Sleep { until, rem }`: completes with 0 at `until`.
- `Futex { key, bitset, deadline, seq }`: completes with 0 when woken, or with the
  personality's `futex_timeout` value at its deadline.
- `Personal(P)`: parked on the personality's terms; woken only explicitly, never by a deadline.

A handler arranges a restartable wait with `Core::wait_for(key, deadline, timeout_result)`
and reports "blocked"; the personality turns it into a blocker with `Core::take_block`.
`Core::restart_deadline` fixes the absolute deadline at the first entry of a call and keeps it
across restarts, so spurious wake-ups never extend a timeout. `Core::wait_objects` is the one
multi-object wait every `poll`/`select`/`epoll_wait` uses: it sleeps on `AnyIo` until any
object changes, the earliest deadline of the watched objects (then the call runs again), or
the caller's deadline (then it returns 0).

`Core::wake` wakes restartable waiters in task-id order and futex waiters in arrival order
(`seq`), at most `limit` of them, filtered by `bitset`. `Core::requeue` moves futex waiters,
`Core::fail_waiters` ends restartable waits on a removed object with an error, and
`Core::expire_deadlines` ends every wait whose deadline passed.

## Address spaces

[`space.rs`](../../crates/guest-kernel/src/space.rs): `AddressSpace<K>` puts a mapping list
(`Vma<K>`) on top of a `GuestMemory`. `K: MapKind` is the personality's description of why a
range is mapped, so ranges can be split, merged and re-protected (`map`, `unmap`, `protect`,
`record`). `find_free_below` places new mappings top-down, with no randomisation.
`duplicate` makes a copy-on-write copy for `fork`.

## Trace and counters

With `SchedConfig::trace`, every syscall writes one line through `Host::diag`:
`[label] name(args) = result`. The personality supplies the name and argument count
(`describe`), the label (`label`) and the error rendering (`show_error`); a number it does not
name prints as `syscall_<n>`. A blocking call prints `? <blocked: ...>`.

[`stats.rs`](../../crates/guest-kernel/src/stats.rs) counts syscall entries, blocks and
blocked virtual time per syscall number and per thread. The counters are host-side only:
nothing a guest sees depends on them, and checkpoints do not save them.

## What a personality supplies

`Personality` must provide `core`/`core_mut`, `finished`, `space(tid)`, `syscall`, `fault`,
`exit_thread`, `exit_group` and `futex_timeout`. Optional hooks, called at fixed points of
`run` (their order is part of determinism): `on_run`, `expire(now)`, `next_deadline`,
`before_run(tid)` (Linux delivers signals here; `false` skips the task), `after_syscall`, the
trace hooks above, and the debugger's `observer`, `process_of`, `symbolize`, `source_line`
and `as_any`.

The embedder supplies a `Host`: `monotonic_ns`, `realtime_ns`, `random`, `stdout`, `stderr`,
`stdin`, `diag`, and optionally `stop` (the debugger; the default resumes at once). It also
supplies a `Frontend`, which makes CPUs and address spaces (`new_cpu`, `new_memory`,
`new_shared_memory`) and may report engine `counters`, and a `Blob` for read-only images.

## Limits

- One instruction set: `Isa::Aarch64` (the syscall instruction is 4 bytes long).
- Single-threaded by design: state is in `Rc`/`RefCell`, nothing is `Send`.
- A syscall completed by a wake (a sleep, a futex, an expired deadline) gets no exit stop.
