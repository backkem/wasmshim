# Known issues

What we know is wrong or missing today, beyond "this system call is not implemented" (for
that, see [syscalls.md](syscalls.md)) and the documented limits of each layer
([design/personality.md](design/personality.md), [design/cpu.md](design/cpu.md)). Version
0.0.0 is experimental; this list is here so that nobody has to find these the hard way.

## Exactness of translated code

The promise is that speed is invisible: a program retires the same instructions and sees the
same results interpreted and translated. Three known holes:

- **Translated code ignores execute permission.** A page that is not executable, and whose
  bytes match a page that was translated, runs translated where the interpreter raises a fault
  (`crates/cpu-aarch64-engine/src/mem.rs`). A program that jumps into its own data on purpose
  can tell the two apart.
- **Read watchpoints are bypassed in translated code.** The fill for a write also installs the
  read entry, and a merged check uses the write check for its loads. This is embedder surface
  (debugging), not something a guest sees.
- **Some reserved encodings execute** where a real CPU raises an undefined-instruction fault.
  Both tiers agree with each other here; they disagree with hardware.

Also in the CPU: fixed-point `SCVTF` and `UCVTF` round twice, so a few results differ from
hardware in the last bit.

## Linux semantics

- **Standard input without `--console`** is a synchronous host call: a read blocks every guest
  thread, `O_NONBLOCK` is ignored, and `poll` always reports it readable. A program that
  multiplexes stdin with sockets or timers stalls. Use `--console`.
- **Pid 1 gets every signal.** Linux delivers to pid 1 only the signals it has a handler for;
  here `kill 1` and Ctrl-C end a program that runs as pid 1, which is what most programs here
  do.
- **`madvise(MADV_DONTNEED)` on a private file mapping** zeroes the pages; Linux reloads them
  from the file.
- **`io_submit` does the I/O before it checks the ring has room**, and **`epoll_wait` with a
  small buffer** returns low descriptors first every time, so high ones can starve.
- **`sendfile` from a pipe or socket with an offset pointer** reads the pipe; Linux returns
  `ESPIPE`.
- **`setsockopt` with a length above four bytes** for an integer option is accepted and not
  stored; **`getsockopt(SO_ERROR)`** clears the error before it checks the length.
- **`kill(-1)` and `kill(-pgid)` do not count zombies**, so they can return `ESRCH` where Linux
  returns success.
- **Netlink:** a malformed message is answered with an error where Linux drops it, and the
  replies to one large write of dump requests are not bounded by a receive buffer.

## Limits that can be got around

`Config::limits` bounds the bytes written to files. Extended attributes (`setxattr`) and
message queue sizes (`msgctl(IPC_SET)`) are not counted against it.

## Images

- **The packer writes members in path order**, so a hard link can come before its target. The
  node mounts such an image correctly; GNU `tar -x` of the same file drops the link.
- **Members under a symlinked directory are dropped** when an image is mounted.
- **`images/alpine-busybox.toml` names packages in a repository that is updated in place**: the
  recipe stops building when a package is replaced. `images/toybox.toml` names a release file
  and keeps working.

## Tests

- **No `cargo test` runs a real program end to end.** CI's smoke job does (toybox, interpreted
  and translated); the unit and differential tests cover the layers.
- **The engine's translated path** (dispatcher, memory fills, block chaining) is exercised only
  in WebAssembly builds, by that smoke job and the page's test.
- **[syscalls.md](syscalls.md) is kept by hand**: no test compares it with the dispatch table.
