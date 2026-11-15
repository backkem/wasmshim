# The Linux personality (`wasmshim-linux`)

`wasmshim-linux` is the Linux application ABI as a library: processes, threads, file
descriptors, a VFS, memory maps, signals, terminals, sockets and `/proc`. It sees syscall
frames and guest memory, never ISA registers; it runs on the [substrate](substrate.md) and
gets its CPUs from a `Frontend`. Programs are stock AArch64 ELF binaries (static, PIE or
dynamically linked) from a tar image ([image-format.md](image-format.md)).

**Guarantee.** The personality never reads a clock, an entropy source or a host file.
Everything comes through `Host`, `Blob` and `Network`, so the same inputs give the same
syscall results in the same order.

## The kernel object

[`Kernel`](../../crates/linux-personality/src/kernel.rs) is one machine:

```rust
let mut k = Kernel::new(Config::default(), Arch::Aarch64, frontend, vfs);
k.spawn(&Spawn { path, argv, envp }, &mut host)?;     // pid 1, stdio on fds 0-2
loop {
    match k.run(&mut host) {
        RunStatus::Exited(status) => break,           // pid 1 ended
        RunStatus::Wait { until } => { /* let time pass to `until`, feed input */ }
    }
}
```

`Config` holds what changes guest-visible behaviour: `hostname`, `quantum` (0 runs as 1),
`ns_per_insn`, `trace`, `host_loopback`, the netlink `interfaces`, `kernel_release`/
`kernel_version`/`kernel_builder` (`uname`, `/proc/version`), `fb_id` and `snapshot_format`.
The run is over when
pid 1 exits; other processes still alive then simply stop. `Kernel::thread_report` lists what
every thread waits on, for a run that stalls.

## Files and the VFS

[`vfs`](../../crates/linux-personality/src/vfs/mod.rs) is one in-memory inode tree. Mounting an
image (`Vfs::mount_image`) reads its index and creates every inode; regular files keep their
bytes in the image and are read on demand. The first write copies a file into memory, so the
image is never modified. Everything the guest creates lives in memory and is gone with the
instance. Files are at most 1 GiB each.

The tree always has `/dev` (`null`, `zero`, `full`, `random`, `urandom`, `tty`, `console`,
`ptmx`, `pts/`, `shm/`, and `stdin`/`stdout`/`stderr`/`fd` links into `/proc/self`) and
`/tmp`. `/dev/hidraw<n>` and `/dev/fb<n>` appear when the embedder attaches input devices or
framebuffers ([`devices.rs`](../../crates/linux-personality/src/devices.rs)).

The guest runs as root. Permission bits, owners and credentials are recorded and reported
(`stat`, `getuid`, `/proc/<pid>/status`) but not enforced, except that `execve` needs an
execute bit. File locks (POSIX, open-file-description and `flock`), inotify (synthesised from
VFS events, exact and delivered before the writing syscall returns), extended attributes in
the `user.`, `trusted.` and `security.` namespaces, and `MAP_SHARED` file mappings with a page
cache shared with `read`/`write` are implemented.

## Processes and threads

Every thread is a substrate task; a `Process` has its address space, descriptor table, signal
handlers, credentials, timers and children ([`task.rs`](../../crates/linux-personality/src/task.rs)).
`clone` supports threads, `fork` (copy-on-write) and `vfork`; namespace flags and
`CLONE_PIDFD` are `EINVAL`. Pids are never reused within one kernel. `execve` loads ELF64
AArch64 images through [`wasmshim-linux-elf`](../../crates/linux-elf/src/lib.rs), follows `#!` up to
four levels, and resolves `/proc/self/exe`. Memory placement is deterministic and top-down
([`mm.rs`](../../crates/linux-personality/src/mm.rs)); there is no address randomisation.
The page at `0xd000_0000`, shown as `[vdso]`, holds only the `rt_sigreturn` trampoline: there
is no vDSO image (`AT_SYSINFO_EHDR` is not passed), so time calls are real syscalls.

## Signals

[`signal.rs`](../../crates/linux-personality/src/signal.rs): standard signals are pending at most
once, real-time signals queue. A handled signal interrupts a blocked syscall (`EINTR`, or a
restart with `SA_RESTART`). Delivery happens before a thread runs guest code, on the arm64
`rt_sigframe` layout with an `fpsimd_context` record. Faults become `SIGSEGV`, `SIGBUS`,
`SIGILL` or `SIGTRAP`. There is no job control: `SIGSTOP`, `SIGTSTP`, `SIGTTIN` and `SIGTTOU`
are ignored by default.

## Terminals

[`objects/tty.rs`](../../crates/linux-personality/src/objects/tty.rs) implements the `n_tty` line
discipline, pseudo-terminals (`/dev/ptmx`, `/dev/pts/N`) and the console, a terminal whose
master is the embedder (`attach_console`, `console_input`, `console_eof`). Sessions, controlling
terminals, the foreground group, `^C`/`^\` and `SIGWINCH` work; background groups are never
stopped, so `^Z` does nothing. Without a console, stdio is the host's streams and terminal
ioctls on it are `ENOTTY`.

## Sockets

- **Local:** `AF_UNIX` (stream, datagram, seqpacket, path and abstract names, `SCM_RIGHTS`) and
  `AF_INET`/`AF_INET6` to "this host" (127.0.0.0/8, `::1`, the unspecified addresses) live
  entirely in the kernel, with deterministic ephemeral ports from 49152.
- **Network:** everything else goes to a [`Network`](../../crates/linux-personality/src/net.rs)
  provider the embedder attaches (`Kernel::set_network`); without one it is `ENETUNREACH`. The
  kernel owns each socket's buffers in a shared `NetSocket` and never blocks on the provider:
  syscalls change that state and `kick` the provider, and the provider moves bytes outside
  syscalls and reports changed sockets through `take_changed`. TCP, UDP and unprivileged ICMP
  echo sockets are supported. `Config::host_loopback` sends loopback traffic that finds no
  local listener to the provider.
- **Netlink:** `NETLINK_ROUTE` `RTM_GETLINK`/`RTM_GETADDR` dumps, enough for `getifaddrs`.

`SOCK_RAW` is `EPERM`; other families are `EAFNOSUPPORT`.

## `/proc` and `/sys`

[`procfs.rs`](../../crates/linux-personality/src/procfs.rs) generates files from live state when
they are opened: `cpuinfo`, `meminfo`, `stat`, `uptime`, `version`, `loadavg`, `filesystems`,
`mounts`, `sys/kernel/*`, `sys/vm/*`, and per process `cmdline`, `comm`, `environ`, `auxv`,
`maps`, `mounts`, `stat`, `statm`, `status`, `limits`, `exe`, `cwd`, `root`, `fd/`, `task/`.
[`sysfs.rs`](../../crates/linux-personality/src/sysfs.rs) is a static skeleton: one CPU, one NUMA
node, transparent huge page settings.

## Limits

`Config::limits` ([`Limits`](../../crates/linux-personality/src/kernel.rs)) bounds what a guest
can make the node hold. At a limit the guest gets the error Linux gives there and the node goes
on. The defaults are far above what the programs run so far need; an embedder lowers them for a
node it does not trust.

| Limit | Default | At the limit |
|---|---|---|
| `tasks`: live threads of every process, zombies included | 4096 | `clone` fails with `EAGAIN`; `RLIMIT_NPROC` reports the value |
| `file_bytes`: the largest regular file | 1 GiB | a write or truncate past it fails with `EFBIG` |
| `total_file_bytes`: bytes of written files held in memory | 2 GiB | growing a file fails with `ENOSPC` |

Written files are the ones created or changed since the image was mounted; a file that is only
read stays in the image and counts nothing, and its first write counts the whole copy. Removing
a file gives its bytes back once nothing has it open.

Not bounded by these: guest memory (`mmap`, `brk`), page-cache objects (a file mapped shared, a
memfd: they hold the pages that were touched) and kernel objects. The address space the
embedder's runtime gives the node bounds those. `statfs` still reports a fixed amount of free
space, and `setrlimit` succeeds without effect: the limits are the embedder's, not the guest's.

## Deliberately not modelled

Mounts, namespaces, `chroot` (`EPERM`), `ptrace`, seccomp, io_uring, permission checks,
guest-set resource limits (`setrlimit` succeeds without effect; see Limits), more than one CPU,
real memory pressure (`sysinfo` and `meminfo` report fixed sizes), job control, `SIGIO`,
core dumps, robust futex lists, priority-inheritance futexes, `ITIMER_VIRTUAL`/`ITIMER_PROF`
expiry, and lock deadlock detection. Unimplemented calls return `ENOSYS`;
[syscalls.md](../syscalls.md) lists what is implemented.

## Known limitations

- A blocking pipe write larger than `PIPE_BUF` returns a short count instead of blocking.
- Host memory use is bounded for tasks and written files (Limits), and per call elsewhere; guest
  memory and page-cache objects are bounded only by the node's address space.
