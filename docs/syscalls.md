# System calls

The Linux personality implements the AArch64 (asm-generic) system calls listed below. The
list is the dispatch `match` in
[`crates/linux-personality/src/syscall/mod.rs`](../crates/linux-personality/src/syscall/mod.rs);
handlers live in the modules next to it. A blank note means the call is implemented as on
Linux for what programs use; a note says where support is partial, where a call is accepted
and has no effect, or where it always fails.

## Calls that are not listed

Every other number returns `-ENOSYS` at once. Nothing else happens: no signal and no
diagnostic. To see such calls, run the node with `--trace`, which writes one line per syscall
to stderr ([design/node.md](design/node.md)). The lines have this form (the calls shown are
examples):

```text
cargo run --release -p wasmshim-run -- --image target/images/alpine-busybox.tar \
    target/wasm32-wasip2/release/wasmshim_linux_node.wasm --trace /bin/busybox ls
...
[linux] [1] openat(0xffffffffffffff9c, 0x..., 0x..., 0x0) = -1 ENOENT
[linux] [1] rseq(0x..., 0x..., 0x..., 0x..., 0x..., 0x...) = -1 ENOSYS
```

A number that is not in the asm-generic table at all prints as `syscall_<n>`. With threads,
the label is `[pid:tid]`.

These named calls are not implemented (`ENOSYS`): `acct`, `add_key`, `adjtimex`, `bpf`,
`cachestat`, `clock_adjtime`, `clone3` (explicitly; C libraries fall back to `clone`),
`delete_module`, `fanotify_init`, `fanotify_mark`, `fchmodat2`, `finit_module`, `fsconfig`,
`fsmount`, `fsopen`, `fspick`, `futex_requeue`, `futex_wait`, `futex_waitv`, `futex_wake`,
`get_mempolicy`, `get_robust_list`, `init_module`, `io_uring_enter`, `io_uring_register`,
`io_uring_setup`, `ioprio_get`, `ioprio_set`, `kcmp`, `kexec_file_load`, `kexec_load`,
`keyctl`, `landlock_add_rule`, `landlock_create_ruleset`, `landlock_restrict_self`,
`listmount`, `lookup_dcookie`, `lsm_get_self_attr`, `lsm_list_modules`, `lsm_set_self_attr`,
`map_shadow_stack`, `mbind`, `memfd_secret`, `migrate_pages`, `mincore`, `mount`,
`mount_setattr`, `move_mount`, `move_pages`, `mseal`, `name_to_handle_at`, `nfsservctl`,
`open_by_handle_at`, `open_tree`, `perf_event_open`, `personality`, `pivot_root`, `pkey_alloc`,
`pkey_free`, `pkey_mprotect`, `process_madvise`, `process_mrelease`, `process_vm_readv`,
`process_vm_writev`, `ptrace`, `quotactl`, `quotactl_fd`, `reboot`, `remap_file_pages`,
`request_key`, `restart_syscall`, `rseq`, `sched_getattr`, `sched_rr_get_interval`,
`sched_setattr`, `seccomp`, `set_mempolicy`, `set_mempolicy_home_node`, `setdomainname`,
`sethostname`, `setns`, `statmount`, `swapoff`, `swapon`, `syslog`, `umount2`, `unshare`,
`userfaultfd`, `vhangup`.

## Files and directories

| Call | Note |
|---|---|
| `openat` | `O_TMPFILE` is `EOPNOTSUPP`. Paths under `/proc` and `/sys` are generated, not read from the VFS. |
| `openat2` | |
| `close` | |
| `close_range` | `CLOSE_RANGE_CLOEXEC` honoured; other flags ignored. |
| `read`, `write`, `readv`, `writev` | |
| `pread64`, `pwrite64`, `preadv`, `pwritev` | |
| `preadv2`, `pwritev2` | `RWF_HIPRI`, `RWF_DSYNC`, `RWF_SYNC` (and `RWF_APPEND`/`RWF_NOAPPEND` for writes) accepted; any other flag, including `RWF_NOWAIT`, is `EOPNOTSUPP`. |
| `lseek` | |
| `dup`, `dup3` | |
| `fcntl` | `F_DUPFD`(`_CLOEXEC`), `F_GETFD`/`F_SETFD`, `F_GETFL`/`F_SETFL`, POSIX and OFD locks, `F_SETOWN`/`F_GETOWN` (recorded; `SIGIO` is never sent), `F_ADD_SEALS`/`F_GET_SEALS`, `F_GETPIPE_SZ` (65536 on a pipe, `EBADF` otherwise). Anything else (`F_SETPIPE_SZ`, leases, `F_NOTIFY`) is `EINVAL`. |
| `ioctl` | `FIOCLEX`, `FIONCLEX`, `FIOASYNC`, `FIONBIO`, `FIONREAD`; terminal requests on terminals; hidraw and framebuffer requests on those devices. Anything else is `ENOTTY`, as is every terminal request on host stdio. |
| `flock` | |
| `pipe2` | A blocking write larger than `PIPE_BUF` (4096) returns a short count instead of blocking. |
| `getdents64` | |
| `newfstatat`, `fstat`, `statx` | |
| `statfs`, `fstatfs` | Fixed tmpfs-like values for every file; the path must resolve. |
| `readlinkat` | |
| `faccessat`, `faccessat2` | The caller is root; only the execute bit is checked. |
| `mkdirat` | |
| `mknodat` | Regular files and FIFOs (a FIFO cannot be opened yet: `ENXIO`); device and socket nodes are `EPERM`. |
| `unlinkat` | |
| `renameat` | |
| `renameat2` | `RENAME_NOREPLACE` only; `RENAME_EXCHANGE` and `RENAME_WHITEOUT` are `EINVAL`. |
| `symlinkat`, `linkat` | |
| `fchmod`, `fchmodat`, `fchown`, `fchownat` | Recorded and reported, never enforced. |
| `utimensat` | |
| `truncate`, `ftruncate` | Files are at most 1 GiB and are allocated when grown. |
| `fallocate` | Allocation (mode 0, `FALLOC_FL_KEEP_SIZE`) and hole punching; other modes are `EOPNOTSUPP`. |
| `setxattr`, `lsetxattr`, `fsetxattr` | `user.`, `trusted.` and `security.` names, kept per inode. `EOPNOTSUPP` on `/proc` files and non-VFS objects. |
| `getxattr`, `lgetxattr`, `fgetxattr` | As above. |
| `listxattr`, `llistxattr`, `flistxattr` | As above. |
| `removexattr`, `lremovexattr`, `fremovexattr` | As above. |
| `copy_file_range` | Regular files. |
| `splice`, `tee`, `vmsplice` | |
| `sendfile` | |
| `readahead` | Checks its descriptor; no other effect. |
| `sync`, `syncfs` | Return 0; there is nothing to write back. |
| `fsync`, `fdatasync`, `fadvise64` | Check the descriptor, then return 0. |
| `sync_file_range` | Checks its arguments, then returns 0. |
| `getcwd`, `chdir`, `fchdir`, `umask` | |
| `chroot` | Always `EPERM`. |
| `inotify_init1`, `inotify_add_watch`, `inotify_rm_watch` | Events synthesised from the VFS, delivered before the writing call returns. |
| `io_setup`, `io_destroy`, `io_submit` | Native AIO: every request completes at submission. |
| `io_getevents`, `io_pgetevents` | |
| `io_cancel` | Always `EINVAL` (nothing is ever in flight). |

## Memory

| Call | Note |
|---|---|
| `brk` | |
| `mmap` | Deterministic top-down placement, no randomisation. `MAP_SHARED` mappings share pages with the file's `read`/`write`. |
| `munmap`, `mprotect` | |
| `mremap` | `MREMAP_FIXED` and `MREMAP_DONTUNMAP` are `EINVAL`. |
| `madvise` | `MADV_DONTNEED`/`MADV_FREE` zero private pages, including private file mappings (the file contents are not restored); `MADV_REMOVE` punches shared mappings; other valid advice returns 0 without effect, unknown advice is `EINVAL`. |
| `msync` | Checks its arguments; shared mappings are their objects, so there is nothing to write. |
| `mlock`, `mlock2`, `munlock`, `mlockall`, `munlockall` | Return 0 without effect. |
| `membarrier` | Returns 0 without effect (one CPU). |
| `memfd_create` | Seals supported. |

## Processes and identity

| Call | Note |
|---|---|
| `clone` | Threads, `fork`, `vfork`. Namespace flags and `CLONE_PIDFD` are `EINVAL`. No process or thread limit. |
| `execve`, `execveat` | ELF64 AArch64 (static, PIE, dynamic); `#!` up to four levels. |
| `exit`, `exit_group` | |
| `wait4`, `waitid` | |
| `set_tid_address` | |
| `set_robust_list` | Returns 0; no robust list is walked at thread exit. |
| `getpid`, `gettid`, `getppid` | |
| `getpgid`, `setpgid`, `getsid`, `setsid` | |
| `getuid`, `geteuid`, `getgid`, `getegid` | |
| `getresuid`, `getresgid` | |
| `setuid`, `setgid`, `setreuid`, `setregid`, `setresuid`, `setresgid`, `setfsuid`, `setfsgid` | Linux's rules for who may change what (`EPERM` included); never enforced on files. |
| `getgroups`, `setgroups` | |
| `capget` | Root's capabilities: effective while the effective uid is 0, permitted while any of the real, effective or saved uids is 0. |
| `capset` | Returns 0 without effect. |
| `prctl` | `PR_SET_NAME`, `PR_GET_NAME`, `PR_GET_DUMPABLE` (1); `PR_SET_PDEATHSIG` and `PR_SET_DUMPABLE` accepted without effect; anything else is `EINVAL`. |
| `prlimit64`, `getrlimit` | Fixed values (`RLIMIT_NOFILE` 1024, `RLIMIT_NPROC` 4096, `RLIMIT_STACK` 8 MiB, `RLIMIT_CORE` 0, others unlimited); a new limit is ignored. |
| `setrlimit` | Returns 0 without effect. |
| `getrusage` | Only `ru_utime`, of the calling thread, whatever `who` says. |
| `sched_yield` | Returns 0; threads switch on instruction quanta. |
| `sched_getaffinity` | One CPU. |
| `sched_setaffinity` | Returns 0 without effect. |
| `sched_getscheduler`, `sched_getparam` | `SCHED_OTHER`, priority 0. |
| `sched_setscheduler`, `sched_setparam` | Return 0 without effect. |
| `sched_get_priority_max`, `sched_get_priority_min` | 0 for `SCHED_OTHER`/`BATCH`/`IDLE`, 99 and 1 for `FIFO`/`RR`. |
| `getcpu` | CPU 0, node 0. |
| `getpriority` | Always 20 (nice 0). |
| `setpriority` | Returns 0 without effect. |
| `pidfd_open`, `pidfd_send_signal`, `pidfd_getfd` | |

## Signals

| Call | Note |
|---|---|
| `rt_sigaction`, `rt_sigprocmask`, `rt_sigreturn`, `rt_sigsuspend`, `rt_sigpending` | `SIGSTOP`, `SIGTSTP`, `SIGTTIN` and `SIGTTOU` are ignored by default (no job control). |
| `rt_sigtimedwait` | |
| `rt_sigqueueinfo`, `rt_tgsigqueueinfo` | |
| `kill`, `tkill`, `tgkill` | Pid 1 gets every signal sent to it, also those it has no handler for (Linux discards those for init). |
| `sigaltstack` | `SS_AUTODISARM` is `EINVAL`. |
| `signalfd4` | |

## Time and timers

| Call | Note |
|---|---|
| `clock_gettime` | `CLOCK_REALTIME`, `_COARSE`, `_ALARM` and `CLOCK_TAI` read the host's wall clock; the `MONOTONIC` and `BOOTTIME` clocks read its monotonic clock; CPU-time clocks are retired instructions times `ns_per_insn`. |
| `clock_getres` | 1 ns for every clock. |
| `clock_settime`, `settimeofday` | `EPERM` (the clocks belong to the host). |
| `gettimeofday` | |
| `nanosleep` | |
| `clock_nanosleep` | Sleeping on a CPU-time clock is `EOPNOTSUPP`. An absolute realtime deadline is converted to the monotonic clock once. |
| `times` | Only `tms_utime`, counted for the whole machine. |
| `getitimer`, `setitimer` | `ITIMER_REAL` fires; `ITIMER_VIRTUAL` and `ITIMER_PROF` are stored but never fire. |
| `timer_create`, `timer_settime`, `timer_gettime`, `timer_getoverrun`, `timer_delete` | CPU-time timers are checked between slices, so they fire within one quantum. |
| `timerfd_create`, `timerfd_settime`, `timerfd_gettime` | |

## Sockets

| Call | Note |
|---|---|
| `socket` | `AF_UNIX` (stream, datagram, seqpacket); `AF_INET`/`AF_INET6` (stream, datagram, ICMP echo datagram); `AF_NETLINK` (`NETLINK_ROUTE` only). `SOCK_RAW` is `EPERM`; other families `EAFNOSUPPORT`. Non-loopback traffic needs a network provider, else `ENETUNREACH`. |
| `socketpair` | `AF_UNIX` only (`EOPNOTSUPP` otherwise). |
| `bind`, `connect`, `accept`, `accept4` | |
| `listen` | `EOPNOTSUPP` on datagram sockets. |
| `getsockname`, `getpeername` | |
| `sendto`, `sendmmsg` | |
| `sendmsg` | Ancillary data: `SCM_RIGHTS` on `AF_UNIX`; anything else is `EOPNOTSUPP`. |
| `recvfrom` | |
| `recvmsg`, `recvmmsg` | |
| `shutdown`, `getsockopt` | |
| `setsockopt` | `SO_RCVTIMEO`/`SO_SNDTIMEO` honoured, integer options remembered for `getsockopt`. A level the socket has no options at is refused (`EOPNOTSUPP` on `AF_UNIX`, `ENOPROTOOPT` otherwise); at the other levels every option is accepted, on local sockets without effect (multicast joins included). |

## Polling and event objects

| Call | Note |
|---|---|
| `ppoll` | |
| `pselect6` | No exceptional conditions are reported (`exceptfds` comes back empty); `nfds` at most 1024. |
| `epoll_create1` | |
| `epoll_ctl` | Regular files are `EPERM`. |
| `epoll_pwait` | |
| `epoll_pwait2` | |
| `eventfd2` | |

## IPC

| Call | Note |
|---|---|
| `futex` | `FUTEX_WAIT`, `FUTEX_WAKE`, their `_BITSET` forms, `FUTEX_REQUEUE`, `FUTEX_CMP_REQUEUE`; private and shared alike. Priority-inheritance operations and `FUTEX_WAKE_OP` are `ENOSYS`. |
| `shmget`, `shmat`, `shmdt`, `shmctl` | |
| `semget`, `semop`, `semtimedop`, `semctl` | `SEM_UNDO` adjustments applied at exit. |
| `msgget`, `msgsnd`, `msgrcv`, `msgctl` | |
| `mq_open`, `mq_unlink`, `mq_timedsend`, `mq_timedreceive`, `mq_notify`, `mq_getsetattr` | |

## Miscellaneous

| Call | Note |
|---|---|
| `getrandom` | From the host's (seeded) entropy; never blocks; at most 16 MiB per call. |
| `uname` | `sysname` `Linux`; node name, release and version from the configuration; machine `aarch64`. |
| `sysinfo` | Uptime from the monotonic clock; fixed 4 GiB total and 3 GiB free memory. |
