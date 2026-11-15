//! Flag and constant values (asm-generic / AArch64 values).

#![allow(missing_docs)]

/// Auxiliary vector keys.
pub mod auxv {
    pub const AT_NULL: u64 = 0;
    pub const AT_IGNORE: u64 = 1;
    pub const AT_PHDR: u64 = 3;
    pub const AT_PHENT: u64 = 4;
    pub const AT_PHNUM: u64 = 5;
    pub const AT_PAGESZ: u64 = 6;
    pub const AT_BASE: u64 = 7;
    pub const AT_FLAGS: u64 = 8;
    pub const AT_ENTRY: u64 = 9;
    pub const AT_UID: u64 = 11;
    pub const AT_EUID: u64 = 12;
    pub const AT_GID: u64 = 13;
    pub const AT_EGID: u64 = 14;
    pub const AT_PLATFORM: u64 = 15;
    pub const AT_HWCAP: u64 = 16;
    pub const AT_CLKTCK: u64 = 17;
    pub const AT_SECURE: u64 = 23;
    pub const AT_RANDOM: u64 = 25;
    pub const AT_HWCAP2: u64 = 26;
    pub const AT_EXECFN: u64 = 31;
    pub const AT_MINSIGSTKSZ: u64 = 51;
}

/// AArch64 `AT_HWCAP` bits.
pub mod hwcap {
    pub const FP: u64 = 1 << 0;
    pub const ASIMD: u64 = 1 << 1;
    pub const EVTSTRM: u64 = 1 << 2;
    pub const AES: u64 = 1 << 3;
    pub const PMULL: u64 = 1 << 4;
    pub const SHA1: u64 = 1 << 5;
    pub const SHA2: u64 = 1 << 6;
    pub const CRC32: u64 = 1 << 7;
    pub const ATOMICS: u64 = 1 << 8;
    pub const FPHP: u64 = 1 << 9;
    pub const ASIMDHP: u64 = 1 << 10;
    pub const CPUID: u64 = 1 << 11;
}

/// `mmap`/`mprotect` protection bits.
pub mod prot {
    pub const PROT_NONE: u64 = 0;
    pub const PROT_READ: u64 = 1;
    pub const PROT_WRITE: u64 = 2;
    pub const PROT_EXEC: u64 = 4;
}

/// `mmap` flags.
pub mod map {
    pub const MAP_SHARED: u64 = 0x01;
    pub const MAP_PRIVATE: u64 = 0x02;
    pub const MAP_SHARED_VALIDATE: u64 = 0x03;
    pub const MAP_TYPE: u64 = 0x0f;
    pub const MAP_FIXED: u64 = 0x10;
    pub const MAP_ANONYMOUS: u64 = 0x20;
    pub const MAP_GROWSDOWN: u64 = 0x0100;
    pub const MAP_DENYWRITE: u64 = 0x0800;
    pub const MAP_NORESERVE: u64 = 0x4000;
    pub const MAP_POPULATE: u64 = 0x8000;
    pub const MAP_STACK: u64 = 0x20000;
    pub const MAP_FIXED_NOREPLACE: u64 = 0x100000;
}

/// `msync` flags.
pub mod msync {
    pub const MS_ASYNC: u64 = 1;
    pub const MS_INVALIDATE: u64 = 2;
    pub const MS_SYNC: u64 = 4;
}

/// `mremap` flags.
pub mod mremap {
    pub const MREMAP_MAYMOVE: u64 = 1;
    pub const MREMAP_FIXED: u64 = 2;
}

/// `open` flags (AArch64 values: `O_DIRECTORY`/`O_NOFOLLOW`/`O_DIRECT` differ from x86).
pub mod open {
    pub const O_ACCMODE: u64 = 0o3;
    pub const O_RDONLY: u64 = 0o0;
    pub const O_WRONLY: u64 = 0o1;
    pub const O_RDWR: u64 = 0o2;
    pub const O_CREAT: u64 = 0o100;
    pub const O_EXCL: u64 = 0o200;
    pub const O_NOCTTY: u64 = 0o400;
    pub const O_TRUNC: u64 = 0o1000;
    pub const O_APPEND: u64 = 0o2000;
    pub const O_NONBLOCK: u64 = 0o4000;
    /// Signal-driven I/O (`FIOASYNC`, `F_SETFL`).
    pub const O_ASYNC: u64 = 0o20000;
    pub const O_DSYNC: u64 = 0o10000;
    pub const O_DIRECTORY: u64 = 0o40000;
    pub const O_NOFOLLOW: u64 = 0o100000;
    pub const O_DIRECT: u64 = 0o200000;
    pub const O_LARGEFILE: u64 = 0o400000;
    pub const O_NOATIME: u64 = 0o1000000;
    pub const O_CLOEXEC: u64 = 0o2000000;
    pub const O_SYNC: u64 = 0o4010000;
    pub const O_PATH: u64 = 0o10000000;
    pub const O_TMPFILE: u64 = 0o20000000 | O_DIRECTORY;
}

/// `*at` syscall conventions.
pub mod at {
    /// "Relative to the current directory" (`-100` as a signed fd).
    pub const AT_FDCWD: i32 = -100;
    pub const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
    pub const AT_REMOVEDIR: u64 = 0x200;
    pub const AT_EACCESS: u64 = 0x200;
    pub const AT_SYMLINK_FOLLOW: u64 = 0x400;
    pub const AT_NO_AUTOMOUNT: u64 = 0x800;
    pub const AT_EMPTY_PATH: u64 = 0x1000;
    pub const AT_STATX_SYNC_TYPE: u64 = 0x6000;
}

/// `access` modes.
pub mod access {
    pub const F_OK: u64 = 0;
    pub const X_OK: u64 = 1;
    pub const W_OK: u64 = 2;
    pub const R_OK: u64 = 4;
}

/// `fcntl` commands and fd flags.
pub mod fcntl {
    pub const F_DUPFD: u64 = 0;
    pub const F_GETFD: u64 = 1;
    pub const F_SETFD: u64 = 2;
    pub const F_GETFL: u64 = 3;
    pub const F_SETFL: u64 = 4;
    pub const F_GETLK: u64 = 5;
    pub const F_SETLK: u64 = 6;
    pub const F_SETLKW: u64 = 7;
    pub const F_SETOWN: u64 = 8;
    pub const F_GETOWN: u64 = 9;
    pub const F_DUPFD_CLOEXEC: u64 = 1030;
    pub const F_GETPIPE_SZ: u64 = 1032;
    pub const F_ADD_SEALS: u64 = 1033;
    pub const F_GET_SEALS: u64 = 1034;
    pub const FD_CLOEXEC: u64 = 1;
}

/// File seals (`F_ADD_SEALS`, `memfd_create`).
pub mod seal {
    pub const F_SEAL_SEAL: u32 = 0x1;
    pub const F_SEAL_SHRINK: u32 = 0x2;
    pub const F_SEAL_GROW: u32 = 0x4;
    pub const F_SEAL_WRITE: u32 = 0x8;
    pub const F_SEAL_FUTURE_WRITE: u32 = 0x10;
    pub const F_SEAL_EXEC: u32 = 0x20;
    pub const ALL: u32 = 0x3f;
}

/// `memfd_create` flags.
pub mod memfd {
    pub const MFD_CLOEXEC: u64 = 0x1;
    pub const MFD_ALLOW_SEALING: u64 = 0x2;
    pub const MFD_HUGETLB: u64 = 0x4;
    pub const MFD_NOEXEC_SEAL: u64 = 0x8;
    pub const MFD_EXEC: u64 = 0x10;
    /// Longest name (`NAME_MAX` minus `"memfd:"`).
    pub const NAME_MAX: usize = 249;
}

/// SysV IPC (`shmget`, ...).
pub mod ipc {
    pub const IPC_CREAT: u64 = 0o1000;
    pub const IPC_EXCL: u64 = 0o2000;
    pub const IPC_NOWAIT: u64 = 0o4000;
    pub const IPC_RMID: u64 = 0;
    pub const IPC_SET: u64 = 1;
    pub const IPC_STAT: u64 = 2;
    pub const IPC_INFO: u64 = 3;
    /// `cmd` flag of the 64-bit structures (Linux's default on arm64; accepted and ignored).
    pub const IPC_64: u64 = 0x100;
    pub const SHM_LOCK: u64 = 11;
    pub const SHM_UNLOCK: u64 = 12;
    pub const SHM_STAT: u64 = 13;
    pub const SHM_INFO: u64 = 14;
    pub const SHM_STAT_ANY: u64 = 15;
    pub const SHM_RDONLY: u64 = 0o10000;
    pub const SHM_RND: u64 = 0o20000;
    pub const SHM_REMAP: u64 = 0o40000;
    pub const SHM_EXEC: u64 = 0o100000;
    /// `SHMLBA` on arm64 (the page size).
    pub const SHMLBA: u64 = 4096;
}

/// File type and permission bits of `st_mode`.
pub mod mode {
    pub const S_IFMT: u32 = 0o170000;
    pub const S_IFSOCK: u32 = 0o140000;
    pub const S_IFLNK: u32 = 0o120000;
    pub const S_IFREG: u32 = 0o100000;
    pub const S_IFBLK: u32 = 0o060000;
    pub const S_IFDIR: u32 = 0o040000;
    pub const S_IFCHR: u32 = 0o020000;
    pub const S_IFIFO: u32 = 0o010000;
    pub const S_ISUID: u32 = 0o4000;
    pub const S_ISGID: u32 = 0o2000;
    pub const S_ISVTX: u32 = 0o1000;
}

/// `d_type` values in `getdents64` records.
pub mod dtype {
    pub const DT_UNKNOWN: u8 = 0;
    pub const DT_FIFO: u8 = 1;
    pub const DT_CHR: u8 = 2;
    pub const DT_DIR: u8 = 4;
    pub const DT_BLK: u8 = 6;
    pub const DT_REG: u8 = 8;
    pub const DT_LNK: u8 = 10;
    pub const DT_SOCK: u8 = 12;
}

/// `lseek` whence values.
pub mod seek {
    pub const SEEK_SET: u64 = 0;
    pub const SEEK_CUR: u64 = 1;
    pub const SEEK_END: u64 = 2;
}

/// Clock ids.
pub mod clock {
    pub const CLOCK_REALTIME: u64 = 0;
    pub const CLOCK_MONOTONIC: u64 = 1;
    pub const CLOCK_PROCESS_CPUTIME_ID: u64 = 2;
    pub const CLOCK_THREAD_CPUTIME_ID: u64 = 3;
    pub const CLOCK_MONOTONIC_RAW: u64 = 4;
    pub const CLOCK_REALTIME_COARSE: u64 = 5;
    pub const CLOCK_MONOTONIC_COARSE: u64 = 6;
    pub const CLOCK_BOOTTIME: u64 = 7;
    pub const CLOCK_REALTIME_ALARM: u64 = 8;
    pub const CLOCK_BOOTTIME_ALARM: u64 = 9;
    pub const CLOCK_TAI: u64 = 11;
    /// `TIMER_ABSTIME` for `clock_nanosleep`.
    pub const TIMER_ABSTIME: u64 = 1;
}

/// Signal numbers, dispositions and flags.
pub mod signal {
    pub const SIGHUP: u32 = 1;
    pub const SIGINT: u32 = 2;
    pub const SIGQUIT: u32 = 3;
    pub const SIGILL: u32 = 4;
    pub const SIGTRAP: u32 = 5;
    pub const SIGABRT: u32 = 6;
    pub const SIGBUS: u32 = 7;
    pub const SIGFPE: u32 = 8;
    pub const SIGKILL: u32 = 9;
    pub const SIGUSR1: u32 = 10;
    pub const SIGSEGV: u32 = 11;
    pub const SIGUSR2: u32 = 12;
    pub const SIGPIPE: u32 = 13;
    pub const SIGALRM: u32 = 14;
    pub const SIGTERM: u32 = 15;
    pub const SIGSTKFLT: u32 = 16;
    pub const SIGCHLD: u32 = 17;
    pub const SIGCONT: u32 = 18;
    pub const SIGSTOP: u32 = 19;
    pub const SIGTSTP: u32 = 20;
    pub const SIGTTIN: u32 = 21;
    pub const SIGTTOU: u32 = 22;
    pub const SIGURG: u32 = 23;
    pub const SIGXCPU: u32 = 24;
    pub const SIGXFSZ: u32 = 25;
    pub const SIGVTALRM: u32 = 26;
    pub const SIGPROF: u32 = 27;
    pub const SIGWINCH: u32 = 28;
    pub const SIGIO: u32 = 29;
    pub const SIGPWR: u32 = 30;
    pub const SIGSYS: u32 = 31;
    /// Highest signal number.
    pub const NSIG: u32 = 64;
    /// The first real-time signal (the kernel's; libcs reserve a few above it).
    pub const SIGRTMIN: u32 = 32;
    pub const SIG_DFL: u64 = 0;
    pub const SIG_IGN: u64 = 1;
    pub const SIG_BLOCK: u64 = 0;
    pub const SIG_UNBLOCK: u64 = 1;
    pub const SIG_SETMASK: u64 = 2;
    pub const SA_NOCLDSTOP: u64 = 0x1;
    pub const SA_NOCLDWAIT: u64 = 0x2;
    pub const SA_SIGINFO: u64 = 0x4;
    pub const SA_RESTORER: u64 = 0x0400_0000;
    pub const SA_ONSTACK: u64 = 0x0800_0000;
    pub const SA_RESTART: u64 = 0x1000_0000;
    pub const SA_NODEFER: u64 = 0x4000_0000;
    pub const SA_RESETHAND: u64 = 0x8000_0000;
    pub const SS_ONSTACK: u32 = 1;
    pub const SS_DISABLE: u32 = 2;
    /// The smallest alternate signal stack `sigaltstack` takes (arm64's
    /// `arch/arm64/include/uapi/asm/signal.h`).
    pub const MINSIGSTKSZ: u64 = 5120;
    /// `si_code` values.
    pub const SI_USER: i32 = 0;
    pub const SI_KERNEL: i32 = 0x80;
    pub const SI_QUEUE: i32 = -1;
    pub const SI_TIMER: i32 = -2;
    pub const SI_MESGQ: i32 = -3;
    pub const SI_TKILL: i32 = -6;
    pub const SEGV_MAPERR: i32 = 1;
    pub const SEGV_ACCERR: i32 = 2;
    pub const BUS_ADRALN: i32 = 1;
    pub const BUS_ADRERR: i32 = 2;
    pub const ILL_ILLOPC: i32 = 1;
    pub const TRAP_BRKPT: i32 = 1;
    pub const CLD_EXITED: i32 = 1;
    pub const CLD_KILLED: i32 = 2;
    pub const CLD_DUMPED: i32 = 3;

    /// The conventional name (`"SIGSEGV"`), or `"SIGRT"` for real-time signals.
    pub const fn name(sig: u32) -> &'static str {
        const NAMES: [&str; 32] = [
            "0",
            "SIGHUP",
            "SIGINT",
            "SIGQUIT",
            "SIGILL",
            "SIGTRAP",
            "SIGABRT",
            "SIGBUS",
            "SIGFPE",
            "SIGKILL",
            "SIGUSR1",
            "SIGSEGV",
            "SIGUSR2",
            "SIGPIPE",
            "SIGALRM",
            "SIGTERM",
            "SIGSTKFLT",
            "SIGCHLD",
            "SIGCONT",
            "SIGSTOP",
            "SIGTSTP",
            "SIGTTIN",
            "SIGTTOU",
            "SIGURG",
            "SIGXCPU",
            "SIGXFSZ",
            "SIGVTALRM",
            "SIGPROF",
            "SIGWINCH",
            "SIGIO",
            "SIGPWR",
            "SIGSYS",
        ];
        if (sig as usize) < NAMES.len() {
            NAMES[sig as usize]
        } else {
            "SIGRT"
        }
    }
}

/// TTY and generic ioctls (asm-generic numbers).
pub mod ioctl {
    pub const TCGETS: u64 = 0x5401;
    pub const TCSETS: u64 = 0x5402;
    pub const TCSETSW: u64 = 0x5403;
    pub const TCSETSF: u64 = 0x5404;
    pub const TCSBRK: u64 = 0x5409;
    pub const TCXONC: u64 = 0x540a;
    pub const TCFLSH: u64 = 0x540b;
    pub const TIOCEXCL: u64 = 0x540c;
    pub const TIOCNXCL: u64 = 0x540d;
    pub const TIOCSCTTY: u64 = 0x540e;
    pub const TIOCGPGRP: u64 = 0x540f;
    pub const TIOCSPGRP: u64 = 0x5410;
    pub const TIOCOUTQ: u64 = 0x5411;
    pub const TIOCSTI: u64 = 0x5412;
    pub const TIOCGWINSZ: u64 = 0x5413;
    pub const TIOCSWINSZ: u64 = 0x5414;
    pub const FIONREAD: u64 = 0x541b;
    pub const TIOCCONS: u64 = 0x541d;
    pub const TIOCPKT: u64 = 0x5420;
    pub const FIONBIO: u64 = 0x5421;
    pub const TIOCNOTTY: u64 = 0x5422;
    pub const TIOCSETD: u64 = 0x5423;
    pub const TIOCGETD: u64 = 0x5424;
    pub const TCSBRKP: u64 = 0x5425;
    pub const TIOCGSID: u64 = 0x5429;
    /// `_IOR('T', 0x2a, struct termios2)`.
    pub const TCGETS2: u64 = 0x802c_542a;
    /// `_IOW('T', 0x2b, struct termios2)`.
    pub const TCSETS2: u64 = 0x402c_542b;
    pub const TCSETSW2: u64 = 0x402c_542c;
    pub const TCSETSF2: u64 = 0x402c_542d;
    /// `_IOR('T', 0x30, unsigned int)`: the pty number of a master.
    pub const TIOCGPTN: u64 = 0x8004_5430;
    /// `_IOW('T', 0x31, int)`: lock (non-zero) or unlock the slave of a master.
    pub const TIOCSPTLCK: u64 = 0x4004_5431;
    /// `_IOR('T', 0x32, unsigned int)`: the device number of the terminal.
    pub const TIOCGDEV: u64 = 0x8004_5432;
    /// `_IOW('T', 0x36, int)`: a master sends a signal to the slave's foreground group.
    pub const TIOCSIG: u64 = 0x4004_5436;
    pub const TIOCVHANGUP: u64 = 0x5437;
    pub const TIOCGPKT: u64 = 0x8004_5438;
    pub const TIOCGPTLCK: u64 = 0x8004_5439;
    pub const TIOCGEXCL: u64 = 0x8004_5440;
    /// `_IO('T', 0x41)`: opens the slave of a master (flags in the argument).
    pub const TIOCGPTPEER: u64 = 0x5441;
    pub const FIONCLEX: u64 = 0x5450;
    pub const FIOCLEX: u64 = 0x5451;
    pub const FIOASYNC: u64 = 0x5452;
}

/// Terminal flags (`struct termios`, asm-generic; octal like the kernel headers).
pub mod termios {
    // c_iflag
    pub const IGNBRK: u32 = 0o1;
    pub const BRKINT: u32 = 0o2;
    pub const ISTRIP: u32 = 0o40;
    pub const INLCR: u32 = 0o100;
    pub const IGNCR: u32 = 0o200;
    pub const ICRNL: u32 = 0o400;
    pub const IUCLC: u32 = 0o1000;
    pub const IXON: u32 = 0o2000;
    pub const IXANY: u32 = 0o4000;
    pub const IXOFF: u32 = 0o10000;
    pub const IMAXBEL: u32 = 0o20000;
    pub const IUTF8: u32 = 0o40000;
    // c_oflag
    pub const OPOST: u32 = 0o1;
    pub const OLCUC: u32 = 0o2;
    pub const ONLCR: u32 = 0o4;
    pub const OCRNL: u32 = 0o10;
    pub const ONOCR: u32 = 0o20;
    pub const ONLRET: u32 = 0o40;
    pub const TABDLY: u32 = 0o14000;
    /// `TAB3`: expand tabs to spaces.
    pub const XTABS: u32 = 0o14000;
    // c_cflag
    pub const CBAUD: u32 = 0o10017;
    pub const B38400: u32 = 0o17;
    pub const CS8: u32 = 0o60;
    pub const CREAD: u32 = 0o200;
    pub const HUPCL: u32 = 0o2000;
    // c_lflag
    pub const ISIG: u32 = 0o1;
    pub const ICANON: u32 = 0o2;
    pub const ECHO: u32 = 0o10;
    pub const ECHOE: u32 = 0o20;
    pub const ECHOK: u32 = 0o40;
    pub const ECHONL: u32 = 0o100;
    pub const NOFLSH: u32 = 0o200;
    pub const TOSTOP: u32 = 0o400;
    pub const ECHOCTL: u32 = 0o1000;
    pub const ECHOPRT: u32 = 0o2000;
    pub const ECHOKE: u32 = 0o4000;
    pub const FLUSHO: u32 = 0o10000;
    pub const PENDIN: u32 = 0o40000;
    pub const IEXTEN: u32 = 0o100000;
    // c_cc indices
    pub const VINTR: usize = 0;
    pub const VQUIT: usize = 1;
    pub const VERASE: usize = 2;
    pub const VKILL: usize = 3;
    pub const VEOF: usize = 4;
    pub const VTIME: usize = 5;
    pub const VMIN: usize = 6;
    pub const VSTART: usize = 8;
    pub const VSTOP: usize = 9;
    pub const VSUSP: usize = 10;
    pub const VEOL: usize = 11;
    pub const VREPRINT: usize = 12;
    pub const VDISCARD: usize = 13;
    pub const VWERASE: usize = 14;
    pub const VLNEXT: usize = 15;
    pub const VEOL2: usize = 16;
    // tcflush / tcflow
    pub const TCIFLUSH: u64 = 0;
    pub const TCOFLUSH: u64 = 1;
    pub const TCIOFLUSH: u64 = 2;
    // The line discipline (`TIOCGETD`).
    pub const N_TTY: u32 = 0;
}

/// `getrandom` flags.
pub mod random {
    pub const GRND_NONBLOCK: u64 = 1;
    pub const GRND_RANDOM: u64 = 2;
    pub const GRND_INSECURE: u64 = 4;
}

/// Resource limits.
pub mod rlimit {
    pub const RLIMIT_CPU: u64 = 0;
    pub const RLIMIT_FSIZE: u64 = 1;
    pub const RLIMIT_DATA: u64 = 2;
    pub const RLIMIT_STACK: u64 = 3;
    pub const RLIMIT_CORE: u64 = 4;
    pub const RLIMIT_RSS: u64 = 5;
    pub const RLIMIT_NPROC: u64 = 6;
    pub const RLIMIT_NOFILE: u64 = 7;
    pub const RLIMIT_MEMLOCK: u64 = 8;
    pub const RLIMIT_AS: u64 = 9;
    pub const RLIM_INFINITY: u64 = u64::MAX;
}

/// `clone` flags.
pub mod clone {
    pub const CSIGNAL: u64 = 0x0000_00ff;
    pub const CLONE_VM: u64 = 0x0000_0100;
    pub const CLONE_FS: u64 = 0x0000_0200;
    pub const CLONE_FILES: u64 = 0x0000_0400;
    pub const CLONE_SIGHAND: u64 = 0x0000_0800;
    pub const CLONE_PIDFD: u64 = 0x0000_1000;
    pub const CLONE_PTRACE: u64 = 0x0000_2000;
    pub const CLONE_VFORK: u64 = 0x0000_4000;
    pub const CLONE_PARENT: u64 = 0x0000_8000;
    pub const CLONE_THREAD: u64 = 0x0001_0000;
    pub const CLONE_NEWNS: u64 = 0x0002_0000;
    pub const CLONE_SYSVSEM: u64 = 0x0004_0000;
    pub const CLONE_SETTLS: u64 = 0x0008_0000;
    pub const CLONE_PARENT_SETTID: u64 = 0x0010_0000;
    pub const CLONE_CHILD_CLEARTID: u64 = 0x0020_0000;
    pub const CLONE_DETACHED: u64 = 0x0040_0000;
    pub const CLONE_UNTRACED: u64 = 0x0080_0000;
    pub const CLONE_CHILD_SETTID: u64 = 0x0100_0000;
}

/// `futex` operations.
pub mod futex {
    pub const FUTEX_WAIT: u64 = 0;
    pub const FUTEX_WAKE: u64 = 1;
    pub const FUTEX_REQUEUE: u64 = 3;
    pub const FUTEX_CMP_REQUEUE: u64 = 4;
    pub const FUTEX_WAKE_OP: u64 = 5;
    pub const FUTEX_WAIT_BITSET: u64 = 9;
    pub const FUTEX_WAKE_BITSET: u64 = 10;
    pub const FUTEX_PRIVATE_FLAG: u64 = 128;
    pub const FUTEX_CLOCK_REALTIME: u64 = 256;
    pub const FUTEX_CMD_MASK: u64 = !(FUTEX_PRIVATE_FLAG | FUTEX_CLOCK_REALTIME);
}

/// `wait4`/`waitid` options.
pub mod wait {
    pub const WNOHANG: u64 = 1;
    pub const WUNTRACED: u64 = 2;
    /// `waitid` spelling of `WUNTRACED`.
    pub const WSTOPPED: u64 = 2;
    pub const WEXITED: u64 = 4;
    pub const WCONTINUED: u64 = 8;
    pub const WNOWAIT: u64 = 0x0100_0000;
    pub const WALL: u64 = 0x4000_0000;
    pub const WCLONE: u64 = 0x8000_0000;
    pub const WNOTHREAD: u64 = 0x2000_0000;
    /// `waitid` id types.
    pub const P_ALL: u64 = 0;
    pub const P_PID: u64 = 1;
    pub const P_PGID: u64 = 2;
    pub const P_PIDFD: u64 = 3;
}

/// `madvise` advice values.
pub mod madvise {
    pub const MADV_NORMAL: u64 = 0;
    pub const MADV_RANDOM: u64 = 1;
    pub const MADV_SEQUENTIAL: u64 = 2;
    pub const MADV_WILLNEED: u64 = 3;
    pub const MADV_DONTNEED: u64 = 4;
    pub const MADV_FREE: u64 = 8;
    pub const MADV_REMOVE: u64 = 9;
}

/// `prctl` options.
pub mod prctl {
    pub const PR_SET_PDEATHSIG: u64 = 1;
    pub const PR_GET_DUMPABLE: u64 = 3;
    pub const PR_SET_DUMPABLE: u64 = 4;
    pub const PR_SET_NAME: u64 = 15;
    pub const PR_GET_NAME: u64 = 16;
}

/// `preadv2`/`pwritev2` flags.
pub mod rwf {
    pub const RWF_HIPRI: u64 = 0x01;
    pub const RWF_DSYNC: u64 = 0x02;
    pub const RWF_SYNC: u64 = 0x04;
    pub const RWF_NOWAIT: u64 = 0x08;
    pub const RWF_APPEND: u64 = 0x10;
    pub const RWF_NOAPPEND: u64 = 0x20;
}
