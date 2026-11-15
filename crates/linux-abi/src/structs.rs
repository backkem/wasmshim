#![allow(missing_docs)]

//! Struct layouts as seen by AArch64 user space (LP64, little-endian).
//!
//! Each struct encodes to (and, where user space passes it in, decodes from) its exact byte
//! layout, so callers copy bytes to and from guest memory without `unsafe`.

/// Little-endian field writer over a fixed-size buffer.
struct Out<const N: usize>([u8; N]);

impl<const N: usize> Out<N> {
    const fn new() -> Self {
        Self([0; N])
    }
    fn u16(&mut self, at: usize, v: u16) -> &mut Self {
        self.0[at..at + 2].copy_from_slice(&v.to_le_bytes());
        self
    }
    fn u32(&mut self, at: usize, v: u32) -> &mut Self {
        self.0[at..at + 4].copy_from_slice(&v.to_le_bytes());
        self
    }
    fn u64(&mut self, at: usize, v: u64) -> &mut Self {
        self.0[at..at + 8].copy_from_slice(&v.to_le_bytes());
        self
    }
    fn bytes(&mut self, at: usize, v: &[u8]) -> &mut Self {
        self.0[at..at + v.len()].copy_from_slice(v);
        self
    }
}

fn get_u64(b: &[u8], at: usize) -> u64 {
    let mut x = [0; 8];
    x.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(x)
}

fn get_u32(b: &[u8], at: usize) -> u32 {
    let mut x = [0; 4];
    x.copy_from_slice(&b[at..at + 4]);
    u32::from_le_bytes(x)
}

fn get_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

/// `struct timespec`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timespec {
    pub sec: i64,
    pub nsec: i64,
}

impl Timespec {
    pub const SIZE: usize = 16;

    /// From nanoseconds.
    pub const fn from_nanos(ns: u64) -> Self {
        Self {
            sec: (ns / 1_000_000_000) as i64,
            nsec: (ns % 1_000_000_000) as i64,
        }
    }

    /// The largest time the kernel represents: `KTIME_MAX` nanoseconds (`i64::MAX`).
    pub const KTIME_MAX: u64 = i64::MAX as u64;

    /// To nanoseconds, or `None` if negative, `nsec` is out of range or the value does not fit.
    pub fn to_nanos(self) -> Option<u64> {
        if self.sec < 0 || self.nsec < 0 || self.nsec >= 1_000_000_000 {
            return None;
        }
        (self.sec as u64)
            .checked_mul(1_000_000_000)
            .and_then(|s| s.checked_add(self.nsec as u64))
    }

    /// To nanoseconds as Linux reads a timeout: `None` if negative or `nsec` is out of range
    /// (`timespec64_valid`), and a valid time past [`Self::KTIME_MAX`] is `KTIME_MAX`
    /// (`timespec64_to_ktime`, include/linux/ktime.h), so a huge timeout waits "forever".
    pub fn to_nanos_saturating(self) -> Option<u64> {
        if self.sec < 0 || self.nsec < 0 || self.nsec >= 1_000_000_000 {
            return None;
        }
        Some(
            (self.sec as u64)
                .saturating_mul(1_000_000_000)
                .saturating_add(self.nsec as u64)
                .min(Self::KTIME_MAX),
        )
    }

    pub fn to_bytes(self) -> [u8; Self::SIZE] {
        Out::new()
            .u64(0, self.sec as u64)
            .u64(8, self.nsec as u64)
            .0
    }

    pub fn from_bytes(b: &[u8; Self::SIZE]) -> Self {
        Self {
            sec: get_u64(b, 0) as i64,
            nsec: get_u64(b, 8) as i64,
        }
    }
}

/// `struct timeval`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timeval {
    pub sec: i64,
    pub usec: i64,
}

impl Timeval {
    pub const SIZE: usize = 16;

    pub const fn from_nanos(ns: u64) -> Self {
        Self {
            sec: (ns / 1_000_000_000) as i64,
            usec: ((ns % 1_000_000_000) / 1000) as i64,
        }
    }

    pub fn to_bytes(self) -> [u8; Self::SIZE] {
        Out::new()
            .u64(0, self.sec as u64)
            .u64(8, self.usec as u64)
            .0
    }
}

/// `struct new_utsname`: six 65-byte NUL-terminated fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Utsname<'a> {
    pub sysname: &'a str,
    pub nodename: &'a str,
    pub release: &'a str,
    pub version: &'a str,
    pub machine: &'a str,
    pub domainname: &'a str,
}

impl Utsname<'_> {
    pub const FIELD: usize = 65;
    pub const SIZE: usize = 6 * Self::FIELD;

    /// Encodes the struct; fields longer than 64 bytes are truncated.
    pub fn to_bytes(&self) -> [u8; 390] {
        let mut out = Out::<390>::new();
        let fields = [
            self.sysname,
            self.nodename,
            self.release,
            self.version,
            self.machine,
            self.domainname,
        ];
        for (i, f) in fields.iter().enumerate() {
            let b = f.as_bytes();
            out.bytes(i * Self::FIELD, &b[..b.len().min(Self::FIELD - 1)]);
        }
        out.0
    }
}

/// asm-generic `struct stat` (used by `fstat`/`newfstatat` on AArch64).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stat {
    pub dev: u64,
    pub ino: u64,
    pub mode: u32,
    pub nlink: u32,
    pub uid: u32,
    pub gid: u32,
    pub rdev: u64,
    pub size: i64,
    pub blksize: i32,
    pub blocks: i64,
    pub atime: Timespec,
    pub mtime: Timespec,
    pub ctime: Timespec,
}

impl Stat {
    pub const SIZE: usize = 128;

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        Out::new()
            .u64(0, self.dev)
            .u64(8, self.ino)
            .u32(16, self.mode)
            .u32(20, self.nlink)
            .u32(24, self.uid)
            .u32(28, self.gid)
            .u64(32, self.rdev)
            // 40: __pad1
            .u64(48, self.size as u64)
            .u32(56, self.blksize as u32)
            // 60: __pad2
            .u64(64, self.blocks as u64)
            .u64(72, self.atime.sec as u64)
            .u64(80, self.atime.nsec as u64)
            .u64(88, self.mtime.sec as u64)
            .u64(96, self.mtime.nsec as u64)
            .u64(104, self.ctime.sec as u64)
            .u64(112, self.ctime.nsec as u64)
            .0
    }
}

/// `struct statx` (256 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Statx {
    pub mask: u32,
    pub blksize: u32,
    pub attributes: u64,
    pub nlink: u32,
    pub uid: u32,
    pub gid: u32,
    pub mode: u16,
    pub ino: u64,
    pub size: u64,
    pub blocks: u64,
    pub attributes_mask: u64,
    pub atime: Timespec,
    pub btime: Timespec,
    pub ctime: Timespec,
    pub mtime: Timespec,
    pub rdev_major: u32,
    pub rdev_minor: u32,
    pub dev_major: u32,
    pub dev_minor: u32,
}

impl Statx {
    pub const SIZE: usize = 256;
    pub const STATX_TYPE: u32 = 0x1;
    pub const STATX_MODE: u32 = 0x2;
    pub const STATX_NLINK: u32 = 0x4;
    pub const STATX_UID: u32 = 0x8;
    pub const STATX_GID: u32 = 0x10;
    pub const STATX_ATIME: u32 = 0x20;
    pub const STATX_MTIME: u32 = 0x40;
    pub const STATX_CTIME: u32 = 0x80;
    pub const STATX_INO: u32 = 0x100;
    pub const STATX_SIZE: u32 = 0x200;
    pub const STATX_BLOCKS: u32 = 0x400;
    pub const STATX_BASIC_STATS: u32 = 0x7ff;
    pub const STATX_BTIME: u32 = 0x800;

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let ts = |out: &mut Out<256>, at: usize, t: Timespec| {
            out.u64(at, t.sec as u64).u32(at + 8, t.nsec as u32);
        };
        let mut out = Out::<256>::new();
        out.u32(0, self.mask)
            .u32(4, self.blksize)
            .u64(8, self.attributes)
            .u32(16, self.nlink)
            .u32(20, self.uid)
            .u32(24, self.gid)
            .u16(28, self.mode)
            .u64(32, self.ino)
            .u64(40, self.size)
            .u64(48, self.blocks)
            .u64(56, self.attributes_mask);
        ts(&mut out, 64, self.atime);
        ts(&mut out, 80, self.btime);
        ts(&mut out, 96, self.ctime);
        ts(&mut out, 112, self.mtime);
        out.u32(128, self.rdev_major)
            .u32(132, self.rdev_minor)
            .u32(136, self.dev_major)
            .u32(140, self.dev_minor);
        out.0
    }
}

/// `struct iovec`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Iovec {
    pub base: u64,
    pub len: u64,
}

impl Iovec {
    pub const SIZE: usize = 16;

    pub fn from_bytes(b: &[u8; Self::SIZE]) -> Self {
        Self {
            base: get_u64(b, 0),
            len: get_u64(b, 8),
        }
    }
}

/// `struct winsize`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Winsize {
    pub rows: u16,
    pub cols: u16,
    pub xpixel: u16,
    pub ypixel: u16,
}

impl Winsize {
    pub const SIZE: usize = 8;

    pub fn to_bytes(self) -> [u8; Self::SIZE] {
        Out::new()
            .u16(0, self.rows)
            .u16(2, self.cols)
            .u16(4, self.xpixel)
            .u16(6, self.ypixel)
            .0
    }

    pub fn from_bytes(b: &[u8; Self::SIZE]) -> Self {
        Self {
            rows: get_u16(b, 0),
            cols: get_u16(b, 2),
            xpixel: get_u16(b, 4),
            ypixel: get_u16(b, 6),
        }
    }
}

/// The kernel's `struct termios` (asm-generic, as used by `TCGETS`/`TCSETS`): 36 bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Termios {
    pub iflag: u32,
    pub oflag: u32,
    pub cflag: u32,
    pub lflag: u32,
    pub line: u8,
    pub cc: [u8; 19],
}

impl Termios {
    pub const SIZE: usize = 36;

    /// A cooked-mode terminal like a fresh Linux console (`stty sane`).
    pub const fn sane() -> Self {
        // ICRNL|IXON|IUTF8 ; OPOST|ONLCR ; B38400|CS8|CREAD|HUPCL ; ISIG|ICANON|ECHO|ECHOE|ECHOK|IEXTEN|ECHOCTL|ECHOKE
        let mut cc = [0u8; 19];
        cc[0] = 3; // VINTR ^C
        cc[1] = 28; // VQUIT ^\
        cc[2] = 127; // VERASE
        cc[3] = 21; // VKILL ^U
        cc[4] = 4; // VEOF ^D
        cc[6] = 1; // VMIN
        cc[8] = 17; // VSTART ^Q
        cc[9] = 19; // VSTOP ^S
        cc[10] = 26; // VSUSP ^Z
        cc[12] = 18; // VREPRINT ^R
        cc[13] = 15; // VDISCARD ^O
        cc[14] = 23; // VWERASE ^W
        cc[15] = 22; // VLNEXT ^V
        Self {
            iflag: 0o2400 | 0o40000,
            oflag: 0o5,
            cflag: 0o277,
            lflag: 0o105073,
            line: 0,
            cc,
        }
    }

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = Out::new();
        out.u32(0, self.iflag)
            .u32(4, self.oflag)
            .u32(8, self.cflag)
            .u32(12, self.lflag);
        out.0[16] = self.line;
        out.bytes(17, &self.cc);
        out.0
    }

    pub fn from_bytes(b: &[u8; Self::SIZE]) -> Self {
        let mut cc = [0u8; 19];
        cc.copy_from_slice(&b[17..36]);
        Self {
            iflag: get_u32(b, 0),
            oflag: get_u32(b, 4),
            cflag: get_u32(b, 8),
            lflag: get_u32(b, 12),
            line: b[16],
            cc,
        }
    }
}

/// The kernel's `struct sigaction` on AArch64 (`rt_sigaction`): handler, flags, restorer, mask.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelSigaction {
    pub handler: u64,
    pub flags: u64,
    pub restorer: u64,
    pub mask: u64,
}

impl KernelSigaction {
    pub const SIZE: usize = 32;

    pub fn to_bytes(self) -> [u8; Self::SIZE] {
        Out::new()
            .u64(0, self.handler)
            .u64(8, self.flags)
            .u64(16, self.restorer)
            .u64(24, self.mask)
            .0
    }

    pub fn from_bytes(b: &[u8; Self::SIZE]) -> Self {
        Self {
            handler: get_u64(b, 0),
            flags: get_u64(b, 8),
            restorer: get_u64(b, 16),
            mask: get_u64(b, 24),
        }
    }
}

/// `struct rlimit`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rlimit {
    pub cur: u64,
    pub max: u64,
}

impl Rlimit {
    pub const SIZE: usize = 16;

    pub fn to_bytes(self) -> [u8; Self::SIZE] {
        Out::new().u64(0, self.cur).u64(8, self.max).0
    }

    pub fn from_bytes(b: &[u8; Self::SIZE]) -> Self {
        Self {
            cur: get_u64(b, 0),
            max: get_u64(b, 8),
        }
    }
}

/// One `getdents64` record header: `d_ino`, `d_off`, `d_reclen`, `d_type`, then the name.
pub struct Dirent64;

impl Dirent64 {
    /// Header size before the name.
    pub const HEADER: usize = 19;

    /// The record length for a name of `name_len` bytes (NUL included, 8-byte aligned).
    pub const fn reclen(name_len: usize) -> usize {
        (Self::HEADER + name_len + 1 + 7) & !7
    }

    /// Encodes one record into `out` (which must be exactly `reclen(name.len())` long).
    pub fn encode(out: &mut [u8], ino: u64, off: u64, dtype: u8, name: &[u8]) {
        out.fill(0);
        out[0..8].copy_from_slice(&ino.to_le_bytes());
        out[8..16].copy_from_slice(&off.to_le_bytes());
        let reclen = out.len() as u16;
        out[16..18].copy_from_slice(&reclen.to_le_bytes());
        out[18] = dtype;
        out[19..19 + name.len()].copy_from_slice(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts() {
        let st = Stat {
            size: 5,
            mode: 0o100644,
            blksize: 4096,
            ..Stat::default()
        }
        .to_bytes();
        assert_eq!(&st[16..20], &0o100644u32.to_le_bytes());
        assert_eq!(&st[48..56], &5u64.to_le_bytes());
        let u = Utsname {
            sysname: "Linux",
            machine: "aarch64",
            ..Utsname::default()
        }
        .to_bytes();
        assert_eq!(&u[..6], b"Linux\0");
        assert_eq!(&u[4 * 65..4 * 65 + 8], b"aarch64\0");
        assert_eq!(Dirent64::reclen(1), 24);
        assert_eq!(Dirent64::reclen(5), 32);
        assert_eq!(
            Timespec::from_nanos(1_500_000_000),
            Timespec {
                sec: 1,
                nsec: 500_000_000
            }
        );
        let t = Termios::sane();
        assert_eq!(Termios::from_bytes(&t.to_bytes()), t);
    }
}
