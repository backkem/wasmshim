//! Error numbers (asm-generic, shared by every architecture we target).

macro_rules! errnos {
    ($($name:ident = $nr:literal,)*) => {
        /// A Linux error number. Syscalls return it negated.
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct Errno(pub u16);

        #[allow(missing_docs)]
        impl Errno {
            $(pub const $name: Self = Self($nr);)*

            /// The symbolic name (`"ENOENT"`), or `None` for an unknown number.
            pub const fn name(self) -> Option<&'static str> {
                match self.0 {
                    $($nr => Some(stringify!($name)),)*
                    _ => None,
                }
            }
        }
    };
}

errnos! {
    EPERM = 1, ENOENT = 2, ESRCH = 3, EINTR = 4, EIO = 5, ENXIO = 6, E2BIG = 7, ENOEXEC = 8,
    EBADF = 9, ECHILD = 10, EAGAIN = 11, ENOMEM = 12, EACCES = 13, EFAULT = 14, ENOTBLK = 15,
    EBUSY = 16, EEXIST = 17, EXDEV = 18, ENODEV = 19, ENOTDIR = 20, EISDIR = 21, EINVAL = 22,
    ENFILE = 23, EMFILE = 24, ENOTTY = 25, ETXTBSY = 26, EFBIG = 27, ENOSPC = 28, ESPIPE = 29,
    EROFS = 30, EMLINK = 31, EPIPE = 32, EDOM = 33, ERANGE = 34, EDEADLK = 35,
    ENAMETOOLONG = 36, ENOLCK = 37, ENOSYS = 38, ENOTEMPTY = 39, ELOOP = 40, ENOMSG = 42,
    EIDRM = 43, ENOSTR = 60, ENODATA = 61, ETIME = 62, ENOSR = 63, ENOLINK = 67,
    EPROTO = 71, EMULTIHOP = 72, EBADMSG = 74, EOVERFLOW = 75, EILSEQ = 84, EUSERS = 87,
    ENOTSOCK = 88, EDESTADDRREQ = 89, EMSGSIZE = 90, EPROTOTYPE = 91, ENOPROTOOPT = 92,
    EPROTONOSUPPORT = 93, ESOCKTNOSUPPORT = 94, EOPNOTSUPP = 95, EPFNOSUPPORT = 96,
    EAFNOSUPPORT = 97, EADDRINUSE = 98, EADDRNOTAVAIL = 99, ENETDOWN = 100,
    ENETUNREACH = 101, ENETRESET = 102, ECONNABORTED = 103, ECONNRESET = 104,
    ENOBUFS = 105, EISCONN = 106, ENOTCONN = 107, ESHUTDOWN = 108, ETOOMANYREFS = 109,
    ETIMEDOUT = 110, ECONNREFUSED = 111, EHOSTDOWN = 112, EHOSTUNREACH = 113,
    EALREADY = 114, EINPROGRESS = 115, ESTALE = 116, EDQUOT = 122, ECANCELED = 125,
    ELIBACC = 79, ELIBBAD = 80, EOWNERDEAD = 130, ENOTRECOVERABLE = 131,
}

impl Errno {
    /// The syscall return value for this error (`-errno` as `u64`).
    pub const fn ret(self) -> u64 {
        (-(self.0 as i64)) as u64
    }

    /// Decodes a raw syscall return value: `Some(errno)` for `-4095..=-1`.
    pub const fn from_ret(ret: u64) -> Option<Self> {
        let r = ret as i64;
        if r < 0 && r >= -4095 {
            Some(Self((-r) as u16))
        } else {
            None
        }
    }
}

impl core::fmt::Debug for Errno {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.name() {
            Some(name) => f.write_str(name),
            None => write!(f, "E{}", self.0),
        }
    }
}
