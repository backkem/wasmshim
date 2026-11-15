//! The Linux application ABI as pure data: syscall numbers, errno values, flags and struct
//! layouts. No logic and no I/O, so every frontend (CPU emulator, WALI, ...) and tool can share it.
//!
//! Values are the asm-generic ones used by AArch64 (and RISC-V). Where AArch64 differs from
//! x86 (open flags, struct stat), the AArch64 value is the one given.

#![no_std]
#![deny(unsafe_code)]

pub mod consts;
mod errno;
pub mod structs;
mod sysno;

pub use errno::Errno;
pub use sysno::Sysno;

/// AArch64-specific ABI facts.
pub mod aarch64 {
    use crate::Sysno;

    /// `e_machine` for AArch64 ELF files.
    pub const EM_AARCH64: u16 = 183;

    /// Maps an AArch64 syscall number (the asm-generic table) to a [`Sysno`].
    pub const fn sysno(nr: u64) -> Option<Sysno> {
        Sysno::from_generic(nr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syscall_table_round_trips() {
        assert_eq!(aarch64::sysno(64), Some(Sysno::write));
        assert_eq!(aarch64::sysno(94), Some(Sysno::exit_group));
        assert_eq!(aarch64::sysno(291), Some(Sysno::statx));
        assert_eq!(aarch64::sysno(244), None);
        assert_eq!(Sysno::rt_sigaction.name(), "rt_sigaction");
        for nr in 0..500 {
            if let Some(s) = Sysno::from_generic(nr) {
                assert_eq!(s.generic(), nr);
            }
        }
    }

    #[test]
    fn errno_encoding() {
        assert_eq!(Errno::ENOENT.ret(), (-2i64) as u64);
        assert_eq!(Errno::from_ret((-38i64) as u64), Some(Errno::ENOSYS));
        assert_eq!(Errno::from_ret(5), None);
        assert_eq!(Errno::EAGAIN.name(), Some("EAGAIN"));
    }
}
