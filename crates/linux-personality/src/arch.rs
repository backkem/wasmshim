//! Per-architecture Linux ABI facts (the "Linux on AArch64" part, not the ISA part).

use wasmshim_linux_abi::Sysno;
use wasmshim_linux_abi::consts::hwcap;

/// The guest architecture as Linux sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// AArch64 (asm-generic syscall table).
    Aarch64,
}

impl Arch {
    /// Maps a raw syscall number to its identity.
    pub fn sysno(self, nr: u64) -> Option<Sysno> {
        match self {
            Self::Aarch64 => wasmshim_linux_abi::aarch64::sysno(nr),
        }
    }

    /// The ELF `e_machine` this architecture executes.
    pub fn elf_machine(self) -> u16 {
        match self {
            Self::Aarch64 => wasmshim_linux_abi::aarch64::EM_AARCH64,
        }
    }

    /// `uname -m` and `AT_PLATFORM`.
    pub fn machine(self) -> &'static str {
        match self {
            Self::Aarch64 => "aarch64",
        }
    }

    /// The instruction set (what the substrate's syscall entry needs to know about it, such
    /// as the syscall instruction's length for restarts).
    pub fn isa(self) -> wasmshim_kernel::Isa {
        match self {
            Self::Aarch64 => wasmshim_kernel::Isa::Aarch64,
        }
    }

    /// Code of the signal-return trampoline: `rt_sigreturn` (the vDSO's `__kernel_rt_sigreturn`).
    pub fn sigreturn_trampoline(self) -> &'static [u8] {
        match self {
            // mov x8, #139 (rt_sigreturn); svc #0
            Self::Aarch64 => &[0x68, 0x11, 0x80, 0xd2, 0x01, 0x00, 0x00, 0xd4],
        }
    }

    /// `AT_HWCAP`: what the CPU backend implements. Deliberately minimal: FP and Advanced SIMD
    /// (required by the AArch64 ABI) only. No LSE atomics advertised (libraries then use
    /// LDXR/STXR, which every backend must support anyway), no CPUID emulation.
    pub fn hwcap(self) -> u64 {
        match self {
            Self::Aarch64 => hwcap::FP | hwcap::ASIMD,
        }
    }
}
