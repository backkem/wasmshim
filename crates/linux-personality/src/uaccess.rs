//! Copying data between guest memory and the personality. Every failure is `EFAULT`.

use wasmshim_cpu::GuestMemory;
use wasmshim_linux_abi::Errno;

use crate::vfs::PATH_MAX;

/// Largest single buffer a syscall copies in one go (bigger requests are clamped by callers).
pub const MAX_IO: usize = 1 << 24;

/// A guest's 64-bit length clamped to [`MAX_IO`], the same on a 32-bit host (where `as usize`
/// would keep only the low bits).
pub(crate) fn io_len(len: u64) -> usize {
    len.min(MAX_IO as u64) as usize
}

/// A guest's 64-bit length as a host size, saturating on a 32-bit host.
pub(crate) fn host_len(len: u64) -> usize {
    usize::try_from(len).unwrap_or(usize::MAX)
}

/// Reads `len` bytes at `addr`.
pub fn read_bytes(mem: &mut dyn GuestMemory, addr: u64, len: usize) -> Result<Vec<u8>, Errno> {
    let mut buf = vec![0; len];
    mem.read(addr, &mut buf).map_err(|_| Errno::EFAULT)?;
    Ok(buf)
}

/// Reads exactly `N` bytes at `addr`.
pub fn read_array<const N: usize>(mem: &mut dyn GuestMemory, addr: u64) -> Result<[u8; N], Errno> {
    let mut buf = [0; N];
    mem.read(addr, &mut buf).map_err(|_| Errno::EFAULT)?;
    Ok(buf)
}

/// Writes `data` at `addr`.
pub fn write_bytes(mem: &mut dyn GuestMemory, addr: u64, data: &[u8]) -> Result<(), Errno> {
    mem.write(addr, data).map_err(|_| Errno::EFAULT)
}

/// Reads a little-endian `u64`.
pub fn read_u64(mem: &mut dyn GuestMemory, addr: u64) -> Result<u64, Errno> {
    mem.read_u64(addr).map_err(|_| Errno::EFAULT)
}

/// Writes a little-endian `u64`.
pub fn write_u64(mem: &mut dyn GuestMemory, addr: u64, v: u64) -> Result<(), Errno> {
    mem.write_u64(addr, v).map_err(|_| Errno::EFAULT)
}

/// Writes a little-endian `u32`.
pub fn write_u32(mem: &mut dyn GuestMemory, addr: u64, v: u32) -> Result<(), Errno> {
    mem.write_u32(addr, v).map_err(|_| Errno::EFAULT)
}

/// Reads a NUL-terminated string of at most `max` bytes (without the NUL).
pub fn read_cstr_max(mem: &dyn GuestMemory, addr: u64, max: usize) -> Result<Vec<u8>, Errno> {
    mem.read_cstr(addr, max)
        .map_err(|_| Errno::EFAULT)?
        .ok_or(Errno::ENAMETOOLONG)
}

/// Reads a path argument (`ENAMETOOLONG` past `PATH_MAX`).
pub fn read_path(mem: &mut dyn GuestMemory, addr: u64) -> Result<Vec<u8>, Errno> {
    read_cstr_max(mem, addr, PATH_MAX)
}

/// Reads a NULL-terminated array of string pointers (`argv`, `envp`).
pub fn read_str_array(mem: &mut dyn GuestMemory, addr: u64) -> Result<Vec<Vec<u8>>, Errno> {
    let mut out = Vec::new();
    if addr == 0 {
        return Ok(out);
    }
    let mut total = 0usize;
    for i in 0.. {
        let p = read_u64(mem, addr + 8 * i)?;
        if p == 0 {
            break;
        }
        let s = read_cstr_max(mem, p, 128 * 1024)?;
        total += s.len() + 1;
        if total > 2 << 20 || i > 1 << 16 {
            return Err(Errno::E2BIG);
        }
        out.push(s);
    }
    Ok(out)
}
