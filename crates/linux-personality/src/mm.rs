//! Address-space bookkeeping: the VMA list on top of a [`GuestMemory`].
//!
//! The [`GuestMemory`] holds pages and permissions; this module remembers *why* each range is
//! mapped (heap, stack, anonymous, file) so `mmap` can place new mappings, `munmap`/`mprotect`
//! can split ranges, and `/proc/self/maps` can be rendered. Placement is deterministic: no
//! randomisation, top-down from a fixed base like Linux's legacy-free layout.

use crate::shm::MemRef;
use std::ops::{Deref, DerefMut};

use wasmshim_cpu::{GuestMemory, MapError, PAGE_SIZE, Perms};
use wasmshim_kernel::checkpoint::{Baseline, Ckpt, CkptError, Reader, Replay, Writer};
use wasmshim_linux_abi::Errno;

// The layout keeps every address below 3.5 GiB, so the whole guest address space fits a
// memory32 linear memory with an offset. A soft-MMU can lift this later (V8's large
// reservations) without changing `GuestMemory`.

/// Top of the initial stack (exclusive).
pub const STACK_TOP: u64 = 0xe000_0000;
/// Stack size reserved below [`STACK_TOP`] (`RLIMIT_STACK` default, 8 MiB).
pub const STACK_SIZE: u64 = 8 << 20;
/// `mmap` allocates top-down below this address.
pub const MMAP_TOP: u64 = 0xd000_0000;
/// Top of the high `mmap` area, for what does not fit below [`MMAP_TOP`] (2^47, like a
/// 47-bit Linux user address space).
pub const HIGH_MMAP_TOP: u64 = 1 << 47;
/// Bottom of the high `mmap` area (4 GiB).
pub const HIGH_MMAP_MIN: u64 = 1 << 32;
/// The signal-return trampoline page (`[vdso]`): handlers installed without `SA_RESTORER`
/// return here, as Linux returns through the vDSO's `__kernel_rt_sigreturn` (glibc on arm64
/// never sets `SA_RESTORER`). One page right above the `mmap` area.
pub const SIGTRAMP: u64 = MMAP_TOP;
/// Load bias for `ET_DYN` executables (PIE, static-pie).
pub const PIE_BASE: u64 = 0x5555_0000;
/// Load bias for the program interpreter (`ld.so`) of dynamically linked programs.
pub const INTERP_BASE: u64 = 0x4000_0000;
/// Lowest address `mmap` hands out (`vm.mmap_min_addr`).
pub const MMAP_MIN: u64 = 0x1_0000;

const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// Rounds up to a page boundary (saturating).
pub const fn page_up(x: u64) -> u64 {
    x.saturating_add(PAGE_MASK) & !PAGE_MASK
}

/// What a mapping is for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VmaKind {
    /// Anonymous memory.
    Anon,
    /// The program break area.
    Heap,
    /// The main thread's stack.
    Stack,
    /// A file mapping (including executable segments).
    File { path: Vec<u8>, offset: u64 },
    /// The signal-return trampoline ([`SIGTRAMP`]).
    Vdso,
    /// A `MAP_SHARED` mapping (of anonymous memory, a file, a memfd, a SysV segment): the
    /// pages of `obj` from byte `offset`. `name` is what `/proc/<pid>/maps` shows, and
    /// `may_write` whether `mprotect` may add `PROT_WRITE` (Linux's `VM_MAYWRITE`).
    Shared {
        obj: MemRef,
        offset: u64,
        name: Vec<u8>,
        may_write: bool,
    },
}

/// Linux's mapping kinds on the substrate's address spaces: offsets move along with splits,
/// file and shared mappings merge only with contiguous offsets, and a fixed-size shared
/// object's pages past its end stay inaccessible (`SIGBUS`, see [`AddressSpace::beyond_object`]).
impl wasmshim_kernel::space::MapKind for VmaKind {
    fn advance(&mut self, by: u64) {
        // `mmap` keeps a file mapping's end within `i64::MAX`; wrapping only guards the rest.
        if let VmaKind::File { offset, .. } | VmaKind::Shared { offset, .. } = self {
            *offset = offset.wrapping_add(by);
        }
    }

    fn continues(&self, len: u64, next: &Self) -> bool {
        match (self, next) {
            (
                VmaKind::File {
                    path: pa,
                    offset: oa,
                },
                VmaKind::File {
                    path: pb,
                    offset: ob,
                },
            ) => pa == pb && oa.checked_add(len) == Some(*ob),
            (
                VmaKind::Shared {
                    obj: xa,
                    offset: oa,
                    name: na,
                    may_write: wa,
                },
                VmaKind::Shared {
                    obj: xb,
                    offset: ob,
                    name: nb,
                    may_write: wb,
                },
            ) => xa == xb && na == nb && wa == wb && oa.checked_add(len) == Some(*ob),
            (ka, kb) => ka == kb,
        }
    }

    fn object_end(&self, start: u64) -> Option<u64> {
        match self {
            VmaKind::Shared { obj, offset, .. } => obj
                .size
                .map(|size| start.saturating_add(page_up(size).saturating_sub(*offset))),
            _ => None,
        }
    }
}

/// One mapping `[start, end)`.
pub type Vma = wasmshim_kernel::space::Vma<VmaKind>;

/// A process address space: the substrate's address space with Linux's mapping kinds (reached
/// through `Deref`: `id`, `mem`, `vmas`, `record`, `covered`, `is_free`, `vma_at`), plus the
/// program break and Linux's layout and errors.
pub struct AddressSpace {
    space: wasmshim_kernel::space::AddressSpace<VmaKind>,
    brk_start: u64,
    brk: u64,
}

impl Deref for AddressSpace {
    type Target = wasmshim_kernel::space::AddressSpace<VmaKind>;
    fn deref(&self) -> &Self::Target {
        &self.space
    }
}

impl DerefMut for AddressSpace {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.space
    }
}

fn map_err(e: MapError) -> Errno {
    match e {
        MapError::Unaligned => Errno::EINVAL,
        MapError::OutOfRange | MapError::NotMapped => Errno::ENOMEM,
        MapError::Unsupported => Errno::ENODEV,
    }
}

impl AddressSpace {
    /// An empty address space over `mem`.
    pub fn new(id: u64, mem: Box<dyn GuestMemory>) -> Self {
        Self {
            space: wasmshim_kernel::space::AddressSpace::new(id, mem),
            brk_start: 0,
            brk: 0,
        }
    }

    /// Saves this address space (the substrate's part, then the program break).
    pub(crate) fn save(
        &self,
        w: &mut Writer,
        baseline: &mut dyn FnMut(&Vma) -> Baseline,
        pages: &mut dyn FnMut(u64, &[u8]),
    ) -> Result<(), CkptError> {
        self.space.save(w, baseline, pages)?;
        self.brk_start.save(w);
        self.brk.save(w);
        Ok(())
    }

    /// Restores what [`AddressSpace::save`] wrote.
    pub(crate) fn load(
        r: &mut Reader<'_>,
        mem: Box<dyn GuestMemory>,
        replay: Replay<'_, VmaKind>,
        pages: &[(u64, &[u8])],
    ) -> Result<Self, CkptError> {
        Ok(Self {
            space: wasmshim_kernel::space::AddressSpace::load(r, mem, replay, pages)?,
            brk_start: Ckpt::load(r)?,
            brk: Ckpt::load(r)?,
        })
    }

    /// A copy-on-write copy of this address space (for `fork`) with id `id`.
    pub fn duplicate(&self, id: u64) -> Self {
        Self {
            space: self.space.duplicate(id),
            brk_start: self.brk_start,
            brk: self.brk,
        }
    }

    /// Maps fresh zero pages at `[start, end)`, replacing whatever was there.
    pub fn map(&mut self, start: u64, end: u64, perms: Perms, kind: VmaKind) -> Result<(), Errno> {
        self.space.map(start, end, perms, kind).map_err(map_err)
    }

    /// Unmaps `[start, end)`; holes are fine.
    pub fn unmap(&mut self, start: u64, end: u64) -> Result<(), Errno> {
        self.space.unmap(start, end).map_err(map_err)
    }

    /// Changes permissions of `[start, end)`; fails with `ENOMEM` if any page is unmapped.
    pub fn protect(&mut self, start: u64, end: u64, perms: Perms) -> Result<(), Errno> {
        self.space.protect(start, end, perms).map_err(map_err)
    }

    /// The highest free, page-aligned range of `len` bytes below [`MMAP_TOP`] (the
    /// memory32-friendly area), or else below [`HIGH_MMAP_TOP`]: large
    /// reservations such as V8's 8 GiB WebAssembly guard regions live above 4 GiB.
    pub fn find_free(&self, len: u64) -> Option<u64> {
        self.space
            .find_free_below(MMAP_TOP, MMAP_MIN, len)
            .or_else(|| {
                self.space
                    .find_free_below(HIGH_MMAP_TOP, HIGH_MMAP_MIN, len)
            })
    }

    /// Sets the initial program break (page-aligned end of the executable).
    pub fn init_brk(&mut self, at: u64) {
        self.brk_start = at;
        self.brk = at;
    }

    /// `brk(2)`: moves the break if possible and returns the (possibly unchanged) break.
    pub fn set_brk(&mut self, want: u64) -> u64 {
        if want < self.brk_start {
            return self.brk;
        }
        let old_end = page_up(self.brk);
        let new_end = page_up(want);
        if new_end > old_end {
            if !self.is_free(old_end, new_end)
                || self
                    .map(old_end, new_end, Perms::RW, VmaKind::Heap)
                    .is_err()
            {
                return self.brk;
            }
        } else if new_end < old_end && self.unmap(new_end, old_end).is_err() {
            return self.brk;
        }
        self.brk = want;
        self.brk
    }

    /// Maps `[start, end)` shared, as the pages of `obj` from byte `offset` (page-aligned).
    #[allow(clippy::too_many_arguments)]
    pub fn map_shared(
        &mut self,
        start: u64,
        end: u64,
        perms: Perms,
        obj: &MemRef,
        offset: u64,
        name: Vec<u8>,
        may_write: bool,
    ) -> Result<(), Errno> {
        self.mem
            .map_shared(
                start,
                end - start,
                perms,
                obj.pages.clone(),
                offset / PAGE_SIZE,
            )
            .map_err(map_err)?;
        if let Some(size) = obj.size {
            // Pages past a fixed-size object fault (SIGBUS, see `beyond_object`).
            let limit = page_up(size);
            let first_beyond = start.saturating_add(limit.saturating_sub(offset));
            if first_beyond < end {
                self.mem
                    .protect(first_beyond, end - first_beyond, Perms::NONE)
                    .map_err(map_err)?;
            }
        }
        self.record(
            start,
            end,
            perms,
            VmaKind::Shared {
                obj: obj.clone(),
                offset,
                name,
                may_write,
            },
        );
        Ok(())
    }

    /// The shared mapping holding `addr`: its object and the byte offset of `addr` in it.
    pub fn shared_at(&self, addr: u64) -> Option<(&MemRef, u64)> {
        let v = self.vma_at(addr)?;
        match &v.kind {
            VmaKind::Shared { obj, offset, .. } => Some((obj, offset + (addr - v.start))),
            _ => None,
        }
    }

    /// `true` if `addr` lies in a shared mapping past the end of its fixed-size object (an
    /// access there is `SIGBUS`, not `SIGSEGV`), and the mapping allows `access`.
    pub fn beyond_object(&self, addr: u64, access: wasmshim_cpu::Access) -> bool {
        let Some(v) = self.vma_at(addr) else {
            return false;
        };
        match &v.kind {
            VmaKind::Shared { obj, offset, .. } => obj.size.is_some_and(|size| {
                offset + (addr - v.start) >= page_up(size) && v.perms.allows(access)
            }),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasmshim_cpu::PagedMemory;

    fn space() -> AddressSpace {
        AddressSpace::new(1, Box::new(PagedMemory::new()))
    }

    #[test]
    fn split_on_unmap_and_protect() {
        let mut a = space();
        a.map(0x10000, 0x20000, Perms::RW, VmaKind::Anon).unwrap();
        a.unmap(0x14000, 0x15000).unwrap();
        let v: Vec<_> = a.vmas().map(|v| (v.start, v.end)).collect();
        assert_eq!(v, [(0x10000, 0x14000), (0x15000, 0x20000)]);
        a.protect(0x16000, 0x17000, Perms::R).unwrap();
        let v: Vec<_> = a.vmas().map(|v| (v.start, v.end, v.perms)).collect();
        assert_eq!(
            v,
            [
                (0x10000, 0x14000, Perms::RW),
                (0x15000, 0x16000, Perms::RW),
                (0x16000, 0x17000, Perms::R),
                (0x17000, 0x20000, Perms::RW)
            ]
        );
        assert_eq!(a.protect(0x13000, 0x16000, Perms::R), Err(Errno::ENOMEM));
        assert!(a.mem.write_u8(0x16000, 1).is_err());
    }

    #[test]
    fn top_down_placement_fills_gaps() {
        let mut a = space();
        let first = a.find_free(0x3000).unwrap();
        assert_eq!(first, MMAP_TOP - 0x3000);
        a.map(first, first + 0x3000, Perms::RW, VmaKind::Anon)
            .unwrap();
        let second = a.find_free(0x1000).unwrap();
        assert_eq!(second, first - 0x1000);
        a.map(second, second + 0x1000, Perms::RW, VmaKind::Anon)
            .unwrap();
        a.unmap(first, first + 0x1000).unwrap();
        assert_eq!(a.find_free(0x1000), Some(first));
        assert_eq!(a.find_free(0x2000), Some(second - 0x2000));
    }

    #[test]
    fn brk_grows_and_shrinks() {
        let mut a = space();
        a.init_brk(0x50_0000);
        assert_eq!(a.set_brk(0), 0x50_0000);
        assert_eq!(a.set_brk(0x50_2100), 0x50_2100);
        a.mem.write_u8(0x50_2000, 1).unwrap();
        assert_eq!(a.set_brk(0x50_1000), 0x50_1000);
        assert!(a.mem.write_u8(0x50_2000, 1).is_err());
        a.map(0x50_3000, 0x50_4000, Perms::RW, VmaKind::Anon)
            .unwrap();
        assert_eq!(
            a.set_brk(0x50_5000),
            0x50_1000,
            "cannot grow into a mapping"
        );
    }
}
