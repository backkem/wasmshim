//! The guest memory contract.

use core::any::Any;
use core::fmt;
use core::ops::BitOr;

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;

/// The guest page size. All mapping operations work in whole pages.
pub const PAGE_SIZE: u64 = 4096;

/// Page permissions (a set of read, write and execute).
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Perms(u8);

impl Perms {
    /// No access (a reserved or `PROT_NONE` page).
    pub const NONE: Self = Self(0);
    /// Readable.
    pub const R: Self = Self(1);
    /// Writable.
    pub const W: Self = Self(2);
    /// Executable.
    pub const X: Self = Self(4);
    /// Readable and writable.
    pub const RW: Self = Self(3);
    /// Readable and executable.
    pub const RX: Self = Self(5);
    /// Everything.
    pub const RWX: Self = Self(7);

    /// Builds a set from three flags.
    pub const fn new(read: bool, write: bool, exec: bool) -> Self {
        Self(read as u8 | (write as u8) << 1 | (exec as u8) << 2)
    }

    /// `self` without the permissions in `other`.
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// `true` if every permission in `other` is also in `self`.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// `true` if this set allows `access`.
    pub const fn allows(self, access: Access) -> bool {
        self.contains(match access {
            Access::Read => Self::R,
            Access::Write => Self::W,
            Access::Execute => Self::X,
        })
    }

    /// Readable?
    pub const fn read(self) -> bool {
        self.contains(Self::R)
    }
    /// Writable?
    pub const fn write(self) -> bool {
        self.contains(Self::W)
    }
    /// Executable?
    pub const fn exec(self) -> bool {
        self.contains(Self::X)
    }
}

impl BitOr for Perms {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl fmt::Debug for Perms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let flag = |on: bool, c: char| if on { c } else { '-' };
        write!(
            f,
            "{}{}{}",
            flag(self.read(), 'r'),
            flag(self.write(), 'w'),
            flag(self.exec(), 'x')
        )
    }
}

/// The kind of memory access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// A data load.
    Read,
    /// A data store.
    Write,
    /// An instruction fetch.
    Execute,
}

/// Why an access failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemFaultKind {
    /// No page is mapped at the address (Linux: `SEGV_MAPERR`).
    Unmapped,
    /// A page is mapped but forbids the access (Linux: `SEGV_ACCERR`).
    Protection,
}

/// A failed guest memory access. `addr` is the first byte that could not be accessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemFault {
    /// The faulting address.
    pub addr: u64,
    /// What the guest tried to do.
    pub access: Access,
    /// Why it failed.
    pub kind: MemFaultKind,
}

impl fmt::Display for MemFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let access = match self.access {
            Access::Read => "read of",
            Access::Write => "write to",
            Access::Execute => "fetch from",
        };
        let kind = match self.kind {
            MemFaultKind::Unmapped => "unmapped",
            MemFaultKind::Protection => "protected",
        };
        write!(f, "{access} {kind} address {:#x}", self.addr)
    }
}

impl core::error::Error for MemFault {}

/// A failed mapping operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapError {
    /// The address or length is not page-aligned, or the length is zero.
    Unaligned,
    /// The range leaves the address space.
    OutOfRange,
    /// `protect` hit a page that is not mapped.
    NotMapped,
    /// This address space cannot map that object (a [`SharedMemory`] made for another kind of
    /// address space, or an address space without shared pages).
    Unsupported,
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Unaligned => "the address or length is not page-aligned, or the length is zero",
            Self::OutOfRange => "the range leaves the address space",
            Self::NotMapped => "a page of the range is not mapped",
            Self::Unsupported => "this address space cannot map that object",
        })
    }
}

impl core::error::Error for MapError {}

/// One page of guest memory contents, shareable copy-on-write between address spaces.
pub type PageRef = Rc<[u8; PAGE_SIZE as usize]>;

/// A page source could not produce a page (its backing store failed). The access that needed
/// the page faults as [`MemFaultKind::Unmapped`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageSourceError;

impl fmt::Display for PageSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the page source could not produce a page")
    }
}

impl core::error::Error for PageSourceError {}

/// Where the contents of lazily filled pages come from: a file, for file-backed mappings.
///
/// [`GuestMemory::map_source`] maps pages whose contents a source produces on their first
/// read, write or fetch, instead of copying them in at map time. A source must be a pure
/// function of the page index: what a page holds must never depend on when (or whether) it
/// is touched, so lazy filling is invisible to the guest (the same bytes, faults and retired
/// instructions) and deterministic.
pub trait PageSource {
    /// The contents of page `index` (in the source's own numbering). A source may hand the
    /// same page to several callers: address spaces copy a page before writing to it.
    fn page(&self, index: u64) -> Result<PageRef, PageSourceError>;
}

/// Mapped pages as a checkpoint sees them ([`GuestMemory::for_each_mapped`]). Only
/// [`PageView::Zero`] covers more than one page.
#[derive(Clone, Copy)]
pub enum PageView<'a> {
    /// Pages that read as zeros: anonymous and never written.
    Zero,
    /// A page with bytes of its own: written, or already read from its source (compare with
    /// the source to tell which).
    Data(&'a [u8; PAGE_SIZE as usize]),
    /// A page not read from its source yet: page `index` of `source`.
    Source(&'a Rc<dyn PageSource>, u64),
    /// Page `index` of a shared memory object (the bytes are the object's).
    Shared(u64),
}

/// A memory object: pages that several address spaces map *shared* (`MAP_SHARED` memory, a
/// file's page cache, a memfd, a SysV segment). Every address space that maps page `i` of an
/// object reads and writes the same bytes, in place, and so do [`SharedMemory::read`] and
/// [`SharedMemory::write`]: a store through one mapping is visible to the next load through
/// any other, with no copying or hand-over (one guest CPU runs at a time, so this is
/// sequentially consistent).
///
/// An object is an unbounded, sparse array of pages that read as zeros until written (or, if
/// it was made over a [`PageSource`], as the source's pages until written); its *size* is its
/// owner's business (a file's length). Objects are made by whoever makes the address spaces
/// (the embedder's frontend), because an address space can only map objects of its own kind
/// ([`GuestMemory::map_shared`]). Single-threaded, like everything here.
pub trait SharedMemory {
    /// Reads `buf.len()` bytes at byte `offset` of the object. Fails only if the object's
    /// page source failed.
    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), PageSourceError>;
    /// Writes `data` at byte `offset` of the object (every mapping sees it at once).
    fn write(&self, offset: u64, data: &[u8]) -> Result<(), PageSourceError>;
    /// Every byte at or after `offset` reads as zero from now on (a file truncated there).
    /// Pages stay allocated while the object lives: mappings may still point at them.
    fn zero_from(&self, offset: u64) -> Result<(), PageSourceError>;
    /// This object as [`Any`], for the address space implementation that recognises it.
    fn into_any(self: Rc<Self>) -> Rc<dyn Any>;

    /// Store tracking, for a reader that keeps a copy of the object (a framebuffer's last
    /// frame): the indices of the pages that may have changed since the previous call,
    /// ascending (a superset is fine: a page counts once it was stored to through any mapping,
    /// [`SharedMemory::write`] or [`SharedMemory::zero_from`], or handed to an engine for direct
    /// writes), and a new period starts. `None`: unknown, compare everything; always the
    /// answer of the first call (tracking starts there) and of objects that do not track (the
    /// default). Speed only: a reader gets the same result either way.
    fn take_dirty(&self) -> Option<Vec<u64>> {
        None
    }
}

/// A guest virtual address space with per-page permissions.
///
/// Data accesses check permissions and fail with a [`MemFault`] that names the first
/// inaccessible byte. An access may span pages. A failed write may have stored the bytes
/// that precede the faulting page (like a real CPU's partial store), so callers that need
/// all-or-nothing behaviour check first with [`GuestMemory::check`].
///
/// Mapping operations take page-aligned ranges. A newly mapped page reads as zeros.
pub trait GuestMemory {
    /// Reads `buf.len()` bytes at `addr`, requiring read permission.
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault>;
    /// Writes `data` at `addr`, requiring write permission.
    fn write(&mut self, addr: u64, data: &[u8]) -> Result<(), MemFault>;
    /// Fetches the little-endian 32-bit instruction word at `addr`, requiring execute
    /// permission. `addr` must be 4-byte aligned (the caller checks that).
    fn fetch(&mut self, addr: u64) -> Result<u32, MemFault>;
    /// Checks that `len` bytes at `addr` allow `access`, without touching them.
    fn check(&self, addr: u64, len: u64, access: Access) -> Result<(), MemFault>;

    /// Writes `data` at `addr` ignoring permissions; only requires the pages to be mapped.
    /// For loaders and the personality (e.g. filling a read-only segment, or patching code
    /// through inspection), never for guest stores. A poke to an executable page is a code
    /// change like a guest store to one: it moves [`GuestMemory::code_epoch`] and the page's
    /// [`GuestMemory::code_version`], so no backend runs the old code.
    fn poke(&mut self, addr: u64, data: &[u8]) -> Result<(), MemFault>;

    /// Maps `len` bytes at `addr` as fresh zero pages with `perms`, replacing any existing mapping.
    fn map(&mut self, addr: u64, len: u64, perms: Perms) -> Result<(), MapError>;
    /// Maps `len` bytes at `addr` with `perms`, replacing any existing mapping, as pages whose
    /// contents are pages `first`, `first + 1`, ... of `source`. What the guest sees is what
    /// [`GuestMemory::map`] followed by a [`GuestMemory::poke`] of every page would give;
    /// implementations should fill each page on its first touch instead (an untouched page
    /// costs nothing), and keep untouched pages shared by [`GuestMemory::duplicate`].
    ///
    /// The default copies every page in now. If the source fails, the range is left unmapped
    /// and the result is [`MapError::OutOfRange`].
    fn map_source(
        &mut self,
        addr: u64,
        len: u64,
        perms: Perms,
        source: Rc<dyn PageSource>,
        first: u64,
    ) -> Result<(), MapError> {
        self.map(addr, len, perms)?;
        let mut at = 0;
        while at < len {
            let stored = match source.page(first + at / PAGE_SIZE) {
                Ok(p) if p.iter().all(|&b| b == 0) => Ok(()),
                Ok(p) => self.poke(addr + at, &p[..]).map_err(|_| PageSourceError),
                Err(e) => Err(e),
            };
            if stored.is_err() {
                let _ = self.unmap(addr, len);
                return Err(MapError::OutOfRange);
            }
            at += PAGE_SIZE;
        }
        Ok(())
    }

    /// Maps `len` bytes at `addr` with `perms`, replacing any existing mapping, as pages
    /// `first`, `first + 1`, ... of the memory object `object`, **shared**: loads and stores go
    /// to the object's pages in place, so every address space (and every other range of this
    /// one) mapping the same object page sees them at once, and [`GuestMemory::duplicate`]
    /// keeps the range shared rather than copy-on-write.
    ///
    /// `object` must have been made for this kind of address space; the default (an address
    /// space without shared pages) and a foreign object fail with [`MapError::Unsupported`].
    fn map_shared(
        &mut self,
        addr: u64,
        len: u64,
        perms: Perms,
        object: Rc<dyn SharedMemory>,
        first: u64,
    ) -> Result<(), MapError> {
        let _ = (addr, len, perms, object, first);
        Err(MapError::Unsupported)
    }

    /// Unmaps every page in the range. Unmapped holes are fine.
    fn unmap(&mut self, addr: u64, len: u64) -> Result<(), MapError>;
    /// Changes the permissions of a fully mapped range.
    fn protect(&mut self, addr: u64, len: u64, perms: Perms) -> Result<(), MapError>;
    /// The permissions of the page holding `addr`, or `None` if it is unmapped.
    fn perms(&self, addr: u64) -> Option<Perms>;

    /// The debugger's watch view. While set, each listed page (any address in it)
    /// forbids the listed accesses (`deny`, read and/or write) to every access through this
    /// address space, as if the guest had protected it: a load or store there fails with
    /// [`MemFaultKind::Protection`], and translated code caching the page drops it.
    /// [`GuestMemory::perms`] still reports the guest's own permissions, a mapping change
    /// (`map`, `unmap`, `protect`) drops the view, and a [`GuestMemory::duplicate`] never
    /// carries it. `set_watch(&[])` clears it. A debugger sets it only while a CPU runs (the
    /// personality's own accesses never see it). Returns `false` if this address space has no
    /// watch view (the default).
    fn set_watch(&mut self, pages: &[(u64, Perms)]) -> bool {
        let _ = pages;
        false
    }

    /// One past the highest valid guest address.
    fn address_limit(&self) -> u64;

    /// A counter that changes whenever code may have changed: a write to an executable page,
    /// or a map/unmap/protect that makes a page executable or stops it being executable.
    /// Values are never reused by another address space (so a cache that outlives an address
    /// space cannot mistake a new one for it). Backends that cache decoded or translated code
    /// compare it.
    fn code_epoch(&self) -> u64;

    /// A version of the code in the page holding `addr`: it changes whenever that page's code
    /// may have changed (as for [`GuestMemory::code_epoch`], but per page), and is never
    /// reused by another address space or another page state. Code caches validate a cached
    /// page with it after [`GuestMemory::code_epoch`] moved, instead of dropping everything
    /// (V8 flips pages between RW and RX all the time).
    ///
    /// The default is the global epoch: correct, but any change invalidates every page.
    fn code_version(&self, addr: u64) -> u64 {
        let _ = addr;
        self.code_epoch()
    }

    /// An independent copy of this address space (contents and permissions), as `fork` needs.
    /// Implementations should share pages copy-on-write.
    fn duplicate(&self) -> Box<dyn GuestMemory>;

    /// This address space as [`Any`], for a backend that must recognise its own memories
    /// (e.g. a remote engine that needs the engine-side handle of the address space it is
    /// asked to run against). The default is `None`.
    fn as_any(&self) -> Option<&dyn Any> {
        None
    }

    /// Mutable [`GuestMemory::as_any`] (for an in-process backend that runs against its own
    /// kind of address space). The default is `None`.
    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        None
    }

    /// Calls `f(addr, pages, perms, view)` for every mapped page in address order, for
    /// checkpoints. Listing changes nothing the guest can observe. The default
    /// is `Err(MapError::Unsupported)`: this address space cannot list its pages.
    fn for_each_mapped(
        &self,
        _f: &mut dyn FnMut(u64, u64, Perms, PageView<'_>),
    ) -> Result<(), MapError> {
        Err(MapError::Unsupported)
    }

    /// Reads a byte.
    fn read_u8(&self, addr: u64) -> Result<u8, MemFault> {
        let mut b = [0; 1];
        self.read(addr, &mut b)?;
        Ok(b[0])
    }
    /// Reads a little-endian `u16`.
    fn read_u16(&self, addr: u64) -> Result<u16, MemFault> {
        let mut b = [0; 2];
        self.read(addr, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    /// Reads a little-endian `u32`.
    fn read_u32(&self, addr: u64) -> Result<u32, MemFault> {
        let mut b = [0; 4];
        self.read(addr, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    /// Reads a little-endian `u64`.
    fn read_u64(&self, addr: u64) -> Result<u64, MemFault> {
        let mut b = [0; 8];
        self.read(addr, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    /// Reads a NUL-terminated string of at most `max` bytes (without the NUL). `Ok(None)`
    /// if no NUL was found within `max` bytes.
    fn read_cstr(&self, addr: u64, max: usize) -> Result<Option<Vec<u8>>, MemFault> {
        let mut out = Vec::new();
        let mut chunk = [0u8; 64];
        let mut at = addr;
        while out.len() < max {
            // Stop each chunk at a page end, so a string just before an unmapped page works.
            let page_left = (PAGE_SIZE - (at % PAGE_SIZE)) as usize;
            let n = page_left.min(chunk.len()).min(max - out.len());
            self.read(at, &mut chunk[..n])?;
            if let Some(end) = chunk[..n].iter().position(|&b| b == 0) {
                out.extend_from_slice(&chunk[..end]);
                return Ok(Some(out));
            }
            out.extend_from_slice(&chunk[..n]);
            at = at.wrapping_add(n as u64);
        }
        Ok(None)
    }

    /// Zeroes `len` bytes at `addr`, requiring write permission.
    fn zero(&mut self, addr: u64, len: u64) -> Result<(), MemFault> {
        let block = [0u8; 256];
        let mut done = 0;
        while done < len {
            let n = (len - done).min(block.len() as u64);
            self.write(addr.wrapping_add(done), &block[..n as usize])?;
            done += n;
        }
        Ok(())
    }

    /// Writes a byte.
    fn write_u8(&mut self, addr: u64, v: u8) -> Result<(), MemFault> {
        self.write(addr, &[v])
    }
    /// Writes a little-endian `u16`.
    fn write_u16(&mut self, addr: u64, v: u16) -> Result<(), MemFault> {
        self.write(addr, &v.to_le_bytes())
    }
    /// Writes a little-endian `u32`.
    fn write_u32(&mut self, addr: u64, v: u32) -> Result<(), MemFault> {
        self.write(addr, &v.to_le_bytes())
    }
    /// Writes a little-endian `u64`.
    fn write_u64(&mut self, addr: u64, v: u64) -> Result<(), MemFault> {
        self.write(addr, &v.to_le_bytes())
    }
}
