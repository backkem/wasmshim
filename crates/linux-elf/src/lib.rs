//! ELF64 loading for Linux user programs.
//!
//! * [`Elf::parse`] validates an ELF64 little-endian image and its program headers;
//!   [`Elf::parse_prefix`] does the same from the start of a file only ([`header_len`] says
//!   how much of it).
//! * [`load_with`] maps its `PT_LOAD` segments into a [`GuestMemory`] the way Linux's
//!   `binfmt_elf` does (page-granular, file bytes from the page-aligned offset, BSS zeroed),
//!   with file pages filled on first touch from a [`PageSource`]; [`load`] copies them from
//!   the whole file in memory.
//! * [`build_stack`] writes the initial process stack: `argc`, `argv`, `envp`, the auxiliary
//!   vector and the strings and random bytes they point to.
//!
//! Every input is untrusted: malformed files are [`ElfError`]s, never panics.

#![no_std]
#![deny(unsafe_code)]

extern crate alloc;

use alloc::rc::Rc;
use alloc::vec::Vec;

use wasmshim_cpu::{
    GuestMemory, MapError, MemFault, PAGE_SIZE, PageRef, PageSource, PageSourceError, Perms,
};

/// `e_type` of a fixed-address executable.
pub const ET_EXEC: u16 = 2;
/// `e_type` of a position-independent executable or shared object.
pub const ET_DYN: u16 = 3;
/// Loadable segment.
pub const PT_LOAD: u32 = 1;
/// Program interpreter path.
pub const PT_INTERP: u32 = 3;
/// Program header table itself.
pub const PT_PHDR: u32 = 6;
/// Thread-local storage template.
pub const PT_TLS: u32 = 7;
/// Stack executability.
pub const PT_GNU_STACK: u32 = 0x6474_e551;

const PAGE_MASK: u64 = PAGE_SIZE - 1;
const EHDR_SIZE: usize = 64;
const PHDR_SIZE: usize = 56;

/// Why an ELF image was rejected or could not be loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElfError {
    /// Not an ELF file at all.
    NotElf,
    /// ELF, but not 64-bit little-endian version 1.
    Unsupported,
    /// Neither `ET_EXEC` nor `ET_DYN`.
    NotExecutable,
    /// A header, table or segment points outside the file or overflows.
    Truncated,
    /// A segment is malformed (e.g. `filesz > memsz`, misaligned offset).
    BadSegment,
    /// The image has no loadable segments.
    NoSegments,
    /// Mapping a segment failed.
    Map(MapError),
    /// Writing segment or stack contents failed.
    Memory(MemFault),
}

impl From<MapError> for ElfError {
    fn from(e: MapError) -> Self {
        Self::Map(e)
    }
}

impl From<MemFault> for ElfError {
    fn from(e: MemFault) -> Self {
        Self::Memory(e)
    }
}

impl core::fmt::Display for ElfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotElf => f.write_str("not an ELF file"),
            Self::Unsupported => f.write_str("not a 64-bit little-endian ELF of version 1"),
            Self::NotExecutable => f.write_str("neither an executable nor a shared object"),
            Self::Truncated => f.write_str("a header, table or segment points outside the file"),
            Self::BadSegment => f.write_str("a malformed segment"),
            Self::NoSegments => f.write_str("no loadable segments"),
            Self::Map(e) => write!(f, "mapping a segment failed: {e}"),
            Self::Memory(e) => write!(f, "writing the image failed: {e}"),
        }
    }
}

impl core::error::Error for ElfError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Map(e) => Some(e),
            Self::Memory(e) => Some(e),
            _ => None,
        }
    }
}

/// One program header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramHeader {
    pub p_type: u32,
    pub flags: u32,
    pub offset: u64,
    pub vaddr: u64,
    pub filesz: u64,
    pub memsz: u64,
    pub align: u64,
}

impl ProgramHeader {
    /// Page permissions from `p_flags` (PF_X = 1, PF_W = 2, PF_R = 4).
    pub fn perms(&self) -> Perms {
        Perms::new(
            self.flags & 4 != 0,
            self.flags & 2 != 0,
            self.flags & 1 != 0,
        )
    }
}

/// A parsed ELF64 image (borrowing the file bytes, or a prefix of them holding the headers).
#[derive(Clone)]
pub struct Elf<'a> {
    bytes: &'a [u8],
    /// The size of the whole file.
    pub file_len: u64,
    /// `e_type` ([`ET_EXEC`] or [`ET_DYN`]).
    pub e_type: u16,
    /// `e_machine` (183 for AArch64).
    pub machine: u16,
    /// `e_entry`, unbiased.
    pub entry: u64,
    /// `e_phoff`.
    pub phoff: u64,
    /// `e_phentsize`.
    pub phentsize: u16,
    /// The program headers.
    pub phdrs: Vec<ProgramHeader>,
}

/// The `N` bytes at `at`; `None` past the end (also when `at + N` overflows: `at` comes from
/// the file).
fn bytes_at<const N: usize>(b: &[u8], at: usize) -> Option<[u8; N]> {
    b.get(at..at.checked_add(N)?)?.try_into().ok()
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    bytes_at(b, at).map(u16::from_le_bytes)
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    bytes_at(b, at).map(u32::from_le_bytes)
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    bytes_at(b, at).map(u64::from_le_bytes)
}

impl core::fmt::Debug for Elf<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Elf")
            .field("len", &self.file_len)
            .field("e_type", &self.e_type)
            .field("machine", &self.machine)
            .field("entry", &self.entry)
            .field("phdrs", &self.phdrs)
            .finish()
    }
}

impl<'a> Elf<'a> {
    /// Parses and validates the ELF header and program header table of a whole file.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, ElfError> {
        Self::parse_prefix(bytes, bytes.len() as u64)
    }

    /// Parses and validates the ELF header and program header table from the first bytes of a
    /// file of `file_len` bytes. The prefix must hold the ELF header, the program headers and
    /// the `PT_INTERP` path ([`header_len`] says how long that is), else the result is
    /// [`ElfError::Truncated`]. Segments are checked against `file_len`.
    pub fn parse_prefix(bytes: &'a [u8], file_len: u64) -> Result<Self, ElfError> {
        if bytes.len() < 4 || &bytes[..4] != b"\x7fELF" {
            return Err(ElfError::NotElf);
        }
        if bytes.len() < EHDR_SIZE {
            return Err(ElfError::Truncated);
        }
        // EI_CLASS = ELFCLASS64, EI_DATA = little-endian, EI_VERSION = 1.
        if bytes[4] != 2 || bytes[5] != 1 || bytes[6] != 1 {
            return Err(ElfError::Unsupported);
        }
        let field = |v: Option<u64>| v.ok_or(ElfError::Truncated);
        let e_type = field(u16_at(bytes, 16).map(u64::from))? as u16;
        if e_type != ET_EXEC && e_type != ET_DYN {
            return Err(ElfError::NotExecutable);
        }
        let machine = field(u16_at(bytes, 18).map(u64::from))? as u16;
        let entry = field(u64_at(bytes, 24))?;
        let phoff = field(u64_at(bytes, 32))?;
        let phentsize = field(u16_at(bytes, 54).map(u64::from))? as u16;
        let phnum = field(u16_at(bytes, 56).map(u64::from))? as usize;
        if usize::from(phentsize) != PHDR_SIZE {
            return Err(ElfError::Unsupported);
        }
        let mut phdrs = Vec::with_capacity(phnum);
        for i in 0..phnum {
            let at = usize::try_from(phoff)
                .ok()
                .and_then(|o| o.checked_add(i * PHDR_SIZE))
                .ok_or(ElfError::Truncated)?;
            let ph = (|| {
                // The whole entry must be there (`at + PHDR_SIZE` may overflow).
                let e: [u8; PHDR_SIZE] = bytes_at(bytes, at)?;
                Some(ProgramHeader {
                    p_type: u32_at(&e, 0)?,
                    flags: u32_at(&e, 4)?,
                    offset: u64_at(&e, 8)?,
                    vaddr: u64_at(&e, 16)?,
                    filesz: u64_at(&e, 32)?,
                    memsz: u64_at(&e, 40)?,
                    align: u64_at(&e, 48)?,
                })
            })()
            .ok_or(ElfError::Truncated)?;
            if ph.p_type == PT_LOAD {
                let file_end = ph
                    .offset
                    .checked_add(ph.filesz)
                    .ok_or(ElfError::Truncated)?;
                if file_end > file_len {
                    return Err(ElfError::Truncated);
                }
                if ph.filesz > ph.memsz || (ph.offset & PAGE_MASK) != (ph.vaddr & PAGE_MASK) {
                    return Err(ElfError::BadSegment);
                }
                ph.vaddr.checked_add(ph.memsz).ok_or(ElfError::BadSegment)?;
            }
            if ph.p_type == PT_INTERP
                && ph
                    .offset
                    .checked_add(ph.filesz)
                    .is_some_and(|end| end > bytes.len() as u64 && end <= file_len)
            {
                // The path is in the file but not in the prefix. (One past the end of the
                // file is ignored.)
                return Err(ElfError::Truncated);
            }
            phdrs.push(ph);
        }
        if !phdrs.iter().any(|p| p.p_type == PT_LOAD) {
            return Err(ElfError::NoSegments);
        }
        Ok(Self {
            bytes,
            file_len,
            e_type,
            machine,
            entry,
            phoff,
            phentsize,
            phdrs,
        })
    }

    /// The `PT_INTERP` path (e.g. `/lib/ld-musl-aarch64.so.1`), if the image has one.
    pub fn interp(&self) -> Option<&'a [u8]> {
        let ph = self.phdrs.iter().find(|p| p.p_type == PT_INTERP)?;
        let start = usize::try_from(ph.offset).ok()?;
        let len = usize::try_from(ph.filesz).ok()?;
        let raw = self.bytes.get(start..start.checked_add(len)?)?;
        Some(raw.split(|&b| b == 0).next().unwrap_or(raw))
    }

    /// `true` for `ET_DYN` (PIE / static-pie / shared object): the caller picks a load bias.
    pub fn is_dyn(&self) -> bool {
        self.e_type == ET_DYN
    }

    /// The lowest and highest+1 virtual addresses covered by `PT_LOAD` segments (unbiased, page-aligned).
    pub fn span(&self) -> (u64, u64) {
        let loads = self.phdrs.iter().filter(|p| p.p_type == PT_LOAD);
        let lo = loads
            .clone()
            .map(|p| p.vaddr & !PAGE_MASK)
            .min()
            .unwrap_or(0);
        let hi = loads
            .map(|p| page_up(p.vaddr.saturating_add(p.memsz)))
            .max()
            .unwrap_or(0);
        (lo, hi)
    }
}

const fn page_up(x: u64) -> u64 {
    x.saturating_add(PAGE_MASK) & !PAGE_MASK
}

/// A mapped segment, for `/proc/self/maps`-style bookkeeping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapping {
    pub start: u64,
    pub end: u64,
    pub perms: Perms,
    /// File offset of `start`.
    pub offset: u64,
}

/// What [`load`] produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedImage {
    /// Entry point (biased).
    pub entry: u64,
    /// Load bias (0 for `ET_EXEC`).
    pub bias: u64,
    /// Address of the program headers in memory (`AT_PHDR`).
    pub phdr: u64,
    /// Number of program headers (`AT_PHNUM`).
    pub phnum: u64,
    /// Program header size (`AT_PHENT`).
    pub phent: u64,
    /// First page after the highest segment: the initial program break.
    pub brk: u64,
    /// The mapped pages, merged per segment.
    pub mappings: Vec<Mapping>,
    /// `false` if `PT_GNU_STACK` asks for a non-executable stack (the default).
    pub exec_stack: bool,
}

/// How many bytes from the start of a file [`Elf::parse_prefix`] needs, judging by `prefix`
/// (at least the ELF header, then the program headers, then the `PT_INTERP` path). Read that
/// many bytes and ask again until the answer stops growing; a malformed header gives an
/// answer no larger than `prefix`, and parsing then reports the error.
pub fn header_len(prefix: &[u8]) -> u64 {
    if prefix.len() < EHDR_SIZE || &prefix[..4] != b"\x7fELF" {
        return EHDR_SIZE as u64;
    }
    let (Some(phoff), Some(phnum)) = (u64_at(prefix, 32), u16_at(prefix, 56)) else {
        return EHDR_SIZE as u64;
    };
    let Some(table_end) = phoff.checked_add(u64::from(phnum) * PHDR_SIZE as u64) else {
        return prefix.len() as u64;
    };
    if table_end > prefix.len() as u64 {
        return table_end;
    }
    let mut need = table_end.max(EHDR_SIZE as u64);
    for i in 0..u64::from(phnum) {
        let at = (phoff + i * PHDR_SIZE as u64) as usize;
        if u32_at(prefix, at) == Some(PT_INTERP)
            && let (Some(off), Some(len)) = (u64_at(prefix, at + 8), u64_at(prefix, at + 32))
            && let Some(end) = off.checked_add(len)
        {
            need = need.max(end);
        }
    }
    need
}

/// File pages for [`load_with`]: `pages(start, end)` is a source whose page `k` holds the
/// file's bytes `start + k * PAGE_SIZE ..` up to `end` (exclusive), and zeros after `end`.
pub type FilePages<'p> = &'p mut dyn FnMut(u64, u64) -> Rc<dyn PageSource>;

/// One `PT_LOAD` segment, placed.
struct Seg {
    /// Mapped pages `[start, end)`.
    start: u64,
    end: u64,
    perms: Perms,
    /// File bytes `[file_start, file_end)` go to `start..data_end`; pages up to `file_pages`
    /// hold file bytes.
    file_start: u64,
    file_end: u64,
    data_end: u64,
    file_pages: u64,
}

/// What fills a run of pages.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fill {
    /// File pages of one segment, from its source.
    Lazy(usize),
    /// Zeros (BSS).
    Zero,
    /// Shared by several segments: mapped as zeros, then the segments' bytes are copied in.
    Shared,
}

/// Maps `elf`'s loadable segments into `mem` at `bias` (ignored for `ET_EXEC`), with the
/// file's contents from `pages` (see [`FilePages`]).
///
/// Pages shared by two segments get the union of their permissions. File bytes come from the
/// page-aligned file offset, as `mmap` of the file would show them, and memory past `filesz`
/// stays zero. A page that belongs to one segment is filled from its source on first touch
/// ([`GuestMemory::map_source`]); the rare page shared by two segments is filled now, with
/// later segments' bytes over earlier ones'.
pub fn load_with(
    mem: &mut dyn GuestMemory,
    elf: &Elf<'_>,
    bias: u64,
    pages: FilePages<'_>,
) -> Result<LoadedImage, ElfError> {
    let bias = if elf.is_dyn() { bias } else { 0 };
    let loads: Vec<&ProgramHeader> = elf.phdrs.iter().filter(|p| p.p_type == PT_LOAD).collect();

    let mut mappings: Vec<Mapping> = Vec::new();
    let mut segs: Vec<Seg> = Vec::new();
    for ph in &loads {
        let start = (ph.vaddr & !PAGE_MASK)
            .checked_add(bias)
            .ok_or(ElfError::BadSegment)?;
        let end = page_up(ph.vaddr.checked_add(ph.memsz).ok_or(ElfError::BadSegment)?)
            .checked_add(bias)
            .ok_or(ElfError::BadSegment)?;
        if end <= start {
            continue;
        }
        // `parse` checked these; `phdrs` is public, so they are checked again.
        let skew = ph.vaddr & PAGE_MASK;
        let file_start = ph.offset.checked_sub(skew).ok_or(ElfError::BadSegment)?;
        let file_end = ph
            .offset
            .checked_add(ph.filesz)
            .ok_or(ElfError::BadSegment)?;
        let data = file_end - file_start;
        if ph.filesz > ph.memsz {
            return Err(ElfError::BadSegment);
        }
        mappings.push(Mapping {
            start,
            end,
            perms: ph.perms(),
            offset: ph.offset & !PAGE_MASK,
        });
        segs.push(Seg {
            start,
            end,
            perms: ph.perms(),
            file_start,
            file_end,
            data_end: start.saturating_add(data),
            file_pages: start.saturating_add(page_up(data)).min(end),
        });
    }

    // Elementary intervals between all segment edges: within one, the covering segments (and
    // so the permissions and the fill) are the same. Adjacent equal intervals form one run.
    let mut edges: Vec<u64> = segs
        .iter()
        .flat_map(|s| [s.start, s.end, s.file_pages])
        .collect();
    edges.sort_unstable();
    edges.dedup();
    let mut runs: Vec<(u64, u64, Perms, Fill)> = Vec::new();
    for w in edges.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mut covering = segs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.start <= a && b <= s.end);
        let Some((first, s)) = covering.next() else {
            continue;
        };
        let mut perms = s.perms;
        let mut shared = false;
        for (_, other) in covering {
            perms = perms | other.perms;
            shared = true;
        }
        let fill = if shared {
            Fill::Shared
        } else if b <= s.file_pages {
            Fill::Lazy(first)
        } else {
            Fill::Zero
        };
        match runs.last_mut() {
            Some(last) if last.1 == a && last.2 == perms && last.3 == fill => last.1 = b,
            _ => runs.push((a, b, perms, fill)),
        }
    }

    let mut sources: Vec<Option<Rc<dyn PageSource>>> = segs.iter().map(|_| None).collect();
    let mut source = |i: usize, segs: &[Seg]| -> Rc<dyn PageSource> {
        sources[i]
            .get_or_insert_with(|| pages(segs[i].file_start, segs[i].file_end))
            .clone()
    };
    for &(a, b, perms, fill) in &runs {
        match fill {
            Fill::Lazy(i) => {
                let src = source(i, &segs);
                mem.map_source(a, b - a, perms, src, (a - segs[i].start) / PAGE_SIZE)?;
            }
            Fill::Zero | Fill::Shared => mem.map(a, b - a, perms)?,
        }
    }
    for &(a, b, _, fill) in &runs {
        if fill != Fill::Shared {
            continue;
        }
        for (i, s) in segs.iter().enumerate() {
            let (from, to) = (a.max(s.start), b.min(s.data_end));
            if from >= to {
                continue;
            }
            let src = source(i, &segs);
            let mut at = from;
            while at < to {
                let index = (at - s.start) / PAGE_SIZE;
                let page = src
                    .page(index)
                    .map_err(|_| ElfError::Map(MapError::OutOfRange))?;
                let off = ((at - s.start) % PAGE_SIZE) as usize;
                let n = (PAGE_SIZE as usize - off).min((to - at) as usize);
                mem.poke(at, &page[off..off + n])?;
                at += n as u64;
            }
        }
    }

    // `AT_PHDR`. `PT_PHDR` is not validated (Linux neither): its address wraps as Linux's
    // `p_vaddr + load_bias` does.
    let phdr = match elf.phdrs.iter().find(|p| p.p_type == PT_PHDR) {
        Some(p) => p.vaddr.wrapping_add(bias),
        None => loads
            .iter()
            .find(|p| {
                p.offset <= elf.phoff
                    && p.offset
                        .checked_add(p.filesz)
                        .is_some_and(|end| elf.phoff < end)
            })
            .map(|p| {
                p.vaddr
                    .wrapping_add(elf.phoff - p.offset)
                    .wrapping_add(bias)
            })
            .unwrap_or(0),
    };
    let brk = mappings.iter().map(|m| m.end).max().unwrap_or(0);
    let exec_stack = elf
        .phdrs
        .iter()
        .find(|p| p.p_type == PT_GNU_STACK)
        .is_some_and(|p| p.flags & 1 != 0);
    Ok(LoadedImage {
        entry: elf.entry.wrapping_add(bias),
        bias,
        phdr,
        phnum: elf.phdrs.len() as u64,
        phent: u64::from(elf.phentsize),
        brk,
        mappings,
        exec_stack,
    })
}

/// Pages of a file held in memory (see [`FilePages`]).
pub struct BytePages {
    bytes: Rc<[u8]>,
    start: u64,
    end: u64,
}

impl BytePages {
    /// Pages of `bytes[start..end]` (clipped to the bytes), zero-padded.
    pub fn new(bytes: Rc<[u8]>, start: u64, end: u64) -> Self {
        Self { bytes, start, end }
    }
}

impl PageSource for BytePages {
    fn page(&self, index: u64) -> Result<PageRef, PageSourceError> {
        let mut page = [0u8; PAGE_SIZE as usize];
        let from = index
            .checked_mul(PAGE_SIZE)
            .and_then(|o| o.checked_add(self.start))
            .ok_or(PageSourceError)?;
        let to = (from.saturating_add(PAGE_SIZE))
            .min(self.end)
            .min(self.bytes.len() as u64);
        if from < to {
            page[..(to - from) as usize].copy_from_slice(&self.bytes[from as usize..to as usize]);
        }
        Ok(Rc::new(page))
    }
}

/// Maps `elf`'s loadable segments into `mem` at `bias` (ignored for `ET_EXEC`), copying the
/// file bytes from the whole file `elf` was parsed from (see [`load_with`]).
pub fn load(mem: &mut dyn GuestMemory, elf: &Elf<'_>, bias: u64) -> Result<LoadedImage, ElfError> {
    if (elf.bytes.len() as u64) < elf.file_len {
        return Err(ElfError::Truncated);
    }
    let bytes: Rc<[u8]> = Rc::from(elf.bytes);
    load_with(mem, elf, bias, &mut |start, end| {
        Rc::new(BytePages::new(bytes.clone(), start, end))
    })
}

/// Everything the initial stack carries.
#[derive(Clone, Debug)]
pub struct StackSpec<'a> {
    /// `argv` strings (without NULs).
    pub argv: &'a [&'a [u8]],
    /// `envp` strings (without NULs).
    pub envp: &'a [&'a [u8]],
    /// The path passed to `execve` (`AT_EXECFN`).
    pub execfn: &'a [u8],
    /// `AT_PLATFORM` string (e.g. `aarch64`).
    pub platform: &'a [u8],
    /// The 16 bytes `AT_RANDOM` points to.
    pub random: [u8; 16],
    /// Auxiliary vector entries other than `AT_RANDOM`, `AT_EXECFN`, `AT_PLATFORM` and
    /// `AT_NULL`, which `build_stack` adds itself.
    pub auxv: &'a [(u64, u64)],
}

/// Writes the initial stack below `top` (exclusive) and returns the new stack pointer, which
/// points at `argc` and is 16-byte aligned. The stack pages must already be mapped.
///
/// Layout, from high to low addresses (as Linux builds it): an 8-byte zero end marker, the
/// `execfn` string, environment strings, argument strings, the platform string, the 16 random
/// bytes, padding, the auxiliary vector, `envp[]` + NULL, `argv[]` + NULL, `argc`.
pub fn build_stack(
    mem: &mut dyn GuestMemory,
    top: u64,
    spec: &StackSpec<'_>,
) -> Result<u64, ElfError> {
    // Addresses wrap rather than panic: a stack too small for the strings ends in a failed
    // write (an unmapped address), never in an overflow.
    let mut p = top.wrapping_sub(8);
    mem.poke(p, &[0; 8])?;
    let mut push_str = |mem: &mut dyn GuestMemory, s: &[u8]| -> Result<u64, ElfError> {
        p = p.wrapping_sub(s.len() as u64 + 1);
        mem.poke(p, s)?;
        mem.poke(p.wrapping_add(s.len() as u64), &[0])?;
        Ok(p)
    };
    let execfn = push_str(mem, spec.execfn)?;
    let mut envp = Vec::with_capacity(spec.envp.len());
    for s in spec.envp.iter().rev() {
        envp.push(push_str(mem, s)?);
    }
    envp.reverse();
    let mut argv = Vec::with_capacity(spec.argv.len());
    for s in spec.argv.iter().rev() {
        argv.push(push_str(mem, s)?);
    }
    argv.reverse();
    let platform = push_str(mem, spec.platform)?;
    p &= !15;
    p = p.wrapping_sub(16);
    let random = p;
    mem.poke(random, &spec.random)?;

    let mut auxv: Vec<(u64, u64)> = spec.auxv.to_vec();
    auxv.push((wasmshim_linux_abi::consts::auxv::AT_RANDOM, random));
    auxv.push((wasmshim_linux_abi::consts::auxv::AT_EXECFN, execfn));
    auxv.push((wasmshim_linux_abi::consts::auxv::AT_PLATFORM, platform));
    auxv.push((wasmshim_linux_abi::consts::auxv::AT_NULL, 0));

    let words = 1 + argv.len() + 1 + envp.len() + 1 + 2 * auxv.len();
    let mut sp = p.wrapping_sub(8 * words as u64) & !15;
    let start = sp;
    let mut table = Vec::with_capacity(words * 8);
    table.extend_from_slice(&(argv.len() as u64).to_le_bytes());
    for a in argv.iter().chain(core::iter::once(&0)) {
        table.extend_from_slice(&a.to_le_bytes());
    }
    for e in envp.iter().chain(core::iter::once(&0)) {
        table.extend_from_slice(&e.to_le_bytes());
    }
    for (k, v) in &auxv {
        table.extend_from_slice(&k.to_le_bytes());
        table.extend_from_slice(&v.to_le_bytes());
    }
    mem.poke(sp, &table)?;
    sp = start;
    Ok(sp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasmshim_cpu::PagedMemory;

    /// A minimal ELF: one RX segment at 0x400000 holding the headers and 8 bytes of code,
    /// and one RW segment with 4 file bytes and 0x2000 of BSS.
    fn tiny_elf() -> Vec<u8> {
        let mut f = alloc::vec![0u8; 0x1010];
        f[..4].copy_from_slice(b"\x7fELF");
        f[4] = 2;
        f[5] = 1;
        f[6] = 1;
        f[16..18].copy_from_slice(&ET_EXEC.to_le_bytes());
        f[18..20].copy_from_slice(&183u16.to_le_bytes());
        f[24..32].copy_from_slice(&0x4000b0u64.to_le_bytes());
        f[32..40].copy_from_slice(&64u64.to_le_bytes());
        f[54..56].copy_from_slice(&56u16.to_le_bytes());
        f[56..58].copy_from_slice(&2u16.to_le_bytes());
        let ph =
            |f: &mut [u8], at: usize, flags: u32, off: u64, vaddr: u64, filesz: u64, memsz: u64| {
                f[at..at + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
                f[at + 4..at + 8].copy_from_slice(&flags.to_le_bytes());
                f[at + 8..at + 16].copy_from_slice(&off.to_le_bytes());
                f[at + 16..at + 24].copy_from_slice(&vaddr.to_le_bytes());
                f[at + 32..at + 40].copy_from_slice(&filesz.to_le_bytes());
                f[at + 40..at + 48].copy_from_slice(&memsz.to_le_bytes());
            };
        ph(&mut f, 64, 5, 0, 0x400000, 0xb8, 0xb8);
        ph(&mut f, 120, 6, 0x1000, 0x411000, 4, 0x2000);
        f[0xb0..0xb8].copy_from_slice(&[0x1f, 0x20, 0x03, 0xd5, 0x1f, 0x20, 0x03, 0xd5]);
        f[0x1000..0x1004].copy_from_slice(b"data");
        f
    }

    #[test]
    fn loads_segments_like_binfmt_elf() {
        let bytes = tiny_elf();
        let elf = Elf::parse(&bytes).unwrap();
        assert_eq!(elf.machine, 183);
        let mut mem = PagedMemory::new();
        let img = load(&mut mem, &elf, 0).unwrap();
        assert_eq!(img.entry, 0x4000b0);
        assert_eq!(img.phdr, 0x400040);
        assert_eq!(img.brk, 0x413000);
        assert_eq!(mem.perms(0x400000), Some(Perms::RX));
        assert_eq!(mem.perms(0x412000), Some(Perms::RW));
        assert_eq!(mem.fetch(0x4000b0).unwrap(), 0xd503_201f);
        let mut b = [0u8; 6];
        mem.read(0x411000, &mut b).unwrap();
        assert_eq!(&b, b"data\0\0");
    }

    /// The loader before lazy pages (copy every segment's bytes in program-header order).
    fn eager_reference(mem: &mut dyn GuestMemory, elf: &Elf<'_>, bytes: &[u8]) {
        let loads: Vec<&ProgramHeader> = elf.phdrs.iter().filter(|p| p.p_type == PT_LOAD).collect();
        let mut by_page = alloc::collections::BTreeMap::new();
        for ph in &loads {
            let mut page = ph.vaddr & !PAGE_MASK;
            while page < page_up(ph.vaddr + ph.memsz) {
                let p: &mut Perms = by_page.entry(page).or_insert(ph.perms());
                *p = *p | ph.perms();
                page += PAGE_SIZE;
            }
        }
        for (page, perms) in by_page {
            mem.map(page, PAGE_SIZE, perms).unwrap();
        }
        for ph in &loads {
            let skew = ph.vaddr & PAGE_MASK;
            let (fs, fe) = (
                (ph.offset - skew) as usize,
                (ph.offset + ph.filesz) as usize,
            );
            mem.poke(ph.vaddr - skew, &bytes[fs..fe]).unwrap();
        }
    }

    /// Two segments sharing their first page: RX `[0, 0xc00)` at 0x400000, and RW file
    /// `[0xe00, 0x1200)` at 0x400e00 with BSS to 0x402600. The file is 0x2000 bytes.
    fn sharing_elf() -> Vec<u8> {
        let mut f: Vec<u8> = (0..0x2000).map(|i| (i % 251) as u8 + 1).collect();
        f[..64].fill(0);
        f[..4].copy_from_slice(b"\x7fELF");
        f[4] = 2;
        f[5] = 1;
        f[6] = 1;
        f[16..18].copy_from_slice(&ET_EXEC.to_le_bytes());
        f[18..20].copy_from_slice(&183u16.to_le_bytes());
        f[24..32].copy_from_slice(&0x400100u64.to_le_bytes());
        f[32..40].copy_from_slice(&64u64.to_le_bytes());
        f[54..56].copy_from_slice(&56u16.to_le_bytes());
        f[56..58].copy_from_slice(&2u16.to_le_bytes());
        let mut ph = |at: usize, flags: u32, off: u64, vaddr: u64, filesz: u64, memsz: u64| {
            f[at..at + 56].fill(0);
            f[at..at + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
            f[at + 4..at + 8].copy_from_slice(&flags.to_le_bytes());
            f[at + 8..at + 16].copy_from_slice(&off.to_le_bytes());
            f[at + 16..at + 24].copy_from_slice(&vaddr.to_le_bytes());
            f[at + 32..at + 40].copy_from_slice(&filesz.to_le_bytes());
            f[at + 40..at + 48].copy_from_slice(&memsz.to_le_bytes());
        };
        ph(64, 5, 0, 0x400000, 0xc00, 0xc00);
        ph(120, 6, 0xe00, 0x400e00, 0x400, 0x1800);
        f
    }

    #[test]
    fn lazy_loading_matches_copying_the_file() {
        for bytes in [tiny_elf(), sharing_elf()] {
            let elf = Elf::parse(&bytes).unwrap();
            let (lo, hi) = elf.span();
            let mut want = PagedMemory::new();
            eager_reference(&mut want, &elf, &bytes);
            let mut got = PagedMemory::new();
            load(&mut got, &elf, 0).unwrap();
            let mut a = alloc::vec![0u8; (hi - lo) as usize];
            let mut b = a.clone();
            let mut page = lo;
            while page < hi {
                assert_eq!(got.perms(page), want.perms(page), "{page:#x}");
                page += PAGE_SIZE;
            }
            for (m, out) in [(&want, &mut a), (&got, &mut b)] {
                let mut page = lo;
                while page < hi {
                    let at = (page - lo) as usize;
                    if let Some(p) = m.page_bytes(page) {
                        out[at..at + PAGE_SIZE as usize].copy_from_slice(p);
                    }
                    page += PAGE_SIZE;
                }
            }
            assert!(a == b);
        }
        // Only the shared page was copied in; the rest fills on touch.
        let bytes = sharing_elf();
        let elf = Elf::parse(&bytes).unwrap();
        let mut mem = PagedMemory::new();
        load(&mut mem, &elf, 0).unwrap();
        assert_eq!(mem.resident_pages(), 1);
        assert_eq!(mem.perms(0x400000), Some(Perms::RWX));
        assert_eq!(mem.read_u8(0x401000).unwrap(), (0x1000 % 251) as u8 + 1);
        assert_eq!(mem.read_u8(0x4011ff).unwrap(), (0x11ff % 251) as u8 + 1);
        assert_eq!(mem.read_u8(0x401200).unwrap(), 0);
        assert_eq!(mem.resident_pages(), 2);
    }

    #[test]
    fn parses_from_a_prefix() {
        let bytes = tiny_elf();
        assert_eq!(header_len(&bytes[..10]), 64);
        assert_eq!(header_len(&bytes[..64]), 64 + 2 * 56);
        assert_eq!(header_len(&bytes), 64 + 2 * 56);
        let prefix = &bytes[..64 + 2 * 56];
        let elf = Elf::parse_prefix(prefix, bytes.len() as u64).unwrap();
        assert_eq!(elf.file_len, 0x1010);
        assert!(matches!(
            Elf::parse_prefix(&bytes[..100], bytes.len() as u64),
            Err(ElfError::Truncated)
        ));
        // A segment past the end of the file.
        assert!(matches!(
            Elf::parse_prefix(prefix, 0x1002),
            Err(ElfError::Truncated)
        ));
        // `load` needs the whole file; `load_with` does not.
        let mut mem = PagedMemory::new();
        assert!(matches!(load(&mut mem, &elf, 0), Err(ElfError::Truncated)));
        let whole: Rc<[u8]> = Rc::from(&bytes[..]);
        let img = load_with(&mut mem, &elf, 0, &mut |s, e| {
            Rc::new(BytePages::new(whole.clone(), s, e))
        })
        .unwrap();
        assert_eq!(img.entry, 0x4000b0);
        assert_eq!(mem.fetch(0x4000b0).unwrap(), 0xd503_201f);
        // A PT_INTERP path beyond the prefix asks for more.
        let mut with_interp = bytes.clone();
        with_interp[120..124].copy_from_slice(&PT_INTERP.to_le_bytes());
        with_interp[128..136].copy_from_slice(&0x1000u64.to_le_bytes());
        with_interp[152..160].copy_from_slice(&8u64.to_le_bytes());
        assert_eq!(header_len(&with_interp[..176]), 0x1008);
        assert!(matches!(
            Elf::parse_prefix(&with_interp[..176], 0x1010),
            Err(ElfError::Truncated)
        ));
        let e = Elf::parse_prefix(&with_interp[..0x1008], 0x1010).unwrap();
        assert_eq!(e.interp(), Some(&b"data"[..]));
    }

    #[test]
    fn rejects_malformed_images() {
        assert!(matches!(Elf::parse(b"MZ"), Err(ElfError::NotElf)));
        let mut bytes = tiny_elf();
        bytes.truncate(0x1002);
        assert!(matches!(Elf::parse(&bytes), Err(ElfError::Truncated)));
        let mut bytes = tiny_elf();
        bytes[56] = 200; // phnum points past the end
        assert!(matches!(Elf::parse(&bytes), Err(ElfError::Truncated)));
        let mut bytes = tiny_elf();
        bytes[4] = 1; // ELFCLASS32
        assert!(matches!(Elf::parse(&bytes), Err(ElfError::Unsupported)));
    }

    /// Writes `v` at `at` in `f`.
    fn set(f: &mut [u8], at: usize, v: &[u8]) {
        f[at..at + v.len()].copy_from_slice(v);
    }

    /// Parses `bytes` and, if that succeeds, loads it: the outcome, which must never panic.
    /// (Images spanning more than 16 MiB are parsed only: mapping a huge executable range
    /// costs time per page in `PagedMemory`.)
    fn parse_and_load(bytes: &[u8]) -> Result<LoadedImage, ElfError> {
        let elf = Elf::parse(bytes)?;
        let (lo, hi) = elf.span();
        let _ = elf.interp();
        let _ = header_len(bytes);
        if hi.saturating_sub(lo) > 16 << 20 {
            return Err(ElfError::NoSegments);
        }
        load(&mut PagedMemory::new(), &elf, 0x5555_0000)
    }

    #[test]
    fn truncated_and_overflowing_headers_are_errors() {
        // Every prefix of a valid file, as a whole file and as a prefix of the whole.
        let good = tiny_elf();
        for n in 0..good.len() {
            let _ = parse_and_load(&good[..n]);
            let _ = Elf::parse_prefix(&good[..n], good.len() as u64);
            let _ = header_len(&good[..n]);
        }
        let max = u64::MAX.to_le_bytes();
        let cases: &[(usize, &[u8], ElfError)] = &[
            // e_phoff at the very end of the address range: the table is past the file.
            (32, &max, ElfError::Truncated),
            (32, &(u64::MAX - 3).to_le_bytes(), ElfError::Truncated),
            (
                32,
                &(usize::MAX as u64 - 55).to_le_bytes(),
                ElfError::Truncated,
            ),
            // A segment whose file range overflows.
            (64 + 8, &max, ElfError::Truncated),
            (64 + 32, &max, ElfError::Truncated),
            // A segment whose memory range overflows.
            (
                120 + 16,
                &(u64::MAX - 0xfff).to_le_bytes(),
                ElfError::BadSegment,
            ),
            (120 + 40, &max, ElfError::BadSegment),
        ];
        for &(at, v, want) in cases {
            let mut f = tiny_elf();
            set(&mut f, at, v);
            assert_eq!(parse_and_load(&f), Err(want), "field at {at}");
        }
        // Segments that pass `parse` but cannot be placed: memory past the address space.
        let mut f = tiny_elf();
        set(&mut f, 120 + 16, &(u64::MAX - 0x1fff_0fff).to_le_bytes());
        set(&mut f, 120 + 8, &0x1000u64.to_le_bytes());
        let elf = Elf::parse(&f).unwrap();
        assert!(matches!(
            load(&mut PagedMemory::new(), &elf, 0),
            Err(ElfError::Map(MapError::OutOfRange))
        ));
        // An unchecked PT_PHDR at the top of the address range: `AT_PHDR` wraps, as on Linux.
        let mut f = tiny_elf();
        set(&mut f, 56, &3u16.to_le_bytes());
        set(&mut f, 176, &PT_PHDR.to_le_bytes());
        set(&mut f, 176 + 16, &max);
        let img = parse_and_load(&f).unwrap();
        assert_eq!(img.phdr, u64::MAX);
        // Every byte of the headers set to the extremes, one at a time.
        for at in 0..176 {
            for b in [0x00, 0x7f, 0x80, 0xff] {
                let mut f = tiny_elf();
                f[at] = b;
                let _ = parse_and_load(&f);
            }
        }
    }

    #[test]
    fn a_stack_too_small_for_its_strings_is_an_error() {
        let mut mem = PagedMemory::new();
        mem.map(0, 0x1000, Perms::RW).unwrap();
        let big = alloc::vec![b'x'; 0x4000];
        let spec = StackSpec {
            argv: &[&big],
            envp: &[],
            execfn: b"/x",
            platform: b"aarch64",
            random: [0; 16],
            auxv: &[],
        };
        for top in [0, 8, 0x1000] {
            assert!(build_stack(&mut mem, top, &spec).is_err(), "top {top:#x}");
        }
    }

    #[test]
    fn stack_layout() {
        let mut mem = PagedMemory::new();
        let top = 0x8000_0000;
        mem.map(top - 0x4000, 0x4000, Perms::RW).unwrap();
        let spec = StackSpec {
            argv: &[b"/bin/hello", b"x"],
            envp: &[b"A=1"],
            execfn: b"/bin/hello",
            platform: b"aarch64",
            random: [7; 16],
            auxv: &[(wasmshim_linux_abi::consts::auxv::AT_PAGESZ, 4096)],
        };
        let sp = build_stack(&mut mem, top, &spec).unwrap();
        assert_eq!(sp % 16, 0);
        assert_eq!(mem.read_u64(sp).unwrap(), 2);
        let argv0 = mem.read_u64(sp + 8).unwrap();
        let mut s = [0u8; 11];
        mem.read(argv0, &mut s).unwrap();
        assert_eq!(&s, b"/bin/hello\0");
        assert_eq!(mem.read_u64(sp + 24).unwrap(), 0); // argv NULL
        let env0 = mem.read_u64(sp + 32).unwrap();
        let mut e = [0u8; 4];
        mem.read(env0, &mut e).unwrap();
        assert_eq!(&e, b"A=1\0");
        assert_eq!(mem.read_u64(sp + 40).unwrap(), 0); // envp NULL
        assert_eq!(
            mem.read_u64(sp + 48).unwrap(),
            wasmshim_linux_abi::consts::auxv::AT_PAGESZ
        );
        assert_eq!(mem.read_u64(sp + 56).unwrap(), 4096);
        assert_eq!(
            mem.read_u64(sp + 64).unwrap(),
            wasmshim_linux_abi::consts::auxv::AT_RANDOM
        );
        let random = mem.read_u64(sp + 72).unwrap();
        assert_eq!(mem.read_u8(random).unwrap(), 7);
    }
}
