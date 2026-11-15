//! A sparse, paged guest address space.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec::Vec;

use core::cell::{Cell, OnceCell};
use core::sync::atomic::{AtomicU64, Ordering};

use crate::memory::{
    Access, GuestMemory, MapError, MemFault, MemFaultKind, PAGE_SIZE, PageRef, PageSource,
    PageSourceError, PageView, Perms, SharedMemory,
};
use crate::shared::{SharedPages, shared_code_epoch};

/// Bits per radix level. Three levels of 12 bits cover 36 bits of page number, so 48-bit addresses.
const LEVEL_BITS: u32 = 12;
const LEVEL_LEN: usize = 1 << LEVEL_BITS;
const LEVEL_MASK: u64 = LEVEL_LEN as u64 - 1;
const PAGE_SHIFT: u32 = 12;
const PAGE_MASK: u64 = PAGE_SIZE - 1;
/// Addresses at or above this fault as unmapped.
const ADDRESS_LIMIT: u64 = 1 << (PAGE_SHIFT + 3 * LEVEL_BITS);

/// Page contents, shared copy-on-write between duplicated address spaces (`fork`) and with
/// page sources.
type PageData = PageRef;

/// Where a page's contents live.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Backing {
    /// In `data`; an empty `data` means "still all zeros".
    #[default]
    Anon,
    /// In `data` once filled; while `data` is empty, in the source range covering the page.
    Source,
    /// In the memory object of the shared range covering the page (`data` stays empty).
    Shared,
}

/// One page-table entry. `perms == None` means unmapped. An empty `data` means "still all
/// zeros" (pages are allocated on first write, so huge reservations cost almost nothing), or,
/// for a [`Backing::Source`] page, "not read from its source yet" (filled on first touch, see
/// [`GuestMemory::map_source`]). `data` is filled through `&self` when a read needs it. Only
/// pages with something in `data` take the fast paths; the others go through the slow paths,
/// which look at `backing`.
#[derive(Clone, Default)]
struct Slot {
    perms: Option<Perms>,
    backing: Backing,
    /// Bumped whenever this page's code may have changed (see [`GuestMemory::code_version`]).
    code_gen: u32,
    data: OnceCell<PageData>,
}

/// Pages `[start, end)` (the map key is `start`) are pages `first..` of `of`: a page source
/// ([`GuestMemory::map_source`]) or a memory object ([`GuestMemory::map_shared`]).
#[derive(Clone)]
struct Ranged<T> {
    end: u64,
    of: T,
    first: u64,
}

type SourceRange = Ranged<Rc<dyn PageSource>>;
type SharedRange = Ranged<Rc<SharedPages>>;
/// A lazy reservation: pages `[start, end)` are mapped anonymous with these (never executable)
/// permissions and have no slots (`first` is unused). See [`LAZY_MIN_PAGES`].
type LazyRange = Ranged<Perms>;

/// Anonymous, non-executable mappings of at least this many pages are kept as one range record
/// ([`LazyRange`]) instead of a slot per page, so reservations of gigabytes or a terabyte
/// (`PROT_NONE` cages, allocator pools, guard regions) cost nothing until they are written.
/// Every page is in at most one of: a lazy range, or a slot with permissions.
const LAZY_MIN_PAGES: u64 = 64;
/// A write (or an executable `protect`) gives a lazy page its slot together with the other lazy
/// pages of its aligned block of this many pages, so touched ranges split at most once per block.
const LAZY_CHUNK: u64 = 64;

/// The index of `p` in [`PagedMemory::templates`].
fn perms_index(p: Perms) -> usize {
    usize::from(p.read()) | usize::from(p.write()) << 1 | usize::from(p.exec()) << 2
}

/// Removes the ranges covering pages `[start, end)` from `map`, splitting ranges at the edges.
fn cut_ranges<T: Clone>(map: &mut BTreeMap<u64, Ranged<T>>, start: u64, end: u64) {
    let mut hits = Vec::new();
    for (&s, r) in map.range(..end).rev() {
        if r.end <= start {
            break;
        }
        hits.push(s);
    }
    for s in hits {
        let Some(r) = map.remove(&s) else {
            continue;
        };
        if s < start {
            map.insert(
                s,
                Ranged {
                    end: start,
                    ..r.clone()
                },
            );
        }
        if r.end > end {
            map.insert(
                end,
                Ranged {
                    end: r.end,
                    first: r.first + (end - s),
                    of: r.of,
                },
            );
        }
    }
}

/// The range of `map` covering page `vpn`, and the page's index in what it maps.
fn range_at<T>(map: &BTreeMap<u64, Ranged<T>>, vpn: u64) -> Option<(&T, u64)> {
    let (&start, r) = map.range(..=vpn).next_back()?;
    (vpn < r.end).then(|| (&r.of, r.first + (vpn - start)))
}

type Leaf = Box<[Slot]>;
type Mid = Box<[Option<Leaf>]>;

/// Page `vpn`'s entry in the page table `root`.
fn slot_in(root: &mut [Option<Mid>], vpn: u64) -> Option<&mut Slot> {
    let (a, b, c) = split(vpn);
    let leaf = root[a].as_mut()?[b].as_mut()?;
    Some(&mut leaf[c])
}

/// A page as the slow paths access it.
enum Target<'a> {
    /// This address space's own page (`resident` counts a page that becomes resident).
    Own {
        slot: &'a mut Slot,
        resident: &'a mut usize,
    },
    /// Page `index` of a memory object.
    Shared {
        slot: &'a mut Slot,
        obj: &'a SharedPages,
        index: u64,
    },
}

/// A sparse 48-bit guest address space: a three-level radix tree of 4 KiB pages with
/// per-page permissions, zero-fill-on-demand, pages filled from a [`PageSource`] on first
/// touch, ranges shared with [`SharedPages`] objects, and copy-on-write duplication.
pub struct PagedMemory {
    root: Box<[Option<Mid>]>,
    /// Source ranges by first page number; they never overlap.
    sources: BTreeMap<u64, SourceRange>,
    /// Shared ranges by first page number; they never overlap.
    shared: BTreeMap<u64, SharedRange>,
    /// Lazy reservations by first page number; they never overlap, and their pages have no
    /// slot with permissions.
    lazy: BTreeMap<u64, LazyRange>,
    /// What a page of a lazy range looks like, by [`perms_index`]: its slot until it gets one.
    templates: [Slot; 8],
    /// Unique per address space (and per duplicate): code versions never collide across them.
    instance: u64,
    epoch: u64,
    /// Pages whose code version moved since [`Self::take_code_changes`] last ran (with the
    /// epoch moves they caused), for engines that keep per-page state; `None` once more
    /// than [`CODE_LOG_MAX`] pages changed (then everything counts as changed).
    code_log: Option<Vec<u64>>,
    resident: Cell<usize>,
    /// The debugger's watch view: pages whose slot permissions are lowered while it
    /// is set, with the guest's own permissions to put back.
    watch: Vec<(u64, Perms)>,
}

/// Backs [`PagedMemory::page_bytes`] reads of pages that were never written.
static ZERO_PAGE: [u8; PAGE_SIZE as usize] = [0; PAGE_SIZE as usize];

/// Source of [`PagedMemory`] instance numbers (only uniqueness matters).
static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);

fn next_instance() -> u64 {
    NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed)
}

impl Clone for PagedMemory {
    /// A copy-on-write copy with its own instance number (so its code versions are its own).
    fn clone(&self) -> Self {
        let instance = next_instance();
        let mut copy = Self {
            root: self.root.clone(),
            sources: self.sources.clone(),
            shared: self.shared.clone(),
            lazy: self.lazy.clone(),
            templates: templates(),
            instance,
            epoch: instance << 32,
            code_log: Some(Vec::new()),
            resident: Cell::new(self.resident.get()),
            watch: Vec::new(),
        };
        // The copy has the guest's permissions, never the debugger's view.
        for &(vpn, saved) in &self.watch {
            if let Some(slot) = copy.slot_mut(vpn) {
                slot.perms = Some(saved);
            }
        }
        copy
    }
}

impl Default for PagedMemory {
    fn default() -> Self {
        Self::new()
    }
}

fn empty_slice<T>(make: impl FnMut() -> T) -> Box<[T]> {
    core::iter::repeat_with(make)
        .take(LEVEL_LEN)
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

/// One slot per permission set: anonymous, never written.
fn templates() -> [Slot; 8] {
    core::array::from_fn(|i| Slot {
        perms: Some(Perms::new(i & 1 != 0, i & 2 != 0, i & 4 != 0)),
        ..Slot::default()
    })
}

fn split(vpn: u64) -> (usize, usize, usize) {
    (
        ((vpn >> (2 * LEVEL_BITS)) & LEVEL_MASK) as usize,
        ((vpn >> LEVEL_BITS) & LEVEL_MASK) as usize,
        (vpn & LEVEL_MASK) as usize,
    )
}

/// Validates a page-aligned range and returns its page numbers.
fn page_range(addr: u64, len: u64) -> Result<core::ops::Range<u64>, MapError> {
    if addr & PAGE_MASK != 0 || len & PAGE_MASK != 0 || len == 0 {
        return Err(MapError::Unaligned);
    }
    let end = addr.checked_add(len).ok_or(MapError::OutOfRange)?;
    if end > ADDRESS_LIMIT {
        return Err(MapError::OutOfRange);
    }
    Ok(addr >> PAGE_SHIFT..end >> PAGE_SHIFT)
}

/// Pages [`PagedMemory`]'s code-change log holds before it gives up (everything changed).
pub const CODE_LOG_MAX: usize = 64;

impl PagedMemory {
    /// Notes that the code of pages `vpns` may have changed (with the epoch move it causes).
    fn log_code(&mut self, vpns: core::ops::Range<u64>) {
        if let Some(log) = &mut self.code_log {
            if log.len() + (vpns.end - vpns.start) as usize > CODE_LOG_MAX {
                self.code_log = None;
            } else {
                log.extend(vpns);
            }
        }
    }

    /// The pages (page numbers) whose code version moved since the last call, or `None` if
    /// too many did to list (treat every page as changed). Engines that keep state per code
    /// page (a chain table) use it to keep the rest when [`GuestMemory::code_epoch`] moves.
    /// Pages of shared objects stored to elsewhere are not listed: their moves show in
    /// [`crate::SharedPages::code_epoch`].
    pub fn take_code_changes(&mut self) -> Option<Vec<u64>> {
        self.code_log.replace(Vec::new())
    }

    /// An empty address space.
    pub fn new() -> Self {
        let instance = next_instance();
        Self {
            root: empty_slice(|| None),
            sources: BTreeMap::new(),
            shared: BTreeMap::new(),
            lazy: BTreeMap::new(),
            templates: templates(),
            instance,
            epoch: instance << 32,
            code_log: Some(Vec::new()),
            resident: Cell::new(0),
            watch: Vec::new(),
        }
    }

    /// The pages of the watch view (page addresses), for caches that must drop them.
    pub fn watched_pages(&self) -> impl Iterator<Item = u64> + '_ {
        self.watch.iter().map(|&(vpn, _)| vpn << PAGE_SHIFT)
    }

    /// Puts the guest's own permissions back on the watched pages (no view).
    fn clear_watch(&mut self) {
        for (vpn, saved) in core::mem::take(&mut self.watch) {
            if let Some(slot) = self.slot_mut(vpn) {
                slot.perms = Some(saved);
            }
        }
    }

    /// Number of pages that hold data (written at least once, or filled from their source).
    pub fn resident_pages(&self) -> usize {
        self.resident.get()
    }

    /// Page `vpn` as reads see it: its own slot, or for a page of a lazy range that range's
    /// template (permissions, all zeros).
    #[inline]
    fn slot(&self, vpn: u64) -> Option<&Slot> {
        match self.own_slot(vpn) {
            Some(s) if s.perms.is_some() => Some(s),
            own => self.lazy_slot(vpn).or(own),
        }
    }

    /// Page `vpn`'s entry in the page table, if its leaf exists (lazy ranges not consulted).
    #[inline]
    fn own_slot(&self, vpn: u64) -> Option<&Slot> {
        let (a, b, c) = split(vpn);
        let leaf = self.root[a].as_ref()?[b].as_ref()?;
        Some(&leaf[c])
    }

    /// The template of the lazy range holding page `vpn`.
    fn lazy_slot(&self, vpn: u64) -> Option<&Slot> {
        if self.lazy.is_empty() {
            return None;
        }
        range_at(&self.lazy, vpn).map(|(&p, _)| &self.templates[perms_index(p)])
    }

    /// `true` if page `vpn` has a slot with permissions (it is mapped and not lazy).
    fn own_mapped(&self, vpn: u64) -> bool {
        self.own_slot(vpn).is_some_and(|s| s.perms.is_some())
    }

    /// Page `vpn`'s own slot (lazy pages have none until [`Self::materialize`]).
    fn slot_mut(&mut self, vpn: u64) -> Option<&mut Slot> {
        slot_in(&mut self.root, vpn)
    }

    /// Gives the lazy pages of `vpn`'s aligned block of [`LAZY_CHUNK`] pages in `vpn`'s lazy
    /// range their own slots (fresh anonymous pages with the range's permissions). A slot left
    /// from an earlier mapping keeps its code generation, so code versions never repeat.
    fn materialize(&mut self, vpn: u64) {
        let Some((&start, r)) = self.lazy.range(..=vpn).next_back() else {
            return;
        };
        if vpn >= r.end {
            return;
        }
        let perms = r.of;
        let lo = (vpn & !(LAZY_CHUNK - 1)).max(start);
        let hi = ((vpn | (LAZY_CHUNK - 1)) + 1).min(r.end);
        cut_ranges(&mut self.lazy, lo, hi);
        for v in lo..hi {
            let slot = self.slot_create(v);
            debug_assert!(slot.data.get().is_none() && slot.perms.is_none());
            slot.perms = Some(perms);
            slot.backing = Backing::Anon;
        }
    }

    /// [`Self::materialize`]s every lazy page of `pages`.
    fn materialize_range(&mut self, pages: core::ops::Range<u64>) {
        let mut vpn = pages.start;
        while vpn < pages.end {
            let hit = self
                .lazy
                .range(..=vpn)
                .next_back()
                .filter(|(_, r)| r.end > vpn)
                .or_else(|| self.lazy.range(vpn..pages.end).next());
            let Some((&s, _)) = hit else {
                return;
            };
            let next = s.max(vpn);
            self.materialize(next);
            vpn = (next | (LAZY_CHUNK - 1)) + 1;
        }
    }

    /// Calls `f(slot)` for every page of `pages` that has a page-table entry, skipping absent
    /// tables, so a huge range costs what is allocated in it.
    fn for_each_own_slot(&mut self, pages: core::ops::Range<u64>, mut f: impl FnMut(&mut Slot)) {
        let mut vpn = pages.start;
        while vpn < pages.end {
            let (a, b, _) = split(vpn);
            let leaf_end = (vpn | LEVEL_MASK) + 1;
            let mid_end = (vpn | ((1 << (2 * LEVEL_BITS)) - 1)) + 1;
            let Some(mid) = self.root[a].as_mut() else {
                vpn = mid_end;
                continue;
            };
            let stop = leaf_end.min(pages.end);
            if let Some(leaf) = mid[b].as_mut() {
                for v in vpn..stop {
                    f(&mut leaf[(v & LEVEL_MASK) as usize]);
                }
            }
            vpn = stop;
        }
    }

    /// Unmaps the own slots of `pages` (dropping their contents) and drops the pages' source,
    /// shared and lazy ranges. Returns whether code changed.
    fn clear(&mut self, pages: core::ops::Range<u64>) -> bool {
        cut_ranges(&mut self.sources, pages.start, pages.end);
        cut_ranges(&mut self.shared, pages.start, pages.end);
        cut_ranges(&mut self.lazy, pages.start, pages.end);
        let mut code = false;
        let mut dropped = 0;
        self.for_each_own_slot(pages, |slot| {
            if slot.data.take().is_some() {
                dropped += 1;
            }
            if slot.perms.is_some_and(Perms::exec) {
                slot.code_gen = slot.code_gen.wrapping_add(1);
                code = true;
            }
            slot.perms = None;
            slot.backing = Backing::Anon;
        });
        self.resident.set(self.resident.get() - dropped);
        code
    }

    /// `true` if every page of `pages` is mapped (own slot or lazy range).
    fn all_mapped(&self, pages: core::ops::Range<u64>) -> bool {
        let mut vpn = pages.start;
        while vpn < pages.end {
            if let Some((&s, r)) = self.lazy.range(..=vpn).next_back()
                && vpn >= s
                && vpn < r.end
            {
                vpn = r.end;
                continue;
            }
            if !self.own_mapped(vpn) {
                return false;
            }
            vpn += 1;
        }
        true
    }

    fn slot_create(&mut self, vpn: u64) -> &mut Slot {
        let (a, b, c) = split(vpn);
        let mid = self.root[a].get_or_insert_with(|| empty_slice(|| None));
        let leaf = mid[b].get_or_insert_with(|| empty_slice(Slot::default));
        &mut leaf[c]
    }

    /// The contents of `slot` (page `vpn`), reading a lazy page from its source on first use.
    /// `Ok(None)`: all zeros. Not for [`Backing::Shared`] pages (their bytes are in the object;
    /// this reports them as an error).
    #[inline]
    fn contents<'s>(
        &self,
        vpn: u64,
        slot: &'s Slot,
    ) -> Result<Option<&'s PageData>, PageSourceError> {
        match (slot.data.get(), slot.backing) {
            (Some(page), _) => Ok(Some(page)),
            (None, Backing::Anon) => Ok(None),
            (None, Backing::Source) => self.fill(vpn, slot).map(Some),
            (None, Backing::Shared) => Err(PageSourceError),
        }
    }

    /// The memory object page that shared page `vpn` maps.
    fn shared_page(&self, vpn: u64) -> Option<(&SharedPages, u64)> {
        range_at(&self.shared, vpn).map(|(o, i)| (&**o, i))
    }

    /// Reads lazy page `vpn` from its source into `slot`.
    #[cold]
    fn fill<'s>(&self, vpn: u64, slot: &'s Slot) -> Result<&'s PageData, PageSourceError> {
        let (source, index) = range_at(&self.sources, vpn).ok_or(PageSourceError)?;
        let page = source.page(index)?;
        let mut fresh = false;
        let data = slot.data.get_or_init(|| {
            fresh = true;
            page
        });
        if fresh {
            self.resident.set(self.resident.get() + 1);
        }
        Ok(data)
    }

    /// Makes page `vpn`'s bytes present: reads a lazy page from its source, allocates a shared
    /// page in its object. `false` if that failed.
    fn ensure_filled(&self, vpn: u64) -> bool {
        match self.slot(vpn) {
            Some(slot) if slot.data.get().is_none() => match slot.backing {
                Backing::Anon => true,
                Backing::Source => self.fill(vpn, slot).is_ok(),
                Backing::Shared => self
                    .shared_page(vpn)
                    .is_some_and(|(o, i)| o.materialize(i).is_ok()),
            },
            _ => true,
        }
    }

    /// Walks `[addr, addr+len)` page by page. `f(page, page_offset, chunk_start, chunk_len)`
    /// runs for each page after `check` accepted its permissions, with a lazy page already
    /// read from its source and a shared page allocated in its object.
    fn walk(
        &mut self,
        addr: u64,
        len: usize,
        access: Access,
        ignore_perms: bool,
        mut f: impl FnMut(Target<'_>, usize, usize, usize),
    ) -> Result<(), MemFault> {
        let mut done = 0usize;
        let mut resident = 0usize;
        let result = loop {
            if done >= len {
                break Ok(());
            }
            let at = addr.wrapping_add(done as u64);
            let fault = |kind| MemFault {
                addr: at,
                access,
                kind,
            };
            if at >= ADDRESS_LIMIT || at < addr {
                break Err(fault(MemFaultKind::Unmapped));
            }
            let vpn = at >> PAGE_SHIFT;
            let offset = (at & PAGE_MASK) as usize;
            let chunk = (PAGE_SIZE as usize - offset).min(len - done);
            let Some(perms) = self.slot(vpn).and_then(|s| s.perms) else {
                break Err(fault(MemFaultKind::Unmapped));
            };
            if !ignore_perms && !perms.allows(access) {
                break Err(fault(MemFaultKind::Protection));
            }
            if !self.own_mapped(vpn) {
                self.materialize(vpn);
            }
            if !self.ensure_filled(vpn) {
                break Err(fault(MemFaultKind::Unmapped));
            }
            // Disjoint borrows: the page table and the shared ranges.
            let shared = range_at(&self.shared, vpn);
            let Some(slot) = slot_in(&mut self.root, vpn) else {
                break Err(fault(MemFaultKind::Unmapped));
            };
            let target = match (slot.backing, shared) {
                (Backing::Shared, Some((obj, index))) => Target::Shared { slot, obj, index },
                (Backing::Shared, None) => break Err(fault(MemFaultKind::Unmapped)),
                _ => Target::Own {
                    slot,
                    resident: &mut resident,
                },
            };
            f(target, offset, done, chunk);
            done += chunk;
        };
        self.resident.set(self.resident.get() + resident);
        result
    }

    fn store(&mut self, addr: u64, data: &[u8], ignore_perms: bool) -> Result<(), MemFault> {
        let mut touched_code = false;
        let result = self.walk(
            addr,
            data.len(),
            Access::Write,
            ignore_perms,
            |target, offset, start, chunk| {
                let bytes = &data[start..start + chunk];
                match target {
                    Target::Own { slot, resident } => {
                        if slot.perms.is_some_and(Perms::exec) {
                            touched_code = true;
                            slot.code_gen = slot.code_gen.wrapping_add(1);
                        }
                        if let Some(page) = writable(slot, resident) {
                            page[offset..offset + chunk].copy_from_slice(bytes);
                        }
                    }
                    Target::Shared { slot, obj, index } => {
                        if slot.perms.is_some_and(Perms::exec) {
                            touched_code = true;
                            slot.code_gen = slot.code_gen.wrapping_add(1);
                        }
                        // The page is allocated (`walk`), so this cannot fail.
                        let _ = obj.with_page_mut(index, |p| {
                            p[offset..offset + chunk].copy_from_slice(bytes);
                        });
                    }
                }
            },
        );
        if touched_code {
            self.epoch += 1;
            let last = addr.saturating_add(data.len().max(1) as u64 - 1);
            self.log_code(addr >> PAGE_SHIFT..(last >> PAGE_SHIFT) + 1);
        }
        result
    }

    /// The bytes of the page holding `addr` if `[addr, addr+N)` lies within one page that
    /// allows `access` and needs no filling. `Ok(None)` means "readable, still all zeros".
    #[inline]
    fn in_page<const N: usize>(&self, addr: u64, access: Access) -> Option<Option<&[u8]>> {
        let offset = (addr & PAGE_MASK) as usize;
        if offset + N > PAGE_SIZE as usize || addr >= ADDRESS_LIMIT {
            return None;
        }
        let slot = self.slot(addr >> PAGE_SHIFT)?;
        if !slot.perms?.allows(access) {
            return None;
        }
        match slot.data.get() {
            Some(p) => Some(Some(&p[offset..offset + N])),
            None if slot.backing == Backing::Anon => Some(None),
            None => None,
        }
    }

    /// Calls `f(address, bytes)` for every resident page (written at least once, or read from
    /// its source), in address order (snapshots and measurements).
    pub fn for_each_resident(&self, mut f: impl FnMut(u64, &[u8])) {
        for (a, mid) in self.root.iter().enumerate() {
            let Some(mid) = mid else { continue };
            for (b, leaf) in mid.iter().enumerate() {
                let Some(leaf) = leaf else { continue };
                for (c, slot) in leaf.iter().enumerate() {
                    if let Some(page) = slot.data.get() {
                        let vpn = ((a as u64) << (2 * LEVEL_BITS))
                            | ((b as u64) << LEVEL_BITS)
                            | c as u64;
                        f(vpn << PAGE_SHIFT, &page[..]);
                    }
                }
            }
        }
    }

    /// The contents of the mapped page holding `addr`, whatever its permissions (`None` if it
    /// is unmapped, or its source failed; a page never written reads as zeros). For code caches
    /// that key translations by page content. `None` for a shared page too: its bytes can change
    /// through other address spaces at any time, so it is never a stable key.
    pub fn page_bytes(&self, addr: u64) -> Option<&[u8]> {
        if addr >= ADDRESS_LIMIT {
            return None;
        }
        let vpn = addr >> PAGE_SHIFT;
        let slot = self.slot(vpn)?;
        slot.perms?;
        Some(match self.contents(vpn, slot).ok()? {
            Some(page) => &page[..],
            None => &ZERO_PAGE[..],
        })
    }

    /// The host address of the page holding `addr` for reading, for engines whose generated
    /// code reads guest pages directly through a software TLB (the DBT inside its own Wasm
    /// instance). The page must be readable. A page never written yet is backed by a shared
    /// all-zero page; a lazy page is read from its source first.
    ///
    /// The address stays valid until the next operation on this address space that may
    /// replace the page: a write or poke (the page may become resident or private),
    /// `map`/`unmap`/`protect`, or a [`PagedMemory::host_write_page`] of the same page.
    pub fn host_read_page(&self, addr: u64) -> Option<usize> {
        if addr >= ADDRESS_LIMIT {
            return None;
        }
        let vpn = addr >> PAGE_SHIFT;
        let slot = self.slot(vpn)?;
        if !slot.perms?.read() {
            return None;
        }
        if slot.backing == Backing::Shared && slot.data.get().is_none() {
            let (obj, index) = self.shared_page(vpn)?;
            return obj.host_page(index).ok();
        }
        Some(match self.contents(vpn, slot).ok()? {
            Some(page) => Rc::as_ptr(page) as *const u8 as usize,
            None => ZERO_PAGE.as_ptr() as usize,
        })
    }

    /// The host address of the page holding `addr` for writing (see
    /// [`PagedMemory::host_read_page`]). The page must be writable and **not executable**
    /// (stores into code must take the slow path, which keeps [`GuestMemory::code_epoch`]
    /// exact). The page becomes resident and private to this address space (copy-on-write is
    /// resolved), except a shared page, which stays its object's (and is refused once the object
    /// has been mapped executable anywhere). The address stays valid until
    /// `map`/`unmap`/`protect`, or until [`GuestMemory::duplicate`] shares the page again.
    pub fn host_write_page(&mut self, addr: u64) -> Option<usize> {
        if addr >= ADDRESS_LIMIT {
            return None;
        }
        let vpn = addr >> PAGE_SHIFT;
        let perms = self.slot(vpn)?.perms?;
        if !perms.write() || perms.exec() {
            return None;
        }
        if !self.own_mapped(vpn) {
            self.materialize(vpn);
        }
        if !self.ensure_filled(vpn) {
            return None;
        }
        if self.slot(vpn)?.backing == Backing::Shared {
            let (obj, index) = self.shared_page(vpn)?;
            return if obj.exec() {
                None
            } else {
                obj.host_write_page(index).ok()
            };
        }
        let mut fresh = 0;
        let ptr = writable(self.slot_mut(vpn)?, &mut fresh)?.as_mut_ptr() as usize;
        self.resident.set(self.resident.get() + fresh);
        Some(ptr)
    }

    /// `true` if the page holding `addr` is mapped shared (a memory object's page): a direct
    /// write entry for it must go when [`SharedPages::write_epoch`] moves.
    pub fn is_shared(&self, addr: u64) -> bool {
        addr < ADDRESS_LIMIT
            && self
                .slot(addr >> PAGE_SHIFT)
                .is_some_and(|s| s.backing == Backing::Shared)
    }

    /// Stores `bytes` at `addr` if that is a plain write within one resident, writable,
    /// non-executable page (the common case). `false`: take the general path.
    #[inline]
    fn store_fast(&mut self, addr: u64, bytes: &[u8]) -> bool {
        let n = bytes.len();
        let offset = (addr & PAGE_MASK) as usize;
        if offset + n > PAGE_SIZE as usize || addr >= ADDRESS_LIMIT {
            return false;
        }
        let Some(slot) = self.slot_mut(addr >> PAGE_SHIFT) else {
            return false;
        };
        match (slot.perms, slot.data.get_mut()) {
            (Some(p), Some(page)) if p.write() && !p.exec() => {
                Rc::make_mut(page)[offset..offset + n].copy_from_slice(bytes);
                true
            }
            _ => false,
        }
    }

    #[inline]
    fn load<const N: usize>(&self, addr: u64) -> Result<[u8; N], MemFault> {
        let mut out = [0; N];
        match self.in_page::<N>(addr, Access::Read) {
            Some(Some(bytes)) => out.copy_from_slice(bytes),
            Some(None) => {}
            None => self.read(addr, &mut out)?,
        }
        Ok(out)
    }

    /// Sets `[addr, addr+len)` to fresh pages with `perms` and `backing` (whose source or shared
    /// range the caller records), dropping their contents, source and shared ranges.
    fn replace(
        &mut self,
        addr: u64,
        len: u64,
        perms: Perms,
        backing: Backing,
    ) -> Result<(), MapError> {
        self.clear_watch();
        let pages = page_range(addr, len)?;
        if backing == Backing::Anon && !perms.exec() && pages.end - pages.start >= LAZY_MIN_PAGES {
            // The new pages are never code, so only pages that were code change code.
            if self.clear(pages.clone()) {
                self.epoch += 1;
                self.log_code(pages.clone());
            }
            self.lazy.insert(
                pages.start,
                Ranged {
                    end: pages.end,
                    of: perms,
                    first: 0,
                },
            );
            return Ok(());
        }
        cut_ranges(&mut self.sources, pages.start, pages.end);
        cut_ranges(&mut self.shared, pages.start, pages.end);
        cut_ranges(&mut self.lazy, pages.start, pages.end);
        let mut code = false;
        let mut dropped = 0;
        for vpn in pages.clone() {
            let slot = self.slot_create(vpn);
            if slot.data.take().is_some() {
                dropped += 1;
            }
            // A fresh page replaces whatever was there: code changed if either side is code.
            if slot.perms.is_some_and(Perms::exec) || perms.exec() {
                slot.code_gen = slot.code_gen.wrapping_add(1);
                code = true;
            }
            slot.perms = Some(perms);
            slot.backing = backing;
        }
        self.resident.set(self.resident.get() - dropped);
        if code {
            self.epoch += 1;
            self.log_code(pages);
        }
        Ok(())
    }
}

/// The page of `slot`, made resident (`resident` counts a fresh one) and private.
/// (Always `Some`.)
fn writable<'s>(
    slot: &'s mut Slot,
    resident: &mut usize,
) -> Option<&'s mut [u8; PAGE_SIZE as usize]> {
    if slot.data.get().is_none() {
        *resident += 1;
        let _ = slot.data.set(Rc::new([0; PAGE_SIZE as usize]));
    }
    slot.data.get_mut().map(Rc::make_mut)
}

/// Hands pages to a [`GuestMemory::for_each_mapped`] callback, joining contiguous zero pages
/// with the same permissions into one run.
struct ZeroRuns<'f> {
    f: &'f mut dyn FnMut(u64, u64, Perms, PageView<'_>),
    /// First page, page count, permissions.
    run: Option<(u64, u64, Perms)>,
}

impl ZeroRuns<'_> {
    fn page(&mut self, vpn: u64, n: u64, perms: Perms, view: PageView<'_>) {
        if let PageView::Zero = view {
            if let Some((s, len, p)) = &mut self.run
                && *p == perms
                && *s + *len == vpn
            {
                *len += n;
                return;
            }
            self.flush();
            self.run = Some((vpn, n, perms));
            return;
        }
        self.flush();
        (self.f)(vpn << PAGE_SHIFT, n, perms, view);
    }

    fn flush(&mut self) {
        if let Some((s, len, p)) = self.run.take() {
            (self.f)(s << PAGE_SHIFT, len, p, PageView::Zero);
        }
    }
}

impl GuestMemory for PagedMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        let mut done = 0usize;
        while done < buf.len() {
            let at = addr.wrapping_add(done as u64);
            let fault = |kind| MemFault {
                addr: at,
                access: Access::Read,
                kind,
            };
            if at >= ADDRESS_LIMIT || at < addr {
                return Err(fault(MemFaultKind::Unmapped));
            }
            let vpn = at >> PAGE_SHIFT;
            let offset = (at & PAGE_MASK) as usize;
            let chunk = (PAGE_SIZE as usize - offset).min(buf.len() - done);
            let slot = self.slot(vpn).ok_or(fault(MemFaultKind::Unmapped))?;
            let perms = slot.perms.ok_or(fault(MemFaultKind::Unmapped))?;
            if !perms.read() {
                return Err(fault(MemFaultKind::Protection));
            }
            let out = &mut buf[done..done + chunk];
            if slot.backing == Backing::Shared {
                let read = self
                    .shared_page(vpn)
                    .ok_or(PageSourceError)
                    .and_then(|(o, i)| {
                        o.with_page(i, |p| out.copy_from_slice(&p[offset..offset + chunk]))
                    });
                if read.is_err() {
                    return Err(fault(MemFaultKind::Unmapped));
                }
                done += chunk;
                continue;
            }
            match self.contents(vpn, slot) {
                Ok(Some(page)) => out.copy_from_slice(&page[offset..offset + chunk]),
                Ok(None) => out.fill(0),
                Err(_) => return Err(fault(MemFaultKind::Unmapped)),
            }
            done += chunk;
        }
        Ok(())
    }

    fn write(&mut self, addr: u64, data: &[u8]) -> Result<(), MemFault> {
        if self.store_fast(addr, data) {
            return Ok(());
        }
        self.store(addr, data, false)
    }

    fn fetch(&mut self, addr: u64) -> Result<u32, MemFault> {
        match self.in_page::<4>(addr, Access::Execute) {
            Some(Some(b)) => Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            Some(None) => Ok(0),
            None => {
                self.check(addr, 4, Access::Execute)?;
                // A misaligned fetch that straddles two pages, a lazy page not read yet, or a
                // shared page.
                let mut b = [0; 4];
                self.walk(addr, 4, Access::Execute, false, |target, off, start, n| {
                    let out = &mut b[start..start + n];
                    match target {
                        Target::Own { slot, .. } => {
                            if let Some(p) = slot.data.get() {
                                out.copy_from_slice(&p[off..off + n]);
                            }
                        }
                        Target::Shared { obj, index, .. } => {
                            let _ = obj.with_page(index, |p| out.copy_from_slice(&p[off..off + n]));
                        }
                    }
                })?;
                Ok(u32::from_le_bytes(b))
            }
        }
    }

    fn check(&self, addr: u64, len: u64, access: Access) -> Result<(), MemFault> {
        let mut at = addr;
        let end = addr.saturating_add(len);
        while at < end {
            let fault = |kind| MemFault {
                addr: at,
                access,
                kind,
            };
            if at >= ADDRESS_LIMIT {
                return Err(fault(MemFaultKind::Unmapped));
            }
            let perms = self
                .slot(at >> PAGE_SHIFT)
                .and_then(|s| s.perms)
                .ok_or(fault(MemFaultKind::Unmapped))?;
            if !perms.allows(access) {
                return Err(fault(MemFaultKind::Protection));
            }
            // A range that overflows saturates at u64::MAX and so runs into ADDRESS_LIMIT.
            at = (at | PAGE_MASK).saturating_add(1);
        }
        Ok(())
    }

    fn poke(&mut self, addr: u64, data: &[u8]) -> Result<(), MemFault> {
        self.store(addr, data, true)
    }

    fn map(&mut self, addr: u64, len: u64, perms: Perms) -> Result<(), MapError> {
        self.replace(addr, len, perms, Backing::Anon)
    }

    fn map_source(
        &mut self,
        addr: u64,
        len: u64,
        perms: Perms,
        source: Rc<dyn PageSource>,
        first: u64,
    ) -> Result<(), MapError> {
        self.replace(addr, len, perms, Backing::Source)?;
        let start = addr >> PAGE_SHIFT;
        self.sources.insert(
            start,
            Ranged {
                end: start + (len >> PAGE_SHIFT),
                of: source,
                first,
            },
        );
        Ok(())
    }

    fn map_shared(
        &mut self,
        addr: u64,
        len: u64,
        perms: Perms,
        object: Rc<dyn SharedMemory>,
        first: u64,
    ) -> Result<(), MapError> {
        let pages = page_range(addr, len)?;
        let object = object
            .into_any()
            .downcast::<SharedPages>()
            .map_err(|_| MapError::Unsupported)?;
        first
            .checked_add(pages.end - pages.start)
            .ok_or(MapError::OutOfRange)?;
        self.replace(addr, len, perms, Backing::Shared)?;
        if perms.exec() {
            object.note_exec();
        }
        self.shared.insert(
            pages.start,
            Ranged {
                end: pages.end,
                of: object,
                first,
            },
        );
        Ok(())
    }

    fn unmap(&mut self, addr: u64, len: u64) -> Result<(), MapError> {
        self.clear_watch();
        let pages = page_range(addr, len)?;
        if self.clear(pages.clone()) {
            self.epoch += 1;
            self.log_code(pages);
        }
        Ok(())
    }

    fn protect(&mut self, addr: u64, len: u64, perms: Perms) -> Result<(), MapError> {
        self.clear_watch();
        let pages = page_range(addr, len)?;
        if !self.all_mapped(pages.clone()) {
            return Err(MapError::NotMapped);
        }
        if perms.exec() {
            // Code gets slots (code versions are per slot).
            self.materialize_range(pages.clone());
        } else if !self.lazy.is_empty() {
            // Lazy pages stay a range, with the new permissions.
            let hits: Vec<(u64, u64)> = self
                .lazy
                .range(..pages.end)
                .rev()
                .take_while(|(_, r)| r.end > pages.start)
                .map(|(&s, r)| (s.max(pages.start), r.end.min(pages.end)))
                .collect();
            cut_ranges(&mut self.lazy, pages.start, pages.end);
            for (s, e) in hits {
                self.lazy.insert(
                    s,
                    Ranged {
                        end: e,
                        of: perms,
                        first: 0,
                    },
                );
            }
        }
        let mut code = false;
        let mut shared = false;
        self.for_each_own_slot(pages.clone(), |slot| {
            // Slots without permissions here belong to lazy pages (all pages are mapped).
            if slot.perms.is_none() {
                return;
            }
            // Gaining or losing execute permission changes what may run there.
            if slot.perms.is_some_and(Perms::exec) != perms.exec() {
                slot.code_gen = slot.code_gen.wrapping_add(1);
                code = true;
            }
            slot.perms = Some(perms);
            shared |= slot.backing == Backing::Shared;
        });
        if shared && perms.exec() {
            // Objects that become executable here keep every store exact from now on.
            for (_, r) in self.shared.range(..pages.end) {
                if r.end > pages.start {
                    r.of.note_exec();
                }
            }
        }
        if code {
            self.epoch += 1;
            self.log_code(pages);
        }
        Ok(())
    }

    fn perms(&self, addr: u64) -> Option<Perms> {
        if addr >= ADDRESS_LIMIT {
            return None;
        }
        let vpn = addr >> PAGE_SHIFT;
        if let Some(&(_, saved)) = self.watch.iter().find(|(v, _)| *v == vpn) {
            return Some(saved);
        }
        self.slot(vpn).and_then(|s| s.perms)
    }

    fn set_watch(&mut self, pages: &[(u64, Perms)]) -> bool {
        self.clear_watch();
        for &(addr, deny) in pages {
            if addr >= ADDRESS_LIMIT {
                continue;
            }
            let vpn = addr >> PAGE_SHIFT;
            if self.watch.iter().any(|(v, _)| *v == vpn) {
                continue;
            }
            // A lazy page gets its own slot now (it would on its first touch anyway).
            if !self.own_mapped(vpn) {
                self.materialize(vpn);
            }
            if let Some(slot) = self.slot_mut(vpn)
                && let Some(guest) = slot.perms
            {
                slot.perms = Some(guest.without(deny));
                self.watch.push((vpn, guest));
            }
        }
        true
    }

    fn address_limit(&self) -> u64 {
        ADDRESS_LIMIT
    }

    fn code_epoch(&self) -> u64 {
        // Stores to executable shared pages anywhere move every address space's epoch.
        self.epoch.wrapping_add(shared_code_epoch())
    }

    fn code_version(&self, addr: u64) -> u64 {
        let vpn = addr >> PAGE_SHIFT;
        // A lazy page's own slot (left from an earlier mapping) carries its code generation;
        // [`Self::materialize`] keeps it.
        match self.own_slot(vpn).or_else(|| self.lazy_slot(vpn)) {
            Some(slot) if addr < ADDRESS_LIMIT => {
                let mut generation = slot.code_gen;
                if slot.backing == Backing::Shared
                    && let Some((obj, _)) = self.shared_page(vpn)
                {
                    // Both counters only grow, so their sum changes whenever either does.
                    generation = generation.wrapping_add(obj.code_gen());
                }
                (self.instance << 32) | u64::from(generation)
            }
            _ => u64::MAX,
        }
    }

    fn duplicate(&self) -> Box<dyn GuestMemory> {
        Box::new(self.clone())
    }

    fn for_each_mapped(
        &self,
        f: &mut dyn FnMut(u64, u64, Perms, PageView<'_>),
    ) -> Result<(), MapError> {
        let mut out = ZeroRuns { f, run: None };
        let mut lazy = self.lazy.iter().peekable();
        for (a, mid) in self.root.iter().enumerate() {
            let Some(mid) = mid else { continue };
            for (b, leaf) in mid.iter().enumerate() {
                let Some(leaf) = leaf else { continue };
                for (c, slot) in leaf.iter().enumerate() {
                    let Some(mut perms) = slot.perms else {
                        continue;
                    };
                    let vpn =
                        ((a as u64) << (2 * LEVEL_BITS)) | ((b as u64) << LEVEL_BITS) | c as u64;
                    // A debugger's watch view lowers slot permissions: list the guest's.
                    if let Some(&(_, saved)) = self.watch.iter().find(|(v, _)| *v == vpn) {
                        perms = saved;
                    }
                    // Lazy ranges before this page (they never overlap pages with slots).
                    while let Some((&s, r)) = lazy.next_if(|&(&s, _)| s < vpn) {
                        out.page(s, r.end - s, r.of, PageView::Zero);
                    }
                    let view = match (slot.data.get(), slot.backing) {
                        (_, Backing::Shared) => match self.shared_page(vpn) {
                            Some((_, index)) => PageView::Shared(index),
                            None => return Err(MapError::Unsupported),
                        },
                        (Some(page), _) => PageView::Data(page),
                        (None, Backing::Anon) => PageView::Zero,
                        (None, Backing::Source) => match range_at(&self.sources, vpn) {
                            Some((source, index)) => PageView::Source(source, index),
                            None => return Err(MapError::Unsupported),
                        },
                    };
                    out.page(vpn, 1, perms, view);
                }
            }
        }
        for (&s, r) in lazy {
            out.page(s, r.end - s, r.of, PageView::Zero);
        }
        out.flush();
        Ok(())
    }

    #[inline]
    fn read_u8(&self, addr: u64) -> Result<u8, MemFault> {
        Ok(self.load::<1>(addr)?[0])
    }
    #[inline]
    fn read_u16(&self, addr: u64) -> Result<u16, MemFault> {
        Ok(u16::from_le_bytes(self.load(addr)?))
    }
    #[inline]
    fn read_u32(&self, addr: u64) -> Result<u32, MemFault> {
        Ok(u32::from_le_bytes(self.load(addr)?))
    }
    #[inline]
    fn read_u64(&self, addr: u64) -> Result<u64, MemFault> {
        Ok(u64::from_le_bytes(self.load(addr)?))
    }
    #[inline]
    fn write_u8(&mut self, addr: u64, v: u8) -> Result<(), MemFault> {
        if self.store_fast(addr, &[v]) {
            return Ok(());
        }
        self.write(addr, &[v])
    }
    #[inline]
    fn write_u16(&mut self, addr: u64, v: u16) -> Result<(), MemFault> {
        if self.store_fast(addr, &v.to_le_bytes()) {
            return Ok(());
        }
        self.write(addr, &v.to_le_bytes())
    }
    #[inline]
    fn write_u32(&mut self, addr: u64, v: u32) -> Result<(), MemFault> {
        if self.store_fast(addr, &v.to_le_bytes()) {
            return Ok(());
        }
        self.write(addr, &v.to_le_bytes())
    }
    #[inline]
    fn write_u64(&mut self, addr: u64, v: u64) -> Result<(), MemFault> {
        if self.store_fast(addr, &v.to_le_bytes()) {
            return Ok(());
        }
        self.write(addr, &v.to_le_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x40_0000;

    #[test]
    fn fresh_mapping_reads_zero_and_allocates_on_write() {
        let mut m = PagedMemory::new();
        m.map(BASE, 2 * PAGE_SIZE, Perms::RW).unwrap();
        assert_eq!(m.read_u64(BASE + 8).unwrap(), 0);
        assert_eq!(m.resident_pages(), 0);
        m.write_u64(BASE + 8, 0x1122_3344_5566_7788).unwrap();
        assert_eq!(m.read_u64(BASE + 8).unwrap(), 0x1122_3344_5566_7788);
        assert_eq!(m.read_u8(BASE + 8).unwrap(), 0x88);
        assert_eq!(m.resident_pages(), 1);
    }

    #[test]
    fn access_spanning_two_pages() {
        let mut m = PagedMemory::new();
        m.map(BASE, 2 * PAGE_SIZE, Perms::RW).unwrap();
        let at = BASE + PAGE_SIZE - 3;
        m.write_u64(at, u64::MAX - 1).unwrap();
        assert_eq!(m.read_u64(at).unwrap(), u64::MAX - 1);
        let mut buf = [0u8; 10];
        m.read(at - 1, &mut buf).unwrap();
        assert_eq!(buf, [0, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0]);
    }

    #[test]
    fn faults_name_the_first_bad_byte() {
        let mut m = PagedMemory::new();
        m.map(BASE, PAGE_SIZE, Perms::R).unwrap();
        let end = BASE + PAGE_SIZE;
        let err = m.read_u64(end - 4).unwrap_err();
        assert_eq!(
            err,
            MemFault {
                addr: end,
                access: Access::Read,
                kind: MemFaultKind::Unmapped
            }
        );
        let err = m.write_u32(BASE, 1).unwrap_err();
        assert_eq!(err.kind, MemFaultKind::Protection);
        assert_eq!(m.fetch(BASE).unwrap_err().access, Access::Execute);
        assert_eq!(
            m.read_u8(u64::MAX).unwrap_err().kind,
            MemFaultKind::Unmapped
        );
    }

    #[test]
    fn poke_ignores_permissions_but_not_holes() {
        let mut m = PagedMemory::new();
        m.map(BASE, PAGE_SIZE, Perms::RX).unwrap();
        m.poke(BASE, &0xd503_201fu32.to_le_bytes()).unwrap();
        assert_eq!(m.fetch(BASE).unwrap(), 0xd503_201f);
        assert!(m.poke(BASE + PAGE_SIZE, &[1]).is_err());
    }

    #[test]
    fn protect_and_unmap() {
        let mut m = PagedMemory::new();
        m.map(BASE, 2 * PAGE_SIZE, Perms::RW).unwrap();
        m.write_u8(BASE, 7).unwrap();
        m.protect(BASE, PAGE_SIZE, Perms::R).unwrap();
        assert_eq!(m.read_u8(BASE).unwrap(), 7);
        assert!(m.write_u8(BASE, 1).is_err());
        assert_eq!(
            m.protect(BASE, 3 * PAGE_SIZE, Perms::R),
            Err(MapError::NotMapped)
        );
        m.unmap(BASE, PAGE_SIZE).unwrap();
        assert_eq!(m.perms(BASE), None);
        assert_eq!(m.perms(BASE + PAGE_SIZE), Some(Perms::RW));
        assert_eq!(m.resident_pages(), 0);
        assert_eq!(
            m.map(BASE + 1, PAGE_SIZE, Perms::R),
            Err(MapError::Unaligned)
        );
        assert_eq!(
            m.map(ADDRESS_LIMIT - PAGE_SIZE, 2 * PAGE_SIZE, Perms::R),
            Err(MapError::OutOfRange)
        );
    }

    #[test]
    fn code_epoch_moves_on_code_writes_and_remaps() {
        let mut m = PagedMemory::new();
        m.map(BASE, PAGE_SIZE, Perms::RWX).unwrap();
        m.map(BASE + PAGE_SIZE, PAGE_SIZE, Perms::RW).unwrap();
        // `code_epoch()` also counts stores to shared code pages in any address space of the
        // process (other tests, in parallel), so "did not move" is read off this space's own
        // counter.
        let (e0, own) = (m.code_epoch(), m.epoch);
        m.write_u8(BASE + PAGE_SIZE, 1).unwrap();
        assert_eq!(m.epoch, own);
        m.write_u8(BASE, 1).unwrap();
        assert!(m.code_epoch() > e0);
    }

    #[test]
    fn duplicate_is_copy_on_write() {
        let mut a = PagedMemory::new();
        a.map(BASE, 2 * PAGE_SIZE, Perms::RW).unwrap();
        a.write_u64(BASE, 1).unwrap();
        let mut b = a.duplicate();
        b.write_u64(BASE, 2).unwrap();
        b.write_u64(BASE + PAGE_SIZE, 3).unwrap();
        assert_eq!(a.read_u64(BASE).unwrap(), 1);
        assert_eq!(a.read_u64(BASE + PAGE_SIZE).unwrap(), 0);
        assert_eq!(b.read_u64(BASE).unwrap(), 2);
        a.unmap(BASE, PAGE_SIZE).unwrap();
        assert_eq!(b.read_u64(BASE).unwrap(), 2);
    }

    #[test]
    fn code_versions_are_per_page_and_per_address_space() {
        let mut m = PagedMemory::new();
        m.map(BASE, 2 * PAGE_SIZE, Perms::RX).unwrap();
        let (v0, v1) = (m.code_version(BASE), m.code_version(BASE + PAGE_SIZE));
        // A write to one code page changes only its version (and the epoch).
        let e = m.code_epoch();
        m.poke(BASE, &[1]).unwrap();
        assert_ne!(m.code_version(BASE), v0);
        assert_eq!(m.code_version(BASE + PAGE_SIZE), v1);
        assert_ne!(m.code_epoch(), e);
        // Data pages never move the epoch.
        m.map(BASE + 4 * PAGE_SIZE, PAGE_SIZE, Perms::RW).unwrap();
        let own = m.epoch;
        m.write_u8(BASE + 4 * PAGE_SIZE, 1).unwrap();
        m.protect(BASE + 4 * PAGE_SIZE, PAGE_SIZE, Perms::R)
            .unwrap();
        assert_eq!(m.epoch, own);
        // Losing execute permission is a code change.
        let v = m.code_version(BASE + PAGE_SIZE);
        m.protect(BASE + PAGE_SIZE, PAGE_SIZE, Perms::R).unwrap();
        assert_ne!(m.code_version(BASE + PAGE_SIZE), v);
        // A copy has versions of its own.
        let c = m.clone();
        assert_ne!(c.code_version(BASE), m.code_version(BASE));
        assert_ne!(c.code_epoch(), m.code_epoch());
    }

    #[test]
    fn code_changes_list_every_page_whose_version_moved() {
        let mut m = PagedMemory::new();
        m.map(BASE, 4 * PAGE_SIZE, Perms::RX).unwrap();
        let _ = m.take_code_changes();
        let vpn = BASE >> PAGE_SHIFT;
        let versions = |m: &PagedMemory| -> Vec<u64> {
            (0..4)
                .map(|i| m.code_version(BASE + i * PAGE_SIZE))
                .collect()
        };
        let check = |m: &mut PagedMemory, before: Vec<u64>| {
            let after = versions(m);
            let listed = m.take_code_changes().expect("listed");
            for i in 0..4u64 {
                if before[i as usize] != after[i as usize] {
                    assert!(
                        listed.contains(&(vpn + i)),
                        "page {i} moved but is not listed"
                    );
                }
            }
        };
        // A store straddling two code pages.
        let v = versions(&m);
        m.poke(BASE + PAGE_SIZE - 2, &[1, 2, 3, 4]).unwrap();
        check(&mut m, v);
        // Protections, remaps and unmaps.
        let v = versions(&m);
        m.protect(BASE + 2 * PAGE_SIZE, PAGE_SIZE, Perms::RW)
            .unwrap();
        check(&mut m, v);
        let v = versions(&m);
        m.map(BASE, PAGE_SIZE, Perms::RX).unwrap();
        check(&mut m, v);
        let v = versions(&m);
        m.unmap(BASE + 3 * PAGE_SIZE, PAGE_SIZE).unwrap();
        check(&mut m, v);
        // Nothing moved: nothing listed.
        assert_eq!(m.take_code_changes(), Some(Vec::new()));
        // Too many pages: not listed.
        m.map(BASE + 16 * PAGE_SIZE, 100 * PAGE_SIZE, Perms::RX)
            .unwrap();
        assert_eq!(m.take_code_changes(), None);
        assert_eq!(m.take_code_changes(), Some(Vec::new()));
    }

    #[test]
    fn host_pages_and_page_bytes() {
        let mut m = PagedMemory::new();
        m.map(BASE, PAGE_SIZE, Perms::RW).unwrap();
        m.map(BASE + PAGE_SIZE, PAGE_SIZE, Perms::RX).unwrap();
        // A page never written reads through the shared zero page.
        assert!(m.host_read_page(BASE).is_some());
        assert_eq!(m.page_bytes(BASE).unwrap(), &[0u8; 4096][..]);
        // Writable, not code: private and resident.
        let w = m.host_write_page(BASE).unwrap();
        assert_eq!(m.resident_pages(), 1);
        assert_eq!(m.host_read_page(BASE), Some(w));
        // Code is never writable through a host page; unmapped pages have none.
        assert_eq!(m.host_write_page(BASE + PAGE_SIZE), None);
        assert_eq!(m.host_read_page(BASE + 2 * PAGE_SIZE), None);
        assert!(m.page_bytes(BASE + 2 * PAGE_SIZE).is_none());
        // A shared page becomes private again on a write.
        let c = m.clone();
        assert_eq!(c.host_read_page(BASE), Some(w));
        assert_ne!(m.host_write_page(BASE), Some(w));
    }

    /// Page-table leaves allocated (4096 slots each).
    fn leaves(m: &PagedMemory) -> usize {
        m.root
            .iter()
            .flatten()
            .map(|mid| mid.iter().flatten().count())
            .sum()
    }

    #[test]
    fn huge_reservations_cost_a_range_until_written() {
        const TIB: u64 = 1 << 40;
        let base = 1 << 42;
        let mut m = PagedMemory::new();
        m.map(base, TIB, Perms::NONE).unwrap();
        assert_eq!(leaves(&m), 0);
        assert_eq!(m.perms(base + TIB / 2), Some(Perms::NONE));
        assert_eq!(m.perms(base + TIB), None);
        assert_eq!(m.read_u8(base).unwrap_err().kind, MemFaultKind::Protection);
        assert_eq!(
            m.check(base + TIB - 8, 16, Access::Read).unwrap_err().kind,
            MemFaultKind::Protection
        );
        // Commit a sub-range (V8 / PartitionAlloc style): still no slots until written.
        let heap = base + 0x1234_5000;
        m.protect(heap, 1 << 30, Perms::RW).unwrap();
        assert_eq!(leaves(&m), 0);
        assert_eq!(m.read_u64(heap + 8).unwrap(), 0);
        assert_eq!(m.perms(heap - PAGE_SIZE), Some(Perms::NONE));
        m.write_u64(heap + 8, 42).unwrap();
        assert_eq!(m.read_u64(heap + 8).unwrap(), 42);
        assert_eq!(m.resident_pages(), 1);
        assert_eq!(leaves(&m), 1);
        // Its block-mates got slots, and still read zero.
        assert_eq!(m.read_u64(heap + PAGE_SIZE).unwrap(), 0);
        m.write_u8(heap + 63 * PAGE_SIZE, 1).unwrap();
        // A copy is copy-on-write and keeps the reservation.
        let mut c = m.clone();
        c.write_u64(heap + 8, 7).unwrap();
        assert_eq!(m.read_u64(heap + 8).unwrap(), 42);
        assert_eq!(c.perms(base + 5 * PAGE_SIZE), Some(Perms::NONE));
        // Protect across materialised and lazy pages; then back to none.
        m.protect(heap, 128 * PAGE_SIZE, Perms::R).unwrap();
        assert!(m.write_u8(heap, 1).is_err());
        assert!(m.write_u8(heap + 100 * PAGE_SIZE, 1).is_err());
        assert_eq!(m.read_u64(heap + 8).unwrap(), 42);
        m.protect(heap, 1 << 30, Perms::NONE).unwrap();
        assert!(m.read_u8(heap + 8).is_err());
        // Unmapping the middle leaves holes; protect over a hole fails.
        m.unmap(heap, 1 << 30).unwrap();
        assert_eq!(m.perms(heap), None);
        assert_eq!(m.resident_pages(), 0);
        assert_eq!(
            m.protect(heap - PAGE_SIZE, 2 * PAGE_SIZE, Perms::RW),
            Err(MapError::NotMapped)
        );
        m.protect(base, PAGE_SIZE, Perms::RW).unwrap();
        // Remapping over everything drops the old pages.
        m.map(base, TIB, Perms::RW).unwrap();
        assert_eq!(m.read_u64(heap + 8).unwrap(), 0);
        m.unmap(base, TIB).unwrap();
        assert_eq!(m.perms(base), None);
    }

    #[test]
    fn lazy_pages_turned_into_code_keep_code_versions_moving() {
        let mut m = PagedMemory::new();
        let len = 256 * PAGE_SIZE;
        m.map(BASE, len, Perms::RWX).unwrap();
        m.poke(BASE, &0xd503_201fu32.to_le_bytes()).unwrap();
        let v = m.code_version(BASE);
        // A lazy remap over code: the epoch moves, and the page's version never comes back.
        let e = m.code_epoch();
        m.map(BASE, len, Perms::NONE).unwrap();
        assert_ne!(m.code_epoch(), e);
        assert_ne!(m.code_version(BASE), v);
        m.protect(BASE, PAGE_SIZE, Perms::RX).unwrap();
        assert_eq!(m.fetch(BASE).unwrap(), 0);
        assert_ne!(m.code_version(BASE), v);
        // Executable protect materialises; the rest of the range stays lazy.
        assert_eq!(m.perms(BASE + PAGE_SIZE), Some(Perms::NONE));
        m.protect(BASE + 200 * PAGE_SIZE, PAGE_SIZE, Perms::RWX)
            .unwrap();
        m.write_u32(BASE + 200 * PAGE_SIZE, 0xd503_201f).unwrap();
        assert_eq!(m.fetch(BASE + 200 * PAGE_SIZE).unwrap(), 0xd503_201f);
        // Host pages work on lazy pages.
        m.protect(BASE + 100 * PAGE_SIZE, PAGE_SIZE, Perms::RW)
            .unwrap();
        assert!(m.host_read_page(BASE + 100 * PAGE_SIZE).is_some());
        assert!(m.host_write_page(BASE + 100 * PAGE_SIZE).is_some());
        assert_eq!(m.host_read_page(BASE + 101 * PAGE_SIZE), None);
    }

    #[test]
    fn check_reports_holes() {
        let mut m = PagedMemory::new();
        m.map(BASE, PAGE_SIZE, Perms::RW).unwrap();
        assert!(m.check(BASE, PAGE_SIZE, Access::Write).is_ok());
        assert_eq!(
            m.check(BASE, PAGE_SIZE + 1, Access::Read).unwrap_err().addr,
            BASE + PAGE_SIZE
        );
        assert!(m.check(u64::MAX - 2, 8, Access::Read).is_err());
    }

    /// Page `i` is filled with byte `i + 1`; counts the pages it produced; fails at `fail`.
    struct Counting {
        reads: Cell<u64>,
        fail: u64,
    }

    impl PageSource for Counting {
        fn page(&self, index: u64) -> Result<PageRef, PageSourceError> {
            if index == self.fail {
                return Err(PageSourceError);
            }
            self.reads.set(self.reads.get() + 1);
            Ok(Rc::new([(index + 1) as u8; PAGE_SIZE as usize]))
        }
    }

    fn source() -> Rc<Counting> {
        Rc::new(Counting {
            reads: Cell::new(0),
            fail: u64::MAX,
        })
    }

    #[test]
    fn source_pages_fill_on_first_touch() {
        let src = source();
        let mut m = PagedMemory::new();
        m.map_source(BASE, 4 * PAGE_SIZE, Perms::RX, src.clone(), 10)
            .unwrap();
        assert_eq!(m.resident_pages(), 0);
        assert_eq!(src.reads.get(), 0);
        // Reads, fetches, host pages and page bytes all fill (once).
        assert_eq!(m.read_u8(BASE).unwrap(), 11);
        assert_eq!(m.fetch(BASE + PAGE_SIZE).unwrap(), 0x0c0c_0c0c);
        assert_eq!(m.page_bytes(BASE + 2 * PAGE_SIZE).unwrap()[5], 13);
        assert!(m.host_read_page(BASE + 3 * PAGE_SIZE).is_some());
        assert_eq!(m.read_u8(BASE + 7).unwrap(), 11);
        assert_eq!(src.reads.get(), 4);
        assert_eq!(m.resident_pages(), 4);
        // A read across two pages.
        let mut b = [0u8; 4];
        m.read(BASE + PAGE_SIZE - 2, &mut b).unwrap();
        assert_eq!(b, [11, 11, 12, 12]);
    }

    #[test]
    fn source_pages_copy_on_write_and_share_with_duplicates() {
        let src = source();
        let mut m = PagedMemory::new();
        m.map_source(BASE, 2 * PAGE_SIZE, Perms::RW, src.clone(), 0)
            .unwrap();
        let mut c = m.clone();
        // A write fills first, then changes only this address space's copy.
        m.write_u8(BASE + 1, 0xaa).unwrap();
        assert_eq!(m.read_u8(BASE).unwrap(), 1);
        assert_eq!(m.read_u8(BASE + 1).unwrap(), 0xaa);
        assert_eq!(c.read_u8(BASE + 1).unwrap(), 1);
        // The duplicate's untouched page fills from the same source.
        assert_eq!(c.read_u8(BASE + PAGE_SIZE).unwrap(), 2);
        let w = c.host_write_page(BASE + PAGE_SIZE).unwrap();
        assert_ne!(w, 0);
        assert_eq!(c.read_u8(BASE + PAGE_SIZE).unwrap(), 2);
        // poke ignores permissions but still fills.
        m.protect(BASE, 2 * PAGE_SIZE, Perms::R).unwrap();
        m.poke(BASE + PAGE_SIZE + 2, &[9]).unwrap();
        assert_eq!(
            m.read(BASE + PAGE_SIZE + 1, &mut [0; 3]).map(|_| ()),
            Ok(())
        );
        let mut b = [0u8; 3];
        m.read(BASE + PAGE_SIZE + 1, &mut b).unwrap();
        assert_eq!(b, [2, 9, 2]);
    }

    #[test]
    fn source_ranges_split_on_remap_and_unmap() {
        let src = source();
        let mut m = PagedMemory::new();
        m.map_source(BASE, 6 * PAGE_SIZE, Perms::R, src.clone(), 0)
            .unwrap();
        // Replace the middle with zeros and unmap one page: the rest keeps its source pages.
        m.map(BASE + 2 * PAGE_SIZE, PAGE_SIZE, Perms::RW).unwrap();
        m.unmap(BASE + 4 * PAGE_SIZE, PAGE_SIZE).unwrap();
        assert_eq!(m.read_u8(BASE + PAGE_SIZE).unwrap(), 2);
        assert_eq!(m.read_u8(BASE + 2 * PAGE_SIZE).unwrap(), 0);
        assert_eq!(m.read_u8(BASE + 3 * PAGE_SIZE).unwrap(), 4);
        assert!(m.read_u8(BASE + 4 * PAGE_SIZE).is_err());
        assert_eq!(m.read_u8(BASE + 5 * PAGE_SIZE).unwrap(), 6);
        assert_eq!(m.sources.len(), 3);
        // A remap over a source range with another source.
        m.map_source(BASE, 2 * PAGE_SIZE, Perms::R, src.clone(), 100)
            .unwrap();
        assert_eq!(m.read_u8(BASE + PAGE_SIZE).unwrap(), 102);
        m.unmap(BASE, 8 * PAGE_SIZE).unwrap();
        assert!(m.sources.is_empty());
        assert_eq!(m.resident_pages(), 0);
    }

    #[test]
    fn a_failing_source_faults_as_unmapped() {
        let src = Rc::new(Counting {
            reads: Cell::new(0),
            fail: 1,
        });
        let mut m = PagedMemory::new();
        m.map_source(BASE, 2 * PAGE_SIZE, Perms::RWX, src, 0)
            .unwrap();
        assert_eq!(m.read_u8(BASE).unwrap(), 1);
        let at = BASE + PAGE_SIZE;
        assert_eq!(m.read_u8(at).unwrap_err().kind, MemFaultKind::Unmapped);
        assert_eq!(m.write_u8(at, 1).unwrap_err().addr, at);
        assert_eq!(m.fetch(at).unwrap_err().kind, MemFaultKind::Unmapped);
        assert!(m.page_bytes(at).is_none());
        assert_eq!(m.perms(at), Some(Perms::RWX));
    }

    #[test]
    fn the_default_map_source_copies_eagerly() {
        /// A memory that only implements the required methods (the default `map_source`).
        struct Plain(PagedMemory);
        impl GuestMemory for Plain {
            fn read(&self, a: u64, b: &mut [u8]) -> Result<(), MemFault> {
                self.0.read(a, b)
            }
            fn write(&mut self, a: u64, d: &[u8]) -> Result<(), MemFault> {
                self.0.write(a, d)
            }
            fn fetch(&mut self, a: u64) -> Result<u32, MemFault> {
                self.0.fetch(a)
            }
            fn check(&self, a: u64, l: u64, x: Access) -> Result<(), MemFault> {
                self.0.check(a, l, x)
            }
            fn poke(&mut self, a: u64, d: &[u8]) -> Result<(), MemFault> {
                self.0.poke(a, d)
            }
            fn map(&mut self, a: u64, l: u64, p: Perms) -> Result<(), MapError> {
                self.0.map(a, l, p)
            }
            fn unmap(&mut self, a: u64, l: u64) -> Result<(), MapError> {
                self.0.unmap(a, l)
            }
            fn protect(&mut self, a: u64, l: u64, p: Perms) -> Result<(), MapError> {
                self.0.protect(a, l, p)
            }
            fn perms(&self, a: u64) -> Option<Perms> {
                self.0.perms(a)
            }
            fn address_limit(&self) -> u64 {
                self.0.address_limit()
            }
            fn code_epoch(&self) -> u64 {
                self.0.code_epoch()
            }
            fn duplicate(&self) -> Box<dyn GuestMemory> {
                self.0.duplicate()
            }
        }
        let src = source();
        let mut m = Plain(PagedMemory::new());
        m.map_source(BASE, 2 * PAGE_SIZE, Perms::R, src.clone(), 4)
            .unwrap();
        assert_eq!(src.reads.get(), 2);
        assert_eq!(m.read_u8(BASE + PAGE_SIZE).unwrap(), 6);
        let failing = Rc::new(Counting {
            reads: Cell::new(0),
            fail: 1,
        });
        assert_eq!(
            m.map_source(BASE, 2 * PAGE_SIZE, Perms::R, failing, 0),
            Err(MapError::OutOfRange)
        );
        assert_eq!(m.perms(BASE), None);
    }

    fn object() -> Rc<SharedPages> {
        Rc::new(SharedPages::new(None))
    }

    #[test]
    fn shared_pages_are_coherent_across_spaces_and_duplicates() {
        let obj = object();
        let mut a = PagedMemory::new();
        let mut b = PagedMemory::new();
        a.map_shared(BASE, 2 * PAGE_SIZE, Perms::RW, obj.clone(), 0)
            .unwrap();
        // b maps the object's second page twice, once read-only.
        b.map_shared(0x20000, PAGE_SIZE, Perms::RW, obj.clone(), 1)
            .unwrap();
        b.map_shared(0x30000, PAGE_SIZE, Perms::R, obj.clone(), 1)
            .unwrap();
        a.write_u64(BASE + PAGE_SIZE + 8, 0x1122).unwrap();
        assert_eq!(b.read_u64(0x20008).unwrap(), 0x1122);
        assert_eq!(b.read_u64(0x30008).unwrap(), 0x1122);
        b.write_u32(0x20010, 7).unwrap();
        assert_eq!(a.read_u32(BASE + PAGE_SIZE + 16).unwrap(), 7);
        assert!(
            b.write_u8(0x30000, 1).is_err(),
            "permissions stay per mapping"
        );
        let mut buf = [0u8; 4];
        obj.read(PAGE_SIZE + 16, &mut buf).unwrap();
        assert_eq!(u32::from_le_bytes(buf), 7);
        obj.write(8, &[9]).unwrap();
        assert_eq!(a.read_u8(BASE + 8).unwrap(), 9);
        // A duplicate (fork) keeps sharing; private pages next to it stay copy-on-write.
        a.map(BASE + 2 * PAGE_SIZE, PAGE_SIZE, Perms::RW).unwrap();
        a.write_u8(BASE + 2 * PAGE_SIZE, 1).unwrap();
        let mut c = a.duplicate();
        c.write_u8(BASE, 42).unwrap();
        c.write_u8(BASE + 2 * PAGE_SIZE, 2).unwrap();
        assert_eq!(a.read_u8(BASE).unwrap(), 42);
        assert_eq!(a.read_u8(BASE + 2 * PAGE_SIZE).unwrap(), 1);
        // Engines see the object's page itself.
        assert_eq!(
            a.host_read_page(BASE + PAGE_SIZE),
            b.host_read_page(0x20000)
        );
        assert_eq!(
            a.host_write_page(BASE + PAGE_SIZE),
            b.host_read_page(0x30000)
        );
        assert_eq!(
            a.page_bytes(BASE),
            None,
            "shared pages are no code cache key"
        );
        assert_eq!(
            a.resident_pages(),
            1,
            "object pages are not the space's own"
        );
    }

    #[test]
    fn shared_ranges_split_and_replace() {
        let obj = object();
        let mut a = PagedMemory::new();
        a.map_shared(BASE, 4 * PAGE_SIZE, Perms::RW, obj.clone(), 10)
            .unwrap();
        a.unmap(BASE + PAGE_SIZE, PAGE_SIZE).unwrap();
        a.map(BASE + 2 * PAGE_SIZE, PAGE_SIZE, Perms::RW).unwrap();
        assert_eq!(a.shared.len(), 2);
        a.write_u8(BASE + 3 * PAGE_SIZE, 5).unwrap();
        a.write_u8(BASE + 2 * PAGE_SIZE, 6).unwrap();
        let mut b = [0u8; 1];
        obj.read(13 * PAGE_SIZE, &mut b).unwrap();
        assert_eq!(b[0], 5);
        obj.read(12 * PAGE_SIZE, &mut b).unwrap();
        assert_eq!(b[0], 0, "the private page replaced the shared one");
        assert_eq!(
            a.map_shared(BASE, PAGE_SIZE, Perms::RW, Rc::new(Foreign), 0),
            Err(MapError::Unsupported)
        );
    }

    struct Foreign;
    impl SharedMemory for Foreign {
        fn read(&self, _: u64, _: &mut [u8]) -> Result<(), PageSourceError> {
            Ok(())
        }
        fn write(&self, _: u64, _: &[u8]) -> Result<(), PageSourceError> {
            Ok(())
        }
        fn zero_from(&self, _: u64) -> Result<(), PageSourceError> {
            Ok(())
        }
        fn into_any(self: Rc<Self>) -> Rc<dyn core::any::Any> {
            self
        }
    }

    #[test]
    fn stores_to_shared_code_move_code_versions_everywhere() {
        let obj = object();
        let mut writer = PagedMemory::new();
        let mut runner = PagedMemory::new();
        writer
            .map_shared(BASE, PAGE_SIZE, Perms::RW, obj.clone(), 0)
            .unwrap();
        runner
            .map_shared(0x40000, PAGE_SIZE, Perms::RX, obj.clone(), 0)
            .unwrap();
        writer.write_u32(BASE, 0xd503_201f).unwrap();
        let (epoch, version) = (runner.code_epoch(), runner.code_version(0x40000));
        assert_eq!(runner.fetch(0x40000).unwrap(), 0xd503_201f);
        writer.write_u32(BASE, 0xd65f_03c0).unwrap();
        assert_ne!(runner.code_epoch(), epoch);
        assert_ne!(runner.code_version(0x40000), version);
        assert_eq!(runner.fetch(0x40000).unwrap(), 0xd65f_03c0);
        assert_eq!(
            writer.host_write_page(BASE),
            None,
            "no direct stores into an object that holds code"
        );
    }

    /// What `for_each_mapped` reports, as text.
    fn listing(m: &PagedMemory) -> Vec<alloc::string::String> {
        let mut out = Vec::new();
        m.for_each_mapped(&mut |addr, n, p, view| {
            let v = match view {
                PageView::Zero => alloc::string::String::from("zero"),
                PageView::Data(d) => alloc::format!("data {}", d[0]),
                PageView::Source(_, i) => alloc::format!("source {i}"),
                PageView::Shared(i) => alloc::format!("shared {i}"),
            };
            out.push(alloc::format!("{addr:#x} {n} {p:?} {v}"));
        })
        .unwrap();
        out
    }

    #[test]
    fn mapped_pages_list_in_order_without_filling() {
        let src = source();
        let mut m = PagedMemory::new();
        // A lazy reservation, then small anonymous mappings around a written page.
        m.map(0x1000_0000, 256 * PAGE_SIZE, Perms::RW).unwrap();
        m.map(BASE, 3 * PAGE_SIZE, Perms::RW).unwrap();
        m.map(BASE + 3 * PAGE_SIZE, PAGE_SIZE, Perms::R).unwrap();
        m.write_u8(BASE + PAGE_SIZE, 7).unwrap();
        m.map_source(0x50_0000, 2 * PAGE_SIZE, Perms::RX, src.clone(), 10)
            .unwrap();
        assert_eq!(m.read_u8(0x50_0000).unwrap(), 11);
        m.map_shared(0x60_0000, PAGE_SIZE, Perms::RW, object(), 3)
            .unwrap();
        // A written page in the middle of the lazy range.
        m.write_u8(0x1000_0000 + 100 * PAGE_SIZE, 9).unwrap();
        let resident = m.resident_pages();
        let reads = src.reads.get();
        let l = listing(&m);
        assert_eq!(src.reads.get(), reads, "listing reads no source");
        assert_eq!(m.resident_pages(), resident);
        let rw = alloc::format!("{:?}", Perms::RW);
        let r = alloc::format!("{:?}", Perms::R);
        let rx = alloc::format!("{:?}", Perms::RX);
        assert_eq!(
            l,
            [
                alloc::format!("0x400000 1 {rw} zero"),
                alloc::format!("0x401000 1 {rw} data 7"),
                alloc::format!("0x402000 1 {rw} zero"),
                alloc::format!("0x403000 1 {r} zero"),
                alloc::format!("0x500000 1 {rx} data 11"),
                alloc::format!("0x501000 1 {rx} source 11"),
                alloc::format!("0x600000 1 {rw} shared 3"),
                // The write gave its block of 64 pages slots; the rest stays lazy. Zero pages
                // join into runs whichever way they are held.
                alloc::format!("0x10000000 100 {rw} zero"),
                alloc::format!("0x10064000 1 {rw} data 9"),
                alloc::format!("0x10065000 155 {rw} zero"),
            ]
        );
        // A debugger's watch view changes nothing listed (the guest's permissions).
        assert!(m.set_watch(&[(BASE + PAGE_SIZE, Perms::W), (0x1000_0000, Perms::RW)]));
        assert_eq!(listing(&m), l);
        m.set_watch(&[]);
    }
}
