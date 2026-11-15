//! An address space inside the engine: guest pages, the software TLB that translated code
//! uses, the translations of its code, and the execution counters that decide what to
//! translate.

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use alloc::rc::Rc;
use wasmshim_aarch64_translate::abi::TLB_ENTRIES;
use wasmshim_cpu::{
    Access, GuestMemory, MapError, MemFault, PAGE_SIZE, PageSource, PagedMemory, Perms,
    SharedMemory,
};

const PAGE_MASK: u64 = PAGE_SIZE - 1;
const TLB: usize = TLB_ENTRIES as usize;
/// Direct-mapped execution counters.
const COUNTERS: usize = 1 << 14;
/// Direct-mapped translation lookup in front of the map.
const FAST: usize = 1 << 12;

/// One TLB entry (layout: [`wasmshim_aarch64_translate::abi`]).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TlbEntry {
    tag: u64,
    addend: u32,
    _pad: u32,
}

const EMPTY: TlbEntry = TlbEntry {
    tag: u64::MAX,
    addend: 0,
    _pad: 0,
};

/// Where a pc's translation is.
#[derive(Clone, Copy)]
struct Tr {
    slot: u32,
    idx: u32,
    region: u32,
}

/// One chain table entry (layout: `wasmshim_aarch64_translate::abi::CHAIN_ENTRY_SIZE`).
#[repr(C)]
#[derive(Clone, Copy)]
struct ChainEntry {
    pc: u64,
    slot: u32,
    block: u32,
    epoch: u64,
}

const CHAIN_EMPTY: ChainEntry = ChainEntry {
    pc: u64::MAX,
    slot: 0,
    block: 0,
    epoch: u64::MAX,
};

const CHAIN: u64 = wasmshim_aarch64_translate::abi::CHAIN_ENTRIES;
const WAYS: u64 = wasmshim_aarch64_translate::abi::CHAIN_WAYS;

/// Pages that may have chain entries: one bit per page number modulo this many (a shared bit
/// only costs a scan).
const CHAIN_PAGE_BITS: u64 = 1 << 16;

fn chain_page_bit(vpn: u64) -> (usize, u64) {
    let i = vpn & (CHAIN_PAGE_BITS - 1);
    ((i / 64) as usize, 1 << (i % 64))
}

/// The first entry of `pc`'s chain set (its [`WAYS`] entries follow).
fn chain_index(pc: u64) -> usize {
    (((pc >> 2) & (CHAIN - 1)) * WAYS) as usize
}

/// "No translation" (negative cache entry).
const NONE: Tr = Tr {
    slot: u32::MAX,
    idx: 0,
    region: 0,
};

struct RegionRec {
    version: u64,
    checked: u32,
    pcs: Vec<u64>,
}

/// A page as the run-time tier sees it ([`EngineMem::hot_page`]): (address space, page address,
/// code version), its bytes, and the interpreted executions of its blocks (page offset, count).
pub type HotPage = ((u64, u64, u64), Vec<u8>, Vec<(u16, u32)>);

/// Engine statistics (never guest-visible).
#[derive(Default)]
pub struct MemStats {
    pub regions: u64,
    pub aot_regions: u64,
    /// Regions installed by the run-time tier.
    pub jit_regions: u64,
    pub killed: u64,
}

/// A chain table entry for a target, seen from the dispatcher ([`EngineMem::chain_state`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainState {
    /// No entry for the target (never dispatched, untranslated, or its slot taken by another).
    Absent,
    /// An entry from an older code epoch.
    Stale,
    /// A current entry.
    Current,
}

pub struct EngineMem {
    pub(crate) mem: PagedMemory,
    /// Read entries, then write entries ([`wasmshim_aarch64_translate::abi::TLB_WRITE`]).
    tlb: Box<[Cell<TlbEntry>]>,
    /// Fill the TLB on successful accesses (only while the slow path of translated code runs).
    pub(crate) filling: Cell<bool>,
    /// TLB indices that may be non-empty.
    filled: RefCell<Vec<u32>>,
    map: BTreeMap<u64, Tr>,
    fast: Box<[(u64, Tr)]>,
    regions: Vec<Option<RegionRec>>,
    epoch: u64,
    /// [`wasmshim_cpu::SharedPages::code_epoch`] at the last `sync`.
    shared_code: u64,
    /// [`wasmshim_cpu::SharedPages::write_epoch`] at the last `sync`.
    shared_writes: u64,
    generation: u32,
    counts: Box<[(u64, u32)]>,
    /// Ahead-of-time translation state per code page (vpn).
    aot_pages: BTreeMap<u64, PageAot>,
    /// The chain table translated code tail-calls through (`wasmshim_aarch64_translate::abi::CHAIN_ENTRIES`).
    chain: Box<[ChainEntry]>,
    /// The epoch chain entries must carry: moves only when code changes are not listed per
    /// page ([`PagedMemory::take_code_changes`]).
    chain_epoch: u64,
    /// Pages that may have chain entries ([`chain_page_bit`]).
    chain_pages: Box<[u64]>,
    pub stats: MemStats,
}

/// Whether a page was looked up in the AOT directory, for which version of its code.
#[derive(Clone, Copy)]
struct PageAot {
    version: u64,
    probes: u32,
    done: bool,
}

/// Distinct untranslated block entries seen in a page before it is hashed and looked up in
/// the AOT directory (hashing costs a few microseconds per page).
const AOT_PROBES: u32 = 2;

fn vpn_index(addr: u64) -> usize {
    ((addr >> 12) as usize) & (TLB - 1)
}

impl EngineMem {
    /// The pages underneath, to read: residency, code versions, page contents. Writes go
    /// through [`GuestMemory`], which keeps the TLB and the translations in step.
    pub fn paged(&self) -> &PagedMemory {
        &self.mem
    }

    pub fn new(mem: PagedMemory) -> Self {
        Self {
            mem,
            tlb: (0..2 * TLB).map(|_| Cell::new(EMPTY)).collect(),
            filling: Cell::new(false),
            filled: RefCell::new(Vec::new()),
            map: BTreeMap::new(),
            // An empty slot is a "no translation" entry for pc `u64::MAX`: a guest may branch
            // there, and its lookup must find nothing (it then faults in the interpreter).
            fast: vec![(u64::MAX, NONE); FAST].into_boxed_slice(),
            regions: Vec::new(),
            epoch: u64::MAX,
            shared_code: 0,
            shared_writes: 0,
            generation: 0,
            counts: vec![(u64::MAX, 0); COUNTERS].into_boxed_slice(),
            aot_pages: BTreeMap::new(),
            chain: vec![CHAIN_EMPTY; (CHAIN * WAYS) as usize].into_boxed_slice(),
            chain_epoch: 0,
            chain_pages: vec![0; (CHAIN_PAGE_BITS / 64) as usize].into_boxed_slice(),
            stats: MemStats::default(),
        }
    }

    /// Address of the chain table, and the epoch its entries must carry.
    pub fn chain(&self) -> (usize, u64) {
        (self.chain.as_ptr() as usize, self.chain_epoch)
    }

    /// Why translated code could not chain to `pc` (statistics only): the chain table's entry
    /// for it, as of the last `sync`.
    pub fn chain_state(&self, pc: u64) -> ChainState {
        let set = chain_index(pc);
        let Some(e) = self.chain[set..set + WAYS as usize]
            .iter()
            .find(|e| e.pc == pc)
        else {
            return ChainState::Absent;
        };
        if e.epoch != self.chain_epoch {
            ChainState::Stale
        } else {
            ChainState::Current
        }
    }

    /// Statistics: how many block entries of `pc`'s page have a translation installed.
    pub fn translated_entries(&self, pc: u64) -> usize {
        let page = pc & !PAGE_MASK;
        self.map
            .range(page..page + PAGE_SIZE)
            .filter(|(_, t)| t.slot != NONE.slot)
            .count()
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Address of the TLB (the `tlb` argument of translated code).
    pub fn tlb_ptr(&self) -> usize {
        self.tlb.as_ptr() as usize
    }

    // ---- TLB ----
    //
    // Invalidation costs what was filled, not the TLB's size: `filled` lists the indices that
    // were set since they were last emptied (with duplicates; compacted when it grows).

    fn set_entry(&self, write: bool, addr: u64, host: usize) {
        let page = addr & !PAGE_MASK;
        let i = vpn_index(addr) + if write { TLB } else { 0 };
        if self.tlb[i].get().tag == u64::MAX {
            let mut filled = self.filled.borrow_mut();
            if filled.len() >= 4 * TLB {
                filled.retain(|&j| self.tlb[j as usize].get().tag != u64::MAX);
                filled.sort_unstable();
                filled.dedup();
            }
            filled.push(i as u32);
        }
        self.tlb[i].set(TlbEntry {
            tag: page,
            addend: (host as u32).wrapping_sub(page as u32),
            _pad: 0,
        });
    }

    fn fill_read(&self, addr: u64) {
        if let Some(host) = self.mem.host_read_page(addr) {
            self.set_entry(false, addr, host);
        }
    }

    /// A write entry, and the read entry for the same (now private, resident) page.
    fn fill_write(&mut self, addr: u64) {
        if let Some(host) = self.mem.host_write_page(addr) {
            self.set_entry(true, addr, host);
            self.set_entry(false, addr, host);
        }
    }

    /// For translated code's TLB-miss path: fills the entry (or, for a write, both entries) of
    /// `addr`'s page and returns the page's host address, if the access is allowed (a write
    /// never to code). Runs no instruction and changes no guest-visible state (a page may become
    /// private or resident, as the store that follows would make it anyway).
    pub fn fill_for_jit(&mut self, addr: u64, write: bool) -> Option<usize> {
        if write {
            let host = self.mem.host_write_page(addr)?;
            self.set_entry(true, addr, host);
            self.set_entry(false, addr, host);
            Some(host)
        } else {
            let host = self.mem.host_read_page(addr)?;
            self.set_entry(false, addr, host);
            Some(host)
        }
    }

    /// Drops the filled entries whose page satisfies `drop` (`writes`: in the write table too).
    fn drop_where(&self, writes: bool, drop: impl Fn(u64) -> bool) {
        let mut filled = self.filled.borrow_mut();
        filled.retain(|&i| {
            let i = i as usize;
            let e = self.tlb[i].get();
            if e.tag == u64::MAX {
                return false;
            }
            if (writes || i < TLB) && drop(e.tag) {
                self.tlb[i].set(EMPTY);
                return false;
            }
            true
        });
    }

    /// Drops the entries of `[addr, addr+len)` (`writes`: in the write table too).
    fn invalidate(&mut self, addr: u64, len: u64, writes: bool) {
        let first = addr & !PAGE_MASK;
        let end = addr.saturating_add(len).saturating_add(PAGE_MASK) & !PAGE_MASK;
        let pages = (end - first) >> 12;
        if pages as usize > self.filled.borrow().len() {
            self.drop_where(writes, |tag| tag >= first && tag < end);
            return;
        }
        for p in 0..pages {
            let page = first + (p << 12);
            let i = vpn_index(page);
            if self.tlb[i].get().tag == page {
                self.tlb[i].set(EMPTY);
            }
            if writes && self.tlb[TLB + i].get().tag == page {
                self.tlb[TLB + i].set(EMPTY);
            }
        }
    }

    /// Pages are about to be shared copy-on-write: no direct writes any more.
    pub fn drop_write_entries(&self) {
        let mut filled = self.filled.borrow_mut();
        filled.retain(|&i| {
            let i = i as usize;
            if i >= TLB {
                self.tlb[i].set(EMPTY);
                return false;
            }
            true
        });
    }

    /// A memory object that tracks its stores started a new period: its pages must not be
    /// written directly any more until they are asked for again (and recorded).
    fn drop_shared_write_entries(&self) {
        let mut filled = self.filled.borrow_mut();
        filled.retain(|&i| {
            let i = i as usize;
            if i < TLB {
                return true;
            }
            let e = self.tlb[i].get();
            if e.tag == u64::MAX {
                return false;
            }
            if self.mem.is_shared(e.tag) {
                self.tlb[i].set(EMPTY);
                return false;
            }
            true
        });
    }

    // ---- translations ----

    fn sync(&mut self) {
        let w = wasmshim_cpu::SharedPages::write_epoch();
        if w != self.shared_writes {
            self.shared_writes = w;
            self.drop_shared_write_entries();
        }
        let e = self.mem.code_epoch();
        if e != self.epoch {
            self.epoch = e;
            self.generation = self.generation.wrapping_add(1);
            // A shared object may have become executable: its pages take the slow path now.
            let shared = wasmshim_cpu::SharedPages::code_epoch();
            let shared_moved = shared != self.shared_code;
            if shared_moved {
                self.shared_code = shared;
                self.drop_write_entries();
            }
            // Chain entries stay valid unless their target's page changed:
            // drop the changed pages' entries, or all of them when the change is not listed.
            match self.mem.take_code_changes() {
                Some(pages) if !shared_moved => {
                    for vpn in pages {
                        self.unchain_page(vpn);
                    }
                }
                _ => {
                    self.chain_epoch = self.chain_epoch.wrapping_add(1);
                    self.chain_pages.fill(0);
                }
            }
        }
    }

    /// Clears the chain entries whose targets are in page `vpn` (if it may have any).
    fn unchain_page(&mut self, vpn: u64) {
        let (word, bit) = chain_page_bit(vpn);
        if self.chain_pages[word] & bit == 0 {
            return;
        }
        self.chain_pages[word] &= !bit;
        // The page's 1 024 instruction slots index 1 024 consecutive sets.
        for i in 0..(PAGE_MASK + 1) / 4 {
            let set = chain_index((vpn << 12) + 4 * i);
            for e in &mut self.chain[set..set + WAYS as usize] {
                if e.pc != u64::MAX && e.pc >> 12 == vpn {
                    *e = CHAIN_EMPTY;
                }
            }
        }
    }

    /// The translation for `pc`, if there is a valid one: `(table slot, block index)`.
    #[inline]
    pub fn lookup(&mut self, pc: u64) -> Option<(u32, u32)> {
        self.sync();
        let h = ((pc >> 2) as usize) & (FAST - 1);
        let tr = if self.fast[h].0 == pc {
            self.fast[h].1
        } else {
            // Cache misses too: most dispatches of cold code find nothing.
            let tr = match self.map.get(&pc).copied() {
                Some(tr) => tr,
                None if self.try_aot(pc) => self.map.get(&pc).copied().unwrap_or(NONE),
                None => NONE,
            };
            self.fast[h] = (pc, tr);
            tr
        };
        if tr.slot == NONE.slot {
            return None;
        }
        let generation = self.generation;
        let rec = self.regions[tr.region as usize].as_mut()?;
        if rec.checked != generation {
            if self.mem.code_version(pc) != rec.version {
                self.kill(tr.region);
                return None;
            }
            rec.checked = generation;
        }
        // Validated for this epoch: translated code may chain straight to it.
        // Into its set: over its own entry, else in front (the older one moves to way 1).
        let entry = ChainEntry {
            pc,
            slot: tr.slot,
            block: tr.idx,
            epoch: self.chain_epoch,
        };
        let set = chain_index(pc);
        let (word, bit) = chain_page_bit(pc >> 12);
        self.chain_pages[word] |= bit;
        if self.chain[set].pc == pc {
            self.chain[set] = entry;
        } else if self.chain[set + 1].pc == pc {
            self.chain[set + 1] = entry;
        } else {
            self.chain[set + 1] = self.chain[set];
            self.chain[set] = entry;
        }
        Some((tr.slot, tr.idx))
    }

    /// Installs the ahead-of-time translations of `pc`'s page, if the directory has its
    /// content. `true` if anything was installed.
    fn try_aot(&mut self, pc: u64) -> bool {
        let Some(dir) = crate::aot::directory() else {
            return false;
        };
        let vpn = pc >> 12;
        let version = self.mem.code_version(pc);
        let st = self.aot_pages.entry(vpn).or_insert(PageAot {
            version,
            probes: 0,
            done: false,
        });
        if st.version != version {
            *st = PageAot {
                version,
                probes: 0,
                done: false,
            };
        }
        if st.done {
            return false;
        }
        st.probes += 1;
        if st.probes < AOT_PROBES {
            return false;
        }
        st.done = true;
        let Some(bytes) = self.mem.page_bytes(pc) else {
            return false;
        };
        // A fast hash finds candidates; a whole-page compare confirms them.
        let entries: Vec<wasmshim_aarch64_translate::aot::Entry> = dir.lookup_page(bytes).collect();
        let page = pc & !PAGE_MASK;
        let mut any = false;
        let mut func = u32::MAX;
        let mut region = 0;
        for e in entries {
            if e.func != func {
                func = e.func;
                region = self.regions.len() as u32;
                self.regions.push(Some(RegionRec {
                    version,
                    checked: self.generation,
                    pcs: Vec::new(),
                }));
                self.stats.regions += 1;
                self.stats.aot_regions += 1;
            }
            let at = page + u64::from(e.offset);
            let tr = Tr {
                slot: e.func,
                idx: u32::from(e.block),
                region,
            };
            self.map.insert(at, tr);
            let h = ((at >> 2) as usize) & (FAST - 1);
            self.fast[h] = (at, tr);
            if let Some(Some(rec)) = self.regions.get_mut(region as usize) {
                rec.pcs.push(at);
            }
            any = true;
        }
        any
    }

    fn kill(&mut self, region: u32) {
        let Some(rec) = self.regions[region as usize].take() else {
            return;
        };
        self.stats.killed += 1;
        for pc in rec.pcs {
            if self.map.get(&pc).is_some_and(|t| t.region == region) {
                self.map.remove(&pc);
            }
            let h = ((pc >> 2) as usize) & (FAST - 1);
            if self.fast[h].0 == pc {
                self.fast[h] = (u64::MAX, NONE);
            }
            let set = chain_index(pc);
            for e in &mut self.chain[set..set + WAYS as usize] {
                if e.pc == pc {
                    *e = CHAIN_EMPTY;
                }
            }
        }
    }

    /// Counts one execution of the block at `pc`; returns the new count.
    #[inline]
    pub fn bump(&mut self, pc: u64) -> u32 {
        let e = &mut self.counts[((pc >> 2) as usize) & (COUNTERS - 1)];
        if e.0 == pc {
            e.1 = e.1.saturating_add(1);
        } else {
            *e = (pc, 1);
        }
        e.1
    }

    fn count(&self, pc: u64) -> u32 {
        let e = self.counts[((pc >> 2) as usize) & (COUNTERS - 1)];
        if e.0 == pc { e.1 } else { 0 }
    }

    /// The run-time tier's view of `pc`'s page: its identity (this address
    /// space, the page address, its code version), a copy of its bytes, and the interpreted
    /// executions of its blocks (page offset, count; blocks that ran at least once). `None` if
    /// the page is not readable.
    pub fn hot_page(&self, pc: u64) -> Option<HotPage> {
        let page = pc & !PAGE_MASK;
        let bytes = self.mem.page_bytes(pc)?.to_vec();
        let version = self.mem.code_version(pc);
        let warm = (0..PAGE_SIZE)
            .step_by(4)
            .filter_map(|o| {
                let n = self.count(page + o);
                (n > 0).then_some((o as u16, n))
            })
            .collect();
        // The address is only a tag: the version is never reused by another address space
        // (`GuestMemory::code_version`), so an `EngineMem` allocated where a freed one was
        // cannot inherit its per-page tier state.
        let space = self as *const Self as usize as u64;
        Some(((space, page, version), bytes, warm))
    }

    /// Installs run-time translations of the page at `page` for code version `version`: each
    /// entry is (page offset, table slot of its region function, block index), grouped by
    /// function (`wasmshim_translate::tier::Install`). Nothing is installed if the page's code
    /// changed since `version` was read.
    pub fn install_page(&mut self, page: u64, version: u64, entries: &[(u16, u32, u16)]) {
        self.sync();
        if self.mem.code_version(page) != version {
            return;
        }
        let mut func = u32::MAX;
        let mut region = 0;
        for &(offset, slot, block) in entries {
            if slot != func {
                func = slot;
                region = self.regions.len() as u32;
                self.regions.push(Some(RegionRec {
                    version,
                    checked: self.generation,
                    pcs: Vec::new(),
                }));
                self.stats.regions += 1;
                self.stats.jit_regions += 1;
            }
            let at = page + u64::from(offset);
            let tr = Tr {
                slot,
                idx: u32::from(block),
                region,
            };
            self.map.insert(at, tr);
            let h = ((at >> 2) as usize) & (FAST - 1);
            self.fast[h] = (at, tr);
            if let Some(Some(rec)) = self.regions.get_mut(region as usize) {
                rec.pcs.push(at);
            }
        }
    }
}

impl GuestMemory for EngineMem {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        let r = self.mem.read(addr, buf);
        if r.is_ok() {
            self.note_read(addr);
        }
        r
    }

    fn write(&mut self, addr: u64, data: &[u8]) -> Result<(), MemFault> {
        let r = self.mem.write(addr, data);
        self.after_write(addr, data.len() as u64, r.is_ok());
        r
    }

    fn fetch(&mut self, addr: u64) -> Result<u32, MemFault> {
        self.mem.fetch(addr)
    }

    fn check(&self, addr: u64, len: u64, access: Access) -> Result<(), MemFault> {
        self.mem.check(addr, len, access)
    }

    fn poke(&mut self, addr: u64, data: &[u8]) -> Result<(), MemFault> {
        let r = self.mem.poke(addr, data);
        self.invalidate(addr, data.len() as u64, false);
        r
    }

    fn map(&mut self, addr: u64, len: u64, perms: Perms) -> Result<(), MapError> {
        let r = self.mem.map(addr, len, perms);
        self.invalidate(addr, len, true);
        r
    }

    fn map_source(
        &mut self,
        addr: u64,
        len: u64,
        perms: Perms,
        source: Rc<dyn PageSource>,
        first: u64,
    ) -> Result<(), MapError> {
        let r = self.mem.map_source(addr, len, perms, source, first);
        self.invalidate(addr, len, true);
        r
    }

    fn map_shared(
        &mut self,
        addr: u64,
        len: u64,
        perms: Perms,
        object: Rc<dyn SharedMemory>,
        first: u64,
    ) -> Result<(), MapError> {
        let r = self.mem.map_shared(addr, len, perms, object, first);
        self.invalidate(addr, len, true);
        r
    }

    fn unmap(&mut self, addr: u64, len: u64) -> Result<(), MapError> {
        let r = self.mem.unmap(addr, len);
        self.invalidate(addr, len, true);
        r
    }

    fn protect(&mut self, addr: u64, len: u64, perms: Perms) -> Result<(), MapError> {
        let r = self.mem.protect(addr, len, perms);
        self.invalidate(addr, len, true);
        r
    }

    fn perms(&self, addr: u64) -> Option<Perms> {
        self.mem.perms(addr)
    }

    /// The watch view is `PagedMemory`'s; the TLB drops the pages whose effective
    /// permissions change (the old view's and the new one's), so translated code takes the
    /// slow path there and faults like the interpreter.
    fn set_watch(&mut self, pages: &[(u64, Perms)]) -> bool {
        let before: alloc::vec::Vec<u64> = self.mem.watched_pages().collect();
        let ok = self.mem.set_watch(pages);
        for page in before.into_iter().chain(pages.iter().map(|&(a, _)| a)) {
            self.invalidate(page, 1, true);
        }
        ok
    }

    fn address_limit(&self) -> u64 {
        self.mem.address_limit()
    }

    fn code_epoch(&self) -> u64 {
        self.mem.code_epoch()
    }

    fn code_version(&self, addr: u64) -> u64 {
        self.mem.code_version(addr)
    }

    fn duplicate(&self) -> Box<dyn GuestMemory> {
        // The pages become shared: no direct writes through this TLB any more.
        self.drop_write_entries();
        Box::new(EngineMem::new(self.mem.clone()))
    }

    fn as_any(&self) -> Option<&dyn core::any::Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn core::any::Any> {
        Some(self)
    }

    fn for_each_mapped(
        &self,
        f: &mut dyn FnMut(u64, u64, Perms, wasmshim_cpu::PageView<'_>),
    ) -> Result<(), MapError> {
        self.mem.for_each_mapped(f)
    }

    // The typed accessors are the interpreter's hot path: keep PagedMemory's fast paths, and
    // fill the TLB while translated code's slow path runs.
    fn read_u8(&self, addr: u64) -> Result<u8, MemFault> {
        let r = self.mem.read_u8(addr);
        if r.is_ok() {
            self.note_read(addr);
        }
        r
    }
    fn read_u16(&self, addr: u64) -> Result<u16, MemFault> {
        let r = self.mem.read_u16(addr);
        if r.is_ok() {
            self.note_read(addr);
        }
        r
    }
    fn read_u32(&self, addr: u64) -> Result<u32, MemFault> {
        let r = self.mem.read_u32(addr);
        if r.is_ok() {
            self.note_read(addr);
        }
        r
    }
    fn read_u64(&self, addr: u64) -> Result<u64, MemFault> {
        let r = self.mem.read_u64(addr);
        if r.is_ok() {
            self.note_read(addr);
        }
        r
    }
    fn write_u8(&mut self, addr: u64, v: u8) -> Result<(), MemFault> {
        let r = self.mem.write_u8(addr, v);
        self.after_write(addr, 1, r.is_ok());
        r
    }
    fn write_u16(&mut self, addr: u64, v: u16) -> Result<(), MemFault> {
        let r = self.mem.write_u16(addr, v);
        self.after_write(addr, 2, r.is_ok());
        r
    }
    fn write_u32(&mut self, addr: u64, v: u32) -> Result<(), MemFault> {
        let r = self.mem.write_u32(addr, v);
        self.after_write(addr, 4, r.is_ok());
        r
    }
    fn write_u64(&mut self, addr: u64, v: u64) -> Result<(), MemFault> {
        let r = self.mem.write_u64(addr, v);
        self.after_write(addr, 8, r.is_ok());
        r
    }
}

impl EngineMem {
    /// After a write through the interpreter or the personality: the page may have become
    /// resident or private (a new host address), so its read entry is stale. While translated
    /// code's slow path runs, refill both entries.
    #[inline]
    fn after_write(&mut self, addr: u64, len: u64, ok: bool) {
        if len <= 16 && (addr & PAGE_MASK) + len <= PAGE_SIZE {
            let i = vpn_index(addr);
            if self.tlb[i].get().tag == addr & !PAGE_MASK {
                self.tlb[i].set(EMPTY);
            }
        } else {
            self.invalidate(addr, len, false);
        }
        if ok && self.filling.get() {
            self.fill_write(addr);
        }
    }

    /// While translated code's slow path runs: a successful read makes its page readable
    /// through the TLB.
    #[inline]
    fn note_read(&self, addr: u64) {
        if self.filling.get() {
            self.fill_read(addr);
        }
    }
}
