//! Translations keyed by page content, and ahead-of-time translation: translated
//! regions in a relocatable Wasm object file that the normal linker links into the module that
//! runs the guest.
//!
//! * **Keys are page contents.** A translation is a function of a 4 KiB code page's bytes
//!   and of which blocks in it are hot ([`PageProfile`]); it is position-independent, so the
//!   translation of a page of `libc.so` serves that page wherever it is loaded, in any image.
//!   The key is the first 16 bytes of the page's SHA-256 ([`page_key`]), with the front half's
//!   translator version in the directory header.
//! * **The object** defines `<prefix>_aot_directory() -> *const u8` and
//!   `<prefix>_aot_directory_len() -> usize` (the [`Directory`]), the front half's helper
//!   functions and one function per region; translated code calls the undefined symbols
//!   `<prefix>_jit_step(pc: i64) -> i32` and `<prefix>_jit_fill`, which the engine defines. The
//!   directory's function fields are table-index relocations, so the linker puts every region
//!   function in the indirect function table and the engine calls them through it.
//! * An object with no pages is valid: engines link one when there is nothing to translate.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use sha2::{Digest, Sha256};

use crate::emit::{Body, Callee, Callees, Helper};
use crate::region::{Block, Isa, Region};

/// `"AOT1"`.
pub const DIRECTORY_MAGIC: u32 = u32::from_le_bytes(*b"AOT1");
/// Content key of a code page.
pub type PageKey = [u8; 16];

/// Bytes per page.
pub const PAGE: usize = crate::region::PAGE as usize;
/// Most blocks per region function.
pub const REGION_BLOCKS: usize = 64;
/// Most instructions per region function.
pub const REGION_INSNS: usize = 2048;

/// The key of a page: the first 16 bytes of its SHA-256.
pub fn page_key(page: &[u8]) -> PageKey {
    let digest = Sha256::digest(page);
    let mut key = [0u8; 16];
    key.copy_from_slice(&digest[..16]);
    key
}

/// A fast 64-bit hash of a page: the directory's first-level key. Four independent
/// multiply-mix lanes over the page's 64-bit words, so it runs at a few hundred bytes per
/// hundred cycles instead of SHA-256's tens of cycles per byte. It only *finds* candidates:
/// a page is accepted only if its whole content equals the translated page, so a collision of
/// this hash costs a page comparison, never a wrong translation.
pub fn page_fast(page: &[u8]) -> u64 {
    const K: [u64; 4] = [
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
        0x1656_67b1_9e37_79f9,
        0xd6e8_feb8_6659_fd93,
    ];
    let mut h = [K[0] ^ page.len() as u64, K[1], K[2], K[3]];
    for (i, w) in page.as_chunks::<8>().0.iter().enumerate() {
        let w = u64::from_le_bytes(*w);
        let l = i & 3;
        h[l] = (h[l] ^ w).wrapping_mul(K[l]).rotate_left(31);
    }
    let mut x = h[0] ^ h[1].rotate_left(17) ^ h[2].rotate_left(29) ^ h[3].rotate_left(43);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x
}

/// A code page and the offsets of its hot block entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageProfile {
    pub bytes: Vec<u8>,
    /// Offsets (multiples of 4) of blocks to translate.
    pub entries: Vec<u16>,
}

/// The regions of one page, ready to emit.
#[derive(Clone, Debug)]
pub struct PageTranslation<A: Isa> {
    pub key: PageKey,
    /// [`page_fast`] of the page.
    pub fast: u64,
    /// The page itself: the directory keeps it, and a lookup compares it byte for byte.
    pub bytes: Vec<u8>,
    /// Regions with block PCs relative to page address 0.
    pub regions: Vec<Region<A>>,
}

impl<A: Isa> PageTranslation<A> {
    /// The translation of `bytes` made of `regions`.
    pub fn new(bytes: &[u8], regions: Vec<Region<A>>) -> Self {
        Self {
            key: page_key(bytes),
            fast: page_fast(bytes),
            bytes: bytes.to_vec(),
            regions,
        }
    }
}

/// Install-time translation (Rosetta 2's shape): the block entries of a
/// code page found without running it, by a linear sweep: the page start, every instruction
/// after one that ends a block (a call's return address, a branch's fall-through, the code
/// after a return), and the targets of direct branches inside the page. Data in a code page
/// decodes to blocks that never run, which costs only size; an entry the sweep missed (an
/// indirect target in the middle of straight-line code) runs interpreted, or in the run-time
/// tier where there is one.
pub fn static_entries<A: Isa>(bytes: &[u8]) -> Vec<u16> {
    let end = bytes.len().min(PAGE);
    let mut entries = alloc::collections::BTreeSet::new();
    entries.insert(0u16);
    let mut off = 0;
    while off < end {
        let Some(inst) = A::decode(bytes, off) else {
            off += A::ALIGN;
            if off < end {
                entries.insert(off as u16);
            }
            continue;
        };
        let len = (A::inst_len(&inst) as usize).max(A::ALIGN);
        if A::ends_block(&inst) {
            let block = Block::<A> {
                pc: off as u64,
                insts: alloc::vec![inst],
            };
            for t in A::successors(&block) {
                if t < end as u64 {
                    entries.insert(t as u16);
                }
            }
            if off + len < end {
                entries.insert((off + len) as u16);
            }
        }
        off += len;
    }
    entries.into_iter().collect()
}

/// Groups the decoded blocks of one page (sorted by PC, distinct PCs; every hot entry starts
/// one) into regions. With `loops`, every single-entry loop nest among
/// the blocks becomes a loop region of its own, entered only at its header. The other blocks,
/// the loops' non-header blocks included (a side exit or a budget exit inside a loop resumes
/// there), are grouped in address order into page regions of at most [`REGION_BLOCKS`] blocks
/// and [`REGION_INSNS`] instructions, every block an entry; a branch from a page region to a
/// loop header leaves the page region and chains into the loop.
pub fn page_regions<A: Isa>(blocks: Vec<Block<A>>, loops: bool) -> Vec<Region<A>> {
    page_regions_capped(blocks, loops, REGION_BLOCKS, REGION_INSNS)
}

/// [`page_regions`] with page regions of at most `max_blocks` blocks and `max_insns`
/// instructions (smaller functions, fewer bytes compiled at a first call).
pub fn page_regions_capped<A: Isa>(
    blocks: Vec<Block<A>>,
    loops: bool,
    max_blocks: usize,
    max_insns: usize,
) -> Vec<Region<A>> {
    let (max_blocks, max_insns) = (max_blocks.max(1), max_insns.max(1));
    let mut regions = Vec::new();
    let mut headers = alloc::collections::BTreeSet::new();
    if loops {
        // Loops are found among the blocks that start at a branch target or are not inside
        // another block, each ending where the next branch target starts (a block falling
        // through into a loop header is not part of the loop). A hot entry in the middle of
        // another block that nothing branches to is where a quantum, a syscall or a side exit
        // resumed: it stays an entry of a page region and is not part of any loop. Page
        // regions keep the longer, overlapping blocks.
        let targets: alloc::collections::BTreeSet<u64> =
            blocks.iter().flat_map(A::successors).collect();
        let mut split = Vec::new();
        let mut covered = 0; // the furthest end of the blocks so far (sorted by pc)
        for b in &blocks {
            let inside = covered > b.pc;
            covered = covered.max(b.end());
            if inside && !targets.contains(&b.pc) {
                continue;
            }
            let mut b = b.clone();
            if let Some(&t) = targets.range(b.pc + 1..b.end()).next() {
                // Keep the instructions before the target.
                let mut at = b.pc;
                let mut keep = 0;
                while keep < b.insts.len() && at < t {
                    at += A::inst_len(&b.insts[keep]);
                    keep += 1;
                }
                b.insts.truncate(keep);
            }
            split.push(b);
        }
        for r in crate::nest::loop_regions(&split) {
            headers.insert(r.blocks[0].pc);
            regions.push(r);
        }
    }
    let mut cur = Region::default();
    let mut insns = 0;
    for b in blocks.into_iter().filter(|b| !headers.contains(&b.pc)) {
        if !cur.blocks.is_empty()
            && (cur.blocks.len() >= max_blocks || insns + b.insts.len() > max_insns)
        {
            regions.push(core::mem::take(&mut cur));
            insns = 0;
        }
        insns += b.insts.len();
        cur.blocks.push(b);
    }
    if !cur.blocks.is_empty() {
        regions.push(cur);
    }
    regions
}

// ---- the directory ----

/// Header: magic, version, page count, entry count (u32 each).
const HEADER: usize = 16;
/// Page record: fast key (u64), key (16), first entry (u32), entry count (u32); sorted by fast
/// key, then key. After the entry records come the pages themselves, 4 KiB each, in record
/// order: a lookup confirms a fast-key candidate by comparing the whole page, which is
/// both cheaper than SHA-256 and exact.
const PAGE_RECORD: usize = 32;
/// Entry record: function (u32 table index), page offset (u16), block index (u16).
const ENTRY_RECORD: usize = 8;

/// A read-only view of a directory (as linked into the engine).
#[derive(Clone, Copy)]
pub struct Directory<'a> {
    bytes: &'a [u8],
    pages: usize,
    /// Number of entry records.
    entries: usize,
    /// Offset of the page contents.
    data: usize,
}

/// One translated block entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    /// Indirect-function-table index of the region function.
    pub func: u32,
    /// Byte offset of the block in the page.
    pub offset: u16,
    /// The block's index in its region (the function's entry argument).
    pub block: u16,
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

impl<'a> Directory<'a> {
    /// Parses a directory; `None` if it is malformed or from another translator `version`.
    pub fn new(bytes: &'a [u8], version: u32) -> Option<Self> {
        if bytes.len() < HEADER
            || u32_at(bytes, 0) != DIRECTORY_MAGIC
            || u32_at(bytes, 4) != version
        {
            return None;
        }
        let pages = u32_at(bytes, 8) as usize;
        let entries = u32_at(bytes, 12) as usize;
        // Checked: on a 32-bit engine a corrupt count overflows `usize`.
        let data = pages
            .checked_mul(PAGE_RECORD)
            .and_then(|p| p.checked_add(entries.checked_mul(ENTRY_RECORD)?))
            .and_then(|r| r.checked_add(HEADER))?;
        let end = pages.checked_mul(PAGE).and_then(|p| p.checked_add(data))?;
        if bytes.len() < end {
            return None;
        }
        Some(Self {
            bytes,
            pages,
            entries,
            data,
        })
    }

    /// Number of translated pages.
    pub fn pages(&self) -> usize {
        self.pages
    }

    fn fast(&self, i: usize) -> u64 {
        let at = HEADER + i * PAGE_RECORD;
        u64::from(u32_at(self.bytes, at)) | u64::from(u32_at(self.bytes, at + 4)) << 32
    }

    /// The stored contents of page record `i`.
    fn page(&self, i: usize) -> &[u8] {
        &self.bytes[self.data + i * PAGE..self.data + (i + 1) * PAGE]
    }

    /// The translated entries of the page with these contents: candidates by [`page_fast`],
    /// confirmed by comparing the whole page with the one that was translated.
    pub fn lookup_page(&self, page: &[u8]) -> impl Iterator<Item = Entry> + '_ {
        let fast = page_fast(page);
        let (mut lo, mut hi) = (0, self.pages);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.fast(mid) < fast {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut found = None;
        if lo < self.pages && self.fast(lo) == fast {
            let mut i = lo;
            while i < self.pages && self.fast(i) == fast {
                if self.page(i) == page {
                    found = Some(i);
                    break;
                }
                i += 1;
            }
        }
        let (first, count) = found.map_or((0, 0), |i| {
            let at = HEADER + i * PAGE_RECORD + 24;
            (
                u32_at(self.bytes, at) as usize,
                u32_at(self.bytes, at + 4) as usize,
            )
        });
        // A page record naming entries past the entry records is malformed: no entries.
        let end = first
            .checked_add(count)
            .filter(|&end| end <= self.entries)
            .unwrap_or(first);
        let base = HEADER + self.pages * PAGE_RECORD;
        (first..end).map(move |e| {
            let at = base + e * ENTRY_RECORD;
            Entry {
                func: u32_at(self.bytes, at),
                offset: u16::from_le_bytes([self.bytes[at + 4], self.bytes[at + 5]]),
                block: u16::from_le_bytes([self.bytes[at + 6], self.bytes[at + 7]]),
            }
        })
    }
}

// ---- the object file ----

const R_WASM_FUNCTION_INDEX_LEB: u8 = 0;
const R_WASM_TABLE_INDEX_I32: u8 = 2;
const R_WASM_MEMORY_ADDR_SLEB: u8 = 4;
const R_WASM_TYPE_INDEX_LEB: u8 = 6;

const SYM_FUNCTION: u8 = 0;
const SYM_DATA: u8 = 1;
const SYM_BINDING_LOCAL: u32 = 0x2;
const SYM_UNDEFINED: u32 = 0x10;

/// How a front half's translations are linked: into an object ([`object`]) or a standalone
/// module ([`crate::module`]).
#[derive(Clone, Copy, Debug)]
pub struct Linking<'a> {
    /// The front half's translator version (in a directory's header: an engine ignores
    /// translations of another version).
    pub version: u32,
    /// The prefix of the symbols the engine defines and reads: `<prefix>_jit_step`,
    /// `<prefix>_jit_fill`, `<prefix>_aot_directory`, ... (`"linux"` for `wasmshim-aarch64-engine`).
    pub prefix: &'a str,
    /// The front half's helper functions ([`crate::ir::Op::Helper`]), in index order.
    pub helpers: &'a [Helper],
    /// Direct chaining: a branch from one region to an entry of another
    /// region of the same page tail-calls it directly (`return_call`) instead of looking it up
    /// in the chain table.
    pub direct_chain: bool,
}

/// A back half: the body of `region` (whose page starts at `region.page()`) calling `callees`,
/// linked with `peers`: the PC of every entry of the other regions of its page, mapped to
/// (region number, block index); a branch to one of them may become a direct `return_call`
/// ([`Callee::Region`]).
pub type BackHalf<'b, A> = dyn FnMut(&Region<A>, Callees, &BTreeMap<u64, (u32, u32)>) -> Body + 'b;

/// For every region of a page, the entries of the page's other regions (direct chaining).
pub fn page_peers<A: Isa>(
    regions: &[Region<A>],
    first: u32,
    direct_chain: bool,
) -> Vec<BTreeMap<u64, (u32, u32)>> {
    let mut peers: Vec<BTreeMap<u64, (u32, u32)>> = alloc::vec![BTreeMap::new(); regions.len()];
    if direct_chain {
        for (i, peer) in peers.iter_mut().enumerate() {
            for (j, other) in regions.iter().enumerate() {
                if i == j {
                    continue;
                }
                for bi in other.entries() {
                    peer.insert(other.blocks[bi].pc, (first + j as u32, bi as u32));
                }
            }
        }
    }
    peers
}

/// The custom section of the branch-hinting proposal: per function, the offsets (from the start
/// of the function body, its locals included) of `if`s and whether their then-arm is likely.
pub const BRANCH_HINT_SECTION: &str = "metadata.code.branch_hint";

/// The hints of one function (after its index): a count, then `offset, 1, likely` each.
pub fn branch_hints(out: &mut Vec<u8>, hints: &[(usize, bool)]) {
    leb(out, hints.len() as u64);
    for &(offset, likely) in hints {
        leb(out, offset as u64);
        leb(out, 1);
        out.push(u8::from(likely));
    }
}

/// Appends `v` as an unsigned LEB128.
pub fn leb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn padded_leb(out: &mut Vec<u8>, v: u32) {
    crate::emit::push_padded_leb(out, v);
}

fn name(out: &mut Vec<u8>, s: &str) {
    leb(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

fn section(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
    out.push(id);
    leb(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

fn custom(out: &mut Vec<u8>, title: &str, payload: &[u8]) {
    let mut p = Vec::new();
    name(&mut p, title);
    p.extend_from_slice(payload);
    section(out, 0, &p);
}

/// Statistics of an emitted object.
#[derive(Clone, Copy, Debug, Default)]
pub struct ObjectStats {
    pub pages: usize,
    pub regions: usize,
    /// Loop regions among them.
    pub loop_regions: usize,
    pub blocks: usize,
    pub insns: usize,
    /// Bytes of function bodies (translated code).
    pub code_bytes: usize,
    /// Registers moved by region prologues and epilogues, summed over regions.
    pub loaded: usize,
    pub stored: usize,
}

/// Emits a relocatable Wasm object with the given page translations (see the module docs),
/// each region's body made by `back`.
pub fn object<A: Isa>(
    pages: &[PageTranslation<A>],
    link: &Linking<'_>,
    back: &mut BackHalf<'_, A>,
) -> (Vec<u8>, ObjectStats) {
    let mut pages: Vec<&PageTranslation<A>> = pages.iter().collect();
    pages.sort_by_key(|p| (p.fast, p.key));
    pages.dedup_by(|a, b| a.key == b.key);
    let prefix = link.prefix;
    let nhelpers = link.helpers.len() as u32;

    // Function index space: [0] <prefix>_jit_step, [1] <prefix>_jit_fill (imports), the
    // helpers, the regions, then the two directory accessors.
    let regions: Vec<&Region<A>> = pages.iter().flat_map(|p| p.regions.iter()).collect();
    let nregions = regions.len() as u32;
    let f_helper0 = 2;
    let f_region0 = f_helper0 + nhelpers;
    let f_dir = f_region0 + nregions;
    let f_dir_len = f_dir + 1;

    // Symbols: 0 step, 1 fill, the helpers, the regions, then dir, dir_len, data.
    let s_step = 0u32;
    let s_fill = 1u32;
    let s_helper0 = 2u32;
    let s_region0 = s_helper0 + nhelpers;
    let s_dir = s_region0 + nregions;
    let s_dir_len = s_dir + 1;
    let s_data = s_dir_len + 1;

    // Types: 0 step, then one per helper, then region, accessors, fill.
    let t_region = 1 + nhelpers;
    let t_accessor = t_region + 1;
    let t_fill = t_accessor + 1;

    // ---- the directory bytes (function fields relocated) ----
    let nentries: usize = regions.iter().map(|r| r.entries().len()).sum();
    let mut dir = Vec::new();
    dir.extend_from_slice(&DIRECTORY_MAGIC.to_le_bytes());
    dir.extend_from_slice(&link.version.to_le_bytes());
    dir.extend_from_slice(&(pages.len() as u32).to_le_bytes());
    dir.extend_from_slice(&(nentries as u32).to_le_bytes());
    let mut first = 0u32;
    for p in &pages {
        let n: usize = p.regions.iter().map(|r| r.entries().len()).sum();
        dir.extend_from_slice(&p.fast.to_le_bytes());
        dir.extend_from_slice(&p.key);
        dir.extend_from_slice(&first.to_le_bytes());
        dir.extend_from_slice(&(n as u32).to_le_bytes());
        first += n as u32;
    }
    let mut data_relocs = Vec::new(); // (offset in dir, symbol)
    let mut stats = ObjectStats {
        pages: pages.len(),
        regions: regions.len(),
        ..ObjectStats::default()
    };
    for (ri, r) in regions.iter().enumerate() {
        if r.nest.is_some() {
            stats.loop_regions += 1;
        }
        for (bi, b) in r.blocks.iter().enumerate() {
            stats.blocks += 1;
            stats.insns += b.insts.len();
            if !r.entries().contains(&bi) {
                continue;
            }
            data_relocs.push((dir.len(), s_region0 + ri as u32));
            dir.extend_from_slice(&0u32.to_le_bytes());
            dir.extend_from_slice(&((b.pc & (PAGE as u64 - 1)) as u16).to_le_bytes());
            dir.extend_from_slice(&(bi as u16).to_le_bytes());
        }
    }
    for p in &pages {
        debug_assert_eq!(p.bytes.len(), PAGE);
        dir.extend_from_slice(&p.bytes);
        dir.resize(dir.len() + PAGE - p.bytes.len().min(PAGE), 0);
    }

    // ---- type section ----
    let mut types = Vec::new();
    leb(&mut types, u64::from(t_fill) + 1);
    let mut sigs: Vec<(&[u8], &[u8])> = Vec::new();
    sigs.push((&[0x7e], &[0x7f])); // step
    for h in link.helpers {
        sigs.push((h.params, h.results));
    }
    sigs.push((&[0x7f, 0x7f, 0x7f], &[0x7f])); // region
    sigs.push((&[], &[0x7f])); // accessors
    sigs.push((&[0x7e, 0x7f, 0x7f], &[0x7f])); // fill
    for (params, results) in sigs {
        types.push(0x60);
        leb(&mut types, params.len() as u64);
        types.extend_from_slice(params);
        leb(&mut types, results.len() as u64);
        types.extend_from_slice(results);
    }

    // ---- imports: memory, <prefix>_jit_step, <prefix>_jit_fill ----
    let step_symbol = alloc::format!("{prefix}_jit_step");
    let fill_symbol = alloc::format!("{prefix}_jit_fill");
    let mut imports = Vec::new();
    leb(&mut imports, 3);
    name(&mut imports, "env");
    name(&mut imports, "__linear_memory");
    imports.push(0x02); // memory
    imports.push(0x00); // no maximum
    leb(&mut imports, 0);
    name(&mut imports, "env");
    name(&mut imports, &step_symbol);
    imports.push(0x00); // function
    leb(&mut imports, 0);
    name(&mut imports, "env");
    name(&mut imports, &fill_symbol);
    imports.push(0x00);
    leb(&mut imports, u64::from(t_fill));

    // ---- function section ----
    let mut funcs = Vec::new();
    leb(&mut funcs, u64::from(nhelpers + nregions) + 2);
    for i in 0..nhelpers {
        leb(&mut funcs, u64::from(1 + i));
    }
    for _ in 0..nregions {
        leb(&mut funcs, u64::from(t_region));
    }
    leb(&mut funcs, u64::from(t_accessor));
    leb(&mut funcs, u64::from(t_accessor));

    // ---- code section (payload built by hand to place relocations) ----
    let mut code = Vec::new();
    let mut code_relocs: Vec<(u8, usize, u32, Option<i32>)> = Vec::new();
    leb(&mut code, u64::from(nhelpers + nregions) + 2);
    let mut put_body = |code: &mut Vec<u8>, body: &[u8], calls: &[(usize, Callee)]| {
        leb(code, body.len() as u64);
        let start = code.len();
        code.extend_from_slice(body);
        for &(off, callee) in calls {
            let (ty, index) = match callee {
                Callee::Step => (R_WASM_FUNCTION_INDEX_LEB, s_step),
                Callee::Fill => (R_WASM_FUNCTION_INDEX_LEB, s_fill),
                Callee::Helper(n) => (R_WASM_FUNCTION_INDEX_LEB, s_helper0 + u32::from(n)),
                // A type index relocation names the type itself.
                Callee::EntryType => (R_WASM_TYPE_INDEX_LEB, t_region),
                Callee::Region(n) => (R_WASM_FUNCTION_INDEX_LEB, s_region0 + n),
            };
            code_relocs.push((ty, start + off, index, None));
        }
        body.len()
    };
    for h in link.helpers {
        put_body(&mut code, &h.body, &[]);
    }
    let callees = Callees {
        step: 0,
        fill: 1,
        helper: f_helper0,
        entry_type: t_region,
    };
    // Regions are linked with the other regions of their page (`Linking::direct_chain`):
    // a branch to one of their entries tail-calls it directly.
    let mut ri = 0u32;
    // Branch hints per region function: (region number, hints).
    let mut region_hints: Vec<(u32, Vec<(usize, bool)>)> = Vec::new();
    for p in &pages {
        let peers = page_peers(&p.regions, ri, link.direct_chain);
        for (i, r) in p.regions.iter().enumerate() {
            let body = back(r, callees, &peers[i]);
            if !body.hints.is_empty() {
                region_hints.push((ri, body.hints.clone()));
            }
            stats.code_bytes += put_body(&mut code, &body.body, &body.calls);
            stats.loaded += body.loaded as usize;
            stats.stored += body.stored as usize;
            ri += 1;
        }
    }
    // <prefix>_aot_directory: i32.const <address of the directory>
    {
        let mut body = Vec::new();
        leb(&mut body, 0); // no locals
        body.push(0x41); // i32.const
        let at = body.len();
        padded_leb(&mut body, 0); // a padded SLEB of 0 is the same bytes
        body.push(0x0b);
        leb(&mut code, body.len() as u64);
        let start = code.len();
        code.extend_from_slice(&body);
        code_relocs.push((R_WASM_MEMORY_ADDR_SLEB, start + at, s_data, Some(0)));
    }
    // <prefix>_aot_directory_len: i32.const <len>
    {
        let mut body = Vec::new();
        leb(&mut body, 0);
        body.push(0x41);
        sleb32(&mut body, dir.len() as i32);
        body.push(0x0b);
        leb(&mut code, body.len() as u64);
        code.extend_from_slice(&body);
    }

    // ---- data section: one segment, the directory ----
    let mut data = Vec::new();
    leb(&mut data, 1);
    data.push(0x00); // active, memory 0
    data.extend_from_slice(&[0x41, 0x00, 0x0b]); // i32.const 0; end
    leb(&mut data, dir.len() as u64);
    let dir_start = data.len();
    data.extend_from_slice(&dir);

    // ---- linking ----
    let mut linking = Vec::new();
    leb(&mut linking, 2); // version
    // WASM_SEGMENT_INFO
    {
        let mut sub = Vec::new();
        leb(&mut sub, 1);
        name(&mut sub, &alloc::format!(".rodata.{prefix}_aot_directory"));
        leb(&mut sub, 2); // alignment 2^2
        leb(&mut sub, 0);
        linking.push(5);
        leb(&mut linking, sub.len() as u64);
        linking.extend_from_slice(&sub);
    }
    // WASM_SYMBOL_TABLE
    {
        let mut sub = Vec::new();
        leb(&mut sub, u64::from(s_data) + 1);
        // step and fill: undefined functions, names from their imports
        for import in 0..2 {
            sub.push(SYM_FUNCTION);
            leb(&mut sub, u64::from(SYM_UNDEFINED));
            leb(&mut sub, import);
        }
        for (i, h) in link.helpers.iter().enumerate() {
            sub.push(SYM_FUNCTION);
            leb(&mut sub, u64::from(SYM_BINDING_LOCAL));
            leb(&mut sub, u64::from(f_helper0 + i as u32));
            name(&mut sub, &alloc::format!("{prefix}_aot_{}", h.name));
        }
        for i in 0..nregions {
            sub.push(SYM_FUNCTION);
            leb(&mut sub, u64::from(SYM_BINDING_LOCAL));
            leb(&mut sub, u64::from(f_region0 + i));
            name(&mut sub, &alloc::format!("{prefix}_aot_region_{i}"));
        }
        sub.push(SYM_FUNCTION);
        leb(&mut sub, 0);
        leb(&mut sub, u64::from(f_dir));
        name(&mut sub, &alloc::format!("{prefix}_aot_directory"));
        sub.push(SYM_FUNCTION);
        leb(&mut sub, 0);
        leb(&mut sub, u64::from(f_dir_len));
        name(&mut sub, &alloc::format!("{prefix}_aot_directory_len"));
        sub.push(SYM_DATA);
        leb(&mut sub, u64::from(SYM_BINDING_LOCAL));
        name(&mut sub, &alloc::format!("{prefix}_aot_directory_data"));
        leb(&mut sub, 0); // segment
        leb(&mut sub, 0); // offset
        leb(&mut sub, dir.len() as u64);
        linking.push(8);
        leb(&mut linking, sub.len() as u64);
        linking.extend_from_slice(&sub);
    }

    // ---- relocations ----
    // Section indices: 0 type, 1 import, 2 function, 3 code, 4 data, 5 linking.
    let mut reloc_code = Vec::new();
    leb(&mut reloc_code, 3);
    leb(&mut reloc_code, code_relocs.len() as u64);
    for (ty, off, sym, addend) in &code_relocs {
        reloc_code.push(*ty);
        leb(&mut reloc_code, *off as u64);
        leb(&mut reloc_code, u64::from(*sym));
        if let Some(a) = addend {
            sleb32(&mut reloc_code, *a);
        }
    }
    let mut reloc_data = Vec::new();
    leb(&mut reloc_data, 4);
    leb(&mut reloc_data, data_relocs.len() as u64);
    for (off, sym) in &data_relocs {
        reloc_data.push(R_WASM_TABLE_INDEX_I32);
        leb(&mut reloc_data, (dir_start + off) as u64);
        leb(&mut reloc_data, u64::from(*sym));
    }

    // ---- branch hints: function indices relocated by the linker ----
    // Section indices: 0 type, 1 import, 2 function, 3 code, 4 data, 5 the hints (when there
    // are any), then linking and the relocation sections.
    let mut hints = Vec::new();
    let mut reloc_hints = Vec::new();
    if !region_hints.is_empty() {
        // Relocation offsets of a custom section count from its content, after its name.
        leb(&mut hints, region_hints.len() as u64);
        let mut relocs = Vec::new();
        for (ri, h) in &region_hints {
            relocs.push((hints.len(), s_region0 + ri));
            padded_leb(&mut hints, f_region0 + ri);
            branch_hints(&mut hints, h);
        }
        leb(&mut reloc_hints, 5);
        leb(&mut reloc_hints, relocs.len() as u64);
        for (off, sym) in relocs {
            reloc_hints.push(R_WASM_FUNCTION_INDEX_LEB);
            leb(&mut reloc_hints, off as u64);
            leb(&mut reloc_hints, u64::from(sym));
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"\0asm");
    out.extend_from_slice(&1u32.to_le_bytes());
    section(&mut out, 1, &types);
    section(&mut out, 2, &imports);
    section(&mut out, 3, &funcs);
    section(&mut out, 10, &code);
    section(&mut out, 11, &data);
    if !hints.is_empty() {
        custom(&mut out, BRANCH_HINT_SECTION, &hints);
    }
    custom(&mut out, "linking", &linking);
    custom(&mut out, "reloc.CODE", &reloc_code);
    custom(&mut out, "reloc.DATA", &reloc_data);
    if !reloc_hints.is_empty() {
        custom(
            &mut out,
            &alloc::format!("reloc.{BRANCH_HINT_SECTION}"),
            &reloc_hints,
        );
    }
    (out, stats)
}

fn sleb32(out: &mut Vec<u8>, mut v: i32) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        let done = (v == 0 && b & 0x40 == 0) || (v == -1 && b & 0x40 != 0);
        if done {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory of one page whose record names entries `first..first + count`, with
    /// `entries` entry records (each function `7`, offset `4 * i`, block `i`).
    fn dir_bytes(page: &[u8], pages: u32, entries: u32, first: u32, count: u32) -> Vec<u8> {
        let mut d = Vec::new();
        for v in [DIRECTORY_MAGIC, 3, pages, entries] {
            d.extend_from_slice(&v.to_le_bytes());
        }
        d.extend_from_slice(&page_fast(page).to_le_bytes());
        d.extend_from_slice(&page_key(page));
        d.extend_from_slice(&first.to_le_bytes());
        d.extend_from_slice(&count.to_le_bytes());
        for i in 0..entries {
            d.extend_from_slice(&7u32.to_le_bytes());
            d.extend_from_slice(&(4 * i as u16).to_le_bytes());
            d.extend_from_slice(&(i as u16).to_le_bytes());
        }
        d.extend_from_slice(page);
        d
    }

    #[test]
    fn a_directory_finds_the_entries_of_a_page() {
        let page = [5u8; PAGE];
        let bytes = dir_bytes(&page, 1, 2, 0, 2);
        let dir = Directory::new(&bytes, 3).expect("well formed");
        let found: Vec<Entry> = dir.lookup_page(&page).collect();
        assert_eq!(
            found,
            [
                Entry {
                    func: 7,
                    offset: 0,
                    block: 0
                },
                Entry {
                    func: 7,
                    offset: 4,
                    block: 1
                },
            ]
        );
        assert_eq!(dir.lookup_page(&[6u8; PAGE]).count(), 0);
        assert!(Directory::new(&bytes, 4).is_none(), "another version");
    }

    #[test]
    fn a_corrupt_directory_is_refused_or_finds_nothing() {
        let page = [5u8; PAGE];
        // Counts whose sizes overflow (on any word size) or exceed the bytes.
        for (pages, entries) in [(u32::MAX, u32::MAX), (1, u32::MAX), (u32::MAX, 0), (2, 2)] {
            let mut bytes = dir_bytes(&page, 1, 2, 0, 2);
            bytes[8..12].copy_from_slice(&pages.to_le_bytes());
            bytes[12..16].copy_from_slice(&entries.to_le_bytes());
            assert!(
                Directory::new(&bytes, 3).is_none(),
                "{pages} pages, {entries} entries"
            );
        }
        assert!(Directory::new(&[0; HEADER - 1], 3).is_none());
        // A page record naming entries past the entry records: no entries, never a panic.
        for (first, count) in [
            (0, 3),
            (2, 1),
            (u32::MAX, 1),
            (1, u32::MAX),
            (u32::MAX, u32::MAX),
        ] {
            let bytes = dir_bytes(&page, 1, 2, first, count);
            let dir = Directory::new(&bytes, 3).expect("the header is consistent");
            assert_eq!(
                dir.lookup_page(&page).count(),
                0,
                "entries {first} + {count}"
            );
        }
    }
}
