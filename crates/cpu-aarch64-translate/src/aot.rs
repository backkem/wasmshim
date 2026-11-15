//! Ahead-of-time translation of A64 pages over the kit's content-keyed pages,
//! directories and objects (`wasmshim_translate::aot`): this module decodes a
//! profiled page into blocks; grouping them into regions, the directory and the object are the
//! kit's.
//!
//! * **The object** defines `linux_aot_directory() -> *const u8` and
//!   `linux_aot_directory_len() -> usize` (the [`Directory`]), one internal `pack` function
//!   and one function per region; translated code calls the undefined symbols
//!   `linux_jit_step(pc: i64) -> i32` and `linux_jit_fill`, which the engine defines.
//! * An object with no pages is valid: engines link one when there is nothing to translate.

use alloc::vec::Vec;

use crate::{Block, ends_block};

pub use wasmshim_translate::aot::{
    BRANCH_HINT_SECTION, DIRECTORY_MAGIC, Entry, ObjectStats, PAGE, PageKey, PageProfile,
    page_fast, page_key,
};

pub use crate::TRANSLATOR_VERSION;

/// The regions of one A64 page, ready to emit.
pub type PageTranslation = wasmshim_translate::aot::PageTranslation<crate::A64>;
/// A read-only view of a directory (as linked into the engine).
pub type Directory<'a> = wasmshim_translate::aot::Directory<'a>;

/// The symbol `linux_jit_step` resolves to.
pub const STEP_SYMBOL: &str = "linux_jit_step";
/// The symbol `linux_jit_fill` resolves to.
pub const FILL_SYMBOL: &str = "linux_jit_fill";
/// The directory accessor symbols.
pub const DIRECTORY_SYMBOL: &str = "linux_aot_directory";
pub const DIRECTORY_LEN_SYMBOL: &str = "linux_aot_directory_len";

/// Longest block.
const MAX_BLOCK: usize = 256;

/// Parses a directory of this translator version; `None` if it is malformed or from another
/// version.
pub fn directory(bytes: &[u8]) -> Option<Directory<'_>> {
    Directory::new(bytes, TRANSLATOR_VERSION)
}

/// Forms the regions of a profiled page: every hot entry starts a block (decoded up to a
/// branch, an `SVC`, the page end or `MAX_BLOCK` instructions). With
/// [`crate::Options::loop_regions`] (the default), every single-entry loop nest among the
/// blocks becomes a loop region of its own, entered only at its header; the
/// rest are grouped into page regions (`wasmshim_translate::aot::page_regions`).
pub fn translate_page(p: &PageProfile) -> PageTranslation {
    translate_page_with(p, crate::Options::default())
}

/// [`translate_page`] with code-shape options.
pub fn translate_page_with(p: &PageProfile, opts: crate::Options) -> PageTranslation {
    let mut entries = p.entries.clone();
    entries.sort_unstable();
    entries.dedup();
    let word = |off: usize| -> Option<u32> {
        p.bytes
            .get(off..off + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let mut blocks = Vec::new();
    // A block cut at MAX_BLOCK instructions (no branch yet) goes on in another block: its
    // continuation is an entry too (the interpreter's blocks are not
    // cut, so a profile never names it; long straight-line code such as OpenSSL's unrolled
    // sha512 rounds stayed interpreted past its first 256 instructions).
    let mut todo: alloc::collections::BTreeSet<usize> =
        entries.iter().map(|&e| usize::from(e) & !3).collect();
    let mut done = alloc::collections::BTreeSet::new();
    while let Some(start) = todo.pop_first() {
        if !done.insert(start) {
            continue;
        }
        let mut off = start;
        let mut insts = Vec::new();
        let mut ended = false;
        while off + 4 <= PAGE && insts.len() < MAX_BLOCK {
            let Some(w) = word(off) else { break };
            let inst = wasmshim_aarch64::decode(w);
            insts.push(inst);
            off += 4;
            if ends_block(&inst) {
                ended = true;
                break;
            }
        }
        if insts.is_empty() {
            continue;
        }
        if !ended && insts.len() == MAX_BLOCK && off + 4 <= PAGE {
            todo.insert(off);
        }
        blocks.push(Block {
            pc: start as u64,
            insts,
        });
    }
    blocks.sort_by_key(|b| b.pc);
    blocks.dedup_by_key(|b| b.pc);
    let regions = wasmshim_translate::aot::page_regions_capped(
        blocks,
        opts.loop_regions && opts.self_loops,
        opts.region_blocks,
        opts.region_insns,
    );
    PageTranslation::new(&p.bytes, regions)
}

/// Emits a relocatable Wasm object with the given page translations (see the module docs).
pub fn object(pages: &[PageTranslation]) -> (Vec<u8>, ObjectStats) {
    object_with(pages, crate::Options::default())
}

/// [`object`] with code-shape [`crate::Options`] (for experiments).
pub fn object_with(pages: &[PageTranslation], opts: crate::Options) -> (Vec<u8>, ObjectStats) {
    let helpers = crate::helpers();
    let link = crate::linking(&helpers, opts.direct_chain);
    wasmshim_translate::aot::object(pages, &link, &mut |r, callees, peers| {
        crate::emit::region_body_linked(r, r.page(), callees, opts, peers)
    })
}
