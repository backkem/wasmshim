//! Code patching on the engine's address space: a
//! permission-ignoring write to a code page invalidates exactly the translations of that page
//! (they run interpreted until something translates the new content), drops the chain entries
//! into that page (translated code cannot chain into a stale entry), and leaves every other page's
//! translations in place. Natively there is no translated code to call, so this checks the
//! bookkeeping the dispatcher relies on: `lookup` is the only way into translated code.

use wasmshim_aarch64::asm::assemble_bytes;
use wasmshim_aarch64_engine::mem::{ChainState, EngineMem};
use wasmshim_cpu::{GuestMemory, PAGE_SIZE, PagedMemory, Perms};

const CODE: u64 = 0x1_0000;
const OTHER: u64 = CODE + PAGE_SIZE;

/// Installs a "translation" of the block at `pc` in table slot `slot` (as the run-time tier
/// does).
fn install(m: &mut EngineMem, pc: u64, slot: u32) {
    let ((_, page, version), _, _) = m.hot_page(pc).expect("a readable page");
    m.install_page(page, version, &[((pc & 0xfff) as u16, slot, 0)]);
}

fn engine_mem() -> EngineMem {
    let mut m = EngineMem::new(PagedMemory::new());
    m.map(CODE, 2 * PAGE_SIZE, Perms::RX).unwrap();
    m.poke(CODE, &assemble_bytes("mov x0, #1; ret", CODE).unwrap())
        .unwrap();
    m.poke(OTHER, &assemble_bytes("mov x0, #7; ret", OTHER).unwrap())
        .unwrap();
    // One "translated" region per page (slots 5 and 6 stand for compiled functions).
    for (pc, slot) in [(CODE, 5), (OTHER, 6)] {
        install(&mut m, pc, slot);
    }
    m
}

#[test]
fn a_patch_kills_the_translations_of_its_page_only() {
    let mut m = engine_mem();
    assert_eq!(m.lookup(CODE), Some((5, 0)));
    assert_eq!(m.lookup(OTHER), Some((6, 0)));

    m.poke(CODE, &assemble_bytes("mov x0, #2", CODE).unwrap())
        .unwrap();
    assert_eq!(m.lookup(CODE), None, "the patched page runs interpreted");
    // A listed code change drops the chain entries into its page (the epoch moves only for
    // changes that are not listed per page).
    assert_ne!(
        m.chain_state(CODE),
        ChainState::Current,
        "chained entries into the patched page are dead"
    );
    assert_eq!(m.lookup(OTHER), Some((6, 0)), "other pages keep theirs");
    assert_eq!(m.stats.killed, 1);
    // The patched bytes are what the interpreter fetches.
    assert_eq!(
        m.fetch(CODE).unwrap().to_le_bytes()[..],
        assemble_bytes("mov x0, #2", CODE).unwrap()[..]
    );

    // A region translated from the new content is valid again.
    install(&mut m, CODE, 9);
    assert_eq!(m.lookup(CODE), Some((9, 0)));
}

#[test]
fn rewriting_identical_bytes_still_invalidates() {
    // The engine does not compare contents: any write to code is a change (the AOT directory
    // is content-keyed, so identical content finds its translation again on the next miss).
    let mut m = engine_mem();
    let same = assemble_bytes("mov x0, #1", CODE).unwrap();
    m.poke(CODE, &same).unwrap();
    assert_eq!(m.lookup(CODE), None);
    assert_eq!(m.lookup(OTHER), Some((6, 0)));
}

#[test]
fn a_write_to_data_changes_no_translation() {
    // Speed stays invisible where nothing is patched: a write elsewhere moves no code epoch.
    let mut m = engine_mem();
    m.map(0x4_0000, PAGE_SIZE, Perms::RW).unwrap();
    let (_, epoch) = {
        assert_eq!(m.lookup(CODE), Some((5, 0)));
        m.chain()
    };
    m.poke(0x4_0000, &[1, 2, 3, 4]).unwrap();
    assert_eq!(m.lookup(CODE), Some((5, 0)));
    assert_eq!(m.chain().1, epoch);
}

#[test]
fn a_new_address_space_never_shares_a_page_identity_with_a_freed_one() {
    // The run-time tier keys its per-page state by (address space, page, code version). The
    // address-space part is the `EngineMem`'s address, which a later allocation may reuse; the
    // version is never reused by another address space, so the key stays unique anyway.
    let mut seen = Vec::new();
    for _ in 0..8 {
        let m = engine_mem();
        let (id, bytes, _) = m.hot_page(CODE).expect("a readable page");
        assert!(!seen.iter().any(|&(_, _, v)| v == id.2), "{id:?} reused");
        seen.push(id);
        assert_eq!(bytes.len() as u64, PAGE_SIZE);
        drop(m);
    }
}

#[test]
fn a_branch_to_the_last_address_finds_no_translation() {
    // Empty fast-lookup slots are keyed by pc `u64::MAX`, which a guest can branch to: the
    // lookup must answer "interpret" (where the fetch faults), not an empty slot's contents.
    let mut fresh = EngineMem::new(PagedMemory::new());
    assert_eq!(fresh.lookup(u64::MAX), None);
    let mut m = engine_mem();
    assert_eq!(m.lookup(u64::MAX), None);
    // Also once a killed translation's slot has been emptied.
    assert_eq!(m.lookup(CODE), Some((5, 0)));
    m.poke(CODE, &assemble_bytes("mov x0, #3", CODE).unwrap())
        .unwrap();
    assert_eq!(m.lookup(CODE), None);
    assert_eq!(m.lookup(u64::MAX), None);
}
