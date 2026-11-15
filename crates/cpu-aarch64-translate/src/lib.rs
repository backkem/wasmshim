//! The AArch64 front half of the translator (over `wasmshim-translate`): the hot blocks of an A64
//! code page are grouped into *regions*, and each region becomes one Wasm function.
//!
//! The generated code runs inside an *engine* instance (`wasmshim-aarch64-engine`) and shares its
//! linear memory: it reads and writes the [`wasmshim_aarch64::Cpu`] register file in place (see
//! [`wasmshim_aarch64::layout`]) and reaches guest pages through a software TLB the engine
//! maintains.
//!
//! **What runs inline.** Integer arithmetic and logic, branches, loads and stores (a TLB check
//! and a host access), single-register exclusives, the system registers this translator knows,
//! scalar floating point (`a64/fp.rs`), Advanced SIMD lane arithmetic (`a64/neon.rs`), fused
//! multiply-add emulated exactly (`a64/fma.rs`), and common SIMD integer and move instructions
//! (`a64/simd.rs`). Every inline case mirrors the `wasmshim-aarch64` interpreter bit for bit; where
//! Wasm leaves a result open the code checks it and falls back: a NaN float result, a lane that
//! saturated, or an FMA outside its exact range hands that instruction to the interpreter
//! (`jit_step`), which applies Arm's rules (and sets `FPSR.QC`). Still executed by the
//! interpreter one instruction at a time: half precision, LSE atomics and compare-and-swap, pair
//! exclusives, the other system instructions, and every encoding not listed in those modules.
//! A memory access the fast path cannot do (a TLB miss, a page-crossing or misaligned access, a
//! store into code) leaves through [`abi::EXIT_MEM`] inside loops and calls `jit_fill` and
//! `jit_step` elsewhere. The interpreter stays the reference implementation; the differential
//! tests (`tests/differential.rs`) compare the two per instruction class.
//!
//! **Retired-instruction exactness.** Each block first checks that the remaining budget covers
//! the whole block, otherwise it exits ([`abi::EXIT_BUDGET`]) and the engine finishes the budget
//! in the interpreter. A block that leaves early (a syscall, a memory side exit, a `jit_step`
//! status) gives back the budget of the instructions it did not retire; unrolled loops check the
//! budget once for all their copies and keep the checked loop as the exact path. So translated
//! code retires exactly what the interpreter would, and stops at exactly the same points. (The
//! scheduler turns retired instructions into virtual time at its configured rate.)
//!
//! **What is A64 and what is the kit's**: the IR and its passes, regions and
//! loop nests, the translated-code ABI, page keys, directories, objects, standalone modules and
//! the run-time tier are `wasmshim-translate`'s, shared by every guest ISA. Here: decoding pages into
//! blocks ([`A64`]), the A64 front half (`a64`: A64 to IR), and the back half's A64 code shape
//! (`emit`: the register file, lazy NZCV and `pack`, block-scoped registers).
//!
//! The translator knows nothing about Linux or any host: it maps instructions to Wasm. The design
//! note is `docs/design/engines-and-translation.md` in this repository; instruction coverage is
//! in `docs/design/cpu.md`.

#![no_std]
#![deny(unsafe_code)]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use wasmshim_aarch64::Inst;

mod a64;
pub mod aot;
mod emit;

/// The contract between translated modules and the engine that runs them (the kit's).
pub use wasmshim_translate::abi;
/// The ISA-neutral IR and its passes (the kit's).
pub use wasmshim_translate::{ir, opt};

pub use wasmshim_translate::Item;
pub use wasmshim_translate::emit::{Body, Callee, Callees};

/// The AArch64 ISA as the kit sees it: fixed 4-byte instructions, blocks ending at branches and
/// `SVC`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct A64;

impl wasmshim_translate::Isa for A64 {
    type Inst = Inst;

    const ALIGN: usize = 4;

    fn inst_len(_inst: &Inst) -> u64 {
        4
    }

    fn decode(bytes: &[u8], offset: usize) -> Option<Inst> {
        let w = bytes.get(offset..offset + 4)?;
        Some(wasmshim_aarch64::decode(u32::from_le_bytes([
            w[0], w[1], w[2], w[3],
        ])))
    }

    fn block_end(pc: u64, insts: &[Inst]) -> u64 {
        pc + 4 * insts.len() as u64
    }

    fn ends_block(inst: &Inst) -> bool {
        ends_block(inst)
    }

    fn successors(block: &Block) -> Vec<u64> {
        successors(block)
    }
}

/// A straight-line run of A64 instructions (the kit's [`wasmshim_translate::Block`]).
pub type Block = wasmshim_translate::Block<A64>;
/// Blocks translated together into one function (the kit's [`wasmshim_translate::Region`]).
pub type Region = wasmshim_translate::Region<A64>;

/// Bumped whenever generated code or the directory format changes meaning: translations made
/// by another version are ignored at run time.
pub const TRANSLATOR_VERSION: u32 = 13;

/// How A64 translations are linked: the engine's symbols are `linux_*` (the
/// `wasmshim-aarch64-engine` defines them), one helper (`pack`).
pub fn linking(
    helpers: &[wasmshim_translate::emit::Helper],
    direct_chain: bool,
) -> wasmshim_translate::aot::Linking<'_> {
    wasmshim_translate::aot::Linking {
        version: TRANSLATOR_VERSION,
        prefix: "linux",
        helpers,
        direct_chain,
    }
}

/// The A64 front half's helper functions, in index order: `pack(kind, x, y, r) -> nzcv`.
pub fn helpers() -> [wasmshim_translate::emit::Helper; 1] {
    [wasmshim_translate::emit::Helper {
        name: "pack",
        params: &[0x7f, 0x7e, 0x7e, 0x7e],
        results: &[0x7f],
        body: emit::pack_body(),
    }]
}

/// Translates `region` into a complete Wasm module (imports per [`abi`], one exported
/// function [`abi::EXPORT_ENTRY`]). Deterministic: the same region gives the same bytes.
pub fn translate(region: &Region) -> Vec<u8> {
    translate_with(region, Options::default())
}

/// [`translate`] with code-shape [`Options`] (experiments; the defaults ship).
pub fn translate_with(region: &Region, opts: Options) -> Vec<u8> {
    let helpers = helpers();
    let link = linking(&helpers, false);
    wasmshim_translate::module::region_module(region, &link, &mut |r, callees, peers| {
        emit::region_body_linked(r, r.page(), callees, opts, peers)
    })
}

pub use emit::Options;

/// The body of a region function (see `emit`'s module docs) calling `callees`, for
/// embedding in a module or an object file.
pub fn region_body(region: &Region, callees: Callees) -> Body {
    emit::region_body(region, region.page(), callees)
}

/// [`region_body`] with code-shape [`Options`].
pub fn region_body_with(region: &Region, callees: Callees, opts: Options) -> Body {
    emit::region_body_with(region, region.page(), callees, opts)
}

/// [`region_body_with`] linked with the other regions of its page (direct chaining).
pub fn region_body_linked(
    region: &Region,
    callees: Callees,
    opts: Options,
    peers: &BTreeMap<u64, (u32, u32)>,
) -> Body {
    emit::region_body_linked(region, region.page(), callees, opts, peers)
}

/// The body of the internal `pack(kind: i32, x: i64, y: i64, r: i64) -> i32` function every
/// region calls (one copy per module).
pub fn pack_body() -> Vec<u8> {
    emit::pack_body()
}

/// Diagnostics: the instructions of a profiled page's hot blocks that
/// the A64 front half leaves to the reference interpreter (an `Op::Step` of their own, not a
/// memory access's slow path), once per occurrence in a block.
pub fn stepped_instructions(p: &aot::PageProfile) -> Vec<Inst> {
    let t = aot::translate_page(p);
    let mut out = Vec::new();
    for r in &t.regions {
        for b in &r.blocks {
            let ir = a64::lower(b, 0, &Options::default());
            for (k, ops) in ir.ops.iter().enumerate() {
                if ops.iter().any(|o| matches!(o, ir::Op::Step { .. })) {
                    out.push(b.insts[k]);
                }
            }
        }
    }
    out
}

/// The run-time tier's front half: a page as a module, in exactly the
/// shape an ahead-of-time build of the same page and entries links.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrontHalf {
    pub opts: Options,
}

impl wasmshim_translate::tier::FrontHalf for FrontHalf {
    fn page_module(&self, bytes: &[u8], entries: &[u16]) -> wasmshim_translate::module::PageModule {
        let page = aot::translate_page_with(
            &aot::PageProfile {
                bytes: bytes.to_vec(),
                entries: entries.to_vec(),
            },
            self.opts,
        );
        let helpers = helpers();
        let link = linking(&helpers, self.opts.direct_chain);
        let opts = self.opts;
        wasmshim_translate::module::page_module(&page, &link, &mut |r, callees, peers| {
            emit::region_body_linked(r, r.page(), callees, opts, peers)
        })
    }
}

/// `true` if `inst` ends a block when regions are formed: branches, and `SVC` (the engine
/// handles the syscall).
pub fn ends_block(inst: &Inst) -> bool {
    inst.is_control_flow() || matches!(inst, Inst::Svc { .. })
}

/// Static successors of a block's last instruction that a region may contain: direct branch
/// targets and the fall-through address. Indirect branches have none.
pub fn successors(block: &Block) -> Vec<u64> {
    let mut out = Vec::new();
    let Some(last) = block.insts.last() else {
        return out;
    };
    let pc = block.end() - 4;
    match *last {
        Inst::Branch { offset, .. } => out.push(pc.wrapping_add(offset as u64)),
        Inst::BranchCond { offset, .. }
        | Inst::CompareBranch { offset, .. }
        | Inst::TestBranch { offset, .. } => {
            out.push(pc.wrapping_add(offset as u64));
            out.push(block.end());
        }
        Inst::BranchReg { .. } | Inst::Svc { .. } => {}
        _ => out.push(block.end()),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aot::{PageProfile, object, page_key, translate_page};

    /// A page of real A64 code: a loop with a load, a compare, flags across blocks, a call,
    /// SIMD and an `SVC`.
    fn page() -> alloc::vec::Vec<u8> {
        let words: [u32; 12] = [
            0xf940_0020, // ldr x0, [x1]
            0xf100_041f, // cmp x0, #1
            0x5400_0061, // b.ne +12
            0x9100_0400, // add x0, x0, #1
            0x1400_0002, // b +8
            0xd100_0400, // sub x0, x0, #1
            0x6e20_8c00, // cmeq v0.16b, v0.16b, v0.16b
            0x9e66_0001, // fmov x1, d0
            0xdac0_0c21, // rev x1, x1
            0x9400_0010, // bl +64
            0xd400_0001, // svc #0
            0xd65f_03c0, // ret
        ];
        let mut bytes = alloc::vec![0u8; aot::PAGE];
        for (i, w) in words.iter().enumerate() {
            bytes[0x100 + 4 * i..0x104 + 4 * i].copy_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    fn validate(wasm: &[u8]) {
        wasmparser::Validator::new()
            .validate_all(wasm)
            .expect("valid Wasm");
    }

    #[test]
    fn long_straight_line_code_continues_in_more_blocks() {
        // 700 `add x0, x0, #1` from 0x40, then `ret`: one hot entry. Blocks stop at 256
        // instructions; the rest are blocks too, not left to the interpreter.
        let mut bytes = alloc::vec![0u8; aot::PAGE];
        for i in 0..700 {
            bytes[0x40 + 4 * i..0x44 + 4 * i].copy_from_slice(&0x9100_0400u32.to_le_bytes());
        }
        bytes[0x40 + 4 * 700..0x44 + 4 * 700].copy_from_slice(&0xd65f_03c0u32.to_le_bytes());
        let t = translate_page(&PageProfile {
            bytes,
            entries: alloc::vec![0x40],
        });
        let mut starts: alloc::vec::Vec<u64> = t
            .regions
            .iter()
            .flat_map(|r| r.blocks.iter().map(|b| b.pc))
            .collect();
        starts.sort_unstable();
        assert_eq!(starts, [0x40, 0x40 + 1024, 0x40 + 2048]);
        let insns: usize = t
            .regions
            .iter()
            .flat_map(|r| r.blocks.iter().map(|b| b.insts.len()))
            .sum();
        assert_eq!(insns, 701);
    }

    #[test]
    fn regions_translate_to_valid_modules() {
        let p = page();
        let t = translate_page(&PageProfile {
            bytes: p,
            entries: alloc::vec![0x100, 0x10c, 0x114, 0x128],
        });
        assert!(!t.regions.is_empty());
        for r in &t.regions {
            validate(&translate(r));
        }
    }

    #[test]
    fn the_directory_finds_translated_pages_by_content() {
        let p = page();
        let t = translate_page(&PageProfile {
            bytes: p.clone(),
            entries: alloc::vec![0x100, 0x114],
        });
        let (obj, stats) = object(&[t]);
        assert_eq!(stats.pages, 1);
        assert_eq!(stats.blocks, 2);
        assert!(stats.code_bytes > 0);
        // An object file is not a module, but its directory is plain data: find it in the
        // data section by its magic and parse it.
        let magic = aot::DIRECTORY_MAGIC.to_le_bytes();
        let at = obj
            .windows(4)
            .position(|w| w == magic)
            .expect("directory in the object");
        let dir = crate::aot::directory(&obj[at..]).expect("directory");
        assert_eq!(dir.pages(), 1);
        let entries: alloc::vec::Vec<_> = dir.lookup_page(&p).collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].offset, 0x100);
        assert_eq!(entries[1].offset, 0x114);
        assert_eq!(dir.lookup_page(&[0u8; 4096]).count(), 0);
        // A page with the same fast key but other content is not accepted: the whole-page compare decides.
        let mut forged = translate_page(&PageProfile {
            bytes: p.clone(),
            entries: alloc::vec![0x100],
        });
        let mut other = p.clone();
        other[0] ^= 1;
        forged.fast = aot::page_fast(&other);
        let (obj, _) = object(&[forged]);
        let at = obj.windows(4).position(|w| w == magic).expect("directory");
        let dir = crate::aot::directory(&obj[at..]).expect("directory");
        assert_eq!(
            dir.lookup_page(&other).count(),
            0,
            "fast-key collision rejected"
        );
        assert_ne!(page_key(&other), page_key(&p));
        // An empty object (no translations) still parses as an object.
        let (empty, s) = object(&[]);
        assert_eq!(s.pages, 0);
        assert!(empty.starts_with(b"\0asm"));
    }
}
#[cfg(test)]
mod nest_tests {
    use super::*;
    use alloc::vec;
    use wasmshim_translate::nest::loop_regions;

    fn blocks(words: &[u32], starts: &[usize]) -> Vec<Block> {
        starts
            .iter()
            .map(|&s| {
                let mut insts = Vec::new();
                for &w in &words[s..] {
                    let i = wasmshim_aarch64::decode(w);
                    insts.push(i);
                    if ends_block(&i) {
                        break;
                    }
                }
                Block {
                    pc: 4 * s as u64,
                    insts,
                }
            })
            .collect()
    }

    /// `arith`'s Collatz loops: an outer loop with an inner self-loop.
    #[test]
    fn collatz_is_one_loop_region_with_the_inner_loop_inside() {
        let words = [
            0xcb0b_010c, // 0x00 sub x12, x8, x11
            0x9100_0529, // 0x04 add x9, x9, #1
            0x8b0b_158b, // 0x08 add x11, x12, x11, lsl #5
            0xeb0a_013f, // 0x0c cmp x9, x10
            0x5400_0180, // 0x10 b.eq 0x40
            0xf100_053f, // 0x14 cmp x9, #1
            0xaa09_03ec, // 0x18 mov x12, x9
            0x54ff_ff20, // 0x1c b.eq 0x00
            0xd341_fd8d, // 0x20 lsr x13, x12, #1
            0x8b0c_058e, // 0x24 add x14, x12, x12, lsl #1
            0xf240_019f, // 0x28 tst x12, #1
            0x9100_0508, // 0x2c add x8, x8, #1
            0x9a8e_15ac, // 0x30 csinc x12, x13, x14, ne
            0xf100_059f, // 0x34 cmp x12, #1
            0x54ff_ff41, // 0x38 b.ne 0x20
            0x17ff_fff1, // 0x3c b 0x00
            0xd65f_03c0, // 0x40 ret
        ];
        let b = blocks(&words, &[0, 5, 8, 15, 16]);
        let regions = loop_regions(&b);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        let pcs: Vec<u64> = r.blocks.iter().map(|b| b.pc).collect();
        assert_eq!(pcs, [0x00, 0x14, 0x20, 0x3c]);
        assert_eq!(
            r.nest,
            Some(vec![Item::Loop(vec![
                Item::Block(0),
                Item::Block(1),
                Item::Block(2),
                Item::Block(3)
            ])])
        );
    }

    /// A loop entered at two blocks is not a loop region; a two-block inner loop nests.
    #[test]
    fn inner_loops_nest_and_two_entries_do_not() {
        // 0: b.eq 8 (exit-ish) ; 1: (inner header) add ; b 2 ; 2: subs ; b.ne 1 ; 3: b 0
        let words = [
            0x5400_0100, // 0x00 b.eq 0x20
            0x9100_0400, // 0x04 add x0, x0, #1
            0x1400_0001, // 0x08 b 0x0c
            0xf100_0421, // 0x0c subs x1, x1, #1
            0x54ff_ffa1, // 0x10 b.ne 0x04
            0x17ff_fffb, // 0x14 b 0x00
            0xd503_201f, // 0x18 nop
            0xd503_201f, // 0x1c nop
            0xd65f_03c0, // 0x20 ret
        ];
        let b = blocks(&words, &[0, 1, 3, 5, 8]);
        let regions = loop_regions(&b);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        let pcs: Vec<u64> = r.blocks.iter().map(|b| b.pc).collect();
        assert_eq!(pcs, [0x00, 0x04, 0x0c, 0x14]);
        assert_eq!(
            r.nest,
            Some(vec![Item::Loop(vec![
                Item::Block(0),
                Item::Loop(vec![Item::Block(1), Item::Block(2)]),
                Item::Block(3),
            ])])
        );
        // Hot blocks branching into the loop at two places (0x0c and 0x00): two entries, so no
        // loop region. (With only the branch to 0x0c, the loop would be entered there.)
        let mut words2 = words.to_vec();
        words2[6] = 0x17ff_fffd; // 0x18 b 0x0c
        words2[7] = 0x17ff_fff9; // 0x1c b 0x00
        let b = blocks(&words2, &[0, 1, 3, 5, 6, 7, 8]);
        assert!(loop_regions(&b).is_empty());
        let b = blocks(&words2, &[0, 1, 3, 5, 6, 8]);
        let r = loop_regions(&b);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].blocks[0].pc, 0x0c, "entered at 0x0c");
    }
}
