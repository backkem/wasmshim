//! A toy guest ISA for the kit's own tests: variable-length instructions (2 or 4 bytes), so
//! nothing in the kit may assume A64's fixed 4-byte words.

use alloc::vec;
use alloc::vec::Vec;

use crate::aot::{Linking, PageTranslation, page_regions};
use crate::emit::{Body, Callee, Helper, push_padded_leb};
use crate::module::page_module;
use crate::nest::loop_regions;
use crate::region::{Block, Isa, Item, Region};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// A 2-byte operation.
    Short,
    /// A 4-byte operation.
    Wide,
    /// An unconditional branch to an absolute address (2 bytes).
    #[allow(dead_code)]
    Jump(u64),
    /// A conditional branch (2 bytes).
    Branch(u64),
    /// An indirect branch (2 bytes).
    Ret,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Toy;

impl Isa for Toy {
    type Inst = Op;
    const ALIGN: usize = 2;
    fn inst_len(inst: &Op) -> u64 {
        if *inst == Op::Wide { 4 } else { 2 }
    }
    /// Encoding: opcode byte (1 short, 2 wide, 3 jump, 4 branch, 5 ret; 0 is data), then an
    /// operand byte (a branch target in units of 2 bytes, absolute in the page).
    fn decode(bytes: &[u8], offset: usize) -> Option<Op> {
        let (op, arg) = (*bytes.get(offset)?, u64::from(*bytes.get(offset + 1)?));
        match op {
            1 => Some(Op::Short),
            2 if offset + 4 <= bytes.len() => Some(Op::Wide),
            3 => Some(Op::Jump(2 * arg)),
            4 => Some(Op::Branch(2 * arg)),
            5 => Some(Op::Ret),
            _ => None,
        }
    }
    fn ends_block(inst: &Op) -> bool {
        matches!(inst, Op::Jump(_) | Op::Branch(_) | Op::Ret)
    }
    fn successors(block: &Block<Self>) -> Vec<u64> {
        match block.insts.last() {
            Some(Op::Jump(t)) => vec![*t],
            Some(Op::Branch(t)) => vec![*t, block.end()],
            Some(Op::Ret) | None => vec![],
            _ => vec![block.end()],
        }
    }
}

/// Blocks decoded from `(pc, op)` pairs at the given starts.
fn blocks(code: &[(u64, Op)], starts: &[u64]) -> Vec<Block<Toy>> {
    starts
        .iter()
        .map(|&s| {
            let mut insts = Vec::new();
            for &(_, op) in code.iter().filter(|(pc, _)| *pc >= s) {
                insts.push(op);
                if Toy::ends_block(&op) {
                    break;
                }
            }
            Block { pc: s, insts }
        })
        .collect()
}

/// `0: op; 2: wide; 6: branch 2 (inner loop 2..8); 8: branch 0 (outer loop); 10: ret`.
fn code() -> Vec<(u64, Op)> {
    vec![
        (0, Op::Short),
        (2, Op::Wide),
        (6, Op::Branch(2)),
        (8, Op::Branch(0)),
        (10, Op::Ret),
    ]
}

#[test]
fn loop_nests_are_found_for_any_isa() {
    let b = blocks(&code(), &[0, 2, 8, 10]);
    assert_eq!(b[1].end(), 8, "a 4-byte and a 2-byte instruction");
    let r = loop_regions(&b);
    assert_eq!(r.len(), 1);
    let pcs: Vec<u64> = r[0].blocks.iter().map(|b| b.pc).collect();
    assert_eq!(pcs, [0, 2, 8]);
    assert_eq!(
        r[0].nest,
        Some(vec![Item::Loop(vec![
            Item::Block(0),
            Item::Block(1),
            Item::Block(2)
        ])])
    );
    // Grouping: the loop region (entered only at 0), and a page region with every other block.
    let regions = page_regions(b, true);
    assert_eq!(regions.len(), 2);
    assert!(regions[0].nest.is_some());
    assert_eq!(
        regions[1].blocks.len(),
        3,
        "non-header loop blocks stay entries of a page region"
    );
    assert_eq!(regions[1].blocks[2].pc, 10);
}

/// A back half that returns exit code 1 at once, with a direct chain to the next region.
fn back(
    r: &Region<Toy>,
    _c: crate::emit::Callees,
    peers: &alloc::collections::BTreeMap<u64, (u32, u32)>,
) -> Body {
    let mut body = Vec::new();
    body.push(0); // no locals
    let mut calls = Vec::new();
    if let Some((_, &(n, _))) = peers.iter().next() {
        // if (ctx == -1) return_call region n
        body.extend_from_slice(&[
            0x20, 2, 0x41, 0x7f, 0x46, 0x04, 0x40, 0x20, 0, 0x20, 1, 0x20, 2, 0x12,
        ]);
        calls.push((body.len(), Callee::Region(n)));
        push_padded_leb(&mut body, 0);
        body.push(0x0b);
    }
    body.extend_from_slice(&[0x10]); // call helper 0
    calls.push((body.len(), Callee::Helper(0)));
    push_padded_leb(&mut body, 0);
    body.extend_from_slice(&[0x0b]); // end
    let _ = r;
    Body {
        body,
        calls,
        ..Body::default()
    }
}

#[test]
fn page_modules_are_valid_and_resolve_their_calls() {
    let b = blocks(&code(), &[0, 2, 8, 10]);
    let page = PageTranslation::new(&[0u8; 4096], page_regions(b, true));
    let helpers = [Helper {
        name: "one",
        params: &[],
        results: &[0x7f],
        body: vec![0, 0x41, 1, 0x0b],
    }];
    let link = Linking {
        version: 1,
        prefix: "toy",
        helpers: &helpers,
        direct_chain: true,
    };
    let m = page_module(&page, &link, &mut back);
    assert_eq!(m.funcs, 2);
    assert_eq!(m.entries, [(0, 0, 0), (2, 1, 0), (8, 1, 1), (10, 1, 2)]);
    wasmparser::Validator::new()
        .validate_all(&m.bytes)
        .expect("a valid module");
    // The engine's functions under the given prefix, from module `engine`.
    let has = |s: &[u8]| m.bytes.windows(s.len()).any(|w| w == s);
    assert!(has(b"toy_jit_step") && has(b"toy_jit_fill") && has(b"engine"));
}

#[test]
fn a_static_sweep_finds_block_entries_without_running_the_code() {
    // 0: short; 2: wide; 6: branch -> 2; 8: short; 10: ret; 12: data (0, 0); 14: short; 16: ret.
    let mut page = vec![0u8; 4096];
    let code: [(usize, [u8; 2]); 7] = [
        (0, [1, 0]),
        (2, [2, 0]),
        (6, [4, 1]),
        (8, [1, 0]),
        (10, [5, 0]),
        (14, [1, 0]),
        (16, [5, 0]),
    ];
    for (at, b) in code {
        page[at..at + 2].copy_from_slice(&b);
    }
    let e = crate::aot::static_entries::<Toy>(&page);
    // The page start, the branch target (2), after the branch (8), after the ret (12), the
    // undecodable data moves on by the alignment (14), after the last ret (18), and so on
    // through the zero-filled rest of the page.
    assert_eq!(&e[..6], [0, 2, 8, 12, 14, 18]);
    assert!(e.iter().all(|&o| (o as usize) < 4096 && o % 2 == 0));
}
