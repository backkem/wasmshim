//! Blocks and regions: what a front half decodes and a back half turns into one Wasm function.
//!
//! A region never crosses a [`PAGE`]-sized unit of guest code: its translation is a function of
//! that unit's bytes (and of which blocks in it are entries), so translations are keyed by the
//! page's content, not by its address ([`crate::aot`]).

use alloc::vec::Vec;
use core::fmt::Debug;

/// The unit of guest code translations are keyed by (bytes).
pub const PAGE: u64 = 4096;

/// What the ISA-neutral parts of the kit need to know about a guest ISA: its decoded
/// instruction type and its control flow. A front half implements it on a marker type.
pub trait Isa: Clone + Copy + Debug + Default + PartialEq + Eq + 'static {
    /// A decoded guest instruction.
    type Inst: Clone + Debug;

    /// The size of `inst` in bytes.
    fn inst_len(inst: &Self::Inst) -> u64;

    /// Decodes the instruction at `offset` of a page's `bytes` (`None` past the end, or if
    /// nothing decodes there: the static entry sweep then moves on by the ISA's smallest
    /// instruction).
    fn decode(bytes: &[u8], offset: usize) -> Option<Self::Inst>;

    /// The smallest instruction size and alignment, in bytes.
    const ALIGN: usize;

    /// The address after a block of `insts` starting at `pc` (an ISA with fixed-size
    /// instructions may compute it directly).
    fn block_end(pc: u64, insts: &[Self::Inst]) -> u64 {
        pc + insts.iter().map(Self::inst_len).sum::<u64>()
    }

    /// `true` if `inst` ends a block when regions are formed (branches, and whatever the
    /// engine must see: a system call, say).
    fn ends_block(inst: &Self::Inst) -> bool;

    /// The static successors of `block`'s last instruction that a region may contain: direct
    /// branch targets and the fall-through address. Indirect branches have none.
    fn successors(block: &Block<Self>) -> Vec<u64>;
}

/// A straight-line run of instructions: every instruction but the last falls through. If the
/// last one is not a branch, the block falls through to [`Block::end`].
#[derive(Clone, Debug)]
pub struct Block<A: Isa> {
    /// Address of the first instruction.
    pub pc: u64,
    /// The decoded instructions.
    pub insts: Vec<A::Inst>,
}

impl<A: Isa> Block<A> {
    /// The address after the last instruction.
    pub fn end(&self) -> u64 {
        A::block_end(self.pc, &self.insts)
    }
}

/// Blocks translated together into one function.
///
/// * A **page region** (`nest: None`): every block is an entry (by index), and a branch to the
///   start of a block of the region stays inside the function (forward branches directly,
///   backward ones through the function's dispatch loop).
/// * A **loop region** (`nest: Some`): a loop nest with a single entry,
///   block 0 (the outer loop's header). Its blocks are laid out as structured Wasm: [`Item`]s
///   in an order where every branch inside the region goes forward or back to the header of a
///   loop that encloses it, so there is no dispatch at all.
#[derive(Clone, Debug)]
pub struct Region<A: Isa> {
    pub blocks: Vec<Block<A>>,
    /// The loop nest of a loop region (indices into `blocks`; the first item is block 0).
    pub nest: Option<Vec<Item>>,
}

impl<A: Isa> Default for Region<A> {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            nest: None,
        }
    }
}

/// One element of a loop region's layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    /// A block (index into [`Region::blocks`]).
    Block(usize),
    /// A loop: its items, the first being its header (a block). Only the header is ever
    /// entered from outside the loop.
    Loop(Vec<Item>),
}

impl<A: Isa> Region<A> {
    /// The page the region lies in (regions never cross a page: their code, and so their
    /// translation, is a function of that page's content).
    pub fn page(&self) -> u64 {
        self.blocks.first().map_or(0, |b| b.pc & !(PAGE - 1))
    }

    /// The blocks the function can be entered at (`ctx.entry`): every block of a page region,
    /// block 0 of a loop region.
    pub fn entries(&self) -> core::ops::Range<usize> {
        match self.nest {
            None => 0..self.blocks.len(),
            Some(_) => 0..self.blocks.len().min(1),
        }
    }
}

/// The blocks of a loop layout in order (indices into the region's blocks).
pub fn flatten(items: &[Item], out: &mut Vec<usize>) {
    for item in items {
        match item {
            Item::Block(b) => out.push(*b),
            Item::Loop(inner) => flatten(inner, out),
        }
    }
}

/// The first block of an item.
pub fn first(item: &Item) -> usize {
    match item {
        Item::Block(b) => *b,
        Item::Loop(inner) => inner.first().map_or(0, first),
    }
}
