//! What a back half returns for one region: a function body with the positions of its call
//! indices, so the same body can be placed in a standalone module ([`crate::module`]) or in a
//! relocatable object ([`crate::aot`]).

use alloc::vec::Vec;

/// The functions a translated body calls, by index in the module it is placed in.
#[derive(Clone, Copy, Debug)]
pub struct Callees {
    /// `jit_step: (pc: i64) -> i32` (the reference interpreter for one instruction).
    pub step: u32,
    /// `jit_fill: (addr: i64, write: i32, size: i32) -> i32` (a TLB fill).
    pub fill: u32,
    /// The front half's first helper ([`crate::ir::Op::Helper`] `n` is `helper + n`).
    pub helper: u32,
    /// The type index of the entry signature `(i32, i32, i32) -> i32` (for chaining).
    pub entry_type: u32,
}

/// Which function a recorded call site calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Callee {
    Step,
    Fill,
    /// The front half's helper `n`.
    Helper(u8),
    /// Not a call: the type index of a `return_call_indirect` (entry signature).
    EntryType,
    /// Not a call: a `return_call` to region function number `n` of the module or object
    /// (direct chaining between the regions of one page).
    Region(u32),
}

/// A function body (locals + code) with the positions of its call indices: `offset` is the
/// position in `body` of a 5-byte padded LEB function (or type) index.
#[derive(Clone, Debug, Default)]
pub struct Body {
    pub body: Vec<u8>,
    pub calls: Vec<(usize, Callee)>,
    /// Registers loaded by the prologue and stored by the epilogue.
    pub loaded: u32,
    pub stored: u32,
    /// Every load of the prologue (context, registers, flags) and store of the shared
    /// epilogue (registers, flags, context; not the conditional flag store or the chain
    /// lookup): the guest-state traffic of one call of the function.
    pub state_loads: u32,
    pub state_stores: u32,
    /// Block-scoped regions: per block, the guest-state loads and stores of its main path.
    /// Empty for regions that keep registers in locals for the whole call.
    pub block_state: Vec<(u32, u32)>,
    /// Branch hints: the offset in `body` of an `if`, and whether its then-arm is the
    /// likely one; ascending.
    pub hints: Vec<(usize, bool)>,
    /// Statistics: per block, the memory accesses off a base register an earlier access
    /// of the block used, unchanged since (their checks could merge).
    pub block_mergeable: Vec<u32>,
}

/// A function the front half supplies to every module or object its regions are placed in
/// ([`crate::ir::Op::Helper`]): A64's `pack`, say.
#[derive(Clone, Debug)]
pub struct Helper {
    /// The symbol's suffix in an object (`<prefix>_aot_<name>`).
    pub name: &'static str,
    /// Parameter and result types as Wasm value-type bytes (`0x7f` i32, `0x7e` i64, ...).
    pub params: &'static [u8],
    pub results: &'static [u8],
    /// The body (locals + code), calling nothing.
    pub body: Vec<u8>,
}

/// Writes `v` as a 5-byte padded unsigned LEB at `at` (a recorded call site).
pub fn patch_padded_leb(code: &mut [u8], at: usize, mut v: u32) {
    for i in 0..5 {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if i < 4 {
            b |= 0x80;
        }
        code[at + i] = b;
    }
}

/// Appends `v` as a 5-byte padded unsigned LEB.
pub fn push_padded_leb(code: &mut Vec<u8>, v: u32) {
    let at = code.len();
    code.resize(at + 5, 0);
    patch_padded_leb(code, at, v);
}
