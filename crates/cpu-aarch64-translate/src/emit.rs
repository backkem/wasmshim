//! Wasm code generation for one region: the back half's A64 code shape. Each block arrives
//! lowered by the A64 front half (`a64.rs`) to the kit's IR and is emitted from it. The defaults
//! of [`Options`] are what ships; the design note is `docs/design/engines-and-translation.md`.
//!
//! ## Function shape
//!
//! ```text
//! f(cpu, tlb, ctx) -> exit
//!   left = ctx.left ; base = ctx.base ; next = ctx.entry (page regions only)
//!   loop regions: load the registers the region uses into locals, and the flags if needed
//!   block $done                 ; exits whose PC is already stored (a failed jit_step)
//!     block $exit               ; exits with the PC in a local
//!       page region: loop $L { br_table next → block j … }   ; blocks end in `br $L` or an exit
//!       loop region: block 0, then the loop nest as structured Wasm `loop`s (one entry)
//!     end
//!     cpu.pc = pc
//!   end
//!   loop regions: store the registers and flags the region wrote
//!   ctx.left = left
//!   a branch to another region of this page: `return_call` it directly
//!   a branch elsewhere: look the target up in the chain table and `return_call_indirect` it
//!     when its entry is current, else return the exit code to the dispatcher
//! ```
//!
//! **Position independence.** `ctx.base` is the runtime address of the region's page; every
//! PC the code needs is `base + offset`. A64 code is position-independent within a page
//! (`ADRP` is page-relative), so one translation serves the same page content at any load
//! address: translations are keyed by page content, not by address.
//!
//! **Registers.** In a page region ([`Options::block_regs`]) a guest register or the flags live
//! in a local only within a block: loaded at the first use, written back at the block's end or
//! at an exit if changed; along a single forward edge ([`Options::carry`]) they stay in locals.
//! Between blocks, across the dispatch loop, the shared exit and calls they live in the CPU. A
//! loop region keeps the registers it uses in locals for the whole call (its back edges carry
//! them), loaded in the prologue and stored in the epilogue. Around `jit_step` the registers the
//! stepped instruction may read are stored and those it may write are reloaded.
//!
//! **Flags** are lazy: a flag-setting instruction records its operands and result and a
//! *kind* ([`kind`]); a condition is evaluated directly from them when the kind is known at
//! translation time (`cmp; b.cond`), otherwise through `pack`, a small internal function that
//! materialises NZCV in the memory format. A loop region that sets the flags before it reads
//! them neither loads nor (until it sets them) stores them ([`Options::flags_on_demand`]).
//!
//! **Memory accesses** check the TLB inline (`tag == (addr + size - 1) & !0xfff` also rejects
//! page-crossing accesses) and the alignment rules; accesses of a block off one base can share
//! one check ([`Options::merge_checks`]). Inside a loop a failed check leaves the region with
//! [`EXIT_MEM`] (no call in the loop, [`Options::mem_exits`]); elsewhere the miss arm calls
//! `jit_fill` and repeats the access, and steps the instruction with `jit_step` if the fill
//! fails.
//!
//! **Budget.** Every block checks that the budget covers it before running ([`EXIT_BUDGET`]
//! otherwise); a block that loops on itself runs as a Wasm `loop` ([`Options::self_loops`]), and
//! small loops check once for [`Options::unroll`] iterations, keeping the checked loop as the
//! exact path. Exits give back the budget of what did not run.
//!
//! The finished body's value locals are packed by live range ([`Options::compact_locals`]).
//! Every `call` uses a 5-byte LEB index, and its position is recorded, so the same body can be
//! placed in a standalone module or in a relocatable object file.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use wasm_encoder::{BlockType, Function, InstructionSink, MemArg, ValType};
use wasmshim_aarch64::{Reg, ZR, layout};

use crate::abi::*;
use crate::{Block, Region};
use wasmshim_translate::region::{first, flatten};

// ---- locals ----
const P_CPU: u32 = 0;
const P_TLB: u32 = 1;
const P_CTX: u32 = 2;
const L_LEFT: u32 = 3; // i64
const L_NEXT: u32 = 4; // i32
const L_BASE: u32 = 5; // i64: runtime address of the region's page
const L_A: u32 = 6; // i64: effective address
const L_PC: u32 = 7; // i64: exit PC
const L_E: u32 = 8; // i32: TLB entry address
const L_H: u32 = 9; // i32: host address
const L_ST: u32 = 10; // i32: jit_step status / exit code
const L_W0: u32 = 13; // i32 temps
const L_TGT: u32 = 15; // i32: direct chain target (index + 1), 0: none
const L_FK: u32 = 16; // i32: flags kind (runtime)
const L_FX: u32 = 17; // i64: flags operands and result
const L_FY: u32 = 18;
const L_FR: u32 = 19;
const L_REG0: u32 = 20; // 33 × i64: X0..X30, (31 unused), SP
const L_V0: u32 = L_REG0 + 33; // v128 temp

/// Largest loop body (instructions) given unchecked copies (`Options::unroll`).
const UNROLL_INSNS: usize = 24;

/// `FK` of a region that has not set the flags yet and did not load them.
const FLAGS_UNTOUCHED: i32 = -1;

/// No linked peers (a standalone region).
static NO_PEERS: BTreeMap<u64, (u32, u32)> = BTreeMap::new();
/// Most peers a region chains to directly; branches to others use the chain table.
const MAX_DIRECT: usize = 8;

const NREGS: usize = 33;
/// The flags' bit in the block-scoped cache masks (registers use bits 0..33).
const FLAGS: u64 = 1 << 63;
const ALL_REGS: u64 = ((1 << NREGS) - 1) & !(1 << ZR);

/// Flag kinds: how `FX`, `FY`, `FR` describe NZCV.
pub(crate) mod kind {
    use wasmshim_aarch64::lazy_flags as l;
    /// `FR` holds NZCV packed like memory: `n | z << 8 | c << 16 | v << 24` (the CPU's
    /// `lazy_flags::NONE`).
    pub const PACKED: i32 = l::NONE as i32;
    pub const LOGIC64: i32 = l::LOGIC64 as i32;
    pub const LOGIC32: i32 = l::LOGIC32 as i32;
}

pub use wasmshim_translate::emit::{Body, Callee, Callees};

/// Translates `region` (whose page starts at `page`) into a function body.
pub(crate) fn region_body(region: &Region, page: u64, callees: Callees) -> Body {
    region_body_with(region, page, callees, Options::default())
}

/// Code-shape options (the defaults are what ships).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// A block that branches back to its own start runs as a Wasm `loop`: the back edge does
    /// the entry check itself and branches to the loop, instead of going through the region's
    /// dispatch loop (`br_table`). Same exits, same accounting. Default on.
    pub self_loops: bool,
    /// In loops (a self-looping block, or any block of a loop region), a memory access that
    /// misses the TLB or fails its check leaves the region ([`EXIT_MEM`]) instead of calling
    /// `jit_fill` and `jit_step`: no call in the loop, so its values stay in registers.
    /// Elsewhere misses stay calls. Default on.
    pub mem_exits: bool,
    /// Region formation (`aot::translate_page_with`): every single-entry loop nest of a page's
    /// hot blocks becomes a region of its own, laid out as structured Wasm loops, and the page
    /// regions branch to it by exiting. Default on.
    pub loop_regions: bool,
    /// Amortised exact accounting: a loop of at most `UNROLL_INSNS` guest
    /// instructions (a self-looping block, or an innermost loop of a loop region) checks the
    /// budget once for `unroll` iterations and runs `unroll - 1` of them as copies without a
    /// check; the checked loop stays the exact slow path. `0` or `1`: off. Default 4.
    pub unroll: u32,
    /// Flags on demand: a region that sets the flags before it ever reads them
    /// does not load them in its prologue, and stores them in its epilogue only once it has
    /// set them. Default on.
    pub flags_on_demand: bool,
    /// Direct chaining: in an object, a branch from one region to an entry of
    /// another region of the same page tail-calls it directly (`return_call`) instead of
    /// looking it up in the chain table. Default on.
    pub direct_chain: bool,
    /// Block-scoped guest state: in a page region, a guest register or the
    /// flags live in a local only within a block (loaded at the first use, written back at the
    /// block's end or an exit if changed); between blocks, across the region's dispatch loop,
    /// its shared exit and its calls, they live in the CPU. No prologue loads or epilogue
    /// stores of registers. Loop regions keep registers in locals for the whole call (their
    /// back edges carry them). Default on.
    pub block_regs: bool,
    /// Block-scoped registers carried along a single forward edge: a block of
    /// a page region whose only forward predecessor is one block starts with the registers that
    /// block holds and it reads; its dispatch landing loads them. Default on.
    pub carry: bool,
    /// Memory accesses of a block off the same base share one TLB check. Default on.
    pub merge_checks: bool,
    /// Expression trees: pure values read once are computed
    /// where they are read and values used next stay on the stack (`local.tee`). Default on.
    pub trees: bool,
    /// Branch hints (default off): a `metadata.code.branch_hint` section marks budget
    /// exits and `jit_step` status exits cold, and with `hint_mem` the TLB-miss arms of memory
    /// accesses too.
    pub branch_hints: bool,
    pub hint_mem: bool,
    /// Scalar FP runs inline (a NaN result steps). Default on; off: every scalar FP
    /// instruction steps.
    pub scalar_fp: bool,
    /// The Advanced SIMD lane arithmetic of `a64/neon.rs` runs inline. Default on;
    /// off: it steps.
    pub neon: bool,
    /// Default off. On: outside loops, a memory access that misses the TLB asks `jit_fill` and
    /// joins the one copy of the access; a failed fill leaves the region (`EXIT_MEM`). Off: the
    /// access is repeated in the miss arm, which flushes and reloads the cached registers
    /// around the calls and steps on a failed fill.
    pub compact_miss: bool,
    /// Value locals packed by live range
    /// (`wasmshim_translate::locals::compact`). Every value of a block otherwise keeps a local for
    /// the whole block, and engines pay per declared local at every merge, so fewer locals
    /// make the module cheaper to compile.
    pub compact_locals: bool,
    /// Most blocks and instructions per page region (`wasmshim_translate::aot`'s
    /// `REGION_BLOCKS`, `REGION_INSNS`): a host may compile a region whole at its first call,
    /// so smaller regions compile fewer bytes at a time.
    pub region_blocks: usize,
    pub region_insns: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            self_loops: true,
            mem_exits: true,
            loop_regions: true,
            unroll: 4,
            flags_on_demand: true,
            direct_chain: true,
            block_regs: true,
            carry: true,
            merge_checks: true,
            trees: true,
            branch_hints: false,
            hint_mem: false,
            scalar_fp: true,
            neon: true,
            compact_miss: false,
            compact_locals: true,
            region_blocks: wasmshim_translate::aot::REGION_BLOCKS,
            region_insns: wasmshim_translate::aot::REGION_INSNS,
        }
    }
}

/// [`region_body`] with `opts`.
pub(crate) fn region_body_with(
    region: &Region,
    page: u64,
    callees: Callees,
    opts: Options,
) -> Body {
    region_body_linked(region, page, callees, opts, &BTreeMap::new())
}

/// [`region_body_with`] for a region linked with the other regions of its page: `peers`
/// maps the PC of every entry of those regions to (region symbol, block index), and a branch
/// out of this region to one of them becomes a direct `return_call` ([`Callee::Region`])
/// instead of a chain-table lookup.
pub(crate) fn region_body_linked<'a>(
    region: &'a Region,
    page: u64,
    callees: Callees,
    opts: Options,
    peers: &'a BTreeMap<u64, (u32, u32)>,
) -> Body {
    // Pass 1 finds the registers and flags the region uses; pass 2 emits with them.
    let mut first = Emitter::new(region, page, callees, Usage::everything());
    first.opts = opts;
    first.peers = peers;
    first.body();
    let usage = first.seen;
    let mut e = Emitter::new(region, page, callees, usage);
    e.opts = opts;
    e.peers = peers;
    e.known_rw = core::mem::take(&mut first.rw);
    e.body();
    e.finish()
}

/// Registers and flags a region touches.
#[derive(Clone, Copy)]
struct Usage {
    /// Registers read or written natively (loaded in the prologue).
    used: u64,
    /// Registers written natively (stored in the epilogue).
    written: u64,
    flags_used: bool,
    flags_written: bool,
    /// Flags read before the region itself set them (with a kind unknown at translation time,
    /// or stored for `jit_step`): only then does the prologue load them.
    flags_in: bool,
}

impl Usage {
    fn everything() -> Self {
        Self {
            used: ALL_REGS,
            written: ALL_REGS,
            flags_used: true,
            flags_written: true,
            flags_in: true,
        }
    }
    fn nothing() -> Self {
        Self {
            used: 0,
            written: 0,
            flags_used: false,
            flags_written: false,
            flags_in: false,
        }
    }
}

const fn bit(r: Reg) -> u64 {
    if r == ZR { 0 } else { 1 << r }
}

const fn reg_off(r: Reg) -> u64 {
    (layout::X + 8 * r as usize) as u64
}

fn mem(offset: u64, align: u32) -> MemArg {
    MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

struct Emitter<'r> {
    code: Vec<u8>,
    calls: Vec<(usize, Callee)>,
    region: &'r Region,
    page: u64,
    callees: Callees,
    entries: BTreeMap<u64, u32>,
    /// What the prologue loads and the epilogue stores (from pass 1).
    usage: Usage,
    /// What this pass saw.
    seen: Usage,
    len: u32,
    /// The Wasm labels enclosing the code being emitted, outermost first: a branch finds its
    /// target here, and its depth is the distance from the top.
    labels: Vec<Label>,
    /// The flags kind if known at this point of the block.
    kind: Option<i32>,
    /// Per block: instructions on the longest path of forward branches from it (the budget
    /// its entry check requires). "Forward" is in layout order ([`Self::order`]).
    reach: Vec<u64>,
    opts: Options,
    /// Entries of the other regions of the page (see [`region_body_linked`]).
    peers: &'r BTreeMap<u64, (u32, u32)>,
    /// The peers this region branches to directly, in order: `L_TGT` = index + 1.
    direct: Vec<(u32, u32)>,
    /// Block-scoped registers: the registers whose local holds the current value, and
    /// those whose local is newer than the CPU's register file. Empty at every block start.
    cached: u64,
    dirty: u64,
    /// The code being emitted is unreachable (after a branch or an exit).
    dead: bool,
    /// The cache states of the enclosing `if`s (see [`Self::if_`]).
    arms: Vec<Arm>,
    /// The instruction being emitted (its PC).
    cur_pc: u64,
    /// The registers each instruction (by PC) reads and writes, as this pass emitted it.
    rw: BTreeMap<u64, (u64, u64)>,
    /// The same, from pass 1: what pass 2 knows about the instructions still to come.
    known_rw: BTreeMap<u64, (u64, u64)>,
    /// Per instruction of the current block: the registers read by a later instruction of
    /// the block before it writes them (block-scoped registers worth keeping cached).
    live_out: Vec<u64>,
    /// Index of the instruction being emitted in its block.
    cur_k: usize,
    /// Branch hints: (offset of an `if` in `code`, then-arm likely).
    hints: Vec<(usize, bool)>,
    /// Statistics: base registers of this block's accesses so far, and the accesses that
    /// reuse one (`Body::block_mergeable`).
    bases: u64,
    mergeable: alloc::collections::BTreeSet<u64>,
    /// Page regions with `Options::carry`: forward predecessors per block in the region, and
    /// the (cached, dirty) registers a block's single forward predecessor hands it.
    fwd_preds: Vec<u32>,
    carry: BTreeMap<usize, (u64, u64)>,
    /// The IR path: the local of each value of the block being emitted, the free locals
    /// per type (i32, i64, v128), the locals in use, the types of the locals added beyond the
    /// fixed ones, the block's value types, and its integer constants.
    vlocal: BTreeMap<u32, u32>,
    vfree: [Vec<u32>; 5],
    vused: Vec<(u32, u8)>,
    vextra: Vec<u8>,
    /// IR path: a value just computed and still on the operand stack, not yet stored to its
    /// local `(value, local)`. If the next thing emitted reads it, a `local.tee` replaces the
    /// `local.set` + `local.get` pair; anything else stores it first.
    pending: Option<(u32, u32)>,
    /// IR path: pure values read once, emitted where they are read instead of through a
    /// local (expression trees; Wasm size).
    deferred: BTreeMap<u32, crate::ir::Op>,
    /// IR path: how often each value of the block is read (`count_reads`).
    reads: Vec<u32>,
    vtypes_block: Vec<crate::ir::Ty>,
    consts: BTreeMap<u32, i64>,
    /// Merged memory checks of the block ([`crate::opt::merged_checks`]): roles by
    /// instruction, and per group its `ok` and host-base locals and its low offset.
    merge: BTreeMap<u32, crate::opt::Merge>,
    groups: BTreeMap<u32, (u32, u32, i64)>,
}

/// The register cache around one `if` (block-scoped registers).
#[derive(Clone, Copy)]
struct Arm {
    /// (cached, dirty) where the `if` began.
    start: (u64, u64),
    /// After `else`: the then-arm's final state, `None` if it ended unreachable.
    then_end: Option<Option<(u64, u64)>>,
}

/// A Wasm label the emitter can branch to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Label {
    /// Exits whose PC is already stored (a failed `jit_step`); carries the exit code.
    Done,
    /// Exits with the PC in `L_PC`; carries the exit code.
    Exit,
    /// A page region's dispatch loop (`br_table` on `L_NEXT`).
    Dispatch,
    /// Where a page region's `br_table` lands for one block (before its entry check).
    Land,
    /// Forward branches to the block (or loop) at this PC land here, after its entry check.
    Fwd(u64),
    /// A loop whose header is at this PC: a back edge does the header's entry check and
    /// branches here.
    Loop(u64),
    /// An unchecked copy of the loop at this PC (amortised accounting): its back edge ends
    /// the copy and falls into the next one, without an entry check.
    Next(u64),
    /// An `if` inside a block.
    If,
    /// The blocks around a memory access with a compact miss path (`Options::compact_miss`).
    Mem,
}

impl<'r> Emitter<'r> {
    fn new(region: &'r Region, page: u64, callees: Callees, usage: Usage) -> Self {
        let entries: BTreeMap<u64, u32> = region
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.pc, i as u32))
            .collect();
        // Layout order: block index for a page region, the flattened nest for a loop region.
        let mut layout = Vec::new();
        match &region.nest {
            None => layout.extend(0..region.blocks.len()),
            Some(items) => flatten(items, &mut layout),
        }
        let mut order = alloc::vec![usize::MAX; region.blocks.len()];
        for (o, &b) in layout.iter().enumerate() {
            order[b] = o;
        }
        // Longest forward path, from the last block (in layout order) back.
        let mut reach = alloc::vec![0u64; region.blocks.len()];
        for &i in layout.iter().rev() {
            let b = &region.blocks[i];
            let further = crate::successors(b)
                .into_iter()
                .filter_map(|s| entries.get(&s).copied())
                .filter(|&j| order[j as usize] > order[i])
                .map(|j| reach[j as usize])
                .max()
                .unwrap_or(0);
            reach[i] = b.insts.len() as u64 + further;
        }
        Self {
            code: Vec::new(),
            calls: Vec::new(),
            region,
            page,
            callees,
            entries,
            usage,
            seen: Usage::nothing(),
            len: 0,
            labels: Vec::new(),
            kind: None,
            reach,
            opts: Options::default(),
            peers: &NO_PEERS,
            direct: Vec::new(),
            cached: 0,
            dirty: 0,
            dead: false,
            arms: Vec::new(),
            cur_pc: 0,
            rw: BTreeMap::new(),
            known_rw: BTreeMap::new(),
            live_out: Vec::new(),
            cur_k: 0,
            hints: Vec::new(),
            bases: 0,
            mergeable: alloc::collections::BTreeSet::new(),
            fwd_preds: Vec::new(),
            carry: BTreeMap::new(),
            vlocal: BTreeMap::new(),
            vfree: [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            vused: Vec::new(),
            vextra: Vec::new(),
            pending: None,
            deferred: BTreeMap::new(),
            reads: Vec::new(),
            vtypes_block: Vec::new(),
            consts: BTreeMap::new(),
            merge: BTreeMap::new(),
            groups: BTreeMap::new(),
        }
    }

    fn s(&mut self) -> InstructionSink<'_> {
        self.flush_pending();
        InstructionSink::new(&mut self.code)
    }

    /// Stores a pending IR value to its local (see `pending`).
    fn flush_pending(&mut self) {
        if let Some((_, l)) = self.pending.take() {
            InstructionSink::new(&mut self.code).local_set(l);
        }
    }

    /// `call f` with a padded index whose position is recorded.
    fn call(&mut self, callee: Callee) {
        self.flush_pending();
        let index = match callee {
            Callee::Step => self.callees.step,
            Callee::Fill => self.callees.fill,
            Callee::Helper(n) => self.callees.helper + u32::from(n),
            Callee::EntryType | Callee::Region(_) => self.callees.helper,
        };
        self.code.push(0x10);
        self.calls.push((self.code.len(), callee));
        push_padded_leb(&mut self.code, index);
    }

    fn finish(self) -> Body {
        let fixed = alloc::vec![
            (1, ValType::I64), // left
            (1, ValType::I32), // next
            (3, ValType::I64), // base, a, pc
            (3, ValType::I32), // e, h, st
            (2, ValType::I64), // t0, t1
            (4, ValType::I32), // w0..w2, fk
            (3, ValType::I64), // fx, fy, fr
            (NREGS as u32, ValType::I64),
            (1, ValType::V128),
        ];
        let mut locals = fixed.clone();
        // The IR path's value locals, in the order they were added.
        let extra: Vec<ValType> = self
            .vextra
            .iter()
            .map(|&t| {
                [
                    ValType::I32,
                    ValType::I64,
                    ValType::V128,
                    ValType::F32,
                    ValType::F64,
                ][t as usize]
            })
            .collect();
        for &ty in &extra {
            match locals.last_mut() {
                Some((n, last)) if *last == ty => *n += 1,
                _ => locals.push((1, ty)),
            }
        }
        let mut f = Function::new(locals);
        let header = f.byte_len();
        let mut code = Vec::new();
        let u = self.usage;
        let mut pro = InstructionSink::new(&mut code);
        pro.local_get(P_CTX)
            .i64_load(mem(CTX_LEFT, 3))
            .local_set(L_LEFT);
        if self.region.nest.is_none() {
            // A loop region has one entry: no `br_table`, no entry index.
            pro.local_get(P_CTX)
                .i32_load(mem(CTX_ENTRY, 2))
                .local_set(L_NEXT);
        }
        pro.local_get(P_CTX)
            .i64_load(mem(CTX_BASE, 3))
            .local_set(L_BASE);
        for r in regs(u.used) {
            pro.local_get(P_CPU)
                .i64_load(mem(reg_off(r), 3))
                .local_set(L_REG0 + u32::from(r));
        }
        // Block-scoped state: flags, like registers, live in the CPU between blocks.
        let block = self.block_regs();
        let flags_written = u.flags_written && !block;
        let load_flags = !block && u.flags_used && (u.flags_in || !self.opts.flags_on_demand);
        if !load_flags && flags_written {
            // The region sets the flags before it reads them: nothing to load; FK = -1
            // says "untouched" until it does, and the epilogue stores nothing then.
            pro.i32_const(FLAGS_UNTOUCHED).local_set(L_FK);
        }
        if load_flags {
            load_lazy_flags(&mut pro);
        }
        debug_assert!(self.pending.is_none(), "an IR value left on the stack");
        let body_start = code.len();
        code.extend_from_slice(&self.code);
        let mut epi = InstructionSink::new(&mut code);
        epi.local_set(L_ST);
        for r in regs(u.written) {
            epi.local_get(P_CPU)
                .local_get(L_REG0 + u32::from(r))
                .i64_store(mem(reg_off(r), 3));
        }
        let mut calls: Vec<(usize, Callee)> = self
            .calls
            .iter()
            .map(|&(o, c)| (header + body_start + o, c))
            .collect();
        let hints: Vec<(usize, bool)> = if self.opts.branch_hints {
            self.hints
                .iter()
                .map(|&(o, likely)| (header + body_start + o, likely))
                .collect()
        } else {
            Vec::new()
        };
        if flags_written {
            if load_flags {
                store_lazy_flags(&mut epi);
            } else {
                epi.local_get(L_FK)
                    .i32_const(FLAGS_UNTOUCHED)
                    .i32_ne()
                    .if_(BlockType::Empty);
                store_lazy_flags(&mut epi);
                epi.end();
            }
        }
        InstructionSink::new(&mut code)
            .local_get(P_CTX)
            .local_get(L_LEFT)
            .i64_store(mem(CTX_LEFT, 3));
        // Direct chaining: a branch to another region of this page tail-calls it; the
        // page is the same, so `ctx.base` stays. (Valid: both regions translate the code this
        // call validated, and code can only change through `jit_step`, which never exits here.)
        for (k, &(region, block)) in self.direct.iter().enumerate() {
            InstructionSink::new(&mut code)
                .local_get(L_TGT)
                .i32_const(k as i32 + 1)
                .i32_eq()
                .if_(BlockType::Empty)
                .local_get(P_CTX)
                .i32_const(block as i32)
                .i32_store(mem(CTX_ENTRY, 2))
                .local_get(P_CPU)
                .local_get(P_TLB)
                .local_get(P_CTX);
            code.push(0x12); // return_call
            calls.push((header + code.len(), Callee::Region(region)));
            push_padded_leb(&mut code, 0);
            InstructionSink::new(&mut code).end();
        }
        let type_at = chain(&mut code, self.callees.entry_type);

        calls.push((header + type_at, Callee::EntryType));
        InstructionSink::new(&mut code).local_get(L_ST).end();
        let mut hints = hints;
        // Every local after the parameters may move: the fixed ones a region does not use go
        // too (a fixed local read before any write keeps its own zero-initialised slot).
        let all: Vec<ValType> = fixed
            .iter()
            .flat_map(|&(n, t)| core::iter::repeat_n(t, n as usize))
            .chain(extra.iter().copied())
            .collect();
        if self.opts.compact_locals
            && let Some(c) = wasmshim_translate::locals::compact(&code, P_CTX + 1, &[], &all)
        {
            // Same instructions, fewer locals: recorded offsets move with the shorter indices.
            f = Function::new(c.locals.iter().copied());
            let header2 = f.byte_len();
            for (o, _) in &mut calls {
                *o = header2 + c.map(*o - header);
            }
            for (o, _) in &mut hints {
                *o = header2 + c.map(*o - header);
            }
            code = c.code;
        }
        f.raw(code);
        Body {
            body: f.into_raw_body(),
            calls,
            loaded: u.used.count_ones(),
            stored: u.written.count_ones(),
            // 3 context loads + registers + 5 flag loads; registers + 4 flag stores + `left`.
            state_loads: 2
                + u32::from(self.region.nest.is_none())
                + u.used.count_ones()
                + if load_flags { 5 } else { 0 },
            state_stores: u.written.count_ones() + if flags_written { 4 } else { 0 } + 1,
            block_state: if block {
                self.region
                    .blocks
                    .iter()
                    .map(|b| {
                        let (mut read, mut written) = (0u64, 0u64);
                        for k in 0..b.insts.len() as u64 {
                            let (r, w) = self
                                .known_rw
                                .get(&(b.pc + 4 * k))
                                .copied()
                                .unwrap_or_default();
                            read |= r & !written;
                            written |= w;
                        }
                        let n = |m: u64| (m & !FLAGS).count_ones();
                        let f = |m: u64, k: u32| if m & FLAGS != 0 { k } else { 0 };
                        (n(read) + f(read, 5), n(written) + f(written, 4))
                    })
                    .collect()
            } else {
                Vec::new()
            },
            hints,
            block_mergeable: self
                .region
                .blocks
                .iter()
                .map(|b| self.mergeable.range(b.pc..b.end()).count() as u32)
                .collect(),
        }
    }

    fn body(&mut self) {
        self.s()
            .block(BlockType::Result(ValType::I32))
            .block(BlockType::Result(ValType::I32));
        self.labels.extend([Label::Done, Label::Exit]);
        match &self.region.nest {
            None => self.page_body(),
            Some(items) => self.loop_body(items),
        }
        // Every path above ended in a branch; then the `$exit` tail.
        self.s().unreachable().end();
        self.labels.pop();
        self.s()
            .local_set(L_ST)
            .local_get(P_CPU)
            .local_get(L_PC)
            .i64_store(mem(layout::PC as u64, 3))
            .local_get(L_ST)
            .end();
        self.labels.pop();
        debug_assert!(self.labels.is_empty());
    }

    /// A page region: every block is an entry, through a dispatch loop.
    ///
    /// ```text
    /// loop $L
    ///   block $F[k-1] block $B[k-1] … block $F[1] block $B[1] block $B[0]
    ///     br_table next …
    ///   end <check 0> <block 0> end <check 1> end <block 1> …
    /// end
    /// ```
    ///
    /// Every block `j > 0` has two labels: `$B[j]` ends before its entry check (the `br_table`
    /// lands there) and `$F[j]` ends after it (forward branches land there, skipping the check).
    fn page_body(&mut self) {
        let region = self.region;
        let k = region.blocks.len();
        self.s().loop_(BlockType::Empty);
        self.labels.push(Label::Dispatch);
        for j in (0..k).rev() {
            if j > 0 {
                self.s().block(BlockType::Empty);
                self.labels.push(Label::Fwd(region.blocks[j].pc));
            }
            self.s().block(BlockType::Empty);
            self.labels.push(Label::Land);
        }
        // From inside $B[0]: $B[0] is 0, $B[j] is 2j - 1.
        let table: Vec<u32> = (0..k as u32)
            .map(|j| if j == 0 { 0 } else { 2 * j - 1 })
            .collect();
        let default = table.last().copied().unwrap_or(0);
        self.s().local_get(L_NEXT).br_table(table, default);
        // Block-scoped registers carried along a forward edge: a block whose only
        // forward predecessor in the region is one block may start with that block's cached
        // registers; its dispatch landing loads them instead.
        if self.block_regs() && self.opts.carry {
            let index: BTreeMap<u64, usize> = region
                .blocks
                .iter()
                .enumerate()
                .map(|(i, b)| (b.pc, i))
                .collect();
            self.fwd_preds = alloc::vec![0; k];
            for (i, b) in region.blocks.iter().enumerate() {
                for s in crate::successors(b) {
                    if let Some(&j) = index.get(&s)
                        && j > i
                    {
                        self.fwd_preds[j] += 1;
                    }
                }
            }
        }
        for (i, block) in region.blocks.iter().enumerate() {
            self.close(); // $B[i]: the dispatch loop lands here
            self.entry_check(i);
            if i > 0 {
                let carried = self.carry.get(&i).copied();
                if let Some((s, _)) = carried {
                    self.fill_cache(s);
                }
                self.close(); // $F[i]: forward branches land here
                if let Some((s, d)) = carried {
                    (self.cached, self.dirty) = (s, d);
                }
            }
            self.block(block);
        }
        self.close(); // the dispatch loop
    }

    /// The registers a block reads before it writes them (as pass 1 saw it).
    fn live_in(&self, j: usize) -> u64 {
        let b = &self.region.blocks[j];
        let (mut read, mut written) = (0u64, 0u64);
        for k in 0..b.insts.len() as u64 {
            let (r, w) = self
                .known_rw
                .get(&(b.pc + 4 * k))
                .copied()
                .unwrap_or((ALL_REGS, 0));
            read |= r & !written;
            written |= w;
        }
        read
    }

    /// If a branch from here to `target` may carry cached registers (see [`Self::page_body`]):
    /// the target block and the registers it would start with.
    fn carry_set(&self, target: u64) -> Option<(usize, u64)> {
        if !self.block_regs() || !self.opts.carry || self.depth_of(Label::Fwd(target)).is_none() {
            return None;
        }
        let j = *self.entries.get(&target)? as usize;
        (self.fwd_preds.get(j) == Some(&1)).then(|| (j, self.cached & self.live_in(j) & !FLAGS))
    }

    /// A loop region: one entry, block 0, then the nest as structured code.
    ///
    /// ```text
    /// <check 0>
    /// loop $H0                       ; the outer loop (a back edge: check 0, br $H0)
    ///   block $F[n-1] … block $F[1]
    ///     <item 0: the header block>
    ///   end <item 1> … end <item n-1>  ; an item is a block or a nested loop of the same shape
    /// end
    /// ```
    fn loop_body(&mut self, items: &[crate::Item]) {
        self.entry_check(0);
        self.items(items);
    }

    /// Lays out `items`: forward branches to item `m` land at the end of its label.
    fn items(&mut self, items: &[crate::Item]) {
        let region = self.region;
        for item in items.iter().skip(1).rev() {
            self.s().block(BlockType::Empty);
            self.labels.push(Label::Fwd(region.blocks[first(item)].pc));
        }
        for (m, item) in items.iter().enumerate() {
            if m > 0 {
                self.close();
            }
            match item {
                crate::Item::Block(b) => self.block(&region.blocks[*b]),
                crate::Item::Loop(inner) => {
                    let header = first(item);
                    self.s().loop_(BlockType::Empty);
                    self.labels.push(Label::Loop(region.blocks[header].pc));
                    // An innermost loop of plain blocks: unchecked copies while the budget
                    // covers them (each iteration takes at most the header's reach).
                    let blocks: Option<Vec<usize>> = inner
                        .iter()
                        .map(|i| match i {
                            crate::Item::Block(b) if !self.loops_on_itself(*b) => Some(*b),
                            _ => None,
                        })
                        .collect();
                    let insns: usize = blocks
                        .iter()
                        .flatten()
                        .map(|&b| region.blocks[b].insts.len())
                        .sum();
                    if blocks.is_some() && insns <= UNROLL_INSNS {
                        let per = self.reach[header];
                        self.unchecked_copies(region.blocks[header].pc, per, |e| e.items(inner));
                    }
                    self.items(inner);
                    self.close();
                }
            }
        }
    }

    /// `true` if block `b` branches to its own start (and so becomes a Wasm `loop`).
    fn loops_on_itself(&self, b: usize) -> bool {
        let block = &self.region.blocks[b];
        self.opts.self_loops && crate::successors(block).contains(&block.pc)
    }

    /// Amortised exact accounting: at the top of the loop at `pc`, if the budget covers
    /// `opts.unroll` more iterations of at most `per` instructions each, runs `unroll - 1` of
    /// them as copies of the body (`body`) whose back edges skip the entry check and fall into
    /// the next copy; the last iteration is the ordinary, checked body that follows. Every
    /// copy starts with at least the loop's reach left, as a checked iteration would, so exits
    /// happen at the same points with the same budget.
    fn unchecked_copies(&mut self, pc: u64, per: u64, body: impl Fn(&mut Self)) {
        let k = u64::from(self.opts.unroll);
        if k < 2 {
            return;
        }
        let j = self.entries[&pc] as usize;
        let need = (k - 1) * per + self.reach[j];
        self.s().local_get(L_LEFT).i64_const(need as i64).i64_ge_s();
        self.if_hint(true);
        for _ in 1..k {
            self.s().block(BlockType::Empty);
            self.labels.push(Label::Next(pc));
            body(self);
            self.close();
        }
        self.end();
    }

    // ---- control ----

    /// Ends the innermost structural label. Code after it is reached by branches, which leave
    /// every register in the register file: the block-scoped cache starts empty.
    fn close(&mut self) {
        self.s().end();
        self.labels.pop();
        self.cached = 0;
        self.dirty = 0;
        self.dead = false;
    }

    /// The branch depth of the innermost label equal to `label`, if one encloses the code.
    fn depth_of(&self, label: Label) -> Option<u32> {
        self.labels
            .iter()
            .rposition(|&l| l == label)
            .map(|i| (self.labels.len() - 1 - i) as u32)
    }

    /// Inside a loop (a self-looping block or a loop region): slow paths become side exits.
    fn in_loop(&self) -> bool {
        self.labels.iter().any(|l| matches!(l, Label::Loop(_)))
    }

    /// Opens an `if` (the condition is on the stack). With block-scoped registers, each arm
    /// starts from the cache state here and [`Self::end`] reconciles the two.
    fn if_(&mut self) {
        self.s().if_(BlockType::Empty);
        self.labels.push(Label::If);
        self.arms.push(Arm {
            start: (self.cached, self.dirty),
            then_end: None,
        });
    }

    /// [`Self::if_`] with a branch hint: `likely` if the then-arm is the hot path. The
    /// hint goes into the module's `metadata.code.branch_hint` section, which lets the
    /// compiler move the cold arm out of the hot path's way; it never changes semantics.
    fn if_hint(&mut self, likely: bool) {
        self.flush_pending();
        self.hints.push((self.code.len(), likely));
        self.if_();
    }

    /// The TLB-hit test of a memory access: hinted likely if `Options::hint_mem`.
    fn if_mem_hint(&mut self) {
        if self.opts.hint_mem {
            self.if_hint(true);
        } else {
            self.if_();
        }
    }

    fn else_(&mut self) {
        let arm = self.arms.last_mut().expect("if");
        arm.then_end = Some((!self.dead).then_some((self.cached, self.dirty)));
        (self.cached, self.dirty) = arm.start;
        self.dead = false;
        self.s().else_();
    }

    /// Closes an `if`. The arm still open reconciles its cache with the other one's (the
    /// then-arm, or the implicit empty else-arm): it loads the registers the other arm left
    /// dirty that it does not hold, and writes back those it left dirty that the other does
    /// not hold. After the `if`, a register is cached if both arms hold it, dirty if either
    /// left it dirty.
    fn end(&mut self) {
        let arm = self.arms.pop().expect("if");
        let other = match arm.then_end {
            Some(then) => then,
            None => Some(arm.start),
        };
        if !self.dead
            && let Some((c_other, d_other)) = other
        {
            // Everything the other arm holds (the slow arm of a memory access gets the fast
            // arm's registers back after its calls), and the flags if it changed them.
            let live = self.live_after();
            self.fill_cache((c_other & live & !FLAGS) | d_other);
            self.flush(self.dirty & !c_other);
            self.cached &= c_other;
            self.dirty = (self.dirty | d_other) & self.cached;
        } else if self.dead
            && let Some(state) = other
        {
            // (If both arms left, so does the `if`.)
            (self.cached, self.dirty) = state;
            self.dead = false;
        }
        self.s().end();
        let l = self.labels.pop();
        debug_assert_eq!(l, Some(Label::If));
    }

    /// Pushes the runtime address of `pc`.
    fn pc(&mut self, pc: u64) {
        let off = pc.wrapping_sub(self.page) as i64;
        self.s().local_get(L_BASE);
        if off != 0 {
            self.s().i64_const(off).i64_add();
        }
    }

    /// Leaves with `code`, the PC on the stack (i64).
    fn exit_pc_on_stack(&mut self, code: i32) {
        self.flush(self.dirty);
        let depth = self.depth_of(Label::Exit).expect("$exit");
        self.s().local_set(L_PC).i32_const(code).br(depth);
        self.dead = true;
    }

    fn exit_to(&mut self, code: i32, pc: u64) {
        self.pc(pc);
        self.exit_pc_on_stack(code);
    }

    /// Leaves with the status in `L_ST`, the PC already stored, giving back `adjust`.
    fn exit_stored(&mut self, adjust: i64) {
        if adjust != 0 {
            self.s()
                .local_get(L_LEFT)
                .i64_const(adjust)
                .i64_add()
                .local_set(L_LEFT);
        }
        self.flush(self.dirty);
        let depth = self.depth_of(Label::Done).expect("$done");
        self.s().local_get(L_ST).br(depth);
        self.dead = true;
    }

    /// Transfers control to `target` at the end of a block.
    fn jump(&mut self, target: u64) {
        // Block-scoped registers: every block ends with the register file up to date, except
        // for the registers it hands to a block it is the only forward predecessor of.
        match self.carry_set(target) {
            Some((j, s)) => {
                self.flush(self.dirty & !s);
                self.carry.insert(j, (s, self.dirty & s));
            }
            None => self.flush(self.dirty),
        }
        self.jump_to(target);
        self.dead = true;
    }

    fn jump_to(&mut self, target: u64) {
        // A block (or loop) later in the layout: its label encloses this code, so a direct
        // branch gets there, skipping its entry check (the check that led here covered it).
        if let Some(depth) = self.depth_of(Label::Fwd(target)) {
            self.s().br(depth);
            return;
        }
        // The back edge of an unchecked copy (amortised accounting): into the next copy.
        if let Some(depth) = self.depth_of(Label::Next(target)) {
            self.s().br(depth);
            return;
        }
        // The header of an enclosing loop (a self-looping block, or a loop of a loop region):
        // the entry check the dispatch loop would do, then straight back to the loop.
        if self.depth_of(Label::Loop(target)).is_some() {
            let j = self.entries[&target] as usize;
            let reach = self.reach[j] as i64;
            self.s().local_get(L_LEFT).i64_const(reach).i64_lt_s();
            self.if_hint(false);
            self.exit_to(EXIT_BUDGET, target);
            self.end();
            let depth = self.depth_of(Label::Loop(target)).expect("loop");
            self.s().br(depth);
            return;
        }
        match (
            self.entries.get(&target).copied(),
            self.depth_of(Label::Dispatch),
        ) {
            // An earlier block of a page region: through the dispatch loop.
            (Some(j), Some(depth)) => {
                self.s().i32_const(j as i32).local_set(L_NEXT).br(depth);
            }
            _ => {
                // Another region of this page: leave with its index, the epilogue
                // tail-calls it directly.
                if let Some(&peer) = self.peers.get(&target) {
                    let k = match self.direct.iter().position(|&d| d == peer) {
                        Some(k) => Some(k),
                        None if self.direct.len() < MAX_DIRECT => {
                            self.direct.push(peer);
                            Some(self.direct.len() - 1)
                        }
                        None => None,
                    };
                    if let Some(k) = k {
                        self.s().i32_const(k as i32 + 1).local_set(L_TGT);
                    }
                }
                self.exit_to(EXIT_JUMP, target)
            }
        }
    }

    // ---- registers ----

    /// Block-scoped registers: in a page region, a guest register lives in its local only
    /// within a block, loaded at its first use and written back at the block's end (or an
    /// exit) if it changed; across blocks, and so across the region's merge points and calls,
    /// it lives in the CPU's register file.
    fn block_regs(&self) -> bool {
        self.opts.block_regs && self.region.nest.is_none()
    }

    /// Writes the cached registers (and [`FLAGS`]) in `mask` back to the CPU.
    fn flush(&mut self, mask: u64) {
        for r in regs(mask & self.dirty) {
            self.s()
                .local_get(P_CPU)
                .local_get(L_REG0 + u32::from(r))
                .i64_store(mem(reg_off(r), 3));
        }
        if mask & self.dirty & FLAGS != 0 {
            self.store_flags_now();
        }
        self.dirty &= !mask;
    }

    /// Stores the flag locals into the CPU, only the fields the kind known here needs.
    fn store_flags_now(&mut self) {
        let cpu = |s: &mut InstructionSink<'_>, local: u32, at: usize| {
            s.local_get(P_CPU)
                .local_get(local)
                .i64_store(mem(at as u64, 3));
        };
        let kind_is = |s: &mut InstructionSink<'_>, k: i32| {
            s.local_get(P_CPU)
                .i32_const(k)
                .i32_store(mem(layout::FLAGS_KIND as u64, 2));
        };
        self.flush_pending();
        let mut s = InstructionSink::new(&mut self.code);
        match self.kind {
            Some(kind::PACKED) => {
                kind_is(&mut s, kind::PACKED);
                s.local_get(P_CPU)
                    .local_get(L_FR)
                    .i32_wrap_i64()
                    .i32_store(mem(layout::NZCV as u64, 2));
            }
            Some(k @ (kind::LOGIC64 | kind::LOGIC32)) => {
                kind_is(&mut s, k);
                cpu(&mut s, L_FR, layout::FLAGS_R);
            }
            Some(k) => {
                kind_is(&mut s, k);
                cpu(&mut s, L_FX, layout::FLAGS_X);
                cpu(&mut s, L_FY, layout::FLAGS_Y);
                cpu(&mut s, L_FR, layout::FLAGS_R);
            }
            None => store_lazy_flags(&mut s),
        }
    }

    /// The registers the rest of the current block may read (the current instruction's own
    /// reads included).
    fn live_after(&self) -> u64 {
        let reads = self.known_rw.get(&self.cur_pc).map_or(ALL_REGS, |rw| rw.0);
        self.live_out
            .get(self.cur_k)
            .map_or(ALL_REGS, |&live| live | reads)
    }

    /// Before a call in a block-scoped region: no cached register is live across it (the dirty
    /// ones are written back, every one is loaded again when used), so no register value has
    /// to survive the call in a callee-saved register or a stack slot. The flags stay.
    fn regs_before_call(&mut self) {
        if self.block_regs() {
            self.flush(self.dirty & !FLAGS);
            self.cached &= FLAGS;
        }
    }

    /// Loads the cached state in `mask` that is not cached yet (registers, [`FLAGS`]).
    fn fill_cache(&mut self, mask: u64) {
        let missing = mask & !self.cached;
        for r in regs(missing) {
            self.s()
                .local_get(P_CPU)
                .i64_load(mem(reg_off(r), 3))
                .local_set(L_REG0 + u32::from(r));
        }
        if missing & FLAGS != 0 {
            load_lazy_flags(&mut self.s());
        }
        self.cached |= missing;
    }

    /// Before code reads the flag locals with a kind unknown at translation time: in a
    /// block-scoped region, they come from the CPU unless this block holds them.
    fn need_flags(&mut self) {
        self.rw.entry(self.cur_pc).or_default().0 |= FLAGS;
        if self.block_regs() {
            self.fill_cache(FLAGS);
        }
    }

    /// The flag locals were (re)written.
    fn flags_written(&mut self) {
        self.rw.entry(self.cur_pc).or_default().1 |= FLAGS;
        if self.block_regs() {
            self.cached |= FLAGS;
            self.dirty |= FLAGS;
        }
    }

    fn get(&mut self, r: Reg) {
        self.rw.entry(self.cur_pc).or_default().0 |= bit(r);
        if r == ZR {
            self.s().i64_const(0);
        } else if self.block_regs() {
            if self.cached & bit(r) == 0 {
                self.cached |= bit(r);
                self.s()
                    .local_get(P_CPU)
                    .i64_load(mem(reg_off(r), 3))
                    .local_tee(L_REG0 + u32::from(r));
            } else {
                self.s().local_get(L_REG0 + u32::from(r));
            }
        } else {
            self.seen.used |= bit(r);
            self.s().local_get(L_REG0 + u32::from(r));
        }
    }

    fn set(&mut self, r: Reg) {
        self.rw.entry(self.cur_pc).or_default().1 |= bit(r);
        self.bases &= !bit(r);
        if r == ZR {
            self.s().drop();
        } else if self.block_regs() {
            self.cached |= bit(r);
            self.dirty |= bit(r);
            self.s().local_set(L_REG0 + u32::from(r));
        } else {
            self.seen.used |= bit(r);
            self.seen.written |= bit(r);
            self.s().local_set(L_REG0 + u32::from(r));
        }
    }

    /// Stores the registers in `mask` that native code may have changed.
    fn store_regs(&mut self, mask: u64) {
        if self.block_regs() {
            self.flush(mask);
            return;
        }
        for r in regs(mask & self.usage.written) {
            self.s()
                .local_get(P_CPU)
                .local_get(L_REG0 + u32::from(r))
                .i64_store(mem(reg_off(r), 3));
        }
    }

    /// Reloads the registers in `mask` that native code keeps in locals.
    fn reload_regs(&mut self, mask: u64) {
        if self.block_regs() {
            // The register file is current; the next use loads it again.
            self.cached &= !mask;
            self.dirty &= !mask;
            return;
        }
        for r in regs(mask & self.usage.used) {
            self.s()
                .local_get(P_CPU)
                .i64_load(mem(reg_off(r), 3))
                .local_set(L_REG0 + u32::from(r));
        }
    }

    // ---- flags ----

    // ---- width-generic arithmetic ----

    // ---- blocks ----

    /// Entering block `i` through the dispatch loop or the function entry: the budget must
    /// cover the longest forward path from here, or none of it runs here (the engine finishes
    /// the budget, exactly, in the interpreter). Forward branches skip this: the check at the
    /// path's start already covered them. (Back edges to a loop header check at the branch.)
    fn entry_check(&mut self, i: usize) {
        let pc = self.region.blocks[i].pc;
        let reach = self.reach[i] as i64;
        self.s().local_get(L_LEFT).i64_const(reach).i64_lt_s();
        self.if_hint(false);
        self.exit_to(EXIT_BUDGET, pc);
        self.end();
    }

    fn block(&mut self, block: &Block) {
        let self_loop = self.opts.self_loops && crate::successors(block).contains(&block.pc);
        if self_loop {
            self.s().loop_(BlockType::Empty);
            self.labels.push(Label::Loop(block.pc));
            if block.insts.len() <= UNROLL_INSNS {
                let n = block.insts.len() as u64;
                self.unchecked_copies(block.pc, n, |e| e.block_body(block));
            }
        }
        self.block_body(block);
        if self_loop {
            // Every path of the body left with a branch; nothing falls out of the loop.
            self.close();
        }
    }

    /// One execution of `block`: its budget, its instructions, the branch out.
    fn block_body(&mut self, block: &Block) {
        self.kind = None;
        self.len = block.insts.len() as u32;
        let n = i64::from(self.len);
        self.s()
            .local_get(L_LEFT)
            .i64_const(n)
            .i64_sub()
            .local_set(L_LEFT);
        // Liveness within the block, from what pass 1 saw each instruction read and write.
        self.bases = 0;
        self.live_out.clear();
        self.live_out.resize(block.insts.len(), 0);
        let mut live = 0u64;
        for k in (0..block.insts.len()).rev() {
            self.live_out[k] = live;
            let (r, w) = self
                .known_rw
                .get(&(block.pc + 4 * k as u64))
                .copied()
                .unwrap_or((ALL_REGS, 0));
            live = r | (live & !w);
        }
        self.block_ir(block);
    }

    // ---- the IR back half ----

    /// The block through the IR: the A64 front half lowers it, the ops are emitted here.
    fn block_ir(&mut self, block: &Block) {
        let ir = crate::a64::lower(block, self.page, &self.opts);
        self.vlocal.clear();
        for f in &mut self.vfree {
            f.clear();
        }
        let used = core::mem::take(&mut self.vused);
        for (l, t) in used {
            self.vfree[t as usize].push(l);
        }
        self.vtypes_block = ir.types.clone();
        self.consts.clear();
        self.groups.clear();
        // How often each value is read as the back half emits it (a memory access's fast
        // path and a division's operands are emitted more than once).
        let mut reads = alloc::vec![0u32; ir.types.len()];
        for ops in &ir.ops {
            crate::ir::count_reads(ops, 1, &mut reads);
        }
        if let crate::ir::Term::Branch(c, ..) = ir.term {
            crate::ir::bump_reads(&mut reads, c, 1);
        }
        self.reads = reads;
        self.deferred.clear();
        self.merge = if self.opts.merge_checks {
            crate::opt::merged_checks(&ir, 4096)
        } else {
            BTreeMap::new()
        };
        for (k, ops) in ir.ops.iter().enumerate() {
            self.cur_pc = block.pc + 4 * k as u64;
            self.cur_k = k;
            self.ir_ops(ops);
            if self.dead {
                return;
            }
        }
        match ir.term {
            crate::ir::Term::Jump(off) => self.jump(self.page.wrapping_add(off)),
            crate::ir::Term::Branch(c, taken, fall) => {
                self.vget(c);
                self.cond_jump(self.page.wrapping_add(taken), self.page.wrapping_add(fall));
            }
            crate::ir::Term::Left => {}
        }
    }

    /// The Wasm local of value `v` (one of a pool reused from block to block).
    fn vloc(&mut self, v: crate::ir::V) -> u32 {
        if let Some(&l) = self.vlocal.get(&v.0) {
            return l;
        }
        let ty = self.vtypes_block[v.0 as usize].index();
        let l = match self.vfree[ty].pop() {
            Some(l) => l,
            None => {
                self.vextra.push(ty as u8);
                L_V0 + self.vextra.len() as u32
            }
        };
        self.vused.push((l, ty as u8));
        self.vlocal.insert(v.0, l);
        l
    }

    fn vget(&mut self, v: crate::ir::V) {
        if v == crate::ir::CPU {
            self.s().local_get(P_CPU);
            return;
        }
        if let Some(op) = self.deferred.get(&v.0).cloned() {
            // Kept for a second emission of the same read (a repeated fast path).
            self.value_op(&op);
            return;
        }
        if let Some((pv, l)) = self.pending
            && pv == v.0
        {
            // The value is still on the stack: keep it there and store it on the way.
            self.pending = None;
            InstructionSink::new(&mut self.code).local_tee(l);
            return;
        }
        let l = self.vloc(v);
        self.s().local_get(l);
    }

    fn vset(&mut self, v: crate::ir::V) {
        let l = self.vloc(v);
        self.flush_pending();
        if self.opts.trees {
            self.pending = Some((v.0, l));
        } else {
            self.s().local_set(l);
        }
    }

    /// Emits a pure value operation, leaving its result on the stack.
    fn value_op(&mut self, op: &crate::ir::Op) {
        use crate::ir::Op;
        match op {
            Op::I32(_, c) => {
                self.s().i32_const(*c);
            }
            Op::I64(_, c) => {
                self.s().i64_const(*c);
            }
            Op::V128(_, c) => {
                self.s().v128_const(*c);
            }
            Op::Pure(_, p, args) => {
                for &a in args {
                    self.vget(a);
                }
                p.emit(&mut self.s());
            }
            Op::Select(_, a, b, c) => {
                self.vget(*a);
                self.vget(*b);
                self.vget(*c);
                self.s().select();
            }
            Op::Lane(_, l, i, args) => {
                for &a in args {
                    self.vget(a);
                }
                l.emit(&mut self.s(), *i);
            }
            Op::Shuffle(_, lanes, a, b) => {
                self.vget(*a);
                self.vget(*b);
                self.s().i8x16_shuffle(*lanes);
            }
            _ => unreachable!("not a pure value operation"),
        }
    }

    fn ir_ops(&mut self, ops: &[crate::ir::Op]) {
        use crate::ir::{Base, Op};
        for op in ops {
            if self.dead {
                return;
            }
            match op {
                Op::I32(d, _)
                | Op::I64(d, _)
                | Op::V128(d, _)
                | Op::Pure(d, ..)
                | Op::Select(d, ..)
                | Op::Lane(d, ..)
                | Op::Shuffle(d, ..) => {
                    match op {
                        Op::I32(_, c) => {
                            self.consts.insert(d.0, i64::from(*c));
                        }
                        Op::I64(_, c) => {
                            self.consts.insert(d.0, *c);
                        }
                        _ => {}
                    }
                    if self.opts.trees && self.reads.get(d.0 as usize) == Some(&1) {
                        // Read once: computed where it is read.
                        self.deferred.insert(d.0, op.clone());
                    } else {
                        self.value_op(op);
                        self.vset(*d);
                    }
                }
                Op::Load(d, l, base, off) => {
                    match base {
                        Base::Cpu => {
                            self.s().local_get(P_CPU);
                        }
                        Base::Val(b) => self.vget(*b),
                    }
                    l.emit(&mut self.s(), *off);
                    self.vset(*d);
                }
                Op::Store(st, base, off, v) => {
                    match base {
                        Base::Cpu => {
                            self.s().local_get(P_CPU);
                        }
                        Base::Val(b) => self.vget(*b),
                    }
                    self.vget(*v);
                    st.emit(&mut self.s(), *off);
                }
                Op::Get(d, slot) => {
                    self.slot_get(*slot);
                    self.vset(*d);
                }
                Op::Set(slot, v) => {
                    self.vget(*v);
                    if slot.0 == crate::a64::FK.0 {
                        self.kind = self.consts.get(&v.0).map(|&c| c as i32);
                    }
                    self.slot_set(*slot);
                }
                Op::Pc(d, off) => {
                    self.pc(self.page.wrapping_add(*off as u64));
                    self.vset(*d);
                }
                Op::Div {
                    d,
                    wide,
                    signed,
                    x,
                    y,
                } => self.ir_div(*d, *wide, *signed, *x, *y),
                Op::If {
                    cond,
                    then,
                    else_,
                    out,
                } => {
                    self.vget(*cond);
                    self.if_();
                    self.ir_ops(then);
                    if !self.dead {
                        for &(d, t, _) in out {
                            self.vget(t);
                            self.vset(d);
                        }
                    }
                    self.else_();
                    self.ir_ops(else_);
                    if !self.dead {
                        for &(d, _, e) in out {
                            self.vget(e);
                            self.vset(d);
                        }
                    }
                    self.end();
                    // What the arms did to the flags kind is known only at run time.
                    if then.iter().chain(else_.iter()).any(sets_kind) {
                        self.kind = None;
                    }
                }
                Op::Mem(m) => self.ir_mem(m),
                Op::Step {
                    pc,
                    k,
                    reads,
                    writes,
                } => {
                    let flags = crate::a64::FLAG_SLOTS;
                    self.call_step(
                        *k,
                        self.page.wrapping_add(*pc),
                        reads & ALL_REGS,
                        writes & ALL_REGS,
                        (reads & flags != 0, writes & flags != 0),
                    );
                }
                Op::Helper(d, index, args) => {
                    debug_assert_eq!(*index, crate::a64::HELPER_PACK);
                    self.regs_before_call();
                    for &a in args {
                        self.vget(a);
                    }
                    self.call(Callee::Helper(0));
                    self.vset(*d);
                }
                Op::Exit { code, pc, giveback } => {
                    if *giveback != 0 {
                        self.s()
                            .local_get(L_LEFT)
                            .i64_const(i64::from(*giveback))
                            .i64_add()
                            .local_set(L_LEFT);
                    }
                    self.vget(*pc);
                    self.exit_pc_on_stack(*code);
                }
            }
        }
    }

    /// Pushes a slot's value (a register through the block-scoped cache, a flag slot from the
    /// flag locals).
    fn slot_get(&mut self, slot: crate::ir::Slot) {
        match slot_local(slot) {
            Some(l) => {
                if self.kind.is_none() {
                    self.need_flags();
                }
                // (Region-scoped registers: the prologue loads the flags if this read may see
                // the incoming ones.)
                self.seen.flags_in |= self.kind.is_none();
                self.seen.flags_used = true;
                self.rw.entry(self.cur_pc).or_default().0 |= FLAGS;
                self.s().local_get(l);
            }
            None => self.get(slot.0),
        }
    }

    /// Pops a value into a slot.
    fn slot_set(&mut self, slot: crate::ir::Slot) {
        match slot_local(slot) {
            Some(l) => {
                self.s().local_set(l);
                self.seen.flags_used = true;
                self.seen.flags_written = true;
                self.rw.entry(self.cur_pc).or_default().1 |= FLAGS;
                self.flags_written();
            }
            None => self.set(slot.0),
        }
    }

    fn ir_div(
        &mut self,
        d: crate::ir::V,
        wide: bool,
        signed: bool,
        x: crate::ir::V,
        y: crate::ir::V,
    ) {
        // y == 0 gives 0; signed MIN / -1 gives MIN (Wasm would trap on both).
        let ty = if wide { ValType::I64 } else { ValType::I32 };
        let zero = |s: &mut InstructionSink<'_>| {
            if wide {
                s.i64_const(0);
            } else {
                s.i32_const(0);
            }
        };
        self.vget(y);
        if wide {
            self.s().i64_eqz();
        } else {
            self.s().i32_eqz();
        }
        self.s().if_(BlockType::Result(ty));
        zero(&mut self.s());
        self.s().else_();
        if signed {
            self.vget(y);
            if wide {
                self.s().i64_const(-1).i64_eq();
            } else {
                self.s().i32_const(-1).i32_eq();
            }
            self.s().if_(BlockType::Result(ty));
            zero(&mut self.s());
            self.vget(x);
            if wide {
                self.s().i64_sub();
            } else {
                self.s().i32_sub();
            }
            self.s().else_();
            self.vget(x);
            self.vget(y);
            if wide {
                self.s().i64_div_s();
            } else {
                self.s().i32_div_s();
            }
            self.s().end();
        } else {
            self.vget(x);
            self.vget(y);
            if wide {
                self.s().i64_div_u();
            } else {
                self.s().i32_div_u();
            }
        }
        self.s().end();
        self.vset(d);
    }

    fn ir_mem(&mut self, m: &crate::ir::Mem) {
        use crate::opt::Merge;
        let check = match m.check {
            crate::ir::Check::None => Check::None,
            crate::ir::Check::Aligned16(_) => Check::SpBase,
            crate::ir::Check::Aligned(n) => Check::Aligned(n),
        };
        // The host address is `L_H` inside the fast path.
        self.vlocal.insert(m.host.0, L_H);
        let role = self.merge.get(&(self.cur_k as u32)).copied();
        if let Some(Merge::Head {
            id,
            lo,
            span,
            write,
            off,
        }) = role
        {
            // One check for the group (merged checks): `span` bytes from base + `lo`.
            self.vget(m.addr);
            self.s().i64_const(lo - off).i64_add().local_set(L_A);
            let table = if write { TLB_WRITE } else { TLB_READ };
            self.tlb_hit(table, span);
            if matches!(check, Check::SpBase) {
                self.get(wasmshim_aarch64::SP);
                self.s().i64_const(15).i64_and().i64_eqz().i32_and();
            }
            let ok = self.tmp_local(0);
            let hg = self.tmp_local(0);
            self.s().local_tee(ok).if_(BlockType::Empty);
            self.host_addr(table);
            self.s().local_get(L_H).local_set(hg).end();
            self.groups.insert(id, (ok, hg, lo));
        }
        let fast = m.fast.clone();
        if let Some(Merge::Head { id, off, .. } | Merge::Member { id, off }) = role
            && let Some(&(ok, hg, lo)) = self.groups.get(&id)
        {
            // In the group's page: the access is the fast path at a known offset.
            self.s().local_get(ok);
            self.if_hint(true);
            self.s()
                .local_get(hg)
                .i32_const((off - lo) as i32)
                .i32_add()
                .local_set(L_H);
            self.ir_ops(&fast);
            self.else_();
            self.ir_guarded(m, check, &fast);
            // The group's locals are redefined after the slow path's calls (the rest of the
            // group takes its own checks), so neither lives across a call.
            self.s()
                .i32_const(0)
                .local_set(ok)
                .i32_const(0)
                .local_set(hg);
            self.end();
            return;
        }
        self.ir_guarded(m, check, &fast);
    }

    /// A memory access through its own TLB check.
    fn ir_guarded(&mut self, m: &crate::ir::Mem, check: Check, fast: &[crate::ir::Op]) {
        self.vget(m.addr);
        self.s().local_set(L_A);
        let table = if m.write { TLB_WRITE } else { TLB_READ };
        self.guarded(
            m.k,
            self.page.wrapping_add(m.pc),
            table,
            m.size,
            check,
            m.reads & ALL_REGS,
            m.writes & ALL_REGS,
            |e| e.ir_ops(fast),
        );
    }

    /// A Wasm local of type `ty` (0: i32, 1: i64, 2: v128) for the rest of the block.
    fn tmp_local(&mut self, ty: usize) -> u32 {
        let l = match self.vfree[ty].pop() {
            Some(l) => l,
            None => {
                self.vextra.push(ty as u8);
                L_V0 + self.vextra.len() as u32
            }
        };
        self.vused.push((l, ty as u8));
        l
    }

    /// The condition (i32) is on the stack: branch to `taken` or `fallthrough`.
    fn cond_jump(&mut self, taken: u64, fallthrough: u64) {
        // Both ways leave the block: write back once, before the branch (the condition stays
        // on the stack), except what a way may carry to its target.
        let keep = [taken, fallthrough]
            .iter()
            .filter_map(|&t| self.carry_set(t))
            .fold(0, |m, (_, s)| m | s);
        self.flush(self.dirty & !keep);
        self.if_();
        self.jump(taken);
        self.end();
        self.jump(fallthrough);
    }

    /// Calls `jit_step` for the instruction at `pc` (index `k`): stores the registers it may
    /// read or write, reloads those it may write (and the flags if it may write them), then
    /// leaves on a nonzero status. Storing the outputs too keeps memory and locals equal on
    /// every outcome: a fault changes nothing, a retired instruction updated memory.
    fn call_step(&mut self, k: u32, pc: u64, reads: u64, writes: u64, flags: (bool, bool)) {
        if self.block_regs() {
            self.regs_before_call();
        } else {
            self.store_regs(reads | writes);
        }
        if flags.0 || flags.1 {
            // The interpreter sees (and, on a partial update, keeps) the current flags.
            self.store_flags();
        }
        self.pc(pc);
        self.call(Callee::Step);
        self.s().local_set(L_ST);
        self.reload_regs(writes);
        if flags.1 && self.block_regs() {
            // The interpreter changed them: the next reader loads them from the CPU.
            self.cached &= !FLAGS;
            self.dirty &= !FLAGS;
            self.kind = None;
        } else if flags.1 {
            self.seen.flags_used = true;
            self.s()
                .local_get(P_CPU)
                .i64_load32_u(mem(layout::NZCV as u64, 2))
                .local_set(L_FR)
                .i32_const(kind::PACKED)
                .local_set(L_FK);
            self.kind = Some(kind::PACKED);
        }
        self.s().local_get(L_ST);
        self.if_hint(false);
        self.exit_stored(i64::from(self.len - k));
        self.end();
    }

    fn store_flags(&mut self) {
        if self.block_regs() {
            // The CPU holds the flags unless this block changed them.
            self.flush(FLAGS);
            return;
        }
        self.seen.flags_in = true;
        self.seen.flags_used = true;
        store_lazy_flags(&mut self.s());
    }

    // ---- memory ----

    /// Pushes the TLB hit test (i32) for `size` bytes at `L_A`, leaving the entry in `L_E`.
    fn tlb_hit(&mut self, table: u64, size: u64) {
        self.s()
            .local_get(P_TLB)
            .local_get(L_A)
            .i64_const(12 - 4)
            .i64_shr_u()
            .i32_wrap_i64()
            .i32_const(((TLB_ENTRIES - 1) << 4) as i32)
            .i32_and()
            .i32_add()
            .local_tee(L_E)
            .i64_load(mem(table, 3))
            .local_get(L_A);
        if size > 1 {
            self.s().i64_const(size as i64 - 1).i64_add();
        }
        self.s().i64_const(!0xfff).i64_and().i64_eq();
    }

    fn host_addr(&mut self, table: u64) {
        self.s()
            .local_get(L_A)
            .i32_wrap_i64()
            .local_get(L_E)
            .i32_load(mem(table + TLB_ADDEND, 2))
            .i32_add()
            .local_set(L_H);
    }

    /// The fast-path/slow-path frame around one memory instruction.
    #[allow(clippy::too_many_arguments)]
    fn guarded(
        &mut self,
        k: u32,
        pc: u64,
        table: u64,
        size: u64,
        check: Check,
        inputs: u64,
        outputs: u64,
        fast: impl Fn(&mut Self),
    ) {
        self.tlb_hit(table, size);
        match check {
            Check::None => {}
            Check::SpBase => {
                // A64: an SP base must be 16-byte aligned (checked on the base, before offsets).
                self.get(wasmshim_aarch64::SP);
                self.s().i64_const(15).i64_and().i64_eqz().i32_and();
            }
            Check::Aligned(n) => {
                self.s()
                    .local_get(L_A)
                    .i64_const(n as i64 - 1)
                    .i64_and()
                    .i64_eqz()
                    .i32_and();
            }
        }
        if self.opts.compact_miss && !(self.opts.mem_exits && self.in_loop()) {
            // Compact miss path: the access exists once.
            //   block $go { block $miss { hit? else br $miss; host address; br $go }
            //     check? fill(addr) -> L_H, br_if $go; side exit (EXIT_MEM) }
            //   the access
            // No register is flushed or reloaded around the fill (it runs no instruction);
            // a failed fill (fault, page crossing, store into code, an SP check) leaves the
            // region with nothing of the instruction run, like a loop's side exit.
            let state = (self.cached, self.dirty);
            let hit = self.tmp_local(0);
            self.s().local_set(hit);
            self.flush_pending();
            self.s().block(BlockType::Empty);
            self.labels.push(Label::Mem);
            self.s().block(BlockType::Empty);
            self.labels.push(Label::Mem);
            self.s().local_get(hit).i32_eqz().br_if(0);
            self.host_addr(table);
            self.s().br(1);
            self.s().end();
            self.labels.pop();
            self.check(check);
            self.s().if_(BlockType::Result(ValType::I32)).local_get(L_A);
            self.s()
                .i32_const(i32::from(table == TLB_WRITE))
                .i32_const(size as i32);
            self.call(Callee::Fill);
            self.s()
                .local_tee(L_H)
                .i32_const(0)
                .i32_ne()
                .else_()
                .i32_const(0)
                .end()
                .br_if(0);
            let back = i64::from(self.len - k);
            self.s()
                .local_get(L_LEFT)
                .i64_const(back)
                .i64_add()
                .local_set(L_LEFT);
            self.exit_to(EXIT_MEM, pc);
            self.s().end();
            self.labels.pop();
            // Only the joining paths continue: the cache is the hit path's.
            (self.cached, self.dirty) = state;
            self.dead = false;
            let _ = (inputs, outputs);
            fast(self);
            return;
        }
        // Hit: the access, inline. Miss (but the check passed): ask the engine to fill the TLB,
        // which gives the host address or 0 (the access needs the interpreter). The hit path
        // stays a straight line; the access code is repeated in the cold arm.
        self.if_mem_hint();
        self.host_addr(table);
        fast(self);
        self.else_();
        if self.opts.mem_exits && self.in_loop() {
            // A side exit: nothing of this instruction ran; give back its budget
            // and the rest of the block's, and leave with its PC. The engine executes it in
            // the interpreter (filling the TLB) and goes on. No call on the loop's path.
            // Only inside loops: elsewhere a miss is cheaper as a call than as an exit (an exit
            // also interprets the rest of the block and re-enters; CPython misses often).
            let back = i64::from(self.len - k);
            self.s()
                .local_get(L_LEFT)
                .i64_const(back)
                .i64_add()
                .local_set(L_LEFT);
            self.exit_to(EXIT_MEM, pc);
            self.end();
            return;
        }
        self.regs_before_call();
        self.check(check);
        self.s().if_(BlockType::Result(ValType::I32)).local_get(L_A);
        self.s()
            .i32_const(i32::from(table == TLB_WRITE))
            .i32_const(size as i32);
        self.call(Callee::Fill);
        self.s()
            .local_tee(L_H)
            .i32_const(0)
            .i32_ne()
            .else_()
            .i32_const(0)
            .end();
        self.if_mem_hint();
        fast(self);
        self.else_();
        self.call_step(k, pc, inputs, outputs, (false, false));
        self.end();
        self.end();
    }

    /// Pushes the extra fast-path condition of `check` (i32 0/1).
    fn check(&mut self, check: Check) {
        match check {
            Check::None => {
                self.s().i32_const(1);
            }
            Check::SpBase => {
                self.get(wasmshim_aarch64::SP);
                self.s().i64_const(15).i64_and().i64_eqz();
            }
            Check::Aligned(n) => {
                self.s()
                    .local_get(L_A)
                    .i64_const(n as i64 - 1)
                    .i64_and()
                    .i64_eqz();
            }
        }
    }
}

/// The chain: on `EXIT_JUMP`, if the chain table has a current entry for the target PC, set
/// up the context and tail-call it. Returns the position of the padded type index.
fn chain(code: &mut Vec<u8>, entry_type: u32) -> usize {
    let entry = |s: &mut InstructionSink<'_>| {
        s.local_get(P_CTX)
            .i32_load(mem(CTX_CHAIN, 2))
            .local_get(L_PC)
            .i64_const(2)
            .i64_shr_u()
            .i32_wrap_i64()
            .i32_const((CHAIN_ENTRIES - 1) as i32)
            .i32_and()
            .i32_const((CHAIN_WAYS * CHAIN_ENTRY_SIZE) as i32)
            .i32_mul()
            .i32_add()
            .local_set(L_E);
    };
    // Does the entry at `L_E + off` hold the target, for the current epoch? (i32)
    let hit = |s: &mut InstructionSink<'_>, off: u64| {
        s.local_get(L_E)
            .i64_load(mem(off, 3))
            .local_get(L_PC)
            .i64_eq()
            .local_get(L_E)
            .i64_load(mem(off + 16, 3))
            .local_get(P_CTX)
            .i64_load(mem(CTX_EPOCH, 3))
            .i64_eq()
            .i32_and();
    };
    debug_assert_eq!(CHAIN_WAYS, 2);
    let mut s = InstructionSink::new(code);
    s.local_get(L_ST)
        .i32_const(EXIT_JUMP)
        .i32_eq()
        .if_(BlockType::Empty);
    entry(&mut s);
    // Way 0 (the most recent), else way 1: `L_E` becomes the matching entry.
    hit(&mut s, 0);
    s.local_set(L_W0);
    s.local_get(L_E);
    s.local_get(L_E)
        .i32_const(CHAIN_ENTRY_SIZE as i32)
        .i32_add();
    s.local_get(L_W0).select().local_set(L_E);
    hit(&mut s, 0);
    s.if_(BlockType::Empty)
        .local_get(P_CTX)
        .local_get(L_E)
        .i32_load(mem(12, 2))
        .i32_store(mem(CTX_ENTRY, 2))
        .local_get(P_CTX)
        .local_get(L_PC)
        .i64_const(!0xfff)
        .i64_and()
        .i64_store(mem(CTX_BASE, 3))
        .local_get(P_CPU)
        .local_get(P_TLB)
        .local_get(P_CTX)
        .local_get(L_E)
        .i32_load(mem(8, 2));
    // return_call_indirect (type entry) (table 0), the type index padded for relocation.
    code.push(0x13);
    let at = code.len();
    push_padded_leb(code, entry_type);
    code.push(0x00);
    InstructionSink::new(code).end().end();
    at
}

/// Loads the CPU's lazy flags into the flag locals; `FR` = packed NZCV when the kind is NONE.
fn load_lazy_flags(s: &mut InstructionSink<'_>) {
    s.local_get(P_CPU)
        .i32_load(mem(layout::FLAGS_KIND as u64, 2))
        .local_set(L_FK)
        .local_get(P_CPU)
        .i64_load(mem(layout::FLAGS_X as u64, 3))
        .local_set(L_FX)
        .local_get(P_CPU)
        .i64_load(mem(layout::FLAGS_Y as u64, 3))
        .local_set(L_FY)
        .local_get(P_CPU)
        .i64_load(mem(layout::FLAGS_R as u64, 3))
        .local_get(P_CPU)
        .i64_load32_u(mem(layout::NZCV as u64, 2))
        .local_get(L_FK)
        .select()
        .local_set(L_FR);
}

/// Stores the flag locals into the CPU's lazy flags (and NZCV itself when the kind is
/// PACKED): no evaluation; the interpreter evaluates lazily when it needs them.
fn store_lazy_flags(s: &mut InstructionSink<'_>) {
    s.local_get(P_CPU)
        .local_get(L_FK)
        .i32_store(mem(layout::FLAGS_KIND as u64, 2))
        .local_get(P_CPU)
        .local_get(L_FX)
        .i64_store(mem(layout::FLAGS_X as u64, 3))
        .local_get(P_CPU)
        .local_get(L_FY)
        .i64_store(mem(layout::FLAGS_Y as u64, 3))
        .local_get(P_CPU)
        .local_get(L_FR)
        .i64_store(mem(layout::FLAGS_R as u64, 3))
        .local_get(L_FK)
        .i32_eqz()
        .if_(BlockType::Empty)
        .local_get(P_CPU)
        .local_get(L_FR)
        .i32_wrap_i64()
        .i32_store(mem(layout::NZCV as u64, 2))
        .end();
}

/// The registers in `mask`, in order.
fn regs(mask: u64) -> impl Iterator<Item = Reg> {
    (0..NREGS as Reg).filter(move |&r| mask & bit(r) != 0)
}

fn push_padded_leb(code: &mut Vec<u8>, mut v: u32) {
    for i in 0..5 {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if i < 4 {
            b |= 0x80;
        }
        code.push(b);
    }
}

/// The body of `pack(kind, x, y, r) -> i32`: NZCV in the memory format from a lazy
/// description (see [`kind`]).
pub(crate) fn pack_body() -> Vec<u8> {
    // params: 0 kind (i32), 1 x, 2 y, 3 r (i64); locals: 4 n, 5 z, 6 c, 7 v (i32)
    let mut f = Function::new([(4, ValType::I32)]);
    let mut c = Vec::new();
    let mut s = InstructionSink::new(&mut c);
    // block per kind; br_table kind (the same function as `a64::pack_body`)
    for _ in 0..7 {
        s.block(BlockType::Empty);
    }
    s.local_get(0).br_table([0, 1, 2, 3, 4, 5, 6], 0).end();
    // PACKED
    s.local_get(3).i32_wrap_i64().return_().end();
    // SUB64, ADD64, SUB32, ADD32
    for (sub, wide) in [(true, true), (false, true), (true, false), (false, false)] {
        let get = |s: &mut InstructionSink<'_>, l: u32| {
            s.local_get(l);
            if !wide {
                s.i32_wrap_i64();
            }
        };
        // n = r < 0, z = r == 0
        get(&mut s, 3);
        if wide {
            s.i64_const(0).i64_lt_s();
        } else {
            s.i32_const(0).i32_lt_s();
        }
        s.local_set(4);
        get(&mut s, 3);
        if wide {
            s.i64_eqz();
        } else {
            s.i32_eqz();
        }
        s.local_set(5);
        // c
        if sub {
            get(&mut s, 1);
            get(&mut s, 2);
            if wide {
                s.i64_ge_u()
            } else {
                s.i32_ge_u()
            };
        } else {
            get(&mut s, 3);
            get(&mut s, 1);
            if wide {
                s.i64_lt_u()
            } else {
                s.i32_lt_u()
            };
        }
        s.local_set(6);
        // v
        get(&mut s, 1);
        get(&mut s, 2);
        if wide {
            s.i64_xor()
        } else {
            s.i32_xor()
        };
        if !sub {
            if wide {
                s.i64_const(-1).i64_xor();
            } else {
                s.i32_const(-1).i32_xor();
            }
        }
        get(&mut s, 1);
        get(&mut s, 3);
        if wide {
            s.i64_xor().i64_and().i64_const(0).i64_lt_s();
        } else {
            s.i32_xor().i32_and().i32_const(0).i32_lt_s();
        }
        s.local_set(7);
        pack_nzcv(&mut s);
        s.return_().end();
    }
    // LOGIC64, LOGIC32: n, z from r; c = v = 0
    for wide in [true, false] {
        s.local_get(3);
        if wide {
            s.i64_const(0).i64_lt_s();
        } else {
            s.i32_wrap_i64().i32_const(0).i32_lt_s();
        }
        s.local_get(3);
        if wide {
            s.i64_eqz();
        } else {
            s.i32_wrap_i64().i32_eqz();
        }
        s.i32_const(8).i32_shl().i32_or().return_();
        if wide {
            s.end();
        }
    }
    s.end();
    f.raw(c);
    f.into_raw_body()
}

fn pack_nzcv(s: &mut InstructionSink<'_>) {
    s.local_get(4)
        .local_get(5)
        .i32_const(8)
        .i32_shl()
        .i32_or()
        .local_get(6)
        .i32_const(16)
        .i32_shl()
        .i32_or()
        .local_get(7)
        .i32_const(24)
        .i32_shl()
        .i32_or();
}

/// The extra condition of a memory access's fast path.
#[derive(Clone, Copy)]
enum Check {
    None,
    /// The base register is SP, which must be 16-byte aligned.
    SpBase,
    /// The address must be aligned to this many bytes (exclusives, acquire/release).
    Aligned(u64),
}

impl Check {}

/// The flag local standing for an A64 flag slot (`None`: a register slot).
fn slot_local(slot: crate::ir::Slot) -> Option<u32> {
    use crate::a64::{FK, FR, FX, FY};
    match slot {
        s if s == FK => Some(L_FK),
        s if s == FX => Some(L_FX),
        s if s == FY => Some(L_FY),
        s if s == FR => Some(L_FR),
        _ => None,
    }
}

/// Whether `op` (or an op nested in it) writes the flags kind.
fn sets_kind(op: &crate::ir::Op) -> bool {
    use crate::ir::Op;
    match op {
        Op::Set(s, _) => *s == crate::a64::FK,
        Op::If { then, else_, .. } => then.iter().chain(else_.iter()).any(sets_kind),
        Op::Mem(m) => m.fast.iter().any(sets_kind),
        Op::Step { writes, .. } => writes & crate::a64::FLAG_SLOTS != 0,
        _ => false,
    }
}
