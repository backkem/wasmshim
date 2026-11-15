//! The AArch64 front half: A64 instructions lowered to the ISA-neutral IR
//! ([`crate::ir`]).
//!
//! Everything A64 is here and nowhere else in the translator's code shape: the register file
//! as slots, NZCV as **lazy flags** (four slots: a kind and the operands and result of the
//! operation that set them, evaluated only where a condition reads them, see [`kind`]), the
//! addressing modes, the exclusive monitor, the system registers this translator knows, and
//! which instructions the reference interpreter executes instead ([`Op::Step`]).
//!
//! The lowering is written like stack-machine Wasm: operands are pushed on the
//! [`Builder`]'s stack, operations pop them, and temporaries are numbered locals.

use alloc::vec::Vec;

use wasm_encoder::{BlockType, MemArg, ValType};
use wasmshim_aarch64::simd::SimdInst;
use wasmshim_aarch64::{
    AddrMode, BitfieldOp, BranchRegOp, Cond, CselOp, Dp1Op, Dp2Op, Dp3Op, ExclusiveOp, Extend,
    Inst, LogicOp, MemOp, MovOp, Reg, Shift, SysReg, ZR, layout,
};

use crate::Block;
use crate::abi::*;
use crate::ir::build::Builder;
use crate::ir::{Check, IrBlock, Mem, PureOps, Slot, Slots, Term, Ty, slot_bit};

mod fma;
mod fp;
mod neon;
mod simd;

// The locals the lowering code names: the builder maps the guest-state base and the flag
// locals; the others are temporaries.
const P_CPU: u32 = 0;
const L_A: u32 = 6; // i64: effective address
const L_PC: u32 = 7; // i64 temp
const L_H: u32 = 9; // i32: host address (in a memory access's fast path)
const L_T0: u32 = 11; // i64 temps
const L_T1: u32 = 12;
const L_W0: u32 = 13; // i32 temps
const L_FK: u32 = 16; // the flag slots
const L_FX: u32 = 17;
const L_FY: u32 = 18;
const L_FR: u32 = 19;
const L_V0: u32 = 53; // v128 temp

/// The slots: X0..X30 are 0..30, SP is 32 (31 is the zero register, never a slot), then the
/// lazy flags.
pub const SP: Slot = Slot(wasmshim_aarch64::SP);
pub const FK: Slot = Slot(33);
pub const FX: Slot = Slot(34);
pub const FY: Slot = Slot(35);
pub const FR: Slot = Slot(36);
/// The flag slots (loaded and stored together).
pub const FLAG_SLOTS: Slots = slot_bit(FK) | slot_bit(FX) | slot_bit(FY) | slot_bit(FR);

/// Helper 0: `pack(kind, x, y, r) -> nzcv` (the module's internal function).
pub const HELPER_PACK: u8 = 0;

const NREGS: usize = 33;
const ALL_REGS: u64 = ((1 << NREGS) - 1) & !(1 << ZR);

/// Flag kinds: how `FX`, `FY`, `FR` describe NZCV.
pub(crate) mod kind {
    use wasmshim_aarch64::lazy_flags as l;
    /// `FR` holds NZCV packed like memory: `n | z << 8 | c << 16 | v << 24` (the CPU's
    /// `lazy_flags::NONE`).
    pub const PACKED: i32 = l::NONE as i32;
    pub const SUB64: i32 = l::SUB64 as i32;
    pub const ADD64: i32 = l::ADD64 as i32;
    pub const SUB32: i32 = l::SUB32 as i32;
    pub const ADD32: i32 = l::ADD32 as i32;
    pub const LOGIC64: i32 = l::LOGIC64 as i32;
    pub const LOGIC32: i32 = l::LOGIC32 as i32;
}

/// Operand width.
#[derive(Clone, Copy, PartialEq, Eq)]
enum W {
    X,
    W,
}

impl W {
    fn of(sf: bool) -> Self {
        if sf { Self::X } else { Self::W }
    }
}

const fn bit(r: Reg) -> u64 {
    if r == ZR { 0 } else { 1 << r }
}

fn mem(offset: u64, align: u32) -> MemArg {
    MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

/// The slots of a set of registers (bits by register number, as [`Slots`]).
fn reg_slots(mask: u64) -> Slots {
    mask & ALL_REGS
}

/// Lowers `block` (in the page at `page`) to IR.
pub(crate) fn lower(block: &Block, page: u64, opts: &crate::Options) -> IrBlock {
    let mut a = A64 {
        b: Builder::new(),
        kind: None,
        len: block.insts.len() as u32,
        page,
        term: None,
        fp: opts.scalar_fp,
        vector: opts.neon,
    };
    a.b.map_local(P_CPU, None);
    a.b.map_local(L_FK, Some((FK, Ty::I32)));
    a.b.map_local(L_FX, Some((FX, Ty::I64)));
    a.b.map_local(L_FY, Some((FY, Ty::I64)));
    a.b.map_local(L_FR, Some((FR, Ty::I64)));
    let mut ops = Vec::new();
    for (k, inst) in block.insts.iter().enumerate() {
        let pc = block.pc + 4 * k as u64;
        let ended = a.inst(k as u32, pc, inst);
        ops.push(a.b.take());
        if ended || a.b.ended {
            break;
        }
    }
    let term = match a.term.take() {
        Some(t) => t,
        None if a.b.ended => Term::Left,
        None => Term::Jump(block.end().wrapping_sub(page)),
    };
    IrBlock {
        ops,
        term,
        types: core::mem::take(&mut a.b.types),
    }
}

/// The lowering state of one block.
struct A64 {
    b: Builder,
    /// The flags kind if known at this point of the block.
    kind: Option<i32>,
    /// Instructions in the block.
    len: u32,
    page: u64,
    /// The block's end, once an instruction ended it.
    term: Option<Term>,
    /// `Options::scalar_fp`, `Options::neon`.
    fp: bool,
    vector: bool,
}

impl A64 {
    fn s(&mut self) -> &mut Builder {
        &mut self.b
    }

    /// Pushes the runtime address of `pc`.
    fn pc(&mut self, pc: u64) {
        let off = pc.wrapping_sub(self.page) as i64;
        self.b.pc(off);
    }

    // ---- registers ----

    fn get(&mut self, r: Reg) {
        if r == ZR {
            self.b.i64_const(0);
        } else {
            self.b.get(Slot(r), Ty::I64);
        }
    }

    fn get_w(&mut self, w: W, r: Reg) {
        if r == ZR {
            self.konst(w, 0);
            return;
        }
        self.get(r);
        if w == W::W {
            self.b.i32_wrap_i64();
        }
    }

    fn set(&mut self, r: Reg) {
        if r == ZR {
            self.b.drop();
        } else {
            self.b.set(Slot(r));
        }
    }

    fn set_w(&mut self, w: W, r: Reg) {
        if w == W::W {
            self.b.i64_extend_i32_u();
        }
        self.set(r);
    }

    // ---- control ----

    /// Ends the block with a branch to `target`.
    fn jump(&mut self, target: u64) {
        self.term = Some(Term::Jump(target.wrapping_sub(self.page)));
    }

    /// The condition (i32) is on the stack: the block ends with a branch to `taken` or
    /// `fallthrough`.
    fn cond_jump(&mut self, taken: u64, fallthrough: u64) {
        let c = self.b.pop();
        self.term = Some(Term::Branch(
            c,
            taken.wrapping_sub(self.page),
            fallthrough.wrapping_sub(self.page),
        ));
    }

    /// The reference interpreter executes the instruction at `pc` (index `k`); it reads the
    /// registers in `reads` and may write those in `writes`, and the flags (`flags`: read,
    /// written).
    fn call_step(&mut self, k: u32, pc: u64, reads: u64, writes: u64, flags: (bool, bool)) {
        let f = |b: bool| if b { FLAG_SLOTS } else { 0 };
        self.b.step(
            pc.wrapping_sub(self.page),
            k,
            reg_slots(reads) | f(flags.0 || flags.1),
            reg_slots(writes) | f(flags.1),
        );
        if flags.1 {
            self.kind = None;
        }
    }

    /// Executes `inst` in the interpreter (`jit_step`).
    fn step(&mut self, k: u32, pc: u64, inst: &Inst) {
        let fx = effects(inst);
        self.call_step(
            k,
            pc,
            fx.gpr_reads,
            fx.gpr_writes,
            (fx.reads_flags, fx.writes_flags),
        );
    }

    /// A guest memory access: `fast` performs it given the host address (`L_H`); the address
    /// is in `L_A`. On a miss the interpreter executes the instruction (which reads `inputs`
    /// and may write `outputs`).
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
        self.b.local_get(L_A);
        let addr = self.b.pop();
        let host = self.b.value(Ty::I32);
        let check = match check {
            Check::Aligned16(_) => Check::Aligned16(SP),
            c => c,
        };
        self.b.begin_fast(Mem {
            pc: pc.wrapping_sub(self.page),
            k,
            addr,
            write: table == TLB_WRITE,
            size,
            check,
            reads: reg_slots(inputs),
            writes: reg_slots(outputs),
            host,
            fast: Vec::new(),
        });
        self.b.push(host).local_set(L_H);
        fast(self);
        self.b.end_fast();
    }
}

/// Check for a base register (the SP alignment rule).
fn check_sp(rn: Reg) -> Check {
    if rn == wasmshim_aarch64::SP {
        Check::Aligned16(SP)
    } else {
        Check::None
    }
}

impl A64 {
    fn flags_set(&mut self, k: i32) {
        self.s().i32_const(k).local_set(L_FK);
        self.kind = Some(k);
    }

    /// Makes `FR` hold packed NZCV (`FK = PACKED`); returns nothing.
    fn materialize(&mut self) {
        if self.kind == Some(kind::PACKED) {
            return;
        }
        if self.kind.is_some() {
            // Not inside `cond`'s run-time kind dispatch (an arm the cache does not track).
        }
        self.s()
            .local_get(L_FK)
            .local_get(L_FX)
            .local_get(L_FY)
            .local_get(L_FR);
        self.b.helper(HELPER_PACK, 4, Ty::I32);
        self.s()
            .i64_extend_i32_u()
            .local_set(L_FR)
            .i32_const(kind::PACKED)
            .local_set(L_FK);
        self.kind = Some(kind::PACKED);
    }

    /// Pushes bit `shift` of packed NZCV in `FR` (i32 0/1).
    fn packed_bit(&mut self, shift: i64) {
        self.s().local_get(L_FR);
        if shift != 0 {
            self.s().i64_const(shift).i64_shr_u();
        }
        self.s().i64_const(1).i64_and().i32_wrap_i64();
    }

    /// Pushes `FX`, `FY` or `FR` at the width of the current kind.
    fn flag_operand(&mut self, local: u32, w: W) {
        self.s().local_get(local);
        if w == W::W {
            self.s().i32_wrap_i64();
        }
    }

    /// Pushes the condition (i32 0/1).
    fn cond(&mut self, cond: Cond) {
        if self.kind.is_none() && cond.0 >> 1 != 7 {
            // The flags were set in another block: test the common kinds at run time and
            // evaluate those directly; only the others go through `pack`.
            self.s()
                .local_get(L_FK)
                .i32_const(kind::SUB64)
                .i32_eq()
                .if_block(BlockType::Result(ValType::I32));
            self.kind = Some(kind::SUB64);
            self.cond_known(cond);
            self.s()
                .else_()
                .local_get(L_FK)
                .i32_const(kind::SUB32)
                .i32_eq()
                .if_block(BlockType::Result(ValType::I32));
            self.kind = Some(kind::SUB32);
            self.cond_known(cond);
            self.s().else_();
            self.kind = None;
            self.cond_known(cond);
            self.s().end().end();
            self.kind = None;
            return;
        }
        self.cond_known(cond);
    }

    /// [`Self::cond`] given what is known about the flags kind at translation time.
    fn cond_known(&mut self, cond: Cond) {
        if cond.0 >> 1 != 7 {
            // A read of the flag locals (for the block's liveness).
        }
        let base = cond.0 >> 1;
        let invert = cond.0 & 1 == 1 && cond.0 != 0b1111;
        if base == 7 {
            self.s().i32_const(1);
            return;
        }
        let k = self.kind;
        let direct = match k {
            Some(kind::SUB64) | Some(kind::SUB32) => {
                let w = if k == Some(kind::SUB64) { W::X } else { W::W };
                let cmp = |e: &mut Self, f: &dyn Fn(&mut Builder)| {
                    e.flag_operand(L_FX, w);
                    e.flag_operand(L_FY, w);
                    let s = e.s();
                    f(s);
                };
                match (base, w) {
                    // EQ: x == y; CS: x >= y (unsigned); HI: x > y (unsigned);
                    // GE: x >= y (signed); GT: x > y (signed).
                    (0, W::X) => cmp(self, &|s| {
                        s.i64_eq();
                    }),
                    (0, W::W) => cmp(self, &|s| {
                        s.i32_eq();
                    }),
                    (1, W::X) => cmp(self, &|s| {
                        s.i64_ge_u();
                    }),
                    (1, W::W) => cmp(self, &|s| {
                        s.i32_ge_u();
                    }),
                    (4, W::X) => cmp(self, &|s| {
                        s.i64_gt_u();
                    }),
                    (4, W::W) => cmp(self, &|s| {
                        s.i32_gt_u();
                    }),
                    (5, W::X) => cmp(self, &|s| {
                        s.i64_ge_s();
                    }),
                    (5, W::W) => cmp(self, &|s| {
                        s.i32_ge_s();
                    }),
                    (6, W::X) => cmp(self, &|s| {
                        s.i64_gt_s();
                    }),
                    (6, W::W) => cmp(self, &|s| {
                        s.i32_gt_s();
                    }),
                    // MI: r < 0.
                    (2, _) => {
                        self.flag_operand(L_FR, w);
                        self.lt_zero(w);
                    }
                    // VS: ((x ^ y) & (x ^ r)) < 0.
                    _ => {
                        self.flag_operand(L_FX, w);
                        self.flag_operand(L_FY, w);
                        self.xor(w);
                        self.flag_operand(L_FX, w);
                        self.flag_operand(L_FR, w);
                        self.xor(w);
                        self.and(w);
                        self.lt_zero(w);
                    }
                }
                true
            }
            Some(kind::LOGIC64) | Some(kind::LOGIC32) => {
                let w = if k == Some(kind::LOGIC64) { W::X } else { W::W };
                match base {
                    0 => {
                        self.flag_operand(L_FR, w);
                        self.eqz(w);
                    }
                    1 | 3 | 4 => {
                        // C = V = 0: CS, VS and HI never hold.
                        self.s().i32_const(0);
                    }
                    2 => {
                        self.flag_operand(L_FR, w);
                        self.lt_zero(w);
                    }
                    5 => {
                        // N == V with V = 0: r >= 0.
                        self.flag_operand(L_FR, w);
                        self.lt_zero(w);
                        self.s().i32_eqz();
                    }
                    _ => {
                        // GT: !Z && N == V: r > 0 (signed).
                        self.flag_operand(L_FR, w);
                        self.konst(w, 0);
                        match w {
                            W::X => self.s().i64_gt_s(),
                            W::W => self.s().i32_gt_s(),
                        };
                    }
                }
                true
            }
            Some(kind::ADD64) | Some(kind::ADD32) if matches!(base, 0..=2) => {
                let w = if k == Some(kind::ADD64) { W::X } else { W::W };
                match base {
                    0 => {
                        self.flag_operand(L_FR, w);
                        self.eqz(w);
                    }
                    1 => {
                        // C: r < x (unsigned).
                        self.flag_operand(L_FR, w);
                        self.flag_operand(L_FX, w);
                        match w {
                            W::X => self.s().i64_lt_u(),
                            W::W => self.s().i32_lt_u(),
                        };
                    }
                    _ => {
                        self.flag_operand(L_FR, w);
                        self.lt_zero(w);
                    }
                }
                true
            }
            _ => false,
        };
        if !direct {
            self.materialize();
            let (n, z, c, v) = (0, 8, 16, 24);
            match base {
                0 => self.packed_bit(z),
                1 => self.packed_bit(c),
                2 => self.packed_bit(n),
                3 => self.packed_bit(v),
                4 => {
                    self.packed_bit(c);
                    self.packed_bit(z);
                    self.s().i32_eqz().i32_and();
                }
                5 => {
                    self.packed_bit(n);
                    self.packed_bit(v);
                    self.s().i32_eq();
                }
                _ => {
                    self.packed_bit(n);
                    self.packed_bit(v);
                    self.s().i32_eq();
                    self.packed_bit(z);
                    self.s().i32_eqz().i32_and();
                }
            }
        }
        if invert {
            self.s().i32_eqz();
        }
    }

    /// Pushes the carry flag (i32 0/1).
    fn carry(&mut self) {
        match self.kind {
            Some(kind::SUB64) => {
                self.s().local_get(L_FX).local_get(L_FY).i64_ge_u();
            }
            Some(kind::SUB32) => {
                self.flag_operand(L_FX, W::W);
                self.flag_operand(L_FY, W::W);
                self.s().i32_ge_u();
            }
            Some(kind::ADD64) => {
                self.s().local_get(L_FR).local_get(L_FX).i64_lt_u();
            }
            Some(kind::ADD32) => {
                self.flag_operand(L_FR, W::W);
                self.flag_operand(L_FX, W::W);
                self.s().i32_lt_u();
            }
            Some(kind::LOGIC64) | Some(kind::LOGIC32) => {
                self.s().i32_const(0);
            }
            _ => {
                self.materialize();
                self.packed_bit(16);
            }
        }
    }

    // ---- width-generic arithmetic ----

    fn konst(&mut self, w: W, v: u64) {
        match w {
            W::X => self.s().i64_const(v as i64),
            W::W => self.s().i32_const(v as u32 as i32),
        };
    }
    fn add(&mut self, w: W) {
        match w {
            W::X => self.s().i64_add(),
            W::W => self.s().i32_add(),
        };
    }
    fn sub(&mut self, w: W) {
        match w {
            W::X => self.s().i64_sub(),
            W::W => self.s().i32_sub(),
        };
    }
    fn and(&mut self, w: W) {
        match w {
            W::X => self.s().i64_and(),
            W::W => self.s().i32_and(),
        };
    }
    fn or(&mut self, w: W) {
        match w {
            W::X => self.s().i64_or(),
            W::W => self.s().i32_or(),
        };
    }
    fn xor(&mut self, w: W) {
        match w {
            W::X => self.s().i64_xor(),
            W::W => self.s().i32_xor(),
        };
    }
    fn not(&mut self, w: W) {
        self.konst(w, u64::MAX);
        self.xor(w);
    }
    fn shift_op(&mut self, w: W, shift: Shift) {
        match (w, shift) {
            (W::X, Shift::Lsl) => self.s().i64_shl(),
            (W::X, Shift::Lsr) => self.s().i64_shr_u(),
            (W::X, Shift::Asr) => self.s().i64_shr_s(),
            (W::X, Shift::Ror) => self.s().i64_rotr(),
            (W::W, Shift::Lsl) => self.s().i32_shl(),
            (W::W, Shift::Lsr) => self.s().i32_shr_u(),
            (W::W, Shift::Asr) => self.s().i32_shr_s(),
            (W::W, Shift::Ror) => self.s().i32_rotr(),
        };
    }
    fn lt_zero(&mut self, w: W) {
        match w {
            W::X => self.s().i64_const(0).i64_lt_s(),
            W::W => self.s().i32_const(0).i32_lt_s(),
        };
    }
    fn eqz(&mut self, w: W) {
        match w {
            W::X => self.s().i64_eqz(),
            W::W => self.s().i32_eqz(),
        };
    }
    /// Zero-extends a value of width `w` to i64 (no-op for X).
    fn widen(&mut self, w: W) {
        if w == W::W {
            self.s().i64_extend_i32_u();
        }
    }

    fn shifted(&mut self, w: W, rm: Reg, shift: Shift, amount: u8) {
        self.get_w(w, rm);
        if amount != 0 {
            self.konst(w, u64::from(amount));
            self.shift_op(w, shift);
        }
    }

    fn extended(&mut self, w: W, rm: Reg, extend: Extend, amount: u8) {
        self.get(rm);
        self.extend(extend, amount);
        if w == W::W {
            self.s().i32_wrap_i64();
        }
    }

    fn extend(&mut self, extend: Extend, amount: u8) {
        match extend {
            Extend::Uxtb => {
                self.s().i64_const(0xff).i64_and();
            }
            Extend::Uxth => {
                self.s().i64_const(0xffff).i64_and();
            }
            Extend::Uxtw => {
                self.s().i64_const(0xffff_ffff).i64_and();
            }
            Extend::Uxtx | Extend::Sxtx => {}
            Extend::Sxtb => {
                self.s().i64_extend8_s();
            }
            Extend::Sxth => {
                self.s().i64_extend16_s();
            }
            Extend::Sxtw => {
                self.s().i64_extend32_s();
            }
        }
        if amount != 0 {
            self.s().i64_const(i64::from(amount)).i64_shl();
        }
    }

    /// Add or subtract with x and y pushed (width `w`); result to `rd`.
    fn add_sub_tail(&mut self, w: W, sub: bool, set_flags: bool, rd: Reg) {
        if set_flags {
            // FY = y, FX = x, FR = x op y (zero-extended for W).
            self.widen(w);
            self.s().local_set(L_FY);
            self.widen(w);
            self.s().local_tee(L_FX).local_get(L_FY);
            if sub {
                self.s().i64_sub();
            } else {
                self.s().i64_add();
            }
            if w == W::W {
                self.s().i64_const(0xffff_ffff).i64_and();
            }
            self.s().local_set(L_FR);
            self.flags_set(match (w, sub) {
                (W::X, true) => kind::SUB64,
                (W::X, false) => kind::ADD64,
                (W::W, true) => kind::SUB32,
                (W::W, false) => kind::ADD32,
            });
            if rd != ZR {
                self.s().local_get(L_FR);
                self.set(rd);
            }
        } else {
            if sub {
                self.sub(w);
            } else {
                self.add(w);
            }
            self.set_w(w, rd);
        }
    }

    fn logic_tail(&mut self, w: W, op: LogicOp, rd: Reg) {
        match op {
            LogicOp::And | LogicOp::Ands => self.and(w),
            LogicOp::Orr => self.or(w),
            LogicOp::Eor => self.xor(w),
        }
        self.widen(w);
        if op == LogicOp::Ands {
            self.s().local_tee(L_FR);
            self.flags_set(if w == W::X {
                kind::LOGIC64
            } else {
                kind::LOGIC32
            });
        }
        self.set(rd);
    }

    // ---- blocks ----

    /// Entering block `i` through the dispatch loop or the function entry: the budget must
    /// cover the longest forward path from here, or none of it runs here (the engine finishes
    /// the budget, exactly, in the interpreter). Forward branches skip this: the check at the
    /// path's start already covered them. (Back edges to a loop header check at the branch.)
    fn inst(&mut self, k: u32, pc: u64, inst: &Inst) -> bool {
        let next = pc.wrapping_add(4);
        match *inst {
            Inst::Adr { rd, imm, page } => {
                if page {
                    self.pc(pc & !0xfff);
                    self.s().i64_const(imm).i64_add();
                } else {
                    self.pc(pc.wrapping_add(imm as u64));
                }
                self.set(rd);
            }
            Inst::AddSubImm {
                sf,
                sub,
                set_flags,
                rd,
                rn,
                imm,
            } => {
                let w = W::of(sf);
                self.get_w(w, rn);
                self.konst(w, imm);
                self.add_sub_tail(w, sub, set_flags, rd);
            }
            Inst::AddSubShifted {
                sf,
                sub,
                set_flags,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let w = W::of(sf);
                self.get_w(w, rn);
                self.shifted(w, rm, shift, amount);
                self.add_sub_tail(w, sub, set_flags, rd);
            }
            Inst::AddSubExtended {
                sf,
                sub,
                set_flags,
                rd,
                rn,
                rm,
                extend,
                amount,
            } => {
                let w = W::of(sf);
                self.get_w(w, rn);
                self.extended(w, rm, extend, amount);
                self.add_sub_tail(w, sub, set_flags, rd);
            }
            Inst::AddSubCarry {
                sf,
                sub,
                set_flags: false,
                rd,
                rn,
                rm,
            } => {
                let w = W::of(sf);
                self.get_w(w, rn);
                self.get_w(w, rm);
                if sub {
                    self.not(w);
                }
                self.add(w);
                self.carry();
                if w == W::X {
                    self.s().i64_extend_i32_u();
                }
                self.add(w);
                self.set_w(w, rd);
            }
            Inst::AddSubCarry {
                sf,
                sub,
                set_flags: true,
                rd,
                rn,
                rm,
            } => {
                // ADCS/SBCS: r = x + y' + c (y' = !y for SBCS); NZCV packed: N, Z from
                // r, C = r < x or (r == x and c), V = (x ^ r) & (y' ^ r) negative.
                const X: u32 = 120;
                const Y: u32 = 121;
                const C: u32 = 122;
                const R: u32 = 123;
                let w = W::of(sf);
                self.get_w(w, rn);
                self.s().local_set(X);
                self.get_w(w, rm);
                if sub {
                    self.not(w);
                }
                self.s().local_set(Y);
                self.carry();
                self.s().local_set(C);
                self.s().local_get(X).local_get(Y);
                self.add(w);
                self.s().local_get(C);
                if w == W::X {
                    self.s().i64_extend_i32_u();
                }
                self.add(w);
                self.s().local_set(R);
                let s = self.s();
                if w == W::X {
                    // N
                    s.local_get(R).i64_const(0).i64_lt_s().i64_extend_i32_u();
                    // Z
                    s.local_get(R)
                        .i64_eqz()
                        .i64_extend_i32_u()
                        .i64_const(8)
                        .i64_shl()
                        .i64_or();
                    // C
                    s.local_get(R)
                        .local_get(X)
                        .i64_lt_u()
                        .local_get(R)
                        .local_get(X)
                        .i64_eq()
                        .local_get(C)
                        .i32_and()
                        .i32_or()
                        .i64_extend_i32_u()
                        .i64_const(16)
                        .i64_shl()
                        .i64_or();
                    // V
                    s.local_get(X)
                        .local_get(R)
                        .i64_xor()
                        .local_get(Y)
                        .local_get(R)
                        .i64_xor()
                        .i64_and()
                        .i64_const(0)
                        .i64_lt_s()
                        .i64_extend_i32_u()
                        .i64_const(24)
                        .i64_shl()
                        .i64_or();
                } else {
                    s.local_get(R).i32_const(0).i32_lt_s().i64_extend_i32_u();
                    s.local_get(R)
                        .i32_eqz()
                        .i64_extend_i32_u()
                        .i64_const(8)
                        .i64_shl()
                        .i64_or();
                    s.local_get(R)
                        .local_get(X)
                        .i32_lt_u()
                        .local_get(R)
                        .local_get(X)
                        .i32_eq()
                        .local_get(C)
                        .i32_and()
                        .i32_or()
                        .i64_extend_i32_u()
                        .i64_const(16)
                        .i64_shl()
                        .i64_or();
                    s.local_get(X)
                        .local_get(R)
                        .i32_xor()
                        .local_get(Y)
                        .local_get(R)
                        .i32_xor()
                        .i32_and()
                        .i32_const(0)
                        .i32_lt_s()
                        .i64_extend_i32_u()
                        .i64_const(24)
                        .i64_shl()
                        .i64_or();
                }
                self.s()
                    .local_set(L_FR)
                    .i32_const(kind::PACKED)
                    .local_set(L_FK);
                self.kind = Some(kind::PACKED);
                self.s().local_get(R);
                self.set_w(w, rd);
            }
            Inst::LogicImm {
                sf,
                op,
                rd,
                rn,
                imm,
            } => {
                let w = W::of(sf);
                self.get_w(w, rn);
                self.konst(w, imm);
                self.logic_tail(w, op, rd);
            }
            Inst::LogicShifted {
                sf,
                op,
                invert,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let w = W::of(sf);
                self.get_w(w, rn);
                self.shifted(w, rm, shift, amount);
                if invert {
                    self.not(w);
                }
                self.logic_tail(w, op, rd);
            }
            Inst::MovWide {
                sf,
                op,
                rd,
                imm16,
                shift,
            } => {
                let imm = u64::from(imm16) << shift;
                let mask = if sf { u64::MAX } else { 0xffff_ffff };
                match op {
                    MovOp::N => {
                        self.s().i64_const((!imm & mask) as i64);
                    }
                    MovOp::Z => {
                        self.s().i64_const(imm as i64);
                    }
                    MovOp::K => {
                        self.get(rd);
                        self.s()
                            .i64_const(!(0xffff_u64 << shift) as i64)
                            .i64_and()
                            .i64_const(imm as i64)
                            .i64_or();
                        if !sf {
                            self.s().i64_const(0xffff_ffff).i64_and();
                        }
                    }
                }
                self.set(rd);
            }
            Inst::Bitfield {
                sf,
                op,
                rd,
                rn,
                immr,
                imms,
                wmask,
                tmask,
            } => self.bitfield(sf, op, rd, rn, immr, imms, wmask, tmask),
            Inst::Extract {
                sf,
                rd,
                rn,
                rm,
                lsb,
            } => {
                if sf {
                    self.get(rm);
                    if lsb != 0 {
                        self.s().i64_const(i64::from(lsb)).i64_shr_u();
                        self.get(rn);
                        self.s().i64_const(64 - i64::from(lsb)).i64_shl().i64_or();
                    }
                } else {
                    self.get(rn);
                    self.s().i64_const(32).i64_shl();
                    self.get(rm);
                    self.s()
                        .i64_const(0xffff_ffff)
                        .i64_and()
                        .i64_or()
                        .i64_const(i64::from(lsb))
                        .i64_shr_u()
                        .i64_const(0xffff_ffff)
                        .i64_and();
                }
                self.set(rd);
            }
            Inst::CondSelect {
                sf,
                op,
                rd,
                rn,
                rm,
                cond,
            } => {
                self.get(rn);
                self.get(rm);
                match op {
                    CselOp::Csel => {}
                    CselOp::Csinc => {
                        self.s().i64_const(1).i64_add();
                    }
                    CselOp::Csinv => {
                        self.s().i64_const(-1).i64_xor();
                    }
                    CselOp::Csneg => {
                        self.s()
                            .local_set(L_T0)
                            .i64_const(0)
                            .local_get(L_T0)
                            .i64_sub();
                    }
                }
                self.cond(cond);
                self.s().select();
                if !sf {
                    self.s().i64_const(0xffff_ffff).i64_and();
                }
                self.set(rd);
            }
            Inst::CondCompare {
                sf,
                sub,
                rn,
                rm,
                imm,
                nzcv,
                cond,
            } => {
                let w = W::of(sf);
                self.cond(cond);
                self.b.if_(None);
                self.get_w(w, rn);
                match imm {
                    Some(i) => self.konst(w, u64::from(i)),
                    None => self.get_w(w, rm),
                }
                self.add_sub_tail(w, sub, true, ZR);
                self.b.else_();
                let packed = u64::from((nzcv >> 3) & 1)
                    | u64::from((nzcv >> 2) & 1) << 8
                    | u64::from((nzcv >> 1) & 1) << 16
                    | u64::from(nzcv & 1) << 24;
                self.s()
                    .i64_const(packed as i64)
                    .local_set(L_FR)
                    .i32_const(kind::PACKED)
                    .local_set(L_FK);
                self.b.end();
                // The kind is only known at run time now.
                self.kind = None;
            }
            Inst::Dp1 {
                sf,
                op: op @ (Dp1Op::Clz | Dp1Op::Cls),
                rd,
                rn,
            } => {
                let w = W::of(sf);
                self.get_w(w, rn);
                if op == Dp1Op::Cls {
                    match w {
                        W::X => {
                            self.s()
                                .local_tee(L_T0)
                                .local_get(L_T0)
                                .i64_const(1)
                                .i64_shl()
                                .i64_xor()
                                .i64_const(1)
                                .i64_or();
                        }
                        W::W => {
                            self.s()
                                .local_tee(L_W0)
                                .local_get(L_W0)
                                .i32_const(1)
                                .i32_shl()
                                .i32_xor()
                                .i32_const(1)
                                .i32_or();
                        }
                    }
                }
                match w {
                    W::X => self.s().i64_clz(),
                    W::W => self.s().i32_clz(),
                };
                self.set_w(w, rd);
            }
            Inst::Dp2 { sf, op, rd, rn, rm } => self.dp2(sf, op, rd, rn, rm),
            Inst::Dp3 {
                sf,
                op,
                rd,
                rn,
                rm,
                ra,
            } if !matches!(op, Dp3Op::Smulh | Dp3Op::Umulh) => {
                self.get(ra);
                for r in [rn, rm] {
                    self.get(r);
                    match op {
                        Dp3Op::Smaddl | Dp3Op::Smsubl => {
                            self.s().i64_extend32_s();
                        }
                        Dp3Op::Umaddl | Dp3Op::Umsubl => {
                            self.s().i64_const(0xffff_ffff).i64_and();
                        }
                        _ => {}
                    }
                }
                self.s().i64_mul();
                match op {
                    Dp3Op::Madd | Dp3Op::Smaddl | Dp3Op::Umaddl => self.s().i64_add(),
                    _ => self.s().i64_sub(),
                };
                if !sf {
                    self.s().i64_const(0xffff_ffff).i64_and();
                }
                self.set(rd);
            }

            // ---- branches ----
            Inst::Branch { link, offset } => {
                if link {
                    self.pc(next);
                    self.set(30);
                }
                self.jump(pc.wrapping_add(offset as u64));
                return true;
            }
            Inst::BranchCond { cond, offset } => {
                self.cond(cond);
                self.cond_jump(pc.wrapping_add(offset as u64), next);
                return true;
            }
            Inst::CompareBranch {
                sf,
                nonzero,
                rt,
                offset,
            } => {
                self.get_w(W::of(sf), rt);
                self.eqz(W::of(sf));
                if nonzero {
                    self.s().i32_eqz();
                }
                self.cond_jump(pc.wrapping_add(offset as u64), next);
                return true;
            }
            Inst::TestBranch {
                nonzero,
                rt,
                bit,
                offset,
            } => {
                self.get(rt);
                self.s().i64_const(1 << bit).i64_and().i64_eqz();
                if nonzero {
                    self.s().i32_eqz();
                }
                self.cond_jump(pc.wrapping_add(offset as u64), next);
                return true;
            }
            Inst::BranchReg { op, rn } => {
                self.get(rn);
                self.s().local_set(L_A);
                if op == BranchRegOp::Blr {
                    self.pc(next);
                    self.set(30);
                }
                self.s().local_get(L_A);
                self.b.exit(EXIT_JUMP, 0);
                return true;
            }
            Inst::Svc { .. } => {
                // The SVC retired: give back the rest of the block.
                self.pc(next);
                self.b.exit(EXIT_SYSCALL, self.len - k - 1);
                return true;
            }
            Inst::Nop => {}
            Inst::Mrs {
                rt,
                reg: SysReg::TpidrEl0,
            } => {
                self.s()
                    .local_get(P_CPU)
                    .i64_load(mem(layout::TPIDR as u64, 3));
                self.set(rt);
            }
            Inst::Mrs { rt, reg } if sysreg_const(reg).is_some() => {
                self.s().i64_const(sysreg_const(reg).unwrap_or(0) as i64);
                self.set(rt);
            }
            Inst::Mrs {
                rt,
                reg: SysReg::CntvctEl0,
            } => {
                self.s()
                    .local_get(P_CPU)
                    .i64_load(mem(layout::COUNTER as u64, 3));
                self.set(rt);
            }
            Inst::DcZva { rt } => self.dc_zva(k, pc, rt),
            Inst::Msr {
                rt,
                reg: SysReg::TpidrEl0,
            } => {
                self.s().local_get(P_CPU);
                self.get(rt);
                self.s().i64_store(mem(layout::TPIDR as u64, 3));
            }

            // ---- loads and stores ----
            Inst::LoadStore {
                op,
                bytes,
                rt,
                rn,
                mode,
            } if gpr_or_simple_vector(op, bytes) => self.load_store(k, pc, op, bytes, rt, rn, mode),
            Inst::LoadLiteral {
                op: op @ (MemOp::Load | MemOp::LoadSigned64 | MemOp::LoadSigned32),
                bytes,
                rt,
                offset,
            } => self.load_literal(k, pc, op, bytes, rt, offset),
            Inst::LoadStorePair {
                load,
                vector,
                signed,
                bytes,
                rt,
                rt2,
                rn,
                mode,
            } => self.pair(k, pc, load, vector, signed, bytes, rt, rt2, rn, mode),
            Inst::Exclusive {
                op,
                bytes,
                pair: false,
                rs,
                rt,
                rn,
                ..
            } => self.exclusive(k, pc, op, bytes, rs, rt, rn),

            _ => {
                if !self.native_extra(k, pc, inst) {
                    self.step(k, pc, inst);
                }
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn bitfield(
        &mut self,
        sf: bool,
        op: BitfieldOp,
        rd: Reg,
        rn: Reg,
        immr: u8,
        imms: u8,
        wmask: u64,
        tmask: u64,
    ) {
        let mask = if sf { u64::MAX } else { 0xffff_ffff };
        // Common special cases first (LSL, LSR, ASR, UBFX, SBFX, UXT*, SXT*): no BFM, so the
        // destination's old value does not matter.
        if op != BitfieldOp::Bfm {
            // T0 = rotated & wmask & tmask; plus, for SBFM, the replicated sign bit outside.
            self.get(rn);
            if sf {
                if immr != 0 {
                    self.s().i64_const(i64::from(immr)).i64_rotr();
                }
            } else {
                self.s()
                    .i32_wrap_i64()
                    .i32_const(i32::from(immr))
                    .i32_rotr()
                    .i64_extend_i32_u();
            }
            self.s().i64_const((wmask & tmask) as i64).i64_and();
            if op == BitfieldOp::Sbfm {
                // top = sign(src bit imms) replicated, masked; result |= top & !tmask.
                self.get(rn);
                self.s()
                    .i64_const(63 - i64::from(imms))
                    .i64_shl()
                    .i64_const(63)
                    .i64_shr_s()
                    .i64_const((!tmask & mask) as i64)
                    .i64_and()
                    .i64_or();
            }
            self.set(rd);
            return;
        }
        // BFM: (dst & !wmask | rotated & wmask) under tmask, dst outside it.
        self.get(rn);
        if sf {
            self.s().i64_const(i64::from(immr)).i64_rotr();
        } else {
            self.s()
                .i32_wrap_i64()
                .i32_const(i32::from(immr))
                .i32_rotr()
                .i64_extend_i32_u();
        }
        self.s().i64_const(wmask as i64).i64_and();
        self.get(rd);
        self.s()
            .local_tee(L_T0)
            .i64_const(!wmask as i64)
            .i64_and()
            .i64_or()
            .i64_const(tmask as i64)
            .i64_and()
            .local_get(L_T0)
            .i64_const(!tmask as i64)
            .i64_and()
            .i64_or()
            .i64_const(mask as i64)
            .i64_and();
        self.set(rd);
    }

    fn dp2(&mut self, sf: bool, op: Dp2Op, rd: Reg, rn: Reg, rm: Reg) {
        let w = W::of(sf);
        match op {
            Dp2Op::Lslv | Dp2Op::Lsrv | Dp2Op::Asrv | Dp2Op::Rorv => {
                self.get_w(w, rn);
                self.get_w(w, rm);
                let shift = match op {
                    Dp2Op::Lslv => Shift::Lsl,
                    Dp2Op::Lsrv => Shift::Lsr,
                    Dp2Op::Asrv => Shift::Asr,
                    _ => Shift::Ror,
                };
                // Wasm masks the count by the width, as A64 does.
                self.shift_op(w, shift);
                self.set_w(w, rd);
            }
            Dp2Op::Udiv | Dp2Op::Sdiv => {
                // A64: y == 0 gives 0; Sdiv(MIN, -1) gives MIN (the IR's total division).
                self.get_w(w, rn);
                self.get_w(w, rm);
                self.b.div(w == W::X, op == Dp2Op::Sdiv);
                self.set_w(w, rd);
            }
        }
    }

    // ---- memory ----

    /// Computes the address of `mode` into `L_A`; returns the post-index (`Some(0)`: pre).
    fn address(&mut self, rn: Reg, mode: AddrMode) -> Option<i64> {
        self.get(rn);
        match mode {
            AddrMode::Offset(imm) | AddrMode::PreIndex(imm) => {
                if imm != 0 {
                    self.s().i64_const(imm).i64_add();
                }
            }
            AddrMode::PostIndex(_) => {}
            AddrMode::Register { rm, extend, shift } => {
                self.get(rm);
                self.extend(extend, shift);
                self.s().i64_add();
            }
        }
        self.s().local_set(L_A);
        match mode {
            AddrMode::PreIndex(_) => Some(0),
            AddrMode::PostIndex(imm) => Some(imm),
            _ => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn load_store(
        &mut self,
        k: u32,
        pc: u64,
        op: MemOp,
        bytes: u8,
        rt: Reg,
        rn: Reg,
        mode: AddrMode,
    ) {
        let store = matches!(op, MemOp::Store | MemOp::StoreV);
        let vector = matches!(op, MemOp::StoreV | MemOp::LoadV);
        if op == MemOp::Prefetch {
            // No access; only the SP alignment rule can fault (no write-back forms).
            if rn == wasmshim_aarch64::SP {
                self.step(k, pc, &Inst::Nop);
            }
            return;
        }
        let wb = self.address(rn, mode);
        let mut inputs = bit(rn);
        if let AddrMode::Register { rm, .. } = mode {
            inputs |= bit(rm);
        }
        let mut outputs = 0;
        if store && !vector {
            inputs |= bit(rt);
        }
        if !store && !vector {
            outputs |= bit(rt);
        }
        if wb.is_some() {
            outputs |= bit(rn);
        }
        let table = if store { TLB_WRITE } else { TLB_READ };
        self.guarded(
            k,
            pc,
            table,
            u64::from(bytes),
            check_sp(rn),
            inputs,
            outputs,
            |e| {
                let vbase = (layout::V + 16 * rt as usize) as u64;
                match op {
                    MemOp::Load | MemOp::LoadSigned64 | MemOp::LoadSigned32 => {
                        e.s().local_get(L_H);
                        e.load_int_at(op, bytes, 0);
                        e.set(rt);
                    }
                    MemOp::Store => {
                        e.s().local_get(L_H);
                        e.get(rt);
                        e.store_int_at(bytes, 0);
                    }
                    MemOp::LoadV => e.load_vector(vbase, bytes, 0),
                    MemOp::StoreV => e.store_vector(vbase, bytes, 0),
                    MemOp::Prefetch => {}
                }
                if let Some(post) = wb {
                    e.s().local_get(L_A);
                    if post != 0 {
                        e.s().i64_const(post).i64_add();
                    }
                    e.set(rn);
                }
            },
        );
    }

    /// `DC ZVA`: zero the 64-byte block holding `Xt`.
    fn dc_zva(&mut self, k: u32, pc: u64, rt: Reg) {
        self.get(rt);
        self.s()
            .i64_const(!(wasmshim_aarch64::DC_ZVA_BYTES as i64 - 1))
            .i64_and()
            .local_set(L_A);
        let n = wasmshim_aarch64::DC_ZVA_BYTES;
        self.guarded(k, pc, TLB_WRITE, n, Check::None, bit(rt), 0, |e| {
            for i in 0..n / 16 {
                e.s()
                    .local_get(L_H)
                    .v128_const(0)
                    .v128_store(mem(16 * i, 0));
            }
        });
    }

    /// `LDXR`/`LDAXR`, `STXR`/`STLXR`, `LDAR`/`LDAPR`, `STLR` (single register): one host
    /// thread, so acquire/release ordering is free; the exclusive monitor is the CPU's.
    #[allow(clippy::too_many_arguments)]
    fn exclusive(
        &mut self,
        k: u32,
        pc: u64,
        op: ExclusiveOp,
        bytes: u8,
        rs: Reg,
        rt: Reg,
        rn: Reg,
    ) {
        let store = matches!(op, ExclusiveOp::StoreExclusive | ExclusiveOp::StoreRelease);
        self.get(rn);
        self.s().local_set(L_A);
        let (inputs, outputs) = match op {
            ExclusiveOp::LoadExclusive | ExclusiveOp::LoadAcquire => (bit(rn), bit(rt)),
            ExclusiveOp::StoreRelease => (bit(rn) | bit(rt), 0),
            ExclusiveOp::StoreExclusive => (bit(rn) | bit(rt), bit(rs)),
        };
        let table = if store { TLB_WRITE } else { TLB_READ };
        let n = u64::from(bytes);
        let monitor = layout::MONITOR as u64;
        self.guarded(
            k,
            pc,
            table,
            n,
            Check::Aligned(n),
            inputs,
            outputs,
            |e| match op {
                ExclusiveOp::LoadExclusive | ExclusiveOp::LoadAcquire => {
                    if op == ExclusiveOp::LoadExclusive {
                        e.s()
                            .local_get(P_CPU)
                            .local_get(L_A)
                            .i64_store(mem(monitor, 3));
                    }
                    e.s().local_get(L_H);
                    e.load_int_at(MemOp::Load, bytes, 0);
                    e.set(rt);
                }
                ExclusiveOp::StoreRelease => {
                    e.s().local_get(L_H);
                    e.get(rt);
                    e.store_int_at(bytes, 0);
                }
                ExclusiveOp::StoreExclusive => {
                    // ok = monitor == addr; the monitor is cleared either way.
                    e.s()
                        .local_get(P_CPU)
                        .i64_load(mem(monitor, 3))
                        .local_get(L_A)
                        .i64_eq()
                        .local_set(L_W0)
                        .local_get(P_CPU)
                        .i64_const(wasmshim_aarch64::NO_MONITOR as i64)
                        .i64_store(mem(monitor, 3))
                        .local_get(L_W0);
                    e.b.if_(None);
                    e.s().local_get(L_H);
                    e.get(rt);
                    e.store_int_at(bytes, 0);
                    e.b.end();
                    e.s().local_get(L_W0).i32_eqz().i64_extend_i32_u();
                    e.set(rs);
                }
            },
        );
    }

    fn load_literal(&mut self, k: u32, pc: u64, op: MemOp, bytes: u8, rt: Reg, offset: i64) {
        self.pc(pc.wrapping_add(offset as u64));
        self.s().local_set(L_A);
        self.guarded(
            k,
            pc,
            TLB_READ,
            u64::from(bytes),
            Check::None,
            0,
            bit(rt),
            |e| {
                e.s().local_get(L_H);
                e.load_int_at(op, bytes, 0);
                e.set(rt);
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn pair(
        &mut self,
        k: u32,
        pc: u64,
        load: bool,
        vector: bool,
        signed: bool,
        bytes: u8,
        rt: Reg,
        rt2: Reg,
        rn: Reg,
        mode: AddrMode,
    ) {
        let wb = self.address(rn, mode);
        let mut inputs = bit(rn);
        let mut outputs = 0;
        if !vector {
            if load {
                outputs |= bit(rt) | bit(rt2);
            } else {
                inputs |= bit(rt) | bit(rt2);
            }
        }
        if wb.is_some() {
            outputs |= bit(rn);
        }
        let table = if load { TLB_READ } else { TLB_WRITE };
        let n = u64::from(bytes);
        self.guarded(k, pc, table, 2 * n, check_sp(rn), inputs, outputs, |e| {
            if vector {
                let v1 = (layout::V + 16 * rt as usize) as u64;
                let v2 = (layout::V + 16 * rt2 as usize) as u64;
                if load {
                    e.load_vector(v1, bytes, 0);
                    e.load_vector(v2, bytes, n);
                } else {
                    e.store_vector(v1, bytes, 0);
                    e.store_vector(v2, bytes, n);
                }
            } else if load {
                let op = if signed {
                    MemOp::LoadSigned64
                } else {
                    MemOp::Load
                };
                e.s().local_get(L_H);
                e.load_int_at(op, bytes, 0);
                e.set(rt);
                e.s().local_get(L_H);
                e.load_int_at(op, bytes, n);
                e.set(rt2);
            } else {
                e.s().local_get(L_H);
                e.get(rt);
                e.store_int_at(bytes, 0);
                e.s().local_get(L_H);
                e.get(rt2);
                e.store_int_at(bytes, n);
            }
            if let Some(post) = wb {
                e.s().local_get(L_A);
                if post != 0 {
                    e.s().i64_const(post).i64_add();
                }
                e.set(rn);
            }
        });
    }

    /// Host address on the stack → the loaded value (i64) for a GPR load.
    fn load_int_at(&mut self, op: MemOp, bytes: u8, offset: u64) {
        let m = mem(offset, 0);
        match (op, bytes) {
            (MemOp::LoadSigned32, 1) => {
                self.s().i32_load8_s(m).i64_extend_i32_u();
            }
            (MemOp::LoadSigned32, 2) => {
                self.s().i32_load16_s(m).i64_extend_i32_u();
            }
            (MemOp::LoadSigned32, _) => {
                self.s().i64_load32_u(m);
            }
            (MemOp::LoadSigned64, 1) => {
                self.s().i64_load8_s(m);
            }
            (MemOp::LoadSigned64, 2) => {
                self.s().i64_load16_s(m);
            }
            (MemOp::LoadSigned64, 4) => {
                self.s().i64_load32_s(m);
            }
            (_, 1) => {
                self.s().i64_load8_u(m);
            }
            (_, 2) => {
                self.s().i64_load16_u(m);
            }
            (_, 4) => {
                self.s().i64_load32_u(m);
            }
            _ => {
                self.s().i64_load(m);
            }
        }
    }

    /// Host address and value (i64) on the stack → a store of `bytes`.
    fn store_int_at(&mut self, bytes: u8, offset: u64) {
        let m = mem(offset, 0);
        match bytes {
            1 => self.s().i64_store8(m),
            2 => self.s().i64_store16(m),
            4 => self.s().i64_store32(m),
            _ => self.s().i64_store(m),
        };
    }

    fn load_vector(&mut self, vbase: u64, bytes: u8, offset: u64) {
        self.s().local_get(P_CPU).local_get(L_H);
        if bytes == 16 {
            self.s()
                .i64_load(mem(offset, 0))
                .i64_store(mem(vbase, 3))
                .local_get(P_CPU)
                .local_get(L_H)
                .i64_load(mem(offset + 8, 0))
                .i64_store(mem(vbase + 8, 3));
        } else {
            self.load_int_at(MemOp::Load, bytes, offset);
            self.s()
                .i64_store(mem(vbase, 3))
                .local_get(P_CPU)
                .i64_const(0)
                .i64_store(mem(vbase + 8, 3));
        }
    }

    fn store_vector(&mut self, vbase: u64, bytes: u8, offset: u64) {
        self.s()
            .local_get(L_H)
            .local_get(P_CPU)
            .i64_load(mem(vbase, 3));
        self.store_int_at(bytes.min(8), offset);
        if bytes == 16 {
            self.s()
                .local_get(L_H)
                .local_get(P_CPU)
                .i64_load(mem(vbase + 8, 3))
                .i64_store(mem(offset + 8, 0));
        }
    }
}

fn sysreg_const(reg: SysReg) -> Option<u64> {
    match reg {
        SysReg::DczidEl0 => Some(wasmshim_aarch64::DCZID_EL0),
        SysReg::CtrEl0 => Some(wasmshim_aarch64::CTR_EL0),
        SysReg::MidrEl1 => Some(wasmshim_aarch64::MIDR_EL1),
        SysReg::CntfrqEl0 => Some(wasmshim_aarch64::COUNTER_FREQUENCY),
        _ => None,
    }
}

/// GPR loads/stores (and prefetch) of any size, and SIMD&FP loads/stores of 1–16 bytes.
fn gpr_or_simple_vector(op: MemOp, bytes: u8) -> bool {
    match op {
        MemOp::LoadV | MemOp::StoreV => matches!(bytes, 1 | 2 | 4 | 8 | 16),
        _ => true,
    }
}

/// What an instruction executed by `jit_step` may read and change, conservatively.
struct Effects {
    gpr_reads: u64,
    gpr_writes: u64,
    reads_flags: bool,
    writes_flags: bool,
}

fn effects(inst: &Inst) -> Effects {
    let all = Effects {
        gpr_reads: ALL_REGS,
        gpr_writes: ALL_REGS,
        reads_flags: true,
        writes_flags: true,
    };
    let none = Effects {
        gpr_reads: 0,
        gpr_writes: 0,
        reads_flags: false,
        writes_flags: false,
    };
    match inst {
        Inst::Simd(s) => match s {
            SimdInst::ThreeSame { .. }
            | SimdInst::ThreeDiff { .. }
            | SimdInst::TwoMisc { .. }
            | SimdInst::AcrossLanes { .. }
            | SimdInst::ScalarPairwise { .. }
            | SimdInst::ScalarDup { .. }
            | SimdInst::ModImm { .. }
            | SimdInst::ShiftImm { .. }
            | SimdInst::Permute { .. }
            | SimdInst::Ext { .. }
            | SimdInst::Tbl { .. }
            | SimdInst::Indexed { .. }
            | SimdInst::FpImm { .. }
            | SimdInst::Fp1 { .. }
            | SimdInst::Fp2 { .. }
            | SimdInst::Fp3 { .. } => none,
            SimdInst::FpCsel { .. } => Effects {
                reads_flags: true,
                ..none
            },
            SimdInst::FpCmp { .. } => Effects {
                writes_flags: true,
                ..none
            },
            SimdInst::FpCondCmp { .. } => Effects {
                reads_flags: true,
                writes_flags: true,
                ..none
            },
            SimdInst::LdStMulti { rn, post, .. } | SimdInst::LdStSingle { rn, post, .. } => {
                let post_reg = post.map_or(0, bit);
                Effects {
                    gpr_reads: bit(*rn) | post_reg,
                    gpr_writes: if post.is_some() { bit(*rn) } else { 0 },
                    ..none
                }
            }
            _ => Effects {
                reads_flags: false,
                writes_flags: false,
                ..all
            },
        },
        Inst::Exclusive {
            rs, rt, rt2, rn, ..
        } => Effects {
            gpr_reads: bit(*rt) | bit(*rt2) | bit(*rn),
            gpr_writes: bit(*rs) | bit(*rt) | bit(*rt2),
            ..none
        },
        Inst::Cas { rs, rt, rn, .. } | Inst::Atomic { rs, rt, rn, .. } => Effects {
            gpr_reads: bit(*rs) | bit(*rt) | bit(*rn),
            gpr_writes: bit(*rs) | bit(*rt),
            ..none
        },
        Inst::Casp { rs, rt, rn, .. } => {
            let both = bit(*rs) | bit(rs + 1) | bit(*rt) | bit(rt + 1);
            Effects {
                gpr_reads: both | bit(*rn),
                gpr_writes: both,
                ..none
            }
        }
        Inst::Nop | Inst::Clrex => none,
        Inst::DcZva { rt } => Effects {
            gpr_reads: bit(*rt),
            ..none
        },
        Inst::Mrs { rt, reg } => Effects {
            gpr_writes: bit(*rt),
            reads_flags: *reg == SysReg::Nzcv,
            ..none
        },
        Inst::Msr { rt, reg } => Effects {
            gpr_reads: bit(*rt),
            writes_flags: *reg == SysReg::Nzcv,
            ..none
        },
        Inst::Dp1 { rd, rn, .. } => Effects {
            gpr_reads: bit(*rn),
            gpr_writes: bit(*rd),
            ..none
        },
        Inst::Dp3 { rd, rn, rm, ra, .. } => Effects {
            gpr_reads: bit(*rn) | bit(*rm) | bit(*ra),
            gpr_writes: bit(*rd),
            ..none
        },
        Inst::AddSubCarry { rd, rn, rm, .. } => Effects {
            gpr_reads: bit(*rn) | bit(*rm),
            gpr_writes: bit(*rd),
            reads_flags: true,
            writes_flags: true,
        },
        _ => all,
    }
}
