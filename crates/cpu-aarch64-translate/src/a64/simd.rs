//! Native translations of the instructions hot code (glibc's NEON string routines, CPython,
//! V8) would otherwise hand to `jit_step`: FP↔GPR moves, byte/bit reversal, high multiplies,
//! and a subset of Advanced SIMD integer operations on Wasm `v128` values.
//!
//! Every case mirrors the interpreter (`wasmshim-aarch64`'s `simd_exec.rs`) exactly, including
//! the zeroing of the upper half for 64-bit (`q == false`) forms and the encodings it treats
//! as undefined: those, and every combination not listed here, still go to `jit_step`.
//! SIMD&FP registers live in memory (`cpu.vreg`), so these touch no cached locals except the
//! general registers they name.

use wasmshim_aarch64::simd::{ModImmOp, SimdInst};
use wasmshim_aarch64::{Dp1Op, Dp3Op, Inst, Reg, VReg, ZR, layout};

use super::{A64, L_T0, L_T1, L_W0, P_CPU, W, mem};
use crate::ir::PureOps;
use crate::ir::build::Builder;

/// Low 64 bits set.
pub(super) const LOW64: i128 = 0xffff_ffff_ffff_ffff;

pub(super) fn voff(n: VReg) -> u64 {
    (layout::V + 16 * usize::from(n & 31)) as u64
}

/// Lane shape of an element size (`size`: 0 = 8 bits … 3 = 64 bits).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Lanes {
    B,
    H,
    S,
    D,
}

impl Lanes {
    pub(super) fn of(size: u8) -> Self {
        match size {
            0 => Self::B,
            1 => Self::H,
            2 => Self::S,
            _ => Self::D,
        }
    }
    pub(super) fn bytes(self) -> usize {
        match self {
            Self::B => 1,
            Self::H => 2,
            Self::S => 4,
            Self::D => 8,
        }
    }
}

/// Byte shuffle indices selecting, for each result element `i` (of `lanes`), the element given
/// by `pick(i)` (`Some((false, k))`: element `k` of the first operand, `true`: of the second).
/// Elements past `count` are don't-care (taken from element 0 of the first operand).
pub(super) fn shuffle(
    lanes: Lanes,
    count: usize,
    pick: impl Fn(usize) -> (bool, usize),
) -> [u8; 16] {
    let s = lanes.bytes();
    let mut out = [0u8; 16];
    for i in 0..16 / s {
        let (second, k) = if i < count { pick(i) } else { (false, 0) };
        for j in 0..s {
            out[i * s + j] = (if second { 16 } else { 0 } + k * s + j) as u8;
        }
    }
    out
}

impl A64 {
    pub(super) fn vload(&mut self, n: VReg) {
        self.s().local_get(P_CPU).v128_load(mem(voff(n), 0));
    }

    /// The v128 on the stack becomes `Vd` (upper half zeroed unless `q`). The CPU pointer must
    /// already be below it: use [`Self::vstore_begin`] first.
    fn vstore_end(&mut self, d: VReg, q: bool) {
        if !q {
            self.s().v128_const(LOW64).v128_and();
        }
        self.s().v128_store(mem(voff(d), 0));
    }

    fn vstore_begin(&mut self) {
        self.s().local_get(P_CPU);
    }

    /// Handles `inst` natively if it is one of this module's cases; `false`: use `jit_step`.
    pub(super) fn native_extra(&mut self, k: u32, pc: u64, inst: &Inst) -> bool {
        match *inst {
            Inst::Dp1 { sf, op, rd, rn } => self.dp1(sf, op, rd, rn),
            Inst::Dp3 {
                op: op @ (Dp3Op::Umulh | Dp3Op::Smulh),
                rd,
                rn,
                rm,
                ..
            } => {
                self.mulh(op == Dp3Op::Smulh, rd, rn, rm);
                true
            }
            Inst::Simd(s) => {
                self.simd(&s)
                    || (self.fp && self.scalar_fp(k, pc, inst, &s))
                    || (self.vector && self.neon(k, pc, inst, &s))
            }
            _ => false,
        }
    }

    // ---- general-register helpers ----

    /// `REV`, `REV16`, `REV32`, `RBIT` (CLZ/CLS are handled by the main emitter).
    fn dp1(&mut self, sf: bool, op: Dp1Op, rd: Reg, rn: Reg) -> bool {
        let w = W::of(sf);
        // Swap adjacent groups of `bits` bits: ((x >> b) & m) | ((x & m) << b).
        let swap = |e: &mut Self, bits: u32, mask: u64| {
            let (t, m, b) = (if w == W::X { L_T0 } else { L_W0 }, mask, bits);
            match w {
                W::X => {
                    e.s()
                        .local_tee(t)
                        .i64_const(i64::from(b))
                        .i64_shr_u()
                        .i64_const(m as i64)
                        .i64_and()
                        .local_get(t)
                        .i64_const(m as i64)
                        .i64_and()
                        .i64_const(i64::from(b))
                        .i64_shl()
                        .i64_or();
                }
                W::W => {
                    e.s()
                        .local_tee(t)
                        .i32_const(b as i32)
                        .i32_shr_u()
                        .i32_const(m as u32 as i32)
                        .i32_and()
                        .local_get(t)
                        .i32_const(m as u32 as i32)
                        .i32_and()
                        .i32_const(b as i32)
                        .i32_shl()
                        .i32_or();
                }
            }
        };
        self.get_w(w, rn);
        match (op, w) {
            (Dp1Op::Rev16, _) => swap(self, 8, 0x00ff_00ff_00ff_00ff),
            (Dp1Op::Rev32, W::X) => {
                swap(self, 8, 0x00ff_00ff_00ff_00ff);
                swap(self, 16, 0x0000_ffff_0000_ffff);
            }
            (Dp1Op::Rev, W::X) => {
                swap(self, 8, 0x00ff_00ff_00ff_00ff);
                swap(self, 16, 0x0000_ffff_0000_ffff);
                self.s().i64_const(32).i64_rotl();
            }
            (Dp1Op::Rev | Dp1Op::Rev32, W::W) => {
                swap(self, 8, 0x00ff_00ff);
                self.s().i32_const(16).i32_rotl();
            }
            (Dp1Op::Rbit, _) => {
                swap(self, 1, 0x5555_5555_5555_5555);
                swap(self, 2, 0x3333_3333_3333_3333);
                swap(self, 4, 0x0f0f_0f0f_0f0f_0f0f);
                swap(self, 8, 0x00ff_00ff_00ff_00ff);
                match w {
                    W::X => {
                        swap(self, 16, 0x0000_ffff_0000_ffff);
                        self.s().i64_const(32).i64_rotl();
                    }
                    W::W => {
                        self.s().i32_const(16).i32_rotl();
                    }
                }
            }
            _ => {
                self.s().drop();
                return false;
            }
        }
        self.set_w(w, rd);
        true
    }

    /// `UMULH`/`SMULH`: the high 64 bits of the 128-bit product, from 32-bit halves.
    fn mulh(&mut self, signed: bool, rd: Reg, rn: Reg, rm: Reg) {
        // T0 = x, T1 = y; W0/W1 unused. Locals beyond the shared temps: reuse L_A and L_PC
        // (dead between memory accesses and exits).
        let (x, y, a, b) = (L_T0, L_T1, super::L_A, super::L_PC);
        self.get(rn);
        self.s().local_set(x);
        self.get(rm);
        self.s().local_set(y);
        let lo = |s: &mut Builder, l: u32| {
            s.local_get(l).i64_const(0xffff_ffff).i64_and();
        };
        let hi = |s: &mut Builder, l: u32| {
            s.local_get(l).i64_const(32).i64_shr_u();
        };
        let s = self.s();
        // a = lo(x) * lo(y); b = hi(x) * lo(y) + (a >> 32)
        lo(s, x);
        lo(s, y);
        s.i64_mul().local_set(a);
        hi(s, x);
        lo(s, y);
        s.i64_mul()
            .local_get(a)
            .i64_const(32)
            .i64_shr_u()
            .i64_add()
            .local_set(b);
        // a = lo(x) * hi(y) + lo(b)
        lo(s, x);
        hi(s, y);
        s.i64_mul();
        lo(s, b);
        s.i64_add().local_set(a);
        // high = hi(x) * hi(y) + hi(b) + hi(a)
        hi(s, x);
        hi(s, y);
        s.i64_mul();
        hi(s, b);
        s.i64_add();
        hi(s, a);
        s.i64_add();
        if signed {
            // signed high = unsigned high - (x < 0 ? y : 0) - (y < 0 ? x : 0)
            s.local_get(y)
                .i64_const(0)
                .local_get(x)
                .i64_const(0)
                .i64_lt_s()
                .select()
                .i64_sub()
                .local_get(x)
                .i64_const(0)
                .local_get(y)
                .i64_const(0)
                .i64_lt_s()
                .select()
                .i64_sub();
        }
        self.set(rd);
    }

    // ---- SIMD ----

    fn simd(&mut self, s: &SimdInst) -> bool {
        match *s {
            SimdInst::FpInt {
                sf,
                ftype,
                rmode,
                opcode,
                rd,
                rn,
            } => self.fmov_general(sf, ftype, rmode, opcode, rd, rn),
            SimdInst::ThreeSame {
                scalar: false,
                q,
                u,
                size,
                opcode,
                rd,
                rn,
                rm,
            } => self.three_same(q, u, size, opcode, rd, rn, rm),
            SimdInst::TwoMisc {
                scalar: false,
                q,
                u,
                size,
                opcode,
                rd,
                rn,
            } => self.two_misc(q, u, size, opcode, rd, rn),
            SimdInst::Copy {
                q,
                op,
                imm5,
                imm4,
                rd,
                rn,
            } => self.copy(q, op, imm5, imm4, rd, rn),
            SimdInst::ModImm { q, op, imm, rd } => {
                let imm128 = i128::from(imm as i64) & LOW64 | (i128::from(imm as i64) << 64);
                self.vstore_begin();
                match op {
                    ModImmOp::Movi => {
                        self.s().v128_const(imm128);
                    }
                    ModImmOp::Mvni => {
                        self.s().v128_const(!imm128);
                    }
                    ModImmOp::Orr => {
                        self.vload(rd);
                        self.s().v128_const(imm128).v128_or();
                    }
                    ModImmOp::Bic => {
                        self.vload(rd);
                        self.s().v128_const(imm128).v128_andnot();
                    }
                }
                self.vstore_end(rd, q);
                true
            }
            SimdInst::Permute {
                q,
                size,
                opcode,
                rd,
                rn,
                rm,
            } => self.permute(q, size, opcode, rd, rn, rm),
            SimdInst::Ext {
                q,
                index,
                rd,
                rn,
                rm,
            } => {
                let bytes = if q { 16 } else { 8 };
                let mut idx = [0u8; 16];
                for (i, o) in idx.iter_mut().enumerate().take(bytes) {
                    let k = i + usize::from(index);
                    *o = if k < bytes {
                        k as u8
                    } else {
                        (16 + k - bytes) as u8
                    };
                }
                self.vstore_begin();
                self.vload(rn);
                self.vload(rm);
                self.s().i8x16_shuffle(idx);
                self.vstore_end(rd, q);
                true
            }
            SimdInst::Tbl {
                q,
                len: 0,
                tbx: false,
                rd,
                rn,
                rm,
            } => {
                // One table register: swizzle gives 0 for indices >= 16, as TBL does.
                self.vstore_begin();
                self.vload(rn);
                self.vload(rm);
                self.s().i8x16_swizzle();
                self.vstore_end(rd, q);
                true
            }
            _ => false,
        }
    }

    /// `FMOV` between general and FP registers (the interpreter's `fp_int` first cases).
    fn fmov_general(&mut self, sf: bool, ftype: u8, rmode: u8, opcode: u8, rd: u8, rn: u8) -> bool {
        match (sf, ftype, rmode, opcode) {
            // FMOV Wd, Sn / Xd, Dn
            (false, 0, 0, 0b110) | (true, 1, 0, 0b110) => {
                self.s().local_get(P_CPU);
                if sf {
                    self.s().i64_load(mem(voff(rn), 3));
                } else {
                    self.s().i64_load32_u(mem(voff(rn), 2));
                }
                self.set(rd);
            }
            // FMOV Sn, Wd / Dd, Xn: the scalar written, every other bit cleared.
            (false, 0, 0, 0b111) | (true, 1, 0, 0b111) => {
                self.s().local_get(P_CPU);
                self.get(rn);
                if !sf {
                    self.s().i64_const(0xffff_ffff).i64_and();
                }
                self.s()
                    .i64_store(mem(voff(rd), 3))
                    .local_get(P_CPU)
                    .i64_const(0)
                    .i64_store(mem(voff(rd) + 8, 3));
            }
            // FMOV Xd, Vn.D[1]
            (true, 2, 1, 0b110) => {
                self.s().local_get(P_CPU).i64_load(mem(voff(rn) + 8, 3));
                self.set(rd);
            }
            // FMOV Vd.D[1], Xn
            (true, 2, 1, 0b111) => {
                self.s().local_get(P_CPU);
                self.get(rn);
                self.s().i64_store(mem(voff(rd) + 8, 3));
            }
            _ => return false,
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn three_same(
        &mut self,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
        rm: VReg,
    ) -> bool {
        let lanes = Lanes::of(size);
        if opcode == 0b00011 {
            // Logical: AND, BIC, ORR, ORN, EOR, BSL, BIT, BIF (size selects).
            self.vstore_begin();
            match (u, size) {
                (false, 0) => {
                    self.vload(rn);
                    self.vload(rm);
                    self.s().v128_and();
                }
                (false, 1) => {
                    self.vload(rn);
                    self.vload(rm);
                    self.s().v128_andnot();
                }
                (false, 2) => {
                    self.vload(rn);
                    self.vload(rm);
                    self.s().v128_or();
                }
                (false, _) => {
                    self.vload(rn);
                    self.vload(rm);
                    self.s().v128_not().v128_or();
                }
                (true, 0) => {
                    self.vload(rn);
                    self.vload(rm);
                    self.s().v128_xor();
                }
                (true, 1) => {
                    // BSL: (d & a) | (!d & b)
                    self.vload(rn);
                    self.vload(rm);
                    self.vload(rd);
                    self.s().v128_bitselect();
                }
                (true, 2) => {
                    // BIT: (d & !b) | (a & b)
                    self.vload(rn);
                    self.vload(rd);
                    self.vload(rm);
                    self.s().v128_bitselect();
                }
                (true, _) => {
                    // BIF: (d & b) | (a & !b)
                    self.vload(rd);
                    self.vload(rn);
                    self.vload(rm);
                    self.s().v128_bitselect();
                }
            }
            self.vstore_end(rd, q);
            return true;
        }
        // Element-wise comparisons and pairwise max/min/add.
        let cmp = match (opcode, u) {
            (0b10001, true) => Some(Cmp::Eq),
            (0b10001, false) => Some(Cmp::Tst),
            (0b00110, false) => Some(Cmp::GtS),
            (0b00110, true) => Some(Cmp::GtU),
            (0b00111, false) => Some(Cmp::GeS),
            (0b00111, true) => Some(Cmp::GeU),
            _ => None,
        };
        if let Some(c) = cmp {
            if lanes == Lanes::D && matches!(c, Cmp::GtU | Cmp::GeU) {
                return false; // no i64x2 unsigned comparison in Wasm SIMD
            }
            self.vstore_begin();
            self.vload(rn);
            self.vload(rm);
            self.compare(c, lanes);
            self.vstore_end(rd, q);
            return true;
        }
        let pairwise = match (opcode, u) {
            (0b10100, true) => Some(Pair::MaxU),
            (0b10100, false) => Some(Pair::MaxS),
            (0b10101, true) => Some(Pair::MinU),
            (0b10101, false) => Some(Pair::MinS),
            (0b10111, false) => Some(Pair::Add),
            _ => None,
        };
        let Some(p) = pairwise else {
            return false;
        };
        if lanes == Lanes::D && p != Pair::Add {
            return false;
        }
        let n = (if q { 16 } else { 8 }) / lanes.bytes();
        let pick = |odd: usize| {
            move |i: usize| {
                if i < n / 2 {
                    (false, 2 * i + odd)
                } else {
                    (true, 2 * (i - n / 2) + odd)
                }
            }
        };
        let evens = shuffle(lanes, n, pick(0));
        let odds = shuffle(lanes, n, pick(1));
        self.vstore_begin();
        self.vload(rn);
        self.vload(rm);
        self.s().i8x16_shuffle(evens);
        self.vload(rn);
        self.vload(rm);
        self.s().i8x16_shuffle(odds);
        let s = self.s();
        match (p, lanes) {
            (Pair::MaxU, Lanes::B) => s.i8x16_max_u(),
            (Pair::MaxU, Lanes::H) => s.i16x8_max_u(),
            (Pair::MaxU, _) => s.i32x4_max_u(),
            (Pair::MaxS, Lanes::B) => s.i8x16_max_s(),
            (Pair::MaxS, Lanes::H) => s.i16x8_max_s(),
            (Pair::MaxS, _) => s.i32x4_max_s(),
            (Pair::MinU, Lanes::B) => s.i8x16_min_u(),
            (Pair::MinU, Lanes::H) => s.i16x8_min_u(),
            (Pair::MinU, _) => s.i32x4_min_u(),
            (Pair::MinS, Lanes::B) => s.i8x16_min_s(),
            (Pair::MinS, Lanes::H) => s.i16x8_min_s(),
            (Pair::MinS, _) => s.i32x4_min_s(),
            (Pair::Add, Lanes::B) => s.i8x16_add(),
            (Pair::Add, Lanes::H) => s.i16x8_add(),
            (Pair::Add, Lanes::S) => s.i32x4_add(),
            (Pair::Add, Lanes::D) => s.i64x2_add(),
        };
        self.vstore_end(rd, q);
        true
    }

    fn compare(&mut self, c: Cmp, lanes: Lanes) {
        let s = self.s();
        match (c, lanes) {
            (Cmp::Eq, Lanes::B) => s.i8x16_eq(),
            (Cmp::Eq, Lanes::H) => s.i16x8_eq(),
            (Cmp::Eq, Lanes::S) => s.i32x4_eq(),
            (Cmp::Eq, Lanes::D) => s.i64x2_eq(),
            (Cmp::Tst, l) => {
                s.v128_and().v128_const(0);
                match l {
                    Lanes::B => s.i8x16_ne(),
                    Lanes::H => s.i16x8_ne(),
                    Lanes::S => s.i32x4_ne(),
                    Lanes::D => s.i64x2_ne(),
                }
            }
            (Cmp::GtS, Lanes::B) => s.i8x16_gt_s(),
            (Cmp::GtS, Lanes::H) => s.i16x8_gt_s(),
            (Cmp::GtS, Lanes::S) => s.i32x4_gt_s(),
            (Cmp::GtS, Lanes::D) => s.i64x2_gt_s(),
            (Cmp::GeS, Lanes::B) => s.i8x16_ge_s(),
            (Cmp::GeS, Lanes::H) => s.i16x8_ge_s(),
            (Cmp::GeS, Lanes::S) => s.i32x4_ge_s(),
            (Cmp::GeS, Lanes::D) => s.i64x2_ge_s(),
            (Cmp::GtU, Lanes::B) => s.i8x16_gt_u(),
            (Cmp::GtU, Lanes::H) => s.i16x8_gt_u(),
            (Cmp::GtU, _) => s.i32x4_gt_u(),
            (Cmp::GeU, Lanes::B) => s.i8x16_ge_u(),
            (Cmp::GeU, Lanes::H) => s.i16x8_ge_u(),
            (Cmp::GeU, _) => s.i32x4_ge_u(),
        };
    }

    fn two_misc(&mut self, q: bool, u: bool, size: u8, opcode: u8, rd: VReg, rn: VReg) -> bool {
        let lanes = Lanes::of(size);
        self.vstore_begin();
        self.vload(rn);
        match (opcode, u) {
            // CNT, NOT (byte elements only).
            (0b00101, false) if size == 0 => {
                self.s().i8x16_popcnt();
            }
            (0b00101, true) if size == 0 => {
                self.s().v128_not();
            }
            // Compare with zero.
            (0b01000, false) => {
                self.s().v128_const(0);
                self.compare(Cmp::GtS, lanes);
            }
            (0b01000, true) => {
                self.s().v128_const(0);
                self.compare(Cmp::GeS, lanes);
            }
            (0b01001, false) => {
                self.s().v128_const(0);
                self.compare(Cmp::Eq, lanes);
            }
            (0b01001, true) => {
                // CMLE #0: x <= 0, i.e. 0 >= x.
                self.s()
                    .local_set(super::L_V0)
                    .v128_const(0)
                    .local_get(super::L_V0);
                self.compare(Cmp::GeS, lanes);
            }
            (0b01010, false) => {
                // CMLT #0: 0 > x.
                self.s()
                    .local_set(super::L_V0)
                    .v128_const(0)
                    .local_get(super::L_V0);
                self.compare(Cmp::GtS, lanes);
            }
            _ => {
                // Not ours: undo the pushes.
                self.s().drop().drop();
                return false;
            }
        }
        self.vstore_end(rd, q);
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn copy(&mut self, q: bool, op: bool, imm5: u8, imm4: u8, rd: u8, rn: u8) -> bool {
        let size = imm5.trailing_zeros();
        if size > 3 {
            return false;
        }
        let lanes = Lanes::of(size as u8);
        let index = (u32::from(imm5) >> (size + 1)) as u8;
        if op {
            // INS (element): Vd.T[index] = Vn.T[imm4 >> size].
            if !q {
                return false;
            }
            let src = imm4 >> size;
            self.vstore_begin();
            self.vload(rd);
            self.vload(rn);
            self.extract(lanes, src, false);
            self.replace(lanes, index);
            self.vstore_end(rd, true);
            return true;
        }
        match imm4 {
            // DUP (element) / DUP (general)
            0b0000 | 0b0001 => {
                if size == 3 && !q {
                    return false;
                }
                self.vstore_begin();
                if imm4 == 0 {
                    self.vload(rn);
                    self.extract(lanes, index, false);
                } else {
                    self.get(rn);
                    if lanes != Lanes::D {
                        self.s().i32_wrap_i64();
                    }
                }
                let s = self.s();
                match lanes {
                    Lanes::B => s.i8x16_splat(),
                    Lanes::H => s.i16x8_splat(),
                    Lanes::S => s.i32x4_splat(),
                    Lanes::D => s.i64x2_splat(),
                };
                self.vstore_end(rd, q);
            }
            // INS (general): Vd.T[index] = Xn
            0b0011 => {
                self.vstore_begin();
                self.vload(rd);
                self.get(rn);
                if lanes != Lanes::D {
                    self.s().i32_wrap_i64();
                }
                self.replace(lanes, index);
                self.vstore_end(rd, true);
            }
            // SMOV / UMOV: to a general register (32 or 64 bits by `q`).
            0b0101 | 0b0111 => {
                let signed = imm4 == 0b0101;
                if rd == ZR {
                    return true; // the result is discarded; nothing else changes
                }
                self.vload(rn);
                self.extract(lanes, index, signed);
                if lanes != Lanes::D {
                    if signed {
                        self.s().i64_extend_i32_s();
                    } else {
                        self.s().i64_extend_i32_u();
                    }
                }
                if !q {
                    self.s().i64_const(0xffff_ffff).i64_and();
                }
                self.set(rd);
            }
            _ => return false,
        }
        true
    }

    /// v128 on the stack → lane `i` (i32 for B/H/S, i64 for D).
    fn extract(&mut self, lanes: Lanes, i: u8, signed: bool) {
        let s = self.s();
        match (lanes, signed) {
            (Lanes::B, false) => s.i8x16_extract_lane_u(i),
            (Lanes::B, true) => s.i8x16_extract_lane_s(i),
            (Lanes::H, false) => s.i16x8_extract_lane_u(i),
            (Lanes::H, true) => s.i16x8_extract_lane_s(i),
            (Lanes::S, _) => s.i32x4_extract_lane(i),
            (Lanes::D, _) => s.i64x2_extract_lane(i),
        };
    }

    /// v128 and a lane value on the stack → the v128 with lane `i` replaced.
    fn replace(&mut self, lanes: Lanes, i: u8) {
        let s = self.s();
        match lanes {
            Lanes::B => s.i8x16_replace_lane(i),
            Lanes::H => s.i16x8_replace_lane(i),
            Lanes::S => s.i32x4_replace_lane(i),
            Lanes::D => s.i64x2_replace_lane(i),
        };
    }

    #[allow(clippy::too_many_arguments)]
    fn permute(&mut self, q: bool, size: u8, opcode: u8, rd: VReg, rn: VReg, rm: VReg) -> bool {
        if size == 3 && !q {
            return false;
        }
        let lanes = Lanes::of(size);
        let n = (if q { 16 } else { 8 }) / lanes.bytes();
        let half = n / 2;
        let idx = match opcode {
            0b001 | 0b101 => shuffle(lanes, n, |i| {
                let k = 2 * i + usize::from(opcode == 0b101);
                if k < n { (false, k) } else { (true, k - n) }
            }),
            0b010 | 0b110 => shuffle(lanes, n, |i| {
                let base = (i & !1) + usize::from(opcode == 0b110);
                (i % 2 == 1, base)
            }),
            0b011 | 0b111 => shuffle(lanes, n, |i| {
                let k = i / 2 + if opcode == 0b111 { half } else { 0 };
                (i % 2 == 1, k)
            }),
            _ => return false,
        };
        self.vstore_begin();
        self.vload(rn);
        self.vload(rm);
        self.s().i8x16_shuffle(idx);
        self.vstore_end(rd, q);
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Cmp {
    Eq,
    Tst,
    GtS,
    GtU,
    GeS,
    GeU,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pair {
    MaxU,
    MaxS,
    MinU,
    MinS,
    Add,
}
