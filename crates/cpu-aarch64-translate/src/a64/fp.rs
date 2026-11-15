//! Scalar floating point in translated code: the scalar arithmetic, conversion, compare
//! and move instructions, so they do not each go to the interpreter one at a time.
//!
//! The interpreter (`wasmshim-aarch64`'s `fp.rs`) computes IEEE 754 results for non-NaN operands,
//! always rounding to nearest even, with no FPSR flags, and applies the Arm ARM's NaN rules
//! explicitly. Wasm's float operations give the same IEEE results for non-NaN operands, but
//! leave NaN bit patterns open. So every float result here is checked: a NaN result (from a NaN
//! operand or an invalid operation) is not stored; the interpreter executes the instruction
//! instead (`jit_step`), producing Arm's NaN. Moves, absolute value and negation work on the bits
//! and need no check; compares turn NaN into "unordered" themselves.
//!
//! `FRINTA` and `FCVTA*` (ties away from zero) use `round_away`; fused multiply-add is `fma.rs`
//! (exact emulation). Not lowered (still `jit_step`): half precision.

use wasmshim_aarch64::simd::SimdInst;
use wasmshim_aarch64::{Cond, Inst, VReg, ZR};

use super::simd::voff;
use super::{A64, L_FK, L_FR, L_T0, L_T1, L_W0, P_CPU, kind, mem};
use crate::ir::PureOps;

/// The precision of an FP type field: `Some(true)` for double, `Some(false)` single.
fn double(ftype: u8) -> Option<bool> {
    match ftype {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

impl A64 {
    /// Handles a scalar FP instruction natively; `false`: use `jit_step`.
    pub(super) fn scalar_fp(&mut self, k: u32, pc: u64, inst: &Inst, s: &SimdInst) -> bool {
        match *s {
            SimdInst::Fp2 {
                ftype,
                opcode,
                rd,
                rn,
                rm,
            } => {
                let Some(d) = double(ftype) else { return false };
                if opcode > 8 {
                    return false;
                }
                self.fload(rn, d);
                self.fload(rm, d);
                let b = self.s();
                match (opcode, d) {
                    (0 | 8, false) => b.f32_mul(),
                    (1, false) => b.f32_div(),
                    (2, false) => b.f32_add(),
                    (3, false) => b.f32_sub(),
                    // FMAX/FMIN, and FMAXNM/FMINNM (which differ only when a NaN is involved,
                    // and then the interpreter decides): -0 < +0 like Wasm's.
                    (4 | 6, false) => b.f32_max(),
                    (5 | 7, false) => b.f32_min(),
                    (0 | 8, true) => b.f64_mul(),
                    (1, true) => b.f64_div(),
                    (2, true) => b.f64_add(),
                    (3, true) => b.f64_sub(),
                    (4 | 6, true) => b.f64_max(),
                    (5 | 7, true) => b.f64_min(),
                    _ => unreachable!(),
                };
                // FNMUL negates the product (a NaN product is the interpreter's, as always).
                self.fresult(k, pc, inst, rd, d, opcode == 8);
                true
            }
            SimdInst::Fp1 {
                ftype,
                opcode,
                rd,
                rn,
            } => self.fp1(k, pc, inst, ftype, opcode, rd, rn),
            SimdInst::FpCmp {
                ftype,
                zero,
                rn,
                rm,
            } => {
                let Some(d) = double(ftype) else { return false };
                self.fload(rn, d);
                self.s().local_set(L_T0);
                if zero {
                    self.fzero(d);
                } else {
                    self.fload(rm, d);
                }
                self.s().local_set(L_T1);
                self.fcmp_flags(d);
                true
            }
            SimdInst::FpCondCmp {
                ftype,
                rn,
                rm,
                cond,
                nzcv,
            } => {
                let Some(d) = double(ftype) else { return false };
                self.cond(cond);
                self.b.if_(None);
                self.fload(rn, d);
                self.s().local_set(L_T0);
                self.fload(rm, d);
                self.s().local_set(L_T1);
                self.fcmp_flags(d);
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
                self.kind = Some(kind::PACKED);
                true
            }
            SimdInst::FpCsel {
                ftype,
                rd,
                rn,
                rm,
                cond,
            } => {
                let Some(d) = double(ftype) else { return false };
                self.fcsel(d, rd, rn, rm, cond);
                true
            }
            SimdInst::FpImm { ftype, bits, rd } => {
                let Some(d) = double(ftype) else { return false };
                let mask = if d { u64::MAX } else { 0xffff_ffff };
                self.s().i64_const((bits & mask) as i64);
                self.bits_store(rd);
                true
            }
            SimdInst::FpInt {
                sf,
                ftype,
                rmode,
                opcode,
                rd,
                rn,
            } => self.fp_int(sf, ftype, rmode, opcode, rd, rn),
            SimdInst::FpFixed {
                sf,
                ftype,
                rmode,
                opcode,
                fbits,
                rd,
                rn,
            } => self.fp_fixed(sf, ftype, rmode, opcode, fbits, rd, rn),
            SimdInst::Fp3 {
                ftype,
                o1,
                o0,
                rd,
                rn,
                rm,
                ra,
            } => {
                let Some(d) = double(ftype) else { return false };
                self.fused(k, pc, inst, d, o1, o0, rd, rn, rm, ra);
                true
            }
            _ => false,
        }
    }

    /// Rounds the float on the stack to an integral value, ties away from zero (`FRINTA`):
    /// `t = trunc(x)`, then one step away from zero when `|x - t| >= 0.5`. `x - t` is exact,
    /// `t ± 1` too (below 2^23 or 2^52, where `x` may have a fraction), and a zero keeps its
    /// sign. A NaN stays a NaN (the caller's check); an infinity stays (`x - t` is a NaN).
    pub(super) fn round_away(&mut self, d: bool) {
        const X: u32 = 80;
        const T: u32 = 81;
        let one = |e: &mut Self, v: f64| {
            if d {
                e.s().i64_const(v.to_bits() as i64).f64_reinterpret_i64();
            } else {
                e.s()
                    .i32_const((v as f32).to_bits() as i32)
                    .f32_reinterpret_i32();
            }
        };
        let s = self.s();
        s.local_tee(X);
        if d {
            s.f64_trunc();
        } else {
            s.f32_trunc();
        }
        s.local_set(T);
        // t + 1, t - 1, t
        self.s().local_get(T);
        one(self, 1.0);
        if d {
            self.s().f64_add();
        } else {
            self.s().f32_add();
        }
        self.s().local_get(T);
        one(self, 1.0);
        if d {
            self.s().f64_sub();
        } else {
            self.s().f32_sub();
        }
        self.s().local_get(T);
        // x - t <= -0.5 ? t - 1 : t
        self.s().local_get(X).local_get(T);
        if d {
            self.s().f64_sub();
        } else {
            self.s().f32_sub();
        }
        one(self, -0.5);
        if d {
            self.s().f64_le();
        } else {
            self.s().f32_le();
        }
        self.s().select();
        // x - t >= 0.5 ? t + 1 : that
        self.s().local_get(X).local_get(T);
        if d {
            self.s().f64_sub();
        } else {
            self.s().f32_sub();
        }
        one(self, 0.5);
        if d {
            self.s().f64_ge();
        } else {
            self.s().f32_ge();
        }
        self.s().select();
    }

    /// `SCVTF`/`UCVTF` from a fixed-point general register and `FCVTZS`/`FCVTZU` to one, as the
    /// interpreter computes them: through a double, scaled by 2^fbits (exact).
    #[allow(clippy::too_many_arguments)]
    fn fp_fixed(
        &mut self,
        sf: bool,
        ftype: u8,
        rmode: u8,
        opcode: u8,
        fbits: u8,
        rd: u8,
        rn: u8,
    ) -> bool {
        let Some(d) = double(ftype) else { return false };
        if fbits > 64 {
            return false;
        }
        let scale = f64::from_bits(u64::from(1023 + u32::from(fbits)) << 52);
        match (rmode, opcode) {
            (0, 2 | 3) => {
                let signed = opcode == 2;
                if rn == ZR {
                    self.s().i64_const(0);
                } else {
                    self.get(rn);
                }
                let b = self.s();
                if !sf {
                    b.i32_wrap_i64();
                }
                match (sf, signed) {
                    (false, true) => b.f64_convert_i32_s(),
                    (false, false) => b.f64_convert_i32_u(),
                    (true, true) => b.f64_convert_i64_s(),
                    (true, false) => b.f64_convert_i64_u(),
                };
                b.i64_const(scale.to_bits() as i64)
                    .f64_reinterpret_i64()
                    .f64_div();
                if d {
                    b.i64_reinterpret_f64();
                } else {
                    b.f32_demote_f64().i32_reinterpret_f32().i64_extend_i32_u();
                }
                self.bits_store(rd);
                true
            }
            (3, 0 | 1) => {
                let signed = opcode == 0;
                self.fload(rn, d);
                if !d {
                    self.s().f64_promote_f32();
                }
                let b = self.s();
                b.i64_const(scale.to_bits() as i64)
                    .f64_reinterpret_i64()
                    .f64_mul()
                    .f64_trunc();
                match (sf, signed) {
                    (false, true) => {
                        b.i32_trunc_sat_f64_s().i64_extend_i32_u();
                    }
                    (false, false) => {
                        b.i32_trunc_sat_f64_u().i64_extend_i32_u();
                    }
                    (true, true) => {
                        b.i64_trunc_sat_f64_s();
                    }
                    (true, false) => {
                        b.i64_trunc_sat_f64_u();
                    }
                }
                self.set(rd);
                true
            }
            _ => false,
        }
    }

    /// Pushes the scalar of `Vn` as a float (`d`: double).
    fn fload(&mut self, n: VReg, d: bool) {
        self.s().local_get(P_CPU);
        if d {
            self.s().i64_load(mem(voff(n), 3)).f64_reinterpret_i64();
        } else {
            self.s().i32_load(mem(voff(n), 2)).f32_reinterpret_i32();
        }
    }

    /// Pushes +0.0.
    fn fzero(&mut self, d: bool) {
        if d {
            self.s().i64_const(0).f64_reinterpret_i64();
        } else {
            self.s().i32_const(0).f32_reinterpret_i32();
        }
    }

    /// Pushes the scalar bits of `Vn` (zero-extended to i64).
    fn bits_load(&mut self, n: VReg, d: bool) {
        self.s().local_get(P_CPU);
        if d {
            self.s().i64_load(mem(voff(n), 3));
        } else {
            self.s().i64_load32_u(mem(voff(n), 2));
        }
    }

    /// `Vd` = the i64 on the stack, every bit above it cleared (the interpreter's `set_scalar`).
    pub(super) fn bits_store(&mut self, d: VReg) {
        self.s()
            .local_set(L_T1)
            .local_get(P_CPU)
            .local_get(L_T1)
            .i64_store(mem(voff(d), 3))
            .local_get(P_CPU)
            .i64_const(0)
            .i64_store(mem(voff(d) + 8, 3));
    }

    /// The float on the stack is the instruction's result: stores it in `Vd` (negated with
    /// `neg`), unless it is a NaN, which the interpreter produces instead.
    fn fresult(&mut self, k: u32, pc: u64, inst: &Inst, rd: VReg, d: bool, neg: bool) {
        self.s().local_tee(L_T0).local_get(L_T0);
        if d {
            self.s().f64_ne();
        } else {
            self.s().f32_ne();
        }
        self.b.if_(None);
        self.step(k, pc, inst);
        self.b.else_();
        self.s().local_get(L_T0);
        if neg {
            if d {
                self.s().f64_neg();
            } else {
                self.s().f32_neg();
            }
        }
        if d {
            self.s().i64_reinterpret_f64();
        } else {
            self.s().i32_reinterpret_f32().i64_extend_i32_u();
        }
        self.bits_store(rd);
        self.b.end();
    }

    /// NZCV from comparing `T0` with `T1` (floats): less 1000, equal 0110, greater 0010,
    /// unordered 0011 (the interpreter's `fp::compare`).
    fn fcmp_flags(&mut self, d: bool) {
        let cmp = |e: &mut Self, op: u8| {
            e.s().local_get(L_T0).local_get(L_T1);
            let b = e.s();
            match (op, d) {
                (0, false) => b.f32_lt(),
                (1, false) => b.f32_eq(),
                (2, false) => b.f32_gt(),
                (0, true) => b.f64_lt(),
                (1, true) => b.f64_eq(),
                _ => b.f64_gt(),
            };
        };
        // N = less
        cmp(self, 0);
        self.s().i64_extend_i32_u();
        // Z = equal
        cmp(self, 1);
        self.s().i64_extend_i32_u().i64_const(8).i64_shl().i64_or();
        // C = not less (equal, greater or unordered)
        cmp(self, 0);
        self.s()
            .i32_const(1)
            .i32_xor()
            .i64_extend_i32_u()
            .i64_const(16)
            .i64_shl()
            .i64_or();
        // V = unordered (neither less, equal nor greater)
        cmp(self, 0);
        cmp(self, 1);
        self.s().i32_or();
        cmp(self, 2);
        self.s()
            .i32_or()
            .i32_const(1)
            .i32_xor()
            .i64_extend_i32_u()
            .i64_const(24)
            .i64_shl()
            .i64_or()
            .local_set(L_FR)
            .i32_const(kind::PACKED)
            .local_set(L_FK);
        self.kind = Some(kind::PACKED);
    }

    /// `FCSEL`: the scalar of `Vn` if `cond` holds, else of `Vm` (bits, no check needed).
    fn fcsel(&mut self, d: bool, rd: VReg, rn: VReg, rm: VReg, cond: Cond) {
        self.cond(cond);
        self.s().local_set(L_W0);
        self.bits_load(rn, d);
        self.bits_load(rm, d);
        self.s().local_get(L_W0).select();
        self.bits_store(rd);
    }

    /// FP data processing with one source.
    #[allow(clippy::too_many_arguments)]
    fn fp1(
        &mut self,
        k: u32,
        pc: u64,
        inst: &Inst,
        ftype: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
    ) -> bool {
        // FCVT between single and double (half precision: the interpreter).
        if opcode & 0b11_1100 == 0b00_0100 {
            let (Some(from), Some(to)) = (double(ftype), double(opcode & 3)) else {
                return false;
            };
            if from == to {
                return false;
            }
            self.fload(rn, from);
            if to {
                self.s().f64_promote_f32();
            } else {
                self.s().f32_demote_f64();
            }
            self.fresult(k, pc, inst, rd, to, false);
            return true;
        }
        let Some(d) = double(ftype) else { return false };
        let sign: u64 = if d { 1 << 63 } else { 1 << 31 };
        match opcode {
            // FMOV, FABS, FNEG: bits.
            0b00_0000..=0b00_0010 => {
                self.bits_load(rn, d);
                match opcode {
                    1 => {
                        self.s().i64_const(!sign as i64).i64_and();
                    }
                    2 => {
                        self.s().i64_const(sign as i64).i64_xor();
                    }
                    _ => {}
                }
                self.bits_store(rd);
                true
            }
            // FSQRT, FRINTN, FRINTP, FRINTM, FRINTZ, FRINTX, FRINTI (round to nearest even:
            // the interpreter ignores FPCR's rounding mode). FRINTA (ties away): the interpreter.
            0b00_0011 | 0b00_1000..=0b00_1011 | 0b00_1110 | 0b00_1111 => {
                self.fload(rn, d);
                let b = self.s();
                match (opcode, d) {
                    (0b00_0011, false) => b.f32_sqrt(),
                    (0b00_1001, false) => b.f32_ceil(),
                    (0b00_1010, false) => b.f32_floor(),
                    (0b00_1011, false) => b.f32_trunc(),
                    (_, false) => b.f32_nearest(),
                    (0b00_0011, true) => b.f64_sqrt(),
                    (0b00_1001, true) => b.f64_ceil(),
                    (0b00_1010, true) => b.f64_floor(),
                    (0b00_1011, true) => b.f64_trunc(),
                    (_, true) => b.f64_nearest(),
                };
                self.fresult(k, pc, inst, rd, d, false);
                true
            }
            // FRINTA: ties away from zero.
            0b00_1100 => {
                self.fload(rn, d);
                self.round_away(d);
                self.fresult(k, pc, inst, rd, d, false);
                true
            }
            _ => false,
        }
    }

    /// Conversions between FP and general registers (the `FMOV`s are `fmov_general`'s).
    #[allow(clippy::too_many_arguments)]
    fn fp_int(&mut self, sf: bool, ftype: u8, rmode: u8, opcode: u8, rd: u8, rn: u8) -> bool {
        let Some(d) = double(ftype) else { return false };
        match (rmode, opcode) {
            // SCVTF, UCVTF: round to nearest even, never a NaN.
            (0, 0b010 | 0b011) => {
                let signed = opcode == 0b010;
                if rn == ZR {
                    self.s().i64_const(0);
                } else {
                    self.get(rn);
                }
                let b = self.s();
                if !sf {
                    b.i32_wrap_i64();
                }
                match (d, sf, signed) {
                    (false, false, true) => b.f32_convert_i32_s(),
                    (false, false, false) => b.f32_convert_i32_u(),
                    (false, true, true) => b.f32_convert_i64_s(),
                    (false, true, false) => b.f32_convert_i64_u(),
                    (true, false, true) => b.f64_convert_i32_s(),
                    (true, false, false) => b.f64_convert_i32_u(),
                    (true, true, true) => b.f64_convert_i64_s(),
                    (true, true, false) => b.f64_convert_i64_u(),
                };
                if d {
                    self.s().i64_reinterpret_f64();
                } else {
                    self.s().i32_reinterpret_f32().i64_extend_i32_u();
                }
                self.bits_store(rd);
                true
            }
            // FCVTNS/NU (nearest), FCVTPS/PU (up), FCVTMS/MU (down), FCVTZS/ZU (zero): round in
            // double (as the interpreter does), then convert saturating, NaN to 0 (Wasm's
            // `trunc_sat` is exactly that).
            // FCVTAS/AU (rmode 0, opcodes 4 and 5): ties away from zero.
            (0..=3, 0b000 | 0b001) | (0, 0b100 | 0b101) => {
                let signed = opcode & 1 == 0;
                self.fload(rn, d);
                if !d {
                    self.s().f64_promote_f32();
                }
                if opcode & 0b100 != 0 {
                    self.round_away(true);
                }
                let b = self.s();
                if opcode & 0b100 == 0 {
                    match rmode {
                        0 => b.f64_nearest(),
                        1 => b.f64_ceil(),
                        2 => b.f64_floor(),
                        _ => b.f64_trunc(),
                    };
                }
                match (sf, signed) {
                    (false, true) => {
                        b.i32_trunc_sat_f64_s().i64_extend_i32_u();
                    }
                    (false, false) => {
                        b.i32_trunc_sat_f64_u().i64_extend_i32_u();
                    }
                    (true, true) => {
                        b.i64_trunc_sat_f64_s();
                    }
                    (true, false) => {
                        b.i64_trunc_sat_f64_u();
                    }
                }
                self.set(rd);
                true
            }
            _ => false,
        }
    }
}
