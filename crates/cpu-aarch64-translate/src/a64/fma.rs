//! Fused multiply-add (`FMADD`, `FMSUB`, `FNMADD`, `FNMSUB`) in translated code. Wasm has no
//! deterministic fused operation, so the one rounding of
//! `a + n * m` is emulated exactly with error-free transformations and rounding to odd
//! (Boldo and Melquiond, "Emulation of FMA and correctly rounded sums: proved algorithms using
//! rounding to odd", IEEE TC 57(4), 2008):
//!
//! * single precision: the product is exact in a double; `RO(p + a)` (the double sum rounded
//!   to odd, from `TwoSum`'s error) then rounded to single is the correctly rounded result
//!   (53 >= 2 * 24 + 2);
//! * double precision: `(uh, ul) = ExactMult(n, m)` (Dekker's product with Veltkamp's split),
//!   `(th, tl) = TwoSum(a, ul)`, `(vh, vl) = TwoSum(uh, th)`, `w = RO(tl + vl)`, result
//!   `RN(vh + w)`. Exact when nothing overflows or underflows: the operands must lie in
//!   `[2^-400, 2^400]` (the addend: zero or `[2^-900, 2^900]`); a zero product is the plain
//!   `a + n * m` (exact); anything else (infinities, NaNs, extreme exponents) is handed to the
//!   interpreter.
//!
//! The interpreter's `fma` (libm) is correctly rounded with Arm's NaN rules, so for every case
//! translated here the two agree bit for bit.

use wasmshim_aarch64::{Inst, VReg};

use super::simd::voff;
use super::{A64, P_CPU, mem};
use crate::ir::PureOps;

/// `2^e` exactly (no `std` here).
fn pow2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

// Temporaries (f64 unless said).
const X: u32 = 90;
const Y: u32 = 91;
const Z: u32 = 92;
const UH: u32 = 93;
const UL: u32 = 94;
const XH: u32 = 95;
const XL: u32 = 96;
const YH: u32 = 97;
const YL: u32 = 98;
const C: u32 = 99;
const TH: u32 = 100;
const TL: u32 = 101;
const VH: u32 = 102;
const VL: u32 = 103;
const BB: u32 = 104;
const S: u32 = 105;
const E: u32 = 106;
const IB: u32 = 107; // i64
const R: u32 = 108;

impl A64 {
    fn f64c(&mut self, v: f64) {
        self.s().i64_const(v.to_bits() as i64).f64_reinterpret_i64();
    }

    /// `(s, e) = TwoSum(a, b)`: `s = RN(a + b)`, `a + b = s + e` exactly.
    fn two_sum(&mut self, a: u32, b: u32, s: u32, e: u32) {
        self.s().local_get(a).local_get(b).f64_add().local_set(s);
        self.s().local_get(s).local_get(a).f64_sub().local_set(BB);
        // e = (a - (s - bb)) + (b - bb)
        self.s()
            .local_get(a)
            .local_get(s)
            .local_get(BB)
            .f64_sub()
            .f64_sub()
            .local_get(b)
            .local_get(BB)
            .f64_sub()
            .f64_add()
            .local_set(e);
    }

    /// Veltkamp's split: `a = hi + lo`, each with at most 26 significant bits.
    fn split(&mut self, a: u32, hi: u32, lo: u32) {
        self.f64c(134_217_729.0);
        self.s().local_get(a).f64_mul().local_set(C);
        self.s()
            .local_get(C)
            .local_get(C)
            .local_get(a)
            .f64_sub()
            .f64_sub()
            .local_set(hi);
        self.s().local_get(a).local_get(hi).f64_sub().local_set(lo);
    }

    /// Pushes `RO(a + b)`: the sum rounded to odd (from `RN` and `TwoSum`'s error: an inexact
    /// sum with an even last bit moves one unit towards the exact value).
    fn round_odd_sum(&mut self, a: u32, b: u32) {
        self.two_sum(a, b, S, E);
        self.s().local_get(S).i64_reinterpret_f64().local_set(IB);
        // bits ± 1 (towards the exact value: + when e and s have the same sign)
        self.s().local_get(IB).i64_const(1).i64_const(-1);
        self.s().local_get(E);
        self.f64c(0.0);
        self.s().f64_gt().local_get(S);
        self.f64c(0.0);
        self.s()
            .f64_gt()
            .i32_eq()
            .select()
            .i64_add()
            .f64_reinterpret_i64();
        self.s().local_get(S);
        // inexact and even
        self.s().local_get(E);
        self.f64c(0.0);
        self.s()
            .f64_ne()
            .local_get(IB)
            .i64_const(1)
            .i64_and()
            .i64_eqz()
            .i32_and()
            .select();
    }

    /// Pushes whether `lo <= |v| <= hi` (i32).
    fn in_range(&mut self, v: u32, lo: f64, hi: f64) {
        self.s().local_get(v).f64_abs();
        self.f64c(lo);
        self.s().f64_ge().local_get(v).f64_abs();
        self.f64c(hi);
        self.s().f64_le().i32_and();
    }

    /// `FMADD` and its negated forms (`o1`, `o0` as the interpreter reads them).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fused(
        &mut self,
        k: u32,
        pc: u64,
        inst: &Inst,
        d: bool,
        o1: bool,
        o0: bool,
        rd: VReg,
        rn: VReg,
        rm: VReg,
        ra: VReg,
    ) {
        // Operands as doubles (singles are promoted exactly), signs flipped as the interpreter
        // does: n negated when o0 != o1, a when o1.
        let load = |e: &mut Self, r: VReg, neg: bool, t: u32| {
            e.s().local_get(P_CPU);
            if d {
                e.s().i64_load(mem(voff(r), 3)).f64_reinterpret_i64();
            } else {
                e.s()
                    .i32_load(mem(voff(r), 2))
                    .f32_reinterpret_i32()
                    .f64_promote_f32();
            }
            if neg {
                e.s().f64_neg();
            }
            e.s().local_set(t);
        };
        load(self, rn, o0 != o1, X);
        load(self, rm, false, Y);
        load(self, ra, o1, Z);
        if !d {
            // p = n * m exactly; RO(p + a); one rounding to single. A non-finite result (an
            // infinite or NaN operand, an overflow) is the interpreter's.
            self.s().local_get(X).local_get(Y).f64_mul().local_set(UH);
            self.round_odd_sum(UH, Z);
            self.s().f32_demote_f64().local_set(R);
            self.s().local_get(R).f32_abs();
            self.s()
                .i32_const(f32::INFINITY.to_bits() as i32)
                .f32_reinterpret_i32()
                .f32_lt();
            self.b.if_(None);
            self.s()
                .local_get(R)
                .i32_reinterpret_f32()
                .i64_extend_i32_u();
            self.bits_store(rd);
            self.b.else_();
            self.step(k, pc, inst);
            self.b.end();
            return;
        }
        // Double: the exact path's range.
        self.in_range(X, pow2(-400), pow2(400));
        self.in_range(Y, pow2(-400), pow2(400));
        self.s().i32_and();
        self.in_range(Z, pow2(-900), pow2(900));
        self.s().local_get(Z);
        self.f64c(0.0);
        self.s().f64_eq().i32_or().i32_and();
        self.b.if_(None);
        {
            self.s().local_get(X).local_get(Y).f64_mul().local_set(UH);
            self.split(X, XH, XL);
            self.split(Y, YH, YL);
            // ul = ((xh*yh - uh) + xh*yl + xl*yh) + xl*yl
            self.s()
                .local_get(XH)
                .local_get(YH)
                .f64_mul()
                .local_get(UH)
                .f64_sub()
                .local_get(XH)
                .local_get(YL)
                .f64_mul()
                .f64_add()
                .local_get(XL)
                .local_get(YH)
                .f64_mul()
                .f64_add()
                .local_get(XL)
                .local_get(YL)
                .f64_mul()
                .f64_add()
                .local_set(UL);
            self.two_sum(Z, UL, TH, TL);
            self.two_sum(UH, TH, VH, VL);
            self.round_odd_sum(TL, VL);
            self.s().local_set(R).local_get(VH).local_get(R).f64_add();
            self.s().i64_reinterpret_f64();
            self.bits_store(rd);
        }
        self.b.else_();
        {
            // A zero product with finite operands: a + (±0), exact in Wasm.
            let finite = |e: &mut Self, v: u32| {
                e.s().local_get(v).f64_abs();
                e.f64c(f64::INFINITY);
                e.s().f64_lt();
            };
            self.s().local_get(X);
            self.f64c(0.0);
            self.s().f64_eq().local_get(Y);
            self.f64c(0.0);
            self.s().f64_eq().i32_or();
            finite(self, X);
            self.s().i32_and();
            finite(self, Y);
            self.s().i32_and();
            finite(self, Z);
            self.s().i32_and();
            self.b.if_(None);
            self.s()
                .local_get(Z)
                .local_get(X)
                .local_get(Y)
                .f64_mul()
                .f64_add()
                .i64_reinterpret_f64();
            self.bits_store(rd);
            self.b.else_();
            self.step(k, pc, inst);
            self.b.end();
        }
        self.b.end();
    }
}
