//! Advanced SIMD arithmetic in translated code: floating point lanes, integer add, multiply,
//! min/max, saturating arithmetic, widening and narrowing, shifts by immediate and reductions
//! across lanes.
//!
//! Every case mirrors the interpreter (`wasmshim-aarch64`'s `simd_exec.rs`) for the encodings it
//! accepts, including which bits of `Vd` a result clears. Float results follow `fp.rs`: Wasm's
//! lane operations give the interpreter's IEEE results for non-NaN lanes, so a NaN in any lane
//! the instruction writes hands the whole instruction to the interpreter (Arm's NaN rules).
//! Saturating integer operations compute the wrapping result and hand the instruction over when
//! a lane saturated (the interpreter then also sets FPSR.QC). Compares, moves and bit operations
//! need no check. Everything not listed (fused multiply-add, rounding shifts, saturating
//! narrows, 64-bit lanes Wasm has no operation for, ties-away rounding) still steps.

use wasmshim_aarch64::simd::SimdInst;
use wasmshim_aarch64::{Inst, VReg};

use super::simd::{LOW64, Lanes, shuffle, voff};
use super::{A64, P_CPU, mem};
use crate::ir::PureOps;

// Temporaries (the IR builder's names for values).
const T_R: u32 = 70;
const T_A: u32 = 71;
const T_B: u32 = 72;
const T_I: u32 = 73;

/// The bits of `Vd` a result writes (the rest are cleared): all, the low half, or one element.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Keep {
    Q,
    Low,
    Scalar(u32),
}

impl Keep {
    fn of(scalar: bool, q: bool, esize: u32) -> Self {
        if scalar {
            Keep::Scalar(esize)
        } else if q {
            Keep::Q
        } else {
            Keep::Low
        }
    }

    fn mask(self) -> Option<i128> {
        match self {
            Keep::Q => None,
            Keep::Low | Keep::Scalar(64) => Some(LOW64),
            Keep::Scalar(b) => Some((1i128 << b) - 1),
        }
    }
}

/// `value` (of `esize` bits) in every lane.
fn splat(esize: u32, value: u64) -> i128 {
    let m = if esize == 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    let mut r = 0u128;
    for i in 0..128 / esize {
        r |= u128::from(value & m) << (i * esize);
    }
    r as i128
}

/// Float lane operations.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fop {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Abd,
}

/// Float lane comparisons (all ones where true).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fcmp {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
}

/// Integer lane operations.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Iop {
    Add,
    Sub,
    Mul,
    MaxS,
    MaxU,
    MinS,
    MinU,
}

impl A64 {
    /// Handles an Advanced SIMD instruction of this module natively; `false`: use `jit_step`.
    pub(super) fn neon(&mut self, k: u32, pc: u64, inst: &Inst, s: &SimdInst) -> bool {
        match *s {
            SimdInst::ThreeSame {
                scalar,
                q,
                u,
                size,
                opcode,
                rd,
                rn,
                rm,
            } => {
                if opcode >= 0b11000 {
                    self.fp_three(k, pc, inst, scalar, q, u, size, opcode, rd, rn, rm)
                } else if !scalar {
                    self.int_three(k, pc, inst, q, u, size, opcode, rd, rn, rm)
                } else {
                    false
                }
            }
            SimdInst::TwoMisc {
                scalar,
                q,
                u,
                size,
                opcode,
                rd,
                rn,
            } => {
                if opcode >= 0b01100 && !matches!(opcode, 0b10010..=0b10100) {
                    self.fp_two(k, pc, inst, scalar, q, u, size, opcode, rd, rn)
                } else {
                    self.int_two(scalar, q, u, size, opcode, rd, rn)
                }
            }
            SimdInst::ShiftImm {
                scalar,
                q,
                u,
                immh,
                immb,
                opcode,
                rd,
                rn,
            } => self.shift_imm(scalar, q, u, immh, immb, opcode, rd, rn),
            SimdInst::AcrossLanes {
                q,
                u,
                size,
                opcode,
                rd,
                rn,
            } => self.across(k, pc, inst, q, u, size, opcode, rd, rn),
            SimdInst::LdStSingle {
                load,
                replicate,
                q,
                esize_log,
                index,
                selem,
                rt,
                rn,
                post,
            } => self.ldst_single(
                k, pc, load, replicate, q, esize_log, index, selem, rt, rn, post,
            ),
            SimdInst::LdStMulti {
                load,
                q,
                size,
                regs,
                selem,
                rt,
                rn,
                post,
            } => {
                if !(1..=4).contains(&regs) || !(1..=4).contains(&selem) || size > 3 {
                    return false;
                }
                let per = if q { 16u64 } else { 8 };
                let total = per * u64::from(regs);
                if selem > 1 {
                    // LD2-LD4/ST2-ST4: element by element, in the interpreter's order (structure
                    // `e`'s member `s` is element `e` of register `rt + r + s`).
                    let ebytes = 1u8 << size;
                    let elements = per / u64::from(ebytes);
                    let rpt = regs / selem;
                    self.simd_mem(k, pc, load, rn, post, total, |e| {
                        let mut off = 0u64;
                        for r in 0..rpt {
                            for el in 0..elements {
                                for s in 0..selem {
                                    let reg = (rt + r + s) % 32;
                                    let at = voff(reg) + el * u64::from(ebytes);
                                    if load {
                                        e.s().local_get(P_CPU).local_get(super::L_H);
                                        e.load_int_at(super::MemOp::Load, ebytes, off);
                                        e.store_int_at(ebytes, at);
                                    } else {
                                        e.s().local_get(super::L_H).local_get(P_CPU);
                                        e.load_int_at_cpu(ebytes, at);
                                        e.store_int_at(ebytes, off);
                                    }
                                    off += u64::from(ebytes);
                                }
                            }
                        }
                        if load && !q {
                            // The 64-bit forms clear the registers' upper halves.
                            for i in 0..regs {
                                e.s()
                                    .local_get(P_CPU)
                                    .i64_const(0)
                                    .i64_store(mem(voff((rt + i) % 32) + 8, 3));
                            }
                        }
                    });
                    return true;
                }
                self.simd_mem(k, pc, load, rn, post, total, |e| {
                    for i in 0..regs {
                        let vbase = voff((rt + i) % 32);
                        let off = per * u64::from(i);
                        if load {
                            e.load_vector(vbase, per as u8, off);
                        } else {
                            e.store_vector(vbase, per as u8, off);
                        }
                    }
                });
                true
            }
            SimdInst::Tbl {
                q,
                len,
                tbx,
                rd,
                rn,
                rm,
            } => {
                // TBL/TBX with one to four table registers: one swizzle per register,
                // its indices moved down by 16 per register before it (a swizzle gives 0 for
                // indices of 16 and up, so exactly one register supplies each in-range lane);
                // TBX keeps Vd's lanes whose index is past the table.
                if len > 3 {
                    return false;
                }
                const IDX: u32 = 90;
                self.vload(rm);
                self.s().local_set(IDX);
                if tbx {
                    self.vload(rd);
                }
                for k in 0..=len {
                    self.vload((rn + k) % 32);
                    self.s().local_get(IDX);
                    if k > 0 {
                        self.s().v128_const(splat(8, u64::from(16 * k))).i8x16_sub();
                    }
                    self.s().i8x16_swizzle();
                    if k > 0 {
                        self.s().v128_or();
                    }
                }
                if tbx {
                    // [d, table result] -> result where idx < limit, else d
                    let limit = 16 * (u64::from(len) + 1);
                    self.s().local_set(T_A).local_set(T_B);
                    self.s().local_get(T_A).local_get(T_B);
                    self.s()
                        .v128_const(splat(8, limit))
                        .local_get(IDX)
                        .i8x16_gt_u()
                        .v128_bitselect();
                }
                self.vput(rd, Keep::of(false, q, 8));
                true
            }
            SimdInst::ScalarDup { imm5, rd, rn } => {
                // DUP (scalar, element): element `index` of Vn, every other bit cleared.
                let size = imm5.trailing_zeros();
                if size > 3 {
                    return false;
                }
                let lanes = Lanes::of(size as u8);
                self.vdup_lane(rn, lanes, usize::from(imm5 >> (size + 1)));
                self.vput(rd, Keep::Scalar(8 << size));
                true
            }
            SimdInst::ScalarPairwise {
                u,
                size,
                opcode,
                rd,
                rn,
            } => self.scalar_pairwise(k, pc, inst, u, size, opcode, rd, rn),
            SimdInst::ThreeDiff {
                q,
                u,
                size,
                opcode,
                rd,
                rn,
                rm,
            } => self.three_diff(q, u, size, opcode, rd, rn, rm),
            SimdInst::Indexed {
                scalar,
                q,
                u,
                size,
                opcode,
                l,
                m,
                h,
                rd,
                rn,
                rm,
            } => self.indexed(
                k,
                pc,
                inst,
                scalar,
                q,
                u,
                size,
                opcode,
                (l, m, h),
                rd,
                rn,
                rm,
            ),
            _ => false,
        }
    }

    // ---- structure loads and stores ----

    /// A SIMD structure access of `total` bytes at `Xn` (SP must be 16-byte aligned), then the
    /// post-index (`Some(ZR)`: by `total`, else by `Xm`). `fast` moves the data given the host
    /// address; a TLB miss or a page crossing runs the instruction in the interpreter.
    #[allow(clippy::too_many_arguments)]
    fn simd_mem(
        &mut self,
        k: u32,
        pc: u64,
        load: bool,
        rn: u8,
        post: Option<u8>,
        total: u64,
        fast: impl Fn(&mut Self),
    ) {
        self.get(rn);
        self.s().local_set(super::L_A);
        let mut inputs = super::bit(rn);
        let mut outputs = 0;
        if let Some(rm) = post {
            inputs |= super::bit(rm);
            outputs |= super::bit(rn);
        }
        let table = if load {
            crate::abi::TLB_READ
        } else {
            crate::abi::TLB_WRITE
        };
        self.guarded(
            k,
            pc,
            table,
            total,
            super::check_sp(rn),
            inputs,
            outputs,
            |e| {
                fast(e);
                if let Some(rm) = post {
                    e.s().local_get(super::L_A);
                    if rm == wasmshim_aarch64::ZR {
                        e.s().i64_const(total as i64);
                    } else {
                        e.get(rm);
                    }
                    e.s().i64_add();
                    e.set(rn);
                }
            },
        );
    }

    /// `LD1`-`LD4` (single structure, one lane or replicated) and `ST1`-`ST4` (one lane).
    #[allow(clippy::too_many_arguments)]
    fn ldst_single(
        &mut self,
        k: u32,
        pc: u64,
        load: bool,
        replicate: bool,
        q: bool,
        esize_log: u8,
        index: u8,
        selem: u8,
        rt: VReg,
        rn: u8,
        post: Option<u8>,
    ) -> bool {
        if esize_log > 3 || !(1..=4).contains(&selem) {
            return false;
        }
        let ebytes = 1u8 << esize_log;
        let lanes = Lanes::of(esize_log);
        let total = u64::from(ebytes) * u64::from(selem);
        self.simd_mem(k, pc, load, rn, post, total, |e| {
            for s in 0..selem {
                let reg = (rt + s) % 32;
                let off = u64::from(s) * u64::from(ebytes);
                if !load {
                    // The lane's bytes from the register file to memory.
                    e.s().local_get(super::L_H).local_get(P_CPU);
                    e.load_int_at_cpu(ebytes, voff(reg) + u64::from(index) * u64::from(ebytes));
                    e.store_int_at(ebytes, off);
                } else if replicate {
                    e.s().local_get(P_CPU).local_get(super::L_H);
                    e.load_int_at(super::MemOp::Load, ebytes, off);
                    let s = e.s();
                    match lanes {
                        Lanes::D => s.i64x2_splat(),
                        Lanes::S => s.i32_wrap_i64().i32x4_splat(),
                        Lanes::H => s.i32_wrap_i64().i16x8_splat(),
                        Lanes::B => s.i32_wrap_i64().i8x16_splat(),
                    };
                    if !q {
                        s.v128_const(LOW64).v128_and();
                    }
                    s.v128_store(mem(voff(reg), 0));
                } else {
                    // One lane of the register, the others kept.
                    e.s().local_get(P_CPU).local_get(super::L_H);
                    e.load_int_at(super::MemOp::Load, ebytes, off);
                    e.store_int_at(ebytes, voff(reg) + u64::from(index) * u64::from(ebytes));
                }
            }
        });
        true
    }

    /// Pushes `bytes` (zero-extended to i64) read at `offset` from the address on the stack.
    fn load_int_at_cpu(&mut self, bytes: u8, offset: u64) {
        let m = mem(offset, 0);
        match bytes {
            1 => self.s().i64_load8_u(m),
            2 => self.s().i64_load16_u(m),
            4 => self.s().i64_load32_u(m),
            _ => self.s().i64_load(m),
        };
    }

    /// Sets FPSR.QC (cumulative saturation) if the v128 on the stack is nonzero.
    fn set_qc_if_any(&mut self) {
        let fpsr = wasmshim_aarch64::layout::FPSR as u64;
        self.s().v128_any_true();
        self.b.if_(None);
        self.s()
            .local_get(P_CPU)
            .local_get(P_CPU)
            .i64_load(mem(fpsr, 3))
            .i64_const(1 << 27)
            .i64_or()
            .i64_store(mem(fpsr, 3));
        self.b.end();
    }

    /// `op` (`MinS`, `MaxS` or `MinU`) of each `lanes` element of the v128 on the stack and the
    /// constant `c`; for doublewords (no Wasm min/max) by compare and `bitselect`.
    fn vclamp(&mut self, lanes: Lanes, op: Iop, c: i64) {
        let esize = 8 * lanes.bytes() as u32;
        let cv = splat(esize, c as u64);
        if Self::has_iop(op, lanes) {
            self.s().v128_const(cv);
            self.iop(op, lanes);
            return;
        }
        const X: u32 = 89;
        let min = splat(64, 1 << 63);
        self.s().local_set(X).v128_const(cv).local_get(X);
        match op {
            // x > c: c, else x
            Iop::MinS => {
                self.s().local_get(X).v128_const(cv).i64x2_gt_s();
            }
            // c > x: c, else x
            Iop::MaxS => {
                self.s().v128_const(cv).local_get(X).i64x2_gt_s();
            }
            // x >u c: c, else x
            _ => {
                self.s()
                    .local_get(X)
                    .v128_const(min)
                    .v128_xor()
                    .v128_const(cv ^ min)
                    .i64x2_gt_s();
            }
        }
        self.s().v128_bitselect();
    }

    // ---- results ----

    /// Stores the v128 on the stack in `Vd`, every bit outside `keep` cleared.
    fn vput(&mut self, d: VReg, keep: Keep) {
        self.s().local_set(T_R).local_get(P_CPU).local_get(T_R);
        if let Some(m) = keep.mask() {
            self.s().v128_const(m).v128_and();
        }
        self.s().v128_store(mem(voff(d), 0));
    }

    /// The v128 on the stack holds float lanes (`dbl`: f64): stored in `Vd` unless a lane in
    /// `keep` is a NaN, which the interpreter produces instead.
    #[allow(clippy::too_many_arguments)]
    fn vfresult(&mut self, k: u32, pc: u64, inst: &Inst, d: VReg, keep: Keep, dbl: bool) {
        self.s().local_tee(T_R).local_get(T_R).local_get(T_R);
        if dbl {
            self.s().f64x2_ne();
        } else {
            self.s().f32x4_ne();
        }
        self.guarded_put(k, pc, inst, d, keep);
    }

    /// Below the top of the stack, the result; on top, a v128 that is nonzero in a lane where
    /// the interpreter must decide (a NaN, a saturation). Stores the result in `Vd` if no lane
    /// in `keep` is set, else steps.
    fn guarded_put(&mut self, k: u32, pc: u64, inst: &Inst, d: VReg, keep: Keep) {
        if let Some(m) = keep.mask() {
            self.s().v128_const(m).v128_and();
        }
        self.s().v128_any_true().local_set(T_I);
        self.s().local_set(T_R);
        self.s().local_get(T_I);
        self.b.if_(None);
        self.step(k, pc, inst);
        self.b.else_();
        self.s().local_get(T_R);
        self.vput(d, keep);
        self.b.end();
    }

    // ---- lane operations ----

    fn fop(&mut self, op: Fop, dbl: bool) {
        let s = self.s();
        match (op, dbl) {
            (Fop::Add, false) => s.f32x4_add(),
            (Fop::Sub | Fop::Abd, false) => s.f32x4_sub(),
            (Fop::Mul, false) => s.f32x4_mul(),
            (Fop::Div, false) => s.f32x4_div(),
            (Fop::Max, false) => s.f32x4_max(),
            (Fop::Min, false) => s.f32x4_min(),
            (Fop::Add, true) => s.f64x2_add(),
            (Fop::Sub | Fop::Abd, true) => s.f64x2_sub(),
            (Fop::Mul, true) => s.f64x2_mul(),
            (Fop::Div, true) => s.f64x2_div(),
            (Fop::Max, true) => s.f64x2_max(),
            (Fop::Min, true) => s.f64x2_min(),
        };
        if op == Fop::Abd {
            if dbl {
                s.f64x2_abs();
            } else {
                s.f32x4_abs();
            }
        }
    }

    fn fcmp(&mut self, c: Fcmp, dbl: bool) {
        let s = self.s();
        match (c, dbl) {
            (Fcmp::Eq, false) => s.f32x4_eq(),
            (Fcmp::Ge, false) => s.f32x4_ge(),
            (Fcmp::Gt, false) => s.f32x4_gt(),
            (Fcmp::Le, false) => s.f32x4_le(),
            (Fcmp::Lt, false) => s.f32x4_lt(),
            (Fcmp::Eq, true) => s.f64x2_eq(),
            (Fcmp::Ge, true) => s.f64x2_ge(),
            (Fcmp::Gt, true) => s.f64x2_gt(),
            (Fcmp::Le, true) => s.f64x2_le(),
            (Fcmp::Lt, true) => s.f64x2_lt(),
        };
    }

    /// `FRINTA` on every lane of the v128 on the stack: [`A64::round_away`]'s steps, the
    /// choices made with `bitselect` (a zero keeps its sign).
    fn vround_away(&mut self, dbl: bool) {
        const X: u32 = 82;
        const T: u32 = 83;
        const D: u32 = 84;
        let c = |v: f64| {
            if dbl {
                splat(64, v.to_bits())
            } else {
                splat(32, u64::from((v as f32).to_bits()))
            }
        };
        self.s().local_tee(X);
        if dbl {
            self.s().f64x2_trunc();
        } else {
            self.s().f32x4_trunc();
        }
        self.s().local_set(T).local_get(X).local_get(T);
        self.fop(Fop::Sub, dbl);
        self.s().local_set(D);
        // t + 1, t - 1, t, (d <= -0.5) -> bitselect; (d >= 0.5) -> bitselect
        self.s().local_get(T).v128_const(c(1.0));
        self.fop(Fop::Add, dbl);
        self.s().local_get(T).v128_const(c(1.0));
        self.fop(Fop::Sub, dbl);
        self.s().local_get(T).local_get(D).v128_const(c(-0.5));
        self.fcmp(Fcmp::Le, dbl);
        self.s().v128_bitselect();
        self.s().local_get(D).v128_const(c(0.5));
        self.fcmp(Fcmp::Ge, dbl);
        self.s().v128_bitselect();
    }

    fn fabs(&mut self, dbl: bool) {
        if dbl {
            self.s().f64x2_abs();
        } else {
            self.s().f32x4_abs();
        }
    }

    /// `op` on lanes of `lanes`; `false` if Wasm has no such operation.
    fn iop(&mut self, op: Iop, lanes: Lanes) -> bool {
        use Lanes::{B, D, H, S};
        let s = self.s();
        match (op, lanes) {
            (Iop::Add, B) => s.i8x16_add(),
            (Iop::Add, H) => s.i16x8_add(),
            (Iop::Add, S) => s.i32x4_add(),
            (Iop::Add, D) => s.i64x2_add(),
            (Iop::Sub, B) => s.i8x16_sub(),
            (Iop::Sub, H) => s.i16x8_sub(),
            (Iop::Sub, S) => s.i32x4_sub(),
            (Iop::Sub, D) => s.i64x2_sub(),
            (Iop::Mul, H) => s.i16x8_mul(),
            (Iop::Mul, S) => s.i32x4_mul(),
            (Iop::Mul, D) => s.i64x2_mul(),
            (Iop::MaxS, B) => s.i8x16_max_s(),
            (Iop::MaxS, H) => s.i16x8_max_s(),
            (Iop::MaxS, S) => s.i32x4_max_s(),
            (Iop::MaxU, B) => s.i8x16_max_u(),
            (Iop::MaxU, H) => s.i16x8_max_u(),
            (Iop::MaxU, S) => s.i32x4_max_u(),
            (Iop::MinS, B) => s.i8x16_min_s(),
            (Iop::MinS, H) => s.i16x8_min_s(),
            (Iop::MinS, S) => s.i32x4_min_s(),
            (Iop::MinU, B) => s.i8x16_min_u(),
            (Iop::MinU, H) => s.i16x8_min_u(),
            (Iop::MinU, S) => s.i32x4_min_u(),
            _ => return false,
        };
        true
    }

    /// Whether [`Self::iop`] has `op` for `lanes` (checked before anything is pushed).
    fn has_iop(op: Iop, lanes: Lanes) -> bool {
        match op {
            Iop::Add | Iop::Sub => true,
            Iop::Mul => lanes != Lanes::B,
            _ => lanes != Lanes::D,
        }
    }

    /// Element `index` of `Vn` (`lanes`) in every lane.
    fn vdup_lane(&mut self, n: VReg, lanes: Lanes, index: usize) {
        self.vload(n);
        self.vload(n);
        let idx = shuffle(lanes, 16 / lanes.bytes(), |_| (false, index));
        self.s().i8x16_shuffle(idx);
    }

    /// The low half of each wide element of the v128 on the stack, packed into the low half
    /// (`XTN`'s shape); with `upper_of`, into the high half above `Vd`'s low half (`XTN2`).
    fn narrow(&mut self, narrow: Lanes, upper_of: Option<VReg>) {
        let s = narrow.bytes();
        let count = 8 / s;
        self.s().local_set(T_A);
        match upper_of {
            Some(d) => {
                self.vload(d);
                self.s().local_get(T_A);
                let mut idx = [0u8; 16];
                for (i, o) in idx.iter_mut().enumerate().take(8) {
                    *o = i as u8;
                }
                for i in 0..count {
                    for j in 0..s {
                        idx[8 + i * s + j] = (16 + 2 * i * s + j) as u8;
                    }
                }
                self.s().i8x16_shuffle(idx);
            }
            None => {
                self.s().local_get(T_A).local_get(T_A);
                let mut idx = [0u8; 16];
                for i in 0..count {
                    for j in 0..s {
                        idx[i * s + j] = (2 * i * s + j) as u8;
                    }
                }
                self.s().i8x16_shuffle(idx);
            }
        }
    }

    // ---- three same ----

    #[allow(clippy::too_many_arguments)]
    fn fp_three(
        &mut self,
        k: u32,
        pc: u64,
        inst: &Inst,
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
        rm: VReg,
    ) -> bool {
        let dbl = size & 1 == 1;
        let hi = size >> 1 == 1;
        let esize = if dbl { 64 } else { 32 };
        if !scalar && !q && dbl {
            return false;
        }
        let keep = Keep::of(scalar, q, esize);
        let pairwise =
            u && matches!(opcode, 0b11000 | 0b11010 | 0b11110) && !(opcode == 0b11010 && hi);
        if pairwise {
            if scalar {
                return false;
            }
            let op = match (opcode, hi) {
                (0b11000, false) | (0b11110, false) => Fop::Max,
                (0b11000, true) | (0b11110, true) => Fop::Min,
                _ => Fop::Add,
            };
            let lanes = if dbl { Lanes::D } else { Lanes::S };
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
            let (evens, odds) = (shuffle(lanes, n, pick(0)), shuffle(lanes, n, pick(1)));
            self.vload(rn);
            self.vload(rm);
            self.s().i8x16_shuffle(evens);
            self.vload(rn);
            self.vload(rm);
            self.s().i8x16_shuffle(odds);
            self.fop(op, dbl);
            self.vfresult(k, pc, inst, rd, keep, dbl);
            return true;
        }
        enum F {
            Op(Fop),
            Cmp(Fcmp, bool),
        }
        let f = match (u, opcode, hi) {
            // FMAXNM/FMINNM differ from FMAX/FMIN only with a NaN operand (then the result is
            // a NaN in Wasm and the interpreter decides). FMULX differs from FMUL only for
            // infinity times zero, a NaN in Wasm.
            (false, 0b11000, false) | (false, 0b11110, false) => F::Op(Fop::Max),
            (false, 0b11000, true) | (false, 0b11110, true) => F::Op(Fop::Min),
            (false, 0b11010, false) => F::Op(Fop::Add),
            (false, 0b11010, true) => F::Op(Fop::Sub),
            (false, 0b11011, false) | (true, 0b11011, false) => F::Op(Fop::Mul),
            (true, 0b11010, true) => F::Op(Fop::Abd),
            (true, 0b11111, false) => F::Op(Fop::Div),
            (false, 0b11100, false) => F::Cmp(Fcmp::Eq, false),
            (true, 0b11100, false) => F::Cmp(Fcmp::Ge, false),
            (true, 0b11100, true) => F::Cmp(Fcmp::Gt, false),
            (true, 0b11101, false) => F::Cmp(Fcmp::Ge, true),
            (true, 0b11101, true) => F::Cmp(Fcmp::Gt, true),
            _ => return false,
        };
        match f {
            F::Op(op) => {
                self.vload(rn);
                self.vload(rm);
                self.fop(op, dbl);
                self.vfresult(k, pc, inst, rd, keep, dbl);
            }
            F::Cmp(c, abs) => {
                self.vload(rn);
                if abs {
                    self.fabs(dbl);
                }
                self.vload(rm);
                if abs {
                    self.fabs(dbl);
                }
                self.fcmp(c, dbl);
                self.vput(rd, keep);
            }
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn int_three(
        &mut self,
        k: u32,
        pc: u64,
        inst: &Inst,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
        rm: VReg,
    ) -> bool {
        let lanes = Lanes::of(size);
        let esize = 8 * lanes.bytes() as u32;
        let keep = Keep::of(false, q, esize);
        let simple = match (opcode, u) {
            (0b10000, false) => Some(Iop::Add),
            (0b10000, true) => Some(Iop::Sub),
            (0b10011, false) => Some(Iop::Mul),
            (0b01100, false) => Some(Iop::MaxS),
            (0b01100, true) => Some(Iop::MaxU),
            (0b01101, false) => Some(Iop::MinS),
            (0b01101, true) => Some(Iop::MinU),
            _ => None,
        };
        if let Some(op) = simple {
            if !Self::has_iop(op, lanes) {
                return false;
            }
            self.vload(rn);
            self.vload(rm);
            self.iop(op, lanes);
            self.vput(rd, keep);
            return true;
        }
        match opcode {
            // MLA, MLS
            0b10010 => {
                if !Self::has_iop(Iop::Mul, lanes) {
                    return false;
                }
                self.vload(rd);
                self.vload(rn);
                self.vload(rm);
                self.iop(Iop::Mul, lanes);
                self.iop(if u { Iop::Sub } else { Iop::Add }, lanes);
                self.vput(rd, keep);
                true
            }
            // SABD, UABD, SABA, UABA: max - min.
            0b01110 | 0b01111 => {
                if lanes == Lanes::D {
                    return false;
                }
                let (max, min) = if u {
                    (Iop::MaxU, Iop::MinU)
                } else {
                    (Iop::MaxS, Iop::MinS)
                };
                if opcode == 0b01111 {
                    self.vload(rd);
                }
                self.vload(rn);
                self.vload(rm);
                self.iop(max, lanes);
                self.vload(rn);
                self.vload(rm);
                self.iop(min, lanes);
                self.iop(Iop::Sub, lanes);
                if opcode == 0b01111 {
                    self.iop(Iop::Add, lanes);
                }
                self.vput(rd, keep);
                true
            }
            // SSHL, USHL of words and doublewords: each lane shifted by the signed low byte of
            // the other's lane (left if positive, right if negative), lane by lane.
            0b01000 => {
                if !matches!(lanes, Lanes::S | Lanes::D) {
                    return false;
                }
                const X: u32 = 85;
                const SH: u32 = 86;
                let dbl = lanes == Lanes::D;
                let bits = if dbl { 64 } else { 32 };
                self.vload(rn);
                self.s().local_set(T_A);
                self.vload(rm);
                self.s().local_set(T_B);
                self.s().local_get(T_A);
                for i in 0..(16 / lanes.bytes()) as u8 {
                    let s = self.s();
                    if dbl {
                        s.local_get(T_A).i64x2_extract_lane(i).local_set(X);
                        s.local_get(T_B)
                            .i64x2_extract_lane(i)
                            .i64_extend8_s()
                            .local_set(SH);
                        // left: sh < 64 ? x << sh : 0
                        s.local_get(X)
                            .local_get(SH)
                            .i64_shl()
                            .i64_const(0)
                            .local_get(SH)
                            .i64_const(64)
                            .i64_lt_u()
                            .select();
                        // right by n = -sh
                        if u {
                            s.local_get(X)
                                .i64_const(0)
                                .local_get(SH)
                                .i64_sub()
                                .i64_shr_u()
                                .i64_const(0)
                                .i64_const(0)
                                .local_get(SH)
                                .i64_sub()
                                .i64_const(64)
                                .i64_lt_u()
                                .select();
                        } else {
                            s.local_get(X)
                                .i64_const(0)
                                .local_get(SH)
                                .i64_sub()
                                .i64_const(63)
                                .i64_const(0)
                                .local_get(SH)
                                .i64_sub()
                                .i64_const(64)
                                .i64_lt_u()
                                .select()
                                .i64_shr_s();
                        }
                        s.local_get(SH).i64_const(0).i64_ge_s().select();
                        s.i64x2_replace_lane(i);
                    } else {
                        s.local_get(T_A).i32x4_extract_lane(i).local_set(X);
                        s.local_get(T_B)
                            .i32x4_extract_lane(i)
                            .i32_const(24)
                            .i32_shl()
                            .i32_const(24)
                            .i32_shr_s()
                            .local_set(SH);
                        s.local_get(X)
                            .local_get(SH)
                            .i32_shl()
                            .i32_const(0)
                            .local_get(SH)
                            .i32_const(bits)
                            .i32_lt_u()
                            .select();
                        if u {
                            s.local_get(X)
                                .i32_const(0)
                                .local_get(SH)
                                .i32_sub()
                                .i32_shr_u()
                                .i32_const(0)
                                .i32_const(0)
                                .local_get(SH)
                                .i32_sub()
                                .i32_const(bits)
                                .i32_lt_u()
                                .select();
                        } else {
                            s.local_get(X)
                                .i32_const(0)
                                .local_get(SH)
                                .i32_sub()
                                .i32_const(bits - 1)
                                .i32_const(0)
                                .local_get(SH)
                                .i32_sub()
                                .i32_const(bits)
                                .i32_lt_u()
                                .select()
                                .i32_shr_s();
                        }
                        s.local_get(SH).i32_const(0).i32_ge_s().select();
                        s.i32x4_replace_lane(i);
                    }
                }
                self.vput(rd, keep);
                true
            }
            // SQADD, UQADD, SQSUB, UQSUB: the saturated result, FPSR.QC set inline where a
            // lane saturated (earlier versions stepped those).
            0b00001 | 0b00101 => {
                const SAT: u32 = 91;
                const MASK: u32 = 92;
                let sub = opcode == 0b00101;
                match lanes {
                    Lanes::B | Lanes::H => {
                        self.vload(rn);
                        self.vload(rm);
                        let s = self.s();
                        match (lanes, sub, u) {
                            (Lanes::B, false, false) => s.i8x16_add_sat_s(),
                            (Lanes::B, false, true) => s.i8x16_add_sat_u(),
                            (Lanes::B, true, false) => s.i8x16_sub_sat_s(),
                            (Lanes::B, true, true) => s.i8x16_sub_sat_u(),
                            (_, false, false) => s.i16x8_add_sat_s(),
                            (_, false, true) => s.i16x8_add_sat_u(),
                            (_, true, false) => s.i16x8_sub_sat_s(),
                            (_, true, true) => s.i16x8_sub_sat_u(),
                        };
                        s.local_set(SAT);
                        // Saturated where it differs from the wrapping result.
                        self.vload(rn);
                        self.vload(rm);
                        self.iop(if sub { Iop::Sub } else { Iop::Add }, lanes);
                        self.s().local_get(SAT).v128_xor();
                        self.qc_where(keep);
                        self.s().local_get(SAT);
                        self.vput(rd, keep);
                    }
                    Lanes::S | Lanes::D => {
                        let sign = splat(esize, 1 << (esize - 1));
                        self.vload(rn);
                        self.vload(rm);
                        self.iop(if sub { Iop::Sub } else { Iop::Add }, lanes);
                        self.s().local_set(T_B);
                        if u {
                            // Unsigned: a carry out of `x + y` (x > r), a borrow in `x - y`
                            // (y > x), compared signed with the sign bits flipped: whole-lane
                            // masks.
                            if sub {
                                self.vload(rm);
                                self.s().v128_const(sign).v128_xor();
                                self.vload(rn);
                                self.s().v128_const(sign).v128_xor();
                            } else {
                                self.vload(rn);
                                self.s().v128_const(sign).v128_xor();
                                self.s().local_get(T_B).v128_const(sign).v128_xor();
                            }
                            if lanes == Lanes::S {
                                self.s().i32x4_gt_s();
                            } else {
                                self.s().i64x2_gt_s();
                            }
                        } else {
                            // Signed: add overflows when r's sign differs from both x's and
                            // y's; sub when x and y differ in sign and r's differs from x's.
                            if sub {
                                self.vload(rn);
                                self.vload(rm);
                                self.s().v128_xor();
                                self.vload(rn);
                                self.s().local_get(T_B).v128_xor().v128_and();
                            } else {
                                self.vload(rn);
                                self.s().local_get(T_B).v128_xor();
                                self.vload(rm);
                                self.s().local_get(T_B).v128_xor().v128_and();
                            }
                            // The sign bit spread over the lane.
                            self.s().i32_const(esize as i32 - 1);
                            if lanes == Lanes::S {
                                self.s().i32x4_shr_s();
                            } else {
                                self.s().i64x2_shr_s();
                            }
                        }
                        self.s().local_set(MASK);
                        // The saturation value: unsigned all ones (add) or zero (sub); signed
                        // the extreme of x's sign (overflow always goes past it).
                        if u {
                            self.s().v128_const(if sub { 0 } else { -1 });
                        } else {
                            self.vload(rn);
                            self.s().i32_const(esize as i32 - 1);
                            if lanes == Lanes::S {
                                self.s().i32x4_shr_s();
                            } else {
                                self.s().i64x2_shr_s();
                            }
                            self.s().v128_const(!sign).v128_xor();
                        }
                        self.s()
                            .local_get(T_B)
                            .local_get(MASK)
                            .v128_bitselect()
                            .local_set(SAT);
                        self.s().local_get(MASK);
                        self.qc_where(keep);
                        self.s().local_get(SAT);
                        self.vput(rd, keep);
                    }
                }
                let _ = (k, pc, inst);
                true
            }
            _ => false,
        }
    }

    /// Sets FPSR.QC if the v128 on the stack is nonzero in a lane `keep` writes.
    fn qc_where(&mut self, keep: Keep) {
        if let Some(m) = keep.mask() {
            self.s().v128_const(m).v128_and();
        }
        self.set_qc_if_any();
    }

    /// Widens the low (or with `high`, the high) half of the v128 on the stack from `lanes`.
    fn vwiden(&mut self, lanes: Lanes, high: bool, signed: bool) {
        let s = self.s();
        match (lanes, high, signed) {
            (Lanes::B, false, true) => s.i16x8_extend_low_i8x16_s(),
            (Lanes::B, true, true) => s.i16x8_extend_high_i8x16_s(),
            (Lanes::B, false, false) => s.i16x8_extend_low_i8x16_u(),
            (Lanes::B, true, false) => s.i16x8_extend_high_i8x16_u(),
            (Lanes::H, false, true) => s.i32x4_extend_low_i16x8_s(),
            (Lanes::H, true, true) => s.i32x4_extend_high_i16x8_s(),
            (Lanes::H, false, false) => s.i32x4_extend_low_i16x8_u(),
            (Lanes::H, true, false) => s.i32x4_extend_high_i16x8_u(),
            (_, false, true) => s.i64x2_extend_low_i32x4_s(),
            (_, true, true) => s.i64x2_extend_high_i32x4_s(),
            (_, false, false) => s.i64x2_extend_low_i32x4_u(),
            (_, true, false) => s.i64x2_extend_high_i32x4_u(),
        };
    }

    /// `ADDP` (scalar, doublewords) and the FP pairwise reductions of two elements.
    #[allow(clippy::too_many_arguments)]
    fn scalar_pairwise(
        &mut self,
        k: u32,
        pc: u64,
        inst: &Inst,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
    ) -> bool {
        let swap = |e: &mut Self, lanes: Lanes| {
            e.s().local_tee(T_A).local_get(T_A).local_get(T_A);
            let n = 16 / lanes.bytes();
            e.s().i8x16_shuffle(shuffle(lanes, n, |i| (false, i ^ 1)));
        };
        if !u && opcode == 0b11011 && size == 3 {
            self.vload(rn);
            swap(self, Lanes::D);
            self.s().i64x2_add();
            self.vput(rd, Keep::Scalar(64));
            return true;
        }
        if !u {
            return false;
        }
        let dbl = size & 1 == 1;
        let op = match (opcode, size >> 1) {
            (0b01100, 0) | (0b01111, 0) => Fop::Max,
            (0b01100, _) | (0b01111, _) => Fop::Min,
            (0b01101, 0) => Fop::Add,
            _ => return false,
        };
        let (lanes, esize) = if dbl { (Lanes::D, 64) } else { (Lanes::S, 32) };
        self.vload(rn);
        swap(self, lanes);
        self.fop(op, dbl);
        self.vfresult(k, pc, inst, rd, Keep::Scalar(esize), dbl);
        true
    }

    /// Advanced SIMD three different: the widening (`L`, `W`) and narrowing (`HN`) forms.
    #[allow(clippy::too_many_arguments)]
    fn three_diff(
        &mut self,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
        rm: VReg,
    ) -> bool {
        if size == 3 {
            return false;
        }
        let lanes = Lanes::of(size);
        let wide = Lanes::of(size + 1);
        let esize = 8 * lanes.bytes() as u32;
        let ext = |e: &mut Self, r: VReg| {
            e.vload(r);
            e.vwiden(lanes, q, !u);
        };
        let abs = |e: &mut Self| {
            let s = e.s();
            match wide {
                Lanes::H => s.i16x8_abs(),
                Lanes::S => s.i32x4_abs(),
                _ => s.i64x2_abs(),
            };
        };
        match opcode {
            // ADDHN, SUBHN, RADDHN, RSUBHN: the high halves of the wide sums.
            0b0100 | 0b0110 => {
                self.vload(rn);
                self.vload(rm);
                self.iop(if opcode == 0b0100 { Iop::Add } else { Iop::Sub }, wide);
                if u {
                    self.s().v128_const(splat(2 * esize, 1 << (esize - 1)));
                    self.iop(Iop::Add, wide);
                }
                self.s().i32_const(esize as i32);
                let s = self.s();
                match wide {
                    Lanes::H => s.i16x8_shr_u(),
                    Lanes::S => s.i32x4_shr_u(),
                    _ => s.i64x2_shr_u(),
                };
                if q {
                    self.narrow(lanes, Some(rd));
                    self.vput(rd, Keep::Q);
                } else {
                    self.narrow(lanes, None);
                    self.vput(rd, Keep::Low);
                }
                return true;
            }
            // SADDL, UADDL, SSUBL, USUBL
            0b0000 | 0b0010 => {
                ext(self, rn);
                ext(self, rm);
                self.iop(if opcode == 0 { Iop::Add } else { Iop::Sub }, wide);
            }
            // SADDW, UADDW, SSUBW, USUBW
            0b0001 | 0b0011 => {
                self.vload(rn);
                ext(self, rm);
                self.iop(if opcode == 1 { Iop::Add } else { Iop::Sub }, wide);
            }
            // SABDL, UABDL, SABAL, UABAL: the difference fits the wide lane.
            0b0111 | 0b0101 => {
                if opcode == 0b0101 {
                    self.vload(rd);
                }
                ext(self, rn);
                ext(self, rm);
                self.iop(Iop::Sub, wide);
                abs(self);
                if opcode == 0b0101 {
                    self.iop(Iop::Add, wide);
                }
            }
            // SMULL, UMULL, SMLAL, UMLAL, SMLSL, UMLSL: the product fits the wide lane.
            0b1100 | 0b1000 | 0b1010 => {
                if opcode != 0b1100 {
                    self.vload(rd);
                }
                ext(self, rn);
                ext(self, rm);
                self.iop(Iop::Mul, wide);
                match opcode {
                    0b1000 => {
                        self.iop(Iop::Add, wide);
                    }
                    0b1010 => {
                        self.iop(Iop::Sub, wide);
                    }
                    _ => {}
                }
            }
            _ => return false,
        }
        self.vput(rd, Keep::Q);
        true
    }

    // ---- two-register miscellaneous ----

    #[allow(clippy::too_many_arguments)]
    fn fp_two(
        &mut self,
        k: u32,
        pc: u64,
        inst: &Inst,
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
    ) -> bool {
        let dbl = size & 1 == 1;
        let hi = size >> 1 == 1;
        // FCVTL, FCVTN (and their 2 forms) between single and double; half precision: the
        // interpreter.
        if !scalar && matches!(opcode, 0b10110 | 0b10111) && !hi && !u {
            if !dbl {
                return false;
            }
            if opcode == 0b10111 {
                self.vload(rn);
                if q {
                    self.s().local_tee(T_A).local_get(T_A);
                    self.s()
                        .i8x16_shuffle(shuffle(Lanes::S, 4, |i| (false, (i + 2) % 4)));
                }
                self.s().f64x2_promote_low_f32x4();
                self.vfresult(k, pc, inst, rd, Keep::Q, true);
            } else {
                self.vload(rn);
                self.s().f32x4_demote_f64x2_zero().local_set(T_B);
                if q {
                    self.vload(rd);
                    self.s().local_get(T_B);
                    let mut idx = [0u8; 16];
                    for (i, o) in idx.iter_mut().enumerate() {
                        *o = if i < 8 { i as u8 } else { 8 + i as u8 };
                    }
                    self.s().i8x16_shuffle(idx);
                } else {
                    self.s().local_get(T_B);
                }
                self.s()
                    .local_get(T_B)
                    .local_get(T_B)
                    .f32x4_ne()
                    .v128_const(LOW64)
                    .v128_and();
                self.guarded_put(k, pc, inst, rd, Keep::Q);
            }
            return true;
        }
        if !scalar && !q && dbl {
            return false;
        }
        let esize: u32 = if dbl { 64 } else { 32 };
        let keep = Keep::of(scalar, q, esize);
        let sign = splat(esize, 1 << (esize - 1));
        let cmp0 = match (u, opcode, hi) {
            (false, 0b01100, true) => Some(Fcmp::Gt),
            (false, 0b01101, true) => Some(Fcmp::Eq),
            (false, 0b01110, true) => Some(Fcmp::Lt),
            (true, 0b01100, true) => Some(Fcmp::Ge),
            (true, 0b01101, true) => Some(Fcmp::Le),
            _ => None,
        };
        if let Some(c) = cmp0 {
            self.vload(rn);
            self.s().v128_const(0);
            self.fcmp(c, dbl);
            self.vput(rd, keep);
            return true;
        }
        match (u, opcode, hi) {
            // FABS, FNEG: bits.
            (false, 0b01111, true) => {
                self.vload(rn);
                self.s().v128_const(!sign).v128_and();
                self.vput(rd, keep);
            }
            (true, 0b01111, true) => {
                self.vload(rn);
                self.s().v128_const(sign).v128_xor();
                self.vput(rd, keep);
            }
            // FSQRT, FRINTN/M/P/Z/X/I (round to nearest even: FPCR's mode is ignored).
            (true, 0b11111, true) | (false, 0b11000 | 0b11001, _) | (true, 0b11001, _) => {
                self.vload(rn);
                let s = self.s();
                match ((u, opcode, hi), dbl) {
                    ((true, 0b11111, _), false) => s.f32x4_sqrt(),
                    ((false, 0b11001, false), false) => s.f32x4_floor(),
                    ((false, 0b11000, true), false) => s.f32x4_ceil(),
                    ((false, 0b11001, true), false) => s.f32x4_trunc(),
                    (_, false) => s.f32x4_nearest(),
                    ((true, 0b11111, _), true) => s.f64x2_sqrt(),
                    ((false, 0b11001, false), true) => s.f64x2_floor(),
                    ((false, 0b11000, true), true) => s.f64x2_ceil(),
                    ((false, 0b11001, true), true) => s.f64x2_trunc(),
                    (_, true) => s.f64x2_nearest(),
                };
                self.vfresult(k, pc, inst, rd, keep, dbl);
            }
            // FCVTN/M/P/Z (S, U) of single lanes: round, then convert saturating, NaN to 0.
            (_, 0b11010 | 0b11011, _) if !dbl => {
                self.vload(rn);
                let s = self.s();
                match (opcode, hi) {
                    (0b11010, false) => s.f32x4_nearest(),
                    (0b11011, false) => s.f32x4_floor(),
                    (0b11010, true) => s.f32x4_ceil(),
                    _ => s.f32x4_trunc(),
                };
                if u {
                    s.i32x4_trunc_sat_f32x4_u();
                } else {
                    s.i32x4_trunc_sat_f32x4_s();
                }
                self.vput(rd, keep);
            }
            // FRINTA: ties away from zero.
            (true, 0b11000, false) => {
                self.vload(rn);
                self.vround_away(dbl);
                self.vfresult(k, pc, inst, rd, keep, dbl);
            }
            // FCVTAS, FCVTAU of single lanes.
            (_, 0b11100, false) if !dbl => {
                self.vload(rn);
                self.vround_away(false);
                if u {
                    self.s().i32x4_trunc_sat_f32x4_u();
                } else {
                    self.s().i32x4_trunc_sat_f32x4_s();
                }
                self.vput(rd, keep);
            }
            // SCVTF, UCVTF of 32-bit lanes (exactly rounded, never a NaN).
            (_, 0b11101, false) if !dbl => {
                self.vload(rn);
                if u {
                    self.s().f32x4_convert_i32x4_u();
                } else {
                    self.s().f32x4_convert_i32x4_s();
                }
                self.vput(rd, keep);
            }
            // The same conversions of doubleword lanes (scalar D or 2D): lane by lane,
            // with Wasm's scalar conversions (Wasm has no 64-bit lane forms).
            (_, 0b11010..=0b11100, _) | (_, 0b11101, false) if dbl => {
                if opcode == 0b11100 && hi {
                    return false;
                }
                const X: u32 = 93;
                self.vload(rn);
                self.s().local_set(X);
                self.s().v128_const(0);
                for lane in 0..if scalar { 1u8 } else { 2 } {
                    self.s().local_get(X).i64x2_extract_lane(lane);
                    if opcode == 0b11101 {
                        if u {
                            self.s().f64_convert_i64_u();
                        } else {
                            self.s().f64_convert_i64_s();
                        }
                        self.s().i64_reinterpret_f64();
                    } else {
                        self.s().f64_reinterpret_i64();
                        match (opcode, hi) {
                            (0b11100, _) => self.round_away(true),
                            (0b11010, false) => {
                                self.s().f64_nearest();
                            }
                            (0b11011, false) => {
                                self.s().f64_floor();
                            }
                            (0b11010, true) => {
                                self.s().f64_ceil();
                            }
                            _ => {
                                self.s().f64_trunc();
                            }
                        }
                        if u {
                            self.s().i64_trunc_sat_f64_u();
                        } else {
                            self.s().i64_trunc_sat_f64_s();
                        }
                    }
                    self.s().i64x2_replace_lane(lane);
                }
                self.vput(rd, keep);
            }
            _ => return false,
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn int_two(
        &mut self,
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
    ) -> bool {
        let lanes = Lanes::of(size);
        let esize = 8 * lanes.bytes() as u32;
        match (opcode, u) {
            // ABS, NEG (vector and scalar).
            (0b01011, _) => {
                self.vload(rn);
                let s = self.s();
                match (u, lanes) {
                    (false, Lanes::B) => s.i8x16_abs(),
                    (false, Lanes::H) => s.i16x8_abs(),
                    (false, Lanes::S) => s.i32x4_abs(),
                    (false, Lanes::D) => s.i64x2_abs(),
                    (true, Lanes::B) => s.i8x16_neg(),
                    (true, Lanes::H) => s.i16x8_neg(),
                    (true, Lanes::S) => s.i32x4_neg(),
                    (true, Lanes::D) => s.i64x2_neg(),
                };
                self.vput(rd, Keep::of(scalar, q, esize));
                true
            }
            _ if scalar => false,
            // SQXTUN, SQXTN, UQXTN (and their 2 forms): clamp each wide element to the narrow
            // range, set FPSR.QC if any was clamped, then narrow like XTN.
            (0b10010, true) | (0b10100, _) => {
                if lanes == Lanes::D {
                    return false;
                }
                const X: u32 = 87;
                const R: u32 = 88;
                let wide = Lanes::of(size + 1);
                let unsigned_in = opcode == 0b10100 && u;
                let (lo, hi): (i64, i64) = if opcode == 0b10100 && !u {
                    (-(1i64 << (esize - 1)), (1i64 << (esize - 1)) - 1)
                } else {
                    (0, (1i64 << esize) - 1)
                };
                self.vload(rn);
                self.s().local_tee(X);
                if unsigned_in {
                    self.vclamp(wide, Iop::MinU, hi);
                } else {
                    self.vclamp(wide, Iop::MaxS, lo);
                    self.vclamp(wide, Iop::MinS, hi);
                }
                self.s().local_tee(R).local_get(X);
                let s = self.s();
                match wide {
                    Lanes::H => s.i16x8_ne(),
                    Lanes::S => s.i32x4_ne(),
                    _ => s.i64x2_ne(),
                };
                self.set_qc_if_any();
                self.s().local_get(R);
                if q {
                    self.narrow(lanes, Some(rd));
                    self.vput(rd, Keep::Q);
                } else {
                    self.narrow(lanes, None);
                    self.vput(rd, Keep::Low);
                }
                true
            }
            // XTN, XTN2
            (0b10010, false) => {
                if lanes == Lanes::D {
                    return false;
                }
                self.vload(rn);
                if q {
                    self.narrow(lanes, Some(rd));
                    self.vput(rd, Keep::Q);
                } else {
                    self.narrow(lanes, None);
                    self.vput(rd, Keep::Low);
                }
                true
            }
            // SHLL, SHLL2: widen and shift left by the element size.
            (0b10011, true) => {
                if lanes == Lanes::D {
                    return false;
                }
                self.vload(rn);
                let s = self.s();
                match (lanes, q) {
                    (Lanes::B, false) => s.i16x8_extend_low_i8x16_u(),
                    (Lanes::B, true) => s.i16x8_extend_high_i8x16_u(),
                    (Lanes::H, false) => s.i32x4_extend_low_i16x8_u(),
                    (Lanes::H, true) => s.i32x4_extend_high_i16x8_u(),
                    (_, false) => s.i64x2_extend_low_i32x4_u(),
                    (_, true) => s.i64x2_extend_high_i32x4_u(),
                };
                s.i32_const(esize as i32);
                match lanes {
                    Lanes::B => s.i16x8_shl(),
                    Lanes::H => s.i32x4_shl(),
                    _ => s.i64x2_shl(),
                };
                self.vput(rd, Keep::Q);
                true
            }
            // SADDLP, UADDLP, SADALP, UADALP of words: each doubleword's two halves,
            // widened (`(v << 32) >> 32` and `v >> 32`, arithmetic when signed), added.
            (0b00010 | 0b00110, _) if lanes == Lanes::S => {
                const V: u32 = 94;
                let acc = opcode == 0b00110;
                if acc {
                    self.vload(rd);
                }
                self.vload(rn);
                self.s().local_tee(V);
                let s = self.s();
                if u {
                    s.v128_const(splat(64, 0xffff_ffff)).v128_and();
                } else {
                    s.i32_const(32).i64x2_shl().i32_const(32).i64x2_shr_s();
                }
                s.local_get(V).i32_const(32);
                if u {
                    s.i64x2_shr_u();
                } else {
                    s.i64x2_shr_s();
                }
                s.i64x2_add();
                if acc {
                    s.i64x2_add();
                }
                self.vput(rd, Keep::of(false, q, 64));
                true
            }
            // SADDLP, UADDLP, SADALP, UADALP of bytes and halfwords.
            (0b00010 | 0b00110, _) => {
                if !matches!(lanes, Lanes::B | Lanes::H) {
                    return false;
                }
                let acc = opcode == 0b00110;
                if acc {
                    self.vload(rd);
                }
                self.vload(rn);
                let s = self.s();
                match (lanes, u) {
                    (Lanes::B, false) => s.i16x8_extadd_pairwise_i8x16_s(),
                    (Lanes::B, true) => s.i16x8_extadd_pairwise_i8x16_u(),
                    (_, false) => s.i32x4_extadd_pairwise_i16x8_s(),
                    (_, true) => s.i32x4_extadd_pairwise_i16x8_u(),
                };
                if acc {
                    if lanes == Lanes::B {
                        s.i16x8_add();
                    } else {
                        s.i32x4_add();
                    }
                }
                self.vput(rd, Keep::of(false, q, esize));
                true
            }
            // REV64, REV32, REV16: reverse elements within containers.
            (0b00000, _) | (0b00001, false) => {
                let container = match (opcode, u) {
                    (0b00000, false) => 64,
                    (0b00000, true) => 32,
                    _ => 16,
                };
                if esize >= container {
                    return false;
                }
                let per = (container / esize) as usize;
                let idx = shuffle(lanes, 16 / lanes.bytes(), |i| {
                    (false, i - i % per + (per - 1 - i % per))
                });
                self.vload(rn);
                self.vload(rn);
                self.s().i8x16_shuffle(idx);
                self.vput(rd, Keep::of(false, q, esize));
                true
            }
            _ => false,
        }
    }

    // ---- shift by immediate ----

    #[allow(clippy::too_many_arguments)]
    fn shift_imm(
        &mut self,
        scalar: bool,
        q: bool,
        u: bool,
        immh: u8,
        immb: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
    ) -> bool {
        if immh == 0 {
            return false;
        }
        let hsb = 7 - immh.leading_zeros();
        let lanes = Lanes::of(hsb as u8);
        let esize = 8u32 << hsb;
        let immhb = (u32::from(immh) << 3) | u32::from(immb);
        let shl = |s: &mut crate::ir::build::Builder, lanes: Lanes| {
            match lanes {
                Lanes::B => s.i8x16_shl(),
                Lanes::H => s.i16x8_shl(),
                Lanes::S => s.i32x4_shl(),
                Lanes::D => s.i64x2_shl(),
            };
        };
        let shr = |s: &mut crate::ir::build::Builder, lanes: Lanes, signed: bool| {
            match (lanes, signed) {
                (Lanes::B, false) => s.i8x16_shr_u(),
                (Lanes::H, false) => s.i16x8_shr_u(),
                (Lanes::S, false) => s.i32x4_shr_u(),
                (Lanes::D, false) => s.i64x2_shr_u(),
                (Lanes::B, true) => s.i8x16_shr_s(),
                (Lanes::H, true) => s.i16x8_shr_s(),
                (Lanes::S, true) => s.i32x4_shr_s(),
                (Lanes::D, true) => s.i64x2_shr_s(),
            };
        };
        // SSHLL, USHLL (and their 2 forms): widen, then shift left.
        if opcode == 0b10100 {
            if scalar || esize == 64 {
                return false;
            }
            let shift = immhb - esize;
            self.vload(rn);
            let s = self.s();
            match (lanes, q, u) {
                (Lanes::B, false, false) => s.i16x8_extend_low_i8x16_s(),
                (Lanes::B, true, false) => s.i16x8_extend_high_i8x16_s(),
                (Lanes::B, false, true) => s.i16x8_extend_low_i8x16_u(),
                (Lanes::B, true, true) => s.i16x8_extend_high_i8x16_u(),
                (Lanes::H, false, false) => s.i32x4_extend_low_i16x8_s(),
                (Lanes::H, true, false) => s.i32x4_extend_high_i16x8_s(),
                (Lanes::H, false, true) => s.i32x4_extend_low_i16x8_u(),
                (Lanes::H, true, true) => s.i32x4_extend_high_i16x8_u(),
                (_, false, false) => s.i64x2_extend_low_i32x4_s(),
                (_, true, false) => s.i64x2_extend_high_i32x4_s(),
                (_, false, true) => s.i64x2_extend_low_i32x4_u(),
                (_, true, true) => s.i64x2_extend_high_i32x4_u(),
            };
            if shift > 0 {
                s.i32_const(shift as i32);
                shl(s, Lanes::of(hsb as u8 + 1));
            }
            self.vput(rd, Keep::Q);
            return true;
        }
        // SHRN, RSHRN (and their 2 forms): shift each wide element right, keep its low half.
        if matches!(opcode, 0b10000 | 0b10001) {
            if scalar || esize == 64 || u {
                return false;
            }
            let wide = Lanes::of(hsb as u8 + 1);
            let shift = 2 * esize - immhb;
            self.vload(rn);
            if opcode == 0b10001 {
                // The rounding constant may carry out of the wide element: only bits below
                // `shift + esize <= 2 * esize` are kept, so wrapping is exact.
                self.s().v128_const(splat(2 * esize, 1 << (shift - 1)));
                self.iop(Iop::Add, wide);
            }
            self.s().i32_const(shift as i32);
            shr(self.s(), wide, false);
            if q {
                self.narrow(lanes, Some(rd));
                self.vput(rd, Keep::Q);
            } else {
                self.narrow(lanes, None);
                self.vput(rd, Keep::Low);
            }
            return true;
        }
        // SCVTF/UCVTF and FCVTZS/FCVTZU, fixed point, of single lanes (vector and scalar): the
        // interpreter's double arithmetic scaled by 2^fbits, which single precision does exactly
        // (no lane can underflow; an overflow to infinity saturates the same way).
        if matches!(opcode, 0b11100 | 0b11111) {
            if esize != 32 {
                return false;
            }
            let fbits = 2 * esize - immhb;
            let keep = Keep::of(scalar, q, esize);
            self.vload(rn);
            if opcode == 0b11100 {
                if u {
                    self.s().f32x4_convert_i32x4_u();
                } else {
                    self.s().f32x4_convert_i32x4_s();
                }
                let c = f32::from_bits((127 - fbits) << 23);
                self.s().v128_const(splat(32, u64::from(c.to_bits())));
                self.fop(Fop::Mul, false);
            } else {
                let c = f32::from_bits((127 + fbits) << 23);
                self.s().v128_const(splat(32, u64::from(c.to_bits())));
                self.fop(Fop::Mul, false);
                if u {
                    self.s().i32x4_trunc_sat_f32x4_u();
                } else {
                    self.s().i32x4_trunc_sat_f32x4_s();
                }
            }
            self.vput(rd, keep);
            return true;
        }
        if !scalar && esize == 64 && !q {
            return false;
        }
        if scalar && esize != 64 {
            return false;
        }
        let keep = Keep::of(scalar, q, esize);
        let right = 2 * esize - immhb;
        let left = immhb.wrapping_sub(esize);
        match (opcode, u) {
            // SSHR, USHR, SSRA, USRA
            (0b00000 | 0b00010, _) => {
                let acc = opcode == 0b00010;
                if acc {
                    self.vload(rd);
                }
                if u && right == esize {
                    self.s().v128_const(0);
                } else {
                    self.vload(rn);
                    self.s().i32_const(right.min(esize - 1) as i32);
                    shr(self.s(), lanes, !u);
                }
                if acc {
                    self.iop(Iop::Add, lanes);
                }
                self.vput(rd, keep);
                true
            }
            // SHL, SLI
            (0b01010, _) => {
                if left >= esize {
                    return false;
                }
                self.vload(rn);
                if left > 0 {
                    self.s().i32_const(left as i32);
                    shl(self.s(), lanes);
                }
                if u {
                    let all = if esize == 64 {
                        u64::MAX
                    } else {
                        (1 << esize) - 1
                    };
                    let m = splat(esize, (all << left) & all);
                    self.vload(rd);
                    self.s().v128_const(m).v128_bitselect();
                }
                self.vput(rd, keep);
                true
            }
            // SRI
            (0b01000, true) => {
                if right == esize {
                    // Nothing is inserted: Vd keeps its bits (and the clearing).
                    self.vload(rd);
                } else {
                    let all = if esize == 64 {
                        u64::MAX
                    } else {
                        (1 << esize) - 1
                    };
                    self.vload(rn);
                    self.s().i32_const(right as i32);
                    shr(self.s(), lanes, false);
                    self.vload(rd);
                    self.s()
                        .v128_const(splat(esize, all >> right))
                        .v128_bitselect();
                }
                self.vput(rd, keep);
                true
            }
            _ => false,
        }
    }

    // ---- across lanes ----

    #[allow(clippy::too_many_arguments)]
    fn across(
        &mut self,
        k: u32,
        pc: u64,
        inst: &Inst,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
    ) -> bool {
        if matches!(opcode, 0b01100 | 0b01111) {
            // FMAXNMV, FMINNMV, FMAXV, FMINV (4S): op(op(a0, a1), op(a2, a3)).
            if !u || !q || size & 1 == 1 {
                return false;
            }
            let op = if size >> 1 == 0 { Fop::Max } else { Fop::Min };
            self.vload(rn);
            self.s().local_tee(T_A).local_get(T_A).local_get(T_A);
            self.s()
                .i8x16_shuffle(shuffle(Lanes::S, 4, |i| (false, i ^ 1)));
            self.fop(op, false);
            self.s().local_tee(T_A).local_get(T_A).local_get(T_A);
            self.s()
                .i8x16_shuffle(shuffle(Lanes::S, 4, |i| (false, i ^ 2)));
            self.fop(op, false);
            self.vfresult(k, pc, inst, rd, Keep::Scalar(32), false);
            return true;
        }
        let lanes = Lanes::of(size);
        let esize = 8 * lanes.bytes() as u32;
        let n = (if q { 16 } else { 8 }) / lanes.bytes();
        if size == 3 || n < 2 {
            return false;
        }
        let count = 16 / lanes.bytes();
        let reduce = |e: &mut Self, op: Iop| {
            let mut half = count / 2;
            while half >= 1 {
                e.s().local_tee(T_A).local_get(T_A).local_get(T_A);
                e.s()
                    .i8x16_shuffle(shuffle(lanes, count, |i| (false, (i + half) % count)));
                e.iop(op, lanes);
                half /= 2;
            }
        };
        // The 64-bit forms: the low half twice for min/max, zeros above it for sums.
        let low_twice = |e: &mut Self| {
            if !q {
                e.s().local_tee(T_A).local_get(T_A);
                e.s().i8x16_shuffle(shuffle(Lanes::D, 2, |_| (false, 0)));
            }
        };
        let low_only = |e: &mut Self| {
            if !q {
                e.s().v128_const(LOW64).v128_and();
            }
        };
        let (out, value_is_i64) = match (opcode, u) {
            // SADDLV, UADDLV
            (0b00011, _) => {
                self.vload(rn);
                low_only(self);
                let s = self.s();
                match lanes {
                    Lanes::B | Lanes::H => {
                        if lanes == Lanes::B {
                            if u {
                                s.i16x8_extadd_pairwise_i8x16_u();
                            } else {
                                s.i16x8_extadd_pairwise_i8x16_s();
                            }
                        }
                        if u {
                            s.i32x4_extadd_pairwise_i16x8_u();
                        } else {
                            s.i32x4_extadd_pairwise_i16x8_s();
                        }
                        // Sum the four 32-bit lanes (wrapping: only 2 * esize bits are kept).
                        s.local_tee(T_A).local_get(T_A).local_get(T_A);
                        s.i8x16_shuffle(shuffle(Lanes::S, 4, |i| (false, i ^ 2)))
                            .i32x4_add();
                        s.local_tee(T_A).local_get(T_A).local_get(T_A);
                        s.i8x16_shuffle(shuffle(Lanes::S, 4, |i| (false, i ^ 1)))
                            .i32x4_add();
                        s.i32x4_extract_lane(0);
                        (2 * esize, false)
                    }
                    _ => {
                        s.local_tee(T_A);
                        if u {
                            s.i64x2_extend_low_i32x4_u();
                        } else {
                            s.i64x2_extend_low_i32x4_s();
                        }
                        s.local_get(T_A);
                        if u {
                            s.i64x2_extend_high_i32x4_u();
                        } else {
                            s.i64x2_extend_high_i32x4_s();
                        }
                        s.i64x2_add().local_tee(T_A).i64x2_extract_lane(0);
                        s.local_get(T_A).i64x2_extract_lane(1).i64_add();
                        (64, true)
                    }
                }
            }
            // SMAXV, UMAXV, SMINV, UMINV, ADDV
            (0b01010 | 0b11010, _) | (0b11011, false) => {
                let op = match (opcode, u) {
                    (0b01010, false) => Iop::MaxS,
                    (0b01010, true) => Iop::MaxU,
                    (0b11010, false) => Iop::MinS,
                    (0b11010, true) => Iop::MinU,
                    _ => Iop::Add,
                };
                if !Self::has_iop(op, lanes) {
                    return false;
                }
                self.vload(rn);
                if op == Iop::Add {
                    low_only(self);
                } else {
                    low_twice(self);
                }
                reduce(self, op);
                let s = self.s();
                match lanes {
                    Lanes::B => s.i8x16_extract_lane_u(0),
                    Lanes::H => s.i16x8_extract_lane_u(0),
                    _ => s.i32x4_extract_lane(0),
                };
                (esize, false)
            }
            _ => return false,
        };
        // The scalar result, zero-extended (`set_scalar`).
        if !value_is_i64 {
            self.s().i64_extend_i32_u();
        }
        if out < 64 {
            self.s().i64_const(((1u64 << out) - 1) as i64).i64_and();
        }
        self.bits_store(rd);
        true
    }

    // ---- by element ----

    #[allow(clippy::too_many_arguments)]
    fn indexed(
        &mut self,
        k: u32,
        pc: u64,
        inst: &Inst,
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        (l, m, h): (bool, bool, bool),
        rd: VReg,
        rn: VReg,
        rm: VReg,
    ) -> bool {
        let fp_op = matches!(opcode, 0b0001 | 0b0101 | 0b1001) && !(u && opcode != 0b1001);
        if fp_op {
            // FMUL, FMULX (by element); FMLA/FMLS are fused: the interpreter.
            if opcode != 0b1001 || size >> 1 != 1 {
                return false;
            }
            let dbl = size & 1 == 1;
            if dbl && l {
                return false;
            }
            let index = if dbl {
                usize::from(h)
            } else {
                usize::from(h) << 1 | usize::from(l)
            };
            let reg = rm | (u8::from(m) << 4);
            let (lanes, esize) = if dbl { (Lanes::D, 64) } else { (Lanes::S, 32) };
            self.vload(rn);
            self.vdup_lane(reg, lanes, index);
            self.fop(Fop::Mul, dbl);
            self.vfresult(k, pc, inst, rd, Keep::of(scalar, q, esize), dbl);
            return true;
        }
        // MUL, MLA, MLS (by element) of halfwords and words.
        if scalar || !matches!(opcode, 0b1000 | 0b0000 | 0b0100) || (opcode == 0b1000 && u) {
            return false;
        }
        let (index, reg) = match size {
            1 => (
                usize::from(h) << 2 | usize::from(l) << 1 | usize::from(m),
                rm,
            ),
            2 => (
                usize::from(h) << 1 | usize::from(l),
                rm | (u8::from(m) << 4),
            ),
            _ => return false,
        };
        let lanes = Lanes::of(size);
        let esize = 8 * lanes.bytes() as u32;
        if opcode != 0b1000 {
            self.vload(rd);
        }
        self.vload(rn);
        self.vdup_lane(reg, lanes, index);
        self.iop(Iop::Mul, lanes);
        match opcode {
            0b0000 => {
                self.iop(Iop::Add, lanes);
            }
            0b0100 => {
                self.iop(Iop::Sub, lanes);
            }
            _ => {}
        }
        self.vput(rd, Keep::of(false, q, esize));
        true
    }
}
