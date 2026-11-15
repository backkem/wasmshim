//! SIMD&FP interpreter.

use wasmshim_cpu::{Fault, GuestMemory, MemFault};

use crate::Cpu;
use crate::fp::{self, CmpOp, Fmt, Op2, Round};
use crate::inst::{SP, ZR};
use crate::simd::{ModImmOp, SimdInst};

type R = Result<(), Fault>;

const LOW64: u128 = u64::MAX as u128;

#[inline]
const fn mask_bits(esize: u32) -> u64 {
    if esize >= 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    }
}

#[inline]
fn elem(v: u128, esize: u32, i: u32) -> u64 {
    ((v >> (i * esize)) as u64) & mask_bits(esize)
}

#[inline]
fn set_elem(v: &mut u128, esize: u32, i: u32, x: u64) {
    let m = u128::from(mask_bits(esize)) << (i * esize);
    *v = (*v & !m) | ((u128::from(x & mask_bits(esize))) << (i * esize));
}

#[inline]
fn sx(x: u64, esize: u32) -> i64 {
    let s = 64 - esize;
    ((x << s) as i64) >> s
}

fn fmt_of(sz: bool) -> Fmt {
    if sz { Fmt::D } else { Fmt::S }
}

fn fmt_of_ftype(ftype: u8) -> Option<Fmt> {
    match ftype {
        0 => Some(Fmt::S),
        1 => Some(Fmt::D),
        _ => None,
    }
}

/// Saturates `v` (an exact result) to an `esize`-bit signed or unsigned value; returns the
/// value and whether it saturated.
fn saturate(v: i128, esize: u32, signed: bool) -> (u64, bool) {
    let (lo, hi) = if signed {
        (-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1)
    } else {
        (0, (1i128 << esize) - 1)
    };
    let c = v.clamp(lo, hi);
    ((c as u64) & mask_bits(esize), c != v)
}

impl Cpu {
    /// An undefined-instruction fault; the caller (the `Inst::Simd` arm of `Cpu::execute`)
    /// fills in the instruction word.
    fn undef(&self, pc: u64) -> Fault {
        Fault::Undefined { pc, insn: 0 }
    }

    #[inline]
    fn v(&self, n: u8) -> u128 {
        self.vreg[usize::from(n & 31)]
    }

    /// Writes a vector result, clearing the upper half for 64-bit (`Q = 0`) forms.
    #[inline]
    fn set_v(&mut self, n: u8, value: u128, q: bool) {
        self.vreg[usize::from(n & 31)] = if q { value } else { value & LOW64 };
    }

    /// Writes a scalar FP/SIMD result (upper bits cleared).
    #[inline]
    fn set_scalar(&mut self, n: u8, value: u64, esize: u32) {
        self.vreg[usize::from(n & 31)] = u128::from(value & mask_bits(esize));
    }

    #[inline]
    fn gp(&self, r: u8) -> u64 {
        if r == 31 { 0 } else { self.x[usize::from(r)] }
    }

    fn dn(&self) -> bool {
        self.fpcr & (1 << 25) != 0
    }

    pub(crate) fn execute_simd(&mut self, mem: &mut dyn GuestMemory, inst: SimdInst, pc: u64) -> R {
        match inst {
            SimdInst::ThreeSame {
                scalar,
                q,
                u,
                size,
                opcode,
                rd,
                rn,
                rm,
            } => self.three_same(scalar, q, u, size, opcode, rd, rn, rm, pc),
            SimdInst::ThreeDiff {
                q,
                u,
                size,
                opcode,
                rd,
                rn,
                rm,
            } => self.three_diff(q, u, size, opcode, rd, rn, rm, pc),
            SimdInst::TwoMisc {
                scalar,
                q,
                u,
                size,
                opcode,
                rd,
                rn,
            } => self.two_misc(scalar, q, u, size, opcode, rd, rn, pc),
            SimdInst::AcrossLanes {
                q,
                u,
                size,
                opcode,
                rd,
                rn,
            } => self.across(q, u, size, opcode, rd, rn, pc),
            SimdInst::ScalarPairwise {
                u,
                size,
                opcode,
                rd,
                rn,
            } => self.scalar_pairwise(u, size, opcode, rd, rn, pc),
            SimdInst::Copy {
                q,
                op,
                imm5,
                imm4,
                rd,
                rn,
            } => self.copy(q, op, imm5, imm4, rd, rn, pc),
            SimdInst::ScalarDup { imm5, rd, rn } => {
                let size = imm5.trailing_zeros();
                let esize = 8 << size;
                let index = u32::from(imm5) >> (size + 1);
                let x = elem(self.v(rn), esize, index);
                self.set_scalar(rd, x, esize);
                Ok(())
            }
            SimdInst::ModImm { q, op, imm, rd } => {
                let imm128 = u128::from(imm) | (u128::from(imm) << 64);
                let old = self.v(rd);
                let r = match op {
                    ModImmOp::Movi => imm128,
                    ModImmOp::Mvni => !imm128,
                    ModImmOp::Orr => old | imm128,
                    ModImmOp::Bic => old & !imm128,
                };
                self.set_v(rd, r, q);
                Ok(())
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
            } => self.shift_imm(scalar, q, u, immh, immb, opcode, rd, rn, pc),
            SimdInst::Permute {
                q,
                size,
                opcode,
                rd,
                rn,
                rm,
            } => {
                let esize = 8u32 << size;
                let n = if q { 128 } else { 64 } / esize;
                let (a, b) = (self.v(rn), self.v(rm));
                let mut r = 0u128;
                let half = n / 2;
                for i in 0..n {
                    let x = match opcode {
                        0b001 | 0b101 => {
                            let k = 2 * i + u32::from(opcode == 0b101);
                            if k < n {
                                elem(a, esize, k)
                            } else {
                                elem(b, esize, k - n)
                            }
                        }
                        0b010 | 0b110 => {
                            let base = (i & !1) + u32::from(opcode == 0b110);
                            if i % 2 == 0 {
                                elem(a, esize, base)
                            } else {
                                elem(b, esize, base)
                            }
                        }
                        0b011 | 0b111 => {
                            let k = i / 2 + if opcode == 0b111 { half } else { 0 };
                            if i % 2 == 0 {
                                elem(a, esize, k)
                            } else {
                                elem(b, esize, k)
                            }
                        }
                        _ => return Err(self.undef(pc)),
                    };
                    set_elem(&mut r, esize, i, x);
                }
                self.set_v(rd, r, q);
                Ok(())
            }
            SimdInst::Ext {
                q,
                index,
                rd,
                rn,
                rm,
            } => {
                let bytes = if q { 16 } else { 8 };
                let (a, b) = (self.v(rn).to_le_bytes(), self.v(rm).to_le_bytes());
                let mut out = [0u8; 16];
                for (i, o) in out.iter_mut().enumerate().take(bytes) {
                    let k = i + usize::from(index);
                    *o = if k < bytes { a[k] } else { b[k - bytes] };
                }
                self.set_v(rd, u128::from_le_bytes(out), q);
                Ok(())
            }
            SimdInst::Tbl {
                q,
                len,
                tbx,
                rd,
                rn,
                rm,
            } => {
                let mut table = [0u8; 64];
                for r in 0..=len {
                    let bytes = self.v((rn + r) % 32).to_le_bytes();
                    table[usize::from(r) * 16..usize::from(r) * 16 + 16].copy_from_slice(&bytes);
                }
                let limit = (usize::from(len) + 1) * 16;
                let idx = self.v(rm).to_le_bytes();
                let mut out = self.v(rd).to_le_bytes();
                for i in 0..if q { 16 } else { 8 } {
                    let k = usize::from(idx[i]);
                    if k < limit {
                        out[i] = table[k];
                    } else if !tbx {
                        out[i] = 0;
                    }
                }
                self.set_v(rd, u128::from_le_bytes(out), q);
                Ok(())
            }
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
            } => self.indexed(scalar, q, u, size, opcode, l, m, h, rd, rn, rm, pc),
            SimdInst::Fp1 {
                ftype,
                opcode,
                rd,
                rn,
            } => self.fp1(ftype, opcode, rd, rn, pc),
            SimdInst::Fp2 {
                ftype,
                opcode,
                rd,
                rn,
                rm,
            } => {
                let f = fmt_of_ftype(ftype).ok_or_else(|| self.undef(pc))?;
                let op = match opcode {
                    0 => Op2::Mul,
                    1 => Op2::Div,
                    2 => Op2::Add,
                    3 => Op2::Sub,
                    4 => Op2::Max,
                    5 => Op2::Min,
                    6 => Op2::MaxNum,
                    7 => Op2::MinNum,
                    8 => Op2::Mul,
                    _ => return Err(self.undef(pc)),
                };
                let esize = f.bits();
                let (a, b) = (elem(self.v(rn), esize, 0), elem(self.v(rm), esize, 0));
                let mut r = fp::op2(f, op, a, b, self.dn());
                if opcode == 8 {
                    r ^= f.sign_bit(); // FNMUL
                }
                self.set_scalar(rd, r, esize);
                Ok(())
            }
            SimdInst::Fp3 {
                ftype,
                o1,
                o0,
                rd,
                rn,
                rm,
                ra,
            } => {
                let f = fmt_of_ftype(ftype).ok_or_else(|| self.undef(pc))?;
                let esize = f.bits();
                let (mut n, m, mut a) = (
                    elem(self.v(rn), esize, 0),
                    elem(self.v(rm), esize, 0),
                    elem(self.v(ra), esize, 0),
                );
                // FMADD: a + n*m; FMSUB: a - n*m; FNMADD: -a - n*m; FNMSUB: -a + n*m.
                if o0 != o1 {
                    n ^= f.sign_bit();
                }
                if o1 {
                    a ^= f.sign_bit();
                }
                let r = fp::fma(f, a, n, m, self.dn());
                self.set_scalar(rd, r, esize);
                Ok(())
            }
            SimdInst::FpCmp {
                ftype,
                zero,
                rn,
                rm,
            } => {
                let f = fmt_of_ftype(ftype).ok_or_else(|| self.undef(pc))?;
                let esize = f.bits();
                let a = elem(self.v(rn), esize, 0);
                let b = if zero { 0 } else { elem(self.v(rm), esize, 0) };
                self.set_nzcv(u64::from(fp::compare(f, a, b)) << 28);
                Ok(())
            }
            SimdInst::FpCondCmp {
                ftype,
                rn,
                rm,
                cond,
                nzcv,
            } => {
                let f = fmt_of_ftype(ftype).ok_or_else(|| self.undef(pc))?;
                let esize = f.bits();
                let flags = if self.cond_holds(cond) {
                    fp::compare(f, elem(self.v(rn), esize, 0), elem(self.v(rm), esize, 0))
                } else {
                    nzcv
                };
                self.set_nzcv(u64::from(flags) << 28);
                Ok(())
            }
            SimdInst::FpCsel {
                ftype,
                rd,
                rn,
                rm,
                cond,
            } => {
                let f = fmt_of_ftype(ftype).ok_or_else(|| self.undef(pc))?;
                let esize = f.bits();
                let src = if self.cond_holds(cond) { rn } else { rm };
                let x = elem(self.v(src), esize, 0);
                self.set_scalar(rd, x, esize);
                Ok(())
            }
            SimdInst::FpImm { ftype, bits, rd } => {
                let f = fmt_of_ftype(ftype).ok_or_else(|| self.undef(pc))?;
                self.set_scalar(rd, bits, f.bits());
                Ok(())
            }
            SimdInst::FpInt {
                sf,
                ftype,
                rmode,
                opcode,
                rd,
                rn,
            } => self.fp_int(sf, ftype, rmode, opcode, rd, rn, pc),
            SimdInst::FpFixed {
                sf,
                ftype,
                rmode,
                opcode,
                fbits,
                rd,
                rn,
            } => {
                let f = fmt_of_ftype(ftype).ok_or_else(|| self.undef(pc))?;
                let width = if sf { 64 } else { 32 };
                let scale = pow2(u32::from(fbits));
                match (rmode, opcode) {
                    (0, 2) | (0, 3) => {
                        let x = self.gp(rn);
                        let signed = opcode == 2;
                        let v = if signed {
                            if sf {
                                x as i64 as f64
                            } else {
                                f64::from(x as i32)
                            }
                        } else if sf {
                            x as f64
                        } else {
                            f64::from(x as u32)
                        };
                        // Exact for |x| < 2^53; larger values round twice (a known limitation).
                        let r = v / scale;
                        let bits = match f {
                            Fmt::S => u64::from((r as f32).to_bits()),
                            Fmt::D => r.to_bits(),
                        };
                        self.set_scalar(rd, bits, f.bits());
                    }
                    (3, 0) | (3, 1) => {
                        let a = elem(self.v(rn), f.bits(), 0);
                        let r = if f.is_nan(a) {
                            0
                        } else {
                            fp::to_int(
                                Fmt::D,
                                (f.to_f64(a) * scale).to_bits(),
                                Round::Zero,
                                opcode == 0,
                                width,
                            )
                        };
                        self.write_gp(rd, sf, r);
                    }
                    _ => return Err(self.undef(pc)),
                }
                Ok(())
            }
            SimdInst::LdStMulti {
                load,
                q,
                size,
                regs,
                selem,
                rt,
                rn,
                post,
            } => self.ldst_multi(mem, load, q, size, regs, selem, rt, rn, post, pc),
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
                mem, load, replicate, q, esize_log, index, selem, rt, rn, post, pc,
            ),
        }
    }

    fn write_gp(&mut self, rd: u8, sf: bool, v: u64) {
        if rd != 31 {
            self.x[usize::from(rd)] = if sf { v } else { v & 0xffff_ffff };
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn three_same(
        &mut self,
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: u8,
        rn: u8,
        rm: u8,
        pc: u64,
    ) -> R {
        let (a, b, d) = (self.v(rn), self.v(rm), self.v(rd));
        if opcode >= 0b11000 {
            return self.fp_three_same(scalar, q, u, size, opcode, rd, a, b, d, pc);
        }
        if opcode == 0b00011 {
            if scalar {
                return Err(self.undef(pc));
            }
            let r = match (u, size) {
                (false, 0) => a & b,
                (false, 1) => a & !b,
                (false, 2) => a | b,
                (false, _) => a | !b,
                (true, 0) => a ^ b,
                (true, 1) => (d & a) | (!d & b), // BSL
                (true, 2) => (d & !b) | (a & b), // BIT
                (true, _) => (d & b) | (a & !b), // BIF
            };
            self.set_v(rd, r, q);
            return Ok(());
        }
        let esize = 8u32 << size;
        let datasize = if scalar {
            esize
        } else if q {
            128
        } else {
            64
        };
        let n = datasize / esize;
        // Pairwise ops work on the concatenation of the operands.
        if matches!(opcode, 0b10100 | 0b10101 | 0b10111) {
            if scalar || (opcode == 0b10111 && u) {
                return Err(self.undef(pc));
            }
            let mut r = 0u128;
            for i in 0..n {
                let (src, k) = if i < n / 2 {
                    (a, 2 * i)
                } else {
                    (b, 2 * (i - n / 2))
                };
                let (x, y) = (elem(src, esize, k), elem(src, esize, k + 1));
                let v = match opcode {
                    0b10111 => x.wrapping_add(y),
                    0b10100 => {
                        if u {
                            x.max(y)
                        } else if sx(x, esize) >= sx(y, esize) {
                            x
                        } else {
                            y
                        }
                    }
                    _ => {
                        if u {
                            x.min(y)
                        } else if sx(x, esize) <= sx(y, esize) {
                            x
                        } else {
                            y
                        }
                    }
                };
                set_elem(&mut r, esize, i, v);
            }
            self.set_v(rd, r, q);
            return Ok(());
        }
        let mut r = 0u128;
        let mut sat = false;
        for i in 0..n {
            let (x, y, acc) = (elem(a, esize, i), elem(b, esize, i), elem(d, esize, i));
            let (sxv, syv) = (i128::from(sx(x, esize)), i128::from(sx(y, esize)));
            let (ux, uy) = (i128::from(x), i128::from(y));
            let all = mask_bits(esize);
            let v = match (opcode, u) {
                (0b00000, _) => (if u { (ux + uy) >> 1 } else { (sxv + syv) >> 1 }) as u64,
                (0b00010, _) => {
                    (if u {
                        (ux + uy + 1) >> 1
                    } else {
                        (sxv + syv + 1) >> 1
                    }) as u64
                }
                (0b00100, _) => (if u { (ux - uy) >> 1 } else { (sxv - syv) >> 1 }) as u64,
                (0b00001, _) | (0b00101, _) => {
                    let exact = match (opcode, u) {
                        (0b00001, false) => sxv + syv,
                        (0b00001, true) => ux + uy,
                        (_, false) => sxv - syv,
                        (_, true) => ux - uy,
                    };
                    let (v, s) = saturate(exact, esize, !u);
                    sat |= s;
                    v
                }
                (0b00110, false) => {
                    if sxv > syv {
                        all
                    } else {
                        0
                    }
                }
                (0b00110, true) => {
                    if ux > uy {
                        all
                    } else {
                        0
                    }
                }
                (0b00111, false) => {
                    if sxv >= syv {
                        all
                    } else {
                        0
                    }
                }
                (0b00111, true) => {
                    if ux >= uy {
                        all
                    } else {
                        0
                    }
                }
                (0b01000, _) | (0b01010, _) => {
                    // SSHL/USHL (and rounding forms): shift by the signed low byte of y.
                    let sh = i64::from(y as u8 as i8);
                    let round = opcode == 0b01010;
                    shift_by(x, esize, sh, !u, round)
                }
                (0b01001, _) => {
                    // SQSHL/UQSHL (register): the exact shift, then saturation (Arm ARM,
                    // shared/functions/integer/SatQ). A left shift by `esize` or more saturates
                    // any nonzero element; the unsigned product is formed in u128 because a
                    // 64-bit element shifted by 64 does not fit an i128.
                    let sh = i64::from(y as u8 as i8);
                    let exact = if sh >= 0 {
                        if u {
                            i128::try_from(u128::from(x) << sh.min(64)).unwrap_or(i128::MAX)
                        } else {
                            sxv << sh.min(64)
                        }
                    } else if u {
                        ux >> (-sh).min(127)
                    } else {
                        sxv >> (-sh).min(127)
                    };
                    let (v, s) = saturate(exact, esize, !u);
                    sat |= s;
                    v
                }
                (0b01100, false) => {
                    if sxv >= syv {
                        x
                    } else {
                        y
                    }
                }
                (0b01100, true) => x.max(y),
                (0b01101, false) => {
                    if sxv <= syv {
                        x
                    } else {
                        y
                    }
                }
                (0b01101, true) => x.min(y),
                (0b01110, _) | (0b01111, _) => {
                    let diff = if u {
                        (ux - uy).unsigned_abs()
                    } else {
                        (sxv - syv).unsigned_abs()
                    } as u64;
                    if opcode == 0b01111 {
                        acc.wrapping_add(diff)
                    } else {
                        diff
                    }
                }
                (0b10000, false) => x.wrapping_add(y),
                (0b10000, true) => x.wrapping_sub(y),
                (0b10001, false) => {
                    if x & y != 0 {
                        all
                    } else {
                        0
                    }
                }
                (0b10001, true) => {
                    if x == y {
                        all
                    } else {
                        0
                    }
                }
                (0b10010, false) => acc.wrapping_add(x.wrapping_mul(y)),
                (0b10010, true) => acc.wrapping_sub(x.wrapping_mul(y)),
                (0b10011, false) => x.wrapping_mul(y),
                (0b10011, true) => {
                    // PMUL (bytes only): carry-less multiply.
                    let mut p = 0u64;
                    for bit in 0..8 {
                        if (y >> bit) & 1 == 1 {
                            p ^= x << bit;
                        }
                    }
                    p
                }
                (0b10110, _) => {
                    // SQDMULH / SQRDMULH
                    let prod = 2 * sxv * syv + if u { 1i128 << (esize - 1) } else { 0 };
                    let (v, s) = saturate(prod >> esize, esize, true);
                    sat |= s;
                    v
                }
                _ => return Err(self.undef(pc)),
            };
            set_elem(&mut r, esize, i, v);
        }
        if sat {
            self.fpsr |= 1 << 27;
        }
        if scalar {
            self.set_scalar(rd, r as u64, esize);
        } else {
            self.set_v(rd, r, q);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fp_three_same(
        &mut self,
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: u8,
        a: u128,
        b: u128,
        d: u128,
        pc: u64,
    ) -> R {
        let f = fmt_of(size & 1 == 1);
        let hi = size >> 1 == 1;
        let esize = f.bits();
        let datasize = if scalar {
            esize
        } else if q {
            128
        } else {
            64
        };
        if !scalar && !q && f == Fmt::D {
            return Err(self.undef(pc));
        }
        let n = datasize / esize;
        let dn = self.dn();
        let all = mask_bits(esize);
        let pairwise =
            u && matches!(opcode, 0b11000 | 0b11010 | 0b11110) && !(opcode == 0b11010 && hi);
        if pairwise {
            if scalar {
                return Err(self.undef(pc));
            }
            let op = match (opcode, hi) {
                (0b11000, false) => Op2::MaxNum,
                (0b11000, true) => Op2::MinNum,
                (0b11010, _) => Op2::Add,
                (0b11110, false) => Op2::Max,
                _ => Op2::Min,
            };
            let mut r = 0u128;
            for i in 0..n {
                let (src, k) = if i < n / 2 {
                    (a, 2 * i)
                } else {
                    (b, 2 * (i - n / 2))
                };
                set_elem(
                    &mut r,
                    esize,
                    i,
                    fp::op2(f, op, elem(src, esize, k), elem(src, esize, k + 1), dn),
                );
            }
            self.set_v(rd, r, q);
            return Ok(());
        }
        let mut r = 0u128;
        for i in 0..n {
            let (x, y, acc) = (elem(a, esize, i), elem(b, esize, i), elem(d, esize, i));
            let v = match (u, opcode, hi) {
                (false, 0b11000, false) => fp::op2(f, Op2::MaxNum, x, y, dn),
                (false, 0b11000, true) => fp::op2(f, Op2::MinNum, x, y, dn),
                (false, 0b11001, false) => fp::fma(f, acc, x, y, dn),
                (false, 0b11001, true) => fp::fma(f, acc, x ^ f.sign_bit(), y, dn),
                (false, 0b11010, false) => fp::op2(f, Op2::Add, x, y, dn),
                (false, 0b11010, true) => fp::op2(f, Op2::Sub, x, y, dn),
                (false, 0b11011, false) => fp::op2(f, Op2::MulX, x, y, dn),
                (false, 0b11100, false) => {
                    if fp::cmp_pred(f, x, y, CmpOp::Eq) {
                        all
                    } else {
                        0
                    }
                }
                (false, 0b11110, false) => fp::op2(f, Op2::Max, x, y, dn),
                (false, 0b11110, true) => fp::op2(f, Op2::Min, x, y, dn),
                (true, 0b11010, true) => fp::op2(f, Op2::Abd, x, y, dn),
                (true, 0b11011, false) => fp::op2(f, Op2::Mul, x, y, dn),
                (true, 0b11100, false) => {
                    if fp::cmp_pred(f, x, y, CmpOp::Ge) {
                        all
                    } else {
                        0
                    }
                }
                (true, 0b11100, true) => {
                    if fp::cmp_pred(f, x, y, CmpOp::Gt) {
                        all
                    } else {
                        0
                    }
                }
                (true, 0b11101, false) => {
                    if fp::cmp_pred(f, x, y, CmpOp::AbsGe) {
                        all
                    } else {
                        0
                    }
                }
                (true, 0b11101, true) => {
                    if fp::cmp_pred(f, x, y, CmpOp::AbsGt) {
                        all
                    } else {
                        0
                    }
                }
                (true, 0b11111, false) => fp::op2(f, Op2::Div, x, y, dn),
                _ => return Err(self.undef(pc)),
            };
            set_elem(&mut r, esize, i, v);
        }
        if scalar {
            self.set_scalar(rd, r as u64, esize);
        } else {
            self.set_v(rd, r, q);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn three_diff(
        &mut self,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: u8,
        rn: u8,
        rm: u8,
        pc: u64,
    ) -> R {
        if size == 3 {
            return Err(self.undef(pc));
        }
        let esize = 8u32 << size;
        let wide = 2 * esize;
        let n = 64 / esize;
        let part = if q { n } else { 0 };
        let (a, b, d) = (self.v(rn), self.v(rm), self.v(rd));
        let ext = |x: u64| -> i128 {
            if u {
                i128::from(x)
            } else {
                i128::from(sx(x, esize))
            }
        };
        let mut r = 0u128;
        match opcode {
            0b0100 | 0b0110 => {
                // ADDHN/SUBHN (RADDHN/RSUBHN with u): high halves into the narrow part.
                let mut out = if q { d & LOW64 } else { 0 };
                for i in 0..n {
                    let (x, y) = (elem(a, wide, i), elem(b, wide, i));
                    let mut s = if opcode == 0b0100 {
                        u128::from(x) + u128::from(y)
                    } else {
                        u128::from(x).wrapping_sub(u128::from(y))
                    };
                    if u {
                        s = s.wrapping_add(1 << (esize - 1));
                    }
                    set_elem(&mut out, esize, part + i, (s >> esize) as u64);
                }
                self.set_v(rd, out, true);
                return Ok(());
            }
            _ => {}
        }
        for i in 0..n {
            let x = if matches!(opcode, 0b0001 | 0b0011) {
                elem(a, wide, i)
            } else {
                elem(a, esize, part + i)
            };
            let y = elem(b, esize, part + i);
            let acc = elem(d, wide, i);
            let xe = if matches!(opcode, 0b0001 | 0b0011) {
                if u {
                    i128::from(x)
                } else {
                    i128::from(sx(x, wide))
                }
            } else {
                ext(x)
            };
            let ye = ext(y);
            let v: i128 = match opcode {
                0b0000 | 0b0001 => xe + ye,
                0b0010 | 0b0011 => xe - ye,
                0b0101 => i128::from(acc) + (xe - ye).abs(),
                0b0111 => (xe - ye).abs(),
                0b1000 => i128::from(acc) + xe * ye,
                0b1010 => i128::from(acc) - xe * ye,
                0b1100 => xe * ye,
                0b1110 if !u && size == 0 => {
                    let mut p = 0u64;
                    for bit in 0..8 {
                        if (y >> bit) & 1 == 1 {
                            p ^= x << bit;
                        }
                    }
                    i128::from(p)
                }
                _ => return Err(self.undef(pc)),
            };
            set_elem(&mut r, wide, i, v as u64);
        }
        self.set_v(rd, r, true);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn two_misc(
        &mut self,
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: u8,
        rn: u8,
        pc: u64,
    ) -> R {
        let a = self.v(rn);
        let fp_op = opcode >= 0b01100 && !matches!(opcode, 0b10010..=0b10100);
        if fp_op {
            return self.fp_two_misc(scalar, q, u, size, opcode, rd, a, pc);
        }
        let esize = 8u32 << size;
        let datasize = if scalar {
            esize
        } else if q {
            128
        } else {
            64
        };
        let n = datasize / esize;
        let all = mask_bits(esize);
        // Narrowing: XTN/SQXTN/UQXTN/SQXTUN write the low or (Q) high half of rd.
        if matches!(opcode, 0b10010 | 0b10100) {
            if size == 3 {
                return Err(self.undef(pc));
            }
            let wide = 2 * esize;
            let cnt = if scalar { 1 } else { 64 / esize };
            let part = if q && !scalar { cnt } else { 0 };
            let mut out = if q && !scalar { self.v(rd) & LOW64 } else { 0 };
            let mut sat = false;
            for i in 0..cnt {
                let x = elem(a, wide, i);
                let v = match (opcode, u) {
                    (0b10010, false) => x,
                    (0b10010, true) => {
                        let (v, s) = saturate(i128::from(sx(x, wide)), esize, false);
                        sat |= s;
                        v
                    }
                    (_, false) => {
                        let (v, s) = saturate(i128::from(sx(x, wide)), esize, true);
                        sat |= s;
                        v
                    }
                    (_, true) => {
                        let (v, s) = saturate(i128::from(x), esize, false);
                        sat |= s;
                        v
                    }
                };
                set_elem(&mut out, esize, part + i, v);
            }
            if sat {
                self.fpsr |= 1 << 27;
            }
            self.set_v(rd, out, true);
            return Ok(());
        }
        if opcode == 0b10011 {
            // SHLL: widen and shift left by esize.
            if !u || size == 3 {
                return Err(self.undef(pc));
            }
            let part = if q { 64 / esize } else { 0 };
            let mut out = 0u128;
            for i in 0..64 / esize {
                set_elem(&mut out, 2 * esize, i, elem(a, esize, part + i) << esize);
            }
            self.set_v(rd, out, true);
            return Ok(());
        }
        if matches!(opcode, 0b00010 | 0b00110) {
            // SADDLP/UADDLP (and accumulating SADALP/UADALP).
            if size == 3 {
                return Err(self.undef(pc));
            }
            let wide = 2 * esize;
            let d = self.v(rd);
            let mut out = 0u128;
            for i in 0..(n / 2) {
                let (x, y) = (elem(a, esize, 2 * i), elem(a, esize, 2 * i + 1));
                let s = if u {
                    x + y
                } else {
                    (sx(x, esize) + sx(y, esize)) as u64
                };
                let s = if opcode == 0b00110 {
                    s.wrapping_add(elem(d, wide, i))
                } else {
                    s
                };
                set_elem(&mut out, wide, i, s);
            }
            self.set_v(rd, out, q);
            return Ok(());
        }
        let mut r = 0u128;
        for i in 0..n {
            let x = elem(a, esize, i);
            let s = sx(x, esize);
            let v = match (opcode, u) {
                (0b00000, _) | (0b00001, false) => {
                    // REV64 / REV32 / REV16: reverse elements within containers.
                    let container = match (opcode, u) {
                        (0b00000, false) => 64,
                        (0b00000, true) => 32,
                        _ => 16,
                    };
                    if esize >= container {
                        return Err(self.undef(pc));
                    }
                    let per = container / esize;
                    let base = i - i % per;
                    elem(a, esize, base + (per - 1 - i % per))
                }
                (0b00100, false) => {
                    let clz = if s < 0 {
                        (!x & all).leading_zeros()
                    } else {
                        x.leading_zeros()
                    };
                    u64::from(clz - (64 - esize)).saturating_sub(1)
                }
                (0b00100, true) => u64::from(x.leading_zeros() - (64 - esize)),
                (0b00101, false) if size == 0 => u64::from(x.count_ones()),
                (0b00101, true) if size == 0 => !x & all,
                (0b00101, true) if size == 1 => u64::from((x as u8).reverse_bits()),
                (0b01000, false) => {
                    if s > 0 {
                        all
                    } else {
                        0
                    }
                }
                (0b01000, true) => {
                    if s >= 0 {
                        all
                    } else {
                        0
                    }
                }
                (0b01001, false) => {
                    if x == 0 {
                        all
                    } else {
                        0
                    }
                }
                (0b01001, true) => {
                    if s <= 0 {
                        all
                    } else {
                        0
                    }
                }
                (0b01010, false) => {
                    if s < 0 {
                        all
                    } else {
                        0
                    }
                }
                (0b01011, false) => s.unsigned_abs() & all,
                (0b01011, true) => (s.wrapping_neg() as u64) & all,
                (0b00111, _) => {
                    let exact = if u {
                        -i128::from(s)
                    } else {
                        i128::from(s).abs()
                    };
                    let (v, sat) = saturate(exact, esize, true);
                    if sat {
                        self.fpsr |= 1 << 27;
                    }
                    v
                }
                _ => return Err(self.undef(pc)),
            };
            set_elem(&mut r, esize, i, v);
        }
        if scalar {
            self.set_scalar(rd, r as u64, esize);
        } else {
            self.set_v(rd, r, q);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fp_two_misc(
        &mut self,
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: u8,
        a: u128,
        pc: u64,
    ) -> R {
        let f = fmt_of(size & 1 == 1);
        let hi = size >> 1 == 1;
        let dn = self.dn();
        // FCVTL / FCVTN between S and D (half precision not supported).
        if !scalar && matches!(opcode, 0b10110 | 0b10111) && !hi && !u {
            if size & 1 == 0 {
                return Err(self.undef(pc)); // half <-> single
            }
            if opcode == 0b10111 {
                let base = if q { 2 } else { 0 };
                let mut out = 0u128;
                for i in 0..2 {
                    set_elem(
                        &mut out,
                        64,
                        i,
                        fp::convert(Fmt::S, Fmt::D, elem(a, 32, base + i), dn),
                    );
                }
                self.set_v(rd, out, true);
            } else {
                let mut out = if q { self.v(rd) & LOW64 } else { 0 };
                let base = if q { 2 } else { 0 };
                for i in 0..2 {
                    set_elem(
                        &mut out,
                        32,
                        base + i,
                        fp::convert(Fmt::D, Fmt::S, elem(a, 64, i), dn),
                    );
                }
                self.set_v(rd, out, true);
            }
            return Ok(());
        }
        let esize = f.bits();
        let datasize = if scalar {
            esize
        } else if q {
            128
        } else {
            64
        };
        if !scalar && !q && f == Fmt::D {
            return Err(self.undef(pc));
        }
        let n = datasize / esize;
        let all = mask_bits(esize);
        let mut r = 0u128;
        for i in 0..n {
            let x = elem(a, esize, i);
            let v = match (u, opcode, hi) {
                (false, 0b01100, true) => {
                    if fp::cmp_pred(f, x, 0, CmpOp::Gt) {
                        all
                    } else {
                        0
                    }
                }
                (false, 0b01101, true) => {
                    if fp::cmp_pred(f, x, 0, CmpOp::Eq) {
                        all
                    } else {
                        0
                    }
                }
                (false, 0b01110, true) => {
                    if fp::cmp_pred(f, x, 0, CmpOp::Lt) {
                        all
                    } else {
                        0
                    }
                }
                (true, 0b01100, true) => {
                    if fp::cmp_pred(f, x, 0, CmpOp::Ge) {
                        all
                    } else {
                        0
                    }
                }
                (true, 0b01101, true) => {
                    if fp::cmp_pred(f, x, 0, CmpOp::Le) {
                        all
                    } else {
                        0
                    }
                }
                (false, 0b01111, true) => x & !f.sign_bit(),
                (true, 0b01111, true) => x ^ f.sign_bit(),
                (true, 0b11111, true) => fp::sqrt(f, x, dn),
                (false, 0b11000, false) => fp::round_int(f, x, Round::Nearest, dn),
                (false, 0b11001, false) => fp::round_int(f, x, Round::Down, dn),
                (false, 0b11000, true) => fp::round_int(f, x, Round::Up, dn),
                (false, 0b11001, true) => fp::round_int(f, x, Round::Zero, dn),
                (true, 0b11000, false) => fp::round_int(f, x, Round::Away, dn),
                (true, 0b11001, _) => fp::round_int(f, x, Round::Nearest, dn),
                (_, 0b11010, false) => fp::to_int(f, x, Round::Nearest, !u, esize),
                (_, 0b11011, false) => fp::to_int(f, x, Round::Down, !u, esize),
                (_, 0b11100, false) => fp::to_int(f, x, Round::Away, !u, esize),
                (_, 0b11010, true) => fp::to_int(f, x, Round::Up, !u, esize),
                (_, 0b11011, true) => fp::to_int(f, x, Round::Zero, !u, esize),
                (_, 0b11101, false) => fp::from_int(f, x, !u, esize),
                _ => return Err(self.undef(pc)),
            };
            set_elem(&mut r, esize, i, v);
        }
        if scalar {
            self.set_scalar(rd, r as u64, esize);
        } else {
            self.set_v(rd, r, q);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn across(&mut self, q: bool, u: bool, size: u8, opcode: u8, rd: u8, rn: u8, pc: u64) -> R {
        let a = self.v(rn);
        if matches!(opcode, 0b01100 | 0b01111) {
            if !u || !q || size & 1 == 1 {
                return Err(self.undef(pc));
            }
            let op = match (opcode, size >> 1) {
                (0b01100, 0) => Op2::MaxNum,
                (0b01100, _) => Op2::MinNum,
                (_, 0) => Op2::Max,
                _ => Op2::Min,
            };
            let dn = self.dn();
            let l = fp::op2(Fmt::S, op, elem(a, 32, 0), elem(a, 32, 1), dn);
            let h = fp::op2(Fmt::S, op, elem(a, 32, 2), elem(a, 32, 3), dn);
            self.set_scalar(rd, fp::op2(Fmt::S, op, l, h, dn), 32);
            return Ok(());
        }
        let esize = 8u32 << size;
        let n = if q { 128 } else { 64 } / esize;
        if size == 3 || n < 2 {
            return Err(self.undef(pc));
        }
        let xs = (0..n).map(|i| elem(a, esize, i));
        let (value, out_size) = match (opcode, u) {
            (0b00011, false) => (xs.map(|x| sx(x, esize)).sum::<i64>() as u64, 2 * esize),
            (0b00011, true) => (xs.sum::<u64>(), 2 * esize),
            (0b01010, false) => (xs.max_by_key(|&x| sx(x, esize)).unwrap_or(0), esize),
            (0b01010, true) => (xs.max().unwrap_or(0), esize),
            (0b11010, false) => (xs.min_by_key(|&x| sx(x, esize)).unwrap_or(0), esize),
            (0b11010, true) => (xs.min().unwrap_or(0), esize),
            (0b11011, false) => (xs.fold(0u64, u64::wrapping_add), esize),
            _ => return Err(self.undef(pc)),
        };
        self.set_scalar(rd, value, out_size);
        Ok(())
    }

    fn scalar_pairwise(&mut self, u: bool, size: u8, opcode: u8, rd: u8, rn: u8, pc: u64) -> R {
        let a = self.v(rn);
        if !u && opcode == 0b11011 && size == 3 {
            self.set_scalar(rd, elem(a, 64, 0).wrapping_add(elem(a, 64, 1)), 64);
            return Ok(());
        }
        if !u {
            return Err(self.undef(pc));
        }
        let f = fmt_of(size & 1 == 1);
        let op = match (opcode, size >> 1) {
            (0b01100, 0) => Op2::MaxNum,
            (0b01100, _) => Op2::MinNum,
            (0b01101, 0) => Op2::Add,
            (0b01111, 0) => Op2::Max,
            (0b01111, _) => Op2::Min,
            _ => return Err(self.undef(pc)),
        };
        let e = f.bits();
        self.set_scalar(
            rd,
            fp::op2(f, op, elem(a, e, 0), elem(a, e, 1), self.dn()),
            e,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn copy(&mut self, q: bool, op: bool, imm5: u8, imm4: u8, rd: u8, rn: u8, pc: u64) -> R {
        let size = imm5.trailing_zeros();
        let esize = 8u32 << size;
        let index = u32::from(imm5) >> (size + 1);
        if op {
            // INS (element)
            if !q {
                return Err(self.undef(pc));
            }
            let src = u32::from(imm4) >> size;
            let x = elem(self.v(rn), esize, src);
            let mut d = self.v(rd);
            set_elem(&mut d, esize, index, x);
            self.vreg[usize::from(rd)] = d;
            return Ok(());
        }
        match imm4 {
            0b0000 | 0b0001 => {
                if size == 3 && !q {
                    return Err(self.undef(pc));
                }
                let x = if imm4 == 0 {
                    elem(self.v(rn), esize, index)
                } else {
                    self.gp(rn) & mask_bits(esize)
                };
                let n = if q { 128 } else { 64 } / esize;
                let mut r = 0u128;
                for i in 0..n {
                    set_elem(&mut r, esize, i, x);
                }
                self.set_v(rd, r, q);
            }
            0b0011 => {
                let mut d = self.v(rd);
                set_elem(&mut d, esize, index, self.gp(rn));
                self.vreg[usize::from(rd)] = d;
            }
            0b0101 => {
                let v = sx(elem(self.v(rn), esize, index), esize) as u64;
                self.write_gp(rd, q, v);
            }
            0b0111 => {
                let v = elem(self.v(rn), esize, index);
                self.write_gp(rd, q, v);
            }
            _ => return Err(self.undef(pc)),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn shift_imm(
        &mut self,
        scalar: bool,
        q: bool,
        u: bool,
        immh: u8,
        immb: u8,
        opcode: u8,
        rd: u8,
        rn: u8,
        pc: u64,
    ) -> R {
        let hsb = 7 - immh.leading_zeros(); // index of the highest set bit of immh (0..3)
        let esize = 8u32 << hsb;
        let immhb = (u32::from(immh) << 3) | u32::from(immb);
        let a = self.v(rn);
        let d = self.v(rd);
        // Widening (SSHLL/USHLL) and narrowing (SHRN family) forms.
        if opcode == 0b10100 {
            if scalar || esize == 64 {
                return Err(self.undef(pc));
            }
            let shift = immhb - esize;
            let part = if q { 64 / esize } else { 0 };
            let mut r = 0u128;
            for i in 0..64 / esize {
                let x = elem(a, esize, part + i);
                let x = if u { x } else { sx(x, esize) as u64 };
                set_elem(&mut r, 2 * esize, i, x << shift);
            }
            self.set_v(rd, r, true);
            return Ok(());
        }
        if (0b10000..=0b10011).contains(&opcode) {
            if esize == 64 {
                return Err(self.undef(pc));
            }
            let wide = 2 * esize;
            let shift = 2 * esize - immhb;
            let cnt = if scalar { 1 } else { 64 / esize };
            let part = if q && !scalar { cnt } else { 0 };
            let mut out = if q && !scalar { d & LOW64 } else { 0 };
            let round = opcode & 1 == 1;
            let mut sat = false;
            for i in 0..cnt {
                let x = elem(a, wide, i);
                let signed_src = !(u && opcode >= 0b10010);
                let mut v = if signed_src {
                    i128::from(sx(x, wide))
                } else {
                    i128::from(x)
                };
                if round {
                    v += 1 << (shift - 1);
                }
                v >>= shift;
                let r = match (opcode >> 1, u) {
                    (0b1000, false) => v as u64 & mask_bits(esize), // SHRN/RSHRN
                    (0b1000, true) => {
                        let (r, s) = saturate(v, esize, false); // SQSHRUN/SQRSHRUN
                        sat |= s;
                        r
                    }
                    (_, signed) => {
                        let (r, s) = saturate(v, esize, !signed); // SQSHRN/UQSHRN (+ rounding)
                        sat |= s;
                        r
                    }
                };
                set_elem(&mut out, esize, part + i, r);
            }
            if sat {
                self.fpsr |= 1 << 27;
            }
            if scalar {
                self.set_scalar(rd, out as u64, esize);
            } else {
                self.set_v(rd, out, true);
            }
            return Ok(());
        }
        if matches!(opcode, 0b11100 | 0b11111) {
            // Fixed-point <-> FP (vector/scalar): esize 32 or 64.
            if esize < 32 {
                return Err(self.undef(pc));
            }
            let f = fmt_of(esize == 64);
            let fbits = 2 * esize - immhb;
            let scale = pow2(fbits);
            let n = if scalar {
                1
            } else if q {
                128 / esize
            } else {
                64 / esize
            };
            let mut r = 0u128;
            for i in 0..n {
                let x = elem(a, esize, i);
                let v = if opcode == 0b11100 {
                    let val = if u { x as f64 } else { sx(x, esize) as f64 } / scale;
                    match f {
                        Fmt::S => u64::from((val as f32).to_bits()),
                        Fmt::D => val.to_bits(),
                    }
                } else if f.is_nan(x) {
                    0
                } else {
                    fp::to_int(
                        Fmt::D,
                        (f.to_f64(x) * scale).to_bits(),
                        Round::Zero,
                        !u,
                        esize,
                    )
                };
                set_elem(&mut r, esize, i, v);
            }
            if scalar {
                self.set_scalar(rd, r as u64, esize);
            } else {
                self.set_v(rd, r, q);
            }
            return Ok(());
        }
        if !scalar && esize == 64 && !q {
            return Err(self.undef(pc));
        }
        if scalar && esize != 64 && !matches!(opcode, 0b01110 | 0b01100) {
            return Err(self.undef(pc));
        }
        let right = 2 * esize - immhb;
        let left = immhb - esize;
        let n = if scalar {
            1
        } else if q {
            128 / esize
        } else {
            64 / esize
        };
        let mut r = 0u128;
        let mut sat = false;
        for i in 0..n {
            let x = elem(a, esize, i);
            let acc = elem(d, esize, i);
            let v = match opcode {
                0b00000 | 0b00010 | 0b00100 | 0b00110 => {
                    let round = opcode & 0b00100 != 0;
                    let shifted = shift_by(x, esize, -i64::from(right), !u, round);
                    if opcode & 0b00010 != 0 {
                        acc.wrapping_add(shifted)
                    } else {
                        shifted
                    }
                }
                0b01000 if u => {
                    // SRI
                    let m = if right >= 64 {
                        0
                    } else {
                        mask_bits(esize) >> right
                    };
                    (acc & !m) | (if right >= 64 { 0 } else { x >> right } & m)
                }
                0b01010 => {
                    let shifted = if left >= 64 { 0 } else { x << left };
                    if u {
                        let m = (mask_bits(esize) << left) & mask_bits(esize);
                        (acc & !m) | (shifted & m)
                    } else {
                        shifted
                    }
                }
                0b01100 | 0b01110 => {
                    // SQSHLU (u, 01100), SQSHL/UQSHL (01110)
                    let src_signed = !(u && opcode == 0b01110);
                    let v = if src_signed {
                        i128::from(sx(x, esize))
                    } else {
                        i128::from(x)
                    };
                    let exact = v << left;
                    let dst_signed = opcode == 0b01110 && !u;
                    let (r, s) = saturate(exact, esize, dst_signed);
                    sat |= s;
                    r
                }
                _ => return Err(self.undef(pc)),
            };
            set_elem(&mut r, esize, i, v & mask_bits(esize));
        }
        if sat {
            self.fpsr |= 1 << 27;
        }
        if scalar {
            self.set_scalar(rd, r as u64, esize);
        } else {
            self.set_v(rd, r, q);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn indexed(
        &mut self,
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        l: bool,
        m: bool,
        h: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        pc: u64,
    ) -> R {
        let a = self.v(rn);
        let d = self.v(rd);
        let fp_op = matches!(opcode, 0b0001 | 0b0101 | 0b1001) && !(u && opcode != 0b1001);
        if fp_op {
            let f = fmt_of(size & 1 == 1);
            if size >> 1 != 1 {
                return Err(self.undef(pc));
            }
            let (index, reg) = if f == Fmt::D {
                if l {
                    return Err(self.undef(pc));
                }
                (u32::from(h), rm | (u8::from(m) << 4))
            } else {
                ((u32::from(h) << 1) | u32::from(l), rm | (u8::from(m) << 4))
            };
            let esize = f.bits();
            let y = elem(self.v(reg), esize, index);
            let n = if scalar {
                1
            } else if q {
                128 / esize
            } else {
                64 / esize
            };
            let dn = self.dn();
            let mut r = 0u128;
            for i in 0..n {
                let x = elem(a, esize, i);
                let v = match (opcode, u) {
                    (0b0001, _) => fp::fma(f, elem(d, esize, i), x, y, dn),
                    (0b0101, _) => fp::fma(f, elem(d, esize, i), x ^ f.sign_bit(), y, dn),
                    (0b1001, false) => fp::op2(f, Op2::Mul, x, y, dn),
                    _ => fp::op2(f, Op2::MulX, x, y, dn),
                };
                set_elem(&mut r, esize, i, v);
            }
            if scalar {
                self.set_scalar(rd, r as u64, esize);
            } else {
                self.set_v(rd, r, q);
            }
            return Ok(());
        }
        let (index, reg) = match size {
            1 => ((u32::from(h) << 2) | (u32::from(l) << 1) | u32::from(m), rm),
            2 => ((u32::from(h) << 1) | u32::from(l), rm | (u8::from(m) << 4)),
            _ => return Err(self.undef(pc)),
        };
        if scalar {
            return Err(self.undef(pc));
        }
        let esize = 8u32 << size;
        let y = elem(self.v(reg), esize, index);
        match opcode {
            0b1000 | 0b0000 | 0b0100 => {
                if opcode == 0b1000 && u {
                    return Err(self.undef(pc));
                }
                let n = if q { 128 } else { 64 } / esize;
                let mut r = 0u128;
                for i in 0..n {
                    let (x, acc) = (elem(a, esize, i), elem(d, esize, i));
                    let p = x.wrapping_mul(y);
                    let v = match opcode {
                        0b1000 => p,
                        0b0000 => acc.wrapping_add(p),
                        _ => acc.wrapping_sub(p),
                    };
                    set_elem(&mut r, esize, i, v);
                }
                self.set_v(rd, r, q);
            }
            0b1010 | 0b0010 | 0b0110 => {
                let part = if q { 64 / esize } else { 0 };
                let wide = 2 * esize;
                let ye = if u {
                    i128::from(y)
                } else {
                    i128::from(sx(y, esize))
                };
                let mut r = 0u128;
                for i in 0..64 / esize {
                    let x = elem(a, esize, part + i);
                    let xe = if u {
                        i128::from(x)
                    } else {
                        i128::from(sx(x, esize))
                    };
                    let acc = i128::from(elem(d, wide, i));
                    let v = match opcode {
                        0b1010 => xe * ye,
                        0b0010 => acc + xe * ye,
                        _ => acc - xe * ye,
                    };
                    set_elem(&mut r, wide, i, v as u64);
                }
                self.set_v(rd, r, true);
            }
            _ => return Err(self.undef(pc)),
        }
        Ok(())
    }

    fn fp1(&mut self, ftype: u8, opcode: u8, rd: u8, rn: u8, pc: u64) -> R {
        let dn = self.dn();
        // FCVT between precisions: opcode 0001xx, target in bits 1:0.
        if opcode & 0b111100 == 0b000100 {
            let from = fmt_of_ftype(ftype).ok_or_else(|| self.undef(pc))?;
            let to = fmt_of_ftype(opcode & 3).ok_or_else(|| self.undef(pc))?;
            if from == to {
                return Err(self.undef(pc));
            }
            let x = elem(self.v(rn), from.bits(), 0);
            self.set_scalar(rd, fp::convert(from, to, x, dn), to.bits());
            return Ok(());
        }
        let f = fmt_of_ftype(ftype).ok_or_else(|| self.undef(pc))?;
        let esize = f.bits();
        let x = elem(self.v(rn), esize, 0);
        let r = match opcode {
            0b000000 => x,
            0b000001 => x & !f.sign_bit(),
            0b000010 => x ^ f.sign_bit(),
            0b000011 => fp::sqrt(f, x, dn),
            0b001000 => fp::round_int(f, x, Round::Nearest, dn),
            0b001001 => fp::round_int(f, x, Round::Up, dn),
            0b001010 => fp::round_int(f, x, Round::Down, dn),
            0b001011 => fp::round_int(f, x, Round::Zero, dn),
            0b001100 => fp::round_int(f, x, Round::Away, dn),
            0b001110 | 0b001111 => fp::round_int(f, x, Round::Nearest, dn),
            _ => return Err(self.undef(pc)),
        };
        self.set_scalar(rd, r, esize);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fp_int(&mut self, sf: bool, ftype: u8, rmode: u8, opcode: u8, rd: u8, rn: u8, pc: u64) -> R {
        let width = if sf { 64 } else { 32 };
        // FMOV between general and FP registers.
        match (sf, ftype, rmode, opcode) {
            (false, 0, 0, 0b110) | (true, 1, 0, 0b110) => {
                let v = elem(self.v(rn), width, 0);
                self.write_gp(rd, sf, v);
                return Ok(());
            }
            (false, 0, 0, 0b111) | (true, 1, 0, 0b111) => {
                let v = self.gp(rn) & mask_bits(width);
                self.set_scalar(rd, v, width);
                return Ok(());
            }
            (true, 2, 1, 0b110) => {
                let v = elem(self.v(rn), 64, 1);
                self.write_gp(rd, true, v);
                return Ok(());
            }
            (true, 2, 1, 0b111) => {
                let mut d = self.v(rd);
                set_elem(&mut d, 64, 1, self.gp(rn));
                self.vreg[usize::from(rd)] = d;
                return Ok(());
            }
            _ => {}
        }
        let f = fmt_of_ftype(ftype).ok_or_else(|| self.undef(pc))?;
        match opcode {
            0b010 | 0b011 if rmode == 0 => {
                let x = self.gp(rn);
                self.set_scalar(rd, fp::from_int(f, x, opcode == 0b010, width), f.bits());
            }
            0b000 | 0b001 | 0b100 | 0b101 => {
                let mode = match (opcode >> 2, rmode) {
                    (1, 0) => Round::Away,
                    (0, 0) => Round::Nearest,
                    (0, 1) => Round::Up,
                    (0, 2) => Round::Down,
                    (0, 3) => Round::Zero,
                    _ => return Err(self.undef(pc)),
                };
                let x = elem(self.v(rn), f.bits(), 0);
                let v = fp::to_int(f, x, mode, opcode & 1 == 0, width);
                self.write_gp(rd, sf, v);
            }
            _ => return Err(self.undef(pc)),
        }
        Ok(())
    }

    /// Base address for a SIMD structure access (SP must be 16-byte aligned).
    fn struct_base(&self, rn: u8, pc: u64) -> Result<u64, Fault> {
        let base = self.x[usize::from(rn)];
        if rn == SP && base & 0xf != 0 {
            return Err(Fault::Alignment { pc, addr: base });
        }
        Ok(base)
    }

    fn post_index(&mut self, rn: u8, post: Option<u8>, total: u64) {
        if let Some(rm) = post {
            let inc = if rm == ZR {
                total
            } else {
                self.x[usize::from(rm)]
            };
            self.x[usize::from(rn)] = self.x[usize::from(rn)].wrapping_add(inc);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ldst_multi(
        &mut self,
        mem: &mut dyn GuestMemory,
        load: bool,
        q: bool,
        size: u8,
        regs: u8,
        selem: u8,
        rt: u8,
        rn: u8,
        post: Option<u8>,
        pc: u64,
    ) -> R {
        let base = self.struct_base(rn, pc)?;
        let mem_fault = |fault: MemFault| Fault::Memory { pc, fault };
        let esize = 8u32 << size;
        let ebytes = u64::from(esize / 8);
        let elements = if q { 128 } else { 64 } / esize;
        let rpt = regs / selem; // repetitions (LD1 with 2-4 registers)
        let total = u64::from(regs) * u64::from(elements) * ebytes;
        let mut buf = vec_buf(total as usize);
        if load {
            mem.read(base, &mut buf).map_err(mem_fault)?;
            let mut out: [u128; 4] = [0; 4];
            let mut off = 0usize;
            for r in 0..rpt {
                for e in 0..elements {
                    for s in 0..selem {
                        let idx = usize::from(r + s);
                        let x = le_u64(&buf[off..off + ebytes as usize]);
                        set_elem(&mut out[idx], esize, e, x);
                        off += ebytes as usize;
                    }
                }
            }
            for (i, v) in out.iter().enumerate().take(usize::from(regs)) {
                self.set_v((rt + i as u8) % 32, *v, q);
            }
        } else {
            let mut off = 0usize;
            for r in 0..rpt {
                for e in 0..elements {
                    for s in 0..selem {
                        let x = elem(self.v((rt + r + s) % 32), esize, e);
                        buf[off..off + ebytes as usize]
                            .copy_from_slice(&x.to_le_bytes()[..ebytes as usize]);
                        off += ebytes as usize;
                    }
                }
            }
            mem.write(base, &buf).map_err(mem_fault)?;
        }
        self.post_index(rn, post, total);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn ldst_single(
        &mut self,
        mem: &mut dyn GuestMemory,
        load: bool,
        replicate: bool,
        q: bool,
        esize_log: u8,
        index: u8,
        selem: u8,
        rt: u8,
        rn: u8,
        post: Option<u8>,
        pc: u64,
    ) -> R {
        let base = self.struct_base(rn, pc)?;
        let mem_fault = |fault: MemFault| Fault::Memory { pc, fault };
        let esize = 8u32 << esize_log;
        let ebytes = (esize / 8) as usize;
        let total = (ebytes * usize::from(selem)) as u64;
        let mut buf = [0u8; 32];
        let buf = &mut buf[..total as usize];
        if load {
            mem.read(base, buf).map_err(mem_fault)?;
            for s in 0..selem {
                let x = le_u64(&buf[usize::from(s) * ebytes..(usize::from(s) + 1) * ebytes]);
                let reg = (rt + s) % 32;
                if replicate {
                    let n = if q { 128 } else { 64 } / esize;
                    let mut r = 0u128;
                    for i in 0..n {
                        set_elem(&mut r, esize, i, x);
                    }
                    self.set_v(reg, r, q);
                } else {
                    let mut d = self.v(reg);
                    set_elem(&mut d, esize, u32::from(index), x);
                    self.vreg[usize::from(reg)] = d;
                }
            }
        } else {
            for s in 0..selem {
                let x = elem(self.v((rt + s) % 32), esize, u32::from(index));
                buf[usize::from(s) * ebytes..(usize::from(s) + 1) * ebytes]
                    .copy_from_slice(&x.to_le_bytes()[..ebytes]);
            }
            mem.write(base, buf).map_err(mem_fault)?;
        }
        self.post_index(rn, post, total);
        Ok(())
    }
}

/// Shifts `x` (an `esize`-bit element) left by `sh` (or right by `-sh`), signed or unsigned,
/// optionally rounding right shifts. Out-of-range shifts give the architectural result.
fn shift_by(x: u64, esize: u32, sh: i64, signed: bool, round: bool) -> u64 {
    let v: i128 = if signed {
        i128::from(sx(x, esize))
    } else {
        i128::from(x)
    };
    let r = if sh >= 0 {
        if sh >= 128 { 0 } else { v << sh }
    } else {
        let n = (-sh).min(127) as u32;
        let rounding = if round && n <= 127 {
            1i128 << (n - 1)
        } else {
            0
        };
        (v + rounding) >> n
    };
    (r as u64) & mask_bits(esize)
}

fn le_u64(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a[..b.len()].copy_from_slice(b);
    u64::from_le_bytes(a)
}

fn vec_buf(n: usize) -> alloc::vec::Vec<u8> {
    alloc::vec![0u8; n]
}

/// `2^n` exactly, for fixed-point scales (`n <= 64`).
fn pow2(n: u32) -> f64 {
    f64::from_bits(u64::from(1023 + n) << 52)
}
