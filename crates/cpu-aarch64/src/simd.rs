//! SIMD&FP instructions: the decoder.
//!
//! Instructions are classified by encoding group (Arm ARM C4.1.9x, "Data Processing --
//! Scalar Floating-Point and Advanced SIMD", and the SIMD load/store structure groups). Each
//! class keeps its raw selector fields (`u`, `size`, `opcode`, ...), and the interpreter
//! (`simd_exec.rs`) dispatches on them. Selector combinations the interpreter does not implement
//! become undefined-instruction faults when executed.

use crate::decode::{bit, bits, rsp};
use crate::inst::{Cond, Inst, Reg, VReg};

/// The four operations of "Advanced SIMD modified immediate".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModImmOp {
    /// `MOVI`/`FMOV` (vector, immediate): replace.
    Movi,
    /// `MVNI`: replace with the inverse.
    Mvni,
    /// `ORR` (vector, immediate).
    Orr,
    /// `BIC` (vector, immediate).
    Bic,
}

/// A decoded SIMD&FP instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdInst {
    /// Advanced SIMD three same (vector) or, with `scalar`, its scalar form.
    ThreeSame {
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
        rm: VReg,
    },
    /// Advanced SIMD three different (widening/narrowing).
    ThreeDiff {
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
        rm: VReg,
    },
    /// Advanced SIMD two-register miscellaneous (vector or scalar).
    TwoMisc {
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
    },
    /// Advanced SIMD across lanes.
    AcrossLanes {
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
    },
    /// Advanced SIMD scalar pairwise (`ADDP d, v.2d`, `FADDP`, ...).
    ScalarPairwise {
        u: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
    },
    /// Advanced SIMD copy: `DUP`, `SMOV`, `UMOV`, `INS`. `op` is bit 29.
    Copy {
        q: bool,
        op: bool,
        imm5: u8,
        imm4: u8,
        rd: VReg,
        rn: Reg,
    },
    /// Advanced SIMD scalar copy (`DUP`/`MOV` scalar from element).
    ScalarDup { imm5: u8, rd: VReg, rn: VReg },
    /// Advanced SIMD modified immediate: `imm` is the expanded 64-bit pattern.
    ModImm {
        q: bool,
        op: ModImmOp,
        imm: u64,
        rd: VReg,
    },
    /// Advanced SIMD shift by immediate (vector or scalar).
    ShiftImm {
        scalar: bool,
        q: bool,
        u: bool,
        immh: u8,
        immb: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
    },
    /// `UZP1/2`, `TRN1/2`, `ZIP1/2`.
    Permute {
        q: bool,
        size: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
        rm: VReg,
    },
    /// `EXT`.
    Ext {
        q: bool,
        index: u8,
        rd: VReg,
        rn: VReg,
        rm: VReg,
    },
    /// `TBL`/`TBX` with `len + 1` table registers.
    Tbl {
        q: bool,
        len: u8,
        tbx: bool,
        rd: VReg,
        rn: VReg,
        rm: VReg,
    },
    /// Advanced SIMD vector x indexed element (`MUL`, `FMLA`, ... by element).
    Indexed {
        scalar: bool,
        q: bool,
        u: bool,
        size: u8,
        opcode: u8,
        l: bool,
        m: bool,
        h: bool,
        rd: VReg,
        rn: VReg,
        rm: VReg,
    },

    /// FP data processing, one source (`FMOV`, `FABS`, `FNEG`, `FSQRT`, `FCVT`, `FRINT*`).
    Fp1 {
        ftype: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
    },
    /// FP data processing, two sources (`FMUL` ... `FNMUL`).
    Fp2 {
        ftype: u8,
        opcode: u8,
        rd: VReg,
        rn: VReg,
        rm: VReg,
    },
    /// FP data processing, three sources (`FMADD`, `FMSUB`, `FNMADD`, `FNMSUB`).
    Fp3 {
        ftype: u8,
        o1: bool,
        o0: bool,
        rd: VReg,
        rn: VReg,
        rm: VReg,
        ra: VReg,
    },
    /// `FCMP`/`FCMPE`, with `zero` comparing against +0.0.
    FpCmp {
        ftype: u8,
        zero: bool,
        rn: VReg,
        rm: VReg,
    },
    /// `FCCMP`/`FCCMPE`.
    FpCondCmp {
        ftype: u8,
        rn: VReg,
        rm: VReg,
        cond: Cond,
        nzcv: u8,
    },
    /// `FCSEL`.
    FpCsel {
        ftype: u8,
        rd: VReg,
        rn: VReg,
        rm: VReg,
        cond: Cond,
    },
    /// `FMOV` (scalar, immediate): `bits` is the expanded value.
    FpImm { ftype: u8, bits: u64, rd: VReg },
    /// Conversions between FP and integer registers (`SCVTF`, `FCVTZS`, `FMOV` general, ...).
    FpInt {
        sf: bool,
        ftype: u8,
        rmode: u8,
        opcode: u8,
        rd: u8,
        rn: u8,
    },
    /// Conversions between FP and fixed point (`fbits` fraction bits).
    FpFixed {
        sf: bool,
        ftype: u8,
        rmode: u8,
        opcode: u8,
        fbits: u8,
        rd: u8,
        rn: u8,
    },

    /// `LD1`-`LD4` / `ST1`-`ST4` (multiple structures). `regs` registers, `selem` elements per
    /// structure. `post` is the post-index: `Some(Reg)` (ZR means "by the transfer size").
    LdStMulti {
        load: bool,
        q: bool,
        size: u8,
        regs: u8,
        selem: u8,
        rt: VReg,
        rn: Reg,
        post: Option<Reg>,
    },
    /// Single-structure (one lane, or replicate with `replicate`) loads and stores.
    LdStSingle {
        load: bool,
        replicate: bool,
        q: bool,
        esize_log: u8,
        index: u8,
        selem: u8,
        rt: VReg,
        rn: Reg,
        post: Option<Reg>,
    },
}

/// `VFPExpandImm` for single (`ftype == 0`), double (`1`) or half (`3`) precision.
pub(crate) fn vfp_expand_imm(imm8: u32, ftype: u8) -> u64 {
    let a = u64::from((imm8 >> 7) & 1);
    let b = u64::from((imm8 >> 6) & 1);
    let cd = u64::from((imm8 >> 4) & 3);
    let efgh = u64::from(imm8 & 0xf);
    match ftype {
        1 => {
            let exp = ((b ^ 1) << 10) | (if b == 1 { 0xff } else { 0 } << 2) | cd;
            (a << 63) | (exp << 52) | (efgh << 48)
        }
        3 => {
            let exp = ((b ^ 1) << 4) | (if b == 1 { 3 } else { 0 } << 2) | cd;
            (a << 15) | (exp << 10) | (efgh << 6)
        }
        _ => {
            let exp = ((b ^ 1) << 7) | (if b == 1 { 0x1f } else { 0 } << 2) | cd;
            (a << 31) | (exp << 23) | (efgh << 19)
        }
    }
}

/// `AdvSIMDExpandImm` plus the operation it selects; `None` for unallocated encodings.
fn mod_imm(op: bool, cmode: u32, imm8: u32, q: bool) -> Option<(ModImmOp, u64)> {
    let i = u64::from(imm8);
    let rep32 = |x: u64| x | (x << 32);
    let rep16 = |x: u64| x | (x << 16) | (x << 32) | (x << 48);
    let (kind_or, imm) = match cmode >> 1 {
        0..=3 => (cmode & 1 == 1, rep32(i << (8 * (cmode >> 1)))),
        4 | 5 => (cmode & 1 == 1, rep16(i << (8 * ((cmode >> 1) & 1)))),
        6 => (
            false,
            rep32(if cmode & 1 == 0 {
                (i << 8) | 0xff
            } else {
                (i << 16) | 0xffff
            }),
        ),
        _ => {
            let imm = match (cmode & 1, op) {
                (0, false) => i * 0x0101_0101_0101_0101,
                (0, true) => (0..8).fold(0u64, |acc, b| {
                    acc | (if (imm8 >> b) & 1 == 1 { 0xff } else { 0 } << (8 * b))
                }),
                (_, false) => rep32(vfp_expand_imm(imm8, 0)),
                (_, true) => {
                    if !q {
                        return None;
                    }
                    vfp_expand_imm(imm8, 1)
                }
            };
            return Some((ModImmOp::Movi, imm));
        }
    };
    let opk = match (op, kind_or) {
        (false, false) => ModImmOp::Movi,
        (true, false) => ModImmOp::Mvni,
        (false, true) => ModImmOp::Orr,
        (true, true) => ModImmOp::Bic,
    };
    Some((opk, imm))
}

/// Decodes the "Data processing — SIMD and floating point" group (op0 = x111).
pub(crate) fn decode(i: u32) -> Inst {
    let s = |inst: SimdInst| Inst::Simd(inst);
    let (rd, rn, rm) = (
        bits(i, 4, 0) as u8,
        bits(i, 9, 5) as u8,
        bits(i, 20, 16) as u8,
    );
    let q = bit(i, 30);
    let u = bit(i, 29);
    let size = bits(i, 23, 22) as u8;

    // ---- scalar floating point and FP<->integer conversions: bits 28:24 = 11110 / 11111, bit 30 = 0 ----
    if bits(i, 28, 24) == 0b11111 && !bit(i, 30) {
        if bit(i, 31) || bit(i, 29) {
            return Inst::Undefined;
        }
        return s(SimdInst::Fp3 {
            ftype: size,
            o1: bit(i, 21),
            o0: bit(i, 15),
            rd,
            rn,
            rm,
            ra: bits(i, 14, 10) as u8,
        });
    }
    if bits(i, 28, 24) == 0b11110 && !bit(i, 30) {
        let ftype = size;
        if bit(i, 29) {
            return Inst::Undefined;
        }
        if !bit(i, 21) {
            // FP <-> fixed point.
            let scale = bits(i, 15, 10);
            let sf = bit(i, 31);
            if !sf && scale < 32 {
                return Inst::Undefined;
            }
            return s(SimdInst::FpFixed {
                sf,
                ftype,
                rmode: bits(i, 20, 19) as u8,
                opcode: bits(i, 18, 16) as u8,
                fbits: (64 - scale) as u8,
                rd,
                rn,
            });
        }
        if bits(i, 15, 10) == 0 {
            return s(SimdInst::FpInt {
                sf: bit(i, 31),
                ftype,
                rmode: bits(i, 20, 19) as u8,
                opcode: bits(i, 18, 16) as u8,
                rd,
                rn,
            });
        }
        if bit(i, 31) {
            return Inst::Undefined;
        }
        if bits(i, 14, 10) == 0b10000 {
            return s(SimdInst::Fp1 {
                ftype,
                opcode: bits(i, 20, 15) as u8,
                rd,
                rn,
            });
        }
        if bits(i, 13, 10) == 0b1000 {
            if bits(i, 15, 14) != 0 || bits(i, 2, 0) != 0 {
                return Inst::Undefined;
            }
            return s(SimdInst::FpCmp {
                ftype,
                zero: bit(i, 3),
                rn,
                rm,
            });
        }
        if bits(i, 12, 10) == 0b100 {
            if bits(i, 9, 5) != 0 {
                return Inst::Undefined;
            }
            return s(SimdInst::FpImm {
                ftype,
                bits: vfp_expand_imm(bits(i, 20, 13), ftype),
                rd,
            });
        }
        return match bits(i, 11, 10) {
            0b01 => s(SimdInst::FpCondCmp {
                ftype,
                rn,
                rm,
                cond: Cond(bits(i, 15, 12) as u8),
                nzcv: bits(i, 3, 0) as u8,
            }),
            0b10 => s(SimdInst::Fp2 {
                ftype,
                opcode: bits(i, 15, 12) as u8,
                rd,
                rn,
                rm,
            }),
            _ => s(SimdInst::FpCsel {
                ftype,
                rd,
                rn,
                rm,
                cond: Cond(bits(i, 15, 12) as u8),
            }),
        };
    }

    // ---- Advanced SIMD: bit 31 = 0 ----
    if bit(i, 31) {
        return Inst::Undefined;
    }
    let scalar = bit(i, 28);
    match bits(i, 28, 24) {
        0b01110 | 0b11110 => {
            if bit(i, 21) {
                if bit(i, 10) {
                    return s(SimdInst::ThreeSame {
                        scalar,
                        q,
                        u,
                        size,
                        opcode: bits(i, 15, 11) as u8,
                        rd,
                        rn,
                        rm,
                    });
                }
                if bits(i, 11, 10) == 0b00 && !scalar {
                    return s(SimdInst::ThreeDiff {
                        q,
                        u,
                        size,
                        opcode: bits(i, 15, 12) as u8,
                        rd,
                        rn,
                        rm,
                    });
                }
                if bits(i, 11, 10) == 0b10 {
                    let opcode = bits(i, 16, 12) as u8;
                    return match bits(i, 20, 17) {
                        0b0000 => s(SimdInst::TwoMisc {
                            scalar,
                            q,
                            u,
                            size,
                            opcode,
                            rd,
                            rn,
                        }),
                        0b1000 if scalar => s(SimdInst::ScalarPairwise {
                            u,
                            size,
                            opcode,
                            rd,
                            rn,
                        }),
                        0b1000 => s(SimdInst::AcrossLanes {
                            q,
                            u,
                            size,
                            opcode,
                            rd,
                            rn,
                        }),
                        _ => Inst::Undefined,
                    };
                }
                return Inst::Undefined;
            }
            // bit 21 = 0: copy (bits 23:21 = 000, bit 15 = 0, bit 10 = 1) and, for vectors, permute/table.
            if bits(i, 23, 21) == 0 && !bit(i, 15) && bit(i, 10) {
                let imm5 = bits(i, 20, 16) as u8;
                if imm5 & 0xf == 0 {
                    return Inst::Undefined;
                }
                if scalar {
                    if u || bits(i, 14, 11) != 0 {
                        return Inst::Undefined;
                    }
                    return s(SimdInst::ScalarDup { imm5, rd, rn });
                }
                return s(SimdInst::Copy {
                    q,
                    op: u,
                    imm5,
                    imm4: bits(i, 14, 11) as u8,
                    rd,
                    rn,
                });
            }
            if !scalar && u && size == 0 && !bit(i, 15) && !bit(i, 10) {
                // EXT: 0 Q 101110 00 0 Rm 0 imm4 0 Rn Rd
                let index = bits(i, 14, 11) as u8;
                if !q && index >= 8 {
                    return Inst::Undefined;
                }
                return s(SimdInst::Ext {
                    q,
                    index,
                    rd,
                    rn,
                    rm,
                });
            }
            if scalar || u {
                return Inst::Undefined;
            }
            if !bit(i, 15) && bits(i, 11, 10) == 0b10 {
                return s(SimdInst::Permute {
                    q,
                    size,
                    opcode: bits(i, 14, 12) as u8,
                    rd,
                    rn,
                    rm,
                });
            }
            if !bit(i, 15) && bits(i, 11, 10) == 0b00 && size == 0 {
                return s(SimdInst::Tbl {
                    q,
                    len: bits(i, 14, 13) as u8,
                    tbx: bit(i, 12),
                    rd,
                    rn,
                    rm,
                });
            }
            Inst::Undefined
        }
        0b01111 | 0b11111 => {
            if !bit(i, 10) {
                // Vector x indexed element.
                return s(SimdInst::Indexed {
                    scalar,
                    q,
                    u,
                    size,
                    opcode: bits(i, 15, 12) as u8,
                    l: bit(i, 21),
                    m: bit(i, 20),
                    h: bit(i, 11),
                    rd,
                    rn,
                    rm: bits(i, 19, 16) as u8,
                });
            }
            let immh = bits(i, 22, 19) as u8;
            if immh == 0 {
                if scalar || bit(i, 23) || bit(i, 11) {
                    return Inst::Undefined;
                }
                let imm8 = (bits(i, 18, 16) << 5) | bits(i, 9, 5);
                return match mod_imm(u, bits(i, 15, 12), imm8, q) {
                    Some((op, imm)) => s(SimdInst::ModImm { q, op, imm, rd }),
                    None => Inst::Undefined,
                };
            }
            if bit(i, 23) {
                return Inst::Undefined;
            }
            s(SimdInst::ShiftImm {
                scalar,
                q,
                u,
                immh,
                immb: bits(i, 18, 16) as u8,
                opcode: bits(i, 15, 11) as u8,
                rd,
                rn,
            })
        }
        _ => Inst::Undefined,
    }
}

/// Decodes "Advanced SIMD load/store multiple structures" and "... single structure" (and
/// their post-indexed forms).
pub(crate) fn decode_ldst_struct(i: u32) -> Inst {
    if bit(i, 31) {
        return Inst::Undefined;
    }
    let q = bit(i, 30);
    let load = bit(i, 22);
    let rt = bits(i, 4, 0) as u8;
    let rn = rsp(i, 5);
    let size = bits(i, 11, 10) as u8;
    let post_bit = bit(i, 23);
    let post = |rm: u32| {
        if rm == 31 {
            Some(crate::inst::ZR)
        } else {
            Some(rm as u8)
        }
    };
    match bits(i, 29, 24) {
        0b001100 => {
            // Multiple structures: bit 21 must be 0; without post-index Rm must be 0.
            if bit(i, 21) || (!post_bit && bits(i, 20, 16) != 0) {
                return Inst::Undefined;
            }
            let (regs, selem) = match bits(i, 15, 12) {
                0b0000 => (4, 4),
                0b0010 => (4, 1),
                0b0100 => (3, 3),
                0b0110 => (3, 1),
                0b0111 => (1, 1),
                0b1000 => (2, 2),
                0b1010 => (2, 1),
                _ => return Inst::Undefined,
            };
            if size == 3 && !q && selem > 1 {
                return Inst::Undefined;
            }
            Inst::Simd(SimdInst::LdStMulti {
                load,
                q,
                size,
                regs,
                selem,
                rt,
                rn,
                post: if post_bit {
                    post(bits(i, 20, 16))
                } else {
                    None
                },
            })
        }
        0b001101 => {
            if !post_bit && bits(i, 20, 16) != 0 {
                return Inst::Undefined;
            }
            let r = bit(i, 21);
            let opcode = bits(i, 15, 13);
            let s_bit = bit(i, 12);
            let selem = ((((opcode & 1) << 1) | u32::from(r)) + 1) as u8;
            let qs = (u32::from(q) << 3) | (u32::from(s_bit) << 2) | u32::from(size);
            let (esize_log, index, replicate) = match opcode >> 1 {
                0 => (0, qs as u8, false),
                1 => {
                    if size & 1 != 0 {
                        return Inst::Undefined;
                    }
                    (1, (qs >> 1) as u8, false)
                }
                2 => match size {
                    0 => (2, (qs >> 2) as u8, false),
                    1 if !s_bit => (3, u8::from(q), false),
                    _ => return Inst::Undefined,
                },
                _ => {
                    if !load || s_bit {
                        return Inst::Undefined;
                    }
                    (size, 0, true)
                }
            };
            Inst::Simd(SimdInst::LdStSingle {
                load,
                replicate,
                q,
                esize_log,
                index,
                selem,
                rt,
                rn,
                post: if post_bit {
                    post(bits(i, 20, 16))
                } else {
                    None
                },
            })
        }
        _ => Inst::Undefined,
    }
}
