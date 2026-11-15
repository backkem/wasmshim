//! The A64 decoder: one 32-bit word in, one [`Inst`] out.
//!
//! The structure follows the encoding-group tables of the Arm ARM (DDI 0487, section C4.1):
//! the top-level `op0` field (bits 28:25) picks a group, and each group has its own small
//! decoder function below. Anything unallocated or not implemented decodes to
//! [`Inst::Undefined`], which the interpreter reports as an undefined-instruction fault.

use crate::inst::*;
use crate::simd;

/// Extracts `bits[hi:lo]` of `x`.
#[inline]
pub(crate) const fn bits(x: u32, hi: u32, lo: u32) -> u32 {
    (x >> lo) & ((1 << (hi - lo + 1)) - 1)
}

/// Bit `n` of `x`.
#[inline]
pub(crate) const fn bit(x: u32, n: u32) -> bool {
    (x >> n) & 1 != 0
}

/// Sign-extends the low `width` bits of `x`.
#[inline]
pub(crate) const fn sext(x: u32, width: u32) -> i64 {
    let shift = 64 - width;
    ((x as i64) << shift) >> shift
}

/// Register field at `lo` where encoding 31 means the zero register.
#[inline]
pub(crate) const fn r(x: u32, lo: u32) -> Reg {
    bits(x, lo + 4, lo) as Reg
}

/// Register field at `lo` where encoding 31 means the stack pointer.
#[inline]
pub(crate) const fn rsp(x: u32, lo: u32) -> Reg {
    let n = bits(x, lo + 4, lo) as Reg;
    if n == 31 { SP } else { n }
}

/// Decodes one instruction word.
pub fn decode(insn: u32) -> Inst {
    match bits(insn, 28, 25) {
        0b1000 | 0b1001 => dp_immediate(insn),
        0b1010 | 0b1011 => branch_system(insn),
        0b0100 | 0b0110 | 0b1100 | 0b1110 => load_store(insn),
        0b0101 | 0b1101 => dp_register(insn),
        0b0111 | 0b1111 => simd::decode(insn),
        _ => Inst::Undefined,
    }
}

// ---------------------------------------------------------------------------------------------
// Data processing — immediate
// ---------------------------------------------------------------------------------------------

fn dp_immediate(i: u32) -> Inst {
    let sf = bit(i, 31);
    match bits(i, 25, 23) {
        0b000 | 0b001 => {
            let imm = sext((bits(i, 23, 5) << 2) | bits(i, 30, 29), 21);
            let page = bit(i, 31);
            Inst::Adr {
                rd: r(i, 0),
                imm: if page { imm << 12 } else { imm },
                page,
            }
        }
        0b010 => {
            let imm12 = u64::from(bits(i, 21, 10));
            let set_flags = bit(i, 29);
            Inst::AddSubImm {
                sf,
                sub: bit(i, 30),
                set_flags,
                // ADDS/SUBS write ZR (CMP/CMN); ADD/SUB write SP (MOV to/from SP).
                rd: if set_flags { r(i, 0) } else { rsp(i, 0) },
                rn: rsp(i, 5),
                imm: if bit(i, 22) { imm12 << 12 } else { imm12 },
            }
        }
        0b100 => {
            let n = bit(i, 22);
            if !sf && n {
                return Inst::Undefined;
            }
            let Some((imm, _)) = decode_bit_masks(
                n,
                bits(i, 15, 10),
                bits(i, 21, 16),
                true,
                if sf { 64 } else { 32 },
            ) else {
                return Inst::Undefined;
            };
            let op = logic_op(bits(i, 30, 29));
            Inst::LogicImm {
                sf,
                op,
                rd: if op == LogicOp::Ands {
                    r(i, 0)
                } else {
                    rsp(i, 0)
                },
                rn: r(i, 5),
                imm,
            }
        }
        0b101 => {
            let hw = bits(i, 22, 21);
            if !sf && hw >= 2 {
                return Inst::Undefined;
            }
            let op = match bits(i, 30, 29) {
                0b00 => MovOp::N,
                0b10 => MovOp::Z,
                0b11 => MovOp::K,
                _ => return Inst::Undefined,
            };
            Inst::MovWide {
                sf,
                op,
                rd: r(i, 0),
                imm16: bits(i, 20, 5) as u16,
                shift: (hw * 16) as u8,
            }
        }
        0b110 => {
            let n = bit(i, 22);
            let immr = bits(i, 21, 16);
            let imms = bits(i, 15, 10);
            if n != sf || (!sf && (immr >= 32 || imms >= 32)) {
                return Inst::Undefined;
            }
            let op = match bits(i, 30, 29) {
                0b00 => BitfieldOp::Sbfm,
                0b01 => BitfieldOp::Bfm,
                0b10 => BitfieldOp::Ubfm,
                _ => return Inst::Undefined,
            };
            let Some((wmask, tmask)) =
                decode_bit_masks(n, imms, immr, false, if sf { 64 } else { 32 })
            else {
                return Inst::Undefined;
            };
            Inst::Bitfield {
                sf,
                op,
                rd: r(i, 0),
                rn: r(i, 5),
                immr: immr as u8,
                imms: imms as u8,
                wmask,
                tmask,
            }
        }
        0b111 => {
            let lsb = bits(i, 15, 10);
            if bits(i, 30, 29) != 0 || bit(i, 21) || bit(i, 22) != sf || (!sf && lsb >= 32) {
                return Inst::Undefined;
            }
            Inst::Extract {
                sf,
                rd: r(i, 0),
                rn: r(i, 5),
                rm: r(i, 16),
                lsb: lsb as u8,
            }
        }
        _ => Inst::Undefined,
    }
}

const fn logic_op(opc: u32) -> LogicOp {
    match opc {
        0b00 => LogicOp::And,
        0b01 => LogicOp::Orr,
        0b10 => LogicOp::Eor,
        _ => LogicOp::Ands,
    }
}

/// `DecodeBitMasks` from the Arm ARM (shared pseudocode). Returns `(wmask, tmask)`, or `None`
/// for a reserved encoding.
pub(crate) fn decode_bit_masks(
    n: bool,
    imms: u32,
    immr: u32,
    immediate: bool,
    datasize: u32,
) -> Option<(u64, u64)> {
    let combined = (u32::from(n) << 6) | (!imms & 0x3f);
    if combined == 0 {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    if len < 1 || (1 << len) > datasize {
        return None;
    }
    let levels = (1u32 << len) - 1;
    if immediate && (imms & levels) == levels {
        return None;
    }
    let s = imms & levels;
    let r = immr & levels;
    let diff = s.wrapping_sub(r) & levels;
    let esize = 1u32 << len;
    let d = diff & levels;
    let welem = ones(s + 1);
    let telem = ones(d + 1);
    let wmask = replicate(ror(welem, r, esize), esize, datasize);
    let tmask = replicate(telem, esize, datasize);
    Some((wmask, tmask))
}

const fn ones(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

fn ror(x: u64, r: u32, width: u32) -> u64 {
    if r == 0 {
        return x;
    }
    let mask = ones(width);
    ((x >> r) | (x << (width - r))) & mask
}

fn replicate(x: u64, esize: u32, datasize: u32) -> u64 {
    let mut out = 0u64;
    let mut at = 0;
    while at < datasize {
        out |= x << at;
        at += esize;
    }
    out & ones(datasize)
}

// ---------------------------------------------------------------------------------------------
// Branches, exception generating and system instructions
// ---------------------------------------------------------------------------------------------

fn branch_system(i: u32) -> Inst {
    match bits(i, 31, 29) {
        0b000 | 0b100 => Inst::Branch {
            link: bit(i, 31),
            offset: sext(bits(i, 25, 0), 26) << 2,
        },
        0b001 | 0b101 => {
            let offset = sext(bits(i, 18, 5), 14) << 2;
            if bit(i, 25) {
                Inst::TestBranch {
                    nonzero: bit(i, 24),
                    rt: r(i, 0),
                    bit: ((bits(i, 31, 31) << 5) | bits(i, 23, 19)) as u8,
                    offset,
                }
            } else {
                Inst::CompareBranch {
                    sf: bit(i, 31),
                    nonzero: bit(i, 24),
                    rt: r(i, 0),
                    offset: sext(bits(i, 23, 5), 19) << 2,
                }
            }
        }
        0b010 => {
            // B.cond (bit 4 = 0) and BC.cond (bit 4 = 1, FEAT_HBC) behave the same here.
            if bit(i, 25) || bit(i, 24) {
                return Inst::Undefined;
            }
            Inst::BranchCond {
                cond: Cond(bits(i, 3, 0) as u8),
                offset: sext(bits(i, 23, 5), 19) << 2,
            }
        }
        0b110 => {
            if bit(i, 25) {
                branch_reg(i)
            } else if bits(i, 25, 24) == 0b00 {
                exception(i)
            } else if bits(i, 25, 22) == 0b0100 {
                system(i)
            } else {
                Inst::Undefined
            }
        }
        _ => Inst::Undefined,
    }
}

fn exception(i: u32) -> Inst {
    let imm = bits(i, 20, 5) as u16;
    match (bits(i, 23, 21), bits(i, 4, 2), bits(i, 1, 0)) {
        (0b000, 0b000, 0b01) => Inst::Svc { imm },
        (0b001, 0b000, 0b00) => Inst::Brk { imm },
        _ => Inst::Undefined,
    }
}

fn branch_reg(i: u32) -> Inst {
    // Only the plain forms: op2 = 11111, op3 = 000000, op4 = 00000.
    if bits(i, 20, 16) != 0b11111 || bits(i, 15, 10) != 0 || bits(i, 4, 0) != 0 {
        return Inst::Undefined;
    }
    let op = match bits(i, 24, 21) {
        0b0000 => BranchRegOp::Br,
        0b0001 => BranchRegOp::Blr,
        0b0010 => BranchRegOp::Ret,
        _ => return Inst::Undefined,
    };
    Inst::BranchReg { op, rn: r(i, 5) }
}

fn system(i: u32) -> Inst {
    let l = bit(i, 21);
    let op0 = bits(i, 20, 19);
    let op1 = bits(i, 18, 16);
    let crn = bits(i, 15, 12);
    let crm = bits(i, 11, 8);
    let op2 = bits(i, 7, 5);
    let rt = r(i, 0);
    match (l, op0) {
        (false, 0b00) => {
            if crn == 0b0010 && op1 == 0b011 && rt == 31 {
                // Hints: NOP, YIELD, WFE/WFI/SEV (no-ops for a single user thread), PAC and BTI
                // hint-space instructions (PAC is "not implemented", so they are NOPs too).
                Inst::Nop
            } else if crn == 0b0011 && op1 == 0b011 && rt == 31 {
                match op2 {
                    0b010 => Inst::Clrex,
                    0b100..=0b111 => Inst::Nop, // DSB, DMB, ISB, SB
                    _ => Inst::Undefined,
                }
            } else {
                Inst::Undefined // MSR (immediate): PSTATE fields, SMSTART/SMSTOP ...
            }
        }
        (false, 0b01) => {
            // SYS: only EL0-accessible cache maintenance by VA (op1 = 3, CRn = 7).
            if op1 == 0b011 && crn == 0b0111 {
                match (crm, op2) {
                    (0b0100, 0b001) => Inst::DcZva { rt },
                    // DC CVAC/CVAU/CVAP/CVADP/CIVAC, IC IVAU: coherent by construction.
                    (0b1010 | 0b1011 | 0b1100 | 0b1101 | 0b1110 | 0b0101, 0b001) => Inst::Nop,
                    _ => Inst::Undefined,
                }
            } else {
                Inst::Undefined
            }
        }
        (_, 0b10 | 0b11) => {
            let reg = sys_reg(op0, op1, crn, crm, op2);
            if l {
                Inst::Mrs { rt, reg }
            } else {
                Inst::Msr { rt, reg }
            }
        }
        _ => Inst::Undefined,
    }
}

fn sys_reg(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> SysReg {
    match (op0, op1, crn, crm, op2) {
        (3, 3, 4, 2, 0) => SysReg::Nzcv,
        (3, 3, 4, 4, 0) => SysReg::Fpcr,
        (3, 3, 4, 4, 1) => SysReg::Fpsr,
        (3, 3, 13, 0, 2) => SysReg::TpidrEl0,
        (3, 3, 13, 0, 3) => SysReg::TpidrroEl0,
        (3, 3, 14, 0, 2) => SysReg::CntvctEl0,
        (3, 3, 14, 0, 0) => SysReg::CntfrqEl0,
        (3, 3, 0, 0, 1) => SysReg::CtrEl0,
        (3, 3, 0, 0, 7) => SysReg::DczidEl0,
        (3, 0, 0, 0, 0) => SysReg::MidrEl1,
        _ => SysReg::Other(((op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2) as u16),
    }
}

// ---------------------------------------------------------------------------------------------
// Data processing — register
// ---------------------------------------------------------------------------------------------

const fn shift(x: u32) -> Shift {
    match x {
        0 => Shift::Lsl,
        1 => Shift::Lsr,
        2 => Shift::Asr,
        _ => Shift::Ror,
    }
}

pub(crate) const fn extend(x: u32) -> Extend {
    match x & 7 {
        0 => Extend::Uxtb,
        1 => Extend::Uxth,
        2 => Extend::Uxtw,
        3 => Extend::Uxtx,
        4 => Extend::Sxtb,
        5 => Extend::Sxth,
        6 => Extend::Sxtw,
        _ => Extend::Sxtx,
    }
}

fn dp_register(i: u32) -> Inst {
    let sf = bit(i, 31);
    let (rd, rn, rm) = (r(i, 0), r(i, 5), r(i, 16));
    if !bit(i, 28) {
        let amount = bits(i, 15, 10) as u8;
        let shifted_form = !bit(i, 24) || !bit(i, 21);
        if shifted_form && !sf && amount >= 32 {
            return Inst::Undefined;
        }
        if !bit(i, 24) {
            return Inst::LogicShifted {
                sf,
                op: logic_op(bits(i, 30, 29)),
                invert: bit(i, 21),
                rd,
                rn,
                rm,
                shift: shift(bits(i, 23, 22)),
                amount,
            };
        }
        let set_flags = bit(i, 29);
        if !bit(i, 21) {
            let sh = bits(i, 23, 22);
            if sh == 0b11 {
                return Inst::Undefined;
            }
            return Inst::AddSubShifted {
                sf,
                sub: bit(i, 30),
                set_flags,
                rd,
                rn,
                rm,
                shift: shift(sh),
                amount,
            };
        }
        let amount = bits(i, 12, 10) as u8;
        if bits(i, 23, 22) != 0 || amount > 4 {
            return Inst::Undefined;
        }
        return Inst::AddSubExtended {
            sf,
            sub: bit(i, 30),
            set_flags,
            rd: if set_flags { rd } else { rsp(i, 0) },
            rn: rsp(i, 5),
            rm,
            extend: extend(bits(i, 15, 13)),
            amount,
        };
    }
    match bits(i, 24, 21) {
        0b0000 => {
            if bits(i, 15, 10) != 0 {
                return Inst::Undefined;
            }
            Inst::AddSubCarry {
                sf,
                sub: bit(i, 30),
                set_flags: bit(i, 29),
                rd,
                rn,
                rm,
            }
        }
        0b0010 => {
            if !bit(i, 29) || bit(i, 10) || bit(i, 4) {
                return Inst::Undefined;
            }
            Inst::CondCompare {
                sf,
                sub: bit(i, 30),
                rn,
                rm,
                imm: if bit(i, 11) {
                    Some(bits(i, 20, 16) as u8)
                } else {
                    None
                },
                nzcv: bits(i, 3, 0) as u8,
                cond: Cond(bits(i, 15, 12) as u8),
            }
        }
        0b0100 => {
            if bit(i, 29) {
                return Inst::Undefined;
            }
            let op = match (bit(i, 30), bits(i, 11, 10)) {
                (false, 0b00) => CselOp::Csel,
                (false, 0b01) => CselOp::Csinc,
                (true, 0b00) => CselOp::Csinv,
                (true, 0b01) => CselOp::Csneg,
                _ => return Inst::Undefined,
            };
            Inst::CondSelect {
                sf,
                op,
                rd,
                rn,
                rm,
                cond: Cond(bits(i, 15, 12) as u8),
            }
        }
        0b0110 => {
            if bit(i, 29) {
                return Inst::Undefined;
            }
            if bit(i, 30) {
                if bits(i, 20, 16) != 0 {
                    return Inst::Undefined;
                }
                let op = match (bits(i, 15, 10), sf) {
                    (0b000000, _) => Dp1Op::Rbit,
                    (0b000001, _) => Dp1Op::Rev16,
                    (0b000010, false) => Dp1Op::Rev,
                    (0b000010, true) => Dp1Op::Rev32,
                    (0b000011, true) => Dp1Op::Rev,
                    (0b000100, _) => Dp1Op::Clz,
                    (0b000101, _) => Dp1Op::Cls,
                    _ => return Inst::Undefined,
                };
                Inst::Dp1 { sf, op, rd, rn }
            } else {
                let op = match bits(i, 15, 10) {
                    0b000010 => Dp2Op::Udiv,
                    0b000011 => Dp2Op::Sdiv,
                    0b001000 => Dp2Op::Lslv,
                    0b001001 => Dp2Op::Lsrv,
                    0b001010 => Dp2Op::Asrv,
                    0b001011 => Dp2Op::Rorv,
                    _ => return Inst::Undefined,
                };
                Inst::Dp2 { sf, op, rd, rn, rm }
            }
        }
        0b1000..=0b1111 => {
            if bits(i, 30, 29) != 0 {
                return Inst::Undefined;
            }
            let o0 = bit(i, 15);
            let op = match (bits(i, 23, 21), o0) {
                (0b000, false) => Dp3Op::Madd,
                (0b000, true) => Dp3Op::Msub,
                (0b001, false) if sf => Dp3Op::Smaddl,
                (0b001, true) if sf => Dp3Op::Smsubl,
                (0b010, false) if sf => Dp3Op::Smulh,
                (0b101, false) if sf => Dp3Op::Umaddl,
                (0b101, true) if sf => Dp3Op::Umsubl,
                (0b110, false) if sf => Dp3Op::Umulh,
                _ => return Inst::Undefined,
            };
            Inst::Dp3 {
                sf,
                op,
                rd,
                rn,
                rm,
                ra: r(i, 10),
            }
        }
        _ => Inst::Undefined,
    }
}

// ---------------------------------------------------------------------------------------------
// Loads and stores
// ---------------------------------------------------------------------------------------------

fn load_store(i: u32) -> Inst {
    let v = bit(i, 26);
    match bits(i, 29, 27) {
        0b001 if !v => {
            if bit(i, 24) {
                return Inst::Undefined; // RCpc / LDAPR (unscaled) etc.: rarely used
            }
            exclusive(i)
        }
        0b001 => simd::decode_ldst_struct(i),
        0b011 if bits(i, 25, 24) == 0 => load_literal(i, v),
        0b101 => load_store_pair(i, v),
        0b111 => load_store_reg(i, v),
        _ => Inst::Undefined,
    }
}

fn exclusive(i: u32) -> Inst {
    let size = bits(i, 31, 30);
    let (o2, l, o1) = (bit(i, 23), bit(i, 22), bit(i, 21));
    let (rs, rt, rt2, rn) = (r(i, 16), r(i, 0), r(i, 10), rsp(i, 5));
    let bytes = 1u8 << size;
    match (o2, o1) {
        (false, false) => Inst::Exclusive {
            op: if l {
                ExclusiveOp::LoadExclusive
            } else {
                ExclusiveOp::StoreExclusive
            },
            bytes,
            pair: false,
            rs,
            rt,
            rt2,
            rn,
        },
        (false, true) if size >= 2 => Inst::Exclusive {
            op: if l {
                ExclusiveOp::LoadExclusive
            } else {
                ExclusiveOp::StoreExclusive
            },
            bytes: 1 << (size & 1 | 2),
            pair: true,
            rs,
            rt,
            rt2,
            rn,
        },
        (false, true) => {
            if bits(i, 14, 10) != 0b11111 || rs % 2 != 0 || rt % 2 != 0 {
                return Inst::Undefined;
            }
            Inst::Casp {
                bytes: if size == 0 { 4 } else { 8 },
                rs,
                rt,
                rn,
            }
        }
        (true, false) => {
            if bits(i, 14, 10) != 0b11111 {
                return Inst::Undefined;
            }
            let op = if l {
                ExclusiveOp::LoadAcquire
            } else {
                ExclusiveOp::StoreRelease
            };
            Inst::Exclusive {
                op,
                bytes,
                pair: false,
                rs: ZR,
                rt,
                rt2: ZR,
                rn,
            }
        }
        (true, true) => {
            if bits(i, 14, 10) != 0b11111 {
                return Inst::Undefined;
            }
            Inst::Cas { bytes, rs, rt, rn }
        }
    }
}

fn load_literal(i: u32, v: bool) -> Inst {
    let offset = sext(bits(i, 23, 5), 19) << 2;
    let rt = r(i, 0);
    let (op, bytes) = match (bits(i, 31, 30), v) {
        (0b00, false) => (MemOp::Load, 4),
        (0b01, false) => (MemOp::Load, 8),
        (0b10, false) => (MemOp::LoadSigned64, 4),
        (0b11, false) => (MemOp::Prefetch, 8),
        (0b00, true) => (MemOp::LoadV, 4),
        (0b01, true) => (MemOp::LoadV, 8),
        (0b10, true) => (MemOp::LoadV, 16),
        _ => return Inst::Undefined,
    };
    Inst::LoadLiteral {
        op,
        bytes,
        rt,
        offset,
    }
}

fn load_store_pair(i: u32, v: bool) -> Inst {
    let opc = bits(i, 31, 30);
    let load = bit(i, 22);
    let (bytes, signed) = match (opc, v) {
        (0b00, false) => (4, false),
        (0b01, false) if load => (4, true), // LDPSW
        (0b10, false) => (8, false),
        (0b00, true) => (4, false),
        (0b01, true) => (8, false),
        (0b10, true) => (16, false),
        _ => return Inst::Undefined,
    };
    let imm = sext(bits(i, 21, 15), 7) * i64::from(bytes);
    let mode = match bits(i, 24, 23) {
        0b00 | 0b10 => AddrMode::Offset(imm), // 00 = non-temporal (LDNP/STNP)
        0b01 => AddrMode::PostIndex(imm),
        _ => AddrMode::PreIndex(imm),
    };
    Inst::LoadStorePair {
        load,
        vector: v,
        signed,
        bytes,
        rt: r(i, 0),
        rt2: r(i, 10),
        rn: rsp(i, 5),
        mode,
    }
}

fn load_store_reg(i: u32, v: bool) -> Inst {
    let size = bits(i, 31, 30);
    let opc = bits(i, 23, 22);
    let rt = r(i, 0);
    let rn = rsp(i, 5);

    if !bit(i, 24) && bit(i, 21) && bits(i, 11, 10) == 0b00 {
        return if v { Inst::Undefined } else { atomic(i) };
    }

    let (op, bytes) = if v {
        match (size, opc) {
            (_, 0b00) => (MemOp::StoreV, 1u8 << size),
            (_, 0b01) => (MemOp::LoadV, 1 << size),
            (0b00, 0b10) => (MemOp::StoreV, 16),
            (0b00, 0b11) => (MemOp::LoadV, 16),
            _ => return Inst::Undefined,
        }
    } else {
        let bytes = 1u8 << size;
        match (size, opc) {
            (_, 0b00) => (MemOp::Store, bytes),
            (_, 0b01) => (MemOp::Load, bytes),
            (0b11, 0b10) => (MemOp::Prefetch, bytes),
            (0b11, 0b11) => return Inst::Undefined,
            (_, 0b10) => (MemOp::LoadSigned64, bytes),
            (0b10, 0b11) => return Inst::Undefined,
            (_, _) => (MemOp::LoadSigned32, bytes),
        }
    };

    let mode = if bit(i, 24) {
        AddrMode::Offset(i64::from(bits(i, 21, 10)) << bytes.trailing_zeros())
    } else if bit(i, 21) {
        if bits(i, 11, 10) != 0b10 {
            return Inst::Undefined; // PAC loads (LDRAA/LDRAB)
        }
        let option = bits(i, 15, 13);
        if option & 0b010 == 0 {
            return Inst::Undefined;
        }
        let shift = if bit(i, 12) {
            bytes.trailing_zeros() as u8
        } else {
            0
        };
        AddrMode::Register {
            rm: r(i, 16),
            extend: extend(option),
            shift,
        }
    } else {
        let imm9 = sext(bits(i, 20, 12), 9);
        match bits(i, 11, 10) {
            0b00 | 0b10 => AddrMode::Offset(imm9), // unscaled (LDUR) and unprivileged (LDTR)
            0b01 => AddrMode::PostIndex(imm9),
            _ => AddrMode::PreIndex(imm9),
        }
    };
    if op == MemOp::Prefetch && !matches!(mode, AddrMode::Offset(_) | AddrMode::Register { .. }) {
        return Inst::Undefined;
    }
    Inst::LoadStore {
        op,
        bytes,
        rt,
        rn,
        mode,
    }
}

fn atomic(i: u32) -> Inst {
    let bytes = 1u8 << bits(i, 31, 30);
    let (rs, rt, rn) = (r(i, 16), r(i, 0), rsp(i, 5));
    let op = match (bit(i, 15), bits(i, 14, 12)) {
        (false, 0b000) => AtomicOp::Add,
        (false, 0b001) => AtomicOp::Clr,
        (false, 0b010) => AtomicOp::Eor,
        (false, 0b011) => AtomicOp::Set,
        (false, 0b100) => AtomicOp::Smax,
        (false, 0b101) => AtomicOp::Smin,
        (false, 0b110) => AtomicOp::Umax,
        (false, 0b111) => AtomicOp::Umin,
        (true, 0b000) => AtomicOp::Swp,
        // LDAPR (FEAT_LRCPC): a load-acquire for our purposes.
        (true, 0b100) if rs == ZR => {
            return Inst::Exclusive {
                op: ExclusiveOp::LoadAcquire,
                bytes,
                pair: false,
                rs: ZR,
                rt,
                rt2: ZR,
                rn,
            };
        }
        _ => return Inst::Undefined,
    };
    Inst::Atomic {
        op,
        bytes,
        rs,
        rt,
        rn,
    }
}
