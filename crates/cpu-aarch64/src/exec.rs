//! The interpreter: executes one decoded [`Inst`].
//!
//! Every instruction either completes (registers, flags, memory and PC updated) or faults
//! with no register or PC change, so a faulting instruction can be re-executed. Loads read
//! all their memory before writing any register; a store that faults on its second page may
//! have written the first (like hardware), which is harmless because re-execution rewrites it.

use wasmshim_cpu::{Event, Fault, GuestMemory, MemFault};

use crate::inst::*;
use crate::{COUNTER_FREQUENCY, Cpu, NO_MONITOR, syscall_frame};

/// `CTR_EL0`: 64-byte I/D cache lines, PIPT I-cache (a Cortex-A72-like value).
pub const CTR_EL0: u64 = 0x8444_c004;
/// `DCZID_EL0`: `DC ZVA` zeroes 2^4 words = 64 bytes.
pub const DCZID_EL0: u64 = 4;
pub const DC_ZVA_BYTES: u64 = 64;
/// `MIDR_EL1` as Linux reports it to EL0 (Arm Cortex-A72 r0p3).
pub const MIDR_EL1: u64 = 0x410f_d083;

type Step = Result<Option<Event>, Fault>;

#[inline]
const fn mask(sf: bool, value: u64) -> u64 {
    if sf { value } else { value as u32 as u64 }
}

#[inline]
fn shift_value(sf: bool, value: u64, shift: Shift, amount: u8) -> u64 {
    let amount = u32::from(amount);
    if sf {
        match shift {
            Shift::Lsl => value << amount,
            Shift::Lsr => value >> amount,
            Shift::Asr => ((value as i64) >> amount) as u64,
            Shift::Ror => value.rotate_right(amount),
        }
    } else {
        let w = value as u32;
        u64::from(match shift {
            Shift::Lsl => w << amount,
            Shift::Lsr => w >> amount,
            Shift::Asr => ((w as i32) >> amount) as u32,
            Shift::Ror => w.rotate_right(amount),
        })
    }
}

#[inline]
pub(crate) fn extend_value(value: u64, extend: Extend, shift: u8) -> u64 {
    let v = match extend {
        Extend::Uxtb => value as u8 as u64,
        Extend::Uxth => value as u16 as u64,
        Extend::Uxtw => value as u32 as u64,
        Extend::Uxtx | Extend::Sxtx => value,
        Extend::Sxtb => value as i8 as i64 as u64,
        Extend::Sxth => value as i16 as i64 as u64,
        Extend::Sxtw => value as i32 as i64 as u64,
    };
    v << shift
}

/// `AddWithCarry` from the Arm ARM: returns the result and NZCV in bits 3:0.
#[inline]
fn add_with_carry(sf: bool, x: u64, y: u64, carry: bool) -> (u64, u8) {
    if sf {
        let (r1, c1) = x.overflowing_add(y);
        let (r, c2) = r1.overflowing_add(u64::from(carry));
        let v = ((x ^ r) & (y ^ r)) >> 63 != 0;
        (r, flags(r >> 63 != 0, r == 0, c1 | c2, v))
    } else {
        let (x, y) = (x as u32, y as u32);
        let (r1, c1) = x.overflowing_add(y);
        let (r, c2) = r1.overflowing_add(u32::from(carry));
        let v = ((x ^ r) & (y ^ r)) >> 31 != 0;
        (u64::from(r), flags(r >> 31 != 0, r == 0, c1 | c2, v))
    }
}

#[inline]
const fn flags(n: bool, z: bool, c: bool, v: bool) -> u8 {
    (n as u8) << 3 | (z as u8) << 2 | (c as u8) << 1 | v as u8
}

/// Sign-extends the low `bytes` bytes of `value`.
#[inline]
const fn sign_extend(value: u64, bytes: u8) -> u64 {
    let shift = 64 - 8 * bytes as u32;
    (((value << shift) as i64) >> shift) as u64
}

impl Cpu {
    #[inline]
    fn get(&self, r: Reg) -> u64 {
        self.x[usize::from(r)]
    }

    #[inline]
    fn set(&mut self, r: Reg, value: u64) {
        self.x[usize::from(r)] = value;
        self.x[usize::from(ZR)] = 0;
    }

    #[inline]
    fn set_flags(&mut self, nzcv: u8) {
        self.n = nzcv & 8 != 0;
        self.z = nzcv & 4 != 0;
        self.c = nzcv & 2 != 0;
        self.v = nzcv & 1 != 0;
    }

    #[inline]
    pub(crate) fn cond_holds(&self, cond: Cond) -> bool {
        let result = match cond.0 >> 1 {
            0 => self.z,
            1 => self.c,
            2 => self.n,
            3 => self.v,
            4 => self.c && !self.z,
            5 => self.n == self.v,
            6 => self.n == self.v && !self.z,
            _ => true,
        };
        if cond.0 & 1 == 1 && cond.0 != 0b1111 {
            !result
        } else {
            result
        }
    }

    /// Computes the access address and the write-back value of an addressing mode.
    #[inline]
    pub(crate) fn address(
        &self,
        pc: u64,
        rn: Reg,
        mode: AddrMode,
    ) -> Result<(u64, Option<u64>), Fault> {
        let base = self.get(rn);
        if rn == SP && base & 0xf != 0 {
            return Err(Fault::Alignment { pc, addr: base });
        }
        Ok(match mode {
            AddrMode::Offset(imm) => (base.wrapping_add(imm as u64), None),
            AddrMode::PreIndex(imm) => {
                let a = base.wrapping_add(imm as u64);
                (a, Some(a))
            }
            AddrMode::PostIndex(imm) => (base, Some(base.wrapping_add(imm as u64))),
            AddrMode::Register { rm, extend, shift } => (
                base.wrapping_add(extend_value(self.get(rm), extend, shift)),
                None,
            ),
        })
    }

    // Inlined into both interpreter loops (`run_raw`, `run_block`): as a call it costs about a
    // third of the interpreter's speed.
    #[inline(always)]
    pub(crate) fn execute(&mut self, mem: &mut dyn GuestMemory, inst: Inst, pc: u64) -> Step {
        let mem_fault = |fault: MemFault| Fault::Memory { pc, fault };
        let mut next = pc.wrapping_add(4);
        match inst {
            Inst::Adr { rd, imm, page } => {
                let base = if page { pc & !0xfff } else { pc };
                self.set(rd, base.wrapping_add(imm as u64));
            }
            Inst::AddSubImm {
                sf,
                sub,
                set_flags,
                rd,
                rn,
                imm,
            } => {
                let y = if sub { !imm } else { imm };
                let (r, nzcv) = add_with_carry(sf, self.get(rn), y, sub);
                if set_flags {
                    self.set_flags(nzcv);
                }
                self.set(rd, mask(sf, r));
            }
            Inst::LogicImm {
                sf,
                op,
                rd,
                rn,
                imm,
            } => {
                let r = self.logic(sf, op, self.get(rn), imm);
                self.set(rd, r);
            }
            Inst::MovWide {
                sf,
                op,
                rd,
                imm16,
                shift,
            } => {
                let imm = u64::from(imm16) << shift;
                let r = match op {
                    MovOp::N => !imm,
                    MovOp::Z => imm,
                    MovOp::K => (self.get(rd) & !(0xffff << shift)) | imm,
                };
                self.set(rd, mask(sf, r));
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
            } => {
                let src = self.get(rn);
                let rotated = if sf {
                    src.rotate_right(u32::from(immr))
                } else {
                    u64::from((src as u32).rotate_right(u32::from(immr)))
                };
                let dst = if op == BitfieldOp::Bfm {
                    self.get(rd)
                } else {
                    0
                };
                let bot = (dst & !wmask) | (rotated & wmask);
                let top = match op {
                    BitfieldOp::Sbfm => {
                        if (src >> imms) & 1 != 0 {
                            mask(sf, u64::MAX)
                        } else {
                            0
                        }
                    }
                    _ => dst,
                };
                self.set(rd, mask(sf, (top & !tmask) | (bot & tmask)));
            }
            Inst::Extract {
                sf,
                rd,
                rn,
                rm,
                lsb,
            } => {
                let (hi, lo) = (self.get(rn), self.get(rm));
                let r = if sf {
                    if lsb == 0 {
                        lo
                    } else {
                        (lo >> lsb) | (hi << (64 - lsb))
                    }
                } else {
                    let both = (u64::from(hi as u32) << 32) | u64::from(lo as u32);
                    u64::from((both >> lsb) as u32)
                };
                self.set(rd, r);
            }

            Inst::Branch { link, offset } => {
                if link {
                    self.set(30, next);
                }
                next = pc.wrapping_add(offset as u64);
            }
            Inst::BranchCond { cond, offset } => {
                if self.cond_holds(cond) {
                    next = pc.wrapping_add(offset as u64);
                }
            }
            Inst::CompareBranch {
                sf,
                nonzero,
                rt,
                offset,
            } => {
                if (mask(sf, self.get(rt)) != 0) == nonzero {
                    next = pc.wrapping_add(offset as u64);
                }
            }
            Inst::TestBranch {
                nonzero,
                rt,
                bit,
                offset,
            } => {
                if ((self.get(rt) >> bit) & 1 != 0) == nonzero {
                    next = pc.wrapping_add(offset as u64);
                }
            }
            Inst::BranchReg { op, rn } => {
                let target = self.get(rn);
                if op == BranchRegOp::Blr {
                    self.set(30, next);
                }
                next = target;
            }
            Inst::Svc { .. } => {
                self.pc = next;
                return Ok(Some(Event::Syscall(syscall_frame(&self.x, next))));
            }
            Inst::Brk { imm } => return Err(Fault::Breakpoint { pc, imm }),
            Inst::Nop => {}
            Inst::Clrex => self.monitor = NO_MONITOR,
            Inst::DcZva { rt } => {
                let addr = self.get(rt) & !(DC_ZVA_BYTES - 1);
                mem.write(addr, &[0; DC_ZVA_BYTES as usize])
                    .map_err(mem_fault)?;
            }
            Inst::Mrs { rt, reg } => {
                let value = match reg {
                    SysReg::Nzcv => self.nzcv(),
                    SysReg::Fpcr => self.fpcr,
                    SysReg::Fpsr => self.fpsr,
                    SysReg::TpidrEl0 => self.tpidr,
                    SysReg::TpidrroEl0 => self.tpidrro,
                    SysReg::CntvctEl0 => self.counter,
                    SysReg::CntfrqEl0 => COUNTER_FREQUENCY,
                    SysReg::CtrEl0 => CTR_EL0,
                    SysReg::DczidEl0 => DCZID_EL0,
                    SysReg::MidrEl1 => MIDR_EL1,
                    SysReg::Other(_) => return Err(self.undefined(mem, pc)),
                };
                self.set(rt, value);
            }
            Inst::Msr { rt, reg } => {
                let value = self.get(rt);
                match reg {
                    SysReg::Nzcv => self.set_nzcv(value),
                    SysReg::Fpcr => self.fpcr = value & 0x07ff_9f00,
                    SysReg::Fpsr => self.fpsr = value & 0xf800_009f,
                    SysReg::TpidrEl0 => self.tpidr = value,
                    _ => return Err(self.undefined(mem, pc)),
                }
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
                let y = shift_value(sf, self.get(rm), shift, amount);
                self.add_sub(sf, sub, set_flags, rd, self.get(rn), y);
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
                let y = extend_value(self.get(rm), extend, amount);
                self.add_sub(sf, sub, set_flags, rd, self.get(rn), y);
            }
            Inst::AddSubCarry {
                sf,
                sub,
                set_flags,
                rd,
                rn,
                rm,
            } => {
                let y = if sub { !self.get(rm) } else { self.get(rm) };
                let (r, nzcv) = add_with_carry(sf, self.get(rn), y, self.c);
                if set_flags {
                    self.set_flags(nzcv);
                }
                self.set(rd, mask(sf, r));
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
                let y = shift_value(sf, self.get(rm), shift, amount);
                let y = if invert { !y } else { y };
                let r = self.logic(sf, op, self.get(rn), y);
                self.set(rd, r);
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
                if self.cond_holds(cond) {
                    let y = imm.map_or_else(|| self.get(rm), u64::from);
                    let y = if sub { !y } else { y };
                    let (_, f) = add_with_carry(sf, self.get(rn), y, sub);
                    self.set_flags(f);
                } else {
                    self.set_flags(nzcv);
                }
            }
            Inst::CondSelect {
                sf,
                op,
                rd,
                rn,
                rm,
                cond,
            } => {
                let r = if self.cond_holds(cond) {
                    self.get(rn)
                } else {
                    let m = self.get(rm);
                    match op {
                        CselOp::Csel => m,
                        CselOp::Csinc => m.wrapping_add(1),
                        CselOp::Csinv => !m,
                        CselOp::Csneg => m.wrapping_neg(),
                    }
                };
                self.set(rd, mask(sf, r));
            }
            Inst::Dp1 { sf, op, rd, rn } => {
                let x = self.get(rn);
                let r = if sf {
                    match op {
                        Dp1Op::Rbit => x.reverse_bits(),
                        Dp1Op::Rev16 => {
                            ((x & 0x00ff_00ff_00ff_00ff) << 8) | ((x >> 8) & 0x00ff_00ff_00ff_00ff)
                        }
                        Dp1Op::Rev32 => {
                            u64::from((x as u32).swap_bytes())
                                | (u64::from(((x >> 32) as u32).swap_bytes()) << 32)
                        }
                        Dp1Op::Rev => x.swap_bytes(),
                        Dp1Op::Clz => u64::from(x.leading_zeros()),
                        Dp1Op::Cls => u64::from(((x ^ (x << 1)) | 1).leading_zeros()),
                    }
                } else {
                    let w = x as u32;
                    u64::from(match op {
                        Dp1Op::Rbit => w.reverse_bits(),
                        Dp1Op::Rev16 => ((w & 0x00ff_00ff) << 8) | ((w >> 8) & 0x00ff_00ff),
                        Dp1Op::Rev32 | Dp1Op::Rev => w.swap_bytes(),
                        Dp1Op::Clz => w.leading_zeros(),
                        Dp1Op::Cls => ((w ^ (w << 1)) | 1).leading_zeros(),
                    })
                };
                self.set(rd, r);
            }
            Inst::Dp2 { sf, op, rd, rn, rm } => {
                let (x, y) = (self.get(rn), self.get(rm));
                let r = if sf {
                    match op {
                        Dp2Op::Udiv => x.checked_div(y).unwrap_or(0),
                        Dp2Op::Sdiv => {
                            if y == 0 {
                                0
                            } else {
                                (x as i64).wrapping_div(y as i64) as u64
                            }
                        }
                        Dp2Op::Lslv => x << (y & 63),
                        Dp2Op::Lsrv => x >> (y & 63),
                        Dp2Op::Asrv => ((x as i64) >> (y & 63)) as u64,
                        Dp2Op::Rorv => x.rotate_right((y & 63) as u32),
                    }
                } else {
                    let (x, y) = (x as u32, y as u32);
                    u64::from(match op {
                        Dp2Op::Udiv => x.checked_div(y).unwrap_or(0),
                        Dp2Op::Sdiv => {
                            if y == 0 {
                                0
                            } else {
                                (x as i32).wrapping_div(y as i32) as u32
                            }
                        }
                        Dp2Op::Lslv => x << (y & 31),
                        Dp2Op::Lsrv => x >> (y & 31),
                        Dp2Op::Asrv => ((x as i32) >> (y & 31)) as u32,
                        Dp2Op::Rorv => x.rotate_right(y & 31),
                    })
                };
                self.set(rd, r);
            }
            Inst::Dp3 {
                sf,
                op,
                rd,
                rn,
                rm,
                ra,
            } => {
                let (n, m, a) = (self.get(rn), self.get(rm), self.get(ra));
                let r = match op {
                    Dp3Op::Madd => a.wrapping_add(n.wrapping_mul(m)),
                    Dp3Op::Msub => a.wrapping_sub(n.wrapping_mul(m)),
                    Dp3Op::Smaddl => {
                        a.wrapping_add((n as i32 as i64).wrapping_mul(m as i32 as i64) as u64)
                    }
                    Dp3Op::Smsubl => {
                        a.wrapping_sub((n as i32 as i64).wrapping_mul(m as i32 as i64) as u64)
                    }
                    Dp3Op::Umaddl => {
                        a.wrapping_add((n as u32 as u64).wrapping_mul(m as u32 as u64))
                    }
                    Dp3Op::Umsubl => {
                        a.wrapping_sub((n as u32 as u64).wrapping_mul(m as u32 as u64))
                    }
                    Dp3Op::Smulh => ((i128::from(n as i64) * i128::from(m as i64)) >> 64) as u64,
                    Dp3Op::Umulh => ((u128::from(n) * u128::from(m)) >> 64) as u64,
                };
                self.set(rd, mask(sf, r));
            }

            Inst::LoadStore {
                op,
                bytes,
                rt,
                rn,
                mode,
            } => {
                let (addr, writeback) = self.address(pc, rn, mode)?;
                self.load_store(mem, pc, op, bytes, rt, addr)?;
                if let Some(wb) = writeback {
                    self.set(rn, wb);
                }
            }
            Inst::LoadLiteral {
                op,
                bytes,
                rt,
                offset,
            } => {
                self.load_store(mem, pc, op, bytes, rt, pc.wrapping_add(offset as u64))?;
            }
            Inst::LoadStorePair {
                load,
                vector,
                signed,
                bytes,
                rt,
                rt2,
                rn,
                mode,
            } => {
                let (addr, writeback) = self.address(pc, rn, mode)?;
                let n = usize::from(bytes);
                let second = addr.wrapping_add(u64::from(bytes));
                if vector {
                    if load {
                        let mut buf = [0u8; 32];
                        mem.read(addr, &mut buf[..2 * n]).map_err(mem_fault)?;
                        self.vreg[usize::from(rt)] = le_u128(&buf[..n]);
                        self.vreg[usize::from(rt2)] = le_u128(&buf[n..2 * n]);
                    } else {
                        let mut buf = [0u8; 32];
                        buf[..16].copy_from_slice(&self.vreg[usize::from(rt)].to_le_bytes());
                        buf[n..n + 16].copy_from_slice(&self.vreg[usize::from(rt2)].to_le_bytes());
                        mem.write(addr, &buf[..2 * n]).map_err(mem_fault)?;
                    }
                } else if load {
                    let (a, b) = if bytes == 8 {
                        (mem.read_u64(addr), mem.read_u64(second))
                    } else {
                        let a = mem.read_u32(addr).map(u64::from);
                        let b = mem.read_u32(second).map(u64::from);
                        (a, b)
                    };
                    let (mut a, mut b) = (a.map_err(mem_fault)?, b.map_err(mem_fault)?);
                    if signed {
                        a = sign_extend(a, 4);
                        b = sign_extend(b, 4);
                    }
                    self.set(rt, a);
                    self.set(rt2, b);
                } else {
                    let (a, b) = (self.get(rt), self.get(rt2));
                    if bytes == 8 {
                        let mut buf = [0u8; 16];
                        buf[..8].copy_from_slice(&a.to_le_bytes());
                        buf[8..].copy_from_slice(&b.to_le_bytes());
                        mem.write(addr, &buf).map_err(mem_fault)?;
                    } else {
                        let mut buf = [0u8; 8];
                        buf[..4].copy_from_slice(&(a as u32).to_le_bytes());
                        buf[4..].copy_from_slice(&(b as u32).to_le_bytes());
                        mem.write(addr, &buf).map_err(mem_fault)?;
                    }
                }
                if let Some(wb) = writeback {
                    self.set(rn, wb);
                }
            }
            Inst::Exclusive {
                op,
                bytes,
                pair,
                rs,
                rt,
                rt2,
                rn,
            } => {
                let addr = self.get(rn);
                let total = if pair {
                    2 * u64::from(bytes)
                } else {
                    u64::from(bytes)
                };
                if !addr.is_multiple_of(total) {
                    return Err(Fault::Alignment { pc, addr });
                }
                match op {
                    ExclusiveOp::LoadExclusive | ExclusiveOp::LoadAcquire => {
                        let a = read_sized(mem, addr, bytes).map_err(mem_fault)?;
                        let b = if pair {
                            read_sized(mem, addr + u64::from(bytes), bytes).map_err(mem_fault)?
                        } else {
                            0
                        };
                        if op == ExclusiveOp::LoadExclusive {
                            self.monitor = addr;
                        }
                        self.set(rt, a);
                        if pair {
                            self.set(rt2, b);
                        }
                    }
                    ExclusiveOp::StoreExclusive => {
                        // The monitor clears once the instruction completes: a store that faults
                        // leaves the reservation as it was (a precise fault), so the instruction
                        // can run again (a debugger's watch trap). Without a debugger
                        // the fault returns from `run_until_event`, which clears it anyway.
                        let ok = self.monitor == addr;
                        if ok {
                            write_sized(mem, addr, bytes, self.get(rt)).map_err(mem_fault)?;
                            if pair {
                                write_sized(mem, addr + u64::from(bytes), bytes, self.get(rt2))
                                    .map_err(mem_fault)?;
                            }
                        } else {
                            // A failing store-exclusive still faults on a bad address on hardware;
                            // check it so a broken loop cannot spin forever silently.
                            mem.check(addr, total, wasmshim_cpu::Access::Write)
                                .map_err(mem_fault)?;
                        }
                        self.monitor = NO_MONITOR;
                        self.set(rs, u64::from(!ok));
                    }
                    ExclusiveOp::StoreRelease => {
                        write_sized(mem, addr, bytes, self.get(rt)).map_err(mem_fault)?;
                    }
                }
            }
            Inst::Cas { bytes, rs, rt, rn } => {
                let addr = self.get(rn);
                if !addr.is_multiple_of(u64::from(bytes)) {
                    return Err(Fault::Alignment { pc, addr });
                }
                let old = read_sized(mem, addr, bytes).map_err(mem_fault)?;
                let size_mask = size_mask(bytes);
                if old == self.get(rs) & size_mask {
                    write_sized(mem, addr, bytes, self.get(rt)).map_err(mem_fault)?;
                } else {
                    mem.check(addr, u64::from(bytes), wasmshim_cpu::Access::Write)
                        .map_err(mem_fault)?;
                }
                self.set(rs, old);
            }
            Inst::Casp { bytes, rs, rt, rn } => {
                let addr = self.get(rn);
                let b = u64::from(bytes);
                if !addr.is_multiple_of(2 * b) {
                    return Err(Fault::Alignment { pc, addr });
                }
                let old0 = read_sized(mem, addr, bytes).map_err(mem_fault)?;
                let old1 = read_sized(mem, addr + b, bytes).map_err(mem_fault)?;
                let m = size_mask(bytes);
                if old0 == self.get(rs) & m && old1 == self.get(rs + 1) & m {
                    write_sized(mem, addr, bytes, self.get(rt)).map_err(mem_fault)?;
                    write_sized(mem, addr + b, bytes, self.get(rt + 1)).map_err(mem_fault)?;
                }
                self.set(rs, old0);
                self.set(rs + 1, old1);
            }
            Inst::Atomic {
                op,
                bytes,
                rs,
                rt,
                rn,
            } => {
                let addr = self.get(rn);
                if !addr.is_multiple_of(u64::from(bytes)) {
                    return Err(Fault::Alignment { pc, addr });
                }
                let old = read_sized(mem, addr, bytes).map_err(mem_fault)?;
                let m = size_mask(bytes);
                let s = self.get(rs) & m;
                let signed = |v: u64| sign_extend(v, bytes) as i64;
                let new = match op {
                    AtomicOp::Add => old.wrapping_add(s),
                    AtomicOp::Clr => old & !s,
                    AtomicOp::Eor => old ^ s,
                    AtomicOp::Set => old | s,
                    AtomicOp::Smax => {
                        if signed(old) >= signed(s) {
                            old
                        } else {
                            s
                        }
                    }
                    AtomicOp::Smin => {
                        if signed(old) <= signed(s) {
                            old
                        } else {
                            s
                        }
                    }
                    AtomicOp::Umax => old.max(s),
                    AtomicOp::Umin => old.min(s),
                    AtomicOp::Swp => s,
                };
                write_sized(mem, addr, bytes, new).map_err(mem_fault)?;
                self.set(rt, old);
            }

            Inst::Simd(s) => match self.execute_simd(mem, s, pc) {
                Ok(()) => {}
                // The SIMD&FP interpreter does not see the word: report it as integer ones do.
                Err(Fault::Undefined { pc, .. }) => return Err(self.undefined(mem, pc)),
                Err(f) => return Err(f),
            },
            Inst::Undefined => return Err(self.undefined(mem, pc)),
        }
        self.pc = next;
        Ok(None)
    }

    /// An undefined-instruction fault carrying the raw word (for diagnostics).
    fn undefined(&self, mem: &mut dyn GuestMemory, pc: u64) -> Fault {
        Fault::Undefined {
            pc,
            insn: mem.fetch(pc).unwrap_or(0),
        }
    }

    #[inline]
    fn add_sub(&mut self, sf: bool, sub: bool, set_flags: bool, rd: Reg, x: u64, y: u64) {
        let y = if sub { !y } else { y };
        let (r, nzcv) = add_with_carry(sf, x, y, sub);
        if set_flags {
            self.set_flags(nzcv);
        }
        self.set(rd, mask(sf, r));
    }

    #[inline]
    fn logic(&mut self, sf: bool, op: LogicOp, x: u64, y: u64) -> u64 {
        let r = mask(
            sf,
            match op {
                LogicOp::And | LogicOp::Ands => x & y,
                LogicOp::Orr => x | y,
                LogicOp::Eor => x ^ y,
            },
        );
        if op == LogicOp::Ands {
            let n = if sf { r >> 63 } else { r >> 31 } != 0;
            self.set_flags(flags(n, r == 0, false, false));
        }
        r
    }

    fn load_store(
        &mut self,
        mem: &mut dyn GuestMemory,
        pc: u64,
        op: MemOp,
        bytes: u8,
        rt: Reg,
        addr: u64,
    ) -> Result<(), Fault> {
        let mem_fault = |fault: MemFault| Fault::Memory { pc, fault };
        match op {
            MemOp::Store => write_sized(mem, addr, bytes, self.get(rt)).map_err(mem_fault)?,
            MemOp::Load => {
                let v = read_sized(mem, addr, bytes).map_err(mem_fault)?;
                self.set(rt, v);
            }
            MemOp::LoadSigned64 => {
                let v = read_sized(mem, addr, bytes).map_err(mem_fault)?;
                self.set(rt, sign_extend(v, bytes));
            }
            MemOp::LoadSigned32 => {
                let v = read_sized(mem, addr, bytes).map_err(mem_fault)?;
                self.set(rt, sign_extend(v, bytes) as u32 as u64);
            }
            MemOp::StoreV => {
                let n = usize::from(bytes);
                let v = self.vreg[usize::from(rt)].to_le_bytes();
                mem.write(addr, &v[..n]).map_err(mem_fault)?;
            }
            MemOp::LoadV => {
                let n = usize::from(bytes);
                let mut buf = [0u8; 16];
                mem.read(addr, &mut buf[..n]).map_err(mem_fault)?;
                self.vreg[usize::from(rt)] = u128::from_le_bytes(buf);
            }
            MemOp::Prefetch => {}
        }
        Ok(())
    }
}

#[inline]
const fn size_mask(bytes: u8) -> u64 {
    if bytes >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * bytes as u32)) - 1
    }
}

#[inline]
pub(crate) fn read_sized(mem: &dyn GuestMemory, addr: u64, bytes: u8) -> Result<u64, MemFault> {
    Ok(match bytes {
        1 => u64::from(mem.read_u8(addr)?),
        2 => u64::from(mem.read_u16(addr)?),
        4 => u64::from(mem.read_u32(addr)?),
        _ => mem.read_u64(addr)?,
    })
}

#[inline]
pub(crate) fn write_sized(
    mem: &mut dyn GuestMemory,
    addr: u64,
    bytes: u8,
    value: u64,
) -> Result<(), MemFault> {
    match bytes {
        1 => mem.write_u8(addr, value as u8),
        2 => mem.write_u16(addr, value as u16),
        4 => mem.write_u32(addr, value as u32),
        _ => mem.write_u64(addr, value),
    }
}

/// Little-endian bytes (at most 16) to a zero-extended `u128`.
#[inline]
pub(crate) fn le_u128(bytes: &[u8]) -> u128 {
    let mut buf = [0u8; 16];
    buf[..bytes.len()].copy_from_slice(bytes);
    u128::from_le_bytes(buf)
}
