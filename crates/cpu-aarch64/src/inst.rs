//! Decoded A64 instructions.
//!
//! The decoder resolves every encoding detail it can up front (register 31 as SP or ZR,
//! scaled and sign-extended immediates, logical-immediate bit masks), so the interpreter
//! only does arithmetic.

/// A general-purpose register operand: 0–30 are `X0`–`X30`, [`ZR`] reads as zero and
/// discards writes, [`SP`] is the stack pointer.
pub type Reg = u8;
/// The zero register (encoding 31 in most operand positions).
pub const ZR: Reg = 31;
/// The stack pointer (encoding 31 in base-register and some arithmetic positions).
pub const SP: Reg = 32;

/// A SIMD&FP register number, `V0`–`V31`.
pub type VReg = u8;

/// A condition code (`EQ`, `NE`, ... in encoding order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cond(pub u8);

/// Shift applied to a register operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shift {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

/// Extension applied to a register operand (`UXTB` ... `SXTX`), in encoding order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extend {
    Uxtb,
    Uxth,
    Uxtw,
    Uxtx,
    Sxtb,
    Sxth,
    Sxtw,
    Sxtx,
}

/// Logical operations (`opc` of the logical instruction classes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Orr,
    Eor,
    /// `AND` that also sets the flags.
    Ands,
}

/// Bitfield move flavours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitfieldOp {
    Sbfm,
    Bfm,
    Ubfm,
}

/// Wide-move flavours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MovOp {
    /// Move inverted.
    N,
    /// Move zero.
    Z,
    /// Move keep.
    K,
}

/// Conditional select flavours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CselOp {
    Csel,
    Csinc,
    Csinv,
    Csneg,
}

/// One-source data processing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dp1Op {
    Rbit,
    Rev16,
    /// `REV32` (64-bit) — byte-reverse each word.
    Rev32,
    /// Byte-reverse the whole register (`REV`, 32 or 64 bits).
    Rev,
    Clz,
    Cls,
}

/// Two-source data processing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dp2Op {
    Udiv,
    Sdiv,
    Lslv,
    Lsrv,
    Asrv,
    Rorv,
}

/// Three-source data processing (multiply-accumulate family).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dp3Op {
    Madd,
    Msub,
    Smaddl,
    Smsubl,
    Umaddl,
    Umsubl,
    Smulh,
    Umulh,
}

/// Branch-to-register flavours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchRegOp {
    Br,
    Blr,
    Ret,
}

/// System registers readable or writable at EL0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SysReg {
    Nzcv,
    Fpcr,
    Fpsr,
    TpidrEl0,
    TpidrroEl0,
    CntvctEl0,
    CntfrqEl0,
    CtrEl0,
    DczidEl0,
    MidrEl1,
    /// Any other `MRS`/`MSR`: undefined at EL0.
    Other(u16),
}

/// Memory-access shapes for single-register loads and stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemOp {
    /// Store `bytes` bytes of a general register.
    Store,
    /// Load and zero-extend.
    Load,
    /// Load and sign-extend to 64 bits.
    LoadSigned64,
    /// Load and sign-extend to 32 bits (upper half zeroed).
    LoadSigned32,
    /// Store a SIMD&FP register.
    StoreV,
    /// Load a SIMD&FP register (upper bits zeroed).
    LoadV,
    /// A prefetch hint: no architectural effect.
    Prefetch,
}

/// Address computation for single-register and pair loads/stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddrMode {
    /// `[base, #imm]`
    Offset(i64),
    /// `[base, #imm]!`
    PreIndex(i64),
    /// `[base], #imm`
    PostIndex(i64),
    /// `[base, Rm, extend #shift]`
    Register { rm: Reg, extend: Extend, shift: u8 },
}

/// Ordering flavour of exclusive / acquire-release accesses (only affects multi-core; kept for tracing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExclusiveOp {
    /// `LDXR`/`LDAXR` (and pair forms).
    LoadExclusive,
    /// `STXR`/`STLXR` (and pair forms); `rs` receives the status.
    StoreExclusive,
    /// `LDAR`/`LDLAR`/`LDAPR`.
    LoadAcquire,
    /// `STLR`/`STLLR`.
    StoreRelease,
}

/// LSE atomic read-modify-write operations (`LDADD` ... `SWP`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    Clr,
    Eor,
    Set,
    Smax,
    Smin,
    Umax,
    Umin,
    Swp,
}

/// A decoded A64 instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inst {
    // ---- data processing, immediate ----
    /// `ADR`/`ADRP`: `rd = (page ? pc & !0xfff : pc) + imm`.
    Adr { rd: Reg, imm: i64, page: bool },
    /// `ADD`/`SUB`(`S`) with an immediate (already shifted).
    AddSubImm {
        sf: bool,
        sub: bool,
        set_flags: bool,
        rd: Reg,
        rn: Reg,
        imm: u64,
    },
    /// `AND`/`ORR`/`EOR`/`ANDS` with a decoded bit-mask immediate.
    LogicImm {
        sf: bool,
        op: LogicOp,
        rd: Reg,
        rn: Reg,
        imm: u64,
    },
    /// `MOVN`/`MOVZ`/`MOVK`.
    MovWide {
        sf: bool,
        op: MovOp,
        rd: Reg,
        imm16: u16,
        shift: u8,
    },
    /// `SBFM`/`BFM`/`UBFM`, with the `DecodeBitMasks` results precomputed.
    Bitfield {
        sf: bool,
        op: BitfieldOp,
        rd: Reg,
        rn: Reg,
        immr: u8,
        imms: u8,
        wmask: u64,
        tmask: u64,
    },
    /// `EXTR`.
    Extract {
        sf: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        lsb: u8,
    },

    // ---- branches, exceptions, system ----
    /// `B`/`BL` to `pc + offset`.
    Branch { link: bool, offset: i64 },
    /// `B.cond`.
    BranchCond { cond: Cond, offset: i64 },
    /// `CBZ`/`CBNZ`.
    CompareBranch {
        sf: bool,
        nonzero: bool,
        rt: Reg,
        offset: i64,
    },
    /// `TBZ`/`TBNZ`.
    TestBranch {
        nonzero: bool,
        rt: Reg,
        bit: u8,
        offset: i64,
    },
    /// `BR`/`BLR`/`RET`.
    BranchReg { op: BranchRegOp, rn: Reg },
    /// `SVC #imm`.
    Svc { imm: u16 },
    /// `BRK #imm`.
    Brk { imm: u16 },
    /// Architectural no-ops at EL0 in this model: `NOP`, `YIELD`, PAC/BTI hints, `DMB`/`DSB`/`ISB`,
    /// cache maintenance by VA.
    Nop,
    /// `CLREX`.
    Clrex,
    /// `DC ZVA, Xt`: zero a 64-byte block.
    DcZva { rt: Reg },
    /// `MRS Xt, <sysreg>`.
    Mrs { rt: Reg, reg: SysReg },
    /// `MSR <sysreg>, Xt`.
    Msr { rt: Reg, reg: SysReg },

    // ---- data processing, register ----
    /// `ADD`/`SUB`(`S`) with a shifted register.
    AddSubShifted {
        sf: bool,
        sub: bool,
        set_flags: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift: Shift,
        amount: u8,
    },
    /// `ADD`/`SUB`(`S`) with an extended register.
    AddSubExtended {
        sf: bool,
        sub: bool,
        set_flags: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        extend: Extend,
        amount: u8,
    },
    /// `ADC`/`SBC`(`S`).
    AddSubCarry {
        sf: bool,
        sub: bool,
        set_flags: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    /// `AND`/`BIC`/`ORR`/`ORN`/`EOR`/`EON`/`ANDS`/`BICS` with a shifted register.
    LogicShifted {
        sf: bool,
        op: LogicOp,
        invert: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift: Shift,
        amount: u8,
    },
    /// `CCMP`/`CCMN` with a register (`imm == None`) or a 5-bit immediate.
    CondCompare {
        sf: bool,
        sub: bool,
        rn: Reg,
        rm: Reg,
        imm: Option<u8>,
        nzcv: u8,
        cond: Cond,
    },
    /// `CSEL`/`CSINC`/`CSINV`/`CSNEG`.
    CondSelect {
        sf: bool,
        op: CselOp,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        cond: Cond,
    },
    /// One-source ops.
    Dp1 {
        sf: bool,
        op: Dp1Op,
        rd: Reg,
        rn: Reg,
    },
    /// Two-source ops.
    Dp2 {
        sf: bool,
        op: Dp2Op,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    /// Three-source ops.
    Dp3 {
        sf: bool,
        op: Dp3Op,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        ra: Reg,
    },

    // ---- loads and stores ----
    /// Single-register load/store of `bytes` (1, 2, 4, 8 or 16).
    LoadStore {
        op: MemOp,
        bytes: u8,
        rt: Reg,
        rn: Reg,
        mode: AddrMode,
    },
    /// `LDR (literal)` and friends: load from `pc + offset`.
    LoadLiteral {
        op: MemOp,
        bytes: u8,
        rt: Reg,
        offset: i64,
    },
    /// `LDP`/`STP`/`LDPSW`/`LDNP`/`STNP` for general (`vector == false`) or SIMD&FP registers.
    LoadStorePair {
        load: bool,
        vector: bool,
        signed: bool,
        bytes: u8,
        rt: Reg,
        rt2: Reg,
        rn: Reg,
        mode: AddrMode,
    },
    /// Exclusive and acquire/release single and pair accesses (`pair` uses `rt2`).
    Exclusive {
        op: ExclusiveOp,
        bytes: u8,
        pair: bool,
        rs: Reg,
        rt: Reg,
        rt2: Reg,
        rn: Reg,
    },
    /// `CAS`/`CASA`/`CASL`/`CASAL` (and byte/half forms).
    Cas {
        bytes: u8,
        rs: Reg,
        rt: Reg,
        rn: Reg,
    },
    /// `CASP` family: compare and swap a pair (`rs`, `rs+1`) with (`rt`, `rt+1`).
    Casp {
        bytes: u8,
        rs: Reg,
        rt: Reg,
        rn: Reg,
    },
    /// LSE atomics `LD<op>` / `SWP`: `rt` receives the old value.
    Atomic {
        op: AtomicOp,
        bytes: u8,
        rs: Reg,
        rt: Reg,
        rn: Reg,
    },

    /// SIMD&FP instructions (see [`crate::simd`]).
    Simd(crate::simd::SimdInst),

    /// An unallocated or unimplemented encoding.
    Undefined,
}

impl Inst {
    /// `true` if executing this instruction may write guest memory (code caches re-check the
    /// memory's code epoch after these, so a store into code is seen by the next instruction).
    #[inline(always)]
    pub fn may_write_memory(&self) -> bool {
        use crate::simd::SimdInst;
        match self {
            Self::LoadStore { op, .. } => matches!(op, MemOp::Store | MemOp::StoreV),
            Self::LoadStorePair { load, .. } => !load,
            Self::Exclusive { op, .. } => {
                matches!(op, ExclusiveOp::StoreExclusive | ExclusiveOp::StoreRelease)
            }
            Self::Cas { .. } | Self::Casp { .. } | Self::Atomic { .. } | Self::DcZva { .. } => true,
            Self::Simd(SimdInst::LdStMulti { load, .. } | SimdInst::LdStSingle { load, .. }) => {
                !load
            }
            _ => false,
        }
    }
}

impl Inst {
    /// `true` for branches (taken or not): the instructions that end a basic block.
    #[inline(always)]
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            Self::Branch { .. }
                | Self::BranchCond { .. }
                | Self::CompareBranch { .. }
                | Self::TestBranch { .. }
                | Self::BranchReg { .. }
        )
    }
}
