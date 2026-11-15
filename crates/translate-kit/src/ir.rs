//! The translator's IR: **ISA-neutral**.
//!
//! A front half (`wasmshim-aarch64-translate`'s for AArch64) lowers each guest instruction of a block to
//! [`Op`]s over values [`V`]; passes ([`crate::opt`]) rewrite them; a back half
//! lowers them to Wasm. The IR knows these things and no more:
//!
//! * **pure arithmetic** on `i32`, `i64` and `v128` values ([`P`]: integer and vector
//!   operations with Wasm's semantics, a generic set that is not any guest's);
//! * **guest state slots** ([`Slot`]): numbered pieces of guest state with a home in the
//!   guest-state memory, which the back half may keep in locals ([`Op::Get`], [`Op::Set`]);
//!   and plain loads and stores of guest-state memory ([`Base::Cpu`]) for state the back half
//!   never caches (vector registers, system registers);
//! * **guest memory accesses with permissions** ([`Op::Mem`]): a fast path through the
//!   software TLB, given a host address, and a slow path in which the front half's reference
//!   interpreter executes the guest instruction;
//! * **the reference interpreter** for one guest instruction ([`Op::Step`]) and **helpers**
//!   the front half supplies as functions ([`Op::Helper`]);
//! * **exits and accounting**: a block retires its instructions; exits ([`Op::Exit`]) give
//!   back the budget of what did not run; blocks end in [`Term`]s.
//!
//! Values are defined once and used within their block (state crosses blocks only through
//! slots), so the back half maps them onto a pool of Wasm locals reset at every block.
//! A guest's semantics (its flags representation, addressing modes, exclusive monitors, system
//! registers) live in its front half: there is no guest-specific operation here.

use alloc::boxed::Box;
use alloc::vec::Vec;

pub mod build;

/// A value: an index into its block's value table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct V(pub u32);

/// The base address of guest-state memory (the CPU structure), a value every block has.
pub const CPU: V = V(u32::MAX);

/// A value type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    I32,
    I64,
    V128,
    /// IEEE single and double. Guest memory and guest state are
    /// integer-typed: a front half moves float bits in and out with the reinterpret operations.
    /// Wasm leaves NaN bit patterns open, so a front half must not let a NaN a float operation
    /// produced reach guest state (it checks and takes its reference path instead).
    F32,
    F64,
}

impl Ty {
    /// The number of value types (for per-type tables).
    pub const COUNT: usize = 5;

    /// A dense index (`0..COUNT`), for per-type tables such as a back half's free locals.
    pub fn index(self) -> usize {
        match self {
            Ty::I32 => 0,
            Ty::I64 => 1,
            Ty::V128 => 2,
            Ty::F32 => 3,
            Ty::F64 => 4,
        }
    }
}

/// A Wasm memory argument for memory 0 (`align`: log2 of the alignment hint).
pub fn memarg(offset: u64, align: u32) -> wasm_encoder::MemArg {
    wasm_encoder::MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

/// A guest state slot (the front half numbers them; at most 64).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Slot(pub u8);

/// A set of slots.
pub type Slots = u64;

/// The bit of `s` in a [`Slots`] set.
pub const fn slot_bit(s: Slot) -> Slots {
    1 << s.0
}

/// What the back half needs to know about a front half's slots.
#[derive(Clone, Debug)]
pub struct SlotInfo {
    /// Per slot: its offset in guest-state memory and its type; `None` for unused numbers.
    pub slots: Vec<Option<(u64, Ty)>>,
    /// Slots the back half must load and store together, through the front half's own code
    /// ([`Grouped`]): a guest's lazily evaluated condition flags, say.
    pub group: Slots,
}

/// The front half's code for a slot group whose memory form is not one word per slot.
pub trait Grouped {
    /// Loads the group into its locals (`local(slot)` gives a slot's local).
    fn load(&self, s: &mut wasm_encoder::InstructionSink<'_>, local: &dyn Fn(Slot) -> u32);
    /// Stores the group from its locals. `known` is the value of the group's first slot if it
    /// is a constant here (a representation tag, letting the store skip what that form does
    /// not use).
    fn store(
        &self,
        s: &mut wasm_encoder::InstructionSink<'_>,
        local: &dyn Fn(Slot) -> u32,
        known: Option<i64>,
    );
    /// Memory operations of a load and of a store (statistics).
    fn traffic(&self) -> (u32, u32);
}

/// Where a plain load or store goes: guest-state memory, or an address value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base {
    Cpu,
    Val(V),
}

macro_rules! pure_ops {
    ($($m:ident => $v:ident [$($a:ident),*] -> $r:ident),* $(,)?) => {
        /// Pure operations: Wasm's numeric and vector instructions without immediates.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum P { $($v),* }

        impl P {
            /// The operand types.
            pub fn args(self) -> &'static [Ty] {
                match self { $(P::$v => &[$(Ty::$a),*]),* }
            }
            /// The result type.
            pub fn ty(self) -> Ty {
                match self { $(P::$v => Ty::$r),* }
            }
            /// Emits the instruction.
            pub fn emit(self, s: &mut wasm_encoder::InstructionSink<'_>) {
                match self { $(P::$v => { s.$m(); }),* }
            }
        }

        /// The stack-style builder methods, one per pure operation.
        pub trait PureOps {
            fn pure(&mut self, p: P) -> &mut Self;
            $(fn $m(&mut self) -> &mut Self { self.pure(P::$v) })*
        }
    };
}

pure_ops! {
    i32_add => I32Add [I32, I32] -> I32,
    i32_sub => I32Sub [I32, I32] -> I32,
    i32_mul => I32Mul [I32, I32] -> I32,
    i32_and => I32And [I32, I32] -> I32,
    i32_or => I32Or [I32, I32] -> I32,
    i32_xor => I32Xor [I32, I32] -> I32,
    i32_shl => I32Shl [I32, I32] -> I32,
    i32_shr_u => I32ShrU [I32, I32] -> I32,
    i32_shr_s => I32ShrS [I32, I32] -> I32,
    i32_rotl => I32Rotl [I32, I32] -> I32,
    i32_rotr => I32Rotr [I32, I32] -> I32,
    i32_clz => I32Clz [I32] -> I32,
    i32_eqz => I32Eqz [I32] -> I32,
    i32_eq => I32Eq [I32, I32] -> I32,
    i32_ne => I32Ne [I32, I32] -> I32,
    i32_lt_s => I32LtS [I32, I32] -> I32,
    i32_lt_u => I32LtU [I32, I32] -> I32,
    i32_gt_s => I32GtS [I32, I32] -> I32,
    i32_gt_u => I32GtU [I32, I32] -> I32,
    i32_ge_s => I32GeS [I32, I32] -> I32,
    i32_ge_u => I32GeU [I32, I32] -> I32,
    i32_wrap_i64 => I32WrapI64 [I64] -> I32,
    i64_add => I64Add [I64, I64] -> I64,
    i64_sub => I64Sub [I64, I64] -> I64,
    i64_mul => I64Mul [I64, I64] -> I64,
    i64_and => I64And [I64, I64] -> I64,
    i64_or => I64Or [I64, I64] -> I64,
    i64_xor => I64Xor [I64, I64] -> I64,
    i64_shl => I64Shl [I64, I64] -> I64,
    i64_shr_u => I64ShrU [I64, I64] -> I64,
    i64_shr_s => I64ShrS [I64, I64] -> I64,
    i64_rotl => I64Rotl [I64, I64] -> I64,
    i64_rotr => I64Rotr [I64, I64] -> I64,
    i64_clz => I64Clz [I64] -> I64,
    i64_eqz => I64Eqz [I64] -> I32,
    i64_eq => I64Eq [I64, I64] -> I32,
    i64_ne => I64Ne [I64, I64] -> I32,
    i64_lt_s => I64LtS [I64, I64] -> I32,
    i64_lt_u => I64LtU [I64, I64] -> I32,
    i64_gt_s => I64GtS [I64, I64] -> I32,
    i64_gt_u => I64GtU [I64, I64] -> I32,
    i64_ge_s => I64GeS [I64, I64] -> I32,
    i64_ge_u => I64GeU [I64, I64] -> I32,
    i64_extend_i32_u => I64ExtendI32U [I32] -> I64,
    i64_extend_i32_s => I64ExtendI32S [I32] -> I64,
    i64_extend8_s => I64Extend8S [I64] -> I64,
    i64_extend16_s => I64Extend16S [I64] -> I64,
    i64_extend32_s => I64Extend32S [I64] -> I64,
    v128_and => V128And [V128, V128] -> V128,
    v128_andnot => V128AndNot [V128, V128] -> V128,
    v128_or => V128Or [V128, V128] -> V128,
    v128_xor => V128Xor [V128, V128] -> V128,
    v128_not => V128Not [V128] -> V128,
    v128_bitselect => V128Bitselect [V128, V128, V128] -> V128,
    i8x16_swizzle => I8x16Swizzle [V128, V128] -> V128,
    i8x16_popcnt => I8x16Popcnt [V128] -> V128,
    i8x16_splat => I8x16Splat [I32] -> V128,
    i16x8_splat => I16x8Splat [I32] -> V128,
    i32x4_splat => I32x4Splat [I32] -> V128,
    i64x2_splat => I64x2Splat [I64] -> V128,
    i8x16_add => I8x16Add [V128, V128] -> V128,
    i16x8_add => I16x8Add [V128, V128] -> V128,
    i32x4_add => I32x4Add [V128, V128] -> V128,
    i64x2_add => I64x2Add [V128, V128] -> V128,
    i8x16_eq => I8x16Eq [V128, V128] -> V128,
    i16x8_eq => I16x8Eq [V128, V128] -> V128,
    i32x4_eq => I32x4Eq [V128, V128] -> V128,
    i64x2_eq => I64x2Eq [V128, V128] -> V128,
    i8x16_ne => I8x16Ne [V128, V128] -> V128,
    i16x8_ne => I16x8Ne [V128, V128] -> V128,
    i32x4_ne => I32x4Ne [V128, V128] -> V128,
    i64x2_ne => I64x2Ne [V128, V128] -> V128,
    i8x16_gt_s => I8x16GtS [V128, V128] -> V128,
    i16x8_gt_s => I16x8GtS [V128, V128] -> V128,
    i32x4_gt_s => I32x4GtS [V128, V128] -> V128,
    i64x2_gt_s => I64x2GtS [V128, V128] -> V128,
    i8x16_ge_s => I8x16GeS [V128, V128] -> V128,
    i16x8_ge_s => I16x8GeS [V128, V128] -> V128,
    i32x4_ge_s => I32x4GeS [V128, V128] -> V128,
    i64x2_ge_s => I64x2GeS [V128, V128] -> V128,
    i8x16_gt_u => I8x16GtU [V128, V128] -> V128,
    i16x8_gt_u => I16x8GtU [V128, V128] -> V128,
    i32x4_gt_u => I32x4GtU [V128, V128] -> V128,
    i8x16_ge_u => I8x16GeU [V128, V128] -> V128,
    i16x8_ge_u => I16x8GeU [V128, V128] -> V128,
    i32x4_ge_u => I32x4GeU [V128, V128] -> V128,
    i8x16_max_s => I8x16MaxS [V128, V128] -> V128,
    i16x8_max_s => I16x8MaxS [V128, V128] -> V128,
    i32x4_max_s => I32x4MaxS [V128, V128] -> V128,
    i8x16_max_u => I8x16MaxU [V128, V128] -> V128,
    i16x8_max_u => I16x8MaxU [V128, V128] -> V128,
    i32x4_max_u => I32x4MaxU [V128, V128] -> V128,
    i8x16_min_s => I8x16MinS [V128, V128] -> V128,
    i16x8_min_s => I16x8MinS [V128, V128] -> V128,
    i32x4_min_s => I32x4MinS [V128, V128] -> V128,
    i8x16_min_u => I8x16MinU [V128, V128] -> V128,
    i16x8_min_u => I16x8MinU [V128, V128] -> V128,
    i32x4_min_u => I32x4MinU [V128, V128] -> V128,
    v128_any_true => V128AnyTrue [V128] -> I32,
    i8x16_sub => I8x16Sub [V128, V128] -> V128,
    i16x8_sub => I16x8Sub [V128, V128] -> V128,
    i32x4_sub => I32x4Sub [V128, V128] -> V128,
    i64x2_sub => I64x2Sub [V128, V128] -> V128,
    i16x8_mul => I16x8Mul [V128, V128] -> V128,
    i32x4_mul => I32x4Mul [V128, V128] -> V128,
    i64x2_mul => I64x2Mul [V128, V128] -> V128,
    i8x16_abs => I8x16Abs [V128] -> V128,
    i16x8_abs => I16x8Abs [V128] -> V128,
    i32x4_abs => I32x4Abs [V128] -> V128,
    i64x2_abs => I64x2Abs [V128] -> V128,
    i8x16_neg => I8x16Neg [V128] -> V128,
    i16x8_neg => I16x8Neg [V128] -> V128,
    i32x4_neg => I32x4Neg [V128] -> V128,
    i64x2_neg => I64x2Neg [V128] -> V128,
    i8x16_add_sat_s => I8x16AddSatS [V128, V128] -> V128,
    i8x16_add_sat_u => I8x16AddSatU [V128, V128] -> V128,
    i8x16_sub_sat_s => I8x16SubSatS [V128, V128] -> V128,
    i8x16_sub_sat_u => I8x16SubSatU [V128, V128] -> V128,
    i16x8_add_sat_s => I16x8AddSatS [V128, V128] -> V128,
    i16x8_add_sat_u => I16x8AddSatU [V128, V128] -> V128,
    i16x8_sub_sat_s => I16x8SubSatS [V128, V128] -> V128,
    i16x8_sub_sat_u => I16x8SubSatU [V128, V128] -> V128,
    i8x16_shl => I8x16Shl [V128, I32] -> V128,
    i8x16_shr_s => I8x16ShrS [V128, I32] -> V128,
    i8x16_shr_u => I8x16ShrU [V128, I32] -> V128,
    i16x8_shl => I16x8Shl [V128, I32] -> V128,
    i16x8_shr_s => I16x8ShrS [V128, I32] -> V128,
    i16x8_shr_u => I16x8ShrU [V128, I32] -> V128,
    i32x4_shl => I32x4Shl [V128, I32] -> V128,
    i32x4_shr_s => I32x4ShrS [V128, I32] -> V128,
    i32x4_shr_u => I32x4ShrU [V128, I32] -> V128,
    i64x2_shl => I64x2Shl [V128, I32] -> V128,
    i64x2_shr_s => I64x2ShrS [V128, I32] -> V128,
    i64x2_shr_u => I64x2ShrU [V128, I32] -> V128,
    i16x8_extend_low_i8x16_s => I16x8ExtendLowI8x16S [V128] -> V128,
    i16x8_extend_high_i8x16_s => I16x8ExtendHighI8x16S [V128] -> V128,
    i16x8_extend_low_i8x16_u => I16x8ExtendLowI8x16U [V128] -> V128,
    i16x8_extend_high_i8x16_u => I16x8ExtendHighI8x16U [V128] -> V128,
    i32x4_extend_low_i16x8_s => I32x4ExtendLowI16x8S [V128] -> V128,
    i32x4_extend_high_i16x8_s => I32x4ExtendHighI16x8S [V128] -> V128,
    i32x4_extend_low_i16x8_u => I32x4ExtendLowI16x8U [V128] -> V128,
    i32x4_extend_high_i16x8_u => I32x4ExtendHighI16x8U [V128] -> V128,
    i64x2_extend_low_i32x4_s => I64x2ExtendLowI32x4S [V128] -> V128,
    i64x2_extend_high_i32x4_s => I64x2ExtendHighI32x4S [V128] -> V128,
    i64x2_extend_low_i32x4_u => I64x2ExtendLowI32x4U [V128] -> V128,
    i64x2_extend_high_i32x4_u => I64x2ExtendHighI32x4U [V128] -> V128,
    i16x8_extadd_pairwise_i8x16_s => I16x8ExtaddPairwiseI8x16S [V128] -> V128,
    i16x8_extadd_pairwise_i8x16_u => I16x8ExtaddPairwiseI8x16U [V128] -> V128,
    i32x4_extadd_pairwise_i16x8_s => I32x4ExtaddPairwiseI16x8S [V128] -> V128,
    i32x4_extadd_pairwise_i16x8_u => I32x4ExtaddPairwiseI16x8U [V128] -> V128,
    f32x4_add => F32x4Add [V128, V128] -> V128,
    f32x4_sub => F32x4Sub [V128, V128] -> V128,
    f32x4_mul => F32x4Mul [V128, V128] -> V128,
    f32x4_div => F32x4Div [V128, V128] -> V128,
    f32x4_min => F32x4Min [V128, V128] -> V128,
    f32x4_max => F32x4Max [V128, V128] -> V128,
    f32x4_abs => F32x4Abs [V128] -> V128,
    f32x4_sqrt => F32x4Sqrt [V128] -> V128,
    f32x4_ceil => F32x4Ceil [V128] -> V128,
    f32x4_floor => F32x4Floor [V128] -> V128,
    f32x4_trunc => F32x4Trunc [V128] -> V128,
    f32x4_nearest => F32x4Nearest [V128] -> V128,
    f32x4_eq => F32x4Eq [V128, V128] -> V128,
    f32x4_ne => F32x4Ne [V128, V128] -> V128,
    f32x4_lt => F32x4Lt [V128, V128] -> V128,
    f32x4_gt => F32x4Gt [V128, V128] -> V128,
    f32x4_le => F32x4Le [V128, V128] -> V128,
    f32x4_ge => F32x4Ge [V128, V128] -> V128,
    f64x2_add => F64x2Add [V128, V128] -> V128,
    f64x2_sub => F64x2Sub [V128, V128] -> V128,
    f64x2_mul => F64x2Mul [V128, V128] -> V128,
    f64x2_div => F64x2Div [V128, V128] -> V128,
    f64x2_min => F64x2Min [V128, V128] -> V128,
    f64x2_max => F64x2Max [V128, V128] -> V128,
    f64x2_abs => F64x2Abs [V128] -> V128,
    f64x2_sqrt => F64x2Sqrt [V128] -> V128,
    f64x2_ceil => F64x2Ceil [V128] -> V128,
    f64x2_floor => F64x2Floor [V128] -> V128,
    f64x2_trunc => F64x2Trunc [V128] -> V128,
    f64x2_nearest => F64x2Nearest [V128] -> V128,
    f64x2_eq => F64x2Eq [V128, V128] -> V128,
    f64x2_ne => F64x2Ne [V128, V128] -> V128,
    f64x2_lt => F64x2Lt [V128, V128] -> V128,
    f64x2_gt => F64x2Gt [V128, V128] -> V128,
    f64x2_le => F64x2Le [V128, V128] -> V128,
    f64x2_ge => F64x2Ge [V128, V128] -> V128,
    i32x4_trunc_sat_f32x4_s => I32x4TruncSatF32x4S [V128] -> V128,
    i32x4_trunc_sat_f32x4_u => I32x4TruncSatF32x4U [V128] -> V128,
    f32x4_convert_i32x4_s => F32x4ConvertI32x4S [V128] -> V128,
    f32x4_convert_i32x4_u => F32x4ConvertI32x4U [V128] -> V128,
    f64x2_promote_low_f32x4 => F64x2PromoteLowF32x4 [V128] -> V128,
    f32x4_demote_f64x2_zero => F32x4DemoteF64x2Zero [V128] -> V128,
    f32_add => F32Add [F32, F32] -> F32,
    f32_sub => F32Sub [F32, F32] -> F32,
    f32_mul => F32Mul [F32, F32] -> F32,
    f32_div => F32Div [F32, F32] -> F32,
    f32_min => F32Min [F32, F32] -> F32,
    f32_max => F32Max [F32, F32] -> F32,
    f32_abs => F32Abs [F32] -> F32,
    f32_neg => F32Neg [F32] -> F32,
    f32_sqrt => F32Sqrt [F32] -> F32,
    f32_ceil => F32Ceil [F32] -> F32,
    f32_floor => F32Floor [F32] -> F32,
    f32_trunc => F32Trunc [F32] -> F32,
    f32_nearest => F32Nearest [F32] -> F32,
    f32_eq => F32Eq [F32, F32] -> I32,
    f32_ne => F32Ne [F32, F32] -> I32,
    f32_lt => F32Lt [F32, F32] -> I32,
    f32_gt => F32Gt [F32, F32] -> I32,
    f32_le => F32Le [F32, F32] -> I32,
    f32_ge => F32Ge [F32, F32] -> I32,
    f64_add => F64Add [F64, F64] -> F64,
    f64_sub => F64Sub [F64, F64] -> F64,
    f64_mul => F64Mul [F64, F64] -> F64,
    f64_div => F64Div [F64, F64] -> F64,
    f64_min => F64Min [F64, F64] -> F64,
    f64_max => F64Max [F64, F64] -> F64,
    f64_abs => F64Abs [F64] -> F64,
    f64_neg => F64Neg [F64] -> F64,
    f64_sqrt => F64Sqrt [F64] -> F64,
    f64_ceil => F64Ceil [F64] -> F64,
    f64_floor => F64Floor [F64] -> F64,
    f64_trunc => F64Trunc [F64] -> F64,
    f64_nearest => F64Nearest [F64] -> F64,
    f64_eq => F64Eq [F64, F64] -> I32,
    f64_ne => F64Ne [F64, F64] -> I32,
    f64_lt => F64Lt [F64, F64] -> I32,
    f64_gt => F64Gt [F64, F64] -> I32,
    f64_le => F64Le [F64, F64] -> I32,
    f64_ge => F64Ge [F64, F64] -> I32,
    f32_convert_i32_s => F32ConvertI32S [I32] -> F32,
    f32_convert_i32_u => F32ConvertI32U [I32] -> F32,
    f32_convert_i64_s => F32ConvertI64S [I64] -> F32,
    f32_convert_i64_u => F32ConvertI64U [I64] -> F32,
    f64_convert_i32_s => F64ConvertI32S [I32] -> F64,
    f64_convert_i32_u => F64ConvertI32U [I32] -> F64,
    f64_convert_i64_s => F64ConvertI64S [I64] -> F64,
    f64_convert_i64_u => F64ConvertI64U [I64] -> F64,
    f32_demote_f64 => F32DemoteF64 [F64] -> F32,
    f64_promote_f32 => F64PromoteF32 [F32] -> F64,
    f32_reinterpret_i32 => F32ReinterpretI32 [I32] -> F32,
    i32_reinterpret_f32 => I32ReinterpretF32 [F32] -> I32,
    f64_reinterpret_i64 => F64ReinterpretI64 [I64] -> F64,
    i64_reinterpret_f64 => I64ReinterpretF64 [F64] -> I64,
    i32_trunc_sat_f32_s => I32TruncSatF32S [F32] -> I32,
    i32_trunc_sat_f32_u => I32TruncSatF32U [F32] -> I32,
    i32_trunc_sat_f64_s => I32TruncSatF64S [F64] -> I32,
    i32_trunc_sat_f64_u => I32TruncSatF64U [F64] -> I32,
    i64_trunc_sat_f32_s => I64TruncSatF32S [F32] -> I64,
    i64_trunc_sat_f32_u => I64TruncSatF32U [F32] -> I64,
    i64_trunc_sat_f64_s => I64TruncSatF64S [F64] -> I64,
    i64_trunc_sat_f64_u => I64TruncSatF64U [F64] -> I64,
}

/// Lane operations (the immediate is the lane).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneOp {
    I8x16ExtractU,
    I8x16ExtractS,
    I16x8ExtractU,
    I16x8ExtractS,
    I32x4Extract,
    I64x2Extract,
    I8x16Replace,
    I16x8Replace,
    I32x4Replace,
    I64x2Replace,
}

impl LaneOp {
    pub fn ty(self) -> Ty {
        match self {
            Self::I64x2Extract => Ty::I64,
            Self::I8x16ExtractU
            | Self::I8x16ExtractS
            | Self::I16x8ExtractU
            | Self::I16x8ExtractS
            | Self::I32x4Extract => Ty::I32,
            _ => Ty::V128,
        }
    }
    pub fn is_replace(self) -> bool {
        self.ty() == Ty::V128
    }

    /// Emits the instruction for `lane`.
    pub fn emit(self, s: &mut wasm_encoder::InstructionSink<'_>, lane: u8) {
        match self {
            Self::I8x16ExtractU => s.i8x16_extract_lane_u(lane),
            Self::I8x16ExtractS => s.i8x16_extract_lane_s(lane),
            Self::I16x8ExtractU => s.i16x8_extract_lane_u(lane),
            Self::I16x8ExtractS => s.i16x8_extract_lane_s(lane),
            Self::I32x4Extract => s.i32x4_extract_lane(lane),
            Self::I64x2Extract => s.i64x2_extract_lane(lane),
            Self::I8x16Replace => s.i8x16_replace_lane(lane),
            Self::I16x8Replace => s.i16x8_replace_lane(lane),
            Self::I32x4Replace => s.i32x4_replace_lane(lane),
            Self::I64x2Replace => s.i64x2_replace_lane(lane),
        };
    }
}

/// Loads (Wasm's, by result and width).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadOp {
    I32,
    I32_8S,
    I32_16S,
    I64,
    I64_8U,
    I64_8S,
    I64_16U,
    I64_16S,
    I64_32U,
    I64_32S,
    V128,
}

impl LoadOp {
    pub fn ty(self) -> Ty {
        match self {
            Self::I32 | Self::I32_8S | Self::I32_16S => Ty::I32,
            Self::V128 => Ty::V128,
            _ => Ty::I64,
        }
    }

    /// Emits the load at `offset` (no alignment hint).
    pub fn emit(self, s: &mut wasm_encoder::InstructionSink<'_>, offset: u64) {
        let m = memarg(offset, 0);
        match self {
            Self::I32 => s.i32_load(m),
            Self::I32_8S => s.i32_load8_s(m),
            Self::I32_16S => s.i32_load16_s(m),
            Self::I64 => s.i64_load(m),
            Self::I64_8U => s.i64_load8_u(m),
            Self::I64_8S => s.i64_load8_s(m),
            Self::I64_16U => s.i64_load16_u(m),
            Self::I64_16S => s.i64_load16_s(m),
            Self::I64_32U => s.i64_load32_u(m),
            Self::I64_32S => s.i64_load32_s(m),
            Self::V128 => s.v128_load(m),
        };
    }
}

/// Stores (Wasm's, by value type and width).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreOp {
    I32,
    I64,
    I64_8,
    I64_16,
    I64_32,
    V128,
}

impl StoreOp {
    /// Emits the store at `offset` (no alignment hint).
    pub fn emit(self, s: &mut wasm_encoder::InstructionSink<'_>, offset: u64) {
        let m = memarg(offset, 0);
        match self {
            Self::I32 => s.i32_store(m),
            Self::I64 => s.i64_store(m),
            Self::I64_8 => s.i64_store8(m),
            Self::I64_16 => s.i64_store16(m),
            Self::I64_32 => s.i64_store32(m),
            Self::V128 => s.v128_store(m),
        };
    }
}

/// One operation. Operations that leave the block's straight line ([`Op::Exit`]) end the
/// list they are in; [`Op::If`] nests lists.
#[derive(Clone, Debug)]
pub enum Op {
    I32(V, i32),
    I64(V, i64),
    V128(V, i128),
    /// `d = p(args...)`.
    Pure(V, P, Vec<V>),
    /// `select(a, b, c)`: `a` if `c != 0`, else `b`.
    Select(V, V, V, V),
    /// Lane extract (`[vector]`) or replace (`[vector, scalar]`).
    Lane(V, LaneOp, u8, Vec<V>),
    Shuffle(V, [u8; 16], V, V),
    /// `d = load(base + offset)`.
    Load(V, LoadOp, Base, u64),
    /// `store(base + offset, v)`.
    Store(StoreOp, Base, u64, V),
    Get(V, Slot),
    Set(Slot, V),
    /// The runtime address of the region's page plus an offset (code is position
    /// independent).
    Pc(V, i64),
    /// `d = x / y` with every case defined: `x / 0 = 0` and `MIN / -1 = MIN` (`signed`), the
    /// guest semantics the front half asked for.
    Div {
        d: V,
        wide: bool,
        signed: bool,
        x: V,
        y: V,
    },
    /// `if cond { then } else { else_ }`, then `out[i].0 = cond ? out[i].1 : out[i].2` (the
    /// values each arm computed last).
    If {
        cond: V,
        then: Vec<Op>,
        else_: Vec<Op>,
        out: Vec<(V, V, V)>,
    },
    /// A guest memory access ([`Mem`]).
    Mem(Box<Mem>),
    /// The reference interpreter executes the instruction at page offset `pc` (index `k` in
    /// its block). It reads slots `reads` and may write slots `writes` (they are in memory
    /// around it); a nonzero status leaves the region, giving back the rest of the block.
    Step {
        pc: u64,
        k: u32,
        reads: Slots,
        writes: Slots,
    },
    /// `d = helper[index](args)` (a function the front half supplies; pure).
    Helper(V, u8, Vec<V>),
    /// Leaves the region with exit `code` at the address `pc`, giving back `giveback`
    /// instructions of the block.
    Exit {
        code: i32,
        pc: V,
        giveback: u32,
    },
}

/// A guest memory access with permissions: the back half checks the TLB (`write`: the write
/// table) for `size` bytes at `addr` plus `check`, runs `fast` with the host address in
/// [`Mem::host`] on a hit, and otherwise fills the TLB or has the reference interpreter
/// execute the instruction (which reads `reads` and may write `writes`).
#[derive(Clone, Debug)]
pub struct Mem {
    pub pc: u64,
    pub k: u32,
    pub addr: V,
    pub write: bool,
    pub size: u64,
    pub check: Check,
    pub reads: Slots,
    pub writes: Slots,
    pub host: V,
    pub fast: Vec<Op>,
}

/// An extra condition of a memory access's fast path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    None,
    /// The value of this slot (a stack pointer used as the base) must be 16-byte aligned.
    Aligned16(Slot),
    /// The address must be aligned to this many bytes.
    Aligned(u64),
}

/// How a block ends (when its ops did not leave).
#[derive(Clone, Debug)]
pub enum Term {
    /// Continue at this page offset (a direct branch or the fall-through).
    Jump(u64),
    /// `cond ? taken : fallthrough`.
    Branch(V, u64, u64),
    /// The ops ended in an [`Op::Exit`] on every path.
    Left,
}

/// One block in IR form.
#[derive(Clone, Debug)]
pub struct IrBlock {
    /// Ops per guest instruction (`ops[k]` lowers instruction `k`).
    pub ops: Vec<Vec<Op>>,
    pub term: Term,
    /// The type of every value.
    pub types: Vec<Ty>,
}

/// Adds `w` to `reads[v]` (saturating; values outside the table are ignored).
pub fn bump_reads(reads: &mut [u32], v: V, w: u32) {
    if let Some(n) = reads.get_mut(v.0 as usize) {
        *n = n.saturating_add(w);
    }
}

/// Adds `w` per read of each value in `ops` to `reads`, weighting what a back half emits more
/// than once: a memory access's address and fast path (its slow arm repeats them) and a
/// division's operands (its guards read them again). A back half uses the counts to decide
/// which values stay on the Wasm stack and which get a local.
pub fn count_reads(ops: &[Op], w: u32, reads: &mut [u32]) {
    for op in ops {
        match op {
            Op::I32(..) | Op::I64(..) | Op::V128(..) | Op::Pc(..) | Op::Get(..) => {}
            Op::Step { .. } => {}
            Op::Pure(_, _, args) | Op::Lane(_, _, _, args) | Op::Helper(_, _, args) => {
                for &a in args {
                    bump_reads(reads, a, w);
                }
            }
            Op::Select(_, a, b, c) => {
                for v in [*a, *b, *c] {
                    bump_reads(reads, v, w);
                }
            }
            Op::Shuffle(_, _, a, b) => {
                bump_reads(reads, *a, w);
                bump_reads(reads, *b, w);
            }
            Op::Load(_, _, base, _) => {
                if let Base::Val(b) = base {
                    bump_reads(reads, *b, w);
                }
            }
            Op::Store(_, base, _, v) => {
                if let Base::Val(b) = base {
                    bump_reads(reads, *b, w);
                }
                bump_reads(reads, *v, w);
            }
            Op::Set(_, v) => bump_reads(reads, *v, w),
            Op::Div { x, y, .. } => {
                bump_reads(reads, *x, 2 * w);
                bump_reads(reads, *y, 2 * w);
            }
            Op::If {
                cond,
                then,
                else_,
                out,
            } => {
                bump_reads(reads, *cond, w);
                count_reads(then, w, reads);
                count_reads(else_, w, reads);
                for &(_, t, e) in out {
                    bump_reads(reads, t, w);
                    bump_reads(reads, e, w);
                }
            }
            Op::Mem(m) => {
                bump_reads(reads, m.addr, 2 * w);
                count_reads(&m.fast, 2 * w, reads);
            }
            Op::Exit { pc, .. } => bump_reads(reads, *pc, w),
        }
    }
}
