//! Per-instruction tests for Advanced SIMD instructions whose results were computed by hand
//! from the Arm Architecture Reference Manual's pseudocode (the interpreter is the reference
//! the translator is compared with, so it needs checks of its own).
//!
//! Encodings are written out field by field next to each test.

use wasmshim_aarch64::Cpu;
use wasmshim_cpu::{Event, Fault, GuestCpu, GuestMemory, PAGE_SIZE, PagedMemory, Perms};

const CODE: u64 = 0x1_0000;
/// FPSR.QC, the cumulative saturation bit.
const QC: u64 = 1 << 27;

/// Runs one instruction with V1 = `a` and V2 = `b`; returns V0 and FPSR.
fn one(insn: u32, a: u128, b: u128) -> (u128, u64) {
    let mut mem = PagedMemory::new();
    mem.map(CODE, PAGE_SIZE, Perms::RX).unwrap();
    mem.poke(CODE, &insn.to_le_bytes()).unwrap();
    let mut cpu = Cpu::new();
    cpu.set_pc(CODE);
    cpu.set_vreg(1, a);
    cpu.set_vreg(2, b);
    assert_eq!(
        cpu.run_until_event(&mut mem, 1),
        Event::Yield,
        "{insn:#010x}"
    );
    (cpu.vreg(0), cpu.fpsr())
}

/// A shift count for the low byte of a shift-by-register operand.
fn sh(n: i8) -> u128 {
    u128::from(n as u8)
}

const TOP: u128 = 0x8000_0000_0000_0000;
const ONES: u128 = 0xffff_ffff_ffff_ffff;

// Three same: 0 Q U 01110 size 1 Rm opcode 1 Rn Rd (vector),
//             01 U 11110 size 1 Rm opcode 1 Rn Rd (scalar); opcode 01001 = (S|U)QSHL.
const UQSHL_D: u32 = 0x7ee2_4c20; // uqshl d0, d1, d2
const SQSHL_D: u32 = 0x5ee2_4c20; // sqshl d0, d1, d2
const UQSHL_2D: u32 = 0x6ee2_4c20; // uqshl v0.2d, v1.2d, v2.2d
const SQSHL_2D: u32 = 0x4ee2_4c20; // sqshl v0.2d, v1.2d, v2.2d
const UQSHL_16B: u32 = 0x6e22_4c20; // uqshl v0.16b, v1.16b, v2.16b

#[test]
fn uqshl_by_register_saturates_a_top_bit_shifted_out() {
    // UnsignedSatQ(x << 64) with x != 0: all ones and QC (the case that wrapped to 0).
    assert_eq!(one(UQSHL_D, TOP, sh(64)), (ONES, QC));
    assert_eq!(one(UQSHL_D, TOP | 1, sh(127)), (ONES, QC));
    assert_eq!(one(UQSHL_D, ONES, sh(100)), (ONES, QC));
    assert_eq!(one(UQSHL_D, TOP, sh(1)), (ONES, QC));
    // In range: exact, no QC.
    assert_eq!(one(UQSHL_D, 1, sh(63)), (TOP, 0));
    assert_eq!(one(UQSHL_D, 0, sh(64)), (0, 0));
    // Negative counts shift right, truncating.
    assert_eq!(one(UQSHL_D, TOP, sh(-1)), (TOP >> 1, 0));
    assert_eq!(one(UQSHL_D, ONES, sh(-64)), (0, 0));
    assert_eq!(one(UQSHL_D, ONES, sh(-128)), (0, 0));
}

#[test]
fn sqshl_by_register_saturates_to_the_signed_bounds() {
    // SignedSatQ: the extreme of the element's sign.
    assert_eq!(one(SQSHL_D, TOP, sh(64)), (TOP, QC));
    assert_eq!(one(SQSHL_D, 1, sh(64)), (ONES >> 1, QC));
    assert_eq!(one(SQSHL_D, TOP >> 1, sh(1)), (ONES >> 1, QC));
    assert_eq!(one(SQSHL_D, ONES, sh(127)), (TOP, QC));
    // In range.
    assert_eq!(one(SQSHL_D, 1, sh(62)), (TOP >> 1, 0));
    assert_eq!(one(SQSHL_D, ONES, sh(63)), (TOP, 0));
    // Right shifts are arithmetic.
    assert_eq!(one(SQSHL_D, ONES, sh(-64)), (ONES, 0));
    assert_eq!(one(SQSHL_D, TOP, sh(-63)), (ONES, 0));
}

#[test]
fn qshl_by_register_works_lane_by_lane() {
    let a = TOP | 1 << 64;
    let b = sh(64) | sh(63) << 64;
    // Lane 0 saturates, lane 1 is exact; QC because one lane saturated.
    assert_eq!(one(UQSHL_2D, a, b), (ONES | TOP << 64, QC));
    assert_eq!(one(SQSHL_2D, a, b), (TOP | (ONES >> 1) << 64, QC));
    // Bytes: 0x80 << 8 saturates to 0xff, 0x01 << 7 is exact, 0x80 >> 7 is 1.
    let a = 0x80 | 0x01 << 8 | 0x80 << 16;
    let b = sh(8) | sh(7) << 8 | sh(-7) << 16;
    assert_eq!(one(UQSHL_16B, a, b), (0xff | 0x80 << 8 | 0x01 << 16, QC));
}

#[test]
fn undefined_simd_instructions_report_their_word() {
    // rev64 with 64-bit elements (size = 11) is unallocated:
    // 0 Q 0 01110 size 10000 00000 10 Rn Rd, Q = 1, Rn = 1, Rd = 0.
    let insn = 0x4ee0_0820;
    let mut mem = PagedMemory::new();
    mem.map(CODE, PAGE_SIZE, Perms::RX).unwrap();
    mem.poke(CODE, &u32::to_le_bytes(insn)).unwrap();
    let mut cpu = Cpu::new();
    cpu.set_pc(CODE);
    assert_eq!(
        cpu.run_until_event(&mut mem, 1),
        Event::Fault(Fault::Undefined { pc: CODE, insn })
    );
    assert_eq!((cpu.pc(), cpu.retired()), (CODE, 0));
}
