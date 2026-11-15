//! Per-instruction tests for the A64 integer ISA.
//!
//! Every encoding below was produced by LLVM's assembler (the comment is the source line) and
//! cross-checked with `llvm-objdump`, so these tests check the decoder and the interpreter
//! together against an independent encoder. They run natively and inside wasmtime
//! (`cargo test --target wasm32-wasip2 -p wasmshim-aarch64`).

use wasmshim_aarch64::Cpu;
use wasmshim_cpu::{
    Access, Event, Fault, GuestCpu, GuestMemory, MemFaultKind, PAGE_SIZE, PagedMemory, Perms,
    SyscallFrame,
};

const CODE: u64 = 0x1_0000;
const DATA: u64 = 0x2_0000;
const STACK_TOP: u64 = 0x5_0000;

struct M {
    cpu: Cpu,
    mem: PagedMemory,
}

impl M {
    fn new(code: &[u32]) -> Self {
        let mut mem = PagedMemory::new();
        mem.map(CODE, PAGE_SIZE, Perms::RX).unwrap();
        mem.map(DATA, 4 * PAGE_SIZE, Perms::RW).unwrap();
        mem.map(STACK_TOP - 4 * PAGE_SIZE, 4 * PAGE_SIZE, Perms::RW)
            .unwrap();
        let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        mem.poke(CODE, &bytes).unwrap();
        let mut cpu = Cpu::new();
        cpu.set_pc(CODE);
        cpu.set_sp(STACK_TOP);
        Self { cpu, mem }
    }

    fn with(mut self, regs: &[(u8, u64)]) -> Self {
        for &(r, v) in regs {
            self.cpu.set_x(r, v);
        }
        self
    }

    /// Runs exactly `n` instructions and expects no event.
    fn step(mut self, n: u64) -> Self {
        assert_eq!(self.cpu.run_until_event(&mut self.mem, n), Event::Yield);
        self
    }

    /// Runs every instruction given to `new` (straight-line code).
    fn run(self, code_len: usize) -> Self {
        self.step(code_len as u64)
    }

    fn x(&self, r: u8) -> u64 {
        self.cpu.x(r)
    }

    fn flags(&self) -> u64 {
        self.cpu.nzcv() >> 28
    }
}

/// Runs a straight-line sequence.
fn run(code: &[u32], regs: &[(u8, u64)]) -> M {
    M::new(code).with(regs).run(code.len())
}

const N: u64 = 8;
const Z: u64 = 4;
const C: u64 = 2;
const V: u64 = 1;

#[test]
fn add_sub_immediate() {
    // add x0, x1, #4 ; subs w2, w3, #1, lsl #12
    let m = run(&[0x9100_1020, 0x7140_0462], &[(1, 10), (3, 0x1000)]);
    assert_eq!(m.x(0), 14);
    assert_eq!(m.x(2), 0);
    assert_eq!(m.flags(), Z | C);
}

#[test]
fn add_sub_immediate_uses_sp() {
    // add x0, sp, w1, uxtw #2 (extended register form, SP base)
    let m = run(&[0x8b21_4be0], &[(1, 0xffff_ffff_0000_0003)]);
    assert_eq!(m.x(0), STACK_TOP + 12);
}

#[test]
fn move_wide() {
    // mov x0, #0x1234 ; movk x0, #0x5678, lsl #16 ; mov w1, #-1 ; mov x2, #-2
    let m = run(
        &[0xd282_4680, 0xf2aa_cf00, 0x1280_0001, 0x9280_0022],
        &[(1, u64::MAX)],
    );
    assert_eq!(m.x(0), 0x5678_1234);
    assert_eq!(m.x(1), 0xffff_ffff);
    assert_eq!(m.x(2), (-2i64) as u64);
}

#[test]
fn logical_immediate() {
    // and x0, x1, #0xff00ff00ff00ff00 ; mov w2, #0x55555555 ; eor x3, x1, #1 ; ands w4, w1, #0x80000000
    let m = run(
        &[0x9208_9c20, 0x3200_f3e2, 0xd240_0023, 0x7201_0024],
        &[(1, 0x1234_5678_9abc_def1)],
    );
    assert_eq!(m.x(0), 0x1200_5600_9a00_de00);
    assert_eq!(m.x(2), 0x5555_5555);
    assert_eq!(m.x(3), 0x1234_5678_9abc_def0);
    assert_eq!(m.x(4), 0x8000_0000);
    assert_eq!(m.flags(), N);
}

#[test]
fn bitfield_aliases() {
    // ubfx x0, x1, #8, #8 ; sbfx x2, x1, #4, #4 ; lsl w3, w1, #3 ; asr x4, x1, #2 ;
    // bfi x5, x1, #16, #8 ; sxtw x6, w1 ; uxtb w7, w1
    let code = [
        0xd348_3c20,
        0x9344_1c22,
        0x531d_7023,
        0x9342_fc24,
        0xb370_1c25,
        0x9340_7c26,
        0x5300_1c27,
    ];
    let m = run(
        &code,
        &[(1, 0x8000_0000_8765_43f9), (5, 0xaaaa_aaaa_aaaa_aaaa)],
    );
    assert_eq!(m.x(0), 0x43);
    assert_eq!(m.x(2), (-1i64) as u64); // nibble 0xf -> -1
    assert_eq!(m.x(3), 0x3b2a_1fc8);
    assert_eq!(m.x(4), 0xe000_0000_21d9_50fe);
    assert_eq!(m.x(5), 0xaaaa_aaaa_aaf9_aaaa);
    assert_eq!(m.x(6), 0xffff_ffff_8765_43f9);
    assert_eq!(m.x(7), 0xf9);
}

#[test]
fn extract_and_rotate() {
    // extr x0, x1, x2, #8 ; ror w3, w1, #4
    let m = run(
        &[0x93c2_2020, 0x1381_1023],
        &[(1, 0x1122_3344_5566_7788), (2, 0x99aa_bbcc_ddee_ff00)],
    );
    assert_eq!(m.x(0), 0x8899_aabb_ccdd_eeff);
    assert_eq!(m.x(3), 0x8556_6778);
}

#[test]
fn flags_and_conditional_select() {
    // adds x0, x1, x2 (overflow) ; cmp x1, x2 ; cset w3, lt ; csel x4, x1, x2, eq ;
    // csinc x5, x1, x2, ne ; csneg w6, w1, w2, ge
    let code = [0xab02_0020];
    let m = run(&code, &[(1, i64::MAX as u64), (2, 1)]);
    assert_eq!(m.x(0), 1 << 63);
    assert_eq!(m.flags(), N | V);

    let code = [
        0xeb02_003f,
        0x1a9f_a7e3,
        0x9a82_0024,
        0x9a82_1425,
        0x5a82_a426,
    ];
    let m = run(&code, &[(1, (-5i64) as u64), (2, 3)]);
    assert_eq!(m.flags(), N | C); // -5 - 3 = -8, no borrow in unsigned terms? 0xfff..fb >= 3 -> C
    assert_eq!(m.x(3), 1); // -5 < 3
    assert_eq!(m.x(4), 3); // not eq -> x2
    assert_eq!(m.x(5), (-5i64) as u64); // ne -> x1
    assert_eq!(m.x(6), (-3i32) as u32 as u64); // not ge -> -w2
}

#[test]
fn conditional_compare() {
    // ccmp x1, #5, #4, ne
    let code = [0xfa45_1824];
    // Z clear (ne holds): compare x1 with 5.
    let m = run(&code, &[(1, 5)]);
    assert_eq!(m.flags(), Z | C);
    // Z set (ne fails): flags = #4.
    let mut m = M::new(&code).with(&[(1, 9)]);
    m.cpu.set_nzcv(Z << 28);
    let m = m.step(1);
    assert_eq!(m.flags(), Z);
}

#[test]
fn multiply_and_divide() {
    // madd x0, x1, x2, x3 ; msub w4, w1, w2, w3 ; smulh x5, x1, x2 ; umulh x6, x1, x2 ;
    // udiv x7, x1, x2 ; sdiv w8, w1, w2 ; umaddl x9, w1, w2, x3 ; smaddl x10, w1, w2, x3
    let code = [
        0x9b02_0c20,
        0x1b02_8c24,
        0x9b42_7c25,
        0x9bc2_7c26,
        0x9ac2_0827,
        0x1ac2_0c28,
        0x9ba2_0c29,
        0x9b22_0c2a,
    ];
    let m = run(&code, &[(1, (-6i64) as u64), (2, 4), (3, 100)]);
    assert_eq!(m.x(0), 76);
    assert_eq!(m.x(4), 124);
    assert_eq!(m.x(5), u64::MAX); // high half of -24
    assert_eq!(m.x(6), 3); // high half of (2^64-6)*4
    assert_eq!(m.x(7), ((-6i64) as u64) / 4);
    assert_eq!(m.x(8), (-1i32) as u32 as u64); // -6 / 4 truncates to -1
    assert_eq!(m.x(9), 100 + 0xffff_fffa * 4);
    assert_eq!(m.x(10), 76);
    // Division by zero yields zero; i64::MIN / -1 wraps.
    let m = run(&[0x9ac2_0827, 0x1ac2_0c28], &[(1, 7), (2, 0)]);
    assert_eq!((m.x(7), m.x(8)), (0, 0));
    let m = run(&[0x1ac2_0c28], &[(1, 0x8000_0000), (2, 0xffff_ffff)]);
    assert_eq!(m.x(8), 0x8000_0000);
}

#[test]
fn one_source_ops() {
    // clz x0, x1 ; cls w2, w1 ; rbit w3, w1 ; rev x4, x1 ; rev16 w5, w1 ; rev32 x6, x1
    let code = [
        0xdac0_1020,
        0x5ac0_1422,
        0x5ac0_0023,
        0xdac0_0c24,
        0x5ac0_0425,
        0xdac0_0826,
    ];
    let m = run(&code, &[(1, 0x0000_1234_f000_0001)]);
    assert_eq!(m.x(0), 19);
    assert_eq!(m.x(2), 3);
    assert_eq!(m.x(3), 0x8000_000f);
    assert_eq!(m.x(4), 0x0100_00f0_3412_0000);
    assert_eq!(m.x(5), 0x00f0_0100);
    assert_eq!(m.x(6), 0x3412_0000_0100_00f0);
    let m = run(&[0x5ac0_1422], &[(1, 0)]);
    assert_eq!(m.x(2), 31);
}

#[test]
fn extended_and_carry_ops() {
    // sub x2, x1, w3, sxtb ; adc x0, x1, x2 ; sbcs w3, w1, w2 ; lsl x4, x1, x2 ; asr w5, w1, w2
    let mut m = M::new(&[0xcb23_8022]).with(&[(1, 100), (3, 0xff)]);
    m = m.step(1);
    assert_eq!(m.x(2), 101);
    let mut m =
        M::new(&[0x9a02_0020, 0x7a02_0023, 0x9ac2_2024, 0x1ac2_2825]).with(&[(1, 10), (2, 66)]);
    m.cpu.set_nzcv(C << 28);
    let m = m.step(4);
    assert_eq!(m.x(0), 77);
    assert_eq!(m.x(3), (10u32.wrapping_sub(66)) as u64); // carry set: no extra borrow
    assert_eq!(m.x(4), 10 << 2); // shift amount taken mod 64
    assert_eq!(m.x(5), 2); // shift amount taken mod 32
}

#[test]
fn branches() {
    // b .+8 (skips one), then mov x0, #0x1234
    let m = M::new(&[0x1400_0002, 0xd282_4680, 0xd282_4680]).step(2);
    assert_eq!(m.cpu.pc(), CODE + 12);
    assert_eq!(m.x(0), 0x1234);
    // bl .+12 sets x30
    let m = M::new(&[0x9400_0003]).step(1);
    assert_eq!((m.cpu.pc(), m.x(30)), (CODE + 12, CODE + 4));
    // cbz x1, .+8 taken / not taken
    assert_eq!(M::new(&[0xb400_0041]).step(1).cpu.pc(), CODE + 8);
    assert_eq!(
        M::new(&[0xb400_0041]).with(&[(1, 1)]).step(1).cpu.pc(),
        CODE + 4
    );
    // cbnz w1 ignores the upper half
    assert_eq!(
        M::new(&[0x3500_0041])
            .with(&[(1, 1 << 32)])
            .step(1)
            .cpu
            .pc(),
        CODE + 4
    );
    // tbnz x1, #63 / tbz w1, #3
    assert_eq!(
        M::new(&[0xb7f8_0041])
            .with(&[(1, 1 << 63)])
            .step(1)
            .cpu
            .pc(),
        CODE + 8
    );
    assert_eq!(
        M::new(&[0x3618_0081]).with(&[(1, 8)]).step(1).cpu.pc(),
        CODE + 4
    );
    assert_eq!(M::new(&[0x3618_0081]).step(1).cpu.pc(), CODE + 16);
    // b.lt after cmp
    let m = M::new(&[0xeb02_003f, 0x5400_004b])
        .with(&[(1, (-1i64) as u64), (2, 0)])
        .step(2);
    assert_eq!(m.cpu.pc(), CODE + 12);
    // br / blr / ret
    let m = M::new(&[0xd61f_0020]).with(&[(1, 0x1234)]).step(1);
    assert_eq!(m.cpu.pc(), 0x1234);
    let m = M::new(&[0xd63f_0020]).with(&[(1, 0x1234)]).step(1);
    assert_eq!((m.cpu.pc(), m.x(30)), (0x1234, CODE + 4));
    let m = M::new(&[0xd65f_03c0]).with(&[(30, 0x4444)]).step(1);
    assert_eq!(m.cpu.pc(), 0x4444);
}

#[test]
fn svc_reports_a_syscall_frame() {
    let mut m =
        M::new(&[0xd400_0001]).with(&[(8, 64), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]);
    let ev = m.cpu.run_until_event(&mut m.mem, 100);
    assert_eq!(
        ev,
        Event::Syscall(SyscallFrame {
            number: 64,
            pc: CODE + 4,
            args: [1, 2, 3, 4, 5, 6]
        })
    );
    assert_eq!(m.cpu.pc(), CODE + 4);
    assert_eq!(m.cpu.retired(), 1);
    m.cpu.set_syscall_result(42);
    assert_eq!(m.x(0), 42);
}

#[test]
fn brk_and_undefined_fault_without_retiring() {
    let mut m = M::new(&[0xd420_0020]);
    assert_eq!(
        m.cpu.run_until_event(&mut m.mem, 10),
        Event::Fault(Fault::Breakpoint { pc: CODE, imm: 1 })
    );
    let mut m = M::new(&[0x0000_0000]);
    assert_eq!(
        m.cpu.run_until_event(&mut m.mem, 10),
        Event::Fault(Fault::Undefined { pc: CODE, insn: 0 })
    );
    assert_eq!((m.cpu.pc(), m.cpu.retired()), (CODE, 0));
}

#[test]
fn fetch_faults() {
    let mut m = M::new(&[]);
    m.cpu.set_pc(DATA);
    match m.cpu.run_until_event(&mut m.mem, 10) {
        Event::Fault(Fault::Memory { pc, fault }) => {
            assert_eq!(
                (pc, fault.access, fault.kind),
                (DATA, Access::Execute, MemFaultKind::Protection)
            );
        }
        other => panic!("{other:?}"),
    }
    m.cpu.set_pc(CODE + 2);
    assert_eq!(
        m.cpu.run_until_event(&mut m.mem, 10),
        Event::Fault(Fault::Alignment {
            pc: CODE + 2,
            addr: CODE + 2
        })
    );
}

#[test]
fn budget_yields_and_resumes() {
    let code = [0x9100_0421u32; 10]; // add x1, x1, #1
    let mut m = M::new(&code);
    assert_eq!(m.cpu.run_until_event(&mut m.mem, 3), Event::Yield);
    assert_eq!((m.x(1), m.cpu.retired()), (3, 3));
    assert_eq!(m.cpu.run_until_event(&mut m.mem, 7), Event::Yield);
    assert_eq!(m.x(1), 10);
}

#[test]
fn loads_and_stores() {
    // str x0, [sp, #-16]! ; ldr x1, [sp], #16
    let m = run(&[0xf81f_0fe0, 0xf841_07e1], &[(0, 0xdead_beef_cafe_f00d)]);
    assert_eq!(m.x(1), 0xdead_beef_cafe_f00d);
    assert_eq!(m.cpu.sp(), STACK_TOP);

    let mut m = M::new(&[
        0x3940_0c22, // ldrb w2, [x1, #3]
        0x3980_0023, // ldrsb x3, [x1]
        0x79c0_0424, // ldrsh w4, [x1, #2]
        0xb980_0425, // ldrsw x5, [x1, #4]
        0xf85f_d028, // ldur x8, [x1, #-3]
    ])
    .with(&[(1, DATA + 8)]);
    m.mem
        .write(
            DATA,
            &[
                1, 2, 3, 4, 5, 6, 7, 8, 0x80, 0x11, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff,
            ],
        )
        .unwrap();
    let m = m.step(5);
    assert_eq!(m.x(2), 0xff);
    assert_eq!(m.x(3), (-128i64) as u64);
    assert_eq!(m.x(4), 0xffff_fffe); // ldrsh w: sign-extend to 32 bits only
    assert_eq!(m.x(5), u64::MAX);
    assert_eq!(
        m.x(8),
        u64::from_le_bytes([6, 7, 8, 0x80, 0x11, 0xfe, 0xff, 0xff])
    );
}

#[test]
fn register_offset_addressing() {
    // ldr x2, [x1, x3, lsl #3] ; ldr w6, [x1, w3, sxtw #2]
    let mut m = M::new(&[0xf863_7822, 0xb863_d826]).with(&[(1, DATA + 64), (3, 0xffff_fffe)]);
    m.mem.write_u64(DATA + 64 + 8 * 0xffff_fffe, 1).ok(); // out of range: must not matter
    m.mem.write_u32(DATA + 56, 0x7777).unwrap();
    let mut m2 = M::new(&[0xb863_d826]).with(&[(1, DATA + 64), (3, 0xffff_fffe)]);
    m2.mem.write_u32(DATA + 56, 0x7777).unwrap();
    let m2 = m2.step(1);
    assert_eq!(m2.x(6), 0x7777);
    // The first load faults (address way out of range) and leaves pc and x2 untouched.
    match m.cpu.run_until_event(&mut m.mem, 1) {
        Event::Fault(Fault::Memory { pc, .. }) => assert_eq!(pc, CODE),
        other => panic!("{other:?}"),
    }
    assert_eq!(m.cpu.pc(), CODE);
}

#[test]
fn pairs() {
    // stp w1, w2, [x0, #-8]! ; ldpsw x3, x4, [x0] ; ldp x1, x2, [x0, #16]
    let mut m = M::new(&[0x29bf_0801, 0x6940_1003, 0xa941_0801]).with(&[
        (0, DATA + 64),
        (1, 0x8000_0000),
        (2, 7),
    ]);
    m.mem.write_u64(DATA + 56 + 16, 11).unwrap();
    m.mem.write_u64(DATA + 56 + 24, 12).unwrap();
    let m = m.step(3);
    assert_eq!(m.x(0), DATA + 56);
    assert_eq!(m.x(3), 0xffff_ffff_8000_0000);
    assert_eq!(m.x(4), 7);
    assert_eq!((m.x(1), m.x(2)), (11, 12));
}

#[test]
fn literal_and_adr() {
    // ldr x7, .+16 (literal) ; adr x0, .+20 ; adrp x1, .+0x3000
    let mut m = M::new(&[
        0x5800_0087,
        0x1000_00a0,
        0xf000_0001,
        0,
        0x1122_3344,
        0x5566_7788,
    ]);
    let _ = &mut m;
    let m = m.step(3);
    assert_eq!(m.x(7), 0x5566_7788_1122_3344);
    assert_eq!(m.x(0), CODE + 4 + 20);
    assert_eq!(m.x(1), (CODE + 8) & !0xfff | 0x3000);
}

#[test]
fn exclusives_and_atomics() {
    // ldxr x1, [x0] ; stxr w2, x3, [x0]
    let mut m = M::new(&[0xc85f_7c01, 0xc802_7c03]).with(&[(0, DATA), (3, 99), (2, 7)]);
    m.mem.write_u64(DATA, 5).unwrap();
    let m = m.step(2);
    assert_eq!((m.x(1), m.x(2)), (5, 0));
    assert_eq!(m.mem.read_u64(DATA).unwrap(), 99);
    // A store-exclusive without a reservation fails.
    let m = M::new(&[0xc802_7c03]).with(&[(0, DATA), (3, 1)]).step(1);
    assert_eq!(m.x(2), 1);
    assert_eq!(m.mem.read_u64(DATA).unwrap(), 0);
    // The reservation does not survive a return from run_until_event.
    let mut m = M::new(&[0xc85f_7c01, 0xc802_7c03]).with(&[(0, DATA), (3, 1)]);
    let _ = m.cpu.run_until_event(&mut m.mem, 1);
    let m = m.step(1);
    assert_eq!(m.x(2), 1);
    // ldxp / stlxp
    let mut m = M::new(&[0xc87f_0801, 0xc823_8801]).with(&[(0, DATA + 16)]);
    m.mem.write_u64(DATA + 16, 1).unwrap();
    m.mem.write_u64(DATA + 24, 2).unwrap();
    let m = m.step(2);
    assert_eq!((m.x(1), m.x(2), m.x(3)), (1, 2, 0));
    assert_eq!(m.mem.read_u64(DATA + 24).unwrap(), 2);
    // misaligned exclusive
    let mut m = M::new(&[0xc85f_7c01]).with(&[(0, DATA + 4)]);
    assert_eq!(
        m.cpu.run_until_event(&mut m.mem, 1),
        Event::Fault(Fault::Alignment {
            pc: CODE,
            addr: DATA + 4
        })
    );

    // cas x1, x2, [x0]: equal -> swap; not equal -> no store
    let mut m = M::new(&[0xc8a1_7c02]).with(&[(0, DATA), (1, 5), (2, 9)]);
    m.mem.write_u64(DATA, 5).unwrap();
    let m = m.step(1);
    assert_eq!((m.x(1), m.mem.read_u64(DATA).unwrap()), (5, 9));
    let mut m = M::new(&[0xc8a1_7c02]).with(&[(0, DATA), (1, 4), (2, 9)]);
    m.mem.write_u64(DATA, 5).unwrap();
    let m = m.step(1);
    assert_eq!((m.x(1), m.mem.read_u64(DATA).unwrap()), (5, 5));
    // casal w1, w2, [x0] compares only 32 bits
    let mut m = M::new(&[0x88e1_fc02]).with(&[(0, DATA), (1, 0xffff_ffff_0000_0005), (2, 9)]);
    m.mem.write_u32(DATA, 5).unwrap();
    let m = m.step(1);
    assert_eq!((m.x(1), m.mem.read_u32(DATA).unwrap()), (5, 9));

    // ldadd w1, w2, [x0] ; swp x1, x2, [x0] ; ldclr ; ldset ; ldsmax ; ldumin
    let cases: &[(u32, u64, u64, u64)] = &[
        (0xb821_0002, 0xffff_fff0, 0x20, 0x10), // ldadd w: wraps at 32 bits
        (0xf821_8002, 7, 3, 7),                 // swp x
        (0xf821_1002, 0b0110, 0b1111, 0b1001),  // ldclr x
        (0xb821_3002, 0b0110, 0b1001, 0b1111),  // ldset w
        (0xb821_4002, 0xffff_ffff, 1, 1),       // ldsmax w: -1 < 1
        (0xf821_7002, 3, 9, 3),                 // ldumin x
    ];
    for &(insn, rs, old, new) in cases {
        let mut m = M::new(&[insn]).with(&[(0, DATA), (1, rs)]);
        m.mem.write_u64(DATA, old).unwrap();
        let m = m.step(1);
        assert_eq!(m.x(2), old, "{insn:#x}");
        assert_eq!(m.mem.read_u64(DATA).unwrap(), new, "{insn:#x}");
    }

    // casp x0, x1, x2, x3, [x4]
    let mut m = M::new(&[0x4820_7c82]).with(&[(4, DATA + 32), (0, 1), (1, 2), (2, 30), (3, 40)]);
    m.mem.write_u64(DATA + 32, 1).unwrap();
    m.mem.write_u64(DATA + 40, 2).unwrap();
    let m = m.step(1);
    assert_eq!(
        (
            m.mem.read_u64(DATA + 32).unwrap(),
            m.mem.read_u64(DATA + 40).unwrap()
        ),
        (30, 40)
    );
    assert_eq!((m.x(0), m.x(1)), (1, 2));
}

#[test]
fn acquire_release_and_barriers() {
    // ldaxr w1, [x0] ; stlr x1, [x0] ; clrex ; nop ; dmb ish ; paciasp ; autiasp ; bti c ; prfm
    let mut m = M::new(&[
        0x885f_fc01,
        0xc89f_fc01,
        0xd503_3f5f,
        0xd503_201f,
        0xd503_3bbf,
        0xd503_233f,
        0xd503_23bf,
        0xd503_245f,
        0xf980_0020,
    ])
    .with(&[(0, DATA), (1, DATA)]);
    m.mem.write_u64(DATA, 0x1_0000_0002).unwrap();
    let m = m.step(9);
    assert_eq!(m.x(1), 2);
    assert_eq!(m.mem.read_u64(DATA).unwrap(), 2);
}

#[test]
fn system_registers() {
    // msr tpidr_el0, x1 ; mrs x0, tpidr_el0 ; msr nzcv, x3 ; mrs x2, nzcv ; mrs x4, cntvct_el0
    let mut m = M::new(&[
        0xd51b_d041,
        0xd53b_d040,
        0xd51b_4203,
        0xd53b_4202,
        0xd53b_e044,
    ])
    .with(&[(1, 0x7000), (3, 0xa000_0000)]);
    m.cpu.set_virtual_time(123_456);
    let m = m.step(5);
    assert_eq!(m.x(0), 0x7000);
    assert_eq!(m.cpu.thread_pointer(), 0x7000);
    assert_eq!(m.x(2), 0xa000_0000);
    assert_eq!(m.x(4), 123_456);
}

#[test]
fn dc_zva_zeroes_a_block() {
    let mut m = M::new(&[0xd50b_7421]).with(&[(1, DATA + 70)]);
    m.mem.write(DATA, &[0xff; 192]).unwrap();
    let m = m.step(1);
    let mut buf = [0u8; 192];
    m.mem.read(DATA, &mut buf).unwrap();
    assert!(buf[..64].iter().all(|&b| b == 0xff));
    assert!(buf[64..128].iter().all(|&b| b == 0));
    assert!(buf[128..].iter().all(|&b| b == 0xff));
}

#[test]
fn simd_and_fp_loads_and_stores() {
    // ldr q0, [x1] ; str q0, [x1, #16] ; ldp q0, q1, [x1] ; ldr s0, [x1, #4] ; stp d0, d1, [sp, #-16]!
    let mut m = M::new(&[
        0x3dc0_0020,
        0x3d80_0420,
        0xad40_0420,
        0xbd40_0420,
        0x6dbf_07e0,
    ])
    .with(&[(1, DATA)]);
    let pattern: Vec<u8> = (0..16).collect();
    m.mem.write(DATA, &pattern).unwrap();
    let m = m.step(5);
    let mut buf = [0u8; 16];
    m.mem.read(DATA + 16, &mut buf).unwrap();
    assert_eq!(buf.as_slice(), pattern.as_slice());
    assert_eq!(m.cpu.vreg(1), u128::from_le_bytes(buf));
    assert_eq!(m.cpu.vreg(0), 0x0706_0504);
    assert_eq!(m.mem.read_u64(STACK_TOP - 16).unwrap(), 0x0706_0504);
    assert_eq!(
        m.mem.read_u64(STACK_TOP - 8).unwrap(),
        u64::from_le_bytes(buf[..8].try_into().unwrap())
    );
}

#[test]
fn misaligned_sp_base_faults() {
    let mut m = M::new(&[0xf841_07e1]); // ldr x1, [sp], #16
    m.cpu.set_sp(STACK_TOP - 8);
    assert_eq!(
        m.cpu.run_until_event(&mut m.mem, 1),
        Event::Fault(Fault::Alignment {
            pc: CODE,
            addr: STACK_TOP - 8
        })
    );
}

#[test]
fn writes_to_xzr_are_discarded() {
    // cmp x1, x2 writes xzr; afterwards x31 still reads zero through "mov w2, #0x55555555"-style ORR with wzr.
    let m = run(&[0xeb02_003f, 0x3200_f3e2], &[(1, 5), (2, 1)]);
    assert_eq!(m.x(2), 0x5555_5555);
    assert_eq!(m.x(31), 0);
}
