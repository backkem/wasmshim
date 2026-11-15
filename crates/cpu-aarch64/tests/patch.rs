//! Code patching: a permission-ignoring write
//! ([`GuestMemory::poke`]) to an executable page is seen by the next instruction fetched from
//! it, although the interpreter caches decoded instructions per page; and patches built with
//! [`wasmshim_aarch64::asm`] run. Runs natively and inside wasmtime.

use wasmshim_aarch64::Cpu;
use wasmshim_aarch64::asm::assemble_bytes;
use wasmshim_cpu::{Event, GuestCpu, GuestMemory, PAGE_SIZE, PagedMemory, Perms};

const CODE: u64 = 0x1_0000;
const F: u64 = CODE + 0x100;
const STACK_TOP: u64 = 0x5_0000;

/// `bl f; svc #0` at CODE, `f: mov x0, #1; ret` at F, on a second code page the code at
/// CODE + PAGE_SIZE (`mov x1, #7; svc #0`).
fn machine() -> (Cpu, PagedMemory) {
    let mut mem = PagedMemory::new();
    mem.map(CODE, 2 * PAGE_SIZE, Perms::RX).unwrap();
    mem.map(STACK_TOP - PAGE_SIZE, PAGE_SIZE, Perms::RW)
        .unwrap();
    mem.poke(CODE, &assemble_bytes("bl 0x10100; svc #0", CODE).unwrap())
        .unwrap();
    mem.poke(F, &assemble_bytes("mov x0, #1; ret", F).unwrap())
        .unwrap();
    mem.poke(
        CODE + PAGE_SIZE,
        &assemble_bytes("mov x1, #7; svc #0", CODE + PAGE_SIZE).unwrap(),
    )
    .unwrap();
    let mut cpu = Cpu::new();
    cpu.set_sp(STACK_TOP);
    (cpu, mem)
}

/// Runs from `pc` to the next event; returns (x0, instructions retired by this run).
fn call(cpu: &mut Cpu, mem: &mut PagedMemory, pc: u64) -> (u64, u64) {
    cpu.set_pc(pc);
    let before = cpu.retired();
    let ev = cpu.run_until_event(mem, 1000);
    assert!(matches!(ev, Event::Syscall(_)), "{ev:?}");
    (cpu.x(0), cpu.retired() - before)
}

#[test]
fn a_patch_is_seen_by_the_next_fetch_through_the_decoded_cache() {
    let (mut cpu, mut mem) = machine();
    // Twice unpatched: the second run comes from the decoded-instruction cache.
    assert_eq!(call(&mut cpu, &mut mem, CODE), (1, 4));
    assert_eq!(call(&mut cpu, &mut mem, CODE), (1, 4));
    // Make f return 2.
    let (epoch, other) = (mem.code_epoch(), mem.code_version(CODE + PAGE_SIZE));
    let version = mem.code_version(F);
    mem.poke(F, &assemble_bytes("mov x0, #2", F).unwrap())
        .unwrap();
    assert_ne!(mem.code_epoch(), epoch, "a write to code moves the epoch");
    assert_ne!(mem.code_version(F), version, "and the page's version");
    assert_eq!(
        mem.code_version(CODE + PAGE_SIZE),
        other,
        "but no other page's: only the patched page is decoded again"
    );
    assert_eq!(call(&mut cpu, &mut mem, CODE), (2, 4));
    // Return early from the caller instead: the `bl` becomes a `mov x0, #3`; one fewer
    // instruction retires.
    mem.poke(CODE, &assemble_bytes("mov x0, #3", CODE).unwrap())
        .unwrap();
    assert_eq!(call(&mut cpu, &mut mem, CODE), (3, 2));
    // The other page still runs (from its cached decode).
    cpu.set_pc(CODE + PAGE_SIZE);
    assert!(matches!(
        cpu.run_until_event(&mut mem, 1000),
        Event::Syscall(_)
    ));
    assert_eq!(cpu.x(1), 7);
}

#[test]
fn a_patch_stays_private_to_its_address_space() {
    // Like ptrace: a copy made before the patch (a forked child) keeps the original code.
    let (mut cpu, mut mem) = machine();
    assert_eq!(call(&mut cpu, &mut mem, CODE), (1, 4));
    let child = mem.duplicate();
    mem.poke(F, &assemble_bytes("mov x0, #2", F).unwrap())
        .unwrap();
    assert_eq!(call(&mut cpu, &mut mem, CODE), (2, 4));
    let mut word = [0; 4];
    child.read(F, &mut word).unwrap();
    assert_eq!(
        word,
        assemble_bytes("mov x0, #1", F).unwrap()[..],
        "the copy still holds the original instruction"
    );
}

#[test]
fn a_patch_that_traps_is_an_ordinary_fault_or_event() {
    // `brk` makes the patched function a breakpoint the guest hits (here: a fault event).
    let (mut cpu, mut mem) = machine();
    mem.poke(F, &assemble_bytes("brk #0", F).unwrap()).unwrap();
    cpu.set_pc(CODE);
    let ev = cpu.run_until_event(&mut mem, 1000);
    assert!(matches!(ev, Event::Fault(_)), "{ev:?}");
    assert_eq!(cpu.pc(), F);
}
