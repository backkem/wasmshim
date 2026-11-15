//! The profiler is the interpreter plus counting: the same instructions, the
//! same events at the same budgets, and the hot blocks of what ran.

use wasmshim_aarch64::Cpu;
use wasmshim_aarch64::asm::assemble_bytes;
use wasmshim_aarch64_engine::mem::EngineMem;
use wasmshim_aarch64_engine::profile::{DEFAULT_THRESHOLD, Profiler};
use wasmshim_cpu::{Event, GuestCpu, GuestMemory, PAGE_SIZE, PagedMemory, Perms};

const CODE: u64 = 0x1_0000;
/// The loop's first instruction (`add x0, x0, #3`, then `subs x1, x1, #1` and `b.ne` back),
/// the entry of the block that runs 100 times.
const LOOP: u64 = CODE + 8;

fn program() -> EngineMem {
    let mut m = EngineMem::new(PagedMemory::new());
    m.map(CODE, PAGE_SIZE, Perms::RX).unwrap();
    let code = "mov x0, #0; mov x1, #100; \
                .word 0x91000c00; .word 0xf1000421; .word 0x54ffffc1; \
                mov x8, #93; svc #0";
    m.poke(CODE, &assemble_bytes(code, CODE).unwrap()).unwrap();
    m
}

/// Runs to the first syscall in slices of `budget`; returns (events, retired, x0).
fn run(cpu: &mut dyn GuestCpu, budget: u64) -> (Vec<Event>, u64, u64) {
    let mut mem = program();
    cpu.set_pc(CODE);
    let mut events = Vec::new();
    loop {
        let e = cpu.run_until_event(&mut mem, budget);
        let done = !matches!(e, Event::Yield);
        events.push(e);
        if done {
            break;
        }
    }
    let wasmshim_cpu::CpuContext::A64(ctx) = cpu.context();
    (events, cpu.retired(), ctx.x[0])
}

#[test]
fn a_profiled_run_is_the_interpreted_run() {
    for budget in [1, 2, 7, 1000] {
        let expect = run(&mut Cpu::new(), budget);
        let p = Profiler::new(DEFAULT_THRESHOLD);
        let got = run(&mut p.cpu(), budget);
        assert_eq!(got, expect, "budget {budget}");
        assert_eq!(expect.2, 300);
    }
}

#[test]
fn the_profile_holds_the_hot_block_only() {
    let p = Profiler::new(DEFAULT_THRESHOLD);
    run(&mut p.cpu(), 1000);
    let prof = p.take();
    assert_eq!(prof.pages.len(), 1);
    let page = prof.pages.values().next().unwrap();
    assert_eq!(page.entries, vec![(LOOP & 0xfff) as u16]);
    assert_eq!(page.bytes.len(), PAGE_SIZE as usize);
    assert_eq!(
        &page.bytes[..4],
        &program().fetch(CODE).unwrap().to_le_bytes()
    );
    assert!(p.take().pages.is_empty(), "take leaves it empty");

    // Below the threshold nothing is hot.
    let p = Profiler::new(101);
    run(&mut p.cpu(), 1000);
    assert!(p.profile().pages.is_empty());
}

#[test]
fn a_misaligned_pc_at_a_page_end_faults_as_in_the_interpreter() {
    // The last three bytes of a page leave less than one instruction to the page's end; the
    // profiler still runs one, so the interpreter raises its alignment fault.
    for offset in [1, 2, 3, PAGE_SIZE - 3, PAGE_SIZE - 2, PAGE_SIZE - 1] {
        let pc = CODE + offset;
        let mut expect = Cpu::new();
        expect.set_pc(pc);
        let want = expect.run_until_event(&mut program(), 1000);
        assert_eq!(
            want,
            Event::Fault(wasmshim_cpu::Fault::Alignment { pc, addr: pc })
        );
        let p = Profiler::new(1);
        let mut cpu = p.cpu();
        cpu.set_pc(pc);
        assert_eq!(cpu.run_until_event(&mut program(), 1000), want);
        assert_eq!(cpu.retired(), expect.retired());
        assert!(p.profile().pages.is_empty(), "a misaligned PC is no block");
    }
}
