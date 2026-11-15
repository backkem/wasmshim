//! The CPU contract's failures are errors a host can print and wrap.

use std::error::Error;

use wasmshim_cpu::{Access, Fault, MapError, MemFault, MemFaultKind, PageSourceError};

#[test]
fn faults_say_what_happened_and_where() {
    let fault = MemFault {
        addr: 0x1000,
        access: Access::Write,
        kind: MemFaultKind::Protection,
    };
    assert_eq!(fault.to_string(), "write to protected address 0x1000");
    assert_eq!(
        Fault::Memory {
            pc: 0x40_0000,
            fault
        }
        .to_string(),
        "write to protected address 0x1000 at pc 0x400000"
    );
    assert_eq!(
        Fault::Undefined {
            pc: 4,
            insn: 0xdead
        }
        .to_string(),
        "undefined instruction 0x0000dead at pc 0x4"
    );
    assert_eq!(
        Fault::Breakpoint { pc: 8, imm: 1 }.to_string(),
        "breakpoint #0x1 at pc 0x8"
    );
    assert_eq!(
        Fault::Alignment {
            pc: 8,
            addr: 0x1001
        }
        .to_string(),
        "misaligned access to 0x1001 at pc 0x8"
    );
}

#[test]
fn every_failure_is_an_error() {
    let errors: [Box<dyn Error>; 4] = [
        Box::new(Fault::Breakpoint { pc: 0, imm: 0 }),
        Box::new(MemFault {
            addr: 0,
            access: Access::Execute,
            kind: MemFaultKind::Unmapped,
        }),
        Box::new(MapError::NotMapped),
        Box::new(PageSourceError),
    ];
    for e in errors {
        assert!(!e.to_string().is_empty());
    }
    assert_eq!(
        MemFault {
            addr: 0x2000,
            access: Access::Execute,
            kind: MemFaultKind::Unmapped,
        }
        .to_string(),
        "fetch from unmapped address 0x2000"
    );
}
