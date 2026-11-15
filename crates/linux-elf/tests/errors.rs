//! A rejected image reports why, and names the memory failure underneath.

use std::error::Error;

use wasmshim_cpu::MapError;
use wasmshim_linux_elf::ElfError;

#[test]
fn a_map_failure_is_the_source_of_the_load_error() {
    let e = ElfError::from(MapError::OutOfRange);
    assert_eq!(
        e.to_string(),
        "mapping a segment failed: the range leaves the address space"
    );
    let source = e.source().expect("the mapping error underneath");
    assert_eq!(source.to_string(), MapError::OutOfRange.to_string());
    assert!(ElfError::NotElf.source().is_none());
    assert_eq!(ElfError::NotElf.to_string(), "not an ELF file");
}
