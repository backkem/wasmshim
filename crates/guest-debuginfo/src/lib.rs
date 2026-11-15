//! Symbols and source lines of a guest program, for debuggers.
//!
//! [`DebugInfo::parse`] reads an ELF file's symbol tables (`.symtab`, else `.dynsym`; Rust names
//! demangled without their hash) and its DWARF line tables (`.debug_line`, versions 2 to 5) into
//! sorted tables, and answers in **link-time addresses**: a personality adds the load bias of
//! the mapping ([`DebugInfo::vaddr_of_offset`] gives the link address of a file offset).
//!
//! Personality-neutral and host-side only: nothing a guest sees depends on it.

#![deny(unsafe_code)]

use std::borrow::Cow;

use object::{Object, ObjectSection, ObjectSegment, ObjectSymbol, SymbolKind};

/// Why a file could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// A function or a data object (a static: what a watchpoint names).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    /// Link-time address.
    pub addr: u64,
    pub size: u64,
    /// Demangled, without a Rust hash (`debuggee::sum`).
    pub name: String,
}

/// One row of a line table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    addr: u64,
    file: u32,
    line: u32,
    stmt: bool,
    /// The end of a sequence: no line from here (until the next sequence's first row).
    end: bool,
}

/// The source position of an address: the row that covers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Line<'a> {
    pub file: &'a str,
    pub line: u32,
    /// Where the row starts (link-time).
    pub addr: u64,
    /// Where the next row starts.
    pub end: u64,
    /// The row starts a statement (a place to stop when stepping by line).
    pub stmt: bool,
}

/// A program's symbols and line tables.
#[derive(Clone, Debug, Default)]
pub struct DebugInfo {
    /// Sorted by address.
    symbols: Vec<Symbol>,
    files: Vec<String>,
    /// Sorted by address (sequence ends before rows at the same address).
    rows: Vec<Row>,
    /// `PT_LOAD` segments: (link address, file offset, file size, executable).
    segments: Vec<(u64, u64, u64, bool)>,
}

/// The page size mappings are aligned to.
const PAGE: u64 = 0x1000;

fn demangle(name: &str) -> String {
    format!("{:#}", rustc_demangle::demangle(name))
}

impl DebugInfo {
    /// Reads `bytes` (an ELF file). A file without symbols or lines gives empty tables.
    pub fn parse(bytes: &[u8]) -> Result<Self, Error> {
        let obj = object::File::parse(bytes).map_err(|e| Error(format!("ELF: {e}")))?;
        let mut out = DebugInfo::default();
        for seg in obj.segments() {
            let (offset, size) = seg.file_range();
            let exec = match seg.flags() {
                object::SegmentFlags::Elf { p_flags, .. } => p_flags.contains(object::elf::PF_X),
                _ => false,
            };
            out.segments.push((seg.address(), offset, size, exec));
        }
        let mut syms: Vec<Symbol> = Vec::new();
        let mut add = |s: object::Symbol<'_, '_>| {
            if !matches!(s.kind(), SymbolKind::Text | SymbolKind::Data) || s.address() == 0 {
                return;
            }
            if let Ok(name) = s.name()
                && !name.is_empty()
            {
                syms.push(Symbol {
                    addr: s.address(),
                    size: s.size(),
                    name: demangle(name),
                });
            }
        };
        if obj.symbols().next().is_some() {
            obj.symbols().for_each(&mut add);
        } else {
            obj.dynamic_symbols().for_each(&mut add);
        }
        syms.sort_by(|a, b| a.addr.cmp(&b.addr).then(b.size.cmp(&a.size)));
        syms.dedup_by(|a, b| a.addr == b.addr);
        out.symbols = syms;
        out.read_lines(&obj)?;
        Ok(out)
    }

    fn read_lines(&mut self, obj: &object::File<'_>) -> Result<(), Error> {
        let endian = if obj.is_little_endian() {
            gimli::RunTimeEndian::Little
        } else {
            gimli::RunTimeEndian::Big
        };
        let section = |id: gimli::SectionId| -> Result<Cow<'_, [u8]>, gimli::Error> {
            Ok(obj
                .section_by_name(id.name())
                .and_then(|s| s.uncompressed_data().ok())
                .unwrap_or(Cow::Borrowed(&[])))
        };
        let owned =
            gimli::DwarfSections::load(section).map_err(|e| Error(format!("DWARF: {e}")))?;
        let dwarf = owned.borrow(|s| gimli::EndianSlice::new(s, endian));
        let err = |e: gimli::Error| Error(format!("DWARF: {e}"));
        let mut files: std::collections::BTreeMap<String, u32> = Default::default();
        let mut units = dwarf.units();
        while let Some(header) = units.next().map_err(err)? {
            let unit = dwarf.unit(header).map_err(err)?;
            let Some(program) = unit.line_program.clone() else {
                continue;
            };
            let mut rows = program.rows();
            let mut seq: Vec<Row> = Vec::new();
            while let Some((header, row)) = rows.next_row().map_err(err)? {
                if row.end_sequence() {
                    seq.push(Row {
                        addr: row.address(),
                        file: 0,
                        line: 0,
                        stmt: false,
                        end: true,
                    });
                    // A sequence at address 0 is code the linker discarded.
                    if seq.first().is_some_and(|r| r.addr != 0) {
                        self.rows.append(&mut seq);
                    }
                    seq.clear();
                    continue;
                }
                let path = match row.file(header) {
                    Some(f) => {
                        let name = dwarf
                            .attr_string(&unit, f.path_name())
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        let dir = f
                            .directory(header)
                            .and_then(|d| dwarf.attr_string(&unit, d).ok())
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        if dir.is_empty() || name.starts_with('/') {
                            name
                        } else {
                            format!("{dir}/{name}")
                        }
                    }
                    None => String::new(),
                };
                let next = files.len() as u32;
                let file = *files.entry(path).or_insert(next);
                seq.push(Row {
                    addr: row.address(),
                    file,
                    line: row.line().map_or(0, |l| l.get() as u32),
                    stmt: row.is_stmt(),
                    end: false,
                });
            }
        }
        let mut names = vec![String::new(); files.len()];
        for (name, i) in files {
            names[i as usize] = name;
        }
        self.files = names;
        self.rows.sort_by_key(|r| (r.addr, !r.end));
        Ok(())
    }

    /// The code symbols, sorted by address.
    pub fn symbols(&self) -> &[Symbol] {
        &self.symbols
    }

    /// Whether the program has line tables.
    pub fn has_lines(&self) -> bool {
        !self.rows.is_empty()
    }

    /// The symbol covering `addr` and the offset into it.
    pub fn symbol_at(&self, addr: u64) -> Option<(&Symbol, u64)> {
        let i = self.symbols.partition_point(|s| s.addr <= addr);
        let s = self.symbols.get(i.checked_sub(1)?)?;
        let off = addr - s.addr;
        (s.size == 0 || off < s.size).then_some((s, off))
    }

    /// The symbol called `name`: its demangled name exactly, or (if unique) the last path
    /// segments (`sum` for `debuggee::sum`).
    pub fn symbol(&self, name: &str) -> Option<&Symbol> {
        if let Some(s) = self.symbols.iter().find(|s| s.name == name) {
            return Some(s);
        }
        let suffix = format!("::{name}");
        let mut hits = self.symbols.iter().filter(|s| s.name.ends_with(&suffix));
        let first = hits.next()?;
        hits.next().is_none().then_some(first)
    }

    /// The line row covering `addr`.
    pub fn line_at(&self, addr: u64) -> Option<Line<'_>> {
        let i = self.rows.partition_point(|r| r.addr <= addr);
        let row = self.rows.get(i.checked_sub(1)?)?;
        if row.end || row.line == 0 {
            return None;
        }
        let end = self.rows.get(i).map_or(u64::MAX, |r| r.addr);
        Some(Line {
            file: &self.files[row.file as usize],
            line: row.line,
            addr: row.addr,
            end,
            stmt: row.stmt,
        })
    }

    /// The statement addresses of `file:line` (`file` matches a path's last components), first
    /// one per sequence of rows, sorted.
    pub fn line_addrs(&self, file: &str, line: u32) -> Vec<u64> {
        let matches = |path: &str| path == file || path.ends_with(&format!("/{file}"));
        let wanted: Vec<u32> = self
            .files
            .iter()
            .enumerate()
            .filter(|(_, p)| matches(p))
            .map(|(i, _)| i as u32)
            .collect();
        let mut out: Vec<u64> = Vec::new();
        let mut prev_match = false;
        for r in &self.rows {
            let m = !r.end && r.stmt && r.line == line && wanted.contains(&r.file);
            if m && !prev_match {
                out.push(r.addr);
            }
            prev_match = m;
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The link-time address of the start of a mapping at file offset `offset` (page-aligned,
    /// as a loader maps it) of an executable (`exec`) or other `PT_LOAD` segment. A mapping
    /// starts in its segment's first page, which may also hold the end of the segment before,
    /// so the segment is the one whose page-aligned range holds the offset, of the right kind,
    /// latest first.
    pub fn vaddr_of_offset(&self, offset: u64, exec: bool) -> Option<u64> {
        self.segments
            .iter()
            .rev()
            .find(|&&(_, off, size, x)| {
                x == exec && offset >= off & !(PAGE - 1) && offset < off + size.max(1)
            })
            .map(|&(vaddr, off, _, _)| vaddr.wrapping_sub(off).wrapping_add(offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_elf_is_an_error_and_empty_tables_answer_nothing() {
        assert!(DebugInfo::parse(b"not an elf").is_err());
        let d = DebugInfo::default();
        assert!(d.symbol_at(0x1000).is_none());
        assert!(d.line_at(0x1000).is_none());
        assert!(d.line_addrs("a.rs", 1).is_empty());
    }

    #[test]
    fn rows_cover_until_the_next_row_and_ends_stop_them() {
        let d = DebugInfo {
            symbols: vec![Symbol {
                addr: 0x100,
                size: 0x20,
                name: "m::f".into(),
            }],
            files: vec!["/src/m.rs".into()],
            rows: vec![
                Row {
                    addr: 0x100,
                    file: 0,
                    line: 3,
                    stmt: true,
                    end: false,
                },
                Row {
                    addr: 0x108,
                    file: 0,
                    line: 4,
                    stmt: true,
                    end: false,
                },
                Row {
                    addr: 0x110,
                    file: 0,
                    line: 3,
                    stmt: true,
                    end: false,
                },
                Row {
                    addr: 0x120,
                    file: 0,
                    line: 0,
                    stmt: false,
                    end: true,
                },
            ],
            segments: vec![(0x0, 0x0, 0x800, false), (0x1d80, 0xd80, 0x400, true)],
        };
        assert_eq!(
            d.symbol_at(0x10c).map(|(s, o)| (s.name.as_str(), o)),
            Some(("m::f", 0xc))
        );
        assert!(d.symbol_at(0x120).is_none());
        assert_eq!(d.symbol("f").map(|s| s.addr), Some(0x100));
        let l = d.line_at(0x10c).unwrap();
        assert_eq!((l.line, l.addr, l.end), (4, 0x108, 0x110));
        assert!(d.line_at(0x120).is_none());
        assert_eq!(d.line_addrs("m.rs", 3), [0x100, 0x110]);
        // The executable segment starts mid-page, in the page the one before ends in.
        assert_eq!(d.vaddr_of_offset(0x0, true), Some(0x1000));
        assert_eq!(d.vaddr_of_offset(0x0, false), Some(0x0));
    }
}
