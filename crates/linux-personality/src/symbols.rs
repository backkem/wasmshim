//! Symbols and source lines of a Linux machine's programs: the
//! ELF symbols and DWARF line tables (`wasmshim-debuginfo`) of every executable file a process
//! maps, at the addresses it maps them (the load bias of a PIE or a shared library included).
//!
//! Names are `name`, `name+0x10`, or `file.rs:12` (the line's first statement). Programs are
//! read from the VFS once and kept (host-side only; nothing a guest sees depends on it).

use std::collections::BTreeMap;
use std::rc::Rc;

use wasmshim_cpu::Perms;
use wasmshim_debuginfo::DebugInfo;

use crate::kernel::{Kernel, Pid};
use crate::mm::VmaKind;

/// Parsed programs by path (`None`: not an ELF file, or unreadable).
#[derive(Default)]
pub(crate) struct Programs {
    by_path: BTreeMap<Vec<u8>, Option<Rc<DebugInfo>>>,
}

/// One executable mapping of a program: its tables and how to turn link addresses into
/// runtime ones.
struct Mapped {
    info: Rc<DebugInfo>,
    path: Vec<u8>,
    start: u64,
    end: u64,
    /// runtime = link + bias.
    bias: u64,
}

/// Where an address is in the source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Where {
    /// The symbol and the offset into it.
    pub symbol: Option<(String, u64)>,
    /// The program (lossy UTF-8 of its path).
    pub program: String,
    /// The source line: file, line, the line's runtime address range and whether it starts a
    /// statement.
    pub line: Option<(String, u32, u64, u64, bool)>,
}

/// A symbol of a program a process maps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolInfo {
    /// Its runtime address.
    pub addr: u64,
    pub size: u64,
    pub name: String,
    /// The program (lossy UTF-8 of its path).
    pub file: String,
}

/// One frame of a call stack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub pc: u64,
    /// The program's file name (the path's last component), where the frame is mapped.
    pub file: Option<String>,
    /// The symbol (no offset), where the frame is mapped and named.
    pub symbol: Option<String>,
}

impl Kernel {
    fn program(&mut self, path: &[u8]) -> Option<Rc<DebugInfo>> {
        if let Some(p) = self.programs.by_path.get(path) {
            return p.clone();
        }
        let parsed = self
            .vfs
            .resolve(0, path, true)
            .ok()
            .and_then(|ino| self.vfs.read_all(ino).ok())
            .and_then(|bytes| DebugInfo::parse(&bytes).ok())
            .map(Rc::new);
        self.programs.by_path.insert(path.to_vec(), parsed.clone());
        parsed
    }

    /// The executable file mappings of `pid`, with their tables.
    fn mapped(&mut self, pid: Pid) -> Vec<Mapped> {
        let Some(pr) = self.procs.get(&pid) else {
            return Vec::new();
        };
        let files: Vec<(Vec<u8>, u64, u64, u64)> = pr
            .mm
            .borrow()
            .vmas()
            .filter(|v| v.perms.contains(Perms::X))
            .filter_map(|v| match &v.kind {
                VmaKind::File { path, offset } => Some((path.clone(), *offset, v.start, v.end)),
                _ => None,
            })
            .collect();
        files
            .into_iter()
            .filter_map(|(path, offset, start, end)| {
                let info = self.program(&path)?;
                let link = info.vaddr_of_offset(offset, true)?;
                Some(Mapped {
                    info,
                    path,
                    start,
                    end,
                    bias: start.wrapping_sub(link),
                })
            })
            .collect()
    }

    /// Where `addr` is in `pid`'s programs.
    pub fn where_is(&mut self, pid: Pid, addr: u64) -> Option<Where> {
        let m = self
            .mapped(pid)
            .into_iter()
            .find(|m| m.start <= addr && addr < m.end)?;
        let link = addr.wrapping_sub(m.bias);
        Some(Where {
            symbol: m.info.symbol_at(link).map(|(s, off)| (s.name.clone(), off)),
            program: String::from_utf8_lossy(&m.path).into_owned(),
            line: m.info.line_at(link).map(|l| {
                (
                    l.file.to_owned(),
                    l.line,
                    l.addr.wrapping_add(m.bias),
                    l.end.wrapping_add(m.bias),
                    l.stmt,
                )
            }),
        })
    }

    /// The symbol `name` in the first of `pid`'s mapped programs that has it.
    pub fn symbol(&mut self, pid: Pid, name: &str) -> Option<SymbolInfo> {
        self.mapped(pid).iter().find_map(|m| {
            let s = m.info.symbol(name)?;
            Some(SymbolInfo {
                addr: s.addr.wrapping_add(m.bias),
                size: s.size,
                name: s.name.clone(),
                file: String::from_utf8_lossy(&m.path).into_owned(),
            })
        })
    }

    /// The statement addresses of source line `file:line` in every program `pid` maps, in
    /// mapping order.
    pub fn line_addresses(&mut self, pid: Pid, file: &str, line: u32) -> Vec<u64> {
        self.mapped(pid)
            .iter()
            .flat_map(|m| {
                m.info
                    .line_addrs(file, line)
                    .into_iter()
                    .map(move |a| a.wrapping_add(m.bias))
            })
            .collect()
    }

    /// A thread's call stack by its frame records (AArch64: x29 chains `[fp]` = the caller's
    /// fp, `[fp + 8]` = the return address), innermost first, at most 64 frames: each its `pc`,
    /// and where it is mapped its `file` (the program) and `symbol` (no offset; a return
    /// address is named by its call instruction, `pc - 4`). Frames without a record (leaf
    /// functions, code built without frame pointers) are not seen: the stacks are as good as
    /// the guest's frame records. Reading changes nothing.
    pub fn stack(&mut self, pid: Pid, ctx: &wasmshim_cpu::CpuContext) -> Vec<Frame> {
        let wasmshim_cpu::CpuContext::A64(c) = ctx;
        let mapped = self.mapped(pid);
        let in_program = |a: u64| mapped.iter().any(|m| m.start <= a && a < m.end);
        let mut pcs = vec![(c.pc, c.pc)];
        let mut fp = c.x[29];
        // A frame record lives on the stack, above SP and not far from it; code built without
        // frame pointers leaves anything in x29, so a record that is not there, or a return
        // address outside the mapped programs, ends the walk.
        let on_stack = |fp: u64| fp >= c.sp && fp - c.sp < 8 << 20 && fp.is_multiple_of(8);
        for _ in 0..63 {
            if !on_stack(fp) {
                break;
            }
            let Some(b) = self
                .with_memory(pid, |mem| crate::debug::read_bytes(mem, fp, 16))
                .flatten()
            else {
                break;
            };
            let next = u64::from_le_bytes(b[0..8].try_into().unwrap_or_default());
            let ret = u64::from_le_bytes(b[8..16].try_into().unwrap_or_default());
            if ret == 0 || !in_program(ret.wrapping_sub(4)) {
                break;
            }
            pcs.push((ret, ret.wrapping_sub(4)));
            if next <= fp {
                break;
            }
            fp = next;
        }
        pcs.into_iter()
            .map(|(pc, at)| {
                let mut frame = Frame {
                    pc,
                    file: None,
                    symbol: None,
                };
                if let Some(mp) = mapped.iter().find(|m| m.start <= at && at < m.end) {
                    let file = String::from_utf8_lossy(&mp.path);
                    frame.file = Some(file.rsplit('/').next().unwrap_or(&file).to_owned());
                    frame.symbol = mp
                        .info
                        .symbol_at(at.wrapping_sub(mp.bias))
                        .map(|(s, _)| s.name.clone());
                }
                frame
            })
            .collect()
    }

    /// `name+0x14 (file.rs:12)` for stop info.
    pub fn symbolize_pc(&mut self, pid: Pid, pc: u64) -> Option<String> {
        let w = self.where_is(pid, pc)?;
        let mut s = match &w.symbol {
            Some((name, 0)) => name.clone(),
            Some((name, off)) => format!("{name}+{off:#x}"),
            None => return None,
        };
        if let Some((file, line, ..)) = &w.line {
            let short = file.rsplit('/').next().unwrap_or(file);
            s.push_str(&format!(" ({short}:{line})"));
        }
        Some(s)
    }

    /// The source line at `pc` and whether `pc` starts a statement of it, for stepping by line.
    pub fn source_line_at(&mut self, pid: Pid, pc: u64) -> Option<(String, u32, bool)> {
        let (file, line, start, _, stmt) = self.where_is(pid, pc)?.line?;
        Some((file, line, stmt && start == pc))
    }

    /// Resolves `spec` (`name`, `name+off`, `file:line`) in `pid` to runtime addresses.
    fn resolve_in(&mut self, pid: Pid, spec: &str) -> Vec<u64> {
        let mapped = self.mapped(pid);
        let mut out = Vec::new();
        if let Some((file, line)) = spec.rsplit_once(':')
            && let Ok(line) = line.parse::<u32>()
        {
            for m in &mapped {
                if let Some(&a) = m.info.line_addrs(file, line).first() {
                    out.push(a.wrapping_add(m.bias));
                }
            }
        } else {
            let (name, off) = match spec.rsplit_once('+') {
                Some((n, o)) => {
                    let off = o
                        .strip_prefix("0x")
                        .map_or_else(|| o.parse().ok(), |h| u64::from_str_radix(h, 16).ok());
                    match off {
                        Some(off) => (n, off),
                        None => (spec, 0),
                    }
                }
                None => (spec, 0),
            };
            for m in &mapped {
                if let Some(s) = m.info.symbol(name) {
                    out.push(s.addr.wrapping_add(m.bias).wrapping_add(off));
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The runtime addresses `spec` (`name`, `name+off`, `file:line`: the line's first
    /// statement) names in process `pid`, or in every process (`None`), sorted, without
    /// duplicates.
    pub fn resolve_addresses(&mut self, pid: Option<Pid>, spec: &str) -> Vec<u64> {
        let pids: Vec<Pid> = match pid {
            Some(p) => vec![p],
            None => self.procs.keys().copied().collect(),
        };
        let mut found: Vec<u64> = pids
            .into_iter()
            .flat_map(|p| self.resolve_in(p, spec))
            .collect();
        found.sort_unstable();
        found.dedup();
        found
    }
}
