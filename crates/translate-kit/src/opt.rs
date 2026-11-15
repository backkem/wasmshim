//! IR passes. ISA-neutral: they see slots, values and memory accesses.
//!
//! * [`merged_checks`]: memory accesses of a block off the same base value, with constant
//!   offsets, whose TLB checks one check of their whole span can replace.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::ir::{Check, IrBlock, Op, P, Slot, V, slot_bit};

/// The role of an instruction's memory access in a merged check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Merge {
    /// The first access of group `id`: before it, one check covers `span` bytes from the
    /// base plus `lo`, in the write table if `write`; its own offset is `off`.
    Head {
        id: u32,
        lo: i64,
        span: u64,
        write: bool,
        off: i64,
    },
    /// A later access of group `id` at offset `off` from the base.
    Member { id: u32, off: i64 },
}

/// Finds groups of memory accesses in `block` that one check can cover: same base value (a
/// slot read with no write of the slot in between), constant offsets, the same extra check
/// (none, or the same alignment of the same slot), and nothing in between that could change
/// what the check found: no reference-interpreter step, helper, structured op or exit, and no
/// store outside the group (a store could make a shared page private, moving it). At most
/// `max_span` bytes per group (within a page anyway). Returns roles by instruction index.
pub fn merged_checks(block: &IrBlock, max_span: u64) -> BTreeMap<u32, Merge> {
    // Values that are slot reads: value -> (slot, version of the slot when read).
    let mut gets: BTreeMap<V, (Slot, u32)> = BTreeMap::new();
    let mut version = [0u32; 64];
    // Address values: value -> (base slot, version, offset).
    let mut addrs: BTreeMap<V, (Slot, u32, i64)> = BTreeMap::new();
    let mut consts: BTreeMap<V, i64> = BTreeMap::new();
    struct Open {
        key: (Slot, u32),
        check: Check,
        members: Vec<(u32, i64, u64, bool)>,
    }
    let mut out = BTreeMap::new();
    let mut open: Vec<Open> = Vec::new();
    let mut next_id = 0u32;
    let close = |o: Open, out: &mut BTreeMap<u32, Merge>, id: &mut u32| {
        if o.members.len() < 2 {
            return;
        }
        let lo = o.members.iter().map(|m| m.1).min().unwrap_or(0);
        let hi = o
            .members
            .iter()
            .map(|m| m.1 + m.2 as i64)
            .max()
            .unwrap_or(0);
        let write = o.members.iter().any(|m| m.3);
        for (i, &(k, off, _, _)) in o.members.iter().enumerate() {
            out.insert(
                k,
                if i == 0 {
                    Merge::Head {
                        id: *id,
                        lo,
                        span: (hi - lo) as u64,
                        write,
                        off,
                    }
                } else {
                    Merge::Member { id: *id, off }
                },
            );
        }
        *id += 1;
    };
    // Every slot written anywhere in `ops` (nested lists included).
    fn writes(ops: &[Op], w: &mut u64) {
        for op in ops {
            match op {
                Op::Set(s, _) => *w |= slot_bit(*s),
                Op::Step { writes: ws, .. } => *w |= ws,
                Op::Mem(m) => {
                    *w |= m.writes;
                    writes(&m.fast, w);
                }
                Op::If { then, else_, .. } => {
                    writes(then, w);
                    writes(else_, w);
                }
                _ => {}
            }
        }
    }
    // A new version of every slot `op` writes.
    let bump = |version: &mut [u32; 64], op: &Op| {
        let mut w = 0;
        writes(core::slice::from_ref(op), &mut w);
        for (s, v) in version.iter_mut().enumerate() {
            if w & (1 << s) != 0 {
                *v += 1;
            }
        }
    };
    for (k, ops) in block.ops.iter().enumerate() {
        for op in ops {
            let mut barrier = false;
            match op {
                Op::I64(d, c) => {
                    consts.insert(*d, *c);
                }
                Op::Get(d, s) => {
                    gets.insert(*d, (*s, version[s.0 as usize]));
                }
                Op::Pure(d, P::I64Add, args) => {
                    let (a, b) = (args[0], args[1]);
                    let based = |v: V| gets.get(&v).copied();
                    if let (Some((s, ver)), Some(&c)) = (based(a), consts.get(&b)) {
                        addrs.insert(*d, (s, ver, c));
                    } else if let (Some(&(s, ver, o)), Some(&c)) = (addrs.get(&a), consts.get(&b)) {
                        addrs.insert(*d, (s, ver, o.wrapping_add(c)));
                    }
                }
                Op::Mem(m) => {
                    let base = addrs
                        .get(&m.addr)
                        .copied()
                        .or_else(|| gets.get(&m.addr).map(|&(s, v)| (s, v, 0)));
                    let simple = matches!(m.check, Check::None | Check::Aligned16(_));
                    let mut joined = false;
                    if let (Some((s, ver, off)), true) = (base, simple) {
                        let pos = open
                            .iter()
                            .position(|o| o.key == (s, ver) && o.check == m.check);
                        if let Some(i) = pos {
                            let o = &open[i];
                            let lo = o.members.iter().map(|x| x.1).min().unwrap_or(off).min(off);
                            let hi = o
                                .members
                                .iter()
                                .map(|x| x.1 + x.2 as i64)
                                .max()
                                .unwrap_or(0)
                                .max(off + m.size as i64);
                            if ((hi - lo) as u64) <= max_span {
                                open[i].members.push((k as u32, off, m.size, m.write));
                                joined = true;
                            }
                        }
                        if !joined {
                            open.push(Open {
                                key: (s, ver),
                                check: m.check,
                                members: alloc::vec![(k as u32, off, m.size, m.write)],
                            });
                            joined = true;
                        }
                    }
                    // A store outside every group could move a page: close the others.
                    if m.write && !joined {
                        barrier = true;
                    } else if m.write {
                        let key = base.map(|(s, v, _)| (s, v));
                        let mut keep = Vec::new();
                        for o in open.drain(..) {
                            if Some(o.key) == key && o.check == m.check {
                                keep.push(o);
                            } else {
                                close(o, &mut out, &mut next_id);
                            }
                        }
                        open = keep;
                    }
                    bump(&mut version, op);
                }
                Op::Set(s, _) => version[s.0 as usize] += 1,
                Op::Step { .. } | Op::Helper(..) | Op::If { .. } | Op::Exit { .. } => {
                    barrier = true;
                    bump(&mut version, op);
                }
                _ => {}
            }
            if barrier {
                for o in open.drain(..) {
                    close(o, &mut out, &mut next_id);
                }
            }
        }
    }
    for o in open.drain(..) {
        close(o, &mut out, &mut next_id);
    }
    out
}
