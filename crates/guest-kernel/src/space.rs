//! Address spaces: the mapping list on top of a [`GuestMemory`].
//!
//! The [`GuestMemory`] holds pages and permissions (and reserves lazily: a range
//! mapped with no access costs one range record, not a slot per page). This module remembers
//! *why* each range is mapped (the personality's [`MapKind`]: Linux's heap, stack, files and
//! shared objects; a Wasmtime platform's anonymous memory), so mappings can be split, merged,
//! re-protected and placed deterministically, and a personality can describe them
//! (`/proc/self/maps`).

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use wasmshim_cpu::{GuestMemory, MapError, PAGE_SIZE, Perms};

/// Rounds up to a page boundary (saturating).
pub const fn page_up(x: u64) -> u64 {
    x.saturating_add(PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// What a mapping is for: the personality's own description.
pub trait MapKind: Clone + PartialEq {
    /// The kind of the part of a mapping that starts `by` bytes into it (a file mapping's
    /// offset moves along).
    fn advance(&mut self, _by: u64) {}

    /// Whether a mapping of this kind and `len` bytes and a mapping of kind `next` right after
    /// it form one mapping (same kind, and contiguous offsets where kinds have them).
    fn continues(&self, _len: u64, next: &Self) -> bool {
        self == next
    }

    /// For a mapping of this kind starting at `start`: the first address past its backing
    /// object, if it has one. Pages from there stay inaccessible whatever `protect` asks
    /// (Linux: touching them is `SIGBUS`).
    fn object_end(&self, _start: u64) -> Option<u64> {
        None
    }
}

/// One mapping `[start, end)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vma<K> {
    pub start: u64,
    pub end: u64,
    pub perms: Perms,
    pub kind: K,
}

/// An address space.
pub struct AddressSpace<K> {
    /// Unique per address space (futex keys use it).
    pub id: u64,
    /// The pages themselves.
    pub mem: Box<dyn GuestMemory>,
    vmas: BTreeMap<u64, Vma<K>>,
}

impl<K: MapKind> AddressSpace<K> {
    /// An empty address space over `mem`.
    pub fn new(id: u64, mem: Box<dyn GuestMemory>) -> Self {
        Self {
            id,
            mem,
            vmas: BTreeMap::new(),
        }
    }

    /// A copy-on-write copy of this address space (for `fork`) with id `id`.
    pub fn duplicate(&self, id: u64) -> Self {
        Self {
            id,
            mem: self.mem.duplicate(),
            vmas: self.vmas.clone(),
        }
    }

    /// The mappings in address order.
    pub fn vmas(&self) -> impl Iterator<Item = &Vma<K>> {
        self.vmas.values()
    }

    /// The mapping that contains `addr`.
    pub fn vma_at(&self, addr: u64) -> Option<&Vma<K>> {
        self.overlapping(addr, addr.saturating_add(1)).next()
    }

    /// Replaces the mapping list as it is, without merging (a checkpoint restores the list
    /// exactly: `/proc/<pid>/maps` shows it). The mappings must not overlap.
    pub(crate) fn set_vmas(&mut self, vmas: Vec<Vma<K>>) {
        self.vmas = vmas.into_iter().map(|v| (v.start, v)).collect();
    }

    /// Records an already-mapped range (e.g. ELF segments mapped by a loader).
    pub fn record(&mut self, start: u64, end: u64, perms: Perms, kind: K) {
        self.cut(start, end);
        self.vmas.insert(
            start,
            Vma {
                start,
                end,
                perms,
                kind,
            },
        );
        self.merge_around(start);
    }

    /// Merges the mapping at `start` with compatible neighbours (same permissions, and a kind
    /// that [`MapKind::continues`]).
    fn merge_around(&mut self, start: u64) {
        let compatible = |a: &Vma<K>, b: &Vma<K>| {
            a.end == b.start && a.perms == b.perms && a.kind.continues(a.end - a.start, &b.kind)
        };
        let mut at = start;
        if let Some((&prev_start, prev)) = self.vmas.range(..start).next_back()
            && let Some(cur) = self.vmas.get(&start)
            && compatible(prev, cur)
        {
            let end = cur.end;
            self.vmas.remove(&start);
            if let Some(p) = self.vmas.get_mut(&prev_start) {
                p.end = end;
            }
            at = prev_start;
        }
        let Some(cur_end) = self.vmas.get(&at).map(|v| v.end) else {
            return;
        };
        if let Some(next) = self.vmas.get(&cur_end)
            && let Some(cur) = self.vmas.get(&at)
            && compatible(cur, next)
        {
            let end = next.end;
            self.vmas.remove(&cur_end);
            if let Some(c) = self.vmas.get_mut(&at) {
                c.end = end;
            }
        }
    }

    /// Maps fresh zero pages at `[start, end)`, replacing whatever was there.
    pub fn map(&mut self, start: u64, end: u64, perms: Perms, kind: K) -> Result<(), MapError> {
        self.mem.map(start, end - start, perms)?;
        self.record(start, end, perms, kind);
        Ok(())
    }

    /// Unmaps `[start, end)`; holes are fine.
    pub fn unmap(&mut self, start: u64, end: u64) -> Result<(), MapError> {
        self.mem.unmap(start, end - start)?;
        self.cut(start, end);
        Ok(())
    }

    /// Changes permissions of `[start, end)`; fails with [`MapError::NotMapped`] if any page
    /// is unmapped. Pages past a mapping's backing object stay inaccessible.
    pub fn protect(&mut self, start: u64, end: u64, perms: Perms) -> Result<(), MapError> {
        if !self.covered(start, end) {
            return Err(MapError::NotMapped);
        }
        self.mem.protect(start, end - start, perms)?;
        let pieces: Vec<Vma<K>> = self.overlapping(start, end).cloned().collect();
        self.cut(start, end);
        for mut v in pieces {
            let s = v.start.max(start);
            let e = v.end.min(end);
            if let Some(first_beyond) = v.kind.object_end(v.start) {
                let b = first_beyond.max(s);
                if b < e {
                    self.mem.protect(b, e - b, Perms::NONE)?;
                }
            }
            v.kind.advance(s - v.start);
            self.vmas.insert(
                s,
                Vma {
                    start: s,
                    end: e,
                    perms,
                    kind: v.kind,
                },
            );
        }
        Ok(())
    }

    /// `true` if every page of `[start, end)` belongs to some mapping.
    pub fn covered(&self, start: u64, end: u64) -> bool {
        let mut at = start;
        for v in self.overlapping(start, end) {
            if v.start > at {
                return false;
            }
            at = at.max(v.end);
        }
        at >= end
    }

    /// `true` if no mapping overlaps `[start, end)`.
    pub fn is_free(&self, start: u64, end: u64) -> bool {
        self.overlapping(start, end).next().is_none()
    }

    /// The highest free, page-aligned range of `len` bytes in `[min, top)` (deterministic
    /// top-down placement).
    pub fn find_free_below(&self, top: u64, min: u64, len: u64) -> Option<u64> {
        let mut top = top;
        for v in self.vmas.values().rev() {
            if v.start >= top {
                continue;
            }
            if v.end <= top && top - v.end >= len {
                return Some(top - len);
            }
            top = top.min(v.start);
        }
        (top >= min.saturating_add(len)).then(|| top - len)
    }

    fn overlapping(&self, start: u64, end: u64) -> impl Iterator<Item = &Vma<K>> {
        let first = self
            .vmas
            .range(..start)
            .next_back()
            .map(|(&k, _)| k)
            .unwrap_or(start);
        self.vmas
            .range(first..end)
            .map(|(_, v)| v)
            .filter(move |v| v.end > start && v.start < end)
    }

    /// Removes `[start, end)` from the mapping list, splitting mappings at the edges.
    fn cut(&mut self, start: u64, end: u64) {
        let hits: Vec<Vma<K>> = self.overlapping(start, end).cloned().collect();
        for v in hits {
            self.vmas.remove(&v.start);
            if v.start < start {
                self.vmas.insert(
                    v.start,
                    Vma {
                        end: start,
                        ..v.clone()
                    },
                );
            }
            if v.end > end {
                let mut kind = v.kind.clone();
                kind.advance(end - v.start);
                self.vmas.insert(
                    end,
                    Vma {
                        start: end,
                        end: v.end,
                        perms: v.perms,
                        kind,
                    },
                );
            }
        }
    }
}

impl<K: MapKind> crate::Space for AddressSpace<K> {
    fn memory(&mut self) -> &mut dyn GuestMemory {
        self.mem.as_mut()
    }
}
