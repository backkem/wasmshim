//! Checkpoints of a running machine: the substrate's state in a deterministic,
//! versioned byte form, which a personality extends with its own objects.
//!
//! * [`Writer`] and [`Reader`]: little-endian, length-prefixed, maps in key order. The same
//!   state always gives the same bytes, whatever engine ran the guest up to it.
//! * [`Ckpt`]: what can be saved and loaded. The substrate implements it for its own types
//!   ([`crate::WaitKey`], [`crate::Blocker`], [`crate::TaskState`], CPU contexts); a
//!   personality implements it for the types it plugs in ([`crate::Types`]) and for its
//!   objects, using [`Objects`] to keep shared objects shared.
//!
//! Loading never trusts the bytes: a malformed checkpoint is an error, never a panic.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::string::String;
use alloc::vec::Vec;

use alloc::rc::Rc;
use wasmshim_cpu::{
    A64Context, CpuContext, GuestCpu, GuestMemory, PAGE_SIZE, PageRef, PageSource, PageView, Perms,
};

use crate::sched::{Isa, SchedConfig};
use crate::space::{AddressSpace, MapKind, Vma};
use crate::task::{Task, TaskState, Tid};
use crate::wait::{Blocker, WaitKey};
use crate::{Core, PendingWait, Types};

/// `"WNCK"`: the first bytes of every checkpoint.
pub const MAGIC: [u8; 4] = *b"WNCK";
/// Bumped whenever the format changes meaning.
pub const VERSION: u32 = 1;

/// A checkpoint that cannot be loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CkptError(pub String);

impl CkptError {
    pub fn new(what: &str) -> Self {
        Self(String::from(what))
    }
}

impl core::fmt::Display for CkptError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "the checkpoint cannot be loaded: {}", self.0)
    }
}

impl core::error::Error for CkptError {}

/// Appends a checkpoint's bytes.
#[derive(Default)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bytes(&mut self, b: &[u8]) {
        (b.len() as u64).save(self);
        self.buf.extend_from_slice(b);
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Reads a checkpoint's bytes.
pub struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8], CkptError> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| CkptError::new("checkpoint truncated"))?;
        let b = &self.buf[self.at..end];
        self.at = end;
        Ok(b)
    }

    pub fn bytes(&mut self) -> Result<&'a [u8], CkptError> {
        let n = u64::load(self)?;
        let n = usize::try_from(n).map_err(|_| CkptError::new("length out of range"))?;
        self.take(n)
    }

    /// Whether every byte was read.
    pub fn done(&self) -> bool {
        self.at == self.buf.len()
    }
}

/// A value a checkpoint holds.
pub trait Ckpt: Sized {
    fn save(&self, w: &mut Writer);
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError>;
}

macro_rules! ckpt_int {
    ($($t:ty),*) => {$(
        impl Ckpt for $t {
            fn save(&self, w: &mut Writer) {
                w.buf.extend_from_slice(&self.to_le_bytes());
            }
            fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
                let b = r.take(core::mem::size_of::<$t>())?;
                let mut a = [0u8; core::mem::size_of::<$t>()];
                a.copy_from_slice(b);
                Ok(<$t>::from_le_bytes(a))
            }
        }
    )*};
}

ckpt_int!(u8, u16, u32, u64, u128, i32, i64);

impl Ckpt for bool {
    fn save(&self, w: &mut Writer) {
        u8::from(*self).save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        match u8::load(r)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(CkptError::new("bad bool")),
        }
    }
}

impl<T: Ckpt> Ckpt for Option<T> {
    fn save(&self, w: &mut Writer) {
        match self {
            None => 0u8.save(w),
            Some(v) => {
                1u8.save(w);
                v.save(w);
            }
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        match u8::load(r)? {
            0 => Ok(None),
            1 => Ok(Some(T::load(r)?)),
            _ => Err(CkptError::new("bad option")),
        }
    }
}

impl<T: Ckpt> Ckpt for Vec<T> {
    fn save(&self, w: &mut Writer) {
        (self.len() as u64).save(w);
        for v in self {
            v.save(w);
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        let n = u64::load(r)?;
        // Every element takes at least a byte: a length beyond the input is malformed.
        if n > (r.buf.len() - r.at) as u64 {
            return Err(CkptError::new("vector length out of range"));
        }
        (0..n).map(|_| T::load(r)).collect()
    }
}

impl<T: Ckpt> Ckpt for VecDeque<T> {
    fn save(&self, w: &mut Writer) {
        (self.len() as u64).save(w);
        for v in self {
            v.save(w);
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(Vec::<T>::load(r)?.into())
    }
}

impl<K: Ckpt + Ord, V: Ckpt> Ckpt for BTreeMap<K, V> {
    fn save(&self, w: &mut Writer) {
        (self.len() as u64).save(w);
        for (k, v) in self {
            k.save(w);
            v.save(w);
        }
    }
    /// Keys must strictly increase, as [`Ckpt::save`] writes them: one state, one byte form.
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        let mut map = BTreeMap::new();
        for (k, v) in Vec::<(K, V)>::load(r)? {
            if map.last_key_value().is_some_and(|(last, _)| *last >= k) {
                return Err(CkptError::new("map keys out of order or repeated"));
            }
            map.insert(k, v);
        }
        Ok(map)
    }
}

impl<K: Ckpt + Ord> Ckpt for BTreeSet<K> {
    fn save(&self, w: &mut Writer) {
        (self.len() as u64).save(w);
        for k in self {
            k.save(w);
        }
    }
    /// Elements must strictly increase, as [`Ckpt::save`] writes them.
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        let mut set = BTreeSet::new();
        for k in Vec::<K>::load(r)? {
            if set.last().is_some_and(|last| *last >= k) {
                return Err(CkptError::new("set elements out of order or repeated"));
            }
            set.insert(k);
        }
        Ok(set)
    }
}

impl<A: Ckpt, B: Ckpt> Ckpt for (A, B) {
    fn save(&self, w: &mut Writer) {
        self.0.save(w);
        self.1.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok((A::load(r)?, B::load(r)?))
    }
}

impl<A: Ckpt, B: Ckpt, C: Ckpt> Ckpt for (A, B, C) {
    fn save(&self, w: &mut Writer) {
        self.0.save(w);
        self.1.save(w);
        self.2.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok((A::load(r)?, B::load(r)?, C::load(r)?))
    }
}

impl<T: Ckpt + Copy + Default, const N: usize> Ckpt for [T; N] {
    fn save(&self, w: &mut Writer) {
        for v in self {
            v.save(w);
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        let mut a = [T::default(); N];
        for v in &mut a {
            *v = T::load(r)?;
        }
        Ok(a)
    }
}

impl Ckpt for A64Context {
    fn save(&self, w: &mut Writer) {
        self.x.save(w);
        self.sp.save(w);
        self.pc.save(w);
        self.flags.save(w);
        self.v.save(w);
        self.fpsr.save(w);
        self.fpcr.save(w);
        self.tp.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(Self {
            x: Ckpt::load(r)?,
            sp: Ckpt::load(r)?,
            pc: Ckpt::load(r)?,
            flags: Ckpt::load(r)?,
            v: Ckpt::load(r)?,
            fpsr: Ckpt::load(r)?,
            fpcr: Ckpt::load(r)?,
            tp: Ckpt::load(r)?,
        })
    }
}

impl Ckpt for CpuContext {
    fn save(&self, w: &mut Writer) {
        match self {
            Self::A64(c) => {
                0u8.save(w);
                c.save(w);
            }
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        match u8::load(r)? {
            0 => Ok(Self::A64(A64Context::load(r)?)),
            _ => Err(CkptError::new("unknown ISA")),
        }
    }
}

impl<K: Ckpt> Ckpt for WaitKey<K> {
    fn save(&self, w: &mut Writer) {
        match self {
            Self::Object(id) => {
                0u8.save(w);
                id.save(w);
            }
            Self::AnyIo => 1u8.save(w),
            Self::Futex(space, addr) => {
                2u8.save(w);
                space.save(w);
                addr.save(w);
            }
            Self::Personal(k) => {
                3u8.save(w);
                k.save(w);
            }
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(match u8::load(r)? {
            0 => Self::Object(Ckpt::load(r)?),
            1 => Self::AnyIo,
            2 => Self::Futex(Ckpt::load(r)?, Ckpt::load(r)?),
            3 => Self::Personal(Ckpt::load(r)?),
            _ => return Err(CkptError::new("bad wait key")),
        })
    }
}

impl<K: Ckpt, P: Ckpt> Ckpt for Blocker<K, P> {
    fn save(&self, w: &mut Writer) {
        match self {
            Self::Sleep { until, rem } => {
                0u8.save(w);
                until.save(w);
                rem.save(w);
            }
            Self::Restart {
                key,
                deadline,
                resume_pc,
                timeout_result,
            } => {
                1u8.save(w);
                key.save(w);
                deadline.save(w);
                resume_pc.save(w);
                timeout_result.save(w);
            }
            Self::Futex {
                key,
                bitset,
                deadline,
                seq,
            } => {
                2u8.save(w);
                key.save(w);
                bitset.save(w);
                deadline.save(w);
                seq.save(w);
            }
            Self::Personal(p) => {
                3u8.save(w);
                p.save(w);
            }
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(match u8::load(r)? {
            0 => Self::Sleep {
                until: Ckpt::load(r)?,
                rem: Ckpt::load(r)?,
            },
            1 => Self::Restart {
                key: Ckpt::load(r)?,
                deadline: Ckpt::load(r)?,
                resume_pc: Ckpt::load(r)?,
                timeout_result: Ckpt::load(r)?,
            },
            2 => Self::Futex {
                key: Ckpt::load(r)?,
                bitset: Ckpt::load(r)?,
                deadline: Ckpt::load(r)?,
                seq: Ckpt::load(r)?,
            },
            3 => Self::Personal(Ckpt::load(r)?),
            _ => return Err(CkptError::new("bad blocker")),
        })
    }
}

impl<K: Ckpt, P: Ckpt> Ckpt for TaskState<K, P> {
    fn save(&self, w: &mut Writer) {
        match self {
            Self::Runnable => 0u8.save(w),
            Self::Blocked(b) => {
                1u8.save(w);
                b.save(w);
            }
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(match u8::load(r)? {
            0 => Self::Runnable,
            1 => Self::Blocked(Ckpt::load(r)?),
            _ => return Err(CkptError::new("bad task state")),
        })
    }
}

/// Shared objects in a checkpoint: each object (an `Rc` in the personality) is saved once and
/// referenced by its index, so sharing (dup'd descriptors, fork-shared tables) survives a
/// restore. Saving keys objects by address; indices follow first-save order, which is
/// deterministic because the walk is.
#[derive(Default)]
pub struct Objects {
    by_addr: BTreeMap<usize, u64>,
}

impl Objects {
    /// The index of the object at `addr` if it was saved (links between saved objects).
    pub fn get(&self, addr: usize) -> Option<u64> {
        self.by_addr.get(&addr).copied()
    }

    /// The index of the object at `addr`, and whether this is its first save.
    pub fn index(&mut self, addr: usize) -> (u64, bool) {
        let next = self.by_addr.len() as u64;
        match self.by_addr.get(&addr) {
            Some(&i) => (i, false),
            None => {
                self.by_addr.insert(addr, next);
                (next, true)
            }
        }
    }
}

/// Writes the header every checkpoint starts with.
pub fn write_header(w: &mut Writer) {
    w.buf.extend_from_slice(&MAGIC);
    VERSION.save(w);
}

/// Checks the header [`write_header`] wrote.
pub fn read_header(r: &mut Reader<'_>) -> Result<(), CkptError> {
    if r.take(4)? != MAGIC {
        return Err(CkptError::new("not a checkpoint"));
    }
    match u32::load(r)? {
        VERSION => Ok(()),
        _ => Err(CkptError::new("unsupported checkpoint version")),
    }
}

impl<K: Ckpt> Ckpt for PendingWait<K> {
    fn save(&self, w: &mut Writer) {
        self.key.save(w);
        self.deadline.save(w);
        self.timeout_result.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(Self {
            key: Ckpt::load(r)?,
            deadline: Ckpt::load(r)?,
            timeout_result: Ckpt::load(r)?,
        })
    }
}

/// Creates a task's CPU on restore, with the given register state.
pub type NewCpu<'a> = &'a mut dyn FnMut(&CpuContext) -> Box<dyn GuestCpu>;

/// Maps one mapping's memory on restore ([`AddressSpace::load`]).
pub type Replay<'a, K> = &'a mut dyn FnMut(&mut dyn GuestMemory, &Vma<K>) -> Result<(), CkptError>;

impl<T: Types> Core<T>
where
    T::Task: Ckpt,
    T::Key: Ckpt,
    T::Park: Ckpt,
{
    /// Saves the substrate's state: every task (its CPU's register state and its own retired
    /// count, not the CPU's internals), the run queue, the id counters and time accounting.
    pub fn save(&self, w: &mut Writer) {
        let cfg = &self.cfg;
        match cfg.isa {
            Isa::Aarch64 => 0u8.save(w),
        }
        cfg.quantum.save(w);
        cfg.ns_per_insn.save(w);
        (self.tasks.len() as u64).save(w);
        for t in self.tasks.values() {
            t.tid.save(w);
            t.cpu.context().save(w);
            t.own_retired().save(w);
            t.state.save(w);
            t.timed.save(w);
            t.last_syscall.save(w);
            t.ext.save(w);
        }
        self.run_queue.save(w);
        self.next_space_id.save(w);
        self.next_wait_seq.save(w);
        self.next_object_id.save(w);
        self.since_yield.save(w);
        self.accrual_start.save(w);
        self.busy_until.save(w);
        self.retired_dead.save(w);
        self.block_on.save(w);
        (self.diags.len() as u64).save(w);
        for d in &self.diags {
            w.bytes(d.as_bytes());
        }
        self.slice_now.save(w);
        self.current.save(w);
        self.current_pc.save(w);
    }

    /// Restores what [`Core::save`] wrote, under `cfg` (which must schedule like the saved
    /// one: same ISA, quantum and time per instruction; tracing may differ), creating each
    /// task's CPU with `new_cpu`.
    pub fn load(
        cfg: SchedConfig,
        r: &mut Reader<'_>,
        new_cpu: NewCpu<'_>,
    ) -> Result<Self, CkptError> {
        let isa = match u8::load(r)? {
            0 => Isa::Aarch64,
            _ => return Err(CkptError::new("unknown ISA")),
        };
        let cfg = cfg.checked();
        if isa != cfg.isa || u64::load(r)? != cfg.quantum || u64::load(r)? != cfg.ns_per_insn {
            return Err(CkptError::new(
                "the checkpoint was taken with another scheduler configuration",
            ));
        }
        let mut core = Self::new(cfg);
        let n = u64::load(r)?;
        for _ in 0..n {
            let tid = Tid::load(r)?;
            let ctx = CpuContext::load(r)?;
            let own = u64::load(r)?;
            let cpu = new_cpu(&ctx);
            // The task's own count is what the guest observes (CPU-time clocks); the new CPU's
            // absolute count is its own business.
            let retired_base = cpu.retired().wrapping_sub(own);
            let state = Ckpt::load(r)?;
            let timed = Ckpt::load(r)?;
            let last_syscall = Ckpt::load(r)?;
            let ext = T::Task::load(r)?;
            let mut t = Task::new(tid, cpu, retired_base, ext);
            t.state = state;
            t.timed = timed;
            t.last_syscall = last_syscall;
            // Saved in tid order: a repeated or out-of-order tid is malformed.
            if core
                .tasks
                .last_key_value()
                .is_some_and(|(&last, _)| last >= tid)
            {
                return Err(CkptError::new("duplicate or unordered task"));
            }
            core.tasks.insert(tid, t);
        }
        core.run_queue = Ckpt::load(r)?;
        if core
            .run_queue
            .iter()
            .any(|tid| !core.tasks.contains_key(tid))
        {
            return Err(CkptError::new("run queue names an unknown task"));
        }
        core.next_space_id = Ckpt::load(r)?;
        core.next_wait_seq = Ckpt::load(r)?;
        core.next_object_id = Ckpt::load(r)?;
        core.since_yield = Ckpt::load(r)?;
        core.accrual_start = Ckpt::load(r)?;
        core.busy_until = Ckpt::load(r)?;
        core.retired_dead = Ckpt::load(r)?;
        core.block_on = Ckpt::load(r)?;
        let n = u64::load(r)?;
        for _ in 0..n {
            let d = r.bytes()?;
            let d = core::str::from_utf8(d).map_err(|_| CkptError::new("bad diagnostic"))?;
            core.diags.push(String::from(d));
        }
        core.slice_now = Ckpt::load(r)?;
        core.current = Ckpt::load(r)?;
        core.current_pc = Ckpt::load(r)?;
        Ok(core)
    }
}

impl Ckpt for Perms {
    fn save(&self, w: &mut Writer) {
        (u8::from(self.read()) | u8::from(self.write()) << 1 | u8::from(self.exec()) << 2).save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        match u8::load(r)? {
            b if b < 8 => Ok(Perms::new(b & 1 != 0, b & 2 != 0, b & 4 != 0)),
            _ => Err(CkptError::new("bad permissions")),
        }
    }
}

impl<K: Ckpt> Ckpt for Vma<K> {
    fn save(&self, w: &mut Writer) {
        self.start.save(w);
        self.end.save(w);
        self.perms.save(w);
        self.kind.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(Self {
            start: Ckpt::load(r)?,
            end: Ckpt::load(r)?,
            perms: Ckpt::load(r)?,
            kind: Ckpt::load(r)?,
        })
    }
}

/// What a mapping holds right after a restore replayed it, before the saved pages are written
/// back: the personality's answer for each mapping ([`AddressSpace::save`]). A checkpoint
/// stores exactly the pages that differ from it.
pub enum Baseline {
    /// Zeros (anonymous memory).
    Zero,
    /// The pages of a source, the mapping's first page being page `first` of it (a private
    /// file mapping). Must be what the restored machine's replay maps.
    Source(Rc<dyn PageSource>, u64),
    /// A shared object's pages (saved with the object, not with the address space).
    Shared,
}

const PAGE: usize = PAGE_SIZE as usize;

/// The bytes of a page source's page, or an error.
fn source_page(s: &Rc<dyn PageSource>, index: u64) -> Result<PageRef, CkptError> {
    s.page(index)
        .map_err(|_| CkptError::new("a page source failed while saving"))
}

/// Collects what [`AddressSpace::save`] stores: permission runs that differ from the
/// mappings', and pages that differ from their baselines.
struct PageDiff<'v, K> {
    vmas: Vec<(&'v Vma<K>, Baseline)>,
    perms: Vec<(u64, u64, Perms)>,
    pages: Vec<(u64, PageRef)>,
}

impl<K> PageDiff<'_, K> {
    fn vma_at(&self, addr: u64) -> Result<usize, CkptError> {
        let i = self.vmas.partition_point(|(v, _)| v.end <= addr);
        match self.vmas.get(i) {
            Some((v, _)) if v.start <= addr => Ok(i),
            _ => Err(CkptError::new("a mapped page outside every mapping")),
        }
    }

    /// `n` pages from `addr` with permissions `perms` that hold `content` (`None`: zeros).
    fn pages(
        &mut self,
        addr: u64,
        n: u64,
        perms: Perms,
        content: Option<&PageRef>,
        shared: bool,
    ) -> Result<(), CkptError> {
        let end = addr + n * PAGE_SIZE;
        let mut at = addr;
        while at < end {
            let i = self.vma_at(at)?;
            let (vma, base) = &self.vmas[i];
            let chunk_end = end.min(vma.end);
            if perms != vma.perms {
                match self.perms.last_mut() {
                    Some((s, len, p)) if *p == perms && *s + *len * PAGE_SIZE == at => {
                        *len += (chunk_end - at) / PAGE_SIZE;
                    }
                    _ => self.perms.push((at, (chunk_end - at) / PAGE_SIZE, perms)),
                }
            }
            match (base, shared) {
                (Baseline::Shared, true) => {}
                (Baseline::Shared, false) | (_, true) => {
                    return Err(CkptError::new(
                        "a shared mapping that does not hold shared pages, or the reverse",
                    ));
                }
                (Baseline::Zero, false) => {
                    if let Some(c) = content
                        && c.iter().any(|&b| b != 0)
                    {
                        self.pages.push((at, c.clone()));
                    }
                }
                (Baseline::Source(s, first), false) => {
                    let mut p = at;
                    while p < chunk_end {
                        let b = source_page(s, first + (p - vma.start) / PAGE_SIZE)?;
                        match content {
                            Some(c) if c[..] != b[..] => self.pages.push((p, c.clone())),
                            None if b.iter().any(|&x| x != 0) => {
                                self.pages.push((p, Rc::new([0; PAGE])));
                            }
                            _ => {}
                        }
                        p += PAGE_SIZE;
                    }
                }
            }
            at = chunk_end;
        }
        Ok(())
    }
}

impl<K: MapKind + Ckpt> AddressSpace<K> {
    /// Saves this address space: its id, its mappings, and the permissions and pages that a
    /// restore would not get back by replaying the mappings (`baseline` says what replaying a
    /// mapping gives). The pages go to `pages` in address order (bulk memory belongs to the
    /// snapshot container's page table); the record keeps their count. Fails if
    /// the memory cannot list its pages or a source fails.
    pub fn save(
        &self,
        w: &mut Writer,
        baseline: &mut dyn FnMut(&Vma<K>) -> Baseline,
        pages: &mut dyn FnMut(u64, &[u8]),
    ) -> Result<(), CkptError> {
        let vmas: Vec<&Vma<K>> = self.vmas().collect();
        let mut diff = PageDiff {
            vmas: vmas.iter().map(|&v| (v, baseline(v))).collect(),
            perms: Vec::new(),
            pages: Vec::new(),
        };
        let mut err = None;
        let listed = self.mem.for_each_mapped(&mut |addr, n, perms, view| {
            if err.is_some() {
                return;
            }
            let r = match view {
                PageView::Zero => diff.pages(addr, n, perms, None, false),
                PageView::Data(d) => diff.pages(addr, 1, perms, Some(&Rc::new(*d)), false),
                PageView::Source(s, i) => match source_page(s, i) {
                    Ok(c) => diff.pages(addr, 1, perms, Some(&c), false),
                    Err(e) => Err(e),
                },
                PageView::Shared(_) => diff.pages(addr, 1, perms, None, true),
            };
            if let Err(e) = r {
                err = Some(e);
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
        listed.map_err(|_| CkptError::new("this address space cannot list its pages"))?;
        self.id.save(w);
        (vmas.len() as u64).save(w);
        for v in &vmas {
            v.save(w);
        }
        diff.perms.save(w);
        (diff.pages.len() as u64).save(w);
        for (addr, page) in &diff.pages {
            pages(*addr, &page[..]);
        }
        Ok(())
    }

    /// Restores what [`AddressSpace::save`] wrote over `mem` (empty): `replay` maps each
    /// mapping's memory the way the saved `baseline` described it (it must not record
    /// mappings: the list is restored as it was), then the saved permissions and `pages` (the
    /// ones `save` gave out, in order) are written back.
    pub fn load(
        r: &mut Reader<'_>,
        mem: Box<dyn GuestMemory>,
        replay: Replay<'_, K>,
        pages: &[(u64, &[u8])],
    ) -> Result<Self, CkptError> {
        let id = u64::load(r)?;
        let vmas: Vec<Vma<K>> = Ckpt::load(r)?;
        if vmas.windows(2).any(|p| p[0].end > p[1].start)
            || vmas
                .iter()
                .any(|v| v.start >= v.end || v.start % PAGE_SIZE != 0 || v.end % PAGE_SIZE != 0)
        {
            return Err(CkptError::new("bad mapping list"));
        }
        let mut space = Self::new(id, mem);
        for v in &vmas {
            replay(&mut *space.mem, v)?;
        }
        space.set_vmas(vmas);
        let perms: Vec<(u64, u64, Perms)> = Ckpt::load(r)?;
        for (addr, n, p) in perms {
            let len = n
                .checked_mul(PAGE_SIZE)
                .ok_or_else(|| CkptError::new("bad permission run"))?;
            space
                .mem
                .protect(addr, len, p)
                .map_err(|_| CkptError::new("bad permission run"))?;
        }
        if u64::load(r)? != pages.len() as u64 {
            return Err(CkptError::new(
                "the address space's page count does not match",
            ));
        }
        for &(addr, bytes) in pages {
            if bytes.len() != PAGE {
                return Err(CkptError::new("a saved page of the wrong size"));
            }
            space
                .mem
                .poke(addr, bytes)
                .map_err(|_| CkptError::new("a saved page outside the restored mappings"))?;
        }
        Ok(space)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round<T: Ckpt + PartialEq + core::fmt::Debug>(v: T) {
        let mut w = Writer::new();
        v.save(&mut w);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(T::load(&mut r).unwrap(), v);
        assert!(r.done());
        // Every strict prefix is an error, never a panic.
        for n in 0..bytes.len() {
            let mut r = Reader::new(&bytes[..n]);
            let _ = T::load(&mut r);
        }
    }

    #[test]
    fn values_round_trip() {
        round(0x1234_5678_9abc_def0u64);
        round(Some(-5i64));
        round(alloc::vec![1u32, 2, 3]);
        let mut m = BTreeMap::new();
        m.insert(3u64, alloc::vec![7u8]);
        m.insert(1u64, alloc::vec![]);
        round(m);
        let mut ctx = A64Context::default();
        ctx.x[30] = 7;
        ctx.v[31] = u128::MAX;
        ctx.pc = 0x40_0000;
        round(CpuContext::A64(ctx));
        let b: TaskState<u32, u8> = TaskState::Blocked(Blocker::Restart {
            key: WaitKey::Futex(2, 0x1000),
            deadline: Some(9),
            resume_pc: 4,
            timeout_result: None,
        });
        round(b);
        round(TaskState::<u32, u8>::Blocked(Blocker::Personal(1)));
    }

    #[test]
    fn malformed_input_is_an_error() {
        let mut r = Reader::new(&[9]);
        assert!(bool::load(&mut r).is_err());
        // A vector claiming more elements than bytes left.
        let mut w = Writer::new();
        u64::MAX.save(&mut w);
        let bytes = w.into_bytes();
        assert!(Vec::<u8>::load(&mut Reader::new(&bytes)).is_err());
    }

    #[test]
    fn shared_objects_are_saved_once() {
        let mut o = Objects::default();
        assert_eq!(o.index(0x100), (0, true));
        assert_eq!(o.index(0x200), (1, true));
        assert_eq!(o.index(0x100), (0, false));
    }
}
