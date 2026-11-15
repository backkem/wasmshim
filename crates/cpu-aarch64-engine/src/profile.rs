//! Profiling runs for ahead-of-time translation.
//!
//! [`ProfCpu`] runs the interpreter one basic block at a time (exactly the interpreter's
//! instruction stream, budgets and events, so a profiling run is the run it profiles) and counts
//! executions per block entry. When a block reaches the threshold, the content of its code page
//! is captured and the block's page offset recorded. The result is a [`Profile`]: pages keyed by
//! content ([`wasmshim_aarch64_translate::aot::page_key`]), each with its hot entries.
//!
//! A node runs it with `--engine profile` and hands the profile out through inspection
//! (`engine:profile`), so translations can cover exactly what the profiled workloads run.

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use wasmshim_aarch64::Cpu;
use wasmshim_aarch64_translate::aot::{PageKey, PageProfile, page_key};
use wasmshim_cpu::{CpuContext, DebugCpu, Event, GuestCpu, GuestMemory};

use crate::mem::EngineMem;

/// The default threshold: blocks executed this often are recorded.
pub const DEFAULT_THRESHOLD: u32 = 16;

/// Hot pages by key, in key order.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Profile {
    pub pages: BTreeMap<PageKey, PageProfile>,
    /// Per page: the block entries counted in it (in every instance of the page with this
    /// content that had a hot block), so a narrower translation can keep the hottest pages
    /// (the browser's nodes).
    pub heat: BTreeMap<PageKey, u64>,
}

impl Profile {
    /// Records the block at `offset` of the page `bytes` as hot.
    pub fn record(&mut self, bytes: &[u8], offset: u16) {
        let key = page_key(bytes);
        let p = self.pages.entry(key).or_insert_with(|| PageProfile {
            bytes: bytes.to_vec(),
            entries: Vec::new(),
        });
        if !p.entries.contains(&offset) {
            p.entries.push(offset);
        }
    }

    /// Adds `other`'s pages and entries.
    pub fn merge(&mut self, other: &Profile) {
        for p in other.pages.values() {
            for &e in &p.entries {
                self.record(&p.bytes, e);
            }
        }
        for (key, h) in &other.heat {
            *self.heat.entry(*key).or_default() += h;
        }
    }

    /// Total hot entries.
    pub fn entries(&self) -> usize {
        self.pages.values().map(|p| p.entries.len()).sum()
    }
}

/// Executions per instruction slot of one code page (one version of it, in one address space).
type Slots = Rc<[Cell<u32>; 1024]>;
/// (address space, code page, code version).
type PageId = (usize, u64, u64);

/// Executions per block entry and the profile so far.
struct Shared {
    threshold: u32,
    counts: BTreeMap<PageId, Slots>,
    /// The content key of each page instance a hot block was recorded in.
    recorded: BTreeMap<PageId, PageKey>,
    profile: Profile,
}

impl Shared {
    /// The profile with each page's heat: the counted entries of its recorded instances.
    fn with_heat(&self) -> Profile {
        let mut p = self.profile.clone();
        for (id, key) in &self.recorded {
            if let Some(slots) = self.counts.get(id) {
                let sum: u64 = slots.iter().map(|c| u64::from(c.get())).sum();
                *p.heat.entry(*key).or_default() += sum;
            }
        }
        p
    }
}

/// The profile of every [`ProfCpu`] made from one [`Profiler`].
#[derive(Clone)]
pub struct Profiler(Rc<RefCell<Shared>>);

impl Profiler {
    /// Blocks executed at least `threshold` times are recorded.
    pub fn new(threshold: u32) -> Self {
        Self(Rc::new(RefCell::new(Shared {
            threshold: threshold.max(1),
            counts: BTreeMap::new(),
            recorded: BTreeMap::new(),
            profile: Profile::default(),
        })))
    }

    /// A new CPU that adds to this profile. Its address spaces must be [`EngineMem`]s (blocks
    /// in other memories run, uncounted).
    pub fn cpu(&self) -> ProfCpu {
        ProfCpu {
            cpu: Box::new(Cpu::new()),
            shared: self.clone(),
            last: None,
            threshold: self.0.borrow().threshold,
        }
    }

    /// The profile so far.
    pub fn profile(&self) -> Profile {
        self.0.borrow().with_heat()
    }

    /// The profile so far, leaving this one empty.
    pub fn take(&self) -> Profile {
        let mut s = self.0.borrow_mut();
        let p = s.with_heat();
        s.profile = Profile::default();
        s.recorded.clear();
        s.counts.clear();
        p
    }
}

/// The interpreter, counting block entries.
pub struct ProfCpu {
    cpu: Box<Cpu>,
    shared: Profiler,
    /// The page the last block started in (most blocks follow one in the same page).
    last: Option<(PageId, Slots)>,
    threshold: u32,
}

impl ProfCpu {
    fn count(&mut self, mem: &mut dyn GuestMemory) {
        let pc = self.cpu.pc();
        // A misaligned PC starts no block: it faults.
        if pc & 3 != 0 {
            return;
        }
        let Some(m) = mem.as_any_mut().and_then(|a| a.downcast_mut::<EngineMem>()) else {
            return;
        };
        let page = (
            m as *const EngineMem as usize,
            pc >> 12,
            m.mem.code_version(pc),
        );
        let slots = match &self.last {
            Some((id, slots)) if *id == page => slots,
            _ => {
                let slots = self
                    .shared
                    .0
                    .borrow_mut()
                    .counts
                    .entry(page)
                    .or_insert_with(|| Rc::new([const { Cell::new(0) }; 1024]))
                    .clone();
                &self.last.insert((page, slots)).1
            }
        };
        let n = &slots[(pc & 0xfff) as usize / 4];
        n.set(n.get() + 1);
        if n.get() == self.threshold
            && let Some(bytes) = m.mem.page_bytes(pc)
        {
            let mut s = self.shared.0.borrow_mut();
            s.profile.record(bytes, (pc & 0xfff) as u16);
            s.recorded.insert(page, page_key(bytes));
        }
    }

    fn run(&mut self, mem: &mut dyn GuestMemory, budget: u64) -> Event {
        let mut left = budget;
        loop {
            if left == 0 {
                break Event::Yield;
            }
            self.count(mem);
            let before = self.cpu.retired();
            // A block that runs into the next page is counted there too: its continuation
            // starts that page's translation, which a profile would
            // otherwise never name. Same instructions either way, in two runs. A misaligned
            // PC in a page's last bytes still gets one instruction's budget, so the
            // interpreter raises its alignment fault instead of the loop never moving.
            let to_page_end = (0x1000 - (self.cpu.pc() & 0xfff)).div_ceil(4);
            let ev = self.cpu.run_block(mem, left.min(to_page_end));
            left -= self.cpu.retired() - before;
            if let Some(ev) = ev {
                break ev;
            }
        }
    }
}

/// Debugger stops: the interpreter's own, uncounted with breakpoints armed (a
/// profile is coverage, never behaviour).
impl DebugCpu for ProfCpu {
    fn run_debug(&mut self, mem: &mut dyn GuestMemory, budget: u64, breaks: &[u64]) -> Event {
        if breaks.is_empty() {
            self.run(mem, budget)
        } else {
            self.cpu.run_debug_raw(mem, budget, breaks)
        }
    }

    fn clear_monitor(&mut self) {
        self.cpu.clear_monitor();
    }
}

impl GuestCpu for ProfCpu {
    fn run_until_event(&mut self, mem: &mut dyn GuestMemory, budget: u64) -> Event {
        let event = self.run(mem, budget);
        self.cpu.clear_monitor();
        event
    }

    fn debug(&mut self) -> Option<&mut dyn DebugCpu> {
        Some(self)
    }

    fn set_syscall_result(&mut self, value: u64) {
        self.cpu.set_syscall_result(value);
    }
    fn pc(&self) -> u64 {
        self.cpu.pc()
    }
    fn set_pc(&mut self, pc: u64) {
        self.cpu.set_pc(pc);
    }
    fn sp(&self) -> u64 {
        self.cpu.sp()
    }
    fn set_sp(&mut self, sp: u64) {
        self.cpu.set_sp(sp);
    }
    fn thread_pointer(&self) -> u64 {
        self.cpu.thread_pointer()
    }
    fn set_thread_pointer(&mut self, tp: u64) {
        self.cpu.set_thread_pointer(tp);
    }
    fn retired(&self) -> u64 {
        self.cpu.retired()
    }
    fn set_virtual_time(&mut self, ns: u64) {
        self.cpu.set_virtual_time(ns);
    }
    fn context(&self) -> CpuContext {
        self.cpu.context()
    }
    fn set_context(&mut self, ctx: &CpuContext) {
        self.cpu.set_context(ctx);
    }
    fn duplicate(&self) -> Box<dyn GuestCpu> {
        Box::new(Self {
            cpu: Box::new((*self.cpu).clone()),
            shared: self.shared.clone(),
            last: None,
            threshold: self.threshold,
        })
    }
}
