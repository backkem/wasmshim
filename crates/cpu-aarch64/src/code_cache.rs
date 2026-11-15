//! The decoded-instruction cache: one lazily filled array of decoded instructions per guest
//! code page.
//!
//! * A page entry remembers the page's [`GuestMemory::code_version`] when it was filled. When
//!   the memory's [`GuestMemory::code_epoch`] moves (a write to code, an `mprotect` that adds or
//!   removes execute permission, ...), entries are not dropped: each page is revalidated against
//!   its own version the next time execution enters it, and only pages that really changed are
//!   decoded again. (V8 flips pages between RW and RX thousands of times during startup.)
//! * The slot of a page is found through a small direct-mapped index in front of a `BTreeMap`;
//!   sequential execution inside a page touches neither.
//! * A cached instruction is only ever produced by a successful execute-permission fetch, and
//!   every loss of execute permission changes the page's version, so cached code never runs
//!   from a page that is no longer executable.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use wasmshim_cpu::{Fault, GuestMemory};

use crate::{Inst, decode};

const PAGE_SHIFT: u32 = 12;
/// Instructions per 4 KiB page.
const PAGE_INSNS: usize = 1024;
/// Pages kept before the whole cache is dropped (bounded memory: about 25 MiB).
const MAX_PAGES: usize = 1024;
/// Direct-mapped `vpn -> slot` index in front of the map.
const INDEX_LEN: usize = 256;

struct CodePage {
    vpn: u64,
    /// The page's code version when its instructions were decoded.
    version: u64,
    /// The cache generation at which `version` was last checked.
    checked: u32,
    /// One bit per instruction: decoded yet?
    decoded: [u64; PAGE_INSNS / 64],
    insts: Box<[Inst]>,
}

/// See the module documentation. Cloning yields an empty cache (it is only a cache).
pub(crate) struct CodeCache {
    /// The memory's code epoch when the cache last looked.
    epoch: u64,
    /// Bumped whenever the epoch moved: pages whose `checked` differs must be revalidated.
    generation: u32,
    pages: Vec<CodePage>,
    map: BTreeMap<u64, u32>,
    index: Box<[(u64, u32)]>,
    /// The page execution is in (`vpn`, slot), or `u64::MAX`.
    cur_vpn: u64,
    cur: u32,
}

impl Clone for CodeCache {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl CodeCache {
    pub(crate) fn new() -> Self {
        Self {
            epoch: u64::MAX,
            generation: 0,
            pages: Vec::new(),
            map: BTreeMap::new(),
            index: alloc::vec![(u64::MAX, 0); INDEX_LEN].into_boxed_slice(),
            cur_vpn: u64::MAX,
            cur: 0,
        }
    }

    /// The code epoch the cache last saw.
    #[inline(always)]
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Notices code changes in `mem` since the last call (cheap when there were none).
    #[inline(always)]
    pub(crate) fn sync(&mut self, mem: &dyn GuestMemory) {
        let epoch = mem.code_epoch();
        if epoch != self.epoch {
            self.epoch = epoch;
            self.generation = self.generation.wrapping_add(1);
            self.cur_vpn = u64::MAX;
        }
    }

    /// The decoded instruction at `pc` (4-byte aligned).
    #[inline(always)]
    pub(crate) fn get(&mut self, mem: &mut dyn GuestMemory, pc: u64) -> Result<Inst, Fault> {
        let vpn = pc >> PAGE_SHIFT;
        if vpn != self.cur_vpn {
            self.enter(mem, vpn, pc);
        }
        let i = ((pc >> 2) as usize) & (PAGE_INSNS - 1);
        let page = &mut self.pages[self.cur as usize];
        if page.decoded[i / 64] & (1 << (i % 64)) != 0 {
            return Ok(page.insts[i]);
        }
        let word = mem.fetch(pc).map_err(|fault| Fault::Memory { pc, fault })?;
        let inst = decode(word);
        page.insts[i] = inst;
        page.decoded[i / 64] |= 1 << (i % 64);
        Ok(inst)
    }

    /// Makes `vpn`'s page current: finds or creates its slot and revalidates it if code may
    /// have changed since it was last checked.
    #[cold]
    fn enter(&mut self, mem: &dyn GuestMemory, vpn: u64, pc: u64) {
        let hash = (vpn as usize) & (INDEX_LEN - 1);
        let slot = match self.index[hash] {
            (v, s) if v == vpn => Some(s),
            _ => self.map.get(&vpn).copied(),
        };
        let slot = match slot {
            Some(s) => {
                let page = &mut self.pages[s as usize];
                if page.checked != self.generation {
                    let version = mem.code_version(pc);
                    if version != page.version {
                        page.version = version;
                        page.decoded = [0; PAGE_INSNS / 64];
                    }
                    page.checked = self.generation;
                }
                s
            }
            None => {
                if self.pages.len() >= MAX_PAGES {
                    self.pages.clear();
                    self.map.clear();
                    self.index.fill((u64::MAX, 0));
                }
                let s = self.pages.len() as u32;
                self.pages.push(CodePage {
                    vpn,
                    version: mem.code_version(pc),
                    checked: self.generation,
                    decoded: [0; PAGE_INSNS / 64],
                    insts: alloc::vec![Inst::Undefined; PAGE_INSNS].into_boxed_slice(),
                });
                self.map.insert(vpn, s);
                s
            }
        };
        debug_assert_eq!(self.pages[slot as usize].vpn, vpn);
        self.index[hash] = (vpn, slot);
        self.cur_vpn = vpn;
        self.cur = slot;
    }
}
