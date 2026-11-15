//! The run-time translation tier: code that ahead-of-time translation did
//! not cover is translated while the guest runs, the way Rosetta 2 pairs install-time
//! translation with a just-in-time translator for what it could not see (generated code, paths
//! no profile took).
//!
//! The engine counts the executions of every block its interpreter runs. When one reaches
//! [`Policy::threshold`], it asks the tier ([`Tier::hot`]); the tier translates the block's page
//! with every hot entry it knows of that page plus the page's warm neighbours (blocks that ran
//! at least [`Policy::warm`] times), in exactly the shape of an ahead-of-time translation (the
//! front half's page translation, [`crate::module::page_module`]), and the host instantiates
//! the module against the engine's memory and table ([`Instantiate`]). The engine then installs
//! the entries ([`Install`]) as it installs ahead-of-time ones, with the page's code version.
//!
//! * **Invisible.** The translation is the same code an ahead-of-time build of the same page
//!   and entries would link, so it retires exactly the instructions the interpreter would and
//!   stops at the same points; only wall time changes. A host that declines costs nothing but
//!   speed.
//! * **The translation cache** is keyed by page content ([`crate::aot::page_key`]): the same
//!   bytes mapped elsewhere (another process, a re-mapped library, a JIT that wrote the same
//!   code again) reuse the compiled functions without a compile.
//! * **Bounded.** A page version is translated at most [`Policy::retranslations`] times, and the
//!   tier stops once the host's table holds [`Policy::max_funcs`] of its functions (a table
//!   never shrinks).
//!
//! The tier is ISA-neutral: a front half supplies [`FrontHalf::page_module`], an engine supplies
//! the page bytes, counts and installs, a host supplies [`Instantiate`].

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::aot::{PageKey, page_key};
use crate::module::PageModule;

/// A host's "instantiate this module against my memory" capability.
pub trait Instantiate {
    /// Compiles `module` (the shape [`crate::module`] documents), instantiates it against the
    /// engine's memory, table and step and fill functions, appends its `funcs` exports `f0`,
    /// `f1`, … to the engine's function table and returns the index of `f0`; `None` if it
    /// declines (no capability, a full table, an invalid module).
    fn instantiate(&mut self, module: &[u8], funcs: u32) -> Option<u32>;
}

/// A front half's page translation as a module.
pub trait FrontHalf {
    /// The module of the page `bytes` with block entries at `entries` (page offsets).
    fn page_module(&self, bytes: &[u8], entries: &[u16]) -> PageModule;
}

/// When and how much to translate.
#[derive(Clone, Copy, Debug)]
pub struct Policy {
    /// Interpreted executions of a block before its page is translated (`0`: never).
    pub threshold: u32,
    /// Interpreted executions that make a block of the same page part of the translation.
    pub warm: u32,
    /// Most translations of one version of one page (each adds entries).
    pub retranslations: u8,
    /// Most functions the tier appends to the host's table.
    pub max_funcs: u32,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            threshold: 0,
            warm: 2,
            retranslations: 8,
            max_funcs: 1 << 20,
        }
    }
}

/// Statistics (never guest-visible).
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Hot blocks the engine reported.
    pub hot: u64,
    /// Pages translated and instantiated.
    pub compiled: u64,
    /// Translations found in the cache (no compile).
    pub cache_hits: u64,
    /// Translations the host declined.
    pub declined: u64,
    /// Hot blocks on pages that reached their retranslation limit, or after `max_funcs`.
    pub refused: u64,
    /// Bytes of Wasm and guest instructions translated.
    pub wasm_bytes: u64,
    pub insns: u64,
    /// Functions appended to the host's table.
    pub funcs: u64,
}

/// What the engine installs: block entries at page offsets, each `(offset, table slot, block
/// index)`, valid for the page's code version the engine read the bytes at.
pub type Install = Vec<(u16, u32, u16)>;

/// A page asked for, waiting for the host (the two-phase form of [`Tier::hot`] for engines that
/// cannot call the host synchronously).
#[derive(Clone, Debug)]
pub struct Request {
    pub module: PageModule,
    key: PageKey,
    entries: Vec<u16>,
}

/// What [`Tier::plan`] decided.
#[derive(Debug)]
pub enum Plan {
    /// Nothing to do (the page is at its limit, or the tier is off).
    None,
    /// A translation from the cache.
    Install(Install),
    /// A module for the host.
    Compile(Request),
}

/// Translations held by the host, by page content.
#[derive(Clone, Debug)]
struct Cached {
    /// The entries translated (sorted).
    entries: Vec<u16>,
    install: Install,
}

/// Per page version: the entries asked for so far and how often it was translated.
#[derive(Clone, Debug, Default)]
struct HotPage {
    entries: Vec<u16>,
    translations: u8,
}

/// The run-time tier of one machine (shared by its CPUs and address spaces).
#[derive(Clone, Debug, Default)]
pub struct Tier {
    pub policy: Policy,
    pub stats: Stats,
    /// By (address space, page address, code version): the engine's identity of a page.
    pages: BTreeMap<(u64, u64, u64), HotPage>,
    cache: BTreeMap<PageKey, Vec<Cached>>,
}

impl Tier {
    /// A tier with `policy`.
    pub fn new(policy: Policy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    /// `true` if the engine should count blocks and report hot ones.
    pub fn enabled(&self) -> bool {
        self.policy.threshold > 0
    }

    /// The block at page offset `offset` of the page `bytes` became hot. `page` identifies the
    /// page version to the engine (address space, page address, code version); `warm` lists the
    /// offsets of other blocks of that page and their interpreted executions. Decides what to
    /// do: install from the cache, compile, or nothing.
    pub fn plan(
        &mut self,
        front: &dyn FrontHalf,
        page: (u64, u64, u64),
        bytes: &[u8],
        offset: u16,
        warm: &[(u16, u32)],
    ) -> Plan {
        self.stats.hot += 1;
        if !self.enabled() || self.stats.funcs >= u64::from(self.policy.max_funcs) {
            self.stats.refused += 1;
            return Plan::None;
        }
        let hot = self.pages.entry(page).or_default();
        if hot.translations >= self.policy.retranslations {
            self.stats.refused += 1;
            return Plan::None;
        }
        let mut wanted = hot.entries.clone();
        wanted.push(offset);
        for &(o, n) in warm {
            if n >= self.policy.warm {
                wanted.push(o);
            }
        }
        wanted.sort_unstable();
        wanted.dedup();
        hot.entries = wanted.clone();
        hot.translations += 1;
        let key = page_key(bytes);
        if let Some(c) = self
            .cache
            .get(&key)
            .and_then(|v| v.iter().find(|c| covers(&c.entries, &wanted)))
        {
            self.stats.cache_hits += 1;
            return Plan::Install(c.install.clone());
        }
        let module = front.page_module(bytes, &wanted);
        Plan::Compile(Request {
            module,
            key,
            entries: wanted,
        })
    }

    /// The host's answer to a [`Plan::Compile`]: the table index of `f0`, or `None` if it
    /// declined. Returns what to install.
    pub fn compiled(&mut self, req: Request, first: Option<u32>) -> Option<Install> {
        let Some(first) = first else {
            self.stats.declined += 1;
            return None;
        };
        self.stats.compiled += 1;
        self.stats.wasm_bytes += req.module.bytes.len() as u64;
        self.stats.insns += req.module.insns as u64;
        self.stats.funcs += u64::from(req.module.funcs);
        let install: Install = req
            .module
            .entries
            .iter()
            .map(|&(off, func, block)| (off, first + func, block))
            .collect();
        self.cache.entry(req.key).or_default().push(Cached {
            entries: req.entries,
            install: install.clone(),
        });
        Some(install)
    }

    /// [`Tier::plan`] and the host call in one step, for engines that can call the host
    /// synchronously.
    pub fn hot(
        &mut self,
        front: &dyn FrontHalf,
        host: &mut dyn Instantiate,
        page: (u64, u64, u64),
        bytes: &[u8],
        offset: u16,
        warm: &[(u16, u32)],
    ) -> Option<Install> {
        match self.plan(front, page, bytes, offset, warm) {
            Plan::None => None,
            Plan::Install(i) => Some(i),
            Plan::Compile(req) => {
                let first = host.instantiate(&req.module.bytes, req.module.funcs);
                self.compiled(req, first)
            }
        }
    }

    /// Forgets the per-page state of pages whose code changed (their translations stay cached
    /// by content: the same bytes may come back).
    pub fn forget(&mut self, keep: impl Fn(&(u64, u64, u64)) -> bool) {
        self.pages.retain(|k, _| keep(k));
    }
}

/// `true` if the sorted `have` contains every element of the sorted `want`.
fn covers(have: &[u16], want: &[u16]) -> bool {
    let mut i = 0;
    for w in want {
        while i < have.len() && have[i] < *w {
            i += 1;
        }
        if i == have.len() || have[i] != *w {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake;
    impl FrontHalf for Fake {
        fn page_module(&self, _bytes: &[u8], entries: &[u16]) -> PageModule {
            PageModule {
                bytes: alloc::vec![0; 8],
                funcs: 1,
                entries: entries
                    .iter()
                    .enumerate()
                    .map(|(i, &o)| (o, 0, i as u16))
                    .collect(),
                insns: entries.len(),
            }
        }
    }
    struct Host(u32);
    impl Instantiate for Host {
        fn instantiate(&mut self, _m: &[u8], funcs: u32) -> Option<u32> {
            let first = self.0;
            self.0 += funcs;
            Some(first)
        }
    }

    #[test]
    fn hot_pages_translate_with_warm_neighbours_and_hit_the_cache() {
        let mut t = Tier::new(Policy {
            threshold: 4,
            ..Policy::default()
        });
        let mut host = Host(100);
        let page = [7u8; 4096];
        let i = t
            .hot(
                &Fake,
                &mut host,
                (1, 0x1000, 0),
                &page,
                0x10,
                &[(0x20, 3), (0x30, 1)],
            )
            .expect("installed");
        assert_eq!(i, [(0x10, 100, 0), (0x20, 100, 1)], "0x30 is not warm");
        // The same content in another address space: from the cache, no compile.
        let i = t
            .hot(&Fake, &mut host, (2, 0x5000, 0), &page, 0x20, &[])
            .expect("cached");
        assert_eq!(i[0].1, 100);
        assert_eq!((t.stats.compiled, t.stats.cache_hits), (1, 1));
        // A new hot entry retranslates with every entry so far.
        let i = t
            .hot(&Fake, &mut host, (1, 0x1000, 0), &page, 0x40, &[])
            .expect("retranslated");
        assert_eq!(i.len(), 3);
        assert_eq!(i[0].1, 101);
    }

    #[test]
    fn pages_stop_at_their_limit_and_declines_cost_nothing() {
        let mut t = Tier::new(Policy {
            threshold: 1,
            retranslations: 1,
            ..Policy::default()
        });
        struct No;
        impl Instantiate for No {
            fn instantiate(&mut self, _m: &[u8], _f: u32) -> Option<u32> {
                None
            }
        }
        let page = [1u8; 4096];
        assert!(t.hot(&Fake, &mut No, (1, 0, 0), &page, 0, &[]).is_none());
        assert_eq!(t.stats.declined, 1);
        assert!(t.hot(&Fake, &mut No, (1, 0, 0), &page, 4, &[]).is_none());
        assert_eq!(t.stats.refused, 1);
        assert!(covers(&[1, 2, 3], &[1, 3]));
        assert!(!covers(&[1, 3], &[2]));
    }
}
