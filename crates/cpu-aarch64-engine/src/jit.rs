//! The run-time translation tier in-process: what ahead-of-time
//! translation did not cover is translated while the guest runs (`wasmshim_translate::tier`) and
//! instantiated by the host through an [`Instantiate`] capability: `translate:jit`
//! for a `wasmshim-linux-node` built with its `jit` feature. Without one the engine never counts
//! blocks and runs the interpreter alone.
//!
//! The tier is invisible: a page's run-time translation is the code an ahead-of-time build of
//! the same page and entries would link, so retired instructions, stops and every guest-visible
//! effect are those of the interpreter. Only wall time changes.

#![allow(unsafe_code)]

extern crate alloc;

use alloc::boxed::Box;

use wasmshim_translate::tier::Tier;
pub use wasmshim_translate::tier::{Instantiate, Policy, Stats};

use crate::mem::EngineMem;
use crate::state::instance_state;

struct Jit {
    tier: Tier,
    host: Box<dyn Instantiate>,
    front: wasmshim_aarch64_translate::FrontHalf,
}

instance_state! {
    /// The tier of this engine instance (one machine per instance).
    fn jit_cell() -> *mut Option<Jit> = None;
}

fn jit() -> Option<&'static mut Jit> {
    // SAFETY: one thread reaches the cell (`state`); the engine never holds this across a call
    // that could re-enter it (translated code calls only `jit_step`/`jit_fill`, which do not
    // touch the tier).
    unsafe { (*jit_cell()).as_mut() }
}

/// Turns the run-time tier on, with `host` instantiating its modules.
pub fn enable(host: Box<dyn Instantiate>, policy: Policy) {
    // SAFETY: as in `jit`.
    unsafe {
        *jit_cell() = Some(Jit {
            tier: Tier::new(policy),
            host,
            front: wasmshim_aarch64_translate::FrontHalf::default(),
        });
    }
}

/// The tier's statistics, if it is on.
pub fn stats() -> Option<Stats> {
    jit().map(|j| j.tier.stats)
}

/// The dispatcher's hot threshold (`0` when the tier is off).
pub(crate) fn threshold() -> u32 {
    jit().map_or(0, |j| j.tier.policy.threshold)
}

/// The block at `pc` of `mem` is hot: translate its page (or take it from the cache) and install
/// what the host instantiated.
pub(crate) fn hot(mem: &mut EngineMem, pc: u64) {
    let Some(j) = jit() else { return };
    let Some((id, bytes, warm)) = mem.hot_page(pc) else {
        return;
    };
    let off = (pc & 0xfff) as u16;
    if let Some(install) = j.tier.hot(&j.front, &mut *j.host, id, &bytes, off, &warm) {
        mem.install_page(id.1, id.2, &install);
    }
}
