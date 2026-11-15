//! The run-time translation tier's host (feature `jit`): the engine's modules
//! are instantiated through `translate:jit`, "instantiate this module against my memory".
//! Browser builds import it; a host that does not serve it declines, and the node
//! interprets what ahead-of-time translation did not cover, exactly as without the feature.

#![allow(unsafe_code)] // generated bindings

wit_bindgen::generate!({ world: "translate:jit/engine", path: "../../crates/translate-kit/wit" });

use translate::jit::jit::{Error, instantiate};

/// `translate:jit` as the engine's [`wasmshim_aarch64_engine::jit::Instantiate`].
struct HostJit {
    /// The host said `unsupported`: stop asking.
    off: bool,
}

impl wasmshim_aarch64_engine::jit::Instantiate for HostJit {
    fn instantiate(&mut self, module: &[u8], funcs: u32) -> Option<u32> {
        if self.off {
            return None;
        }
        match instantiate(module, funcs) {
            Ok(first) => Some(first),
            Err(Error::Unsupported) => {
                self.off = true;
                None
            }
            Err(Error::Invalid(_)) => None,
        }
    }
}

/// Turns the tier on (`threshold` 0: off).
pub fn enable(threshold: u32) {
    if threshold == 0 {
        return;
    }
    wasmshim_aarch64_engine::jit::enable(
        Box::new(HostJit { off: false }),
        wasmshim_aarch64_engine::jit::Policy {
            threshold,
            ..Default::default()
        },
    );
}
