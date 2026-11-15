//! `wasmshim-translate`: the ISA-neutral half of a guest-code → WebAssembly translator.
//!
//! A translator is a **front half** per guest ISA, next to its CPU (`wasmshim-aarch64-translate` for
//! AArch64), over this kit:
//!
//! * [`ir`]: the IR: pure operations, guest-state slots, guest memory with
//!   permissions, reference-interpreter steps, helpers and exits; [`ir::build`] writes it the way
//!   Wasm is written; [`opt`] holds its passes.
//! * [`region`]: blocks and regions (page regions and loop regions), over an [`Isa`] the front
//!   half describes; [`nest`] finds a page's single-entry loop nests.
//! * [`abi`]: the contract between translated code and the engine that runs it (context block,
//!   exit codes, software TLB, chain table).
//! * [`emit`]: what every back half returns (function bodies with their call sites).
//! * [`aot`]: translations keyed by page content; the directory an engine looks pages up in; the
//!   relocatable object that links translations ahead of time.
//! * [`module`]: the same translations as standalone modules, for the run-time tier.
//! * [`tier`]: the run-time translation tier: hot untranslated code found by the engine is
//!   translated as it runs and instantiated through the host's "instantiate this module against
//!   my memory" capability, with a content-keyed translation cache.
//!
//! Nothing here knows a guest ISA, an operating system or a host. Every tier retires exactly the
//! instructions the reference interpreter would: which tier ran is invisible to the guest.

#![no_std]
#![deny(unsafe_code)]

extern crate alloc;

pub mod abi;
pub mod aot;
pub mod emit;
pub mod ir;
pub mod locals;
pub mod module;
pub mod nest;
pub mod opt;
pub mod region;
pub mod tier;
#[cfg(test)]
mod toy;

pub use region::{Block, Isa, Item, PAGE, Region};
