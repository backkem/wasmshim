//! The A64 execution engine: the `wasmshim-aarch64` interpreter plus translated code
//! (`wasmshim-aarch64-translate` over `wasmshim-translate`) behind one dispatcher, with a software TLB
//! for translated code.
//!
//! * [`local`]: the engine in-process, as a [`wasmshim_cpu::GuestCpu`] over [`mem::EngineMem`]
//!   address spaces. Translations come **ahead of time**: an object file made by
//!   [`wasmshim_aarch64_translate::aot::object`] is linked into the module ([`aot`]); pages whose content
//!   matches one of its keys run translated. This is what `wasmshim-linux-node` runs.
//! * `remote` (feature `core-module`): the same engine as a core Wasm module with no imports
//!   that a host instantiates and drives through the engine core ABI
//!   ([`wasmshim_cpu_remote::abi`]), with run-time translation compiled by the host. An optional,
//!   development-only tier.
//!
//! Both tiers retire exactly the same instructions and stop at exactly the same points as the
//! interpreter (the translated code checks the retired-instruction budget per block), so which
//! tier ran is invisible to the guest.
//!
//! The crate depends on no OS and no host. The symbols it exports to translated code
//! (`linux_jit_step`, `linux_jit_fill`) and expects from the linked translations
//! (`linux_aot_directory`) carry a `linux_` prefix because the first personality built on it
//! was Linux; they are names, not a dependency on Linux.

#![no_std]

// Per-thread engine state (`state`) on targets that have threads.
#[cfg(not(all(target_arch = "wasm32", not(target_feature = "atomics"))))]
extern crate std;

pub mod aot;
pub mod dispatch;
pub mod jit;
pub mod local;
pub mod mem;
pub mod profile;
#[cfg(feature = "core-module")]
pub mod remote;
mod state;
