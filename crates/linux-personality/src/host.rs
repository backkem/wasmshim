//! What the personality needs from its host, and how execution engines plug in: the syscall
//! substrate's host contract (`wasmshim-kernel`), re-exported unchanged.
//!
//! The personality is a pure library: it never reads a clock, an entropy source or a file on its
//! own. Everything nondeterministic or external comes through [`Host`], so the same seed and the
//! same host answers always give the same guest behaviour.

pub use wasmshim_kernel::{Blob, BlobError, Frontend, Host};
