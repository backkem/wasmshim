//! What a personality needs from its host, and how execution engines plug in.
//!
//! The substrate and its personalities are pure libraries: they never read a clock, an entropy
//! source or a file on their own. Everything nondeterministic or external comes through
//! [`Host`], so the same seed and the same host answers always give the same guest behaviour.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;

use wasmshim_cpu::{GuestCpu, GuestMemory, PageSource, PageSourceError, SharedMemory, SharedPages};

/// Services the embedder provides (in `wasmshim-linux-node`: WASI clocks/random/stdio).
pub trait Host {
    /// Monotonic time in nanoseconds (virtual under a deterministic host).
    fn monotonic_ns(&mut self) -> u64;
    /// Wall-clock time in nanoseconds since the Unix epoch.
    fn realtime_ns(&mut self) -> u64;
    /// Fills `buf` with (seeded) random bytes.
    fn random(&mut self, buf: &mut [u8]);
    /// Writes guest output to the host's stdout.
    fn stdout(&mut self, data: &[u8]);
    /// Writes guest output to the host's stderr.
    fn stderr(&mut self, data: &[u8]);
    /// Reads from the host's stdin into `buf`; returns 0 at end of input.
    fn stdin(&mut self, buf: &mut [u8]) -> usize;
    /// A diagnostic line from the personality itself (syscall trace, fatal signals). Never
    /// mixed into the guest's own stdout/stderr.
    fn diag(&mut self, line: &str);

    /// The embedder's debugger at an observer stop ([`crate::debug`]): the machine stands
    /// still until it answers. A stop that does not hold (`stop.hold == false`) reports a
    /// probe's hit for its action and is answered at once. The default resumes at once and
    /// changes nothing: an embedder without a debugger is never stopped.
    fn stop(
        &mut self,
        stop: &crate::debug::Stop,
        at: &mut dyn crate::debug::StopTarget,
    ) -> crate::debug::Resume {
        let _ = (stop, at);
        crate::debug::Resume::default()
    }
}
/// Read-only random access to a blob (a root filesystem image).
pub trait Blob {
    /// Total size in bytes.
    fn len(&self) -> u64;
    /// `true` if the blob is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Fills `buf` from `offset`. Fails if the range leaves the blob or the host errs.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), BlobError>;
}

/// A blob read failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobError {
    /// The range is outside the blob.
    OutOfRange,
    /// The host could not serve the read.
    Host,
}

impl core::fmt::Display for BlobError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::OutOfRange => "read outside the blob",
            Self::Host => "the host failed to read the blob",
        })
    }
}

impl core::error::Error for BlobError {}

/// An execution frontend: how guest code runs. A personality never names an ISA backend;
/// the embedder injects one (e.g. the `wasmshim-aarch64` interpreter with `PagedMemory`).
pub trait Frontend {
    /// A fresh CPU with every register zero.
    fn new_cpu(&self) -> Box<dyn GuestCpu>;
    /// The engine's counters, by name (`stat:engine`): which tier retired how
    /// much, translations, slow paths. Read-only and never guest-visible; the default has none.
    fn counters(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }
    /// A fresh, empty address space.
    fn new_memory(&self) -> Box<dyn GuestMemory>;
    /// A new memory object that this frontend's address spaces can map shared
    /// ([`GuestMemory::map_shared`]), whose pages start as `source`'s (`(source, pages)`: the
    /// first `pages` pages of a pure page source; zeros past them). The default makes the
    /// [`SharedPages`] that `wasmshim_cpu::PagedMemory` (and the engines built on it) maps.
    fn new_shared_memory(
        &self,
        source: Option<(Rc<dyn PageSource>, u64)>,
    ) -> Result<Rc<dyn SharedMemory>, PageSourceError> {
        Ok(Rc::new(SharedPages::new(source.map(|(s, _)| s))))
    }
}
