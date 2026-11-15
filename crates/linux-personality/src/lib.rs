//! A Linux userspace personality: the Linux *application* ABI implemented as a library.
//!
//! It knows what Linux means (processes, threads, file descriptors, the VFS, memory maps,
//! signals) but not how guest code executes: it sees [`wasmshim_cpu::SyscallFrame`]s and a
//! [`wasmshim_cpu::GuestMemory`], never ISA registers. Execution engines plug in through
//! [`Frontend`]; clocks, entropy and stdio come from the embedder through [`Host`].
//!
//! ```text
//!   Kernel::spawn(path, argv, envp)      load an ELF from the VFS into a fresh address space
//!   loop { match Kernel::run(&mut host) {
//!       RunStatus::Exited(status) => break,
//!       RunStatus::Wait { until }  => advance virtual time to `until`, then run again,
//!   } }
//! ```
//!
//! Everything is deterministic: the scheduler switches threads after a fixed number of
//! retired instructions, and all time and randomness come from the [`Host`].

#![deny(unsafe_code)]

mod arch;
mod checkpoint;
pub mod debug;
pub mod devices;
mod exec;
mod fd;
mod host;
mod ipc;
mod kernel;
mod locks;
mod mm;
pub mod net;
mod objects;
mod pipe;
mod procfs;
mod shm;
mod siginfo;
mod signal;
mod symbols;
mod syscall;
mod sysfs;
mod task;
mod uaccess;
pub mod vfs;

pub use arch::Arch;
pub use checkpoint::{SNAPSHOT_KIND, SNAPSHOT_VERSION, Snapshot};
pub use host::{Blob, BlobError, Frontend, Host};
pub use kernel::{Config, ExitStatus, Kernel, Limits, Pid, RunStatus, Spawn, SpawnError};
pub use mm::SIGTRAMP;
pub use objects::netlink::NetInterface;
pub use objects::tty::{ConsoleDecoder, ConsoleEvent, READING_HINT};
