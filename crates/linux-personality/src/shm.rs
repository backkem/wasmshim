//! Memory objects: the pages behind every `MAP_SHARED` mapping.
//!
//! A [`MemObject`] wraps a frontend's [`SharedMemory`]: one set of pages that every address
//! space mapping it reads and writes in place. Shared anonymous memory, a regular file's page
//! cache (made when the file is first mapped `MAP_SHARED`; from then on `read`/`write` go
//! through the same pages), memfds, `/dev/shm` files and SysV segments are all objects. Their
//! ids key futex words in them, so processes sharing an object wake each other.

use std::fmt;
use std::rc::Rc;

use wasmshim_cpu::SharedMemory;

/// A memory object.
pub struct MemObject {
    /// Unique per kernel (futex keys; `Kernel::new_object_id`).
    pub id: u64,
    /// The pages.
    pub pages: Rc<dyn SharedMemory>,
    /// For anonymous and SysV objects, the bytes it holds: mapped pages past it raise `SIGBUS`
    /// (a shared anonymous mapping grown by `mremap`). Files have a length instead.
    pub size: Option<u64>,
}

/// A shared handle to a memory object.
pub type MemRef = Rc<MemObject>;

impl fmt::Debug for MemObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemObject({})", self.id)
    }
}

impl PartialEq for MemObject {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for MemObject {}

pub use crate::ipc::{IPC_PRIVATE, IpcPerm};
use crate::ipc::{IpcObject, IpcTable};

/// `shm_perm.mode` bit of a segment marked for removal.
pub const SHM_DEST: u32 = 0o1000;
/// `SHMMNI`: most segments.
pub const SHMMNI: u64 = 4096;
/// `SHMMAX` and `SHMALL` (Linux's defaults: `ULONG_MAX - 2^24`).
pub const SHMMAX: u64 = u64::MAX - (1 << 24);

/// One SysV shared memory segment.
#[derive(Clone, Debug)]
pub struct ShmSegment {
    pub perm: IpcPerm,
    /// The size asked for (mappings round it up to pages).
    pub size: u64,
    pub obj: MemRef,
    pub atime: i64,
    pub dtime: i64,
    pub ctime: i64,
    pub cpid: u32,
    pub lpid: u32,
}

impl IpcObject for ShmSegment {
    fn perm(&self) -> &IpcPerm {
        &self.perm
    }
    fn perm_mut(&mut self) -> &mut IpcPerm {
        &mut self.perm
    }
    fn removed(&self) -> bool {
        self.perm.mode & SHM_DEST != 0
    }
}

/// The SysV shared memory segments.
pub type ShmTable = IpcTable<ShmSegment>;

impl Default for ShmTable {
    fn default() -> Self {
        Self::new(SHMMNI as u32)
    }
}

impl ShmTable {
    /// The segment whose object is `obj_id`.
    pub fn by_object(&self, obj_id: u64) -> Option<u32> {
        self.objs
            .iter()
            .find(|(_, s)| s.obj.id == obj_id)
            .map(|(&i, _)| i)
    }
}
