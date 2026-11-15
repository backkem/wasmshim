//! SysV IPC ids (shared memory, semaphores, message queues): one table
//! per kind, Linux's id scheme (`seq * 32768 + slot`), keys and the `ipc64_perm` block.

use std::collections::BTreeMap;

/// `IPC_PRIVATE`.
pub const IPC_PRIVATE: i32 = 0;
/// Distance between successive ids of one slot (Linux's `IPCMNI`).
pub const IPC_SEQ_MULTIPLIER: u32 = 32768;

/// The permission block of an object (`struct ipc64_perm`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IpcPerm {
    pub key: i32,
    pub uid: u32,
    pub gid: u32,
    pub cuid: u32,
    pub cgid: u32,
    pub mode: u32,
    pub seq: u16,
}

impl IpcPerm {
    /// A new object's block: created by `uid`/`gid` with the low 9 bits of `flags` as mode.
    pub fn new(key: i32, uid: u32, gid: u32, flags: u64) -> Self {
        Self {
            key,
            uid,
            gid,
            cuid: uid,
            cgid: gid,
            mode: (flags & 0o777) as u32,
            seq: 0,
        }
    }

    /// `struct ipc64_perm` (48 bytes).
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(48);
        for v in [
            self.key as u32,
            self.uid,
            self.gid,
            self.cuid,
            self.cgid,
            self.mode,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&self.seq.to_le_bytes());
        b.resize(48, 0);
        b
    }

    /// Applies an `IPC_SET` of `uid`, `gid` and the mode bits read from a guest `ipc64_perm`.
    pub fn set_from(&mut self, raw: &[u8]) {
        let word = |i: usize| {
            raw.get(i..i + 4)
                .map_or(0, |w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        };
        self.uid = word(4);
        self.gid = word(8);
        self.mode = (self.mode & !0o777) | (word(20) & 0o777);
    }
}

/// An object in an [`IpcTable`].
pub trait IpcObject {
    fn perm(&self) -> &IpcPerm;
    fn perm_mut(&mut self) -> &mut IpcPerm;
    /// Removed (`IPC_RMID`) but still alive (only shared memory lingers while attached).
    fn removed(&self) -> bool {
        false
    }
}

/// The objects of one kind (one IPC namespace per kernel).
#[derive(Debug)]
pub struct IpcTable<T> {
    /// Objects by slot.
    pub objs: BTreeMap<u32, T>,
    /// Bumped for every new object, as Linux's per-table sequence.
    pub next_seq: u16,
    /// Most objects.
    pub max: u32,
}

impl<T> IpcTable<T> {
    /// An empty table of at most `max` objects.
    pub fn new(max: u32) -> Self {
        Self {
            objs: BTreeMap::new(),
            next_seq: 0,
            max,
        }
    }
}

impl<T: IpcObject> IpcTable<T> {
    /// The id of slot `index`.
    pub fn id_of(&self, index: u32) -> Option<i32> {
        let o = self.objs.get(&index)?;
        i32::try_from(u32::from(o.perm().seq) * IPC_SEQ_MULTIPLIER + index).ok()
    }

    /// The slot of `id`, if it names a live object.
    pub fn index_of(&self, id: i32) -> Option<u32> {
        let id = u32::try_from(id).ok()?;
        let index = id % IPC_SEQ_MULTIPLIER;
        let o = self.objs.get(&index)?;
        (u32::from(o.perm().seq) == id / IPC_SEQ_MULTIPLIER).then_some(index)
    }

    /// The slot of the object with `key` (never `IPC_PRIVATE`), unless removed.
    pub fn find_key(&self, key: i32) -> Option<u32> {
        self.objs
            .iter()
            .find(|(_, o)| o.perm().key == key && !o.removed())
            .map(|(&i, _)| i)
    }

    /// Adds `obj` in the lowest free slot with a fresh sequence number; its id.
    pub fn insert(&mut self, mut obj: T) -> Option<i32> {
        let index = (0..self.max).find(|i| !self.objs.contains_key(i))?;
        self.next_seq = self.next_seq.wrapping_add(1);
        obj.perm_mut().seq = self.next_seq;
        self.objs.insert(index, obj);
        self.id_of(index)
    }

    /// Highest slot in use (`IPC_INFO` and friends), 0 with none.
    pub fn max_index(&self) -> u64 {
        self.objs.keys().next_back().map_or(0, |&i| u64::from(i))
    }
}
