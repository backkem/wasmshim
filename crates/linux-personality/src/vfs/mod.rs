//! The virtual filesystem: one in-memory inode tree whose initial contents come from a
//! read-only tar image.
//!
//! Regular files from the image keep their bytes in the blob (`FileData::Image`) and are read
//! lazily. The first write copies a file into memory (`FileData::Mem`), so the image itself is
//! never modified; that is the writable overlay, without a second tree.
//! Everything the guest creates lives in memory and disappears with the instance.
//!
//! The guest runs as root, so permission bits are recorded but not enforced, except that
//! `execve` needs an execute bit.

pub(crate) mod ckpt;
pub mod index;
mod pages;
pub mod tar;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::rc::Rc;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{dtype, mode, seal};
use wasmshim_linux_abi::structs::{Stat, Timespec};

use crate::host::Blob;
pub use pages::FileContents;
use pages::SharedBlob;

/// An inode number.
pub type Ino = u64;

/// The root directory's inode.
pub const ROOT: Ino = 1;
/// Maximum symlinks followed while resolving one path (Linux: 40).
const MAX_SYMLINKS: usize = 40;
/// Maximum length of one path component.
const NAME_MAX: usize = 255;
/// Maximum length of a path.
pub const PATH_MAX: usize = 4096;
/// Limits on file data ([`crate::Limits`]'s file fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataLimits {
    /// The largest regular file.
    pub file_bytes: u64,
    /// The most bytes of written files held in memory.
    pub total_bytes: u64,
}

impl Default for DataLimits {
    fn default() -> Self {
        let limits = crate::Limits::default();
        Self {
            file_bytes: limits.file_bytes,
            total_bytes: limits.total_file_bytes,
        }
    }
}
/// `st_dev` for every file (one filesystem).
const DEV: u64 = 0x2a;

/// Character devices the VFS implements itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// `/dev/null`
    Null,
    /// `/dev/zero`
    Zero,
    /// `/dev/random`, `/dev/urandom` (seeded by the host).
    Random,
    /// `/dev/full`
    Full,
    /// `/dev/tty`: the opener's controlling terminal.
    Tty,
    /// `/dev/console`: the node's console terminal when one is attached, else the host's
    /// plain stdio.
    Console,
    /// `/dev/ptmx`: each open creates a pseudo-terminal pair and returns its master.
    Ptmx,
    /// `/dev/pts/<n>`: a pseudo-terminal slave.
    Pts(u32),
    /// `/dev/hidraw<n>`: a HID device the embedder attached.
    HidRaw(u32),
    /// `/dev/fb<n>`: a framebuffer the embedder attached.
    Fb(u32),
}

impl Device {
    fn rdev(self) -> (u32, u32) {
        match self {
            Self::Null => (1, 3),
            Self::Zero => (1, 5),
            Self::Full => (1, 7),
            Self::Random => (1, 9),
            Self::Tty => (5, 0),
            Self::Console => (5, 1),
            Self::Ptmx => (5, 2),
            Self::Pts(n) => (136, n),
            // hidraw's major is allocated at boot on Linux; this is a fixed stand-in.
            Self::HidRaw(n) => (241, n),
            Self::Fb(n) => (29, n),
        }
    }
}

/// Where a regular file's bytes are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileData {
    /// Still in the image blob.
    Image { offset: u64, len: u64 },
    /// In memory (created by the guest, or copied up on first write). Shared with the file's
    /// mappings, which keep the contents of the moment they were made (copy on write).
    Mem(Rc<Vec<u8>>),
    /// In a memory object: the file's page cache, made when it was first mapped
    /// `MAP_SHARED` (or for a memfd), so its shared mappings and `read`/`write` see the same
    /// bytes. `len` is the file's length; the object is zero past it.
    Pages { mem: crate::shm::MemRef, len: u64 },
}

/// What an inode is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    /// A directory: names to inodes (without `.` and `..`).
    Dir(BTreeMap<Vec<u8>, Ino>),
    File(FileData),
    Symlink(Vec<u8>),
    Device(Device),
    /// A FIFO or socket node from the image (opening it is not supported yet).
    Special,
    /// An AF_UNIX socket bound to this path (the kernel maps the inode to the socket).
    Socket,
}

/// One inode.
#[derive(Clone, Debug)]
pub struct Inode {
    pub ino: Ino,
    /// File type and permission bits (`st_mode`).
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub nlink: u32,
    pub rdev: (u32, u32),
    pub atime: Timespec,
    pub mtime: Timespec,
    pub ctime: Timespec,
    /// The containing directory (for `..`); the root is its own parent.
    pub parent: Ino,
    /// Open file descriptions referring to this inode (it outlives its last name while open).
    pub opens: u32,
    /// `F_ADD_SEALS` seals; `None` for a file that cannot be sealed (only memfds can).
    pub seals: Option<u32>,
    /// Extended attributes by full name (`user.mime_type`), in name order.
    pub xattrs: BTreeMap<Vec<u8>, Vec<u8>>,
    pub node: Node,
}

impl Inode {
    /// `true` for directories.
    pub fn is_dir(&self) -> bool {
        matches!(self.node, Node::Dir(_))
    }
}

/// A directory entry as `getdents64` reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry {
    pub name: Vec<u8>,
    pub ino: Ino,
    pub dtype: u8,
}

/// The filesystem.
pub struct Vfs {
    inodes: BTreeMap<Ino, Inode>,
    next_ino: Ino,
    blob: Option<SharedBlob>,
    /// Change notifications for inotify, recorded only while someone watches (`Some`).
    events: Option<Vec<FsEvent>>,
    next_cookie: u32,
    /// Page-cache objects made from a file still in the image, by object id: the file's place
    /// in the image (offset, length). A checkpoint saves only the pages that differ from it
    /// (compact states); a restore reads the rest from the image again.
    origins: BTreeMap<u64, (u64, u64)>,
    /// Directories removed (`rmdir`, or replaced by `rename`) that are kept, empty and with no
    /// links, while something may still refer to them: an open descriptor or a working
    /// directory. [`Vfs::reap_dead_dirs`] frees the ones nothing refers to. Not saved: a
    /// restore finds them again (directories with no links).
    dead_dirs: BTreeSet<Ino>,
    limits: DataLimits,
    /// Bytes of written files held in memory: the lengths of the `FileData::Mem` files, kept
    /// up to date wherever one grows, shrinks, appears or goes ([`Vfs::recount_written`] is the
    /// same number counted from scratch).
    written: u64,
}

impl Default for Vfs {
    fn default() -> Self {
        Self::new()
    }
}

fn dtype_of(mode: u32) -> u8 {
    match mode & mode::S_IFMT {
        mode::S_IFDIR => dtype::DT_DIR,
        mode::S_IFREG => dtype::DT_REG,
        mode::S_IFLNK => dtype::DT_LNK,
        mode::S_IFCHR => dtype::DT_CHR,
        mode::S_IFBLK => dtype::DT_BLK,
        mode::S_IFIFO => dtype::DT_FIFO,
        mode::S_IFSOCK => dtype::DT_SOCK,
        _ => dtype::DT_UNKNOWN,
    }
}

/// Splits a path into components, dropping empty ones.
fn components(path: &[u8]) -> impl DoubleEndedIterator<Item = &[u8]> {
    path.split(|&b| b == b'/').filter(|c| !c.is_empty())
}

impl Vfs {
    /// A filesystem with an empty root, `/dev` (null, zero, full, random, urandom, tty,
    /// console) and a sticky `/tmp`.
    pub fn new() -> Self {
        let mut vfs = Self {
            inodes: BTreeMap::new(),
            next_ino: ROOT,
            blob: None,
            events: None,
            next_cookie: 0,
            origins: BTreeMap::new(),
            dead_dirs: BTreeSet::new(),
            limits: DataLimits::default(),
            written: 0,
        };
        let epoch = Timespec::default();
        let root = vfs.alloc(
            mode::S_IFDIR | 0o755,
            ROOT,
            Node::Dir(BTreeMap::new()),
            epoch,
        );
        debug_assert_eq!(root, ROOT);
        vfs.inode_mut(ROOT).nlink = 2;
        let dev = vfs.mkdir_at(ROOT, b"dev", 0o755, epoch).unwrap_or(ROOT);
        for (name, d) in [
            (&b"null"[..], Device::Null),
            (b"zero", Device::Zero),
            (b"full", Device::Full),
            (b"random", Device::Random),
            (b"urandom", Device::Random),
            (b"tty", Device::Tty),
            (b"console", Device::Console),
            (b"ptmx", Device::Ptmx),
        ] {
            let perm = if d == Device::Console { 0o600 } else { 0o666 };
            let ino = vfs.alloc(mode::S_IFCHR | perm, dev, Node::Device(d), epoch);
            vfs.inode_mut(ino).rdev = d.rdev();
            vfs.link_into(dev, name, ino);
        }
        // devpts: slaves appear here while their pty exists.
        let _ = vfs.mkdir_at(dev, b"pts", 0o755, epoch);
        // As udev/devtmpfs set them up: the stdio aliases (`curl -D /dev/stderr`), `/dev/fd`,
        // and a sticky `/dev/shm` (POSIX shared memory lives there).
        for (name, target) in [
            (&b"stdin"[..], &b"/proc/self/fd/0"[..]),
            (b"stdout", b"/proc/self/fd/1"),
            (b"stderr", b"/proc/self/fd/2"),
            (b"fd", b"/proc/self/fd"),
        ] {
            let _ = vfs.symlink(dev, name, target, epoch);
        }
        let _ = vfs.mkdir_at(dev, b"shm", 0o1777, epoch);
        let _ = vfs.mkdir_at(ROOT, b"tmp", 0o1777, epoch);
        vfs
    }

    fn alloc(&mut self, mode: u32, parent: Ino, node: Node, now: Timespec) -> Ino {
        let ino = self.next_ino;
        self.next_ino += 1;
        let nlink = if matches!(node, Node::Dir(_)) { 2 } else { 1 };
        let rdev = (0, 0);
        self.inodes.insert(
            ino,
            Inode {
                ino,
                mode,
                uid: 0,
                gid: 0,
                nlink,
                rdev,
                atime: now,
                mtime: now,
                ctime: now,
                parent,
                opens: 0,
                seals: None,
                xattrs: BTreeMap::new(),
                node,
            },
        );
        ino
    }

    /// Whether `ino` is a directory (`false` for a number that names nothing).
    pub fn is_dir(&self, ino: Ino) -> bool {
        self.inodes.get(&ino).is_some_and(Inode::is_dir)
    }

    /// The inode `ino`. Panics only on a dangling number, which would be a VFS bug.
    pub(crate) fn inode(&self, ino: Ino) -> &Inode {
        &self.inodes[&ino]
    }

    fn inode_mut(&mut self, ino: Ino) -> &mut Inode {
        self.inodes.get_mut(&ino).expect("dangling inode number")
    }

    fn dir_mut(&mut self, dir: Ino) -> Result<&mut BTreeMap<Vec<u8>, Ino>, Errno> {
        match &mut self.inode_mut(dir).node {
            Node::Dir(entries) => Ok(entries),
            _ => Err(Errno::ENOTDIR),
        }
    }

    fn link_into(&mut self, dir: Ino, name: &[u8], ino: Ino) {
        if let Ok(entries) = self.dir_mut(dir) {
            entries.insert(name.to_vec(), ino);
        }
        if self.inode(ino).is_dir() {
            self.inode_mut(ino).parent = dir;
            self.inode_mut(dir).nlink += 1;
        }
    }

    /// Looks `name` up in directory `dir`.
    pub fn lookup(&self, dir: Ino, name: &[u8]) -> Result<Ino, Errno> {
        match &self.inode(dir).node {
            Node::Dir(entries) => match name {
                b"." => Ok(dir),
                b".." => Ok(self.inode(dir).parent),
                _ => entries.get(name).copied().ok_or(Errno::ENOENT),
            },
            _ => Err(Errno::ENOTDIR),
        }
    }

    /// Resolves `path` relative to `cwd`. Symlinks in the middle are always followed; the
    /// last component's symlink only if `follow`. A symlink into `/proc` (which is not in the
    /// VFS) ends in `ENOENT` here; see [`Vfs::resolve_escape`].
    pub fn resolve(&self, cwd: Ino, path: &[u8], follow: bool) -> Result<Ino, Errno> {
        match self.resolve_escape(cwd, path, follow)? {
            Resolved::Ino(ino) => Ok(ino),
            Resolved::Escape(_) => Err(Errno::ENOENT),
        }
    }

    /// [`Vfs::resolve`], but a followed symlink whose target is under `/proc` (generated at
    /// the path layer, outside the VFS: `/dev/stdout`, Docker-style `access.log -> /dev/stdout`
    /// chains end there) returns the rest of the walk as an absolute path to continue with.
    pub fn resolve_escape(&self, cwd: Ino, path: &[u8], follow: bool) -> Result<Resolved, Errno> {
        if path.is_empty() {
            return Err(Errno::ENOENT);
        }
        if path.len() >= PATH_MAX {
            return Err(Errno::ENAMETOOLONG);
        }
        let trailing_slash = path.ends_with(b"/");
        let mut cur = if path[0] == b'/' { ROOT } else { cwd };
        let mut todo: VecDeque<Vec<u8>> = components(path).map(<[u8]>::to_vec).collect();
        let mut links = 0;
        while let Some(name) = todo.pop_front() {
            if name.len() > NAME_MAX {
                return Err(Errno::ENAMETOOLONG);
            }
            let next = self.lookup(cur, &name)?;
            let is_last = todo.is_empty();
            if let Node::Symlink(target) = &self.inode(next).node
                && (!is_last || follow || trailing_slash)
            {
                links += 1;
                if links > MAX_SYMLINKS {
                    return Err(Errno::ELOOP);
                }
                if target == b"/proc" || target.starts_with(b"/proc/") {
                    let mut rest = target.clone();
                    for c in &todo {
                        rest.push(b'/');
                        rest.extend_from_slice(c);
                    }
                    if trailing_slash {
                        rest.push(b'/');
                    }
                    return Ok(Resolved::Escape(rest));
                }
                if target.first() == Some(&b'/') {
                    cur = ROOT;
                }
                for c in components(target).rev() {
                    todo.push_front(c.to_vec());
                }
                continue;
            }
            cur = next;
        }
        if trailing_slash && !self.inode(cur).is_dir() {
            return Err(Errno::ENOTDIR);
        }
        Ok(Resolved::Ino(cur))
    }

    /// Resolves everything but the last component: returns the parent directory and the final
    /// name (which may not exist). `ENOENT` for `/` itself or an empty path.
    pub fn resolve_parent(&self, cwd: Ino, path: &[u8]) -> Result<(Ino, Vec<u8>), Errno> {
        let trimmed = {
            let mut p = path;
            while p.len() > 1 && p.ends_with(b"/") {
                p = &p[..p.len() - 1];
            }
            p
        };
        let (dir_path, name) = match trimmed.iter().rposition(|&b| b == b'/') {
            Some(i) => (&trimmed[..i.max(1)], &trimmed[i + 1..]),
            None => (&b"."[..], trimmed),
        };
        if name.is_empty() {
            return Err(Errno::ENOENT);
        }
        if name.len() > NAME_MAX {
            return Err(Errno::ENAMETOOLONG);
        }
        let dir = if trimmed.starts_with(b"/") && dir_path == b"/" {
            ROOT
        } else {
            self.resolve(cwd, dir_path, true)?
        };
        if !self.inode(dir).is_dir() {
            return Err(Errno::ENOTDIR);
        }
        Ok((dir, name.to_vec()))
    }

    /// Mounts a tar image as the initial contents: every member is created (parents too),
    /// later members replace earlier ones. Returns the number of members. An image that starts
    /// with an index ([`index`]) is mounted from it, without reading every header.
    pub fn mount_image(&mut self, mut blob: Box<dyn Blob>) -> Result<usize, tar::TarError> {
        let entries = index::entries(blob.as_mut())?;
        // A hard link names a member archived before it (tar(5)), but archives sorted by path
        // can list a link before its target. Such links are made once the rest is in place,
        // unless a later member took their name.
        let mut later = Vec::new();
        for e in &entries {
            if !self.add_image_entry(e, true) {
                later.push(e);
            }
        }
        while !later.is_empty() {
            let before = later.len();
            later.retain(|e| !self.add_image_entry(e, false));
            if later.len() == before {
                break;
            }
        }
        self.blob = Some(Rc::new(RefCell::new(blob)));
        Ok(entries.len())
    }

    /// Adds one member, replacing what has its name (or, unless `replace`, leaving it).
    /// `false` for a hard link whose target is not there (yet).
    fn add_image_entry(&mut self, e: &tar::Entry, replace: bool) -> bool {
        let now = Timespec {
            sec: e.mtime,
            nsec: 0,
        };
        let parts: Vec<&[u8]> = components(&e.path).filter(|c| *c != b".").collect();
        let Some((name, dirs)) = parts.split_last() else {
            // The root itself ("./"): take its mode.
            if e.kind == tar::EntryKind::Dir {
                self.inode_mut(ROOT).mode = mode::S_IFDIR | e.mode;
            }
            return true;
        };
        if parts.iter().any(|c| *c == b"..") {
            return true;
        }
        let mut dir = ROOT;
        for d in dirs {
            dir = match self.lookup(dir, d) {
                Ok(ino) if self.inode(ino).is_dir() => ino,
                _ => match self.mkdir_at(dir, d, 0o755, now) {
                    Ok(ino) => ino,
                    Err(_) => return true,
                },
            };
        }
        if e.kind == tar::EntryKind::Dir
            && let Ok(existing) = self.lookup(dir, name)
            && self.inode(existing).is_dir()
        {
            let i = self.inode_mut(existing);
            i.mode = mode::S_IFDIR | e.mode;
            i.uid = e.uid;
            i.gid = e.gid;
            i.mtime = now;
            return true;
        }
        if let Ok(old) = self.lookup(dir, name) {
            if !replace {
                return true;
            }
            let _ = self.unlink_any(dir, name);
            // A replaced directory: nothing can refer to it while an image mounts.
            if self.dead_dirs.remove(&old) {
                self.forget(old);
            }
        }
        let ino = match e.kind {
            tar::EntryKind::Dir => match self.mkdir_at(dir, name, e.mode, now) {
                Ok(ino) => ino,
                Err(_) => return true,
            },
            tar::EntryKind::HardLink => {
                let Ok(target) = self.resolve(ROOT, &e.link, false) else {
                    return false;
                };
                if self.inode(target).is_dir() {
                    return true;
                }
                self.link_into(dir, name, target);
                self.inode_mut(target).nlink += 1;
                return true;
            }
            tar::EntryKind::File => {
                let ino = self.alloc(
                    mode::S_IFREG | e.mode,
                    dir,
                    Node::File(FileData::Image {
                        offset: e.data_offset,
                        len: e.size,
                    }),
                    now,
                );
                self.link_into(dir, name, ino);
                ino
            }
            tar::EntryKind::Symlink => {
                let ino = self.alloc(
                    mode::S_IFLNK | 0o777,
                    dir,
                    Node::Symlink(e.link.clone()),
                    now,
                );
                self.link_into(dir, name, ino);
                ino
            }
            tar::EntryKind::CharDev | tar::EntryKind::BlockDev | tar::EntryKind::Fifo => {
                let ty = match e.kind {
                    tar::EntryKind::CharDev => mode::S_IFCHR,
                    tar::EntryKind::BlockDev => mode::S_IFBLK,
                    _ => mode::S_IFIFO,
                };
                let ino = self.alloc(ty | e.mode, dir, Node::Special, now);
                self.inode_mut(ino).rdev = (e.dev_major, e.dev_minor);
                self.link_into(dir, name, ino);
                ino
            }
        };
        let i = self.inode_mut(ino);
        i.uid = e.uid;
        i.gid = e.gid;
        true
    }

    fn mkdir_at(&mut self, dir: Ino, name: &[u8], perm: u32, now: Timespec) -> Result<Ino, Errno> {
        if self.lookup(dir, name).is_ok() {
            return Err(Errno::EEXIST);
        }
        let ino = self.alloc(
            mode::S_IFDIR | (perm & 0o7777),
            dir,
            Node::Dir(BTreeMap::new()),
            now,
        );
        self.link_into(dir, name, ino);
        Ok(ino)
    }

    /// `mkdir`.
    pub fn mkdir(&mut self, dir: Ino, name: &[u8], perm: u32, now: Timespec) -> Result<Ino, Errno> {
        if name == b"." || name == b".." {
            return Err(Errno::EEXIST);
        }
        self.dir_mut(dir)?;
        self.check_live(dir)?;
        let ino = self.mkdir_at(dir, name, perm, now)?;
        self.touch(dir, now);
        self.note_child(dir, name, ino, fsmask::IN_CREATE, 0);
        Ok(ino)
    }

    /// Where an `O_CREAT` open of `name` in `dir` creates its file when `name` does not
    /// resolve: `name` itself if it does not exist, else the missing name a chain of dangling
    /// symlinks ends at (Linux follows a trailing symlink to create; up to [`MAX_SYMLINKS`]).
    pub(crate) fn creation_target(
        &self,
        mut dir: Ino,
        mut name: Vec<u8>,
    ) -> Result<(Ino, Vec<u8>), Errno> {
        for _ in 0..=MAX_SYMLINKS {
            let ino = match self.lookup(dir, &name) {
                Err(Errno::ENOENT) => return Ok((dir, name)),
                Err(e) => return Err(e),
                Ok(ino) => ino,
            };
            let Node::Symlink(target) = &self.inode(ino).node else {
                return Err(Errno::ENOENT);
            };
            (dir, name) = self.resolve_parent(dir, target)?;
        }
        Err(Errno::ELOOP)
    }

    /// Creates an empty regular file (`O_CREAT`).
    pub fn create(
        &mut self,
        dir: Ino,
        name: &[u8],
        perm: u32,
        now: Timespec,
    ) -> Result<Ino, Errno> {
        if name == b"." || name == b".." {
            return Err(Errno::EISDIR);
        }
        self.dir_mut(dir)?;
        self.check_live(dir)?;
        if self.lookup(dir, name).is_ok() {
            return Err(Errno::EEXIST);
        }
        let ino = self.alloc(
            mode::S_IFREG | (perm & 0o7777),
            dir,
            Node::File(FileData::Mem(Rc::default())),
            now,
        );
        self.link_into(dir, name, ino);
        self.touch(dir, now);
        self.note_child(dir, name, ino, fsmask::IN_CREATE, 0);
        Ok(ino)
    }

    /// A regular file with no name (a memfd): it lives while open file descriptions refer to
    /// it ([`Vfs::retain`]), with `data` as its contents.
    pub fn create_unlinked(&mut self, perm: u32, data: FileData, now: Timespec) -> Ino {
        if let FileData::Mem(v) = &data {
            self.written += v.len() as u64;
        }
        let ino = self.alloc(mode::S_IFREG | (perm & 0o7777), ROOT, Node::File(data), now);
        self.inode_mut(ino).nlink = 0;
        ino
    }

    /// Sets the seals of `ino` (see [`Inode::seals`]).
    pub fn set_seals(&mut self, ino: Ino, seals: Option<u32>) {
        self.inode_mut(ino).seals = seals;
    }

    /// Creates the inode a `bind` to a path makes.
    pub fn mksock(
        &mut self,
        dir: Ino,
        name: &[u8],
        perm: u32,
        now: Timespec,
    ) -> Result<Ino, Errno> {
        self.dir_mut(dir)?;
        self.check_live(dir)?;
        if self.lookup(dir, name).is_ok() {
            return Err(Errno::EEXIST);
        }
        let ino = self.alloc(mode::S_IFSOCK | (perm & 0o7777), dir, Node::Socket, now);
        self.link_into(dir, name, ino);
        self.touch(dir, now);
        self.note_child(dir, name, ino, fsmask::IN_CREATE, 0);
        Ok(ino)
    }

    /// `mknod` of a FIFO: a node like a FIFO from the image (opening it is not supported).
    pub(crate) fn mkfifo(
        &mut self,
        dir: Ino,
        name: &[u8],
        perm: u32,
        now: Timespec,
    ) -> Result<Ino, Errno> {
        self.dir_mut(dir)?;
        self.check_live(dir)?;
        if self.lookup(dir, name).is_ok() {
            return Err(Errno::EEXIST);
        }
        let ino = self.alloc(mode::S_IFIFO | (perm & 0o7777), dir, Node::Special, now);
        self.link_into(dir, name, ino);
        self.touch(dir, now);
        self.note_child(dir, name, ino, fsmask::IN_CREATE, 0);
        Ok(ino)
    }

    /// `symlink`.
    pub fn symlink(
        &mut self,
        dir: Ino,
        name: &[u8],
        target: &[u8],
        now: Timespec,
    ) -> Result<Ino, Errno> {
        self.dir_mut(dir)?;
        self.check_live(dir)?;
        if self.lookup(dir, name).is_ok() {
            return Err(Errno::EEXIST);
        }
        let ino = self.alloc(
            mode::S_IFLNK | 0o777,
            dir,
            Node::Symlink(target.to_vec()),
            now,
        );
        self.link_into(dir, name, ino);
        self.touch(dir, now);
        self.note_child(dir, name, ino, fsmask::IN_CREATE, 0);
        Ok(ino)
    }

    /// `link`: a new name for a non-directory.
    pub fn link(&mut self, ino: Ino, dir: Ino, name: &[u8], now: Timespec) -> Result<(), Errno> {
        if self.inode(ino).is_dir() {
            return Err(Errno::EPERM);
        }
        self.dir_mut(dir)?;
        self.check_live(dir)?;
        if self.lookup(dir, name).is_ok() {
            return Err(Errno::EEXIST);
        }
        self.link_into(dir, name, ino);
        let i = self.inode_mut(ino);
        i.nlink += 1;
        i.ctime = now;
        self.touch(dir, now);
        self.note_child(dir, name, ino, fsmask::IN_CREATE, 0);
        self.note(ino, fsmask::IN_ATTRIB);
        Ok(())
    }

    /// `unlink` (non-directories) or `rmdir` (empty directories), as selected by `want_dir`.
    pub fn remove(
        &mut self,
        dir: Ino,
        name: &[u8],
        want_dir: bool,
        now: Timespec,
    ) -> Result<(), Errno> {
        if name == b"." {
            return Err(Errno::EINVAL);
        }
        if name == b".." {
            return Err(Errno::ENOTEMPTY);
        }
        let ino = self.lookup(dir, name)?;
        let is_dir = self.inode(ino).is_dir();
        match (want_dir, is_dir) {
            (true, false) => return Err(Errno::ENOTDIR),
            (false, true) => return Err(Errno::EISDIR),
            (true, true) => {
                if matches!(&self.inode(ino).node, Node::Dir(e) if !e.is_empty()) {
                    return Err(Errno::ENOTEMPTY);
                }
            }
            (false, false) => {}
        }
        let gone = is_dir || self.inode(ino).nlink <= 1;
        // Linux's order: the link count (IN_ATTRIB) for a file, the parent's IN_DELETE, then
        // IN_DELETE_SELF once the inode has no names left.
        if !is_dir {
            self.note(ino, fsmask::IN_ATTRIB);
        }
        self.note_child(dir, name, ino, fsmask::IN_DELETE, 0);
        self.unlink_any(dir, name)?;
        self.touch(dir, now);
        if gone {
            self.note(ino, fsmask::IN_DELETE_SELF);
        }
        Ok(())
    }

    fn unlink_any(&mut self, dir: Ino, name: &[u8]) -> Result<(), Errno> {
        let ino = self.dir_mut(dir)?.remove(name).ok_or(Errno::ENOENT)?;
        let is_dir = self.inode(ino).is_dir();
        if is_dir {
            self.inode_mut(dir).nlink -= 1;
            // Drop the subtree's inodes (only reachable through this name; `rmdir` and
            // `rename` only remove empty directories, a mounting image may replace a full one).
            let entries = match &mut self.inode_mut(ino).node {
                Node::Dir(entries) => std::mem::take(entries),
                _ => BTreeMap::new(),
            };
            let mut stack: Vec<Ino> = entries
                .values()
                .copied()
                .filter(|&c| self.inode(c).is_dir() || self.inode(c).nlink <= 1)
                .collect();
            while let Some(i) = stack.pop() {
                if let Some(inode) = self.inodes.get(&i)
                    && let Node::Dir(entries) = &inode.node
                {
                    stack.extend(
                        entries
                            .values()
                            .copied()
                            .filter(|&c| self.inode(c).is_dir() || self.inode(c).nlink <= 1),
                    );
                }
                self.forget(i);
            }
            // The directory itself stays, empty and with no links, while a descriptor or a
            // working directory may still refer to it (rmdir(2) does not wait for them); see
            // `reap_dead_dirs`.
            self.inode_mut(ino).nlink = 0;
            self.dead_dirs.insert(ino);
        } else {
            let i = self.inode_mut(ino);
            i.nlink = i.nlink.saturating_sub(1);
            // An inode with no names left is freed now, or when its last open file
            // description closes (see `release`).
            if i.nlink == 0 && i.opens == 0 {
                self.forget(ino);
            }
        }
        Ok(())
    }

    /// devpts: creates `/dev/pts/<n>` (mode 0620, as `grantpt` would leave it).
    pub fn add_pts(&mut self, n: u32, now: Timespec) -> Result<Ino, Errno> {
        let dir = self.resolve(ROOT, b"/dev/pts", true)?;
        let name = n.to_string().into_bytes();
        if self.lookup(dir, &name).is_ok() {
            return Err(Errno::EEXIST);
        }
        let d = Device::Pts(n);
        let ino = self.alloc(mode::S_IFCHR | 0o620, dir, Node::Device(d), now);
        self.inode_mut(ino).rdev = d.rdev();
        self.link_into(dir, &name, ino);
        Ok(ino)
    }

    /// Adds a device node `/dev/<name>` (an embedder attaching hardware: hidraw, fb).
    pub fn add_device(&mut self, name: &[u8], d: Device, perm: u32) {
        let Ok(dir) = self.resolve(ROOT, b"/dev", true) else {
            return;
        };
        if self.lookup(dir, name).is_ok() {
            return;
        }
        let ino = self.alloc(
            mode::S_IFCHR | perm,
            dir,
            Node::Device(d),
            Timespec::default(),
        );
        self.inode_mut(ino).rdev = d.rdev();
        self.link_into(dir, name, ino);
    }

    /// devpts: removes `/dev/pts/<n>` (the pty is gone; open slaves keep the inode).
    pub fn remove_pts(&mut self, n: u32) {
        if let Ok(dir) = self.resolve(ROOT, b"/dev/pts", true) {
            let name = n.to_string().into_bytes();
            if matches!(self.lookup(dir, &name).map(|i| &self.inode(i).node), Ok(Node::Device(Device::Pts(m))) if *m == n)
            {
                let _ = self.unlink_any(dir, &name);
            }
        }
    }

    /// Records one more open file description of `ino`.
    pub fn retain(&mut self, ino: Ino) {
        if let Some(i) = self.inodes.get_mut(&ino) {
            i.opens += 1;
        }
    }

    /// Drops one open file description; frees the inode if it has no names left either.
    pub fn release(&mut self, ino: Ino) {
        if let Some(i) = self.inodes.get_mut(&ino) {
            i.opens = i.opens.saturating_sub(1);
            if i.opens == 0 && i.nlink == 0 && !i.is_dir() {
                self.forget(ino);
            }
        }
    }

    /// `true` if the inode still exists.
    pub fn exists(&self, ino: Ino) -> bool {
        self.inodes.contains_key(&ino)
    }

    /// `true` for a directory that was removed while in use (Linux's `IS_DEADDIR`).
    pub(crate) fn is_dead_dir(&self, ino: Ino) -> bool {
        self.dead_dirs.contains(&ino)
    }

    /// `ENOENT` if `dir` was removed: nothing can be created in a deleted directory.
    fn check_live(&self, dir: Ino) -> Result<(), Errno> {
        if self.is_dead_dir(dir) {
            Err(Errno::ENOENT)
        } else {
            Ok(())
        }
    }

    /// Frees the removed directories nothing refers to any more: no open file description,
    /// and not `in_use` (a working directory). A kept one keeps its removed ancestors too,
    /// since its `..` still leads there.
    pub(crate) fn reap_dead_dirs(&mut self, in_use: impl Fn(Ino) -> bool) {
        if self.dead_dirs.is_empty() {
            return;
        }
        let mut keep = BTreeSet::new();
        for &d in &self.dead_dirs {
            if self.inodes.get(&d).is_some_and(|i| i.opens > 0) || in_use(d) {
                let mut x = d;
                while self.dead_dirs.contains(&x) && keep.insert(x) {
                    x = self.inodes.get(&x).map_or(ROOT, |i| i.parent);
                }
            }
        }
        let gone: Vec<Ino> = self.dead_dirs.difference(&keep).copied().collect();
        for d in gone {
            self.dead_dirs.remove(&d);
            self.forget(d);
        }
    }

    /// Finds the removed directories again after a restore (directories with no links).
    pub(crate) fn find_dead_dirs(&mut self) {
        self.dead_dirs = self
            .inodes
            .values()
            .filter(|i| i.is_dir() && i.nlink == 0)
            .map(|i| i.ino)
            .collect();
    }

    /// `rename`.
    pub fn rename(
        &mut self,
        odir: Ino,
        oname: &[u8],
        ndir: Ino,
        nname: &[u8],
        now: Timespec,
    ) -> Result<(), Errno> {
        let ino = self.lookup(odir, oname)?;
        if matches!(oname, b"." | b"..") || matches!(nname, b"." | b"..") {
            return Err(Errno::EBUSY);
        }
        self.check_live(ndir)?;
        if self.inode(ino).is_dir() {
            // A directory cannot move into its own subtree.
            let mut d = ndir;
            loop {
                if d == ino {
                    return Err(Errno::EINVAL);
                }
                if d == ROOT {
                    break;
                }
                d = self.inode(d).parent;
            }
        }
        if let Ok(existing) = self.lookup(ndir, nname) {
            if existing == ino {
                return Ok(());
            }
            let (src_dir, dst_dir) = (self.inode(ino).is_dir(), self.inode(existing).is_dir());
            if src_dir && !dst_dir {
                return Err(Errno::ENOTDIR);
            }
            if !src_dir && dst_dir {
                return Err(Errno::EISDIR);
            }
            if dst_dir && matches!(&self.inode(existing).node, Node::Dir(e) if !e.is_empty()) {
                return Err(Errno::ENOTEMPTY);
            }
            self.unlink_any(ndir, nname)?;
        }
        self.dir_mut(odir)?.remove(oname);
        if self.inode(ino).is_dir() {
            self.inode_mut(odir).nlink -= 1;
        }
        self.link_into(ndir, nname, ino);
        if self.events.is_some() {
            self.next_cookie = self.next_cookie.wrapping_add(1).max(1);
            let cookie = self.next_cookie;
            self.note_child(odir, oname, ino, fsmask::IN_MOVED_FROM, cookie);
            self.note_child(ndir, nname, ino, fsmask::IN_MOVED_TO, cookie);
            self.note(ino, fsmask::IN_MOVE_SELF);
        }
        self.touch(odir, now);
        self.touch(ndir, now);
        self.inode_mut(ino).ctime = now;
        Ok(())
    }

    fn touch(&mut self, ino: Ino, now: Timespec) {
        let i = self.inode_mut(ino);
        i.mtime = now;
        i.ctime = now;
    }

    /// Symlink target.
    pub fn readlink(&self, ino: Ino) -> Result<&[u8], Errno> {
        match &self.inode(ino).node {
            Node::Symlink(t) => Ok(t),
            _ => Err(Errno::EINVAL),
        }
    }

    /// Size in bytes (`st_size`).
    pub fn size(&self, ino: Ino) -> u64 {
        match &self.inode(ino).node {
            Node::File(FileData::Image { len, .. }) => *len,
            Node::File(FileData::Mem(v)) => v.len() as u64,
            Node::File(FileData::Pages { len, .. }) => *len,
            Node::Symlink(t) => t.len() as u64,
            Node::Dir(_) => 4096,
            _ => 0,
        }
    }

    /// Reads file bytes at `offset`; returns how many were read (0 at or past the end).
    pub fn read(&mut self, ino: Ino, offset: u64, buf: &mut [u8]) -> Result<usize, Errno> {
        let blob = &self.blob;
        match &self.inodes.get(&ino).ok_or(Errno::EBADF)?.node {
            Node::File(FileData::Mem(v)) => {
                let start = usize::try_from(offset).unwrap_or(usize::MAX).min(v.len());
                let n = buf.len().min(v.len() - start);
                buf[..n].copy_from_slice(&v[start..start + n]);
                Ok(n)
            }
            Node::File(FileData::Image { offset: base, len }) => {
                if offset >= *len {
                    return Ok(0);
                }
                let n = buf.len().min((len - offset) as usize);
                let blob = blob.as_ref().ok_or(Errno::EIO)?;
                let mut blob = blob.try_borrow_mut().map_err(|_| Errno::EIO)?;
                blob.read_at(base + offset, &mut buf[..n])
                    .map_err(|_| Errno::EIO)?;
                Ok(n)
            }
            Node::File(FileData::Pages { mem, len }) => {
                if offset >= *len {
                    return Ok(0);
                }
                let n = buf.len().min((len - offset) as usize);
                mem.pages
                    .read(offset, &mut buf[..n])
                    .map_err(|_| Errno::EIO)?;
                Ok(n)
            }
            Node::Dir(_) => Err(Errno::EISDIR),
            _ => Err(Errno::EINVAL),
        }
    }

    /// The contents of regular file `ino` as they are now, for mappings (their private copy:
    /// later writes to the file do not show in them). Nothing is read or copied, except for a
    /// file in a memory object, whose bytes are copied now (the object keeps changing).
    pub fn file_contents(&mut self, ino: Ino) -> Result<FileContents, Errno> {
        match &self.inodes.get(&ino).ok_or(Errno::EBADF)?.node {
            Node::File(FileData::Mem(v)) => Ok(FileContents::Mem(v.clone())),
            Node::File(FileData::Image { offset, len }) => Ok(FileContents::Image {
                blob: self.blob.clone().ok_or(Errno::EIO)?,
                base: *offset,
                len: *len,
            }),
            Node::File(FileData::Pages { .. }) => {
                Ok(FileContents::Mem(Rc::new(self.read_all(ino)?)))
            }
            Node::Dir(_) => Err(Errno::EISDIR),
            _ => Err(Errno::EINVAL),
        }
    }

    /// Every page-cache object (a file mapped shared, a memfd) with its file's length, by inode
    /// order (a checkpoint saves them ahead of the filesystem).
    pub fn page_objects(&self) -> Vec<(crate::shm::MemRef, u64)> {
        self.inodes
            .values()
            .filter_map(|i| match &i.node {
                Node::File(FileData::Pages { mem, len }) => Some((mem.clone(), *len)),
                _ => None,
            })
            .collect()
    }

    /// The memory object holding file `ino`, if it has one.
    pub fn file_pages(&self, ino: Ino) -> Option<crate::shm::MemRef> {
        match &self.inodes.get(&ino)?.node {
            Node::File(FileData::Pages { mem, .. }) => Some(mem.clone()),
            _ => None,
        }
    }

    /// Moves regular file `ino` into the memory object `mem`, which must already hold its
    /// current bytes (see [`Vfs::file_contents`]): from now on the object is the file.
    pub fn set_file_pages(&mut self, ino: Ino, mem: crate::shm::MemRef) -> Result<(), Errno> {
        let len = self.size(ino);
        let id = mem.id;
        let origin = match &mut self.inode_mut(ino).node {
            node @ Node::File(_) => {
                let (origin, in_memory) = match node {
                    Node::File(FileData::Image { offset, len }) => (Some((*offset, *len)), 0),
                    Node::File(FileData::Mem(v)) => (None, v.len() as u64),
                    _ => (None, 0),
                };
                *node = Node::File(FileData::Pages { mem, len });
                // The bytes live in the memory object now.
                self.written -= in_memory;
                origin
            }
            Node::Dir(_) => return Err(Errno::EISDIR),
            _ => return Err(Errno::EINVAL),
        };
        if let Some(o) = origin {
            self.origins.insert(id, o);
        }
        Ok(())
    }

    /// Where page-cache object `id` came from in the image (offset, length), if it was made
    /// from a file still in the image (its baseline).
    pub fn page_origin(&self, id: u64) -> Option<(u64, u64)> {
        self.origins.get(&id).copied()
    }

    /// Records that page-cache object `id` came from the image at `origin` (a restore).
    pub(crate) fn set_page_origin(&mut self, id: u64, origin: (u64, u64)) {
        self.origins.insert(id, origin);
    }

    /// The image's bytes at `origin` (offset, length) as pages, zeros after: the baseline of a
    /// page-cache object made from an image file. `None` without an image.
    pub fn image_pages(&self, origin: (u64, u64)) -> Option<Rc<dyn wasmshim_cpu::PageSource>> {
        let (base, len) = origin;
        let c = FileContents::Image {
            blob: self.blob.clone()?,
            base,
            len,
        };
        Some(c.pages(0, len))
    }

    /// Reads a whole regular file.
    pub fn read_all(&mut self, ino: Ino) -> Result<Vec<u8>, Errno> {
        let size = usize::try_from(self.size(ino)).map_err(|_| Errno::EFBIG)?;
        let mut buf = vec![0; size];
        let n = self.read(ino, 0, &mut buf)?;
        buf.truncate(n);
        Ok(buf)
    }

    /// Sets the limits on file data. They apply to what is written from now on.
    pub fn set_limits(&mut self, limits: DataLimits) {
        self.limits = limits;
    }

    /// Bytes of written files held in memory, the number [`DataLimits::total_bytes`] bounds:
    /// files created or modified since the image was mounted. Files still read from the image
    /// do not count, and neither do page-cache objects (a file mapped shared, a memfd), which
    /// hold only the pages that were touched.
    pub fn written_bytes(&self) -> u64 {
        self.written
    }

    /// [`Vfs::written_bytes`] counted from the inodes (after a restore, and in tests).
    pub fn recount_written(&self) -> u64 {
        self.inodes
            .values()
            .map(|i| match &i.node {
                Node::File(FileData::Mem(v)) => v.len() as u64,
                _ => 0,
            })
            .sum()
    }

    /// Accounts for `bytes` more of written file data, or fails with `ENOSPC` at the limit.
    fn reserve(&mut self, bytes: u64) -> Result<(), Errno> {
        let total = self.written.checked_add(bytes).ok_or(Errno::ENOSPC)?;
        if total > self.limits.total_bytes {
            return Err(Errno::ENOSPC);
        }
        self.written = total;
        Ok(())
    }

    /// Removes inode `ino` for good.
    fn forget(&mut self, ino: Ino) {
        if let Some(Inode {
            node: Node::File(FileData::Mem(v)),
            ..
        }) = self.inodes.remove(&ino)
        {
            self.written -= v.len() as u64;
        }
    }

    /// Copies an image file into memory so it can be modified.
    fn copy_up(&mut self, ino: Ino) -> Result<&mut Vec<u8>, Errno> {
        if let Node::File(FileData::Image { len, .. }) = self.inode(ino).node {
            self.reserve(len)?;
            let data = match self.read_all(ino) {
                Ok(data) => data,
                Err(e) => {
                    self.written -= len;
                    return Err(e);
                }
            };
            // A short read (a damaged image) holds less than was reserved.
            self.written -= len - data.len() as u64;
            self.inode_mut(ino).node = Node::File(FileData::Mem(Rc::new(data)));
        }
        match &mut self.inode_mut(ino).node {
            Node::File(FileData::Mem(v)) => Ok(Rc::make_mut(v)),
            Node::Dir(_) => Err(Errno::EISDIR),
            _ => Err(Errno::EINVAL),
        }
    }

    /// Checks a write of `[offset, end)` to `ino` against its seals (`memfd`): `EPERM` if the
    /// file is write-sealed, or the write would grow a grow-sealed file.
    fn check_write_seals(&self, ino: Ino, end: u64) -> Result<(), Errno> {
        let seals = self.inode(ino).seals.unwrap_or(0);
        if seals & (seal::F_SEAL_WRITE | seal::F_SEAL_FUTURE_WRITE) != 0
            || (seals & seal::F_SEAL_GROW != 0 && end > self.size(ino))
        {
            return Err(Errno::EPERM);
        }
        Ok(())
    }

    /// Writes at `offset` (extending with zeros if needed).
    pub fn write(
        &mut self,
        ino: Ino,
        offset: u64,
        data: &[u8],
        now: Timespec,
    ) -> Result<usize, Errno> {
        let start = usize::try_from(offset).map_err(|_| Errno::EFBIG)?;
        let end = start.checked_add(data.len()).ok_or(Errno::EFBIG)?;
        if end as u64 > self.limits.file_bytes {
            return Err(Errno::EFBIG);
        }
        self.check_write_seals(ino, end as u64)?;
        if let Node::File(FileData::Pages { mem, len }) = &mut self.inode_mut(ino).node {
            if end as u64 > *len {
                // What a mapping stored past the old end does not become file data.
                mem.pages.zero_from(*len).map_err(|_| Errno::EIO)?;
                *len = end as u64;
            }
            mem.pages.write(offset, data).map_err(|_| Errno::EIO)?;
        } else {
            let old = self.copy_up(ino)?.len();
            if old < end {
                self.reserve((end - old) as u64)?;
            }
            let v = self.copy_up(ino)?;
            if old < end {
                v.resize(end, 0);
            }
            v[start..end].copy_from_slice(data);
        }
        self.touch(ino, now);
        self.note(ino, fsmask::IN_MODIFY);
        Ok(data.len())
    }

    /// Sets the file size.
    pub fn truncate(&mut self, ino: Ino, len: u64, now: Timespec) -> Result<(), Errno> {
        if len > self.limits.file_bytes {
            return Err(Errno::EFBIG);
        }
        let seals = self.inode(ino).seals.unwrap_or(0);
        let size = self.size(ino);
        if (len < size && seals & seal::F_SEAL_SHRINK != 0)
            || (len > size && seals & seal::F_SEAL_GROW != 0)
        {
            return Err(Errno::EPERM);
        }
        if let Node::File(FileData::Pages { mem, len: old }) = &mut self.inode_mut(ino).node {
            // Shrinking clears the tail; growing clears what mappings stored past the old end.
            mem.pages.zero_from(len.min(*old)).map_err(|_| Errno::EIO)?;
            *old = len;
        } else {
            let len = usize::try_from(len).map_err(|_| Errno::EFBIG)?;
            let old = self.copy_up(ino)?.len();
            if old < len {
                self.reserve((len - old) as u64)?;
            } else {
                self.written -= (old - len) as u64;
            }
            self.copy_up(ino)?.resize(len, 0);
        }
        self.touch(ino, now);
        self.note(ino, fsmask::IN_MODIFY);
        Ok(())
    }

    /// Changes permission bits.
    pub fn chmod(&mut self, ino: Ino, perm: u32, now: Timespec) {
        let i = self.inode_mut(ino);
        i.mode = (i.mode & mode::S_IFMT) | (perm & 0o7777);
        i.ctime = now;
        self.note(ino, fsmask::IN_ATTRIB);
    }

    /// Changes ownership (`u32::MAX` keeps a field).
    pub fn chown(&mut self, ino: Ino, uid: u32, gid: u32, now: Timespec) {
        let i = self.inode_mut(ino);
        if uid != u32::MAX {
            i.uid = uid;
        }
        if gid != u32::MAX {
            i.gid = gid;
        }
        i.ctime = now;
        self.note(ino, fsmask::IN_ATTRIB);
    }

    /// Sets access and modification times (`None` keeps one).
    pub fn set_times(
        &mut self,
        ino: Ino,
        atime: Option<Timespec>,
        mtime: Option<Timespec>,
        now: Timespec,
    ) {
        let i = self.inode_mut(ino);
        if let Some(a) = atime {
            i.atime = a;
        }
        if let Some(m) = mtime {
            i.mtime = m;
        }
        i.ctime = now;
        self.note(ino, fsmask::IN_ATTRIB);
    }

    /// `struct stat` for an inode.
    pub fn stat(&self, ino: Ino) -> Stat {
        let i = self.inode(ino);
        let size = self.size(ino);
        Stat {
            dev: DEV,
            ino,
            mode: i.mode,
            nlink: i.nlink,
            uid: i.uid,
            gid: i.gid,
            rdev: u64::from(i.rdev.0) << 8 | u64::from(i.rdev.1),
            size: size as i64,
            blksize: 4096,
            blocks: size.div_ceil(512) as i64,
            atime: i.atime,
            mtime: i.mtime,
            ctime: i.ctime,
        }
    }

    /// Directory listing including `.` and `..`, in name order after those two.
    pub fn readdir(&self, ino: Ino) -> Result<Vec<DirEntry>, Errno> {
        let i = self.inode(ino);
        let Node::Dir(entries) = &i.node else {
            return Err(Errno::ENOTDIR);
        };
        // Listing a removed directory is ENOENT (fs/readdir.c: `IS_DEADDIR`).
        self.check_live(ino)?;
        let mut out = vec![
            DirEntry {
                name: b".".to_vec(),
                ino,
                dtype: dtype::DT_DIR,
            },
            DirEntry {
                name: b"..".to_vec(),
                ino: i.parent,
                dtype: dtype::DT_DIR,
            },
        ];
        for (name, &child) in entries {
            out.push(DirEntry {
                name: name.clone(),
                ino: child,
                dtype: dtype_of(self.inode(child).mode),
            });
        }
        Ok(out)
    }

    /// The canonical absolute path of `path` (all symlinks resolved), like `realpath(3)`.
    pub fn realpath(&self, cwd: Ino, path: &[u8]) -> Result<Vec<u8>, Errno> {
        let mut path = path.to_vec();
        let mut base = cwd;
        for _ in 0..MAX_SYMLINKS {
            let (dir, name) = self.resolve_parent(base, &path)?;
            let ino = self.lookup(dir, &name)?;
            if let Node::Symlink(target) = &self.inode(ino).node {
                path = target.clone();
                base = dir;
                continue;
            }
            let mut out = self.path_of(dir);
            if out != b"/" {
                out.push(b'/');
            }
            if name != b"." {
                out.extend_from_slice(&name);
            }
            return Ok(out);
        }
        Err(Errno::ELOOP)
    }

    /// The absolute path of a directory (for `getcwd`).
    pub fn path_of(&self, dir: Ino) -> Vec<u8> {
        let mut parts = Vec::new();
        let mut d = dir;
        while d != ROOT {
            let parent = self.inode(d).parent;
            let Node::Dir(entries) = &self.inode(parent).node else {
                break;
            };
            match entries.iter().find(|(_, i)| **i == d) {
                Some((name, _)) => parts.push(name.clone()),
                None => break,
            }
            d = parent;
        }
        if parts.is_empty() {
            return b"/".to_vec();
        }
        let mut out = Vec::new();
        for p in parts.iter().rev() {
            out.push(b'/');
            out.extend_from_slice(p);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::tar::tests::{MemBlob, build_tar};
    use super::*;

    #[test]
    fn changes_are_recorded_only_while_watched_and_in_linux_order() {
        let mut v = image();
        let t = Timespec::default();
        let tmp = v.resolve(ROOT, b"/tmp", true).expect("tmp");
        v.create(tmp, b"quiet", 0o644, t).expect("create");
        assert!(
            v.take_events().is_empty(),
            "nothing recorded before a watch"
        );
        v.watch_events();
        let f = v.create(tmp, b"a", 0o644, t).expect("create");
        v.write(f, 0, b"x", t).expect("write");
        v.rename(tmp, b"a", tmp, b"b", t).expect("rename");
        v.remove(tmp, b"b", false, t).expect("unlink");
        let masks: Vec<u32> = v
            .take_events()
            .iter()
            .map(|e| match e {
                FsEvent::Child { mask, .. } | FsEvent::Inode { mask, .. } => *mask,
            })
            .collect();
        use fsmask::*;
        assert_eq!(
            masks,
            [
                IN_CREATE,
                IN_MODIFY,
                IN_MOVED_FROM,
                IN_MOVED_TO,
                IN_MOVE_SELF,
                IN_ATTRIB,
                IN_DELETE,
                IN_DELETE_SELF
            ]
        );
    }

    fn image() -> Vfs {
        let tar = build_tar(&[
            ("./", b'5', 0o755, b""),
            ("./bin/", b'5', 0o755, b""),
            ("./bin/busybox", b'0', 0o755, b"ELF..."),
            ("./bin/sh", b'2', 0o777, b"busybox"),
            ("./bin/ls", b'1', 0o755, b"./bin/busybox"),
            ("./etc/hostname", b'0', 0o644, b"box\n"),
            ("./lib", b'2', 0o777, b"/usr/lib"),
            ("./usr/lib/libc.so", b'0', 0o755, b"libc"),
        ]);
        let mut vfs = Vfs::new();
        assert_eq!(vfs.mount_image(Box::new(MemBlob(tar))).unwrap(), 8);
        vfs
    }

    #[test]
    fn a_hard_link_listed_before_its_target_is_made() {
        // Path order puts `bin/a` before `bin/zz`, the file it links to; `bin/b` links to a
        // file that never comes; `bin/c` links to `bin/zz` but a later member takes its name.
        let tar = build_tar(&[
            ("bin/", b'5', 0o755, b""),
            ("bin/a", b'1', 0o755, b"bin/zz"),
            ("bin/b", b'1', 0o755, b"bin/nowhere"),
            ("bin/c", b'1', 0o755, b"bin/zz"),
            ("bin/c", b'2', 0o777, b"zz"),
            ("bin/zz", b'0', 0o755, b"the file"),
        ]);
        let mut vfs = Vfs::new();
        assert_eq!(vfs.mount_image(Box::new(MemBlob(tar))).unwrap(), 6);
        let zz = vfs.resolve(ROOT, b"/bin/zz", true).unwrap();
        assert_eq!(vfs.resolve(ROOT, b"/bin/a", false), Ok(zz));
        assert_eq!(vfs.inode(zz).nlink, 2);
        assert_eq!(vfs.resolve(ROOT, b"/bin/b", false), Err(Errno::ENOENT));
        // The later symlink keeps its name.
        let c = vfs.resolve(ROOT, b"/bin/c", false).unwrap();
        assert!(matches!(vfs.inode(c).node, Node::Symlink(_)));
    }

    #[test]
    fn resolves_paths_and_symlinks() {
        let vfs = image();
        let bb = vfs.resolve(ROOT, b"/bin/busybox", true).unwrap();
        assert_eq!(vfs.resolve(ROOT, b"/bin/sh", true), Ok(bb));
        assert_eq!(vfs.resolve(ROOT, b"bin/ls", true), Ok(bb));
        assert_eq!(vfs.inode(bb).nlink, 2);
        assert_ne!(vfs.resolve(ROOT, b"/bin/sh", false), Ok(bb));
        assert!(vfs.resolve(ROOT, b"/lib/libc.so", true).is_ok());
        assert_eq!(
            vfs.resolve(ROOT, b"/etc/hostname/", true),
            Err(Errno::ENOTDIR)
        );
        assert_eq!(vfs.resolve(ROOT, b"/nope", true), Err(Errno::ENOENT));
        let etc = vfs.resolve(ROOT, b"/etc", true).unwrap();
        assert_eq!(vfs.resolve(etc, b"../bin/./busybox", true), Ok(bb));
        assert_eq!(vfs.resolve(ROOT, b"/..", true), Ok(ROOT));
        assert_eq!(vfs.path_of(etc), b"/etc");
        assert_eq!(
            vfs.resolve_parent(ROOT, b"/tmp/x"),
            Ok((vfs.resolve(ROOT, b"/tmp", true).unwrap(), b"x".to_vec()))
        );
        assert_eq!(vfs.resolve_parent(ROOT, b"x"), Ok((ROOT, b"x".to_vec())));
    }

    #[test]
    fn symlink_loops_are_detected() {
        let mut vfs = Vfs::new();
        vfs.symlink(ROOT, b"a", b"b", Timespec::default()).unwrap();
        vfs.symlink(ROOT, b"b", b"a", Timespec::default()).unwrap();
        assert_eq!(vfs.resolve(ROOT, b"/a", true), Err(Errno::ELOOP));
    }

    #[test]
    fn copy_up_on_write_leaves_the_image_alone() {
        let mut vfs = image();
        let host = vfs.resolve(ROOT, b"/etc/hostname", true).unwrap();
        assert_eq!(vfs.read_all(host).unwrap(), b"box\n");
        vfs.write(host, 4, b"x", Timespec::default()).unwrap();
        assert_eq!(vfs.read_all(host).unwrap(), b"box\nx");
        vfs.truncate(host, 2, Timespec::default()).unwrap();
        assert_eq!(vfs.read_all(host).unwrap(), b"bo");
    }

    #[test]
    fn create_rename_remove() {
        let mut vfs = image();
        let now = Timespec::default();
        let tmp = vfs.resolve(ROOT, b"/tmp", true).unwrap();
        let f = vfs.create(tmp, b"x", 0o644, now).unwrap();
        vfs.write(f, 0, b"hi\n", now).unwrap();
        assert_eq!(vfs.create(tmp, b"x", 0o644, now), Err(Errno::EEXIST));
        let d = vfs.mkdir(tmp, b"d", 0o755, now).unwrap();
        vfs.rename(tmp, b"x", d, b"y", now).unwrap();
        assert_eq!(vfs.resolve(ROOT, b"/tmp/d/y", true), Ok(f));
        assert_eq!(vfs.remove(tmp, b"d", true, now), Err(Errno::ENOTEMPTY));
        assert_eq!(vfs.rename(tmp, b"d", d, b"inner", now), Err(Errno::EINVAL));
        vfs.remove(d, b"y", false, now).unwrap();
        vfs.remove(tmp, b"d", true, now).unwrap();
        let names: Vec<_> = vfs
            .readdir(tmp)
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names, [b".".to_vec(), b"..".to_vec()]);
        let devnull = vfs.resolve(ROOT, b"/dev/null", true).unwrap();
        assert_eq!(vfs.stat(devnull).rdev, 0x103);
    }
}

/// The outcome of [`Vfs::resolve_escape`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolved {
    /// The path names this inode.
    Ino(Ino),
    /// A symlink led out of the VFS, into `/proc`: continue with this absolute path.
    Escape(Vec<u8>),
}

/// inotify event bits (`<linux/inotify.h>`), as the VFS and the kernel report them.
pub mod fsmask {
    pub const IN_ACCESS: u32 = 0x1;
    pub const IN_MODIFY: u32 = 0x2;
    pub const IN_ATTRIB: u32 = 0x4;
    pub const IN_CLOSE_WRITE: u32 = 0x8;
    pub const IN_CLOSE_NOWRITE: u32 = 0x10;
    pub const IN_OPEN: u32 = 0x20;
    pub const IN_MOVED_FROM: u32 = 0x40;
    pub const IN_MOVED_TO: u32 = 0x80;
    pub const IN_CREATE: u32 = 0x100;
    pub const IN_DELETE: u32 = 0x200;
    pub const IN_DELETE_SELF: u32 = 0x400;
    pub const IN_MOVE_SELF: u32 = 0x800;
    pub const IN_UNMOUNT: u32 = 0x2000;
    pub const IN_Q_OVERFLOW: u32 = 0x4000;
    pub const IN_IGNORED: u32 = 0x8000;
    pub const IN_ONLYDIR: u32 = 0x0100_0000;
    pub const IN_DONT_FOLLOW: u32 = 0x0200_0000;
    pub const IN_EXCL_UNLINK: u32 = 0x0400_0000;
    pub const IN_MASK_CREATE: u32 = 0x1000_0000;
    pub const IN_MASK_ADD: u32 = 0x2000_0000;
    pub const IN_ISDIR: u32 = 0x4000_0000;
    pub const IN_ONESHOT: u32 = 0x8000_0000;
    /// Every event a watch can ask for.
    pub const IN_ALL_EVENTS: u32 = 0xfff;
}

/// A change in the VFS, as inotify sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FsEvent {
    /// Something happened to `name` (inode `ino`) in directory `dir`: watchers of `dir` get
    /// it with the name, watchers of `ino` without.
    Child {
        dir: Ino,
        name: Vec<u8>,
        ino: Ino,
        mask: u32,
        cookie: u32,
    },
    /// Something happened to `ino` itself (written, attributes, opened, closed, deleted).
    /// `parent` is the directory entry it was reached by when the event happened (its
    /// watchers hear about it by name); `None` for the self-only events.
    Inode {
        ino: Ino,
        mask: u32,
        isdir: bool,
        parent: Option<(Ino, Vec<u8>)>,
    },
}

impl Vfs {
    /// Starts (or keeps) recording change events for inotify.
    pub fn watch_events(&mut self) {
        self.events.get_or_insert_with(Vec::new);
    }

    /// Takes the recorded events (empty when nobody watches).
    pub fn take_events(&mut self) -> Vec<FsEvent> {
        self.events.as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// `true` if events are recorded.
    pub fn has_events(&self) -> bool {
        self.events.as_ref().is_some_and(|e| !e.is_empty())
    }

    /// Records an event on `ino` itself.
    pub fn note(&mut self, ino: Ino, mask: u32) {
        if self.events.is_none() {
            return;
        }
        let selfish = mask & (fsmask::IN_DELETE_SELF | fsmask::IN_MOVE_SELF) != 0;
        let parent = if selfish {
            None
        } else {
            self.name_in_parent(ino)
        };
        let isdir = self.inodes.get(&ino).is_some_and(Inode::is_dir);
        if let Some(ev) = self.events.as_mut() {
            ev.push(FsEvent::Inode {
                ino,
                mask,
                isdir,
                parent,
            });
        }
    }

    fn note_child(&mut self, dir: Ino, name: &[u8], ino: Ino, mask: u32, cookie: u32) {
        if self.events.is_none() {
            return;
        }
        let isdir = if self.inodes.get(&ino).is_some_and(Inode::is_dir) {
            fsmask::IN_ISDIR
        } else {
            0
        };
        if let Some(ev) = self.events.as_mut() {
            ev.push(FsEvent::Child {
                dir,
                name: name.to_vec(),
                ino,
                mask: mask | isdir,
                cookie,
            });
        }
    }

    /// One name of `ino` in its parent directory, if it has one (for events reported to the
    /// parent's watchers when only the inode is known).
    pub fn name_in_parent(&self, ino: Ino) -> Option<(Ino, Vec<u8>)> {
        let parent = self.inodes.get(&ino)?.parent;
        if parent == ino {
            return None;
        }
        match &self.inodes.get(&parent)?.node {
            Node::Dir(entries) => entries
                .iter()
                .find(|(_, i)| **i == ino)
                .map(|(n, _)| (parent, n.clone())),
            _ => None,
        }
    }
}

/// `openat2`'s `RESOLVE_*` rules for [`Vfs::resolve_restricted`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Restrict {
    /// `RESOLVE_BENEATH`: the walk may not leave the start directory (`EXDEV`).
    pub beneath: bool,
    /// `RESOLVE_IN_ROOT`: the start directory is the root (absolute paths and `..` stay in).
    pub in_root: bool,
    /// `RESOLVE_NO_SYMLINKS`: following any symlink is `ELOOP`.
    pub no_symlinks: bool,
    /// `RESOLVE_NO_XDEV`: entering another mount (`/proc`) is `EXDEV`.
    pub no_xdev: bool,
}

/// Where [`Vfs::resolve_restricted`] ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Walked {
    /// The final component `name` in directory `parent` (it may not exist), with the trailing
    /// symlink already followed if asked.
    Entry { parent: Ino, name: Vec<u8> },
    /// The start itself (a path of only `.`/`..` components, or `/` under `RESOLVE_IN_ROOT`).
    Dir(Ino),
    /// The walk entered `/proc` (not in the VFS): the rest, as an absolute path.
    Proc(Vec<u8>),
}

impl Vfs {
    /// Resolves `path` from `start` under `openat2`'s rules. `..` is tracked against the walk
    /// itself, so `RESOLVE_BENEATH` and `RESOLVE_IN_ROOT` hold through symlinks too.
    pub fn resolve_restricted(
        &self,
        start: Ino,
        path: &[u8],
        follow: bool,
        rules: Restrict,
    ) -> Result<Walked, Errno> {
        if path.is_empty() {
            return Err(Errno::ENOENT);
        }
        if path.len() >= PATH_MAX {
            return Err(Errno::ENAMETOOLONG);
        }
        let confined = rules.beneath || rules.in_root;
        // The directories walked from the root of the walk (`start` when confined).
        let mut stack: Vec<Ino> = Vec::new();
        let mut cur = start;
        let restart = |stack: &mut Vec<Ino>, cur: &mut Ino| -> Result<(), Errno> {
            if rules.beneath {
                return Err(Errno::EXDEV);
            }
            stack.clear();
            *cur = if rules.in_root { start } else { ROOT };
            Ok(())
        };
        if path[0] == b'/' {
            restart(&mut stack, &mut cur)?;
        }
        let trailing_slash = path.ends_with(b"/");
        let mut todo: VecDeque<Vec<u8>> = components(path).map(<[u8]>::to_vec).collect();
        let mut links = 0;
        let mut last = None;
        while let Some(name) = todo.pop_front() {
            if name.len() > NAME_MAX {
                return Err(Errno::ENAMETOOLONG);
            }
            let is_last = todo.is_empty();
            if !self.inode(cur).is_dir() {
                return Err(Errno::ENOTDIR);
            }
            match name.as_slice() {
                b"." => {
                    last = None;
                    continue;
                }
                b".." => {
                    last = None;
                    if let Some(up) = stack.pop() {
                        cur = up;
                    } else if confined {
                        if rules.beneath {
                            return Err(Errno::EXDEV);
                        }
                        // IN_ROOT: `..` of the root is the root.
                    } else {
                        cur = self.inode(cur).parent;
                    }
                    continue;
                }
                _ => {}
            }
            // `/proc` is another filesystem, mounted on the real root.
            if cur == ROOT && !(rules.in_root && start != ROOT) && name == b"proc" {
                if rules.no_xdev {
                    return Err(Errno::EXDEV);
                }
                let mut rest = b"/proc".to_vec();
                for c in &todo {
                    rest.push(b'/');
                    rest.extend_from_slice(c);
                }
                return Ok(Walked::Proc(rest));
            }
            let next = match self.lookup(cur, &name) {
                Ok(n) => n,
                Err(Errno::ENOENT) if is_last => {
                    last = Some(name);
                    break;
                }
                Err(e) => return Err(e),
            };
            if let Node::Symlink(target) = &self.inode(next).node
                && (!is_last || follow || trailing_slash)
            {
                if rules.no_symlinks {
                    return Err(Errno::ELOOP);
                }
                links += 1;
                if links > MAX_SYMLINKS {
                    return Err(Errno::ELOOP);
                }
                if target.first() == Some(&b'/') {
                    restart(&mut stack, &mut cur)?;
                }
                for c in components(target).rev() {
                    todo.push_front(c.to_vec());
                }
                last = None;
                continue;
            }
            if is_last {
                last = Some(name);
                break;
            }
            stack.push(cur);
            cur = next;
        }
        match last {
            Some(name) => Ok(Walked::Entry { parent: cur, name }),
            None => Ok(Walked::Dir(cur)),
        }
    }

    /// The value of extended attribute `name` of `ino`.
    pub fn get_xattr(&self, ino: Ino, name: &[u8]) -> Option<&[u8]> {
        self.inode(ino).xattrs.get(name).map(Vec::as_slice)
    }

    /// Sets (`Some`) or removes (`None`) extended attribute `name` of `ino`.
    pub fn set_xattr(&mut self, ino: Ino, name: &[u8], value: Option<Vec<u8>>, now: Timespec) {
        let i = self.inode_mut(ino);
        match value {
            Some(v) => {
                i.xattrs.insert(name.to_vec(), v);
            }
            None => {
                i.xattrs.remove(name);
            }
        }
        i.ctime = now;
        self.note(ino, fsmask::IN_ATTRIB);
    }

    /// Zeroes `[offset, offset + len)` of regular file `ino`, within its size (a punched hole).
    pub fn zero_range(
        &mut self,
        ino: Ino,
        offset: u64,
        len: u64,
        now: Timespec,
    ) -> Result<(), Errno> {
        let end = offset.saturating_add(len).min(self.size(ino));
        let zeros = [0u8; 4096];
        let mut at = offset;
        while at < end {
            let n = (end - at).min(zeros.len() as u64) as usize;
            self.write(ino, at, &zeros[..n], now)?;
            at += n as u64;
        }
        Ok(())
    }
}
