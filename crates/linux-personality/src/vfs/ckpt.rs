//! Checkpoints of the filesystem: every inode as it is, files still in the
//! image as references into it (the restored machine mounts the same image first), files in
//! memory with their bytes.

use std::collections::BTreeMap;
use std::rc::Rc;

use wasmshim_kernel::checkpoint::{Ckpt, CkptError, Reader, Writer};
use wasmshim_linux_abi::structs::Timespec;

use super::{Device, FileData, FsEvent, Inode, Node, Vfs};

/// Saves a [`Timespec`] (a foreign type: no [`Ckpt`] impl).
pub(crate) fn save_time(t: &Timespec, w: &mut Writer) {
    t.sec.save(w);
    t.nsec.save(w);
}

/// Loads what [`save_time`] wrote.
pub(crate) fn load_time(r: &mut Reader<'_>) -> Result<Timespec, CkptError> {
    Ok(Timespec {
        sec: Ckpt::load(r)?,
        nsec: Ckpt::load(r)?,
    })
}

impl Ckpt for Device {
    fn save(&self, w: &mut Writer) {
        let (tag, n) = match self {
            Self::Null => (0u8, 0),
            Self::Zero => (1, 0),
            Self::Random => (2, 0),
            Self::Full => (3, 0),
            Self::Tty => (4, 0),
            Self::Console => (5, 0),
            Self::Ptmx => (6, 0),
            Self::Pts(n) => (7, *n),
            Self::HidRaw(n) => (8, *n),
            Self::Fb(n) => (9, *n),
        };
        tag.save(w);
        n.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        let tag = u8::load(r)?;
        let n = u32::load(r)?;
        Ok(match tag {
            0 => Self::Null,
            1 => Self::Zero,
            2 => Self::Random,
            3 => Self::Full,
            4 => Self::Tty,
            5 => Self::Console,
            6 => Self::Ptmx,
            7 => Self::Pts(n),
            8 => Self::HidRaw(n),
            9 => Self::Fb(n),
            _ => return Err(CkptError::new("bad device")),
        })
    }
}

thread_local! {
    /// The page-cache objects a restore made before loading the filesystem, by id
    /// ([`crate::kernel::Kernel::restore`] reads them first; files and shared mappings name them).
    pub(crate) static RESTORED_PAGES: std::cell::RefCell<std::collections::BTreeMap<u64, crate::shm::MemRef>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}

/// A file's data. A page-cache object (`FileData::Pages`: a file mapped shared, a memfd) is
/// named by its id; the kernel's checkpoint saves the objects ahead of the filesystem.
fn save_data(d: &FileData, w: &mut Writer) -> Result<(), CkptError> {
    match d {
        FileData::Image { offset, len } => {
            0u8.save(w);
            offset.save(w);
            len.save(w);
        }
        FileData::Mem(v) => {
            1u8.save(w);
            w.bytes(v);
        }
        FileData::Pages { mem, len } => {
            2u8.save(w);
            (mem.id, *len).save(w);
        }
    }
    Ok(())
}

fn load_data(r: &mut Reader<'_>) -> Result<FileData, CkptError> {
    Ok(match u8::load(r)? {
        0 => FileData::Image {
            offset: Ckpt::load(r)?,
            len: Ckpt::load(r)?,
        },
        1 => FileData::Mem(Rc::new(r.bytes()?.to_vec())),
        2 => {
            let (id, len): (u64, u64) = Ckpt::load(r)?;
            let mem = RESTORED_PAGES
                .with(|m| m.borrow().get(&id).cloned())
                .ok_or_else(|| CkptError::new("a file in a page-cache object not saved"))?;
            FileData::Pages { mem, len }
        }
        _ => return Err(CkptError::new("bad file data")),
    })
}

fn save_node(n: &Node, w: &mut Writer) -> Result<(), CkptError> {
    match n {
        Node::Dir(entries) => {
            0u8.save(w);
            entries.save(w);
        }
        Node::File(d) => {
            1u8.save(w);
            save_data(d, w)?;
        }
        Node::Symlink(t) => {
            2u8.save(w);
            t.save(w);
        }
        Node::Device(d) => {
            3u8.save(w);
            d.save(w);
        }
        Node::Special => 4u8.save(w),
        Node::Socket => 5u8.save(w),
    }
    Ok(())
}

fn load_node(r: &mut Reader<'_>) -> Result<Node, CkptError> {
    Ok(match u8::load(r)? {
        0 => Node::Dir(Ckpt::load(r)?),
        1 => Node::File(load_data(r)?),
        2 => Node::Symlink(Ckpt::load(r)?),
        3 => Node::Device(Ckpt::load(r)?),
        4 => Node::Special,
        5 => Node::Socket,
        _ => return Err(CkptError::new("bad inode")),
    })
}

fn save_inode(i: &Inode, w: &mut Writer) -> Result<(), CkptError> {
    i.ino.save(w);
    i.mode.save(w);
    i.uid.save(w);
    i.gid.save(w);
    i.nlink.save(w);
    i.rdev.save(w);
    save_time(&i.atime, w);
    save_time(&i.mtime, w);
    save_time(&i.ctime, w);
    i.parent.save(w);
    i.opens.save(w);
    i.seals.save(w);
    i.xattrs.save(w);
    save_node(&i.node, w)
}

fn load_inode(r: &mut Reader<'_>) -> Result<Inode, CkptError> {
    Ok(Inode {
        ino: Ckpt::load(r)?,
        mode: Ckpt::load(r)?,
        uid: Ckpt::load(r)?,
        gid: Ckpt::load(r)?,
        nlink: Ckpt::load(r)?,
        rdev: Ckpt::load(r)?,
        atime: load_time(r)?,
        mtime: load_time(r)?,
        ctime: load_time(r)?,
        parent: Ckpt::load(r)?,
        opens: Ckpt::load(r)?,
        seals: Ckpt::load(r)?,
        xattrs: Ckpt::load(r)?,
        node: load_node(r)?,
    })
}

impl Ckpt for FsEvent {
    fn save(&self, w: &mut Writer) {
        match self {
            Self::Child {
                dir,
                name,
                ino,
                mask,
                cookie,
            } => {
                0u8.save(w);
                dir.save(w);
                name.save(w);
                ino.save(w);
                mask.save(w);
                cookie.save(w);
            }
            Self::Inode {
                ino,
                mask,
                isdir,
                parent,
            } => {
                1u8.save(w);
                ino.save(w);
                mask.save(w);
                isdir.save(w);
                parent.save(w);
            }
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(match u8::load(r)? {
            0 => Self::Child {
                dir: Ckpt::load(r)?,
                name: Ckpt::load(r)?,
                ino: Ckpt::load(r)?,
                mask: Ckpt::load(r)?,
                cookie: Ckpt::load(r)?,
            },
            1 => Self::Inode {
                ino: Ckpt::load(r)?,
                mask: Ckpt::load(r)?,
                isdir: Ckpt::load(r)?,
                parent: Ckpt::load(r)?,
            },
            _ => return Err(CkptError::new("bad filesystem event")),
        })
    }
}

/// Checks that every directory's chain of parents (`..`) is made of directories and ends at the
/// root, as it does in a filesystem the VFS built (a removed directory still in use keeps its
/// removed ancestors). Walks such as `getcwd` follow these chains and would never end on a
/// cycle. Every parent is known to exist.
fn reach_the_root(inodes: &BTreeMap<super::Ino, Inode>) -> Result<(), CkptError> {
    let mut reach = std::collections::BTreeSet::from([super::ROOT]);
    for dir in inodes.values().filter(|i| i.is_dir()) {
        let mut chain = Vec::new();
        let mut d = dir;
        while !reach.contains(&d.ino) {
            // A chain longer than the inodes there are has a cycle.
            if chain.len() > inodes.len() {
                return Err(CkptError::new("a directory does not lead to the root"));
            }
            chain.push(d.ino);
            d = match inodes.get(&d.parent) {
                Some(p) if p.is_dir() => p,
                _ => return Err(CkptError::new("a directory's parent is not a directory")),
            };
        }
        reach.extend(chain);
    }
    Ok(())
}

impl Vfs {
    /// Saves every inode and the counters. Fails on what is not checkpointed yet (files with
    /// shared mappings, memfds).
    pub fn save(&self, w: &mut Writer) -> Result<(), CkptError> {
        self.blob.is_some().save(w);
        (self.inodes.len() as u64).save(w);
        for i in self.inodes.values() {
            save_inode(i, w)?;
        }
        self.next_ino.save(w);
        self.events.save(w);
        self.next_cookie.save(w);
        Ok(())
    }

    /// Replaces this filesystem's inodes and counters with saved ones. If the checkpoint was
    /// taken with an image, this filesystem must have mounted the same image (its files'
    /// bytes are read from it).
    pub fn load_saved(&mut self, r: &mut Reader<'_>) -> Result<(), CkptError> {
        let had_image = bool::load(r)?;
        if had_image && self.blob.is_none() {
            return Err(CkptError::new("the checkpoint needs its image mounted"));
        }
        let n = u64::load(r)?;
        let mut inodes = BTreeMap::new();
        for _ in 0..n {
            let i = load_inode(r)?;
            if inodes.insert(i.ino, i).is_some() {
                return Err(CkptError::new("duplicate inode"));
            }
        }
        let next_ino = Ckpt::load(r)?;
        let events = Ckpt::load(r)?;
        let next_cookie = Ckpt::load(r)?;
        // Every name and parent must point at an inode.
        let known = |ino| inodes.contains_key(&ino);
        for i in inodes.values() {
            let dangling = match &i.node {
                Node::Dir(entries) => entries.values().any(|&e| !known(e)),
                _ => false,
            };
            if dangling || !known(i.parent) {
                return Err(CkptError::new("a directory entry names a missing inode"));
            }
        }
        if !inodes.get(&super::ROOT).is_some_and(Inode::is_dir) {
            return Err(CkptError::new("no root directory"));
        }
        reach_the_root(&inodes)?;
        self.inodes = inodes;
        self.written = self.recount_written();
        self.next_ino = next_ino;
        self.events = events;
        self.next_cookie = next_cookie;
        self.find_dead_dirs();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::tar::tests::{MemBlob, build_tar};

    fn now() -> Timespec {
        Timespec { sec: 5, nsec: 6 }
    }

    fn image() -> Vec<u8> {
        build_tar(&[
            ("bin/", b'5', 0o755, b""),
            ("bin/hello", b'0', 0o755, b"hello from the image"),
        ])
    }

    fn saved(v: &Vfs) -> Vec<u8> {
        let mut w = Writer::new();
        v.save(&mut w).unwrap();
        w.into_bytes()
    }

    #[test]
    fn a_filesystem_round_trips_over_its_image() {
        let mut v = Vfs::new();
        v.mount_image(Box::new(MemBlob(image()))).unwrap();
        let bin = v.resolve(super::super::ROOT, b"/bin", true).unwrap();
        let f = v.create(bin, b"new", 0o644, now()).unwrap();
        v.write(f, 3, b"abc", now()).unwrap();
        v.mkdir(super::super::ROOT, b"work", 0o700, now()).unwrap();
        v.symlink(super::super::ROOT, b"link", b"/bin/hello", now())
            .unwrap();
        v.set_xattr(f, b"user.x", Some(b"y".to_vec()), now());
        v.watch_events();
        let hello = v.resolve(super::super::ROOT, b"/bin/hello", true).unwrap();
        v.note(hello, 2);
        let bytes = saved(&v);

        let mut d = Vfs::new();
        d.mount_image(Box::new(MemBlob(image()))).unwrap();
        d.load_saved(&mut Reader::new(&bytes)).unwrap();
        assert_eq!(saved(&d), bytes);
        assert_eq!(d.read_all(f).unwrap(), b"\0\0\0abc");
        assert_eq!(d.read_all(hello).unwrap(), b"hello from the image");
        assert_eq!(
            d.realpath(super::super::ROOT, b"/link").unwrap(),
            b"/bin/hello"
        );
        assert_eq!(d.take_events(), v.take_events());
        // A new file gets the next inode number, as it would have.
        let a = v.create(super::super::ROOT, b"a", 0o644, now()).unwrap();
        let b = d.create(super::super::ROOT, b"a", 0o644, now()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn a_removed_directory_in_use_survives_a_restore() {
        let root = super::super::ROOT;
        let mut v = Vfs::new();
        v.mount_image(Box::new(MemBlob(image()))).unwrap();
        let d = v.mkdir(root, b"gone", 0o755, now()).unwrap();
        v.retain(d);
        v.remove(root, b"gone", true, now()).unwrap();
        v.reap_dead_dirs(|_| false);
        assert!(v.is_dead_dir(d), "an open description keeps it");
        let bytes = saved(&v);

        let mut r = Vfs::new();
        r.mount_image(Box::new(MemBlob(image()))).unwrap();
        r.load_saved(&mut Reader::new(&bytes)).unwrap();
        assert!(r.is_dead_dir(d));
        assert_eq!(r.stat(d).nlink, 0);
        assert_eq!(
            r.create(d, b"f", 0o644, now()),
            Err(wasmshim_linux_abi::Errno::ENOENT)
        );
        r.release(d);
        r.reap_dead_dirs(|_| false);
        assert!(!r.exists(d), "freed once nothing refers to it");
    }

    #[test]
    fn restoring_needs_the_image_and_whole_bytes() {
        let mut v = Vfs::new();
        v.mount_image(Box::new(MemBlob(image()))).unwrap();
        let bytes = saved(&v);
        assert!(Vfs::new().load_saved(&mut Reader::new(&bytes)).is_err());
        for n in 0..bytes.len() {
            let mut d = Vfs::new();
            d.mount_image(Box::new(MemBlob(image()))).unwrap();
            assert!(d.load_saved(&mut Reader::new(&bytes[..n])).is_err());
        }
    }

    #[test]
    fn directories_that_never_reach_the_root_are_refused() {
        let root = super::super::ROOT;
        let mut v = Vfs::new();
        let a = v.mkdir(root, b"a", 0o755, now()).unwrap();
        let b = v.mkdir(a, b"b", 0o755, now()).unwrap();
        let f = v.create(b, b"f", 0o644, now()).unwrap();
        // A legitimate tree loads.
        let good = saved(&v);
        assert!(Vfs::new().load_saved(&mut Reader::new(&good)).is_ok());
        // `b` lists `a`, and `a`'s parent is `b`: `..` goes round for ever, and so would
        // `getcwd` in either. Every name still points at an inode.
        let Node::Dir(entries) = &mut v.inode_mut(b).node else {
            unreachable!()
        };
        entries.insert(b"a".to_vec(), a);
        v.inode_mut(a).parent = b;
        let cycle = saved(&v);
        assert!(Vfs::new().load_saved(&mut Reader::new(&cycle)).is_err());
        // A directory whose parent is a file.
        v.inode_mut(a).parent = f;
        let bad = saved(&v);
        assert!(Vfs::new().load_saved(&mut Reader::new(&bad)).is_err());
    }
}
