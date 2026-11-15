//! Checkpoints of a Linux machine: the filesystem, the substrate's threads,
//! and Linux's processes and objects, with sharing preserved (one saved object per `Rc`:
//! `dup`ed descriptors, fork-shared tables, epoll interests), in one snapshot container of kind
//! [`SNAPSHOT_KIND`] whose body has version [`SNAPSHOT_VERSION`].
//!
//! **Saved:** host stdio; VFS files and directories (image files as references into the image);
//! pipes; `/proc` files; eventfds, timerfds, epoll sets, signalfds and pidfds; inotify instances
//! and their watches; netlink sockets; every address space (private file mappings are replayed
//! from the restored filesystem; only pages that differ are stored); memory objects (shared
//! anonymous memory, files mapped shared and memfds, also ones unlinked while mapped); SysV
//! shared memory segments; native AIO contexts; file locks (POSIX, open-file-description and
//! `flock`); HID devices and framebuffers with the frames not taken yet; local sockets (AF_UNIX,
//! and AF_INET/AF_INET6 ones on the in-kernel loopback) unconnected, listening or connected, with
//! their queued bytes, datagrams and connections not accepted yet; and the network provider's
//! idle or listening sockets with nothing queued, with the provider's own records.
//!
//! **Refused** (the checkpoint fails and changes nothing; [`Kernel::checkpoint_blockers`] lists
//! every obstacle): pseudo-terminals and the console, SysV semaphores and message queues, POSIX
//! message queues, provider sockets that are connecting, connected, closing, failed or have data
//! queued, local sockets with descriptors in flight, a shared mapping of a file outside the
//! filesystem, more memory objects than a checkpoint holds, and sockets of a provider that cannot
//! checkpoint them.
//!
//! The design note, with the restore protocol and the trust model, is
//! `docs/design/checkpoints.md` in this repository.

use std::any::Any;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use wasmshim_cpu::{CpuContext, GuestCpu, GuestMemory, MapError};
use wasmshim_kernel::checkpoint::{
    Baseline, Ckpt, CkptError, Objects, Reader, Writer, read_header, write_header,
};
use wasmshim_linux_abi::structs::KernelSigaction;

use crate::fd::{FdTable, FileKind, FileRef, OpenFile, Stdio};
use crate::kernel::{ExitStatus, Kernel, LinuxKey, LinuxThread, Park, Pid, Process, Zombie};
use crate::mm::{AddressSpace, Vma, VmaKind};
use crate::objects::epoll::{Epoll, Interest};
use crate::objects::eventfd::EventFd;
use crate::objects::signalfd::SignalFd;
use crate::objects::timerfd::TimerFd;
use crate::pipe::Pipe;
use crate::syscall::{Creds, ITimer};
use crate::vfs::{ROOT, Vfs};

pub use wasmshim_snapshot::Snapshot;

/// The snapshot container's kind for a Linux machine.
pub const SNAPSHOT_KIND: &str = "linux";
/// The body's version: bumped when what the body holds changes meaning.
pub const SNAPSHOT_VERSION: u64 = 4;

fn unsupported(what: &str) -> CkptError {
    CkptError(format!("cannot checkpoint {what} yet"))
}

impl Ckpt for LinuxKey {
    fn save(&self, w: &mut Writer) {
        match self {
            Self::Children(p) => (0u8, *p).save(w),
            Self::Signals(p) => (1u8, *p).save(w),
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        match <(u8, Pid)>::load(r)? {
            (0, p) => Ok(Self::Children(p)),
            (1, p) => Ok(Self::Signals(p)),
            _ => Err(CkptError::new("bad wait key")),
        }
    }
}

impl Ckpt for Park {
    fn save(&self, w: &mut Writer) {
        match self {
            Self::Vfork { child } => (0u8, *child).save(w),
            Self::SigSuspend => (1u8, 0u32).save(w),
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        match <(u8, Pid)>::load(r)? {
            (0, child) => Ok(Self::Vfork { child }),
            (1, _) => Ok(Self::SigSuspend),
            _ => Err(CkptError::new("bad parking")),
        }
    }
}

impl Ckpt for LinuxThread {
    fn save(&self, w: &mut Writer) {
        self.pid.save(w);
        self.clear_child_tid.save(w);
        self.sigmask.save(w);
        self.pending.save(w);
        self.sigwait.save(w);
        self.saved_mask.save(w);
        self.altstack.save(w);
        self.name.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(Self {
            pid: Ckpt::load(r)?,
            clear_child_tid: Ckpt::load(r)?,
            sigmask: Ckpt::load(r)?,
            pending: Ckpt::load(r)?,
            sigwait: Ckpt::load(r)?,
            saved_mask: Ckpt::load(r)?,
            altstack: Ckpt::load(r)?,
            name: Ckpt::load(r)?,
        })
    }
}

impl Ckpt for ExitStatus {
    fn save(&self, w: &mut Writer) {
        match self {
            Self::Code(c) => (0u8, u32::from(*c)).save(w),
            Self::Signal(s) => (1u8, *s).save(w),
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        match <(u8, u32)>::load(r)? {
            (0, c) => Ok(Self::Code(c as u8)),
            (1, s) => Ok(Self::Signal(s)),
            _ => Err(CkptError::new("bad exit status")),
        }
    }
}

impl Ckpt for Zombie {
    fn save(&self, w: &mut Writer) {
        self.ppid.save(w);
        self.pgid.save(w);
        self.uid.save(w);
        self.status.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(Self {
            ppid: Ckpt::load(r)?,
            pgid: Ckpt::load(r)?,
            uid: Ckpt::load(r)?,
            status: Ckpt::load(r)?,
        })
    }
}

impl Ckpt for Creds {
    fn save(&self, w: &mut Writer) {
        for id in [
            self.ruid, self.euid, self.suid, self.fsuid, self.rgid, self.egid, self.sgid,
            self.fsgid,
        ] {
            id.save(w);
        }
        self.groups.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(Self {
            ruid: Ckpt::load(r)?,
            euid: Ckpt::load(r)?,
            suid: Ckpt::load(r)?,
            fsuid: Ckpt::load(r)?,
            rgid: Ckpt::load(r)?,
            egid: Ckpt::load(r)?,
            sgid: Ckpt::load(r)?,
            fsgid: Ckpt::load(r)?,
            groups: Ckpt::load(r)?,
        })
    }
}

impl Ckpt for ITimer {
    fn save(&self, w: &mut Writer) {
        self.expiry.save(w);
        self.interval.save(w);
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(Self {
            expiry: Ckpt::load(r)?,
            interval: Ckpt::load(r)?,
        })
    }
}

thread_local! {
    /// The shared objects of the address space being loaded, by id ([`VmaKind`]'s load names
    /// them; [`Loader::process`] defines them first).
    static LOADING_OBJS: RefCell<BTreeMap<u64, crate::shm::MemRef>> =
        const { RefCell::new(BTreeMap::new()) };
}

/// Mapping kinds. A shared mapping names its object by id; the object itself is saved (once,
/// by identity) just before the address space that maps it.
impl Ckpt for VmaKind {
    fn save(&self, w: &mut Writer) {
        match self {
            Self::Anon => 0u8.save(w),
            Self::Heap => 1u8.save(w),
            Self::Stack => 2u8.save(w),
            Self::File { path, offset } => {
                3u8.save(w);
                path.save(w);
                offset.save(w);
            }
            Self::Vdso => 4u8.save(w),
            Self::Shared {
                obj,
                offset,
                name,
                may_write,
            } => {
                5u8.save(w);
                (obj.id, *offset).save(w);
                name.save(w);
                may_write.save(w);
            }
        }
    }
    fn load(r: &mut Reader<'_>) -> Result<Self, CkptError> {
        Ok(match u8::load(r)? {
            0 => Self::Anon,
            1 => Self::Heap,
            2 => Self::Stack,
            3 => Self::File {
                path: Ckpt::load(r)?,
                offset: Ckpt::load(r)?,
            },
            4 => Self::Vdso,
            5 => {
                let (id, offset): (u64, u64) = Ckpt::load(r)?;
                let obj = LOADING_OBJS
                    .with(|m| m.borrow().get(&id).cloned())
                    .ok_or_else(|| CkptError::new("a shared mapping of an object not saved"))?;
                Self::Shared {
                    obj,
                    offset,
                    name: Ckpt::load(r)?,
                    may_write: Ckpt::load(r)?,
                }
            }
            _ => return Err(CkptError::new("bad mapping kind")),
        })
    }
}

fn save_sigaction(a: &KernelSigaction, w: &mut Writer) {
    (a.handler, a.flags, a.restorer).save(w);
    a.mask.save(w);
}

fn load_sigaction(r: &mut Reader<'_>) -> Result<KernelSigaction, CkptError> {
    let (handler, flags, restorer) = Ckpt::load(r)?;
    Ok(KernelSigaction {
        handler,
        flags,
        restorer,
        mask: Ckpt::load(r)?,
    })
}

/// What a private file mapping holds when a restore replays it: the file at its path in the
/// filesystem (which is restored first), from the mapping's offset. A file that is gone, or
/// is no regular file, replays as zeros (the checkpoint then stores every page that is not).
fn file_pages(vfs: &mut Vfs, path: &[u8], offset: u64) -> Option<Rc<dyn wasmshim_cpu::PageSource>> {
    let ino = vfs.resolve(ROOT, path, true).ok()?;
    if vfs.file_pages(ino).is_some() {
        return None;
    }
    let contents = vfs.file_contents(ino).ok()?;
    Some(contents.pages(offset, contents.len()))
}

fn baseline(vfs: &mut Vfs, v: &Vma) -> Baseline {
    match &v.kind {
        VmaKind::File { path, offset } => match file_pages(vfs, path, *offset) {
            Some(pages) => Baseline::Source(pages, 0),
            None => Baseline::Zero,
        },
        VmaKind::Shared { .. } => Baseline::Shared,
        _ => Baseline::Zero,
    }
}

fn replay(vfs: &mut Vfs, mem: &mut dyn GuestMemory, v: &Vma) -> Result<(), CkptError> {
    let len = v.end - v.start;
    let r: Result<(), MapError> = match &v.kind {
        VmaKind::File { path, offset } => match file_pages(vfs, path, *offset) {
            Some(pages) => mem.map_source(v.start, len, v.perms, pages, 0),
            None => mem.map(v.start, len, v.perms),
        },
        VmaKind::Shared { obj, offset, .. } => mem.map_shared(
            v.start,
            len,
            v.perms,
            obj.pages.clone(),
            offset / wasmshim_cpu::PAGE_SIZE,
        ),
        _ => mem.map(v.start, len, v.perms),
    };
    r.map_err(|_| CkptError::new("a mapping does not fit the restored address space"))
}

/// Saving: shared objects by address, defined where first met.
struct Saver<'k> {
    objs: Objects,
    vfs: &'k mut Vfs,
    /// Address spaces saved so far (a space's pages are keyed by its number).
    spaces: u64,
    /// Memory objects saved so far (their pages are keyed by `OBJECT_SPACES` + their number).
    obj_spaces: u64,
    /// The pages for the container, keyed `space << 48 | address` (increasing).
    pages: Vec<(u64, Vec<u8>)>,
    /// The network provider, for its records of provider sockets.
    net: Option<&'k dyn crate::net::Network>,
    /// A provider socket was saved (the provider's own record is then required).
    inet: bool,
    /// The ids of the filesystem's page-cache objects (saved ahead of it).
    files: std::collections::BTreeSet<u64>,
    /// Page-cache objects of files that are gone (unlinked while mapped: Chromium's `/dev/shm`
    /// files), by id, with the extent their mappings reach: saved with their pages.
    orphans: BTreeMap<u64, u64>,
    /// Connected local sockets and their peers, linked after every object is saved.
    peers: Vec<(
        crate::objects::unix::UnixRef,
        std::rc::Weak<RefCell<crate::objects::unix::UnixSocket>>,
    )>,
}

fn save_addr(a: &Option<std::net::SocketAddr>, w: &mut Writer) {
    match a {
        None => 0u8.save(w),
        Some(std::net::SocketAddr::V4(v)) => {
            4u8.save(w);
            v.ip().octets().save(w);
            v.port().save(w);
        }
        Some(std::net::SocketAddr::V6(v)) => {
            6u8.save(w);
            v.ip().octets().save(w);
            v.port().save(w);
            (v.flowinfo(), v.scope_id()).save(w);
        }
    }
}

/// Memory objects' pages go into the container as address spaces' do (deduplicated and
/// compressed with them), each object under a space of its own from here up.
const OBJECT_SPACES: u64 = 1 << 15;

/// A local socket's state tag with this bit: its record ends with what it has queued (bytes,
/// datagrams, connections not accepted yet). Without it, the record is as
/// before, so a socket with nothing queued saves the same bytes as it always did.
const QUEUED: u8 = 0x80;

/// Where a memory object's pages go: the container's pages, under the next object space.
struct ObjectPages<'a> {
    next: &'a mut u64,
    pages: &'a mut Vec<(u64, Vec<u8>)>,
}

/// Saves a memory object's first `len` bytes (writing its object space's number): the pages
/// that differ from `baseline` (the image's bytes, for a page cache made from a file still in
/// the image: a restore reads the rest from the image again), or without one the pages that
/// are not zero.
fn save_object_pages(
    obj: &crate::shm::MemObject,
    len: u64,
    baseline: Option<&Rc<dyn wasmshim_cpu::PageSource>>,
    out: ObjectPages<'_>,
    w: &mut Writer,
) -> Result<(), CkptError> {
    let n = *out.next;
    if n >= OBJECT_SPACES {
        return Err(unsupported("more memory objects than a checkpoint holds"));
    }
    *out.next += 1;
    n.save(w);
    let space = (OBJECT_SPACES + n) << SPACE_SHIFT;
    let count = crate::mm::page_up(len) / wasmshim_cpu::PAGE_SIZE;
    let mut page = vec![0u8; wasmshim_cpu::PAGE_SIZE as usize];
    for i in 0..count {
        obj.pages
            .read(i * wasmshim_cpu::PAGE_SIZE, &mut page)
            .map_err(|_| CkptError::new("a shared object's pages failed"))?;
        let differs = match baseline {
            Some(b) => {
                let base = b
                    .page(i)
                    .map_err(|_| CkptError::new("the image failed under a page cache"))?;
                page[..] != base[..]
            }
            None => page.iter().any(|&b| b != 0),
        };
        if differs {
            out.pages
                .push((space | (i * wasmshim_cpu::PAGE_SIZE), page.clone()));
        }
    }
    Ok(())
}

/// A new memory object (this frontend's kind) as [`save_object_pages`] saved it: `baseline`'s
/// pages (read as they are touched) with the saved ones over them.
fn load_object_pages(
    frontend: &dyn wasmshim_kernel::Frontend,
    len: u64,
    baseline: Option<(Rc<dyn wasmshim_cpu::PageSource>, u64)>,
    pages: &mut BTreeMap<u64, Vec<(u64, &[u8])>>,
    r: &mut Reader<'_>,
) -> Result<Rc<dyn wasmshim_cpu::SharedMemory>, CkptError> {
    let n = u64::load(r)?;
    if n >= OBJECT_SPACES {
        return Err(CkptError::new("a bad object space"));
    }
    let mem = frontend
        .new_shared_memory(baseline)
        .map_err(|_| CkptError::new("a shared object could not be made"))?;
    for (offset, bytes) in pages.remove(&(OBJECT_SPACES + n)).unwrap_or_default() {
        if bytes.len() as u64 != wasmshim_cpu::PAGE_SIZE || offset >= crate::mm::page_up(len) {
            return Err(CkptError::new("a bad page of a shared object"));
        }
        mem.write(offset, bytes)
            .map_err(|_| CkptError::new("a shared object's page could not be written"))?;
    }
    Ok(mem)
}

/// The baseline of a page cache made from the image at `origin`, as the object's source.
#[allow(clippy::type_complexity)]
fn origin_source(
    vfs: &Vfs,
    origin: Option<(u64, u64)>,
) -> Result<Option<(Rc<dyn wasmshim_cpu::PageSource>, u64)>, CkptError> {
    origin
        .map(|o| {
            vfs.image_pages(o)
                .map(|p| (p, crate::mm::page_up(o.1) / wasmshim_cpu::PAGE_SIZE))
                .ok_or_else(|| CkptError::new("a page cache from an image, without the image"))
        })
        .transpose()
}

fn save_sock_name(n: &Option<crate::objects::unix::SockName>, w: &mut Writer) {
    use crate::objects::unix::SockName;
    match n {
        None => 0u8.save(w),
        Some(SockName::Unix(p)) => {
            1u8.save(w);
            w.bytes(p);
        }
        Some(SockName::Inet(a)) => {
            2u8.save(w);
            save_addr(&Some(*a), w);
        }
    }
}

fn load_sock_name(r: &mut Reader<'_>) -> Result<Option<crate::objects::unix::SockName>, CkptError> {
    use crate::objects::unix::SockName;
    Ok(match u8::load(r)? {
        0 => None,
        1 => Some(SockName::Unix(r.bytes()?.to_vec())),
        2 => Some(SockName::Inet(
            load_addr(r)?.ok_or_else(|| CkptError::new("bad socket name"))?,
        )),
        _ => return Err(CkptError::new("bad socket name")),
    })
}

fn save_unix_name(n: &Option<crate::objects::unix::UnixName>, w: &mut Writer) {
    use crate::objects::unix::UnixName;
    match n {
        None => 0u8.save(w),
        Some(UnixName::Ino(i)) => (1u8, *i).save(w),
        Some(UnixName::Abstract(b)) => {
            2u8.save(w);
            w.bytes(b);
        }
        Some(UnixName::Port(stream, v6, port)) => {
            3u8.save(w);
            (*stream, *v6, *port).save(w);
        }
    }
}

fn load_unix_name(r: &mut Reader<'_>) -> Result<Option<crate::objects::unix::UnixName>, CkptError> {
    use crate::objects::unix::UnixName;
    Ok(match u8::load(r)? {
        0 => None,
        1 => Some(UnixName::Ino(u64::load(r)?)),
        2 => Some(UnixName::Abstract(r.bytes()?.to_vec())),
        3 => {
            let (stream, v6, port) = Ckpt::load(r)?;
            Some(UnixName::Port(stream, v6, port))
        }
        _ => return Err(CkptError::new("bad socket registration")),
    })
}

fn load_addr(r: &mut Reader<'_>) -> Result<Option<std::net::SocketAddr>, CkptError> {
    use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
    Ok(match u8::load(r)? {
        0 => None,
        4 => {
            let ip: [u8; 4] = Ckpt::load(r)?;
            Some(SocketAddr::V4(SocketAddrV4::new(
                Ipv4Addr::from(ip),
                u16::load(r)?,
            )))
        }
        6 => {
            let ip: [u8; 16] = Ckpt::load(r)?;
            let port = u16::load(r)?;
            let (flow, scope) = Ckpt::load(r)?;
            Some(SocketAddr::V6(SocketAddrV6::new(
                Ipv6Addr::from(ip),
                port,
                flow,
                scope,
            )))
        }
        _ => return Err(CkptError::new("bad socket address")),
    })
}

/// Page keys: an address space's number above the 48-bit guest address.
const SPACE_SHIFT: u32 = 48;

impl Saver<'_> {
    /// Saves a reference to `rc`; the first time also its contents (`body`).
    fn rc<T: ?Sized>(
        &mut self,
        w: &mut Writer,
        rc: &Rc<T>,
        body: impl FnOnce(&mut Self, &mut Writer) -> Result<(), CkptError>,
    ) -> Result<(), CkptError> {
        let (index, first) = self.objs.index(Rc::as_ptr(rc) as *const () as usize);
        index.save(w);
        first.save(w);
        if first {
            body(self, w)?;
        }
        Ok(())
    }

    fn file(&mut self, w: &mut Writer, f: &FileRef) -> Result<(), CkptError> {
        self.rc(w, f, |s, w| {
            let of = f.borrow();
            s.kind(w, &of.kind)?;
            of.flags.save(w);
            of.offset.save(w);
            of.path.save(w);
            of.owner.save(w);
            of.dir_cursor.save(w);
            Ok(())
        })
    }

    fn kind(&mut self, w: &mut Writer, k: &FileKind) -> Result<(), CkptError> {
        match k {
            FileKind::Stdio(s) => {
                0u8.save(w);
                (*s as u8).save(w);
            }
            FileKind::Inode { ino, device } => {
                1u8.save(w);
                ino.save(w);
                device.save(w);
            }
            FileKind::Pipe { pipe, write } => {
                2u8.save(w);
                write.save(w);
                self.rc(w, pipe, |_, w| {
                    let p = pipe.borrow();
                    p.id.save(w);
                    p.generation.save(w);
                    p.buf.save(w);
                    p.readers.save(w);
                    p.writers.save(w);
                    Ok(())
                })?;
            }
            FileKind::Proc { data } => {
                3u8.save(w);
                w.bytes(data);
            }
            FileKind::ProcDir { path } => {
                4u8.save(w);
                path.save(w);
            }
            FileKind::EventFd(e) => {
                5u8.save(w);
                self.rc(w, e, |_, w| {
                    let e = e.borrow();
                    ((e.id, e.count), (e.semaphore, e.generation)).save(w);
                    Ok(())
                })?;
            }
            FileKind::TimerFd(t) => {
                6u8.save(w);
                self.rc(w, t, |_, w| {
                    let t = t.borrow();
                    (t.id, t.realtime, t.next).save(w);
                    (t.interval, t.ticks, t.total).save(w);
                    Ok(())
                })?;
            }
            FileKind::Epoll(e) => {
                7u8.save(w);
                self.rc(w, e, |s, w| {
                    let e = e.borrow();
                    e.id.save(w);
                    (e.interests.len() as u64).save(w);
                    for (fd, i) in &e.interests {
                        fd.save(w);
                        match i.file.upgrade() {
                            Some(f) => {
                                true.save(w);
                                s.file(w, &f)?;
                            }
                            None => false.save(w),
                        }
                        (i.events, i.data, i.last_gen).save(w);
                        i.disabled.save(w);
                    }
                    Ok(())
                })?;
            }
            FileKind::SignalFd(sfd) => {
                8u8.save(w);
                self.rc(w, sfd, |_, w| {
                    sfd.borrow().mask.save(w);
                    Ok(())
                })?;
            }
            FileKind::PidFd { pid } => {
                9u8.save(w);
                pid.save(w);
            }
            FileKind::Inotify(i) => {
                13u8.save(w);
                self.rc(w, i, |_, w| {
                    let i = i.borrow();
                    (i.id, i.generation, i.next_wd).save(w);
                    i.watches.save(w);
                    i.queue.save(w);
                    i.overflowed.save(w);
                    Ok(())
                })?;
            }
            FileKind::MQueue(_) => return Err(unsupported("a message queue")),
            FileKind::Unix(u) => {
                11u8.save(w);
                self.unix_socket(w, u)?;
            }
            FileKind::Netlink(n) => {
                12u8.save(w);
                self.rc(w, n, |_, w| {
                    let n = n.borrow();
                    (n.id, n.port, n.groups).save(w);
                    n.queue.save(w);
                    n.generation.save(w);
                    Ok(())
                })?;
            }
            // A provider socket that is idle or listening, with nothing queued;
            // the provider adds its own record (which transport sockets stand behind it).
            FileKind::Inet(sock) => {
                10u8.save(w);
                self.net_socket(w, sock)?;
            }
            FileKind::Tty { .. } => return Err(unsupported("a terminal")),
            FileKind::HidRaw(h) => {
                14u8.save(w);
                self.rc(w, h, |_, w| {
                    let h = h.borrow();
                    (h.id, h.dev, h.ino).save(w);
                    h.queue.save(w);
                    h.generation.save(w);
                    Ok(())
                })?;
            }
        }
        Ok(())
    }

    /// A provider socket: idle or listening with nothing queued, plus the
    /// provider's record of it.
    fn net_socket(&mut self, w: &mut Writer, sock: &crate::net::NetRef) -> Result<(), CkptError> {
        self.rc(w, sock, |s, w| {
            let n = sock.borrow();
            if let Some(what) = n.not_quiescent() {
                return Err(unsupported(what));
            }
            s.inet = true;
            let net = s
                .net
                .ok_or_else(|| unsupported("a socket without a network"))?;
            let record = net
                .checkpoint(sock)
                .map_err(|e| CkptError(format!("cannot checkpoint a socket: {e}")))?;
            let proto: u8 = match n.proto {
                crate::net::Proto::Tcp => 0,
                crate::net::Proto::Udp => 1,
                crate::net::Proto::Icmp => 2,
            };
            (n.id, proto).save(w);
            (n.v6, n.v6only, n.state == crate::net::NetState::Listening).save(w);
            save_addr(&n.local, w);
            save_addr(&n.remote, w);
            (n.rx_cap as u64, n.tx_cap as u64).save(w);
            (n.shut_wr, n.shut_rd, n.linger_zero).save(w);
            (n.rcvtimeo, n.sndtimeo).save(w);
            (n.backlog as u64, n.dgrams_dropped).save(w);
            (n.nodelay, n.keepalive).save(w);
            n.opts.save(w);
            n.generation.save(w);
            w.bytes(&record);
            Ok(())
        })
    }

    /// A local socket (AF_UNIX, or an AF_INET/AF_INET6 one on the in-kernel loopback with
    /// its network half): unconnected or listening, nothing queued. Its bound
    /// name is registered again on restore.
    fn unix_socket(
        &mut self,
        w: &mut Writer,
        u: &crate::objects::unix::UnixRef,
    ) -> Result<(), CkptError> {
        use crate::objects::unix::UnixState;
        self.rc(w, u, |s, w| {
            let b = u.borrow();
            // (state, backlog): unconnected, listening, connected (its peer is linked after
            // every object is saved: the two ends refer to each other). With `QUEUED` set, the
            // record ends with what is queued: stream bytes and datagrams not
            // read yet, and a listener's connections not accepted yet. A socket with nothing
            // queued has no `QUEUED` flag and no such trailer.
            let queued = !b.rx.is_empty()
                || !b.msgs.is_empty()
                || matches!(&b.state, UnixState::Listening { queue, .. } if !queue.is_empty());
            let state: (u8, u64) = match &b.state {
                UnixState::Unconnected => (0, 0),
                UnixState::Listening { backlog, .. } => (1, *backlog as u64),
                UnixState::Connected => {
                    s.peers.push((u.clone(), b.peer.clone()));
                    (2, 0)
                }
            };
            if !b.rx_fds.is_empty() || !b.msg_fds.is_empty() {
                return Err(unsupported("a local socket with descriptions in flight"));
            }
            let state = (state.0 | if queued { QUEUED } else { 0 }, state.1);
            (b.id, b.ty.raw(), b.family).save(w);
            state.save(w);
            save_sock_name(&b.name, w);
            save_unix_name(&b.reg, w);
            save_sock_name(&b.peer_name, w);
            (b.rx_read, (b.read_shut, b.write_shut, b.peer_gone)).save(w);
            (b.owner, b.v6only, b.generation).save(w);
            (b.rcvtimeo, b.sndtimeo, b.nodelay).save(w);
            b.opts.save(w);
            (b.reset, b.reset_pending).save(w);
            match &b.net {
                Some(n) => {
                    true.save(w);
                    s.net_socket(w, n)?;
                }
                None => false.save(w),
            }
            if queued {
                let (a, z) = b.rx.as_slices();
                w.bytes(&[a, z].concat());
                (b.msgs.len() as u64).save(w);
                for (data, from) in &b.msgs {
                    w.bytes(data);
                    save_sock_name(from, w);
                }
                let queue: Vec<crate::objects::unix::UnixRef> = match &b.state {
                    UnixState::Listening { queue, .. } => queue.iter().cloned().collect(),
                    _ => Vec::new(),
                };
                drop(b);
                (queue.len() as u64).save(w);
                for q in &queue {
                    s.unix_socket(w, q)?;
                }
            }
            Ok(())
        })
    }

    /// A shared memory object: an anonymous one (`MAP_SHARED | MAP_ANONYMOUS`, a
    /// fixed size) by identity, with its pages that are not zero. A file's page cache, a memfd
    /// or a `/dev/shm` file (no fixed size) is refused.
    fn mem_object(&mut self, w: &mut Writer, obj: &crate::shm::MemRef) -> Result<(), CkptError> {
        self.rc(w, obj, |s, w| match obj.size {
            // A file's page cache or a memfd: saved with the filesystem, named by id.
            None if s.files.contains(&obj.id) => {
                (1u8, obj.id).save(w);
                Ok(())
            }
            None if s.orphans.contains_key(&obj.id) => {
                let extent = s.orphans[&obj.id];
                (2u8, obj.id).save(w);
                extent.save(w);
                let origin = s.vfs.page_origin(obj.id);
                let base = origin.and_then(|o| s.vfs.image_pages(o));
                origin.filter(|_| base.is_some()).save(w);
                let out = ObjectPages {
                    next: &mut s.obj_spaces,
                    pages: &mut s.pages,
                };
                save_object_pages(obj, extent, base.as_ref(), out, w)
            }
            None => Err(unsupported(
                "a shared mapping of a file outside the filesystem",
            )),
            Some(size) => {
                (0u8, obj.id).save(w);
                size.save(w);
                let out = ObjectPages {
                    next: &mut s.obj_spaces,
                    pages: &mut s.pages,
                };
                save_object_pages(obj, size, None, out, w)
            }
        })
    }

    fn process(&mut self, w: &mut Writer, p: &Process) -> Result<(), CkptError> {
        (p.ppid, p.pgid, p.sid).save(w);
        self.rc(w, &p.mm, |s, w| {
            let mm = p.mm.borrow();
            // The shared objects it maps: defined before the mappings that name them.
            let objs: Vec<crate::shm::MemRef> = mm
                .vmas()
                .filter_map(|v| match &v.kind {
                    VmaKind::Shared { obj, .. } => Some(obj.clone()),
                    _ => None,
                })
                .collect();
            (objs.len() as u64).save(w);
            for obj in &objs {
                s.mem_object(w, obj)?;
            }
            let vfs = &mut *s.vfs;
            let space = s.spaces << SPACE_SHIFT;
            s.spaces += 1;
            let pages = &mut s.pages;
            mm.save(w, &mut |v| baseline(vfs, v), &mut |addr, page| {
                pages.push((space | addr, page.to_vec()))
            })
        })?;
        self.rc(w, &p.fds, |s, w| {
            let fds = p.fds.borrow();
            (fds.iter().count() as u64).save(w);
            for (fd, slot) in fds.iter() {
                fd.save(w);
                slot.cloexec.save(w);
                s.file(w, &slot.file)?;
            }
            Ok(())
        })?;
        p.cwd.save(w);
        p.umask.save(w);
        self.rc(w, &p.sigactions, |_, w| {
            for a in p.sigactions.borrow().iter() {
                save_sigaction(a, w);
            }
            Ok(())
        })?;
        for b in [&p.exe, &p.comm, &p.cmdline, &p.environ, &p.auxv] {
            b.save(w);
        }
        p.threads.save(w);
        p.children.save(w);
        p.exit_signal.save(w);
        p.pending.save(w);
        p.vfork_parent.save(w);
        p.creds.save(w);
        p.itimers.save(w);
        p.timers.save(w);
        p.semundo.save(w);
        Ok(())
    }
}

/// Loading: the objects in the order they were defined.
struct Loader<'k> {
    objs: Vec<Option<Rc<dyn Any>>>,
    vfs: &'k mut Vfs,
    frontend: &'k dyn wasmshim_kernel::Frontend,
    /// Address spaces loaded so far.
    spaces: u64,
    /// The container's pages by address space number: (address, bytes).
    pages: BTreeMap<u64, Vec<(u64, &'k [u8])>>,
    /// Provider sockets loaded so far, with their provider records.
    inet: Vec<(crate::net::NetRef, Vec<u8>)>,
    /// Bound socket names to register again.
    names: Vec<(
        crate::objects::unix::UnixName,
        std::rc::Weak<RefCell<crate::objects::unix::UnixSocket>>,
    )>,
    /// How many object definitions enclose the one being read.
    depth: u32,
}

/// The deepest nesting of object definitions a restore reads. A machine's own state nests a few
/// levels (a descriptor table, an epoll set's interests at most `EP_MAX_NESTS` deep, a
/// listening socket's queue); a body nested deeper is malformed, and reading it recursively
/// could exhaust the stack.
const MAX_OBJECT_DEPTH: u32 = 32;

impl Loader<'_> {
    fn rc<T: 'static>(
        &mut self,
        r: &mut Reader<'_>,
        body: impl FnOnce(&mut Self, &mut Reader<'_>) -> Result<Rc<T>, CkptError>,
    ) -> Result<Rc<T>, CkptError> {
        let index = u64::load(r)?;
        let first = bool::load(r)?;
        let index = usize::try_from(index).map_err(|_| CkptError::new("bad object index"))?;
        if first {
            if index != self.objs.len() {
                return Err(CkptError::new("objects out of order"));
            }
            if self.depth >= MAX_OBJECT_DEPTH {
                return Err(CkptError::new("objects nested too deep"));
            }
            // Reserve the slot: objects defined inside come after it.
            self.objs.push(None);
            self.depth += 1;
            let v = body(self, r);
            self.depth -= 1;
            let v = v?;
            self.objs[index] = Some(v.clone());
            Ok(v)
        } else {
            self.objs
                .get(index)
                .cloned()
                .flatten()
                .and_then(|o| o.downcast::<T>().ok())
                .ok_or_else(|| CkptError::new("a reference to a missing or different object"))
        }
    }

    fn file(&mut self, r: &mut Reader<'_>) -> Result<FileRef, CkptError> {
        self.rc(r, |s, r| {
            let kind = s.kind(r)?;
            Ok(Rc::new(RefCell::new(OpenFile {
                kind,
                flags: Ckpt::load(r)?,
                offset: Ckpt::load(r)?,
                path: Ckpt::load(r)?,
                owner: Ckpt::load(r)?,
                dir_cursor: Ckpt::load(r)?,
            })))
        })
    }

    fn kind(&mut self, r: &mut Reader<'_>) -> Result<FileKind, CkptError> {
        Ok(match u8::load(r)? {
            0 => FileKind::Stdio(match u8::load(r)? {
                0 => Stdio::In,
                1 => Stdio::Out,
                2 => Stdio::Err,
                _ => return Err(CkptError::new("bad stdio stream")),
            }),
            1 => FileKind::Inode {
                ino: Ckpt::load(r)?,
                device: Ckpt::load(r)?,
            },
            2 => {
                let write = bool::load(r)?;
                let pipe = self.rc(r, |_, r| {
                    Ok(Rc::new(RefCell::new(Pipe {
                        id: Ckpt::load(r)?,
                        generation: Ckpt::load(r)?,
                        buf: Ckpt::load(r)?,
                        readers: Ckpt::load(r)?,
                        writers: Ckpt::load(r)?,
                    })))
                })?;
                FileKind::Pipe { pipe, write }
            }
            3 => FileKind::Proc {
                data: Rc::new(r.bytes()?.to_vec()),
            },
            4 => FileKind::ProcDir {
                path: Ckpt::load(r)?,
            },
            5 => FileKind::EventFd(self.rc(r, |_, r| {
                let ((id, count), (semaphore, generation)) = Ckpt::load(r)?;
                Ok(Rc::new(RefCell::new(EventFd {
                    id,
                    count,
                    semaphore,
                    generation,
                })))
            })?),
            6 => FileKind::TimerFd(self.rc(r, |_, r| {
                let (id, realtime, next) = Ckpt::load(r)?;
                let (interval, ticks, total) = Ckpt::load(r)?;
                Ok(Rc::new(RefCell::new(TimerFd {
                    id,
                    realtime,
                    next,
                    interval,
                    ticks,
                    total,
                })))
            })?),
            7 => FileKind::Epoll(self.rc(r, |s, r| {
                let id = u64::load(r)?;
                let n = u64::load(r)?;
                let mut interests = BTreeMap::new();
                for _ in 0..n {
                    let fd = i32::load(r)?;
                    let file = match bool::load(r)? {
                        true => Rc::downgrade(&s.file(r)?),
                        false => std::rc::Weak::new(),
                    };
                    let (events, data, last_gen) = Ckpt::load(r)?;
                    let disabled = Ckpt::load(r)?;
                    interests.insert(
                        fd,
                        Interest {
                            file,
                            events,
                            data,
                            last_gen,
                            disabled,
                        },
                    );
                }
                Ok(Rc::new(RefCell::new(Epoll { id, interests })))
            })?),
            8 => FileKind::SignalFd(self.rc(r, |_, r| {
                Ok(Rc::new(RefCell::new(SignalFd {
                    mask: Ckpt::load(r)?,
                })))
            })?),
            9 => FileKind::PidFd {
                pid: Ckpt::load(r)?,
            },
            10 => FileKind::Inet(self.net_socket(r)?),
            11 => FileKind::Unix(self.unix_socket(r)?),
            12 => FileKind::Netlink(self.rc(r, |_, r| {
                let (id, port, groups) = Ckpt::load(r)?;
                Ok(Rc::new(RefCell::new(crate::objects::netlink::Netlink {
                    id,
                    port,
                    groups,
                    queue: Ckpt::load(r)?,
                    generation: Ckpt::load(r)?,
                })))
            })?),
            14 => FileKind::HidRaw(self.rc(r, |_, r| {
                let (id, dev, ino) = Ckpt::load(r)?;
                Ok(Rc::new(RefCell::new(crate::devices::HidOpen {
                    id,
                    dev,
                    ino,
                    queue: Ckpt::load(r)?,
                    generation: Ckpt::load(r)?,
                })))
            })?),
            13 => FileKind::Inotify(self.rc(r, |_, r| {
                let (id, generation, next_wd) = Ckpt::load(r)?;
                Ok(Rc::new(RefCell::new(crate::objects::inotify::Inotify {
                    id,
                    generation,
                    next_wd,
                    watches: Ckpt::load(r)?,
                    queue: Ckpt::load(r)?,
                    overflowed: Ckpt::load(r)?,
                })))
            })?),
            _ => return Err(CkptError::new("bad file kind")),
        })
    }

    fn net_socket(&mut self, r: &mut Reader<'_>) -> Result<crate::net::NetRef, CkptError> {
        self.rc(r, |s, r| {
            use crate::net::{NetSocket, NetState, Proto};
            let (id, proto): (u64, u8) = Ckpt::load(r)?;
            let (v6, v6only, listening): (bool, bool, bool) = Ckpt::load(r)?;
            let proto = match proto {
                0 => Proto::Tcp,
                1 => Proto::Udp,
                2 => Proto::Icmp,
                _ => return Err(CkptError::new("bad socket protocol")),
            };
            let mut n = NetSocket::new(id, proto, v6);
            n.v6only = v6only;
            n.state = if listening {
                NetState::Listening
            } else {
                NetState::Idle
            };
            n.local = load_addr(r)?;
            n.remote = load_addr(r)?;
            let (rx_cap, tx_cap): (u64, u64) = Ckpt::load(r)?;
            (n.rx_cap, n.tx_cap) = (rx_cap as usize, tx_cap as usize);
            (n.shut_wr, n.shut_rd, n.linger_zero) = Ckpt::load(r)?;
            (n.rcvtimeo, n.sndtimeo) = Ckpt::load(r)?;
            let (backlog, dropped): (u64, u64) = Ckpt::load(r)?;
            (n.backlog, n.dgrams_dropped) = (backlog as usize, dropped);
            (n.nodelay, n.keepalive) = Ckpt::load(r)?;
            n.opts = Ckpt::load(r)?;
            n.generation = Ckpt::load(r)?;
            let record = r.bytes()?.to_vec();
            let sock = n.shared();
            s.inet.push((sock.clone(), record));
            Ok(sock)
        })
    }

    fn unix_socket(
        &mut self,
        r: &mut Reader<'_>,
    ) -> Result<crate::objects::unix::UnixRef, CkptError> {
        use crate::objects::unix::{SockType, UnixSocket, UnixState};
        self.rc(r, |s, r| {
            let (id, ty, family): (u64, u32, u16) = Ckpt::load(r)?;
            let ty = match ty {
                1 => SockType::Stream,
                2 => SockType::Dgram,
                5 => SockType::SeqPacket,
                _ => return Err(CkptError::new("bad socket type")),
            };
            let (state, backlog): (u8, u64) = Ckpt::load(r)?;
            let name = load_sock_name(r)?;
            let reg = load_unix_name(r)?;
            let peer_name = load_sock_name(r)?;
            let (rx_read, (read_shut, write_shut, peer_gone)) = Ckpt::load(r)?;
            let (owner, v6only, generation) = Ckpt::load(r)?;
            let (rcvtimeo, sndtimeo, nodelay) = Ckpt::load(r)?;
            let opts = Ckpt::load(r)?;
            let (reset, reset_pending) = Ckpt::load(r)?;
            let net = if bool::load(r)? {
                Some(s.net_socket(r)?)
            } else {
                None
            };
            let mut u = UnixSocket::new(id, family, ty, owner);
            let mut queue = std::collections::VecDeque::new();
            if state & QUEUED != 0 {
                u.rx = r.bytes()?.iter().copied().collect();
                for _ in 0..u64::load(r)? {
                    let data = r.bytes()?.to_vec();
                    u.msgs.push_back((data, load_sock_name(r)?));
                }
                for _ in 0..u64::load(r)? {
                    queue.push_back(s.unix_socket(r)?);
                }
            }
            u.state = match state & !QUEUED {
                0 => UnixState::Unconnected,
                1 => UnixState::Listening {
                    backlog: backlog as usize,
                    queue,
                },
                2 => UnixState::Connected,
                _ => return Err(CkptError::new("bad local socket state")),
            };
            u.name = name;
            u.reg = reg.clone();
            u.peer_name = peer_name;
            (u.rx_read, u.read_shut, u.write_shut, u.peer_gone) =
                (rx_read, read_shut, write_shut, peer_gone);
            (u.v6only, u.generation) = (v6only, generation);
            (u.rcvtimeo, u.sndtimeo, u.nodelay) = (rcvtimeo, sndtimeo, nodelay);
            u.opts = opts;
            (u.reset, u.reset_pending) = (reset, reset_pending);
            u.net = net;
            let u = Rc::new(RefCell::new(u));
            if let Some(key) = reg {
                s.names.push((key, Rc::downgrade(&u)));
            }
            Ok(u)
        })
    }

    fn mem_object(&mut self, r: &mut Reader<'_>) -> Result<crate::shm::MemRef, CkptError> {
        self.rc(r, |s, r| {
            let (tag, id): (u8, u64) = Ckpt::load(r)?;
            if tag == 1 {
                return crate::vfs::ckpt::RESTORED_PAGES
                    .with(|m| m.borrow().get(&id).cloned())
                    .ok_or_else(|| CkptError::new("a shared mapping of a file not saved"));
            }
            if tag == 2 {
                let extent = u64::load(r)?;
                let origin: Option<(u64, u64)> = Ckpt::load(r)?;
                let source = origin_source(s.vfs, origin)?;
                let pages = load_object_pages(s.frontend, extent, source, &mut s.pages, r)?;
                if let Some(o) = origin {
                    s.vfs.set_page_origin(id, o);
                }
                return Ok(Rc::new(crate::shm::MemObject {
                    id,
                    pages,
                    size: None,
                }));
            }
            let size = u64::load(r)?;
            let pages = load_object_pages(s.frontend, size, None, &mut s.pages, r)?;
            Ok(Rc::new(crate::shm::MemObject {
                id,
                pages,
                size: Some(size),
            }))
        })
    }

    fn process(&mut self, r: &mut Reader<'_>) -> Result<Process, CkptError> {
        let (ppid, pgid, sid) = Ckpt::load(r)?;
        let mm = self.rc(r, |s, r| {
            LOADING_OBJS.with(|m| m.borrow_mut().clear());
            for _ in 0..u64::load(r)? {
                let obj = s.mem_object(r)?;
                LOADING_OBJS.with(|m| m.borrow_mut().insert(obj.id, obj));
            }
            let vfs = &mut *s.vfs;
            let pages = s.pages.remove(&s.spaces).unwrap_or_default();
            s.spaces += 1;
            let mm = AddressSpace::load(
                r,
                s.frontend.new_memory(),
                &mut |mem, v| replay(vfs, mem, v),
                &pages,
            )?;
            LOADING_OBJS.with(|m| m.borrow_mut().clear());
            Ok(Rc::new(RefCell::new(mm)))
        })?;
        let fds = self.rc(r, |s, r| {
            let mut fds = FdTable::new();
            let n = u64::load(r)?;
            for _ in 0..n {
                let fd = i32::load(r)?;
                let cloexec = bool::load(r)?;
                let file = s.file(r)?;
                fds.insert_at(fd, file, cloexec)
                    .map_err(|_| CkptError::new("bad descriptor"))?;
            }
            Ok(Rc::new(RefCell::new(fds)))
        })?;
        let cwd = Ckpt::load(r)?;
        let umask = Ckpt::load(r)?;
        let sigactions = self.rc(r, |_, r| {
            let mut a = [KernelSigaction::default(); 64];
            for s in &mut a {
                *s = load_sigaction(r)?;
            }
            Ok(Rc::new(RefCell::new(a)))
        })?;
        Ok(Process {
            ppid,
            pgid,
            sid,
            mm,
            fds,
            cwd,
            umask,
            sigactions,
            exe: Ckpt::load(r)?,
            comm: Ckpt::load(r)?,
            cmdline: Ckpt::load(r)?,
            environ: Ckpt::load(r)?,
            auxv: Ckpt::load(r)?,
            threads: Ckpt::load(r)?,
            children: Ckpt::load(r)?,
            exit_signal: Ckpt::load(r)?,
            pending: Ckpt::load(r)?,
            vfork_parent: Ckpt::load(r)?,
            creds: Ckpt::load(r)?,
            itimers: Ckpt::load(r)?,
            timers: Ckpt::load(r)?,
            semundo: Ckpt::load(r)?,
        })
    }
}

impl Kernel {
    /// Saves the whole machine as a snapshot container (kind
    /// [`SNAPSHOT_KIND`]): the filesystem, threads, processes and their objects in the body,
    /// the address spaces' stored pages as content-addressed pages. Fails, and changes
    /// nothing, if the machine holds something not checkpointed yet. The embedder adds its own
    /// `meta` (which image) before encoding.
    pub fn checkpoint(&mut self) -> Result<Snapshot, CkptError> {
        self.refuse_unsupported()?;
        let mut w = Writer::new();
        write_header(&mut w);
        // Page-cache objects (files mapped shared, memfds) ahead of the filesystem,
        // whose files and the shared mappings of them name them by id.
        let mut files = self.vfs.page_objects();
        files.sort_by_key(|(m, _)| m.id);
        files.dedup_by_key(|(m, _)| m.id);
        // A page cache made from a file still in the image keeps only the pages that differ
        // from the image (a restore reads the rest from it again).
        let (mut obj_spaces, mut obj_pages) = (0, Vec::new());
        (files.len() as u64).save(&mut w);
        for (mem, len) in &files {
            let origin = self.vfs.page_origin(mem.id);
            let base = origin.and_then(|o| self.vfs.image_pages(o));
            (mem.id, *len, origin.filter(|_| base.is_some())).save(&mut w);
            let out = ObjectPages {
                next: &mut obj_spaces,
                pages: &mut obj_pages,
            };
            save_object_pages(mem, *len, base.as_ref(), out, &mut w)?;
        }
        self.vfs.save(&mut w)?;
        self.core.save(&mut w);
        (self.next_pid, self.next_ephemeral).save(&mut w);
        self.init_exit.save(&mut w);
        self.sig_generation.save(&mut w);
        for (seq, max) in [
            (self.shm.next_seq, self.shm.max),
            (self.sem.next_seq, self.sem.max),
            (self.msg.next_seq, self.msg.max),
        ] {
            (seq, max).save(&mut w);
        }
        self.zombies.save(&mut w);
        // AIO contexts (version 2): by address space and ring address; their rings are
        // ordinary anonymous memory, saved with the pages.
        (self.aio.len() as u64).save(&mut w);
        for ((space, ring), c) in &self.aio {
            (*space, *ring).save(&mut w);
            (c.id, c.nr, c.ring_len).save(&mut w);
            c.done.save(&mut w);
        }
        let orphans = self.orphan_objects(&files);
        let mut s = Saver {
            objs: Objects::default(),
            vfs: &mut self.vfs,
            spaces: 0,
            pages: obj_pages,
            obj_spaces,
            files: files.iter().map(|(m, _)| m.id).collect(),
            orphans,
            net: self.net.as_deref(),
            inet: false,
            peers: Vec::new(),
        };
        (self.procs.len() as u64).save(&mut w);
        for (pid, p) in &self.procs {
            pid.save(&mut w);
            s.process(&mut w, p)?;
        }
        // Connected local sockets and their peers, by object index: a peer nobody
        // holds any more is gone (none), one held only in flight cannot be saved.
        let ptr = |u: &crate::objects::unix::UnixRef| Rc::as_ptr(u) as *const () as usize;
        (s.peers.len() as u64).save(&mut w);
        for (u, peer) in &s.peers {
            let me = s
                .objs
                .get(ptr(u))
                .ok_or_else(|| CkptError::new("a connected socket that was not saved"))?;
            let other = match peer.upgrade() {
                None => None,
                Some(p) => Some(
                    s.objs
                        .get(ptr(&p))
                        .ok_or_else(|| unsupported("a socket connected to one in flight"))?,
                ),
            };
            (me, other).save(&mut w);
        }
        // SysV shared memory segments by slot (Xvfb's MIT-SHM): their objects by identity, the
        // same as the mappings of them (attachments are counted from the mappings).
        (self.shm.objs.len() as u64).save(&mut w);
        for (slot, seg) in &self.shm.objs {
            let p = &seg.perm;
            (*slot, (p.key, p.uid, p.gid), (p.cuid, p.cgid, p.mode)).save(&mut w);
            (p.seq, seg.size).save(&mut w);
            s.mem_object(&mut w, &seg.obj)?;
            (seg.atime, seg.dtime, seg.ctime).save(&mut w);
            (seg.cpid, seg.lpid).save(&mut w);
        }
        // inotify's watchers by inode, in delivery order, as (instance, watch descriptor).
        (self.inotify_watches.len() as u64).save(&mut w);
        for (ino, list) in &self.inotify_watches {
            let live: Vec<(u64, i32)> = list
                .iter()
                .filter_map(|(i, wd)| i.upgrade().map(|i| (i, *wd)))
                .map(|(i, wd)| {
                    s.objs
                        .get(Rc::as_ptr(&i) as *const () as usize)
                        .map(|idx| (idx, wd))
                        .ok_or_else(|| unsupported("an inotify instance nobody holds"))
                })
                .collect::<Result<_, _>>()?;
            ino.save(&mut w);
            live.save(&mut w);
        }
        // File locks, in table order; a description's lock names the description's object.
        (self.locks.locks.len() as u64).save(&mut w);
        for l in &self.locks.locks {
            use crate::locks::Owner;
            let desc = |id: usize| {
                s.objs
                    .get(id)
                    .ok_or_else(|| unsupported("a lock of a description nobody holds"))
            };
            let (tag, who) = match l.owner {
                Owner::Posix(pid) => (0u8, u64::from(pid)),
                Owner::Ofd(id) => (1, desc(id)?),
                Owner::Flock(id) => (2, desc(id)?),
            };
            (l.ino, (tag, who), (l.start, l.end, l.write)).save(&mut w);
        }
        // Display and input devices: the HID devices with their
        // open descriptions, the framebuffers with their memory (by identity: the mappings of
        // `/dev/fb<n>` share it) and their vblank clocks, and frames not taken yet.
        (self.hid.len() as u64).save(&mut w);
        for d in &self.hid {
            let i = &d.info;
            (i.name.as_bytes().to_vec(), i.phys.as_bytes().to_vec()).save(&mut w);
            (i.bus, i.vendor, i.product).save(&mut w);
            i.descriptor.save(&mut w);
            let opens: Vec<u64> = d
                .opens
                .iter()
                .filter_map(std::rc::Weak::upgrade)
                .map(|o| {
                    s.objs
                        .get(Rc::as_ptr(&o) as *const () as usize)
                        .ok_or_else(|| unsupported("a hidraw description nobody holds"))
                })
                .collect::<Result<_, _>>()?;
            opens.save(&mut w);
        }
        (self.fbs.len() as u64).save(&mut w);
        for fb in &self.fbs {
            let m = fb.mode;
            (fb.id, (m.width, m.height), (m.refresh_num, m.refresh_den)).save(&mut w);
            (fb.stride, fb.size).save(&mut w);
            s.mem_object(&mut w, &fb.mem)?;
            w.bytes(&fb.shown);
            (fb.blank, fb.full_next).save(&mut w);
            (fb.t0, fb.next).save(&mut w);
        }
        (self.fb_frames.len() as u64).save(&mut w);
        for f in &self.fb_frames {
            (f.index, f.at, f.vblank).save(&mut w);
            (f.y, f.height).save(&mut w);
            w.bytes(&f.pixels);
        }
        // The provider's own record, required once a provider socket was saved.
        let provider = match s.net.map(|n| n.checkpoint_provider()) {
            Some(Ok(b)) => Some(b),
            Some(Err(e)) if s.inet => {
                return Err(CkptError(format!("cannot checkpoint the network: {e}")));
            }
            _ => None,
        };
        provider.save(&mut w);
        let mut snap = Snapshot::new_in(&self.cfg.snapshot_format, SNAPSHOT_KIND, SNAPSHOT_VERSION);
        snap.body = w.into_bytes();
        s.pages.sort_by_key(|(k, _)| *k);
        for (key, page) in &s.pages {
            snap.add_page(*key, page)
                .map_err(|e| CkptError(format!("snapshot pages: {e:?}")))?;
        }
        Ok(snap)
    }

    /// After [`Kernel::restore`] and [`Kernel::set_network`]: hands the restored provider
    /// sockets back to the provider, its own record first, in the order they were saved. A
    /// machine restored without provider sockets needs no call.
    pub fn reattach_network(&mut self) -> Result<(), CkptError> {
        let (provider, socks) = std::mem::take(&mut self.net_restored);
        if provider.is_none() && socks.is_empty() {
            return Ok(());
        }
        let net = self
            .net
            .as_mut()
            .ok_or_else(|| CkptError::new("restored sockets need a network provider"))?;
        if let Some(p) = provider {
            net.restore_provider(&p)
                .map_err(|e| CkptError(format!("restoring the network: {e}")))?;
        }
        for (sock, record) in socks {
            net.restore(&sock, &record)
                .map_err(|e| CkptError(format!("restoring a socket: {e}")))?;
        }
        Ok(())
    }

    /// The page-cache objects that shared mappings hold but no file does any more (unlinked
    /// while mapped), by id, with the furthest byte a mapping of them reaches.
    fn orphan_objects(&self, files: &[(crate::shm::MemRef, u64)]) -> BTreeMap<u64, u64> {
        let named: std::collections::BTreeSet<u64> = files.iter().map(|(m, _)| m.id).collect();
        let mut out = BTreeMap::new();
        for p in self.procs.values() {
            for v in p.mm.borrow().vmas() {
                if let VmaKind::Shared { obj, offset, .. } = &v.kind
                    && obj.size.is_none()
                    && !named.contains(&obj.id)
                {
                    let reach = offset.saturating_add(v.end - v.start);
                    let e = out.entry(obj.id).or_insert(0);
                    *e = reach.max(*e);
                }
            }
        }
        out
    }

    /// Everything that keeps this machine from a checkpoint now, with how many of each
    /// (what a workload's ready point still needs, in one look; [`Kernel::checkpoint`]
    /// stops at the first). Empty does not promise success (a socket's provider may refuse).
    pub fn checkpoint_blockers(&self) -> Vec<(String, usize)> {
        let mut out: BTreeMap<String, usize> = BTreeMap::new();
        let mut add = |what: &str| *out.entry(what.to_owned()).or_default() += 1;
        for (bad, what) in [
            (!self.ptys.is_empty(), "pseudo-terminals"),
            (self.console.is_some(), "a console terminal"),
            (!self.sem.objs.is_empty(), "SysV semaphores"),
            (!self.msg.objs.is_empty(), "SysV message queues"),
            (!self.mqueues.is_empty(), "POSIX message queues"),
        ] {
            if bad {
                add(what);
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for p in self.procs.values() {
            if !seen.insert(Rc::as_ptr(&p.fds) as usize) {
                continue;
            }
            for (_, slot) in p.fds.borrow().iter() {
                if !seen.insert(Rc::as_ptr(&slot.file) as usize) {
                    continue;
                }
                match &slot.file.borrow().kind {
                    FileKind::MQueue(_) => add("a message queue"),
                    FileKind::Tty { .. } => add("a terminal"),
                    FileKind::Inet(n) => {
                        if let Some(why) = n.borrow().not_quiescent() {
                            add(why);
                        }
                    }
                    FileKind::Unix(u) => {
                        let u = u.borrow();
                        // Queued bytes, datagrams and connections are saved;
                        // descriptions in flight are not.
                        if !u.rx_fds.is_empty() || !u.msg_fds.is_empty() {
                            add("a local socket with descriptions in flight");
                        }
                        if let Some(why) = u.net.as_ref().and_then(|n| n.borrow().not_quiescent()) {
                            add(why);
                        }
                    }
                    _ => {}
                }
            }
        }
        out.into_iter().collect()
    }

    fn refuse_unsupported(&self) -> Result<(), CkptError> {
        let checks = [
            (!self.ptys.is_empty(), "pseudo-terminals"),
            (self.console.is_some(), "a console terminal"),
            (!self.sem.objs.is_empty(), "SysV semaphores"),
            (!self.msg.objs.is_empty(), "SysV message queues"),
            (!self.mqueues.is_empty(), "POSIX message queues"),
        ];
        match checks.iter().find(|(bad, _)| *bad) {
            Some((_, what)) => Err(unsupported(what)),
            None => Ok(()),
        }
    }

    /// A machine restored from a [`Kernel::checkpoint`] container (complete: every chunk
    /// present). `vfs` must have mounted the image the checkpoint was taken over (if any);
    /// `cfg` must schedule like the saved machine's. Every thread gets a fresh CPU from
    /// `frontend` with its saved registers.
    pub fn restore(
        cfg: crate::kernel::Config,
        arch: crate::arch::Arch,
        frontend: Box<dyn wasmshim_kernel::Frontend>,
        vfs: Vfs,
        snap: &Snapshot,
    ) -> Result<Self, CkptError> {
        snap.expect(SNAPSHOT_KIND, SNAPSHOT_VERSION)
            .map_err(|e| CkptError(format!("not this kind of snapshot: {e:?}")))?;
        let mut pages: BTreeMap<u64, Vec<(u64, &[u8])>> = BTreeMap::new();
        for (key, bytes) in snap.pages() {
            let bytes = bytes.ok_or_else(|| CkptError::new("the snapshot lacks a page's chunk"))?;
            pages
                .entry(key >> SPACE_SHIFT)
                .or_default()
                .push((key & ((1 << SPACE_SHIFT) - 1), bytes));
        }
        let mut k = Kernel::new(cfg, arch, frontend, vfs);
        let r = &mut Reader::new(&snap.body);
        read_header(r)?;
        let mut files = BTreeMap::new();
        for _ in 0..u64::load(r)? {
            let (id, len, origin): (u64, u64, Option<(u64, u64)>) = Ckpt::load(r)?;
            let source = origin_source(&k.vfs, origin)?;
            let pages = load_object_pages(&*k.frontend, len, source, &mut pages, r)?;
            if let Some(o) = origin {
                k.vfs.set_page_origin(id, o);
            }
            files.insert(
                id,
                Rc::new(crate::shm::MemObject {
                    id,
                    pages,
                    size: None,
                }),
            );
        }
        crate::vfs::ckpt::RESTORED_PAGES.with(|m| *m.borrow_mut() = files);
        let loaded = Self::restore_body(&mut k, r, pages);
        crate::vfs::ckpt::RESTORED_PAGES.with(|m| m.borrow_mut().clear());
        loaded?;
        Ok(k)
    }

    /// [`Kernel::restore`] after the page-cache objects (the filesystem onwards).
    fn restore_body(
        k: &mut Kernel,
        r: &mut Reader<'_>,
        pages: BTreeMap<u64, Vec<(u64, &[u8])>>,
    ) -> Result<(), CkptError> {
        k.vfs.load_saved(r)?;
        let sched = k.core.cfg;
        let frontend = &*k.frontend;
        let mut new_cpu = |ctx: &CpuContext| -> Box<dyn GuestCpu> {
            let mut cpu = frontend.new_cpu();
            cpu.set_context(ctx);
            cpu
        };
        k.core = wasmshim_kernel::Core::load(sched, r, &mut new_cpu)?;
        (k.next_pid, k.next_ephemeral) = Ckpt::load(r)?;
        k.init_exit = Ckpt::load(r)?;
        k.sig_generation = Ckpt::load(r)?;
        (k.shm.next_seq, k.shm.max) = Ckpt::load(r)?;
        (k.sem.next_seq, k.sem.max) = Ckpt::load(r)?;
        (k.msg.next_seq, k.msg.max) = Ckpt::load(r)?;
        k.zombies = Ckpt::load(r)?;
        for _ in 0..u64::load(r)? {
            let key: (u64, u64) = Ckpt::load(r)?;
            let (id, nr, ring_len) = Ckpt::load(r)?;
            let done = Ckpt::load(r)?;
            k.aio.insert(
                key,
                crate::syscall::AioCtx {
                    id,
                    nr,
                    done,
                    ring_len,
                },
            );
        }
        let mut l = Loader {
            objs: Vec::new(),
            vfs: &mut k.vfs,
            frontend: &*k.frontend,
            spaces: 0,
            pages,
            inet: Vec::new(),
            names: Vec::new(),
            depth: 0,
        };
        let n = u64::load(r)?;
        let mut procs = BTreeMap::new();
        for _ in 0..n {
            let pid = Pid::load(r)?;
            let p = l.process(r)?;
            procs.insert(pid, p);
        }
        // Connected local sockets: link each to its peer.
        let unix = |i: u64| -> Result<crate::objects::unix::UnixRef, CkptError> {
            l.objs
                .get(i as usize)
                .cloned()
                .flatten()
                .and_then(|o| {
                    o.downcast::<RefCell<crate::objects::unix::UnixSocket>>()
                        .ok()
                })
                .ok_or_else(|| CkptError::new("a link to a missing or different object"))
        };
        for _ in 0..u64::load(r)? {
            let (me, other): (u64, Option<u64>) = Ckpt::load(r)?;
            let u = unix(me)?;
            let peer = match other {
                Some(p) => Rc::downgrade(&unix(p)?),
                None => std::rc::Weak::new(),
            };
            u.borrow_mut().peer = peer;
        }
        for _ in 0..u64::load(r)? {
            let (slot, (key, uid, gid), (cuid, cgid, mode)): (
                u32,
                (i32, u32, u32),
                (u32, u32, u32),
            ) = Ckpt::load(r)?;
            let (seq, size): (u16, u64) = Ckpt::load(r)?;
            let obj = l.mem_object(r)?;
            let (atime, dtime, ctime) = Ckpt::load(r)?;
            let (cpid, lpid) = Ckpt::load(r)?;
            k.shm.objs.insert(
                slot,
                crate::shm::ShmSegment {
                    perm: crate::shm::IpcPerm {
                        key,
                        uid,
                        gid,
                        cuid,
                        cgid,
                        mode,
                        seq,
                    },
                    size,
                    obj,
                    atime,
                    dtime,
                    ctime,
                    cpid,
                    lpid,
                },
            );
        }
        let obj = |i: u64| l.objs.get(i as usize).cloned().flatten();
        for _ in 0..u64::load(r)? {
            let ino: crate::vfs::Ino = Ckpt::load(r)?;
            let list: Vec<(u64, i32)> = Ckpt::load(r)?;
            let mut watchers = Vec::new();
            for (i, wd) in list {
                let i = obj(i)
                    .and_then(|o| {
                        o.downcast::<RefCell<crate::objects::inotify::Inotify>>()
                            .ok()
                    })
                    .ok_or_else(|| CkptError::new("a watch of a missing inotify instance"))?;
                watchers.push((Rc::downgrade(&i), wd));
            }
            k.inotify_watches.insert(ino, watchers);
        }
        for _ in 0..u64::load(r)? {
            use crate::locks::{Lock, Owner};
            let (ino, (tag, who), (start, end, write)): (u64, (u8, u64), (u64, u64, bool)) =
                Ckpt::load(r)?;
            let desc = |i: u64| -> Result<usize, CkptError> {
                obj(i)
                    .and_then(|o| o.downcast::<RefCell<crate::fd::OpenFile>>().ok())
                    .map(|f| crate::syscall::description_id(&f))
                    .ok_or_else(|| CkptError::new("a lock of a missing description"))
            };
            let owner = match tag {
                0 => Owner::Posix(who as u32),
                1 => Owner::Ofd(desc(who)?),
                2 => Owner::Flock(desc(who)?),
                _ => return Err(CkptError::new("bad lock owner")),
            };
            k.locks.locks.push(Lock {
                ino,
                owner,
                start,
                end,
                write,
            });
        }
        for _ in 0..u64::load(r)? {
            let (name, phys): (Vec<u8>, Vec<u8>) = Ckpt::load(r)?;
            let (bus, vendor, product) = Ckpt::load(r)?;
            let descriptor = Ckpt::load(r)?;
            let opens: Vec<u64> = Ckpt::load(r)?;
            let opens = opens
                .into_iter()
                .map(|i| {
                    l.objs
                        .get(i as usize)
                        .cloned()
                        .flatten()
                        .and_then(|o| o.downcast::<RefCell<crate::devices::HidOpen>>().ok())
                        .map(|o| Rc::downgrade(&o))
                        .ok_or_else(|| CkptError::new("a missing hidraw description"))
                })
                .collect::<Result<_, _>>()?;
            k.hid.push(crate::devices::HidDevice {
                info: crate::devices::HidInfo {
                    name: String::from_utf8(name).map_err(|_| CkptError::new("bad HID name"))?,
                    phys: String::from_utf8(phys).map_err(|_| CkptError::new("bad HID phys"))?,
                    bus,
                    vendor,
                    product,
                    descriptor,
                },
                opens,
            });
        }
        for _ in 0..u64::load(r)? {
            let (id, (width, height), (refresh_num, refresh_den)) = Ckpt::load(r)?;
            let (stride, size) = Ckpt::load(r)?;
            let mem = l.mem_object(r)?;
            let shown = r.bytes()?.to_vec();
            let (blank, full_next) = Ckpt::load(r)?;
            let (t0, next) = Ckpt::load(r)?;
            let scratch = vec![0; shown.len()];
            k.fbs.push(crate::devices::Framebuffer {
                id,
                mode: crate::devices::FbMode {
                    width,
                    height,
                    refresh_num,
                    refresh_den,
                },
                stride,
                size,
                mem,
                shown,
                scratch,
                blank,
                full_next,
                t0,
                next,
            });
        }
        for _ in 0..u64::load(r)? {
            let (index, at, vblank) = Ckpt::load(r)?;
            let (y, height) = Ckpt::load(r)?;
            k.fb_frames.push(crate::devices::FbFrame {
                index,
                at,
                vblank,
                y,
                height,
                pixels: r.bytes()?.to_vec(),
            });
        }
        if !l.pages.is_empty() {
            return Err(CkptError::new(
                "pages of an address space or object the snapshot does not have",
            ));
        }
        let inet = std::mem::take(&mut l.inet);
        let names = std::mem::take(&mut l.names);
        drop(l);
        let provider = Ckpt::load(r)?;
        if !r.done() {
            return Err(CkptError::new("trailing bytes after the checkpoint"));
        }
        k.procs = procs;
        k.net_restored = (provider, inet);
        k.unix_names = names.into_iter().collect();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, VecDeque};

    use wasmshim_cpu::{A64Context, Event, PagedMemory, Perms};

    use super::*;
    use crate::arch::Arch;
    use crate::fd::open_file;
    use crate::kernel::{Config, Thread};
    use crate::siginfo::SigInfo;
    use crate::vfs::tar::tests::{MemBlob, build_tar};

    /// A CPU that only holds registers.
    #[derive(Clone, Default)]
    struct Regs(A64Context, u64);

    impl GuestCpu for Regs {
        fn run_until_event(&mut self, _mem: &mut dyn GuestMemory, _budget: u64) -> Event {
            Event::Yield
        }
        fn set_syscall_result(&mut self, value: u64) {
            self.0.x[0] = value;
        }
        fn pc(&self) -> u64 {
            self.0.pc
        }
        fn set_pc(&mut self, pc: u64) {
            self.0.pc = pc;
        }
        fn sp(&self) -> u64 {
            self.0.sp
        }
        fn set_sp(&mut self, sp: u64) {
            self.0.sp = sp;
        }
        fn thread_pointer(&self) -> u64 {
            self.0.tp
        }
        fn set_thread_pointer(&mut self, tp: u64) {
            self.0.tp = tp;
        }
        fn retired(&self) -> u64 {
            self.1
        }
        fn set_virtual_time(&mut self, _ns: u64) {}
        fn context(&self) -> CpuContext {
            CpuContext::A64(self.0.clone())
        }
        fn set_context(&mut self, ctx: &CpuContext) {
            let CpuContext::A64(c) = ctx;
            self.0 = c.clone();
        }
        fn duplicate(&self) -> Box<dyn GuestCpu> {
            Box::new(self.clone())
        }
    }

    struct Front;
    impl wasmshim_kernel::Frontend for Front {
        fn new_cpu(&self) -> Box<dyn GuestCpu> {
            Box::new(Regs::default())
        }
        fn new_memory(&self) -> Box<dyn GuestMemory> {
            Box::new(PagedMemory::new())
        }
    }

    fn image() -> Vec<u8> {
        build_tar(&[
            ("lib/", b'5', 0o755, b""),
            ("lib/libx.so", b'0', 0o755, &[7u8; 5000]),
        ])
    }

    fn vfs() -> Vfs {
        let mut v = Vfs::new();
        v.mount_image(Box::new(MemBlob(image()))).unwrap();
        v
    }

    fn process(k: &mut Kernel, ppid: Pid, fds: Rc<RefCell<FdTable>>) -> Process {
        let mm = k.new_address_space(Box::new(PagedMemory::new()));
        Process {
            ppid,
            pgid: 1,
            sid: 1,
            mm: Rc::new(RefCell::new(mm)),
            fds,
            cwd: ROOT,
            umask: 0o022,
            sigactions: Rc::new(RefCell::new([KernelSigaction::default(); 64])),
            exe: b"/bin/x".to_vec(),
            comm: b"x".to_vec(),
            cmdline: b"x\0".to_vec(),
            environ: Vec::new(),
            auxv: Vec::new(),
            threads: BTreeSet::new(),
            children: BTreeSet::new(),
            exit_signal: 17,
            pending: Default::default(),
            vfork_parent: None,
            creds: Creds::default(),
            itimers: Default::default(),
            timers: Default::default(),
            semundo: Default::default(),
        }
    }

    /// Two processes sharing a descriptor table (as after `clone(CLONE_FILES)`), a pipe with
    /// bytes in it, an eventfd and an epoll set over both, pending signals, an address space
    /// with written anonymous pages and a private file mapping.
    fn machine() -> Kernel {
        let mut k = Kernel::new(Config::default(), Arch::Aarch64, Box::new(Front), vfs());
        let fds = Rc::new(RefCell::new(FdTable::new()));
        let pipe = Rc::new(RefCell::new(Pipe {
            id: k.new_object_id(),
            generation: 3,
            buf: VecDeque::from(b"queued".to_vec()),
            readers: 1,
            writers: 2,
        }));
        let r = open_file(
            FileKind::Pipe {
                pipe: pipe.clone(),
                write: false,
            },
            0,
            b"pipe:[1]",
        );
        let wr = open_file(FileKind::Pipe { pipe, write: true }, 1, b"pipe:[1]");
        let ev = open_file(
            FileKind::EventFd(Rc::new(RefCell::new(EventFd {
                id: k.new_object_id(),
                count: 5,
                semaphore: false,
                generation: 1,
            }))),
            2,
            b"anon_inode:[eventfd]",
        );
        let mut interests = BTreeMap::new();
        for (fd, f) in [(3, &r), (5, &ev)] {
            interests.insert(
                fd,
                Interest {
                    file: Rc::downgrade(f),
                    events: 1,
                    data: fd as u64 * 10,
                    last_gen: Some(1),
                    disabled: false,
                },
            );
        }
        let ep = open_file(
            FileKind::Epoll(Rc::new(RefCell::new(Epoll {
                id: k.new_object_id(),
                interests,
            }))),
            2,
            b"anon_inode:[eventpoll]",
        );
        {
            let mut t = fds.borrow_mut();
            t.insert_at(
                0,
                open_file(FileKind::Stdio(Stdio::In), 0, b"/dev/console"),
                false,
            )
            .unwrap();
            t.insert_at(3, r, false).unwrap();
            t.insert_at(4, wr.clone(), true).unwrap();
            t.insert_at(7, wr, false).unwrap(); // a dup
            t.insert_at(5, ev, false).unwrap();
            t.insert_at(6, ep, true).unwrap();
        }
        let pid1 = k.alloc_pid();
        let mut p1 = process(&mut k, 0, fds.clone());
        p1.pending.push(SigInfo::new(10, 0));
        {
            let mut mm = p1.mm.borrow_mut();
            mm.map(0x10000, 0x14000, Perms::RW, VmaKind::Anon).unwrap();
            mm.mem.write_u64(0x11008, 0xfeed).unwrap();
            let lib = k.vfs.resolve(ROOT, b"/lib/libx.so", true).unwrap();
            let contents = k.vfs.file_contents(lib).unwrap();
            mm.map(
                0x40000,
                0x42000,
                Perms::RW,
                VmaKind::File {
                    path: b"/lib/libx.so".to_vec(),
                    offset: 0,
                },
            )
            .unwrap();
            mm.mem
                .map_source(0x40000, 0x2000, Perms::RW, contents.pages(0, 5000), 0)
                .unwrap();
            mm.mem.write_u8(0x41000 + 9, 1).unwrap(); // past the file's end: stored
            mm.mem.read_u8(0x40000).unwrap(); // read only: replayed
        }
        k.procs.insert(pid1, p1);
        let mut cpu = Regs::default();
        cpu.0.x[0] = 42;
        cpu.0.pc = 0x40_1000;
        cpu.1 = 1000;
        let mut t = Thread::new(pid1, Box::new(cpu), 0, LinuxThread::new(pid1, 1 << 9));
        t.pending.push(SigInfo::new(12, 0));
        k.add_thread(t);
        let pid2 = k.alloc_pid();
        let p2 = process(&mut k, pid1, fds);
        k.procs.insert(pid2, p2);
        k.procs.get_mut(&pid1).unwrap().children.insert(pid2);
        k.add_thread(Thread::new(
            pid2,
            Box::new(Regs::default()),
            0,
            LinuxThread::new(pid2, 0),
        ));
        k
    }

    /// The checkpoint as container bytes.
    fn saved(k: &mut Kernel) -> Vec<u8> {
        k.checkpoint().unwrap().encode()
    }

    fn restored(bytes: &[u8]) -> Result<Kernel, CkptError> {
        let snap = Snapshot::decode(bytes).map_err(|e| CkptError(format!("{e:?}")))?;
        Kernel::restore(
            Config::default(),
            Arch::Aarch64,
            Box::new(Front),
            vfs(),
            &snap,
        )
    }

    #[test]
    fn a_machine_round_trips_with_sharing_intact() {
        let mut k = machine();
        let bytes = saved(&mut k);
        let mut d = restored(&bytes).unwrap();
        assert_eq!(saved(&mut d), bytes, "restore, then save: the same bytes");

        {
            let (p1, p2) = (&d.procs[&1], &d.procs[&2]);
            assert!(
                Rc::ptr_eq(&p1.fds, &p2.fds),
                "a shared descriptor table stays shared"
            );
            let fds = p1.fds.borrow();
            assert_eq!(fds.iter().count(), 6);
            let (r3, w4, w7, e6) = (
                fds.get(3).unwrap(),
                fds.get(4).unwrap(),
                fds.get(7).unwrap(),
                fds.get(6).unwrap(),
            );
            assert!(
                Rc::ptr_eq(&w4, &w7),
                "dup'ed descriptors share one description"
            );
            let pipe_of = |f: &FileRef| match &f.borrow().kind {
                FileKind::Pipe { pipe, .. } => pipe.clone(),
                _ => panic!("not a pipe"),
            };
            let (rp, wp) = (pipe_of(&r3), pipe_of(&w4));
            assert!(Rc::ptr_eq(&rp, &wp), "both ends name one pipe");
            assert_eq!(rp.borrow().buf, VecDeque::from(b"queued".to_vec()));
            let watched = match &e6.borrow().kind {
                FileKind::Epoll(ep) => ep.borrow().interests[&3].file.upgrade().unwrap(),
                _ => panic!("fd 6 is the epoll set"),
            };
            assert!(Rc::ptr_eq(&watched, &r3), "interests name the fd's file");

            assert_eq!(d.core.tasks[&1].pending.set, 1 << 11);
            assert_eq!(p1.pending.set, 1 << 9);
            assert_eq!(d.core.tasks[&1].cpu.pc(), 0x40_1000);
            assert_eq!(d.core.tasks[&1].own_retired(), 1000);
            let mm = p1.mm.borrow();
            assert_eq!(mm.mem.read_u64(0x11008).unwrap(), 0xfeed);
            assert_eq!(mm.mem.read_u8(0x40000 + 4999).unwrap(), 7);
            assert_eq!(mm.mem.read_u8(0x41000 + 9).unwrap(), 1);
            assert_eq!(mm.mem.read_u8(0x41000 + 10).unwrap(), 7);
            assert_eq!(mm.mem.read_u8(0x41000 + 1000).unwrap(), 0);
        }
        // Ids keep counting where they were.
        assert_eq!(d.alloc_pid(), k.alloc_pid());
        assert_eq!(d.new_object_id(), k.new_object_id());
    }

    #[test]
    fn unsupported_state_and_damaged_bytes_fail_loudly() {
        let mut k = machine();
        let bytes = saved(&mut k);
        for n in (0..bytes.len()).step_by(97) {
            assert!(restored(&bytes[..n]).is_err(), "prefix of {n} bytes");
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(restored(&longer).is_err(), "trailing bytes");
        // A console terminal is refused (not checkpointed yet); without it the machine saves.
        k.attach_console(24, 80);
        let e = k.checkpoint().unwrap_err();
        assert!(e.0.contains("console"), "{e:?}");
        assert!(
            k.checkpoint_blockers()
                .iter()
                .any(|(what, n)| what.contains("console") && *n == 1)
        );
        k.console = None;
        assert!(k.checkpoint().is_ok());
    }

    #[test]
    fn objects_nested_past_any_machine_are_refused() {
        // A crafted body: an epoll set watching an epoll set watching ... far deeper than
        // `epoll_ctl` allows. Each level is a few bytes; reading it must fail, not recurse
        // until the stack runs out.
        let mut w = Writer::new();
        for level in 0..100_000u64 {
            // The file: a new object, of kind epoll.
            (2 * level, true, 7u8).save(&mut w);
            // Its epoll set: a new object with one interest, whose file follows.
            (2 * level + 1, true).save(&mut w);
            (level, 1u64, (0i32, true)).save(&mut w);
        }
        let bytes = w.into_bytes();
        let mut vfs = vfs();
        let mut l = Loader {
            objs: Vec::new(),
            vfs: &mut vfs,
            frontend: &Front,
            spaces: 0,
            pages: BTreeMap::new(),
            inet: Vec::new(),
            names: Vec::new(),
            depth: 0,
        };
        let e = l.file(&mut Reader::new(&bytes)).expect_err("refused");
        assert!(e.0.contains("nested too deep"), "{e:?}");
        assert_eq!(l.depth, 0);
    }

    /// A netlink socket, an inotify instance with a watch (and the kernel's
    /// watcher table) and file locks by a process and by a description round-trip.
    #[test]
    fn netlink_inotify_and_locks_round_trip() {
        use crate::locks::{Lock, Owner};
        use crate::objects::inotify::Inotify;
        use crate::objects::netlink::Netlink;
        let mut k = machine();
        let nl = open_file(
            FileKind::Netlink(Rc::new(RefCell::new(Netlink {
                id: k.new_object_id(),
                port: 77,
                groups: 1,
                queue: VecDeque::from(vec![b"answer".to_vec()]),
                generation: 2,
            }))),
            2,
            b"socket:[netlink]",
        );
        let ino = ROOT;
        let inotify = Rc::new(RefCell::new(Inotify {
            id: k.new_object_id(),
            generation: 1,
            next_wd: 2,
            watches: BTreeMap::from([(1, (ino, 0x2))]),
            queue: VecDeque::new(),
            overflowed: false,
        }));
        k.inotify_watches
            .insert(ino, vec![(Rc::downgrade(&inotify), 1)]);
        let inf = open_file(FileKind::Inotify(inotify), 0, b"anon_inode:inotify");
        let pid = *k.procs.keys().next().unwrap();
        {
            let p = &k.procs[&pid];
            let mut t = p.fds.borrow_mut();
            t.insert_at(20, nl, false).unwrap();
            t.insert_at(21, inf.clone(), false).unwrap();
        }
        k.locks.locks.push(Lock {
            ino,
            owner: Owner::Posix(pid),
            start: 0,
            end: 10,
            write: true,
        });
        k.locks.locks.push(Lock {
            ino,
            owner: Owner::Flock(crate::syscall::description_id(&inf)),
            start: 0,
            end: u64::MAX,
            write: false,
        });
        // A SysV segment nobody maps (its object is defined with the table).
        let seg_obj = k.new_mem_object(None, Some(4096)).unwrap();
        seg_obj.pages.write(0, b"mit-shm").unwrap();
        k.shm.objs.insert(
            3,
            crate::shm::ShmSegment {
                perm: crate::shm::IpcPerm::new(42, 0, 0, 0o600),
                size: 100,
                obj: seg_obj,
                atime: 1,
                dtime: 2,
                ctime: 3,
                cpid: pid,
                lpid: pid,
            },
        );
        let bytes = saved(&mut k);
        let mut r = restored(&bytes).expect("restore");
        assert_eq!(saved(&mut r), bytes, "save(restore(s)) == s");
        let f = r.procs[&pid].fds.borrow().get(21).unwrap();
        let mut buf = [0u8; 7];
        r.shm.objs[&3].obj.pages.read(0, &mut buf).unwrap();
        assert_eq!(&buf, b"mit-shm");
        assert_eq!(
            r.locks.locks[1].owner,
            Owner::Flock(crate::syscall::description_id(&f))
        );
        let FileKind::Inotify(i) = &f.borrow().kind else {
            panic!("not inotify")
        };
        let w = &r.inotify_watches[&ino];
        assert!(Rc::ptr_eq(&w[0].0.upgrade().unwrap(), i));
    }

    /// Display and input devices round-trip: a HID device with an open
    /// description and a queued report (still linked to the device: a report after the restore
    /// reaches it), and a framebuffer mapped shared (one object for the device and the
    /// mapping), its vblank clock and a frame not taken yet.
    #[test]
    fn display_and_input_devices_round_trip() {
        let mut k = machine();
        let pid = *k.procs.keys().next().unwrap();
        let fb = k
            .attach_fb(
                crate::devices::FbMode {
                    width: 4,
                    height: 2,
                    refresh_num: 60,
                    refresh_den: 1,
                },
                0,
            )
            .unwrap();
        let hid = k.attach_hid(crate::devices::HidInfo::boot_keyboard("kbd", "usb-1"));
        let ino = k.vfs.resolve(ROOT, b"/dev/hidraw0", true).unwrap();
        let open = open_file(k.hidraw_open(hid, ino).unwrap(), 2, b"/dev/hidraw0");
        k.procs[&pid]
            .fds
            .borrow_mut()
            .insert_at(22, open, false)
            .unwrap();
        k.hid_input(hid, &[0, 0, 4, 0, 0, 0, 0, 0]);
        let (mem, size) = k.fb_object(fb).unwrap();
        k.procs[&pid]
            .mm
            .borrow_mut()
            .map_shared(
                0xd0000,
                0xd0000 + size,
                Perms::RW,
                &mem,
                0,
                b"/dev/fb0".to_vec(),
                true,
            )
            .unwrap();
        k.procs[&pid]
            .mm
            .borrow_mut()
            .mem
            .write(0xd0000, &[9; 32])
            .unwrap();
        k.expire_vblanks(20_000_000);
        assert_eq!(k.fb_frames.len(), 1);
        let bytes = saved(&mut k);
        let mut r = restored(&bytes).expect("restore");
        assert_eq!(saved(&mut r), bytes, "save(restore(s)) == s");
        assert_eq!(r.hid_devices(), k.hid_devices());
        assert_eq!(r.next_vblank(), k.next_vblank());
        assert_eq!(r.take_frames(), k.take_frames());
        // The restored screen, whole, for an embedder whose display starts empty.
        let screens = r.fb_screens(7);
        assert_eq!(screens.len(), 1);
        assert_eq!((screens[0].y, screens[0].height, screens[0].at), (0, 2, 7));
        assert_eq!(&screens[0].pixels[..32], &[9; 32]);
        assert_eq!(screens, k.fb_screens(7));
        let mapped = match &r.procs[&pid].mm.borrow().vmas().last().unwrap().kind {
            VmaKind::Shared { obj, .. } => obj.clone(),
            _ => panic!("not the framebuffer mapping"),
        };
        assert!(Rc::ptr_eq(&mapped, &r.fb_object(fb).unwrap().0));
        r.hid_input(hid, &[0, 0, 5, 0, 0, 0, 0, 0]);
        let f = r.procs[&pid].fds.borrow().get(22).unwrap();
        let FileKind::HidRaw(h) = &f.borrow().kind else {
            panic!("not hidraw")
        };
        let mut buf = [0u8; 8];
        assert_eq!(r.hidraw_read(&h.clone(), &mut buf, true), Ok(8));
        assert_eq!(buf[2], 4);
        assert_eq!(r.hidraw_read(&h.clone(), &mut buf, true), Ok(8));
        assert_eq!(buf[2], 5);
    }

    /// A file mapped shared (its page cache, as Chromium's `/dev/shm` files and
    /// memfds are) round-trips: the restored file and the restored mapping are one object, and
    /// a store through the mapping is what the file reads.
    #[test]
    fn a_shared_file_mapping_round_trips_with_its_file() {
        let mut k = machine();
        let ino = k.vfs.resolve(ROOT, b"/lib/libx.so", true).unwrap();
        let obj = k.file_object(ino).unwrap();
        obj.pages.write(100, b"dirty").unwrap();
        let pid = *k.procs.keys().next().unwrap();
        k.procs[&pid]
            .mm
            .borrow_mut()
            .map_shared(
                0xb0000,
                0xb2000,
                Perms::RW,
                &obj,
                0,
                b"/lib/libx.so".to_vec(),
                true,
            )
            .unwrap();
        // Only the page that differs from the image is saved (the file's second page is the
        // image's: a restore reads it from there again).
        let snap = k.checkpoint().unwrap();
        let object_pages: Vec<u64> = snap
            .pages()
            .map(|(key, _)| key)
            .filter(|key| key >> SPACE_SHIFT >= OBJECT_SPACES)
            .collect();
        assert_eq!(object_pages.len(), 1, "{object_pages:x?}");
        let bytes = saved(&mut k);
        let mut r = restored(&bytes).expect("restore");
        assert_eq!(saved(&mut r), bytes, "save(restore(s)) == s");
        let mut page2 = [0u8; 4];
        r.vfs.read(ino, 4096, &mut page2).unwrap();
        assert_eq!(page2, [7; 4], "the image's bytes, read again");
        let file = r.vfs.file_pages(ino).expect("still the file's page cache");
        let mapped = {
            let mm = r.procs[&pid].mm.borrow();
            let v = mm
                .vmas()
                .find(|v| matches!(v.kind, VmaKind::Shared { .. }))
                .unwrap();
            match &v.kind {
                VmaKind::Shared { obj, .. } => obj.clone(),
                _ => unreachable!(),
            }
        };
        assert!(
            Rc::ptr_eq(&file, &mapped),
            "one object for the file and its mapping"
        );
        r.procs[&pid]
            .mm
            .borrow_mut()
            .mem
            .write(0xb0000, b"Z")
            .unwrap();
        let mut buf = [0u8; 5];
        let n = r.vfs.read(ino, 100, &mut buf).unwrap();
        assert_eq!((n, &buf), (5, b"dirty"));
        r.vfs.read(ino, 0, &mut buf[..1]).unwrap();
        assert_eq!(buf[0], b'Z');
    }

    /// The page cache of a file unlinked while mapped (Chromium's `/dev/shm`
    /// files) is saved with its pages as far as its mappings reach.
    #[test]
    fn a_mapping_of_an_unlinked_file_keeps_its_pages() {
        let mut k = machine();
        let obj = k.new_mem_object(None, None).unwrap();
        obj.pages.write(4096 + 1, b"orphan").unwrap();
        let pid = *k.procs.keys().next().unwrap();
        k.procs[&pid]
            .mm
            .borrow_mut()
            .map_shared(
                0xc0000,
                0xc2000,
                Perms::RW,
                &obj,
                0,
                b"/dev/shm/x (deleted)".to_vec(),
                true,
            )
            .unwrap();
        let bytes = saved(&mut k);
        let mut r = restored(&bytes).expect("restore");
        assert_eq!(saved(&mut r), bytes, "save(restore(s)) == s");
        let mut buf = [0u8; 6];
        r.procs[&pid]
            .mm
            .borrow()
            .mem
            .read(0xc1001, &mut buf)
            .unwrap();
        assert_eq!(&buf, b"orphan");
    }

    /// A shared anonymous object mapped by two processes is saved once, with its
    /// pages, and both restored processes map the same restored object.
    #[test]
    fn a_shared_anonymous_mapping_round_trips_shared() {
        let mut k = machine();
        let obj = k.new_mem_object(None, Some(3 * 4096)).unwrap();
        obj.pages.write(4096 + 8, b"nginx shm").unwrap();
        let pids: Vec<Pid> = k.procs.keys().copied().collect();
        for (i, pid) in pids.iter().enumerate() {
            let at = 0x90000 + i as u64 * 0x10000;
            k.procs[pid]
                .mm
                .borrow_mut()
                .map_shared(
                    at,
                    at + 3 * 4096,
                    Perms::RW,
                    &obj,
                    0,
                    b"/dev/zero (deleted)".to_vec(),
                    true,
                )
                .unwrap();
        }
        let bytes = saved(&mut k);
        let mut r = restored(&bytes).expect("restore");
        assert_eq!(saved(&mut r), bytes, "save(restore(s)) == s");
        let objs: Vec<crate::shm::MemRef> = pids
            .iter()
            .map(|pid| {
                let mm = r.procs[pid].mm.borrow();
                let v = mm
                    .vmas()
                    .find(|v| matches!(v.kind, VmaKind::Shared { .. }))
                    .unwrap();
                match &v.kind {
                    VmaKind::Shared { obj, .. } => obj.clone(),
                    _ => unreachable!(),
                }
            })
            .collect();
        assert!(
            Rc::ptr_eq(&objs[0], &objs[1]),
            "one object for both processes"
        );
        assert_eq!(objs[0].id, obj.id);
        let mut buf = [0u8; 9];
        objs[0].pages.read(4096 + 8, &mut buf).unwrap();
        assert_eq!(&buf, b"nginx shm");
        // A store through one process is seen through the other.
        r.procs[&pids[0]]
            .mm
            .borrow_mut()
            .mem
            .write(0x90000, b"x")
            .unwrap();
        let mut one = [0u8; 1];
        r.procs[&pids[1]]
            .mm
            .borrow()
            .mem
            .read(0xa0000, &mut one)
            .unwrap();
        assert_eq!(&one, b"x");
    }
}
