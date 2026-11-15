//! [`SharedPages`]: the memory objects [`crate::PagedMemory`] maps shared.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::rc::Rc;
use alloc::vec::Vec;

use core::any::Any;
use core::cell::{Cell, RefCell};
use core::sync::atomic::{AtomicU64, Ordering};

use crate::memory::{PAGE_SIZE, PageSource, PageSourceError, SharedMemory};

const PAGE: usize = PAGE_SIZE as usize;

/// One page of an object. Boxed, so its address is stable for as long as the object lives:
/// engines keep host addresses of mapped pages in their TLBs ([`crate::PagedMemory::host_read_page`]).
type Frame = Box<RefCell<[u8; PAGE]>>;

/// Moves whenever code in *some* shared page may have changed (a store to an object that is
/// mapped executable anywhere). Every [`crate::PagedMemory`] adds it to its code epoch, so code
/// caches revalidate: a store through one address space's writable mapping can change code
/// that another address space (or another range of the same one) executes.
static SHARED_CODE: AtomicU64 = AtomicU64::new(0);

/// The global shared-code counter (see [`SHARED_CODE`]).
pub(crate) fn shared_code_epoch() -> u64 {
    SHARED_CODE.load(Ordering::Relaxed)
}

/// Moves when an object that tracks its stores ([`SharedMemory::take_dirty`]) starts a new
/// period after handing pages out for direct writes: engines drop their direct write entries
/// for shared pages, so the next store to such a page asks again (and is recorded).
static SHARED_WRITES: AtomicU64 = AtomicU64::new(0);

/// A memory object for [`crate::PagedMemory`]: a sparse set of pages, each allocated on its
/// first write (or first direct host access) and then kept, at a fixed host address, until the
/// object is dropped. Pages never written read as zeros, or as the pages of the object's
/// [`PageSource`] (a file's page cache starts as the file's bytes and fills page by page).
pub struct SharedPages {
    frames: RefCell<BTreeMap<u64, Frame>>,
    source: Option<Rc<dyn PageSource>>,
    /// Source pages at or above this index read as zeros (truncated away).
    source_limit: Cell<u64>,
    /// Some mapping of this object has been executable: stores must keep code caches exact.
    exec: Cell<bool>,
    /// Bumped on every store while `exec` (part of the mapped pages' code versions).
    code_gen: Cell<u32>,
    /// Store tracking ([`SharedMemory::take_dirty`]): the pages stored to, or handed out for
    /// direct writes, since the last take; `None` until the first take.
    dirty: RefCell<Option<BTreeSet<u64>>>,
    /// A page was handed out for direct writes since the last take.
    handed: Cell<bool>,
    /// Pages that cannot be listed changed since the last take (a truncation over a source).
    all: Cell<bool>,
}

impl Default for SharedPages {
    fn default() -> Self {
        Self::new(None)
    }
}

impl SharedPages {
    /// An object whose unwritten pages read as `source`'s pages (or as zeros without one).
    /// The source must be pure (see [`PageSource`]).
    pub fn new(source: Option<Rc<dyn PageSource>>) -> Self {
        Self {
            frames: RefCell::new(BTreeMap::new()),
            source,
            source_limit: Cell::new(u64::MAX),
            exec: Cell::new(false),
            code_gen: Cell::new(0),
            dirty: RefCell::new(None),
            handed: Cell::new(false),
            all: Cell::new(false),
        }
    }

    /// A counter that moves when some object that tracks its stores starts a new period after
    /// handing pages out for direct writes. An engine that keeps direct write entries for shared
    /// pages drops them when it moves (before it runs code again), so every later store to such a
    /// page is recorded ([`SharedMemory::take_dirty`]).
    pub fn write_epoch() -> u64 {
        SHARED_WRITES.load(Ordering::Relaxed)
    }

    /// Records a store (or a hand-out for direct writes) to page `index` while tracking.
    fn mark(&self, index: u64) {
        if let Some(set) = self.dirty.borrow_mut().as_mut() {
            set.insert(index);
        }
    }

    /// A counter that moves whenever code in some shared page anywhere may have changed (part of
    /// every [`crate::PagedMemory`]'s code epoch). An engine that keeps direct write entries for
    /// shared pages drops them when it moves: the object may have become executable.
    pub fn code_epoch() -> u64 {
        shared_code_epoch()
    }

    /// Number of pages allocated so far.
    pub fn resident_pages(&self) -> usize {
        self.frames.borrow().len()
    }

    /// The initial contents of page `index`.
    fn initial(&self, index: u64) -> Result<[u8; PAGE], PageSourceError> {
        match &self.source {
            Some(src) if index < self.source_limit.get() => Ok(*src.page(index)?),
            _ => Ok([0; PAGE]),
        }
    }

    /// Allocates page `index` if it is not yet.
    pub(crate) fn materialize(&self, index: u64) -> Result<(), PageSourceError> {
        if self.frames.borrow().contains_key(&index) {
            return Ok(());
        }
        let page = self.initial(index)?;
        self.frames
            .borrow_mut()
            .insert(index, Box::new(RefCell::new(page)));
        Ok(())
    }

    /// Runs `f` on the contents of page `index` (without allocating it).
    pub(crate) fn with_page<R>(
        &self,
        index: u64,
        f: impl FnOnce(&[u8; PAGE]) -> R,
    ) -> Result<R, PageSourceError> {
        if let Some(frame) = self.frames.borrow().get(&index) {
            return Ok(f(&frame.borrow()));
        }
        let page = self.initial(index)?;
        Ok(f(&page))
    }

    /// Runs `f` on page `index` for writing (allocating it first).
    pub(crate) fn with_page_mut<R>(
        &self,
        index: u64,
        f: impl FnOnce(&mut [u8; PAGE]) -> R,
    ) -> Result<R, PageSourceError> {
        self.materialize(index)?;
        self.stored();
        self.mark(index);
        let frames = self.frames.borrow();
        let frame = frames.get(&index).ok_or(PageSourceError)?;
        Ok(f(&mut frame.borrow_mut()))
    }

    /// The host address of page `index` (allocating it), for engines' direct reads.
    pub(crate) fn host_page(&self, index: u64) -> Result<usize, PageSourceError> {
        self.materialize(index)?;
        let frames = self.frames.borrow();
        let frame = frames.get(&index).ok_or(PageSourceError)?;
        Ok(frame.as_ptr() as usize)
    }

    /// The host address of page `index` for an engine's direct writes: while the object tracks
    /// its stores, the page counts as stored to (the engine writes it without telling us).
    pub(crate) fn host_write_page(&self, index: u64) -> Result<usize, PageSourceError> {
        let host = self.host_page(index)?;
        if self.dirty.borrow().is_some() {
            self.mark(index);
            self.handed.set(true);
        }
        Ok(host)
    }

    /// Records a store: code in any executable mapping of this object may have changed.
    fn stored(&self) {
        if self.exec.get() {
            self.code_gen.set(self.code_gen.get().wrapping_add(1));
            SHARED_CODE.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Some mapping of this object became executable.
    pub(crate) fn note_exec(&self) {
        if !self.exec.get() {
            self.exec.set(true);
            SHARED_CODE.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// `true` once some mapping of this object has been executable (direct host writes are
    /// refused from then on, so every store keeps code caches exact).
    pub(crate) fn exec(&self) -> bool {
        self.exec.get()
    }

    /// The object's code generation (see [`crate::GuestMemory::code_version`]).
    pub(crate) fn code_gen(&self) -> u32 {
        self.code_gen.get()
    }
}

/// Splits `[offset, offset + len)` into `(page index, offset in page, start in the buffer,
/// length)` pieces.
fn pieces(offset: u64, len: usize) -> impl Iterator<Item = (u64, usize, usize, usize)> {
    let mut done = 0usize;
    core::iter::from_fn(move || {
        if done >= len {
            return None;
        }
        let at = offset.checked_add(done as u64)?;
        let in_page = (at % PAGE_SIZE) as usize;
        let n = (PAGE - in_page).min(len - done);
        let piece = (at / PAGE_SIZE, in_page, done, n);
        done += n;
        Some(piece)
    })
}

impl SharedMemory for SharedPages {
    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), PageSourceError> {
        if offset.checked_add(buf.len() as u64).is_none() {
            return Err(PageSourceError);
        }
        for (index, at, start, n) in pieces(offset, buf.len()) {
            self.with_page(index, |p| {
                buf[start..start + n].copy_from_slice(&p[at..at + n]);
            })?;
        }
        Ok(())
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<(), PageSourceError> {
        if offset.checked_add(data.len() as u64).is_none() {
            return Err(PageSourceError);
        }
        for (index, at, start, n) in pieces(offset, data.len()) {
            self.with_page_mut(index, |p| {
                p[at..at + n].copy_from_slice(&data[start..start + n]);
            })?;
        }
        Ok(())
    }

    fn zero_from(&self, offset: u64) -> Result<(), PageSourceError> {
        let index = offset / PAGE_SIZE;
        let at = (offset % PAGE_SIZE) as usize;
        let mut first_whole = index;
        if at != 0 {
            // The page holding `offset` keeps its head; its tail becomes zero.
            let has_tail = self.with_page(index, |p| p[at..].iter().any(|&b| b != 0))?;
            if has_tail {
                self.with_page_mut(index, |p| p[at..].fill(0))?;
            }
            first_whole = index + 1;
        }
        if first_whole < self.source_limit.get() {
            if self.source.is_some() {
                // Unallocated source pages from here on read as zeros now: not listable.
                self.all.set(true);
            }
            self.source_limit.set(first_whole);
        }
        let mut stored = false;
        for (&index, frame) in self.frames.borrow().range(first_whole..) {
            let mut page = frame.borrow_mut();
            if page.iter().any(|&b| b != 0) {
                page.fill(0);
                stored = true;
                self.mark(index);
            }
        }
        if stored {
            self.stored();
        }
        Ok(())
    }

    fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }

    fn take_dirty(&self) -> Option<Vec<u64>> {
        let taken = self.dirty.borrow_mut().replace(BTreeSet::new());
        let all = self.all.replace(false);
        // Pages handed out for direct writes may be written again without telling us: engines
        // drop those entries before they run again. On the first take any earlier hand-out
        // went unrecorded, so they drop theirs too.
        if self.handed.replace(false) || taken.is_none() {
            SHARED_WRITES.fetch_add(1, Ordering::Relaxed);
        }
        match taken {
            Some(set) if !all => Some(set.into_iter().collect()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::PageRef;

    struct Ones;
    impl PageSource for Ones {
        fn page(&self, _index: u64) -> Result<PageRef, PageSourceError> {
            Ok(Rc::new([1; PAGE]))
        }
    }

    #[test]
    fn reads_writes_and_zeroes() {
        let o = SharedPages::new(None);
        let mut b = [9u8; 10];
        o.read(4090, &mut b).unwrap();
        assert_eq!(b, [0; 10]);
        assert_eq!(o.resident_pages(), 0, "reads allocate nothing");
        o.write(4090, &[7; 10]).unwrap();
        assert_eq!(o.resident_pages(), 2);
        o.read(4090, &mut b).unwrap();
        assert_eq!(b, [7; 10]);
        o.zero_from(4093).unwrap();
        o.read(4090, &mut b).unwrap();
        assert_eq!(b, [7, 7, 7, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn tracks_stores_and_direct_write_hand_outs() {
        let o = SharedPages::new(None);
        o.write(0, &[1]).unwrap();
        let e0 = SharedPages::write_epoch();
        assert_eq!(o.take_dirty(), None, "the first take starts tracking");
        assert!(
            SharedPages::write_epoch() > e0,
            "earlier hand-outs went unrecorded"
        );
        assert_eq!(o.take_dirty(), Some(Vec::new()));
        o.write(PAGE_SIZE + 5, &[2; 4096]).unwrap();
        let mut b = [0u8; 8];
        o.read(9 * PAGE_SIZE, &mut b).unwrap();
        o.host_page(7).unwrap();
        assert_eq!(
            o.take_dirty(),
            Some(alloc::vec![1, 2]),
            "reads count for nothing"
        );
        let e1 = SharedPages::write_epoch();
        o.host_write_page(4).unwrap();
        assert_eq!(
            SharedPages::write_epoch(),
            e1,
            "moves at the take, not the hand-out"
        );
        assert_eq!(o.take_dirty(), Some(alloc::vec![4]));
        assert!(
            SharedPages::write_epoch() > e1,
            "engines drop their write entries"
        );
        let e2 = SharedPages::write_epoch();
        o.write(3 * PAGE_SIZE, &[3]).unwrap();
        o.zero_from(PAGE_SIZE + 8).unwrap();
        assert_eq!(o.take_dirty(), Some(alloc::vec![1, 2, 3]));
        assert_eq!(SharedPages::write_epoch(), e2, "no hand-out, no move");
        let s = SharedPages::new(Some(Rc::new(Ones)));
        assert_eq!(s.take_dirty(), None);
        s.zero_from(PAGE_SIZE).unwrap();
        assert_eq!(
            s.take_dirty(),
            None,
            "source pages read as zeros now: unlistable"
        );
        assert_eq!(s.take_dirty(), Some(Vec::new()));
    }

    #[test]
    fn source_pages_fill_and_truncate_away() {
        let o = SharedPages::new(Some(Rc::new(Ones)));
        let mut b = [0u8; 4];
        o.read(2 * PAGE_SIZE, &mut b).unwrap();
        assert_eq!(b, [1; 4]);
        o.write(0, &[5]).unwrap();
        o.read(0, &mut b).unwrap();
        assert_eq!(b, [5, 1, 1, 1]);
        o.zero_from(PAGE_SIZE + 2).unwrap();
        o.read(PAGE_SIZE, &mut b).unwrap();
        assert_eq!(b, [1, 1, 0, 0]);
        o.read(3 * PAGE_SIZE, &mut b).unwrap();
        assert_eq!(b, [0; 4], "source pages past the truncation read as zeros");
    }
}
