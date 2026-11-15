//! File pages for lazy mappings: a [`PageSource`] over a file's bytes as they were when the
//! mapping was made (`execve`'s segments, `mmap` of a file).

use std::cell::RefCell;
use std::rc::Rc;

use wasmshim_cpu::{PAGE_SIZE, PageRef, PageSource, PageSourceError};

use crate::host::Blob;

/// The image blob, shared by the VFS and the file pages that read from it.
pub type SharedBlob = Rc<RefCell<Box<dyn Blob>>>;

/// A regular file's contents, frozen at the moment they were taken (cheap: nothing is read or
/// copied). See [`super::Vfs::file_contents`].
#[derive(Clone)]
pub enum FileContents {
    /// `len` bytes at `base` in the image blob (which never changes).
    Image {
        blob: SharedBlob,
        base: u64,
        len: u64,
    },
    /// An in-memory file's contents (a later write to the file copies it first).
    Mem(Rc<Vec<u8>>),
}

impl FileContents {
    /// The size in bytes.
    pub fn len(&self) -> u64 {
        match self {
            Self::Image { len, .. } => *len,
            Self::Mem(v) => v.len() as u64,
        }
    }

    /// `true` for an empty file.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pages from byte `start`, with the file's bytes up to `end` (clipped to its size) and
    /// zeros after: page `k` holds bytes `start + k * PAGE_SIZE ..`. Nothing is read until a
    /// page is asked for.
    pub fn pages(&self, start: u64, end: u64) -> Rc<dyn PageSource> {
        Rc::new(FilePages {
            data: self.clone(),
            start,
            end: end.min(self.len()),
        })
    }
}

/// Pages of a file from byte `start`, with its bytes up to `end` and zeros after.
struct FilePages {
    data: FileContents,
    start: u64,
    end: u64,
}

impl PageSource for FilePages {
    fn page(&self, index: u64) -> Result<PageRef, PageSourceError> {
        let mut page = [0u8; PAGE_SIZE as usize];
        let from = index
            .checked_mul(PAGE_SIZE)
            .and_then(|o| o.checked_add(self.start))
            .ok_or(PageSourceError)?;
        let to = from.saturating_add(PAGE_SIZE).min(self.end);
        if from < to {
            let n = (to - from) as usize;
            match &self.data {
                FileContents::Image { blob, base, .. } => blob
                    .try_borrow_mut()
                    .map_err(|_| PageSourceError)?
                    .read_at(base + from, &mut page[..n])
                    .map_err(|_| PageSourceError)?,
                FileContents::Mem(v) => {
                    page[..n].copy_from_slice(&v[from as usize..to as usize]);
                }
            }
        }
        Ok(Rc::new(page))
    }
}
