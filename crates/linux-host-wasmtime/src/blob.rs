//! `linux:host/blob` served from files: a host that keeps root filesystem images on disk (or
//! in a content-addressed store) instead of compiling them into the node. Reads are positional
//! `(name, offset, len)`, so any store that can serve byte ranges can stand behind it.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;

use wasmtime::component::{HasData, Linker};

mod bindings {
    wasmtime::component::bindgen!({
        world: "node-host",
        path: "../../wit/linux-host",
    });
}

use bindings::linux::host::blob;

/// Named read-only blobs backed by files.
#[derive(Default)]
pub struct FileBlobs {
    files: BTreeMap<String, (File, u64)>,
}

impl FileBlobs {
    /// Serves `path` as blob `name`.
    pub fn add(&mut self, name: &str, path: impl Into<PathBuf>) -> std::io::Result<()> {
        let path = path.into();
        let file = File::open(&path)?;
        let len = file.metadata()?.len();
        self.files.insert(name.to_owned(), (file, len));
        Ok(())
    }
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(f, buf, offset)
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(f, buf, offset)
}

impl blob::Host for FileBlobs {
    fn size(&mut self, name: String) -> Result<u64, blob::Error> {
        self.files
            .get(&name)
            .map(|(_, len)| *len)
            .ok_or(blob::Error::NotFound)
    }

    fn read(&mut self, name: String, offset: u64, len: u32) -> Result<Vec<u8>, blob::Error> {
        let (file, size) = self.files.get(&name).ok_or(blob::Error::NotFound)?;
        let end = offset
            .checked_add(u64::from(len))
            .ok_or(blob::Error::OutOfRange)?;
        if end > *size {
            return Err(blob::Error::OutOfRange);
        }
        let mut buf = vec![0u8; len as usize];
        let mut done = 0;
        while done < buf.len() {
            match read_at(file, &mut buf[done..], offset + done as u64) {
                Ok(0) => return Err(blob::Error::Io("short read".into())),
                Ok(n) => done += n,
                Err(e) => return Err(blob::Error::Io(e.to_string())),
            }
        }
        Ok(buf)
    }
}

/// Links `linux:host/blob` (served by the [`FileBlobs`] that `get` returns).
pub fn add_to_linker<T: 'static>(
    linker: &mut Linker<T>,
    get: fn(&mut T) -> &mut FileBlobs,
) -> wasmtime::Result<()> {
    struct Blobs;
    impl HasData for Blobs {
        type Data<'a> = &'a mut FileBlobs;
    }
    blob::add_to_linker::<T, Blobs>(linker, get)
}
