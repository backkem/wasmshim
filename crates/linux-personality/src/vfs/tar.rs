//! A streaming reader for (uncompressed) POSIX tar archives: ustar, pax (`x`/`g`) and GNU
//! long names (`L`/`K`). Only headers are read; file contents stay in the blob and are
//! referenced by offset.

use crate::host::{Blob, BlobError};

/// The tar block size.
pub(crate) const BLOCK: u64 = 512;

/// One pax extended-header record: key and value.
type PaxRecord = (Vec<u8>, Vec<u8>);

/// The kind of a tar entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    File,
    HardLink,
    Symlink,
    CharDev,
    BlockDev,
    Dir,
    Fifo,
}

/// One archive member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub path: Vec<u8>,
    pub kind: EntryKind,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub mtime: i64,
    pub size: u64,
    /// Offset of the content in the blob.
    pub data_offset: u64,
    /// Symlink target or hard-link source.
    pub link: Vec<u8>,
    pub dev_major: u32,
    pub dev_minor: u32,
}

/// A malformed archive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TarError {
    #[error("blob read failed: {0}")]
    Blob(#[from] BlobError),
    #[error("bad tar header at offset {0}")]
    BadHeader(u64),
}

/// Parses an octal (or GNU base-256) numeric field.
fn number(field: &[u8]) -> Option<u64> {
    if field.first().is_some_and(|b| b & 0x80 != 0) {
        // GNU base-256: big-endian binary after the marker bit.
        let mut v: u64 = u64::from(field[0] & 0x7f);
        for &b in &field[1..] {
            v = v.checked_mul(256)?.checked_add(u64::from(b))?;
        }
        return Some(v);
    }
    let mut v: u64 = 0;
    let mut seen = false;
    for &b in field {
        match b {
            b'0'..=b'7' => {
                v = v.checked_mul(8)?.checked_add(u64::from(b - b'0'))?;
                seen = true;
            }
            b' ' | 0 if !seen => {}
            b' ' | 0 => break,
            _ => return None,
        }
    }
    Some(v)
}

fn cstr(field: &[u8]) -> &[u8] {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    &field[..end]
}

/// Iterates the members of a tar archive stored in a blob.
pub struct TarReader<'b> {
    blob: &'b mut dyn Blob,
    pos: u64,
    done: bool,
}

impl<'b> TarReader<'b> {
    pub fn new(blob: &'b mut dyn Blob) -> Self {
        Self {
            blob,
            pos: 0,
            done: false,
        }
    }

    fn read_pax(&mut self, size: u64, at: u64) -> Result<Vec<PaxRecord>, TarError> {
        let len = usize::try_from(size).map_err(|_| TarError::BadHeader(at))?;
        if len > 1 << 20 {
            return Err(TarError::BadHeader(at));
        }
        let mut data = vec![0u8; len];
        self.blob.read_at(self.pos, &mut data)?;
        let mut out = Vec::new();
        let mut rest = &data[..];
        while !rest.is_empty() {
            let sp = rest
                .iter()
                .position(|&b| b == b' ')
                .ok_or(TarError::BadHeader(at))?;
            let n: usize = std::str::from_utf8(&rest[..sp])
                .ok()
                .and_then(|s| s.parse().ok())
                .ok_or(TarError::BadHeader(at))?;
            if n <= sp + 1 || n > rest.len() {
                return Err(TarError::BadHeader(at));
            }
            let record = &rest[sp + 1..n - 1]; // without the trailing newline
            if let Some(eq) = record.iter().position(|&b| b == b'=') {
                out.push((record[..eq].to_vec(), record[eq + 1..].to_vec()));
            }
            rest = &rest[n..];
        }
        Ok(out)
    }

    fn read_long(&mut self, size: u64, at: u64) -> Result<Vec<u8>, TarError> {
        let len = usize::try_from(size).map_err(|_| TarError::BadHeader(at))?;
        if len > 1 << 16 {
            return Err(TarError::BadHeader(at));
        }
        let mut data = vec![0u8; len];
        self.blob.read_at(self.pos, &mut data)?;
        Ok(cstr(&data).to_vec())
    }

    /// Where the next header is read (after the last member: where the archive's data ends).
    pub fn pos(&self) -> u64 {
        self.pos
    }

    /// Continues reading headers at `pos` (a member boundary).
    pub fn seek(&mut self, pos: u64) {
        self.pos = pos;
        self.done = false;
    }

    /// Reads bytes of the archive (e.g. a member's contents) without moving the reader.
    pub fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), BlobError> {
        self.blob.read_at(offset, buf)
    }

    /// Moves past the `size` bytes of data of the member whose header is at `at`, padded to a
    /// block. The data must lie within the blob: a size that reaches past it (or wraps a
    /// position, like a GNU base-256 size near 2^64) is a bad header.
    fn skip_data(&mut self, size: u64, at: u64) -> Result<(), TarError> {
        let bad = TarError::BadHeader(at);
        if self
            .pos
            .checked_add(size)
            .is_none_or(|end| end > self.blob.len())
        {
            return Err(bad);
        }
        let padded = size.checked_next_multiple_of(BLOCK).ok_or(bad)?;
        self.pos = self.pos.checked_add(padded).ok_or(bad)?;
        Ok(())
    }

    /// The next member, `Ok(None)` at the end of the archive.
    pub fn next_entry(&mut self) -> Result<Option<Entry>, TarError> {
        let mut long_name: Option<Vec<u8>> = None;
        let mut long_link: Option<Vec<u8>> = None;
        let mut pax_size: Option<u64> = None;
        loop {
            let header_end = self.pos.checked_add(BLOCK);
            if self.done || header_end.is_none_or(|end| end > self.blob.len()) {
                self.done = true;
                return Ok(None);
            }
            let at = self.pos;
            let mut h = [0u8; 512];
            self.blob.read_at(at, &mut h)?;
            if h.iter().all(|&b| b == 0) {
                self.done = true;
                return Ok(None);
            }
            let stored = number(&h[148..156]).ok_or(TarError::BadHeader(at))?;
            let sum: u64 = h
                .iter()
                .enumerate()
                .map(|(i, &b)| {
                    if (148..156).contains(&i) {
                        32
                    } else {
                        u64::from(b)
                    }
                })
                .sum();
            if stored != sum {
                return Err(TarError::BadHeader(at));
            }
            let mut size = number(&h[124..136]).ok_or(TarError::BadHeader(at))?;
            let typeflag = h[156];
            // In the blob: checked when the loop began.
            self.pos = at + BLOCK;
            let data_offset = self.pos;
            match typeflag {
                b'x' => {
                    for (k, v) in self.read_pax(size, at)? {
                        match &k[..] {
                            b"path" => long_name = Some(v),
                            b"linkpath" => long_link = Some(v),
                            b"size" => {
                                pax_size = std::str::from_utf8(&v).ok().and_then(|s| s.parse().ok())
                            }
                            _ => {}
                        }
                    }
                    self.skip_data(size, at)?;
                    continue;
                }
                b'g' => {
                    self.skip_data(size, at)?;
                    continue;
                }
                b'L' => {
                    long_name = Some(self.read_long(size, at)?);
                    self.skip_data(size, at)?;
                    continue;
                }
                b'K' => {
                    long_link = Some(self.read_long(size, at)?);
                    self.skip_data(size, at)?;
                    continue;
                }
                _ => {}
            }
            if let Some(s) = pax_size {
                size = s;
            }
            let kind = match typeflag {
                b'0' | 0 | b'7' => EntryKind::File,
                b'1' => EntryKind::HardLink,
                b'2' => EntryKind::Symlink,
                b'3' => EntryKind::CharDev,
                b'4' => EntryKind::BlockDev,
                b'5' => EntryKind::Dir,
                b'6' => EntryKind::Fifo,
                _ => {
                    // Unknown member types are skipped, like GNU tar does with a warning.
                    self.skip_data(size, at)?;
                    long_name = None;
                    long_link = None;
                    pax_size = None;
                    continue;
                }
            };
            let path = long_name.take().unwrap_or_else(|| {
                let name = cstr(&h[0..100]);
                let prefix = if &h[257..262] == b"ustar" {
                    cstr(&h[345..500])
                } else {
                    &[]
                };
                if prefix.is_empty() {
                    name.to_vec()
                } else {
                    let mut p = prefix.to_vec();
                    p.push(b'/');
                    p.extend_from_slice(name);
                    p
                }
            });
            let link = long_link
                .take()
                .unwrap_or_else(|| cstr(&h[157..257]).to_vec());
            let field = |r: core::ops::Range<usize>| number(&h[r]).ok_or(TarError::BadHeader(at));
            let has_data = matches!(kind, EntryKind::File);
            let entry = Entry {
                path,
                kind,
                mode: (field(100..108)? & 0o7777) as u32,
                uid: field(108..116)? as u32,
                gid: field(116..124)? as u32,
                mtime: field(136..148)? as i64,
                size: if has_data { size } else { 0 },
                data_offset,
                link,
                dev_major: field(329..337).unwrap_or(0) as u32,
                dev_minor: field(337..345).unwrap_or(0) as u32,
            };
            self.skip_data(size, at)?;
            return Ok(Some(entry));
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// An in-memory blob for tests.
    pub struct MemBlob(pub Vec<u8>);

    impl Blob for MemBlob {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
        fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), BlobError> {
            let start = usize::try_from(offset).map_err(|_| BlobError::OutOfRange)?;
            let src = self
                .0
                .get(start..start + buf.len())
                .ok_or(BlobError::OutOfRange)?;
            buf.copy_from_slice(src);
            Ok(())
        }
    }

    /// Builds a ustar archive (test helper): `(path, typeflag, mode, data_or_link)`.
    pub fn build_tar(entries: &[(&str, u8, u32, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for &(path, ty, mode, data) in entries {
            let mut h = [0u8; 512];
            h[..path.len()].copy_from_slice(path.as_bytes());
            let oct = |h: &mut [u8; 512], at: usize, len: usize, v: u64| {
                let s = format!("{:0width$o}\0", v, width = len - 1);
                h[at..at + len].copy_from_slice(s.as_bytes());
            };
            oct(&mut h, 100, 8, u64::from(mode));
            oct(&mut h, 108, 8, 0);
            oct(&mut h, 116, 8, 0);
            let size = if ty == b'0' { data.len() as u64 } else { 0 };
            oct(&mut h, 124, 12, size);
            oct(&mut h, 136, 12, 1_700_000_000);
            h[156] = ty;
            if ty == b'2' || ty == b'1' {
                h[157..157 + data.len()].copy_from_slice(data);
            }
            h[257..263].copy_from_slice(b"ustar\0");
            h[263..265].copy_from_slice(b"00");
            h[148..156].fill(b' ');
            let sum: u64 = h.iter().map(|&b| u64::from(b)).sum();
            let s = format!("{sum:06o}\0 ");
            h[148..156].copy_from_slice(s.as_bytes());
            out.extend_from_slice(&h);
            if ty == b'0' {
                out.extend_from_slice(data);
                out.resize(out.len().div_ceil(512) * 512, 0);
            }
        }
        out.extend_from_slice(&[0; 1024]);
        out
    }

    #[test]
    fn reads_members() {
        let tar = build_tar(&[
            ("bin/", b'5', 0o755, b""),
            ("bin/hello", b'0', 0o755, b"hi there"),
            ("bin/sh", b'2', 0o777, b"hello"),
        ]);
        let mut blob = MemBlob(tar);
        let mut r = TarReader::new(&mut blob);
        let d = r.next_entry().unwrap().unwrap();
        assert_eq!((d.path.as_slice(), d.kind), (&b"bin/"[..], EntryKind::Dir));
        let f = r.next_entry().unwrap().unwrap();
        assert_eq!((f.size, f.mode, f.mtime), (8, 0o755, 1_700_000_000));
        let s = r.next_entry().unwrap().unwrap();
        assert_eq!(
            (s.kind, s.link.as_slice()),
            (EntryKind::Symlink, &b"hello"[..])
        );
        assert_eq!(r.next_entry().unwrap(), None);
        let data_off = f.data_offset as usize;
        assert_eq!(&blob.0[data_off..data_off + 8], b"hi there");
    }

    /// `tar` with the size of the member whose header is at `at` set to `size` in GNU base-256
    /// (a marker bit, then big-endian binary), and its typeflag set to `ty`.
    fn with_size(mut tar: Vec<u8>, at: usize, ty: u8, size: u64) -> Vec<u8> {
        let h = &mut tar[at..at + 512];
        h[124..136].fill(0);
        h[124] = 0x80;
        h[128..136].copy_from_slice(&size.to_be_bytes());
        h[156] = ty;
        h[148..156].fill(b' ');
        let sum: u64 = h.iter().map(|&b| u64::from(b)).sum();
        h[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
        tar
    }

    #[test]
    fn a_member_reaching_past_the_blob_is_a_bad_header() {
        // A size near 2^64 would move the position back onto the same header (and loop for
        // ever), or past the end of the blob: both are malformed archives.
        for ty in *b"g05Z" {
            for size in [u64::MAX - 511, u64::MAX - 1023, u64::MAX, 1 << 40, 2000] {
                let tar = build_tar(&[("x", b'0', 0o644, b"1"), ("y", b'0', 0o644, b"2")]);
                let tar = with_size(tar, 1024, ty, size);
                let mut blob = MemBlob(tar);
                let mut r = TarReader::new(&mut blob);
                assert!(r.next_entry().unwrap().is_some());
                assert_eq!(
                    r.next_entry(),
                    Err(TarError::BadHeader(1024)),
                    "typeflag {ty} size {size:#x}"
                );
            }
        }
    }

    #[test]
    fn rejects_bad_checksum() {
        let mut tar = build_tar(&[("x", b'0', 0o644, b"1")]);
        tar[0] = b'y';
        let mut blob = MemBlob(tar);
        assert_eq!(
            TarReader::new(&mut blob).next_entry(),
            Err(TarError::BadHeader(0))
        );
    }
}
