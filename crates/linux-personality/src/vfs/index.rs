//! The image index: a table of every member of a tar image, stored as the image's
//! first member, so mounting reads one small member instead of every header of the archive.
//!
//! An indexed image is still an ordinary tar archive. Its first member is a regular file named
//! [`INDEX_NAME`] whose contents are the index; the rest of the archive is unchanged except that
//! it starts later. The index lists, in archive order, exactly the entries [`TarReader`] would
//! return for the members after it (with their contents' offsets in the whole archive), and
//! where those members end, so a mount through the index builds the same tree as a scan. The
//! index member itself never appears in the filesystem. Images without one are scanned.
//!
//! **Concatenated archives** (an embedder may compose an image from a base tar, layer tars
//! and the user's files, each without its end-of-archive blocks; later members win) are read
//! as: the index at offset 0 for the region it describes, then a scan of everything after that
//! region. An index anywhere else (a layer cut from another indexed image carries that image's
//! index, with offsets into *that* image) is ignored, as is any index that does not decode or
//! describes more than the archive holds.
//!
//! Format (version 2, little-endian): the magic `LXIMGIX2`, `end: u64` (where the described
//! members end), a `u32` entry count, then per entry: `kind: u8`, `mode`, `uid`, `gid: u32`,
//! `mtime: i64`, `size`, `data_offset: u64`, `dev_major`, `dev_minor: u32`, `path_len: u32`,
//! `path`, `link_len: u32`, `link`.

use super::tar::{BLOCK, Entry, EntryKind, TarError, TarReader};
use crate::host::{Blob, BlobError};

/// The path of the index member (the archive's first member).
pub const INDEX_NAME: &[u8] = b".linux-image-index";
const MAGIC: &[u8; 8] = b"LXIMGIX2";
/// Largest index we read.
const MAX_INDEX: u64 = 64 << 20;

fn kind_code(k: EntryKind) -> u8 {
    match k {
        EntryKind::File => 0,
        EntryKind::HardLink => 1,
        EntryKind::Symlink => 2,
        EntryKind::CharDev => 3,
        EntryKind::BlockDev => 4,
        EntryKind::Dir => 5,
        EntryKind::Fifo => 6,
    }
}

fn kind_of(code: u8) -> Option<EntryKind> {
    Some(match code {
        0 => EntryKind::File,
        1 => EntryKind::HardLink,
        2 => EntryKind::Symlink,
        3 => EntryKind::CharDev,
        4 => EntryKind::BlockDev,
        5 => EntryKind::Dir,
        6 => EntryKind::Fifo,
        _ => return None,
    })
}

/// Encodes `entries` (content offsets moved by `shift`) and the end of the members they
/// describe.
pub fn encode(entries: &[Entry], shift: u64, end: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(24 + entries.len() * 96);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&end.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        out.push(kind_code(e.kind));
        for v in [e.mode, e.uid, e.gid] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&e.mtime.to_le_bytes());
        out.extend_from_slice(&e.size.to_le_bytes());
        out.extend_from_slice(&(e.data_offset + shift).to_le_bytes());
        for v in [e.dev_major, e.dev_minor] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for s in [&e.path, &e.link] {
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s);
        }
    }
    out
}

/// A cursor over the index bytes; every read is checked.
struct Cursor<'a>(&'a [u8]);

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.0.len() < n {
            return None;
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Some(head)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn bytes(&mut self) -> Option<Vec<u8>> {
        let n = self.u32()? as usize;
        Some(self.take(n)?.to_vec())
    }
}

/// Decodes an index of an archive of `archive_len` bytes: its entries and where the members
/// it describes end. That end, and every content range, must lie in the archive.
pub fn decode(data: &[u8], archive_len: u64) -> Option<(Vec<Entry>, u64)> {
    let mut c = Cursor(data);
    if c.take(8)? != MAGIC {
        return None;
    }
    let end = c.u64()?;
    if end > archive_len {
        return None;
    }
    let count = c.u32()? as usize;
    // Each entry takes at least 53 bytes: a count larger than that is malformed.
    if count > data.len() / 53 {
        return None;
    }
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let kind = kind_of(c.u8()?)?;
        let (mode, uid, gid) = (c.u32()?, c.u32()?, c.u32()?);
        let mtime = c.u64()? as i64;
        let size = c.u64()?;
        let data_offset = c.u64()?;
        let (dev_major, dev_minor) = (c.u32()?, c.u32()?);
        let path = c.bytes()?;
        let link = c.bytes()?;
        if data_offset.checked_add(size)? > end {
            return None;
        }
        entries.push(Entry {
            path,
            kind,
            mode,
            uid,
            gid,
            mtime,
            size,
            data_offset,
            link,
            dev_major,
            dev_minor,
        });
    }
    c.0.is_empty().then_some((entries, end))
}

/// The members of the archive in `blob`, in order: those an index at offset 0 describes (one
/// read), then every member after that region by reading its header (see the module docs for
/// concatenated archives). Index members themselves are never listed.
pub fn entries(blob: &mut dyn Blob) -> Result<Vec<Entry>, TarError> {
    let archive_len = blob.len();
    let mut reader = TarReader::new(blob);
    let mut entries = Vec::new();
    let mut first = true;
    while let Some(e) = reader.next_entry()? {
        let is_index = e.path == INDEX_NAME && e.kind == EntryKind::File;
        if is_index && first && e.size <= MAX_INDEX {
            let mut data = vec![0u8; e.size as usize];
            reader.read_at(e.data_offset, &mut data)?;
            if let Some((listed, end)) = decode(&data, archive_len)
                && end >= reader.pos()
            {
                entries.extend(listed);
                reader.seek(end);
            }
        }
        first = false;
        if !is_index {
            entries.push(e);
        }
    }
    Ok(entries)
}

/// `tar` with an index member in front (see the module docs). An archive that already has
/// one is returned with its index rebuilt.
pub fn with_index(tar: &[u8]) -> Result<Vec<u8>, TarError> {
    let mut blob = SliceBlob(tar);
    let mut reader = TarReader::new(&mut blob);
    let mut entries = Vec::new();
    let mut body_start = 0;
    while let Some(e) = reader.next_entry()? {
        if e.path == INDEX_NAME && e.kind == EntryKind::File {
            if entries.is_empty() {
                // An old index: drop it, and everything moves forward.
                body_start = e.data_offset + e.size.div_ceil(BLOCK) * BLOCK;
            }
            continue;
        }
        entries.push(e);
    }
    let data_end = reader.pos();
    let len = encode(&entries, 0, 0).len() as u64;
    let shift = BLOCK + len.div_ceil(BLOCK) * BLOCK;
    let index = encode(
        &entries
            .iter()
            .map(|e| Entry {
                data_offset: e.data_offset - body_start,
                ..e.clone()
            })
            .collect::<Vec<_>>(),
        shift,
        data_end - body_start + shift,
    );
    let mut out = Vec::with_capacity(tar.len() + shift as usize);
    out.extend_from_slice(&header(INDEX_NAME, index.len() as u64));
    out.extend_from_slice(&index);
    out.resize(shift as usize, 0);
    out.extend_from_slice(&tar[body_start as usize..]);
    Ok(out)
}

/// A ustar header for a read-only regular file (fixed metadata: the image is deterministic).
fn header(name: &[u8], size: u64) -> [u8; 512] {
    let mut h = [0u8; 512];
    h[..name.len()].copy_from_slice(name);
    let mut octal = |at: usize, len: usize, v: u64| {
        let s = format!("{:0width$o}\0", v, width = len - 1);
        h[at..at + len].copy_from_slice(s.as_bytes());
    };
    octal(100, 8, 0o444);
    octal(108, 8, 0);
    octal(116, 8, 0);
    octal(124, 12, size);
    octal(136, 12, 0);
    h[156] = b'0';
    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");
    h[148..156].fill(b' ');
    let sum: u64 = h.iter().map(|&b| u64::from(b)).sum();
    h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
    h
}

/// A blob over bytes in memory.
pub struct SliceBlob<'a>(pub &'a [u8]);

impl Blob for SliceBlob<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), BlobError> {
        let start = usize::try_from(offset).map_err(|_| BlobError::OutOfRange)?;
        let src = start
            .checked_add(buf.len())
            .and_then(|end| self.0.get(start..end))
            .ok_or(BlobError::OutOfRange)?;
        buf.copy_from_slice(src);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::tar::tests::build_tar;
    use super::*;

    fn sample() -> Vec<u8> {
        build_tar(&[
            ("./", b'5', 0o755, b""),
            ("bin/", b'5', 0o755, b""),
            ("bin/hello", b'0', 0o755, b"hello world\n"),
            ("bin/hi", b'2', 0o777, b"hello"),
            ("etc/hosts", b'0', 0o644, &[b'x'; 700]),
            ("bin/again", b'1', 0o644, b"bin/hello"),
        ])
    }

    #[test]
    fn an_indexed_archive_lists_the_same_entries() {
        let tar = sample();
        let plain = entries(&mut SliceBlob(&tar)).unwrap();
        let indexed = with_index(&tar).unwrap();
        assert_eq!(indexed.len() % 512, 0);
        let via_index = entries(&mut SliceBlob(&indexed)).unwrap();
        assert_eq!(via_index.len(), plain.len());
        for (a, b) in plain.iter().zip(&via_index) {
            assert_eq!(a.path, b.path);
            let shifted = b.data_offset - (indexed.len() - tar.len()) as u64;
            assert_eq!(
                Entry {
                    data_offset: shifted,
                    ..b.clone()
                },
                *a
            );
            assert_eq!(
                tar[a.data_offset as usize..(a.data_offset + a.size) as usize],
                indexed[b.data_offset as usize..(b.data_offset + b.size) as usize]
            );
        }
        // A scan of the indexed archive (as an index-unaware reader sees it, minus the index
        // member) gives the same entries as the index.
        let mut blob = SliceBlob(&indexed);
        let mut reader = TarReader::new(&mut blob);
        let mut scanned = Vec::new();
        while let Some(e) = reader.next_entry().unwrap() {
            if e.path != INDEX_NAME {
                scanned.push(e);
            }
        }
        assert_eq!(scanned, via_index);
        // Re-indexing is idempotent.
        assert_eq!(with_index(&indexed).unwrap(), indexed);
    }

    #[test]
    fn malformed_indexes_are_rejected() {
        let tar = with_index(&sample()).unwrap();
        let len = tar.len() as u64;
        let index_len = u64::from_str_radix(
            std::str::from_utf8(&tar[124..135])
                .unwrap()
                .trim_matches('\0'),
            8,
        )
        .unwrap() as usize;
        let index = &tar[512..512 + index_len];
        assert!(decode(index, len).is_some());
        assert!(decode(&index[..index.len() - 1], len).is_none());
        assert!(decode(index, 100).is_none());
        let mut bad = index.to_vec();
        bad[0] = b'X';
        assert!(decode(&bad, len).is_none());
        let mut bad = index.to_vec();
        bad[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(&bad, len).is_none());
        // An index claiming more than the archive holds.
        let mut bad = index.to_vec();
        bad[8..16].copy_from_slice(&(len + 1).to_le_bytes());
        assert!(decode(&bad, len).is_none());
    }

    /// `tar` up to where its members end (without the end-of-archive blocks).
    fn data_part(tar: &[u8]) -> &[u8] {
        let mut blob = SliceBlob(tar);
        let mut reader = TarReader::new(&mut blob);
        while reader.next_entry().unwrap().is_some() {}
        let end = reader.pos() as usize;
        &tar[..end]
    }

    /// The index member's record (header, contents, padding) at the start of an indexed tar.
    fn index_record(tar: &[u8]) -> &[u8] {
        let size = u64::from_str_radix(
            std::str::from_utf8(&tar[124..135])
                .unwrap()
                .trim_matches('\0'),
            8,
        )
        .unwrap();
        &tar[..(BLOCK + size.div_ceil(BLOCK) * BLOCK) as usize]
    }

    /// A layered composition: a base image, a layer
    /// cut from another indexed image (its index member first, then the members it adds or
    /// changes) and the user's files, each without end-of-archive blocks.
    #[test]
    fn concatenated_archives_mount_every_part() {
        use crate::vfs::{ROOT, Vfs};
        let base = with_index(&sample()).unwrap();
        let added: &[(&str, u8, u32, &[u8])] = &[
            ("usr/", b'5', 0o755, b""),
            ("usr/bin/", b'5', 0o755, b""),
            ("usr/bin/python3", b'0', 0o755, b"python\n"),
            ("etc/hosts", b'0', 0o644, b"layer hosts\n"),
        ];
        let mut full = data_part(&sample()).to_vec();
        full.extend_from_slice(&build_tar(added));
        let image = with_index(&full).unwrap();
        let mut layer = index_record(&image).to_vec();
        layer.extend_from_slice(data_part(&build_tar(added)));
        let extra = build_tar(&[
            ("home/", b'5', 0o755, b""),
            ("home/notes", b'0', 0o644, b"mine\n"),
        ]);
        let mut composed = data_part(&base).to_vec();
        composed.extend_from_slice(&layer);
        composed.extend_from_slice(&extra);

        let mut vfs = Vfs::new();
        vfs.mount_image(Box::new(super::super::tar::tests::MemBlob(composed)))
            .unwrap();
        let read = |vfs: &mut Vfs, path: &[u8]| {
            let ino = vfs.resolve(ROOT, path, true).unwrap();
            vfs.read_all(ino).unwrap()
        };
        assert_eq!(read(&mut vfs, b"/bin/hello"), b"hello world\n");
        assert_eq!(read(&mut vfs, b"/usr/bin/python3"), b"python\n");
        assert_eq!(read(&mut vfs, b"/etc/hosts"), b"layer hosts\n");
        assert_eq!(read(&mut vfs, b"/home/notes"), b"mine\n");
        assert!(vfs.resolve(ROOT, b"/.linux-image-index", true).is_err());

        // The base alone mounts through its index; with no index at the start, everything is
        // scanned (and a stray index member is never a file).
        let parts = entries(&mut SliceBlob(&base)).unwrap();
        assert_eq!(
            parts.len(),
            entries(&mut SliceBlob(&sample())).unwrap().len()
        );
        let mut stray = data_part(&sample()).to_vec();
        stray.extend_from_slice(&layer);
        let listed = entries(&mut SliceBlob(&stray)).unwrap();
        assert!(listed.iter().any(|e| e.path == b"usr/bin/python3"));
        assert!(listed.iter().all(|e| e.path != INDEX_NAME));
    }
}
