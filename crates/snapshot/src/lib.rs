//! One snapshot container for every saved machine: a Linux checkpoint, or the state of any
//! other emulated machine an embedder chooses to save. A snapshot is one deterministic CBOR map
//! (`wasmshim-cbor`):
//!
//! | key | value |
//! |---|---|
//! | `"format"` | the container format: [`DEFAULT_FORMAT`], or the name an embedder gives its own (`Snapshot::new_in`) |
//! | `"kind"` | the kind of state, named by whoever saves it (`"linux"`, ...) |
//! | `"version"` | the kind's own body version |
//! | `"meta"` | text → scalar: what the state is valid for (an image, a program) |
//! | `"body"` | the kind's own state, without bulk memory |
//! | `"pages"` | `[key, digest]` pairs, keys strictly increasing: bulk memory by reference |
//! | `"chunks"` | digest → bytes, each referenced chunk once (absent in a manifest) |
//! | `"encoding"` | `"deflate"`: `body` and every chunk are raw DEFLATE (RFC 1951) of their bytes (absent: as they are) |
//!
//! A chunk is named by the SHA-256 of its bytes (one content-addressing scheme). The
//! snapshot's identity ([`Snapshot::id`]) is the SHA-256 of its manifest (the encoding without
//! `"chunks"`), so it is the same whether the chunks travel inline or live in a store.
//!
//! The compact encoding ([`Snapshot::encode_deflated`]) is a property of the
//! bytes on disk and on the wire only: a chunk keeps the name of its plain bytes and the
//! identity is that of the plain manifest, so a snapshot is the same snapshot in either
//! encoding.
//!
//! The container never interprets a body. Decoding is strict: anything the encoder would not
//! produce, a missing, corrupt or unreferenced chunk, or another kind or version is an error.

#![no_std]

extern crate alloc;

mod sha256;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

pub use sha256::sha256;
use wasmshim_cbor::Value;

/// A chunk's name: the SHA-256 of its bytes.
pub type Digest = [u8; 32];

/// The container format this crate writes and reads unless told another name: an embedder that
/// keeps states under its own name (states it already saved) passes that name to
/// [`Snapshot::new_in`] and [`Snapshot::decode_in`]. The format names the container, never a
/// body: the bytes are the same layout under any name.
pub const DEFAULT_FORMAT: &str = "snapshot/1";

/// The compact encoding's name (`"encoding"`), and its DEFLATE level: fixed, since the bytes it
/// writes are pinned (one snapshot, one compact byte form).
const DEFLATE: &str = "deflate";
const DEFLATE_LEVEL: u8 = 6;
/// The most a deflated field may inflate to (a bound for untrusted bytes).
const INFLATE_LIMIT: usize = 1 << 31;
/// The most all deflated fields of one snapshot may inflate to together: more than a 32-bit
/// host can hold, so every snapshot such a host could load still loads, while a small file can
/// no longer demand 2 GiB for every chunk it lists before the hashes are checked.
const INFLATE_TOTAL: u64 = 1 << 32;

fn deflate(bytes: &[u8]) -> Vec<u8> {
    miniz_oxide::deflate::compress_to_vec(bytes, DEFLATE_LEVEL)
}

/// Inflates one field to at most [`INFLATE_LIMIT`] bytes and at most `*left`, which it then
/// reduces by what it produced.
fn inflate(bytes: &[u8], left: &mut u64) -> Result<Vec<u8>, Error> {
    let limit = usize::try_from(*left).map_or(INFLATE_LIMIT, |l| l.min(INFLATE_LIMIT));
    let out = miniz_oxide::inflate::decompress_to_vec_with_limit(bytes, limit)
        .map_err(|_| Error::Malformed("a deflated field"))?;
    *left -= out.len() as u64;
    Ok(out)
}

/// Why a snapshot cannot be read or used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Not deterministic CBOR.
    Cbor,
    /// Not this container (or another container format).
    Format,
    /// A field is missing or has the wrong shape.
    Malformed(&'static str),
    /// Page keys must strictly increase.
    PageOrder,
    /// A chunk a page names is not there.
    MissingChunk(Digest),
    /// A chunk's bytes do not hash to its name.
    CorruptChunk(Digest),
    /// A chunk no page names.
    UnusedChunk(Digest),
    /// The snapshot is of another kind.
    Kind { found: String, want: &'static str },
    /// The kind's body version is not the one this reader knows.
    Version { found: u64, want: u64 },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        /// The first bytes of a chunk's name, enough to find it.
        fn short(d: &Digest) -> impl core::fmt::Display + '_ {
            struct Short<'a>(&'a Digest);
            impl core::fmt::Display for Short<'_> {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    self.0[..6].iter().try_for_each(|b| write!(f, "{b:02x}"))
                }
            }
            Short(d)
        }
        match self {
            Self::Cbor => f.write_str("not deterministic CBOR"),
            Self::Format => f.write_str("not a snapshot of this container format"),
            Self::Malformed(what) => write!(f, "malformed snapshot: {what}"),
            Self::PageOrder => f.write_str("page keys do not strictly increase"),
            Self::MissingChunk(d) => write!(f, "chunk {}… is missing", short(d)),
            Self::CorruptChunk(d) => write!(f, "chunk {}… does not hash to its name", short(d)),
            Self::UnusedChunk(d) => write!(f, "chunk {}… is named by no page", short(d)),
            Self::Kind { found, want } => write!(f, "a `{found}` snapshot, not `{want}`"),
            Self::Version { found, want } => {
                write!(f, "body version {found}, this reader knows {want}")
            }
        }
    }
}

impl core::error::Error for Error {}

/// A saved machine.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// The container format's name (`"format"`).
    pub format: String,
    pub kind: String,
    pub version: u64,
    /// What the state is only valid for, as scalars (integers, text, bytes, booleans).
    pub meta: BTreeMap<String, Value>,
    pub body: Vec<u8>,
    pages: Vec<(u64, Digest)>,
    chunks: BTreeMap<Digest, Vec<u8>>,
}

impl Snapshot {
    /// An empty snapshot of `kind`, body version `version`, in the [`DEFAULT_FORMAT`].
    pub fn new(kind: &str, version: u64) -> Self {
        Self::new_in(DEFAULT_FORMAT, kind, version)
    }

    /// An empty snapshot of `kind`, body version `version`, in container format `format`.
    pub fn new_in(format: &str, kind: &str, version: u64) -> Self {
        Self {
            format: String::from(format),
            kind: String::from(kind),
            version,
            ..Self::default()
        }
    }

    /// Checks the kind and version before a reader looks at the body.
    pub fn expect(&self, kind: &'static str, version: u64) -> Result<(), Error> {
        if self.kind != kind {
            return Err(Error::Kind {
                found: self.kind.clone(),
                want: kind,
            });
        }
        if self.version != version {
            return Err(Error::Version {
                found: self.version,
                want: version,
            });
        }
        Ok(())
    }

    /// Adds page `key` with `bytes`; keys must strictly increase. Equal contents share one
    /// chunk.
    pub fn add_page(&mut self, key: u64, bytes: &[u8]) -> Result<(), Error> {
        if self.pages.last().is_some_and(|&(k, _)| k >= key) {
            return Err(Error::PageOrder);
        }
        let d = sha256(bytes);
        self.chunks.entry(d).or_insert_with(|| bytes.to_vec());
        self.pages.push((key, d));
        Ok(())
    }

    /// The pages in key order with their bytes (`None` while a chunk is not loaded: a
    /// manifest).
    pub fn pages(&self) -> impl Iterator<Item = (u64, Option<&[u8]>)> + '_ {
        self.pages
            .iter()
            .map(|(k, d)| (*k, self.chunks.get(d).map(Vec::as_slice)))
    }

    /// The number of pages.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// The distinct chunks (by name) and their total size in bytes.
    pub fn chunk_stats(&self) -> (usize, usize) {
        (self.chunks.len(), self.chunks.values().map(Vec::len).sum())
    }

    /// Every chunk a page names is present.
    pub fn is_complete(&self) -> bool {
        self.pages.iter().all(|(_, d)| self.chunks.contains_key(d))
    }

    fn encode_with(&self, chunks: bool, deflated: bool) -> Vec<u8> {
        let field = |b: &[u8]| {
            if deflated {
                Value::Bytes(deflate(b))
            } else {
                Value::Bytes(b.to_vec())
            }
        };
        let meta = Value::Map(
            self.meta
                .iter()
                .map(|(k, v)| (Value::Text(k.clone()), v.clone()))
                .collect(),
        );
        let pages = Value::Array(
            self.pages
                .iter()
                .map(|(k, d)| Value::Array(alloc::vec![Value::Uint(*k), Value::Bytes(d.to_vec())]))
                .collect(),
        );
        let mut entries = alloc::vec![
            (Value::text("format"), Value::Text(self.format.clone())),
            (Value::text("kind"), Value::Text(self.kind.clone())),
            (Value::text("version"), Value::Uint(self.version)),
            (Value::text("meta"), meta),
            (Value::text("body"), field(&self.body)),
            (Value::text("pages"), pages),
        ];
        if deflated {
            entries.push((Value::text("encoding"), Value::text(DEFLATE)));
        }
        if chunks {
            entries.push((
                Value::text("chunks"),
                Value::Map(
                    self.chunks
                        .iter()
                        .map(|(d, b)| (Value::Bytes(d.to_vec()), field(b)))
                        .collect(),
                ),
            ));
        }
        Value::Map(entries).encode()
    }

    /// The whole snapshot, chunks inline.
    pub fn encode(&self) -> Vec<u8> {
        self.encode_with(true, false)
    }

    /// The whole snapshot, chunks inline, compact: the body and every chunk deflated. The same
    /// snapshot as [`Snapshot::encode`]'s (same chunk names, same identity).
    pub fn encode_deflated(&self) -> Vec<u8> {
        self.encode_with(true, true)
    }

    /// The manifest: everything but the chunks (plain).
    pub fn manifest(&self) -> Vec<u8> {
        self.encode_with(false, false)
    }

    /// The snapshot's identity: the SHA-256 of its manifest.
    pub fn id(&self) -> Digest {
        sha256(&self.manifest())
    }

    /// The manifest and the chunks, for a content-addressed store.
    pub fn split(&self) -> (Vec<u8>, BTreeMap<Digest, Vec<u8>>) {
        (self.manifest(), self.chunks.clone())
    }

    /// Reads a snapshot in the [`DEFAULT_FORMAT`] (a whole one, or a manifest whose chunks then
    /// are absent).
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        Self::decode_in(DEFAULT_FORMAT, bytes)
    }

    /// Reads a snapshot in container format `format` (any other is [`Error::Format`]).
    pub fn decode_in(format: &str, bytes: &[u8]) -> Result<Self, Error> {
        Self::decode_limited(format, bytes, INFLATE_TOTAL)
    }

    /// [`Snapshot::decode_in`], deflated fields inflating to at most `total` bytes in all.
    fn decode_limited(format: &str, bytes: &[u8], total: u64) -> Result<Self, Error> {
        let v = wasmshim_cbor::decode(bytes).map_err(|_| Error::Cbor)?;
        let Value::Map(entries) = &v else {
            return Err(Error::Format);
        };
        let known = [
            "format", "kind", "version", "meta", "body", "pages", "chunks", "encoding",
        ];
        if entries
            .iter()
            .any(|(k, _)| !matches!(k, Value::Text(t) if known.contains(&t.as_str())))
        {
            return Err(Error::Malformed("an unknown field"));
        }
        if v.get("format").and_then(Value::as_text) != Some(format) {
            return Err(Error::Format);
        }
        let deflated = match v.get("encoding") {
            None => false,
            Some(e) if e.as_text() == Some(DEFLATE) => true,
            Some(_) => return Err(Error::Malformed("encoding")),
        };
        let mut left = total;
        let mut field = |b: &[u8]| {
            if deflated {
                inflate(b, &mut left)
            } else {
                Ok(b.to_vec())
            }
        };
        let kind = v
            .get("kind")
            .and_then(Value::as_text)
            .ok_or(Error::Malformed("kind"))?;
        let version = v
            .get("version")
            .and_then(Value::as_uint)
            .ok_or(Error::Malformed("version"))?;
        let body = field(
            v.get("body")
                .and_then(Value::as_bytes)
                .ok_or(Error::Malformed("body"))?,
        )?;
        let mut meta = BTreeMap::new();
        match v.get("meta") {
            Some(Value::Map(m)) => {
                for (k, val) in m {
                    let Value::Text(k) = k else {
                        return Err(Error::Malformed("meta key"));
                    };
                    if matches!(val, Value::Array(_) | Value::Map(_)) {
                        return Err(Error::Malformed("meta value"));
                    }
                    meta.insert(k.clone(), val.clone());
                }
            }
            _ => return Err(Error::Malformed("meta")),
        }
        let mut pages = Vec::new();
        match v.get("pages") {
            Some(Value::Array(ps)) => {
                for p in ps {
                    let Value::Array(pair) = p else {
                        return Err(Error::Malformed("page"));
                    };
                    let [Value::Uint(k), Value::Bytes(d)] = pair.as_slice() else {
                        return Err(Error::Malformed("page"));
                    };
                    let d: Digest = d
                        .as_slice()
                        .try_into()
                        .map_err(|_| Error::Malformed("digest"))?;
                    if pages.last().is_some_and(|&(last, _)| last >= *k) {
                        return Err(Error::PageOrder);
                    }
                    pages.push((*k, d));
                }
            }
            _ => return Err(Error::Malformed("pages")),
        }
        let mut chunks = BTreeMap::new();
        match v.get("chunks") {
            None => {}
            Some(Value::Map(cs)) => {
                for (d, b) in cs {
                    let (Value::Bytes(d), Value::Bytes(b)) = (d, b) else {
                        return Err(Error::Malformed("chunk"));
                    };
                    let d: Digest = d
                        .as_slice()
                        .try_into()
                        .map_err(|_| Error::Malformed("digest"))?;
                    let b = field(b)?;
                    if sha256(&b) != d {
                        return Err(Error::CorruptChunk(d));
                    }
                    chunks.insert(d, b);
                }
                let named: alloc::collections::BTreeSet<&Digest> =
                    pages.iter().map(|(_, d)| d).collect();
                if let Some(d) = chunks.keys().find(|d| !named.contains(d)) {
                    return Err(Error::UnusedChunk(*d));
                }
                if let Some((_, d)) = pages.iter().find(|(_, d)| !chunks.contains_key(d)) {
                    return Err(Error::MissingChunk(*d));
                }
            }
            Some(_) => return Err(Error::Malformed("chunks")),
        }
        Ok(Self {
            format: String::from(format),
            kind: String::from(kind),
            version,
            meta,
            body,
            pages,
            chunks,
        })
    }

    /// A whole snapshot from its manifest (in the [`DEFAULT_FORMAT`]) and a store: `get` returns
    /// a chunk's bytes by name.
    pub fn join(
        manifest: &[u8],
        get: impl FnMut(&Digest) -> Option<Vec<u8>>,
    ) -> Result<Self, Error> {
        Self::join_in(DEFAULT_FORMAT, manifest, get)
    }

    /// [`Snapshot::join`] for container format `format`.
    pub fn join_in(
        format: &str,
        manifest: &[u8],
        mut get: impl FnMut(&Digest) -> Option<Vec<u8>>,
    ) -> Result<Self, Error> {
        let mut s = Self::decode_in(format, manifest)?;
        if !s.chunks.is_empty() {
            return Err(Error::Malformed("a manifest with chunks"));
        }
        for (_, d) in &s.pages {
            if s.chunks.contains_key(d) {
                continue;
            }
            let b = get(d).ok_or(Error::MissingChunk(*d))?;
            if sha256(&b) != *d {
                return Err(Error::CorruptChunk(*d));
            }
            s.chunks.insert(*d, b);
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Snapshot {
        let mut s = Snapshot::new("linux", 1);
        s.meta.insert("image".into(), Value::text("conformance"));
        s.meta.insert("pages".into(), Value::Uint(3));
        s.body = b"WNCK body".to_vec();
        s.add_page(0x1000, &[1; 4096]).unwrap();
        s.add_page(0x5000, &[2; 4096]).unwrap();
        s.add_page(0x9000, &[1; 4096]).unwrap(); // the same bytes as 0x1000
        s
    }

    #[test]
    fn round_trips_and_dedups() {
        let s = sample();
        assert_eq!(s.chunk_stats(), (2, 8192));
        let bytes = s.encode();
        let d = Snapshot::decode(&bytes).unwrap();
        assert_eq!(d, s);
        assert_eq!(d.encode(), bytes, "one byte form");
        d.expect("linux", 1).unwrap();
        assert!(d.expect("gb", 1).is_err());
        assert!(d.expect("linux", 2).is_err());
        let pages: Vec<_> = d.pages().map(|(k, b)| (k, b.unwrap()[0])).collect();
        assert_eq!(pages, [(0x1000, 1), (0x5000, 2), (0x9000, 1)]);
    }

    #[test]
    fn identity_does_not_depend_on_where_chunks_live() {
        let s = sample();
        let (manifest, chunks) = s.split();
        let m = Snapshot::decode(&manifest).unwrap();
        assert!(!m.is_complete());
        assert_eq!(m.id(), s.id());
        let j = Snapshot::join(&manifest, |d| chunks.get(d).cloned()).unwrap();
        assert_eq!(j, s);
        assert_eq!(
            Snapshot::join(&manifest, |_| None),
            Err(Error::MissingChunk(sha256(&[1; 4096])))
        );
        assert!(matches!(
            Snapshot::join(&manifest, |_| Some(alloc::vec![0; 4096])),
            Err(Error::CorruptChunk(_))
        ));
        let mut other = s.clone();
        other.body.push(0);
        assert_ne!(other.id(), s.id());
    }

    #[test]
    fn the_compact_encoding_is_the_same_snapshot() {
        let mut s = sample();
        s.body = alloc::vec![7; 10_000];
        let plain = s.encode();
        let compact = s.encode_deflated();
        assert!(
            compact.len() < plain.len() / 4,
            "{} vs {}",
            compact.len(),
            plain.len()
        );
        let d = Snapshot::decode(&compact).unwrap();
        assert_eq!(d, s, "the same snapshot");
        assert_eq!(d.id(), s.id(), "the same identity");
        assert_eq!(d.encode_deflated(), compact, "one compact byte form");
        assert_eq!(d.encode(), plain);
        // Another encoding name, or a deflated field that does not inflate, is refused.
        let mut v = wasmshim_cbor::decode(&compact).unwrap();
        if let Value::Map(entries) = &mut v {
            for (k, val) in entries.iter_mut() {
                if k.as_text() == Some("encoding") {
                    *val = Value::text("zstd");
                }
            }
        }
        assert_eq!(
            Snapshot::decode(&v.encode()),
            Err(Error::Malformed("encoding"))
        );
        let mut v = wasmshim_cbor::decode(&compact).unwrap();
        if let Value::Map(entries) = &mut v {
            for (k, val) in entries.iter_mut() {
                if k.as_text() == Some("body") {
                    *val = Value::Bytes(alloc::vec![0xff, 0xff, 0xff]);
                }
            }
        }
        assert_eq!(
            Snapshot::decode(&v.encode()),
            Err(Error::Malformed("a deflated field"))
        );
    }

    #[test]
    fn deflated_fields_share_one_inflate_budget() {
        let mut s = sample();
        s.body = alloc::vec![7; 10_000];
        let compact = s.encode_deflated();
        // The body and the two distinct 4 KiB chunks: 18_192 bytes in all.
        let plain = 10_000 + 2 * 4096;
        let d = Snapshot::decode_limited(DEFAULT_FORMAT, &compact, plain).unwrap();
        assert_eq!(d, s, "a budget of exactly the plain size");
        // One byte less: the last chunk no longer fits, though every field alone would.
        assert_eq!(
            Snapshot::decode_limited(DEFAULT_FORMAT, &compact, plain - 1),
            Err(Error::Malformed("a deflated field"))
        );
        // The plain encoding is bounded by its own size: no budget applies.
        assert_eq!(
            Snapshot::decode_limited(DEFAULT_FORMAT, &s.encode(), 0).unwrap(),
            s
        );
    }

    #[test]
    fn decoding_is_strict() {
        let s = sample();
        let bytes = s.encode();
        for n in 0..bytes.len() {
            assert!(
                Snapshot::decode(&bytes[..n]).is_err(),
                "prefix of {n} bytes"
            );
        }
        // A flipped byte inside a chunk's contents: corrupt.
        let mut bad = bytes.clone();
        let at = bad.len() - 10;
        bad[at] ^= 1;
        assert!(Snapshot::decode(&bad).is_err());
        let mut out_of_order = Snapshot::new("x", 1);
        out_of_order.add_page(2, b"a").unwrap();
        assert_eq!(out_of_order.add_page(2, b"b"), Err(Error::PageOrder));
        assert_eq!(Snapshot::decode(b"\xa0"), Err(Error::Format));
    }
}
