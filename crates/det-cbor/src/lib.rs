//! Deterministic CBOR, a small subset of it: unsigned and negative integers, byte and text
//! strings, arrays, maps, booleans and null, in the deterministic encoding of RFC 8949 §4.2
//! (shortest heads, definite lengths, map keys sorted by their encoded bytes). [`decode`]
//! accepts exactly that encoding and nothing else (no indefinite lengths, tags or floats), so
//! every value has one byte form.
//!
//! `no_std`, no dependencies, so any crate may share it.
//! [`build`] writes values straight to bytes (what an engine replying to a read needs), and
//! [`write_head`] and [`write_map`] are the primitives a fuller CBOR profile can build on.

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// A decoded value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Uint(u64),
    /// The negative integer `-1 - n`.
    Nint(u64),
    Bytes(Vec<u8>),
    Text(String),
    Array(Vec<Value>),
    /// In encoded-key order once encoded (the encoder sorts).
    Map(Vec<(Value, Value)>),
    Bool(bool),
    Null,
}

impl Value {
    pub fn text(s: impl Into<String>) -> Value {
        Value::Text(s.into())
    }

    pub fn as_uint(&self) -> Option<u64> {
        match self {
            Value::Uint(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// The value of text key `key` in a map.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(entries) => entries
                .iter()
                .find(|(k, _)| k.as_text() == Some(key))
                .map(|(_, v)| v),
            _ => None,
        }
    }

    /// The deterministic encoding.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write(&mut out);
        out
    }

    /// Appends the deterministic encoding to `out`.
    pub fn write(&self, out: &mut Vec<u8>) {
        match self {
            Value::Uint(n) => write_head(out, 0, *n),
            Value::Nint(n) => write_head(out, 1, *n),
            Value::Bytes(b) => {
                write_head(out, 2, b.len() as u64);
                out.extend_from_slice(b);
            }
            Value::Text(s) => {
                write_head(out, 3, s.len() as u64);
                out.extend_from_slice(s.as_bytes());
            }
            Value::Array(items) => {
                write_head(out, 4, items.len() as u64);
                for i in items {
                    i.write(out);
                }
            }
            Value::Map(entries) => write_map(
                out,
                entries
                    .iter()
                    .map(|(k, v)| (k.encode(), v.encode()))
                    .collect(),
            ),
            Value::Bool(false) => out.push(0xf4),
            Value::Bool(true) => out.push(0xf5),
            Value::Null => out.push(0xf6),
        }
    }
}

/// Appends a head: major type `major` with argument `n`, in its shortest form.
pub fn write_head(out: &mut Vec<u8>, major: u8, n: u64) {
    let m = major << 5;
    if n < 24 {
        out.push(m | n as u8);
    } else if n <= 0xff {
        out.extend_from_slice(&[m | 24, n as u8]);
    } else if n <= 0xffff {
        out.push(m | 25);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else if n <= 0xffff_ffff {
        out.push(m | 26);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    } else {
        out.push(m | 27);
        out.extend_from_slice(&n.to_be_bytes());
    }
}

/// Appends a map of already-encoded `(key, value)` pairs: sorted by their encoded keys (RFC 8949
/// §4.2.1), and of equal keys only the last (a map's keys are unique).
pub fn write_map(out: &mut Vec<u8>, mut pairs: Vec<(Vec<u8>, Vec<u8>)>) {
    // A stable sort keeps equal keys in their given order; the last of each run wins.
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut kept: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(pairs.len());
    for p in pairs {
        match kept.last_mut() {
            Some(last) if last.0 == p.0 => *last = p,
            _ => kept.push(p),
        }
    }
    write_head(out, 5, kept.len() as u64);
    for (k, v) in kept {
        out.extend_from_slice(&k);
        out.extend_from_slice(&v);
    }
}

/// Why bytes are not a value of this subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    Truncated,
    /// A head that is not the shortest, an indefinite length, a tag, a float or a reserved code.
    NotDeterministic,
    BadUtf8,
    TooDeep,
    TrailingBytes,
    /// Map keys out of order or repeated.
    MapOrder,
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Truncated => "the bytes end inside a value",
            Self::NotDeterministic => "not deterministic CBOR (a long head, an indefinite length, a tag, a float or a reserved code)",
            Self::BadUtf8 => "a text string is not UTF-8",
            Self::TooDeep => "nested too deep",
            Self::TrailingBytes => "bytes follow the value",
            Self::MapOrder => "map keys out of order or repeated",
        })
    }
}

impl core::error::Error for DecodeError {}

/// Decodes exactly one value that fills `bytes`.
pub fn decode(bytes: &[u8]) -> Result<Value, DecodeError> {
    let mut at = 0;
    let mut budget = bytes.len();
    let v = item(bytes, &mut at, 0, &mut budget)?;
    if at != bytes.len() {
        return Err(DecodeError::TrailingBytes);
    }
    Ok(v)
}

const MAX_DEPTH: u32 = 32;

/// How many of an array's or map's `n` items to reserve room for up front. A length is
/// checked only against the bytes left, and every nesting level sees the same bytes, so the
/// levels draw on one `budget` (the input's length in items, each item being at least a byte):
/// a truncated input never reserves room for more items than it has bytes. Beyond the budget a
/// container grows as its items actually decode.
fn reserve(n: u64, budget: &mut usize) -> usize {
    let k = usize::try_from(n).unwrap_or(usize::MAX).min(*budget);
    *budget -= k;
    k
}

fn take<'a>(b: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], DecodeError> {
    let end = at.checked_add(n).ok_or(DecodeError::Truncated)?;
    let s = b.get(*at..end).ok_or(DecodeError::Truncated)?;
    *at = end;
    Ok(s)
}

fn read_head(b: &[u8], at: &mut usize) -> Result<(u8, u64, u8), DecodeError> {
    let first = take(b, at, 1)?[0];
    let (major, info) = (first >> 5, first & 0x1f);
    let n = match info {
        0..=23 => u64::from(info),
        24 => {
            let v = u64::from(take(b, at, 1)?[0]);
            if v < 24 {
                return Err(DecodeError::NotDeterministic);
            }
            v
        }
        25 => {
            let s = take(b, at, 2)?;
            let v = u64::from(u16::from_be_bytes([s[0], s[1]]));
            if v <= 0xff {
                return Err(DecodeError::NotDeterministic);
            }
            v
        }
        26 => {
            let s = take(b, at, 4)?;
            let v = u64::from(u32::from_be_bytes([s[0], s[1], s[2], s[3]]));
            if v <= 0xffff {
                return Err(DecodeError::NotDeterministic);
            }
            v
        }
        27 => {
            let s = take(b, at, 8)?;
            let mut a = [0u8; 8];
            a.copy_from_slice(s);
            let v = u64::from_be_bytes(a);
            if v <= 0xffff_ffff {
                return Err(DecodeError::NotDeterministic);
            }
            v
        }
        _ => return Err(DecodeError::NotDeterministic),
    };
    Ok((major, n, info))
}

fn item(b: &[u8], at: &mut usize, depth: u32, budget: &mut usize) -> Result<Value, DecodeError> {
    if depth > MAX_DEPTH {
        return Err(DecodeError::TooDeep);
    }
    let (major, n, info) = read_head(b, at)?;
    let len = |n: u64| usize::try_from(n).map_err(|_| DecodeError::Truncated);
    Ok(match major {
        0 => Value::Uint(n),
        1 => Value::Nint(n),
        2 => Value::Bytes(take(b, at, len(n)?)?.to_vec()),
        3 => {
            let s = take(b, at, len(n)?)?;
            Value::Text(String::from(
                core::str::from_utf8(s).map_err(|_| DecodeError::BadUtf8)?,
            ))
        }
        4 => {
            // Each item is at least one byte: a length beyond the input is truncated.
            if n > (b.len() - *at) as u64 {
                return Err(DecodeError::Truncated);
            }
            let mut items = Vec::with_capacity(reserve(n, budget));
            for _ in 0..n {
                items.push(item(b, at, depth + 1, budget)?);
            }
            Value::Array(items)
        }
        5 => {
            if n > (b.len() - *at) as u64 {
                return Err(DecodeError::Truncated);
            }
            let mut entries = Vec::with_capacity(reserve(n, budget));
            let mut last: Option<&[u8]> = None;
            for _ in 0..n {
                let k0 = *at;
                let k = item(b, at, depth + 1, budget)?;
                let key_bytes = &b[k0..*at];
                if last.is_some_and(|l| l >= key_bytes) {
                    return Err(DecodeError::MapOrder);
                }
                last = Some(key_bytes);
                let v = item(b, at, depth + 1, budget)?;
                entries.push((k, v));
            }
            Value::Map(entries)
        }
        7 => match info {
            20 => Value::Bool(false),
            21 => Value::Bool(true),
            22 => Value::Null,
            _ => return Err(DecodeError::NotDeterministic),
        },
        _ => return Err(DecodeError::NotDeterministic),
    })
}

/// `n` as a CBOR unsigned integer.
pub fn uint(n: u64) -> Vec<u8> {
    Value::Uint(n).encode()
}

/// `null`.
pub fn null() -> Vec<u8> {
    Value::Null.encode()
}

/// Values written straight to bytes, for engines that reply to reads without building a
/// [`Value`]: the same bytes `Value::encode` gives.
pub mod build {
    use alloc::string::String;
    use alloc::vec::Vec;

    use super::{Value, decode, write_head, write_map};

    pub fn uint(n: u64) -> Vec<u8> {
        let mut out = Vec::new();
        write_head(&mut out, 0, n);
        out
    }

    pub fn bytes(b: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(b.len() + 9);
        write_head(&mut out, 2, b.len() as u64);
        out.extend_from_slice(b);
        out
    }

    pub fn text(s: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(s.len() + 9);
        write_head(&mut out, 3, s.len() as u64);
        out.extend_from_slice(s.as_bytes());
        out
    }

    pub fn boolean(b: bool) -> Vec<u8> {
        alloc::vec![if b { 0xf5 } else { 0xf4 }]
    }

    pub fn null() -> Vec<u8> {
        alloc::vec![0xf6]
    }

    /// An array of already-encoded items.
    pub fn array(items: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        write_head(&mut out, 4, items.len() as u64);
        for i in items {
            out.extend_from_slice(i);
        }
        out
    }

    /// A map of text keys to already-encoded values (keys sorted by their encoding).
    pub fn map(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        write_map(
            &mut out,
            entries.iter().map(|(k, v)| (text(k), v.clone())).collect(),
        );
        out
    }

    /// One scalar (an integer, a string, a boolean or null), or `None` (a container, or not
    /// this encoding): what an engine's write accepts.
    pub fn scalar(b: &[u8]) -> Option<Value> {
        match decode(b).ok()? {
            Value::Array(_) | Value::Map(_) => None,
            v => Some(v),
        }
    }

    /// A map of text keys to scalars (a probe's condition), in encoded order; the empty input
    /// is the empty map. Anything else is `None`.
    pub fn text_map(b: &[u8]) -> Option<Vec<(String, Value)>> {
        if b.is_empty() {
            return Some(Vec::new());
        }
        let Value::Map(entries) = decode(b).ok()? else {
            return None;
        };
        entries
            .into_iter()
            .map(|(k, v)| match (k, v) {
                (_, Value::Array(_) | Value::Map(_)) => None,
                (Value::Text(k), v) => Some((k, v)),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn round_trips_in_the_deterministic_form() {
        let v = Value::Map(vec![
            (Value::text("to"), Value::Uint(1)),
            (
                Value::text("a"),
                Value::Array(vec![Value::Null, Value::Bool(true)]),
            ),
            (Value::text("b"), Value::Bytes(vec![1, 2, 3])),
            (Value::text("n"), Value::Nint(4)),
            (Value::text("big"), Value::Uint(1 << 40)),
        ]);
        let bytes = v.encode();
        let back = decode(&bytes).unwrap();
        assert_eq!(back.get("to"), Some(&Value::Uint(1)));
        assert_eq!(back.get("big"), Some(&Value::Uint(1 << 40)));
        assert_eq!(back.encode(), bytes);
        let Value::Map(entries) = back else { panic!() };
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_text().unwrap()).collect();
        assert_eq!(keys, ["a", "b", "n", "to", "big"]);
    }

    #[test]
    fn known_encodings() {
        assert_eq!(uint(0), [0x00]);
        assert_eq!(uint(23), [0x17]);
        assert_eq!(uint(24), [0x18, 24]);
        assert_eq!(uint(500), [0x19, 0x01, 0xf4]);
        assert_eq!(null(), [0xf6]);
        assert_eq!(Value::text("c4").encode(), [0x62, b'c', b'4']);
        assert_eq!(Value::Nint(0).encode(), [0x20]);
        for n in [
            0u64,
            23,
            24,
            255,
            256,
            65_535,
            65_536,
            u32::MAX as u64 + 1,
            u64::MAX,
        ] {
            assert_eq!(build::uint(n), uint(n));
        }
        assert_eq!(build::text("a+start"), Value::text("a+start").encode());
        assert_eq!(build::bytes(&[1, 2]), Value::Bytes(vec![1, 2]).encode());
        assert_eq!(build::boolean(true), Value::Bool(true).encode());
        // Keys sorted by encoding: "b" before "aa".
        assert_eq!(
            build::map(&[("aa", build::uint(1)), ("b", build::uint(2))])[1..3],
            [0x61, b'b']
        );
    }

    #[test]
    fn equal_keys_keep_the_last() {
        let v = Value::Map(vec![
            (Value::text("k"), Value::Uint(1)),
            (Value::text("k"), Value::Uint(2)),
        ]);
        assert_eq!(v.encode(), [0xa1, 0x61, b'k', 0x02]);
    }

    #[test]
    fn narrow_decoders() {
        assert_eq!(build::scalar(&build::uint(7)), Some(Value::Uint(7)));
        assert_eq!(build::scalar(&build::array(&[])), None);
        assert_eq!(build::text_map(&[]), Some(Vec::new()));
        let m = build::map(&[("when", build::text("exit")), ("tid", build::uint(3))]);
        assert_eq!(
            build::text_map(&m),
            Some(vec![
                (String::from("tid"), Value::Uint(3)),
                (String::from("when"), Value::text("exit")),
            ])
        );
    }

    #[test]
    fn rejects_non_deterministic_and_malformed_input() {
        assert_eq!(decode(&[0x18, 5]), Err(DecodeError::NotDeterministic));
        assert_eq!(decode(&[0x9f]), Err(DecodeError::NotDeterministic));
        assert_eq!(decode(&[0x01, 0x02]), Err(DecodeError::TrailingBytes));
        assert_eq!(decode(&[0x62, b'a']), Err(DecodeError::Truncated));
        assert_eq!(
            decode(&[0x9b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]),
            Err(DecodeError::Truncated)
        );
        assert_eq!(
            decode(&[0xa2, 0x61, b'b', 0, 0x61, b'a', 0]),
            Err(DecodeError::MapOrder)
        );
        assert_eq!(decode(&[0xc1, 0x00]), Err(DecodeError::NotDeterministic));
        assert_eq!(decode(&[0x62, 0xff, 0xfe]), Err(DecodeError::BadUtf8));
        assert_eq!(decode(&[]), Err(DecodeError::Truncated));
    }

    #[test]
    fn truncated_nested_lengths_reserve_no_more_than_the_input() {
        // 32 nested arrays each claiming 1000 items (no more than the bytes left after its
        // head), the innermost complete, the rest truncated.
        let mut bytes = Vec::new();
        for _ in 0..32 {
            bytes.extend_from_slice(&[0x99, 0x03, 0xe8]);
        }
        bytes.extend_from_slice(&[0; 1000]);
        assert_eq!(decode(&bytes), Err(DecodeError::Truncated));
        // Before the shared budget each level reserved its own 1000 items; now all levels
        // together reserve at most one item per input byte.
        let (mut at, mut budget) = (0, bytes.len());
        assert_eq!(
            item(&bytes, &mut at, 0, &mut budget),
            Err(DecodeError::Truncated)
        );
        assert_eq!(
            budget, 0,
            "1000 for the outermost, the other 96 bytes for the next"
        );
        let mut budget = 10;
        assert_eq!((reserve(4, &mut budget), budget), (4, 6));
        assert_eq!((reserve(u64::MAX, &mut budget), budget), (6, 0));
        assert_eq!((reserve(3, &mut budget), budget), (0, 0));
        // Valid containers decode as before once the budget is spent.
        let v = Value::Array(vec![Value::Array(vec![Value::Uint(1); 50]); 3]);
        assert_eq!(decode(&v.encode()), Ok(v));
    }
}
