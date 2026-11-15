//! Serves one compiled-in tar image as the `linux:host/blob` named `"rootfs"`, also as words
//! (`linux:host/blob-words`, for a node composed with it).

#![deny(unsafe_code)]

/// The embedded image (unused by the unit tests, which test `read` directly).
#[cfg_attr(test, allow(dead_code))]
static ROOTFS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rootfs.tar"));

/// The only blob name this provider knows.
pub const ROOTFS_NAME: &str = "rootfs";

/// `blob.read` semantics on a byte slice (pure, testable).
pub fn read(data: &[u8], offset: u64, len: u32) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(usize::try_from(len).ok()?)?;
    data.get(start..end)
}

/// `blob-words.read-words` packing: `bytes` as little-endian words, the last padded with zeros.
pub fn words(bytes: &[u8]) -> Vec<u64> {
    let (whole, tail) = bytes.as_chunks::<8>();
    let mut out: Vec<u64> = whole.iter().map(|w| u64::from_le_bytes(*w)).collect();
    if !tail.is_empty() {
        let mut last = [0u8; 8];
        last[..tail.len()].copy_from_slice(tail);
        out.push(u64::from_le_bytes(last));
    }
    out
}

#[cfg(not(test))]
mod component {
    #![allow(unsafe_code)] // generated bindings

    wit_bindgen::generate!({ world: "image-provider", path: "../../wit/linux-host" });

    use exports::linux::host::blob::{Error, Guest};

    struct Provider;

    impl Guest for Provider {
        fn size(name: String) -> Result<u64, Error> {
            if name == super::ROOTFS_NAME {
                Ok(super::ROOTFS.len() as u64)
            } else {
                Err(Error::NotFound)
            }
        }

        fn read(name: String, offset: u64, len: u32) -> Result<Vec<u8>, Error> {
            if name != super::ROOTFS_NAME {
                return Err(Error::NotFound);
            }
            super::read(super::ROOTFS, offset, len)
                .map(<[u8]>::to_vec)
                .ok_or(Error::OutOfRange)
        }
    }

    impl exports::linux::host::blob_words::Guest for Provider {
        fn read_words(name: String, offset: u64, len: u32) -> Result<Vec<u64>, Error> {
            if name != super::ROOTFS_NAME {
                return Err(Error::NotFound);
            }
            super::read(super::ROOTFS, offset, len)
                .map(super::words)
                .ok_or(Error::OutOfRange)
        }
    }

    export!(Provider);
}

#[cfg(test)]
mod tests {
    #[test]
    fn ranges() {
        let data = [1u8, 2, 3, 4];
        assert_eq!(super::read(&data, 1, 2), Some(&[2u8, 3][..]));
        assert_eq!(super::read(&data, 3, 2), None);
        assert_eq!(super::read(&data, u64::MAX, 1), None);
        assert_eq!(super::read(&data, 4, 0), Some(&[][..]));
    }

    #[test]
    fn words() {
        assert_eq!(super::words(&[]), Vec::<u64>::new());
        assert_eq!(
            super::words(&[1, 2, 3, 4, 5, 6, 7, 8]),
            [0x0807_0605_0403_0201]
        );
        assert_eq!(
            super::words(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            [0x0807_0605_0403_0201, 0x0a09]
        );
    }
}
