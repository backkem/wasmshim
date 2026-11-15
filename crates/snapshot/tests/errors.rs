//! A snapshot that cannot be read says why.

use wasmshim_snapshot::Error;

#[test]
fn errors_name_the_chunk_and_the_kind() {
    let mut digest = [0u8; 32];
    digest[..3].copy_from_slice(&[0xab, 0xcd, 0x01]);
    assert_eq!(
        Error::MissingChunk(digest).to_string(),
        "chunk abcd01000000… is missing"
    );
    let e = Error::Kind {
        found: "gb".into(),
        want: "linux",
    };
    assert_eq!(e.to_string(), "a `gb` snapshot, not `linux`");
    let boxed: Box<dyn std::error::Error> = Box::new(Error::Malformed("pages"));
    assert_eq!(boxed.to_string(), "malformed snapshot: pages");
}
