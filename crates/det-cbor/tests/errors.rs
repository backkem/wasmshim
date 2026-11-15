//! A decode failure is an error a caller can print and wrap.

use wasmshim_cbor::{DecodeError, decode};

#[test]
fn decode_errors_are_errors() {
    // 0x18 0x01: the integer 1 with a one-byte head it does not need.
    let e = decode(&[0x18, 0x01]).unwrap_err();
    assert_eq!(e, DecodeError::NotDeterministic);
    let boxed: Box<dyn std::error::Error> = Box::new(e);
    assert!(boxed.to_string().starts_with("not deterministic CBOR"));
    assert_eq!(
        decode(&[0x01, 0x02]).unwrap_err().to_string(),
        "bytes follow the value"
    );
}
