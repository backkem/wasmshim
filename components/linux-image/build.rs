//! Copies the image named by `LINUX_IMAGE` into OUT_DIR so `include_bytes!` can embed it.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=LINUX_IMAGE");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let dest = out.join("rootfs.tar");
    match std::env::var_os("LINUX_IMAGE") {
        Some(src) => {
            println!("cargo:rerun-if-changed={}", PathBuf::from(&src).display());
            std::fs::copy(&src, &dest)
                .unwrap_or_else(|e| panic!("cannot copy LINUX_IMAGE {src:?}: {e}"));
        }
        // Without an image the component serves an empty archive (two zero blocks).
        None => std::fs::write(&dest, [0u8; 1024]).expect("writing OUT_DIR/rootfs.tar"),
    }
}
