//! For wasm32 artifacts of this crate (the core module): link the ahead-of-time translations
//! and emit a plain core module (no component wrapper, no entry point) that exports its
//! growable function table. `$LINUX_AOT_OBJ` is the path of an object file the translator
//! produced; when it is unset an empty object is linked and the module carries no
//! ahead-of-time translations.
fn main() {
    println!("cargo:rerun-if-env-changed=LINUX_AOT_OBJ");
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("wasm32") {
        return;
    }
    let obj = match std::env::var("LINUX_AOT_OBJ") {
        Ok(path) => {
            println!("cargo:rerun-if-changed={path}");
            std::path::PathBuf::from(path)
        }
        Err(_) => {
            let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
            let path = out.join("linux-aot-empty.o");
            let (bytes, _) = wasmshim_aarch64_translate::aot::object(&[]);
            std::fs::write(&path, bytes).expect("write the empty AOT object");
            path
        }
    };
    println!("cargo:rustc-link-arg={}", obj.display());
    {
        for a in [
            "--skip-wit-component",
            "--export-table",
            "--growable-table",
            "--no-entry",
        ] {
            println!("cargo:rustc-link-arg={a}");
        }
    }
}
