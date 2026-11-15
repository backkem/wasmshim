//! For wasm32 artifacts (the component, guest tests): link the ahead-of-time translations
//! (`$LINUX_AOT_OBJ` is the path of an object file the translator produced). When it is unset,
//! or this crate is built as an embedder's dependency, an empty object is linked and the node
//! carries no ahead-of-time translations.
fn main() {
    println!("cargo:rerun-if-env-changed=LINUX_AOT_OBJ");
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("wasm32") {
        return;
    }
    // Only the plain node (feature `export`) is a final component. Built without it, as an
    // embedder's dependency, this crate's own cdylib is never used: it links the empty object,
    // and the embedder's component links the translations itself.
    let own = std::env::var_os("CARGO_FEATURE_EXPORT").is_some();
    let obj = match std::env::var("LINUX_AOT_OBJ").ok().filter(|_| own) {
        Some(path) => {
            println!("cargo:rerun-if-changed={path}");
            std::path::PathBuf::from(path)
        }
        None => {
            let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
            let path = out.join("linux-aot-empty.o");
            let (bytes, _) = wasmshim_aarch64_translate::aot::object(&[]);
            std::fs::write(&path, bytes).expect("write the empty AOT object");
            path
        }
    };
    println!("cargo:rustc-link-arg={}", obj.display());
    // The run-time tier (feature `jit`): its host instantiates modules against
    // this core module, so the module exports its function table (growable: the host appends
    // to it) and the step and fill functions translated code imports.
    if std::env::var_os("CARGO_FEATURE_JIT").is_some() {
        for arg in [
            "--export-table",
            "--growable-table",
            "--export=linux_jit_step",
            "--export=linux_jit_fill",
        ] {
            println!("cargo:rustc-link-arg={arg}");
        }
    }
}
