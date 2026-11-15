//! The ahead-of-time translations linked into this module.
//!
//! Every Wasm artifact that contains the engine links an object made by
//! [`wasmshim_aarch64_translate::aot::object`] (an empty one when nothing was translated), which defines
//! the two accessor functions below. Natively there are no translations.

#![allow(unsafe_code)]

use wasmshim_aarch64_translate::aot::Directory;

#[cfg(target_arch = "wasm32")]
unsafe extern "C" {
    fn linux_aot_directory() -> *const u8;
    fn linux_aot_directory_len() -> usize;
}

/// The linked directory, if it holds any translation of this translator version.
pub fn directory() -> Option<Directory<'static>> {
    #[cfg(target_arch = "wasm32")]
    {
        // SAFETY: the linked object defines both functions; the directory is a read-only data
        // segment of this module that lives as long as the instance.
        let bytes = unsafe {
            core::slice::from_raw_parts(linux_aot_directory(), linux_aot_directory_len())
        };
        wasmshim_aarch64_translate::aot::directory(bytes).filter(|d| d.pages() > 0)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        None
    }
}
