//! The engine as a core module (wasm32 only; natively this crate is empty). The glue: the allocator, the panic handler, the global engine and the exports
//! of the engine core ABI ([`wasmshim_cpu_remote::abi`]) and of translated code's imports
//! (`wasmshim_aarch64_translate::abi`).

#![cfg_attr(target_arch = "wasm32", no_std)]
#![allow(unsafe_code)]

#[cfg(target_arch = "wasm32")]
mod module {

    use wasmshim_cpu_remote::abi::{self, Io, STATUS_COMPILE, STATUS_DONE};

    use wasmshim_aarch64_engine::remote::{Engine, Outcome};

    #[global_allocator]
    static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo) -> ! {
        core::arch::wasm32::unreachable()
    }

    static mut IO: Io = Io {
        service: 0,
        nargs: 0,
        input_len: 0,
        nvalues: 0,
        output_ptr: 0,
        output_len: 0,
        req_ptr: 0,
        req_len: 0,
        req_funcs: 0,
        req_slot: 0,
        args: [0; abi::MAX_WORDS],
        values: [0; abi::MAX_WORDS],
    };

    static mut ENGINE: Option<Engine> = None;

    /// The engine. One host thread; the dispatcher never holds this borrow across a translated
    /// call that may re-enter through `jit_step` with another one (see `dispatch.rs`).
    fn engine() -> &'static mut Engine {
        // SAFETY: single-threaded core module; see above.
        let p = &raw mut ENGINE;
        unsafe { (*p).get_or_insert_with(Engine::new) }
    }

    fn io() -> &'static mut Io {
        // SAFETY: single-threaded; only the exports below touch it, one at a time.
        let p = &raw mut IO;
        unsafe { &mut *p }
    }

    fn publish(outcome: Outcome) -> i32 {
        let e = engine();
        let io = io();
        match outcome {
            Outcome::Done => {
                let n = e.values.len().min(abi::MAX_WORDS);
                io.values[..n].copy_from_slice(&e.values[..n]);
                io.nvalues = n as u32;
                io.output_ptr = e.output.as_ptr() as u32;
                io.output_len = e.output.len() as u32;
                STATUS_DONE
            }
            Outcome::Compile => {
                io.req_ptr = e.req.as_ptr() as u32;
                io.req_len = e.req.len() as u32;
                io.req_funcs = e.req_funcs;
                io.req_slot = abi::SLOT_DECLINED;
                STATUS_COMPILE
            }
        }
    }

    /// `() -> i32`: the address of the call block.
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_io() -> i32 {
        (&raw mut IO) as i32
    }

    /// `(len) -> i32`: an input buffer of at least `len` bytes.
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_input(len: i32) -> i32 {
        let e = engine();
        e.input.clear();
        e.input.resize(len.max(0) as usize, 0);
        e.input.as_mut_ptr() as i32
    }

    /// `() -> i32`: performs (or, after a compile request, continues) the call in the block.
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_call() -> i32 {
        let e = engine();
        let io = io();
        let outcome = if io.req_len != 0 {
            // The host answered a compile request.
            let slot = io.req_slot;
            io.req_len = 0;
            io.req_ptr = 0;
            e.resume(slot)
        } else {
            let n = (io.nargs as usize).min(abi::MAX_WORDS);
            let args = io.args;
            let input_len = (io.input_len as usize).min(e.input.len());
            e.begin(io.service, &args[..n], input_len)
        };
        publish(outcome)
    }

    /// `(pc: i64) -> i32`: translated code's slow path (see `wasmshim_aarch64_translate::abi::IMPORT_STEP`).
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_step(pc: i64) -> i32 {
        wasmshim_aarch64_engine::dispatch::step(pc as u64)
    }
}

/// `memcmp` for `core` (no libc in this module).
///
/// # Safety
///
/// `a` and `b` must be valid for `n` bytes.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memcmp(
    a: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    n: usize,
) -> i32 {
    let (a, b) = (a.cast::<u8>(), b.cast::<u8>());
    for i in 0..n {
        // SAFETY: per the contract.
        let (x, y) = unsafe { (*a.add(i), *b.add(i)) };
        if x != y {
            return i32::from(x) - i32::from(y);
        }
    }
    0
}
