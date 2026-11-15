//! Where the engine keeps the state of its one machine.
//!
//! Translated code re-enters the engine through exported symbols (`linux_jit_step`,
//! `linux_jit_fill`) that carry no state pointer, so that state cannot live in a value the caller
//! holds: it is global to the engine instance. An engine instance is a Wasm instance: one
//! machine, one thread.
//!
//! * On single-threaded wasm32 (the only place translated code exists) each piece of state is
//!   one cell.
//! * Everywhere else (native builds and their tests, a wasm32 target with threads) it is one
//!   cell **per thread**, so two tests that run engines on two threads never share it.
//!
//! Either way a cell is reached by one thread only. What remains for the code that uses one is
//! the aliasing rule: never hold a reference into a cell across a call that can re-enter the
//! engine. Every use says why it does not.

/// Declares `fn $name() -> *mut $ty`: the address of this thread's cell, initialised to `$init`
/// (a constant expression). The pointer is valid for as long as the thread runs.
macro_rules! instance_state {
    ($(#[$doc:meta])* $vis:vis fn $name:ident() -> *mut $ty:ty = $init:expr;) => {
        $(#[$doc])*
        #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
        $vis fn $name() -> *mut $ty {
            struct Cell(core::cell::UnsafeCell<$ty>);
            // SAFETY: this target has no threads, so the cell is never shared between two.
            unsafe impl Sync for Cell {}
            static STATE: Cell = Cell(core::cell::UnsafeCell::new($init));
            STATE.0.get()
        }

        $(#[$doc])*
        #[cfg(not(all(target_arch = "wasm32", not(target_feature = "atomics"))))]
        $vis fn $name() -> *mut $ty {
            std::thread_local! {
                static STATE: core::cell::UnsafeCell<$ty> =
                    const { core::cell::UnsafeCell::new($init) };
            }
            STATE.with(core::cell::UnsafeCell::get)
        }
    };
}

pub(crate) use instance_state;
