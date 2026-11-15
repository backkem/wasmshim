//! The engine's global state is per thread natively (`src/state.rs`): engines on two threads do
//! not share it.

use wasmshim_aarch64_engine::dispatch;

#[test]
fn counters_are_per_thread() {
    let before = dispatch::fills();
    std::thread::spawn(|| {
        assert_eq!(
            dispatch::fills(),
            0,
            "a new thread starts from its own zero"
        );
        for _ in 0..3 {
            // No dispatch is running on this thread: the fill is counted and refused.
            assert_eq!(dispatch::fill(0x1000, false, 8), 0);
        }
        assert_eq!(dispatch::fills(), 3);
    })
    .join()
    .unwrap();
    assert_eq!(
        dispatch::fills(),
        before,
        "another thread's fills are not ours"
    );
}

#[test]
fn a_step_outside_a_dispatch_is_refused() {
    // `step` is what translated code calls back into; with no dispatch running on this thread
    // there is no CPU to step, on whatever thread a dispatch may be running.
    let worker = std::thread::spawn(|| dispatch::step(0x40_0000));
    assert_eq!(worker.join().unwrap(), dispatch::STATUS_FAULT);
    assert_eq!(dispatch::step(0x40_0000), dispatch::STATUS_FAULT);
}
