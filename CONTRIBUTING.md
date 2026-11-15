# Contributing

Thanks for looking. This is an experimental project, so the bar is "small, tested and
deterministic", not "finished".

## Set up

Rust comes from `rust-toolchain.toml`. `wasm-tools` 1.261.0 validates the built node, and
`curl` downloads image artifacts:

```sh
cargo install --locked wasm-tools@1.261.0
```

## Before you open a pull request

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test
cargo xtask build         # the node builds and validates as a component
cargo xtask check-deps    # the layers still point one way
```

## What we look for

- **Determinism is a feature, flakiness is a bug.** No wall clock, no host randomness, no
  `HashMap` order in anything a guest can observe.
- **A guest never panics the host side.** System-call arguments, guest pointers and lengths,
  ELF files and images are untrusted. Bad input is an errno or a fault, never an `unwrap`.
- **Follow Linux, and say where you do not.** Cite the man page or the kernel source for a
  behaviour, and write down what is not modelled.
- **Respect the layers** (`cargo xtask check-deps`). Execution (`wasmshim-cpu*`,
  `wasmshim-aarch64*`) never depends on the Linux side or on `wasmshim-kernel`; `wasmshim-kernel`
  depends only on `wasmshim-cpu`; a personality never names a CPU.
- **Interpreter and translator agree.** A change to one comes with the other, and the
  differential test (`crates/cpu-aarch64-translate/tests/differential.rs`) covers it.
- **`unsafe` stays where it is**: the engine's dispatch files and generated bindings, each
  block with a `SAFETY` comment.

## License

Contributions are accepted under the project's dual license, MIT or Apache-2.0, without
additional terms.
