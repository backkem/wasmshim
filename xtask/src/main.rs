//! `cargo xtask` for this workspace (run from its root).
//!
//! ```text
//! cargo xtask build            build the plain node (components/linux-node, wasm32-wasip2,
//!                              release) and `wasm-tools validate` it
//! cargo xtask image <recipe>   fetch, verify and assemble images/<recipe>.tar from
//!                              images/<recipe>.toml into target/images/ (downloads cached in
//!                              .cache/artifacts)
//! cargo xtask aot <recipe> [PATH...]
//!                              translate the code of the image's programs (all of them, or the
//!                              files at PATH) ahead of time and build a node with the
//!                              translations linked in: target/aot/<recipe>/ (see `aot.rs`)
//! cargo xtask check-deps       the dependency rules between the layers (see `deps.rs`)
//! cargo xtask web [--no-build] [--test]
//!                              the browser page in web/dist: `image toybox`, `build`, then
//!                              `npm ci` (when the lockfile changed) and `npm run build` in web/;
//!                              --no-build skips the Rust steps, --test also builds `wasmshim-run`
//!                              and runs `npm test` (headless Chromium, compared with wasmshim-run).
//!                              Needs Node 24 and npm, and a Chromium for --test
//! ```
//!
//! A built node runs an image with `wasmshim-run`:
//! `cargo run --release -p wasmshim-run -- --image target/images/<recipe>.tar
//! target/wasm32-wasip2/release/wasmshim_linux_node.wasm /bin/busybox echo hello`.

mod aot;
mod deps;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use anyhow::{Context as _, Result, bail};
use wasmshim_image_pack::{NoExtension, Packer};

const TARGET: &str = "wasm32-wasip2";

/// The workspace root.
fn root() -> PathBuf {
    let xtask = Path::new(env!("CARGO_MANIFEST_DIR"));
    xtask.parent().unwrap_or(xtask).to_path_buf()
}

fn image(args: &[String]) -> Result<()> {
    let [name] = args else {
        bail!("usage: cargo xtask image <recipe>");
    };
    let root = root();
    let packer = Packer::new(
        root.join("images"),
        root.join(".cache/artifacts"),
        root.join("target/images"),
    );
    packer.build(name, &NoExtension)?;
    Ok(())
}

fn check_deps(args: &[String]) -> Result<()> {
    if !args.is_empty() {
        bail!("usage: cargo xtask check-deps");
    }
    let violations = deps::check(&root())?;
    for v in &violations {
        eprintln!("check-deps: {v}");
    }
    if !violations.is_empty() {
        bail!("{} dependency rule violation(s)", violations.len());
    }
    println!("xtask check-deps: ok");
    Ok(())
}

/// Builds the node (`wasm32-wasip2`, release), with the translations of `aot` linked in when
/// given, validates it and returns its path.
fn build_node(root: &Path, aot: Option<&Path>) -> Result<PathBuf> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(root).args([
        "build",
        "--release",
        "--target",
        TARGET,
        "-p",
        "wasmshim-linux-node",
    ]);
    // The node's build script links `$LINUX_AOT_OBJ` (an empty object without it) and reruns
    // when it changes.
    match aot {
        Some(object) => cmd.env("LINUX_AOT_OBJ", object),
        None => cmd.env_remove("LINUX_AOT_OBJ"),
    };
    let status = cmd.status()?;
    if !status.success() {
        bail!("cargo build -p wasmshim-linux-node failed: {status}");
    }
    let target =
        std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from);
    let node = target
        .join(TARGET)
        .join("release")
        .join("wasmshim_linux_node.wasm");
    let out = Command::new("wasm-tools")
        .arg("validate")
        .arg(&node)
        .output()
        .context("wasm-tools (on PATH) validates the node")?;
    if !out.status.success() {
        bail!(
            "{}: invalid: {}",
            node.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(node)
}

fn build(args: &[String]) -> Result<()> {
    if !args.is_empty() {
        bail!("usage: cargo xtask build");
    }
    let node = build_node(&root(), None)?;
    println!("xtask build: {} (valid)", node.display());
    Ok(())
}

/// `cargo xtask aot <recipe> [PATH...]`: a node that runs the image's programs as translated
/// code (see `aot.rs`).
fn aot(args: &[String]) -> Result<()> {
    let Some((name, paths)) = args.split_first().filter(|(n, _)| !n.starts_with('-')) else {
        bail!("usage: cargo xtask aot <recipe> [PATH...]");
    };
    let root = root();
    let tar = root.join("target/images").join(format!("{name}.tar"));
    let image = std::fs::read(&tar)
        .with_context(|| format!("{} (run `cargo xtask image {name}` first)", tar.display()))?;
    let started = std::time::Instant::now();
    let (pages, swept) = aot::sweep(&image, paths)?;
    if pages.is_empty() {
        bail!("{}: no AArch64 code to translate", tar.display());
    }
    let (object, stats) = aot::translate(&pages);
    let dir = root.join("target/aot").join(name);
    std::fs::create_dir_all(&dir)?;
    let object_path = dir.join("aot.o");
    std::fs::write(&object_path, &object)?;
    println!(
        "xtask aot: {} files, {} code pages, {} blocks, {} instructions -> {} KiB of Wasm in {:.1}s",
        swept.files,
        swept.pages,
        stats.blocks,
        stats.insns,
        stats.code_bytes / 1024,
        started.elapsed().as_secs_f64()
    );
    let node = dir.join("wasmshim_linux_node.wasm");
    std::fs::copy(build_node(&root, Some(&object_path))?, &node)?;
    // The default build output stays the plain node.
    build_node(&root, None)?;
    println!("xtask aot: {} (valid)", node.display());
    println!(
        "run it: cargo run --release -p wasmshim-run -- --image {} {} PROGRAM [ARGS...]",
        tar.display(),
        node.display()
    );
    Ok(())
}

/// `npm` (`npm.cmd` on Windows, where Command does not search for `.cmd` files).
fn npm() -> Command {
    Command::new(if cfg!(windows) { "npm.cmd" } else { "npm" })
}

/// Runs `cmd`, failing on a non-zero status.
fn run(cmd: &mut Command) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("running {:?}", cmd.get_program()))?;
    if !status.success() {
        bail!("{:?} failed: {status}", cmd.get_program());
    }
    Ok(())
}

/// The image the page ships (`images/<WEB_IMAGE>.toml`).
const WEB_IMAGE: &str = "toybox";

/// The browser page: the image and the node, then web/package.json's scripts.
fn web(args: &[String]) -> Result<()> {
    let mut no_build = false;
    let mut test = false;
    for a in args {
        match a.as_str() {
            "--no-build" => no_build = true,
            "--test" => test = true,
            other => bail!("usage: cargo xtask web [--no-build] [--test] (unknown option {other})"),
        }
    }
    let root = root();
    let web = root.join("web");
    if npm().arg("--version").output().is_err() {
        bail!("web: needs Node 24 and npm on PATH");
    }
    if !no_build {
        image(&[WEB_IMAGE.to_owned()])?;
        build(&[])?;
        if test {
            // The test compares every run with the native runner.
            let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
            run(Command::new(cargo).current_dir(&root).args([
                "build",
                "--release",
                "-p",
                "wasmshim-run",
            ]))?;
        }
    }
    // A fresh install when there is none, or when the lockfile changed since the last one.
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let stale = match (
        modified(&web.join("node_modules").join(".package-lock.json")),
        modified(&web.join("package-lock.json")),
    ) {
        (Some(installed), Some(lock)) => lock > installed,
        _ => true,
    };
    if stale {
        run(npm().current_dir(&web).arg("ci"))?;
    }
    run(npm().current_dir(&web).args(["run", "build"]))?;
    if test {
        run(npm().current_dir(&web).arg("test"))?;
    }
    println!("xtask web: the page is in {}", web.join("dist").display());
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("build") => build(&args[1..]),
        Some("image") => image(&args[1..]),
        Some("aot") => aot(&args[1..]),
        Some("check-deps") => check_deps(&args[1..]),
        Some("web") => web(&args[1..]),
        _ => Err(anyhow::anyhow!(
            "usage: cargo xtask build | cargo xtask image <recipe> | \
             cargo xtask aot <recipe> [PATH...] | cargo xtask check-deps | \
             cargo xtask web [--no-build] [--test]"
        )),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xtask: {e:#}");
            ExitCode::FAILURE
        }
    }
}
