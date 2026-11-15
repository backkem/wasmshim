//! Root filesystem images: uncompressed tar archives with an index in front, built from TOML
//! recipes of pinned upstream artifacts.
//!
//! A [`Packer`] names three directories: where recipes live (`<name>.toml`, plus the source
//! trees they copy), the artifact cache (`<cache>/<sha256>/<file>`), and where images go
//! (`<out>/<name>.tar` and `<out>/<name>.sha256`). Every artifact is named by URL and SHA-256,
//! downloaded once over https with the system `curl`, verified before use, and then only read:
//! nothing from an artifact is executed or extracted onto the host filesystem. The output is
//! deterministic (sorted members, fixed mtime and owners, see [`tar`]), so the same recipe gives
//! the same bytes on every host.
//!
//! Recipe keys and artifact kinds the packer does not know come from an [`Extension`]: its own
//! recipe fields, its own `kind`s, and members it adds at fixed points of the build.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

pub mod recipe;
pub mod tar;

pub use recipe::{Artifact, Kind, Recipe};
use tar::Member;

/// Hex SHA-256.
pub fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Runs a command, failing with its name if it does not succeed.
fn run(cmd: &mut Command) -> Result<()> {
    let shown = format!("{cmd:?}");
    let status = cmd.status().with_context(|| format!("starting {shown}"))?;
    if !status.success() {
        bail!("{shown} failed with {status}");
    }
    Ok(())
}

/// The members of an image as they are merged: later inserts replace earlier paths, and the
/// archive is written in path order (so parents come first).
pub type Members = BTreeMap<String, Member>;

/// Recipe keys and artifact kinds on top of the packer's own.
///
/// A build calls the hooks in this order: [`Extension::artifact`] for each artifact of a kind
/// the packer does not know (in recipe order, between the packer's own artifacts), then
/// [`Extension::after_artifacts`], then the packer adds the recipe's symlinks, directories,
/// trees and files, then [`Extension::after_files`]. Each hook may replace what came before it.
pub trait Extension {
    /// The extension's top-level recipe keys (every key the packer does not know is handed to
    /// it). Use `#[serde(deny_unknown_fields)]` so a misspelt key is an error.
    type Fields: DeserializeOwned;

    /// Flattens includes: `own` is the recipe's fields, `includes` its included recipes' (each
    /// already flattened), in include order.
    fn flatten(own: Self::Fields, includes: Vec<Self::Fields>) -> Self::Fields;

    /// The members of an artifact of custom kind `a.kind`, before `strip`, `exclude` and
    /// `prefix` are applied; `None` when the kind is unknown to this extension too.
    fn artifact(&self, packer: &Packer, a: &Artifact) -> Result<Option<Vec<(String, Member)>>> {
        let _ = (packer, a);
        Ok(None)
    }

    /// Adds members after the artifacts.
    fn after_artifacts(
        &self,
        packer: &Packer,
        recipe: &Recipe<Self::Fields>,
        members: &mut Members,
    ) -> Result<()> {
        let _ = (packer, recipe, members);
        Ok(())
    }

    /// Adds members last, after the recipe's symlinks, directories, trees and files.
    fn after_files(
        &self,
        packer: &Packer,
        recipe: &Recipe<Self::Fields>,
        members: &mut Members,
    ) -> Result<()> {
        let _ = (packer, recipe, members);
        Ok(())
    }
}

/// The packer's recipes alone: no extra keys, no custom kinds.
pub struct NoExtension;

/// [`NoExtension`]'s fields: none, so any unknown recipe key is an error.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoFields {}

impl Extension for NoExtension {
    type Fields = NoFields;

    fn flatten(own: NoFields, _includes: Vec<NoFields>) -> NoFields {
        own
    }
}

/// Where a packer reads recipes and artifacts and writes images.
#[derive(Clone, Debug)]
pub struct Packer {
    /// Recipes (`<name>.toml`) and the source trees their `[[tree]]` entries name.
    pub recipes: PathBuf,
    /// The download cache: one fresh directory per artifact, `<cache>/<sha256>/`.
    pub cache: PathBuf,
    /// The output directory for `<name>.tar` and `<name>.sha256`.
    pub out: PathBuf,
}

impl Packer {
    pub fn new(
        recipes: impl Into<PathBuf>,
        cache: impl Into<PathBuf>,
        out: impl Into<PathBuf>,
    ) -> Self {
        Self {
            recipes: recipes.into(),
            cache: cache.into(),
            out: out.into(),
        }
    }

    /// Downloads `url` (https only) into the cache if it is not there yet, and returns its
    /// bytes once they match `sha256`.
    pub fn fetch(&self, url: &str, sha256: &str) -> Result<Vec<u8>> {
        if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!("{url}: sha256 must be 64 hex digits");
        }
        if !url.starts_with("https://") {
            bail!("{url}: only https URLs are allowed");
        }
        let file = url
            .rsplit('/')
            .next()
            .filter(|f| !f.is_empty())
            .context("URL has no file name")?;
        let dir = self.cache.join(sha256);
        let path = dir.join(file);
        if !path.is_file() {
            std::fs::create_dir_all(&dir)?;
            let partial = dir.join(format!("{file}.partial"));
            println!("fetching {url}");
            run(Command::new("curl")
                .args([
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--proto",
                    "=https",
                    "-o",
                ])
                .arg(&partial)
                .arg(url))
            .context("downloading with curl")?;
            std::fs::rename(&partial, &path)?;
        }
        let data = std::fs::read(&path)?;
        let got = sha256_hex(&data);
        if got != sha256 {
            bail!(
                "{url}: SHA-256 mismatch: expected {sha256}, got {got} (cached at {})",
                path.display()
            );
        }
        Ok(data)
    }

    /// Writes `tar`, with the image index put in front, as `<out>/<name>.tar` and its SHA-256
    /// as `<out>/<name>.sha256`; returns the tar's path.
    pub fn publish(&self, name: &str, tar: &[u8]) -> Result<PathBuf> {
        publish(&self.out, name, tar)
    }

    /// Builds recipe `name` with `ext`; returns the image's path.
    pub fn build<E: Extension>(&self, name: &str, ext: &E) -> Result<PathBuf> {
        recipe::build(self, name, ext)
    }

    /// Loads recipe `name` with its includes flattened (includes first).
    pub fn load<E: Extension>(&self, name: &str) -> Result<Recipe<E::Fields>> {
        recipe::load::<E>(self, name, 0)
    }
}

/// Writes `tar`, with the image index put in front, as `<dir>/<name>.tar` and its SHA-256 as
/// `<dir>/<name>.sha256`; returns the tar's path.
pub fn publish(dir: &Path, name: &str, tar: &[u8]) -> Result<PathBuf> {
    // The index goes in front: mounting reads it instead of every header.
    let tar = &wasmshim_linux::vfs::index::with_index(tar)
        .map_err(|e| anyhow::anyhow!("indexing image {name}: {e}"))?;
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{name}.tar"));
    std::fs::write(&path, tar).with_context(|| format!("writing {}", path.display()))?;
    let hash = sha256_hex(tar);
    std::fs::write(dir.join(format!("{name}.sha256")), format!("{hash}\n"))?;
    println!(
        "image {name}: {} ({} bytes, sha256 {hash})",
        path.display(),
        tar.len()
    );
    Ok(path)
}
