//! Recipes: `<recipes>/<name>.toml` → `<out>/<name>.tar`.
//!
//! A recipe lists pinned artifacts (`[[artifact]]`: URL, SHA-256, kind), merged in order (later
//! ones replace earlier paths), then symlinks, directories, source trees and small text files
//! declared inline. `include` names recipes whose entries come first. Archives are decoded in
//! memory (gzip, xz, zstd, tar, `ar`) and re-packed deterministically.

use std::io::Read as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::tar::{Member, TarWriter, read_all};
use crate::{Extension, Members, Packer};

/// A recipe with its includes flattened; `ext` holds the extension's own keys.
#[derive(Debug)]
pub struct Recipe<F> {
    pub description: String,
    pub artifacts: Vec<Artifact>,
    /// Symlinks added after the artifacts (e.g. Debian's merged-/usr `bin -> usr/bin`).
    pub symlinks: Vec<Symlink>,
    /// Small text files a package's maintainer scripts or the installer would have written
    /// (`/etc/hosts`, `/etc/nsswitch.conf`), declared inline.
    pub files: Vec<TextFile>,
    /// Directories an installer script would have created (`install -d /run/nginx`).
    pub dirs: Vec<Dir>,
    /// Source trees next to the recipes (e.g. a demo project), copied to `path`.
    pub trees: Vec<Tree>,
    /// The extension's keys.
    pub ext: F,
}

/// The packer's own top-level keys; every other key belongs to the extension.
const RECIPE_KEYS: &[&str] = &[
    "description",
    "include",
    "artifact",
    "symlink",
    "file",
    "dir",
    "tree",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecipe {
    description: String,
    /// Recipes whose entries come first (by name, in order).
    #[serde(default)]
    include: Vec<String>,
    #[serde(default, rename = "artifact")]
    artifacts: Vec<toml::Table>,
    #[serde(default, rename = "symlink")]
    symlinks: Vec<Symlink>,
    #[serde(default, rename = "file")]
    files: Vec<TextFile>,
    #[serde(default, rename = "dir")]
    dirs: Vec<Dir>,
    #[serde(default, rename = "tree")]
    trees: Vec<Tree>,
}

/// One `[[file]]`: a text file, mode 0644.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextFile {
    pub path: String,
    pub content: String,
}

/// One `[[dir]]`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dir {
    pub path: String,
    /// Octal permission bits as a string (`"755"`, the default).
    #[serde(default)]
    pub mode: Option<String>,
}

/// One `[[tree]]`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tree {
    /// Where the tree goes in the image.
    pub path: String,
    /// The directory, relative to the recipes directory. Text only: CRLF line ends become LF,
    /// so a checkout with `core.autocrlf` builds the same image.
    pub source: String,
}

/// One `[[symlink]]`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Symlink {
    pub path: String,
    pub target: String,
}

/// One `[[artifact]]`.
#[derive(Debug)]
pub struct Artifact {
    /// Where an upstream artifact comes from (https only); may be empty for a custom kind.
    pub url: String,
    pub sha256: String,
    pub kind: Kind,
    /// Leading path components to drop (a tarball's top-level directory).
    pub strip: usize,
    /// Where the (stripped) tree goes in the image, e.g. `usr/local`.
    pub prefix: String,
    /// Paths (after `strip`, before `prefix`) to leave out: a prefix match per entry.
    pub exclude: Vec<String>,
    /// For `kind = "file"`: where the single downloaded file goes in the image.
    pub path: Option<String>,
    /// For `kind = "tar-zst"`: the byte offset of the zstd frame inside the artifact (a tarball
    /// embedded in a binary; 0 for a plain `.tar.zst`).
    pub offset: usize,
    /// Keys the packer does not know: allowed only for a custom kind, whose extension reads
    /// them.
    pub extra: toml::Table,
}

/// The packer's own artifact keys.
const ARTIFACT_KEYS: &[&str] = &[
    "url", "sha256", "kind", "strip", "prefix", "exclude", "path", "offset",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArtifact {
    #[serde(default)]
    url: String,
    sha256: String,
    kind: String,
    #[serde(default)]
    strip: usize,
    #[serde(default)]
    prefix: String,
    #[serde(default)]
    exclude: Vec<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    offset: usize,
}

/// Moves the entries of `table` whose keys are not in `known` into a table of their own.
fn split_off_unknown(table: &mut toml::Table, known: &[&str]) -> toml::Table {
    let mut rest = toml::Table::new();
    let unknown: Vec<String> = table
        .keys()
        .filter(|k| !known.contains(&k.as_str()))
        .cloned()
        .collect();
    for key in unknown {
        if let Some(v) = table.remove(&key) {
            rest.insert(key, v);
        }
    }
    rest
}

impl Artifact {
    /// Parses one `[[artifact]]` table.
    pub fn parse(mut table: toml::Table) -> Result<Self> {
        let extra = split_off_unknown(&mut table, ARTIFACT_KEYS);
        let raw: RawArtifact = table.try_into()?;
        let kind = Kind::from_name(&raw.kind);
        if !matches!(kind, Kind::Custom(_)) && !extra.is_empty() {
            let keys: Vec<&str> = extra.keys().map(String::as_str).collect();
            bail!("{}: unknown artifact key(s) {}", raw.url, keys.join(", "));
        }
        Ok(Self {
            url: raw.url,
            sha256: raw.sha256,
            kind,
            strip: raw.strip,
            prefix: raw.prefix,
            exclude: raw.exclude,
            path: raw.path,
            offset: raw.offset,
            extra,
        })
    }

    /// Where a member lands, or `None` when stripped or excluded.
    pub fn place(&self, path: &str) -> Option<String> {
        let mut parts = path.split('/').filter(|p| !p.is_empty());
        for _ in 0..self.strip {
            parts.next()?;
        }
        let rel = parts.collect::<Vec<_>>().join("/");
        if rel.is_empty() || self.exclude.iter().any(|e| rel.starts_with(e.as_str())) {
            return None;
        }
        Some(if self.prefix.is_empty() {
            rel
        } else {
            format!("{}/{rel}", self.prefix.trim_end_matches('/'))
        })
    }
}

/// An artifact's `kind`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// `tar-gz`: a gzip-compressed tar (Alpine minirootfs, nodejs.org tarballs).
    TarGz,
    /// `apk`: an Alpine package: concatenated gzip streams of one tar; metadata entries are
    /// dropped.
    Apk,
    /// `deb`: a Debian package: an `ar` archive whose `data.tar.xz` (or `.gz`) holds the files.
    Deb,
    /// `file`: one file, installed (mode 0755) at the artifact's `path`: a release binary.
    File,
    /// `tar-zst`: a zstd-compressed tar starting at the artifact's `offset` (a payload embedded
    /// in a binary, which its launcher would otherwise extract on every first start).
    TarZst,
    /// Any other name: an [`Extension`] builds it.
    Custom(String),
}

impl Kind {
    /// The kind a recipe's `kind = "..."` names.
    pub fn from_name(name: &str) -> Self {
        match name {
            "tar-gz" => Self::TarGz,
            "apk" => Self::Apk,
            "deb" => Self::Deb,
            "file" => Self::File,
            "tar-zst" => Self::TarZst,
            other => Self::Custom(other.to_owned()),
        }
    }
}

/// Decodes one artifact of the packer's own kinds into tar members.
fn members(a: &Artifact, data: &[u8]) -> Result<Vec<(String, Member)>> {
    let kind = &a.kind;
    if *kind == Kind::File {
        let path = a.path.clone().context("kind = \"file\" needs a `path`")?;
        return Ok(vec![(
            path,
            Member::File {
                mode: 0o755,
                data: data.to_vec(),
            },
        )]);
    }
    let mut tar = Vec::new();
    if *kind == Kind::TarZst {
        let frame = data
            .get(a.offset..)
            .context("tar-zst: offset past the end of the artifact")?;
        if !frame.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
            bail!("tar-zst: no zstd frame at offset {}", a.offset);
        }
        ruzstd::decoding::StreamingDecoder::new(frame)
            .map_err(|e| anyhow::anyhow!("unzstd: {e}"))?
            .read_to_end(&mut tar)
            .context("unzstd")?;
        return read_all(&tar);
    }
    if *kind == Kind::Deb {
        let (name, member) = deb_data(data)?;
        if name.ends_with(".xz") {
            lzma_rs::xz_decompress(&mut std::io::Cursor::new(member), &mut tar)
                .map_err(|e| anyhow::anyhow!("unxz: {e:?}"))?;
        } else if name.ends_with(".gz") {
            flate2::read::MultiGzDecoder::new(member)
                .read_to_end(&mut tar)
                .context("gunzip")?;
        } else {
            bail!("unsupported .deb data member {name}");
        }
        return read_all(&tar);
    }
    flate2::read::MultiGzDecoder::new(data)
        .read_to_end(&mut tar)
        .context("gunzip")?;
    let mut entries = read_all(&tar)?;
    if *kind == Kind::Apk {
        // .PKGINFO, .SIGN.*, .pre-install, ... are package metadata, not files.
        entries.retain(|(p, _)| !p.starts_with('.'));
    }
    Ok(entries)
}

/// Loads `<recipes>/<name>.toml` with its includes flattened (includes first).
pub(crate) fn load<E: Extension>(
    packer: &Packer,
    name: &str,
    depth: usize,
) -> Result<Recipe<E::Fields>> {
    if depth > 8 {
        bail!("recipe include depth exceeded at {name}");
    }
    let recipe_path: &Path = &packer.recipes.join(format!("{name}.toml"));
    let text = std::fs::read_to_string(recipe_path)
        .with_context(|| format!("no image recipe {}", recipe_path.display()))?;
    let parse = || -> Result<(RawRecipe, E::Fields, Vec<Artifact>)> {
        let mut table: toml::Table = toml::from_str(&text)?;
        let ext = split_off_unknown(&mut table, RECIPE_KEYS);
        let mut recipe: RawRecipe = table.try_into()?;
        let artifacts = std::mem::take(&mut recipe.artifacts)
            .into_iter()
            .map(Artifact::parse)
            .collect::<Result<_>>()?;
        Ok((recipe, ext.try_into()?, artifacts))
    };
    let (mut recipe, own_ext, mut own_artifacts) =
        parse().with_context(|| format!("parsing {}", recipe_path.display()))?;
    let mut artifacts = Vec::new();
    let mut symlinks = Vec::new();
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    let mut trees = Vec::new();
    let mut includes = Vec::new();
    for inc in std::mem::take(&mut recipe.include) {
        let base = load::<E>(packer, &inc, depth + 1)?;
        trees.extend(base.trees);
        artifacts.extend(base.artifacts);
        symlinks.extend(base.symlinks);
        files.extend(base.files);
        dirs.extend(base.dirs);
        includes.push(base.ext);
    }
    artifacts.append(&mut own_artifacts);
    symlinks.append(&mut recipe.symlinks);
    files.append(&mut recipe.files);
    dirs.append(&mut recipe.dirs);
    trees.append(&mut recipe.trees);
    Ok(Recipe {
        description: recipe.description,
        artifacts,
        symlinks,
        files,
        dirs,
        trees,
        ext: E::flatten(own_ext, includes),
    })
}

/// Builds `<recipes>/<name>.toml` (see [`Extension`] for where its hooks run).
pub(crate) fn build<E: Extension>(packer: &Packer, name: &str, ext: &E) -> Result<PathBuf> {
    let recipe = load::<E>(packer, name, 0)?;
    println!("image {name}: {}", recipe.description);
    // Later artifacts replace earlier paths; the output is sorted by path, so parents come first.
    let mut merged = Members::new();
    for a in &recipe.artifacts {
        let entries = match &a.kind {
            Kind::Custom(kind) => ext
                .artifact(packer, a)
                .with_context(|| format!("reading {}", a.url))?
                .with_context(|| format!("{}: unknown artifact kind `{kind}`", a.url))?,
            _ => {
                let data = packer.fetch(&a.url, &a.sha256)?;
                members(a, &data).with_context(|| format!("reading {}", a.url))?
            }
        };
        println!("  {} members from {}", entries.len(), a.url);
        for (path, m) in entries {
            let Some(path) = a.place(&path) else {
                continue;
            };
            // Hard links name another member: it moves with the tree.
            let m = match m {
                Member::HardLink { target } => match a.place(&target) {
                    Some(target) => Member::HardLink { target },
                    None => continue,
                },
                m => m,
            };
            merged.insert(path, m);
        }
    }
    ext.after_artifacts(packer, &recipe, &mut merged)?;
    let mut w = TarWriter::new();
    for s in &recipe.symlinks {
        merged.insert(
            s.path.clone(),
            Member::Symlink {
                target: s.target.clone(),
            },
        );
    }
    for d in &recipe.dirs {
        let mode = match &d.mode {
            Some(m) => u32::from_str_radix(m, 8)
                .with_context(|| format!("dir {}: mode {m} is not octal", d.path))?,
            None => 0o755,
        };
        merged.insert(d.path.clone(), Member::Dir { mode });
    }
    for t in &recipe.trees {
        let dir = packer.recipes.join(&t.source);
        let mut members = Vec::new();
        tree_members(&dir, t.path.trim_end_matches('/'), &mut members)
            .with_context(|| format!("tree {}", dir.display()))?;
        println!(
            "  tree: {} file(s) from {} into /{}",
            members.len(),
            t.source,
            t.path
        );
        merged.extend(members);
    }
    for f in &recipe.files {
        merged.insert(
            f.path.clone(),
            Member::File {
                mode: 0o644,
                data: f.content.clone().into_bytes(),
            },
        );
    }
    ext.after_files(packer, &recipe, &mut merged)?;
    for (path, m) in &merged {
        w.add(path, m)?;
    }
    packer.publish(name, &w.finish())
}

/// Finds the `data.tar.*` member of a `.deb` (`ar` format: an 8-byte magic, then 60-byte
/// headers each followed by the member, padded to an even size).
fn deb_data(deb: &[u8]) -> Result<(String, &[u8])> {
    if !deb.starts_with(b"!<arch>\n") {
        bail!("not an ar archive");
    }
    let mut pos = 8;
    while pos + 60 <= deb.len() {
        let header = &deb[pos..pos + 60];
        let name = String::from_utf8_lossy(&header[..16])
            .trim_end()
            .trim_end_matches('/')
            .to_owned();
        let size: usize = String::from_utf8_lossy(&header[48..58])
            .trim()
            .parse()
            .context("bad ar member size")?;
        let start = pos + 60;
        let body = deb
            .get(start..start + size)
            .context("truncated ar member")?;
        if name.starts_with("data.tar") {
            return Ok((name, body));
        }
        pos = start + size + (size & 1);
    }
    bail!("no data.tar member")
}

/// The directories and files under `dir` (sorted), as members under `prefix`. Build outputs
/// (`node_modules`, `dist`) are skipped; text line ends are normalised to LF.
fn tree_members(dir: &Path, prefix: &str, out: &mut Vec<(String, Member)>) -> Result<()> {
    out.push((prefix.to_owned(), Member::Dir { mode: 0o755 }));
    let mut children: Vec<_> = std::fs::read_dir(dir)?.collect::<std::io::Result<_>>()?;
    children.sort_by_key(|e| e.file_name());
    for e in children {
        let name = e.file_name().to_string_lossy().into_owned();
        let path = format!("{prefix}/{name}");
        if e.file_type()?.is_dir() {
            if name != "node_modules" && name != "dist" {
                tree_members(&e.path(), &path, out)?;
            }
        } else {
            let data = std::fs::read(e.path())?;
            let mut text = Vec::with_capacity(data.len());
            for (i, &b) in data.iter().enumerate() {
                if !(b == b'\r' && data.get(i + 1) == Some(&b'\n')) {
                    text.push(b);
                }
            }
            out.push((
                path,
                Member::File {
                    mode: 0o644,
                    data: text,
                },
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoExtension;

    /// A fresh directory of recipes for one test.
    fn dir(test: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("linux-image-pack-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("recipes/tree/sub")).unwrap();
        d
    }

    fn packer(d: &Path) -> Packer {
        Packer::new(d.join("recipes"), d.join("cache"), d.join("out"))
    }

    fn write(d: &Path, name: &str, text: &str) {
        std::fs::write(d.join("recipes").join(name), text).unwrap();
    }

    fn file(data: &str) -> Member {
        Member::File {
            mode: 0o644,
            data: data.as_bytes().to_vec(),
        }
    }

    #[test]
    fn builds_a_recipe_without_downloads() {
        let d = dir("plain");
        write(
            &d,
            "base.toml",
            "description = \"base\"\n\
             [[file]]\npath = \"etc/hosts\"\ncontent = \"base\\n\"\n\
             [[dir]]\npath = \"tmp\"\nmode = \"1777\"\n",
        );
        write(
            &d,
            "top.toml",
            "description = \"top\"\ninclude = [\"base\"]\n\
             [[symlink]]\npath = \"bin\"\ntarget = \"usr/bin\"\n\
             [[file]]\npath = \"etc/hosts\"\ncontent = \"top\\n\"\n\
             [[tree]]\npath = \"srv/app/\"\nsource = \"tree\"\n",
        );
        std::fs::write(d.join("recipes/tree/sub/a.txt"), "one\r\ntwo\n").unwrap();
        let path = packer(&d).build("top", &NoExtension).unwrap();
        let tar = std::fs::read(&path).unwrap();
        let members = read_all(&tar).unwrap();
        assert_eq!(
            members,
            vec![
                (
                    "bin".into(),
                    Member::Symlink {
                        target: "usr/bin".into()
                    }
                ),
                ("etc/hosts".into(), file("top\n")),
                ("srv/app".into(), Member::Dir { mode: 0o755 }),
                ("srv/app/sub".into(), Member::Dir { mode: 0o755 }),
                ("srv/app/sub/a.txt".into(), file("one\ntwo\n")),
                ("tmp".into(), Member::Dir { mode: 0o1777 }),
            ]
        );
        let sha = std::fs::read_to_string(d.join("out/top.sha256")).unwrap();
        assert_eq!(sha, format!("{}\n", crate::sha256_hex(&tar)));
        // The same recipe, the same bytes.
        packer(&d).build("top", &NoExtension).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), tar);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn unknown_keys_and_kinds_are_errors() {
        let d = dir("unknown");
        write(&d, "key.toml", "description = \"x\"\nprograms = [\"a\"]\n");
        assert!(packer(&d).load::<NoExtension>("key").is_err());
        write(
            &d,
            "akey.toml",
            "description = \"x\"\n[[artifact]]\nurl = \"https://example.org/a.tar.gz\"\n\
             sha256 = \"00\"\nkind = \"tar-gz\"\nsource = \"a\"\n",
        );
        let err = packer(&d).load::<NoExtension>("akey").unwrap_err();
        assert!(format!("{err:#}").contains("unknown artifact key(s) source"));
        write(
            &d,
            "kind.toml",
            "description = \"x\"\n[[artifact]]\nsha256 = \"00\"\nkind = \"mystery\"\n",
        );
        let err = packer(&d).build("kind", &NoExtension).unwrap_err();
        assert!(format!("{err:#}").contains("unknown artifact kind `mystery`"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// An extension with a custom kind (`inline`: its `text` as one file) and a key (`last`:
    /// paths written last).
    struct Inline;

    #[derive(Debug, Default, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct InlineFields {
        #[serde(default)]
        last: Vec<String>,
    }

    impl Extension for Inline {
        type Fields = InlineFields;

        fn flatten(mut own: InlineFields, includes: Vec<InlineFields>) -> InlineFields {
            let mut last: Vec<String> = includes.into_iter().flat_map(|i| i.last).collect();
            last.append(&mut own.last);
            InlineFields { last }
        }

        fn artifact(
            &self,
            _packer: &Packer,
            a: &Artifact,
        ) -> Result<Option<Vec<(String, Member)>>> {
            if a.kind != Kind::Custom("inline".into()) {
                return Ok(None);
            }
            let text = a.extra["text"].as_str().context("text")?;
            Ok(Some(vec![("x/inline".into(), file(text))]))
        }

        fn after_artifacts(
            &self,
            _packer: &Packer,
            _recipe: &Recipe<InlineFields>,
            members: &mut Members,
        ) -> Result<()> {
            members.insert("etc/a".into(), file("after-artifacts"));
            members.insert("etc/b".into(), file("after-artifacts"));
            Ok(())
        }

        fn after_files(
            &self,
            _packer: &Packer,
            recipe: &Recipe<InlineFields>,
            members: &mut Members,
        ) -> Result<()> {
            for p in &recipe.ext.last {
                members.insert(p.clone(), file("last"));
            }
            Ok(())
        }
    }

    #[test]
    fn extension_hooks_run_in_order() {
        let d = dir("ext");
        write(&d, "base.toml", "description = \"b\"\nlast = [\"etc/c\"]\n");
        write(
            &d,
            "top.toml",
            "description = \"t\"\ninclude = [\"base\"]\nlast = [\"etc/b\"]\n\
             [[artifact]]\nsha256 = \"00\"\nkind = \"inline\"\ntext = \"hi\"\nprefix = \"opt\"\n\
             [[file]]\npath = \"etc/a\"\ncontent = \"file\"\n\
             [[file]]\npath = \"etc/c\"\ncontent = \"file\"\n",
        );
        let recipe = packer(&d).load::<Inline>("top").unwrap();
        assert_eq!(recipe.ext.last, ["etc/c", "etc/b"]);
        let path = packer(&d).build("top", &Inline).unwrap();
        let members = read_all(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(
            members,
            vec![
                ("etc/a".into(), file("file")),
                ("etc/b".into(), file("last")),
                ("etc/c".into(), file("last")),
                ("opt/x/inline".into(), file("hi")),
            ]
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
