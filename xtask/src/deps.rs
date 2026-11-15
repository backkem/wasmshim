//! The dependency firewall between the layers of this workspace.
//!
//! * execution (`wasmshim-cpu*`, `wasmshim-aarch64*`) never depends on the Linux side
//!   (`wasmshim-linux*`, `wasmshim-wasmtime`, `wasmshim-image-pack`, `wasmshim-run`) or on
//!   `wasmshim-kernel`;
//! * `wasmshim-kernel` (the substrate) depends on `wasmshim-cpu` only, never on a CPU or a
//!   personality;
//! * `wasmshim-linux` (the personality) never names an execution backend (`wasmshim-aarch64*`).
//!
//! Direct dependencies of every kind count (normal, build, dev, per target).

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::{Context as _, Result};

/// An execution crate: the CPU contract and the AArch64 front end, engines and translator.
fn executes(package: &str) -> bool {
    package.starts_with("wasmshim-cpu") || is_backend(package)
}

/// A crate that implements the CPU contract for one instruction set.
fn is_backend(package: &str) -> bool {
    package.starts_with("wasmshim-aarch64")
}

/// A crate of the Linux side: the personality, its ABI and ELF loader, the node and its hosts.
fn is_linux(package: &str) -> bool {
    package.starts_with("wasmshim-linux")
        || matches!(
            package,
            "wasmshim-wasmtime" | "wasmshim-image-pack" | "wasmshim-run"
        )
}

/// The rule that forbids `package` to depend on `dependency`, if one does.
pub fn rule_for(package: &str, dependency: &str) -> Option<&'static str> {
    if executes(package) && is_linux(dependency) {
        return Some(
            "execution (wasmshim-cpu*, wasmshim-aarch64*) never depends on the Linux side",
        );
    }
    if executes(package) && dependency == "wasmshim-kernel" {
        return Some("execution sits below the substrate: no wasmshim-kernel");
    }
    if package == "wasmshim-kernel" && (is_linux(dependency) || is_backend(dependency)) {
        return Some("wasmshim-kernel depends on wasmshim-cpu only");
    }
    if package == "wasmshim-linux" && is_backend(dependency) {
        return Some("a personality never names an execution backend (wasmshim-aarch64*)");
    }
    None
}

/// The names in every dependency table of a manifest (`[dependencies]`, `[dev-dependencies]`,
/// `[build-dependencies]` and their `[target.<cfg>.*]` forms).
pub fn dependencies(manifest: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_deps = false;
    for line in manifest.lines().map(str::trim) {
        if let Some(table) = line.strip_prefix('[') {
            in_deps = table.trim_end_matches(']').ends_with("dependencies");
        } else if in_deps && !line.starts_with('#') {
            // `name = ...`, `name.workspace = true` or `name = { ... }`.
            let name: String = line
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
                .collect();
            if !name.is_empty() {
                names.insert(name);
            }
        }
    }
    names
}

/// Every violated rule in the workspace at `root`, as `package -> dependency: rule`.
pub fn check(root: &Path) -> Result<Vec<String>> {
    Ok(violations(&members(root)?))
}

/// The violated rules among `members` (`(package name, manifest text)`), and nothing about
/// dependencies that are not members.
pub fn violations(members: &[(String, String)]) -> Vec<String> {
    let ours: BTreeSet<&str> = members.iter().map(|(name, _)| name.as_str()).collect();
    let mut violations = Vec::new();
    for (package, manifest) in members {
        for dependency in dependencies(manifest) {
            if !ours.contains(dependency.as_str()) {
                continue;
            }
            if let Some(rule) = rule_for(package, &dependency) {
                violations.push(format!("{package} -> {dependency}: {rule}"));
            }
        }
    }
    violations
}

/// The `name` of a manifest's `[package]` table.
pub fn package_name(manifest: &str) -> Option<String> {
    let mut in_package = false;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            in_package = line == "[package]";
        } else if in_package
            && let Some((key, value)) = line.split_once('=')
            && key.trim() == "name"
        {
            return Some(value.trim().trim_matches('"').to_owned());
        }
    }
    None
}

/// `(package name, manifest text)` of every crate under `crates/` and `components/`. The name is
/// the manifest's, not the directory's: the directories are named after what they hold
/// (`crates/guest-kernel` is `wasmshim-kernel`), and dependencies are written by package.
fn members(root: &Path) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for group in ["crates", "components"] {
        let dir = root.join(group);
        for entry in fs::read_dir(&dir).with_context(|| dir.display().to_string())? {
            let path = entry?.path();
            let manifest = path.join("Cargo.toml");
            if !manifest.is_file() {
                continue;
            }
            let text = fs::read_to_string(&manifest)?;
            let name = package_name(&text)
                .with_context(|| format!("{}: no [package] name", manifest.display()))?;
            out.push((name, text));
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layers_point_one_way() {
        assert!(rule_for("wasmshim-aarch64", "wasmshim-linux-abi").is_some());
        assert!(rule_for("wasmshim-cpu", "wasmshim-kernel").is_some());
        assert!(rule_for("wasmshim-kernel", "wasmshim-aarch64").is_some());
        assert!(rule_for("wasmshim-kernel", "wasmshim-linux").is_some());
        assert!(rule_for("wasmshim-linux", "wasmshim-aarch64-engine").is_some());

        assert!(rule_for("wasmshim-kernel", "wasmshim-cpu").is_none());
        assert!(rule_for("wasmshim-linux", "wasmshim-kernel").is_none());
        assert!(rule_for("wasmshim-aarch64-engine", "wasmshim-cpu-remote").is_none());
        assert!(rule_for("wasmshim-linux-node", "wasmshim-aarch64").is_none());
        assert!(rule_for("wasmshim-aarch64-engine", "wasmshim-run").is_some());
        assert!(rule_for("wasmshim-cpu-remote", "wasmshim-kernel").is_some());
    }

    #[test]
    fn dependency_tables_of_every_kind_are_read() {
        let manifest = r#"
            [package]
            name = "x"
            [dependencies]
            wasmshim-cpu.workspace = true
            libm = "0.2"
            [target.'cfg(target_arch = "wasm32")'.dependencies]
            dlmalloc = { version = "0.2" }
            [dev-dependencies]
            # a comment
            wasmshim-linux-abi = { workspace = true }
            [features]
            profile = []
        "#;
        let names: Vec<String> = dependencies(manifest).into_iter().collect();
        assert_eq!(
            names,
            ["dlmalloc", "libm", "wasmshim-cpu", "wasmshim-linux-abi"]
        );
    }

    #[test]
    fn this_workspace_keeps_the_rules() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        assert_eq!(check(root).unwrap(), Vec::<String>::new());
    }

    /// The check must see this workspace's packages and the edges between them, or it passes
    /// whatever the manifests say.
    #[test]
    fn the_check_sees_the_workspace_s_packages_and_edges() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let members = members(root).unwrap();
        let names: BTreeSet<&str> = members.iter().map(|(n, _)| n.as_str()).collect();
        for package in [
            "wasmshim-cpu",
            "wasmshim-kernel",
            "wasmshim-linux",
            "wasmshim-aarch64",
        ] {
            assert!(names.contains(package), "{package} is not among {names:?}");
        }
        let edges = members
            .iter()
            .flat_map(|(_, manifest)| dependencies(manifest))
            .filter(|d| names.contains(d.as_str()))
            .count();
        assert!(
            edges >= 20,
            "only {edges} dependencies between members were read"
        );
    }

    #[test]
    fn a_planted_violation_is_reported() {
        let member = |name: &str, deps: &str| {
            (
                name.to_owned(),
                format!("[package]\nname = \"{name}\"\n[dependencies]\n{deps}\n"),
            )
        };
        let members = [
            member("wasmshim-cpu", ""),
            member("wasmshim-kernel", "wasmshim-cpu.workspace = true"),
            // The interpreter reaching into the personality: forbidden.
            member("wasmshim-aarch64", "wasmshim-linux = { workspace = true }"),
            member(
                "wasmshim-linux",
                "wasmshim-kernel.workspace = true\nlibm = \"0.2\"",
            ),
        ];
        let found = violations(&members);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].starts_with("wasmshim-aarch64 -> wasmshim-linux: "),
            "{found:?}"
        );
    }

    #[test]
    fn a_package_s_name_is_read_from_its_manifest() {
        let manifest =
            "[package]\nname = \"wasmshim-kernel\"\nversion = \"0.0.0\"\n[lib]\nname = \"x\"\n";
        assert_eq!(package_name(manifest).as_deref(), Some("wasmshim-kernel"));
        assert_eq!(package_name("[workspace]\nmembers = []\n"), None);
    }
}
