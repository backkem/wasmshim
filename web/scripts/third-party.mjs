// The licences of the code the page ships: the Rust crates compiled into its WebAssembly, and
// the npm packages whose code runs in the tab. MIT and Apache-2.0 both ask that their text
// travels with binary copies, so the notices page carries every licence file these packages
// publish, not a pointer to a lock file.
//
// The crates come from `cargo metadata` (the dependency closure of the packages the page is
// built from, for the page's targets); their licence files are read from the crate sources
// cargo has on disk. The npm packages are read from node_modules.

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';

/** File names that carry a licence, a copyright statement or a notice. */
const LICENCE_FILE = /^(LICEN[CS]E|COPYING|COPYRIGHT|NOTICE|UNLICENSE|PATENTS)([-._].*)?$/i;

/** The licence files in `dir`: `[{ file, text }]`, by name. */
export function licenceFiles(dir) {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => LICENCE_FILE.test(f) && statSync(join(dir, f)).isFile())
    .sort()
    .map((file) => ({ file, text: readFileSync(join(dir, file), 'utf8') }));
}

/**
 * The third-party crates in the dependency closure of the workspace packages `roots` at
 * `workspace`, for each of `targets`: `[{ name, version, license, repository, files }]`, by
 * name. Only normal dependencies count: build scripts and tests are not in the artifact.
 */
export function rustCrates(workspace, roots, targets) {
  const found = new Map();
  for (const target of targets) {
    const out = execFileSync('cargo', ['metadata', '--format-version', '1', '--locked', '--filter-platform', target], {
      cwd: workspace,
      encoding: 'utf8',
      maxBuffer: 512 * 1024 * 1024,
    });
    const meta = JSON.parse(out);
    const packages = new Map(meta.packages.map((p) => [p.id, p]));
    const nodes = new Map(meta.resolve.nodes.map((n) => [n.id, n]));
    // This workspace's members are its own code; a path dependency that is not a member (vendored
    // third-party source) is third-party like a registry crate.
    const own = new Set(meta.workspace_members);
    const todo = meta.packages.filter((p) => own.has(p.id) && roots.includes(p.name)).map((p) => p.id);
    const missing = roots.filter((r) => !meta.packages.some((p) => own.has(p.id) && p.name === r));
    if (missing.length > 0) throw new Error(`third-party: no workspace package named ${missing.join(', ')}`);
    // A procedural macro runs in the compiler: neither it nor what it depends on is in the
    // artifact.
    const isMacro = (id) => packages.get(id)?.targets.some((t) => t.kind.includes('proc-macro'));
    const seen = new Set();
    while (todo.length > 0) {
      const id = todo.pop();
      if (seen.has(id) || isMacro(id)) continue;
      seen.add(id);
      for (const dep of nodes.get(id)?.deps ?? []) {
        if (dep.dep_kinds.some((k) => k.kind === null)) todo.push(dep.pkg);
      }
    }
    for (const id of seen) {
      const p = packages.get(id);
      if (own.has(id)) continue;
      found.set(`${p.name} ${p.version}`, {
        name: p.name,
        version: p.version,
        license: p.license ?? '(no licence field)',
        repository: p.repository ?? '',
        files: licenceFiles(dirname(p.manifest_path)),
      });
    }
  }
  return [...found.values()].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : a.version < b.version ? -1 : 1));
}

/**
 * The npm packages `names` as installed under `web/node_modules`:
 * `[{ name, version, license, repository, files }]`. A package that publishes no licence file
 * takes the one in `fallback[name]` (a file under `web/licenses`), and fails without one.
 */
export function npmPackages(web, names, fallback = {}) {
  return names.map((name) => {
    const dir = join(web, 'node_modules', ...name.split('/'));
    const manifest = JSON.parse(readFileSync(join(dir, 'package.json'), 'utf8'));
    let files = licenceFiles(dir);
    if (files.length === 0) {
      const file = fallback[name];
      if (!file) throw new Error(`third-party: ${name} publishes no licence file and has no fallback`);
      files = [{ file, text: readFileSync(join(web, 'licenses', file), 'utf8') }];
    }
    const repository = typeof manifest.repository === 'string' ? manifest.repository : (manifest.repository?.url ?? '');
    return { name, version: manifest.version, license: manifest.license ?? '(no licence field)', repository, files };
  });
}

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

/** One package as a notices-page section: a line of facts and its licence files, folded. */
export function packageHtml(p) {
  const where = /^https?:\/\//.test(p.repository) ? `<a href="${esc(p.repository)}">${esc(p.repository)}</a>` : '';
  const files =
    p.files.length === 0
      ? '<p>The published package carries no licence file; the expression above is its licence.</p>'
      : p.files.map((f) => `<details>\n<summary>${esc(f.file)}</summary>\n<pre>${esc(f.text)}</pre>\n</details>`).join('\n');
  return `<section>\n<h3>${esc(p.name)} ${esc(p.version)} <span class="licence">${esc(p.license)}</span></h3>\n<p>${where}</p>\n${files}\n</section>`;
}
