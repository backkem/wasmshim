// Third-party notices for the image the page ships (images/toybox.toml): every artifact of the
// recipe, verified against its pinned SHA-256 in the download cache (`cargo xtask image toybox`
// fills .cache/artifacts/<sha256>/<file>), with what we know about it from upstream: licence,
// version, source. The facts are in UPSTREAM below, each with where it was checked; the licence
// texts are upstream's own files, in licenses/.
//
//   node scripts/notices.mjs      prints the notices as text
//
// A recipe artifact without an UPSTREAM entry fails the build: nothing ships without a notice.

import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { licenceFiles, npmPackages, packageHtml, rustCrates } from './third-party.mjs';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = resolve(web, '..');

/**
 * Upstream facts per artifact URL. Checked 2026-11-14: the release directory and its README,
 * the release announcement's sha256sums (lists.landley.net, toybox, 2026-October, 010015 for
 * the binaries and the source tarball), the toolchain README, and every URL below (HTTP 200).
 */
const UPSTREAM = {
  'https://landley.net/toybox/downloads/binaries/0.8.15/toybox-aarch64': [
    {
      name: 'toybox',
      version: '0.8.15',
      license: '0BSD',
      licenseFile: 'toybox-0.8.15-LICENSE',
      licenseFrom: 'LICENSE in the release source archive',
      project: 'https://landley.net/toybox/',
      sources: [
        ['Release source archive (SHA-256 44e2bab43dfe2c171743297631ce558c6d9ed38b87a66fd751cef617574b375e)', 'https://landley.net/toybox/downloads/toybox-0.8.15.tar.gz'],
        ['Git, at the release tag', 'https://github.com/landley/toybox/tree/0.8.15'],
      ],
      note: 'The upstream release binary for aarch64, statically linked; used unmodified.',
    },
    {
      name: 'musl',
      version: '1.2.6',
      license: 'MIT',
      licenseFile: 'musl-1.2.6-COPYRIGHT',
      licenseFrom: 'COPYRIGHT in the musl 1.2.6 release',
      project: 'https://musl.libc.org/',
      sources: [
        ['Release source archive', 'https://musl.libc.org/releases/musl-1.2.6.tar.gz'],
        ["The toolchain sources toybox's binaries are built with, all in one archive", 'https://landley.net/toybox/downloads/binaries/toolchains/19-06-2026/musl-cross-make.tar.xz'],
      ],
      note:
        'The C library, linked statically into the toybox binary. The version is the one the README of toybox\'s ' +
        'binary toolchains names ("Currently built from: musl 1.2.6, linux 6.15.7, gcc 15.1.0, binutils 2.44", for the ' +
        'toolchains of 19-06-2026, also the latest); the toybox release itself does not say which toolchain built it. ' +
        "The compiler's runtime support (libgcc, GPL-3.0 with the GCC Runtime Library Exception, which sets no " +
        'conditions on the program) may be linked in too.',
    },
  ],
};

/** The recipe's description and artifacts (the keys this needs; the packer validates the rest). */
export function readRecipe(path) {
  const text = readFileSync(path, 'utf8');
  const recipe = { description: '', artifacts: [] };
  let current = null;
  for (const raw of text.split('\n')) {
    const line = raw.replace(/\s+#.*$/, '').trim();
    if (line === '' || line.startsWith('#')) continue;
    if (line === '[[artifact]]') {
      current = {};
      recipe.artifacts.push(current);
      continue;
    }
    if (line.startsWith('[')) {
      current = null;
      continue;
    }
    const m = /^([A-Za-z0-9_]+)\s*=\s*"(.*)"$/.exec(line);
    if (!m) continue;
    if (current) current[m[1]] = m[2];
    else if (m[1] === 'description') recipe.description = m[2];
  }
  return recipe;
}

/**
 * Every artifact of `recipe`, verified against its SHA-256, with its upstream facts:
 * `{ recipe, description, artifacts: [{ file, url, sha256, bytes, path, parts: [...] }] }`.
 */
export function collect(recipeName = 'toybox', cache = join(root, '.cache', 'artifacts')) {
  const recipe = readRecipe(join(root, 'images', `${recipeName}.toml`));
  const artifacts = recipe.artifacts.map((a) => {
    const file = basename(new URL(a.url).pathname);
    const cached = join(cache, a.sha256, file);
    if (!existsSync(cached)) throw new Error(`notices: ${cached} is missing: run \`cargo xtask image ${recipeName}\` first`);
    const bytes = readFileSync(cached);
    const got = createHash('sha256').update(bytes).digest('hex');
    if (got !== a.sha256) throw new Error(`notices: ${cached}: SHA-256 ${got}, the recipe pins ${a.sha256}`);
    const parts = UPSTREAM[a.url];
    if (!parts) throw new Error(`notices: no upstream facts for ${a.url} (scripts/notices.mjs, UPSTREAM)`);
    for (const p of parts) {
      if (!existsSync(join(web, 'licenses', p.licenseFile))) throw new Error(`notices: licenses/${p.licenseFile} is missing`);
    }
    return { file, url: a.url, sha256: a.sha256, bytes: bytes.length, path: a.path, parts };
  });
  return { recipe: recipeName, description: recipe.description, artifacts };
}

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const link = (url) => `<a href="${esc(url)}">${esc(url)}</a>`;

/** What the page ships besides the image: this repository's licence, its crates, its npm code. */
export function shipped() {
  return {
    own: licenceFiles(root),
    // The node is the one Rust artifact of the page.
    crates: rustCrates(root, ['wasmshim-linux-node'], ['wasm32-wasip2']),
    // The kit's modules run in the tab, and the node's glue is generated by jco. None of the
    // three publishes a licence file in its npm package: their repositories' are in licenses/.
    npm: npmPackages(web, ['codelib-web', '@bytecodealliance/jco', '@bytecodealliance/jco-transpile'], {
      'codelib-web': 'codelib-LICENSE',
      '@bytecodealliance/jco': 'jco-LICENSE',
      '@bytecodealliance/jco-transpile': 'jco-LICENSE',
    }),
  };
}

/**
 * The notices page. `image` is what the page serves: `{ file, sha256, bytes, size }`; `rest`
 * is [`shipped`]'s answer.
 */
export function noticesHtml(info, image, rest) {
  const part = (p) => `<section>
<h3>${esc(p.name)} ${esc(p.version)} <span class="licence">${esc(p.license)}</span></h3>
<p>${esc(p.note)}</p>
<dl>
  <dt>Project</dt><dd>${link(p.project)}</dd>
${p.sources.map(([what, url]) => `  <dt>${esc(what)}</dt><dd>${link(url)}</dd>`).join('\n')}
</dl>
<details>
<summary>Licence (${esc(p.licenseFrom)})</summary>
<pre>${esc(readFileSync(join(web, 'licenses', p.licenseFile), 'utf8'))}</pre>
</details>
</section>`;
  const sections = info.artifacts
    .map(
      (a) => `<h2><code>/${esc(a.path ?? a.file)}</code></h2>
<dl>
  <dt>Downloaded from</dt><dd>${link(a.url)}</dd>
  <dt>SHA-256</dt><dd><code class="hash">${esc(a.sha256)}</code></dd>
  <dt>Size</dt><dd>${a.bytes.toLocaleString('en-US')} bytes</dd>
</dl>
${a.parts.map(part).join('\n')}`,
    )
    .join('\n');
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Third-party notices</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect x='1' y='1' width='14' height='14' rx='3' fill='%23353a8c'/%3E%3Cpath d='M4 5l3 3-3 3M8 11h4' stroke='white' stroke-width='1.5' fill='none'/%3E%3C/svg%3E">
  <link rel="stylesheet" href="style.css">
</head>
<body class="notices">
<header>
  <h1>Third-party notices</h1>
  <p><a href="./">Back to the page</a></p>
</header>
<main>
<p>
  The programs on this page run from a root filesystem image built from the recipe
  ${link(`https://github.com/backkem/wasmshim/blob/main/images/${info.recipe}.toml`)}
  (${esc(info.description)}). The page serves it as <code>${esc(image.file)}</code>
  (${image.bytes.toLocaleString('en-US')} bytes gzipped, ${image.size.toLocaleString('en-US')} bytes
  as a tar with SHA-256 <code class="hash">${esc(image.sha256)}</code>).
</p>
<p>
  The software below is not part of wasmshim and keeps its own licence. The image's other files
  (<code>/etc/os-release</code>, <code>/etc/passwd</code>, <code>/etc/group</code>,
  <code>/etc/hostname</code> and its directories) are written by the recipe itself and are under
  wasmshim's licence.
</p>
${sections}
<h2>The node</h2>
<p>
  The node (<code>node/</code>) is compiled from ${link('https://github.com/backkem/wasmshim')},
  under the MIT licence or the Apache License 2.0, at your option:
</p>
${rest.own.map((f) => `<details>\n<summary>${esc(f.file)}</summary>\n<pre>${esc(f.text)}</pre>\n</details>`).join('\n')}
<p>
  It is linked with the Rust standard library (MIT or Apache-2.0, the same two texts;
  ${link('https://github.com/rust-lang/rust/blob/master/COPYRIGHT')}) and with these
  ${rest.crates.length} crates, each under its own licence:
</p>
${rest.crates.map(packageHtml).join('\n')}
<h2>The code that runs it in the tab</h2>
<p>
  The page runs the node on codelib-web, a browser host for WebAssembly components, and the
  JavaScript around the node's modules is generated by jco.
</p>
${rest.npm.map(packageHtml).join('\n')}
</main>
</body>
</html>
`;
}

/** The notices as plain text (the CLI). */
export function noticesText(info) {
  const lines = [`${info.recipe}: ${info.description}`];
  for (const a of info.artifacts) {
    lines.push('', `/${a.path ?? a.file}`, `  url     ${a.url}`, `  sha256  ${a.sha256}`, `  bytes   ${a.bytes}`);
    for (const p of a.parts) {
      lines.push(`  ${p.name} ${p.version}  ${p.license}  ${p.project}`);
      for (const [what, url] of p.sources) lines.push(`    ${what}: ${url}`);
    }
  }
  return lines.join('\n');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log(noticesText(collect(process.argv[2] ?? 'toybox')));
}
