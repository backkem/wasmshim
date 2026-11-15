// Assembles the page in dist/ (`npm run build`; `cargo xtask web` builds the Rust inputs first):
//
//   dist/index.html, style.css          the page (page/)
//   dist/main.js, worker.js, node.js    src/, compiled by tsc
//   dist/codelib-web/dist/              the kit's browser modules (no transpile step, no CLI)
//   dist/node/                          wasmshim_linux_node.wasm transpiled with the kit; its core modules
//                                       without DWARF (`.debug_*` sections; the name section stays)
//   dist/toybox.tar.gz                  the root filesystem image (images/toybox.toml), gzipped
//                                       (the page inflates it)
//   dist/NOTICES.html                   third-party notices, from the image recipe (notices.mjs)
//   dist/assets.json                    what the page downloads, with sizes
//
// Every URL in the page is relative, so the site works under any base path (GitHub Pages serves
// it from /wasmshim/). The Rust inputs come from the workspace's target directory
// ($CARGO_TARGET_DIR, else ../target): `cargo xtask build` and `cargo xtask image toybox`.

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

import { transpile } from 'codelib-web/transpile';

import { collect, noticesHtml, shipped } from './notices.mjs';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(web, 'dist');
const target = process.env.CARGO_TARGET_DIR ? resolve(process.env.CARGO_TARGET_DIR) : join(web, '..', 'target');
const nodeWasm = join(target, 'wasm32-wasip2', 'release', 'wasmshim_linux_node.wasm');
const IMAGE = 'toybox';
const imageTar = join(target, 'images', `${IMAGE}.tar`);

const missing = [nodeWasm, imageTar].filter((p) => !existsSync(p));
if (missing.length > 0) {
  console.error(`build: missing ${missing.map((p) => relative(web, p)).join(', ')}`);
  console.error(`run \`cargo xtask build\` and \`cargo xtask image ${IMAGE}\` first (or \`cargo xtask web\`)`);
  process.exit(2);
}

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

// The TypeScript: tsc from this package's own install.
const require = createRequire(import.meta.url);
const tsc = join(dirname(require.resolve('typescript/package.json')), 'bin', 'tsc');
try {
  execFileSync(process.execPath, [tsc, '-p', web], { stdio: 'inherit' });
} catch {
  console.error('build: tsc failed');
  process.exit(1);
}

for (const f of ['index.html', 'style.css']) cpSync(join(web, 'page', f), join(out, f));

// The kit's browser modules, as `codelib-web browser-files` lists and copies them (everything but
// its Node-only modules), at the path the binding imports them from.
const walk = (dir) => readdirSync(dir).flatMap((n) => (statSync(join(dir, n)).isDirectory() ? walk(join(dir, n)) : [join(dir, n)]));
const kitCli = join(dirname(require.resolve('codelib-web/package.json')), 'dist', 'cli.js');
const copied = execFileSync(process.execPath, [kitCli, 'browser-files', '-o', join(out, 'codelib-web', 'dist')], { encoding: 'utf8' });
const kitFiles = copied.split('\n').filter((l) => l.startsWith('  ')).length;

/**
 * A core module without its DWARF: custom sections named `.debug_*` (the release build keeps line
 * tables, most of the module's bytes). Everything else, the name section included, is unchanged.
 */
function withoutDebug(bytes) {
  const leb = (at) => {
    let n = 0;
    let shift = 0;
    for (;;) {
      const b = bytes[at++];
      n += (b & 0x7f) * 2 ** shift;
      shift += 7;
      if ((b & 0x80) === 0) return [n, at];
    }
  };
  const keep = [bytes.subarray(0, 8)];
  for (let at = 8; at < bytes.length; ) {
    const start = at;
    const id = bytes[at++];
    const [size, body] = leb(at);
    const end = body + size;
    let drop = false;
    if (id === 0) {
      const [len, name] = leb(body);
      drop = Buffer.from(bytes.subarray(name, name + len)).toString('utf8').startsWith('.debug_');
    }
    if (!drop) keep.push(bytes.subarray(start, end));
    at = end;
  }
  return Buffer.concat(keep);
}

// The node, transpiled. As `codelib-web transpile <wasm> -o <dir> --name wasmshim-linux-node
// --no-wasi-async`: WASI 0.2's blocking functions stay synchronous imports (the kit's virtual
// world never blocks them; the node waits through WASI 0.3's `wait-until`).
const nodeDir = join(out, 'node');
const t0 = performance.now();
const manifest = await transpile(nodeWasm, { outDir: nodeDir, name: 'wasmshim-linux-node', noWasiAsync: true });
rmSync(join(nodeDir, 'interfaces'), { recursive: true, force: true });
rmSync(join(nodeDir, 'wasmshim-linux-node.d.ts'), { force: true });
const coreModules = manifest.coreModules.map((file) => {
  const path = join(nodeDir, file);
  const before = statSync(path).size;
  const stripped = withoutDebug(readFileSync(path));
  writeFileSync(path, stripped);
  return { file, bytes: stripped.length, debugBytes: before - stripped.length };
});
const patches = manifest.patches.map((p) => `${p.name}x${p.count}`).join(' ');
console.log(`  wasmshim-linux-node: ${manifest.coreModules.length} core modules, ${Math.round(performance.now() - t0)} ms; patches: ${patches || '(none)'}`);

// The image, gzipped: about half the bytes over the wire; the page inflates it with
// DecompressionStream.
const tar = readFileSync(imageTar);
const gz = gzipSync(tar, { level: 9 });
writeFileSync(join(out, `${IMAGE}.tar.gz`), gz);
const image = {
  file: `${IMAGE}.tar.gz`,
  bytes: gz.length,
  size: tar.length,
  sha256: createHash('sha256').update(tar).digest('hex'),
};

// The notices, from the recipe and the downloaded artifacts' own metadata.
const rest = shipped();
writeFileSync(join(out, 'NOTICES.html'), noticesHtml(collect(IMAGE), image, rest));
console.log(`  notices: ${rest.crates.length} crates (${rest.crates.filter((c) => c.files.length === 0).length} without a licence file), ${rest.npm.length} npm packages`);

const assets = {
  node: {
    module: `node/${manifest.module}`,
    moduleBytes: statSync(join(nodeDir, manifest.module)).size,
    component: manifest.component,
    coreModules: coreModules.map(({ file, bytes }) => ({ file, path: `node/${file}`, bytes })),
  },
  image,
};
writeFileSync(join(out, 'assets.json'), `${JSON.stringify(assets, null, 2)}\n`);

const total = walk(out).reduce((n, f) => n + statSync(f).size, 0);
const debug = coreModules.reduce((n, c) => n + c.debugBytes, 0);
console.log(`build: ${kitFiles} kit modules, the node (${(debug / 1e6).toFixed(1)} MB of DWARF left out), the image ${(tar.length / 1e6).toFixed(1)} MB (${(gz.length / 1e6).toFixed(1)} MB gzipped); ${(total / 1e6).toFixed(1)} MB -> ${relative(web, out)}`);
