// The page end to end in headless Chromium: serve the built site (dist/, from `npm run build`)
// under /wasmshim/ as GitHub Pages does, run the examples, and compare each with `wasmshim-run`
// running the same node on the same image natively.
//
//   node test/run.mjs
//
// LINUX_RUN=<path>: the native runner (default: ../target/release/wasmshim-run if it is there; build
// it with `cargo build --release -p wasmshim-run`). Without one, the outputs are compared with the
// values below, recorded from it. DEMO_URL=<url>: test a deployed site instead of the local
// build. CHROMIUM=<path>: the browser (default: /usr/bin/chromium if present, else Playwright's).

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

import { serve } from '../scripts/serve.mjs';

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const site = path.join(web, 'dist');
const target = process.env.CARGO_TARGET_DIR ? path.resolve(process.env.CARGO_TARGET_DIR) : path.join(web, '..', 'target');
const deployed = process.env.DEMO_URL;
if (!deployed && !fs.existsSync(path.join(site, 'main.js'))) {
  console.error('dist/ is not built: run `npm run build` first');
  process.exit(2);
}

// The native side: wasmshim-run with the node and image the page was built from.
const linuxRun = process.env.LINUX_RUN ?? path.join(target, 'release', 'wasmshim-run');
const nodeWasm = path.join(target, 'wasm32-wasip2', 'release', 'wasmshim_linux_node.wasm');
const imageTar = path.join(target, 'images', 'toybox.tar');
const haveNative = [linuxRun, nodeWasm, imageTar].every((p) => fs.existsSync(p));

/** Runs `args` natively: stdout, stderr (without wasmshim-run's timing line), exit status, virtual ns. */
function native(args, seed = '1') {
  const r = spawnSync(linuxRun, ['--seed', seed, '--image', imageTar, nodeWasm, ...args], { encoding: 'utf8' });
  const lines = r.stderr.split('\n');
  const timing = lines.find((l) => l.startsWith('wasmshim-run: load='));
  return {
    stdout: r.stdout,
    stderr: lines.filter((l) => l !== timing).join('\n'),
    exitCode: r.status,
    virtualNs: /virtual=(\d+)ns/.exec(timing ?? '')?.[1] ?? null,
  };
}

/** What wasmshim-run printed for these commands (seed 1) when this test was written. */
const RECORDED = {
  'uname -a': { stdout: 'Linux linux 6.6.0 #1 SMP PREEMPT aarch64 Toybox\n', exitCode: 0 },
};

const server = deployed ? null : await serve(site, { port: 0, base: '/wasmshim/' });
const url = deployed ?? `http://127.0.0.1:${server.address().port}/wasmshim/`;
const executablePath = process.env.CHROMIUM ?? (fs.existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined);
const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(m.text());
});
page.on('response', (r) => {
  if (r.status() >= 400) errors.push(`HTTP ${r.status()}: ${r.url()}`);
});
const external = [];
page.on('request', (r) => {
  if (!r.url().startsWith(url) && !r.url().startsWith('data:') && !r.url().startsWith('blob:')) external.push(r.url());
});

const steps = [];
async function step(name, fn) {
  const t0 = performance.now();
  try {
    const detail = await fn();
    const ms = Math.round(performance.now() - t0);
    steps.push({ name, ok: true, ms, detail });
    console.log(`[pass] ${name} (${ms} ms)${detail ? ` ${JSON.stringify(detail)}` : ''}`);
  } catch (e) {
    steps.push({ name, ok: false, error: e.message });
    console.log(`[FAIL] ${name}: ${e.message.split('\n').slice(0, 8).join('\n    ')}`);
  }
}

/** Runs `command` on the page (the command field, then Run); resolves to the page's record. */
async function run(command, seed = '1') {
  await page.fill('#command', command);
  await page.fill('#seed', seed);
  await page.click('#run');
  await page.waitForFunction(() => window.lastRun !== undefined, null, { timeout: 300_000 });
  const record = await page.evaluate(() => window.lastRun);
  if (record.outcome !== 'ok') throw new Error(`run ended ${record.outcome}: ${await page.textContent('#status')}`);
  return { ...record, status: await page.textContent('#status') };
}

/** The page's arguments for `command`, as wasmshim-run gets them. */
const argsOf = (command) => page.evaluate((c) => window.lastRun?.args ?? null, command);

/**
 * Runs `command` in the page and natively (seed `seed`): the same stdout, stderr, exit status
 * and virtual time are required. (Not guaranteed in general, but true of these commands: none
 * of them depends on random bytes, which the two hosts draw from different generators.)
 */
async function compare(command, key, seed = '1') {
  const b = await run(command, seed);
  const args = await argsOf(command);
  const n = haveNative ? native(args, seed) : { ...RECORDED[key], stderr: undefined, virtualNs: undefined };
  if (n.stdout === undefined) throw new Error(`no native run and no recorded output for ${key}`);
  if (b.stdout !== n.stdout) throw new Error(`stdout differs:\n browser ${JSON.stringify(b.stdout)}\n native  ${JSON.stringify(n.stdout)}`);
  if (b.exitCode !== n.exitCode) throw new Error(`exit status: browser ${b.exitCode}, native ${n.exitCode}`);
  if (n.stderr !== undefined && b.stderr !== n.stderr) throw new Error(`stderr differs:\n browser ${JSON.stringify(b.stderr)}\n native  ${JSON.stringify(n.stderr)}`);
  if (n.virtualNs !== undefined && b.virtualNs !== n.virtualNs) throw new Error(`virtual time: browser ${b.virtualNs} ns, native ${n.virtualNs} ns`);
  return { b, n };
}

const brief = (b, n) => ({
  stdout: b.stdout.length > 80 ? `${b.stdout.slice(0, 77)}...` : b.stdout,
  exit: b.exitCode,
  virtualNs: b.virtualNs,
  ...(n?.virtualNs !== undefined ? { nativeVirtualNs: n.virtualNs } : {}),
  wallMs: b.wallMs,
});

await step('loads and compiles the node and the image', async () => {
  const t0 = performance.now();
  await page.goto(url);
  const notice = await page.isVisible('#loading');
  await page.waitForFunction(() => document.body.dataset.state !== 'loading', null, { timeout: 120_000 });
  const state = await page.evaluate(() => document.body.dataset.state);
  if (state !== 'ready') throw new Error(`state ${state}: ${await page.textContent('#status')}`);
  return { readyMs: Math.round(performance.now() - t0), loadingNoticeShown: notice, size: await page.textContent('#size'), native: haveNative ? linuxRun : 'recorded values' };
});

await step('uname -a, as wasmshim-run prints it', async () => {
  const { b, n } = await compare('/bin/toybox uname -a', 'uname -a');
  if (b.stdout !== 'Linux linux 6.6.0 #1 SMP PREEMPT aarch64 Toybox\n') throw new Error(`stdout ${JSON.stringify(b.stdout)}`);
  return brief(b, n);
});

await step("sh -c 'echo hello; ls /'", async () => {
  const { b, n } = await compare("/bin/toybox sh -c 'echo hello; ls /'", 'echo; ls');
  if (!b.stdout.startsWith('hello\n') || !b.stdout.includes('\netc\n')) throw new Error(`stdout ${JSON.stringify(b.stdout)}`);
  return brief(b, n);
});

await step('cat /etc/os-release', async () => {
  const { b, n } = await compare('/bin/toybox cat /etc/os-release', 'os-release');
  if (!b.stdout.includes('PRETTY_NAME="toybox 0.8.15 on wasmshim"')) throw new Error(`stdout ${JSON.stringify(b.stdout)}`);
  return brief(b, n);
});

await step('a shell loop', async () => {
  const { b, n } = await compare("/bin/toybox sh -c 'i=0; while [ $i -lt 5 ]; do echo $i; i=$((i+1)); done'", 'loop');
  if (b.stdout !== '0\n1\n2\n3\n4\n' || b.exitCode !== 0) throw new Error(`stdout ${JSON.stringify(b.stdout)}, exit ${b.exitCode}`);
  return brief(b, n);
});

await step('sha256sum of the binary itself', async () => {
  const { b, n } = await compare('/bin/toybox sha256sum /bin/toybox', 'sha256sum');
  if (!b.stdout.startsWith('be360e78774c9ad599f2c0a606a2a7579840f9957ffccdc9898b0dd8e3f87e88 ')) throw new Error(`stdout ${JSON.stringify(b.stdout)}`);
  return brief(b, n);
});

await step('sleep 10: ten virtual seconds in far less wall time', async () => {
  const { b, n } = await compare('/bin/toybox sleep 10', 'sleep');
  const v = BigInt(b.virtualNs);
  if (v < 10_000_000_000n || v > 10_100_000_000n) throw new Error(`virtual ${b.virtualNs} ns`);
  if (b.wallMs > 2_000) throw new Error(`wall ${b.wallMs} ms`);
  return { ...brief(b, n), status: b.status };
});

await step('one seed: the same bytes and the same virtual time', async () => {
  const command = "/bin/toybox sh -c 'uuidgen; mcookie; date'";
  const a = await run(command, '7');
  const again = await run(command, '7');
  if (a.stdout !== again.stdout || a.stderr !== again.stderr) throw new Error(`output differs: ${JSON.stringify(a.stdout)} vs ${JSON.stringify(again.stdout)}`);
  if (a.virtualNs !== again.virtualNs) throw new Error(`virtual ${a.virtualNs} vs ${again.virtualNs}`);
  const other = await run(command, '8');
  if (other.stdout === a.stdout) throw new Error('another seed printed the same random bytes');
  const n = haveNative ? native(await argsOf(command), '7') : null;
  return { stdout: a.stdout, virtualNs: a.virtualNs, otherSeed: other.stdout.split('\n')[0], ...(n ? { native: n.stdout, nativeVirtualNs: n.virtualNs } : {}) };
});

await step('a program that does not exist: non-zero exit status', async () => {
  const missing = await compare('/bin/nope', 'nope');
  if (missing.b.exitCode !== 125 || !missing.b.stderr.includes('/bin/nope')) throw new Error(`exit ${missing.b.exitCode}, stderr ${JSON.stringify(missing.b.stderr)}`);
  const unknown = await compare('/bin/toybox nope', 'toybox nope');
  if (unknown.b.exitCode !== 127) throw new Error(`toybox nope: exit ${unknown.b.exitCode}`);
  const exit3 = await compare("/bin/toybox sh -c 'exit 3'", 'exit 3');
  if (exit3.b.exitCode !== 3) throw new Error(`exit 3: ${exit3.b.exitCode}`);
  return { '/bin/nope': [missing.b.exitCode, missing.b.stderr.trim()], 'toybox nope': [unknown.b.exitCode, unknown.b.stderr.trim()], 'exit 3': exit3.b.exitCode, status: missing.b.status };
});

await step('the shell: lines typed on a console', async () => {
  await page.click('#shell');
  await page.waitForSelector('#typing:not([hidden])');
  await page.waitForFunction(() => document.getElementById('output').textContent.length > 0, null, { timeout: 60_000 });
  const send = async (line) => {
    await page.fill('#line', line);
    await page.press('#line', 'Enter');
  };
  await send('echo typed-$((40+2))');
  await page.waitForFunction(() => document.getElementById('output').textContent.includes('typed-42'), null, { timeout: 60_000 });
  await send('exit 5');
  await page.waitForFunction(() => window.lastRun !== undefined, null, { timeout: 60_000 });
  const r = await page.evaluate(() => window.lastRun);
  if (r.exitCode !== 5) throw new Error(`exit ${r.exitCode} (${r.outcome}): ${await page.textContent('#status')}`);
  return { screen: (await page.textContent('#output')).slice(0, 200), exit: r.exitCode, virtualNs: r.virtualNs };
});

await step('the notices page', async () => {
  await page.goto(new URL('NOTICES.html', url).href);
  const text = await page.textContent('body');
  for (const want of ['toybox 0.8.15', '0BSD', 'musl 1.2.6', 'MIT', 'be360e78774c9ad599f2c0a606a2a7579840f9957ffccdc9898b0dd8e3f87e88', 'toybox-0.8.15.tar.gz']) {
    if (!text.includes(want)) throw new Error(`no "${want}"`);
  }
  // The licence texts of what the page ships travel with it: this repository's two, a crate
  // linked into the node, the kit and jco.
  for (const want of [
    'Apache License',
    'Permission is hereby granted, free of charge',
    'wit-bindgen',
    'codelib-web',
    'codelib contributors',
    '@bytecodealliance/jco',
    'LLVM Exceptions to the Apache 2.0 License',
  ]) {
    if (!text.includes(want)) throw new Error(`the notices lack "${want}"`);
  }
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  if (overflow > 0) throw new Error(`scrolls sideways by ${overflow} px`);
  await page.goto(url);
  await page.waitForFunction(() => document.body.dataset.state === 'ready', null, { timeout: 120_000 });
});

await step('a browser without JSPI gets a clear message', async () => {
  const other = await browser.newPage();
  await other.addInitScript(() => {
    delete WebAssembly.Suspending;
    delete WebAssembly.promising;
  });
  await other.goto(url);
  await other.waitForFunction(() => document.body.dataset.state === 'unsupported', null, { timeout: 10_000 });
  const shown = await other.isVisible('#nojspi');
  const disabled = await other.isDisabled('#run');
  await other.close();
  if (!shown || !disabled) throw new Error(`message shown ${shown}, Run disabled ${disabled}`);
});

await step('the input line shows only while the shell runs', async () => {
  if (await page.isVisible('#typing')) throw new Error('the input line is visible with no shell running');
});

await step('fits a phone screen', async () => {
  await run('/bin/toybox ls -l /proc/self/');
  fs.mkdirSync(path.join(web, 'test-results'), { recursive: true });
  const widths = {};
  for (const [name, file] of [['page', ''], ['notices', 'NOTICES.html']]) {
    if (file) await page.goto(new URL(file, url).href);
    await page.setViewportSize({ width: 360, height: 780 });
    widths[name] = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    await page.screenshot({ path: path.join(web, `test-results/phone-${name}.png`), fullPage: true });
  }
  await page.goBack();
  await page.setViewportSize({ width: 1100, height: 900 });
  await page.waitForFunction(() => document.body.dataset.state === 'ready', null, { timeout: 120_000 });
  await run('/bin/toybox sh -c \'echo hello; ls /\'');
  await page.screenshot({ path: path.join(web, 'test-results/desktop.png'), fullPage: true });
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.screenshot({ path: path.join(web, 'test-results/desktop-dark.png'), fullPage: true });
  const wide = Object.entries(widths).filter(([, w]) => w > 0);
  if (wide.length > 0) throw new Error(`scrolls sideways: ${JSON.stringify(widths)}`);
});

await step('the page made no external requests', async () => {
  if (external.length > 0) throw new Error(external.join(', '));
});

await browser.close();
server?.close();
if (errors.length > 0) console.log(`page errors:\n  ${errors.join('\n  ')}`);
const failed = steps.filter((s) => !s.ok).length;
fs.mkdirSync(path.join(web, 'test-results'), { recursive: true });
fs.writeFileSync(path.join(web, 'test-results/page.json'), `${JSON.stringify({ steps, errors }, null, 2)}\n`);
console.log(failed === 0 ? `page test: all ${steps.length} steps passed` : `page test: ${failed} of ${steps.length} steps failed`);
process.exit(failed === 0 && errors.length === 0 ? 0 : 1);
