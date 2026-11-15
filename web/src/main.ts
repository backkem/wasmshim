// The page: downloads and compiles the node and the image once, then runs each command in a
// fresh Worker (worker.ts), showing the guest's output as it arrives and, at the end, the exit
// status, the virtual time and the wall time. "Shell" runs toybox's shell on a console, with
// keystrokes sent line by line.

import { hasJspi, JSPI_MISSING } from './codelib-web/dist/jspi.js';
import { runInWorker, type WorkerRun } from './codelib-web/dist/worker/index.js';
import type { NodeSource } from './node.js';
import type { Done, Init, Input, Output } from './worker.js';

/** What the build wrote next to the page (scripts/build.mjs). */
interface Assets {
  node: { module: string; moduleBytes: number; coreModules: { file: string; path: string; bytes: number }[] };
  image: { file: string; bytes: number; size: number; sha256: string };
}

/** The last run, as plain values (for tests and the console). */
interface RunRecord {
  args: string[];
  seed: string;
  stdout: string;
  stderr: string;
  exitCode: number | null;
  virtualNs: string | null;
  wallMs: number | null;
  outcome: string;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const commandInput = $<HTMLInputElement>('command');
const seedInput = $<HTMLInputElement>('seed');
const runButton = $<HTMLButtonElement>('run');
const shellButton = $<HTMLButtonElement>('shell');
const stopButton = $<HTMLButtonElement>('stop');
const output = $<HTMLPreElement>('output');
const status = $('status');
const loading = $('loading');
const typing = $<HTMLFormElement>('typing');
const lineInput = $<HTMLInputElement>('line');

function setStatus(text: string, state: string): void {
  status.textContent = text;
  document.body.dataset.state = state;
}

const mb = (n: number) => `${(n / 1e6).toFixed(1)} MB`;

/** `10000000000` ns as `10.000000 s`. */
function seconds(ns: bigint): string {
  const s = ns / 1_000_000_000n;
  const frac = String(ns % 1_000_000_000n).padStart(9, '0').slice(0, 6);
  return `${s}.${frac} s`;
}

/**
 * Splits a command line into arguments as a POSIX shell would for plain words: single quotes
 * keep everything literally, double quotes keep spaces (a backslash escapes `"`, `\`, `$` and
 * a backtick inside them), and a backslash outside quotes escapes the next character. No
 * expansion: `$` and globs reach the program as they are.
 */
export function splitArgs(line: string): string[] {
  const args: string[] = [];
  let cur = '';
  let word = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i]!;
    if (c === "'") {
      const end = line.indexOf("'", i + 1);
      if (end < 0) throw new Error('unterminated single quote');
      cur += line.slice(i + 1, end);
      i = end;
      word = true;
    } else if (c === '"') {
      word = true;
      for (i++; ; i++) {
        if (i >= line.length) throw new Error('unterminated double quote');
        const d = line[i]!;
        if (d === '"') break;
        if (d === '\\' && i + 1 < line.length && '"\\$`'.includes(line[i + 1]!)) cur += line[++i];
        else cur += d;
      }
    } else if (c === '\\' && i + 1 < line.length) {
      cur += line[++i];
      word = true;
    } else if (/\s/.test(c)) {
      if (word) args.push(cur);
      cur = '';
      word = false;
    } else {
      cur += c;
      word = true;
    }
  }
  if (word) args.push(cur);
  return args;
}

/** Appends guest output; a console's output is cleaned of terminal control sequences. */
class Screen {
  private readonly decoders = { stdout: new TextDecoder(), stderr: new TextDecoder() };
  text = { stdout: '', stderr: '' };
  console = false;

  clear(consoleMode: boolean): void {
    this.decoders.stdout = new TextDecoder();
    this.decoders.stderr = new TextDecoder();
    this.text = { stdout: '', stderr: '' };
    this.console = consoleMode;
    output.replaceChildren();
  }

  write(stream: 'stdout' | 'stderr', bytes: Uint8Array): void {
    let s = this.decoders[stream].decode(bytes, { stream: true });
    this.text[stream] += s;
    if (this.console) {
      // A terminal's output: CRLF line ends, CSI sequences (colours, cursor moves), and
      // "\b \b" for an erased character.
      s = s.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '').replace(/\x1b[()][0-9A-Za-z]/g, '').replace(/\r\n/g, '\n').replace(/\r/g, '');
    }
    const atBottom = output.scrollTop + output.clientHeight >= output.scrollHeight - 4;
    for (const part of s.split(/(\x08)/)) {
      if (part === '\x08') {
        const last = output.lastChild;
        if (last?.textContent) last.textContent = last.textContent.slice(0, -1);
        continue;
      }
      if (part === '') continue;
      const last = output.lastElementChild as HTMLElement | null;
      if (last?.dataset.stream === stream) last.append(part);
      else {
        const span = document.createElement('span');
        span.dataset.stream = stream;
        span.textContent = part;
        output.append(span);
      }
    }
    if (atBottom) output.scrollTop = output.scrollHeight;
  }
}

async function load(): Promise<{ node: NodeSource; image: Uint8Array }> {
  const assets = (await (await fetch('assets.json')).json()) as Assets;
  const nodeBytes = assets.node.moduleBytes + assets.node.coreModules.reduce((n, c) => n + c.bytes, 0);
  $('size').textContent = `the node, ${mb(nodeBytes)}, and the image, ${mb(assets.image.bytes)}`;
  const modules = Promise.all(assets.node.coreModules.map(async (c) => [c.file, await WebAssembly.compileStreaming(fetch(c.path))] as const));
  const image = (async () => {
    const r = await fetch(assets.image.file);
    if (!r.ok || !r.body) throw new Error(`${assets.image.file}: HTTP ${r.status}`);
    const raw = new Uint8Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    if (raw.length !== assets.image.size) throw new Error(`${assets.image.file}: ${raw.length} bytes, expected ${assets.image.size}`);
    return raw;
  })();
  return {
    node: { url: new URL(assets.node.module, document.baseURI).href, modules: Object.fromEntries(await modules) },
    image: await image,
  };
}

async function main(): Promise<void> {
  if (!hasJspi()) {
    loading.hidden = true;
    $('nojspi').hidden = false;
    $('nojspi-detail').textContent = JSPI_MISSING;
    setStatus('This browser cannot run the node.', 'unsupported');
    return;
  }
  setStatus('Loading…', 'loading');
  let loaded: Awaited<ReturnType<typeof load>>;
  try {
    loaded = await load();
  } catch (err) {
    loading.hidden = true;
    setStatus(`Loading failed: ${err instanceof Error ? err.message : String(err)}`, 'error');
    return;
  }
  loading.hidden = true;
  const screen = new Screen();
  let current: WorkerRun<Done> | null = null;

  const idle = (on: boolean) => {
    runButton.disabled = !on;
    shellButton.disabled = !on;
    stopButton.hidden = on;
    for (const b of document.querySelectorAll<HTMLButtonElement>('[data-command]')) b.disabled = !on;
  };
  idle(true);
  setStatus('Ready. Type a command or pick an example, then press Run.', 'ready');

  async function start(args: string[], interactive: boolean): Promise<void> {
    const seed = seedInput.value.trim() || '1';
    if (!/^\d+$/.test(seed) || BigInt(seed) >= 1n << 64n) {
      setStatus('The seed is a whole number from 0 to 2^64 - 1.', 'error');
      return;
    }
    idle(false);
    screen.clear(interactive);
    delete (window as { lastRun?: unknown }).lastRun;
    const record: RunRecord = { args, seed, stdout: '', stderr: '', exitCode: null, virtualNs: null, wallMs: null, outcome: 'running' };
    setStatus(interactive ? 'Starting the shell…' : 'Running…', 'running');
    const nodeArgs = interactive ? ['--console', ...args] : args;
    current = runInWorker<Init, Done>(new URL('worker.js', import.meta.url), {
      // The image is copied into the Worker (structured clone); the page keeps its own.
      init: { node: loaded.node, image: loaded.image, args: nodeArgs, seed, interactive },
      startTimeoutMs: 60_000,
      // A shell waits for its user; a command gets five minutes.
      ...(interactive ? {} : { timeoutMs: 300_000 }),
      onMessage: (m) => {
        const o = m as Output;
        if (o.type === 'output') screen.write(o.stream, o.bytes);
      },
    });
    if (interactive) {
      typing.hidden = false;
      lineInput.value = '';
      lineInput.focus();
      setStatus('The shell is running: type a line and press Enter. Ctrl-D ends it.', 'shell');
    }
    const outcome = await current.outcome;
    current = null;
    typing.hidden = true;
    idle(true);
    record.stdout = screen.text.stdout;
    record.stderr = screen.text.stderr;
    record.outcome = outcome.status;
    if (outcome.status === 'ok') {
      const v = outcome.value;
      record.exitCode = v.exitCode;
      record.virtualNs = v.virtualNs;
      record.wallMs = v.wallMs;
      setStatus(`Exit status ${v.exitCode} · virtual time ${seconds(BigInt(v.virtualNs))} · wall time ${v.wallMs} ms`, v.exitCode === 0 ? 'done' : 'failed');
    } else if (outcome.status === 'error') {
      setStatus(`The node failed: ${outcome.error.message}`, 'error');
    } else if (outcome.status === 'timeout') {
      setStatus(`Stopped: the run took longer than ${outcome.limitMs / 1000} s of wall time.`, 'error');
    } else {
      setStatus('Stopped.', 'error');
    }
    (window as { lastRun?: unknown }).lastRun = record;
  }

  const runCommand = () => {
    let args: string[];
    try {
      args = splitArgs(commandInput.value);
    } catch (err) {
      setStatus(`The command line has an ${err instanceof Error ? err.message : String(err)}.`, 'error');
      return;
    }
    if (args.length === 0) {
      setStatus('Type a command first, for example /bin/toybox uname -a.', 'error');
      return;
    }
    void start(args, false);
  };

  runButton.addEventListener('click', runCommand);
  commandInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !runButton.disabled) runCommand();
  });
  seedInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !runButton.disabled) runCommand();
  });
  for (const b of document.querySelectorAll<HTMLButtonElement>('[data-command]')) {
    b.addEventListener('click', () => {
      commandInput.value = b.dataset.command!;
      runCommand();
    });
  }
  shellButton.addEventListener('click', () => void start(['/bin/toybox', 'sh', '-i'], true));
  stopButton.addEventListener('click', () => current?.stop());

  const send = (input: Input) => current?.post(input);
  typing.addEventListener('submit', (e) => {
    e.preventDefault();
    send({ type: 'input', bytes: new TextEncoder().encode(`${lineInput.value}\r`) });
    lineInput.value = '';
  });
  lineInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && (e.key === 'c' || e.key === 'd')) {
      e.preventDefault();
      send({ type: 'input', bytes: new Uint8Array([e.key === 'c' ? 3 : 4]) });
    }
  });
  $<HTMLButtonElement>('ctrl-c').addEventListener('click', () => send({ type: 'input', bytes: new Uint8Array([3]) }));
  $<HTMLButtonElement>('ctrl-d').addEventListener('click', () => send({ type: 'input', bytes: new Uint8Array([4]) }));
}

void main();
