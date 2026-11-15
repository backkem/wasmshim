// Runs `wasmshim-linux-node` (components/linux-node), transpiled with the codelib-web kit, the way
// `wasmshim-run` (crates/linux-run) runs it natively:
//
// - WASI from the kit's `wasiImports` over `virtualWorld({ seed })`: a monotonic clock that
//   starts at 0 and moves only when the node waits, jumping to its next timer; a wall clock at
//   2026-01-01T00:00:00Z plus that; seeded entropy; the command line as arguments;
// - `linux:host/blob` from images held in memory (the root filesystem is blob `rootfs`);
// - `wasi:cli/run`, called once; the exit status is the guest's (`wasi:cli/exit`), or 0 when
//   `run` returns.
//
// One run per JavaScript realm: jco's runtime and the kit's event loop are global, so call
// `runNode` once per Worker (worker.ts).

import { instantiate } from './codelib-web/dist/load.js';
import { WasiExit, wasiImports } from './codelib-web/dist/wasi/index.js';
import { virtualWorld } from './codelib-web/dist/worlds.js';

/** Where the transpiled node is: its JS module and its core modules, compiled. */
export interface NodeSource {
  /** URL of the transpiled `wasmshim-linux-node.js`. */
  url: string;
  /** Its core modules by the file name the glue asks for. */
  modules: Record<string, WebAssembly.Module>;
}

export interface RunOptions {
  node: NodeSource;
  /** Blobs by name; the node reads its root filesystem from `rootfs` (its `--image` option). */
  blobs: Record<string, Uint8Array>;
  /** The node's command line, without the program name: options, then the guest's argv. */
  args: string[];
  /** Seeds both random streams (as `wasmshim-run --seed`). */
  seed: bigint;
  stdout?: (bytes: Uint8Array) => void;
  stderr?: (bytes: Uint8Array) => void;
  /**
   * Keystrokes for a node started with `--console`: called for the next chunk, resolves to
   * `undefined` at the end of input. Without it, stdin is empty.
   */
  stdin?: () => Promise<Uint8Array | undefined>;
  /** Called once the node is instantiated, just before `run`. */
  onRunning?: () => void;
}

export interface RunResult {
  /** The guest's exit status (128 + signal for a signal death); 125 when the node itself failed. */
  exitCode: number;
  /** Virtual time at the end of the run, in nanoseconds. */
  virtualNs: bigint;
  /** Wall-clock time of `run` itself (after instantiation), in milliseconds. */
  wallMs: number;
}

/** A blob read's error, as jco lowers a thrown value into `result::err`. */
type BlobError = { tag: 'not-found' } | { tag: 'out-of-range' } | { tag: 'io'; val: string };

/** `linux:host/blob` (wit/linux-host/host.wit) over blobs in memory; also `blob-words`. */
export function blobImports(blobs: Record<string, Uint8Array>): Record<string, Record<string, unknown>> {
  const find = (name: string): Uint8Array => {
    const b = blobs[name];
    if (b === undefined) throw { tag: 'not-found' } satisfies BlobError;
    return b;
  };
  const range = (name: string, offset: bigint, len: number): Uint8Array => {
    const b = find(name);
    if (offset < 0n || offset + BigInt(len) > BigInt(b.length)) throw { tag: 'out-of-range' } satisfies BlobError;
    const at = Number(offset);
    // A copy: jco lowers the list into the guest's memory and may keep no view of ours.
    return b.slice(at, at + len);
  };
  return {
    'linux:host/blob': {
      size: (name: string) => BigInt(find(name).length),
      read: range,
    },
    // A node built with `blob-words` (composed with its image provider) reads this instead; the
    // plain node does not import it.
    'linux:host/blob-words': {
      readWords: (name: string, offset: bigint, len: number) => {
        const bytes = range(name, offset, len);
        const padded = new Uint8Array(Math.ceil(len / 8) * 8);
        padded.set(bytes);
        return new BigUint64Array(padded.buffer);
      },
    },
  };
}

/** Instantiates the node and runs it to its exit. */
export async function runNode(o: RunOptions): Promise<RunResult> {
  const world = virtualWorld({
    seed: o.seed,
    // As wasmshim-run: the node's own name first; it skips it.
    args: ['linux-node', ...o.args],
    stdout: o.stdout ?? (() => undefined),
    stderr: o.stderr ?? (() => undefined),
    // The kit waits for a pending read `patient`ly: while the guest has timers, virtual time
    // runs; when it waits for keystrokes alone, the loop waits for them in wall time.
    ...(o.stdin ? { stdin: o.stdin } : {}),
  });
  const wasi = wasiImports(world);
  // The status the guest exits with. The kit's `exit` throws `WasiExit` to unwind the guest;
  // jco may wrap what reaches us, so the status is kept here as well.
  let exited: number | undefined;
  const exitImports = wasi['wasi:cli/exit']!;
  const exit = {
    ...exitImports,
    exit: (status: { tag: string }) => {
      exited ??= status?.tag === 'ok' ? 0 : 1;
      throw new WasiExit(exited);
    },
    exitWithCode: (code: number) => {
      exited ??= code;
      throw new WasiExit(code);
    },
  };
  const node = await instantiate(o.node.url, { ...wasi, 'wasi:cli/exit': exit, ...blobImports(o.blobs) }, { modules: o.node.modules });
  o.onRunning?.();
  const t0 = performance.now();
  let exitCode: number;
  try {
    await node.call('wasi:cli/run@0.3.0#run');
    exitCode = exited ?? 0;
  } catch (err) {
    if (exited !== undefined) exitCode = exited;
    else if (err instanceof WasiExit) exitCode = err.code;
    // `run` returned `err` (jco throws it as a ComponentError): status 1, as wasmshim-run says.
    else if ((err as { name?: string } | null)?.name === 'ComponentError') exitCode = 1;
    else throw err;
  }
  const wallMs = performance.now() - t0;
  world.dispose();
  return { exitCode, virtualNs: world.now(), wallMs };
}
