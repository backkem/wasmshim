// One run of the node in a Worker (the kit's `serveRun`; the page starts it with `runInWorker`).
// `init` carries the compiled node, the image and the command line; the guest's output streams
// back as messages while it runs; keystrokes come in as messages for a `--console` run; the exit
// status and the times are the run's value.

import { serveRun } from './codelib-web/dist/worker/index.js';
import { runNode, type NodeSource } from './node.js';

export interface Init {
  node: NodeSource;
  /** The root filesystem image (an uncompressed tar), served as blob `rootfs`. */
  image: Uint8Array;
  /** The node's command line, without its own name. */
  args: string[];
  /** A decimal u64. */
  seed: string;
  /** Keystrokes arrive as `Input` messages (the node runs with `--console`). */
  interactive?: boolean;
}

export interface Done {
  exitCode: number;
  /** Virtual nanoseconds at the end, as a decimal string. */
  virtualNs: string;
  wallMs: number;
}

/** What the Worker posts while the guest runs. */
export type Output = { type: 'output'; stream: 'stdout' | 'stderr'; bytes: Uint8Array };

/** What the page posts to an interactive run. */
export type Input = { type: 'input'; bytes: Uint8Array } | { type: 'eof' };

serveRun<Init, Done>(async (init, ctx) => {
  const post = (stream: Output['stream']) => (bytes: Uint8Array) => {
    const copy = bytes.slice();
    ctx.post({ type: 'output', stream, bytes: copy } satisfies Output, [copy.buffer]);
  };

  // Keystrokes, queued until the node's console pump asks for them.
  const typed: Uint8Array[] = [];
  let ended = false;
  let wake: (() => void) | null = null;
  ctx.onMessage((m) => {
    const input = m as Input;
    if (input.type === 'input') typed.push(input.bytes);
    else ended = true;
    wake?.();
    wake = null;
  });
  const stdin = async (): Promise<Uint8Array | undefined> => {
    while (typed.length === 0 && !ended) await new Promise<void>((r) => (wake = r));
    return typed.shift();
  };

  const result = await runNode({
    node: init.node,
    blobs: { rootfs: init.image },
    args: init.args,
    seed: BigInt(init.seed),
    stdout: post('stdout'),
    stderr: post('stderr'),
    ...(init.interactive ? { stdin } : {}),
    onRunning: () => ctx.running(),
  });
  return { exitCode: result.exitCode, virtualNs: String(result.virtualNs), wallMs: Math.round(result.wallMs) };
});
