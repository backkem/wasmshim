// A static file server for the built page:
//
//   node scripts/serve.mjs [DIR] [PORT] [BASE]     (defaults: dist 8080 /wasmshim/)
//
// Serves DIR under the path BASE, as GitHub Pages serves the site. Also the test's server.

import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  // The image, gzipped: served as bytes, never with a Content-Encoding (the page inflates it).
  '.gz': 'application/gzip',
};

/** Starts a server for `dir` under `base`; resolves to it once listening. */
export function serve(dir, { port = 8080, base = '/wasmshim/', host = '127.0.0.1' } = {}) {
  const root = resolve(dir);
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://x');
    if (url.pathname === base.replace(/\/$/, '')) {
      res.writeHead(301, { location: base }).end();
      return;
    }
    if (!url.pathname.startsWith(base)) {
      res.writeHead(404).end();
      return;
    }
    let file = join(root, decodeURIComponent(url.pathname.slice(base.length)));
    if (existsSync(file) && statSync(file).isDirectory()) file = join(file, 'index.html');
    if (!(file === root || file.startsWith(root + sep)) || !existsSync(file)) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream', 'content-length': statSync(file).size });
    createReadStream(file).pipe(res);
  });
  return new Promise((r) => server.listen(port, host, () => r(server)));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [dir = 'dist', port = '8080', base = '/wasmshim/'] = process.argv.slice(2);
  const server = await serve(dir, { port: Number(port), base });
  console.log(`serving ${dir} at http://127.0.0.1:${server.address().port}${base}`);
}
