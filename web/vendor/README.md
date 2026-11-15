# Vendored packages

## `codelib-web-0.1.2.tgz`

The browser host kit the page runs on: the pinned jco (1.42.0, its code generator
`@bytecodealliance/jco-transpile` 0.24.0) with its patches to jco's output (`codelib-web
transpile`), the event loop the patched output runs on (`codelib-web/loop`), virtual WASI 0.2
and 0.3 over a `World` (`codelib-web/wasi`, `codelib-web/worlds`), the loader
(`codelib-web/load`), the Worker runner (`codelib-web/worker`) and the list of its browser
modules (`codelib-web browser-files`, which the build copies next to the page). `package.json` installs it
from here (`"codelib-web": "file:vendor/codelib-web-0.1.2.tgz"`), so a clone builds without
reaching any registry for it.

| | |
|---|---|
| Source | <https://github.com/backkem/codelib>, directory `web`, commit `d5672f1ee1d4a5f436a2532c70712c35c5d129df` ("web: 0.1.2, settle() reports idle again and exportNames finds camel-only instances") |
| Built with | `npm ci && npm pack` in that directory |
| SHA-256 | `20ccc00327de63f4e6a6b00e2dd8148ba7316b3859d8447e1d28cf0b5b777661` |
| npm integrity | recorded in `package-lock.json` |
| Licence | MIT |

The kit documents its patches and the jco defects behind them in the package itself
(`node_modules/codelib-web/docs/jco-patches.md` after `npm ci`).

A new kit version gets a new file name: replace the tarball, update `package.json`, regenerate
the lockfile (`npm install`) and update this table.
