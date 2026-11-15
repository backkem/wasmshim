# Root filesystem images

A root filesystem is one **uncompressed tar archive** whose first member is an index. The node
reads it through the `linux:host/blob` interface; the personality mounts it as the initial
contents of its in-memory VFS ([personality.md](personality.md)). Images are built from TOML
recipes of pinned upstream artifacts by `wasmshim-image-pack`.

**Guarantees.** An image is still an ordinary tar archive that any `tar` can list. The same
recipe gives the same bytes on every host. Nothing from a downloaded artifact is executed or
extracted onto the host filesystem: archives are only read, in memory.

## The format

```text
offset 0     ustar header  ".linux-image-index"     regular file
             index data    "LXIMGIX2", end, entry count, one record per member
             ...           the archive's members, unchanged
end          two zero blocks
```

The index ([`vfs/index.rs`](../../crates/linux-personality/src/vfs/index.rs), format version 2,
little-endian) holds the magic `LXIMGIX2`, `end: u64` (where the described members end), a
`u32` entry count, then per member: `kind: u8` (file, hard link, symlink, character device,
block device, directory, FIFO), `mode`, `uid`, `gid: u32`, `mtime: i64`, `size`,
`data_offset: u64` (where its contents start in the whole archive), `dev_major`,
`dev_minor: u32`, and the path and link target, each as a `u32` length and bytes.

Mounting (`Vfs::mount_image`) reads the index with one blob read instead of every header, then
scans whatever follows `end` header by header. An archive without an index is scanned
entirely. Concatenated archives (a base image followed by layer archives, each without its end
blocks; later members win) therefore work: only an index at offset 0 is used, and an index
anywhere else, one that does not decode, or one that claims more than the archive holds is
ignored. The index member never appears in the filesystem. A hard link listed before the
member it names (an archive sorted by path) is made once every other member is in place, unless
a later member took its name. The tar reader
([`vfs/tar.rs`](../../crates/linux-personality/src/vfs/tar.rs)) understands ustar, pax (`x`, `g`)
and GNU long names (`L`, `K`); it reads headers only, and file contents stay in the blob,
referenced by offset, until the guest reads them. A member whose data would reach past the end
of the blob is a bad header, and mounting fails.

`wasmshim_linux::vfs::index::with_index` puts an index in front of any tar (rebuilding an
existing one).

## Recipes

A recipe is `images/<name>.toml` ([example](../../images/alpine-busybox.toml)), loaded by
[`recipe.rs`](../../crates/linux-image-pack/src/recipe.rs):

| Key | Meaning |
|---|---|
| `description` | Required text. |
| `include` | Recipes whose entries come first (at most 8 levels deep). |
| `[[artifact]]` | `url` (https only), `sha256`, `kind`; optional `strip` (leading path components), `prefix`, `exclude` (path prefixes), `path` (for `kind = "file"`), `offset` (for `tar-zst`). |
| `[[symlink]]` | `path`, `target`. |
| `[[dir]]` | `path`, optional octal `mode` (default `"755"`). |
| `[[file]]` | `path`, `content`: a small text file, mode 0644. |
| `[[tree]]` | `path` in the image, `source` directory next to the recipes; text only, CRLF becomes LF. |

Artifact kinds: `tar-gz`, `apk` (Alpine package; metadata members dropped), `deb` (its
`data.tar.xz` or `.gz`), `file` (one file installed with mode 0755 at `path`) and `tar-zst`
(a zstd tar starting at `offset`, for a payload embedded in a binary). With the plain packer,
unknown keys and kinds are errors.
Decoding uses pure-Rust gzip, xz and zstd.

Artifacts merge in recipe order and later paths replace earlier ones; then symlinks,
directories, trees and files are added. The output is written in path order with a fixed
mtime (2026-01-01T00:00:00Z) and owner 0:0, so it is byte-for-byte reproducible.

## Building

```sh
cargo xtask image alpine-busybox
```

[`Packer`](../../crates/linux-image-pack/src/lib.rs) downloads each artifact once with the
system `curl` (`--proto =https`) into `.cache/artifacts/<sha256>/<file>`, checks its SHA-256
before every use, and writes `target/images/<name>.tar` with the index in front plus
`target/images/<name>.sha256`. A mismatch is an error naming the cached file.

`wasmshim_image_pack::Extension` lets an embedder add its own recipe keys, artifact kinds and
members at fixed points of a build; `NoExtension` is the plain packer.

## Serving an image to a node

- **A host serves the file.** `wasmshim-run --image <tar>` serves it as the blob `rootfs`
  (`wasmshim-wasmtime::blob::FileBlobs`), which is the node's default `--image` name.
- **Composed in.** [`components/linux-image`](../../components/linux-image/src/lib.rs) compiles
  one image into a component (set `LINUX_IMAGE=<path>` when building; without it, an empty
  archive) exporting `linux:host/blob` and `linux:host/blob-words`. Composed with a node built
  with the `blob-words` feature, the pair is a self-contained WASI 0.3 command.

The node reads the blob through a cache of 32 KiB chunks (256 of them); reads of 128 KiB or
more go to the host directly ([`components/linux-node/src/lib.rs`](../../components/linux-node/src/lib.rs)).
Blobs never change while a node runs.

## Limits

- Files the guest writes live in memory; the image itself is read-only and nothing is written
  back to it.
- The pinned SHA-256, not the transport, is what makes a download trustworthy.
- The packer's tar reader does not verify header checksums, and `[[tree]] source` is not
  confined to the recipes directory: recipes are trusted input.
