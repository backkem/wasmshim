# Checkpoints

A running Linux machine can be saved and restored. `Kernel::checkpoint` writes the whole
machine (filesystem, threads, processes and their objects, memory) into one snapshot
container; `Kernel::restore` builds a new `Kernel` from it that continues exactly where the
saved one stopped.

**Guarantees.** A checkpoint is deterministic: the same state always gives the same bytes,
whatever engine ran the guest up to it (only register state is saved, never interpreter caches
or translations, so a checkpoint taken under one engine restores under another). A checkpoint
either succeeds or fails without changing the machine. Decoding never panics on truncated or
malformed bytes; see the trust model below for what it does not check.

## Three layers

```text
snapshot::Snapshot            container: kind, version, meta, body, content-addressed pages
  body = wasmshim-kernel bytes   header "WNCK" + version, the substrate's state (Core::save)
         + Linux's objects    filesystem, processes, descriptors, sockets, IPC, devices
  pages                       memory pages that differ from what replaying the mappings gives
```

**The container** ([`wasmshim-snapshot`](../../crates/snapshot/src/lib.rs)) is one deterministic CBOR map
([`wasmshim-cbor`](../../crates/det-cbor/src/lib.rs)): `format`, `kind` (`"linux"`), `version` (the
body's version, 4 for Linux), `meta` (text keys, scalar values: what the state is valid for),
`body`, `pages` (`[key, digest]` pairs in increasing key order) and `chunks` (digest to bytes).
A chunk is named by the SHA-256 of its bytes, so equal pages are stored once.
`Snapshot::id` is the SHA-256 of the manifest (everything but the chunks): the same whether the
chunks travel inline (`encode`), are deflated (`encode_deflated`, same identity) or live in a
content-addressed store (`split`, `join`). Decoding is strict: an unknown field, an
unreferenced, missing or corrupt chunk, out-of-order pages, or another kind or version is an
error.

**The substrate's part** ([`guest-kernel/src/checkpoint.rs`](../../crates/guest-kernel/src/checkpoint.rs)):
`Writer`/`Reader` (little-endian, length-prefixed, maps in key order) and the `Ckpt` trait.
`Core::save` writes the scheduler configuration (which a restore must match: same ISA, quantum
and `ns_per_insn`), every task's register state, own retired count, blocker, timed deadline and
personality state, the run queue, id counters, time accounting and pending diagnostics.
`AddressSpace::save` writes the mapping list and stores only the pages a restore would not get
back by replaying the mappings: the personality says what each mapping holds after a replay
(`Baseline::Zero`, `Source` for a private file mapping, `Shared` for an object saved with the
object). `Objects` keeps sharing: each shared object (`Rc`) is saved once and referenced by
index, so `dup`ed descriptors, fork-shared tables and epoll interests stay shared.

**The Linux part** ([`linux-personality/src/checkpoint.rs`](../../crates/linux-personality/src/checkpoint.rs)):
page-cache objects, then the VFS (image files as references into the image, created or
modified files with their bytes), the substrate, processes with their address spaces,
descriptor tables, handlers and timers, then connected socket pairs, SysV shared memory,
inotify watches, file locks, HID devices and framebuffers, and the network provider's record.
Each address space and memory object gets its own key space (`space << 48 | address`) in
`pages`.

## What is saved

Host stdio; VFS files and directories; pipes; `/proc` files (their generated contents);
eventfds, timerfds, epoll sets, signalfds, pidfds, inotify instances and watches; every
address space; private file mappings (replayed from the restored filesystem); shared anonymous
memory, files mapped shared and memfds (including ones unlinked while mapped); SysV shared
memory; native AIO contexts; file locks; netlink sockets; HID devices and framebuffers with
their pending frames; local sockets (`AF_UNIX`, and loopback `AF_INET`/`AF_INET6`) unconnected,
listening or connected, with queued bytes, datagrams and unaccepted connections; and provider
sockets that are idle or listening with nothing queued, together with the provider's own
record (`Network::checkpoint`, `checkpoint_provider`).

## What blocks a checkpoint

`Kernel::checkpoint` fails with a `CkptError` naming the first obstacle, and
`Kernel::checkpoint_blockers` lists all of them with counts:

- pseudo-terminals or an attached console;
- SysV semaphores or message queues, POSIX message queues;
- a provider socket that is connecting, connected, closing, failed or has data queued;
- a local socket with descriptors in flight (`SCM_RIGHTS` not yet received);
- a network provider that cannot checkpoint its sockets;
- a shared mapping of a file outside the filesystem, or more than 2^15 memory objects.

## Restoring

```rust
let snap = Snapshot::decode(&bytes)?;          // or decode_in(format, ...) for another name
let mut vfs = Vfs::new();
vfs.mount_image(image_blob)?;                  // the image the checkpoint was taken over
let mut k = Kernel::restore(cfg, Arch::Aarch64, frontend, vfs, &snap)?;
k.set_network(provider);                       // if it had provider sockets
k.reattach_network()?;                         // hands them back, the provider's record first
```

Every thread gets a fresh CPU from `frontend` with its saved registers. The restored machine's
times (deadlines, busy-until, slice start) are instants on the saved machine's monotonic
clock: the embedder must make `Host::monotonic_ns` continue from the saved instant (in
`wasmshim-linux-node`, the `SHIFT` offset, see [node.md](node.md)). Entropy is not part of the
checkpoint; an embedder that wants the guest's random stream to continue must restore its own
generator. Profile counters start from zero.

## Trust model

Restore checks structure, not meaning. It rejects bad magic, versions, kinds, truncated input,
trailing bytes, duplicate tasks or inodes, overlapping or unaligned mappings, dangling
directory entries, directories whose parents never lead to the root, objects nested more than
32 definitions deep (a machine's own state nests about half that), page counts that do not
match and chunks that do not hash to their names.
It does **not** check:

- that the mounted image is the one the checkpoint was taken over (only that one is mounted
  when the checkpoint had one): put the image's hash into `meta` and compare it yourself;
- referential integrity of process state (pids, a process's cwd or descriptor inodes), so a
  crafted snapshot can make the restored personality panic later.

Maps and sets read through the substrate's `Ckpt` impls, and the substrate's tasks, must
arrive in the order they are saved in (strictly increasing keys): unsorted or repeated entries
are an error. Deflated fields may inflate to 2 GiB each and to 4 GiB in
all.

Treat a snapshot as trusted input: restore only what you produced or verified.
