//! The contract between translated code and the engine that runs it: the entry signature, the
//! context block, exit codes, the software TLB and the chain table. A front half decides how its
//! guest state is laid out in the engine's memory; this part is the same for every guest ISA.
//!
//! The symbol names keep the `linux_` prefix of the first engine that defined them
//! (`wasmshim-aarch64-engine`); they are names, not a dependency on Linux. Another engine may use other
//! names through [`crate::aot::Linking::prefix`].

/// The module name of every import of a translated module.
pub const IMPORT_MODULE: &str = "engine";
/// The engine's linear memory (register file, TLB, guest pages).
pub const IMPORT_MEMORY: &str = "memory";
/// `linux_jit_step: (pc: i64) -> i32`. Sets the CPU's `pc` and executes that one instruction in
/// the engine's interpreter (the registers and flags it uses are in memory when it is
/// called). `0`: it completed and nothing else happened; any other value is returned
/// unchanged by the translated function (see [`STEP_STATUS_MIN`]).
pub const IMPORT_STEP: &str = "linux_jit_step";
/// `jit_fill: (addr: i64, write: i32, size: i32) -> i32`: makes the page of `addr`
/// accessible through the TLB (read, or `write`) and returns the host address of `addr`,
/// or 0 if the access would fault, cross a page, or (for a write) reach code. It runs no
/// instruction; translated code then performs the access itself (or, on 0, `jit_step`s it).
pub const IMPORT_FILL: &str = "linux_jit_fill";
/// The exported entry of a region module:
/// `f0: (cpu: i32, tlb: i32, ctx: i32) -> i32` (an exit code).
pub const EXPORT_ENTRY: &str = "f0";

/// `ctx + 0`: the remaining instruction budget (`i64`, in and out).
pub const CTX_LEFT: u64 = 0;
/// `ctx + 8`: the index of the block to start at (`i32`, in).
pub const CTX_ENTRY: u64 = 8;
/// `ctx + 16`: the runtime address of the region's page (`i64`, in). Translated code is
/// position-independent: every PC it computes is relative to this.
pub const CTX_BASE: u64 = 16;
/// `ctx + 24`: the address of the chain table (`i32`, in): see [`CHAIN_ENTRIES`].
pub const CTX_CHAIN: u64 = 24;
/// `ctx + 32`: the code epoch the chain table's entries must carry to be used (`i64`, in).
pub const CTX_EPOCH: u64 = 32;
/// Size of the context block.
pub const CTX_SIZE: usize = 40;

/// Sets of the chain table, indexed by `(pc >> 2) & (CHAIN_ENTRIES - 1)`; each set holds
/// [`CHAIN_WAYS`] consecutive entries (the set's address is the table's plus
/// `index * CHAIN_WAYS * CHAIN_ENTRY_SIZE`). An entry is `pc: u64`, `slot: u32` (function
/// table index), `block: u32`, `epoch: u64`. On an [`EXIT_JUMP`] whose target has an entry
/// with the current epoch in its set, translated code tail-calls the target
/// (`return_call_indirect`) instead of returning: the engine fills entries for translations it
/// validated (the most recent first in its set) and clears those it drops.
pub const CHAIN_ENTRIES: u64 = 8192;
/// Entries per chain set (two-way, so that two hot targets 32 KiB apart no
/// longer evict each other).
pub const CHAIN_WAYS: u64 = 2;
/// Bytes per chain entry.
pub const CHAIN_ENTRY_SIZE: u64 = 24;
/// The table translated code tail-calls through (the module's function table).
pub const IMPORT_TABLE: &str = "__indirect_function_table";

/// Continue at the CPU's `pc` (a branch left the region, or an indirect branch).
pub const EXIT_JUMP: i32 = 1;
/// The budget does not cover the next block: the CPU's `pc` is that block's first
/// instruction and nothing of it ran. The engine finishes the budget in the interpreter.
pub const EXIT_BUDGET: i32 = 2;
/// An `SVC` retired; the CPU's `pc` is the next instruction.
pub const EXIT_SYSCALL: i32 = 3;
/// A memory instruction translated code does not do inline (a TLB miss, a page-crossing or
/// misaligned access, a store into code): nothing of it ran, the CPU's `pc` is that
/// instruction and the budget of it and the rest of its block is given back. The engine
/// executes it in the interpreter, filling the TLB, and goes on (a side exit keeps
/// the slow path's call out of hot loops; regions built without `Options::mem_exits` call
/// `jit_fill` and `jit_step` instead).
pub const EXIT_MEM: i32 = 4;
/// `jit_step` statuses at or above this are returned as exit codes. The translated code
/// treats the stepped instruction as **not** retired (it gives back the budget of it and
/// of the rest of its block); an engine status meaning "it did retire" (e.g. "it wrote
/// into code") must take one more instruction off the budget itself.
pub const STEP_STATUS_MIN: i32 = 16;

/// Entries per TLB (direct mapped by virtual page number).
pub const TLB_ENTRIES: u64 = 2048;
/// Bytes per TLB entry: `tag: u64` (the page's virtual address, or `u64::MAX` when
/// empty), then `addend: u32` (host address of the page minus the low 32 bits of the
/// page's virtual address, wrapping), then 4 bytes of padding.
pub const TLB_ENTRY_SIZE: u64 = 16;
/// Offset of the read TLB from the `tlb` argument (pages that may be read).
pub const TLB_READ: u64 = 0;
/// Offset of the write TLB from the `tlb` argument (pages that may be written: never
/// executable ones, so stores into code always reach the interpreter).
pub const TLB_WRITE: u64 = TLB_ENTRIES * TLB_ENTRY_SIZE;
/// Total bytes of the two TLBs.
pub const TLB_SIZE: usize = (2 * TLB_ENTRIES * TLB_ENTRY_SIZE) as usize;
/// Offset of the addend in an entry.
pub const TLB_ADDEND: u64 = 8;
