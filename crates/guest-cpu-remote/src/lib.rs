//! A [`GuestCpu`] and [`GuestMemory`] whose state lives in an *execution engine*: a core Wasm
//! module that a host instantiates on its own memory.
//!
//! Three layers, each usable without the others:
//!
//! * [`abi`]: the **engine core ABI** between any host and any engine module. The host knows
//!   only this: how to pass a call (a service number, up to 16 `u64` arguments and an input
//!   byte string) and get its reply (up to 16 `u64` values and an output byte string), and how
//!   to serve the engine's requests to compile translated code into its function table. It
//!   never learns the guest ISA or what the services mean.
//! * [`service`]: the services of the A64 engine (`wasmshim-aarch64-engine`): address spaces,
//!   CPUs, `run`. Both the engine and this crate's client implement them.
//! * [`RemoteCpu`] / [`RemoteMemory`]: the client, over any [`Link`] (a WIT import inside a
//!   component, or direct wasmtime calls in a native embedder). One `run_until_event` is one
//!   call; the syscall frame comes back by value and the result goes out with the next run.

#![no_std]
#![deny(unsafe_code)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::any::Any;
use core::cell::RefCell;

use wasmshim_cpu::{
    A64Context, Access, CpuContext, Event, Fault, GuestCpu, GuestMemory, MapError, MemFault,
    MemFaultKind, PAGE_SIZE, PageSource, PageSourceError, Perms, SharedMemory, SyscallFrame,
};

/// The engine core ABI (host ↔ engine module). See the crate documentation.
///
/// An engine module has **no imports**. It exports:
/// * `memory` (memory 0, 32-bit): its own and the guest's memory, shared with translated code;
/// * `EXPORT_TABLE`: a growable `funcref` table where the host installs translated code;
/// * `EXPORT_IO` `() -> i32`: the address of its `Io` block (fixed for its lifetime);
/// * `EXPORT_INPUT` `(len: i32) -> i32`: the address of a buffer of at least `len` bytes for
///   the next call's input;
/// * `EXPORT_CALL` `() -> i32`: performs the call described by the `Io` block and returns
///   `STATUS_DONE` (the reply is in the block) or `STATUS_COMPILE` (see below).
///
/// **Compile requests.** On `STATUS_COMPILE`, `req_ptr`/`req_len` hold a core module that
/// may import only functions and the memory that the engine exports (module name `engine`),
/// and exports `req_funcs` functions named `f0`, `f1`, …. The host compiles and instantiates it
/// against the engine instance, appends the functions to the table, writes the first index to
/// `req_slot` (or `SLOT_DECLINED`) and calls `EXPORT_CALL` again to continue the same
/// call. The engine decides what to translate and when; the host decides whether to compile.
/// Neither choice may change what the guest observes.
pub mod abi {
    use core::mem::offset_of;

    /// The engine's memory export.
    pub const EXPORT_MEMORY: &str = "memory";
    /// The engine's function table export.
    pub const EXPORT_TABLE: &str = "__indirect_function_table";
    /// `() -> i32`: address of the [`Io`] block.
    pub const EXPORT_IO: &str = "jit_io";
    /// `(len: i32) -> i32`: address of an input buffer.
    pub const EXPORT_INPUT: &str = "jit_input";
    /// `() -> i32`: run (or continue) the call.
    pub const EXPORT_CALL: &str = "jit_call";
    /// Translated modules import from this module name.
    pub const IMPORT_MODULE: &str = "engine";

    /// The call completed.
    pub const STATUS_DONE: i32 = 0;
    /// The engine asks for a module to be compiled before it can continue.
    pub const STATUS_COMPILE: i32 = 1;
    /// `req_slot` value: the host did not install the module.
    pub const SLOT_DECLINED: u32 = u32::MAX;

    /// Maximum arguments and reply values per call.
    pub const MAX_WORDS: usize = 16;

    /// The call block shared by host and engine (little-endian, `#[repr(C)]`, the same
    /// layout on every target).
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Io {
        /// In: the service number.
        pub service: u32,
        /// In: number of valid `args`.
        pub nargs: u32,
        /// In: bytes of input written at `jit_input(len)`.
        pub input_len: u32,
        /// Out: number of valid `values`.
        pub nvalues: u32,
        /// Out: the output bytes (valid until the next call).
        pub output_ptr: u32,
        pub output_len: u32,
        /// Compile request: the module bytes and the number of exported functions.
        pub req_ptr: u32,
        pub req_len: u32,
        pub req_funcs: u32,
        /// Compile reply: first table index of `f0`, or [`SLOT_DECLINED`].
        pub req_slot: u32,
        pub args: [u64; MAX_WORDS],
        pub values: [u64; MAX_WORDS],
    }

    pub const IO_SERVICE: usize = offset_of!(Io, service);
    pub const IO_NARGS: usize = offset_of!(Io, nargs);
    pub const IO_INPUT_LEN: usize = offset_of!(Io, input_len);
    pub const IO_NVALUES: usize = offset_of!(Io, nvalues);
    pub const IO_OUTPUT_PTR: usize = offset_of!(Io, output_ptr);
    pub const IO_OUTPUT_LEN: usize = offset_of!(Io, output_len);
    pub const IO_REQ_PTR: usize = offset_of!(Io, req_ptr);
    pub const IO_REQ_LEN: usize = offset_of!(Io, req_len);
    pub const IO_REQ_FUNCS: usize = offset_of!(Io, req_funcs);
    pub const IO_REQ_SLOT: usize = offset_of!(Io, req_slot);
    pub const IO_ARGS: usize = offset_of!(Io, args);
    pub const IO_VALUES: usize = offset_of!(Io, values);
    pub const IO_SIZE: usize = core::mem::size_of::<Io>();

    const _: () = assert!(IO_ARGS == 40 && IO_VALUES == 168 && IO_SIZE == 296);
}

/// Services of the A64 engine. Arguments and values are `u64`s; "fault" values are
/// `[0]` (ok) or `[1, addr, access, kind]` ([`encode_fault`]).
pub mod service {
    /// `() -> [mem]`
    pub const NEW_MEM: u32 = 1;
    /// `(mem) -> [mem]`: a copy-on-write copy.
    pub const DUP_MEM: u32 = 2;
    /// `(mem)`
    pub const FREE_MEM: u32 = 3;
    /// `() -> [cpu]`
    pub const NEW_CPU: u32 = 4;
    /// `(cpu, PENDING...) -> [cpu]`: applies the pending writes, then copies.
    pub const DUP_CPU: u32 = 5;
    /// `(cpu)`
    pub const FREE_CPU: u32 = 6;
    /// `(mem, addr, len, perms) -> [map error]`
    pub const MAP: u32 = 7;
    /// `(mem, addr, len) -> [map error]`
    pub const UNMAP: u32 = 8;
    /// `(mem, addr, len, perms) -> [map error]`
    pub const PROTECT: u32 = 9;
    /// `(mem, addr, len) -> fault`, output: the bytes.
    pub const READ: u32 = 10;
    /// `(mem, addr)`, input: the bytes `-> fault`
    pub const WRITE: u32 = 11;
    /// `(mem, addr)`, input: the bytes `-> fault` (ignores permissions)
    pub const POKE: u32 = 12;
    /// `(mem, addr, len, access) -> fault`
    pub const CHECK: u32 = 13;
    /// `(mem, addr) -> [perms bits, or 255 if unmapped]`
    pub const PERMS: u32 = 14;
    /// `(mem, addr, max) -> fault ++ [found]`, output: the string without the NUL.
    pub const READ_CSTR: u32 = 15;
    /// `(mem, addr, len) -> fault`
    pub const ZERO: u32 = 16;
    /// `(mem) -> [code epoch]`
    pub const CODE_EPOCH: u32 = 17;
    /// `(mem, addr) -> [code version]`
    pub const CODE_VERSION: u32 = 18;
    /// `(mem, addr) -> fault ++ [word]`
    pub const FETCH: u32 = 19;
    /// `(cpu, mem, budget, virtual time, PENDING...) -> [kind, retired, pc, sp, tp, payload...]`
    /// (see [`super::run_kind`]).
    pub const RUN: u32 = 20;
    /// `(cpu, PENDING...)`, output: the context ([`super::encode_context`]).
    pub const GET_CONTEXT: u32 = 21;
    /// `(cpu)`, input: the context.
    pub const SET_CONTEXT: u32 = 22;
    /// `() -> [counter...]`: engine statistics (never guest-visible).
    pub const STATS: u32 = 23;
    /// `(key, value)`: engine tuning (never guest-visible). Keys in [`super::config`].
    pub const CONFIG: u32 = 24;
    /// `()`, output: text, the instructions translated code handed to the interpreter, by
    /// shape and count (engines built with step statistics; diagnostics only).
    pub const STEP_SHAPES: u32 = 25;
    /// `() -> id`: a new memory object ([`wasmshim_cpu::SharedPages`] in the engine).
    pub const SHARED_NEW: u32 = 26;
    /// `(id)`: drops the client's handle (mappings keep the object alive).
    pub const SHARED_FREE: u32 = 27;
    /// `(id, offset, len) -> status; output`: reads object bytes (status 0: ok).
    pub const SHARED_READ: u32 = 28;
    /// `(id, offset; input) -> status`: writes object bytes.
    pub const SHARED_WRITE: u32 = 29;
    /// `(id, offset) -> status`: zeroes the object from `offset` on.
    pub const SHARED_ZERO_FROM: u32 = 30;
    /// `(mem, addr, len, perms, id, first) -> map error code`: [`wasmshim_cpu::GuestMemory::map_shared`].
    pub const MAP_SHARED: u32 = 31;

    /// The pending register writes that ride along with `RUN`, `DUP_CPU` and `GET_CONTEXT`:
    /// `mask, x0, pc, sp, tp` (bits of `mask`: 1 = x0, 2 = pc, 4 = sp, 8 = tp).
    pub const PENDING_WORDS: usize = 5;
}

/// [`service::CONFIG`] keys.
pub mod config {
    /// Executions of a block before its region is translated (`0`: never translate).
    pub const THRESHOLD: u64 = 1;
    /// Maximum blocks per region.
    pub const REGION_BLOCKS: u64 = 2;
}

/// `RUN` reply kinds.
pub mod run_kind {
    /// The budget ran out.
    pub const YIELD: u64 = 0;
    /// Payload: number, `args[6]`.
    pub const SYSCALL: u64 = 1;
    /// Payload: fault pc, addr, access, kind.
    pub const FAULT_MEMORY: u64 = 2;
    /// Payload: fault pc, insn.
    pub const FAULT_UNDEFINED: u64 = 3;
    /// Payload: fault pc, imm.
    pub const FAULT_BREAKPOINT: u64 = 4;
    /// Payload: fault pc, addr.
    pub const FAULT_ALIGNMENT: u64 = 5;
}

/// Encodes an [`Event`] as `RUN` reply words (after `kind`, `retired`, `pc`, `sp`, `tp`).
pub fn encode_event(event: &Event, out: &mut Vec<u64>) -> u64 {
    match *event {
        Event::Yield => run_kind::YIELD,
        Event::Syscall(f) => {
            out.push(f.number);
            out.extend_from_slice(&f.args);
            run_kind::SYSCALL
        }
        Event::Fault(Fault::Memory { pc, fault }) => {
            out.extend_from_slice(&[
                pc,
                fault.addr,
                access_code(fault.access),
                kind_code(fault.kind),
            ]);
            run_kind::FAULT_MEMORY
        }
        Event::Fault(Fault::Undefined { pc, insn }) => {
            out.extend_from_slice(&[pc, u64::from(insn)]);
            run_kind::FAULT_UNDEFINED
        }
        Event::Fault(Fault::Breakpoint { pc, imm }) => {
            out.extend_from_slice(&[pc, u64::from(imm)]);
            run_kind::FAULT_BREAKPOINT
        }
        Event::Fault(Fault::Alignment { pc, addr }) => {
            out.extend_from_slice(&[pc, addr]);
            run_kind::FAULT_ALIGNMENT
        }
    }
}

/// Decodes [`encode_event`]'s words (`pc` is the CPU's PC after the run).
pub fn decode_event(kind: u64, pc: u64, p: &[u64]) -> Event {
    let at = |i: usize| p.get(i).copied().unwrap_or(0);
    match kind {
        run_kind::SYSCALL => Event::Syscall(SyscallFrame {
            number: at(0),
            pc,
            args: [at(1), at(2), at(3), at(4), at(5), at(6)],
        }),
        run_kind::FAULT_MEMORY => Event::Fault(Fault::Memory {
            pc: at(0),
            fault: MemFault {
                addr: at(1),
                access: access_of(at(2)),
                kind: kind_of(at(3)),
            },
        }),
        run_kind::FAULT_UNDEFINED => Event::Fault(Fault::Undefined {
            pc: at(0),
            insn: at(1) as u32,
        }),
        run_kind::FAULT_BREAKPOINT => Event::Fault(Fault::Breakpoint {
            pc: at(0),
            imm: at(1) as u16,
        }),
        run_kind::FAULT_ALIGNMENT => Event::Fault(Fault::Alignment {
            pc: at(0),
            addr: at(1),
        }),
        _ => Event::Yield,
    }
}

pub fn access_code(a: Access) -> u64 {
    match a {
        Access::Read => 0,
        Access::Write => 1,
        Access::Execute => 2,
    }
}

pub fn access_of(v: u64) -> Access {
    match v {
        1 => Access::Write,
        2 => Access::Execute,
        _ => Access::Read,
    }
}

fn kind_code(k: MemFaultKind) -> u64 {
    match k {
        MemFaultKind::Unmapped => 0,
        MemFaultKind::Protection => 1,
    }
}

fn kind_of(v: u64) -> MemFaultKind {
    if v == 1 {
        MemFaultKind::Protection
    } else {
        MemFaultKind::Unmapped
    }
}

/// `[0]` or `[1, addr, access, kind]`.
pub fn encode_fault(r: Result<(), MemFault>, out: &mut Vec<u64>) {
    match r {
        Ok(()) => out.push(0),
        Err(f) => out.extend_from_slice(&[1, f.addr, access_code(f.access), kind_code(f.kind)]),
    }
}

/// The inverse of [`encode_fault`]; returns the result and the number of words used.
pub fn decode_fault(v: &[u64]) -> (Result<(), MemFault>, usize) {
    if v.first().copied().unwrap_or(1) == 0 {
        return (Ok(()), 1);
    }
    let at = |i: usize| v.get(i).copied().unwrap_or(0);
    (
        Err(MemFault {
            addr: at(1),
            access: access_of(at(2)),
            kind: kind_of(at(3)),
        }),
        4,
    )
}

pub fn map_error_code(r: Result<(), MapError>) -> u64 {
    match r {
        Ok(()) => 0,
        Err(MapError::Unaligned) => 1,
        Err(MapError::OutOfRange) => 2,
        Err(MapError::NotMapped) => 3,
        Err(MapError::Unsupported) => 4,
    }
}

pub fn map_error_of(v: u64) -> Result<(), MapError> {
    match v {
        0 => Ok(()),
        1 => Err(MapError::Unaligned),
        2 => Err(MapError::OutOfRange),
        4 => Err(MapError::Unsupported),
        _ => Err(MapError::NotMapped),
    }
}

pub fn perms_code(p: Perms) -> u64 {
    u64::from(p.read()) | u64::from(p.write()) << 1 | u64::from(p.exec()) << 2
}

pub fn perms_of(v: u64) -> Perms {
    Perms::new(v & 1 != 0, v & 2 != 0, v & 4 != 0)
}

/// Bytes of an encoded [`CpuContext`].
pub const CONTEXT_BYTES: usize = 8 * (31 + 6) + 16 * 32;

/// Serializes a context: x0–x30, sp, pc, flags, fpsr, fpcr, tp (u64 LE), then v0–v31 (u128 LE).
pub fn encode_context(c: &CpuContext, out: &mut Vec<u8>) {
    let CpuContext::A64(c) = c;
    for x in c.x {
        out.extend_from_slice(&x.to_le_bytes());
    }
    for v in [c.sp, c.pc, c.flags, c.fpsr, c.fpcr, c.tp] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in c.v {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// The inverse of [`encode_context`] (missing bytes read as zero).
pub fn decode_context(b: &[u8]) -> CpuContext {
    let u64_at = |i: usize| {
        let mut w = [0u8; 8];
        if let Some(s) = b.get(i..i + 8) {
            w.copy_from_slice(s);
        }
        u64::from_le_bytes(w)
    };
    let mut c = A64Context::default();
    for (i, x) in c.x.iter_mut().enumerate() {
        *x = u64_at(8 * i);
    }
    let o = 8 * 31;
    c.sp = u64_at(o);
    c.pc = u64_at(o + 8);
    c.flags = u64_at(o + 16);
    c.fpsr = u64_at(o + 24);
    c.fpcr = u64_at(o + 32);
    c.tp = u64_at(o + 40);
    let o = 8 * 37;
    for (i, v) in c.v.iter_mut().enumerate() {
        let mut w = [0u8; 16];
        if let Some(s) = b.get(o + 16 * i..o + 16 * i + 16) {
            w.copy_from_slice(s);
        }
        *v = u128::from_le_bytes(w);
    }
    CpuContext::A64(c)
}

/// A reply: values and output bytes. Reused across calls to avoid allocations.
#[derive(Clone, Debug, Default)]
pub struct Reply {
    pub values: Vec<u64>,
    pub output: Vec<u8>,
}

/// How a client reaches an engine: one call per [`abi`] call.
pub trait Link {
    /// Performs `service(args, input)` and fills `reply`. Panics (traps) if the engine is
    /// gone or broken: a CPU backend has no way to report that to its caller, and a broken
    /// engine is as fatal as a broken interpreter.
    fn call(&mut self, service: u32, args: &[u64], input: &[u8], reply: &mut Reply);
}

/// The shared connection: link plus a reusable reply.
pub struct Conn<L> {
    link: L,
    reply: Reply,
}

/// A handle to an engine, shared by every CPU and address space it hosts.
pub struct Remote<L: Link> {
    conn: Rc<RefCell<Conn<L>>>,
}

impl<L: Link> Clone for Remote<L> {
    fn clone(&self) -> Self {
        Self {
            conn: self.conn.clone(),
        }
    }
}

impl<L: Link + 'static> Remote<L> {
    /// Wraps a link to a freshly created engine.
    pub fn new(link: L) -> Self {
        Self {
            conn: Rc::new(RefCell::new(Conn {
                link,
                reply: Reply::default(),
            })),
        }
    }

    /// Calls a service; `f` reads the reply.
    pub fn call<R>(
        &self,
        service: u32,
        args: &[u64],
        input: &[u8],
        f: impl FnOnce(&Reply) -> R,
    ) -> R {
        let mut c = self.conn.borrow_mut();
        let Conn { link, reply } = &mut *c;
        link.call(service, args, input, reply);
        f(reply)
    }

    fn value(&self, service: u32, args: &[u64]) -> u64 {
        self.call(service, args, &[], |r| {
            r.values.first().copied().unwrap_or(0)
        })
    }

    /// A fresh address space.
    pub fn new_memory(&self) -> RemoteMemory<L> {
        RemoteMemory {
            remote: self.clone(),
            id: self.value(service::NEW_MEM, &[]),
        }
    }

    /// A fresh CPU with every register zero.
    /// A new memory object in the engine, holding `source`'s first `pages` pages (copied in
    /// now: the engine cannot call back for them).
    pub fn new_shared(
        &self,
        source: Option<(Rc<dyn PageSource>, u64)>,
    ) -> Result<RemoteShared<L>, PageSourceError> {
        let object = RemoteShared {
            remote: self.clone(),
            id: self.value(service::SHARED_NEW, &[]),
        };
        if let Some((source, pages)) = source {
            for index in 0..pages {
                let page = source.page(index)?;
                if page.iter().any(|&b| b != 0) {
                    object.write(index * PAGE_SIZE, &page[..])?;
                }
            }
        }
        Ok(object)
    }

    pub fn new_cpu(&self) -> RemoteCpu<L> {
        RemoteCpu {
            remote: self.clone(),
            id: self.value(service::NEW_CPU, &[]),
            pending: Pending::default(),
            cached: Cached::default(),
            vtime: 0,
        }
    }

    /// Runs `f` on the link (e.g. to read host-side statistics).
    pub fn with_link<R>(&self, f: impl FnOnce(&mut L) -> R) -> R {
        f(&mut self.conn.borrow_mut().link)
    }

    /// Engine tuning ([`config`]); never guest-visible.
    pub fn configure(&self, key: u64, value: u64) {
        self.call(service::CONFIG, &[key, value], &[], |_| ());
    }

    /// Engine statistics (implementation-defined counters).
    pub fn stats(&self) -> Vec<u64> {
        self.call(service::STATS, &[], &[], |r| r.values.clone())
    }
}

/// An address space inside the engine.
pub struct RemoteMemory<L: Link> {
    remote: Remote<L>,
    id: u64,
}

impl<L: Link> RemoteMemory<L> {
    /// The engine-side handle.
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl<L: Link> Drop for RemoteMemory<L> {
    fn drop(&mut self) {
        let mut c = self.remote.conn.borrow_mut();
        let Conn { link, reply } = &mut *c;
        link.call(service::FREE_MEM, &[self.id], &[], reply);
    }
}

impl<L: Link + 'static> RemoteMemory<L> {
    fn fault_call(&self, service: u32, args: &[u64], input: &[u8]) -> Result<(), MemFault> {
        self.remote
            .call(service, args, input, |r| decode_fault(&r.values).0)
    }

    fn map_call(&self, service: u32, args: &[u64]) -> Result<(), MapError> {
        map_error_of(self.remote.value(service, args))
    }
}

impl<L: Link + 'static> GuestMemory for RemoteMemory<L> {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        self.remote.call(
            service::READ,
            &[self.id, addr, buf.len() as u64],
            &[],
            |r| {
                let (res, _) = decode_fault(&r.values);
                if res.is_ok() {
                    let n = buf.len().min(r.output.len());
                    buf[..n].copy_from_slice(&r.output[..n]);
                }
                res
            },
        )
    }

    fn write(&mut self, addr: u64, data: &[u8]) -> Result<(), MemFault> {
        self.fault_call(service::WRITE, &[self.id, addr], data)
    }

    fn fetch(&mut self, addr: u64) -> Result<u32, MemFault> {
        self.remote
            .call(service::FETCH, &[self.id, addr], &[], |r| {
                let (res, n) = decode_fault(&r.values);
                res.map(|()| r.values.get(n).copied().unwrap_or(0) as u32)
            })
    }

    fn check(&self, addr: u64, len: u64, access: Access) -> Result<(), MemFault> {
        self.fault_call(
            service::CHECK,
            &[self.id, addr, len, access_code(access)],
            &[],
        )
    }

    fn poke(&mut self, addr: u64, data: &[u8]) -> Result<(), MemFault> {
        self.fault_call(service::POKE, &[self.id, addr], data)
    }

    fn map(&mut self, addr: u64, len: u64, perms: Perms) -> Result<(), MapError> {
        self.map_call(service::MAP, &[self.id, addr, len, perms_code(perms)])
    }

    fn map_shared(
        &mut self,
        addr: u64,
        len: u64,
        perms: Perms,
        object: Rc<dyn SharedMemory>,
        first: u64,
    ) -> Result<(), MapError> {
        let object = object
            .into_any()
            .downcast::<RemoteShared<L>>()
            .map_err(|_| MapError::Unsupported)?;
        self.map_call(
            service::MAP_SHARED,
            &[self.id, addr, len, perms_code(perms), object.id, first],
        )
    }

    fn unmap(&mut self, addr: u64, len: u64) -> Result<(), MapError> {
        self.map_call(service::UNMAP, &[self.id, addr, len])
    }

    fn protect(&mut self, addr: u64, len: u64, perms: Perms) -> Result<(), MapError> {
        self.map_call(service::PROTECT, &[self.id, addr, len, perms_code(perms)])
    }

    fn perms(&self, addr: u64) -> Option<Perms> {
        match self.remote.value(service::PERMS, &[self.id, addr]) {
            255 => None,
            v => Some(perms_of(v)),
        }
    }

    fn address_limit(&self) -> u64 {
        1 << 48
    }

    fn code_epoch(&self) -> u64 {
        self.remote.value(service::CODE_EPOCH, &[self.id])
    }

    fn code_version(&self, addr: u64) -> u64 {
        self.remote.value(service::CODE_VERSION, &[self.id, addr])
    }

    fn duplicate(&self) -> Box<dyn GuestMemory> {
        Box::new(RemoteMemory {
            remote: self.remote.clone(),
            id: self.remote.value(service::DUP_MEM, &[self.id]),
        })
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn read_cstr(&self, addr: u64, max: usize) -> Result<Option<Vec<u8>>, MemFault> {
        self.remote
            .call(service::READ_CSTR, &[self.id, addr, max as u64], &[], |r| {
                let (res, n) = decode_fault(&r.values);
                res.map(|()| (r.values.get(n).copied() == Some(1)).then(|| r.output.clone()))
            })
    }

    fn zero(&mut self, addr: u64, len: u64) -> Result<(), MemFault> {
        self.fault_call(service::ZERO, &[self.id, addr, len], &[])
    }
}

/// Register writes not yet sent to the engine.
#[derive(Clone, Copy, Default)]
struct Pending {
    mask: u64,
    x0: u64,
    pc: u64,
    sp: u64,
    tp: u64,
}

impl Pending {
    fn words(&self) -> [u64; service::PENDING_WORDS] {
        [self.mask, self.x0, self.pc, self.sp, self.tp]
    }
}

/// Engine state known without a call (valid after any call that returned it).
#[derive(Clone, Copy, Default)]
struct Cached {
    pc: u64,
    sp: u64,
    tp: u64,
    retired: u64,
}

/// A CPU inside the engine.
pub struct RemoteCpu<L: Link> {
    remote: Remote<L>,
    id: u64,
    pending: Pending,
    cached: Cached,
    vtime: u64,
}

impl<L: Link> Drop for RemoteCpu<L> {
    fn drop(&mut self) {
        let mut c = self.remote.conn.borrow_mut();
        let Conn { link, reply } = &mut *c;
        link.call(service::FREE_CPU, &[self.id], &[], reply);
    }
}

impl<L: Link + 'static> RemoteCpu<L> {
    /// The engine-side handle.
    pub fn id(&self) -> u64 {
        self.id
    }

    fn take_pending(&mut self) -> [u64; service::PENDING_WORDS] {
        let w = self.pending.words();
        if self.pending.mask & 2 != 0 {
            self.cached.pc = self.pending.pc;
        }
        if self.pending.mask & 4 != 0 {
            self.cached.sp = self.pending.sp;
        }
        if self.pending.mask & 8 != 0 {
            self.cached.tp = self.pending.tp;
        }
        self.pending = Pending::default();
        w
    }
}

impl<L: Link + 'static> GuestCpu for RemoteCpu<L> {
    fn run_until_event(&mut self, mem: &mut dyn GuestMemory, budget: u64) -> Event {
        let mem_id = mem
            .as_any()
            .and_then(|a| a.downcast_ref::<RemoteMemory<L>>())
            .map(RemoteMemory::id)
            .expect("a remote CPU runs only against an address space of its own engine");
        let p = self.take_pending();
        let args = [
            self.id, mem_id, budget, self.vtime, p[0], p[1], p[2], p[3], p[4],
        ];
        let (event, cached) = self.remote.call(service::RUN, &args, &[], |r| {
            let v = &r.values;
            let at = |i: usize| v.get(i).copied().unwrap_or(0);
            let cached = Cached {
                retired: at(1),
                pc: at(2),
                sp: at(3),
                tp: at(4),
            };
            (
                decode_event(at(0), cached.pc, v.get(5..).unwrap_or(&[])),
                cached,
            )
        });
        self.cached = cached;
        event
    }

    fn set_syscall_result(&mut self, value: u64) {
        self.pending.mask |= 1;
        self.pending.x0 = value;
    }

    fn pc(&self) -> u64 {
        if self.pending.mask & 2 != 0 {
            self.pending.pc
        } else {
            self.cached.pc
        }
    }

    fn set_pc(&mut self, pc: u64) {
        self.pending.mask |= 2;
        self.pending.pc = pc;
    }

    fn sp(&self) -> u64 {
        if self.pending.mask & 4 != 0 {
            self.pending.sp
        } else {
            self.cached.sp
        }
    }

    fn set_sp(&mut self, sp: u64) {
        self.pending.mask |= 4;
        self.pending.sp = sp;
    }

    fn thread_pointer(&self) -> u64 {
        if self.pending.mask & 8 != 0 {
            self.pending.tp
        } else {
            self.cached.tp
        }
    }

    fn set_thread_pointer(&mut self, tp: u64) {
        self.pending.mask |= 8;
        self.pending.tp = tp;
    }

    fn retired(&self) -> u64 {
        self.cached.retired
    }

    fn set_virtual_time(&mut self, ns: u64) {
        self.vtime = ns;
    }

    fn context(&self) -> CpuContext {
        let p = self.pending.words();
        let args = [self.id, p[0], p[1], p[2], p[3], p[4]];
        self.remote.call(service::GET_CONTEXT, &args, &[], |r| {
            decode_context(&r.output)
        })
    }

    fn set_context(&mut self, ctx: &CpuContext) {
        self.pending = Pending::default();
        let mut bytes = Vec::with_capacity(CONTEXT_BYTES);
        encode_context(ctx, &mut bytes);
        self.remote
            .call(service::SET_CONTEXT, &[self.id], &bytes, |_| ());
        // The engine masks flags and FP control like the interpreter; read back what stuck.
        self.cached.pc = ctx.pc();
        self.cached.sp = ctx.sp();
        self.cached.tp = ctx.thread_pointer();
    }

    fn duplicate(&self) -> Box<dyn GuestCpu> {
        let p = self.pending.words();
        let args = [self.id, p[0], p[1], p[2], p[3], p[4]];
        let id = self.remote.value(service::DUP_CPU, &args);
        let mut cached = self.cached;
        if p[0] & 2 != 0 {
            cached.pc = p[2];
        }
        if p[0] & 4 != 0 {
            cached.sp = p[3];
        }
        if p[0] & 8 != 0 {
            cached.tp = p[4];
        }
        Box::new(RemoteCpu {
            remote: self.remote.clone(),
            id,
            pending: Pending::default(),
            cached,
            vtime: self.vtime,
        })
    }
}

/// A memory object in the engine ([`Remote::new_shared`]), mappable by its [`RemoteMemory`]s.
pub struct RemoteShared<L: Link> {
    remote: Remote<L>,
    id: u64,
}

impl<L: Link> Drop for RemoteShared<L> {
    fn drop(&mut self) {
        let mut c = self.remote.conn.borrow_mut();
        let Conn { link, reply } = &mut *c;
        link.call(service::SHARED_FREE, &[self.id], &[], reply);
    }
}

impl<L: Link + 'static> SharedMemory for RemoteShared<L> {
    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), PageSourceError> {
        self.remote.call(
            service::SHARED_READ,
            &[self.id, offset, buf.len() as u64],
            &[],
            |r| {
                if r.values.first() != Some(&0) || r.output.len() < buf.len() {
                    return Err(PageSourceError);
                }
                buf.copy_from_slice(&r.output[..buf.len()]);
                Ok(())
            },
        )
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<(), PageSourceError> {
        self.remote.call(
            service::SHARED_WRITE,
            &[self.id, offset],
            data,
            |r| match r.values.first() {
                Some(0) => Ok(()),
                _ => Err(PageSourceError),
            },
        )
    }

    fn zero_from(&self, offset: u64) -> Result<(), PageSourceError> {
        match self
            .remote
            .value(service::SHARED_ZERO_FROM, &[self.id, offset])
        {
            0 => Ok(()),
            _ => Err(PageSourceError),
        }
    }

    fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_round_trips() {
        let mut c = A64Context::default();
        for (i, x) in c.x.iter_mut().enumerate() {
            *x = 0x1111 * i as u64;
        }
        c.sp = 1;
        c.pc = 2;
        c.flags = 0xf000_0000;
        c.fpsr = 3;
        c.fpcr = 4;
        c.tp = 5;
        for (i, v) in c.v.iter_mut().enumerate() {
            *v = u128::MAX - i as u128;
        }
        let mut bytes = Vec::new();
        let c = CpuContext::A64(c);
        encode_context(&c, &mut bytes);
        assert_eq!(bytes.len(), CONTEXT_BYTES);
        assert_eq!(decode_context(&bytes), c);
    }

    #[test]
    fn events_and_faults_round_trip() {
        let events = [
            Event::Yield,
            Event::Syscall(SyscallFrame {
                number: 64,
                pc: 0x1004,
                args: [1, 2, 3, 4, 5, 6],
            }),
            Event::Fault(Fault::Memory {
                pc: 0x2000,
                fault: MemFault {
                    addr: 0xdead,
                    access: Access::Write,
                    kind: MemFaultKind::Protection,
                },
            }),
            Event::Fault(Fault::Undefined { pc: 8, insn: 0 }),
            Event::Fault(Fault::Breakpoint { pc: 12, imm: 7 }),
            Event::Fault(Fault::Alignment { pc: 16, addr: 3 }),
        ];
        for e in events {
            let mut words = Vec::new();
            let kind = encode_event(&e, &mut words);
            let pc = match e {
                Event::Syscall(f) => f.pc,
                _ => 0,
            };
            assert_eq!(decode_event(kind, pc, &words), e);
        }
        let mut v = Vec::new();
        encode_fault(Ok(()), &mut v);
        assert_eq!(decode_fault(&v), (Ok(()), 1));
        let f = MemFault {
            addr: 9,
            access: Access::Execute,
            kind: MemFaultKind::Unmapped,
        };
        v.clear();
        encode_fault(Err(f), &mut v);
        assert_eq!(decode_fault(&v), (Err(f), 4));
        for p in [Perms::NONE, Perms::R, Perms::RW, Perms::RX, Perms::RWX] {
            assert_eq!(perms_of(perms_code(p)), p);
        }
        for e in [
            Ok(()),
            Err(MapError::Unaligned),
            Err(MapError::OutOfRange),
            Err(MapError::NotMapped),
        ] {
            assert_eq!(map_error_of(map_error_code(e)), e);
        }
    }
}
