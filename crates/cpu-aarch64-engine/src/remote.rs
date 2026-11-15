//! The engine as a host-instantiated core module: its services
//! ([`wasmshim_cpu_remote::service`]) over handles to CPUs and address spaces, and `RUN` with
//! run-time translation (the host compiles what the engine asks for). This is the optional
//! run-time translation tier; the other path links translations ahead of time into the node.

#![allow(unsafe_code)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use alloc::rc::Rc;
use wasmshim_aarch64::Cpu;

use wasmshim_cpu::{Event, GuestCpu, GuestMemory, PagedMemory, SharedMemory, SharedPages};
use wasmshim_cpu_remote::{
    abi, config, decode_context, encode_context, encode_event, encode_fault, map_error_code,
    perms_code, perms_of, service,
};

use crate::dispatch::{Stop, Tier, dispatch};
use crate::mem::EngineMem;
use wasmshim_translate::tier::{Plan, Policy, Request, Tier as RunTier};

/// How the host's call ended.
pub enum Outcome {
    Done,
    /// A module waits in `req` for the host to compile.
    Compile,
}

struct RunState {
    cpu: u32,
    mem: u32,
    left: u64,
}

#[derive(Default)]
struct Stats {
    calls: u64,
    declined: u64,
    regions: u64,
    killed: u64,
}

pub struct Engine {
    cpus: Vec<Option<Box<Cpu>>>,
    mems: Vec<Option<Box<EngineMem>>>,
    /// Memory objects the client holds handles to.
    shareds: Vec<Option<Rc<SharedPages>>>,
    pub input: Vec<u8>,
    pub output: Vec<u8>,
    pub values: Vec<u64>,
    run: Option<RunState>,
    /// The run-time tier and its pending request: (address space, page
    /// identity, request).
    jit: RunTier,
    compile: Option<(u32, (u64, u64, u64), Request)>,
    pub req: Vec<u8>,
    pub req_funcs: u32,
    tier: Tier,
    stats: Stats,
}

/// Default executions before a block's region is translated (tiering is invisible to the
/// guest: only wall-clock time depends on it).
const DEFAULT_THRESHOLD: u32 = 1000;

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        Self {
            cpus: Vec::new(),
            mems: Vec::new(),
            shareds: Vec::new(),
            input: Vec::new(),
            output: Vec::new(),
            values: Vec::new(),
            run: None,
            jit: RunTier::new(Policy {
                threshold: DEFAULT_THRESHOLD,
                ..Policy::default()
            }),
            compile: None,
            req: Vec::new(),
            req_funcs: 0,
            tier: Tier {
                threshold: DEFAULT_THRESHOLD,
                ..Tier::default()
            },
            stats: Stats::default(),
        }
    }

    fn put<T>(v: &mut Vec<Option<T>>, x: T) -> u64 {
        if let Some(i) = v.iter().position(Option::is_none) {
            v[i] = Some(x);
            i as u64
        } else {
            v.push(Some(x));
            (v.len() - 1) as u64
        }
    }

    fn mem(&mut self, id: u64) -> &mut EngineMem {
        self.mems
            .get_mut(id as usize)
            .and_then(Option::as_mut)
            .expect("no such address space")
    }

    fn shared(&self, id: u64) -> Option<Rc<SharedPages>> {
        self.shareds.get(id as usize).and_then(Option::clone)
    }

    fn cpu(&mut self, id: u64) -> &mut Cpu {
        self.cpus
            .get_mut(id as usize)
            .and_then(Option::as_mut)
            .expect("no such cpu")
    }

    /// Applies the pending register writes of a call (`mask, x0, pc, sp, tp`).
    fn apply_pending(cpu: &mut Cpu, p: &[u64]) {
        let at = |i: usize| p.get(i).copied().unwrap_or(0);
        let mask = at(0);
        if mask & 1 != 0 {
            cpu.set_syscall_result(at(1));
        }
        if mask & 2 != 0 {
            cpu.set_pc(at(2));
        }
        if mask & 4 != 0 {
            cpu.set_sp(at(3));
        }
        if mask & 8 != 0 {
            cpu.set_thread_pointer(at(4));
        }
    }

    /// Starts a call.
    pub fn begin(&mut self, svc: u32, args: &[u64], input_len: usize) -> Outcome {
        self.stats.calls += 1;
        self.values.clear();
        self.output.clear();
        let at = |i: usize| args.get(i).copied().unwrap_or(0);
        match svc {
            service::NEW_MEM => {
                let id = Self::put(&mut self.mems, Box::new(EngineMem::new(PagedMemory::new())));
                self.values.push(id);
            }
            service::DUP_MEM => {
                let src = self.mem(at(0));
                // The pages become shared: no direct writes through the source's TLB.
                src.drop_write_entries();
                let copy = EngineMem::new(src.mem.clone());
                let id = Self::put(&mut self.mems, Box::new(copy));
                self.values.push(id);
            }
            service::FREE_MEM => {
                if let Some(slot) = self.mems.get_mut(at(0) as usize)
                    && let Some(m) = slot.take()
                {
                    self.stats.regions += m.stats.regions;
                    self.stats.killed += m.stats.killed;
                }
            }
            service::NEW_CPU => {
                let id = Self::put(&mut self.cpus, Box::new(Cpu::new()));
                self.values.push(id);
            }
            service::DUP_CPU => {
                let cpu = self.cpu(at(0));
                Self::apply_pending(cpu, args.get(1..).unwrap_or_default());
                let copy = Box::new(cpu.clone());
                let id = Self::put(&mut self.cpus, copy);
                self.values.push(id);
            }
            service::FREE_CPU => {
                if let Some(slot) = self.cpus.get_mut(at(0) as usize) {
                    *slot = None;
                }
            }
            service::MAP => {
                let r = self.mem(at(0)).map(at(1), at(2), perms_of(at(3)));
                self.values.push(map_error_code(r));
            }
            service::SHARED_NEW => {
                let id = Self::put(&mut self.shareds, Rc::new(SharedPages::new(None)));
                self.values.push(id);
            }
            service::SHARED_FREE => {
                if let Some(slot) = self.shareds.get_mut(at(0) as usize) {
                    *slot = None;
                }
            }
            service::SHARED_READ => {
                let mut out = core::mem::take(&mut self.output);
                out.resize(at(2) as usize, 0);
                let ok = self
                    .shared(at(0))
                    .is_some_and(|o| o.read(at(1), &mut out).is_ok());
                self.output = out;
                self.values.push(u64::from(!ok));
            }
            service::SHARED_WRITE => {
                let input = core::mem::take(&mut self.input);
                let ok = self
                    .shared(at(0))
                    .is_some_and(|o| o.write(at(1), &input[..input_len]).is_ok());
                self.input = input;
                self.values.push(u64::from(!ok));
            }
            service::SHARED_ZERO_FROM => {
                let ok = self
                    .shared(at(0))
                    .is_some_and(|o| o.zero_from(at(1)).is_ok());
                self.values.push(u64::from(!ok));
            }
            service::MAP_SHARED => {
                let r = match self.shared(at(4)) {
                    Some(o) => {
                        let o: Rc<dyn SharedMemory> = o;
                        self.mem(at(0))
                            .map_shared(at(1), at(2), perms_of(at(3)), o, at(5))
                    }
                    None => Err(wasmshim_cpu::MapError::Unsupported),
                };
                self.values.push(map_error_code(r));
            }
            service::UNMAP => {
                let r = self.mem(at(0)).unmap(at(1), at(2));
                self.values.push(map_error_code(r));
            }
            service::PROTECT => {
                let r = self.mem(at(0)).protect(at(1), at(2), perms_of(at(3)));
                self.values.push(map_error_code(r));
            }
            service::READ => {
                let mut out = core::mem::take(&mut self.output);
                out.resize(at(2) as usize, 0);
                let r = self.mem(at(0)).mem.read(at(1), &mut out);
                self.output = out;
                encode_fault(r, &mut self.values);
            }
            service::WRITE | service::POKE => {
                let input = core::mem::take(&mut self.input);
                let m = self.mem(at(0));
                let r = if svc == service::WRITE {
                    m.write(at(1), &input[..input_len])
                } else {
                    m.poke(at(1), &input[..input_len])
                };
                self.input = input;
                encode_fault(r, &mut self.values);
            }
            service::CHECK => {
                let r = self
                    .mem(at(0))
                    .check(at(1), at(2), wasmshim_cpu_remote::access_of(at(3)));
                encode_fault(r, &mut self.values);
            }
            service::PERMS => {
                let p = self.mem(at(0)).perms(at(1));
                self.values.push(p.map_or(255, perms_code));
            }
            service::READ_CSTR => {
                let r = self.mem(at(0)).mem.read_cstr(at(1), at(2) as usize);
                match r {
                    Ok(Some(s)) => {
                        self.values.extend_from_slice(&[0, 1]);
                        self.output = s;
                    }
                    Ok(None) => self.values.extend_from_slice(&[0, 0]),
                    Err(f) => encode_fault(Err(f), &mut self.values),
                }
            }
            service::ZERO => {
                let r = self.mem(at(0)).zero(at(1), at(2));
                encode_fault(r, &mut self.values);
            }
            service::CODE_EPOCH => {
                let e = self.mem(at(0)).code_epoch();
                self.values.push(e);
            }
            service::CODE_VERSION => {
                let v = self.mem(at(0)).code_version(at(1));
                self.values.push(v);
            }
            service::FETCH => {
                let r = self.mem(at(0)).fetch(at(1));
                encode_fault(r.map(|_| ()), &mut self.values);
                if let Ok(w) = r {
                    self.values.push(u64::from(w));
                }
            }
            service::GET_CONTEXT => {
                let cpu = self.cpu(at(0));
                Self::apply_pending(cpu, args.get(1..).unwrap_or_default());
                let ctx = cpu.context();
                encode_context(&ctx, &mut self.output);
            }
            service::SET_CONTEXT => {
                let ctx = decode_context(&self.input[..input_len]);
                self.cpu(at(0)).set_context(&ctx);
            }
            service::STATS => {
                let (mut regions, mut killed) = (self.stats.regions, self.stats.killed);
                for m in self.mems.iter().flatten() {
                    regions += m.stats.regions;
                    killed += m.stats.killed;
                }
                let t = &self.tier.stats;
                let s = &self.stats;
                self.values.extend_from_slice(&[
                    s.calls,
                    t.jit_calls,
                    t.interp_blocks,
                    t.compiles,
                    t.wasm_bytes,
                    s.declined,
                    t.budget_exits,
                    regions,
                    killed,
                    t.steps,
                    t.insns_jit,
                    t.insns_interp,
                    t.insns_translated,
                ]);
                let classes = crate::dispatch::step_classes();
                // (16 words fit a reply: two step classes, then the TLB fills.)
                self.values.extend_from_slice(&classes[..2]);
                self.values.push(crate::dispatch::fills());
            }
            service::STEP_SHAPES => {
                #[cfg(feature = "step-stats")]
                {
                    let (_, v) = crate::dispatch::step_stats();
                    for (s, n) in v.iter().take(60) {
                        self.output
                            .extend_from_slice(alloc::format!("{n:>9} {s}\n").as_bytes());
                    }
                }
            }
            service::CONFIG => match at(0) {
                config::THRESHOLD => {
                    let t = at(1).min(u64::from(u32::MAX)) as u32;
                    self.tier.threshold = t;
                    self.jit.policy.threshold = t;
                }
                // Accepted and ignored: this engine has no region size to set.
                config::REGION_BLOCKS => {}
                _ => {}
            },
            service::RUN => {
                let cpu = self.cpu(at(0));
                Self::apply_pending(cpu, args.get(4..).unwrap_or_default());
                cpu.set_virtual_time(at(3));
                self.run = Some(RunState {
                    cpu: at(0) as u32,
                    mem: at(1) as u32,
                    left: at(2),
                });
                // Validate the handles before the dispatcher takes raw pointers.
                let _ = self.mem(at(1));
                return self.dispatch();
            }
            _ => {}
        }
        Outcome::Done
    }

    /// Continues a call after a compile request (`slot`: where the host installed it).
    pub fn resume(&mut self, slot: u32) -> Outcome {
        if let Some((mem, (_, page, version), req)) = self.compile.take() {
            let first = (slot != abi::SLOT_DECLINED).then_some(slot);
            match self.jit.compiled(req, first) {
                Some(install) => self
                    .mem(u64::from(mem))
                    .install_page(page, version, &install),
                None => self.stats.declined += 1,
            }
        }
        if self.run.is_some() {
            self.dispatch()
        } else {
            Outcome::Done
        }
    }
    /// `RUN`: the shared dispatcher, until an event or a compile request.
    fn dispatch(&mut self) -> Outcome {
        let (cpu_id, mem_id, mut left) = {
            let r = self.run.as_ref().expect("a run in progress");
            (r.cpu, r.mem, r.left)
        };
        let cpu: *mut Cpu = &mut **self.cpus[cpu_id as usize].as_mut().expect("cpu");
        let mem: *mut EngineMem = &mut **self.mems[mem_id as usize].as_mut().expect("mem");
        let front = wasmshim_aarch64_translate::FrontHalf::default();
        loop {
            // SAFETY: both point into boxes owned by `self` that nothing frees during the call.
            match unsafe { dispatch(cpu, mem, &mut left, &mut self.tier) } {
                Stop::Event(event) => {
                    self.finish(cpu, event);
                    return Outcome::Done;
                }
                Stop::Hot(pc) => {
                    // SAFETY: as above; the dispatcher holds no reference any more.
                    let m = unsafe { &mut *mem };
                    let Some((id, bytes, warm)) = m.hot_page(pc) else {
                        continue;
                    };
                    let off = (pc & 0xfff) as u16;
                    match self.jit.plan(&front, id, &bytes, off, &warm) {
                        Plan::None => {}
                        Plan::Install(install) => m.install_page(id.1, id.2, &install),
                        Plan::Compile(req) => {
                            self.req = req.module.bytes.clone();
                            self.req_funcs = req.module.funcs;
                            let s = &mut self.tier.stats;
                            s.compiles += 1;
                            s.wasm_bytes += self.req.len() as u64;
                            s.insns_translated += req.module.insns as u64;
                            self.compile = Some((mem_id, id, req));
                            if let Some(r) = self.run.as_mut() {
                                r.left = left;
                            }
                            return Outcome::Compile;
                        }
                    }
                }
            }
        }
    }

    fn finish(&mut self, cpu: *mut Cpu, event: Event) {
        self.run = None;
        // SAFETY: as in `dispatch`; no translated call is in progress any more.
        let cpu = unsafe { &mut *cpu };
        cpu.clear_monitor();
        self.values.clear();
        self.values.extend_from_slice(&[
            0,
            cpu.retired(),
            cpu.pc(),
            cpu.sp(),
            cpu.thread_pointer(),
        ]);
        let mut payload = Vec::new();
        self.values[0] = encode_event(&event, &mut payload);
        self.values.extend_from_slice(&payload);
    }
}
