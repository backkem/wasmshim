//! A stack-style IR builder: a front half writes operations the way Wasm is written (operands
//! pushed, operations popping them), and gets SSA [`Op`]s.
//!
//! Temporaries are numbered locals (`local_set`, `local_get`, `local_tee`) mapping to values;
//! slot accesses go through [`Builder::get`] and [`Builder::set`]. Structured `if`s with or
//! without a result become [`Op::If`]; a value set into a temporary inside an arm is visible
//! only in that arm.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::*;

/// One open nested list: an arm of an `if`, or a memory access's fast path.
struct Frame {
    kind: FrameKind,
    /// The list that was being written when the frame opened.
    outer: Vec<Op>,
    /// Temporaries as they were when the frame opened (restored when it closes).
    temps: BTreeMap<u32, V>,
    /// The operand stack depth when the frame opened.
    depth: usize,
    ended: bool,
}

enum FrameKind {
    /// `if` with an optional result type; `then` holds the then-arm once `else` is seen.
    If {
        cond: V,
        result: Option<Ty>,
        then: Option<(Vec<Op>, Option<V>, bool)>,
    },
    /// A memory access's fast path.
    Fast(Mem),
}

/// The builder of one instruction's (or one block's) ops.
pub struct Builder {
    pub types: Vec<Ty>,
    pub ops: Vec<Op>,
    stack: Vec<V>,
    temps: BTreeMap<u32, V>,
    frames: Vec<Frame>,
    /// The current list ended in an exit: the rest of it is unreachable.
    pub ended: bool,
    /// Locals that stand for slots or the guest-state base ([`Self::map_local`]).
    mapped: BTreeMap<u32, Option<(Slot, Ty)>>,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            ops: Vec::new(),
            stack: Vec::new(),
            temps: BTreeMap::new(),
            frames: Vec::new(),
            ended: false,
            mapped: BTreeMap::new(),
        }
    }

    /// A new value of type `ty`.
    pub fn value(&mut self, ty: Ty) -> V {
        self.types.push(ty);
        V(self.types.len() as u32 - 1)
    }

    pub fn ty(&self, v: V) -> Ty {
        if v == CPU {
            Ty::I32
        } else {
            self.types[v.0 as usize]
        }
    }

    pub fn push(&mut self, v: V) -> &mut Self {
        self.stack.push(v);
        self
    }

    pub fn pop(&mut self) -> V {
        self.stack
            .pop()
            .expect("IR builder: operand stack underflow")
    }

    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    fn op(&mut self, op: Op) {
        if !self.ended {
            self.ops.push(op);
        }
    }

    /// Takes the ops written so far (one instruction's) and starts a new list.
    pub fn take(&mut self) -> Vec<Op> {
        debug_assert!(
            self.frames.is_empty(),
            "IR builder: open frame at an instruction's end"
        );
        debug_assert!(
            self.stack.is_empty(),
            "IR builder: operands left: {:?}",
            self.stack
        );
        self.temps.clear();
        core::mem::take(&mut self.ops)
    }

    // ---- constants and pure operations ----

    pub fn i32_const(&mut self, c: i32) -> &mut Self {
        let d = self.value(Ty::I32);
        self.op(Op::I32(d, c));
        self.push(d)
    }

    pub fn i64_const(&mut self, c: i64) -> &mut Self {
        let d = self.value(Ty::I64);
        self.op(Op::I64(d, c));
        self.push(d)
    }

    pub fn v128_const(&mut self, c: i128) -> &mut Self {
        let d = self.value(Ty::V128);
        self.op(Op::V128(d, c));
        self.push(d)
    }

    pub fn select(&mut self) -> &mut Self {
        let c = self.pop();
        let b = self.pop();
        let a = self.pop();
        let d = self.value(self.ty(a));
        self.op(Op::Select(d, a, b, c));
        self.push(d)
    }

    pub fn drop(&mut self) -> &mut Self {
        self.pop();
        self
    }

    pub fn lane(&mut self, op: LaneOp, lane: u8) -> &mut Self {
        let args = if op.is_replace() {
            let x = self.pop();
            let v = self.pop();
            alloc::vec![v, x]
        } else {
            alloc::vec![self.pop()]
        };
        let d = self.value(op.ty());
        self.op(Op::Lane(d, op, lane, args));
        self.push(d)
    }

    pub fn i8x16_shuffle(&mut self, lanes: [u8; 16]) -> &mut Self {
        let b = self.pop();
        let a = self.pop();
        let d = self.value(Ty::V128);
        self.op(Op::Shuffle(d, lanes, a, b));
        self.push(d)
    }

    // ---- temporaries ----

    /// Makes local number `l` stand for a slot (`Some`) or the guest-state base (`None`).
    pub fn map_local(&mut self, l: u32, slot: Option<(Slot, Ty)>) {
        self.mapped.insert(l, slot);
    }

    pub fn local_set(&mut self, t: u32) -> &mut Self {
        if let Some(&m) = self.mapped.get(&t) {
            let (s, _) = m.expect("IR builder: the guest-state base is read-only");
            return self.set(s);
        }
        let v = self.pop();
        self.temps.insert(t, v);
        self
    }

    pub fn local_tee(&mut self, t: u32) -> &mut Self {
        if let Some(&m) = self.mapped.get(&t) {
            let (s, _) = m.expect("IR builder: the guest-state base is read-only");
            let v = self.pop();
            self.op(Op::Set(s, v));
            return self.push(v);
        }
        let v = self.pop();
        self.temps.insert(t, v);
        self.push(v)
    }

    pub fn local_get(&mut self, t: u32) -> &mut Self {
        if let Some(&m) = self.mapped.get(&t) {
            return match m {
                Some((s, ty)) => self.get(s, ty),
                None => self.cpu(),
            };
        }
        let v = *self
            .temps
            .get(&t)
            .unwrap_or_else(|| panic!("IR builder: temporary {t} read before it was set"));
        self.push(v)
    }

    /// Pushes the guest-state base (the operand of [`Self::load`] and [`Self::store`] for
    /// guest-state memory).
    pub fn cpu(&mut self) -> &mut Self {
        self.push(CPU)
    }

    // ---- slots ----

    pub fn get(&mut self, s: Slot, ty: Ty) -> &mut Self {
        let d = self.value(ty);
        self.op(Op::Get(d, s));
        self.push(d)
    }

    pub fn set(&mut self, s: Slot) -> &mut Self {
        let v = self.pop();
        self.op(Op::Set(s, v));
        self
    }

    // ---- memory ----

    /// `[base] -> [value]`.
    pub fn load(&mut self, op: LoadOp, offset: u64) -> &mut Self {
        let base = self.pop();
        let d = self.value(op.ty());
        let base = if base == CPU {
            Base::Cpu
        } else {
            Base::Val(base)
        };
        self.op(Op::Load(d, op, base, offset));
        self.push(d)
    }

    /// `[base, value] -> []`.
    pub fn store(&mut self, op: StoreOp, offset: u64) -> &mut Self {
        let v = self.pop();
        let base = self.pop();
        let base = if base == CPU {
            Base::Cpu
        } else {
            Base::Val(base)
        };
        self.op(Op::Store(op, base, offset, v));
        self
    }

    // ---- the rest ----

    /// `[] -> [the runtime address of page offset off]`.
    pub fn pc(&mut self, off: i64) -> &mut Self {
        let d = self.value(Ty::I64);
        self.op(Op::Pc(d, off));
        self.push(d)
    }

    /// `[x, y] -> [x / y]` with every case defined.
    pub fn div(&mut self, wide: bool, signed: bool) -> &mut Self {
        let y = self.pop();
        let x = self.pop();
        let d = self.value(if wide { Ty::I64 } else { Ty::I32 });
        self.op(Op::Div {
            d,
            wide,
            signed,
            x,
            y,
        });
        self.push(d)
    }

    /// `[args...] -> [helper(args)]`.
    pub fn helper(&mut self, index: u8, args: usize, ty: Ty) -> &mut Self {
        let at = self.stack.len() - args;
        let a: Vec<V> = self.stack.drain(at..).collect();
        let d = self.value(ty);
        self.op(Op::Helper(d, index, a));
        self.push(d)
    }

    pub fn step(&mut self, pc: u64, k: u32, reads: Slots, writes: Slots) {
        self.op(Op::Step {
            pc,
            k,
            reads,
            writes,
        });
    }

    /// `[pc] -> leaves`: the current list ends.
    pub fn exit(&mut self, code: i32, giveback: u32) {
        let pc = self.pop();
        self.op(Op::Exit { code, pc, giveback });
        self.ended = true;
    }

    // ---- structure ----

    /// `[cond] -> []`: opens an `if` (with a `result`, its arms each leave one value).
    pub fn if_(&mut self, result: Option<Ty>) -> &mut Self {
        let cond = self.pop();
        let outer = core::mem::take(&mut self.ops);
        self.frames.push(Frame {
            kind: FrameKind::If {
                cond,
                result,
                then: None,
            },
            outer,
            temps: self.temps.clone(),
            depth: self.stack.len(),
            ended: self.ended,
        });
        self
    }

    pub fn else_(&mut self) -> &mut Self {
        let f = self.frames.last_mut().expect("IR builder: else without if");
        let FrameKind::If { result, then, .. } = &mut f.kind else {
            panic!("IR builder: else in a fast path");
        };
        let v = result.map(|_| self.stack.pop().expect("then-arm result"));
        *then = Some((core::mem::take(&mut self.ops), v, self.ended));
        self.temps = f.temps.clone();
        self.ended = f.ended;
        debug_assert_eq!(self.stack.len(), f.depth);
        self
    }

    pub fn end(&mut self) -> &mut Self {
        let f = self.frames.pop().expect("IR builder: end without if");
        let FrameKind::If { cond, result, then } = f.kind else {
            panic!("IR builder: end of a fast path (use end_fast)");
        };
        // The arm being closed: the else-arm, or (without `else`) the then-arm.
        let cur_v = result.map(|_| self.stack.pop().expect("arm result"));
        let cur_ops = core::mem::take(&mut self.ops);
        let cur_ended = self.ended;
        let (then_ops, then_v, then_ended, else_ops, else_v, else_ended) = match then {
            Some((t_ops, t_v, t_ended)) => (t_ops, t_v, t_ended, cur_ops, cur_v, cur_ended),
            // No `else`: an empty else-arm, which falls through.
            None => (cur_ops, cur_v, cur_ended, Vec::new(), None, f.ended),
        };
        self.ops = f.outer;
        self.temps = f.temps;
        self.ended = f.ended || (then_ended && else_ended);
        let mut out = Vec::new();
        let res = result.map(|ty| {
            let d = self.value(ty);
            let t = then_v.or(else_v).expect("an arm's value");
            let e = else_v.or(then_v).expect("an arm's value");
            out.push((d, t, e));
            d
        });
        if !f.ended {
            self.ops.push(Op::If {
                cond,
                then: then_ops,
                else_: else_ops,
                out,
            });
        }
        if let Some(d) = res {
            self.push(d);
        }
        self
    }

    /// Opens a memory access's fast path: `mem.host` is the host address inside it.
    pub fn begin_fast(&mut self, mem: Mem) {
        let outer = core::mem::take(&mut self.ops);
        self.frames.push(Frame {
            kind: FrameKind::Fast(mem),
            outer,
            temps: self.temps.clone(),
            depth: self.stack.len(),
            ended: self.ended,
        });
    }

    pub fn end_fast(&mut self) {
        let f = self
            .frames
            .pop()
            .expect("IR builder: end_fast without begin_fast");
        let FrameKind::Fast(mut mem) = f.kind else {
            panic!("IR builder: end_fast of an if");
        };
        debug_assert_eq!(self.stack.len(), f.depth);
        mem.fast = core::mem::take(&mut self.ops);
        self.ops = f.outer;
        // Temporaries set in the fast path (the address, say) stay visible: the slow path
        // computes nothing the rest of the instruction reads.
        self.ended = f.ended;
        if !self.ended {
            self.ops.push(Op::Mem(alloc::boxed::Box::new(mem)));
        }
    }
}

/// Wasm-named wrappers (so lowering code reads like the Wasm it stands for).
impl Builder {
    pub fn i64_load(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.load(LoadOp::I64, m.offset)
    }
    pub fn i64_load8_u(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.load(LoadOp::I64_8U, m.offset)
    }
    pub fn i64_load8_s(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.load(LoadOp::I64_8S, m.offset)
    }
    pub fn i64_load16_u(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.load(LoadOp::I64_16U, m.offset)
    }
    pub fn i64_load16_s(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.load(LoadOp::I64_16S, m.offset)
    }
    pub fn i64_load32_u(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.load(LoadOp::I64_32U, m.offset)
    }
    pub fn i64_load32_s(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.load(LoadOp::I64_32S, m.offset)
    }
    pub fn i32_load(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.load(LoadOp::I32, m.offset)
    }
    pub fn i32_load8_s(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.load(LoadOp::I32_8S, m.offset)
    }
    pub fn i32_load16_s(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.load(LoadOp::I32_16S, m.offset)
    }
    pub fn v128_load(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.load(LoadOp::V128, m.offset)
    }
    pub fn i64_store(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.store(StoreOp::I64, m.offset)
    }
    pub fn i64_store8(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.store(StoreOp::I64_8, m.offset)
    }
    pub fn i64_store16(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.store(StoreOp::I64_16, m.offset)
    }
    pub fn i64_store32(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.store(StoreOp::I64_32, m.offset)
    }
    pub fn i32_store(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.store(StoreOp::I32, m.offset)
    }
    pub fn v128_store(&mut self, m: wasm_encoder::MemArg) -> &mut Self {
        self.store(StoreOp::V128, m.offset)
    }
    pub fn i8x16_extract_lane_u(&mut self, i: u8) -> &mut Self {
        self.lane(LaneOp::I8x16ExtractU, i)
    }
    pub fn i8x16_extract_lane_s(&mut self, i: u8) -> &mut Self {
        self.lane(LaneOp::I8x16ExtractS, i)
    }
    pub fn i16x8_extract_lane_u(&mut self, i: u8) -> &mut Self {
        self.lane(LaneOp::I16x8ExtractU, i)
    }
    pub fn i16x8_extract_lane_s(&mut self, i: u8) -> &mut Self {
        self.lane(LaneOp::I16x8ExtractS, i)
    }
    pub fn i32x4_extract_lane(&mut self, i: u8) -> &mut Self {
        self.lane(LaneOp::I32x4Extract, i)
    }
    pub fn i64x2_extract_lane(&mut self, i: u8) -> &mut Self {
        self.lane(LaneOp::I64x2Extract, i)
    }
    pub fn i8x16_replace_lane(&mut self, i: u8) -> &mut Self {
        self.lane(LaneOp::I8x16Replace, i)
    }
    pub fn i16x8_replace_lane(&mut self, i: u8) -> &mut Self {
        self.lane(LaneOp::I16x8Replace, i)
    }
    pub fn i32x4_replace_lane(&mut self, i: u8) -> &mut Self {
        self.lane(LaneOp::I32x4Replace, i)
    }
    pub fn i64x2_replace_lane(&mut self, i: u8) -> &mut Self {
        self.lane(LaneOp::I64x2Replace, i)
    }
    /// Wasm's `if` with a block type.
    pub fn if_block(&mut self, bt: wasm_encoder::BlockType) -> &mut Self {
        let result = match bt {
            wasm_encoder::BlockType::Result(wasm_encoder::ValType::I32) => Some(Ty::I32),
            wasm_encoder::BlockType::Result(wasm_encoder::ValType::I64) => Some(Ty::I64),
            wasm_encoder::BlockType::Result(wasm_encoder::ValType::V128) => Some(Ty::V128),
            _ => None,
        };
        self.if_(result)
    }
}

impl PureOps for Builder {
    fn pure(&mut self, p: P) -> &mut Self {
        let n = p.args().len();
        let at = self.stack.len() - n;
        let args: Vec<V> = self.stack.drain(at..).collect();
        let d = self.value(p.ty());
        self.op(Op::Pure(d, p, args));
        self.push(d)
    }
}
