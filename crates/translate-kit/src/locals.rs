//! Local compaction: a translated function's value locals packed into as few
//! Wasm locals as their live ranges allow.
//!
//! A back half gives every value of a block a local of its own for the whole block, so a
//! region function declares hundreds of locals, almost all of them dead most of the time.
//! Engines pay for declared locals at every control-flow merge (V8's Liftoff reconciles each
//! one; TurboFan and Cranelift build a phi per local and merge), so a region compiled at the
//! first call of one of its blocks costs far more than its size says: in the browser that cost
//! sits on the guest's critical path.
//!
//! [`compact`] reads a finished body, computes the liveness of the locals it may move over the
//! body's structured control flow, takes each local's live range as the hull of the points
//! where it is live, used or defined, and assigns the ranges to slots per type by linear scan.
//! The rewritten body executes exactly the same instructions (only local indices change), so
//! it computes the same values with the same fuel. A local live at entry (read before any
//! write on some path: Wasm zero-initialises locals) keeps a slot of its own. Anything the pass
//! does not model (exception handling, GC branches) leaves the body as it was.

use alloc::vec;
use alloc::vec::Vec;

use wasm_encoder::ValType;
use wasmparser::{BinaryReader, Operator, OperatorsReader};

/// A compacted body: its code, its new local declarations, and where each byte of the old code
/// moved (for offsets recorded during emission: calls, hints).
pub struct Compacted {
    pub code: Vec<u8>,
    pub locals: Vec<(u32, ValType)>,
    /// `(old offset, old length, new length)` of each rewritten local immediate, in order.
    edits: Vec<(usize, u8, u8)>,
}

impl Compacted {
    /// The new position of old code offset `p` (which must not lie inside a rewritten local
    /// immediate: call indices and branch opcodes never do).
    pub fn map(&self, p: usize) -> usize {
        let mut delta: isize = 0;
        for &(o, old, new) in &self.edits {
            if o >= p {
                break;
            }
            delta += isize::from(new) - isize::from(old);
        }
        p.wrapping_add_signed(delta)
    }

    /// Locals declared after compaction, beyond the fixed ones.
    pub fn moved_locals(&self, fixed: usize) -> u32 {
        self.locals[fixed..].iter().map(|&(n, _)| n).sum()
    }
}

#[derive(Clone, Copy)]
enum Access {
    None,
    Get(u32),
    Def(u32),
}

/// What a control instruction does to the flow (instruction indices).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Plain,
    Block,
    Loop,
    If,
    Else,
    End,
    Br(u32),
    BrIf(u32),
    BrTable,
    Exit,
}

struct Ins {
    kind: Kind,
    access: Access,
    /// Byte offset of the instruction and of its end.
    at: usize,
    end: usize,
}

/// Packs the locals of `code` (a function body's instructions after its local declarations)
/// numbered from `params + Σ fixed` on (`extra`: their types, in index order) into as few
/// locals as their live ranges allow. Returns `None` when the body uses something the pass
/// does not model, or when nothing would change.
pub fn compact(
    code: &[u8],
    params: u32,
    fixed: &[(u32, ValType)],
    extra: &[ValType],
) -> Option<Compacted> {
    let first = params + fixed.iter().map(|&(n, _)| n).sum::<u32>();
    let n_extra = extra.len();
    if n_extra < 2 {
        return None;
    }
    let mut ins: Vec<Ins> = Vec::new();
    let mut table_targets: Vec<Vec<u32>> = Vec::new();
    let mut table_of: Vec<u32> = Vec::new();
    let mut reader = OperatorsReader::new(BinaryReader::new(code, 0));
    while !reader.eof() {
        let at = reader.original_position() as usize;
        let op = reader.read().ok()?;
        let end = reader.original_position() as usize;
        let movable = |i: u32| i >= first && ((i - first) as usize) < n_extra;
        let mut access = Access::None;
        let kind = match op {
            Operator::LocalGet { local_index } if movable(local_index) => {
                access = Access::Get(local_index - first);
                Kind::Plain
            }
            Operator::LocalSet { local_index } | Operator::LocalTee { local_index }
                if movable(local_index) =>
            {
                access = Access::Def(local_index - first);
                Kind::Plain
            }
            Operator::Block { .. } => Kind::Block,
            Operator::Loop { .. } => Kind::Loop,
            Operator::If { .. } => Kind::If,
            Operator::Else => Kind::Else,
            Operator::End => Kind::End,
            Operator::Br { relative_depth } => Kind::Br(relative_depth),
            Operator::BrIf { relative_depth } => Kind::BrIf(relative_depth),
            Operator::BrTable { targets } => {
                let mut t: Vec<u32> = Vec::new();
                for d in targets.targets() {
                    t.push(d.ok()?);
                }
                t.push(targets.default());
                table_of.push(ins.len() as u32);
                table_targets.push(t);
                Kind::BrTable
            }
            Operator::Return
            | Operator::Unreachable
            | Operator::ReturnCall { .. }
            | Operator::ReturnCallIndirect { .. }
            | Operator::ReturnCallRef { .. } => Kind::Exit,
            Operator::Try { .. }
            | Operator::Catch { .. }
            | Operator::CatchAll
            | Operator::Delegate { .. }
            | Operator::Rethrow { .. }
            | Operator::Throw { .. }
            | Operator::ThrowRef
            | Operator::TryTable { .. }
            | Operator::BrOnNull { .. }
            | Operator::BrOnNonNull { .. }
            | Operator::BrOnCast { .. }
            | Operator::BrOnCastFail { .. } => return None,
            _ => Kind::Plain,
        };
        ins.push(Ins {
            kind,
            access,
            at,
            end,
        });
    }
    let n = ins.len();
    if n == 0 || ins[n - 1].kind != Kind::End {
        return None;
    }

    // Structure: each block/loop/if's `end` (and an `if`'s `else`); the function body is the
    // outermost block (its `end` the last instruction).
    let mut end_of = vec![u32::MAX; n];
    let mut else_of = vec![u32::MAX; n];
    let mut owner = vec![u32::MAX; n]; // for `else`/`end`: the instruction that opened it
    let mut stack: Vec<u32> = vec![u32::MAX]; // u32::MAX: the function's own block
    for (i, x) in ins.iter().enumerate() {
        match x.kind {
            Kind::Block | Kind::Loop | Kind::If => stack.push(i as u32),
            Kind::Else => {
                let o = *stack.last()?;
                if o == u32::MAX || ins[o as usize].kind != Kind::If {
                    return None;
                }
                else_of[o as usize] = i as u32;
                owner[i] = o;
            }
            Kind::End => {
                let o = stack.pop()?;
                owner[i] = o;
                if o != u32::MAX {
                    end_of[o as usize] = i as u32;
                } else if i != n - 1 {
                    return None;
                }
            }
            _ => {}
        }
    }
    if !stack.is_empty() {
        return None;
    }
    // Branch targets need the frames open at each branch: walk again with the stack.
    let target = |frames: &[u32], d: u32| -> Option<u32> {
        let f = *frames.get(frames.len().checked_sub(1 + d as usize)?)?;
        Some(if f == u32::MAX {
            (n - 1) as u32
        } else if ins[f as usize].kind == Kind::Loop {
            f + 1
        } else {
            end_of[f as usize]
        })
    };
    let mut succ: Vec<[u32; 2]> = vec![[u32::MAX; 2]; n];
    let mut multi: Vec<(u32, Vec<u32>)> = Vec::new(); // br_table successors
    let mut frames: Vec<u32> = vec![u32::MAX];
    let mut table_k = 0;
    for i in 0..n {
        let next = i as u32 + 1;
        let s = match ins[i].kind {
            Kind::Plain | Kind::Block | Kind::Loop => [next, u32::MAX],
            Kind::If => {
                let e = else_of[i];
                [next, if e != u32::MAX { e + 1 } else { end_of[i] }]
            }
            Kind::Else => [end_of[owner[i] as usize], u32::MAX],
            Kind::End => {
                if i == n - 1 {
                    [u32::MAX; 2]
                } else {
                    [next, u32::MAX]
                }
            }
            Kind::Br(d) => [target(&frames, d)?, u32::MAX],
            Kind::BrIf(d) => [target(&frames, d)?, next],
            Kind::BrTable => {
                let mut t: Vec<u32> = Vec::new();
                for &d in &table_targets[table_k] {
                    t.push(target(&frames, d)?);
                }
                table_k += 1;
                t.sort_unstable();
                t.dedup();
                multi.push((i as u32, t));
                [u32::MAX; 2]
            }
            Kind::Exit => [u32::MAX; 2],
        };
        succ[i] = s;
        match ins[i].kind {
            Kind::Block | Kind::Loop | Kind::If => frames.push(i as u32),
            Kind::End => {
                frames.pop();
            }
            _ => {}
        }
    }
    let _ = table_of;

    // Basic blocks: leaders at 0, at every branch target, and after every instruction that
    // does not simply fall through.
    let mut leader = vec![false; n + 1];
    leader[0] = true;
    let mut multi_k = 0;
    for i in 0..n {
        let straight = succ[i] == [i as u32 + 1, u32::MAX];
        if straight {
            continue;
        }
        leader[i + 1] = true;
        if ins[i].kind == Kind::BrTable {
            for &t in &multi[multi_k].1 {
                leader[t as usize] = true;
            }
            multi_k += 1;
        } else {
            for t in succ[i] {
                if t != u32::MAX {
                    leader[t as usize] = true;
                }
            }
        }
    }
    let starts: Vec<usize> = (0..n).filter(|&i| leader[i]).collect();
    let nb = starts.len();
    let mut block_of = vec![0u32; n + 1];
    for (b, &s) in starts.iter().enumerate() {
        let e = starts.get(b + 1).copied().unwrap_or(n);
        for slot in &mut block_of[s..e] {
            *slot = b as u32;
        }
    }
    let block_end = |b: usize| starts.get(b + 1).copied().unwrap_or(n);
    let mut bsucc: Vec<Vec<u32>> = vec![Vec::new(); nb];
    let mut multi_k = 0;
    for (b, succs) in bsucc.iter_mut().enumerate() {
        let last = block_end(b) - 1;
        if ins[last].kind == Kind::BrTable {
            while multi[multi_k].0 as usize != last {
                multi_k += 1;
            }
            for &t in &multi[multi_k].1 {
                succs.push(block_of[t as usize]);
            }
        } else {
            for t in succ[last] {
                if t != u32::MAX && (t as usize) < n {
                    succs.push(block_of[t as usize]);
                }
            }
        }
    }

    // Liveness over the movable locals (bitsets).
    let words = n_extra.div_ceil(64);
    let set = |v: &mut [u64], l: u32| v[(l / 64) as usize] |= 1 << (l % 64);
    let clear = |v: &mut [u64], l: u32| v[(l / 64) as usize] &= !(1 << (l % 64));
    let mut uses = vec![0u64; nb * words];
    let mut defs = vec![0u64; nb * words];
    for b in 0..nb {
        let (u, d) = (
            &mut uses[b * words..(b + 1) * words],
            &mut defs[b * words..(b + 1) * words],
        );
        for i in (starts[b]..block_end(b)).rev() {
            match ins[i].access {
                Access::Get(l) => set(u, l),
                Access::Def(l) => {
                    clear(u, l);
                    set(d, l);
                }
                Access::None => {}
            }
        }
    }
    let mut live_in = vec![0u64; nb * words];
    let mut live_out = vec![0u64; nb * words];
    loop {
        let mut changed = false;
        for b in (0..nb).rev() {
            let mut out = vec![0u64; words];
            for &s in &bsucc[b] {
                let s = s as usize;
                for (o, x) in out.iter_mut().zip(&live_in[s * words..(s + 1) * words]) {
                    *o |= x;
                }
            }
            for w in 0..words {
                let inn = uses[b * words + w] | (out[w] & !defs[b * words + w]);
                if inn != live_in[b * words + w] {
                    live_in[b * words + w] = inn;
                    changed = true;
                }
                live_out[b * words + w] = out[w];
            }
        }
        if !changed {
            break;
        }
    }

    // Live ranges: the hull of the instructions where each local is live, used or defined.
    let mut lo = vec![u32::MAX; n_extra];
    let mut hi = vec![0u32; n_extra];
    let mut touch = |l: u32, i: u32| {
        let l = l as usize;
        lo[l] = lo[l].min(i);
        hi[l] = hi[l].max(i);
    };
    let bits = |v: &[u64]| {
        let mut out = Vec::new();
        for (w, &x) in v.iter().enumerate() {
            let mut x = x;
            while x != 0 {
                let t = x.trailing_zeros();
                out.push(w as u32 * 64 + t);
                x &= x - 1;
            }
        }
        out
    };
    for b in 0..nb {
        let (s, e) = (starts[b] as u32, block_end(b) as u32 - 1);
        for l in bits(&live_out[b * words..(b + 1) * words]) {
            touch(l, e);
        }
        for l in bits(&live_in[b * words..(b + 1) * words]) {
            touch(l, s);
        }
    }
    for (i, x) in ins.iter().enumerate() {
        match x.access {
            Access::Get(l) | Access::Def(l) => touch(l, i as u32),
            Access::None => {}
        }
    }
    // Locals live at entry keep a slot of their own for the whole body.
    let pinned = bits(&live_in[..words]);
    for &l in &pinned {
        lo[l as usize] = 0;
        hi[l as usize] = n as u32;
    }

    // Linear scan per type.
    let type_index = |t: ValType| match t {
        ValType::I32 => 0usize,
        ValType::I64 => 1,
        ValType::V128 => 2,
        ValType::F32 => 3,
        ValType::F64 => 4,
        ValType::Ref(_) => 5,
    };
    let mut order: Vec<u32> = (0..n_extra as u32)
        .filter(|&l| lo[l as usize] != u32::MAX)
        .collect();
    order.sort_by_key(|&l| (lo[l as usize], l));
    let mut slot_of = vec![u32::MAX; n_extra];
    let mut slots_end: [Vec<u32>; 6] = Default::default(); // per type: each slot's last end
    for &l in &order {
        let t = type_index(extra[l as usize]);
        if t == 5 {
            return None;
        }
        let ends = &mut slots_end[t];
        // The lowest free slot (small indices encode short).
        let free = ends.iter().position(|&e| e < lo[l as usize]);
        let s = match free {
            Some(s) => s,
            None => {
                ends.push(0);
                ends.len() - 1
            }
        };
        ends[s] = hi[l as usize];
        slot_of[l as usize] = s as u32;
    }
    let tys = [
        ValType::I32,
        ValType::I64,
        ValType::V128,
        ValType::F32,
        ValType::F64,
    ];
    let mut base = [0u32; 5];
    let mut locals: Vec<(u32, ValType)> = fixed.to_vec();
    let mut next = 0u32;
    for (t, ty) in tys.iter().enumerate() {
        base[t] = next;
        let count = slots_end[t].len() as u32;
        if count > 0 {
            locals.push((count, *ty));
        }
        next += count;
    }
    if next as usize >= n_extra {
        return None;
    }
    let new_index = |l: u32| first + base[type_index(extra[l as usize])] + slot_of[l as usize];

    // Rewrite.
    let mut out = Vec::with_capacity(code.len());
    let mut edits = Vec::new();
    let mut copied = 0;
    for x in &ins {
        let l = match x.access {
            Access::Get(l) | Access::Def(l) => l,
            Access::None => continue,
        };
        let imm = x.at + 1;
        out.extend_from_slice(&code[copied..imm]);
        let before = out.len();
        leb(&mut out, new_index(l));
        let new_len = (out.len() - before) as u8;
        edits.push((imm, (x.end - imm) as u8, new_len));
        copied = x.end;
    }
    out.extend_from_slice(&code[copied..]);
    Some(Compacted {
        code: out,
        locals,
        edits,
    })
}

fn leb(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_encoder::{BlockType, Function, Instruction};

    fn body(f: impl FnOnce(&mut Vec<Instruction<'static>>)) -> Vec<u8> {
        let mut ins = Vec::new();
        f(&mut ins);
        let mut func = Function::new([]);
        for i in &ins {
            func.instruction(i);
        }
        let raw = func.into_raw_body();
        raw[1..].to_vec() // drop the empty locals vector
    }

    #[test]
    fn disjoint_ranges_share_a_local() {
        // param 0; fixed: none; extra 0..3 i64: a, b, c used one after another.
        let code = body(|v| {
            for l in 1..4u32 {
                v.push(Instruction::I64Const(l as i64));
                v.push(Instruction::LocalSet(l));
                v.push(Instruction::LocalGet(l));
                v.push(Instruction::Drop);
            }
            v.push(Instruction::End);
        });
        let c = compact(&code, 1, &[], &[ValType::I64; 3]).expect("compacted");
        assert_eq!(c.locals, vec![(1, ValType::I64)]);
    }

    #[test]
    fn a_value_around_a_loop_keeps_its_local() {
        // a is set before the loop and read in it; b lives only inside the loop body before
        // the read of a: they overlap, so two locals.
        let code = body(|v| {
            v.push(Instruction::I64Const(1));
            v.push(Instruction::LocalSet(1)); // a
            v.push(Instruction::Loop(BlockType::Empty));
            v.push(Instruction::I64Const(2));
            v.push(Instruction::LocalSet(2)); // b
            v.push(Instruction::LocalGet(2));
            v.push(Instruction::Drop);
            v.push(Instruction::LocalGet(1));
            v.push(Instruction::I32WrapI64);
            v.push(Instruction::BrIf(0));
            v.push(Instruction::End);
            v.push(Instruction::End);
        });
        assert!(
            compact(&code, 1, &[], &[ValType::I64; 2]).is_none(),
            "nothing to share"
        );
    }

    #[test]
    fn a_local_read_before_its_write_stays_alone() {
        // b is read first (zero) on one path: it keeps its own slot.
        let code = body(|v| {
            v.push(Instruction::LocalGet(0));
            v.push(Instruction::If(BlockType::Empty));
            v.push(Instruction::I64Const(5));
            v.push(Instruction::LocalSet(2));
            v.push(Instruction::End);
            v.push(Instruction::LocalGet(2));
            v.push(Instruction::Drop);
            v.push(Instruction::I64Const(1));
            v.push(Instruction::LocalSet(3));
            v.push(Instruction::LocalGet(3));
            v.push(Instruction::Drop);
            v.push(Instruction::I64Const(1));
            v.push(Instruction::LocalSet(4));
            v.push(Instruction::LocalGet(4));
            v.push(Instruction::Drop);
            v.push(Instruction::End);
        });
        let c = compact(&code, 2, &[], &[ValType::I64; 3]).expect("compacted");
        // b (pinned) alone; c and d share.
        assert_eq!(c.locals, vec![(2, ValType::I64)]);
    }

    #[test]
    fn offsets_move_with_shorter_indices() {
        let code = body(|v| {
            for l in 0..200u32 {
                v.push(Instruction::I32Const(l as i32));
                v.push(Instruction::LocalSet(l));
                v.push(Instruction::LocalGet(l));
                v.push(Instruction::Drop);
            }
            v.push(Instruction::Nop);
            v.push(Instruction::End);
        });
        let nop = code.len() - 2;
        let c = compact(&code, 0, &[], &[ValType::I32; 200]).expect("compacted");
        assert_eq!(c.locals, vec![(1, ValType::I32)]);
        assert_eq!(c.code[c.map(nop)], 0x01);
        assert_eq!(c.code.len(), c.map(code.len()));
    }
}
