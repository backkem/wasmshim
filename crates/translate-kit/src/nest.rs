//! Loop regions: the single-entry loop nests among a page's hot blocks.
//!
//! The hot blocks of a page and their static edges (direct branches and fall-throughs to other
//! hot blocks) form a graph. Every strongly connected component with a cycle is a loop. Its
//! **header** is the one block entered from outside it (a loop entered at two places is left
//! alone; one entered only from other pages gets its lowest block). Without the edges back into
//! the header, the loop's blocks split into components again: single blocks (perhaps looping
//! on themselves) and inner loops, each of which must again have exactly one header. Laid out
//! in topological order of those components, every edge of the nest goes forward or back to
//! the header of a loop that encloses it, which is what structured Wasm (`block` / `loop`)
//! expresses without any dispatch.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::region::{Block, Isa, Item, Region, flatten};

/// Most blocks of a loop region.
const MAX_BLOCKS: usize = 64;
/// Most instructions of a loop region.
const MAX_INSNS: usize = 2048;

/// The loop regions of `blocks` (one page's hot blocks, distinct PCs): block 0 of each is the
/// loop's header, its only entry.
pub fn loop_regions<A: Isa>(blocks: &[Block<A>]) -> Vec<Region<A>> {
    let index: BTreeMap<u64, usize> = blocks.iter().enumerate().map(|(i, b)| (b.pc, i)).collect();
    let succ: Vec<Vec<usize>> = blocks
        .iter()
        .map(|b| {
            let mut s: Vec<usize> = A::successors(b)
                .into_iter()
                .filter_map(|t| index.get(&t).copied())
                .collect();
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();
    let all: Vec<usize> = (0..blocks.len()).collect();
    let mut out = Vec::new();
    for comp in sccs(&all, &succ, None) {
        if !cyclic(&comp, &succ) {
            continue;
        }
        let insns: usize = comp.iter().map(|&b| blocks[b].insts.len()).sum();
        if comp.len() > MAX_BLOCKS || insns > MAX_INSNS {
            continue;
        }
        let entered: Vec<usize> = comp
            .iter()
            .copied()
            .filter(|&n| {
                (0..blocks.len()).any(|p| comp.binary_search(&p).is_err() && succ[p].contains(&n))
            })
            .collect();
        let header = match entered.as_slice() {
            [h] => *h,
            [] => comp
                .iter()
                .copied()
                .min_by_key(|&n| blocks[n].pc)
                .unwrap_or(0),
            _ => continue,
        };
        let Some(items) = nest(&comp, header, &succ) else {
            continue;
        };
        // Region-local indices in layout order (the header first).
        let mut layout = Vec::new();
        flatten(&items, &mut layout);
        let local: BTreeMap<usize, usize> =
            layout.iter().enumerate().map(|(i, &g)| (g, i)).collect();
        out.push(Region {
            blocks: layout.iter().map(|&g| blocks[g].clone()).collect(),
            nest: Some(vec![Item::Loop(remap(&items, &local))]),
        });
    }
    out
}

/// The layout of the loop `comp` (sorted block indices) entered at `header`, or `None` if an
/// inner loop has more than one entry.
fn nest(comp: &[usize], header: usize, succ: &[Vec<usize>]) -> Option<Vec<Item>> {
    // Without the back edges into the header, in topological order (Tarjan finds components
    // in reverse topological order; the header's is the only source).
    let mut parts = sccs(comp, succ, Some(header));
    parts.reverse();
    let mut items = Vec::new();
    for part in parts {
        if part.len() == 1 {
            // A block, perhaps looping on itself (the emitter makes that a `loop`).
            items.push(Item::Block(part[0]));
            continue;
        }
        let entered: Vec<usize> = part
            .iter()
            .copied()
            .filter(|&n| {
                comp.iter().any(|&p| {
                    part.binary_search(&p).is_err() && n != header && succ[p].contains(&n)
                })
            })
            .collect();
        let [inner] = entered.as_slice() else {
            return None;
        };
        items.push(Item::Loop(nest(&part, *inner, succ)?));
    }
    (items.first() == Some(&Item::Block(header))).then_some(items)
}

fn remap(items: &[Item], local: &BTreeMap<usize, usize>) -> Vec<Item> {
    items
        .iter()
        .map(|item| match item {
            Item::Block(b) => Item::Block(local[b]),
            Item::Loop(inner) => Item::Loop(remap(inner, local)),
        })
        .collect()
}

/// `true` if the component has a cycle: more than one block, or a block branching to itself.
fn cyclic(comp: &[usize], succ: &[Vec<usize>]) -> bool {
    comp.len() > 1 || succ[comp[0]].contains(&comp[0])
}

/// Strongly connected components of the subgraph on `nodes` (sorted), without edges into
/// `cut`; each component sorted, the components in reverse topological order (Tarjan).
/// Deterministic: nodes and edges are visited in index order.
fn sccs(nodes: &[usize], succ: &[Vec<usize>], cut: Option<usize>) -> Vec<Vec<usize>> {
    struct St<'a> {
        nodes: &'a [usize],
        succ: &'a [Vec<usize>],
        cut: Option<usize>,
        index: BTreeMap<usize, (usize, usize)>, // node -> (index, lowlink)
        stack: Vec<usize>,
        on: BTreeMap<usize, bool>,
        next: usize,
        out: Vec<Vec<usize>>,
    }
    fn visit(s: &mut St<'_>, v: usize) {
        s.index.insert(v, (s.next, s.next));
        s.next += 1;
        s.stack.push(v);
        s.on.insert(v, true);
        let succ = s.succ;
        for &w in &succ[v] {
            if s.nodes.binary_search(&w).is_err() || Some(w) == s.cut {
                continue;
            }
            match s.index.get(&w).copied() {
                None => {
                    visit(s, w);
                    let lw = s.index[&w].1;
                    let e = s.index.get_mut(&v).expect("visited");
                    e.1 = e.1.min(lw);
                }
                Some((iw, _)) if s.on.get(&w).copied().unwrap_or(false) => {
                    let e = s.index.get_mut(&v).expect("visited");
                    e.1 = e.1.min(iw);
                }
                Some(_) => {}
            }
        }
        let (iv, lv) = s.index[&v];
        if iv == lv {
            let mut comp = Vec::new();
            while let Some(w) = s.stack.pop() {
                s.on.insert(w, false);
                comp.push(w);
                if w == v {
                    break;
                }
            }
            comp.sort_unstable();
            s.out.push(comp);
        }
    }
    let mut s = St {
        nodes,
        succ,
        cut,
        index: BTreeMap::new(),
        stack: Vec::new(),
        on: BTreeMap::new(),
        next: 0,
        out: Vec::new(),
    };
    // The cut node (a header) first, so its component comes out last (the source).
    for &n in cut.iter().chain(nodes.iter()) {
        if !s.index.contains_key(&n) {
            visit(&mut s, n);
        }
    }
    s.out
}
