//! Translations as standalone Wasm modules, for the run-time tier ([`crate::tier`]): what a host
//! compiles and instantiates against the engine's memory and table.
//!
//! A module imports, all from module [`abi::IMPORT_MODULE`]: the engine's memory
//! ([`abi::IMPORT_MEMORY`]), `<prefix>_jit_step` and `<prefix>_jit_fill`, and the engine's
//! function table ([`abi::IMPORT_TABLE`]). It exports its region functions as `f0`, `f1`, …; the
//! host appends them to the table, in that order, and reports the index of `f0`. Nothing else is
//! imported or exported, so a host can check a module before it compiles it.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use wasm_encoder::{
    CodeSection, CustomSection, EntityType, ExportKind, ExportSection, FunctionSection,
    ImportSection, MemoryType, Module, RefType, TableType, TypeSection, ValType,
};

use crate::abi;
use crate::aot::{
    BRANCH_HINT_SECTION, BackHalf, Linking, PAGE, PageTranslation, branch_hints, leb, page_peers,
};
use crate::emit::{Callee, Callees, patch_padded_leb};
use crate::region::{Isa, Region};

/// A translated page as a module.
#[derive(Clone, Debug)]
pub struct PageModule {
    pub bytes: Vec<u8>,
    /// Exported region functions (`f0` … `f<funcs - 1>`).
    pub funcs: u32,
    /// Every entry: `(page offset, function number, block index)`, in region order.
    pub entries: Vec<(u16, u32, u16)>,
    /// Guest instructions translated (statistics).
    pub insns: usize,
}

fn val(b: u8) -> ValType {
    match b {
        0x7f => ValType::I32,
        0x7e => ValType::I64,
        0x7d => ValType::F32,
        0x7c => ValType::F64,
        _ => ValType::V128,
    }
}

/// One module with every region of `page` (linked with each other for direct chaining), each
/// region's body made by `back`. Deterministic: the same page gives the same bytes.
pub fn page_module<A: Isa>(
    page: &PageTranslation<A>,
    link: &Linking<'_>,
    back: &mut BackHalf<'_, A>,
) -> PageModule {
    let regions: Vec<&Region<A>> = page.regions.iter().collect();
    let peers = page_peers(&page.regions, 0, link.direct_chain);
    let mut entries = Vec::new();
    let mut insns = 0;
    for (ri, r) in regions.iter().enumerate() {
        insns += r.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
        for bi in r.entries() {
            entries.push((
                (r.blocks[bi].pc & (PAGE as u64 - 1)) as u16,
                ri as u32,
                bi as u16,
            ));
        }
    }
    let bytes = build(&regions, &peers, link, back);
    PageModule {
        bytes,
        funcs: regions.len() as u32,
        entries,
        insns,
    }
}

/// One module with the single region `region` (no direct chaining): the shape of the run-time
/// translation of a single region.
pub fn region_module<A: Isa>(
    region: &Region<A>,
    link: &Linking<'_>,
    back: &mut BackHalf<'_, A>,
) -> Vec<u8> {
    build(&[region], &[BTreeMap::new()], link, back)
}

fn build<A: Isa>(
    regions: &[&Region<A>],
    peers: &[BTreeMap<u64, (u32, u32)>],
    link: &Linking<'_>,
    back: &mut BackHalf<'_, A>,
) -> Vec<u8> {
    let nhelpers = link.helpers.len() as u32;
    let nregions = regions.len() as u32;
    // Types: 0 entry, 1 step, the helpers, then fill.
    let mut types = TypeSection::new();
    types
        .ty()
        .function([ValType::I32, ValType::I32, ValType::I32], [ValType::I32]);
    types.ty().function([ValType::I64], [ValType::I32]);
    for h in link.helpers {
        types.ty().function(
            h.params.iter().map(|&b| val(b)),
            h.results.iter().map(|&b| val(b)),
        );
    }
    let t_fill = 2 + nhelpers;
    types
        .ty()
        .function([ValType::I64, ValType::I32, ValType::I32], [ValType::I32]);
    let mut imports = ImportSection::new();
    imports.import(
        abi::IMPORT_MODULE,
        abi::IMPORT_MEMORY,
        EntityType::Memory(MemoryType {
            minimum: 0,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );
    let step = alloc::format!("{}_jit_step", link.prefix);
    let fill = alloc::format!("{}_jit_fill", link.prefix);
    imports.import(abi::IMPORT_MODULE, &step, EntityType::Function(1));
    imports.import(abi::IMPORT_MODULE, &fill, EntityType::Function(t_fill));
    imports.import(
        abi::IMPORT_MODULE,
        abi::IMPORT_TABLE,
        EntityType::Table(TableType {
            element_type: RefType::FUNCREF,
            minimum: 0,
            maximum: None,
            table64: false,
            shared: false,
        }),
    );
    // Functions: 0 step, 1 fill (imported), the regions, then the helpers.
    let f_region0 = 2;
    let f_helper0 = f_region0 + nregions;
    let mut funcs = FunctionSection::new();
    for _ in 0..nregions {
        funcs.function(0);
    }
    for i in 0..nhelpers {
        funcs.function(2 + i);
    }
    let mut exports = ExportSection::new();
    for i in 0..nregions {
        exports.export(&alloc::format!("f{i}"), ExportKind::Func, f_region0 + i);
    }
    let callees = Callees {
        step: 0,
        fill: 1,
        helper: f_helper0,
        entry_type: 0,
    };
    let mut code = CodeSection::new();
    let mut hints = Vec::new();
    for (i, r) in regions.iter().enumerate() {
        let mut body = back(r, callees, &peers[i]);
        // Resolve every recorded call site (relocations, in an object).
        for &(at, callee) in &body.calls {
            let index = match callee {
                Callee::Step => 0,
                Callee::Fill => 1,
                Callee::Helper(n) => f_helper0 + u32::from(n),
                Callee::EntryType => 0,
                Callee::Region(n) => f_region0 + n,
            };
            patch_padded_leb(&mut body.body, at, index);
        }
        if !body.hints.is_empty() {
            hints.push((f_region0 + i as u32, body.hints.clone()));
        }
        code.raw(&body.body);
    }
    for h in link.helpers {
        code.raw(&h.body);
    }
    let mut m = Module::new();
    m.section(&types)
        .section(&imports)
        .section(&funcs)
        .section(&exports)
        .section(&code);
    if !hints.is_empty() {
        let mut payload = Vec::new();
        leb(&mut payload, hints.len() as u64);
        for (f, h) in &hints {
            leb(&mut payload, u64::from(*f));
            branch_hints(&mut payload, h);
        }
        m.section(&CustomSection {
            name: BRANCH_HINT_SECTION.into(),
            data: payload.into(),
        });
    }
    m.finish()
}
