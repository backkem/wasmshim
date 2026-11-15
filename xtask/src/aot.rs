//! Ahead-of-time translation of an image's programs: the object a node links to run them as
//! translated code.
//!
//! No run is needed. Every code page of the image's AArch64 ELF files is translated, with the
//! block entries a linear sweep finds (`wasmshim_translate::aot::static_entries`): the page
//! start, the instruction after each block end, and the targets of direct branches inside the
//! page. A block the sweep missed (an indirect target in the middle of straight-line code) runs
//! in the interpreter, with the same result.
//!
//! A page is found again at run time by its content, so the object also serves any other image
//! that maps the same bytes. A code page is the 4 KiB of the file at a page-aligned offset of an
//! executable `PT_LOAD` segment (zeros past the end of the file), which is what the loader maps.
//! A file the loader would reject has no code pages.

use std::collections::BTreeSet;

use anyhow::{Result, bail};
use wasmshim_aarch64_translate::A64;
use wasmshim_aarch64_translate::aot::{PageProfile, object, translate_page};
use wasmshim_image_pack::tar::{self, Member};
use wasmshim_linux_elf::{Elf, PT_LOAD};
use wasmshim_translate::aot::{ObjectStats, PAGE, page_key, static_entries};

/// `e_machine` of an AArch64 file.
const EM_AARCH64: u16 = 183;

/// `p_flags`: the segment is executable.
const PF_X: u32 = 1;

/// What a sweep found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Swept {
    /// AArch64 ELF files with code pages.
    pub files: usize,
    /// Distinct code pages.
    pub pages: usize,
    /// Block entries in them.
    pub entries: usize,
}

/// The page-aligned file offsets of an AArch64 ELF file's executable segments; empty for
/// anything else.
fn code_pages(file: &[u8]) -> Vec<usize> {
    let Ok(elf) = Elf::parse(file) else {
        return Vec::new();
    };
    if elf.machine != EM_AARCH64 {
        return Vec::new();
    }
    let mut pages = BTreeSet::new();
    for ph in &elf.phdrs {
        if ph.p_type != PT_LOAD || ph.flags & PF_X == 0 || ph.filesz == 0 {
            continue;
        }
        let (Ok(start), Some(end)) = (
            usize::try_from(ph.offset),
            ph.offset
                .checked_add(ph.filesz)
                .and_then(|e| usize::try_from(e).ok()),
        ) else {
            continue;
        };
        pages.extend((start & !(PAGE - 1)..end.min(file.len())).step_by(PAGE));
    }
    pages.into_iter().collect()
}

/// The code pages of the files of the archive `image` whose path is in `paths` (every AArch64
/// ELF file when `paths` is empty), each once, with their statically found entries.
pub fn sweep(image: &[u8], paths: &[String]) -> Result<(Vec<PageProfile>, Swept)> {
    let wanted = |name: &str| {
        paths.is_empty()
            || paths
                .iter()
                .any(|p| p.trim_start_matches('/') == name.trim_start_matches('/'))
    };
    let mut found = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut pages = Vec::new();
    let mut swept = Swept::default();
    for (name, member) in tar::read_all(image)? {
        let name = name.trim_start_matches("./");
        let Member::File { data, .. } = member else {
            continue;
        };
        if !wanted(name) {
            continue;
        }
        found.insert(name.trim_start_matches('/').to_owned());
        let offsets = code_pages(&data);
        if offsets.is_empty() {
            continue;
        }
        swept.files += 1;
        for off in offsets {
            let mut bytes = data[off..(off + PAGE).min(data.len())].to_vec();
            bytes.resize(PAGE, 0);
            if !seen.insert(page_key(&bytes)) {
                continue;
            }
            let entries = static_entries::<A64>(&bytes);
            swept.pages += 1;
            swept.entries += entries.len();
            pages.push(PageProfile { bytes, entries });
        }
    }
    if let Some(missing) = paths
        .iter()
        .find(|p| !found.contains(p.trim_start_matches('/')))
    {
        bail!("{missing}: no such file in the image");
    }
    Ok((pages, swept))
}

/// Translates `pages` into one relocatable object (what `$LINUX_AOT_OBJ` names when the node is
/// built).
pub fn translate(pages: &[PageProfile]) -> (Vec<u8>, ObjectStats) {
    let translated: Vec<_> = pages.iter().map(translate_page).collect();
    object(&translated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasmshim_image_pack::tar::TarWriter;

    /// A minimal AArch64 ELF64 executable: one executable segment of `code` at file offset
    /// 0x1000 and address 0x41000.
    fn elf(code: &[u32]) -> Vec<u8> {
        let mut f = vec![0u8; 0x1000];
        f[..4].copy_from_slice(b"\x7fELF");
        f[4] = 2; // 64-bit
        f[5] = 1; // little-endian
        f[6] = 1; // version
        f[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
        f[18..20].copy_from_slice(&EM_AARCH64.to_le_bytes());
        f[20..24].copy_from_slice(&1u32.to_le_bytes());
        f[24..32].copy_from_slice(&0x41000u64.to_le_bytes()); // entry
        f[32..40].copy_from_slice(&64u64.to_le_bytes()); // phoff
        f[52..54].copy_from_slice(&64u16.to_le_bytes()); // ehsize
        f[54..56].copy_from_slice(&56u16.to_le_bytes()); // phentsize
        f[56..58].copy_from_slice(&1u16.to_le_bytes()); // phnum
        let bytes = (code.len() * 4) as u64;
        let ph = 64;
        f[ph..ph + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
        f[ph + 4..ph + 8].copy_from_slice(&5u32.to_le_bytes()); // R + X
        f[ph + 8..ph + 16].copy_from_slice(&0x1000u64.to_le_bytes()); // offset
        f[ph + 16..ph + 24].copy_from_slice(&0x41000u64.to_le_bytes()); // vaddr
        f[ph + 24..ph + 32].copy_from_slice(&0x41000u64.to_le_bytes()); // paddr
        f[ph + 32..ph + 40].copy_from_slice(&bytes.to_le_bytes()); // filesz
        f[ph + 40..ph + 48].copy_from_slice(&bytes.to_le_bytes()); // memsz
        f[ph + 48..ph + 56].copy_from_slice(&0x1000u64.to_le_bytes()); // align
        for word in code {
            f.extend_from_slice(&word.to_le_bytes());
        }
        f
    }

    fn image(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut tar = TarWriter::new();
        for (path, data) in files {
            let member = Member::File {
                mode: 0o755,
                data: data.to_vec(),
            };
            tar.add(path, &member).unwrap();
        }
        tar.finish()
    }

    // add x0, x0, #1; ret
    const PROGRAM: [u32; 2] = [0x9100_0400, 0xd65f_03c0];

    #[test]
    fn a_program_s_code_pages_are_found_once() {
        let program = elf(&PROGRAM);
        // The same program twice and a text file: one code page.
        let tar = image(&[
            ("bin/one", &program),
            ("bin/two", &program),
            ("etc/motd", b"hello\n"),
        ]);
        let (pages, swept) = sweep(&tar, &[]).unwrap();
        assert_eq!(swept.files, 2);
        assert_eq!(swept.pages, 1);
        assert_eq!(pages.len(), 1);
        assert_eq!(&pages[0].bytes[..4], &PROGRAM[0].to_le_bytes());
        assert_eq!(pages[0].bytes.len(), PAGE, "zeros past the end of the file");
        // The page start is always an entry.
        assert_eq!(pages[0].entries.first(), Some(&0));
        assert_eq!(swept.entries, pages[0].entries.len());
    }

    #[test]
    fn paths_select_files_and_a_missing_one_is_an_error() {
        let tar = image(&[("bin/one", &elf(&PROGRAM)), ("etc/motd", b"hello\n")]);
        let (_, swept) = sweep(&tar, &["/bin/one".into()]).unwrap();
        assert_eq!(swept.files, 1);
        // A file that is no program is found, with nothing to translate.
        let (pages, swept) = sweep(&tar, &["etc/motd".into()]).unwrap();
        assert!(pages.is_empty());
        assert_eq!(swept, Swept::default());
        let e = sweep(&tar, &["/bin/nope".into()]).unwrap_err();
        assert_eq!(e.to_string(), "/bin/nope: no such file in the image");
    }

    #[test]
    fn the_object_s_directory_finds_the_page_by_its_content() {
        let tar = image(&[("bin/one", &elf(&PROGRAM))]);
        let (pages, _) = sweep(&tar, &[]).unwrap();
        let (object, stats) = translate(&pages);
        assert_eq!(stats.pages, 1);
        assert!(stats.blocks >= 1);
        let magic = wasmshim_translate::aot::DIRECTORY_MAGIC.to_le_bytes();
        let at = object
            .windows(4)
            .position(|w| w == magic)
            .expect("a directory in the object");
        let directory = wasmshim_aarch64_translate::aot::directory(&object[at..]).unwrap();
        assert_eq!(directory.pages(), 1);
        assert!(directory.lookup_page(&pages[0].bytes).next().is_some());
    }
}
