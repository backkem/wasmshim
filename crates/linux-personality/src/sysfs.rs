//! `/sys`, a static skeleton for the probes runtimes make: the CPU and NUMA
//! topology of the one virtual CPU, and transparent huge page settings. Served by the same
//! path layer as `/proc` ([`crate::procfs`]); read-only.

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::dtype;

use crate::procfs::ProcNode;

/// `MIDR_EL1` matching `/proc/cpuinfo` (implementer 0x41, architecture 0xf, part 0xd08, r0p3).
const MIDR_EL1: u64 = 0x410f_d083;

fn dir(entries: &[(&str, bool)]) -> Result<ProcNode, Errno> {
    Ok(ProcNode::Dir(
        entries
            .iter()
            .map(|(n, is_dir)| {
                (
                    n.as_bytes().to_vec(),
                    if *is_dir {
                        dtype::DT_DIR
                    } else {
                        dtype::DT_REG
                    },
                )
            })
            .collect(),
    ))
}

fn file(s: &str) -> Result<ProcNode, Errno> {
    Ok(ProcNode::File(s.as_bytes().to_vec()))
}

/// The node at `/sys/<parts...>`.
pub(crate) fn lookup(parts: &[&str]) -> Result<ProcNode, Errno> {
    const TOPOLOGY: [&str; 11] = [
        "cluster_id",
        "core_cpus",
        "core_cpus_list",
        "core_id",
        "core_siblings",
        "core_siblings_list",
        "package_cpus",
        "package_cpus_list",
        "physical_package_id",
        "thread_siblings",
        "thread_siblings_list",
    ];
    match parts {
        [] => dir(&[("devices", true), ("kernel", true)]),
        ["devices"] => dir(&[("system", true)]),
        ["devices", "system"] => dir(&[("cpu", true), ("node", true)]),
        ["devices", "system", "cpu"] => dir(&[
            ("cpu0", true),
            ("isolated", false),
            ("kernel_max", false),
            ("offline", false),
            ("online", false),
            ("possible", false),
            ("present", false),
        ]),
        [
            "devices",
            "system",
            "cpu",
            "online" | "possible" | "present",
        ] => file("0\n"),
        ["devices", "system", "cpu", "offline" | "isolated"] => file("\n"),
        ["devices", "system", "cpu", "kernel_max"] => file("511\n"),
        ["devices", "system", "cpu", "cpu0"] => dir(&[("regs", true), ("topology", true)]),
        ["devices", "system", "cpu", "cpu0", "topology"] => {
            let entries: Vec<(&str, bool)> = TOPOLOGY.iter().map(|n| (*n, false)).collect();
            dir(&entries)
        }
        ["devices", "system", "cpu", "cpu0", "topology", name] if TOPOLOGY.contains(name) => {
            // CPU lists and ids are "0", CPU masks "1".
            if name.ends_with("_list") || name.ends_with("_id") {
                file("0\n")
            } else {
                file("1\n")
            }
        }
        ["devices", "system", "cpu", "cpu0", "regs"] => dir(&[("identification", true)]),
        ["devices", "system", "cpu", "cpu0", "regs", "identification"] => {
            dir(&[("midr_el1", false), ("revidr_el1", false)])
        }
        [
            "devices",
            "system",
            "cpu",
            "cpu0",
            "regs",
            "identification",
            "midr_el1",
        ] => file(&format!("0x{MIDR_EL1:016x}\n")),
        [
            "devices",
            "system",
            "cpu",
            "cpu0",
            "regs",
            "identification",
            "revidr_el1",
        ] => file("0x0000000000000000\n"),
        ["devices", "system", "node"] => dir(&[
            ("has_cpu", false),
            ("has_memory", false),
            ("has_normal_memory", false),
            ("node0", true),
            ("online", false),
            ("possible", false),
        ]),
        [
            "devices",
            "system",
            "node",
            "online" | "possible" | "has_cpu" | "has_memory" | "has_normal_memory",
        ] => file("0\n"),
        ["devices", "system", "node", "node0"] => dir(&[("cpulist", false), ("cpumap", false)]),
        ["devices", "system", "node", "node0", "cpulist"] => file("0\n"),
        ["devices", "system", "node", "node0", "cpumap"] => file("1\n"),
        ["kernel"] => dir(&[("mm", true)]),
        ["kernel", "mm"] => dir(&[("transparent_hugepage", true)]),
        ["kernel", "mm", "transparent_hugepage"] => dir(&[
            ("defrag", false),
            ("enabled", false),
            ("hpage_pmd_size", false),
            ("shmem_enabled", false),
            ("use_zero_page", false),
        ]),
        ["kernel", "mm", "transparent_hugepage", "enabled"] => file("always [madvise] never\n"),
        ["kernel", "mm", "transparent_hugepage", "defrag"] => {
            file("always defer defer+madvise [madvise] never\n")
        }
        ["kernel", "mm", "transparent_hugepage", "hpage_pmd_size"] => file("2097152\n"),
        ["kernel", "mm", "transparent_hugepage", "use_zero_page"] => file("1\n"),
        ["kernel", "mm", "transparent_hugepage", "shmem_enabled"] => {
            file("always within_size advise [never] deny force\n")
        }
        _ => Err(Errno::ENOENT),
    }
}
