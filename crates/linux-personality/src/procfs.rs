//! `/proc`, generated from live kernel state.
//!
//! Paths under `/proc` (absolute, or relative to an open `/proc` directory) never reach the
//! VFS: [`Kernel::proc_target`] recognises them and [`Kernel::proc_lookup`] produces a node.
//! A file's contents are generated once, when it is opened (a snapshot, as most Linux proc files
//! behave for one `read` loop). `self` and `thread-self` resolve to the caller.

use std::fmt::Write as _;

use wasmshim_cpu::Perms;
use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{dtype, mode};
use wasmshim_linux_abi::structs::Stat;

use crate::fd::FileKind;
use crate::kernel::{Kernel, Pid, ThreadState, Tid};
use crate::mm::VmaKind;

/// A generated node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ProcNode {
    /// A directory and its entries (name, `d_type`).
    Dir(Vec<(Vec<u8>, u8)>),
    /// A file and its contents.
    File(Vec<u8>),
    /// A symlink and its target.
    Link(Vec<u8>),
}

/// `st_dev` of procfs.
const PROC_DEV: u64 = 0x16;

fn fnv(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn dir(names: &[&str]) -> ProcNode {
    ProcNode::Dir(
        names
            .iter()
            .map(|n| (n.as_bytes().to_vec(), dtype::DT_REG))
            .collect(),
    )
}

impl ProcNode {
    /// A `struct stat` for this node at `path`.
    pub(crate) fn stat(&self, path: &[u8]) -> Stat {
        let (mode_bits, size) = match self {
            Self::Dir(_) => (mode::S_IFDIR | 0o555, 0),
            Self::File(_) => (mode::S_IFREG | 0o444, 0),
            Self::Link(t) => (mode::S_IFLNK | 0o777, t.len() as i64),
        };
        Stat {
            dev: PROC_DEV,
            ino: fnv(path) & 0x7fff_ffff,
            mode: mode_bits,
            nlink: 1,
            blksize: 1024,
            size,
            ..Stat::default()
        }
    }
}

impl Kernel {
    /// If `(dirfd, path)` names something under `/proc`, the normalised absolute path with
    /// `self`/`thread-self` expanded.
    /// devtmpfs's stdio links: `/dev/stdin`, `/dev/stdout`, `/dev/stderr` and `/dev/fd` point
    /// into `/proc/self/fd`. The VFS holds them as symlinks (for `ls`/`readlink`); a lookup that
    /// follows them (`follow_last`, or anything below `/dev/fd/`) continues in `/proc` here,
    /// where the VFS cannot.
    pub(crate) fn dev_alias(path: &[u8], follow_last: bool) -> Option<Vec<u8>> {
        let rest = path.strip_prefix(b"/dev/")?;
        let (name, tail) = match rest.iter().position(|&b| b == b'/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, &b""[..]),
        };
        let target: &[u8] = match name {
            b"stdin" => b"/proc/self/fd/0",
            b"stdout" => b"/proc/self/fd/1",
            b"stderr" => b"/proc/self/fd/2",
            b"fd" => b"/proc/self/fd",
            _ => return None,
        };
        if tail.iter().all(|&b| b == b'/') && !follow_last {
            return None;
        }
        let mut out = target.to_vec();
        out.extend_from_slice(tail);
        Some(out)
    }

    pub(crate) fn proc_target(&self, tid: Tid, dirfd: u64, path: &[u8]) -> Option<Vec<u8>> {
        let base: Vec<u8> = if path.starts_with(b"/") {
            Vec::new()
        } else if dirfd as i32 == wasmshim_linux_abi::consts::at::AT_FDCWD {
            return None;
        } else {
            let file = self.file(tid, dirfd).ok()?;
            let f = file.borrow();
            match &f.kind {
                FileKind::ProcDir { path } => path.clone(),
                _ => return None,
            }
        };
        let mut parts: Vec<Vec<u8>> = base
            .split(|&b| b == b'/')
            .filter(|c| !c.is_empty())
            .map(<[u8]>::to_vec)
            .collect();
        for c in path.split(|&b| b == b'/').filter(|c| !c.is_empty()) {
            match c {
                b"." => {}
                b".." => {
                    parts.pop();
                }
                _ => parts.push(c.to_vec()),
            }
        }
        match parts.first().map(Vec::as_slice) {
            Some(b"proc") => {}
            // sysfs is generated here too (`crate::sysfs`).
            Some(b"sys") => {
                let mut out = Vec::new();
                for p in &parts {
                    out.push(b'/');
                    out.extend_from_slice(p);
                }
                return Some(out);
            }
            _ => return None,
        }
        let pid = self.pid_of(tid);
        if parts.get(1).map(Vec::as_slice) == Some(b"self") {
            parts[1] = pid.to_string().into_bytes();
        } else if parts.get(1).map(Vec::as_slice) == Some(b"thread-self") {
            parts.splice(
                1..2,
                [
                    pid.to_string().into_bytes(),
                    b"task".to_vec(),
                    tid.to_string().into_bytes(),
                ],
            );
        }
        let mut out = Vec::new();
        for p in &parts {
            out.push(b'/');
            out.extend_from_slice(p);
        }
        Some(out)
    }

    /// The node at a normalised `/proc/...` path.
    pub(crate) fn proc_lookup(
        &self,
        path: &[u8],
        now_ns: u64,
        wall_ns: u64,
    ) -> Result<ProcNode, Errno> {
        let text = std::str::from_utf8(path).map_err(|_| Errno::ENOENT)?;
        if text.split('/').find(|c| !c.is_empty()) == Some("sys") {
            let parts: Vec<&str> = text.split('/').filter(|c| !c.is_empty()).skip(1).collect();
            return crate::sysfs::lookup(&parts);
        }
        let parts: Vec<&str> = text.split('/').filter(|c| !c.is_empty()).skip(1).collect();
        let file = |s: String| Ok(ProcNode::File(s.into_bytes()));
        match parts.as_slice() {
            [] => {
                let mut entries: Vec<(Vec<u8>, u8)> = ["self", "thread-self", "mounts"].iter().map(|n| (n.as_bytes().to_vec(), dtype::DT_LNK)).collect();
                for n in ["cpuinfo", "meminfo", "stat", "uptime", "version", "loadavg", "filesystems"] {
                    entries.push((n.as_bytes().to_vec(), dtype::DT_REG));
                }
                entries.push((b"sys".to_vec(), dtype::DT_DIR));
                for pid in self.procs.keys() {
                    entries.push((pid.to_string().into_bytes(), dtype::DT_DIR));
                }
                Ok(ProcNode::Dir(entries))
            }
            ["self"] | ["thread-self"] => Err(Errno::ENOENT), // expanded by proc_target
            ["mounts"] => Ok(ProcNode::Link(b"self/mounts".to_vec())),
            ["cpuinfo"] => file(format!(
                "processor\t: 0\nBogoMIPS\t: 2000.00\nFeatures\t: {}\nCPU implementer\t: 0x41\nCPU architecture: 8\nCPU variant\t: 0x0\nCPU part\t: 0xd08\nCPU revision\t: 3\n\n",
                "fp asimd"
            )),
            ["meminfo"] => file(
                "MemTotal:        4194304 kB\nMemFree:         3145728 kB\nMemAvailable:    3145728 kB\nBuffers:               0 kB\nCached:                0 kB\nSwapCached:            0 kB\nActive:                0 kB\nInactive:              0 kB\nSwapTotal:             0 kB\nSwapFree:              0 kB\nShmem:                 0 kB\n"
                    .to_owned(),
            ),
            ["stat"] => file(format!(
                "cpu  0 0 0 0 0 0 0 0 0 0\ncpu0 0 0 0 0 0 0 0 0 0 0\nintr 0\nctxt 0\nbtime {}\nprocesses {}\nprocs_running 1\nprocs_blocked 0\n",
                (wall_ns - now_ns) / 1_000_000_000,
                self.procs.len()
            )),
            ["uptime"] => file(format!("{}.{:02} 0.00\n", now_ns / 1_000_000_000, (now_ns / 10_000_000) % 100)),
            ["version"] => file(format!("Linux version {} ({}) {}\n", self.cfg.kernel_release, self.cfg.kernel_builder, self.cfg.kernel_version)),
            ["loadavg"] => file(format!("0.00 0.00 0.00 1/{} {}\n", self.core.tasks.len(), self.procs.keys().max().copied().unwrap_or(1))),
            ["filesystems"] => file("nodev\tproc\nnodev\ttmpfs\n".to_owned()),
            ["sys"] => Ok(ProcNode::Dir(vec![(b"kernel".to_vec(), dtype::DT_DIR), (b"vm".to_vec(), dtype::DT_DIR)])),
            ["sys", "kernel"] => Ok(dir(&["hostname", "osrelease", "ostype", "version", "pid_max", "ngroups_max", "cap_last_cap"])),
            ["sys", "kernel", "hostname"] => file(format!("{}\n", self.cfg.hostname)),
            ["sys", "kernel", "osrelease"] => file(format!("{}\n", self.cfg.kernel_release)),
            ["sys", "kernel", "ostype"] => file("Linux\n".to_owned()),
            ["sys", "kernel", "version"] => file(format!("{}\n", self.cfg.kernel_version)),
            ["sys", "kernel", "pid_max"] => file("4194304\n".to_owned()),
            ["sys", "kernel", "ngroups_max"] => file("65536\n".to_owned()),
            ["sys", "kernel", "cap_last_cap"] => file("40\n".to_owned()),
            ["sys", "vm"] => Ok(dir(&["overcommit_memory", "max_map_count"])),
            ["sys", "vm", "overcommit_memory"] => file("0\n".to_owned()),
            ["sys", "vm", "max_map_count"] => file("65530\n".to_owned()),
            [pid, rest @ ..] => {
                let pid: Pid = pid.parse().map_err(|_| Errno::ENOENT)?;
                if !self.procs.contains_key(&pid) {
                    return Err(Errno::ENOENT);
                }
                match rest {
                    ["task"] => {
                        let p = &self.procs[&pid];
                        Ok(ProcNode::Dir(p.threads.iter().map(|t| (t.to_string().into_bytes(), dtype::DT_DIR)).collect()))
                    }
                    ["task", tid, more @ ..] => {
                        let tid: Tid = tid.parse().map_err(|_| Errno::ENOENT)?;
                        if !self.procs[&pid].threads.contains(&tid) {
                            return Err(Errno::ENOENT);
                        }
                        self.proc_pid(pid, Some(tid), more, now_ns)
                    }
                    _ => self.proc_pid(pid, None, rest, now_ns),
                }
            }
        }
    }

    /// `/proc/<pid>/...` (and `/proc/<pid>/task/<tid>/...`).
    pub(crate) fn proc_pid(
        &self,
        pid: Pid,
        tid: Option<Tid>,
        rest: &[&str],
        now_ns: u64,
    ) -> Result<ProcNode, Errno> {
        let p = &self.procs[&pid];
        let file = |s: String| Ok(ProcNode::File(s.into_bytes()));
        match rest {
            [] => {
                let mut e: Vec<(Vec<u8>, u8)> = ["exe", "cwd", "root"].iter().map(|n| (n.as_bytes().to_vec(), dtype::DT_LNK)).collect();
                for n in ["cmdline", "comm", "environ", "auxv", "maps", "mounts", "stat", "statm", "status", "limits"] {
                    e.push((n.as_bytes().to_vec(), dtype::DT_REG));
                }
                for n in ["fd", "task"] {
                    e.push((n.as_bytes().to_vec(), dtype::DT_DIR));
                }
                Ok(ProcNode::Dir(e))
            }
            ["exe"] => Ok(ProcNode::Link(p.exe.clone())),
            ["cwd"] => Ok(ProcNode::Link(self.vfs.path_of(p.cwd))),
            ["root"] => Ok(ProcNode::Link(b"/".to_vec())),
            ["cmdline"] => Ok(ProcNode::File(p.cmdline.clone())),
            ["environ"] => Ok(ProcNode::File(p.environ.clone())),
            ["auxv"] => Ok(ProcNode::File(p.auxv.clone())),
            ["comm"] => {
                let mut c = p.comm.clone();
                c.push(b'\n');
                Ok(ProcNode::File(c))
            }
            ["mounts"] => file("rootfs / rootfs rw 0 0\nproc /proc proc rw,nosuid,nodev,noexec 0 0\nsysfs /sys sysfs ro,nosuid,nodev,noexec 0 0\ntmpfs /dev/shm tmpfs rw,nosuid,nodev 0 0\ntmpfs /tmp tmpfs rw 0 0\n".to_owned()),
            ["limits"] => file(format!(
                "Limit                     Soft Limit           Hard Limit           Units     \nMax stack size            {:<21}unlimited            bytes     \nMax open files            {:<21}{:<21}files     \n",
                crate::mm::STACK_SIZE,
                crate::fd::MAX_FDS,
                crate::fd::MAX_FDS
            )),
            ["fd"] => {
                let fds = p.fds.borrow();
                Ok(ProcNode::Dir(fds.iter().map(|(fd, _)| (fd.to_string().into_bytes(), dtype::DT_LNK)).collect()))
            }
            ["fd", n] => {
                let fd: i32 = n.parse().map_err(|_| Errno::ENOENT)?;
                let file = p.fds.borrow().get(fd).map_err(|_| Errno::ENOENT)?;
                let f = file.borrow();
                Ok(ProcNode::Link(match &f.kind {
                    FileKind::Pipe { pipe, .. } => format!("pipe:[{}]", pipe.borrow().id).into_bytes(),
                    FileKind::Stdio(_) => b"/dev/console".to_vec(),
                    FileKind::Inode { ino, .. } if self.vfs.exists(*ino) && self.vfs.inode(*ino).is_dir() => self.vfs.path_of(*ino),
                    _ => f.path.clone(),
                }))
            }
            ["maps"] => {
                let mm = p.mm.borrow();
                let mut out = String::new();
                for v in mm.vmas() {
                    let perms = |on: bool, c: char| if on { c } else { '-' };
                    let (offset, name) = match &v.kind {
                        VmaKind::File { path, offset } => (*offset, String::from_utf8_lossy(path).into_owned()),
                        VmaKind::Heap => (0, "[heap]".to_owned()),
                        VmaKind::Stack => (0, "[stack]".to_owned()),
                        VmaKind::Anon => (0, String::new()),
                        VmaKind::Vdso => (0, "[vdso]".to_owned()),
                        VmaKind::Shared { offset, name, .. } => (*offset, String::from_utf8_lossy(name).into_owned()),
                    };
                    let mut line = format!(
                        "{:08x}-{:08x} {}{}{}{} {:08x} 00:00 0",
                        v.start,
                        v.end,
                        perms(v.perms.contains(Perms::R), 'r'),
                        perms(v.perms.contains(Perms::W), 'w'),
                        perms(v.perms.contains(Perms::X), 'x'),
                        if matches!(v.kind, VmaKind::Shared { .. }) { 's' } else { 'p' },
                        offset
                    );
                    if !name.is_empty() {
                        while line.len() < 73 {
                            line.push(' ');
                        }
                        line.push_str(&name);
                    }
                    line.push('\n');
                    out.push_str(&line);
                }
                file(out)
            }
            ["statm"] => {
                let pages = p.mm.borrow().vmas().map(|v| (v.end - v.start) / 4096).sum::<u64>();
                file(format!("{pages} {pages} 0 0 0 {pages} 0\n"))
            }
            ["stat"] => {
                let comm = String::from_utf8_lossy(&p.comm).into_owned();
                let threads = p.threads.len();
                let vsize: u64 = p.mm.borrow().vmas().map(|v| v.end - v.start).sum();
                let id = tid.unwrap_or(pid);
                let start = now_ns / 10_000_000;
                // The controlling terminal's device number and foreground group (`ps`'s TTY).
                let (tty_nr, tpgid) = match self.ctty_of(pid) {
                    Some(t) => {
                        let t = t.borrow();
                        (t.rdev(), t.pgrp.map_or(-1, i64::from))
                    }
                    None => (0, -1),
                };
                let mut s = format!("{id} ({comm}) R {} {} {} {tty_nr} {tpgid} 4194304 0 0 0 0 0 0 0 0 20 0 {threads} 0 {start} {vsize} {} 18446744073709551615", p.ppid, p.pgid, p.sid, vsize / 4096);
                for _ in 0..27 {
                    s.push_str(" 0");
                }
                s.push('\n');
                file(s)
            }
            ["status"] => {
                let id = tid.unwrap_or(pid);
                let t = self.core.tasks.get(&id);
                let state = match t.map(|t| t.state) {
                    Some(ThreadState::Blocked(_)) => "S (sleeping)",
                    _ => "R (running)",
                };
                let sigmask = t.map_or(0, |t| t.sigmask);
                let vsize: u64 = p.mm.borrow().vmas().map(|v| v.end - v.start).sum::<u64>() / 1024;
                let c = &p.creds;
                let groups: String = c.groups.iter().map(|g| format!("{g} ")).collect();
                let cap = |on: bool| if on { "000001ffffffffff" } else { "0000000000000000" };
                let cap_eff = cap(c.euid == 0);
                let cap_prm = cap([c.ruid, c.euid, c.suid].contains(&0));
                let mut s = String::new();
                let _ = write!(
                    s,
                    "Name:\t{}\nUmask:\t{:04o}\nState:\t{state}\nTgid:\t{pid}\nNgid:\t0\nPid:\t{id}\nPPid:\t{}\nTracerPid:\t0\nUid:\t{}\t{}\t{}\t{}\nGid:\t{}\t{}\t{}\t{}\nFDSize:\t64\nGroups:\t{groups}\nVmPeak:\t{vsize} kB\nVmSize:\t{vsize} kB\nVmRSS:\t{vsize} kB\nThreads:\t{}\nSigQ:\t0/15000\nSigPnd:\t{:016x}\nShdPnd:\t{:016x}\nSigBlk:\t{sigmask:016x}\nSigIgn:\t0000000000000000\nSigCgt:\t0000000000000000\nCapInh:\t0000000000000000\nCapPrm:\t{cap_prm}\nCapEff:\t{cap_eff}\nCapBnd:\t000001ffffffffff\nCpus_allowed:\t1\nCpus_allowed_list:\t0\n",
                    String::from_utf8_lossy(&p.comm),
                    p.umask,
                    p.ppid,
                    c.ruid,
                    c.euid,
                    c.suid,
                    c.fsuid,
                    c.rgid,
                    c.egid,
                    c.sgid,
                    c.fsgid,
                    p.threads.len(),
                    t.map_or(0, |t| t.pending.set),
                    p.pending.set,
                );
                file(s)
            }
            _ => Err(Errno::ENOENT),
        }
    }
}
