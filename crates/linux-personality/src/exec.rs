//! `execve`: loading a program into a fresh address space.

use std::cell::RefCell;
use std::rc::Rc;

use wasmshim_cpu::{GuestCpu, PAGE_SIZE, Perms};
use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{auxv, signal};
use wasmshim_linux_abi::structs::KernelSigaction;

use crate::host::Host;
use crate::kernel::{Kernel, Pid, Spawn, Tid};
use crate::mm::{INTERP_BASE, PIE_BASE, SIGTRAMP, STACK_SIZE, STACK_TOP, VmaKind, page_up};
use crate::vfs::{Node, ROOT};

const MAX_SHEBANG_DEPTH: usize = 4;

/// Where `execveat` finds the program, instead of the path from the cwd.
#[derive(Clone, Debug)]
pub(crate) struct ExecStart {
    /// The directory a relative path starts from (`dirfd`).
    pub dir: crate::vfs::Ino,
    /// Follow a trailing symlink (not `AT_SYMLINK_NOFOLLOW`).
    pub follow: bool,
    /// `AT_EMPTY_PATH`: the file itself, and the name `/proc/<pid>/exe` shows.
    pub file: Option<(crate::vfs::Ino, Vec<u8>)>,
}

impl Kernel {
    /// Loads `path` into a fresh address space for `pid` and points `cpu` at its entry.
    /// On error nothing about the process has changed. On success the process gets a new
    /// address space and handler table, close-on-exec descriptors are closed, other threads
    /// are gone, and a `vfork` parent is released.
    pub(crate) fn exec_image(
        &mut self,
        pid: Pid,
        caller: Option<Tid>,
        cpu: &mut Box<dyn GuestCpu>,
        program: Spawn,
        start: Option<ExecStart>,
        host: &mut dyn Host,
    ) -> Result<(), Errno> {
        let cwd = self.procs.get(&pid).map_or(ROOT, |p| p.cwd);
        let mut start = start;
        let mut exe_name = None;
        let mut start_dir = cwd;
        let Spawn {
            path,
            mut argv,
            envp,
        } = program;
        let mut path = path;
        let mut bytes;
        let mut file;
        let mut depth = 0;
        // `/proc/self/exe` (and `/proc/<pid>/exe`) name the program itself: procfs is not in
        // the VFS, so the link is followed here (Chromium starts its helpers this way).
        let mut alias = self.proc_exe_alias(pid, &path);
        // The path the program was found by (its /proc/<pid>/exe).
        let mut looked_up;
        loop {
            let lookup = alias.take().unwrap_or_else(|| path.clone());
            looked_up = lookup.clone();
            let ino = match start.take() {
                Some(ExecStart {
                    file: Some((ino, name)),
                    ..
                }) => {
                    exe_name = Some(name);
                    ino
                }
                Some(ExecStart { dir, follow, .. }) => {
                    start_dir = dir;
                    let ino = self.vfs.resolve(dir, &lookup, follow)?;
                    if !follow && matches!(self.vfs.inode(ino).node, Node::Symlink(_)) {
                        return Err(Errno::ELOOP);
                    }
                    ino
                }
                None => {
                    start_dir = cwd;
                    exe_name = None;
                    self.vfs.resolve(cwd, &lookup, true)?
                }
            };
            let inode = self.vfs.inode(ino);
            // execve(2): EACCES "The file or a script interpreter is not a regular file"
            // (a directory too: may_open refuses to execute one).
            if !matches!(inode.node, Node::File(_)) {
                return Err(Errno::EACCES);
            }
            if inode.mode & 0o111 == 0 {
                return Err(Errno::EACCES);
            }
            file = self.vfs.file_contents(ino)?;
            bytes = self.read_head(ino)?;
            if !bytes.starts_with(b"#!") {
                break;
            }
            depth += 1;
            if depth > MAX_SHEBANG_DEPTH {
                return Err(Errno::ELOOP);
            }
            let line_end = bytes
                .iter()
                .position(|&b| b == b'\n')
                .unwrap_or(bytes.len())
                .min(256);
            let (interp, arg) = parse_shebang(&bytes[2..line_end]).ok_or(Errno::ENOEXEC)?;
            let mut new_argv = vec![interp.clone()];
            new_argv.extend(arg);
            new_argv.push(path.clone());
            new_argv.extend(argv.into_iter().skip(1));
            argv = new_argv;
            path = interp;
        }

        let elf = wasmshim_linux_elf::Elf::parse_prefix(&bytes, file.len())
            .map_err(|_| Errno::ENOEXEC)?;
        if elf.machine != self.arch.elf_machine() {
            return Err(Errno::ENOEXEC);
        }
        // A dynamically linked program: load its interpreter (`ld.so`) too and start there.
        let interp = match elf.interp() {
            Some(ipath) => {
                let ino = self
                    .vfs
                    .resolve(ROOT, ipath, true)
                    .map_err(|_| Errno::ENOENT)?;
                if !matches!(self.vfs.inode(ino).node, Node::File(_)) {
                    return Err(Errno::EACCES);
                }
                let head = self.read_head(ino)?;
                Some((ipath.to_vec(), head, self.vfs.file_contents(ino)?))
            }
            None => None,
        };
        let mem = self.frontend.new_memory();
        let mut mm = self.new_address_space(mem);
        // Segments are filled from the file on first touch.
        let image = wasmshim_linux_elf::load_with(mm.mem.as_mut(), &elf, PIE_BASE, &mut |s, e| {
            file.pages(s, e)
        })
        .map_err(|_| Errno::ENOEXEC)?;
        let canonical = match exe_name {
            Some(name) => name,
            None => self
                .vfs
                .realpath(start_dir, &looked_up)
                .unwrap_or(looked_up),
        };
        for m in &image.mappings {
            mm.record(
                m.start,
                m.end,
                m.perms,
                VmaKind::File {
                    path: canonical.clone(),
                    offset: m.offset,
                },
            );
        }
        let (entry, base) = match &interp {
            Some((ipath, ibytes, ifile)) => {
                let ielf = wasmshim_linux_elf::Elf::parse_prefix(ibytes, ifile.len())
                    .map_err(|_| Errno::ELIBBAD)?;
                if ielf.machine != self.arch.elf_machine()
                    || !ielf.is_dyn()
                    || ielf.interp().is_some()
                {
                    return Err(Errno::ELIBBAD);
                }
                let loaded = wasmshim_linux_elf::load_with(
                    mm.mem.as_mut(),
                    &ielf,
                    INTERP_BASE,
                    &mut |s, e| ifile.pages(s, e),
                )
                .map_err(|_| Errno::ELIBBAD)?;
                for m in &loaded.mappings {
                    mm.record(
                        m.start,
                        m.end,
                        m.perms,
                        VmaKind::File {
                            path: self
                                .vfs
                                .realpath(ROOT, ipath)
                                .unwrap_or_else(|_| ipath.clone()),
                            offset: m.offset,
                        },
                    );
                }
                (loaded.entry, loaded.bias)
            }
            None => (image.entry, 0),
        };
        mm.init_brk(page_up(image.brk));
        let stack_perms = if image.exec_stack {
            Perms::RWX
        } else {
            Perms::RW
        };
        mm.map(
            STACK_TOP - STACK_SIZE,
            STACK_TOP,
            stack_perms,
            VmaKind::Stack,
        )?;
        mm.map(SIGTRAMP, SIGTRAMP + PAGE_SIZE, Perms::RX, VmaKind::Vdso)?;
        mm.mem
            .poke(SIGTRAMP, self.arch.sigreturn_trampoline())
            .map_err(|_| Errno::ENOMEM)?;

        let mut random = [0u8; 16];
        host.random(&mut random);
        let aux = [
            (auxv::AT_HWCAP, self.arch.hwcap()),
            (auxv::AT_PAGESZ, PAGE_SIZE),
            (auxv::AT_CLKTCK, 100),
            (auxv::AT_PHDR, image.phdr),
            (auxv::AT_PHENT, image.phent),
            (auxv::AT_PHNUM, image.phnum),
            (auxv::AT_BASE, base),
            (auxv::AT_FLAGS, 0),
            (auxv::AT_ENTRY, image.entry),
            (auxv::AT_UID, 0),
            (auxv::AT_EUID, 0),
            (auxv::AT_GID, 0),
            (auxv::AT_EGID, 0),
            (auxv::AT_SECURE, 0),
            (auxv::AT_HWCAP2, 0),
            (auxv::AT_MINSIGSTKSZ, crate::signal::SIGFRAME_SIZE),
        ];
        let mut aux_bytes: Vec<u8> = Vec::new();
        for (k, v) in aux.iter().chain(&[(auxv::AT_NULL, 0)]) {
            aux_bytes.extend_from_slice(&k.to_le_bytes());
            aux_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let argv_refs: Vec<&[u8]> = argv.iter().map(Vec::as_slice).collect();
        let envp_refs: Vec<&[u8]> = envp.iter().map(Vec::as_slice).collect();
        let spec = wasmshim_linux_elf::StackSpec {
            argv: &argv_refs,
            envp: &envp_refs,
            execfn: &path,
            platform: self.arch.machine().as_bytes(),
            random,
            auxv: &aux,
        };
        let sp = wasmshim_linux_elf::build_stack(mm.mem.as_mut(), STACK_TOP, &spec)
            .map_err(|_| Errno::E2BIG)?;

        // Point of no return.
        let mut fresh = self.frontend.new_cpu();
        fresh.set_pc(entry);
        fresh.set_sp(sp);
        *cpu = fresh;
        let exe = canonical;
        let mut closed = Vec::new();
        let mut vfork_parent = None;
        let mut others = Vec::new();
        if let Some(p) = self.procs.get_mut(&pid) {
            // A fresh address space and handler table: never the ones shared with a vfork
            // parent or with other threads.
            p.mm = Rc::new(RefCell::new(mm));
            let comm_src = path.rsplit(|&b| b == b'/').next().unwrap_or(&path);
            p.comm = comm_src[..comm_src.len().min(15)].to_vec();
            p.exe = exe;
            p.cmdline = nul_join(&argv);
            p.environ = nul_join(&envp);
            p.auxv = aux_bytes.clone();
            p.timers = Default::default();
            // Handlers are reset to default on exec; ignored signals stay ignored.
            let mut actions = *p.sigactions.borrow();
            for a in actions.iter_mut() {
                if a.handler != signal::SIG_IGN {
                    *a = KernelSigaction::default();
                }
            }
            p.sigactions = Rc::new(RefCell::new(actions));
            // A private descriptor table (unshared from a CLONE_FILES parent), minus cloexec.
            let table = p.fds.borrow().clone();
            p.fds = Rc::new(RefCell::new(table));
            closed = p.fds.borrow_mut().take_cloexec();
            vfork_parent = p.vfork_parent.take();
            others = p.threads.iter().copied().collect();
        }
        for fd in closed {
            self.locks_fd_closed(pid, &fd.file);
            self.release_file(&fd.file);
        }
        // Other threads die; the caller's thread is the one that keeps running.
        for tid in others {
            if Some(tid) != caller {
                self.drop_thread(tid);
            }
        }
        if let Some(parent) = vfork_parent {
            self.finish_vfork(parent, pid);
        }
        Ok(())
    }
}

/// The most of a file's start that `execve` reads: a `#!` line, or the ELF headers.
const MAX_HEAD: u64 = 1 << 20;

impl Kernel {
    /// What `/proc/self/exe`, `/proc/thread-self/exe` or `/proc/<pid>/exe` (as seen by process
    /// `pid`) points at: that process's executable. `None` for any other path.
    fn proc_exe_alias(&self, pid: Pid, path: &[u8]) -> Option<Vec<u8>> {
        let rest = path.strip_prefix(b"/proc/")?.strip_suffix(b"/exe")?;
        let target = match rest {
            b"self" | b"thread-self" => pid,
            n => std::str::from_utf8(n).ok()?.parse().ok()?,
        };
        self.procs
            .get(&target)
            .map(|p| p.exe.clone())
            .filter(|e| !e.is_empty())
    }

    /// The start of file `ino`: enough for a `#!` line (the first page) or for the ELF and
    /// program headers and the interpreter path ([`wasmshim_linux_elf::header_len`]).
    fn read_head(&mut self, ino: crate::vfs::Ino) -> Result<Vec<u8>, Errno> {
        let size = self.vfs.size(ino);
        let mut want = size.min(PAGE_SIZE);
        let mut head = Vec::new();
        // Each round reads more only when the headers point further (at most three rounds:
        // the ELF header, the program headers, the interpreter path).
        for _ in 0..4 {
            head.resize(want as usize, 0);
            let n = self.vfs.read(ino, 0, &mut head)?;
            head.truncate(n);
            if head.starts_with(b"#!") {
                break;
            }
            let need = wasmshim_linux_elf::header_len(&head)
                .min(size)
                .min(MAX_HEAD);
            if need <= head.len() as u64 {
                break;
            }
            want = need;
        }
        Ok(head)
    }
}

/// Splits a `#!` line (without the `#!`) into the interpreter and its optional single
/// argument, like Linux: everything after the first run of blanks is one argument.
fn parse_shebang(line: &[u8]) -> Option<(Vec<u8>, Option<Vec<u8>>)> {
    let line = line.trim_ascii();
    let end = line
        .iter()
        .position(|&b| b == b' ' || b == b'\t')
        .unwrap_or(line.len());
    let interp = &line[..end];
    if interp.is_empty() {
        return None;
    }
    let rest = line[end..].trim_ascii();
    Some((interp.to_vec(), (!rest.is_empty()).then(|| rest.to_vec())))
}

/// `a\0b\0` style joining for `/proc/<pid>/{cmdline,environ}`.
fn nul_join(items: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for i in items {
        out.extend_from_slice(i);
        out.push(0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::parse_shebang;

    #[test]
    fn shebang() {
        assert_eq!(parse_shebang(b"/bin/sh"), Some((b"/bin/sh".to_vec(), None)));
        assert_eq!(
            parse_shebang(b" /usr/bin/env  python3 -u "),
            Some((b"/usr/bin/env".to_vec(), Some(b"python3 -u".to_vec())))
        );
        assert_eq!(parse_shebang(b"   "), None);
    }
}
