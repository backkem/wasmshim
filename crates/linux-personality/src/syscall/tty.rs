//! Terminals in the kernel: opening `/dev/ptmx`, `/dev/pts/<n>`, `/dev/tty` and the console,
//! reading and writing both sides, the terminal ioctls, sessions and controlling terminals,
//! and the console API the embedder drives.
//!
//! Job control is basic: a session leader acquires a controlling terminal by opening one (or
//! `TIOCSCTTY`), `tcsetpgrp` picks the foreground group, and `^C`/`^\` signal it, as do window
//! size changes (`SIGWINCH`). A hang-up (the master closing) sends `SIGHUP` and `SIGCONT` to the
//! session leader; a session leader exiting sends `SIGHUP` to the foreground group. Background
//! groups are never stopped (`SIGTTIN`/`SIGTTOU` are not sent, and stop signals are ignored),
//! so `^Z` does nothing.

use std::cell::RefCell;
use std::rc::Rc;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::termios::{TCIFLUSH, TCIOFLUSH, TCOFLUSH, VMIN, VTIME};
use wasmshim_linux_abi::consts::{ioctl, open, signal};
use wasmshim_linux_abi::structs::{Termios, Winsize};

use super::SysResult;
use crate::fd::{FileKind, FileRef, open_file};
use crate::host::Host;
use crate::kernel::{Kernel, Pid, Tid, WaitKey};
use crate::objects::tty::{ConsoleDecoder, ConsoleEvent, OUTPUT_CAPACITY, Tty, TtyKind, TtyRef};
use crate::objects::{EPOLLHUP, EPOLLIN, EPOLLOUT, EPOLLRDNORM, EPOLLWRNORM};
use crate::uaccess::{read_array, write_bytes, write_u32};
use crate::vfs::{Device, Ino};

/// `sizeof(struct termios2)`: `struct termios` plus input and output speeds.
const TERMIOS2_SIZE: usize = Termios::SIZE + 8;
/// `TIOCPKT_DATA`: the status byte in front of data read from a master in packet mode.
const TIOCPKT_DATA: u8 = 0;

impl Kernel {
    // ---- the console (the embedder's side) ----

    /// Attaches a console terminal of `rows` x `cols`: the initial process gets it on fds 0, 1
    /// and 2 as its controlling terminal, instead of the host's plain streams. Call before
    /// [`Kernel::spawn`]. Input then comes from [`Kernel::console_input`]; output goes to
    /// [`Host::stdout`], already processed by the terminal (`\n` becomes `\r\n`, echo, ...).
    pub fn attach_console(&mut self, rows: u16, cols: u16) {
        let ino = self
            .vfs
            .resolve(crate::vfs::ROOT, b"/dev/console", true)
            .unwrap_or(0);
        let id = self.new_object_id();
        let ws = Winsize {
            rows,
            cols,
            xpixel: 0,
            ypixel: 0,
        };
        self.console = Some(Rc::new(RefCell::new(Tty::new(
            id,
            TtyKind::Console,
            ino,
            ws,
        ))));
    }

    /// Whether a console is attached.
    pub fn has_console(&self) -> bool {
        self.console.is_some()
    }

    /// Bytes from the terminal attached to the console (keystrokes, raw: `\r` for Enter, `^C`
    /// as 0x03). In-band window size reports (`ESC [ 48 ; rows ; cols ; ypix ; xpix t`) are
    /// taken out and applied. Echo goes to [`Host::stdout`] at once.
    pub fn console_input(&mut self, data: &[u8], host: &mut dyn Host) {
        let Some(console) = self.console.clone() else {
            return;
        };
        let now = host.monotonic_ns();
        for event in ConsoleDecoder.decode(data) {
            match event {
                ConsoleEvent::Input(bytes) => {
                    let fx = console.borrow_mut().input(&bytes, now);
                    if let Some(s) = fx.signal {
                        self.signal_tty_pgrp(&console, s);
                    }
                }
                ConsoleEvent::Resize(ws) => self.set_winsize(&console, ws),
            }
        }
        let id = console.borrow().id;
        self.object_changed(id);
        self.flush_console(host);
    }

    /// The console's input ended (the terminal detached): a partial line is handed over, and
    /// reads return end of file once everything typed is read.
    pub fn console_eof(&mut self) {
        let Some(console) = self.console.clone() else {
            return;
        };
        let id = {
            let mut c = console.borrow_mut();
            if !c.input_eof {
                // Like ^D: a partial line becomes readable.
                let eof = c.termios.cc[wasmshim_linux_abi::consts::termios::VEOF];
                if c.canonical() && eof != 0 {
                    c.input(&[eof], 0);
                }
                c.input_eof = true;
            }
            c.generation += 1;
            c.id
        };
        self.object_changed(id);
    }

    /// The console's window size changed: `TIOCGWINSZ` reports it, and the foreground group
    /// gets `SIGWINCH`.
    pub fn console_resize(&mut self, rows: u16, cols: u16) {
        if let Some(console) = self.console.clone() {
            let ws = Winsize {
                rows,
                cols,
                xpixel: 0,
                ypixel: 0,
            };
            self.set_winsize(&console, ws);
        }
    }

    /// Enables the console's "a reader is waiting for input" hint ([`crate::READING_HINT`] on
    /// stdout whenever a read, `poll`, `select` or `epoll_wait` blocks on the console with
    /// nothing typed; once per input). Off by default.
    pub fn set_console_hints(&mut self, on: bool) {
        if let Some(c) = &self.console {
            c.borrow_mut().hints = on;
        }
    }

    /// A thread is about to block waiting to read `file`: if it is the console, remember it
    /// for the hint.
    pub(crate) fn note_input_wait(&self, file: &FileRef) {
        if let FileKind::Tty {
            tty, master: false, ..
        } = &file.borrow().kind
            && let Ok(mut t) = tty.try_borrow_mut()
            && t.kind == TtyKind::Console
        {
            t.wants_input = true;
        }
    }

    /// Hands console output to the host.
    pub(crate) fn flush_console(&mut self, host: &mut dyn Host) {
        if let Some(console) = &self.console {
            let mut c = console.borrow_mut();
            if c.wants_input {
                c.wants_input = false;
                if c.hints && !c.hint_sent {
                    c.hint_sent = true;
                    c.out.extend(crate::objects::tty::READING_HINT);
                }
            }
            if !c.out.is_empty() {
                let data: Vec<u8> = c.out.drain(..).collect();
                drop(c);
                host.stdout(&data);
            }
        }
    }

    // ---- sessions ----

    /// The controlling terminal of process `pid`: the terminal of its session.
    pub(crate) fn ctty_of(&self, pid: Pid) -> Option<TtyRef> {
        let sid = self.procs.get(&pid)?.sid;
        self.console
            .iter()
            .chain(self.ptys.values())
            .find(|t| t.borrow().session == Some(sid))
            .cloned()
    }

    /// Sends `s` to every process in group `pgid`.
    pub(crate) fn kill_pgrp(&mut self, pgid: Pid, s: u32) {
        let members: Vec<Pid> = self
            .procs
            .iter()
            .filter(|(_, p)| p.pgid == pgid)
            .map(|(&q, _)| q)
            .collect();
        for p in members {
            let _ = self.send_to_process(p, s);
        }
    }

    fn signal_tty_pgrp(&mut self, tty: &TtyRef, s: u32) {
        let pgrp = tty.borrow().pgrp;
        if let Some(pg) = pgrp {
            self.kill_pgrp(pg, s);
        }
    }

    fn set_winsize(&mut self, tty: &TtyRef, ws: Winsize) {
        let changed = {
            let mut t = tty.borrow_mut();
            let changed = t.winsize != ws;
            t.winsize = ws;
            changed
        };
        if changed {
            self.signal_tty_pgrp(tty, signal::SIGWINCH);
        }
    }

    /// A session leader opening a terminal without `O_NOCTTY` acquires it, if neither has one.
    fn maybe_acquire_ctty(&mut self, pid: Pid, tty: &TtyRef, flags: u64) {
        if flags & open::O_NOCTTY != 0 {
            return;
        }
        let Some(p) = self.procs.get(&pid) else {
            return;
        };
        if p.sid != pid || self.ctty_of(pid).is_some() || tty.borrow().session.is_some() {
            return;
        }
        let pgid = p.pgid;
        let mut t = tty.borrow_mut();
        t.session = Some(pid);
        t.pgrp = Some(pgid);
    }

    /// Process `pid`, a session leader, exited: its terminal loses the session, and the
    /// foreground group gets `SIGHUP` (Linux's `disassociate_ctty(1)` for a pty).
    pub(crate) fn session_leader_exited(&mut self, pid: Pid) {
        let tty = self
            .console
            .iter()
            .chain(self.ptys.values())
            .find(|t| t.borrow().session == Some(pid))
            .cloned();
        if let Some(tty) = tty {
            let pgrp = {
                let mut t = tty.borrow_mut();
                t.session = None;
                t.pgrp.take()
            };
            if let Some(pg) = pgrp {
                self.kill_pgrp(pg, signal::SIGHUP);
                self.kill_pgrp(pg, signal::SIGCONT);
            }
        }
    }

    /// The master side went away: the slave hangs up. Its session leader gets `SIGHUP` and
    /// `SIGCONT`; slave reads then see end of file and writes fail with `EIO`.
    fn hangup(&mut self, tty: &TtyRef) {
        let (session, id) = {
            let mut t = tty.borrow_mut();
            t.hung_up = true;
            t.pgrp = None;
            t.generation += 1;
            (t.session.take(), t.id)
        };
        if let Some(sid) = session {
            let _ = self.send_to_process(sid, signal::SIGHUP);
            let _ = self.send_to_process(sid, signal::SIGCONT);
        }
        self.object_changed(id);
    }

    // ---- opening ----

    /// Opens a terminal device node. `Ok(None)`: not a terminal here (`/dev/console` without
    /// an attached console keeps meaning the host's plain stdio).
    pub(crate) fn tty_open(
        &mut self,
        tid: Tid,
        device: Device,
        ino: Ino,
        flags: u64,
        now: wasmshim_linux_abi::structs::Timespec,
    ) -> Result<Option<FileKind>, Errno> {
        let pid = self.pid_of(tid);
        let (tty, master) = match device {
            Device::Ptmx => {
                let n = (0u32..)
                    .find(|n| !self.ptys.contains_key(n))
                    .ok_or(Errno::ENOSPC)?;
                let pts = self.vfs.add_pts(n, now)?;
                let id = self.new_object_id();
                let tty = Rc::new(RefCell::new(Tty::new(
                    id,
                    TtyKind::Pty(n),
                    pts,
                    Winsize::default(),
                )));
                self.ptys.insert(n, tty.clone());
                (tty, true)
            }
            Device::Pts(n) => {
                let tty = self.ptys.get(&n).cloned().ok_or(Errno::EIO)?;
                {
                    let t = tty.borrow();
                    if t.locked || t.hung_up || t.masters == 0 {
                        return Err(Errno::EIO);
                    }
                }
                self.maybe_acquire_ctty(pid, &tty, flags);
                (tty, false)
            }
            Device::Tty => (self.ctty_of(pid).ok_or(Errno::ENXIO)?, false),
            Device::Console => match self.console.clone() {
                Some(c) => {
                    self.maybe_acquire_ctty(pid, &c, flags);
                    (c, false)
                }
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        if !master {
            let mut t = tty.borrow_mut();
            t.slaves += 1;
            t.slave_closed = false;
        }
        Ok(Some(FileKind::Tty { tty, master, ino }))
    }

    /// `TIOCGPTPEER`: a new descriptor for the slave of master `tty`.
    fn open_peer(&mut self, tid: Tid, tty: &TtyRef, flags: u64) -> SysResult {
        let (n, ino) = {
            let t = tty.borrow();
            if t.locked || t.hung_up {
                return Err(Errno::EIO);
            }
            match t.kind {
                TtyKind::Pty(n) => (n, t.ino),
                TtyKind::Console => return Err(Errno::ENOTTY),
            }
        };
        let pid = self.pid_of(tid);
        self.maybe_acquire_ctty(pid, tty, flags);
        {
            let mut t = tty.borrow_mut();
            t.slaves += 1;
            t.slave_closed = false;
        }
        self.vfs.retain(ino);
        let file = open_file(
            FileKind::Tty {
                tty: tty.clone(),
                master: false,
                ino,
            },
            flags & (open::O_ACCMODE | open::O_NONBLOCK),
            format!("/dev/pts/{n}").as_bytes(),
        );
        let fd = self
            .fds(tid)
            .borrow_mut()
            .insert(file, flags & open::O_CLOEXEC != 0, 0)?;
        Ok(fd as u64)
    }

    /// The last reference to a terminal description closed.
    pub(crate) fn tty_closed(&mut self, tty: &TtyRef, master: bool, ino: Ino) {
        self.vfs.release(ino);
        let (kind, masters, slaves, id) = {
            let mut t = tty.borrow_mut();
            if master {
                t.masters = t.masters.saturating_sub(1);
            } else {
                t.slaves = t.slaves.saturating_sub(1);
                if t.slaves == 0 {
                    t.slave_closed = true;
                }
            }
            t.generation += 1;
            (t.kind, t.masters, t.slaves, t.id)
        };
        if let TtyKind::Pty(n) = kind {
            if master && masters == 0 {
                self.vfs.remove_pts(n);
                self.hangup(tty);
            }
            if masters == 0 && slaves == 0 {
                self.ptys.remove(&n);
            }
        }
        self.object_changed(id);
    }

    /// A reopen through `/proc/<pid>/fd/<n>` (a new description of the same side).
    pub(crate) fn tty_reopened(&mut self, tty: &TtyRef, master: bool, ino: Ino) {
        self.vfs.retain(ino);
        let mut t = tty.borrow_mut();
        if master {
            t.masters += 1;
        } else {
            t.slaves += 1;
        }
    }

    // ---- reading and writing ----

    pub(crate) fn tty_read(
        &mut self,
        tty: &TtyRef,
        master: bool,
        buf: &mut [u8],
        nonblock: bool,
        now: u64,
    ) -> Result<usize, Errno> {
        let mut t = tty.borrow_mut();
        let id = t.id;
        if master {
            if t.out.is_empty() {
                if t.slave_closed {
                    return Err(Errno::EIO);
                }
                if nonblock {
                    return Err(Errno::EAGAIN);
                }
                drop(t);
                return Err(self.wait_for(WaitKey::Object(id), None, None));
            }
            let mut n = 0;
            if t.packet && !buf.is_empty() {
                buf[0] = TIOCPKT_DATA;
                n = 1;
            }
            let take = (buf.len() - n).min(t.out.len());
            for (dst, src) in buf[n..].iter_mut().zip(t.out.drain(..take)) {
                *dst = src;
            }
            t.generation += 1;
            drop(t);
            self.object_changed(id);
            return Ok(n + take);
        }
        if buf.is_empty() {
            return Ok(0);
        }
        if t.hung_up {
            return Ok(0);
        }
        let mut deadline = None;
        let mut timeout_result = None;
        if t.canonical() {
            if let Some(data) = t.read(buf.len()) {
                buf[..data.len()].copy_from_slice(&data);
                drop(t);
                self.object_changed(id);
                return Ok(data.len());
            }
        } else {
            let vmin = usize::from(t.termios.cc[VMIN]);
            let vtime = u64::from(t.termios.cc[VTIME]) * 100_000_000;
            let avail = t.raw_available();
            let enough = avail > 0 && (avail >= vmin.min(buf.len()) || vmin == 0);
            let interbyte = avail > 0 && vtime > 0 && now >= t.last_input.saturating_add(vtime);
            if enough || interbyte {
                let data = t.read(buf.len()).unwrap_or_default();
                buf[..data.len()].copy_from_slice(&data);
                drop(t);
                self.object_changed(id);
                return Ok(data.len());
            }
            match (vmin, vtime, avail) {
                (0, 0, _) => return Ok(0),
                // Wait up to VTIME for the first byte; then return what there is (maybe 0).
                (0, _, _) => {
                    deadline = Some(now.saturating_add(vtime));
                    timeout_result = Some(0);
                }
                // MIN with an inter-byte timer that runs from the last byte received.
                (_, t2, a) if t2 > 0 && a > 0 => {
                    deadline = Some(t.last_input.saturating_add(t2));
                }
                _ => {}
            }
        }
        if t.input_eof {
            return Ok(0);
        }
        if nonblock {
            return Err(Errno::EAGAIN);
        }
        t.wants_input = true;
        drop(t);
        Err(self.wait_for(WaitKey::Object(id), deadline, timeout_result))
    }

    pub(crate) fn tty_write(
        &mut self,
        tty: &TtyRef,
        master: bool,
        data: &[u8],
        nonblock: bool,
        now: u64,
    ) -> Result<usize, Errno> {
        if master {
            let (fx, id) = {
                let mut t = tty.borrow_mut();
                (t.input(data, now), t.id)
            };
            if let Some(s) = fx.signal {
                self.signal_tty_pgrp(tty, s);
            }
            self.object_changed(id);
            return Ok(data.len());
        }
        let mut t = tty.borrow_mut();
        let id = t.id;
        if t.hung_up {
            return Err(Errno::EIO);
        }
        let n = match t.kind {
            TtyKind::Console => data.len(),
            TtyKind::Pty(_) => {
                if t.masters == 0 {
                    return Err(Errno::EIO);
                }
                let space = OUTPUT_CAPACITY.saturating_sub(t.out.len());
                if space == 0 {
                    if nonblock {
                        return Err(Errno::EAGAIN);
                    }
                    drop(t);
                    return Err(self.wait_for(WaitKey::Object(id), None, None));
                }
                data.len().min(space)
            }
        };
        t.output(&data[..n]);
        t.generation += 1;
        drop(t);
        self.object_changed(id);
        Ok(n)
    }

    /// Readiness of a terminal description as `EPOLL*` bits.
    pub(crate) fn tty_events(t: &Tty, master: bool) -> (u32, u64) {
        let mut ev = 0;
        if master {
            if !t.out.is_empty() {
                ev |= EPOLLIN | EPOLLRDNORM;
            }
            if t.slave_closed {
                ev |= EPOLLHUP;
            }
            ev |= EPOLLOUT | EPOLLWRNORM;
            return (ev, t.generation);
        }
        if t.hung_up {
            return (
                EPOLLIN | EPOLLRDNORM | EPOLLOUT | EPOLLWRNORM | EPOLLHUP,
                t.generation,
            );
        }
        let readable = if t.canonical() {
            t.read_ready()
        } else {
            // Like n_tty_poll: with VMIN > 0 and no VTIME, MIN bytes must be there.
            let vmin = usize::from(t.termios.cc[VMIN]);
            let need = if t.termios.cc[VTIME] == 0 && vmin > 0 {
                vmin
            } else {
                1
            };
            t.raw_available() >= need
        };
        if readable || t.input_eof {
            ev |= EPOLLIN | EPOLLRDNORM;
        }
        if matches!(t.kind, TtyKind::Console) || t.out.len() < OUTPUT_CAPACITY {
            ev |= EPOLLOUT | EPOLLWRNORM;
        }
        (ev, t.generation)
    }

    // ---- ioctls ----

    #[allow(clippy::too_many_lines)]
    pub(crate) fn tty_ioctl(&mut self, tid: Tid, file: &FileRef, req: u64, arg: u64) -> SysResult {
        let FileKind::Tty { tty, master, .. } = file.borrow().kind.clone() else {
            return Err(Errno::ENOTTY);
        };
        let pid = self.pid_of(tid);
        let mm = self.mm(tid);
        let is_ctty = self.ctty_of(pid).is_some_and(|c| Rc::ptr_eq(&c, &tty));
        match req {
            ioctl::TCGETS => {
                let t = tty.borrow().termios;
                write_bytes(mm.borrow_mut().mem.as_mut(), arg, &t.to_bytes())?;
                Ok(0)
            }
            ioctl::TCGETS2 => {
                let (t, (ispeed, ospeed)) = {
                    let b = tty.borrow();
                    (b.termios, b.speeds)
                };
                let mut out = [0u8; TERMIOS2_SIZE];
                out[..Termios::SIZE].copy_from_slice(&t.to_bytes());
                out[36..40].copy_from_slice(&ispeed.to_le_bytes());
                out[40..44].copy_from_slice(&ospeed.to_le_bytes());
                write_bytes(mm.borrow_mut().mem.as_mut(), arg, &out)?;
                Ok(0)
            }
            ioctl::TCSETS
            | ioctl::TCSETSW
            | ioctl::TCSETSF
            | ioctl::TCSETS2
            | ioctl::TCSETSW2
            | ioctl::TCSETSF2 => {
                let two = matches!(req, ioctl::TCSETS2 | ioctl::TCSETSW2 | ioctl::TCSETSF2);
                let raw = crate::uaccess::read_bytes(
                    mm.borrow_mut().mem.as_mut(),
                    arg,
                    if two { TERMIOS2_SIZE } else { Termios::SIZE },
                )?;
                let mut b = [0u8; Termios::SIZE];
                b.copy_from_slice(&raw[..Termios::SIZE]);
                let new = Termios::from_bytes(&b);
                let id = {
                    let mut t = tty.borrow_mut();
                    if matches!(req, ioctl::TCSETSF | ioctl::TCSETSF2) {
                        t.flush_input();
                    }
                    t.set_termios(new);
                    if two {
                        let speed = |at: usize| {
                            u32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]])
                        };
                        t.speeds = (speed(36), speed(40));
                    }
                    t.id
                };
                self.object_changed(id);
                Ok(0)
            }
            ioctl::TIOCGWINSZ => {
                let ws = tty.borrow().winsize;
                write_bytes(mm.borrow_mut().mem.as_mut(), arg, &ws.to_bytes())?;
                Ok(0)
            }
            ioctl::TIOCSWINSZ => {
                let ws = Winsize::from_bytes(&read_array(mm.borrow_mut().mem.as_mut(), arg)?);
                self.set_winsize(&tty, ws);
                Ok(0)
            }
            ioctl::TIOCGPGRP => {
                if !master && !is_ctty {
                    return Err(Errno::ENOTTY);
                }
                let pg = tty.borrow().pgrp.unwrap_or(0);
                write_u32(mm.borrow_mut().mem.as_mut(), arg, pg)?;
                Ok(0)
            }
            ioctl::TIOCSPGRP => {
                let sid = self.procs.get(&pid).map(|p| p.sid);
                if !is_ctty || tty.borrow().session != sid {
                    return Err(Errno::ENOTTY);
                }
                let pg = mm.borrow().mem.read_u32(arg).map_err(|_| Errno::EFAULT)? as i32;
                if pg < 0 {
                    return Err(Errno::EINVAL);
                }
                let pg = pg as u32;
                let mut members = self.procs.values().filter(|p| p.pgid == pg).peekable();
                if members.peek().is_none() {
                    return Err(Errno::ESRCH);
                }
                if members.any(|p| Some(p.sid) != sid) {
                    return Err(Errno::EPERM);
                }
                tty.borrow_mut().pgrp = Some(pg);
                Ok(0)
            }
            ioctl::TIOCGSID => {
                if !master && !is_ctty {
                    return Err(Errno::ENOTTY);
                }
                let sid = tty.borrow().session.ok_or(Errno::ENOTTY)?;
                write_u32(mm.borrow_mut().mem.as_mut(), arg, sid)?;
                Ok(0)
            }
            ioctl::TIOCSCTTY => {
                let Some(p) = self.procs.get(&pid) else {
                    return Err(Errno::ESRCH);
                };
                if is_ctty {
                    return Ok(0);
                }
                if p.sid != pid || self.ctty_of(pid).is_some() {
                    return Err(Errno::EPERM);
                }
                let pgid = p.pgid;
                let owner = tty.borrow().session;
                if owner.is_some() {
                    // Stealing another session's terminal needs `arg == 1` (and root).
                    if arg != 1 {
                        return Err(Errno::EPERM);
                    }
                }
                let mut t = tty.borrow_mut();
                t.session = Some(pid);
                t.pgrp = Some(pgid);
                Ok(0)
            }
            ioctl::TIOCNOTTY => {
                if !is_ctty {
                    return Err(Errno::ENOTTY);
                }
                if self.procs.get(&pid).is_some_and(|p| p.sid == pid) {
                    self.session_leader_exited(pid);
                }
                Ok(0)
            }
            ioctl::FIONREAD => {
                let n = {
                    let t = tty.borrow();
                    if master {
                        t.out.len()
                    } else {
                        t.readable_bytes()
                    }
                };
                write_u32(mm.borrow_mut().mem.as_mut(), arg, n as u32)?;
                Ok(0)
            }
            ioctl::TIOCOUTQ => {
                let n = {
                    let t = tty.borrow();
                    if master { 0 } else { t.out.len() }
                };
                write_u32(mm.borrow_mut().mem.as_mut(), arg, n as u32)?;
                Ok(0)
            }
            ioctl::TCFLSH => {
                let id = {
                    let mut t = tty.borrow_mut();
                    // On a master the queues are seen from the other side.
                    let (input, output) = match (arg, master) {
                        (TCIFLUSH, false) | (TCOFLUSH, true) => (true, false),
                        (TCOFLUSH, false) | (TCIFLUSH, true) => (false, true),
                        (TCIOFLUSH, _) => (true, true),
                        _ => return Err(Errno::EINVAL),
                    };
                    if input {
                        t.flush_input();
                    }
                    if output {
                        t.out.clear();
                    }
                    t.id
                };
                self.object_changed(id);
                Ok(0)
            }
            ioctl::TCXONC => match arg {
                0..=3 => Ok(0),
                _ => Err(Errno::EINVAL),
            },
            ioctl::TCSBRK | ioctl::TCSBRKP => Ok(0),
            ioctl::TIOCSTI => {
                let c = crate::uaccess::read_bytes(mm.borrow_mut().mem.as_mut(), arg, 1)?;
                let now = self.core.slice_now;
                let (fx, id) = {
                    let mut t = tty.borrow_mut();
                    (t.input(&c, now), t.id)
                };
                if let Some(s) = fx.signal {
                    self.signal_tty_pgrp(&tty, s);
                }
                self.object_changed(id);
                Ok(0)
            }
            ioctl::TIOCEXCL | ioctl::TIOCNXCL => {
                tty.borrow_mut().exclusive = req == ioctl::TIOCEXCL;
                Ok(0)
            }
            ioctl::TIOCGEXCL => {
                let x = u32::from(tty.borrow().exclusive);
                write_u32(mm.borrow_mut().mem.as_mut(), arg, x)?;
                Ok(0)
            }
            ioctl::TIOCGETD => {
                write_u32(
                    mm.borrow_mut().mem.as_mut(),
                    arg,
                    wasmshim_linux_abi::consts::termios::N_TTY,
                )?;
                Ok(0)
            }
            ioctl::TIOCSETD => {
                let d = mm.borrow().mem.read_u32(arg).map_err(|_| Errno::EFAULT)?;
                if d == wasmshim_linux_abi::consts::termios::N_TTY {
                    Ok(0)
                } else {
                    Err(Errno::EINVAL)
                }
            }
            ioctl::TIOCGDEV => {
                let dev = tty.borrow().rdev() as u32;
                write_u32(mm.borrow_mut().mem.as_mut(), arg, dev)?;
                Ok(0)
            }
            ioctl::TIOCCONS => Ok(0),
            ioctl::TIOCVHANGUP => {
                self.hangup(&tty);
                Ok(0)
            }
            // Master-only requests.
            ioctl::TIOCGPTN if master => match tty.borrow().kind {
                TtyKind::Pty(n) => {
                    write_u32(mm.borrow_mut().mem.as_mut(), arg, n)?;
                    Ok(0)
                }
                TtyKind::Console => Err(Errno::ENOTTY),
            },
            ioctl::TIOCSPTLCK if master => {
                let lock = mm.borrow().mem.read_u32(arg).map_err(|_| Errno::EFAULT)? != 0;
                tty.borrow_mut().locked = lock;
                Ok(0)
            }
            ioctl::TIOCGPTLCK if master => {
                let l = u32::from(tty.borrow().locked);
                write_u32(mm.borrow_mut().mem.as_mut(), arg, l)?;
                Ok(0)
            }
            ioctl::TIOCPKT if master => {
                let on = mm.borrow().mem.read_u32(arg).map_err(|_| Errno::EFAULT)? != 0;
                tty.borrow_mut().packet = on;
                Ok(0)
            }
            ioctl::TIOCGPKT if master => {
                let p = u32::from(tty.borrow().packet);
                write_u32(mm.borrow_mut().mem.as_mut(), arg, p)?;
                Ok(0)
            }
            ioctl::TIOCSIG if master => {
                let s = u32::try_from(arg).map_err(|_| Errno::EINVAL)?;
                if s == 0 || s > signal::NSIG {
                    return Err(Errno::EINVAL);
                }
                self.signal_tty_pgrp(&tty, s);
                Ok(0)
            }
            ioctl::TIOCGPTPEER if master => self.open_peer(tid, &tty, arg),
            _ => Err(Errno::ENOTTY),
        }
    }
}
