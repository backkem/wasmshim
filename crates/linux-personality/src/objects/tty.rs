//! Terminals: the line discipline (Linux's `n_tty`), pseudo-terminal pairs and the console.
//!
//! A [`Tty`] is one terminal: termios, window size, its session and foreground process group,
//! the input queue the slave reads, and the output queue its master reads. Two kinds exist:
//!
//! * a **pseudo-terminal** (`/dev/ptmx` + `/dev/pts/N`): the master is a file descriptor in the
//!   guest (telnetd, sshd, `script`, tmux). Master writes are typed input; slave writes, after
//!   output processing, queue for the master to read.
//! * the **console**: the master is the embedder (a terminal pane attached to the node). Its
//!   input arrives through [`crate::Kernel::console_input`], and slave output goes straight to
//!   the host's stdout.
//!
//! This module is pure: it processes bytes and reports what the kernel must do (signals to
//! send). Blocking, waking, sessions and file descriptors live in `syscall/tty.rs`.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use wasmshim_linux_abi::consts::termios::*;
use wasmshim_linux_abi::structs::{Termios, Winsize};

use crate::kernel::Pid;

/// Largest line in canonical mode (Linux's `N_TTY_BUF_SIZE - 1`).
pub(crate) const MAX_CANON: usize = 4095;
/// Input bytes a terminal holds before a master write is cut short.
pub(crate) const INPUT_CAPACITY: usize = 65536;
/// Output bytes a pty holds for its master before slave writes block.
pub(crate) const OUTPUT_CAPACITY: usize = 65536;

/// The termios of a fresh Linux pseudo-terminal slave (`tty_std_termios` with devpts's
/// `B38400 | CS8 | CREAD`): cooked, echoing, `^C` interrupts, `\n` becomes `\r\n` on output.
pub(crate) fn default_termios() -> Termios {
    let mut cc = [0u8; 19];
    cc[VINTR] = 0x03;
    cc[VQUIT] = 0x1c;
    cc[VERASE] = 0x7f;
    cc[VKILL] = 0x15;
    cc[VEOF] = 0x04;
    cc[VMIN] = 1;
    cc[VSTART] = 0x11;
    cc[VSTOP] = 0x13;
    cc[VSUSP] = 0x1a;
    cc[VREPRINT] = 0x12;
    cc[VDISCARD] = 0x0f;
    cc[VWERASE] = 0x17;
    cc[VLNEXT] = 0x16;
    Termios {
        iflag: ICRNL | IXON,
        oflag: OPOST | ONLCR,
        cflag: B38400 | CS8 | CREAD,
        lflag: ISIG | ICANON | ECHO | ECHOE | ECHOK | ECHOCTL | ECHOKE | IEXTEN,
        line: 0,
        cc,
    }
}

/// Which terminal this is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TtyKind {
    /// `/dev/pts/<n>`, with a master descriptor in the guest.
    Pty(u32),
    /// The node's console (`/dev/console`), whose master is the embedder.
    Console,
}

/// What processing input asked the kernel to do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct InputEffects {
    /// A signal for the foreground process group (`^C`, `^\`, `^Z`).
    pub signal: Option<u32>,
}

/// One terminal.
#[derive(Debug)]
pub(crate) struct Tty {
    /// Object id: the wait key of everything that blocks on this terminal.
    pub id: u64,
    /// Change generation (edge-triggered epoll).
    pub generation: u64,
    pub kind: TtyKind,
    /// The slave's VFS inode (`/dev/pts/<n>` or `/dev/console`).
    pub ino: u64,
    pub termios: Termios,
    /// `termios2`'s input and output speeds.
    pub speeds: (u32, u32),
    pub winsize: Winsize,
    /// The session this is the controlling terminal of.
    pub session: Option<Pid>,
    /// The foreground process group.
    pub pgrp: Option<Pid>,
    /// `TIOCSPTLCK`: the slave cannot be opened while locked (a fresh pty is locked).
    pub locked: bool,
    /// `TIOCEXCL`.
    pub exclusive: bool,
    /// `TIOCPKT` on the master.
    pub packet: bool,
    /// Open master descriptions (a pty), or 1 while the console is attached.
    pub masters: u32,
    /// Open slave descriptions.
    pub slaves: u32,
    /// The last slave description closed (a master read then fails with `EIO`).
    pub slave_closed: bool,
    /// Hung up: the master went away (slave reads see end of file, writes `EIO`).
    pub hung_up: bool,
    /// The console's input ended: once drained, reads return end of file.
    pub input_eof: bool,
    /// Output stopped by `^S` (`IXON`).
    pub stopped: bool,
    /// Input bytes readable by the slave.
    rq: VecDeque<u8>,
    /// Canonical mode: lengths of the complete lines at the front of `rq` (0 = end of file).
    lines: VecDeque<usize>,
    /// Canonical mode: the line being edited.
    edit: Vec<u8>,
    /// The output column where `edit` starts (for erasing tabs).
    edit_column: u32,
    /// The next input byte is taken literally (`^V`).
    lnext: bool,
    /// Bytes for the master to read (a pty) or for the host (the console; drained at once).
    pub out: VecDeque<u8>,
    /// The output column (for tabs and `ONOCR`).
    column: u32,
    /// Monotonic time of the last input byte (`VTIME`'s inter-byte timer).
    pub last_input: u64,
    /// Console only: emit [`READING_HINT`] when a reader blocks with nothing to read.
    pub hints: bool,
    /// A reader blocked on this terminal with nothing to read (for the hint).
    pub wants_input: bool,
    /// The hint went out since the last input.
    pub hint_sent: bool,
}

/// A shared handle to a terminal.
pub(crate) type TtyRef = Rc<RefCell<Tty>>;

/// The console's optional "a reader is waiting for input" hint: a private OSC
/// sequence on the output, which terminals ignore (xterm.js: `registerOscHandler(5379, ...)`).
pub const READING_HINT: &[u8] = b"\x1b]5379;reading\x07";

fn is_ctrl(c: u8) -> bool {
    c < 0x20 || c == 0x7f
}

impl Tty {
    /// A terminal with Linux's default settings.
    pub fn new(id: u64, kind: TtyKind, ino: u64, winsize: Winsize) -> Self {
        Self {
            id,
            generation: 0,
            kind,
            ino,
            termios: default_termios(),
            speeds: (38400, 38400),
            winsize,
            session: None,
            pgrp: None,
            locked: matches!(kind, TtyKind::Pty(_)),
            exclusive: false,
            packet: false,
            masters: 1,
            slaves: 0,
            slave_closed: false,
            hung_up: false,
            input_eof: false,
            stopped: false,
            rq: VecDeque::new(),
            lines: VecDeque::new(),
            edit: Vec::new(),
            edit_column: 0,
            lnext: false,
            out: VecDeque::new(),
            column: 0,
            last_input: 0,
            hints: false,
            wants_input: false,
            hint_sent: false,
        }
    }

    fn l(&self, flag: u32) -> bool {
        self.termios.lflag & flag != 0
    }

    fn i(&self, flag: u32) -> bool {
        self.termios.iflag & flag != 0
    }

    fn cc(&self, index: usize) -> u8 {
        self.termios.cc[index]
    }

    /// `c` is the (enabled) special character at `index`.
    fn is(&self, c: u8, index: usize) -> bool {
        let v = self.cc(index);
        v != 0 && v == c
    }

    pub fn canonical(&self) -> bool {
        self.l(ICANON)
    }

    /// Bytes `FIONREAD` reports.
    pub fn readable_bytes(&self) -> usize {
        self.rq.len()
    }

    /// Whether a read can return something now (data, or end of file).
    pub fn read_ready(&self) -> bool {
        if self.canonical() {
            !self.lines.is_empty()
        } else {
            !self.rq.is_empty()
        }
    }

    /// Bytes waiting in the input queue (non-canonical reads).
    pub fn raw_available(&self) -> usize {
        self.rq.len()
    }

    // ---- output processing (OPOST) ----

    /// Appends `data` to the output queue, processed as `c_oflag` says.
    pub fn output(&mut self, data: &[u8]) {
        let oflag = self.termios.oflag;
        if oflag & OPOST == 0 {
            self.out.extend(data);
            return;
        }
        for &c in data {
            match c {
                b'\n' => {
                    if oflag & ONLRET != 0 {
                        self.column = 0;
                    }
                    if oflag & ONLCR != 0 {
                        self.out.extend(b"\r\n");
                        self.column = 0;
                    } else {
                        self.out.push_back(b'\n');
                    }
                }
                b'\r' => {
                    if oflag & ONOCR != 0 && self.column == 0 {
                        continue;
                    }
                    if oflag & OCRNL != 0 {
                        self.out.push_back(b'\n');
                        if oflag & ONLRET != 0 {
                            self.column = 0;
                        }
                    } else {
                        self.out.push_back(b'\r');
                        self.column = 0;
                    }
                }
                b'\t' => {
                    let spaces = 8 - (self.column % 8);
                    if oflag & TABDLY == XTABS {
                        self.out.extend(std::iter::repeat_n(b' ', spaces as usize));
                    } else {
                        self.out.push_back(b'\t');
                    }
                    self.column += spaces;
                }
                0x08 => {
                    self.column = self.column.saturating_sub(1);
                    self.out.push_back(c);
                }
                _ => {
                    let c = if oflag & OLCUC != 0 {
                        c.to_ascii_uppercase()
                    } else {
                        c
                    };
                    // UTF-8 continuation bytes and control characters take no column.
                    if !is_ctrl(c) && (c & 0xc0 != 0x80 || !self.i(IUTF8)) {
                        self.column += 1;
                    }
                    self.out.push_back(c);
                }
            }
        }
    }

    /// Echoes one input byte: control characters as `^X` with `ECHOCTL`.
    fn echo_char(&mut self, c: u8) {
        if self.l(ECHOCTL) && is_ctrl(c) && c != b'\t' && c != b'\n' {
            self.output(&[b'^', c ^ 0x40]);
        } else {
            self.output(&[c]);
        }
    }

    /// The columns `bytes` occupy when echoed from `start`.
    fn columns(&self, start: u32, bytes: &[u8]) -> u32 {
        let mut col = start;
        for &c in bytes {
            if c == b'\t' {
                col = (col | 7) + 1;
            } else if is_ctrl(c) {
                if self.l(ECHOCTL) {
                    col += 2;
                }
            } else if c & 0xc0 != 0x80 || !self.i(IUTF8) {
                col += 1;
            }
        }
        col
    }

    // ---- input processing (n_tty receive) ----

    /// Processes bytes typed at the terminal (master writes, console input).
    pub fn input(&mut self, data: &[u8], now: u64) -> InputEffects {
        let mut fx = InputEffects::default();
        if !data.is_empty() {
            self.hint_sent = false;
            self.last_input = now;
        }
        for &byte in data {
            self.input_byte(byte, &mut fx);
        }
        self.generation += 1;
        fx
    }

    fn input_byte(&mut self, byte: u8, fx: &mut InputEffects) {
        let mut c = if self.i(ISTRIP) { byte & 0x7f } else { byte };
        if self.lnext {
            // After ^V (echoed as "^\b"), any byte is data; its echo overwrites the caret.
            self.lnext = false;
            self.store(c);
            return;
        }
        if self.i(IXON) {
            if self.is(c, VSTOP) {
                self.stopped = true;
                return;
            }
            if self.is(c, VSTART) {
                self.stopped = false;
                return;
            }
            if self.stopped && self.i(IXANY) {
                self.stopped = false;
            }
        }
        if self.l(ISIG) {
            let signal = if self.is(c, VINTR) {
                Some(wasmshim_linux_abi::consts::signal::SIGINT)
            } else if self.is(c, VQUIT) {
                Some(wasmshim_linux_abi::consts::signal::SIGQUIT)
            } else if self.is(c, VSUSP) {
                Some(wasmshim_linux_abi::consts::signal::SIGTSTP)
            } else {
                None
            };
            if let Some(s) = signal {
                if !self.l(NOFLSH) {
                    self.flush_input();
                    self.out.clear();
                }
                self.stopped = false;
                if self.l(ECHO) {
                    self.echo_char(c);
                }
                fx.signal = Some(s);
                return;
            }
        }
        if c == b'\r' {
            if self.i(IGNCR) {
                return;
            }
            if self.i(ICRNL) {
                c = b'\n';
            }
        } else if c == b'\n' && self.i(INLCR) {
            c = b'\r';
        }
        if !self.canonical() {
            if self.rq.len() >= INPUT_CAPACITY {
                return;
            }
            if self.l(ECHO) {
                self.echo_char(c);
            }
            self.rq.push_back(c);
            return;
        }
        if self.is(c, VERASE) || self.is(c, VKILL) || (self.is(c, VWERASE) && self.l(IEXTEN)) {
            self.erase(c);
            return;
        }
        if self.is(c, VLNEXT) && self.l(IEXTEN) {
            self.lnext = true;
            if self.l(ECHO) && self.l(ECHOCTL) {
                self.output(b"^\x08");
            }
            return;
        }
        if self.is(c, VREPRINT) && self.l(IEXTEN) && self.l(ECHO) {
            self.echo_char(c);
            self.output(b"\n");
            let line = self.edit.clone();
            self.edit_column = self.column;
            for b in line {
                self.echo_char(b);
            }
            return;
        }
        if c == b'\n' {
            if self.l(ECHO) || self.l(ECHONL) {
                self.output(b"\n");
            }
            self.edit.push(b'\n');
            self.commit_line();
            return;
        }
        if self.is(c, VEOF) {
            self.commit_line();
            return;
        }
        if self.is(c, VEOL) || (self.is(c, VEOL2) && self.l(IEXTEN)) {
            if self.l(ECHO) {
                self.echo_char(c);
            }
            self.edit.push(c);
            self.commit_line();
            return;
        }
        self.store(c);
    }

    /// Adds an ordinary byte to the line being edited (canonical) or the input queue.
    fn store(&mut self, c: u8) {
        if !self.canonical() {
            if self.rq.len() < INPUT_CAPACITY {
                if self.l(ECHO) {
                    self.echo_char(c);
                }
                self.rq.push_back(c);
            }
            return;
        }
        if self.edit.len() >= MAX_CANON {
            // The line is full: Linux drops the byte (and rings the bell with IMAXBEL).
            if self.i(IMAXBEL) && self.l(ECHO) {
                self.output(b"\x07");
            }
            return;
        }
        if self.edit.is_empty() {
            self.edit_column = self.column;
        }
        if self.l(ECHO) {
            self.echo_char(c);
        }
        self.edit.push(c);
    }

    /// The line being edited becomes readable (an empty one is an end of file).
    fn commit_line(&mut self) {
        let len = self.edit.len();
        self.rq.extend(self.edit.drain(..));
        self.lines.push_back(len);
    }

    /// `VERASE`, `VWERASE` and `VKILL` in canonical mode.
    fn erase(&mut self, c: u8) {
        let kill = self.is(c, VKILL);
        let word = !kill && self.is(c, VWERASE);
        if self.edit.is_empty() {
            return;
        }
        if kill && !self.l(ECHO) {
            self.edit.clear();
            return;
        }
        if kill && !(self.l(ECHOK) && self.l(ECHOKE) && self.l(ECHOE)) {
            // No visual kill: the kill character is echoed (then a newline with ECHOK).
            self.edit.clear();
            self.echo_char(c);
            if self.l(ECHOK) {
                self.output(b"\n");
            }
            return;
        }
        let mut seen_word = false;
        while let Some(&last) = self.edit.last() {
            if word {
                let in_word = last.is_ascii_alphanumeric() || last == b'_';
                if in_word {
                    seen_word = true;
                } else if seen_word {
                    break;
                }
            }
            // One character: a whole UTF-8 sequence with IUTF8.
            let mut len = 1;
            if self.i(IUTF8) {
                while len < self.edit.len() && self.edit[self.edit.len() - len] & 0xc0 == 0x80 {
                    len += 1;
                }
            }
            let start = self.edit.len() - len;
            let before = self.columns(self.edit_column, &self.edit[..start]);
            let after = self.columns(self.edit_column, &self.edit);
            self.edit.truncate(start);
            if self.l(ECHO) {
                if !kill && !self.l(ECHOE) {
                    self.echo_char(c);
                } else if last == b'\t' {
                    // A tab is erased by moving back over the columns it took.
                    for _ in before..after {
                        self.output(b"\x08");
                    }
                } else {
                    for _ in before..after {
                        self.output(b"\x08 \x08");
                    }
                }
            }
            if !kill && !word {
                break;
            }
        }
    }

    /// Discards unread input (`tcflush(TCIFLUSH)`, signals without `NOFLSH`).
    pub fn flush_input(&mut self) {
        self.rq.clear();
        self.lines.clear();
        self.edit.clear();
        self.lnext = false;
        self.generation += 1;
    }

    /// Applies new termios, moving input between the canonical and raw representations.
    pub fn set_termios(&mut self, t: Termios) {
        let was = self.canonical();
        self.termios = t;
        let now = self.canonical();
        if was && !now {
            // Everything typed becomes readable at once (Linux: canon_head = read_head).
            self.rq.extend(self.edit.drain(..));
            self.lines.clear();
        } else if !was && now {
            // Unread raw input becomes the start of the line being edited.
            let mut pending: Vec<u8> = self.rq.drain(..).collect();
            pending.append(&mut self.edit);
            self.edit = pending;
            self.lines.clear();
            self.edit_column = self.column;
        }
        self.generation += 1;
    }

    /// Takes up to `max` readable bytes. In canonical mode at most one line; `Some(0)` is an
    /// end of file. `None`: nothing to read now.
    pub fn read(&mut self, max: usize) -> Option<Vec<u8>> {
        if self.canonical() {
            let first = *self.lines.front()?;
            let n = first.min(max);
            let data: Vec<u8> = self.rq.drain(..n).collect();
            if n == first {
                self.lines.pop_front();
            } else if let Some(l) = self.lines.front_mut() {
                *l -= n;
            }
            self.generation += 1;
            Some(data)
        } else {
            if self.rq.is_empty() {
                return None;
            }
            let n = self.rq.len().min(max);
            self.generation += 1;
            Some(self.rq.drain(..n).collect())
        }
    }

    /// The device number (`st_rdev` encoding).
    pub fn rdev(&self) -> u64 {
        match self.kind {
            TtyKind::Pty(n) => (136 << 8) | (u64::from(n) & 0xff) | ((u64::from(n) & !0xff) << 12),
            TtyKind::Console => (5 << 8) | 1,
        }
    }
}

/// Splits a console input stream into typed bytes and in-band window size reports.
///
/// The console contract reports the terminal size in band, in the format of the
/// "in-band resize" terminal protocol (DEC private mode 2048: `CSI 48 ; rows ; cols ; ypixels ;
/// xpixels t`, the pixel fields optional). A report must arrive within one input chunk; a
/// partial one at the end of a chunk is ordinary input.
#[derive(Debug, Default)]
pub struct ConsoleDecoder;

/// One element of decoded console input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConsoleEvent {
    /// Bytes the user typed.
    Input(Vec<u8>),
    /// A window size report.
    Resize(Winsize),
}

impl ConsoleDecoder {
    /// Decodes one chunk of console input.
    pub fn decode(&mut self, data: &[u8]) -> Vec<ConsoleEvent> {
        let mut events = Vec::new();
        let mut plain = Vec::new();
        let mut i = 0;
        while i < data.len() {
            if data[i] == 0x1b
                && let Some((size, len)) = parse_resize(&data[i..])
            {
                if !plain.is_empty() {
                    events.push(ConsoleEvent::Input(std::mem::take(&mut plain)));
                }
                events.push(ConsoleEvent::Resize(size));
                i += len;
                continue;
            }
            plain.push(data[i]);
            i += 1;
        }
        if !plain.is_empty() {
            events.push(ConsoleEvent::Input(plain));
        }
        events
    }
}

/// Parses `ESC [ 48 ; rows ; cols [; ypix ; xpix] t` at the start of `s`.
fn parse_resize(s: &[u8]) -> Option<(Winsize, usize)> {
    let body = s.strip_prefix(b"\x1b[48;")?;
    let end = body
        .iter()
        .position(|&b| !(b.is_ascii_digit() || b == b';'))?;
    if body[end] != b't' {
        return None;
    }
    let fields: Vec<u16> = body[..end]
        .split(|&b| b == b';')
        .map(|f| std::str::from_utf8(f).ok()?.parse().ok())
        .collect::<Option<_>>()?;
    let (rows, cols, ypixel, xpixel) = match fields.as_slice() {
        [r, c] => (*r, *c, 0, 0),
        [r, c, y, x] => (*r, *c, *y, *x),
        _ => return None,
    };
    Some((
        Winsize {
            rows,
            cols,
            xpixel,
            ypixel,
        },
        5 + end + 1,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tty() -> Tty {
        Tty::new(1, TtyKind::Pty(0), 0, Winsize::default())
    }

    fn out(t: &mut Tty) -> Vec<u8> {
        t.out.drain(..).collect()
    }

    #[test]
    fn cooked_lines_echo_and_onlcr() {
        let mut t = tty();
        t.input(b"ab\r", 0);
        assert_eq!(out(&mut t), b"ab\r\n");
        assert_eq!(t.read(100), Some(b"ab\n".to_vec()));
        assert_eq!(t.read(100), None);
        t.input(b"x\x04", 0); // ^D ends a non-empty line without a newline
        assert_eq!(t.read(100), Some(b"x".to_vec()));
        t.input(b"\x04", 0); // ^D on an empty line: end of file
        assert_eq!(t.read(100), Some(Vec::new()));
    }

    #[test]
    fn partial_reads_stay_within_a_line() {
        let mut t = tty();
        t.input(b"hello\nworld\n", 0);
        assert_eq!(t.read(3), Some(b"hel".to_vec()));
        assert_eq!(t.read(100), Some(b"lo\n".to_vec()));
        assert_eq!(t.read(100), Some(b"world\n".to_vec()));
    }

    #[test]
    fn erase_kill_and_werase() {
        let mut t = tty();
        t.input(b"abc\x7f", 0);
        assert_eq!(out(&mut t), b"abc\x08 \x08");
        t.input(b"d e\x17", 0); // ^W erases "e"
        assert_eq!(out(&mut t), b"d e\x08 \x08");
        t.input(b"\x15", 0); // ^U erases the whole line (ECHOKE)
        assert_eq!(out(&mut t), b"\x08 \x08".repeat(4));
        t.input(b"ok\n", 0);
        assert_eq!(t.read(100), Some(b"ok\n".to_vec()));
    }

    #[test]
    fn control_characters_echo_as_caret() {
        let mut t = tty();
        t.input(b"\x01", 0);
        assert_eq!(out(&mut t), b"^A");
        t.input(b"\x7f", 0); // erases both columns
        assert_eq!(out(&mut t), b"\x08 \x08\x08 \x08");
    }

    #[test]
    fn interrupt_flushes_and_signals() {
        let mut t = tty();
        t.input(b"partial", 0);
        let fx = t.input(b"\x03", 0);
        assert_eq!(fx.signal, Some(wasmshim_linux_abi::consts::signal::SIGINT));
        assert_eq!(t.read(100), None);
        assert!(out(&mut t).ends_with(b"^C"));
    }

    #[test]
    fn raw_mode_and_switching() {
        let mut t = tty();
        t.input(b"ab", 0); // still being edited
        let mut raw = t.termios;
        raw.lflag &= !(ICANON | ECHO);
        t.set_termios(raw);
        assert_eq!(t.read(1), Some(b"a".to_vec()));
        t.input(b"\x7f", 0); // no special meaning now
        assert_eq!(t.read(100), Some(b"b\x7f".to_vec()));
        assert_eq!(out(&mut t), b"ab", "only the cooked echo of \"ab\"");
    }

    #[test]
    fn literal_next_and_tabs() {
        let mut t = tty();
        t.input(b"\x16\x03", 0); // ^V ^C: a literal ^C
        assert_eq!(t.read(100), None);
        t.input(b"\n", 0);
        assert_eq!(t.read(100), Some(b"\x03\n".to_vec()));
        let mut x = t.termios;
        x.oflag |= XTABS;
        t.set_termios(x);
        out(&mut t);
        t.output(b"ab\tc\n");
        assert_eq!(out(&mut t), b"ab      c\r\n");
    }

    #[test]
    fn console_decoder_splits_resizes() {
        let mut d = ConsoleDecoder;
        let ev = d.decode(b"ls\x1b[48;40;120;0;0t\r\x1b[A");
        assert_eq!(
            ev,
            vec![
                ConsoleEvent::Input(b"ls".to_vec()),
                ConsoleEvent::Resize(Winsize {
                    rows: 40,
                    cols: 120,
                    xpixel: 0,
                    ypixel: 0
                }),
                ConsoleEvent::Input(b"\r\x1b[A".to_vec()),
            ]
        );
        assert_eq!(
            d.decode(b"\x1b[48;24;80t"),
            vec![ConsoleEvent::Resize(Winsize {
                rows: 24,
                cols: 80,
                xpixel: 0,
                ypixel: 0
            })]
        );
        // Not a report: passed through.
        assert_eq!(
            d.decode(b"\x1b[48;1x"),
            vec![ConsoleEvent::Input(b"\x1b[48;1x".to_vec())]
        );
    }
}
