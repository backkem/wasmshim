//! The node's standard streams, connected to this process's.
//!
//! **Output** is written through as the guest writes it, so a long run shows its progress and
//! a trap loses nothing.
//!
//! **Input** is handed to the guest only when the node can do nothing else: every timer has
//! fired and every task waits. Then the runner waits (in wall time) for the next piece of
//! input and delivers exactly one piece: a line, at most [`PIECE`] bytes, or the end of input.
//! So what the guest reads, and the virtual time at which it reads it, depend on the bytes of
//! the input alone, never on how fast they arrive: a run with piped input repeats exactly. The
//! price: a guest that waits for input with a timeout always sees the timeout first.
//!
//! Both WASI versions are served from the same state: 0.2's `wasi:io` streams (what Rust's
//! `std::io::stdin` reads) and 0.3's byte streams (what the node's console reads).

use std::collections::VecDeque;
use std::future::Future;
use std::io::{self, BufRead as _, Read as _, Write as _};
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use bytes::Bytes;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use wasmtime_wasi::cli::{IsTerminal, StdinStream, StdoutStream};
use wasmtime_wasi::p2::{InputStream, OutputStream, Pollable, StreamError, StreamResult};

/// The most bytes delivered to the guest at once.
const PIECE: usize = 64 * 1024;

/// How much a 0.2 guest may write per call.
const WRITE_PERMIT: usize = 1024 * 1024;

// ---- output ----

/// This process's stdout or stderr.
#[derive(Clone, Copy)]
pub enum HostOutput {
    Stdout,
    Stderr,
}

impl HostOutput {
    /// Writes all of `bytes` now. Stdout is flushed so that output appears as it is produced
    /// (stderr is unbuffered).
    fn put(self, bytes: &[u8]) -> io::Result<()> {
        match self {
            Self::Stdout => {
                let mut out = io::stdout().lock();
                out.write_all(bytes)?;
                out.flush()
            }
            Self::Stderr => io::stderr().lock().write_all(bytes),
        }
    }
}

impl IsTerminal for HostOutput {
    fn is_terminal(&self) -> bool {
        match self {
            Self::Stdout => io::IsTerminal::is_terminal(&io::stdout()),
            Self::Stderr => io::IsTerminal::is_terminal(&io::stderr()),
        }
    }
}

impl StdoutStream for HostOutput {
    fn async_stream(&self) -> Box<dyn AsyncWrite + Send + Sync> {
        Box::new(*self)
    }

    fn p2_stream(&self) -> Box<dyn OutputStream> {
        Box::new(*self)
    }
}

impl AsyncWrite for HostOutput {
    fn poll_write(
        self: Pin<&mut Self>,
        _: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Poll::Ready(self.put(buf).map(|()| buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

#[async_trait::async_trait]
impl OutputStream for HostOutput {
    fn write(&mut self, bytes: Bytes) -> StreamResult<()> {
        // A reader that went away (a closed pipe) closes the stream for the guest.
        self.put(&bytes).map_err(|_| StreamError::Closed)
    }

    fn flush(&mut self) -> StreamResult<()> {
        Ok(())
    }

    fn check_write(&mut self) -> StreamResult<usize> {
        Ok(WRITE_PERMIT)
    }
}

#[async_trait::async_trait]
impl Pollable for HostOutput {
    async fn ready(&mut self) {}
}

// ---- input ----

/// This process's stdin, delivered piece by piece (see the module).
#[derive(Clone, Default)]
pub struct HostInput(Arc<Shared>);

#[derive(Default)]
struct Shared {
    state: Mutex<Input>,
    /// Signalled by the reader thread when a piece (or the end) has arrived.
    arrived: Condvar,
}

#[derive(Default)]
struct Input {
    /// Whether the reader thread runs. It starts with the first wait, so a run that never reads
    /// its input never touches this process's stdin.
    reading: bool,
    /// Pieces the reader thread has read and the guest has not been given.
    arrived: VecDeque<Vec<u8>>,
    /// The reader thread saw the end of input (or an error).
    arrived_end: bool,
    /// Bytes delivered to the guest and not yet read by it.
    ready: VecDeque<u8>,
    /// The end of input was delivered.
    end: bool,
    /// Guest reads that wait for a delivery.
    waiting: Vec<Waker>,
}

impl HostInput {
    fn lock(&self) -> MutexGuard<'_, Input> {
        self.0.state.lock().expect("stdin state")
    }

    /// Whether a guest read waits for input.
    pub fn wanted(&self) -> bool {
        !self.lock().waiting.is_empty()
    }

    /// Waits for the next piece of input and gives it to the guest (wakes its reads). Called
    /// when the node can do nothing else and [`wanted`](Self::wanted) says a read waits.
    pub fn deliver(&self) {
        let mut state = self.lock();
        if !state.reading {
            state.reading = true;
            let shared = self.0.clone();
            std::thread::spawn(move || read_stdin(&shared));
        }
        while state.arrived.is_empty() && !state.arrived_end {
            state = self.0.arrived.wait(state).expect("stdin state");
        }
        match state.arrived.pop_front() {
            Some(piece) => state.ready.extend(piece),
            None => state.end = true,
        }
        let waiting = std::mem::take(&mut state.waiting);
        drop(state);
        waiting.into_iter().for_each(Waker::wake);
    }

    /// Up to `max` delivered bytes: `Some(bytes)` (empty at the end of input), or `None` when
    /// the guest has to wait for a delivery (`waker` is kept then).
    fn take(&self, max: usize, waker: Option<&Waker>) -> Option<Vec<u8>> {
        let mut state = self.lock();
        if !state.ready.is_empty() {
            let n = max.min(state.ready.len());
            return Some(state.ready.drain(..n).collect());
        }
        if state.end {
            return Some(Vec::new());
        }
        if let Some(waker) = waker {
            state.waiting.push(waker.clone());
        }
        None
    }
}

/// The reader thread: this process's stdin in pieces, until it ends.
fn read_stdin(shared: &Shared) {
    let mut stdin = io::stdin().lock();
    loop {
        let mut piece = Vec::new();
        let read = (&mut stdin)
            .take(PIECE as u64)
            .read_until(b'\n', &mut piece);
        let mut state = shared.state.lock().expect("stdin state");
        match read {
            Ok(n) if n > 0 => state.arrived.push_back(piece),
            // The end of input, or a read error, which ends it too.
            _ => state.arrived_end = true,
        }
        let ended = state.arrived_end;
        drop(state);
        shared.arrived.notify_all();
        if ended {
            return;
        }
    }
}

impl IsTerminal for HostInput {
    fn is_terminal(&self) -> bool {
        io::IsTerminal::is_terminal(&io::stdin())
    }
}

impl StdinStream for HostInput {
    fn async_stream(&self) -> Box<dyn AsyncRead + Send + Sync> {
        Box::new(self.clone())
    }

    fn p2_stream(&self) -> Box<dyn InputStream> {
        Box::new(self.clone())
    }
}

impl AsyncRead for HostInput {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.take(buf.remaining(), Some(cx.waker())) {
            // Nothing put into `buf` is the end of input.
            Some(bytes) => {
                buf.put_slice(&bytes);
                Poll::Ready(Ok(()))
            }
            None => Poll::Pending,
        }
    }
}

#[async_trait::async_trait]
impl InputStream for HostInput {
    fn read(&mut self, size: usize) -> StreamResult<Bytes> {
        match self.take(size, None) {
            Some(bytes) if bytes.is_empty() && size > 0 => Err(StreamError::Closed),
            Some(bytes) => Ok(bytes.into()),
            // Not ready: an empty read, and the guest waits on the pollable.
            None => Ok(Bytes::new()),
        }
    }
}

#[async_trait::async_trait]
impl Pollable for HostInput {
    async fn ready(&mut self) {
        Delivered(self.clone()).await;
    }
}

/// Resolves once input (or its end) has been delivered.
struct Delivered(HostInput);

impl Future for Delivered {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let mut state = self.0.lock();
        if !state.ready.is_empty() || state.end {
            return Poll::Ready(());
        }
        state.waiting.push(cx.waker().clone());
        Poll::Pending
    }
}
