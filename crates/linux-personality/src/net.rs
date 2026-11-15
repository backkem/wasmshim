//! The network provider contract: how AF_INET/AF_INET6 traffic that is *not* for this host
//! leaves the personality.
//!
//! Traffic to "this host" (127.0.0.0/8, `::1`, the unspecified addresses) never leaves the
//! kernel: it uses the in-kernel loopback. Everything else goes to a [`Network`]
//! that the embedder attaches with [`crate::Kernel::set_network`]; without one, such traffic is
//! `ENETUNREACH`.
//!
//! The split is a "kernel facade":
//!
//! * The kernel owns every socket's buffers and state in a shared [`NetSocket`], computes
//!   readiness from it, and never blocks on the provider. Syscalls only look at and change
//!   that state, then [`Network::kick`] the provider.
//! * The provider moves bytes and connections between those buffers and the real transport
//!   *outside* syscalls (in `wasmshim-linux-node`: async pump tasks over `wasi:sockets@0.3`), bumps
//!   [`NetSocket::generation`] and reports the ids it changed through
//!   [`Network::take_changed`]. The kernel collects them at the start of every
//!   [`crate::Kernel::run`] and wakes exactly the threads that wait on those sockets.
//!
//! Nothing here names WASI: the same contract can be served by a browser host, a native
//! host, or a test double.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::rc::Rc;

use wasmshim_linux_abi::Errno;

/// Object ids at or above this value are allocated by the provider (for connections it
/// accepted), never by the kernel, so the two id spaces cannot collide.
pub const PROVIDER_ID_BASE: u64 = 1 << 62;

/// Default per-socket buffer sizes (the kernel's `SO_RCVBUF`/`SO_SNDBUF`) for provider
/// sockets: 64 KiB, on top of the provider's own window.
pub const DEFAULT_BUF: usize = 64 << 10;

/// Datagrams queued per direction before `EAGAIN` (receive: further datagrams are dropped).
pub const DGRAM_QUEUE: usize = 64;

/// A shared handle to a provider socket.
pub type NetRef = Rc<RefCell<NetSocket>>;

/// What a restore brought back for the provider: its own record, and each provider
/// socket with its record, in the order they were saved.
pub type Restored = (Option<Vec<u8>>, Vec<(NetRef, Vec<u8>)>);

/// Transport protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proto {
    Tcp,
    Udp,
    /// An unprivileged ICMP echo ("ping") socket: `SOCK_DGRAM` + `IPPROTO_ICMP` /
    /// `IPPROTO_ICMPV6`. Datagrams are whole ICMP messages; the port of an address is the echo
    /// identifier (local) or 0 (remote), as on Linux. The provider picks the identifier when
    /// the bound port is 0 and overwrites it and the checksum on every request.
    Icmp,
}

/// Life cycle of a provider socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetState {
    /// Created (maybe bound); not connected, not listening.
    Idle,
    /// TCP: the provider is connecting.
    Connecting,
    /// TCP: connected (UDP: has a default peer).
    Connected,
    /// TCP: the provider is accepting connections into [`NetSocket::accepted`].
    Listening,
    /// TCP: connecting failed; the reason is in [`NetSocket::error`].
    Failed,
}

/// A socket option the kernel forwards to the provider. Options a provider cannot honour are
/// accepted and ignored (graceful degradation, e.g. `TCP_NODELAY` on stock wasmtime).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetOption {
    /// `TCP_NODELAY`.
    NoDelay(bool),
    /// `SO_KEEPALIVE`.
    KeepAlive(bool),
    /// `IP_TTL` / `IPV6_UNICAST_HOPS`.
    HopLimit(u8),
}

/// The state one socket shares between the kernel and the provider.
///
/// Ownership of each field is one-sided, which is what keeps the two halves simple: the
/// kernel fills `tx`/`dgrams_out` and sets the `shut_*`/`closed`/`abort` requests; the
/// provider fills `rx`/`dgrams_in`/`accepted` and sets `state`, `rx_end`, `tx_err`,
/// `tx_done`, `error`, `local`/`remote`. Either side may drain what the other fills.
#[derive(Debug)]
pub struct NetSocket {
    /// The kernel object id: the wait key of the threads blocked on this socket.
    pub id: u64,
    pub proto: Proto,
    /// An `AF_INET6` socket (addresses are reported as IPv6, IPv4 peers as v4-mapped).
    pub v6: bool,
    /// `IPV6_V6ONLY`: an IPv6 wildcard socket that does *not* also serve IPv4.
    pub v6only: bool,
    pub state: NetState,
    /// Bound local address (set by the provider after `bind`/`connect`).
    pub local: Option<SocketAddr>,
    /// Peer address (connected sockets).
    pub remote: Option<SocketAddr>,

    /// TCP receive buffer (provider → kernel).
    pub rx: VecDeque<u8>,
    pub rx_cap: usize,
    /// The receive direction ended: `Ok` at the peer's FIN, `Err` after a reset or error.
    pub rx_end: Option<Result<(), Errno>>,
    /// TCP send buffer (kernel → provider).
    pub tx: VecDeque<u8>,
    pub tx_cap: usize,
    /// Bytes the provider took from `tx` and has not finished sending (count against `tx_cap`).
    pub tx_inflight: usize,
    /// The send direction failed (`EPIPE`, `ECONNRESET`).
    pub tx_err: Option<Errno>,
    /// The provider finished the send direction (sent its FIN after `shut_wr` or `closed`).
    pub tx_done: bool,

    /// `shutdown(SHUT_WR)`: send FIN once `tx` drains.
    pub shut_wr: bool,
    /// `shutdown(SHUT_RD)`: reads return EOF; arriving data is discarded.
    pub shut_rd: bool,
    /// The last descriptor closed: finish sending (linger), then release everything.
    pub closed: bool,
    /// Abortive close (`SO_LINGER {1, 0}`, close with unread data): reset the connection.
    pub abort: bool,
    /// `SO_LINGER {1, 0}` is set: `close` aborts.
    pub linger_zero: bool,
    /// `SO_RCVTIMEO` / `SO_SNDTIMEO` in ns (blocking calls then fail with `EAGAIN`).
    pub rcvtimeo: Option<u64>,
    pub sndtimeo: Option<u64>,
    /// Pending `SO_ERROR` (a failed connect).
    pub error: Option<Errno>,
    /// A blocking `connect` is waiting for this socket (restarts return its result once).
    pub connect_waiter: bool,

    /// TCP listener: accepted connections not yet taken by `accept`.
    pub accepted: VecDeque<NetRef>,
    /// TCP listener: how many connections may wait in `accepted`.
    pub backlog: usize,

    /// UDP and ICMP: received datagrams with their source and, if the provider knows it, the
    /// TTL / hop limit they arrived with (`IP_RECVTTL`, `IPV6_RECVHOPLIMIT`).
    pub dgrams_in: VecDeque<(Vec<u8>, SocketAddr, Option<u8>)>,
    /// UDP: datagrams to send, with an explicit destination or `None` for the default peer.
    pub dgrams_out: VecDeque<(Vec<u8>, Option<SocketAddr>)>,
    /// UDP: datagrams dropped because `dgrams_in` was full.
    pub dgrams_dropped: u64,

    /// `TCP_NODELAY` as last set (reported back by `getsockopt`).
    pub nodelay: bool,
    /// `SO_KEEPALIVE` as last set.
    pub keepalive: bool,
    /// Other integer options as last set (`SO_REUSEADDR`, `TCP_CORK`, ...), read back by
    /// `getsockopt`: (level, option) to value.
    pub opts: std::collections::BTreeMap<(u64, u64), u32>,
    /// Bumped by whoever changes readiness-relevant state (edge-triggered epoll compares it).
    pub generation: u64,
}

impl NetSocket {
    /// A fresh socket with the default buffer sizes.
    pub fn new(id: u64, proto: Proto, v6: bool) -> Self {
        Self {
            id,
            proto,
            v6,
            v6only: false,
            state: NetState::Idle,
            local: None,
            remote: None,
            rx: VecDeque::new(),
            rx_cap: DEFAULT_BUF,
            rx_end: None,
            tx: VecDeque::new(),
            tx_cap: DEFAULT_BUF,
            tx_inflight: 0,
            tx_err: None,
            tx_done: false,
            shut_wr: false,
            shut_rd: false,
            closed: false,
            abort: false,
            linger_zero: false,
            rcvtimeo: None,
            sndtimeo: None,
            error: None,
            connect_waiter: false,
            accepted: VecDeque::new(),
            backlog: 1,
            dgrams_in: VecDeque::new(),
            dgrams_out: VecDeque::new(),
            dgrams_dropped: 0,
            nodelay: false,
            keepalive: false,
            opts: std::collections::BTreeMap::new(),
            generation: 0,
        }
    }

    /// A shared handle.
    pub fn shared(self) -> NetRef {
        Rc::new(RefCell::new(self))
    }

    /// Free space in the receive buffer (what the provider may read next).
    pub fn rx_space(&self) -> usize {
        self.rx_cap.saturating_sub(self.rx.len())
    }

    /// Free space in the send buffer (what `write` may queue).
    pub fn tx_space(&self) -> usize {
        self.tx_cap.saturating_sub(self.tx.len() + self.tx_inflight)
    }

    /// Records a change that affects readiness.
    pub fn touch(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Queues a received datagram (dropping it, like a full `SO_RCVBUF`, when the queue is full).
    pub fn push_datagram(&mut self, data: Vec<u8>, from: SocketAddr) {
        self.push_datagram_hops(data, from, None);
    }

    /// [`push_datagram`](Self::push_datagram) with the TTL / hop limit it arrived with.
    pub fn push_datagram_hops(&mut self, data: Vec<u8>, from: SocketAddr, hops: Option<u8>) {
        if self.dgrams_in.len() >= DGRAM_QUEUE {
            self.dgrams_dropped += 1;
        } else {
            self.dgrams_in.push_back((data, from, hops));
        }
        self.touch();
    }
}

/// A network provider (see the module docs). Every method is synchronous and never blocks:
/// asynchronous work (connecting, accepting, moving bytes) happens later, outside syscalls,
/// and is reported through [`Network::take_changed`].
pub trait Network {
    /// Creates the provider socket for `sock` if needed and binds it to `addr`. On success
    /// `sock.local` is the bound address (with the chosen port when `addr`'s port was 0). A
    /// UDP socket starts receiving into `dgrams_in` once bound.
    fn bind(&mut self, sock: &NetRef, addr: SocketAddr) -> Result<(), Errno>;

    /// Starts listening on a bound TCP socket: accepted connections (fresh [`NetSocket`]s with
    /// provider-allocated ids, already `Connected`) appear in `sock.accepted`, at most
    /// `sock.backlog` of them at a time.
    fn listen(&mut self, sock: &NetRef) -> Result<(), Errno>;

    /// TCP: starts connecting to `addr` (`state` is `Connecting` on return, and later
    /// `Connected` or `Failed` with `error` set). UDP: sets the default peer, synchronously.
    fn connect(&mut self, sock: &NetRef, addr: SocketAddr) -> Result<(), Errno>;

    /// The kernel changed `sock` (queued data, freed buffer space, shut down, closed or
    /// aborted it): the provider looks at it again. After a kick with `closed` set, the kernel
    /// never names this socket again; the provider finishes sending and releases it.
    fn kick(&mut self, sock: &NetRef);

    /// Applies a socket option. A provider that lacks one returns `Ok(())` and ignores it.
    fn set_option(&mut self, sock: &NetRef, opt: NetOption) -> Result<(), Errno>;

    /// The ids of the sockets whose state the provider changed since the last call, in a
    /// deterministic order.
    fn take_changed(&mut self) -> Vec<u64>;

    /// The provider's own record of a quiescent `sock` (idle or listening, nothing queued) for
    /// a checkpoint: opaque bytes [`Network::restore`] understands. A provider
    /// that cannot checkpoint its sockets refuses, and the checkpoint fails loudly.
    fn checkpoint(&self, sock: &NetRef) -> Result<Vec<u8>, String> {
        let _ = sock;
        Err("this network provider cannot checkpoint sockets".to_owned())
    }

    /// Re-attaches a socket restored from a checkpoint to the provider's side of it, from the
    /// record [`Network::checkpoint`] gave (a listener accepts again, a UDP socket receives).
    fn restore(&mut self, sock: &NetRef, saved: &[u8]) -> Result<(), String> {
        let _ = (sock, saved);
        Err("this network provider cannot restore sockets".to_owned())
    }

    /// Provider-wide state for a checkpoint (its id allocator, for example), and its restore.
    fn checkpoint_provider(&self) -> Result<Vec<u8>, String> {
        Err("this network provider cannot checkpoint".to_owned())
    }

    fn restore_provider(&mut self, saved: &[u8]) -> Result<(), String> {
        let _ = saved;
        Err("this network provider cannot restore".to_owned())
    }
}

impl NetSocket {
    /// Why this socket cannot be checkpointed now (idle or listening sockets with
    /// nothing queued or in flight only), if it cannot.
    pub fn not_quiescent(&self) -> Option<&'static str> {
        if !matches!(self.state, NetState::Idle | NetState::Listening) {
            return Some("a network connection");
        }
        if !self.rx.is_empty()
            || !self.tx.is_empty()
            || self.tx_inflight > 0
            || !self.accepted.is_empty()
            || !self.dgrams_in.is_empty()
            || !self.dgrams_out.is_empty()
        {
            return Some("a socket with data or connections queued");
        }
        if self.rx_end.is_some()
            || self.tx_err.is_some()
            || self.tx_done
            || self.closed
            || self.abort
            || self.error.is_some()
            || self.connect_waiter
        {
            return Some("a socket that is closing or failed");
        }
        None
    }
}
