//! The network provider over `wasi:sockets@0.3` (feature `net`): the kernel facade's pumps.
//!
//! The personality owns every socket's buffers ([`NetSocket`]); this module moves bytes,
//! datagrams and connections between those buffers and WASI sockets in async *pump* tasks
//! (`wit_bindgen::spawn_local`, inside `run`'s task). Syscalls only call the synchronous
//! [`Network`] methods, which create/bind/listen WASI sockets, spawn pumps and wake them.
//!
//! These async driver rules hold throughout:
//! 1. wakes are level-triggered ([`Notify`] generations);
//! 2. a waiting pump registers its waker at every poll (`Notify::changed`);
//! 3. no `RefCell` borrow is held across an await, an import call or a future drop;
//! 4. no timer is cancelled (the run loop's timers are never dropped, see `component.rs`);
//! 5. a cancelled wait removes itself (`Changed` deregisters on drop). The only futures this
//!    module drops while pending are stream reads/writes and UDP receives raced against a
//!    close or abort: a cancel, off the hot path.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::future::{Future, poll_fn};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::pin::{Pin, pin};
use std::rc::Rc;
use std::task::Poll;

use wasip3::sockets::types::{
    ErrorCode, IpAddress, IpAddressFamily, IpSocketAddress, Ipv4SocketAddress, Ipv6SocketAddress,
    TcpSocket, UdpSocket,
};
use wasip3::wit_bindgen::{FutureReader, StreamReader, StreamResult, StreamWriter};
use wasmshim_linux::net::{
    NetOption, NetRef, NetSocket, NetState, Network, PROVIDER_ID_BASE, Proto,
};
use wasmshim_linux_abi::Errno;

use crate::{Notify, Timers};

/// Bytes handed to a WASI send stream per write.
const TX_CHUNK: usize = 64 << 10;
/// Consecutive failed UDP receives after which a receive pump gives up.
const UDP_RX_ERRORS: u32 = 16;

/// State shared by the provider, its pump tasks and the run loop.
#[derive(Default)]
pub struct Shared {
    /// Wakes the run loop: a socket changed, or a timer fired.
    pub wake: Notify,
    /// Timers in flight for the run loop.
    pub timers: Timers,
    changed: RefCell<BTreeSet<u64>>,
    next_id: Cell<u64>,
    /// Pump tasks alive (connect, accept, receive and send pumps).
    pumps: Cell<usize>,
    socks: RefCell<BTreeMap<u64, Entry>>,
    /// What the embedder's socket provider adds beyond `wasi:sockets` ([`SocketExt`]).
    ext: RefCell<Option<Rc<dyn SocketExt>>>,
}

/// What a socket provider may offer beyond standard `wasi:sockets@0.3`: TCP abort (RST) and
/// `TCP_NODELAY`, ICMP echo sockets, and the provider-side records of a quiescent checkpoint.
/// The embedder supplies it ([`Shared::set_socket_ext`]); without it an abort degrades to a FIN,
/// `TCP_NODELAY` is accepted and ignored, a ping socket gets `EACCES`, and checkpoints of
/// network sockets are refused.
///
/// **Embedder surface, unstable**: nothing in this repository implements it.
pub trait SocketExt {
    /// Aborts a connection (RST).
    fn abort(&self, socket: &TcpSocket);
    /// Sets `TCP_NODELAY`.
    fn set_no_delay(&self, socket: &TcpSocket, on: bool);
    /// Creates an ICMP echo socket of `family`; `None` when the provider has none.
    fn icmp_create(
        &self,
        family: IpAddressFamily,
    ) -> Option<Result<Rc<dyn IcmpSocket>, ErrorCode>> {
        let _ = family;
        None
    }
    /// The provider's id of a TCP socket, as a checkpoint names it.
    fn tcp_id(&self, socket: &TcpSocket) -> Option<u32> {
        let _ = socket;
        None
    }
    /// The provider's id of a UDP socket, as a checkpoint names it.
    fn udp_id(&self, socket: &UdpSocket) -> Option<u32> {
        let _ = socket;
        None
    }
    /// Takes back a restored listening TCP socket and its stream of accepted connections.
    fn reclaim_listener(&self, id: u32) -> Result<(TcpSocket, StreamReader<TcpSocket>), String> {
        let _ = id;
        Err("this network provider cannot restore sockets".to_owned())
    }
    /// Takes back a restored TCP socket.
    fn reclaim_tcp(&self, id: u32) -> Result<TcpSocket, String> {
        let _ = id;
        Err("this network provider cannot restore sockets".to_owned())
    }
    /// Takes back a restored UDP socket.
    fn reclaim_udp(&self, id: u32) -> Result<UdpSocket, String> {
        let _ = id;
        Err("this network provider cannot restore sockets".to_owned())
    }
    /// Whether the provider checkpoints sockets ([`Network::checkpoint`] and friends).
    fn checkpoints(&self) -> bool {
        false
    }
}

/// A datagram an ICMP echo socket received: its bytes, its sender and the hop limit it came
/// with.
pub struct IcmpDatagram {
    pub data: Vec<u8>,
    pub remote_address: IpAddress,
    /// The IPv4 TTL or IPv6 hop limit, when the provider knows it.
    pub hop_limit: Option<u8>,
}

/// An ICMP echo socket of a provider ([`SocketExt::icmp_create`]).
pub trait IcmpSocket {
    fn bind(&self, local: IpAddress, identifier: u16) -> Result<(), ErrorCode>;
    fn identifier(&self) -> Result<u16, ErrorCode>;
    fn set_unicast_hop_limit(&self, value: u8) -> Result<(), ErrorCode>;
    fn receive_datagram(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<IcmpDatagram, ErrorCode>> + '_>>;
    fn send(
        &self,
        data: Vec<u8>,
        remote: Option<IpAddress>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ErrorCode>> + '_>>;
}

impl Shared {
    /// Installs the provider's extensions (once, before the kernel opens a socket).
    pub fn set_socket_ext(&self, ext: Option<Rc<dyn SocketExt>>) {
        *self.ext.borrow_mut() = ext;
    }

    fn ext(&self) -> Option<Rc<dyn SocketExt>> {
        self.ext.borrow().clone()
    }

    /// Network work is in flight: a blocked guest may still be woken by it.
    pub fn busy(&self) -> bool {
        self.pumps.get() > 0
    }

    fn alloc_id(&self) -> u64 {
        let n = self.next_id.get();
        self.next_id.set(n + 1);
        PROVIDER_ID_BASE + n
    }

    fn entry(&self, id: u64) -> Option<Entry> {
        self.socks.borrow().get(&id).cloned()
    }
}

/// Reports a change of `s` to the kernel and wakes the run loop.
fn mark(shared: &Shared, s: &NetRef) {
    let id = {
        let mut b = s.borrow_mut();
        b.touch();
        b.id
    };
    shared.changed.borrow_mut().insert(id);
    shared.wake.notify();
}

/// The WASI side of one kernel socket.
#[derive(Clone)]
struct Entry {
    handle: Handle,
    /// Wakes this socket's pumps (the kernel kicked it).
    pump: Rc<Notify>,
}

#[derive(Clone)]
enum Handle {
    /// One socket, or two for a dual-stack (`[::]` without `IPV6_V6ONLY`) listener.
    Tcp(Vec<Rc<TcpSocket>>),
    /// One socket, or two for a dual-stack wildcard.
    Udp(Vec<Rc<UdpSocket>>),
    /// An ICMP echo socket ([`SocketExt::icmp_create`]).
    Icmp(Rc<dyn IcmpSocket>),
}

/// Counts a live pump task.
struct PumpGuard(Rc<Shared>);

impl Drop for PumpGuard {
    fn drop(&mut self) {
        self.0.pumps.set(self.0.pumps.get().saturating_sub(1));
        // The run loop may be waiting only because this pump was alive.
        self.0.wake.notify();
    }
}

fn spawn(shared: &Rc<Shared>, task: impl Future<Output = ()> + 'static) {
    shared.pumps.set(shared.pumps.get() + 1);
    let guard = PumpGuard(shared.clone());
    wit_bindgen::spawn_local(async move {
        let _guard = guard;
        task.await;
    });
}

/// Polls both futures; the first to finish wins and the other is dropped (cancelled).
async fn race<A: Future, B: Future>(a: A, b: B) -> Result<A::Output, B::Output> {
    let mut a = pin!(a);
    let mut b = pin!(b);
    poll_fn(move |cx| {
        if let Poll::Ready(x) = a.as_mut().poll(cx) {
            return Poll::Ready(Ok(x));
        }
        if let Poll::Ready(y) = b.as_mut().poll(cx) {
            return Poll::Ready(Err(y));
        }
        Poll::Pending
    })
    .await
}

/// Completes once `stop(socket)` holds (checked at every wake of `pump`).
async fn until(s: &NetRef, pump: &Notify, stop: fn(&NetSocket) -> bool) {
    loop {
        let g = pump.generation();
        if stop(&s.borrow()) {
            return;
        }
        pump.changed(g).await;
    }
}

fn is_closed(s: &NetSocket) -> bool {
    s.closed
}

fn is_aborted(s: &NetSocket) -> bool {
    s.abort
}

// ---- addresses and errors ----

fn family(a: &SocketAddr) -> IpAddressFamily {
    match a {
        SocketAddr::V4(_) => IpAddressFamily::Ipv4,
        SocketAddr::V6(_) => IpAddressFamily::Ipv6,
    }
}

fn to_wasi(a: SocketAddr) -> IpSocketAddress {
    match a {
        SocketAddr::V4(v4) => {
            let o = v4.ip().octets();
            IpSocketAddress::Ipv4(Ipv4SocketAddress {
                port: v4.port(),
                address: (o[0], o[1], o[2], o[3]),
            })
        }
        SocketAddr::V6(v6) => {
            let s = v6.ip().segments();
            IpSocketAddress::Ipv6(Ipv6SocketAddress {
                port: v6.port(),
                flow_info: v6.flowinfo(),
                address: (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]),
                scope_id: v6.scope_id(),
            })
        }
    }
}

fn from_wasi(a: IpSocketAddress) -> SocketAddr {
    match a {
        IpSocketAddress::Ipv4(v4) => {
            let (a, b, c, d) = v4.address;
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(a, b, c, d)), v4.port)
        }
        IpSocketAddress::Ipv6(v6) => {
            let (a, b, c, d, e, f, g, h) = v6.address;
            SocketAddr::new(IpAddr::V6(Ipv6Addr::new(a, b, c, d, e, f, g, h)), v6.port)
        }
    }
}

/// The Linux errno for a WASI socket error.
fn errno(e: &ErrorCode) -> Errno {
    match e {
        ErrorCode::AccessDenied => Errno::EACCES,
        ErrorCode::NotSupported => Errno::EOPNOTSUPP,
        ErrorCode::InvalidArgument => Errno::EINVAL,
        ErrorCode::OutOfMemory => Errno::ENOMEM,
        ErrorCode::Timeout => Errno::ETIMEDOUT,
        ErrorCode::InvalidState => Errno::EINVAL,
        ErrorCode::AddressNotBindable => Errno::EADDRNOTAVAIL,
        ErrorCode::AddressInUse => Errno::EADDRINUSE,
        ErrorCode::RemoteUnreachable => Errno::EHOSTUNREACH,
        ErrorCode::ConnectionRefused => Errno::ECONNREFUSED,
        ErrorCode::ConnectionBroken => Errno::EPIPE,
        ErrorCode::ConnectionReset => Errno::ECONNRESET,
        ErrorCode::ConnectionAborted => Errno::ECONNABORTED,
        ErrorCode::DatagramTooLarge => Errno::EMSGSIZE,
        ErrorCode::Other(_) => Errno::EIO,
    }
}

// ---- the provider ----

/// [`Network`] over `wasi:sockets@0.3`.
pub struct WasiNet {
    shared: Rc<Shared>,
}

impl WasiNet {
    pub fn new(shared: Rc<Shared>) -> Self {
        Self { shared }
    }

    fn insert(&self, id: u64, handle: Handle) -> Rc<Notify> {
        let pump = Rc::new(Notify::default());
        self.shared.socks.borrow_mut().insert(
            id,
            Entry {
                handle,
                pump: pump.clone(),
            },
        );
        pump
    }
}

impl Network for WasiNet {
    fn bind(&mut self, sock: &NetRef, addr: SocketAddr) -> Result<(), Errno> {
        let (id, proto, v6only) = {
            let b = sock.borrow();
            (b.id, b.proto, b.v6only)
        };
        // `[::]` without IPV6_V6ONLY also serves IPv4. WASI sockets are always v6-only, so two
        // sockets on one port stand in for Linux's dual stack. IPv4 binds first: a provider
        // whose wildcards overlap across families then still serves IPv4, the common case;
        // the IPv6 twin is best effort.
        let dual = matches!(addr, SocketAddr::V6(a) if a.ip().is_unspecified()) && !v6only;
        let first = if dual {
            SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), addr.port())
        } else {
            addr
        };
        let twin = |port| SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), port);
        // What getsockname reports: the requested address with the bound port.
        let report = |local: SocketAddr| {
            if dual {
                SocketAddr::new(addr.ip(), local.port())
            } else {
                local
            }
        };
        match proto {
            Proto::Tcp => {
                let t = TcpSocket::create(family(&first)).map_err(|e| errno(&e))?;
                t.bind(to_wasi(first)).map_err(|e| errno(&e))?;
                let local = t.get_local_address().map(from_wasi).unwrap_or(first);
                let mut socks = vec![Rc::new(t)];
                if dual
                    && let Ok(t6) = TcpSocket::create(IpAddressFamily::Ipv6)
                    && t6.bind(to_wasi(twin(local.port()))).is_ok()
                {
                    socks.push(Rc::new(t6));
                }
                if sock.borrow().nodelay {
                    for t in &socks {
                        set_no_delay(&self.shared, t, true);
                    }
                }
                sock.borrow_mut().local = Some(report(local));
                self.insert(id, Handle::Tcp(socks));
            }
            Proto::Udp => {
                let u = UdpSocket::create(family(&first)).map_err(|e| errno(&e))?;
                u.bind(to_wasi(first)).map_err(|e| errno(&e))?;
                let local = u.get_local_address().map(from_wasi).unwrap_or(first);
                let mut socks = vec![Rc::new(u)];
                if dual
                    && let Ok(u6) = UdpSocket::create(IpAddressFamily::Ipv6)
                    && u6.bind(to_wasi(twin(local.port()))).is_ok()
                {
                    socks.push(Rc::new(u6));
                }
                sock.borrow_mut().local = Some(report(local));
                let pump = self.insert(id, Handle::Udp(socks.clone()));
                for u in &socks {
                    spawn(
                        &self.shared,
                        udp_rx_pump(self.shared.clone(), sock.clone(), u.clone(), pump.clone()),
                    );
                }
                spawn(
                    &self.shared,
                    udp_tx_pump(self.shared.clone(), sock.clone(), socks, pump),
                );
            }
            Proto::Icmp => self.bind_icmp(sock, addr)?,
        }
        Ok(())
    }

    fn listen(&mut self, sock: &NetRef) -> Result<(), Errno> {
        let (id, backlog) = {
            let b = sock.borrow();
            (b.id, b.backlog)
        };
        let Some(Entry {
            handle: Handle::Tcp(socks),
            pump,
        }) = self.shared.entry(id)
        else {
            return Err(Errno::EINVAL);
        };
        let mut streams = Vec::new();
        for t in &socks {
            // A provider without a real backlog still works; it may refuse simultaneous SYNs.
            let _ = t.set_listen_backlog_size(backlog.max(1) as u64);
            streams.push(t.listen().map_err(|e| errno(&e))?);
        }
        sock.borrow_mut().state = NetState::Listening;
        for st in streams {
            spawn(
                &self.shared,
                accept_pump(self.shared.clone(), sock.clone(), pump.clone(), st),
            );
        }
        Ok(())
    }

    fn connect(&mut self, sock: &NetRef, addr: SocketAddr) -> Result<(), Errno> {
        let (id, proto) = {
            let b = sock.borrow();
            (b.id, b.proto)
        };
        let fam = family(&addr);
        match proto {
            Proto::Tcp => {
                // A socket bound earlier is reused when its family fits (a v6 socket bound
                // to `[::]` connecting to IPv4 uses its IPv4 twin, or a fresh socket).
                let existing = match self.shared.entry(id) {
                    Some(Entry {
                        handle: Handle::Tcp(socks),
                        ..
                    }) => socks.into_iter().find(|t| t.get_address_family() == fam),
                    _ => None,
                };
                let t = match existing {
                    Some(t) => t,
                    None => Rc::new(TcpSocket::create(fam).map_err(|e| errno(&e))?),
                };
                let pump = self.insert(id, Handle::Tcp(vec![t.clone()]));
                {
                    let mut b = sock.borrow_mut();
                    b.state = NetState::Connecting;
                    b.remote = Some(addr);
                }
                spawn(
                    &self.shared,
                    connect_task(self.shared.clone(), sock.clone(), t, pump, addr),
                );
                Ok(())
            }
            Proto::Udp => {
                let Some(Entry {
                    handle: Handle::Udp(socks),
                    ..
                }) = self.shared.entry(id)
                else {
                    return Err(Errno::EINVAL);
                };
                let u = socks
                    .iter()
                    .find(|u| u.get_address_family() == fam)
                    .ok_or(Errno::EAFNOSUPPORT)?;
                u.connect(to_wasi(addr)).map_err(|e| errno(&e))?;
                let mut b = sock.borrow_mut();
                b.state = NetState::Connected;
                b.remote = Some(addr);
                if let Ok(l) = u.get_local_address() {
                    b.local = Some(from_wasi(l));
                }
                Ok(())
            }
            // The kernel keeps an ICMP socket's default destination and names it per request.
            Proto::Icmp => Err(Errno::EINVAL),
        }
    }

    fn kick(&mut self, sock: &NetRef) {
        let (id, closed) = {
            let b = sock.borrow();
            (b.id, b.closed)
        };
        let entry = if closed {
            // The kernel is done with this socket; its pumps finish on their own.
            self.shared.socks.borrow_mut().remove(&id)
        } else {
            self.shared.entry(id)
        };
        if let Some(e) = entry {
            e.pump.notify();
        }
    }

    fn set_option(&mut self, sock: &NetRef, opt: NetOption) -> Result<(), Errno> {
        let id = sock.borrow().id;
        let Some(entry) = self.shared.entry(id) else {
            // Not created yet: the setting is kept in the NetSocket only.
            return Ok(());
        };
        match (&entry.handle, opt) {
            (Handle::Tcp(socks), NetOption::NoDelay(on)) => {
                for t in socks {
                    set_no_delay(&self.shared, t, on);
                }
            }
            (Handle::Tcp(socks), NetOption::KeepAlive(on)) => {
                for t in socks {
                    let _ = t.set_keep_alive_enabled(on);
                }
            }
            (Handle::Tcp(socks), NetOption::HopLimit(h)) => {
                for t in socks {
                    let _ = t.set_hop_limit(h);
                }
            }
            (Handle::Udp(socks), NetOption::HopLimit(h)) => {
                for u in socks {
                    let _ = u.set_unicast_hop_limit(h);
                }
            }
            (Handle::Icmp(i), NetOption::HopLimit(h)) => {
                let _ = i.set_unicast_hop_limit(h);
            }
            _ => {}
        }
        Ok(())
    }

    fn take_changed(&mut self) -> Vec<u64> {
        std::mem::take(&mut *self.shared.changed.borrow_mut())
            .into_iter()
            .collect()
    }

    /// The provider's ids of the WASI sockets behind `sock` (one, or two for a dual-stack
    /// wildcard). An idle socket that never bound has none.
    fn checkpoint(&self, sock: &NetRef) -> Result<Vec<u8>, String> {
        let Some(ext) = self.shared.ext().filter(|e| e.checkpoints()) else {
            return Err("this network provider cannot checkpoint sockets".to_owned());
        };
        let id = sock.borrow().id;
        let mut out = Vec::new();
        let no_id = || "the provider has no id for a socket".to_owned();
        match self.shared.entry(id).map(|e| e.handle) {
            None => out.push(0),
            Some(Handle::Tcp(socks)) => {
                out.push(1);
                out.push(socks.len() as u8);
                for t in &socks {
                    out.extend_from_slice(&ext.tcp_id(t).ok_or_else(no_id)?.to_le_bytes());
                }
            }
            Some(Handle::Udp(socks)) => {
                out.push(2);
                out.push(socks.len() as u8);
                for u in &socks {
                    out.extend_from_slice(&ext.udp_id(u).ok_or_else(no_id)?.to_le_bytes());
                }
            }
            // A checkpoint restores no echo session: a bound ICMP socket refuses the checkpoint,
            // as the netstack refuses an app ICMP socket. One that never bound (no netstack
            // socket yet) is saved like an idle TCP or UDP socket.
            Some(Handle::Icmp(_)) => {
                return Err(
                    "an ICMP echo socket is bound (quiescent snapshots restore no echo session)"
                        .into(),
                );
            }
        }
        Ok(out)
    }

    /// Takes the restored WASI sockets back from the netstack and restarts their pumps: an
    /// accept pump per listening socket, the receive pumps and the send pump of a UDP socket.
    fn restore(&mut self, sock: &NetRef, saved: &[u8]) -> Result<(), String> {
        let Some(ext) = self.shared.ext().filter(|e| e.checkpoints()) else {
            return Err("this network provider cannot restore sockets".to_owned());
        };
        let (id, listening) = {
            let b = sock.borrow();
            (b.id, b.state == NetState::Listening)
        };
        let (&kind, rest) = saved.split_first().ok_or("an empty socket record")?;
        if kind == 0 {
            return Ok(());
        }
        let (&n, ids) = rest.split_first().ok_or("a short socket record")?;
        if ids.len() != usize::from(n) * 4 {
            return Err("a malformed socket record".to_owned());
        }
        let ids: Vec<u32> = ids
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        match kind {
            1 if listening => {
                let mut socks = Vec::new();
                let mut streams = Vec::new();
                for i in ids {
                    let (t, st) = ext.reclaim_listener(i)?;
                    socks.push(Rc::new(t));
                    streams.push(st);
                }
                let pump = self.insert(id, Handle::Tcp(socks));
                for st in streams {
                    spawn(
                        &self.shared,
                        accept_pump(self.shared.clone(), sock.clone(), pump.clone(), st),
                    );
                }
            }
            1 => {
                let socks = ids
                    .into_iter()
                    .map(|i| ext.reclaim_tcp(i).map(Rc::new))
                    .collect::<Result<Vec<_>, _>>()?;
                self.insert(id, Handle::Tcp(socks));
            }
            2 => {
                let socks = ids
                    .into_iter()
                    .map(|i| ext.reclaim_udp(i).map(Rc::new))
                    .collect::<Result<Vec<_>, _>>()?;
                let pump = self.insert(id, Handle::Udp(socks.clone()));
                for u in &socks {
                    spawn(
                        &self.shared,
                        udp_rx_pump(self.shared.clone(), sock.clone(), u.clone(), pump.clone()),
                    );
                }
                spawn(
                    &self.shared,
                    udp_tx_pump(self.shared.clone(), sock.clone(), socks, pump),
                );
            }
            k => return Err(format!("a socket record of kind {k}")),
        }
        Ok(())
    }

    /// The provider-wide state: the id allocator (accepted connections get ids from it).
    fn checkpoint_provider(&self) -> Result<Vec<u8>, String> {
        if !self.shared.ext().is_some_and(|e| e.checkpoints()) {
            return Err("this network provider cannot checkpoint".to_owned());
        }
        if !self.shared.changed.borrow().is_empty() {
            return Err("socket changes the kernel has not collected yet".to_owned());
        }
        Ok(self.shared.next_id.get().to_le_bytes().to_vec())
    }

    fn restore_provider(&mut self, saved: &[u8]) -> Result<(), String> {
        if !self.shared.ext().is_some_and(|e| e.checkpoints()) {
            return Err("this network provider cannot restore".to_owned());
        }
        let b: [u8; 8] = saved
            .try_into()
            .map_err(|_| "a malformed provider record")?;
        self.shared.next_id.set(u64::from_le_bytes(b));
        Ok(())
    }
}

// ---- the provider's extensions ([`SocketExt`]), degrading gracefully without them ----

fn set_no_delay(shared: &Shared, t: &TcpSocket, on: bool) {
    // WASI 0.3 has no TCP_NODELAY knob: without the extension it is accepted and ignored.
    if let Some(ext) = shared.ext() {
        ext.set_no_delay(t, on);
    }
}

fn abort(shared: &Shared, t: &TcpSocket) {
    // Without the extension an abort degrades to dropping the streams (FIN).
    if let Some(ext) = shared.ext() {
        ext.abort(t);
    }
}

// ICMP echo sockets ([`SocketExt::icmp_create`]): the guest's unprivileged ping sockets.
// Addresses carry the echo identifier as their port (local) or 0 (remote), as Linux reports
// them. Without the extension (stock wasmtime) a ping socket cannot reach another host: EACCES,
// as on a Linux whose `ping_group_range` excludes the user.

fn ip_to_wasi(ip: IpAddr) -> IpAddress {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            IpAddress::Ipv4((o[0], o[1], o[2], o[3]))
        }
        IpAddr::V6(v6) => {
            let s = v6.segments();
            IpAddress::Ipv6((s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]))
        }
    }
}

fn ip_from_wasi(ip: IpAddress) -> IpAddr {
    match ip {
        IpAddress::Ipv4((a, b, c, d)) => IpAddr::V4(Ipv4Addr::new(a, b, c, d)),
        IpAddress::Ipv6((a, b, c, d, e, f, g, h)) => {
            IpAddr::V6(Ipv6Addr::new(a, b, c, d, e, f, g, h))
        }
    }
}

impl WasiNet {
    fn bind_icmp(&mut self, sock: &NetRef, addr: SocketAddr) -> Result<(), Errno> {
        let Some(ext) = self.shared.ext() else {
            return Err(Errno::EACCES);
        };
        let id = sock.borrow().id;
        let Some(created) = ext.icmp_create(family(&addr)) else {
            return Err(Errno::EACCES);
        };
        let i = created.map_err(|e| errno(&e))?;
        i.bind(ip_to_wasi(addr.ip()), addr.port())
            .map_err(|e| errno(&e))?;
        let ident = i.identifier().map_err(|e| errno(&e))?;
        // IP_TTL / IPV6_UNICAST_HOPS set before the socket had its network half.
        let hops = {
            let b = sock.borrow();
            [(0, 2), (41, 16)]
                .iter()
                .find_map(|k| b.opts.get(k).copied())
                .and_then(|h| u8::try_from(h).ok())
                .filter(|&h| h > 0)
        };
        if let Some(h) = hops {
            let _ = i.set_unicast_hop_limit(h);
        }
        sock.borrow_mut().local = Some(SocketAddr::new(addr.ip(), ident));
        let pump = self.insert(id, Handle::Icmp(i.clone()));
        spawn(
            &self.shared,
            icmp_rx_pump(self.shared.clone(), sock.clone(), i.clone(), pump.clone()),
        );
        spawn(
            &self.shared,
            icmp_tx_pump(self.shared.clone(), sock.clone(), i, pump),
        );
        Ok(())
    }
}

async fn icmp_rx_pump(shared: Rc<Shared>, s: NetRef, i: Rc<dyn IcmpSocket>, pump: Rc<Notify>) {
    let mut errors = 0;
    loop {
        if s.borrow().closed {
            break;
        }
        // With the reply's TTL, for IP_RECVTTL.
        match race(i.receive_datagram(), until(&s, &pump, is_closed)).await {
            Err(()) => break,
            Ok(Ok(d)) => {
                errors = 0;
                let from = SocketAddr::new(ip_from_wasi(d.remote_address), 0);
                s.borrow_mut().push_datagram_hops(d.data, from, d.hop_limit);
                mark(&shared, &s);
            }
            Ok(Err(_)) => {
                errors += 1;
                if errors >= UDP_RX_ERRORS {
                    break;
                }
                wit_bindgen::yield_async().await;
            }
        }
    }
}

async fn icmp_tx_pump(shared: Rc<Shared>, s: NetRef, i: Rc<dyn IcmpSocket>, pump: Rc<Notify>) {
    loop {
        let g = pump.generation();
        let (next, closed) = {
            let mut b = s.borrow_mut();
            (b.dgrams_out.pop_front(), b.closed)
        };
        let Some((data, to)) = next else {
            if closed {
                break;
            }
            pump.changed(g).await;
            continue;
        };
        mark(&shared, &s); // room in the send queue
        let Some(dst) = to else {
            continue;
        };
        // Like Linux, a request the network refuses later is simply lost.
        let _ = i.send(data, Some(ip_to_wasi(dst.ip()))).await;
    }
}

// ---- pumps ----

async fn connect_task(
    shared: Rc<Shared>,
    s: NetRef,
    t: Rc<TcpSocket>,
    pump: Rc<Notify>,
    addr: SocketAddr,
) {
    let r = t.connect(to_wasi(addr)).await;
    let ok = {
        let mut b = s.borrow_mut();
        match &r {
            Ok(()) => {
                b.state = NetState::Connected;
                b.local = t.get_local_address().ok().map(from_wasi);
            }
            Err(e) => {
                b.state = NetState::Failed;
                b.error = Some(errno(e));
            }
        }
        r.is_ok()
    };
    if ok {
        let nodelay = s.borrow().nodelay;
        if nodelay {
            set_no_delay(&shared, &t, true);
        }
        start_connection(&shared, &s, t, pump);
    }
    mark(&shared, &s);
}

/// Starts the receive and send pumps of a connected socket.
fn start_connection(shared: &Rc<Shared>, s: &NetRef, t: Rc<TcpSocket>, pump: Rc<Notify>) {
    let (rx, rx_done) = t.receive();
    let (tx, tx_reader) = wasip3::wit_stream::new::<u8>();
    let tx_done = t.send(tx_reader);
    spawn(
        shared,
        rx_pump(shared.clone(), s.clone(), pump.clone(), rx, rx_done),
    );
    spawn(
        shared,
        tx_pump(shared.clone(), s.clone(), t, pump, tx, tx_done),
    );
}

async fn accept_pump(
    shared: Rc<Shared>,
    listener: NetRef,
    pump: Rc<Notify>,
    mut incoming: StreamReader<TcpSocket>,
) {
    loop {
        let g = pump.generation();
        let (closed, full) = {
            let l = listener.borrow();
            (l.closed, l.accepted.len() >= l.backlog)
        };
        if closed {
            break;
        }
        if full {
            pump.changed(g).await;
            continue;
        }
        let next = race(incoming.next(), until(&listener, &pump, is_closed)).await;
        let Ok(Some(t)) = next else {
            break;
        };
        let (v6, nodelay) = {
            let l = listener.borrow();
            (l.v6, l.nodelay)
        };
        let id = shared.alloc_id();
        let mut c = NetSocket::new(id, Proto::Tcp, v6);
        c.state = NetState::Connected;
        c.local = t.get_local_address().ok().map(from_wasi);
        c.remote = t.get_remote_address().ok().map(from_wasi);
        c.nodelay = nodelay;
        let c = c.shared();
        if nodelay {
            set_no_delay(&shared, &t, true);
        }
        let t = Rc::new(t);
        let conn_pump = Rc::new(Notify::default());
        shared.socks.borrow_mut().insert(
            id,
            Entry {
                handle: Handle::Tcp(vec![t.clone()]),
                pump: conn_pump.clone(),
            },
        );
        start_connection(&shared, &c, t, conn_pump);
        listener.borrow_mut().accepted.push_back(c);
        mark(&shared, &listener);
    }
}

async fn rx_pump(
    shared: Rc<Shared>,
    s: NetRef,
    pump: Rc<Notify>,
    mut rx: StreamReader<u8>,
    done: FutureReader<Result<(), ErrorCode>>,
) {
    let mut aborted = false;
    loop {
        let g = pump.generation();
        let (space, abort, discard) = {
            let b = s.borrow();
            (b.rx_space(), b.abort, b.closed || b.shut_rd)
        };
        if abort {
            aborted = true;
            break;
        }
        // After close or SHUT_RD the pump keeps reading and discards, so the peer can
        // finish sending.
        let space = if discard { TX_CHUNK } else { space };
        if space == 0 {
            pump.changed(g).await; // the guest read: the window opened
            continue;
        }
        let read = race(
            rx.read(Vec::with_capacity(space)),
            until(&s, &pump, is_aborted),
        )
        .await;
        let Ok((result, data)) = read else {
            aborted = true;
            break;
        };
        if !data.is_empty() {
            {
                let mut b = s.borrow_mut();
                if !(b.closed || b.shut_rd) {
                    b.rx.extend(data);
                }
            }
            mark(&shared, &s);
        }
        if !matches!(result, StreamResult::Complete(_)) {
            break;
        }
    }
    drop(rx);
    if aborted {
        let mut b = s.borrow_mut();
        b.rx_end.get_or_insert(Err(Errno::ECONNRESET));
        drop(b);
        mark(&shared, &s);
        return;
    }
    let end = done.await;
    s.borrow_mut().rx_end = Some(end.map_err(|e| errno(&e)));
    mark(&shared, &s);
}

async fn tx_pump(
    shared: Rc<Shared>,
    s: NetRef,
    t: Rc<TcpSocket>,
    pump: Rc<Notify>,
    mut tx: StreamWriter<u8>,
    done: FutureReader<Result<(), ErrorCode>>,
) {
    let mut aborted = false;
    loop {
        let g = pump.generation();
        let (chunk, fin, abort) = {
            let mut b = s.borrow_mut();
            let n = b.tx.len().min(TX_CHUNK);
            let chunk: Vec<u8> = b.tx.drain(..n).collect();
            b.tx_inflight = n;
            (chunk, b.shut_wr || b.closed, b.abort)
        };
        if abort {
            aborted = true;
            break;
        }
        if chunk.is_empty() {
            if fin {
                break;
            }
            pump.changed(g).await; // the guest wrote, shut down or closed
            continue;
        }
        let wrote = race(tx.write_all(chunk), until(&s, &pump, is_aborted)).await;
        let Ok(rest) = wrote else {
            aborted = true;
            break;
        };
        let broken = !rest.is_empty();
        {
            let mut b = s.borrow_mut();
            b.tx_inflight = 0;
            if broken {
                b.tx_err.get_or_insert(Errno::EPIPE);
            }
        }
        mark(&shared, &s);
        if broken {
            break;
        }
    }
    if aborted {
        // RST where the provider can ([`SocketExt::abort`]), else a FIN.
        abort(&shared, &t);
    }
    drop(tx); // FIN after the drained send buffer
    let end = if aborted {
        Err(ErrorCode::ConnectionReset)
    } else {
        done.await
    };
    {
        let mut b = s.borrow_mut();
        b.tx_done = true;
        b.tx_inflight = 0;
        if let Err(e) = &end
            && !aborted
        {
            b.tx_err.get_or_insert(errno(e));
        }
    }
    mark(&shared, &s);
}

async fn udp_rx_pump(shared: Rc<Shared>, s: NetRef, u: Rc<UdpSocket>, pump: Rc<Notify>) {
    let mut errors = 0;
    loop {
        if s.borrow().closed {
            break;
        }
        let got = race(u.receive(), until(&s, &pump, is_closed)).await;
        match got {
            Err(()) => break,
            Ok(Ok((data, from))) => {
                errors = 0;
                s.borrow_mut().push_datagram(data, from_wasi(from));
                mark(&shared, &s);
            }
            Ok(Err(_)) => {
                // ICMP errors and the like: Linux reports them asynchronously, if at all.
                errors += 1;
                if errors >= UDP_RX_ERRORS {
                    break;
                }
                wit_bindgen::yield_async().await;
            }
        }
    }
}

async fn udp_tx_pump(shared: Rc<Shared>, s: NetRef, socks: Vec<Rc<UdpSocket>>, pump: Rc<Notify>) {
    loop {
        let g = pump.generation();
        let (next, closed, peer) = {
            let mut b = s.borrow_mut();
            (b.dgrams_out.pop_front(), b.closed, b.remote)
        };
        let Some((data, to)) = next else {
            if closed {
                break;
            }
            pump.changed(g).await;
            continue;
        };
        mark(&shared, &s); // room in the send queue
        let Some(dst) = to.or(peer) else {
            continue;
        };
        let fam = family(&dst);
        let Some(u) = socks.iter().find(|u| u.get_address_family() == fam) else {
            continue;
        };
        // Like Linux, a failed send is not reported to this datagram's sender.
        let _ = u.send(data, to.map(to_wasi)).await;
    }
}
