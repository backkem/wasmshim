//! AF_UNIX sockets, entirely inside the personality.
//!
//! A connected pair is two `UnixSocket`s pointing at each other (weakly). Sending writes into
//! the *peer's* receive queue, so a writer that finds it full waits on the peer's key, and a
//! reader that drains its queue wakes it. Path names live in the VFS as socket inodes; abstract
//! names (leading NUL) in a kernel table.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::{Rc, Weak};

use wasmshim_linux_abi::Errno;

use super::{EPOLLERR, EPOLLHUP, EPOLLIN, EPOLLOUT, EPOLLRDHUP, EPOLLRDNORM, EPOLLWRNORM};
use crate::fd::FileRef;
use crate::kernel::{Kernel, Pid};

/// Per-socket receive capacity (Linux's default `SO_RCVBUF` for AF_UNIX).
pub(crate) const UNIX_BUF: usize = 212_992;

pub(crate) type UnixRef = Rc<RefCell<UnixSocket>>;

/// Socket type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SockType {
    Stream,
    Dgram,
    SeqPacket,
}

impl SockType {
    pub(crate) fn raw(self) -> u32 {
        match self {
            Self::Stream => 1,
            Self::Dgram => 2,
            Self::SeqPacket => 5,
        }
    }
}

/// Connection state.
#[derive(Debug)]
pub(crate) enum UnixState {
    Unconnected,
    Listening {
        backlog: usize,
        queue: VecDeque<UnixRef>,
    },
    Connected,
}

/// One AF_UNIX socket.
#[derive(Debug)]
pub(crate) struct UnixSocket {
    pub id: u64,
    pub ty: SockType,
    pub state: UnixState,
    /// Address family (`AF_UNIX`, `AF_INET`, `AF_INET6`).
    pub family: u16,
    /// Bound (local) address.
    pub name: Option<SockName>,
    /// Where the bound name is registered (removed on close).
    pub reg: Option<UnixName>,
    pub peer: Weak<RefCell<UnixSocket>>,
    /// The peer's address as of connection time (still reported after it closed).
    pub peer_name: Option<SockName>,
    /// Stream data.
    pub rx: VecDeque<u8>,
    /// Datagrams and seqpackets, with the sender's address.
    pub msgs: VecDeque<(Vec<u8>, Option<SockName>)>,
    /// Descriptions in flight (`SCM_RIGHTS`) on the stream, with the absolute stream positions
    /// `[start, end)` of the write that carried them (counted by `rx_read`).
    pub rx_fds: VecDeque<(u64, u64, Vec<FileRef>)>,
    /// Stream bytes consumed so far (the position of the first byte in `rx`).
    pub rx_read: u64,
    /// Descriptions in flight with each message of `msgs` (same order; empty for none).
    pub msg_fds: VecDeque<Vec<FileRef>>,
    /// No more data will arrive (peer closed or shut down its writing, or we shut reading).
    pub read_shut: bool,
    /// We shut down writing.
    pub write_shut: bool,
    /// The peer is gone.
    pub peer_gone: bool,
    /// The creating process (`SO_PEERCRED`).
    pub owner: Pid,
    /// `IPV6_V6ONLY` (an AF_INET6 socket that does not accept IPv4).
    pub v6only: bool,
    /// An ICMP echo ("ping") socket (`SOCK_DGRAM` + `IPPROTO_ICMP[V6]`): its
    /// messages are ICMP echo requests and replies, its port is the echo identifier.
    pub icmp: bool,
    /// The TTL / hop limit of the last datagram received, if known: what `recvmsg` reports as
    /// `IP_TTL` / `IPV6_HOPLIMIT` when the socket asked (`IP_RECVTTL`, `IPV6_RECVHOPLIMIT`).
    pub last_hops: Option<u8>,
    pub generation: u64,
    /// The network half of an AF_INET/AF_INET6 listener or UDP socket bound beyond the
    /// loopback: accepted connections and datagrams from other hosts arrive there.
    pub net: Option<crate::net::NetRef>,
    /// `SO_RCVTIMEO` / `SO_SNDTIMEO` in ns.
    pub rcvtimeo: Option<u64>,
    pub sndtimeo: Option<u64>,
    /// `TCP_NODELAY`, carried to the network half.
    pub nodelay: bool,
    /// Other integer options as last set, read back by `getsockopt`.
    pub opts: std::collections::BTreeMap<(u64, u64), u32>,
    /// Loopback TCP only: we wrote to a peer that had closed, so (as on Linux, where the peer
    /// answers with RST) the connection is reset: further sends fail with `EPIPE`.
    pub reset: bool,
    /// The reset's `ECONNRESET`, not yet reported by a receive.
    pub reset_pending: bool,
}

impl UnixSocket {
    pub(crate) fn new(id: u64, family: u16, ty: SockType, owner: Pid) -> Self {
        Self {
            id,
            ty,
            state: UnixState::Unconnected,
            family,
            name: None,
            reg: None,
            peer: Weak::new(),
            peer_name: None,
            rx: VecDeque::new(),
            msgs: VecDeque::new(),
            rx_fds: VecDeque::new(),
            rx_read: 0,
            msg_fds: VecDeque::new(),
            read_shut: false,
            write_shut: false,
            peer_gone: false,
            owner,
            v6only: false,
            icmp: false,
            last_hops: None,
            generation: 0,
            net: None,
            rcvtimeo: None,
            sndtimeo: None,
            nodelay: false,
            opts: std::collections::BTreeMap::new(),
            reset: false,
            reset_pending: false,
        }
    }

    /// An AF_INET/AF_INET6 stream on the in-kernel loopback: TCP's semantics, not AF_UNIX's.
    pub(crate) fn is_tcp(&self) -> bool {
        self.family != 1 && self.ty == SockType::Stream
    }

    pub(crate) fn queued(&self) -> usize {
        self.rx.len() + self.msgs.iter().map(|(m, _)| m.len()).sum::<usize>()
    }

    pub(crate) fn has_data(&self) -> bool {
        !self.rx.is_empty() || !self.msgs.is_empty() || self.net_has_datagram()
    }

    /// `FIONREAD`: stream bytes waiting, or the size of the next datagram.
    pub(crate) fn readable_bytes(&self) -> usize {
        if self.ty == SockType::Stream {
            return self.rx.len();
        }
        match self.msgs.front() {
            Some((m, _)) => m.len(),
            None => self.net.as_ref().map_or(0, |n| {
                n.borrow().dgrams_in.front().map_or(0, |(d, ..)| d.len())
            }),
        }
    }

    /// A datagram from another host is waiting.
    fn net_has_datagram(&self) -> bool {
        self.net
            .as_ref()
            .is_some_and(|n| !n.borrow().dgrams_in.is_empty())
    }

    /// Connections accepted by the network half are waiting.
    pub(crate) fn net_has_accepted(&self) -> bool {
        self.net
            .as_ref()
            .is_some_and(|n| !n.borrow().accepted.is_empty())
    }

    /// Readiness (`EPOLL*` bits) and generation.
    pub(crate) fn events(&self) -> (u32, u64) {
        let mut ev = 0;
        match &self.state {
            UnixState::Listening { queue, .. } => {
                if !queue.is_empty() || self.net_has_accepted() {
                    ev |= EPOLLIN | EPOLLRDNORM;
                }
            }
            UnixState::Connected => {
                if self.has_data() || self.read_shut {
                    ev |= EPOLLIN | EPOLLRDNORM;
                }
                if self.read_shut {
                    ev |= EPOLLRDHUP;
                }
                let writable = match self.peer.upgrade() {
                    Some(p) => p.borrow().queued() < UNIX_BUF,
                    // A TCP peer that closed still accepts a write (it answers with RST).
                    None => self.is_tcp() && !self.reset,
                };
                if writable && !self.write_shut {
                    ev |= EPOLLOUT | EPOLLWRNORM;
                }
                let hup = if self.is_tcp() {
                    self.reset || (self.read_shut && self.write_shut)
                } else {
                    self.peer_gone || (self.read_shut && self.write_shut)
                };
                if hup {
                    ev |= EPOLLHUP;
                }
                if self.reset {
                    ev |= EPOLLERR;
                }
            }
            UnixState::Unconnected => {
                if self.ty == SockType::Dgram {
                    ev |= EPOLLOUT | EPOLLWRNORM;
                    if self.has_data() {
                        ev |= EPOLLIN | EPOLLRDNORM;
                    }
                } else {
                    ev |= EPOLLOUT | EPOLLHUP;
                }
            }
        }
        let net_gen = self.net.as_ref().map_or(0, |n| n.borrow().generation);
        (ev, self.generation.wrapping_add(net_gen))
    }
}

/// Wakes a socket's waiters (after its state changed).
pub(crate) fn touch(k: &mut Kernel, s: &UnixRef) {
    let id = {
        let mut b = s.borrow_mut();
        b.generation += 1;
        b.id
    };
    k.object_changed(id);
}

impl Kernel {
    /// Drops references to descriptions that were in flight (`SCM_RIGHTS`) and not delivered,
    /// closing each one whose last reference this was.
    pub(crate) fn release_files(&mut self, files: Vec<FileRef>) {
        for f in files {
            self.release_file(&f);
        }
    }

    /// Connects two sockets to each other.
    pub(crate) fn unix_pair(a: &UnixRef, b: &UnixRef) {
        a.borrow_mut().peer = Rc::downgrade(b);
        b.borrow_mut().peer = Rc::downgrade(a);
        let (na, nb) = (a.borrow().name.clone(), b.borrow().name.clone());
        a.borrow_mut().peer_name = nb;
        b.borrow_mut().peer_name = na;
        a.borrow_mut().state = UnixState::Connected;
        b.borrow_mut().state = UnixState::Connected;
    }

    /// Sends on a socket: stream bytes or one message. `to` is an explicit destination for
    /// unconnected datagram sockets.
    pub(crate) fn unix_send(
        &mut self,
        s: &UnixRef,
        data: &[u8],
        to: Option<UnixRef>,
        nonblock: bool,
    ) -> Result<usize, Errno> {
        self.unix_send_with(s, data, to, nonblock, Vec::new())
    }

    /// [`Kernel::unix_send`] passing descriptions (`SCM_RIGHTS`): they arrive with the first
    /// byte (stream) or with the message. On failure they are released.
    pub(crate) fn unix_send_with(
        &mut self,
        s: &UnixRef,
        data: &[u8],
        to: Option<UnixRef>,
        nonblock: bool,
        fds: Vec<FileRef>,
    ) -> Result<usize, Errno> {
        let mut fds = Some(fds);
        let r = self.unix_send_inner(s, data, to, nonblock, &mut fds);
        // Not queued (an error, or no byte on a stream): the references go away here.
        self.release_files(fds.unwrap_or_default());
        r
    }

    fn unix_send_inner(
        &mut self,
        s: &UnixRef,
        data: &[u8],
        to: Option<UnixRef>,
        nonblock: bool,
        fds: &mut Option<Vec<FileRef>>,
    ) -> Result<usize, Errno> {
        let (ty, write_shut, name) = {
            let b = s.borrow();
            (b.ty, b.write_shut, b.name.clone())
        };
        if write_shut {
            return Err(Errno::EPIPE);
        }
        // The peer is looked up first: a `borrow()` in the scrutinee would live through the
        // whole match, and the "peer gone" arm borrows mutably.
        let peer = if to.is_none() {
            s.borrow().peer.upgrade()
        } else {
            None
        };
        let target = match to {
            Some(t) => t,
            None => match peer {
                Some(p) => p,
                None => {
                    let mut b = s.borrow_mut();
                    if b.is_tcp()
                        && b.peer_gone
                        && !b.reset
                        && matches!(b.state, UnixState::Connected)
                    {
                        // Linux: the first write to a closed TCP peer succeeds; the peer
                        // resets the connection (undici then retries an idempotent request).
                        b.reset = true;
                        b.reset_pending = true;
                        b.read_shut = true;
                        b.generation += 1;
                        drop(b);
                        touch(self, s);
                        return Ok(data.len());
                    }
                    return Err(match (&b.state, b.peer_gone) {
                        (_, true) => Errno::EPIPE,
                        (UnixState::Connected, false) => Errno::EPIPE,
                        _ if ty == SockType::Dgram => Errno::EDESTADDRREQ,
                        _ => Errno::ENOTCONN,
                    });
                }
            },
        };
        let n = {
            let mut t = target.borrow_mut();
            if t.read_shut && ty == SockType::Stream {
                return Err(Errno::EPIPE);
            }
            let space = UNIX_BUF.saturating_sub(t.queued());
            let needed = if ty == SockType::Stream {
                1
            } else {
                data.len().max(1)
            };
            if ty != SockType::Stream && data.len() > UNIX_BUF {
                return Err(Errno::EMSGSIZE);
            }
            if space < needed {
                if nonblock {
                    return Err(Errno::EAGAIN);
                }
                let id = t.id;
                drop(t);
                let timeout = s.borrow().sndtimeo;
                return Err(self.wait_sock(id, timeout));
            }
            if ty == SockType::Stream {
                let n = data.len().min(space);
                if n > 0
                    && let Some(f) = fds.take_if(|f| !f.is_empty())
                {
                    let at = t.rx_read + t.rx.len() as u64;
                    t.rx_fds.push_back((at, at + n as u64, f));
                }
                t.rx.extend(&data[..n]);
                n
            } else {
                t.msgs.push_back((data.to_vec(), name));
                t.msg_fds.push_back(fds.take().unwrap_or_default());
                data.len()
            }
        };
        touch(self, &target);
        Ok(n)
    }

    /// Receives from a socket. Returns (bytes, sender address, the message's whole length:
    /// more than the bytes when a datagram was truncated).
    pub(crate) fn unix_recv(
        &mut self,
        s: &UnixRef,
        buf: &mut [u8],
        peek: bool,
        nonblock: bool,
    ) -> Result<(usize, Option<SockName>, usize), Errno> {
        let mut fds = Vec::new();
        let r = self.unix_recv_with(s, buf, peek, nonblock, &mut fds);
        // A plain read discards descriptions in flight, as on Linux.
        self.release_files(fds);
        r
    }

    /// [`Kernel::unix_recv`] that also takes the descriptions in flight with what it reads
    /// (`SCM_RIGHTS`) into `fds`. As on Linux, a stream read that reaches a write carrying
    /// descriptions takes them and stops at that write's end, so each set arrives with exactly
    /// one read. A peek takes none.
    pub(crate) fn unix_recv_with(
        &mut self,
        s: &UnixRef,
        buf: &mut [u8],
        peek: bool,
        nonblock: bool,
        fds: &mut Vec<FileRef>,
    ) -> Result<(usize, Option<SockName>, usize), Errno> {
        let result = {
            let mut b = s.borrow_mut();
            if matches!(b.state, UnixState::Listening { .. }) {
                return Err(Errno::EINVAL);
            }
            if !b.has_data() {
                if b.reset_pending {
                    b.reset_pending = false;
                    return Err(Errno::ECONNRESET);
                }
                if b.read_shut {
                    return Ok((0, None, 0));
                }
                if b.ty != SockType::Dgram && !matches!(b.state, UnixState::Connected) {
                    return Err(Errno::ENOTCONN);
                }
                if nonblock {
                    return Err(Errno::EAGAIN);
                }
                let (id, timeout) = (b.id, b.rcvtimeo);
                drop(b);
                return Err(self.wait_sock(id, timeout));
            }
            if b.ty == SockType::Stream {
                // Linux: a read that reaches a write carrying descriptions takes them and stops
                // at the end of that write.
                let head = b.rx_read;
                let mark = b.rx_fds.front().map(|&(start, end, _)| (start, end));
                let limit = mark.map_or(b.rx.len(), |(_, end)| {
                    (end.saturating_sub(head) as usize).min(b.rx.len())
                });
                let n = buf.len().min(limit);
                for (dst, src) in buf.iter_mut().zip(b.rx.iter().take(n)) {
                    *dst = *src;
                }
                if !peek {
                    if mark.is_some_and(|(start, _)| start < head + n as u64)
                        && let Some((_, _, f)) = b.rx_fds.pop_front()
                    {
                        fds.extend(f);
                    }
                    b.rx.drain(..n);
                    b.rx_read += n as u64;
                }
                (n, None, n)
            } else {
                let local = if peek {
                    b.msgs.front().cloned()
                } else {
                    let m = b.msgs.pop_front();
                    if m.is_some() {
                        fds.extend(b.msg_fds.pop_front().unwrap_or_default());
                    }
                    m
                };
                // This host's ICMP echo replies (answered in the kernel) left with TTL 64.
                let local_hops = (local.is_some() && b.icmp).then_some(64);
                // Datagrams from other hosts queue in the network half.
                let remote = || {
                    let net = b.net.as_ref()?;
                    let mut n = net.borrow_mut();
                    let (data, from, hops) = if peek {
                        n.dgrams_in.front().cloned()
                    } else {
                        n.dgrams_in.pop_front()
                    }?;
                    Some((data, Some(SockName::Inet(from)), hops))
                };
                let (msg, from, hops) = match local {
                    Some((msg, from)) => (msg, from, local_hops),
                    None => remote().unwrap_or_default(),
                };
                b.last_hops = hops;
                let n = buf.len().min(msg.len());
                buf[..n].copy_from_slice(&msg[..n]);
                (n, from, msg.len())
            }
        };
        if !peek {
            touch(self, s);
        }
        Ok(result)
    }

    /// The last description of a socket closed.
    pub(crate) fn unix_closed(&mut self, s: &UnixRef) {
        // Descriptions still in flight to this socket are closed with it.
        let in_flight: Vec<FileRef> = {
            let mut b = s.borrow_mut();
            let stream = std::mem::take(&mut b.rx_fds)
                .into_iter()
                .flat_map(|(_, _, f)| f);
            let msgs = std::mem::take(&mut b.msg_fds).into_iter().flatten();
            stream.chain(msgs).collect()
        };
        self.release_files(in_flight);
        let (peer, queue, bound) = {
            let mut b = s.borrow_mut();
            let queue = match std::mem::replace(&mut b.state, UnixState::Unconnected) {
                UnixState::Listening { queue, .. } => queue,
                _ => VecDeque::new(),
            };
            (b.peer.upgrade(), queue, b.reg.take())
        };
        if let Some(key) = bound {
            self.unix_names.remove(&key);
        }
        let net = s.borrow_mut().net.take();
        if let Some(net) = net {
            self.inet_closed(&net);
        }
        // Pending connections that were never accepted see their peer vanish.
        for pending in queue {
            if let Some(client) = pending.borrow().peer.upgrade() {
                let mut c = client.borrow_mut();
                c.read_shut = true;
                c.peer_gone = true;
                drop(c);
                touch(self, &client);
            }
        }
        if let Some(p) = peer {
            {
                let mut pb = p.borrow_mut();
                pb.read_shut = true;
                pb.peer_gone = true;
            }
            touch(self, &p);
        }
    }
}

/// How a bound name is registered.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum UnixName {
    /// A path, by its VFS socket inode.
    Ino(u64),
    /// An abstract name (with its leading NUL).
    Abstract(Vec<u8>),
    /// An AF_INET/AF_INET6 port on the loopback: (stream?, IPv6 socket?, port). TCP and UDP,
    /// and v4 and v6 sockets, are separate spaces; dual-stack rules are applied on top.
    Port(bool, bool, u16),
}

/// A socket address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SockName {
    /// `sun_path` bytes (abstract names start with NUL).
    Unix(Vec<u8>),
    /// An IP address and port.
    Inet(std::net::SocketAddr),
}
