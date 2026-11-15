//! Socket syscalls.
//!
//! * In-kernel local sockets: AF_UNIX, and AF_INET/AF_INET6 on the loopback (127.0.0.0/8,
//!   `::1`, and the unspecified addresses as "this host"). Both families share one
//!   implementation (`objects::unix`): a connected pair of receive queues, a listener's accept
//!   queue, datagram queues. They differ only in naming: paths and abstract names, or ports with
//!   deterministic ephemeral allocation (from 49152).
//! * Everything else goes to the network provider (`crate::net`): a TCP `connect` to
//!   another host turns the description into a provider connection (`FileKind::Inet`), and a
//!   listener or UDP socket bound beyond the loopback gets a network half next to its local
//!   name, so it serves both in-node and remote peers. Without a provider such traffic is
//!   `ENETUNREACH`.

use std::cell::RefCell;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use std::rc::Rc;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::{open, signal};
use wasmshim_linux_abi::structs::{Iovec, Timespec};

use super::SysResult;
use crate::fd::{FileKind, FileRef, open_file};
use crate::host::Host;
use crate::kernel::{Kernel, Tid, WaitKey};
use crate::net::{NetOption, NetRef, NetSocket, NetState, Proto};
use crate::objects::unix::{SockName, SockType, UnixName, UnixRef, UnixSocket, UnixState, touch};
use crate::uaccess::{read_array, read_bytes, write_bytes, write_u32};
use crate::vfs::Node;

const AF_UNIX: u16 = 1;
const AF_INET: u16 = 2;
const AF_INET6: u16 = 10;
use super::netlink::AF_NETLINK;
const SOCK_TYPE_MASK: u64 = 0xf;
const SOCK_RAW: u64 = 3;
const IPPROTO_ICMP: u64 = 1;
const IPPROTO_ICMPV6: u64 = 58;
const SOCK_NONBLOCK: u64 = open::O_NONBLOCK;
const SOCK_CLOEXEC: u64 = open::O_CLOEXEC;
const MSG_PEEK: u64 = 0x2;
const MSG_TRUNC: u64 = 0x20;
const MSG_DONTWAIT: u64 = 0x40;
const MSG_NOSIGNAL: u64 = 0x4000;
const SOL_SOCKET: u64 = 1;
const SCM_RIGHTS: u32 = 1;
const SCM_CREDENTIALS: u32 = 2;
/// Most descriptions one `SCM_RIGHTS` message may carry (Linux's `SCM_MAX_FD`).
const SCM_MAX_FD: usize = 253;
const MSG_CTRUNC: u32 = 0x8;
const MSG_CMSG_CLOEXEC: u64 = 0x4000_0000;
/// `sizeof(struct cmsghdr)` on 64-bit: `cmsg_len` (8), `cmsg_level` (4), `cmsg_type` (4).
const CMSG_HDR: usize = 16;

/// Where a send on an in-kernel socket goes.
enum Route {
    /// Into this socket's receive queue (`None`: the connected peer).
    Socket(Option<UnixRef>),
    /// Already handled elsewhere (a network datagram, or lost): this many bytes.
    Done(usize),
}

/// What a receive of `n` bytes from a message of `full` bytes returns: recv(2), MSG_TRUNC
/// "return the real length of the packet or datagram, even when it was longer than the passed
/// buffer".
fn received(n: usize, full: usize, flags: u64) -> u64 {
    if flags & MSG_TRUNC != 0 {
        full.max(n) as u64
    } else {
        n as u64
    }
}

/// `CMSG_ALIGN`: control messages are 8-byte aligned on 64-bit.
fn cmsg_align(n: usize) -> usize {
    (n + 7) & !7
}
const SO_TYPE: u64 = 3;
const SO_ERROR: u64 = 4;
const SO_SNDBUF: u64 = 7;
const SO_RCVBUF: u64 = 8;
const SO_KEEPALIVE: u64 = 9;
const SO_LINGER: u64 = 13;
const SO_RCVTIMEO: u64 = 20;
const SO_SNDTIMEO: u64 = 21;
const SO_RCVTIMEO_NEW: u64 = 66;
const SO_SNDTIMEO_NEW: u64 = 67;
const SO_PEERCRED: u64 = 17;
const SO_ACCEPTCONN: u64 = 30;
const SO_PROTOCOL: u64 = 38;
const SO_DOMAIN: u64 = 39;
const SO_BINDTODEVICE: u64 = 25;
const IPPROTO_IP: u64 = 0;
const IP_TTL: u64 = 2;
const IPPROTO_TCP: u64 = 6;
const TCP_NODELAY: u64 = 1;
const TCP_CONGESTION: u64 = 13;
const TCP_ULP: u64 = 31;
const TCP_FASTOPEN_KEY: u64 = 33;
const IPPROTO_IPV6: u64 = 41;
const IPV6_UNICAST_HOPS: u64 = 16;
const IPV6_V6ONLY: u64 = 26;
/// IPv6 options a zero length removes: `IPV6_2292PKTOPTIONS`, `IPV6_HOPOPTS`,
/// `IPV6_RTHDRDSTOPTS`, `IPV6_RTHDR`, `IPV6_DSTOPTS`.
const IPV6_STICKY_OPTIONS: [u64; 5] = [6, 54, 55, 57, 59];
const SUN_PATH_MAX: usize = 108;
/// `struct msghdr` on LP64.
const MSGHDR_SIZE: usize = 56;
/// First ephemeral port (`net.ipv4.ip_local_port_range`'s usual start).
const EPHEMERAL_START: u16 = 49152;

/// `setsockopt`'s length check for integer options: `optlen < sizeof(int)` is EINVAL at the
/// socket, TCP, UDP and IPv6 levels (net/core/sock.c `sk_setsockopt`, net/ipv4/tcp.c
/// `do_tcp_setsockopt`, net/ipv4/udp.c `udp_lib_setsockopt`, net/ipv6/ipv6_sockglue.c). The IP
/// level takes a single byte, and a length of 0 as the value 0 (net/ipv4/ip_sockglue.c
/// `do_ip_setsockopt`). Options that take a string, or a length of 0 to remove them, are not
/// integers.
fn sockopt_int_len(level: u64, opt: u64, len: u32) -> Result<(), Errno> {
    /// `IPPROTO_UDP`.
    const IPPROTO_UDP: u64 = 17;
    let integer = match level {
        SOL_SOCKET => opt != SO_BINDTODEVICE,
        IPPROTO_TCP => !matches!(opt, TCP_CONGESTION | TCP_ULP | TCP_FASTOPEN_KEY),
        IPPROTO_UDP => true,
        IPPROTO_IPV6 => !(len == 0 && IPV6_STICKY_OPTIONS.contains(&opt)),
        _ => false,
    };
    if integer && len < 4 {
        return Err(Errno::EINVAL);
    }
    Ok(())
}

fn sock_type(raw: u64) -> Result<SockType, Errno> {
    match raw & SOCK_TYPE_MASK {
        1 => Ok(SockType::Stream),
        2 => Ok(SockType::Dgram),
        5 => Ok(SockType::SeqPacket),
        _ => Err(Errno::ESOCKTNOSUPPORT),
    }
}

/// Is this address "this host" (loopback or unspecified)?
fn is_local(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_unspecified(),
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6
                    .to_ipv4_mapped()
                    .is_some_and(|v4| v4.is_loopback() || v4.is_unspecified())
        }
    }
}

/// Encodes an address for a socket of `family` (an unnamed socket encodes as the family's
/// "nothing": `sa_family` alone, or the unspecified address and port 0).
fn encode(name: Option<&SockName>, family: u16) -> Vec<u8> {
    if family == AF_NETLINK {
        // Everything a netlink socket receives comes from the kernel (port 0).
        return super::netlink::sockaddr_nl(0, 0);
    }
    let mut out = family.to_le_bytes().to_vec();
    match (name, family) {
        (Some(SockName::Unix(path)), _) => {
            out.extend_from_slice(path);
            if path.first() != Some(&0) {
                out.push(0);
            }
        }
        (inet, AF_INET) => {
            let a = match inet {
                Some(SockName::Inet(SocketAddr::V4(a))) => *a,
                Some(SockName::Inet(SocketAddr::V6(a))) => SocketAddrV4::new(
                    a.ip().to_ipv4_mapped().unwrap_or(Ipv4Addr::UNSPECIFIED),
                    a.port(),
                ),
                _ => SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0),
            };
            out.extend_from_slice(&a.port().to_be_bytes());
            out.extend_from_slice(&a.ip().octets());
            out.extend_from_slice(&[0; 8]);
        }
        (inet, AF_INET6) => {
            let a = match inet {
                Some(SockName::Inet(SocketAddr::V6(a))) => *a,
                Some(SockName::Inet(SocketAddr::V4(a))) => {
                    SocketAddrV6::new(a.ip().to_ipv6_mapped(), a.port(), 0, 0)
                }
                _ => SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, 0, 0, 0),
            };
            out.extend_from_slice(&a.port().to_be_bytes());
            out.extend_from_slice(&a.flowinfo().to_be_bytes());
            out.extend_from_slice(&a.ip().octets());
            out.extend_from_slice(&a.scope_id().to_le_bytes());
        }
        _ => {}
    }
    out
}

/// A socket behind a descriptor.
enum Sock {
    /// In-kernel (maybe with a network half).
    Local(UnixRef),
    /// A provider TCP connection.
    Net(NetRef),
}

/// The address a provider sees: IPv4-mapped IPv6 addresses become plain IPv4 (WASI rejects
/// mapped addresses; an `AF_INET6` socket reaches IPv4 peers that way).
fn unmap(a: SocketAddr) -> SocketAddr {
    match a {
        SocketAddr::V6(v6) => match v6.ip().to_ipv4_mapped() {
            Some(v4) => SocketAddr::new(IpAddr::V4(v4), v6.port()),
            None => a,
        },
        v4 => v4,
    }
}

/// The unspecified address of a family.
fn wildcard(family: u16) -> IpAddr {
    if family == AF_INET6 {
        IpAddr::V6(Ipv6Addr::UNSPECIFIED)
    } else {
        IpAddr::V4(Ipv4Addr::UNSPECIFIED)
    }
}

/// Is this a loopback address (including IPv4-mapped loopback)?
fn is_loopback(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => {
            v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
    }
}

impl Kernel {
    /// Is `ip` one of this node's interface addresses (`cfg.interfaces`, IPv4-mapped included)?
    /// Linux routes those through `lo` (the `local` table), so they are "this host" too: a
    /// program that connects to its own interface address (K3s' supervisor dialling its
    /// advertise address) reaches its own listeners in the kernel.
    fn own_addr(&self, ip: IpAddr) -> bool {
        let ip = match ip {
            IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
            v4 => v4,
        };
        self.cfg
            .interfaces
            .iter()
            .any(|i| i.addrs.iter().any(|(a, _)| *a == ip))
    }

    /// Is this address "this host": loopback, unspecified, or an address of our own interfaces?
    fn this_host(&self, ip: IpAddr) -> bool {
        is_local(ip) || self.own_addr(ip)
    }

    fn sock_of(&self, tid: Tid, fd: u64) -> Result<(UnixRef, bool), Errno> {
        match self.any_sock(tid, fd)? {
            (Sock::Local(s), nb) => Ok((s, nb)),
            // A provider connection is connected and bound: what local-only calls want fails.
            (Sock::Net(_), _) => Err(Errno::EINVAL),
        }
    }

    fn any_sock(&self, tid: Tid, fd: u64) -> Result<(Sock, bool), Errno> {
        let file = self.file(tid, fd)?;
        let f = file.borrow();
        let nb = f.flags & open::O_NONBLOCK != 0;
        match &f.kind {
            FileKind::Unix(s) => Ok((Sock::Local(s.clone()), nb)),
            FileKind::Inet(s) => Ok((Sock::Net(s.clone()), nb)),
            _ => Err(Errno::ENOTSOCK),
        }
    }

    fn install_net(&mut self, tid: Tid, s: NetRef, flags: u64) -> SysResult {
        let file = open_file(
            FileKind::Inet(s),
            open::O_RDWR | (flags & SOCK_NONBLOCK),
            b"socket:",
        );
        Ok(self
            .fds(tid)
            .borrow_mut()
            .insert(file, flags & SOCK_CLOEXEC != 0, 0)? as u64)
    }

    /// Gives `s` (an AF_INET/AF_INET6 socket bound beyond the loopback, or about to talk to
    /// another host) its network half, bound to `addr` (port 0: the provider chooses).
    fn attach_net(&mut self, s: &UnixRef, addr: SocketAddr) -> Result<NetRef, Errno> {
        if let Some(n) = s.borrow().net.clone() {
            return Ok(n);
        }
        let (id, proto, v6) = {
            let b = s.borrow();
            let proto = if b.icmp {
                Proto::Icmp
            } else if b.ty == SockType::Dgram {
                Proto::Udp
            } else {
                Proto::Tcp
            };
            (b.id, proto, b.family == AF_INET6)
        };
        let ns = NetSocket::new(id, proto, v6).shared();
        inherit_options(&s.borrow(), &mut ns.borrow_mut());
        let net = self.net.as_mut().ok_or(Errno::ENETUNREACH)?;
        if let Err(e) = net.bind(&ns, unmap(addr)) {
            ns.borrow_mut().closed = true;
            net.kick(&ns);
            return Err(e);
        }
        s.borrow_mut().net = Some(ns.clone());
        Ok(ns)
    }

    /// Binds an AF_INET/AF_INET6 socket. Loopback addresses stay in the kernel; with a provider,
    /// the unspecified address (and an address of our own interfaces) binds both the loopback
    /// name and a network half (provider first, so that it may choose the port), and another
    /// host address binds only the network half.
    fn bind_inet(&mut self, s: &UnixRef, a: SocketAddr) -> Result<(), Errno> {
        if s.borrow().icmp {
            return self.bind_icmp(s, a);
        }
        let (family, stream, v6only) = {
            let b = s.borrow();
            (b.family, b.ty != SockType::Dgram, b.v6only)
        };
        let v6 = family == AF_INET6;
        if self.net.is_none() || is_loopback(a.ip()) {
            if !self.this_host(a.ip()) {
                return Err(Errno::EADDRNOTAVAIL);
            }
            if a.port() == 0 {
                return self.autobind(s, a.ip());
            }
            if !self.port_free(stream, v6, a.port(), a.ip(), v6only) {
                return Err(Errno::EADDRINUSE);
            }
            return self.register_inet(s, a);
        }
        // The unspecified address and our own interface addresses are reachable from this host
        // too (through the kernel) as well as from the network.
        let both = a.ip().is_unspecified() || self.own_addr(a.ip());
        if both && a.port() != 0 && !self.port_free(stream, v6, a.port(), a.ip(), v6only) {
            return Err(Errno::EADDRINUSE);
        }
        let ns = self.attach_net(s, a)?;
        let port = ns.borrow().local.map_or(a.port(), |l| l.port());
        let name = SocketAddr::new(a.ip(), port);
        if both {
            if !self.port_free(stream, v6, port, a.ip(), v6only) {
                s.borrow_mut().net = None;
                self.inet_closed(&ns);
                return Err(Errno::EADDRINUSE);
            }
            self.register_inet(s, name)
        } else {
            s.borrow_mut().name = Some(SockName::Inet(name));
            Ok(())
        }
    }

    /// Registers a loopback name for `s`.
    fn register_inet(&mut self, s: &UnixRef, a: SocketAddr) -> Result<(), Errno> {
        let key = {
            let mut b = s.borrow_mut();
            let key = UnixName::Port(b.ty != SockType::Dgram, b.family == AF_INET6, a.port());
            b.name = Some(SockName::Inet(a));
            b.reg = Some(key.clone());
            key
        };
        self.unix_names.insert(key, Rc::downgrade(s));
        Ok(())
    }

    /// Does loopback traffic without an in-node peer go to the provider (`Config::host_loopback`)?
    fn host_loopback(&self) -> bool {
        self.cfg.host_loopback && self.net.is_some()
    }

    /// A destination the provider can reach: the unspecified address means this host.
    fn concrete(a: SocketAddr, family: u16) -> SocketAddr {
        if a.ip().is_unspecified() {
            SocketAddr::new(Self::loopback(family), a.port())
        } else {
            a
        }
    }

    /// Drops a socket's network half (it now talks only to this host).
    fn detach_net(&mut self, s: &UnixRef) {
        let net = s.borrow_mut().net.take();
        if let Some(net) = net {
            self.inet_closed(&net);
        }
    }

    fn install_sock(&mut self, tid: Tid, s: UnixRef, flags: u64) -> SysResult {
        let file = open_file(
            FileKind::Unix(s),
            open::O_RDWR | (flags & SOCK_NONBLOCK),
            b"socket:",
        );
        Ok(self
            .fds(tid)
            .borrow_mut()
            .insert(file, flags & SOCK_CLOEXEC != 0, 0)? as u64)
    }

    /// Reads a socket address of the socket's `family` from user memory.
    fn read_addr(&self, tid: Tid, addr: u64, len: u64, family: u16) -> Result<SockName, Errno> {
        if !(2..=128).contains(&len) {
            return Err(Errno::EINVAL);
        }
        let raw = read_bytes(self.mm(tid).borrow_mut().mem.as_mut(), addr, len as usize)?;
        let fam = u16::from_le_bytes([raw[0], raw[1]]);
        let port = |r: &[u8]| u16::from_be_bytes([r[2], r[3]]);
        match (fam, family) {
            (AF_UNIX, AF_UNIX) => {
                let path = &raw[2..raw.len().min(2 + SUN_PATH_MAX)];
                match path.first() {
                    None => Err(Errno::EINVAL),
                    Some(0) => Ok(SockName::Unix(path.to_vec())),
                    Some(_) => Ok(SockName::Unix(
                        path[..path.iter().position(|&b| b == 0).unwrap_or(path.len())].to_vec(),
                    )),
                }
            }
            (AF_INET, AF_INET | AF_INET6) if raw.len() >= 8 => {
                let ip = Ipv4Addr::new(raw[4], raw[5], raw[6], raw[7]);
                Ok(SockName::Inet(SocketAddr::V4(SocketAddrV4::new(
                    ip,
                    port(&raw),
                ))))
            }
            (AF_INET6, AF_INET6) if raw.len() >= 24 => {
                let mut o = [0u8; 16];
                o.copy_from_slice(&raw[8..24]);
                Ok(SockName::Inet(SocketAddr::V6(SocketAddrV6::new(
                    Ipv6Addr::from(o),
                    port(&raw),
                    0,
                    0,
                ))))
            }
            _ => Err(Errno::EAFNOSUPPORT),
        }
    }

    /// Writes an address to (`addr`, `*addrlen`), truncating like the kernel.
    pub(super) fn put_sockaddr(
        &self,
        tid: Tid,
        addr: u64,
        addrlen: u64,
        bytes: &[u8],
    ) -> Result<(), Errno> {
        if addr == 0 || addrlen == 0 {
            return Ok(());
        }
        let mm = self.mm(tid);
        let mut mm = mm.borrow_mut();
        let cap = mm.mem.read_u32(addrlen).map_err(|_| Errno::EFAULT)? as usize;
        write_bytes(mm.mem.as_mut(), addr, &bytes[..bytes.len().min(cap)])?;
        write_u32(mm.mem.as_mut(), addrlen, bytes.len() as u32)
    }

    /// The live socket registered under `key`, if any.
    fn registered(&self, key: &UnixName) -> Option<UnixRef> {
        self.unix_names.get(key).and_then(std::rc::Weak::upgrade)
    }

    /// The socket bound to a local inet `dst`, applying Linux's matching rules: a socket bound to
    /// a specific address only receives traffic for that address; one bound to the unspecified
    /// address receives all of its family; a dual-stack (not `IPV6_V6ONLY`) `[::]` socket also
    /// receives IPv4.
    fn lookup_inet(&self, dst: SocketAddr, stream: bool) -> Option<UnixRef> {
        let v4 = match dst.ip() {
            IpAddr::V4(a) => Some(a),
            IpAddr::V6(a) => a.to_ipv4_mapped(),
        };
        let matches = |s: &UnixRef, ip: IpAddr| match &s.borrow().name {
            Some(SockName::Inet(bound)) => bound.ip().is_unspecified() || bound.ip() == ip,
            _ => false,
        };
        if let Some(v4) = v4 {
            if let Some(s) = self.registered(&UnixName::Port(stream, false, dst.port()))
                && matches(&s, IpAddr::V4(v4))
            {
                return Some(s);
            }
            return self
                .registered(&UnixName::Port(stream, true, dst.port()))
                .filter(|s| {
                    let b = s.borrow();
                    !b.v6only
                        && matches!(&b.name, Some(SockName::Inet(a)) if a.ip().is_unspecified())
                });
        }
        self.registered(&UnixName::Port(stream, true, dst.port()))
            .filter(|s| matches(s, dst.ip()))
    }

    fn lookup_sock(&self, tid: Tid, name: &SockName, stream: bool) -> Result<UnixRef, Errno> {
        let key = match name {
            SockName::Inet(a) if self.this_host(a.ip()) => {
                return self.lookup_inet(*a, stream).ok_or(Errno::ECONNREFUSED);
            }
            SockName::Inet(_) => return Err(Errno::ENETUNREACH),
            SockName::Unix(p) if p.first() == Some(&0) => UnixName::Abstract(p.clone()),
            SockName::Unix(p) => {
                let cwd = self.procs[&self.pid_of(tid)].cwd;
                let ino = self.vfs.resolve(cwd, p, true)?;
                if !matches!(self.vfs.inode(ino).node, Node::Socket) {
                    return Err(Errno::ECONNREFUSED);
                }
                UnixName::Ino(ino)
            }
        };
        self.registered(&key).ok_or(Errno::ECONNREFUSED)
    }

    /// Can a socket of family `v6` (with `v6only`) bind `ip:port`? The two families share ports
    /// only through dual-stack wildcard sockets.
    fn port_free(&self, stream: bool, v6: bool, port: u16, ip: IpAddr, v6only: bool) -> bool {
        if self.registered(&UnixName::Port(stream, v6, port)).is_some() {
            return false;
        }
        match self.registered(&UnixName::Port(stream, !v6, port)) {
            None => true,
            Some(other) => {
                let o = other.borrow();
                let other_wild =
                    matches!(&o.name, Some(SockName::Inet(a)) if a.ip().is_unspecified());
                if v6 {
                    // A dual-stack wildcard v6 socket collides with any v4 user of the port.
                    v6only || !ip.is_unspecified()
                } else {
                    // A v4 socket collides with a dual-stack wildcard v6 socket.
                    o.v6only || !other_wild
                }
            }
        }
    }

    /// Gives an unbound inet socket an ephemeral port on `ip`.
    fn autobind(&mut self, s: &UnixRef, ip: IpAddr) -> Result<(), Errno> {
        if s.borrow().name.is_some() {
            return Ok(());
        }
        let (stream, v6, v6only) = {
            let b = s.borrow();
            (b.ty != SockType::Dgram, b.family == AF_INET6, b.v6only)
        };
        let mut port = self.next_ephemeral.max(EPHEMERAL_START);
        for _ in 0..(u16::MAX - EPHEMERAL_START) {
            if self.port_free(stream, v6, port, ip, v6only) {
                self.next_ephemeral = if port == u16::MAX {
                    EPHEMERAL_START
                } else {
                    port + 1
                };
                let key = UnixName::Port(stream, v6, port);
                let mut b = s.borrow_mut();
                b.name = Some(SockName::Inet(SocketAddr::new(ip, port)));
                b.reg = Some(key.clone());
                drop(b);
                self.unix_names.insert(key, Rc::downgrade(s));
                return Ok(());
            }
            port = if port == u16::MAX {
                EPHEMERAL_START
            } else {
                port + 1
            };
        }
        Err(Errno::EADDRINUSE)
    }

    fn loopback(family: u16) -> IpAddr {
        if family == AF_INET6 {
            IpAddr::V6(Ipv6Addr::LOCALHOST)
        } else {
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        }
    }

    pub(crate) fn sys_socket(&mut self, tid: Tid, domain: u64, ty: u64, proto: u64) -> SysResult {
        let family = u16::try_from(domain).map_err(|_| Errno::EAFNOSUPPORT)?;
        if family == AF_NETLINK {
            return self.netlink_socket(tid, ty, proto);
        }
        if matches!(family, AF_INET | AF_INET6) && ty & SOCK_TYPE_MASK == SOCK_RAW {
            // Raw sockets need CAP_NET_RAW, which no guest has here. Like an unprivileged Linux
            // process: EPERM, so ping falls back to an
            // ICMP datagram socket.
            return Err(Errno::EPERM);
        }
        let kind = sock_type(ty)?;
        let mut icmp = false;
        match family {
            AF_UNIX => {}
            AF_INET | AF_INET6 => {
                if kind == SockType::SeqPacket {
                    return Err(Errno::ESOCKTNOSUPPORT);
                }
                // Unprivileged ICMP echo ("ping") sockets (net/ipv4/ping.c).
                let icmp_proto = if family == AF_INET6 {
                    IPPROTO_ICMPV6
                } else {
                    IPPROTO_ICMP
                };
                icmp = kind == SockType::Dgram && proto == icmp_proto;
                if !icmp && !matches!(proto, 0 | 6 | 17) {
                    return Err(Errno::EPROTONOSUPPORT);
                }
            }
            _ => return Err(Errno::EAFNOSUPPORT),
        }
        let id = self.new_object_id();
        let mut sock = UnixSocket::new(id, family, kind, self.pid_of(tid));
        sock.icmp = icmp;
        self.install_sock(tid, Rc::new(RefCell::new(sock)), ty)
    }

    pub(crate) fn sys_socketpair(
        &mut self,
        tid: Tid,
        domain: u64,
        ty: u64,
        _proto: u64,
        sv: u64,
    ) -> SysResult {
        if domain != u64::from(AF_UNIX) {
            return Err(Errno::EOPNOTSUPP);
        }
        let kind = sock_type(ty)?;
        let pid = self.pid_of(tid);
        let (ia, ib) = (self.new_object_id(), self.new_object_id());
        let a = Rc::new(RefCell::new(UnixSocket::new(ia, AF_UNIX, kind, pid)));
        let b = Rc::new(RefCell::new(UnixSocket::new(ib, AF_UNIX, kind, pid)));
        Self::unix_pair(&a, &b);
        let fa = self.install_sock(tid, a, ty)? as i32;
        let fb = self.install_sock(tid, b, ty)? as i32;
        let mut out = [0u8; 8];
        out[..4].copy_from_slice(&fa.to_le_bytes());
        out[4..].copy_from_slice(&fb.to_le_bytes());
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), sv, &out)?;
        Ok(0)
    }

    pub(crate) fn sys_bind(
        &mut self,
        tid: Tid,
        fd: u64,
        addr: u64,
        len: u64,
        host: &mut dyn Host,
    ) -> SysResult {
        if let Some((n, _)) = self.netlink_of(tid, fd)? {
            return self.netlink_bind(tid, &n, addr, len);
        }
        let (s, _) = self.sock_of(tid, fd)?;
        if s.borrow().name.is_some() {
            return Err(Errno::EINVAL);
        }
        let family = s.borrow().family;
        let name = self.read_addr(tid, addr, len, family)?;
        let (name, key) = match name {
            SockName::Unix(p) if p.first() == Some(&0) => {
                let key = UnixName::Abstract(p.clone());
                if self
                    .unix_names
                    .get(&key)
                    .is_some_and(|w| w.strong_count() > 0)
                {
                    return Err(Errno::EADDRINUSE);
                }
                (SockName::Unix(p), key)
            }
            SockName::Unix(p) => {
                let cwd = self.procs[&self.pid_of(tid)].cwd;
                let (dir, base) = self.vfs.resolve_parent(cwd, &p)?;
                if self.vfs.lookup(dir, &base).is_ok() {
                    return Err(Errno::EADDRINUSE);
                }
                let now = Timespec::from_nanos(host.realtime_ns());
                let umask = self.procs[&self.pid_of(tid)].umask;
                let ino = self.vfs.mksock(dir, &base, 0o777 & !umask, now)?;
                (SockName::Unix(p), UnixName::Ino(ino))
            }
            SockName::Inet(a) => {
                self.bind_inet(&s, a)?;
                return Ok(0);
            }
        };
        {
            let mut b = s.borrow_mut();
            b.name = Some(name);
            b.reg = Some(key.clone());
        }
        self.unix_names.insert(key, Rc::downgrade(&s));
        Ok(0)
    }

    pub(crate) fn sys_listen(&mut self, tid: Tid, fd: u64, backlog: u64) -> SysResult {
        let (s, _) = self.sock_of(tid, fd)?;
        let (family, ty) = {
            let b = s.borrow();
            (b.family, b.ty)
        };
        if ty == SockType::Dgram {
            return Err(Errno::EOPNOTSUPP);
        }
        if family != AF_UNIX && s.borrow().name.is_none() {
            self.bind_inet(&s, SocketAddr::new(wildcard(family), 0))?;
        }
        let backlog = (backlog as i32).clamp(1, 4096) as usize;
        let net = {
            let mut b = s.borrow_mut();
            if b.name.is_none() || matches!(b.state, UnixState::Connected) {
                return Err(Errno::EINVAL);
            }
            if let UnixState::Listening {
                backlog: current, ..
            } = &mut b.state
            {
                *current = backlog;
            } else {
                b.state = UnixState::Listening {
                    backlog,
                    queue: Default::default(),
                };
            }
            b.net.clone()
        };
        if let Some(ns) = net {
            let fresh = {
                let mut n = ns.borrow_mut();
                n.backlog = backlog;
                n.state != NetState::Listening
            };
            if fresh {
                if let Some(provider) = self.net.as_mut() {
                    provider.listen(&ns)?;
                }
            } else {
                self.net_kick(&ns);
            }
        }
        Ok(0)
    }

    pub(crate) fn sys_accept4(
        &mut self,
        tid: Tid,
        fd: u64,
        addr: u64,
        addrlen: u64,
        flags: u64,
    ) -> SysResult {
        if flags & !(SOCK_NONBLOCK | SOCK_CLOEXEC) != 0 {
            return Err(Errno::EINVAL);
        }
        let (s, nonblock) = self.sock_of(tid, fd)?;
        let conn = {
            let mut b = s.borrow_mut();
            let UnixState::Listening { queue, .. } = &mut b.state else {
                return Err(Errno::EINVAL);
            };
            match queue.pop_front() {
                Some(c) => c,
                None => {
                    // A connection from another host?
                    let remote = b
                        .net
                        .as_ref()
                        .and_then(|n| n.borrow_mut().accepted.pop_front());
                    if let Some(c) = remote {
                        let (v6, listener) = (b.family == AF_INET6, b.net.clone());
                        drop(b);
                        c.borrow_mut().v6 = v6;
                        if let Some(l) = listener {
                            l.borrow_mut().touch();
                            self.net_kick(&l); // room in the backlog
                        }
                        let peer = c.borrow().remote.map(SockName::Inet);
                        let family = if v6 { AF_INET6 } else { AF_INET };
                        self.put_sockaddr(tid, addr, addrlen, &encode(peer.as_ref(), family))?;
                        return self.install_net(tid, c, flags);
                    }
                    if nonblock {
                        return Err(Errno::EAGAIN);
                    }
                    let (id, timeout) = (b.id, b.rcvtimeo);
                    drop(b);
                    return Err(self.wait_sock(id, timeout));
                }
            }
        };
        touch(self, &s);
        let family = s.borrow().family;
        let peer = conn.borrow().peer_name.clone();
        self.put_sockaddr(tid, addr, addrlen, &encode(peer.as_ref(), family))?;
        self.install_sock(tid, conn, flags)
    }

    pub(crate) fn sys_connect(&mut self, tid: Tid, fd: u64, addr: u64, len: u64) -> SysResult {
        if self.netlink_of(tid, fd)?.is_some() {
            // Only the kernel answers; a default destination changes nothing.
            return Ok(0);
        }
        let (s, nonblock) = match self.any_sock(tid, fd)? {
            (Sock::Local(s), nb) => (s, nb),
            (Sock::Net(ns), nb) => return self.connect_again(&ns, nb),
        };
        let (family, ty) = {
            let b = s.borrow();
            (b.family, b.ty)
        };
        let name = self.read_addr(tid, addr, len, family)?;
        if s.borrow().icmp {
            return self.connect_icmp(&s, name).map(|()| 0);
        }
        if let SockName::Inet(a) = &name
            && !self.this_host(a.ip())
        {
            return self.connect_remote(tid, fd, &s, *a, nonblock);
        }
        let target = match self.lookup_sock(tid, &name, ty != SockType::Dgram) {
            Ok(t) => t,
            Err(Errno::ECONNREFUSED) if self.host_loopback() => {
                if let SockName::Inet(a) = name {
                    return self.connect_remote(tid, fd, &s, Self::concrete(a, family), nonblock);
                }
                return Err(Errno::ECONNREFUSED);
            }
            Err(e) => return Err(e),
        };
        if family != AF_UNIX {
            // It talks to this host from now on.
            self.detach_net(&s);
        }
        if let SockName::Inet(a) = &name {
            let local = if a.ip().is_unspecified() {
                Self::loopback(family)
            } else {
                a.ip()
            };
            self.autobind(&s, local)?;
        }
        if ty == SockType::Dgram {
            // A default destination.
            s.borrow_mut().peer = Rc::downgrade(&target);
            return Ok(0);
        }
        if matches!(s.borrow().state, UnixState::Connected) {
            return Err(Errno::EISCONN);
        }
        {
            let t = target.borrow();
            if t.ty != ty {
                return Err(Errno::EPROTOTYPE);
            }
            let UnixState::Listening { backlog, queue } = &t.state else {
                return Err(Errno::ECONNREFUSED);
            };
            if queue.len() >= *backlog {
                if nonblock {
                    return Err(Errno::EAGAIN);
                }
                let id = t.id;
                drop(t);
                return Err(self.wait_for(WaitKey::Object(id), None, None));
            }
        }
        // The server side of the connection, waiting in the listener's queue. Its local name
        // is the address the client actually reached.
        let id = self.new_object_id();
        let (owner, tfamily, tname) = {
            let t = target.borrow();
            (t.owner, t.family, t.name.clone())
        };
        let server = Rc::new(RefCell::new(UnixSocket::new(id, tfamily, ty, owner)));
        server.borrow_mut().name = match (&tname, &name) {
            (Some(SockName::Inet(l)), SockName::Inet(dst)) => {
                let ip = if dst.ip().is_unspecified() {
                    Self::loopback(tfamily)
                } else {
                    dst.ip()
                };
                Some(SockName::Inet(SocketAddr::new(ip, l.port())))
            }
            _ => tname,
        };
        Self::unix_pair(&s, &server);
        if let UnixState::Listening { queue, .. } = &mut target.borrow_mut().state {
            queue.push_back(server);
        }
        touch(self, &target);
        Ok(0)
    }

    /// `connect` to another host, through the provider. A TCP socket becomes a provider
    /// connection (its description switches to `FileKind::Inet`, so every descriptor and epoll
    /// interest follows); a UDP socket gets a remote default peer on its network half.
    fn connect_remote(
        &mut self,
        tid: Tid,
        fd: u64,
        s: &UnixRef,
        a: SocketAddr,
        nonblock: bool,
    ) -> SysResult {
        if self.net.is_none() {
            return Err(Errno::ENETUNREACH);
        }
        let dst = unmap(a);
        let (family, ty, bound) = {
            let b = s.borrow();
            let bound = match &b.name {
                Some(SockName::Inet(l)) => Some(*l),
                _ => None,
            };
            (b.family, b.ty, bound)
        };
        if bound.is_some_and(|l| is_loopback(l.ip())) && !is_loopback(dst.ip()) {
            // A socket bound to the loopback cannot reach other hosts.
            return Err(Errno::EINVAL);
        }
        let any = SocketAddr::new(wildcard(family), 0);
        if ty == SockType::Dgram {
            let ns = self.attach_net(s, bound.unwrap_or(any))?;
            if let Some(provider) = self.net.as_mut() {
                provider.connect(&ns, dst)?;
            }
            let local = ns.borrow().local;
            let mut b = s.borrow_mut();
            b.peer = std::rc::Weak::new();
            if b.name.is_none() {
                b.name = local.map(SockName::Inet);
            }
            return Ok(0);
        }
        if matches!(s.borrow().state, UnixState::Connected) {
            return Err(Errno::EISCONN);
        }
        if matches!(s.borrow().state, UnixState::Listening { .. }) {
            return Err(Errno::EINVAL);
        }
        let half = s.borrow_mut().net.take();
        let ns = match half {
            Some(ns) => ns,
            None => {
                let b = s.borrow();
                let ns = NetSocket::new(b.id, Proto::Tcp, family == AF_INET6).shared();
                inherit_options(&b, &mut ns.borrow_mut());
                ns
            }
        };
        // No longer a loopback name: it is a connection to another host now.
        let reg = s.borrow_mut().reg.take();
        if let Some(key) = reg {
            self.unix_names.remove(&key);
        }
        let started = match self.net.as_mut() {
            Some(provider) => provider.connect(&ns, dst),
            None => Err(Errno::ENETUNREACH),
        };
        if let Err(e) = started {
            self.inet_closed(&ns);
            return Err(e);
        }
        self.file(tid, fd)?.borrow_mut().kind = FileKind::Inet(ns.clone());
        if nonblock {
            return Err(Errno::EINPROGRESS);
        }
        let id = {
            let mut n = ns.borrow_mut();
            n.connect_waiter = true;
            n.id
        };
        Err(self.wait_for(WaitKey::Object(id), None, None))
    }

    /// `connect` on a provider connection: a blocking `connect` restarting, or a repeated one.
    fn connect_again(&mut self, ns: &NetRef, nonblock: bool) -> SysResult {
        let mut n = ns.borrow_mut();
        match n.state {
            NetState::Connecting if n.connect_waiter && !nonblock => {
                let id = n.id;
                drop(n);
                Err(self.wait_for(WaitKey::Object(id), None, None))
            }
            NetState::Connecting => Err(Errno::EALREADY),
            NetState::Connected if n.connect_waiter => {
                n.connect_waiter = false;
                Ok(0)
            }
            NetState::Failed => {
                n.connect_waiter = false;
                Err(n.error.take().unwrap_or(Errno::ECONNREFUSED))
            }
            _ => Err(Errno::EISCONN),
        }
    }

    pub(crate) fn sys_getsockname(
        &mut self,
        tid: Tid,
        fd: u64,
        addr: u64,
        addrlen: u64,
        peer: bool,
    ) -> SysResult {
        if let Some((n, _)) = self.netlink_of(tid, fd)? {
            return self.netlink_name(tid, &n, addr, addrlen, peer);
        }
        let s = match self.any_sock(tid, fd)? {
            (Sock::Local(s), _) => s,
            (Sock::Net(ns), _) => {
                let n = ns.borrow();
                let family = if n.v6 { AF_INET6 } else { AF_INET };
                let name = if peer {
                    if n.state != NetState::Connected {
                        return Err(Errno::ENOTCONN);
                    }
                    n.remote
                } else {
                    n.local
                };
                drop(n);
                let bytes = encode(name.map(SockName::Inet).as_ref(), family);
                self.put_sockaddr(tid, addr, addrlen, &bytes)?;
                return Ok(0);
            }
        };
        let family = s.borrow().family;
        let name = if peer {
            let b = s.borrow();
            let remote_peer = b.net.as_ref().and_then(|n| {
                let n = n.borrow();
                (n.state == NetState::Connected)
                    .then_some(n.remote)
                    .flatten()
            });
            if let Some(r) = remote_peer {
                Some(SockName::Inet(r))
            } else if !matches!(b.state, UnixState::Connected) && b.peer.upgrade().is_none() {
                return Err(Errno::ENOTCONN);
            } else {
                b.peer
                    .upgrade()
                    .and_then(|p| p.borrow().name.clone())
                    .or_else(|| b.peer_name.clone())
            }
        } else {
            s.borrow().name.clone()
        };
        self.put_sockaddr(tid, addr, addrlen, &encode(name.as_ref(), family))?;
        Ok(0)
    }

    /// Common send path (`sendto`, `sendmsg`).
    fn sock_send(
        &mut self,
        tid: Tid,
        fd: u64,
        data: &[u8],
        flags: u64,
        dest: Option<SockName>,
        fds: Vec<FileRef>,
    ) -> SysResult {
        let (sock, nb) = self.any_sock(tid, fd)?;
        let nonblock = nb || flags & MSG_DONTWAIT != 0;
        let result = match sock {
            Sock::Net(ns) => {
                self.release_files(fds);
                self.inet_send(&ns, data, nonblock)
            }
            Sock::Local(s) => self.local_send(tid, &s, data, nonblock, dest, fds),
        };
        match result {
            Err(Errno::EPIPE) => {
                if flags & MSG_NOSIGNAL == 0 {
                    let _ = self.send_to_thread(tid, signal::SIGPIPE);
                }
                Err(Errno::EPIPE)
            }
            r => r.map(|n| n as u64),
        }
    }

    /// Sends on an in-kernel socket (a UDP datagram for another host goes to its network half).
    fn local_send(
        &mut self,
        tid: Tid,
        s: &UnixRef,
        data: &[u8],
        nonblock: bool,
        dest: Option<SockName>,
        fds: Vec<FileRef>,
    ) -> Result<usize, Errno> {
        // Unless delivered in-kernel, descriptions in flight go nowhere.
        match self.local_route(tid, s, data, nonblock, dest) {
            Ok(Route::Socket(to)) => self.unix_send_with(s, data, to, nonblock, fds),
            Ok(Route::Done(n)) => {
                self.release_files(fds);
                Ok(n)
            }
            Err(e) => {
                self.release_files(fds);
                Err(e)
            }
        }
    }

    /// Where [`Kernel::local_send`] delivers.
    fn local_route(
        &mut self,
        tid: Tid,
        s: &UnixRef,
        data: &[u8],
        nonblock: bool,
        dest: Option<SockName>,
    ) -> Result<Route, Errno> {
        let (family, ty, icmp) = {
            let b = s.borrow();
            (b.family, b.ty, b.icmp)
        };
        if icmp {
            return self.icmp_send(s, data, dest, nonblock).map(Route::Done);
        }
        let to = match dest {
            Some(SockName::Inet(a)) if ty == SockType::Dgram && !self.this_host(a.ip()) => {
                return self
                    .udp_send_remote(s, data, Some(unmap(a)), nonblock)
                    .map(Route::Done);
            }
            Some(name) if ty == SockType::Dgram => {
                if let SockName::Inet(a) = &name {
                    let local = if a.ip().is_unspecified() {
                        Self::loopback(family)
                    } else {
                        a.ip()
                    };
                    self.autobind(s, local)?;
                }
                match self.lookup_sock(tid, &name, false) {
                    Ok(t) => Some(t),
                    Err(Errno::ECONNREFUSED) if family != AF_UNIX && self.host_loopback() => {
                        if let SockName::Inet(a) = name {
                            let to = unmap(Self::concrete(a, family));
                            return self
                                .udp_send_remote(s, data, Some(to), nonblock)
                                .map(Route::Done);
                        }
                        return Ok(Route::Done(data.len()));
                    }
                    // UDP to a closed local port: the datagram is lost.
                    Err(Errno::ECONNREFUSED) if family != AF_UNIX => {
                        return Ok(Route::Done(data.len()));
                    }
                    Err(e) => return Err(e),
                }
            }
            Some(_) if family == AF_UNIX => return Err(Errno::EISCONN),
            None if ty == SockType::Dgram && Self::remote_peer(s).is_some() => {
                return self
                    .udp_send_remote(s, data, None, nonblock)
                    .map(Route::Done);
            }
            _ => None,
        };
        Ok(Route::Socket(to))
    }

    /// The remote default peer of a UDP socket connected to another host.
    fn remote_peer(s: &UnixRef) -> Option<SocketAddr> {
        let b = s.borrow();
        let n = b.net.as_ref()?.borrow();
        (n.state == NetState::Connected)
            .then_some(n.remote)
            .flatten()
    }

    /// Queues a datagram for another host on `s`'s network half (attached on first use).
    fn udp_send_remote(
        &mut self,
        s: &UnixRef,
        data: &[u8],
        to: Option<SocketAddr>,
        nonblock: bool,
    ) -> Result<usize, Errno> {
        /// The largest UDP payload over IPv4.
        const MAX_DATAGRAM: usize = 65_507;
        if data.len() > MAX_DATAGRAM {
            return Err(Errno::EMSGSIZE);
        }
        let (family, bound) = {
            let b = s.borrow();
            let bound = match &b.name {
                Some(SockName::Inet(l)) => Some(*l),
                _ => None,
            };
            (b.family, bound)
        };
        if bound.is_some_and(|l| is_loopback(l.ip())) && to.is_none_or(|t| !is_loopback(t.ip())) {
            return Err(Errno::EINVAL);
        }
        if self.net.is_none() {
            return Err(Errno::ENETUNREACH);
        }
        let ns = self.attach_net(s, bound.unwrap_or(SocketAddr::new(wildcard(family), 0)))?;
        if bound.is_none() {
            // Implicitly bound now: the provider chose the port.
            let local = ns.borrow().local;
            s.borrow_mut().name = local.map(SockName::Inet);
        }
        {
            let mut n = ns.borrow_mut();
            if n.dgrams_out.len() >= crate::net::DGRAM_QUEUE {
                if nonblock {
                    return Err(Errno::EAGAIN);
                }
                let (id, timeout) = (n.id, n.sndtimeo);
                drop(n);
                return Err(self.wait_sock(id, timeout));
            }
            n.dgrams_out.push_back((data.to_vec(), to));
            n.touch();
        }
        self.net_kick(&ns);
        Ok(data.len())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sys_sendto(
        &mut self,
        tid: Tid,
        fd: u64,
        buf: u64,
        len: u64,
        flags: u64,
        dest: u64,
        destlen: u64,
    ) -> SysResult {
        if let Some((n, _)) = self.netlink_of(tid, fd)? {
            let data = read_bytes(
                self.mm(tid).borrow_mut().mem.as_mut(),
                buf,
                crate::uaccess::io_len(len),
            )?;
            return self.netlink_send(tid, &n, &data).map(|n| n as u64);
        }
        let family = self.sock_family(tid, fd)?;
        let data = read_bytes(
            self.mm(tid).borrow_mut().mem.as_mut(),
            buf,
            crate::uaccess::io_len(len),
        )?;
        let dest = if dest == 0 {
            None
        } else {
            Some(self.read_addr(tid, dest, destlen, family)?)
        };
        self.sock_send(tid, fd, &data, flags, dest, Vec::new())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sys_recvfrom(
        &mut self,
        tid: Tid,
        fd: u64,
        buf: u64,
        len: u64,
        flags: u64,
        src: u64,
        srclen: u64,
    ) -> SysResult {
        let mut data = vec![0u8; crate::uaccess::io_len(len)];
        let mut fds = Vec::new();
        let r = self.sock_recv(tid, fd, &mut data, flags, &mut fds);
        // No control buffer: descriptions in flight are discarded, as on Linux.
        self.release_files(fds);
        let (n, from, full, family) = r?;
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), buf, &data[..n])?;
        if src != 0 {
            self.put_sockaddr(tid, src, srclen, &encode(from.as_ref(), family))?;
        }
        Ok(received(n, full, flags))
    }

    /// The address family of a socket descriptor.
    fn sock_family(&self, tid: Tid, fd: u64) -> Result<u16, Errno> {
        Ok(match self.any_sock(tid, fd)? {
            (Sock::Local(s), _) => s.borrow().family,
            (Sock::Net(n), _) => {
                if n.borrow().v6 {
                    AF_INET6
                } else {
                    AF_INET
                }
            }
        })
    }

    /// Common receive path (`recvfrom`, `recvmsg`): (bytes, source, the message's whole length,
    /// family). A datagram longer than `buf` was truncated to it.
    fn sock_recv(
        &mut self,
        tid: Tid,
        fd: u64,
        buf: &mut [u8],
        flags: u64,
        fds: &mut Vec<FileRef>,
    ) -> Result<(usize, Option<SockName>, usize, u16), Errno> {
        if let Some((n, nb)) = self.netlink_of(tid, fd)? {
            let nonblock = nb || flags & MSG_DONTWAIT != 0;
            let (len, full) = self.netlink_recv(&n, buf, flags & MSG_PEEK != 0, nonblock)?;
            return Ok((len, None, full, super::netlink::AF_NETLINK));
        }
        let (sock, nb) = self.any_sock(tid, fd)?;
        let nonblock = nb || flags & MSG_DONTWAIT != 0;
        let peek = flags & MSG_PEEK != 0;
        match sock {
            Sock::Net(ns) => {
                let n = self.inet_recv(&ns, buf, peek, nonblock)?;
                let b = ns.borrow();
                let family = if b.v6 { AF_INET6 } else { AF_INET };
                Ok((n, b.remote.map(SockName::Inet), n, family))
            }
            Sock::Local(s) => {
                let family = s.borrow().family;
                let (n, from, full) = self.unix_recv_with(&s, buf, peek, nonblock, fds)?;
                let from = from.or_else(|| {
                    s.borrow()
                        .peer
                        .upgrade()
                        .and_then(|p| p.borrow().name.clone())
                });
                Ok((n, from, full, family))
            }
        }
    }

    fn msghdr(&self, tid: Tid, msg: u64) -> Result<[u8; MSGHDR_SIZE], Errno> {
        read_array(self.mm(tid).borrow_mut().mem.as_mut(), msg)
    }

    fn msg_iovs(&self, tid: Tid, h: &[u8; MSGHDR_SIZE]) -> Result<Vec<Iovec>, Errno> {
        let get = |at: usize| u64::from_le_bytes(h[at..at + 8].try_into().unwrap_or_default());
        let (iov, cnt) = (get(16), get(24));
        if cnt > 1024 {
            return Err(Errno::EMSGSIZE);
        }
        let mm = self.mm(tid);
        let mut mm = mm.borrow_mut();
        (0..cnt)
            .map(|i| {
                Ok(Iovec::from_bytes(&read_array::<16>(
                    mm.mem.as_mut(),
                    iov + 16 * i,
                )?))
            })
            .collect()
    }

    pub(crate) fn sys_sendmsg(&mut self, tid: Tid, fd: u64, msg: u64, flags: u64) -> SysResult {
        let netlink = self.netlink_of(tid, fd)?;
        let family = match netlink {
            Some(_) => super::netlink::AF_NETLINK,
            None => self.sock_family(tid, fd)?,
        };
        let h = self.msghdr(tid, msg)?;
        let get = |at: usize| u64::from_le_bytes(h[at..at + 8].try_into().unwrap_or_default());
        let (name, namelen) = (
            get(0),
            u64::from(u32::from_le_bytes(h[8..12].try_into().unwrap_or_default())),
        );
        // Gathered whole like `writev`, and capped like it: at most `MAX_IO` bytes in all.
        let mut data = Vec::new();
        for v in self.msg_iovs(tid, &h)? {
            let room = crate::uaccess::MAX_IO.saturating_sub(data.len());
            data.extend(read_bytes(
                self.mm(tid).borrow_mut().mem.as_mut(),
                v.base,
                crate::uaccess::io_len(v.len).min(room),
            )?);
        }
        let fds = if get(40) == 0 {
            Vec::new()
        } else if netlink.is_none() && family == AF_UNIX {
            self.scm_rights_in(tid, get(32), get(40))?
        } else {
            // Ancillary data on other families (IP_PKTINFO, ...) is not supported.
            return Err(Errno::EOPNOTSUPP);
        };
        if let Some((n, _)) = netlink {
            // The destination can only be the kernel.
            let _ = (name, namelen);
            return self.netlink_send(tid, &n, &data).map(|n| n as u64);
        }
        let dest = if name == 0 {
            None
        } else {
            match self.read_addr(tid, name, namelen, family) {
                Ok(a) => Some(a),
                Err(e) => {
                    self.release_files(fds);
                    return Err(e);
                }
            }
        };
        self.sock_send(tid, fd, &data, flags, dest, fds)
    }

    /// The descriptions a `sendmsg` passes (`SCM_RIGHTS`), from its control buffer. Credentials
    /// (`SCM_CREDENTIALS`) are accepted and ignored: the receiver learns the sender's through
    /// `SO_PEERCRED` anyway.
    fn scm_rights_in(&mut self, tid: Tid, control: u64, len: u64) -> Result<Vec<FileRef>, Errno> {
        if len > 64 * 1024 {
            return Err(Errno::ENOBUFS);
        }
        let buf = read_bytes(
            self.mm(tid).borrow_mut().mem.as_mut(),
            control,
            len as usize,
        )?;
        let mut fds = Vec::new();
        let mut at = 0;
        let mut bad = None;
        while at + CMSG_HDR <= buf.len() {
            let word = |i: usize| u32::from_le_bytes(buf[i..i + 4].try_into().unwrap_or_default());
            let clen = u64::from_le_bytes(buf[at..at + 8].try_into().unwrap_or_default());
            let (level, ty) = (u64::from(word(at + 8)), word(at + 12));
            let Some(end) = usize::try_from(clen)
                .ok()
                .filter(|&c| c >= CMSG_HDR)
                .and_then(|c| at.checked_add(c))
                .filter(|&e| e <= buf.len())
            else {
                bad = Some(Errno::EINVAL);
                break;
            };
            match (level, ty) {
                (SOL_SOCKET, SCM_RIGHTS) => {
                    for raw in buf[at + CMSG_HDR..end].as_chunks::<4>().0 {
                        if fds.len() >= SCM_MAX_FD {
                            bad = Some(Errno::EINVAL);
                            break;
                        }
                        let fd = i32::from_le_bytes(*raw);
                        match self.fds(tid).borrow().get(fd) {
                            Ok(f) => fds.push(f),
                            Err(e) => {
                                bad = Some(e);
                                break;
                            }
                        }
                    }
                }
                (SOL_SOCKET, SCM_CREDENTIALS) => {}
                _ => bad = Some(Errno::EINVAL),
            }
            if bad.is_some() {
                break;
            }
            at += cmsg_align(end - at);
        }
        match bad {
            Some(e) => {
                self.release_files(fds);
                Err(e)
            }
            None => Ok(fds),
        }
    }

    /// Delivers received descriptions (`SCM_RIGHTS`) into the receiver's descriptor table and
    /// writes their control message to `msg`'s control buffer, setting `msg_controllen` to the
    /// space used. Returns `MSG_CTRUNC` if some did not fit (those are closed, as on Linux).
    fn scm_rights_out(
        &mut self,
        tid: Tid,
        msg: u64,
        h: &[u8; MSGHDR_SIZE],
        mut fds: Vec<FileRef>,
        flags: u64,
    ) -> Result<u32, Errno> {
        let get = |at: usize| u64::from_le_bytes(h[at..at + 8].try_into().unwrap_or_default());
        let (control, cap) = (get(32), get(40) as usize);
        let room = if control == 0 {
            0
        } else {
            cap.saturating_sub(CMSG_HDR) / 4
        };
        let truncated = fds.len() > room;
        let dropped = fds.split_off(fds.len().min(room));
        self.release_files(dropped);
        let cloexec = flags & MSG_CMSG_CLOEXEC != 0;
        let mut installed = Vec::with_capacity(fds.len());
        for f in fds {
            let r = self.fds(tid).borrow_mut().insert(f.clone(), cloexec, 0);
            match r {
                Ok(fd) => installed.push(fd),
                Err(_) => self.release_file(&f),
            }
        }
        let mut out = Vec::new();
        if !installed.is_empty() {
            let clen = CMSG_HDR + 4 * installed.len();
            out.extend_from_slice(&(clen as u64).to_le_bytes());
            out.extend_from_slice(&(SOL_SOCKET as u32).to_le_bytes());
            out.extend_from_slice(&SCM_RIGHTS.to_le_bytes());
            for fd in &installed {
                out.extend_from_slice(&fd.to_le_bytes());
            }
            // The aligned space used, but never more than the buffer holds.
            out.resize(cmsg_align(clen).min(cap), 0);
        }
        let mm = self.mm(tid);
        let mut m = mm.borrow_mut();
        if !out.is_empty() {
            write_bytes(m.mem.as_mut(), control, &out)?;
        }
        crate::uaccess::write_u64(m.mem.as_mut(), msg + 40, out.len() as u64)?;
        Ok(if truncated { MSG_CTRUNC } else { 0 })
    }

    /// `IP_RECVTTL` / `IPV6_RECVHOPLIMIT`: after a datagram whose
    /// TTL / hop limit is known, an `IP_TTL` / `IPV6_HOPLIMIT` control message (an `int`) goes to
    /// `msg`'s control buffer, which held nothing else (only AF_UNIX sockets pass descriptions).
    /// Returns `MSG_CTRUNC` if it did not fit.
    fn hop_limit_out(
        &mut self,
        tid: Tid,
        fd: u64,
        msg: u64,
        h: &[u8; MSGHDR_SIZE],
    ) -> Result<u32, Errno> {
        /// `IP_RECVTTL` and the `IP_TTL` control message; `IPV6_RECVHOPLIMIT`, `IPV6_HOPLIMIT`.
        const IP_RECVTTL: u64 = 12;
        const IPV6_RECVHOPLIMIT: u64 = 51;
        const IPV6_HOPLIMIT: u32 = 52;
        const IPV6_2292HOPLIMIT: u32 = 8;
        let Ok((Sock::Local(s), _)) = self.any_sock(tid, fd) else {
            return Ok(0);
        };
        let (level, ty, hops) = {
            let b = s.borrow();
            let Some(hops) = b.last_hops else {
                return Ok(0);
            };
            let asked = |level, opt| b.opts.get(&(level, opt)).is_some_and(|&v| v != 0);
            match b.family {
                AF_INET if asked(IPPROTO_IP, IP_RECVTTL) => (IPPROTO_IP, IP_TTL as u32, hops),
                AF_INET6 if asked(IPPROTO_IPV6, IPV6_RECVHOPLIMIT) => {
                    (IPPROTO_IPV6, IPV6_HOPLIMIT, hops)
                }
                // RFC 2292's older API, which busybox `ping -6` uses: the control message has
                // the option's own number.
                AF_INET6 if asked(IPPROTO_IPV6, u64::from(IPV6_2292HOPLIMIT)) => {
                    (IPPROTO_IPV6, IPV6_2292HOPLIMIT, hops)
                }
                _ => return Ok(0),
            }
        };
        let get = |at: usize| u64::from_le_bytes(h[at..at + 8].try_into().unwrap_or_default());
        let (control, cap) = (get(32), get(40) as usize);
        let clen = CMSG_HDR + 4;
        if control == 0 || cap < clen {
            return Ok(MSG_CTRUNC);
        }
        let mut out = Vec::with_capacity(cmsg_align(clen));
        out.extend_from_slice(&(clen as u64).to_le_bytes());
        out.extend_from_slice(&(level as u32).to_le_bytes());
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&i32::from(hops).to_le_bytes());
        out.resize(cmsg_align(clen).min(cap), 0);
        let mm = self.mm(tid);
        let mut m = mm.borrow_mut();
        write_bytes(m.mem.as_mut(), control, &out)?;
        crate::uaccess::write_u64(m.mem.as_mut(), msg + 40, out.len() as u64)?;
        Ok(0)
    }

    pub(crate) fn sys_recvmsg(&mut self, tid: Tid, fd: u64, msg: u64, flags: u64) -> SysResult {
        let h = self.msghdr(tid, msg)?;
        let iovs = self.msg_iovs(tid, &h)?;
        // The guest's lengths may add up past 2^64: the sum saturates (it is capped anyway).
        let total = iovs.iter().fold(0u64, |t, v| t.saturating_add(v.len));
        let mut data = vec![0u8; crate::uaccess::io_len(total)];
        let mut fds = Vec::new();
        let (n, from, full, family) = self.sock_recv(tid, fd, &mut data, flags, &mut fds)?;
        let trunc = full > n;
        let mut ctrunc = self.scm_rights_out(tid, msg, &h, fds, flags)?;
        ctrunc |= self.hop_limit_out(tid, fd, msg, &h)?;
        let mut off = 0;
        for v in &iovs {
            if off >= n {
                break;
            }
            let take = crate::uaccess::host_len(v.len).min(n - off);
            write_bytes(
                self.mm(tid).borrow_mut().mem.as_mut(),
                v.base,
                &data[off..off + take],
            )?;
            off += take;
        }
        let name = u64::from_le_bytes(h[0..8].try_into().unwrap_or_default());
        let mm = self.mm(tid);
        let mut m = mm.borrow_mut();
        if name != 0 {
            let addr = encode(from.as_ref(), family);
            let cap = u32::from_le_bytes(h[8..12].try_into().unwrap_or_default()) as usize;
            write_bytes(m.mem.as_mut(), name, &addr[..addr.len().min(cap)])?;
            write_u32(m.mem.as_mut(), msg + 8, addr.len() as u32)?;
        }
        write_u32(
            m.mem.as_mut(),
            msg + 48,
            if trunc { MSG_TRUNC as u32 } else { 0 } | ctrunc,
        )?;
        Ok(received(n, full, flags))
    }

    pub(crate) fn sys_shutdown(&mut self, tid: Tid, fd: u64, how: u64) -> SysResult {
        let s = match self.any_sock(tid, fd)? {
            (Sock::Local(s), _) => s,
            (Sock::Net(ns), _) => return self.inet_shutdown(&ns, how),
        };
        if !matches!(s.borrow().state, UnixState::Connected) {
            return Err(Errno::ENOTCONN);
        }
        let peer = s.borrow().peer.upgrade();
        match how {
            0 => s.borrow_mut().read_shut = true,
            1 => s.borrow_mut().write_shut = true,
            2 => {
                let mut b = s.borrow_mut();
                b.read_shut = true;
                b.write_shut = true;
            }
            _ => return Err(Errno::EINVAL),
        }
        if how >= 1
            && let Some(p) = &peer
        {
            p.borrow_mut().read_shut = true;
            touch(self, p);
        }
        touch(self, &s);
        Ok(0)
    }

    pub(crate) fn sys_getsockopt(
        &mut self,
        tid: Tid,
        fd: u64,
        level: u64,
        opt: u64,
        val: u64,
        len: u64,
    ) -> SysResult {
        let int = |v: u32| v.to_le_bytes().to_vec();
        if self.netlink_of(tid, fd)?.is_some() {
            return self.netlink_getsockopt(tid, level, opt, val, len);
        }
        let (sock, _) = self.any_sock(tid, fd)?;
        let bytes: Vec<u8> = match sock {
            Sock::Net(ns) => {
                let mut n = ns.borrow_mut();
                match (level, opt) {
                    (SOL_SOCKET, SO_TYPE) => int(1),
                    (SOL_SOCKET, SO_ERROR) => int(u32::from(n.error.take().map_or(0, |e| e.0))),
                    (SOL_SOCKET, SO_SNDBUF) => int(n.tx_cap as u32),
                    (SOL_SOCKET, SO_RCVBUF) => int(n.rx_cap as u32),
                    (SOL_SOCKET, SO_KEEPALIVE) => int(u32::from(n.keepalive)),
                    (SOL_SOCKET, SO_LINGER) => {
                        let mut l = int(u32::from(n.linger_zero));
                        l.extend_from_slice(&[0; 4]);
                        l
                    }
                    (SOL_SOCKET, SO_DOMAIN) => {
                        int(u32::from(if n.v6 { AF_INET6 } else { AF_INET }))
                    }
                    (SOL_SOCKET, SO_PROTOCOL) => int(6),
                    (IPPROTO_TCP, TCP_NODELAY) => int(u32::from(n.nodelay)),
                    (SOL_SOCKET, SO_RCVTIMEO | SO_RCVTIMEO_NEW) => timeval(n.rcvtimeo),
                    (SOL_SOCKET, SO_SNDTIMEO | SO_SNDTIMEO_NEW) => timeval(n.sndtimeo),
                    (IPPROTO_IPV6, IPV6_V6ONLY) => int(u32::from(n.v6only)),
                    (SOL_SOCKET, _) | (IPPROTO_TCP, _) | (IPPROTO_IP | IPPROTO_IPV6, _) => {
                        int(n.opts.get(&(level, opt)).copied().unwrap_or(0))
                    }
                    _ => return Err(Errno::ENOPROTOOPT),
                }
            }
            Sock::Local(s) => {
                let b = s.borrow();
                match (level, opt) {
                    (SOL_SOCKET, SO_TYPE) => int(b.ty.raw()),
                    (SOL_SOCKET, SO_ERROR) => int(0),
                    (SOL_SOCKET, SO_SNDBUF | SO_RCVBUF) => {
                        int(crate::objects::unix::UNIX_BUF as u32)
                    }
                    (SOL_SOCKET, SO_ACCEPTCONN) => {
                        int(u32::from(matches!(b.state, UnixState::Listening { .. })))
                    }
                    (SOL_SOCKET, SO_DOMAIN) => int(u32::from(b.family)),
                    (SOL_SOCKET, SO_PROTOCOL) => int(match (b.family, b.ty) {
                        (AF_UNIX, _) => 0,
                        (AF_INET6, _) if b.icmp => IPPROTO_ICMPV6 as u32,
                        _ if b.icmp => IPPROTO_ICMP as u32,
                        (_, SockType::Dgram) => 17,
                        _ => 6,
                    }),
                    (SOL_SOCKET, SO_PEERCRED) => {
                        // struct ucred { pid, uid, gid }
                        let pid = b.peer.upgrade().map_or(0, |p| p.borrow().owner);
                        let mut c = pid.to_le_bytes().to_vec();
                        c.extend_from_slice(&[0; 8]);
                        c
                    }
                    (IPPROTO_TCP, TCP_NODELAY) => int(u32::from(b.nodelay)),
                    (SOL_SOCKET, SO_RCVTIMEO | SO_RCVTIMEO_NEW) => timeval(b.rcvtimeo),
                    (SOL_SOCKET, SO_SNDTIMEO | SO_SNDTIMEO_NEW) => timeval(b.sndtimeo),
                    (IPPROTO_IPV6, IPV6_V6ONLY) => int(u32::from(b.v6only)),
                    (SOL_SOCKET, _) | (IPPROTO_TCP, _) | (IPPROTO_IP | IPPROTO_IPV6, _) => {
                        int(b.opts.get(&(level, opt)).copied().unwrap_or(0))
                    }
                    _ => return Err(Errno::ENOPROTOOPT),
                }
            }
        };
        self.put_sockopt(tid, level, val, len, &bytes)?;
        Ok(0)
    }

    /// Copies an option's value out like Linux: `*optlen` (an int; negative is EINVAL) caps the
    /// bytes written, and the length written back is what was copied (net/core/sock.c
    /// `sk_getsockopt`: `if (len > lv) len = lv`). At the IP level a buffer shorter than an
    /// int gets a value below 256 as one byte (net/ipv4/ip_sockglue.c `do_ip_getsockopt`).
    pub(super) fn put_sockopt(
        &self,
        tid: Tid,
        level: u64,
        val: u64,
        lenp: u64,
        bytes: &[u8],
    ) -> Result<(), Errno> {
        let mm = self.mm(tid);
        let mut mm = mm.borrow_mut();
        let cap = mm.mem.read_u32(lenp).map_err(|_| Errno::EFAULT)? as i32;
        let cap = usize::try_from(cap).map_err(|_| Errno::EINVAL)?;
        let small = <[u8; 4]>::try_from(bytes).is_ok_and(|v| u32::from_le_bytes(v) < 256);
        let n = if level == IPPROTO_IP && small && (1..4).contains(&cap) {
            1
        } else {
            bytes.len().min(cap)
        };
        write_bytes(mm.mem.as_mut(), val, &bytes[..n])?;
        write_u32(mm.mem.as_mut(), lenp, n as u32)
    }

    pub(crate) fn sys_setsockopt(
        &mut self,
        tid: Tid,
        fd: u64,
        level: u64,
        opt: u64,
        val: u64,
        len: u64,
    ) -> SysResult {
        /// `SOL_NETLINK`, `IPPROTO_UDP`, `IPPROTO_RAW`.
        const SOL_NETLINK: u64 = 270;
        const IPPROTO_UDP: u64 = 17;
        const IPPROTO_RAW: u64 = 255;
        let netlink = self.netlink_of(tid, fd)?.is_some();
        // `optlen` is an int; a negative one is EINVAL (net/socket.c, `__sys_setsockopt`).
        let len = u32::try_from(len as i32).map_err(|_| Errno::EINVAL)?;
        if netlink {
            // Buffer sizes and the like: accepted, nothing to tune. A level the socket has no
            // options at is ENOPROTOOPT (net/netlink/af_netlink.c, `netlink_setsockopt`).
            return match level {
                SOL_SOCKET => sockopt_int_len(level, opt, len).map(|()| 0),
                SOL_NETLINK => Ok(0),
                _ => Err(Errno::ENOPROTOOPT),
            };
        }
        let (sock, _) = self.any_sock(tid, fd)?;
        // Options at a level the socket does not have: AF_UNIX sockets have only SOL_SOCKET's
        // (net/socket.c: no protocol `setsockopt`, EOPNOTSUPP); IP sockets have the IP,
        // transport and ICMPv6 levels too (anything else is ENOPROTOOPT).
        let unix = matches!(&sock, Sock::Local(s) if s.borrow().family == AF_UNIX);
        match level {
            SOL_SOCKET => {}
            _ if unix => return Err(Errno::EOPNOTSUPP),
            IPPROTO_IP | IPPROTO_IPV6 | IPPROTO_TCP | IPPROTO_UDP | IPPROTO_ICMPV6
            | IPPROTO_RAW => {}
            _ => return Err(Errno::ENOPROTOOPT),
        }
        sockopt_int_len(level, opt, len)?;
        let int = match len {
            // The IP level reads a single byte when that is all there is.
            1..=3 if level == IPPROTO_IP => {
                read_bytes(self.mm(tid).borrow_mut().mem.as_mut(), val, 1)?
                    .first()
                    .copied()
                    .map_or(0, u32::from)
            }
            0..=3 => 0,
            _ => self
                .mm(tid)
                .borrow()
                .mem
                .read_u32(val)
                .map_err(|_| Errno::EFAULT)?,
        };
        // SO_RCVTIMEO / SO_SNDTIMEO: a 64-bit `struct timeval` (old and new numbers alike).
        let timeout = if matches!(
            (level, opt),
            (
                SOL_SOCKET,
                SO_RCVTIMEO | SO_SNDTIMEO | SO_RCVTIMEO_NEW | SO_SNDTIMEO_NEW
            )
        ) {
            if len < 16 {
                return Err(Errno::EINVAL);
            }
            let tv: [u8; 16] = read_array(self.mm(tid).borrow_mut().mem.as_mut(), val)?;
            let sec = i64::from_le_bytes(tv[..8].try_into().unwrap_or_default());
            let usec = i64::from_le_bytes(tv[8..].try_into().unwrap_or_default());
            if sec < 0 || !(0..1_000_000).contains(&usec) {
                return Err(Errno::EDOM);
            }
            let ns = (sec as u64)
                .saturating_mul(1_000_000_000)
                .saturating_add(usec as u64 * 1000);
            Some((ns > 0).then_some(ns))
        } else {
            None
        };
        let receive = matches!(opt, SO_RCVTIMEO | SO_RCVTIMEO_NEW);
        // Integer options are remembered for getsockopt; boolean ones read back as 0 or 1.
        let remembered = (len == 4
            && matches!(level, SOL_SOCKET | IPPROTO_TCP | IPPROTO_IP | IPPROTO_IPV6))
        .then(|| {
            let boolean = matches!(
                (level, opt),
                (SOL_SOCKET, 2 | 6 | 10 | 15) | (IPPROTO_TCP, 3)
            );
            if boolean { u32::from(int != 0) } else { int }
        });
        // The provider half this option applies to, if any.
        let net = match &sock {
            Sock::Net(n) => Some(n.clone()),
            Sock::Local(s) => {
                let mut b = s.borrow_mut();
                if let Some(v) = remembered {
                    b.opts.insert((level, opt), v);
                }
                match (level, opt) {
                    (IPPROTO_IPV6, IPV6_V6ONLY) => b.v6only = int != 0,
                    (IPPROTO_TCP, TCP_NODELAY) => b.nodelay = int != 0,
                    _ => {}
                }
                if let Some(t) = timeout {
                    if receive {
                        b.rcvtimeo = t;
                    } else {
                        b.sndtimeo = t;
                    }
                }
                b.net.clone()
            }
        };
        if let (Some(t), Some(n)) = (timeout, &net) {
            let mut b = n.borrow_mut();
            if receive {
                b.rcvtimeo = t;
            } else {
                b.sndtimeo = t;
            }
        }
        // Everything else is accepted: the loopback has no knobs (SO_REUSEADDR is implicit).
        let Some(n) = net else {
            return Ok(0);
        };
        if let Some(v) = remembered {
            n.borrow_mut().opts.insert((level, opt), v);
        }
        let option = match (level, opt) {
            (IPPROTO_TCP, TCP_NODELAY) => {
                n.borrow_mut().nodelay = int != 0;
                Some(NetOption::NoDelay(int != 0))
            }
            (SOL_SOCKET, SO_KEEPALIVE) => {
                n.borrow_mut().keepalive = int != 0;
                Some(NetOption::KeepAlive(int != 0))
            }
            (SOL_SOCKET, SO_LINGER) if len >= 8 => {
                let secs = self
                    .mm(tid)
                    .borrow()
                    .mem
                    .read_u32(val + 4)
                    .map_err(|_| Errno::EFAULT)?;
                n.borrow_mut().linger_zero = int != 0 && secs == 0;
                None
            }
            (SOL_SOCKET, SO_SNDBUF) => {
                n.borrow_mut().tx_cap = (int as usize).clamp(4096, 4 << 20);
                None
            }
            (SOL_SOCKET, SO_RCVBUF) => {
                n.borrow_mut().rx_cap = (int as usize).clamp(4096, 4 << 20);
                None
            }
            (IPPROTO_IP, IP_TTL) | (IPPROTO_IPV6, IPV6_UNICAST_HOPS) if int <= 255 => {
                Some(NetOption::HopLimit(int as u8))
            }
            _ => None,
        };
        if let Some(o) = option
            && let Some(provider) = self.net.as_mut()
        {
            provider.set_option(&n, o)?;
        }
        Ok(0)
    }
}

// ---- ICMP echo ("ping") sockets ----
//
// Linux's unprivileged ping sockets (net/ipv4/ping.c): `SOCK_DGRAM` + `IPPROTO_ICMP` (AF_INET)
// or `IPPROTO_ICMPV6` (AF_INET6). The guest sends ICMP echo requests, header first, and
// receives echo replies, header first; the port of the bound address is the echo identifier,
// which the kernel overwrites in every request together with the checksum. Requests to "this
// host" are answered in the kernel; everything else goes to the provider's ICMP socket
// (`Proto::Icmp`), whose identifier is the one on the wire.
//
// Not modelled: the error queue (`IP_RECVERR` is accepted, `MSG_ERRQUEUE` finds nothing), a
// connected socket's `SO_ERROR` from ICMP errors, and identifier clashes between sockets that
// bound the same identifier explicitly on the loopback.

/// The Internet checksum (RFC 1071) of `parts`, concatenated.
fn inet_checksum(parts: &[&[u8]]) -> u16 {
    let mut sum: u32 = 0;
    let mut odd: Option<u8> = None;
    for part in parts {
        for &byte in *part {
            match odd.take() {
                Some(hi) => sum += u32::from(u16::from_be_bytes([hi, byte])),
                None => odd = Some(byte),
            }
        }
    }
    if let Some(hi) = odd {
        sum += u32::from(u16::from_be_bytes([hi, 0]));
    }
    while sum > 0xffff {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}

impl Kernel {
    /// `bind` on an ICMP socket: the port is the echo identifier (0: the kernel or the provider
    /// picks one). Loopback addresses stay in the kernel; others bind the provider's socket.
    fn bind_icmp(&mut self, s: &UnixRef, a: SocketAddr) -> Result<(), Errno> {
        if s.borrow().name.is_some() {
            return Err(Errno::EINVAL);
        }
        if self.net.is_some() && !is_loopback(a.ip()) {
            let ns = self.attach_net(s, a)?;
            let ident = ns.borrow().local.map_or(a.port(), |l| l.port());
            s.borrow_mut().name = Some(SockName::Inet(SocketAddr::new(a.ip(), ident)));
            return Ok(());
        }
        if !self.this_host(a.ip()) {
            return Err(Errno::EADDRNOTAVAIL);
        }
        let ident = match a.port() {
            0 => Self::icmp_ident(s),
            i => i,
        };
        s.borrow_mut().name = Some(SockName::Inet(SocketAddr::new(a.ip(), ident)));
        Ok(())
    }

    /// The identifier the kernel picks for a loopback ICMP socket: derived from its object id,
    /// so deterministic, never 0.
    fn icmp_ident(s: &UnixRef) -> u16 {
        (s.borrow().id % 0xffff) as u16 + 1
    }

    /// `connect` on an ICMP socket: a default destination (no traffic).
    fn connect_icmp(&mut self, s: &UnixRef, name: SockName) -> Result<(), Errno> {
        let SockName::Inet(a) = name else {
            return Err(Errno::EINVAL);
        };
        let local = if self.this_host(a.ip()) {
            Self::loopback(s.borrow().family)
        } else {
            wildcard(s.borrow().family)
        };
        if s.borrow().name.is_none() {
            self.bind_icmp(s, SocketAddr::new(local, 0))?;
        }
        s.borrow_mut().peer_name = Some(SockName::Inet(SocketAddr::new(a.ip(), 0)));
        Ok(())
    }

    /// Sends one ICMP echo request on an ICMP socket.
    fn icmp_send(
        &mut self,
        s: &UnixRef,
        data: &[u8],
        dest: Option<SockName>,
        nonblock: bool,
    ) -> Result<usize, Errno> {
        let (family, peer) = {
            let b = s.borrow();
            (b.family, b.peer_name.clone())
        };
        let v6 = family == AF_INET6;
        let request = if v6 { 128 } else { 8 };
        if data.len() < 8 || data[0] != request || data[1] != 0 {
            return Err(Errno::EINVAL);
        }
        let dst = match dest.or(peer) {
            Some(SockName::Inet(a)) => a.ip(),
            Some(_) => return Err(Errno::EINVAL),
            None => return Err(Errno::EDESTADDRREQ),
        };
        if !self.this_host(dst) {
            let to = unmap(SocketAddr::new(dst, 0));
            return self.udp_send_remote(s, data, Some(to), nonblock);
        }
        // This host answers by itself. An unbound socket binds the unspecified address, as
        // Linux's autobind does.
        if s.borrow().name.is_none() {
            self.bind_icmp(s, SocketAddr::new(wildcard(family), 0))?;
        }
        let ident = match &s.borrow().name {
            Some(SockName::Inet(l)) => l.port(),
            _ => 0,
        };
        let from = if dst.is_unspecified() {
            Self::loopback(family)
        } else {
            dst
        };
        let mut reply = data.to_vec();
        reply[0] = if v6 { 129 } else { 0 };
        reply[2..4].fill(0);
        reply[4..6].copy_from_slice(&ident.to_be_bytes());
        let sum = match from {
            IpAddr::V6(a) => {
                // The ICMPv6 pseudo-header: source, destination, length, next header.
                let len = (reply.len() as u32).to_be_bytes();
                inet_checksum(&[&a.octets(), &a.octets(), &len, &[0, 0, 0, 58], &reply])
            }
            IpAddr::V4(_) => inet_checksum(&[&reply]),
        };
        reply[2..4].copy_from_slice(&sum.to_be_bytes());
        {
            let mut b = s.borrow_mut();
            if b.queued() + reply.len() > crate::objects::unix::UNIX_BUF {
                // A full receive queue drops the reply, as Linux does.
                return Ok(data.len());
            }
            b.msgs
                .push_back((reply, Some(SockName::Inet(SocketAddr::new(from, 0)))));
            b.msg_fds.push_back(Vec::new());
        }
        touch(self, s);
        Ok(data.len())
    }
}

/// Options set on a socket before it had a network half carry over to it.
fn inherit_options(s: &UnixSocket, n: &mut NetSocket) {
    n.opts = s.opts.clone();
    n.v6only = s.v6only;
    n.nodelay = s.nodelay;
    n.rcvtimeo = s.rcvtimeo;
    n.sndtimeo = s.sndtimeo;
}

/// A timeout as a 64-bit `struct timeval` (zero: none).
fn timeval(ns: Option<u64>) -> Vec<u8> {
    let ns = ns.unwrap_or(0);
    let mut v = (ns / 1_000_000_000).to_le_bytes().to_vec();
    v.extend_from_slice(&((ns % 1_000_000_000) / 1000).to_le_bytes());
    v
}

/// `recvmmsg`: after the first datagram, take only what is already queued.
const MSG_WAITFORONE: u64 = 0x10000;
/// `sizeof(struct mmsghdr)`: a `msghdr` (56 bytes) and `msg_len`, padded.
const MMSGHDR_SIZE: u64 = 64;
/// Most messages per call (`UIO_MAXIOV`).
const MMSG_MAX: u64 = 1024;

impl Kernel {
    /// `sendmmsg(fd, msgvec, vlen, flags)`: sends each message like `sendmsg`, recording its
    /// length in `msg_len`. Stops at the first failure: its error if nothing was sent, else the
    /// count (a later message that would block ends the call too).
    pub(crate) fn sys_sendmmsg(
        &mut self,
        tid: Tid,
        fd: u64,
        vec: u64,
        vlen: u64,
        flags: u64,
    ) -> SysResult {
        let mut sent = 0;
        for i in 0..vlen.min(MMSG_MAX) {
            let at = vec + i * MMSGHDR_SIZE;
            match self.sys_sendmsg(tid, fd, at, flags) {
                Ok(n) => {
                    crate::uaccess::write_u32(
                        self.mm(tid).borrow_mut().mem.as_mut(),
                        at + 56,
                        n as u32,
                    )?;
                    sent += 1;
                }
                Err(e) if sent == 0 => return Err(e),
                Err(_) => {
                    // A wait set up by the failed send is not taken.
                    self.cancel_wait();
                    break;
                }
            }
        }
        Ok(sent)
    }

    /// `recvmmsg(fd, msgvec, vlen, flags, timeout)`: receives into each message like `recvmsg`.
    /// Only the first receive may block (and restart); later ones take what is queued, as with
    /// `MSG_WAITFORONE`. The timeout is checked for validity (Linux only looks at it between
    /// datagrams, so it never ends a wait).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sys_recvmmsg(
        &mut self,
        tid: Tid,
        fd: u64,
        vec: u64,
        vlen: u64,
        flags: u64,
        timeout: u64,
    ) -> SysResult {
        if timeout != 0 {
            let ts = wasmshim_linux_abi::structs::Timespec::from_bytes(
                &crate::uaccess::read_array(self.mm(tid).borrow_mut().mem.as_mut(), timeout)?,
            );
            ts.to_nanos_saturating().ok_or(Errno::EINVAL)?;
        }
        let flags = flags & !MSG_WAITFORONE;
        let mut got = 0;
        for i in 0..vlen.min(MMSG_MAX) {
            let at = vec + i * MMSGHDR_SIZE;
            let f = if got == 0 {
                flags
            } else {
                flags | MSG_DONTWAIT
            };
            match self.sys_recvmsg(tid, fd, at, f) {
                Ok(n) => {
                    crate::uaccess::write_u32(
                        self.mm(tid).borrow_mut().mem.as_mut(),
                        at + 56,
                        n as u32,
                    )?;
                    got += 1;
                }
                Err(e) if got == 0 => return Err(e),
                Err(_) => {
                    self.cancel_wait();
                    break;
                }
            }
        }
        Ok(got)
    }
}
