//! `AF_NETLINK` sockets, `NETLINK_ROUTE` only: enough of rtnetlink for programs to list the
//! node's network interfaces and addresses (glibc's `getifaddrs`, so libuv's
//! `uv_interface_addresses`, Node's `os.networkInterfaces()`, Python's `socket.if_nameindex`,
//! `ip link`/`ip addr`).
//!
//! A request is a sequence of `nlmsghdr` messages written with `sendto`/`sendmsg`/`write`;
//! answers are queued as datagrams and read back with `recvmsg`/`recvfrom`/`read`. Implemented:
//! `RTM_GETLINK` and `RTM_GETADDR` dumps (`NLM_F_DUMP`), which answer with one `RTM_NEWLINK`
//! per interface (one `RTM_NEWADDR` per address) and then `NLMSG_DONE`. Any other request is
//! answered with `NLMSG_ERROR` `EOPNOTSUPP`, or `EINVAL` for a malformed one. Nothing changes the
//! interfaces, so there are no notifications (multicast groups are accepted and stay silent).

use std::collections::VecDeque;
use std::net::IpAddr;

use wasmshim_linux_abi::Errno;

use crate::objects::{EPOLLIN, EPOLLOUT, EPOLLRDNORM, EPOLLWRNORM};

pub const NETLINK_ROUTE: u64 = 0;

const NLMSG_ERROR: u16 = 2;
const NLMSG_DONE: u16 = 3;
const RTM_NEWLINK: u16 = 16;
const RTM_GETLINK: u16 = 18;
const RTM_NEWADDR: u16 = 20;
const RTM_GETADDR: u16 = 22;
const NLM_F_MULTI: u16 = 0x2;

const NLM_F_DUMP: u16 = 0x300;

const IFLA_ADDRESS: u16 = 1;
const IFLA_BROADCAST: u16 = 2;
const IFLA_IFNAME: u16 = 3;
const IFLA_MTU: u16 = 4;
const IFLA_TXQLEN: u16 = 13;
const IFLA_OPERSTATE: u16 = 16;
const IFA_ADDRESS: u16 = 1;
const IFA_LOCAL: u16 = 2;
const IFA_LABEL: u16 = 3;
const IFA_BROADCAST: u16 = 4;

const ARPHRD_ETHER: u16 = 1;
const ARPHRD_LOOPBACK: u16 = 772;
const IFF_UP: u32 = 0x1;
const IFF_BROADCAST: u32 = 0x2;
const IFF_LOOPBACK: u32 = 0x8;
const IFF_RUNNING: u32 = 0x40;
const IFF_MULTICAST: u32 = 0x1000;
const IFF_LOWER_UP: u32 = 0x1_0000;
const IF_OPER_UP: u8 = 6;
const IF_OPER_UNKNOWN: u8 = 0;
const IFA_F_PERMANENT: u8 = 0x80;
const RT_SCOPE_UNIVERSE: u8 = 0;
const RT_SCOPE_LINK: u8 = 253;
const RT_SCOPE_HOST: u8 = 254;
const AF_INET: u8 = 2;
const AF_INET6: u8 = 10;

/// One network interface as the node presents it (`lo` is always index 1, and implicit).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetInterface {
    pub name: String,
    pub mac: [u8; 6],
    pub mtu: u32,
    /// Addresses with their prefix lengths.
    pub addrs: Vec<(IpAddr, u8)>,
}

impl NetInterface {
    /// `NAME,mac=02:00:00:00:00:01[,mtu=N][,addr=A/P]...` (wasmshim-linux-node's `--iface`).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut parts = spec.split(',');
        let name = parts.next().unwrap_or_default().to_owned();
        if name.is_empty() || name.len() > 15 || name == "lo" {
            return Err(format!(
                "--iface {spec}: a name of 1 to 15 bytes other than lo"
            ));
        }
        let mut iface = Self {
            name,
            mac: [0; 6],
            mtu: 1500,
            addrs: Vec::new(),
        };
        for p in parts {
            let (k, v) = p
                .split_once('=')
                .ok_or(format!("--iface {spec}: {p} is not key=value"))?;
            match k {
                "mac" => {
                    let bytes: Vec<u8> = v
                        .split(':')
                        .map(|b| u8::from_str_radix(b, 16))
                        .collect::<Result<_, _>>()
                        .map_err(|_| format!("--iface {spec}: bad mac {v}"))?;
                    iface.mac = bytes
                        .try_into()
                        .map_err(|_| format!("--iface {spec}: bad mac {v}"))?;
                }
                "mtu" => iface.mtu = v.parse().map_err(|_| format!("--iface {spec}: bad mtu"))?,
                "addr" => {
                    let (a, len) = v.split_once('/').unwrap_or((v, ""));
                    let addr: IpAddr = a
                        .parse()
                        .map_err(|_| format!("--iface {spec}: bad address {v}"))?;
                    let max = if addr.is_ipv4() { 32 } else { 128 };
                    let len = if len.is_empty() {
                        max
                    } else {
                        len.parse::<u8>()
                            .ok()
                            .filter(|l| *l <= max)
                            .ok_or(format!("--iface {spec}: bad prefix in {v}"))?
                    };
                    iface.addrs.push((addr, len));
                }
                _ => return Err(format!("--iface {spec}: unknown key {k}")),
            }
        }
        Ok(iface)
    }
}

/// An `AF_NETLINK` socket's state: its port id and the datagrams waiting to be read.
#[derive(Debug, Default)]
pub struct Netlink {
    /// Object id (readiness waits).
    pub id: u64,
    /// The port id (`nl_pid`): set by `bind`, or the process id on first use.
    pub port: u32,
    pub groups: u32,
    pub queue: VecDeque<Vec<u8>>,
    /// Bumped whenever an answer is queued (edge-triggered epoll).
    pub generation: u64,
}

fn align4(n: usize) -> usize {
    (n + 3) & !3
}

/// The length of the message whose header starts `data`, if it is well formed: at least a
/// header and no longer than `data` (`NLMSG_OK`, include/uapi/linux/netlink.h). Compared as
/// `u64`, never added to a position, so a length near 2^32 cannot wrap on a 32-bit host.
fn message_len(data: &[u8]) -> Option<usize> {
    let h: &[u8; 4] = data.get(..4)?.try_into().ok()?;
    let len = u32::from_le_bytes(*h);
    (len >= 16 && u64::from(len) <= data.len() as u64).then_some(len as usize)
}

/// How far the next message starts after one of `len` bytes, with `rest` bytes left:
/// `NLMSG_ALIGN(len)`, at most `rest` (net/netlink/af_netlink.c, `netlink_rcv_skb`).
fn message_advance(len: usize, rest: usize) -> usize {
    len.checked_add(3).map_or(rest, |n| n & !3).min(rest)
}

/// Appends one route attribute.
fn attr(out: &mut Vec<u8>, ty: u16, data: &[u8]) {
    out.extend_from_slice(&((4 + data.len()) as u16).to_le_bytes());
    out.extend_from_slice(&ty.to_le_bytes());
    out.extend_from_slice(data);
    out.resize(align4(out.len()), 0);
}

/// One message: header (length filled in) and body.
fn message(ty: u16, flags: u16, seq: u32, port: u32, body: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(16 + body.len());
    m.extend_from_slice(&((16 + body.len()) as u32).to_le_bytes());
    m.extend_from_slice(&ty.to_le_bytes());
    m.extend_from_slice(&flags.to_le_bytes());
    m.extend_from_slice(&seq.to_le_bytes());
    m.extend_from_slice(&port.to_le_bytes());
    m.extend_from_slice(body);
    m.resize(align4(m.len()), 0);
    m
}

/// `lo` (index 1) and the configured interfaces (index 2, 3, ...).
fn all(ifaces: &[NetInterface]) -> Vec<(u32, NetInterface, bool)> {
    let lo = NetInterface {
        name: "lo".into(),
        mac: [0; 6],
        mtu: 65536,
        addrs: vec![
            (IpAddr::V4([127, 0, 0, 1].into()), 8),
            (IpAddr::V6(std::net::Ipv6Addr::LOCALHOST), 128),
        ],
    };
    std::iter::once((1, lo, true))
        .chain(
            ifaces
                .iter()
                .enumerate()
                .map(|(i, f)| (i as u32 + 2, f.clone(), false)),
        )
        .collect()
}

fn link_body(index: u32, f: &NetInterface, lo: bool) -> Vec<u8> {
    let mut b = Vec::new();
    b.push(0); // ifi_family: AF_UNSPEC
    b.push(0);
    b.extend_from_slice(&(if lo { ARPHRD_LOOPBACK } else { ARPHRD_ETHER }).to_le_bytes());
    b.extend_from_slice(&(index as i32).to_le_bytes());
    let flags = if lo {
        IFF_UP | IFF_LOOPBACK | IFF_RUNNING | IFF_LOWER_UP
    } else {
        IFF_UP | IFF_BROADCAST | IFF_RUNNING | IFF_MULTICAST | IFF_LOWER_UP
    };
    b.extend_from_slice(&flags.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // ifi_change
    let mut name = f.name.as_bytes().to_vec();
    name.push(0);
    attr(&mut b, IFLA_IFNAME, &name);
    attr(&mut b, IFLA_MTU, &f.mtu.to_le_bytes());
    attr(&mut b, IFLA_TXQLEN, &1000u32.to_le_bytes());
    attr(
        &mut b,
        IFLA_OPERSTATE,
        &[if lo { IF_OPER_UNKNOWN } else { IF_OPER_UP }],
    );
    attr(&mut b, IFLA_ADDRESS, &f.mac);
    attr(&mut b, IFLA_BROADCAST, &if lo { [0; 6] } else { [0xff; 6] });
    b
}

fn addr_body(index: u32, f: &NetInterface, lo: bool, addr: IpAddr, len: u8) -> Vec<u8> {
    let mut b = Vec::new();
    let scope = if lo {
        RT_SCOPE_HOST
    } else if matches!(addr, IpAddr::V6(a) if (a.segments()[0] & 0xffc0) == 0xfe80) {
        RT_SCOPE_LINK
    } else {
        RT_SCOPE_UNIVERSE
    };
    b.push(if addr.is_ipv4() { AF_INET } else { AF_INET6 });
    b.push(len);
    b.push(IFA_F_PERMANENT);
    b.push(scope);
    b.extend_from_slice(&index.to_le_bytes());
    match addr {
        IpAddr::V4(a) => {
            attr(&mut b, IFA_ADDRESS, &a.octets());
            attr(&mut b, IFA_LOCAL, &a.octets());
            if !lo && len < 31 {
                let bcast = u32::from(a) | (u32::MAX >> len);
                attr(&mut b, IFA_BROADCAST, &bcast.to_be_bytes());
            }
            let mut label = f.name.as_bytes().to_vec();
            label.push(0);
            attr(&mut b, IFA_LABEL, &label);
        }
        IpAddr::V6(a) => attr(&mut b, IFA_ADDRESS, &a.octets()),
    }
    b
}

impl Netlink {
    /// Handles the messages of one write, queueing the answers. `Err` only for a write too
    /// short to hold a header.
    pub fn request(&mut self, data: &[u8], ifaces: &[NetInterface]) -> Result<(), Errno> {
        if data.len() < 16 {
            return Err(Errno::EINVAL);
        }
        let mut at = 0;
        while data.len() - at >= 16 {
            let h = &data[at..];
            let ty = u16::from_le_bytes([h[4], h[5]]);
            let flags = u16::from_le_bytes([h[6], h[7]]);
            let seq = u32::from_le_bytes([h[8], h[9], h[10], h[11]]);
            let Some(len) = message_len(h) else {
                self.error(seq, &h[..16], Errno::EINVAL);
                break;
            };
            let dump = flags & NLM_F_DUMP == NLM_F_DUMP;
            match ty {
                RTM_GETLINK | RTM_GETADDR if dump => {
                    // Asked for one family only (`rtgenmsg`/`ifaddrmsg` family byte)?
                    let family = h.get(16).copied().filter(|_| len > 16).unwrap_or(0);
                    let mut datagram = Vec::new();
                    for (index, f, lo) in all(ifaces) {
                        if ty == RTM_GETLINK {
                            datagram.extend(message(
                                RTM_NEWLINK,
                                NLM_F_MULTI,
                                seq,
                                self.port,
                                &link_body(index, &f, lo),
                            ));
                            continue;
                        }
                        for &(addr, plen) in &f.addrs {
                            let fam = if addr.is_ipv4() { AF_INET } else { AF_INET6 };
                            if family == 0 || family == fam {
                                datagram.extend(message(
                                    RTM_NEWADDR,
                                    NLM_F_MULTI,
                                    seq,
                                    self.port,
                                    &addr_body(index, &f, lo, addr, plen),
                                ));
                            }
                        }
                    }
                    if !datagram.is_empty() {
                        self.queue.push_back(datagram);
                    }
                    self.queue.push_back(message(
                        NLMSG_DONE,
                        NLM_F_MULTI,
                        seq,
                        self.port,
                        &0i32.to_le_bytes(),
                    ));
                }
                _ => self.error(seq, &h[..16], Errno::EOPNOTSUPP),
            }
            // `len` is at least a header, so every message moves `at` forward.
            at += message_advance(len, h.len());
        }
        self.generation += 1;
        Ok(())
    }

    /// Queues an `NLMSG_ERROR` for the request whose header is `header`.
    fn error(&mut self, seq: u32, header: &[u8], e: Errno) {
        let mut body = (-i32::from(e.0)).to_le_bytes().to_vec();
        body.extend_from_slice(header);
        self.queue
            .push_back(message(NLMSG_ERROR, 0, seq, self.port, &body));
    }

    /// Readiness: readable while answers wait; always writable.
    pub fn events(&self) -> (u32, u64) {
        let mut ev = EPOLLOUT | EPOLLWRNORM;
        if !self.queue.is_empty() {
            ev |= EPOLLIN | EPOLLRDNORM;
        }
        (ev, self.generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(ty: u16, family: u8) -> Vec<u8> {
        let mut body = vec![family, 0, 0, 0];
        body.resize(16, 0);
        message(ty, 1 | NLM_F_DUMP, 7, 0, &body)
    }

    /// (type, flags, seq, port, body) of every message in a datagram.
    fn messages(d: &[u8]) -> Vec<(u16, u16, u32, u32, Vec<u8>)> {
        let mut out = Vec::new();
        let mut at = 0;
        while at + 16 <= d.len() {
            let len = u32::from_le_bytes(d[at..at + 4].try_into().unwrap()) as usize;
            out.push((
                u16::from_le_bytes([d[at + 4], d[at + 5]]),
                u16::from_le_bytes([d[at + 6], d[at + 7]]),
                u32::from_le_bytes(d[at + 8..at + 12].try_into().unwrap()),
                u32::from_le_bytes(d[at + 12..at + 16].try_into().unwrap()),
                d[at + 16..at + len].to_vec(),
            ));
            at += align4(len);
        }
        out
    }

    fn eth0() -> NetInterface {
        NetInterface::parse("eth0,mac=02:00:00:00:00:02,addr=10.0.0.2/24,addr=fd00::2/64").unwrap()
    }

    #[test]
    fn parses_interfaces() {
        let f = eth0();
        assert_eq!(f.mac, [2, 0, 0, 0, 0, 2]);
        assert_eq!(f.mtu, 1500);
        assert_eq!(f.addrs.len(), 2);
        assert!(NetInterface::parse("lo,mac=00:00:00:00:00:00").is_err());
        assert!(NetInterface::parse("eth0,addr=10.0.0.1/33").is_err());
        assert!(NetInterface::parse("eth0,colour=blue").is_err());
    }

    #[test]
    fn dumps_links_then_done() {
        let mut nl = Netlink {
            port: 42,
            ..Netlink::default()
        };
        nl.request(&request(RTM_GETLINK, 0), &[eth0()]).unwrap();
        assert_eq!(nl.queue.len(), 2);
        let links = messages(&nl.queue[0]);
        assert_eq!(links.len(), 2);
        assert!(
            links
                .iter()
                .all(|m| m.0 == RTM_NEWLINK && m.2 == 7 && m.3 == 42)
        );
        // ifinfomsg: index 2 is eth0, an Ethernet device.
        assert_eq!(
            u16::from_le_bytes([links[1].4[2], links[1].4[3]]),
            ARPHRD_ETHER
        );
        assert_eq!(i32::from_le_bytes(links[1].4[4..8].try_into().unwrap()), 2);
        assert!(links[1].4.windows(5).any(|w| w == b"eth0\0"));
        let done = messages(&nl.queue[1]);
        assert_eq!(done[0].0, NLMSG_DONE);
    }

    #[test]
    fn dumps_addresses_by_family() {
        let mut nl = Netlink::default();
        nl.request(&request(RTM_GETADDR, 0), &[eth0()]).unwrap();
        let addrs = messages(&nl.queue[0]);
        // lo: 127.0.0.1 and ::1; eth0: 10.0.0.2/24 and fd00::2/64.
        assert_eq!(addrs.len(), 4);
        assert_eq!(&addrs[2].4[..2], &[AF_INET, 24]);
        assert!(addrs[2].4.windows(4).any(|w| w == [10, 0, 0, 255]));
        nl.queue.clear();
        nl.request(&request(RTM_GETADDR, AF_INET6), &[eth0()])
            .unwrap();
        assert_eq!(messages(&nl.queue[0]).len(), 2);
    }

    #[test]
    fn other_requests_get_an_error() {
        let mut nl = Netlink::default();
        nl.request(&message(24, 1, 3, 0, &[0; 12]), &[]).unwrap();
        let e = messages(&nl.queue[0]);
        assert_eq!(e[0].0, NLMSG_ERROR);
        assert_eq!(
            i32::from_le_bytes(e[0].4[..4].try_into().unwrap()),
            -i32::from(Errno::EOPNOTSUPP.0)
        );
        assert!(nl.request(&[0; 8], &[]).is_err());
    }

    #[test]
    fn a_length_past_the_data_ends_the_request() {
        // NLMSG_OK: a length below the header or beyond the data is malformed. Lengths near
        // 2^32 (which wrap a 32-bit position) are refused like any other.
        for bad in [0u32, 15, 33, 0xffff_fff0, 0xffff_fffc, u32::MAX] {
            let mut nl = Netlink::default();
            let mut data = request(RTM_GETLINK, 0);
            let mut second = request(RTM_GETLINK, 0);
            second[..4].copy_from_slice(&bad.to_le_bytes());
            data.extend_from_slice(&second);
            nl.request(&data, &[]).unwrap();
            // The first request's dump and its end, then one error, and nothing more.
            assert_eq!(nl.queue.len(), 3, "length {bad:#x}");
            let e = messages(&nl.queue[2]);
            assert_eq!(e[0].0, NLMSG_ERROR);
            assert_eq!(
                i32::from_le_bytes(e[0].4[..4].try_into().unwrap()),
                -i32::from(Errno::EINVAL.0)
            );
        }
    }

    #[test]
    fn message_arithmetic_holds_at_the_limits_of_usize() {
        // A message ending the data, unaligned: the next position is the end.
        assert_eq!(message_advance(17, 17), 17);
        assert_eq!(message_advance(16, 40), 16);
        assert_eq!(message_advance(17, 40), 20);
        // On a 32-bit host a guest's length can be `usize::MAX`: aligning it must not wrap.
        assert_eq!(message_advance(usize::MAX, 64), 64);
        assert_eq!(message_advance(usize::MAX - 2, 64), 64);
        let mut h = [0u8; 16];
        h[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(message_len(&h), None);
        h[..4].copy_from_slice(&16u32.to_le_bytes());
        assert_eq!(message_len(&h), Some(16));
        assert_eq!(message_len(&h[..15]), None);
    }
}
