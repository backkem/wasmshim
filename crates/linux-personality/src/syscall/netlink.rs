//! `AF_NETLINK` sockets (`NETLINK_ROUTE`): the syscall side of
//! [`crate::objects::netlink`]. The socket calls of `net.rs` hand a netlink descriptor here
//! first.

use std::cell::RefCell;
use std::rc::Rc;

use wasmshim_linux_abi::Errno;
use wasmshim_linux_abi::consts::open;

use super::SysResult;
use crate::fd::{FileKind, open_file};
use crate::kernel::{Kernel, Tid};
use crate::objects::netlink::{NETLINK_ROUTE, Netlink};
use crate::uaccess::read_bytes;

pub(super) const AF_NETLINK: u16 = 16;
const SOCK_TYPE_MASK: u64 = 0xf;
const SOCK_DGRAM: u64 = 2;
const SOCK_RAW: u64 = 3;
const SOL_SOCKET: u64 = 1;
const SO_TYPE: u64 = 3;
const SO_ERROR: u64 = 4;
const SO_SNDBUF: u64 = 7;
const SO_RCVBUF: u64 = 8;
const SO_PROTOCOL: u64 = 38;
const SO_DOMAIN: u64 = 39;
/// The buffer sizes reported (Linux's defaults).
const BUF: u32 = 212_992;

pub(super) type NetlinkRef = Rc<RefCell<Netlink>>;

/// `struct sockaddr_nl` of port `pid` with multicast `groups`.
pub(super) fn sockaddr_nl(pid: u32, groups: u32) -> Vec<u8> {
    let mut a = AF_NETLINK.to_le_bytes().to_vec();
    a.extend_from_slice(&[0, 0]);
    a.extend_from_slice(&pid.to_le_bytes());
    a.extend_from_slice(&groups.to_le_bytes());
    a
}

impl Kernel {
    /// `socket(AF_NETLINK, type, proto)`.
    pub(super) fn netlink_socket(&mut self, tid: Tid, ty: u64, proto: u64) -> SysResult {
        if !matches!(ty & SOCK_TYPE_MASK, SOCK_DGRAM | SOCK_RAW) {
            return Err(Errno::ESOCKTNOSUPPORT);
        }
        if proto != NETLINK_ROUTE {
            return Err(Errno::EPROTONOSUPPORT);
        }
        let s = Rc::new(RefCell::new(Netlink {
            id: self.new_object_id(),
            ..Netlink::default()
        }));
        let file = open_file(
            FileKind::Netlink(s),
            open::O_RDWR | (ty & open::O_NONBLOCK),
            b"socket:",
        );
        Ok(self
            .fds(tid)
            .borrow_mut()
            .insert(file, ty & open::O_CLOEXEC != 0, 0)? as u64)
    }

    /// The netlink socket behind `fd`, if it is one (and whether it is non-blocking).
    pub(super) fn netlink_of(
        &self,
        tid: Tid,
        fd: u64,
    ) -> Result<Option<(NetlinkRef, bool)>, Errno> {
        let file = self.file(tid, fd)?;
        let f = file.borrow();
        Ok(match &f.kind {
            FileKind::Netlink(n) => Some((n.clone(), f.flags & open::O_NONBLOCK != 0)),
            _ => None,
        })
    }

    /// The port a socket that was never bound gets on first use: the process id.
    fn netlink_autobind(&self, tid: Tid, n: &NetlinkRef) {
        let mut s = n.borrow_mut();
        if s.port == 0 {
            s.port = self.pid_of(tid);
        }
    }

    /// `bind`: `nl_pid` 0 means "the process id".
    pub(super) fn netlink_bind(&self, tid: Tid, n: &NetlinkRef, addr: u64, len: u64) -> SysResult {
        if len < 12 {
            return Err(Errno::EINVAL);
        }
        let raw = read_bytes(self.mm(tid).borrow_mut().mem.as_mut(), addr, 12)?;
        if u16::from_le_bytes([raw[0], raw[1]]) != AF_NETLINK {
            return Err(Errno::EINVAL);
        }
        let pid = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        let groups = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
        {
            let mut s = n.borrow_mut();
            s.port = pid;
            s.groups = groups;
        }
        self.netlink_autobind(tid, n);
        Ok(0)
    }

    /// `getsockname` (this socket) or `getpeername` (the kernel, port 0).
    pub(super) fn netlink_name(
        &self,
        tid: Tid,
        n: &NetlinkRef,
        addr: u64,
        addrlen: u64,
        peer: bool,
    ) -> SysResult {
        self.netlink_autobind(tid, n);
        let s = n.borrow();
        let bytes = if peer {
            sockaddr_nl(0, 0)
        } else {
            sockaddr_nl(s.port, s.groups)
        };
        drop(s);
        self.put_sockaddr(tid, addr, addrlen, &bytes)?;
        Ok(0)
    }

    /// A request to the kernel (`sendto`, `sendmsg`, `write`); every byte is consumed.
    pub(crate) fn netlink_send(
        &mut self,
        tid: Tid,
        n: &NetlinkRef,
        data: &[u8],
    ) -> Result<usize, Errno> {
        self.netlink_autobind(tid, n);
        let id = {
            let mut s = n.borrow_mut();
            s.request(data, &self.cfg.interfaces)?;
            s.id
        };
        self.object_changed(id);
        Ok(data.len())
    }

    /// The next answer (one datagram), cut to `buf`: (bytes, the datagram's whole length).
    pub(crate) fn netlink_recv(
        &mut self,
        n: &NetlinkRef,
        buf: &mut [u8],
        peek: bool,
        nonblock: bool,
    ) -> Result<(usize, usize), Errno> {
        let mut s = n.borrow_mut();
        let Some(d) = (if peek {
            s.queue.front().cloned()
        } else {
            s.queue.pop_front()
        }) else {
            let id = s.id;
            drop(s);
            return Err(if nonblock {
                Errno::EAGAIN
            } else {
                self.wait_sock(id, None)
            });
        };
        let len = d.len().min(buf.len());
        buf[..len].copy_from_slice(&d[..len]);
        Ok((len, d.len()))
    }

    /// `getsockopt` on a netlink socket: the `SOL_SOCKET` basics.
    pub(super) fn netlink_getsockopt(
        &self,
        tid: Tid,
        level: u64,
        name: u64,
        val: u64,
        lenp: u64,
    ) -> SysResult {
        if level != SOL_SOCKET {
            return Err(Errno::ENOPROTOOPT);
        }
        let v = match name {
            SO_TYPE => SOCK_RAW as u32,
            SO_ERROR => 0,
            SO_SNDBUF | SO_RCVBUF => BUF,
            SO_PROTOCOL => NETLINK_ROUTE as u32,
            SO_DOMAIN => u32::from(AF_NETLINK),
            _ => return Err(Errno::ENOPROTOOPT),
        };
        self.put_sockopt(tid, level, val, lenp, &v.to_le_bytes())?;
        Ok(0)
    }
}
