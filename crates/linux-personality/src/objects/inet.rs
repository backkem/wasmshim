//! TCP connections through the network provider: the kernel half of the facade.
//!
//! A [`NetSocket`] holds the connection's buffers. Syscalls here only look at and change that
//! state and then kick the provider; they never wait for it. Blocking calls park the thread on
//! the socket's key, and the provider's change reports ([`crate::net::Network::take_changed`])
//! wake it. Readiness is computed from the state, exactly like the in-kernel sockets.

use wasmshim_linux_abi::Errno;

use super::{EPOLLERR, EPOLLHUP, EPOLLIN, EPOLLOUT, EPOLLRDHUP, EPOLLRDNORM, EPOLLWRNORM};
use crate::kernel::Kernel;
use crate::net::{NetRef, NetSocket, NetState, Proto};

/// Readiness (`EPOLL*` bits) of a provider socket, and its generation.
pub(crate) fn events(s: &NetSocket) -> (u32, u64) {
    let mut ev = 0;
    match (s.proto, s.state) {
        (Proto::Udp | Proto::Icmp, _) => {
            if !s.dgrams_in.is_empty() {
                ev |= EPOLLIN | EPOLLRDNORM;
            }
            if s.dgrams_out.len() < crate::net::DGRAM_QUEUE {
                ev |= EPOLLOUT | EPOLLWRNORM;
            }
        }
        (Proto::Tcp, NetState::Listening) => {
            if !s.accepted.is_empty() {
                ev |= EPOLLIN | EPOLLRDNORM;
            }
        }
        (Proto::Tcp, NetState::Idle) => ev |= EPOLLOUT | EPOLLHUP,
        (Proto::Tcp, NetState::Connecting) => {}
        (Proto::Tcp, NetState::Failed) => ev |= EPOLLOUT | EPOLLERR | EPOLLHUP,
        (Proto::Tcp, NetState::Connected) => {
            let ended = s.rx_end.is_some() || s.shut_rd;
            if !s.rx.is_empty() || ended {
                ev |= EPOLLIN | EPOLLRDNORM;
            }
            if ended {
                ev |= EPOLLRDHUP;
            }
            let failed = matches!(s.rx_end, Some(Err(_))) || s.tx_err.is_some();
            if failed {
                ev |= EPOLLERR | EPOLLHUP;
            }
            if ended && s.shut_wr && s.tx_done {
                ev |= EPOLLHUP;
            }
            if !s.shut_wr && !failed && s.tx_space() > 0 {
                ev |= EPOLLOUT | EPOLLWRNORM;
            }
        }
    }
    (ev, s.generation)
}

impl Kernel {
    /// Tells the provider that the kernel changed `s`.
    pub(crate) fn net_kick(&mut self, s: &NetRef) {
        if let Some(net) = self.net.as_mut() {
            net.kick(s);
        }
    }

    /// `write`/`send` on a TCP connection: queue into the send buffer.
    pub(crate) fn inet_send(
        &mut self,
        s: &NetRef,
        data: &[u8],
        nonblock: bool,
    ) -> Result<usize, Errno> {
        let n = {
            let mut b = s.borrow_mut();
            match b.state {
                NetState::Connected => {}
                NetState::Connecting => {
                    if nonblock {
                        return Err(Errno::EAGAIN);
                    }
                    let (id, timeout) = (b.id, b.sndtimeo);
                    drop(b);
                    return Err(self.wait_sock(id, timeout));
                }
                NetState::Failed => {
                    return Err(b.error.take().unwrap_or(Errno::ECONNREFUSED));
                }
                NetState::Idle | NetState::Listening => return Err(Errno::ENOTCONN),
            }
            if let Some(Err(e)) = b.rx_end
                && e == Errno::ECONNRESET
            {
                // The peer reset the connection; Linux reports it once, then EPIPE.
                b.rx_end = Some(Ok(()));
                return Err(e);
            }
            if b.shut_wr || b.tx_err.is_some() {
                return Err(Errno::EPIPE);
            }
            let space = b.tx_space();
            if space == 0 {
                if nonblock {
                    return Err(Errno::EAGAIN);
                }
                let (id, timeout) = (b.id, b.sndtimeo);
                drop(b);
                return Err(self.wait_sock(id, timeout));
            }
            let n = space.min(data.len());
            b.tx.extend(&data[..n]);
            b.touch();
            n
        };
        self.net_kick(s);
        Ok(n)
    }

    /// `read`/`recv` on a TCP connection. Returns the byte count (0 at end of stream).
    pub(crate) fn inet_recv(
        &mut self,
        s: &NetRef,
        buf: &mut [u8],
        peek: bool,
        nonblock: bool,
    ) -> Result<usize, Errno> {
        let n = {
            let mut b = s.borrow_mut();
            match b.state {
                NetState::Connected => {}
                NetState::Connecting => {
                    if nonblock {
                        return Err(Errno::EAGAIN);
                    }
                    let (id, timeout) = (b.id, b.rcvtimeo);
                    drop(b);
                    return Err(self.wait_sock(id, timeout));
                }
                NetState::Failed => {
                    return Err(b.error.take().unwrap_or(Errno::ECONNREFUSED));
                }
                NetState::Idle | NetState::Listening => return Err(Errno::ENOTCONN),
            }
            if b.shut_rd && b.rx.is_empty() {
                return Ok(0);
            }
            if b.rx.is_empty() {
                match b.rx_end {
                    Some(Ok(())) => return Ok(0),
                    Some(Err(e)) => {
                        // Reported once; afterwards the stream reads as ended.
                        b.rx_end = Some(Ok(()));
                        return Err(e);
                    }
                    None => {}
                }
                if buf.is_empty() {
                    return Ok(0);
                }
                if nonblock {
                    return Err(Errno::EAGAIN);
                }
                let (id, timeout) = (b.id, b.rcvtimeo);
                drop(b);
                return Err(self.wait_sock(id, timeout));
            }
            let n = buf.len().min(b.rx.len());
            for (dst, src) in buf.iter_mut().zip(b.rx.iter()) {
                *dst = *src;
            }
            if !peek {
                b.rx.drain(..n);
                b.touch();
            }
            n
        };
        if !peek {
            // The receive window opened: the provider may read more.
            self.net_kick(s);
        }
        Ok(n)
    }

    /// `shutdown(how)` on a TCP connection.
    pub(crate) fn inet_shutdown(&mut self, s: &NetRef, how: u64) -> Result<u64, Errno> {
        {
            let mut b = s.borrow_mut();
            if b.state != NetState::Connected {
                return Err(Errno::ENOTCONN);
            }
            match how {
                0 => b.shut_rd = true,
                1 => b.shut_wr = true,
                2 => {
                    b.shut_rd = true;
                    b.shut_wr = true;
                }
                _ => return Err(Errno::EINVAL),
            }
            if b.shut_rd {
                // Linux sends nothing for SHUT_RD; later data is discarded.
                b.rx.clear();
            }
            b.touch();
        }
        let id = s.borrow().id;
        self.net_kick(s);
        self.object_changed(id);
        Ok(0)
    }

    /// The last descriptor of a provider socket closed. Unread data or `SO_LINGER {1, 0}`
    /// aborts the connection (RST, where the provider can); otherwise the provider drains the
    /// send buffer, sends FIN and releases the socket (Linux's lingering close).
    pub(crate) fn inet_closed(&mut self, s: &NetRef) {
        let accepted: Vec<NetRef> = {
            let mut b = s.borrow_mut();
            b.closed = true;
            if b.proto == Proto::Tcp
                && b.state == NetState::Connected
                && (b.linger_zero || !b.rx.is_empty())
            {
                b.abort = true;
            }
            b.touch();
            b.accepted.drain(..).collect()
        };
        // Connections the guest never accepted are reset, as Linux does on listener close.
        for c in &accepted {
            {
                let mut cb = c.borrow_mut();
                cb.closed = true;
                cb.abort = true;
            }
            self.net_kick(c);
        }
        self.net_kick(s);
    }
}
