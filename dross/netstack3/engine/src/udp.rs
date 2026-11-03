use super::*;
use either::Either;
use netstack3_datagram::{PendingDatagramSocketError, SocketInfo as UdpInfo};
use netstack3_ip::socket::IpSockSendError;
use netstack3_udp::{SendToError, UdpRemotePort, UdpSocketId};
use oscar_netstack3_native::datagrams::UdpRx;
#[derive(Clone)]
enum CoreUdp {
    V4(UdpSocketId<Ipv4, WeakDeviceId<Bindings>, Bindings>),
    V6(UdpSocketId<Ipv6, WeakDeviceId<Bindings>, Bindings>),
}
pub(super) struct DatagramSocket {
    pub(super) id: SocketId,
    core: CoreUdp,
    pub(super) remote: Option<IpEndpoint>,
    read_closed: bool,
    write_closed: bool,
    error: Option<Error>,
}
fn pending_error(e: PendingDatagramSocketError) -> Error {
    match e {
        PendingDatagramSocketError::PortUnreachable => Error::Refused,
        PendingDatagramSocketError::PacketTooBig => Error::MessageTooLarge,
        PendingDatagramSocketError::Aborted => Error::Reset,
        _ => Error::NoRoute,
    }
}
fn send_error(e: SendToError) -> Error {
    match e {
        SendToError::NotWriteable => Error::BrokenPipe,
        SendToError::LocalAddress(e) => local_error(e),
        SendToError::Send(IpSockSendError::Mtu) => Error::MessageTooLarge,
        SendToError::CreateSock(_) | SendToError::Send(_) => Error::NoRoute,
        _ => Error::Invalid,
    }
}
fn local_error(e: netstack3_core::error::LocalAddressError) -> Error {
    match e {
        netstack3_core::error::LocalAddressError::AddressInUse => Error::AddressInUse,
        netstack3_core::error::LocalAddressError::FailedToAllocateLocalPort => Error::NoResources,
        _ => Error::AddressUnavailable,
    }
}
impl Engine {
    pub(super) fn udp_index(&self, id: SocketId) -> Option<usize> {
        self.datagrams.iter().position(|s| s.id == id)
    }
    fn datagram_index(&self, id: SocketId) -> Result<usize, Error> {
        self.udp_index(id).ok_or(Error::BadSocket)
    }
    pub fn create_udp(&mut self, family: Family) -> Result<SocketId, Error> {
        if self.accounted() >= MAX_SOCKETS || self.bindings.failed() {
            return Err(Error::NoResources);
        }
        if family == Family::Ipv6 && self.ipv6.is_none() {
            return Err(Error::Unsupported);
        }
        self.datagrams
            .try_reserve(1)
            .map_err(|_| Error::NoResources)?;
        let id = SocketId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or(Error::NoResources)?;
        let core = match family {
            Family::Ipv4 => CoreUdp::V4(self.api().udp::<Ipv4>().create_with(UdpRx::default())),
            Family::Ipv6 => {
                let c = self.api().udp::<Ipv6>().create_with(UdpRx::default());
                self.api()
                    .udp::<Ipv6>()
                    .set_dual_stack_enabled(&c, false)
                    .map_err(|_| Error::Invalid)?;
                CoreUdp::V6(c)
            }
        };
        self.datagrams.push(DatagramSocket {
            id,
            core,
            remote: None,
            read_closed: false,
            write_closed: false,
            error: None,
        });
        Ok(id)
    }
    pub fn udp_bind(&mut self, id: SocketId, local: IpEndpoint) -> Result<(), Error> {
        let i = self.datagram_index(id)?;
        match (self.datagrams[i].core.clone(), local) {
            (CoreUdp::V4(c), IpEndpoint::V4(e)) => {
                if e.address != [0; 4] && !self.is_local(e.address) {
                    return Err(Error::AddressUnavailable);
                }
                self.ensure_local_alias(e.address)?;
                self.api()
                    .udp::<Ipv4>()
                    .listen(
                        &c,
                        SpecifiedAddr::new(Ipv4Addr::new(e.address)).map(ZonedAddr::Unzoned),
                        NonZeroU16::new(e.port),
                    )
                    .map_err(|e| match e {
                        Either::Left(_) => Error::WrongState,
                        Either::Right(e) => local_error(e),
                    })
            }
            (CoreUdp::V6(c), IpEndpoint::V6(e)) => {
                if e.address != [0; 16] && !self.is_local_ip(local) {
                    return Err(Error::AddressUnavailable);
                }
                let addr = self.zone6(e)?;
                self.api()
                    .udp::<Ipv6>()
                    .listen(&c, addr, NonZeroU16::new(e.port))
                    .map_err(|e| match e {
                        Either::Left(_) => Error::WrongState,
                        Either::Right(e) => local_error(e),
                    })
            }
            _ => Err(Error::Invalid),
        }
    }
    pub fn udp_connect(&mut self, id: SocketId, remote: Option<IpEndpoint>) -> Result<(), Error> {
        let i = self.datagram_index(id)?;
        if remote.is_some_and(|r| !self.link && !self.is_local_ip(r)) {
            return Err(Error::NoRoute);
        }
        if remote.is_none() {
            if self.datagrams[i].remote.is_none() {
                return Ok(());
            }
            match self.datagrams[i].core.clone() {
                CoreUdp::V4(c) => self.api().udp::<Ipv4>().disconnect(&c),
                CoreUdp::V6(c) => self.api().udp::<Ipv6>().disconnect(&c),
            }
            .map_err(|_| Error::NotConnected)?;
            self.datagrams[i].remote = None;
            return Ok(());
        }
        let result = match (self.datagrams[i].core.clone(), remote.unwrap()) {
            (CoreUdp::V4(c), IpEndpoint::V4(e)) => {
                if e.address == [0; 4] || e.port == 0 {
                    return Err(Error::Invalid);
                }
                self.ensure_local_alias(e.address)?;
                self.api().udp::<Ipv4>().connect(
                    &c,
                    Some(ZonedAddr::Unzoned(
                        SpecifiedAddr::new(Ipv4Addr::new(e.address)).unwrap(),
                    )),
                    UdpRemotePort::Set(NonZeroU16::new(e.port).unwrap()),
                )
            }
            (CoreUdp::V6(c), IpEndpoint::V6(e)) => {
                if e.address == [0; 16] || e.port == 0 {
                    return Err(Error::Invalid);
                }
                let addr = self.zone6(e)?;
                self.api().udp::<Ipv6>().connect(
                    &c,
                    addr,
                    UdpRemotePort::Set(NonZeroU16::new(e.port).unwrap()),
                )
            }
            _ => return Err(Error::Invalid),
        };
        result.map_err(|e| match e {
            netstack3_datagram::ConnectError::Ip(_) => Error::NoRoute,
            netstack3_datagram::ConnectError::CouldNotAllocateLocalPort => Error::NoResources,
            netstack3_datagram::ConnectError::SockAddrConflict => Error::AddressInUse,
            _ => Error::Invalid,
        })?;
        self.datagrams[i].remote = remote;
        Ok(())
    }
    /// Success means the whole packet was accepted by IP, not acknowledged
    /// delivery; UDP can lose packets in neighbor and network queues.
    pub fn send_to(
        &mut self,
        id: SocketId,
        body: &[u8],
        remote: Option<IpEndpoint>,
    ) -> Result<usize, Error> {
        let i = self.datagram_index(id)?;
        if self.datagrams[i].write_closed {
            return Err(Error::BrokenPipe);
        }
        if let Some(e) = self.udp_error(id, false)? {
            return Err(e);
        }
        let remote = remote
            .or(self.datagrams[i].remote)
            .ok_or(Error::NotConnected)?;
        if !self.link && !self.is_local_ip(remote) {
            return Err(Error::NoRoute);
        }
        // Keep UDP from filling core's much larger loopback queue between
        // event-loop turns. A queued local delivery must be pumped first.
        if self.is_local_ip(remote) {
            self.loopback_pending |= self.bindings.take_rx_pending();
            if self.loopback_pending {
                return Err(Error::WouldBlock);
            }
        }
        let max = self.config.mtu as usize
            - match remote {
                IpEndpoint::V4(_) => 28,
                IpEndpoint::V6(_) => 48,
            };
        if body.len() > max {
            return Err(Error::MessageTooLarge);
        }
        if self.bindings.frame_count() >= oscar_netstack3_native::MAX_FRAMES {
            return Err(Error::WouldBlock);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(body.len())
            .map_err(|_| Error::NoResources)?;
        bytes.extend_from_slice(body);
        let result = match (self.datagrams[i].core.clone(), remote) {
            (CoreUdp::V4(c), IpEndpoint::V4(e)) => {
                if e.port == 0 || e.address == [0; 4] {
                    return Err(Error::Invalid);
                }
                self.ensure_local_alias(e.address)?;
                self.api().udp::<Ipv4>().send_to(
                    &c,
                    Some(ZonedAddr::Unzoned(
                        SpecifiedAddr::new(Ipv4Addr::new(e.address)).unwrap(),
                    )),
                    UdpRemotePort::Set(NonZeroU16::new(e.port).unwrap()),
                    Buf::new(bytes, ..),
                    (),
                )
            }
            (CoreUdp::V6(c), IpEndpoint::V6(e)) => {
                if e.port == 0 || e.address == [0; 16] {
                    return Err(Error::Invalid);
                }
                let addr = self.zone6(e)?;
                self.api().udp::<Ipv6>().send_to(
                    &c,
                    addr,
                    UdpRemotePort::Set(NonZeroU16::new(e.port).unwrap()),
                    Buf::new(bytes, ..),
                    (),
                )
            }
            _ => return Err(Error::Invalid),
        };
        result.map_err(send_error)?;
        Ok(body.len())
    }
    pub fn recv_from(
        &mut self,
        id: SocketId,
        out: &mut [u8],
        peek: bool,
    ) -> Result<DatagramRead, Error> {
        let i = self.datagram_index(id)?;
        if self.datagrams[i].read_closed {
            return Err(Error::BrokenPipe);
        }
        let result =
            match &self.datagrams[i].core {
                CoreUdp::V4(c) => c.external_data().receive(out, peek).map(
                    |(m, interface, copied, original_len)| DatagramRead {
                        copied,
                        original_len,
                        truncated: copied < original_len,
                        sender: IpEndpoint::V4(Endpoint {
                            address: m.src_ip.ipv4_bytes(),
                            port: m.src_port.map_or(0, NonZeroU16::get),
                        }),
                        local: IpEndpoint::V4(Endpoint {
                            address: m.dst_ip.ipv4_bytes(),
                            port: m.dst_port.get(),
                        }),
                        interface,
                    },
                ),
                CoreUdp::V6(c) => c.external_data().receive(out, peek).map(
                    |(m, interface, copied, original_len)| {
                        let a = m.src_ip.ipv6_bytes();
                        let b = m.dst_ip.ipv6_bytes();
                        DatagramRead {
                            copied,
                            original_len,
                            truncated: copied < original_len,
                            sender: IpEndpoint::V6(Endpoint6 {
                                address: a,
                                port: m.src_port.map_or(0, NonZeroU16::get),
                                scope_id: if ipv6::link_local(a) { interface } else { 0 },
                            }),
                            local: IpEndpoint::V6(Endpoint6 {
                                address: b,
                                port: m.dst_port.get(),
                                scope_id: if ipv6::link_local(b) { interface } else { 0 },
                            }),
                            interface,
                        }
                    },
                ),
            };
        result.ok_or_else(|| {
            self.udp_error(id, false)
                .ok()
                .flatten()
                .unwrap_or(Error::WouldBlock)
        })
    }
    pub(super) fn udp_error(&mut self, id: SocketId, take: bool) -> Result<Option<Error>, Error> {
        let i = self.datagram_index(id)?;
        let e = match &self.datagrams[i].core {
            CoreUdp::V4(c) => c.external_data().error(take),
            CoreUdp::V6(c) => c.external_data().error(take),
        }
        .map(pending_error);
        if let Some(e) = e {
            if self.datagrams[i].error.is_none() {
                self.datagrams[i].error = Some(e);
            }
        }
        Ok(if take {
            self.datagrams[i].error.take()
        } else {
            self.datagrams[i].error
        })
    }
    pub(super) fn udp_shutdown(&mut self, id: SocketId, dir: Shutdown) -> Result<(), Error> {
        let i = self.datagram_index(id)?;
        if matches!(dir, Shutdown::Read | Shutdown::Both) {
            self.datagrams[i].read_closed = true;
            match &self.datagrams[i].core {
                CoreUdp::V4(c) => c.external_data().shutdown_receive(),
                CoreUdp::V6(c) => c.external_data().shutdown_receive(),
            };
        }
        if matches!(dir, Shutdown::Write | Shutdown::Both) {
            self.datagrams[i].write_closed = true;
        }
        Ok(())
    }
    pub(super) fn udp_readiness(&mut self, id: SocketId) -> Result<Readiness, Error> {
        let i = self.datagram_index(id)?;
        let error = self.udp_error(id, false)?.is_some();
        self.loopback_pending |= self.bindings.take_rx_pending();
        let s = &self.datagrams[i];
        let readable = match &s.core {
            CoreUdp::V4(c) => c.external_data().readable(),
            CoreUdp::V6(c) => c.external_data().readable(),
        };
        Ok(Readiness {
            readable: readable || s.read_closed || error,
            writable: (!s.write_closed
                && !self.loopback_pending
                && self.bindings.frame_count() < oscar_netstack3_native::MAX_FRAMES)
                || error,
            connected: s.remote.is_some(),
            hangup: s.read_closed && s.write_closed,
            error,
            accept: false,
        })
    }
    pub(super) fn udp_close(&mut self, id: SocketId) -> Result<(), Error> {
        let i = self.datagram_index(id)?;
        let s = self.datagrams.swap_remove(i);
        match s.core {
            CoreUdp::V4(c) => {
                let _ = self.api().udp::<Ipv4>().close(c);
            }
            CoreUdp::V6(c) => {
                let _ = self.api().udp::<Ipv6>().close(c);
            }
        };
        Ok(())
    }
    pub(super) fn udp_local_addr(&mut self, id: SocketId) -> Result<IpEndpoint, Error> {
        let i = self.datagram_index(id)?;
        Ok(match self.datagrams[i].core.clone() {
            CoreUdp::V4(c) => {
                let (address, port) = match self.api().udp::<Ipv4>().get_info(&c) {
                    UdpInfo::Unbound => ([0; 4], 0),
                    UdpInfo::Listener(x) => (
                        x.local_ip.map_or([0; 4], |a| a.addr().ipv4_bytes()),
                        x.local_identifier.get(),
                    ),
                    UdpInfo::Connected(x) => {
                        (x.local_ip.addr().ipv4_bytes(), x.local_identifier.get())
                    }
                };
                IpEndpoint::V4(Endpoint { address, port })
            }
            CoreUdp::V6(c) => {
                let (address, port) = match self.api().udp::<Ipv6>().get_info(&c) {
                    UdpInfo::Unbound => ([0; 16], 0),
                    UdpInfo::Listener(x) => (
                        x.local_ip.map_or([0; 16], |a| a.addr().ipv6_bytes()),
                        x.local_identifier.get(),
                    ),
                    UdpInfo::Connected(x) => {
                        (x.local_ip.addr().ipv6_bytes(), x.local_identifier.get())
                    }
                };
                IpEndpoint::V6(Endpoint6 {
                    address,
                    port,
                    scope_id: if ipv6::link_local(address) { 1 } else { 0 },
                })
            }
        })
    }
    pub(super) fn udp_link_down(&mut self) {
        for i in 0..self.datagrams.len() {
            if self.datagrams[i]
                .remote
                .is_some_and(|r| !self.is_local_ip(r))
            {
                self.datagrams[i].error = Some(Error::NoRoute);
            }
        }
    }
    pub(super) fn udp_uses_alias(&mut self, alias: [u8; 4]) -> bool {
        for i in 0..self.datagrams.len() {
            let id = self.datagrams[i].id;
            if matches!(self.udp_local_addr(id),Ok(IpEndpoint::V4(e)) if e.address==alias)
                || matches!(self.datagrams[i].remote,Some(IpEndpoint::V4(e)) if e.address==alias)
            {
                return true;
            }
        }
        false
    }
    pub fn udp_queue_usage(&self, id: SocketId) -> Result<(usize, usize, u64), Error> {
        let i = self.datagram_index(id)?;
        Ok(match &self.datagrams[i].core {
            CoreUdp::V4(c) => c.external_data().counters(),
            CoreUdp::V6(c) => c.external_data().counters(),
        })
    }
    pub fn random_u16(&mut self) -> Result<u16, Error> {
        let mut bytes = [0; 2];
        dross_rt::sys::random(&mut bytes).map_err(|_| Error::NoResources)?;
        Ok(u16::from_ne_bytes(bytes))
    }
    pub fn random_u64(&mut self) -> Result<u64, Error> {
        let mut bytes = [0; 8];
        dross_rt::sys::random(&mut bytes).map_err(|_| Error::NoResources)?;
        Ok(u64::from_ne_bytes(bytes))
    }
}
