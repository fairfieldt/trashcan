use super::*;
use netstack3_ip::device::Ipv6DeviceConfigurationUpdate;

const LOOPBACK: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
pub(crate) fn link_local(a: [u8; 16]) -> bool {
    a[0] == 0xfe && a[1] & 0xc0 == 0x80
}
fn mapped(a: [u8; 16]) -> bool {
    a[..10] == [0; 10] && a[10..12] == [255, 255]
}
fn route6(
    subnet: Subnet<Ipv6Addr>,
    device: DeviceId<Bindings>,
    gateway: Option<SpecifiedAddr<Ipv6Addr>>,
    metric: u32,
) -> Entry<Ipv6Addr, DeviceId<Bindings>> {
    Entry {
        subnet,
        device,
        gateway,
        metric: Metric::ExplicitMetric(RawMetric(metric)),
        route_preference: Default::default(),
    }
}
impl Engine {
    /// Enable once, before creating IPv6 sockets. Static addresses and gateways
    /// must be supplied explicitly; router advertisements do not configure them.
    pub fn configure_ipv6(&mut self, config: Ipv6Config) -> Result<(), Error> {
        if self.ipv6.is_some() {
            return Err(Error::WrongState);
        }
        if self.config.mtu < 1280 || config.prefix > 128 {
            return Err(Error::Invalid);
        }
        if let Some(a) = config.address {
            if a == [0; 16] || a == LOOPBACK || a[0] == 255 || mapped(a) || link_local(a) {
                return Err(Error::Invalid);
            }
        }
        if let Some(g) = config.gateway {
            if g == [0; 16] || g == LOOPBACK || g[0] == 255 || mapped(g) {
                return Err(Error::Invalid);
            }
        }
        let ethernet: DeviceId<Bindings> = self.ethernet.as_ref().unwrap().clone().into();
        let lo: DeviceId<Bindings> = self.loopback.as_ref().unwrap().clone().into();
        self.link_local = Mac::new(self.config.mac)
            .to_ipv6_link_local()
            .addr()
            .ipv6_bytes();
        // DAD stays enabled on Ethernet. Loopback has no competing link peers.
        for (device, dad) in [
            (ethernet.clone(), Some(NonZeroU16::new(1).unwrap())),
            (lo.clone(), None),
        ] {
            let enabled = device == lo || self.link;
            self.api()
                .device_ip::<Ipv6>()
                .update_configuration(
                    &device,
                    Ipv6DeviceConfigurationUpdate {
                        ip_config: IpDeviceConfigurationUpdate {
                            ip_enabled: Some(enabled),
                            dad_transmits: Some(dad),
                            ..Default::default()
                        },
                        max_router_solicitations: Some(None),
                        ..Default::default()
                    },
                )
                .map_err(|_| Error::Invalid)?;
        }
        for (device, a, prefix) in [
            (ethernet.clone(), self.link_local, 64),
            (lo.clone(), LOOPBACK, 128),
        ] {
            self.api()
                .device_ip::<Ipv6>()
                .add_ip_addr_subnet(
                    &device,
                    AddrSubnet::new(Ipv6Addr::from_bytes(a), prefix).map_err(|_| Error::Invalid)?,
                )
                .map_err(|_| Error::AddressUnavailable)?;
        }
        let mut routes = vec![
            route6(
                Subnet::new(Ipv6Addr::from_bytes(LOOPBACK), 128).unwrap(),
                lo.clone(),
                None,
                0,
            ),
            route6(
                Subnet::new(
                    Ipv6Addr::from_bytes([0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    64,
                )
                .unwrap(),
                ethernet.clone(),
                None,
                100,
            ),
        ];
        if let Some(a) = config.address {
            let addr = Ipv6Addr::from_bytes(a);
            self.api()
                .device_ip::<Ipv6>()
                .add_ip_addr_subnet(
                    &ethernet,
                    AddrSubnet::new(addr, config.prefix).map_err(|_| Error::Invalid)?,
                )
                .map_err(|_| Error::AddressUnavailable)?;
            self.api()
                .device_ip::<Ipv6>()
                .add_ip_addr_subnet(&lo, AddrSubnet::new(addr, 128).unwrap())
                .map_err(|_| Error::AddressUnavailable)?;
            routes.push(route6(
                Subnet::new(addr.mask(config.prefix), config.prefix).unwrap(),
                ethernet.clone(),
                None,
                100,
            ));
            routes.push(route6(Subnet::new(addr, 128).unwrap(), lo.clone(), None, 0));
        }
        if let Some(g) = config.gateway {
            routes.push(route6(
                Subnet::new(Ipv6Addr::from_bytes([0; 16]), 0).unwrap(),
                ethernet,
                Some(SpecifiedAddr::new(Ipv6Addr::from_bytes(g)).unwrap()),
                100,
            ));
        }
        let table = self.api().routes::<Ipv6>().main_table_id();
        self.api().routes::<Ipv6>().set_routes(
            &table,
            routes
                .into_iter()
                .map(|r| r.with_generation(Generation::initial()))
                .collect(),
        );
        drop(table);
        self.ipv6 = Some(config);
        Ok(())
    }
    /// Sticky diagnostic for a core-reported IPv6 DAD or ongoing conflict.
    pub fn ipv6_dad_failed(&self) -> bool {
        self.bindings.ipv6_dad_failed()
    }
    /// True only after DAD assigned every configured Ethernet address. A DAD
    /// conflict keeps this false; the caller must apply a bounded startup timeout.
    pub fn ipv6_ready(&mut self) -> bool {
        let Some(config) = self.ipv6 else {
            return false;
        };
        let link_local = self.link_local;
        let device = self.ethernet.as_ref().unwrap().clone().into();
        let mut link = false;
        let mut global = config.address.is_none();
        self.api()
            .device_ip::<Ipv6>()
            .for_each_assigned_ip_addr_subnet(&device, |a| {
                let a = a.addr().ipv6_bytes();
                link |= a == link_local;
                global |= config.address == Some(a);
            });
        link && global
    }
    pub fn ipv6_link_local(&self) -> Option<Endpoint6> {
        self.ipv6.map(|_| Endpoint6 {
            address: self.link_local,
            port: 0,
            scope_id: 1,
        })
    }
    pub fn is_local_ip(&self, e: IpEndpoint) -> bool {
        match e {
            IpEndpoint::V4(e) => self.is_local(e.address),
            IpEndpoint::V6(e) => self.ipv6.is_some_and(|c| {
                e.address == LOOPBACK
                    || c.address == Some(e.address)
                    || (e.address == self.link_local && e.scope_id == 1)
            }),
        }
    }
    pub(crate) fn zone6(
        &self,
        e: Endpoint6,
    ) -> Result<Option<ZonedAddr<SpecifiedAddr<Ipv6Addr>, DeviceId<Bindings>>>, Error> {
        if self.ipv6.is_none() {
            return Err(Error::Unsupported);
        }
        if mapped(e.address) || e.address[0] == 255 {
            return Err(Error::Unsupported);
        }
        if e.address == [0; 16] {
            return if e.scope_id == 0 {
                Ok(None)
            } else {
                Err(Error::Invalid)
            };
        }
        let addr = SpecifiedAddr::new(Ipv6Addr::from_bytes(e.address)).unwrap();
        if link_local(e.address) {
            if e.scope_id != 1 {
                return Err(Error::Invalid);
            }
            Ok(Some(ZonedAddr::Zoned(
                AddrAndZone::new(addr, self.ethernet.as_ref().unwrap().clone().into())
                    .ok_or(Error::Invalid)?,
            )))
        } else if e.scope_id != 0 {
            Err(Error::Invalid)
        } else {
            Ok(Some(ZonedAddr::Unzoned(addr)))
        }
    }
    pub(crate) fn bind6(&mut self, id: SocketId, address: Endpoint6) -> Result<(), Error> {
        let i = self.index(id)?;
        if self.sockets[i].mode != Mode::New {
            return Err(Error::WrongState);
        }
        if address.address != [0; 16] && !self.is_local_ip(IpEndpoint::V6(address)) {
            return Err(Error::AddressUnavailable);
        }
        let addr = self.zone6(address)?;
        let CoreSocket::V6(core) = self.sockets[i].core.clone() else {
            return Err(Error::Invalid);
        };
        self.api()
            .tcp::<Ipv6>()
            .bind(&core, addr, NonZeroU16::new(address.port))
            .map_err(bind_error)?;
        self.sockets[i].mode = Mode::Bound;
        Ok(())
    }
    pub(crate) fn connect6(&mut self, id: SocketId, remote: Endpoint6) -> Result<(), Error> {
        let i = self.index(id)?;
        if remote.port == 0 || remote.address == [0; 16] {
            return Err(Error::Invalid);
        }
        if !self.link && !self.is_local_ip(IpEndpoint::V6(remote)) {
            return Err(Error::NoRoute);
        }
        let addr = self.zone6(remote)?;
        if matches!(self.sockets[i].mode, Mode::Listening(_)) {
            return Err(Error::WrongState);
        }
        if self.sockets[i]
            .remote
            .is_some_and(|e| e != IpEndpoint::V6(remote))
        {
            return Err(Error::Invalid);
        }
        self.refresh(i);
        if let Some(e) = self.sockets[i].error {
            return Err(e);
        }
        if self.sockets[i].mode == Mode::Connected {
            return Ok(());
        }
        let CoreSocket::V6(core) = self.sockets[i].core.clone() else {
            return Err(Error::Invalid);
        };
        let result =
            self.api()
                .tcp::<Ipv6>()
                .connect(&core, addr, NonZeroU16::new(remote.port).unwrap());
        self.sockets[i].remote = Some(IpEndpoint::V6(remote));
        match result {
            Ok(()) | Err(ConnectError::Pending) => {
                self.sockets[i].mode = Mode::Connecting;
                Err(Error::WouldBlock)
            }
            Err(ConnectError::Completed) => {
                self.sockets[i].mode = Mode::Connected;
                self.attach_buffers(i);
                Ok(())
            }
            Err(e) => {
                let e = connect_error(e);
                self.sockets[i].error = Some(e);
                Err(e)
            }
        }
    }
    pub(crate) fn accept6(&mut self, id: SocketId) -> Result<(SocketId, IpEndpoint), Error> {
        let i = self.index(id)?;
        if !matches!(self.sockets[i].mode, Mode::Listening(_)) {
            return Err(Error::WrongState);
        }
        if self.accounted() >= MAX_SOCKETS {
            return Err(Error::NoResources);
        }
        let CoreSocket::V6(core) = self.sockets[i].core.clone() else {
            return Err(Error::Invalid);
        };
        let (child, peer, buffers) =
            self.api()
                .tcp::<Ipv6>()
                .accept(&core)
                .map_err(|e| match e {
                    AcceptError::WouldBlock => Error::WouldBlock,
                    _ => Error::WrongState,
                })?;
        let address = peer.ip.addr().ipv6_bytes();
        let remote = IpEndpoint::V6(Endpoint6 {
            address,
            port: peer.port.get(),
            scope_id: if link_local(address) { 1 } else { 0 },
        });
        if !self.link && !self.is_local_ip(remote) {
            let cookie = child.socket_cookie().export_value();
            self.api()
                .tcp::<Ipv6>()
                .disconnect_bound(&IpSocketMatcher::<()>::Cookie(SocketCookieMatcher {
                    cookie,
                    invert: false,
                }));
            self.api().tcp::<Ipv6>().close(child);
            return Err(Error::NoRoute);
        }
        let child = self.allocate(
            CoreSocket::V6(child),
            SocketExtra::default(),
            Some(buffers),
            Mode::Connected,
            Some(remote),
        )?;
        Ok((child, remote))
    }
    pub(crate) fn local_addr6(&mut self, id: SocketId) -> Result<Endpoint6, Error> {
        let i = self.index(id)?;
        let CoreSocket::V6(core) = self.sockets[i].core.clone() else {
            return Err(Error::Invalid);
        };
        let (address, port) = match self.api().tcp::<Ipv6>().get_info(&core) {
            SocketInfo::Unbound(_) => ([0; 16], 0),
            SocketInfo::Bound(x) => (
                x.addr.map_or([0; 16], |a| a.addr().ipv6_bytes()),
                x.port.get(),
            ),
            SocketInfo::Connection(x) => {
                (x.local_addr.ip.addr().ipv6_bytes(), x.local_addr.port.get())
            }
        };
        Ok(Endpoint6 {
            address,
            port,
            scope_id: if link_local(address) { 1 } else { 0 },
        })
    }
}
/// Reject all fragments, including atomic fragments, before core can retain
/// them. Walk only well-sized extension headers within the declared payload.
pub(crate) fn valid_packet(p: &[u8]) -> bool {
    if p.len() < 40 || p[0] >> 4 != 6 {
        return false;
    }
    let len = u16::from_be_bytes([p[4], p[5]]) as usize;
    if len == 0 || 40 + len > p.len() {
        return false;
    }
    let end = 40 + len;
    let mut at = 40;
    let mut next = p[6];
    for _ in 0..16 {
        match next {
            44 => return false,
            0 | 43 | 60 | 51 => {
                if at + 2 > end {
                    return false;
                }
                let size = if next == 51 {
                    (p[at + 1] as usize + 2) * 4
                } else {
                    (p[at + 1] as usize + 1) * 8
                };
                if at + size > end {
                    return false;
                }
                next = p[at];
                at += size;
            }
            6 | 17 | 58 | 59 => return true,
            _ => return false,
        }
    }
    false
}
