//! Bounded IPv4/TCP engine shared by native and compatibility service adapters.
//! The caller supplies monotonic nanoseconds and bounded Ethernet frames. No RPC,
//! Linux errno, NIC authority, or polling thread is part of this library.
#![no_std]
#![recursion_limit = "256"]
extern crate alloc;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::num::{NonZeroU16, NonZeroU64, NonZeroUsize};
use net_types::ethernet::Mac;
use net_types::ip::{AddrSubnet, IpAddress as _, Ipv4, Ipv4Addr, Mtu, Subnet};
use net_types::{SpecifiedAddr, UnicastAddr, ZonedAddr};
use netstack3_base::{RngContext as _, TcpSocketState, WorkQueueReport};
use netstack3_core::ip::IidSecret;
use netstack3_core::socket::{IpSocketMatcher, ShutdownType, SocketCookieMatcher};
use netstack3_core::{CoreApi, StackState, StackStateBuilder};
use netstack3_device::ethernet::{
    EthernetCreationProperties, EthernetDeviceId, EthernetLinkDevice, MaxEthernetFrameSize,
    RecvEthernetFrameMeta,
};
use netstack3_device::loopback::{LoopbackCreationProperties, LoopbackDevice, LoopbackDeviceId};
use netstack3_device::{DeviceId, WeakDeviceId};
use netstack3_ip::device::IpDeviceConfigurationUpdate;
use netstack3_ip::{Entry, Generation, Metric, RawMetric};
use netstack3_tcp::{
    AcceptError, BindError, ConnectError, ConnectionError, ListenError, SocketInfo, TcpSocketId,
};
use oscar_netstack3_native::buffers::{ClientBuffers, SocketExtra};
use oscar_netstack3_native::time::Instant;
use oscar_netstack3_native::{Bindings, DeviceState, Identifier, TxAllocator};
use packet::Buf;

pub const MAX_SOCKETS: usize = 32;
pub const MAX_BACKLOG: usize = 4;
pub const BUFFER_BYTES: usize = 32 * 1024;
const ABANDON_NS: u64 = 5_000_000_000;
const TIMER_BATCH: usize = 64;
const LOOPBACK_BATCH: usize = 8;
type CoreSocket = TcpSocketId<Ipv4, WeakDeviceId<Bindings>, Bindings>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SocketId(u64);
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Endpoint {
    pub address: [u8; 4],
    pub port: u16,
}
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub mac: [u8; 6],
    pub address: [u8; 4],
    pub prefix: u8,
    pub gateway: Option<[u8; 4]>,
    pub mtu: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    WouldBlock,
    Invalid,
    WrongState,
    AddressInUse,
    AddressUnavailable,
    NoRoute,
    Refused,
    Reset,
    TimedOut,
    NoResources,
    NotConnected,
    BrokenPipe,
    BadSocket,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shutdown {
    Read,
    Write,
    Both,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Readiness {
    pub readable: bool,
    pub writable: bool,
    pub accept: bool,
    pub connected: bool,
    pub hangup: bool,
    pub error: bool,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Pump {
    pub more_work: bool,
    pub failed: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    New,
    Bound,
    Listening(usize),
    Connecting,
    Connected,
}
struct Socket {
    id: SocketId,
    core: CoreSocket,
    extra: SocketExtra,
    buffers: Option<ClientBuffers>,
    mode: Mode,
    remote: Option<Endpoint>,
    error: Option<Error>,
    send_closed: bool,
    receive_closed: bool,
    closing: Option<u64>,
}

pub struct Engine {
    core: StackState<Bindings>,
    bindings: Bindings,
    ethernet: Option<EthernetDeviceId<Bindings>>,
    loopback: Option<LoopbackDeviceId<Bindings>>,
    sockets: Vec<Socket>,
    next_id: u64,
    config: Config,
    link: bool,
    now: u64,
    loopback_pending: bool,
    aliases: Vec<[u8; 4]>,
}
impl Engine {
    pub fn new(config: Config, now_ns: u64) -> Result<Self, Error> {
        if config.prefix > 32 || config.mtu < 576 || config.mtu > 2000 || config.address == [0; 4] {
            return Err(Error::Invalid);
        }
        let mac = UnicastAddr::new(Mac::new(config.mac)).ok_or(Error::Invalid)?;
        let mut sockets = Vec::new();
        sockets
            .try_reserve_exact(MAX_SOCKETS)
            .map_err(|_| Error::NoResources)?;
        let mut bindings = Bindings::with_tcp_bytes(BUFFER_BYTES);
        bindings.set_now(Instant(now_ns));
        let mut builder = StackStateBuilder::default();
        builder
            .ipv6_builder()
            .slaac_stable_secret_key(IidSecret::new_random(&mut bindings.rng()));
        let core = builder.build_with_ctx(&mut bindings);
        let ethernet = core
            .api(&mut bindings)
            .device::<EthernetLinkDevice>()
            .add_device(
                Identifier(NonZeroU64::new(1).unwrap()),
                EthernetCreationProperties {
                    mac,
                    max_frame_size: MaxEthernetFrameSize::from_mtu(Mtu::new(config.mtu))
                        .ok_or(Error::Invalid)?,
                    tx_offload_spec: Default::default(),
                },
                RawMetric(100),
                DeviceState,
                TxAllocator,
            );
        let loopback = core
            .api(&mut bindings)
            .device::<LoopbackDevice>()
            .add_device(
                Identifier(NonZeroU64::new(2).unwrap()),
                LoopbackCreationProperties {
                    mtu: Mtu::new(config.mtu),
                },
                RawMetric(0),
                DeviceState,
                netstack3_device::queue::BufVecU8Allocator,
            );
        for device in [
            DeviceId::from(ethernet.clone()),
            DeviceId::from(loopback.clone()),
        ] {
            core.api(&mut bindings)
                .device_ip::<Ipv4>()
                .update_configuration(
                    &device,
                    IpDeviceConfigurationUpdate {
                        ip_enabled: Some(true),
                        ..Default::default()
                    }
                    .into(),
                )
                .map_err(|_| Error::Invalid)?;
        }
        let address = Ipv4Addr::new(config.address);
        core.api(&mut bindings)
            .device_ip::<Ipv4>()
            .add_ip_addr_subnet(
                &ethernet.clone().into(),
                AddrSubnet::new(address, config.prefix).map_err(|_| Error::Invalid)?,
            )
            .map_err(|_| Error::Invalid)?;
        for (address, prefix) in [(Ipv4Addr::new([127, 0, 0, 1]), 8), (address, 32)] {
            core.api(&mut bindings)
                .device_ip::<Ipv4>()
                .add_ip_addr_subnet(
                    &loopback.clone().into(),
                    AddrSubnet::new(address, prefix).map_err(|_| Error::Invalid)?,
                )
                .map_err(|_| Error::Invalid)?;
        }
        let mut routes = vec![
            route(
                Subnet::new(address.mask(config.prefix), config.prefix)
                    .map_err(|_| Error::Invalid)?,
                ethernet.clone().into(),
                None,
                100,
            ),
            route(
                Subnet::new(Ipv4Addr::new([127, 0, 0, 0]), 8).unwrap(),
                loopback.clone().into(),
                None,
                0,
            ),
            route(
                Subnet::new(address, 32).unwrap(),
                loopback.clone().into(),
                None,
                0,
            ),
        ];
        if let Some(gateway) = config.gateway {
            let gateway = SpecifiedAddr::new(Ipv4Addr::new(gateway)).ok_or(Error::Invalid)?;
            routes.push(route(
                Subnet::new(Ipv4Addr::new([0; 4]), 0).unwrap(),
                ethernet.clone().into(),
                Some(gateway),
                100,
            ));
        }
        let table = core.api(&mut bindings).routes::<Ipv4>().main_table_id();
        core.api(&mut bindings).routes::<Ipv4>().set_routes(
            &table,
            routes
                .into_iter()
                .map(|entry| entry.with_generation(Generation::initial()))
                .collect(),
        );
        drop(table);
        Ok(Self {
            core,
            bindings,
            ethernet: Some(ethernet),
            loopback: Some(loopback),
            sockets,
            next_id: 1,
            config,
            link: true,
            now: now_ns,
            loopback_pending: false,
            aliases: Vec::new(),
        })
    }
    fn api(&mut self) -> CoreApi<'_, &mut Bindings> {
        self.core.api(&mut self.bindings)
    }
    fn index(&self, id: SocketId) -> Result<usize, Error> {
        self.sockets
            .iter()
            .position(|s| s.id == id && s.closing.is_none())
            .ok_or(Error::BadSocket)
    }
    fn accounted(&self) -> usize {
        self.sockets.len()
            + self
                .sockets
                .iter()
                .map(|s| {
                    if let Mode::Listening(n) = s.mode {
                        n
                    } else {
                        0
                    }
                })
                .sum::<usize>()
    }
    fn allocate(
        &mut self,
        core: CoreSocket,
        extra: SocketExtra,
        buffers: Option<ClientBuffers>,
        mode: Mode,
        remote: Option<Endpoint>,
    ) -> Result<SocketId, Error> {
        let id = SocketId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or(Error::NoResources)?;
        self.sockets.push(Socket {
            id,
            core,
            extra,
            buffers,
            mode,
            remote,
            error: None,
            send_closed: false,
            receive_closed: false,
            closing: None,
        });
        Ok(id)
    }
    pub fn create(&mut self) -> Result<SocketId, Error> {
        if self.accounted() >= MAX_SOCKETS || self.bindings.failed() {
            return Err(Error::NoResources);
        }
        let extra = SocketExtra::default();
        let core = self.api().tcp::<Ipv4>().create(extra.clone());
        self.allocate(core, extra, None, Mode::New, None)
    }
    pub fn bind(&mut self, id: SocketId, address: Endpoint) -> Result<(), Error> {
        let i = self.index(id)?;
        if self.sockets[i].mode != Mode::New {
            return Err(Error::WrongState);
        }
        if address.address != [0; 4] && !self.is_local(address.address) {
            return Err(Error::AddressUnavailable);
        }
        self.ensure_local_alias(address.address)?;
        let core = self.sockets[i].core.clone();
        let addr = SpecifiedAddr::new(Ipv4Addr::new(address.address)).map(ZonedAddr::Unzoned);
        self.api()
            .tcp::<Ipv4>()
            .bind(&core, addr, NonZeroU16::new(address.port))
            .map_err(bind_error)?;
        self.sockets[i].mode = Mode::Bound;
        Ok(())
    }
    pub fn listen(&mut self, id: SocketId, backlog: usize) -> Result<(), Error> {
        let i = self.index(id)?;
        let backlog = backlog.clamp(1, MAX_BACKLOG);
        if self.sockets[i].mode == Mode::New {
            self.bind(id, Endpoint::default())?;
        }
        if self.sockets[i].mode != Mode::Bound {
            return Err(Error::WrongState);
        }
        if self.accounted() + backlog > MAX_SOCKETS {
            return Err(Error::NoResources);
        }
        let core = self.sockets[i].core.clone();
        self.api()
            .tcp::<Ipv4>()
            .listen(&core, NonZeroUsize::new(backlog).unwrap())
            .map_err(|e| match e {
                ListenError::ListenerExists => Error::AddressInUse,
                _ => Error::WrongState,
            })?;
        self.sockets[i].mode = Mode::Listening(backlog);
        Ok(())
    }
    pub fn connect(&mut self, id: SocketId, remote: Endpoint) -> Result<(), Error> {
        let i = self.index(id)?;
        if remote.port == 0 || remote.address == [0; 4] {
            return Err(Error::Invalid);
        }
        if !self.link && !self.is_local(remote.address) {
            return Err(Error::NoRoute);
        }
        if matches!(self.sockets[i].mode, Mode::Listening(_)) {
            return Err(Error::WrongState);
        }
        if let Some(old) = self.sockets[i].remote {
            if old != remote {
                return Err(Error::Invalid);
            }
        }
        self.refresh(i);
        if let Some(error) = self.sockets[i].error {
            return Err(error);
        }
        if self.sockets[i].mode == Mode::Connected {
            return Ok(());
        }
        self.ensure_local_alias(remote.address)?;
        // Preserve the conventional loopback source when adding a 127/8 alias.
        if remote.address[0] == 127 && self.sockets[i].mode == Mode::New {
            self.bind(
                id,
                Endpoint {
                    address: [127, 0, 0, 1],
                    port: 0,
                },
            )?;
        }
        let core = self.sockets[i].core.clone();
        let result = self.api().tcp::<Ipv4>().connect(
            &core,
            Some(ZonedAddr::Unzoned(
                SpecifiedAddr::new(Ipv4Addr::new(remote.address)).ok_or(Error::Invalid)?,
            )),
            NonZeroU16::new(remote.port).unwrap(),
        );
        self.sockets[i].remote = Some(remote);
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
            Err(error) => {
                let error = connect_error(error);
                self.sockets[i].error = Some(error);
                Err(error)
            }
        }
    }
    pub fn accept(&mut self, id: SocketId) -> Result<(SocketId, Endpoint), Error> {
        let i = self.index(id)?;
        if !matches!(self.sockets[i].mode, Mode::Listening(_)) {
            return Err(Error::WrongState);
        }
        if self.accounted() >= MAX_SOCKETS {
            return Err(Error::NoResources);
        }
        let core = self.sockets[i].core.clone();
        let (child, peer, buffers) =
            self.api()
                .tcp::<Ipv4>()
                .accept(&core)
                .map_err(|e| match e {
                    AcceptError::WouldBlock => Error::WouldBlock,
                    _ => Error::WrongState,
                })?;
        let remote = Endpoint {
            address: peer.ip.addr().ipv4_bytes(),
            port: peer.port.get(),
        };
        if !self.link && !self.is_local(remote.address) {
            let cookie = child.socket_cookie().export_value();
            self.api()
                .tcp::<Ipv4>()
                .disconnect_bound(&IpSocketMatcher::<()>::Cookie(SocketCookieMatcher {
                    cookie,
                    invert: false,
                }));
            self.api().tcp::<Ipv4>().close(child);
            return Err(Error::NoRoute);
        }
        let child = self.allocate(
            child,
            SocketExtra::default(),
            Some(buffers),
            Mode::Connected,
            Some(remote),
        )?;
        Ok((child, remote))
    }
    fn attach_buffers(&mut self, i: usize) {
        if self.sockets[i].buffers.is_none() {
            self.sockets[i].buffers = self.sockets[i].extra.take();
        }
    }
    fn refresh(&mut self, i: usize) {
        let core = self.sockets[i].core.clone();
        if let Some(error) = self.api().tcp::<Ipv4>().get_socket_error(&core) {
            if self.sockets[i].error.is_none() {
                self.sockets[i].error = Some(connection_error(error));
            }
        }
        let state = self.api().tcp::<Ipv4>().get_tcp_info(&core).state;
        if self.sockets[i].mode == Mode::Connecting && connected(state) {
            self.sockets[i].mode = Mode::Connected;
            self.attach_buffers(i);
        }
    }
    pub fn read(
        &mut self,
        id: SocketId,
        destination: &mut [u8],
        peek: bool,
    ) -> Result<usize, Error> {
        let i = self.index(id)?;
        self.refresh(i);
        if destination.is_empty() {
            return Ok(0);
        }
        if self.sockets[i].receive_closed {
            return Ok(0);
        }
        if let Some(buffers) = &self.sockets[i].buffers {
            let count = if peek {
                buffers.peek(destination)
            } else {
                buffers.read(destination)
            };
            if count > 0 {
                if !peek {
                    let core = self.sockets[i].core.clone();
                    self.api().tcp::<Ipv4>().on_receive_buffer_read(&core);
                }
                return Ok(count);
            }
        }
        if let Some(error) = self.sockets[i].error {
            return Err(error);
        }
        let core = self.sockets[i].core.clone();
        let state = self.api().tcp::<Ipv4>().get_tcp_info(&core).state;
        if eof(state) && matches!(self.sockets[i].mode, Mode::Connected) {
            Ok(0)
        } else if matches!(
            self.sockets[i].mode,
            Mode::New | Mode::Bound | Mode::Listening(_)
        ) {
            Err(Error::NotConnected)
        } else {
            Err(Error::WouldBlock)
        }
    }
    pub fn write(&mut self, id: SocketId, source: &[u8]) -> Result<usize, Error> {
        let i = self.index(id)?;
        self.refresh(i);
        if self.sockets[i].send_closed {
            return Err(Error::BrokenPipe);
        }
        if let Some(error) = self.sockets[i].error {
            return Err(error);
        }
        if self.sockets[i].mode != Mode::Connected {
            return Err(Error::NotConnected);
        }
        if source.is_empty() {
            return Ok(0);
        }
        let core = self.sockets[i].core.clone();
        let count = self
            .api()
            .tcp::<Ipv4>()
            .with_send_buffer(&core, |buffer| buffer.enqueue(source))
            .ok_or(Error::BrokenPipe)?;
        if count == 0 {
            return Err(Error::WouldBlock);
        }
        self.api().tcp::<Ipv4>().do_send(&core);
        Ok(count)
    }
    pub fn shutdown(&mut self, id: SocketId, direction: Shutdown) -> Result<(), Error> {
        let i = self.index(id)?;
        self.refresh(i);
        if self.sockets[i].mode != Mode::Connected {
            return Err(Error::NotConnected);
        }
        let core = self.sockets[i].core.clone();
        let operation = match direction {
            Shutdown::Read => ShutdownType::Receive,
            Shutdown::Write => ShutdownType::Send,
            Shutdown::Both => ShutdownType::SendAndReceive,
        };
        self.api()
            .tcp::<Ipv4>()
            .shutdown(&core, operation)
            .map_err(|_| Error::NotConnected)?;
        if matches!(direction, Shutdown::Read | Shutdown::Both) {
            self.sockets[i].receive_closed = true;
        }
        if matches!(direction, Shutdown::Write | Shutdown::Both) {
            self.sockets[i].send_closed = true;
        }
        Ok(())
    }
    pub fn readiness(&mut self, id: SocketId) -> Result<Readiness, Error> {
        let i = self.index(id)?;
        self.refresh(i);
        let socket = &self.sockets[i];
        if let Mode::Listening(_) = socket.mode {
            return Ok(Readiness {
                readable: socket.extra.accept_ready(),
                accept: socket.extra.accept_ready(),
                error: socket.error.is_some(),
                ..Default::default()
            });
        }
        let buffered = socket.buffers.as_ref().map_or(0, ClientBuffers::readable);
        let core = socket.core.clone();
        let state = self.api().tcp::<Ipv4>().get_tcp_info(&core).state;
        let writable = self
            .api()
            .tcp::<Ipv4>()
            .with_send_buffer(&core, |buffer| {
                use netstack3_tcp::Buffer as _;
                let x = buffer.limits();
                x.len < x.capacity
            })
            .unwrap_or(false);
        let socket = &self.sockets[i];
        let is_connected = socket.mode == Mode::Connected;
        Ok(Readiness {
            readable: buffered > 0 || (is_connected && (eof(state) || socket.receive_closed)),
            writable: (is_connected && !socket.send_closed && writable) || socket.error.is_some(),
            accept: false,
            connected: is_connected,
            hangup: matches!(socket.mode, Mode::Connected | Mode::Connecting)
                && state == TcpSocketState::Close,
            error: socket.error.is_some(),
        })
    }
    pub fn local_addr(&mut self, id: SocketId) -> Result<Endpoint, Error> {
        let i = self.index(id)?;
        let core = self.sockets[i].core.clone();
        Ok(match self.api().tcp::<Ipv4>().get_info(&core) {
            SocketInfo::Unbound(_) => Endpoint::default(),
            SocketInfo::Bound(info) => Endpoint {
                address: info.addr.map_or([0; 4], |ip| ip.addr().ipv4_bytes()),
                port: info.port.get(),
            },
            SocketInfo::Connection(info) => Endpoint {
                address: info.local_addr.ip.addr().ipv4_bytes(),
                port: info.local_addr.port.get(),
            },
        })
    }
    pub fn peer_addr(&mut self, id: SocketId) -> Result<Endpoint, Error> {
        let i = self.index(id)?;
        self.sockets[i].remote.ok_or(Error::NotConnected)
    }
    pub fn peek_error(&mut self, id: SocketId) -> Result<Option<Error>, Error> {
        let i = self.index(id)?;
        self.refresh(i);
        Ok(self.sockets[i].error)
    }
    pub fn take_error(&mut self, id: SocketId) -> Result<Option<Error>, Error> {
        let i = self.index(id)?;
        self.refresh(i);
        Ok(self.sockets[i].error.take())
    }
    pub fn close(&mut self, id: SocketId) -> Result<(), Error> {
        let i = self.index(id)?;
        if self.sockets[i].mode == Mode::Connecting {
            self.force(i);
            self.remove(i);
            return Ok(());
        }
        if matches!(
            self.sockets[i].mode,
            Mode::New | Mode::Bound | Mode::Listening(_)
        ) {
            self.remove(i);
            return Ok(());
        }
        let core = self.sockets[i].core.clone();
        let _ = self.api().tcp::<Ipv4>().shutdown(&core, ShutdownType::Send);
        self.sockets[i].send_closed = true;
        self.sockets[i].closing = Some(self.now.saturating_add(ABANDON_NS));
        Ok(())
    }
    pub fn abort(&mut self, id: SocketId) -> Result<(), Error> {
        let i = self.index(id)?;
        self.force(i);
        self.remove(i);
        Ok(())
    }
    fn force(&mut self, i: usize) {
        let cookie = self.sockets[i].core.socket_cookie().export_value();
        self.api()
            .tcp::<Ipv4>()
            .disconnect_bound(&IpSocketMatcher::<()>::Cookie(SocketCookieMatcher {
                cookie,
                invert: false,
            }));
    }
    fn remove(&mut self, i: usize) {
        let socket = self.sockets.swap_remove(i);
        self.api().tcp::<Ipv4>().close(socket.core);
    }
    pub fn set_link(&mut self, up: bool) {
        if self.link && !up {
            for i in 0..self.sockets.len() {
                if self.sockets[i]
                    .remote
                    .is_some_and(|r| !self.is_local(r.address))
                {
                    self.force(i);
                    self.sockets[i].error = Some(Error::NoRoute);
                }
            }
        }
        self.link = up;
    }
    pub fn is_local(&self, address: [u8; 4]) -> bool {
        address[0] == 127 || address == self.config.address
    }
    fn ensure_local_alias(&mut self, address: [u8; 4]) -> Result<(), Error> {
        if address[0] != 127 || address == [127, 0, 0, 1] || self.aliases.contains(&address) {
            return Ok(());
        }
        self.prune_aliases();
        if self.aliases.len() >= MAX_SOCKETS {
            return Err(Error::NoResources);
        }
        self.aliases
            .try_reserve(1)
            .map_err(|_| Error::NoResources)?;
        let device = self.loopback.as_ref().unwrap().clone().into();
        self.api()
            .device_ip::<Ipv4>()
            .add_ip_addr_subnet(
                &device,
                AddrSubnet::new(Ipv4Addr::new(address), 32).map_err(|_| Error::Invalid)?,
            )
            .map_err(|_| Error::AddressUnavailable)?;
        self.aliases.push(address);
        Ok(())
    }
    fn prune_aliases(&mut self) {
        let mut n = 0;
        while n < self.aliases.len() {
            let alias = self.aliases[n];
            let mut used = false;
            for i in 0..self.sockets.len() {
                let core = self.sockets[i].core.clone();
                let local = match self.api().tcp::<Ipv4>().get_info(&core) {
                    SocketInfo::Bound(x) => x.addr.map(|x| x.addr().ipv4_bytes()),
                    SocketInfo::Connection(x) => Some(x.local_addr.ip.addr().ipv4_bytes()),
                    _ => None,
                };
                if local == Some(alias)
                    || self.sockets[i].remote.is_some_and(|x| x.address == alias)
                {
                    used = true;
                    break;
                }
            }
            if used {
                n += 1
            } else {
                let device = self.loopback.as_ref().unwrap().clone().into();
                let _ = self
                    .api()
                    .device_ip::<Ipv4>()
                    .del_ip_addr(&device, SpecifiedAddr::new(Ipv4Addr::new(alias)).unwrap());
                self.aliases.swap_remove(n);
            }
        }
    }
    pub fn receive_ethernet(&mut self, frame: &[u8]) -> Result<(), Error> {
        if !self.link
            || frame.len() < 14
            || frame.len() > self.config.mtu as usize + 18
            || frame.len() > 2048
        {
            return Err(Error::Invalid);
        }
        // No IP fragment retention in v1. TCP uses the configured path MTU.
        if frame[12..14] == [8, 0]
            && (frame.len() < 34 || (u16::from_be_bytes([frame[20], frame[21]]) & 0x3fff) != 0)
        {
            return Err(Error::Invalid);
        }
        if frame[12..14] != [8, 0] && frame[12..14] != [8, 6] {
            return Err(Error::Invalid);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(frame.len())
            .map_err(|_| Error::NoResources)?;
        bytes.extend_from_slice(frame);
        let device = self.ethernet.as_ref().unwrap().clone();
        self.api().device::<EthernetLinkDevice>().receive_frame(
            RecvEthernetFrameMeta {
                device_id: device,
                parsing_context: Default::default(),
                gso_info: None,
            },
            Buf::new(bytes, ..),
        );
        Ok(())
    }
    /// Bounded metadata diagnostics; no client identifiers or packet payloads.
    pub fn resources(&mut self) -> Resources {
        let mut resources = Resources::default();
        self.api().filter().inspect_state(&mut ResourceInspector {
            result: &mut resources,
            ipv4: false,
            conntrack: false,
        });
        resources.queued_frames = self.bindings.frame_count();
        resources.scheduled_timers = self.bindings.timer_count();
        resources
    }
    pub fn take_ethernet(&mut self) -> Option<Vec<u8>> {
        self.bindings.pop_frame().map(|f| f.0)
    }
    pub fn next_deadline(&self) -> Option<u64> {
        self.bindings
            .next_deadline()
            .map(|x| x.0)
            .into_iter()
            .chain(self.sockets.iter().filter_map(|s| s.closing))
            .min()
    }
    pub fn pump(&mut self, now_ns: u64) -> Pump {
        self.now = self.now.max(now_ns);
        self.bindings.set_now(Instant(self.now));
        let mut timer_work = 0;
        while timer_work < TIMER_BATCH {
            let Some((dispatch, unique)) = self.bindings.take_due_timer() else {
                break;
            };
            self.api().handle_timer(dispatch, unique);
            timer_work += 1;
        }
        self.loopback_pending |= self.bindings.take_rx_pending();
        if self.loopback_pending {
            for _ in 0..LOOPBACK_BATCH {
                let id = self.loopback.as_ref().unwrap().clone();
                match self.api().receive_queue().handle_queued_frames(&id) {
                    WorkQueueReport::AllDone => {
                        self.loopback_pending = false;
                        break;
                    }
                    WorkQueueReport::Pending => {}
                }
            }
        }
        self.loopback_pending |= self.bindings.take_rx_pending();
        let mut i = 0;
        while i < self.sockets.len() {
            self.refresh(i);
            if let Some(deadline) = self.sockets[i].closing {
                let core = self.sockets[i].core.clone();
                let state = self.api().tcp::<Ipv4>().get_tcp_info(&core).state;
                if state == TcpSocketState::Close || deadline <= self.now {
                    if state != TcpSocketState::Close {
                        self.force(i)
                    }
                    self.remove(i);
                    continue;
                }
            }
            i += 1;
        }
        self.prune_aliases();
        Pump {
            more_work: self.loopback_pending
                || self
                    .bindings
                    .next_deadline()
                    .is_some_and(|x| x.0 <= self.now),
            failed: self.bindings.failed(),
        }
    }
}
fn route(
    subnet: Subnet<Ipv4Addr>,
    device: DeviceId<Bindings>,
    gateway: Option<SpecifiedAddr<Ipv4Addr>>,
    metric: u32,
) -> Entry<Ipv4Addr, DeviceId<Bindings>> {
    Entry {
        subnet,
        device,
        gateway,
        metric: Metric::ExplicitMetric(RawMetric(metric)),
        route_preference: Default::default(),
    }
}
fn connected(state: TcpSocketState) -> bool {
    matches!(
        state,
        TcpSocketState::Established
            | TcpSocketState::FinWait1
            | TcpSocketState::FinWait2
            | TcpSocketState::CloseWait
            | TcpSocketState::Closing
            | TcpSocketState::LastAck
            | TcpSocketState::TimeWait
    )
}
fn eof(state: TcpSocketState) -> bool {
    matches!(
        state,
        TcpSocketState::CloseWait
            | TcpSocketState::Closing
            | TcpSocketState::LastAck
            | TcpSocketState::TimeWait
            | TcpSocketState::Close
    )
}
fn bind_error(error: BindError) -> Error {
    match error {
        BindError::AlreadyBound => Error::WrongState,
        BindError::LocalAddressError(netstack3_core::error::LocalAddressError::AddressInUse) => {
            Error::AddressInUse
        }
        BindError::LocalAddressError(
            netstack3_core::error::LocalAddressError::FailedToAllocateLocalPort,
        ) => Error::NoResources,
        _ => Error::AddressUnavailable,
    }
}
fn connect_error(error: ConnectError) -> Error {
    match error {
        ConnectError::NoPort => Error::NoResources,
        ConnectError::NoRoute => Error::NoRoute,
        ConnectError::Pending => Error::WouldBlock,
        ConnectError::ConnectionExists => Error::AddressInUse,
        ConnectError::ConnectionError(error) => connection_error(error),
        ConnectError::Aborted => Error::Refused,
        _ => Error::WrongState,
    }
}
fn connection_error(error: ConnectionError) -> Error {
    match error {
        ConnectionError::ConnectionRefused => Error::Refused,
        ConnectionError::ConnectionReset | ConnectionError::Aborted => Error::Reset,
        ConnectionError::TimedOut => Error::TimedOut,
        _ => Error::NoRoute,
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        while !self.sockets.is_empty() {
            self.force(0);
            self.remove(0);
        }
        let table = self.api().routes::<Ipv4>().main_table_id();
        self.api().routes::<Ipv4>().set_routes(&table, Vec::new());
        drop(table);
        if let Some(device) = self.ethernet.take() {
            let _ = self
                .api()
                .device::<EthernetLinkDevice>()
                .remove_device(device);
        }
        if let Some(device) = self.loopback.take() {
            let _ = self.api().device::<LoopbackDevice>().remove_device(device);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Resources {
    pub conntrack_entries: usize,
    pub conntrack_capacity: usize,
    pub conntrack_limit_hits: u64,
    pub conntrack_limit_drops: u64,
    pub connection_object_bytes: usize,
    pub connection_alignment: usize,
    pub tuple_bytes: usize,
    pub queued_frames: usize,
    pub scheduled_timers: usize,
}
struct ResourceInspector<'a> {
    result: &'a mut Resources,
    ipv4: bool,
    conntrack: bool,
}
impl netstack3_core::inspect::Inspector for ResourceInspector<'_> {
    type ChildInspector<'a> = ResourceInspector<'a>;
    fn record_child<F: FnOnce(&mut Self::ChildInspector<'_>)>(&mut self, name: &str, f: F) {
        if self.conntrack && name == "connections" {
            return;
        }
        let ipv4 = if name == "IPv4" {
            true
        } else if name == "IPv6" {
            false
        } else {
            self.ipv4
        };
        let conntrack = self.conntrack || name == "conntrack";
        f(&mut ResourceInspector {
            result: self.result,
            ipv4,
            conntrack,
        });
    }
    fn record_unnamed_child<F: FnOnce(&mut Self::ChildInspector<'_>)>(&mut self, _f: F) {}
    fn record_usize<T: Into<usize>>(&mut self, name: &str, value: T) {
        if !self.ipv4 || !self.conntrack {
            return;
        }
        let value = value.into();
        match name {
            "num_entries" => self.result.conntrack_entries = value,
            "table_capacity" => self.result.conntrack_capacity = value,
            "connection_object_bytes" => self.result.connection_object_bytes = value,
            "connection_alignment" => self.result.connection_alignment = value,
            "tuple_bytes" => self.result.tuple_bytes = value,
            _ => {}
        }
    }
    fn record_uint<T: Into<u64>>(&mut self, name: &str, value: T) {
        if !self.ipv4 || !self.conntrack {
            return;
        }
        match name {
            "table_limit_hits" => self.result.conntrack_limit_hits = value.into(),
            "table_limit_drops" => self.result.conntrack_limit_drops = value.into(),
            _ => {}
        }
    }
    fn record_int<T: Into<i64>>(&mut self, _: &str, _: T) {}
    fn record_double<T: Into<f64>>(&mut self, _: &str, _: T) {}
    fn record_str(&mut self, _: &str, _: &str) {}
    fn record_string(&mut self, _: &str, _: String) {}
    fn record_bool(&mut self, _: &str, _: bool) {}
}
