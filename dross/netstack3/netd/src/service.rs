use crate::{ethernet::Ethernet, seed::Seed};
use alloc::vec::Vec;
use dross_rt::{
    Error, Handle,
    deadline::Deadline,
    owned::{self, EventQueue, Message, OwnedHandle},
    sys,
};
use oscar_netstack3_engine::{Engine, Error as NetError, IpEndpoint, SocketId};

pub const MAX_CONNECTIONS: usize = 32;
pub const CHANNEL_RIGHTS: u32 = dross_abi::rights::READ
    | dross_abi::rights::WRITE
    | dross_abi::rights::WAIT
    | dross_abi::rights::TRANSFER;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    New,
    Stream,
    Listen,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Legacy(Phase),
    Stream { rights: u32 },
    Listener { rights: u32, accepted: u32 },
    Datagram { rights: u32 },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Snapshot {
    pub signals: u32,
    pub error: i32,
    pub local: Option<IpEndpoint>,
    pub peer: Option<IpEndpoint>,
}
pub struct Connection {
    pub endpoint: OwnedHandle,
    pub socket: SocketId,
    pub role: Role,
    pub reply: Option<Message>,
    pub error: Option<NetError>,
    pub connecting: bool,
    pub connect_deadline: Option<u64>,
    pub revision: u64,
    pub snapshot: Snapshot,
    pub observer: Option<crate::native::Observer>,
    pub dead: bool,
    pub engine_alive: bool,
    pub retire_after_reply: bool,
    pub last_id: u64,
    pub wait: Option<crate::wait::Registration>,
}
impl Connection {
    pub fn new(
        endpoint: OwnedHandle,
        socket: SocketId,
        role: Role,
        wait: crate::wait::Registration,
    ) -> Self {
        Self {
            endpoint,
            socket,
            role,
            reply: None,
            error: None,
            connecting: false,
            connect_deadline: None,
            revision: 1,
            snapshot: Snapshot::default(),
            observer: None,
            dead: false,
            engine_alive: true,
            retire_after_reply: false,
            last_id: 0,
            wait: Some(wait),
        }
    }
    pub fn refresh(&mut self, engine: &mut Engine) {
        if !self.engine_alive {
            return;
        }
        let readiness = match engine.readiness(self.socket) {
            Ok(r) => r,
            Err(_) => {
                self.dead = true;
                return;
            }
        };
        if readiness.error {
            if let Ok(Some(e)) = engine.peek_error(self.socket) {
                if self.error.is_none() {
                    self.error = Some(e);
                }
            }
        }
        if readiness.connected || readiness.error || readiness.hangup {
            self.connecting = false;
            self.connect_deadline = None;
        }
        let mut signals = 0;
        if matches!(
            self.role,
            Role::Listener { .. } | Role::Legacy(Phase::Listen)
        ) {
            if readiness.accept {
                signals |= 8
            }
        } else {
            if readiness.readable {
                signals |= 1
            }
            if readiness.writable {
                signals |= 2
            }
            if readiness.connected && !matches!(self.role, Role::Datagram { .. }) {
                signals |= 4
            }
        }
        if readiness.hangup {
            signals |= 16
        }
        if self.error.is_some() {
            signals |= 32
        }
        let snapshot = Snapshot {
            signals,
            error: self.error.map(crate::native::status).unwrap_or(0),
            local: engine.local_addr_ip(self.socket).ok(),
            peer: engine.peer_addr_ip(self.socket).ok(),
        };
        if snapshot != self.snapshot {
            self.snapshot = snapshot;
            match self.revision.checked_add(1) {
                Some(r) => self.revision = r,
                None => self.dead = true,
            }
        }
    }
    pub fn expire_connect(&mut self, engine: &mut Engine, now: u64) {
        if self.connecting && self.connect_deadline.is_some_and(|d| now >= d) {
            self.refresh(engine);
            if !self.connecting {
                return;
            }
            self.error = Some(NetError::TimedOut);
            let _ = engine.abort(self.socket);
            self.engine_alive = false;
            self.connecting = false;
            self.connect_deadline = None;
            self.snapshot.signals = 16 | 32;
            self.snapshot.error = crate::native::status(NetError::TimedOut);
            match self.revision.checked_add(1) {
                Some(r) => self.revision = r,
                None => self.dead = true,
            }
        }
    }
    pub fn flush_reply(&mut self) -> bool {
        if let Some(m) = self.reply.take() {
            match owned::try_send(self.endpoint.as_handle(), m) {
                Ok(()) => {}
                Err(e) if e.error == Error::ShouldWait => {
                    self.reply = Some(e.message);
                    return false;
                }
                Err(_) => {
                    self.dead = true;
                    return false;
                }
            }
        }
        if self.retire_after_reply {
            self.dead = true;
            return false;
        }
        true
    }
}
pub struct Factory {
    pub endpoint: OwnedHandle,
    pub rights: u32,
    pub reply: Option<Message>,
    pub last_id: u64,
    pub dead: bool,
    pub wait: Option<crate::wait::Registration>,
}
pub struct Daemon {
    pub engine: Engine,
    pub connections: Vec<Connection>,
    pub factories: Vec<Factory>,
    pub resolver: Option<oscar_resolver::Resolver>,
    pub resolver_factory: Option<Factory>,
    pub queries: Vec<crate::query::Query>,
    pub next_observer: u64,
    pub processing: bool,
    // Last field: endpoint registrations drop before this owned queue.
    pub queue: EventQueue,
}
impl Daemon {
    pub fn admit(&mut self, connection: Connection) -> Result<(), Error> {
        if self.connections.len() + usize::from(self.processing) >= MAX_CONNECTIONS {
            return Err(Error::NoResources);
        }
        if connection.wait.is_none() {
            return Err(Error::InvalidArgs);
        }
        self.connections.push(connection);
        Ok(())
    }
    pub fn observer_id(&mut self) -> Result<u64, Error> {
        let id = self.next_observer;
        self.next_observer = self
            .next_observer
            .checked_add(1)
            .ok_or(Error::NoResources)?;
        Ok(id)
    }
}
pub fn message_with_space(length: usize) -> Result<Message, Error> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| Error::NoMemory)?;
    bytes.resize(length, 0);
    Message::try_from_bytes(&bytes)
}
fn ipv4(s: &str) -> Result<[u8; 4], Error> {
    let mut result = [0; 4];
    let mut parts = s.split('.');
    for v in &mut result {
        *v = parts
            .next()
            .ok_or(Error::InvalidArgs)?
            .parse()
            .map_err(|_| Error::InvalidArgs)?;
    }
    if parts.next().is_some() {
        return Err(Error::InvalidArgs);
    }
    Ok(result)
}
fn ipv6(s: &str) -> Result<[u8; 16], Error> {
    s.parse::<core::net::Ipv6Addr>()
        .map(|ip| ip.octets())
        .map_err(|_| Error::InvalidArgs)
}
pub fn run(boot: Handle) -> Result<(), Error> {
    let mut seed = Seed::read(boot)?;
    let ethernet = seed.take("ethernet").ok_or(Error::BadHandle)?;
    let ready = seed.take("ready").ok_or(Error::BadHandle)?;
    let mut legacy = seed.take("service");
    let mut nic = Ethernet::new(ethernet)?;
    let address = ipv4(seed.value("ip").unwrap_or("10.0.2.15"))?;
    let prefix = seed
        .value("prefix")
        .unwrap_or("24")
        .parse::<u8>()
        .map_err(|_| Error::InvalidArgs)?;
    if prefix > 32 {
        return Err(Error::InvalidArgs);
    }
    let gateway = seed
        .value("gateway")
        .or_else(|| {
            if seed.value("ip").is_none() {
                Some("10.0.2.2")
            } else {
                None
            }
        })
        .map(ipv4)
        .transpose()?;
    let mut engine = Engine::new(
        oscar_netstack3_engine::Config {
            mac: nic.info.mac,
            address,
            prefix,
            gateway,
            mtu: nic.info.mtu as u32,
        },
        sys::clock(),
    )
    .map_err(|_| Error::NoResources)?;
    let address6 = seed.value("ip6").map(ipv6).transpose()?;
    let prefix6 = seed
        .value("prefix6")
        .unwrap_or("64")
        .parse::<u8>()
        .map_err(|_| Error::InvalidArgs)?;
    if prefix6 > 128 {
        return Err(Error::InvalidArgs);
    }
    let gateway6 = seed.value("gateway6").map(ipv6).transpose()?;
    engine
        .configure_ipv6(oscar_netstack3_engine::Ipv6Config {
            address: address6,
            prefix: prefix6,
            gateway: gateway6,
        })
        .map_err(|_| Error::InvalidArgs)?;
    engine.set_link(nic.info.link.up);
    let queue = EventQueue::create(128)?;
    ipv6_startup(&mut engine, &mut nic, &queue)?;
    let mut connections = Vec::new();
    connections
        .try_reserve_exact(MAX_CONNECTIONS)
        .map_err(|_| Error::NoMemory)?;
    let mut factories = Vec::new();
    factories
        .try_reserve_exact(4)
        .map_err(|_| Error::NoMemory)?;
    for (name, rights) in [
        ("native/connect", 1),
        ("native/listen", 2),
        ("native/network", 3),
        ("native/datagram", 4),
    ] {
        if let Some(endpoint) = seed.take(name) {
            let wait = crate::wait::Registration::new(
                queue.as_handle().as_raw(),
                endpoint.as_handle(),
                dross_abi::event::READABLE | dross_abi::event::PEER_CLOSED,
            )?;
            factories.push(Factory {
                endpoint,
                rights,
                reply: None,
                last_id: 0,
                dead: false,
                wait: Some(wait),
            });
        }
    }
    let resolver_endpoint = seed.take("native/resolver");
    let resolver = if resolver_endpoint.is_some() {
        let literal = seed.value("dns-server").ok_or(Error::InvalidArgs)?;
        let port = seed
            .value("dns-port")
            .unwrap_or("53")
            .parse::<u16>()
            .map_err(|_| Error::InvalidArgs)?;
        let scope = seed
            .value("dns-scope")
            .unwrap_or("0")
            .parse::<u32>()
            .map_err(|_| Error::InvalidArgs)?;
        let server = if literal.contains(':') {
            IpEndpoint::V6(oscar_netstack3_engine::Endpoint6 {
                address: ipv6(literal)?,
                port,
                scope_id: scope,
            })
        } else {
            if scope != 0 {
                return Err(Error::InvalidArgs);
            }
            IpEndpoint::V4(oscar_netstack3_engine::Endpoint {
                address: ipv4(literal)?,
                port,
            })
        };
        Some(oscar_resolver::Resolver::new(server).map_err(|_| Error::InvalidArgs)?)
    } else {
        None
    };
    let resolver_factory = resolver_endpoint
        .map(|endpoint| {
            let wait = crate::wait::Registration::new(
                queue.as_handle().as_raw(),
                endpoint.as_handle(),
                dross_abi::event::READABLE | dross_abi::event::PEER_CLOSED,
            )?;
            Ok::<_, Error>(Factory {
                endpoint,
                rights: 1,
                reply: None,
                last_id: 0,
                dead: false,
                wait: Some(wait),
            })
        })
        .transpose()?;
    let mut queries = Vec::new();
    queries
        .try_reserve_exact(oscar_resolver::MAX_JOBS)
        .map_err(|_| Error::NoMemory)?;
    if legacy.is_none() && factories.is_empty() && resolver_factory.is_none() {
        return Err(Error::BadHandle);
    }
    if seed.remaining() != 0 {
        return Err(Error::AccessDenied);
    }
    drop(seed);
    let mut daemon = Daemon {
        engine,
        connections,
        factories,
        resolver,
        resolver_factory,
        queries,
        next_observer: 1,
        processing: false,
        queue,
    };
    let mut legacy_wait = None;
    let mut nic_wait = None;
    let mut delivered = None;
    if let Some(root) = &legacy {
        crate::wait::Registration::update(
            &mut legacy_wait,
            daemon.queue.as_handle().as_raw(),
            root.as_handle(),
            dross_abi::event::READABLE | dross_abi::event::PEER_CLOSED,
            None,
        )?;
    }
    nic.wait(&mut nic_wait, daemon.queue.as_handle().as_raw(), None)?;
    sys::channel_write(ready.as_handle().as_raw(), b"ready", &[])?;
    drop(ready);
    sys::log("netd: Netstack3 ready");
    let mut reported_dad_failure = false;
    loop {
        let mut more = nic.service(&mut daemon.engine)?;
        let progress = daemon.engine.pump(sys::clock());
        if progress.failed {
            return Err(Error::NoResources);
        }
        more |= progress.more_work;
        if let Some(resolver) = &mut daemon.resolver {
            more |= resolver.pump(sys::clock(), &mut daemon.engine);
        }
        if let Some(root) = legacy.as_ref() {
            for batch in 0..8 {
                match owned::receive(root.as_handle(), Deadline::IMMEDIATE) {
                    Ok(m) => {
                        crate::legacy::admit(&mut daemon, m)?;
                        if batch == 7 {
                            more = true;
                        }
                    }
                    Err(Error::ShouldWait | Error::TimedOut) => break,
                    Err(Error::PeerClosed) => {
                        legacy.take();
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        crate::native::factories(&mut daemon)?;
        crate::query::process(&mut daemon)?;
        let count = daemon.connections.len();
        for _ in 0..count {
            let mut c = daemon.connections.remove(0);
            daemon.processing = true;
            c.expire_connect(&mut daemon.engine, sys::clock());
            c.refresh(&mut daemon.engine);
            crate::native::notify(&mut c);
            if !c.dead && c.flush_reply() {
                match owned::receive(c.endpoint.as_handle(), Deadline::IMMEDIATE) {
                    Ok(m) => match c.role {
                        Role::Legacy(_) => crate::legacy::handle(&mut daemon, &mut c, m),
                        _ => crate::native::handle(&mut daemon, &mut c, m),
                    },
                    Err(Error::ShouldWait | Error::TimedOut) => {}
                    Err(_) => c.dead = true,
                }
            }
            daemon.processing = false;
            if !c.dead {
                daemon.connections.push(c);
            } else {
                if c.engine_alive {
                    let _ = daemon.engine.close(c.socket);
                }
            }
        }
        let progress = daemon.engine.pump(sys::clock());
        if progress.failed {
            return Err(Error::NoResources);
        }
        more |= progress.more_work;
        if let Some(resolver) = &mut daemon.resolver {
            more |= resolver.pump(sys::clock(), &mut daemon.engine);
        }
        more |= nic.service(&mut daemon.engine)?;
        for c in &mut daemon.connections {
            c.expire_connect(&mut daemon.engine, sys::clock());
            c.refresh(&mut daemon.engine);
            crate::native::notify(c);
        }
        crate::query::refresh_all(&mut daemon);
        if daemon.engine.ipv6_dad_failed() && !reported_dad_failure {
            sys::log("netd: IPv6 duplicate address detected; local and IPv4 service retained");
            reported_dad_failure = true;
        }
        if legacy.is_none()
            && daemon.factories.is_empty()
            && daemon.connections.is_empty()
            && daemon.resolver_factory.is_none()
            && daemon.queries.is_empty()
        {
            return Ok(());
        }
        let mut deadline = daemon.engine.next_deadline().unwrap_or(u64::MAX);
        if let Some(resolver) = &daemon.resolver {
            if let Some(d) = resolver.next_deadline() {
                deadline = deadline.min(d);
            }
        }
        for c in &daemon.connections {
            if let Some(d) = c.connect_deadline {
                deadline = deadline.min(d)
            }
        }
        if more {
            sys::yield_now();
            continue;
        }
        use dross_abi::event::{PEER_CLOSED, READABLE, WRITABLE};
        let queue = daemon.queue.as_handle().as_raw();
        if let Some(root) = &legacy {
            crate::wait::Registration::update(
                &mut legacy_wait,
                queue,
                root.as_handle(),
                READABLE | PEER_CLOSED,
                delivered,
            )?;
        } else {
            legacy_wait.take();
        }
        nic.wait(&mut nic_wait, queue, delivered)?;
        let mut failed_endpoint = false;
        for f in &mut daemon.factories {
            let signals = PEER_CLOSED
                | if f.reply.is_some() {
                    WRITABLE
                } else {
                    READABLE
                };
            if crate::wait::Registration::update(
                &mut f.wait,
                queue,
                f.endpoint.as_handle(),
                signals,
                delivered,
            )
            .is_err()
            {
                f.dead = true;
                failed_endpoint = true;
            }
        }
        for c in &mut daemon.connections {
            let signals = PEER_CLOSED
                | if c.reply.is_some() {
                    WRITABLE
                } else {
                    READABLE
                };
            if crate::wait::Registration::update(
                &mut c.wait,
                queue,
                c.endpoint.as_handle(),
                signals,
                delivered,
            )
            .is_err()
            {
                c.dead = true;
                failed_endpoint = true;
            }
            let bad_watch = c
                .observer
                .as_mut()
                .is_some_and(|o| o.wait(queue, delivered).is_err());
            if bad_watch {
                c.observer.take();
            }
        }
        failed_endpoint |= crate::query::wait(&mut daemon, delivered);
        if failed_endpoint {
            sys::yield_now();
            continue;
        }
        delivered = None;
        match daemon.queue.wait_until(Deadline::at_monotonic_ns(deadline)) {
            Ok(event) => delivered = Some(event.subscription),
            Err(Error::TimedOut) => {}
            Err(e) => return Err(e),
        }
    }
}

// DAD runs before registering client endpoints, so waiting here cannot consume
// and disarm a client's queued request event. Only real NIC/core deadlines wake.
fn ipv6_startup(engine: &mut Engine, nic: &mut Ethernet, queue: &EventQueue) -> Result<(), Error> {
    let deadline = sys::after_ms(5000);
    let mut wait = None;
    let mut delivered = None;
    loop {
        let mut more = nic.service(engine)?;
        let p = engine.pump(sys::clock());
        if p.failed {
            return Err(Error::NoResources);
        }
        more |= p.more_work;
        if engine.ipv6_dad_failed() {
            sys::log("netd: IPv6 duplicate address detected during startup");
            return Err(Error::InvalidArgs);
        }
        if engine.ipv6_ready() || !nic.info.link.up {
            return Ok(());
        }
        if sys::clock() >= deadline {
            return Err(Error::TimedOut);
        }
        if more {
            sys::yield_now();
            continue;
        }
        nic.wait(&mut wait, queue.as_handle().as_raw(), delivered)?;
        delivered = None;
        let wake = engine.next_deadline().unwrap_or(deadline).min(deadline);
        match queue.wait_until(Deadline::at_monotonic_ns(wake)) {
            Ok(e) => delivered = Some(e.subscription),
            Err(Error::TimedOut) => {}
            Err(e) => return Err(e),
        }
    }
}
