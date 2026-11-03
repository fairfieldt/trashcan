//! Native capability-scoped networking over the common framed protocol.
use crate::service::{
    CHANNEL_RIGHTS, Connection, Daemon, Factory, MAX_CONNECTIONS, Role, message_with_space,
};
use dross_rt::{
    Error, Handle,
    deadline::Deadline,
    owned::{self, Channel, Message, OwnedHandle},
    protocol::{
        ProtocolError,
        frame::{Frame, Header},
        network as wire,
        schema::*,
    },
    sys,
};
use oscar_netstack3_engine::{Endpoint, Error as NetError, Shutdown};
const STREAM_ALL: u32 = 127;
const LISTENER_ALL: u32 = 7;
pub fn status(e: NetError) -> i32 {
    match e {
        NetError::WouldBlock => STATUS_WOULD_BLOCK,
        NetError::Invalid => STATUS_MALFORMED,
        NetError::WrongState | NetError::BadSocket => STATUS_BAD_STATE,
        NetError::AddressInUse => STATUS_ADDRESS_IN_USE,
        NetError::AddressUnavailable | NetError::NoRoute => STATUS_UNREACHABLE,
        NetError::Refused => STATUS_REFUSED,
        NetError::Reset => STATUS_RESET,
        NetError::TimedOut => STATUS_TIMED_OUT,
        NetError::NoResources => STATUS_LIMIT,
        NetError::NotConnected => STATUS_NOT_CONNECTED,
        NetError::BrokenPipe => STATUS_CLOSED,
    }
}
fn u32at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn u64at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn reply(h: Header, body: usize, handles: usize) -> Result<Message, Error> {
    let mut m = message_with_space(32 + body)?;
    Header::new(
        h.protocol,
        1,
        KIND_REPLY,
        h.ordinal,
        h.request_id,
        body,
        handles,
    )
    .map_err(|_| Error::InvalidArgs)?
    .encode(m.bytes_mut())
    .map_err(|_| Error::InvalidArgs)?;
    Ok(m)
}
fn error(h: Header, value: i32) -> Result<Message, Error> {
    let mut m = reply(h, 4, 0)?;
    m.bytes_mut()[32..].copy_from_slice(&value.to_le_bytes());
    Ok(m)
}
fn decode_endpoint(b: &[u8]) -> Result<Endpoint, i32> {
    let endpoint = wire::Endpoint::decode(b).map_err(|_| STATUS_MALFORMED)?;
    match endpoint.address {
        wire::Address::V4(address) => Ok(Endpoint {
            address,
            port: endpoint.port,
        }),
        wire::Address::V6 { .. } => Err(STATUS_UNSUPPORTED),
    }
}
fn valid_request(m: &Message, protocol: u32, last: &mut u64) -> Result<Header, Option<Header>> {
    let f = Frame::parse(m.bytes(), m.handle_count()).map_err(|_| None)?;
    let h = f.header;
    if h.kind != KIND_REQUEST || h.request_id <= *last {
        return Err(None);
    }
    *last = h.request_id;
    if !h.supported(protocol, 1) {
        return Err(Some(h));
    }
    Ok(h)
}
fn body_error(m: &Message) -> Option<i32> {
    match wire::decode_request(m.bytes(), m.handle_count()) {
        Ok(_) => None,
        Err(ProtocolError::Unsupported) => Some(STATUS_UNSUPPORTED),
        Err(ProtocolError::Limit | ProtocolError::Allocation) => Some(STATUS_LIMIT),
        Err(_) => Some(STATUS_MALFORMED),
    }
}
fn known(protocol: u32, op: u32) -> bool {
    match protocol {
        PROTOCOL_NETWORK => op <= 2,
        PROTOCOL_TCP_STREAM => (1..=9).contains(&op),
        PROTOCOL_TCP_LISTENER => [1, 4, 5, 6, 7].contains(&op),
        _ => false,
    }
}
fn prepare_channel(
    d: &Daemon,
) -> Result<(OwnedHandle, OwnedHandle, crate::wait::Registration), i32> {
    if d.connections.len() + usize::from(d.processing) >= MAX_CONNECTIONS {
        return Err(STATUS_LIMIT);
    }
    let (server, client) = Channel::pair().map_err(|_| STATUS_LIMIT)?;
    let wait = crate::wait::Registration::new(
        d.queue.as_handle().as_raw(),
        server.as_handle().as_untyped(),
        dross_abi::event::READABLE | dross_abi::event::PEER_CLOSED,
    )
    .map_err(|_| STATUS_LIMIT)?;
    let client = client
        .try_duplicate(CHANNEL_RIGHTS)
        .map_err(|_| STATUS_LIMIT)?
        .into_untyped();
    Ok((server.into_untyped(), client, wait))
}
fn factory(d: &mut Daemon, f: &mut Factory, m: Message) {
    let h = match valid_request(&m, PROTOCOL_NETWORK, &mut f.last_id) {
        Ok(h) => h,
        Err(Some(h)) => {
            f.reply = error(h, STATUS_UNSUPPORTED).ok();
            return;
        }
        Err(None) => {
            f.dead = true;
            return;
        }
    };
    if !known(h.protocol, h.ordinal) {
        f.reply = error(h, STATUS_UNSUPPORTED).ok();
        return;
    }
    if let Some(s) = body_error(&m) {
        if s == STATUS_MALFORMED {
            f.dead = true
        } else {
            f.reply = error(h, s).ok();
        }
        return;
    }
    let b = &m.bytes()[32..];
    let deadline = u64at(b, 0);
    let mut fallback = match error(h, STATUS_LIMIT) {
        Ok(m) => m,
        Err(_) => {
            f.dead = true;
            return;
        }
    };
    let result = (|| -> Result<Message, i32> {
        if deadline <= sys::clock() {
            return Err(STATUS_TIMED_OUT);
        }
        if h.ordinal == NETWORK_DESCRIBE {
            let mut out = reply(h, 16, 0).map_err(|_| STATUS_LIMIT)?;
            out.bytes_mut()[36..40].copy_from_slice(&f.rights.to_le_bytes());
            out.bytes_mut()[40..44].copy_from_slice(&(MAX_NETWORK_BYTES as u32).to_le_bytes());
            out.bytes_mut()[44..48].copy_from_slice(&1u32.to_le_bytes());
            return Ok(out);
        }
        let need = if h.ordinal == NETWORK_CONNECT { 1 } else { 2 };
        if f.rights & need == 0 {
            return Err(STATUS_DENIED);
        }
        let ep = decode_endpoint(&b[8..32])?;
        let rights = u32at(b, 32);
        let accepted = if h.ordinal == NETWORK_LISTEN {
            u32at(b, 36)
        } else {
            0
        };
        let mask = if h.ordinal == NETWORK_CONNECT {
            STREAM_ALL
        } else {
            LISTENER_ALL
        };
        if rights & !mask != 0 || h.ordinal == NETWORK_LISTEN && accepted & !STREAM_ALL != 0 {
            return Err(STATUS_DENIED);
        }
        let (server, client, wait) = prepare_channel(d)?;
        let mut out = reply(h, 8, 1).map_err(|_| STATUS_LIMIT)?;
        out.try_push_handle(client).map_err(|_| STATUS_LIMIT)?;
        out.bytes_mut()[36..40].copy_from_slice(&rights.to_le_bytes());
        let socket = d.engine.create().map_err(status)?;
        let setup = if h.ordinal == NETWORK_CONNECT {
            match d.engine.connect(socket, ep) {
                Ok(()) => Ok(false),
                Err(NetError::WouldBlock) => Ok(true),
                Err(e) => Err(e),
            }
        } else {
            d.engine
                .bind(socket, ep)
                .and_then(|()| d.engine.listen(socket, u32at(b, 40) as usize))
                .map(|()| false)
        };
        let connecting = match setup {
            Ok(c) => c,
            Err(e) => {
                let _ = d.engine.abort(socket);
                return Err(status(e));
            }
        };
        let role = if h.ordinal == NETWORK_CONNECT {
            Role::Stream { rights }
        } else {
            Role::Listener { rights, accepted }
        };
        let mut c = Connection::new(server, socket, role, wait);
        c.connecting = connecting;
        c.connect_deadline = if connecting { Some(deadline) } else { None };
        c.refresh(&mut d.engine);
        if d.admit(c).is_err() {
            let _ = d.engine.abort(socket);
            return Err(STATUS_LIMIT);
        }
        Ok(out)
    })();
    f.reply = Some(match result {
        Ok(m) => m,
        Err(s) => {
            fallback.bytes_mut()[32..36].copy_from_slice(&s.to_le_bytes());
            fallback
        }
    });
}
pub fn factories(d: &mut Daemon) -> Result<(), Error> {
    let count = d.factories.len();
    for _ in 0..count {
        let mut f = d.factories.remove(0);
        if let Some(m) = f.reply.take() {
            match owned::try_send(f.endpoint.as_handle(), m) {
                Ok(()) => {}
                Err(e) if e.error == Error::ShouldWait => f.reply = Some(e.message),
                Err(_) => f.dead = true,
            }
        }
        if !f.dead && f.reply.is_none() {
            match owned::receive(f.endpoint.as_handle(), Deadline::IMMEDIATE) {
                Ok(m) => factory(d, &mut f, m),
                Err(Error::ShouldWait | Error::TimedOut) => {}
                Err(_) => f.dead = true,
            }
        }
        if !f.dead {
            d.factories.push(f)
        }
    }
    Ok(())
}

pub struct Observer {
    sink: OwnedHandle,
    id: u64,
    key: u64,
    mask: u32,
    armed: bool,
    pending: Option<[u8; 64]>,
    wait: Option<crate::wait::Registration>,
}
impl Observer {
    pub fn wait(&mut self, queue: Handle, delivered: Option<u64>) -> Result<(), Error> {
        use dross_abi::event::{PEER_CLOSED, READABLE, WRITABLE};
        let signals = READABLE | PEER_CLOSED | if self.pending.is_some() { WRITABLE } else { 0 };
        crate::wait::Registration::update(
            &mut self.wait,
            queue,
            self.sink.as_handle(),
            signals,
            delivered,
        )
    }
}
fn event(observer: &Observer, c: &Connection) -> [u8; 64] {
    wire::encode_event(wire::Readiness {
        subscription: observer.id,
        key: observer.key,
        revision: c.revision,
        signals: c.snapshot.signals,
        error: c.snapshot.error,
    })
    .expect("validated daemon snapshot")
}
pub fn notify(c: &mut Connection) {
    let Some(mut o) = c.observer.take() else {
        return;
    };
    // Event receiver is a channel peer, not an RPC endpoint. Any reverse traffic
    // retires the watch; every attached capability is closed before it is dropped.
    let mut bytes = [0; 1];
    let mut handles = [Handle(0); 64];
    let closed = match sys::channel_read(o.sink.as_handle().as_raw(), &mut bytes, &mut handles, 0) {
        Err(Error::ShouldWait | Error::TimedOut) => false,
        Ok((_, nh)) => {
            for h in &handles[..nh] {
                let _ = sys::close(*h);
            }
            true
        }
        Err(_) => true,
    };
    if closed {
        return;
    }
    if o.armed && c.snapshot.signals & o.mask != 0 {
        o.pending = Some(event(&o, c));
        o.armed = false;
    }
    if let Some(bytes) = o.pending {
        match sys::channel_write(o.sink.as_handle().as_raw(), &bytes, &[]) {
            Ok(()) => o.pending = None,
            Err(Error::ShouldWait) => {}
            Err(_) => return,
        }
    }
    c.observer = Some(o);
}
fn info(h: Header, c: &Connection) -> Result<Message, i32> {
    let rights = match c.role {
        Role::Stream { rights } | Role::Listener { rights, .. } => rights,
        _ => 0,
    };
    let endpoint = |e: Endpoint| wire::Endpoint::v4(e.address, e.port);
    let body = wire::encode_info(
        h.protocol,
        wire::Info {
            rights,
            signals: c.snapshot.signals,
            error: c.snapshot.error,
            revision: c.revision,
            local: c.snapshot.local.map(endpoint),
            peer: c.snapshot.peer.map(endpoint),
        },
    )
    .map_err(|_| STATUS_BAD_STATE)?;
    let mut m = reply(h, 80, 0).map_err(|_| STATUS_LIMIT)?;
    m.bytes_mut()[32..].copy_from_slice(&body);
    Ok(m)
}
pub fn handle(d: &mut Daemon, c: &mut Connection, mut m: Message) {
    let protocol = if matches!(c.role, Role::Listener { .. }) {
        PROTOCOL_TCP_LISTENER
    } else {
        PROTOCOL_TCP_STREAM
    };
    let h = match valid_request(&m, protocol, &mut c.last_id) {
        Ok(h) => h,
        Err(Some(h)) => {
            c.reply = error(h, STATUS_UNSUPPORTED).ok();
            return;
        }
        Err(None) => {
            c.dead = true;
            return;
        }
    };
    if !known(h.protocol, h.ordinal) {
        c.reply = error(h, STATUS_UNSUPPORTED).ok();
        return;
    }
    if let Some(s) = body_error(&m) {
        if s == STATUS_MALFORMED {
            c.dead = true
        } else {
            c.reply = error(h, s).ok();
        }
        return;
    }
    let mut body = [0u8; 24];
    let copied = (m.bytes().len() - 32).min(body.len());
    body[..copied].copy_from_slice(&m.bytes()[32..32 + copied]);
    let b = &body[..copied];
    let deadline = u64at(b, 0);
    let mut fallback = match error(h, STATUS_LIMIT) {
        Ok(m) => m,
        Err(_) => {
            c.dead = true;
            return;
        }
    };
    let result = (|| -> Result<Message, i32> {
        if deadline <= sys::clock() {
            return Err(STATUS_TIMED_OUT);
        }
        let rights = match c.role {
            Role::Stream { rights } | Role::Listener { rights, .. } => rights,
            _ => 0,
        };
        let listener = matches!(c.role, Role::Listener { .. });
        let need = if listener {
            match h.ordinal {
                1 => 1,
                4 => 4,
                _ => 2,
            }
        } else {
            match h.ordinal {
                1 => 1,
                2 => 2,
                3 => 16,
                4 => 8,
                8 => 32,
                9 => 64,
                _ => 4,
            }
        };
        if rights & need == 0 {
            return Err(STATUS_DENIED);
        }
        match h.ordinal {
            TCP_STREAM_INFO => info(h, c),
            TCP_STREAM_OBSERVE => {
                notify(c);
                if c.observer.is_some() {
                    return Err(STATUS_BAD_STATE);
                }
                let mask = u32at(b, 8);
                let allowed = if listener {
                    8 | 16 | 32
                } else {
                    1 | 2 | 4 | 16 | 32
                };
                if mask == 0 || mask & !allowed != 0 {
                    return Err(STATUS_MALFORMED);
                }
                let key = u64at(b, 16);
                let mut out = reply(h, 16, 0).map_err(|_| STATUS_LIMIT)?;
                let id = d.observer_id().map_err(|_| STATUS_LIMIT)?;
                let sink = m.take_handle(0).ok_or(STATUS_MALFORMED)?;
                let wait = crate::wait::Registration::new(
                    d.queue.as_handle().as_raw(),
                    sink.as_handle(),
                    dross_abi::event::READABLE | dross_abi::event::PEER_CLOSED,
                )
                .map_err(|e| match e {
                    Error::NoMemory | Error::NoResources => STATUS_LIMIT,
                    Error::PeerClosed => STATUS_CLOSED,
                    _ => STATUS_MALFORMED,
                })?;
                out.bytes_mut()[40..48].copy_from_slice(&id.to_le_bytes());
                c.observer = Some(Observer {
                    sink,
                    id,
                    key,
                    mask,
                    armed: true,
                    pending: None,
                    wait: Some(wait),
                });
                Ok(out)
            }
            TCP_STREAM_REARM => {
                let id = u64at(b, 8);
                let revision = u64at(b, 16);
                let out = reply(h, 4, 0).map_err(|_| STATUS_LIMIT)?;
                let Some(o) = c.observer.as_ref() else {
                    return Err(STATUS_BAD_STATE);
                };
                if o.id != id || o.pending.is_some() || o.armed {
                    return Err(STATUS_BAD_STATE);
                }
                let immediate = c.snapshot.signals & o.mask != 0 || revision != c.revision;
                let pending = if immediate { Some(event(o, c)) } else { None };
                let o = c.observer.as_mut().unwrap();
                o.pending = pending;
                o.armed = !immediate;
                Ok(out)
            }
            TCP_STREAM_CANCEL => {
                let id = u64at(b, 8);
                let out = reply(h, 4, 0).map_err(|_| STATUS_LIMIT)?;
                if c.observer.as_ref().is_none_or(|o| o.id != id) {
                    return Err(STATUS_BAD_STATE);
                }
                c.observer.take();
                Ok(out)
            }
            TCP_LISTENER_ACCEPT if listener => {
                let accepted = match c.role {
                    Role::Listener { accepted, .. } => accepted,
                    _ => 0,
                };
                let requested = u32at(b, 8);
                if requested & !accepted != 0 {
                    return Err(STATUS_DENIED);
                }
                let (server, client, wait) = prepare_channel(d)?;
                let mut out = reply(h, 8, 1).map_err(|_| STATUS_LIMIT)?;
                out.try_push_handle(client).map_err(|_| STATUS_LIMIT)?;
                out.bytes_mut()[36..40].copy_from_slice(&requested.to_le_bytes());
                let (socket, _) = d.engine.accept(c.socket).map_err(status)?;
                let mut child =
                    Connection::new(server, socket, Role::Stream { rights: requested }, wait);
                child.refresh(&mut d.engine);
                if d.admit(child).is_err() {
                    let _ = d.engine.close(socket);
                    return Err(STATUS_LIMIT);
                }
                Ok(out)
            }
            TCP_STREAM_READ => {
                let max = u32at(b, 8) as usize;
                let mut out = reply(h, 16 + max, 0).map_err(|_| STATUS_LIMIT)?;
                let count = if c.engine_alive {
                    d.engine
                        .read(c.socket, &mut out.bytes_mut()[48..], false)
                        .map_err(status)?
                } else {
                    return Err(c.error.map(status).unwrap_or(STATUS_CLOSED));
                };
                out.bytes_mut()[36..40].copy_from_slice(&u32::from(count == 0).to_le_bytes());
                out.bytes_mut()[40..44].copy_from_slice(&(count as u32).to_le_bytes());
                out.truncate_bytes(48 + count);
                out.bytes_mut()[12..16].copy_from_slice(&((16 + count) as u32).to_le_bytes());
                Ok(out)
            }
            TCP_STREAM_WRITE => {
                let mut out = reply(h, 8, 0).map_err(|_| STATUS_LIMIT)?;
                let count = if c.engine_alive {
                    d.engine.write(c.socket, &m.bytes()[48..]).map_err(status)?
                } else {
                    return Err(c.error.map(status).unwrap_or(STATUS_CLOSED));
                };
                out.bytes_mut()[36..40].copy_from_slice(&(count as u32).to_le_bytes());
                Ok(out)
            }
            TCP_STREAM_SHUTDOWN_SEND | TCP_STREAM_SHUTDOWN_RECEIVE => {
                let out = reply(h, 4, 0).map_err(|_| STATUS_LIMIT)?;
                if !c.engine_alive {
                    return Err(c.error.map(status).unwrap_or(STATUS_CLOSED));
                }
                d.engine
                    .shutdown(
                        c.socket,
                        if h.ordinal == TCP_STREAM_SHUTDOWN_SEND {
                            Shutdown::Write
                        } else {
                            Shutdown::Read
                        },
                    )
                    .map_err(status)?;
                Ok(out)
            }
            TCP_STREAM_ABORT => {
                let out = reply(h, 4, 0).map_err(|_| STATUS_LIMIT)?;
                if c.engine_alive {
                    d.engine.abort(c.socket).map_err(status)?;
                    c.engine_alive = false;
                }
                c.observer.take();
                c.retire_after_reply = true;
                Ok(out)
            }
            _ => Err(STATUS_UNSUPPORTED),
        }
    })();
    c.reply = Some(match result {
        Ok(m) => m,
        Err(s) => {
            fallback.bytes_mut()[32..36].copy_from_slice(&s.to_le_bytes());
            fallback
        }
    });
}
