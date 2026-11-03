//! Native TCP capability protocol and serial Rust clients.
//!
//! Network grants CONNECT/LISTEN authority; each stream or listener has its own
//! server-enforced operation rights. These are separate from channel rights.
//! No descriptor numbers, socket addresses or errno values cross this interface.
//!
//! All RPCs carry one absolute monotonic deadline. A confirmed write means bytes
//! entered the local TCP queue, not that the peer received them. A reply failure
//! after send retires the control connection and reports an uncertain outcome;
//! writes are never replayed automatically. UDP, resolution and administration
//! are separate future protocols, not successful no-op operations here.
use super::{
    ProtocolError,
    frame::{Frame, Header, get16, get32, get64},
    schema::*,
};
use crate::{
    Error,
    deadline::Deadline,
    owned::{self, BorrowedHandle, Channel, EventQueue, Message, OwnedHandle, Untyped},
};
use core::sync::atomic::{AtomicU64, Ordering};

pub const CONNECT: u32 = 1;
pub const LISTEN: u32 = 2;
pub const NETWORK_RIGHTS: u32 = CONNECT | LISTEN;
pub mod stream_rights {
    pub const READ: u32 = 1;
    pub const WRITE: u32 = 2;
    pub const OBSERVE: u32 = 4;
    pub const INFO: u32 = 8;
    pub const SHUTDOWN_SEND: u32 = 16;
    pub const SHUTDOWN_RECEIVE: u32 = 32;
    pub const ABORT: u32 = 64;
    pub const ALL: u32 = 127;
}
pub mod listener_rights {
    pub const ACCEPT: u32 = 1;
    pub const OBSERVE: u32 = 2;
    pub const INFO: u32 = 4;
    pub const ALL: u32 = 7;
}
pub mod signals {
    pub const READABLE: u32 = 1;
    pub const WRITABLE: u32 = 2;
    pub const CONNECTED: u32 = 4;
    pub const ACCEPTABLE: u32 = 8;
    pub const PEER_CLOSED: u32 = 16;
    pub const ERROR: u32 = 32;
    pub const STREAM: u32 = READABLE | WRITABLE | CONNECTED | PEER_CLOSED | ERROR;
    pub const LISTENER: u32 = ACCEPTABLE | PEER_CLOSED | ERROR;
}
pub const MAX_INLINE: usize = MAX_NETWORK_BYTES;
pub const MAX_BACKLOG: u32 = 4;
pub const TRANSPORT_RIGHTS: u32 = dross_abi::rights::READ
    | dross_abi::rights::WRITE
    | dross_abi::rights::WAIT
    | dross_abi::rights::TRANSFER;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Address {
    V4([u8; 4]),
    V6 { octets: [u8; 16], scope: u32 },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub address: Address,
    pub port: u16,
}
impl Endpoint {
    pub const fn v4(octets: [u8; 4], port: u16) -> Self {
        Self {
            address: Address::V4(octets),
            port,
        }
    }
    pub const fn v6(octets: [u8; 16], scope: u32, port: u16) -> Self {
        Self {
            address: Address::V6 { octets, scope },
            port,
        }
    }
    /// 24 bytes: family u16 (4/6), port u16 LE, scope u32 LE, address octets.
    pub fn encode(self) -> [u8; 24] {
        let mut out = [0; 24];
        out[2..4].copy_from_slice(&self.port.to_le_bytes());
        match self.address {
            Address::V4(a) => {
                out[..2].copy_from_slice(&4u16.to_le_bytes());
                out[8..12].copy_from_slice(&a);
            }
            Address::V6 { octets, scope } => {
                out[..2].copy_from_slice(&6u16.to_le_bytes());
                out[4..8].copy_from_slice(&scope.to_le_bytes());
                out[8..].copy_from_slice(&octets);
            }
        }
        out
    }
    /// IPv6 is a valid wire family even when a service supports only IPv4.
    pub fn decode(bytes: &[u8]) -> Result<Self, ProtocolError> {
        if bytes.len() != 24 {
            return Err(ProtocolError::Malformed);
        }
        let port = get16(bytes, 2);
        let scope = get32(bytes, 4);
        match get16(bytes, 0) {
            4 if scope == 0 && bytes[12..].iter().all(|&b| b == 0) => {
                Ok(Self::v4(bytes[8..12].try_into().unwrap(), port))
            }
            6 => Ok(Self::v6(bytes[8..24].try_into().unwrap(), scope, port)),
            _ => Err(ProtocolError::Malformed),
        }
    }
}
fn rights_mask(protocol: u32) -> Result<u32, ProtocolError> {
    match protocol {
        PROTOCOL_NETWORK => Ok(NETWORK_RIGHTS),
        PROTOCOL_TCP_STREAM => Ok(stream_rights::ALL),
        PROTOCOL_TCP_LISTENER => Ok(listener_rights::ALL),
        _ => Err(ProtocolError::Unsupported),
    }
}
fn signal_mask(protocol: u32) -> Result<u32, ProtocolError> {
    match protocol {
        PROTOCOL_TCP_STREAM => Ok(signals::STREAM),
        PROTOCOL_TCP_LISTENER => Ok(signals::LISTENER),
        _ => Err(ProtocolError::Unsupported),
    }
}
fn valid_rights(rights: u32, mask: u32) -> Result<(), ProtocolError> {
    if rights & !mask == 0 {
        Ok(())
    } else {
        Err(ProtocolError::Malformed)
    }
}
fn status_valid(status: i32) -> bool {
    status <= 0
}
fn append(message: &mut Message, bytes: &[u8]) -> Result<(), ProtocolError> {
    message.try_extend_bytes(bytes).map_err(|e| {
        if e == Error::NoMemory {
            ProtocolError::Allocation
        } else {
            ProtocolError::Limit
        }
    })
}
fn frame_message(
    protocol: u32,
    kind: u16,
    ordinal: u32,
    id: u64,
    parts: &[&[u8]],
    handles: usize,
) -> Result<Message, ProtocolError> {
    let size = parts
        .iter()
        .try_fold(0usize, |n, b| n.checked_add(b.len()))
        .ok_or(ProtocolError::Limit)?;
    let header = Header::new(protocol, 1, kind, ordinal, id, size, handles)?;
    let mut bytes = [0; 32];
    header.encode(&mut bytes)?;
    let mut message = Message::try_from_bytes(&bytes).map_err(|_| ProtocolError::Allocation)?;
    for part in parts {
        append(&mut message, part)?;
    }
    Ok(message)
}
/// Build a status reply. Error replies have no trailing bytes or attachments.
pub fn reply_message(
    protocol: u32,
    ordinal: u32,
    id: u64,
    status: i32,
    payload: &[u8],
    handles: usize,
) -> Result<Message, ProtocolError> {
    if !status_valid(status) || (status != 0 && (!payload.is_empty() || handles != 0)) {
        return Err(ProtocolError::Malformed);
    }
    frame_message(
        protocol,
        KIND_REPLY,
        ordinal,
        id,
        &[&status.to_le_bytes(), payload],
        handles,
    )
}
fn request_message(
    protocol: u32,
    ordinal: u32,
    id: u64,
    deadline: Deadline,
    parts: &[&[u8]],
    handles: usize,
) -> Result<Message, ProtocolError> {
    let size = parts
        .iter()
        .try_fold(8usize, |n, b| n.checked_add(b.len()))
        .ok_or(ProtocolError::Limit)?;
    let header = Header::new(protocol, 1, KIND_REQUEST, ordinal, id, size, handles)?;
    let mut bytes = [0; 32];
    header.encode(&mut bytes)?;
    let mut message = Message::try_from_bytes(&bytes).map_err(|_| ProtocolError::Allocation)?;
    append(&mut message, &deadline.as_monotonic_ns().to_le_bytes())?;
    for part in parts {
        append(&mut message, part)?;
    }
    Ok(message)
}

#[derive(Debug, PartialEq, Eq)]
pub enum Operation<'a> {
    Describe,
    Connect {
        remote: Endpoint,
        rights: u32,
    },
    Listen {
        local: Endpoint,
        rights: u32,
        stream_rights: u32,
        backlog: u32,
    },
    Accept {
        rights: u32,
    },
    Read {
        maximum: u32,
    },
    Write {
        bytes: &'a [u8],
    },
    ShutdownSend,
    ShutdownReceive,
    Abort,
    Info,
    Observe {
        mask: u32,
        key: u64,
    },
    Rearm {
        subscription: u64,
        revision: u64,
    },
    Cancel {
        subscription: u64,
    },
}
#[derive(Debug)]
pub struct WireRequest<'a> {
    pub header: Header,
    pub deadline: Deadline,
    pub operation: Operation<'a>,
}
/// Structural decoding never takes attachments. The service owns the Message
/// throughout validation and closes unused attachments on every rejected path.
pub fn decode_request(bytes: &[u8], handles: usize) -> Result<WireRequest<'_>, ProtocolError> {
    let f = Frame::parse(bytes, handles)?;
    let h = f.header;
    if !h.supported(h.protocol, 1) || h.kind != KIND_REQUEST {
        return Err(ProtocolError::Unsupported);
    }
    rights_mask(h.protocol)?;
    if f.body.len() < 8 {
        return Err(ProtocolError::Malformed);
    }
    let deadline = Deadline::at_monotonic_ns(get64(f.body, 0));
    let b = &f.body[8..];
    let mut expected_handles = 0;
    let operation = match (h.protocol, h.ordinal) {
        (PROTOCOL_NETWORK, NETWORK_DESCRIBE) if b.is_empty() => Operation::Describe,
        (PROTOCOL_NETWORK, NETWORK_CONNECT) if b.len() == 32 && get32(b, 28) == 0 => {
            let remote = Endpoint::decode(&b[..24])?;
            if remote.port == 0 {
                return Err(ProtocolError::Malformed);
            }
            let rights = get32(b, 24);
            valid_rights(rights, stream_rights::ALL)?;
            Operation::Connect { remote, rights }
        }
        (PROTOCOL_NETWORK, NETWORK_LISTEN) if b.len() == 40 && get32(b, 36) == 0 => {
            let local = Endpoint::decode(&b[..24])?;
            let rights = get32(b, 24);
            let sr = get32(b, 28);
            let backlog = get32(b, 32);
            valid_rights(rights, listener_rights::ALL)?;
            valid_rights(sr, stream_rights::ALL)?;
            if !(1..=MAX_BACKLOG).contains(&backlog) {
                return Err(ProtocolError::Limit);
            }
            Operation::Listen {
                local,
                rights,
                stream_rights: sr,
                backlog,
            }
        }
        (PROTOCOL_TCP_LISTENER, TCP_LISTENER_ACCEPT) if b.len() == 8 && get32(b, 4) == 0 => {
            let rights = get32(b, 0);
            valid_rights(rights, stream_rights::ALL)?;
            Operation::Accept { rights }
        }
        (PROTOCOL_TCP_STREAM, TCP_STREAM_READ) if b.len() == 8 && get32(b, 4) == 0 => {
            let maximum = get32(b, 0);
            if maximum == 0 || maximum as usize > MAX_INLINE {
                return Err(ProtocolError::Limit);
            }
            Operation::Read { maximum }
        }
        (PROTOCOL_TCP_STREAM, TCP_STREAM_WRITE) if b.len() >= 8 && get32(b, 4) == 0 => {
            let n = get32(b, 0) as usize;
            if n > MAX_INLINE || b.len() != 8 + n {
                return Err(ProtocolError::Malformed);
            }
            Operation::Write { bytes: &b[8..] }
        }
        (PROTOCOL_TCP_STREAM, TCP_STREAM_SHUTDOWN_SEND) if b.is_empty() => Operation::ShutdownSend,
        (PROTOCOL_TCP_STREAM, TCP_STREAM_SHUTDOWN_RECEIVE) if b.is_empty() => {
            Operation::ShutdownReceive
        }
        (PROTOCOL_TCP_STREAM, TCP_STREAM_ABORT) if b.is_empty() => Operation::Abort,
        (PROTOCOL_TCP_STREAM | PROTOCOL_TCP_LISTENER, 4) if b.is_empty() => Operation::Info,
        (PROTOCOL_TCP_STREAM | PROTOCOL_TCP_LISTENER, 5) if b.len() == 16 && get32(b, 4) == 0 => {
            let mask = get32(b, 0);
            if mask == 0 || mask & !signal_mask(h.protocol)? != 0 {
                return Err(ProtocolError::Malformed);
            }
            expected_handles = 1;
            Operation::Observe {
                mask,
                key: get64(b, 8),
            }
        }
        (PROTOCOL_TCP_STREAM | PROTOCOL_TCP_LISTENER, 6)
            if b.len() == 16 && get64(b, 0) != 0 && get64(b, 8) != 0 =>
        {
            Operation::Rearm {
                subscription: get64(b, 0),
                revision: get64(b, 8),
            }
        }
        (PROTOCOL_TCP_STREAM | PROTOCOL_TCP_LISTENER, 7) if b.len() == 8 && get64(b, 0) != 0 => {
            Operation::Cancel {
                subscription: get64(b, 0),
            }
        }
        (PROTOCOL_NETWORK, 0..=2)
        | (PROTOCOL_TCP_STREAM, 1..=9)
        | (PROTOCOL_TCP_LISTENER, 1 | 4..=7) => return Err(ProtocolError::Malformed),
        _ => return Err(ProtocolError::Unsupported),
    };
    if handles != expected_handles {
        return Err(ProtocolError::Malformed);
    }
    Ok(WireRequest {
        header: h,
        deadline,
        operation,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Info {
    pub rights: u32,
    pub signals: u32,
    pub error: i32,
    pub revision: u64,
    pub local: Option<Endpoint>,
    pub peer: Option<Endpoint>,
}
pub fn encode_info(protocol: u32, info: Info) -> Result<[u8; 80], ProtocolError> {
    valid_rights(info.rights, rights_mask(protocol)?)?;
    if info.signals & !signal_mask(protocol)? != 0
        || info.revision == 0
        || !status_valid(info.error)
        || (info.error != 0) != (info.signals & signals::ERROR != 0)
    {
        return Err(ProtocolError::Malformed);
    }
    let mut out = [0; 80];
    out[4..8].copy_from_slice(&info.rights.to_le_bytes());
    out[8..12].copy_from_slice(&info.signals.to_le_bytes());
    out[12..16].copy_from_slice(&info.error.to_le_bytes());
    out[16..24].copy_from_slice(&info.revision.to_le_bytes());
    let mut flags = 0u32;
    if let Some(local) = info.local {
        out[24..48].copy_from_slice(&local.encode());
        flags |= 1;
    }
    if let Some(peer) = info.peer {
        out[48..72].copy_from_slice(&peer.encode());
        flags |= 2;
    }
    out[72..76].copy_from_slice(&flags.to_le_bytes());
    Ok(out)
}
pub fn decode_info(protocol: u32, b: &[u8]) -> Result<Info, ProtocolError> {
    if b.len() != 80 || get32(b, 0) != 0 || get32(b, 76) != 0 || get32(b, 72) & !3 != 0 {
        return Err(ProtocolError::Malformed);
    }
    let flags = get32(b, 72);
    let local = if flags & 1 != 0 {
        Some(Endpoint::decode(&b[24..48])?)
    } else {
        if b[24..48].iter().any(|&v| v != 0) {
            return Err(ProtocolError::Malformed);
        }
        None
    };
    let peer = if flags & 2 != 0 {
        Some(Endpoint::decode(&b[48..72])?)
    } else {
        if b[48..72].iter().any(|&v| v != 0) {
            return Err(ProtocolError::Malformed);
        }
        None
    };
    let info = Info {
        rights: get32(b, 4),
        signals: get32(b, 8),
        error: get32(b, 12) as i32,
        revision: get64(b, 16),
        local,
        peer,
    };
    encode_info(protocol, info)?;
    Ok(info)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readiness {
    pub subscription: u64,
    pub key: u64,
    pub revision: u64,
    pub signals: u32,
    pub error: i32,
}
/// Fixed 64-byte packet: no allocation is needed after a readiness transition.
pub fn encode_event(event: Readiness) -> Result<[u8; 64], ProtocolError> {
    if event.subscription == 0
        || event.revision == 0
        || event.signals & !(signals::STREAM | signals::LISTENER) != 0
        || !status_valid(event.error)
        || (event.error != 0) != (event.signals & signals::ERROR != 0)
    {
        return Err(ProtocolError::Malformed);
    }
    let mut out = [0; 64];
    Header::new(
        PROTOCOL_NETWORK_OBSERVER,
        1,
        KIND_EVENT,
        NETWORK_OBSERVER_READY,
        0,
        32,
        0,
    )?
    .encode(&mut out[..32])?;
    out[32..40].copy_from_slice(&event.subscription.to_le_bytes());
    out[40..48].copy_from_slice(&event.key.to_le_bytes());
    out[48..56].copy_from_slice(&event.revision.to_le_bytes());
    out[56..60].copy_from_slice(&event.signals.to_le_bytes());
    out[60..64].copy_from_slice(&event.error.to_le_bytes());
    Ok(out)
}
pub fn decode_event(bytes: &[u8], handles: usize) -> Result<Readiness, ProtocolError> {
    let f = Frame::parse(bytes, handles)?;
    if !f.header.supported(PROTOCOL_NETWORK_OBSERVER, 1)
        || f.header.kind != KIND_EVENT
        || f.header.ordinal != NETWORK_OBSERVER_READY
        || f.body.len() != 32
        || handles != 0
    {
        return Err(ProtocolError::Malformed);
    }
    let b = f.body;
    let e = Readiness {
        subscription: get64(b, 0),
        key: get64(b, 8),
        revision: get64(b, 16),
        signals: get32(b, 24),
        error: get32(b, 28) as i32,
    };
    encode_event(e)?;
    Ok(e)
}

#[derive(Debug, PartialEq, Eq)]
pub enum ReadReply<'a> {
    Data(&'a [u8]),
    Eof,
    Error(i32),
}
pub fn decode_read_reply(
    body: &[u8],
    handles: usize,
    maximum: usize,
) -> Result<ReadReply<'_>, ProtocolError> {
    if body.len() < 4 || handles != 0 || maximum == 0 || maximum > MAX_INLINE {
        return Err(ProtocolError::Malformed);
    }
    let status = get32(body, 0) as i32;
    if status > 0 {
        return Err(ProtocolError::Malformed);
    }
    if status != 0 {
        return if body.len() == 4 {
            Ok(ReadReply::Error(status))
        } else {
            Err(ProtocolError::Malformed)
        };
    }
    if body.len() < 16 || get32(body, 12) != 0 {
        return Err(ProtocolError::Malformed);
    }
    let count = get32(body, 8) as usize;
    if count > maximum || body.len() != 16 + count {
        return Err(ProtocolError::Malformed);
    }
    match (get32(body, 4), count) {
        (0, 1..) => Ok(ReadReply::Data(&body[16..])),
        (1, 0) => Ok(ReadReply::Eof),
        _ => Err(ProtocolError::Malformed),
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteReply {
    Accepted(usize),
    Error(i32),
}
pub fn decode_write_reply(
    body: &[u8],
    handles: usize,
    offered: usize,
) -> Result<WriteReply, ProtocolError> {
    if body.len() < 4 || handles != 0 || offered > MAX_INLINE {
        return Err(ProtocolError::Malformed);
    }
    let status = get32(body, 0) as i32;
    if status > 0 {
        return Err(ProtocolError::Malformed);
    }
    if status != 0 {
        return if body.len() == 4 {
            Ok(WriteReply::Error(status))
        } else {
            Err(ProtocolError::Malformed)
        };
    }
    if body.len() != 8 {
        return Err(ProtocolError::Malformed);
    }
    let count = get32(body, 4) as usize;
    if count > offered || (offered != 0 && count == 0) {
        return Err(ProtocolError::Malformed);
    }
    Ok(WriteReply::Accepted(count))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrantReply {
    Granted(u32),
    Error(i32),
}
pub fn decode_grant_reply(
    body: &[u8],
    handles: usize,
    requested: u32,
) -> Result<GrantReply, ProtocolError> {
    if body.len() < 4 {
        return Err(ProtocolError::Malformed);
    }
    let status = get32(body, 0) as i32;
    if status > 0 {
        return Err(ProtocolError::Malformed);
    }
    if status != 0 {
        return if body.len() == 4 && handles == 0 {
            Ok(GrantReply::Error(status))
        } else {
            Err(ProtocolError::Malformed)
        };
    }
    if body.len() != 8 || handles != 1 || get32(body, 4) & !requested != 0 {
        return Err(ProtocolError::Malformed);
    }
    Ok(GrantReply::Granted(get32(body, 4)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetworkError {
    NotSent(Error),
    OutcomeUnknown(Error),
    InvalidReply(ProtocolError),
    Protocol(ProtocolError),
    /// Native service STATUS_LIMIT: bounded admission/resource exhaustion.
    NoResources,
    WouldBlock,
    Denied,
    Unsupported,
    TimedOut,
    Refused,
    Reset,
    Closed,
    NotConnected,
    AddressInUse,
    Unreachable,
    Unavailable,
    Service(i32),
    Retired,
}
impl From<ProtocolError> for NetworkError {
    fn from(e: ProtocolError) -> Self {
        Self::Protocol(e)
    }
}
pub fn status_error(status: i32) -> NetworkError {
    match status {
        STATUS_LIMIT => NetworkError::NoResources,
        STATUS_WOULD_BLOCK => NetworkError::WouldBlock,
        STATUS_DENIED => NetworkError::Denied,
        STATUS_UNSUPPORTED => NetworkError::Unsupported,
        STATUS_TIMED_OUT => NetworkError::TimedOut,
        STATUS_REFUSED => NetworkError::Refused,
        STATUS_RESET => NetworkError::Reset,
        STATUS_CLOSED => NetworkError::Closed,
        STATUS_NOT_CONNECTED => NetworkError::NotConnected,
        STATUS_ADDRESS_IN_USE => NetworkError::AddressInUse,
        STATUS_UNREACHABLE => NetworkError::Unreachable,
        STATUS_UNAVAILABLE => NetworkError::Unavailable,
        n => NetworkError::Service(n),
    }
}
// After CONNECT admission, a later unsent control RPC cannot establish that
// the overall connection attempt was never submitted to the network.
fn admitted_error(error: NetworkError) -> NetworkError {
    match error {
        NetworkError::NotSent(error) => NetworkError::OutcomeUnknown(error),
        other => other,
    }
}
static NEXT_COOKIE: AtomicU64 = AtomicU64::new(1);
#[derive(Debug)]
struct Rpc {
    endpoint: Option<OwnedHandle>,
    protocol: u32,
    next: u64,
    cookie: u64,
}
impl Rpc {
    fn new(endpoint: OwnedHandle, protocol: u32) -> Result<Self, NetworkError> {
        let cookie = NEXT_COOKIE
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| NetworkError::Protocol(ProtocolError::Limit))?;
        Ok(Self {
            endpoint: Some(endpoint),
            protocol,
            next: 1,
            cookie,
        })
    }
    fn retired(&self) -> bool {
        self.endpoint.is_none()
    }
    fn retire(&mut self) {
        self.endpoint.take();
    }
    fn invalid<T>(&mut self, error: ProtocolError) -> Result<T, NetworkError> {
        self.retire();
        Err(NetworkError::InvalidReply(error))
    }
    fn call(
        &mut self,
        ordinal: u32,
        parts: &[&[u8]],
        attachment: Option<OwnedHandle>,
        deadline: Deadline,
    ) -> Result<Message, NetworkError> {
        if self.endpoint.is_none() {
            return Err(NetworkError::Retired);
        }
        if self.next == u64::MAX {
            self.retire();
            return Err(NetworkError::Retired);
        }
        let id = self.next;
        self.next += 1;
        if deadline.has_elapsed() {
            return Err(NetworkError::NotSent(Error::TimedOut));
        }
        let mut message = request_message(
            self.protocol,
            ordinal,
            id,
            deadline,
            parts,
            usize::from(attachment.is_some()),
        )?;
        if let Some(cap) = attachment {
            message
                .try_push_handle(cap)
                .map_err(|_| NetworkError::Protocol(ProtocolError::Allocation))?;
        }
        let endpoint = self.endpoint.as_ref().unwrap();
        let mut queue = None;
        loop {
            if deadline.has_elapsed() {
                return Err(NetworkError::NotSent(Error::TimedOut));
            }
            match owned::try_send(endpoint.as_handle(), message) {
                Ok(()) => break,
                Err(failure) => {
                    message = failure.message;
                    if failure.error != Error::ShouldWait {
                        return Err(NetworkError::NotSent(failure.error));
                    }
                    if queue.is_none() {
                        queue = Some(EventQueue::create(1).map_err(NetworkError::NotSent)?);
                    }
                    let registration = queue
                        .as_ref()
                        .unwrap()
                        .subscribe(
                            endpoint.as_handle(),
                            dross_abi::event::WRITABLE | dross_abi::event::PEER_CLOSED,
                            0,
                        )
                        .map_err(NetworkError::NotSent)?;
                    queue
                        .as_ref()
                        .unwrap()
                        .wait_until(deadline)
                        .map_err(NetworkError::NotSent)?;
                    drop(registration);
                }
            }
        }
        let reply = match owned::receive(endpoint.as_handle(), deadline) {
            Ok(m) => m,
            Err(error) => {
                self.retire();
                return Err(NetworkError::OutcomeUnknown(error));
            }
        };
        let frame = match Frame::parse(reply.bytes(), reply.handle_count()) {
            Ok(f) => f,
            Err(e) => return self.invalid(e.into()),
        };
        if !frame.header.supported(self.protocol, 1)
            || frame.header.kind != KIND_REPLY
            || frame.header.ordinal != ordinal
            || frame.header.request_id != id
            || frame.body.len() < 4
        {
            return self.invalid(ProtocolError::Malformed);
        }
        let status = get32(frame.body, 0) as i32;
        if !status_valid(status) {
            return self.invalid(ProtocolError::Malformed);
        }
        if status != 0 {
            if frame.body.len() != 4 || reply.handle_count() != 0 {
                return self.invalid(ProtocolError::Malformed);
            }
            return Err(status_error(status));
        }
        Ok(reply)
    }
    fn simple(&mut self, ordinal: u32, deadline: Deadline) -> Result<(), NetworkError> {
        let m = self.call(ordinal, &[], None, deadline)?;
        if m.bytes().len() != 36 || m.handle_count() != 0 {
            return self.invalid(ProtocolError::Malformed);
        }
        Ok(())
    }
    fn info(&mut self, deadline: Deadline) -> Result<Info, NetworkError> {
        let m = self.call(4, &[], None, deadline)?;
        if m.handle_count() != 0 {
            return self.invalid(ProtocolError::Malformed);
        }
        match decode_info(self.protocol, &m.bytes()[32..]) {
            Ok(info) => Ok(info),
            Err(error) => self.invalid(error),
        }
    }
    fn observe(
        &mut self,
        mask: u32,
        key: u64,
        deadline: Deadline,
    ) -> Result<Observation, NetworkError> {
        if mask == 0 || mask & !signal_mask(self.protocol)? != 0 {
            return Err(ProtocolError::Malformed.into());
        }
        let (receiver, server) = Channel::pair().map_err(NetworkError::NotSent)?;
        let receiver = receiver
            .try_duplicate(
                dross_abi::rights::READ | dross_abi::rights::WAIT | dross_abi::rights::TRANSFER,
            )
            .map_err(NetworkError::NotSent)?;
        let mut body = [0; 16];
        body[..4].copy_from_slice(&mask.to_le_bytes());
        body[8..].copy_from_slice(&key.to_le_bytes());
        let m = self.call(5, &[&body], Some(server.into_untyped()), deadline)?;
        if m.bytes().len() != 48
            || m.handle_count() != 0
            || get32(m.bytes(), 36) != 0
            || get64(m.bytes(), 40) == 0
        {
            return self.invalid(ProtocolError::Malformed);
        }
        Ok(Observation {
            receiver: Some(receiver),
            subscription: get64(m.bytes(), 40),
            key,
            source: self.cookie,
            protocol: self.protocol,
            last_revision: 0,
            dequeued: false,
        })
    }
    fn rearm(&mut self, watch: &mut Observation, deadline: Deadline) -> Result<(), NetworkError> {
        if watch.source != self.cookie || !watch.dequeued || watch.receiver.is_none() {
            return Err(NetworkError::NotSent(Error::InvalidArgs));
        }
        let mut body = [0; 16];
        body[..8].copy_from_slice(&watch.subscription.to_le_bytes());
        body[8..].copy_from_slice(&watch.last_revision.to_le_bytes());
        let m = self.call(6, &[&body], None, deadline)?;
        if m.bytes().len() != 36 || m.handle_count() != 0 {
            return self.invalid(ProtocolError::Malformed);
        }
        watch.dequeued = false;
        Ok(())
    }
    fn cancel(&mut self, watch: &mut Observation, deadline: Deadline) -> Result<(), NetworkError> {
        if watch.source != self.cookie || watch.receiver.is_none() {
            return Err(NetworkError::NotSent(Error::InvalidArgs));
        }
        let m = self.call(7, &[&watch.subscription.to_le_bytes()], None, deadline)?;
        if m.bytes().len() != 36 || m.handle_count() != 0 {
            return self.invalid(ProtocolError::Malformed);
        }
        watch.receiver.take();
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetworkInfo {
    pub rights: u32,
    pub maximum_inline: u32,
    pub families: u32,
}
/// A fresh factory connection with one request/reply reader. Never wrap a
/// previously used connection in a new client, which would reuse request IDs.
#[derive(Debug)]
pub struct Network {
    rpc: Rpc,
}
#[derive(Debug)]
pub struct TcpStream {
    rpc: Rpc,
}
#[derive(Debug)]
pub struct TcpListener {
    rpc: Rpc,
}
fn received_endpoint(
    rpc: &mut Rpc,
    mut message: Message,
    allowed: u32,
) -> Result<OwnedHandle, NetworkError> {
    match decode_grant_reply(&message.bytes()[32..], message.handle_count(), allowed) {
        Ok(GrantReply::Granted(_)) if message.remaining_handles() == 1 => message
            .take_handle(0)
            .ok_or(NetworkError::InvalidReply(ProtocolError::Malformed)),
        Ok(GrantReply::Error(status)) => Err(status_error(status)),
        _ => rpc.invalid(ProtocolError::Malformed),
    }
}
impl Network {
    pub fn new(endpoint: OwnedHandle) -> Result<Self, NetworkError> {
        Ok(Self {
            rpc: Rpc::new(endpoint, PROTOCOL_NETWORK)?,
        })
    }
    pub fn is_retired(&self) -> bool {
        self.rpc.retired()
    }
    pub fn describe(&mut self, deadline: Deadline) -> Result<NetworkInfo, NetworkError> {
        let m = self.rpc.call(NETWORK_DESCRIBE, &[], None, deadline)?;
        if m.bytes().len() != 48 || m.handle_count() != 0 {
            return self.rpc.invalid(ProtocolError::Malformed);
        }
        let b = m.bytes();
        let info = NetworkInfo {
            rights: get32(b, 36),
            maximum_inline: get32(b, 40),
            families: get32(b, 44),
        };
        if info.rights & !NETWORK_RIGHTS != 0
            || info.maximum_inline == 0
            || info.maximum_inline as usize > MAX_INLINE
            || info.families & !3 != 0
        {
            return self.rpc.invalid(ProtocolError::Malformed);
        }
        Ok(info)
    }
    /// Admission returns a connecting object. Its absolute handshake deadline
    /// is retained by the service even after this immediate reply completes.
    pub fn begin_connect(
        &mut self,
        remote: Endpoint,
        rights: u32,
        deadline: Deadline,
    ) -> Result<TcpStream, NetworkError> {
        valid_rights(rights, stream_rights::ALL)?;
        if remote.port == 0 {
            return Err(ProtocolError::Malformed.into());
        }
        let mut tail = [0; 32];
        tail[..24].copy_from_slice(&remote.encode());
        tail[24..28].copy_from_slice(&rights.to_le_bytes());
        let reply = self.rpc.call(NETWORK_CONNECT, &[&tail], None, deadline)?;
        let cap = received_endpoint(&mut self.rpc, reply, rights)?;
        Ok(TcpStream {
            rpc: Rpc::new(cap, PROTOCOL_TCP_STREAM)?,
        })
    }
    pub fn connect_tcp(
        &mut self,
        remote: Endpoint,
        rights: u32,
        deadline: Deadline,
    ) -> Result<TcpStream, NetworkError> {
        // INFO/OBSERVE are needed by this waiting convenience function. They
        // are never silently added to the rights the caller requested.
        if rights & (stream_rights::INFO | stream_rights::OBSERVE)
            != (stream_rights::INFO | stream_rights::OBSERVE)
        {
            return Err(NetworkError::Denied);
        }
        let mut stream = self.begin_connect(remote, rights, deadline)?;
        let info = stream.info(deadline).map_err(admitted_error)?;
        if info.error != 0 {
            return Err(status_error(info.error));
        }
        if info.signals & signals::CONNECTED != 0 {
            return Ok(stream);
        }
        let mut watch = stream
            .observe(
                signals::CONNECTED | signals::ERROR | signals::PEER_CLOSED,
                0,
                deadline,
            )
            .map_err(admitted_error)?;
        loop {
            let event = watch.receive(deadline).map_err(admitted_error)?;
            if event.error != 0 {
                return Err(status_error(event.error));
            }
            if event.signals & signals::CONNECTED != 0 {
                return Ok(stream);
            }
            if event.signals & signals::PEER_CLOSED != 0 {
                return Err(NetworkError::Closed);
            }
            stream.rearm(&mut watch, deadline).map_err(admitted_error)?;
        }
    }
    pub fn listen_tcp(
        &mut self,
        local: Endpoint,
        rights: u32,
        accepted_rights: u32,
        backlog: u32,
        deadline: Deadline,
    ) -> Result<TcpListener, NetworkError> {
        valid_rights(rights, listener_rights::ALL)?;
        valid_rights(accepted_rights, stream_rights::ALL)?;
        if !(1..=MAX_BACKLOG).contains(&backlog) {
            return Err(ProtocolError::Limit.into());
        }
        let mut tail = [0; 40];
        tail[..24].copy_from_slice(&local.encode());
        tail[24..28].copy_from_slice(&rights.to_le_bytes());
        tail[28..32].copy_from_slice(&accepted_rights.to_le_bytes());
        tail[32..36].copy_from_slice(&backlog.to_le_bytes());
        let reply = self.rpc.call(NETWORK_LISTEN, &[&tail], None, deadline)?;
        let cap = received_endpoint(&mut self.rpc, reply, rights)?;
        Ok(TcpListener {
            rpc: Rpc::new(cap, PROTOCOL_TCP_LISTENER)?,
        })
    }
}
/// Read data retains the received message allocation; no second allocation can
/// fail after the service has consumed bytes. This is ordinary private memory.
#[derive(Debug)]
pub struct ReadData {
    message: Message,
}
impl ReadData {
    pub fn bytes(&self) -> &[u8] {
        &self.message.bytes()[48..]
    }
    pub fn len(&self) -> usize {
        self.bytes().len()
    }
    pub fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }
}
#[derive(Debug)]
pub enum ReadResult {
    Data(ReadData),
    Eof,
}
#[derive(Debug)]
pub struct WriteAllError {
    pub error: NetworkError,
    pub accepted: usize,
}
impl WriteAllError {
    pub fn latest_request_may_have_succeeded(&self) -> bool {
        matches!(
            self.error,
            NetworkError::OutcomeUnknown(_) | NetworkError::InvalidReply(_)
        )
    }
}
impl TcpStream {
    /// For explicitly delegated fresh stream connections, never reused control
    /// channels. Server checks the protocol and logical rights on every call.
    pub fn new(endpoint: OwnedHandle) -> Result<Self, NetworkError> {
        Ok(Self {
            rpc: Rpc::new(endpoint, PROTOCOL_TCP_STREAM)?,
        })
    }
    pub fn is_retired(&self) -> bool {
        self.rpc.retired()
    }
    pub fn info(&mut self, deadline: Deadline) -> Result<Info, NetworkError> {
        self.rpc.info(deadline)
    }
    pub fn read(&mut self, maximum: usize, deadline: Deadline) -> Result<ReadResult, NetworkError> {
        if maximum == 0 || maximum > MAX_INLINE {
            return Err(ProtocolError::Limit.into());
        }
        let mut tail = [0; 8];
        tail[..4].copy_from_slice(&(maximum as u32).to_le_bytes());
        let m = self.rpc.call(TCP_STREAM_READ, &[&tail], None, deadline)?;
        match decode_read_reply(&m.bytes()[32..], m.handle_count(), maximum) {
            Ok(ReadReply::Data(_)) => Ok(ReadResult::Data(ReadData { message: m })),
            Ok(ReadReply::Eof) => Ok(ReadResult::Eof),
            Ok(ReadReply::Error(status)) => Err(status_error(status)),
            Err(error) => self.rpc.invalid(error),
        }
    }
    /// Returns bytes accepted into the local send queue, possibly a prefix.
    pub fn write(&mut self, bytes: &[u8], deadline: Deadline) -> Result<usize, NetworkError> {
        if bytes.len() > MAX_INLINE {
            return Err(ProtocolError::Limit.into());
        }
        let mut tail = [0; 8];
        tail[..4].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
        let m = self
            .rpc
            .call(TCP_STREAM_WRITE, &[&tail, bytes], None, deadline)?;
        match decode_write_reply(&m.bytes()[32..], m.handle_count(), bytes.len()) {
            Ok(WriteReply::Accepted(count)) => Ok(count),
            Ok(WriteReply::Error(status)) => Err(status_error(status)),
            Err(error) => self.rpc.invalid(error),
        }
    }
    /// Wait for local capacity, without replaying an uncertain write.
    /// `accepted` counts only acknowledged local-queue prefixes; a reply loss or
    /// invalid reply may hide additional accepted bytes. Requires OBSERVE when
    /// backpressure occurs. The original deadline is preserved throughout.
    pub fn write_all(&mut self, bytes: &[u8], deadline: Deadline) -> Result<(), WriteAllError> {
        let mut accepted = 0;
        let mut watch: Option<Observation> = None;
        while accepted < bytes.len() {
            let end = accepted + (bytes.len() - accepted).min(MAX_INLINE);
            match self.write(&bytes[accepted..end], deadline) {
                Ok(count) => accepted += count,
                Err(NetworkError::WouldBlock) => {
                    let wait = (|| {
                        if watch.is_none() {
                            watch = Some(self.observe(
                                signals::WRITABLE | signals::ERROR,
                                0,
                                deadline,
                            )?);
                        }
                        let observer = watch.as_mut().unwrap();
                        if observer.dequeued {
                            self.rearm(observer, deadline)?;
                        }
                        let event = observer.receive(deadline)?;
                        if event.error != 0 {
                            return Err(status_error(event.error));
                        }
                        Ok(())
                    })();
                    if let Err(error) = wait {
                        return Err(WriteAllError { error, accepted });
                    }
                }
                Err(error) => return Err(WriteAllError { error, accepted }),
            }
        }
        Ok(())
    }
    /// Wait for data or EOF with the original deadline through every retry.
    pub fn read_wait(
        &mut self,
        maximum: usize,
        deadline: Deadline,
    ) -> Result<ReadResult, NetworkError> {
        let mut watch: Option<Observation> = None;
        loop {
            match self.read(maximum, deadline) {
                Err(NetworkError::WouldBlock) => {
                    if watch.is_none() {
                        watch = Some(self.observe(
                            signals::READABLE | signals::ERROR | signals::PEER_CLOSED,
                            0,
                            deadline,
                        )?);
                    }
                    let observer = watch.as_mut().unwrap();
                    if observer.dequeued {
                        self.rearm(observer, deadline)?;
                    }
                    let event = observer.receive(deadline)?;
                    if event.error != 0 && event.signals & signals::READABLE == 0 {
                        return Err(status_error(event.error));
                    }
                }
                result => return result,
            }
        }
    }
    pub fn shutdown_send(&mut self, deadline: Deadline) -> Result<(), NetworkError> {
        self.rpc.simple(TCP_STREAM_SHUTDOWN_SEND, deadline)
    }
    pub fn shutdown_receive(&mut self, deadline: Deadline) -> Result<(), NetworkError> {
        self.rpc.simple(TCP_STREAM_SHUTDOWN_RECEIVE, deadline)
    }
    pub fn abort(&mut self, deadline: Deadline) -> Result<(), NetworkError> {
        self.rpc.simple(TCP_STREAM_ABORT, deadline)?;
        self.rpc.retire();
        Ok(())
    }
    pub fn observe(
        &mut self,
        mask: u32,
        key: u64,
        deadline: Deadline,
    ) -> Result<Observation, NetworkError> {
        self.rpc.observe(mask, key, deadline)
    }
    pub fn rearm(
        &mut self,
        watch: &mut Observation,
        deadline: Deadline,
    ) -> Result<(), NetworkError> {
        self.rpc.rearm(watch, deadline)
    }
    pub fn cancel(
        &mut self,
        watch: &mut Observation,
        deadline: Deadline,
    ) -> Result<(), NetworkError> {
        self.rpc.cancel(watch, deadline)
    }
}
impl TcpListener {
    pub fn new(endpoint: OwnedHandle) -> Result<Self, NetworkError> {
        Ok(Self {
            rpc: Rpc::new(endpoint, PROTOCOL_TCP_LISTENER)?,
        })
    }
    pub fn is_retired(&self) -> bool {
        self.rpc.retired()
    }
    pub fn info(&mut self, deadline: Deadline) -> Result<Info, NetworkError> {
        self.rpc.info(deadline)
    }
    pub fn accept(&mut self, rights: u32, deadline: Deadline) -> Result<TcpStream, NetworkError> {
        valid_rights(rights, stream_rights::ALL)?;
        let mut body = [0; 8];
        body[..4].copy_from_slice(&rights.to_le_bytes());
        let reply = self
            .rpc
            .call(TCP_LISTENER_ACCEPT, &[&body], None, deadline)?;
        let cap = received_endpoint(&mut self.rpc, reply, rights)?;
        TcpStream::new(cap)
    }
    pub fn accept_wait(
        &mut self,
        rights: u32,
        deadline: Deadline,
    ) -> Result<TcpStream, NetworkError> {
        let mut watch: Option<Observation> = None;
        loop {
            match self.accept(rights, deadline) {
                Err(NetworkError::WouldBlock) => {
                    if watch.is_none() {
                        watch = Some(self.observe(
                            signals::ACCEPTABLE | signals::ERROR | signals::PEER_CLOSED,
                            0,
                            deadline,
                        )?);
                    }
                    let observer = watch.as_mut().unwrap();
                    if observer.dequeued {
                        self.rearm(observer, deadline)?;
                    }
                    let event = observer.receive(deadline)?;
                    if event.error != 0 {
                        return Err(status_error(event.error));
                    }
                    if event.signals & signals::PEER_CLOSED != 0 {
                        return Err(NetworkError::Closed);
                    }
                }
                result => return result,
            }
        }
    }
    pub fn observe(
        &mut self,
        mask: u32,
        key: u64,
        deadline: Deadline,
    ) -> Result<Observation, NetworkError> {
        self.rpc.observe(mask, key, deadline)
    }
    pub fn rearm(
        &mut self,
        watch: &mut Observation,
        deadline: Deadline,
    ) -> Result<(), NetworkError> {
        self.rpc.rearm(watch, deadline)
    }
    pub fn cancel(
        &mut self,
        watch: &mut Observation,
        deadline: Deadline,
    ) -> Result<(), NetworkError> {
        self.rpc.cancel(watch, deadline)
    }
}
/// Independent event receiver; its timeout leaves the watch registered. Drop
/// closes the receiver, allowing the service to remove the watch without a
/// blocking control RPC. The service keeps at most one unsent event. Historical
/// packets may still occupy the existing bounded kernel channel queue.
pub struct Observation {
    receiver: Option<Channel>,
    subscription: u64,
    key: u64,
    source: u64,
    protocol: u32,
    last_revision: u64,
    dequeued: bool,
}
impl Observation {
    pub fn id(&self) -> u64 {
        self.subscription
    }
    pub fn as_handle(&self) -> Option<BorrowedHandle<'_, Untyped>> {
        self.receiver.as_ref().map(|e| e.as_handle().as_untyped())
    }
    pub fn receive(&mut self, deadline: Deadline) -> Result<Readiness, NetworkError> {
        let receiver = self.receiver.as_ref().ok_or(NetworkError::Retired)?;
        if self.dequeued {
            return Err(NetworkError::NotSent(Error::Busy));
        }
        let m = owned::receive(receiver.as_handle().as_untyped(), deadline)
            .map_err(NetworkError::NotSent)?;
        let event = match decode_event(m.bytes(), m.handle_count()) {
            Ok(e) => e,
            Err(error) => {
                self.receiver.take();
                return Err(NetworkError::InvalidReply(error));
            }
        };
        if event.subscription != self.subscription
            || event.key != self.key
            || event.revision < self.last_revision
            || event.signals & !signal_mask(self.protocol)? != 0
        {
            self.receiver.take();
            return Err(NetworkError::InvalidReply(ProtocolError::Malformed));
        }
        self.last_revision = event.revision;
        self.dequeued = true;
        Ok(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoints_and_reserved_bytes() {
        let address = Endpoint::v4([127, 0, 0, 1], 443);
        assert_eq!(Endpoint::decode(&address.encode()), Ok(address));
        let mut bad = address.encode();
        bad[12] = 1;
        assert!(Endpoint::decode(&bad).is_err());
        bad = address.encode();
        bad[4] = 1;
        assert!(Endpoint::decode(&bad).is_err());
        let ipv6 = Endpoint::v6([0xff; 16], 9, 80);
        assert_eq!(Endpoint::decode(&ipv6.encode()), Ok(ipv6));
        for n in 0..24 {
            assert!(Endpoint::decode(&address.encode()[..n]).is_err());
        }
    }
    #[test]
    fn event_and_info_validation() {
        let e = Readiness {
            subscription: 4,
            key: 7,
            revision: 1,
            signals: signals::WRITABLE,
            error: 0,
        };
        let wire = encode_event(e).unwrap();
        assert_eq!(decode_event(&wire, 0), Ok(e));
        for n in 0..64 {
            assert!(decode_event(&wire[..n], 0).is_err());
        }
        assert!(decode_event(&wire, 1).is_err());
        let mut bad = wire;
        bad[63] = 0x7f;
        assert!(decode_event(&bad, 0).is_err());
        let info = Info {
            rights: stream_rights::ALL,
            signals: signals::CONNECTED,
            error: 0,
            revision: 8,
            local: Some(Endpoint::v4([127, 0, 0, 1], 123)),
            peer: None,
        };
        let wire = encode_info(PROTOCOL_TCP_STREAM, info).unwrap();
        assert_eq!(decode_info(PROTOCOL_TCP_STREAM, &wire), Ok(info));
        let mut bad = wire;
        bad[72] = 0;
        assert!(decode_info(PROTOCOL_TCP_STREAM, &bad).is_err());
    }
    #[test]
    fn request_lengths_and_authority_fields() {
        let m = request_message(
            PROTOCOL_TCP_STREAM,
            TCP_STREAM_WRITE,
            1,
            Deadline::FOREVER,
            &[&[3, 0, 0, 0, 0, 0, 0, 0], b"abc"],
            0,
        )
        .unwrap();
        assert!(matches!(
            decode_request(m.bytes(), 0).unwrap().operation,
            Operation::Write { bytes: b"abc" }
        ));
        assert!(decode_request(m.bytes(), 1).is_err());
        let mut bad = m.bytes().to_vec();
        bad[44] = 1;
        assert!(decode_request(&bad, 0).is_err());
        let m = request_message(
            PROTOCOL_TCP_STREAM,
            TCP_STREAM_OBSERVE,
            2,
            Deadline::FOREVER,
            &[&[8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            1,
        )
        .unwrap();
        assert!(decode_request(m.bytes(), 1).is_err());
    }
}
