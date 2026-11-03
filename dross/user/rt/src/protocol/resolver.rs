//! Bounded asynchronous A and AAAA jobs through an explicit resolver capability.
//! Dropping a query cancels its I/O. COLLECT preserves the original expiry.
use super::{
    ProtocolError,
    frame::{Frame, Header, get16, get32, get64},
    network::{self, Address, Endpoint, NetworkError, Observation, Rpc},
    schema::*,
};
use crate::{
    deadline::Deadline,
    owned::{Message, OwnedHandle},
};
pub const LOOKUP: u32 = 1;
pub const MAX_NAME: usize = MAX_RESOLVER_NAME_BYTES;
pub const MAX_RECORDS: usize = MAX_RESOLVER_RECORDS;
pub mod rights {
    pub const COLLECT: u32 = 1;
    pub const OBSERVE: u32 = 2;
    pub const CANCEL: u32 = 4;
    pub const INFO: u32 = 8;
    pub const ALL: u32 = 15;
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryType {
    A,
    Aaaa,
}
impl QueryType {
    pub fn wire(self) -> u16 {
        match self {
            Self::A => 1,
            Self::Aaaa => 28,
        }
    }
    pub fn decode(v: u16) -> Result<Self, ProtocolError> {
        match v {
            1 => Ok(Self::A),
            28 => Ok(Self::Aaaa),
            _ => Err(ProtocolError::Unsupported),
        }
    }
    fn matches(self, e: Endpoint) -> bool {
        e.port == 0
            && match (self, e.address) {
                (Self::A, Address::V4(_)) => true,
                (Self::Aaaa, Address::V6 { scope: 0, .. }) => true,
                _ => false,
            }
    }
}
/// ASCII labels of 1..63 LDH/underscore bytes, with optional final dot.
/// The root is `.`. A 254-byte input must end in a dot.
pub fn valid_name(name: &str) -> bool {
    if name == "." {
        return true;
    }
    if name.is_empty() || name.len() > MAX_NAME || !name.is_ascii() {
        return false;
    }
    let n = name.strip_suffix('.').unwrap_or(name);
    !n.is_empty()
        && n.len() <= 253
        && n.split('.').all(|l| {
            !l.is_empty()
                && l.len() <= 63
                && l.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        })
}
#[derive(Debug, PartialEq, Eq)]
pub enum Operation<'a> {
    Describe,
    Lookup {
        name: &'a str,
        kind: QueryType,
        rights: u32,
    },
    Collect,
    Cancel,
    Info,
    Observe {
        mask: u32,
        key: u64,
    },
    Rearm {
        subscription: u64,
        revision: u64,
    },
    Unobserve {
        subscription: u64,
    },
}
#[derive(Debug)]
pub struct WireRequest<'a> {
    pub header: Header,
    pub deadline: Deadline,
    pub operation: Operation<'a>,
}
pub fn decode_request(bytes: &[u8], handles: usize) -> Result<WireRequest<'_>, ProtocolError> {
    let f = Frame::parse(bytes, handles)?;
    let h = f.header;
    if !h.supported(h.protocol, 1) || h.kind != KIND_REQUEST {
        return Err(ProtocolError::Unsupported);
    }
    if f.body.len() < 8 {
        return Err(ProtocolError::Malformed);
    }
    let b = f.body;
    let operation = match (h.protocol, h.ordinal) {
        (PROTOCOL_RESOLVER, RESOLVER_DESCRIBE) if b.len() == 8 && handles == 0 => {
            Operation::Describe
        }
        (PROTOCOL_RESOLVER, RESOLVER_LOOKUP) => {
            if b.len() < 16 || handles != 0 {
                return Err(ProtocolError::Malformed);
            }
            let kind = QueryType::decode(get16(b, 8))?;
            let count = get16(b, 10) as usize;
            let rights = get32(b, 12);
            if rights & !rights::ALL != 0 || b.len() != 16 + count {
                return Err(ProtocolError::Malformed);
            }
            let name = core::str::from_utf8(&b[16..]).map_err(|_| ProtocolError::Malformed)?;
            if !valid_name(name) {
                return Err(ProtocolError::Malformed);
            }
            Operation::Lookup { name, kind, rights }
        }
        (PROTOCOL_RESOLVE_QUERY, RESOLVE_QUERY_COLLECT) if b.len() == 8 && handles == 0 => {
            Operation::Collect
        }
        (PROTOCOL_RESOLVE_QUERY, RESOLVE_QUERY_CANCEL) if b.len() == 8 && handles == 0 => {
            Operation::Cancel
        }
        (PROTOCOL_RESOLVE_QUERY, 4..=7) => {
            let common = network::decode_request_as(bytes, handles, PROTOCOL_RESOLVE_QUERY)?;
            match common.operation {
                network::Operation::Info => Operation::Info,
                network::Operation::Observe { mask, key } => Operation::Observe { mask, key },
                network::Operation::Rearm {
                    subscription,
                    revision,
                } => Operation::Rearm {
                    subscription,
                    revision,
                },
                network::Operation::Cancel { subscription } => {
                    Operation::Unobserve { subscription }
                }
                _ => return Err(ProtocolError::Unsupported),
            }
        }
        (PROTOCOL_RESOLVER, 0..=1) | (PROTOCOL_RESOLVE_QUERY, 1 | 3..=7) => {
            return Err(ProtocolError::Malformed);
        }
        _ => return Err(ProtocolError::Unsupported),
    };
    Ok(WireRequest {
        header: h,
        deadline: Deadline::at_monotonic_ns(get64(b, 0)),
        operation,
    })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryState {
    Pending,
    Complete,
    Failed,
    Canceled,
}
impl QueryState {
    pub fn wire(self) -> u32 {
        match self {
            Self::Pending => 0,
            Self::Complete => 1,
            Self::Failed => 2,
            Self::Canceled => 3,
        }
    }
    fn decode(v: u32) -> Result<Self, ProtocolError> {
        match v {
            0 => Ok(Self::Pending),
            1 => Ok(Self::Complete),
            2 => Ok(Self::Failed),
            3 => Ok(Self::Canceled),
            _ => Err(ProtocolError::Malformed),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryInfo {
    pub rights: u32,
    pub state: QueryState,
    pub error: i32,
    pub revision: u64,
    pub expires_at: u64,
}
pub fn encode_info(i: QueryInfo) -> Result<[u8; 32], ProtocolError> {
    if i.rights & !rights::ALL != 0
        || i.revision == 0
        || match i.state {
            QueryState::Pending => i.error != 0 || i.expires_at != 0,
            QueryState::Complete => i.error != 0,
            QueryState::Failed => i.error >= 0 || i.error == STATUS_CANCELED || i.expires_at != 0,
            QueryState::Canceled => i.error != STATUS_CANCELED || i.expires_at != 0,
        }
    {
        return Err(ProtocolError::Malformed);
    }
    let mut b = [0; 32];
    b[4..8].copy_from_slice(&i.rights.to_le_bytes());
    b[8..12].copy_from_slice(&i.state.wire().to_le_bytes());
    b[12..16].copy_from_slice(&i.error.to_le_bytes());
    b[16..24].copy_from_slice(&i.revision.to_le_bytes());
    b[24..32].copy_from_slice(&i.expires_at.to_le_bytes());
    Ok(b)
}
pub fn decode_info(b: &[u8]) -> Result<QueryInfo, ProtocolError> {
    if b.len() != 32 || get32(b, 0) != 0 {
        return Err(ProtocolError::Malformed);
    }
    let i = QueryInfo {
        rights: get32(b, 4),
        state: QueryState::decode(get32(b, 8))?,
        error: get32(b, 12) as i32,
        revision: get64(b, 16),
        expires_at: get64(b, 24),
    };
    encode_info(i)?;
    Ok(i)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Description {
    pub rights: u32,
    pub max_name: u32,
    pub max_records: u32,
    pub query_types: u32,
}
pub fn decode_description(b: &[u8], handles: usize) -> Result<Description, ProtocolError> {
    if b.len() != 20
        || handles != 0
        || get32(b, 0) != 0
        || get32(b, 4) & !LOOKUP != 0
        || get32(b, 8) == 0
        || get32(b, 8) > MAX_NAME as u32
        || get32(b, 12) == 0
        || get32(b, 12) > MAX_RECORDS as u32
        || get32(b, 16) & !3 != 0
    {
        return Err(ProtocolError::Malformed);
    }
    Ok(Description {
        rights: get32(b, 4),
        max_name: get32(b, 8),
        max_records: get32(b, 12),
        query_types: get32(b, 16),
    })
}
pub fn encode_answers_prefix(
    kind: QueryType,
    count: usize,
    ttl_seconds: u32,
    expires_at: u64,
) -> Result<[u8; 24], ProtocolError> {
    if count == 0 || count > MAX_RECORDS {
        return Err(ProtocolError::Limit);
    }
    let mut b = [0; 24];
    b[4..6].copy_from_slice(&kind.wire().to_le_bytes());
    b[6..8].copy_from_slice(&(count as u16).to_le_bytes());
    b[8..12].copy_from_slice(&ttl_seconds.to_le_bytes());
    b[16..24].copy_from_slice(&expires_at.to_le_bytes());
    Ok(b)
}
#[derive(Debug, PartialEq, Eq)]
pub enum CollectReply<'a> {
    Ready(AnswerView<'a>),
    Error(i32),
}
#[derive(Debug, PartialEq, Eq)]
pub struct AnswerView<'a> {
    pub kind: QueryType,
    pub ttl_seconds: u32,
    pub expires_at: u64,
    addresses: &'a [u8],
}
impl AnswerView<'_> {
    pub fn len(&self) -> usize {
        self.addresses.len() / 24
    }
    pub fn is_empty(&self) -> bool {
        self.addresses.is_empty()
    }
    pub fn addresses(&self) -> impl Iterator<Item = Endpoint> + '_ {
        self.addresses
            .chunks_exact(24)
            .map(|b| Endpoint::decode(b).unwrap())
    }
}
pub fn decode_collect_reply(
    b: &[u8],
    handles: usize,
    expected: QueryType,
) -> Result<CollectReply<'_>, ProtocolError> {
    if b.len() < 4 || handles != 0 {
        return Err(ProtocolError::Malformed);
    }
    let status = get32(b, 0) as i32;
    if status > 0 {
        return Err(ProtocolError::Malformed);
    }
    if status != 0 {
        return if b.len() == 4 {
            Ok(CollectReply::Error(status))
        } else {
            Err(ProtocolError::Malformed)
        };
    }
    if b.len() < 24 || get32(b, 12) != 0 {
        return Err(ProtocolError::Malformed);
    }
    let kind = QueryType::decode(get16(b, 4))?;
    let count = get16(b, 6) as usize;
    if kind != expected || count == 0 || count > MAX_RECORDS || b.len() != 24 + count * 24 {
        return Err(ProtocolError::Malformed);
    }
    for bytes in b[24..].chunks_exact(24) {
        if !kind.matches(Endpoint::decode(bytes)?) {
            return Err(ProtocolError::Malformed);
        }
    }
    Ok(CollectReply::Ready(AnswerView {
        kind,
        ttl_seconds: get32(b, 8),
        expires_at: get64(b, 16),
        addresses: &b[24..],
    }))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveError {
    Client(NetworkError),
    NameNotFound,
    NoData,
    TemporaryFailure,
    ResponseInvalid,
    Canceled,
}
impl From<NetworkError> for ResolveError {
    fn from(e: NetworkError) -> Self {
        match e {
            NetworkError::Service(STATUS_NAME_NOT_FOUND) => Self::NameNotFound,
            NetworkError::Service(STATUS_NO_DATA) => Self::NoData,
            NetworkError::Service(STATUS_TEMPORARY_FAILURE) => Self::TemporaryFailure,
            NetworkError::Service(STATUS_DNS_RESPONSE_INVALID) => Self::ResponseInvalid,
            NetworkError::Service(STATUS_CANCELED) => Self::Canceled,
            e => Self::Client(e),
        }
    }
}
impl From<ProtocolError> for ResolveError {
    fn from(e: ProtocolError) -> Self {
        Self::Client(e.into())
    }
}
#[derive(Debug)]
pub struct Answers {
    message: Message,
    kind: QueryType,
    ttl_seconds: u32,
    expires_at: u64,
}
impl Answers {
    pub fn kind(&self) -> QueryType {
        self.kind
    }
    pub fn len(&self) -> usize {
        (self.message.bytes().len() - 56) / 24
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn expires_at_monotonic_ns(&self) -> u64 {
        self.expires_at
    }
    pub fn ttl_seconds_remaining(&self) -> u32 {
        self.ttl_seconds.min(
            (self.expires_at.saturating_sub(crate::sys::clock()) / 1_000_000_000)
                .min(u32::MAX as u64) as u32,
        )
    }
    pub fn addresses(&self) -> impl Iterator<Item = Endpoint> + '_ {
        self.message.bytes()[56..]
            .chunks_exact(24)
            .map(|b| Endpoint::decode(b).unwrap())
    }
}
#[derive(Debug)]
pub struct Resolver {
    rpc: Rpc,
}
impl Resolver {
    pub fn new(endpoint: OwnedHandle) -> Result<Self, ResolveError> {
        Ok(Self {
            rpc: Rpc::new(endpoint, PROTOCOL_RESOLVER)?,
        })
    }
    pub fn is_retired(&self) -> bool {
        self.rpc.retired()
    }
    pub fn describe(&mut self, deadline: Deadline) -> Result<Description, ResolveError> {
        let m = self.rpc.call(RESOLVER_DESCRIBE, &[], None, deadline)?;
        match decode_description(&m.bytes()[32..], m.handle_count()) {
            Ok(d) => Ok(d),
            Err(e) => Err(self.rpc.invalid::<()>(e).unwrap_err().into()),
        }
    }
    pub fn lookup(
        &mut self,
        name: &str,
        kind: QueryType,
        requested: u32,
        deadline: Deadline,
    ) -> Result<ResolveQuery, ResolveError> {
        if !valid_name(name) || requested & !rights::ALL != 0 {
            return Err(ProtocolError::Malformed.into());
        }
        let mut b = [0; 8];
        b[..2].copy_from_slice(&kind.wire().to_le_bytes());
        b[2..4].copy_from_slice(&(name.len() as u16).to_le_bytes());
        b[4..8].copy_from_slice(&requested.to_le_bytes());
        let mut m = self
            .rpc
            .call(RESOLVER_LOOKUP, &[&b, name.as_bytes()], None, deadline)?;
        match network::decode_grant_reply(&m.bytes()[32..], m.handle_count(), requested) {
            Ok(network::GrantReply::Granted(_)) => Ok(ResolveQuery {
                rpc: Rpc::new(
                    m.take_handle(0).ok_or(ProtocolError::Malformed)?,
                    PROTOCOL_RESOLVE_QUERY,
                )?,
                kind,
            }),
            _ => Err(self
                .rpc
                .invalid::<()>(ProtocolError::Malformed)
                .unwrap_err()
                .into()),
        }
    }
    /// Uses a temporary query watch; use lookup for independent job ownership.
    pub fn lookup_wait(
        &mut self,
        name: &str,
        kind: QueryType,
        requested: u32,
        deadline: Deadline,
    ) -> Result<Answers, ResolveError> {
        if requested & (rights::COLLECT | rights::OBSERVE) != (rights::COLLECT | rights::OBSERVE) {
            return Err(NetworkError::Denied.into());
        }
        let mut job = self.lookup(name, kind, requested, deadline)?;
        job.wait(deadline).map_err(|e| match e {
            ResolveError::Client(e) => network::admitted_error(e).into(),
            e => e,
        })
    }
}
#[derive(Debug)]
pub struct ResolveQuery {
    rpc: Rpc,
    kind: QueryType,
}
impl ResolveQuery {
    pub fn new(endpoint: OwnedHandle, kind: QueryType) -> Result<Self, ResolveError> {
        Ok(Self {
            rpc: Rpc::new(endpoint, PROTOCOL_RESOLVE_QUERY)?,
            kind,
        })
    }
    pub fn is_retired(&self) -> bool {
        self.rpc.retired()
    }
    pub fn collect(&mut self, deadline: Deadline) -> Result<Answers, ResolveError> {
        let m = self.rpc.call(RESOLVE_QUERY_COLLECT, &[], None, deadline)?;
        match decode_collect_reply(&m.bytes()[32..], m.handle_count(), self.kind) {
            Ok(CollectReply::Ready(v)) => {
                let (kind, ttl_seconds, expires_at) = (v.kind, v.ttl_seconds, v.expires_at);
                Ok(Answers {
                    message: m,
                    kind,
                    ttl_seconds,
                    expires_at,
                })
            }
            Ok(CollectReply::Error(s)) => Err(network::status_error(s).into()),
            Err(e) => Err(self.rpc.invalid::<()>(e).unwrap_err().into()),
        }
    }
    pub fn info(&mut self, deadline: Deadline) -> Result<QueryInfo, ResolveError> {
        let m = self.rpc.call(RESOLVE_QUERY_INFO, &[], None, deadline)?;
        if m.handle_count() != 0 {
            return Err(self
                .rpc
                .invalid::<()>(ProtocolError::Malformed)
                .unwrap_err()
                .into());
        }
        match decode_info(&m.bytes()[32..]) {
            Ok(i) => Ok(i),
            Err(e) => Err(self.rpc.invalid::<()>(e).unwrap_err().into()),
        }
    }
    /// ACK stops I/O. INFO and COLLECT remain available with terminal Canceled.
    pub fn cancel(&mut self, deadline: Deadline) -> Result<(), ResolveError> {
        self.rpc
            .simple(RESOLVE_QUERY_CANCEL, deadline)
            .map_err(Into::into)
    }
    pub fn observe(
        &mut self,
        mask: u32,
        key: u64,
        deadline: Deadline,
    ) -> Result<Observation, ResolveError> {
        self.rpc.observe(mask, key, deadline).map_err(Into::into)
    }
    pub fn rearm(
        &mut self,
        watch: &mut Observation,
        deadline: Deadline,
    ) -> Result<(), ResolveError> {
        self.rpc.rearm(watch, deadline).map_err(Into::into)
    }
    pub fn unobserve(
        &mut self,
        watch: &mut Observation,
        deadline: Deadline,
    ) -> Result<(), ResolveError> {
        self.rpc.cancel(watch, deadline).map_err(Into::into)
    }
    /// Uses one temporary watch. An event timeout leaves the query owned; an
    /// uncertain control reply retires the transport, as for every RPC.
    pub fn wait(&mut self, deadline: Deadline) -> Result<Answers, ResolveError> {
        let mut watch: Option<Observation> = None;
        loop {
            match self.collect(deadline) {
                Err(ResolveError::Client(NetworkError::WouldBlock)) => {
                    if watch.is_none() {
                        watch = Some(self.observe(
                            network::signals::COMPLETE | network::signals::ERROR,
                            0,
                            deadline,
                        )?)
                    }
                    let observer = watch.as_mut().unwrap();
                    if observer.dequeued {
                        self.rearm(observer, deadline)?
                    }
                    observer.receive(deadline).map_err(ResolveError::from)?;
                }
                result => return result,
            }
        }
    }
}
