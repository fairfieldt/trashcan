//! Capability-scoped directory prototype. Clients have one reader and retire
//! their endpoint after an uncertain reply. Registration is a separately seeded
//! authority. Each successful connect transfers a fresh, narrowed channel pair;
//! the directory is absent from the resulting service data path.
use super::{
    ProtocolError,
    bootstrap::BootstrapV2,
    frame::{Frame, Header, get16, get32, get64},
    schema::*,
    valid_name,
};
use crate::{
    Error,
    deadline::{Deadline, Duration},
    owned::{self, Channel, Message, OwnedHandle},
    sys,
};
use alloc::{string::String, vec::Vec};

pub const ENDPOINT_RIGHTS: u32 = dross_abi::rights::READ
    | dross_abi::rights::WRITE
    | dross_abi::rights::TRANSFER
    | dross_abi::rights::DUPLICATE
    | dross_abi::rights::WAIT;
#[derive(Debug, PartialEq, Eq)]
pub struct Rule {
    pub name: String,
    pub protocol: u32,
    pub rights: u32,
}
/// Application status is a distinct domain, never a native syscall Error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientError {
    NotSent(Error),
    OutcomeUnknown(Error),
    Protocol(ProtocolError),
    InvalidReply(ProtocolError),
    Status(i32),
    Retired,
}
impl From<ProtocolError> for ClientError {
    fn from(e: ProtocolError) -> Self {
        Self::Protocol(e)
    }
}
impl From<Error> for ClientError {
    fn from(e: Error) -> Self {
        Self::NotSent(e)
    }
}
fn allocation(_: impl core::fmt::Debug) -> ProtocolError {
    ProtocolError::Allocation
}
fn copy_name(s: &str) -> Result<String, ProtocolError> {
    let mut n = String::new();
    n.try_reserve_exact(s.len()).map_err(allocation)?;
    n.push_str(s);
    Ok(n)
}
fn rights_valid(r: u32) -> bool {
    r & !ENDPOINT_RIGHTS == 0 && r & dross_abi::rights::TRANSFER != 0
}
fn put(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), ProtocolError> {
    out.try_reserve(bytes.len()).map_err(allocation)?;
    out.extend_from_slice(bytes);
    Ok(())
}
pub fn encode_rules(rules: &[Rule]) -> Result<Vec<u8>, ProtocolError> {
    if rules.len() > MAX_VIEW_NAMES {
        return Err(ProtocolError::Limit);
    }
    let mut out = Vec::new();
    put(&mut out, &(rules.len() as u16).to_le_bytes())?;
    put(&mut out, &[0, 0])?;
    for (i, r) in rules.iter().enumerate() {
        if !valid_name(&r.name)
            || r.protocol == 0
            || !rights_valid(r.rights)
            || rules[..i].iter().any(|p| p.name == r.name)
        {
            return Err(ProtocolError::Malformed);
        }
        put(&mut out, &r.protocol.to_le_bytes())?;
        put(&mut out, &r.rights.to_le_bytes())?;
        put(&mut out, &(r.name.len() as u16).to_le_bytes())?;
        put(&mut out, &[0, 0])?;
        put(&mut out, r.name.as_bytes())?;
    }
    Ok(out)
}
pub fn decode_rules(bytes: &[u8]) -> Result<Vec<Rule>, ProtocolError> {
    if bytes.len() < 4 || get16(bytes, 2) != 0 {
        return Err(ProtocolError::Malformed);
    }
    let count = get16(bytes, 0) as usize;
    if count > MAX_VIEW_NAMES {
        return Err(ProtocolError::Limit);
    }
    let mut out: Vec<Rule> = Vec::new();
    out.try_reserve_exact(count).map_err(allocation)?;
    let mut at = 4;
    for _ in 0..count {
        if bytes.len() - at < 12 {
            return Err(ProtocolError::Malformed);
        }
        let protocol = get32(bytes, at);
        let rights = get32(bytes, at + 4);
        let nl = get16(bytes, at + 8) as usize;
        let reserved = get16(bytes, at + 10);
        at += 12;
        if nl > bytes.len() - at {
            return Err(ProtocolError::Malformed);
        }
        let name =
            core::str::from_utf8(&bytes[at..at + nl]).map_err(|_| ProtocolError::Malformed)?;
        if reserved != 0
            || !valid_name(name)
            || protocol == 0
            || !rights_valid(rights)
            || out.iter().any(|r| r.name == name)
        {
            return Err(ProtocolError::Malformed);
        }
        out.push(Rule {
            name: copy_name(name)?,
            protocol,
            rights,
        });
        at += nl;
    }
    if at != bytes.len() {
        return Err(ProtocolError::Malformed);
    }
    Ok(out)
}
fn name_body(name: &str, protocol: u32, rights: u32) -> Result<Vec<u8>, ProtocolError> {
    if !valid_name(name) || protocol == 0 || !rights_valid(rights) {
        return Err(ProtocolError::Malformed);
    }
    let mut b = Vec::new();
    put(&mut b, &protocol.to_le_bytes())?;
    put(&mut b, &rights.to_le_bytes())?;
    put(&mut b, &(name.len() as u16).to_le_bytes())?;
    put(&mut b, &[0, 0])?;
    put(&mut b, name.as_bytes())?;
    Ok(b)
}
fn parse_name(b: &[u8]) -> Result<(&str, u32, u32), ProtocolError> {
    if b.len() < 12 || get16(b, 10) != 0 || b.len() != 12 + get16(b, 8) as usize {
        return Err(ProtocolError::Malformed);
    }
    let name = core::str::from_utf8(&b[12..]).map_err(|_| ProtocolError::Malformed)?;
    let p = get32(b, 0);
    let r = get32(b, 4);
    if !valid_name(name) || p == 0 || !rights_valid(r) {
        return Err(ProtocolError::Malformed);
    }
    Ok((name, p, r))
}
/// Build one framed message. Attachment owners are added separately; `handles`
/// is the final count, and send/receiver validation rejects a mismatch.
pub fn message(
    protocol: u32,
    major: u16,
    kind: u16,
    ordinal: u32,
    id: u64,
    body: &[u8],
    handles: usize,
) -> Result<Message, ProtocolError> {
    let h = Header::new(protocol, major, kind, ordinal, id, body.len(), handles)?;
    let mut bytes = [0u8; 32];
    h.encode(&mut bytes)?;
    let mut m = Message::try_from_bytes(&bytes).map_err(allocation)?;
    m.try_extend_bytes(body).map_err(allocation)?;
    Ok(m)
}
fn attach(m: &mut Message, h: OwnedHandle) -> Result<(), ProtocolError> {
    m.try_push_handle(h).map_err(|_| ProtocolError::Allocation)
}
fn reply(
    h: Header,
    status: i32,
    extra: &[u8],
    cap: Option<OwnedHandle>,
) -> Result<Message, ProtocolError> {
    let mut b = Vec::new();
    put(&mut b, &status.to_le_bytes())?;
    put(&mut b, extra)?;
    let mut m = message(
        PROTOCOL_DIRECTORY,
        MAJOR_DIRECTORY,
        KIND_REPLY,
        h.ordinal,
        h.request_id,
        &b,
        usize::from(cap.is_some()),
    )?;
    if let Some(c) = cap {
        attach(&mut m, c)?;
    }
    Ok(m)
}

pub struct Client {
    endpoint: Option<OwnedHandle>,
    next_id: u64,
}
pub struct Connection {
    pub protocol: u32,
    pub instance: u64,
    pub endpoint: OwnedHandle,
}
impl Client {
    /// `endpoint` must be fresh and have no other reader or previous requests.
    pub fn new(endpoint: OwnedHandle) -> Self {
        Self {
            endpoint: Some(endpoint),
            next_id: 1,
        }
    }
    pub fn is_retired(&self) -> bool {
        self.endpoint.is_none()
    }
    /// One in-flight request, one reader. A reply timeout or malformed reply
    /// retires this local endpoint; reconnect requires a separately supplied view.
    /// Attachment arguments are consumed on every path, including pre-send
    /// failure. Use owned::try_send directly when unsent-capability recovery is
    /// required. This RPC layer never retries a successfully sent operation.
    pub fn call(
        &mut self,
        ordinal: u32,
        body: &[u8],
        cap: Option<OwnedHandle>,
        deadline: Deadline,
    ) -> Result<Message, ClientError> {
        if self.next_id == u64::MAX {
            self.endpoint.take();
            return Err(ClientError::Retired);
        }
        let id = self.next_id;
        self.next_id += 1;
        let mut m = message(
            PROTOCOL_DIRECTORY,
            MAJOR_DIRECTORY,
            KIND_REQUEST,
            ordinal,
            id,
            body,
            usize::from(cap.is_some()),
        )?;
        if let Some(c) = cap {
            attach(&mut m, c)?;
        }
        let endpoint = self.endpoint.as_ref().ok_or(ClientError::Retired)?;
        loop {
            match owned::try_send(endpoint.as_handle(), m) {
                Ok(()) => break,
                Err(e) => {
                    m = e.message;
                    if e.error != Error::ShouldWait {
                        return Err(e.error.into());
                    }
                    if deadline.has_elapsed() {
                        return Err(Error::TimedOut.into());
                    }
                    sys::yield_now();
                }
            }
        }
        let result = (|| {
            let m = owned::receive(endpoint.as_handle(), deadline)
                .map_err(ClientError::OutcomeUnknown)?;
            let f = Frame::parse(m.bytes(), m.handle_count())
                .map_err(|e| ClientError::InvalidReply(e.into()))?;
            if !f.header.supported(PROTOCOL_DIRECTORY, MAJOR_DIRECTORY)
                || f.header.kind != KIND_REPLY
                || f.header.ordinal != ordinal
                || f.header.request_id != id
                || f.body.len() < 4
            {
                return Err(ClientError::InvalidReply(ProtocolError::Malformed));
            }
            let status = get32(f.body, 0) as i32;
            if status != STATUS_OK {
                if f.body.len() != 4 || m.handle_count() != 0 {
                    return Err(ClientError::InvalidReply(ProtocolError::Malformed));
                }
                return Err(ClientError::Status(status));
            }
            Ok(m)
        })();
        if matches!(
            result,
            Err(ClientError::OutcomeUnknown(_) | ClientError::InvalidReply(_))
        ) {
            self.endpoint.take();
        }
        result
    }
    pub fn connect(
        &mut self,
        name: &str,
        protocol: u32,
        rights: u32,
        deadline: Deadline,
    ) -> Result<Connection, ClientError> {
        let body = name_body(name, protocol, rights)?;
        let mut m = self.call(DIRECTORY_CONNECT, &body, None, deadline)?;
        let f = Frame::parse(m.bytes(), m.handle_count()).map_err(ProtocolError::from)?;
        if f.body.len() != 16
            || m.handle_count() != 1
            || get32(f.body, 4) != protocol
            || get64(f.body, 8) == 0
        {
            self.endpoint.take();
            return Err(ClientError::InvalidReply(ProtocolError::Malformed));
        }
        let instance = get64(f.body, 8);
        Ok(Connection {
            protocol,
            instance,
            endpoint: m.take_handle(0).ok_or(ProtocolError::Malformed)?,
        })
    }
    pub fn delegate(&mut self, rules: &[Rule], deadline: Deadline) -> Result<Client, ClientError> {
        let b = encode_rules(rules)?;
        let mut m = self.call(DIRECTORY_DELEGATE, &b, None, deadline)?;
        if m.bytes().len() != 36 || m.handle_count() != 1 {
            self.endpoint.take();
            return Err(ClientError::InvalidReply(ProtocolError::Malformed));
        }
        Ok(Client::new(
            m.take_handle(0).ok_or(ProtocolError::Malformed)?,
        ))
    }
    pub fn list(&mut self, deadline: Deadline) -> Result<Vec<Rule>, ClientError> {
        let m = self.call(DIRECTORY_LIST, &[], None, deadline)?;
        if m.handle_count() != 0 {
            self.endpoint.take();
            return Err(ClientError::InvalidReply(ProtocolError::Malformed));
        }
        match decode_rules(&m.bytes()[36..]) {
            Ok(r) => Ok(r),
            Err(e) => {
                self.endpoint.take();
                Err(ClientError::InvalidReply(e))
            }
        }
    }
    pub fn register(
        &mut self,
        name: &str,
        protocol: u32,
        rights: u32,
        provider: OwnedHandle,
        deadline: Deadline,
    ) -> Result<(), ClientError> {
        let b = name_body(name, protocol, rights)?;
        let m = self.call(DIRECTORY_REGISTER, &b, Some(provider), deadline)?;
        if m.bytes().len() != 36 || m.handle_count() != 0 {
            self.endpoint.take();
            return Err(ClientError::InvalidReply(ProtocolError::Malformed));
        }
        Ok(())
    }
}

struct View {
    id: u64,
    endpoint: OwnedHandle,
    rules: Vec<Rule>,
    admin: bool,
    last_id: u64,
    replies: Vec<Message>,
}
struct Provider {
    name: String,
    protocol: u32,
    rights: u32,
    instance: u64,
    endpoint: Option<OwnedHandle>,
}
struct Setup {
    id: u64,
    view: u64,
    provider: usize,
    instance: u64,
    request: Header,
    client: Option<OwnedHandle>,
    offer: Option<Message>,
    deadline: Deadline,
}
/// Single-threaded bounded broker. Pending handoffs expire after two seconds, or
/// the bootstrap's `handoff-ms` (u32) for a provider that can be busy for longer;
/// an unresponsive provider control connection is retired before its IDs can be
/// forgotten. Delivered service endpoints have no broker-held copies.
pub struct Server {
    views: Vec<View>,
    providers: Vec<Provider>,
    pending: Vec<Setup>,
    next: u64,
    notification: OwnedHandle,
    handoff: Duration,
}
impl Server {
    pub fn new() -> Result<Self, Error> {
        let notification = unsafe { OwnedHandle::from_raw(sys::notify_create()?) };
        Ok(Self {
            views: Vec::new(),
            providers: Vec::new(),
            pending: Vec::new(),
            next: 1,
            notification,
            handoff: Duration::from_secs(2),
        })
    }
    fn id(&mut self) -> Result<u64, ProtocolError> {
        let id = self.next;
        self.next = self.next.checked_add(1).ok_or(ProtocolError::Limit)?;
        Ok(id)
    }
    fn bind(&self, h: &OwnedHandle) -> Result<(), ProtocolError> {
        sys::channel_bind(
            h.as_handle().as_raw(),
            self.notification.as_handle().as_raw(),
            1,
        )
        .map_err(|_| ProtocolError::WrongProtocol)
    }
    pub fn add_view(
        &mut self,
        endpoint: OwnedHandle,
        rules: Vec<Rule>,
        admin: bool,
    ) -> Result<(), ProtocolError> {
        if self.views.len() >= MAX_VIEWS {
            return Err(ProtocolError::Limit);
        }
        encode_rules(&rules)?;
        self.bind(&endpoint)?;
        let id = self.id()?;
        self.views.try_reserve(1).map_err(allocation)?;
        let mut replies = Vec::new();
        replies
            .try_reserve_exact(MAX_PENDING_REPLIES)
            .map_err(allocation)?;
        self.views.push(View {
            id,
            endpoint,
            rules,
            admin,
            last_id: 0,
            replies,
        });
        Ok(())
    }
    pub fn add_provider(
        &mut self,
        name: &str,
        protocol: u32,
        rights: u32,
        endpoint: OwnedHandle,
    ) -> Result<(), ProtocolError> {
        if !valid_name(name) || protocol == 0 || !rights_valid(rights) {
            return Err(ProtocolError::Malformed);
        }
        self.bind(&endpoint)?;
        let index = self.providers.iter().position(|p| p.name == name);
        if index.is_none() && self.providers.len() >= MAX_PROVIDERS {
            return Err(ProtocolError::Limit);
        }
        let new_name = copy_name(name)?;
        if index.is_none() {
            self.providers.try_reserve(1).map_err(allocation)?;
        }
        let instance = self.id()?;
        if let Some(i) = index {
            self.retire_provider(i, STATUS_UNAVAILABLE);
            self.providers[i] = Provider {
                name: new_name,
                protocol,
                rights,
                instance,
                endpoint: Some(endpoint),
            };
        } else {
            self.providers.push(Provider {
                name: new_name,
                protocol,
                rights,
                instance,
                endpoint: Some(endpoint),
            });
        }
        Ok(())
    }
    /// Explicit seed format: provider/NAME cap (OPRV), service/NAME value u32
    /// protocol; view/LABEL cap (ODIR), rules/LABEL value encoded Rules; admin/
    /// LABEL cap (ODIR). No global namespace or registration handle is inherited.
    pub fn from_bootstrap(mut b: BootstrapV2) -> Result<Self, ProtocolError> {
        let mut s = Self::new().map_err(allocation)?;
        if let Some(v) = b.value("handoff-ms") {
            if v.len() != 4 {
                return Err(ProtocolError::Malformed);
            }
            s.handoff = Duration::from_millis(get32(v, 0) as u64);
        }
        while let Some((name, cap)) = b.take_next() {
            if let Some(n) = name.strip_prefix("provider/") {
                if cap.protocol != PROTOCOL_PROVIDER {
                    return Err(ProtocolError::WrongProtocol);
                }
                let mut key = String::new();
                key.try_reserve_exact(8 + n.len()).map_err(allocation)?;
                key.push_str("service/");
                key.push_str(n);
                let v = b.value(&key).ok_or(ProtocolError::Missing)?;
                if v.len() != 4 {
                    return Err(ProtocolError::Malformed);
                }
                s.add_provider(n, get32(v, 0), ENDPOINT_RIGHTS, cap.handle)?;
            } else if let Some(n) = name.strip_prefix("view/") {
                if cap.protocol != PROTOCOL_DIRECTORY {
                    return Err(ProtocolError::WrongProtocol);
                }
                let mut key = String::new();
                key.try_reserve_exact(6 + n.len()).map_err(allocation)?;
                key.push_str("rules/");
                key.push_str(n);
                let r = decode_rules(b.value(&key).ok_or(ProtocolError::Missing)?)?;
                s.add_view(cap.handle, r, false)?;
            } else if name.starts_with("admin/") {
                if cap.protocol != PROTOCOL_DIRECTORY {
                    return Err(ProtocolError::WrongProtocol);
                }
                s.add_view(cap.handle, Vec::new(), true)?;
            } else {
                return Err(ProtocolError::Malformed);
            }
        }
        if s.views.is_empty() {
            return Err(ProtocolError::Missing);
        }
        Ok(s)
    }
    fn queue(&mut self, view: u64, message: Message) {
        if let Some(v) = self.views.iter_mut().find(|v| v.id == view) {
            if v.replies.len() < MAX_PENDING_REPLIES {
                v.replies.push(message);
                return;
            }
            self.close_view(view);
        }
    }
    fn answer(
        &mut self,
        view: u64,
        h: Header,
        status: i32,
        extra: &[u8],
        cap: Option<OwnedHandle>,
    ) {
        match reply(h, status, extra, cap) {
            Ok(m) => self.queue(view, m),
            Err(_) => self.close_view(view),
        }
    }
    fn close_view(&mut self, id: u64) {
        if let Some(i) = self.views.iter().position(|v| v.id == id) {
            self.views.remove(i);
        }
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].view == id {
                if self.pending[i].offer.is_some() {
                    self.pending.remove(i);
                    continue;
                }
                self.pending[i].client.take();
                self.pending[i].view = 0;
            }
            i += 1;
        }
    }
    fn retire_provider(&mut self, index: usize, status: i32) {
        self.providers[index].endpoint.take();
        while let Some(i) = self.pending.iter().position(|p| p.provider == index) {
            let p = self.pending.remove(i);
            self.answer(p.view, p.request, status, &[], None);
        }
    }
    fn handle(&mut self, view: u64, mut m: Message) {
        let Some(vi) = self.views.iter().position(|v| v.id == view) else {
            return;
        };
        let f = match Frame::parse(m.bytes(), m.handle_count()) {
            Ok(f) => f,
            Err(_) => {
                self.close_view(view);
                return;
            }
        };
        let h = f.header;
        if h.kind != KIND_REQUEST || h.request_id <= self.views[vi].last_id {
            self.close_view(view);
            return;
        }
        self.views[vi].last_id = h.request_id;
        if !h.supported(PROTOCOL_DIRECTORY, MAJOR_DIRECTORY) {
            self.answer(view, h, STATUS_UNSUPPORTED, &[], None);
            return;
        }
        if self.views[vi].admin
            && h.ordinal != DIRECTORY_DESCRIBE
            && h.ordinal != DIRECTORY_REGISTER
        {
            self.answer(view, h, STATUS_DENIED, &[], None);
            return;
        }
        if h.ordinal == DIRECTORY_REGISTER && !self.views[vi].admin {
            self.answer(view, h, STATUS_DENIED, &[], None);
            return;
        }
        match h.ordinal {
            DIRECTORY_DESCRIBE => {
                if !f.body.is_empty() || m.handle_count() != 0 {
                    self.close_view(view);
                    return;
                }
                let ops: &[u32] = if self.views[vi].admin {
                    &[DIRECTORY_DESCRIBE, DIRECTORY_REGISTER]
                } else {
                    &[
                        DIRECTORY_DESCRIBE,
                        DIRECTORY_CONNECT,
                        DIRECTORY_DELEGATE,
                        DIRECTORY_LIST,
                    ]
                };
                let mut b = Vec::new();
                let result = (|| {
                    put(&mut b, &(ops.len() as u16).to_le_bytes())?;
                    put(&mut b, &[0, 0])?;
                    for op in ops {
                        put(&mut b, &op.to_le_bytes())?;
                    }
                    Ok::<_, ProtocolError>(())
                })();
                if result.is_err() {
                    self.answer(view, h, STATUS_LIMIT, &[], None)
                } else {
                    self.answer(view, h, STATUS_OK, &b, None)
                }
            }
            DIRECTORY_LIST => {
                if !f.body.is_empty() || m.handle_count() != 0 {
                    self.close_view(view);
                    return;
                }
                match encode_rules(&self.views[vi].rules) {
                    Ok(b) => self.answer(view, h, STATUS_OK, &b, None),
                    Err(_) => self.answer(view, h, STATUS_LIMIT, &[], None),
                }
            }
            DIRECTORY_DELEGATE => {
                if m.handle_count() != 0 {
                    self.close_view(view);
                    return;
                }
                let rules = match decode_rules(f.body) {
                    Ok(r) => r,
                    Err(_) => {
                        self.close_view(view);
                        return;
                    }
                };
                if rules.iter().any(|r| {
                    !self.views[vi].rules.iter().any(|a| {
                        a.name == r.name && a.protocol == r.protocol && r.rights & !a.rights == 0
                    })
                }) {
                    self.answer(view, h, STATUS_DENIED, &[], None);
                    return;
                }
                let result = (|| {
                    let (client, server) = Channel::pair().map_err(allocation)?;
                    let client = client
                        .try_duplicate(ENDPOINT_RIGHTS & !dross_abi::rights::DUPLICATE)
                        .map_err(allocation)?
                        .into_untyped();
                    self.add_view(server.into_untyped(), rules, false)?;
                    Ok::<_, ProtocolError>(client)
                })();
                match result {
                    Ok(c) => self.answer(view, h, STATUS_OK, &[], Some(c)),
                    Err(_) => self.answer(view, h, STATUS_LIMIT, &[], None),
                }
            }
            DIRECTORY_REGISTER => {
                if m.handle_count() != 1 {
                    self.close_view(view);
                    return;
                }
                let (name, protocol, rights) = match parse_name(f.body) {
                    Ok(v) => v,
                    Err(_) => {
                        self.close_view(view);
                        return;
                    }
                };
                let name = match copy_name(name) {
                    Ok(n) => n,
                    Err(_) => {
                        self.answer(view, h, STATUS_LIMIT, &[], None);
                        return;
                    }
                };
                let Some(c) = m.take_handle(0) else {
                    self.close_view(view);
                    return;
                };
                let status = match self.add_provider(&name, protocol, rights, c) {
                    Ok(()) => STATUS_OK,
                    Err(ProtocolError::WrongProtocol) => STATUS_MALFORMED,
                    Err(_) => STATUS_LIMIT,
                };
                self.answer(view, h, status, &[], None);
            }
            DIRECTORY_CONNECT => {
                if m.handle_count() != 0 {
                    self.close_view(view);
                    return;
                }
                let (name, protocol, rights) = match parse_name(f.body) {
                    Ok(v) => v,
                    Err(_) => {
                        self.close_view(view);
                        return;
                    }
                };
                if !self.views[vi]
                    .rules
                    .iter()
                    .any(|r| r.name == name && r.protocol == protocol && rights & !r.rights == 0)
                {
                    self.answer(view, h, STATUS_DENIED, &[], None);
                    return;
                }
                let Some(pi) = self
                    .providers
                    .iter()
                    .position(|p| p.name == name && p.protocol == protocol && p.endpoint.is_some())
                else {
                    self.answer(view, h, STATUS_UNAVAILABLE, &[], None);
                    return;
                };
                if rights & !self.providers[pi].rights != 0 {
                    self.answer(view, h, STATUS_UNAVAILABLE, &[], None);
                    return;
                }
                if self.pending.len() >= MAX_PENDING_SETUPS
                    || self.pending.iter().filter(|p| p.view == view).count()
                        + self.views[vi].replies.len()
                        >= MAX_CLIENT_PENDING
                {
                    self.answer(view, h, STATUS_LIMIT, &[], None);
                    return;
                }
                let result = (|| {
                    self.pending.try_reserve(1).map_err(allocation)?;
                    let (client, server) = Channel::pair().map_err(allocation)?;
                    let client = client
                        .try_duplicate(rights)
                        .map_err(allocation)?
                        .into_untyped();
                    let server = server.into_untyped();
                    let id = self.id()?;
                    let instance = self.providers[pi].instance;
                    let mut body = [0u8; 16];
                    body[..4].copy_from_slice(&protocol.to_le_bytes());
                    body[4..8].copy_from_slice(&rights.to_le_bytes());
                    body[8..].copy_from_slice(&instance.to_le_bytes());
                    let mut offer = message(
                        PROTOCOL_PROVIDER,
                        MAJOR_PROVIDER,
                        KIND_REQUEST,
                        PROVIDER_OFFER,
                        id,
                        &body,
                        1,
                    )?;
                    attach(&mut offer, server)?;
                    Ok::<_, ProtocolError>(Setup {
                        id,
                        view,
                        provider: pi,
                        instance,
                        request: h,
                        client: Some(client),
                        offer: Some(offer),
                        deadline: Deadline::from_now(self.handoff),
                    })
                })();
                match result {
                    Ok(p) => self.pending.push(p),
                    Err(_) => self.answer(view, h, STATUS_LIMIT, &[], None),
                }
            }
            _ => self.answer(view, h, STATUS_UNSUPPORTED, &[], None),
        }
    }
    /// Nonblocking bounded work: at most four receives per connection per step.
    pub fn step(&mut self) {
        let mut i = 0;
        while i < self.views.len() {
            let id = self.views[i].id;
            let mut close = false;
            while !self.views[i].replies.is_empty() {
                let m = self.views[i].replies.remove(0);
                match owned::try_send(self.views[i].endpoint.as_handle(), m) {
                    Ok(()) => {}
                    Err(e) if e.error == Error::ShouldWait => {
                        self.views[i].replies.insert(0, e.message);
                        break;
                    }
                    Err(_) => {
                        close = true;
                        break;
                    }
                }
            }
            if close {
                self.close_view(id);
                continue;
            }
            for _ in 0..4 {
                let Some(v) = self.views.iter().find(|v| v.id == id) else {
                    break;
                };
                match owned::receive(v.endpoint.as_handle(), Deadline::IMMEDIATE) {
                    Ok(m) => self.handle(id, m),
                    Err(Error::ShouldWait | Error::TimedOut) => break,
                    Err(_) => {
                        self.close_view(id);
                        break;
                    }
                }
            }
            if self.views.get(i).is_some_and(|v| v.id == id) {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.pending.len() {
            let pi = self.pending[i].provider;
            if self.pending[i].deadline.has_elapsed() {
                self.retire_provider(pi, STATUS_TIMED_OUT);
                continue;
            }
            if let Some(offer) = self.pending[i].offer.take() {
                let result = self.providers[pi]
                    .endpoint
                    .as_ref()
                    .map(|e| owned::try_send(e.as_handle(), offer));
                match result {
                    Some(Ok(())) => {}
                    Some(Err(e)) if e.error == Error::ShouldWait => {
                        self.pending[i].offer = Some(e.message)
                    }
                    _ => {
                        self.retire_provider(pi, STATUS_UNAVAILABLE);
                        continue;
                    }
                }
            }
            i += 1;
        }
        for pi in 0..self.providers.len() {
            for _ in 0..4 {
                let Some(endpoint) = self.providers[pi].endpoint.as_ref() else {
                    break;
                };
                let m = match owned::receive(endpoint.as_handle(), Deadline::IMMEDIATE) {
                    Ok(m) => m,
                    Err(Error::ShouldWait | Error::TimedOut) => break,
                    Err(_) => {
                        self.retire_provider(pi, STATUS_UNAVAILABLE);
                        break;
                    }
                };
                let f = match Frame::parse(m.bytes(), m.handle_count()) {
                    Ok(f) => f,
                    Err(_) => {
                        self.retire_provider(pi, STATUS_MALFORMED);
                        break;
                    }
                };
                let found = self.pending.iter().position(|p| {
                    p.provider == pi
                        && p.instance == self.providers[pi].instance
                        && p.id == f.header.request_id
                        && p.offer.is_none()
                });
                if !f.header.supported(PROTOCOL_PROVIDER, MAJOR_PROVIDER)
                    || f.header.kind != KIND_REPLY
                    || f.header.ordinal != PROVIDER_OFFER
                    || m.handle_count() != 0
                    || f.body.len() != 4
                    || found.is_none()
                {
                    self.retire_provider(pi, STATUS_MALFORMED);
                    break;
                }
                let p = self.pending.remove(found.unwrap());
                let status = get32(f.body, 0) as i32;
                if status == STATUS_OK {
                    let mut extra = [0u8; 12];
                    extra[..4].copy_from_slice(&self.providers[pi].protocol.to_le_bytes());
                    extra[4..].copy_from_slice(&p.instance.to_le_bytes());
                    if p.client.is_some() {
                        self.answer(p.view, p.request, STATUS_OK, &extra, p.client);
                    }
                } else {
                    self.answer(p.view, p.request, STATUS_UNAVAILABLE, &[], None);
                }
            }
        }
    }
    /// The prototype exits once all supplied/delegated views close and pending
    /// handoffs drain. Periodic 5ms wakeups also bound outbound backpressure.
    pub fn run(mut self) -> Result<(), Error> {
        loop {
            self.step();
            if self.views.is_empty() && self.pending.is_empty() {
                return Ok(());
            }
            match sys::notify_wait(
                self.notification.as_handle().as_raw(),
                Deadline::from_now(Duration::from_millis(5)).as_monotonic_ns(),
            ) {
                Ok(_) | Err(Error::TimedOut) => {}
                Err(e) => return Err(e),
            }
        }
    }
}

/// Provider-side validation and ACK helper; service ownership moves only once.
pub struct Offer {
    pub id: u64,
    pub protocol: u32,
    pub rights: u32,
    pub instance: u64,
    pub endpoint: OwnedHandle,
}
pub fn decode_offer(mut m: Message) -> Result<Offer, ProtocolError> {
    let f = Frame::parse(m.bytes(), m.handle_count())?;
    if !f.header.supported(PROTOCOL_PROVIDER, MAJOR_PROVIDER)
        || f.header.kind != KIND_REQUEST
        || f.header.ordinal != PROVIDER_OFFER
        || f.body.len() != 16
        || m.handle_count() != 1
    {
        return Err(ProtocolError::Malformed);
    }
    let (id, protocol, rights, instance) = (
        f.header.request_id,
        get32(f.body, 0),
        get32(f.body, 4),
        get64(f.body, 8),
    );
    if protocol == 0 || !rights_valid(rights) || instance == 0 {
        return Err(ProtocolError::Malformed);
    }
    Ok(Offer {
        id,
        protocol,
        rights,
        instance,
        endpoint: m.take_handle(0).ok_or(ProtocolError::Malformed)?,
    })
}
pub fn offer_reply(id: u64, status: i32) -> Result<Message, ProtocolError> {
    message(
        PROTOCOL_PROVIDER,
        MAJOR_PROVIDER,
        KIND_REPLY,
        PROVIDER_OFFER,
        id,
        &status.to_le_bytes(),
        0,
    )
}
