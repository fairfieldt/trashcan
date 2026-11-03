//! Bounded resolver jobs share the service queue and observer implementation.
use crate::{
    native, observer,
    service::{Daemon, Factory},
    wait::Registration,
};
use dross_rt::{
    Error,
    deadline::Deadline,
    owned::{self, Message, OwnedHandle},
    protocol::{resolver as wire, schema::*},
    sys,
};
use oscar_resolver::{Error as ResolveError, JobId, State};
pub struct Query {
    endpoint: OwnedHandle,
    job: JobId,
    rights: u32,
    reply: Option<Message>,
    last_id: u64,
    revision: u64,
    state: State,
    observer: Option<observer::Observer>,
    wait: Option<Registration>,
    dead: bool,
}
pub fn status(e: ResolveError) -> i32 {
    match e {
        ResolveError::WouldBlock => STATUS_WOULD_BLOCK,
        ResolveError::Invalid => STATUS_MALFORMED,
        ResolveError::NoResources => STATUS_LIMIT,
        ResolveError::Unreachable => STATUS_UNREACHABLE,
        ResolveError::TimedOut => STATUS_TIMED_OUT,
        ResolveError::Canceled => STATUS_CANCELED,
        ResolveError::NameNotFound => STATUS_NAME_NOT_FOUND,
        ResolveError::NoData => STATUS_NO_DATA,
        ResolveError::TemporaryFailure => STATUS_TEMPORARY_FAILURE,
        ResolveError::InvalidResponse => STATUS_DNS_RESPONSE_INVALID,
        ResolveError::BadJob => STATUS_BAD_STATE,
    }
}
impl Query {
    fn view(&self) -> observer::View {
        let (signals, error) = match self.state {
            State::Pending => (0, 0),
            State::Complete => (64, 0),
            State::Failed(e) => (96, status(e)),
            State::Canceled => (96, STATUS_CANCELED),
        };
        observer::View {
            revision: self.revision,
            signals,
            error,
        }
    }
    fn refresh(&mut self, r: &oscar_resolver::Resolver) {
        let Ok(state) = r.state(self.job) else {
            self.dead = true;
            return;
        };
        if state != self.state {
            self.state = state;
            match self.revision.checked_add(1) {
                Some(r) => self.revision = r,
                None => self.dead = true,
            }
        }
        let view = self.view();
        observer::notify(&mut self.observer, view);
    }
}
fn flush(endpoint: &OwnedHandle, reply: &mut Option<Message>, dead: &mut bool) {
    if let Some(m) = reply.take() {
        match owned::try_send(endpoint.as_handle(), m) {
            Ok(()) => {}
            Err(e) if e.error == Error::ShouldWait => *reply = Some(e.message),
            Err(_) => *dead = true,
        }
    }
}
fn decode_error(e: dross_rt::protocol::ProtocolError) -> i32 {
    use dross_rt::protocol::ProtocolError as E;
    match e {
        E::Unsupported => STATUS_UNSUPPORTED,
        E::Allocation | E::Limit => STATUS_LIMIT,
        _ => STATUS_MALFORMED,
    }
}
fn factory(d: &mut Daemon, f: &mut Factory, m: Message) {
    let h = match native::valid_request(&m, PROTOCOL_RESOLVER, &mut f.last_id) {
        Ok(h) => h,
        Err(Some(h)) => {
            f.reply = native::error(h, STATUS_UNSUPPORTED).ok();
            return;
        }
        Err(None) => {
            f.dead = true;
            return;
        }
    };
    let fallback = match native::error(h, STATUS_LIMIT) {
        Ok(m) => m,
        Err(_) => {
            f.dead = true;
            return;
        }
    };
    let result = (|| -> Result<Message, i32> {
        let req = wire::decode_request(m.bytes(), m.handle_count()).map_err(decode_error)?;
        if native::u64at(&m.bytes()[32..], 0) <= sys::clock() {
            return Err(STATUS_TIMED_OUT);
        }
        match req.operation {
            wire::Operation::Describe => {
                let mut out = native::reply(h, 20, 0).map_err(|_| STATUS_LIMIT)?;
                for (at, value) in [
                    (36, f.rights),
                    (40, wire::MAX_NAME as u32),
                    (44, wire::MAX_RECORDS as u32),
                    (48, 3),
                ] {
                    out.bytes_mut()[at..at + 4].copy_from_slice(&value.to_le_bytes());
                }
                Ok(out)
            }
            wire::Operation::Lookup { name, kind, rights } => {
                if f.rights & 1 == 0 {
                    return Err(STATUS_DENIED);
                }
                if d.queries.len() >= oscar_resolver::MAX_JOBS {
                    return Err(STATUS_LIMIT);
                }
                let (endpoint, client, wait) =
                    native::prepare_transport(d.queue.as_handle().as_raw())?;
                let mut out = native::reply(h, 8, 1).map_err(|_| STATUS_LIMIT)?;
                out.bytes_mut()[36..40].copy_from_slice(&rights.to_le_bytes());
                out.try_push_handle(client).map_err(|_| STATUS_LIMIT)?;
                let kind = match kind {
                    wire::QueryType::A => oscar_dns_wire::QueryType::A,
                    wire::QueryType::Aaaa => oscar_dns_wire::QueryType::Aaaa,
                };
                let job = d
                    .resolver
                    .as_mut()
                    .ok_or(STATUS_UNAVAILABLE)?
                    .start(
                        name,
                        kind,
                        native::u64at(&m.bytes()[32..], 0),
                        sys::clock(),
                        &mut d.engine,
                    )
                    .map_err(status)?;
                d.queries.push(Query {
                    endpoint,
                    job,
                    rights,
                    reply: None,
                    last_id: 0,
                    revision: 1,
                    state: State::Pending,
                    observer: None,
                    wait: Some(wait),
                    dead: false,
                });
                Ok(out)
            }
            _ => Err(STATUS_UNSUPPORTED),
        }
    })();
    f.reply = Some(result.unwrap_or_else(|s| {
        let mut m = fallback;
        m.bytes_mut()[32..36].copy_from_slice(&s.to_le_bytes());
        m
    }));
}
fn handle(d: &mut Daemon, q: &mut Query, mut m: Message) {
    let h = match native::valid_request(&m, PROTOCOL_RESOLVE_QUERY, &mut q.last_id) {
        Ok(h) => h,
        Err(Some(h)) => {
            q.reply = native::error(h, STATUS_UNSUPPORTED).ok();
            return;
        }
        Err(None) => {
            q.dead = true;
            return;
        }
    };
    let fallback = match native::error(h, STATUS_LIMIT) {
        Ok(m) => m,
        Err(_) => {
            q.dead = true;
            return;
        }
    };
    let result = (|| -> Result<Message, i32> {
        let req = wire::decode_request(m.bytes(), m.handle_count()).map_err(decode_error)?;
        if native::u64at(&m.bytes()[32..], 0) <= sys::clock() {
            return Err(STATUS_TIMED_OUT);
        }
        let need = match req.operation {
            wire::Operation::Collect => 1,
            wire::Operation::Cancel => 4,
            wire::Operation::Info => 8,
            wire::Operation::Observe { .. }
            | wire::Operation::Rearm { .. }
            | wire::Operation::Unobserve { .. } => 2,
            _ => return Err(STATUS_UNSUPPORTED),
        };
        if q.rights & need == 0 {
            return Err(STATUS_DENIED);
        }
        q.refresh(d.resolver.as_ref().ok_or(STATUS_UNAVAILABLE)?);
        match req.operation {
            wire::Operation::Collect => {
                let r = d.resolver.as_ref().ok_or(STATUS_UNAVAILABLE)?;
                let answer = r.collect(q.job).map_err(status)?;
                let kind = match r.query_type(q.job).map_err(status)? {
                    oscar_dns_wire::QueryType::A => wire::QueryType::A,
                    oscar_dns_wire::QueryType::Aaaa => wire::QueryType::Aaaa,
                };
                let prefix = wire::encode_answers_prefix(
                    kind,
                    answer.answer.addresses.len(),
                    answer.remaining_ttl(sys::clock()),
                    answer.expires_at,
                )
                .map_err(decode_error)?;
                let mut out = native::reply(h, 24 + 24 * answer.answer.addresses.len(), 0)
                    .map_err(|_| STATUS_LIMIT)?;
                out.bytes_mut()[32..56].copy_from_slice(&prefix);
                for (bytes, address) in out.bytes_mut()[56..]
                    .chunks_exact_mut(24)
                    .zip(&answer.answer.addresses)
                {
                    use oscar_dns_wire::Address;
                    let e = match address {
                        Address::V4(a) => dross_rt::protocol::network::Endpoint::v4(*a, 0),
                        Address::V6(a) => dross_rt::protocol::network::Endpoint::v6(*a, 0, 0),
                    };
                    bytes.copy_from_slice(&e.encode());
                }
                Ok(out)
            }
            wire::Operation::Info => {
                let state = match q.state {
                    State::Pending => wire::QueryState::Pending,
                    State::Complete => wire::QueryState::Complete,
                    State::Failed(_) => wire::QueryState::Failed,
                    State::Canceled => wire::QueryState::Canceled,
                };
                let expires_at = d
                    .resolver
                    .as_ref()
                    .and_then(|r| r.collect(q.job).ok())
                    .map(|r| r.expires_at)
                    .unwrap_or(0);
                let body = wire::encode_info(wire::QueryInfo {
                    rights: q.rights,
                    state,
                    error: q.view().error,
                    revision: q.revision,
                    expires_at,
                })
                .map_err(decode_error)?;
                let mut out = native::reply(h, 32, 0).map_err(|_| STATUS_LIMIT)?;
                out.bytes_mut()[32..].copy_from_slice(&body);
                Ok(out)
            }
            wire::Operation::Cancel => {
                let out = native::reply(h, 4, 0).map_err(|_| STATUS_LIMIT)?;
                d.resolver
                    .as_mut()
                    .ok_or(STATUS_UNAVAILABLE)?
                    .cancel(q.job, &mut d.engine)
                    .map_err(status)?;
                q.refresh(d.resolver.as_ref().unwrap());
                Ok(out)
            }
            wire::Operation::Observe { mask, key } => {
                let view = q.view();
                observer::notify(&mut q.observer, view);
                if q.observer.is_some() {
                    return Err(STATUS_BAD_STATE);
                }
                if mask == 0 || mask & !112 != 0 {
                    return Err(STATUS_MALFORMED);
                }
                let mut out = native::reply(h, 16, 0).map_err(|_| STATUS_LIMIT)?;
                let id = d.observer_id().map_err(|_| STATUS_LIMIT)?;
                let sink = m.take_handle(0).ok_or(STATUS_MALFORMED)?;
                let watch = observer::Observer::register(
                    sink,
                    d.queue.as_handle().as_raw(),
                    id,
                    key,
                    mask,
                )?;
                out.bytes_mut()[40..48].copy_from_slice(&id.to_le_bytes());
                q.observer = Some(watch);
                observer::notify(&mut q.observer, view);
                Ok(out)
            }
            wire::Operation::Rearm {
                subscription,
                revision,
            } => {
                let out = native::reply(h, 4, 0).map_err(|_| STATUS_LIMIT)?;
                let view = q.view();
                observer::rearm(&mut q.observer, subscription, revision, view)?;
                Ok(out)
            }
            wire::Operation::Unobserve { subscription } => {
                let out = native::reply(h, 4, 0).map_err(|_| STATUS_LIMIT)?;
                observer::cancel(&mut q.observer, subscription)?;
                Ok(out)
            }
            _ => Err(STATUS_UNSUPPORTED),
        }
    })();
    q.reply = Some(result.unwrap_or_else(|s| {
        let mut m = fallback;
        m.bytes_mut()[32..36].copy_from_slice(&s.to_le_bytes());
        m
    }));
}
pub fn process(d: &mut Daemon) -> Result<(), Error> {
    if let Some(mut f) = d.resolver_factory.take() {
        flush(&f.endpoint, &mut f.reply, &mut f.dead);
        if !f.dead && f.reply.is_none() {
            match owned::receive(f.endpoint.as_handle(), Deadline::IMMEDIATE) {
                Ok(m) => factory(d, &mut f, m),
                Err(Error::ShouldWait | Error::TimedOut) => {}
                Err(_) => f.dead = true,
            }
        }
        if !f.dead {
            d.resolver_factory = Some(f);
        }
    }
    let count = d.queries.len();
    for _ in 0..count {
        let mut q = d.queries.remove(0);
        q.refresh(d.resolver.as_ref().unwrap());
        flush(&q.endpoint, &mut q.reply, &mut q.dead);
        if !q.dead && q.reply.is_none() {
            match owned::receive(q.endpoint.as_handle(), Deadline::IMMEDIATE) {
                Ok(m) => handle(d, &mut q, m),
                Err(Error::ShouldWait | Error::TimedOut) => {}
                Err(_) => q.dead = true,
            }
        }
        if q.dead {
            let _ = d.resolver.as_mut().unwrap().remove(q.job, &mut d.engine);
        } else {
            d.queries.push(q);
        }
    }
    Ok(())
}
pub fn refresh_all(d: &mut Daemon) {
    if let Some(r) = &d.resolver {
        for q in &mut d.queries {
            q.refresh(r);
        }
    }
}
pub fn wait(d: &mut Daemon, delivered: Option<u64>) -> bool {
    use dross_abi::event::{PEER_CLOSED, READABLE, WRITABLE};
    let queue = d.queue.as_handle().as_raw();
    let mut failed = false;
    if let Some(f) = &mut d.resolver_factory {
        if Registration::update(
            &mut f.wait,
            queue,
            f.endpoint.as_handle(),
            PEER_CLOSED
                | if f.reply.is_some() {
                    WRITABLE
                } else {
                    READABLE
                },
            delivered,
        )
        .is_err()
        {
            f.dead = true;
            failed = true;
        }
    }
    for q in &mut d.queries {
        if Registration::update(
            &mut q.wait,
            queue,
            q.endpoint.as_handle(),
            PEER_CLOSED
                | if q.reply.is_some() {
                    WRITABLE
                } else {
                    READABLE
                },
            delivered,
        )
        .is_err()
        {
            q.dead = true;
            failed = true;
        }
        if q.observer
            .as_mut()
            .is_some_and(|o| o.wait(queue, delivered).is_err())
        {
            q.observer.take();
        }
    }
    failed
}
