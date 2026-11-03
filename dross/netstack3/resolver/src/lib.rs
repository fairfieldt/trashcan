//! A bounded forwarding resolver over the shared Netstack3 engine. No ambient
//! resolver configuration, global socket lookup, cache, RPC, or blocking waits.
#![no_std]
extern crate alloc;
use alloc::vec::Vec;
use oscar_dns_wire::{self as wire, Answer, Name, QueryType, Question, Response};
use oscar_netstack3_engine::{
    Endpoint, Endpoint6, Engine, Error as NetError, Family, IpEndpoint, SocketId,
};

pub const MAX_JOBS: usize = 8;
pub const MAX_LIFETIME_NS: u64 = 10_000_000_000;
const UDP_RETRY_NS: [u64; 3] = [2_000_000_000, 4_000_000_000, 4_000_000_000];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    WouldBlock,
    Invalid,
    NoResources,
    Unreachable,
    TimedOut,
    Canceled,
    NameNotFound,
    NoData,
    TemporaryFailure,
    InvalidResponse,
    BadJob,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Pending,
    Complete,
    Failed(Error),
    Canceled,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JobId(u64);
#[derive(Debug)]
pub struct Resolved {
    pub answer: Answer,
    pub expires_at: u64,
}
impl Resolved {
    pub fn remaining_ttl(&self, now: u64) -> u32 {
        (self.expires_at.saturating_sub(now) / 1_000_000_000).min(u32::MAX as u64) as u32
    }
}
enum Phase {
    Udp {
        attempts: usize,
        next_send: u64,
        blocked: bool,
    },
    Tcp {
        sent: usize,
        received: usize,
        expected: Option<usize>,
    },
    Terminal,
}
struct Job {
    id: JobId,
    question: Question,
    deadline: u64,
    socket: Option<SocketId>,
    phase: Phase,
    request: Vec<u8>,
    buffer: Vec<u8>,
    history: Vec<Name>,
    aliases: usize,
    expires_at: u64,
    result: Option<Result<Resolved, Error>>,
}
pub struct Resolver {
    server: IpEndpoint,
    jobs: Vec<Job>,
    next: u64,
}
impl Resolver {
    pub fn new(server: IpEndpoint) -> Result<Self, Error> {
        let valid = match server {
            IpEndpoint::V4(e) => e.port != 0 && e.address != [0; 4] && e.address[0] < 224,
            IpEndpoint::V6(e) => {
                e.port != 0
                    && e.address != [0; 16]
                    && e.address[0] != 255
                    && if e.address[0] == 0xfe && e.address[1] & 0xc0 == 0x80 {
                        e.scope_id != 0
                    } else {
                        e.scope_id == 0
                    }
            }
        };
        if !valid {
            return Err(Error::Invalid);
        }
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(MAX_JOBS)
            .map_err(|_| Error::NoResources)?;
        Ok(Self {
            server,
            jobs,
            next: 1,
        })
    }
    pub fn server(&self) -> IpEndpoint {
        self.server
    }
    pub fn start(
        &mut self,
        name: &str,
        kind: QueryType,
        deadline: u64,
        now: u64,
        engine: &mut Engine,
    ) -> Result<JobId, Error> {
        if deadline <= now {
            return Err(Error::TimedOut);
        }
        if self.jobs.len() == MAX_JOBS {
            return Err(Error::NoResources);
        }
        let name = Name::from_ascii(name).map_err(|_| Error::Invalid)?;
        let question = Question {
            id: engine.random_u16().map_err(net_error)?,
            name,
            kind,
        };
        let request = question.encode().map_err(wire_error)?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(wire::MAX_MESSAGE + 2)
            .map_err(|_| Error::NoResources)?;
        buffer.resize(wire::MAX_MESSAGE + 2, 0);
        let mut history = Vec::new();
        history
            .try_reserve_exact(wire::MAX_ALIASES + 1)
            .map_err(|_| Error::NoResources)?;
        history.push(name);
        let id = JobId(self.next);
        let next = self.next.checked_add(1).ok_or(Error::NoResources)?;
        let socket = engine.create_udp(self.server.family()).map_err(net_error)?;
        if let Err(e) = engine.udp_bind(socket, wildcard(self.server.family())) {
            let _ = engine.abort(socket);
            return Err(net_error(e));
        }
        self.next = next;
        self.jobs.push(Job {
            id,
            question,
            deadline: deadline.min(now.saturating_add(MAX_LIFETIME_NS)),
            socket: Some(socket),
            phase: Phase::Udp {
                attempts: 0,
                next_send: now,
                blocked: false,
            },
            request,
            buffer,
            history,
            aliases: 0,
            expires_at: u64::MAX,
            result: None,
        });
        Ok(id)
    }
    fn get(&self, id: JobId) -> Result<&Job, Error> {
        self.jobs.iter().find(|j| j.id == id).ok_or(Error::BadJob)
    }
    pub fn state(&self, id: JobId) -> Result<State, Error> {
        Ok(match &self.get(id)?.result {
            None => State::Pending,
            Some(Ok(_)) => State::Complete,
            Some(Err(Error::Canceled)) => State::Canceled,
            Some(Err(e)) => State::Failed(*e),
        })
    }
    pub fn collect(&self, id: JobId) -> Result<&Resolved, Error> {
        match &self.get(id)?.result {
            None => Err(Error::WouldBlock),
            Some(Ok(a)) => Ok(a),
            Some(Err(e)) => Err(*e),
        }
    }
    pub fn query_type(&self, id: JobId) -> Result<QueryType, Error> {
        Ok(self.get(id)?.question.kind)
    }
    pub fn deadline(&self, id: JobId) -> Result<u64, Error> {
        Ok(self.get(id)?.deadline)
    }
    pub fn cancel(&mut self, id: JobId, engine: &mut Engine) -> Result<(), Error> {
        let job = self
            .jobs
            .iter_mut()
            .find(|j| j.id == id)
            .ok_or(Error::BadJob)?;
        job.finish(Err(Error::Canceled), engine);
        Ok(())
    }
    pub fn remove(&mut self, id: JobId, engine: &mut Engine) -> Result<(), Error> {
        let i = self
            .jobs
            .iter()
            .position(|j| j.id == id)
            .ok_or(Error::BadJob)?;
        let mut job = self.jobs.swap_remove(i);
        job.finish(Err(Error::Canceled), engine);
        Ok(())
    }
    pub fn next_deadline(&self) -> Option<u64> {
        self.jobs
            .iter()
            .filter(|j| j.result.is_none())
            .map(|j| match j.phase {
                Phase::Udp { next_send, .. } => j.deadline.min(next_send),
                _ => j.deadline,
            })
            .min()
    }
    /// Returns true only when bounded local work remains. Network progress is
    /// otherwise driven by Ethernet readiness and actual engine/query deadlines.
    pub fn pump(&mut self, now: u64, engine: &mut Engine) -> bool {
        let mut more = false;
        for job in &mut self.jobs {
            more |= job.pump(self.server, now, engine);
        }
        more
    }
    pub fn shutdown(&mut self, engine: &mut Engine) {
        for job in &mut self.jobs {
            job.finish(Err(Error::Canceled), engine);
        }
        self.jobs.clear();
    }
}
fn wildcard(family: Family) -> IpEndpoint {
    match family {
        Family::Ipv4 => IpEndpoint::V4(Endpoint {
            address: [0; 4],
            port: 0,
        }),
        Family::Ipv6 => IpEndpoint::V6(Endpoint6 {
            address: [0; 16],
            port: 0,
            scope_id: 0,
        }),
    }
}
fn net_error(e: NetError) -> Error {
    match e {
        NetError::WouldBlock => Error::WouldBlock,
        NetError::NoResources => Error::NoResources,
        NetError::TimedOut => Error::TimedOut,
        NetError::Invalid | NetError::WrongState => Error::Invalid,
        _ => Error::Unreachable,
    }
}
fn wire_error(e: wire::Error) -> Error {
    match e {
        wire::Error::Limit | wire::Error::NoMemory => Error::NoResources,
        _ => Error::InvalidResponse,
    }
}
impl Job {
    fn retire_socket(&mut self, engine: &mut Engine) {
        if let Some(socket) = self.socket.take() {
            let _ = engine.abort(socket);
        }
    }
    fn finish(&mut self, result: Result<Resolved, Error>, engine: &mut Engine) {
        self.retire_socket(engine);
        self.phase = Phase::Terminal;
        self.result = Some(result);
        self.request = Vec::new();
        self.buffer = Vec::new();
        self.history = Vec::new();
    }
    fn tcp(&mut self, server: IpEndpoint, engine: &mut Engine) -> Result<(), Error> {
        let mut request = Vec::new();
        request
            .try_reserve_exact(self.request.len() + 2)
            .map_err(|_| Error::NoResources)?;
        request.extend_from_slice(&(self.request.len() as u16).to_be_bytes());
        request.extend_from_slice(&self.request);
        self.retire_socket(engine);
        let socket = engine.create_tcp(server.family()).map_err(net_error)?;
        match engine.connect_ip(socket, server) {
            Ok(()) | Err(NetError::WouldBlock) => {}
            Err(e) => {
                let _ = engine.abort(socket);
                return Err(net_error(e));
            }
        }
        self.socket = Some(socket);
        self.request = request;
        self.phase = Phase::Tcp {
            sent: 0,
            received: 0,
            expected: None,
        };
        Ok(())
    }
    fn response(
        &mut self,
        response: Response,
        now: u64,
        engine: &mut Engine,
    ) -> Result<bool, Error> {
        match response {
            Response::Answer(answer) => {
                if self.aliases + answer.aliases > wire::MAX_ALIASES
                    || self.history[..self.history.len() - 1].contains(&answer.canonical_name)
                {
                    return Err(Error::InvalidResponse);
                }
                let expires_at = self
                    .expires_at
                    .min(now.saturating_add(answer.ttl as u64 * 1_000_000_000));
                self.finish(Ok(Resolved { answer, expires_at }), engine);
                Ok(false)
            }
            Response::Alias { .. } => Err(Error::InvalidResponse),
            Response::NameNotFound => Err(Error::NameNotFound),
            Response::NoData => Err(Error::NoData),
            Response::TemporaryFailure => Err(Error::TemporaryFailure),
            Response::Truncated => Err(Error::InvalidResponse),
        }
    }
    fn pump(&mut self, server: IpEndpoint, now: u64, engine: &mut Engine) -> bool {
        if self.result.is_some() {
            return false;
        }
        if now >= self.deadline {
            self.finish(Err(Error::TimedOut), engine);
            return false;
        }
        let result = self.progress(server, now, engine);
        match result {
            Ok(more) => more,
            Err(e) => {
                self.finish(Err(e), engine);
                false
            }
        }
    }
    fn progress(
        &mut self,
        server: IpEndpoint,
        now: u64,
        engine: &mut Engine,
    ) -> Result<bool, Error> {
        let socket = self.socket.ok_or(Error::BadJob)?;
        if matches!(self.phase, Phase::Udp { .. }) {
            for batch in 0..8 {
                match engine.recv_from(socket, &mut self.buffer, false) {
                    Ok(packet) => {
                        if packet.sender != server || packet.truncated {
                            if batch == 7 {
                                return Ok(true);
                            }
                            continue;
                        }
                        match wire::parse(&self.buffer[..packet.copied], &self.question) {
                            Ok(Response::Truncated) => {
                                self.tcp(server, engine)?;
                                return Ok(true);
                            }
                            Ok(response) => {
                                return self.response_with_server(response, server, now, engine);
                            }
                            Err(wire::Error::Limit | wire::Error::NoMemory) => {
                                return Err(Error::NoResources);
                            }
                            Err(_) => {}
                        }
                        if batch == 7 {
                            return Ok(true);
                        }
                    }
                    Err(NetError::WouldBlock) => break,
                    Err(e) => return Err(net_error(e)),
                }
            }
            if let Phase::Udp {
                attempts,
                next_send,
                blocked,
            } = &mut self.phase
            {
                if *blocked || now >= *next_send {
                    if *attempts == UDP_RETRY_NS.len() {
                        return Err(Error::TimedOut);
                    }
                    match engine.send_to(socket, &self.request, Some(server)) {
                        Ok(n) if n == self.request.len() => {
                            *next_send = now.saturating_add(UDP_RETRY_NS[*attempts]);
                            *attempts += 1;
                            *blocked = false;
                        }
                        Ok(_) => return Err(Error::InvalidResponse),
                        Err(NetError::WouldBlock) => {
                            *blocked = true;
                            *next_send = u64::MAX;
                            return Ok(false);
                        }
                        Err(e) => return Err(net_error(e)),
                    }
                }
            }
            return Ok(false);
        }
        if let Phase::Tcp {
            sent,
            received,
            expected,
        } = &mut self.phase
        {
            let readiness = engine.readiness(socket).map_err(net_error)?;
            if readiness.error {
                return Err(engine
                    .peek_error(socket)
                    .map_err(net_error)?
                    .map(net_error)
                    .unwrap_or(Error::Unreachable));
            }
            if !readiness.connected {
                return Ok(false);
            }
            if *sent < self.request.len() {
                match engine.write(socket, &self.request[*sent..]) {
                    Ok(n) => *sent += n,
                    Err(NetError::WouldBlock) => {}
                    Err(e) => return Err(net_error(e)),
                }
            }
            if *sent < self.request.len() {
                return Ok(false);
            }
            for _ in 0..8 {
                let end = expected.map(|n| n + 2).unwrap_or(2);
                if *received == end {
                    if expected.is_none() {
                        let n = u16::from_be_bytes(self.buffer[..2].try_into().unwrap()) as usize;
                        if n < 12 {
                            return Err(Error::InvalidResponse);
                        }
                        if n > wire::MAX_MESSAGE {
                            return Err(Error::NoResources);
                        }
                        *expected = Some(n);
                        continue;
                    }
                    let response =
                        wire::parse(&self.buffer[2..end], &self.question).map_err(wire_error)?;
                    return self.response_with_server(response, server, now, engine);
                }
                match engine.read(socket, &mut self.buffer[*received..end], false) {
                    Ok(0) => return Err(Error::InvalidResponse),
                    Ok(n) => *received += n,
                    Err(NetError::WouldBlock) => return Ok(false),
                    Err(e) => return Err(net_error(e)),
                }
            }
            return Ok(true);
        }
        Ok(false)
    }
    fn response_with_server(
        &mut self,
        response: Response,
        server: IpEndpoint,
        now: u64,
        engine: &mut Engine,
    ) -> Result<bool, Error> {
        if let Response::Alias { name, ttl, hops } = response {
            if self.history.contains(&name) || self.aliases + hops > wire::MAX_ALIASES {
                return Err(Error::InvalidResponse);
            }
            self.aliases += hops;
            self.history.push(name);
            self.expires_at = self
                .expires_at
                .min(now.saturating_add(ttl as u64 * 1_000_000_000));
            let question = Question {
                id: engine.random_u16().map_err(net_error)?,
                name,
                kind: self.question.kind,
            };
            let request = question.encode().map_err(wire_error)?;
            self.retire_socket(engine);
            let socket = engine.create_udp(server.family()).map_err(net_error)?;
            if let Err(e) = engine.udp_bind(socket, wildcard(server.family())) {
                let _ = engine.abort(socket);
                return Err(net_error(e));
            }
            self.question = question;
            self.request = request;
            self.socket = Some(socket);
            self.phase = Phase::Udp {
                attempts: 0,
                next_send: now,
                blocked: false,
            };
            return Ok(true);
        }
        self.response(response, now, engine)
    }
}
