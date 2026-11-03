#![no_std]
#![no_main]
#![recursion_limit = "256"]
extern crate alloc;
use alloc::{vec, vec::Vec};
use dross_rt::Handle;
use oscar_dns_wire::{
    self as wire, Address, Name, QueryType, Question, RecordData, ResourceRecord,
};
use oscar_netstack3_engine::{
    Config, Endpoint, Engine, Error as NetError, Family, IpEndpoint, SocketId,
};
use oscar_resolver::{Error, JobId, Resolved, Resolver, State};
dross_rt::main!(main);
const SECOND: u64 = 1_000_000_000;
struct Peer {
    stream: SocketId,
    request: Vec<u8>,
    reply: Vec<u8>,
    sent: usize,
}
struct Lab {
    engine: Engine,
    resolver: Resolver,
    udp: SocketId,
    other: SocketId,
    listener: SocketId,
    peers: Vec<Peer>,
    pending: Vec<(SocketId, Vec<u8>, IpEndpoint)>,
    now: u64,
    drops: usize,
    tcp_count: usize,
    requests: usize,
    alias_delay: bool,
}
impl Lab {
    fn new() -> Self {
        let mut engine = Engine::new(
            Config {
                mac: [2, 0, 0, 0, 0, 1],
                address: [192, 0, 2, 1],
                prefix: 24,
                gateway: None,
                mtu: 1500,
            },
            0,
        )
        .unwrap();
        let server = IpEndpoint::V4(Endpoint {
            address: [127, 0, 0, 1],
            port: 53053,
        });
        let udp = engine.create_udp(Family::Ipv4).unwrap();
        engine.udp_bind(udp, server).unwrap();
        let other = engine.create_udp(Family::Ipv4).unwrap();
        engine
            .udp_bind(
                other,
                IpEndpoint::V4(Endpoint {
                    address: [127, 0, 0, 1],
                    port: 53054,
                }),
            )
            .unwrap();
        let listener = engine.create_tcp(Family::Ipv4).unwrap();
        engine.bind_ip(listener, server).unwrap();
        engine.listen(listener, 2).unwrap();
        Self {
            engine,
            resolver: Resolver::new(server).unwrap(),
            udp,
            other,
            listener,
            peers: Vec::new(),
            pending: Vec::new(),
            now: 0,
            drops: 0,
            tcp_count: 0,
            requests: 0,
            alias_delay: false,
        }
    }
    fn start(&mut self, name: &str, kind: QueryType) -> JobId {
        self.resolver
            .start(
                name,
                kind,
                self.now + 9 * SECOND,
                self.now,
                &mut self.engine,
            )
            .unwrap()
    }
    fn records(question: &Question) -> Vec<ResourceRecord> {
        let name = question.name.to_ascii().unwrap();
        let data = if question.kind == QueryType::A {
            RecordData::A([192, 0, 2, 77])
        } else {
            RecordData::Aaaa([0x20, 1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77])
        };
        if name == "alias.test" {
            vec![ResourceRecord {
                name: question.name,
                ttl: 1,
                data: RecordData::Cname(Name::from_ascii("target.test").unwrap()),
            }]
        } else {
            vec![ResourceRecord {
                name: question.name,
                ttl: 60,
                data,
            }]
        }
    }
    fn step(&mut self) -> bool {
        let mut work = self.engine.pump(self.now).more_work;
        work |= self.resolver.pump(self.now, &mut self.engine);
        let mut bytes = [0u8; 2048];
        for _ in 0..16 {
            match self.engine.recv_from(self.udp, &mut bytes, false) {
                Ok(packet) => {
                    work = true;
                    self.requests += 1;
                    let q = wire::decode_question(&bytes[..packet.copied]).unwrap();
                    let name = q.name.to_ascii().unwrap();
                    if name == "timeout.test" || name == "cancel.test" {
                        continue;
                    }
                    if name == "drop.test" && self.drops == 0 {
                        self.drops += 1;
                        continue;
                    }
                    if name == "target.test" && !self.alias_delay {
                        self.now += 2 * SECOND;
                        self.alias_delay = true;
                    }
                    if name == "foreign.test" {
                        let mut wrong = q;
                        wrong.id = wrong.id.wrapping_add(1);
                        let bad = wire::encode_response(&wrong, 0, false, &Self::records(&wrong))
                            .unwrap();
                        self.pending.push((self.udp, bad, packet.sender));
                        let bad = wire::encode_response(&q, 0, false, &Self::records(&q)).unwrap();
                        self.pending.push((self.other, bad, packet.sender));
                        wrong = q;
                        wrong.name = Name::from_ascii("wrong.test").unwrap();
                        let bad = wire::encode_response(&wrong, 0, false, &Self::records(&wrong))
                            .unwrap();
                        self.pending.push((self.udp, bad, packet.sender));
                    }
                    let (rcode, tc, records) = match name.as_str() {
                        "nx.test" => (3, false, vec![]),
                        "empty.test" => (0, false, vec![]),
                        "tcp.test" => (0, true, vec![]),
                        _ => (0, false, Self::records(&q)),
                    };
                    let reply = wire::encode_response(&q, rcode, tc, &records).unwrap();
                    self.pending.push((self.udp, reply, packet.sender));
                }
                Err(NetError::WouldBlock) => break,
                Err(e) => panic!("DNS receive {e:?}"),
            }
        }
        for _ in 0..16 {
            if self.pending.is_empty() {
                break;
            }
            let (socket, bytes, remote) = &self.pending[0];
            match self.engine.send_to(*socket, bytes, Some(*remote)) {
                Ok(n) => {
                    assert_eq!(n, bytes.len());
                    self.pending.remove(0);
                    work = true;
                }
                Err(NetError::WouldBlock) => break,
                Err(e) => panic!("DNS reply {e:?}"),
            }
        }
        loop {
            match self.engine.accept_ip(self.listener) {
                Ok((stream, _)) => {
                    self.tcp_count += 1;
                    self.peers.push(Peer {
                        stream,
                        request: Vec::new(),
                        reply: Vec::new(),
                        sent: 0,
                    });
                    work = true
                }
                Err(NetError::WouldBlock) => break,
                Err(e) => panic!("DNS accept {e:?}"),
            }
        }
        let mut i = 0;
        while i < self.peers.len() {
            let peer = &mut self.peers[i];
            if peer.reply.is_empty() {
                match self.engine.read(peer.stream, &mut bytes, false) {
                    Ok(0) => panic!("early DNS client close"),
                    Ok(n) => {
                        peer.request.extend_from_slice(&bytes[..n]);
                        work = true
                    }
                    Err(NetError::WouldBlock) => {}
                    Err(e) => panic!("DNS read {e:?}"),
                }
                if peer.request.len() >= 2 {
                    let len = u16::from_be_bytes(peer.request[..2].try_into().unwrap()) as usize;
                    if peer.request.len() == len + 2 {
                        let q = wire::decode_question(&peer.request[2..]).unwrap();
                        let reply =
                            wire::encode_response(&q, 0, false, &Self::records(&q)).unwrap();
                        peer.reply
                            .extend_from_slice(&(reply.len() as u16).to_be_bytes());
                        peer.reply.extend_from_slice(&reply);
                    }
                }
            }
            if !peer.reply.is_empty() {
                let end = (peer.sent + 3).min(peer.reply.len());
                match self.engine.write(peer.stream, &peer.reply[peer.sent..end]) {
                    Ok(n) => {
                        peer.sent += n;
                        work = true
                    }
                    Err(NetError::WouldBlock) => {}
                    Err(e) => panic!("DNS write {e:?}"),
                }
                if peer.sent == peer.reply.len() {
                    let peer = self.peers.swap_remove(i);
                    self.engine.close(peer.stream).unwrap();
                    continue;
                }
            }
            i += 1;
        }
        work |= self.engine.pump(self.now).more_work;
        work
    }
    fn finish(&mut self, id: JobId) -> Result<&Resolved, Error> {
        for _ in 0..2000 {
            if self.resolver.state(id).unwrap() != State::Pending {
                return self.resolver.collect(id);
            }
            if !self.step() {
                let next = self
                    .resolver
                    .next_deadline()
                    .into_iter()
                    .chain(self.engine.next_deadline())
                    .min()
                    .unwrap_or(self.now + SECOND);
                self.now = next.max(self.now + 1);
            }
        }
        panic!("DNS finite event budget exceeded")
    }
}
fn main(_boot: Handle) -> i32 {
    let mut lab = Lab::new();
    for (name, kind) in [
        ("a.test", QueryType::A),
        ("v6.test", QueryType::Aaaa),
        ("foreign.test", QueryType::A),
        ("drop.test", QueryType::A),
        ("tcp.test", QueryType::A),
        ("alias.test", QueryType::A),
    ] {
        let id = lab.start(name, kind);
        let answer = lab.finish(id).unwrap();
        assert_eq!(answer.answer.addresses.len(), 1);
        match kind {
            QueryType::A => assert_eq!(answer.answer.addresses[0], Address::V4([192, 0, 2, 77])),
            QueryType::Aaaa => assert!(matches!(answer.answer.addresses[0], Address::V6(_))),
        }
        let expires = answer.expires_at;
        if name == "alias.test" {
            assert_eq!(lab.resolver.collect(id).unwrap().remaining_ttl(lab.now), 0);
        }
        lab.now += SECOND;
        assert_eq!(lab.resolver.collect(id).unwrap().expires_at, expires);
        lab.resolver.remove(id, &mut lab.engine).unwrap();
        dross_rt::println!("resolver-test: {name} PASS");
    }
    assert_eq!(lab.drops, 1);
    assert_eq!(lab.tcp_count, 1);
    for (name, error) in [
        ("nx.test", Error::NameNotFound),
        ("empty.test", Error::NoData),
        ("timeout.test", Error::TimedOut),
    ] {
        let id = lab.start(name, QueryType::A);
        assert_eq!(lab.finish(id).err(), Some(error));
        lab.resolver.remove(id, &mut lab.engine).unwrap();
    }
    let id = lab.start("cancel.test", QueryType::A);
    lab.step();
    lab.resolver.cancel(id, &mut lab.engine).unwrap();
    assert_eq!(lab.resolver.state(id), Ok(State::Canceled));
    assert_eq!(lab.resolver.collect(id).err(), Some(Error::Canceled));
    lab.step();
    let count = lab.requests;
    for _ in 0..8 {
        lab.now += SECOND;
        lab.step();
    }
    assert_eq!(lab.requests, count);
    lab.resolver.remove(id, &mut lab.engine).unwrap();
    let mut ids = Vec::new();
    for _ in 0..oscar_resolver::MAX_JOBS {
        ids.push(lab.start("cancel.test", QueryType::A));
    }
    assert_eq!(
        lab.resolver.start(
            "a.test",
            QueryType::A,
            lab.now + SECOND,
            lab.now,
            &mut lab.engine
        ),
        Err(Error::NoResources)
    );
    for id in ids {
        lab.resolver.remove(id, &mut lab.engine).unwrap();
    }
    lab.resolver.shutdown(&mut lab.engine);
    for id in [lab.udp, lab.other, lab.listener] {
        lab.engine.abort(id).unwrap();
    }
    dross_rt::println!(
        "resolver-test: PASS A AAAA spoof rejection retry TCP fallback CNAME TTL cancel timeout admission"
    );
    0
}
