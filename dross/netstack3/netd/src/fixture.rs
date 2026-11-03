//! Actual-daemon acceptance for packet semantics and asynchronous DNS jobs.
use super::{Failure, check, transfer};
use alloc::vec::Vec;
use dross_rt::{
    Error,
    deadline::{Deadline, Duration},
    protocol::{
        datagram::{self, DatagramSocket},
        network::{
            self, Endpoint, Network, NetworkError, listener_rights as lr, signals,
            stream_rights as sr,
        },
        resolver::{self, QueryState, QueryType, ResolveError, Resolver},
    },
    sys,
};
use oscar_dns_wire::{self as dns, Name, RecordData, ResourceRecord};
fn wire<T>(r: Result<T, dns::Error>) -> Result<T, Failure> {
    r.map_err(|_| Failure::Check("DNS fixture encoding"))
}
fn loopback6(port: u16) -> Endpoint {
    let mut a = [0; 16];
    a[15] = 1;
    Endpoint::v6(a, 0, port)
}
fn udp(factory: &mut Network, local: Endpoint, deadline: Deadline) -> Result<(), Failure> {
    let mut a = factory.open_datagram(local, datagram::rights::ALL, deadline)?;
    let mut b = factory.open_datagram(local, datagram::rights::ALL, deadline)?;
    let aa = a
        .info(deadline)?
        .local
        .ok_or(Failure::Check("UDP local a"))?;
    let ba = b
        .info(deadline)?
        .local
        .ok_or(Failure::Check("UDP local b"))?;
    check(
        aa.port != 0 && ba.port != 0 && aa != ba,
        "distinct UDP ephemeral ports",
    )?;
    check(
        matches!(b.receive(0, deadline), Err(NetworkError::WouldBlock)),
        "UDP empty queue is WouldBlock",
    )?;
    let mut ready = b.observe(signals::READABLE | signals::ERROR, 71, deadline)?;
    let payload = b"binary\0packet\xffboundary";
    check(
        a.send_to(ba, payload, deadline)? == payload.len(),
        "UDP atomic send",
    )?;
    let event = ready.receive(deadline)?;
    check(
        event.key == 71 && event.signals & signals::READABLE != 0,
        "UDP engine observer",
    )?;
    let got = b.receive_wait(5, deadline)?;
    let m = got.metadata();
    check(
        got.bytes() == &payload[..5]
            && m.original_len == payload.len() as u32
            && m.copied_len == 5
            && m.truncated,
        "UDP truncates and consumes one packet",
    )?;
    check(
        m.sender == aa && m.local == ba && m.interface == 2,
        "UDP exact source destination and loopback interface",
    )?;
    b.cancel_observation(&mut ready, deadline)?;
    check(
        matches!(b.receive(64, deadline), Err(NetworkError::WouldBlock)),
        "UDP truncated tail not queued",
    )?;
    check(a.send_to(ba, &[], deadline)? == 0, "UDP zero length send")?;
    let empty = b.receive_wait(0, deadline)?;
    check(
        empty.is_empty() && !empty.metadata().truncated && empty.metadata().original_len == 0,
        "UDP empty packet is not EOF",
    )?;
    a.send_to(ba, b"discard", deadline)?;
    let discarded = b.receive_wait(0, deadline)?;
    check(
        discarded.is_empty()
            && discarded.metadata().original_len == 7
            && discarded.metadata().truncated,
        "UDP maxCopy zero discards nonempty packet",
    )?;
    a.send_to(ba, b"first", deadline)?;
    a.send_to(ba, b"second", deadline)?;
    check(
        b.receive_wait(64, deadline)?.bytes() == b"first"
            && b.receive_wait(64, deadline)?.bytes() == b"second",
        "UDP packet boundaries and order",
    )?;
    check(
        matches!(
            a.send_to(ba, &[0u8; 1600], deadline),
            Err(NetworkError::MessageTooLarge)
        ),
        "UDP no fragmentation MTU bound",
    )?;
    let mut narrow = factory.open_datagram(
        local,
        datagram::rights::RECEIVE | datagram::rights::INFO,
        deadline,
    )?;
    check(
        matches!(
            narrow.send_to(ba, b"denied", deadline),
            Err(NetworkError::Denied)
        ),
        "UDP attenuated send denied",
    )?;
    drop(a);
    drop(b);
    drop(narrow);
    Ok(())
}
fn receive_question(
    socket: &mut DatagramSocket,
    deadline: Deadline,
) -> Result<(dns::Question, Endpoint), Failure> {
    let packet = socket.receive_wait(512, deadline)?;
    Ok((
        wire(dns::decode_question(packet.bytes()))?,
        packet.metadata().sender,
    ))
}
fn send_answer(
    socket: &mut DatagramSocket,
    question: &dns::Question,
    destination: Endpoint,
    rcode: u8,
    truncated: bool,
    records: &[ResourceRecord],
    deadline: Deadline,
) -> Result<(), Failure> {
    let packet = wire(dns::encode_response(question, rcode, truncated, records))?;
    check(
        socket.send_to(destination, &packet, deadline)? == packet.len(),
        "DNS fixture atomic response",
    )
}
fn tcp_read(
    stream: &mut network::TcpStream,
    n: usize,
    deadline: Deadline,
) -> Result<Vec<u8>, Failure> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(n).map_err(|_| Error::NoMemory)?;
    while bytes.len() < n {
        match stream.read_wait((n - bytes.len()).min(network::MAX_INLINE), deadline)? {
            network::ReadResult::Data(data) => bytes.extend_from_slice(data.bytes()),
            network::ReadResult::Eof => return Err(Failure::Check("DNS TCP early EOF")),
        }
    }
    Ok(bytes)
}
fn dns_tests(
    datagrams: &mut Network,
    listen: &mut Network,
    resolver: &mut Resolver,
    server: Endpoint,
    deadline: Deadline,
) -> Result<(), Failure> {
    check(
        resolver.describe(deadline)?.rights == resolver::LOOKUP,
        "explicit resolver grant",
    )?;
    let mut udp = datagrams.open_datagram(server, datagram::rights::ALL, deadline)?;
    let mut tcp = listen.listen_tcp(server, lr::ALL, sr::ALL, 2, deadline)?;
    let mut query = resolver.lookup(
        "alias.oscar.test.",
        QueryType::A,
        resolver::rights::ALL,
        deadline,
    )?;
    check(
        matches!(
            query.collect(deadline),
            Err(ResolveError::Client(NetworkError::WouldBlock))
        ),
        "DNS immediate pending query cap",
    )?;
    let mut complete = query.observe(signals::COMPLETE | signals::ERROR, 81, deadline)?;
    let (q, source) = receive_question(&mut udp, deadline)?;
    let canonical = wire(Name::from_ascii("canonical.oscar.test"))?;
    let records = [
        ResourceRecord {
            name: canonical,
            ttl: 90,
            data: RecordData::A([192, 0, 2, 44]),
        },
        ResourceRecord {
            name: q.name,
            ttl: 30,
            data: RecordData::Cname(canonical),
        },
    ];
    // Same answer from a different UDP source cannot complete this query.
    let mut impostor = datagrams.open_datagram(
        Endpoint { port: 0, ..server },
        datagram::rights::ALL,
        deadline,
    )?;
    send_answer(&mut impostor, &q, source, 0, false, &records, deadline)?;
    check(
        matches!(
            query.collect(deadline),
            Err(ResolveError::Client(NetworkError::WouldBlock))
        ),
        "DNS ignores foreign source",
    )?;
    drop(impostor);
    let mut wrong_name = q.clone();
    wrong_name.name = wire(Name::from_ascii("other.oscar.test"))?;
    send_answer(&mut udp, &wrong_name, source, 0, false, &records, deadline)?;
    check(
        matches!(
            query.collect(deadline),
            Err(ResolveError::Client(NetworkError::WouldBlock))
        ),
        "DNS ignores foreign question",
    )?;
    // A foreign ID and question must not complete this query.
    let mut foreign = q.clone();
    foreign.id ^= 1;
    send_answer(&mut udp, &foreign, source, 0, false, &records, deadline)?;
    check(
        matches!(
            query.collect(deadline),
            Err(ResolveError::Client(NetworkError::WouldBlock))
        ),
        "DNS ignores foreign transaction",
    )?;
    send_answer(&mut udp, &q, source, 0, false, &records, deadline)?;
    let event = complete.receive(deadline)?;
    check(
        event.key == 81 && event.signals & signals::COMPLETE != 0 && event.error == 0,
        "DNS terminal observer",
    )?;
    let answers = query.wait(deadline)?;
    check(
        answers.len() == 1
            && answers.addresses().next() == Some(Endpoint::v4([192, 0, 2, 44], 0))
            && answers.ttl_seconds_remaining() <= 30,
        "DNS unordered CNAME A and minimum TTL",
    )?;
    let expiry = answers.expires_at_monotonic_ns();
    check(
        query.collect(deadline)?.expires_at_monotonic_ns() == expiry,
        "DNS collect preserves absolute expiry",
    )?;
    query.unobserve(&mut complete, deadline)?;
    drop(query);
    let mut chained = resolver.lookup(
        "chain.oscar.test",
        QueryType::A,
        resolver::rights::ALL,
        deadline,
    )?;
    let (first, source) = receive_question(&mut udp, deadline)?;
    let next = wire(Name::from_ascii("next.oscar.test"))?;
    send_answer(
        &mut udp,
        &first,
        source,
        0,
        false,
        &[ResourceRecord {
            name: first.name,
            ttl: 2,
            data: RecordData::Cname(next),
        }],
        deadline,
    )?;
    let (followup, source) = receive_question(&mut udp, deadline)?;
    check(followup.name == next, "DNS follows CNAME question")?;
    send_answer(
        &mut udp,
        &followup,
        source,
        0,
        false,
        &[ResourceRecord {
            name: next,
            ttl: 90,
            data: RecordData::A([192, 0, 2, 45]),
        }],
        deadline,
    )?;
    let answer = chained.wait(deadline)?;
    check(
        answer.ttl_seconds_remaining() <= 2
            && answer.addresses().next() == Some(Endpoint::v4([192, 0, 2, 45], 0)),
        "DNS CNAME followup keeps original TTL expiry",
    )?;
    drop(chained);
    let mut query = resolver.lookup(
        "v6.oscar.test",
        QueryType::Aaaa,
        resolver::rights::ALL,
        deadline,
    )?;
    let (q, source) = receive_question(&mut udp, deadline)?;
    let mut address = [0; 16];
    address[0] = 0x20;
    address[1] = 1;
    address[2] = 0x0d;
    address[3] = 0xb8;
    address[15] = 44;
    send_answer(
        &mut udp,
        &q,
        source,
        0,
        false,
        &[ResourceRecord {
            name: q.name,
            ttl: 60,
            data: RecordData::Aaaa(address),
        }],
        deadline,
    )?;
    check(
        query.wait(deadline)?.addresses().next() == Some(Endpoint::v6(address, 0, 0)),
        "DNS AAAA exact bytes",
    )?;
    drop(query);
    let mut query = resolver.lookup(
        "tcp.oscar.test",
        QueryType::A,
        resolver::rights::ALL,
        deadline,
    )?;
    let (q, source) = receive_question(&mut udp, deadline)?;
    send_answer(&mut udp, &q, source, 0, true, &[], deadline)?;
    let mut stream = tcp.accept_wait(sr::ALL, deadline)?;
    let length = tcp_read(&mut stream, 2, deadline)?;
    let length = u16::from_be_bytes(length.try_into().unwrap()) as usize;
    let question = wire(dns::decode_question(&tcp_read(
        &mut stream,
        length,
        deadline,
    )?))?;
    let response = wire(dns::encode_response(
        &question,
        0,
        false,
        &[ResourceRecord {
            name: question.name,
            ttl: 12,
            data: RecordData::A([203, 0, 113, 9]),
        }],
    ))?;
    // Split the TCP length and payload to exercise the real partial IO path.
    stream.write_all(&(response.len() as u16).to_be_bytes()[..1], deadline)?;
    stream.write_all(&(response.len() as u16).to_be_bytes()[1..], deadline)?;
    for part in response.chunks(7) {
        stream.write_all(part, deadline)?;
    }
    check(
        query.wait(deadline)?.addresses().next() == Some(Endpoint::v4([203, 0, 113, 9], 0)),
        "DNS truncated UDP TCP fallback",
    )?;
    drop(query);
    drop(stream);
    for (name, rcode, expected) in [
        ("absent.oscar.test", 3, ResolveError::NameNotFound),
        ("empty.oscar.test", 0, ResolveError::NoData),
    ] {
        let mut query = resolver.lookup(name, QueryType::A, resolver::rights::ALL, deadline)?;
        let (q, source) = receive_question(&mut udp, deadline)?;
        send_answer(&mut udp, &q, source, rcode, false, &[], deadline)?;
        check(
            query.wait(deadline).err() == Some(expected),
            "DNS negative status",
        )?;
    }
    let mut query = resolver.lookup(
        "cancel.oscar.test",
        QueryType::A,
        resolver::rights::ALL,
        deadline,
    )?;
    let _ = receive_question(&mut udp, deadline)?;
    query.cancel(deadline)?;
    check(
        !query.is_retired()
            && query.info(deadline)?.state == QueryState::Canceled
            && query.collect(deadline).err() == Some(ResolveError::Canceled),
        "DNS cancel retains terminal cap",
    )?;
    drop(query);
    let short = Deadline::from_now(Duration::from_millis(80));
    let mut query = resolver.lookup(
        "timeout.oscar.test",
        QueryType::A,
        resolver::rights::ALL,
        short,
    )?;
    let _ = receive_question(&mut udp, deadline)?;
    let mut event = query.observe(signals::COMPLETE | signals::ERROR, 82, deadline)?;
    let ready = event.receive(deadline)?;
    check(
        ready.error == dross_rt::protocol::schema::STATUS_TIMED_OUT
            && query.info(deadline)?.state == QueryState::Failed,
        "DNS original deadline drives completion",
    )?;
    check(
        query.collect(deadline).err() == Some(ResolveError::Client(NetworkError::TimedOut)),
        "DNS timeout collect",
    )?;
    drop(query);
    drop(event);
    // Terminal jobs occupy their admission slot until their capability closes.
    let mut held = Vec::new();
    held.try_reserve_exact(8).map_err(|_| Error::NoMemory)?;
    for _ in 0..8 {
        let mut job = resolver.lookup(
            "held.oscar.test",
            QueryType::A,
            resolver::rights::ALL,
            deadline,
        )?;
        let _ = receive_question(&mut udp, deadline)?;
        job.cancel(deadline)?;
        check(
            job.info(deadline)?.state == QueryState::Canceled,
            "retained canceled query state",
        )?;
        held.push(job);
    }
    check(
        matches!(
            resolver.lookup(
                "overflow.oscar.test",
                QueryType::A,
                resolver::rights::ALL,
                deadline
            ),
            Err(ResolveError::Client(NetworkError::NoResources))
        ),
        "eight retained jobs bound admission",
    )?;
    drop(held.pop());
    // Closing a cap and a new request can be observed in either order. LIMIT
    // is a known unsatisfied request, so retrying admission is safe.
    let mut reclaimed = loop {
        match resolver.lookup(
            "reclaimed.oscar.test",
            QueryType::A,
            resolver::rights::ALL,
            deadline,
        ) {
            Ok(job) => break job,
            Err(ResolveError::Client(NetworkError::NoResources)) if !deadline.has_elapsed() => {
                sys::yield_now()
            }
            Err(e) => return Err(e.into()),
        }
    };
    reclaimed.cancel(deadline)?;
    drop(reclaimed);
    drop(held);
    drop(tcp);
    drop(udp);
    Ok(())
}
pub fn run(
    connect: &mut Network,
    listen: &mut Network,
    datagrams: &mut Network,
    resolver: &mut Resolver,
    server: Endpoint,
    deadline: Deadline,
) -> Result<(), Failure> {
    check(
        datagrams.describe(deadline)?.rights == network::DATAGRAM,
        "restricted datagram factory",
    )?;
    check(
        matches!(
            connect.open_datagram(Endpoint::v4([127, 0, 0, 1], 0), 15, deadline),
            Err(NetworkError::Denied)
        ),
        "TCP connect cap cannot grant UDP",
    )?;
    udp(datagrams, Endpoint::v4([127, 0, 0, 1], 0), deadline)?;
    udp(datagrams, loopback6(0), deadline)?;
    let mut listener = listen.listen_tcp(loopback6(0), lr::ALL, sr::ALL, 1, deadline)?;
    let local = listener
        .info(deadline)?
        .local
        .ok_or(Failure::Check("IPv6 listener local"))?;
    let mut client = connect.connect_tcp(local, sr::ALL, deadline)?;
    let mut accepted = listener.accept_wait(sr::ALL, deadline)?;
    transfer(
        &mut client,
        &mut accepted,
        b"native IPv6 TCP\0exact",
        deadline,
    )?;
    client.shutdown_send(deadline)?;
    super::eof(&mut accepted, deadline)?;
    accepted.shutdown_send(deadline)?;
    super::eof(&mut client, deadline)?;
    drop(client);
    drop(accepted);
    drop(listener);
    dns_tests(datagrams, listen, resolver, server, deadline)?;
    sys::log(
        "native-net-test: PASS UDP4/UDP6/packet-boundaries/zero/truncation/rights/observer/TCP6/DNS-A-AAAA-CNAME-TTL/source-ID/TCP-fallback/negative/cancel/deadline/retained8/reclaim",
    );
    Ok(())
}

/// A finite real-peer proof. The peer's host listeners bind only tailnet IPs;
/// the explicitly configured guest gateways route directly to that peer on LAN.
pub fn external(seed: &mut super::seed::Seed) -> Result<(), Failure> {
    use network::Address;
    let mut connect = Network::new(seed.take("connect").ok_or(Error::BadHandle)?)?;
    let mut datagrams = Network::new(seed.take("datagram").ok_or(Error::BadHandle)?)?;
    let mut resolver = Resolver::new(seed.take("resolver").ok_or(Error::BadHandle)?)?;
    check(
        seed.remaining() == 0,
        "external fixture has explicit minimal grants",
    )?;
    let peer4 = seed
        .value("peer4")
        .ok_or(Error::InvalidArgs)?
        .parse::<core::net::Ipv4Addr>()
        .map_err(|_| Error::InvalidArgs)?
        .octets();
    let peer6 = seed
        .value("peer6")
        .ok_or(Error::InvalidArgs)?
        .parse::<core::net::Ipv6Addr>()
        .map_err(|_| Error::InvalidArgs)?
        .octets();
    let udp_port = seed
        .value("peer-udp-port")
        .unwrap_or("37001")
        .parse::<u16>()
        .map_err(|_| Error::InvalidArgs)?;
    let tcp_port = seed
        .value("peer-tcp-port")
        .unwrap_or("37002")
        .parse::<u16>()
        .map_err(|_| Error::InvalidArgs)?;
    let deadline = Deadline::from_now(Duration::from_secs(35));
    check(
        connect.describe(deadline)?.rights == network::CONNECT
            && datagrams.describe(deadline)?.rights == network::DATAGRAM
            && resolver.describe(deadline)?.rights == resolver::LOOKUP,
        "external capability isolation",
    )?;
    for remote in [
        Endpoint::v4(peer4, tcp_port),
        Endpoint::v6(peer6, 0, tcp_port),
    ] {
        let mut stream = connect.connect_tcp(remote, sr::ALL, deadline)?;
        let mut bytes = [0; 16 * 1024 + 19];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(71) ^ (i >> 8) as u8;
        }
        stream.write_all(&bytes, deadline)?;
        stream.shutdown_send(deadline)?;
        let echoed = tcp_read(&mut stream, bytes.len(), deadline)?;
        check(echoed == bytes, "external TCP exact binary echo")?;
        super::eof(&mut stream, deadline)?;
        check(
            stream.info(deadline)?.peer == Some(remote),
            "external TCP peer address",
        )?;
        sys::log(if matches!(remote.address, Address::V4(_)) {
            "native-net-test: external TCP4 exact16403/halfclose/EOF PASS"
        } else {
            "native-net-test: external TCP6 exact16403/halfclose/EOF PASS"
        });
    }
    for remote in [
        Endpoint::v4(peer4, udp_port),
        Endpoint::v6(peer6, 0, udp_port),
    ] {
        let wildcard = match remote.address {
            Address::V4(_) => Endpoint::v4([0; 4], 0),
            Address::V6 { .. } => Endpoint::v6([0; 16], 0, 0),
        };
        let mut socket = datagrams.open_datagram(wildcard, datagram::rights::ALL, deadline)?;
        let port = socket
            .info(deadline)?
            .local
            .ok_or(Failure::Check("external UDP bound port"))?
            .port;
        let mut bytes = [0; 1007];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(73) ^ (i >> 8) as u8;
        }
        for packet in [&bytes[..], &[][..]] {
            check(
                socket.send_to(remote, packet, deadline)? == packet.len(),
                "external UDP atomic send",
            )?;
            let response = socket.receive_wait(1200, deadline)?;
            let m = response.metadata();
            check(
                response.bytes() == packet
                    && m.original_len == packet.len() as u32
                    && !m.truncated
                    && m.sender == remote
                    && m.local.port == port
                    && m.interface == 1,
                "external UDP exact packet/source/interface",
            )?;
            check(
                match m.local.address {
                    Address::V4(a) => a != [0; 4],
                    Address::V6 { octets, .. } => octets != [0; 16],
                },
                "external UDP concrete destination",
            )?;
        }
        sys::log(if matches!(remote.address, Address::V4(_)) {
            "native-net-test: external UDP4 exact1007/zero/source/interface PASS"
        } else {
            "native-net-test: external UDP6 exact1007/zero/source/interface PASS"
        });
    }
    let mut expected6 = [0; 16];
    expected6[0] = 0x20;
    expected6[1] = 1;
    expected6[2] = 0x0d;
    expected6[3] = 0xb8;
    expected6[15] = 0x44;
    for name in ["oscar.test", "tcp.oscar.test"] {
        for (kind, expected) in [
            (QueryType::A, Endpoint::v4([192, 0, 2, 44], 0)),
            (QueryType::Aaaa, Endpoint::v6(expected6, 0, 0)),
        ] {
            let answers = resolver.lookup_wait(name, kind, resolver::rights::ALL, deadline)?;
            check(
                answers.len() == 1
                    && answers.addresses().next() == Some(expected)
                    && answers.ttl_seconds_remaining() <= 30,
                "external DNS exact answer and TTL",
            )?;
        }
        sys::log(if name == "oscar.test" {
            "native-net-test: external DNS6 UDP A/AAAA exactanswers PASS"
        } else {
            "native-net-test: external DNS6 UDP-TC/TCP A/AAAA exactanswers PASS"
        });
    }
    sys::log("native-net-test: external PASS TCP4/TCP6/UDP4/UDP6/DNS6-UDP-TCP/native-capabilities");
    Ok(())
}
