//! Client tests against deliberately constrained or malformed peers. The netd
//! fixture separately exercises real UDP, IPv6, and recursive DNS transport.
use super::*;
use dross_rt::{
    deadline::{Deadline, Duration},
    owned::{self, Channel, Message},
    protocol::{datagram as udp, frame::Header, network as net, resolver as dns, schema::*},
};
fn end() -> Deadline {
    Deadline::from_now(Duration::from_secs(10))
}
fn request(server: &Channel) -> Message {
    server.receive(end()).unwrap()
}
fn header(m: &Message) -> Header {
    dross_rt::protocol::frame::Frame::parse(m.bytes(), m.handle_count())
        .unwrap()
        .header
}
fn response(server: &Channel, h: Header, status: i32, payload: &[u8]) {
    server
        .try_send(
            net::reply_message(h.protocol, h.ordinal, h.request_id, status, payload, 0).unwrap(),
        )
        .unwrap()
}

pub fn run(t: &mut Tally) {
    let (client, server) = Channel::pair().unwrap();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        for original in [0, 5] {
            let m = request(&server);
            let decoded = udp::decode_request(m.bytes(), m.handle_count()).unwrap();
            assert!(matches!(
                decoded.operation,
                udp::Operation::Receive { maximum: 3 }
            ));
            let metadata = udp::Metadata {
                sender: net::Endpoint::v6([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 2, 0),
                local: net::Endpoint::v6([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 2, 99),
                interface: 2,
                original_len: original,
                copied_len: original.min(3),
                truncated: original > 3,
            };
            let prefix = udp::encode_metadata(metadata).unwrap();
            let mut reply = net::reply_message(
                PROTOCOL_DATAGRAM_SOCKET,
                2,
                decoded.header.request_id,
                0,
                &prefix[4..],
                0,
            )
            .unwrap();
            reply
                .try_extend_bytes(&b"abc"[..original.min(3) as usize])
                .unwrap();
            // Append changes body size, so explicitly encode the final wire header.
            Header::new(
                PROTOCOL_DATAGRAM_SOCKET,
                1,
                KIND_REPLY,
                2,
                decoded.header.request_id,
                72 + original.min(3) as usize,
                0,
            )
            .unwrap()
            .encode(reply.bytes_mut())
            .unwrap();
            server.try_send(reply).unwrap();
        }
    })
    .unwrap();
    let mut socket = udp::DatagramSocket::new(client.into_untyped()).unwrap();
    let empty = socket.receive(3, end()).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.metadata().original_len, 0);
    assert!(!empty.metadata().truncated);
    let packet = socket.receive(3, end()).unwrap();
    assert_eq!(packet.bytes(), b"abc");
    assert_eq!(packet.metadata().original_len, 5);
    assert!(packet.metadata().truncated);
    assert_eq!(packet.metadata().sender.port, 0);
    worker.join().unwrap();
    t.check(
        "native UDP empty packets and truncation preserve boundaries",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let mut socket = udp::DatagramSocket::new(client.into_untyped()).unwrap();
    assert_eq!(
        socket.send_to(
            net::Endpoint::v4([127, 0, 0, 1], 1),
            b"x",
            Deadline::IMMEDIATE
        ),
        Err(net::NetworkError::NotSent(Error::TimedOut))
    );
    assert_eq!(
        server.receive(Deadline::IMMEDIATE).unwrap_err(),
        Error::TimedOut
    );
    assert!(!socket.is_retired());
    drop(socket);
    drop(server);
    let (client, server) = Channel::pair().unwrap();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let m = request(&server);
        let h = header(&m);
        response(&server, h, 0, &2u32.to_le_bytes())
    })
    .unwrap();
    let mut socket = udp::DatagramSocket::new(client.into_untyped()).unwrap();
    assert!(matches!(
        socket.send_to(net::Endpoint::v4([127, 0, 0, 1], 1), b"abc", end()),
        Err(net::NetworkError::InvalidReply(_))
    ));
    assert!(socket.is_retired());
    worker.join().unwrap();
    t.check(
        "native UDP deadlines submit nothing and partial sends retire",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let (attachment, observer) = Channel::pair().unwrap();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let m = request(&server);
        let h = header(&m);
        let mut reply =
            net::reply_message(h.protocol, h.ordinal, h.request_id, 0, &[0; 68], 1).unwrap();
        reply.try_push_handle(attachment.into_untyped()).unwrap();
        server.try_send(reply).unwrap()
    })
    .unwrap();
    let mut socket = udp::DatagramSocket::new(client.into_untyped()).unwrap();
    assert!(matches!(
        socket.receive(3, end()),
        Err(net::NetworkError::InvalidReply(_))
    ));
    assert!(socket.is_retired());
    worker.join().unwrap();
    assert_eq!(
        observer.receive(Deadline::IMMEDIATE).unwrap_err(),
        Error::PeerClosed
    );
    t.check(
        "native UDP malformed attached replies close every capability",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let m = request(&server);
        assert!(matches!(
            dns::decode_request(m.bytes(), 0).unwrap().operation,
            dns::Operation::Collect
        ));
        response(&server, header(&m), STATUS_WOULD_BLOCK, &[]);
        let m = request(&server);
        assert!(matches!(
            dns::decode_request(m.bytes(), 0).unwrap().operation,
            dns::Operation::Cancel
        ));
        response(&server, header(&m), 0, &[]);
        let m = request(&server);
        let body = dns::encode_info(dns::QueryInfo {
            rights: 15,
            state: dns::QueryState::Canceled,
            error: STATUS_CANCELED,
            revision: 2,
            expires_at: 0,
        })
        .unwrap();
        response(&server, header(&m), 0, &body[4..]);
        for _ in 0..2 {
            let m = request(&server);
            assert!(matches!(
                dns::decode_request(m.bytes(), 0).unwrap().operation,
                dns::Operation::Collect
            ));
            response(&server, header(&m), STATUS_CANCELED, &[])
        }
    })
    .unwrap();
    let mut query = dns::ResolveQuery::new(client.into_untyped(), dns::QueryType::A).unwrap();
    assert_eq!(
        query.collect(end()).unwrap_err(),
        dns::ResolveError::Client(net::NetworkError::WouldBlock)
    );
    query.cancel(end()).unwrap();
    assert!(!query.is_retired());
    assert_eq!(query.info(end()).unwrap().state, dns::QueryState::Canceled);
    for _ in 0..2 {
        assert_eq!(
            query.collect(end()).unwrap_err(),
            dns::ResolveError::Canceled
        )
    }
    worker.join().unwrap();
    t.check(
        "native DNS cancel retains repeatable terminal query status",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let (query_client, query_server) = Channel::pair().unwrap();
    let expires = sys::clock() + 2_000_000_000;
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let m = request(&server);
        let decoded = dns::decode_request(m.bytes(), 0).unwrap();
        assert!(matches!(
            decoded.operation,
            dns::Operation::Lookup {
                name: "example.test.",
                kind: dns::QueryType::Aaaa,
                rights: 15
            }
        ));
        let h = header(&m);
        let mut reply = net::reply_message(
            h.protocol,
            h.ordinal,
            h.request_id,
            0,
            &15u32.to_le_bytes(),
            1,
        )
        .unwrap();
        reply.try_push_handle(query_client.into_untyped()).unwrap();
        server.try_send(reply).unwrap();
        for _ in 0..2 {
            let m = request(&query_server);
            let prefix = dns::encode_answers_prefix(dns::QueryType::Aaaa, 1, 2, expires).unwrap();
            let address =
                net::Endpoint::v6([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 0, 0).encode();
            let mut body = [0; 44];
            body[..20].copy_from_slice(&prefix[4..]);
            body[20..].copy_from_slice(&address);
            response(&query_server, header(&m), 0, &body)
        }
    })
    .unwrap();
    let mut resolver = dns::Resolver::new(client.into_untyped()).unwrap();
    assert_eq!(
        resolver
            .lookup(
                "example.test.",
                dns::QueryType::Aaaa,
                15,
                Deadline::IMMEDIATE
            )
            .unwrap_err(),
        dns::ResolveError::Client(net::NetworkError::NotSent(Error::TimedOut))
    );
    let mut query = resolver
        .lookup("example.test.", dns::QueryType::Aaaa, 15, end())
        .unwrap();
    let first = query.collect(end()).unwrap();
    let second = query.collect(end()).unwrap();
    assert_eq!(first.expires_at_monotonic_ns(), expires);
    assert_eq!(second.expires_at_monotonic_ns(), expires);
    assert_eq!(second.len(), 1);
    assert_eq!(second.kind(), dns::QueryType::Aaaa);
    assert!(second.ttl_seconds_remaining() <= 2);
    assert_eq!(second.addresses().next().unwrap().port, 0);
    worker.join().unwrap();
    t.check(
        "native DNS jobs transfer ownership and keep fixed result expiry",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let (attachment, observer) = Channel::pair().unwrap();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let m = request(&server);
        let h = header(&m);
        let prefix = dns::encode_answers_prefix(dns::QueryType::Aaaa, 1, 1, 123).unwrap();
        let mut body = [0; 44];
        body[..20].copy_from_slice(&prefix[4..]);
        body[20..].copy_from_slice(&net::Endpoint::v4([1, 2, 3, 4], 0).encode());
        let mut reply =
            net::reply_message(h.protocol, h.ordinal, h.request_id, 0, &body, 1).unwrap();
        reply.try_push_handle(attachment.into_untyped()).unwrap();
        server.try_send(reply).unwrap()
    })
    .unwrap();
    let mut query = dns::ResolveQuery::new(client.into_untyped(), dns::QueryType::Aaaa).unwrap();
    assert!(matches!(
        query.collect(end()),
        Err(dns::ResolveError::Client(net::NetworkError::InvalidReply(
            _
        )))
    ));
    assert!(query.is_retired());
    worker.join().unwrap();
    assert_eq!(
        observer.receive(Deadline::IMMEDIATE).unwrap_err(),
        Error::PeerClosed
    );
    t.check(
        "native DNS malformed results close attachments and retire",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let mut m = request(&server);
        let h = header(&m);
        assert!(matches!(
            dns::decode_request(m.bytes(), 1).unwrap().operation,
            dns::Operation::Observe { mask: 64, key: 17 }
        ));
        let sink = m.take_handle(0).unwrap();
        let mut body = [0; 12];
        body[4..].copy_from_slice(&9u64.to_le_bytes());
        response(&server, h, 0, &body);
        let event = net::encode_event(net::Readiness {
            subscription: 9,
            key: 17,
            revision: 2,
            signals: net::signals::COMPLETE,
            error: 0,
        })
        .unwrap();
        owned::try_send(sink.as_handle(), Message::try_from_bytes(&event).unwrap()).unwrap();
        let m = request(&server);
        assert!(matches!(
            dns::decode_request(m.bytes(), 0).unwrap().operation,
            dns::Operation::Rearm {
                subscription: 9,
                revision: 2
            }
        ));
        response(&server, header(&m), 0, &[]);
        let event = net::encode_event(net::Readiness {
            subscription: 9,
            key: 17,
            revision: 3,
            signals: net::signals::COMPLETE | net::signals::ERROR,
            error: STATUS_CANCELED,
        })
        .unwrap();
        owned::try_send(sink.as_handle(), Message::try_from_bytes(&event).unwrap()).unwrap();
        let m = request(&server);
        assert!(matches!(
            dns::decode_request(m.bytes(), 0).unwrap().operation,
            dns::Operation::Unobserve { subscription: 9 }
        ));
        response(&server, header(&m), 0, &[]);
    })
    .unwrap();
    let mut query = dns::ResolveQuery::new(client.into_untyped(), dns::QueryType::A).unwrap();
    let mut watch = query.observe(net::signals::COMPLETE, 17, end()).unwrap();
    assert_eq!(watch.receive(end()).unwrap().revision, 2);
    query.rearm(&mut watch, end()).unwrap();
    let event = watch.receive(end()).unwrap();
    assert_eq!(event.error, STATUS_CANCELED);
    assert_eq!(event.revision, 3);
    query.unobserve(&mut watch, end()).unwrap();
    assert_eq!(
        watch.receive(end()).unwrap_err(),
        net::NetworkError::Retired
    );
    assert!(!query.is_retired());
    worker.join().unwrap();
    t.check(
        "native DNS one-shot observations rearm and cancel separately",
        true,
        "",
    );
}
