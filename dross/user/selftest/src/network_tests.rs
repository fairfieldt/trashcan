//! Adverse-peer tests of the actual native clients, without substituting a TCP
//! engine. Production transport/authority tests run in native-net-test.
use super::*;
use dross_rt::deadline::{Deadline, Duration};
use dross_rt::owned::{self, Channel, Message};
use dross_rt::protocol::network::{
    self as net, Info, Network, NetworkError, Operation, ReadResult, Readiness, TcpStream,
};
use dross_rt::protocol::schema::*;

fn deadline() -> Deadline {
    Deadline::from_now(Duration::from_secs(10))
}
fn header(message: &Message) -> dross_rt::protocol::frame::Header {
    net::decode_request(message.bytes(), message.handle_count())
        .unwrap()
        .header
}
fn reply(server: &Channel, h: dross_rt::protocol::frame::Header, payload: &[u8]) {
    server
        .try_send(net::reply_message(h.protocol, h.ordinal, h.request_id, 0, payload, 0).unwrap())
        .unwrap();
}

pub fn run(t: &mut Tally) {
    // Fill the new stream's destination queue before starting the deadline.
    // CONNECT admission succeeds, but follow-up INFO cannot be submitted.
    let (factory_client, factory_server) = Channel::pair().unwrap();
    let (stream_client, stream_server) = Channel::pair().unwrap();
    for _ in 0..dross_abi::CHANNEL_DEPTH {
        stream_client.try_send(Message::new()).unwrap();
    }
    let (done, worker_done) = Channel::pair().unwrap();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let request = factory_server.receive(Deadline::FOREVER).unwrap();
        let h = header(&request);
        let mut response = net::reply_message(
            PROTOCOL_NETWORK,
            h.ordinal,
            h.request_id,
            0,
            &net::stream_rights::ALL.to_le_bytes(),
            1,
        )
        .unwrap();
        response
            .try_push_handle(stream_client.into_untyped())
            .unwrap();
        factory_server.try_send(response).unwrap();
        worker_done.receive(Deadline::FOREVER).unwrap();
        // All queued packets are the original filler: no INFO request entered.
        for _ in 0..dross_abi::CHANNEL_DEPTH {
            assert!(
                stream_server
                    .receive(Deadline::IMMEDIATE)
                    .unwrap()
                    .bytes()
                    .is_empty()
            );
        }
        assert_eq!(
            stream_server.receive(Deadline::IMMEDIATE).unwrap_err(),
            Error::PeerClosed
        );
    })
    .unwrap();
    let mut network = Network::new(factory_client.into_untyped()).unwrap();
    let error = network
        .connect_tcp(
            net::Endpoint::v4([127, 0, 0, 1], 80),
            net::stream_rights::ALL,
            Deadline::from_now(Duration::from_millis(100)),
        )
        .unwrap_err();
    assert_eq!(error, NetworkError::OutcomeUnknown(Error::TimedOut));
    done.try_send(Message::new()).unwrap();
    worker.join().unwrap();
    t.check(
        "native network admitted connect never reports a follow-up as unsent",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let mut stream = TcpStream::new(client.into_untyped()).unwrap();
    assert_eq!(
        stream.write(b"not sent", Deadline::IMMEDIATE),
        Err(NetworkError::NotSent(Error::TimedOut))
    );
    assert_eq!(
        server.receive(Deadline::IMMEDIATE).unwrap_err(),
        Error::TimedOut
    );
    assert!(!stream.is_retired());
    drop(server);
    drop(stream);
    t.check(
        "native network expired deadline submits no request",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let (attachment, observer) = Channel::pair().unwrap();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let request = server.receive(Deadline::FOREVER).unwrap();
        let h = header(&request);
        let info = net::encode_info(
            PROTOCOL_TCP_STREAM,
            Info {
                rights: net::stream_rights::ALL,
                signals: net::signals::CONNECTED,
                error: 0,
                revision: 1,
                local: None,
                peer: None,
            },
        )
        .unwrap();
        let mut response =
            net::reply_message(h.protocol, h.ordinal, h.request_id + 1, 0, &info[4..], 1).unwrap();
        response.try_push_handle(attachment.into_untyped()).unwrap();
        server.try_send(response).unwrap();
    })
    .unwrap();
    let mut stream = TcpStream::new(client.into_untyped()).unwrap();
    assert!(matches!(
        stream.info(deadline()),
        Err(NetworkError::InvalidReply(_))
    ));
    assert!(stream.is_retired());
    assert_eq!(
        observer.receive(Deadline::IMMEDIATE).unwrap_err(),
        Error::PeerClosed
    );
    worker.join().unwrap();
    t.check(
        "native network wrong reply retires connection and closes attachments",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let operation_deadline = deadline();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let request = server.receive(Deadline::FOREVER).unwrap();
        let r = net::decode_request(request.bytes(), request.handle_count()).unwrap();
        assert_eq!(r.deadline, operation_deadline);
        assert!(matches!(r.operation, Operation::Write { bytes: b"abcdef" }));
        let h = r.header;
        drop(request);
        reply(&server, h, &2u32.to_le_bytes());
        let request = server.receive(Deadline::FOREVER).unwrap();
        let r = net::decode_request(request.bytes(), request.handle_count()).unwrap();
        assert_eq!(r.deadline, operation_deadline);
        assert!(matches!(r.operation, Operation::Write { bytes: b"cdef" }));
        // The second batch may have been accepted. Lose the reply, forcing the
        // client to preserve its confirmed prefix and never replay that batch.
    })
    .unwrap();
    let mut stream = TcpStream::new(client.into_untyped()).unwrap();
    let error = stream.write_all(b"abcdef", operation_deadline).unwrap_err();
    assert_eq!(error.accepted, 2);
    assert_eq!(error.error, NetworkError::OutcomeUnknown(Error::PeerClosed));
    assert!(stream.is_retired());
    assert_eq!(
        stream.write(b"cdef", deadline()),
        Err(NetworkError::Retired)
    );
    worker.join().unwrap();
    t.check(
        "native network ambiguous write preserves confirmed progress without replay",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let operation_deadline = deadline();
    let worker=thread::spawn_joinable(u64::MAX,move||{
        let request=server.receive(Deadline::FOREVER).unwrap();let r=net::decode_request(request.bytes(),0).unwrap();assert_eq!(r.deadline,operation_deadline);assert!(matches!(r.operation,Operation::Write{bytes:b"abc"}));
        server.try_send(net::reply_message(PROTOCOL_TCP_STREAM,r.header.ordinal,r.header.request_id,STATUS_WOULD_BLOCK,&[],0).unwrap()).unwrap();
        let mut request=server.receive(Deadline::FOREVER).unwrap();let r=net::decode_request(request.bytes(),1).unwrap();assert_eq!(r.deadline,operation_deadline);assert!(matches!(r.operation,Operation::Observe{mask,key:0} if mask==net::signals::WRITABLE|net::signals::ERROR));let h=r.header;let sink=request.take_handle(0).unwrap();let mut registration=[0;12];registration[4..].copy_from_slice(&11u64.to_le_bytes());reply(&server,h,&registration);
        let event=net::encode_event(Readiness{subscription:11,key:0,revision:1,signals:net::signals::WRITABLE,error:0}).unwrap();owned::try_send(sink.as_handle(),Message::try_from_bytes(&event).unwrap()).unwrap();
        let request=server.receive(Deadline::FOREVER).unwrap();let r=net::decode_request(request.bytes(),0).unwrap();assert_eq!(r.deadline,operation_deadline);assert!(matches!(r.operation,Operation::Write{bytes:b"abc"}));reply(&server,r.header,&1u32.to_le_bytes());
        let request=server.receive(Deadline::FOREVER).unwrap();let r=net::decode_request(request.bytes(),0).unwrap();assert_eq!(r.deadline,operation_deadline);assert!(matches!(r.operation,Operation::Write{bytes:b"bc"}));server.try_send(net::reply_message(PROTOCOL_TCP_STREAM,r.header.ordinal,r.header.request_id,STATUS_WOULD_BLOCK,&[],0).unwrap()).unwrap();
        let request=server.receive(Deadline::FOREVER).unwrap();let r=net::decode_request(request.bytes(),0).unwrap();assert_eq!(r.deadline,operation_deadline);assert!(matches!(r.operation,Operation::Rearm{subscription:11,revision:1}));reply(&server,r.header,&[]);
        owned::try_send(sink.as_handle(),Message::try_from_bytes(&event).unwrap()).unwrap();
        let request=server.receive(Deadline::FOREVER).unwrap();let r=net::decode_request(request.bytes(),0).unwrap();assert_eq!(r.deadline,operation_deadline);assert!(matches!(r.operation,Operation::Write{bytes:b"bc"}));reply(&server,r.header,&2u32.to_le_bytes());
        assert_eq!(owned::receive(sink.as_handle(),Deadline::FOREVER).unwrap_err(),Error::PeerClosed);
    }).unwrap();
    let mut stream = TcpStream::new(client.into_untyped()).unwrap();
    stream.write_all(b"abc", operation_deadline).unwrap();
    worker.join().unwrap();
    t.check(
        "native network write_all waits and retries only the unaccepted suffix",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let request = server.receive(Deadline::FOREVER).unwrap();
        let h = header(&request);
        let mut data = [0; 15];
        data[4..8].copy_from_slice(&3u32.to_le_bytes());
        data[12..].copy_from_slice(b"abc");
        reply(&server, h, &data);
        let request = server.receive(Deadline::FOREVER).unwrap();
        let h = header(&request);
        let mut body = [0; 12];
        body[..4].copy_from_slice(&1u32.to_le_bytes());
        reply(&server, h, &body);
    })
    .unwrap();
    let mut stream = TcpStream::new(client.into_untyped()).unwrap();
    match stream.read(16, deadline()).unwrap() {
        ReadResult::Data(data) => assert_eq!(data.bytes(), b"abc"),
        _ => panic!("expected data before EOF"),
    }
    assert!(matches!(stream.read(16, deadline()), Ok(ReadResult::Eof)));
    worker.join().unwrap();
    assert_eq!(
        stream.read(16, deadline()).unwrap_err(),
        NetworkError::NotSent(Error::PeerClosed)
    );
    t.check(
        "native network explicit EOF differs from service loss",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let request = server.receive(Deadline::FOREVER).unwrap();
        reply(&server, header(&request), &[0; 12]);
    })
    .unwrap();
    let mut stream = TcpStream::new(client.into_untyped()).unwrap();
    assert!(matches!(
        stream.read(16, deadline()),
        Err(NetworkError::InvalidReply(_))
    ));
    assert!(stream.is_retired());
    worker.join().unwrap();
    t.check(
        "native network zero DATA cannot masquerade as EOF",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let (new_client, new_server) = Channel::pair().unwrap();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let request = server.receive(Deadline::FOREVER).unwrap();
        let h = header(&request);
        let mut response = net::reply_message(
            h.protocol,
            h.ordinal,
            h.request_id,
            0,
            &net::stream_rights::ALL.to_le_bytes(),
            1,
        )
        .unwrap();
        response.try_push_handle(new_client.into_untyped()).unwrap();
        server.try_send(response).unwrap();
        assert_eq!(
            new_server.receive(Deadline::FOREVER).unwrap_err(),
            Error::PeerClosed
        );
    })
    .unwrap();
    let mut network = Network::new(client.into_untyped()).unwrap();
    assert!(matches!(
        network.begin_connect(
            net::Endpoint::v4([127, 0, 0, 1], 1),
            net::stream_rights::READ,
            deadline()
        ),
        Err(NetworkError::InvalidReply(_))
    ));
    assert!(network.is_retired());
    worker.join().unwrap();
    t.check(
        "native network rejects broader logical grants and closes returned object",
        true,
        "",
    );

    let (client, server) = Channel::pair().unwrap();
    let (gate, worker_gate) = Channel::pair().unwrap();
    let operation_deadline = deadline();
    let worker = thread::spawn_joinable(u64::MAX, move || {
        let mut request = server.receive(Deadline::FOREVER).unwrap();
        let r = net::decode_request(request.bytes(), request.handle_count()).unwrap();
        let h = r.header;
        assert_eq!(r.deadline, operation_deadline);
        assert!(matches!(
            r.operation,
            Operation::Observe {
                mask: net::signals::READABLE,
                key: 99
            }
        ));
        let sink = request.take_handle(0).unwrap();
        let mut body = [0; 12];
        body[4..].copy_from_slice(&7u64.to_le_bytes());
        reply(&server, h, &body);
        worker_gate.receive(Deadline::FOREVER).unwrap();
        let event = net::encode_event(Readiness {
            subscription: 7,
            key: 99,
            revision: 3,
            signals: net::signals::READABLE,
            error: 0,
        })
        .unwrap();
        owned::try_send(sink.as_handle(), Message::try_from_bytes(&event).unwrap()).unwrap();
        let request = server.receive(Deadline::FOREVER).unwrap();
        let r = net::decode_request(request.bytes(), 0).unwrap();
        assert_eq!(r.deadline, operation_deadline);
        assert!(matches!(
            r.operation,
            Operation::Rearm {
                subscription: 7,
                revision: 3
            }
        ));
        reply(&server, r.header, &[]);
        let event = net::encode_event(Readiness {
            subscription: 7,
            key: 99,
            revision: 4,
            signals: 0,
            error: 0,
        })
        .unwrap();
        owned::try_send(sink.as_handle(), Message::try_from_bytes(&event).unwrap()).unwrap();
        let request = server.receive(Deadline::FOREVER).unwrap();
        let r = net::decode_request(request.bytes(), 0).unwrap();
        assert!(matches!(r.operation, Operation::Cancel { subscription: 7 }));
        reply(&server, r.header, &[]);
    })
    .unwrap();
    let mut stream = TcpStream::new(client.into_untyped()).unwrap();
    let mut watch = stream
        .observe(net::signals::READABLE, 99, operation_deadline)
        .unwrap();
    assert_eq!(
        watch.receive(Deadline::IMMEDIATE),
        Err(NetworkError::NotSent(Error::TimedOut))
    );
    assert_eq!(
        stream.rearm(&mut watch, operation_deadline),
        Err(NetworkError::NotSent(Error::InvalidArgs))
    );
    gate.try_send(Message::new()).unwrap();
    let event = watch.receive(operation_deadline).unwrap();
    assert_eq!(event.revision, 3);
    let (other, other_peer) = Channel::pair().unwrap();
    let mut other = TcpStream::new(other.into_untyped()).unwrap();
    assert_eq!(
        other.rearm(&mut watch, operation_deadline),
        Err(NetworkError::NotSent(Error::InvalidArgs))
    );
    drop(other);
    drop(other_peer);
    stream.rearm(&mut watch, operation_deadline).unwrap();
    let event = watch.receive(operation_deadline).unwrap();
    assert_eq!(event.revision, 4);
    assert_eq!(event.signals, 0);
    stream.cancel(&mut watch, operation_deadline).unwrap();
    assert_eq!(
        watch.receive(Deadline::IMMEDIATE),
        Err(NetworkError::Retired)
    );
    worker.join().unwrap();
    t.check(
        "native network observer timeout, revision rearm and source identity",
        true,
        "",
    );
}
