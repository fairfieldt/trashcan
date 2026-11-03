//! Native wire acceptance: actual daemon RPC, rights and observer lifecycle.
#![no_std]
#![no_main]
extern crate alloc;
#[path = "../fixture.rs"]
mod extended;
#[allow(dead_code)]
#[path = "../seed.rs"]
mod seed;
use alloc::vec::Vec;
use dross_rt::{
    Error, Handle,
    deadline::{Deadline, Duration},
    owned::{self, Channel, Message, OwnedHandle},
    protocol::network::{
        self, Endpoint, Network, NetworkError, ReadResult, TcpStream, listener_rights as lr,
        signals, stream_rights as sr,
    },
    protocol::{
        frame::{Frame, Header},
        schema::*,
    },
    sys,
};
dross_rt::main!(main);
#[derive(Debug)]
enum Failure {
    Network(NetworkError),
    Sys(Error),
    Resolve(dross_rt::protocol::resolver::ResolveError),
    Check(&'static str),
}
impl From<NetworkError> for Failure {
    fn from(e: NetworkError) -> Self {
        Self::Network(e)
    }
}
impl From<dross_rt::protocol::resolver::ResolveError> for Failure {
    fn from(e: dross_rt::protocol::resolver::ResolveError) -> Self {
        Self::Resolve(e)
    }
}
impl From<network::WriteAllError> for Failure {
    fn from(e: network::WriteAllError) -> Self {
        Self::Network(e.error)
    }
}
impl From<Error> for Failure {
    fn from(e: Error) -> Self {
        Self::Sys(e)
    }
}
fn check(value: bool, name: &'static str) -> Result<(), Failure> {
    if value {
        Ok(())
    } else {
        Err(Failure::Check(name))
    }
}
fn transient(e: NetworkError) -> bool {
    matches!(e, NetworkError::WouldBlock)
}
fn transfer(
    sender: &mut TcpStream,
    receiver: &mut TcpStream,
    bytes: &[u8],
    deadline: Deadline,
) -> Result<(), Failure> {
    let mut written = 0;
    let mut read = 0;
    while written < bytes.len() || read < bytes.len() {
        let mut progress = false;
        if written < bytes.len() {
            let end = (written + network::MAX_INLINE).min(bytes.len());
            match sender.write(&bytes[written..end], deadline) {
                Ok(n) => {
                    check(n > 0, "write progress")?;
                    written += n;
                    progress = true;
                }
                Err(e) if transient(e) => {}
                Err(e) => return Err(e.into()),
            }
        }
        if read < bytes.len() {
            match receiver.read((bytes.len() - read).min(network::MAX_INLINE), deadline) {
                Ok(ReadResult::Data(data)) => {
                    check(
                        data.bytes() == &bytes[read..read + data.len()],
                        "exact binary stream bytes",
                    )?;
                    read += data.len();
                    progress = true
                }
                Ok(ReadResult::Eof) => return Err(Failure::Check("unexpected EOF")),
                Err(e) if transient(e) => {}
                Err(e) => return Err(e.into()),
            }
        }
        if deadline.has_elapsed() {
            return Err(Failure::Check("transfer deadline"));
        }
        if !progress {
            sys::yield_now();
        }
    }
    Ok(())
}
fn eof(stream: &mut TcpStream, deadline: Deadline) -> Result<(), Failure> {
    loop {
        match stream.read(64, deadline) {
            Ok(ReadResult::Eof) => return Ok(()),
            Ok(ReadResult::Data(_)) => return Err(Failure::Check("unexpected bytes before EOF")),
            Err(e) if transient(e) => {
                if deadline.has_elapsed() {
                    return Err(Failure::Check("EOF deadline"));
                }
                sys::yield_now()
            }
            Err(e) => return Err(e.into()),
        }
    }
}
fn raw_request(
    protocol: u32,
    ordinal: u32,
    id: u64,
    body: &[u8],
    handles: usize,
) -> Result<Message, Failure> {
    let mut header = [0; 32];
    Header::new(protocol, 1, KIND_REQUEST, ordinal, id, body.len(), handles)
        .map_err(|_| Failure::Check("raw header"))?
        .encode(&mut header)
        .map_err(|_| Failure::Check("raw header encode"))?;
    let mut m = Message::try_from_bytes(&header)?;
    m.try_extend_bytes(body)?;
    Ok(m)
}
fn raw_call(
    channel: &OwnedHandle,
    mut request: Message,
    deadline: Deadline,
) -> Result<Message, Failure> {
    loop {
        match owned::try_send(channel.as_handle(), request) {
            Ok(()) => break,
            Err(e) => {
                request = e.message;
                if e.error != Error::ShouldWait {
                    return Err(e.error.into());
                }
                if deadline.has_elapsed() {
                    return Err(Failure::Check("raw send deadline"));
                }
                sys::yield_now();
            }
        }
    }
    Ok(owned::receive(channel.as_handle(), deadline)?)
}
fn no_wait_observer(
    factory: &OwnedHandle,
    remote: Endpoint,
    deadline: Deadline,
) -> Result<(), Failure> {
    let mut body = [0; 40];
    body[..8].copy_from_slice(&deadline.as_monotonic_ns().to_le_bytes());
    body[8..32].copy_from_slice(&remote.encode());
    body[32..36].copy_from_slice(&sr::ALL.to_le_bytes());
    let response = raw_call(
        factory,
        raw_request(PROTOCOL_NETWORK, NETWORK_CONNECT, u64::MAX - 1, &body, 0)?,
        deadline,
    )?;
    let mut response = response;
    let frame = Frame::parse(response.bytes(), response.handle_count())
        .map_err(|_| Failure::Check("raw connect framing"))?;
    check(
        frame.body.len() == 8 && frame.body[..4] == [0; 4] && response.handle_count() == 1,
        "raw connect grant",
    )?;
    let stream = response
        .take_handle(0)
        .ok_or(Failure::Check("raw stream cap"))?;
    let (receiver, sink) = Channel::pair()?;
    let weak = sink
        .try_duplicate(
            dross_abi::rights::READ | dross_abi::rights::WRITE | dross_abi::rights::TRANSFER,
        )?
        .into_untyped();
    drop(sink);
    let mut body = [0; 24];
    body[..8].copy_from_slice(&deadline.as_monotonic_ns().to_le_bytes());
    body[8..12].copy_from_slice(&signals::WRITABLE.to_le_bytes());
    body[16..24].copy_from_slice(&47u64.to_le_bytes());
    let mut request = raw_request(PROTOCOL_TCP_STREAM, TCP_STREAM_OBSERVE, 1, &body, 1)?;
    request
        .try_push_handle(weak)
        .map_err(|e| Failure::Sys(e.error))?;
    let response = raw_call(&stream, request, deadline)?;
    let frame = Frame::parse(response.bytes(), response.handle_count())
        .map_err(|_| Failure::Check("observer rejection framing"))?;
    check(
        frame.body.len() == 4
            && i32::from_le_bytes(frame.body.try_into().unwrap()) < 0
            && response.handle_count() == 0,
        "observer without WAIT rejected before ACK",
    )?;
    check(
        receiver.receive(Deadline::IMMEDIATE).err() == Some(Error::PeerClosed),
        "rejected observer releases sink",
    )?;
    Ok(())
}
fn run(boot: Handle) -> Result<(), Failure> {
    let mut seed = seed::Seed::read(boot)?;
    if seed.value("mode") == Some("external") {
        return extended::external(&mut seed);
    }
    let connect_cap = seed.take("connect").ok_or(Error::BadHandle)?;
    // This alias has no concurrent reader. At the final raw-wire test the SDK
    // owner is dropped first, then a terminal high request ID avoids reuse.
    let raw_factory = connect_cap.try_duplicate(network::TRANSPORT_RIGHTS)?;
    let mut connect = Network::new(connect_cap)?;
    let mut listen = Network::new(seed.take("listen").ok_or(Error::BadHandle)?)?;
    let mut datagram = Network::new(seed.take("datagram").ok_or(Error::BadHandle)?)?;
    let mut resolver = dross_rt::protocol::resolver::Resolver::new(
        seed.take("resolver").ok_or(Error::BadHandle)?,
    )?;
    let dns_ip = seed.value("dns-server").unwrap_or("127.0.0.1");
    let dns_port = seed
        .value("dns-port")
        .unwrap_or("53053")
        .parse::<u16>()
        .map_err(|_| Error::InvalidArgs)?;
    let dns_endpoint = if dns_ip.contains(':') {
        Endpoint::v6(
            dns_ip
                .parse::<core::net::Ipv6Addr>()
                .map_err(|_| Error::InvalidArgs)?
                .octets(),
            0,
            dns_port,
        )
    } else {
        Endpoint::v4(
            dns_ip
                .parse::<core::net::Ipv4Addr>()
                .map_err(|_| Error::InvalidArgs)?
                .octets(),
            dns_port,
        )
    };
    check(seed.remaining() == 0, "explicit bootstrap grants only")?;
    drop(seed);
    let deadline = Deadline::from_now(Duration::from_secs(35));
    let ci = connect.describe(deadline)?;
    let li = listen.describe(deadline)?;
    check(
        ci.rights == network::CONNECT && li.rights == network::LISTEN,
        "restricted factory grants",
    )?;
    check(
        matches!(
            connect.listen_tcp(
                Endpoint::v4([127, 0, 0, 1], 0),
                lr::ALL,
                sr::ALL,
                1,
                deadline
            ),
            Err(NetworkError::Denied)
        ),
        "connect cannot listen",
    )?;
    check(
        matches!(
            listen.begin_connect(Endpoint::v4([127, 0, 0, 1], 9), sr::ALL, deadline),
            Err(NetworkError::Denied)
        ),
        "listen cannot connect",
    )?;
    extended::run(
        &mut connect,
        &mut listen,
        &mut datagram,
        &mut resolver,
        dns_endpoint,
        deadline,
    )?;
    let mut listener = listen.listen_tcp(
        Endpoint::v4([127, 0, 0, 1], 0),
        lr::ALL,
        sr::ALL,
        2,
        deadline,
    )?;
    let local = listener
        .info(deadline)?
        .local
        .ok_or(Failure::Check("listener local address"))?;
    check(local.port != 0, "ephemeral listener port")?;
    let mut watch = listener.observe(signals::ACCEPTABLE | signals::ERROR, 41, deadline)?;
    check(
        matches!(
            watch.receive(Deadline::IMMEDIATE),
            Err(NetworkError::NotSent(Error::ShouldWait | Error::TimedOut))
        ),
        "initial observer not falsely ready",
    )?;
    check(
        matches!(
            listener.observe(signals::ACCEPTABLE, 99, deadline),
            Err(NetworkError::Service(
                dross_rt::protocol::schema::STATUS_BAD_STATE
            ))
        ),
        "one observer per object",
    )?;
    let mut client = connect.begin_connect(local, sr::ALL, deadline)?;
    let mut connected = client.observe(
        signals::CONNECTED | signals::ERROR | signals::PEER_CLOSED,
        42,
        deadline,
    )?;
    let event = connected.receive(deadline)?;
    check(
        event.error == 0 && event.signals & signals::CONNECTED != 0,
        "connecting object readiness",
    )?;
    client.cancel(&mut connected, deadline)?;
    let event = watch.receive(deadline)?;
    check(
        event.key == 41 && event.signals & signals::ACCEPTABLE != 0,
        "listener engine readiness event",
    )?;
    listener.rearm(&mut watch, deadline)?;
    check(
        watch.receive(deadline)?.signals & signals::ACCEPTABLE != 0,
        "ready rearm immediate",
    )?;
    let mut server = loop {
        match listener.accept(sr::ALL, deadline) {
            Ok(s) => break s,
            Err(NetworkError::WouldBlock) => sys::yield_now(),
            Err(e) => return Err(e.into()),
        }
        if deadline.has_elapsed() {
            return Err(Failure::Check("accept deadline"));
        }
    };
    listener.cancel(&mut watch, deadline)?;
    check(
        matches!(server.read(32, deadline), Err(NetworkError::WouldBlock)),
        "empty live socket is WouldBlock",
    )?;
    check(client.write(&[], deadline)? == 0, "empty write")?;
    let mut readable = server.observe(
        signals::READABLE | signals::ERROR | signals::PEER_CLOSED,
        43,
        deadline,
    )?;
    let length = 96 * 1024 + 17;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(length)
        .map_err(|_| Error::NoMemory)?;
    for i in 0..length {
        payload.push((i as u8).wrapping_mul(67) ^ (i >> 8) as u8);
    }
    transfer(&mut client, &mut server, &payload, deadline)?;
    let first = readable.receive(deadline)?;
    check(
        first.error == 0 && first.signals & signals::READABLE != 0,
        "readiness transition survives read",
    )?;
    server.rearm(&mut readable, deadline)?;
    let later = readable.receive(deadline)?;
    check(
        later.revision > first.revision,
        "rearm catches changed revision",
    )?;
    server.cancel(&mut readable, deadline)?;
    // Closing a receiver must permit immediate replacement even when the service
    // has not yet reached its end-of-loop housekeeping.
    let old = server.observe(signals::READABLE, 44, deadline)?;
    drop(old);
    let mut replacement = server.observe(signals::READABLE, 45, deadline)?;
    server.cancel(&mut replacement, deadline)?;
    client.shutdown_send(deadline)?;
    eof(&mut server, deadline)?;
    check(
        matches!(client.write(b"closed", deadline), Err(NetworkError::Closed)),
        "shutdown-send rejects further writes",
    )?;
    transfer(
        &mut server,
        &mut client,
        b"reply after peer half-close\0exact",
        deadline,
    )?;
    server.shutdown_send(deadline)?;
    eof(&mut client, deadline)?;
    drop(client);
    drop(server);
    drop(listener);
    // Narrow accepted streams cannot gain write authority, and the listener
    // cannot grant rights beyond its bootstrap-authorized accepted-stream view.
    let accepted = sr::READ | sr::INFO | sr::OBSERVE;
    let mut narrow = listen.listen_tcp(
        Endpoint::v4([127, 0, 0, 1], 0),
        lr::ALL,
        accepted,
        1,
        deadline,
    )?;
    let local = narrow.info(deadline)?.local.unwrap();
    let mut client = connect.connect_tcp(local, sr::ALL, deadline)?;
    check(
        matches!(
            narrow.accept(sr::WRITE, deadline),
            Err(NetworkError::Denied)
        ),
        "accepted rights cannot expand",
    )?;
    let mut server = loop {
        match narrow.accept(accepted, deadline) {
            Ok(s) => break s,
            Err(NetworkError::WouldBlock) => sys::yield_now(),
            Err(e) => return Err(e.into()),
        }
        if deadline.has_elapsed() {
            return Err(Failure::Check("narrow accept deadline"));
        }
    };
    check(
        matches!(server.write(b"denied", deadline), Err(NetworkError::Denied)),
        "narrow stream write denied",
    )?;
    check(
        server.info(deadline)?.rights == accepted,
        "narrow stream grant",
    )?;
    transfer(&mut client, &mut server, b"narrow read\0", deadline)?;
    let mut probe = client.observe(signals::WRITABLE, 46, deadline)?;
    check(
        probe.receive(deadline)?.signals & signals::WRITABLE != 0,
        "helper watch cleaned before reobserve",
    )?;
    client.cancel(&mut probe, deadline)?;
    client.abort(deadline)?;
    check(client.is_retired(), "abort retires local control")?;
    drop(connect);
    no_wait_observer(&raw_factory, local, deadline)?;
    check(
        listen.describe(deadline)?.rights == network::LISTEN,
        "bad observer cannot kill factory",
    )?;
    check(
        narrow.info(deadline)?.rights == lr::ALL,
        "bad observer cannot kill another object",
    )?;
    dross_rt::sys::log(
        "native-net-test: PASS rights/connecting/readiness/rearm/close/binary96KiB/EOF/halfclose/attenuation/observer-isolation",
    );
    Ok(())
}
fn main(boot: Handle) -> i32 {
    match run(boot) {
        Ok(()) => 0,
        Err(error) => {
            let detail = match error {
                Failure::Network(e) => alloc::format!("network {e:?}"),
                Failure::Resolve(e) => alloc::format!("resolver {e:?}"),
                Failure::Sys(e) => alloc::format!("syscall {e:?}"),
                Failure::Check(name) => alloc::format!("check {name}"),
            };
            dross_rt::sys::log(&alloc::format!("native-net-test: FAILED {detail}"));
            1
        }
    }
}
