//! Native public-protocol acceptance. The broker is a separate process; echo
//! providers are test threads with explicitly moved raw ownership at the boundary.
use super::Tally;
use alloc::{string::String, vec::Vec};
use dross_abi::{Error, rights};
use dross_rt::{
    Handle,
    bootstrap::Bootstrap,
    deadline::{Deadline, Duration},
    launch::{self, SpawnOptions},
    owned::{self, Channel, Memory, Message, OwnedHandle, Process},
    protocol::{
        self,
        bootstrap::{BootstrapV2, Builder},
        directory::{self, Client, ClientError, Rule},
        frame::Frame,
        schema::*,
    },
    sys, thread,
};
const RW: u32 = rights::READ | rights::WRITE | rights::TRANSFER | rights::WAIT;
fn end() -> Deadline {
    Deadline::from_now(Duration::from_secs(5))
}
fn ok(v: bool) -> Result<(), Error> {
    if v { Ok(()) } else { Err(Error::InvalidArgs) }
}
fn pe(_: protocol::ProtocolError) -> Error {
    Error::InvalidArgs
}
fn ce(e: ClientError) -> Error {
    match e {
        ClientError::NotSent(e) | ClientError::OutcomeUnknown(e) => e,
        _ => Error::InvalidArgs,
    }
}
fn rule(rights: u32) -> Rule {
    Rule {
        name: String::from("echo"),
        protocol: PROTOCOL_ECHO,
        rights,
    }
}
fn put_handle(b: &mut Builder, name: &str, p: u32, h: OwnedHandle) -> Result<(), Error> {
    b.try_handle(name, p, h).map_err(|_| Error::InvalidArgs)
}
fn launch_image(raw: Handle, name: &str, b: Builder) -> Result<Process, Error> {
    let image = unsafe { OwnedHandle::from_raw(sys::dup(raw, rights::READ)?) }
        .try_into_memory()
        .map_err(|e| e.error)?;
    let m = b.build().map_err(|_| Error::InvalidArgs)?;
    launch::spawn(image.as_handle(), name, m, SpawnOptions::default()).map_err(|e| e.error)
}
fn send(h: &OwnedHandle, mut m: Message) -> Result<(), Error> {
    let until = end();
    loop {
        match owned::try_send(h.as_handle(), m) {
            Ok(()) => return Ok(()),
            Err(e) => {
                m = e.message;
                if e.error != Error::ShouldWait {
                    return Err(e.error);
                }
                if until.has_elapsed() {
                    return Err(Error::TimedOut);
                }
                sys::yield_now();
            }
        }
    }
}
fn echo(endpoint: &OwnedHandle, id: u64, body: &[u8]) -> Result<(), Error> {
    send(
        endpoint,
        directory::message(
            PROTOCOL_ECHO,
            MAJOR_ECHO,
            KIND_REQUEST,
            ECHO_ECHO,
            id,
            body,
            0,
        )
        .map_err(pe)?,
    )?;
    let m = owned::receive(endpoint.as_handle(), end())?;
    let f = Frame::parse(m.bytes(), m.handle_count()).map_err(|_| Error::InvalidArgs)?;
    ok(f.header.supported(PROTOCOL_ECHO, MAJOR_ECHO)
        && f.header.kind == KIND_REPLY
        && f.header.ordinal == ECHO_ECHO
        && f.header.request_id == id
        && m.handle_count() == 0
        && f.body == body)
}

fn provider(control: Channel, stall: bool) -> Result<thread::JoinHandle, Error> {
    // The typed channel's sole owner moves into the new thread as a raw slot.
    // On THREAD_START failure it is re-adopted and closed here.
    let raw = control.into_raw();
    match thread::spawn_joinable(u64::MAX, move || {
        let mut control = Some(unsafe { OwnedHandle::from_raw(raw) });
        let mut connections: Vec<OwnedHandle> = Vec::new();
        let sleep =
            unsafe { OwnedHandle::<owned::Untyped>::from_raw(sys::notify_create().unwrap()) };
        if let Some(c) = &control {
            let _ = sys::channel_bind(c.as_handle().as_raw(), sleep.as_handle().as_raw(), 1);
        }
        loop {
            if let Some(c) = &control {
                match owned::receive(c.as_handle(), Deadline::IMMEDIATE) {
                    Ok(m) => match directory::decode_offer(m) {
                        Ok(offer) => {
                            if offer.protocol != PROTOCOL_ECHO {
                                break;
                            }
                            let _ = sys::channel_bind(
                                offer.endpoint.as_handle().as_raw(),
                                sleep.as_handle().as_raw(),
                                1,
                            );
                            connections.push(offer.endpoint);
                            if !stall
                                && send(c, directory::offer_reply(offer.id, STATUS_OK).unwrap())
                                    .is_err()
                            {
                                break;
                            }
                        }
                        Err(_) => break,
                    },
                    Err(Error::ShouldWait | Error::TimedOut) => {}
                    Err(_) => {
                        control.take();
                    }
                }
            }
            let mut i = 0;
            while i < connections.len() {
                match owned::receive(connections[i].as_handle(), Deadline::IMMEDIATE) {
                    Ok(m) => {
                        let f = match Frame::parse(m.bytes(), m.handle_count()) {
                            Ok(f) => f,
                            Err(_) => {
                                connections.remove(i);
                                continue;
                            }
                        };
                        if !f.header.supported(PROTOCOL_ECHO, MAJOR_ECHO)
                            || f.header.kind != KIND_REQUEST
                            || f.header.ordinal != ECHO_ECHO
                            || m.handle_count() != 0
                            || f.body.len() > MAX_ECHO_BYTES
                        {
                            connections.remove(i);
                            continue;
                        }
                        let reply = directory::message(
                            PROTOCOL_ECHO,
                            MAJOR_ECHO,
                            KIND_REPLY,
                            ECHO_ECHO,
                            f.header.request_id,
                            f.body,
                            0,
                        )
                        .unwrap();
                        if send(&connections[i], reply).is_err() {
                            connections.remove(i);
                            continue;
                        }
                    }
                    Err(Error::ShouldWait | Error::TimedOut) => {}
                    Err(_) => {
                        connections.remove(i);
                        continue;
                    }
                }
                i += 1;
            }
            if control.is_none() && connections.is_empty() {
                break;
            }
            let _ = sys::notify_wait(
                sleep.as_handle().as_raw(),
                Deadline::from_now(Duration::from_millis(2)).as_monotonic_ns(),
            );
        }
    }) {
        Ok(t) => Ok(t),
        Err(e) => {
            drop(unsafe { OwnedHandle::<owned::Untyped>::from_raw(raw) });
            Err(e)
        }
    }
}
fn bootstrap_checks() -> Result<(), Error> {
    let memory = Memory::create(4096, 0)?.into_untyped();
    let number = memory.as_handle().as_raw();
    let mut b = Builder::new();
    put_handle(&mut b, "memory", PROTOCOL_ECHO, memory)?;
    b.try_value("argument", b"hello\0world").map_err(pe)?;
    let second = Memory::create(4096, 0)?.into_untyped();
    let duplicate = b
        .try_handle("memory", PROTOCOL_ECHO, second)
        .err()
        .ok_or(Error::InvalidArgs)?;
    drop(duplicate.handle);
    let m = b.build().map_err(|_| Error::InvalidArgs)?;
    let mut b = BootstrapV2::decode(m).map_err(pe)?;
    ok(b.value("argument") == Some(b"hello\0world"))?;
    let memory = b.take("memory").ok_or(Error::BadHandle)?;
    ok(b.take("memory").is_none() && b.remaining_handles() == 0)?;
    drop(b);
    ok(sys::mem_size(number) == Ok(4096))?;
    drop(memory);
    ok(sys::mem_size(number) == Err(Error::BadHandle))?;
    // Invalid attachment references must drop the complete owned message.
    let memory = Memory::create(4096, 0)?.into_untyped();
    let number = memory.as_handle().as_raw();
    let mut b = Builder::new();
    put_handle(&mut b, "memory", PROTOCOL_ECHO, memory)?;
    let mut m = b.build().map_err(|_| Error::InvalidArgs)?;
    // header32 + bodycount4 + record8 + name6 + protocol4 => index.
    m.bytes_mut()[54..56].copy_from_slice(&1u16.to_le_bytes());
    ok(BootstrapV2::decode(m).is_err())?;
    ok(sys::mem_size(number) == Err(Error::BadHandle))
}
fn connection_body(name: &str) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&PROTOCOL_ECHO.to_le_bytes());
    b.extend_from_slice(&RW.to_le_bytes());
    b.extend_from_slice(&(name.len() as u16).to_le_bytes());
    b.extend_from_slice(&[0, 0]);
    b.extend_from_slice(name.as_bytes());
    b
}
fn directory_checks(t: &mut Tally, b: &Bootstrap) -> Result<(), Error> {
    let image = b.handle("module/serviced").ok_or(Error::NotFound)?;
    let (pc, ps) = Channel::pair()?;
    let mut worker = provider(pc, false)?;
    let (client, server) = Channel::pair()?;
    let (admin, admin_server) = Channel::pair()?;
    let (cc, cs) = Channel::pair()?;
    let (malformed, malformed_server) = Channel::pair()?;
    let (abandon, abandon_server) = Channel::pair()?;
    let mut seed = Builder::new();
    put_handle(
        &mut seed,
        "provider/echo",
        PROTOCOL_PROVIDER,
        ps.into_untyped(),
    )?;
    seed.try_value("service/echo", &PROTOCOL_ECHO.to_le_bytes())
        .map_err(pe)?;
    let rules = directory::encode_rules(&[rule(directory::ENDPOINT_RIGHTS)]).map_err(pe)?;
    for (label, endpoint) in [
        ("rust", server),
        ("c", cs),
        ("malformed", malformed_server),
        ("abandon", abandon_server),
    ] {
        put_handle(
            &mut seed,
            &alloc::format!("view/{label}"),
            PROTOCOL_DIRECTORY,
            endpoint.into_untyped(),
        )?;
        seed.try_value(&alloc::format!("rules/{label}"), &rules)
            .map_err(pe)?;
    }
    put_handle(
        &mut seed,
        "admin/registrar",
        PROTOCOL_DIRECTORY,
        admin_server.into_untyped(),
    )?;
    let daemon = launch_image(image, "serviced-test", seed)?;
    let mut c = Client::new(client.into_untyped());
    let mut a = Client::new(admin.into_untyped());
    let cprocess = if let Some(image) = b.handle("module/c-api-test") {
        let mut boot = Builder::new();
        put_handle(
            &mut boot,
            "directory",
            PROTOCOL_DIRECTORY,
            cc.into_untyped(),
        )?;
        Some(launch_image(image, "c-api-test", boot)?)
    } else {
        drop(cc);
        None
    };
    ok(c.list(end()).map_err(ce)? == [rule(directory::ENDPOINT_RIGHTS)])?;
    ok(matches!(
        c.connect("forbidden", PROTOCOL_ECHO, RW, end()),
        Err(ClientError::Status(STATUS_DENIED))
    ))?;
    let first = c.connect("echo", PROTOCOL_ECHO, RW, end()).map_err(ce)?;
    echo(&first.endpoint, 1, b"Rust\0echo\xff")?;
    ok(
        sys::dup(first.endpoint.as_handle().as_raw(), rights::DUPLICATE)
            == Err(Error::AccessDenied),
    )?;
    t.check(
        "service directory fresh restricted connect and binary roundtrip",
        true,
        "",
    );
    let mut delegated = c.delegate(&[rule(RW)], end()).map_err(ce)?;
    ok(delegated.list(end()).map_err(ce)? == [rule(RW)])?;
    ok(matches!(
        delegated.connect("echo", PROTOCOL_ECHO, RW | rights::DUPLICATE, end()),
        Err(ClientError::Status(STATUS_DENIED))
    ))?;
    let second = delegated
        .connect("echo", PROTOCOL_ECHO, RW, end())
        .map_err(ce)?;
    echo(&second.endpoint, 2, b"delegated")?;
    ok(matches!(
        c.delegate(
            &[Rule {
                name: String::from("forbidden"),
                protocol: PROTOCOL_ECHO,
                rights: RW
            }],
            end()
        ),
        Err(ClientError::Status(STATUS_DENIED))
    ))?;
    let (denied_provider, denied_server) = Channel::pair()?;
    ok(matches!(
        delegated.register(
            "echo",
            PROTOCOL_ECHO,
            RW,
            denied_server.into_untyped(),
            end()
        ),
        Err(ClientError::Status(STATUS_DENIED))
    ))?;
    ok(denied_provider.receive(end()).err() == Some(Error::PeerClosed))?;
    t.check(
        "service delegation cannot widen names rights or registration authority",
        true,
        "",
    );
    // Framing violations terminate exactly the offending view, not the broker.
    let mut bad = directory::message(
        PROTOCOL_DIRECTORY,
        MAJOR_DIRECTORY,
        KIND_REQUEST,
        DIRECTORY_LIST,
        1,
        &[],
        0,
    )
    .map_err(pe)?;
    bad.bytes_mut()[28] = 1;
    malformed.try_send(bad).map_err(|e| e.error)?;
    ok(malformed.receive(end()).err() == Some(Error::PeerClosed))?;
    drop(malformed);
    let mut unsupported = directory::message(
        PROTOCOL_DIRECTORY,
        MAJOR_DIRECTORY,
        KIND_REQUEST,
        999,
        1,
        &[],
        0,
    )
    .map_err(pe)?;
    unsupported.bytes_mut()[26] = 1;
    abandon.try_send(unsupported).map_err(|e| e.error)?;
    let m = abandon.receive(end())?;
    let f = Frame::parse(m.bytes(), 0).map_err(|_| Error::InvalidArgs)?;
    ok(f.body == STATUS_UNSUPPORTED.to_le_bytes())?;
    drop(abandon);
    echo(&first.endpoint, 3, b"other view survived")?;
    t.check(
        "service malformed view isolation and Unsupported reply",
        true,
        "",
    );
    if let Some(child) = cprocess {
        ok(child.wait_until(end())? == 0)?;
        t.check(
            "native C bootstrap directory and binary service roundtrip",
            true,
            "",
        );
    }
    // Replacing registration changes future instance IDs. Existing connections
    // retain the old provider's endpoint and keep working independently.
    let (pc, ps) = Channel::pair()?;
    let mut replacement = provider(pc, false)?;
    a.register(
        "echo",
        PROTOCOL_ECHO,
        directory::ENDPOINT_RIGHTS,
        ps.into_untyped(),
        end(),
    )
    .map_err(ce)?;
    let third = c.connect("echo", PROTOCOL_ECHO, RW, end()).map_err(ce)?;
    ok(first.instance != third.instance)?;
    echo(&first.endpoint, 4, b"old provider still direct")?;
    echo(&third.endpoint, 5, b"new provider")?;
    t.check(
        "service explicit reconnect replacement keeps existing connections",
        true,
        "",
    );
    // No ACK: bounded setup expiration retires the control endpoint, releases
    // retained client backing, and leaves the view reusable for explicit retry.
    let (pc, ps) = Channel::pair()?;
    let mut stalled = provider(pc, true)?;
    a.register(
        "echo",
        PROTOCOL_ECHO,
        directory::ENDPOINT_RIGHTS,
        ps.into_untyped(),
        end(),
    )
    .map_err(ce)?;
    ok(matches!(
        c.connect("echo", PROTOCOL_ECHO, RW, end()),
        Err(ClientError::Status(STATUS_TIMED_OUT))
    ))?;
    ok(matches!(
        c.connect("echo", PROTOCOL_ECHO, RW, end()),
        Err(ClientError::Status(STATUS_UNAVAILABLE))
    ))?;
    stalled.wait_until(end().as_monotonic_ns())?;
    t.check(
        "service unresponsive provider setup deadline and closure cleanup",
        true,
        "",
    );
    drop(first);
    drop(second);
    drop(third);
    drop(delegated);
    drop(c);
    drop(a);
    ok(daemon.wait_until(end())? == 0)?;
    worker.wait_until(end().as_monotonic_ns())?;
    replacement.wait_until(end().as_monotonic_ns())?;
    t.check(
        "service all directory views close and broker exits cleanly",
        true,
        "",
    );
    Ok(())
}

fn regression_checks() -> Result<(), Error> {
    // Deterministic retirement regression: [A/other, A/echo, B/echo]. A's
    // outbound kernel queue and eight broker replies are full. Retiring echo
    // answers A, closes saturated A, and removes an EARLIER other setup. B's
    // shifted echo setup must still be retired rather than reach a replacement.
    let mut server = directory::Server::new()?;
    let (other_control, other_server) = Channel::pair()?;
    server
        .add_provider(
            "other",
            PROTOCOL_ECHO,
            directory::ENDPOINT_RIGHTS,
            other_server.into_untyped(),
        )
        .map_err(pe)?;
    let (echo_control, echo_server) = Channel::pair()?;
    server
        .add_provider(
            "echo",
            PROTOCOL_ECHO,
            directory::ENDPOINT_RIGHTS,
            echo_server.into_untyped(),
        )
        .map_err(pe)?;
    let (a, sa) = Channel::pair()?;
    let (b, sb) = Channel::pair()?;
    for _ in 0..dross_abi::CHANNEL_DEPTH {
        sa.try_send(Message::try_from_bytes(b"full")?)
            .map_err(|e| e.error)?;
    }
    server
        .add_view(
            sa.into_untyped(),
            alloc::vec![
                rule(RW),
                Rule {
                    name: String::from("other"),
                    protocol: PROTOCOL_ECHO,
                    rights: RW
                }
            ],
            false,
        )
        .map_err(pe)?;
    server
        .add_view(sb.into_untyped(), alloc::vec![rule(RW)], false)
        .map_err(pe)?;
    a.try_send(
        directory::message(
            PROTOCOL_DIRECTORY,
            MAJOR_DIRECTORY,
            KIND_REQUEST,
            DIRECTORY_CONNECT,
            1,
            &connection_body("other"),
            0,
        )
        .map_err(pe)?,
    )
    .map_err(|e| e.error)?;
    server.step();
    a.try_send(
        directory::message(
            PROTOCOL_DIRECTORY,
            MAJOR_DIRECTORY,
            KIND_REQUEST,
            DIRECTORY_CONNECT,
            2,
            &connection_body("echo"),
            0,
        )
        .map_err(pe)?,
    )
    .map_err(|e| e.error)?;
    b.try_send(
        directory::message(
            PROTOCOL_DIRECTORY,
            MAJOR_DIRECTORY,
            KIND_REQUEST,
            DIRECTORY_CONNECT,
            1,
            &connection_body("echo"),
            0,
        )
        .map_err(pe)?,
    )
    .map_err(|e| e.error)?;
    server.step();
    for id in 3..11 {
        a.try_send(
            directory::message(
                PROTOCOL_DIRECTORY,
                MAJOR_DIRECTORY,
                KIND_REQUEST,
                999,
                id,
                &[],
                0,
            )
            .map_err(pe)?,
        )
        .map_err(|e| e.error)?;
    }
    server.step();
    server.step();
    drop(echo_control);
    server.step();
    server.step();
    let m = b.receive(end())?;
    let f = Frame::parse(m.bytes(), m.handle_count()).map_err(|_| Error::InvalidArgs)?;
    ok(f.header.request_id == 1
        && f.body == STATUS_UNAVAILABLE.to_le_bytes()
        && m.handle_count() == 0)?;
    let abandoned = directory::decode_offer(other_control.receive(end())?).map_err(pe)?;
    ok(owned::receive(abandoned.endpoint.as_handle(), end()).err() == Some(Error::PeerClosed))?;
    drop(abandoned);
    let (new_control, new_server) = Channel::pair()?;
    server
        .add_provider(
            "echo",
            PROTOCOL_ECHO,
            directory::ENDPOINT_RIGHTS,
            new_server.into_untyped(),
        )
        .map_err(pe)?;
    server.step();
    ok(matches!(
        new_control.receive(Deadline::IMMEDIATE),
        Err(Error::ShouldWait | Error::TimedOut)
    ))?;
    // Closing a waiting client releases its fresh pending endpoint. The already
    // delivered offer is held only until provider ACK, then forgotten.
    b.try_send(
        directory::message(
            PROTOCOL_DIRECTORY,
            MAJOR_DIRECTORY,
            KIND_REQUEST,
            DIRECTORY_CONNECT,
            2,
            &connection_body("echo"),
            0,
        )
        .map_err(pe)?,
    )
    .map_err(|e| e.error)?;
    server.step();
    let offer = directory::decode_offer(new_control.receive(end())?).map_err(pe)?;
    drop(b);
    server.step();
    ok(owned::receive(offer.endpoint.as_handle(), end()).err() == Some(Error::PeerClosed))?;
    new_control
        .try_send(directory::offer_reply(offer.id, STATUS_OK).map_err(pe)?)
        .map_err(|e| e.error)?;
    server.step();
    drop(offer);
    drop(a);
    drop(other_control);
    drop(new_control);
    server.run()?;
    // A timeout after successful send is explicitly uncertain, and the local
    // client retires so a delayed reply cannot match a new request.
    let (client, peer) = Channel::pair()?;
    let mut client = Client::new(client.into_untyped());
    let r = client.call(
        DIRECTORY_LIST,
        &[],
        None,
        Deadline::from_now(Duration::from_millis(10)),
    );
    ok(matches!(
        r,
        Err(ClientError::OutcomeUnknown(Error::TimedOut))
    ))?;
    ok(client.is_retired())?;
    let _ = peer.receive(end())?;
    ok(peer.receive(end()).err() == Some(Error::PeerClosed))?;
    Ok(())
}

pub fn run(t: &mut Tally, b: &Bootstrap) {
    let r = regression_checks();
    if let Err(e) = r {
        dross_rt::println!("service regression: {e:?}");
    }
    t.check(
        "service saturated retirement closed-client cleanup and uncertain timeout",
        r.is_ok(),
        "",
    );
    let r = bootstrap_checks();
    t.check(
        "owned bootstrap v2 takes once and closes rejected attachments",
        r.is_ok(),
        "",
    );
    let r = directory_checks(t, b);
    if let Err(e) = r {
        dross_rt::println!("service tests: {e:?}");
    }
    t.check("service directory process acceptance", r.is_ok(), "");
}
