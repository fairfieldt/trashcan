//! Private ONET v1 adapter. Linux errno and sockaddr layouts stay here; the
//! engine and native network protocol do not depend on them.
use crate::service::{Connection, Daemon, MAX_CONNECTIONS, Phase, Role, message_with_space};
use dross_rt::{
    Error,
    owned::{Channel, Message},
    sys,
};
use oscar_netstack3_engine::{Endpoint, Error as NetError, Shutdown};
const MAX_IO: usize = 60 * 1024;
pub fn errno(error: NetError) -> i64 {
    match error {
        NetError::WouldBlock => 11,
        NetError::Invalid | NetError::WrongState => 22,
        NetError::AddressInUse => 98,
        NetError::AddressUnavailable => 99,
        NetError::NoRoute => 101,
        NetError::Refused => 111,
        NetError::Reset => 104,
        NetError::TimedOut => 110,
        NetError::NoResources => 105,
        NetError::NotConnected => 107,
        NetError::BrokenPipe => 32,
        NetError::BadSocket => 9,
    }
}
fn endpoint(b: &[u8]) -> Result<Endpoint, NetError> {
    if b.len() != 16 || b[..2] != [2, 0] {
        return Err(NetError::Invalid);
    }
    Ok(Endpoint {
        address: b[4..8].try_into().unwrap(),
        port: u16::from_be_bytes([b[2], b[3]]),
    })
}
fn sockaddr(e: Endpoint) -> [u8; 16] {
    let mut b = [0; 16];
    b[0] = 2;
    b[2..4].copy_from_slice(&e.port.to_be_bytes());
    b[4..8].copy_from_slice(&e.address);
    b
}
fn output(value: u64, body: &[u8]) -> Result<Message, Error> {
    let mut m = message_with_space(8 + body.len())?;
    m.bytes_mut()[..8].copy_from_slice(&(value as i64).to_le_bytes());
    m.bytes_mut()[8..].copy_from_slice(body);
    Ok(m)
}
pub fn admit(d: &mut Daemon, mut m: Message) -> Result<(), Error> {
    if m.bytes() != b"ONET\x01\0\0\0" || m.handle_count() != 1 {
        return Ok(());
    }
    let endpoint = m.take_handle(0).ok_or(Error::BadHandle)?;
    let wait = match crate::wait::Registration::new(
        d.queue.as_handle().as_raw(),
        endpoint.as_handle(),
        dross_abi::event::READABLE | dross_abi::event::PEER_CLOSED,
    ) {
        Ok(wait) => wait,
        Err(_) => {
            let _ =
                sys::channel_write(endpoint.as_handle().as_raw(), &(-105i64).to_le_bytes(), &[]);
            return Ok(());
        }
    };
    let socket = if d.connections.len() + usize::from(d.processing) >= MAX_CONNECTIONS {
        Err(NetError::NoResources)
    } else {
        d.engine.create()
    };
    let socket = match socket {
        Ok(s) => s,
        Err(e) => {
            let _ = sys::channel_write(
                endpoint.as_handle().as_raw(),
                &(-errno(e)).to_le_bytes(),
                &[],
            );
            return Ok(());
        }
    };
    if d.admit(Connection::new(
        endpoint,
        socket,
        Role::Legacy(Phase::New),
        wait,
    ))
    .is_err()
    {
        let _ = d.engine.close(socket);
    }
    Ok(())
}
pub fn handle(d: &mut Daemon, c: &mut Connection, m: Message) {
    if m.handle_count() != 0 || m.bytes().len() < 4 {
        c.dead = true;
        return;
    }
    let bytes = m.bytes();
    let op = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    let arg = &bytes[4..];
    // Allocate the ordinary scalar/error reply before any state-changing call.
    let mut scalar = match output(0, &[]) {
        Ok(m) => m,
        Err(_) => {
            c.dead = true;
            return;
        }
    };
    let result = (|| -> Result<Option<Message>, NetError> {
        match op {
            0 => {
                if !arg.is_empty() {
                    return Err(NetError::Invalid);
                }
            }
            1 => {
                let ep = endpoint(arg)?;
                if !matches!(c.role, Role::Legacy(Phase::New)) {
                    return Err(NetError::WrongState);
                }
                d.engine.bind(c.socket, ep)?;
            }
            2 => {
                if arg.len() != 4 || !matches!(c.role, Role::Legacy(Phase::New)) {
                    return Err(NetError::WrongState);
                }
                let backlog = u32::from_le_bytes(arg.try_into().unwrap()) as usize;
                d.engine.listen(c.socket, backlog.clamp(1, 4))?;
                c.role = Role::Legacy(Phase::Listen);
            }
            3 => {
                let ep = endpoint(arg)?;
                if let Some(error) = c.error {
                    return Err(error);
                }
                if matches!(c.role, Role::Legacy(Phase::Listen)) {
                    return Err(NetError::WrongState);
                }
                match d.engine.connect(c.socket, ep) {
                    Ok(()) => {
                        c.connecting = false;
                        c.role = Role::Legacy(Phase::Stream);
                    }
                    Err(NetError::WouldBlock) => {
                        c.connecting = true;
                        c.role = Role::Legacy(Phase::Stream);
                        return Err(NetError::WouldBlock);
                    }
                    Err(e) => {
                        if matches!(
                            e,
                            NetError::Refused
                                | NetError::Reset
                                | NetError::TimedOut
                                | NetError::NoRoute
                        ) {
                            c.error = Some(e);
                        }
                        return Err(e);
                    }
                }
            }
            4 => {
                if !arg.is_empty() || !matches!(c.role, Role::Legacy(Phase::Listen)) {
                    return Err(NetError::WrongState);
                }
                if d.connections.len() + usize::from(d.processing) >= MAX_CONNECTIONS {
                    return Err(NetError::NoResources);
                }
                let (server, client) = Channel::pair().map_err(|_| NetError::NoResources)?;
                let wait = crate::wait::Registration::new(
                    d.queue.as_handle().as_raw(),
                    server.as_handle().as_untyped(),
                    dross_abi::event::READABLE | dross_abi::event::PEER_CLOSED,
                )
                .map_err(|_| NetError::NoResources)?;
                let mut response = message_with_space(24).map_err(|_| NetError::NoResources)?;
                response
                    .try_push_handle(client.into_untyped())
                    .map_err(|_| NetError::NoResources)?;
                let (socket, peer) = d.engine.accept(c.socket)?;
                response.bytes_mut()[8..24].copy_from_slice(&sockaddr(peer));
                if d.admit(Connection::new(
                    server.into_untyped(),
                    socket,
                    Role::Legacy(Phase::Stream),
                    wait,
                ))
                .is_err()
                {
                    let _ = d.engine.close(socket);
                    return Err(NetError::NoResources);
                }
                return Ok(Some(response));
            }
            5 => {
                if arg.len() != 8 {
                    return Err(NetError::Invalid);
                }
                if !matches!(c.role, Role::Legacy(Phase::Stream)) {
                    return Err(NetError::NotConnected);
                }
                let max = (u32::from_le_bytes(arg[..4].try_into().unwrap()) as usize).min(MAX_IO);
                let peek = u32::from_le_bytes(arg[4..].try_into().unwrap()) != 0;
                let mut response =
                    message_with_space(8 + max).map_err(|_| NetError::NoResources)?;
                let count = d
                    .engine
                    .read(c.socket, &mut response.bytes_mut()[8..], peek)?;
                response.truncate_bytes(8 + count);
                response.bytes_mut()[..8].copy_from_slice(&(count as i64).to_le_bytes());
                return Ok(Some(response));
            }
            6 => {
                if !matches!(c.role, Role::Legacy(Phase::Stream)) {
                    return Err(NetError::NotConnected);
                }
                let count = d.engine.write(c.socket, &arg[..arg.len().min(MAX_IO)])?;
                scalar.bytes_mut()[..8].copy_from_slice(&(count as i64).to_le_bytes());
            }
            7 => {
                if !arg.is_empty() {
                    return Err(NetError::Invalid);
                }
                c.refresh(&mut d.engine);
                let flags = match c.role {
                    Role::Legacy(Phase::New) => 4,
                    Role::Legacy(Phase::Listen) => {
                        if c.snapshot.signals & 8 != 0 {
                            1
                        } else {
                            0
                        }
                    }
                    _ => {
                        u64::from(c.snapshot.signals & 1 != 0)
                            | if c.snapshot.signals & 2 != 0 { 4 } else { 0 }
                            | if c.snapshot.signals & 16 != 0 { 16 } else { 0 }
                            | if c.error.is_some() { 8 | 4 } else { 0 }
                    }
                };
                scalar.bytes_mut()[..8].copy_from_slice(&(flags as i64).to_le_bytes());
            }
            8 => {
                if arg.len() != 4 || !matches!(c.role, Role::Legacy(Phase::Stream)) {
                    return Err(NetError::NotConnected);
                }
                let direction = match u32::from_le_bytes(arg.try_into().unwrap()) {
                    0 => Shutdown::Read,
                    1 => Shutdown::Write,
                    2 => Shutdown::Both,
                    _ => return Err(NetError::Invalid),
                };
                d.engine.shutdown(c.socket, direction)?;
            }
            9 | 10 => {
                if !arg.is_empty() {
                    return Err(NetError::Invalid);
                }
                let ep = if op == 9 {
                    d.engine.local_addr(c.socket)?
                } else {
                    d.engine.peer_addr(c.socket)?
                };
                return Ok(Some(
                    output(0, &sockaddr(ep)).map_err(|_| NetError::NoResources)?,
                ));
            }
            11 => {
                if !arg.is_empty() {
                    return Err(NetError::Invalid);
                }
                c.refresh(&mut d.engine);
                let value = c
                    .error
                    .take()
                    .or_else(|| d.engine.take_error(c.socket).ok().flatten())
                    .map(errno)
                    .unwrap_or(0);
                let _ = d.engine.take_error(c.socket);
                scalar.bytes_mut()[..8].copy_from_slice(&value.to_le_bytes());
            }
            _ => {
                scalar.bytes_mut()[..8].copy_from_slice(&(-38i64).to_le_bytes());
            }
        }
        Ok(None)
    })();
    match result {
        Ok(Some(reply)) => c.reply = Some(reply),
        Ok(None) => c.reply = Some(scalar),
        Err(error) => {
            scalar.bytes_mut()[..8].copy_from_slice(&(-errno(error)).to_le_bytes());
            c.reply = Some(scalar);
        }
    }
}
