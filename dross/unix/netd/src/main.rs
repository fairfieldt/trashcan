#![no_std]
#![no_main]
extern crate alloc;
mod ethernet;
use alloc::{collections::BTreeMap, vec, vec::Vec};
use dross_rt::{Error, Handle, bootstrap::Bootstrap, sys};
use smoltcp::{
    iface::{Config, Interface, SocketHandle, SocketSet},
    socket::tcp,
    time::Instant,
    wire::{EthernetAddress, IpAddress, IpCidr, IpEndpoint, IpListenEndpoint, Ipv4Address},
};
dross_rt::main!(main);
const MAX_SOCKETS: usize = 32;
const BUFFER: usize = 32768;
struct Ports {
    owners: BTreeMap<u16, Handle>,
    next: u16,
}
impl Ports {
    fn reserve(&mut self, requested: u16, owner: Handle) -> Result<u16, i64> {
        if requested != 0 {
            if self.owners.contains_key(&requested) {
                return Err(98);
            }
            self.owners.insert(requested, owner);
            return Ok(requested);
        }
        for _ in 49152..=65535 {
            let port = self.next;
            self.next = self.next.wrapping_add(1).max(49152);
            if let alloc::collections::btree_map::Entry::Vacant(slot) = self.owners.entry(port) {
                slot.insert(owner);
                return Ok(port);
            }
        }
        Err(99)
    }
    fn release(&mut self, owner: Handle) {
        self.owners.retain(|_, value| *value != owner);
    }
}
#[derive(Clone, Copy)]
enum Mode {
    New,
    Stream(SocketHandle),
    Listen,
}
struct Connection {
    ch: Handle,
    mode: Mode,
    bound: IpListenEndpoint,
    listening: Vec<SocketHandle>,
    backlog: usize,
    connecting: bool,
    error: u32,
    dead: bool,
}
fn socket(set: &mut SocketSet<'static>) -> SocketHandle {
    let mut socket = tcp::Socket::new(
        tcp::SocketBuffer::new(vec![0; BUFFER]),
        tcp::SocketBuffer::new(vec![0; BUFFER]),
    );
    socket.set_timeout(Some(smoltcp::time::Duration::from_secs(60)));
    set.add(socket)
}
fn endpoint(bytes: &[u8]) -> Result<IpEndpoint, i64> {
    if bytes.len() != 16 || bytes[..2] != [2, 0] {
        return Err(22);
    }
    Ok(IpEndpoint::new(
        IpAddress::v4(bytes[4], bytes[5], bytes[6], bytes[7]),
        u16::from_be_bytes([bytes[2], bytes[3]]),
    ))
}
fn sockaddr(endpoint: IpEndpoint) -> [u8; 16] {
    let mut bytes = [0; 16];
    bytes[0] = 2;
    bytes[2..4].copy_from_slice(&endpoint.port.to_be_bytes());
    let IpAddress::Ipv4(ip) = endpoint.addr;
    bytes[4..8].copy_from_slice(&ip.octets());
    bytes
}
fn reply(
    ch: Handle,
    status: Result<u64, i64>,
    bytes: &[u8],
    handles: &[Handle],
) -> Result<(), Error> {
    let mut out = match status {
        Ok(v) => (v as i64).to_le_bytes().to_vec(),
        Err(e) => (-e).to_le_bytes().to_vec(),
    };
    out.extend_from_slice(bytes);
    sys::channel_write(ch, &out, handles)
}
fn handle(
    c: &mut Connection,
    bytes: &[u8],
    set: &mut SocketSet<'static>,
    iface: &mut Interface,
    ports: &mut Ports,
    notify: Handle,
) -> Result<Option<Connection>, i64> {
    if bytes.len() < 4 {
        return Err(22);
    }
    let op = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    let arg = &bytes[4..];
    match op {
        0 => {
            if !arg.is_empty() {
                return Err(22);
            }
            reply(c.ch, Ok(0), &[], &[]).map_err(|_| 5)?;
        }
        1 => {
            let ep = endpoint(arg)?;
            if !matches!(c.mode, Mode::New) || c.bound.port != 0 {
                return Err(22);
            }
            if ep.addr != IpAddress::v4(0, 0, 0, 0) && !iface.has_ip_addr(ep.addr) {
                return Err(99);
            }
            let port = ports.reserve(ep.port, c.ch)?;
            c.bound = IpListenEndpoint {
                addr: if ep.addr == IpAddress::v4(0, 0, 0, 0) {
                    None
                } else {
                    Some(ep.addr)
                },
                port,
            };
            reply(c.ch, Ok(0), &[], &[]).map_err(|_| 5)?;
        }
        2 => {
            if arg.len() != 4 || !matches!(c.mode, Mode::New) {
                return Err(22);
            }
            if c.bound.port == 0 {
                c.bound.port = ports.reserve(0, c.ch)?;
            }
            c.backlog = (u32::from_le_bytes(arg.try_into().unwrap()) as usize).clamp(1, 4);
            c.mode = Mode::Listen;
            for _ in 0..c.backlog {
                let h = socket(set);
                set.get_mut::<tcp::Socket>(h)
                    .listen(c.bound)
                    .map_err(|_| 98)?;
                c.listening.push(h);
            }
            reply(c.ch, Ok(0), &[], &[]).map_err(|_| 5)?;
        }
        3 => {
            let remote = endpoint(arg)?;
            if matches!(c.mode, Mode::New) {
                let h = socket(set);
                if c.bound.port == 0 {
                    c.bound.port = ports.reserve(0, c.ch)?;
                }
                if set
                    .get_mut::<tcp::Socket>(h)
                    .connect(iface.context(), remote, c.bound)
                    .is_err()
                {
                    set.remove(h);
                    return Err(22);
                }
                c.mode = Mode::Stream(h);
                c.connecting = true;
            }
            let Mode::Stream(h) = c.mode else {
                return Err(22);
            };
            let sock = set.get::<tcp::Socket>(h);
            if sock.state() == tcp::State::Established {
                c.connecting = false;
                reply(c.ch, Ok(0), &[], &[]).map_err(|_| 5)?;
            } else if sock.state() == tcp::State::Closed {
                c.error = 111;
                return Err(111);
            } else {
                return Err(11);
            }
        }
        4 => {
            if !matches!(c.mode, Mode::Listen) {
                return Err(22);
            }
            let Some(index) = c
                .listening
                .iter()
                .position(|h| set.get::<tcp::Socket>(*h).state() == tcp::State::Established)
            else {
                return Err(11);
            };
            let h = c.listening.remove(index);
            let peer = set.get::<tcp::Socket>(h).remote_endpoint().ok_or(107)?;
            let (mine, theirs) = sys::channel_create().map_err(|_| 12)?;
            sys::channel_bind(mine, notify, 2).map_err(|_| 5)?;
            reply(c.ch, Ok(0), &sockaddr(peer), &[theirs]).map_err(|_| 5)?;
            let replacement = socket(set);
            set.get_mut::<tcp::Socket>(replacement)
                .listen(c.bound)
                .map_err(|_| 98)?;
            c.listening.push(replacement);
            return Ok(Some(Connection {
                ch: mine,
                mode: Mode::Stream(h),
                bound: IpListenEndpoint {
                    addr: None,
                    port: 0,
                },
                listening: Vec::new(),
                backlog: 0,
                connecting: false,
                error: 0,
                dead: false,
            }));
        }
        5 => {
            if arg.len() != 8 {
                return Err(22);
            }
            let len = u32::from_le_bytes(arg[..4].try_into().unwrap()).min(60 * 1024) as usize;
            let peek = u32::from_le_bytes(arg[4..].try_into().unwrap()) != 0;
            let Mode::Stream(h) = c.mode else {
                return Err(107);
            };
            let sock = set.get_mut::<tcp::Socket>(h);
            let mut out = vec![0; len];
            let n = if sock.can_recv() {
                if peek {
                    sock.peek_slice(&mut out).map_err(|_| 5)?
                } else {
                    sock.recv_slice(&mut out).map_err(|_| 5)?
                }
            } else if !sock.may_recv() {
                0
            } else {
                return Err(11);
            };
            reply(c.ch, Ok(n as u64), &out[..n], &[]).map_err(|_| 5)?;
        }
        6 => {
            let Mode::Stream(h) = c.mode else {
                return Err(107);
            };
            let sock = set.get_mut::<tcp::Socket>(h);
            if !sock.can_send() {
                return Err(if sock.may_send() { 11 } else { 32 });
            }
            let n = sock.send_slice(arg).map_err(|_| 32)?;
            reply(c.ch, Ok(n as u64), &[], &[]).map_err(|_| 5)?;
        }
        7 => {
            let flags = match c.mode {
                Mode::New => 4,
                Mode::Listen => {
                    if c.listening
                        .iter()
                        .any(|h| set.get::<tcp::Socket>(*h).state() == tcp::State::Established)
                    {
                        1
                    } else {
                        0
                    }
                }
                Mode::Stream(h) => {
                    let s = set.get::<tcp::Socket>(h);
                    if c.connecting && s.state() == tcp::State::Closed {
                        c.error = 111;
                    }
                    u64::from(s.can_recv() || !s.may_recv() && !c.connecting)
                        | if s.can_send() { 4 } else { 0 }
                        | if !s.is_open() { 16 } else { 0 }
                        | if c.error != 0 { 8 | 4 } else { 0 }
                }
            };
            reply(c.ch, Ok(flags), &[], &[]).map_err(|_| 5)?;
        }
        8 => {
            if arg.len() != 4 {
                return Err(22);
            }
            let Mode::Stream(h) = c.mode else {
                return Err(107);
            };
            set.get_mut::<tcp::Socket>(h).close();
            reply(c.ch, Ok(0), &[], &[]).map_err(|_| 5)?;
        }
        9 | 10 => {
            let ep = match c.mode {
                Mode::Stream(h) => {
                    let s = set.get::<tcp::Socket>(h);
                    if op == 9 {
                        s.local_endpoint()
                    } else {
                        s.remote_endpoint()
                    }
                }
                _ if op == 9 => Some(IpEndpoint::new(
                    c.bound.addr.unwrap_or(IpAddress::v4(0, 0, 0, 0)),
                    c.bound.port,
                )),
                _ => None,
            }
            .ok_or(107)?;
            reply(c.ch, Ok(0), &sockaddr(ep), &[]).map_err(|_| 5)?;
        }
        11 => {
            let error = core::mem::take(&mut c.error);
            reply(c.ch, Ok(error as u64), &[], &[]).map_err(|_| 5)?;
        }
        _ => return Err(38),
    }
    Ok(None)
}
fn main(bootstrap: Handle) -> i32 {
    let b = Bootstrap::read(bootstrap).expect("netd bootstrap");
    let root = b.handle("service").expect("network service capability");
    let ready = b.handle("ready").expect("ready report");
    let channel = b.handle("ethernet").expect("Ethernet capability");
    let mut nic = ethernet::Ethernet::new(channel);
    let mut config = Config::new(EthernetAddress(nic.info.mac).into());
    let mut seed = [0; 8];
    sys::random(&mut seed).expect("network entropy");
    config.random_seed = u64::from_le_bytes(seed);
    let mut iface = Interface::new(
        config,
        &mut nic,
        Instant::from_micros((sys::clock() / 1000) as i64),
    );
    let ip: Ipv4Address = b
        .value("ip")
        .unwrap_or("10.0.2.15")
        .parse()
        .expect("IPv4 address");
    let prefix = b.value_u64("prefix").unwrap_or(24);
    assert!(prefix <= 32);
    iface.update_ip_addrs(|ips| {
        ips.push(IpCidr::new(ip.into(), prefix as u8)).unwrap();
    });
    if let Some(gateway) = b.value("gateway").or_else(|| {
        if b.value("ip").is_none() {
            Some("10.0.2.2")
        } else {
            None
        }
    }) {
        iface
            .routes_mut()
            .add_default_ipv4_route(gateway.parse().expect("IPv4 gateway"))
            .unwrap();
    }
    let mut retired: Vec<(SocketHandle, u64)> = Vec::new();
    let mut set = SocketSet::new(Vec::new());
    let mut connections: Vec<Connection> = Vec::new();
    let mut ports = Ports {
        owners: BTreeMap::new(),
        next: 49152 + (u64::from_le_bytes(seed) % 16384) as u16,
    };
    let notify = sys::notify_create().unwrap();
    sys::channel_bind(root, notify, 1).unwrap();
    sys::channel_bind(channel, notify, 4).unwrap();
    sys::channel_write(ready, b"ready", &[]).unwrap();
    let _ = sys::close(ready);
    let mut bytes = vec![0; 65536];
    let mut hs = [Handle(0); 64];
    loop {
        if nic.service().is_err() {
            return 1;
        }
        iface.poll(
            Instant::from_micros((sys::clock() / 1000) as i64),
            &mut nic,
            &mut set,
        );
        for _ in 0..8 {
            match sys::channel_read(root, &mut bytes, &mut hs, 0) {
                Ok((8, 1)) if bytes[..8] == *b"ONET\x01\0\0\0" => {
                    if connections.len() >= MAX_SOCKETS {
                        let _ = sys::close(hs[0]);
                        continue;
                    }
                    sys::channel_bind(hs[0], notify, 2).unwrap();
                    connections.push(Connection {
                        ch: hs[0],
                        mode: Mode::New,
                        bound: IpListenEndpoint {
                            addr: None,
                            port: 0,
                        },
                        listening: Vec::new(),
                        backlog: 0,
                        connecting: false,
                        error: 0,
                        dead: false,
                    });
                }
                Err(Error::TimedOut | Error::ShouldWait) => break,
                Err(Error::PeerClosed) => return 0,
                Ok((_, n)) => {
                    for h in &hs[..n] {
                        let _ = sys::close(*h);
                    }
                }
                Err(_) => return 1,
            }
        }
        let mut accepted = Vec::new();
        let capacity = MAX_SOCKETS.saturating_sub(connections.len());
        for c in &mut connections {
            for _ in 0..8 {
                match sys::channel_read(c.ch, &mut bytes, &mut hs, 0) {
                    Ok((n, 0)) => {
                        if bytes.get(..4) != Some(&4u32.to_le_bytes()) || accepted.len() < capacity
                        {
                            match handle(c, &bytes[..n], &mut set, &mut iface, &mut ports, notify) {
                                Ok(Some(new)) => accepted.push(new),
                                Ok(None) => {}
                                Err(e) => {
                                    if reply(c.ch, Err(e), &[], &[]).is_err() {
                                        c.dead = true;
                                        break;
                                    }
                                }
                            }
                        } else {
                            let _ = reply(c.ch, Err(24), &[], &[]);
                        }
                    }
                    Err(Error::TimedOut | Error::ShouldWait) => break,
                    Err(_) => {
                        c.dead = true;
                        break;
                    }
                    Ok((_, n)) => {
                        for h in &hs[..n] {
                            let _ = sys::close(*h);
                        }
                        c.dead = true;
                        break;
                    }
                }
            }
        }
        connections.extend(accepted);
        connections.retain(|c| {
            if !c.dead {
                return true;
            }
            if let Mode::Stream(h) = c.mode {
                set.get_mut::<tcp::Socket>(h).close();
                retired.push((h, sys::after_ms(5000)));
            }
            for &h in &c.listening {
                set.get_mut::<tcp::Socket>(h).abort();
                set.remove(h);
            }
            ports.release(c.ch);
            let _ = sys::close(c.ch);
            false
        });
        retired.retain(|&(h, deadline)| {
            if set.get::<tcp::Socket>(h).state() == tcp::State::Closed || sys::clock() >= deadline {
                set.get_mut::<tcp::Socket>(h).abort();
                set.remove(h);
                false
            } else {
                true
            }
        });
        while retired.len() > 32 {
            let (h, _) = retired.remove(0);
            set.get_mut::<tcp::Socket>(h).abort();
            set.remove(h);
        }
        iface.poll(
            Instant::from_micros((sys::clock() / 1000) as i64),
            &mut nic,
            &mut set,
        );
        if nic.service().is_err() {
            return 1;
        }
        let _ = sys::notify_wait(notify, sys::after_ms(1));
    }
}
