#![no_std]
#![recursion_limit = "256"]
#![no_main]
extern crate alloc;
use alloc::vec::Vec;
use dross_rt::Handle;
use oscar_netstack3_engine::{
    BUFFER_BYTES, Config, Endpoint, Engine, Error, MAX_SOCKETS, Shutdown, SocketId,
};
dross_rt::main!(main);
const SECOND: u64 = 1_000_000_000;
fn config(index: u8) -> Config {
    Config {
        mac: [2, 0, 0, 0, 0, index],
        address: [192, 0, 2, index],
        prefix: 24,
        gateway: None,
        mtu: 1500,
    }
}
struct Lab {
    engines: [Engine; 2],
    now: u64,
    steps: usize,
    arps: usize,
    syn_drops: usize,
    data_drops: usize,
    drop_syn: bool,
    drop_data: bool,
}
impl Lab {
    fn new(drop_syn: bool, drop_data: bool) -> Self {
        Self {
            engines: [
                Engine::new(config(1), 0).unwrap(),
                Engine::new(config(2), 0).unwrap(),
            ],
            now: 0,
            steps: 0,
            arps: 0,
            syn_drops: 0,
            data_drops: 0,
            drop_syn,
            drop_data,
        }
    }
    fn step(&mut self) -> bool {
        self.steps += 1;
        assert!(self.steps < 10000, "engine test exceeded event budget");
        let mut more = false;
        for e in &mut self.engines {
            let p = e.pump(self.now);
            assert!(!p.failed);
            more |= p.more_work;
        }
        for sender in 0..2 {
            if let Some(frame) = self.engines[sender].take_ethernet() {
                if frame[12..14] == [8, 6] {
                    self.arps += 1;
                }
                if sender == 0 && frame[12..14] == [8, 0] && frame.len() >= 54 && frame[23] == 6 {
                    let ip = usize::from(frame[14] & 15) * 4;
                    let tcp = 14 + ip;
                    let syn = frame[tcp + 13] & 2 != 0;
                    let total = usize::from(u16::from_be_bytes([frame[16], frame[17]]));
                    let payload = total > ip + usize::from(frame[tcp + 12] >> 4) * 4;
                    if self.drop_syn && syn {
                        self.drop_syn = false;
                        self.syn_drops += 1;
                        return true;
                    }
                    if self.drop_data && payload {
                        self.drop_data = false;
                        self.data_drops += 1;
                        return true;
                    }
                }
                self.engines[1 - sender].receive_ethernet(&frame).unwrap();
                return true;
            }
        }
        if more {
            return true;
        }
        let next = self.engines.iter().filter_map(Engine::next_deadline).min();
        if let Some(next) = next {
            assert!(next >= self.now);
            assert!(next <= self.now.saturating_add(300 * SECOND));
            self.now = next;
            true
        } else {
            false
        }
    }
    fn until(&mut self, mut done: impl FnMut(&mut Self) -> bool) {
        for _ in 0..10000 {
            if done(self) {
                return;
            }
            assert!(self.step(), "engine stalled without a deadline");
        }
        panic!("engine condition failed")
    }
    fn receive(&mut self, side: usize, socket: SocketId, expected: &[u8]) {
        let mut offset = 0;
        let mut scratch = [0; 4096];
        self.until(|lab| {
            match lab.engines[side].read(socket, &mut scratch, false) {
                Ok(n) => {
                    assert!(n > 0);
                    assert!(offset + n <= expected.len());
                    assert_eq!(&scratch[..n], &expected[offset..offset + n]);
                    offset += n
                }
                Err(Error::WouldBlock) => {}
                e => panic!("read failed: {:?}", e),
            }
            offset == expected.len()
        });
    }
}
fn wire(drop_syn: bool, drop_data: bool) {
    let mut lab = Lab::new(drop_syn, drop_data);
    let listener = lab.engines[1].create().unwrap();
    lab.engines[1]
        .bind(
            listener,
            Endpoint {
                address: [192, 0, 2, 2],
                port: 22222,
            },
        )
        .unwrap();
    lab.engines[1].listen(listener, 4).unwrap();
    assert_eq!(lab.engines[1].accept(listener), Err(Error::WouldBlock));
    let client = lab.engines[0].create().unwrap();
    let remote = Endpoint {
        address: [192, 0, 2, 2],
        port: 22222,
    };
    assert_eq!(
        lab.engines[0].connect(client, remote),
        Err(Error::WouldBlock)
    );
    let mut accepted = None;
    lab.until(|lab| {
        if lab.engines[1].readiness(listener).unwrap().accept {
            accepted = Some(lab.engines[1].accept(listener).unwrap());
            true
        } else {
            false
        }
    });
    let (server, peer) = accepted.unwrap();
    assert_eq!(peer.address, [192, 0, 2, 1]);
    lab.until(|lab| lab.engines[0].connect(client, remote) == Ok(()));
    assert!(lab.arps >= 2, "dynamic ARP request/reply absent");
    let request: Vec<u8> = (0..if drop_data { 512 } else { 9217 })
        .map(|n| (n * 37 + 3) as u8)
        .collect();
    let reply: Vec<u8> = (0..4099).map(|n| (n * 19 + 7) as u8).collect();
    assert_eq!(lab.engines[0].write(client, &request), Ok(request.len()));
    assert_eq!(lab.engines[1].write(server, &reply), Ok(reply.len()));
    lab.until(|lab| lab.engines[1].readiness(server).unwrap().readable);
    let mut peek = [0; 17];
    assert_eq!(lab.engines[1].read(server, &mut peek, true), Ok(17));
    assert_eq!(&peek, &request[..17]);
    assert_eq!(lab.engines[1].read(server, &mut peek, true), Ok(17));
    assert_eq!(&peek, &request[..17]);
    lab.receive(1, server, &request);
    lab.receive(0, client, &reply);
    // Wait for ACKs before the next admission test, then force a partial write.
    lab.until(|lab| {
        lab.engines[0].readiness(client).unwrap().writable
            && lab.engines[1].readiness(server).unwrap().writable
    });
    let large: Vec<u8> = (0..BUFFER_BYTES + 1000)
        .map(|n| (n * 13 + 11) as u8)
        .collect();
    let written = lab.engines[0].write(client, &large).unwrap();
    assert!(written > 0 && written <= BUFFER_BYTES);
    assert_eq!(lab.engines[0].write(client, b"x"), Err(Error::WouldBlock));
    lab.receive(1, server, &large[..written]);
    let mut offset = written;
    while offset < large.len() {
        lab.until(|lab| lab.engines[0].readiness(client).unwrap().writable);
        let n = lab.engines[0].write(client, &large[offset..]).unwrap();
        assert!(n > 0);
        lab.receive(1, server, &large[offset..offset + n]);
        offset += n;
    }
    // A second large response crosses the 32 KiB ring boundary as well.
    let mut offset = 0;
    while offset < large.len() {
        lab.until(|lab| lab.engines[1].readiness(server).unwrap().writable);
        let n = lab.engines[1].write(server, &large[offset..]).unwrap();
        assert!(n > 0);
        lab.receive(0, client, &large[offset..offset + n]);
        offset += n;
    }
    lab.engines[0].shutdown(client, Shutdown::Write).unwrap();
    assert_eq!(lab.engines[0].write(client, b"x"), Err(Error::BrokenPipe));
    lab.until(|lab| matches!(lab.engines[1].read(server, &mut peek, false), Ok(0)));
    lab.engines[1].write(server, b"after-fin").unwrap();
    lab.receive(0, client, b"after-fin");
    lab.engines[1].shutdown(server, Shutdown::Write).unwrap();
    lab.until(|lab| matches!(lab.engines[0].read(client, &mut peek, false), Ok(0)));
    lab.engines[0].close(client).unwrap();
    lab.engines[1].close(server).unwrap();
    lab.engines[1].close(listener).unwrap();
    lab.now = lab.now.saturating_add(6 * SECOND);
    for e in &mut lab.engines {
        assert!(!e.pump(lab.now).failed);
    }
    assert_eq!(lab.syn_drops, usize::from(drop_syn));
    assert_eq!(lab.data_drops, usize::from(drop_data));
    dross_rt::println!(
        "engine-test: wire syn-loss={} data-loss={} ARP={} PASS",
        drop_syn,
        drop_data,
        lab.arps
    );
}
fn local(address: [u8; 4]) {
    let mut e = Engine::new(config(1), 0).unwrap();
    let listener = e.create().unwrap();
    e.bind(
        listener,
        Endpoint {
            address,
            port: 22333,
        },
    )
    .unwrap();
    e.listen(listener, 1).unwrap();
    let client = e.create().unwrap();
    let remote = Endpoint {
        address,
        port: 22333,
    };
    assert_eq!(e.connect(client, remote), Err(Error::WouldBlock));
    let mut accepted = None;
    for _ in 0..64 {
        assert!(!e.pump(0).failed);
        if e.readiness(listener).unwrap().accept {
            accepted = Some(e.accept(listener).unwrap());
            break;
        }
    }
    let (server, peer) = accepted.expect("loopback accept");
    assert_eq!(
        peer.address,
        if address[0] == 127 {
            [127, 0, 0, 1]
        } else {
            address
        }
    );
    assert_eq!(e.connect(client, remote), Ok(()));
    assert_eq!(e.write(client, b"local bytes"), Ok(11));
    let mut bytes = [0; 32];
    let mut count = 0;
    for _ in 0..64 {
        e.pump(0);
        match e.read(server, &mut bytes, false) {
            Ok(n) => {
                count = n;
                break;
            }
            Err(Error::WouldBlock) => {}
            x => panic!("local read {:?}", x),
        }
    }
    assert_eq!(&bytes[..count], b"local bytes");
    assert!(
        e.take_ethernet().is_none(),
        "local packet leaked to physical NIC"
    );
    e.abort(client).unwrap();
    e.abort(server).unwrap();
    e.close(listener).unwrap();
    dross_rt::println!("engine-test: loopback {:?} PASS", address);
}
fn admission() {
    let mut e = Engine::new(config(1), 0).unwrap();
    let mut ids = Vec::new();
    for _ in 0..MAX_SOCKETS {
        ids.push(e.create().unwrap());
    }
    assert_eq!(e.create(), Err(Error::NoResources));
    for id in ids {
        e.close(id).unwrap();
        assert_eq!(e.readiness(id), Err(Error::BadSocket));
    }
    let listener = e.create().unwrap();
    e.bind(
        listener,
        Endpoint {
            address: [192, 0, 2, 1],
            port: 22444,
        },
    )
    .unwrap();
    e.listen(listener, 4).unwrap();
    let mut ids = Vec::new();
    for _ in 0..MAX_SOCKETS - 5 {
        ids.push(e.create().unwrap());
    }
    assert_eq!(e.create(), Err(Error::NoResources));
    for id in ids {
        e.close(id).unwrap();
    }
    e.close(listener).unwrap();
    let mut cfg = config(1);
    cfg.gateway = Some([192, 0, 2, 254]);
    let mut gateway = Engine::new(cfg, 0).unwrap();
    let s = gateway.create().unwrap();
    assert_eq!(
        gateway.connect(
            s,
            Endpoint {
                address: [198, 51, 100, 9],
                port: 22
            }
        ),
        Err(Error::WouldBlock)
    );
    let frame = gateway.take_ethernet().expect("gateway ARP");
    assert_eq!(&frame[12..14], &[8, 6]);
    assert_eq!(&frame[38..42], &[192, 0, 2, 254]);
    gateway.abort(s).unwrap();
    dross_rt::println!("engine-test: admission and gateway ARP PASS");
}
fn pressure_and_link_loss() {
    let mut e = Engine::new(config(1), 0).unwrap();
    let socket = e.create().unwrap();
    assert_eq!(
        e.connect(
            socket,
            Endpoint {
                address: [192, 0, 2, 2],
                port: 22
            }
        ),
        Err(Error::WouldBlock)
    );
    e.set_link(false);
    assert_eq!(e.peek_error(socket), Ok(Some(Error::NoRoute)));
    let state = e.readiness(socket).unwrap();
    assert!(state.error && state.hangup);
    assert_eq!(e.take_error(socket), Ok(Some(Error::NoRoute)));
    assert_eq!(e.peek_error(socket), Ok(None));
    e.abort(socket).unwrap();
    e.set_link(true);
    let mut arp = [0u8; 42];
    arp[..6].fill(255);
    arp[6..12].copy_from_slice(&[2, 9, 9, 9, 9, 9]);
    arp[12..14].copy_from_slice(&[8, 6]);
    arp[14..22].copy_from_slice(&[0, 1, 8, 0, 6, 4, 0, 1]);
    arp[22..28].copy_from_slice(&[2, 9, 9, 9, 9, 9]);
    arp[38..42].copy_from_slice(&[192, 0, 2, 1]);
    for i in 1..1100u16 {
        arp[28..32].copy_from_slice(&[192, 0, (i / 250) as u8 + 3, (i % 250) as u8 + 1]);
        e.receive_ethernet(&arp).unwrap();
    }
    assert!(!e.pump(0).failed, "network pressure faulted bindings");
    let mut count = 0;
    while e.take_ethernet().is_some() {
        count += 1;
    }
    assert!(count <= 64);
    let mut fragment = [0u8; 42];
    fragment[12..14].copy_from_slice(&[8, 0]);
    fragment[20] = 0x20;
    assert_eq!(e.receive_ethernet(&fragment), Err(Error::Invalid));
    assert_eq!(e.receive_ethernet(&[0; 2050]), Err(Error::Invalid));
    dross_rt::println!("engine-test: link loss, ARP pressure and fragment limits PASS");
}
fn passive_link_loss() {
    let mut lab = Lab::new(false, false);
    let listener = lab.engines[1].create().unwrap();
    lab.engines[1]
        .bind(
            listener,
            Endpoint {
                address: [192, 0, 2, 2],
                port: 22555,
            },
        )
        .unwrap();
    lab.engines[1].listen(listener, 4).unwrap();
    let client = lab.engines[0].create().unwrap();
    assert_eq!(
        lab.engines[0].connect(
            client,
            Endpoint {
                address: [192, 0, 2, 2],
                port: 22555
            }
        ),
        Err(Error::WouldBlock)
    );
    lab.until(|lab| lab.engines[1].readiness(listener).unwrap().accept);
    lab.engines[1].set_link(false);
    assert_eq!(lab.engines[1].accept(listener), Err(Error::NoRoute));
    assert!(!lab.engines[1].readiness(listener).unwrap().accept);
    lab.engines[1].set_link(true);
    lab.engines[0].abort(client).unwrap();
    lab.engines[1].close(listener).unwrap();
    dross_rt::println!("engine-test: queued passive connection link loss PASS");
}
fn checksum(bytes: &[u8]) -> u16 {
    let mut sum = 0u32;
    for chunk in bytes.chunks(2) {
        sum += u32::from(u16::from_be_bytes([chunk[0], *chunk.get(1).unwrap_or(&0)]));
    }
    while sum >> 16 != 0 {
        sum = (sum & 65535) + (sum >> 16);
    }
    !(sum as u16)
}
fn conntrack_pressure() {
    let mut e = Engine::new(config(1), 0).unwrap();
    let listener = e.create().unwrap();
    e.bind(
        listener,
        Endpoint {
            address: [192, 0, 2, 1],
            port: 22666,
        },
    )
    .unwrap();
    e.listen(listener, 4).unwrap();
    let mut arp = [0u8; 42];
    arp[..6].fill(255);
    arp[6..12].copy_from_slice(&[2, 9, 9, 9, 9, 9]);
    arp[12..14].copy_from_slice(&[8, 6]);
    arp[14..22].copy_from_slice(&[0, 1, 8, 0, 6, 4, 0, 1]);
    arp[22..28].copy_from_slice(&[2, 9, 9, 9, 9, 9]);
    arp[28..32].copy_from_slice(&[192, 0, 2, 2]);
    arp[38..42].copy_from_slice(&[192, 0, 2, 1]);
    e.receive_ethernet(&arp).unwrap();
    while e.take_ethernet().is_some() {}
    let mut syn = [0u8; 54];
    syn[..6].copy_from_slice(&[2, 0, 0, 0, 0, 1]);
    syn[6..12].copy_from_slice(&[2, 9, 9, 9, 9, 9]);
    syn[12..14].copy_from_slice(&[8, 0]);
    syn[14] = 0x45;
    syn[16..18].copy_from_slice(&40u16.to_be_bytes());
    syn[20] = 0x40;
    syn[22] = 64;
    syn[23] = 6;
    syn[26..30].copy_from_slice(&[192, 0, 2, 2]);
    syn[30..34].copy_from_slice(&[192, 0, 2, 1]);
    let sum = checksum(&syn[14..34]);
    syn[24..26].copy_from_slice(&sum.to_be_bytes());
    syn[36..38].copy_from_slice(&22666u16.to_be_bytes());
    syn[46] = 0x50;
    syn[47] = 2;
    syn[48..50].copy_from_slice(&16384u16.to_be_bytes());
    for i in 0..1500u16 {
        syn[34..36].copy_from_slice(&(20000 + i).to_be_bytes());
        syn[38..42].copy_from_slice(&u32::from(i).to_be_bytes());
        syn[50..52].fill(0);
        let mut pseudo = [0u8; 32];
        pseudo[..8].copy_from_slice(&syn[26..34]);
        pseudo[9] = 6;
        pseudo[10..12].copy_from_slice(&20u16.to_be_bytes());
        pseudo[12..].copy_from_slice(&syn[34..]);
        let sum = checksum(&pseudo);
        syn[50..52].copy_from_slice(&sum.to_be_bytes());
        e.receive_ethernet(&syn).unwrap();
    }
    assert!(!e.pump(0).failed);
    let stats = e.resources();
    assert_eq!(stats.conntrack_entries, 1024);
    assert!(stats.conntrack_limit_hits > 0);
    assert!(stats.queued_frames <= 64);
    dross_rt::println!("engine-test: conntrack saturation {:?} PASS", stats);
    e.close(listener).unwrap();
    for _ in 0..8 {
        if !e.pump(31 * SECOND).more_work {
            break;
        }
    }
    assert!(!e.pump(31 * SECOND).failed);
    assert!(e.resources().conntrack_entries <= 1024);
    // Abandoning an in-progress connection releases its admitted socket now.
    for _ in 0..MAX_SOCKETS * 2 {
        let id = e.create().unwrap();
        assert_eq!(
            e.connect(
                id,
                Endpoint {
                    address: [192, 0, 2, 200],
                    port: 22
                }
            ),
            Err(Error::WouldBlock)
        );
        e.close(id).unwrap();
        assert_eq!(e.readiness(id), Err(Error::BadSocket));
    }
    let id = e.create().unwrap();
    let mut byte = [0u8; 1];
    assert_eq!(e.shutdown(id, Shutdown::Read), Err(Error::NotConnected));
    assert_eq!(e.shutdown(id, Shutdown::Write), Err(Error::NotConnected));
    assert_eq!(e.read(id, &mut byte, false), Err(Error::NotConnected));
    assert_eq!(e.write(id, b"x"), Err(Error::NotConnected));
    assert_eq!(
        e.connect(
            id,
            Endpoint {
                address: [192, 0, 2, 200],
                port: 22
            }
        ),
        Err(Error::WouldBlock)
    );
    assert_eq!(e.shutdown(id, Shutdown::Read), Err(Error::NotConnected));
    assert_eq!(e.shutdown(id, Shutdown::Write), Err(Error::NotConnected));
    assert_eq!(e.read(id, &mut byte, false), Err(Error::WouldBlock));
    assert_eq!(e.write(id, b"x"), Err(Error::NotConnected));
    e.close(id).unwrap();
    dross_rt::println!("engine-test: pending connect cancellation and negative shutdown PASS");
}
fn main(_bootstrap: Handle) -> i32 {
    dross_rt::println!("engine-test: production engine regressions");
    for loss in [(false, false), (true, false), (false, true)] {
        wire(loss.0, loss.1);
    }
    local([127, 0, 0, 1]);
    local([127, 0, 0, 2]);
    local([192, 0, 2, 1]);
    admission();
    pressure_and_link_loss();
    conntrack_pressure();
    passive_link_loss();
    dross_rt::println!(
        "engine-test: PASS dynamic ARP TCP peek bounded writes half-close local routes admission gateway"
    );
    0
}
