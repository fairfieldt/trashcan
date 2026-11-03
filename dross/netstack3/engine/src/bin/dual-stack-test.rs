#![no_std]
#![no_main]
extern crate alloc;
use alloc::vec::Vec;
use dross_rt::Handle;
use oscar_netstack3_engine::{
    Config, Endpoint, Endpoint6, Engine, Error, Family, IpEndpoint, Ipv6Config, MAX_SOCKETS,
    Shutdown, SocketId,
};
dross_rt::main!(main);
const MS: u64 = 1_000_000;
const SECOND: u64 = 1000 * MS;
const LOOP: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
fn global(n: u8) -> [u8; 16] {
    [0xfd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n]
}
fn v6(a: [u8; 16], port: u16, scope: u32) -> IpEndpoint {
    IpEndpoint::V6(Endpoint6 {
        address: a,
        port,
        scope_id: scope,
    })
}
fn v4(n: u8, port: u16) -> IpEndpoint {
    IpEndpoint::V4(Endpoint {
        address: [192, 0, 2, n],
        port,
    })
}
fn config(n: u8) -> Config {
    Config {
        mac: [2, 0, 0, 0, 0, n],
        address: [192, 0, 2, n],
        prefix: 24,
        gateway: None,
        mtu: 1500,
    }
}
struct Lab {
    e: [Engine; 2],
    now: u64,
    ndp: usize,
    frames: usize,
}
impl Lab {
    fn new() -> Self {
        let mut e = [
            Engine::new(config(1), 0).unwrap(),
            Engine::new(config(2), 0).unwrap(),
        ];
        for (n, e) in e.iter_mut().enumerate() {
            e.configure_ipv6(Ipv6Config {
                address: Some(global(n as u8 + 1)),
                prefix: 64,
                gateway: None,
            })
            .unwrap();
            assert!(!e.ipv6_ready());
        }
        Self {
            e,
            now: 0,
            ndp: 0,
            frames: 0,
        }
    }
    fn step(&mut self) {
        for e in &mut self.e {
            assert!(!e.pump(self.now).failed);
        }
        for sender in 0..2 {
            for _ in 0..8 {
                let Some(f) = self.e[sender].take_ethernet() else {
                    break;
                };
                self.frames += 1;
                if f.len() > 54
                    && f[12..14] == [0x86, 0xdd]
                    && f[20] == 58
                    && matches!(f[54], 135 | 136)
                {
                    self.ndp += 1;
                }
                self.e[1 - sender].receive_ethernet(&f).unwrap();
            }
        }
        self.now += MS;
    }
    fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
    fn ready(&mut self) {
        for _ in 0..5000 {
            if self.e[0].ipv6_ready() && self.e[1].ipv6_ready() {
                return;
            }
            self.step();
        }
        panic!("DAD deadline exceeded");
    }
    fn send(&mut self, n: usize, id: SocketId, body: &[u8], remote: IpEndpoint) -> usize {
        for _ in 0..5000 {
            match self.e[n].send_to(id, body, Some(remote)) {
                Ok(count) => return count,
                Err(Error::WouldBlock) => self.step(),
                other => panic!("datagram send: {:?}", other),
            }
        }
        panic!("UDP send deadline exceeded")
    }
    fn receive(
        &mut self,
        n: usize,
        id: SocketId,
        out: &mut [u8],
        peek: bool,
    ) -> oscar_netstack3_engine::DatagramRead {
        for _ in 0..5000 {
            match self.e[n].recv_from(id, out, peek) {
                Ok(x) => return x,
                Err(Error::WouldBlock) => self.step(),
                other => panic!("datagram receive: {:?}", other),
            }
        }
        panic!("UDP deadline exceeded")
    }
    fn connect(
        &mut self,
        client: usize,
        server: usize,
        remote: IpEndpoint,
        port: u16,
    ) -> (SocketId, SocketId, SocketId) {
        let family = remote.family();
        let l = self.e[server].create_tcp(family).unwrap();
        let wildcard = match family {
            Family::Ipv4 => IpEndpoint::V4(Endpoint {
                address: [0; 4],
                port,
            }),
            Family::Ipv6 => v6([0; 16], port, 0),
        };
        self.e[server].bind_ip(l, wildcard).unwrap();
        self.e[server].listen(l, 2).unwrap();
        let c = self.e[client].create_tcp(family).unwrap();
        assert_eq!(self.e[client].connect_ip(c, remote), Err(Error::WouldBlock));
        let mut accepted = None;
        for _ in 0..5000 {
            self.step();
            if accepted.is_none() {
                match self.e[server].accept_ip(l) {
                    Ok((s, _)) => accepted = Some(s),
                    Err(Error::WouldBlock) => {}
                    x => panic!("accept: {:?}", x),
                }
            }
            if accepted.is_some() && self.e[client].connect_ip(c, remote) == Ok(()) {
                return (c, accepted.unwrap(), l);
            }
        }
        panic!("TCP6 deadline exceeded")
    }
}
fn tcp6() {
    let mut lab = Lab::new();
    lab.ready();
    let dad_ndp = lab.ndp;
    for remote in [
        v6(global(2), 23001, 0),
        v6(lab.e[1].ipv6_link_local().unwrap().address, 23002, 1),
    ] {
        let (c, s, l) = lab.connect(
            0,
            1,
            remote,
            match remote {
                IpEndpoint::V6(e) => e.port,
                _ => unreachable!(),
            },
        );
        assert_eq!(lab.e[0].write(c, b"ipv6 outbound"), Ok(13));
        let mut buf = [0; 64];
        let mut got = false;
        for _ in 0..5000 {
            lab.step();
            match lab.e[1].read(s, &mut buf, false) {
                Ok(13) => {
                    assert_eq!(&buf[..13], b"ipv6 outbound");
                    got = true;
                    break;
                }
                Err(Error::WouldBlock) => {}
                x => panic!("TCP6 read {:?}", x),
            }
        }
        assert!(got);
        assert_eq!(lab.e[1].write(s, b"ipv6 return"), Ok(11));
        let mut got = false;
        for _ in 0..5000 {
            lab.step();
            match lab.e[0].read(c, &mut buf, false) {
                Ok(11) => {
                    assert_eq!(&buf[..11], b"ipv6 return");
                    got = true;
                    break;
                }
                Err(Error::WouldBlock) => {}
                x => panic!("TCP6 reverse {:?}", x),
            }
        }
        assert!(got);
        lab.e[0].shutdown(c, Shutdown::Write).unwrap();
        let mut eof = false;
        for _ in 0..5000 {
            lab.step();
            if lab.e[1].read(s, &mut buf, false) == Ok(0) {
                eof = true;
                break;
            }
        }
        assert!(eof);
        lab.e[1].close(s).unwrap();
        lab.e[0].close(c).unwrap();
        lab.e[1].close(l).unwrap();
        lab.now += 6 * SECOND;
        lab.steps(10);
    }
    assert!(lab.ndp > dad_ndp, "TCP6 did not resolve neighbors");
    dross_rt::println!(
        "dual-stack-test: IPv6 TCP global/link-local NDP bytes half-close PASS frames={} ndp={}",
        lab.frames,
        lab.ndp
    );
}
fn udp(family: Family, linklocal: bool) {
    let mut lab = Lab::new();
    lab.ready();
    let a = lab.e[0].create_udp(family).unwrap();
    let b = lab.e[1].create_udp(family).unwrap();
    let wildcard = match family {
        Family::Ipv4 => IpEndpoint::V4(Endpoint {
            address: [0; 4],
            port: 24000,
        }),
        Family::Ipv6 => v6([0; 16], 24000, 0),
    };
    lab.e[1].udp_bind(b, wildcard).unwrap();
    let remote = match family {
        Family::Ipv4 => v4(2, 24000),
        Family::Ipv6 => v6(
            if linklocal {
                lab.e[1].ipv6_link_local().unwrap().address
            } else {
                global(2)
            },
            24000,
            if linklocal { 1 } else { 0 },
        ),
    };
    assert_eq!(lab.e[0].send_to(a, b"whole-datagram", Some(remote)), Ok(14));
    let mut buf = [0; 4];
    let m = lab.receive(1, b, &mut buf, true);
    assert_eq!(m.copied, 4);
    assert_eq!(m.original_len, 14);
    assert!(m.truncated);
    assert_eq!(&buf, b"whol");
    assert_eq!(m.local, remote);
    assert_eq!(m.interface, 1);
    let mut all = [0; 64];
    let m2 = lab.receive(1, b, &mut all, false);
    assert_eq!(m2.original_len, 14);
    assert!(!m2.truncated);
    assert_eq!(&all[..14], b"whole-datagram");
    assert_eq!(m2.sender, m.sender);
    assert_eq!(
        lab.e[1].recv_from(b, &mut all, false),
        Err(Error::WouldBlock)
    );
    assert_eq!(lab.e[1].send_to(b, b"", Some(m.sender)), Ok(0));
    let empty = lab.receive(0, a, &mut all, false);
    assert_eq!(empty.copied, 0);
    assert_eq!(empty.original_len, 0);
    assert!(!empty.truncated);
    assert_eq!(empty.sender, remote);
    lab.e[0].udp_connect(a, Some(remote)).unwrap();
    assert_eq!(lab.e[0].peer_addr_ip(a), Ok(remote));
    assert_eq!(lab.e[0].send_to(a, b"connected", None), Ok(9));
    assert_eq!(lab.receive(1, b, &mut all, false).copied, 9);
    lab.e[0].udp_connect(a, None).unwrap();
    assert_eq!(lab.e[0].send_to(a, b"x", None), Err(Error::NotConnected));
    let over = [0; 1501];
    assert_eq!(
        lab.e[0].send_to(a, &over, Some(remote)),
        Err(Error::MessageTooLarge)
    );
    for n in 0..40 {
        assert_eq!(lab.e[0].send_to(a, &[n], Some(remote)), Ok(1));
        lab.steps(2);
    }
    assert_eq!(lab.e[1].udp_queue_usage(b), Ok((16, 16, 24)));
    for n in 0..16 {
        let m = lab.receive(1, b, &mut all, false);
        assert_eq!(m.copied, 1);
        assert_eq!(all[0], n);
    }
    assert_eq!(
        lab.e[1].recv_from(b, &mut all, false),
        Err(Error::WouldBlock)
    );
    lab.e[1].shutdown(b, Shutdown::Read).unwrap();
    assert_eq!(lab.e[0].send_to(a, b"discard", Some(remote)), Ok(7));
    lab.steps(10);
    assert_eq!(lab.e[1].udp_queue_usage(b).unwrap().0, 0);
    assert_eq!(
        lab.e[1].recv_from(b, &mut all, false),
        Err(Error::BrokenPipe)
    );
    lab.e[0].abort(a).unwrap();
    lab.e[1].close(b).unwrap();
    dross_rt::println!(
        "dual-stack-test: UDP {:?} linklocal={} packet metadata peek truncation empty connect disconnect RX cap shutdown PASS",
        family,
        linklocal
    );
}
fn loopback() {
    let mut lab = Lab::new();
    lab.ready();
    for address in [LOOP, global(1)] {
        let remote = v6(address, 25000, 0);
        let (c, s, l) = lab.connect(0, 0, remote, 25000);
        assert_eq!(lab.e[0].write(c, b"local6"), Ok(6));
        let mut buf = [0; 16];
        let mut got = false;
        for _ in 0..100 {
            lab.step();
            if lab.e[0].read(s, &mut buf, false) == Ok(6) {
                got = true;
                break;
            }
        }
        assert!(got);
        assert_eq!(&buf[..6], b"local6");
        for id in [c, s, l] {
            lab.e[0].abort(id).unwrap();
        }
        let a = lab.e[0].create_udp(Family::Ipv6).unwrap();
        let b = lab.e[0].create_udp(Family::Ipv6).unwrap();
        lab.e[0].udp_bind(b, remote).unwrap();
        assert_eq!(lab.send(0, a, b"", remote), 0);
        let m = lab.receive(0, b, &mut buf, false);
        assert_eq!(m.copied, 0);
        assert_eq!(m.local, remote);
        // Core transports local traffic over loopback while retaining the
        // device owning the destination. Our global address is assigned on
        // both interfaces; ::1 belongs exclusively to loopback.
        if address == LOOP {
            assert_eq!(m.interface, 2);
        } else {
            assert!(matches!(m.interface, 1 | 2));
        }
        lab.e[0].close(a).unwrap();
        lab.e[0].close(b).unwrap();
    }
    dross_rt::println!("dual-stack-test: IPv6 ::1/self-address TCP UDP loopback PASS");
}
fn policy() {
    let mut e = Engine::new(config(1), 0).unwrap();
    assert_eq!(e.create_tcp(Family::Ipv6), Err(Error::Unsupported));
    e.configure_ipv6(Ipv6Config {
        address: None,
        prefix: 0,
        gateway: None,
    })
    .unwrap();
    let mut ids = Vec::new();
    for i in 0..MAX_SOCKETS {
        ids.push(
            match i % 4 {
                0 => e.create(),
                1 => e.create_tcp(Family::Ipv6),
                2 => e.create_udp(Family::Ipv4),
                _ => e.create_udp(Family::Ipv6),
            }
            .unwrap(),
        );
    }
    assert_eq!(e.create(), Err(Error::NoResources));
    assert_eq!(e.create_udp(Family::Ipv6), Err(Error::NoResources));
    for id in ids {
        e.abort(id).unwrap();
    }
    let id = e.create_tcp(Family::Ipv6).unwrap();
    let ll = e.ipv6_link_local().unwrap();
    assert_eq!(e.connect_ip(id, v6(ll.address, 22, 0)), Err(Error::Invalid));
    assert_eq!(e.connect_ip(id, v6(global(2), 22, 1)), Err(Error::Invalid));
    e.abort(id).unwrap();
    let mut packet = [0u8; 14 + 40 + 16];
    packet[12..14].copy_from_slice(&[0x86, 0xdd]);
    packet[14] = 0x60;
    packet[18..20].copy_from_slice(&16u16.to_be_bytes());
    packet[20] = 44;
    assert_eq!(e.receive_ethernet(&packet), Err(Error::Invalid));
    packet[20] = 60;
    packet[54] = 44;
    packet[55] = 0;
    assert_eq!(e.receive_ethernet(&packet), Err(Error::Invalid));
    packet[54] = 17;
    packet[55] = 255;
    assert_eq!(e.receive_ethernet(&packet), Err(Error::Invalid));
    let mut small = config(1);
    small.mtu = 1200;
    let mut small = Engine::new(small, 0).unwrap();
    assert_eq!(
        small.configure_ipv6(Ipv6Config {
            address: None,
            prefix: 0,
            gateway: None
        }),
        Err(Error::Invalid)
    );
    e.random_u16().unwrap();
    e.random_u64().unwrap();
    dross_rt::println!("dual-stack-test: dual-family resources {:?}", e.resources());
    dross_rt::println!(
        "dual-stack-test: shared TCP/UDP family quota IPv6 configuration scopes min-MTU fragment policy real entropy PASS"
    );
}
fn dad_conflict_and_link() {
    let mut engines = [
        Engine::new(config(1), 0).unwrap(),
        Engine::new(config(2), 0).unwrap(),
    ];
    for e in &mut engines {
        e.configure_ipv6(Ipv6Config {
            address: Some(global(9)),
            prefix: 64,
            gateway: None,
        })
        .unwrap();
    }
    let mut lab = Lab {
        e: engines,
        now: 0,
        ndp: 0,
        frames: 0,
    };
    lab.steps(2500);
    assert!(
        !lab.e[0].ipv6_ready() && !lab.e[1].ipv6_ready(),
        "duplicate address became ready"
    );
    assert!(lab.e[0].ipv6_dad_failed() && lab.e[1].ipv6_dad_failed());
    drop(lab);
    let mut lab = Lab::new();
    lab.ready();
    lab.e[0].set_link(false);
    assert!(!lab.e[0].ipv6_ready());
    lab.now += 2 * SECOND;
    lab.e[0].pump(lab.now);
    assert!(!lab.e[0].ipv6_ready());
    lab.e[0].set_link(true);
    assert!(!lab.e[0].ipv6_ready());
    lab.ready();
    dross_rt::println!(
        "dual-stack-test: duplicate IPv6 conflict stays unready and link-up repeats DAD PASS"
    );
}
fn gateway6() {
    let mut e = Engine::new(config(1), 0).unwrap();
    e.configure_ipv6(Ipv6Config {
        address: Some(global(1)),
        prefix: 64,
        gateway: Some(global(254)),
    })
    .unwrap();
    for _ in 0..16 {
        while e.take_ethernet().is_some() {}
        if !e.pump(2 * SECOND).more_work {
            break;
        }
    }
    assert!(e.ipv6_ready());
    let c = e.create_tcp(Family::Ipv6).unwrap();
    let offlink = [0x20, 1, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9];
    assert_eq!(e.connect_ip(c, v6(offlink, 443, 0)), Err(Error::WouldBlock));
    e.pump(2 * SECOND);
    let mut found = false;
    while let Some(f) = e.take_ethernet() {
        if f.len() >= 78 && f[12..14] == [0x86, 0xdd] && f[20] == 58 && f[54] == 135 {
            assert_eq!(&f[62..78], &global(254));
            found = true;
        }
    }
    assert!(found, "default IPv6 route did not resolve gateway");
    e.abort(c).unwrap();
    dross_rt::println!(
        "dual-stack-test: IPv6 off-link route resolves configured gateway with NDP PASS"
    );
}
fn main(_: Handle) -> i32 {
    dross_rt::println!("dual-stack-test: native dual-stack UDP/TCP simulated Ethernet");
    policy();
    dad_conflict_and_link();
    gateway6();
    tcp6();
    udp(Family::Ipv4, false);
    udp(Family::Ipv6, false);
    udp(Family::Ipv6, true);
    loopback();
    dross_rt::println!(
        "dual-stack-test: PASS UDP IPv4 IPv6 NDP DAD scoped link-local whole-datagram bounds TCP6"
    );
    0
}
