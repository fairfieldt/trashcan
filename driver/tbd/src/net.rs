//! ThunderboltIP: Ethernet frames between two hosts over a DMA path, compatible with Linux's thunderbolt_net
//! (drivers/net/thunderbolt/main.c, 7.3). Here the host side of the network is a TAP device.
//!
//! Both hosts log in to each other over XDomain messages (each login carries the HopID the sender transmits
//! on); once both logins are through, each side starts its rings and routes the two paths. A packet is split into
//! frames of at most 4 KiB, each with a 12-byte header (size, index, packet id, count) and the start/end PDFs 1
//! and 2.

use crate::ctl::{Frame, PDF_XDOMAIN_RESP, xd_uuid};
use crate::domain::{Domain, Peer};
use crate::nhi::{DESC_BUFFER_OVERRUN, DESC_CRC_ERROR, Kind, MAX_FRAME_SIZE, Ring, RingConfig};
use crate::property::Dir;
use crate::uuid::Uuid;
use crate::xdomain::fill_header;
use anyhow::{Context, Result, bail};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Network property directory: c66189ca-1cce-4195-bdb8-49592e5f5a4f.
pub const DIR_UUID: Uuid = Uuid::init(0xc66189ca, 0x1cce, 0x4195, [0xbd, 0xb8, 0x49, 0x59, 0x2e, 0x5f, 0x5a, 0x4f]);
/// ThunderboltIP protocol: 798f589e-3616-8a47-97c6-5664a920c8dd.
pub const SVC_UUID: Uuid = Uuid::init(0x798f589e, 0x3616, 0x8a47, [0x97, 0xc6, 0x56, 0x64, 0xa9, 0x20, 0xc8, 0xdd]);

pub const LOGIN: u32 = 0;
pub const LOGIN_RESPONSE: u32 = 1;
pub const LOGOUT: u32 = 2;
pub const STATUS: u32 = 3;

const PROTO_VERSION: u32 = 1;
const E2E: u32 = 1 << 0;
const MATCH_FRAGS_ID: u32 = 1 << 1;
const FRAMES_64K: u32 = 1 << 2;

const PDF_FRAME_START: u8 = 1;
const PDF_FRAME_END: u8 = 2;
const HEADER: usize = 12;
const MAX_PAYLOAD: usize = MAX_FRAME_SIZE - HEADER;
const MAX_PACKET: usize = 64 * 1024 + 64;
const RING_SIZE: u32 = 256;
/// The receive ring absorbs bursts while the TAP write of a 64 KiB packet runs (1024 frames: 4 MiB).
const RX_RING_SIZE: u32 = 1024;

const LOGIN_DELAY: Duration = Duration::from_millis(4500);
const LOGIN_TIMEOUT: Duration = Duration::from_millis(500);
const LOGOUT_TIMEOUT: Duration = Duration::from_millis(1000);
const LOGIN_RETRIES: u32 = 60;

#[derive(Clone, Debug)]
pub struct NetConfig {
    /// TAP interface name (`%d` takes the next free number).
    pub ifname: String,
    pub mtu: u32,
    /// Full end-to-end flow control on our RX ring, if the other host announces it too.
    pub e2e: bool,
    /// Spin on the rings and the TAP device instead of sleeping.
    pub busy_poll: bool,
}

/// Our `network` service directory (what thunderbolt_net registers).
pub fn property_dir() -> Dir {
    let mut d = Dir::new(Some(DIR_UUID));
    d.add_immediate("prtcid", 1).add_immediate("prtcvers", 1).add_immediate("prtcrevs", 1);
    d.add_immediate("prtcstns", MATCH_FRAGS_ID | FRAMES_64K | E2E);
    d
}

/// `struct thunderbolt_ip_header` (17 dwords) in a packet of `len` dwords.
fn packet(len: usize, route: u64, seq: u32, initiator: Uuid, target: Uuid, ty: u32, command_id: u32) -> Vec<u32> {
    let mut w = vec![0u32; len];
    fill_header(&mut w, route, seq, SVC_UUID);
    w[7..11].copy_from_slice(&initiator.words());
    w[11..15].copy_from_slice(&target.words());
    w[15] = ty;
    w[16] = command_id;
    w
}

fn ip_type(f: &Frame) -> Option<u32> {
    (xd_uuid(f) == Some(SVC_UUID) && f.words.len() >= 17).then(|| f.words[15])
}

/// Logins and logouts from the other host (sent, like all ThunderboltIP packets, as XDomain responses).
pub fn is_request(f: &Frame) -> bool {
    f.pdf == PDF_XDOMAIN_RESP && matches!(ip_type(f), Some(LOGIN | LOGOUT))
}

/// Per-peer connection state shared with the event thread.
#[derive(Default)]
struct Login {
    sent: bool,
    received: bool,
    remote_transmit_path: u32,
    /// The other host logged out: tear down and wait for a new login.
    logout: bool,
}

struct Conn {
    login: Mutex<Login>,
    cv: std::sync::Condvar,
    command_id: AtomicU32,
    mac: [u8; 6],
}

static CONNS: Mutex<Vec<(u32, Arc<Conn>)>> = Mutex::new(Vec::new());

fn conn(port: u32) -> Option<Arc<Conn>> {
    CONNS.lock().unwrap().iter().find(|c| c.0 == port).map(|c| c.1.clone())
}

/// Answers a login or logout (`tbnet_handle_packet`).
pub fn handle_request(d: &Domain, peer: &Peer, f: &Frame) {
    let Some(c) = conn(peer.port) else { return };
    if f.words.len() < 17 {
        return;
    }
    let initiator = Uuid::from_words(&f.words[7..11]);
    let target = Uuid::from_words(&f.words[11..15]);
    if initiator != peer.remote_uuid() || target != d.local_uuid {
        return;
    }
    let route = f.route();
    let seq = crate::xdomain::seq_of(f);
    let command_id = f.words[16];
    match f.words[15] {
        LOGIN if f.words.len() >= 19 => {
            let mut w = packet(25, route, seq, d.local_uuid, peer.remote_uuid(), LOGIN_RESPONSE, command_id);
            let mut mac = [0u8; 8];
            mac[..6].copy_from_slice(&c.mac);
            w[18] = u32::from_le_bytes(mac[..4].try_into().unwrap());
            w[19] = u32::from_le_bytes(mac[4..].try_into().unwrap());
            w[20] = 6;
            if let Err(e) = d.ctl.send(&Frame { pdf: PDF_XDOMAIN_RESP, words: w }) {
                eprintln!("tbd: login response: {e:#}");
                return;
            }
            let mut l = c.login.lock().unwrap();
            l.received = true;
            l.remote_transmit_path = f.words[18];
            c.cv.notify_all();
        }
        LOGOUT => {
            let id = c.command_id.fetch_add(1, Ordering::Relaxed) + 1;
            let w = packet(18, route, seq, d.local_uuid, peer.remote_uuid(), STATUS, id);
            let _ = d.ctl.send(&Frame { pdf: PDF_XDOMAIN_RESP, words: w });
            let mut l = c.login.lock().unwrap();
            l.logout = true;
            c.cv.notify_all();
        }
        _ => {}
    }
}

fn matcher(route: u64, ty: u32, command_id: Option<u32>) -> crate::ctl::Matcher {
    Box::new(move |f: &Frame| {
        f.pdf == PDF_XDOMAIN_RESP && f.route() == route && ip_type(f) == Some(ty) && command_id.is_none_or(|c| f.words[16] == c)
    })
}

/// A locally administered MAC from the host UUID and the port (like `tbnet_generate_mac`, not the same bytes).
fn mac_for(uuid: Uuid, port: u32) -> [u8; 6] {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in uuid.0 {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    let b = h.to_le_bytes();
    [((port - 1) / 2) as u8 * 16 | 0x02, b[0], b[1], b[2], b[3], b[4]]
}

pub fn spawn(d: Arc<Domain>, peer: Arc<Peer>, cfg: NetConfig) {
    std::thread::Builder::new()
        .name(format!("tbd-net{}", peer.port))
        .spawn(move || {
            if let Err(e) = run(&d, &peer, &cfg) {
                eprintln!("port {}: ThunderboltIP stopped: {e:#}", peer.port);
            }
        })
        .expect("spawn net");
}

fn run(d: &Arc<Domain>, peer: &Arc<Peer>, cfg: &NetConfig) -> Result<()> {
    let c = Arc::new(Conn {
        login: Mutex::new(Login::default()),
        cv: std::sync::Condvar::new(),
        command_id: AtomicU32::new(0),
        mac: mac_for(d.local_uuid, peer.port),
    });
    CONNS.lock().unwrap().retain(|x| x.0 != peer.port);
    CONNS.lock().unwrap().push((peer.port, c.clone()));
    let tap = Arc::new(crate::tap::Tap::open(&cfg.ifname, c.mac, cfg.mtu)?);
    println!("port {}: ThunderboltIP on {} (MTU {})", peer.port, tap.name, cfg.mtu);
    let (remote_max, prtcstns) = {
        let s = peer.state.lock().unwrap();
        let r = s.remote.as_ref().unwrap();
        (r.max_hopid, r.service("network", 1).and_then(|n| n.immediate("prtcstns")).unwrap_or(0))
    };
    let local_transmit_path = peer.alloc_hop(false, None, remote_max)?;
    let result = (|| -> Result<()> {
        loop {
            login(d, peer, &c, local_transmit_path)?;
            connected(d, peer, &c, &tap, local_transmit_path, prtcstns, cfg)?;
            if peer.gone() {
                return Ok(());
            }
            println!("port {}: ThunderboltIP logged out, waiting for a new login", peer.port);
        }
    })();
    peer.free_hop(false, local_transmit_path);
    CONNS.lock().unwrap().retain(|x| x.0 != peer.port);
    result
}

/// Sends logins until one is answered and the other host has logged in to us (`tbnet_login_work`).
fn login(d: &Domain, peer: &Peer, c: &Conn, transmit_path: u32) -> Result<()> {
    {
        let mut l = c.login.lock().unwrap();
        l.sent = false;
        l.logout = false;
    }
    std::thread::sleep(Duration::from_millis(1000));
    let mut retries = 0;
    let mut next = Instant::now();
    loop {
        if peer.gone() {
            bail!("unplugged");
        }
        let (sent, received) = {
            let l = c.login.lock().unwrap();
            (l.sent, l.received)
        };
        if sent && received {
            return Ok(());
        }
        if !sent && Instant::now() >= next {
            let id = c.command_id.fetch_add(1, Ordering::Relaxed) + 1;
            let mut w = packet(23, peer.route, retries % 4, d.local_uuid, peer.remote_uuid(), LOGIN, id);
            w[17] = PROTO_VERSION;
            w[18] = transmit_path;
            match d.ctl.request(&Frame { pdf: PDF_XDOMAIN_RESP, words: w }, matcher(peer.route, LOGIN_RESPONSE, Some(id)), LOGIN_TIMEOUT) {
                Ok(_) => {
                    c.login.lock().unwrap().sent = true;
                    continue;
                }
                Err(_) => {
                    retries += 1;
                    if retries >= LOGIN_RETRIES {
                        bail!("ThunderboltIP login timed out");
                    }
                    next = Instant::now() + LOGIN_DELAY;
                }
            }
        }
        let l = c.login.lock().unwrap();
        let _ = c.cv.wait_timeout(l, Duration::from_millis(100)).unwrap();
    }
}

/// Rings up, paths routed, packets moving until a logout or unplug (`tbnet_connected_work`).
fn connected(d: &Arc<Domain>, peer: &Arc<Peer>, c: &Arc<Conn>, tap: &Arc<crate::tap::Tap>, tx_path: u32, prtcstns: u32, cfg: &NetConfig) -> Result<()> {
    let rx_path = c.login.lock().unwrap().remote_transmit_path;
    let local_max = d.router.read().unwrap().port(peer.port).max_in_hop;
    peer.alloc_hop(true, Some(rx_path), local_max).context("RX HopID")?;
    let e2e = cfg.e2e && prtcstns & E2E != 0;
    let mut ring_cfg =
        RingConfig { hop: None, size: RING_SIZE, frame: true, e2e: false, sof_mask: 0, eof_mask: 0, interrupt: !cfg.busy_poll };
    let tx = d.nhi.ring(Kind::Tx, ring_cfg)?;
    ring_cfg.e2e = e2e;
    ring_cfg.size = RX_RING_SIZE;
    ring_cfg.sof_mask = 1 << PDF_FRAME_START;
    ring_cfg.eof_mask = 1 << PDF_FRAME_END;
    let rx = d.nhi.ring(Kind::Rx, ring_cfg)?;
    let (tx_ring, rx_ring) = (tx.hop, rx.hop);
    let result = (|| -> Result<()> {
        d.enable_paths(peer, tx_path, tx_ring, rx_path, rx_ring)?;
        println!(
            "port {}: ThunderboltIP up: TX ring {tx_ring} -> HopID {tx_path}, HopID {rx_path} -> RX ring {rx_ring}{}",
            peer.port,
            if e2e { ", E2E" } else { "" }
        );
        let stop = Arc::new(AtomicBool::new(false));
        let frags_id = prtcstns & MATCH_FRAGS_ID != 0;
        let (t, s, spin) = (tap.clone(), stop.clone(), cfg.busy_poll);
        let txh = std::thread::Builder::new().name(format!("tbd-net{}-tx", peer.port)).spawn(move || tx_loop(tx, &t, &s, frags_id, spin))?;
        let (t, s) = (tap.clone(), stop.clone());
        let rxh = std::thread::Builder::new().name(format!("tbd-net{}-rx", peer.port)).spawn(move || rx_loop(rx, &t, &s))?;
        {
            let mut l = c.login.lock().unwrap();
            while !l.logout && !peer.gone() {
                l = c.cv.wait_timeout(l, Duration::from_millis(200)).unwrap().0;
            }
            l.received = false;
            l.sent = false;
        }
        stop.store(true, Ordering::Relaxed);
        let _ = txh.join();
        let _ = rxh.join();
        Ok(())
    })();
    d.disable_paths(peer, tx_ring, rx_path);
    peer.free_hop(true, rx_path);
    result
}

/// Packets from the TAP device to the TX ring, split into frames (`tbnet_start_xmit`).
fn tx_loop(mut tx: Ring, tap: &crate::tap::Tap, stop: &AtomicBool, frags_id: bool, spin: bool) {
    use crate::tap::{VNET_F_NEEDS_CSUM, VNET_HDR, finish_checksum};
    let mut raw = vec![0u8; VNET_HDR + MAX_PACKET];
    let mut frame_id: u16 = 0;
    let mut bad = 0u64;
    while !stop.load(Ordering::Relaxed) {
        let wait = if spin { Duration::ZERO } else { Duration::from_millis(100) };
        let n = match tap.read(&mut raw, wait) {
            Ok(0) if spin => {
                std::hint::spin_loop();
                continue;
            }
            Ok(n) if n <= VNET_HDR + 14 => continue,
            Ok(n) => n,
            Err(e) => {
                eprintln!("tbd: TAP read: {e}");
                return;
            }
        };
        let (hdr, rest) = raw.split_at_mut(VNET_HDR);
        let buf = &mut rest[..n - VNET_HDR];
        let n = buf.len();
        if hdr[0] & VNET_F_NEEDS_CSUM != 0 {
            let start = u16::from_le_bytes([hdr[6], hdr[7]]) as usize;
            let offset = u16::from_le_bytes([hdr[8], hdr[9]]) as usize;
            if !finish_checksum(buf, start, offset) {
                bad += 1;
                if bad.is_power_of_two() {
                    eprintln!("tbd: ThunderboltIP: {bad} packets whose checksum could not be filled in, dropped");
                }
                continue;
            }
        }
        let count = n.div_ceil(MAX_PAYLOAD);
        // Wait for room for the whole packet.
        loop {
            tx.tx_reap();
            if tx.size - 1 - tx.in_flight() >= count as u32 {
                break;
            }
            if stop.load(Ordering::Relaxed) {
                return;
            }
            tx.wait(Duration::from_millis(10));
        }
        for (i, chunk) in buf[..n].chunks(MAX_PAYLOAD).enumerate() {
            let slot = tx.tx_slot().expect("room checked");
            slot[0..4].copy_from_slice(&(chunk.len() as u32).to_le_bytes());
            slot[4..6].copy_from_slice(&(i as u16).to_le_bytes());
            slot[6..8].copy_from_slice(&frame_id.to_le_bytes());
            slot[8..12].copy_from_slice(&(count as u32).to_le_bytes());
            slot[HEADER..HEADER + chunk.len()].copy_from_slice(chunk);
            tx.tx_submit(HEADER + chunk.len(), PDF_FRAME_START, PDF_FRAME_END);
        }
        if frags_id {
            frame_id = frame_id.wrapping_add(1);
        }
    }
}

/// Frames from the RX ring put back together into packets for the TAP device (`tbnet_poll`, `tbnet_check_frame`).
/// A packet's frames stay in their slots until it is complete and written out from there; a frame that does not
/// fit the packet so far drops the packet and itself, as Linux does.
fn rx_loop(mut rx: Ring, tap: &crate::tap::Tap, stop: &AtomicBool) {
    // Frames of the packet being assembled, still in the ring, and its (count, last index, id).
    let mut held = 0u32;
    let mut cur: Option<(u32, u16, u16)> = None;
    let mut errors = 0u64;
    while !stop.load(Ordering::Relaxed) {
        while let Some(f) = rx.rx_peek_nth(held) {
            let ok = (|| {
                if f.flags & (DESC_CRC_ERROR | DESC_BUFFER_OVERRUN) != 0 || f.len <= HEADER {
                    return None;
                }
                let data = rx.rx_data(&f);
                let size = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
                let index = u16::from_le_bytes(data[4..6].try_into().unwrap());
                let id = u16::from_le_bytes(data[6..8].try_into().unwrap());
                let count = u32::from_le_bytes(data[8..12].try_into().unwrap());
                if size == 0 || size > f.len - HEADER {
                    return None;
                }
                match cur {
                    Some((c, i, pid)) if count != c || index != i + 1 || id != pid => return None,
                    Some(_) => {}
                    None if count == 0 || count > 17 || index != 0 => return None,
                    None => {}
                }
                cur = Some((count, index, id));
                Some(index as u32 == count - 1)
            })();
            match ok {
                Some(false) => held += 1,
                Some(true) => {
                    held += 1;
                    let parts: Vec<&[u8]> = (0..held)
                        .map(|k| {
                            let g = rx.rx_peek_nth(k).unwrap();
                            let d = rx.rx_data(&g);
                            let size = u32::from_le_bytes(d[0..4].try_into().unwrap()) as usize;
                            &d[HEADER..HEADER + size]
                        })
                        .collect();
                    if let Err(e) = tap.write_parts(&parts) {
                        eprintln!("tbd: TAP write: {e}");
                    }
                    drop(parts);
                    for _ in 0..held {
                        rx.rx_release();
                    }
                    held = 0;
                    cur = None;
                }
                None => {
                    for _ in 0..=held {
                        rx.rx_release();
                    }
                    held = 0;
                    cur = None;
                    errors += 1;
                    if errors.is_power_of_two() {
                        eprintln!("tbd: ThunderboltIP: {errors} bad frames so far");
                    }
                }
            }
        }
        rx.wait_nth(held, Duration::from_millis(10));
    }
}

/// Logs out of the other host (`tbnet_logout_request`), used when tbd stops.
pub fn logout(d: &Domain, peer: &Peer) {
    let Some(c) = conn(peer.port) else { return };
    for _ in 0..3 {
        let id = c.command_id.fetch_add(1, Ordering::Relaxed) + 1;
        let w = packet(17, peer.route, 0, d.local_uuid, peer.remote_uuid(), LOGOUT, id);
        if d.ctl.request(&Frame { pdf: PDF_XDOMAIN_RESP, words: w }, matcher(peer.route, STATUS, None), LOGOUT_TIMEOUT).is_ok() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn login_layout() {
        let mut w = packet(23, 1, 0, Uuid([1; 16]), Uuid([2; 16]), LOGIN, 5);
        w[17] = PROTO_VERSION;
        w[18] = 8;
        let f = Frame { pdf: PDF_XDOMAIN_RESP, words: w };
        // length counts everything after route and length_sn: 20 dwords.
        assert_eq!(f.words[2] & 0x3f, 20);
        assert!(is_request(&f));
        assert_eq!(f.encode().len(), 23 * 4 + 4);
    }
}
