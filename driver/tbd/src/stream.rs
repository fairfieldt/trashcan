//! USB4STREAM: raw frames between two hosts on HopIDs both sides agree on, compatible with Linux's
//! thunderbolt_stream (drivers/thunderbolt/stream.c, 7.3, with kernel/patches-7.3/0005-0007).
//!
//! Frames carry the writer's bytes as they are: DATA (end PDF 2, up to 4 KiB) and CLOSE (end PDF 3, 256 zero
//! bytes) when the writer closes; the start PDF is 1. The receive ring has end-to-end flow control, the transmit
//! ring none. Each stream is announced in the `stream` service directory as a subdirectory named after it, holding
//! its `inhopid` and `outhopid`.
//!
//! Clients reach a stream through a `SOCK_SEQPACKET` Unix socket, one client at a time, as with an exclusive open of
//! `/dev/tbstreamN`: every message is one frame each way, and a CLOSE from the other host arrives as an empty
//! message (a read of 0 bytes, which a stream device reader sees too). Paths are routed while a client is
//! connected; when it goes, a CLOSE is sent and the paths come down.

use crate::domain::{Domain, Peer};
use crate::nhi::{DESC_BUFFER_OVERRUN, DESC_CRC_ERROR, Kind, MAX_FRAME_SIZE, Ring, RingConfig};
use crate::property::Dir;
use crate::uuid::Uuid;
use anyhow::{Context, Result};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// Stream property directory: 3a1cb984-c4d9-4469-a277-ce2fdfd11f0d.
pub const DIR_UUID: Uuid = Uuid::init(0x3a1cb984, 0xc4d9, 0x4469, [0xa2, 0x77, 0xce, 0x2f, 0xdf, 0xd1, 0x1f, 0x0d]);

pub const PDF_FRAME_START: u8 = 1;
pub const PDF_DATA: u8 = 2;
pub const PDF_CLOSE: u8 = 3;
const CLOSE_SIZE: usize = 256;

#[derive(Clone, Debug)]
pub struct StreamSpec {
    /// At most 8 characters: it is a property key.
    pub name: String,
    pub in_hop: Option<u32>,
    pub out_hop: Option<u32>,
    pub ring: u32,
    pub socket_dir: PathBuf,
    /// Poll the rings instead of taking interrupts (lower latency, a busy core per direction).
    pub busy_poll: bool,
}

/// The `stream` service directory (what thunderbolt_stream registers).
pub fn property_dir() -> Dir {
    let mut d = Dir::new(Some(DIR_UUID));
    d.add_immediate("prtcid", 1).add_immediate("prtcvers", 1).add_immediate("prtcrevs", 0).add_immediate("prtcstns", 0);
    d
}

pub fn spawn(d: Arc<Domain>, peer: Arc<Peer>, spec: StreamSpec) {
    std::thread::Builder::new()
        .name(format!("tbd-st{}-{}", peer.port, spec.name))
        .spawn(move || {
            if let Err(e) = run(&d, &peer, &spec) {
                eprintln!("port {}: stream {}: {e:#}", peer.port, spec.name);
            }
        })
        .expect("spawn stream");
}

#[derive(Default)]
pub struct Counters {
    pub rx_frames: AtomicU64,
    pub rx_bytes: AtomicU64,
    pub rx_errors: AtomicU64,
    pub tx_frames: AtomicU64,
    pub tx_bytes: AtomicU64,
}

fn run(d: &Arc<Domain>, peer: &Arc<Peer>, spec: &StreamSpec) -> Result<()> {
    let (remote_max, local_max) = {
        let s = peer.state.lock().unwrap();
        (s.remote.as_ref().unwrap().max_hopid, d.router.read().unwrap().port(peer.port).max_in_hop)
    };
    let in_hop = peer.alloc_hop(true, spec.in_hop, local_max).context("input HopID")?;
    let out_hop = match peer.alloc_hop(false, spec.out_hop, remote_max) {
        Ok(h) => h,
        Err(e) => {
            peer.free_hop(true, in_hop);
            return Err(e.context("output HopID"));
        }
    };
    let mut entry = Dir::new(None);
    let mut sub = Dir::new(Some(Uuid::random()));
    sub.add_immediate("inhopid", in_hop).add_immediate("outhopid", out_hop);
    entry.add_dir(&spec.name, sub);
    d.set_service_props(peer, &format!("stream/{}", spec.name), entry);

    std::fs::create_dir_all(&spec.socket_dir)?;
    let path = spec.socket_dir.join(format!("{}.{}", peer.name(), spec.name));
    let listener = listen(&path)?;
    println!("port {}: stream {} (in HopID {in_hop}, out HopID {out_hop}) on {}", peer.port, spec.name, path.display());
    let counters = Arc::new(Counters::default());
    let result = (|| -> Result<()> {
        while !peer.gone() {
            let Some(client) = accept(&listener, Duration::from_millis(200))? else { continue };
            if let Err(e) = serve(d, peer, spec, in_hop, out_hop, client, &counters) {
                eprintln!("port {}: stream {}: {e:#}", peer.port, spec.name);
            }
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(&path);
    peer.free_hop(true, in_hop);
    peer.free_hop(false, out_hop);
    result
}

fn listen(path: &std::path::Path) -> Result<OwnedFd> {
    let _ = std::fs::remove_file(path);
    // SAFETY: socket/bind/listen on a descriptor this function owns, with a NUL-terminated sun_path.
    unsafe {
        let fd = libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0);
        anyhow::ensure!(fd >= 0, "socket: {}", std::io::Error::last_os_error());
        let fd = OwnedFd::from_raw_fd(fd);
        let mut addr: libc::sockaddr_un = std::mem::zeroed();
        addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
        let p = path.as_os_str().as_encoded_bytes();
        anyhow::ensure!(p.len() < addr.sun_path.len(), "socket path {} too long", path.display());
        for (i, &b) in p.iter().enumerate() {
            addr.sun_path[i] = b as libc::c_char;
        }
        let len = std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t;
        anyhow::ensure!(libc::bind(fd.as_raw_fd(), (&raw const addr).cast(), len) == 0, "bind {}: {}", path.display(), std::io::Error::last_os_error());
        anyhow::ensure!(libc::listen(fd.as_raw_fd(), 1) == 0, "listen: {}", std::io::Error::last_os_error());
        Ok(fd)
    }
}

fn poll_in(fd: i32, timeout: Duration) -> bool {
    let mut p = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
    // SAFETY: one pollfd.
    unsafe { libc::poll(&mut p, 1, timeout.as_millis() as libc::c_int) > 0 }
}

fn accept(l: &OwnedFd, timeout: Duration) -> Result<Option<OwnedFd>> {
    if !poll_in(l.as_raw_fd(), timeout) {
        return Ok(None);
    }
    // SAFETY: accept on a listening socket.
    let fd = unsafe { libc::accept4(l.as_raw_fd(), std::ptr::null_mut(), std::ptr::null_mut(), libc::SOCK_CLOEXEC) };
    if fd < 0 {
        return Ok(None);
    }
    // SAFETY: a new descriptor from accept4.
    Ok(Some(unsafe { OwnedFd::from_raw_fd(fd) }))
}

/// One client: rings up, paths routed, frames both ways until the client goes or the peer is unplugged
/// (`tbstream_dev_start`, read/write, `tbstream_dev_stop`).
fn serve(d: &Arc<Domain>, peer: &Arc<Peer>, spec: &StreamSpec, in_hop: u32, out_hop: u32, client: OwnedFd, c: &Arc<Counters>) -> Result<()> {
    let interrupt = !spec.busy_poll;
    let tx = d.nhi.ring(Kind::Tx, RingConfig { hop: None, size: spec.ring, frame: true, e2e: false, sof_mask: 0, eof_mask: 0, interrupt })?;
    let rx = d.nhi.ring(
        Kind::Rx,
        RingConfig {
            hop: None,
            size: spec.ring,
            frame: true,
            e2e: true,
            sof_mask: 1 << PDF_FRAME_START,
            eof_mask: 1 << PDF_DATA | 1 << PDF_CLOSE,
            interrupt,
        },
    )?;
    let (tx_ring, rx_ring) = (tx.hop, rx.hop);
    d.enable_paths(peer, out_hop, tx_ring, in_hop, rx_ring)?;
    println!("port {}: stream {} open: TX ring {tx_ring} -> HopID {out_hop}, HopID {in_hop} -> RX ring {rx_ring}", peer.port, spec.name);
    let client = Arc::new(client);
    let stop = Arc::new(AtomicBool::new(false));
    let (cl, st, cc, p) = (client.clone(), stop.clone(), c.clone(), peer.clone());
    let rxh = std::thread::Builder::new().name("tbd-st-rx".into()).spawn(move || rx_loop(rx, &cl, &st, &cc, &p))?;
    tx_loop(tx, &client, &stop, c, peer);
    stop.store(true, Ordering::Relaxed);
    let _ = rxh.join();
    d.disable_paths(peer, tx_ring, in_hop);
    println!(
        "port {}: stream {} closed: sent {} frames ({} bytes), received {} ({} bytes, {} bad)",
        peer.port,
        spec.name,
        c.tx_frames.load(Ordering::Relaxed),
        c.tx_bytes.load(Ordering::Relaxed),
        c.rx_frames.load(Ordering::Relaxed),
        c.rx_bytes.load(Ordering::Relaxed),
        c.rx_errors.load(Ordering::Relaxed)
    );
    Ok(())
}

/// Client messages to DATA frames. A client that shuts down its writing side (or sends an empty message) sends a
/// CLOSE and can still read; one that closes the socket ends the session, with a CLOSE if it had not sent one.
fn tx_loop(mut tx: Ring, client: &OwnedFd, stop: &AtomicBool, c: &Counters, peer: &Peer) {
    let fd = client.as_raw_fd();
    let mut buf = vec![0u8; MAX_FRAME_SIZE];
    let mut closed = false;
    loop {
        if peer.gone() || stop.load(Ordering::Relaxed) {
            break;
        }
        tx.tx_reap();
        if closed {
            // Only a hangup is left to wait for.
            if hung_up(fd, Duration::from_millis(100)) {
                break;
            }
            continue;
        }
        // Room for the next frame before taking it off the socket.
        if tx.full() {
            tx.wait(Duration::from_millis(5));
            continue;
        }
        if !poll_in(fd, Duration::from_millis(100)) {
            continue;
        }
        // SAFETY: recv into buf.
        let n = unsafe { libc::recv(fd, buf.as_mut_ptr().cast(), buf.len(), 0) };
        if n < 0 {
            break;
        }
        if n == 0 {
            send_close(&mut tx);
            closed = true;
            if hung_up(fd, Duration::ZERO) {
                break;
            }
            continue;
        }
        let n = n as usize;
        let slot = tx.tx_slot().expect("room checked");
        slot[..n].copy_from_slice(&buf[..n]);
        tx.tx_submit(n, PDF_FRAME_START, PDF_DATA);
        c.tx_frames.fetch_add(1, Ordering::Relaxed);
        c.tx_bytes.fetch_add(n as u64, Ordering::Relaxed);
    }
    stop.store(true, Ordering::Relaxed);
    if !closed {
        send_close(&mut tx);
    }
    // Let the ring drain before it stops (`tbstream_dev_stop`).
    for _ in 0..100 {
        tx.tx_reap();
        if tx.empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn hung_up(fd: i32, timeout: Duration) -> bool {
    let mut p = libc::pollfd { fd, events: 0, revents: 0 };
    // SAFETY: one pollfd.
    unsafe { libc::poll(&mut p, 1, timeout.as_millis() as libc::c_int) > 0 && p.revents & (libc::POLLHUP | libc::POLLERR) != 0 }
}

/// `tbstream_dev_send_close`: a 256-byte frame of zeros with the CLOSE PDF.
fn send_close(tx: &mut Ring) {
    for _ in 0..100 {
        tx.tx_reap();
        if let Some(slot) = tx.tx_slot() {
            slot[..CLOSE_SIZE].fill(0);
            tx.tx_submit(CLOSE_SIZE, PDF_FRAME_START, PDF_CLOSE);
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// DATA frames to client messages; a CLOSE as an empty message.
fn rx_loop(mut rx: Ring, client: &OwnedFd, stop: &AtomicBool, c: &Counters, peer: &Peer) {
    let fd = client.as_raw_fd();
    while !stop.load(Ordering::Relaxed) && !peer.gone() {
        while let Some(f) = rx.rx_peek() {
            if f.flags & (DESC_CRC_ERROR | DESC_BUFFER_OVERRUN) != 0 {
                c.rx_errors.fetch_add(1, Ordering::Relaxed);
            }
            let data = if f.eof == PDF_CLOSE { &[][..] } else { rx.rx_data(&f) };
            // SAFETY: send of data's bytes; MSG_NOSIGNAL so a gone client is an error, not SIGPIPE.
            let n = unsafe { libc::send(fd, data.as_ptr().cast(), data.len(), libc::MSG_NOSIGNAL) };
            let len = data.len();
            rx.rx_release();
            if n < 0 {
                return;
            }
            c.rx_frames.fetch_add(1, Ordering::Relaxed);
            c.rx_bytes.fetch_add(len as u64, Ordering::Relaxed);
        }
        rx.wait(Duration::from_millis(10));
    }
}
