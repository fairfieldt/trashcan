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
    let ring_path = spec.socket_dir.join(format!("{}.{}.ring", peer.name(), spec.name));
    let info_path = spec.socket_dir.join(format!("{}.{}.info", peer.name(), spec.name));
    let listener = listen(&path)?;
    let ring_listener = listen(&ring_path)?;
    std::fs::write(
        &info_path,
        format!("ring={}\nlocal={}\npeer={}\nin_hopid={in_hop}\nout_hopid={out_hop}\n", spec.ring, d.local_uuid, peer.remote_uuid()),
    )?;
    println!("port {}: stream {} (in HopID {in_hop}, out HopID {out_hop}) on {}{{,.ring}}", peer.port, spec.name, path.display());
    let counters = Arc::new(Counters::default());
    let result = (|| -> Result<()> {
        while !peer.gone() {
            let mut fds = [
                libc::pollfd { fd: listener.as_raw_fd(), events: libc::POLLIN, revents: 0 },
                libc::pollfd { fd: ring_listener.as_raw_fd(), events: libc::POLLIN, revents: 0 },
            ];
            // SAFETY: two pollfds.
            if unsafe { libc::poll(fds.as_mut_ptr(), 2, 200) } <= 0 {
                continue;
            }
            let direct = fds[1].revents != 0;
            let Some(client) = accept(if direct { &ring_listener } else { &listener }, Duration::ZERO)? else { continue };
            let r = if direct {
                serve_direct(d, peer, spec, in_hop, out_hop, client, &[&listener, &ring_listener])
            } else {
                serve(d, peer, spec, in_hop, out_hop, client, &counters)
            };
            if let Err(e) = r {
                eprintln!("port {}: stream {}: {e:#}", peer.port, spec.name);
            }
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&ring_path);
    let _ = std::fs::remove_file(&info_path);
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
        // Clients in the render group (who use the GPUs and the stream devices too) may connect.
        let cpath = std::ffi::CString::new(p).unwrap();
        let g = libc::getgrnam(c"render".as_ptr());
        if !g.is_null() {
            libc::chown(cpath.as_ptr(), u32::MAX, (*g).gr_gid);
            libc::chmod(cpath.as_ptr(), 0o660);
        }
        Ok(fd)
    }
}

fn poll_in(fd: i32, timeout: Duration) -> bool {
    let mut p = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
    // SAFETY: one pollfd.
    unsafe { libc::poll(&mut p, 1, timeout.as_millis() as libc::c_int) > 0 }
}

fn accept(l: &OwnedFd, timeout: Duration) -> Result<Option<OwnedFd>> {
    // Never block in accept4: the listening sockets are blocking, and a zero timeout means "only if one waits".
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
    tx_loop(tx, &client, &stop, c, peer, spec.busy_poll);
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

/// A client message: up to this many bytes, cut into 4 KiB DATA frames.
pub const MAX_MESSAGE: usize = 64 * 1024;
const MESSAGE_FRAMES: u32 = (MAX_MESSAGE / MAX_FRAME_SIZE) as u32;

/// Client messages to DATA frames. A client that shuts down its writing side (or sends an empty message) sends a
/// CLOSE and can still read; one that closes the socket ends the session, with a CLOSE if it had not sent one.
fn tx_loop(mut tx: Ring, client: &OwnedFd, stop: &AtomicBool, c: &Counters, peer: &Peer, spin: bool) {
    let fd = client.as_raw_fd();
    let mut buf = vec![0u8; MAX_MESSAGE];
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
        // Room for the largest message before taking one off the socket.
        if tx.tx_room() < MESSAGE_FRAMES.min(tx.size - 1) {
            tx.wait(Duration::from_millis(5));
            continue;
        }
        if !spin && !poll_in(fd, Duration::from_millis(100)) {
            continue;
        }
        // SAFETY: recv into buf.
        let n = unsafe { libc::recv(fd, buf.as_mut_ptr().cast(), buf.len(), if spin { libc::MSG_DONTWAIT } else { 0 }) };
        if n < 0 {
            if spin && std::io::Error::last_os_error().kind() == std::io::ErrorKind::WouldBlock {
                std::hint::spin_loop();
                continue;
            }
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
        for chunk in buf[..n].chunks(MAX_FRAME_SIZE) {
            let slot = tx.tx_slot().expect("room checked");
            slot[..chunk.len()].copy_from_slice(chunk);
            tx.tx_submit(chunk.len(), PDF_FRAME_START, PDF_DATA);
            c.tx_frames.fetch_add(1, Ordering::Relaxed);
        }
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

/// Shared memory for a stream's two rings, handed to a client: descriptors, then buffers, page-aligned pieces.
struct SharedRings {
    fd: OwnedFd,
    ptr: std::ptr::NonNull<u8>,
    len: usize,
    layout: tbd_client::Layout,
}

impl SharedRings {
    fn new(ring: u32) -> Result<SharedRings> {
        let page = 4096u64;
        let desc = (ring as u64 * 16).div_ceil(page) * page;
        let bufs = ring as u64 * MAX_FRAME_SIZE as u64;
        let layout = tbd_client::Layout {
            ring: ring as u64,
            frame_bytes: MAX_FRAME_SIZE as u64,
            tx_desc: 0,
            rx_desc: desc,
            tx_buf: 2 * desc,
            rx_buf: 2 * desc + bufs,
            mem_size: 2 * desc + 2 * bufs,
            ..Default::default()
        };
        let len = layout.mem_size as usize;
        // SAFETY: a new memfd of len bytes, mapped shared; the mapping lives until drop.
        unsafe {
            let fd = libc::memfd_create(c"tbd-stream".as_ptr(), libc::MFD_CLOEXEC);
            anyhow::ensure!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
            let fd = OwnedFd::from_raw_fd(fd);
            anyhow::ensure!(libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) == 0, "ftruncate: {}", std::io::Error::last_os_error());
            let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd.as_raw_fd(), 0);
            anyhow::ensure!(p != libc::MAP_FAILED, "mmap: {}", std::io::Error::last_os_error());
            Ok(SharedRings { fd, ptr: std::ptr::NonNull::new(p.cast()).unwrap(), len, layout })
        }
    }

    /// Maps one piece for the NHI.
    fn map(&self, dma: &dyn gpud_platform::DmaSpace, off: u64, len: u64, access: gpud_platform::DmaAccess) -> Result<gpud_platform::DmaBuffer> {
        let len = len.div_ceil(4096) * 4096;
        anyhow::ensure!(off + len <= self.len as u64, "shared ring piece outside the memory");
        // SAFETY: inside our mapping, which outlives the DMA buffers (they are dropped with the rings, before us).
        unsafe {
            let p = std::ptr::NonNull::new_unchecked(self.ptr.as_ptr().add(off as usize));
            Ok(dma.map_external(p, len as usize, access)?)
        }
    }
}

impl Drop for SharedRings {
    fn drop(&mut self) {
        // SAFETY: unmaps what new mapped.
        unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.len) };
    }
}

/// One client that runs the rings itself (tbd-client `Stream`): rings on shared memory, paths routed, everything
/// handed over; when the client goes, the TX ring comes back here for the CLOSE.
fn serve_direct(d: &Arc<Domain>, peer: &Arc<Peer>, spec: &StreamSpec, in_hop: u32, out_hop: u32, client: OwnedFd, listeners: &[&OwnedFd]) -> Result<()> {
    use gpud_platform::DmaAccess;
    let fd = client.as_raw_fd();
    let refuse = |why: &str| {
        let _ = tbd_client::send_fds(fd, format!("error: {why}").as_bytes(), &[]);
        anyhow::anyhow!("ring client refused: {why}")
    };
    if !poll_in(fd, Duration::from_secs(2)) {
        return Err(refuse("no hello"));
    }
    let mut hello = [0u8; 64];
    // SAFETY: recv into hello.
    let n = unsafe { libc::recv(fd, hello.as_mut_ptr().cast(), hello.len(), 0) };
    if n <= 0 || &hello[..n as usize] != tbd_client::HELLO {
        return Err(refuse("unknown hello"));
    }
    let shared = SharedRings::new(spec.ring)?;
    let mut l = shared.layout;
    let dma = d.nhi.dma();
    let descs = l.ring * 16;
    let bufs = l.ring * l.frame_bytes;
    let txd = shared.map(dma.as_ref(), l.tx_desc, descs, DmaAccess::ReadWrite)?;
    let rxd = shared.map(dma.as_ref(), l.rx_desc, descs, DmaAccess::ReadWrite)?;
    let txb = shared.map(dma.as_ref(), l.tx_buf, bufs, DmaAccess::DeviceReads)?;
    let rxb = shared.map(dma.as_ref(), l.rx_buf, bufs, DmaAccess::DeviceWrites)?;
    l.tx_buf_iova = txb.iova();
    l.rx_buf_iova = rxb.iova();
    let mut tx = d.nhi.ring_on(
        Kind::Tx,
        RingConfig { hop: None, size: spec.ring, frame: true, e2e: false, sof_mask: 0, eof_mask: 0, interrupt: false },
        Some((txd, txb)),
    )?;
    let rx = d.nhi.ring_on(
        Kind::Rx,
        RingConfig {
            hop: None,
            size: spec.ring,
            frame: true,
            e2e: true,
            sof_mask: 1 << PDF_FRAME_START,
            eof_mask: 1 << PDF_DATA | 1 << PDF_CLOSE,
            interrupt: true,
        },
        Some((rxd, rxb)),
    )?;
    // Linux's stream rings throttle interrupts to one per 8 us (TBSTREAM_DEV_THROTTLING).
    let Some(irq) = rx.arm_interrupt(8192) else { return Err(refuse("no MSI-X vector for the ring")) };
    let Some(bar) = d.config.bar.clone() else { return Err(refuse("no BAR to hand over")) };
    l.bar_size = bar.size;
    l.bar_offset = bar.offset;
    l.tx_reg = tx.index_reg() as u64;
    l.rx_reg = rx.index_reg() as u64;
    l.tx_head = tx.head as u64;
    l.tx_tail = tx.tail as u64;
    l.rx_head = rx.head as u64;
    l.rx_tail = rx.tail as u64;
    l.flags = 1;
    let (tx_ring, rx_ring) = (tx.hop, rx.hop);
    d.enable_paths(peer, out_hop, tx_ring, in_hop, rx_ring)?;
    let sent = tbd_client::send_fds(fd, &l.encode(), &[shared.fd.as_raw_fd(), bar.fd.as_raw_fd(), irq]);
    if sent.is_ok() {
        println!("port {}: stream {} handed to a client: TX ring {tx_ring} -> HopID {out_hop}, HopID {in_hop} -> RX ring {rx_ring}", peer.port, spec.name);
        while !peer.gone() && !hung_up(fd, Duration::from_millis(100)) {
            // One client at a time, as an exclusive open of the device: others hear so at once.
            for l in listeners {
                if let Ok(Some(other)) = accept(l, Duration::ZERO) {
                    let _ = tbd_client::send_fds(other.as_raw_fd(), b"error: the stream is in use", &[]);
                }
            }
        }
    }
    // The client is gone: its last frames out, then our CLOSE.
    tx.resync_tx();
    for _ in 0..100 {
        if tx.tx_hw_idle() {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    tx.tx_reap();
    if !peer.gone() {
        send_close(&mut tx);
        for _ in 0..100 {
            tx.tx_reap();
            if tx.empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    drop(tx);
    drop(rx);
    d.disable_paths(peer, tx_ring, in_hop);
    println!("port {}: stream {}: ring client gone", peer.port, spec.name);
    sent.map_err(Into::into)
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

/// DATA frames to client messages, as many consecutive frames per message as are there (up to 64 KiB, sent from
/// the ring buffers); a CLOSE as an empty message.
fn rx_loop(mut rx: Ring, client: &OwnedFd, stop: &AtomicBool, c: &Counters, peer: &Peer) {
    let fd = client.as_raw_fd();
    let max = MESSAGE_FRAMES as usize;
    while !stop.load(Ordering::Relaxed) && !peer.gone() {
        while let Some(first) = rx.rx_peek() {
            let mut iov: Vec<libc::iovec> = Vec::with_capacity(max);
            let mut frames = 0;
            let mut bytes = 0;
            if first.eof == PDF_CLOSE {
                frames = 1;
            } else {
                while iov.len() < max {
                    let Some(f) = rx.rx_peek_nth(frames) else { break };
                    if f.eof == PDF_CLOSE {
                        break;
                    }
                    if f.flags & (DESC_CRC_ERROR | DESC_BUFFER_OVERRUN) != 0 {
                        c.rx_errors.fetch_add(1, Ordering::Relaxed);
                    }
                    let d = rx.rx_data(&f);
                    iov.push(libc::iovec { iov_base: d.as_ptr() as *mut _, iov_len: d.len() });
                    bytes += d.len();
                    frames += 1;
                }
            }
            // SAFETY: sendmsg of buffers in completed RX slots, which stay ours until rx_release; MSG_NOSIGNAL so a
            // gone client is an error, not SIGPIPE.
            let n = unsafe {
                let mut msg: libc::msghdr = std::mem::zeroed();
                msg.msg_iov = iov.as_mut_ptr();
                msg.msg_iovlen = iov.len() as _;
                libc::sendmsg(fd, &msg, libc::MSG_NOSIGNAL)
            };
            for _ in 0..frames {
                rx.rx_release();
            }
            if n < 0 {
                return;
            }
            c.rx_frames.fetch_add(frames as u64, Ordering::Relaxed);
            c.rx_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        }
        rx.wait(Duration::from_millis(10));
    }
}
