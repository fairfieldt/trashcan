//! A tbd stream run in the client's own process.
//!
//! tbd (driver/tbd) offers each USB4STREAM stream on two Unix sockets: `PEER.NAME`, where tbd moves the frames, and
//! `PEER.NAME.ring`, where it hands the stream's two rings over. The client sends [`HELLO`]; tbd answers with a
//! [`Layout`] and three descriptors: the shared memory holding both rings' descriptors and buffers (mapped for the
//! NHI by tbd), the NHI's register BAR (for the rings' index registers), and the eventfd of the RX ring's MSI-X
//! vector. From then on the client posts and completes frames itself, with no system call on the data path. When
//! the client closes the socket (or dies), tbd takes the TX ring back, sends the CLOSE frame and takes the paths
//! down.
//!
//! The client is trusted: the BAR reaches every ring of the NHI.
//!
//! Reads and writes behave as on Linux's `/dev/tbstreamN` opened `O_NONBLOCK`: a write is cut into DATA frames of up
//! to 4 KiB, a read returns the bytes of consecutive frames back to back, a CLOSE from the other host reads once as
//! 0 bytes, and both fail with `WouldBlock` rather than wait. `fd()` becomes readable when frames arrive.

pub mod ring;

#[cfg(target_os = "linux")]
use ring::{MAX_FRAME_SIZE, RingCore};
#[cfg(target_os = "linux")]
use std::io::{self, IoSlice};
#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
#[cfg(target_os = "linux")]
use std::path::Path;
#[cfg(target_os = "linux")]
use std::ptr::NonNull;

pub const HELLO: &[u8] = b"tbd-ring 1";
const MAGIC: u64 = u64::from_le_bytes(*b"tbdring1");

pub const PDF_FRAME_START: u8 = 1;
pub const PDF_DATA: u8 = 2;
pub const PDF_CLOSE: u8 = 3;

/// Where everything is, as tbd hands it over (little-endian u64s on the socket).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    pub ring: u64,
    pub frame_bytes: u64,
    pub mem_size: u64,
    pub tx_desc: u64,
    pub rx_desc: u64,
    pub tx_buf: u64,
    pub rx_buf: u64,
    pub tx_buf_iova: u64,
    pub rx_buf_iova: u64,
    pub bar_size: u64,
    /// Offsets of the rings' index registers in the BAR.
    pub tx_reg: u64,
    pub rx_reg: u64,
    pub tx_head: u64,
    pub tx_tail: u64,
    pub rx_head: u64,
    pub rx_tail: u64,
    /// Bit 0: RX completions raise the eventfd.
    pub flags: u64,
    /// Where the BAR starts in its descriptor (a VFIO device: the region's offset).
    pub bar_offset: u64,
}

const FIELDS: usize = 19;

impl Layout {
    pub fn encode(&self) -> Vec<u8> {
        let v = [
            MAGIC,
            self.ring,
            self.frame_bytes,
            self.mem_size,
            self.tx_desc,
            self.rx_desc,
            self.tx_buf,
            self.rx_buf,
            self.tx_buf_iova,
            self.rx_buf_iova,
            self.bar_size,
            self.tx_reg,
            self.rx_reg,
            self.tx_head,
            self.tx_tail,
            self.rx_head,
            self.rx_tail,
            self.flags,
            self.bar_offset,
        ];
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    pub fn decode(b: &[u8]) -> Option<Layout> {
        if b.len() != FIELDS * 8 {
            return None;
        }
        let v: Vec<u64> = b.chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        (v[0] == MAGIC).then(|| Layout {
            ring: v[1],
            frame_bytes: v[2],
            mem_size: v[3],
            tx_desc: v[4],
            rx_desc: v[5],
            tx_buf: v[6],
            rx_buf: v[7],
            tx_buf_iova: v[8],
            rx_buf_iova: v[9],
            bar_size: v[10],
            tx_reg: v[11],
            rx_reg: v[12],
            tx_head: v[13],
            tx_tail: v[14],
            rx_head: v[15],
            rx_tail: v[16],
            flags: v[17],
            bar_offset: v[18],
        })
    }
}

#[cfg(target_os = "linux")]
/// Sends `data` with descriptors on a Unix socket.
pub fn send_fds(sock: RawFd, data: &[u8], fds: &[RawFd]) -> io::Result<()> {
    let space = unsafe { libc::CMSG_SPACE((fds.len() * 4) as u32) } as usize;
    let mut cbuf = vec![0u8; space.max(1)];
    let mut iov = libc::iovec { iov_base: data.as_ptr() as *mut _, iov_len: data.len() };
    // SAFETY: a msghdr over iov and cbuf, which outlive the call; the control message is filled within cbuf.
    unsafe {
        let mut msg: libc::msghdr = std::mem::zeroed();
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        if !fds.is_empty() {
            msg.msg_control = cbuf.as_mut_ptr().cast();
            msg.msg_controllen = space as _;
            let c = libc::CMSG_FIRSTHDR(&msg);
            (*c).cmsg_level = libc::SOL_SOCKET;
            (*c).cmsg_type = libc::SCM_RIGHTS;
            (*c).cmsg_len = libc::CMSG_LEN((fds.len() * 4) as u32) as _;
            std::ptr::copy_nonoverlapping(fds.as_ptr(), libc::CMSG_DATA(c).cast::<RawFd>(), fds.len());
        }
        if libc::sendmsg(sock, &msg, libc::MSG_NOSIGNAL) < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
/// Receives one message and the descriptors that came with it.
pub fn recv_fds(sock: RawFd, buf: &mut [u8]) -> io::Result<(usize, Vec<OwnedFd>)> {
    let space = unsafe { libc::CMSG_SPACE(16 * 4) } as usize;
    let mut cbuf = vec![0u8; space];
    let mut iov = libc::iovec { iov_base: buf.as_mut_ptr().cast(), iov_len: buf.len() };
    // SAFETY: recvmsg into iov and cbuf; descriptors in SCM_RIGHTS messages are new and ours.
    unsafe {
        let mut msg: libc::msghdr = std::mem::zeroed();
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cbuf.as_mut_ptr().cast();
        msg.msg_controllen = space as _;
        let n = libc::recvmsg(sock, &mut msg, libc::MSG_CMSG_CLOEXEC);
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut fds = Vec::new();
        let mut c = libc::CMSG_FIRSTHDR(&msg);
        while !c.is_null() {
            if (*c).cmsg_level == libc::SOL_SOCKET && (*c).cmsg_type == libc::SCM_RIGHTS {
                let count = ((*c).cmsg_len as usize - libc::CMSG_LEN(0) as usize) / 4;
                let p = libc::CMSG_DATA(c).cast::<RawFd>();
                for i in 0..count {
                    fds.push(OwnedFd::from_raw_fd(p.add(i).read_unaligned()));
                }
            }
            c = libc::CMSG_NXTHDR(&msg, c);
        }
        Ok((n as usize, fds))
    }
}

#[cfg(target_os = "linux")]
struct Map {
    ptr: NonNull<u8>,
    len: usize,
}

#[cfg(target_os = "linux")]
impl Map {
    fn new(fd: &OwnedFd, len: usize, offset: u64) -> io::Result<Map> {
        // SAFETY: a shared read-write mapping of len bytes of fd at offset.
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd.as_raw_fd(), offset as libc::off_t)
        };
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Map { ptr: NonNull::new(p.cast()).unwrap(), len })
    }

    fn at(&self, off: u64, len: u64) -> io::Result<NonNull<u8>> {
        if off.checked_add(len).is_none_or(|end| end > self.len as u64) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "tbd ring layout outside its mapping"));
        }
        // SAFETY: in bounds (checked).
        Ok(unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(off as usize)) })
    }
}

#[cfg(target_os = "linux")]
impl Drop for Map {
    fn drop(&mut self) {
        // SAFETY: unmaps exactly what new mapped.
        unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.len) };
    }
}

#[cfg(target_os = "linux")]
/// A stream whose rings run in this process.
pub struct Stream {
    // Dropped first: tbd takes the rings back when the socket closes.
    sock: OwnedFd,
    tx: RingCore,
    rx: RingCore,
    irq: OwnedFd,
    /// Bytes of the frame at the RX tail already returned by `read`.
    offset: usize,
    /// TX watchdog: the tail when progress was last seen, and when.
    tx_seen: (u32, std::time::Instant),
    /// Times the watchdog rang the doorbell again.
    pub tx_kicks: u64,
    _mem: Map,
    _bar: Map,
}

// SAFETY: the stream owns its mappings and descriptors; one thread drives it at a time (`&mut self`).
#[cfg(target_os = "linux")]
unsafe impl Send for Stream {}

#[cfg(target_os = "linux")]
fn bad(why: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, why.into())
}

#[cfg(target_os = "linux")]
impl Stream {
    /// Takes over the stream offered at `path` (a tbd `.ring` socket).
    pub fn connect(path: &Path) -> io::Result<Stream> {
        use std::os::unix::ffi::OsStrExt;
        // SAFETY: socket and connect with a NUL-terminated path inside sun_path.
        let sock = unsafe {
            let fd = libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0);
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            let sock = OwnedFd::from_raw_fd(fd);
            let mut addr: libc::sockaddr_un = std::mem::zeroed();
            addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
            let p = path.as_os_str().as_bytes();
            if p.len() >= addr.sun_path.len() {
                return Err(bad("socket path too long"));
            }
            for (i, &b) in p.iter().enumerate() {
                addr.sun_path[i] = b as libc::c_char;
            }
            let len = std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t;
            if libc::connect(sock.as_raw_fd(), (&raw const addr).cast(), len) != 0 {
                return Err(io::Error::last_os_error());
            }
            sock
        };
        send_fds(sock.as_raw_fd(), HELLO, &[])?;
        let mut buf = [0u8; 512];
        let (n, mut fds) = recv_fds(sock.as_raw_fd(), &mut buf)?;
        let Some(l) = Layout::decode(&buf[..n]) else {
            return Err(bad(format!("tbd: {}", String::from_utf8_lossy(&buf[..n]))));
        };
        if fds.len() != 3 {
            return Err(bad(format!("tbd sent {} descriptors, not 3", fds.len())));
        }
        let irq = fds.pop().unwrap();
        let bar_fd = fds.pop().unwrap();
        let mem_fd = fds.pop().unwrap();
        let mem = Map::new(&mem_fd, l.mem_size as usize, 0)?;
        let bar = Map::new(&bar_fd, l.bar_size as usize, l.bar_offset)?;
        let ring = l.ring as u32;
        let fb = l.frame_bytes as usize;
        if ring < 2 || fb != MAX_FRAME_SIZE {
            return Err(bad(format!("ring of {ring}, frames of {fb}")));
        }
        let descs = l.ring * 16;
        let bufs = l.ring * l.frame_bytes;
        let interrupt = l.flags & 1 != 0;
        // SAFETY: the layout's areas are inside the mappings (checked by `at`), mapped for the NHI by tbd; the
        // register offsets are the rings' index registers.
        let (mut tx, mut rx) = unsafe {
            (
                RingCore::new(true, ring, fb, mem.at(l.tx_desc, descs)?, mem.at(l.tx_buf, bufs)?, l.tx_buf_iova, bar.at(l.tx_reg, 4)?.cast(), true),
                RingCore::new(false, ring, fb, mem.at(l.rx_desc, descs)?, mem.at(l.rx_buf, bufs)?, l.rx_buf_iova, bar.at(l.rx_reg, 4)?.cast(), interrupt),
            )
        };
        tx.head = l.tx_head as u32;
        tx.tail = l.tx_tail as u32;
        rx.head = l.rx_head as u32;
        rx.tail = l.rx_tail as u32;
        let tx_seen = (tx.tail, std::time::Instant::now());
        Ok(Stream { sock, tx, rx, irq, offset: 0, tx_seen, tx_kicks: 0, _mem: mem, _bar: bar })
    }

    /// Readable (for `poll`) when the RX ring has completed frames.
    pub fn fd(&self) -> RawFd {
        self.irq.as_raw_fd()
    }

    /// The receive ring's size, in frames.
    pub fn ring_size(&self) -> u32 {
        self.rx.size
    }

    /// Falcon Ridge's TX engine sometimes stops with frames still posted, until the next doorbell (seen at the end
    /// of bursts; nothing is lost). When frames are pending and none has gone for 50 us, ring it again with the
    /// same index. Runs on every read and write: a caller waiting in poll() comes back within milliseconds.
    fn tx_watch(&mut self) {
        self.tx.tx_reap();
        let now = std::time::Instant::now();
        if self.tx.empty() || self.tx.tail != self.tx_seen.0 {
            self.tx_seen = (self.tx.tail, now);
        } else if now.duration_since(self.tx_seen.1) > std::time::Duration::from_micros(50) {
            self.tx.tx_kick();
            self.tx_kicks += 1;
            self.tx_seen.1 = now;
        }
    }

    /// Bytes of consecutive frames; 0 for a CLOSE (only when it comes first); `WouldBlock` when nothing is there.
    pub fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.tx_watch();
        // Clear the eventfd before looking at the ring: a frame that lands after the look raises it again.
        let mut v = 0u64;
        // SAFETY: an 8-byte read from a non-blocking eventfd.
        unsafe { libc::read(self.irq.as_raw_fd(), (&mut v as *mut u64).cast(), 8) };
        let mut n = 0;
        while n < buf.len() {
            let Some(f) = self.rx.rx_peek() else { break };
            if f.eof == PDF_CLOSE {
                if n == 0 {
                    self.rx.rx_release();
                    self.offset = 0;
                    return Ok(0);
                }
                break;
            }
            let data = &self.rx.rx_data(&f)[self.offset..];
            let k = data.len().min(buf.len() - n);
            buf[n..n + k].copy_from_slice(&data[..k]);
            n += k;
            if k == data.len() {
                self.rx.rx_release();
                self.offset = 0;
            } else {
                self.offset += k;
            }
        }
        if n == 0 { Err(io::ErrorKind::WouldBlock.into()) } else { Ok(n) }
    }

    /// Sends as many bytes as the TX ring has room for, in DATA frames of up to 4 KiB.
    pub fn writev(&mut self, bufs: &[IoSlice]) -> io::Result<usize> {
        self.tx_watch();
        let mut n = 0;
        let mut parts = bufs.iter().map(|b| &b[..]).filter(|b| !b.is_empty());
        let mut cur: &[u8] = parts.next().unwrap_or(&[]);
        while !cur.is_empty() {
            let Some(slot) = self.tx.tx_slot() else { break };
            let mut len = 0;
            while len < slot.len() && !cur.is_empty() {
                let k = cur.len().min(slot.len() - len);
                slot[len..len + k].copy_from_slice(&cur[..k]);
                len += k;
                cur = &cur[k..];
                if cur.is_empty() {
                    cur = parts.next().unwrap_or(&[]);
                }
            }
            // A doorbell per frame, as Linux rings it: Falcon Ridge's TX engine can stop partway through frames
            // posted under one doorbell and wait for the next.
            self.tx.tx_submit(len, PDF_FRAME_START, PDF_DATA);
            n += len;
        }
        if n == 0 && bufs.iter().any(|b| !b.is_empty()) { Err(io::ErrorKind::WouldBlock.into()) } else { Ok(n) }
    }

    pub fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.writev(&[IoSlice::new(data)])
    }

    /// Waits until a frame arrives or `timeout_ms` passes (-1: no limit), waking every 2 ms for the TX watchdog
    /// while frames are going out.
    pub fn wait_readable(&mut self, timeout_ms: i32) -> bool {
        let deadline = (timeout_ms >= 0).then(|| std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms as u64));
        loop {
            self.tx_watch();
            if self.rx.rx_peek().is_some() {
                return true;
            }
            let left = deadline.map(|d| d.saturating_duration_since(std::time::Instant::now()).as_millis() as i32);
            if left == Some(0) {
                return false;
            }
            let step = if self.tx.empty() { left.unwrap_or(-1) } else { left.map_or(2, |l| l.min(2)) };
            let mut p = libc::pollfd { fd: self.irq.as_raw_fd(), events: libc::POLLIN, revents: 0 };
            // SAFETY: one pollfd.
            unsafe { libc::poll(&mut p, 1, step) };
        }
    }

    /// Frames sent and not yet taken by the NHI.
    pub fn tx_pending(&mut self) -> u32 {
        self.tx.tx_reap();
        self.tx.in_flight()
    }

    /// The rings' indexes and the descriptors around the RX tail, for diagnosing a stall.
    pub fn debug_state(&self) -> String {
        format!(
            "tx {}..{} {} | rx {}..{} offset {} {}",
            self.tx.tail,
            self.tx.head,
            self.tx.debug_descs(2),
            self.rx.tail,
            self.rx.head,
            self.offset,
            self.rx.debug_descs(2)
        )
    }

    /// The socket to tbd (closing it ends the stream).
    pub fn control_fd(&self) -> RawFd {
        self.sock.as_raw_fd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_round_trips() {
        let l = Layout { ring: 256, frame_bytes: 4096, mem_size: 1 << 21, tx_reg: 0x28, rx_reg: 0x8028, flags: 1, ..Default::default() };
        assert_eq!(Layout::decode(&l.encode()), Some(l));
        assert_eq!(Layout::decode(b"error: busy"), None);
    }
}
