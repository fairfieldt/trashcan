//! Stream checks that run on either end of a cable: on a tbd stream socket or on a Linux `/dev/tbstreamN`.
//!
//! `echo` returns every byte it reads. `bench` sends a pseudo-random byte stream through an echo with at most a
//! window of bytes unanswered (a stream has no transmit flow control: more than the other side's receive ring holds
//! is lost), checks every byte that comes back, and reports the rate. `ping` measures round trips of one message.

use anyhow::{Context, Result, bail, ensure};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::Path;
use std::time::{Duration, Instant};

/// A stream device or a tbd stream socket; a read of 0 bytes is a CLOSE.
pub struct Endpoint {
    fd: OwnedFd,
    socket: bool,
}

pub const FRAME: usize = 4096;

impl Endpoint {
    pub fn open(path: &Path) -> Result<Endpoint> {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::FileTypeExt;
        let meta = std::fs::metadata(path).with_context(|| path.display().to_string())?;
        if meta.file_type().is_socket() {
            // SAFETY: a socket and connect with a NUL-terminated path.
            unsafe {
                let fd = libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0);
                ensure!(fd >= 0, "socket: {}", io::Error::last_os_error());
                let fd = OwnedFd::from_raw_fd(fd);
                let mut addr: libc::sockaddr_un = std::mem::zeroed();
                addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
                let p = path.as_os_str().as_bytes();
                ensure!(p.len() < addr.sun_path.len(), "socket path too long");
                for (i, &b) in p.iter().enumerate() {
                    addr.sun_path[i] = b as libc::c_char;
                }
                let len = std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t;
                ensure!(libc::connect(fd.as_raw_fd(), (&raw const addr).cast(), len) == 0, "connect {}: {}", path.display(), io::Error::last_os_error());
                Ok(Endpoint { fd, socket: true })
            }
        } else {
            let f = std::fs::OpenOptions::new().read(true).write(true).open(path).with_context(|| path.display().to_string())?;
            Ok(Endpoint { fd: f.into(), socket: false })
        }
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        // SAFETY: a read into buf.
        let n = unsafe { libc::read(self.fd.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) };
        if n < 0 { Err(io::Error::last_os_error()) } else { Ok(n as usize) }
    }

    /// Writes all of `data`, in frames of at most 4 KiB on a socket.
    pub fn write_all(&self, mut data: &[u8]) -> io::Result<()> {
        while !data.is_empty() {
            let len = if self.socket { data.len().min(FRAME) } else { data.len() };
            // SAFETY: a write of data's first len bytes.
            let n = unsafe { libc::write(self.fd.as_raw_fd(), data.as_ptr().cast(), len) };
            if n < 0 {
                return Err(io::Error::last_os_error());
            }
            data = &data[n as usize..];
        }
        Ok(())
    }

    /// Sends a CLOSE: shuts down the writing side of a socket; a device sends one when it is closed.
    pub fn close_write(&self) {
        if self.socket {
            // SAFETY: shutdown of our socket's writing side.
            unsafe { libc::shutdown(self.fd.as_raw_fd(), libc::SHUT_WR) };
        }
    }

    pub fn wait_readable(&self, timeout: Duration) -> bool {
        let mut p = libc::pollfd { fd: self.fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        // SAFETY: one pollfd.
        unsafe { libc::poll(&mut p, 1, timeout.as_millis() as libc::c_int) > 0 }
    }
}

/// xorshift64*: the same bytes on both ends from the same seed.
struct Pattern(u64);

impl Pattern {
    fn fill(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let v = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
    }
}

/// Echoes until a CLOSE (or the device reports end of file), `sessions` times (0: forever).
pub fn echo(path: &Path, sessions: u32) -> Result<()> {
    let mut n = 0;
    loop {
        let ep = Endpoint::open(path)?;
        let mut buf = vec![0u8; 1 << 20];
        let mut total = 0u64;
        loop {
            match ep.read(&mut buf) {
                Ok(0) => break,
                Ok(k) => {
                    ep.write_all(&buf[..k])?;
                    total += k as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e.into()),
            }
        }
        ep.close_write();
        eprintln!("echo: {total} bytes");
        drop(ep);
        n += 1;
        if sessions != 0 && n >= sessions {
            return Ok(());
        }
    }
}

/// Sends `bytes` through an echo with at most `window` bytes unanswered and checks them.
pub fn bench(path: &Path, bytes: u64, window: usize, write: usize) -> Result<()> {
    ensure!(window >= write && write > 0, "window {window} smaller than a write of {write}");
    let ep = Endpoint::open(path)?;
    let seed = 0x9e37_79b9_7f4a_7c15;
    let (mut tx_pat, mut rx_pat) = (Pattern(seed), Pattern(seed));
    let mut out = vec![0u8; write];
    let mut inbuf = vec![0u8; 1 << 20];
    let mut expect = vec![0u8; 1 << 20];
    let (mut sent, mut got) = (0u64, 0u64);
    let t0 = Instant::now();
    let mut last = Instant::now();
    while got < bytes {
        while sent < bytes && sent - got + write as u64 <= window as u64 {
            let n = write.min((bytes - sent) as usize);
            tx_pat.fill(&mut out[..n]);
            ep.write_all(&out[..n])?;
            sent += n as u64;
        }
        if !ep.wait_readable(Duration::from_secs(2)) {
            bail!("no echo for 2 s: {got} of {sent} bytes back (frames lost: the window is larger than a ring holds?)");
        }
        let k = ep.read(&mut inbuf)?;
        ensure!(k > 0, "CLOSE after {got} bytes");
        rx_pat.fill(&mut expect[..k]);
        if inbuf[..k] != expect[..k] {
            let at = inbuf[..k].iter().zip(&expect[..k]).position(|(a, b)| a != b).unwrap();
            bail!("echoed byte {} differs", got + at as u64);
        }
        got += k as u64;
        last = Instant::now();
    }
    let secs = last.duration_since(t0).as_secs_f64();
    println!(
        "{got} bytes echoed intact in {secs:.3} s: {:.1} MB/s each way ({:.2} Gb/s), window {window}, writes of {write}",
        got as f64 / secs / 1e6,
        got as f64 * 8.0 / secs / 1e9
    );
    ep.close_write();
    let _ = ep.wait_readable(Duration::from_secs(1));
    Ok(())
}

/// Round trips of `size`-byte messages through an echo.
pub fn ping(path: &Path, count: usize, size: usize) -> Result<()> {
    let ep = Endpoint::open(path)?;
    let mut msg = vec![0u8; size];
    let mut buf = vec![0u8; size.max(FRAME)];
    let mut rtts = Vec::with_capacity(count);
    for i in 0..count + 10 {
        msg[..8.min(size)].copy_from_slice(&(i as u64).to_le_bytes()[..8.min(size)]);
        let t = Instant::now();
        ep.write_all(&msg)?;
        let mut got = 0;
        while got < size {
            ensure!(ep.wait_readable(Duration::from_secs(2)), "no echo for 2 s (message {i})");
            let k = ep.read(&mut buf[got..])?;
            ensure!(k > 0, "CLOSE during message {i}");
            got += k;
        }
        ensure!(buf[..size] == msg[..], "message {i} came back different");
        if i >= 10 {
            rtts.push(t.elapsed());
        }
    }
    rtts.sort();
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    let pct = |p: f64| us(rtts[((rtts.len() - 1) as f64 * p) as usize]);
    println!(
        "{count} round trips of {size} bytes: min {:.1} us, median {:.1}, p99 {:.1}, max {:.1}",
        us(rtts[0]),
        pct(0.5),
        pct(0.99),
        us(*rtts.last().unwrap())
    );
    ep.close_write();
    let _ = ep.wait_readable(Duration::from_secs(1));
    Ok(())
}
