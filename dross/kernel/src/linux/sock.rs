//! Unix sockets (stream, datagram and seqpacket; paths in one namespace, no filesystem entries) with fd passing
//! (`SCM_RIGHTS`), and pipes, which are one-way streams.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use spin::Mutex;

use super::fs::{File, Kind, POLLERR, POLLHUP, POLLIN, POLLOUT};
use super::*;
use crate::sched::WaitCell;

const SOCK_STREAM: u32 = 1;
const SOCK_DGRAM: u32 = 2;
const SOCK_SEQPACKET: u32 = 5;
const SOCK_NONBLOCK: u32 = 0o4000;
const SOCK_CLOEXEC: u32 = 0o2000000;
const MSG_PEEK: u64 = 2;
const MSG_TRUNC: u64 = 0x20;
const MSG_DONTWAIT: u64 = 0x40;
const MSG_CMSG_CLOEXEC: u64 = 0x4000_0000;
/// Bytes a socket may hold unread before writers wait.
const RX_LIMIT: usize = 8 << 20;

struct Packet {
    data: Vec<u8>,
    off: usize,
    fds: Vec<Arc<File>>,
}

struct Rx {
    packets: VecDeque<Packet>,
    bytes: usize,
    /// The writer went away or shut down: reads end with EOF once the queue is empty.
    closed: bool,
}

/// One end of a connected socket pair or pipe: it receives into `rx`, sends into the peer's.
pub struct Endpoint {
    ty: u32,
    rx: WaitCell<Rx>,
    peer: Mutex<Weak<Endpoint>>,
    pipe: bool,
    /// A pipe's write end (its peer is the read end; a read end has no peer).
    writer: bool,
}

impl Endpoint {
    fn new(ty: u32, pipe: bool, writer: bool) -> Endpoint {
        Endpoint { ty, rx: WaitCell::new(Rx { packets: VecDeque::new(), bytes: 0, closed: false }), peer: Mutex::new(Weak::new()), pipe, writer }
    }

    pub fn pair(ty: u32) -> (Arc<Endpoint>, Arc<Endpoint>) {
        let a = Arc::new(Endpoint::new(ty, false, false));
        let b = Arc::new(Endpoint::new(ty, false, false));
        *a.peer.lock() = Arc::downgrade(&b);
        *b.peer.lock() = Arc::downgrade(&a);
        (a, b)
    }

    /// (read end, write end).
    pub fn pipe() -> (Arc<Endpoint>, Arc<Endpoint>) {
        let r = Arc::new(Endpoint::new(SOCK_STREAM, true, false));
        let w = Arc::new(Endpoint::new(SOCK_STREAM, true, true));
        *w.peer.lock() = Arc::downgrade(&r);
        (r, w)
    }

    pub fn is_pipe(&self) -> bool {
        self.pipe
    }

    pub fn send(&self, data: &[u8], fds: Vec<Arc<File>>, nonblock: bool) -> Result<usize, Errno> {
        let peer = self.peer.lock().upgrade().ok_or(EPIPE)?;
        if peer.rx.with(|r| r.closed) {
            return Err(EPIPE);
        }
        let mut item = Some(Packet { data: data.to_vec(), off: 0, fds });
        let n = data.len();
        let r = peer.rx.wait(if nonblock { 0 } else { u64::MAX }, |rx| {
            if rx.bytes > 0 && rx.bytes + n > RX_LIMIT {
                return None;
            }
            rx.bytes += n;
            rx.packets.push_back(item.take().unwrap());
            Some(())
        });
        match r {
            Ok(()) => {
                // Wake readers (wait only wakes on update).
                peer.rx.update(|_| {});
                Ok(n)
            }
            Err(WaitError::TimedOut) => Err(EAGAIN),
            Err(e) => Err(wait_errno(e)),
        }
    }

    /// Receives up to `max` bytes and the fds of the packet they come from. Returns (data, fds, full length of
    /// the packet for datagram types).
    pub fn recv(&self, max: usize, flags: u64, nonblock: bool, _: bool) -> Result<(Vec<u8>, Vec<Arc<File>>, usize), Errno> {
        let peek = flags & MSG_PEEK != 0;
        let stream = self.ty == SOCK_STREAM;
        let r = self.rx.wait(if nonblock || flags & MSG_DONTWAIT != 0 { 0 } else { u64::MAX }, |rx| {
            if rx.packets.is_empty() {
                // EOF once the writer is gone (its drop sets `closed`) or shut down.
                return if rx.closed { Some((Vec::new(), Vec::new(), 0)) } else { None };
            }
            if stream {
                let mut out = Vec::new();
                let mut fds = Vec::new();
                let mut i = 0;
                while out.len() < max && i < rx.packets.len() {
                    let pk = &mut rx.packets[i];
                    // fds arrive with the first byte of their packet; don't merge past a packet that has some.
                    if i > 0 && !pk.fds.is_empty() {
                        break;
                    }
                    let n = (max - out.len()).min(pk.data.len() - pk.off);
                    out.extend_from_slice(&pk.data[pk.off..pk.off + n]);
                    if !peek {
                        fds.append(&mut pk.fds);
                        pk.off += n;
                    }
                    i += 1;
                }
                if !peek {
                    rx.bytes -= out.len();
                    while rx.packets.front().is_some_and(|p| p.off == p.data.len()) {
                        rx.packets.pop_front();
                    }
                }
                let n = out.len();
                Some((out, fds, n))
            } else {
                let pk = if peek { None } else { rx.packets.pop_front() };
                let (data, fds) = match pk {
                    Some(p) => {
                        rx.bytes -= p.data.len();
                        (p.data, p.fds)
                    }
                    None => (rx.packets.front().unwrap().data.clone(), Vec::new()),
                };
                let full = data.len();
                Some((data[..full.min(max)].to_vec(), fds, full))
            }
        });
        match r {
            Ok(v) => {
                // Writers waiting for room.
                if !peek {
                    self.rx.update(|_| {});
                }
                Ok(v)
            }
            Err(WaitError::TimedOut) => Err(EAGAIN),
            Err(e) => Err(wait_errno(e)),
        }
    }

    pub fn shutdown_write(&self) {
        if let Some(peer) = self.peer.lock().upgrade() {
            peer.rx.update(|r| r.closed = true);
        }
    }

    pub fn poll(&self, me: &Arc<Thread>, events: u16) -> u16 {
        let mut r = self.rx.poll(me, |rx| {
            let mut r = 0;
            if !rx.packets.is_empty() {
                r |= POLLIN;
            }
            if rx.closed {
                r |= POLLIN | POLLHUP;
            }
            r
        });
        // Writable while the peer is there; a pipe's write end without its reader is an error.
        let peer_alive = self.peer.lock().strong_count() > 0;
        if peer_alive {
            r |= POLLOUT;
        } else if self.writer {
            r |= POLLERR;
        }
        r & (events | POLLERR | POLLHUP)
    }

    pub fn unpoll(&self, me: &Arc<Thread>) {
        self.rx.unregister(me);
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        if let Some(peer) = self.peer.lock().upgrade() {
            peer.rx.update(|r| r.closed = true);
        }
    }
}

pub struct Listener {
    ty: u32,
    path: String,
    queue: WaitCell<VecDeque<Arc<Endpoint>>>,
}

impl Listener {
    pub fn poll(&self, me: &Arc<Thread>) -> u16 {
        self.queue.poll(me, |q| if q.is_empty() { 0 } else { POLLIN })
    }

    pub fn unpoll(&self, me: &Arc<Thread>) {
        self.queue.unregister(me);
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        let mut ns = NAMES.lock();
        if ns.get(&self.path).is_some_and(|w| w.strong_count() == 0) {
            ns.remove(&self.path);
        }
    }
}

static NAMES: Mutex<BTreeMap<String, Weak<Listener>>> = Mutex::new(BTreeMap::new());

/// A socket that isn't connected yet.
pub enum State {
    Fresh { ty: u32 },
    Bound { ty: u32, path: String },
}

fn read_path(p: &Process, addr: u64, len: u64) -> Result<String, Errno> {
    if len < 2 || len > 110 {
        return Err(EINVAL);
    }
    let b = read_user(p, addr, len)?;
    if u16::from_le_bytes([b[0], b[1]]) != 1 {
        return Err(EAFNOSUPPORT);
    }
    let raw = &b[2..];
    // Abstract names (leading NUL) and paths share the namespace.
    let raw = if raw.first() == Some(&0) { &raw[1..] } else { raw };
    let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
    String::from_utf8(raw[..end].to_vec()).map_err(|_| EINVAL)
}

fn socket_type(f: &File) -> Option<u32> {
    match &f.kind {
        Kind::Socket(s) => Some(match &*s.lock() {
            State::Fresh { ty } | State::Bound { ty, .. } => *ty,
        }),
        Kind::Stream(e) => Some(e.ty),
        Kind::Listener(l) => Some(l.ty),
        _ => None,
    }
}

/// sendmsg/recvmsg's header.
struct MsgHdr {
    iov: Vec<(u64, u64)>,
    control: u64,
    controllen: u64,
}

fn read_msghdr(p: &Process, ptr: u64) -> Result<MsgHdr, Errno> {
    let b = read_user(p, ptr, 56)?;
    let u = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    Ok(MsgHdr { iov: read_iov(p, u(16), u(24))?, control: u(32), controllen: u(40) })
}

/// The fds of every SCM_RIGHTS message in a control buffer.
fn rights_in(p: &Process, control: u64, len: u64) -> Result<Vec<Arc<File>>, Errno> {
    let mut fds = Vec::new();
    if control == 0 || len < 16 {
        return Ok(fds);
    }
    let b = read_user(p, control, len.min(4096))?;
    let mut at = 0;
    while at + 16 <= b.len() {
        let clen = u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize;
        let (level, ty) = (i32::from_le_bytes(b[at + 8..at + 12].try_into().unwrap()), i32::from_le_bytes(b[at + 12..at + 16].try_into().unwrap()));
        let Some(end) = at.checked_add(clen) else { return Err(EINVAL) };
        if clen < 16 || end > b.len() {
            break;
        }
        if level == 1 && ty == 1 {
            for c in b[at + 16..end].chunks_exact(4) {
                fds.push(linux(p).files.get(i32::from_le_bytes(c.try_into().unwrap()))?);
            }
        }
        at += clen.next_multiple_of(8);
    }
    Ok(fds)
}

pub fn syscall(p: &Arc<Process>, n: u64, a: [u64; 6]) -> R {
    let files = &linux(p).files;
    let get = |fd: u64| files.get(fd as i32);
    match n {
        // socket
        41 => {
            if a[0] != 1 {
                return Err(EAFNOSUPPORT);
            }
            let ty = a[1] as u32;
            let base = ty & 0xf;
            if ![SOCK_STREAM, SOCK_DGRAM, SOCK_SEQPACKET].contains(&base) {
                return Err(EINVAL);
            }
            files.insert(File::new(Kind::Socket(Mutex::new(State::Fresh { ty: base })), ty & SOCK_NONBLOCK), ty & SOCK_CLOEXEC != 0)
        }
        // socketpair
        53 => {
            if a[0] != 1 {
                return Err(EAFNOSUPPORT);
            }
            let ty = a[1] as u32;
            let (x, y) = Endpoint::pair(ty & 0xf);
            let fx = files.insert(File::new(Kind::Stream(x), ty & SOCK_NONBLOCK), ty & SOCK_CLOEXEC != 0)?;
            let fy = files.insert(File::new(Kind::Stream(y), ty & SOCK_NONBLOCK), ty & SOCK_CLOEXEC != 0)?;
            write_user(p, a[3], &[(fx as u32).to_le_bytes(), (fy as u32).to_le_bytes()].concat()).map(|_| 0)
        }
        // bind
        49 => {
            let f = get(a[0])?;
            let path = read_path(p, a[1], a[2])?;
            let Kind::Socket(s) = &f.kind else { return Err(EINVAL) };
            let mut s = s.lock();
            let State::Fresh { ty } = *s else { return Err(EINVAL) };
            if NAMES.lock().get(&path).is_some_and(|w| w.strong_count() > 0) {
                return Err(EADDRINUSE);
            }
            *s = State::Bound { ty, path };
            Ok(0)
        }
        // listen: the fd becomes a listener in place.
        50 => {
            let f = get(a[0])?;
            let Kind::Socket(s) = &f.kind else { return if matches!(f.kind, Kind::Listener(_)) { Ok(0) } else { Err(EINVAL) } };
            let State::Bound { ty, path } = &*s.lock() else { return Err(EINVAL) };
            let l = Arc::new(Listener { ty: *ty, path: path.clone(), queue: WaitCell::new(VecDeque::new()) });
            NAMES.lock().insert(path.clone(), Arc::downgrade(&l));
            let nf = File::new(Kind::Listener(l), f.flags.load(core::sync::atomic::Ordering::Relaxed));
            files.fds_replace(a[0] as i32, nf)?;
            Ok(0)
        }
        // accept, accept4
        43 | 288 => {
            let f = get(a[0])?;
            let Kind::Listener(l) = &f.kind else { return Err(EINVAL) };
            let r = l.queue.wait(if f.nonblocking() { 0 } else { u64::MAX }, |q| q.pop_front());
            let e = match r {
                Ok(e) => e,
                Err(WaitError::TimedOut) => return Err(EAGAIN),
                Err(e) => return Err(wait_errno(e)),
            };
            // The peer is unnamed: an address of just the family.
            if a[1] != 0 && a[2] != 0 {
                write_user(p, a[1], &1u16.to_le_bytes())?;
                write_u32(p, a[2], 2)?;
            }
            let flags = if n == 288 { a[3] as u32 } else { 0 };
            files.insert(File::new(Kind::Stream(e), flags & SOCK_NONBLOCK), flags & SOCK_CLOEXEC != 0)
        }
        // connect
        42 => {
            let f = get(a[0])?;
            let path = read_path(p, a[1], a[2])?;
            let Kind::Socket(s) = &f.kind else { return Err(EINVAL) };
            let ty = match &*s.lock() {
                State::Fresh { ty } | State::Bound { ty, .. } => *ty,
            };
            let l = NAMES.lock().get(&path).and_then(|w| w.upgrade()).ok_or(ECONNREFUSED)?;
            if l.ty != ty {
                return Err(ECONNREFUSED);
            }
            let (mine, theirs) = Endpoint::pair(ty);
            l.queue.update(|q| q.push_back(theirs));
            let nf = File::new(Kind::Stream(mine), f.flags.load(core::sync::atomic::Ordering::Relaxed));
            files.fds_replace(a[0] as i32, nf)?;
            Ok(0)
        }
        // sendto, sendmsg
        44 | 46 => {
            let f = get(a[0])?;
            let Kind::Stream(e) = &f.kind else { return Err(if socket_type(&f).is_some() { ENOTCONN } else { ENOTSOCK }) };
            let (data, fds, flags) = if n == 44 {
                (read_user(p, a[1], a[2].min(16 << 20))?, Vec::new(), a[3])
            } else {
                let h = read_msghdr(p, a[1])?;
                (gather(p, &h.iov)?, rights_in(p, h.control, h.controllen)?, a[2])
            };
            e.send(&data, fds, f.nonblocking() || flags & MSG_DONTWAIT != 0).map(|n| n as u64)
        }
        // recvfrom, recvmsg
        45 | 47 => {
            let f = get(a[0])?;
            let Kind::Stream(e) = &f.kind else { return Err(if socket_type(&f).is_some() { ENOTCONN } else { ENOTSOCK }) };
            if n == 45 {
                let (data, _, full) = e.recv(a[2] as usize, a[3], f.nonblocking(), false)?;
                write_user(p, a[1], &data)?;
                if a[4] != 0 && a[5] != 0 {
                    write_u32(p, a[5], 0)?;
                }
                return Ok(if a[3] & MSG_TRUNC != 0 { full } else { data.len() } as u64);
            }
            let h = read_msghdr(p, a[1])?;
            let total: u64 = h.iov.iter().map(|v| v.1).sum();
            let (data, fds, full) = e.recv(total as usize, a[2], f.nonblocking(), false)?;
            scatter(p, &h.iov, &data)?;
            let mut out_flags = 0u32;
            if full > data.len() {
                out_flags |= MSG_TRUNC as u32;
            }
            // Control: one SCM_RIGHTS message with the received fds, if they fit.
            let mut clen_written = 0u64;
            if !fds.is_empty() {
                let need = 16 + fds.len() as u64 * 4;
                if h.control != 0 && h.controllen >= need {
                    let mut c = Vec::new();
                    c.extend_from_slice(&need.to_le_bytes());
                    c.extend_from_slice(&1i32.to_le_bytes());
                    c.extend_from_slice(&1i32.to_le_bytes());
                    for fl in fds {
                        let fd = files.insert(fl, a[2] & MSG_CMSG_CLOEXEC != 0)?;
                        c.extend_from_slice(&(fd as u32).to_le_bytes());
                    }
                    write_user(p, h.control, &c)?;
                    clen_written = need.next_multiple_of(8).min(h.controllen);
                } else {
                    out_flags |= 8; // MSG_CTRUNC: the fds are dropped
                }
            }
            write_u64(p, a[1] + 40, clen_written)?;
            write_u32(p, a[1] + 48, out_flags)?;
            write_u32(p, a[1] + 8, 0)?;
            Ok(if a[2] & MSG_TRUNC != 0 { full } else { data.len() } as u64)
        }
        // shutdown
        48 => {
            let f = get(a[0])?;
            let Kind::Stream(e) = &f.kind else { return Err(ENOTCONN) };
            if a[1] >= 1 {
                e.shutdown_write();
            }
            Ok(0)
        }
        // getsockname, getpeername: an unnamed AF_UNIX address.
        51 | 52 => {
            get(a[0])?;
            if a[1] != 0 && a[2] != 0 {
                write_user(p, a[1], &1u16.to_le_bytes())?;
                write_u32(p, a[2], 2)?;
            }
            Ok(0)
        }
        // setsockopt: accepted; getsockopt: SO_TYPE and SO_ERROR answered, the rest zero.
        54 => get(a[0]).map(|_| 0),
        55 => {
            let f = get(a[0])?;
            let v = match (a[1], a[2]) {
                (1, 3) => socket_type(&f).ok_or(ENOTSOCK)?,
                _ => 0,
            };
            write_u32(p, a[3], v)?;
            write_u32(p, a[4], 4)?;
            Ok(0)
        }
        _ => Err(ENOSYS),
    }
}
