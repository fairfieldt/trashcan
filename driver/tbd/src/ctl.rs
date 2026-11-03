//! The control channel: ring 0 of the NHI, carrying config space reads and writes to routers, hot plug events,
//! and the XDomain messages between hosts.
//!
//! A frame is a PDF (the packet type, in the descriptor's SOF/EOF fields) and dwords. Linux builds packets as C
//! structs on a little-endian host and sends every dword big-endian, then the CRC-32C of those bytes, big-endian
//! (ctl.c `tb_ctl_tx`). Here packets are `Vec<u32>` holding the struct's dwords, so bitfields and byte arrays land
//! where Linux puts them.

use crate::crc::crc32c;
use crate::nhi::{FRAME_SIZE, Kind, Nhi, Ring, RingConfig};
use crate::uuid::Uuid;
use anyhow::{Result, bail};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const PDF_READ: u8 = 1;
pub const PDF_WRITE: u8 = 2;
pub const PDF_ERROR: u8 = 3;
pub const PDF_NOTIFY_ACK: u8 = 4;
pub const PDF_EVENT: u8 = 5;
pub const PDF_XDOMAIN_REQ: u8 = 6;
pub const PDF_XDOMAIN_RESP: u8 = 7;

pub const SPACE_HOPS: u32 = 0;
pub const SPACE_PORT: u32 = 1;
pub const SPACE_SWITCH: u32 = 2;

pub const ERROR_INVALID_CONFIG_SPACE: u32 = 2;
pub const ERROR_ACK_PLUG_EVENT: u32 = 7;

/// Bit 63 of a route is set on replies.
pub const ROUTE_REPLY: u64 = 1 << 63;

const RETRIES: u32 = 4;
pub const TIMEOUT: Duration = Duration::from_millis(1000);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub pdf: u8,
    pub words: Vec<u32>,
}

impl Frame {
    pub fn route(&self) -> u64 {
        ((self.words[0] as u64) << 32 | self.words[1] as u64) & !ROUTE_REPLY
    }

    /// Wire bytes: big-endian dwords, then the CRC.
    pub fn encode(&self) -> Vec<u8> {
        let mut b: Vec<u8> = self.words.iter().flat_map(|w| w.to_be_bytes()).collect();
        let crc = crc32c(&b);
        b.extend_from_slice(&crc.to_be_bytes());
        b
    }

    /// From wire bytes with the CRC at the end; `None` if the size or CRC is wrong.
    pub fn decode(pdf: u8, bytes: &[u8]) -> Option<Frame> {
        if bytes.len() < 12 || bytes.len() % 4 != 0 {
            return None;
        }
        let (data, crc) = bytes.split_at(bytes.len() - 4);
        if crc32c(data) != u32::from_be_bytes(crc.try_into().unwrap()) {
            return None;
        }
        Some(Frame { pdf, words: data.chunks(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect() })
    }
}

/// The two route dwords of a config packet header (`struct tb_cfg_header`: 22 bits of route_hi).
pub fn cfg_header(route: u64) -> [u32; 2] {
    [((route >> 32) as u32) & 0x3f_ffff, route as u32]
}

/// `struct tb_cfg_address`.
pub fn cfg_address(offset: u32, length: u32, port: u32, space: u32, seq: u32) -> u32 {
    (offset & 0x1fff) | (length & 0x3f) << 13 | (port & 0x3f) << 19 | (space & 3) << 25 | (seq & 3) << 27
}

/// What a request waits for.
pub type Matcher = Box<dyn Fn(&Frame) -> bool + Send>;

struct Pending {
    id: u64,
    matcher: Matcher,
    reply: Sender<Frame>,
}

/// A packet nobody asked for: plug events, error notifications and XDomain messages.
pub type Event = Frame;

pub struct Ctl {
    tx: Mutex<Ring>,
    pending: Mutex<Vec<Pending>>,
    next_id: AtomicU64,
}

/// Errors from routers.
#[derive(Debug)]
pub enum CfgError {
    Timeout,
    /// TB_CFG_PKG_ERROR from a router: (error code, port).
    Router(u32, u32),
    Malformed(String),
}

impl std::fmt::Display for CfgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CfgError::Timeout => write!(f, "timed out"),
            CfgError::Router(e, p) => write!(f, "router error {e} on port {p}"),
            CfgError::Malformed(s) => write!(f, "malformed reply: {s}"),
        }
    }
}

impl std::error::Error for CfgError {}

impl Ctl {
    /// Starts ring 0 both ways and the receive thread; unsolicited packets come out of the returned channel. The
    /// classifier decides which packets are events (the rest go to waiting requests).
    pub fn start(nhi: &Arc<Nhi>, is_event: fn(&Frame) -> bool) -> Result<(Arc<Ctl>, Receiver<Event>)> {
        let raw = RingConfig {
            hop: Some(0),
            size: 10,
            frame: false,
            e2e: false,
            sof_mask: 0xffff,
            eof_mask: 0xffff,
            interrupt: true,
        };
        let tx = nhi.ring(Kind::Tx, raw)?;
        let rx = nhi.ring(Kind::Rx, raw)?;
        let ctl = Arc::new(Ctl { tx: Mutex::new(tx), pending: Mutex::new(Vec::new()), next_id: AtomicU64::new(0) });
        let (events_tx, events) = mpsc::channel();
        let c = ctl.clone();
        std::thread::Builder::new().name("tbd-ctl-rx".into()).spawn(move || c.rx_loop(rx, events_tx, is_event))?;
        Ok((ctl, events))
    }

    fn rx_loop(&self, mut rx: Ring, events: Sender<Event>, is_event: fn(&Frame) -> bool) {
        loop {
            while let Some(f) = rx.rx_peek() {
                let frame = Frame::decode(f.eof, &rx.rx_data(&f)[..f.len]);
                rx.rx_release();
                let Some(frame) = frame else {
                    eprintln!("tbd: control RX: bad frame (PDF {}, {} bytes, flags {:#x}), dropped", f.eof, f.len, f.flags);
                    continue;
                };
                if trace() {
                    eprintln!("tbd: ctl rx pdf {} {:08x?}", frame.pdf, frame.words);
                }
                if is_event(&frame) {
                    if events.send(frame).is_err() {
                        return;
                    }
                    continue;
                }
                let mut pending = self.pending.lock().unwrap();
                if let Some(i) = pending.iter().position(|p| (p.matcher)(&frame)) {
                    let p = pending.swap_remove(i);
                    let _ = p.reply.send(frame);
                } else if trace() {
                    eprintln!("tbd: ctl rx: no request for pdf {} route {:#x}", frame.pdf, frame.route());
                }
            }
            rx.wait(Duration::from_millis(10));
        }
    }

    /// Sends a frame without waiting for anything.
    pub fn send(&self, frame: &Frame) -> Result<()> {
        let bytes = frame.encode();
        if bytes.len() > FRAME_SIZE {
            bail!("control frame of {} bytes", bytes.len());
        }
        if trace() {
            eprintln!("tbd: ctl tx pdf {} {:08x?}", frame.pdf, frame.words);
        }
        let deadline = Instant::now() + TIMEOUT;
        let mut tx = self.tx.lock().unwrap();
        loop {
            tx.tx_reap();
            if let Some(slot) = tx.tx_slot() {
                slot[..bytes.len()].copy_from_slice(&bytes);
                break;
            }
            if Instant::now() > deadline {
                bail!("control TX ring stuck full");
            }
            tx.wait(Duration::from_millis(1));
        }
        tx.tx_submit(bytes.len(), frame.pdf, frame.pdf);
        Ok(())
    }

    /// Sends `frame` and waits for the first packet `matcher` accepts.
    pub fn request(&self, frame: &Frame, matcher: Matcher, timeout: Duration) -> Result<Frame, CfgError> {
        let (reply, rx) = mpsc::channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.pending.lock().unwrap().push(Pending { id, matcher, reply });
        if let Err(e) = self.send(frame) {
            self.forget(id);
            return Err(CfgError::Malformed(e.to_string()));
        }
        match rx.recv_timeout(timeout) {
            Ok(f) => Ok(f),
            Err(_) => {
                self.forget(id);
                // A reply may have raced in between the timeout and the removal.
                rx.try_recv().map_err(|_| CfgError::Timeout)
            }
        }
    }

    fn forget(&self, id: u64) {
        self.pending.lock().unwrap().retain(|p| p.id != id);
    }

    /// Config space read (`tb_cfg_read_raw`, with its retries on timeout). Returns the data and the port the
    /// router answered from (its upstream port).
    pub fn read(&self, route: u64, port: u32, space: u32, offset: u32, len: u32) -> Result<(Vec<u32>, u32), CfgError> {
        assert!((1..=60).contains(&len), "config read of {len} dwords");
        for seq in 0..RETRIES {
            let [h0, h1] = cfg_header(route);
            let req = Frame { pdf: PDF_READ, words: vec![h0, h1, cfg_address(offset, len, port, space, seq)] };
            let m = cfg_matcher(PDF_READ, route, seq, 3 + len as usize);
            match self.request(&req, m, TIMEOUT) {
                Ok(f) => return parse_cfg_reply(&f, space, offset, len).map(|p| (f.words[3..].to_vec(), p)),
                Err(CfgError::Timeout) => std::thread::sleep(Duration::from_micros(50)),
                Err(e) => return Err(e),
            }
        }
        Err(CfgError::Timeout)
    }

    pub fn write(&self, route: u64, port: u32, space: u32, offset: u32, data: &[u32]) -> Result<(), CfgError> {
        let len = data.len() as u32;
        assert!((1..=60).contains(&len), "config write of {len} dwords");
        for seq in 0..RETRIES {
            let [h0, h1] = cfg_header(route);
            let mut words = vec![h0, h1, cfg_address(offset, len, port, space, seq)];
            words.extend_from_slice(data);
            let req = Frame { pdf: PDF_WRITE, words };
            let m = cfg_matcher(PDF_WRITE, route, seq, 3);
            match self.request(&req, m, TIMEOUT) {
                Ok(f) => return parse_cfg_reply(&f, space, offset, len).map(|_| ()),
                Err(CfgError::Timeout) => std::thread::sleep(Duration::from_micros(50)),
                Err(e) => return Err(e),
            }
        }
        Err(CfgError::Timeout)
    }

    /// Acknowledges a hot plug event (`tb_cfg_ack_plug`).
    pub fn ack_plug(&self, route: u64, port: u32, unplug: bool) -> Result<()> {
        let [h0, h1] = cfg_header(route);
        let pg = if unplug { 3 } else { 2 };
        self.send(&Frame { pdf: PDF_ERROR, words: vec![h0, h1, ERROR_ACK_PLUG_EVENT | (port & 0x3f) << 8 | pg << 30] })
    }

    /// Acknowledges an error notification (`tb_cfg_ack_notification`).
    pub fn ack_notification(&self, route: u64) -> Result<()> {
        let [h0, h1] = cfg_header(route);
        self.send(&Frame { pdf: PDF_NOTIFY_ACK, words: vec![h0, h1] })
    }
}

/// Matches the reply to a read or write: same PDF, route, sequence number and size; or an error from that route.
fn cfg_matcher(pdf: u8, route: u64, seq: u32, words: usize) -> Matcher {
    Box::new(move |f: &Frame| {
        if f.words.len() < 3 || f.route() != route {
            return false;
        }
        if f.pdf == PDF_ERROR {
            return true;
        }
        f.pdf == pdf && f.words.len() == words && (f.words[2] >> 27) & 3 == seq
    })
}

fn parse_cfg_reply(f: &Frame, space: u32, offset: u32, len: u32) -> Result<u32, CfgError> {
    if f.pdf == PDF_ERROR {
        return Err(CfgError::Router(f.words[2] & 0xff, (f.words[2] >> 8) & 0x3f));
    }
    if f.words[0] >> 22 != 1 << 9 {
        return Err(CfgError::Malformed(format!("header {:#x}", f.words[0])));
    }
    let a = f.words[2];
    let (r_off, r_len, r_space) = (a & 0x1fff, (a >> 13) & 0x3f, (a >> 25) & 3);
    if r_off != offset || r_len != len || r_space != space || a >> 29 != 0 {
        return Err(CfgError::Malformed(format!("address {a:#x} for space {space} offset {offset:#x} length {len}")));
    }
    Ok((a >> 19) & 0x3f)
}

/// XDomain packets start with route, length/sequence and the protocol UUID (`struct tb_xdp_header`, and the
/// ThunderboltIP header that shares it).
pub fn xd_uuid(f: &Frame) -> Option<Uuid> {
    (f.words.len() >= 7).then(|| Uuid::from_words(&f.words[3..7]))
}

pub fn trace() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var_os("TBD_TRACE").is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_request_layout() {
        // Read 5 dwords of switch space at route 0: header, then offset 0 | len 5 << 13 | space 2 << 25.
        let f = Frame { pdf: PDF_READ, words: vec![0, 0, cfg_address(0, 5, 0, SPACE_SWITCH, 1)] };
        assert_eq!(f.words[2], 5 << 13 | 2 << 25 | 1 << 27);
        let bytes = f.encode();
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[8..12], &(5u32 << 13 | 2 << 25 | 1 << 27).to_be_bytes());
        assert_eq!(Frame::decode(PDF_READ, &bytes).unwrap(), f);
        let mut bad = bytes.clone();
        bad[0] ^= 1;
        assert!(Frame::decode(PDF_READ, &bad).is_none());
    }

    #[test]
    fn reply_parsing() {
        let reply = Frame {
            pdf: PDF_READ,
            words: vec![1 << 31, 0, cfg_address(0, 2, 5, SPACE_SWITCH, 0), 0x156d8086, 0x30505],
        };
        assert!(cfg_matcher(PDF_READ, 0, 0, 5)(&reply));
        assert!(!cfg_matcher(PDF_READ, 0, 1, 5)(&reply));
        assert_eq!(parse_cfg_reply(&reply, SPACE_SWITCH, 0, 2).unwrap(), 5);
        let err = Frame { pdf: PDF_ERROR, words: vec![1 << 31, 0, ERROR_INVALID_CONFIG_SPACE | 3 << 8] };
        assert!(cfg_matcher(PDF_READ, 0, 2, 5)(&err));
        assert!(matches!(parse_cfg_reply(&err, 1, 0, 8), Err(CfgError::Router(2, 3))));
    }
}
