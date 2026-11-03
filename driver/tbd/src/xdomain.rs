//! XDomain: the protocol between two hosts' connection managers on one cable (UUID exchange, lane bonding,
//! property blocks), as Linux's drivers/thunderbolt/xdomain.c speaks it (7.3).
//!
//! Each side discovers the other: it asks for the UUID, bonds the two lanes (the host with the lower UUID asks),
//! reads the other's property block, and tells it to read its own. Requests from the other host are answered by
//! the event thread (`Domain::handle_xdp`), never behind a request of our own.

use crate::ctl::{Frame, Matcher, PDF_XDOMAIN_REQ, PDF_XDOMAIN_RESP, xd_uuid};
use crate::property::Dir;
use crate::uuid::Uuid;

/// The XDomain discovery protocol: b638d70e-42ff-40bb-97c2-90e2c0b2ff07.
pub const XDP_UUID: Uuid = Uuid::init(0xb638d70e, 0x42ff, 0x40bb, [0x97, 0xc2, 0x90, 0xe2, 0xc0, 0xb2, 0xff, 0x07]);

pub const UUID_REQUEST_OLD: u32 = 1;
pub const UUID_RESPONSE: u32 = 2;
pub const PROPERTIES_REQUEST: u32 = 3;
pub const PROPERTIES_RESPONSE: u32 = 4;
pub const PROPERTIES_CHANGED_REQUEST: u32 = 5;
pub const PROPERTIES_CHANGED_RESPONSE: u32 = 6;
pub const ERROR_RESPONSE: u32 = 7;
pub const UUID_REQUEST: u32 = 12;
pub const LINK_STATE_STATUS_REQUEST: u32 = 15;
pub const LINK_STATE_STATUS_RESPONSE: u32 = 16;
pub const LINK_STATE_CHANGE_REQUEST: u32 = 17;
pub const LINK_STATE_CHANGE_RESPONSE: u32 = 18;

pub const ERROR_UNKNOWN_DOMAIN: u32 = 2;
pub const ERROR_NOT_SUPPORTED: u32 = 3;
pub const ERROR_NOT_READY: u32 = 4;

/// Dwords of property data in one response: (256 - CRC - the 72-byte response header) / 4.
pub const PROPERTIES_MAX_DATA: usize = 45;
/// Largest property block accepted.
pub const PROPERTIES_MAX_LENGTH: usize = 500;

/// HopIDs 0-7 are reserved on lane adapters; the default maximum when the peer announces none.
pub const MIN_HOPID: u32 = 8;
pub const DEFAULT_MAX_HOPID: u32 = 15;

/// The common header of every XDomain packet: route, length (dwords after these three) and sequence number, then
/// the protocol UUID. `words` is the whole packet, sized by the caller; the header is filled in.
pub fn fill_header(words: &mut [u32], route: u64, seq: u32, uuid: Uuid) {
    words[0] = (route >> 32) as u32;
    words[1] = route as u32;
    words[2] = (words.len() as u32 - 3) & 0x3f | (seq & 3) << 27;
    words[3..7].copy_from_slice(&uuid.words());
}

pub fn seq_of(f: &Frame) -> u32 {
    (f.words[2] >> 27) & 3
}

/// An XDP packet of `len` dwords with its type.
pub fn xdp(len: usize, route: u64, seq: u32, ty: u32) -> Vec<u32> {
    let mut w = vec![0u32; len];
    fill_header(&mut w, route, seq, XDP_UUID);
    w[7] = ty;
    w
}

pub fn xdp_type(f: &Frame) -> Option<u32> {
    (xd_uuid(f) == Some(XDP_UUID) && f.words.len() >= 8).then(|| f.words[7])
}

/// Matches the response to an XDP request: same route, protocol, sequence number, and the expected type or an
/// error response.
pub fn xdp_matcher(route: u64, seq: u32, response: u32) -> Matcher {
    Box::new(move |f: &Frame| {
        f.pdf == PDF_XDOMAIN_RESP
            && f.route() == route
            && seq_of(f) == seq
            && matches!(xdp_type(f), Some(t) if t == response || t == ERROR_RESPONSE)
    })
}

/// Maps an XDP error response to what Linux returns for it (`tb_xdp_handle_error`).
pub fn xdp_error(f: &Frame) -> Option<u32> {
    (xdp_type(f) == Some(ERROR_RESPONSE) && f.words.len() >= 9).then(|| f.words[8])
}

pub fn uuid_request(route: u64, seq: u32) -> Frame {
    Frame { pdf: PDF_XDOMAIN_REQ, words: xdp(8, route, seq, UUID_REQUEST) }
}

pub fn uuid_response(route: u64, seq: u32, local: Uuid) -> Frame {
    let mut w = xdp(14, route, seq, UUID_RESPONSE);
    w[8..12].copy_from_slice(&local.words());
    w[12] = (route >> 32) as u32;
    w[13] = route as u32;
    Frame { pdf: PDF_XDOMAIN_RESP, words: w }
}

pub fn error_response(route: u64, seq: u32, error: u32) -> Frame {
    let mut w = xdp(9, route, seq, ERROR_RESPONSE);
    w[8] = error;
    Frame { pdf: PDF_XDOMAIN_RESP, words: w }
}

pub fn properties_request(route: u64, seq: u32, src: Uuid, dst: Uuid, offset: u16) -> Frame {
    let mut w = xdp(17, route, seq, PROPERTIES_REQUEST);
    w[8..12].copy_from_slice(&src.words());
    w[12..16].copy_from_slice(&dst.words());
    w[16] = offset as u32;
    Frame { pdf: PDF_XDOMAIN_REQ, words: w }
}

/// One page of our property block (`tb_xdp_properties_response`).
pub fn properties_response(route: u64, seq: u32, src: Uuid, dst: Uuid, offset: usize, block: &[u32], generation: u32) -> Frame {
    let len = (block.len() - offset).min(PROPERTIES_MAX_DATA);
    let mut w = xdp(18 + len, route, seq, PROPERTIES_RESPONSE);
    w[8..12].copy_from_slice(&src.words());
    w[12..16].copy_from_slice(&dst.words());
    w[16] = offset as u32 | (block.len() as u32) << 16;
    w[17] = generation;
    w[18..].copy_from_slice(&block[offset..offset + len]);
    Frame { pdf: PDF_XDOMAIN_RESP, words: w }
}

/// A properties response page: (offset, total length, generation, data).
pub fn parse_properties_response(f: &Frame) -> Option<(usize, usize, u32, &[u32])> {
    if f.words.len() < 18 {
        return None;
    }
    let len = (f.words[2] & 0x3f) as usize + 3;
    if len != f.words.len() {
        return None;
    }
    Some(((f.words[16] & 0xffff) as usize, (f.words[16] >> 16) as usize, f.words[17], &f.words[18..]))
}

pub fn properties_changed_request(route: u64, seq: u32, src: Uuid) -> Frame {
    let mut w = xdp(12, route, seq, PROPERTIES_CHANGED_REQUEST);
    w[8..12].copy_from_slice(&src.words());
    Frame { pdf: PDF_XDOMAIN_REQ, words: w }
}

pub fn properties_changed_response(route: u64, seq: u32) -> Frame {
    Frame { pdf: PDF_XDOMAIN_RESP, words: xdp(8, route, seq, PROPERTIES_CHANGED_RESPONSE) }
}

pub fn link_state_status_request(route: u64, seq: u32) -> Frame {
    Frame { pdf: PDF_XDOMAIN_REQ, words: xdp(8, route, seq, LINK_STATE_STATUS_REQUEST) }
}

/// Supported and target lane width and speed, as `LANE_ADP_CS_0/1` encode them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinkState {
    pub slw: u8,
    pub tlw: u8,
    pub sls: u8,
    pub tls: u8,
}

pub fn link_state_status_response(route: u64, seq: u32, s: LinkState) -> Frame {
    let mut w = xdp(10, route, seq, LINK_STATE_STATUS_RESPONSE);
    w[9] = s.slw as u32 | (s.tlw as u32) << 8 | (s.sls as u32) << 16 | (s.tls as u32) << 24;
    Frame { pdf: PDF_XDOMAIN_RESP, words: w }
}

pub fn parse_link_state_status(f: &Frame) -> Option<(u32, LinkState)> {
    (f.words.len() >= 10).then(|| {
        let v = f.words[9];
        (f.words[8], LinkState { slw: v as u8, tlw: (v >> 8) as u8, sls: (v >> 16) as u8, tls: (v >> 24) as u8 })
    })
}

pub fn link_state_change_request(route: u64, seq: u32, tlw: u8, tls: u8) -> Frame {
    let mut w = xdp(9, route, seq, LINK_STATE_CHANGE_REQUEST);
    w[8] = tlw as u32 | (tls as u32) << 8;
    Frame { pdf: PDF_XDOMAIN_REQ, words: w }
}

pub fn link_state_change_response(route: u64, seq: u32, status: u32) -> Frame {
    let mut w = xdp(9, route, seq, LINK_STATE_CHANGE_RESPONSE);
    w[8] = status;
    Frame { pdf: PDF_XDOMAIN_RESP, words: w }
}

/// What the other host published, the parts we use.
#[derive(Clone, Debug)]
pub struct Remote {
    pub vendor: u32,
    pub device: u32,
    pub vendor_name: Option<String>,
    pub device_name: Option<String>,
    pub max_hopid: u32,
    pub generation: u32,
    pub dir: Dir,
}

impl Remote {
    /// `populate_properties`: vendorid and deviceid are required.
    pub fn from_dir(dir: Dir, generation: u32) -> Option<Remote> {
        Some(Remote {
            vendor: dir.immediate("vendorid")?,
            device: dir.immediate("deviceid")?,
            vendor_name: dir.text("vendorid").map(str::to_owned),
            device_name: dir.text("deviceid").map(str::to_owned),
            max_hopid: dir.immediate("maxhopid").unwrap_or(DEFAULT_MAX_HOPID),
            generation,
            dir,
        })
    }

    /// A service directory with this key and protocol ID.
    pub fn service(&self, key: &str, prtcid: u32) -> Option<&Dir> {
        self.dir.dir(key).filter(|d| d.immediate("prtcid") == Some(prtcid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_request_matches_linux_layout() {
        // tb_xdp_uuid: header (route, length_sn, uuid, type) = 8 dwords; length counts the 5 after route/length.
        let f = uuid_request(1, 2);
        assert_eq!(f.words[..3], [0, 1, 5 | 2 << 27]);
        assert_eq!(f.words[7], UUID_REQUEST);
        assert_eq!(xdp_type(&f), Some(UUID_REQUEST));
        let r = uuid_response(1, 2, Uuid([7; 16]));
        assert_eq!(r.words.len(), 14);
        assert_eq!(r.words[2] & 0x3f, 11);
        let mut reply = r.clone();
        reply.words[0] |= 1 << 31;
        assert!(xdp_matcher(1, 2, UUID_RESPONSE)(&reply));
        assert!(!xdp_matcher(1, 3, UUID_RESPONSE)(&reply));
    }

    #[test]
    fn properties_pages() {
        let block: Vec<u32> = (0..100).collect();
        let p = properties_response(1, 0, Uuid([1; 16]), Uuid([2; 16]), 90, &block, 77);
        let (off, total, gen_, data) = parse_properties_response(&p).unwrap();
        assert_eq!((off, total, gen_), (90, 100, 77));
        assert_eq!(data, &block[90..]);
        let p = properties_response(1, 0, Uuid([1; 16]), Uuid([2; 16]), 0, &block, 77);
        assert_eq!(p.encode().len(), 256);
    }
}
