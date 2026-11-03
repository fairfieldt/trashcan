//! A TAP device: the host's network stack sees ThunderboltIP as an Ethernet interface. Linux only; a microkernel
//! build plugs its own network stack in here.
//!
//! Every frame comes with a virtio-net header (`IFF_VNET_HDR`), and the device takes TCP segmentation and checksum
//! offload, so the kernel hands over TSO packets of up to 64 KiB with their checksums left to us, as it does to
//! thunderbolt_net (which sends such a packet whole, in 4 KiB frames).

use std::io;
use std::time::Duration;

/// `struct virtio_net_hdr`, the 10 bytes ahead of every frame.
pub const VNET_HDR: usize = 10;
pub const VNET_F_NEEDS_CSUM: u8 = 1;

pub struct Tap {
    pub name: String,
    #[cfg(target_os = "linux")]
    fd: std::os::fd::OwnedFd,
}

#[cfg(target_os = "linux")]
mod sys {
    pub const TUNSETIFF: libc::c_ulong = 0x4004_54ca;
    pub const IFF_TAP: libc::c_short = 0x0002;
    pub const IFF_NO_PI: libc::c_short = 0x1000;
    pub const IFF_VNET_HDR: libc::c_short = 0x4000;
    pub const TUNSETOFFLOAD: libc::c_ulong = 0x4004_54d0;
    pub const TUN_F_CSUM: libc::c_uint = 0x01;
    pub const TUN_F_TSO4: libc::c_uint = 0x02;
    pub const TUN_F_TSO6: libc::c_uint = 0x04;

    #[repr(C)]
    pub struct IfreqFlags {
        pub name: [u8; 16],
        pub flags: libc::c_short,
        pub pad: [u8; 22],
    }

    #[repr(C)]
    pub struct IfreqMtu {
        pub name: [u8; 16],
        pub mtu: libc::c_int,
        pub pad: [u8; 20],
    }

    #[repr(C)]
    pub struct IfreqHwaddr {
        pub name: [u8; 16],
        pub family: libc::sa_family_t,
        pub data: [u8; 14],
        pub pad: [u8; 8],
    }
}

#[cfg(target_os = "linux")]
impl Tap {
    pub fn open(name: &str, mac: [u8; 6], mtu: u32) -> anyhow::Result<Tap> {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        let os = |op: &str| anyhow::anyhow!("{op}: {}", io::Error::last_os_error());
        // SAFETY: plain syscalls on descriptors this function owns, with properly sized ifreq structs.
        unsafe {
            let fd = libc::open(c"/dev/net/tun".as_ptr(), libc::O_RDWR | libc::O_CLOEXEC);
            if fd < 0 {
                return Err(os("open /dev/net/tun"));
            }
            let fd = OwnedFd::from_raw_fd(fd);
            let mut req = sys::IfreqFlags { name: [0; 16], flags: sys::IFF_TAP | sys::IFF_NO_PI | sys::IFF_VNET_HDR, pad: [0; 22] };
            req.name[..name.len().min(15)].copy_from_slice(&name.as_bytes()[..name.len().min(15)]);
            if libc::ioctl(fd.as_raw_fd(), sys::TUNSETIFF, &mut req) < 0 {
                return Err(os("TUNSETIFF"));
            }
            let ifname = String::from_utf8_lossy(&req.name).trim_end_matches('\0').to_owned();
            let offload = sys::TUN_F_CSUM | sys::TUN_F_TSO4 | sys::TUN_F_TSO6;
            if libc::ioctl(fd.as_raw_fd(), sys::TUNSETOFFLOAD, offload as libc::c_ulong) < 0 {
                return Err(os("TUNSETOFFLOAD"));
            }
            let sock = libc::socket(libc::AF_INET, libc::SOCK_DGRAM | libc::SOCK_CLOEXEC, 0);
            if sock < 0 {
                return Err(os("socket"));
            }
            let sock = OwnedFd::from_raw_fd(sock);
            let mut hw = sys::IfreqHwaddr { name: req.name, family: libc::ARPHRD_ETHER, data: [0; 14], pad: [0; 8] };
            hw.data[..6].copy_from_slice(&mac);
            if libc::ioctl(sock.as_raw_fd(), libc::SIOCSIFHWADDR, &hw) < 0 {
                return Err(os("SIOCSIFHWADDR"));
            }
            let m = sys::IfreqMtu { name: req.name, mtu: mtu as libc::c_int, pad: [0; 20] };
            if libc::ioctl(sock.as_raw_fd(), libc::SIOCSIFMTU, &m) < 0 {
                return Err(os("SIOCSIFMTU"));
            }
            let mut up = sys::IfreqFlags { name: req.name, flags: 0, pad: [0; 22] };
            if libc::ioctl(sock.as_raw_fd(), libc::SIOCGIFFLAGS, &mut up) < 0 {
                return Err(os("SIOCGIFFLAGS"));
            }
            up.flags |= (libc::IFF_UP | libc::IFF_RUNNING) as libc::c_short;
            if libc::ioctl(sock.as_raw_fd(), libc::SIOCSIFFLAGS, &up) < 0 {
                return Err(os("SIOCSIFFLAGS"));
            }
            Ok(Tap { name: ifname, fd })
        }
    }

    /// One frame behind its virtio-net header, or 0 bytes if none arrives within `timeout`.
    pub fn read(&self, buf: &mut [u8], timeout: Duration) -> io::Result<usize> {
        use std::os::fd::AsRawFd;
        let mut p = libc::pollfd { fd: self.fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        // SAFETY: one pollfd; a read into buf of at most its length.
        unsafe {
            let r = libc::poll(&mut p, 1, timeout.as_millis() as libc::c_int);
            if r < 0 {
                let e = io::Error::last_os_error();
                return if e.kind() == io::ErrorKind::Interrupted { Ok(0) } else { Err(e) };
            }
            if r == 0 {
                return Ok(0);
            }
            let n = libc::read(self.fd.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len());
            if n < 0 { Err(io::Error::last_os_error()) } else { Ok(n as usize) }
        }
    }

    /// A frame, with an empty virtio-net header (no offload: checksums are complete, nothing to segment).
    pub fn write(&self, frame: &[u8]) -> io::Result<()> {
        self.write_parts(&[frame])
    }

    /// A frame from several pieces (the payloads of the ThunderboltIP frames it came in).
    pub fn write_parts(&self, parts: &[&[u8]]) -> io::Result<()> {
        use std::os::fd::AsRawFd;
        let hdr = [0u8; VNET_HDR];
        let mut iov = Vec::with_capacity(parts.len() + 1);
        iov.push(libc::iovec { iov_base: hdr.as_ptr() as *mut _, iov_len: hdr.len() });
        iov.extend(parts.iter().map(|p| libc::iovec { iov_base: p.as_ptr() as *mut _, iov_len: p.len() }));
        // SAFETY: a writev of buffers that outlive the call.
        let n = unsafe { libc::writev(self.fd.as_raw_fd(), iov.as_ptr(), iov.len() as libc::c_int) };
        if n < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
    }
}

#[cfg(not(target_os = "linux"))]
impl Tap {
    pub fn open(_: &str, _: [u8; 6], _: u32) -> anyhow::Result<Tap> {
        anyhow::bail!("TAP devices need Linux")
    }

    pub fn read(&self, _: &mut [u8], _: Duration) -> io::Result<usize> {
        Err(io::ErrorKind::Unsupported.into())
    }

    pub fn write(&self, _: &[u8]) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }

    pub fn write_parts(&self, _: &[&[u8]]) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }
}

/// Fills in the checksums of a frame the kernel left them to (virtio-net `NEEDS_CSUM`): the IPv4 header checksum
/// and the TCP or UDP checksum at `csum_start + csum_offset`, computed afresh over the pseudo-header and the whole
/// transport segment (`tbnet_xmit_csum_and_map`). Returns false for frames it does not understand.
pub fn finish_checksum(frame: &mut [u8], csum_start: usize, csum_offset: usize) -> bool {
    let mut l3 = 14;
    let mut ethertype = u16::from_be_bytes([frame[12], frame[13]]);
    if ethertype == 0x8100 {
        if frame.len() < 18 {
            return false;
        }
        ethertype = u16::from_be_bytes([frame[16], frame[17]]);
        l3 = 18;
    }
    let field = csum_start + csum_offset;
    if field + 2 > frame.len() || csum_start > frame.len() || csum_start < l3 {
        return false;
    }
    let seg_len = frame.len() - csum_start;
    let mut sum: u32 = 0;
    match ethertype {
        0x0800 => {
            if frame.len() < l3 + 20 {
                return false;
            }
            let ihl = (frame[l3] & 0xf) as usize * 4;
            let tot = (frame.len() - l3) as u16;
            frame[l3 + 2..l3 + 4].copy_from_slice(&tot.to_be_bytes());
            frame[l3 + 10..l3 + 12].fill(0);
            let ip = !fold(add(0, &frame[l3..l3 + ihl]));
            frame[l3 + 10..l3 + 12].copy_from_slice(&ip.to_be_bytes());
            sum = add(sum, &frame[l3 + 12..l3 + 20]);
            sum += frame[l3 + 9] as u32 + seg_len as u32;
        }
        0x86dd => {
            if frame.len() < l3 + 40 {
                return false;
            }
            let plen = (frame.len() - l3 - 40) as u16;
            frame[l3 + 4..l3 + 6].copy_from_slice(&plen.to_be_bytes());
            sum = add(sum, &frame[l3 + 8..l3 + 40]);
            // The transport protocol: the next header, unless extension headers come first (then trust the
            // transport header position and take TCP, which TSO is for).
            let next = frame[l3 + 6];
            let proto = if csum_start == l3 + 40 { next as u32 } else { 6 };
            sum += proto + (seg_len >> 16) as u32 + (seg_len & 0xffff) as u32;
        }
        _ => return false,
    }
    frame[field..field + 2].fill(0);
    sum = add(sum, &frame[csum_start..]);
    let mut c = !fold(sum);
    // UDP sends a computed 0 as all ones.
    if c == 0 && csum_offset == 6 {
        c = 0xffff;
    }
    frame[field..field + 2].copy_from_slice(&c.to_be_bytes());
    true
}

/// One's complement sum of big-endian 16-bit words, eight bytes at a time (the sum of 32-bit halves folds to the
/// same 16-bit result).
fn add(sum: u32, data: &[u8]) -> u32 {
    let mut acc = sum as u64;
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        let v = u64::from_be_bytes(c.try_into().unwrap());
        acc += (v >> 32) + (v & 0xffff_ffff);
    }
    let mut rest = chunks.remainder().chunks_exact(2);
    for c in &mut rest {
        acc += u16::from_be_bytes([c[0], c[1]]) as u64;
    }
    if let [b] = rest.remainder() {
        acc += (*b as u64) << 8;
    }
    while acc > 0xffff_ffff {
        acc = (acc & 0xffff_ffff) + (acc >> 32);
    }
    acc as u32
}

fn fold(mut sum: u32) -> u16 {
    while sum > 0xffff {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    sum as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An Ethernet/IPv4/UDP frame with known checksums (IP 0xb861, UDP 0x8a3c... computed independently below).
    fn frame() -> Vec<u8> {
        let mut f = vec![0u8; 14];
        f[12] = 0x08;
        // IPv4: 20-byte header, total length 32, TTL 64, UDP, 10.0.0.1 -> 10.0.0.2.
        f.extend_from_slice(&[0x45, 0, 0, 32, 0, 0, 0x40, 0, 64, 17, 0, 0, 10, 0, 0, 1, 10, 0, 0, 2]);
        // UDP 1234 -> 5678, length 12, payload "test".
        f.extend_from_slice(&[0x04, 0xd2, 0x16, 0x2e, 0, 12, 0, 0]);
        f.extend_from_slice(b"test");
        f
    }

    #[test]
    fn checksums_verify() {
        let mut f = frame();
        assert!(finish_checksum(&mut f, 34, 6));
        // A correct IPv4 header sums to 0xffff, and so does the UDP segment with its pseudo-header.
        assert_eq!(fold(add(0, &f[14..34])), 0xffff);
        let mut s = add(0, &f[26..34]);
        s += 17 + 12;
        s = add(s, &f[34..]);
        assert_eq!(fold(s), 0xffff);
        // Garbage in the checksum fields first gives the same result.
        let mut g = frame();
        g[24] = 0x12;
        g[40] = 0x34;
        assert!(finish_checksum(&mut g, 34, 6));
        assert_eq!(f, g);
    }
}
