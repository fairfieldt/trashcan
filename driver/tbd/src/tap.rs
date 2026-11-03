//! A TAP device: the host's network stack sees ThunderboltIP as an Ethernet interface. Linux only; a microkernel
//! build plugs its own network stack in here.

use std::io;
use std::time::Duration;

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
            let mut req = sys::IfreqFlags { name: [0; 16], flags: sys::IFF_TAP | sys::IFF_NO_PI, pad: [0; 22] };
            req.name[..name.len().min(15)].copy_from_slice(&name.as_bytes()[..name.len().min(15)]);
            if libc::ioctl(fd.as_raw_fd(), sys::TUNSETIFF, &mut req) < 0 {
                return Err(os("TUNSETIFF"));
            }
            let ifname = String::from_utf8_lossy(&req.name).trim_end_matches('\0').to_owned();
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

    /// One frame, or 0 bytes if none arrives within `timeout`.
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

    pub fn write(&self, frame: &[u8]) -> io::Result<()> {
        use std::os::fd::AsRawFd;
        // SAFETY: a write of frame's bytes.
        let n = unsafe { libc::write(self.fd.as_raw_fd(), frame.as_ptr().cast(), frame.len()) };
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
}
