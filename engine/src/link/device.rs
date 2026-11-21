//! USB4STREAM devices: finding the stream of the cable that carries a TCP connection, opening it for one link at a
//! time, and closing it off the caller's thread.
//!
//! What the driver (drivers/thunderbolt/stream.c with kernel/patches-7.3/0005) does and the link relies on:
//! - a write is cut into frames of up to 4 KiB; every frame takes one slot of the other side's receive ring (256 by
//!   default), and with no Tx flow control (0005) a frame that finds the ring full is lost or arrives damaged;
//! - frames flagged with a CRC error or a buffer overrun are passed to the reader like good ones;
//! - `read` returns the bytes of consecutive frames back to back;
//! - a close sends a CLOSE frame, which the other side reads once as end of file, and then waits up to 0.5 s for the
//!   receive ring to drain, which it never does, so every close takes at least half a second;
//! - an fd kept open while the XDomain connection goes away and comes back stays valid but carries nothing;
//! - with `busy_poll` set in the stream's configfs group the rings run without interrupts: `read` and `write` look
//!   at the rings themselves (a frame is read as soon as the controller has completed it, without the interrupt and
//!   the work item that otherwise come first), and `poll()` on the device only returns an error, so a link on such
//!   a device sleeps in short naps instead (`Frames::fd` is none).
//!
//! `tblink` (nodes/macpro) creates the stream `trashcan` with HopIDs 15/15 on every link to another node.
//!
//! With tbd (driver/tbd, the userspace Thunderbolt driver) owning a controller instead, its links are TAP interfaces:
//! tbd says which peer each one leads to in `/run/tbd/net/IFACE`, and offers the stream's rings on
//! `/run/tbd/PEER.trashcan.ring`. Opening that runs the stream's rings in this process (`tbd_client::Stream`), with
//! the same read and write behaviour as the device, and closing it sends the CLOSE.
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{self, IoSlice, Read, Write},
    net::SocketAddr,
    os::unix::{fs::OpenOptionsExt, io::AsRawFd},
    path::{Path, PathBuf},
};

/// The stream name under each XDomain service's configfs group (at most 8 characters: an XDomain property key).
pub const STREAM_NAME: &str = "trashcan";

/// A frame device: a USB4STREAM fd, or the simulator in tests.
pub trait Frames: Send {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize>;
    fn writev(&mut self, bufs: &[IoSlice]) -> io::Result<usize>;
    /// For poll(): none when nothing wakes a sleeper on this device (busy-polled rings, the simulator).
    fn fd(&self) -> Option<i32> {
        None
    }
    /// How much later than its arrival a frame may become readable on this device without being held back by the
    /// cable: nothing where `read` looks at the ring itself, the interrupt and the work item that complete a frame
    /// on an interrupt-driven stream device (over 100 us for one frame in ten after 200 us of quiet on the Mac
    /// Pros, whose idle cores sleep).
    fn late(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}

/// A stream device and whether its rings are busy-polled.
pub struct Device(File, bool);

impl Frames for Device {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
    fn writev(&mut self, bufs: &[IoSlice]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }
    fn fd(&self) -> Option<i32> {
        (!self.1).then(|| self.0.as_raw_fd())
    }
    fn late(&self) -> std::time::Duration {
        if self.1 {
            std::time::Duration::ZERO
        } else {
            std::time::Duration::from_millis(1)
        }
    }
}

/// The stream of one cable as one end sees it.
#[derive(Clone, Debug)]
pub struct Found {
    /// What `Streams::open` opens (a device path).
    pub path: String,
    /// The receive ring size, in frames.
    pub ring: u32,
    /// UUIDs of this end's controller and of the other end's (`unique_id`), so both ends can check that they found
    /// the same cable.
    pub local: String,
    pub peer: String,
    /// For messages, e.g. `0-3.1/trashcan /dev/tbstream2`.
    pub label: String,
    /// The stream's rings are busy-polled (`busy_poll` in its configfs group).
    pub polled: bool,
}

/// Where a link finds and opens its stream.
pub trait Streams: Send + Sync {
    /// The stream of the cable whose network interface holds `local`. Err("") when that is not a Thunderbolt
    /// link, else why it has no usable stream.
    fn find(&self, local: &SocketAddr) -> Result<Found, String>;
    fn open(&self, found: &Found) -> io::Result<Box<dyn Frames>>;
}

/// The node's streams, from sysfs and configfs (`tblink`).
pub struct Sysfs;

fn read_trim(p: &Path) -> Option<String> {
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_owned())
}

/// The network interface that holds `addr`.
#[cfg(target_os = "linux")]
fn interface_of(addr: &std::net::IpAddr) -> Option<String> {
    let std::net::IpAddr::V4(v4) = addr else { return None };
    let mut list: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: getifaddrs fills `list`, which is walked read-only and freed once.
    unsafe {
        if libc::getifaddrs(&mut list) != 0 {
            return None;
        }
        let mut found = None;
        let mut p = list;
        while !p.is_null() {
            let a = (*p).ifa_addr;
            if !a.is_null() && (*a).sa_family as i32 == libc::AF_INET {
                let sin = &*(a as *const libc::sockaddr_in);
                if u32::from_be(sin.sin_addr.s_addr) == u32::from(*v4) {
                    found = Some(std::ffi::CStr::from_ptr((*p).ifa_name).to_string_lossy().into_owned());
                    break;
                }
            }
            p = (*p).ifa_next;
        }
        libc::freeifaddrs(list);
        found
    }
}

#[cfg(not(target_os = "linux"))]
fn interface_of(_: &std::net::IpAddr) -> Option<String> {
    None
}

impl Sysfs {
    /// The stream of the XDomain connection behind network interface `iface`, or why there is none.
    pub fn of_interface(iface: &str) -> Result<Found, String> {
        let service = std::fs::canonicalize(format!("/sys/class/net/{iface}/device")).map_err(|e| format!("{iface}: {e}"))?;
        let xd_dir = service.parent().ok_or("no XDomain")?;
        let xd = xd_dir.file_name().and_then(|n| n.to_str()).ok_or("no XDomain")?.to_owned();
        let (domain, _) = xd.split_once('-').ok_or_else(|| format!("{iface} is not on a Thunderbolt XDomain ({xd})"))?;
        let peer = read_trim(&xd_dir.join("unique_id")).ok_or("no peer UUID")?;
        let local = read_trim(Path::new(&format!("/sys/bus/thunderbolt/devices/{domain}-0/unique_id"))).ok_or("no host UUID")?;
        // The stream service's index (.0 or .1) follows discovery order.
        let mut svc = None;
        for e in std::fs::read_dir(xd_dir).map_err(|e| e.to_string())?.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with(&format!("{xd}.")) && read_trim(&e.path().join("key")).as_deref() == Some("stream") {
                svc = Some(name);
            }
        }
        let svc = svc.ok_or_else(|| format!("{xd} has no stream service (thunderbolt_stream not loaded on both ends?)"))?;
        let cfg = PathBuf::from(format!("/sys/kernel/config/thunderbolt/stream/{svc}/{STREAM_NAME}"));
        let num = |a: &str| read_trim(&cfg.join(a)).and_then(|s| s.parse::<i64>().ok());
        let index = num("index").ok_or_else(|| format!("no stream {svc}/{STREAM_NAME} (tblink creates it)"))?;
        let (inh, outh) = (num("in_hopid").unwrap_or(0), num("out_hopid").unwrap_or(0));
        if inh <= 0 || outh <= 0 {
            return Err(format!("stream {svc}/{STREAM_NAME} has no HopIDs ({inh}, {outh})"));
        }
        let ring = num("ring_size").unwrap_or(256) as u32;
        let polled = num("busy_poll") == Some(1);
        let path = format!("/dev/tbstream{index}");
        if !Path::new(&path).exists() {
            return Err(format!("{path} is missing"));
        }
        let how = if polled { ", busy-polled" } else { "" };
        Ok(Found { label: format!("{svc}/{STREAM_NAME} {path}{how}"), path, ring, local, peer, polled })
    }
}

/// Where tbd keeps its sockets and status files.
pub const TBD_DIR: &str = "/run/tbd";

impl Sysfs {
    /// The tbd stream of the link behind `iface` (a tbd TAP interface), or why there is none.
    pub fn of_tbd(iface: &str) -> Result<Found, String> {
        let kv = |p: &Path| -> Option<BTreeMap<String, String>> {
            let s = std::fs::read_to_string(p).ok()?;
            Some(s.lines().filter_map(|l| l.split_once('=')).map(|(k, v)| (k.to_owned(), v.trim().to_owned())).collect())
        };
        let net = kv(&Path::new(TBD_DIR).join("net").join(iface)).ok_or("not a tbd interface")?;
        let peer = net.get("peer").ok_or("tbd names no peer")?;
        let info_path = Path::new(TBD_DIR).join(format!("{peer}.{STREAM_NAME}.info"));
        let info = kv(&info_path).ok_or_else(|| format!("no tbd stream {peer}.{STREAM_NAME} (tbd run --stream {STREAM_NAME}:15:15)"))?;
        let path = Path::new(TBD_DIR).join(format!("{peer}.{STREAM_NAME}.ring"));
        if !path.exists() {
            return Err(format!("{} is missing", path.display()));
        }
        Ok(Found {
            label: format!("tbd {peer}.{STREAM_NAME}.ring"),
            path: path.to_string_lossy().into_owned(),
            ring: info.get("ring").and_then(|r| r.parse().ok()).unwrap_or(256),
            local: info.get("local").cloned().unwrap_or_default(),
            peer: info.get("peer").cloned().unwrap_or_default(),
            polled: false,
        })
    }
}

impl Streams for Sysfs {
    fn find(&self, local: &SocketAddr) -> Result<Found, String> {
        match interface_of(&local.ip()) {
            Some(iface) if iface.starts_with("thunderbolt") => Self::of_interface(&iface).map_err(|why| format!("{iface}: {why}")),
            Some(iface) if Path::new(TBD_DIR).join("net").join(&iface).exists() => {
                Self::of_tbd(&iface).map_err(|why| format!("{iface}: {why}"))
            }
            _ => Err(String::new()),
        }
    }

    fn open(&self, found: &Found) -> io::Result<Box<dyn Frames>> {
        if found.path.ends_with(".ring") {
            return open_tbd(Path::new(&found.path));
        }
        Ok(Box::new(open_device(Path::new(&found.path), found.polled)?))
    }
}

/// A stream whose rings tbd handed to this process.
#[cfg(target_os = "linux")]
pub struct TbdStream(tbd_client::Stream);

#[cfg(target_os = "linux")]
impl Frames for TbdStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
    fn writev(&mut self, bufs: &[IoSlice]) -> io::Result<usize> {
        self.0.writev(bufs)
    }
    fn fd(&self) -> Option<i32> {
        Some(self.0.fd())
    }
}

/// Takes over a tbd stream (one process at a time: tbd turns others away at once) and drops what it already holds.
#[cfg(target_os = "linux")]
pub fn open_tbd(path: &Path) -> io::Result<Box<dyn Frames>> {
    let s = tbd_client::Stream::connect(path).map_err(|e| {
        if e.to_string().contains("in use") {
            io::Error::new(io::ErrorKind::AddrInUse, format!("{} is in use by another process", path.display()))
        } else {
            e
        }
    })?;
    let mut d = TbdStream(s);
    drain(&mut d);
    Ok(Box::new(d))
}

#[cfg(not(target_os = "linux"))]
pub fn open_tbd(path: &Path) -> io::Result<Box<dyn Frames>> {
    Err(io::Error::new(io::ErrorKind::Unsupported, format!("{}: tbd streams need Linux", path.display())))
}

/// Open a stream device for this process alone (an exclusive `flock`: a second process on the same cable is told
/// so at once and keeps to TCP) and drop whatever its receive ring already holds.
pub fn open_device(path: &Path, polled: bool) -> io::Result<Device> {
    let f = OpenOptions::new().read(true).write(true).custom_flags(libc::O_NONBLOCK).open(path)?;
    // SAFETY: flock on an fd this function owns.
    if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let e = io::Error::last_os_error();
        return Err(if e.kind() == io::ErrorKind::WouldBlock {
            io::Error::new(io::ErrorKind::AddrInUse, format!("{} is in use by another process", path.display()))
        } else {
            e
        });
    }
    let mut d = Device(f, polled);
    drain(&mut d);
    Ok(d)
}

/// Read and drop what a newly opened device holds (frames and CLOSEs of earlier sessions), at most 64 MiB.
pub fn drain(d: &mut dyn Frames) {
    let mut buf = vec![0u8; 1 << 20];
    for _ in 0..64 {
        if d.read(&mut buf).is_err() {
            break;
        }
    }
}

/// Close a device on a thread of its own: the driver's close waits at least 0.5 s.
pub fn close_later(d: Box<dyn Frames>) {
    let _ = std::thread::Builder::new().name("link-close".into()).spawn(move || drop(d));
}
