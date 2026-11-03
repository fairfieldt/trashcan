//! tbd: a userspace Thunderbolt 2 driver for the Mac Pro 6,1. It owns one Falcon Ridge controller through VFIO and
//! connects to other hosts on its cables: ThunderboltIP networking (a TAP device) and USB4STREAM streams, both
//! compatible with Linux's thunderbolt_net and thunderbolt_stream on the other end. See driver/tbd/README.md.

// Off Linux only the protocol code and its tests build.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(target_os = "linux")]
mod bench;
mod crc;
mod ctl;
#[cfg(target_os = "linux")]
mod domain;
#[cfg(target_os = "linux")]
mod net;
mod nhi;
mod property;
#[cfg(target_os = "linux")]
mod stream;
mod switch;
#[cfg(target_os = "linux")]
mod tap;
mod uuid;
mod xdomain;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Userspace Thunderbolt 2 driver: ThunderboltIP and USB4STREAM to other hosts")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Open an NHI bound to vfio-pci, start the control channel and read the host router: ports, link states,
    /// UUID, routed hops. Changes nothing on the router.
    Probe {
        /// PCI address of the NHI, e.g. 0000:16:00.0
        address: String,
    },
    /// Run the driver: discover the hosts on the cables, then serve ThunderboltIP and streams to them.
    Run {
        /// PCI address of the NHI, e.g. 0000:16:00.0
        address: String,
        /// The name other hosts see (default: this host's name).
        #[arg(long)]
        hostname: Option<String>,
        /// TAP interface for ThunderboltIP; none turns networking off.
        #[arg(long, default_value = "tbd%d")]
        net: String,
        #[arg(long, default_value_t = 1500)]
        mtu: u32,
        /// Streams to offer, NAME[:IN_HOPID:OUT_HOPID] (default HopIDs: the first free ones).
        #[arg(long = "stream")]
        streams: Vec<String>,
        #[arg(long, default_value_t = 256)]
        ring: u32,
        /// Spin on the rings (and the stream sockets and TAP device) instead of sleeping: lower latency, a busy
        /// core per ring.
        #[arg(long)]
        busy_poll: bool,
        /// Directory for the stream sockets.
        #[arg(long, default_value = "/run/tbd")]
        socket_dir: PathBuf,
        /// Keep the links as they are at start (by default they go down and up once, so the other hosts start
        /// afresh with tbd instead of the driver that ran before).
        #[arg(long)]
        no_bounce: bool,
        /// Keep each link on one lane.
        #[arg(long)]
        no_bonding: bool,
    },
    /// Connect to a stream socket: standard input goes out as DATA frames (then a CLOSE), received frames to
    /// standard output until the other host's CLOSE.
    Cat {
        socket: PathBuf,
    },
    /// Echo everything read from a stream (a tbd socket or a Linux /dev/tbstreamN) back to it, until a CLOSE;
    /// then open it again.
    Echo {
        path: PathBuf,
        /// Stop after this many sessions (0: never).
        #[arg(long, default_value_t = 0)]
        sessions: u32,
    },
    /// Send bytes through an echo on the other host and check every one that comes back.
    Bench {
        path: PathBuf,
        #[arg(long, default_value_t = 1 << 30)]
        bytes: u64,
        /// Bytes unanswered at most; more than the other side's receive ring (256 frames of 4 KiB) loses frames.
        #[arg(long, default_value_t = 512 << 10)]
        window: usize,
        #[arg(long, default_value_t = 4096)]
        write: usize,
    },
    /// Round trips of one message through an echo on the other host.
    Ping {
        path: PathBuf,
        #[arg(long, default_value_t = 10000)]
        count: usize,
        #[arg(long, default_value_t = 64)]
        size: usize,
        /// Spin for the echo instead of sleeping in poll().
        #[arg(long)]
        spin: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    let r = match cli.command {
        Command::Probe { address } => probe(&address),
        Command::Run { address, hostname, net, mtu, streams, ring, busy_poll, socket_dir, no_bounce, no_bonding } => {
            run(RunArgs { address, hostname, net, mtu, streams, ring, busy_poll, socket_dir, no_bounce, no_bonding })
        }
        Command::Cat { socket } => cat(&socket),
        #[cfg(target_os = "linux")]
        Command::Echo { path, sessions } => bench::echo(&path, sessions),
        #[cfg(target_os = "linux")]
        Command::Bench { path, bytes, window, write } => bench::bench(&path, bytes, window, write),
        #[cfg(target_os = "linux")]
        Command::Ping { path, count, size, spin } => bench::ping(&path, count, size, spin),
        #[cfg(not(target_os = "linux"))]
        Command::Echo { .. } | Command::Bench { .. } | Command::Ping { .. } => Err(anyhow::anyhow!("needs Linux")),
    };
    if let Err(e) = r {
        eprintln!("tbd: {e:#}");
        std::process::exit(1);
    }
}

#[allow(dead_code)]
struct RunArgs {
    address: String,
    hostname: Option<String>,
    net: String,
    mtu: u32,
    streams: Vec<String>,
    ring: u32,
    busy_poll: bool,
    socket_dir: PathBuf,
    no_bounce: bool,
    no_bonding: bool,
}

#[cfg(target_os = "linux")]
fn probe(address: &str) -> Result<()> {
    use crate::ctl::Ctl;
    use crate::switch::Router;
    let (nhi, irqs, _dev) = domain::open_nhi(address)?;
    println!(
        "NHI {address}: {} paths, REG_CAPS {:#x}, {}",
        nhi.hop_count,
        nhi.mmio().read32(nhi::REG_CAPS),
        if matches!(irqs, domain::Irqs::Msix(_)) { "MSI-X" } else { "MSI" }
    );
    irqs.attach(&nhi);
    let (ctl, events) = Ctl::start(&nhi, |f| matches!(f.pdf, ctl::PDF_EVENT | ctl::PDF_XDOMAIN_REQ))?;
    std::thread::spawn(move || {
        for e in events {
            println!("(event: PDF {} from route {:#x})", e.pdf, e.route());
        }
    });
    let r = Router::open(ctl, 0)?;
    let hdr = r.read(0, ctl::SPACE_SWITCH, 0, 5)?;
    println!("host router {:04x}:{:04x}, upstream port {}, max port {}, header {:08x?}", r.vendor, r.device, r.upstream_port, r.max_port, hdr);
    println!("plug events cap at {:#x}, link controller cap at {:#x}", r.cap_plug_events, r.cap_lc);
    println!("host UUID {}", r.uuid()?);
    for p in &r.ports {
        if p.disabled {
            println!("  port {:2}: not implemented", p.number);
            continue;
        }
        print!("  port {:2}: type {:#08x}, HopIDs in/out {}/{}, credits {}", p.number, p.kind, p.max_in_hop, p.max_out_hop, p.total_credits);
        if p.is_lane() && p.cap_phy != 0 {
            let [cs0, cs1] = r.lane_cs(p)?;
            print!(
                ", state {}, width {} (target {}), speed {:#x}, lane {}",
                r.port_state(p)?,
                (cs1 >> 20) & 0x3f,
                (cs1 >> 4) & 3,
                (cs1 >> 16) & 0xf,
                if cs1 & 1 << 14 != 0 { "disabled" } else { "enabled" }
            );
            let _ = cs0;
        }
        println!();
        if p.kind == switch::TYPE_PORT || p.kind == switch::TYPE_NHI {
            for h in 1..=p.max_in_hop.min(15) {
                let hop = r.read_hop(p.number, h)?;
                if hop.enable {
                    println!("      hop {h:2} -> port {} hop {}, credits {}, fc in/eg {}/{}", hop.out_port, hop.next_hop, hop.initial_credits, hop.ingress_fc as u8, hop.egress_fc as u8);
                }
            }
        }
    }
    nhi.quiesce();
    Ok(())
}

#[cfg(target_os = "linux")]
fn parse_stream(s: &str, a: &RunArgs) -> Result<stream::StreamSpec> {
    let parts: Vec<&str> = s.split(':').collect();
    let hop = |v: &str| -> Result<Option<u32>> { if v == "-1" || v.is_empty() { Ok(None) } else { Ok(Some(v.parse()?)) } };
    let (name, in_hop, out_hop) = match parts[..] {
        [n] => (n, None, None),
        [n, i, o] => (n, hop(i)?, hop(o)?),
        _ => anyhow::bail!("--stream {s:?}: expected NAME or NAME:IN:OUT"),
    };
    anyhow::ensure!(!name.is_empty() && name.len() <= 8, "stream name {name:?}: 1 to 8 characters");
    Ok(stream::StreamSpec { name: name.into(), in_hop, out_hop, ring: a.ring, socket_dir: a.socket_dir.clone(), busy_poll: a.busy_poll })
}

#[cfg(target_os = "linux")]
static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// With `TBD_KMSG=1`, standard output and error also go to the kernel log, one line per record, so a netconsole
/// receiver has tbd's last words if the machine dies (needs `sysctl kernel.printk_devkmsg=on`).
#[cfg(target_os = "linux")]
fn tee_to_kmsg() -> Result<()> {
    use std::io::{BufRead, Write};
    use std::os::fd::{AsRawFd, FromRawFd};
    let mut kmsg = std::fs::OpenOptions::new().write(true).open("/dev/kmsg")?;
    for target in [libc::STDOUT_FILENO, libc::STDERR_FILENO] {
        let mut fds = [0; 2];
        // SAFETY: a pipe whose write end replaces stdout/stderr; the read end is owned by the thread below.
        unsafe {
            anyhow::ensure!(libc::pipe(fds.as_mut_ptr()) == 0, "pipe");
            let orig = libc::dup(target);
            libc::dup2(fds[1], target);
            libc::close(fds[1]);
            let reader = std::io::BufReader::new(std::fs::File::from_raw_fd(fds[0]));
            let mut out = std::fs::File::from_raw_fd(orig);
            let mut k = kmsg.try_clone()?;
            std::thread::spawn(move || {
                for line in reader.lines().map_while(Result::ok) {
                    let _ = writeln!(out, "{line}");
                    let _ = k.write_all(format!("tbd: {line}\n").as_bytes());
                }
            });
        }
    }
    let _ = kmsg.as_raw_fd();
    let _ = kmsg.flush();
    Ok(())
}

#[cfg(target_os = "linux")]
fn run(a: RunArgs) -> Result<()> {
    use std::sync::atomic::Ordering;
    if std::env::var_os("TBD_KMSG").is_some() {
        tee_to_kmsg()?;
    }
    extern "C" fn on_signal(_: libc::c_int) {
        STOP.store(true, Ordering::Relaxed);
    }
    // SAFETY: the handler only stores to an atomic.
    unsafe {
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
    }
    let hostname = match &a.hostname {
        Some(h) => h.clone(),
        None => std::fs::read_to_string("/proc/sys/kernel/hostname")?.trim().to_owned(),
    };
    let streams = a.streams.iter().map(|s| parse_stream(s, &a)).collect::<Result<Vec<_>>>()?;
    let network = (a.net != "none").then(|| net::NetConfig { ifname: a.net.clone(), mtu: a.mtu, e2e: true, busy_poll: a.busy_poll });
    let config = domain::Config { hostname, network, streams, bounce: !a.no_bounce, bonding: !a.no_bonding };
    let (nhi, irqs, _dev) = domain::open_nhi(&a.address)?;
    let d = domain::Domain::start(nhi.clone(), irqs, config)?;
    while !STOP.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    println!("stopping");
    d.shutdown();
    nhi.quiesce();
    Ok(())
}

#[cfg(target_os = "linux")]
fn cat(socket: &std::path::Path) -> Result<()> {
    use std::io::{Read, Write};
    use std::os::unix::ffi::OsStrExt;
    // SAFETY: a socket and connect with a NUL-terminated path, then reads and writes on it.
    let fd = unsafe {
        let fd = libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0);
        anyhow::ensure!(fd >= 0, "socket: {}", std::io::Error::last_os_error());
        let mut addr: libc::sockaddr_un = std::mem::zeroed();
        addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
        for (i, &b) in socket.as_os_str().as_bytes().iter().enumerate() {
            addr.sun_path[i] = b as libc::c_char;
        }
        let len = std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t;
        anyhow::ensure!(libc::connect(fd, (&raw const addr).cast(), len) == 0, "connect {}: {}", socket.display(), std::io::Error::last_os_error());
        fd
    };
    let writer = std::thread::spawn(move || {
        let mut buf = vec![0u8; stream::MAX_MESSAGE];
        let mut stdin = std::io::stdin().lock();
        loop {
            let n = match stdin.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            // SAFETY: send of buf's first n bytes.
            if unsafe { libc::send(fd, buf.as_ptr().cast(), n, libc::MSG_NOSIGNAL) } < 0 {
                break;
            }
        }
        // Our side is done: tbd sends a CLOSE, and we keep reading until the other host's.
        // SAFETY: shutdown of the socket's writing side.
        unsafe { libc::shutdown(fd, libc::SHUT_WR) };
    });
    let mut buf = vec![0u8; stream::MAX_MESSAGE];
    let mut out = std::io::stdout().lock();
    loop {
        // SAFETY: recv into buf.
        let n = unsafe { libc::recv(fd, buf.as_mut_ptr().cast(), buf.len(), 0) };
        if n <= 0 {
            break;
        }
        out.write_all(&buf[..n as usize])?;
    }
    out.flush()?;
    let _ = writer.join();
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn probe(_: &str) -> Result<()> {
    anyhow::bail!("tbd needs Linux (VFIO)")
}

#[cfg(not(target_os = "linux"))]
fn run(_: RunArgs) -> Result<()> {
    anyhow::bail!("tbd needs Linux (VFIO)")
}

#[cfg(not(target_os = "linux"))]
fn cat(_: &std::path::Path) -> Result<()> {
    anyhow::bail!("tbd needs Linux")
}
