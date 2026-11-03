//! ethd: a userspace Ethernet driver for the Mac Pro 6,1's Broadcom BCM57762 ports (Linux's tg3 family). It owns
//! one port through VFIO and gives the host a TAP interface on it, or runs a bounded self-test against a peer.
//! See driver/ethd/README.md.

// Off Linux only the chip logic and its tests build.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

mod hw;
mod netlog;
mod oeth;
mod nic;
mod phy;
mod regs;
#[cfg(test)]
mod sim;
mod tap;
#[cfg(test)]
mod tests;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Userspace Ethernet driver for the Mac Pro 6,1's BCM57762 ports")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Identify the chip, its station address and the PHY's link, without resetting anything or giving the chip
    /// any memory.
    Probe {
        /// PCI address of the port, e.g. 0000:0c:00.0
        address: String,
    },
    /// Print the registers tg3 shows for `ethtool -d` (without the clear-on-read statistics), "offset value" per
    /// line.
    Regs { address: String },
    /// Bring the port up and exchange a bounded number of unicast test frames (EtherType 0x88b5) with a peer that
    /// echoes them (peer.py), then halt the port.
    Selftest {
        address: String,
        /// The port's station address; ethd refuses any other port.
        #[arg(long)]
        expect_mac: String,
        /// The echoing peer's MAC address (unicast only).
        #[arg(long)]
        peer: String,
        #[arg(long, default_value_t = 1000)]
        count: u32,
        /// Frame size in bytes, 60-1514.
        #[arg(long, default_value_t = 1514)]
        size: usize,
        /// Frames in flight at most.
        #[arg(long, default_value_t = 64)]
        window: u32,
        /// Also dump the registers after bring-up to this file.
        #[arg(long)]
        regs_out: Option<std::path::PathBuf>,
    },
    /// Send the kernel log (/dev/kmsg) to the LAN as UDP datagrams to PORT, from the start of the log, until
    /// SIGINT or SIGTERM: a console for machines without one (like Linux's netconsole, whose port it uses).
    Netlog {
        address: String,
        /// The source IPv4 address in the datagrams (macOS drops broadcasts from 0.0.0.0).
        #[arg(long, default_value = "0.0.0.0")]
        ip: std::net::Ipv4Addr,
        /// The receiver (default: broadcast).
        #[arg(long, default_value = "255.255.255.255")]
        to: std::net::Ipv4Addr,
        /// The receiver's MAC address (default: broadcast).
        #[arg(long, default_value = "ff:ff:ff:ff:ff:ff")]
        to_mac: String,
        #[arg(long, default_value_t = 6666)]
        port: u16,
    },
    /// On dross: be the NIC behind Oscar's network stack, frames between the port and the OETH channel named
    /// `ethernet` in the bootstrap message, until SIGINT or SIGTERM or the stack goes away. With --netlog-to it
    /// also sends the kernel log out of the port, as `netlog` does.
    Serve {
        address: String,
        #[arg(long)]
        netlog_to: Option<std::net::Ipv4Addr>,
        #[arg(long, default_value = "ff:ff:ff:ff:ff:ff")]
        netlog_mac: String,
        #[arg(long, default_value = "0.0.0.0")]
        netlog_ip: std::net::Ipv4Addr,
    },
    /// On dross: check an OETH driver from the stack's side: take its Info, ARP for TARGET from IP through the
    /// channel named `ethernet`, and wait for the reply.
    OethArp {
        #[arg(long)]
        ip: std::net::Ipv4Addr,
        #[arg(long)]
        target: std::net::Ipv4Addr,
    },
    /// Run the driver: frames between the port and a TAP interface until SIGINT or SIGTERM.
    Run {
        address: String,
        /// The port's station address; ethd refuses any other port.
        #[arg(long)]
        expect_mac: String,
        /// TAP interface name (an existing persistent TAP is reused as it is configured).
        #[arg(long, default_value = "ethd%d")]
        tap: String,
        /// Accept every multicast frame (IPv6 needs it); by default only broadcast and our address.
        #[arg(long)]
        all_multi: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    #[cfg(target_os = "linux")]
    return linux::main(cli);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cli.command;
        anyhow::bail!("ethd drives hardware through VFIO, which needs Linux")
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use crate::hw::{Hw, PciHw};
    use crate::nic::{self, Nic, Options, fmt_mac, parse_mac};
    use crate::phy;
    use crate::tap::Tap;
    use anyhow::{Context, bail, ensure};
    use gpud_platform::{DmaSpace, IrqEvent};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    static STOP: AtomicBool = AtomicBool::new(false);

    extern "C" fn on_signal(_: libc::c_int) {
        STOP.store(true, Ordering::SeqCst);
    }

    fn catch_signals() {
        // SAFETY: the handler only stores to an atomic.
        unsafe {
            libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
            libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
        }
    }

    struct Port {
        hw: Arc<PciHw>,
        container: Box<dyn DmaSpace>,
    }

    fn open_port(address: &str) -> Result<Port> {
        let addr = address.parse()?;
        let (container, dev) = gpud_platform::open(addr, 1 << 20..1 << 40).with_context(|| format!("open {address}"))?;
        let hw = Arc::new(PciHw::new(dev)?);
        nic::identify(&*hw)?;
        Ok(Port { hw, container })
    }

    /// Starts the port: interrupts routed, chip reset and running. Drop the Nic before the port.
    fn start(port: &Port, expect_mac: &str, opts: Options) -> Result<(Nic, IrqEvent)> {
        let want = parse_mac(expect_mac)?;
        let irq = port.hw.dev.enable_msi()?;
        let hw: Arc<dyn Hw> = port.hw.clone();
        let nic = Nic::open(hw, &*port.container, Some(want), opts)?;
        Ok((nic, irq))
    }

    pub fn main(cli: Cli) -> Result<()> {
        match cli.command {
            Command::Probe { address } => probe(&address),
            Command::Regs { address } => {
                let port = open_port(&address)?;
                for (off, v) in nic::dump_regs(&*port.hw) {
                    println!("{off:04x} {v:08x}");
                }
                Ok(())
            }
            Command::Selftest { address, expect_mac, peer, count, size, window, regs_out } => {
                selftest(&address, &expect_mac, &peer, count, size, window, regs_out)
            }
            Command::Run { address, expect_mac, tap, all_multi } => run(&address, &expect_mac, &tap, all_multi),
            Command::Netlog { address, ip, to, to_mac, port } => netlog(&address, ip, to, &to_mac, port),
            Command::Serve { address, netlog_to, netlog_mac, netlog_ip } => serve(&address, netlog_to.map(|to| (to, netlog_mac, netlog_ip))),
            Command::OethArp { ip, target } => oeth_arp(ip, target),
        }
    }

    fn ethernet_channel() -> Result<dross_rt::Handle> {
        gpud_platform::dross::bootstrap_handle("ethernet").context("no `ethernet` channel in the bootstrap message (dross only)")
    }

    fn os(e: dross_rt::Error) -> anyhow::Error {
        anyhow::anyhow!("dross: {e:?}")
    }

    /// Waits until `irq` fires, the channel has a message (its bound notification's fd `n_fd`), or `ms` pass.
    fn wait_either(irq: &IrqEvent, n_fd: i32, ms: i32) {
        let mut fds = [
            libc::pollfd { fd: n_fd, events: libc::POLLIN, revents: 0 },
            libc::pollfd { fd: irq.raw_fd().unwrap_or(-1), events: libc::POLLIN, revents: 0 },
        ];
        // SAFETY: poll on two descriptors this process owns.
        unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, ms) };
    }

    type NetlogTo = (std::net::Ipv4Addr, String, std::net::Ipv4Addr);

    fn serve(address: &str, netlog: Option<NetlogTo>) -> Result<()> {
        use crate::oeth::{self, Message};
        use dross_rt::sys;
        use std::collections::VecDeque;
        use std::io::Read;
        catch_signals();
        let eth = ethernet_channel()?;
        let port = open_port(address)?;
        let mac = nic::station_address(&*port.hw)?;
        let opts = Options { all_multi: true, link_wait: Duration::from_secs(10) };
        let (nic, irq) = start(&port, &fmt_mac(&mac), opts)?;
        let mut link = nic.link();
        eprintln!("serve: {} ({}) for the network stack", fmt_mac(&mac), link);
        let n = sys::notify_create().map_err(os)?;
        sys::channel_bind(eth, n, 1).map_err(os)?;
        let n_fd = sys::handle_fd(n).map_err(os)?;
        let send = |m: &[u8]| sys::channel_write(eth, m, &[]);
        send(&oeth::info(&mac, 1500, link.up, link.speed)).map_err(os)?;

        // The kernel log, if asked for, read by a thread as in `netlog`.
        let log = match netlog {
            Some((to, to_mac, ip)) => {
                let dst = (parse_mac(&to_mac)?, to.octets());
                let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
                std::thread::spawn(move || {
                    let Ok(mut kmsg) = std::fs::File::open("/dev/kmsg") else { return };
                    let mut buf = vec![0u8; 8192];
                    while let Ok(n) = kmsg.read(&mut buf) {
                        if n == 0 || tx.send(buf[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                });
                Some((rx, dst, ip))
            }
            None => None,
        };
        let mut log_id = 0u16;

        // Frames the stack's channel had no room for yet (bounded, oldest dropped), and frames for the wire.
        let mut to_stack: VecDeque<Vec<u8>> = VecDeque::new();
        let mut to_wire: VecDeque<Vec<u8>> = VecDeque::new();
        let mut buf = vec![0u8; 2048];
        let mut hs = [dross_rt::Handle(0); 1];
        let (mut up, mut down, mut dropped) = (0u64, 0u64, 0u64);
        let mut last_link_check = Instant::now();
        let mut last_report = Instant::now();
        let mut loops = 0u64;
        'outer: while !STOP.load(Ordering::SeqCst) {
            loops += 1;
            if last_report.elapsed() > Duration::from_secs(2) {
                last_report = Instant::now();
                eprintln!("serve: {loops} loops, {up} frames to the stack, {down} to the wire, {} and {} queued", to_stack.len(), to_wire.len());
            }
            wait_either(&irq, n_fd, 20);
            let _ = irq.take();
            // Drain the notification's fd so it polls quiet until the next message.
            let mut pf = libc::pollfd { fd: n_fd, events: libc::POLLIN, revents: 0 };
            let mut b8 = [0u8; 8];
            // SAFETY: poll and read on our own descriptor; the read only happens when it is readable.
            unsafe {
                if libc::poll(&mut pf, 1, 0) > 0 {
                    libc::read(n_fd, b8.as_mut_ptr().cast(), 8);
                }
            }
            // From the stack.
            loop {
                match sys::channel_read(eth, &mut buf, &mut hs, 0) {
                    Ok((len, _)) => {
                        if let Some(Message::Frame(f)) = oeth::parse(&buf[..len]) {
                            to_wire.push_back(f.to_vec());
                        }
                    }
                    Err(dross_rt::Error::TimedOut) => break,
                    Err(dross_rt::Error::PeerClosed) => break 'outer,
                    Err(e) => return Err(os(e)),
                }
            }
            // The kernel log.
            if let Some((rx, dst, ip)) = &log {
                let mut batch = Vec::new();
                for r in rx.try_iter() {
                    if let Some(m) = netlog::kmsg_message(&r) {
                        if batch.len() + m.len() > 1400 && !batch.is_empty() {
                            to_wire.push_back(netlog::udp_frame(&mac, ip.octets(), &dst.0, dst.1, 6666, &batch, log_id));
                            log_id = log_id.wrapping_add(1);
                            batch.clear();
                        }
                        batch.extend_from_slice(&m[..m.len().min(1400)]);
                        batch.push(b'\n');
                    }
                }
                if !batch.is_empty() {
                    to_wire.push_back(netlog::udp_frame(&mac, ip.octets(), &dst.0, dst.1, 6666, &batch, log_id));
                    log_id = log_id.wrapping_add(1);
                }
            }
            while let Some(f) = to_wire.front() {
                let mut f = f.clone();
                f.resize(f.len().max(60), 0);
                if !nic.send(&f)? {
                    break;
                }
                to_wire.pop_front();
                down += 1;
            }
            // From the wire.
            let serviced = nic.service(&mut |f: &[u8]| {
                if to_stack.len() >= 1024 {
                    to_stack.pop_front();
                    dropped += 1;
                }
                to_stack.push_back(oeth::frame(&f[..f.len().min(oeth::MAX_FRAME)]));
            })?;
            while let Some(m) = to_stack.front() {
                match send(m) {
                    Ok(()) => {
                        to_stack.pop_front();
                        up += 1;
                    }
                    Err(dross_rt::Error::ShouldWait) => break,
                    Err(dross_rt::Error::PeerClosed) => break 'outer,
                    Err(e) => return Err(os(e)),
                }
            }
            let now_link = match serviced.link {
                Some(l) => Some(l),
                None if last_link_check.elapsed() > Duration::from_secs(1) => {
                    last_link_check = Instant::now();
                    Some(nic.check_link()?)
                }
                None => None,
            };
            if let Some(l) = now_link.filter(|l| l.up != link.up || l.speed != link.speed) {
                link = l;
                eprintln!("serve: link {link}");
                let _ = send(&oeth::link(link.up, link.speed));
            }
        }
        eprintln!("serve: {up} frames to the stack, {down} to the wire, {dropped} dropped");
        nic.halt()?;
        Ok(())
    }

    fn oeth_arp(ip: std::net::Ipv4Addr, target: std::net::Ipv4Addr) -> Result<()> {
        use crate::oeth::{self, Message};
        use dross_rt::sys;
        let eth = ethernet_channel()?;
        let mut buf = vec![0u8; 2048];
        let mut hs = [dross_rt::Handle(0); 1];
        let deadline = |ms: u64| sys::clock() + ms * 1_000_000;
        let (len, _) = sys::channel_read(eth, &mut buf, &mut hs, deadline(20_000)).map_err(os)?;
        let Some(Message::Info { mac, mtu, link, mbps }) = oeth::parse(&buf[..len]) else {
            bail!("the driver's first message is not Info")
        };
        println!("oeth: driver {} mtu {mtu} link {link} {mbps} Mb/s", fmt_mac(&mac));
        let t0 = Instant::now();
        for attempt in 0..5 {
            sys::channel_write(eth, &oeth::frame(&oeth::arp_request(&mac, ip.octets(), target.octets())), &[]).map_err(os)?;
            let until = deadline(1000);
            while sys::clock() < until {
                let Ok((len, _)) = sys::channel_read(eth, &mut buf, &mut hs, until) else { break };
                if let Some(Message::Frame(f)) = oeth::parse(&buf[..len]) {
                    if let Some(who) = oeth::arp_reply_from(f, target.octets()) {
                        println!("oeth: {target} is at {} (attempt {}, {:.2} ms)", fmt_mac(&who), attempt + 1, t0.elapsed().as_secs_f64() * 1e3);
                        return Ok(());
                    }
                }
            }
        }
        bail!("no ARP reply from {target}")
    }

    fn netlog(
        address: &str,
        ip: std::net::Ipv4Addr,
        to: std::net::Ipv4Addr,
        to_mac: &str,
        udp_port: u16,
    ) -> Result<()> {
        use std::io::Read;
        use std::sync::mpsc;
        let dst = (parse_mac(to_mac)?, to.octets());
        catch_signals();
        let port = open_port(address)?;
        let mac = nic::station_address(&*port.hw)?;
        let opts = Options { all_multi: false, link_wait: Duration::from_secs(10) };
        let (nic, irq) = start(&port, &fmt_mac(&mac), opts)?;
        ensure!(nic.link().up, "no link");
        eprintln!("netlog: {} up ({}), sending from {ip} to {to}:{udp_port}", fmt_mac(&mac), nic.link());

        // A thread reads the log, one record per read; the main thread batches and sends.
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        std::thread::spawn(move || {
            let Ok(mut kmsg) = std::fs::File::open("/dev/kmsg") else { return };
            let mut buf = vec![0u8; 8192];
            while let Ok(n) = kmsg.read(&mut buf) {
                if n == 0 || tx.send(buf[..n].to_vec()).is_err() {
                    break;
                }
            }
        });
        let mut id = 0u16;
        let mut batch: Vec<u8> = Vec::new();
        let mut sent = 0u64;
        while !STOP.load(Ordering::SeqCst) {
            let first = rx.recv_timeout(Duration::from_millis(20));
            let records = first.into_iter().chain(rx.try_iter());
            for r in records {
                if let Some(m) = netlog::kmsg_message(&r) {
                    if batch.len() + m.len() + 1 > 1400 && !batch.is_empty() {
                        flush(&nic, &mac, ip, dst, udp_port, &mut batch, &mut id, &mut sent)?;
                    }
                    batch.extend_from_slice(&m[..m.len().min(1400)]);
                    batch.push(b'\n');
                }
            }
            if !batch.is_empty() {
                flush(&nic, &mac, ip, dst, udp_port, &mut batch, &mut id, &mut sent)?;
            }
            nic.service(&mut |_| {})?;
            let _ = irq.take();
        }
        eprintln!("netlog: {sent} datagrams");
        nic.halt()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn flush(
        nic: &Nic,
        mac: &[u8; 6],
        ip: std::net::Ipv4Addr,
        dst: ([u8; 6], [u8; 4]),
        port: u16,
        batch: &mut Vec<u8>,
        id: &mut u16,
        sent: &mut u64,
    ) -> Result<()> {
        let frame = netlog::udp_frame(mac, ip.octets(), &dst.0, dst.1, port, batch, *id);
        // A full ring drains as the chip sends; wait for room rather than drop log lines.
        let t0 = Instant::now();
        while !nic.send(&frame)? {
            nic.service(&mut |_| {})?;
            ensure!(t0.elapsed() < Duration::from_secs(2), "send ring stuck");
            std::thread::sleep(Duration::from_millis(1));
        }
        *id = id.wrapping_add(1);
        *sent += 1;
        batch.clear();
        Ok(())
    }

    fn probe(address: &str) -> Result<()> {
        let port = open_port(address)?;
        let hw = &*port.hw;
        let id = nic::identify(hw)?;
        // Reading NIC SRAM needs the memory arbiter on (it is, unless something turned it off).
        let mac = nic::station_address(hw)?;
        println!("{address}: {:04x}:{:04x}, chip revision {:#x}", id.vendor, id.device, id.chip_rev_id);
        println!("station address {}", fmt_mac(&mac));
        let mi_mode = hw.read32(crate::regs::MAC_MI_MODE);
        if mi_mode & crate::regs::MAC_MI_MODE_AUTO_POLL != 0 {
            println!("PHY: the MAC is polling it (MI_MODE {mi_mode:#x}); not read");
            return Ok(());
        }
        println!("PHY {:#010x}", phy::id(hw)?);
        let regs = [
            ("BMCR", 0x00),
            ("BMSR", 0x01),
            ("ADVERTISE", 0x04),
            ("LPA", 0x05),
            ("CTRL1000", 0x09),
            ("STAT1000", 0x0a),
        ];
        let line: Vec<String> =
            regs.iter().map(|(n, r)| Ok(format!("{n} {:04x}", phy::read(hw, *r)?))).collect::<Result<_>>()?;
        println!("PHY registers: {}", line.join(", "));
        match phy::read_link(hw)? {
            Some(link) => println!("link {link}"),
            None => println!("link up, but not negotiated with ethd's advertisement"),
        }
        Ok(())
    }

    const ETHERTYPE: [u8; 2] = [0x88, 0xb5];

    fn selftest(
        address: &str,
        expect_mac: &str,
        peer: &str,
        count: u32,
        size: usize,
        window: u32,
        regs_out: Option<std::path::PathBuf>,
    ) -> Result<()> {
        let peer = parse_mac(peer)?;
        ensure!(nic::valid_unicast(&peer), "the peer must be a unicast address");
        ensure!((60..=nic::MAX_FRAME).contains(&size), "frame size {size}: 60-1514");
        catch_signals();
        let port = open_port(address)?;
        let opts = Options { all_multi: false, link_wait: Duration::from_secs(10) };
        let t0 = Instant::now();
        let (nic, irq) = start(&port, expect_mac, opts)?;
        println!(
            "up in {:.0} ms: chip {:#x}, {}, link {}",
            t0.elapsed().as_secs_f64() * 1e3,
            nic.id.chip_rev_id,
            fmt_mac(&nic.mac),
            nic.link()
        );
        if let Some(path) = regs_out {
            let text: String = nic.dump_regs().iter().map(|(o, v)| format!("{o:04x} {v:08x}\n")).collect();
            std::fs::write(&path, text)?;
        }
        ensure!(nic.link().up, "no link");

        let tag: u32 = std::process::id() ^ (t0.elapsed().as_nanos() as u32);
        let frame = |i: u32| {
            let mut f = Vec::with_capacity(size);
            f.extend_from_slice(&peer);
            f.extend_from_slice(&nic.mac);
            f.extend_from_slice(&ETHERTYPE);
            f.extend_from_slice(b"ethd-selftest");
            f.extend_from_slice(&tag.to_be_bytes());
            f.extend_from_slice(&i.to_be_bytes());
            // A pattern that depends on the frame number, so corruption shows.
            while f.len() < size {
                f.push((f.len() as u32 ^ i) as u8);
            }
            f
        };
        let (mut sent, mut echoed, mut bad, mut dup, mut other) = (0u32, 0u32, 0u32, 0u32, 0u32);
        let mut seen = vec![false; count as usize];
        let start = Instant::now();
        let mut last_progress = Instant::now();
        while echoed < count && !STOP.load(Ordering::SeqCst) {
            while sent < count && sent - echoed.min(sent) < window {
                if !nic.send(&frame(sent))? {
                    break;
                }
                sent += 1;
            }
            match irq.wait(Some(Duration::from_millis(100))) {
                Ok(_) | Err(gpud_platform::Error::Timeout) => {}
                Err(e) => return Err(e.into()),
            }
            nic.service(&mut |f: &[u8]| {
                if f.len() < 35 || f[12..14] != ETHERTYPE || f[14..27] != *b"ethd-selftest" {
                    other += 1;
                    return;
                }
                let i = u32::from_be_bytes(f[31..35].try_into().unwrap());
                if u32::from_be_bytes(f[27..31].try_into().unwrap()) != tag || i >= count {
                    other += 1;
                    return;
                }
                // The echo comes back with the addresses swapped; the rest must be ours, byte for byte.
                let want = frame(i);
                if f[..6] != want[6..12] || f[6..12] != want[..6] || f[12..] != want[12..] {
                    bad += 1;
                } else if seen[i as usize] {
                    dup += 1;
                } else {
                    seen[i as usize] = true;
                    echoed += 1;
                    last_progress = Instant::now();
                }
            })?;
            if last_progress.elapsed() > Duration::from_secs(2) {
                println!("no echo for 2 s; stopping");
                break;
            }
        }
        let secs = start.elapsed().as_secs_f64();
        let s = nic.stats();
        println!(
            "sent {sent}, echoed {echoed}, corrupt {bad}, duplicates {dup}, other frames {other}, in {:.3} s: {:.0} frames/s each way, {:.1} Mb/s each way",
            secs,
            echoed as f64 / secs,
            echoed as f64 * size as f64 * 8.0 / secs / 1e6
        );
        println!(
            "driver: tx {} frames, rx {} frames, rx errors {}, ring full {}, services {}, link changes {}",
            s.tx_frames, s.rx_frames, s.rx_errors, s.tx_ring_full, s.irq_services, s.link_changes
        );
        let pending = nic.tx_pending();
        drop(nic);
        port.hw.dev.disable_irqs()?;
        println!("halted ({pending} sends were still queued)");
        if echoed != count || bad != 0 || dup != 0 {
            bail!("self-test failed");
        }
        Ok(())
    }

    fn run(address: &str, expect_mac: &str, tap_name: &str, all_multi: bool) -> Result<()> {
        catch_signals();
        let port = open_port(address)?;
        let opts = Options { all_multi, link_wait: Duration::from_secs(5) };
        let (nic, irq) = start(&port, expect_mac, opts)?;
        let tap = Tap::open(tap_name, nic.mac, 1500)?;
        println!("{address}: {} on {}, link {}", fmt_mac(&nic.mac), tap.name, nic.link());
        let nic = Arc::new(nic);
        let tap = Arc::new(tap);

        // Host to wire: one frame per TAP read.
        let tx = {
            let (nic, tap) = (nic.clone(), tap.clone());
            std::thread::spawn(move || -> Result<()> {
                let mut buf = vec![0u8; 65536];
                while !STOP.load(Ordering::SeqCst) {
                    let n = tap.read(&mut buf, Duration::from_millis(100))?;
                    if n == 0 {
                        continue;
                    }
                    if n > nic::MAX_FRAME {
                        continue;
                    }
                    // A full ring drains at line rate; wait for it rather than drop.
                    while !nic.send(&buf[..n])? {
                        if STOP.load(Ordering::SeqCst) {
                            return Ok(());
                        }
                        std::thread::sleep(Duration::from_micros(50));
                    }
                }
                Ok(())
            })
        };

        // Wire to host, and the link.
        let mut last_link = Instant::now();
        let mut result = Ok(());
        while !STOP.load(Ordering::SeqCst) {
            match irq.wait(Some(Duration::from_millis(100))) {
                Ok(_) | Err(gpud_platform::Error::Timeout) => {}
                Err(e) => {
                    result = Err(e.into());
                    break;
                }
            }
            let mut tap_err = None;
            let r = nic.service(&mut |f: &[u8]| {
                if let Err(e) = tap.write(f) {
                    tap_err.get_or_insert(e);
                }
            });
            match r {
                Err(e) => {
                    result = Err(e);
                    break;
                }
                Ok(s) => {
                    if let Some(link) = s.link {
                        println!("link {link}");
                    }
                }
            }
            if let Some(e) = tap_err {
                eprintln!("ethd: TAP write: {e}");
            }
            if last_link.elapsed() > Duration::from_secs(1) {
                last_link = Instant::now();
                let before = nic.link();
                let now = nic.check_link()?;
                if now != before {
                    println!("link {now}");
                }
            }
            if tx.is_finished() {
                break;
            }
        }
        STOP.store(true, Ordering::SeqCst);
        let tx_result = tx.join().unwrap_or_else(|_| Err(anyhow::anyhow!("TAP thread panicked")));
        let s = nic.stats();
        println!(
            "tx {} frames {} bytes, rx {} frames {} bytes, rx errors {}, link changes {}",
            s.tx_frames, s.tx_bytes, s.rx_frames, s.rx_bytes, s.rx_errors, s.link_changes
        );
        drop(Arc::try_unwrap(nic).map_err(|_| anyhow::anyhow!("NIC still shared"))?);
        port.hw.dev.disable_irqs()?;
        println!("halted");
        result.and(tx_result)
    }
}
