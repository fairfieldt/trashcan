//! ethd: a userspace Ethernet driver for the Mac Pro 6,1's Broadcom BCM57762 ports (Linux's tg3 family). It owns
//! one port through VFIO and gives the host a TAP interface on it, or runs a bounded self-test against a peer.
//! See driver/ethd/README.md.

// Off Linux only the chip logic and its tests build.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

mod hw;
mod netlog;
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
        }
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
