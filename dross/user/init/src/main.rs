//! init: the first process. It holds the root resources and starts everything else with only what each needs.
#![no_std]
#![no_main]

extern crate alloc;

use dross_abi::rights;
use dross_rt::bootstrap::{self, Bootstrap, Builder};
use dross_rt::{println, sys, Handle};

dross_rt::main!(main);

fn main(boot: Handle) -> i32 {
    let b = Bootstrap::read(boot).expect("bootstrap");
    let names: alloc::vec::Vec<_> = b.handles.iter().map(|(n, _)| n.as_str()).collect();
    println!("started with {}", names.join(" "));
    let system = b.handle("resource/system").expect("system resource");
    let ioport = b.handle("resource/ioport").expect("ioport resource");

    let mut ok = true;
    // Linux programs from init.conf: `start MODULE ARGS...` runs one; `check MODULE CODE ARGS...` runs one and
    // requires its exit code. Each gets a connection to pcid (bootstrap handle `pci`).
    let conf = b.handle("module/init.conf").map(read_module).unwrap_or_default();
    let all: alloc::vec::Vec<alloc::vec::Vec<&str>> =
        conf.lines().map(|l| l.split_whitespace().collect::<alloc::vec::Vec<_>>()).filter(|w| w.first().is_some_and(|f| !f.starts_with('#'))).collect();
    let lines: alloc::vec::Vec<&alloc::vec::Vec<&str>> = all.iter().filter(|w| matches!(w[0], "start" | "check" | "native" | "native-check")).collect();
    // `channel NAME` makes a channel; `@NAME=HANDLE` on a start or check line gives that program the next end of
    // it as bootstrap handle HANDLE (so a driver and its client can be connected, e.g. ethd and netd).
    let mut channels: Channels = alloc::vec::Vec::new();
    for w in all.iter().filter(|w| w[0] == "channel" && w.len() == 2) {
        let (a, b) = sys::channel_create().unwrap();
        channels.push((w[1], alloc::vec![b, a]));
    }
    // `mount NAME PATH`: a channel whose one end serves a read-only filesystem at PATH (Unix kernels); the other
    // end goes to the filesystem server as `@NAME=service`.
    for w in all.iter().filter(|w| w[0] == "mount" && w.len() == 3) {
        let (kernel_end, server_end) = sys::channel_create().unwrap();
        match sys::unix_mount(system, kernel_end, w[2]) {
            Ok(()) => channels.push((w[1], alloc::vec![server_end])),
            Err(e) => println!("init.conf: mount {} {}: {e:?}", w[1], w[2]),
        }
    }
    // `netservice NAME`: a channel whose one end becomes the kernel's network service (Unix kernels); the other
    // end goes to the network stack as `@NAME=service`.
    for w in all.iter().filter(|w| w[0] == "netservice" && w.len() == 2) {
        let (kernel_end, stack_end) = sys::channel_create().unwrap();
        match sys::unix_net_register(system, kernel_end) {
            Ok(()) => channels.push((w[1], alloc::vec![stack_end])),
            Err(e) => println!("init.conf: netservice {}: {e:?} (a kernel without the unix feature?)", w[1]),
        }
    }

    // pcid gets the ECAM, physical memory for BARs, interrupts and the IOMMU; each driver a channel to it.
    let (Some(pcid), Some(ecam)) = (b.handle("module/pcid"), b.value_u64("ecam")) else {
        println!("no pcid or no ECAM: no drivers");
        sys::system_exit(system, if ok { 0 } else { 1 });
    };
    let buses = b.value("ecam-buses").unwrap_or("0 0");
    let last: u64 = buses.split(' ').nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    let physical = b.handle("resource/physical").unwrap();
    let mut pb = Builder::default()
        .handle("resource/ecam", sys::resource_sub(physical, ecam, (last + 1) << 20).unwrap())
        .handle("resource/physical", sys::dup(physical, rights::ALL).unwrap())
        .handle("resource/irq", sys::dup(b.handle("resource/irq").unwrap(), rights::ALL).unwrap())
        .handle("resource/iommu", sys::dup(b.handle("resource/iommu").unwrap(), rights::ALL).unwrap())
        .value("ecam", &alloc::format!("{ecam:#x}"))
        .value("ecam-buses", buses);
    let test = b.value("hypervisor") == Some("1");
    // QEMU's edu device only exists under QEMU; init.conf can start usbd itself (`native usbd @seat=input`).
    let conf_usbd = lines.iter().any(|w| w[0] == "native" && w.get(1) == Some(&"usbd"));
    let drivers: alloc::vec::Vec<&str> = ["edutest", "usbd"].into_iter().filter(|&d| (d != "edutest" || test) && (d != "usbd" || !conf_usbd)).collect();
    let mut driver_ends = alloc::vec::Vec::new();
    for &d in &drivers {
        let (mine, theirs) = sys::channel_create().unwrap();
        pb = pb.handle(&alloc::format!("client/{d}"), mine);
        driver_ends.push(theirs);
    }
    let mut conf_ends = alloc::vec::Vec::new();
    for (i, _) in lines.iter().enumerate() {
        let (mine, theirs) = sys::channel_create().unwrap();
        pb = pb.handle(&alloc::format!("client/conf{i}"), mine);
        conf_ends.push(theirs);
    }
    bootstrap::spawn(pcid, "pcid", pb, u64::MAX).expect("spawn pcid");
    // init.conf's `start` and `native` lines up to the first `wait` (services such as ethd's network log) before
    // anything else, so their output covers the rest of the run; the rest run in order below.
    let mut ends: alloc::vec::Vec<Option<Handle>> = conf_ends.into_iter().map(Some).collect();
    let early = all.iter().position(|w| w[0] == "wait").unwrap_or(all.len());
    let mut li = 0;
    for words in &all[..early] {
        if matches!(words[0], "start" | "native") {
            ok &= start(&b, &mut channels, words, ends[li].take().unwrap());
        }
        li += matches!(words[0], "start" | "native" | "check" | "native-check") as usize;
    }

    if let Some(img) = b.handle("module/selftest") {
        let com1 = sys::resource_sub(ioport, 0x3f8, 8).unwrap();
        let mut bb = Builder::default().handle("module/selftest", sys::dup(img, rights::ALL).unwrap()).handle("resource/com1", com1)
            .handle("resource/iommu", sys::dup(b.handle("resource/iommu").unwrap(), rights::ALL).unwrap())
            .handle("resource/irq", sys::dup(b.handle("resource/irq").unwrap(), rights::ALL).unwrap())
            .handle("account/test", sys::account_create(system, 8192).unwrap())
            .handle("resource/system", sys::dup(system, rights::ALL).unwrap());
        if let Some(image) = b.handle("module/posixtest") { bb = bb.handle("module/posixtest", sys::dup(image, rights::ALL).unwrap()); }
        // API conformance programs get executable images only; the self-test
        // provisions their service endpoints explicitly for each test.
        for module in ["serviced", "c-api-test"] {
            let name = alloc::format!("module/{module}");
            if let Some(image) = b.handle(&name) {
                bb = bb.handle(&name, sys::dup(image, rights::READ | rights::DUPLICATE | rights::TRANSFER).unwrap());
            }
        }
        let p = bootstrap::spawn(img, "selftest", bb, 0).expect("spawn selftest");
        let code = sys::process_wait(p, dross_rt::FOREVER);
        println!("selftest exited with {code:?}");
        ok &= code == Ok(0);
    }
    // The drivers' unit tests (Linux programs), if the boot volume has them.
    let mut tests: alloc::vec::Vec<(alloc::string::String, Handle)> = alloc::vec::Vec::new();
    for (name, h) in &b.handles {
        if let Some(m) = name.strip_prefix("module/").filter(|m| m.starts_with("test-")) {
            tests.push((alloc::string::String::from(m), *h));
        }
    }
    tests.sort();
    for (m, img) in tests {
        let t0 = sys::clock();
        let code = sys::spawn_linux(img, &[m.as_str(), "--test-threads=4"], &["RUST_BACKTRACE=0"], None).and_then(|p| sys::process_wait(p, sys::after_ms(60_000)));
        println!("{m} exited with {code:?} after {} ms", (sys::clock() - t0) / 1_000_000);
        ok &= code == Ok(0);
    }

    // Under QEMU the run's harness types on the emulated keyboard and moves the mouse; usbd reports when it saw
    // both, and init waits for that.
    let mut usb_report = None;
    for (d, end) in drivers.iter().zip(driver_ends) {
        let Some(img) = b.handle(&alloc::format!("module/{d}")) else {
            continue;
        };
        let mut db = Builder::default().handle("pci", end);
        if *d == "usbd" && test {
            let (mine, theirs) = sys::channel_create().unwrap();
            db = db.handle("report", theirs);
            usb_report = Some(mine);
        }
        let p = bootstrap::spawn(img, d, db, u64::MAX).expect("spawn driver");
        if *d == "edutest" {
            let code = sys::process_wait(p, dross_rt::FOREVER);
            println!("{d} exited with {code:?}");
            ok &= code == Ok(0);
        }
    }
    // Then init.conf's `check`, `sleep MS`, `wait CHANNEL [MS]` (one message from the program holding the
    // channel's other end, e.g. netd's `ready`), the remaining `start` lines and `reboot` in order (`reboot` ends
    // the run when everything else is done: QEMU exits, hardware resets).
    let mut reboot = false;
    let mut li = 0;
    for (i, words) in all.iter().enumerate() {
        match words[0] {
            "wait" => {
                let ms = words.get(2).and_then(|v| v.parse().ok()).unwrap_or(30_000);
                let Some(end) = words.get(1).and_then(|c| channels.iter_mut().find(|(n, _)| n == c)).and_then(|(_, e)| e.pop()) else {
                    println!("init.conf: wait: no free end of channel {:?}", words.get(1));
                    ok = false;
                    continue;
                };
                let mut buf = [0u8; 64];
                let got = sys::channel_read(end, &mut buf, &mut [], sys::after_ms(ms));
                println!("{}: {:?}", words.join(" "), got.map(|(n, _)| core::str::from_utf8(&buf[..n]).unwrap_or("?")));
                ok &= got.is_ok();
            }
            "start" | "native" if i > early => {
                ok &= start(&b, &mut channels, words, ends[li].take().unwrap());
                li += 1;
            }
            "sleep" => {
                let ms = words.get(1).and_then(|v| v.parse().ok()).unwrap_or(1000);
                let _ = sys::notify_wait(sys::notify_create().unwrap(), sys::after_ms(ms));
            }
            "reboot" => reboot = true,
            "channel" | "netservice" | "mount" => {}
            "start" | "native" => li += 1,
            "check" | "native-check" => {
                let end = ends[li].take().unwrap();
                li += 1;
                let want = words.get(2).and_then(|c| c.parse::<i64>().ok());
                let mut words_argv = alloc::vec![words[1]];
                words_argv.extend_from_slice(&words[3..]);
                let args = take_ends(&b, &mut channels, &words_argv);
                let process = if words[0] == "native-check" {
                    let _ = sys::close(end);
                    launch_native(&b, &args, None)
                } else {
                    launch(&b, &args, end)
                };
                match process {
                    Some(p) => {
                        let code = sys::process_wait(p, sys::after_ms(600_000));
                        println!("{} exited with {code:?}", args.argv.join(" "));
                        ok &= code.ok() == want;
                        let _ = sys::close(p);
                    }
                    None => ok = false,
                }
            }
            _ => println!("init.conf: unknown line {:?}", words.join(" ")),
        }
    }
    if let Some(r) = usb_report {
        let mut buf = [0u8; 64];
        let mut hs = [Handle(0); 1];
        let got = sys::channel_read(r, &mut buf, &mut hs, sys::after_ms(120_000));
        println!("usbd: {:?}", got.map(|(n, _)| core::str::from_utf8(&buf[..n]).unwrap_or("?")));
        ok &= got.is_ok();
    }
    if !test {
        println!("{}", if ok { "all checks passed" } else { "checks FAILED" });
        if !reboot {
            let n = sys::notify_create().unwrap();
            let _ = sys::notify_wait(n, dross_rt::FOREVER);
        }
    }
    println!("{}", if ok { "all checks passed" } else { "checks FAILED" });
    sys::system_exit(system, if ok { 0 } else { 1 })
}

type Channels<'a> = alloc::vec::Vec<(&'a str, alloc::vec::Vec<Handle>)>;

/// A program's arguments, handles and bootstrap values, from an init.conf line.
#[derive(Default)]
struct Args {
    argv: alloc::vec::Vec<alloc::string::String>,
    handles: alloc::vec::Vec<(alloc::string::String, Handle)>,
    values: alloc::vec::Vec<(alloc::string::String, alloc::string::String)>,
}

/// A `start` or `native` line: runs the program; false if it could not be started.
fn start(b: &Bootstrap, channels: &mut Channels, words: &[&str], pci: Handle) -> bool {
    let args = take_ends(b, channels, &words[1..]);
    if words[0] == "native" { launch_native(b, &args, Some(pci)).is_some() } else { launch(b, &args, pci).is_some() }
}

/// A line's arguments: `@NAME=HANDLE` passes the next free end of channel NAME, or else a duplicate of init's
/// own handle NAME (`module/...`, `display`), as bootstrap handle HANDLE; `%KEY=VALUE` passes a value, and
/// `%KEY=$NAME` init's own value NAME (the kernel's `display-width` and so on); the rest are arguments.
fn take_ends(b: &Bootstrap, channels: &mut Channels, words: &[&str]) -> Args {
    let mut args = Args::default();
    for a in words {
        if let Some((k, v)) = a.strip_prefix('%').and_then(|kv| kv.split_once('=')) {
            let v = match v.strip_prefix('$') {
                Some(name) => match b.value(name) {
                    Some(v) => v,
                    None => {
                        println!("init.conf: no value {name}");
                        continue;
                    }
                },
                None => v,
            };
            args.values.push((alloc::string::String::from(k), alloc::string::String::from(v)));
        } else if let Some((chan, name)) = a.strip_prefix('@').and_then(|r| r.split_once('=')) {
            let end = channels.iter_mut().find(|(n, _)| *n == chan).and_then(|(_, ends)| ends.pop());
            match end.or_else(|| b.handle(chan).and_then(|h| sys::dup(h, rights::ALL).ok())) {
                Some(h) => args.handles.push((alloc::string::String::from(name), h)),
                None => println!("init.conf: no free end of channel {chan} and no handle of that name"),
            }
        } else {
            args.argv.push(alloc::string::String::from(*a));
        }
    }
    args
}

/// A text module (init.conf) as a string.
fn read_module(h: Handle) -> alloc::string::String {
    let Ok(size) = sys::mem_size(h) else {
        return Default::default();
    };
    let Ok(at) = sys::vm_map(h, 0, size, 0, dross_abi::vm::READ) else {
        return Default::default();
    };
    // SAFETY: a read-only mapping of `size` bytes that stays.
    let bytes = unsafe { core::slice::from_raw_parts(at as *const u8, size as usize) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    alloc::string::String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Starts the Linux program `argv[0]` (a module) with a pcid connection, the line's handles and values in its
/// bootstrap message.
fn launch(b: &Bootstrap, args: &Args, pci: Handle) -> Option<Handle> {
    let argv: alloc::vec::Vec<&str> = args.argv.iter().map(|s| s.as_str()).collect();
    let Some(img) = b.handle(&alloc::format!("module/{}", argv[0])) else {
        println!("init.conf: no module {}", argv[0]);
        return None;
    };
    let boot = builder(args, Some(pci)).send().ok()?;
    let env = ["DROSS_BOOTSTRAP=1", "RUST_BACKTRACE=0", "PATH=/bin:/sbin:/usr/bin:/usr/sbin", "HOME=/root", "TERM=xterm"];
    match sys::spawn_linux(img, &argv, &env, Some(boot)) {
        Ok(p) => Some(p),
        Err(e) => {
            println!("{}: {e:?}", argv.join(" "));
            None
        }
    }
}

/// Starts a native module with explicit handles/values and an optional pcid connection.
fn launch_native(b: &Bootstrap, args: &Args, pci: Option<Handle>) -> Option<Handle> {
    let Some(img) = b.handle(&alloc::format!("module/{}", args.argv[0])) else {
        println!("init.conf: no module {}", args.argv[0]);
        return None;
    };
    match bootstrap::spawn(img, &args.argv[0], builder(args, pci), u64::MAX) {
        Ok(process) => Some(process),
        Err(e) => {
            println!("{}: {e:?}", args.argv.join(" "));
            None
        }
    }
}

fn builder(args: &Args, pci: Option<Handle>) -> Builder {
    let mut bb = Builder::default();
    if let Some(pci) = pci {
        if args.argv.first().is_some_and(|name| name == "unix-netd") {
            // The network service receives a port from the NIC driver, not PCI access.
            let _ = sys::close(pci);
        } else {
            bb = bb.handle("pci", pci);
        }
    }
    for (name, h) in &args.handles {
        bb = bb.handle(name, *h);
    }
    for (k, v) in &args.values {
        bb = bb.value(k, v);
    }
    bb
}
