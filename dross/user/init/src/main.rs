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
    let lines: alloc::vec::Vec<&alloc::vec::Vec<&str>> = all.iter().filter(|w| matches!(w[0], "start" | "check")).collect();
    // `channel NAME` makes a channel; `@NAME=HANDLE` on a start or check line gives that program the next end of
    // it as bootstrap handle HANDLE (so a driver and its client can be connected, e.g. ethd and netd).
    let mut channels: alloc::vec::Vec<(&str, alloc::vec::Vec<Handle>)> = alloc::vec::Vec::new();
    for w in all.iter().filter(|w| w[0] == "channel" && w.len() == 2) {
        let (a, b) = sys::channel_create().unwrap();
        channels.push((w[1], alloc::vec![b, a]));
    }
    let mut take_ends = |args: &[&str]| -> (alloc::vec::Vec<alloc::string::String>, alloc::vec::Vec<(alloc::string::String, Handle)>) {
        let mut argv = alloc::vec::Vec::new();
        let mut extra = alloc::vec::Vec::new();
        for a in args {
            if let Some((chan, name)) = a.strip_prefix('@').and_then(|r| r.split_once('=')) {
                match channels.iter_mut().find(|(n, _)| *n == chan).and_then(|(_, ends)| ends.pop()) {
                    Some(end) => extra.push((alloc::string::String::from(name), end)),
                    None => println!("init.conf: no free end of channel {chan}"),
                }
            } else {
                argv.push(alloc::string::String::from(*a));
            }
        }
        (argv, extra)
    };

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
    // QEMU's edu device only exists under QEMU.
    let drivers: &[&str] = if test { &["edutest", "usbd"] } else { &["usbd"] };
    let mut driver_ends = alloc::vec::Vec::new();
    for &d in drivers {
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
    // init.conf's `start` lines (services such as ethd's network log) before anything else, so their output
    // covers the rest of the run.
    let mut ends: alloc::vec::Vec<Option<Handle>> = conf_ends.into_iter().map(Some).collect();
    for (i, words) in lines.iter().enumerate() {
        if words[0] == "start" {
            let (argv, extra) = take_ends(&words[1..]);
            let argv: alloc::vec::Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
            ok &= launch(&b, &argv, ends[i].take().unwrap(), &extra).is_some();
        }
    }

    if let Some(img) = b.handle("module/selftest") {
        let com1 = sys::resource_sub(ioport, 0x3f8, 8).unwrap();
        let bb = Builder::default().handle("module/selftest", sys::dup(img, rights::ALL).unwrap()).handle("resource/com1", com1);
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
    // Then init.conf's `check`, `sleep MS` and `reboot` lines in order (`reboot` ends the run when everything
    // else is done: QEMU exits, hardware resets).
    let mut reboot = false;
    let mut li = 0;
    for words in &all {
        match words[0] {
            "sleep" => {
                let ms = words.get(1).and_then(|v| v.parse().ok()).unwrap_or(1000);
                let _ = sys::notify_wait(sys::notify_create().unwrap(), sys::after_ms(ms));
            }
            "reboot" => reboot = true,
            "channel" => {}
            "start" => li += 1,
            "check" => {
                let end = ends[li].take().unwrap();
                li += 1;
                let want = words.get(2).and_then(|c| c.parse::<i64>().ok());
                let mut words_argv = alloc::vec![words[1]];
                words_argv.extend_from_slice(&words[3..]);
                let (argv, extra) = take_ends(&words_argv);
                let argv: alloc::vec::Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
                match launch(&b, &argv, end, &extra) {
                    Some(p) => {
                        let code = sys::process_wait(p, sys::after_ms(600_000));
                        println!("{} exited with {code:?}", argv.join(" "));
                        ok &= code.ok() == want;
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

/// Starts the Linux program `argv[0]` (a module) with a pcid connection in its bootstrap message.
fn launch(b: &Bootstrap, argv: &[&str], pci: Handle, extra: &[(alloc::string::String, Handle)]) -> Option<Handle> {
    let Some(img) = b.handle(&alloc::format!("module/{}", argv[0])) else {
        println!("init.conf: no module {}", argv[0]);
        return None;
    };
    let mut bb = Builder::default().handle("pci", pci);
    for (name, h) in extra {
        bb = bb.handle(name, *h);
    }
    let boot = bb.send().ok()?;
    match sys::spawn_linux(img, argv, &["DROSS_BOOTSTRAP=1", "RUST_BACKTRACE=0"], Some(boot)) {
        Ok(p) => Some(p),
        Err(e) => {
            println!("{}: {e:?}", argv.join(" "));
            None
        }
    }
}
