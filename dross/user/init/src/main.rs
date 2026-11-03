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
        let code = sys::spawn_linux(img, &[m.as_str(), "--test-threads=4"], &["RUST_BACKTRACE=0"], None).and_then(|p| sys::process_wait(p, sys::after_ms(600_000)));
        println!("{m} exited with {code:?} after {} ms", (sys::clock() - t0) / 1_000_000);
        ok &= code == Ok(0);
    }

    // Linux programs from init.conf: `start MODULE ARGS...` runs one; `check MODULE CODE ARGS...` runs one and
    // requires its exit code. Each gets a connection to pcid (bootstrap handle `pci`).
    let conf = b.handle("module/init.conf").map(read_module).unwrap_or_default();
    let lines: alloc::vec::Vec<alloc::vec::Vec<&str>> =
        conf.lines().map(|l| l.split_whitespace().collect::<alloc::vec::Vec<_>>()).filter(|w| matches!(w.first(), Some(&"start") | Some(&"check"))).collect();

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
    let drivers = ["edutest", "usbd"];
    let mut driver_ends = alloc::vec::Vec::new();
    for d in drivers {
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
    // Under QEMU the run's harness types on the emulated keyboard and moves the mouse; usbd reports when it saw
    // both, and init waits for that.
    let test = b.value("hypervisor") == Some("1");
    let mut usb_report = None;
    for (d, end) in drivers.iter().zip(driver_ends) {
        let Some(img) = b.handle(&alloc::format!("module/{d}")) else { continue };
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
    for (words, end) in lines.iter().zip(conf_ends) {
        let check = words[0] == "check";
        let module = words[1];
        let (want, args) = if check { (words.get(2).and_then(|c| c.parse::<i64>().ok()), &words[3..]) } else { (None, &words[2..]) };
        let Some(img) = b.handle(&alloc::format!("module/{module}")) else {
            println!("init.conf: no module {module}");
            ok = false;
            continue;
        };
        let mut argv = alloc::vec![module];
        argv.extend_from_slice(args);
        let boot = Builder::default().handle("pci", end).send().unwrap();
        let p = match sys::spawn_linux(img, &argv, &["DROSS_BOOTSTRAP=1", "RUST_BACKTRACE=0"], Some(boot)) {
            Ok(p) => p,
            Err(e) => {
                println!("{}: {e:?}", argv.join(" "));
                ok = false;
                continue;
            }
        };
        if check {
            let code = sys::process_wait(p, sys::after_ms(600_000));
            println!("{} exited with {code:?}", argv.join(" "));
            ok &= code.ok() == want;
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
        let n = sys::notify_create().unwrap();
        let _ = sys::notify_wait(n, dross_rt::FOREVER);
    }
    println!("{}", if ok { "all checks passed" } else { "checks FAILED" });
    sys::system_exit(system, if ok { 0 } else { 1 })
}

/// A text module (init.conf) as a string.
fn read_module(h: Handle) -> alloc::string::String {
    let Ok(size) = sys::mem_size(h) else { return Default::default() };
    let Ok(at) = sys::vm_map(h, 0, size, 0, dross_abi::vm::READ) else { return Default::default() };
    // SAFETY: a read-only mapping of `size` bytes that stays.
    let bytes = unsafe { core::slice::from_raw_parts(at as *const u8, size as usize) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    alloc::string::String::from_utf8_lossy(&bytes[..end]).into_owned()
}
