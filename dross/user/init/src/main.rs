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
    let drivers = ["edutest"];
    let mut driver_ends = alloc::vec::Vec::new();
    for d in drivers {
        let (mine, theirs) = sys::channel_create().unwrap();
        pb = pb.handle(&alloc::format!("client/{d}"), mine);
        driver_ends.push(theirs);
    }
    bootstrap::spawn(pcid, "pcid", pb, u64::MAX).expect("spawn pcid");
    for (d, end) in drivers.iter().zip(driver_ends) {
        let Some(img) = b.handle(&alloc::format!("module/{d}")) else { continue };
        let p = bootstrap::spawn(img, d, Builder::default().handle("pci", end), u64::MAX).expect("spawn driver");
        let code = sys::process_wait(p, dross_rt::FOREVER);
        println!("{d} exited with {code:?}");
        ok &= code == Ok(0);
    }
    println!("{}", if ok { "all checks passed" } else { "checks FAILED" });
    sys::system_exit(system, if ok { 0 } else { 1 })
}
