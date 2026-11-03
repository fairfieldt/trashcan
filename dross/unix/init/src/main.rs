#![no_std]
#![no_main]
extern crate alloc;
use dross_rt::{
    Handle,
    bootstrap::{self, Bootstrap, Builder},
    println, sys,
};
dross_rt::main!(main);
fn main(handle: Handle) -> i32 {
    let b = Bootstrap::read(handle).expect("unix bootstrap");
    let (pci_server, pci_client) = sys::channel_create().unwrap();
    let physical = b.handle("resource/physical").unwrap();
    let ecam = b.value_u64("ecam").expect("ECAM");
    let buses = b.value("ecam-buses").unwrap();
    let last: u64 = buses.split(' ').nth(1).unwrap().parse().unwrap();
    let dup = |name| sys::dup(b.handle(name).unwrap(), dross_abi::rights::ALL).unwrap();
    let pb = Builder::default()
        .handle(
            "resource/ecam",
            sys::resource_sub(physical, ecam, (last + 1) << 20).unwrap(),
        )
        .handle("resource/physical", dup("resource/physical"))
        .handle("resource/irq", dup("resource/irq"))
        .handle("resource/iommu", dup("resource/iommu"))
        .handle("client/netd", pci_server)
        .value("ecam", b.value("ecam").unwrap())
        .value("ecam-buses", buses);
    bootstrap::spawn(b.handle("module/pcid").unwrap(), "pcid", pb, u64::MAX).unwrap();
    let (service, daemon) = sys::channel_create().unwrap();
    let (report, ready) = sys::channel_create().unwrap();
    // The supervisor authorizes registration; netd never receives the system resource.
    let (status, _value): (i64, u64);
    // SAFETY: native dross syscall ABI, arguments are capabilities owned by this init.
    unsafe {
        core::arch::asm!("syscall", inlateout("rax") (dross_abi::sys::UNIX_NET_REGISTER | dross_abi::NATIVE) => status,
        in("rdi") b.handle("resource/system").unwrap().0 as u64, in("rsi") service.0 as u64, lateout("rdx") _value, out("rcx") _, out("r11") _, options(nostack));
    }
    assert_eq!(status, 0, "register network service");
    let (driver, ethernet) = sys::channel_create().unwrap();
    let db = Builder::default()
        .handle("pci", pci_client)
        .handle("ethernet", driver);
    bootstrap::spawn(b.handle("module/unix-nic").unwrap(), "nic", db, u64::MAX).unwrap();
    let nb = Builder::default()
        .handle("ethernet", ethernet)
        .handle("service", daemon)
        .handle("ready", ready);
    bootstrap::spawn(b.handle("module/unix-netd").unwrap(), "netd", nb, u64::MAX).unwrap();
    let mut bytes = [0; 64];
    sys::channel_read(report, &mut bytes, &mut [], sys::after_ms(10000)).expect("netd ready");
    let mut bytes = [0; 4096];
    let (front, stream) = sys::channel_create().unwrap();
    let message = Builder::default()
        .handle("terminal", stream)
        .value("columns", "80")
        .value("rows", "24")
        .send()
        .unwrap();
    let local = sys::spawn_linux(
        b.handle("module/unix-terminal-bridge").unwrap(),
        &["unix-terminal-bridge"],
        &["PATH=/bin:/usr/bin", "HOME=/root", "TERM=xterm"],
        Some(message),
    )
    .unwrap();
    sys::channel_write(front, b"printf 'LOCAL_%s\\n' BRIDGE_OK\r", &[]).unwrap();
    let deadline = sys::after_ms(10000);
    let mut transcript = alloc::vec::Vec::new();
    loop {
        let (n, _) =
            sys::channel_read(front, &mut bytes, &mut [], deadline).expect("local terminal output");
        transcript.extend_from_slice(&bytes[..n]);
        assert!(transcript.len() < 8192);
        if transcript.windows(17).any(|v| v == b"LOCAL_BRIDGE_OK\r\n") {
            break;
        }
    }
    sys::channel_write(front, b"exit\r", &[]).unwrap();
    assert_eq!(sys::process_wait(local, sys::after_ms(5000)), Ok(0));
    let _ = sys::close(front);
    println!("local PTY bridge passed");
    let image = b.handle("module/unix-probe").expect("probe image");
    let p = sys::spawn_linux(
        image,
        &["/bin/unix-probe"],
        &[
            "PATH=/bin:/usr/bin:/sbin:/usr/sbin",
            "HOME=/root",
            "TERM=xterm",
        ],
        None,
    )
    .expect("spawn probe");
    let code = sys::process_wait(p, sys::after_ms(120000));
    println!("unix probe exited {code:?}");
    sys::system_exit(
        b.handle("resource/system").unwrap(),
        if code == Ok(0) { 0 } else { 1 },
    )
}
