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
    println!("{}", if ok { "all checks passed" } else { "checks FAILED" });
    sys::system_exit(system, if ok { 0 } else { 1 })
}
