//! Test-image supervisor only: does not configure hardware or change installed init.
#![no_std]
#![no_main]
extern crate alloc;
use dross_rt::{
    Handle,
    bootstrap::{Bootstrap, Builder},
    sys,
};
dross_rt::main!(main);
fn main(h: Handle) -> i32 {
    let b = Bootstrap::read(h).expect("test bootstrap");
    let _ = sys::close(h);
    let boot = Builder::default()
        .handle("display", b.handle("display").unwrap())
        .value("stride", b.value("display-stride").unwrap())
        .value("width", b.value("display-width").unwrap())
        .value("height", b.value("display-height").unwrap())
        .handle("windowd", b.handle("module/oscar-windowd").unwrap())
        .handle("doom", b.handle("module/oscar-doom").unwrap())
        .send()
        .unwrap();
    let process = sys::spawn_linux(
        b.handle("module/oscar-windowcheck").unwrap(),
        &["oscar-windowcheck"],
        &[],
        Some(boot),
    )
    .unwrap();
    let code = sys::process_wait(process, dross_rt::FOREVER).unwrap_or(-1);
    dross_rt::println!("desktop test: {}", if code == 0 { "ok" } else { "failed" });
    sys::system_exit(b.handle("resource/system").unwrap(), u64::from(code != 0))
}
