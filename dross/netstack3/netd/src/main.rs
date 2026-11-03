//! One native Netstack3 engine, with the private ONET Unix adapter and explicit
//! native factory/socket channels. No driver discovery or PCI authority.
#![no_std]
#![no_main]
extern crate alloc;
mod ethernet;
mod legacy;
mod native;
mod seed;
mod service;
mod wait;
use dross_rt::Handle;
dross_rt::main!(main);
fn main(boot: Handle) -> i32 {
    match service::run(boot) {
        Ok(()) => 0,
        Err(error) => {
            dross_rt::sys::log(&alloc::format!("netd: stopped: {error:?}"));
            1
        }
    }
}
