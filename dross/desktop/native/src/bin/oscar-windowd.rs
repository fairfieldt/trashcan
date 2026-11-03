#![no_std]
#![no_main]
use dross_rt::{Handle, bootstrap::Bootstrap, sys};
dross_rt::main!(main);
fn main(bootstrap: Handle) -> i32 {
    let b = Bootstrap::read(bootstrap).expect("windowd bootstrap");
    let _ = sys::close(bootstrap);
    oscar_desktop_native::windowd::run(b, |server, display| {
        display.present(server.render());
        Ok(())
    })
}
