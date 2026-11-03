#![no_std]
#![no_main]
dross_rt::main!(main);
fn main(bootstrap: dross_rt::Handle) -> i32 {
    oscar_desktop_native::client_main(bootstrap, false)
}
