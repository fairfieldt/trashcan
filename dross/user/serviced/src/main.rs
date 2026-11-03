#![no_std]
#![no_main]
extern crate alloc;
use dross_rt::{
    Handle,
    deadline::{Deadline, Duration},
    owned::{self, OwnedHandle},
    protocol::{bootstrap::BootstrapV2, directory::Server},
};
dross_rt::main!(main);
fn main(boot: Handle) -> i32 {
    // Entry transfers the process's unique bootstrap slot to this owner.
    let boot = unsafe { OwnedHandle::from_raw(boot) };
    let result = (|| {
        let message = owned::receive(boot.as_handle(), Deadline::from_now(Duration::from_secs(5)))
            .map_err(|_| ())?;
        let bootstrap = BootstrapV2::decode(message).map_err(|_| ())?;
        let server = Server::from_bootstrap(bootstrap).map_err(|_| ())?;
        drop(boot);
        server.run().map_err(|_| ())
    })();
    if result.is_ok() {
        0
    } else {
        dross_rt::sys::log("serviced: bootstrap or transport failed");
        1
    }
}
