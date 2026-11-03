//! Explicit opt-in process graph. The existing init/boot configuration never loads it.
#![no_std]
#![no_main]
extern crate alloc;
use dross_abi::rights;
use dross_rt::{
    FOREVER, Handle,
    bootstrap::{self, Bootstrap, Builder},
    sys,
};
dross_rt::main!(main);

fn main(bootstrap: Handle) -> i32 {
    let b = Bootstrap::read(bootstrap).expect("session bootstrap");
    let _ = sys::close(bootstrap);
    let (server, client) = sys::channel_create().expect("window connection");
    let (control, lifetime) = sys::channel_create().expect("server lifetime");
    let mut wb = Builder::default()
        .handle("display", b.handle("display").expect("exclusive display"))
        .handle("input", b.handle("input").expect("seat input"))
        .handle("client/trash", server)
        .handle("control", lifetime);
    for name in ["width", "height", "stride", "format", "offset"] {
        if let Some(value) = b.value(name) {
            wb = wb.value(name, value);
        }
    }
    let windowd = bootstrap::spawn(
        b.handle("module/oscar-windowd").expect("windowd image"),
        "oscar-windowd",
        wb,
        u64::MAX,
    )
    .expect("spawn windowd");
    let external = b.value("app") == Some("terminal");
    let name = if external { "oscar-terminal" } else { "trash" };
    let image = b
        .handle(if external {
            "module/oscar-terminal"
        } else {
            "module/trash"
        })
        .expect("app image");
    let mut ab = Builder::default().handle("window", client);
    if external {
        ab = ab.handle("terminal", b.handle("terminal").expect("terminal stream"));
    }
    for (n, h) in &b.handles {
        if n.starts_with("file/") {
            ab = ab.handle(
                n,
                sys::dup(*h, rights::READ | rights::MAP | rights::TRANSFER)
                    .expect("read-only file"),
            );
        }
    }
    if let Some(width) = b.value("client-width") {
        ab = ab.value("width", width);
    }
    if let Some(height) = b.value("client-height") {
        ab = ab.value("height", height);
    }
    let app = match bootstrap::spawn(image, name, ab, u64::MAX) {
        Ok(p) => p,
        Err(_) => {
            let _ = sys::close(control);
            let _ = sys::process_wait(windowd, FOREVER);
            return 1;
        }
    };
    let result = sys::process_wait(app, FOREVER).unwrap_or(1);
    let _ = sys::close(control);
    let _ = sys::process_wait(windowd, FOREVER);
    result as i32
}
