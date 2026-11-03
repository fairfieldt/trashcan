//! Session supervisor. The caller provisions display, seat and program images.
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
    if let Some(scanout) = b.handle("scanout") {
        wb = wb.handle("scanout", scanout);
    }
    for name in ["width", "height", "stride", "format", "offset", "gpu-bus"] {
        if let Some(value) = b.value(name) {
            wb = wb.value(name, value);
        }
    }
    // More applications' window connections (`client/NAME`, e.g. the chat window, a Linux program init starts).
    for (n, h) in &b.handles {
        if n.starts_with("client/") {
            wb = wb.handle(n, *h);
        }
    }
    // Supplying the GPU server image explicitly selects the Linux-compat adapter.
    // Applications still receive the same native channel and surface capabilities.
    let windowd = if let Some(image) = b.handle("module/oscar-windowd-gpu") {
        let environment = graphics_environment(&b);
        let env: alloc::vec::Vec<&str> = environment.iter().map(|s| s.as_str()).collect();
        let child = wb.send().expect("windowd bootstrap");
        sys::spawn_linux(image, &["oscar-windowd-gpu"], &env, Some(child))
            .expect("spawn GPU windowd")
    } else {
        bootstrap::spawn(
            b.handle("module/oscar-windowd").expect("windowd image"),
            "oscar-windowd",
            wb,
            u64::MAX,
        )
        .expect("spawn windowd")
    };
    let doom = b.value("app") == Some("doom");
    let vulkan = b.value("app") == Some("vulkan");
    let external = b.value("app") == Some("terminal");
    let name = if vulkan {
        "oscar-vkdraw"
    } else if doom {
        "oscar-doom"
    } else if external {
        "oscar-terminal"
    } else {
        "trash"
    };
    let image = b
        .handle(if vulkan {
            "module/oscar-vkdraw"
        } else if doom {
            "module/oscar-doom"
        } else if external {
            "module/oscar-terminal"
        } else {
            "module/trash"
        })
        .expect("app image");
    let mut ab = Builder::default().handle("window", client);
    if let Some(bus) = b.value("gpu-bus") {
        ab = ab.value("gpu-bus", bus);
    }
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
    let started = if vulkan {
        let environment = graphics_environment(&b);
        let env: alloc::vec::Vec<&str> = environment.iter().map(|s| s.as_str()).collect();
        let boot = ab.send().expect("Vulkan app bootstrap");
        sys::spawn_linux(image, &["oscar-vkdraw"], &env, Some(boot))
    } else if doom {
        let boot = ab.send().expect("Doom bootstrap");
        sys::spawn_linux(
            image,
            &[
                "oscar-doom",
                "-iwad",
                b.value("iwad").unwrap_or("/freedoom1.wad"),
                "-warp",
                "1",
                "1",
                "-nosound",
            ],
            &[],
            Some(boot),
        )
    } else {
        bootstrap::spawn(image, name, ab, u64::MAX)
    };
    let app = match started {
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

fn graphics_environment(b: &Bootstrap) -> alloc::vec::Vec<alloc::string::String> {
    let mut environment: alloc::vec::Vec<alloc::string::String> = [
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "VK_DRIVER_FILES",
        "GPUD_SOCKET",
    ]
    .iter()
    .filter_map(|key| b.value(key).map(|value| alloc::format!("{key}={value}")))
    .collect();
    environment.push("DROSS_BOOTSTRAP=1".into());
    environment
}
