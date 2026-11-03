#![no_std]
#![no_main]
extern crate alloc;

use alloc::{collections::VecDeque, vec, vec::Vec};
use dross_rt::{Error, Handle, bootstrap::Bootstrap, sys};
use oscar_desktop::{
    canvas::Canvas,
    protocol::{MAX_PACKET, Message, Packet},
    server::{Client, Delivery, MAX_WINDOWS, Server},
};
use oscar_desktop_native::{idle, read, send};
dross_rt::main!(main);

struct Connection {
    id: Client,
    channel: Handle,
    pending: VecDeque<Packet>,
    dead: bool,
}
fn queue(connections: &mut [Connection], out: Vec<Delivery>) {
    for delivery in out {
        if let Some(c) = connections
            .iter_mut()
            .find(|c| c.id == delivery.client && !c.dead)
        {
            if c.pending.len() >= 32 {
                c.dead = true;
            } else {
                c.pending.push_back(delivery.packet);
            }
        }
    }
}
struct Display {
    at: *mut u32,
    stride: usize,
    rgb: bool,
}
impl Display {
    fn present(&mut self, canvas: &Canvas) {
        for (y, line) in canvas.pixels().chunks(canvas.width() as usize).enumerate() {
            for (x, &pixel) in line.iter().enumerate() {
                let pixel = if self.rgb {
                    ((pixel & 255) << 16) | (pixel & 0xff00) | ((pixel >> 16) & 255)
                } else {
                    pixel
                };
                // SAFETY: main validated stride, height and mapping size. This is
                // the exclusive display capability; writes to MMIO are volatile.
                unsafe {
                    self.at.add(y * self.stride + x).write_volatile(pixel);
                }
            }
        }
    }
}
fn main(bootstrap: Handle) -> i32 {
    let b = Bootstrap::read(bootstrap).expect("windowd bootstrap");
    let _ = sys::close(bootstrap);
    let width: u32 = b
        .value_u64("width")
        .expect("display width")
        .try_into()
        .expect("width");
    let height: u32 = b
        .value_u64("height")
        .expect("display height")
        .try_into()
        .expect("height");
    let stride = b.value_u64("stride").expect("display stride");
    let rgb = match b.value("format").expect("display format") {
        "rgb" => true,
        "bgr" => false,
        _ => return 1,
    };
    let mut server = Server::new(width, height).expect("display dimensions");
    let framebuffer = b.handle("display").expect("exclusive display memory");
    let len = sys::mem_size(framebuffer).expect("framebuffer size");
    let offset = b.value_u64("offset").unwrap_or(0);
    let Some(required) = stride
        .checked_mul(height as u64)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(offset))
    else {
        return 1;
    };
    if stride < width as u64 || required > len || !offset.is_multiple_of(4) {
        return 1;
    }
    let at = sys::vm_map(
        framebuffer,
        0,
        len,
        0,
        dross_abi::vm::READ | dross_abi::vm::WRITE,
    )
    .expect("map framebuffer");
    let mut display = Display {
        at: (at + offset) as *mut u32,
        stride: stride as usize,
        rgb,
    };
    let notify = sys::notify_create().expect("seat notification");
    let input = b.handle("input").expect("trusted seat channel");
    sys::channel_bind(input, notify, 1).expect("bind seat");
    let control = b.handle("control").expect("supervisor lifetime channel");
    sys::channel_bind(control, notify, 1 << 63).expect("bind supervisor");
    let mut connections = Vec::new();
    for (name, h) in &b.handles {
        if name.starts_with("client/") {
            if connections.len() == MAX_WINDOWS {
                return 1;
            }
            let id = connections.len() as u32 + 1;
            sys::channel_bind(*h, notify, 1 << id).expect("bind client");
            connections.push(Connection {
                id,
                channel: *h,
                pending: VecDeque::new(),
                dead: false,
            });
        }
    }
    let mut bytes = vec![0; MAX_PACKET];
    loop {
        match read(control, &mut bytes, 0) {
            Err(e) if idle(e) => {}
            Err(Error::PeerClosed) => return 0,
            _ => return 1,
        }
        for _ in 0..16 {
            match read(input, &mut bytes, 0) {
                Ok(n) => match Packet::decode(&bytes[..n]) {
                    Ok(Packet {
                        message: Message::Input(input),
                        ..
                    }) => queue(&mut connections, server.input(input)),
                    _ => return 1,
                },
                Err(e) if idle(e) => break,
                Err(_) => return 1,
            }
        }
        for i in 0..connections.len() {
            if connections[i].dead {
                continue;
            }
            for _ in 0..8 {
                match read(connections[i].channel, &mut bytes, 0) {
                    Ok(n) => match Packet::decode(&bytes[..n]) {
                        Ok(p) => {
                            let id = connections[i].id;
                            queue(&mut connections, server.request(id, p));
                        }
                        Err(_) => {
                            connections[i].dead = true;
                            break;
                        }
                    },
                    Err(e) if idle(e) => break,
                    Err(_) => {
                        connections[i].dead = true;
                        break;
                    }
                }
            }
            while let Some(packet) = connections[i].pending.front() {
                match send(connections[i].channel, packet) {
                    Ok(()) => {
                        connections[i].pending.pop_front();
                    }
                    Err(Error::ShouldWait) => break,
                    Err(_) => {
                        connections[i].dead = true;
                        break;
                    }
                }
            }
        }
        for i in 0..connections.len() {
            if connections[i].dead && connections[i].channel.0 != 0 {
                let out = server.disconnect(connections[i].id);
                queue(&mut connections, out);
                let _ = sys::close(connections[i].channel);
                connections[i].channel = Handle(0);
                connections[i].pending.clear();
            }
        }
        if server.is_dirty() {
            display.present(server.render());
        }
        // A deadline also retries backpressure: channel bindings signal readable,
        // not writable. No GPU queues or waits on unbounded host events.
        match sys::notify_wait(notify, sys::after_ms(16)) {
            Ok(_) => {}
            Err(e) if idle(e) => {}
            Err(_) => return 1,
        }
    }
}
