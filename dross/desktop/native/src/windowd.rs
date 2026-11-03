use crate::{idle, read};
use alloc::{collections::VecDeque, vec, vec::Vec};
use dross_rt::{Error, Handle, bootstrap::Bootstrap, sys};
use oscar_desktop::{
    canvas::Canvas,
    protocol::{MAX_PACKET, Message, Packet},
    server::{Client, Delivery, MAX_WINDOWS, Server},
};

struct Pending {
    packet: Packet,
    memory: Option<Handle>,
}
impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(h) = self.memory.take() {
            let _ = sys::close(h);
        }
    }
}
struct Surface {
    window: u32,
    memory: Handle,
    at: u64,
    size: u64,
    pixels: Vec<u32>,
}
impl Drop for Surface {
    fn drop(&mut self) {
        let _ = sys::vm_unmap(self.at, self.size);
        let _ = sys::close(self.memory);
    }
}
struct Connection {
    id: Client,
    channel: Handle,
    pending: VecDeque<Pending>,
    surfaces: Vec<Surface>,
    dead: bool,
}
impl Drop for Connection {
    fn drop(&mut self) {
        if self.channel.0 != 0 {
            let _ = sys::close(self.channel);
        }
    }
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
                let memory =
                    if let Message::CreatedShared { width, height } = delivery.packet.message {
                        let size = (width as u64 * height as u64 * 4).next_multiple_of(4096);
                        let allocate = (|| -> Result<_, Error> {
                            let memory = sys::mem_create(size, 0)?;
                            let at = match sys::vm_map(memory, 0, size, 0, dross_abi::vm::READ) {
                                Ok(at) => at,
                                Err(e) => {
                                    let _ = sys::close(memory);
                                    return Err(e);
                                }
                            };
                            let mut surface = Surface {
                                window: delivery.packet.window,
                                memory,
                                at,
                                size,
                                pixels: vec![0; (width * height) as usize],
                            };
                            let client = match sys::dup(
                                memory,
                                dross_abi::rights::READ
                                    | dross_abi::rights::WRITE
                                    | dross_abi::rights::MAP
                                    | dross_abi::rights::TRANSFER,
                            ) {
                                Ok(h) => h,
                                Err(e) => {
                                    return Err(e);
                                }
                            };
                            // Initialize only kernel-zeroed backing; no untrusted pixel data is followed as pointers.
                            surface.pixels.fill(0);
                            c.surfaces.push(surface);
                            Ok(client)
                        })();
                        match allocate {
                            Ok(h) => Some(h),
                            Err(_) => {
                                c.dead = true;
                                continue;
                            }
                        }
                    } else {
                        None
                    };
                if matches!(delivery.packet.message, Message::Closed) {
                    c.surfaces.retain(|s| s.window != delivery.packet.window);
                }
                c.pending.push_back(Pending {
                    packet: delivery.packet,
                    memory,
                });
            }
        }
    }
}
pub struct Display {
    at: *mut u32,
    stride: usize,
    rgb: bool,
    width: u32,
    height: u32,
}
impl Display {
    pub fn present(&mut self, canvas: &Canvas) {
        self.present_pixels(canvas.pixels(), canvas.width());
    }
    pub fn present_pixels(&mut self, pixels: &[u32], width: u32) {
        assert_eq!(width, self.width);
        assert_eq!(pixels.len(), self.width as usize * self.height as usize);
        for (y, line) in pixels.chunks(width as usize).enumerate() {
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
pub fn run(
    b: Bootstrap,
    mut render: impl FnMut(&mut Server, &mut Display) -> Result<(), Error>,
) -> i32 {
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
        width,
        height,
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
                surfaces: Vec::new(),
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
                            let out = if matches!(p.message, Message::PresentShared) {
                                let result = connections[i]
                                    .surfaces
                                    .iter_mut()
                                    .find(|s| s.window == p.window)
                                    .ok_or(oscar_desktop::protocol::Fault::Ownership)
                                    .and_then(|surface| {
                                        for (j, pixel) in surface.pixels.iter_mut().enumerate() {
                                            // SAFETY: this server allocated a fixed-size mapping; volatile reads
                                            // snapshot client-owned pixels, never scene data or addresses.
                                            *pixel = unsafe {
                                                (surface.at as *const u32).add(j).read_volatile()
                                            };
                                        }
                                        server.present_shared(
                                            id,
                                            p.window,
                                            p.serial,
                                            &surface.pixels,
                                        )
                                    });
                                match result {
                                    Ok(d) => vec![d],
                                    Err(e) => vec![Delivery {
                                        client: id,
                                        packet: Packet::new(p.serial, p.window, Message::Error(e)),
                                    }],
                                }
                            } else {
                                server.request(id, p)
                            };
                            queue(&mut connections, out);
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
            while let Some(pending) = connections[i].pending.front() {
                let encoded = match pending.packet.encode() {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        connections[i].dead = true;
                        break;
                    }
                };
                let handles = pending.memory.as_slice();
                match sys::channel_write(connections[i].channel, &encoded, handles) {
                    Ok(()) => {
                        let mut sent = connections[i].pending.pop_front().unwrap();
                        sent.memory = None;
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
                connections[i].surfaces.clear();
            }
        }
        if server.is_dirty() && render(&mut server, &mut display).is_err() {
            return 1;
        }
        // A deadline also retries backpressure: channel bindings signal readable,
        // not writable. Rendering adapters must bound any GPU waits.
        match sys::notify_wait(notify, sys::after_ms(16)) {
            Ok(_) => {}
            Err(e) if idle(e) => {}
            Err(_) => return 1,
        }
    }
}
