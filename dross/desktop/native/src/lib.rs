//! Dross channel adapter. Native programs receive only pre-created capabilities;
//! this crate neither enumerates devices nor alters the boot supervisor.
#![no_std]
extern crate alloc;

use alloc::{format, string::String, vec, vec::Vec};
use dross_rt::{Error, Handle, bootstrap::Bootstrap, sys};
use oscar_desktop::{
    app::Trash,
    protocol::{Fault, MAX_PACKET, Packet},
    shell::Environment,
};

pub fn send(ch: Handle, packet: &Packet) -> Result<(), Error> {
    let bytes = packet.encode().map_err(|_| Error::InvalidArgs)?;
    sys::channel_write(ch, &bytes, &[])
}
pub fn read(ch: Handle, bytes: &mut [u8], deadline: u64) -> Result<usize, Error> {
    let mut handles = [Handle(0); dross_abi::MSG_HANDLES];
    let (n, nh) = sys::channel_read(ch, bytes, &mut handles, deadline)?;
    for &h in &handles[..nh] {
        let _ = sys::close(h);
    }
    if nh != 0 {
        return Err(Error::InvalidArgs);
    }
    Ok(n)
}
pub fn idle(error: Error) -> bool {
    matches!(error, Error::ShouldWait | Error::TimedOut)
}

/// The module directory is the native shell's read-only namespace. No invented files.
struct Modules {
    entries: Vec<(String, Handle)>,
}
impl Environment for Modules {
    fn pwd(&self) -> String {
        String::from("/")
    }
    fn cd(&mut self, path: &str) -> Result<(), String> {
        if matches!(path, "/" | "." | "..") {
            Ok(())
        } else {
            Err(String::from("directory not found"))
        }
    }
    fn list(&mut self, path: &str) -> Result<String, String> {
        self.cd(path)?;
        Ok(self
            .entries
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join("\n"))
    }
    fn read(&mut self, path: &str) -> Result<String, String> {
        let name = path.trim_start_matches('/');
        let h = self
            .entries
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, h)| *h)
            .ok_or_else(|| String::from("file not found"))?;
        let len = sys::mem_size(h).map_err(|e| format!("{e:?}"))?;
        if len == 0 {
            return Ok(String::new());
        }
        let at = sys::vm_map(h, 0, len, 0, dross_abi::vm::READ).map_err(|e| format!("{e:?}"))?;
        // SAFETY: the kernel validated this read-only memory object mapping. It is
        // live until vm_unmap below; the string owns a copy before that unmap.
        let bytes =
            unsafe { core::slice::from_raw_parts(at as *const u8, len.min(32 * 1024) as usize) };
        let result = core::str::from_utf8(bytes)
            .map(String::from)
            .map_err(|_| String::from("file is not UTF-8"));
        let _ = sys::vm_unmap(at, len);
        result
    }
}

pub fn client_main(bootstrap: Handle, external: bool) -> i32 {
    let b = Bootstrap::read(bootstrap).expect("desktop bootstrap");
    let _ = sys::close(bootstrap);
    let window = b.handle("window").expect("window channel");
    let stream = if external {
        Some(b.handle("terminal").expect("terminal byte stream"))
    } else {
        None
    };
    let width = b
        .value_u64("width")
        .unwrap_or(760)
        .try_into()
        .expect("width");
    let height = b
        .value_u64("height")
        .unwrap_or(500)
        .try_into()
        .expect("height");
    let mut env = Modules {
        entries: b
            .handles
            .iter()
            .filter_map(|(n, h)| n.strip_prefix("file/").map(|n| (String::from(n), *h)))
            .collect(),
    };
    let mut app = Trash::new(width, height).expect("client dimensions");
    if external {
        app.use_byte_stream();
    }
    let notify = sys::notify_create().expect("client notification");
    sys::channel_bind(window, notify, 1).expect("bind window");
    if let Some(ch) = stream {
        sys::channel_bind(ch, notify, 2).expect("bind terminal");
    }
    send(window, &app.create()).expect("create window");
    let mut bytes = vec![0; MAX_PACKET];
    let mut pending = Vec::new();
    loop {
        // Bounded work per source, so an output flood cannot starve keyboard or paint.
        for _ in 0..8 {
            match read(window, &mut bytes, 0) {
                Ok(n) => {
                    let packet = Packet::decode(&bytes[..n]).expect("window packet");
                    app.receive(packet, &mut env).expect("window response");
                }
                Err(e) if idle(e) => break,
                Err(Error::PeerClosed) => return 0,
                Err(_) => return 1,
            }
        }
        if let Some(ch) = stream {
            for _ in 0..8 {
                match read(ch, &mut bytes, 0) {
                    Ok(n) => app.feed(&bytes[..n]),
                    Err(e) if idle(e) => break,
                    Err(Error::PeerClosed) => {
                        app.closed = true;
                        break;
                    }
                    Err(_) => return 1,
                }
            }
            if pending.is_empty() {
                pending = app.take_input();
            }
            if !pending.is_empty() {
                match sys::channel_write(ch, &pending, &[]) {
                    Ok(()) => pending.clear(),
                    Err(Error::ShouldWait) => {}
                    Err(_) => return 1,
                }
            }
        }
        if app.closed {
            let _ = send(window, &app.close_packet());
            let _ = sys::close(window);
            return 0;
        }
        // At most 36 packets per frame, below CHANNEL_DEPTH=256. If the server is
        // stalled, abandon this connection instead of blocking or queueing frames.
        if app
            .frame(&mut env, |p| send(window, &p).map_err(|_| Fault::Limit))
            .is_err()
        {
            return 1;
        }
        match sys::notify_wait(notify, sys::after_ms(16)) {
            Ok(_) => {}
            Err(e) if idle(e) => {}
            Err(_) => return 1,
        }
    }
}
