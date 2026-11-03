//! Provisional OSCR shared-window client. Native channels/memory are the ABI;
//! this module works in native and Linux-compat programs without Mesa or DRM.
use alloc::{vec, vec::Vec};
use dross_rt::{Error, Handle, sys};
use oscar_desktop::protocol::{Input, MAX_PACKET, Message, Packet};

pub enum Event {
    Input(Input),
    Focus(bool),
    Close,
}
pub struct Window {
    channel: Handle,
    id: u32,
    width: u32,
    height: u32,
    memory: Handle,
    at: u64,
    size: u64,
    serial: u32,
    in_flight: bool,
    bytes: Vec<u8>,
}
impl Window {
    pub fn open(channel: Handle, width: u32, height: u32, title: &str) -> Result<Self, Error> {
        let result = Self::open_inner(channel, width, height, title);
        if result.is_err() {
            let _ = sys::close(channel);
        }
        result
    }
    fn open_inner(channel: Handle, width: u32, height: u32, title: &str) -> Result<Self, Error> {
        if width < 120
            || height < 80
            || width > 2048
            || height > 2048
            || width as u64 * height as u64 > 512 * 1024
        {
            return Err(Error::InvalidArgs);
        }
        let packet = Packet::new(
            1,
            0,
            Message::CreateShared {
                width,
                height,
                title: title.into(),
            },
        );
        sys::channel_write(
            channel,
            &packet.encode().map_err(|_| Error::InvalidArgs)?,
            &[],
        )?;
        let mut bytes = vec![0; MAX_PACKET];
        let mut handles = [Handle(0); dross_abi::MSG_HANDLES];
        let (n, nh) = sys::channel_read(channel, &mut bytes, &mut handles, sys::after_ms(2000))?;
        let decoded = Packet::decode(&bytes[..n]);
        let valid = matches!(&decoded,Ok(Packet {serial:1,window,message:Message::CreatedShared{width:w,height:h}}) if *window!=0 && *w==width && *h==height);
        if !valid || nh != 1 {
            for &h in &handles[..nh] {
                let _ = sys::close(h);
            }
            return Err(Error::InvalidArgs);
        }
        let size = (width as u64 * height as u64 * 4).next_multiple_of(4096);
        let memory = handles[0];
        let at = match sys::mem_size(memory).and_then(|n| {
            if n == size {
                sys::vm_map(
                    memory,
                    0,
                    size,
                    0,
                    dross_abi::vm::READ | dross_abi::vm::WRITE,
                )
            } else {
                Err(Error::InvalidArgs)
            }
        }) {
            Ok(at) => at,
            Err(e) => {
                let _ = sys::close(memory);
                return Err(e);
            }
        };
        Ok(Self {
            channel,
            id: decoded.unwrap().window,
            width,
            height,
            memory,
            at,
            size,
            serial: 1,
            in_flight: false,
            bytes,
        })
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn can_present(&self) -> bool {
        !self.in_flight
    }
    /// Copies one complete frame into the shared buffer. Reuse is forbidden until
    /// FrameDone acknowledges a server-owned snapshot, independently of GPU/scanout.
    pub fn present(&mut self, pixels: &[u32]) -> Result<(), Error> {
        if self.in_flight {
            return Err(Error::ShouldWait);
        }
        if pixels.len() != (self.width * self.height) as usize {
            return Err(Error::InvalidArgs);
        }
        for (i, &p) in pixels.iter().enumerate() {
            // SAFETY: fixed-size server-owned memory mapped writable above.
            unsafe {
                (self.at as *mut u32).add(i).write_volatile(p);
            }
        }
        self.present_ready()
    }
    /// Mapping for a native-capable GPU import. Its backing belongs to windowd;
    /// callers must keep this Window alive while using its process mapping.
    pub fn shared_memory(&self) -> (*mut u8, u64) {
        (self.at as *mut u8, self.size)
    }
    /// Present a frame already written by the CPU or a completed GPU operation.
    /// Calling this before the GPU's completion fence is a protocol violation.
    pub fn present_ready(&mut self) -> Result<(), Error> {
        if self.in_flight {
            return Err(Error::ShouldWait);
        }
        let serial = self.serial.checked_add(1).ok_or(Error::NoResources)?;
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        let packet = Packet::new(serial, self.id, Message::PresentShared)
            .encode()
            .map_err(|_| Error::InvalidArgs)?;
        sys::channel_write(self.channel, &packet, &[])?;
        self.serial = serial;
        self.in_flight = true;
        Ok(())
    }
    pub fn poll(&mut self, deadline: u64) -> Result<Option<Event>, Error> {
        let mut handles = [Handle(0); dross_abi::MSG_HANDLES];
        let (n, nh) = sys::channel_read(self.channel, &mut self.bytes, &mut handles, deadline)?;
        for &h in &handles[..nh] {
            let _ = sys::close(h);
        }
        if nh != 0 {
            return Err(Error::InvalidArgs);
        }
        let p = Packet::decode(&self.bytes[..n]).map_err(|_| Error::InvalidArgs)?;
        if p.window != self.id {
            return Err(Error::InvalidArgs);
        }
        Ok(match p.message {
            Message::FrameDone if self.in_flight && p.serial == self.serial => {
                self.in_flight = false;
                None
            }
            Message::Input(input) => Some(Event::Input(input)),
            Message::Focus(f) => Some(Event::Focus(f)),
            Message::CloseRequested | Message::Closed => Some(Event::Close),
            _ => return Err(Error::InvalidArgs),
        })
    }
}
impl Drop for Window {
    fn drop(&mut self) {
        if let Ok(bytes) = Packet::new(self.serial, self.id, Message::Close).encode() {
            let _ = sys::channel_write(self.channel, &bytes, &[]);
        }
        let _ = sys::vm_unmap(self.at, self.size);
        let _ = sys::close(self.memory);
        let _ = sys::close(self.channel);
    }
}
