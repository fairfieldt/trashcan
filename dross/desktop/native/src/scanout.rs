//! Provisional exclusive current-mode display channel. The capability is provided
//! by the supervisor; application/window channels cannot reach it.
use dross_rt::{Error, Handle, sys};
pub struct Scanout {
    channel: Handle,
    memory: Handle,
    at: u64,
    bytes: u64,
    width: u32,
    height: u32,
    pitch: u32,
    pub frame: u32,
    pub presents: u64,
    pub index: u32,
    released: bool,
}
impl Scanout {
    pub fn acquire(channel: Handle) -> Result<Self, Error> {
        let result = Self::acquire_inner(channel);
        if result.is_err() {
            let _ = sys::close(channel);
        }
        result
    }
    fn acquire_inner(channel: Handle) -> Result<Self, Error> {
        let mut request = [0u8; 8];
        request[4..].copy_from_slice(&u32::MAX.to_le_bytes());
        sys::channel_write(channel, &request, &[])?;
        let mut reply = [0u8; 48];
        let mut handles = [Handle(0); 64];
        let (n, nh) = sys::channel_read(channel, &mut reply, &mut handles, sys::after_ms(2000))?;
        let status = i64::from_le_bytes(reply[..8].try_into().unwrap());
        if status != 0 || n != 48 || nh != 1 {
            for &h in &handles[..nh] {
                let _ = sys::close(h);
            }
            return Err(Error::NoResources);
        }
        let word = |at| u32::from_le_bytes(reply[at..at + 4].try_into().unwrap());
        let (index, width, height, pitch, frame) =
            (word(8), word(12), word(16), word(20), word(32));
        let bytes = u64::from_le_bytes(reply[40..48].try_into().unwrap());
        if width == 0
            || height == 0
            || pitch < width
            || width > 4096
            || height > 4096
            || bytes < pitch as u64 * height as u64 * 4
            || bytes > 64 << 20
        {
            let _ = sys::close(handles[0]);
            return Err(Error::InvalidArgs);
        }
        let memory = handles[0];
        let at = match sys::vm_map(memory, 0, bytes, 0, 3) {
            Ok(at) => at,
            Err(error) => {
                let _ = sys::close(memory);
                return Err(error);
            }
        };
        Ok(Self {
            channel,
            memory,
            at,
            bytes,
            width,
            height,
            pitch,
            frame,
            index,
            released: false,
            presents: 0,
        })
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    /// Export only this owner's process mapping to a native-memory GPU import.
    /// The server retains the allocation until copy and scanout completion.
    pub fn shared_memory(&self) -> (*mut u8, u64, u32) {
        (self.at as *mut u8, self.bytes, self.pitch)
    }
    /// Publish staging only after the compositor's GPU completion fence.
    pub fn present_ready(&mut self) -> Result<u32, Error> {
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        let mut request = [0u8; 8];
        request[..4].copy_from_slice(&1u32.to_le_bytes());
        sys::channel_write(self.channel, &request, &[])?;
        let mut reply = [0u8; 12];
        let mut handles = [Handle(0); 64];
        let (n, nh) =
            sys::channel_read(self.channel, &mut reply, &mut handles, sys::after_ms(1000))?;
        for &h in &handles[..nh] {
            let _ = sys::close(h);
        }
        if n != 12 || nh != 0 || i64::from_le_bytes(reply[..8].try_into().unwrap()) != 0 {
            return Err(Error::NoResources);
        }
        let frame = u32::from_le_bytes(reply[8..].try_into().unwrap());
        if frame == self.frame {
            return Err(Error::InvalidArgs);
        }
        self.frame = frame;
        self.presents += 1;
        Ok(self.frame)
    }
    /// Restore the retained firmware plane and wait for driver acknowledgement.
    pub fn release(&mut self) -> Result<(), Error> {
        if self.released {
            return Ok(());
        }
        self.released = true;
        let mut request = [0u8; 8];
        request[..4].copy_from_slice(&2u32.to_le_bytes());
        sys::channel_write(self.channel, &request, &[])?;
        let mut reply = [0u8; 8];
        let mut handles = [Handle(0); 64];
        let (n, nh) =
            sys::channel_read(self.channel, &mut reply, &mut handles, sys::after_ms(1000))?;
        for &h in &handles[..nh] {
            let _ = sys::close(h);
        }
        if n != 8 || nh != 0 || i64::from_le_bytes(reply) != 0 {
            return Err(Error::NoResources);
        }
        Ok(())
    }
}
impl Drop for Scanout {
    fn drop(&mut self) {
        let _ = self.release();
        let _ = sys::close(self.channel);
        let _ = sys::vm_unmap(self.at, self.bytes);
        let _ = sys::close(self.memory);
    }
}
