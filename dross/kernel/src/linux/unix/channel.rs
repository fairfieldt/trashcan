//! A byte-stream fd over an explicitly supplied native channel capability.
use super::*;
use crate::ipc::{ChannelEnd, Message, ReadError};
use alloc::collections::VecDeque;
pub struct Stream {
    end: Arc<ChannelEnd>,
    pending: Mutex<VecDeque<u8>>,
    reader: WaitCell<bool>,
}
struct Guard<'a>(&'a WaitCell<bool>);
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.0.update(|v| *v = false);
    }
}
impl Stream {
    pub fn new(end: Arc<ChannelEnd>) -> Self {
        Self {
            end,
            pending: Mutex::new(VecDeque::new()),
            reader: WaitCell::new(false),
        }
    }
    pub fn read(&self, size: usize, nb: bool) -> Result<Vec<u8>, Errno> {
        if size == 0 {
            return Ok(Vec::new());
        }
        self.reader
            .wait(if nb { 0 } else { u64::MAX }, |busy| {
                if !*busy {
                    *busy = true;
                    Some(())
                } else {
                    None
                }
            })
            .map_err(|e| {
                if e == WaitError::TimedOut {
                    EAGAIN
                } else {
                    EINTR
                }
            })?;
        let _guard = Guard(&self.reader);
        if self.pending.lock().is_empty() {
            match self.end.read(
                dross_abi::MSG_BYTES,
                dross_abi::MSG_HANDLES,
                if nb { 0 } else { u64::MAX },
            ) {
                Ok(message) => {
                    if !message.handles.is_empty() {
                        return Err(EINVAL);
                    }
                    self.pending.lock().extend(message.bytes);
                }
                Err(ReadError::PeerClosed) => return Ok(Vec::new()),
                Err(ReadError::Wait(WaitError::TimedOut)) => return Err(EAGAIN),
                Err(ReadError::Wait(_)) => return Err(EINTR),
                Err(_) => return Err(EINVAL),
            }
        }
        let mut pending = self.pending.lock();
        let size = size.min(pending.len());
        Ok(pending.drain(..size).collect())
    }
    pub fn write(&self, bytes: &[u8], nb: bool) -> Result<usize, Errno> {
        let count = bytes.len().min(60 * 1024);
        if count == 0 {
            return Ok(0);
        }
        loop {
            match self.end.write(Message {
                bytes: bytes[..count].to_vec(),
                handles: Vec::new(),
            }) {
                Ok(()) => return Ok(count),
                Err((dross_abi::Error::ShouldWait, _)) if !nb => {
                    sched::sleep_ns(1_000_000).map_err(|_| EINTR)?;
                }
                Err((dross_abi::Error::ShouldWait, _)) => return Err(EAGAIN),
                Err(_) => return Err(EPIPE),
            }
        }
    }
    pub fn poll(&self, events: u16) -> u16 {
        let flags = self.end.readiness()
            | if !self.pending.lock().is_empty() {
                1
            } else {
                0
            };
        flags & (events | 0x18)
    }
}
pub fn fd(p: &Process, end: Arc<ChannelEnd>) -> R {
    linux(p).files.insert(
        super::super::fs::File::new(
            super::super::fs::Kind::Channel(Arc::new(Stream::new(end))),
            2,
        ),
        true,
    )
}
