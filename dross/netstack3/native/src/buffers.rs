//! Fixed-capacity TCP payload buffers. Allocation happens at socket creation;
//! core's passive-open constructor is infallible, so process OOM can still abort.
use crate::sync::Mutex;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};
use netstack3_base::{FragmentedPayload, Payload};
use netstack3_tcp::{
    AcceptQueueChange, Buffer, BufferLimits, BufferSizes, IntoBuffers, ListenerNotifier,
    ReceiveBuffer, SendBuffer,
};

pub const TCP_BYTES: usize = 16 * 1024;
pub const MAX_TCP_BYTES: usize = 32 * 1024;

#[derive(Debug)]
pub struct Ring {
    storage: Vec<u8>,
    head: usize,
    length: usize,
    wraps: usize,
}
impl Ring {
    fn new(requested: usize) -> Self {
        let capacity = requested.min(MAX_TCP_BYTES);
        assert!(capacity > 0);
        Self {
            storage: vec![0; capacity],
            head: 0,
            length: 0,
            wraps: 0,
        }
    }
    fn write<P: Payload>(&mut self, offset: usize, payload: &P) -> usize {
        let free = self.storage.len() - self.length;
        if offset >= free {
            return 0;
        }
        let count = payload.len().min(free - offset);
        if count == 0 {
            return 0;
        }
        let start = (self.head + self.length + offset) % self.storage.len();
        let first = count.min(self.storage.len() - start);
        payload.partial_copy(0, &mut self.storage[start..start + first]);
        if first < count {
            self.wraps += 1;
            payload.partial_copy(first, &mut self.storage[..count - first]);
        }
        count
    }
    pub fn wraps(&self) -> usize {
        self.wraps
    }
    pub fn enqueue(&mut self, bytes: &[u8]) -> usize {
        let count = self.write(0, &bytes);
        self.length += count;
        count
    }
    pub fn read(&mut self, destination: &mut [u8]) -> usize {
        let count = self.peek(destination);
        self.discard(count);
        count
    }
    pub fn peek(&self, destination: &mut [u8]) -> usize {
        let count = self.length.min(destination.len());
        let first = count.min(self.storage.len() - self.head);
        destination[..first].copy_from_slice(&self.storage[self.head..self.head + first]);
        destination[first..count].copy_from_slice(&self.storage[..count - first]);
        count
    }
    fn discard(&mut self, count: usize) {
        assert!(count <= self.length);
        self.head = (self.head + count) % self.storage.len();
        self.length -= count;
    }
}
impl Buffer for Ring {
    fn limits(&self) -> BufferLimits {
        BufferLimits {
            capacity: self.storage.len(),
            len: self.length,
        }
    }
    fn target_capacity(&self) -> usize {
        self.storage.len()
    }
    fn request_capacity(&mut self, _size: usize) { /* Fixed capacity, permitted by Buffer contract. */
    }
}
impl SendBuffer for Ring {
    type Payload<'a> = FragmentedPayload<'a, 2>;
    fn mark_read(&mut self, count: usize) {
        self.discard(count);
    }
    fn peek_with<'a, F, R>(&'a mut self, offset: usize, f: F) -> R
    where
        F: FnOnce(Self::Payload<'a>) -> R,
    {
        assert!(offset <= self.length);
        let start = (self.head + offset) % self.storage.len();
        let length = self.length - offset;
        let first = length.min(self.storage.len() - start);
        f(FragmentedPayload::new([
            &self.storage[start..start + first],
            &self.storage[..length - first],
        ]))
    }
}

#[derive(Debug, Clone)]
pub struct Rx(pub Arc<Mutex<Ring>>);
impl Buffer for Rx {
    fn limits(&self) -> BufferLimits {
        self.0.lock().limits()
    }
    fn target_capacity(&self) -> usize {
        self.0.lock().target_capacity()
    }
    fn request_capacity(&mut self, _size: usize) {}
}
impl ReceiveBuffer for Rx {
    fn write_at<P: Payload>(&mut self, offset: usize, data: &P) -> usize {
        self.0.lock().write(offset, data)
    }
    fn make_readable(&mut self, count: usize, _outstanding: bool) {
        let mut ring = self.0.lock();
        assert!(count <= ring.storage.len() - ring.length);
        ring.length += count;
    }
}

#[derive(Clone, Debug)]
pub struct ClientBuffers {
    pub receive: Rx,
}
impl ClientBuffers {
    pub fn read(&self, destination: &mut [u8]) -> usize {
        self.receive.0.lock().read(destination)
    }
    pub fn peek(&self, destination: &mut [u8]) -> usize {
        self.receive.0.lock().peek(destination)
    }
    pub fn readable(&self) -> usize {
        self.receive.0.lock().limits().len
    }
}

#[derive(Clone)]
pub struct SocketExtra(pub Arc<Mutex<Option<ClientBuffers>>>, Arc<AtomicBool>);
impl Default for SocketExtra {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(None)), Arc::new(AtomicBool::new(false)))
    }
}
impl fmt::Debug for SocketExtra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SocketExtra").finish_non_exhaustive()
    }
}
impl SocketExtra {
    pub fn take(&self) -> Option<ClientBuffers> {
        self.0.lock().take()
    }
    pub fn accept_ready(&self) -> bool {
        self.1.load(Ordering::Acquire)
    }
}
impl IntoBuffers<Rx, Ring> for SocketExtra {
    fn into_buffers(self, sizes: BufferSizes) -> (Rx, Ring) {
        let (receive, send, client) = new_buffers(sizes);
        *self.0.lock() = Some(client);
        (receive, send)
    }
}
impl ListenerNotifier for SocketExtra {
    fn accept_queue_changed(&mut self, change: AcceptQueueChange) {
        let ready = match change {
            AcceptQueueChange::ConnectionAdded { .. } => true,
            AcceptQueueChange::ConnectionRemoved { is_empty } => !is_empty,
        };
        self.1.store(ready, Ordering::Release);
    }
}
pub fn new_buffers(sizes: BufferSizes) -> (Rx, Ring, ClientBuffers) {
    let receive = Rx(Arc::new(Mutex::new(Ring::new(sizes.receive))));
    let client = ClientBuffers {
        receive: receive.clone(),
    };
    (receive, Ring::new(sizes.send), client)
}
