//! Bounded, whole-datagram receive storage shared with the engine.
use crate::sync::Mutex;
use alloc::{collections::VecDeque, sync::Arc, vec::Vec};
use core::fmt;
use net_types::ip::Ip;
use netstack3_datagram::PendingDatagramSocketError;
use netstack3_udp::{ReceiveUdpError, UdpPacketMeta};
pub const MAX_DATAGRAMS: usize = 16;
pub const MAX_DATAGRAM_BYTES: usize = 32 * 1024;
pub struct Datagram<I: Ip> {
    pub meta: UdpPacketMeta<I>,
    pub interface: u32,
    pub body: Vec<u8>,
}
struct Queue<I: Ip> {
    packets: VecDeque<Datagram<I>>,
    bytes: usize,
    error: Option<PendingDatagramSocketError>,
    drops: u64,
    closed: bool,
}
pub struct UdpRx<I: Ip>(Arc<Mutex<Queue<I>>>);
impl<I: Ip> Clone for UdpRx<I> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<I: Ip> fmt::Debug for UdpRx<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("UdpRx")
    }
}
impl<I: Ip> Default for UdpRx<I> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Queue {
            packets: VecDeque::new(),
            bytes: 0,
            error: None,
            drops: 0,
            closed: false,
        })))
    }
}
impl<I: Ip> UdpRx<I> {
    pub(crate) fn push(
        &self,
        meta: UdpPacketMeta<I>,
        interface: u32,
        body: &[u8],
    ) -> Result<(), ReceiveUdpError> {
        let mut q = self.0.lock();
        if q.closed || q.packets.len() >= MAX_DATAGRAMS || body.len() > MAX_DATAGRAM_BYTES - q.bytes
        {
            q.drops = q.drops.saturating_add(1);
            return Err(ReceiveUdpError::QueueFull);
        }
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(body.len()).is_err() || q.packets.try_reserve(1).is_err() {
            q.drops = q.drops.saturating_add(1);
            return Err(ReceiveUdpError::QueueFull);
        }
        bytes.extend_from_slice(body);
        q.bytes += bytes.len();
        q.packets.push_back(Datagram {
            meta,
            interface,
            body: bytes,
        });
        Ok(())
    }
    pub(crate) fn set_error(&self, e: PendingDatagramSocketError) {
        self.0.lock().error = Some(e);
    }
    pub fn error(&self, take: bool) -> Option<PendingDatagramSocketError> {
        let mut q = self.0.lock();
        if take { q.error.take() } else { q.error }
    }
    pub fn readable(&self) -> bool {
        !self.0.lock().packets.is_empty()
    }
    pub fn shutdown_receive(&self) {
        let mut q = self.0.lock();
        q.closed = true;
        q.packets.clear();
        q.bytes = 0;
    }
    pub fn clear(&self) {
        let mut q = self.0.lock();
        q.packets.clear();
        q.bytes = 0;
    }
    pub fn counters(&self) -> (usize, usize, u64) {
        let q = self.0.lock();
        (q.packets.len(), q.bytes, q.drops)
    }
    /// Copy only the requested prefix; consuming a truncated datagram discards its tail.
    pub fn receive(
        &self,
        out: &mut [u8],
        peek: bool,
    ) -> Option<(UdpPacketMeta<I>, u32, usize, usize)> {
        let mut q = self.0.lock();
        let p = q.packets.front()?;
        let n = out.len().min(p.body.len());
        out[..n].copy_from_slice(&p.body[..n]);
        let result = (p.meta.clone(), p.interface, n, p.body.len());
        if !peek {
            let p = q.packets.pop_front().unwrap();
            q.bytes -= p.body.len();
        }
        Some(result)
    }
}
