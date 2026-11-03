//! Bounded OETH transport. Driver link events are propagated even when TCP
//! admission is full. No incoming attachments are retained or ignored.
use alloc::{collections::VecDeque, vec::Vec};
use dross_rt::{Error, Handle, owned::OwnedHandle, sys};
use oscar_ethernet::{Info, MAX_PACKET, Packet, decode};
use oscar_netstack3_engine::Engine;
const MAX_TX: usize = 32;
pub struct Ethernet {
    channel: OwnedHandle,
    pub info: Info,
    tx: VecDeque<Vec<u8>>,
    pub dropped_rx: u64,
}
impl Ethernet {
    pub fn new(channel: OwnedHandle) -> Result<Self, Error> {
        let mut bytes = [0u8; MAX_PACKET];
        let n = receive(&channel, &mut bytes, sys::after_ms(15000))?;
        let Packet::Info(info) = decode(&bytes[..n]).map_err(|_| Error::InvalidArgs)? else {
            return Err(Error::InvalidArgs);
        };
        let mut tx = VecDeque::new();
        tx.try_reserve_exact(MAX_TX).map_err(|_| Error::NoMemory)?;
        Ok(Self {
            channel,
            info,
            tx,
            dropped_rx: 0,
        })
    }
    pub fn service(&mut self, engine: &mut Engine) -> Result<bool, Error> {
        let mut more = false;
        let mut bytes = [0u8; MAX_PACKET];
        for i in 0..32 {
            match receive(&self.channel, &mut bytes, 0) {
                Ok(n) => {
                    match decode(&bytes[..n]).map_err(|_| Error::InvalidArgs)? {
                        Packet::Frame(frame) => match engine.receive_ethernet(frame) {
                            Ok(()) => {}
                            Err(
                                oscar_netstack3_engine::Error::Invalid
                                | oscar_netstack3_engine::Error::NoResources,
                            ) => {
                                self.dropped_rx = self.dropped_rx.saturating_add(1);
                            }
                            Err(_) => return Err(Error::InvalidArgs),
                        },
                        Packet::Link(link) => {
                            self.info.link = link;
                            engine.set_link(link.up);
                        }
                        Packet::Info(info)
                            if info.mac == self.info.mac && info.mtu == self.info.mtu =>
                        {
                            self.info.link = info.link;
                            engine.set_link(info.link.up);
                        }
                        _ => return Err(Error::InvalidArgs),
                    }
                    if i == 31 {
                        more = true;
                    }
                }
                Err(Error::ShouldWait | Error::TimedOut) => break,
                Err(e) => return Err(e),
            }
        }
        self.flush()?;
        while self.tx.len() < MAX_TX {
            let Some(frame) = engine.take_ethernet() else {
                break;
            };
            if !(14..=oscar_ethernet::MAX_FRAME).contains(&frame.len()) {
                return Err(Error::InvalidArgs);
            }
            self.tx.push_back(frame);
        }
        let filled = self.tx.len() == MAX_TX;
        self.flush()?;
        more |= filled && self.tx.len() < MAX_TX;
        Ok(more)
    }
    fn flush(&mut self) -> Result<(), Error> {
        if !self.info.link.up {
            self.tx.clear();
            return Ok(());
        }
        while let Some(frame) = self.tx.front() {
            let mut packet = [0u8; MAX_PACKET];
            packet[..8].copy_from_slice(b"OETH\x01\0\x02\0");
            packet[8..8 + frame.len()].copy_from_slice(frame);
            match sys::channel_write(
                self.channel.as_handle().as_raw(),
                &packet[..8 + frame.len()],
                &[],
            ) {
                Ok(()) => {
                    self.tx.pop_front();
                }
                Err(Error::ShouldWait) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
    pub fn wait(
        &self,
        slot: &mut Option<crate::wait::Registration>,
        queue: Handle,
        delivered: Option<u64>,
    ) -> Result<(), Error> {
        use dross_abi::event::{PEER_CLOSED, READABLE, WRITABLE};
        let signals = READABLE | PEER_CLOSED | if self.tx.is_empty() { 0 } else { WRITABLE };
        crate::wait::Registration::update(slot, queue, self.channel.as_handle(), signals, delivered)
    }
}
fn receive(ch: &OwnedHandle, bytes: &mut [u8], deadline: u64) -> Result<usize, Error> {
    let mut handles = [Handle(0); 64];
    let (n, nh) = sys::channel_read(ch.as_handle().as_raw(), bytes, &mut handles, deadline)?;
    if nh != 0 {
        for h in &handles[..nh] {
            let _ = sys::close(*h);
        }
        return Err(Error::InvalidArgs);
    }
    Ok(n)
}
