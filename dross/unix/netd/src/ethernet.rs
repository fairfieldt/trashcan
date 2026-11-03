//! Bounded frame-channel adapter; ethd and the QEMU NIC server share this contract.
use alloc::{collections::VecDeque, vec, vec::Vec};
use dross_rt::{Error, Handle, sys};
use oscar_ethernet::{Info, MAX_PACKET, Packet, decode, encode};
use smoltcp::{
    phy::{self, DeviceCapabilities, Medium},
    time::Instant,
};
pub struct Ethernet {
    channel: Handle,
    pub info: Info,
    rx: VecDeque<Vec<u8>>,
    tx: VecDeque<Vec<u8>>,
}
impl Ethernet {
    pub fn new(channel: Handle) -> Self {
        let mut bytes = [0; MAX_PACKET];
        let (n, h) = sys::channel_read(channel, &mut bytes, &mut [], sys::after_ms(15000))
            .expect("NIC info");
        assert_eq!(h, 0);
        let Ok(Packet::Info(info)) = decode(&bytes[..n]) else {
            panic!("NIC info packet");
        };
        Self {
            channel,
            info,
            rx: VecDeque::new(),
            tx: VecDeque::new(),
        }
    }
    pub fn service(&mut self) -> Result<(), Error> {
        let mut bytes = [0; MAX_PACKET];
        for _ in 0..32 {
            if self.rx.len() == 32 {
                break;
            }
            match sys::channel_read(self.channel, &mut bytes, &mut [], 0) {
                Ok((n, 0)) => match decode(&bytes[..n]) {
                    Ok(Packet::Frame(frame)) => self.rx.push_back(frame.to_vec()),
                    Ok(Packet::Link(link)) => self.info.link = link,
                    _ => return Err(Error::InvalidArgs),
                },
                Err(Error::TimedOut | Error::ShouldWait) => break,
                Err(e) => return Err(e),
                _ => return Err(Error::InvalidArgs),
            }
        }
        while let Some(frame) = self.tx.front() {
            let packet = encode(Packet::Frame(frame)).map_err(|_| Error::InvalidArgs)?;
            match sys::channel_write(self.channel, &packet, &[]) {
                Ok(()) => {
                    self.tx.pop_front();
                }
                Err(Error::ShouldWait) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}
pub struct Rx(Vec<u8>);
pub struct Tx<'a>(&'a mut VecDeque<Vec<u8>>);
impl phy::RxToken for Rx {
    fn consume<R, F: FnOnce(&[u8]) -> R>(self, f: F) -> R {
        f(&self.0)
    }
}
impl phy::TxToken for Tx<'_> {
    fn consume<R, F: FnOnce(&mut [u8]) -> R>(self, len: usize, f: F) -> R {
        let mut bytes = vec![0; len];
        let out = f(&mut bytes);
        self.0.push_back(bytes);
        out
    }
}
impl phy::Device for Ethernet {
    type RxToken<'a> = Rx;
    type TxToken<'a> = Tx<'a>;
    fn receive(&mut self, _: Instant) -> Option<(Rx, Tx<'_>)> {
        if self.tx.len() == 32 {
            return None;
        }
        self.rx.pop_front().map(|f| (Rx(f), Tx(&mut self.tx)))
    }
    fn transmit(&mut self, _: Instant) -> Option<Tx<'_>> {
        if self.tx.len() == 32 || !self.info.link.up {
            None
        } else {
            Some(Tx(&mut self.tx))
        }
    }
    fn capabilities(&self) -> DeviceCapabilities {
        let mut caps = DeviceCapabilities::default();
        caps.medium = Medium::Ethernet;
        caps.max_transmission_unit = 1514;
        caps
    }
}
