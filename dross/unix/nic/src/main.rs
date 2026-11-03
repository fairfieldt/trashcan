#![no_std]
#![no_main]
extern crate alloc;
mod e1000;
use alloc::{collections::VecDeque, vec::Vec};
use dross_rt::{Error, Handle, bootstrap::Bootstrap, pci::Pci, sys};
use oscar_ethernet::{Info, MAX_PACKET, Packet, decode, encode};
dross_rt::main!(main);
fn main(bootstrap: Handle) -> i32 {
    let b = Bootstrap::read(bootstrap).expect("nic bootstrap");
    let channel = b.handle("ethernet").expect("Ethernet channel");
    let mut nic = e1000::Nic::new(Pci::new(b.handle("pci").expect("PCI")));
    let mut link = nic.link();
    sys::channel_write(
        channel,
        &encode(Packet::Info(Info {
            mac: nic.mac,
            mtu: 1500,
            link,
        }))
        .unwrap(),
        &[],
    )
    .unwrap();
    let notify = sys::notify_create().unwrap();
    sys::channel_bind(channel, notify, 1).unwrap();
    let mut tx = VecDeque::<Vec<u8>>::new();
    let mut pending: Option<Vec<u8>> = None;
    let mut bytes = [0; MAX_PACKET];
    let mut hs = [Handle(0); 1];
    loop {
        for _ in 0..32 {
            if tx.len() == 32 {
                break;
            }
            match sys::channel_read(channel, &mut bytes, &mut hs, 0) {
                Ok((n, 0)) => match decode(&bytes[..n]) {
                    Ok(Packet::Frame(frame)) => tx.push_back(frame.to_vec()),
                    _ => return 1,
                },
                Err(Error::TimedOut | Error::ShouldWait) => break,
                Err(Error::PeerClosed) => return 0,
                Ok((_, n)) => {
                    for h in &hs[..n] {
                        let _ = sys::close(*h);
                    }
                    return 1;
                }
                Err(_) => return 1,
            }
        }
        while nic.available() {
            let Some(frame) = tx.pop_front() else {
                break;
            };
            nic.send(&frame);
        }
        for _ in 0..32 {
            if pending.is_none() {
                pending = nic.read_frame().map(|f| encode(Packet::Frame(&f)).unwrap());
            }
            let Some(frame) = pending.as_ref() else {
                break;
            };
            match sys::channel_write(channel, frame, &[]) {
                Ok(()) => pending = None,
                Err(Error::ShouldWait) => break,
                Err(_) => return 0,
            }
        }
        let now = nic.link();
        if now != link && pending.is_none() {
            let frame = encode(Packet::Link(now)).unwrap();
            if sys::channel_write(channel, &frame, &[]).is_ok() {
                link = now;
            }
        }
        let _ = sys::notify_wait(notify, sys::after_ms(1));
    }
}
