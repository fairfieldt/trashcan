//! Regression driver: check a block view, then proxy it unchanged to fsd.
#![no_std]
#![no_main]
extern crate alloc;
use dross_abi::vm;
use dross_rt::bootstrap::Bootstrap;
use dross_rt::{Handle, println, sys};
dross_rt::main!(main);
fn request(ch: Handle, req: &[u8]) -> i64 {
    sys::channel_write(ch, req, &[]).expect("request");
    let mut out = [0u8; 24];
    let (n, _) = sys::channel_read(ch, &mut out, &mut [], dross_rt::FOREVER).expect("reply");
    assert!(n >= 8);
    i64::from_le_bytes(out[..8].try_into().unwrap())
}
fn main(boot: Handle) -> i32 {
    let b = Bootstrap::read(boot).expect("bootstrap");
    let raw = b.handle("block").expect("block");
    let service = b.handle("service").expect("service");
    let ready = b.handle("ready").expect("ready");
    let expected = b.value_u64("blocks").expect("blocks");
    sys::channel_write(raw, &[0], &[]).expect("info");
    let (mut info, mut hs) = ([0u8; 20], [Handle(0); 1]);
    let (n, nh) =
        sys::channel_read(raw, &mut info, &mut hs, dross_rt::FOREVER).expect("info reply");
    assert_eq!((n, nh), (20, 1));
    assert_eq!(i64::from_le_bytes(info[..8].try_into().unwrap()), 0);
    let blocks = u64::from_le_bytes(info[12..20].try_into().unwrap());
    assert_eq!(blocks, expected);
    let window = hs[0];
    let bytes = sys::mem_size(window).expect("window");
    let at = sys::vm_map(window, 0, bytes, 0, vm::READ | vm::WRITE).expect("window map");
    // If an invalid WRITE is accidentally accepted, the outside-image sentinels
    // checked by the host test catch the corruption as well as this status check.
    unsafe { core::ptr::write_bytes(at as *mut u8, 0x5a, bytes as usize) };
    for (op, lba, count) in [
        (1u8, blocks, 1u32),
        (2, blocks, 1),
        (2, u64::MAX, 1),
        (2, 0, 0),
        (2, blocks - 1, 2),
        (2, 0, u32::MAX),
    ] {
        let mut req = alloc::vec![op];
        req.extend_from_slice(&lba.to_le_bytes());
        req.extend_from_slice(&count.to_le_bytes());
        assert_eq!(request(raw, &req), -22);
    }
    println!("block range bounds: ok");
    sys::channel_write(ready, b"ready", &[]).expect("ready");
    let mut shared = false;
    let mut req = [0u8; 16];
    loop {
        let (n, _) = match sys::channel_read(service, &mut req, &mut [], dross_rt::FOREVER) {
            Ok(r) => r,
            Err(_) => return 0,
        };
        if req[..n] == [0] {
            let transfer = if shared { None } else { Some(window) };
            if sys::channel_write(service, &info, transfer.as_slice()).is_err() {
                return 0;
            }
            shared = true;
            continue;
        }
        sys::channel_write(raw, &req[..n], &[]).expect("forward request");
        let mut reply = [0u8; 24];
        let (n, _) =
            sys::channel_read(raw, &mut reply, &mut [], dross_rt::FOREVER).expect("forward reply");
        if sys::channel_write(service, &reply[..n], &[]).is_err() {
            return 0;
        }
    }
}
