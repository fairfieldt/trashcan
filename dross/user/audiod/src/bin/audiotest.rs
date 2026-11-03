#![no_std]
#![no_main]
extern crate alloc;
use audiod::{CLOSE, INFO, PLAY, Request};
use dross_rt::{Error, Handle, bootstrap::Bootstrap, println, sys};
dross_rt::main!(main);
fn main(boot: Handle) -> i32 {
    match run(boot) {
        Ok(()) => {
            println!("ok");
            0
        }
        Err(e) => {
            println!("failed {e:?}");
            1
        }
    }
}
fn call(ch: Handle, r: Request, handles: &[Handle], want: i32) -> sys::Result<()> {
    sys::channel_write(ch, &r.encode(), handles)?;
    let mut bytes = [0; 32];
    let mut hs = [Handle(0); 4];
    let (n, nh) = sys::channel_read(ch, &mut bytes, &mut hs, sys::after_ms(3000))?;
    for &h in &hs[..nh] {
        let _ = sys::close(h);
    }
    if n != 32
        || nh != 0
        || u32::from_le_bytes(bytes[0..4].try_into().unwrap()) != 1
        || u64::from_le_bytes(bytes[8..16].try_into().unwrap()) != r.id
        || i32::from_le_bytes(bytes[4..8].try_into().unwrap()) != want
    {
        return Err(Error::InvalidArgs);
    }
    if want == 0 && u32::from_le_bytes(bytes[16..20].try_into().unwrap()) != r.frames {
        return Err(Error::InvalidArgs);
    }
    Ok(())
}
fn run(boot: Handle) -> sys::Result<()> {
    let b = Bootstrap::read(boot)?;
    let ch = b.handle("audio").ok_or(Error::BadHandle)?;
    call(
        ch,
        Request {
            op: INFO,
            id: 1,
            deadline: sys::after_ms(2000),
            frames: 0,
        },
        &[],
        0,
    )?;
    // Malformed handle count, undersized object and already expired deadline must fail consistently.
    call(
        ch,
        Request {
            op: PLAY,
            id: 2,
            deadline: sys::after_ms(2000),
            frames: 48,
        },
        &[],
        Error::InvalidArgs as i32,
    )?;
    let short = sys::mem_create(4096, 0)?;
    call(
        ch,
        Request {
            op: PLAY,
            id: 3,
            deadline: sys::after_ms(2000),
            frames: 4800,
        },
        &[short],
        Error::OutOfRange as i32,
    )?;
    let expired = sys::mem_create(4096, 0)?;
    call(
        ch,
        Request {
            op: PLAY,
            id: 4,
            deadline: 1,
            frames: 48,
        },
        &[expired],
        Error::TimedOut as i32,
    )?;
    let aborted = sys::mem_create(192512, 0)?;
    call(
        ch,
        Request {
            op: PLAY,
            id: 40,
            deadline: sys::after_ms(40),
            frames: 48000,
        },
        &[aborted],
        Error::TimedOut as i32,
    )?;
    // Two different lengths exercise stream reset, stale-MSI draining and DMA reuse.
    for (id, frames) in [(5, 12_000u32), (6, 24_000)] {
        let len = frames as u64 * 4;
        let mapped = len.next_multiple_of(4096);
        let memory = sys::mem_create(mapped, 0)?;
        let at = sys::vm_map(
            memory,
            0,
            mapped,
            0,
            dross_abi::vm::READ | dross_abi::vm::WRITE,
        )?;
        let samples =
            unsafe { core::slice::from_raw_parts_mut(at as *mut i16, frames as usize * 2) };
        // Deterministic low-volume sine via32-entry lookup: left440Hz, right660Hz.
        const SINE: [i16; 32] = [
            0, 293, 574, 833, 1061, 1247, 1386, 1471, 1500, 1471, 1386, 1247, 1061, 833, 574, 293,
            0, -293, -574, -833, -1061, -1247, -1386, -1471, -1500, -1471, -1386, -1247, -1061,
            -833, -574, -293,
        ];
        let mut left = 0u32;
        let mut right = 0u32;
        for f in 0..frames as usize {
            samples[f * 2] = SINE[(left >> 27) as usize];
            samples[f * 2 + 1] = SINE[(right >> 27) as usize];
            left = left.wrapping_add(((440u64 << 32) / 48000) as u32);
            right = right.wrapping_add(((660u64 << 32) / 48000) as u32);
        }
        sys::vm_unmap(at, mapped)?;
        call(
            ch,
            Request {
                op: PLAY,
                id,
                deadline: sys::after_ms(2000),
                frames,
            },
            &[memory],
            0,
        )?;
    }
    call(
        ch,
        Request {
            op: CLOSE,
            id: 7,
            deadline: sys::after_ms(2000),
            frames: 0,
        },
        &[],
        0,
    )?;
    Ok(())
}
