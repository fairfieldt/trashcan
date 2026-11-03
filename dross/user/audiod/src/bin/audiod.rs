#![no_std]
#![no_main]
extern crate alloc;
use audiod::{CLOSE, INFO, MESSAGE_BYTES, PLAY, Request};
use dross_rt::{Error, Handle, bootstrap::Bootstrap, pci::Pci, println, sys};
dross_rt::main!(main);

fn main(boot: Handle) -> i32 {
    match run(boot) {
        Ok(()) => 0,
        Err(e) => {
            println!("failed {e:?}; exiting for pcid fail-stop cleanup");
            1
        }
    }
}
fn run(boot: Handle) -> sys::Result<()> {
    let b = Bootstrap::read(boot)?;
    let service = b.handle("service").ok_or(Error::BadHandle)?;
    let pci = Pci::with_timeout(b.handle("pci").ok_or(Error::BadHandle)?, 1000);
    let mut hda = audiod::hda::Hda::open(&pci)?;
    println!("service ready");
    let mut bytes = [0u8; MESSAGE_BYTES];
    let mut handles = [Handle(0); 64];
    loop {
        let (n, nh) = match sys::channel_read(service, &mut bytes, &mut handles, dross_rt::FOREVER)
        {
            Ok(v) => v,
            Err(Error::PeerClosed) => {
                hda.shutdown()?;
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        let request = Request::decode(&bytes[..n]);
        let id = if n >= 16 {
            u64::from_le_bytes(bytes[8..16].try_into().unwrap())
        } else {
            0
        };
        let mut close = false;
        let outcome = (|| {
            let r = request.ok_or(Error::InvalidArgs)?;
            if (r.op == PLAY && nh != 1) || (r.op != PLAY && nh != 0) {
                return Err(Error::InvalidArgs);
            }
            let now = sys::clock();
            if r.deadline <= now {
                return Err(Error::TimedOut);
            }
            if r.deadline > now.saturating_add(2_000_000_000) {
                return Err(Error::OutOfRange);
            }
            match r.op {
                PLAY => {
                    let memory = handles[0];
                    let len = r.frames as u64 * 4;
                    let size = sys::mem_size(memory)?;
                    if size < len {
                        return Err(Error::OutOfRange);
                    }
                    let mapped = len.next_multiple_of(4096);
                    let at = sys::vm_map(memory, 0, mapped, 0, dross_abi::vm::READ)?;
                    // The snapshot prevents further client writes changing an active hardware buffer.
                    // Client must avoid racing the snapshot; this contract does not promise atomic writes.
                    // SAFETY: this read mapping stays valid until snapshot returns.
                    let staged = unsafe { hda.snapshot(at as *const u8, len as usize) };
                    sys::vm_unmap(at, mapped)?;
                    sys::close(memory)?;
                    handles[0] = Handle(0);
                    staged?;
                    hda.play(r.frames, r.deadline)
                }
                INFO => Ok(0),
                CLOSE => {
                    close = true;
                    hda.shutdown()?;
                    Ok(0)
                }
                _ => Err(Error::InvalidArgs),
            }
        })();
        for &h in &handles[..nh] {
            if h.0 != 0 {
                sys::close(h)?;
            }
        }
        if close {
            if let Err(error) = outcome {
                return Err(error);
            }
        }
        if !hda.stopped || hda.failed {
            return Err(Error::NoResources);
        }
        let (status, frames) = match outcome {
            Ok(frames) => (0, frames),
            Err(e) => (e as i32, 0),
        };
        sys::channel_write(service, &audiod::reply(id, status, frames), &[])?;
        if close {
            return Ok(());
        }
    }
}
