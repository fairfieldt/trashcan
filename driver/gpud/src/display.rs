//! Exclusive native display capability. Never reachable through the public GPU
//! render socket: only init's explicitly supplied channel may change scanout.
use crate::{hw::AmdgpuDevice, serve::Shared};
use dross_rt::{Error, Handle, sys};
use std::{
    ffi::c_void,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Info {
    index: u32,
    width: u32,
    height: u32,
    pitch: u32,
    control: u32,
    swap: u32,
    frame: u32,
    flags: u32,
    base: u64,
    bytes: u64,
}
unsafe extern "C" {
    fn gpud_display_info(adev: *mut AmdgpuDevice, index: u32, out: *mut Info) -> i32;
    fn gpud_display_acquire(adev: *mut AmdgpuDevice, index: u32, cpu: *mut *mut c_void, bytes: *mut u64) -> i32;
    fn gpud_display_upload(adev: *mut AmdgpuDevice, index: u32, buffer: u32, ring: *mut i32, seq: *mut u64) -> i32;
    fn gpud_display_flip(adev: *mut AmdgpuDevice, index: u32, buffer: u32, frame: *mut u32) -> i32;
    fn gpud_display_release(adev: *mut AmdgpuDevice, index: u32) -> i32;
    fn gpud_fence_wait(adev: *mut AmdgpuDevice, ring: i32, seq: u64, timeout: i64) -> i32;
}
pub fn start(shared: Arc<Shared>) {
    if !gpud_platform::on_dross() {
        return;
    }
    let Some(channel) = gpud_platform::dross::bootstrap_handle("display") else { return };
    std::thread::spawn(move || serve(shared, channel));
}
fn reply(ch: Handle, status: i64, payload: &[u8], memory: Option<Handle>) -> Result<(), Error> {
    let mut bytes = status.to_le_bytes().to_vec();
    bytes.extend_from_slice(payload);
    let handles = memory.as_slice();
    match sys::channel_write(ch, &bytes, handles) {
        Ok(()) => Ok(()),
        Err(e) => {
            if let Some(h) = memory {
                let _ = sys::close(h);
            }
            Err(e)
        }
    }
}
fn serve(shared: Arc<Shared>, channel: Handle) {
    let mut owned = None;
    let mut next = 0;
    let mut quarantined = false;
    let mut pending = None;
    let mut release_requested = false;
    loop {
        let mut request = [0u8; 8];
        let mut handles = [Handle(0); 64];
        let result = sys::channel_read(channel, &mut request, &mut handles, sys::after_ms(500));
        let (n, nh) = match result {
            Ok(r) => r,
            Err(Error::TimedOut | Error::ShouldWait) => continue,
            Err(_) => break,
        };
        for &h in &handles[..nh] {
            let _ = sys::close(h);
        }
        if n != 8 || nh != 0 {
            break;
        }
        let op = u32::from_le_bytes(request[..4].try_into().unwrap());
        let argument = u32::from_le_bytes(request[4..].try_into().unwrap());
        let access = shared.accepting.read().unwrap();
        if !*access {
            break;
        }
        match op {
            0 if owned.is_none() => {
                let mut info = Info::default();
                let found = (0..6).find(|&index| {
                    (argument == u32::MAX || argument == index)
                        && unsafe { gpud_display_info(shared.adev(), index, &mut info) } == 0
                        && info.flags & 3 == 3
                });
                if found.is_none() {
                    let _ = reply(channel, -libc::ENODEV as i64, &[], None);
                    continue;
                }
                let mut cpu = std::ptr::null_mut();
                let mut bytes = 0;
                let r = unsafe { gpud_display_acquire(shared.adev(), info.index, &mut cpu, &mut bytes) };
                if r != 0 {
                    let _ = reply(channel, r as i64, &[], None);
                    continue;
                }
                owned = Some(info.index);
                shared.display_live.store(true, Ordering::Release);
                let export = sys::vm_memory(cpu as u64).and_then(|(memory, offset, remaining)| {
                    if offset != 0 || remaining < bytes {
                        let _ = sys::close(memory);
                        return Err(Error::InvalidArgs);
                    }
                    let duplicate = sys::dup(
                        memory,
                        dross_abi::rights::READ
                            | dross_abi::rights::WRITE
                            | dross_abi::rights::MAP
                            | dross_abi::rights::TRANSFER,
                    );
                    let _ = sys::close(memory);
                    duplicate
                });
                let memory = match export {
                    Ok(h) => h,
                    Err(_) => break,
                };
                let mut payload = Vec::new();
                for v in
                    [info.index, info.width, info.height, info.pitch, info.control, info.swap, info.frame, info.flags]
                {
                    payload.extend_from_slice(&v.to_le_bytes());
                }
                payload.extend_from_slice(&bytes.to_le_bytes());
                if reply(channel, 0, &payload, Some(memory)).is_err() {
                    break;
                }
                crate::hw::log(
                    "info",
                    &format!("display {} acquired: {}x{}, pitch {}", info.index, info.width, info.height, info.pitch),
                );
            }
            1 if owned.is_some() && !quarantined => {
                let index = owned.unwrap();
                let mut ring = 0;
                let mut sequence = 0;
                let mut r = unsafe { gpud_display_upload(shared.adev(), index, next, &mut ring, &mut sequence) };
                if r == 0 {
                    shared.submitted.lock().unwrap().record(ring, sequence);
                    pending = Some((ring, sequence));
                    r = unsafe { gpud_fence_wait(shared.adev(), ring, sequence, 500_000_000) };
                }
                let mut frame = 0;
                if r == 0 {
                    pending = None;
                    r = unsafe { gpud_display_flip(shared.adev(), index, next, &mut frame) };
                }
                if r != 0 {
                    quarantined = true;
                    crate::hw::log("error", &format!("display present failed {r}; buffers quarantined"));
                } else {
                    next ^= 1;
                }
                if reply(channel, r as i64, &frame.to_le_bytes(), None).is_err() {
                    break;
                }
            }
            2 => {
                release_requested = true;
                break;
            }
            _ => {
                if reply(channel, -libc::EINVAL as i64, &[], None).is_err() {
                    break;
                }
            }
        }
    }
    if let Some(index) = owned {
        if let Some((ring, sequence)) = pending {
            // A timeout is not cancellation. Keep the backing alive and poll
            // bounded waits until the copy has actually stopped reading it.
            crate::hw::log("error", "display owner closed with unfinished copy; backing retained");
            loop {
                let _access = shared.accepting.read().unwrap();
                if unsafe { gpud_fence_wait(shared.adev(), ring, sequence, 10_000_000) } == 0 {
                    break;
                }
                drop(_access);
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        let _access = shared.accepting.read().unwrap();
        let result = unsafe { gpud_display_release(shared.adev(), index) };
        if result == 0 {
            shared.display_live.store(false, Ordering::Release);
        }
        if release_requested {
            let _ = reply(channel, result as i64, &[], None);
        }
        crate::hw::log(
            if result == 0 { "info" } else { "error" },
            &format!("display {index} firmware restore: {result}"),
        );
    }
    let _ = sys::close(channel);
}
