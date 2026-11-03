//! Development-only native control capability. Not on the public render socket.
use crate::serve::Shared;
use dross_rt::{Error, Handle, sys};
use std::sync::{Arc, atomic::Ordering};
unsafe extern "C" {
    fn gpud_fence_read(adev: *mut crate::hw::AmdgpuDevice, ring: i32) -> u64;
}
pub fn start(shared: Arc<Shared>) {
    if !gpud_platform::on_dross() {
        return;
    }
    let Some(channel) = gpud_platform::dross::bootstrap_handle("supervisor") else {
        return;
    };
    std::thread::spawn(move || {
        if sys::channel_write(channel, &0i64.to_le_bytes(), &[]).is_err() {
            return;
        }
        loop {
            let mut bytes = [0u8; 4];
            let mut handles = [Handle(0); 64];
            let (n, nh) = match sys::channel_read(channel, &mut bytes, &mut handles, sys::after_ms(500)) {
                Ok(v) => v,
                Err(Error::TimedOut | Error::ShouldWait) => continue,
                Err(_) => return,
            };
            for &handle in &handles[..nh] {
                let _ = sys::close(handle);
            }
            if n != 4 || nh != 0 || u32::from_le_bytes(bytes) != 1 {
                let _ = sys::channel_write(channel, &(-libc::EINVAL as i64).to_le_bytes(), &[]);
                continue;
            }
            let mut accepting = shared.accepting.write().unwrap();
            // Keep a live/poisoned epoch intact. The caller must close applications
            // and restore firmware scanout before replacing this driver process.
            if shared.display_live.load(Ordering::Acquire)
                || shared.vmids.lock().unwrap().iter().any(|&used| used)
                || !shared.submitted.lock().unwrap().complete(|ring| unsafe { gpud_fence_read(shared.adev(), ring) })
            {
                let _ = sys::channel_write(channel, &(-libc::EBUSY as i64).to_le_bytes(), &[]);
                continue;
            }
            *accepting = false;
            shared.device.shutdown();
            let _ = sys::channel_write(channel, &0i64.to_le_bytes(), &[]);
            crate::hw::log("info", "supervisor: drained GPU stopped");
            std::process::exit(0);
        }
    });
}
