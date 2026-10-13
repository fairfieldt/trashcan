//! Bounds for work submitted ahead (`GpuStage::presubmit`): such a command buffer waits on the GPU for a host-set
//! event, and amdgpu resets a GPU whose job runs past its timeout (about 10 s), waiting included. Work is submitted
//! ahead only once its input is on the way, so a gate is normally released within one decode step; a watchdog thread
//! releases any gate armed for longer than `DEADLINE` (the GPU then computes garbage and the stage refuses to go on, so
//! its session ends), and a SIGINT, SIGTERM or SIGHUP sets every gate before the process ends
//! (`qwen::gpu::register_gate`, as for the 27B's tensor-parallel gates).
use crate::vulkan::Device;
use anyhow::{bail, Result};
use ash::vk;
use std::{
    sync::{Mutex, Once},
    time::{Duration, Instant},
};

/// Longest a gate may stay armed.
pub const DEADLINE: Duration = Duration::from_secs(2);

struct Gate {
    dev: vk::Device,
    set: vk::PFN_vkSetEvent,
    event: vk::Event,
    armed: Option<Instant>,
    fired: bool,
}

static GATES: Mutex<Vec<Gate>> = Mutex::new(Vec::new());

/// Watch `event`, a gate of `gpu` (also released on SIGINT, SIGTERM and SIGHUP).
pub fn register(gpu: &Device, event: vk::Event) {
    crate::qwen::gpu::register_gate(gpu, event);
    GATES.lock().unwrap().push(Gate { dev: gpu.dev.handle(), set: gpu.dev.fp_v1_0().set_event, event, armed: None, fired: false });
}

/// Forget the gates of `gpu` (its device is going away).
pub fn unregister(gpu: &Device) {
    crate::qwen::gpu::unregister_gates(gpu);
    GATES.lock().unwrap().retain(|g| g.dev != gpu.dev.handle());
}

/// Work waiting for `event` was submitted: release it after `DEADLINE` unless `disarm` comes first.
pub fn arm(event: vk::Event) {
    static WATCHDOG: Once = Once::new();
    WATCHDOG.call_once(|| {
        std::thread::spawn(|| loop {
            std::thread::sleep(Duration::from_millis(50));
            for g in GATES.lock().unwrap().iter_mut() {
                if g.armed.is_some_and(|t| t.elapsed() > DEADLINE) {
                    let _ = unsafe { (g.set)(g.dev, g.event) };
                    (g.armed, g.fired) = (None, true);
                    eprintln!("qwen4: work submitted ahead got no input for {DEADLINE:?}; released it (the stage stops)");
                }
            }
        });
    });
    if let Some(g) = GATES.lock().unwrap().iter_mut().find(|g| g.event == event) {
        (g.armed, g.fired) = (Some(Instant::now()), false);
    }
}

/// The host is about to release `event` itself; an error when the watchdog already did.
pub fn disarm(event: vk::Event) -> Result<()> {
    if let Some(g) = GATES.lock().unwrap().iter_mut().find(|g| g.event == event) {
        g.armed = None;
        if std::mem::take(&mut g.fired) {
            bail!("work submitted ahead was released by the watchdog after {DEADLINE:?} without its input");
        }
    }
    Ok(())
}
