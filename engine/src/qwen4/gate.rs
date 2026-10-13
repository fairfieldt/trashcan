//! Bounds for work submitted ahead (`GpuStage::presubmit`): such a command buffer waits on the GPU for a host-set
//! event, and amdgpu resets a GPU whose job runs past its timeout, waiting included (`lockup_timeout`, 2 s by default
//! on the Macs' 7.2 kernels; the reset loses VRAM). Work is submitted ahead only once its input is on the way, so a
//! gate is normally released within one decode step; a watchdog thread releases any gate armed for longer than
//! `DEADLINE`, well inside the kernel's timeout (the GPU then computes garbage and the stage refuses to go on, so its
//! session ends), and a SIGINT, SIGTERM or SIGHUP sets every gate before the process ends (`qwen::gpu::register_gate`,
//! as for the 27B's tensor-parallel gates). A process killed otherwise leaves its gates armed, and the kernel resets
//! the GPU.
use crate::vulkan::Device;
use anyhow::{bail, Result};
use ash::vk;
use std::{
    sync::{Mutex, MutexGuard, Once},
    time::{Duration, Instant},
};

/// Longest a gate may stay armed.
pub const DEADLINE: Duration = Duration::from_secs(1);

/// How often the watchdog looks.
const TICK: Duration = Duration::from_millis(20);

/// amdgpu's job timeout on the Macs' kernels.
const KERNEL_TIMEOUT: Duration = Duration::from_secs(2);

const _: () = assert!(
    DEADLINE.as_millis() + TICK.as_millis() <= KERNEL_TIMEOUT.as_millis() / 2 + 100,
    "the watchdog must release a gate long before the kernel resets the GPU"
);

struct Gate {
    dev: vk::Device,
    set: vk::PFN_vkSetEvent,
    event: vk::Event,
    armed: Option<Instant>,
    fired: bool,
}

static GATES: Mutex<Vec<Gate>> = Mutex::new(Vec::new());

/// The gates, also after a thread panicked while holding them: the watchdog must keep going.
fn gates() -> MutexGuard<'static, Vec<Gate>> {
    GATES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Watch `event`, a gate of `gpu` (also released on SIGINT, SIGTERM and SIGHUP).
pub fn register(gpu: &Device, event: vk::Event) {
    crate::qwen::gpu::register_gate(gpu, event);
    register_raw(gpu.dev.handle(), gpu.dev.fp_v1_0().set_event, event);
}

fn register_raw(dev: vk::Device, set: vk::PFN_vkSetEvent, event: vk::Event) {
    gates().push(Gate { dev, set, event, armed: None, fired: false });
}

/// Forget the gates of `gpu` (its device is going away).
pub fn unregister(gpu: &Device) {
    crate::qwen::gpu::unregister_gates(gpu);
    unregister_raw(gpu.dev.handle());
}

fn unregister_raw(dev: vk::Device) {
    gates().retain(|g| g.dev != dev);
}

/// Work waiting for `event` was submitted: release it after `DEADLINE` unless `disarm` comes first.
pub fn arm(event: vk::Event) {
    static WATCHDOG: Once = Once::new();
    WATCHDOG.call_once(|| {
        std::thread::Builder::new()
            .name("qwen4-gates".into())
            .spawn(|| loop {
                std::thread::sleep(TICK);
                let released = release_expired();
                // Printed after every expired gate is set and the lock is free (stderr may block).
                for _ in 0..released {
                    eprintln!("qwen4: work submitted ahead got no input for {DEADLINE:?}; released it (the stage stops)");
                }
            })
            .expect("starting the gate watchdog");
    });
    if let Some(g) = gates().iter_mut().find(|g| g.event == event) {
        (g.armed, g.fired) = (Some(Instant::now()), false);
    }
}

/// Set every gate armed for longer than `DEADLINE`; how many.
fn release_expired() -> usize {
    let mut released = 0;
    for g in gates().iter_mut() {
        if g.armed.is_some_and(|t| t.elapsed() >= DEADLINE) {
            let _ = unsafe { (g.set)(g.dev, g.event) };
            (g.armed, g.fired) = (None, true);
            released += 1;
        }
    }
    released
}

/// The host is about to release `event` itself; an error when the watchdog already did.
pub fn disarm(event: vk::Event) -> Result<()> {
    if let Some(g) = gates().iter_mut().find(|g| g.event == event) {
        g.armed = None;
        if std::mem::take(&mut g.fired) {
            bail!("work submitted ahead was released by the watchdog after {DEADLINE:?} without its input");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! The watchdog with simulated gates: a stand-in for vkSetEvent records when each event was set.
    use super::*;
    use ash::vk::Handle;
    use std::{sync::Arc, thread};

    static SETS: Mutex<Vec<(u64, Instant)>> = Mutex::new(Vec::new());

    unsafe extern "system" fn record_set(_dev: vk::Device, event: vk::Event) -> vk::Result {
        SETS.lock().unwrap().push((event.as_raw(), Instant::now()));
        vk::Result::SUCCESS
    }

    /// When the watchdog set `event`, at most once.
    fn set_at(event: vk::Event) -> Option<Instant> {
        let sets: Vec<Instant> = SETS.lock().unwrap().iter().filter(|s| s.0 == event.as_raw()).map(|s| s.1).collect();
        assert!(sets.len() <= 1, "event {event:?} set {} times", sets.len());
        sets.first().copied()
    }

    /// Simulated gates on their own simulated device (every test uses another `base`).
    fn simulated(base: u64, count: u64) -> (vk::Device, Vec<vk::Event>) {
        let dev = vk::Device::from_raw(base);
        let events: Vec<vk::Event> = (1..=count).map(|i| vk::Event::from_raw(base + i)).collect();
        for &e in &events {
            register_raw(dev, record_set, e);
        }
        (dev, events)
    }

    /// The release of a gate armed at `armed` came after `DEADLINE` and at most this late.
    const SLACK: Duration = Duration::from_millis(100);

    fn assert_released_in_time(event: vk::Event, armed: Instant) {
        let at = set_at(event).unwrap_or_else(|| panic!("{event:?} was never released"));
        let after = at - armed;
        assert!(after >= DEADLINE && after <= DEADLINE + SLACK, "{event:?} released after {after:?}");
    }

    #[test]
    fn a_gate_left_armed_is_released_once_after_the_deadline() {
        let (dev, ev) = simulated(0x1000, 1);
        let armed = Instant::now();
        arm(ev[0]);
        thread::sleep(DEADLINE / 2);
        assert!(set_at(ev[0]).is_none(), "released before the deadline");
        thread::sleep(DEADLINE + SLACK);
        assert_released_in_time(ev[0], armed);
        // The stage learns that its work ran without input, once.
        assert!(disarm(ev[0]).is_err());
        assert!(disarm(ev[0]).is_ok());
        unregister_raw(dev);
    }

    #[test]
    fn a_gate_disarmed_in_time_is_left_alone() {
        let (dev, ev) = simulated(0x2000, 1);
        arm(ev[0]);
        thread::sleep(DEADLINE * 9 / 10);
        disarm(ev[0]).unwrap();
        thread::sleep(DEADLINE + SLACK);
        assert!(set_at(ev[0]).is_none(), "a disarmed gate was released");
        unregister_raw(dev);
    }

    #[test]
    fn arming_again_restarts_the_deadline() {
        let (dev, ev) = simulated(0x3000, 1);
        arm(ev[0]);
        thread::sleep(DEADLINE * 3 / 4);
        disarm(ev[0]).unwrap();
        let armed = Instant::now();
        arm(ev[0]);
        thread::sleep(DEADLINE * 3 / 4);
        assert!(set_at(ev[0]).is_none(), "the first arming's deadline released the second");
        thread::sleep(DEADLINE / 2 + SLACK);
        assert_released_in_time(ev[0], armed);
        unregister_raw(dev);
    }

    #[test]
    fn every_abandoned_gate_is_released_in_time_while_others_cycle() {
        // Twelve stages' gates armed and abandoned 25 ms apart, while four threads arm and disarm their own gates
        // every millisecond (decode steps); the cycling gates are never released, every abandoned one in time.
        let (dev, abandoned) = simulated(0x4000, 12);
        let (cdev, cycling) = simulated(0x5000, 4);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let threads: Vec<_> = cycling
            .iter()
            .map(|&e| {
                let stop = stop.clone();
                thread::spawn(move || {
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        arm(e);
                        thread::sleep(Duration::from_millis(1));
                        disarm(e).expect("a cycling gate was released");
                    }
                })
            })
            .collect();
        let mut armed = Vec::new();
        for &e in &abandoned {
            armed.push(Instant::now());
            arm(e);
            thread::sleep(Duration::from_millis(25));
        }
        thread::sleep(DEADLINE + SLACK);
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for t in threads {
            t.join().unwrap();
        }
        for (&e, &t) in abandoned.iter().zip(&armed) {
            assert_released_in_time(e, t);
        }
        for &e in &cycling {
            assert!(set_at(e).is_none(), "a cycling gate was released");
        }
        unregister_raw(dev);
        unregister_raw(cdev);
    }

    #[test]
    fn a_panic_while_holding_the_gates_does_not_stop_the_watchdog() {
        let (dev, ev) = simulated(0x6000, 1);
        let _ = thread::spawn(|| {
            let _held = GATES.lock().unwrap();
            panic!("poisoning the gates on purpose");
        })
        .join();
        let armed = Instant::now();
        arm(ev[0]);
        thread::sleep(DEADLINE + SLACK + TICK);
        assert_released_in_time(ev[0], armed);
        unregister_raw(dev);
    }
}
