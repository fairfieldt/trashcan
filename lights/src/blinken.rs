//! Blinkenlights: a 1950s computer console driven by what the Mac is doing.
//!
//! The USB, Thunderbolt and Ethernet rows are 3-bit registers that count while GPU 0, GPU 1 and the network
//! work; the box edges and HDMI flicker like core memory; the audio jacks alternate like RUN/WAIT lamps; the
//! power lamp stays lit. Lamps glow up and cool down like incandescent bulbs.

use crate::panel::{COUNT, Frame, idx};
use crate::rng::Rng;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// GPU job and network packet counters (debugfs needs root).
pub struct Activity {
    fence_files: Vec<PathBuf>,
    last_fences: Vec<u64>,
    last_packets: u64,
    last: Instant,
    discovered: Instant,
}

#[derive(Clone, Copy, Default)]
pub struct Sample {
    /// Fences signaled per second on each GPU (all rings)
    pub gpu: [f32; 2],
    /// Network packets per second (all interfaces but loopback and Tailscale)
    pub net: f32,
}

impl Sample {
    pub fn gpu_total(&self) -> f32 {
        self.gpu[0] + self.gpu[1]
    }
}

/// The first two GPUs' fence files.
fn fence_files() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = (0..16)
        .map(|n| PathBuf::from(format!("/sys/kernel/debug/dri/{n}/amdgpu_fence_info")))
        .filter(|p| p.exists())
        .collect();
    files.truncate(2);
    files
}

impl Activity {
    pub fn new() -> Activity {
        let fence_files = fence_files();
        let last_fences = fence_files.iter().map(read_fences).collect();
        Activity {
            fence_files,
            last_fences,
            last_packets: read_packets(),
            last: Instant::now(),
            discovered: Instant::now(),
        }
    }

    pub fn gpus(&self) -> usize {
        self.fence_files.len()
    }

    /// Looks for the GPUs again every 10 s while fewer than two are watched or one has gone (amdgpu loaded after
    /// the node started, or reloaded). True if the set changed.
    fn rediscover(&mut self) -> bool {
        let lost = self.fence_files.iter().any(|p| !p.exists());
        if !(lost || self.fence_files.len() < 2) || self.discovered.elapsed().as_secs() < 10 {
            return false;
        }
        self.discovered = Instant::now();
        let files = fence_files();
        if files == self.fence_files {
            return false;
        }
        self.last_fences = files.iter().map(read_fences).collect();
        self.fence_files = files;
        true
    }

    /// Rates since the previous call.
    pub fn sample(&mut self) -> Sample {
        if self.rediscover() {
            println!("portlights: {} GPUs watched", self.gpus());
        }
        let dt = self.last.elapsed().as_secs_f32().max(1e-3);
        self.last = Instant::now();
        let mut s = Sample::default();
        for (i, p) in self.fence_files.iter().enumerate() {
            let now = read_fences(p);
            // a counter that went back (driver reloaded, a ring wrapped) counts as no work this time
            s.gpu[i] = now.saturating_sub(self.last_fences[i]) as f32 / dt;
            self.last_fences[i] = now;
        }
        let packets = read_packets();
        s.net = packets.saturating_sub(self.last_packets) as f32 / dt;
        self.last_packets = packets;
        s
    }
}

/// Sum of "Last signaled fence" over a GPU's rings.
fn read_fences(path: &PathBuf) -> u64 {
    let text = fs::read_to_string(path).unwrap_or_default();
    text.lines()
        .filter_map(|l| l.strip_prefix("Last signaled fence"))
        .filter_map(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok())
        .sum()
}

fn read_packets() -> u64 {
    let text = fs::read_to_string("/proc/net/dev").unwrap_or_default();
    text.lines()
        .skip(2)
        .filter_map(|l| l.split_once(':'))
        .filter(|(name, _)| {
            let name = name.trim();
            name != "lo" && !name.starts_with("tailscale")
        })
        .map(|(_, v)| {
            let f: Vec<u64> = v
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            f.get(1).copied().unwrap_or(0) + f.get(9).copied().unwrap_or(0)
        })
        .sum()
}

pub struct Blinken {
    glow: Frame,
    lit: [bool; COUNT],
    registers: [u8; 3],
    since_tick: f32,
    ticks: u64,
    rng: Rng,
}

impl Blinken {
    pub fn new(seed: u64) -> Blinken {
        Blinken {
            glow: [0.0; COUNT],
            lit: [false; COUNT],
            registers: [0; 3],
            since_tick: 0.0,
            ticks: 0,
            rng: Rng::new(seed),
        }
    }

    /// Advances by `dt` seconds. `busy` 0..1 sets the pace; at 0 every lamp goes out.
    pub fn render(&mut self, dt: f32, busy: f32, s: &Sample) -> Frame {
        let running = busy > 0.0;
        let rate = 3.0 + 9.0 * busy;
        self.since_tick += dt;
        if self.since_tick >= 1.0 / rate {
            self.since_tick = 0.0;
            self.ticks += 1;
            if running {
                self.tick(busy, s);
            } else {
                self.lit = [false; COUNT];
            }
        }
        for i in 0..COUNT {
            let target = if self.lit[i] { 1.0 } else { 0.0 };
            // incandescent: about 30 ms to light, 90 ms to cool
            let tau = if target > self.glow[i] { 0.03 } else { 0.09 };
            self.glow[i] += (target - self.glow[i]) * (1.0 - (-dt / tau).exp());
        }
        self.glow
    }

    /// True once every lamp has cooled down.
    pub fn dark(&self) -> bool {
        self.glow.iter().all(|&g| g < 0.01)
    }

    fn tick(&mut self, busy: f32, s: &Sample) {
        let counting = [s.gpu[0] > 0.5, s.gpu[1] > 0.5, s.net > 20.0];
        for (r, &on) in counting.iter().enumerate() {
            if on {
                self.registers[r] = self.registers[r].wrapping_add(1) & 7;
            }
            for (bit, &led) in idx::ROWS3[r].iter().enumerate() {
                self.lit[led] = self.registers[r] & (4 >> bit) != 0;
            }
        }
        let flip = 0.15 + 0.5 * busy;
        for &led in idx::EDGES.iter().chain([idx::HDMI].iter()) {
            if self.rng.chance(flip) {
                self.lit[led] = !self.lit[led];
            }
        }
        let run = self.ticks.is_multiple_of(2);
        self.lit[idx::LINE_OUT] = run;
        self.lit[idx::HEADPHONE] = !run;
        self.lit[idx::POWER] = true;
        self.lit[idx::POWER_2] = self.rng.chance(0.3 + 0.6 * busy);
    }
}

/// How busy the GPUs are, 0..1: 1 fence/s barely registers, 1000/s is flat out.
pub fn busyness(s: &Sample) -> f32 {
    ((1.0 + s.gpu_total()).log10() / 3.0).clamp(0.0, 1.0)
}
