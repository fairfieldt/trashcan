//! Audio in (raw s16le stereo from a command) and its spectrum.

use std::collections::VecDeque;
use std::f32::consts::PI;
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

pub struct Capture {
    rx: mpsc::Receiver<Vec<i16>>,
    child: Child,
}

impl Capture {
    /// Runs `cmd` with sh; its stdout must be s16le stereo at `rate`. A source faster than real time (a file)
    /// is paced to real time.
    pub fn start(cmd: &str, rate: u32) -> std::io::Result<Capture> {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .stdout(Stdio::piped())
            .spawn()?;
        let mut out = child.stdout.take().expect("piped stdout");
        let (tx, rx) = mpsc::sync_channel(64);
        thread::spawn(move || {
            let start = Instant::now();
            let mut frames: u64 = 0;
            let mut buf = vec![0u8; 4096];
            let mut carry: Option<u8> = None;
            loop {
                let n = match out.read(&mut buf) {
                    Ok(0) | Err(_) => return,
                    Ok(n) => n,
                };
                let mut bytes: Vec<u8> = carry.take().into_iter().collect();
                bytes.extend_from_slice(&buf[..n]);
                if bytes.len() % 2 == 1 {
                    carry = bytes.pop();
                }
                let samples: Vec<i16> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| i16::from_le_bytes([b[0], b[1]]))
                    .collect();
                frames += samples.len() as u64 / 2;
                if tx.send(samples).is_err() {
                    return;
                }
                let ahead = frames as f64 / rate as f64 - start.elapsed().as_secs_f64();
                if ahead > 0.02 {
                    thread::sleep(Duration::from_secs_f64(ahead - 0.01));
                }
            }
        });
        Ok(Capture { rx, child })
    }

    /// Samples received since the last call (interleaved left, right). None once the source has ended.
    pub fn take(&mut self) -> Option<Vec<i16>> {
        let mut all = Vec::new();
        loop {
            match self.rx.try_recv() {
                Ok(s) => all.extend(s),
                Err(mpsc::TryRecvError::Empty) => return Some(all),
                Err(mpsc::TryRecvError::Disconnected) => {
                    return if all.is_empty() { None } else { Some(all) };
                }
            }
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

const N: usize = 2048;

/// In-place radix-2 FFT of `re`/`im` (length N).
fn fft(re: &mut [f32; N], im: &mut [f32; N]) {
    let mut j = 0;
    for i in 1..N {
        let mut bit = N >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= N {
        let ang = -2.0 * PI / len as f32;
        for start in (0..N).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (ang * k as f32).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * c - im[b] * s;
                let ti = re[b] * s + im[b] * c;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

pub struct Spectrum {
    /// Per band, 0..1: both channels, left, right
    pub mono: Vec<f32>,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    /// Overall loudness 0..1 per channel
    pub level: [f32; 2],
    /// A kick drum (bass onset) in this frame
    pub beat: bool,
}

pub struct Analyzer {
    rate: f32,
    left: VecDeque<f32>,
    right: VecDeque<f32>,
    window: Vec<f32>,
    /// FFT bin range per band
    bands: Vec<(usize, usize)>,
    /// Per-band weight: +3 dB/octave, since music has less power per bin higher up
    tilt: Vec<f32>,
    /// Slowly decaying maximum weighted band power per channel (automatic gain)
    max: [f32; 3],
    shown: [Vec<f32>; 3],
    level_max: f32,
    level: [f32; 2],
    bass_history: VecDeque<f32>,
    since_beat: f32,
}

impl Analyzer {
    pub fn new(rate: u32, bands: usize) -> Analyzer {
        let rate = rate as f32;
        let (lo, hi) = (40.0f32, 14000.0f32.min(rate / 2.0));
        let bin = |f: f32| ((f / rate * N as f32).round() as usize).clamp(1, N / 2 - 1);
        let edges: Vec<usize> = (0..=bands)
            .map(|i| bin(lo * (hi / lo).powf(i as f32 / bands as f32)))
            .collect();
        let ranges: Vec<(usize, usize)> = (0..bands)
            .map(|i| (edges[i], edges[i + 1].max(edges[i] + 1)))
            .collect();
        let tilt = ranges
            .iter()
            .map(|&(a, b)| (a + b) as f32 / 2.0 * rate / N as f32 / 1000.0)
            .collect();
        let window = (0..N)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (N - 1) as f32).cos())
            .collect();
        Analyzer {
            rate,
            left: VecDeque::from(vec![0.0; N]),
            right: VecDeque::from(vec![0.0; N]),
            window,
            bands: ranges,
            tilt,
            max: [1e-7; 3],
            shown: std::array::from_fn(|_| vec![0.0; bands]),
            level_max: 1e-4,
            level: [0.0; 2],
            bass_history: VecDeque::new(),
            since_beat: 1.0,
        }
    }

    pub fn push(&mut self, interleaved: &[i16]) {
        for pair in interleaved.as_chunks::<2>().0 {
            self.left.push_back(pair[0] as f32 / 32768.0);
            self.right.push_back(pair[1] as f32 / 32768.0);
        }
        while self.left.len() > N {
            self.left.pop_front();
            self.right.pop_front();
        }
    }

    fn band_powers(&self, signal: impl Fn(usize) -> f32) -> Vec<f32> {
        let mut re = [0.0f32; N];
        let mut im = [0.0f32; N];
        for i in 0..N {
            re[i] = signal(i) * self.window[i];
        }
        fft(&mut re, &mut im);
        self.bands
            .iter()
            .map(|&(a, b)| {
                (a..b).map(|k| re[k] * re[k] + im[k] * im[k]).sum::<f32>() / (b - a) as f32
            })
            .collect()
    }

    /// The spectrum of the latest N samples, `dt` seconds after the previous call.
    pub fn analyze(&mut self, dt: f32) -> Spectrum {
        let powers = [
            self.band_powers(|i| 0.5 * (self.left[i] + self.right[i])),
            self.band_powers(|i| self.left[i]),
            self.band_powers(|i| self.right[i]),
        ];
        let rms = |ch: &VecDeque<f32>| (ch.iter().map(|v| v * v).sum::<f32>() / N as f32).sqrt();
        let rms = [rms(&self.left), rms(&self.right)];
        // silence: let everything fall instead of amplifying the noise floor
        let silent = rms[0].max(rms[1]) < 1e-3;

        let decay = (-dt / 8.0).exp(); // gain recovers over ~8 s after a loud passage
        let attack = 1.0 - (-dt / 0.03).exp();
        let release = 1.0 - (-dt / 0.25).exp();
        let mut out: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        for c in 0..3 {
            let weighted: Vec<f32> = powers[c]
                .iter()
                .zip(&self.tilt)
                .map(|(p, w)| p * w)
                .collect();
            let peak = weighted.iter().copied().fold(0.0, f32::max);
            self.max[c] = (self.max[c] * decay).max(peak).max(1e-7);
            for (b, &p) in weighted.iter().enumerate() {
                // 30 dB of range below the recent loudest band
                let v = if silent {
                    0.0
                } else {
                    (1.0 + 10.0 * (p / self.max[c]).max(1e-9).log10() / 30.0).clamp(0.0, 1.0)
                };
                let s = &mut self.shown[c][b];
                *s += (v - *s) * if v > *s { attack } else { release };
                out[c].push(*s);
            }
        }

        self.level_max = (self.level_max * decay).max(rms[0]).max(rms[1]).max(1e-4);
        for ch in 0..2 {
            let v = if silent {
                0.0
            } else {
                (rms[ch] / self.level_max).sqrt()
            };
            self.level[ch] +=
                (v - self.level[ch]) * if v > self.level[ch] { attack } else { release };
        }

        // a kick: bass power well above its average over the last second
        let bass = powers[0]
            .iter()
            .take((powers[0].len() / 6).max(1))
            .sum::<f32>();
        let average = if self.bass_history.is_empty() {
            bass
        } else {
            self.bass_history.iter().sum::<f32>() / self.bass_history.len() as f32
        };
        self.bass_history.push_back(bass);
        while self.bass_history.len() as f32 > 1.0 / dt.max(1e-3) {
            self.bass_history.pop_front();
        }
        self.since_beat += dt;
        let beat = !silent && bass > 1.5 * average && self.since_beat > 0.25;
        if beat {
            self.since_beat = 0.0;
        }
        let _ = self.rate;
        let [mono, left, right] = out;
        Spectrum {
            mono,
            left,
            right,
            level: self.level,
            beat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_finds_a_sine() {
        let mut re = [0.0f32; N];
        let mut im = [0.0f32; N];
        for (i, v) in re.iter_mut().enumerate() {
            *v = (2.0 * PI * 64.0 * i as f32 / N as f32).sin();
        }
        fft(&mut re, &mut im);
        let mag: Vec<f32> = (0..N / 2)
            .map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt())
            .collect();
        let peak = mag
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(peak, 64);
    }

    #[test]
    fn a_tone_lights_its_band() {
        let rate = 48000;
        let mut a = Analyzer::new(rate, 6);
        let mut t = 0usize;
        let mut spec = None;
        for _ in 0..40 {
            let block: Vec<i16> = (0..1200)
                .flat_map(|_| {
                    t += 1;
                    let v = ((2.0 * PI * 1000.0 * t as f32 / rate as f32).sin() * 8000.0) as i16;
                    [v, v]
                })
                .collect();
            a.push(&block);
            spec = Some(a.analyze(0.025));
        }
        let s = spec.unwrap();
        let loudest = s
            .mono
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .unwrap()
            .0;
        // 40 Hz-14 kHz in 6 log bands: 1 kHz falls in band 3
        assert_eq!(loudest, 3, "{:?}", s.mono);
    }
}
