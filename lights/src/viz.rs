//! Music visualizer: frames for a row of Macs from the audio's spectrum.

use crate::audio::Spectrum;
use crate::panel::{COUNT, Frame, LEDS, ROWS, idx};

#[derive(Clone, Copy, PartialEq)]
pub enum Style {
    /// One frequency band per Mac, bars rising from the power light; left/right LEDs follow the channels
    Spectrum,
    /// Every Mac a stereo VU meter
    Meter,
    /// Bass pulses that ripple along the row on each kick, over a dim spectrum
    Wave,
}

impl Style {
    pub fn parse(s: &str) -> Option<Style> {
        match s {
            "spectrum" => Some(Style::Spectrum),
            "meter" => Some(Style::Meter),
            "wave" => Some(Style::Wave),
            _ => None,
        }
    }
}

pub struct Viz {
    style: Style,
    columns: usize,
    /// Falling peak marker per column, 0..1
    peaks: Vec<f32>,
    /// Ripple fronts: position along the row (columns) and strength
    ripples: Vec<(f32, f32)>,
}

/// Height of the bar for `h` at LED `i`: 1 below the bar's top, fractional at its top row, 0 above.
fn bar(i: usize, h: f32) -> f32 {
    let from_bottom = (ROWS - 1 - LEDS[i].y) as f32; // power light 0, audio jacks ROWS - 1
    (h * ROWS as f32 - from_bottom).clamp(0.0, 1.0)
}

fn row_of(h: f32) -> f32 {
    (h * ROWS as f32).floor().min(ROWS as f32 - 1.0)
}

impl Viz {
    pub fn new(style: Style, columns: usize) -> Viz {
        Viz {
            style,
            columns,
            peaks: vec![0.0; columns],
            ripples: Vec::new(),
        }
    }

    /// The band shown by `column`: the spectrum's bands spread over the row.
    fn band(&self, values: &[f32], column: usize) -> f32 {
        let per = values.len() as f32 / self.columns as f32;
        let a = (column as f32 * per) as usize;
        let b = (((column + 1) as f32 * per) as usize)
            .max(a + 1)
            .min(values.len());
        values[a..b].iter().copied().fold(0.0, f32::max)
    }

    pub fn render(&mut self, s: &Spectrum, dt: f32) -> Vec<Frame> {
        match self.style {
            Style::Spectrum => (0..self.columns)
                .map(|c| self.spectrum(s, c, dt, 1.0))
                .collect(),
            Style::Meter => (0..self.columns).map(|c| self.meter(s, c, dt)).collect(),
            Style::Wave => self.wave(s, dt),
        }
    }

    fn spectrum(&mut self, s: &Spectrum, c: usize, dt: f32, gain: f32) -> Frame {
        let (mono, left, right) = (
            self.band(&s.mono, c),
            self.band(&s.left, c),
            self.band(&s.right, c),
        );
        self.columns_bars(c, [left, mono, right], dt, gain)
    }

    fn meter(&mut self, s: &Spectrum, c: usize, dt: f32) -> Frame {
        let [l, r] = s.level;
        self.columns_bars(c, [l, 0.5 * (l + r), r], dt, 1.0)
    }

    /// Bars for the left, center and right LEDs, and the column's falling peak marker.
    fn columns_bars(
        &mut self,
        c: usize,
        [left, mono, right]: [f32; 3],
        dt: f32,
        gain: f32,
    ) -> Frame {
        let peak = &mut self.peaks[c];
        *peak = (*peak - 0.6 * dt).max(mono);
        let peak_row = row_of(*peak);
        let mut f = [0.0; COUNT];
        for (i, led) in LEDS.iter().enumerate() {
            let h = if led.x < 0.0 {
                left
            } else if led.x > 0.0 {
                right
            } else {
                mono
            };
            let mut v = bar(i, h);
            if *peak > 0.05 && (ROWS - 1 - led.y) as f32 == peak_row {
                v = v.max(0.6);
            }
            f[i] = v * gain;
        }
        f
    }

    fn wave(&mut self, s: &Spectrum, dt: f32) -> Vec<Frame> {
        let speed = 10.0; // columns per second
        if s.beat {
            let bass = s
                .mono
                .iter()
                .take((s.mono.len() / 4).max(1))
                .copied()
                .fold(0.0, f32::max);
            self.ripples.push((0.0, 0.4 + 0.6 * bass));
        }
        for r in &mut self.ripples {
            r.0 += speed * dt;
            r.1 *= (-dt / 0.6).exp();
        }
        let end = self.columns as f32 + 1.0;
        self.ripples.retain(|r| r.0 < end && r.1 > 0.02);
        let mut frames = Vec::with_capacity(self.columns);
        for c in 0..self.columns {
            let mut f = self.spectrum(s, c, dt, 0.25);
            let hit: f32 = self
                .ripples
                .iter()
                .map(|&(pos, strength)| strength * (1.0 - (pos - c as f32).abs()).max(0.0))
                .sum();
            if hit > 0.0 {
                for (i, v) in f.iter_mut().enumerate() {
                    // the edges of each port box ring brightest
                    let weight = if idx::EDGES.contains(&i)
                        || idx::ROWS3.iter().any(|r| r[0] == i || r[2] == i)
                    {
                        1.0
                    } else {
                        0.6
                    };
                    *v = v.max((hit * weight).min(1.0));
                }
            }
            frames.push(f);
        }
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bars_fill_from_the_power_light() {
        assert_eq!(bar(idx::POWER, 0.1), 1.0);
        assert_eq!(bar(idx::LINE_OUT, 0.1), 0.0);
        assert_eq!(bar(idx::LINE_OUT, 1.0), 1.0);
        assert_eq!(bar(idx::HDMI, 0.0), 0.0);
    }
}
