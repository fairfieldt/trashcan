//! The Mac Pro 6,1's port lights, through applesmc's `port_lights_levels` (trashcan kernel patches 0009 and 0010).

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;

pub const COUNT: usize = 20;
/// Rows from the top of the I/O panel (the audio jacks, 0) to the bottom (the power light, ROWS - 1).
pub const ROWS: u8 = 12;
const DIR: &str = "/sys/devices/platform/applesmc.768";

pub struct Led {
    pub name: &'static str,
    /// -1 left, 0 center, 1 right (seen from behind the Mac)
    pub x: f32,
    pub y: u8,
}

/// In SMC order (the driver's LED index).
pub const LEDS: [Led; COUNT] = [
    Led {
        name: "line-out",
        x: -0.5,
        y: 0,
    },
    Led {
        name: "headphone",
        x: 0.5,
        y: 0,
    },
    Led {
        name: "usb",
        x: 0.0,
        y: 2,
    },
    Led {
        name: "thunderbolt",
        x: 0.0,
        y: 5,
    },
    Led {
        name: "ethernet",
        x: 0.0,
        y: 8,
    },
    Led {
        name: "hdmi",
        x: 0.0,
        y: 10,
    },
    Led {
        name: "usb-box-right",
        x: 1.0,
        y: 2,
    },
    Led {
        name: "usb-box-left",
        x: -1.0,
        y: 2,
    },
    Led {
        name: "thunderbolt-box-right",
        x: 1.0,
        y: 5,
    },
    Led {
        name: "thunderbolt-box-left",
        x: -1.0,
        y: 5,
    },
    Led {
        name: "ethernet-box-right",
        x: 1.0,
        y: 8,
    },
    Led {
        name: "ethernet-box-left",
        x: -1.0,
        y: 8,
    },
    Led {
        name: "usb-box-top",
        x: 0.0,
        y: 1,
    },
    Led {
        name: "usb-box-bottom",
        x: 0.0,
        y: 3,
    },
    Led {
        name: "thunderbolt-box-top",
        x: 0.0,
        y: 4,
    },
    Led {
        name: "thunderbolt-box-bottom",
        x: 0.0,
        y: 6,
    },
    Led {
        name: "ethernet-box-top",
        x: 0.0,
        y: 7,
    },
    Led {
        name: "ethernet-box-bottom",
        x: 0.0,
        y: 9,
    },
    Led {
        name: "power",
        x: 0.0,
        y: 11,
    },
    Led {
        name: "power-2",
        x: 0.0,
        y: 11,
    },
];

/// Indices by role, for effects that address the panel by function.
pub mod idx {
    pub const LINE_OUT: usize = 0;
    pub const HEADPHONE: usize = 1;
    pub const HDMI: usize = 5;
    pub const POWER: usize = 18;
    pub const POWER_2: usize = 19;
    /// Left, center, right of each port box: USB, Thunderbolt, Ethernet.
    pub const ROWS3: [[usize; 3]; 3] = [[7, 2, 6], [9, 3, 8], [11, 4, 10]];
    /// Top and bottom of each box.
    pub const EDGES: [usize; 6] = [12, 13, 14, 15, 16, 17];
}

/// Linear brightness 0..1 per LED.
pub type Frame = [f32; COUNT];

/// Brightness 0..1 to the driver's 0..255, with gamma so fades look even.
pub fn encode(v: f32) -> u8 {
    (v.clamp(0.0, 1.0).powf(2.2) * 255.0).round() as u8
}

pub struct Panel {
    levels: File,
    control: File,
    power: Option<File>,
    shown: Option<[u8; COUNT]>,
}

impl Panel {
    pub fn open() -> io::Result<Panel> {
        let open = |name: &str, read: bool| {
            OpenOptions::new()
                .read(read)
                .write(true)
                .open(format!("{DIR}/{name}"))
        };
        Ok(Panel {
            levels: open("port_lights_levels", false)?,
            control: open("port_lights", true)?,
            power: open("power_light", true).ok(),
            shown: None,
        })
    }

    /// Shows a frame. LEDs that moved by less than a step keep their value: each change costs an SMC
    /// transaction (about 0.4 ms).
    pub fn show(&mut self, frame: &Frame) -> io::Result<()> {
        let mut next = self.shown.unwrap_or([0; COUNT]);
        let mut changed = self.shown.is_none();
        for (i, &v) in frame.iter().enumerate() {
            let want = encode(v);
            let have = next[i];
            if want != have && (want.abs_diff(have) >= 3 || want == 0 || want == 255) {
                next[i] = want;
                changed = true;
            }
        }
        if changed {
            let text: Vec<String> = next.iter().map(|v| v.to_string()).collect();
            self.levels.write_at(text.join(" ").as_bytes(), 0)?;
            self.shown = Some(next);
        }
        Ok(())
    }

    /// Fails once the driver has gone away (unloaded or reloaded): these files then return errors.
    pub fn check(&self) -> io::Result<()> {
        let mut buf = [0u8; 16];
        self.control.read_at(&mut buf, 0).map(|_| ())
    }

    /// Hands the lights back to the SMC (they light up on motion again).
    pub fn release(&mut self) -> io::Result<()> {
        self.shown = None;
        self.control.write_at(b"auto", 0).map(|_| ())
    }

    /// The power light's mode while the lights are the SMC's: off, on, breathe or bright.
    pub fn power_light(&mut self, mode: &str) -> io::Result<()> {
        match &self.power {
            Some(f) => f.write_at(mode.as_bytes(), 0).map(|_| ()),
            None => Ok(()),
        }
    }
}
