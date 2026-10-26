//! portlights: the Mac Pro 6,1's port lights as blinkenlights while the GPUs work, and as a music visualizer
//! across a row of Macs.

mod audio;
mod blinken;
mod net;
mod panel;
mod rng;
mod viz;

use panel::{COUNT, Frame, Panel};
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const USAGE: &str = "usage:
  portlights node [--idle auto|blinken|off] [--idle-power breathe|on|off|bright] [--listen [ADDR:]PORT]
      run on each Mac: blinkenlights while the GPUs work, otherwise the lights are the SMC's
      (--idle auto, the power light breathing) or dark (off); with --listen, also frames from a conductor
      (UDP, port 7465 by default)
  portlights conduct [--row FILE | --nodes A,B,...] [--style spectrum|meter|wave] [--audio CMD] [--rate HZ] [--fps N]
      visualize audio across the Macs in the row, left to right as seen from behind ('-' for a gap);
      the row file (default ~/.config/portlights/row, else /etc/portlights/row; one Mac per line, HOST or
      HOST:PORT) is re-read when it changes; CMD prints s16le stereo at HZ (default: parec from the default
      output's monitor)
  portlights blinken [--busy 0-1] [--secs N]   blinkenlights here, without GPU work
  portlights test                              light the LEDs one by one, in order
  portlights auto                              give the lights back to the SMC";

const SYSTEM_ROW: &str = "/etc/portlights/row";

/// The row file: ~/.config/portlights/row (or under $XDG_CONFIG_HOME) if it exists, else /etc/portlights/row.
fn default_row() -> String {
    let config = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|d| !d.is_empty())
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.config")));
    if let Some(dir) = config {
        let user = format!("{dir}/portlights/row");
        if std::path::Path::new(&user).exists() {
            return user;
        }
    }
    SYSTEM_ROW.into()
}

const DEFAULT_AUDIO: &str = "parec --device=@DEFAULT_MONITOR@ --format=s16le --rate=48000 --channels=2 --raw --latency-msec=20";

static STOP: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    fn signal(signum: i32, handler: usize) -> usize;
}

extern "C" fn on_signal(_: i32) {
    STOP.store(true, Ordering::SeqCst);
}

fn stopping() -> bool {
    STOP.load(Ordering::SeqCst)
}

fn catch_signals() {
    for sig in [2, 15] {
        // SIGINT, SIGTERM
        unsafe { signal(sig, on_signal as extern "C" fn(i32) as usize) };
    }
}

struct Args(Vec<String>);

impl Args {
    fn take(&mut self, flag: &str) -> Option<String> {
        let i = self.0.iter().position(|a| a == flag)?;
        if i + 1 >= self.0.len() {
            die(&format!("{flag} needs a value"));
        }
        let v = self.0.remove(i + 1);
        self.0.remove(i);
        Some(v)
    }

    fn num<T: std::str::FromStr>(&mut self, flag: &str, default: T) -> T {
        match self.take(flag) {
            Some(v) => v
                .parse()
                .unwrap_or_else(|_| die(&format!("{flag}: not a number: {v}"))),
            None => default,
        }
    }

    fn done(&self) {
        if let Some(a) = self.0.first() {
            die(&format!("unexpected argument {a}"));
        }
    }
}

fn die(msg: &str) -> ! {
    eprintln!("portlights: {msg}\n{USAGE}");
    std::process::exit(2);
}

fn open_panel() -> Panel {
    Panel::open().unwrap_or_else(|e| {
        eprintln!("portlights: no port lights ({e}); needs applesmc with trashcan kernel patches 0009 and 0010");
        std::process::exit(1);
    })
}

/// Calls `f` every 1/fps seconds with the time since the previous call, until a signal arrives.
fn every(fps: f32, mut f: impl FnMut(f32) -> bool) {
    let period = Duration::from_secs_f32(1.0 / fps);
    let mut next = Instant::now();
    let mut last = Instant::now();
    while !stopping() {
        let dt = last.elapsed().as_secs_f32();
        last = Instant::now();
        if !f(dt) {
            return;
        }
        next += period;
        let now = Instant::now();
        if next > now {
            thread::sleep(next - now);
        } else {
            next = now;
        }
    }
}

fn main() {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        die("no command");
    }
    let cmd = argv.remove(0);
    let mut args = Args(argv);
    catch_signals();
    match cmd.as_str() {
        "node" => {
            let idle = args.take("--idle").unwrap_or_else(|| "auto".into());
            let idle_power = args
                .take("--idle-power")
                .unwrap_or_else(|| "breathe".into());
            let listen = args.take("--listen").map(|a| listen_addr(&a));
            args.done();
            if !["auto", "blinken", "off"].contains(&idle.as_str()) {
                die("--idle is auto, blinken or off");
            }
            if !["off", "on", "breathe", "bright"].contains(&idle_power.as_str()) {
                die("--idle-power is off, on, breathe or bright");
            }
            node(&idle, &idle_power, listen);
        }
        "conduct" => {
            let nodes = args.take("--nodes");
            let row_file = args.take("--row").unwrap_or_else(default_row);
            let style = args.take("--style").unwrap_or_else(|| "spectrum".into());
            let style = viz::Style::parse(&style)
                .unwrap_or_else(|| die("--style is spectrum, meter or wave"));
            let audio = args.take("--audio").unwrap_or_else(|| DEFAULT_AUDIO.into());
            let rate = args.num("--rate", 48000u32);
            let fps = args.num("--fps", 40.0f32);
            args.done();
            conduct(Row::new(nodes, row_file), style, &audio, rate, fps);
        }
        "blinken" => {
            let busy = args.num("--busy", 0.6f32).clamp(0.05, 1.0);
            let secs = args.num("--secs", 30.0f32);
            args.done();
            let mut panel = open_panel();
            let mut b = blinken::Blinken::new(rng::seed());
            let sample = blinken::Sample {
                gpu: [10f32.powf(3.0 * busy); 2],
                net: 100.0,
            };
            let start = Instant::now();
            every(50.0, |dt| {
                let busy = if start.elapsed().as_secs_f32() < secs {
                    busy
                } else {
                    0.0
                };
                let frame = b.render(dt, busy, &sample);
                let _ = panel.show(&frame);
                busy > 0.0 || !b.dark()
            });
            let _ = panel.release();
        }
        "test" => {
            args.done();
            let mut panel = open_panel();
            let mut frame: Frame = [0.0; COUNT];
            let _ = panel.show(&frame);
            for (i, led) in panel::LEDS.iter().enumerate() {
                if stopping() {
                    break;
                }
                println!("{i:2} {}", led.name);
                frame[i] = 1.0;
                let _ = panel.show(&frame);
                thread::sleep(Duration::from_millis(1500));
            }
            thread::sleep(Duration::from_secs(2));
            let _ = panel.release();
        }
        "auto" => {
            args.done();
            if let Err(e) = open_panel().release() {
                eprintln!("portlights: {e}");
                std::process::exit(1);
            }
        }
        _ => die(&format!("unknown command {cmd}")),
    }
}

/// Opens the port lights, waiting for them: applesmc may load after this starts at boot, or be reloaded.
fn wait_for_panel() -> Option<Panel> {
    let mut said = false;
    loop {
        match Panel::open() {
            Ok(p) => return Some(p),
            Err(e) => {
                if !said {
                    eprintln!(
                        "portlights: no port lights ({e}); needs applesmc with trashcan kernel patches 0009 and 0010"
                    );
                    said = true;
                }
                for _ in 0..10 {
                    if stopping() {
                        return None;
                    }
                    thread::sleep(Duration::from_secs(1));
                }
            }
        }
    }
}

/// `--listen` value: PORT (all interfaces) or ADDR:PORT.
fn listen_addr(arg: &str) -> SocketAddr {
    let target = if arg.parse::<u16>().is_ok() {
        format!("0.0.0.0:{arg}")
    } else {
        arg.to_string()
    };
    target
        .to_socket_addrs()
        .ok()
        .and_then(|mut a| a.next())
        .unwrap_or_else(|| die(&format!("--listen: not PORT or ADDR:PORT: {arg}")))
}

/// The node daemon: conductor frames first (with --listen), then blinkenlights while the GPUs work, then idle.
fn node(idle: &str, idle_power: &str, listen: Option<SocketAddr>) {
    let sock = listen.map(|addr| {
        let sock = UdpSocket::bind(addr).unwrap_or_else(|e| {
            eprintln!("portlights: cannot listen on UDP {addr}: {e}");
            std::process::exit(1);
        });
        sock.set_nonblocking(true).expect("nonblocking socket");
        sock
    });
    let Some(mut panel) = wait_for_panel() else {
        return;
    };
    let mut activity = blinken::Activity::new();
    let mut lamps = blinken::Blinken::new(rng::seed());
    let conductor = match listen {
        Some(addr) => format!("conductor frames on UDP {addr}"),
        None => "no conductor".into(),
    };
    println!(
        "portlights: node, {conductor}, {} GPUs watched, idle {idle} (power light {idle_power})",
        activity.gpus()
    );

    let mut remote: Option<(u32, Frame, Instant)> = None;
    let mut sample = blinken::Sample::default();
    let mut sampled = Instant::now();
    let mut active_at: Option<Instant> = None;
    let mut checked = Instant::now();
    let mut mode = "";
    let mut buf = [0u8; 64];
    every(50.0, |dt| {
        while let Some(Ok((n, _))) = sock.as_ref().map(|s| s.recv_from(&mut buf)) {
            if let Some((seq, frame)) = net::decode(&buf[..n])
                && remote
                    .as_ref()
                    .is_none_or(|r| net::newer(seq, r.0) || r.2.elapsed().as_secs() > 1)
            {
                remote = Some((seq, frame, Instant::now()));
            }
        }
        if sampled.elapsed() >= Duration::from_millis(100) {
            sampled = Instant::now();
            sample = activity.sample();
            if sample.gpu_total() >= 1.0 {
                active_at = Some(Instant::now());
            }
        }

        let (now_mode, result) = if let Some((_, frame, _)) =
            remote.filter(|r| r.2.elapsed() < Duration::from_millis(500))
        {
            ("conductor", panel.show(&frame))
        } else {
            let working = active_at.is_some_and(|t| t.elapsed() < Duration::from_secs(10));
            let busy = if working || idle == "blinken" {
                blinken::busyness(&sample).max(0.05)
            } else {
                0.0
            };
            let frame = lamps.render(dt, busy, &sample);
            if busy > 0.0 || !lamps.dark() {
                ("blinkenlights", panel.show(&frame))
            } else if idle == "off" {
                ("off", panel.show(&[0.0; COUNT]))
            } else if mode != "smc" {
                // the SMC's again: lit on motion, the power light breathing (or as --idle-power says)
                checked = Instant::now();
                (
                    "smc",
                    panel.release().and_then(|_| panel.power_light(idle_power)),
                )
            } else if checked.elapsed() >= Duration::from_secs(2) {
                // nothing is written while idle: notice a driver reload, which resets the power light
                checked = Instant::now();
                ("smc", panel.check())
            } else {
                ("smc", Ok(()))
            }
        };
        if now_mode != mode {
            println!("portlights: {now_mode}");
            mode = now_mode;
        }
        if let Err(e) = result {
            // the driver may have been reloaded: open it again and start over
            eprintln!("portlights: {e}");
            thread::sleep(Duration::from_secs(1));
            match wait_for_panel() {
                Some(p) => panel = p,
                None => return false,
            }
            mode = "";
        }
        true
    });
    let _ = panel.release();
    let _ = panel.power_light("on");
}

fn resolve(name: &str) -> Option<SocketAddr> {
    let target = if name.contains(':') {
        name.to_string()
    } else {
        format!("{name}:{}", net::PORT)
    };
    let addrs: Vec<SocketAddr> = target.to_socket_addrs().ok()?.collect();
    addrs
        .iter()
        .find(|a| a.is_ipv4())
        .or(addrs.first())
        .copied()
}

/// Parses a row file: one Mac per line, left to right as seen from behind; `-` is a gap, `#` starts a comment.
fn parse_row(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect()
}

/// The Macs in the row and their addresses, from `--nodes` or a row file that is re-read when it changes.
struct Row {
    file: Option<String>,
    modified: Option<std::time::SystemTime>,
    names: Vec<String>,
    targets: Vec<Option<SocketAddr>>,
    checked: Instant,
}

impl Row {
    fn new(nodes: Option<String>, file: String) -> Row {
        let mut row = Row {
            file: None,
            modified: None,
            names: Vec::new(),
            targets: Vec::new(),
            checked: Instant::now(),
        };
        match nodes {
            Some(n) => row.set(n.split(',').map(|s| s.trim().to_string()).collect()),
            None => {
                row.file = Some(file);
                if !row.reload() {
                    die(&format!(
                        "no Macs in {} (one per line, or use --nodes)",
                        row.file.as_deref().unwrap_or("")
                    ));
                }
            }
        }
        row
    }

    fn set(&mut self, names: Vec<String>) {
        self.targets = names
            .iter()
            .map(|n| {
                if n == "-" {
                    return None;
                }
                let a = resolve(n);
                if a.is_none() {
                    eprintln!("portlights: cannot resolve {n}; will retry");
                }
                a
            })
            .collect();
        println!("portlights: row {}", names.join(" "));
        self.names = names;
    }

    /// Re-reads the row file if it changed. False if it can't be read or names no Macs.
    fn reload(&mut self) -> bool {
        let Some(path) = self.file.clone() else {
            return true;
        };
        let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        if modified.is_some() && modified == self.modified {
            return true;
        }
        let names = match std::fs::read_to_string(&path) {
            Ok(text) => parse_row(&text),
            Err(e) => {
                eprintln!("portlights: {path}: {e}");
                return false;
            }
        };
        self.modified = modified;
        if names.is_empty() {
            eprintln!("portlights: {path} names no Macs; keeping the previous row");
            return false;
        }
        if names != self.names {
            self.set(names);
        }
        true
    }

    /// Every couple of seconds: picks up row file edits and retries names that didn't resolve.
    fn refresh(&mut self) {
        if self.checked.elapsed() < Duration::from_secs(2) {
            return;
        }
        self.checked = Instant::now();
        self.reload();
        for (t, n) in self.targets.iter_mut().zip(&self.names) {
            if t.is_none() && n != "-" {
                *t = resolve(n);
            }
        }
    }
}

/// The conductor: audio in, a frame per Mac out.
fn conduct(mut row: Row, style: viz::Style, audio_cmd: &str, rate: u32, fps: f32) {
    let sock = UdpSocket::bind("0.0.0.0:0").expect("UDP socket");
    // a frame that cannot go out now is dropped; the next one is 25 ms away
    sock.set_nonblocking(true).expect("nonblocking socket");
    let mut capture = audio::Capture::start(audio_cmd, rate).unwrap_or_else(|e| {
        eprintln!("portlights: cannot start audio ({audio_cmd}): {e}");
        std::process::exit(1);
    });
    let mut columns = row.names.len();
    let mut analyzer = audio::Analyzer::new(rate, columns * 3);
    let mut viz = viz::Viz::new(style, columns);
    let mut seq: u32 = 0;
    println!("portlights: conducting at {fps} fps");
    every(fps, |dt| {
        let Some(samples) = capture.take() else {
            println!("portlights: audio ended");
            return false;
        };
        row.refresh();
        if row.names.len() != columns {
            columns = row.names.len();
            analyzer = audio::Analyzer::new(rate, columns * 3);
            viz = viz::Viz::new(style, columns);
        }
        analyzer.push(&samples);
        let spectrum = analyzer.analyze(dt.max(1e-3));
        let frames = viz.render(&spectrum, dt);
        seq = seq.wrapping_add(1);
        for (target, frame) in row.targets.iter().zip(&frames) {
            if let Some(addr) = target {
                let _ = sock.send_to(&net::encode(seq, frame), addr);
            }
        }
        true
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_file() {
        let text = "# left to right, from behind\nincite\n  invite  # moved 10-19\n\n-\nalight\n";
        assert_eq!(parse_row(text), ["incite", "invite", "-", "alight"]);
    }
}
