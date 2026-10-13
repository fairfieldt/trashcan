//! Hold this machine's amdgpu GPUs at the `high` DPM level, and its CPUs at the `performance` cpufreq governor, while
//! work arrives, and put both back after `IDLE` without any. In a pipeline of many stages each GPU works a few
//! milliseconds of every decode token; on `auto` it falls to 300/150 MHz in between and every layer starts slow
//! (Flash-Next on six Macs: 8.1 against 13.5 tok/s, results/qwen4/chain-20261012). Held high, an idle D700 draws about
//! 35 W more, hence the timeout. Likewise a machine's host work for a decode token (receiving the residual, copies,
//! submissions, sending it on) comes in short bursts, between which schedutil lets the package fall to its lowest
//! frequency (results/qwen4/decode-host-20261013).
//!
//! The level files belong to root, so writes go through `sudo -n tee` (about 20 ms; the clocks are up when it
//! returns). They run on a background thread: `busy` never blocks the caller. A root helper started with the first
//! clocks of a process puts the levels back however the process ends (a signal, a crash): it polls for the process
//! twice a second.
use std::{
    os::unix::process::CommandExt,
    path::PathBuf,
    process::{Command, Stdio},
    sync::{Arc, Condvar, Mutex, Once},
    thread,
    time::{Duration, Instant},
};

const IDLE: Duration = Duration::from_secs(1);

struct State {
    last: Instant,
    high: bool,
    stop: bool,
}

pub struct Clocks {
    shared: Arc<(Mutex<State>, Condvar)>,
    worker: Option<thread::JoinHandle<()>>,
}

/// The DPM level files of every amdgpu device.
fn level_files() -> Vec<PathBuf> {
    let Ok(cards) = std::fs::read_dir("/sys/class/drm") else { return Vec::new() };
    let mut files: Vec<PathBuf> = cards
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("card") && !n.contains('-')))
        .filter(|p| {
            std::fs::read_link(p.join("device/driver")).is_ok_and(|d| d.file_name().is_some_and(|n| n == "amdgpu"))
        })
        .map(|p| p.join("device/power_dpm_force_performance_level"))
        .filter(|f| f.exists())
        .collect();
    files.sort();
    files
}

fn set(files: &[PathBuf], level: &str) -> bool {
    let child = Command::new("sudo")
        .args(["-n", "tee"])
        .args(files)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    let Ok(mut child) = child else { return false };
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        let _ = writeln!(stdin, "{level}");
    }
    child.wait().is_ok_and(|s| s.success())
}

/// The governor file of every cpufreq policy and the governor it had, when `performance` is available.
struct Cpus {
    files: Vec<PathBuf>,
    saved: Vec<String>,
}

impl Cpus {
    fn find() -> Option<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir("/sys/devices/system/cpu/cpufreq")
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("policy")))
            .map(|p| p.join("scaling_governor"))
            .filter(|f| f.exists())
            .collect();
        files.sort();
        let available = std::fs::read_to_string(files.first()?.with_file_name("scaling_available_governors")).ok()?;
        if !available.split_whitespace().any(|g| g == "performance") {
            return None;
        }
        let saved = files
            .iter()
            .map(|f| std::fs::read_to_string(f).map(|s| s.trim().to_string()))
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        Some(Self { files, saved })
    }

    fn hold(&self) -> bool {
        set(&self.files, "performance")
    }

    /// Each policy's governor as found, one write per distinct governor.
    fn restore(&self) -> bool {
        let mut values: Vec<&String> = self.saved.iter().collect();
        values.sort();
        values.dedup();
        values.into_iter().fold(true, |ok, v| {
            let files: Vec<PathBuf> =
                self.files.iter().zip(&self.saved).filter(|(_, s)| *s == v).map(|(f, _)| f.clone()).collect();
            set(&files, v) && ok
        })
    }
}

/// Start the root helper that puts the GPUs back on `auto` and the CPUs on their governors once this process has
/// ended. It runs in a session of its own: a signal to this process's group leaves it running, and without a
/// terminal sudo needs no pseudo-terminal.
fn guardian(gpus: &[PathBuf], cpus: Option<&Cpus>) {
    let mut script = format!("while kill -0 {} 2>/dev/null; do sleep 0.5; done\n", std::process::id());
    for f in gpus {
        script.push_str(&format!("echo auto > '{}'\n", f.display()));
    }
    for (f, v) in cpus.iter().flat_map(|c| c.files.iter().zip(&c.saved)) {
        script.push_str(&format!("echo {v} > '{}'\n", f.display()));
    }
    let mut command = Command::new("sudo");
    command.args(["-n", "sh", "-c", &script]).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    // setsid is async-signal-safe.
    unsafe {
        command.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let child = command.spawn();
    if child.is_err() {
        eprintln!("clocks: cannot start the helper that restores the clock levels when this process ends");
    }
}

impl Clocks {
    /// None when this machine has no amdgpu devices or the level cannot be written (reported once). The CPUs are held
    /// too where the governors can be read and `performance` is available; a failed write is reported once and the
    /// CPUs are left alone from then on.
    pub fn new() -> Option<Self> {
        let files = level_files();
        if files.is_empty() {
            return None;
        }
        if !set(&files, "auto") {
            eprintln!("clocks: cannot write {} through sudo -n; GPUs stay on amdgpu's own levels", files[0].display());
            return None;
        }
        let mut cpus = Cpus::find();
        // The governors found by the first clocks of the process are the ones to restore: later clocks find what
        // the earlier ones restored.
        static GUARDIAN: Once = Once::new();
        GUARDIAN.call_once(|| guardian(&files, cpus.as_ref()));
        let shared = Arc::new((Mutex::new(State { last: Instant::now(), high: false, stop: false }), Condvar::new()));
        let s = shared.clone();
        let worker = thread::spawn(move || {
            let (lock, cv) = &*s;
            let mut st = lock.lock().unwrap();
            let mut level = false;
            let put = |level: bool, cpus: &mut Option<Cpus>| {
                set(&files, if level { "high" } else { "auto" });
                if let Some(c) = cpus.as_ref() {
                    let ok = if level { c.hold() } else { c.restore() };
                    if !ok {
                        eprintln!("clocks: cannot write {} through sudo -n; CPUs stay on their governors", c.files[0].display());
                        c.restore();
                        *cpus = None;
                    }
                }
            };
            loop {
                if st.stop {
                    break;
                }
                if st.high != level {
                    level = st.high;
                    drop(st);
                    put(level, &mut cpus);
                    st = lock.lock().unwrap();
                    continue;
                }
                if level {
                    let idle = st.last.elapsed();
                    if idle >= IDLE {
                        st.high = false;
                        continue;
                    }
                    st = cv.wait_timeout(st, IDLE - idle).unwrap().0;
                } else {
                    st = cv.wait(st).unwrap();
                }
            }
            if level {
                put(false, &mut cpus);
            }
        });
        Some(Self { shared, worker: Some(worker) })
    }

    /// Work is arriving: raise the clocks if they are not already held, and restart the idle timeout.
    pub fn busy(&self) {
        let (lock, cv) = &*self.shared;
        let mut st = lock.lock().unwrap();
        st.last = Instant::now();
        if !st.high {
            st.high = true;
            cv.notify_one();
        }
    }
}

impl Drop for Clocks {
    fn drop(&mut self) {
        let (lock, cv) = &*self.shared;
        lock.lock().unwrap().stop = true;
        cv.notify_one();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}
