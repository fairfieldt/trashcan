//! Hold this process's amdgpu GPUs at the `high` DPM level while work arrives, and return them to `auto` after
//! `IDLE` without any. In a pipeline of many stages each GPU works a few milliseconds of every decode token; on `auto`
//! it falls to 300/150 MHz in between and every layer starts slow (Flash-Next on six Macs: 8.1 against 13.5 tok/s,
//! results/qwen4/chain-20261012). Held high, an idle D700 draws about 35 W more, hence the timeout.
//!
//! Only the GPUs the process uses: two engine processes on one Mac, one per GPU, each holding both levels, set each
//! other's GPU to `auto` when one started, went idle or ended, while the other was still working.
//!
//! The level file belongs to root, so writes go through `sudo -n tee` (about 20 ms; the clocks are up when it
//! returns). They run on a background thread: `busy` never blocks the caller.
use std::{
    path::PathBuf,
    process::{Command, Stdio},
    sync::{Arc, Condvar, Mutex},
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

/// The DPM level files of the amdgpu devices at the PCI addresses `pci` (as sysfs names them, `0000:06:00.0`).
fn level_files(pci: &[String]) -> Vec<PathBuf> {
    let Ok(cards) = std::fs::read_dir("/sys/class/drm") else { return Vec::new() };
    let mut files: Vec<PathBuf> = cards
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("card") && !n.contains('-')))
        .filter(|p| {
            std::fs::read_link(p.join("device/driver")).is_ok_and(|d| d.file_name().is_some_and(|n| n == "amdgpu"))
        })
        .filter(|p| {
            std::fs::canonicalize(p.join("device"))
                .is_ok_and(|d| d.file_name().and_then(|n| n.to_str()).is_some_and(|n| pci.iter().any(|a| a == n)))
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

impl Clocks {
    /// Hold the GPUs at the PCI addresses `pci` (`Instance::pci_address`). None when none of them is an amdgpu device
    /// or the level cannot be written (reported once).
    pub fn new(pci: &[String]) -> Option<Self> {
        let files = level_files(pci);
        if files.is_empty() {
            return None;
        }
        if !set(&files, "auto") {
            eprintln!("clocks: cannot write {} through sudo -n; GPUs stay on amdgpu's own levels", files[0].display());
            return None;
        }
        let shared = Arc::new((Mutex::new(State { last: Instant::now(), high: false, stop: false }), Condvar::new()));
        let s = shared.clone();
        let worker = thread::spawn(move || {
            let (lock, cv) = &*s;
            let mut st = lock.lock().unwrap();
            let mut level = false;
            loop {
                if st.stop {
                    break;
                }
                if st.high != level {
                    level = st.high;
                    drop(st);
                    set(&files, if level { "high" } else { "auto" });
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
                set(&files, "auto");
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
