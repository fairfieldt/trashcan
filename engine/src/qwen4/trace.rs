//! Host timeline of the chain (`TRASHCAN_Q4_TRACE=PREFIX`, given to every process): each process records
//! CLOCK_MONOTONIC timestamps at the steps of a forward (frames received and sent, stage submissions, fence waits,
//! the GPU's own start and end through calibrated timestamps, head scoring) and writes them to
//! `PREFIX-HOST.json` when its session (node) or command (driver) ends. The driver also measures every machine's
//! clock offset by ping-pong along the chain (`Chain::sync_clocks`) before and after the timed steps.
//! `results/qwen4/decode-host-20261013/tools/timeline.py` aligns the files and accounts for a decode step.
use std::sync::{Mutex, OnceLock};

type Events = Mutex<Vec<(&'static str, u64, u64)>>;

static TRACE: OnceLock<Option<(String, Events)>> = OnceLock::new();

fn trace() -> Option<&'static (String, Events)> {
    TRACE
        .get_or_init(|| {
            std::env::var("TRASHCAN_Q4_TRACE").ok().map(|p| (p, Mutex::new(Vec::with_capacity(1 << 18))))
        })
        .as_ref()
}

pub fn on() -> bool {
    trace().is_some()
}

pub fn now() -> u64 {
    crate::vulkan::host_ns()
}

/// Record `name` (with an argument such as a position or stage index) now.
pub fn mark(name: &'static str, arg: u64) {
    if let Some((_, ev)) = trace() {
        let t = now();
        ev.lock().unwrap().push((name, arg, t));
    }
}

/// Record `name` at host time `t`.
pub fn mark_at(name: &'static str, arg: u64, t: u64) {
    if let Some((_, ev)) = trace() {
        ev.lock().unwrap().push((name, arg, t));
    }
}

fn hostname() -> String {
    let mut buf = [0u8; 256];
    let ok = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) } == 0;
    let n = buf.iter().position(|&b| b == 0).unwrap_or(0);
    if ok && n > 0 {
        String::from_utf8_lossy(&buf[..n]).into_owned()
    } else {
        "local".into()
    }
}

/// Write the events so far (and `extra`, e.g. the driver's clock offsets) to `PREFIX-HOST.json`, then forget them.
pub fn write(role: &str, extra: serde_json::Value) {
    let Some((prefix, ev)) = trace() else { return };
    let events: Vec<_> = std::mem::take(&mut *ev.lock().unwrap());
    let host = hostname();
    let path = format!("{prefix}-{host}.json");
    let v = serde_json::json!({"host": host, "role": role, "extra": extra,
        "events": events.iter().map(|(n, a, t)| serde_json::json!([n, a, t])).collect::<Vec<_>>()});
    match std::fs::write(&path, v.to_string()) {
        Ok(()) => eprintln!("trace: {} events in {path}", events.len()),
        Err(e) => eprintln!("trace: cannot write {path}: {e}"),
    }
}
