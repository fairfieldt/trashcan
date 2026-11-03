//! Opt-in Unix personality extensions. Default hardware boots retain the original ABI.
pub mod channel;
pub mod entropy;
mod files;
mod multiplex;
pub mod mount;
pub mod net;
mod processes;
pub mod pty;
mod signals;
pub mod vfs;

use super::*;
use alloc::{
    string::String,
    sync::{Arc, Weak},
    vec::Vec,
};
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use spin::Mutex;

pub struct State {
    pub vfork: Mutex<Option<Arc<WaitCell<bool>>>>,
    pub cwd: Mutex<String>,
    pub mask: AtomicU32,
    pub credentials: Mutex<Credentials>,
    pub parent: Mutex<Weak<Process>>,
    pub children: WaitCell<Vec<Arc<Process>>>,
    pub pgid: AtomicU32,
    pub sid: AtomicU32,
    pub pending: AtomicU64,
    pub term_signal: AtomicU32,
    pub wait_event: AtomicU32,
    pub stopped: WaitCell<bool>,
    pub ctty: Mutex<Option<Arc<pty::Pty>>>,
    pub saved_signals: Mutex<Vec<signals::Saved>>,
}
#[derive(Clone)]
pub struct Credentials {
    pub uid: [u32; 3],
    pub gid: [u32; 3],
    pub groups: Vec<u32>,
}
impl State {
    pub fn new() -> Self {
        Self {
            vfork: Mutex::new(None),
            cwd: Mutex::new(String::from("/")),
            mask: AtomicU32::new(0o022),
            credentials: Mutex::new(Credentials {
                uid: [0; 3],
                gid: [0; 3],
                groups: Vec::new(),
            }),
            parent: Mutex::new(Weak::new()),
            children: WaitCell::new(Vec::new()),
            pgid: AtomicU32::new(0),
            sid: AtomicU32::new(0),
            pending: AtomicU64::new(0),
            term_signal: AtomicU32::new(0),
            wait_event: AtomicU32::new(0),
            stopped: WaitCell::new(false),
            ctty: Mutex::new(None),
            saved_signals: Mutex::new(Vec::new()),
        }
    }
}

pub fn state(p: &Process) -> &State {
    &linux(p).unix
}
pub fn dispatch(p: &Arc<Process>, frame: &mut Frame, n: u64, a: [u64; 6]) -> Option<R> {
    if state(p).vfork.lock().is_some() && matches!(n, 9..=12 | 25 | 56..=58) {
        return Some(Err(ENOSYS));
    }
    if let Some(result) = processes::dispatch(p, frame, n, a) {
        return Some(result);
    }
    if let Some(result) = signals::dispatch(p, frame, n, a) {
        return Some(result);
    }
    if let Some(result) = net::dispatch(p, n, a) {
        return Some(result);
    }
    if let Some(result) = multiplex::dispatch(p, n, a) {
        return Some(result);
    }
    files::dispatch(p, n, a)
}
pub fn exited(p: &Process) {
    if p.linux.is_none() {
        return;
    }
    if let Some(gate) = state(p).vfork.lock().take() {
        gate.update(|ready| *ready = true);
    }
    if let Some(parent) = state(p).parent.lock().upgrade() {
        parent.linux.as_ref().unwrap().unix.children.update(|_| ());
        signals::send(&parent, 17);
    }
}
pub fn user_return(frame: &mut Frame) {
    signals::user_return(frame);
}
pub fn interrupted(p: &Process) -> bool {
    state(p).pending.load(Ordering::Acquire) & !*linux(p).sigmask.lock() != 0
}
pub fn signal_group(pgid: u32, sig: u32) {
    processes::signal_group(pgid, sig);
}

pub fn pgid(p: &Process) -> u32 {
    let value = state(p).pgid.load(Ordering::Relaxed);
    if value == 0 { p.pid() } else { value }
}
pub fn sid(p: &Process) -> u32 {
    let value = state(p).sid.load(Ordering::Relaxed);
    if value == 0 { p.pid() } else { value }
}
