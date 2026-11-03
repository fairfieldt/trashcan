//! Threads pinned to a CPU. Kernel-owned user stacks are reaped after exit,
//! including detached threads and threads whose status handles remain open.
use alloc::boxed::Box;
use crate::sys::{self, Handle};
const STACK: u64 = 256 << 10;

extern "C" fn trampoline(arg: u64) -> ! {
    // SAFETY: spawn transfers this box only after THREAD_START succeeds.
    let f = unsafe { Box::from_raw(arg as *mut Box<dyn FnOnce() + Send>) };
    f();
    sys::thread_exit()
}

pub struct JoinHandle { handle: Option<Handle> }
impl JoinHandle {
    /// Borrow the status capability; ownership remains with this wrapper.
    pub fn as_handle(&self) -> Handle { self.handle.expect("already joined") }
    /// Timeout leaves the thread and handle intact, allowing a later retry.
    pub fn wait_until(&mut self, deadline_ns: u64) -> sys::Result<()> {
        let Some(handle) = self.handle else { return Ok(()) };
        sys::thread_wait(handle, deadline_ns)?;
        self.handle = None;
        sys::close(handle)
    }
    pub fn join(mut self) -> sys::Result<()> { self.wait_until(dross_abi::FOREVER) }
    /// Transfer ownership of the raw status handle to the caller.
    pub fn detach(mut self) -> Handle { self.handle.take().unwrap() }
}
impl Drop for JoinHandle {
    fn drop(&mut self) { if let Some(handle) = self.handle.take() { let _ = sys::close(handle); } }
}

pub fn spawn_joinable(cpu: u64, f: impl FnOnce() + Send + 'static) -> sys::Result<JoinHandle> {
    let boxed: Box<Box<dyn FnOnce() + Send>> = Box::new(Box::new(f));
    let arg = Box::into_raw(boxed) as u64;
    match sys::thread_start(trampoline as *const () as u64, arg, cpu, STACK) {
        Ok(handle) => Ok(JoinHandle { handle: Some(handle) }),
        Err(error) => {
            // SAFETY: a failed start never published a thread or consumed its argument.
            drop(unsafe { Box::from_raw(arg as *mut Box<dyn FnOnce() + Send>) });
            Err(error)
        }
    }
}

/// Detached compatibility API. Close or wait/reap the returned status handle;
/// the kernel reclaims the user stack independently of that handle's lifetime.
pub fn spawn(cpu: u64, f: impl FnOnce() + Send + 'static) -> sys::Result<Handle> {
    spawn_joinable(cpu, f).map(JoinHandle::detach)
}
