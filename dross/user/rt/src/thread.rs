//! Threads in the current process, each pinned to a CPU.

use alloc::boxed::Box;
use dross_abi::vm;

use crate::sys::{self, Handle};

const STACK: u64 = 256 << 10;

extern "C" fn trampoline(arg: u64) -> ! {
    // SAFETY: `spawn` leaked this box for the thread.
    let f = unsafe { Box::from_raw(arg as *mut Box<dyn FnOnce() + Send>) };
    f();
    sys::thread_exit()
}

/// Runs `f` on a new thread on `cpu` (`u64::MAX`: the kernel chooses). The stack is never freed.
pub fn spawn(cpu: u64, f: impl FnOnce() + Send + 'static) -> sys::Result<Handle> {
    let m = sys::mem_create(STACK, 0)?;
    let mapped = sys::vm_map(m, 0, STACK, 0, vm::READ | vm::WRITE);
    let _ = sys::close(m);
    let base = mapped?;
    let b: Box<Box<dyn FnOnce() + Send>> = Box::new(Box::new(f));
    let arg = Box::into_raw(b) as u64;
    // 16-byte aligned minus the return address slot, as at a call.
    match sys::thread_create(trampoline as *const () as u64, base + STACK - 8, arg, cpu) {
        Ok(h) => Ok(h),
        Err(e) => {
            // SAFETY: no thread was created, so ownership never left this function.
            drop(unsafe { Box::from_raw(arg as *mut Box<dyn FnOnce() + Send>) });
            let _ = sys::vm_unmap(base, STACK);
            Err(e)
        }
    }
}
