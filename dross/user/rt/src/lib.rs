//! The runtime every dross program links: system call wrappers (`sys`), the entry point, a heap, threads,
//! bootstrap messages, and the client side of pcid with the platform interface drivers use (`pci`).
//!
//! Without the `runtime` feature (entry point, heap, panic handler) it is a plain library, for crates such as
//! gpud-platform's dross backend.
#![no_std]

extern crate alloc;

pub mod bootstrap;
pub mod deadline;
pub mod launch;
pub mod owned;
pub mod pci;
pub mod protocol;
pub mod sys;
pub mod thread;

pub use dross_abi::{Error, FOREVER};
pub use sys::Handle;

#[cfg(feature = "runtime")]
mod runtime {
    use crate::{sys, Handle};
    use linked_list_allocator::LockedHeap;

    #[global_allocator]
    static HEAP: LockedHeap = LockedHeap::empty();
    const HEAP_SIZE: u64 = 16 << 20;

    unsafe extern "Rust" {
        fn __dross_main(bootstrap: Handle) -> i32;
    }

    #[unsafe(no_mangle)]
    extern "C" fn _start(bootstrap: u64) -> ! {
        let m = match sys::mem_create(HEAP_SIZE, 0) {
            Ok(memory) => memory,
            Err(_) => { sys::log("out of memory for program heap"); sys::exit(-1); }
        };
        let at = match sys::vm_map(m, 0, HEAP_SIZE, 0, dross_abi::vm::READ | dross_abi::vm::WRITE) {
            Ok(address) => address,
            Err(_) => { let _ = sys::close(m); sys::log("out of memory for program heap mapping"); sys::exit(-1); }
        };
        let _ = sys::close(m);
        // SAFETY: a fresh private mapping, used for nothing else.
        unsafe { HEAP.lock().init(at as *mut u8, HEAP_SIZE as usize) };
        // SAFETY: defined by the program through `main!`.
        let code = unsafe { __dross_main(Handle(bootstrap as u32)) };
        sys::exit(code as i64)
    }

    #[panic_handler]
    fn panic(p: &core::panic::PanicInfo) -> ! {
        sys::log(&alloc::format!("panic: {p}"));
        sys::exit(-1)
    }
}

/// Names the program's main function: `fn main(bootstrap: Handle) -> i32`.
#[macro_export]
macro_rules! main {
    ($f:path) => {
        #[unsafe(no_mangle)]
        fn __dross_main(h: $crate::Handle) -> i32 {
            $f(h)
        }
    };
}

#[macro_export]
macro_rules! println {
    ($($t:tt)*) => { $crate::sys::log(&alloc::format!($($t)*)) };
}

/// A lock for short critical sections between threads of one process: spins, yielding the CPU.
pub struct Lock<T> {
    held: core::sync::atomic::AtomicBool,
    value: core::cell::UnsafeCell<T>,
}

// SAFETY: access to `value` is serialized by `held`.
unsafe impl<T: Send> Sync for Lock<T> {}
unsafe impl<T: Send> Send for Lock<T> {}

impl<T> Lock<T> {
    pub const fn new(v: T) -> Self {
        Lock { held: core::sync::atomic::AtomicBool::new(false), value: core::cell::UnsafeCell::new(v) }
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        use core::sync::atomic::Ordering;
        while self.held.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            sys::yield_now();
        }
        // SAFETY: held exclusively until the store below.
        let r = f(unsafe { &mut *self.value.get() });
        self.held.store(false, Ordering::Release);
        r
    }
}
