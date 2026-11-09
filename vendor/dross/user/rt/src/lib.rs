//! The Rust runtime and native SDK: system call wrappers (`sys`), the entry point, a heap, threads,
//! bootstrap messages, and the client side of pcid with the platform interface drivers use (`pci`).
//!
//! Without the `runtime` feature (entry point, heap, panic handler) it is a plain library, for crates such as
//! gpud-platform's dross backend.
#![no_std]

extern crate alloc;
pub use dross_abi as abi;

pub mod bootstrap;
pub mod deadline;
pub mod events;
pub mod launch;
pub use dross_abi::io;
pub mod owned;
pub mod pci;
pub mod protocol;
pub mod sync;
pub mod sys;
pub mod thread;

pub use dross_abi::{Error, FOREVER};
pub use sys::Handle;

#[cfg(any(feature = "runtime", feature = "hosted"))]
pub mod heap;

#[cfg(feature = "runtime")]
mod runtime {
    use crate::{Handle, heap::ProgramHeap, sys};
    use core::fmt::{self, Write};

    #[global_allocator]
    static HEAP: ProgramHeap = ProgramHeap::new();

    unsafe extern "Rust" {
        fn __dross_main(bootstrap: Handle) -> i32;
    }

    #[unsafe(no_mangle)]
    extern "C" fn _start(bootstrap: u64) -> ! {
        // A program without room for its heap is out of its budget, which isn't a bug in the program.
        if !HEAP.add_first_arena() {
            sys::log("out of memory for program heap");
            sys::exit(-1);
        }
        // SAFETY: defined by the program through `main!`.
        let code = unsafe { __dross_main(Handle(bootstrap as u32)) };
        sys::exit(code as i64)
    }

    /// Bytes currently allocated from the program heap.
    pub fn heap_used() -> usize {
        HEAP.used()
    }

    /// Text in a fixed buffer, cut off when full.
    struct Message {
        bytes: [u8; 1024],
        length: usize,
    }
    impl Write for Message {
        fn write_str(&mut self, text: &str) -> fmt::Result {
            let room = self.bytes.len() - self.length;
            let fits = text.floor_char_boundary(room);
            self.bytes[self.length..self.length + fits].copy_from_slice(&text.as_bytes()[..fits]);
            self.length += fits;
            Ok(())
        }
    }

    #[panic_handler]
    fn panic(p: &core::panic::PanicInfo) -> ! {
        // Without allocating, which can be why this program is panicking.
        let mut message = Message { bytes: [0; 1024], length: 0 };
        let _ = write!(message, "panic: {p}");
        sys::log(core::str::from_utf8(&message.bytes[..message.length]).unwrap_or("panic"));
        sys::exit(-1)
    }
}

#[cfg(feature = "runtime")]
pub use runtime::heap_used;

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
