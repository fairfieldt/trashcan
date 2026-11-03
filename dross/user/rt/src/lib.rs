//! The runtime every dross program links: system call wrappers (`sys`), the entry point, a heap, threads,
//! bootstrap messages, and the client side of pcid with the platform interface drivers use (`pci`).
#![no_std]

extern crate alloc;

pub mod bootstrap;
pub mod pci;
pub mod sys;
pub mod thread;

pub use dross_abi::{Error, FOREVER};
pub use sys::Handle;

use linked_list_allocator::LockedHeap;

#[global_allocator]
static HEAP: LockedHeap = LockedHeap::empty();
const HEAP_SIZE: u64 = 16 << 20;

unsafe extern "Rust" {
    fn __dross_main(bootstrap: Handle) -> i32;
}

#[unsafe(no_mangle)]
extern "C" fn _start(bootstrap: u64) -> ! {
    let m = sys::mem_create(HEAP_SIZE, 0).expect("heap");
    let at = sys::vm_map(m, 0, HEAP_SIZE, 0, dross_abi::vm::READ | dross_abi::vm::WRITE).expect("heap map");
    // SAFETY: a fresh private mapping, used for nothing else.
    unsafe { HEAP.lock().init(at as *mut u8, HEAP_SIZE as usize) };
    // SAFETY: defined by the program through `main!`.
    let code = unsafe { __dross_main(Handle(bootstrap as u32)) };
    sys::exit(code as i64)
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

#[panic_handler]
fn panic(p: &core::panic::PanicInfo) -> ! {
    sys::log(&alloc::format!("panic: {p}"));
    sys::exit(-1)
}
