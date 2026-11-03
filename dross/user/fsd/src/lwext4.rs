//! Narrow FFI surface; lwext4 descriptors stay in C. The server calls it serially.
use core::ffi::{c_char, c_int, c_void};
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct Info {
    pub inode: u64,
    pub size: u64,
    pub atime: u64,
    pub mtime: u64,
    pub ctime: u64,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub links: u32,
}
#[repr(C)]
#[derive(Default)]
pub struct Stat {
    pub size: u64,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub mtime: u32,
}
#[repr(C)]
pub struct DirEntry {
    pub inode: u32,
    pub length: u16,
    pub name_length: u8,
    pub kind: u8,
    pub name: [u8; 255],
}
unsafe extern "C" {
    pub fn oscar_info(path: *const c_char, info: *mut Info) -> c_int;
    pub fn oscar_link(path: *const c_char, next: *const c_char) -> c_int;
    pub fn oscar_open(path: *const c_char, flags: c_int, fd: *mut c_int) -> c_int;
    pub fn oscar_close(fd: c_int) -> c_int;
    pub fn oscar_fd_info(fd: c_int, info: *mut Info) -> c_int;
    pub fn oscar_fd_read(
        fd: c_int,
        off: u64,
        buf: *mut c_void,
        len: usize,
        done: *mut usize,
    ) -> c_int;
    pub fn oscar_fd_write(
        fd: c_int,
        off: u64,
        buf: *const c_void,
        len: usize,
        done: *mut usize,
        end: *mut u64,
    ) -> c_int;
    pub fn oscar_fd_truncate(fd: c_int, size: u64) -> c_int;
    pub fn oscar_fd_setattr(fd: c_int, mode: u32, uid: u32, gid: u32) -> c_int;
    pub fn oscar_fd_times(fd: c_int, flags: u32, atime: u64, mtime: u64) -> c_int;

    pub fn oscar_setattr(path: *const c_char, mode: u32, uid: u32, gid: u32) -> c_int;
    pub fn oscar_error() -> c_int;
    pub fn oscar_mount(size: u32, blocks: u64, readonly: c_int) -> c_int;
    pub fn oscar_sync() -> c_int;
    pub fn oscar_unmount() -> c_int;
    pub fn oscar_stat(path: *const c_char, stat: *mut Stat) -> c_int;
    pub fn oscar_read(
        path: *const c_char,
        off: u64,
        buf: *mut c_void,
        len: usize,
        done: *mut usize,
    ) -> c_int;
    pub fn oscar_write(
        path: *const c_char,
        off: u64,
        buf: *const c_void,
        len: usize,
        done: *mut usize,
    ) -> c_int;
    pub fn oscar_create(path: *const c_char, mode: u32, uid: u32, gid: u32) -> c_int;
    pub fn oscar_truncate(path: *const c_char, size: u64) -> c_int;
    pub fn oscar_remove(path: *const c_char, directory: c_int) -> c_int;
    pub fn oscar_rename(path: *const c_char, next: *const c_char, directory: c_int) -> c_int;
    pub fn oscar_symlink(path: *const c_char, target: *const c_char) -> c_int;
    pub fn oscar_readlink(
        path: *const c_char,
        buf: *mut c_char,
        len: usize,
        done: *mut usize,
    ) -> c_int;
    pub fn oscar_dir_open(path: *const c_char) -> c_int;
    pub fn oscar_dir_next() -> *const DirEntry;
    pub fn oscar_dir_close() -> c_int;
}
// C allocations use the process heap and report exhaustion as null. The stored
// layout makes all success/error paths compatible with the same deallocator.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    let Some(total) = size.checked_add(16) else {
        return core::ptr::null_mut();
    };
    let Ok(layout) = alloc::alloc::Layout::from_size_align(total, 16) else {
        return core::ptr::null_mut();
    };
    let p = unsafe { alloc::alloc::alloc(layout) };
    if p.is_null() {
        return p.cast();
    }
    unsafe {
        p.cast::<usize>().write(total);
        p.add(16).cast()
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    let base = unsafe { p.cast::<u8>().sub(16) };
    let total = unsafe { base.cast::<usize>().read() };
    unsafe {
        alloc::alloc::dealloc(
            base,
            alloc::alloc::Layout::from_size_align_unchecked(total, 16),
        )
    };
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calloc(n: usize, size: usize) -> *mut c_void {
    let Some(bytes) = n.checked_mul(size) else {
        return core::ptr::null_mut();
    };
    let p = unsafe { malloc(bytes) };
    if !p.is_null() {
        unsafe { core::ptr::write_bytes(p.cast::<u8>(), 0, bytes) }
    }
    p
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn realloc(p: *mut c_void, size: usize) -> *mut c_void {
    if p.is_null() {
        return unsafe { malloc(size) };
    }
    if size == 0 {
        unsafe { free(p) };
        return core::ptr::null_mut();
    }
    let next = unsafe { malloc(size) };
    if next.is_null() {
        return next;
    }
    let old = unsafe { p.cast::<u8>().sub(16).cast::<usize>().read() } - 16;
    unsafe {
        core::ptr::copy_nonoverlapping(p.cast::<u8>(), next.cast::<u8>(), old.min(size));
        free(p)
    };
    next
}
