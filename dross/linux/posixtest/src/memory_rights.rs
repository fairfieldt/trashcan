//! Native capability rights must survive Linux mappings and descriptor adapters.
use dross_abi::{rights, vm, Error};
use dross_rt::sys;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

pub fn check() {
    let memory = sys::mem_create(8192, 0).unwrap();
    let original = sys::vm_map(memory, 0, 8192, 0, vm::READ | vm::WRITE).unwrap();
    // SAFETY: our writable allocation, kept mapped until the end of the check.
    unsafe { (original as *mut u8).write_volatile(42); }
    let read_only = sys::dup(memory, rights::READ | rights::MAP | rights::DUPLICATE).unwrap();
    assert_eq!(sys::vm_map(read_only, 0, 8192, 0, vm::READ | vm::WRITE), Err(Error::AccessDenied));
    assert_eq!(sys::vm_map(read_only, 0, 8192, 0, vm::READ | vm::EXECUTE), Err(Error::AccessDenied));
    let native = sys::vm_map(read_only, 0, 8192, 0, vm::READ).unwrap();
    // SAFETY: libc receives mapped, page-aligned ranges and valid local buffers.
    unsafe {
        assert_eq!(libc::mprotect(native as *mut _, 8192, libc::PROT_READ | libc::PROT_WRITE), -1);
        assert_eq!(*libc::__errno_location(), libc::EACCES);
        // Splitting the mapping must preserve its ceiling on both surviving pieces.
        assert_eq!(libc::mprotect(native as *mut _, 4096, libc::PROT_NONE), 0);
        assert_eq!(libc::mprotect(native as *mut _, 4096, libc::PROT_READ | libc::PROT_EXEC), -1);
        assert_eq!(*libc::__errno_location(), libc::EACCES);
        assert_eq!(libc::mprotect(native as *mut _, 8192, libc::PROT_READ), 0);
    }
    let derived = sys::vm_memory(native).unwrap().0;
    assert_eq!(sys::vm_map(derived, 0, 4096, 0, vm::READ | vm::WRITE), Err(Error::AccessDenied));
    assert_eq!(sys::vm_map(derived, 0, 4096, 0, vm::READ | vm::EXECUTE), Err(Error::AccessDenied));
    let fd = unsafe { OwnedFd::from_raw_fd(sys::handle_fd(read_only).unwrap()) };
    // The descriptor retains its restricted rights even after its source handles close.
    sys::close(read_only).unwrap();
    sys::close(derived).unwrap();
    unsafe {
        let fd = fd.as_raw_fd();
        assert_eq!(libc::pwrite(fd, b"!".as_ptr().cast(), 1, 0), -1);
        assert_eq!(*libc::__errno_location(), libc::EBADF);
        assert_eq!(libc::ftruncate(fd, 4096), -1);
        assert_eq!(libc::mmap(std::ptr::null_mut(), 8192, libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED, fd, 0), libc::MAP_FAILED);
        assert_eq!(*libc::__errno_location(), libc::EACCES);
        let shared = libc::mmap(std::ptr::null_mut(), 8192, libc::PROT_READ, libc::MAP_SHARED, fd, 0);
        assert_ne!(shared, libc::MAP_FAILED);
        assert_eq!(shared.cast::<u8>().read_volatile(), 42);
        assert_eq!(libc::mprotect(shared, 8192, libc::PROT_READ | libc::PROT_WRITE), -1);
        assert_eq!(*libc::__errno_location(), libc::EACCES);
        assert_eq!(libc::mprotect(shared, 8192, libc::PROT_READ | libc::PROT_EXEC), -1);
        assert_eq!(*libc::__errno_location(), libc::EACCES);
        assert_eq!(libc::munmap(shared, 8192), 0);
        // A readable source may be copied into privately owned writable memory.
        let private = libc::mmap(std::ptr::null_mut(), 4096, libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE, fd, 0);
        assert_ne!(private, libc::MAP_FAILED);
        private.cast::<u8>().write_volatile(99);
        assert_eq!((original as *const u8).read_volatile(), 42);
        assert_eq!(libc::munmap(private, 4096), 0);
    }
    let no_read = sys::dup(memory, rights::MAP).unwrap();
    let unreadable = unsafe { OwnedFd::from_raw_fd(sys::handle_fd(no_read).unwrap()) };
    unsafe {
        let mut byte = 0u8;
        assert_eq!(libc::pread(unreadable.as_raw_fd(), (&mut byte as *mut u8).cast(), 1, 0), -1);
        assert_eq!(*libc::__errno_location(), libc::EBADF);
        assert_eq!(libc::mmap(std::ptr::null_mut(), 4096, libc::PROT_READ,
            libc::MAP_PRIVATE, unreadable.as_raw_fd(), 0), libc::MAP_FAILED);
        assert_eq!(*libc::__errno_location(), libc::EACCES);
    }
    drop(unreadable);
    drop(fd);
    sys::close(no_read).unwrap();
    // mprotect split this native mapping; Linux munmap accepts the full range.
    assert_eq!(unsafe { libc::munmap(native as *mut _, 8192) }, 0);
    sys::vm_unmap(original, 8192).unwrap();
    sys::close(memory).unwrap();
}

pub fn deadline_saturation() {
    let mut fds = [-1; 2];
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
    let reader = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let writer = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    let worker = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(unsafe { libc::write(writer.as_raw_fd(), b"x".as_ptr().cast(), 1) }, 1);
        writer
    });
    let mut event = libc::pollfd { fd: reader.as_raw_fd(), events: libc::POLLIN, revents: 0 };
    let timeout = libc::timespec { tv_sec: i64::MAX, tv_nsec: 0 };
    // An enormous relative duration must remain pending until actual readiness,
    // rather than wrapping to a past deadline and returning a timeout.
    let result = unsafe { libc::ppoll(&mut event, 1, &timeout, std::ptr::null()) };
    let writer = worker.join().unwrap();
    assert_eq!(result, 1);
    assert_ne!(event.revents & libc::POLLIN, 0);
    drop(writer);
}
