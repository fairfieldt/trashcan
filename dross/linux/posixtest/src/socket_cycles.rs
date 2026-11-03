//! Queued descriptor ownership must not turn unreachable sockets into roots.
use dross_rt::{Handle, sys};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::mpsc;
use std::time::Duration;

fn pair(ty: i32) -> (OwnedFd, OwnedFd) {
    let mut fds = [-1; 2];
    assert_eq!(
        unsafe { libc::socketpair(libc::AF_UNIX, ty | libc::SOCK_CLOEXEC, 0, fds.as_mut_ptr()) },
        0
    );
    unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) }
}
fn cycle() {
    let (a, b) = pair(libc::SOCK_SEQPACKET);
    let memory = sys::mem_create(64 * 1024, 0).unwrap();
    let payload = unsafe { OwnedFd::from_raw_fd(sys::handle_fd(memory).unwrap()) };
    sys::close(memory).unwrap();
    assert_eq!(super::send_fd(a.as_raw_fd(), b.as_raw_fd()), 1);
    assert_eq!(super::send_fd(a.as_raw_fd(), payload.as_raw_fd()), 1);
    drop(payload);
    drop(b);
    drop(a);
}
fn failed_receive_group() {
    let (a, b) = pair(libc::SOCK_SEQPACKET);
    let before = sys::account_info(Handle(0)).unwrap()[0];
    let memory = sys::mem_create(65536, 0).unwrap();
    let payload = unsafe { OwnedFd::from_raw_fd(sys::handle_fd(memory).unwrap()) };
    sys::close(memory).unwrap();
    unsafe {
        let mut byte = b'g';
        let mut iov = libc::iovec {
            iov_base: (&mut byte as *mut u8).cast(),
            iov_len: 1,
        };
        let mut control = [0u64; 4];
        let mut message: libc::msghdr = std::mem::zeroed();
        message.msg_iov = &mut iov;
        message.msg_iovlen = 1;
        message.msg_control = control.as_mut_ptr().cast();
        message.msg_controllen = libc::CMSG_SPACE(8) as _;
        let header = libc::CMSG_FIRSTHDR(&message);
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN(8) as _;
        let descriptors = libc::CMSG_DATA(header).cast::<i32>();
        descriptors.write(payload.as_raw_fd());
        descriptors.add(1).write(payload.as_raw_fd());
        assert_eq!(libc::sendmsg(a.as_raw_fd(), &message, 0), 1);
        drop(payload);
        assert!(sys::account_info(Handle(0)).unwrap()[0] >= before + 65536);
        // A non-null inaccessible control pointer fails only at copyout. It
        // must not leave either received descriptor installed without a number.
        message.msg_control = 1usize as *mut _;
        assert_eq!(libc::recvmsg(b.as_raw_fd(), &mut message, 0), -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EFAULT)
        );
    }
    assert!(sys::account_info(Handle(0)).unwrap()[0] <= before + 8192);
}

pub fn check() {
    // Failure to copy out either descriptor must publish neither endpoint and
    // must roll back the pair's account reservation.
    let baseline = sys::account_info(Handle(0)).unwrap()[0];
    for _ in 0..4 {
        assert_eq!(
            unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, std::ptr::null_mut()) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EFAULT)
        );
        assert_eq!(unsafe { libc::pipe(std::ptr::null_mut()) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EFAULT)
        );
    }
    assert_eq!(sys::account_info(Handle(0)).unwrap()[0], baseline);
    failed_receive_group();
    cycle(); // Warm descriptor tables/allocator before comparing backing charges.
    let before = sys::account_info(Handle(0)).unwrap()[0];
    for _ in 0..32 {
        cycle();
    }
    assert!(sys::account_info(Handle(0)).unwrap()[0] <= before + 8192);
    // A dup is an external root even when a queued right is the only other
    // reference to the same open description. Receiving it must still work.
    let (a, b) = pair(libc::SOCK_SEQPACKET);
    assert_eq!(super::send_fd(a.as_raw_fd(), b.as_raw_fd()), 1);
    let retained = unsafe { OwnedFd::from_raw_fd(libc::dup(b.as_raw_fd())) };
    assert!(retained.as_raw_fd() >= 0);
    drop(b);
    let received = super::recv_fd(retained.as_raw_fd()).unwrap();
    assert_eq!(
        unsafe { libc::send(a.as_raw_fd(), b"r".as_ptr().cast(), 1, libc::MSG_NOSIGNAL) },
        1
    );
    let mut byte = 0u8;
    assert_eq!(
        unsafe { libc::recv(received.as_raw_fd(), (&mut byte as *mut u8).cast(), 1, 0) },
        1
    );
    assert_eq!(byte, b'r');
    drop(received);
    drop(retained);
    drop(a);
    // Closing or shutting down a full receiver must wake a blocked sender.
    // The sender may finish entering its wait concurrently with the close.
    for shutdown in [false, true] {
        for _ in 0..4 {
            let (a, b) = pair(libc::SOCK_STREAM);
            let bytes = [0u8; 65536];
            loop {
                let n = unsafe {
                    libc::send(
                        a.as_raw_fd(),
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL,
                    )
                };
                if n < 0 {
                    assert_eq!(
                        std::io::Error::last_os_error().raw_os_error(),
                        Some(libc::EAGAIN)
                    );
                    break;
                }
                assert!(n > 0);
            }
            let (started, start) = mpsc::channel();
            let (finished, result) = mpsc::channel();
            let worker = std::thread::spawn(move || {
                started.send(()).unwrap();
                let n = unsafe {
                    libc::send(a.as_raw_fd(), b"x".as_ptr().cast(), 1, libc::MSG_NOSIGNAL)
                };
                let errno = std::io::Error::last_os_error().raw_os_error();
                finished.send((n, errno)).unwrap();
            });
            start.recv_timeout(Duration::from_secs(2)).unwrap();
            std::thread::sleep(Duration::from_millis(5));
            if shutdown {
                assert_eq!(unsafe { libc::shutdown(b.as_raw_fd(), libc::SHUT_RD) }, 0);
            }
            drop(b);
            let (n, errno) = result.recv_timeout(Duration::from_secs(2)).unwrap();
            assert_eq!(n, -1);
            assert_eq!(errno, Some(libc::EPIPE));
            worker.join().unwrap();
        }
    }
}
