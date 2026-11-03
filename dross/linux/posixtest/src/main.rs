//! An ordinary Rust `std` program, built as a static Linux executable, that checks what gpud, tbd and ethd use of
//! Linux on dross: threads and their synchronization, sleeping and clocks, files, Unix sockets (stream and
//! seqpacket), fd passing of a shared memfd, pipes with poll, eventfd, unwinding, thread-locals, randomness.

use std::cell::Cell;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

mod memory_rights;
mod socket_cycles;
mod account_budget;

struct Tally(u32, u32);

impl Tally {
    fn check(&mut self, name: &str, ok: bool, detail: String) {
        println!("{} {name} {detail}", if ok { "ok  " } else { "FAIL" });
        if ok { self.0 += 1 } else { self.1 += 1 }
    }
}

fn memfd_with(text: &[u8]) -> OwnedFd {
    // SAFETY: plain libc calls on a fresh memfd.
    unsafe {
        let fd = libc::memfd_create(c"posixtest".as_ptr(), libc::MFD_CLOEXEC);
        assert!(fd >= 0);
        assert_eq!(libc::ftruncate(fd, 4096), 0);
        let p = libc::mmap(std::ptr::null_mut(), 4096, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0);
        assert_ne!(p, libc::MAP_FAILED);
        std::ptr::copy_nonoverlapping(text.as_ptr(), p.cast(), text.len());
        libc::munmap(p, 4096);
        OwnedFd::from_raw_fd(fd)
    }
}

fn send_fd(sock: i32, fd: i32) -> isize {
    // SAFETY: a msghdr with one byte and one SCM_RIGHTS message, on the stack.
    unsafe {
        let mut byte = [b'x'];
        let mut iov = libc::iovec { iov_base: byte.as_mut_ptr().cast(), iov_len: 1 };
        let mut buf = [0u64; 4];
        let mut m: libc::msghdr = std::mem::zeroed();
        m.msg_iov = &mut iov;
        m.msg_iovlen = 1;
        m.msg_control = buf.as_mut_ptr().cast();
        m.msg_controllen = libc::CMSG_SPACE(4) as _;
        let c = libc::CMSG_FIRSTHDR(&m);
        (*c).cmsg_level = libc::SOL_SOCKET;
        (*c).cmsg_type = libc::SCM_RIGHTS;
        (*c).cmsg_len = libc::CMSG_LEN(4) as _;
        *(libc::CMSG_DATA(c) as *mut i32) = fd;
        libc::sendmsg(sock, &m, 0)
    }
}

fn recv_fd(sock: i32) -> Option<OwnedFd> {
    // SAFETY: as in send_fd.
    unsafe {
        let mut byte = [0u8];
        let mut iov = libc::iovec { iov_base: byte.as_mut_ptr().cast(), iov_len: 1 };
        let mut buf = [0u64; 4];
        let mut m: libc::msghdr = std::mem::zeroed();
        m.msg_iov = &mut iov;
        m.msg_iovlen = 1;
        m.msg_control = buf.as_mut_ptr().cast();
        m.msg_controllen = std::mem::size_of_val(&buf) as _;
        if libc::recvmsg(sock, &mut m, 0) != 1 {
            return None;
        }
        let c = libc::CMSG_FIRSTHDR(&m);
        if c.is_null() || (*c).cmsg_type != libc::SCM_RIGHTS {
            return None;
        }
        Some(OwnedFd::from_raw_fd(*(libc::CMSG_DATA(c) as *const i32)))
    }
}

fn main() {
    if account_budget::run(&std::env::args().collect::<Vec<_>>()) { return; }
    let mut t = Tally(0, 0);
    memory_rights::check();
    t.check("native memory rights through Linux adapters", true, "mapping ceilings, descriptor rights, private copies".into());
    memory_rights::deadline_saturation();
    t.check("large relative deadline saturates", true, "ppoll wakes for data instead of overflowing".into());
    let args: Vec<String> = std::env::args().collect();
    t.check("args and environment", args.get(1).map(String::as_str) == Some("--check") && std::env::var("DROSS_BOOTSTRAP").as_deref() == Ok("1"), format!("{args:?}"));

    // A partial Linux unmap must release unused backing, while surviving pages retain data.
    let charged = || dross_rt::sys::account_info(dross_rt::Handle(0)).unwrap()[0];
    let baseline = charged();
    // SAFETY: checked anonymous mapping, then disjoint page-aligned unmaps.
    let reclaimed = unsafe {
        let size = 16 * 4096;
        let p = libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0);
        assert_ne!(p, libc::MAP_FAILED);
        let first = p.cast::<u8>();
        let last = first.add(size - 4096);
        first.write_volatile(17); last.write_volatile(93);
        let allocated = charged() >= baseline + size as u64 && charged() <= baseline + size as u64 + 3 * 4096;
        assert_eq!(libc::munmap(first.add(4096).cast(), size - 8192), 0);
        let partial = charged() >= baseline + 8192 && charged() <= baseline + 8192 + 3 * 4096
            && first.read_volatile() == 17 && last.read_volatile() == 93;
        assert_eq!(libc::munmap(first.cast(), 4096), 0);
        assert_eq!(libc::munmap(last.cast(), 4096), 0);
        allocated && partial && charged() == baseline
    };
    t.check("partial unmap releases backing", reclaimed, String::new());

    // Threads.
    let sum: u64 = (0..8u64).map(|i| std::thread::spawn(move || (0..1000).map(|j| i * j).sum::<u64>())).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).sum();
    t.check("threads", sum == 13_986_000, String::new());
    let counter = Arc::new(Mutex::new(0u64));
    let hs: Vec<_> = (0..8).map(|_| {
        let c = counter.clone();
        std::thread::spawn(move || (0..20_000).for_each(|_| *c.lock().unwrap() += 1))
    }).collect();
    hs.into_iter().for_each(|h| h.join().unwrap());
    t.check("mutex", *counter.lock().unwrap() == 160_000, String::new());
    let pair = Arc::new((Mutex::new(0u32), Condvar::new()));
    let p2 = pair.clone();
    let h = std::thread::spawn(move || {
        for i in 0..200 {
            let (m, c) = &*p2;
            let mut g = c.wait_while(m.lock().unwrap(), |v| *v % 2 == 0).unwrap();
            *g = i * 2 + 2;
            c.notify_all();
        }
    });
    for i in 0..200 {
        let (m, c) = &*pair;
        let mut g = c.wait_while(m.lock().unwrap(), |v| *v != i * 2).unwrap();
        *g += 1;
        c.notify_all();
    }
    h.join().unwrap();
    t.check("condvar", *pair.0.lock().unwrap() == 400, String::new());
    let (tx, rx) = mpsc::channel();
    let h = std::thread::spawn(move || (0..1000u32).for_each(|i| tx.send(i).unwrap()));
    let got: u32 = rx.iter().sum();
    h.join().unwrap();
    t.check("channel", got == 499_500, String::new());
    thread_local!(static TLS: Cell<u32> = const { Cell::new(0) });
    TLS.set(1);
    let other = std::thread::spawn(|| { TLS.set(2); TLS.get() }).join().unwrap();
    t.check("thread-locals", TLS.get() == 1 && other == 2, String::new());

    // Time.
    let t0 = Instant::now();
    std::thread::sleep(Duration::from_millis(20));
    let el = t0.elapsed();
    t.check("sleep", el >= Duration::from_millis(20), format!("(20 ms took {el:?})"));

    // Files.
    let text = std::fs::read_to_string("/etc/dross-test.txt");
    t.check("read file", text.as_deref().ok() == Some("dross rootfs test file\n"), format!("{text:?}"));
    let names: Vec<String> = std::fs::read_dir("/etc").map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()).unwrap_or_default();
    t.check("read_dir", names.iter().any(|n| n == "dross-test.txt"), format!("{names:?}"));
    t.check("missing file", std::fs::File::open("/etc/nothing").err().map(|e| e.kind()) == Some(std::io::ErrorKind::NotFound), String::new());
    let len = std::fs::metadata("/etc/dross-test.txt").map(|m| m.len());
    t.check("metadata", len.as_ref().ok() == Some(&23), format!("{len:?}"));
    let vfct = std::fs::metadata("/sys/firmware/acpi/tables/APIC").map(|m| m.len()).unwrap_or(0);
    t.check("firmware tables", vfct > 44, format!("(APIC {vfct} bytes)"));

    // The kernel log, as /dev/kmsg records from the oldest kept.
    let first = std::fs::File::open("/dev/kmsg").and_then(|mut f| {
        let mut b = vec![0u8; 8192];
        let n = f.read(&mut b)?;
        Ok(String::from_utf8_lossy(&b[..n]).into_owned())
    });
    t.check("kernel log", first.as_deref().is_ok_and(|r| r.starts_with("6,0,") && r.contains(";dross")), format!("{first:?}"));

    // Unix sockets.
    let path = "/run/posixtest.sock";
    let listener = UnixListener::bind(path).unwrap();
    let h = std::thread::spawn(move || {
        let mut s = UnixStream::connect(path).unwrap();
        s.write_all(b"ping").unwrap();
        let mut b = [0u8; 4];
        s.read_exact(&mut b).unwrap();
        b == *b"pong"
    });
    let (mut c, _) = listener.accept().unwrap();
    let mut b = [0u8; 4];
    c.read_exact(&mut b).unwrap();
    c.write_all(b"pong").unwrap();
    t.check("unix stream", b == *b"ping" && h.join().unwrap(), String::new());

    // A memfd passed over a seqpacket pair, then mapped by the receiver.
    let mut sv = [0i32; 2];
    // SAFETY: socketpair into sv.
    let r = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0, sv.as_mut_ptr()) };
    for (lengths, label) in [
        ([usize::MAX, 1], "recvmsg iovec overflow rejected"),
        ([i64::MAX as usize, 1], "recvmsg iovec above ssize_t rejected"),
    ] {
        let mut bad_iov = lengths.map(|iov_len| libc::iovec { iov_base: std::ptr::null_mut(), iov_len });
        // SAFETY: msg and its two iovecs are valid; invalid buffer lengths must be rejected before copying.
        let rejected = unsafe {
            let mut msg: libc::msghdr = std::mem::zeroed();
            msg.msg_iov = bad_iov.as_mut_ptr();
            msg.msg_iovlen = bad_iov.len() as _;
            r == 0 && libc::recvmsg(sv[1], &mut msg, libc::MSG_DONTWAIT) == -1 && *libc::__errno_location() == libc::EINVAL
        };
        t.check(label, rejected, String::new());
    }
    // A later control header can overflow its offset even when the first is valid.
    let malformed = unsafe {
        let mut control = [0u64; 4];
        control[0] = 16;
        control[2] = u64::MAX;
        let mut byte = 0u8;
        let mut iov = libc::iovec { iov_base: (&mut byte as *mut u8).cast(), iov_len: 1 };
        let mut msg: libc::msghdr = std::mem::zeroed();
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        msg.msg_controllen = std::mem::size_of_val(&control) as _;
        libc::sendmsg(sv[0], &msg, 0) == -1
    };
    t.check("overflowing control message", malformed, String::new());
    let mfd = memfd_with(b"shared memory");
    let large_iov = [libc::iovec { iov_base: std::ptr::null_mut(), iov_len: 64 << 20 }, libc::iovec { iov_base: std::ptr::null_mut(), iov_len: 1 }];
    let oversized_iov = unsafe { libc::writev(mfd.as_raw_fd(), large_iov.as_ptr(), 2) == -1 && *libc::__errno_location() == libc::EINVAL };
    t.check("oversized iovec rejected", oversized_iov, String::new());
    // Invalid or unrepresentable sizes must fail without damaging the live file.
    let bad_sizes = unsafe { libc::ftruncate(mfd.as_raw_fd(), -1) == -1 && libc::ftruncate(mfd.as_raw_fd(), i64::MAX) == -1 };
    t.check("memfd oversized truncate", bad_sizes, String::new());
    let sent = send_fd(sv[0], mfd.as_raw_fd());
    let got = recv_fd(sv[1]);
    let text = got.map(|fd| {
        // SAFETY: mapping the received memfd.
        unsafe {
            let p = libc::mmap(std::ptr::null_mut(), 4096, libc::PROT_READ, libc::MAP_SHARED, fd.as_raw_fd(), 0);
            let s = std::slice::from_raw_parts(p as *const u8, 13).to_vec();
            libc::munmap(p, 4096);
            s
        }
    });
    t.check("fd passing", r == 0 && sent == 1 && text.as_deref() == Some(b"shared memory".as_slice()), String::new());

    // A pipe with poll.
    let mut p = [0i32; 2];
    // SAFETY: plain libc calls on the pipe.
    let (before, after, n) = unsafe {
        libc::pipe2(p.as_mut_ptr(), libc::O_CLOEXEC);
        let mut pf = libc::pollfd { fd: p[0], events: libc::POLLIN, revents: 0 };
        let before = libc::poll(&mut pf, 1, 0);
        let w = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            libc::write(p[1], b"x".as_ptr().cast(), 1)
        });
        let after = libc::poll(&mut pf, 1, 5000);
        let mut b = [0u8; 1];
        let n = libc::read(p[0], b.as_mut_ptr().cast(), 1);
        w.join().unwrap();
        (before, after, n + (pf.revents as isize & 1) * 10)
    };
    t.check("pipe and poll", before == 0 && after == 1 && n == 11, format!("({before} {after} {n})"));

    // eventfd
    // SAFETY: plain libc calls on the eventfd.
    let v = unsafe {
        let e = libc::eventfd(0, 0);
        let one = 3u64.to_ne_bytes();
        libc::write(e, one.as_ptr().cast(), 8);
        let mut b = [0u8; 8];
        libc::read(e, b.as_mut_ptr().cast(), 8);
        u64::from_ne_bytes(b)
    };
    t.check("eventfd", v == 3, String::new());

    socket_cycles::check();
    t.check("SCM_RIGHTS lifetimes and peer-close races", true, String::new());

    // Unwinding.
    let caught = std::panic::catch_unwind(|| panic!("expected panic")).is_err();
    t.check("unwinding", caught, String::new());

    let mut a = [0u8; 16];
    let mut b = [0u8; 16];
    // SAFETY: getrandom into the buffers.
    unsafe {
        libc::getrandom(a.as_mut_ptr().cast(), 16, 0);
        libc::getrandom(b.as_mut_ptr().cast(), 16, 0);
    }
    t.check("getrandom", a != b && a != [0; 16], String::new());

    println!("{} passed, {} failed", t.0, t.1);
    std::process::exit(if t.1 == 0 { 0 } else { 1 });
}
