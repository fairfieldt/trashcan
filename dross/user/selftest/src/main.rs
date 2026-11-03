//! Checks the kernel from user space: memory, channels, notifications, deadlines, threads on every CPU, extended
//! state across switches, preemption, rights, port I/O, and that faults end only the faulting process. Run by
//! init at boot; exits 0 when everything passed.
#![no_std]
#![no_main]

extern crate alloc;

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};
use dross_abi::{rights, vm, Error};
use dross_rt::bootstrap::{self, Bootstrap, Builder};
use dross_rt::{println, sys, thread, Handle, FOREVER};

dross_rt::main!(main);

fn main(boot: Handle) -> i32 {
    let b = Bootstrap::read(boot).expect("bootstrap");
    match b.value("mode").unwrap_or("main") {
        "exit42" => 42,
        "page-fault" => {
            // SAFETY: deliberately touches an unmapped address.
            unsafe { (0x10 as *mut u64).write_volatile(1) };
            0
        }
        "port-denied" => {
            // SAFETY: deliberately uses a port this process may not use.
            unsafe { core::arch::asm!("out dx, al", in("dx") 0x80u16, in("al") 0u8) };
            0
        }
        "read-only" => {
            let m = sys::mem_create(4096, 0).unwrap();
            let ro = sys::dup(m, rights::ALL & !rights::WRITE).unwrap();
            assert_eq!(sys::vm_map(ro, 0, 4096, 0, vm::READ | vm::WRITE), Err(Error::AccessDenied));
            let a = sys::vm_map(ro, 0, 4096, 0, vm::READ).unwrap();
            // SAFETY: deliberately writes a read-only mapping.
            unsafe { (a as *mut u64).write_volatile(1) };
            0
        }
        "serial" => {
            let r = b.handle("resource/com1").expect("com1");
            sys::ioport_enable(r, 0x3f8, 8).expect("ioport_enable");
            for &c in b"selftest: this line went straight to COM1 from user mode\r\n" {
                // SAFETY: COM1, which this process was given.
                unsafe {
                    while { let s: u8; core::arch::asm!("in al, dx", in("dx") 0x3fdu16, out("al") s); s } & 0x20 == 0 {}
                    core::arch::asm!("out dx, al", in("dx") 0x3f8u16, in("al") c);
                }
            }
            0
        }
        _ => run_all(&b),
    }
}

struct Tally {
    passed: u32,
    failed: u32,
}

impl Tally {
    fn check(&mut self, name: &str, ok: bool, detail: &str) {
        if ok {
            self.passed += 1;
            println!("ok   {name} {detail}");
        } else {
            self.failed += 1;
            println!("FAIL {name} {detail}");
        }
    }
}

fn run_all(b: &Bootstrap) -> i32 {
    let mut t = Tally { passed: 0, failed: 0 };
    let me = b.handle("module/selftest").expect("own image");
    let cpus = sys::cpu_count();

    // Memory: two mappings of one object see the same bytes.
    let m = sys::mem_create(64 << 10, 0).unwrap();
    let a = sys::vm_map(m, 0, 64 << 10, 0, vm::READ | vm::WRITE).unwrap();
    let c = sys::vm_map(m, 0, 64 << 10, 0, vm::READ).unwrap();
    // SAFETY: both mappings cover 64 KiB.
    let same = unsafe {
        for i in 0..8192u64 {
            (a as *mut u64).add(i as usize).write_volatile(i * 3);
        }
        (0..8192u64).all(|i| (c as *const u64).add(i as usize).read_volatile() == i * 3)
    };
    let unmapped = sys::vm_unmap(a, 64 << 10).is_ok() && sys::vm_unmap(a, 64 << 10) == Err(Error::NotFound);
    t.check("memory", same && unmapped, "");

    // A channel carries bytes and a handle; the handle works on the other side.
    let (x, y) = sys::channel_create().unwrap();
    let n = sys::notify_create().unwrap();
    let n2 = sys::dup(n, rights::ALL).unwrap();
    sys::channel_write(x, b"hello", &[n2]).unwrap();
    // The handle moved: it is gone here (the slot may be reused by the read below).
    let gone = sys::notify_signal(n2, 1) == Err(Error::BadHandle);
    let mut buf = [0u8; 16];
    let mut hs = [Handle(0); 4];
    let (nb, nh) = sys::channel_read(y, &mut buf, &mut hs, FOREVER).unwrap();
    sys::notify_signal(hs[0], 1 << 5).unwrap();
    let bits = sys::notify_wait(n, FOREVER).unwrap();
    t.check("channel", &buf[..nb] == b"hello" && nh == 1 && bits == 1 << 5 && gone, "");
    sys::close(x).unwrap();
    t.check("peer closed", sys::channel_read(y, &mut buf, &mut hs, FOREVER) == Err(Error::PeerClosed), "");

    // Rights: a handle without SIGNAL can't signal.
    let weak = sys::dup(n, rights::WAIT | rights::DUPLICATE).unwrap();
    t.check("rights", sys::notify_signal(weak, 1) == Err(Error::AccessDenied), "");

    // A deadline.
    let t0 = sys::clock();
    let r = sys::notify_wait(n, sys::after_ms(20));
    let waited = sys::clock() - t0;
    t.check("deadline", r == Err(Error::TimedOut) && waited >= 20_000_000, &alloc::format!("(20 ms wait took {} us)", waited / 1000));

    // A thread on every CPU signals its bit.
    let done = sys::notify_create().unwrap();
    let t0 = sys::clock();
    for cpu in 0..cpus {
        let d = sys::dup(done, rights::ALL).unwrap();
        thread::spawn(cpu as u64, move || {
            sys::notify_signal(d, 1 << cpu).unwrap();
        })
        .unwrap();
    }
    let want = if cpus == 64 { u64::MAX } else { (1u64 << cpus) - 1 };
    let mut got = 0;
    while got != want {
        match sys::notify_wait(done, sys::after_ms(5000)) {
            Ok(b) => got |= b,
            Err(_) => break,
        }
    }
    t.check("threads", got == want, &alloc::format!("({} cpus in {} us)", cpus, (sys::clock() - t0) / 1000));

    // Round trips to a thread on another CPU.
    if cpus > 1 {
        let (p, q) = sys::channel_create().unwrap();
        thread::spawn(1, move || {
            let mut b = [0u8; 8];
            let mut h = [Handle(0); 1];
            while let Ok((n, _)) = sys::channel_read(q, &mut b, &mut h, FOREVER) {
                if sys::channel_write(q, &b[..n], &[]).is_err() {
                    break;
                }
            }
            sys::thread_exit();
        })
        .unwrap();
        const N: u64 = 2000;
        let t0 = sys::clock();
        let mut ok = true;
        for i in 0..N {
            sys::channel_write(p, &i.to_le_bytes(), &[]).unwrap();
            let (n, _) = sys::channel_read(p, &mut buf, &mut hs, FOREVER).unwrap();
            ok &= n == 8 && buf[..8] == i.to_le_bytes();
        }
        let ns = (sys::clock() - t0) / N;
        t.check("round trips", ok, &alloc::format!("(cpu 0 <-> cpu 1: {}.{} us each)", ns / 1000, ns % 1000 / 100));
        sys::close(p).unwrap();
    }

    // Extended state survives switches between threads on one CPU.
    let cpu = (cpus - 1) as u64;
    let fpu_done = sys::notify_create().unwrap();
    for k in 1..=2u64 {
        let d = sys::dup(fpu_done, rights::ALL).unwrap();
        thread::spawn(cpu, move || {
            let pattern = 0x1111_1111_1111_1111u64 * k;
            let mut ok = true;
            for _ in 0..200 {
                let out: u64;
                // SAFETY: xmm0 is caller-saved and this code uses no other SSE.
                unsafe {
                    core::arch::asm!("movq xmm0, {}", in(reg) pattern);
                    sys::yield_now();
                    core::arch::asm!("movq {}, xmm0", out(reg) out);
                }
                ok &= out == pattern;
            }
            sys::notify_signal(d, if ok { 1 << k } else { 1 << (k + 8) }).unwrap();
        })
        .unwrap();
    }
    let mut got = 0;
    while got & 0b110 != 0b110 && got >> 8 == 0 {
        got |= sys::notify_wait(fpu_done, sys::after_ms(5000)).unwrap_or(1 << 15);
    }
    t.check("extended state", got == 0b110, "");

    // Preemption: two threads that never block both run on one CPU.
    let counts = Arc::new([AtomicU64::new(0), AtomicU64::new(0)]);
    let stop = sys::clock() + 300_000_000;
    let pre = sys::notify_create().unwrap();
    for k in 0..2 {
        let c = counts.clone();
        let d = sys::dup(pre, rights::ALL).unwrap();
        thread::spawn(cpu, move || {
            while sys::clock() < stop {
                c[k].fetch_add(1, Ordering::Relaxed);
            }
            sys::notify_signal(d, 1 << k).unwrap();
        })
        .unwrap();
    }
    let mut got = 0;
    while got != 0b11 {
        got |= sys::notify_wait(pre, sys::after_ms(5000)).unwrap_or(0b100);
        if got & 0b100 != 0 {
            break;
        }
    }
    let (c0, c1) = (counts[0].load(Ordering::Relaxed), counts[1].load(Ordering::Relaxed));
    t.check("preemption", got == 0b11 && c0 > 0 && c1 > 0, &alloc::format!("({c0} and {c1} loops)"));

    // Children: exit codes, and faults that end only the child.
    for (mode, want) in [("exit42", 42), ("page-fault", -1014), ("port-denied", -1013), ("read-only", -1014), ("serial", 0)] {
        let mut bb = Builder::default().value("mode", mode);
        if mode == "serial" {
            bb = bb.handle("resource/com1", sys::dup(b.handle("resource/com1").unwrap(), rights::ALL).unwrap());
        }
        let img = sys::dup(me, rights::ALL).unwrap();
        let code = bootstrap::spawn(img, mode, bb, u64::MAX).and_then(|p| sys::process_wait(p, sys::after_ms(10_000)));
        t.check(&alloc::format!("child {mode}"), code == Ok(want), &alloc::format!("(exit {code:?})"));
    }

    println!("{} passed, {} failed", t.passed, t.failed);
    if t.failed == 0 { 0 } else { 1 }
}
