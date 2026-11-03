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

/// One-way latencies (ns, sorted) of 1000 messages from cpu 0 to a thread on cpu 1 that waits for each blocked
/// in the kernel or spinning on non-blocking reads, of its replies (cpu 0 always blocks), and the round trips;
/// `pause`: 50 us between messages, so both sides have settled. The TSCs agree to well under a microsecond.
fn one_way(blocking: bool, pause: bool) -> (alloc::vec::Vec<u64>, alloc::vec::Vec<u64>, alloc::vec::Vec<u64>) {
    let (p, q) = sys::channel_create().unwrap();
    let out = Arc::new(dross_rt::Lock::new(alloc::vec::Vec::new()));
    let o = out.clone();
    thread::spawn(1, move || {
        let mut b = [0u8; 8];
        let mut h = [Handle(0); 1];
        loop {
            let r = sys::channel_read(q, &mut b, &mut h, if blocking { FOREVER } else { 0 });
            let now = sys::clock();
            match r {
                Ok(_) => {
                    o.with(|v| v.push(now - u64::from_le_bytes(b)));
                    if sys::channel_write(q, &sys::clock().to_le_bytes(), &[]).is_err() {
                        break;
                    }
                }
                Err(Error::TimedOut) => core::hint::spin_loop(),
                Err(_) => break,
            }
        }
        sys::thread_exit();
    })
    .unwrap();
    let mut b = [0u8; 8];
    let mut h = [Handle(0); 1];
    let (mut back, mut rt) = (alloc::vec::Vec::new(), alloc::vec::Vec::new());
    for _ in 0..1000 {
        if pause {
            let until = sys::clock() + 50_000;
            while sys::clock() < until {}
        }
        let t0 = sys::clock();
        sys::channel_write(p, &t0.to_le_bytes(), &[]).unwrap();
        sys::channel_read(p, &mut b, &mut h, FOREVER).unwrap();
        let now = sys::clock();
        back.push(now - u64::from_le_bytes(b));
        rt.push(now - t0);
    }
    sys::close(p).unwrap();
    let mut v = out.with(core::mem::take);
    for l in [&mut v, &mut back, &mut rt] {
        l.sort_unstable();
        if l.is_empty() {
            l.push(0);
        }
    }
    (v, back, rt)
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
    let first = sys::notify_create().unwrap();
    let second = sys::notify_create().unwrap();
    let failed = sys::channel_write(y, b"", &[second, first]) == Err(Error::PeerClosed);
    let restored = sys::notify_signal(first, 1).is_ok() && sys::notify_signal(second, 2).is_ok()
        && sys::notify_wait(first, 0) == Ok(1) && sys::notify_wait(second, 0) == Ok(2);
    t.check("failed transfer preserves handles", failed && restored, "");
    sys::close(first).unwrap();
    sys::close(second).unwrap();

    // Race a sender transferring a handle with the peer closing on another CPU.
    if cpus > 1 {
        struct Race { phase: AtomicU64, send: AtomicU64, handle: AtomicU64, result: AtomicU64 }
        let race = Arc::new(Race { phase: AtomicU64::new(0), send: AtomicU64::new(0), handle: AtomicU64::new(0), result: AtomicU64::new(0) });
        let other = race.clone();
        thread::spawn(1, move || {
            for _ in 0..512 {
                while other.phase.load(Ordering::Acquire) != 1 { core::hint::spin_loop(); }
                let send = Handle(other.send.load(Ordering::Relaxed) as u32);
                let handle = Handle(other.handle.load(Ordering::Relaxed) as u32);
                other.phase.store(2, Ordering::Release);
                let good = match sys::channel_write(send, b"race", &[handle]) {
                    Ok(()) => sys::notify_signal(handle, 1) == Err(Error::BadHandle),
                    Err(Error::PeerClosed) => sys::notify_signal(handle, 1).is_ok() && sys::close(handle).is_ok(),
                    _ => false,
                };
                let good = good && sys::close(send).is_ok();
                other.result.store(good as u64, Ordering::Relaxed);
                other.phase.store(3, Ordering::Release);
                while other.phase.load(Ordering::Acquire) != 0 { core::hint::spin_loop(); }
                other.phase.store(4, Ordering::Release);
            }
        }).unwrap();
        let mut good = true;
        let wait_for = |wanted: u64, different: bool| {
            let deadline = sys::after_ms(5000);
            while (race.phase.load(Ordering::Acquire) == wanted) == different {
                if sys::clock() >= deadline { return false; }
                core::hint::spin_loop();
            }
            true
        };
        for _ in 0..512 {
            let (send, recv) = sys::channel_create().unwrap();
            let n = sys::notify_create().unwrap();
            let h = sys::dup(n, rights::ALL).unwrap();
            race.send.store(send.0 as u64, Ordering::Relaxed);
            race.handle.store(h.0 as u64, Ordering::Relaxed);
            race.phase.store(1, Ordering::Release);
            if !wait_for(1, true) { good = false; break; }
            good &= sys::close(recv).is_ok();
            if !wait_for(3, false) { good = false; break; }
            good &= race.result.load(Ordering::Relaxed) == 1;
            good &= sys::close(n).is_ok();
            race.phase.store(0, Ordering::Release);
            if !wait_for(4, false) { good = false; break; }
        }
        t.check("channel close and transfer race", good, "(512 attempts)");
    }

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

    // Every CPU's timer: a 10 ms deadline on each.
    let waits = Arc::new((0..cpus).map(|_| AtomicU64::new(0)).collect::<alloc::vec::Vec<_>>());
    let tdone = sys::notify_create().unwrap();
    for cpu in 0..cpus {
        let (w, d) = (waits.clone(), sys::dup(tdone, rights::ALL).unwrap());
        thread::spawn(cpu as u64, move || {
            let n = sys::notify_create().unwrap();
            let t0 = sys::clock();
            let r = sys::notify_wait(n, sys::after_ms(10));
            w[cpu as usize].store(if r == Err(Error::TimedOut) { (sys::clock() - t0) / 1000 } else { u64::MAX - 1 }, Ordering::Relaxed);
            sys::notify_signal(d, 1 << cpu).unwrap();
        })
        .unwrap();
    }
    let mut got = 0;
    let t_end = sys::after_ms(2000);
    while got != want && sys::clock() < t_end {
        got |= sys::notify_wait(tdone, t_end).unwrap_or(0);
    }
    let us: alloc::vec::Vec<u64> = waits.iter().map(|w| w.load(Ordering::Relaxed)).collect();
    // On hardware within 5 ms; under QEMU on macOS the host delays timers by tens of ms, so only that they fire.
    let vm = core::arch::x86_64::__cpuid(1).ecx & 1 << 31 != 0;
    let late = if vm { 1_000_000 } else { 15_000 };
    let fine = got == want && us.iter().all(|&u| (10_000..late).contains(&u));
    t.check("timers", fine, &alloc::format!("(10 ms on each cpu took {:?} us; 0 = never woke)", us));

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
        let mut ok = true;
        let mut each = alloc::vec::Vec::with_capacity(N as usize);
        for i in 0..N {
            let t0 = sys::clock();
            sys::channel_write(p, &i.to_le_bytes(), &[]).unwrap();
            let (n, _) = sys::channel_read(p, &mut buf, &mut hs, FOREVER).unwrap();
            each.push(sys::clock() - t0);
            ok &= n == 8 && buf[..8] == i.to_le_bytes();
        }
        // The median: a mean would be dominated by the odd time slice lost to another thread on the CPU.
        each.sort_unstable();
        let ns = each[each.len() / 2];
        t.check("round trips", ok, &alloc::format!("(cpu 0 <-> cpu 1: median {}.{} us)", ns / 1000, ns % 1000 / 100));
        sys::close(p).unwrap();
        let pct = |l: &alloc::vec::Vec<u64>| alloc::format!("min {} p50 {} p99 {} max {} ns", l[0], l[l.len() / 2], l[l.len() * 99 / 100], l[l.len() - 1]);
        for (blocking, pause) in [(true, true), (false, true), (true, false)] {
            let (there, back, rt) = one_way(blocking, pause);
            t.check(
                &alloc::format!("latency {} {}", if blocking { "blocked" } else { "spinning" }, if pause { "paced" } else { "back to back" }),
                there.len() == 1000,
                &alloc::format!("(cpu 0 -> 1: {}; 1 -> 0: {}; round trip: {})", pct(&there), pct(&back), pct(&rt)),
            );
        }
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
