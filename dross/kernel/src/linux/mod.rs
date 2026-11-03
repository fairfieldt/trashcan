//! The Linux system call interface, for static Linux executables (Rust `std` and C on musl): how gpud, tbd and
//! ethd run on dross unchanged. A process started with `SPAWN_LINUX` gets this interface for `syscall`; its
//! native dross calls (handles, pcid, DMA) stay reachable with `NATIVE` set in the number.
//!
//! Covered: memory (mmap of anonymous memory and memfds, munmap and mprotect of any range), threads (clone,
//! futex, TLS, exit with clear-tid), files (a read-only tree from the `rootfs.tar` boot module, the ACPI tables
//! under /sys/firmware/acpi/tables, /dev/null, logs on stdout and stderr), Unix sockets (stream, datagram and
//! seqpacket, with fd passing), pipes, memfd, eventfd, poll, time, sleeping, randomness. Signals are accepted and
//! never delivered (except an abort, which ends the process). The optional `unix` feature adds fork, exec,
//! signals, writable files, PTYs and TCP capability RPC; see unix/mod.rs and ../../unix/README.md.

mod exec;
mod fs;
mod sock;
#[cfg(feature = "unix")]
pub mod unix;

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use dross_abi::vm;
use spin::Mutex;

use crate::interrupts::Frame;
use crate::process::Process;
use crate::sched::{self, Thread, WaitCell, WaitError};
use crate::{cpu, time};

pub use exec::spawn;
pub use fs::{init_rootfs, memory_fd, notification_fd, publish_table, Files};

pub type Errno = i64;
pub const ENOENT: Errno = 2;
pub const EINTR: Errno = 4;
pub const EBADF: Errno = 9;
pub const EAGAIN: Errno = 11;
pub const ENOMEM: Errno = 12;
pub const EACCES: Errno = 13;
pub const EFAULT: Errno = 14;
pub const EEXIST: Errno = 17;
pub const ENOTDIR: Errno = 20;
pub const EISDIR: Errno = 21;
pub const EINVAL: Errno = 22;
pub const EMFILE: Errno = 24;
pub const ENOTTY: Errno = 25;
pub const ESPIPE: Errno = 29;
pub const EPIPE: Errno = 32;
pub const ENOSYS: Errno = 38;
pub const ENOTSOCK: Errno = 88;
pub const EAFNOSUPPORT: Errno = 97;
pub const EADDRINUSE: Errno = 98;
pub const ENOTCONN: Errno = 107;
pub const ETIMEDOUT: Errno = 110;
pub const ECONNREFUSED: Errno = 111;

pub type R = Result<u64, Errno>;
const TRACE: bool = false;

/// What a Linux process has beyond a dross one.
pub struct Linux {
    pub files: Files,
    #[cfg(feature = "unix")]
    pub unix: unix::State,
    sigmask: Mutex<u64>,
    sigactions: Mutex<[[u64; 4]; 64]>,
}

impl Linux {
    pub fn new(name: &str) -> Linux {
        Linux { #[cfg(feature = "unix")] unix: unix::State::new(), files: Files::new(name), sigmask: Mutex::new(0), sigactions: Mutex::new([[0; 4]; 64]) }
    }
}

fn linux(p: &Process) -> &Linux {
    p.linux.as_ref().expect("not a Linux process")
}

fn fault<T>(_: dross_abi::Error) -> Result<T, Errno> {
    Err(EFAULT)
}

pub fn read_user(p: &Process, ptr: u64, len: u64) -> Result<Vec<u8>, Errno> {
    if len > 64 << 20 {
        return Err(EINVAL);
    }
    let mut v = alloc::vec![0u8; len as usize];
    // SAFETY: with_user checked the range is mapped readable while its lock is held.
    p.aspace.with_user(ptr, len, false, |src| unsafe { core::ptr::copy_nonoverlapping(src, v.as_mut_ptr(), len as usize) }).or_else(fault)?;
    Ok(v)
}

pub fn write_user(p: &Process, ptr: u64, data: &[u8]) -> Result<(), Errno> {
    // SAFETY: with_user checked the range is mapped writable while its lock is held.
    p.aspace.with_user(ptr, data.len() as u64, true, |dst| unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) }).or_else(fault)
}

pub fn read_u64(p: &Process, ptr: u64) -> Result<u64, Errno> {
    Ok(u64::from_le_bytes(read_user(p, ptr, 8)?.try_into().unwrap()))
}

pub fn read_u32(p: &Process, ptr: u64) -> Result<u32, Errno> {
    Ok(u32::from_le_bytes(read_user(p, ptr, 4)?.try_into().unwrap()))
}

pub fn write_u64(p: &Process, ptr: u64, v: u64) -> Result<(), Errno> {
    write_user(p, ptr, &v.to_le_bytes())
}

pub fn write_u32(p: &Process, ptr: u64, v: u32) -> Result<(), Errno> {
    write_user(p, ptr, &v.to_le_bytes())
}

/// A NUL-terminated string from user memory (read page by page, so it may end near the end of a mapping).
pub fn read_cstr(p: &Process, ptr: u64, max: usize) -> Result<String, Errno> {
    let mut out = Vec::new();
    let mut at = ptr;
    while out.len() < max {
        let n = (4096 - at % 4096).min((max - out.len()) as u64);
        let chunk = read_user(p, at, n)?;
        if let Some(z) = chunk.iter().position(|&b| b == 0) {
            out.extend_from_slice(&chunk[..z]);
            return String::from_utf8(out).map_err(|_| EINVAL);
        }
        out.extend_from_slice(&chunk);
        at += n;
    }
    Err(EINVAL)
}

/// (base, len) pairs of an iovec array.
pub fn read_iov(p: &Process, iov: u64, n: u64) -> Result<Vec<(u64, u64)>, Errno> {
    if n > 1024 {
        return Err(EINVAL);
    }
    let raw = read_user(p, iov, n * 16)?;
    Ok(raw.chunks_exact(16).map(|c| (u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..].try_into().unwrap()))).collect())
}

/// Scatters `data` over iovecs; returns the bytes written.
pub fn scatter(p: &Process, iov: &[(u64, u64)], data: &[u8]) -> Result<usize, Errno> {
    let mut done = 0;
    for &(base, len) in iov {
        if done == data.len() {
            break;
        }
        let n = (len as usize).min(data.len() - done);
        write_user(p, base, &data[done..done + n])?;
        done += n;
    }
    Ok(done)
}

pub fn gather(p: &Process, iov: &[(u64, u64)]) -> Result<Vec<u8>, Errno> {
    let mut v = Vec::new();
    for &(base, len) in iov {
        v.extend_from_slice(&read_user(p, base, len)?);
    }
    Ok(v)
}

fn wait_errno(e: WaitError) -> Errno {
    match e {
        WaitError::TimedOut => ETIMEDOUT,
        WaitError::Killed => EINTR,
    }
}

/// A wall clock for programs that want one: boot is taken to be 2026-01-01 (the Mac Pro's RTC isn't read yet).
const BOOT_EPOCH_S: u64 = 1_767_225_600;

fn timespec(ns: u64) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&(ns / 1_000_000_000).to_le_bytes());
    b[8..].copy_from_slice(&(ns % 1_000_000_000).to_le_bytes());
    b
}

fn read_timespec(p: &Process, ptr: u64) -> Result<u64, Errno> {
    let b = read_user(p, ptr, 16)?;
    let (s, ns) = (i64::from_le_bytes(b[..8].try_into().unwrap()), i64::from_le_bytes(b[8..].try_into().unwrap()));
    if s < 0 || !(0..1_000_000_000).contains(&ns) {
        return Err(EINVAL);
    }
    Ok((s as u64).saturating_mul(1_000_000_000).saturating_add(ns as u64))
}

/// The monotonic clock (boot-relative) in ns; `realtime` adds the epoch.
fn clock_ns(clock: u64) -> u64 {
    let now = time::now_ns();
    match clock {
        0 | 5 => now + BOOT_EPOCH_S * 1_000_000_000,
        _ => now,
    }
}

/// One futex: its waiters in arrival order, each with a ticket, and the tickets woken but not yet back.
struct Futex {
    waiters: alloc::collections::VecDeque<(u64, Arc<Thread>)>,
    woken: Vec<u64>,
    next: u64,
}

type FutexKey = (usize, u64);
static FUTEXES: Mutex<alloc::collections::BTreeMap<FutexKey, Arc<Mutex<Futex>>>> = Mutex::new(alloc::collections::BTreeMap::new());

/// Futexes are keyed by the memory object and offset behind the address, so shared mappings work too.
fn futex_of(p: &Process, addr: u64) -> Result<Arc<Mutex<Futex>>, Errno> {
    let (mem, off) = p.aspace.lookup(addr).ok_or(EFAULT)?;
    let key = (Arc::as_ptr(&mem) as usize, off);
    Ok(FUTEXES.lock().entry(key).or_insert_with(|| Arc::new(Mutex::new(Futex { waiters: Default::default(), woken: Vec::new(), next: 0 }))).clone())
}

/// Wakes up to `n` waiters, oldest first. Returns how many.
fn futex_wake(p: &Process, addr: u64, n: u64) -> Result<u64, Errno> {
    let fx = futex_of(p, addr)?;
    let mut g = fx.lock();
    let k = (n as usize).min(g.waiters.len());
    for _ in 0..k {
        let (ticket, t) = g.waiters.pop_front().unwrap();
        g.woken.push(ticket);
        sched::wake(&t);
    }
    Ok(k as u64)
}

/// FUTEX_WAIT: sleeps while `*addr == val`, until a wake, the deadline (TSC) or a kill.
fn futex_wait(p: &Process, addr: u64, val: u32, deadline_tsc: u64) -> R {
    let fx = futex_of(p, addr)?;
    let me = sched::current();
    let ticket = {
        let mut g = fx.lock();
        // Checked under the futex's lock, which wakers take too: a wake after the store this compares against
        // finds the waiter queued.
        if read_u32(p, addr)? != val {
            return Err(EAGAIN);
        }
        let t = g.next;
        g.next += 1;
        g.waiters.push_back((t, me.clone()));
        sched::prepare_block(&me, deadline_tsc);
        t
    };
    loop {
        sched::schedule();
        let mut g = fx.lock();
        if let Some(i) = g.woken.iter().position(|&t| t == ticket) {
            g.woken.swap_remove(i);
            return Ok(0);
        }
        let timed_out = cpu::rdtsc() >= deadline_tsc;
        if timed_out || me.killed() {
            g.waiters.retain(|(t, _)| *t != ticket);
            return Err(if timed_out { ETIMEDOUT } else { EINTR });
        }
        // Woken for something else (a timer of an earlier wait): wait again.
        sched::prepare_block(&me, deadline_tsc);
    }
}

/// Ends the calling thread the Linux way: clear-tid, then exit.
pub fn exit_thread(p: &Process, me: &Thread) -> ! {
    let ct = me.clear_tid.load(core::sync::atomic::Ordering::Relaxed);
    if ct != 0 {
        let _ = write_u32(p, ct, 0);
        let _ = futex_wake(p, ct, u64::MAX);
    }
    sched::exit_current()
}

#[cfg(not(feature = "unix"))]
fn random_u64() -> u64 {
    let mut v = 0u64;
    let mut ok = 0u8;
    if cpu::cpuid(1, 0).2 & 1 << 30 != 0 {
        // SAFETY: RDRAND exists (CPUID.1:ECX.30).
        unsafe { core::arch::asm!("rdrand {}", "setc {}", out(reg) v, out(reg_byte) ok) };
    }
    if ok == 0 {
        // SplitMix64 over the TSC.
        let mut z = cpu::rdtsc().wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        v = z ^ (z >> 31);
    }
    v
}

pub fn random_bytes(buf: &mut [u8]) {
    #[cfg(feature = "unix")]
    { unix::entropy::fill(buf).expect("secure entropy unavailable"); }
    #[cfg(not(feature = "unix"))]
    {
        for c in buf.chunks_mut(8) {
            let r = random_u64().to_le_bytes();
            c.copy_from_slice(&r[..c.len()]);
        }
    }
}

fn prot_flags(prot: u64) -> u64 {
    (prot & 1 != 0) as u64 * vm::READ | (prot & 2 != 0) as u64 * vm::WRITE | (prot & 4 != 0) as u64 * vm::EXECUTE
}

fn mmap(p: &Process, addr: u64, len: u64, prot: u64, flags: u64, fd: u64, off: u64) -> R {
    const MAP_SHARED: u64 = 1;
    const MAP_FIXED: u64 = 0x10;
    const MAP_ANONYMOUS: u64 = 0x20;
    const MAP_FIXED_NOREPLACE: u64 = 0x10_0000;
    if len == 0 || off % 4096 != 0 {
        return Err(EINVAL);
    }
    let len = len.next_multiple_of(4096);
    let fixed = flags & (MAP_FIXED | MAP_FIXED_NOREPLACE) != 0;
    if fixed && addr % 4096 != 0 {
        return Err(EINVAL);
    }
    let at = if fixed { addr } else { 0 };
    let f = prot_flags(prot) | if cfg!(feature = "unix") && flags & MAP_SHARED != 0 { crate::paging::FORK_SHARED } else { 0 };
    let r = if flags & MAP_ANONYMOUS != 0 {
        if f == 0 {
            p.aspace.reserve(len, at, flags & MAP_FIXED != 0)
        } else {
            let m = crate::memory::Memory::anonymous(len, false, false).map_err(|_| ENOMEM)?;
            if flags & MAP_FIXED != 0 { p.aspace.map_replace(Arc::new(m), 0, len, at, f) } else { p.aspace.map(Arc::new(m), 0, len, at, f) }
        }
    } else {
        let file = linux(p).files.get(fd as i32)?;
        let (mem, shared_ok) = file.backing(off, len)?;
        let mem = if flags & MAP_SHARED != 0 {
            if !shared_ok {
                return Err(EACCES);
            }
            mem
        } else {
            // A private copy.
            let copy = crate::memory::Memory::anonymous(len, false, false).map_err(|_| ENOMEM)?;
            let mut buf = alloc::vec![0u8; len.min(mem.size().saturating_sub(off)) as usize];
            mem.read(off, &mut buf).map_err(|_| EINVAL)?;
            copy.write(0, &buf).map_err(|_| EINVAL)?;
            return p.aspace.map(Arc::new(copy), 0, len, at, f).map_err(|_| ENOMEM);
        };
        if flags & MAP_FIXED != 0 { p.aspace.map_replace(mem, off, len, at, f) } else { p.aspace.map(mem, off, len, at, f) }
    };
    r.map_err(|e| match e {
        dross_abi::Error::Busy => EEXIST,
        dross_abi::Error::OutOfRange => EINVAL,
        _ => ENOMEM,
    })
}

fn clone(p: &Arc<Process>, f: &Frame, flags: u64, stack: u64, ptid: u64, ctid: u64, tls: u64) -> R {
    const CLONE_VM: u64 = 0x100;
    const CLONE_THREAD: u64 = 0x10000;
    const CLONE_SETTLS: u64 = 0x80000;
    const CLONE_PARENT_SETTID: u64 = 0x10_0000;
    const CLONE_CHILD_CLEARTID: u64 = 0x20_0000;
    const CLONE_CHILD_SETTID: u64 = 0x100_0000;
    if flags & (CLONE_VM | CLONE_THREAD) != CLONE_VM | CLONE_THREAD {
        // No new processes from Linux programs.
        return Err(ENOSYS);
    }
    // The child resumes where the parent does, with rax 0 and its own stack.
    let mut regs = [0u64; 22];
    // SAFETY: Frame is 22 u64s.
    unsafe { core::ptr::copy_nonoverlapping(f as *const Frame as *const u64, regs.as_mut_ptr(), 22) };
    regs[14] = 0; // rax
    regs[17] = f.rcx; // syscall return address
    regs[19] = f.r11; // flags
    regs[20] = stack;
    let fs = if flags & CLONE_SETTLS != 0 { tls } else { sched::current().fs_base.load(core::sync::atomic::Ordering::Relaxed) };
    let t = sched::spawn_with(p.clone(), sched::pick_cpu(), regs, fs);
    if flags & CLONE_CHILD_CLEARTID != 0 {
        t.clear_tid.store(ctid, core::sync::atomic::Ordering::Relaxed);
    }
    if flags & CLONE_PARENT_SETTID != 0 {
        write_u32(p, ptid, t.tid)?;
    }
    if flags & CLONE_CHILD_SETTID != 0 {
        write_u32(p, ctid, t.tid)?;
    }
    Ok(t.tid as u64)
}

/// A Linux system call from a Linux process. Returns the value for rax.
pub fn syscall(f: &mut Frame) -> u64 {
    let p = sched::current_process();
    let a = [f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.r9];
    let n = f.rax;
    #[cfg(feature = "unix")] { f.error = n; }
    #[cfg(feature = "unix")]
    let r = match unix::dispatch(&p, f, n, a) { Some(r) => r, None => dispatch(&p, f, n, a) };
    #[cfg(not(feature = "unix"))]
    let r = dispatch(&p, f, n, a);
    if TRACE { crate::println!("TRACE {} tid {} cpu {}: {} {:x?} -> {:x?}", p.name, sched::current().tid, crate::cpu::index(), n, &a[..4], r); }
    match r {
        Ok(v) => v,
        Err(e) => (-e) as u64,
    }
}

fn dispatch(p: &Arc<Process>, f: &Frame, n: u64, a: [u64; 6]) -> R {
    let l = linux(p);
    match n {
        // Files and I/O.
        0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 32 | 33 | 72 | 77 | 78 | 79 | 80 | 83 | 89 | 90 | 91 | 92 | 93 | 95
        | 217 | 257 | 258 | 260 | 262 | 263 | 267 | 268 | 269 | 284 | 285 | 290 | 292 | 293 | 319 | 332 | 439 | 7 | 271 | 87 => {
            fs::syscall(p, n, a)
        }
        // Sockets.
        41..=55 | 288 => sock::syscall(p, n, a),
        9 => mmap(p, a[0], a[1], a[2], a[3], a[4], a[5]),
        10 => p.aspace.protect(a[0] & !0xfff, (a[1] + (a[0] & 0xfff)).next_multiple_of(4096), prot_flags(a[2])).map(|_| 0).map_err(|_| ENOMEM),
        11 => p.aspace.unmap_range(a[0], a[1].next_multiple_of(4096)).map(|_| 0).map_err(|_| EINVAL),
        // brk: no heap that way (musl falls back to mmap).
        12 => Ok(0),
        25 => Err(ENOMEM),
        26 | 28 | 149 | 150 | 151 | 152 => Ok(0),
        // Signals: kept, never delivered.
        13 => {
            let sig = a[0] as usize;
            if sig == 0 || sig > 64 {
                return Err(EINVAL);
            }
            let old = l.sigactions.lock()[sig - 1];
            if a[2] != 0 {
                write_user(p, a[2], &old.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
            }
            if a[1] != 0 {
                let new = read_user(p, a[1], 32)?;
                let mut v = [0u64; 4];
                for (i, c) in new.chunks_exact(8).enumerate() {
                    v[i] = u64::from_le_bytes(c.try_into().unwrap());
                }
                l.sigactions.lock()[sig - 1] = v;
            }
            Ok(0)
        }
        14 => {
            let mut m = l.sigmask.lock();
            if a[2] != 0 {
                write_u64(p, a[2], *m)?;
            }
            if a[1] != 0 {
                let set = read_u64(p, a[1])?;
                *m = match a[0] {
                    0 => *m | set,
                    1 => *m & !set,
                    2 => set,
                    _ => return Err(EINVAL),
                };
            }
            Ok(0)
        }
        131 => {
            if a[1] != 0 {
                // No alternate stack in use.
                let mut ss = [0u8; 24];
                ss[8..12].copy_from_slice(&2u32.to_le_bytes());
                write_user(p, a[1], &ss)?;
            }
            Ok(0)
        }
        // rt_sigtimedwait, pause, rt_sigsuspend: no signal ever comes; sleep until the deadline or a kill.
        128 | 34 | 130 => {
            let deadline = if n == 128 && a[2] != 0 { time::tsc_at_ns(time::now_ns() + read_timespec(p, a[2])?) } else { u64::MAX };
            static NOTHING: WaitCell<()> = WaitCell::new(());
            match NOTHING.wait(deadline, |_| None::<()>) {
                Err(WaitError::TimedOut) => Err(EAGAIN),
                _ => Err(EINTR),
            }
        }
        127 => write_u64(p, a[0], 0).map(|_| 0),
        // kill, tkill, tgkill: signal 0 checks; anything else to ourselves ends the process (abort, raise).
        62 | 200 | 234 => {
            let sig = if n == 234 { a[2] } else { a[1] };
            if sig == 0 {
                return Ok(0);
            }
            crate::println!("{}: signal {} raised: process ends", p.name, sig);
            p.kill(128 + sig as i64);
            Ok(0)
        }
        // Threads and processes.
        56 => clone(p, f, a[0], a[1], a[2], a[3], a[4]),
        57 | 58 | 59 | 61 | 435 => Err(ENOSYS),
        60 => {
            let me = sched::current();
            exit_thread(p, &me)
        }
        231 => {
            p.kill(a[0] as i32 as i64);
            Ok(0)
        }
        39 => Ok(p.pid() as u64),
        110 => Ok(1),
        186 => Ok(sched::current().tid as u64),
        218 => {
            let me = sched::current();
            me.clear_tid.store(a[0], core::sync::atomic::Ordering::Relaxed);
            Ok(me.tid as u64)
        }
        158 => match a[0] {
            0x1002 => {
                sched::set_fs_base(a[1]);
                Ok(0)
            }
            0x1003 => write_u64(p, a[1], sched::current().fs_base.load(core::sync::atomic::Ordering::Relaxed)).map(|_| 0),
            _ => Err(EINVAL),
        },
        202 => {
            let (addr, op, val) = (a[0], a[1] & 0x7f, a[2] as u32);
            match op {
                0 | 9 => {
                    let deadline = if a[3] == 0 {
                        u64::MAX
                    } else {
                        let ts = read_timespec(p, a[3])?;
                        if op == 9 {
                            // Absolute, on the monotonic clock (or realtime with FUTEX_CLOCK_REALTIME).
                            let now = if a[1] & 256 != 0 { clock_ns(0) } else { time::now_ns() };
                            time::tsc_at_ns(time::now_ns() + ts.saturating_sub(now))
                        } else {
                            time::tsc_at_ns(time::now_ns().saturating_add(ts))
                        }
                    };
                    futex_wait(p, addr, val, deadline)
                }
                1 | 10 => futex_wake(p, addr, a[2]),
                // Requeue: wake the waiters instead of moving them (a spurious wakeup is allowed).
                3 | 4 => futex_wake(p, addr, u64::MAX),
                // Wake-op: the operation on the second word isn't done; wake both sets.
                5 => Ok(futex_wake(p, addr, a[2])? + futex_wake(p, a[4], a[3])?),
                _ => Err(ENOSYS),
            }
        }
        273 => Ok(0),
        // reboot: restart resets the machine; the Ctrl-Alt-Del settings are accepted.
        169 => {
            if a[0] != 0xfee1_dead {
                return Err(EINVAL);
            }
            match a[2] as u32 {
                0x0123_4567 => crate::qemu::reset(&alloc::format!("reboot by {}", p.name), true, 300),
                0x89ab_cdef | 0 => Ok(0),
                _ => Err(EINVAL),
            }
        }
        // fsync, fdatasync, sync, syncfs: files live in memory.
        74 | 75 | 162 | 306 => Ok(0),
        274 | 334 | 324 => Err(ENOSYS),
        24 => {
            sched::request_resched();
            Ok(0)
        }
        // sched_setaffinity: placement is decided at creation (threads are pinned); accept.
        203 => Ok(0),
        204 => {
            let n = crate::smp::online() as usize;
            let len = (a[1] as usize).min(128);
            let mut mask = alloc::vec![0u8; len];
            for c in 0..n.min(len * 8) {
                mask[c / 8] |= 1 << (c % 8);
            }
            write_user(p, a[2], &mask)?;
            Ok(len as u64)
        }
        309 => {
            let c = cpu::this().index as u32;
            if a[0] != 0 {
                write_u32(p, a[0], c)?;
            }
            if a[1] != 0 {
                write_u32(p, a[1], 0)?;
            }
            Ok(0)
        }
        // Time.
        228 => write_user(p, a[1], &timespec(clock_ns(a[0]))).map(|_| 0),
        229 => {
            if a[1] != 0 {
                write_user(p, a[1], &timespec(1))?;
            }
            Ok(0)
        }
        96 => {
            let ns = clock_ns(0);
            let mut tv = [0u8; 16];
            tv[..8].copy_from_slice(&(ns / 1_000_000_000).to_le_bytes());
            tv[8..].copy_from_slice(&(ns / 1000 % 1_000_000).to_le_bytes());
            write_user(p, a[0], &tv).map(|_| 0)
        }
        201 => {
            let s = clock_ns(0) / 1_000_000_000;
            if a[0] != 0 {
                write_u64(p, a[0], s)?;
            }
            Ok(s)
        }
        35 | 230 => {
            let (ts, abs, clock) = if n == 35 { (a[0], false, 1) } else { (a[2], a[1] & 1 != 0, a[0]) };
            let t = read_timespec(p, ts)?;
            let now = clock_ns(clock);
            let wait = if abs { t.saturating_sub(now) } else { t };
            match sched::sleep_ns(wait) {
                Ok(()) => Ok(0),
                Err(_) => Err(EINTR),
            }
        }
        // Identity and limits.
        102 | 104 | 107 | 108 | 111 | 112 | 109 | 105 | 106 | 115 => Ok(0),
        63 => {
            let mut u = [0u8; 390];
            for (i, s) in ["Linux", "dross", "6.12.0-dross", "#1 SMP dross", "x86_64", "(none)"].iter().enumerate() {
                u[i * 65..i * 65 + s.len()].copy_from_slice(s.as_bytes());
            }
            write_user(p, a[0], &u).map(|_| 0)
        }
        97 | 302 => {
            let (res, new, old) = if n == 97 { (a[0], 0, a[1]) } else { (a[1], a[2], a[3]) };
            let _ = new;
            if old != 0 {
                let (cur, max) = match res {
                    3 => (8u64 << 20, u64::MAX),
                    7 => (1024, 4096),
                    _ => (u64::MAX, u64::MAX),
                };
                let mut b = [0u8; 16];
                b[..8].copy_from_slice(&cur.to_le_bytes());
                b[8..].copy_from_slice(&max.to_le_bytes());
                write_user(p, old, &b)?;
            }
            Ok(0)
        }
        160 => Ok(0),
        98 => write_user(p, a[1], &[0u8; 144]).map(|_| 0),
        157 => Ok(0),
        318 => {
            let len = a[1].min(1 << 20) as usize;
            let mut b = alloc::vec![0u8; len];
            random_bytes(&mut b);
            write_user(p, a[0], &b)?;
            Ok(len as u64)
        }
        _ => {
            crate::println!("{}: Linux system call {} not supported", p.name, n);
            Err(ENOSYS)
        }
    }
}
