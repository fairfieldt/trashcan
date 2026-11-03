use super::*;
use alloc::boxed::Box;
#[derive(Clone)]
#[repr(align(64))]
pub struct Fpu([u8; 4096]);
#[derive(Clone)]
pub struct Saved {
    regs: [u64; 22],
    mask: u64,
    fpu: Box<Fpu>,
}

pub fn send(p: &Arc<Process>, sig: u32) {
    let action = linux(p).sigactions.lock()[sig as usize - 1];
    // SIGCONT resumes a stopped process even when blocked or ignored.
    if sig == 18 {
        state(p).stopped.update(|v| *v = false);
        state(p).wait_event.store(0xffff, Ordering::Release);
    }
    if action[0] == 1 || action[0] == 0 && matches!(sig, 17 | 23 | 28) {
        return;
    }
    if sig == 9 {
        state(p).term_signal.store(sig, Ordering::Relaxed);
        p.kill(128 + sig as i64);
        return;
    }
    state(p)
        .pending
        .fetch_or(1u64 << (sig - 1), Ordering::AcqRel);
    p.wake_threads();
}
pub fn dispatch(p: &Arc<Process>, f: &mut Frame, n: u64, a: [u64; 6]) -> Option<R> {
    if !matches!(n, 13..=15 | 34 | 127 | 130) {
        return None;
    }
    Some(inner(p, f, n, a))
}
fn inner(p: &Arc<Process>, f: &mut Frame, n: u64, a: [u64; 6]) -> R {
    match n {
        15 => {
            let Some(saved) = state(p).saved_signals.lock().pop() else {
                return Err(EFAULT);
            };
            // SAFETY: this is a kernel-owned saved frame, never an untrusted sigcontext.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    saved.regs.as_ptr(),
                    f as *mut Frame as *mut u64,
                    22,
                );
            }
            *linux(p).sigmask.lock() = saved.mask;
            let mask = cpu::xsave_mask();
            // SAFETY: aligned XSAVE image captured by this kernel, not user-writable.
            unsafe {
                core::arch::asm!("xrstor64 [{}]", in(reg) saved.fpu.0.as_ptr(), in("eax") mask as u32, in("edx") (mask >> 32) as u32);
            }
            Ok(f.rax)
        }
        13 => {
            if a[0] == 0 || a[0] > 64 || a[0] == 9 || a[0] == 19 || a[3] != 8 {
                Err(EINVAL)
            } else {
                let index = a[0] as usize - 1;
                if a[2] != 0 {
                    write_user(
                        p,
                        a[2],
                        &linux(p).sigactions.lock()[index]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                }
                if a[1] != 0 {
                    let raw = read_user(p, a[1], 32)?;
                    let mut action = [0; 4];
                    for (i, bytes) in raw.as_chunks::<8>().0.iter().enumerate() {
                        action[i] = u64::from_le_bytes(*bytes);
                    }
                    // This first ABI implements ordinary handlers. It must not pretend
                    // to supply Linux's siginfo/ucontext or alternate-stack structures.
                    if action[0] > 1 && action[1] & (4 | 0x0800_0000) != 0 {
                        return Err(ENOSYS);
                    }
                    linux(p).sigactions.lock()[index] = action;
                }
                Ok(0)
            }
        }
        14 => {
            if a[3] != 8 {
                Err(EINVAL)
            } else {
                let old = *linux(p).sigmask.lock();
                if a[2] != 0 {
                    write_u64(p, a[2], old)?;
                }
                if a[1] != 0 {
                    let bits = read_u64(p, a[1])?;
                    let unmaskable = (1 << 8) | (1 << 18);
                    *linux(p).sigmask.lock() = match a[0] {
                        0 => old | bits,
                        1 => old & !bits,
                        2 => bits,
                        _ => return Err(EINVAL),
                    } & !unmaskable;
                }
                Ok(0)
            }
        }
        127 => write_u64(p, a[0], state(p).pending.load(Ordering::Acquire)).map(|_| 0),
        34 | 130 => {
            let old = *linux(p).sigmask.lock();
            if n == 130 {
                *linux(p).sigmask.lock() = read_u64(p, a[0])? & !((1 << 8) | (1 << 18));
            }
            while !super::interrupted(p) && !sched::current().killed() {
                let _ = sched::sleep_ns(1_000_000);
            }
            if n == 130 {
                *linux(p).sigmask.lock() = old;
            }
            Err(EINTR)
        }
        _ => Err(ENOSYS),
    }
}
pub fn user_return(f: &mut Frame) {
    let p = sched::current_process();
    if p.linux.is_none() {
        return;
    }
    if state(&p).stopped.with(|v| *v) {
        let _ = state(&p)
            .stopped
            .wait_uninterruptible(u64::MAX, |v| if !*v { Some(()) } else { None });
    }
    loop {
        let pending = state(&p).pending.load(Ordering::Acquire) & !*linux(&p).sigmask.lock();
        if pending == 0 {
            return;
        }
        let index = pending.trailing_zeros() as usize;
        let sig = index as u32 + 1;
        state(&p)
            .pending
            .fetch_and(!(1u64 << index), Ordering::AcqRel);
        let action = linux(&p).sigactions.lock()[index];
        if action[0] == 1 || action[0] == 0 && matches!(sig, 17 | 18 | 23 | 28) {
            continue;
        }
        if action[0] == 0 && matches!(sig, 19..=22) {
            state(&p).stopped.update(|v| *v = true);
            state(&p)
                .wait_event
                .store((sig << 8) | 0x7f, Ordering::Release);
            if let Some(parent) = state(&p).parent.lock().upgrade() {
                send(&parent, 17);
            }
            let _ = state(&p)
                .stopped
                .wait_uninterruptible(u64::MAX, |v| if !*v { Some(()) } else { None });
            continue;
        }
        if action[0] == 0 {
            state(&p).term_signal.store(sig, Ordering::Relaxed);
            p.kill(128 + sig as i64);
            return;
        }
        if action[2] == 0 || action[0] >= dross_abi::USER_TOP || action[2] >= dross_abi::USER_TOP {
            p.kill(139);
            return;
        }
        let mut regs = [0; 22];
        // SAFETY: initialized repr(C) user frame, saved before changing user registers.
        unsafe {
            core::ptr::copy_nonoverlapping(f as *const Frame as *const u64, regs.as_mut_ptr(), 22);
        }
        if action[1] & 0x1000_0000 != 0
            && f.vector == 0x100
            && f.rax as i64 == -EINTR
            && matches!(f.error, 0 | 1 | 19 | 20 | 43 | 61 | 288)
        {
            regs[14] = f.error;
            regs[17] = f.rip.saturating_sub(2);
        }
        let sp = match f.rsp.checked_sub(128 + 16) {
            Some(v) => (v & !15) - 8,
            None => {
                p.kill(139);
                return;
            }
        };
        if write_u64(&p, sp, action[2]).is_err() {
            p.kill(139);
            return;
        }
        let mut saved = state(&p).saved_signals.lock();
        if saved.len() == 8 {
            p.kill(139);
            return;
        }
        let mask = *linux(&p).sigmask.lock();
        let mut fpu = Box::new(Fpu([0; 4096]));
        let components = cpu::xsave_mask();
        // SAFETY: aligned, private 4 KiB XSAVE area, as used by the scheduler.
        unsafe {
            core::arch::asm!("xsave64 [{}]", in(reg) fpu.0.as_mut_ptr(), in("eax") components as u32, in("edx") (components >> 32) as u32);
        }
        saved.push(Saved { regs, mask, fpu });
        drop(saved);
        *linux(&p).sigmask.lock() = mask
            | action[3]
            | if action[1] & 0x4000_0000 == 0 {
                1u64 << index
            } else {
                0
            };
        if action[1] & 0x8000_0000 != 0 {
            linux(&p).sigactions.lock()[index] = [0; 4];
        }
        f.rsp = sp;
        f.rip = action[0];
        f.rdi = sig as u64;
        return;
    }
}
