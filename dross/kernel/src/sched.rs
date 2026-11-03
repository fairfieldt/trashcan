//! Threads and the scheduler. Every thread is pinned to one CPU; each CPU has its own run queue, timers and
//! idle thread, so scheduling never touches another CPU's state except to wake a thread there (its run queue
//! lock, then a reschedule IPI). The kernel runs with interrupts off; a thread enters the scheduler when it
//! blocks, yields or exits, or at the end of an interrupt (timer, reschedule IPI) taken in user mode or idle.
//!
//! A CPU arms its timer only when something needs it: the end of the running thread's time slice when others
//! are waiting, or the earliest deadline of a thread sleeping there. An idle CPU with no deadlines takes no
//! timer interrupts.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::arch::global_asm;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use spin::{Mutex, MutexGuard};

use crate::cpu::{self, PerCpu};
use crate::process::Process;
use crate::{apic, interrupts, mem, time};

/// 64 KiB: debug builds take a lot of stack in the system call paths.
const KSTACK_PAGES: usize = 16;
/// Time slice when other threads wait for the CPU.
const SLICE_NS: u64 = 10_000_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum State {
    Running = 0,
    Runnable = 1,
    /// Registered with a wait cell, still on the CPU.
    Blocking = 2,
    Blocked = 3,
    /// Woken between `Blocking` and leaving the CPU: it keeps running.
    Woken = 4,
    Dead = 5,
}

pub struct Thread {
    /// A number for Linux programs (`gettid`).
    pub tid: u32,
    /// The user FS base (thread-local storage), loaded at every switch.
    pub fs_base: AtomicU64,
    /// Linux `CLONE_CHILD_CLEARTID`: zeroed and woken (futex) when the thread exits.
    pub clear_tid: AtomicU64,
    /// None for the idle threads.
    pub process: Option<Arc<Process>>,
    pub cpu: u32,
    kstack: u64,
    ksp: UnsafeCell<u64>,
    /// Changed only under the run queue lock of `cpu`.
    state: AtomicU8,
    killed: AtomicBool,
    /// XSAVE area (one page, 64-byte aligned), 0 for kernel threads.
    xsave: u64,
}

// SAFETY: `ksp` is only touched by the thread's own CPU in `schedule`, with interrupts off.
unsafe impl Sync for Thread {}
unsafe impl Send for Thread {}

impl Thread {
    pub fn kstack_top(&self) -> u64 {
        self.kstack + (KSTACK_PAGES * 4096) as u64
    }

    fn state(&self) -> State {
        // SAFETY: only valid values are stored.
        unsafe { core::mem::transmute(self.state.load(Ordering::Acquire)) }
    }

    fn set(&self, s: State) {
        self.state.store(s as u8, Ordering::Release);
    }

    pub fn killed(&self) -> bool {
        self.killed.load(Ordering::Acquire)
    }

    /// Makes the thread leave at its next return from the kernel (waking it if it waits).
    pub fn kill(self: &Arc<Self>) {
        self.killed.store(true, Ordering::Release);
        wake(self);
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        mem::free_stack(self.kstack, KSTACK_PAGES);
        if self.xsave != 0 {
            mem::free_virt(self.xsave, 1);
        }
    }
}

struct RunQueue {
    current: Option<Arc<Thread>>,
    idle: Option<Arc<Thread>>,
    queue: VecDeque<Arc<Thread>>,
    /// Sleeping threads with a deadline (TSC), unordered: few at a time.
    timers: Vec<(u64, Arc<Thread>)>,
    /// A thread that exited on this CPU, freed once the CPU is off its stack.
    reap: Option<Arc<Thread>>,
    /// The page table root loaded on this CPU.
    cr3: u64,
    slice_end: u64,
}

pub struct Cpu {
    rq: Mutex<RunQueue>,
    need_resched: AtomicBool,
    /// Set by another CPU asking this one to flush its TLB.
    pub flush_pending: AtomicBool,
}

impl Cpu {
    pub const fn new() -> Self {
        Cpu {
            rq: Mutex::new(RunQueue {
                current: None,
                idle: None,
                queue: VecDeque::new(),
                timers: Vec::new(),
                reap: None,
                cr3: 0,
                slice_end: 0,
            }),
            need_resched: AtomicBool::new(false),
            flush_pending: AtomicBool::new(false),
        }
    }
}

fn cpu_of(t: &Thread) -> &'static PerCpu {
    cpu::get(t.cpu as usize).expect("thread on a CPU that is not up")
}

/// The thread running on this CPU.
pub fn current() -> Arc<Thread> {
    cpu::this().sched.rq.lock().current.clone().expect("no current thread")
}

pub fn current_process() -> Arc<Process> {
    current().process.clone().expect("not in a process")
}

/// Turns the code running on this CPU (boot flow) into its idle thread.
pub fn init_cpu() {
    let pc = cpu::this();
    let idle = Arc::new(Thread {
        tid: 0,
        fs_base: AtomicU64::new(0),
        clear_tid: AtomicU64::new(0),
        process: None,
        cpu: pc.index,
        kstack: 0,
        ksp: UnsafeCell::new(0),
        state: AtomicU8::new(State::Running as u8),
        killed: AtomicBool::new(false),
        xsave: 0,
    });
    // The idle thread runs on the CPU's boot stack, which is never freed: keep it out of `Drop`.
    core::mem::forget(idle.clone());
    let mut rq = pc.sched.rq.lock();
    rq.current = Some(idle.clone());
    rq.idle = Some(idle);
    rq.cr3 = x86_64::registers::control::Cr3::read().0.start_address().as_u64();
}

global_asm!(
    r#"
.global switch_stack
// switch_stack(save: *mut u64 (rdi), next: u64 (rsi))
switch_stack:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov [rdi], rsp
    mov rsp, rsi
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

// A new thread's first switch returns here, with rsp at its user frame.
.global thread_start
thread_start:
    call after_switch
    jmp isr_return
"#
);

unsafe extern "C" {
    fn switch_stack(save: *mut u64, next: u64);
    fn thread_start();
}

/// Creates a user thread of `process` on `cpu`, starting at `rip` with `rsp` and `arg` in rdi, and makes it
/// runnable.
pub fn spawn_user(process: Arc<Process>, cpu_index: u32, rip: u64, rsp: u64, arg: u64) -> Arc<Thread> {
    let mut regs = [0u64; 22];
    regs[9] = arg;
    regs[17] = rip;
    regs[20] = rsp;
    spawn_with(process, cpu_index, regs, 0)
}

/// Creates a user thread that starts with the registers `regs` (`interrupts::Frame` layout; segment selectors
/// and flags are set here) and FS base `fs_base`.
pub fn spawn_with(process: Arc<Process>, cpu_index: u32, mut regs: [u64; 22], fs_base: u64) -> Arc<Thread> {
    let kstack = mem::alloc_stack(KSTACK_PAGES);
    let xsave = mem::alloc_zeroed(1, 4096);
    // The state a new thread starts with: x87 and SSE in their initial state (XSTATE_BV 0), except that XRSTOR
    // always loads MXCSR from memory, so it must hold the default (all exceptions masked), not zero.
    // SAFETY: the fresh XSAVE page.
    unsafe {
        (xsave as *mut u16).write(0x037f);
        ((xsave + 24) as *mut u32).write(0x1f80);
    }
    let top = kstack + (KSTACK_PAGES * 4096) as u64;
    regs[15] = 0x100; // vector: not an exception
    regs[16] = 0;
    regs[18] = cpu::USER_CODE as u64;
    regs[19] = 0x202; // IF
    regs[21] = cpu::USER_DATA as u64;
    // The user frame, then switch_stack's six registers and its return address.
    let frame = top - (regs.len() * 8) as u64;
    // SAFETY: inside the fresh stack.
    unsafe {
        core::ptr::copy_nonoverlapping(regs.as_ptr(), frame as *mut u64, regs.len());
        let sw = (frame - 7 * 8) as *mut u64;
        sw.add(6).write(thread_start as *const () as u64);
    }
    let t = Arc::new(Thread {
        tid: NEXT_TID.fetch_add(1, Ordering::Relaxed),
        fs_base: AtomicU64::new(fs_base),
        clear_tid: AtomicU64::new(0),
        process: Some(process.clone()),
        cpu: cpu_index,
        kstack,
        ksp: UnsafeCell::new(frame - 7 * 8),
        state: AtomicU8::new(State::Blocked as u8),
        killed: AtomicBool::new(false),
        xsave,
    });
    process.add_thread(&t);
    wake(&t);
    t
}

static NEXT_TID: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(100);

/// Sets the current thread's FS base (Linux `arch_prctl(ARCH_SET_FS)`).
pub fn set_fs_base(v: u64) {
    current().fs_base.store(v, Ordering::Relaxed);
    // SAFETY: IA32_FS_BASE; user code reads it through FS.
    unsafe { cpu::wrmsr(0xc000_0100, v) };
}

/// The CPU for a new thread when the caller doesn't choose: round robin over the CPUs that are up.
pub fn pick_cpu() -> u32 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = crate::smp::online();
    (NEXT.fetch_add(1, Ordering::Relaxed) % n as u64) as u32
}

/// Makes `t` runnable (or keeps it running if it hasn't left its CPU yet).
pub fn wake(t: &Arc<Thread>) {
    let pc = cpu_of(t);
    let mut rq = pc.sched.rq.lock();
    match t.state() {
        State::Blocking => t.set(State::Woken),
        State::Blocked => {
            t.set(State::Runnable);
            rq.timers.retain(|(_, w)| !Arc::ptr_eq(w, t));
            rq.queue.push_front(t.clone());
            drop(rq);
            if t.cpu == cpu::this().index {
                pc.sched.need_resched.store(true, Ordering::Release);
            } else {
                apic::send_ipi(pc.apic_id, interrupts::RESCHED as u64);
            }
        }
        _ => {}
    }
}

/// Registers the current thread as about to block, with a deadline (TSC, `u64::MAX` for none). Then either
/// `schedule()` (blocks unless woken meanwhile) or `cancel_block()`.
pub fn prepare_block(me: &Arc<Thread>, deadline_tsc: u64) {
    let mut rq = cpu::this().sched.rq.lock();
    me.set(State::Blocking);
    if deadline_tsc != u64::MAX {
        rq.timers.push((deadline_tsc, me.clone()));
    }
}

/// Takes back a `prepare_block` that turned out not to be needed.
pub fn cancel_block(me: &Arc<Thread>) {
    let mut rq = cpu::this().sched.rq.lock();
    if matches!(me.state(), State::Blocking | State::Woken) {
        me.set(State::Running);
    }
    rq.timers.retain(|(_, w)| !Arc::ptr_eq(w, me));
}

/// Leaves the CPU for good.
pub fn exit_current() -> ! {
    let me = current();
    if let Some(p) = &me.process {
        p.thread_exited(&me);
    }
    me.set(State::Dead);
    drop(me);
    schedule();
    unreachable!("a dead thread was scheduled");
}

/// Called at the end of every interrupt and system call: reschedules if asked to.
pub fn preempt_point() {
    let pc = cpu::this();
    if pc.sched.need_resched.swap(false, Ordering::AcqRel) {
        schedule();
    }
}

pub fn request_resched() {
    cpu::this().sched.need_resched.store(true, Ordering::Release);
}

/// The timer fired: wake expired sleepers; the time slice may be over.
pub fn timer_interrupt() {
    let pc = cpu::this();
    let now = cpu::rdtsc();
    let mut expired = Vec::new();
    {
        let mut rq = pc.sched.rq.lock();
        rq.timers.retain(|(d, t)| {
            if *d <= now {
                expired.push(t.clone());
                false
            } else {
                true
            }
        });
    }
    for t in expired {
        wake(&t);
    }
    pc.sched.need_resched.store(true, Ordering::Release);
}

/// Picks the next thread on this CPU and switches to it.
pub fn schedule() {
    let pc: &'static PerCpu = cpu::this();
    let mut rq = pc.sched.rq.lock();
    let cur = rq.current.clone().expect("no current thread");
    let is_idle = cur.process.is_none();
    match cur.state() {
        State::Running if !is_idle => {
            cur.set(State::Runnable);
            rq.queue.push_back(cur.clone());
        }
        State::Running => {}
        State::Blocking => cur.set(State::Blocked),
        State::Woken => {
            cur.set(State::Running);
            arm_timer(&mut rq);
            return;
        }
        State::Dead => {}
        s => panic!("current thread in state {s:?}"),
    }
    let next = rq.queue.pop_front().unwrap_or_else(|| rq.idle.clone().unwrap());
    next.set(State::Running);
    if Arc::ptr_eq(&next, &cur) {
        arm_timer(&mut rq);
        return;
    }
    rq.slice_end = cpu::rdtsc() + time::ns_to_tsc(SLICE_NS);
    arm_timer(&mut rq);
    if cur.state() == State::Dead {
        rq.reap = Some(cur.clone());
    }
    rq.current = Some(next.clone());

    // Address space, kernel stack, ports, extended state.
    let cr3 = match &next.process {
        Some(p) => p.aspace.pml4(),
        None => crate::paging::kernel_pml4(),
    };
    if cr3 != rq.cr3 {
        rq.cr3 = cr3;
        if let Some(p) = &next.process {
            p.aspace.loaded_on(pc.index);
        }
        // SAFETY: a PML4 with the kernel half shared.
        unsafe { core::arch::asm!("mov cr3, {}", in(reg) cr3) };
    }
    if let Some(p) = &next.process {
        // SAFETY: IA32_FS_BASE, the next thread's TLS pointer.
        unsafe { cpu::wrmsr(0xc000_0100, next.fs_base.load(Ordering::Relaxed)) };
        let me = cpu::this();
        me.set_kernel_stack(next.kstack_top());
        let ports = p.ioports.lock();
        me.load_iomap(Arc::as_ptr(p) as u64, ports.as_deref());
    }
    let mask = cpu::xsave_mask();
    // SAFETY: the XSAVE areas are 64-byte aligned pages of their threads.
    unsafe {
        if cur.xsave != 0 && cur.state() != State::Dead {
            core::arch::asm!("xsave64 [{}]", in(reg) cur.xsave, in("eax") mask as u32, in("edx") (mask >> 32) as u32);
        }
        if next.xsave != 0 {
            core::arch::asm!("xrstor64 [{}]", in(reg) next.xsave, in("eax") mask as u32, in("edx") (mask >> 32) as u32);
        }
    }
    let save = cur.ksp.get();
    // SAFETY: next's ksp was saved when it left its CPU (this one) or prepared by spawn_user.
    let next_sp = unsafe { *next.ksp.get() };
    drop(next);
    drop(cur);
    MutexGuard::leak(rq);
    // SAFETY: the guard was leaked above; release the lock before leaving this stack. Interrupts are off, so
    // nothing on this CPU can take it in between; other CPUs only queue threads here.
    unsafe { pc.sched.rq.force_unlock() };
    // SAFETY: switches stacks; `save` stays valid (the thread is referenced from the run queue or reap slot).
    unsafe { switch_stack(save, next_sp) };
    after_switch();
}

/// Runs on the new thread right after a switch: frees a thread that exited.
#[unsafe(no_mangle)]
extern "C" fn after_switch() {
    let r = cpu::this().sched.rq.lock().reap.take();
    drop(r);
}

/// Programs this CPU's timer for the earliest of: the slice end (only when others wait), sleep deadlines.
fn arm_timer(rq: &mut RunQueue) {
    let mut at = rq.timers.iter().map(|(d, _)| *d).min().unwrap_or(u64::MAX);
    if !rq.queue.is_empty() {
        at = at.min(rq.slice_end);
    }
    if at == u64::MAX {
        apic::timer_disarm();
    } else {
        apic::timer_arm(at);
    }
}

/// The idle loop of a CPU: halt until an interrupt leaves something to run.
pub fn idle() -> ! {
    loop {
        schedule();
        x86_64::instructions::interrupts::enable_and_hlt();
        x86_64::instructions::interrupts::disable();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitError {
    TimedOut,
    Killed,
}

/// Object state and the threads waiting on it, under one lock: a waiter checks the state and registers
/// atomically, and every change through `update` wakes the waiters, so no wakeup is lost.
pub struct WaitCell<T> {
    inner: Mutex<(T, Vec<Arc<Thread>>)>,
}

impl<T> WaitCell<T> {
    pub const fn new(v: T) -> Self {
        WaitCell { inner: Mutex::new((v, Vec::new())) }
    }

    /// Reads or changes the state without waking anyone.
    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(&mut self.inner.lock().0)
    }

    /// Changes the state and wakes every waiter (they re-check).
    pub fn update<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut g = self.inner.lock();
        let r = f(&mut g.0);
        for t in g.1.drain(..) {
            wake(&t);
        }
        r
    }

    /// For waiting on several cells (poll): checks the state with `ready`; if it reports nothing (0), registers
    /// `me` to be woken by the next `update`. The caller has called `prepare_block` already.
    pub fn poll(&self, me: &Arc<Thread>, ready: impl FnOnce(&mut T) -> u16) -> u16 {
        let mut g = self.inner.lock();
        let r = ready(&mut g.0);
        if r == 0 {
            g.1.push(me.clone());
        }
        r
    }

    pub fn unregister(&self, me: &Arc<Thread>) {
        self.inner.lock().1.retain(|t| !Arc::ptr_eq(t, me));
    }

    /// Blocks until `f` returns `Some`, the deadline (TSC) passes or the thread is killed.
    pub fn wait<R>(&self, deadline_tsc: u64, mut f: impl FnMut(&mut T) -> Option<R>) -> Result<R, WaitError> {
        let me = current();
        loop {
            let mut g = self.inner.lock();
            if let Some(r) = f(&mut g.0) {
                return Ok(r);
            }
            if me.killed() {
                return Err(WaitError::Killed);
            }
            if cpu::rdtsc() >= deadline_tsc {
                return Err(WaitError::TimedOut);
            }
            g.1.push(me.clone());
            prepare_block(&me, deadline_tsc);
            drop(g);
            schedule();
            self.inner.lock().1.retain(|t| !Arc::ptr_eq(t, &me));
        }
    }
}

/// Sleeps `ns` nanoseconds (or until killed).
pub fn sleep_ns(ns: u64) -> Result<(), WaitError> {
    static NOTHING: WaitCell<()> = WaitCell::new(());
    let deadline = time::tsc_at_ns(time::now_ns().saturating_add(ns));
    match NOTHING.wait(deadline, |_| None::<()>) {
        Err(WaitError::TimedOut) => Ok(()),
        Err(e) => Err(e),
        Ok(()) => Ok(()),
    }
}

/// A TSC deadline from a deadline in ns of `time::now_ns` (`u64::MAX` stays forever).
pub fn deadline_from_ns(ns: u64) -> u64 {
    if ns == u64::MAX { u64::MAX } else { time::tsc_at_ns(ns) }
}
