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

const KSTACK_PAGES: usize = 4;
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
        mem::free_virt(self.kstack, KSTACK_PAGES);
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
    let kstack = mem::alloc_zeroed(KSTACK_PAGES, 4096);
    let xsave = mem::alloc_zeroed(1, 4096);
    let top = kstack + (KSTACK_PAGES * 4096) as u64;
    // The user frame (interrupts::Frame layout), then switch_stack's six registers and its return address.
    let frame_words: usize = 22;
    let frame = top - (frame_words * 8) as u64;
    // SAFETY: inside the fresh stack.
    unsafe {
        let f = frame as *mut u64;
        f.add(9).write(arg); // rdi
        f.add(15).write(0x100); // vector: not an exception
        f.add(17).write(rip);
        f.add(18).write(cpu::USER_CODE as u64);
        f.add(19).write(0x202); // IF
        f.add(20).write(rsp);
        f.add(21).write(cpu::USER_DATA as u64);
        let sw = (frame - 7 * 8) as *mut u64;
        sw.add(6).write(thread_start as *const () as u64);
    }
    let t = Arc::new(Thread {
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

/// Registers the current thread as about to block, with a deadline (TSC, `u64::MAX` for none).
fn prepare_block(me: &Arc<Thread>, deadline_tsc: u64) {
    let mut rq = cpu::this().sched.rq.lock();
    me.set(State::Blocking);
    if deadline_tsc != u64::MAX {
        rq.timers.push((deadline_tsc, me.clone()));
    }
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

/// A TSC deadline from a deadline in ns of `time::now_ns` (`u64::MAX` stays forever).
pub fn deadline_from_ns(ns: u64) -> u64 {
    if ns == u64::MAX { u64::MAX } else { time::tsc_at_ns(ns) }
}
