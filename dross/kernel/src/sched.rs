//! Threads and the scheduler. Every thread is pinned to one CPU; each CPU has its own run queue, timers and
//! idle thread, so scheduling never touches another CPU's state except to wake a thread there (its run queue
//! lock, then a reschedule IPI). The kernel runs with interrupts off; a thread enters the scheduler when it
//! blocks, yields or exits, or at the end of an interrupt (timer, reschedule IPI) taken in user mode or idle.
//!
//! A CPU arms its timer only when something needs it: the end of the running thread's time slice when others
//! are waiting, or the earliest deadline of a thread sleeping there. An idle CPU with no deadlines takes no
//! timer interrupts.

use alloc::collections::VecDeque;
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use core::arch::global_asm;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use spin::{Mutex, MutexGuard};
use dross_abi::Error;

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

struct UserStack {
    process: alloc::sync::Weak<Process>,
    memory: Arc<crate::memory::Memory>,
    base: u64,
    len: u64,
}
impl Drop for UserStack {
    fn drop(&mut self) {
        if let Some(process) = self.process.upgrade() {
            process.aspace.release_stack(self.base, self.len, &self.memory);
        }
    }
}

pub struct Thread {
    /// A number for Linux programs (`gettid`).
    pub tid: u32,
    pub signals: Arc<crate::events::Signals>,
    pub exited: WaitCell<bool>,
    user_stack: Mutex<Option<UserStack>>,
    kernel_charge: Mutex<Option<crate::account::Charge>>,
    /// The user FS base (thread-local storage), loaded at every switch.
    pub fs_base: AtomicU64,
    /// Linux `CLONE_CHILD_CLEARTID`: zeroed and woken (futex) when the thread exits.
    pub clear_tid: AtomicU64,
    /// None for the idle threads.
    pub process: Option<Arc<Process>>,
    pub cpu: u32,
    kstack: AtomicU64,
    ksp: UnsafeCell<u64>,
    /// Changed only under the run queue lock of `cpu`.
    state: AtomicU8,
    killed: AtomicBool,
    /// XSAVE area (one page, 64-byte aligned), 0 for kernel threads.
    xsave: AtomicU64,
}

// SAFETY: `ksp` is only touched by the thread's own CPU in `schedule`, with interrupts off.
unsafe impl Sync for Thread {}
unsafe impl Send for Thread {}

impl Thread {
    pub fn kstack_top(&self) -> u64 {
        self.kstack.load(Ordering::Relaxed) + (KSTACK_PAGES * 4096) as u64
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

    /// Called only after this thread has stopped executing on its stack. Handles
    /// may retain the thread's identity without retaining its execution storage.
    fn release_execution_resources(&self) {
        let stack = self.kstack.swap(0, Ordering::Relaxed);
        if stack != 0 { mem::free_stack(stack, KSTACK_PAGES); }
        let xsave = self.xsave.swap(0, Ordering::Relaxed);
        if xsave != 0 { mem::free_virt(xsave, 1); }
        let stack = self.user_stack.lock().take();
        drop(stack);
        self.kernel_charge.lock().take();
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        self.release_execution_resources();
    }
}

struct ObjectTimer {
    owner: Weak<crate::timer::Timer>,
    deadline: u64,
    generation: u64,
}
struct RunQueue {
    current: Option<Arc<Thread>>,
    idle: Option<Arc<Thread>>,
    queue: VecDeque<Arc<Thread>>,
    /// Sleeping threads with a deadline (TSC), unordered: few at a time.
    timers: Vec<(u64, Arc<Thread>)>,
    object_timers: [Option<ObjectTimer>; dross_abi::TIMER_LIMIT],
    /// A thread that exited on this CPU, freed once the CPU is off its stack.
    reap: Option<Arc<Thread>>,
    /// The page table root loaded on this CPU.
    cr3: u64,
    slice_end: u64,
}

pub struct Cpu {
    rq: Mutex<RunQueue>,
    need_resched: AtomicBool,
    /// When the timer last asked for a reschedule (TSC; 0 once served), to report long non-preemptible stretches.
    resched_at: AtomicU64,
    /// The last system call entered on this CPU (for that report).
    pub last_syscall: AtomicU64,
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
                object_timers: [const { None }; dross_abi::TIMER_LIMIT],
                reap: None,
                cr3: 0,
                slice_end: 0,
            }),
            need_resched: AtomicBool::new(false),
            resched_at: AtomicU64::new(0),
            last_syscall: AtomicU64::new(0),
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

pub fn try_current_process() -> Option<Arc<Process>> {
    cpu::this().sched.rq.lock().current.as_ref().and_then(|t| t.process.clone())
}

#[cfg(feature = "audit-test")]
pub fn audit_test() {
    let process = Process::new("stopped-thread-publication").unwrap();
    process.kill(0);
    assert!(matches!(spawn_user(process, 0, 0, 0, 0), Err(Error::PeerClosed)));
    crate::println!("sched: stopped process rejects thread publication");
}

/// Turns the code running on this CPU (boot flow) into its idle thread.
pub fn init_cpu() {
    let pc = cpu::this();
    let idle = Arc::new(Thread {
        tid: 0,
        signals: crate::events::Signals::new(0),
        exited: WaitCell::new(false),
        user_stack: Mutex::new(None),
        kernel_charge: Mutex::new(None),
        fs_base: AtomicU64::new(0),
        clear_tid: AtomicU64::new(0),
        process: None,
        cpu: pc.index,
        kstack: AtomicU64::new(0),
        ksp: UnsafeCell::new(0),
        state: AtomicU8::new(State::Running as u8),
        killed: AtomicBool::new(false),
        xsave: AtomicU64::new(0),
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
pub fn spawn_user(process: Arc<Process>, cpu_index: u32, rip: u64, rsp: u64, arg: u64) -> Result<Arc<Thread>, Error> {
    let mut regs = [0u64; 22];
    regs[9] = arg;
    regs[17] = rip;
    regs[20] = rsp;
    spawn_with(process, cpu_index, regs, 0)
}

/// A stack with a thread lifetime. Existing caller-owned THREAD_CREATE is unchanged.
pub fn spawn_owned(process: Arc<Process>, cpu_index: u32, entry: u64, arg: u64, size: u64) -> Result<Arc<Thread>, Error> {
    if cpu::get(cpu_index as usize).is_none() || size < 4096 || size > 16 << 20 || size % 4096 != 0 { return Err(Error::InvalidArgs); }
    let memory = Arc::new(crate::memory::Memory::anonymous_in(size, false, false, process.account.clone())?);
    let base = process.aspace.map(memory.clone(), 0, size, 0, dross_abi::vm::READ | dross_abi::vm::WRITE)?;
    let stack = UserStack { process: Arc::downgrade(&process), memory, base, len: size };
    let mut regs = [0u64; 22]; regs[9] = arg; regs[17] = entry; regs[20] = base + size - 8;
    spawn_with_stack(process, cpu_index, regs, 0, Some(stack))
}

/// Creates a user thread that starts with the registers `regs` (`interrupts::Frame` layout; segment selectors
/// and flags are set here) and FS base `fs_base`.
pub fn spawn_with(process: Arc<Process>, cpu_index: u32, regs: [u64; 22], fs_base: u64) -> Result<Arc<Thread>, Error> {
    spawn_with_stack(process, cpu_index, regs, fs_base, None)
}
fn spawn_with_stack(process: Arc<Process>, cpu_index: u32, mut regs: [u64; 22], fs_base: u64, user_stack: Option<UserStack>) -> Result<Arc<Thread>, Error> {
    if cpu::get(cpu_index as usize).is_none() { return Err(Error::InvalidArgs); }
    let kernel_charge = process.account.reserve(((KSTACK_PAGES + 1) * 4096) as u64)?;
    let kstack = mem::try_alloc_stack(KSTACK_PAGES)?;
    let xsave = match mem::try_alloc_zeroed(1, 4096) {
        Ok(xsave) => xsave,
        Err(e) => { mem::free_stack(kstack, KSTACK_PAGES); return Err(e); }
    };
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
        signals: crate::events::Signals::new(0),
        exited: WaitCell::new(false),
        user_stack: Mutex::new(user_stack),
        kernel_charge: Mutex::new(Some(kernel_charge)),
        fs_base: AtomicU64::new(fs_base),
        clear_tid: AtomicU64::new(0),
        process: Some(process.clone()),
        cpu: cpu_index,
        kstack: AtomicU64::new(kstack),
        ksp: UnsafeCell::new(frame - 7 * 8),
        state: AtomicU8::new(State::Blocked as u8),
        killed: AtomicBool::new(false),
        xsave: AtomicU64::new(xsave),
    });
    // Once published, kill() may wake it on another CPU before our wake below.
    // Its exit must already have a matching scheduler count to remove.
    let count = THREADS[cpu_index as usize].fetch_add(1, Ordering::Relaxed) as usize + 1;
    // Reserve scheduling/wait storage before publication. IRQ wakeups and timed
    // blocking then use at most one slot per live thread without allocation.
    let reserved = {
        let mut rq = cpu::get(cpu_index as usize).unwrap().sched.rq.lock();
        let queue_extra = (count + 1).saturating_sub(rq.queue.len());
        let timer_extra = (count + 1).saturating_sub(rq.timers.len());
        rq.queue.try_reserve(queue_extra).and_then(|_| rq.timers.try_reserve(timer_extra))
    };
    if reserved.is_err() {
        THREADS[cpu_index as usize].fetch_sub(1, Ordering::Relaxed);
        return Err(Error::NoMemory);
    }
    if let Err(e) = process.add_thread(&t) {
        THREADS[cpu_index as usize].fetch_sub(1, Ordering::Relaxed);
        return Err(e);
    }
    wake(&t);
    Ok(t)
}

/// Called before a same-CPU fork child can run: inherit the live XSAVE state.
#[cfg(feature = "unix")]
pub fn copy_fpu_to(child: &Thread) {
    assert_eq!(child.cpu, cpu::index());
    let mask = cpu::xsave_mask();
    // SAFETY: the child is queued on this CPU and cannot run while the kernel has
    // interrupts disabled. Its XSAVE page is aligned and not used concurrently.
    unsafe { core::arch::asm!("xsave64 [{}]", in(reg) child.xsave.load(Ordering::Relaxed), in("eax") mask as u32, in("edx") (mask >> 32) as u32); }
}

static NEXT_TID: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(100);

/// Sets the current thread's FS base (Linux `arch_prctl(ARCH_SET_FS)`).
pub fn set_fs_base(v: u64) {
    current().fs_base.store(v, Ordering::Relaxed);
    // SAFETY: IA32_FS_BASE; user code reads it through FS.
    unsafe { cpu::wrmsr(0xc000_0100, v) };
}

/// User threads living on each CPU.
static THREADS: [core::sync::atomic::AtomicU32; cpu::MAX_CPUS] = [const { core::sync::atomic::AtomicU32::new(0) }; cpu::MAX_CPUS];

/// The CPU for a thread whose creator doesn't choose: the one with the fewest threads, the highest-numbered on a
/// tie (CPU 0 takes the boot work and most interrupts). Threads never move, so this decides for their lifetime.
pub fn pick_cpu() -> u32 {
    let n = crate::smp::online() as usize;
    (0..n).rev().min_by_key(|&c| THREADS[c].load(Ordering::Relaxed)).unwrap_or(0) as u32
}

/// Makes `t` runnable (or keeps it running if it hasn't left its CPU yet).
pub fn wake(t: &Arc<Thread>) {
    let pc = cpu_of(t);
    let mut rq = pc.sched.rq.lock();
    match t.state() {
        State::Blocking => {
            t.set(State::Woken);
            rq.timers.retain(|(_, w)| !Arc::ptr_eq(w, t));
        },
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
        THREADS[me.cpu as usize].fetch_sub(1, Ordering::Relaxed);
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
        let asked = pc.sched.resched_at.swap(0, Ordering::Relaxed);
        if asked != 0 && time::ms_since(asked) > 5.0 {
            let me = current();
            let name = me.process.as_ref().map_or("idle", |p| p.name.as_str());
            crate::println!("sched: cpu{} rescheduled {:.1} ms after its timer ({}, last system call {})",
                pc.index, time::ms_since(asked), name, pc.sched.last_syscall.load(Ordering::Relaxed));
        }
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
    // Bound each interrupt and allocate no temporary expiry list. Remaining due
    // work keeps the APIC deadline armed for the next interrupt.
    for _ in 0..32 {
        let due = {
            let mut rq = pc.sched.rq.lock();
            let Some(entry) = rq.object_timers.iter_mut().flatten().find(|e| e.deadline != u64::MAX && e.deadline <= now) else { break };
            let deadline = core::mem::replace(&mut entry.deadline, u64::MAX);
            entry.owner.upgrade().map(|timer| (timer, entry.generation, deadline))
        };
        if let Some((timer, generation, deadline)) = due { timer.expire(generation, deadline, now); }
    }
    for _ in 0..64 {
        let due = {
            let mut rq = pc.sched.rq.lock();
            let Some(index) = rq.timers.iter().position(|(deadline, _)| *deadline <= now) else { break };
            rq.timers.swap_remove(index).1
        };
        wake(&due);
    }
    let _ = pc.sched.resched_at.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
    pc.sched.need_resched.store(true, Ordering::Release);
}

/// Picks the next thread on this CPU and switches to it.
pub fn schedule() {
    let pc: &'static PerCpu = cpu::this();
    pc.sched.resched_at.store(0, Ordering::Relaxed);
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
        let cur_xsave = cur.xsave.load(Ordering::Relaxed);
        let next_xsave = next.xsave.load(Ordering::Relaxed);
        if cur_xsave != 0 && cur.state() != State::Dead {
            core::arch::asm!("xsave64 [{}]", in(reg) cur_xsave, in("eax") mask as u32, in("edx") (mask >> 32) as u32);
        }
        if next_xsave != 0 {
            core::arch::asm!("xrstor64 [{}]", in(reg) next_xsave, in("eax") mask as u32, in("edx") (mask >> 32) as u32);
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
    if let Some(t) = &r {
        t.release_execution_resources();
        t.exited.update(|exited| { *exited = true; t.signals.set(dross_abi::event::EXITED, dross_abi::event::EXITED); });
        if let Some(p) = &t.process { p.thread_stopped(); }
    }
    drop(r);
    crate::ipc::reap_deferred();
    crate::linux::reap_socket_cycles();
    crate::heap::reap_deferred();
}

/// Programs this CPU's timer for the earliest of: the slice end (only when others wait), sleep deadlines.
fn arm_timer(rq: &mut RunQueue) {
    let mut at = rq.timers.iter().map(|(d, _)| *d).chain(rq.object_timers.iter().flatten().map(|e| e.deadline)).min().unwrap_or(u64::MAX);
    if !rq.queue.is_empty() {
        at = at.min(rq.slice_end);
    }
    if at == u64::MAX {
        apic::timer_disarm();
    } else {
        apic::timer_arm(at);
    }
}

/// Timer slots are reserved once. Arm/reset/expiry never allocate entries.
pub fn register_timer(timer: &Arc<crate::timer::Timer>) -> Result<usize, Error> {
    let mut rq = cpu::this().sched.rq.lock();
    let slot = rq.object_timers.iter_mut().position(|entry| entry.as_ref().is_none_or(|e| e.owner.strong_count() == 0)).ok_or(Error::NoResources)?;
    rq.object_timers[slot] = Some(ObjectTimer { owner: Arc::downgrade(timer), deadline: u64::MAX, generation: 0 });
    Ok(slot)
}
pub fn set_timer(timer: &crate::timer::Timer, generation: u64, deadline: u64) {
    let pc = cpu::get(timer.cpu as usize).unwrap();
    {
        let mut rq = pc.sched.rq.lock();
        let slot = timer.slot.load(Ordering::Relaxed);
        let entry = rq.object_timers[slot].as_mut().unwrap();
        assert_eq!(entry.owner.as_ptr(), timer as *const _);
        entry.generation = generation;
        entry.deadline = deadline;
        if pc.index == cpu::index() { arm_timer(&mut rq); }
    }
    if pc.index != cpu::index() { apic::send_ipi(pc.apic_id, interrupts::RESCHED as u64); }
}
pub fn unregister_timer(timer: &crate::timer::Timer) {
    let slot = timer.slot.load(Ordering::Relaxed);
    if slot == usize::MAX { return; }
    let pc = cpu::get(timer.cpu as usize).unwrap();
    {
        let mut rq = pc.sched.rq.lock();
        if rq.object_timers[slot].as_ref().is_some_and(|e| core::ptr::eq(e.owner.as_ptr(), timer)) {
            rq.object_timers[slot] = None;
        }
        if pc.index == cpu::index() { arm_timer(&mut rq); }
    }
    if pc.index != cpu::index() { apic::send_ipi(pc.apic_id, interrupts::RESCHED as u64); }
}

/// The idle loop of a CPU: halt until an interrupt leaves something to run.
pub fn idle() -> ! {
    loop {
        schedule();
        // Post-switch reclamation can publish process exit and wake a waiter on
        // this CPU after schedule selected idle. Run it before halting.
        crate::ipc::reap_deferred();
        crate::linux::reap_socket_cycles();
        crate::heap::reap_deferred();
        if crate::ipc::has_deferred() || crate::linux::socket_cycles_deferred() || crate::heap::has_retired() || !cpu::this().sched.rq.lock().queue.is_empty() { continue; }
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
    pub fn wait<R>(&self, deadline_tsc: u64, f: impl FnMut(&mut T) -> Option<R>) -> Result<R, WaitError> {
        self.wait_inner(deadline_tsc, f, true, None)
    }

    /// Kernel transactions defer ordinary signals until their reply is consumed.
    /// Process death and the deadline still interrupt the wait.
    #[cfg(feature = "unix")]
    pub fn wait_uninterruptible<R>(&self, deadline_tsc: u64, f: impl FnMut(&mut T) -> Option<R>) -> Result<R, WaitError> {
        self.wait_inner(deadline_tsc, f, false, None)
    }

    /// Serialize a state transaction with an external ownership graph, but
    /// release that lock before sleeping. State check and waiter registration
    /// remain atomic with writers.
    pub fn wait_serialized<R>(&self, deadline_tsc: u64, interruptible: bool, serial: &Mutex<()>, f: impl FnMut(&mut T) -> Option<R>) -> Result<R, WaitError> {
        self.wait_inner(deadline_tsc, f, interruptible, Some(serial))
    }

    fn wait_inner<R>(&self, deadline_tsc: u64, mut f: impl FnMut(&mut T) -> Option<R>, _interruptible: bool, serial: Option<&Mutex<()>>) -> Result<R, WaitError> {
        let me = current();
        loop {
            let serial = serial.map(Mutex::lock);
            let mut g = self.inner.lock();
            if let Some(r) = f(&mut g.0) {
                return Ok(r);
            }
            #[cfg(feature = "unix")]
            if _interruptible && me.process.as_ref().is_some_and(|p| p.linux.is_some() && crate::linux::unix::interrupted(p)) { return Err(WaitError::Killed); }
            if me.killed() {
                return Err(WaitError::Killed);
            }
            if cpu::rdtsc() >= deadline_tsc {
                return Err(WaitError::TimedOut);
            }
            g.1.push(me.clone());
            prepare_block(&me, deadline_tsc);
            drop(g);
            drop(serial);
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
