//! Per-CPU state (GDT, TSS with interrupt stacks and the I/O permission bitmap, scheduler state), reached
//! through GS in the kernel; the IDT; CPUID, MSRs and the extended (FPU/SSE/AVX) state.

use core::arch::x86_64::__cpuid_count;
use core::fmt;
use core::sync::atomic::{AtomicU32, Ordering};
use x86_64::instructions::segmentation::{Segment, CS, DS, ES, SS};
use x86_64::instructions::tables::{lidt, load_tss};
use x86_64::registers::control::{Cr4, Cr4Flags};
use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable};
use x86_64::structures::tss::TaskStateSegment;
use x86_64::structures::DescriptorTablePointer;
use x86_64::VirtAddr;

pub const MAX_CPUS: usize = 64;

pub const KERNEL_CODE: u16 = 0x08;
pub const KERNEL_DATA: u16 = 0x10;
pub const USER_DATA: u16 = 0x18 | 3;
pub const USER_CODE: u16 = 0x20 | 3;

/// Interrupt stack table slots (1-based in the IDT, 0-based in the TSS).
const IST_DOUBLE_FAULT: u8 = 1;
const IST_NMI: u8 = 2;
const IST_MACHINE_CHECK: u8 = 3;
const IST_PAGES: usize = 4;

pub fn cpuid(leaf: u32, sub: u32) -> (u32, u32, u32, u32) {
    let r = __cpuid_count(leaf, sub);
    (r.eax, r.ebx, r.ecx, r.edx)
}

pub fn rdmsr(msr: u32) -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: callers name MSRs that exist on the CPUs `Features::require` accepts.
    unsafe { core::arch::asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi, options(nomem, nostack)) };
    (hi as u64) << 32 | lo as u64
}

/// # Safety
/// The MSR must exist and the value must be valid for it.
pub unsafe fn wrmsr(msr: u32, v: u64) {
    unsafe { core::arch::asm!("wrmsr", in("ecx") msr, in("eax") v as u32, in("edx") (v >> 32) as u32, options(nostack)) };
}

pub fn rdtsc() -> u64 {
    // SAFETY: always available (checked by `require`).
    unsafe { core::arch::x86_64::_rdtsc() }
}

pub fn halt_forever() -> ! {
    loop {
        x86_64::instructions::interrupts::disable();
        x86_64::instructions::hlt();
    }
}

#[derive(Clone, Copy)]
pub struct Features {
    pub vendor: [u8; 12],
    pub family: u32,
    pub model: u32,
    pub x2apic: bool,
    pub tsc_deadline: bool,
    pub invariant_tsc: bool,
    pub xsave: bool,
    pub avx: bool,
    pub pdpe1gb: bool,
    pub nx: bool,
    pub hypervisor: bool,
}

impl Features {
    pub fn detect() -> Self {
        let (max, b, c, d) = cpuid(0, 0);
        let mut vendor = [0; 12];
        vendor[0..4].copy_from_slice(&b.to_le_bytes());
        vendor[4..8].copy_from_slice(&d.to_le_bytes());
        vendor[8..12].copy_from_slice(&c.to_le_bytes());
        let (a1, _, c1, _) = if max >= 1 { cpuid(1, 0) } else { (0, 0, 0, 0) };
        let family = (a1 >> 8 & 0xf) + if a1 >> 8 & 0xf == 0xf { a1 >> 20 & 0xff } else { 0 };
        let model = (a1 >> 4 & 0xf) | if family >= 6 { (a1 >> 16 & 0xf) << 4 } else { 0 };
        let (ext_max, ..) = cpuid(0x8000_0000, 0);
        let (_, _, _, d81) = if ext_max >= 0x8000_0001 { cpuid(0x8000_0001, 0) } else { (0, 0, 0, 0) };
        let (_, _, _, d87) = if ext_max >= 0x8000_0007 { cpuid(0x8000_0007, 0) } else { (0, 0, 0, 0) };
        Features {
            vendor,
            family,
            model,
            x2apic: c1 & 1 << 21 != 0,
            tsc_deadline: c1 & 1 << 24 != 0,
            xsave: c1 & 1 << 26 != 0,
            avx: c1 & 1 << 28 != 0,
            hypervisor: c1 & 1 << 31 != 0,
            pdpe1gb: d81 & 1 << 26 != 0,
            nx: d81 & 1 << 20 != 0,
            invariant_tsc: d87 & 1 << 8 != 0,
        }
    }

    /// What the kernel can't run without (all present on the Xeon E5-2697 v2).
    pub fn require(&self) {
        assert!(self.x2apic, "the kernel needs x2APIC");
        assert!(self.nx, "the kernel needs NX");
        assert!(self.xsave, "the kernel needs XSAVE");
    }

    pub fn brand(&self) -> &str {
        core::str::from_utf8(&self.vendor).unwrap_or("?")
    }
}

impl fmt::Display for Features {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (on, name) in [
            (self.x2apic, "x2apic"),
            (self.tsc_deadline, "tsc-deadline"),
            (self.invariant_tsc, "invariant-tsc"),
            (self.xsave, "xsave"),
            (self.avx, "avx"),
            (self.pdpe1gb, "1g-pages"),
            (self.hypervisor, "hypervisor"),
        ] {
            write!(f, "{}{} ", if on { "+" } else { "-" }, name)?;
        }
        Ok(())
    }
}

/// One per CPU. The first fields are at fixed offsets for the entry code (`gs:[n]`).
#[repr(C, align(4096))]
pub struct PerCpu {
    /// gs:0
    pub self_ptr: u64,
    /// gs:8: top of the running thread's kernel stack (`syscall` switches to it).
    pub kstack_top: u64,
    /// gs:16: the user stack pointer while `syscall` entry switches stacks.
    pub user_rsp: u64,
    pub index: u32,
    pub apic_id: u32,
    pub sched: crate::sched::Cpu,
    gdt: GlobalDescriptorTable,
    tss: TaskStateSegment,
    /// The I/O permission bitmap, right after the TSS (one bit per port, set = denied, plus the
    /// all-ones byte the CPU reads past the end).
    iomap: [u8; 8193],
    /// The process whose ports `iomap` grants (0: none).
    iomap_owner: u64,
}

// The entry code reads these as gs:[8] and gs:[16].
const _: () = assert!(core::mem::offset_of!(PerCpu, kstack_top) == 8 && core::mem::offset_of!(PerCpu, user_rsp) == 16);

/// The APIC ID of each CPU index, for IPIs; `u32::MAX` when the slot is unused.
pub static APIC_IDS: [AtomicU32; MAX_CPUS] = [const { AtomicU32::new(u32::MAX) }; MAX_CPUS];
static PERCPUS: [core::sync::atomic::AtomicU64; MAX_CPUS] = [const { core::sync::atomic::AtomicU64::new(0) }; MAX_CPUS];

const IA32_GS_BASE: u32 = 0xc000_0101;
const IA32_KERNEL_GS_BASE: u32 = 0xc000_0102;
const IA32_EFER: u32 = 0xc000_0080;
const IA32_STAR: u32 = 0xc000_0081;
const IA32_LSTAR: u32 = 0xc000_0082;
const IA32_FMASK: u32 = 0xc000_0084;

/// This CPU's block. Only valid in the kernel (GS is the kernel's) after `init_cpu`.
pub fn this() -> &'static mut PerCpu {
    let p: u64;
    // SAFETY: gs:0 holds the block's own address.
    unsafe { core::arch::asm!("mov {}, gs:[0]", out(reg) p, options(nostack, readonly, preserves_flags)) };
    // SAFETY: the block lives as long as the CPU; each CPU only touches its own mutable fields with interrupts off.
    unsafe { &mut *(p as *mut PerCpu) }
}

pub fn get(index: usize) -> Option<&'static PerCpu> {
    let p = PERCPUS.get(index)?.load(Ordering::Acquire);
    // SAFETY: as in `this`; shared access to the fields other CPUs may read (locks inside `sched`).
    (p != 0).then(|| unsafe { &*(p as *const PerCpu) })
}

/// The index of the CPU this runs on (0 before `init_cpu` ran on it).
pub fn index() -> u32 {
    if rdmsr(IA32_GS_BASE) == 0 {
        return 0;
    }
    this().index
}

impl PerCpu {
    pub fn set_kernel_stack(&mut self, top: u64) {
        self.kstack_top = top;
        self.tss.privilege_stack_table[0] = VirtAddr::new(top);
    }

    /// Loads `owner`'s port bitmap (None: no ports) unless it is already loaded.
    pub fn load_iomap(&mut self, owner: u64, map: Option<&[u8; 8192]>) {
        let owner = if map.is_some() { owner } else { 0 };
        if self.iomap_owner == owner {
            return;
        }
        match map {
            Some(m) => self.iomap[..8192].copy_from_slice(m),
            None => self.iomap[..8192].fill(0xff),
        }
        self.iomap_owner = owner;
    }
}

#[repr(C, align(16))]
struct Idt([u64; 512]);

static mut IDT: Idt = Idt([0; 512]);

unsafe extern "C" {
    static ISR_TABLE: [u64; 256];
    fn syscall_entry();
}

fn build_idt() {
    // SAFETY: written once by the BSP before any CPU loads it.
    let idt = unsafe { &mut *core::ptr::addr_of_mut!(IDT) };
    for v in 0..256 {
        // SAFETY: ISR_TABLE is the stub table from build.rs.
        let handler = unsafe { ISR_TABLE[v] };
        let ist = match v {
            8 => IST_DOUBLE_FAULT,
            2 => IST_NMI,
            18 => IST_MACHINE_CHECK,
            _ => 0,
        } as u64;
        // A present, ring-0, 64-bit interrupt gate.
        idt.0[v * 2] = (handler & 0xffff) | (KERNEL_CODE as u64) << 16 | ist << 32 | 0x8e << 40 | (handler >> 16 & 0xffff) << 48;
        idt.0[v * 2 + 1] = handler >> 32;
    }
}

fn load_idt() {
    let ptr = DescriptorTablePointer { limit: (size_of::<Idt>() - 1) as u16, base: VirtAddr::new(core::ptr::addr_of!(IDT) as u64) };
    // SAFETY: IDT is a complete table of valid gates.
    unsafe { lidt(&ptr) };
}

/// Size of the XSAVE area for the state the kernel enables (x87, SSE, AVX).
pub fn xsave_size() -> usize {
    cpuid(0xd, 0).1 as usize
}

/// The components saved and restored on every switch between user threads.
pub fn xsave_mask() -> u64 {
    if Features::detect().avx { 0b111 } else { 0b11 }
}

/// Extended state for user code: SSE and, where present, AVX, saved with XSAVE.
fn init_fpu() {
    // SAFETY: enabling FXSR/XMM exceptions/XSAVE in CR4, then XCR0 with the components the CPU has.
    unsafe {
        Cr4::update(|f| *f |= Cr4Flags::OSFXSR | Cr4Flags::OSXMMEXCPT_ENABLE | Cr4Flags::OSXSAVE);
        let mask = xsave_mask();
        core::arch::asm!("xsetbv", in("ecx") 0, in("eax") mask as u32, in("edx") (mask >> 32) as u32);
        // CR0: monitor coprocessor on, emulation off, no lazy switching (TS stays clear).
        let mut cr0: u64;
        core::arch::asm!("mov {}, cr0", out(reg) cr0);
        cr0 = (cr0 | 1 << 1) & !(1 << 2 | 1 << 3);
        core::arch::asm!("mov cr0, {}", in(reg) cr0);
    }
}

/// `syscall`: kernel CS/SS from STAR[47:32]; user SS = STAR[63:48] + 8, CS = + 16 (the GDT order is
/// kernel code, kernel data, user data, user code). IF, TF, DF and AC are cleared on entry.
fn init_syscall() {
    // SAFETY: the architectural syscall MSRs.
    unsafe {
        wrmsr(IA32_EFER, rdmsr(IA32_EFER) | 1);
        wrmsr(IA32_STAR, (KERNEL_DATA as u64) << 48 | (KERNEL_CODE as u64) << 32);
        wrmsr(IA32_LSTAR, syscall_entry as *const () as u64);
        wrmsr(IA32_FMASK, 1 << 9 | 1 << 8 | 1 << 10 | 1 << 18);
    }
}

/// GDT, TSS, IDT, GS, extended state and `syscall` for the CPU this runs on.
pub fn init_cpu(index: u32, apic_id: u32) {
    let pages = size_of::<PerCpu>().div_ceil(4096);
    let pc = crate::mem::alloc_zeroed(pages, 4096) as *mut PerCpu;
    // SAFETY: a fresh, zeroed allocation of the right size, never freed. Zero is a valid start for every field
    // but the ones written here.
    let pc: &'static mut PerCpu = unsafe {
        core::ptr::addr_of_mut!((*pc).tss).write(TaskStateSegment::new());
        core::ptr::addr_of_mut!((*pc).gdt).write(GlobalDescriptorTable::new());
        core::ptr::addr_of_mut!((*pc).sched).write(crate::sched::Cpu::new());
        &mut *pc
    };
    pc.self_ptr = pc as *mut PerCpu as u64;
    pc.index = index;
    pc.apic_id = apic_id;
    pc.iomap.fill(0xff);
    pc.tss.iomap_base = size_of::<TaskStateSegment>() as u16;
    for slot in [IST_DOUBLE_FAULT, IST_NMI, IST_MACHINE_CHECK] {
        let stack = crate::mem::alloc_zeroed(IST_PAGES, 4096);
        pc.tss.interrupt_stack_table[slot as usize - 1] = VirtAddr::new(stack + (IST_PAGES * 4096) as u64);
    }
    // SAFETY: tss and iomap live as long as the CPU and are adjacent (repr(C)).
    let (tss, iomap): (&'static TaskStateSegment, &'static [u8]) = unsafe { (&*core::ptr::addr_of!(pc.tss), &*core::ptr::addr_of!(pc.iomap)) };
    let code = pc.gdt.append(Descriptor::kernel_code_segment());
    let data = pc.gdt.append(Descriptor::kernel_data_segment());
    pc.gdt.append(Descriptor::user_data_segment());
    pc.gdt.append(Descriptor::user_code_segment());
    let tss_sel = pc.gdt.append(Descriptor::tss_segment_with_iomap(tss, iomap).expect("iomap"));
    assert_eq!((code.0, data.0), (KERNEL_CODE, KERNEL_DATA));
    // SAFETY: the GDT is static for this CPU's lifetime; the selectors are from it.
    unsafe {
        pc.gdt.load_unsafe();
        CS::set_reg(code);
        SS::set_reg(data);
        DS::set_reg(data);
        ES::set_reg(data);
        load_tss(tss_sel);
        wrmsr(IA32_GS_BASE, pc as *const PerCpu as u64);
        wrmsr(IA32_KERNEL_GS_BASE, 0);
    }
    load_idt();
    init_fpu();
    init_syscall();
    PERCPUS[index as usize].store(pc as *const PerCpu as u64, Ordering::Release);
    APIC_IDS[index as usize].store(apic_id, Ordering::Release);
}

pub fn init_bsp() {
    build_idt();
    let apic_id = cpuid(1, 0).1 >> 24;
    init_cpu(0, apic_id);
    crate::println!("cpu0: gdt, tss, idt loaded; xsave area {} bytes", xsave_size());
}
