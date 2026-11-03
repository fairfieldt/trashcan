//! Per-CPU state (GDT, TSS with interrupt stacks, the CPU's index in GS base), the IDT, CPUID and MSRs.

use core::arch::x86_64::__cpuid_count;
use core::fmt;
use core::sync::atomic::{AtomicU32, Ordering};
use x86_64::instructions::segmentation::{Segment, CS, DS, ES, SS};
use x86_64::instructions::tables::{lidt, load_tss};
use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable};
use x86_64::structures::tss::TaskStateSegment;
use x86_64::structures::DescriptorTablePointer;
use x86_64::VirtAddr;

pub const MAX_CPUS: usize = 64;

pub const KERNEL_CODE: u16 = 0x08;

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

/// One per CPU, in frames of its own.
#[repr(C, align(64))]
pub struct PerCpu {
    pub index: u32,
    pub apic_id: u32,
    tss: TaskStateSegment,
    gdt: GlobalDescriptorTable,
}

/// The APIC ID of each CPU index, for IPIs; `u32::MAX` when the slot is unused.
pub static APIC_IDS: [AtomicU32; MAX_CPUS] = [const { AtomicU32::new(u32::MAX) }; MAX_CPUS];

const IA32_GS_BASE: u32 = 0xc000_0101;

/// The index of the CPU this runs on (0 before `init_cpu` ran on it).
pub fn index() -> u32 {
    let p = rdmsr(IA32_GS_BASE) as *const PerCpu;
    if p.is_null() {
        return 0;
    }
    // SAFETY: GS base is only ever set to this CPU's PerCpu.
    unsafe { (*p).index }
}

#[repr(C, align(16))]
struct Idt([u64; 512]);

static mut IDT: Idt = Idt([0; 512]);

unsafe extern "C" {
    static ISR_TABLE: [u64; 256];
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

/// GDT, TSS, IDT and GS base for the CPU this runs on.
pub fn init_cpu(index: u32, apic_id: u32) {
    let pages = size_of::<PerCpu>().div_ceil(4096);
    let pc = crate::mem::alloc_zeroed(pages, 4096) as *mut PerCpu;
    // SAFETY: a fresh allocation of the right size, never freed.
    let pc: &'static mut PerCpu = unsafe {
        pc.write(PerCpu { index, apic_id, tss: TaskStateSegment::new(), gdt: GlobalDescriptorTable::new() });
        &mut *pc
    };
    for slot in [IST_DOUBLE_FAULT, IST_NMI, IST_MACHINE_CHECK] {
        let stack = crate::mem::alloc_zeroed(IST_PAGES, 4096);
        pc.tss.interrupt_stack_table[slot as usize - 1] = VirtAddr::new(stack + (IST_PAGES * 4096) as u64);
    }
    // SAFETY: tss lives as long as the CPU (never freed).
    let tss: &'static TaskStateSegment = unsafe { &*core::ptr::addr_of!(pc.tss) };
    let code = pc.gdt.append(Descriptor::kernel_code_segment());
    let data = pc.gdt.append(Descriptor::kernel_data_segment());
    pc.gdt.append(Descriptor::user_data_segment());
    pc.gdt.append(Descriptor::user_code_segment());
    let tss_sel = pc.gdt.append(Descriptor::tss_segment(tss));
    assert_eq!(code.0, KERNEL_CODE);
    // SAFETY: the GDT is static for this CPU's lifetime; the selectors are from it.
    unsafe {
        pc.gdt.load_unsafe();
        CS::set_reg(code);
        SS::set_reg(data);
        DS::set_reg(data);
        ES::set_reg(data);
        load_tss(tss_sel);
        wrmsr(IA32_GS_BASE, pc as *const PerCpu as u64);
    }
    load_idt();
    APIC_IDS[index as usize].store(apic_id, Ordering::Release);
}

pub fn init_bsp() {
    build_idt();
    let apic_id = cpuid(1, 0).1 >> 24;
    init_cpu(0, apic_id);
    crate::println!("cpu0: gdt, tss, idt loaded");
}
