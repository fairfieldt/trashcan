//! dross, the trashcan microkernel (x86-64, Mac Pro 6,1). For now: bring-up of one machine from the UEFI loader's
//! handover to all CPUs running with timers and IPIs; the plan is `dross/README.md`.
#![no_std]
#![no_main]

mod apic;
mod cpu;
mod fbcon;
mod firmware;
mod interrupts;
mod log;
mod mem;
mod qemu;
mod smp;
mod time;

use bootinfo::BootInfo;

/// The loader calls this on its own stack, with interrupts off, on the loader's page tables.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.entry")]
pub extern "sysv64" fn kernel_entry(info: &'static BootInfo) -> ! {
    assert_eq!(info.magic, bootinfo::MAGIC, "bad BootInfo");
    log::init(&info.framebuffer);
    println!("dross {}", env!("CARGO_PKG_VERSION"));
    let features = cpu::Features::detect();
    println!("cpu: {} (family {:#x} model {:#x}), {}", features.brand(), features.family, features.model, features);
    features.require();

    mem::init(info);
    cpu::init_bsp();
    mem::reclaim_boot_services(info);
    let acpi = firmware::Acpi::find(info);
    time::init(&acpi);
    apic::init(&acpi);
    interrupts::enable();
    apic::timer_check();
    let cpus = smp::start_aps(info, &acpi);
    smp::ipi_check();

    println!(
        "boot ok: {} cpus, {} MiB free, tsc {}.{:03} MHz, {:.1} ms since loader",
        cpus,
        mem::free_bytes() >> 20,
        time::tsc_hz() / 1_000_000,
        time::tsc_hz() / 1_000 % 1_000,
        time::ms_since(info.tsc_at_entry)
    );
    qemu::exit(true);
}

#[panic_handler]
fn panic(p: &core::panic::PanicInfo) -> ! {
    x86_64::instructions::interrupts::disable();
    log::emergency(format_args!("panic on cpu {}: {}", cpu::index(), p));
    qemu::exit(false);
}
