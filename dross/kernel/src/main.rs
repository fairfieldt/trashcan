//! dross, the trashcan microkernel (x86-64, Mac Pro 6,1). For now: bring-up of one machine from the UEFI loader's
//! handover to all CPUs running with timers and IPIs; the plan is `dross/README.md`.
#![no_std]
#![no_main]

extern crate alloc;

mod apic;
mod cpu;
mod fbcon;
mod firmware;
mod heap;
mod interrupts;
mod iommu;
mod ipc;
mod irq;
mod linux;
mod log;
mod mem;
mod memory;
mod object;
mod paging;
mod process;
mod qemu;
mod resource;
mod sched;
mod smp;
mod syscall;
mod time;

use alloc::sync::Arc;
use alloc::vec::Vec;
use bootinfo::BootInfo;

use crate::ipc::{ChannelEnd, Message};
use crate::memory::Memory;
use crate::object::{Entry, Object};
use crate::resource::{Kind, Resource};

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
    heap::init();
    mem::record_ram(info);
    cpu::init_bsp();
    mem::reclaim_boot_services(info);
    let acpi = firmware::Acpi::find(info);
    time::init(&acpi);
    apic::init(&acpi);
    sched::init_cpu();
    println!("apic: cpu 0: 1 ms timer took {:?} us", apic::timer_selftest());
    let cpus = smp::start_aps(info, &acpi);
    smp::ipi_check();
    paging::init_kernel_space();
    iommu::init(&acpi);
    firmware::power_on_after_power_loss(&acpi);

    println!(
        "boot ok: {} cpus, {} MiB free, tsc {}.{:03} MHz, {:.1} ms since loader",
        cpus,
        mem::free_bytes() >> 20,
        time::tsc_hz() / 1_000_000,
        time::tsc_hz() / 1_000 % 1_000,
        time::ms_since(info.tsc_at_entry)
    );
    // Linux programs' files: the rootfs module and the firmware tables.
    if let Some(m) = info.modules().iter().find(|m| m.name() == "rootfs.tar") {
        // SAFETY: the loader's module pages, which are never freed.
        linux::init_rootfs(unsafe { core::slice::from_raw_parts(mem::virt(m.phys) as *const u8, m.size as usize) });
    }
    for t in [&Some(acpi.madt), &acpi.hpet, &acpi.mcfg, &acpi.dmar, &acpi.fadt, &acpi.dsdt, &acpi.vfct].into_iter().flatten() {
        linux::publish_table(core::str::from_utf8(&t.header.signature).unwrap_or("?"), t.bytes);
    }
    start_init(info, &acpi);
    sched::idle()
}

/// Starts the `init` module with a bootstrap message: one line per handle (`@kind/name`) in handle order,
/// then `key value` lines.
fn start_init(info: &BootInfo, acpi: &firmware::Acpi) {
    let Some(m) = info.modules().iter().find(|m| m.name() == "init") else {
        println!("no init module: nothing to run");
        qemu::exit(false);
    };
    let image = Memory::module(m.phys, m.size);
    let (mine, theirs) = ChannelEnd::pair();
    let mut text = alloc::string::String::new();
    let mut handles = Vec::new();
    let mut add = |name: &str, o: Object, text: &mut alloc::string::String| {
        text.push('@');
        text.push_str(name);
        text.push('\n');
        handles.push(Entry::new(o));
    };
    for m in info.modules().iter().filter(|m| m.name() != "init" && m.name() != "rootfs.tar") {
        add(&alloc::format!("module/{}", m.name()), Object::Memory(Arc::new(Memory::module(m.phys, m.size))), &mut text);
    }
    for (name, kind, size) in [
        ("physical", Kind::Physical, 1u64 << 52),
        ("ioport", Kind::IoPort, 0x10000),
        ("irq", Kind::Irq, 1),
        ("iommu", Kind::Iommu, 1),
        ("system", Kind::System, 1),
    ] {
        add(&alloc::format!("resource/{name}"), Object::Resource(Arc::new(Resource::root(kind, size))), &mut text);
    }
    text.push_str(&alloc::format!("rsdp {:#x}\ncpus {}\nhypervisor {}\n", info.rsdp, smp::online(), cpu::Features::detect().hypervisor as u8));
    if let Some(r) = acpi.mcfg.as_ref().and_then(|t| ::acpi::mcfg::ranges(t).ok()).and_then(|mut r| r.next()) {
        text.push_str(&alloc::format!("ecam {:#x}\necam-buses {} {}\n", r.base, r.start_bus, r.end_bus));
    }
    if let Some(v) = &acpi.vfct {
        text.push_str(&alloc::format!("vfct {:#x} {}\n", v.bytes.as_ptr() as u64 - bootinfo::DIRECT_MAP, v.bytes.len()));
    }
    if let Err((e, _)) = mine.write(Message { bytes: text.into_bytes(), handles }) {
        panic!("init bootstrap: {e:?}");
    }
    match process::spawn("init", &image, 0, Some(Entry::new(Object::Channel(Arc::new(theirs))))) {
        Ok(_) => println!("init started"),
        Err(e) => panic!("init: {e:?}"),
    }
    // The kernel's end stays open so init can read the message.
    core::mem::forget(mine);
}

#[panic_handler]
fn panic(p: &core::panic::PanicInfo) -> ! {
    x86_64::instructions::interrupts::disable();
    log::emergency(format_args!("panic on cpu {}: {}", cpu::index(), p));
    qemu::exit(false);
}
