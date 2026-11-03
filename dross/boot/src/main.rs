//! The UEFI loader: reads `\dross.elf` from the volume it was started from, loads it at its link address,
//! builds the page tables described in `bootinfo`, leaves boot services and jumps to the kernel.
//!
//! Everything handed over (kernel image, page tables, stack, `BootInfo`, the AP trampoline page) is allocated
//! as `bootinfo::EFI_KERNEL_TYPE`, so the memory map tells the kernel what it must keep.
#![no_std]
#![no_main]

extern crate alloc;

use bootinfo::{BootInfo, Framebuffer, Module, PixelFormat, Region, DIRECT_MAP, EFI_KERNEL_TYPE, MAX_MODULES, MAX_REGIONS};
use core::arch::asm;
use core::ptr;
use log::info;
use uefi::boot::{self, AllocateType, MemoryType};
use uefi::mem::memory_map::MemoryMap;
use uefi::prelude::*;
use uefi::proto::console::gop::{self, GraphicsOutput};
use uefi::table::cfg::ConfigTableEntry;

const PAGE: u64 = 4096;
const KERNEL_TYPE: MemoryType = MemoryType::custom(EFI_KERNEL_TYPE);
const STACK_PAGES: usize = 16;

const PRESENT: u64 = 1;
const WRITABLE: u64 = 1 << 1;
const HUGE: u64 = 1 << 7;
const NO_EXECUTE: u64 = 1 << 63;

/// Pages the kernel keeps, zeroed. `below`: highest address + 1 the allocation may reach (0: anywhere).
fn alloc_pages(count: usize, below: u64) -> u64 {
    let ty = if below == 0 { AllocateType::AnyPages } else { AllocateType::MaxAddress(below - 1) };
    let p = boot::allocate_pages(ty, KERNEL_TYPE, count).expect("out of memory").as_ptr();
    // SAFETY: freshly allocated, identity-mapped by the firmware.
    unsafe { ptr::write_bytes(p, 0, count * PAGE as usize) };
    p as u64
}

/// Page tables in pages below 4 GiB (the AP trampoline loads CR3 from 32-bit code).
struct Tables {
    pml4: u64,
}

impl Tables {
    fn table() -> u64 {
        alloc_pages(1, 1 << 32)
    }

    fn entry(table: u64, index: u64) -> *mut u64 {
        (table + index * 8) as *mut u64
    }

    /// The next level under `table[index]`, created if absent.
    fn next(table: u64, index: u64) -> u64 {
        let e = Self::entry(table, index);
        // SAFETY: page tables are identity-mapped pages this loader allocated.
        unsafe {
            if *e & PRESENT == 0 {
                *e = Self::table() | PRESENT | WRITABLE;
            }
            *e & 0x000f_ffff_ffff_f000
        }
    }

    fn map_4k(&self, virt: u64, phys: u64, flags: u64) {
        let pdpt = Self::next(self.pml4, (virt >> 39) & 511);
        let pd = Self::next(pdpt, (virt >> 30) & 511);
        let pt = Self::next(pd, (virt >> 21) & 511);
        // SAFETY: as in `next`.
        unsafe { *Self::entry(pt, (virt >> 12) & 511) = phys | flags | PRESENT };
    }
}

struct Elf<'a>(&'a [u8]);

struct Segment {
    vaddr: u64,
    offset: u64,
    filesz: u64,
    memsz: u64,
    flags: u32,
}

impl<'a> Elf<'a> {
    fn u16(&self, o: usize) -> u16 {
        u16::from_le_bytes(self.0[o..o + 2].try_into().unwrap())
    }
    fn u32(&self, o: usize) -> u32 {
        u32::from_le_bytes(self.0[o..o + 4].try_into().unwrap())
    }
    fn u64(&self, o: usize) -> u64 {
        u64::from_le_bytes(self.0[o..o + 8].try_into().unwrap())
    }

    fn check(&self) {
        assert!(self.0.len() > 64 && &self.0[..4] == b"\x7fELF", "dross.elf is not ELF");
        // 64-bit, little-endian, executable, x86-64.
        assert!(self.0[4] == 2 && self.0[5] == 1 && self.u16(16) == 2 && self.u16(18) == 0x3e, "dross.elf is not an x86-64 executable");
    }

    fn entry(&self) -> u64 {
        self.u64(24)
    }

    fn loads(&self) -> impl Iterator<Item = Segment> + '_ {
        let (phoff, phentsize, phnum) = (self.u64(32) as usize, self.u16(54) as usize, self.u16(56) as usize);
        (0..phnum).map(move |i| phoff + i * phentsize).filter(|&p| self.u32(p) == 1).map(move |p| Segment {
            flags: self.u32(p + 4),
            offset: self.u64(p + 8),
            vaddr: self.u64(p + 16),
            filesz: self.u64(p + 32),
            memsz: self.u64(p + 40),
        })
    }
}

#[entry]
fn main() -> Status {
    uefi::helpers::init().unwrap();
    info!("dross loader: firmware {} rev {:#x}", uefi::system::firmware_vendor(), uefi::system::firmware_revision());

    #[cfg(feature = "bundle")]
    let (image, modules) = load_bundle();
    #[cfg(not(feature = "bundle"))]
    let (image, modules) = {
        let mut fs = uefi::fs::FileSystem::new(boot::get_image_file_system(boot::image_handle()).expect("no boot volume"));
        let image = fs.read(cstr16!("\\dross.elf")).expect("cannot read \\dross.elf");
        (image, load_modules(&mut fs))
    };
    let elf = Elf(&image);
    elf.check();

    // The kernel image, one contiguous block from its lowest to highest page.
    let lo = elf.loads().map(|s| s.vaddr & !(PAGE - 1)).min().expect("no loadable segments");
    let hi = elf.loads().map(|s| (s.vaddr + s.memsz).next_multiple_of(PAGE)).max().unwrap();
    let kernel_pages = ((hi - lo) / PAGE) as usize;
    let kernel_phys = alloc_pages(kernel_pages, 0);
    for s in elf.loads() {
        // SAFETY: inside the block just allocated; the file range was bounds-checked by the slice.
        unsafe {
            let src = &image[s.offset as usize..(s.offset + s.filesz) as usize];
            ptr::copy_nonoverlapping(src.as_ptr(), (kernel_phys + s.vaddr - lo) as *mut u8, src.len());
        }
    }
    info!("kernel: {} KiB at {:#x}, linked at {:#x}, entry {:#x}", kernel_pages * 4, kernel_phys, lo, elf.entry());

    // Highest address in the memory map, MMIO included, rounded up to 1 GiB; at least 4 GiB so the local
    // APIC, I/O APICs, HPET and ECAM below 4 GiB are covered.
    let phys_limit = {
        let mm = boot::memory_map(MemoryType::LOADER_DATA).expect("no memory map");
        mm.entries().map(|d| d.phys_start + d.page_count * PAGE).max().unwrap_or(0).max(1 << 32).next_multiple_of(1 << 30)
    };

    let tables = Tables { pml4: Tables::table() };
    // Direct map with 2 MiB pages, shared by the identity half (PML4 slot 0) and the direct map (slot 256).
    for addr in (0..phys_limit).step_by(1 << 21) {
        let pdpt = Tables::next(tables.pml4, (DIRECT_MAP >> 39) & 511);
        let pd = Tables::next(pdpt, (addr >> 30) & 511);
        // SAFETY: as in `Tables::next`.
        unsafe { *Tables::entry(pd, (addr >> 21) & 511) = addr | PRESENT | WRITABLE | HUGE };
    }
    // SAFETY: as above.
    unsafe { *Tables::entry(tables.pml4, 0) = *Tables::entry(tables.pml4, (DIRECT_MAP >> 39) & 511) };
    // The kernel image at its link address, with its segments' permissions.
    for s in elf.loads() {
        let mut flags = 0;
        if s.flags & 2 != 0 {
            flags |= WRITABLE;
        }
        if s.flags & 1 == 0 {
            flags |= NO_EXECUTE;
        }
        let first = s.vaddr & !(PAGE - 1);
        for v in (first..s.vaddr + s.memsz).step_by(PAGE as usize) {
            tables.map_4k(v, kernel_phys + v - lo, flags);
        }
    }

    let stack = alloc_pages(STACK_PAGES, 0);
    let trampoline = alloc_pages(1, 0x10_0000);
    let info_pages = size_of::<BootInfo>().div_ceil(PAGE as usize);
    let info_phys = alloc_pages(info_pages, 0);

    let rsdp = uefi::system::with_config_table(|t: &[ConfigTableEntry]| {
        t.iter().find(|e| e.guid == ConfigTableEntry::ACPI2_GUID).map_or(0, |e| e.address as u64)
    });
    let framebuffer = framebuffer();
    info!("rsdp {:#x}, framebuffer {}x{} at {:#x}, direct map up to {} GiB", rsdp, framebuffer.width, framebuffer.height, framebuffer.base, phys_limit >> 30);
    info!("leaving boot services");

    // SAFETY: nothing from boot services is used after this point.
    let mm = unsafe { boot::exit_boot_services(None) };

    // SAFETY: info_phys is a zeroed allocation of `info_pages` pages, identity-mapped.
    let bi = unsafe { &mut *(info_phys as *mut BootInfo) };
    bi.magic = bootinfo::MAGIC;
    bi.rsdp = rsdp;
    bi.framebuffer = framebuffer;
    bi.phys_limit = phys_limit;
    bi.kernel_phys = kernel_phys;
    bi.kernel_pages = kernel_pages as u64;
    bi.trampoline = trampoline;
    bi.module_count = modules.len() as u64;
    bi.modules[..modules.len()].copy_from_slice(&modules);
    bi.pml4 = tables.pml4;
    let mut n = 0;
    for d in mm.entries() {
        let r = Region { start: d.phys_start, pages: d.page_count, kind: bootinfo::classify(d.ty.0), efi_type: d.ty.0 };
        // Merge with the previous region when contiguous and of the same kind and type.
        if n > 0 {
            let p = &mut bi.regions[n - 1];
            if p.end() == r.start && p.kind == r.kind && p.efi_type == r.efi_type {
                p.pages += r.pages;
                continue;
            }
        }
        if n == MAX_REGIONS {
            break;
        }
        bi.regions[n] = r;
        n += 1;
    }
    bi.region_count = n as u64;

    // SAFETY: the new tables map this code (identity half), the stack and BootInfo (direct map) and the
    // kernel. NXE must be on before tables with NX bits are live.
    unsafe {
        asm!("cli");
        let (lo32, hi32): (u32, u32);
        asm!("rdmsr", in("ecx") 0xc000_0080u32, out("eax") lo32, out("edx") hi32);
        asm!("wrmsr", in("ecx") 0xc000_0080u32, in("eax") lo32 | 1 << 11, in("edx") hi32);
        let (t_lo, t_hi): (u32, u32);
        asm!("rdtsc", out("eax") t_lo, out("edx") t_hi);
        bi.tsc_at_entry = (t_hi as u64) << 32 | t_lo as u64;
        asm!(
            "mov cr3, {pml4}",
            "mov rsp, {stack}",
            "xor ebp, ebp",
            "call {entry}",
            "ud2",
            pml4 = in(reg) tables.pml4,
            stack = in(reg) DIRECT_MAP + stack + STACK_PAGES as u64 * PAGE,
            entry = in(reg) elf.entry(),
            in("rdi") DIRECT_MAP + info_phys,
            options(noreturn),
        );
    }
}

fn framebuffer() -> Framebuffer {
    let none = Framebuffer { base: 0, size: 0, width: 0, height: 0, stride: 0, format: PixelFormat::Bgr };
    let Ok(handle) = boot::get_handle_for_protocol::<GraphicsOutput>() else { return none };
    // Not exclusive: an exclusive open would take the display from the firmware console the log goes to.
    let params = boot::OpenProtocolParams { handle, agent: boot::image_handle(), controller: None };
    // SAFETY: only read here, while the console keeps using it.
    let Ok(mut g) = (unsafe { boot::open_protocol::<GraphicsOutput>(params, boot::OpenProtocolAttributes::GetProtocol) }) else { return none };
    let mode = g.current_mode_info();
    let format = match mode.pixel_format() {
        gop::PixelFormat::Rgb => PixelFormat::Rgb,
        gop::PixelFormat::Bgr => PixelFormat::Bgr,
        _ => return none,
    };
    let (w, h) = mode.resolution();
    let mut fb = g.frame_buffer();
    Framebuffer { base: fb.as_mut_ptr() as u64, size: fb.size() as u64, width: w as u32, height: h as u32, stride: mode.stride() as u32, format }
}

/// Every file in `\\modules`, each in pages of its own.
#[cfg(not(feature = "bundle"))]
fn load_modules(fs: &mut uefi::fs::FileSystem) -> alloc::vec::Vec<Module> {
    let mut out = alloc::vec::Vec::new();
    let Ok(dir) = fs.read_dir(cstr16!("\\modules")) else {
        info!("no \\modules directory");
        return out;
    };
    let names: alloc::vec::Vec<_> = dir.flatten().filter(|f| !f.is_directory()).map(|f| alloc::string::String::from(f.file_name())).collect();
    for name in names {
        if out.len() == MAX_MODULES {
            break;
        }
        let mut path = alloc::string::String::from("\\modules\\");
        path.push_str(&name);
        let p = uefi::CString16::try_from(path.as_str()).unwrap();
        let data = fs.read(p.as_ref()).expect("cannot read a module");
        out.push(place_module(&name, &data));
    }
    out
}

/// The kernel and modules built into the loader (`build.sh` with NETBOOT=1), for a network boot, where there is
/// no volume to read them from. Format: "DROSSBND", then per file a u16 name length, the name, a u64 size and
/// the data; a zero name length ends it.
#[cfg(feature = "bundle")]
fn load_bundle() -> (alloc::vec::Vec<u8>, alloc::vec::Vec<Module>) {
    static BUNDLE: &[u8] = include_bytes!(env!("DROSS_BUNDLE"));
    assert!(BUNDLE.starts_with(b"DROSSBND"), "bad bundle");
    let (mut at, mut image, mut out) = (8, alloc::vec::Vec::new(), alloc::vec::Vec::new());
    loop {
        let n = u16::from_le_bytes(BUNDLE[at..at + 2].try_into().unwrap()) as usize;
        if n == 0 {
            break;
        }
        let name = core::str::from_utf8(&BUNDLE[at + 2..at + 2 + n]).expect("bad bundle name");
        let size = u64::from_le_bytes(BUNDLE[at + 2 + n..at + 10 + n].try_into().unwrap()) as usize;
        let data = &BUNDLE[at + 10 + n..at + 10 + n + size];
        at += 10 + n + size;
        if name == "dross.elf" {
            image = data.to_vec();
        } else if out.len() < MAX_MODULES {
            out.push(place_module(name, data));
        }
    }
    assert!(!image.is_empty(), "no dross.elf in the bundle");
    (image, out)
}

/// A module in pages of its own.
fn place_module(name: &str, data: &[u8]) -> Module {
    let phys = alloc_pages(data.len().div_ceil(PAGE as usize).max(1), 0);
    // SAFETY: a fresh allocation of at least data.len() bytes.
    unsafe { ptr::copy_nonoverlapping(data.as_ptr(), phys as *mut u8, data.len()) };
    let mut m = Module { name: [0; 32], phys, size: data.len() as u64 };
    let n = name.len().min(31);
    m.name[..n].copy_from_slice(&name.as_bytes()[..n]);
    info!("module {}: {} KiB at {:#x}", name, data.len() / 1024, phys);
    m
}
