//! The handover from the UEFI loader (`boot`) to the kernel: one `BootInfo` in memory the kernel owns, its
//! address in `rdi` at the kernel's entry point.
//!
//! When the kernel starts, the loader's page tables are live: all physical memory below `phys_limit` is mapped
//! write-back at `DIRECT_MAP` and at its identity address (the identity half is only for the AP trampoline and
//! goes away once the APs are up), and the kernel image is at its link address. Every address in `BootInfo` is
//! physical unless the field says otherwise.
#![no_std]

/// Where the direct map of physical memory starts (PML4 slot 256).
pub const DIRECT_MAP: u64 = 0xffff_8000_0000_0000;

pub const MAGIC: u64 = u64::from_le_bytes(*b"DROSSBI2");

/// UEFI memory types the loader allocates its handover in, so the kernel can tell them apart (UEFI reserves
/// 0x8000_0000 and up for OS loaders).
pub const EFI_KERNEL_TYPE: u32 = 0x8000_0001;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    /// Free for the kernel: conventional memory and what boot services and the loader used.
    Usable = 0,
    /// The kernel image, its boot stack, the page tables, this `BootInfo`, the AP trampoline page.
    Kernel = 1,
    /// ACPI tables (reclaimable once parsed, but kept: drivers read VFCT and the DSDT later).
    AcpiTables = 2,
    AcpiNvs = 3,
    /// UEFI runtime services code and data, MMIO, reserved, unusable: never touched.
    Reserved = 4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub start: u64,
    pub pages: u64,
    pub kind: RegionKind,
    /// The UEFI memory type, for the log.
    pub efi_type: u32,
}

impl Region {
    pub const fn end(&self) -> u64 {
        self.start + self.pages * 4096
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb = 0,
    Bgr = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Framebuffer {
    /// 0 when the firmware gave no linear framebuffer.
    pub base: u64,
    pub size: u64,
    pub width: u32,
    pub height: u32,
    /// Pixels per scan line.
    pub stride: u32,
    pub format: PixelFormat,
}

pub const MAX_REGIONS: usize = 512;
pub const MAX_MODULES: usize = 32;

/// A file from the boot volume's `\modules` directory, loaded for the kernel (user programs).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Module {
    /// UTF-8, NUL-padded.
    pub name: [u8; 32],
    pub phys: u64,
    pub size: u64,
}

impl Module {
    pub fn name(&self) -> &str {
        let n = self.name.iter().position(|&b| b == 0).unwrap_or(32);
        core::str::from_utf8(&self.name[..n]).unwrap_or("?")
    }
}

#[repr(C)]
pub struct BootInfo {
    pub magic: u64,
    /// The ACPI 2.0+ RSDP, 0 if the firmware has none.
    pub rsdp: u64,
    pub framebuffer: Framebuffer,
    /// Physical memory below this is in the direct map.
    pub phys_limit: u64,
    pub kernel_phys: u64,
    pub kernel_pages: u64,
    /// One 4 KiB page below 1 MiB for the AP startup trampoline.
    pub trampoline: u64,
    /// The PML4 the kernel starts on.
    pub pml4: u64,
    /// TSC when the loader jumped to the kernel.
    pub tsc_at_entry: u64,
    pub module_count: u64,
    pub modules: [Module; MAX_MODULES],
    pub region_count: u64,
    pub regions: [Region; MAX_REGIONS],
}

impl BootInfo {
    pub fn regions(&self) -> &[Region] {
        &self.regions[..self.region_count as usize]
    }

    pub fn modules(&self) -> &[Module] {
        &self.modules[..self.module_count as usize]
    }
}

/// Maps a UEFI memory type to what the kernel may do with it.
pub const fn classify(efi_type: u32) -> RegionKind {
    match efi_type {
        // LoaderCode, LoaderData, BootServicesCode, BootServicesData, ConventionalMemory.
        1..=4 | 7 => RegionKind::Usable,
        9 => RegionKind::AcpiTables,
        10 => RegionKind::AcpiNvs,
        EFI_KERNEL_TYPE => RegionKind::Kernel,
        _ => RegionKind::Reserved,
    }
}
