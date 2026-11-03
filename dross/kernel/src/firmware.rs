//! Finds the ACPI tables through the direct map and logs what the kernel will rely on.

use acpi::dmar::{Dmar, Remapping};
use acpi::fadt::Fadt;
use acpi::madt::Madt;
use acpi::{mcfg, Header, Rsdp, Sdt};
use bootinfo::BootInfo;

use crate::mem::virt;
use crate::println;

pub struct Acpi {
    pub madt: Sdt<'static>,
    pub hpet: Option<Sdt<'static>>,
    pub mcfg: Option<Sdt<'static>>,
    pub dmar: Option<Sdt<'static>>,
    pub fadt: Option<Sdt<'static>>,
    pub dsdt: Option<Sdt<'static>>,
    /// The VBIOS images (gpud reads them later).
    pub vfct: Option<Sdt<'static>>,
}

/// # Safety
/// `phys` must point at memory the firmware filled with a table that stays in place.
unsafe fn table(phys: u64) -> Option<Sdt<'static>> {
    let head = unsafe { core::slice::from_raw_parts(virt(phys) as *const u8, Header::LEN) };
    let len = Header::parse(head).ok()?.length as usize;
    Sdt::parse(unsafe { core::slice::from_raw_parts(virt(phys) as *const u8, len) }).ok()
}

impl Acpi {
    pub fn find(info: &BootInfo) -> Acpi {
        assert!(info.rsdp != 0, "the firmware gave no ACPI 2.0 RSDP");
        // SAFETY: the RSDP is firmware memory in the direct map.
        let rsdp = Rsdp::parse(unsafe { core::slice::from_raw_parts(virt(info.rsdp) as *const u8, Rsdp::LEN_V2) }).expect("bad RSDP");
        let root_phys = if rsdp.xsdt != 0 { rsdp.xsdt } else { rsdp.rsdt as u64 };
        // SAFETY: as above, for every table the root lists.
        let root = unsafe { table(root_phys) }.expect("bad XSDT");
        let (mut madt, mut hpet, mut mcfg, mut dmar, mut fadt, mut vfct) = (None, None, None, None, None, None);
        let mut names = [[b' '; 5]; 32];
        let mut n = 0;
        for phys in acpi::root_entries(&root) {
            let Some(t) = (unsafe { table(phys) }) else { continue };
            if n < names.len() {
                names[n][..4].copy_from_slice(&t.header.signature);
                names[n][4] = if t.checksum_ok { b' ' } else { b'!' };
                n += 1;
            }
            match &t.header.signature {
                b"APIC" => madt = Some(t),
                b"HPET" => hpet = Some(t),
                b"MCFG" => mcfg = Some(t),
                b"DMAR" => dmar = Some(t),
                b"FACP" => fadt = Some(t),
                b"VFCT" => vfct = Some(t),
                _ => {}
            }
        }
        println!(
            "acpi: rev {} oem {} tables {}(! = bad checksum)",
            rsdp.revision,
            core::str::from_utf8(&rsdp.oem_id).unwrap_or("?"),
            Names(&names[..n])
        );
        let dsdt = fadt.as_ref().and_then(|f| Fadt::parse(f).ok()).and_then(|f| unsafe { table(f.dsdt) });
        let a = Acpi { madt: madt.expect("no MADT"), hpet, mcfg, dmar, fadt, dsdt, vfct };
        a.log();
        a
    }

    fn log(&self) {
        let m = Madt::parse(&self.madt).unwrap();
        println!("acpi: {} cpus, local apic {:#x}, 8259s {}", m.cpus().count(), m.local_apic_address, if m.flags & 1 != 0 { "present" } else { "absent" });
        if let Some(t) = &self.mcfg {
            for r in mcfg::ranges(t).unwrap() {
                println!("acpi: ecam {:#x} segment {} buses {:#x}-{:#x}", r.base, r.segment, r.start_bus, r.end_bus);
            }
        }
        if let Some(t) = &self.dmar {
            let d = Dmar::parse(t).unwrap();
            let units = d.structures().filter(|s| matches!(s, Ok(Remapping::Unit { .. }))).count();
            let rmrr = d.structures().filter(|s| matches!(s, Ok(Remapping::Reserved { .. }))).count();
            println!(
                "acpi: dmar: {} remapping units, {} reserved regions, {}-bit DMA, interrupt remapping {}",
                units,
                rmrr,
                d.host_address_width,
                if d.interrupt_remapping() { "yes" } else { "no" }
            );
        }
        if let Some(f) = self.fadt.as_ref().and_then(|f| Fadt::parse(f).ok()) {
            if let Some((g, v)) = f.reset.filter(|(g, _)| g.space == acpi::Gas::IO) {
                crate::qemu::set_reset(g.address as u16, v);
            }
            let s5 = self.dsdt.as_ref().and_then(|d| acpi::s5_sleep_type(d.bytes));
            println!("acpi: fadt: sci {}, pm1a {:#x}, reset {:?}, s5 {:?}", f.sci_irq, f.pm1a_control, f.reset.map(|(g, v)| (g.space, g.address, v)), s5);
        }
        if let Some(v) = &self.vfct {
            println!("acpi: vfct {} bytes", v.bytes.len());
        }
    }
}

struct Names<'a>(&'a [[u8; 5]]);

impl core::fmt::Display for Names<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        for n in self.0 {
            f.write_str(core::str::from_utf8(n).unwrap_or("?"))?;
        }
        Ok(())
    }
}

/// On a Mac Pro 6,1 (Intel C600 LPC bridge at 00:1f.0), power on when AC power returns: clears AFTERG3_EN in
/// GEN_PMCON_3, so a test machine on a switched outlet comes back when the switch does.
pub fn power_on_after_power_loss(acpi: &Acpi) {
    let Some(r) = acpi.mcfg.as_ref().and_then(|t| mcfg::ranges(t).ok()).and_then(|mut r| r.next()) else {
        return;
    };
    let Some(phys) = r.function(0, 0x1f, 0) else {
        return;
    };
    let cfg = crate::mem::map_mmio(phys, 4096);
    // SAFETY: the LPC bridge's config space, just mapped.
    unsafe {
        let id = (cfg as *const u32).read_volatile();
        if id != 0x1d41_8086 {
            return;
        }
        let reg = (cfg + 0xa4) as *mut u8;
        let v = reg.read_volatile();
        if v & 1 != 0 {
            reg.write_volatile(v & !1);
        }
        println!("chipset: power on after power loss (GEN_PMCON_3 {:#x} -> {:#x})", v, reg.read_volatile());
    }
}
