//! The Mac Pro 6,1's static tables (read from /sys/firmware/acpi/tables under Linux 7.3). The expected values
//! agree with Linux on the same machine: APIC IDs as in /proc/cpuinfo, the DMAR unit, ECAM and HPET bases.

use acpi::dmar::{scopes, Dmar, Remapping, Scope};
use acpi::fadt::Fadt;
use acpi::hpet::Hpet;
use acpi::madt::{Entry, Madt};
use acpi::{mcfg, s5_sleep_type, Gas, Sdt};

fn table(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/macpro61/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

#[test]
fn checksums() {
    for name in ["APIC", "DMAR", "FACP", "HPET", "MCFG"] {
        let b = table(name);
        let t = Sdt::parse(&b).unwrap();
        assert_eq!(&t.header.signature, name.as_bytes());
        assert_eq!(&t.header.oem_id, b"APPLE ");
        assert!(t.checksum_ok, "{name}");
    }
}

#[test]
fn madt() {
    let b = table("APIC");
    let m = Madt::parse(&Sdt::parse(&b).unwrap()).unwrap();
    assert_eq!(m.local_apic_address, 0xfee0_0000);
    assert_eq!(m.flags & 1, 1);
    // 12 cores, 2 threads each; the second half of the IDs starts at 16.
    let cpus: Vec<u32> = m.cpus().collect();
    let expect: Vec<u32> = (0..12).chain(16..28).collect();
    assert_eq!(cpus, expect);
    let ioapics: Vec<_> = m.entries().filter_map(|e| match e.unwrap() {
        Entry::IoApic { id, address, gsi_base } => Some((id, address, gsi_base)),
        _ => None,
    }).collect();
    assert_eq!(ioapics, [(8, 0xfec0_0000, 0), (9, 0xfec0_1000, 24)]);
    let overrides: Vec<_> = m.entries().filter_map(|e| match e.unwrap() {
        Entry::Override { source, gsi, flags, .. } => Some((source, gsi, flags)),
        _ => None,
    }).collect();
    // The PIT on GSI 2; the SCI level-triggered, active high.
    assert_eq!(overrides, [(0, 2, 0), (9, 9, 0xd)]);
    let nmis = m.entries().filter(|e| matches!(e, Ok(Entry::LocalApicNmi { lint: 1, .. }))).count();
    assert_eq!(nmis, 24);
    assert!(m.entries().all(|e| e.is_ok()));
}

#[test]
fn dmar() {
    let b = table("DMAR");
    let d = Dmar::parse(&Sdt::parse(&b).unwrap()).unwrap();
    assert_eq!(d.host_address_width, 46);
    // No interrupt remapping on this machine.
    assert!(!d.interrupt_remapping());
    let s: Vec<_> = d.structures().map(|s| s.unwrap()).collect();
    assert_eq!(s.len(), 3);
    assert!(matches!(s[0], Remapping::Unit { segment: 0, base: 0xdfff_8000, include_all: true, .. }));
    let Remapping::Reserved { base, limit, scopes: sc, .. } = s[1] else { panic!() };
    assert_eq!((base, limit), (0x7e7d_3000, 0x7e7d_5fff));
    // The two USB controllers (00:1d.0 EHCI, 00:1a.0).
    let devs: Vec<Scope> = scopes(sc).collect();
    assert_eq!(devs.iter().map(|x| (x.kind, x.start_bus, x.path)).collect::<Vec<_>>(), [(1, 0, &[0x1d, 0][..]), (1, 0, &[0x1a, 0][..])]);
    let Remapping::Ats { all_ports: false, scopes: sc, .. } = s[2] else { panic!() };
    assert_eq!(scopes(sc).map(|x| x.path[0]).collect::<Vec<_>>(), [1, 2, 3]);
}

#[test]
fn fadt() {
    let b = table("FACP");
    let f = Fadt::parse(&Sdt::parse(&b).unwrap()).unwrap();
    assert_eq!(f.dsdt, 0x7e58_3000);
    assert_eq!(f.sci_irq, 9);
    assert_eq!((f.pm1a_event, f.pm1a_control, f.pm1b_control, f.pm_timer), (0x400, 0x404, 0, 0x408));
    // MSI is supported; ASPM must not be touched.
    assert_eq!(f.iapc_boot_arch & Fadt::IAPC_MSI_NOT_SUPPORTED, 0);
    assert_eq!(f.iapc_boot_arch, 1 << 4);
    // Reset through the 0xcf9 reset control register.
    assert_eq!(f.reset, Some((Gas { space: Gas::IO, bit_width: 8, bit_offset: 0, access_size: 1, address: 0xcf9 }, 6)));
}

#[test]
fn hpet() {
    let b = table("HPET");
    let h = Hpet::parse(&Sdt::parse(&b).unwrap()).unwrap();
    assert_eq!((h.base.space, h.base.address), (Gas::MEMORY, 0xfed0_0000));
    assert_eq!(h.event_timer_block_id >> 16, 0x8086);
}

#[test]
fn mcfg() {
    let b = table("MCFG");
    let r: Vec<_> = mcfg::ranges(&Sdt::parse(&b).unwrap()).unwrap().collect();
    assert_eq!(r.len(), 1);
    assert_eq!((r[0].base, r[0].segment, r[0].start_bus, r[0].end_bus), (0xe000_0000, 0, 0, 0xff));
    // 02:00.0, the first D700.
    assert_eq!(r[0].function(2, 0, 0), Some(0xe020_0000));
}

#[test]
fn s5() {
    // The `\_S5` package as it is in the Mac Pro 6,1's DSDT (the DSDT itself is not in the repository).
    let aml = [0x06, 0x0a, 0x00, 0x08, b'_', b'S', b'5', b'_', 0x12, 0x08, 0x03, 0x0a, 0x07, 0x0a, 0x07, 0x0a, 0x00, 0x14];
    assert_eq!(s5_sleep_type(&aml), Some((7, 7)));
    assert_eq!(s5_sleep_type(&[0x08, b'_', b'S', b'5', b'_', 0x12, 0x06, 0x04, 0x00, 0x01, 0x00, 0x00]), Some((0, 1)));
    assert_eq!(s5_sleep_type(b"no sleep here"), None);
}
