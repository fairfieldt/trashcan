//! The platform interface of gpud, the userspace GPU driver server.
//!
//! The driver (Rust today, imported Linux C later through a compatibility layer) reaches the hardware only
//! through this crate:
//!
//! - [`PciDevice`]: one PCI function owned by this process: its config space, its regions (BARs, expansion ROM),
//!   mapping a memory BAR ([`Mmio`]), and its interrupts ([`IrqEvent`]).
//! - [`DmaSpace`]: the device's view of memory (an IOMMU domain): pinned host memory at device addresses
//!   ([`DmaBuffer`]).
//!
//! The interface names hardware operations, never Linux objects: no file descriptors, paths or sysfs in its
//! signatures. On Linux the backend is VFIO (`vfio`, type1 IOMMU); on dross, the trashcan microkernel, it is
//! `dross` (feature `dross`, without `std`): config access through pcid, BARs as memory objects mapped into the
//! driver, the function's own IOMMU domain fed with memory objects the driver owns, MSIs bound to notifications.
//! `mock` implements them in memory for tests. Everything but `vfio` and `mock` builds without `std`.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod prelude {
    pub use alloc::boxed::Box;
    pub use alloc::format;
    pub use alloc::string::{String, ToString};
    pub use alloc::vec;
    pub use alloc::vec::Vec;
}

pub mod dma;
pub mod error;
pub mod iova;
pub mod irq;
pub mod mmio;
#[cfg(feature = "std")]
pub mod mock;
pub mod pci;
pub mod vbios;
#[cfg(all(feature = "std", target_os = "linux"))]
pub mod vfio;
#[cfg(feature = "dross")]
pub mod dross;

pub use dma::{DmaAccess, DmaBuffer, DmaSpace};
pub use error::{Error, Result};
pub use irq::IrqEvent;
pub use mmio::Mmio;
pub use pci::{PciAddress, PciDevice, Region, RegionInfo, ResetCapability};

/// Opens a PCI function with the backend this process runs on: dross, when dross started the process (it sets
/// `DROSS_BOOTSTRAP`), else VFIO on Linux. `iova` is the device address range to hand out (VFIO; dross gives
/// each function a domain of its own and ignores it).
#[cfg(feature = "std")]
pub fn open(address: PciAddress, iova: core::ops::Range<u64>) -> Result<(Box<dyn DmaSpace>, Box<dyn PciDevice>)> {
    #[cfg(feature = "dross")]
    if dross::started_by_dross() {
        let _ = iova;
        let (dev, dma) = dross::open_from_env(address)?;
        return Ok((Box::new(dma), Box::new(dev)));
    }
    #[cfg(target_os = "linux")]
    {
        let (container, dev) = vfio::open(address, iova)?;
        Ok((Box::new(container), Box::new(dev)))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = iova;
        Err(Error::Unsupported(format!("{address}: no PCI backend on this platform")))
    }
}

/// Whether this process runs on dross.
#[cfg(feature = "std")]
pub fn on_dross() -> bool {
    #[cfg(feature = "dross")]
    return dross::started_by_dross();
    #[cfg(not(feature = "dross"))]
    false
}
