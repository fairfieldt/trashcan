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
