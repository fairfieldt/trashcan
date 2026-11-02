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
//! signatures. On Linux the backend is VFIO ([`vfio`], type1 IOMMU); a microkernel backend implements the same
//! traits over its capabilities (config access from a PCI service, device frames mapped into the server's address
//! space, an IOMMU domain fed with frames the server owns, MSIs bound to notification objects). [`mock`] implements
//! them in memory for tests.

pub mod dma;
pub mod error;
pub mod iova;
pub mod irq;
pub mod mmio;
pub mod mock;
pub mod pci;
pub mod vbios;
#[cfg(target_os = "linux")]
pub mod vfio;

pub use dma::{DmaAccess, DmaBuffer, DmaSpace};
pub use error::{Error, Result};
pub use irq::IrqEvent;
pub use mmio::Mmio;
pub use pci::{PciAddress, PciDevice, Region, RegionInfo, ResetCapability};
