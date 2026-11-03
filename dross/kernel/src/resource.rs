//! Resources: the right to reach a range of something the kernel guards. init gets one root of each kind and
//! hands out narrower ones (`RESOURCE_SUB`).

use dross_abi::Error;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Physical address ranges outside RAM (device registers, firmware tables): `MEM_PHYSICAL`.
    Physical,
    /// I/O ports: `IOPORT_ENABLE`.
    IoPort,
    /// Interrupt vectors: `IRQ_MSI`.
    Irq,
    /// The IOMMU: attaching devices to DMA domains.
    Iommu,
    /// Reset, power off, ending a QEMU run.
    System,
}

pub struct Resource {
    pub kind: Kind,
    pub base: u64,
    pub size: u64,
}

impl Resource {
    pub const fn root(kind: Kind, size: u64) -> Resource {
        Resource { kind, base: 0, size }
    }

    pub fn covers(&self, base: u64, size: u64) -> bool {
        base >= self.base && base.checked_add(size).is_some_and(|e| e <= self.base + self.size)
    }

    pub fn sub(&self, base: u64, size: u64) -> Result<Resource, Error> {
        if size == 0 || !self.covers(base, size) {
            return Err(Error::OutOfRange);
        }
        Ok(Resource { kind: self.kind, base, size })
    }

    pub fn check(&self, kind: Kind, base: u64, size: u64) -> Result<(), Error> {
        if self.kind != kind {
            return Err(Error::WrongType);
        }
        if !self.covers(base, size) {
            return Err(Error::AccessDenied);
        }
        Ok(())
    }
}
