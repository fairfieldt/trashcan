//! The VFIO ioctls and structures gpud uses, from Linux's `include/uapi/linux/vfio.h` (7.3).

#![allow(non_camel_case_types, non_upper_case_globals)]

use libc::c_ulong;

const VFIO_TYPE: u32 = b';' as u32;
const VFIO_BASE: u32 = 100;

/// `_IO(VFIO_TYPE, VFIO_BASE + nr)`: no direction or size bits.
const fn io(nr: u32) -> c_ulong {
    ((VFIO_TYPE << 8) | (VFIO_BASE + nr)) as c_ulong
}

pub const VFIO_API_VERSION: i32 = 0;
pub const VFIO_TYPE1v2_IOMMU: c_ulong = 3;

pub const VFIO_GET_API_VERSION: c_ulong = io(0);
pub const VFIO_CHECK_EXTENSION: c_ulong = io(1);
pub const VFIO_SET_IOMMU: c_ulong = io(2);
pub const VFIO_GROUP_GET_STATUS: c_ulong = io(3);
pub const VFIO_GROUP_SET_CONTAINER: c_ulong = io(4);
pub const VFIO_GROUP_GET_DEVICE_FD: c_ulong = io(6);
pub const VFIO_DEVICE_GET_INFO: c_ulong = io(7);
pub const VFIO_DEVICE_GET_REGION_INFO: c_ulong = io(8);
pub const VFIO_DEVICE_GET_IRQ_INFO: c_ulong = io(9);
pub const VFIO_DEVICE_SET_IRQS: c_ulong = io(10);
pub const VFIO_DEVICE_GET_PCI_HOT_RESET_INFO: c_ulong = io(12);
pub const VFIO_IOMMU_GET_INFO: c_ulong = io(12);
pub const VFIO_IOMMU_MAP_DMA: c_ulong = io(13);
pub const VFIO_IOMMU_UNMAP_DMA: c_ulong = io(14);

pub const VFIO_GROUP_FLAGS_VIABLE: u32 = 1 << 0;
pub const VFIO_DEVICE_FLAGS_RESET: u32 = 1 << 0;
pub const VFIO_DEVICE_FLAGS_PCI: u32 = 1 << 1;
pub const VFIO_REGION_INFO_FLAG_READ: u32 = 1 << 0;
pub const VFIO_REGION_INFO_FLAG_WRITE: u32 = 1 << 1;
pub const VFIO_REGION_INFO_FLAG_MMAP: u32 = 1 << 2;
pub const VFIO_IRQ_SET_DATA_NONE: u32 = 1 << 0;
pub const VFIO_IRQ_SET_DATA_EVENTFD: u32 = 1 << 2;
pub const VFIO_IRQ_SET_ACTION_TRIGGER: u32 = 1 << 5;
pub const VFIO_IOMMU_INFO_CAPS: u32 = 1 << 1;
pub const VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE: u16 = 1;
pub const VFIO_IOMMU_TYPE1_INFO_DMA_AVAIL: u16 = 3;
pub const VFIO_DMA_MAP_FLAG_READ: u32 = 1 << 0;
pub const VFIO_DMA_MAP_FLAG_WRITE: u32 = 1 << 1;

/// Region indices of a vfio-pci device.
pub const VFIO_PCI_BAR0_REGION_INDEX: u32 = 0;
pub const VFIO_PCI_ROM_REGION_INDEX: u32 = 6;
pub const VFIO_PCI_CONFIG_REGION_INDEX: u32 = 7;
pub const VFIO_PCI_MSI_IRQ_INDEX: u32 = 1;

#[repr(C)]
#[derive(Default)]
pub struct vfio_group_status {
    pub argsz: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct vfio_device_info {
    pub argsz: u32,
    pub flags: u32,
    pub num_regions: u32,
    pub num_irqs: u32,
    pub cap_offset: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct vfio_region_info {
    pub argsz: u32,
    pub flags: u32,
    pub index: u32,
    pub cap_offset: u32,
    pub size: u64,
    pub offset: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct vfio_irq_info {
    pub argsz: u32,
    pub flags: u32,
    pub index: u32,
    pub count: u32,
}

/// `vfio_irq_set` with room for one eventfd.
#[repr(C)]
#[derive(Default)]
pub struct vfio_irq_set_one {
    pub argsz: u32,
    pub flags: u32,
    pub index: u32,
    pub start: u32,
    pub count: u32,
    pub fd: i32,
}

#[repr(C)]
#[derive(Default)]
pub struct vfio_iommu_type1_info {
    pub argsz: u32,
    pub flags: u32,
    pub iova_pgsizes: u64,
    pub cap_offset: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct vfio_iommu_type1_dma_map {
    pub argsz: u32,
    pub flags: u32,
    pub vaddr: u64,
    pub iova: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct vfio_iommu_type1_dma_unmap {
    pub argsz: u32,
    pub flags: u32,
    pub iova: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct vfio_pci_hot_reset_info {
    pub argsz: u32,
    pub flags: u32,
    pub count: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct vfio_pci_dependent_device {
    pub group_id: u32,
    pub segment: u16,
    pub bus: u8,
    pub devfn: u8,
}

const _: () = {
    // The kernel's sizes for these structs (x86-64).
    assert!(size_of::<vfio_group_status>() == 8);
    assert!(size_of::<vfio_device_info>() == 24);
    assert!(size_of::<vfio_region_info>() == 32);
    assert!(size_of::<vfio_irq_info>() == 16);
    assert!(size_of::<vfio_irq_set_one>() == 24);
    assert!(size_of::<vfio_iommu_type1_info>() == 24);
    assert!(size_of::<vfio_iommu_type1_dma_map>() == 32);
    assert!(size_of::<vfio_iommu_type1_dma_unmap>() == 24);
    assert!(size_of::<vfio_pci_hot_reset_info>() == 12);
    assert!(size_of::<vfio_pci_dependent_device>() == 8);
    assert!(VFIO_GET_API_VERSION == 0x3b64);
    assert!(VFIO_IOMMU_UNMAP_DMA == 0x3b72);
};
