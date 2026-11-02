use crate::Result;
use std::ops::Range;
use std::ptr::NonNull;

/// Mappings of this size or more start on a multiple of it in the device address space (IOMMU superpages, and
/// GPU page table fragments later).
pub const LARGE_ALIGN: usize = 2 << 20;

/// Device address alignment for a mapping of `size` bytes.
pub fn iova_align(size: usize) -> u64 {
    if size >= LARGE_ALIGN { LARGE_ALIGN as u64 } else { 4096 }
}

/// What the device may do with a mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DmaAccess {
    DeviceReads,
    DeviceWrites,
    ReadWrite,
}

/// The device's view of memory: an IOMMU domain on Linux (a VFIO container), shared by every device attached to it
/// (both GPUs of a Mac Pro, later, so each can reach the other's BAR).
pub trait DmaSpace: Send + Sync {
    /// Zeroed host memory, pinned and mapped for the device. The size is rounded up to the page size, and to
    /// [`LARGE_ALIGN`] for buffers of that size or more.
    fn alloc(&self, size: usize, access: DmaAccess) -> Result<DmaBuffer>;

    /// Maps memory this process already has: a file mapping, or another device's BAR for peer-to-peer.
    ///
    /// # Safety
    /// `ptr..ptr+size` must stay mapped until the returned buffer is dropped; the device may read or write it at
    /// any time until then.
    unsafe fn map_external(&self, ptr: NonNull<u8>, size: usize, access: DmaAccess) -> Result<DmaBuffer>;

    /// Device address ranges this space hands out.
    fn iova_ranges(&self) -> Vec<Range<u64>>;

    /// Page sizes the IOMMU can map (a bitmap: bit n set means 2^n bytes).
    fn page_sizes(&self) -> u64;
}

/// Host memory the device reaches at `iova()`. Unmapped from the device (and freed, if the space allocated it)
/// when dropped; drop it only once the device has stopped using it.
pub struct DmaBuffer {
    ptr: NonNull<u8>,
    len: usize,
    iova: u64,
    release: Option<Box<dyn FnOnce() + Send + Sync>>,
}

// SAFETY: the buffer is plain memory; the device's concurrent access is ordered by the driver (fences).
unsafe impl Send for DmaBuffer {}
unsafe impl Sync for DmaBuffer {}

impl DmaBuffer {
    /// # Safety
    /// `ptr..ptr+len` must be valid host memory mapped for the device at `iova` until `release` runs.
    pub unsafe fn from_raw(ptr: NonNull<u8>, len: usize, iova: u64, release: Box<dyn FnOnce() + Send + Sync>) -> Self {
        DmaBuffer { ptr, len, iova, release: Some(release) }
    }

    pub fn iova(&self) -> u64 {
        self.iova
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// A dword the device may have written: a fence, a read pointer.
    pub fn read32(&self, offset: usize) -> u32 {
        assert!(offset.is_multiple_of(4) && offset + 4 <= self.len, "DMA read at {offset:#x} outside {:#x}", self.len);
        // SAFETY: in bounds and aligned; the memory lives while self does.
        unsafe { self.ptr.as_ptr().add(offset).cast::<u32>().read_volatile() }
    }

    pub fn write32(&self, offset: usize, value: u32) {
        assert!(offset.is_multiple_of(4) && offset + 4 <= self.len, "DMA write at {offset:#x} outside {:#x}", self.len);
        // SAFETY: as in read32.
        unsafe { self.ptr.as_ptr().add(offset).cast::<u32>().write_volatile(value) }
    }

    /// The buffer as bytes, for filling it before the device sees it.
    ///
    /// # Safety
    /// The device must not access the buffer while the slice lives.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// # Safety
    /// The device must not write the buffer while the slice lives.
    pub unsafe fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for DmaBuffer {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release();
        }
    }
}
