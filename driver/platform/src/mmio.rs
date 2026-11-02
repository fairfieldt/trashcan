use std::ptr::NonNull;

/// A mapped device memory region (a BAR), accessed with volatile loads and stores.
///
/// Offsets are byte offsets; an access outside the region or a misaligned one is a driver bug and panics, as slice
/// indexing does. The C compatibility layer takes [`Mmio::as_ptr`] for `readl`/`writel`.
pub struct Mmio {
    ptr: NonNull<u8>,
    len: usize,
    // Unmaps the region when dropped (a munmap guard, or the backing memory of a mock).
    _backing: Box<dyn Send + Sync>,
}

// SAFETY: the mapping is device memory shared with the hardware; ordering between threads is the driver's job, as
// it is for the registers themselves.
unsafe impl Send for Mmio {}
unsafe impl Sync for Mmio {}

impl Mmio {
    /// # Safety
    /// `ptr..ptr+len` must stay mapped and valid for volatile access while `backing` lives.
    pub unsafe fn from_raw(ptr: NonNull<u8>, len: usize, backing: Box<dyn Send + Sync>) -> Self {
        Mmio { ptr, len, _backing: backing }
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

    #[inline]
    fn check(&self, offset: usize, size: usize) {
        assert!(
            offset.is_multiple_of(size) && offset.checked_add(size).is_some_and(|end| end <= self.len),
            "MMIO access of {size} bytes at {offset:#x} outside a {:#x}-byte region or misaligned",
            self.len
        );
    }

    #[inline]
    pub fn read32(&self, offset: usize) -> u32 {
        self.check(offset, 4);
        // SAFETY: in bounds and aligned (checked); the mapping is live while self is.
        unsafe { self.ptr.as_ptr().add(offset).cast::<u32>().read_volatile() }
    }

    #[inline]
    pub fn write32(&self, offset: usize, value: u32) {
        self.check(offset, 4);
        // SAFETY: as in read32.
        unsafe { self.ptr.as_ptr().add(offset).cast::<u32>().write_volatile(value) }
    }

    /// Copies dwords out of the region (`buf.len()` dwords from `offset`).
    pub fn read_dwords(&self, offset: usize, buf: &mut [u32]) {
        self.check(offset, 4);
        self.check(offset + (buf.len().max(1) - 1) * 4, 4);
        for (i, v) in buf.iter_mut().enumerate() {
            *v = self.read32(offset + i * 4);
        }
    }

    pub fn write_dwords(&self, offset: usize, data: &[u32]) {
        self.check(offset, 4);
        self.check(offset + (data.len().max(1) - 1) * 4, 4);
        for (i, &v) in data.iter().enumerate() {
            self.write32(offset + i * 4, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::mock::AlignedMemory;
    use std::panic::AssertUnwindSafe;

    #[test]
    fn reads_back_writes_and_bounds_checks() {
        let mmio = AlignedMemory::new(64).into_mmio();
        mmio.write32(0, 0xdead_beef);
        mmio.write_dwords(8, &[1, 2, 3]);
        let mut out = [0u32; 3];
        mmio.read_dwords(8, &mut out);
        assert_eq!(mmio.read32(0), 0xdead_beef);
        assert_eq!(out, [1, 2, 3]);
        let panics = |f: &dyn Fn()| std::panic::catch_unwind(AssertUnwindSafe(f)).is_err();
        assert!(panics(&|| _ = mmio.read32(64)));
        assert!(panics(&|| _ = mmio.read32(2)));
        assert!(panics(&|| mmio.write_dwords(56, &[0; 3])));
    }
}
