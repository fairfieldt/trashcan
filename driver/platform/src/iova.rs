//! Device address (IOVA) allocation: first fit over the usable ranges, freed ranges merge with their neighbours.

use crate::{Error, Result};
use std::ops::Range;

pub struct IovaAllocator {
    // Free ranges, sorted by start and never touching each other.
    free: Vec<Range<u64>>,
}

impl IovaAllocator {
    /// `usable`: the ranges the IOMMU translates, already clipped to what the device can address.
    pub fn new(usable: &[Range<u64>]) -> Self {
        let mut a = IovaAllocator { free: Vec::new() };
        for r in usable {
            if r.start < r.end {
                a.release(r.clone());
            }
        }
        a
    }

    pub fn free_ranges(&self) -> &[Range<u64>] {
        &self.free
    }

    /// `align` is a power of two.
    pub fn alloc(&mut self, size: u64, align: u64) -> Result<u64> {
        assert!(align.is_power_of_two() && size > 0);
        for i in 0..self.free.len() {
            let r = self.free[i].clone();
            let Some(start) = r.start.checked_next_multiple_of(align) else { continue };
            let Some(end) = start.checked_add(size) else { continue };
            if end > r.end {
                continue;
            }
            match (start == r.start, end == r.end) {
                (true, true) => {
                    self.free.remove(i);
                }
                (true, false) => self.free[i].start = end,
                (false, true) => self.free[i].end = start,
                (false, false) => {
                    self.free[i].end = start;
                    self.free.insert(i + 1, end..r.end);
                }
            }
            return Ok(start);
        }
        Err(Error::Exhausted(format!("no free {size:#x}-byte range aligned to {align:#x}")))
    }

    /// Returns a range from [`alloc`](Self::alloc). Freeing what isn't allocated is a bug and panics.
    pub fn release(&mut self, r: Range<u64>) {
        let i = self.free.partition_point(|f| f.start < r.start);
        let overlaps_prev = i > 0 && self.free[i - 1].end > r.start;
        let overlaps_next = i < self.free.len() && self.free[i].start < r.end;
        assert!(!overlaps_prev && !overlaps_next, "IOVA range {r:x?} freed twice");
        let merge_prev = i > 0 && self.free[i - 1].end == r.start;
        let merge_next = i < self.free.len() && self.free[i].start == r.end;
        match (merge_prev, merge_next) {
            (true, true) => {
                self.free[i - 1].end = self.free[i].end;
                self.free.remove(i);
            }
            (true, false) => self.free[i - 1].end = r.end,
            (false, true) => self.free[i].start = r.start,
            (false, false) => self.free.insert(i, r),
        }
    }
}

/// `a` minus the ranges in `holes` (both half-open).
pub fn subtract(a: Range<u64>, holes: &[Range<u64>]) -> Vec<Range<u64>> {
    let mut out = vec![a];
    for h in holes {
        out = out
            .into_iter()
            .flat_map(|r| {
                if h.end <= r.start || h.start >= r.end {
                    vec![r]
                } else {
                    [r.start..h.start.max(r.start), h.end.min(r.end)..r.end]
                        .into_iter()
                        .filter(|p| p.start < p.end)
                        .collect()
                }
            })
            .collect();
    }
    out
}

/// The intersection of two sets of ranges (each sorted and non-overlapping).
pub fn intersect(a: &[Range<u64>], b: &[Range<u64>]) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    for x in a {
        for y in b {
            let r = x.start.max(y.start)..x.end.min(y.end);
            if r.start < r.end {
                out.push(r);
            }
        }
    }
    out.sort_by_key(|r| r.start);
    out
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)] // one usable range is the common case
mod tests {
    use super::*;

    #[test]
    fn first_fit_with_alignment_and_merge() {
        let mut a = IovaAllocator::new(&[0x1000..0x10000, 0x20000..0x40000]);
        let x = a.alloc(0x1000, 0x1000).unwrap();
        assert_eq!(x, 0x1000);
        let y = a.alloc(0x8000, 0x8000).unwrap();
        assert_eq!(y, 0x8000);
        assert_eq!(a.free_ranges(), &[0x2000..0x8000, 0x20000..0x40000]);
        let z = a.alloc(0x10000, 0x10000).unwrap();
        assert_eq!(z, 0x20000);
        a.release(y..y + 0x8000);
        a.release(x..x + 0x1000);
        a.release(z..z + 0x10000);
        assert_eq!(a.free_ranges(), &[0x1000..0x10000, 0x20000..0x40000]);
        assert!(matches!(a.alloc(0x40000, 0x1000), Err(Error::Exhausted(_))));
    }

    #[test]
    fn splits_a_range_around_an_aligned_block() {
        let mut a = IovaAllocator::new(&[0x1000..0x400000]);
        let x = a.alloc(0x200000, 0x200000).unwrap();
        assert_eq!(x, 0x200000);
        assert_eq!(a.free_ranges(), &[0x1000..0x200000]);
        a.release(x..x + 0x200000);
        assert_eq!(a.free_ranges(), &[0x1000..0x400000]);
    }

    #[test]
    #[should_panic(expected = "freed twice")]
    fn double_free_panics() {
        let mut a = IovaAllocator::new(&[0x1000..0x10000]);
        let x = a.alloc(0x1000, 0x1000).unwrap();
        a.release(x..x + 0x1000);
        a.release(x..x + 0x1000);
    }

    #[test]
    fn range_arithmetic() {
        // The x86 MSI window cut out of a 40-bit device range.
        let usable = subtract(0x10_0000..1 << 40, &[0xfee0_0000..0xff00_0000]);
        assert_eq!(usable, vec![0x10_0000..0xfee0_0000, 0xff00_0000..1 << 40]);
        let iommu = [0..0xfee0_0000, 0xfef0_0000..1 << 48];
        assert_eq!(intersect(&usable, &iommu), vec![0x10_0000..0xfee0_0000, 0xff00_0000..1 << 40]);
    }
}
