//! Physical memory in 4 KiB frames, one bit each (set: in use). 64 GiB takes 2 MiB of bitmap.
//!
//! Allocation is first fit from a moving hint, with any power-of-two alignment and an optional address
//! ceiling, so the same allocator serves page tables, 2 MiB DMA buffers and the below-4 GiB pages the AP
//! trampoline needs. Linear scans are fine for a machine that allocates its big buffers once.
#![no_std]

pub const FRAME: u64 = 4096;

pub struct Frames<'a> {
    bits: &'a mut [u64],
    frames: u64,
    free: u64,
    hint: u64,
}

impl<'a> Frames<'a> {
    /// Words of bitmap needed to cover physical memory below `limit`.
    pub const fn words_for(limit: u64) -> usize {
        (limit.div_ceil(FRAME)).div_ceil(64) as usize
    }

    /// Everything starts in use; `release` the usable ranges.
    pub fn new(bits: &'a mut [u64]) -> Self {
        bits.fill(!0);
        let frames = bits.len() as u64 * 64;
        Frames { bits, frames, free: 0, hint: 0 }
    }

    pub fn free_frames(&self) -> u64 {
        self.free
    }

    fn is_used(&self, f: u64) -> bool {
        self.bits[(f / 64) as usize] & (1 << (f % 64)) != 0
    }

    fn set(&mut self, f: u64, used: bool) {
        let w = &mut self.bits[(f / 64) as usize];
        let m = 1u64 << (f % 64);
        if used {
            *w |= m
        } else {
            *w &= !m
        }
    }

    /// Sets frames [first, last) to `used`, a word at a time where it can; returns how many changed.
    fn fill(&mut self, first: u64, last: u64, used: bool) -> u64 {
        let mut changed = 0;
        let mut f = first;
        while f < last {
            let w = (f / 64) as usize;
            let lo = f % 64;
            let hi = (last - (f - lo)).min(64);
            let mask = if hi - lo == 64 { !0 } else { ((1u64 << (hi - lo)) - 1) << lo };
            let old = self.bits[w];
            self.bits[w] = if used { old | mask } else { old & !mask };
            changed += ((old ^ self.bits[w]) & mask).count_ones() as u64;
            f += hi - lo;
        }
        changed
    }

    /// Marks [start, end) free; partial frames at the edges stay in use. Frames past the bitmap are ignored.
    pub fn release(&mut self, start: u64, end: u64) {
        let first = start.div_ceil(FRAME);
        let last = (end / FRAME).min(self.frames);
        if first < last {
            self.free += self.fill(first, last, false);
        }
    }

    /// Marks [start, end) in use, whole frames covering it.
    pub fn reserve(&mut self, start: u64, end: u64) {
        let first = start / FRAME;
        let last = end.div_ceil(FRAME).min(self.frames);
        if first < last {
            self.free -= self.fill(first, last, true);
        }
    }

    /// `count` contiguous frames aligned to `align` bytes (a power of two, at least 4096), ending at or below
    /// `ceiling` (exclusive; `u64::MAX` for none). Returns the physical address.
    pub fn alloc(&mut self, count: u64, align: u64, ceiling: u64) -> Option<u64> {
        assert!(align.is_power_of_two() && align >= FRAME && count > 0);
        let step = align / FRAME;
        let top = (ceiling / FRAME).min(self.frames);
        // Two passes: from the hint, then from 0.
        for from in [self.hint, 0] {
            let mut f = from.next_multiple_of(step);
            while f + count <= top {
                match (f..f + count).find(|&g| self.is_used(g)) {
                    None => {
                        for g in f..f + count {
                            self.set(g, true);
                        }
                        self.free -= count;
                        self.hint = f + count;
                        return Some(f * FRAME);
                    }
                    Some(used) => f = (used + 1).next_multiple_of(step),
                }
            }
        }
        None
    }

    pub fn free(&mut self, addr: u64, count: u64) {
        let f = addr / FRAME;
        for g in f..f + count {
            assert!(self.is_used(g), "double free of frame {g:#x}");
            self.set(g, false);
        }
        self.free += count;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn alloc_align_ceiling_free() {
        let mut bits = [0u64; Frames::words_for(64 << 20)];
        let mut fr = Frames::new(&mut bits);
        assert_eq!(fr.free_frames(), 0);
        fr.release(0x1000, 0x9f000);
        fr.release(0x10_0000 + 1, 64 << 20); // the partial first frame stays used
        let free0 = fr.free_frames();
        assert_eq!(free0, 0x9e + (64 << 20) / FRAME - 0x101);
        // Below 1 MiB.
        let low = fr.alloc(1, FRAME, 0x10_0000).unwrap();
        assert!(low < 0x10_0000 && low >= 0x1000);
        // 2 MiB aligned.
        let big = fr.alloc(512, 2 << 20, u64::MAX).unwrap();
        assert_eq!(big % (2 << 20), 0);
        assert_eq!(fr.free_frames(), free0 - 513);
        fr.free(big, 512);
        fr.free(low, 1);
        assert_eq!(fr.free_frames(), free0);
        // Reserving a free range takes it out; reserving twice is harmless.
        fr.reserve(0x20_0000, 0x40_0000);
        fr.reserve(0x20_0000, 0x40_0000);
        assert_eq!(fr.free_frames(), free0 - 512);
        // Nothing that big.
        assert_eq!(fr.alloc(1 << 20, FRAME, u64::MAX), None);
        // Word-straddling ranges.
        fr.reserve(0x3f000, 0x41000 + 64 * FRAME);
        assert_eq!(fr.free_frames(), free0 - 512 - 66);
        fr.release(0x3f000, 0x41000 + 64 * FRAME);
        assert_eq!(fr.free_frames(), free0 - 512);
        // Exhaust and recover.
        let mut n = 0;
        while fr.alloc(1, FRAME, u64::MAX).is_some() {
            n += 1;
        }
        assert_eq!(n, free0 - 512);
        assert_eq!(fr.free_frames(), 0);
    }
}
