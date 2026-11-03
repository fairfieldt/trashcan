//! Memory objects: frames a process can map and hand to devices. Anonymous memory is pinned and zeroed when
//! created (no paging, no lazy allocation); device memory is a physical range outside RAM, mapped uncached.

use alloc::vec::Vec;
use dross_abi::Error;

use crate::mem;

const PAGE: u64 = 4096;

enum Kind {
    /// Owned frames, one per page.
    Pages(Vec<u64>),
    /// Owned contiguous frames.
    Contiguous(u64),
    /// Frames the kernel doesn't own (boot modules): never freed.
    Borrowed(u64),
    /// A device's registers or other non-RAM range.
    Device(u64),
    /// The firmware's framebuffer: mapped with the caching the firmware's MTRRs give it (write-combining on the
    /// Mac Pro), and while mapped anywhere the kernel log stays off it (`fbcon::Claim`).
    Display(u64),
}

pub struct Memory {
    size: u64,
    kind: Kind,
    writable: bool,
}

impl Memory {
    pub fn anonymous(size: u64, contiguous: bool, below_4g: bool) -> Result<Memory, Error> {
        if size == 0 || size % PAGE != 0 || size > 1 << 40 {
            return Err(Error::InvalidArgs);
        }
        let pages = (size / PAGE) as usize;
        let ceiling = if below_4g { 1 << 32 } else { u64::MAX };
        let kind = if contiguous {
            let align = if size >= 2 << 20 { 2 << 20 } else { PAGE };
            let p = mem::alloc_phys(pages, align, ceiling).ok_or(Error::NoMemory)?;
            mem::zero_phys(p, pages);
            Kind::Contiguous(p)
        } else {
            let mut v = Vec::new();
            v.try_reserve_exact(pages).map_err(|_| Error::NoMemory)?;
            for _ in 0..pages {
                match mem::alloc_phys(1, PAGE, ceiling) {
                    Some(p) => {
                        mem::zero_phys(p, 1);
                        v.push(p);
                    }
                    None => {
                        for p in v {
                            mem::free_phys(p, 1);
                        }
                        return Err(Error::NoMemory);
                    }
                }
            }
            Kind::Pages(v)
        };
        Ok(Memory { size, kind, writable: true })
    }

    /// A boot module: read-only, its frames stay where the loader put them.
    pub fn module(phys: u64, size: u64) -> Memory {
        Memory { size: size.next_multiple_of(PAGE), kind: Kind::Borrowed(phys), writable: false }
    }

    pub fn device(phys: u64, size: u64) -> Memory {
        Memory { size, kind: Kind::Device(phys), writable: true }
    }

    pub fn display(phys: u64, size: u64) -> Memory {
        Memory { size, kind: Kind::Display(phys), writable: true }
    }

    pub fn is_display(&self) -> bool {
        matches!(self.kind, Kind::Display(_))
    }

    /// RAM the kernel can copy (not device registers or the framebuffer).
    pub fn is_ram(&self) -> bool {
        !matches!(self.kind, Kind::Device(_) | Kind::Display(_))
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn writable(&self) -> bool {
        self.writable
    }

    pub fn uncached(&self) -> bool {
        matches!(self.kind, Kind::Device(_))
    }

    pub fn page_phys(&self, i: usize) -> u64 {
        match &self.kind {
            Kind::Pages(v) => v[i],
            Kind::Contiguous(p) | Kind::Borrowed(p) | Kind::Device(p) | Kind::Display(p) => p + i as u64 * PAGE,
        }
    }

    /// Copies out of RAM-backed memory (the kernel reads ELF images this way).
    pub fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), Error> {
        if !self.is_ram() {
            return Err(Error::WrongType);
        }
        if offset.checked_add(buf.len() as u64).is_none_or(|e| e > self.size) {
            return Err(Error::OutOfRange);
        }
        let mut done = 0;
        while done < buf.len() {
            let at = offset + done as u64;
            let in_page = (PAGE - at % PAGE) as usize;
            let n = in_page.min(buf.len() - done);
            let src = mem::virt(self.page_phys((at / PAGE) as usize) + at % PAGE) as *const u8;
            // SAFETY: inside one page of this object, in the direct map.
            unsafe { core::ptr::copy_nonoverlapping(src, buf[done..].as_mut_ptr(), n) };
            done += n;
        }
        Ok(())
    }

    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), Error> {
        if !matches!(self.kind, Kind::Pages(_) | Kind::Contiguous(_)) {
            return Err(Error::WrongType);
        }
        if offset.checked_add(data.len() as u64).is_none_or(|e| e > self.size) {
            return Err(Error::OutOfRange);
        }
        let mut done = 0;
        while done < data.len() {
            let at = offset + done as u64;
            let n = ((PAGE - at % PAGE) as usize).min(data.len() - done);
            let dst = mem::virt(self.page_phys((at / PAGE) as usize) + at % PAGE) as *mut u8;
            // SAFETY: as in `read`.
            unsafe { core::ptr::copy_nonoverlapping(data[done..].as_ptr(), dst, n) };
            done += n;
        }
        Ok(())
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        match &self.kind {
            Kind::Pages(v) => v.iter().for_each(|&p| mem::free_phys(p, 1)),
            Kind::Contiguous(p) => mem::free_phys(*p, (self.size / PAGE) as usize),
            Kind::Borrowed(_) | Kind::Device(_) | Kind::Display(_) => {}
        }
    }
}
