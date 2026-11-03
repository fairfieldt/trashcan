//! Memory objects: frames a process can map and hand to devices. Anonymous memory is pinned and zeroed when
//! created (no paging, no lazy allocation); device memory is a physical range outside RAM, mapped uncached.

use alloc::vec::Vec;
use alloc::sync::Arc;
use crate::account::{Account, Charge};
use dross_abi::Error;

use crate::mem;

const PAGE: u64 = 4096;

enum Kind {
    /// Owned frames, one per page.
    Pages(Vec<u64>),
    /// Linux anonymous ranges: independent page ownership permits partial unmap.
    SharedPages(Vec<Arc<OwnedPage>>),
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

#[cfg(feature = "unix")]
pub struct PageCopies(alloc::collections::BTreeMap<u64, Arc<OwnedPage>>);

#[cfg(feature = "unix")]
impl PageCopies {
    pub fn new() -> Self { Self(alloc::collections::BTreeMap::new()) }
}

struct OwnedPage {
    phys: u64,
    _charge: Charge,
}

impl Drop for OwnedPage {
    fn drop(&mut self) { mem::free_phys(self.phys, 1); }
}

pub struct Memory {
    size: u64,
    kind: Kind,
    writable: bool,
    _charge: Option<Charge>,
}

/// Reclaim only cache-owned references, outside the frame allocator lock.
fn allocate(pages: usize, align: u64, ceiling: u64) -> Option<u64> {
    if let Some(p) = mem::alloc_phys(pages, align, ceiling) { return Some(p); }
    if reclaim_memory() { return mem::alloc_phys(pages, align, ceiling); }
    None
}

fn reclaim_memory() -> bool {
    let mut reclaimed = crate::ipc::collect_cycles() != 0;
    reclaimed |= crate::linux::reclaim_socket_cycles() != 0;
    #[cfg(feature = "unix")]
    { reclaimed |= crate::linux::unix::vfs::reclaim_content_cache() != 0; }
    reclaimed
}
fn ensure_free(bytes: u64) -> Result<(), Error> {
    if bytes > mem::free_bytes() { reclaim_memory(); }
    if bytes > mem::free_bytes() { Err(Error::NoMemory) } else { Ok(()) }
}

impl Memory {
    pub fn anonymous(size: u64, contiguous: bool, below_4g: bool) -> Result<Memory, Error> {
        Self::anonymous_in(size, contiguous, below_4g, crate::account::current())
    }

    pub fn anonymous_in(size: u64, contiguous: bool, below_4g: bool, account: Arc<Account>) -> Result<Memory, Error> {
        if size == 0 || size % PAGE != 0 || size > 1 << 40 {
            return Err(Error::InvalidArgs);
        }
        let charge = account.reserve(size)?;
        ensure_free(size)?;
        let pages = (size / PAGE) as usize;
        let ceiling = if below_4g { 1 << 32 } else { u64::MAX };
        let kind = if contiguous {
            let align = if size >= 2 << 20 { 2 << 20 } else { PAGE };
            let p = allocate(pages, align, ceiling).ok_or(Error::NoMemory)?;
            mem::zero_phys(p, pages);
            Kind::Contiguous(p)
        } else {
            let mut v = Vec::new();
            v.try_reserve_exact(pages).map_err(|_| Error::NoMemory)?;
            ensure_free(size)?;
            for _ in 0..pages {
                match allocate(1, PAGE, ceiling) {
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
        Ok(Memory { size, kind, writable: true, _charge: Some(charge) })
    }

    /// Anonymous Linux memory can release pages independently when a mapping splits.
    /// Explicit handles to the original object still retain that object's full backing.
    pub fn anonymous_paged(size: u64) -> Result<Memory, Error> {
        Self::anonymous_paged_in(size, crate::account::current())
    }

    pub fn anonymous_paged_in(size: u64, account: Arc<Account>) -> Result<Memory, Error> {
        let mut memory = Self::anonymous_in(size, false, false, account)?;
        let mut pages = Vec::new();
        pages.try_reserve_exact((size / PAGE) as usize).map_err(|_| Error::NoMemory)?;
        let Kind::Pages(frames) = core::mem::replace(&mut memory.kind, Kind::Pages(Vec::new())) else { unreachable!() };
        let mut charge = memory._charge.take().unwrap();
        for phys in frames { pages.push(Arc::new(OwnedPage { phys, _charge: charge.split(PAGE) })); }
        memory.kind = Kind::SharedPages(pages);
        Ok(memory)
    }

    /// Eager fork copies each physical page once, preserving aliases across views.
    #[cfg(feature = "unix")]
    pub fn copy_for_fork(&self, account: Arc<Account>, copies: &mut PageCopies) -> Result<Memory, Error> {
        if !self.is_ram() { return Err(Error::AccessDenied); }
        let mut pages = Vec::new();
        pages.try_reserve_exact((self.size / PAGE) as usize).map_err(|_| Error::NoMemory)?;
        for index in 0..self.size / PAGE {
            let source = self.page_phys(index as usize);
            let page = if let Some(page) = copies.0.get(&source) { page.clone() } else {
                let charge = account.reserve(PAGE)?;
                let phys = allocate(1, PAGE, u64::MAX).ok_or(Error::NoMemory)?;
                // SAFETY: both frames are resident RAM, the destination is fresh.
                unsafe { core::ptr::copy_nonoverlapping(mem::virt(source) as *const u8, mem::virt(phys) as *mut u8, PAGE as usize) };
                let page = Arc::new(OwnedPage { phys, _charge: charge });
                copies.0.insert(source, page.clone());
                page
            };
            pages.push(page);
        }
        Ok(Memory { size: self.size, kind: Kind::SharedPages(pages), writable: self.writable, _charge: None })
    }

    /// Returns an independent view for range-owned backing; explicit native objects
    /// retain their original whole-object ownership contract.
    pub fn slice(&self, offset: u64, len: u64) -> Result<Option<Memory>, Error> {
        if offset % PAGE != 0 || len % PAGE != 0 || offset.checked_add(len).is_none_or(|end| end > self.size) {
            return Err(Error::OutOfRange);
        }
        let Kind::SharedPages(pages) = &self.kind else { return Ok(None) };
        let start = (offset / PAGE) as usize;
        let end = ((offset + len) / PAGE) as usize;
        let mut view = Vec::new();
        view.try_reserve_exact(end - start).map_err(|_| Error::NoMemory)?;
        view.extend(pages[start..end].iter().cloned());
        Ok(Some(Memory { size: len, kind: Kind::SharedPages(view), writable: self.writable, _charge: None }))
    }

    /// A boot module: read-only, its frames stay where the loader put them.
    pub fn module(phys: u64, size: u64) -> Memory {
        Memory { size: size.next_multiple_of(PAGE), kind: Kind::Borrowed(phys), writable: false, _charge: None }
    }

    pub fn device(phys: u64, size: u64) -> Memory {
        Memory { size, kind: Kind::Device(phys), writable: true, _charge: None }
    }

    pub fn display(phys: u64, size: u64) -> Memory {
        Memory { size, kind: Kind::Display(phys), writable: true, _charge: None }
    }

    /// The same memory, no longer writable (a filesystem's file contents).
    #[cfg(feature = "unix")]
    pub fn read_only(mut self) -> Memory {
        self.writable = false;
        self
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
            Kind::SharedPages(v) => v[i].phys,
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

    /// Copies `len` bytes from `src` at `src_off` to `off`, page by page (no buffer the size of the copy).
    pub fn copy_from(&self, off: u64, src: &Memory, src_off: u64, len: u64) -> Result<(), Error> {
        if !matches!(self.kind, Kind::Pages(_) | Kind::SharedPages(_) | Kind::Contiguous(_)) || !src.is_ram() {
            return Err(Error::WrongType);
        }
        if off.checked_add(len).is_none_or(|e| e > self.size) || src_off.checked_add(len).is_none_or(|e| e > src.size) {
            return Err(Error::OutOfRange);
        }
        let mut done = 0;
        while done < len {
            let (d, s) = (off + done, src_off + done);
            let n = (PAGE - d % PAGE).min(PAGE - s % PAGE).min(len - done);
            let from = mem::virt(src.page_phys((s / PAGE) as usize) + s % PAGE) as *const u8;
            let to = mem::virt(self.page_phys((d / PAGE) as usize) + d % PAGE) as *mut u8;
            // SAFETY: inside one page of each object, in the direct map; distinct objects or ranges.
            unsafe { core::ptr::copy(from, to, n as usize) };
            done += n;
        }
        Ok(())
    }

    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), Error> {
        if !matches!(self.kind, Kind::Pages(_) | Kind::SharedPages(_) | Kind::Contiguous(_)) {
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
            Kind::SharedPages(_) | Kind::Borrowed(_) | Kind::Device(_) | Kind::Display(_) => {}
        }
    }
}
