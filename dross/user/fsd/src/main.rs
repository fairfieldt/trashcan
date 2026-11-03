//! fsd serves ext4 through lwext4 in its own process. USB block reads, writes
//! and flushes go through an explicitly supplied channel. Mounts default to
//! read-only; `%writable=1` enables mutation on a supported journaled filesystem.
#![no_std]
#![no_main]

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use dross_abi::vm;
use dross_rt::bootstrap::Bootstrap;
use dross_rt::{Handle, println, sys};
mod lwext4;
use alloc::ffi::CString;
use core::sync::atomic::{AtomicPtr, Ordering};
use lwext4::Stat;

dross_rt::main!(main);

/// GPT's Linux filesystem data partition type, as stored on disk.
const LINUX_FS: [u8; 16] = [
    0xaf, 0x3d, 0xc6, 0x0f, 0x83, 0x84, 0x72, 0x47, 0x8e, 0x79, 0x3d, 0x69, 0xd8, 0x47, 0x7d, 0xe4,
];
/// Listing replies stay under a channel message.
const REPLY_MAX: usize = 60 * 1024;

/// The block device: reads land in its window, which doubles as a read-ahead cache of the last range read.
struct Disk {
    ch: Handle,
    window: &'static mut [u8],
    block_size: u64,
    blocks: u64,
    /// The cached range, in blocks.
    cached: Option<(u64, u64)>,
    /// The partition's first byte and its length.
    start: u64,
    len: u64,
    failed: bool,
}

impl Disk {
    fn open(ch: Handle) -> Result<Disk, i64> {
        let mut reply = [0u8; 20];
        let mut hs = [Handle(0); 1];
        // The USB driver may still be looking for the disk (no device: -19): ask again for 10 s.
        let mut tries = 0;
        let (n, nh, status) = loop {
            sys::channel_write(ch, &[0], &[]).map_err(|_| -5)?;
            let (n, nh) =
                sys::channel_read(ch, &mut reply, &mut hs, dross_rt::FOREVER).map_err(|_| -5)?;
            let status = i64::from_le_bytes(reply[..8].try_into().unwrap());
            tries += 1;
            if status != -19 || tries == 50 {
                break (n, nh, status);
            }
            let _ = sys::notify_wait(sys::notify_create().map_err(|_| -5)?, sys::after_ms(200));
        };
        if status < 0 || n < 20 || nh != 1 {
            return Err(if status < 0 { status } else { -5 });
        }
        let block_size = u32::from_le_bytes(reply[8..12].try_into().unwrap()) as u64;
        let blocks = u64::from_le_bytes(reply[12..20].try_into().unwrap());
        let size = sys::mem_size(hs[0]).map_err(|_| -5)?;
        let at = sys::vm_map(hs[0], 0, size, 0, vm::READ | vm::WRITE).map_err(|_| -12)?;
        // SAFETY: a mapping of the device's window that stays for the life of the program.
        let window = unsafe { core::slice::from_raw_parts_mut(at as *mut u8, size as usize) };
        Ok(Disk {
            ch,
            window,
            block_size,
            blocks,
            cached: None,
            start: 0,
            len: block_size.checked_mul(blocks).ok_or(-22)?,
            failed: false,
        })
    }

    /// Bytes from the disk (not the partition) at `at`.
    fn read_disk(&mut self, mut at: u64, mut dst: &mut [u8]) -> Result<(), i64> {
        let per_window = self.window.len() as u64 / self.block_size;
        while !dst.is_empty() {
            let lba = at / self.block_size;
            let hit = self
                .cached
                .filter(|&(first, count)| lba >= first && lba < first + count);
            let (first, count) = match hit {
                Some(c) => c,
                None => {
                    let count = per_window.min(self.blocks.saturating_sub(lba));
                    if count == 0 {
                        return Err(-22);
                    }
                    let mut req = [0u8; 13];
                    req[0] = 1;
                    req[1..9].copy_from_slice(&lba.to_le_bytes());
                    req[9..13].copy_from_slice(&(count as u32).to_le_bytes());
                    self.cached = None;
                    sys::channel_write(self.ch, &req, &[]).map_err(|_| -5)?;
                    let mut reply = [0u8; 8];
                    sys::channel_read(self.ch, &mut reply, &mut [], dross_rt::FOREVER)
                        .map_err(|_| -5)?;
                    let status = i64::from_le_bytes(reply);
                    if status < 0 {
                        return Err(status);
                    }
                    self.cached = Some((lba, count));
                    (lba, count)
                }
            };
            let off = (at - first * self.block_size) as usize;
            let n = dst.len().min((count * self.block_size) as usize - off);
            dst[..n].copy_from_slice(&self.window[off..off + n]);
            dst = &mut dst[n..];
            at += n as u64;
        }
        Ok(())
    }

    /// Picks the partition: number `want` (from 1), else the first Linux filesystem; the whole disk without GPT.
    fn partition(&mut self, want: Option<u64>) -> Result<(), i64> {
        let mut header = alloc::vec![0u8; 92];
        self.read_disk(self.block_size, &mut header)?;
        if &header[..8] != b"EFI PART" {
            println!("no GPT: the filesystem is the whole disk");
            return Ok(());
        }
        let entries = u64::from_le_bytes(header[72..80].try_into().unwrap());
        let count = u32::from_le_bytes(header[80..84].try_into().unwrap()) as u64;
        let size = u32::from_le_bytes(header[84..88].try_into().unwrap()) as u64;
        if !(128..=4096).contains(&size) || !size.is_multiple_of(128) {
            return Err(-22);
        }
        let entries_at = entries.checked_mul(self.block_size).ok_or(-22)?;
        let mut e = alloc::vec![0u8; size as usize];
        for i in 0..count.min(128) {
            self.read_disk(entries_at.checked_add(i * size).ok_or(-22)?, &mut e)?;
            if e[..16].iter().all(|&b| b == 0) {
                continue;
            }
            let chosen = match want {
                Some(n) => n == i + 1,
                None => e[..16] == LINUX_FS,
            };
            if chosen {
                let first = u64::from_le_bytes(e[32..40].try_into().unwrap());
                let last = u64::from_le_bytes(e[40..48].try_into().unwrap());
                if first > last || last >= self.blocks {
                    return Err(-22);
                }
                self.start = first.checked_mul(self.block_size).ok_or(-22)?;
                self.len = (last - first + 1).checked_mul(self.block_size).ok_or(-22)?;
                println!("partition {}: {} MiB", i + 1, self.len >> 20);
                return Ok(());
            }
        }
        Err(-2)
    }
}

static DISK: AtomicPtr<Disk> = AtomicPtr::new(core::ptr::null_mut());
impl Disk {
    fn command(&mut self, op: u8, lba: u64, count: u32) -> Result<(), i64> {
        let mut request = alloc::vec![op];
        if op != 3 {
            request.extend_from_slice(&lba.to_le_bytes());
            request.extend_from_slice(&count.to_le_bytes());
        }
        sys::channel_write(self.ch, &request, &[]).map_err(|_| 5)?;
        let mut reply = [0u8; 8];
        let (n, _) =
            sys::channel_read(self.ch, &mut reply, &mut [], dross_rt::FOREVER).map_err(|_| 5)?;
        if n != 8 {
            return Err(5);
        }
        let e = i64::from_le_bytes(reply);
        if e < 0 { Err(-e) } else { Ok(()) }
    }
    fn blocks(
        &mut self,
        write: bool,
        lba: u64,
        count: u32,
        buf: *mut u8,
        len: usize,
    ) -> Result<(), i64> {
        let size = self.block_size;
        let end = lba
            .checked_add(count as u64)
            .and_then(|v| v.checked_mul(size))
            .ok_or(22)?;
        if end > self.len || len != (count as u64 * size) as usize {
            return Err(22);
        }
        if write && self.failed {
            return Err(5);
        }
        self.cached = None;
        let mut block = lba + self.start / size;
        let mut at = 0;
        while at < len {
            let n = (len - at).min(self.window.len());
            let count = (n / size as usize) as u32;
            if write {
                // SAFETY: lwext4 supplies a readable buffer of `len` bytes.
                // Do not create a mutable slice for write-source memory.
                unsafe { core::ptr::copy_nonoverlapping(buf.add(at), self.window.as_mut_ptr(), n) };
            }
            if let Err(e) = self.command(if write { 2 } else { 1 }, block, count) {
                if write {
                    self.failed = true
                }
                return Err(e);
            }
            if !write {
                // SAFETY: the read callback supplies an exclusive output buffer.
                unsafe { core::ptr::copy_nonoverlapping(self.window.as_ptr(), buf.add(at), n) };
            }
            at += n;
            block += count as u64;
        }
        Ok(())
    }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn fsd_blocks(write: i32, at: u64, count: u32, buf: *mut u8) -> i32 {
    let disk = DISK.load(Ordering::Relaxed);
    if disk.is_null() {
        return 19;
    }
    let d = unsafe { &mut *disk };
    let Some(len) = (count as usize).checked_mul(d.block_size as usize) else {
        return 22;
    };
    d.blocks(write != 0, at, count, buf, len).err().unwrap_or(0) as i32
}
#[unsafe(no_mangle)]
unsafe extern "C" fn fsd_flush() -> i32 {
    let disk = DISK.load(Ordering::Relaxed);
    if disk.is_null() {
        return 19;
    }
    unsafe { &mut *disk }.command(3, 0, 0).err().unwrap_or(0) as i32
}
#[unsafe(no_mangle)]
unsafe extern "C" fn fsd_integrity(file: *const core::ffi::c_char, line: u32) {
    let file = unsafe { core::ffi::CStr::from_ptr(file) }
        .to_str()
        .unwrap_or("unknown");
    println!("metadata checksum failed at {}:{}", file, line);
}
fn check(code: i32) -> Result<(), i64> {
    let code = if code == 0 {
        unsafe { lwext4::oscar_error() }
    } else {
        code
    };
    if code == 0 {
        Ok(())
    } else {
        Err(-(code as i64))
    }
}
fn path(bytes: &[u8]) -> Result<CString, i64> {
    let s = core::str::from_utf8(bytes).map_err(|_| -22)?;
    if !s.starts_with('/') || s.split('/').any(|v| v == "..") || bytes.len() > 4096 {
        return Err(-22);
    }
    CString::new(bytes).map_err(|_| -22)
}
fn stat(p: &CString) -> Result<Stat, i64> {
    let mut s = Stat::default();
    check(unsafe { lwext4::oscar_stat(p.as_ptr(), &mut s) })?;
    Ok(s)
}
fn list(p: &CString, cookie: u64) -> Result<Vec<u8>, i64> {
    check(unsafe { lwext4::oscar_dir_open(p.as_ptr()) })?;
    let result = (|| {
        let mut out = 0i64.to_le_bytes().to_vec();
        out.extend_from_slice(&0u64.to_le_bytes());
        let mut i = 0u64;
        loop {
            let next = unsafe { lwext4::oscar_dir_next() };
            if next.is_null() {
                break;
            }
            let e = unsafe { &*next };
            let name = &e.name[..e.name_length as usize];
            if name == b"." || name == b".." {
                continue;
            }
            if i < cookie {
                i += 1;
                continue;
            }
            let full = CString::new(alloc::format!(
                "{}/{}",
                p.to_str().map_err(|_| -22)?.trim_end_matches('/'),
                core::str::from_utf8(name).map_err(|_| -22)?
            ))
            .map_err(|_| -22)?;
            let s = stat(&full)?;
            let mut target = Vec::new();
            if s.mode & 0o170000 == 0o120000 {
                target.resize(4096, 0);
                let mut n = 0;
                check(unsafe {
                    lwext4::oscar_readlink(
                        full.as_ptr(),
                        target.as_mut_ptr().cast(),
                        target.len(),
                        &mut n,
                    )
                })?;
                target.truncate(n);
            }
            if out.len() + 40 + name.len() + target.len() > REPLY_MAX {
                out[8..16].copy_from_slice(&i.to_le_bytes());
                break;
            }
            out.extend_from_slice(&s.mode.to_le_bytes());
            out.extend_from_slice(&s.uid.to_le_bytes());
            out.extend_from_slice(&s.gid.to_le_bytes());
            out.extend_from_slice(&s.size.to_le_bytes());
            out.extend_from_slice(&(s.mtime as u64 * 1_000_000_000).to_le_bytes());
            out.extend_from_slice(&(e.inode as u64).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&(target.len() as u16).to_le_bytes());
            out.extend_from_slice(name);
            out.extend_from_slice(&target);
            i += 1;
        }
        Ok(out)
    })();
    let closed = check(unsafe { lwext4::oscar_dir_close() });
    result.and_then(|v| closed.map(|_| v))
}
fn file_info(fd: i32) -> Result<lwext4::Info, i64> {
    let mut info = lwext4::Info::default();
    check(unsafe { lwext4::oscar_fd_info(fd, &mut info) })?;
    Ok(info)
}
fn encode_info(info: lwext4::Info, out: &mut Vec<u8>) {
    for v in [info.inode, info.size, info.atime, info.mtime, info.ctime] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in [info.mode, info.uid, info.gid, info.links] {
        out.extend_from_slice(&v.to_le_bytes());
    }
}
fn snapshot_file(fd: i32) -> Result<(Vec<u8>, Handle), i64> {
    let size = file_info(fd)?.size;
    let bytes = (size.checked_add(4095).ok_or(-27)? & !4095).max(4096);
    let mem = sys::mem_create(bytes, 0).map_err(|_| -12)?;
    let at = match sys::vm_map(mem, 0, bytes, 0, vm::READ | vm::WRITE) {
        Ok(at) => at,
        Err(_) => {
            let _ = sys::close(mem);
            return Err(-12);
        }
    };
    let mut off = 0;
    let mut result = Ok(());
    while off < size {
        let n = (size - off).min(1 << 20) as usize;
        let mut done = 0;
        if let Err(e) =
            check(unsafe { lwext4::oscar_fd_read(fd, off, (at + off) as *mut _, n, &mut done) })
        {
            result = Err(e);
            break;
        }
        if done != n {
            result = Err(-5);
            break;
        }
        off += n as u64;
    }
    let unmap = sys::vm_unmap(at, bytes).map_err(|_| -5);
    if let Err(e) = result.and(unmap) {
        let _ = sys::close(mem);
        return Err(e);
    }
    let mut out = 0i64.to_le_bytes().to_vec();
    out.extend_from_slice(&size.to_le_bytes());
    Ok((out, mem))
}
fn snapshot(p: &CString) -> Result<(Vec<u8>, Handle), i64> {
    let mut fd = 0;
    check(unsafe { lwext4::oscar_open(p.as_ptr(), 0, &mut fd) })?;
    let result = snapshot_file(fd);
    let closed = check(unsafe { lwext4::oscar_close(fd) });
    match (result, closed) {
        (Ok((_, h)), Err(e)) => {
            let _ = sys::close(h);
            Err(e)
        }
        (Ok(v), Ok(())) => Ok(v),
        (Err(e), _) => Err(e),
    }
}
fn handle_file(req: &[u8], fd: i32) -> Result<(Vec<u8>, Option<Handle>), i64> {
    let u32at = |at: usize| -> Result<u32, i64> {
        Ok(u32::from_le_bytes(
            req.get(at..at + 4).ok_or(-22)?.try_into().unwrap(),
        ))
    };
    let u64at = |at: usize| -> Result<u64, i64> {
        Ok(u64::from_le_bytes(
            req.get(at..at + 8).ok_or(-22)?.try_into().unwrap(),
        ))
    };
    let mut out = 0i64.to_le_bytes().to_vec();
    match u32at(0)? {
        1 => {
            let len = (u32at(12)? as usize).min(60 * 1024);
            out.resize(8 + len, 0);
            let mut n = 0;
            check(unsafe {
                lwext4::oscar_fd_read(fd, u64at(4)?, out[8..].as_mut_ptr().cast(), len, &mut n)
            })?;
            out.truncate(8 + n);
        }
        2 => {
            let data = req.get(12..).ok_or(-22)?;
            let (mut n, mut end) = (0, 0);
            check(unsafe {
                lwext4::oscar_fd_write(
                    fd,
                    u64at(4)?,
                    data.as_ptr().cast(),
                    data.len(),
                    &mut n,
                    &mut end,
                )
            })?;
            out.extend_from_slice(&end.to_le_bytes());
            out.extend_from_slice(&(n as u64).to_le_bytes());
        }
        3 => encode_info(file_info(fd)?, &mut out),
        4 => check(unsafe { lwext4::oscar_fd_truncate(fd, u64at(4)?) })?,
        5 => check(unsafe { lwext4::oscar_fd_setattr(fd, u32at(4)?, u32at(8)?, u32at(12)?) })?,
        6 => check(unsafe { lwext4::oscar_fd_times(fd, u32at(4)?, u64at(8)?, u64at(16)?) })?,
        7 => check(unsafe { lwext4::oscar_sync() })?,
        8 => {
            let (out, h) = snapshot_file(fd)?;
            return Ok((out, Some(h)));
        }
        _ => return Err(-22),
    }
    Ok((out, None))
}
fn handle(req: &[u8], readonly: bool) -> Result<(Vec<u8>, Option<Handle>), i64> {
    let u32at = |at: usize| -> Result<u32, i64> {
        Ok(u32::from_le_bytes(
            req.get(at..at + 4).ok_or(-22)?.try_into().unwrap(),
        ))
    };
    let u64at = |at: usize| -> Result<u64, i64> {
        Ok(u64::from_le_bytes(
            req.get(at..at + 8).ok_or(-22)?.try_into().unwrap(),
        ))
    };
    let op = u32at(0)?;
    if readonly && matches!(op, 3..=8 | 12 | 13 | 16) {
        return Err(-30);
    }
    let mut out = 0i64.to_le_bytes().to_vec();
    match op {
        1 => {
            let p = path(req.get(12..).ok_or(-22)?)?;
            return Ok((list(&p, u64at(4)?)?, None));
        }
        2 => {
            let p = path(req.get(4..).ok_or(-22)?)?;
            let (out, h) = snapshot(&p)?;
            return Ok((out, Some(h)));
        }
        3 => {
            let p = path(req.get(16..).ok_or(-22)?)?;
            check(unsafe { lwext4::oscar_create(p.as_ptr(), u32at(4)?, u32at(8)?, u32at(12)?) })?;
        }
        4 => {
            let p = path(req.get(12..).ok_or(-22)?)?;
            check(unsafe { lwext4::oscar_truncate(p.as_ptr(), u64at(4)?) })?;
        }
        5 => {
            let length = u32at(12)? as usize;
            let p = path(req.get(16..16 + length).ok_or(-22)?)?;
            let data = req.get(16 + length..).ok_or(-22)?;
            let mut n = 0;
            let offset = if u64at(4)? == u64::MAX {
                stat(&p)?.size
            } else {
                u64at(4)?
            };
            check(unsafe {
                lwext4::oscar_write(p.as_ptr(), offset, data.as_ptr().cast(), data.len(), &mut n)
            })?;
            out.extend_from_slice(&(offset + n as u64).to_le_bytes());
            out.extend_from_slice(&(n as u64).to_le_bytes());
        }
        6 => {
            let p = path(req.get(8..).ok_or(-22)?)?;
            check(unsafe { lwext4::oscar_remove(p.as_ptr(), u32at(4)? as i32) })?;
        }
        7 => {
            let length = u32at(8)? as usize;
            let a = path(req.get(12..12 + length).ok_or(-22)?)?;
            let b = path(req.get(12 + length..).ok_or(-22)?)?;
            check(unsafe { lwext4::oscar_rename(a.as_ptr(), b.as_ptr(), u32at(4)? as i32) })?;
        }
        8 => {
            let length = u32at(4)? as usize;
            let p = path(req.get(8..8 + length).ok_or(-22)?)?;
            let target = CString::new(req.get(8 + length..).ok_or(-22)?).map_err(|_| -22)?;
            check(unsafe { lwext4::oscar_symlink(p.as_ptr(), target.as_ptr()) })?;
        }
        9 => {
            check(unsafe { lwext4::oscar_sync() })?;
        }
        10 => {
            let p = path(req.get(16..).ok_or(-22)?)?;
            let len = (u32at(12)? as usize).min(60 * 1024);
            out.resize(8 + len, 0);
            let mut n = 0;
            check(unsafe {
                lwext4::oscar_read(
                    p.as_ptr(),
                    u64at(4)?,
                    out[8..].as_mut_ptr().cast(),
                    len,
                    &mut n,
                )
            })?;
            out.truncate(8 + n);
        }
        11 => {
            out.push(!readonly as u8);
        }
        12 => {
            let p = path(req.get(16..).ok_or(-22)?)?;
            check(unsafe { lwext4::oscar_setattr(p.as_ptr(), u32at(4)?, u32at(8)?, u32at(12)?) })?;
        }
        13 => {
            let p = path(req.get(24..).ok_or(-22)?)?;
            let mut fd = 0;
            check(unsafe { lwext4::oscar_open(p.as_ptr(), 0, &mut fd) })?;
            let e = check(unsafe { lwext4::oscar_fd_times(fd, u32at(4)?, u64at(8)?, u64at(16)?) });
            let closed = check(unsafe { lwext4::oscar_close(fd) });
            e.and(closed)?;
        }
        15 => {
            let p = path(req.get(4..).ok_or(-22)?)?;
            let mut info = lwext4::Info::default();
            check(unsafe { lwext4::oscar_info(p.as_ptr(), &mut info) })?;
            encode_info(info, &mut out);
        }
        16 => {
            let length = u32at(4)? as usize;
            let a = path(req.get(8..8 + length).ok_or(-22)?)?;
            let b = path(req.get(8 + length..).ok_or(-22)?)?;
            check(unsafe { lwext4::oscar_link(a.as_ptr(), b.as_ptr()) })?;
        }
        _ => return Err(-22),
    }
    Ok((out, None))
}
fn main(boot: Handle) -> i32 {
    let b = Bootstrap::read(boot).expect("bootstrap");
    let (Some(block), Some(service)) = (b.handle("block"), b.handle("service")) else {
        return 1;
    };
    let mut disk = match Disk::open(block) {
        Ok(d) => d,
        Err(e) => {
            println!("disk: {e}");
            return 1;
        }
    };
    if let Err(e) = disk.partition(b.value_u64("partition")) {
        println!("partition: {e}");
        return 1;
    }
    if disk.start % disk.block_size as u64 != 0 || disk.len % disk.block_size as u64 != 0 {
        return 1;
    }
    if let Some(expected) = b.value("uuid") {
        let expected: Vec<_> = expected.bytes().filter(|b| *b != b'-').collect();
        let mut uuid = [0u8; 16];
        if expected.len() != 32 {
            return 1;
        }
        for (i, pair) in expected.as_chunks::<2>().0.iter().enumerate() {
            let digit = |b: u8| -> Option<u8> {
                match b {
                    b'0'..=b'9' => Some(b - b'0'),
                    b'a'..=b'f' => Some(b - b'a' + 10),
                    b'A'..=b'F' => Some(b - b'A' + 10),
                    _ => None,
                }
            };
            let (Some(a), Some(b)) = (digit(pair[0]), digit(pair[1])) else {
                return 1;
            };
            uuid[i] = a << 4 | b;
        }
        let mut superblock = [0u8; 1024];
        if disk
            .start
            .checked_add(1024)
            .is_none_or(|at| disk.read_disk(at, &mut superblock).is_err())
            || superblock[104..120] != uuid
        {
            println!("filesystem UUID mismatch");
            return 1;
        }
    }
    let block_size = disk.block_size;
    let blocks = disk.len / block_size as u64;
    // Read-only remains the default. Writable mounts must be requested explicitly.
    let readonly = b.value("writable") != Some("1");
    let disk = Box::into_raw(Box::new(disk));
    DISK.store(disk, Ordering::Relaxed);
    if let Err(e) =
        check(unsafe { lwext4::oscar_mount(block_size as u32, blocks, readonly as i32) })
    {
        println!("ext4 mount: {e}");
        return 1;
    }
    println!("ext4 ready");
    let ready = match sys::notify_create() {
        Ok(h) => h,
        Err(_) => return 1,
    };
    if sys::channel_bind(service, ready, 1).is_err() {
        return 1;
    }
    let mut clients: Vec<(Handle, i32)> = Vec::new();
    let mut req = alloc::vec![0u8;dross_abi::MSG_BYTES];
    loop {
        let mut progress = false;
        // Observe closed file capabilities before allocating another inode.
        let mut i = 0;
        while i < clients.len() {
            let (ch, fd) = clients[i];
            match sys::channel_read(ch, &mut req, &mut [], 0) {
                Ok((n, _)) => {
                    progress = true;
                    let (out, h) = handle_file(&req[..n], fd)
                        .unwrap_or_else(|e| (e.to_le_bytes().to_vec(), None));
                    if sys::channel_write(ch, &out, h.as_slice()).is_err() {
                        if let Some(h) = h {
                            let _ = sys::close(h);
                        }
                        let _ = unsafe { lwext4::oscar_close(fd) };
                        let _ = sys::close(ch);
                        clients.swap_remove(i);
                        continue;
                    }
                }
                Err(dross_rt::Error::TimedOut | dross_rt::Error::ShouldWait) => {}
                Err(_) => {
                    progress = true;
                    let _ = unsafe { lwext4::oscar_close(fd) };
                    let _ = sys::close(ch);
                    clients.swap_remove(i);
                    continue;
                }
            }
            i += 1;
        }
        let mut hs = [Handle(0); 1];
        match sys::channel_read(service, &mut req, &mut hs, 0) {
            Ok((n, nh)) => {
                progress = true;
                let op = req
                    .get(..4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
                let result = if op == Some(14) && nh == 1 {
                    let ch = hs[0];
                    let opened = (|| {
                        if clients.len() >= 256 {
                            return Err(-24);
                        }
                        clients.try_reserve(1).map_err(|_| -12)?;
                        let flags =
                            u32::from_le_bytes(req.get(4..8).ok_or(-22)?.try_into().unwrap());
                        if readonly && flags & 3 != 0 {
                            return Err(-30);
                        }
                        let p = path(req.get(8..n).ok_or(-22)?)?;
                        let mut fd = 0;
                        check(unsafe {
                            lwext4::oscar_open(p.as_ptr(), (flags & 3) as i32, &mut fd)
                        })?;
                        if sys::channel_bind(ch, ready, 1).is_err() {
                            let _ = unsafe { lwext4::oscar_close(fd) };
                            return Err(-5);
                        }
                        clients.push((ch, fd));
                        Ok((0i64.to_le_bytes().to_vec(), None))
                    })();
                    if opened.is_err() {
                        let _ = sys::close(ch);
                    }
                    opened
                } else {
                    for h in hs.into_iter().take(nh) {
                        let _ = sys::close(h);
                    }
                    handle(&req[..n], readonly)
                };
                let (out, h) = result.unwrap_or_else(|e| (e.to_le_bytes().to_vec(), None));
                if sys::channel_write(service, &out, h.as_slice()).is_err() {
                    if let Some(h) = h {
                        let _ = sys::close(h);
                    }
                    break;
                }
            }
            Err(dross_rt::Error::TimedOut | dross_rt::Error::ShouldWait) => {}
            Err(_) => break,
        }
        if !progress && sys::notify_wait(ready, dross_rt::FOREVER).is_err() {
            break;
        }
    }
    for (ch, fd) in clients {
        let _ = unsafe { lwext4::oscar_close(fd) };
        let _ = sys::close(ch);
    }
    let _ = unsafe { lwext4::oscar_unmount() };
    0
}
