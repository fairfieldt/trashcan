//! fsd: an ext4 filesystem, read-only, served to the kernel's Linux layer (`kernel/src/linux/unix/mount.rs` has
//! the protocol). It reads the disk through a block device on `block` (usbd's mass storage), finds the partition
//! on a GPT disk, and reads the filesystem with `ext4-view`.
//!
//! Bootstrap: `block` (the block device), `service` (the kernel's mount); value `partition`, a GPT partition
//! number (from 1), else the first Linux filesystem partition, else the whole disk.
#![no_std]
#![no_main]

extern crate alloc;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use dross_abi::vm;
use dross_rt::bootstrap::Bootstrap;
use dross_rt::{println, sys, Handle};
use ext4_view::{Ext4, Ext4Error, Ext4Read, FileType};

dross_rt::main!(main);

/// GPT's Linux filesystem data partition type, as stored on disk.
const LINUX_FS: [u8; 16] = [0xaf, 0x3d, 0xc6, 0x0f, 0x83, 0x84, 0x72, 0x47, 0x8e, 0x79, 0x3d, 0x69, 0xd8, 0x47, 0x7d, 0xe4];
/// Listing replies stay under a channel message.
const REPLY_MAX: usize = 60 * 1024;

#[derive(Debug)]
struct DiskError(i64);

impl core::fmt::Display for DiskError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "block device error {}", self.0)
    }
}

impl core::error::Error for DiskError {}

/// The block device: reads land in its window, which doubles as a read-ahead cache of the last range read.
struct Disk {
    ch: Handle,
    window: &'static [u8],
    block_size: u64,
    blocks: u64,
    /// The cached range, in blocks.
    cached: Option<(u64, u64)>,
    /// The partition's first byte and its length.
    start: u64,
    len: u64,
}

impl Disk {
    fn open(ch: Handle) -> Result<Disk, i64> {
        let mut reply = [0u8; 20];
        let mut hs = [Handle(0); 1];
        // The USB driver may still be looking for the disk (no device: -19): ask again for 10 s.
        let mut tries = 0;
        let (n, nh, status) = loop {
            sys::channel_write(ch, &[0], &[]).map_err(|_| -5)?;
            let (n, nh) = sys::channel_read(ch, &mut reply, &mut hs, dross_rt::FOREVER).map_err(|_| -5)?;
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
        let at = sys::vm_map(hs[0], 0, size, 0, vm::READ).map_err(|_| -12)?;
        // SAFETY: a mapping of the device's window that stays for the life of the program.
        let window = unsafe { core::slice::from_raw_parts(at as *const u8, size as usize) };
        Ok(Disk { ch, window, block_size, blocks, cached: None, start: 0, len: block_size * blocks })
    }

    /// Bytes from the disk (not the partition) at `at`.
    fn read_disk(&mut self, mut at: u64, mut dst: &mut [u8]) -> Result<(), i64> {
        let per_window = self.window.len() as u64 / self.block_size;
        while !dst.is_empty() {
            let lba = at / self.block_size;
            let hit = self.cached.filter(|&(first, count)| lba >= first && lba < first + count);
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
                    sys::channel_read(self.ch, &mut reply, &mut [], dross_rt::FOREVER).map_err(|_| -5)?;
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
        let mut e = alloc::vec![0u8; size as usize];
        for i in 0..count.min(128) {
            self.read_disk(entries * self.block_size + i * size, &mut e)?;
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
                self.start = first * self.block_size;
                self.len = (last + 1 - first) * self.block_size;
                println!("partition {}: {} MiB", i + 1, self.len >> 20);
                return Ok(());
            }
        }
        Err(-2)
    }
}

impl Ext4Read for Disk {
    fn read(&mut self, start: u64, dst: &mut [u8]) -> Result<(), Box<dyn core::error::Error + Send + Sync>> {
        if start.checked_add(dst.len() as u64).is_none_or(|end| end > self.len) {
            return Err(Box::new(DiskError(-22)));
        }
        self.read_disk(self.start + start, dst).map_err(|e| Box::new(DiskError(e)) as _)
    }
}

fn errno(e: &Ext4Error) -> i64 {
    match e {
        Ext4Error::NotFound => -2,
        Ext4Error::NotADirectory => -20,
        Ext4Error::IsADirectory => -21,
        Ext4Error::FileTooLarge => -27,
        Ext4Error::TooManySymlinks => -40,
        Ext4Error::PathTooLong => -36,
        _ => -5,
    }
}

fn type_bits(t: FileType) -> u32 {
    if t.is_dir() {
        0o040000
    } else if t.is_symlink() {
        0o120000
    } else if t.is_char_dev() {
        0o020000
    } else if t.is_block_dev() {
        0o060000
    } else if t.is_fifo() {
        0o010000
    } else if t.is_socket() {
        0o140000
    } else {
        0o100000
    }
}

/// `LIST`: entries of the directory from index `cookie`, as many as fit.
fn list(fs: &Ext4, path: &str, cookie: u64) -> Result<Vec<u8>, i64> {
    let mut out = 0i64.to_le_bytes().to_vec();
    out.extend_from_slice(&0u64.to_le_bytes());
    let mut next = 0u64;
    for (i, entry) in fs.read_dir(path).map_err(|e| errno(&e))?.enumerate().skip(cookie as usize) {
        let entry = entry.map_err(|e| errno(&e))?;
        let name = entry.file_name();
        let Ok(name) = name.as_str() else { continue };
        if name == "." || name == ".." {
            continue;
        }
        let m = entry.metadata().map_err(|e| errno(&e))?;
        let target = if m.is_symlink() {
            fs.read_link(entry.path().as_path()).map(|t| Vec::from(t.as_ref())).unwrap_or_default()
        } else {
            Vec::new()
        };
        let modified = m.modified();
        let mtime = (modified.seconds().max(0) as u64) * 1_000_000_000 + modified.nanoseconds() as u64;
        let mut e = Vec::with_capacity(32 + name.len() + target.len());
        e.extend_from_slice(&(type_bits(m.file_type()) | m.mode() as u32).to_le_bytes());
        e.extend_from_slice(&m.uid().to_le_bytes());
        e.extend_from_slice(&m.gid().to_le_bytes());
        e.extend_from_slice(&m.len().to_le_bytes());
        e.extend_from_slice(&mtime.to_le_bytes());
        e.extend_from_slice(&(name.len() as u16).to_le_bytes());
        e.extend_from_slice(&(target.len() as u16).to_le_bytes());
        e.extend_from_slice(name.as_bytes());
        e.extend_from_slice(&target);
        if out.len() + e.len() > REPLY_MAX {
            next = i as u64;
            break;
        }
        out.extend_from_slice(&e);
    }
    out[8..16].copy_from_slice(&next.to_le_bytes());
    Ok(out)
}

/// `OPEN`: the whole file in a new memory object, which goes to the kernel.
fn open(fs: &Ext4, path: &str) -> Result<(Vec<u8>, Handle), i64> {
    let mut file = fs.open(path).map_err(|e| errno(&e))?;
    let size = file.metadata().len();
    let mem_size = size.next_multiple_of(4096).max(4096);
    let mem = sys::mem_create(mem_size, 0).map_err(|_| -12)?;
    let at = sys::vm_map(mem, 0, mem_size, 0, vm::READ | vm::WRITE).map_err(|_| -12)?;
    // SAFETY: the mapping just made, unmapped below before the memory leaves.
    let buf = unsafe { core::slice::from_raw_parts_mut(at as *mut u8, size as usize) };
    let t0 = sys::clock();
    let mut done = 0;
    let mut result = Ok(());
    while done < buf.len() {
        match file.read_bytes(&mut buf[done..]) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) => {
                result = Err(errno(&e));
                break;
            }
        }
    }
    let _ = sys::vm_unmap(at, mem_size);
    if let Err(e) = result {
        let _ = sys::close(mem);
        return Err(e);
    }
    if size >= 64 << 20 {
        let ms = ((sys::clock() - t0) / 1_000_000).max(1);
        println!("{path}: {} MiB in {ms} ms ({} MB/s)", size >> 20, size / 1000 / ms);
    }
    let mut reply = 0i64.to_le_bytes().to_vec();
    reply.extend_from_slice(&size.to_le_bytes());
    Ok((reply, mem))
}

fn main(boot: Handle) -> i32 {
    let b = Bootstrap::read(boot).expect("bootstrap");
    let (Some(block), Some(service)) = (b.handle("block"), b.handle("service")) else {
        println!("needs block and service");
        return 1;
    };
    let mut disk = match Disk::open(block) {
        Ok(d) => d,
        Err(e) => {
            println!("no disk ({e})");
            return 1;
        }
    };
    if let Err(e) = disk.partition(b.value_u64("partition")) {
        println!("no partition ({e})");
        return 1;
    }
    let fs = match Ext4::load(Box::new(disk)) {
        Ok(fs) => fs,
        Err(e) => {
            println!("not an ext4 filesystem: {e}");
            return 1;
        }
    };
    println!("ext4 \"{}\" ready", fs.label().to_str().unwrap_or("?"));
    let mut req = alloc::vec![0u8; 4096 + 12];
    loop {
        let n = match sys::channel_read(service, &mut req, &mut [], dross_rt::FOREVER) {
            Ok((n, _)) => n,
            Err(_) => return 0,
        };
        let op = (n >= 4).then(|| u32::from_le_bytes(req[..4].try_into().unwrap()));
        let path = |from: usize| String::from_utf8_lossy(&req[from.min(n)..n]).into_owned();
        let (reply, handle) = match op {
            Some(1) if n >= 12 => {
                let cookie = u64::from_le_bytes(req[4..12].try_into().unwrap());
                (list(&fs, &path(12), cookie), None)
            }
            Some(2) => match open(&fs, &path(4)) {
                Ok((r, h)) => (Ok(r), Some(h)),
                Err(e) => (Err(e), None),
            },
            _ => (Err(-22), None),
        };
        let bytes = reply.unwrap_or_else(|e| e.to_le_bytes().to_vec());
        if sys::channel_write(service, &bytes, handle.as_slice()).is_err() {
            return 0;
        }
    }
}
