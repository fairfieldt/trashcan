//! Processes: an address space, a handle table, threads. `spawn` loads a static ELF executable from a memory
//! object and starts its first thread with one handle (a bootstrap channel) in rdi.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};
use dross_abi::{vm, Error, USER_TOP};
use spin::Mutex;

use crate::memory::Memory;
use crate::object::{Entry, HandleTable};
use crate::paging::AddressSpace;
use crate::sched::{self, Thread, WaitCell};

const STACK_SIZE: u64 = 1 << 20;
const STACK_TOP: u64 = 0x0000_7fff_ffff_0000;

pub struct Process {
    pub name: String,
    pub aspace: AddressSpace,
    pub handles: Mutex<HandleTable>,
    threads: Mutex<Vec<Weak<Thread>>>,
    live: AtomicU32,
    /// The exit code, once every thread has gone.
    pub exit: WaitCell<Option<i64>>,
    code: Mutex<Option<i64>>,
    /// Linux programs' state (fds, signal dispositions); None for dross programs.
    pub linux: Option<crate::linux::Linux>,
    /// The first thread's id (Linux `getpid`).
    pid: AtomicU32,
    /// The I/O ports this process may use (one bit per port, clear = allowed), if any.
    pub ioports: Mutex<Option<Box<[u8; 8192]>>>,
}

impl Process {
    pub fn new(name: &str) -> Arc<Process> {
        Self::create(name, None)
    }

    pub fn new_linux(name: &str) -> Arc<Process> {
        Self::create(name, Some(crate::linux::Linux::new(name)))
    }

    pub fn pid(&self) -> u32 {
        self.pid.load(Ordering::Relaxed)
    }

    fn create(name: &str, linux: Option<crate::linux::Linux>) -> Arc<Process> {
        Arc::new(Process {
            linux,
            pid: AtomicU32::new(0),
            name: String::from(name),
            aspace: AddressSpace::new(),
            handles: Mutex::new(HandleTable::new()),
            threads: Mutex::new(Vec::new()),
            live: AtomicU32::new(0),
            exit: WaitCell::new(None),
            code: Mutex::new(None),
            ioports: Mutex::new(None),
        })
    }

    pub fn add_thread(&self, t: &Arc<Thread>) {
        let _ = self.pid.compare_exchange(0, t.tid, Ordering::Relaxed, Ordering::Relaxed);
        self.live.fetch_add(1, Ordering::AcqRel);
        let mut v = self.threads.lock();
        v.retain(|w| w.strong_count() > 0);
        v.push(Arc::downgrade(t));
    }

    pub fn thread_exited(&self, _t: &Thread) {
        if self.live.fetch_sub(1, Ordering::AcqRel) == 1 {
            let dropped = self.handles.lock().clear();
            drop(dropped);
            if let Some(l) = &self.linux {
                drop(l.files.clear());
            }
            let code = self.code.lock().unwrap_or(0);
            self.exit.update(|c| *c = Some(code));
        }
    }

    /// Ends the process: every thread leaves at its next return from the kernel.
    pub fn kill(&self, code: i64) {
        self.code.lock().get_or_insert(code);
        let threads: Vec<_> = self.threads.lock().iter().filter_map(|w| w.upgrade()).collect();
        for t in threads {
            t.kill();
        }
    }

    pub fn enable_ports(&self, first: u16, count: u32) {
        let mut p = self.ioports.lock();
        let map = p.get_or_insert_with(|| Box::new([0xff; 8192]));
        for port in first as u32..first as u32 + count {
            map[(port / 8) as usize] &= !(1 << (port % 8));
        }
    }
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// Loads the ELF executable in `image` into a new process, gives it `bootstrap` as handle 1 and starts its
/// first thread on `cpu`.
pub fn spawn(name: &str, image: &Memory, cpu: u32, bootstrap: Option<Entry>) -> Result<Arc<Process>, Error> {
    let mut hdr = [0u8; 64];
    image.read(0, &mut hdr).map_err(|_| Error::BadElf)?;
    if &hdr[..4] != b"\x7fELF" || hdr[4] != 2 || hdr[5] != 1 || u16_at(&hdr, 16) != 2 || u16_at(&hdr, 18) != 0x3e {
        return Err(Error::BadElf);
    }
    let entry = u64_at(&hdr, 24);
    let (phoff, phentsize, phnum) = (u64_at(&hdr, 32), u16_at(&hdr, 54) as u64, u16_at(&hdr, 56) as u64);
    if phentsize < 56 || phnum > 64 {
        return Err(Error::BadElf);
    }
    let p = Process::new(name);
    for i in 0..phnum {
        let mut ph = [0u8; 56];
        image.read(phoff + i * phentsize, &mut ph).map_err(|_| Error::BadElf)?;
        if u32_at(&ph, 0) != 1 {
            continue;
        }
        let (flags, offset, vaddr, filesz, memsz) = (u32_at(&ph, 4), u64_at(&ph, 8), u64_at(&ph, 16), u64_at(&ph, 32), u64_at(&ph, 40));
        let start = vaddr & !0xfff;
        let end = vaddr.checked_add(memsz).ok_or(Error::BadElf)?.next_multiple_of(4096);
        if filesz > memsz || end > USER_TOP || start == 0 {
            return Err(Error::BadElf);
        }
        let seg = Memory::anonymous(end - start, false, false)?;
        let mut buf = alloc::vec![0u8; filesz as usize];
        image.read(offset, &mut buf).map_err(|_| Error::BadElf)?;
        seg.write(vaddr - start, &buf)?;
        let mut f = vm::READ;
        if flags & 2 != 0 {
            f |= vm::WRITE;
        }
        if flags & 1 != 0 {
            f |= vm::EXECUTE;
        }
        p.aspace.map(Arc::new(seg), 0, end - start, start, f)?;
    }
    let stack = Memory::anonymous(STACK_SIZE, false, false)?;
    p.aspace.map(Arc::new(stack), 0, STACK_SIZE, STACK_TOP - STACK_SIZE, vm::READ | vm::WRITE)?;
    let arg = match bootstrap {
        Some(e) => p.handles.lock().insert(e) as u64,
        None => 0,
    };
    // As if called: the return address slot keeps rsp 8 mod 16 at entry.
    sched::spawn_user(p.clone(), cpu, entry, STACK_TOP - 8, arg);
    Ok(p)
}
