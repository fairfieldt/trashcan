//! System call entry and dispatch (the interface is `dross-abi`).

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::arch::global_asm;
use dross_abi::{rights, sys, vm, Error, MSG_BYTES, MSG_HANDLES};

use crate::interrupts::Frame;
use crate::iommu::Domain;
use crate::ipc::{ChannelEnd, Message, Notification, ReadError};
use crate::irq::Interrupt;
use crate::memory::Memory;
use crate::object::{get_obj, Entry, Object};
use crate::process::{self, Process};
use crate::resource::Kind;
use crate::sched::{self, WaitError};
use crate::{cpu, mem, println, time};

global_asm!(
    r#"
.global syscall_entry
syscall_entry:
    swapgs
    mov gs:[16], rsp
    mov rsp, gs:[8]
    push 0x1b
    push qword ptr gs:[16]
    push r11
    push 0x23
    push rcx
    push 0
    push 0x100
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    mov rdi, rsp
    cld
    call syscall_dispatch
    jmp isr_return
"#
);

type R = Result<u64, Error>;

fn wait_err(e: WaitError) -> Error {
    match e {
        WaitError::TimedOut => Error::TimedOut,
        // The thread leaves on the way out of the kernel.
        WaitError::Killed => Error::PeerClosed,
    }
}

fn copy_in(p: &Process, ptr: u64, len: u64) -> Result<Vec<u8>, Error> {
    if len > MSG_BYTES as u64 * 16 {
        return Err(Error::InvalidArgs);
    }
    if len == 0 { return Ok(Vec::new()); }
    let mut v = alloc::vec![0u8; len as usize];
    // SAFETY: with_user checked the range is mapped readable while the lock is held.
    p.aspace.with_user(ptr, len, false, |src| unsafe { core::ptr::copy_nonoverlapping(src, v.as_mut_ptr(), len as usize) })?;
    Ok(v)
}

fn copy_out(p: &Process, ptr: u64, data: &[u8]) -> Result<(), Error> {
    if data.is_empty() { return Ok(()); }
    // SAFETY: with_user checked the range is mapped writable while the lock is held.
    p.aspace.with_user(ptr, data.len() as u64, true, |dst| unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) })
}

fn insert(p: &Process, o: Object) -> u64 {
    p.handles.lock().insert(Entry::new(o)) as u64
}

#[unsafe(no_mangle)]
extern "C" fn syscall_dispatch(f: &mut Frame) {
    let is_linux = sched::current_process().linux.is_some();
    crate::cpu::this().sched.last_syscall.store(f.rax, core::sync::atomic::Ordering::Relaxed);
    if is_linux && f.rax & dross_abi::NATIVE == 0 {
        f.rax = crate::linux::syscall(f);
    } else {
        let (rax, rdx) = match dispatch(f.rax & !dross_abi::NATIVE, [f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.r9]) {
            Ok(v) => (0, v),
            Err(e) => (e as i64 as u64, 0),
        };
        f.rax = rax;
        f.rdx = rdx;
    }
    #[cfg(feature = "unix")]
    crate::linux::unix::user_return(f);
    let me = sched::current();
    if me.killed() {
        drop(me);
        sched::exit_current();
    }
    drop(me);
    sched::preempt_point();
}

fn dispatch(n: u64, a: [u64; 6]) -> R {
    let p = sched::current_process();
    match n {
        #[cfg(feature = "unix")]
        sys::UNIX_NET_REGISTER => {
            let t = p.handles.lock();
            let resource = get_obj!(t, a[0] as u32, Resource, rights::WRITE)?;
            resource.check(Kind::System, 0, 1)?;
            let ch = get_obj!(t, a[1] as u32, Channel, rights::READ | rights::WRITE)?;
            drop(t); crate::linux::unix::net::register(ch).map(|_| 0)
        }
        #[cfg(feature = "unix")]
        sys::UNIX_MOUNT => {
            let t = p.handles.lock();
            let resource = get_obj!(t, a[0] as u32, Resource, rights::WRITE)?;
            resource.check(Kind::System, 0, 1)?;
            let ch = get_obj!(t, a[1] as u32, Channel, rights::READ | rights::WRITE)?;
            drop(t);
            let path = copy_in(&p, a[2], a[3].min(4096))?;
            let path = core::str::from_utf8(&path).map_err(|_| Error::InvalidArgs)?;
            crate::linux::unix::vfs::mount(path, crate::linux::unix::mount::Mount::new(ch)).map(|_| 0).map_err(|_| Error::InvalidArgs)
        }
        #[cfg(feature = "unix")]
        sys::RANDOM => {
            if a[1] > MSG_BYTES as u64 { return Err(Error::InvalidArgs); }
            let mut bytes = alloc::vec![0; a[1] as usize];
            crate::linux::unix::entropy::fill(&mut bytes).map_err(|_| Error::ShouldWait)?;
            copy_out(&p, a[0], &bytes).map(|_| a[1])
        }
        sys::LOG => {
            let b = copy_in(&p, a[0], a[1].min(4096))?;
            println!("{}: {}", p.name, core::str::from_utf8(&b).unwrap_or("(not utf-8)").trim_end());
            Ok(0)
        }
        sys::PROCESS_EXIT => {
            p.kill(a[0] as i64);
            Ok(0)
        }
        sys::THREAD_EXIT => {
            drop(p);
            sched::exit_current()
        }
        sys::THREAD_CREATE => {
            let cpu = if a[3] == u64::MAX { sched::pick_cpu() } else { a[3] as u32 };
            if cpu >= crate::smp::online() {
                return Err(Error::InvalidArgs);
            }
            let t = sched::spawn_user(p.clone(), cpu, a[0], a[1], a[2]);
            Ok(insert(&p, Object::Thread(t)))
        }
        sys::YIELD => {
            sched::request_resched();
            Ok(0)
        }
        sys::CLOCK => Ok(time::now_ns()),
        sys::CPU_COUNT => Ok(crate::smp::online() as u64),
        sys::HANDLE_CLOSE => {
            let e = p.handles.lock().remove(a[0] as u32)?;
            drop(e);
            Ok(0)
        }
        sys::HANDLE_DUP => {
            let mut t = p.handles.lock();
            let e = t.get(a[0] as u32)?;
            if e.rights & rights::DUPLICATE == 0 {
                return Err(Error::AccessDenied);
            }
            let d = Entry { object: e.object.clone(), rights: e.rights & a[1] as u32 };
            Ok(t.insert(d) as u64)
        }
        sys::MEM_CREATE => {
            let m = Memory::anonymous(a[0], a[1] & vm::CONTIGUOUS != 0, a[1] & vm::BELOW_4G != 0)?;
            Ok(insert(&p, Object::Memory(Arc::new(m))))
        }
        sys::MEM_SIZE => {
            let m = get_obj!(p.handles.lock(), a[0] as u32, Memory, 0)?;
            Ok(m.size())
        }
        sys::MEM_PHYSICAL => {
            let r = get_obj!(p.handles.lock(), a[0] as u32, Resource, 0)?;
            let (phys, size) = (a[1], a[2]);
            if phys % 4096 != 0 || size % 4096 != 0 || size == 0 {
                return Err(Error::InvalidArgs);
            }
            r.check(Kind::Physical, phys, size)?;
            if mem::overlaps_ram(phys, size) {
                return Err(Error::AccessDenied);
            }
            Ok(insert(&p, Object::Memory(Arc::new(Memory::device(phys, size)))))
        }
        sys::VM_MAP => {
            let (m, r) = {
                let t = p.handles.lock();
                let e = t.get(a[0] as u32)?;
                let Object::Memory(m) = &e.object else { return Err(Error::WrongType) };
                (m.clone(), e.rights)
            };
            let flags = a[4];
            if r & rights::MAP == 0 || (flags & vm::WRITE != 0 && r & rights::WRITE == 0) || (flags & vm::WRITE != 0 && !m.writable()) {
                return Err(Error::AccessDenied);
            }
            p.aspace.map(m, a[1], a[2], a[3], flags)
        }
        sys::VM_UNMAP => p.aspace.unmap(a[0], a[1]).map(|_| 0),
        sys::VM_MEMORY => {
            let (m, offset, flags, rest) = p.aspace.lookup_mapping(a[0]).ok_or(Error::NotFound)?;
            let mut out = offset.to_le_bytes().to_vec();
            out.extend_from_slice(&rest.to_le_bytes());
            copy_out(&p, a[1], &out)?;
            let write = if flags & vm::WRITE != 0 && m.writable() { rights::WRITE } else { 0 };
            let r = rights::READ | rights::MAP | rights::TRANSFER | rights::DUPLICATE | write;
            Ok(p.handles.lock().insert(Entry { object: Object::Memory(m), rights: r }) as u64)
        }
        sys::SPAWN => {
            let (image, boot) = {
                let mut t = p.handles.lock();
                let image = get_obj!(t, a[0] as u32, Memory, rights::READ)?;
                let boot = if a[3] != 0 {
                    let e = t.get(a[3] as u32)?;
                    if e.rights & rights::TRANSFER == 0 {
                        return Err(Error::AccessDenied);
                    }
                    Some(t.remove(a[3] as u32)?)
                } else {
                    None
                };
                (image, boot)
            };
            let name = copy_in(&p, a[1], a[2].min(64))?;
            let name = core::str::from_utf8(&name).map_err(|_| Error::InvalidArgs)?;
            let cpu = if a[4] == u64::MAX { sched::pick_cpu() } else { a[4] as u32 };
            if cpu >= crate::smp::online() {
                return Err(Error::InvalidArgs);
            }
            let child = process::spawn(name, &image, cpu, boot)?;
            Ok(insert(&p, Object::Process(child)))
        }
        sys::PROCESS_WAIT => {
            let c = get_obj!(p.handles.lock(), a[0] as u32, Process, rights::WAIT)?;
            let code = c.exit.wait(sched::deadline_from_ns(a[1]), |c| *c).map_err(wait_err)?;
            Ok(code as u64)
        }
        sys::CHANNEL_CREATE => {
            let (x, y) = ChannelEnd::pair();
            let mut t = p.handles.lock();
            let hx = t.insert(Entry::new(Object::Channel(Arc::new(x))));
            let hy = t.insert(Entry::new(Object::Channel(Arc::new(y))));
            Ok(hx as u64 | (hy as u64) << 32)
        }
        sys::CHANNEL_WRITE => {
            let ch = get_obj!(p.handles.lock(), a[0] as u32, Channel, rights::WRITE)?;
            if a[2] > MSG_BYTES as u64 || a[4] > MSG_HANDLES as u64 {
                return Err(Error::InvalidArgs);
            }
            let bytes = copy_in(&p, a[1], a[2])?;
            let hs = copy_in(&p, a[3], a[4] * 4)?;
            let hs: Vec<u32> = hs.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            // Take all the handles or none.
            let mut t = p.handles.lock();
            let handles = {
                for (i, &h) in hs.iter().enumerate() {
                    let e = t.get(h)?;
                    if e.rights & rights::TRANSFER == 0 || hs[..i].contains(&h) {
                        return Err(Error::AccessDenied);
                    }
                    if let Object::Channel(c) = &e.object {
                        if Arc::ptr_eq(c, &ch) {
                            return Err(Error::InvalidArgs);
                        }
                    }
                }
                hs.iter().map(|&h| t.remove(h).unwrap()).collect::<Vec<_>>()
            };
            match ch.write(Message { bytes, handles }) {
                Ok(()) => Ok(0),
                Err((e, m)) => {
                    // Give the handles back.
                    for (h, entry) in hs.into_iter().zip(m.handles) {
                        t.restore(h, entry);
                    }
                    Err(e)
                }
            }
        }
        sys::CHANNEL_READ => {
            let ch = get_obj!(p.handles.lock(), a[0] as u32, Channel, rights::READ)?;
            let m = match ch.read(a[2] as usize, a[4] as usize, sched::deadline_from_ns(a[5])) {
                Ok(m) => m,
                Err(ReadError::Wait(e)) => return Err(wait_err(e)),
                Err(ReadError::PeerClosed) => return Err(Error::PeerClosed),
                Err(ReadError::TooSmall(b, h)) => {
                    let _ = (b, h);
                    return Err(Error::BufferTooSmall);
                }
            };
            let (nb, nh) = (m.bytes.len(), m.handles.len());
            copy_out(&p, a[1], &m.bytes)?;
            let mut t = p.handles.lock();
            let ids: Vec<u32> = m.handles.into_iter().map(|e| t.insert(e)).collect();
            let hs: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            if let Err(e) = copy_out(&p, a[3], &hs) {
                // No caller can observe or reuse these slots while the table is locked.
                let removed: Vec<Entry> = ids.into_iter().map(|id| t.remove(id).unwrap()).collect();
                drop(t);
                drop(removed);
                return Err(e);
            }
            Ok(nb as u64 | (nh as u64) << 32)
        }
        sys::CHANNEL_BIND => {
            let t = p.handles.lock();
            let ch = get_obj!(t, a[0] as u32, Channel, rights::READ)?;
            let n = get_obj!(t, a[1] as u32, Notification, rights::SIGNAL)?;
            drop(t);
            ch.bind(n, a[2]);
            Ok(0)
        }
        sys::NOTIFY_CREATE => Ok(insert(&p, Object::Notification(Arc::new(Notification::new())))),
        sys::NOTIFY_SIGNAL => {
            let n = get_obj!(p.handles.lock(), a[0] as u32, Notification, rights::SIGNAL)?;
            n.signal(a[1]);
            Ok(0)
        }
        sys::NOTIFY_WAIT => {
            let n = get_obj!(p.handles.lock(), a[0] as u32, Notification, rights::WAIT)?;
            drop(p);
            n.wait(sched::deadline_from_ns(a[1])).map_err(wait_err)
        }
        sys::RESOURCE_SUB => {
            let r = get_obj!(p.handles.lock(), a[0] as u32, Resource, rights::DUPLICATE)?;
            let s = r.sub(a[1], a[2])?;
            Ok(insert(&p, Object::Resource(Arc::new(s))))
        }
        sys::IOPORT_ENABLE => {
            let r = get_obj!(p.handles.lock(), a[0] as u32, Resource, 0)?;
            r.check(Kind::IoPort, a[1], a[2])?;
            if a[2] == 0 || a[1] + a[2] > 0x10000 {
                return Err(Error::InvalidArgs);
            }
            p.enable_ports(a[1] as u16, a[2] as u32);
            // Reload this CPU's bitmap now; other CPUs load it at their next switch.
            let ports = p.ioports.lock();
            let pc = cpu::this();
            pc.load_iomap(0, None);
            pc.load_iomap(Arc::as_ptr(&p) as u64, ports.as_deref());
            Ok(0)
        }
        sys::IRQ_MSI => {
            let t = p.handles.lock();
            let r = get_obj!(t, a[0] as u32, Resource, 0)?;
            let n = get_obj!(t, a[1] as u32, Notification, rights::SIGNAL)?;
            drop(t);
            r.check(Kind::Irq, 0, 1)?;
            let cpu = if a[3] == u64::MAX { sched::pick_cpu() } else { a[3] as u32 };
            let irq = Interrupt::msi(cpu, n, a[2])?;
            let mut out = [0u8; 12];
            out[..8].copy_from_slice(&irq.msi_address.to_le_bytes());
            out[8..].copy_from_slice(&irq.msi_data.to_le_bytes());
            copy_out(&p, a[4], &out)?;
            Ok(insert(&p, Object::Interrupt(Arc::new(irq))))
        }
        sys::DMA_DOMAIN_CREATE => {
            let r = get_obj!(p.handles.lock(), a[0] as u32, Resource, 0)?;
            r.check(Kind::Iommu, 0, 1)?;
            if !crate::iommu::present() {
                return Err(Error::NotFound);
            }
            Ok(insert(&p, Object::Domain(Arc::new(Domain::new()))))
        }
        sys::DMA_ATTACH => {
            let t = p.handles.lock();
            let d = get_obj!(t, a[0] as u32, Domain, 0)?;
            let r = get_obj!(t, a[1] as u32, Resource, 0)?;
            drop(t);
            r.check(Kind::Iommu, 0, 1)?;
            d.attach(a[2] as u16).map(|_| 0)
        }
        sys::DMA_MAP => {
            let t = p.handles.lock();
            let d = get_obj!(t, a[0] as u32, Domain, rights::MAP)?;
            let m = get_obj!(t, a[1] as u32, Memory, rights::MAP)?;
            drop(t);
            d.map(m, a[2], a[3], a[4], a[5]).map(|_| 0)
        }
        sys::DMA_UNMAP => {
            let d = get_obj!(p.handles.lock(), a[0] as u32, Domain, rights::MAP)?;
            d.unmap(a[1], a[2]).map(|_| 0)
        }
        sys::SPAWN_LINUX => {
            let (image, boot) = {
                let mut t = p.handles.lock();
                let image = get_obj!(t, a[0] as u32, Memory, rights::READ)?;
                let boot = if a[5] != 0 {
                    if t.get(a[5] as u32)?.rights & rights::TRANSFER == 0 {
                        return Err(Error::AccessDenied);
                    }
                    Some(t.remove(a[5] as u32)?)
                } else {
                    None
                };
                (image, boot)
            };
            let argv = copy_in(&p, a[1], a[2].min(64 << 10))?;
            let envp = copy_in(&p, a[3], a[4].min(64 << 10))?;
            let child = crate::linux::spawn(&image, &argv, &envp, boot)?;
            Ok(insert(&p, Object::Process(child)))
        }
        sys::HANDLE_FD => {
            if p.linux.is_none() {
                return Err(Error::WrongType);
            }
            let o = p.handles.lock().get(a[0] as u32)?.clone();
            match o.object {
                #[cfg(feature = "unix")]
                Object::Channel(ch) if o.rights & (rights::READ | rights::WRITE) == (rights::READ | rights::WRITE) => crate::linux::unix::channel::fd(&p, ch),
                Object::Notification(n) if o.rights & rights::WAIT != 0 => crate::linux::notification_fd(&p, n),
                Object::Memory(m) if o.rights & rights::MAP != 0 => crate::linux::memory_fd(&p, m),
                Object::Notification(_) | Object::Memory(_) => return Err(Error::AccessDenied),
                _ => return Err(Error::WrongType),
            }
            .map_err(|_| Error::NoResources)
        }
        sys::SYSTEM_EXIT => {
            let r = get_obj!(p.handles.lock(), a[0] as u32, Resource, 0)?;
            r.check(Kind::System, 0, 1)?;
            println!("system: exit requested by {} with code {}", p.name, a[1]);
            crate::qemu::exit(a[1] == 0)
        }
        _ => Err(Error::BadSyscall),
    }
}
