//! System call entry and dispatch (the interface is `dross-abi`).

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::arch::global_asm;
use dross_abi::{dma, rights, sys, vm, Error, MSG_BYTES, MSG_HANDLES};

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
    let mut v = Vec::new();
    v.try_reserve_exact(len as usize).map_err(|_| Error::NoMemory)?;
    v.resize(len as usize, 0);
    // SAFETY: with_user checked the range is mapped readable while the lock is held.
    p.aspace.with_user(ptr, len, false, |src| unsafe { core::ptr::copy_nonoverlapping(src, v.as_mut_ptr(), len as usize) })?;
    Ok(v)
}

fn copy_out(p: &Process, ptr: u64, data: &[u8]) -> Result<(), Error> {
    if data.is_empty() { return Ok(()); }
    // SAFETY: with_user checked the range is mapped writable while the lock is held.
    p.aspace.with_user(ptr, data.len() as u64, true, |dst| unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) })
}

/// Reserved slots cannot be stolen by another thread while creation or receive
/// blocks. Every error releases unused identities without allocating rollback.
struct IncomingHandles {
    process: Arc<Process>, token: u64, ids: [u32; MSG_HANDLES],
}
impl IncomingHandles {
    fn new(process: Arc<Process>, count: usize) -> Result<Self, Error> {
        let (token, ids) = if count == 0 { (0, [0; MSG_HANDLES]) } else { process.handles.lock().reserve_handles(count)? };
        Ok(Self { process, token, ids })
    }
    fn publish(&mut self, entries: impl IntoIterator<Item = Entry>) -> Result<(), Error> {
        let mut table = self.process.handles.lock();
        if self.ids.iter().filter(|&&id| id != 0).any(|&id| !table.reserved(id, self.token)) { return Err(Error::PeerClosed); }
        for entry in entries {
            let id = self.ids.iter_mut().find(|id| **id != 0).expect("reserved receive capacity");
            assert!(table.finish_transfer(*id, self.token, Some(entry)).is_none()); *id = 0;
        }
        Ok(())
    }
    fn one(&mut self, entry: Entry) -> Result<u64, Error> {
        let id = self.ids.iter().find(|&&id| id != 0).copied().ok_or(Error::NoResources)?;
        self.publish(core::iter::once(entry))?; Ok(id as u64)
    }
}
impl Drop for IncomingHandles {
    fn drop(&mut self) {
        if self.ids.iter().all(|&id| id == 0) { return; }
        let mut table = self.process.handles.lock();
        for &id in self.ids.iter().filter(|&&id| id != 0) { assert!(table.finish_transfer(id, self.token, None).is_none()); }
        table.compact();
    }
}
fn insert(incoming: &mut IncomingHandles, object: Object) -> R { incoming.one(Entry::new(object)) }

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
    crate::ipc::reap_deferred();
    crate::linux::reap_socket_cycles();
    crate::heap::reap_deferred();
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
    let outputs = match n {
        sys::CHANNEL_CREATE => 2,
        sys::CHANNEL_READ => (a[4] as usize).min(MSG_HANDLES),
        sys::THREAD_START | sys::THREAD_CREATE | sys::TIMER_CREATE | sys::EVENT_QUEUE_CREATE
        | sys::ACCOUNT_CREATE | sys::ACCOUNT_CHILD | sys::MEM_CREATE | sys::MEM_CREATE_IN
        | sys::MEM_PHYSICAL | sys::VM_MEMORY | sys::SPAWN | sys::SPAWN_LINUX | sys::SPAWN_LINUX_IN
        | sys::HANDLE_DUP | sys::NOTIFY_CREATE | sys::RESOURCE_SUB | sys::IRQ_MSI | sys::DMA_DOMAIN_CREATE => 1,
        _ => 0,
    };
    let mut incoming = IncomingHandles::new(p.clone(), outputs)?;
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
        sys::THREAD_START => {
            let cpu = if a[2] == u64::MAX { sched::pick_cpu() } else { u32::try_from(a[2]).map_err(|_| Error::InvalidArgs)? };
            if cpu >= crate::smp::online() { return Err(Error::InvalidArgs); }
            let thread = sched::spawn_owned(p.clone(), cpu, a[0], a[1], a[3])?;
            insert(&mut incoming, Object::Thread(thread))
        }
        sys::THREAD_WAIT => {
            let thread = get_obj!(p.handles.lock(), u32::try_from(a[0]).map_err(|_| Error::BadHandle)?, Thread, rights::WAIT)?;
            if Arc::ptr_eq(&thread, &sched::current()) { return Err(Error::Busy); }
            thread.exited.wait(sched::deadline_from_ns(a[1]), |done| done.then_some(())).map_err(wait_err).map(|_| 0)
        }
        sys::TIMER_CREATE => insert(&mut incoming, Object::Timer(crate::timer::Timer::new()?)),
        sys::TIMER_ARM | sys::TIMER_CANCEL => {
            let timer = get_obj!(p.handles.lock(), u32::try_from(a[0]).map_err(|_| Error::BadHandle)?, Timer, rights::WRITE)?;
            if n == sys::TIMER_ARM { timer.arm(a[1]) } else { timer.cancel() }.map(|_| 0)
        }
        sys::EVENT_QUEUE_CREATE => {
            let capacity = usize::try_from(a[0]).map_err(|_| Error::InvalidArgs)?;
            insert(&mut incoming, Object::EventQueue(Arc::new(crate::events::EventQueue::new(capacity)?)))
        }
        sys::OBJECT_SUBSCRIBE => {
            let (queue, source, life) = {
                let handles = p.handles.lock();
                let queue = get_obj!(handles, u32::try_from(a[0]).map_err(|_| Error::BadHandle)?, EventQueue, rights::WRITE)?;
                let entry = handles.get(u32::try_from(a[1]).map_err(|_| Error::BadHandle)?)?;
                if entry.rights & rights::WAIT == 0 { return Err(Error::AccessDenied); }
                let (source, supported) = match &entry.object {
                    Object::Channel(c) => (c.signals(), dross_abi::event::READABLE | dross_abi::event::WRITABLE | dross_abi::event::PEER_CLOSED),
                    Object::Notification(n) => (n.signals.clone(), dross_abi::event::PENDING),
                    Object::Process(p) => (p.signals.clone(), dross_abi::event::EXITED),
                    Object::Thread(t) => (t.signals.clone(), dross_abi::event::EXITED),
                    Object::Timer(t) => (t.signals.clone(), dross_abi::event::EXPIRED),
                    _ => return Err(Error::WrongType),
                };
                if a[2] == 0 || a[2] & !supported != 0 { return Err(Error::InvalidArgs); }
                (queue, source, entry.life.as_ref().unwrap().identity.clone())
            };
            queue.subscribe(source, life, a[2], a[3])
        }
        sys::SUBSCRIPTION_REARM | sys::SUBSCRIPTION_CANCEL => {
            let queue = get_obj!(p.handles.lock(), u32::try_from(a[0]).map_err(|_| Error::BadHandle)?, EventQueue, rights::WRITE)?;
            if n == sys::SUBSCRIPTION_REARM { queue.rearm(a[1]) } else { queue.cancel(a[1]) }.map(|_| 0)
        }
        sys::EVENT_QUEUE_WAIT => {
            let queue = get_obj!(p.handles.lock(), u32::try_from(a[0]).map_err(|_| Error::BadHandle)?, EventQueue, rights::READ | rights::WAIT)?;
            queue.wait(sched::deadline_from_ns(a[2]), |event| {
                let mut out = [0u8; 24];
                out[..8].copy_from_slice(&event.subscription.to_le_bytes());
                out[8..16].copy_from_slice(&event.key.to_le_bytes());
                out[16..].copy_from_slice(&event.observed.to_le_bytes());
                copy_out(&p, a[1], &out)
            }).map(|_| 0)
        }
        sys::ACCOUNT_CREATE => {
            let resource = get_obj!(p.handles.lock(), u32::try_from(a[0]).map_err(|_| Error::BadHandle)?, Resource, rights::WRITE)?;
            resource.check(Kind::System, 0, 1)?;
            insert(&mut incoming, Object::Account(crate::account::Account::child(p.account.clone(), a[1])?))
        }
        sys::ACCOUNT_CHILD => {
            let parent = get_obj!(p.handles.lock(), u32::try_from(a[0]).map_err(|_| Error::BadHandle)?, Account, rights::WRITE)?;
            insert(&mut incoming, Object::Account(crate::account::Account::child(parent, a[1])?))
        }
        sys::ACCOUNT_INFO => {
            let account = if a[0] == 0 { p.account.clone() } else {
                get_obj!(p.handles.lock(), u32::try_from(a[0]).map_err(|_| Error::BadHandle)?, Account, rights::READ)?
            };
            let mut bytes = [0; 24];
            for (out, value) in bytes.chunks_exact_mut(8).zip(account.stats()) { out.copy_from_slice(&value.to_le_bytes()); }
            copy_out(&p, a[1], &bytes).map(|_| 0)
        }
        sys::MEM_CREATE_IN => {
            let account = get_obj!(p.handles.lock(), u32::try_from(a[0]).map_err(|_| Error::BadHandle)?, Account, rights::WRITE)?;
            if a[2] & !(vm::CONTIGUOUS | vm::BELOW_4G) != 0 { return Err(Error::InvalidArgs); }
            let m = Memory::anonymous_in(a[1], a[2] & vm::CONTIGUOUS != 0, a[2] & vm::BELOW_4G != 0, account)?;
            insert(&mut incoming, Object::Memory(Arc::new(m)))
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
            drop(incoming);
            drop(p);
            sched::exit_current()
        }
        sys::THREAD_CREATE => {
            let cpu = if a[3] == u64::MAX { sched::pick_cpu() } else { u32::try_from(a[3]).map_err(|_| Error::InvalidArgs)? };
            if cpu >= crate::smp::online() {
                return Err(Error::InvalidArgs);
            }
            let t = sched::spawn_user(p.clone(), cpu, a[0], a[1], a[2])?;
            insert(&mut incoming, Object::Thread(t))
        }
        sys::YIELD => {
            sched::request_resched();
            Ok(0)
        }
        sys::CLOCK => Ok(time::now_ns()),
        sys::CPU_COUNT => Ok(crate::smp::online() as u64),
        sys::HANDLE_CLOSE => {
            let e = { let mut table = p.handles.lock(); let entry = table.remove(a[0] as u32)?; table.compact(); entry };
            drop(e);
            crate::ipc::collect_cycles();
            Ok(0)
        }
        sys::HANDLE_DUP => {
            let t = p.handles.lock();
            let e = t.get(a[0] as u32)?;
            if e.rights & rights::DUPLICATE == 0 {
                return Err(Error::AccessDenied);
            }
            let d = Entry::with_rights(e.object.clone(), e.rights & a[1] as u32);
            drop(t); incoming.one(d)
        }
        sys::MEM_CREATE => {
            let m = Memory::anonymous(a[0], a[1] & vm::CONTIGUOUS != 0, a[1] & vm::BELOW_4G != 0)?;
            insert(&mut incoming, Object::Memory(Arc::new(m)))
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
            insert(&mut incoming, Object::Memory(Arc::new(Memory::device(phys, size))))
        }
        sys::VM_MAP => {
            let (m, r) = {
                let t = p.handles.lock();
                let e = t.get(a[0] as u32)?;
                let Object::Memory(m) = &e.object else { return Err(Error::WrongType) };
                (m.clone(), e.rights)
            };
            let flags = a[4];
            if flags & !(vm::READ | vm::WRITE | vm::EXECUTE) != 0 { return Err(Error::InvalidArgs); }
            let max_flags = (if r & rights::READ != 0 { vm::READ } else { 0 })
                | (if r & rights::WRITE != 0 && m.writable() { vm::WRITE } else { 0 })
                | (if r & rights::EXECUTE != 0 { vm::EXECUTE } else { 0 });
            if r & rights::MAP == 0 || flags & !max_flags != 0 || (flags != 0 && flags & vm::READ == 0) {
                return Err(Error::AccessDenied);
            }
            p.aspace.map_restricted(m, a[1], a[2], a[3], flags, max_flags)
        }
        sys::VM_UNMAP => p.aspace.unmap(a[0], a[1]).map(|_| 0),
        sys::VM_MEMORY => {
            let (m, offset, flags, rest) = p.aspace.lookup_mapping(a[0]).ok_or(Error::NotFound)?;
            let mut out = offset.to_le_bytes().to_vec();
            out.extend_from_slice(&rest.to_le_bytes());
            copy_out(&p, a[1], &out)?;
            let write = if flags & vm::WRITE != 0 && m.writable() { rights::WRITE } else { 0 };
            let r = rights::MAP | rights::TRANSFER | rights::DUPLICATE | write
                | if flags & vm::READ != 0 { rights::READ } else { 0 }
                | if flags & vm::EXECUTE != 0 { rights::EXECUTE } else { 0 };
            incoming.one(Entry::with_rights(Object::Memory(m), r))
        }
        sys::SPAWN => {
            let name = copy_in(&p, a[1], a[2].min(64))?;
            let name = core::str::from_utf8(&name).map_err(|_| Error::InvalidArgs)?;
            let cpu = if a[4] == u64::MAX { sched::pick_cpu() } else { u32::try_from(a[4]).map_err(|_| Error::InvalidArgs)? };
            if cpu >= crate::smp::online() {
                return Err(Error::InvalidArgs);
            }
            let selected = if a[5] == 0 { None } else { Some(get_obj!(p.handles.lock(), u32::try_from(a[5]).map_err(|_| Error::BadHandle)?, Account, rights::WRITE)?) };
            let (image, boot) = {
                let mut t = p.handles.lock();
                let image = get_obj!(t, a[0] as u32, Memory, rights::READ)?;
                let boot = if a[3] != 0 {
                    let e = t.get(a[3] as u32)?;
                    if e.rights & rights::TRANSFER == 0 {
                        return Err(Error::AccessDenied);
                    }
                    Some(t.reserve_transfer(a[3] as u32)?)
                } else {
                    None
                };
                (image, boot)
            };
            let child = process::spawn_in(name, &image, cpu, boot.as_ref().map(|(entry, _)| entry.clone()), selected);
            if let Some((boot, token)) = boot {
                let restore = if child.is_err() { Some(boot) } else { drop(boot); None };
                let left = p.handles.lock().finish_transfer(a[3] as u32, token, restore);
                drop(left);
            }
            let child = child?;
            insert(&mut incoming, Object::Process(child))
        }
        sys::PROCESS_WAIT => {
            let c = get_obj!(p.handles.lock(), a[0] as u32, Process, rights::WAIT)?;
            let code = c.exit.wait(sched::deadline_from_ns(a[1]), |c| *c).map_err(wait_err)?;
            Ok(code as u64)
        }
        sys::CHANNEL_COLLECT => {
            let resource = get_obj!(p.handles.lock(), u32::try_from(a[0]).map_err(|_| Error::BadHandle)?, Resource, rights::WRITE)?;
            resource.check(Kind::System, 0, 1)?;
            Ok(crate::ipc::collect_cycles() as u64)
        }
        sys::CHANNEL_CREATE => {
            let (x, y) = ChannelEnd::try_pair()?;
            let hx = incoming.one(Entry::new(Object::Channel(x)))?;
            let hy = incoming.one(Entry::new(Object::Channel(y)))?;
            Ok(hx | hy << 32)
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
                }
                hs.iter().map(|&h| t.remove(h).unwrap()).collect::<Vec<_>>()
            };
            match ch.write(Message::new(bytes, handles)) {
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
            let mut ids = [0u8; MSG_HANDLES * 4];
            for (out, id) in ids.chunks_exact_mut(4).zip(incoming.ids.iter()).take(nh) { out.copy_from_slice(&id.to_le_bytes()); }
            copy_out(&p, a[3], &ids[..nh * 4])?;
            incoming.publish(m.handles)?;
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
        sys::NOTIFY_CREATE => insert(&mut incoming, Object::Notification(Arc::new(Notification::new()?))),
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
            insert(&mut incoming, Object::Resource(Arc::new(s)))
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
            let cpu = if a[3] == u64::MAX { sched::pick_cpu() } else { u32::try_from(a[3]).map_err(|_| Error::InvalidArgs)? };
            let irq = Interrupt::msi(cpu, n, a[2])?;
            let mut out = [0u8; 12];
            out[..8].copy_from_slice(&irq.msi_address.to_le_bytes());
            out[8..].copy_from_slice(&irq.msi_data.to_le_bytes());
            copy_out(&p, a[4], &out)?;
            insert(&mut incoming, Object::Interrupt(Arc::new(irq)))
        }
        sys::DMA_DOMAIN_CREATE => {
            let r = get_obj!(p.handles.lock(), a[0] as u32, Resource, 0)?;
            r.check(Kind::Iommu, 0, 1)?;
            if !crate::iommu::present() {
                return Err(Error::NotFound);
            }
            insert(&mut incoming, Object::Domain(Arc::new(Domain::new()?)))
        }
        sys::DMA_ATTACH => {
            let t = p.handles.lock();
            let d = get_obj!(t, a[0] as u32, Domain, 0)?;
            let r = get_obj!(t, a[1] as u32, Resource, 0)?;
            drop(t);
            r.check(Kind::Iommu, 0, 1)?;
            let bdf = u16::try_from(a[2]).map_err(|_| Error::InvalidArgs)?;
            d.attach(bdf).map(|_| 0)
        }
        sys::DMA_MAP => {
            let t = p.handles.lock();
            let d = get_obj!(t, a[0] as u32, Domain, rights::MAP)?;
            let e = t.get(a[1] as u32)?;
            let Object::Memory(m) = &e.object else { return Err(Error::WrongType) };
            if a[5] & !(dma::READ | dma::WRITE) != 0 { return Err(Error::InvalidArgs); }
            if e.rights & rights::MAP == 0 || (a[5] & dma::READ != 0 && e.rights & rights::READ == 0)
                || (a[5] & dma::WRITE != 0 && (e.rights & rights::WRITE == 0 || !m.writable())) {
                return Err(Error::AccessDenied);
            }
            let m = m.clone();
            drop(t);
            d.map(m, a[2], a[3], a[4], a[5]).map(|_| 0)
        }
        sys::DMA_UNMAP => {
            let d = get_obj!(p.handles.lock(), a[0] as u32, Domain, rights::MAP)?;
            d.unmap(a[1], a[2]).map(|_| 0)
        }
        sys::SPAWN_LINUX | sys::SPAWN_LINUX_IN => {
            let (a, selected) = if n == sys::SPAWN_LINUX_IN {
                let account = get_obj!(p.handles.lock(), u32::try_from(a[0]).map_err(|_| Error::BadHandle)?, Account, rights::WRITE)?;
                let bytes = copy_in(&p, a[1], 48)?;
                let mut args = [0u64; 6];
                for (arg, bytes) in args.iter_mut().zip(bytes.chunks_exact(8)) { *arg = u64::from_le_bytes(bytes.try_into().unwrap()); }
                (args, Some(account))
            } else { (a, None) };
            let argv = copy_in(&p, a[1], a[2].min(64 << 10))?;
            let envp = copy_in(&p, a[3], a[4].min(64 << 10))?;
            let (image, boot) = {
                let mut t = p.handles.lock();
                let image = get_obj!(t, a[0] as u32, Memory, rights::READ)?;
                let boot = if a[5] != 0 {
                    if t.get(a[5] as u32)?.rights & rights::TRANSFER == 0 {
                        return Err(Error::AccessDenied);
                    }
                    Some(t.reserve_transfer(a[5] as u32)?)
                } else {
                    None
                };
                (image, boot)
            };
            let child = crate::linux::spawn_in(&image, &argv, &envp, boot.as_ref().map(|(entry, _)| entry.clone()), selected);
            if let Some((boot, token)) = boot {
                let restore = if child.is_err() { Some(boot) } else { drop(boot); None };
                let left = p.handles.lock().finish_transfer(a[5] as u32, token, restore);
                drop(left);
            }
            let child = child?;
            insert(&mut incoming, Object::Process(child))
        }
        sys::HANDLE_FD => {
            if p.linux.is_none() {
                return Err(Error::WrongType);
            }
            let (object, held_rights) = {
                let handles = p.handles.lock();
                let entry = handles.get(a[0] as u32)?;
                (entry.object.clone(), entry.rights)
            };
            match object {
                #[cfg(feature = "unix")]
                Object::Channel(ch) if held_rights & (rights::READ | rights::WRITE) == (rights::READ | rights::WRITE) => crate::linux::unix::channel::fd(&p, ch),
                Object::Notification(n) if held_rights & rights::WAIT != 0 => crate::linux::notification_fd(&p, n),
                Object::Memory(m) if held_rights & rights::MAP != 0 => crate::linux::memory_fd(&p, m, held_rights),
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
