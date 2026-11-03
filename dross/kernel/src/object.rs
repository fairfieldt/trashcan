//! Handles: a process names kernel objects by small integers, each with rights. Holding a handle is the
//! permission to use the object; resources (device memory, ports, interrupts, the IOMMU) are objects too.

use alloc::sync::Arc;
use alloc::vec::Vec;
use dross_abi::{rights, Error};

use crate::ipc::{ChannelEnd, Notification};
use crate::iommu::Domain;
use crate::irq::Interrupt;
use crate::memory::Memory;
use crate::process::Process;
use crate::resource::Resource;
use crate::sched::Thread;

/// Variants that are never matched on still keep their object alive while a handle names it.
#[derive(Clone)]
#[allow(dead_code)]
pub enum Object {
    Timer(Arc<crate::timer::Timer>),
    EventQueue(Arc<crate::events::EventQueue>),
    Account(Arc<crate::account::Account>),
    Process(Arc<Process>),
    Thread(Arc<Thread>),
    Memory(Arc<Memory>),
    Channel(Arc<ChannelEnd>),
    Notification(Arc<Notification>),
    Resource(Arc<Resource>),
    Interrupt(Arc<Interrupt>),
    Domain(Arc<Domain>),
}

#[derive(Clone)]
pub struct Entry {
    pub object: Object,
    pub rights: u32,
    pub life: Option<crate::events::HandleLease>,
}

impl Entry {
    pub fn with_rights(object: Object, rights: u32) -> Self {
        let life = matches!(object, Object::Channel(_) | Object::Notification(_) | Object::Process(_) | Object::Thread(_) | Object::Timer(_)).then(crate::events::HandleLease::new);
        Self { object, rights, life }
    }

    pub fn new(object: Object) -> Entry {
        let r = match &object {
            Object::Memory(m) if !m.writable() => rights::ALL & !rights::WRITE,
            _ => rights::ALL,
        };
        Self::with_rights(object, r)
    }
}

pub struct HandleTable {
    slots: Vec<Slot>,
    next_transfer: u64,
    account: Arc<crate::account::Account>,
    storage: Option<crate::account::Charge>,
}

enum Slot { Empty, Occupied(Entry), Reserved(u64) }

/// Keeps released capabilities alive until after the table lock is dropped.
pub struct ClosedHandles { _slots: Vec<Slot>, _storage: Option<crate::account::Charge> }

impl HandleTable {
    pub fn new(account: Arc<crate::account::Account>) -> Self {
        HandleTable { slots: Vec::new(), next_transfer: 0, account, storage: None }
    }

    /// Reserve real slot identities before a fallible or blocking operation.
    /// Capacity includes a conservative allowance for each logical handle's
    /// identity metadata, and remains charged until the table is released.
    pub fn reserve_handles(&mut self, count: usize) -> Result<(u64, [u32; dross_abi::MSG_HANDLES]), Error> {
        const LIMIT: usize = 65536;
        if count > dross_abi::MSG_HANDLES { return Err(Error::InvalidArgs); }
        let mut ids = [0; dross_abi::MSG_HANDLES];
        if count == 0 { return Ok((0, ids)); }
        let empty = self.slots.iter().skip(1).filter(|s| matches!(s, Slot::Empty)).count();
        let required = self.slots.len().max(1).checked_add(count.saturating_sub(empty)).ok_or(Error::NoResources)?;
        if required > LIMIT { return Err(Error::NoResources); }
        if required > self.slots.capacity() {
            let capacity = required.max(self.slots.capacity().saturating_mul(2)).max(16).min(LIMIT);
            let bytes = (capacity - self.slots.capacity()) * (core::mem::size_of::<Slot>() + 128);
            let charge = self.account.reserve(bytes as u64)?;
            self.slots.try_reserve_exact(capacity - self.slots.len()).map_err(|_| Error::NoMemory)?;
            if let Some(storage) = &mut self.storage { storage.absorb(charge); } else { self.storage = Some(charge); }
        }
        let token = self.next_transfer.checked_add(1).ok_or(Error::NoResources)?;
        self.next_transfer = token;
        if self.slots.is_empty() { self.slots.push(Slot::Empty); }
        for id in ids.iter_mut().take(count) {
            let index = if let Some(index) = self.slots.iter().skip(1).position(|s| matches!(s, Slot::Empty)) { index + 1 }
                else { self.slots.push(Slot::Empty); self.slots.len() - 1 };
            self.slots[index] = Slot::Reserved(token); *id = index as u32;
        }
        Ok((token, ids))
    }
    /// Return unused trailing capacity when a table becomes sparse. A failed
    /// compaction leaves existing handles and their storage charge unchanged.
    pub fn compact(&mut self) {
        let len = self.slots.iter().rposition(|slot| !matches!(slot, Slot::Empty)).map_or(0, |i| i + 1);
        if len == 0 {
            self.slots = Vec::new(); self.storage.take(); return;
        }
        let old_capacity = self.slots.capacity();
        if old_capacity <= 64 || len > old_capacity / 4 { return; }
        let capacity = len.next_power_of_two().max(16);
        let mut slots = Vec::new();
        if slots.try_reserve_exact(capacity).is_err() { return; }
        self.slots.truncate(len);
        slots.extend(self.slots.drain(..));
        let released = (old_capacity - slots.capacity()) * (core::mem::size_of::<Slot>() + 128);
        self.slots = slots;
        drop(self.storage.as_mut().expect("handle storage charge").split(released as u64));
    }

    pub fn reserved(&self, handle: u32, token: u64) -> bool {
        matches!(self.slots.get(handle as usize), Some(Slot::Reserved(id)) if *id == token)
    }
    pub fn insert(&mut self, entry: Entry) -> Result<u32, Error> {
        let (token, ids) = self.reserve_handles(1)?;
        assert!(self.finish_transfer(ids[0], token, Some(entry)).is_none());
        Ok(ids[0])
    }

    pub fn get(&self, h: u32) -> Result<&Entry, Error> {
        match self.slots.get(h as usize) { Some(Slot::Occupied(e)) => Ok(e), _ => Err(Error::BadHandle) }
    }

    pub fn remove(&mut self, h: u32) -> Result<Entry, Error> {
        let slot = self.slots.get_mut(h as usize).ok_or(Error::BadHandle)?;
        if !matches!(slot, Slot::Occupied(_)) { return Err(Error::BadHandle); }
        match core::mem::replace(slot, Slot::Empty) { Slot::Occupied(e) => Ok(e), _ => unreachable!() }
    }

    /// Restore a removed handle while the table lock has remained held.
    pub fn restore(&mut self, h: u32, e: Entry) {
        let slot = &mut self.slots[h as usize];
        assert!(matches!(slot, Slot::Empty));
        *slot = Slot::Occupied(e);
    }

    /// Keep a moved handle's number unavailable while a fallible spawn runs.
    pub fn reserve_transfer(&mut self, h: u32) -> Result<(Entry, u64), Error> {
        let slot = self.slots.get_mut(h as usize).ok_or(Error::BadHandle)?;
        if !matches!(slot, Slot::Occupied(_)) { return Err(Error::BadHandle); }
        self.next_transfer = self.next_transfer.checked_add(1).ok_or(Error::NoResources)?;
        let token = self.next_transfer;
        match core::mem::replace(slot, Slot::Reserved(token)) { Slot::Occupied(e) => Ok((e, token)), _ => unreachable!() }
    }

    /// Restore on failure, or release the reserved number on success. Process exit can cancel it.
    pub fn finish_transfer(&mut self, h: u32, token: u64, entry: Option<Entry>) -> Option<Entry> {
        let Some(slot) = self.slots.get_mut(h as usize) else { return entry };
        if matches!(&*slot, Slot::Reserved(id) if *id == token) {
            *slot = entry.map_or(Slot::Empty, Slot::Occupied);
            None
        } else { entry }
    }

    /// Drops every handle (process exit); returns them so they are dropped outside the table's lock.
    pub fn clear(&mut self) -> ClosedHandles {
        ClosedHandles { _slots: core::mem::take(&mut self.slots), _storage: self.storage.take() }
    }
}

#[cfg(feature = "audit-test")]
pub fn audit_test() {
    let mut t = HandleTable::new(crate::account::current());
    let entry = || Entry::new(Object::Notification(Arc::new(Notification::new().unwrap())));
    let h = t.insert(entry()).unwrap();
    let (old, token) = t.reserve_transfer(h).unwrap();
    let next = t.insert(entry()).unwrap();
    assert_ne!(h, next);
    assert!(t.finish_transfer(h, token, Some(old)).is_none());
    assert!(t.get(h).is_ok());
    let (old, old_token) = t.reserve_transfer(h).unwrap();
    drop(t.clear());
    assert_eq!(t.insert(entry()).unwrap(), h);
    let (current, current_token) = t.reserve_transfer(h).unwrap();
    assert!(t.finish_transfer(h, old_token, Some(old)).is_some());
    assert!(t.get(h).is_err());
    assert!(t.finish_transfer(h, current_token, Some(current)).is_none());
    assert!(t.get(h).is_ok());
    let budget = crate::account::Account::new(32 << 10);
    let object = Arc::new(Notification::new().unwrap());
    let mut table = HandleTable::new(budget.clone());
    let mut count = 0;
    loop {
        match table.insert(Entry::new(Object::Notification(object.clone()))) {
            Ok(_) => count += 1,
            Err(Error::NoMemory) => break,
            result => panic!("handle accounting: {result:?}"),
        }
    }
    assert!(count > 0 && count < 1024);
    assert!(budget.stats()[0] <= 32 << 10);
    drop(table.clear()); assert_eq!(budget.stats()[0], 0);
    crate::println!("handles: bounded fallible storage charges and rollback passed");
    crate::println!("handles: transfer reservation tests passed");
}

/// Gets the object of handle `h` if it is of the wanted kind and the handle has `need` rights.
macro_rules! get_obj {
    ($table:expr, $h:expr, $kind:ident, $need:expr) => {{
        let table = &$table;
        let e = table.get($h)?;
        if e.rights & ($need) != ($need) {
            Err(dross_abi::Error::AccessDenied)
        } else if let $crate::object::Object::$kind(o) = &e.object {
            Ok(o.clone())
        } else {
            Err(dross_abi::Error::WrongType)
        }
    }};
}
pub(crate) use get_obj;
