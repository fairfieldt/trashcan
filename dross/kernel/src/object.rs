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
}

impl Entry {
    pub fn new(object: Object) -> Entry {
        let r = match &object {
            Object::Memory(m) if !m.writable() => rights::ALL & !rights::WRITE,
            _ => rights::ALL,
        };
        Entry { object, rights: r }
    }
}

pub struct HandleTable {
    slots: Vec<Slot>,
    next_transfer: u64,
}

enum Slot { Empty, Occupied(Entry), Reserved(u64) }

impl HandleTable {
    pub fn new() -> Self {
        // Slot 0 is never a handle.
        HandleTable { slots: alloc::vec![Slot::Empty], next_transfer: 0 }
    }

    pub fn insert(&mut self, e: Entry) -> u32 {
        if let Some(i) = self.slots.iter().skip(1).position(|s| matches!(s, Slot::Empty)) {
            self.slots[i + 1] = Slot::Occupied(e);
            return (i + 1) as u32;
        }
        self.slots.push(Slot::Occupied(e));
        (self.slots.len() - 1) as u32
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
    pub fn clear(&mut self) -> Vec<Entry> {
        self.slots.drain(1..).filter_map(|s| match s { Slot::Occupied(e) => Some(e), _ => None }).collect()
    }
}

#[cfg(feature = "audit-test")]
pub fn audit_test() {
    let mut t = HandleTable::new();
    let entry = || Entry::new(Object::Notification(Arc::new(Notification::new())));
    let h = t.insert(entry());
    let (old, token) = t.reserve_transfer(h).unwrap();
    let next = t.insert(entry());
    assert_ne!(h, next);
    assert!(t.finish_transfer(h, token, Some(old)).is_none());
    assert!(t.get(h).is_ok());
    let (old, old_token) = t.reserve_transfer(h).unwrap();
    drop(t.clear());
    assert_eq!(t.insert(entry()), h);
    let (current, current_token) = t.reserve_transfer(h).unwrap();
    assert!(t.finish_transfer(h, old_token, Some(old)).is_some());
    assert!(t.get(h).is_err());
    assert!(t.finish_transfer(h, current_token, Some(current)).is_none());
    assert!(t.get(h).is_ok());
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
