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
    slots: Vec<Option<Entry>>,
}

impl HandleTable {
    pub fn new() -> Self {
        // Slot 0 is never a handle.
        HandleTable { slots: alloc::vec![None] }
    }

    pub fn insert(&mut self, e: Entry) -> u32 {
        if let Some(i) = self.slots.iter().skip(1).position(|s| s.is_none()) {
            self.slots[i + 1] = Some(e);
            return (i + 1) as u32;
        }
        self.slots.push(Some(e));
        (self.slots.len() - 1) as u32
    }

    pub fn get(&self, h: u32) -> Result<&Entry, Error> {
        self.slots.get(h as usize).and_then(|s| s.as_ref()).ok_or(Error::BadHandle)
    }

    pub fn remove(&mut self, h: u32) -> Result<Entry, Error> {
        self.slots.get_mut(h as usize).and_then(|s| s.take()).ok_or(Error::BadHandle)
    }

    /// Restore a removed handle while the table lock has remained held.
    pub fn restore(&mut self, h: u32, e: Entry) {
        let slot = &mut self.slots[h as usize];
        assert!(slot.is_none());
        *slot = Some(e);
    }

    /// Drops every handle (process exit); returns them so they are dropped outside the table's lock.
    pub fn clear(&mut self) -> Vec<Entry> {
        self.slots.drain(..).flatten().collect()
    }
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
