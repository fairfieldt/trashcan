//! Device interrupts as notifications: an MSI vector on a chosen CPU signals bits on a notification. The driver
//! (or pcid for it) writes the returned address and data into the device's MSI capability. Without interrupt
//! remapping (the Mac Pro has none) the address names the CPU directly.

use alloc::sync::Arc;
use dross_abi::Error;
use spin::Mutex;

use crate::cpu::{self, MAX_CPUS};
use crate::ipc::Notification;

pub const FIRST_VECTOR: u8 = 0x40;
pub const LAST_VECTOR: u8 = 0xdf;

type Slot = Option<(Arc<Notification>, u64)>;
static TABLES: [Mutex<[Slot; 256]>; MAX_CPUS] = [const { Mutex::new([const { None }; 256]) }; MAX_CPUS];

pub struct Interrupt {
    cpu: u32,
    vector: u8,
    pub msi_address: u64,
    pub msi_data: u32,
}

impl Interrupt {
    pub fn msi(cpu_index: u32, notification: Arc<Notification>, bits: u64) -> Result<Interrupt, Error> {
        let pc = cpu::get(cpu_index as usize).ok_or(Error::InvalidArgs)?;
        if pc.apic_id > 0xff {
            return Err(Error::NotFound);
        }
        let mut t = TABLES[cpu_index as usize].lock();
        let v = (FIRST_VECTOR..=LAST_VECTOR).find(|&v| t[v as usize].is_none()).ok_or(Error::NoResources)?;
        t[v as usize] = Some((notification, bits));
        Ok(Interrupt {
            cpu: cpu_index,
            vector: v,
            // Fixed delivery, physical destination, edge.
            msi_address: 0xfee0_0000 | (pc.apic_id as u64) << 12,
            msi_data: v as u32,
        })
    }
}

impl Drop for Interrupt {
    fn drop(&mut self) {
        TABLES[self.cpu as usize].lock()[self.vector as usize] = None;
    }
}

/// A device vector fired on this CPU. Returns false if nothing owns it.
pub fn dispatch(vector: u8) -> bool {
    let slot = TABLES[cpu::this().index as usize].lock()[vector as usize].clone();
    match slot {
        Some((n, bits)) => {
            n.signal(bits);
            true
        }
        None => false,
    }
}
