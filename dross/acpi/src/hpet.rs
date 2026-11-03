//! HPET: the High Precision Event Timer's registers.

use crate::{u8_at, u16_at, u32_at, Error, Gas, Sdt};

#[derive(Clone, Copy, Debug)]
pub struct Hpet {
    /// The capabilities register as the firmware reports it (vendor, comparator count, counter size).
    pub event_timer_block_id: u32,
    pub base: Gas,
    pub number: u8,
    pub min_tick: u16,
}

impl Hpet {
    pub fn parse(t: &Sdt<'_>) -> Result<Self, Error> {
        if &t.header.signature != b"HPET" {
            return Err(Error::BadSignature);
        }
        let b = t.bytes;
        Ok(Hpet { event_timer_block_id: u32_at(b, 36)?, base: Gas::parse(b, 40)?, number: u8_at(b, 52)?, min_tick: u16_at(b, 53)? })
    }
}
