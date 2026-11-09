//! Privileged S3 boundary. This does not freeze drivers or prepare AML/RTC.
use crate::Error;
pub const VERSION: u32 = 1;
pub const PREFLIGHT: u32 = 0;
pub const ENTER: u32 = 1;
/// For PREFLIGHT this means validation only. Only an ENTER return permits
/// pcid's Restore: IRQ/IOMMU/CPU restoration completed with driver BME off.
pub const KERNEL_READY: u64 = 1;
pub const RESUMED: u64 = 2;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Request {
    pub version: u32,
    pub operation: u32,
    pub flags: u64,
    pub earliest_fire_ns: u64,
    pub pm1a_sleep_value: u64,
    pub pm1b_sleep_value: u64,
    pub reserved: [u64; 3],
}
impl Request {
    pub const BYTES: usize = 64;
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != Self::BYTES {
            return Err(Error::InvalidArgs);
        }
        let d = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let q = |at| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        Ok(Self {
            version: d(0),
            operation: d(4),
            flags: q(8),
            earliest_fire_ns: q(16),
            pm1a_sleep_value: q(24),
            pm1b_sleep_value: q(32),
            reserved: [q(40), q(48), q(56)],
        })
    }
    pub fn validate(&self, now: u64) -> Result<(), Error> {
        if self.version != VERSION
            || self.operation > ENTER
            || self.flags != 0
            || self.reserved != [0; 3]
        {
            return Err(Error::InvalidArgs);
        }
        if now
            .checked_add(1_000_000_000)
            .is_none_or(|minimum| self.earliest_fire_ns <= minimum)
        {
            return Err(Error::TimedOut);
        }
        if now
            .checked_add(30_000_000_000)
            .is_none_or(|maximum| self.earliest_fire_ns > maximum)
        {
            return Err(Error::InvalidArgs);
        }
        Ok(())
    }
}
/// The private AML owner already wrote S3 SLP_TYP with EN clear. Kernel only
/// accepts the same live 16-bit value plus EN, with SCI enabled and GBL_RLS
/// clear. The system-capability owner attests that AML selected S3, not S4/S5.
pub fn validate_control(live: u16, requested: u64) -> Result<(), Error> {
    if live & 1 == 0 || live & (0x2000 | 4) != 0 || requested != (live as u64 | 0x2000) {
        Err(Error::InvalidArgs)
    } else {
        Ok(())
    }
}
/// A translated device must not retain a device-side translation requester
/// while the IOMMU is stopped. Reject malformed/cyclic capability lists too.
pub fn check_extended_caps(
    mut read32: impl FnMut(usize) -> u32,
    mut read16: impl FnMut(usize) -> u16,
) -> Result<(), Error> {
    let mut at = 0x100;
    let mut seen = [0u64; 16];
    loop {
        if !(0x100..=0xffc).contains(&at) || at & 3 != 0 {
            return Err(Error::InvalidArgs);
        }
        let word = at / 4;
        let mask = 1u64 << (word % 64);
        if seen[word / 64] & mask != 0 {
            return Err(Error::InvalidArgs);
        }
        seen[word / 64] |= mask;
        let header = read32(at);
        if header == 0 && at == 0x100 {
            return Ok(());
        }
        if header == u32::MAX {
            return Err(Error::NotFound);
        }
        if header & 0xffff == 0 || header & 0xffff == 0xffff || header >> 16 & 15 == 0 {
            return Err(Error::InvalidArgs);
        }
        let enabled = match header & 0xffff {
            0x000f => {
                if at + 8 > 4096 {
                    return Err(Error::InvalidArgs);
                }
                read16(at + 6) & 0x8000 != 0
            }
            0x0013 => {
                if at + 6 > 4096 {
                    return Err(Error::InvalidArgs);
                }
                read16(at + 4) & 1 != 0
            }
            0x001b => {
                if at + 8 > 4096 {
                    return Err(Error::InvalidArgs);
                }
                read16(at + 6) & 1 != 0
            }
            _ => false,
        };
        if enabled {
            return Err(Error::Busy);
        }
        let next = (header >> 20) as usize;
        if next == 0 {
            return Ok(());
        }
        at = next;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn control_accepts_only_prepared_value_plus_enable() {
        assert_eq!(validate_control(0xc01, 0x2c01), Ok(()));
        for (live, request) in [
            (0xc00, 0x2c00),
            (0x2c01, 0x2c01),
            (0xc05, 0x2c05),
            (0xc01, 0x3001),
            (0xc01, 0x12c01),
        ] {
            assert_eq!(validate_control(live, request), Err(Error::InvalidArgs));
        }
    }
    #[test]
    fn version_reserved_and_recovery_deadline_are_checked() {
        let mut r = Request {
            version: 1,
            earliest_fire_ns: 19_000_000_000,
            ..Request::default()
        };
        assert_eq!(r.validate(0), Ok(()));
        r.reserved[2] = 1;
        assert_eq!(r.validate(0), Err(Error::InvalidArgs));
        r.reserved = [0; 3];
        assert_eq!(r.validate(18_000_000_000), Err(Error::TimedOut));
        r.earliest_fire_ns = 31_000_000_000;
        assert_eq!(r.validate(0), Err(Error::InvalidArgs));
        assert!(Request::decode(&[0; 63]).is_err());
        assert_eq!(core::mem::size_of::<Request>(), 64);
    }
    #[test]
    fn active_requesters_and_capability_cycles_fail_closed() {
        for (id, offset, bit) in [(0xf, 6, 0x8000), (0x13, 4, 1), (0x1b, 6, 1)] {
            assert_eq!(
                check_extended_caps(
                    |_| id | 1 << 16,
                    |at| if at == 0x100 + offset { bit } else { 0 }
                ),
                Err(Error::Busy)
            );
            assert_eq!(check_extended_caps(|_| id | 1 << 16, |_| 0), Ok(()));
        }
        assert_eq!(
            check_extended_caps(|_| 0x1001_0010, |_| 0),
            Err(Error::InvalidArgs)
        );
        assert_eq!(check_extended_caps(|_| 0x10010, |_| 0), Ok(()));
        assert_eq!(
            check_extended_caps(|_| u32::MAX, |_| 0),
            Err(Error::NotFound)
        );
        assert_eq!(
            check_extended_caps(|_| 0x10, |_| 0),
            Err(Error::InvalidArgs)
        );
        assert_eq!(
            check_extended_caps(|at| if at == 0x100 { 0x1201_0010 } else { 0 }, |_| 0),
            Err(Error::InvalidArgs)
        );
    }
}
