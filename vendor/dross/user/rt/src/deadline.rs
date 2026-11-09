//! Monotonic absolute deadlines and relative intervals.
//!
//! Compute a deadline once, then pass it unchanged through retries. A timeout
//! ends the wait; it does not cancel a previously submitted service operation.

/// A relative interval, in nanoseconds. Conversions saturate rather than wrap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration(u64);
impl Duration {
    pub const ZERO: Self = Self(0);
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis.saturating_mul(1_000_000))
    }
    pub const fn from_secs(seconds: u64) -> Self {
        Self(seconds.saturating_mul(1_000_000_000))
    }
    pub const fn as_nanos(self) -> u64 {
        self.0
    }
}

/// An absolute instant on Oscar's monotonic clock, or an infinite wait.
/// Wall-clock timestamps must not be passed to this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deadline(u64);
impl Deadline {
    /// Check current state and return immediately if it is not ready.
    pub const IMMEDIATE: Self = Self(0);
    pub const FOREVER: Self = Self(u64::MAX);
    /// Construct from CLOCK nanoseconds. `u64::MAX` means FOREVER.
    pub const fn at_monotonic_ns(nanos: u64) -> Self {
        Self(nanos)
    }
    pub const fn as_monotonic_ns(self) -> u64 {
        self.0
    }
    /// Saturating absolute deadline from a supplied monotonic clock reading.
    /// Useful when several operations must share the same start instant.
    pub const fn after_at(now_ns: u64, interval: Duration) -> Self {
        Self(now_ns.saturating_add(interval.0))
    }
    pub fn from_now(interval: Duration) -> Self {
        Self::after_at(crate::sys::clock(), interval)
    }
    pub const fn is_forever(self) -> bool {
        self.0 == u64::MAX
    }
    pub fn has_elapsed(self) -> bool {
        !self.is_forever() && crate::sys::clock() >= self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn units_and_overflow() {
        assert_eq!(Duration::from_millis(3).as_nanos(), 3_000_000);
        assert_eq!(Duration::from_secs(u64::MAX).as_nanos(), u64::MAX);
        assert_eq!(
            Deadline::after_at(u64::MAX - 2, Duration::from_nanos(3)),
            Deadline::FOREVER
        );
        assert_eq!(
            Deadline::after_at(7, Duration::from_nanos(9)).as_monotonic_ns(),
            16
        );
        assert_eq!(Deadline::at_monotonic_ns(0), Deadline::IMMEDIATE);
    }
}
