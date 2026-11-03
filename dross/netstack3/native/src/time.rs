use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use netstack3_base::{AtomicInstant as AtomicInstantTrait, Instant as InstantTrait};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Instant(pub u64);
impl netstack3_base::InspectableValue for Instant {
    fn record<I: netstack3_base::Inspector>(&self, name: &str, inspector: &mut I) {
        inspector.record_uint(name, self.0);
    }
}
impl InstantTrait for Instant {
    fn checked_duration_since(&self, earlier: Self) -> Option<Duration> {
        self.0.checked_sub(earlier.0).map(Duration::from_nanos)
    }
    fn checked_add(&self, duration: Duration) -> Option<Self> {
        self.0
            .checked_add(u64::try_from(duration.as_nanos()).ok()?)
            .map(Self)
    }
    fn saturating_add(&self, duration: Duration) -> Self {
        Self(
            self.0
                .saturating_add(duration.as_nanos().min(u128::from(u64::MAX)) as u64),
        )
    }
    fn checked_sub(&self, duration: Duration) -> Option<Self> {
        self.0
            .checked_sub(u64::try_from(duration.as_nanos()).ok()?)
            .map(Self)
    }
}
#[derive(Debug)]
pub struct AtomicInstant(AtomicU64);
impl AtomicInstantTrait<Instant> for AtomicInstant {
    fn new(instant: Instant) -> Self {
        Self(AtomicU64::new(instant.0))
    }
    fn load(&self, ordering: Ordering) -> Instant {
        Instant(self.0.load(ordering))
    }
    fn store(&self, instant: Instant, ordering: Ordering) {
        self.0.store(instant.0, ordering);
    }
    fn store_max(&self, instant: Instant, ordering: Ordering) {
        self.0.fetch_max(instant.0, ordering);
    }
}
