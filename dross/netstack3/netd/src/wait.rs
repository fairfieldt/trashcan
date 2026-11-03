//! One-shot kernel readiness registrations. Queue ownership stays in Daemon,
//! declared after endpoints so every registration is cancelled before its queue
//! closes. A registration follows one owned endpoint, never a reused raw slot.
use dross_rt::{
    Error, Handle,
    owned::{BorrowedHandle, Untyped},
    sys,
};
pub struct Registration {
    queue: Handle,
    id: u64,
    signals: u64,
}
impl Drop for Registration {
    fn drop(&mut self) {
        let _ = sys::subscription_cancel(self.queue, self.id);
    }
}
impl Registration {
    pub fn new(
        queue: Handle,
        source: BorrowedHandle<'_, Untyped>,
        signals: u64,
    ) -> Result<Self, Error> {
        let id = sys::object_subscribe(queue, source.as_raw(), signals, 0)?;
        Ok(Self { queue, id, signals })
    }
    pub fn update(
        slot: &mut Option<Self>,
        queue: Handle,
        source: BorrowedHandle<'_, Untyped>,
        signals: u64,
        delivered: Option<u64>,
    ) -> Result<(), Error> {
        if slot.as_ref().is_some_and(|s| s.signals != signals) {
            slot.take();
        }
        if let Some(s) = slot {
            if delivered == Some(s.id) {
                sys::subscription_rearm(queue, s.id)?;
            }
            return Ok(());
        }
        *slot = Some(Self::new(queue, source, signals)?);
        Ok(())
    }
}
