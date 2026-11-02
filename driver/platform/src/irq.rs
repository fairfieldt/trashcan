use crate::Result;
use std::time::Duration;

/// A notification raised by a device interrupt (an eventfd on Linux, a notification object on a microkernel).
pub struct IrqEvent {
    inner: Box<dyn IrqWait>,
}

/// What a backend provides behind [`IrqEvent`].
pub trait IrqWait: Send + Sync {
    /// Waits until the interrupt fires at least once, then returns how many times it fired since the last wait or
    /// take (≥ 1). `None` waits forever.
    fn wait(&self, timeout: Option<Duration>) -> Result<u64>;

    /// How many times it fired since the last wait or take, without waiting (0 if not at all).
    fn take(&self) -> Result<u64>;
}

impl IrqEvent {
    pub fn new(inner: Box<dyn IrqWait>) -> Self {
        IrqEvent { inner }
    }

    /// Returns `Err(Error::Timeout)` if nothing arrives in time.
    pub fn wait(&self, timeout: Option<Duration>) -> Result<u64> {
        self.inner.wait(timeout)
    }

    pub fn take(&self) -> Result<u64> {
        self.inner.take()
    }
}
