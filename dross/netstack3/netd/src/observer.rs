//! Shared one-shot observers for socket and resolver-job controls. Events use
//! fixed storage; no allocation follows an observable state transition.
use dross_rt::{
    Error, Handle,
    owned::OwnedHandle,
    protocol::{network as wire, schema::*},
    sys,
};
#[derive(Clone, Copy)]
pub struct View {
    pub revision: u64,
    pub signals: u32,
    pub error: i32,
}
pub struct Observer {
    sink: OwnedHandle,
    id: u64,
    key: u64,
    mask: u32,
    armed: bool,
    pending: Option<[u8; 64]>,
    wait: Option<crate::wait::Registration>,
}
impl Observer {
    pub fn register(
        sink: OwnedHandle,
        queue: Handle,
        id: u64,
        key: u64,
        mask: u32,
    ) -> Result<Self, i32> {
        let wait = crate::wait::Registration::new(
            queue,
            sink.as_handle(),
            dross_abi::event::READABLE | dross_abi::event::PEER_CLOSED,
        )
        .map_err(|e| match e {
            Error::NoMemory | Error::NoResources => STATUS_LIMIT,
            Error::PeerClosed => STATUS_CLOSED,
            _ => STATUS_MALFORMED,
        })?;
        Ok(Self {
            sink,
            id,
            key,
            mask,
            armed: true,
            pending: None,
            wait: Some(wait),
        })
    }
    pub fn wait(&mut self, queue: Handle, delivered: Option<u64>) -> Result<(), Error> {
        use dross_abi::event::{PEER_CLOSED, READABLE, WRITABLE};
        let signals = READABLE | PEER_CLOSED | if self.pending.is_some() { WRITABLE } else { 0 };
        crate::wait::Registration::update(
            &mut self.wait,
            queue,
            self.sink.as_handle(),
            signals,
            delivered,
        )
    }
    fn event(&self, view: View) -> [u8; 64] {
        wire::encode_event(wire::Readiness {
            subscription: self.id,
            key: self.key,
            revision: view.revision,
            signals: view.signals,
            error: view.error,
        })
        .expect("validated service snapshot")
    }
}
pub fn notify(slot: &mut Option<Observer>, view: View) {
    let Some(mut o) = slot.take() else { return };
    let mut bytes = [0; 1];
    let mut handles = [Handle(0); 64];
    let closed = match sys::channel_read(o.sink.as_handle().as_raw(), &mut bytes, &mut handles, 0) {
        Err(Error::ShouldWait | Error::TimedOut) => false,
        Ok((_, nh)) => {
            for h in &handles[..nh] {
                let _ = sys::close(*h);
            }
            true
        }
        Err(_) => true,
    };
    if closed {
        return;
    }
    if o.armed && view.signals & o.mask != 0 {
        o.pending = Some(o.event(view));
        o.armed = false;
    }
    if let Some(bytes) = o.pending {
        match sys::channel_write(o.sink.as_handle().as_raw(), &bytes, &[]) {
            Ok(()) => o.pending = None,
            Err(Error::ShouldWait) => {}
            Err(_) => return,
        }
    }
    *slot = Some(o);
}
pub fn rearm(slot: &mut Option<Observer>, id: u64, revision: u64, view: View) -> Result<(), i32> {
    let Some(o) = slot.as_mut() else {
        return Err(STATUS_BAD_STATE);
    };
    if o.id != id || o.pending.is_some() || o.armed {
        return Err(STATUS_BAD_STATE);
    }
    let immediate = view.signals & o.mask != 0 || revision != view.revision;
    if immediate {
        o.pending = Some(o.event(view));
    }
    o.armed = !immediate;
    Ok(())
}
pub fn cancel(slot: &mut Option<Observer>, id: u64) -> Result<(), i32> {
    if slot.as_ref().is_none_or(|o| o.id != id) {
        return Err(STATUS_BAD_STATE);
    }
    slot.take();
    Ok(())
}
