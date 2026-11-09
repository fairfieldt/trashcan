//! Trusted Dross bootstrap lifecycle. HTTP has no shutdown endpoint.
use anyhow::Result;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, Weak,
};

static STOP: AtomicBool = AtomicBool::new(false);
static HTTP: Mutex<Option<Weak<tiny_http::Server>>> = Mutex::new(None);

pub fn stopping() -> bool {
    STOP.load(Ordering::Acquire)
}

pub fn register_server(server: &Arc<tiny_http::Server>) {
    *HTTP.lock().unwrap() = Some(Arc::downgrade(server));
    // Covers a request received before or while the listener was installed.
    if stopping() {
        server.unblock();
    }
}

#[cfg(feature = "dross")]
fn request_stop() {
    STOP.store(true, Ordering::Release);
    if let Some(server) = HTTP.lock().unwrap().as_ref().and_then(Weak::upgrade) {
        server.unblock();
    }
}

pub struct Control {
    #[cfg(feature = "dross")]
    notification: dross_rt::Handle,
    #[cfg(feature = "dross")]
    worker: std::thread::JoinHandle<Result<DrossTransport>>,
}

/// Read before any DRM call, so the shim consumes DROSS_HANDLES rather than
/// racing for handle 1. A Dross build still runs normally under Linux/VFIO.
pub fn start() -> Result<Option<Control>> {
    #[cfg(feature = "dross")]
    if std::env::var("DROSS_BOOTSTRAP").as_deref() == Ok("1") {
        use dross_rt::{bootstrap::Bootstrap, sys, Handle};
        let boot = Bootstrap::read(Handle(1)).map_err(|e| anyhow::anyhow!("bootstrap: {e:?}"))?;
        let channel = boot
            .handle("control")
            .ok_or_else(|| anyhow::anyhow!("server requires trusted control"))?;
        eprintln!(
            "trashcan: dross source {}; trusted control ready",
            option_env!("TRASHCAN_BUILD_SHA").unwrap_or("unrecorded")
        );
        let notification =
            sys::notify_create().map_err(|e| anyhow::anyhow!("control notify: {e:?}"))?;
        let transport = DrossTransport {
            channel,
            notification,
        };
        sys::channel_bind(channel, notification, 1)
            .map_err(|e| anyhow::anyhow!("control bind: {e:?}"))?;
        let worker = std::thread::Builder::new()
            .name("server-control".into())
            .spawn(move || wait_for_finish(transport, request_stop))?;
        return Ok(Some(Control {
            notification,
            worker,
        }));
    }
    Ok(None)
}

/// Called only after the serving scope and all its model/device owners dropped.
/// The supervisor must also wait for normal process exit before draining gpud.
pub fn finish(control: Option<Control>, success: bool) -> Result<()> {
    *HTTP.lock().unwrap() = None;
    #[cfg(feature = "dross")]
    if let Some(control) = control {
        dross_rt::sys::notify_signal(control.notification, 2)
            .map_err(|e| anyhow::anyhow!("control finish: {e:?}"))?;
        let mut transport = control
            .worker
            .join()
            .map_err(|_| anyhow::anyhow!("control worker panicked"))??;
        transport.reply(success)?;
    }
    #[cfg(not(feature = "dross"))]
    let _ = (control, success);
    Ok(())
}

#[cfg(any(feature = "dross", test))]
enum Event {
    Stop,
    Finished,
}
#[cfg(any(feature = "dross", test))]
trait Transport: Send {
    fn next(&mut self) -> Result<Event>;
    fn reply(&mut self, success: bool) -> Result<()>;
}

#[cfg(any(feature = "dross", test))]
fn wait_for_finish<T: Transport>(mut transport: T, stopping: impl Fn()) -> Result<T> {
    loop {
        match transport.next() {
            Ok(Event::Stop) => stopping(),
            Ok(Event::Finished) => return Ok(transport),
            // A lost/invalid control asks the model to stop, but never claims
            // successful drain or acknowledges a fabricated stop request.
            Err(error) => {
                stopping();
                return Err(error);
            }
        }
    }
}

#[cfg(feature = "dross")]
struct DrossTransport {
    channel: dross_rt::Handle,
    notification: dross_rt::Handle,
}
#[cfg(feature = "dross")]
impl Transport for DrossTransport {
    fn next(&mut self) -> Result<Event> {
        use dross_rt::sys;
        loop {
            let bits = sys::notify_wait(self.notification, dross_rt::FOREVER)
                .map_err(|e| anyhow::anyhow!("control wait: {e:?}"))?;
            if bits & 2 != 0 {
                return Ok(Event::Finished);
            }
            if bits & 1 == 0 {
                continue;
            }
            let mut bytes = [0u8; 4];
            let got = sys::channel_read(self.channel, &mut bytes, &mut [], 0);
            anyhow::ensure!(
                got == Ok((4, 0)) && &bytes == b"stop",
                "invalid or disconnected control: {got:?}"
            );
            return Ok(Event::Stop);
        }
    }
    fn reply(&mut self, success: bool) -> Result<()> {
        let status = if success { 0i64 } else { -1i64 };
        dross_rt::sys::channel_write(self.channel, &status.to_le_bytes(), &[])
            .map_err(|e| anyhow::anyhow!("control reply: {e:?}"))
    }
}
#[cfg(feature = "dross")]
impl Drop for DrossTransport {
    fn drop(&mut self) {
        let _ = dross_rt::sys::close(self.channel);
        let _ = dross_rt::sys::close(self.notification);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    struct Mock {
        input: mpsc::Receiver<Result<Event>>,
        replies: mpsc::Sender<bool>,
    }
    impl Transport for Mock {
        fn next(&mut self) -> Result<Event> {
            self.input.recv()?
        }
        fn reply(&mut self, success: bool) -> Result<()> {
            self.replies.send(success)?;
            Ok(())
        }
    }
    #[test]
    fn stop_does_not_acknowledge_before_owners_finish() {
        let (send, input) = mpsc::channel();
        let (replies, receive) = mpsc::channel();
        let stopped = Arc::new(AtomicBool::new(false));
        let observed = stopped.clone();
        let worker = std::thread::spawn(move || {
            wait_for_finish(Mock { input, replies }, || {
                observed.store(true, Ordering::Release)
            })
        });
        send.send(Ok(Event::Stop)).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !stopped.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "stop was not delivered"
            );
            std::thread::yield_now();
        }
        assert_eq!(receive.try_recv(), Err(mpsc::TryRecvError::Empty));
        send.send(Ok(Event::Finished)).unwrap();
        let mut transport = worker.join().unwrap().unwrap();
        transport.reply(true).unwrap();
        assert!(receive.recv().unwrap());
    }
    #[test]
    fn disconnected_control_stops_without_success_reply() {
        let (send, input) = mpsc::channel();
        let (replies, receive) = mpsc::channel();
        let stopped = AtomicBool::new(false);
        send.send(Err(anyhow::anyhow!("peer closed"))).unwrap();
        assert!(wait_for_finish(Mock { input, replies }, || stopped
            .store(true, Ordering::Release))
        .is_err());
        assert!(stopped.load(Ordering::Acquire));
        assert_eq!(receive.try_recv(), Err(mpsc::TryRecvError::Disconnected));
    }
}
