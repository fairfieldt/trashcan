//! The maintenance thread: while the application does not drive a link (it computes, or waits elsewhere), this
//! thread keeps its protocol going every millisecond: it reads what arrives (so the other side's sends keep their
//! credits), acknowledges, answers keepalives, retransmits, and opens stream devices for new epochs, which can take
//! a second and never happens on an application thread.
use super::{proto::Sink, Inner};
use std::{
    sync::{Arc, Mutex, Once, Weak},
    time::{Duration, Instant},
};

/// The application drove the link this recently: leave it alone.
const APP_QUIET: Duration = Duration::from_millis(1);
const TICK: Duration = Duration::from_millis(1);

static LINKS: Mutex<Vec<Weak<Inner>>> = Mutex::new(Vec::new());
static START: Once = Once::new();

pub fn register(link: &Arc<Inner>) {
    LINKS.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(link));
    START.call_once(|| {
        std::thread::Builder::new().name("link-service".into()).spawn(run).expect("link service thread");
    });
}

fn run() {
    loop {
        let links: Vec<Arc<Inner>> = {
            let mut l = LINKS.lock().unwrap_or_else(|e| e.into_inner());
            l.retain(|w| w.strong_count() > 0);
            l.iter().filter_map(Weak::upgrade).collect()
        };
        let mut busy = false;
        for link in &links {
            if let Ok(mut st) = link.st.try_lock() {
                let now = Instant::now();
                if !st.is_closed() && now - st.last_app >= APP_QUIET {
                    busy |= st.pump(now, &mut Sink::none()).unwrap_or(false);
                }
            }
            // Opening a device waits for an earlier close of it (half a second or more): never on this thread.
            if let Some(job) = link.take_job() {
                let link = link.clone();
                let _ = std::thread::Builder::new().name("link-open".into()).spawn(move || link.run_job(job));
            }
        }
        drop(links);
        if busy {
            std::thread::yield_now();
        } else {
            std::thread::sleep(TICK);
        }
    }
}
