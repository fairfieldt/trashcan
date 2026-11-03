use super::*;
use dross_abi::{event as e, EVENT_LIMIT};
fn take(q: Handle, id: u64, key: u64, bit: u64) {
    let event = sys::event_queue_wait(q, sys::after_ms(5000)).unwrap();
    assert_eq!(event.subscription, id); assert_eq!(event.key, key); assert_ne!(event.observed & bit, 0);
}
fn empty(q: Handle) { assert_eq!(sys::event_queue_wait(q, 0), Err(Error::TimedOut)); }
fn close(handles: &[Handle]) { for &h in handles { sys::close(h).unwrap(); } }
pub fn run(t: &mut Tally, boot: &Bootstrap) {
    let q = sys::event_queue_create(4).unwrap();
    let (a, b) = sys::channel_create().unwrap();
    let id = sys::object_subscribe(q, a, e::WRITABLE, 7).unwrap();
    assert_eq!(sys::subscription_rearm(q, id), Err(Error::Busy));
    take(q, id, 7, e::WRITABLE); empty(q);
    sys::subscription_rearm(q, id).unwrap(); take(q, id, 7, e::WRITABLE);
    sys::subscription_cancel(q, id).unwrap();
    t.check("event register-ready and one-shot rearm", true, "");

    let id = sys::object_subscribe(q, a, e::READABLE, 8).unwrap();
    assert_eq!(sys::event_queue_wait(q, sys::after_ms(5)), Err(Error::TimedOut));
    sys::channel_write(b, b"x", &[]).unwrap();
    assert_eq!(sys::event_queue_wait(q, 0).unwrap().subscription, id);
    sys::subscription_rearm(q, id).unwrap(); take(q, id, 8, e::READABLE);
    sys::channel_read(a, &mut [0], &mut [], 0).unwrap();
    sys::subscription_rearm(q, id).unwrap(); empty(q);
    sys::subscription_cancel(q, id).unwrap();
    sys::channel_write(b, b"y", &[]).unwrap(); empty(q);
    let queued = sys::object_subscribe(q, a, e::READABLE, 9).unwrap();
    sys::subscription_cancel(q, queued).unwrap(); empty(q);
    assert_eq!(sys::subscription_cancel(q, queued), Err(Error::NotFound));
    sys::channel_read(a, &mut [0], &mut [], 0).unwrap();
    t.check("event timeout preserves subscription; cancel removes packets", true, "");

    let sibling = sys::dup(a, rights::ALL).unwrap();
    let closed = sys::object_subscribe(q, a, e::READABLE, 10).unwrap();
    sys::close(a).unwrap();
    let (replacement, other) = sys::channel_create().unwrap();
    assert_eq!(replacement, a);
    sys::channel_write(other, b"new", &[]).unwrap(); sys::channel_write(b, b"old", &[]).unwrap(); empty(q);
    assert_eq!(sys::subscription_rearm(q, closed), Err(Error::PeerClosed));
    sys::subscription_cancel(q, closed).unwrap();
    let queued = sys::object_subscribe(q, sibling, e::READABLE, 11).unwrap();
    sys::close(sibling).unwrap(); take(q, queued, 11, e::READABLE);
    assert_eq!(sys::subscription_rearm(q, queued), Err(Error::PeerClosed));
    sys::subscription_cancel(q, queued).unwrap();
    let id = sys::object_subscribe(q, b, e::PEER_CLOSED, 12).unwrap();
    take(q, id, 12, e::PEER_CLOSED); sys::subscription_cancel(q, id).unwrap(); close(&[b, replacement, other]);
    t.check("event source identity, close and slot reuse", true, "");

    let n = sys::notify_create().unwrap();
    let wait_only = sys::dup(n, rights::WAIT).unwrap();
    let signal_only = sys::dup(n, rights::SIGNAL).unwrap();
    let reader = sys::dup(q, rights::READ | rights::WAIT).unwrap(); let writer = sys::dup(q, rights::WRITE).unwrap();
    assert_eq!(sys::object_subscribe(q, signal_only, e::PENDING, 0), Err(Error::AccessDenied));
    assert_eq!(sys::object_subscribe(reader, n, e::PENDING, 0), Err(Error::AccessDenied));
    assert_eq!(sys::event_queue_wait(writer, 0), Err(Error::AccessDenied));
    assert_eq!(sys::notify_signal(wait_only, 1), Err(Error::AccessDenied));
    let id = sys::object_subscribe(writer, wait_only, e::PENDING, 13).unwrap();
    sys::notify_signal(n, 0x40).unwrap(); take(reader, id, 13, e::PENDING);
    assert_eq!(sys::notify_wait(n, 0).unwrap(), 0x40);
    sys::subscription_rearm(q, id).unwrap(); empty(q); sys::notify_signal(n, 1).unwrap(); take(q, id, 13, e::PENDING);
    sys::subscription_cancel(q, id).unwrap(); close(&[wait_only, signal_only, reader, writer]);
    let pending = sys::object_subscribe(q, n, e::PENDING, 99).unwrap();
    let status: i64;
    // SAFETY: deliberate invalid output tests transactional dequeue.
    unsafe {
        core::arch::asm!("syscall", inlateout("rax") dross_abi::sys::EVENT_QUEUE_WAIT as i64 => status,
            in("rdi") q.0 as u64, in("rsi") 1u64, inlateout("rdx") 0u64 => _,
            in("r10") 0u64, in("r8") 0u64, in("r9") 0u64, out("rcx") _, out("r11") _, options(nostack));
    }
    assert_eq!(status, Error::InvalidArgs as i64);
    take(q, pending, 99, e::PENDING); sys::subscription_cancel(q, pending).unwrap();
    t.check("event WAIT authority grants no operation rights", true, "");

    let limited = sys::event_queue_create(1).unwrap();
    assert_eq!(sys::event_queue_create(0), Err(Error::InvalidArgs));
    assert_eq!(sys::event_queue_create(EVENT_LIMIT + 1), Err(Error::InvalidArgs));
    let id = sys::object_subscribe(limited, n, e::PENDING, 14).unwrap();
    assert_eq!(sys::object_subscribe(limited, n, e::PENDING, 15), Err(Error::NoResources));
    take(limited, id, 14, e::PENDING);
    assert_eq!(sys::object_subscribe(limited, n, e::PENDING, 15), Err(Error::NoResources));
    sys::subscription_cancel(limited, id).unwrap();
    let new = sys::object_subscribe(limited, n, e::PENDING, 15).unwrap(); assert_ne!(new, id);
    assert_eq!(sys::subscription_rearm(limited, id), Err(Error::NotFound)); sys::subscription_cancel(limited, new).unwrap();
    assert_eq!(sys::object_subscribe(q, n, e::READABLE, 0), Err(Error::InvalidArgs));
    let m = sys::mem_create(4096, 0).unwrap();
    assert_eq!(sys::object_subscribe(q, m, e::PENDING, 0), Err(Error::WrongType)); close(&[m, limited, n]);
    t.check("event bounded resources and stable subscription IDs", true, "");
    let n = sys::notify_create().unwrap();
    let full = sys::event_queue_create(EVENT_LIMIT).unwrap();
    let extra = sys::event_queue_create(1).unwrap();
    let mut ids = alloc::vec::Vec::new();
    for key in 0..EVENT_LIMIT { ids.push(sys::object_subscribe(full, n, e::PENDING, key as u64).unwrap()); }
    assert_eq!(sys::object_subscribe(extra, n, e::PENDING, 0), Err(Error::NoResources));
    sys::notify_signal(n, 1).unwrap();
    for (key, &id) in ids.iter().enumerate() { take(full, id, key as u64, e::PENDING); }
    empty(full);
    // Destroying a queue releases registration capacity even with no explicit cancel.
    sys::close(full).unwrap();
    let id = sys::object_subscribe(extra, n, e::PENDING, 0).unwrap(); take(extra, id, 0, e::PENDING);
    sys::subscription_cancel(extra, id).unwrap(); close(&[extra, n]);
    t.check("event saturation delivers every reserved packet", true, "");

    let (a, b) = sys::channel_create().unwrap(); let (sender, receiver) = sys::channel_create().unwrap();
    let id = sys::object_subscribe(q, a, e::READABLE, 19).unwrap();
    sys::channel_write(sender, &[], &[a]).unwrap();
    sys::channel_write(b, b"transfer", &[]).unwrap(); take(q, id, 19, e::READABLE);
    let mut received = [Handle(0)];
    sys::channel_read(receiver, &mut [], &mut received, 0).unwrap();
    sys::subscription_rearm(q, id).unwrap(); take(q, id, 19, e::READABLE);
    sys::close(received[0]).unwrap();
    assert_eq!(sys::subscription_rearm(q, id), Err(Error::PeerClosed));
    sys::subscription_cancel(q, id).unwrap(); close(&[b, sender, receiver]);
    t.check("event subscriptions follow transferred handle identity", true, "");


    let n = sys::notify_create().unwrap(); let (control, worker_end) = sys::channel_create().unwrap();
    let worker = thread::spawn((sys::cpu_count() - 1) as u64, move || {
        for _ in 0..384 {
            sys::channel_read(worker_end, &mut [0], &mut [], FOREVER).unwrap();
            sys::notify_signal(n, 1).unwrap(); sys::channel_write(worker_end, b"a", &[]).unwrap();
        }
        sys::close(worker_end).unwrap();
    }).unwrap();
    let tq = sys::event_queue_create(1).unwrap(); let tid = sys::object_subscribe(tq, worker, e::EXITED, 16).unwrap();
    for i in 0..384 {
        let _ = sys::notify_wait(n, 0);
        if i % 2 == 0 { sys::channel_write(control, b"r", &[]).unwrap(); }
        let id = sys::object_subscribe(q, n, e::PENDING, i).unwrap();
        if i % 2 != 0 { sys::channel_write(control, b"r", &[]).unwrap(); }
        if i < 256 { take(q, id, i, e::PENDING); }
        sys::subscription_cancel(q, id).unwrap();
        sys::channel_read(control, &mut [0], &mut [], sys::after_ms(5000)).unwrap(); empty(q);
    }
    take(tq, tid, 16, e::EXITED); sys::subscription_cancel(tq, tid).unwrap(); close(&[worker, tq, control, n]);
    t.check("event register/cancel races and thread exit", true, "");

    let image = boot.handle("module/selftest").unwrap(); let (control, child_end) = sys::channel_create().unwrap();
    let child = bootstrap::spawn(image, "event-child", Builder::default().value("mode", "event-child").handle("control", child_end), u64::MAX).unwrap();
    let id = sys::object_subscribe(q, child, e::EXITED, 17).unwrap(); sys::channel_write(control, b"x", &[]).unwrap();
    take(q, id, 17, e::EXITED); assert_eq!(sys::process_wait(child, 0), Ok(42));
    sys::subscription_rearm(q, id).unwrap(); take(q, id, 17, e::EXITED);
    sys::subscription_cancel(q, id).unwrap(); close(&[child, control]);
    t.check("event process exit keeps status in process API", true, "");

    let (control, worker_end) = sys::channel_create().unwrap();
    let worker = thread::spawn((sys::cpu_count() - 1) as u64, move || {
        for _ in 0..256 {
            let mut command = [0u8; 8];
            sys::channel_read(worker_end, &mut command, &mut [], FOREVER).unwrap();
            let source = Handle(u32::from_le_bytes(command[4..].try_into().unwrap()));
            if command[0] == 0 { sys::close(source).unwrap(); } else { sys::notify_signal(source, 1).unwrap(); }
            sys::channel_write(worker_end, b"a", &[]).unwrap();
        }
        sys::close(worker_end).unwrap();
    }).unwrap();
    let completion = sys::event_queue_create(1).unwrap();
    let done = sys::object_subscribe(completion, worker, e::EXITED, 20).unwrap();
    for iteration in 0..256 {
        let n = sys::notify_create().unwrap();
        let keeper = sys::dup(n, rights::ALL).unwrap();
        let mut command = [0u8; 8]; command[4..].copy_from_slice(&n.0.to_le_bytes());
        if iteration < 128 {
            sys::notify_signal(n, 1).unwrap();
            let id = sys::object_subscribe(q, n, e::PENDING, 21).unwrap(); take(q, id, 21, e::PENDING);
            sys::channel_write(control, &command, &[]).unwrap();
            match sys::subscription_rearm(q, id) {
                Ok(()) => take(q, id, 21, e::PENDING),
                Err(Error::PeerClosed) => empty(q),
                result => panic!("rearm/close: {result:?}"),
            }
            sys::channel_read(control, &mut [0], &mut [], sys::after_ms(5000)).unwrap();
            sys::subscription_cancel(q, id).unwrap();
        } else {
            let temporary = sys::event_queue_create(1).unwrap();
            sys::object_subscribe(temporary, n, e::PENDING, 22).unwrap();
            command[0] = 1;
            sys::channel_write(control, &command, &[]).unwrap();
            sys::close(temporary).unwrap();
            sys::channel_read(control, &mut [0], &mut [], sys::after_ms(5000)).unwrap();
            sys::close(n).unwrap();
        }
        sys::close(keeper).unwrap();
    }
    take(completion, done, 20, e::EXITED); sys::subscription_cancel(completion, done).unwrap();
    close(&[completion, worker, control]);
    t.check("event source-close/rearm and queue-close/signal races", true, "");

    let (a, b) = sys::channel_create().unwrap(); let capacity = sys::event_queue_create(1).unwrap();
    let id = sys::object_subscribe(capacity, a, e::WRITABLE, 18).unwrap(); take(capacity, id, 18, e::WRITABLE);
    for _ in 0..dross_abi::CHANNEL_DEPTH { sys::channel_write(a, b"x", &[]).unwrap(); }
    sys::subscription_rearm(capacity, id).unwrap(); empty(capacity);
    sys::channel_read(b, &mut [0], &mut [], 0).unwrap(); take(capacity, id, 18, e::WRITABLE);
    sys::subscription_cancel(capacity, id).unwrap(); close(&[capacity, a, b, q]);
    t.check("event writable readiness follows channel capacity", true, "");
}
