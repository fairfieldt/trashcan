use super::*;
use dross_abi::event as e;
fn expiry(queue: Handle, id: u64) {
    let event = sys::event_queue_wait(queue, FOREVER).unwrap();
    assert_eq!(event.subscription, id); assert_ne!(event.observed & e::EXPIRED, 0);
}
fn empty(queue: Handle) { assert_eq!(sys::event_queue_wait(queue, 0), Err(Error::TimedOut)); }
pub fn run(t: &mut Tally) {
    let timer = sys::timer_create().unwrap();
    let queue = sys::event_queue_create(1).unwrap();
    let id = sys::object_subscribe(queue, timer, e::EXPIRED, 1).unwrap();
    empty(queue);
    let deadline = sys::after_ms(5);
    sys::timer_arm(timer, deadline).unwrap();
    // No wait deadline: only the timer object's APIC interrupt can produce this event.
    expiry(queue, id); assert!(sys::clock() >= deadline);
    sys::subscription_rearm(queue, id).unwrap(); expiry(queue, id);
    sys::timer_cancel(timer).unwrap(); sys::subscription_rearm(queue, id).unwrap(); empty(queue);
    t.check("timer IRQ expiry, absolute deadline and expired latch", true, "");

    sys::timer_arm(timer, sys::after_ms(5)).unwrap();
    let deadline = sys::after_ms(30);
    sys::timer_arm(timer, deadline).unwrap();
    expiry(queue, id); assert!(sys::clock() >= deadline);
    sys::timer_cancel(timer).unwrap(); sys::subscription_rearm(queue, id).unwrap();
    sys::timer_arm(timer, sys::after_ms(5)).unwrap(); sys::timer_cancel(timer).unwrap();
    assert_eq!(sys::event_queue_wait(queue, sys::after_ms(10)), Err(Error::TimedOut));
    sys::timer_arm(timer, 0).unwrap(); sys::timer_cancel(timer).unwrap();
    expiry(queue, id); // Cancel does not retract a historical queued observation.
    sys::subscription_rearm(queue, id).unwrap(); sys::timer_arm(timer, FOREVER).unwrap(); empty(queue);
    t.check("timer reset and cancel suppress stale expiry generations", true, "");

    let observer = sys::dup(timer, rights::WAIT).unwrap();
    let controller = sys::dup(timer, rights::WRITE).unwrap();
    assert_eq!(sys::timer_arm(observer, 0), Err(Error::AccessDenied));
    assert_eq!(sys::timer_cancel(observer), Err(Error::AccessDenied));
    assert_eq!(sys::object_subscribe(queue, controller, e::EXPIRED, 0), Err(Error::AccessDenied));
    sys::close(observer).unwrap(); sys::close(controller).unwrap();
    sys::subscription_cancel(queue, id).unwrap();
    let id = sys::object_subscribe(queue, timer, e::EXPIRED, 2).unwrap();
    let (control, worker_end) = sys::channel_create().unwrap();
    let worker = thread::spawn((sys::cpu_count() - 1) as u64, move || {
        sys::timer_arm(timer, sys::after_ms(5)).unwrap();
        sys::channel_write(worker_end, b"armed", &[]).unwrap();
        for _ in 0..64 {
            sys::channel_read(worker_end, &mut [], &mut [], FOREVER).unwrap();
            sys::timer_arm(timer, 0).unwrap();
            sys::timer_cancel(timer).unwrap();
            sys::channel_write(worker_end, &[], &[]).unwrap();
        }
        sys::close(worker_end).unwrap();
    }).unwrap();
    sys::channel_read(control, &mut [0; 5], &mut [], sys::after_ms(5000)).unwrap();
    expiry(queue, id);
    sys::subscription_cancel(queue, id).unwrap();
    for _ in 0..64 {
        sys::timer_cancel(timer).unwrap();
        let id = sys::object_subscribe(queue, timer, e::EXPIRED, 4).unwrap();
        sys::timer_arm(timer, sys::after_ms(1)).unwrap();
        sys::channel_write(control, &[], &[]).unwrap();
        sys::channel_read(control, &mut [], &mut [], sys::after_ms(5000)).unwrap();
        sys::timer_cancel(timer).unwrap(); sys::subscription_cancel(queue, id).unwrap();
        let id = sys::object_subscribe(queue, timer, e::EXPIRED, 5).unwrap(); empty(queue);
        let deadline = sys::after_ms(2); sys::timer_arm(timer, deadline).unwrap(); expiry(queue, id);
        assert!(sys::clock() >= deadline); sys::subscription_cancel(queue, id).unwrap();
    }
    let done = sys::object_subscribe(queue, worker, e::EXITED, 6).unwrap();
    let event = sys::event_queue_wait(queue, sys::after_ms(5000)).unwrap(); assert_eq!(event.subscription, done);
    sys::subscription_cancel(queue, done).unwrap();
    sys::close(worker).unwrap(); sys::close(control).unwrap();
    t.check("timer cross-CPU reset/cancel races and WAIT authority", true, "");

    sys::timer_cancel(timer).unwrap();
    let id = sys::object_subscribe(queue, timer, e::EXPIRED, 7).unwrap();
    sys::timer_arm(timer, sys::after_ms(5)).unwrap(); sys::close(timer).unwrap();
    assert_eq!(sys::subscription_rearm(queue, id), Err(Error::PeerClosed));
    assert_eq!(sys::event_queue_wait(queue, sys::after_ms(10)), Err(Error::TimedOut));
    sys::subscription_cancel(queue, id).unwrap(); sys::close(queue).unwrap();
    let mut timers = alloc::vec::Vec::new();
    for _ in 0..dross_abi::TIMER_LIMIT { timers.push(sys::timer_create().unwrap()); }
    assert_eq!(sys::timer_create(), Err(Error::NoResources));
    for timer in timers { sys::close(timer).unwrap(); }
    sys::close(sys::timer_create().unwrap()).unwrap();
    t.check("timer close cancels deadlines and releases bounded slots", true, "");
}
