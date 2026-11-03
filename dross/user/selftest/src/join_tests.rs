use super::*;
fn used() -> u64 { sys::account_info(Handle(0)).unwrap()[0] }
pub fn run(t: &mut Tally) {
    let before = used();
    for _ in 0..32 {
        let (control, worker_end) = sys::channel_create().unwrap();
        let mut worker = thread::spawn_joinable((sys::cpu_count() - 1) as u64, move || {
            sys::channel_read(worker_end, &mut [], &mut [], FOREVER).unwrap();
            sys::close(worker_end).unwrap();
        }).unwrap();
        assert!(used() >= before + (256 << 10));
        assert_eq!(worker.wait_until(0), Err(Error::TimedOut));
        sys::channel_write(control, &[], &[]).unwrap();
        worker.wait_until(sys::after_ms(5000)).unwrap();
        sys::close(control).unwrap();
        assert_eq!(used(), before);
    }
    t.check("native join retries deadline and reaps user stacks", true, "");

    let (control, worker_end) = sys::channel_create().unwrap();
    let worker = thread::spawn_joinable((sys::cpu_count() - 1) as u64, move || {
        sys::channel_read(worker_end, &mut [], &mut [], FOREVER).unwrap(); sys::close(worker_end).unwrap();
    }).unwrap();
    let observer = sys::dup(worker.as_handle(), rights::WAIT).unwrap();
    drop(worker); // Detach; retained observation must not retain its user stack after exit.
    sys::channel_write(control, &[], &[]).unwrap();
    sys::thread_wait(observer, sys::after_ms(5000)).unwrap();
    sys::close(control).unwrap();
    assert_eq!(used(), before);
    let no_wait = sys::dup(observer, 0).err(); // WAIT-only cannot duplicate.
    assert_eq!(no_wait, Some(Error::AccessDenied));
    sys::close(observer).unwrap();
    t.check("detached native thread stack is reclaimed with retained status", true, "");

    let (control, worker_end) = sys::channel_create().unwrap();
    let worker = thread::spawn_joinable((sys::cpu_count() - 1) as u64, move || {
        let mut handle = [0u8; 4];
        sys::channel_read(worker_end, &mut handle, &mut [], FOREVER).unwrap();
        let handle = Handle(u32::from_le_bytes(handle));
        assert_eq!(sys::thread_wait(handle, 0), Err(Error::Busy));
        sys::close(worker_end).unwrap();
    }).unwrap();
    sys::channel_write(control, &worker.as_handle().0.to_le_bytes(), &[]).unwrap();
    worker.join().unwrap(); sys::close(control).unwrap();
    assert_eq!(used(), before);
    t.check("native self-join is rejected", true, "");
}
