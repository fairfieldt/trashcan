use super::*;
fn used() -> u64 { sys::account_info(Handle(0)).unwrap()[0] }
pub fn run(t: &mut Tally, boot: &Bootstrap) {
    let system = boot.handle("resource/system").unwrap();
    let before = used();
    let (a, b) = sys::channel_create().unwrap();
    let memory = sys::mem_create(64 << 10, 0).unwrap();
    sys::channel_write(a, b"cycle", &[b, memory]).unwrap();
    assert!(used() >= before + (64 << 10) + 1024);
    sys::channel_collect(system).unwrap();
    assert_eq!(used(), before + 1024);
    assert_eq!(sys::channel_write(a, &[], &[]), Err(Error::PeerClosed));
    sys::close(a).unwrap(); assert_eq!(used(), before);
    t.check("unreachable self-queued channel releases backing with live peer", true, "");

    let (a, b) = sys::channel_create().unwrap();
    let (c, d) = sys::channel_create().unwrap();
    let memory = sys::mem_create(64 << 10, 0).unwrap();
    sys::channel_write(c, &[], &[b, memory]).unwrap();
    sys::channel_write(a, &[], &[d]).unwrap();
    sys::channel_collect(system).unwrap();
    for sender in [a, c] { assert_eq!(sys::channel_write(sender, &[], &[]), Err(Error::PeerClosed)); sys::close(sender).unwrap(); }
    assert_eq!(used(), before);
    t.check("mutual queued capability cycles reclaim memory", true, "");

    let (a, b) = sys::channel_create().unwrap();
    let receiver = sys::dup(b, rights::ALL).unwrap();
    sys::channel_write(a, b"live", &[b]).unwrap();
    sys::channel_collect(system).unwrap();
    let mut bytes = [0; 4]; let mut handles = [Handle(0)];
    assert_eq!(sys::channel_read(receiver, &mut bytes, &mut handles, 0), Ok((4, 1)));
    assert_eq!(&bytes, b"live");
    // Sending the writer itself is legitimate too: the peer can receive it.
    sys::channel_write(a, &[], &[a]).unwrap();
    sys::channel_collect(system).unwrap();
    let mut writer = [Handle(0)];
    sys::channel_read(receiver, &mut [], &mut writer, 0).unwrap();
    for handle in [receiver, handles[0], writer[0]] { sys::close(handle).unwrap(); }
    assert_eq!(used(), before);
    t.check("collector preserves reachable cycles and ordinary endpoint transfer", true, "");

    let reduced = sys::dup(system, rights::READ).unwrap();
    assert_eq!(sys::channel_collect(reduced), Err(Error::AccessDenied)); sys::close(reduced).unwrap();
    let worker = thread::spawn_joinable((sys::cpu_count() - 1) as u64, move || {
        for _ in 0..64 {
            let (a, b) = sys::channel_create().unwrap();
            sys::channel_write(a, &[], &[b]).unwrap();
            sys::close(a).unwrap();
        }
    }).unwrap();
    for _ in 0..64 { sys::channel_collect(system).unwrap(); sys::yield_now(); }
    worker.join().unwrap(); sys::channel_collect(system).unwrap();
    assert_eq!(used(), before);
    t.check("concurrent channel close and collection preserve authority and reclaim", true, "");
}
