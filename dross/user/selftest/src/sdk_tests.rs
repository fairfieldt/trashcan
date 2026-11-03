use super::*;
use alloc::vec::Vec;
use dross_abi::{CHANNEL_DEPTH, MSG_BYTES, MSG_HANDLES, event};
use dross_rt::deadline::{Deadline, Duration};
use dross_rt::owned::{self, Channel, EventQueue, Memory, Message, Timer};

pub fn run(t: &mut Tally) {
    let (a, b) = Channel::pair().unwrap();
    let duplicate = a
        .as_handle()
        .try_duplicate(rights::READ | rights::WRITE | rights::WAIT)
        .unwrap();
    assert_eq!(
        duplicate.try_duplicate(rights::ALL).unwrap_err(),
        Error::AccessDenied
    );
    let original = a.as_handle().as_raw();
    drop(a);
    assert_eq!(
        sys::channel_write(original, &[], &[]),
        Err(Error::BadHandle)
    );
    duplicate
        .try_send(Message::try_from_bytes(b"owned").unwrap())
        .unwrap();
    assert_eq!(b.receive(Deadline::IMMEDIATE).unwrap().bytes(), b"owned");
    let raw = duplicate.into_raw();
    // SAFETY: into_raw transferred exactly one valid channel owner to this scope.
    let duplicate = unsafe { Channel::from_raw(raw) };
    drop(duplicate);
    assert_eq!(
        b.receive(Deadline::IMMEDIATE).unwrap_err(),
        Error::PeerClosed
    );
    t.check(
        "SDK owned drop, borrowed duplication and raw roundtrip",
        true,
        "",
    );

    let (sender, peer) = Channel::pair().unwrap();
    let memory = Memory::create(4096, 0).unwrap();
    let original = memory.as_handle().as_raw();
    let mut message = Message::try_from_bytes(b"retry").unwrap();
    message.try_push_handle(memory.into_untyped()).unwrap();
    drop(peer);
    let error = sender.try_send(message).unwrap_err();
    assert_eq!(error.error, Error::PeerClosed);
    assert_eq!(error.message.bytes(), b"retry");
    assert_eq!(sys::mem_size(original), Ok(4096));
    let (sender, peer) = Channel::pair().unwrap();
    sender.try_send(error.message).unwrap();
    assert_eq!(sys::mem_size(original), Err(Error::BadHandle));
    let mut received = peer.receive(Deadline::IMMEDIATE).unwrap();
    let memory = received.take_handle(0).unwrap().try_into_memory().unwrap();
    assert!(received.take_handle(0).is_none());
    assert_eq!(memory.size(), Ok(4096));
    assert_eq!(received.handles_len(), 1);
    assert_eq!(received.remaining_handles(), 0);
    assert_eq!(
        sender.try_send(received).unwrap_err().error,
        Error::InvalidArgs
    );
    t.check(
        "SDK failed send returns ownership and successful send transfers it",
        true,
        "",
    );

    let (attachment, observer) = Channel::pair().unwrap();
    let mut malformed = Message::try_from_bytes(b"not a valid service header").unwrap();
    malformed
        .try_push_handle(attachment.into_untyped())
        .unwrap();
    sender.try_send(malformed).unwrap();
    let malformed = peer.receive(Deadline::IMMEDIATE).unwrap();
    assert_eq!(malformed.handle_count(), 1);
    // Rejected input still owns and closes its unused attachments.
    drop(malformed);
    assert_eq!(
        observer.receive(Deadline::IMMEDIATE).unwrap_err(),
        Error::PeerClosed
    );
    let (attachment, observer) = Channel::pair().unwrap();
    let mut message = Message::new();
    message.try_push_handle(attachment.into_untyped()).unwrap();
    sender.try_send(message).unwrap();
    let mut message = peer.receive(Deadline::IMMEDIATE).unwrap();
    let wrong = message
        .take_handle(0)
        .unwrap()
        .try_into_memory()
        .unwrap_err();
    assert_eq!(wrong.error, Error::WrongType);
    assert_eq!(
        observer.receive(Deadline::IMMEDIATE).unwrap_err(),
        Error::TimedOut
    );
    drop(wrong);
    assert_eq!(
        observer.receive(Deadline::IMMEDIATE).unwrap_err(),
        Error::PeerClosed
    );
    t.check(
        "SDK rejected input and failed type checks own every attachment",
        true,
        "",
    );

    let (sender, peer) = Channel::pair().unwrap();
    for _ in 0..CHANNEL_DEPTH {
        sender.try_send(Message::new()).unwrap();
    }
    let mut pending = Message::try_from_bytes(b"full").unwrap();
    pending.try_push_handle(memory.into_untyped()).unwrap();
    let pending = sender.try_send(pending).unwrap_err();
    assert_eq!(pending.error, Error::ShouldWait);
    assert_eq!(pending.message.remaining_handles(), 1);
    peer.receive(Deadline::IMMEDIATE).unwrap();
    sender.try_send(pending.message).unwrap();
    for _ in 1..CHANNEL_DEPTH {
        peer.receive(Deadline::IMMEDIATE).unwrap();
    }
    let mut received = peer.receive(Deadline::IMMEDIATE).unwrap();
    assert_eq!(received.bytes(), b"full");
    assert_eq!(
        received
            .take_handle(0)
            .unwrap()
            .try_into_memory()
            .unwrap()
            .size(),
        Ok(4096)
    );
    t.check(
        "SDK channel backpressure preserves the retry message",
        true,
        "",
    );

    let memory = Memory::create(4096, 0).unwrap();
    let mut message = Message::new();
    for _ in 0..MSG_HANDLES {
        message
            .try_push_handle(memory.try_duplicate(rights::ALL).unwrap().into_untyped())
            .unwrap();
    }
    let extra = memory.try_duplicate(rights::ALL).unwrap().into_untyped();
    let raw = extra.as_handle().as_raw();
    let overflow = message.try_push_handle(extra).unwrap_err();
    assert_eq!(overflow.error, Error::InvalidArgs);
    assert_eq!(sys::mem_size(raw), Ok(4096));
    drop(overflow);
    assert_eq!(sys::mem_size(raw), Err(Error::BadHandle));
    let mut bytes = Message::try_from_bytes(b"kept").unwrap();
    let maximum = [0u8; MSG_BYTES];
    assert_eq!(bytes.try_extend_bytes(&maximum), Err(Error::InvalidArgs));
    assert_eq!(bytes.bytes(), b"kept");
    drop(message);
    drop(memory);
    t.check(
        "SDK message bounds preserve bytes and rejected handle ownership",
        true,
        "",
    );

    let memory = Memory::create(4096, 0).unwrap();
    let mapping = memory.map(0, 4096, vm::READ | vm::WRITE).unwrap();
    let address = mapping.address();
    drop(memory);
    // SAFETY: fresh backing was never delegated or mapped elsewhere; this scope
    // owns the only mapping and accesses only its first byte.
    unsafe {
        mapping.as_mut_ptr().unwrap().write_volatile(0x5a);
        assert_eq!(mapping.as_ptr().read_volatile(), 0x5a);
    }
    mapping.unmap().unwrap();
    assert_eq!(sys::vm_memory(address), Err(Error::NotFound));
    let memory = Memory::create(4096, 0).unwrap();
    let mapping = memory.map(0, 4096, vm::READ).unwrap();
    assert!(mapping.as_mut_ptr().is_none());
    drop(mapping);
    drop(memory);
    t.check(
        "SDK mappings outlive handles and expose no safe shared references",
        true,
        "",
    );

    let (sender, peer) = Channel::pair().unwrap();
    let memory = Memory::create(4096, 0).unwrap();
    let mut queued = Message::try_from_bytes(b"still queued").unwrap();
    queued.try_push_handle(memory.into_untyped()).unwrap();
    sender.try_send(queued).unwrap();
    let mut storage: Vec<Message> = Vec::new();
    storage.try_reserve_exact(512).unwrap();
    let mut exhausted = false;
    for _ in 0..512 {
        match Message::try_from_bytes(&maximum) {
            Ok(message) => storage.push(message),
            Err(Error::NoMemory) => {
                exhausted = true;
                break;
            }
            Err(error) => panic!("SDK storage: {error:?}"),
        }
    }
    let result = peer.receive(Deadline::IMMEDIATE);
    // Return heap space before assertions or formatting can need it.
    drop(storage);
    assert!(exhausted);
    assert_eq!(result.unwrap_err(), Error::NoMemory);
    let mut queued = peer.receive(Deadline::IMMEDIATE).unwrap();
    assert_eq!(queued.bytes(), b"still queued");
    assert_eq!(
        queued
            .take_handle(0)
            .unwrap()
            .try_into_memory()
            .unwrap()
            .size(),
        Ok(4096)
    );
    t.check(
        "SDK receive allocation failure leaves queued attachments intact",
        true,
        "",
    );

    let queue = EventQueue::create(1).unwrap();
    let timer = Timer::create().unwrap();
    let subscription = queue
        .subscribe(timer.as_handle(), event::EXPIRED, 71)
        .unwrap();
    assert_eq!(
        queue.wait_until(Deadline::IMMEDIATE).unwrap_err(),
        Error::TimedOut
    );
    let deadline = Deadline::from_now(Duration::from_millis(20));
    timer.arm(deadline).unwrap();
    let event = queue.wait_until(Deadline::FOREVER).unwrap();
    assert_eq!(event.subscription, subscription.id());
    assert_eq!(event.key, 71);
    assert!(sys::clock() >= deadline.as_monotonic_ns());
    subscription.rearm().unwrap();
    assert_eq!(
        queue.wait_until(Deadline::IMMEDIATE).unwrap().subscription,
        subscription.id()
    );
    timer.cancel().unwrap();
    subscription.rearm().unwrap();
    assert_eq!(queue.wait_until(deadline).unwrap_err(), Error::TimedOut);
    assert_eq!(queue.wait_until(deadline).unwrap_err(), Error::TimedOut);
    drop(timer);
    assert_eq!(subscription.rearm(), Err(Error::PeerClosed));
    drop(subscription);
    assert_eq!(
        Deadline::after_at(u64::MAX - 1, Duration::from_millis(1)),
        Deadline::FOREVER
    );
    t.check(
        "SDK absolute deadlines, timer latch and registration cleanup",
        true,
        "",
    );

    let transport = Memory::create(4096, 0).unwrap().into_untyped();
    let error = owned::try_send(
        transport.as_handle(),
        Message::try_from_bytes(b"wrong").unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.error, Error::WrongType);
    assert_eq!(error.message.bytes(), b"wrong");
    t.check(
        "SDK untyped transports retain kernel type checking",
        true,
        "",
    );
}
