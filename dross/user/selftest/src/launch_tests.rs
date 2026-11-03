//! Native launch ownership, including recovery after bootstrap was queued.
use super::Tally;
use dross_abi::{Error, rights};
use dross_rt::deadline::{Deadline, Duration};
use dross_rt::launch::{self, LaunchStage, ReturnedBootstrap, SpawnOptions};
use dross_rt::owned::{Memory, Message, OwnedHandle};
use dross_rt::{bootstrap::Bootstrap, sys};

pub fn run(t: &mut Tally, bootstrap: &Bootstrap) {
    let result = checks(bootstrap);
    t.check(
        "owned process launch and bootstrap recovery",
        result.is_ok(),
        "",
    );
}

fn checks(bootstrap: &Bootstrap) -> Result<(), Error> {
    // A private zeroed page is not an ELF. Failure happens after enqueue, so the
    // returned child endpoint must retain the capabilities queued for bootstrap.
    let bad_image = Memory::create(4096, 0)?;
    let attachment = Memory::create(4096, 0)?.into_untyped();
    let old_number = attachment.as_handle().as_raw();
    let mut message = Message::try_from_bytes(b"recover")?;
    message.try_push_handle(attachment).map_err(|e| e.error)?;
    let failure = launch::spawn(
        bad_image.as_handle(),
        "bad-image",
        message,
        SpawnOptions::default(),
    )
    .expect_err("non-ELF launch must fail");
    assert_eq!(failure.stage, LaunchStage::Spawn);
    assert_eq!(failure.error, Error::BadElf);
    assert_eq!(sys::mem_size(old_number), Err(Error::BadHandle));
    let ReturnedBootstrap::Queued(channel) = failure.bootstrap else {
        return Err(Error::InvalidArgs);
    };
    let mut recovered = channel.receive(Deadline::IMMEDIATE)?;
    assert_eq!(recovered.bytes(), b"recover");
    let attachment = recovered.take_handle(0).ok_or(Error::BadHandle)?;
    assert_eq!(sys::mem_size(attachment.as_handle().as_raw()), Ok(4096));
    drop(attachment);
    assert_eq!(
        channel.receive(Deadline::IMMEDIATE).err(),
        Some(Error::PeerClosed)
    );

    let attachment = Memory::create(4096, 0)?.into_untyped();
    let number = attachment.as_handle().as_raw();
    let mut message = Message::new();
    message.try_push_handle(attachment).map_err(|e| e.error)?;
    let failure = launch::spawn(bad_image.as_handle(), "", message, SpawnOptions::default())
        .expect_err("empty name must fail before enqueue");
    assert_eq!(failure.stage, LaunchStage::Validate);
    assert!(matches!(failure.bootstrap, ReturnedBootstrap::Unsent(_)));
    assert_eq!(sys::mem_size(number), Ok(4096));
    drop(failure);
    assert_eq!(sys::mem_size(number), Err(Error::BadHandle));

    let image = sys::dup(
        bootstrap.handle("module/selftest").ok_or(Error::NotFound)?,
        rights::READ,
    )?;
    // SAFETY: HANDLE_DUP returned a fresh slot, initially kept untyped. MEM_SIZE
    // in try_into_memory verifies the object before exposing a typed image.
    let image = unsafe { OwnedHandle::from_raw(image) }
        .try_into_memory()
        .map_err(|e| e.error)?;
    let message = Message::try_from_bytes(b"mode exit42\n")?;
    let child = launch::spawn(
        image.as_handle(),
        "owned-launch-child",
        message,
        SpawnOptions::default(),
    )
    .map_err(|e| e.error)?;
    assert_eq!(
        child.wait_until(Deadline::from_now(Duration::from_secs(5)))?,
        42
    );
    assert!(image.size()? > 4096);
    Ok(())
}
