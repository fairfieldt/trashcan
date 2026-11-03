//! Isolated audio acceptance supervisor. Never changes the ordinary boot policy.
#![no_std]
#![no_main]
extern crate alloc;
use dross_abi::rights;
use dross_rt::{
    Error, Handle,
    bootstrap::{self, Bootstrap, Builder},
    println, sys,
};
dross_rt::main!(main);
fn main(boot: Handle) -> i32 {
    let b = Bootstrap::read(boot).expect("audio test bootstrap");
    let result = run(&b);
    println!(
        "audio test: {}",
        if result.is_ok() { "ok" } else { "FAILED" }
    );
    if let Err(e) = result {
        println!("audio test: {e:?}");
    }
    sys::system_exit(
        b.handle("resource/system").unwrap(),
        if result.is_ok() { 0 } else { 1 },
    );
}
fn run(b: &Bootstrap) -> sys::Result<()> {
    // A lost pcid response must poison all clones of its untagged connection.
    let (probe, peer) = sys::channel_create()?;
    let bounded = dross_rt::pci::Pci::with_timeout(probe, 5);
    let clone = bounded.clone();
    if !matches!(bounded.list(), Err(Error::TimedOut))
        || !matches!(clone.list(), Err(Error::PeerClosed))
    {
        return Err(Error::InvalidArgs);
    }
    let mut request = [0u8; 32];
    let mut handles = [Handle(0); 1];
    if sys::channel_read(peer, &mut request, &mut handles, sys::after_ms(10))? != (32, 0) {
        return Err(Error::InvalidArgs);
    }
    if sys::channel_read(peer, &mut request, &mut handles, sys::after_ms(10))
        != Err(Error::PeerClosed)
    {
        return Err(Error::InvalidArgs);
    }
    sys::close(peer)?;
    println!("audio test: PCI timeout closes and poisons shared endpoint");
    let ecam = b.value_u64("ecam").ok_or(Error::NotFound)?;
    let buses = b.value("ecam-buses").ok_or(Error::NotFound)?;
    let last: u64 = buses
        .split(' ')
        .nth(1)
        .and_then(|v| v.parse().ok())
        .ok_or(Error::InvalidArgs)?;
    let (pcid_end, audio_pci) = sys::channel_create()?;
    let pb = Builder::default()
        .handle(
            "resource/ecam",
            sys::resource_sub(
                b.handle("resource/physical").unwrap(),
                ecam,
                (last + 1) << 20,
            )?,
        )
        .handle(
            "resource/physical",
            sys::dup(b.handle("resource/physical").unwrap(), rights::ALL)?,
        )
        .handle(
            "resource/irq",
            sys::dup(b.handle("resource/irq").unwrap(), rights::ALL)?,
        )
        .handle(
            "resource/iommu",
            sys::dup(b.handle("resource/iommu").unwrap(), rights::ALL)?,
        )
        .value("ecam", &alloc::format!("{ecam:#x}"))
        .value("ecam-buses", buses)
        .handle("client/audio", pcid_end);
    bootstrap::spawn(b.handle("module/pcid").unwrap(), "pcid", pb, u64::MAX)?;
    let (server, client) = sys::channel_create()?;
    let audio = bootstrap::spawn(
        b.handle("module/audiod").unwrap(),
        "audiod",
        Builder::default()
            .handle("pci", audio_pci)
            .handle("service", server),
        u64::MAX,
    )?;
    let test = bootstrap::spawn(
        b.handle("module/audiotest").unwrap(),
        "audiotest",
        Builder::default().handle("audio", client),
        u64::MAX,
    )?;
    if sys::process_wait(test, sys::after_ms(15000))? != 0 {
        return Err(Error::InvalidArgs);
    }
    if sys::process_wait(audio, sys::after_ms(3000))? != 0 {
        return Err(Error::InvalidArgs);
    }
    // RELEASE has its own reply path and must retire a timed-out endpoint too.
    let (probe, peer) = sys::channel_create()?;
    let bounded = dross_rt::pci::Pci::with_timeout(probe, 5);
    let clone = bounded.clone();
    if bounded.release(0, audio) != Err(Error::TimedOut)
        || !matches!(clone.list(), Err(Error::PeerClosed))
    {
        return Err(Error::InvalidArgs);
    }
    if sys::channel_read(peer, &mut request, &mut handles, sys::after_ms(10))? != (32, 1) {
        return Err(Error::InvalidArgs);
    }
    sys::close(handles[0])?;
    if sys::channel_read(peer, &mut request, &mut handles, sys::after_ms(10))
        != Err(Error::PeerClosed)
    {
        return Err(Error::InvalidArgs);
    }
    sys::close(peer)?;
    println!("audio test: RELEASE timeout closes and poisons shared endpoint");
    println!("audio test: client and driver exited successfully");
    Ok(())
}
