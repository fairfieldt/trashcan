use super::*;
const APP_LIMIT: u64 = 32 << 20;
fn empty_account(account: Handle) {
    let deadline = sys::after_ms(2000);
    while sys::account_info(account).unwrap()[0] != 0 && sys::clock() < deadline { sys::yield_now(); }
    assert_eq!(sys::account_info(account).unwrap()[0], 0);
}
pub fn probe(boot: &Bootstrap) -> i32 {
    assert_eq!(sys::account_info(Handle(0)).unwrap()[2], APP_LIMIT);
    assert_eq!(sys::mem_create(APP_LIMIT, 0), Err(Error::NoMemory));
    assert!(sys::account_create(Handle(0), u64::MAX).is_err());
    let image = boot.handle("module/selftest").unwrap();
    let nested = bootstrap::spawn(image, "budget-nested", Builder::default().value("mode", "exit42"), u64::MAX);
    match nested {
        Err(Error::NoMemory) => (),
        Ok(child) => {
            // Its startup heap would exceed the common ancestor. Failure ends
            // only that child; a new process must not acquire an unlimited budget.
            assert_eq!(sys::process_wait(child, sys::after_ms(5000)).unwrap(), -1);
            sys::close(child).unwrap();
        }
        result => panic!("budget spawn: {result:?}"),
    }
    assert_eq!(sys::mem_create(APP_LIMIT, 0), Err(Error::NoMemory));
    0
}
pub fn hold(boot: &Bootstrap) -> i32 {
    let control = boot.handle("control").unwrap();
    sys::channel_write(control, b"ready", &[]).unwrap();
    sys::channel_read(control, &mut [], &mut [], FOREVER).unwrap();
    let stats = sys::account_info(Handle(0)).unwrap();
    sys::channel_write(control, &stats[0].to_le_bytes(), &[]).unwrap();
    sys::channel_read(control, &mut [], &mut [], FOREVER).unwrap();
    0
}
pub fn run(t: &mut Tally, boot: &Bootstrap) {
    let system = boot.handle("resource/system").unwrap();
    let parent = sys::account_create(system, 16384).unwrap();
    let a = sys::account_child(parent, 12288).unwrap(); let b = sys::account_child(parent, 12288).unwrap();
    let baseline = sys::account_info(parent).unwrap()[0];
    let memory = sys::mem_create_in(a, 8192, 0).unwrap();
    assert_eq!(sys::mem_create_in(b, 8192, 0), Err(Error::NoMemory));
    assert_eq!(sys::account_info(b).unwrap()[0], 0);
    assert_eq!(sys::account_info(parent).unwrap()[0], baseline + 8192);
    sys::close(memory).unwrap(); assert_eq!(sys::account_info(parent).unwrap()[0], baseline);
    let reader = sys::dup(parent, rights::READ).unwrap();
    assert_eq!(sys::account_child(reader, 4096), Err(Error::AccessDenied));
    for h in [reader, a, b] { sys::close(h).unwrap(); }
    assert_eq!(sys::account_info(parent).unwrap()[0], 0); sys::close(parent).unwrap();
    t.check("ancestor budgets reserve atomically and roll back siblings", true, "");

    let image = boot.handle("module/selftest").unwrap();
    let account = sys::account_create(system, APP_LIMIT).unwrap();
    let child = bootstrap::spawn_in(account, image, "budget-probe", Builder::default().value("mode", "budget-probe")
        .handle("module/selftest", sys::dup(image, rights::ALL).unwrap()), u64::MAX).unwrap();
    assert_eq!(sys::process_wait(child, sys::after_ms(10000)), Ok(0));
    assert!(sys::account_info(account).unwrap()[1] <= APP_LIMIT);
    sys::close(child).unwrap(); empty_account(account);
    sys::close(account).unwrap();
    t.check("selected default budget cannot be escaped by native spawn", true, "");

    let account = sys::account_create(system, 64 << 20).unwrap();
    let mut children = alloc::vec::Vec::new();
    for _ in 0..2 {
        let (control, child_end) = sys::channel_create().unwrap();
        let child = bootstrap::spawn_in(account, image, "shared-budget", Builder::default().value("mode", "budget-hold").handle("control", child_end), u64::MAX).unwrap();
        sys::channel_read(control, &mut [0; 5], &mut [], sys::after_ms(5000)).unwrap();
        children.push((child, control));
    }
    let shared = sys::account_info(account).unwrap()[0]; assert!(shared >= 32 << 20);
    for &(_, control) in &children {
        sys::channel_write(control, &[], &[]).unwrap(); let mut bytes = [0u8; 8];
        sys::channel_read(control, &mut bytes, &mut [], sys::after_ms(5000)).unwrap();
        assert_eq!(u64::from_le_bytes(bytes), shared);
    }
    for (child, control) in children {
        sys::channel_write(control, &[], &[]).unwrap(); assert_eq!(sys::process_wait(child, sys::after_ms(5000)), Ok(0));
        sys::close(child).unwrap(); sys::close(control).unwrap();
    }
    empty_account(account); sys::close(account).unwrap();
    t.check("supervisor-selected shared accounts fund multiple applications", true, "");

    if let Some(image) = boot.handle("module/posixtest") {
        let account = sys::account_create(system, APP_LIMIT).unwrap();
        let child = sys::spawn_linux_in(account, image, &["posixtest", "--budget-probe"], &[], None).unwrap();
        assert_eq!(sys::process_wait(child, sys::after_ms(10000)), Ok(0)); sys::close(child).unwrap();
        empty_account(account); sys::close(account).unwrap();
        t.check("Linux spawn, fork and exec preserve ancestor budget", true, "");
    }
}
