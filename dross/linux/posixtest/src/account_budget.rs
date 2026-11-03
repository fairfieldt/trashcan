use dross_rt::{sys, Handle, Error};
const LIMIT: u64 = 32 << 20;
fn check() {
    assert_eq!(sys::account_info(Handle(0)).unwrap()[2], LIMIT);
    assert_eq!(sys::mem_create(LIMIT, 0), Err(Error::NoMemory));
}
pub fn run(args: &[String]) -> bool {
    match args.get(1).map(String::as_str) {
        Some("--budget-exec") => { check(); println!("budget: exec retained its account"); true }
        Some("--budget-probe") => {
            check();
            // SAFETY: single-threaded test; child only checks its account and execs.
            unsafe {
                let child = libc::fork();
                if child < 0 {
                    assert_eq!(*libc::__errno_location(), libc::ENOSYS);
                    println!("budget: default Linux personality has no fork");
                    return true;
                }
                if child == 0 {
                    check();
                    libc::execl(c"/usr/bin/dross-budget-probe".as_ptr(), c"budget-probe".as_ptr(), c"--budget-exec".as_ptr(), core::ptr::null::<libc::c_char>());
                    libc::_exit(77);
                }
                let mut status = 0; assert_eq!(libc::waitpid(child, &mut status, 0), child);
                assert!(libc::WIFEXITED(status)); assert_eq!(libc::WEXITSTATUS(status), 0);
            }
            check(); println!("budget: fork/exec stayed within the ancestor limit"); true
        }
        _ => false,
    }
}
