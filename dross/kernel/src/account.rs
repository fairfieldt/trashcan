//! Hierarchical allocation budgets. Limits grant permission, not reserved RAM.
//! Each backing reservation charges its leaf and every ancestor, with rollback.
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};
use dross_abi::Error;
const MAX_DEPTH: usize = 32;
pub const METADATA_BYTES: u64 = 128;
pub struct Account {
    parent: Option<Arc<Account>>,
    limit: u64,
    used: AtomicU64,
    peak: AtomicU64,
    depth: usize,
    _metadata: Option<Charge>,
}
impl Account {
    /// Boot/audit roots only. User-created accounts always have a parent.
    pub fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self { parent: None, limit, used: AtomicU64::new(0), peak: AtomicU64::new(0), depth: 1, _metadata: None })
    }
    pub fn child(parent: Arc<Account>, limit: u64) -> Result<Arc<Self>, Error> {
        if parent.depth >= MAX_DEPTH { return Err(Error::NoResources); }
        let metadata = parent.reserve(METADATA_BYTES)?;
        Ok(Arc::new(Self { depth: parent.depth + 1, parent: Some(parent), limit,
            used: AtomicU64::new(0), peak: AtomicU64::new(0), _metadata: Some(metadata) }))
    }
    pub fn inherited() -> Result<Arc<Self>, Error> {
        let parent = current(); Self::child(parent.clone(), parent.limit)
    }
    pub fn stats(&self) -> [u64; 3] {
        let used = self.used.load(Ordering::Acquire);
        [used, self.peak.load(Ordering::Acquire).max(used), self.limit]
    }
    pub fn reserve(self: &Arc<Self>, bytes: u64) -> Result<Charge, Error> {
        let mut reserved = 0;
        let mut node = Some(self.as_ref());
        while let Some(account) = node {
            let mut used = account.used.load(Ordering::Acquire);
            loop {
                let Some(next) = used.checked_add(bytes).filter(|&v| v <= account.limit) else {
                    let mut rollback = Some(self.as_ref());
                    for _ in 0..reserved {
                        let account = rollback.unwrap(); account.used.fetch_sub(bytes, Ordering::AcqRel);
                        rollback = account.parent.as_deref();
                    }
                    return Err(Error::NoMemory);
                };
                match account.used.compare_exchange_weak(used, next, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => { account.peak.fetch_max(next, Ordering::Relaxed); break; }
                    Err(actual) => used = actual,
                }
            }
            reserved += 1;
            node = account.parent.as_deref();
        }
        Ok(Charge { account: self.clone(), bytes })
    }
}
pub struct Charge { account: Arc<Account>, bytes: u64 }
impl Charge {
    pub fn absorb(&mut self, mut other: Charge) {
        assert!(Arc::ptr_eq(&self.account, &other.account));
        self.bytes = self.bytes.checked_add(other.bytes).expect("charge overflow");
        other.bytes = 0;
    }
    pub fn split(&mut self, bytes: u64) -> Charge {
        assert!(bytes <= self.bytes); self.bytes -= bytes;
        Charge { account: self.account.clone(), bytes }
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        let mut node = Some(self.account.as_ref());
        while let Some(account) = node {
            account.used.fetch_sub(self.bytes, Ordering::AcqRel); node = account.parent.as_deref();
        }
    }
}
pub fn current() -> Arc<Account> {
    if let Some(p) = crate::sched::try_current_process() { return p.account.clone(); }
    static BOOT: spin::Mutex<Option<Arc<Account>>> = spin::Mutex::new(None);
    BOOT.lock().get_or_insert_with(|| Account::new(u64::MAX)).clone()
}
#[cfg(feature = "audit-test")]
pub fn audit_test() {
    let account = Account::new(4096);
    let charge = account.reserve(4096).unwrap();
    assert!(matches!(account.reserve(1), Err(Error::NoMemory)));
    assert_eq!(account.stats(), [4096, 4096, 4096]); drop(charge);
    assert_eq!(account.stats(), [0, 4096, 4096]);
    let parent = Account::new(16384);
    let a = Account::child(parent.clone(), 12288).unwrap();
    let b = Account::child(parent.clone(), 12288).unwrap();
    let baseline = parent.stats()[0];
    let charge = a.reserve(8192).unwrap();
    assert!(matches!(b.reserve(8192), Err(Error::NoMemory)));
    assert_eq!(b.stats()[0], 0); assert_eq!(parent.stats()[0], baseline + 8192);
    drop(charge); drop(a); drop(b); assert_eq!(parent.stats()[0], 0);
    crate::println!("memory: ancestor budget reservation and rollback passed");
}
