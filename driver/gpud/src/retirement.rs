//! Last uses of backing memory. SI fences are wrapping 32-bit counters. A buffer
//! may be retired only after every ring that could reach it has passed its fence.
use std::collections::HashMap;

#[derive(Default)]
pub struct LastUse {
    rings: HashMap<i32, u64>,
}
impl LastUse {
    pub fn record(&mut self, ring: i32, sequence: u64) {
        self.rings
            .entry(ring)
            .and_modify(|old| {
                // Concurrent submissions can publish their fences out of order.
                if (sequence as u32).wrapping_sub(*old as u32) as i32 > 0 {
                    *old = sequence;
                }
            })
            .or_insert(sequence);
    }
    pub fn complete(&self, mut read: impl FnMut(i32) -> u64) -> bool {
        self.rings.iter().all(|(&ring, &sequence)| (read(ring) as u32).wrapping_sub(sequence as u32) as i32 >= 0)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_rings_must_complete_and_out_of_order_publication_cannot_retire_early() {
        let mut use_ = LastUse::default();
        use_.record(0, 8);
        use_.record(1, 4);
        use_.record(0, 7);
        assert!(!use_.complete(|ring| if ring == 0 { 7 } else { 4 }));
        assert!(!use_.complete(|ring| if ring == 0 { 8 } else { 3 }));
        assert!(use_.complete(|ring| if ring == 0 { 8 } else { 4 }));
    }
    #[test]
    fn wrapping_fences_and_quarantined_backing() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        struct Backing(Arc<AtomicUsize>);
        impl Drop for Backing {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let mut backing = Some(Backing(drops.clone()));
        let mut use_ = LastUse::default();
        use_.record(0, u32::MAX as u64);
        use_.record(0, 1);
        use_.record(0, u32::MAX as u64);
        if use_.complete(|_| 0) {
            drop(backing.take());
        }
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(backing.is_some());
        if use_.complete(|_| 1) {
            drop(backing.take());
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
