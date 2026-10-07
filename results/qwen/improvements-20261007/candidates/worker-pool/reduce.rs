//! Persistent host workers for tensor-parallel reductions. Jobs borrow mapped GPU buffers only until
//! all completion counters are acquired; each worker owns a disjoint, 16-float-aligned range of every destination.
use anyhow::{ensure, Result};
use std::cell::UnsafeCell;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};
use std::thread::{self, JoinHandle};

pub const LOCAL: usize = 8;

#[derive(Clone, Copy)]
struct Job {
    sources: [*const f32; LOCAL],
    targets: [*mut f32; LOCAL + 1],
    sources_n: usize,
    targets_n: usize,
    len: usize,
    size: usize,
}
// The caller keeps all buffers alive until every completion is acquired. Sources are read-only (except source 0
// may equal target 0); destination chunks are disjoint and cannot be read by GPUs until run returns.
unsafe impl Send for Job {}

impl Job {
    fn chunk(self, worker: usize) {
        let at = worker * self.size;
        let n = self.size.min(self.len.saturating_sub(at));
        if n == 0 {
            return;
        }
        unsafe {
            let out = std::slice::from_raw_parts_mut(self.targets[0].add(at), n);
            if self.sources[0] != self.targets[0].cast_const() {
                out.copy_from_slice(std::slice::from_raw_parts(self.sources[0].add(at), n));
            }
            for src in &self.sources[1..self.sources_n] {
                for (a, b) in out
                    .iter_mut()
                    .zip(std::slice::from_raw_parts(src.add(at), n))
                {
                    *a += b;
                }
            }
            for dst in &self.targets[1..self.targets_n] {
                std::ptr::copy_nonoverlapping(out.as_ptr(), dst.add(at), n);
            }
        }
    }
}

struct Shared {
    job: UnsafeCell<Option<Job>>,
    generation: AtomicUsize,
    done: Vec<AtomicUsize>,
    wake: Condvar,
    sleep: Mutex<()>,
}
// A release publication makes the job visible; the caller never replaces it before every worker's
// release completion has been acquired. The mutex prevents a lost wake when an idle worker sleeps.
unsafe impl Sync for Shared {}

pub struct Workers {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
    #[cfg(target_os = "linux")]
    affinity: Option<libc::cpu_set_t>,
}

impl Workers {
    pub fn new(threads: usize) -> Result<Self> {
        ensure!((1..=16).contains(&threads), "1 to 16 reduction threads");
        let cpus = std::env::var("TRASHCAN_TP_CPUS")
            .ok()
            .map(|v| {
                v.split(',')
                    .map(str::parse::<usize>)
                    .collect::<std::result::Result<Vec<_>, _>>()
            })
            .transpose()?;
        if let Some(cpus) = &cpus {
            ensure!(
                cpus.len() >= threads && cpus.iter().all(|&c| c < 1024),
                "TRASHCAN_TP_CPUS needs one CPU per reduction thread"
            );
        }
        #[cfg(target_os = "linux")]
        let affinity = if let Some(cpus) = &cpus {
            let mut old: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            ensure!(
                unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&old), &mut old) } == 0,
                "reading CPU affinity failed"
            );
            ensure!(
                cpus.iter()
                    .all(|&cpu| unsafe { libc::CPU_ISSET(cpu, &old) }),
                "worker CPUs are outside the allowed CPU set"
            );
            ensure!(
                cpus.iter()
                    .enumerate()
                    .all(|(i, cpu)| !cpus[..i].contains(cpu)),
                "worker CPUs must be distinct"
            );
            pin(cpus[threads - 1]);
            Some(old)
        } else {
            None
        };
        let shared = Arc::new(Shared {
            job: UnsafeCell::new(None),
            generation: AtomicUsize::new(0),
            done: (0..threads - 1).map(|_| AtomicUsize::new(0)).collect(),
            wake: Condvar::new(),
            sleep: Mutex::new(()),
        });
        let handles = (0..threads - 1)
            .map(|i| {
                let shared = shared.clone();
                let cpu = cpus.as_ref().map(|cpus| cpus[i]);
                thread::spawn(move || {
                    #[cfg(target_os = "linux")]
                    if let Some(cpu) = cpu {
                        pin(cpu);
                    }
                    #[cfg(not(target_os = "linux"))]
                    let _ = cpu;
                    let mut seen = 0;
                    loop {
                        let mut lock = shared.sleep.lock().unwrap();
                        while shared.generation.load(Ordering::Acquire) == seen {
                            lock = shared.wake.wait(lock).unwrap();
                        }
                        seen = shared.generation.load(Ordering::Acquire);
                        drop(lock);
                        if seen == usize::MAX {
                            break;
                        }
                        let j = unsafe { (*shared.job.get()).unwrap() };
                        j.chunk(i);
                        shared.done[i].store(seen, Ordering::Release);
                    }
                })
            })
            .collect();
        Ok(Self {
            shared,
            handles,
            #[cfg(target_os = "linux")]
            affinity,
        })
    }

    pub fn threads(&self) -> usize {
        self.handles.len() + 1
    }

    /// Sum sources in order and optionally copy to mapped per-rank destinations. Source zero may equal
    /// out.
    ///
    /// # Safety
    /// Other sources must not alias out; targets must be disjoint and valid for out.len() floats.
    /// All pointed-to memory must remain accessible until this method returns.
    pub unsafe fn sum(
        &mut self,
        out: &mut [f32],
        sources: &[*const f32],
        targets: &[*mut f32],
        parallel: bool,
    ) {
        assert!(!sources.is_empty() && sources.len() <= LOCAL && targets.len() <= LOCAL);
        let threads = if parallel { self.threads() } else { 1 };
        let mut j = Job {
            sources: [std::ptr::null(); LOCAL],
            targets: [std::ptr::null_mut(); LOCAL + 1],
            sources_n: sources.len(),
            targets_n: targets.len() + 1,
            len: out.len(),
            size: out.len().div_ceil(threads).next_multiple_of(16),
        };
        j.sources[..sources.len()].copy_from_slice(sources);
        j.targets[0] = out.as_mut_ptr();
        j.targets[1..targets.len() + 1].copy_from_slice(targets);
        if threads == 1 {
            j.chunk(0);
        } else {
            let lock = self.shared.sleep.lock().unwrap();
            *self.shared.job.get() = Some(j);
            let generation = self.shared.generation.fetch_add(1, Ordering::Release) + 1;
            self.shared.wake.notify_all();
            drop(lock);
            j.chunk(threads - 1);
            for done in &self.shared.done {
                while done.load(Ordering::Acquire) != generation {
                    std::hint::spin_loop();
                }
            }
            // All workers have finished accessing the borrowed buffers before this method returns.
        }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        let lock = self.shared.sleep.lock().unwrap();
        self.shared.generation.store(usize::MAX, Ordering::Release);
        self.shared.wake.notify_all();
        drop(lock);
        for h in self.handles.drain(..) {
            h.join().unwrap();
        }
        #[cfg(target_os = "linux")]
        if let Some(old) = &self.affinity {
            unsafe {
                libc::sched_setaffinity(0, std::mem::size_of_val(old), old);
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn pin(cpu: usize) {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::CPU_SET(cpu, &mut set);
        assert_eq!(
            libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set),
            0,
            "setting worker CPU affinity failed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_sums_reuse_workers_and_copy_uneven_chunks() {
        for threads in [1, 2, 6, 16] {
            let mut pool = Workers::new(threads).unwrap();
            for len in [0, 1, 17, 257, 65537] {
                let a: Vec<_> = (0..len).map(|i| i as f32 / 7.).collect();
                let b: Vec<_> = (0..len).map(|i| -(i as f32) / 9.).collect();
                let c = vec![0.01; len];
                let mut out = vec![0.; len];
                let (mut x, mut y) = (vec![0.; len], vec![0.; len]);
                unsafe {
                    pool.sum(&mut out, &[a.as_ptr(), b.as_ptr()], &[], true);
                    let ptr = out.as_ptr();
                    pool.sum(
                        &mut out,
                        &[ptr, c.as_ptr()],
                        &[x.as_mut_ptr(), y.as_mut_ptr()],
                        true,
                    );
                }
                let want: Vec<_> = a.iter().zip(&b).map(|(a, b)| (a + b) + 0.01).collect();
                assert_eq!(out, want);
                assert_eq!(x, want);
                assert_eq!(y, want);
                unsafe {
                    pool.sum(&mut out, &[a.as_ptr()], &[], false);
                }
                assert_eq!(out, a);
            }
        }
    }
}
