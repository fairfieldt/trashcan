//! Persistent host workers for tensor-parallel reductions. Jobs borrow mapped GPU buffers only until
//! both barriers complete; each worker owns a disjoint, 16-float-aligned range of every destination.
use anyhow::{ensure, Result};
use std::sync::{Arc, Barrier, Mutex};
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
// The caller keeps all buffers alive until the finish barrier. Sources are read-only (except source 0
// may equal target 0); destination chunks are disjoint and cannot be read by GPUs until run returns.
unsafe impl Send for Job {}

impl Job {
    fn chunk(self, worker: usize) {
        let at = worker * self.size;
        let n = self.size.min(self.len.saturating_sub(at));
        if n == 0 { return; }
        unsafe {
            let out = std::slice::from_raw_parts_mut(self.targets[0].add(at), n);
            if self.sources[0] != self.targets[0].cast_const() {
                out.copy_from_slice(std::slice::from_raw_parts(self.sources[0].add(at), n));
            }
            for src in &self.sources[1..self.sources_n] {
                for (a, b) in out.iter_mut().zip(std::slice::from_raw_parts(src.add(at), n)) {
                    *a += b;
                }
            }
            for dst in &self.targets[1..self.targets_n] {
                std::ptr::copy_nonoverlapping(out.as_ptr(), dst.add(at), n);
            }
        }
    }
}

pub struct Workers {
    start: Arc<Barrier>,
    finish: Arc<Barrier>,
    job: Arc<Mutex<Option<Job>>>,
    handles: Vec<JoinHandle<()>>,
    #[cfg(target_os = "linux")]
    affinity: Option<libc::cpu_set_t>,
}

impl Workers {
    pub fn new(threads: usize) -> Result<Self> {
        ensure!((1..=16).contains(&threads), "1 to 16 reduction threads");
        let cpus = std::env::var("TRASHCAN_TP_CPUS").ok().map(|v| v.split(',').map(str::parse::<usize>).collect::<std::result::Result<Vec<_>, _>>()).transpose()?;
        if let Some(cpus) = &cpus { ensure!(cpus.len() >= threads && cpus.iter().all(|&c| c < 1024), "TRASHCAN_TP_CPUS needs one CPU per reduction thread"); }
        #[cfg(target_os = "linux")]
        let affinity = if let Some(cpus) = &cpus {
            let mut old: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            ensure!(unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&old), &mut old) } == 0, "reading CPU affinity failed");
            pin(cpus[threads - 1]);
            Some(old)
        } else { None };
        let start = Arc::new(Barrier::new(threads));
        let finish = Arc::new(Barrier::new(threads));
        let job = Arc::new(Mutex::new(None::<Job>));
        let handles = (0..threads - 1).map(|i| {
            let (start, finish, job) = (start.clone(), finish.clone(), job.clone());
            let cpu = cpus.as_ref().map(|cpus| cpus[i]);
            thread::spawn(move || {
                #[cfg(target_os = "linux")]
                if let Some(cpu) = cpu { pin(cpu); }
                #[cfg(not(target_os = "linux"))]
                let _ = cpu;
                loop {
                start.wait();
                let Some(j) = *job.lock().unwrap() else { break };
                j.chunk(i);
                finish.wait();
                }
            })
        }).collect();
        Ok(Self { start, finish, job, handles, #[cfg(target_os = "linux")] affinity })
    }

    pub fn threads(&self) -> usize { self.handles.len() + 1 }

    /// Sum sources in their existing order and optionally copy to mapped per-rank destinations. The sum
    /// may itself be the first source. Additional targets must be disjoint and valid for sum.len() floats.
    pub unsafe fn sum(&mut self, out: &mut [f32], sources: &[*const f32], targets: &[*mut f32], parallel: bool) {
        assert!(!sources.is_empty() && sources.len() <= LOCAL && targets.len() <= LOCAL);
        let threads = if parallel { self.threads() } else { 1 };
        let mut j = Job { sources: [std::ptr::null(); LOCAL], targets: [std::ptr::null_mut(); LOCAL + 1],
            sources_n: sources.len(), targets_n: targets.len() + 1, len: out.len(),
            size: out.len().div_ceil(threads).next_multiple_of(16) };
        j.sources[..sources.len()].copy_from_slice(sources);
        j.targets[0] = out.as_mut_ptr();
        j.targets[1..targets.len() + 1].copy_from_slice(targets);
        if threads == 1 {
            j.chunk(0);
        } else {
            *self.job.lock().unwrap() = Some(j);
            self.start.wait();
            j.chunk(threads - 1);
            self.finish.wait();
            // No worker can access the borrowed buffers after the finish barrier.
            *self.job.lock().unwrap() = None;
        }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        *self.job.lock().unwrap() = None;
        self.start.wait();
        for h in self.handles.drain(..) { h.join().unwrap(); }
        #[cfg(target_os = "linux")]
        if let Some(old) = &self.affinity { unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(old), old); } }
    }
}

#[cfg(target_os = "linux")]
fn pin(cpu: usize) {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::CPU_SET(cpu, &mut set);
        assert_eq!(libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set), 0, "setting worker CPU affinity failed");
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
                    pool.sum(&mut out, &[ptr, c.as_ptr()], &[x.as_mut_ptr(), y.as_mut_ptr()], true);
                }
                let want: Vec<_> = a.iter().zip(&b).map(|(a, b)| (a + b) + 0.01).collect();
                assert_eq!(out, want);
                assert_eq!(x, want);
                assert_eq!(y, want);
                unsafe { pool.sum(&mut out, &[a.as_ptr()], &[], false); }
                assert_eq!(out, a);
            }
        }
    }
}
