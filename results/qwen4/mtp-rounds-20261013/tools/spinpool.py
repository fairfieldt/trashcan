#!/usr/bin/env python3
"""Candidate: pool workers spin a while for the next job before sleeping (applied temporarily to build a test binary)."""
import sys
p = sys.argv[1] + '/engine/src/qwen4/draft.rs'
s = open(p).read()
old_start = s.index('struct Shared {')
old_end = s.index('/// A pointer for writing disjoint parts of one buffer from several tasks.')
new = '''struct Shared {
    /// Bumped for every job; workers spin on it for a while before sleeping.
    gen: std::sync::atomic::AtomicU64,
    job: Mutex<Option<Arc<Job>>>,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
}

/// Spins of a worker waiting for the next job before it sleeps (about 100 us): the jobs of a draft step follow each
/// other within microseconds.
const SPINS: u32 = 40_000;

/// Worker threads that run the parallel loops of a draft step (spawning threads for every product would cost more
/// than the products).
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

fn work(job: &Job) {
    loop {
        let i = job.next.fetch_add(1, Ordering::Relaxed);
        if i >= job.tasks {
            return;
        }
        // The caller keeps `f` alive until every task is done.
        unsafe { (*job.f)(i) };
        job.done.fetch_add(1, Ordering::Release);
    }
}

impl Pool {
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            gen: std::sync::atomic::AtomicU64::new(0),
            job: Mutex::new(None),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let workers = (1..threads.max(1))
            .map(|_| {
                let s = shared.clone();
                std::thread::spawn(move || {
                    let mut seen = 0u64;
                    loop {
                        let mut spins = 0;
                        loop {
                            let g = s.gen.load(Ordering::SeqCst);
                            if g != seen {
                                seen = g;
                                break;
                            }
                            if spins < SPINS {
                                spins += 1;
                                std::hint::spin_loop();
                                continue;
                            }
                            let mut l = s.lock.lock().unwrap();
                            s.sleepers.fetch_add(1, Ordering::SeqCst);
                            while s.gen.load(Ordering::SeqCst) == seen {
                                l = s.wake.wait(l).unwrap();
                            }
                            s.sleepers.fetch_sub(1, Ordering::SeqCst);
                        }
                        let job = s.job.lock().unwrap().clone();
                        match job {
                            Some(j) => work(&j),
                            None => return,
                        }
                    }
                })
            })
            .collect();
        Self { shared, workers }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    fn publish(&self, job: Option<Arc<Job>>) {
        *self.shared.job.lock().unwrap() = job;
        self.shared.gen.fetch_add(1, Ordering::SeqCst);
        if self.shared.sleepers.load(Ordering::SeqCst) > 0 {
            let _l = self.shared.lock.lock().unwrap();
            self.shared.wake.notify_all();
        }
    }

    /// Run `f(0..tasks)` on every thread, the caller's included; returns when all are done.
    pub fn run(&self, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        if tasks == 0 {
            return;
        }
        // Lifetime erased: `run` waits for every task before returning.
        let f: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let job = Arc::new(Job {
            f,
            tasks,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
        });
        if tasks > 1 && !self.workers.is_empty() {
            self.publish(Some(job.clone()));
        }
        work(&job);
        while job.done.load(Ordering::Acquire) < tasks {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // Workers that wake for this generation find no job and stop; a sleeping one is woken by the notify (it
        // rechecks the generation under the lock).
        *self.shared.job.lock().unwrap() = None;
        self.shared.gen.fetch_add(1, Ordering::SeqCst);
        {
            let _l = self.shared.lock.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

'''
s = s[:old_start] + new + s[old_end:]
open(p, 'w').write(s)
print('spin pool applied')
