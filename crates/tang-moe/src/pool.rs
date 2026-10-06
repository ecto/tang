//! A spinning job pool for the CPU miss path: one pinned worker per P-core (E-cores opt-in),
//! and the calling thread works too.
//!
//! A dispatch is one 64-bit control word, `epoch(32) | njobs(16) | next(16)`. Workers claim job
//! `next` with a CAS on the whole word, so a claim can never land in a stale epoch. Completions
//! go to a separate counter on its own cache line. Idle workers spin for `spin` (default
//! 20 ms) after their last job, then park; a dispatch unparks only workers that parked.
//! Nothing here takes a lock on the hot path, and nothing is shared with CUDA driver threads.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[repr(align(64))]
struct Padded<T>(T);

type Job<'a> = &'a (dyn Fn(usize) + Sync);

struct Shared {
    ctrl: Padded<AtomicU64>,
    done: Padded<AtomicU32>,
    /// Points at the dispatching thread's `Job` for the current epoch.
    job: Padded<AtomicPtr<()>>,
    parked: Vec<Padded<AtomicBool>>,
    stop: AtomicBool,
    spin: Duration,
}

const NEXT_MASK: u64 = 0xffff;

fn unpack(w: u64) -> (u32, usize, usize) {
    (
        (w >> 32) as u32,
        ((w >> 16) & 0xffff) as usize,
        (w & NEXT_MASK) as usize,
    )
}

impl Shared {
    /// Claim and run jobs of the current epoch until none are left. Returns jobs run.
    fn work(&self) -> usize {
        let mut ran = 0;
        loop {
            let w = self.ctrl.0.load(Ordering::Acquire);
            let (_, n, next) = unpack(w);
            if next >= n {
                return ran;
            }
            if self
                .ctrl
                .0
                .compare_exchange_weak(w, w + 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            // SAFETY: the epoch's job outlives every claim: `run` waits for all completions.
            let job: Job = unsafe { *(self.job.0.load(Ordering::Acquire) as *const Job) };
            job(next);
            ran += 1;
            self.done.0.fetch_add(1, Ordering::Release);
        }
    }
}

/// Which CPUs to use, from sysfs on a hybrid Intel part.
#[derive(Clone, Debug)]
pub struct Topology {
    /// First hardware thread of each P-core.
    pub pcores: Vec<usize>,
    /// E-core CPUs.
    pub ecores: Vec<usize>,
}

impl Topology {
    pub fn detect() -> Topology {
        let parse = |s: &str| -> Vec<usize> {
            let mut v = Vec::new();
            for part in s.trim().split(',') {
                if let Some((a, b)) = part.split_once('-') {
                    if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                        v.extend(a..=b);
                    }
                } else if let Ok(a) = part.parse() {
                    v.push(a);
                }
            }
            v
        };
        let read = |p: &str| std::fs::read_to_string(p).ok().map(|s| parse(&s));
        match (
            read("/sys/devices/cpu_core/cpus"),
            read("/sys/devices/cpu_atom/cpus"),
        ) {
            (Some(p), Some(e)) => {
                // Keep one hardware thread per core (the lowest in its sibling list).
                let first = |c: &usize| {
                    read(&format!(
                        "/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
                    ))
                    .and_then(|s| s.first().copied())
                    .is_none_or(|f| f == *c)
                };
                Topology {
                    pcores: p.iter().filter(|c| first(c)).copied().collect(),
                    ecores: e,
                }
            }
            _ => {
                let n = std::thread::available_parallelism().map_or(1, |n| n.get());
                Topology {
                    pcores: (0..n).collect(),
                    ecores: Vec::new(),
                }
            }
        }
    }
}

pub fn pin_to(cpu: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
    let _ = cpu;
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    epoch: u32,
    threads: usize,
}

impl Pool {
    /// Pin the calling thread to `cpus[0]` and start a worker on each other CPU in `cpus`.
    pub fn new(cpus: &[usize], spin: Duration) -> Pool {
        assert!(!cpus.is_empty());
        pin_to(cpus[0]);
        let shared = Arc::new(Shared {
            ctrl: Padded(AtomicU64::new(0)),
            done: Padded(AtomicU32::new(0)),
            job: Padded(AtomicPtr::new(std::ptr::null_mut())),
            parked: (1..cpus.len())
                .map(|_| Padded(AtomicBool::new(false)))
                .collect(),
            stop: AtomicBool::new(false),
            spin,
        });
        let workers = cpus[1..]
            .iter()
            .enumerate()
            .map(|(i, &cpu)| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("tang-moe-{cpu}"))
                    .spawn(move || worker(sh, i, cpu))
                    .unwrap()
            })
            .collect();
        Pool {
            shared,
            workers,
            epoch: 0,
            threads: cpus.len(),
        }
    }

    /// The default pool: every P-core, plus E-cores if `TANG_MOE_ECORES=1`.
    pub fn default_cpus() -> Vec<usize> {
        let t = Topology::detect();
        let mut cpus = t.pcores;
        if std::env::var("TANG_MOE_ECORES").is_ok_and(|v| v == "1") {
            cpus.extend(t.ecores);
        }
        cpus
    }

    /// Threads that run jobs, the caller included.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `f(0..njobs)` across the pool and the calling thread; returns when all are done.
    pub fn run(&mut self, njobs: usize, f: &(dyn Fn(usize) + Sync)) {
        if njobs == 0 {
            return;
        }
        assert!(njobs <= 0xffff);
        let sh = &*self.shared;
        let job: Job = f;
        sh.job
            .0
            .store(&job as *const Job as *mut (), Ordering::Release);
        sh.done.0.store(0, Ordering::Relaxed);
        self.epoch = self.epoch.wrapping_add(1);
        sh.ctrl.0.store(
            ((self.epoch as u64) << 32) | ((njobs as u64) << 16),
            Ordering::SeqCst,
        );
        for (i, p) in sh.parked.iter().enumerate() {
            if p.0.swap(false, Ordering::SeqCst) {
                self.workers[i].thread().unpark();
            }
        }
        sh.work();
        while sh.done.0.load(Ordering::Acquire) < njobs as u32 {
            std::hint::spin_loop();
        }
    }
}

fn worker(sh: Arc<Shared>, i: usize, cpu: usize) {
    pin_to(cpu);
    let mut idle_since = Instant::now();
    let mut polls = 0u32;
    while !sh.stop.load(Ordering::Relaxed) {
        if sh.work() > 0 {
            idle_since = Instant::now();
            continue;
        }
        std::hint::spin_loop();
        polls = polls.wrapping_add(1);
        if !polls.is_multiple_of(1024) || idle_since.elapsed() < sh.spin {
            continue;
        }
        // Park, re-checking for work after announcing it (pairs with `run`'s swap).
        sh.parked[i].0.store(true, Ordering::SeqCst);
        let (_, n, next) = unpack(sh.ctrl.0.load(Ordering::SeqCst));
        if next < n || sh.stop.load(Ordering::SeqCst) {
            sh.parked[i].0.store(false, Ordering::SeqCst);
            continue;
        }
        std::thread::park();
        sh.parked[i].0.store(false, Ordering::SeqCst);
        idle_since = Instant::now();
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        for w in self.workers.drain(..) {
            w.thread().unpark();
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn runs_every_job_once_across_epochs_and_parking() {
        let n = std::thread::available_parallelism()
            .map_or(2, |n| n.get())
            .min(4);
        let cpus: Vec<usize> = (0..n).collect();
        let mut pool = Pool::new(&cpus, Duration::from_millis(1));
        for round in 0..200 {
            let jobs = 1 + round % 37;
            let hits: Vec<AtomicUsize> = (0..jobs).map(|_| AtomicUsize::new(0)).collect();
            pool.run(jobs, &|j| {
                hits[j].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                "round {round}"
            );
            if round % 50 == 0 {
                std::thread::sleep(Duration::from_millis(5)); // let workers park
            }
        }
    }
}
