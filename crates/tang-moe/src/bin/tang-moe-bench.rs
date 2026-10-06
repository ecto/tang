//! Microbenchmarks for the GPU-only offload design. See crates/tang-moe/BENCH.md.
//!
//! ```text
//! tang-moe-bench register [--max-gb 20]
//! tang-moe-bench mapped|copy|window|dram|all [--arena-gb 16] [--iters 30] [--no-thp]
//! tang-moe-bench graph
//! tang-moe-bench spin
//! ```

#[path = "../bench/missbench.rs"]
mod missbench;

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use cudarc::driver::sys;
use tang_moe::args;
use tang_moe::gpu::{check, launch, DevBuf, Event, Gpu, Graph, Result, Stream};
use tang_moe::{kernels, ArenaOptions, Geometry, HostArena};

const BLOB: usize = Geometry::FLASH_NEXT_Q2_0.blob_bytes;
const ROWS: usize = 1280 + 2560;
const GB: usize = 1 << 30;

struct Opts {
    arena_gb: usize,
    iters: usize,
    thp: bool,
    max_gb: usize,
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().cloned().unwrap_or_else(|| "all".into());
    let mut o = Opts {
        arena_gb: 16,
        iters: 30,
        thp: true,
        max_gb: 20,
    };
    let mut i = 1;
    while i < argv.len() {
        let val = |i: usize| -> usize { argv[i + 1].parse().expect("number") };
        match argv[i].as_str() {
            "--arena-gb" => {
                o.arena_gb = val(i);
                i += 1
            }
            "--iters" => {
                o.iters = val(i);
                i += 1
            }
            "--max-gb" => {
                o.max_gb = val(i);
                i += 1
            }
            "--no-thp" => o.thp = false,
            a => panic!("unknown argument {a}"),
        }
        i += 1;
    }
    if let Err(e) = run(&cmd, &o) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(cmd: &str, o: &Opts) -> Result<()> {
    let gpu = Gpu::new()?;
    gpu.bind()?;
    print_env(&gpu)?;
    let parts: Vec<&str> = if cmd == "all" {
        vec!["mapped", "window", "copy", "dram", "pcie", "graph", "spin"]
    } else {
        cmd.split(',').collect()
    };
    let needs_arena = parts
        .iter()
        .any(|p| matches!(*p, "mapped" | "window" | "copy" | "dram"));
    let bench = if needs_arena {
        Some(Bench::new(&gpu, o)?)
    } else {
        None
    };
    for p in parts {
        match p {
            "register" => register(&gpu, o)?,
            "graph" => graph(&gpu)?,
            "spin" => spin(&gpu)?,
            "pcie" => pcie(&gpu)?,
            "size" => size(&gpu)?,
            "cpu" => missbench::cpu(Some(&gpu), o.iters)?,
            "missw" => missbench::missw(&gpu, o.iters)?,
            "mapped" => bench.as_ref().unwrap().mapped(o)?,
            "window" => bench.as_ref().unwrap().window(o)?,
            "copy" => bench.as_ref().unwrap().copy(o)?,
            "dram" => bench.as_ref().unwrap().dram()?,
            c => panic!("unknown command {c}"),
        }
    }
    Ok(())
}

fn print_env(gpu: &Gpu) -> Result<()> {
    let (free, total) = gpu.mem_info()?;
    let mut rl = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rl) };
    println!(
        "# gpu: {} free / {} total MiB; RLIMIT_MEMLOCK cur={} max={}",
        free >> 20,
        total >> 20,
        fmt_lim(rl.rlim_cur),
        fmt_lim(rl.rlim_max)
    );
    if let Ok(m) = std::fs::read_to_string("/proc/meminfo") {
        let pick: Vec<&str> = m
            .lines()
            .filter(|l| l.starts_with("MemAvailable") || l.starts_with("HugePages_Total"))
            .collect();
        println!("# host: {}", pick.join("; "));
    }
    Ok(())
}

fn fmt_lim(v: u64) -> String {
    if v == u64::MAX {
        "unlimited".into()
    } else {
        format!("{} MiB", v >> 20)
    }
}

// ---------- stats ----------

struct Summary {
    med: f64,
    p10: f64,
    p90: f64,
    p99: f64,
    max: f64,
}

fn summarize(mut v: Vec<f64>) -> Summary {
    v.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    Summary {
        med: q(0.5),
        p10: q(0.1),
        p90: q(0.9),
        p99: q(0.99),
        max: *v.last().unwrap(),
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// `n` distinct values in `[lo, hi)`.
    fn distinct(&mut self, n: usize, lo: usize, hi: usize) -> Vec<i32> {
        let mut out: Vec<i32> = Vec::with_capacity(n);
        while out.len() < n {
            let v = (lo + (self.next() as usize) % (hi - lo)) as i32;
            if !out.contains(&v) {
                out.push(v);
            }
        }
        out
    }
}

// ---------- register ----------

fn register(gpu: &Gpu, o: &Opts) -> Result<()> {
    println!("\n## register: cuMemHostRegister(PORTABLE|DEVICEMAP) before first touch");
    println!("| size | backing | result | chunk | regions | seconds | GB/s pinned | huge pages |");
    println!("|---|---|---|---|---|---|---|---|");
    for gb in [1, 2, 4, 8, 12, 16, 20] {
        if gb > o.max_gb {
            break;
        }
        let mut a = HostArena::new(
            gb * GB,
            ArenaOptions {
                try_hugetlb: true,
                thp: o.thp,
            },
        )
        .map_err(|e| tang_moe::gpu::Error(format!("mmap: {e}")))?;
        let backing = a.backing();
        match a.register(&gpu.ctx, &[4 * GB, GB, 256 << 20]) {
            Ok(r) => {
                let huge = a.huge_bytes().unwrap_or(0);
                println!(
                    "| {gb} GiB | {backing:?} | ok | {} MiB | {} | {:.2} | {:.1} | {:.0}% |",
                    r.chunk >> 20,
                    r.regions,
                    r.seconds,
                    gb as f64 * 1.073_741_824 / r.seconds,
                    100.0 * huge as f64 / a.len() as f64
                );
                if !r.refused.is_empty() {
                    println!("|   | refused first: {:?} |||||||", r.refused);
                }
            }
            Err(e) => println!("| {gb} GiB | {backing:?} | **refused** {e} ||||||"),
        }
        let t = Instant::now();
        drop(a);
        println!(
            "|   | unregister+munmap {:.2} s |||||||",
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

// ---------- the expert benches ----------

struct Bench {
    arena: HostArena,
    host_slots: usize,
    vram_slots: usize,
    _vram: DevBuf,
    /// Pointer table: host slots, then VRAM slots.
    d_ptrs: DevBuf,
    d_ids: DevBuf,
    d_x: DevBuf,
    d_xsum: DevBuf,
    d_out: DevBuf,
    x: Vec<i8>,
    xsum: Vec<f32>,
    q2: sys::CUfunction,
    stream_sum: sys::CUfunction,
    _m: tang_moe::gpu::Module,
}

const VRAM_SLOTS: usize = 1024; // 1.4 GB of VRAM-resident blobs to scatter across
const MAX_N: usize = 48 * 32; // ids per launch / window

impl Bench {
    fn new(gpu: &Gpu, o: &Opts) -> Result<Self> {
        let mut arena = HostArena::new(
            o.arena_gb * GB,
            ArenaOptions {
                try_hugetlb: true,
                thp: o.thp,
            },
        )
        .map_err(|e| tang_moe::gpu::Error(format!("mmap: {e}")))?;
        let r = arena.register(&gpu.ctx, &[4 * GB, GB, 256 << 20])?;
        println!(
            "# arena: {} GiB {:?}, registered in {:.2} s as {} region(s) of {} MiB; {:.0}% huge pages",
            arena.len() / GB,
            arena.backing(),
            r.seconds,
            r.regions,
            r.chunk >> 20,
            100.0 * arena.huge_bytes().unwrap_or(0) as f64 / arena.len() as f64
        );
        let host_slots = arena.len() / BLOB;
        let t = Instant::now();
        fill(&arena, host_slots);
        println!(
            "# filled {host_slots} host blobs in {:.1} s",
            t.elapsed().as_secs_f64()
        );

        let vram = DevBuf::alloc(VRAM_SLOTS * BLOB)?;
        // Same contents as the first VRAM_SLOTS host blobs.
        for s in 0..VRAM_SLOTS {
            let src = unsafe { arena.as_ptr().add(s * BLOB) };
            check(
                unsafe {
                    sys::cuMemcpyHtoD_v2(vram.ptr + (s * BLOB) as u64, src as *const c_void, BLOB)
                },
                "seed vram",
            )?;
        }
        let mut ptrs: Vec<u64> = (0..host_slots)
            .map(|s| arena.device_ptr(s * BLOB).unwrap())
            .collect();
        ptrs.extend((0..VRAM_SLOTS).map(|s| vram.ptr + (s * BLOB) as u64));
        let d_ptrs = DevBuf::from_slice(&ptrs)?;

        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let x: Vec<i8> = (0..2560).map(|_| (rng.next() % 255) as i8).collect();
        let xsum: Vec<f32> = (0..40)
            .map(|g| x[g * 64..(g + 1) * 64].iter().map(|&v| v as f32).sum())
            .collect();
        let m = gpu.module(kernels::BENCH)?;
        let b = Bench {
            host_slots,
            vram_slots: VRAM_SLOTS,
            d_ptrs,
            d_ids: DevBuf::zeroed(MAX_N * 4)?,
            d_x: DevBuf::from_slice(&x)?,
            d_xsum: DevBuf::from_slice(&xsum)?,
            d_out: DevBuf::zeroed(MAX_N * ROWS * 4)?,
            x,
            xsum,
            q2: m.func("q2_expert")?,
            stream_sum: m.func("stream_sum")?,
            _m: m,
            arena,
            _vram: vram,
        };
        b.self_check()?;
        Ok(b)
    }

    /// The kernel's output for one host blob and one VRAM blob matches a CPU reference.
    fn self_check(&self) -> Result<()> {
        let s = Stream::new()?;
        let host_id = (self.host_slots - 1) as i32;
        let vram_id = (self.host_slots + 3) as i32; // VRAM slot 3 = host blob 3
        self.d_ids.write(0, &[host_id, vram_id])?;
        self.q2_launch(2, &s)?;
        s.sync()?;
        let out: Vec<f32> = self.d_out.read(0, 2 * ROWS)?;
        for (j, blob) in [(0, self.host_slots - 1), (1, 3)] {
            let want = reference(
                unsafe { std::slice::from_raw_parts(self.arena.as_ptr().add(blob * BLOB), BLOB) },
                &self.x,
                &self.xsum,
            );
            for r in [0, 1, 639, 1279, 1280, 2000, ROWS - 1] {
                let (g, w) = (out[j * ROWS + r], want[r]);
                assert!(
                    (g - w).abs() <= 1e-3 * w.abs().max(1.0),
                    "q2_expert row {r} blob {blob}: gpu {g} cpu {w}"
                );
            }
        }
        println!("# q2_expert matches the CPU reference on a mapped-host blob and a VRAM blob");
        Ok(())
    }

    fn q2_launch(&self, n: usize, s: &Stream) -> Result<()> {
        unsafe {
            launch(
                self.q2,
                (320, n as u32, 1),
                (256, 1, 1),
                0,
                s,
                args![
                    self.d_ptrs.ptr,
                    self.d_ids.ptr,
                    self.d_x.ptr,
                    self.d_xsum.ptr,
                    self.d_out.ptr
                ],
            )
        }
    }

    fn q2_launch_at(&self, ids_off: usize, n: usize, s: &Stream) -> Result<()> {
        unsafe {
            launch(
                self.q2,
                (320, n as u32, 1),
                (256, 1, 1),
                0,
                s,
                args![
                    self.d_ptrs.ptr,
                    self.d_ids.ptr + (ids_off * 4) as u64,
                    self.d_x.ptr,
                    self.d_xsum.ptr,
                    self.d_out.ptr + (ids_off * ROWS * 4) as u64
                ],
            )
        }
    }

    fn stream_launch(&self, n: usize, s: &Stream) -> Result<()> {
        let vecs = (BLOB / 16) as i32;
        unsafe {
            launch(
                self.stream_sum,
                (128, n as u32, 1),
                (256, 1, 1),
                0,
                s,
                args![self.d_ptrs.ptr, self.d_ids.ptr, vecs, self.d_out.ptr],
            )
        }
    }

    /// (a) The kernel reads blobs from mapped host memory vs from VRAM.
    fn mapped(&self, o: &Opts) -> Result<()> {
        println!("\n## (a) expert kernel reading blobs in place: mapped host (PCIe) vs VRAM");
        println!(
            "host blobs scattered at random over the {} GiB arena; VRAM blobs over {:.1} GB. \
             GB/s = n × 1.3824 MB / kernel time (events). median [p10–p90] of {} runs after 3 warm-up.",
            self.arena.len() / GB,
            (self.vram_slots * BLOB) as f64 / 1e9,
            o.iters
        );
        println!("| kernel | where | n blobs | cold GB/s | cold µs | warm GB/s | warm µs |");
        println!("|---|---|---|---|---|---|---|");
        let s = Stream::new()?;
        let (e0, e1) = (Event::new(true)?, Event::new(true)?);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for kernel in ["q2_expert", "stream_sum"] {
            for host in [true, false] {
                for n in [1usize, 4, 16, 64, 128] {
                    let (lo, hi) = if host {
                        (0, self.host_slots)
                    } else {
                        (self.host_slots, self.host_slots + self.vram_slots)
                    };
                    let mut res = Vec::new();
                    for warm in [false, true] {
                        let fixed = rng.distinct(n, lo, hi);
                        let mut t = Vec::new();
                        for it in 0..o.iters + 3 {
                            let ids = if warm {
                                fixed.clone()
                            } else {
                                rng.distinct(n, lo, hi)
                            };
                            self.d_ids.write(0, &ids)?;
                            e0.record(&s)?;
                            if kernel == "q2_expert" {
                                self.q2_launch(n, &s)?;
                            } else {
                                self.stream_launch(n, &s)?;
                            }
                            e1.record(&s)?;
                            e1.sync()?;
                            if it >= 3 {
                                t.push(e1.since(&e0)? as f64 * 1e3);
                            }
                        }
                        let sm = summarize(t);
                        let gbs = |us: f64| (n * BLOB) as f64 / us / 1e3;
                        res.push(format!(
                            "{:.1} [{:.1}–{:.1}] | {:.0}",
                            gbs(sm.med),
                            gbs(sm.p90),
                            gbs(sm.p10),
                            sm.med
                        ));
                    }
                    println!(
                        "| {kernel} | {} | {n} | {} | {} |",
                        if host { "host" } else { "VRAM" },
                        res[0],
                        res[1]
                    );
                }
            }
        }
        Ok(())
    }

    /// A decode window's expert work: 48 launches (one per layer), each over `hits` VRAM
    /// blobs and `misses` mapped-host blobs, replayed as one CUDA graph.
    fn window(&self, o: &Opts) -> Result<()> {
        println!("\n## (a') a window's expert pass: 48 layer launches in one graph, hits from VRAM + misses read over PCIe");
        println!(
            "fresh random ids every replay (written to device memory before the replay). \
             median [p10–p90] of {} replays.",
            o.iters
        );
        println!("| hits/layer | misses/layer | misses/window | window ms | Δ vs 0 misses ms | µs per miss |");
        println!("|---|---|---|---|---|---|");
        let s = Stream::new()?;
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut base = None;
        for (hits, misses) in [
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 6),
            (0, 2),
            (0, 3),
        ] {
            let per = hits + misses;
            let g = Graph::capture(&s, |s| {
                for l in 0..48 {
                    self.q2_launch_at(l * per, per, s)?;
                }
                Ok(())
            })?;
            let mut t = Vec::new();
            for it in 0..o.iters + 3 {
                let mut ids = Vec::with_capacity(48 * per);
                for _ in 0..48 {
                    ids.extend(rng.distinct(
                        hits,
                        self.host_slots,
                        self.host_slots + self.vram_slots,
                    ));
                    ids.extend(rng.distinct(misses, 0, self.host_slots));
                }
                self.d_ids.write(0, &ids)?;
                let t0 = Instant::now();
                g.launch(&s)?;
                s.sync()?;
                if it >= 3 {
                    t.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            let sm = summarize(t);
            let (delta, per_miss) = match (hits, base) {
                (24, None) => {
                    base = Some(sm.med);
                    ("—".to_string(), "—".to_string())
                }
                (24, Some(b)) => (
                    format!("{:.2}", sm.med - b),
                    format!("{:.0}", (sm.med - b) * 1e3 / (48 * misses) as f64),
                ),
                _ => (
                    "—".into(),
                    format!("{:.0}", sm.med * 1e3 / (48 * misses) as f64),
                ),
            };
            println!(
                "| {hits} | {misses} | {} | {:.2} [{:.2}–{:.2}] | {delta} | {per_miss} |",
                48 * misses,
                sm.med,
                sm.p10,
                sm.p90
            );
        }
        Ok(())
    }

    /// (b) Copy misses into VRAM staging with cuMemcpyHtoDAsync, then run the kernel there.
    fn copy(&self, o: &Opts) -> Result<()> {
        println!("\n## (b) misses copied to VRAM staging (cuMemcpyHtoDAsync per blob), then processed there");
        println!("| n blobs | copy µs | copy GB/s | copy+kernel µs | effective GB/s | same n read in place over PCIe µs |");
        println!("|---|---|---|---|---|---|");
        let staging = DevBuf::alloc(128 * BLOB)?;
        // Pointer table for the staging slots appended after the bench table.
        let stage_ptrs: Vec<u64> = (0..128).map(|i| staging.ptr + (i * BLOB) as u64).collect();
        let d_stage_ptrs = DevBuf::from_slice(&stage_ptrs)?;
        let s = Stream::new()?;
        let (e0, e1, e2) = (Event::new(true)?, Event::new(true)?, Event::new(true)?);
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        for n in [1usize, 4, 16, 64, 128] {
            let (mut tc, mut tk, mut tm) = (Vec::new(), Vec::new(), Vec::new());
            for it in 0..o.iters + 3 {
                // Copy fresh random blobs into staging, then run the kernel on the staging copy.
                let ids = rng.distinct(n, 0, self.host_slots);
                e0.record(&s)?;
                for (j, &id) in ids.iter().enumerate() {
                    let src = unsafe { self.arena.as_ptr().add(id as usize * BLOB) };
                    check(
                        unsafe {
                            sys::cuMemcpyHtoDAsync_v2(
                                staging.ptr + (j * BLOB) as u64,
                                src as *const c_void,
                                BLOB,
                                s.0,
                            )
                        },
                        "copy",
                    )?;
                }
                e1.record(&s)?;
                unsafe {
                    launch(
                        self.q2,
                        (320, n as u32, 1),
                        (256, 1, 1),
                        0,
                        &s,
                        args![
                            d_stage_ptrs.ptr,
                            0u64,
                            self.d_x.ptr,
                            self.d_xsum.ptr,
                            self.d_out.ptr
                        ],
                    )
                }?;
                e2.record(&s)?;
                e2.sync()?;
                let (c, k) = (e1.since(&e0)? as f64 * 1e3, e2.since(&e0)? as f64 * 1e3);
                // The same count of other fresh blobs, read in place over PCIe.
                let ids2 = rng.distinct(n, 0, self.host_slots);
                self.d_ids.write(0, &ids2)?;
                e0.record(&s)?;
                self.q2_launch(n, &s)?;
                e1.record(&s)?;
                e1.sync()?;
                if it >= 3 {
                    tc.push(c);
                    tk.push(k);
                    tm.push(e1.since(&e0)? as f64 * 1e3);
                }
            }
            let (c, k, m) = (summarize(tc), summarize(tk), summarize(tm));
            let gbs = |us: f64| (n * BLOB) as f64 / us / 1e3;
            println!(
                "| {n} | {:.0} [{:.0}–{:.0}] | {:.1} | {:.0} | {:.1} | {:.0} |",
                c.med,
                c.p10,
                c.p90,
                gbs(c.med),
                k.med,
                gbs(k.med),
                m.med
            );
        }

        println!("\n### (b') overlap: copies or mapped reads on stream B while a VRAM expert kernel runs on stream A");
        println!("busy = q2_expert over 256 VRAM blobs (354 MB); side = 64 random host blobs (88 MB). wall from one event before both to the join.");
        println!("| side work | busy alone µs | side alone µs | both µs | busy slowed to µs |");
        println!("|---|---|---|---|---|");
        let (sa, sb) = (Stream::new()?, Stream::new()?);
        let busy_ids: Vec<i32> = (0..256).map(|i| (self.host_slots + i) as i32).collect();
        // ids buffer: [0..256) busy, [256..320) side.
        for side in ["copy", "mapped read"] {
            let (mut ta, mut tb, mut tboth, mut tbusy_in) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for it in 0..o.iters + 3 {
                let mut ids = busy_ids.clone();
                ids.extend(rng.distinct(64, 0, self.host_slots));
                self.d_ids.write(0, &ids)?;
                let side_ids = ids[256..].to_vec();
                let run_busy = |st: &Stream| self.q2_launch_at(0, 256, st);
                let run_side = |st: &Stream| -> Result<()> {
                    if side == "copy" {
                        for (j, &id) in side_ids.iter().enumerate() {
                            let src = unsafe { self.arena.as_ptr().add(id as usize * BLOB) };
                            check(
                                unsafe {
                                    sys::cuMemcpyHtoDAsync_v2(
                                        staging.ptr + ((j % 128) * BLOB) as u64,
                                        src as *const c_void,
                                        BLOB,
                                        st.0,
                                    )
                                },
                                "copy",
                            )?;
                        }
                        Ok(())
                    } else {
                        self.q2_launch_at(256, 64, st)
                    }
                };
                // alone
                e0.record(&sa)?;
                run_busy(&sa)?;
                e1.record(&sa)?;
                e1.sync()?;
                let a = e1.since(&e0)? as f64 * 1e3;
                e0.record(&sb)?;
                run_side(&sb)?;
                e1.record(&sb)?;
                e1.sync()?;
                let b = e1.since(&e0)? as f64 * 1e3;
                // together
                let fresh: Vec<i32> = rng.distinct(64, 0, self.host_slots);
                self.d_ids.write(256 * 4, &fresh)?;
                e0.record(&sa)?;
                sb.wait(&e0)?;
                run_side(&sb)?;
                run_busy(&sa)?;
                e2.record(&sa)?;
                e1.record(&sb)?;
                sa.wait(&e1)?;
                let end = Event::new(true)?;
                end.record(&sa)?;
                end.sync()?;
                if it >= 3 {
                    ta.push(a);
                    tb.push(b);
                    tboth.push(end.since(&e0)? as f64 * 1e3);
                    tbusy_in.push(e2.since(&e0)? as f64 * 1e3);
                }
            }
            println!(
                "| {side} | {:.0} | {:.0} | {:.0} | {:.0} |",
                summarize(ta).med,
                summarize(tb).med,
                summarize(tboth).med,
                summarize(tbusy_in).med
            );
        }
        Ok(())
    }

    /// (e) CPU read bandwidth over the arena, and H2D/D2H copy rates.
    fn dram(&self) -> Result<()> {
        println!(
            "\n## (e) host DRAM read bandwidth over the pinned arena (AVX2 sum), and copy rates"
        );
        let (pcores, ecores) = cpu_sets();
        println!("# P-core first threads: {pcores:?}; E-cores: {ecores:?}");
        let mut all_p: Vec<usize> = Vec::new();
        for c in pcores.iter() {
            all_p.push(*c);
            all_p.push(*c + 1);
        }
        let mut p_and_e = pcores.clone();
        p_and_e.extend(ecores.iter().copied());
        println!("| threads | GB/s median [min–max] of 5 |");
        println!("|---|---|");
        for (name, cpus) in [
            ("1 P-core", vec![pcores[0]]),
            ("8 P-cores (1 thread each)", pcores.clone()),
            ("16 P threads (HT)", all_p),
            ("8 P + 8 E", p_and_e),
        ] {
            let mut v = Vec::new();
            for _ in 0..5 {
                let t = Instant::now();
                let s = par_sum(&self.arena, &cpus);
                v.push(self.arena.len() as f64 / t.elapsed().as_secs_f64() / 1e9);
                std::hint::black_box(s);
            }
            let sm = summarize(v.clone());
            let mn = v.iter().cloned().fold(f64::MAX, f64::min);
            println!("| {name} | {:.1} [{:.1}–{:.1}] |", sm.med, mn, sm.max);
        }

        println!("\n| copy | GB/s median of 5 |");
        println!("|---|---|");
        let n = GB;
        let d = DevBuf::alloc(n)?;
        let pageable = vec![1u8; n];
        for (name, src, dir) in [
            (
                "H2D 1 GiB from pinned arena",
                self.arena.as_ptr() as *const u8,
                0,
            ),
            (
                "D2H 1 GiB to pinned arena",
                self.arena.as_ptr() as *const u8,
                1,
            ),
            ("H2D 1 GiB from pageable Vec", pageable.as_ptr(), 0),
        ] {
            let mut v = Vec::new();
            for _ in 0..5 {
                let t = Instant::now();
                if dir == 0 {
                    check(
                        unsafe { sys::cuMemcpyHtoD_v2(d.ptr, src as *const c_void, n) },
                        "h2d",
                    )?;
                } else {
                    check(
                        unsafe { sys::cuMemcpyDtoH_v2(src as *mut c_void, d.ptr, n) },
                        "d2h",
                    )?;
                }
                v.push(n as f64 / t.elapsed().as_secs_f64() / 1e9);
            }
            println!("| {name} | {:.1} |", summarize(v).med);
        }
        // D2H wrote junk into blob 0..; restore the fill so later benches stay sane.
        fill_range(&self.arena, 0, n.div_ceil(BLOB) + 1);
        Ok(())
    }
}

/// Q2_0-like blobs: random codes, bf16 scales of ~0.01.
fn fill(arena: &HostArena, slots: usize) {
    let threads = 16;
    let per = slots.div_ceil(threads);
    std::thread::scope(|sc| {
        for t in 0..threads {
            let lo = t * per;
            let hi = ((t + 1) * per).min(slots);
            sc.spawn(move || fill_range(arena, lo, hi));
        }
    });
}

fn fill_range(arena: &HostArena, lo: usize, hi: usize) {
    let bytes = unsafe { arena.bytes_mut() };
    for s in lo..hi.min(arena.len() / BLOB) {
        let blob = &mut bytes[s * BLOB..(s + 1) * BLOB];
        let mut r = Rng(0x9e37_79b9 ^ (s as u64 + 1).wrapping_mul(0x100_0000_01b3));
        let (codes, scales) = blob.split_at_mut(1_228_800);
        for c in codes.as_chunks_mut::<8>().0 {
            c.copy_from_slice(&r.next().to_le_bytes());
        }
        for sc in scales.as_chunks_mut::<2>().0 {
            sc.copy_from_slice(&0x3c23u16.to_le_bytes());
        }
    }
}

/// CPU reference for `q2_expert` on one blob.
fn reference(blob: &[u8], x: &[i8], xsum: &[f32]) -> Vec<f32> {
    let group = |gi: usize, xg: usize| -> f32 {
        let c = &blob[gi * 16..gi * 16 + 16];
        let mut s = 0i32;
        for j in 0..4 {
            for sh in 0..4 {
                for b in 0..4 {
                    let code = (c[4 * j + b] >> (2 * sh)) & 3;
                    s += code as i32 * x[xg * 64 + 16 * j + 4 * sh + b] as i32;
                }
            }
        }
        let d = u16::from_le_bytes([blob[1_228_800 + gi * 2], blob[1_228_800 + gi * 2 + 1]]);
        f32::from_bits((d as u32) << 16) * (s as f32 - xsum[xg])
    };
    let mut out = vec![0f32; ROWS];
    for (r, o) in out.iter_mut().enumerate().take(1280) {
        *o = (0..40).map(|g| group(r * 40 + g, g)).sum();
    }
    for r in 0..2560 {
        out[1280 + r] = (0..10).map(|g| group(51_200 + r * 10 + g, g)).sum();
    }
    out
}

/// (first thread of each P-core, E-cores), from sysfs on a hybrid Intel; falls back to
/// "every other CPU" if the hybrid nodes are missing.
fn cpu_sets() -> (Vec<usize>, Vec<usize>) {
    let parse = |s: &str| -> Vec<usize> {
        let mut v = Vec::new();
        for part in s.trim().split(',') {
            if let Some((a, b)) = part.split_once('-') {
                v.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap());
            } else if let Ok(a) = part.parse() {
                v.push(a);
            }
        }
        v
    };
    let p = std::fs::read_to_string("/sys/devices/cpu_core/cpus").map(|s| parse(&s));
    let e = std::fs::read_to_string("/sys/devices/cpu_atom/cpus").map(|s| parse(&s));
    match (p, e) {
        (Ok(p), Ok(e)) => (p.into_iter().step_by(2).collect(), e),
        _ => ((0..8).map(|i| 2 * i).collect(), Vec::new()),
    }
}

fn pin_to(cpu: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
    let _ = cpu;
}

fn par_sum(arena: &HostArena, cpus: &[usize]) -> u64 {
    let n = cpus.len();
    let per = (arena.len() / n) & !4095;
    let base = arena.as_ptr() as usize;
    std::thread::scope(|sc| {
        let hs: Vec<_> = cpus
            .iter()
            .enumerate()
            .map(|(i, &cpu)| {
                sc.spawn(move || {
                    pin_to(cpu);
                    sum_bytes((base + i * per) as *const u8, per)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().unwrap())
            .fold(0, u64::wrapping_add)
    })
}

fn sum_bytes(p: *const u8, n: usize) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { sum_avx2(p, n) };
    }
    let s = unsafe { std::slice::from_raw_parts(p as *const u64, n / 8) };
    s.iter().fold(0u64, |a, &b| a.wrapping_add(b))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(p: *const u8, n: usize) -> u64 {
    use std::arch::x86_64::*;
    let mut a = [_mm256_setzero_si256(); 4];
    let mut q = p as *const __m256i;
    let end = p.add(n & !127) as *const __m256i;
    while q < end {
        a[0] = _mm256_add_epi64(a[0], _mm256_load_si256(q));
        a[1] = _mm256_add_epi64(a[1], _mm256_load_si256(q.add(1)));
        a[2] = _mm256_add_epi64(a[2], _mm256_load_si256(q.add(2)));
        a[3] = _mm256_add_epi64(a[3], _mm256_load_si256(q.add(3)));
        q = q.add(4);
    }
    let s = _mm256_add_epi64(_mm256_add_epi64(a[0], a[1]), _mm256_add_epi64(a[2], a[3]));
    let mut out = [0u64; 4];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, s);
    out.iter().fold(0u64, |x, &y| x.wrapping_add(y))
}

// ---------- (c) graphs ----------

fn graph(gpu: &Gpu) -> Result<()> {
    println!("\n## (c) CUDA graph overhead: K dependent small kernels on one stream");
    println!("wall time host-side (launch to sync), median of 20; per-kernel = wall / K.");
    println!("tiny = 1 block × 32 threads, 1 RMW; medium = 256 × 256 threads, 64K floats RMW.");
    println!("| kernel | K | eager ms | eager µs/kernel | graph ms | graph µs/kernel | fused ms |");
    println!("|---|---|---|---|---|---|---|");
    let m = gpu.module(kernels::BENCH)?;
    let (tiny, medium, fused) = (m.func("tiny")?, m.func("medium")?, m.func("tiny_fused")?);
    let s = Stream::new()?;
    let step = DevBuf::zeroed(4)?;
    for (kname, f) in [("tiny", tiny), ("medium", medium)] {
        for k in [500usize, 1000, 2000] {
            let buf = DevBuf::zeroed((k * 32).max(65536) * 4)?;
            let (grid, block) = if kname == "tiny" { (1, 32) } else { (256, 256) };
            let enqueue = |s: &Stream| -> Result<()> {
                for i in 0..k as i32 {
                    unsafe {
                        launch(
                            f,
                            (grid, 1, 1),
                            (block, 1, 1),
                            0,
                            s,
                            args![buf.ptr, step.ptr, i],
                        )
                    }?;
                }
                Ok(())
            };
            let mut eager = Vec::new();
            for it in 0..23 {
                let t = Instant::now();
                enqueue(&s)?;
                s.sync()?;
                if it >= 3 {
                    eager.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            let g = Graph::capture(&s, enqueue)?;
            let mut gt = Vec::new();
            for it in 0..23 {
                let t = Instant::now();
                g.launch(&s)?;
                s.sync()?;
                if it >= 3 {
                    gt.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            let fused_ms = if kname == "tiny" {
                let mut ft = Vec::new();
                for it in 0..23 {
                    let t = Instant::now();
                    unsafe {
                        launch(
                            fused,
                            (1, 1, 1),
                            (32, 1, 1),
                            0,
                            &s,
                            args![buf.ptr, step.ptr, k as i32],
                        )
                    }?;
                    s.sync()?;
                    if it >= 3 {
                        ft.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                }
                format!("{:.3}", summarize(ft).med)
            } else {
                "—".into()
            };
            let (e, gg) = (summarize(eager).med, summarize(gt).med);
            println!(
                "| {kname} | {k} | {e:.3} | {:.2} | {gg:.3} | {:.2} | {fused_ms} |",
                e * 1e3 / k as f64,
                gg * 1e3 / k as f64
            );
        }
    }

    // Correctness: per-step scalar in device memory, changed between replays.
    let k = 1000usize;
    let buf = DevBuf::zeroed(k * 32 * 4)?;
    let g = Graph::capture(&s, |s| {
        for i in 0..k as i32 {
            unsafe {
                launch(
                    tiny,
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    s,
                    args![buf.ptr, step.ptr, i],
                )
            }?;
        }
        Ok(())
    })?;
    let reps = 20;
    let sum_steps: i32 = (1..=reps).sum();
    let verify = |label: &str| -> Result<bool> {
        let got: Vec<f32> = buf.read(0, k * 32)?;
        let ok = (0..k).all(|i| {
            let want = (sum_steps + reps * i as i32) as f32;
            (0..32).all(|t| got[i * 32 + t] == want)
        });
        println!(
            "graph replay, step scalar in device memory rewritten {label} between {reps} replays of {k} kernels: {}",
            if ok { "correct" } else { "WRONG" }
        );
        Ok(ok)
    };
    // Stream-ordered: the write is enqueued on the graph's stream (pageable source; the
    // driver stages it before returning, so the host value can change right after).
    println!();
    for r in 0..reps {
        step.write_async(0, &[r + 1], &s)?;
        g.launch(&s)?;
    }
    s.sync()?;
    let ok = verify("with cuMemcpyHtoDAsync on the graph's stream")?;
    // Not stream-ordered: a synchronous cuMemcpyHtoD runs on the legacy stream, which a
    // non-blocking stream does not wait for, so it races the queued replays.
    let zero = vec![0f32; k * 32];
    buf.write(0, &zero)?;
    for r in 0..reps {
        step.write(0, &[r + 1])?;
        g.launch(&s)?;
    }
    s.sync()?;
    verify(
        "with a synchronous cuMemcpyHtoD (legacy stream, NOT ordered with the non-blocking stream)",
    )?;
    assert!(ok);
    Ok(())
}

// ---------- cache sizing ----------

/// Size the Flash-Next slot arena the way a real load would: allocate stand-ins for the dense
/// weights (3.5 GB), MTP (1.0 GB) and KV/state (1.0 GB), then let `ExpertCache::new` take the
/// rest minus a 0.7 GB reserve. Host copies alias a 2 GiB arena (sizing only). Then seed all
/// slots and time it, and time one full adaptation's worth of swaps (96).
fn size(gpu: &Gpu) -> Result<()> {
    use tang_moe::{AdaptParams, ExpertCache};
    println!("\n## slot arena sizing for Flash-Next on this card");
    let geo = Geometry::FLASH_NEXT_Q2_0;
    let mut a = HostArena::new(2 * GB, ArenaOptions::default())
        .map_err(|e| tang_moe::gpu::Error(format!("mmap: {e}")))?;
    a.register(&gpu.ctx, &[])?;
    fill(&a, a.len() / BLOB);
    let host_slots = a.len() / BLOB;
    let offsets: Vec<usize> = (0..geo.keys()).map(|k| (k % host_slots) * BLOB).collect();
    let (free0, total) = gpu.mem_info()?;
    let stand_ins = [3_500_000_000usize, 1_000_000_000, 1_000_000_000];
    let _bufs: Vec<DevBuf> = stand_ins
        .iter()
        .map(|&n| DevBuf::zeroed(n))
        .collect::<Result<_>>()?;
    let reserve = 700_000_000;
    let (mut cache, r) = ExpertCache::new(
        gpu,
        geo,
        AdaptParams::default(),
        &a,
        &offsets,
        reserve,
        None,
    )?;
    println!(
        "free at start {:.2} GB of {:.2} GB; after dense+MTP+KV stand-ins (5.5 GB) {:.2} GB; reserve {:.1} GB",
        free0 as f64 / 1e9,
        total as f64 / 1e9,
        r.free_before as f64 / 1e9,
        reserve as f64 / 1e9
    );
    println!(
        "slots: first estimate {} → after touch free {:.2} GB → final {} slots ({:.2} GB, {:.1}% of {} experts); free left {:.2} GB",
        r.slots_first,
        r.free_after_touch as f64 / 1e9,
        r.slots,
        (r.slots * BLOB) as f64 / 1e9,
        100.0 * r.slots as f64 / geo.keys() as f64,
        geo.keys(),
        r.free_final as f64 / 1e9
    );
    let main = Stream::new()?;
    let t = Instant::now();
    cache.seed(0..geo.keys() as u32, &main)?;
    let secs = t.elapsed().as_secs_f64();
    println!(
        "seeding {} slots from pinned host: {:.2} s ({:.1} GB/s)",
        r.slots,
        secs,
        (r.slots * BLOB) as f64 / secs / 1e9
    );
    // One adaptation: make 96 non-resident keys hot.
    let hot: Vec<u32> = (r.slots as u32..r.slots as u32 + 96)
        .flat_map(|k| [k; 4])
        .collect();
    for _ in 0..3 {
        cache.record(&hot);
        cache.between_windows(&main)?;
    }
    cache.record(&hot);
    let t = Instant::now();
    let n = cache.between_windows(&main)?;
    let plan_ms = t.elapsed().as_secs_f64() * 1e3;
    cache.settle(&main)?;
    let secs = t.elapsed().as_secs_f64();
    println!(
        "one adaptation: {n} swaps; plan + publish {plan_ms:.2} ms on the host; copies landed after {:.1} ms ({:.1} GB/s)",
        secs * 1e3,
        (n * BLOB) as f64 / secs / 1e9
    );
    Ok(())
}

// ---------- PCIe diagnosis ----------

/// Where does host→device bandwidth go: copy size, a small region repeated (IOTLB-resident),
/// and registered mmap memory vs driver-allocated pinned memory (cuMemHostAlloc).
fn pcie(gpu: &Gpu) -> Result<()> {
    println!("\n## PCIe diagnosis: H2D copy engine and kernel reads of mapped memory");
    let len = GB;
    let mut a = HostArena::new(len, ArenaOptions::default())
        .map_err(|e| tang_moe::gpu::Error(format!("mmap: {e}")))?;
    a.register(&gpu.ctx, &[])?;
    unsafe { std::ptr::write_bytes(a.as_ptr(), 1, len) };
    let mut hp: *mut c_void = std::ptr::null_mut();
    check(
        unsafe {
            sys::cuMemHostAlloc(
                &mut hp,
                len,
                sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP,
            )
        },
        "cuMemHostAlloc",
    )?;
    unsafe { std::ptr::write_bytes(hp as *mut u8, 1, len) };
    let mut hp_dev: u64 = 0;
    check(
        unsafe { sys::cuMemHostGetDevicePointer_v2(&mut hp_dev, hp, 0) },
        "dev ptr",
    )?;
    let d = DevBuf::alloc(len)?;
    let s = Stream::new()?;
    let (e0, e1) = (Event::new(true)?, Event::new(true)?);
    let h2d = |src: *const u8, chunk: usize, span: usize, total: usize| -> Result<f64> {
        let mut v = Vec::new();
        for _ in 0..5 {
            e0.record(&s)?;
            let mut done = 0;
            let mut off = 0;
            while done < total {
                check(
                    unsafe {
                        sys::cuMemcpyHtoDAsync_v2(
                            d.ptr + off as u64,
                            src.add(off) as *const c_void,
                            chunk,
                            s.0,
                        )
                    },
                    "h2d",
                )?;
                done += chunk;
                off += chunk;
                if off + chunk > span {
                    off = 0;
                }
            }
            e1.record(&s)?;
            e1.sync()?;
            v.push(total as f64 / (e1.since(&e0)? as f64 * 1e-3) / 1e9);
        }
        Ok(summarize(v).med)
    };
    println!("| source | copy size | span touched | GB/s |");
    println!("|---|---|---|---|");
    for (name, src) in [
        ("registered mmap (THP)", a.as_ptr() as *const u8),
        ("cuMemHostAlloc", hp as *const u8),
    ] {
        for (chunk, span) in [
            (64 << 10, 2 << 20),
            (2 << 20, 2 << 20),
            (1_382_400, len),
            (64 << 20, len),
            (len, len),
        ] {
            let g = h2d(src, chunk, span, len)?;
            println!(
                "| {name} | {} KiB | {} MiB | {g:.1} |",
                chunk >> 10,
                span >> 20
            );
        }
    }

    // Kernel reads: one 1.38 MB blob at a time, same blob repeatedly vs walking the GiB.
    let m = gpu.module(kernels::BENCH)?;
    let f = m.func("stream_sum")?;
    let out = DevBuf::zeroed(1024)?;
    println!("\n| source | kernel read pattern | GB/s |");
    println!("|---|---|---|");
    for (name, base) in [
        ("registered mmap (THP)", a.device_ptr(0).unwrap()),
        ("cuMemHostAlloc", hp_dev),
    ] {
        for (pat, n, stride) in [
            ("1 blob, same one each launch", 1usize, 0usize),
            ("64 blobs in one launch, contiguous", 64, BLOB),
        ] {
            let ptrs: Vec<u64> = (0..n).map(|i| base + (i * stride) as u64).collect();
            let dp = DevBuf::from_slice(&ptrs)?;
            let vecs = (BLOB / 16) as i32;
            let mut v = Vec::new();
            for it in 0..13 {
                e0.record(&s)?;
                unsafe {
                    launch(
                        f,
                        (128, n as u32, 1),
                        (256, 1, 1),
                        0,
                        &s,
                        args![dp.ptr, 0u64, vecs, out.ptr],
                    )
                }?;
                e1.record(&s)?;
                e1.sync()?;
                if it >= 3 {
                    v.push((n * BLOB) as f64 / (e1.since(&e0)? as f64 * 1e-3) / 1e9);
                }
            }
            println!("| {name} | {pat} | {:.1} |", summarize(v).med);
        }
    }
    unsafe { sys::cuMemFreeHost(hp) };
    Ok(())
}

// ---------- (d) spin ----------

fn spin(gpu: &Gpu) -> Result<()> {
    println!("\n## (d) device spin-wait on a mapped flag");
    let mut a = HostArena::new(2 << 20, ArenaOptions::default())
        .map_err(|e| tang_moe::gpu::Error(format!("mmap: {e}")))?;
    a.register(&gpu.ctx, &[])?;
    let (flag, ack) = unsafe {
        let p = a.as_ptr();
        std::ptr::write_bytes(p, 0, 4096);
        (
            &*(p as *const AtomicU32),
            &*(p.add(128) as *const AtomicU32),
        )
    };
    let (dflag, dack) = (a.device_ptr(0).unwrap(), a.device_ptr(128).unwrap());
    let m = gpu.module(kernels::BENCH)?;
    let (pp, wait) = (m.func("pingpong")?, m.func("wait_flag")?);
    let s = Stream::new()?;
    pin_to(cpu_sets().0[1]);

    let rounds = 20_000u32;
    unsafe {
        launch(
            pp,
            (1, 1, 1),
            (1, 1, 1),
            0,
            &s,
            args![dflag, dack, rounds as i32],
        )
    }?;
    let mut rtt = Vec::with_capacity(rounds as usize);
    for r in 1..=rounds {
        let t = Instant::now();
        flag.store(r, Ordering::Release);
        while ack.load(Ordering::Acquire) < r {
            std::hint::spin_loop();
        }
        rtt.push(t.elapsed().as_secs_f64() * 1e6);
    }
    s.sync()?;
    let sm = summarize(rtt);
    println!("| measurement | median µs | p90 | p99 | max |");
    println!("|---|---|---|---|---|");
    println!(
        "| round trip: host writes flag → spinning kernel sees it, writes ack → host sees ack ({rounds} rounds) | {:.2} | {:.2} | {:.2} | {:.1} |",
        sm.med, sm.p90, sm.p99, sm.max
    );

    flag.store(0, Ordering::Release);
    let mut lat = Vec::new();
    for t in 1..=300u32 {
        unsafe { launch(wait, (1, 1, 1), (1, 1, 1), 0, &s, args![dflag, t]) }?;
        let t_spin = Instant::now();
        while t_spin.elapsed().as_micros() < 200 {
            std::hint::spin_loop();
        }
        let t0 = Instant::now();
        flag.store(t, Ordering::Release);
        while !s.idle() {
            std::hint::spin_loop();
        }
        lat.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let sm = summarize(lat);
    println!(
        "| host writes flag → waiting kernel exits → cuStreamQuery reports idle (300 trials) | {:.2} | {:.2} | {:.2} | {:.1} |",
        sm.med, sm.p90, sm.p99, sm.max
    );
    Ok(())
}
