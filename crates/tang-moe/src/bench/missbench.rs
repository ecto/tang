//! CPU miss path benchmarks: `cpu` (expert kernel throughput) and `missw` (a synthetic decode
//! window through the doorbell). Included by tang-moe-bench.

use std::time::{Duration, Instant};

use tang_moe::contract::{
    expert_ref, f32_to_f16, quantize_act, ExpertBlob, MoePlan, QAct, FF, HIDDEN, TOPK,
};
use tang_moe::doorbell::{kernels_src, seq, serve_layer, LayerTimes, Mailbox, Mb};
use tang_moe::gpu::{launch, DevBuf, Error, Gpu, Graph, Result, Stream};
use tang_moe::miss::{build_plan, MissExec, MissJob};
use tang_moe::pool::{Pool, Topology};
use tang_moe::q2cpu::Isa;
use tang_moe::{args, ArenaOptions, HostArena};

use super::{summarize, Rng, GB};

const BLOB: usize = ExpertBlob::BYTES;

/// Resident-mode host arena for Flash-Next on mew: the 12,235 experts not in the 12,341 VRAM
/// slots, ≈ 16.9 GB.
pub const HOST_SLOTS: usize = 24_576 - 12_341;

pub struct Arena {
    pub a: HostArena,
    pub slots: usize,
}

impl Arena {
    pub fn new(gpu: Option<&Gpu>, slots: usize) -> Result<Arena> {
        let mut a = HostArena::new(slots * BLOB, ArenaOptions::default())
            .map_err(|e| Error(format!("mmap: {e}")))?;
        let t = Instant::now();
        if let Some(g) = gpu {
            a.register(&g.ctx, &[4 * GB, GB])?;
        }
        let reg = t.elapsed().as_secs_f64();
        let t = Instant::now();
        fill_q2(&a, slots);
        println!(
            "# arena: {slots} blobs, {:.1} GB, {:?}{}, {:.0}% huge pages; filled in {:.1} s",
            (slots * BLOB) as f64 / 1e9,
            a.backing(),
            if gpu.is_some() {
                format!(", registered in {reg:.1} s")
            } else {
                String::new()
            },
            100.0 * a.huge_bytes().unwrap_or(0) as f64 / a.len() as f64,
            t.elapsed().as_secs_f64()
        );
        Ok(Arena { a, slots })
    }

    pub fn blob(&self, slot: usize) -> &[u8] {
        assert!(slot < self.slots);
        unsafe { std::slice::from_raw_parts(self.a.as_ptr().add(slot * BLOB), BLOB) }
    }
}

/// Valid repacked Q2_0 blobs: random codes, fp16 scales in [0.01, 0.03).
fn fill_q2(a: &HostArena, slots: usize) {
    let threads = 16;
    let per = slots.div_ceil(threads);
    let base = a.as_ptr() as usize;
    std::thread::scope(|sc| {
        for t in 0..threads {
            sc.spawn(move || {
                for s in t * per..((t + 1) * per).min(slots) {
                    let b = unsafe {
                        std::slice::from_raw_parts_mut((base + s * BLOB) as *mut u8, BLOB)
                    };
                    let mut r = Rng(0x9e37_79b9 ^ (s as u64 + 1).wrapping_mul(0x100_0000_01b3));
                    for (off, n, k) in [
                        (ExpertBlob::GATE, FF, HIDDEN),
                        (ExpertBlob::UP, FF, HIDDEN),
                        (ExpertBlob::DOWN, HIDDEN, FF),
                    ] {
                        let codes = n * k / 4;
                        for c in b[off..off + codes].as_chunks_mut::<8>().0 {
                            c.copy_from_slice(&r.next().to_le_bytes());
                        }
                        for sc in b[off + codes..off + codes + n * k / 32]
                            .as_chunks_mut::<2>()
                            .0
                        {
                            let d = 0.01 + (r.next() % 1000) as f32 * 2e-5;
                            sc.copy_from_slice(&f32_to_f16(d).to_le_bytes());
                        }
                    }
                }
            });
        }
    });
}

fn acts(rng: &mut Rng, t: usize) -> Vec<u32> {
    let x: Vec<f32> = (0..t * HIDDEN)
        .map(|_| ((rng.next() % 20001) as f32 / 10000.0 - 1.0) * 2.0)
        .collect();
    quantize_act(&x, t, HIDDEN)
}

fn pools() -> Vec<(&'static str, Vec<usize>)> {
    let topo = Topology::detect();
    let mut pe = topo.pcores.clone();
    pe.extend(&topo.ecores);
    vec![
        ("1 P-core", vec![topo.pcores[0]]),
        ("8 P-cores", topo.pcores.clone()),
        ("8 P + 8 E", pe),
    ]
}

/// (a) Expert kernel throughput and parity.
pub fn cpu(gpu: Option<&Gpu>, iters: usize) -> Result<()> {
    println!("\n## (a) CPU expert kernels: Q2_0 blobs from a resident-mode arena");
    let arena = Arena::new(gpu, HOST_SLOTS)?;
    let topo = Topology::detect();
    println!(
        "# P-cores {:?}, E-cores {:?}; best ISA {:?}",
        topo.pcores,
        topo.ecores,
        Isa::detect()
    );
    let mut rng = Rng(0x5eed);

    // Parity against the scalar reference (the spec).
    let t = 2;
    let xq = acts(&mut rng, t);
    let blobs: Vec<usize> = (0..4)
        .map(|_| (rng.next() as usize) % arena.slots)
        .collect();
    let toks: Vec<(usize, usize)> = (0..t).map(|tt| (tt, tt * TOPK)).collect();
    let toks_per: Vec<Vec<(usize, usize)>> = (0..4)
        .map(|e| toks.iter().map(|&(tt, d)| (tt, d + e)).collect())
        .collect();
    for isa in Isa::available() {
        let mut ex = MissExec::new(Pool::new(&topo.pcores, Duration::from_millis(20)), isa);
        let jobs: Vec<MissJob> = blobs
            .iter()
            .zip(&toks_per)
            .map(|(&b, tl)| MissJob {
                blob: arena.blob(b),
                toks: tl,
            })
            .collect();
        let mut out = vec![0f32; MoePlan::PARTS_ROWS * HIDDEN];
        unsafe { ex.run(&xq, t, &jobs, out.as_mut_ptr()) };
        let (mut worst, mut off) = (0f32, 0usize);
        for (e, &b) in blobs.iter().enumerate() {
            for (tt, dst) in &toks_per[e] {
                let want = expert_ref(arena.blob(b), &xq, t, *tt);
                let got = &out[dst * HIDDEN..(dst + 1) * HIDDEN];
                let mx = want.iter().fold(0f32, |a, v| a.max(v.abs()));
                for (g, w) in got.iter().zip(&want) {
                    let r = (g - w).abs() / mx;
                    worst = worst.max(r);
                    if r > 1e-5 {
                        off += 1;
                    }
                }
            }
        }
        println!(
            "# parity {isa:?} vs scalar spec, 4 experts × 2 tokens × 2560 outputs: max |Δ|/max|y| = {worst:.2e}; outputs beyond 1e-5: {off}"
        );
    }

    println!("\nGB/s = expert weight bytes (1.3824 MB each) / wall time of `MissExec::run` (both phases + quantize), fresh random blobs every call. Median [p10–p90] of {iters}.");
    println!("| threads | ISA | experts × tokens each | µs | GB/s |");
    println!("|---|---|---|---|---|");
    for (name, cpus) in pools() {
        let isas: Vec<Isa> = if cpus.len() == 1 {
            Isa::available()
                .into_iter()
                .filter(|&i| i != Isa::Lane)
                .collect()
        } else {
            vec![Isa::detect()]
        };
        for isa in isas {
            let mut ex = MissExec::new(Pool::new(&cpus, Duration::from_millis(20)), isa);
            for (ne, t) in [(16, 1), (16, 2), (16, 4)] {
                let xq = acts(&mut rng, t);
                let toks: Vec<Vec<(usize, usize)>> = (0..ne)
                    .map(|e| {
                        (0..t)
                            .map(|tt| (tt, (tt * TOPK + e) % MoePlan::SHARED_ROW))
                            .collect()
                    })
                    .collect();
                let mut out = vec![0f32; MoePlan::PARTS_ROWS * HIDDEN];
                let mut v = Vec::new();
                for it in 0..iters + 5 {
                    let ids: Vec<usize> = (0..ne)
                        .map(|_| (rng.next() as usize) % arena.slots)
                        .collect();
                    let jobs: Vec<MissJob> = ids
                        .iter()
                        .zip(&toks)
                        .map(|(&b, tl)| MissJob {
                            blob: arena.blob(b),
                            toks: tl,
                        })
                        .collect();
                    let t0 = Instant::now();
                    unsafe { ex.run(&xq, t, &jobs, out.as_mut_ptr()) };
                    if it >= 5 {
                        v.push(t0.elapsed().as_secs_f64() * 1e6);
                    }
                }
                let s = summarize(v);
                let gbs = |us: f64| (ne * BLOB) as f64 / us / 1e3;
                println!(
                    "| {name} | {isa:?} | {ne} × {t} | {:.0} | {:.1} [{:.1}–{:.1}] |",
                    s.med,
                    gbs(s.med),
                    gbs(s.p90),
                    gbs(s.p10)
                );
            }
        }
    }

    println!("\nPer-layer latency at realistic sizes (8 P-cores, best ISA): M distinct missed experts, each serving 1 token; fresh blobs each call. Median [p10–p90] of {iters}.");
    println!("| M | µs | phase A µs | quantize µs | phase B µs | GB/s |");
    println!("|---|---|---|---|---|---|");
    let mut ex = MissExec::new(
        Pool::new(&topo.pcores, Duration::from_millis(20)),
        Isa::detect(),
    );
    let xq = acts(&mut rng, 3);
    // Warm the pool and caches first: the first few dozen calls after creating a pool run
    // ~3x slower (measured: M = 1 at 120 µs cold vs 33 µs warm).
    {
        let toks = [vec![(0usize, 0usize)], vec![(1, TOPK)]];
        let mut out = vec![0f32; MoePlan::PARTS_ROWS * HIDDEN];
        for _ in 0..100 {
            let jobs: Vec<MissJob> = toks
                .iter()
                .map(|tl| MissJob {
                    blob: arena.blob((rng.next() as usize) % arena.slots),
                    toks: tl,
                })
                .collect();
            unsafe { ex.run(&xq, 3, &jobs, out.as_mut_ptr()) };
        }
    }
    for m in [1usize, 2, 3, 4, 8] {
        let toks: Vec<Vec<(usize, usize)>> = (0..m)
            .map(|e| vec![(e % 3, (e % 3) * TOPK + e / 3)])
            .collect();
        let mut out = vec![0f32; MoePlan::PARTS_ROWS * HIDDEN];
        let (mut v, mut a, mut q, mut b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for it in 0..iters + 5 {
            let ids: Vec<usize> = (0..m)
                .map(|_| (rng.next() as usize) % arena.slots)
                .collect();
            let jobs: Vec<MissJob> = ids
                .iter()
                .zip(&toks)
                .map(|(&bb, tl)| MissJob {
                    blob: arena.blob(bb),
                    toks: tl,
                })
                .collect();
            let t0 = Instant::now();
            let tm = unsafe { ex.run(&xq, 3, &jobs, out.as_mut_ptr()) };
            if it >= 5 {
                v.push(t0.elapsed().as_secs_f64() * 1e6);
                a.push(tm.phase_a_us);
                q.push(tm.quant_us);
                b.push(tm.phase_b_us);
            }
        }
        let s = summarize(v);
        println!(
            "| {m} | {:.0} [{:.0}–{:.0}] | {:.0} | {:.1} | {:.0} | {:.1} |",
            s.med,
            s.p10,
            s.p90,
            summarize(a).med,
            summarize(q).med,
            summarize(b).med,
            (m * BLOB) as f64 / s.med / 1e3
        );
    }
    Ok(())
}

/// (b) A synthetic window: 48 layers in one graph, GPU stand-in + doorbell + CPU misses.
pub fn missw(gpu: &Gpu, iters: usize) -> Result<()> {
    const LAYERS: usize = 48;
    const T: usize = 3;
    const D: usize = 24; // distinct experts per layer per window (T = 3, top-10)
    const VSLOTS: usize = 1024;
    println!("\n## (b) synthetic window: 48 layers, T = {T}, {D} distinct experts per layer, M of them missed");
    let arena = Arena::new(Some(gpu), HOST_SLOTS)?;
    let topo = Topology::detect();
    let mb = Mailbox::new(gpu)?;
    let m = gpu.module(&kernels_src())?;
    let stamp = m.func("stamp")?;
    // TANG_MOE_STAMPS=1 adds globaltimer stamps around each step (7 extra tiny kernels per
    // layer, ~0.3 ms per window) and prints a per-step GPU timeline.
    let stamping = std::env::var("TANG_MOE_STAMPS").is_ok();
    let stamps = DevBuf::zeroed(LAYERS * 8 * 8)?;
    let (publish, wait, copy_rows, dense, hits) = (
        m.func("db_publish")?,
        m.func("db_wait")?,
        m.func("db_copy_rows")?,
        m.func("dense_read")?,
        m.func("plan_hits")?,
    );
    let vram = DevBuf::zeroed(VSLOTS * BLOB)?;
    let dense_bytes: usize = 128 << 20;
    let dense_buf = DevBuf::zeroed(dense_bytes)?;
    let parts = DevBuf::zeroed(MoePlan::PARTS_ROWS * HIDDEN * 4)?;
    let sink = DevBuf::zeroed(64)?;
    let win = DevBuf::zeroed(4)?;
    let lx = QAct { m: T, k: HIDDEN };
    let xq_words = lx.words();
    let d_ids = DevBuf::zeroed(LAYERS * T * TOPK * 4)?;
    let d_xq = DevBuf::zeroed(LAYERS * xq_words * 4)?;
    let d_plans = DevBuf::zeroed(LAYERS * MoePlan::WORDS * 4)?;
    let mut rng = Rng(0xfeed_f00d);
    let mut xq_all = Vec::with_capacity(LAYERS * xq_words);
    for _ in 0..LAYERS {
        xq_all.extend(acts(&mut rng, T));
    }
    d_xq.write(0, &xq_all)?;
    let s = Stream::new()?;

    let vram_addr = |l: usize, e: u32| vram.ptr + (((l * 256 + e as usize) % VSLOTS) * BLOB) as u64;
    let host_slot = |l: usize, e: u32| (l * 256 + e as usize - 256) % arena.slots;
    let dense_vecs = (dense_bytes / 16) as i64;

    // Per-layer routing: D candidates, M from the non-resident half (256..512), shuffled; token
    // t takes candidates t·10 .. t·10+9 (mod D).
    let route = |rng: &mut Rng, misses: usize| -> Vec<u32> {
        let mut ids = Vec::with_capacity(LAYERS * T * TOPK);
        for _ in 0..LAYERS {
            let mut cand: Vec<u32> = Vec::with_capacity(D);
            while cand.len() < D - misses {
                let e = (rng.next() % 256) as u32;
                if !cand.contains(&e) {
                    cand.push(e);
                }
            }
            while cand.len() < D {
                let e = 256 + (rng.next() % 256) as u32;
                if !cand.contains(&e) {
                    cand.push(e);
                }
            }
            for i in (1..D).rev() {
                cand.swap(i, (rng.next() as usize) % (i + 1));
            }
            for tt in 0..T {
                for j in 0..TOPK {
                    ids.push(cand[(tt * TOPK + j) % D]);
                }
            }
        }
        ids
    };

    // Graph without the doorbell: dense + hits from a plan already in VRAM.
    let g_plain = Graph::capture(&s, |s| {
        for l in 0..LAYERS {
            unsafe {
                launch(
                    dense,
                    (82 * 4, 1, 1),
                    (256, 1, 1),
                    0,
                    s,
                    args![dense_buf.ptr, dense_vecs, sink.ptr],
                )?;
                let plan = d_plans.ptr + (l * MoePlan::WORDS * 4) as u64;
                launch(
                    hits,
                    (32, MoePlan::CAP as u32, 1),
                    (256, 1, 1),
                    0,
                    s,
                    args![plan, sink.ptr],
                )?;
            }
        }
        Ok(())
    })?;
    // Graph with the doorbell.
    let mbd = mb.device();
    let d_plan = DevBuf::zeroed(MoePlan::WORDS * 4)?;
    let d_list = DevBuf::zeroed((1 + MoePlan::CAP) * 4)?;
    let (plan_src, plan_n) = (Mb::PLAN as i32, MoePlan::WORDS as i32);
    let (list_src, list_n) = (Mb::CPU_ROWS as i32, (1 + MoePlan::CAP) as i32);
    let g_db = Graph::capture(&s, |s| {
        for l in 0..LAYERS {
            let (li, ti, xw) = (l as i32, T as i32, xq_words as i32);
            let ids_l = d_ids.ptr + (l * T * TOPK * 4) as u64;
            let xq_l = d_xq.ptr + (l * xq_words * 4) as u64;
            let (fa, fb) = (Mb::FLAG_A as i32, Mb::FLAG_B as i32);
            let st = |i: i32| -> Result<()> {
                if !stamping {
                    return Ok(());
                }
                unsafe {
                    launch(
                        stamp,
                        (1, 1, 1),
                        (1, 1, 1),
                        0,
                        s,
                        args![stamps.ptr, (l as i32) * 8 + i],
                    )
                }
            };
            st(0)?;
            unsafe {
                launch(
                    dense,
                    (82 * 4, 1, 1),
                    (256, 1, 1),
                    0,
                    s,
                    args![dense_buf.ptr, dense_vecs, sink.ptr],
                )?;
            }
            st(1)?;
            unsafe {
                launch(
                    publish,
                    (1, 1, 1),
                    (256, 1, 1),
                    0,
                    s,
                    args![mbd, ids_l, xq_l, xw, win.ptr, li, ti],
                )?;
            }
            st(2)?;
            unsafe {
                launch(
                    wait,
                    (1, 1, 1),
                    (256, 1, 1),
                    0,
                    s,
                    args![mbd, fa, win.ptr, li, plan_src, plan_n, d_plan.ptr],
                )?;
            }
            st(3)?;
            unsafe {
                launch(
                    hits,
                    (32, MoePlan::CAP as u32, 1),
                    (256, 1, 1),
                    0,
                    s,
                    args![d_plan.ptr, sink.ptr],
                )?;
            }
            st(4)?;
            unsafe {
                launch(
                    wait,
                    (1, 1, 1),
                    (256, 1, 1),
                    0,
                    s,
                    args![mbd, fb, win.ptr, li, list_src, list_n, d_list.ptr],
                )?;
            }
            st(5)?;
            unsafe {
                launch(
                    copy_rows,
                    (MoePlan::CAP as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    s,
                    args![mbd, d_list.ptr, parts.ptr],
                )?;
            }
            st(6)?;
        }
        Ok(())
    })?;

    let mut exec = MissExec::new(
        Pool::new(&topo.pcores, Duration::from_millis(20)),
        Isa::detect(),
    );
    let mut wseq: u32 = 0;
    println!("GPU per layer: dense stand-in (stream 128 MiB of VRAM) + hits stand-in (read each resident group's 1.38 MB blob from VRAM). Host serves with 8 P-cores ({:?}). Window = host wall, graph launch → stream sync. Median [p10–p90] of {iters} windows; CPU = Σ over layers of FLAG_A → FLAG_B.", Isa::detect());
    println!("| variant | M/layer | misses/window | window ms | CPU ms | exposed ms (vs M=0) | hidden | plan+handoff µs/layer |");
    println!("|---|---|---|---|---|---|---|---|");

    // Plain graph baseline (plans built by the host, misses excluded).
    let mut plain = Vec::new();
    for it in 0..iters + 3 {
        let ids = route(&mut rng, 0);
        let mut plans = vec![0u32; LAYERS * MoePlan::WORDS];
        for l in 0..LAYERS {
            let addr = |e: u32| if e < 256 { vram_addr(l, e) } else { 0 };
            build_plan(
                &ids[l * T * TOPK..],
                T,
                addr,
                0,
                &mut plans[l * MoePlan::WORDS..],
            );
        }
        d_plans.write(0, &plans)?;
        let t0 = Instant::now();
        g_plain.launch(&s)?;
        s.sync()?;
        if it >= 3 {
            plain.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    let base_plain = summarize(plain);
    println!(
        "| GPU work only, no doorbell | 0 | 0 | {:.2} [{:.2}–{:.2}] | — | — | — | — |",
        base_plain.med, base_plain.p10, base_plain.p90
    );

    let mut base_db = None;
    for misses in [0usize, 1, 2, 3, 4, 8] {
        let (mut wt, mut ct, mut ht) = (Vec::new(), Vec::new(), Vec::new());
        let mut parity_done = false;
        for it in 0..iters + 3 {
            let ids = route(&mut rng, misses);
            d_ids.write_async(0, &ids, &s)?;
            wseq = wseq.wrapping_add(1);
            win.write_async(0, &[wseq], &s)?;
            s.sync()?;
            let t0 = Instant::now();
            g_db.launch(&s)?;
            let mut lts: Vec<LayerTimes> = Vec::with_capacity(LAYERS);
            for l in 0..LAYERS {
                let addr = |e: u32| if e < 256 { vram_addr(l, e) } else { 0 };
                let host = |e: u32| arena.blob(host_slot(l, e));
                let lt = serve_layer(&mb, seq(wseq, l), &s, &mut exec, &addr, &host, 0);
                match lt {
                    Ok(lt) => lts.push(lt),
                    Err(e) => {
                        mb.abort();
                        let _ = s.sync();
                        return Err(e);
                    }
                }
                if l == LAYERS - 1 && misses > 0 && !parity_done {
                    // Check the rows the GPU copied for this layer against the reference.
                    s.sync()?;
                    parity_done = true;
                    let (t, ids_l, xq) = mb.request();
                    let mut plan = vec![0u32; MoePlan::WORDS];
                    let missed = build_plan(ids_l, t, addr, 0, &mut plan);
                    let mut worst = 0f32;
                    for mm in &missed {
                        for &(tok, dst) in &mm.toks {
                            let want = expert_ref(host(mm.expert), xq, t, tok);
                            let got: Vec<f32> = parts.read(dst * HIDDEN * 4, HIDDEN)?;
                            let mx = want.iter().fold(0f32, |a, v| a.max(v.abs()));
                            for (g, w) in got.iter().zip(&want) {
                                worst = worst.max((g - w).abs() / mx);
                            }
                        }
                    }
                    if it == 0 {
                        println!("# M={misses}: rows the GPU received for layer 47 vs scalar spec: max |Δ|/max|y| = {worst:.2e}");
                    }
                }
            }
            s.sync()?;
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            if it >= 3 {
                wt.push(wall);
                ct.push(lts.iter().map(|l| l.cpu_us).sum::<f64>() / 1e3);
                ht.push(lts.iter().map(|l| l.plan_us).sum::<f64>() / LAYERS as f64);
            }
        }
        if stamping {
            let v: Vec<u64> = stamps.read(0, LAYERS * 8)?;
            for l in [0usize, 1, 2, 24, 47] {
                let b = &v[l * 8..l * 8 + 7];
                let d: Vec<f64> = (1..7)
                    .map(|i| (b[i] as f64 - b[i - 1] as f64) / 1e3)
                    .collect();
                println!("# stamps M={misses} layer {l}: dense {:.1} publish {:.1} waitA {:.1} hits {:.1} waitB {:.1} copy {:.1} µs", d[0], d[1], d[2], d[3], d[4], d[5]);
            }
        }
        let (w, c, h) = (summarize(wt), summarize(ct), summarize(ht));
        let (exposed, hidden) = match base_db {
            None => {
                base_db = Some(w.med);
                ("—".to_string(), "—".to_string())
            }
            Some(b) => {
                let ex = w.med - b;
                (
                    format!("{ex:.2}"),
                    format!("{:.0}%", (100.0 * (c.med - ex) / c.med).clamp(0.0, 100.0)),
                )
            }
        };
        println!(
            "| doorbell | {misses} | {} | {:.2} [{:.2}–{:.2}] | {:.2} | {exposed} | {hidden} | {:.1} |",
            misses * LAYERS,
            w.med,
            w.p10,
            w.p90,
            c.med,
            h.med
        );
    }
    println!(
        "\nhandoff cost: doorbell with no misses − GPU work only = {:.2} ms per window ({:.1} µs per layer)",
        base_db.unwrap() - base_plain.med,
        (base_db.unwrap() - base_plain.med) * 1e3 / LAYERS as f64
    );
    Ok(())
}
