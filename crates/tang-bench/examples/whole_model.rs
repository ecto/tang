//! Whole-model compilation spike: `layers` SmolLM2-135M-like blocks, an
//! MSE loss, the full backward and AdamW on every projection weight as
//! one `TensorGraph`, one program, one submission per step.
//!
//! Reports kernels per step, cross-layer fusion (stacked forward vs
//! `layers ×` one block), memory (one buffer per node vs liveness-planned
//! slots vs planned + in-place updates) and step time.
//!
//! Usage: `whole_model [layers] [seq] [wgpu|cuda|stats]`. `stats` builds
//! and plans without a GPU. `cuda` needs `--features cuda`.
use std::time::Instant;

use tang_bench::ir_block::{compiled_block, BlockDims};
use tang_bench::ir_model::{
    compile, compile_with_barriers, donations, program_stats, train_step, AdamW, TrainStep,
};
use tang_compile::tensor::memplan::MemoryPlan;
use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{TensorGraph, TensorId};

fn mb(b: usize) -> f64 {
    b as f64 / (1 << 20) as f64
}

fn dims(seq: usize) -> BlockDims {
    BlockDims {
        seq,
        kv_len: seq,
        rope: true,
        bias: false,
        d_model: 576,
        n_heads: 9,
        n_kv_heads: 3,
        head_dim: 64,
        ff_dim: 1536,
    }
}

/// Forward kernels: `layers` stacked blocks vs one block.
fn fusion_report(seq: usize, layers: usize) {
    let d = dims(seq);
    let (g1, y1) = compiled_block(&d);
    let k1 = g1.to_wgsl(&[y1]).kernels.len();
    let step = train_step(&d, layers, AdamW::default());
    let h = step.hidden;
    let (gs, ys) = compile(&step.graph, &[h]);
    let ks = gs.to_wgsl(&ys).kernels.len();
    let (gb, yb) = compile_with_barriers(&step.graph, &[h], &step.boundaries);
    let kb = gb.to_wgsl(&yb).kernels.len();
    println!("forward kernels with block outputs materialized: {kb}");
    println!(
        "forward kernels: one block {k1}, {layers} stacked {ks} (vs {} = {layers} × {k1}): {} saved across block boundaries",
        layers * k1,
        (layers * k1) as i64 - ks as i64
    );
}

struct Built {
    step: TrainStep,
    g: TensorGraph,
    outs: Vec<TensorId>,
    prog: WgslProgram,
    naive: MemoryPlan,
    planned: MemoryPlan,
    inplace: MemoryPlan,
}

fn build(seq: usize, layers: usize, cuda: bool) -> Built {
    let t = Instant::now();
    let step = train_step(&dims(seq), layers, AdamW::default());
    let outs = step.outputs();
    let t_graph = t.elapsed();
    let t = Instant::now();
    let barriers = if std::env::var("NOBARRIER").is_ok() {
        vec![]
    } else {
        step.boundaries.clone()
    };
    let (g, couts) = compile_with_barriers(&step.graph, &outs, &barriers);
    let t_passes = t.elapsed();
    let t = Instant::now();
    let prog = if cuda {
        #[cfg(feature = "cuda")]
        {
            g.to_cuda_with(&couts, true)
        }
        #[cfg(not(feature = "cuda"))]
        {
            unreachable!()
        }
    } else {
        g.to_wgsl(&couts)
    };
    let t_lower = t.elapsed();
    let t = Instant::now();
    let naive = MemoryPlan::naive(&prog);
    let planned = g.plan_memory(&prog, &couts, &[]);
    let inplace = g.plan_memory(&prog, &couts, &donations(&g, &step, &couts));
    let t_plan = t.elapsed();
    let s = program_stats(&prog);
    if let Some(k) = prog.kernels.iter().max_by_key(|k| k.reads.len()) {
        let c = g.comp_at(k.node.index()).unwrap();
        println!(
            "  widest kernel: node {:?} reads {} buffers (shape {:?}, red {:?}, {} args)",
            k.node,
            k.reads.len(),
            g.shape(k.node),
            c.red,
            c.args.len()
        );
        if std::env::var("WIDE").is_ok() {
            for a in &c.args {
                let s = a.src.index();
                eprintln!(
                    "    arg {:?} shape {:?} red {:?} readers-of-it {} forms {:?}",
                    a.src,
                    g.shape(a.src),
                    g.comp_at(s).map(|c| c.red.clone()),
                    prog.kernels
                        .iter()
                        .filter(|k| k.reads.contains(&a.src))
                        .count(),
                    a.view
                        .forms
                        .iter()
                        .map(|f| (f.coefs.clone(), f.offset))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    println!(
        "layers {layers} seq {seq}: {} graph nodes -> {} compiled; {} kernels ({} distinct sources, {} up to renaming), {} buffers",
        step.graph.nodes().len(),
        g.nodes().len(),
        s.kernels,
        s.distinct_sources,
        s.unique_kernels,
        s.buffers
    );
    println!(
        "  build {:.2?} passes {:.2?} lower {:.2?} plan {:.2?}",
        t_graph, t_passes, t_lower, t_plan
    );
    let p = naive.stats.persistent_bytes;
    println!(
        "  memory: persistent {:.1} MB (params, m, v, data); intermediates naive {:.1} MB, planned {:.1} MB ({} slots), planned+in-place {:.1} MB ({} in-place updates, live peak {:.1} MB)",
        mb(p),
        mb(naive.stats.planned_bytes),
        mb(planned.stats.planned_bytes),
        planned.slot_words.len() - planned.slot_input.iter().filter(|s| s.is_some()).count(),
        mb(inplace.stats.planned_bytes),
        inplace.inplace.len(),
        mb(planned.stats.live_peak_bytes),
    );
    println!(
        "  peak total: naive {:.1} MB, planned {:.1} MB, planned+in-place {:.1} MB",
        mb(naive.total_bytes()),
        mb(planned.total_bytes()),
        mb(inplace.total_bytes()),
    );
    Built {
        step,
        g,
        outs: couts,
        prog,
        naive,
        planned,
        inplace,
    }
}

fn clone_prog(p: &WgslProgram) -> WgslProgram {
    use tang_compile::tensor::wgsl::{BufferSpec, WgslKernel};
    WgslProgram {
        kernels: p
            .kernels
            .iter()
            .map(|k| WgslKernel {
                source: k.source.clone(),
                node: k.node,
                reads: k.reads.clone(),
                points: k.points,
                count: k.count,
                workgroup_size: k.workgroup_size,
            })
            .collect(),
        buffers: p
            .buffers
            .iter()
            .map(|b| BufferSpec {
                node: b.node,
                len: b.len,
                format: b.format,
                input: b.input,
            })
            .collect(),
        aliases: p.aliases.clone(),
    }
}

fn clone_plan(p: &MemoryPlan) -> MemoryPlan {
    MemoryPlan {
        order: p.order.clone(),
        slot: p.slot.clone(),
        slot_words: p.slot_words.clone(),
        slot_input: p.slot_input.clone(),
        inplace: p.inplace.clone(),
        stats: p.stats,
    }
}

fn run_wgpu(b: &Built, layers: usize, seq: usize) {
    use tang_bench::ir_model::PlannedExecutable;
    use tang_gpu::{GpuDevice, KernelCache};
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
    let time = |exe: &PlannedExecutable, cache: &mut KernelCache| {
        let mut best = f64::INFINITY;
        exe.run(&device, cache);
        let _ = exe.read(&device, b.outs[0]);
        for _ in 0..3 {
            let t = Instant::now();
            for _ in 0..3 {
                exe.run(&device, cache);
            }
            let _ = exe.read(&device, b.outs[0]);
            best = best.min(t.elapsed().as_secs_f64() * 1e3 / 3.0);
        }
        best
    };
    let t = Instant::now();
    let naive = PlannedExecutable::new(
        &device,
        clone_prog(&b.prog),
        clone_plan(&b.naive),
        &b.step.inputs,
    );
    naive.run(&device, &mut cache);
    let loss_n = naive.read(&device, b.outs[0]);
    let p_n = naive.read(&device, b.outs[1]);
    println!("  wgpu first run (shader compiles) {:.2?}", t.elapsed());
    let planned = PlannedExecutable::new(
        &device,
        clone_prog(&b.prog),
        clone_plan(&b.planned),
        &b.step.inputs,
    );
    planned.run(&device, &mut cache);
    let loss_p = planned.read(&device, b.outs[0]);
    let p_p = planned.read(&device, b.outs[1]);
    let dp = p_n
        .iter()
        .zip(&p_p)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  loss naive {} planned {} (max |Δp'| {dp:e})",
        loss_n[0], loss_p[0]
    );
    if std::env::var("CPU").is_ok() {
        let inputs: Vec<Vec<f64>> = b
            .step
            .inputs
            .iter()
            .map(|v| v.iter().map(|&x| x as f64).collect())
            .collect();
        let want = b.g.eval(&inputs, &b.outs[..2]);
        let dp = want[1]
            .iter()
            .zip(&p_p)
            .map(|(a, b)| (a - *b as f64).abs())
            .fold(0.0, f64::max);
        println!("  cpu loss {} (max |Δp'| gpu vs cpu {dp:e})", want[0][0]);
    }
    let tn = time(&naive, &mut cache);
    let tp = time(&planned, &mut cache);
    drop(naive);
    drop(planned);
    // one block's step, timed alone
    let one = build(seq, 1, false);
    let e1 = PlannedExecutable::new(&device, one.prog, one.planned, &one.step.inputs);
    let mut best = f64::INFINITY;
    e1.run(&device, &mut cache);
    let _ = e1.read(&device, one.outs[0]);
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..5 {
            e1.run(&device, &mut cache);
        }
        let _ = e1.read(&device, one.outs[0]);
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / 5.0);
    }
    println!(
        "  wgpu step: naive {tn:.2} ms, planned {tp:.2} ms; one block's step {best:.2} ms × {layers} = {:.2} ms",
        best * layers as f64
    );
}

#[cfg(feature = "cuda")]
fn run_cuda(b: &Built, layers: usize, seq: usize) {
    use tang_bench::ir_model::cuda::PlannedCuda;
    let ctx = tang_gpu::cudarc::driver::CudaContext::new(0).expect("cuda");
    let t = Instant::now();
    let mut naive = PlannedCuda::new(
        &ctx,
        clone_prog(&b.prog),
        clone_plan(&b.naive),
        &b.step.inputs,
    );
    println!(
        "  cuda: {} kernels, {} distinct sources compiled in {:.2?}",
        b.prog.kernels.len(),
        naive.compiled,
        t.elapsed()
    );
    naive.run();
    let loss_n = naive.read(b.outs[0]);
    let p_n = naive.read(b.outs[1]);
    let tn = naive.time(5, 5);
    naive.capture();
    let tng = naive.time(5, 5);
    drop(naive);
    let mut inplace = PlannedCuda::new(
        &ctx,
        clone_prog(&b.prog),
        clone_plan(&b.inplace),
        &b.step.inputs,
    );
    inplace.run();
    let loss_i = inplace.read(b.outs[0]);
    let p_i = inplace.read(b.outs[1]);
    let dp = p_n
        .iter()
        .zip(&p_i)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  loss naive {} planned+in-place {} (max |Δp'| {dp:e})",
        loss_n[0], loss_i[0]
    );
    let ti = inplace.time(5, 5);
    inplace.capture();
    let tig = inplace.time(5, 5);
    drop(inplace);
    let one = build(seq, 1, true);
    let mut e1 = PlannedCuda::new(&ctx, one.prog, one.inplace, &one.step.inputs);
    let t1 = e1.time(10, 5);
    e1.capture();
    let t1g = e1.time(10, 5);
    println!(
        "  cuda step: naive {tn:.2} ms (graph {tng:.2}), planned+in-place {ti:.2} ms (graph {tig:.2}); one block's step {t1:.2} ms (graph {t1g:.2}) × {layers} = {:.2} ms",
        t1 * layers as f64
    );
    let _ = &b.g;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layers: usize = args.get(1).map_or(4, |s| s.parse().unwrap());
    let seq: usize = args.get(2).map_or(128, |s| s.parse().unwrap());
    let mode = args.get(3).map_or("wgpu", |s| s.as_str());
    if std::env::var("FUSION").is_ok() {
        fusion_report(seq, layers);
    }
    let b = build(seq, layers, mode.starts_with("cuda"));
    match mode {
        "wgpu" => run_wgpu(&b, layers, seq),
        // nvrtc only (no GPU): fill the PTX cache outside the GPU lock
        #[cfg(feature = "cuda")]
        "cuda-compile" => {
            let t = Instant::now();
            tang_bench::ir_model::cuda::compile_all(&b.prog);
            tang_bench::ir_model::cuda::compile_all(&build(seq, 1, true).prog);
            println!("  nvrtc {:.2?}", t.elapsed());
        }
        #[cfg(feature = "cuda")]
        "cuda" => run_cuda(&b, layers, seq),
        _ => {}
    }
}
