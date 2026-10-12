//! The transformer block and the matmul candidates through tang-gpu's
//! program runner. With `--features cuda` on an NVIDIA GPU that is CUDA
//! (`to_cuda`: wmma tensor-core tiles + transpiled WGSL, tuned per
//! kernel); otherwise wgpu (`TANG_BACKEND=wgpu` forces it). Checked
//! against the CPU evaluation and timed.
fn main() {
    use tang_bench::ir_block::{compiled_block, train_loss_graph, BlockDims};
    use tang_compile::tensor::TensorGraph;
    use tang_gpu::program::{Program, Runtime};

    let rt = Runtime::select().expect("gpu");
    println!("backend: {:?} — {}", rt.kind(), rt.name());
    let tuning = |name: String| {
        std::path::PathBuf::from(format!(
            "{}/target/tang-tuning-{name}.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
    };
    // order-sensitive checksum of a result, for comparing runs bit for bit
    let digest = |v: &[f32]| -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for x in v {
            x.to_bits().hash(&mut h);
        }
        h.finish()
    };
    let mut rng = tang_bench::Rng::new(7);
    let mut rand = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect() };

    // matmul: wmma tile vs CPU
    for (m, k, n) in [(1024usize, 1024usize, 1024usize), (512, 576, 1536)] {
        let mut g = TensorGraph::new();
        let a = g.input(&[m, k]);
        let b = g.input(&[k, n]);
        let c = g.matmul(a, b).unwrap();
        g.schedule_auto(&[c]);
        let (av, bv) = (rand(m * k), rand(k * n));
        let exe = Program::compile(&rt, &g, &[c], &[&av, &bv]);
        let native = exe.program.kernels[0].source.contains("wmma");
        let ms = exe.time(&rt, 50);
        println!("matmul {m}x{k}x{n}: {ms:.3} ms (native wmma: {native})");
    }

    // block forward
    for seq in [128usize, 512] {
        let dims = BlockDims {
            seq,
            kv_len: seq,
            rope: true,
            bias: true,
            d_model: 576,
            n_heads: 9,
            n_kv_heads: 3,
            head_dim: 64,
            ff_dim: 1536,
        };
        let (g, y) = compiled_block(&dims);
        let program = rt.lower(&g, &[y]);
        let n_native = program
            .kernels
            .iter()
            .filter(|k| k.source.contains("wmma"))
            .count();
        let n_inputs = program.buffers.iter().filter(|b| b.input.is_some()).count();
        let inputs: Vec<Vec<f32>> = (0..n_inputs)
            .map(|k| {
                let len = program
                    .buffers
                    .iter()
                    .find(|b| b.input == Some(k))
                    .unwrap()
                    .len;
                rand(len).iter().map(|x| x * 0.1).collect()
            })
            .collect();
        let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        drop(program);
        let (exe, report) = Program::tuned(
            &rt,
            &g,
            &[y],
            &refs,
            Some(&tuning(format!("cuda-block-{seq}"))),
        );
        println!(
            "block seq={seq}: tuned{}, {} of {n_native} native kernels kept",
            if report.from_cache { " (cached)" } else { "" },
            report.native_kept
        );
        exe.run(&rt);
        let got = exe.read(&rt, y);
        println!("block seq={seq}: output digest {:016x}", digest(&got));
        if seq == 128 {
            let f64s: Vec<Vec<f64>> = inputs
                .iter()
                .map(|v| v.iter().map(|&x| x as f64).collect())
                .collect();
            let want = g.eval(&f64s, &[y]).remove(0);
            let err = got
                .iter()
                .zip(&want)
                .map(|(&x, &y)| (x as f64 - y).abs())
                .fold(0.0, f64::max);
            let scale = want.iter().map(|y| y.abs()).fold(0.0, f64::max);
            println!("block seq={seq}: max err {err:.2e} (max |ref| {scale:.2})");
        }
        let ms = exe.time(&rt, 50);
        println!(
            "block seq={seq}: fwd {ms:.3} ms ({} kernels, {n_native} on tensor cores)",
            exe.program.kernels.len()
        );
        for (i, k) in exe.program.kernels.iter().enumerate() {
            let node = &g.nodes()[k.node.index()];
            println!(
                "  {:7.3} ms  node {:?} shape {:?} split {} tile {:?} {}",
                exe.time_kernel(&rt, i, 20),
                k.node,
                node.shape,
                node.split,
                node.tile,
                if k.source.contains("wmma") {
                    "wmma"
                } else {
                    ""
                }
            );
        }
        // training step: loss + every gradient, same pipeline
        let (tg, outs, _) = train_loss_graph(&dims);
        let (tg, outs) = tg.stabilize_softmax(&outs);
        let (tg, outs) = tg.substitute_maps(&outs);
        let (tg, outs) = tg.eliminate_common(&outs);
        let (tg, outs) = tg.merge_siblings(&outs);
        let (tg, outs) = tg.relayout(&outs);
        let (tg, outs) = tg.merge_reduce_dims(&outs);
        let (mut tg, outs) = tg.split_reduce_dims(&outs);
        tg.schedule_auto(&outs);
        let program = rt.lower(&tg, &outs);
        let n_native = program
            .kernels
            .iter()
            .filter(|k| k.source.contains("wmma"))
            .count();
        let n_kernels = program.kernels.len();
        let target = vec![0.0f32; seq * 576];
        let mut refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        refs.push(&target);
        drop(program);
        let (train, report) = Program::tuned(
            &rt,
            &tg,
            &outs,
            &refs,
            Some(&tuning(format!("cuda-train-{seq}"))),
        );
        let kept = report.native_kept;
        train.run(&rt);
        let mut all = Vec::new();
        for &o in &outs {
            all.extend(train.read(&rt, o));
        }
        println!(
            "block seq={seq}: train loss {:.6}, outputs digest {:016x}",
            all[0],
            digest(&all)
        );
        let ms = train.time(&rt, 50);
        println!(
            "block seq={seq}: fwd+bwd {ms:.3} ms ({n_kernels} kernels, {kept} of {n_native} native kept)"
        );
        let mut slow: Vec<(f64, usize)> = (0..n_kernels)
            .map(|i| (train.time_kernel(&rt, i, 10), i))
            .collect();
        slow.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let total: f64 = slow.iter().map(|x| x.0).sum();
        println!("  (sum of kernels {total:.3} ms)");
        for (ms, i) in slow.iter().take(14) {
            let k = &train.program.kernels[*i];
            let node = &tg.nodes()[k.node.index()];
            let c = tg.comp_at(k.node.index());
            let args: Vec<String> = c
                .map(|c| {
                    c.args
                        .iter()
                        .map(|a| {
                            format!(
                                "{}{:?}",
                                a.src.index(),
                                tg.nodes()[a.src.index()]
                                    .shape
                                    .iter()
                                    .map(|d| d.n())
                                    .collect::<Vec<_>>()
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            println!(
                "  {ms:7.3} ms  node {:?} shape {:?} red {:?} monoid {:?} split {} tile {:?} args [{}] {}",
                k.node,
                node.shape.iter().map(|d| d.n()).collect::<Vec<_>>(),
                c.map(|c| c.red.iter().map(|d| d.n()).collect::<Vec<_>>()),
                c.and_then(|c| c.monoid.as_ref().map(|m| m.kind())),
                node.split,
                node.tile,
                args.join(" "),
                if k.source.contains("wmma") {
                    "wmma"
                } else {
                    ""
                }
            );
        }
    }
}
