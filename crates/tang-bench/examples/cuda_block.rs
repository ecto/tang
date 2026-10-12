//! The transformer block and the matmul candidates on CUDA: the derived
//! program (`to_cuda`: wmma tensor-core tiles + transpiled WGSL) run
//! through the driver, checked against the CPU evaluation and timed.
//! Needs `--features cuda` and an NVIDIA GPU.
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::CudaContext;
    use tang_bench::ir_block::{compiled_block, train_loss_graph, BlockDims};
    use tang_bench::ir_cuda::CudaExecutable;
    use tang_compile::tensor::TensorGraph;

    let ctx = CudaContext::new(0).expect("cuda");
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
        let exe = CudaExecutable::new(&ctx, g.to_cuda(&[c]), &[&av, &bv]);
        let native = exe.program.kernels[0].source.contains("wmma");
        let ms = exe.time(50);
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
        let program = g.to_cuda(&[y]);
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
        let (exe, kept) = CudaExecutable::new_tuned(&ctx, &g, &[y], &refs);
        println!("block seq={seq}: tuned, {kept} of {n_native} native kernels kept");
        exe.run();
        let got = exe.read(y);
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
        let ms = exe.time(50);
        println!(
            "block seq={seq}: fwd {ms:.3} ms ({} kernels, {n_native} on tensor cores)",
            exe.program.kernels.len()
        );
        for (i, k) in exe.program.kernels.iter().enumerate() {
            let node = &g.nodes()[k.node.index()];
            println!(
                "  {:7.3} ms  node {:?} shape {:?} split {} tile {:?} {}",
                exe.time_kernel(i, 20),
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
        let program = tg.to_cuda(&outs);
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
        let (train, kept) = CudaExecutable::new_tuned(&ctx, &tg, &outs, &refs);
        let ms = train.time(50);
        println!(
            "block seq={seq}: fwd+bwd {ms:.3} ms ({n_kernels} kernels, {kept} of {n_native} native kept)"
        );
        let mut slow: Vec<(f64, usize)> = (0..n_kernels)
            .map(|i| (train.time_kernel(i, 10), i))
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
