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
    use tang_bench::ir_block::{compiled_block, BlockDims};
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
        let exe = CudaExecutable::new(&ctx, program, &refs);
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
    }
}
