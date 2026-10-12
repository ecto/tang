//! Per-kernel native-vs-transpiled times for the training step on CUDA:
//! which tiled kernels lower to tensor cores, and how each lowering does.
//! `cargo run --release -p tang-bench --features cuda --example cuda_train_kernels [seq]`
//! (`TANG_DUMP=<dir>` writes every native kernel's source there.)
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::CudaContext;
    use tang_bench::ir_block::{train_loss_graph, BlockDims};
    use tang_bench::ir_cuda::CudaExecutable;

    let seq: usize = std::env::args().nth(1).map_or(512, |s| s.parse().unwrap());
    let ctx = CudaContext::new(0).expect("cuda");
    let mut rng = tang_bench::Rng::new(7);
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
    let (tg, outs, _) = train_loss_graph(&dims);
    let (tg, outs) = tg.stabilize_softmax(&outs);
    let (tg, outs) = tg.substitute_maps(&outs);
    let (tg, outs) = tg.eliminate_common(&outs);
    let (tg, outs) = tg.merge_siblings(&outs);
    let (tg, outs) = tg.relayout(&outs);
    let (tg, outs) = tg.merge_reduce_dims(&outs);
    let (mut tg, outs) = tg.split_reduce_dims(&outs);
    tg.schedule_auto(&outs);
    let native = tg.to_cuda_with(&outs, true);
    let plain = tg.to_cuda_with(&outs, false);
    let n_inputs = native.buffers.iter().filter(|b| b.input.is_some()).count();
    let inputs: Vec<Vec<f32>> = (0..n_inputs)
        .map(|k| {
            let len = native
                .buffers
                .iter()
                .find(|b| b.input == Some(k))
                .unwrap()
                .len;
            (0..len).map(|_| (rng.f32() * 2.0 - 1.0) * 0.1).collect()
        })
        .collect();
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    if let Ok(dir) = std::env::var("TANG_DUMP") {
        for (i, (k, kp)) in native.kernels.iter().zip(&plain.kernels).enumerate() {
            let n = k.node.index();
            std::fs::write(format!("{dir}/k{i:02}_n{n}.cu"), &k.source).unwrap();
            std::fs::write(format!("{dir}/k{i:02}_n{n}_plain.cu"), &kp.source).unwrap();
        }
    }
    let n = CudaExecutable::new(&ctx, native, &refs);
    let p = CudaExecutable::new(&ctx, plain, &refs);
    n.run();
    p.run();
    if let Ok(path) = std::env::var("TANG_CHECK") {
        // every output (loss and gradients) against the f64 evaluation,
        // which is slow: computed once into `path` (run without the GPU
        // lock), then compared on later runs
        let want: Vec<Vec<f64>> = match std::fs::read(&path) {
            Ok(bytes) => {
                let flat: Vec<f64> = bytes
                    .chunks(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let mut at = 0;
                outs.iter()
                    .map(|o| {
                        let len = n.read(*o).len();
                        at += len;
                        flat[at - len..at].to_vec()
                    })
                    .collect()
            }
            Err(_) => {
                let f64s: Vec<Vec<f64>> = inputs
                    .iter()
                    .map(|v| v.iter().map(|&x| x as f64).collect())
                    .collect();
                let want = tg.eval(&f64s, &outs);
                let bytes: Vec<u8> = want
                    .iter()
                    .flatten()
                    .flat_map(|x| x.to_le_bytes())
                    .collect();
                std::fs::write(&path, bytes).unwrap();
                println!("wrote reference to {path}");
                return;
            }
        };
        for (o, w) in outs.iter().zip(&want) {
            let got = n.read(*o);
            let err = got
                .iter()
                .zip(w)
                .map(|(&x, &y)| (x as f64 - y).abs())
                .fold(0.0, f64::max);
            let scale = w.iter().map(|y| y.abs()).fold(0.0, f64::max);
            println!(
                "check {o:?}: max err {err:.2e} / max |ref| {scale:.2e} = {:.2e}",
                err / scale.max(1e-30)
            );
        }
    }
    let (mut tot_best, mut tot_plain) = (0.0, 0.0);
    for i in 0..n.program.kernels.len() {
        let k = &n.program.kernels[i];
        let node = &tg.nodes()[k.node.index()];
        let tp = p.time_kernel(i, 10);
        let differs = k.source != p.program.kernels[i].source;
        let tn = if differs { n.time_kernel(i, 10) } else { tp };
        tot_best += tn.min(tp);
        tot_plain += tp;
        let c = tg.comp_at(k.node.index());
        println!(
            "k{i:02} n{:<4} plain {:>7.3} native {:>7.3} {:4} shape {:?} red {:?} monoid {:?} tile {:?} split {}",
            k.node.index(),
            tp,
            tn,
            if differs {
                "yes"
            } else if node.tile.0 > 0 {
                "NONE"
            } else {
                "-"
            },
            node.shape.iter().map(|d| d.n()).collect::<Vec<_>>(),
            c.map(|c| c.red.iter().map(|d| d.n()).collect::<Vec<_>>()),
            c.and_then(|c| c.monoid.as_ref().map(|m| m.kind())),
            node.tile,
            node.split,
        );
    }
    println!(
        "sum plain {tot_plain:.3} ms, sum best {tot_best:.3} ms, {} kernels",
        n.program.kernels.len()
    );
}
