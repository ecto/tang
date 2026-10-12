//! Print the CUDA tensor-core kernel tang-compile derives for a matmul
//! (optionally with a bias + silu epilogue), for `scripts/cuda_bench.cu`:
//! `dump_cuda_mma M K N [epilogue]`.
use tang::Scalar;
use tang_compile::tensor::cuda::{MmaShape, MMA_SHAPES};
use tang_compile::tensor::TensorGraph;
use tang_expr::ExprId;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dims: Vec<usize> = args.iter().filter_map(|s| s.parse().ok()).collect();
    let dim = |i: usize| dims.get(i).copied().unwrap_or(1024);
    let (m, k, n) = (dim(0), dim(1), dim(2));
    let epilogue = args.iter().any(|s| s == "epi");
    let mut g = TensorGraph::new();
    let a = g.input(&[m, k]);
    let b = g.input(&[k, n]);
    let c = g.matmul(a, b).unwrap();
    let y = if epilogue {
        let bias = g.input(&[n]);
        let bias1 = g.reshape(bias, &[1, n]).unwrap();
        let bb = g.broadcast(bias1, &[m, n]).unwrap();
        g.map(&[c, bb], |v| {
            let s = v[0] + v[1];
            s / (ExprId::from_f64(1.0) + (-s).exp())
        })
        .unwrap()
    } else {
        c
    };
    let (mut g, outs) = g.substitute_maps(&[y]);
    g.schedule_auto(&outs);
    // TANG_MMA_SHAPE: an index into MMA_SHAPES, or bm,bn,warps_m,warps_n[,chunk[,stages]]
    let shape = match std::env::var("TANG_MMA_SHAPE") {
        Ok(v) => {
            let f: Vec<usize> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            if f.len() == 1 {
                MMA_SHAPES[f[0]]
            } else {
                let mut sh = MmaShape::new(f[0], f[1], f[2], f[3]);
                if let Some(&c) = f.get(4) {
                    sh.chunk = c;
                }
                if let Some(&st) = f.get(5) {
                    sh.stages = st;
                }
                sh
            }
        }
        Err(_) => MMA_SHAPES[0],
    };
    let (src, reads) = g
        .cuda_mma_program_shaped(&outs, shape)
        .expect("matmul-shaped tile");
    println!("#define TANG_M {m}\n#define TANG_K {k}\n#define TANG_N {n}");
    println!("#define TANG_EPILOGUE {}", epilogue as u32);
    // the kernel takes its read buffers in binding order; name them by
    // graph input so the harness can pass a, b, bias positionally
    let names: Vec<String> = reads
        .iter()
        .map(|t| match g.nodes()[t.index()].op {
            tang_compile::tensor::Op::Input(k) => ["a", "b", "bias"][k].to_string(),
            _ => panic!("read of a non-input"),
        })
        .collect();
    println!(
        "#define TANG_LAUNCH(a, b, bias, out) main0<<<TANG_TILES, TANG_THREADS, TANG_SMEM>>>({}, out, 0u)",
        names.join(", ")
    );
    print!("{src}");
}
