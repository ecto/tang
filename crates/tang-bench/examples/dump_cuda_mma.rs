//! Print the CUDA tensor-core kernel tang-compile derives for a matmul
//! (optionally with a bias + silu epilogue), for `scripts/cuda_bench.cu`:
//! `dump_cuda_mma M K N [epilogue]`.
use tang::Scalar;
use tang_compile::tensor::TensorGraph;
use tang_expr::ExprId;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dim = |i: usize, d: usize| args.get(i).map_or(d, |s| s.parse().unwrap());
    let (m, k, n) = (dim(1, 1024), dim(2, 1024), dim(3, 1024));
    let epilogue = args.get(4).is_some();
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
    let (src, reads) = g.cuda_mma_program(&outs).expect("matmul-shaped tile");
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
        "#define TANG_LAUNCH(a, b, bias, out) tang_kernel<<<TANG_TILES, TANG_THREADS>>>({}, out)",
        names.join(", ")
    );
    print!("{src}");
}
