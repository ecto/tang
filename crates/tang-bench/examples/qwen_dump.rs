//! Dump a small Qwen-shaped step program's kernels:
//! `cargo run -p tang-bench --example qwen_dump -- <dir> [seq]`.
fn main() {
    use tang_bench::ir_qwen::*;
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("output dir");
    let seq: usize = args.next().map_or(1, |s| s.parse().unwrap());
    let dims = QwenDims {
        d_model: 128,
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        ff: 192,
        vocab: 256,
        layers: 1,
        ctx: if seq == 1 { 16 } else { seq },
        eps: 1e-6,
        theta: 1e4,
    };
    let dg = step_graph(&dims, seq);
    let (g, outs) = compiled_decode(&dg);
    let opts = decode_opts(&dg, &outs, dims.layers);
    let p = g.to_cuda_decode(&outs, &opts);
    for (i, (k, n)) in p.program.kernels.iter().zip(&p.native).enumerate() {
        let node = &g.nodes()[k.node.index()];
        let red = g
            .tiled_reduction(k.node.index())
            .and_then(|r| g.comp_at(r))
            .map(|c| c.red.clone());
        println!(
            "k{i:02} native={n} shape={:?} red={:?} reads={} {}",
            node.shape,
            red,
            k.reads.len(),
            k.source.lines().nth(0).unwrap_or("")
        );
        std::fs::write(format!("{dir}/k{i:02}.cu"), &k.source).unwrap();
    }
}
