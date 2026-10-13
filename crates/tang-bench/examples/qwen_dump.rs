//! Dump the decode program's kernels for a small Qwen-shaped graph:
//! `cargo run -p tang-bench --example qwen_dump -- <dir>`.
fn main() {
    use tang_bench::ir_qwen::*;
    let dims = QwenDims {
        d_model: 128,
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        ff: 192,
        vocab: 256,
        layers: 1,
        ctx: 16,
        eps: 1e-6,
        theta: 1e4,
    };
    let dg = decode_graph(&dims);
    let (g, outs) = compiled_decode(&dg);
    let opts = decode_opts(&dg, &outs, dims.layers);
    let p = g.to_cuda_decode(&outs, &opts);
    let dir = std::env::args().nth(1).expect("output dir");
    for (i, (k, n)) in p.program.kernels.iter().zip(&p.native).enumerate() {
        println!(
            "k{i:02} native={n} points={} count={} wg={} reads={}",
            k.points,
            k.count,
            k.workgroup_size,
            k.reads.len()
        );
        std::fs::write(format!("{dir}/k{i:02}.cu"), &k.source).unwrap();
    }
}
