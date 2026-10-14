//! Dump the tiny Qwen decode program's kernels (CUDA spelling) to a dir.
use tang_bench::ir_qwen::{compiled_decode, decode_graph, decode_opts, QwenDims};
fn main() {
    let dir = std::env::args().nth(1).expect("usage: dump_decode DIR");
    std::fs::create_dir_all(&dir).unwrap();
    let dims = QwenDims {
        d_model: 128,
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        ff: 192,
        vocab: 256,
        layers: 2,
        ctx: 16,
        eps: 1e-6,
        theta: 1e4,
        kv_f16: true,
    };
    let dg = decode_graph(&dims);
    let (g, outs) = compiled_decode(&dg, true);
    let opts = decode_opts(&dg, &outs);
    let p = match std::env::var("DUMP_TARGET").as_deref() {
        Ok("glsl") => g.to_glsl_decode(&outs, &opts),
        _ => g.to_cuda_decode(&outs, &opts),
    };
    let native_ext = if std::env::var("DUMP_TARGET").as_deref() == Ok("glsl") {
        "comp"
    } else {
        "cu"
    };
    for (i, k) in p.program.kernels.iter().enumerate() {
        let ext = if p.native[i] { native_ext } else { "wgsl" };
        std::fs::write(format!("{dir}/k{i:03}.{ext}"), &k.source).unwrap();
    }
    println!(
        "{} kernels, {} native",
        p.native.len(),
        p.native.iter().filter(|n| **n).count()
    );
}
