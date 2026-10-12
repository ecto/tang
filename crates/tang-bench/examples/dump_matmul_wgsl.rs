//! Print the WGSL of one matmul candidate: `dump_matmul_wgsl [name]`.
use tang_bench::ir_gpu::matmul_candidates;
fn main() {
    let want = std::env::args().nth(1).unwrap_or_else(|| "reg4".into());
    for (name, g, c) in matmul_candidates(64, 64, 64) {
        if name == want {
            for k in g.to_wgsl(&[c]).kernels {
                println!("{}", k.source);
            }
        }
    }
}
