//! Print the WGSL that tang-compile derives for causal attention.
fn main() {
    let (g, y) = tang_bench::ir_gpu::flash_attention(4, 2, 2);
    for k in g.to_wgsl(&[y]).kernels {
        println!(
            "// ===== kernel for node {:?}, reads {:?}, count {}\n{}",
            k.node, k.reads, k.count, k.source
        );
    }
}
