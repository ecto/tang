//! Print the WGSL that tang-compile derives for tiled causal attention.
fn main() {
    let (g, y) = tang_bench::ir_gpu::flash_attention_tiled(128, 4, 4);
    for k in g.to_wgsl(&[y]).kernels {
        println!(
            "// ===== kernel for node {:?}, reads {:?}, count {}\n{}",
            k.node, k.reads, k.count, k.source
        );
    }
}
