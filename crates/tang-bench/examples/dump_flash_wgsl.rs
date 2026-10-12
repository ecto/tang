//! Print the kernels tang-compile derives for tiled causal attention.
//! `dump_flash_wgsl [n] [d] [full]`; with `full` the WGSL is printed too.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(128);
    let d = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(4);
    let full = args.get(3).is_some();
    let (g, y) = tang_bench::ir_gpu::flash_attention_tiled(n, d, d);
    let prog = g.to_wgsl(&[y]);
    for k in &prog.kernels {
        let node = &g.nodes()[k.node.index()];
        println!(
            "// ===== kernel node {} shape {:?} inline-closure reads {:?} count {} wg {} split {} tile {:?}",
            k.node.index(),
            node.shape.iter().map(|d| d.n()).collect::<Vec<_>>(),
            k.reads,
            k.count,
            k.workgroup_size,
            node.split,
            node.tile
        );
        if full {
            println!("{}", k.source);
        }
    }
    println!("materialized: {:?}", g.materialized(&[y]));
}
