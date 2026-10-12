//! Write every kernel of the block's `to_cuda` program to `DIR/k{i}.cu`
//! (default `/tmp/tang_cuda`), for inspection with nvcc/ptxas.
use tang_bench::ir_block::{compiled_block, BlockDims};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/tang_cuda".into());
    let seq = std::env::args().nth(2).map_or(128, |s| s.parse().unwrap());
    std::fs::create_dir_all(&dir).unwrap();
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
    let (g, y) = compiled_block(&dims);
    for (i, k) in g.to_cuda(&[y]).kernels.iter().enumerate() {
        let node = &g.nodes()[k.node.index()];
        let path = format!("{dir}/k{i}.cu");
        std::fs::write(&path, &k.source).unwrap();
        println!(
            "{path}: node {:?} tile {:?} wg {} count {}",
            k.node, node.tile, k.workgroup_size, k.count
        );
    }
}
