//! Print the comps tang-compile derives for attention after the full
//! pass pipeline (`flash_attention_auto` minus the schedule):
//! `dump_attention_graph [n] [d]`.
use tang_compile::tensor::{Op, TensorGraph, TensorId};

fn dump(g: &TensorGraph, outs: &[TensorId]) {
    let mat: Vec<usize> = g
        .materialized(outs)
        .iter()
        .map(|(t, _)| t.index())
        .collect();
    for (i, n) in g.nodes().iter().enumerate() {
        let shape: Vec<usize> = n.shape.iter().map(|d| d.n()).collect();
        match &n.op {
            Op::Comp(c) => {
                let red: Vec<usize> = c.red.iter().map(|d| d.n()).collect();
                println!(
                    "#{i} comp {shape:?} tuple {} red {red:?} inline {} mat {} monoid {:?}",
                    c.tuple,
                    n.inline,
                    mat.contains(&i),
                    c.monoid.as_ref().map(|m| (m.kind(), m.width))
                );
                for (k, a) in c.args.iter().enumerate() {
                    let f: Vec<(Vec<i64>, i64)> = a
                        .view
                        .forms
                        .iter()
                        .map(|f| (f.coefs.clone(), f.offset))
                        .collect();
                    println!(
                        "    arg{k} <- #{} guard {} {f:?}",
                        a.src.index(),
                        a.view.guard
                    );
                }
                for (k, o) in c.body_outs.iter().enumerate() {
                    println!("    out{k} = {}", c.body.fmt_expr(*o));
                }
            }
            Op::Input(k) => println!("#{i} input{k} {shape:?}"),
            Op::Iota(d) => println!("#{i} iota{d} {shape:?}"),
            _ => println!("#{i} other {shape:?}"),
        }
    }
    println!(
        "outputs {:?}",
        outs.iter().map(|t| t.index()).collect::<Vec<_>>()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(128);
    let d = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(64);
    let hk = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(1);
    let gq = args.get(4).and_then(|a| a.parse().ok()).unwrap_or(1);
    let causal = args.get(5).is_none_or(|a| a != "full");
    let (g, y) = tang_bench::ir_gpu::fused_attention_heads(hk, gq, n, d, d, causal);
    dump(&g, &[y]);
    kernels("heads", &g, y);
    if args.get(6).is_none_or(|a| a != "all") {
        return;
    }
    for (name, (g, y)) in [
        ("flash", tang_bench::ir_gpu::flash_attention(n, d, d)),
        ("auto", tang_bench::ir_gpu::flash_attention_auto(n, d, d)),
    ] {
        kernels(name, &g, y);
        if name == "auto" {
            dump(&g, &[y]);
        }
    }
}

#[allow(dead_code)]
fn kernels(name: &str, g: &TensorGraph, y: TensorId) {
    let p = g.to_wgsl(&[y]);
    for k in &p.kernels {
        let n = &g.nodes()[k.node.index()];
        println!(
            "{name}: kernel #{} reads {:?} count {} wg {} split {} tile {:?} inline {}",
            k.node.index(),
            k.reads.iter().map(|t| t.index()).collect::<Vec<_>>(),
            k.count,
            k.workgroup_size,
            n.split,
            n.tile,
            n.inline
        );
    }
}
