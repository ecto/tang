//! The native Metal backend against the f64 interpreter (and against
//! `eval_emulated` for f16 schedules), for every emitter path: one thread
//! per point, workgroup split, tiles with register tiles, multi-dim
//! reductions with guarded/dilated windowed staging, epilogue fusion,
//! online-softmax attention, backward graphs, and fragment lowering.
//! Random inputs, every output element compared.
#![cfg(feature = "metal")]

use tang::Scalar;
use tang_bench::ir_msl::{Executable, MslDevice};
use tang_compile::tensor::{Arg, Form, Format, Monoid, Precision, TensorGraph, TensorId, View};

fn dev() -> Option<MslDevice> {
    let d = MslDevice::new();
    if d.is_none() {
        eprintln!("no Metal device, skipping");
    }
    d
}

fn rand(n: usize, rng: &mut tang_bench::Rng) -> Vec<f32> {
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn f64s(v: &[Vec<f32>]) -> Vec<Vec<f64>> {
    v.iter()
        .map(|x| x.iter().map(|&y| y as f64).collect())
        .collect()
}

/// Run on Metal and compare every element of every output with `want`,
/// relative to each output's largest magnitude.
fn check(
    dev: &MslDevice,
    name: &str,
    g: &TensorGraph,
    outs: &[TensorId],
    inputs: &[Vec<f32>],
    want: &[Vec<f64>],
    tol: f64,
) -> Executable {
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let exe = Executable::new(dev, g.to_msl(outs), &refs);
    exe.run(dev);
    for (o, w) in outs.iter().zip(want) {
        let got = exe.read(*o);
        assert_eq!(got.len(), w.len(), "{name}: length");
        // an all-but-zero output (a masked gradient) is judged absolutely
        let scale = w.iter().fold(1e-6f64, |m, v| m.max(v.abs()));
        for (i, (a, b)) in got.iter().zip(w).enumerate() {
            assert!(
                (*a as f64 - b).abs() <= tol * scale,
                "{name} out {}[{i}]: {a} vs {b} (scale {scale})",
                o.index()
            );
        }
    }
    exe
}

fn matmul_graph(m: usize, k: usize, n: usize) -> (TensorGraph, TensorId) {
    let mut g = TensorGraph::new();
    let a = g.input(&[m, k]);
    let b = g.input(&[k, n]);
    let c = g.matmul(a, b).unwrap();
    (g, c)
}

type Mapping = (&'static str, u32, (u32, u32, u32), (u32, u32));

const MAPPINGS: [Mapping; 6] = [
    ("point", 0, (0, 0, 0), (1, 1)),
    ("split", 64, (0, 0, 0), (1, 1)),
    ("tile", 0, (16, 16, 32), (1, 1)),
    ("tile 2x2", 0, (16, 16, 32), (2, 2)),
    ("tile 4x4", 0, (16, 16, 32), (4, 4)),
    ("rows", 0, (64, 1, 8), (1, 1)),
];

#[test]
fn matmul_every_mapping_f32() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(1);
    for (m, k, n) in [(100, 100, 100), (128, 576, 1536), (70, 50, 90)] {
        let (g, c) = matmul_graph(m, k, n);
        let inputs = vec![rand(m * k, &mut rng), rand(k * n, &mut rng)];
        let want = g.eval(&f64s(&inputs), &[c]);
        for (name, split, tile, reg) in MAPPINGS {
            let mut gv = g.clone();
            gv.set_schedule(c, split, tile, reg);
            check(
                &dev,
                &format!("{m}x{k}x{n} {name}"),
                &gv,
                &[c],
                &inputs,
                &want,
                1e-5,
            );
        }
    }
}

/// f16 buffers are native `half`; f16 accumulators round every combine.
/// The interpreter's emulation predicts both to within an f16 ulp or two.
#[test]
fn f16_storage_and_accum_match_emulation() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(7);
    let (m, k, n) = (48, 96, 80);
    let inputs = vec![
        rand(m * k, &mut rng),
        rand(k * n, &mut rng),
        rand(n, &mut rng),
    ];
    let half = Precision::storage(Format::F16);
    for accum in [Format::F32, Format::F16] {
        for (name, split, tile, reg) in MAPPINGS {
            let mut g = TensorGraph::new();
            let ga = g.input(&[m, k]);
            let gb = g.input(&[k, n]);
            let gbias = g.input(&[1, n]);
            let y = g.matmul(ga, gb).unwrap();
            let bb = g.broadcast(gbias, &[m, n]).unwrap();
            let z = g.map(&[y, bb], |v| v[0] * v[0] + v[1]).unwrap();
            for t in [ga, gb, gbias, z] {
                g.set_precision(t, half);
            }
            g.set_precision(
                y,
                Precision {
                    storage: Format::F16,
                    accum,
                },
            );
            g.set_schedule(y, split, tile, reg);
            let emu = g.eval_emulated(&f64s(&inputs), &[y, z]);
            // the emulation sums in a fixed order; the GPU's split/tile
            // order differs, so allow a couple of f16 ulps of the result
            let tol = if accum == Format::F16 { 2e-2 } else { 2e-3 };
            check(
                &dev,
                &format!("f16 {accum:?} {name}"),
                &g,
                &[y, z],
                &inputs,
                &emu,
                tol,
            );
        }
    }
}

/// Direct conv `y[o,t] = Σ_{c,k} w[o,c,k]·x[c, t + d·k − p]`, guarded.
fn conv_graph(ci: usize, co: usize, t: usize, k: usize, dil: i64) -> (TensorGraph, TensorId) {
    let pad = dil * (k as i64 - 1) / 2;
    let mut g = TensorGraph::new();
    let x = g.input(&[ci, t]);
    let w = g.input(&[co, ci, k]);
    let xv = View {
        forms: vec![
            Form::unit(4, 2),
            Form {
                coefs: vec![0, 1, 0, dil],
                offset: -pad,
            },
        ],
        guard: true,
    };
    let y = g
        .comp(
            &[co, t],
            &[ci, k],
            vec![
                Arg { src: x, view: xv },
                Arg {
                    src: w,
                    view: View::pick(4, &[Some(0), Some(2), Some(3)]),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    (g, y)
}

#[test]
fn conv_guarded_dilated_and_backward_f32() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(11);
    for &(ci, co, t, k, dil) in &[(5, 7, 37, 3, 1), (40, 33, 70, 7, 3), (19, 64, 130, 5, 2)] {
        let (g, y) = conv_graph(ci, co, t, k, dil);
        let inputs = vec![rand(ci * t, &mut rng), rand(co * ci * k, &mut rng)];
        let want = g.eval(&f64s(&inputs), &[y]);
        for (name, split, tile, reg) in MAPPINGS {
            let mut gv = g.clone();
            gv.set_schedule(y, split, tile, reg);
            let name = format!("conv {ci}x{co}x{t} k{k} d{dil} {name}");
            check(&dev, &name, &gv, &[y], &inputs, &want, 1e-5);
        }
        // forward + backward, auto-scheduled, then every candidate per node
        let mut gb = g.clone();
        let grads = gb
            .backward(y, &[TensorId::from_index(0), TensorId::from_index(1)])
            .unwrap();
        let mut outs = vec![y];
        outs.extend(grads);
        let want = gb.eval(&f64s(&inputs), &outs);
        let (gb, outs) = gb.substitute_maps(&outs);
        let (mut gb, outs) = gb.eliminate_common(&outs);
        gb.schedule_auto(&outs);
        check(&dev, "conv fwd+bwd auto", &gb, &outs, &inputs, &want, 1e-4);
        for o in &outs {
            for (split, tile, reg) in gb.schedule_candidates(*o) {
                let mut gv = gb.clone();
                gv.set_schedule(*o, split, tile, reg);
                let name = format!("bwd node {} {split} {tile:?} {reg:?}", o.index());
                check(&dev, &name, &gv, &outs, &inputs, &want, 1e-4);
            }
        }
    }
}

/// A tiled conv with a fused epilogue (bias + snake), f32.
#[test]
fn conv_epilogue_fused_f32() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(5);
    let (ci, co, t, k, dil) = (32, 48, 100, 7, 3);
    let (mut g, y) = conv_graph(ci, co, t, k, dil);
    let bias = g.input(&[co, 1]);
    let bb = g.broadcast(bias, &[co, t]).unwrap();
    let z = g
        .map(&[y, bb], |v| {
            let s = v[0] + v[1];
            s + s.sin() * s.sin()
        })
        .unwrap();
    let inputs = vec![
        rand(ci * t, &mut rng),
        rand(co * ci * k, &mut rng),
        rand(co, &mut rng),
    ];
    let want = g.eval(&f64s(&inputs), &[z]);
    let (g, outs) = g.substitute_maps(&[z]);
    let mut g = g;
    g.schedule_auto(&outs);
    let z = outs[0];
    let exe = check(&dev, "conv epilogue", &g, &[z], &inputs, &want, 1e-5);
    assert_eq!(exe.program.kernels.len(), 1, "epilogue should fuse");
}

/// Online-softmax attention (one fused reduction) under every mapping
/// the WGSL path runs, plus the materialized naive form.
#[test]
fn attention_online_softmax_f32() {
    let Some(dev) = dev() else { return };
    use tang_bench::ir_gpu::{flash_attention, naive_attention};
    let mut rng = tang_bench::Rng::new(9);
    for (n, d) in [(64, 16), (130, 32)] {
        let inputs = vec![
            rand(n * d, &mut rng),
            rand(n * d, &mut rng),
            rand(n * d, &mut rng),
        ];
        let (g, y) = naive_attention(n, d, d);
        let want = g.eval(&f64s(&inputs), &[y]);
        check(&dev, "naive", &g, &[y], &inputs, &want, 1e-5);
        let (g, y) = flash_attention(n, d, d);
        check(&dev, "flash point", &g, &[y], &inputs, &want, 1e-5);
        let mut gs = g.clone();
        gs.schedule_split(&[y], 256, 64);
        check(&dev, "flash split", &gs, &[y], &inputs, &want, 1e-5);
        let mut gt = g.clone();
        gt.schedule_tile_2d(&[y], d as u32, 4, 32, 64);
        check(&dev, "flash tiled", &gt, &[y], &inputs, &want, 1e-5);
        let (ga, ya) = tang_bench::ir_gpu::flash_attention_auto(n, d, d);
        check(&dev, "flash auto", &ga, &[ya], &inputs, &want, 1e-5);
    }
}

/// The transformer block's backward graph on Metal vs the interpreter.
#[test]
fn block_backward_f32() {
    let Some(dev) = dev() else { return };
    use tang_bench::ir_block::{transformer_block, BlockDims};
    let dims = BlockDims {
        seq: 12,
        kv_len: 12,
        rope: true,
        bias: true,
        d_model: 64,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 16,
        ff_dim: 48,
    };
    let (mut g, out) = transformer_block(&dims);
    let mut rng = tang_bench::Rng::new(31);
    let n_inputs = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, tang_compile::tensor::Op::Input(_)))
        .count();
    let inputs: Vec<Vec<f32>> = (0..n_inputs)
        .map(|k| {
            let node = g
                .nodes()
                .iter()
                .find(|n| matches!(n.op, tang_compile::tensor::Op::Input(j) if j == k))
                .unwrap();
            let len: usize = node.shape.iter().map(|d| d.n()).product();
            (0..len).map(|_| (rng.f32() * 2.0 - 1.0) * 0.5).collect()
        })
        .collect();
    let (xi, wq) = (TensorId::from_index(0), TensorId::from_index(1));
    let grads = g.backward(out, &[xi, wq]).unwrap();
    let want = g.eval(&f64s(&inputs), &grads);
    let (g, outs) = g.substitute_maps(&grads);
    let (mut g, outs) = g.eliminate_common(&outs);
    g.schedule_auto(&outs);
    check(&dev, "block backward", &g, &outs, &inputs, &want, 1e-4);
}

// ---------------------------------------------------------------------------
// fragment lowering
// ---------------------------------------------------------------------------

fn f16_operands(g: &mut TensorGraph, ts: &[TensorId]) {
    for &t in ts {
        g.set_precision(t, Precision::storage(Format::F16));
    }
}

#[test]
fn fragment_matmul_ragged() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(21);
    for (m, k, n) in [
        (100, 100, 100),
        (128, 576, 1536),
        (64, 64, 64),
        (7, 300, 33),
    ] {
        let (mut g, c) = matmul_graph(m, k, n);
        let inputs = vec![rand(m * k, &mut rng), rand(k * n, &mut rng)];
        assert!(
            g.fragment_plan(c).is_none(),
            "f32 operands must not get fragments"
        );
        f16_operands(&mut g, &[TensorId::from_index(0), TensorId::from_index(1)]);
        assert!(g.fragment_plan(c).is_some());
        g.set_fragment(c, true);
        let emu = g.eval_emulated(&f64s(&inputs), &[c]);
        let exe = check(
            &dev,
            &format!("frag {m}x{k}x{n}"),
            &g,
            &[c],
            &inputs,
            &emu,
            1e-5,
        );
        assert!(exe.program.kernels[0].fragment);
        // f16 output
        g.set_precision(c, Precision::storage(Format::F16));
        let emu = g.eval_emulated(&f64s(&inputs), &[c]);
        check(
            &dev,
            &format!("frag {m}x{k}x{n} f16 out"),
            &g,
            &[c],
            &inputs,
            &emu,
            1e-3,
        );
    }
}

#[test]
fn fragment_batched_matmul() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(22);
    let (bt, m, k, n) = (3, 70, 40, 90);
    let mut g = TensorGraph::new();
    let a = g.input(&[bt, m, k]);
    let b = g.input(&[bt, k, n]);
    let c = g
        .comp(
            &[bt, m, n],
            &[k],
            vec![
                Arg {
                    src: a,
                    view: View::pick(4, &[Some(0), Some(1), Some(3)]),
                },
                Arg {
                    src: b,
                    view: View::pick(4, &[Some(0), Some(3), Some(2)]),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    f16_operands(&mut g, &[a, b]);
    g.set_fragment(c, true);
    let inputs = vec![rand(bt * m * k, &mut rng), rand(bt * k * n, &mut rng)];
    let emu = g.eval_emulated(&f64s(&inputs), &[c]);
    let exe = check(&dev, "batched frag", &g, &[c], &inputs, &emu, 1e-5);
    assert!(exe.program.kernels[0].fragment);
}

/// Conv as implicit GEMM: `w[o, (c,k)] · x[(c,k), t]`, the `x` operand
/// gathered through its guarded, dilated view (zero padding at staging),
/// with a fused bias epilogue.
#[test]
fn fragment_conv_implicit_gemm_with_epilogue() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(23);
    for &(ci, co, t, k, dil) in &[(5, 7, 37, 3, 1), (40, 33, 70, 7, 3), (64, 128, 200, 7, 1)] {
        let (mut g, y) = conv_graph(ci, co, t, k, dil);
        let bias = g.input(&[co, 1]);
        let bb = g.broadcast(bias, &[co, t]).unwrap();
        let z = g.map(&[y, bb], |v| v[0] + v[1]).unwrap();
        let inputs = vec![
            rand(ci * t, &mut rng),
            rand(co * ci * k, &mut rng),
            rand(co, &mut rng),
        ];
        f16_operands(&mut g, &[TensorId::from_index(0), TensorId::from_index(1)]);
        let emu = g.eval_emulated(&f64s(&inputs), &[z]);
        // unfused: the conv itself on fragments
        let mut gu = g.clone();
        gu.set_fragment(y, true);
        let exe = check(&dev, "frag conv", &gu, &[z], &inputs, &emu, 1e-5);
        assert!(exe.program.kernels.iter().any(|k| k.fragment));
        // fused: the map takes the conv as its epilogue core
        let mut gf = g.clone();
        gf.set_inline(y, true);
        gf.set_fragment(z, true);
        assert!(gf.fragment_plan(z).is_some(), "epilogue plan");
        let exe = check(&dev, "frag conv epilogue", &gf, &[z], &inputs, &emu, 1e-5);
        // the conv has no kernel of its own: it is the map's epilogue core
        assert!(exe.program.kernels.iter().all(|k| k.node != y));
        assert!(exe
            .program
            .kernels
            .iter()
            .any(|k| k.node == z && k.fragment));
    }
}

/// Transposed operands take the transposed device-load path (`a` stored
/// `[k, m]`, `b` stored `[n, k]`: q·kᵀ's layout), interior and ragged.
#[test]
fn fragment_transposed_operands() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(24);
    for (m, k, n) in [(128, 64, 256), (100, 72, 90)] {
        let mut g = TensorGraph::new();
        let at = g.input(&[k, m]);
        let bt = g.input(&[n, k]);
        let a = g.transpose(at, &[1, 0]).unwrap();
        let b = g.transpose(bt, &[1, 0]).unwrap();
        let c = g.matmul(a, b).unwrap();
        let inputs = vec![rand(m * k, &mut rng), rand(k * n, &mut rng)];
        let want = g.eval(&f64s(&inputs), &[c]);
        let (mut g, outs) = g.substitute_maps(&[c]);
        let c = outs[0];
        f16_operands(&mut g, &[TensorId::from_index(0), TensorId::from_index(1)]);
        g.set_fragment(c, true);
        assert!(g.fragment_plan(c).is_some());
        let emu = g.eval_emulated(&f64s(&inputs), &[c]);
        let exe = check(
            &dev,
            &format!("frag T {m}x{k}x{n}"),
            &g,
            &[c],
            &inputs,
            &emu,
            1e-5,
        );
        assert!(exe.program.kernels.iter().any(|k| k.fragment));
        let _ = want;
    }
}

/// The backend dimension: the same graph through WGSL and MSL, both
/// checked against the f64 reference.
#[test]
fn pick_backend_runs_both() {
    let Some(dev) = dev() else { return };
    let Ok(wgpu) = tang_gpu::GpuDevice::new_sync() else {
        return;
    };
    let mut cache = tang_gpu::KernelCache::new();
    let mut rng = tang_bench::Rng::new(25);
    let (mut g, c) = matmul_graph(96, 80, 112);
    g.schedule_auto(&[c]);
    let inputs = vec![rand(96 * 80, &mut rng), rand(80 * 112, &mut rng)];
    let want = g.eval(&f64s(&inputs), &[c]);
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let (_, w, m) = tang_bench::ir_msl::pick_backend(
        &wgpu,
        &mut cache,
        &dev,
        &g,
        &[c],
        &refs,
        &want,
        tang_compile::tensor::ErrorBudget::rel(1e-5),
        10,
    );
    assert!(w.is_some() && m.is_some());
}
