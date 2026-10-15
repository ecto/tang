//! `ExprGraph::to_wgsl_split_f64` on a device with `SHADER_F64` (skipped
//! elsewhere): every op, including the WGSL f64 `sin`/`exp2`/`log2`/`atan2`
//! helpers, split across kernels, against f64 evaluation on the CPU.

use tang_expr::graph::ExprGraph;
use tang_gpu::{GpuBuffer, GpuDevice, KernelCache};

#[test]
fn split_f64_matches_cpu() {
    let Ok(device) = GpuDevice::new_sync() else {
        return;
    };
    if !device.shader_f64 {
        eprintln!("skipped: {} has no SHADER_F64", device.name);
        return;
    }
    let mut g = ExprGraph::new();
    let (x, y) = (g.var(0), g.var(1));
    let xy = g.mul(x, y);
    let s = g.sin(xy);
    let ax = g.mul(x, x);
    let e = g.exp2(ax);
    let yy = g.mul(y, y);
    let one = g.lit(1.0);
    let pos = g.add(yy, one);
    let l = g.log2(pos);
    let at = g.atan2(y, x);
    let sq = g.sqrt(pos);
    let r = g.recip(sq);
    let n = g.neg(at);
    let sel = g.select(x, s, n);
    let a1 = g.add(sel, e);
    let a2 = g.mul(a1, r);
    let a3 = g.add(a2, l);
    let outs = [a3, s, e, l, at, xy];
    let split = g.to_wgsl_split_f64(&outs, 2, 4);
    assert!(
        split.parts.len() > 2,
        "split into {} parts",
        split.parts.len()
    );
    let pts: Vec<[f64; 2]> = (0..512)
        .map(|i| {
            let t = i as f64 / 511.0;
            [-30.0 + 60.0 * t, 3.0 * (17.0 * t).sin()]
        })
        .collect();
    let host: Vec<f64> = pts.iter().flat_map(|p| p.iter().copied()).collect();
    let as32 = |v: &[f64]| bytemuck::cast_slice::<f64, f32>(v).to_vec();
    let inputs = GpuBuffer::from_slice(&device, &as32(&host));
    let scratch = GpuBuffer::from_slice(&device, &vec![0.0f32; 2 * pts.len() * split.n_scratch]);
    let out = GpuBuffer::from_slice(&device, &vec![0.0f32; 2 * pts.len() * outs.len()]);
    let mut cache = KernelCache::new();
    let b = pts.len() as u32;
    for src in &split.parts {
        cache.dispatch_buffers(
            &device,
            src,
            &[&inputs, &scratch, &out],
            &[1, 2],
            &[b, 0, 0, 0],
            b,
            256,
        );
    }
    cache.flush(&device);
    let got32 = out.to_vec_sync(&device);
    let got: &[f64] = bytemuck::cast_slice(&got32);
    let mut worst: f64 = 0.0;
    for (e, p) in pts.iter().enumerate() {
        for (k, o) in outs.iter().enumerate() {
            let want: f64 = g.eval(*o, p);
            let err = (got[e * outs.len() + k] - want).abs() / want.abs().max(1.0);
            worst = worst.max(err);
        }
    }
    eprintln!(
        "{}: {} parts, worst relative error {worst:e}",
        device.name,
        split.parts.len()
    );
    assert!(worst < 1e-13, "worst {worst:e}");
}
