//! Tensor-core matmul tiles by operand format: TF32 (f32 operands)
//! against m16n8k16 f16 / bf16 (half operands, fp32 accumulation), and
//! f16 activations with f32 weights converted while staging, in every
//! block shape. Fastest single launch each (`Program::time_kernel`).
//! `compile` runs nvrtc only (no GPU).
use tang_compile::tensor::cuda::MMA_SHAPES;
use tang_compile::tensor::{Format, Precision, TensorGraph};
use tang_gpu::program::{Program, Runtime};

fn main() {
    let compile_only = std::env::args().nth(1).as_deref() == Some("compile");
    let rt = (!compile_only).then(|| Runtime::select().expect("gpu"));
    let cases = [
        (128usize, 576usize, 1536usize),
        (512, 576, 1536),
        (512, 1536, 576),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
    ];
    let mut rng = tang_bench::Rng::new(3);
    for (m, k, n) in cases {
        let mut g = TensorGraph::new();
        let a = g.input(&[m, k]);
        let b = g.input(&[k, n]);
        let c = g.matmul(a, b).unwrap();
        let (mut g, outs) = g.substitute_maps(&[c]);
        g.schedule_auto(&outs);
        let c = outs[0];
        let av: Vec<f32> = (0..m * k).map(|_| rng.f32() - 0.5).collect();
        let bv: Vec<f32> = (0..k * n).map(|_| rng.f32() - 0.5).collect();
        let configs = [
            ("tf32", Format::F32, Format::F32),
            ("f16", Format::F16, Format::F16),
            ("bf16", Format::BF16, Format::BF16),
            ("f16 x f32", Format::F16, Format::F32),
        ];
        let mut line = format!("{m}x{k}x{n}:");
        for (name, fa, fb) in configs {
            let mut v = g.clone();
            v.set_precision(a, Precision::storage(fa));
            v.set_precision(b, Precision::storage(fb));
            let mut best = (f64::INFINITY, 0);
            for s in 0..MMA_SHAPES.len() {
                let p = v.to_cuda_variant(&[c], Some(s));
                if compile_only {
                    tang_gpu::cuda::precompile(
                        p.kernels.iter().map(|k| k.source.as_str()),
                        "sm_86",
                    );
                    continue;
                }
                let rt = rt.as_ref().unwrap();
                let p = Program::new(rt, p, &[&av, &bv]);
                let t = p.time_kernel(rt, 0, 20);
                if t < best.0 {
                    best = (t, s);
                }
            }
            if !compile_only {
                let tflops = 2.0 * (m * k * n) as f64 / best.0 / 1e9;
                line.push_str(&format!(
                    "  {name} {:.4} ms (shape {}, {tflops:.1} TFLOP/s)",
                    best.0, best.1
                ));
            }
        }
        println!("{line}");
    }
}
