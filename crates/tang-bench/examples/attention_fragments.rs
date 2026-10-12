//! Derived flash attention on the native Metal backend: the best f32
//! schedule (`schedule_auto`, MSL) vs simdgroup fragments (q/k/v f16, f32
//! accumulate). d = dv = 64; 1, 4 and 8 heads (`ATT_HEADS`); causal and not.
//! Kernel-only, 20 launches per sample, best of three; each fragment run
//! is checked against the f32 kernel.
//!
//!   cargo run --release -p tang-bench --features metal --example attention_fragments [n...]
use tang_bench::ir_gpu::{flash_attention_heads_auto, flash_attention_heads_fragment};
use tang_bench::ir_msl::{Executable, MslDevice};

fn main() {
    let dev = MslDevice::new().expect("metal");
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let ns = if args.is_empty() {
        vec![512, 1024, 2048, 4096]
    } else {
        args
    };
    let d = 64;
    println!("{} — attention, d = dv = {d}, kernel-only ms", dev.name);
    println!("heads causal     n   f32-auto   fragment  speedup  TFLOP/s(frag)  max|diff|");
    let heads: Vec<usize> = std::env::var("ATT_HEADS")
        .unwrap_or_else(|_| "1,4,8".into())
        .split(',')
        .map(|h| h.parse().unwrap())
        .collect();
    let causals: Vec<bool> = match std::env::var("ATT_CAUSAL").as_deref() {
        Ok("0") => vec![false],
        Ok("1") => vec![true],
        _ => vec![false, true],
    };
    let samples: usize = std::env::var("ATT_SAMPLES").map_or(3, |s| s.parse().unwrap());
    for &h in &heads {
        for &causal in &causals {
            for &n in &ns {
                let mut rng = tang_bench::Rng::new(5);
                let mut r = |k: usize| (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect::<Vec<f32>>();
                let inputs = [r(h * n * d), r(h * n * d), r(h * n * d)];
                let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
                let iters = 20;
                let best = |exe: &Executable| {
                    (0..samples)
                        .map(|_| exe.time(&dev, iters))
                        .fold(f64::INFINITY, f64::min)
                };
                let (g, y) = flash_attention_heads_auto(h, 1, n, d, d, causal);
                let e32 = Executable::new(&dev, g.to_msl(&[y]), &refs);
                e32.run(&dev);
                let (gf, yf) = flash_attention_heads_fragment(h, 1, n, d, d, causal);
                let ef = Executable::new(&dev, gf.to_msl(&[yf]), &refs);
                ef.run(&dev);
                assert!(ef.program.kernels[0].fragment);
                let diff = e32
                    .read(y)
                    .iter()
                    .zip(ef.read(yf))
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                let (t32, tf) = (best(&e32), best(&ef));
                let flop = 4.0 * (h * n * n * d) as f64 * if causal { 0.5 } else { 1.0 };
                println!(
                    "{h:5} {causal:6} {n:5} {t32:10.3} {tf:10.3} {:7.2}x {:10.2} {diff:12.2e}",
                    t32 / tf,
                    flop / tf / 1e9
                );
            }
        }
    }
}
