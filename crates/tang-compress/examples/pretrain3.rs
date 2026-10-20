//! `pretrain3` — train the TCG3 learned model offline.
//!
//! ```text
//! pretrain3 OUT.tcm --hidden 32,16 --psnr 40,60,80,100 --lanes 8 --block 32 a.f32:64x128x128 b.f32:64x128x128 ...
//! ```
use tang_compress::codec::ErrorBound;
use tang_compress::gpuwav3::{samples3, Gpu3Config};
use tang_compress::learned::{mean_bits_float, train, TrainConfig};

fn opt(args: &[String], k: &str, d: &str) -> String {
    args.iter().position(|a| a == k).map(|i| args[i + 1].clone()).unwrap_or_else(|| d.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = &args[0];
    let hidden: Vec<usize> = opt(&args, "--hidden", "32,16").split(',').map(|v| v.parse().unwrap()).collect();
    let psnr: Vec<f64> = opt(&args, "--psnr", "40,50,60,70,80,90,100").split(',').map(|v| v.parse().unwrap()).collect();
    let lanes: usize = opt(&args, "--lanes", "8").parse().unwrap();
    let block: usize = opt(&args, "--block", "32").parse().unwrap();
    let per: usize = opt(&args, "--per", "100000").parse().unwrap();
    let epochs: usize = opt(&args, "--epochs", "8").parse().unwrap();
    let mut smp = Vec::new();
    for a in args.iter().filter(|a| a.contains(".f32:")) {
        let (path, d) = a.split_once(':').unwrap();
        let dims: Vec<usize> = d.split('x').map(|s| s.parse().unwrap()).collect();
        let b = std::fs::read(path).unwrap();
        let x: Vec<f32> = b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
        for &p in &psnr {
            let cfg = Gpu3Config { block, block2d: block * 4, lanes, ..Gpu3Config::new(ErrorBound::Psnr(p)) };
            let s = samples3(&x, &dims, &cfg, 16, per);
            eprintln!("{path} {p} dB: {} samples", s.len());
            smp.extend(s);
        }
    }
    let tc = TrainConfig { hidden, epochs, per_epoch: 3_000_000, batch: 256, lr: 3e-3, apm: false, seed: 7 };
    let t = std::time::Instant::now();
    let net = train(&smp, &tc);
    eprintln!("trained on {} samples in {:.1}s: {:.4} bits/sample (float view), {} params", smp.len(), t.elapsed().as_secs_f64(), mean_bits_float(&net, &smp), net.num_params());
    std::fs::write(out, net.to_bytes()).unwrap();
}
