//! Offline pretraining of the learned coefficient model.
//!
//! `pretrain OUT.tcm --dims 64x128x128 --psnr 40,60,80,100 [--hidden 64,32]
//!  [--per 150000] [--epochs 10] FILE.f32...`
//!
//! For every file × PSNR target the wavelet encoder's rate control picks `Δ`,
//! the coefficients are quantized and in-band predicted exactly as when
//! coding, and a seeded random subset of `(features, residual)` samples is
//! kept. The network is trained on the union and written to OUT.
use std::sync::Arc;
use cram_codec::codec::ErrorBound;
use cram_codec::coefcode::fit_predictors;
use cram_codec::learned::{collect, mean_bits_float, train, Net, Sample, TrainConfig};
use cram_codec::wavcodec::{encode_f32, quantize, Coeffs, WavConfig};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let out = &a[0];
    let mut dims = vec![64, 128, 128];
    let mut psnrs = vec![40.0, 60.0, 80.0, 100.0];
    let mut cfg = TrainConfig::pretrain();
    let mut per = 150_000usize;
    let mut files = Vec::new();
    let mut i = 1;
    while i < a.len() {
        match a[i].as_str() {
            "--dims" => dims = a[i + 1].split('x').map(|s| s.parse().unwrap()).collect(),
            "--psnr" => psnrs = a[i + 1].split(',').map(|s| s.parse().unwrap()).collect(),
            "--hidden" => cfg.hidden = a[i + 1].split(',').map(|s| s.parse().unwrap()).collect(),
            "--per" => per = a[i + 1].parse().unwrap(),
            "--epochs" => cfg.epochs = a[i + 1].parse().unwrap(),
            "--per-epoch" => cfg.per_epoch = a[i + 1].parse().unwrap(),
            f => {
                files.push(f.to_string());
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    let mut samples: Vec<Sample> = Vec::new();
    let mut seed = 12345u64;
    for f in &files {
        let b = std::fs::read(f).unwrap();
        let x: Vec<f32> = b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let co = Coeffs::new(&xd, &dims);
        for &p in &psnrs {
            let wc = WavConfig::new(ErrorBound::Psnr(p));
            let (_, st) = encode_f32(&x, &dims, &wc);
            let (q, _) = quantize(&co, st.delta, wc.rho);
            let preds = fit_predictors(&co.p, &q);
            let mut s = collect(&co.p, &q, &preds);
            // seeded partial Fisher–Yates subsample
            let k = per.min(s.len());
            for j in 0..k {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = j + (seed >> 11) as usize % (s.len() - j);
                s.swap(j, r);
            }
            s.truncate(k);
            eprintln!(
                "{f} psnr {p}: delta {:.4e}, hand {:.4} bpv, {} samples",
                st.delta,
                st.bits_per_value,
                s.len()
            );
            samples.extend(s);
        }
    }
    let t = std::time::Instant::now();
    let net = train(&samples, &cfg);
    eprintln!(
        "trained {} params on {} samples in {:.1}s: {:.4} bits/sample (float view)",
        net.num_params(),
        samples.len(),
        t.elapsed().as_secs_f64(),
        mean_bits_float(&net, &samples)
    );
    std::fs::write(out, Arc::new(net).to_bytes()).unwrap();
    let _ = Net::from_bytes(&std::fs::read(out).unwrap()).unwrap();
}
