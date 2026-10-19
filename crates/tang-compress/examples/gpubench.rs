//! `gpubench` — tc-gpu measurements.
//!
//! ```text
//! gpubench ratio FIELD.f32 AxBxC [--bricks 32,64,128] [--lanes 32] [--threads 16]
//! ```
//! `ratio` prints one JSON line per (target, codec): serial tc-wavelet (hand)
//! and tc-gpu per brick size, each verified by decoding.

use std::time::Instant;
use tang_compress::codec::ErrorBound;
use tang_compress::gpuwav::{self, GpuConfig};
use tang_compress::{wavcodec, Decoded};

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

fn err(x: &[f32], y: &[f32]) -> (f64, f64) {
    let mut se = 0.0;
    let mut mx = 0.0f64;
    for (a, b) in x.iter().zip(y) {
        let e = *a as f64 - *b as f64;
        se += e * e;
        mx = mx.max(e.abs());
    }
    ((se / x.len() as f64).sqrt(), mx)
}

fn opt(args: &[String], k: &str, d: &str) -> String {
    args.iter().position(|a| a == k).map(|i| args[i + 1].clone()).unwrap_or_else(|| d.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args[0].as_str();
    let x = read_f32(&args[1]);
    let dims: Vec<usize> = args[2].split('x').map(|s| s.parse().unwrap()).collect();
    let name = std::path::Path::new(&args[1]).file_stem().unwrap().to_string_lossy().to_string();
    let bricks: Vec<usize> = opt(&args, "--bricks", "64").split(',').map(|s| s.parse().unwrap()).collect();
    let lanes: usize = opt(&args, "--lanes", "32").parse().unwrap();
    let threads: usize = opt(&args, "--threads", "16").parse().unwrap();
    let targets = opt(&args, "--targets", "p60,p80,p100,m1e-3,m1e-4");
    let serial = !args.iter().any(|a| a == "--no-serial");
    let (lo, hi) = x.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let range = (hi - lo) as f64;
    match mode {
        "ratio" => {
            for t in targets.split(',') {
                let bound = if let Some(p) = t.strip_prefix('p') {
                    ErrorBound::Psnr(p.parse().unwrap())
                } else {
                    ErrorBound::MaxAbs(t[1..].parse::<f64>().unwrap() * range)
                };
                if serial {
                    let t0 = Instant::now();
                    let (b, _) = wavcodec::encode_f32(&x, &dims, &wavcodec::WavConfig::new(bound));
                    let enc_s = t0.elapsed().as_secs_f64();
                    let t0 = Instant::now();
                    let Decoded::F32 { data, .. } = wavcodec::decode(&b).unwrap() else { panic!() };
                    let dec_s = t0.elapsed().as_secs_f64();
                    let (rmse, mx) = err(&x, &data);
                    println!(
                        "{{\"field\":\"{name}\",\"target\":\"{t}\",\"codec\":\"tc-wavelet\",\"brick\":0,\"bpv\":{},\"psnr\":{},\"max_abs_rel\":{},\"enc_s\":{enc_s},\"dec_s\":{dec_s}}}",
                        8.0 * b.len() as f64 / x.len() as f64,
                        20.0 * (range / rmse).log10(),
                        mx / range
                    );
                }
                for &bk in &bricks {
                    let cfg = GpuConfig { brick: bk, lanes, ..GpuConfig::new(bound) };
                    let t0 = Instant::now();
                    let (b, st) = gpuwav::encode_f32_mt(&x, &dims, &cfg, threads);
                    let enc_s = t0.elapsed().as_secs_f64();
                    let t0 = Instant::now();
                    let (_, y) = gpuwav::decode_f32(&b, 1).unwrap();
                    let dec_s = t0.elapsed().as_secs_f64();
                    let (rmse, mx) = err(&x, &y);
                    if let ErrorBound::MaxAbs(bb) = bound {
                        assert!(mx <= bb, "{name} {t}: {mx} > {bb}");
                    }
                    println!(
                        "{{\"field\":\"{name}\",\"target\":\"{t}\",\"codec\":\"tc-gpu\",\"brick\":{bk},\"lanes\":{lanes},\"bpv\":{},\"psnr\":{},\"max_abs_rel\":{},\"enc_s\":{enc_s},\"dec_s\":{dec_s},\"hdr\":{},\"idx\":{},\"outl\":{},\"esc\":{}}}",
                        st.bits_per_value,
                        20.0 * (range / rmse).log10(),
                        mx / range,
                        st.header_bytes,
                        st.index_bytes,
                        st.outlier_bytes,
                        st.n_escapes
                    );
                }
            }
        }
        _ => panic!("mode"),
    }
}
