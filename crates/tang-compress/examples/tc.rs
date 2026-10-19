//! `tc` — encode/decode raw little-endian f32 files with the tang-compress
//! wavelet or Tucker codecs (the QTT codec has its own `qtt` example).
//!
//! ```text
//! tc encode IN.f32 OUT.tc --dims 64x128x128 --codec wavelet --psnr 60
//! tc encode IN.f32 OUT.tc --dims 64x128x128 --codec wavelet --max-rel 1e-4
//! tc decode IN.tc OUT.f32
//! ```
//!
//! `encode` prints one JSON line (bytes, bpv, self-check errors, encode
//! seconds); `decode` prints the decode seconds. Options: `--rho r`,
//! `--model hand|overfit|pretrained:PATH`.

use std::time::Instant;
use tang_compress::codec::ErrorBound;
use tang_compress::{decode_any, wavcodec, Decoded};

fn usage() -> ! {
    eprintln!("usage: tc encode IN.f32 OUT --dims AxBxC --codec wavelet|tucker BOUND [--rho r] [--model m]\n       tc decode IN OUT.f32");
    std::process::exit(2)
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("encode") if args.len() >= 3 => {
            let x = read_f32(&args[1]);
            let range = {
                let (a, b) = x.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
                (b - a) as f64
            };
            let mut dims = vec![x.len()];
            let mut bound = None;
            let mut codec = "wavelet".to_string();
            let mut rho = None;
            let mut model = "hand".to_string();
            let mut i = 3;
            while i < args.len() {
                let v = args.get(i + 1).cloned().unwrap_or_else(|| usage());
                match args[i].as_str() {
                    "--dims" => dims = v.split('x').map(|s| s.parse().unwrap()).collect(),
                    "--psnr" => bound = Some(ErrorBound::Psnr(v.parse().unwrap())),
                    "--rmse" => bound = Some(ErrorBound::Rmse(v.parse().unwrap())),
                    "--max-abs" => bound = Some(ErrorBound::MaxAbs(v.parse().unwrap())),
                    "--max-rel" => bound = Some(ErrorBound::MaxAbs(v.parse::<f64>().unwrap() * range)),
                    "--codec" => codec = v,
                    "--rho" => rho = Some(v.parse().unwrap()),
                    "--model" => model = v,
                    _ => usage(),
                }
                i += 2;
            }
            let bound = bound.unwrap_or_else(|| usage());
            let t = Instant::now();
            let (bytes, json) = match codec.as_str() {
                "wavelet" => {
                    let mut cfg = wavcodec::WavConfig::new(bound);
                    if let Some(r) = rho {
                        cfg.rho = r;
                    }
                    cfg.model = tang_compress::learned::parse_model_arg(&model);
                    let (b, s) = wavcodec::encode_f32(&x, &dims, &cfg);
                    let j = format!(
                        "\"coef_bytes\":{},\"model_bytes\":{},\"outlier_bytes\":{},\"n_outliers\":{},\"delta\":{},\"rmse\":{},\"max_abs\":{},\"passes\":{}",
                        s.coef_bytes, s.model_bytes, s.outlier_bytes, s.n_outliers, s.delta, s.rmse, s.max_abs, s.passes
                    );
                    (b, j)
                }
                "tucker" => {
                    let (b, s) = tang_compress::tucker::encode_f32(&x, &dims, bound);
                    let j = format!(
                        "\"core_bytes\":{},\"factor_bytes\":{},\"ranks\":{:?},\"delta\":{},\"rmse\":{},\"max_abs\":{}",
                        s.core_bytes, s.factor_bytes, s.ranks, s.delta, s.rmse, s.max_abs
                    );
                    (b, j)
                }
                _ => usage(),
            };
            let enc = t.elapsed().as_secs_f64();
            std::fs::write(&args[2], &bytes).unwrap();
            println!("{{\"bytes\":{},\"bpv\":{},\"encode_s\":{enc},{json}}}", bytes.len(), 8.0 * bytes.len() as f64 / x.len() as f64);
        }
        Some("decode") if args.len() == 3 => {
            let bytes = std::fs::read(&args[1]).unwrap();
            let t = Instant::now();
            let Decoded::F32 { data, .. } = decode_any(&bytes).expect("decode") else { panic!("not an f32 stream") };
            let dec = t.elapsed().as_secs_f64();
            let out: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&args[2], out).unwrap();
            println!("{{\"decode_s\":{dec}}}");
        }
        _ => usage(),
    }
}
