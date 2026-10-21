//! `cram` — general-purpose compressor front door ([`cram_codec::container`]).
//!
//! ```text
//! cram c IN OUT [--hint auto|bytes|text|f32:AxBxC[:maxabs=E|psnr=DB|rmse=E]]
//! cram d IN OUT
//! cram bench IN [--hint ...]     # CSV: every arm alone, then the container
//! ```

use cram_codec::codec::ErrorBound;
use cram_codec::container::{self, Arm, Hint};
use std::time::Instant;

fn usage() -> ! {
    eprintln!("usage: cram c IN OUT [--hint H] | cram d IN OUT | cram bench IN [--hint H]\n  H = auto|bytes|text|f32:AxBxC[:maxabs=E|psnr=DB|rmse=E]");
    std::process::exit(2)
}

fn parse_hint(s: &str) -> Hint {
    match s {
        "auto" => Hint::Auto,
        "bytes" => Hint::Bytes,
        "text" => Hint::Text,
        _ => {
            let Some(rest) = s.strip_prefix("f32:") else {
                usage()
            };
            let mut parts = rest.splitn(2, ':');
            let dims: Vec<usize> = parts
                .next()
                .unwrap()
                .split('x')
                .map(|d| d.parse().unwrap_or_else(|_| usage()))
                .collect();
            let bound = parts.next().map(|b| {
                let (k, v) = b.split_once('=').unwrap_or_else(|| usage());
                let v: f64 = v.parse().unwrap_or_else(|_| usage());
                match k {
                    "maxabs" => ErrorBound::MaxAbs(v),
                    "psnr" => ErrorBound::Psnr(v),
                    "rmse" => ErrorBound::Rmse(v),
                    _ => usage(),
                }
            });
            Hint::F32Field { dims, bound }
        }
    }
}

fn hint_arg(args: &[String]) -> Hint {
    args.iter()
        .position(|a| a == "--hint")
        .map(|i| parse_hint(args.get(i + 1).unwrap_or_else(|| usage())))
        .unwrap_or(Hint::Auto)
}

fn read(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| {
        eprintln!("read {p}: {e}");
        std::process::exit(1)
    })
}

fn mbs(bytes: usize, s: f64) -> f64 {
    bytes as f64 / 1e6 / s.max(1e-9)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("c") if args.len() >= 3 => {
            let x = read(&args[1]);
            let t = Instant::now();
            let (c, rep) = container::compress_report(&x, &hint_arg(&args)).unwrap_or_else(|e| {
                eprintln!("cram: {e}");
                std::process::exit(1)
            });
            let s = t.elapsed().as_secs_f64();
            std::fs::write(&args[2], &c).unwrap();
            let tried: Vec<String> = rep
                .tried
                .iter()
                .map(|(a, n)| format!("{}={n}", a.name()))
                .collect();
            println!(
                "arm={} in={} out={} ratio={:.4} MB/s={:.2} tried=[{}]",
                rep.pick.name(),
                x.len(),
                c.len(),
                x.len() as f64 / c.len() as f64,
                mbs(x.len(), s),
                tried.join(" ")
            );
        }
        Some("d") if args.len() >= 3 => {
            let c = read(&args[1]);
            let t = Instant::now();
            let x = container::decompress(&c).unwrap_or_else(|e| {
                eprintln!("cram: {e}");
                std::process::exit(1)
            });
            let s = t.elapsed().as_secs_f64();
            std::fs::write(&args[2], &x).unwrap();
            let arm = Arm::from_tag(c[4]).unwrap();
            println!(
                "arm={} in={} out={} ratio={:.4} MB/s={:.2}",
                arm.name(),
                c.len(),
                x.len(),
                x.len() as f64 / c.len() as f64,
                mbs(x.len(), s)
            );
        }
        Some("bench") if args.len() >= 2 => {
            let x = read(&args[1]);
            let hint = hint_arg(&args);
            let name = std::path::Path::new(&args[1])
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let mut arms = container::LOSSLESS_ARMS.to_vec();
            if matches!(hint, Hint::F32Field { bound: Some(_), .. }) {
                arms.push(Arm::CramLossy);
            }
            for arm in arms {
                let t = Instant::now();
                let p = container::encode_arm(arm, &x, &hint).unwrap();
                let ce = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let y = container::decode_arm(arm, &p, x.len()).unwrap();
                let de = t.elapsed().as_secs_f64();
                assert!(!arm.is_lossless() || y == x, "{arm:?} did not round-trip");
                println!(
                    "{name},{},{},{},{:.4},{:.2},{:.2},",
                    arm.name(),
                    x.len(),
                    p.len(),
                    x.len() as f64 / p.len() as f64,
                    mbs(x.len(), ce),
                    mbs(x.len(), de)
                );
            }
            let t = Instant::now();
            let (c, rep) = container::compress_report(&x, &hint).unwrap();
            let ce = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let y = container::decompress(&c).unwrap();
            let de = t.elapsed().as_secs_f64();
            assert!(!rep.pick.is_lossless() || y == x, "container did not round-trip");
            println!(
                "{name},cram,{},{},{:.4},{:.2},{:.2},{}",
                x.len(),
                c.len(),
                x.len() as f64 / c.len() as f64,
                mbs(x.len(), ce),
                mbs(x.len(), de),
                rep.pick.name()
            );
        }
        _ => usage(),
    }
}
