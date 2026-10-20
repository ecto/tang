//! KV-cache codec CLI (see `tang_compress::kvcodec`).
//!
//!     kvc encode IN.f32 OUT --shape HxTxD --kind k|v (--rmse r | --maxabs e | --bpv b)
//!         [--post] [--sink n] [--sinkdiv n] [--block n] [--noar] [--theta t] [--prefill n]
//!     kvc decode IN OUT.f32 [--range t0:t1] [--reps n]
//!
//! `--prefill n` fixes the header from the first n tokens and appends the rest
//! one token at a time (the streaming path); default is one shot.

use std::time::Instant;

use tang_compress::kvcodec::{step_for_bpv, Kind, KvEncoder, KvParams, KvStream};

fn usage() -> ! {
    eprintln!("usage: kvc encode IN OUT --shape HxTxD --kind k|v (--rmse r|--maxabs e|--bpv b) [--post] [--sink n] [--sinkdiv n] [--block n] [--noar] [--theta t] [--prefill n]\n       kvc decode IN OUT [--range t0:t1] [--reps n]");
    std::process::exit(2)
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .map(|i| args.get(i + 1).cloned().unwrap_or_else(|| usage()))
}

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        usage()
    }
    match args[1].as_str() {
        "encode" => {
            let x = read_f32(&args[2]);
            let shape: Vec<usize> = flag(&args, "--shape")
                .unwrap_or_else(|| usage())
                .split('x')
                .map(|s| s.parse().unwrap())
                .collect();
            let (h, n, d) = (shape[0], shape[1], shape[2]);
            let kind = match flag(&args, "--kind").as_deref() {
                Some("k") => Kind::K,
                Some("v") => Kind::V,
                _ => usage(),
            };
            let mut p = KvParams::new(kind, h, d, 1.0);
            p.pre_rope = !args.iter().any(|a| a == "--post");
            p.ar = !args.iter().any(|a| a == "--noar");
            p.gpu = args.iter().any(|a| a == "--gpu");
            if let Some(v) = flag(&args, "--sink") {
                p.sink = v.parse().unwrap();
            }
            if let Some(v) = flag(&args, "--sinkdiv") {
                p.sink_div = v.parse().unwrap();
            }
            if let Some(v) = flag(&args, "--block") {
                p.block = v.parse().unwrap();
            }
            if let Some(v) = flag(&args, "--theta") {
                p.theta = v.parse().unwrap();
            }
            let t = Instant::now();
            p.step = if let Some(r) = flag(&args, "--rmse") {
                r.parse::<f64>().unwrap() * 12f64.sqrt()
            } else if let Some(e) = flag(&args, "--maxabs") {
                // pre-RoPE pair rotation can move a per-element error by sqrt(2)
                let e: f64 = e.parse().unwrap();
                if kind == Kind::K && p.pre_rope {
                    e * std::f64::consts::SQRT_2
                } else {
                    2.0 * e
                }
            } else if let Some(b) = flag(&args, "--bpv") {
                step_for_bpv(p.clone(), &x, n, b.parse().unwrap())
            } else {
                usage()
            };
            let pre: usize = flag(&args, "--prefill")
                .map(|v| v.parse().unwrap())
                .unwrap_or(n);
            let slice = |t0: usize, t1: usize| -> Vec<f32> {
                let mut v = Vec::with_capacity(h * (t1 - t0) * d);
                for hh in 0..h {
                    v.extend_from_slice(&x[(hh * n + t0) * d..(hh * n + t1) * d]);
                }
                v
            };
            let mut e = KvEncoder::new(p.clone(), &slice(0, pre), pre);
            let ta = Instant::now();
            for t in pre..n {
                e.append(&slice(t, t + 1), 1);
            }
            let append_s = ta.elapsed().as_secs_f64();
            let bytes = e.to_bytes();
            let enc = t.elapsed().as_secs_f64();
            std::fs::write(&args[3], &bytes).unwrap();
            println!(
                "{{\"bytes\":{},\"bpv\":{},\"step\":{},\"encode_s\":{enc},\"append_s\":{append_s},\"appended\":{}}}",
                bytes.len(),
                8.0 * bytes.len() as f64 / x.len() as f64,
                p.step,
                n - pre
            );
        }
        "decode" => {
            let bytes = std::fs::read(&args[2]).unwrap();
            let s = KvStream::parse(&bytes).unwrap();
            let (t0, t1) = match flag(&args, "--range") {
                Some(r) => {
                    let (a, b) = r.split_once(':').unwrap();
                    (a.parse().unwrap(), b.parse().unwrap())
                }
                None => (0, s.tokens),
            };
            let reps: usize = flag(&args, "--reps")
                .map(|v| v.parse().unwrap())
                .unwrap_or(1);
            let t = Instant::now();
            let mut y = Vec::new();
            for _ in 0..reps {
                y = s.decode_range(t0, t1);
            }
            let dec = t.elapsed().as_secs_f64() / reps as f64;
            let out: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&args[3], out).unwrap();
            println!(
                "{{\"decode_s\":{dec},\"ns_per_value\":{}}}",
                1e9 * dec / y.len().max(1) as f64
            );
        }
        _ => usage(),
    }
}
