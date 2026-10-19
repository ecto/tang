//! `qtt` — encode/decode raw little-endian f32 files with the QTT codec.
//!
//! ```text
//! cargo run --release -p tang-compress --example qtt -- encode in.f32 out.qtt --dims 256x384x384 --psnr 60
//! cargo run --release -p tang-compress --example qtt -- decode out.qtt back.f32
//! cargo run --release -p tang-compress --example qtt -- bench in.f32 --dims 512x512 --max-rel 1e-3
//! ```
//!
//! Bounds: `--psnr dB` (range-normalised), `--rmse x`, `--max-abs x`,
//! `--max-rel r` (max-abs = r·range). Options: `--order morton|row`,
//! `--pad mixed|pow2`, `--gd` (Gavish–Donoho rank cap), `--max-rank n`.
//! `encode` and `bench` print one JSON line of stats (sizes, ranks, errors,
//! single-thread wall times).

use std::time::Instant;
use tang_compress::{decode, encode_f32, Config, Decoded, DigitOrder, ErrorBound, PadPolicy};

fn usage() -> ! {
    eprintln!("usage: qtt encode IN.f32 OUT.qtt --dims AxBxC BOUND [opts]\n       qtt decode IN.qtt OUT.f32\n       qtt bench IN.f32 --dims AxBxC BOUND [opts]");
    std::process::exit(2)
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn parse(args: &[String], data: &[f32]) -> (Vec<usize>, Config) {
    let mut dims = None;
    let mut bound = None;
    let mut order = DigitOrder::Morton;
    let mut pad = PadPolicy::MixedRadix;
    let mut gd = false;
    let mut max_rank = 512;
    let range = {
        let (a, b) = data
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        (b - a) as f64
    };
    let mut i = 0;
    while i < args.len() {
        let v = || args.get(i + 1).cloned().unwrap_or_else(|| usage());
        match args[i].as_str() {
            "--dims" => {
                dims = Some(
                    v().split('x')
                        .map(|s| s.parse().unwrap())
                        .collect::<Vec<usize>>(),
                )
            }
            "--psnr" => bound = Some(ErrorBound::Psnr(v().parse().unwrap())),
            "--rmse" => bound = Some(ErrorBound::Rmse(v().parse().unwrap())),
            "--max-abs" => bound = Some(ErrorBound::MaxAbs(v().parse().unwrap())),
            "--max-rel" => bound = Some(ErrorBound::MaxAbs(v().parse::<f64>().unwrap() * range)),
            "--order" => {
                order = match v().as_str() {
                    "morton" => DigitOrder::Morton,
                    "row" => DigitOrder::RowMajor,
                    _ => usage(),
                }
            }
            "--pad" => {
                pad = match v().as_str() {
                    "mixed" => PadPolicy::MixedRadix,
                    "pow2" => PadPolicy::Pow2,
                    _ => usage(),
                }
            }
            "--max-rank" => max_rank = v().parse().unwrap(),
            "--gd" => {
                gd = true;
                i += 1;
                continue;
            }
            _ => usage(),
        }
        i += 2;
    }
    let dims = dims.unwrap_or_else(|| vec![data.len()]);
    assert_eq!(
        dims.iter().product::<usize>(),
        data.len(),
        "--dims does not match file size"
    );
    let mut cfg = Config::new(bound.unwrap_or_else(|| usage()));
    cfg.order = order;
    cfg.pad = pad;
    cfg.gavish_donoho = gd;
    cfg.max_rank = max_rank;
    (dims, cfg)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("encode") if args.len() >= 3 => {
            let x = read_f32(&args[1]);
            let (dims, cfg) = parse(&args[3..], &x);
            let t = Instant::now();
            let (bytes, st) = encode_f32(&x, &dims, &cfg);
            let enc = t.elapsed().as_secs_f64();
            std::fs::write(&args[2], &bytes).unwrap();
            println!("{}", json(&st, enc, None));
        }
        Some("decode") if args.len() == 3 => {
            let bytes = std::fs::read(&args[1]).unwrap();
            let t = Instant::now();
            let Decoded::F32 { data, .. } = decode(&bytes).expect("decode") else {
                panic!("not an f32 stream")
            };
            let dec = t.elapsed().as_secs_f64();
            let out: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&args[2], out).unwrap();
            println!("{{\"decode_s\":{dec}}}");
        }
        Some("bench") if args.len() >= 2 => {
            let x = read_f32(&args[1]);
            let (dims, cfg) = parse(&args[2..], &x);
            let t = Instant::now();
            let (bytes, st) = encode_f32(&x, &dims, &cfg);
            let enc = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let Decoded::F32 { data, .. } = decode(&bytes).unwrap() else {
                unreachable!()
            };
            let dec = t.elapsed().as_secs_f64();
            let mx = x
                .iter()
                .zip(&data)
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .fold(0.0, f64::max);
            assert_eq!(
                mx, st.max_abs,
                "decode differs from the encoder's self-check"
            );
            println!("{}", json(&st, enc, Some(dec)));
        }
        _ => usage(),
    }
}

fn json(st: &tang_compress::EncodeStats, enc: f64, dec: Option<f64>) -> String {
    let ranks: Vec<String> = st.ranks.iter().map(|r| r.to_string()).collect();
    let modes: Vec<String> = st.modes.iter().map(|r| r.to_string()).collect();
    let psnr = 20.0 * (st.range / st.rmse).log10();
    format!(
        "{{\"bytes\":{},\"bpv\":{},\"header_bytes\":{},\"core_bytes\":{},\"residual_bytes\":{},\"n_residual\":{},\"rmse\":{},\"psnr\":{},\"max_abs\":{},\"range\":{},\"delta\":{},\"passes\":{},\"encode_s\":{},\"decode_s\":{},\"ranks\":[{}],\"modes\":[{}]}}",
        st.bytes, st.bits_per_value, st.header_bytes, st.core_bytes, st.residual_bytes, st.n_residual,
        st.rmse, if psnr.is_finite() { psnr } else { 999.0 }, st.max_abs, st.range, st.delta, st.passes, enc,
        dec.map_or("null".into(), |d| d.to_string()), ranks.join(","), modes.join(",")
    )
}
