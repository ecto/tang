//! `corpus` — write the browser gate corpus (native): for every field and
//! target, a TCG1, a static TCG3 and a learned TCG3 stream, plus
//! `manifest.json` with the native CPU decode hash and native decode times.
//!
//! ```text
//! corpus FIELDS.txt OUT_DIR MODEL.tcm [--targets p60,p80,p100,m1e-3,m1e-4] [--threads 8]
//! ```
//! `FIELDS.txt`: one `path AxBxC` per line.

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Instant;
use cram_codec::codec::ErrorBound;
use cram_codec::gpudec::GpuDecoder;
use cram_codec::gpuwav::{self, GpuConfig};
use cram_codec::gpuwav3::{self, Gpu3Config, Model3};
use cram_codec::learned;
use cram_codec_wasm::{decode_cpu_bits, decoder_key, fnv1a_bits, format_of, Format};

fn opt(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| d.to_string())
}

fn best(reps: usize, mut f: impl FnMut() -> f64) -> f64 {
    (0..reps).map(|_| f()).fold(f64::INFINITY, f64::min)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = std::path::PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out).unwrap();
    let net = Arc::new(
        learned::Net::from_bytes(&std::fs::read(&args[2]).expect("model")).expect("parse model"),
    );
    learned::register(net.clone());
    let targets = opt(&args, "--targets", "p60,p80,p100,m1e-3,m1e-4");
    let threads: usize = opt(&args, "--threads", "8").parse().unwrap();
    let mut decs: Vec<((usize, usize), GpuDecoder)> = Vec::new();
    let mut man = String::from("[\n");
    for line in std::fs::read_to_string(&args[0])
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
    {
        let (path, d) = line.split_once(' ').unwrap();
        let dims: Vec<usize> = d.trim().split('x').map(|s| s.parse().unwrap()).collect();
        let name = std::path::Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .replace("-64x128x128", "");
        let b = std::fs::read(path).unwrap();
        let x: Vec<f32> = b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        assert_eq!(x.len(), dims.iter().product::<usize>(), "{path}");
        let (lo, hi) = x
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        let range = (hi - lo) as f64;
        let two_d = dims.len() == 2;
        for t in targets.split(',') {
            let bound = if let Some(p) = t.strip_prefix('p') {
                ErrorBound::Psnr(p.parse().unwrap())
            } else {
                ErrorBound::MaxAbs(t[1..].parse::<f64>().unwrap() * range)
            };
            let bk = if two_d { 128 } else { 32 };
            let mut streams: Vec<(&str, Vec<u8>)> = vec![
                (
                    "tcg1",
                    gpuwav::encode_f32_mt(
                        &x,
                        &dims,
                        &GpuConfig {
                            brick: if two_d { 512 } else { 64 },
                            lanes: 32,
                            ..GpuConfig::new(bound)
                        },
                        threads,
                    )
                    .0,
                ),
                (
                    "tcg3s",
                    gpuwav3::encode3_f32(
                        &x,
                        &dims,
                        &Gpu3Config {
                            block: bk,
                            block2d: bk,
                            lanes: 1,
                            ..Gpu3Config::new(bound)
                        },
                        threads,
                    )
                    .0,
                ),
                (
                    "tcg3l",
                    gpuwav3::encode3_f32(
                        &x,
                        &dims,
                        &Gpu3Config {
                            block: bk,
                            block2d: bk,
                            lanes: 8,
                            model: Model3::Pretrained(net.clone()),
                            ..Gpu3Config::new(bound)
                        },
                        threads,
                    )
                    .0,
                ),
            ];
            // --ship: the TCG3 streams again, carrying their decoder (learned: with the MLP)
            if args.iter().any(|a| a == "--ship") {
                let w: Vec<(&str, Vec<u8>)> = streams[1..]
                    .iter()
                    .map(|(f, s)| {
                        let ps = gpuwav3::parse3(s).unwrap();
                        (
                            if *f == "tcg3s" { "tcs3s" } else { "tcs3l" },
                            cram_codec::shipdec::wrap(
                                s,
                                &cram_codec::shipdec::builtin_program(
                                    ps.net.as_ref(),
                                    ps.net.is_some() && ps.ad.wa > 0,
                                ),
                            ),
                        )
                    })
                    .collect();
                streams.extend(w);
            }
            for (fmt, s) in streams {
                let file = format!("{name}_{t}_{fmt}.tc");
                std::fs::write(out.join(&file), &s).unwrap();
                let cpu = decode_cpu_bits(&s).unwrap();
                let h = fnv1a_bits(&cpu);
                let cpu_s = best(2, || {
                    let t0 = Instant::now();
                    decode_cpu_bits(&s).unwrap();
                    t0.elapsed().as_secs_f64()
                });
                let key = decoder_key(&s).unwrap();
                let key = (key.0, key.1.max(2).next_power_of_two());
                if !decs.iter().any(|d| d.0 == key) {
                    decs.push((key, GpuDecoder::new(key.0, key.1).expect("gpu")));
                }
                let g = &decs.iter().find(|d| d.0 == key).unwrap().1;
                let v3 = format_of(&s).unwrap() == Format::V3;
                let dec = |s: &[u8]| {
                    if v3 {
                        g.decode3(s).unwrap()
                    } else {
                        g.decode(s).unwrap()
                    }
                };
                let (gv, _) = dec(&s);
                assert_eq!(fnv1a_bits(&gv), h, "native GPU != CPU on {file}");
                let mut kern = f64::INFINITY;
                let gpu_s = best(3, || {
                    let t0 = Instant::now();
                    let (_, tm) = dec(&s);
                    kern = kern.min(tm.kernels_s);
                    t0.elapsed().as_secs_f64()
                });
                let _ = writeln!(
                    man,
                    "  {{\"file\":\"{file}\",\"field\":\"{name}\",\"target\":\"{t}\",\"format\":\"{fmt}\",\"dims\":{dims:?},\"cells\":{},\"bytes\":{},\"hash\":\"{h:016x}\",\"native_cpu1_s\":{cpu_s},\"native_gpu_s\":{gpu_s},\"native_gpu_kernels_s\":{kern}}},",
                    x.len(),
                    s.len()
                );
                eprintln!("{file} {} B gpu {gpu_s:.4}s cpu {cpu_s:.3}s", s.len());
            }
        }
    }
    if man.ends_with(",\n") {
        man.truncate(man.len() - 2);
        man.push('\n');
    }
    man.push_str("]\n");
    std::fs::write(out.join("manifest.json"), man).unwrap();
}
