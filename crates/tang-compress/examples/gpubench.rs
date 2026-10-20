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
    let lanes: usize = opt(&args, "--lanes", "16").parse().unwrap();
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
                    let cfg = GpuConfig { brick: bk, lanes, adaptive: args.iter().any(|a| a == "--adaptive"), ..GpuConfig::new(bound) };
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
        "speed" => {
            // encode once, then time decoders on the same stream
            let t = opt(&args, "--target", "p80");
            let bound = if let Some(p) = t.strip_prefix('p') {
                ErrorBound::Psnr(p.parse().unwrap())
            } else {
                ErrorBound::MaxAbs(t[1..].parse::<f64>().unwrap() * range)
            };
            let reps: usize = opt(&args, "--reps", "3").parse().unwrap();
            for &bk in &bricks {
                let cfg = GpuConfig { brick: bk, lanes, adaptive: args.iter().any(|a| a == "--adaptive"), ..GpuConfig::new(bound) };
                let t0 = Instant::now();
                let (b, st) = gpuwav::encode_f32_mt(&x, &dims, &cfg, threads);
                let enc_s = t0.elapsed().as_secs_f64();
                let gb = x.len() as f64 * 4.0 / 1e9;
                let best = |f: &mut dyn FnMut() -> f64| (0..reps).map(|_| f()).fold(f64::INFINITY, f64::min);
                let ps = gpuwav::parse(&b).unwrap();
                let mut cpu_bits = Vec::new();
                let cpu1 = best(&mut || {
                    let t0 = Instant::now();
                    cpu_bits = gpuwav::decode_box_parsed(&ps, &b, [0; 3], ps.d3, 1).unwrap();
                    t0.elapsed().as_secs_f64()
                });
                let cpumt = best(&mut || {
                    let t0 = Instant::now();
                    let v = gpuwav::decode_box_parsed(&ps, &b, [0; 3], ps.d3, threads).unwrap();
                    assert!(v == cpu_bits);
                    t0.elapsed().as_secs_f64()
                });
                // random access: 64^3 (or 64^2) box at the field centre
                let lo: [usize; 3] = std::array::from_fn(|a| if ps.d3[a] > 64 { ps.d3[a] / 2 - 32 } else { 0 });
                let hi: [usize; 3] = std::array::from_fn(|a| (lo[a] + 64).min(ps.d3[a]));
                let mut sub = Vec::new();
                let ra = best(&mut || {
                    let t0 = Instant::now();
                    sub = gpuwav::decode_box_bits(&b, lo, hi, 1).unwrap();
                    t0.elapsed().as_secs_f64()
                });
                let ra_mt = best(&mut || {
                    let t0 = Instant::now();
                    let _ = gpuwav::decode_box_bits(&b, lo, hi, threads).unwrap();
                    t0.elapsed().as_secs_f64()
                });
                let mut k = 0;
                for z in lo[0]..hi[0] {
                    for y in lo[1]..hi[1] {
                        for xx in lo[2]..hi[2] {
                            assert_eq!(sub[k], cpu_bits[(z * ps.d3[1] + y) * ps.d3[2] + xx], "random access mismatch");
                            k += 1;
                        }
                    }
                }
                let mut gpu_json = String::new();
                #[cfg(feature = "gpu")]
                {
                    let maxl = ps.shapes.iter().flat_map(|s| s.dims).max().unwrap();
                    let g = tang_compress::gpudec::GpuDecoder::new(lanes, maxl).expect("gpu");
                    // warm-up + check
                    let (v, _) = g.decode(&b).unwrap();
                    let diff = v.iter().zip(&cpu_bits).filter(|(a, b)| a != b).count();
                    assert_eq!(diff, 0, "GPU != CPU in {diff} cells");
                    let mut tt = tang_compress::gpudec::GpuTimes::default();
                    let mut tot = f64::INFINITY;
                    for _ in 0..reps {
                        let t0 = Instant::now();
                        let (v, t) = g.decode(&b).unwrap();
                        let w = t0.elapsed().as_secs_f64();
                        assert!(v == cpu_bits);
                        if w < tot {
                            tot = w;
                            tt = t;
                        }
                    }
                    let ps2 = gpuwav::parse(&b).unwrap();
                    let mut t2 = tang_compress::gpudec::GpuTimes::default();
                    let pr = g.prepare(&ps2, &b, &mut t2);
                    let kern = best(&mut || g.run(&pr));
                    gpu_json = format!(
                        ",\"gpu\":\"{}\",\"gpu_kernels_s\":{kern},\"gpu_kernels_gbs\":{},\"gpu_total_s\":{tot},\"gpu_total_gbs\":{},\"gpu_parse_s\":{},\"gpu_upload_s\":{},\"gpu_readback_s\":{},\"gpu_bit_identical\":true",
                        g.name,
                        gb / kern,
                        gb / tot,
                        tt.parse_s,
                        tt.upload_s,
                        tt.readback_s
                    );
                }
                println!(
                    "{{\"field\":\"{name}\",\"dims\":\"{}\",\"target\":\"{t}\",\"brick\":{bk},\"lanes\":{lanes},\"bpv\":{},\"enc_s\":{enc_s},\"cpu1_s\":{cpu1},\"cpu1_gbs\":{},\"cpumt_s\":{cpumt},\"cpumt_gbs\":{},\"threads\":{threads},\"ra_box\":\"{:?}-{:?}\",\"ra_s\":{ra},\"ra_mt_s\":{ra_mt}{gpu_json}}}",
                    args[2],
                    st.bits_per_value,
                    gb / cpu1,
                    gb / cpumt,
                    lo,
                    hi
                );
            }
        }
        _ => panic!("mode"),
    }
}
