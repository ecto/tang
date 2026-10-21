//! `gpubench` — tc-gpu measurements.
//!
//! ```text
//! gpubench ratio FIELD.f32 AxBxC [--bricks 32,64,128] [--lanes 32] [--threads 16]
//! ```
//! `ratio` prints one JSON line per (target, codec): serial tc-wavelet (hand)
//! and tc-gpu per brick size, each verified by decoding.

use cram_codec::codec::ErrorBound;
use cram_codec::gpuwav::{self, GpuConfig};
use cram_codec::gpuwav3::{self, Gpu3Config, Model3};
use std::time::Instant;

fn cfg3_of(bound: ErrorBound, bk: usize, args: &[String]) -> Gpu3Config {
    let m = opt(args, "--model3", "static");
    let model = if m == "static" {
        Model3::Static
    } else if let Some(h) = m.strip_prefix("overfit") {
        let hidden = h.strip_prefix(':').map_or(vec![16], |h| {
            h.split(',').map(|v| v.parse().unwrap()).collect()
        });
        Model3::Overfit {
            hidden,
            per_epoch: 200_000,
            epochs: 4,
        }
    } else if let Some(p) = m.strip_prefix("pretrained:") {
        let net = std::sync::Arc::new(
            cram_codec::learned::Net::from_bytes(&std::fs::read(p).expect("model")).expect("parse"),
        );
        cram_codec::learned::register(net.clone());
        Model3::Pretrained(net)
    } else {
        panic!("model3 {m}")
    };
    Gpu3Config {
        block: bk,
        block2d: bk,
        resctx: !args.iter().any(|a| a == "--nores"),
        lanes: opt(args, "--l3", "1").parse().unwrap(),
        compact: !args.iter().any(|a| a == "--nocompact"),
        axis: !args.iter().any(|a| a == "--noaxis"),
        model,
        ..Gpu3Config::new(bound)
    }
}
use cram_codec::{wavcodec, Decoded};

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
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
    args.iter()
        .position(|a| a == k)
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| d.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args[0].as_str();
    if mode == "cpu3" {
        // gpubench cpu3 STREAM... [pretrained:PATH]: CPU decode vs saved hash (TC_CPU_IR=1: net via tang-expr)
        for p in args.iter().filter_map(|a| a.strip_prefix("pretrained:")) {
            cram_codec::learned::register(std::sync::Arc::new(
                cram_codec::learned::Net::from_bytes(&std::fs::read(p).unwrap()).unwrap(),
            ));
        }
        let threads: usize = opt(&args, "--threads", "16").parse().unwrap();
        for f in args[1..]
            .iter()
            .filter(|a| a.ends_with(".tcg3") || a.ends_with(".tcs3"))
        {
            let b = std::fs::read(f).unwrap();
            let want = std::fs::read_to_string(format!("{f}.hash")).unwrap();
            let t0 = Instant::now();
            let (_, y) = gpuwav3::decode3_f32(&b, threads).unwrap();
            let s = t0.elapsed().as_secs_f64();
            let h = y.iter().fold(0xcbf29ce484222325u64, |h, v| {
                (h ^ v.to_bits() as u64).wrapping_mul(0x100000001b3)
            });
            println!(
                "{{\"stream\":\"{f}\",\"identical\":{},\"dec_s\":{s},\"gbs\":{}}}",
                format!("{h:016x}") == want.trim(),
                y.len() as f64 * 4.0 / 1e9 / s
            );
        }
        return;
    }
    if mode == "ship3" {
        // gpubench ship3 STREAM.tcg3... [--mlp] [pretrained:PATH]: write STREAM.tcs3 carrying the
        // built-in decoder (with the net's MLP as a graph when --mlp); report bytes and compile/cache time
        for p in args.iter().filter_map(|a| a.strip_prefix("pretrained:")) {
            cram_codec::learned::register(std::sync::Arc::new(
                cram_codec::learned::Net::from_bytes(&std::fs::read(p).unwrap()).unwrap(),
            ));
        }
        let mlp = args.iter().any(|a| a == "--mlp");
        for f in args[1..].iter().filter(|a| a.ends_with(".tcg3")) {
            let b = std::fs::read(f).unwrap();
            let ps = gpuwav3::parse3(&b).unwrap();
            let prog = cram_codec::shipdec::builtin_program(
                if mlp { ps.net.as_ref() } else { None },
                ps.net.is_some() && ps.ad.wa > 0,
            );
            let w = cram_codec::shipdec::wrap(&b, &prog);
            let out = format!("{}.tcs3", f.trim_end_matches(".tcg3"));
            std::fs::write(&out, &w).unwrap();
            std::fs::copy(format!("{f}.hash"), format!("{out}.hash")).unwrap();
            let t0 = Instant::now();
            let d = cram_codec::shipdec::compile(&prog).unwrap();
            let verify_lower_s = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let _ = cram_codec::shipdec::compile_cached(&prog).unwrap();
            let first_s = t1.elapsed().as_secs_f64();
            let t2 = Instant::now();
            let _ = cram_codec::shipdec::compile_cached(&prog).unwrap();
            let hit_s = t2.elapsed().as_secs_f64();
            #[cfg(not(feature = "gpu"))]
            let gpu = String::new();
            #[cfg(feature = "gpu")]
            let gpu = {
                let ws = gpuwav3::parse3(&w).unwrap();
                let g = cram_codec::gpudec::GpuDecoder::new(
                    ws.lanes,
                    ws.d3.iter().copied().max().unwrap(),
                )
                .unwrap();
                let mut tt = cram_codec::gpudec::GpuTimes::default();
                let pr = g.prepare3(&ws, &w, &mut tt);
                let cold = g.warm3(&pr);
                let hot = g.warm3(&pr);
                format!(",\"gpu_pipeline_cold_s\":{cold},\"gpu_pipeline_hit_s\":{hot}")
            };
            println!(
                "{{\"out\":\"{out}\",\"stream_bytes\":{},\"program_bytes\":{},\"added_bytes\":{},\"mlp\":{},\"verify_lower_s\":{verify_lower_s},\"cache_miss_s\":{first_s},\"cache_hit_s\":{hit_s},\"wgsl_chars\":{}{gpu}}}",
                b.len(),
                prog.len(),
                w.len() - b.len(),
                d.mlp.is_some(),
                d.wgsl_res.len() + d.wgsl_mlp.as_ref().map_or(0, |m| m.len())
            );
        }
        return;
    }
    #[cfg(feature = "gpu")]
    if mode == "gpu3" {
        // gpubench gpu3 STREAM... [--model3 pretrained:PATH]: GPU decode vs saved hash, kernel timing
        for p in args.iter().filter_map(|a| a.strip_prefix("pretrained:")) {
            cram_codec::learned::register(std::sync::Arc::new(
                cram_codec::learned::Net::from_bytes(&std::fs::read(p).unwrap()).unwrap(),
            ));
        }
        let reps: usize = opt(&args, "--reps", "3").parse().unwrap();
        for f in args[1..]
            .iter()
            .filter(|a| a.ends_with(".tcg3") || a.ends_with(".tcs3"))
        {
            let b = std::fs::read(f).unwrap();
            let want = std::fs::read_to_string(format!("{f}.hash")).unwrap();
            let ps = gpuwav3::parse3(&b).unwrap();
            let g =
                cram_codec::gpudec::GpuDecoder::new(ps.lanes, ps.d3.iter().copied().max().unwrap())
                    .unwrap();
            let (v, _) = g.decode3(&b).unwrap();
            let h = v.iter().fold(0xcbf29ce484222325u64, |h, &x| {
                (h ^ x as u64).wrapping_mul(0x100000001b3)
            });
            let mut t2 = cram_codec::gpudec::GpuTimes::default();
            let pr = g.prepare3(&ps, &b, &mut t2);
            let kern = (0..reps).map(|_| g.run3(&pr)).fold(f64::INFINITY, f64::min);
            let mut tot = f64::INFINITY;
            for _ in 0..reps {
                let t0 = Instant::now();
                let _ = g.decode3(&b).unwrap();
                tot = tot.min(t0.elapsed().as_secs_f64());
            }
            let gb = v.len() as f64 * 4.0 / 1e9;
            println!("{{\"stream\":\"{f}\",\"identical\":{},\"kernels_s\":{kern},\"kernels_gbs\":{},\"total_gbs\":{}}}", format!("{h:016x}") == want.trim(), gb / kern, gb / tot);
        }
        return;
    }
    let x = read_f32(&args[1]);
    let dims: Vec<usize> = args[2].split('x').map(|s| s.parse().unwrap()).collect();
    let name = std::path::Path::new(&args[1])
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let bricks: Vec<usize> = opt(&args, "--bricks", "64")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let lanes: usize = opt(&args, "--lanes", "16").parse().unwrap();
    let threads: usize = opt(&args, "--threads", "16").parse().unwrap();
    let targets = opt(&args, "--targets", "p60,p80,p100,m1e-3,m1e-4");
    let serial = !args.iter().any(|a| a == "--no-serial");
    let (lo, hi) = x
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
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
                    let Decoded::F32 { data, .. } = wavcodec::decode(&b).unwrap() else {
                        panic!()
                    };
                    let dec_s = t0.elapsed().as_secs_f64();
                    let (rmse, mx) = err(&x, &data);
                    println!(
                        "{{\"field\":\"{name}\",\"target\":\"{t}\",\"codec\":\"tc-wavelet\",\"brick\":0,\"bpv\":{},\"psnr\":{},\"max_abs_rel\":{},\"enc_s\":{enc_s},\"dec_s\":{dec_s}}}",
                        8.0 * b.len() as f64 / x.len() as f64,
                        20.0 * (range / rmse).log10(),
                        mx / range
                    );
                }
                let v3 = args.iter().any(|a| a == "--v3");
                for &bk in &bricks {
                    let cfg = GpuConfig {
                        brick: bk,
                        lanes,
                        adaptive: args.iter().any(|a| a == "--adaptive"),
                        ..GpuConfig::new(bound)
                    };
                    let cfg3 = cfg3_of(bound, bk, &args);
                    let t0 = Instant::now();
                    let (b, st) = if v3 {
                        gpuwav3::encode3_f32(&x, &dims, &cfg3, threads)
                    } else {
                        gpuwav::encode_f32_mt(&x, &dims, &cfg, threads)
                    };
                    let enc_s = t0.elapsed().as_secs_f64();
                    let t0 = Instant::now();
                    let (_, y) = if v3 {
                        gpuwav3::decode3_f32(&b, 1).unwrap()
                    } else {
                        gpuwav::decode_f32(&b, 1).unwrap()
                    };
                    let dec_s = t0.elapsed().as_secs_f64();
                    let (rmse, mx) = err(&x, &y);
                    if let ErrorBound::MaxAbs(bb) = bound {
                        assert!(mx <= bb, "{name} {t}: {mx} > {bb}");
                    }
                    println!(
                        "{{\"field\":\"{name}\",\"target\":\"{t}\",\"codec\":\"{}\",\"brick\":{bk},\"lanes\":{lanes},\"bpv\":{},\"psnr\":{},\"max_abs_rel\":{},\"enc_s\":{enc_s},\"dec_s\":{dec_s},\"hdr\":{},\"idx\":{},\"outl\":{},\"esc\":{}}}",
                        if v3 { "tc-gpu3" } else { "tc-gpu" },
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
                let cfg = GpuConfig {
                    brick: bk,
                    lanes,
                    adaptive: args.iter().any(|a| a == "--adaptive"),
                    ..GpuConfig::new(bound)
                };
                let t0 = Instant::now();
                let (b, st) = gpuwav::encode_f32_mt(&x, &dims, &cfg, threads);
                let enc_s = t0.elapsed().as_secs_f64();
                let gb = x.len() as f64 * 4.0 / 1e9;
                let best = |f: &mut dyn FnMut() -> f64| {
                    (0..reps).map(|_| f()).fold(f64::INFINITY, f64::min)
                };
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
                let lo: [usize; 3] =
                    std::array::from_fn(|a| if ps.d3[a] > 64 { ps.d3[a] / 2 - 32 } else { 0 });
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
                            assert_eq!(
                                sub[k],
                                cpu_bits[(z * ps.d3[1] + y) * ps.d3[2] + xx],
                                "random access mismatch"
                            );
                            k += 1;
                        }
                    }
                }
                #[allow(unused_mut, unused_assignments)]
                let mut gpu_json = String::new();
                #[cfg(feature = "gpu")]
                {
                    let maxl = ps.shapes.iter().flat_map(|s| s.dims).max().unwrap();
                    let g = cram_codec::gpudec::GpuDecoder::new(lanes, maxl).expect("gpu");
                    // warm-up + check
                    let (v, _) = g.decode(&b).unwrap();
                    let diff = v.iter().zip(&cpu_bits).filter(|(a, b)| a != b).count();
                    assert_eq!(diff, 0, "GPU != CPU in {diff} cells");
                    let mut tt = cram_codec::gpudec::GpuTimes::default();
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
                    let mut t2 = cram_codec::gpudec::GpuTimes::default();
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
        #[cfg(feature = "gpu")]
        "verify" => {
            // CPU == GPU byte-for-byte, error bounds, and random-access crops
            let g = cram_codec::gpudec::GpuDecoder::new(lanes, 1024).expect("gpu");
            for t in targets.split(',') {
                let bound = if let Some(p) = t.strip_prefix('p') {
                    ErrorBound::Psnr(p.parse().unwrap())
                } else {
                    ErrorBound::MaxAbs(t[1..].parse::<f64>().unwrap() * range)
                };
                for &bk in &bricks {
                    let cfg = GpuConfig {
                        brick: bk,
                        lanes,
                        ..GpuConfig::new(bound)
                    };
                    let (b, st) = gpuwav::encode_f32_mt(&x, &dims, &cfg, threads);
                    let ps = gpuwav::parse(&b).unwrap();
                    let cpu = gpuwav::decode_box_parsed(&ps, &b, [0; 3], ps.d3, threads).unwrap();
                    let (gpu, _) = g.decode(&b).unwrap();
                    let diff = cpu.iter().zip(&gpu).filter(|(a, b)| a != b).count();
                    let y: Vec<f32> = cpu.iter().map(|&v| f32::from_bits(v)).collect();
                    let (rmse, mx) = err(&x, &y);
                    let ok_bound = match bound {
                        ErrorBound::MaxAbs(bb) => mx <= bb,
                        ErrorBound::Psnr(db) => 20.0 * (range / rmse).log10() >= db - 1e-9,
                        _ => true,
                    };
                    // random-access sub-box straddling bricks
                    let lo: [usize; 3] = std::array::from_fn(|a| ps.d3[a] / 3);
                    let hi: [usize; 3] =
                        std::array::from_fn(|a| (lo[a] + 64).min(ps.d3[a]).max(lo[a] + 1));
                    let sub = gpuwav::decode_box_bits(&b, lo, hi, 1).unwrap();
                    let mut ra_ok = true;
                    let mut k = 0;
                    for z in lo[0]..hi[0] {
                        for yy in lo[1]..hi[1] {
                            for xx in lo[2]..hi[2] {
                                ra_ok &= sub[k] == cpu[(z * ps.d3[1] + yy) * ps.d3[2] + xx];
                                k += 1;
                            }
                        }
                    }
                    println!(
                        "{{\"field\":\"{name}\",\"target\":\"{t}\",\"brick\":{bk},\"bpv\":{},\"cells\":{},\"gpu_diff_cells\":{diff},\"bound_ok\":{ok_bound},\"ra_ok\":{ra_ok},\"max_abs_rel\":{}}}",
                        st.bits_per_value,
                        x.len(),
                        mx / range
                    );
                    assert!(diff == 0 && ok_bound && ra_ok, "{name} {t} {bk}");
                }
            }
        }
        #[cfg(feature = "gpu")]
        "verify3" | "speed3" => {
            let speed = mode == "speed3";
            let maxl = dims.iter().copied().max().unwrap();
            let g = cram_codec::gpudec::GpuDecoder::new(lanes, maxl).expect("gpu");
            let reps: usize = opt(&args, "--reps", "3").parse().unwrap();
            let best =
                |f: &mut dyn FnMut() -> f64| (0..reps).map(|_| f()).fold(f64::INFINITY, f64::min);
            let gb = x.len() as f64 * 4.0 / 1e9;
            for t in targets.split(',') {
                let bound = if let Some(p) = t.strip_prefix('p') {
                    ErrorBound::Psnr(p.parse().unwrap())
                } else {
                    ErrorBound::MaxAbs(t[1..].parse::<f64>().unwrap() * range)
                };
                for &bk in &bricks {
                    let cfg = cfg3_of(bound, bk, &args);
                    let t0 = Instant::now();
                    let (b, st) = gpuwav3::encode3_f32(&x, &dims, &cfg, threads);
                    let enc_s = t0.elapsed().as_secs_f64();
                    let ps = gpuwav3::parse3(&b).unwrap();
                    let full = |th: usize| {
                        gpuwav3::decode3_box_parsed(&ps, &b, [0; 3], ps.d3, th)
                            .unwrap()
                            .bits
                    };
                    let cpu = full(threads);
                    let (gpu, _) = g.decode3(&b).unwrap();
                    let diff = cpu.iter().zip(&gpu).filter(|(a, b)| a != b).count();
                    if diff > 0 {
                        let i = cpu.iter().zip(&gpu).position(|(a, b)| a != b).unwrap();
                        eprintln!(
                            "first diff at {i}: cpu {} gpu {}",
                            f32::from_bits(cpu[i]),
                            f32::from_bits(gpu[i])
                        );
                    }
                    let y: Vec<f32> = cpu.iter().map(|&v| f32::from_bits(v)).collect();
                    let (rmse, mx) = err(&x, &y);
                    let ok_bound = match bound {
                        ErrorBound::MaxAbs(bb) => mx <= bb,
                        ErrorBound::Psnr(db) => 20.0 * (range / rmse).log10() >= db - 1e-9,
                        _ => true,
                    };
                    // random access: 64^3 (2D: 64^2) box, off-centre so it straddles blocks
                    let lo: [usize; 3] =
                        std::array::from_fn(|a| if ps.d3[a] > 64 { ps.d3[a] / 3 } else { 0 });
                    let hi: [usize; 3] = std::array::from_fn(|a| (lo[a] + 64).min(ps.d3[a]));
                    let mut sub = None;
                    let ra = best(&mut || {
                        let t0 = Instant::now();
                        sub = Some(gpuwav3::decode3_box(&b, lo, hi, 1).unwrap());
                        t0.elapsed().as_secs_f64()
                    });
                    let ra_mt = best(&mut || {
                        let t0 = Instant::now();
                        let _ = gpuwav3::decode3_box(&b, lo, hi, threads).unwrap();
                        t0.elapsed().as_secs_f64()
                    });
                    let sub = sub.unwrap();
                    let mut ra_ok = true;
                    let mut k = 0;
                    for z in lo[0]..hi[0] {
                        for yy in lo[1]..hi[1] {
                            for xx in lo[2]..hi[2] {
                                ra_ok &= sub.bits[k] == cpu[(z * ps.d3[1] + yy) * ps.d3[2] + xx];
                                k += 1;
                            }
                        }
                    }
                    let mut sp = String::new();
                    if speed {
                        let cpu1 = best(&mut || {
                            let t0 = Instant::now();
                            let _ = full(1);
                            t0.elapsed().as_secs_f64()
                        });
                        let cpumt = best(&mut || {
                            let t0 = Instant::now();
                            let _ = full(threads);
                            t0.elapsed().as_secs_f64()
                        });
                        let mut tt = cram_codec::gpudec::GpuTimes::default();
                        let tot = best(&mut || {
                            let t0 = Instant::now();
                            let (v, t) = g.decode3(&b).unwrap();
                            assert!(v == cpu);
                            tt = t;
                            t0.elapsed().as_secs_f64()
                        });
                        let mut t2 = cram_codec::gpudec::GpuTimes::default();
                        let pr = g.prepare3(&ps, &b, &mut t2);
                        let kern = best(&mut || g.run3(&pr));
                        sp = format!(
                            ",\"enc_s\":{enc_s},\"cpu1_gbs\":{},\"cpumt_gbs\":{},\"gpu_kernels_gbs\":{},\"gpu_total_gbs\":{},\"gpu_readback_s\":{},\"ra_s\":{ra},\"ra_mt_s\":{ra_mt},\"ra_bytes\":{},\"ra_blocks\":{},\"total_blocks\":{}",
                            gb / cpu1,
                            gb / cpumt,
                            gb / kern,
                            gb / tot,
                            tt.readback_s,
                            sub.bytes_read,
                            sub.blocks,
                            ps.blocks.len()
                        );
                    }
                    println!(
                        "{{\"field\":\"{name}\",\"target\":\"{t}\",\"block\":{bk},\"bpv\":{},\"bytes\":{},\"cells\":{},\"gpu_diff_cells\":{diff},\"bound_ok\":{ok_bound},\"ra_ok\":{ra_ok},\"ra_bytes\":{},\"ra_s\":{ra}{sp}}}",
                        st.bits_per_value,
                        b.len(),
                        x.len(),
                        sub.bytes_read
                    );
                    assert!(diff == 0 && ok_bound && ra_ok, "{name} {t} {bk}");
                }
            }
        }
        "encode3" => {
            // write a v3 stream for later GPU-only timing: gpubench encode3 FIELD dims OUT --bricks B --targets T ...
            let out = opt(&args, "--out", "out.tcg3");
            let t = targets.split(',').next().unwrap();
            let bound = if let Some(p) = t.strip_prefix('p') {
                ErrorBound::Psnr(p.parse().unwrap())
            } else {
                ErrorBound::MaxAbs(t[1..].parse::<f64>().unwrap() * range)
            };
            let (b, st) =
                gpuwav3::encode3_f32(&x, &dims, &cfg3_of(bound, bricks[0], &args), threads);
            std::fs::write(&out, &b).unwrap();
            let (_, y) = gpuwav3::decode3_f32(&b, threads).unwrap();
            let h = y.iter().fold(0xcbf29ce484222325u64, |h, v| {
                (h ^ v.to_bits() as u64).wrapping_mul(0x100000001b3)
            });
            std::fs::write(format!("{out}.hash"), format!("{h:016x}")).unwrap();
            println!(
                "{{\"out\":\"{out}\",\"bpv\":{},\"hash\":\"{h:016x}\"}}",
                st.bits_per_value
            );
        }
        _ => panic!("mode"),
    }
}
