//! Track D: Cool-chic export (`.ccx`) → `TCI1` stream → CPU decode, with
//! rate / PSNR and timing.
//!
//! `cargo run --release --example imgcodec -- IN.ccx [LANES] [TILE_W] [OUT.tci]`

use cram_codec::img;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let c = img::Ccx::parse(&std::fs::read(&a[1]).unwrap()).unwrap();
    let lanes = a.get(2).map_or(32, |s| s.parse().unwrap());
    // TILE as W or WxH (0 = whole grid)
    let (tile_w, tile_h) = a.get(3).map_or((0, 0), |s| match s.split_once('x') {
        Some((w, h)) => (w.parse().unwrap(), h.parse().unwrap()),
        None => (s.parse().unwrap(), 0),
    });
    let t0 = Instant::now();
    let (bytes, st) = img::encode(&c, lanes, tile_w, tile_h);
    let te = t0.elapsed().as_secs_f64();
    if let Some(p) = a.get(4) {
        std::fs::write(p, &bytes).unwrap();
    }
    let t0 = Instant::now();
    let (h, w, rgb) = img::decode(&bytes).unwrap();
    let td = t0.elapsed().as_secs_f64();
    let px = (h * w) as f64;
    let cc = &c.t["bytes"].1;
    let refrgb: Vec<u8> = c.t["ref_rgb"].1.iter().map(|&v| v as u8).collect();
    let orig: Option<Vec<u8>> =
        c.t.get("orig_rgb")
            .map(|t| t.1.iter().map(|&v| v as u8).collect());
    let lat = st.latent_bytes;
    println!(
        "{h}x{w} lanes {lanes} tile {tile_w}x{tile_h}: {} B = {:.4} bpp (model {} B, latents {} B, ideal {:.0} B) | coolchic {} B = {:.4} bpp (nn {} latent {})",
        st.bytes,
        st.bytes as f64 * 8.0 / px,
        st.model_bytes,
        lat,
        st.ideal_bits / 8.0,
        cc[0],
        cc[0] * 8.0 / px,
        cc[1],
        cc[2]
    );
    if let Some(o) = &orig {
        println!(
            "PSNR ours {:.3} dB | coolchic ref {:.3} dB | ours vs ref {:.2} dB",
            img::psnr(&rgb, o),
            img::psnr(&refrgb, o),
            img::psnr(&rgb, &refrgb)
        );
    } else {
        println!("ours vs coolchic ref {:.2} dB", img::psnr(&rgb, &refrgb));
    }
    #[cfg(feature = "gpu")]
    {
        let gpu = cram_codec::imggpu::ImgGpu::new().unwrap();
        let p = img::parse(&bytes).unwrap();
        let pr = gpu.prepare(&p);
        if std::env::var("TC_DUMP_WGSL").is_ok() {
            std::fs::write("/tmp/tci.wgsl", &pr.src).unwrap();
        }
        let mut best = f64::MAX;
        for _ in 0..5 {
            best = best.min(gpu.run(&pr));
        }
        if std::env::var("TC_PROF").is_ok() {
            let names = gpu.plan_names(&pr);
            let mut prev = 0.0;
            for k in 1..=names.len() {
                let t = (0..3)
                    .map(|_| gpu.run_upto(&pr, k))
                    .fold(f64::MAX, f64::min);
                println!(
                    "  {:>8} cum {:7.2} ms (+{:.2})",
                    names[k - 1],
                    t * 1e3,
                    (t - prev) * 1e3
                );
                prev = t;
            }
        }
        let g = gpu.readback(&pr);
        let same = g == rgb;
        let diff = g.iter().zip(&rgb).filter(|(a, b)| a != b).count();
        println!("GPU {}: decode {:.2} ms (best of 5), compile {:.2} s, byte-identical to CPU: {same} ({diff} bytes differ)", gpu.name, best * 1e3, pr.compile_s);
    }
    println!(
        "encode {te:.2}s, CPU decode {td:.3}s; grid ideal bytes {:?}",
        st.grid_bits
            .iter()
            .map(|b| (b / 8.0).round() as i64)
            .collect::<Vec<_>>()
    );
}
