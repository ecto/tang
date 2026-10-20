//! Track D: Cool-chic export (`.ccx`) → `TCI1` stream → CPU decode, with
//! rate / PSNR and timing.
//!
//! `cargo run --release --example imgcodec -- IN.ccx [LANES] [TILE_W] [OUT.tci]`

use std::time::Instant;
use tang_compress::img;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let c = img::Ccx::parse(&std::fs::read(&a[1]).unwrap()).unwrap();
    let lanes = a.get(2).map_or(32, |s| s.parse().unwrap());
    let tile_w = a.get(3).map_or(0, |s| s.parse().unwrap());
    let t0 = Instant::now();
    let (bytes, st) = img::encode(&c, lanes, tile_w);
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
    let orig: Option<Vec<u8>> = c.t.get("orig_rgb").map(|t| t.1.iter().map(|&v| v as u8).collect());
    let lat = st.latent_bytes;
    println!(
        "{h}x{w} lanes {lanes} tile {tile_w}: {} B = {:.4} bpp (model {} B, latents {} B, ideal {:.0} B) | coolchic {} B = {:.4} bpp (nn {} latent {})",
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
        println!("PSNR ours {:.3} dB | coolchic ref {:.3} dB | ours vs ref {:.2} dB", img::psnr(&rgb, o), img::psnr(&refrgb, o), img::psnr(&rgb, &refrgb));
    } else {
        println!("ours vs coolchic ref {:.2} dB", img::psnr(&rgb, &refrgb));
    }
    println!("encode {te:.2}s, CPU decode {td:.3}s; grid ideal bytes {:?}", st.grid_bits.iter().map(|b| (b / 8.0).round() as i64).collect::<Vec<_>>());
}
