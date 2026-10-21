//! Encode a synthetic 2D field under each error-bound mode, decode it and
//! check the bound.
//!
//! ```text
//! cargo run --release -p cram-codec --example roundtrip
//! ```
use cram_codec::{decode, encode_f32, Config, Decoded, ErrorBound};

fn main() {
    let (h, w) = (128, 128);
    let x: Vec<f32> = (0..h * w)
        .map(|k| {
            let (i, j) = ((k / w) as f32, (k % w) as f32);
            (i * 0.05).sin() * (j * 0.03).cos() + 0.1 * (i * j * 1e-3).sin()
        })
        .collect();

    for bound in [
        ErrorBound::MaxAbs(1e-3),
        ErrorBound::Rmse(1e-4),
        ErrorBound::Psnr(60.0),
    ] {
        let (bytes, stats) = encode_f32(&x, &[h, w], &Config::new(bound));
        let Decoded::F32 { data, .. } = decode(&bytes).expect("decode") else {
            unreachable!("encoded f32")
        };
        let max_abs = x
            .iter()
            .zip(&data)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        let rmse = (x
            .iter()
            .zip(&data)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / x.len() as f64)
            .sqrt();
        println!(
            "{bound:?}: {} bytes ({:.3} bits/value, ratio {:.1}x), max|err| {max_abs:.2e}, rmse {rmse:.2e}",
            stats.bytes,
            stats.bits_per_value,
            32.0 / stats.bits_per_value
        );
        if let ErrorBound::MaxAbs(e) = bound {
            assert!(max_abs <= e);
        }
    }
}
