use tang::{Dual, Scalar};
use wasm_bindgen::prelude::*;

/// Evaluate f(x) = x * sin(x) using Dual<f64>.
/// Returns [value, derivative].
#[wasm_bindgen]
pub fn dual_eval(x: f64) -> Box<[f64]> {
    let d = Dual::var(x);
    let y = d * d.sin();
    Box::new([y.real, y.dual])
}

/// Generate SVG path `d` attribute for f(x) = x * sin(x).
/// Maps math coordinates to SVG pixel coordinates.
#[wasm_bindgen]
pub fn curve_svg_path(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    w: f64,
    h: f64,
    steps: u32,
) -> String {
    let mut d = String::with_capacity(steps as usize * 20);
    for i in 0..steps {
        let t = i as f64 / (steps - 1) as f64;
        let x = x_min + (x_max - x_min) * t;
        let y = x * x.sin();
        let sx = (x - x_min) / (x_max - x_min) * w;
        let sy = h - (y - y_min) / (y_max - y_min) * h;
        if i == 0 {
            d.push_str(&format!("M{:.1},{:.1}", sx, sy));
        } else {
            d.push_str(&format!(" L{:.1},{:.1}", sx, sy));
        }
    }
    d
}

/// Convert math coordinates to SVG pixel coordinates.
/// Returns [sx, sy].
#[wasm_bindgen]
pub fn to_svg(x: f64, y: f64, x_min: f64, x_max: f64, y_min: f64, y_max: f64, w: f64, h: f64) -> Box<[f64]> {
    let sx = (x - x_min) / (x_max - x_min) * w;
    let sy = h - (y - y_min) / (y_max - y_min) * h;
    Box::new([sx, sy])
}

/// Compute tangent line endpoints in SVG coordinates.
/// Returns [x1, y1, x2, y2].
#[wasm_bindgen]
pub fn tangent_svg(
    x: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    w: f64,
    h: f64,
    half_len: f64,
) -> Box<[f64]> {
    let d = Dual::var(x);
    let y_d = d * d.sin();
    let y_val = y_d.real;
    let slope = y_d.dual;

    let x1 = x - half_len;
    let y1 = y_val - slope * half_len;
    let x2 = x + half_len;
    let y2 = y_val + slope * half_len;

    let sx1 = (x1 - x_min) / (x_max - x_min) * w;
    let sy1 = h - (y1 - y_min) / (y_max - y_min) * h;
    let sx2 = (x2 - x_min) / (x_max - x_min) * w;
    let sy2 = h - (y2 - y_min) / (y_max - y_min) * h;

    Box::new([sx1, sy1, sx2, sy2])
}
