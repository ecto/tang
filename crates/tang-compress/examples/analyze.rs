//! Bit breakdown of the hand context model on a raw f32 3D field.
//! `analyze IN.f32 AxBxC PSNR`
use tang_compress::coefcode::{code_bands, fit_predictors, HandModel, Meter};
use tang_compress::wavcodec::{quantize, Coeffs};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let b = std::fs::read(&a[0]).unwrap();
    let x: Vec<f64> = b
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c) as f64)
        .collect();
    let dims: Vec<usize> = a[1].split('x').map(|s| s.parse().unwrap()).collect();
    let delta: f64 = a[2].parse().unwrap();
    let rho: f64 = a.get(3).map_or(0.45, |s| s.parse().unwrap());
    let co = Coeffs::new(&x, &dims);
    let (q, _) = quantize(&co, delta, rho);
    let mut m = Meter::default();
    let mut qq = q.clone();
    let preds = if std::env::var("TC_NOPRED").is_ok() {
        vec![]
    } else {
        fit_predictors(&co.p, &q)
    };
    code_bands(&co.p, &mut qq, &preds, &mut m, &mut HandModel::default());
    let n = x.len() as f64;
    let nz = q.iter().filter(|&&v| v != 0).count();
    println!(
        "nonzero {:.4}  sig {:.4} hi {:.4} low {:.4} sign {:.4}  total {:.4} bpv",
        nz as f64 / n,
        m.bits[0] / n,
        m.bits[1] / n,
        m.bits[2] / n,
        m.bits[3] / n,
        m.bits.iter().sum::<f64>() / n
    );
    // per band
    for (bi, band) in co.p.bands.iter().enumerate() {
        let _ = bi;
        let [_, ny, nx] = co.p.dims;
        let mut cnt = 0;
        let mut nzc = 0;
        let mut sa = 0u64;
        for z in band.start[0]..band.end[0] {
            for y in band.start[1]..band.end[1] {
                for xx in band.start[2]..band.end[2] {
                    let v = q[(z * ny + y) * nx + xx];
                    cnt += 1;
                    if v != 0 {
                        nzc += 1;
                    }
                    sa += v.unsigned_abs() as u64;
                }
            }
        }
        if band.level <= 1 {
            println!(
                "  L{} m{:03b} n={} nz={:.3} mean|q|={:.2}",
                band.level,
                band.mask,
                cnt,
                nzc as f64 / cnt as f64,
                sa as f64 / cnt as f64
            );
        }
    }
}
