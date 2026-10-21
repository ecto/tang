//! Lorenzo-predictor codec for pointwise (max-abs) bounds, and for RMSE /
//! PSNR targets through rate control of the bound.
//!
//! Aimed at rough fields with one smooth (or repeating) axis, such as
//! particle trajectories stored time × particle, where interpolating across
//! the rough axis costs more than it predicts.
//!
//! 1. **Prediction**, in row-major order on f32 reconstructions. Mode 0 is
//!    the full N-D Lorenzo predictor (inclusion–exclusion over the axes whose
//!    coordinate is > 0); mode `k + 1` is the previous point along axis `k`
//!    (the N-D predictor where that coordinate is 0). The mode is picked by
//!    real stream size.
//! 2. **Quantization**, `q = round((x − x̃)/2b)`, checked on the exact f32
//!    reconstruction; failures are escaped raw (as in [`crate::interp`]).
//! 3. **Coding.** Adaptive binary range coding: a zero flag with contexts
//!    (symbol class of the previous point along the prediction axis and of
//!    the side neighbour, `log₂` of the last step along the prediction axis
//!    in quanta), a sign bit keyed on the previous point's sign, and an
//!    adaptive Elias-gamma magnitude keyed on the two neighbour classes.

use crate::abac::{BitModel, Decoder, Encoder, UintModel};
use crate::bitio::{ByteReader, ByteWriter, DecodeError};
use crate::codec::Decoded;
use crate::interp::{quant, rate_control, ESC};

/// Stream magic of the Lorenzo codec.
pub const MAGIC: &[u8; 4] = b"TCL1";

const ACT: usize = 9;
/// Neighbour symbol classes: 0 zero, 1 `|q| = 1`, 2 larger or escaped, 3 none.
const CLS: usize = 4;

struct Models {
    zero: Vec<BitModel>,
    sign: Vec<BitModel>,
    mag: Vec<UintModel>,
}

impl Models {
    fn new() -> Self {
        Self {
            zero: vec![BitModel::default(); CLS * CLS * ACT],
            sign: vec![BitModel::default(); 3],
            mag: vec![UintModel::default(); CLS * CLS],
        }
    }
}

/// Coding context of one point.
#[derive(Clone, Copy)]
struct Ctx {
    up: usize,
    side: usize,
    act: usize,
    /// 0 none or zero, 1 positive, 2 negative (previous point along the axis).
    sgn: usize,
}

/// Per-point symbol record: class in bits 0–1, negative sign in bit 2.
fn record(q: i64, esc: bool) -> u8 {
    let c = if esc || q.unsigned_abs() > 1 {
        2
    } else {
        q.unsigned_abs() as u8
    };
    c | ((q < 0) as u8) << 2
}

fn strides(dims: &[usize]) -> Vec<usize> {
    let mut st = vec![1usize; dims.len()];
    for j in (0..dims.len().saturating_sub(1)).rev() {
        st[j] = st[j + 1] * dims[j + 1];
    }
    st
}

/// Number of prediction modes for `nd` axes (1D has only the previous point).
fn modes(nd: usize) -> u8 {
    if nd == 1 {
        1
    } else {
        nd as u8 + 1
    }
}

/// Visit every point in row-major order with `f(index, prediction, ctx)`,
/// which returns `(reconstruction, record)`.
fn walk(
    dims: &[usize],
    mode: u8,
    b: f64,
    r: &mut [f32],
    rec: &mut [u8],
    mut f: impl FnMut(usize, f64, Ctx) -> (f32, u8),
) {
    let nd = dims.len();
    let st = strides(dims);
    // prediction axis (mode 0: the slowest), side axis: the fastest other one
    let k = if mode == 0 { 0 } else { mode as usize - 1 };
    let side = if k == nd - 1 {
        nd.checked_sub(2)
    } else {
        Some(nd - 1)
    };
    let step = 2.0 * b;
    let n: usize = dims.iter().product();
    let mut c = vec![0usize; nd];
    for i in 0..n {
        let along = c[k] > 0;
        let pred = if along && mode != 0 {
            r[i - st[k]] as f64
        } else {
            // N-D Lorenzo over the axes with coordinate > 0
            let live: Vec<usize> = (0..nd).filter(|&j| c[j] > 0).map(|j| st[j]).collect();
            let mut p = 0f64;
            for m in 1u32..(1 << live.len()) {
                let off: usize = (0..live.len())
                    .filter(|&t| m >> t & 1 == 1)
                    .map(|t| live[t])
                    .sum();
                let v = r[i - off] as f64;
                p += if m.count_ones() % 2 == 1 { v } else { -v };
            }
            p
        };
        let (up, sgn, act) = if along {
            let u = rec[i - st[k]];
            let act = if c[k] > 1 {
                let d = (r[i - st[k]] as f64 - r[i - 2 * st[k]] as f64).abs() / step;
                let u = d as u64;
                ((u64::BITS - u.leading_zeros()) as usize).min(ACT - 2)
            } else {
                ACT - 1
            };
            let sgn = if u & 3 == 0 { 0 } else { 1 + (u >> 2) as usize };
            ((u & 3) as usize, sgn, act)
        } else {
            (CLS - 1, 0, ACT - 1)
        };
        let sd = match side {
            Some(s) if c[s] > 0 => (rec[i - st[s]] & 3) as usize,
            _ => CLS - 1,
        };
        let (v, q) = f(
            i,
            pred,
            Ctx {
                up,
                side: sd,
                act,
                sgn,
            },
        );
        r[i] = v;
        rec[i] = q;
        for j in (0..nd).rev() {
            c[j] += 1;
            if c[j] < dims[j] {
                break;
            }
            c[j] = 0;
        }
    }
}

fn put(e: &mut Encoder, m: &mut Models, cx: Ctx, q: i64, raw: Option<f32>) {
    let z = (cx.up * CLS + cx.side) * ACT + cx.act;
    if q == 0 && raw.is_none() {
        e.encode(false, &mut m.zero[z]);
        return;
    }
    e.encode(true, &mut m.zero[z]);
    e.encode(q < 0, &mut m.sign[cx.sgn]);
    let mc = cx.up * CLS + cx.side;
    match raw {
        Some(v) => {
            m.mag[mc].encode(e, ESC - 1);
            e.encode_raw(v.to_bits() as u64, 32);
        }
        None => m.mag[mc].encode(e, q.unsigned_abs() - 1),
    }
}

/// `Ok(q)` or `Err(raw value)`.
fn get(d: &mut Decoder, m: &mut Models, cx: Ctx) -> Result<i64, f32> {
    let z = (cx.up * CLS + cx.side) * ACT + cx.act;
    if !d.decode(&mut m.zero[z]) {
        return Ok(0);
    }
    let neg = d.decode(&mut m.sign[cx.sgn]);
    let a = m.mag[cx.up * CLS + cx.side].decode(d) + 1;
    if a >= ESC {
        return Err(f32::from_bits(d.decode_raw(32) as u32));
    }
    Ok(if neg { -(a as i64) } else { a as i64 })
}

/// One stream at bound `b` in prediction `mode`, with its squared error.
fn encode_sse(x: &[f32], dims: &[usize], b: f64, mode: u8) -> (Vec<u8>, f64) {
    let n = x.len();
    let mut r = vec![0f32; n];
    let mut rc = vec![0u8; n];
    let mut m = Models::new();
    let mut e = Encoder::new();
    let step = 2.0 * b;
    walk(dims, mode, b, &mut r, &mut rc, |i, p, cx| {
        let (q, v, esc) = quant(x[i], p, step, b);
        put(&mut e, &mut m, cx, q, esc.then_some(x[i]));
        (v, record(q, esc))
    });
    let mut w = ByteWriter::new();
    w.bytes(MAGIC);
    w.u8(dims.len() as u8);
    for &d in dims {
        w.varint(d as u64);
    }
    w.f64(b);
    w.u8(mode);
    w.bytes(&e.finish());
    let sse = x
        .iter()
        .zip(&r)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum();
    (w.buf, sse)
}

fn check(x: &[f32], dims: &[usize]) {
    assert_eq!(
        x.len(),
        dims.iter().product::<usize>(),
        "dims do not match data"
    );
    assert!(!dims.is_empty() && dims.len() <= 8);
}

/// The prediction mode with the smallest stream at bound `b`, and that stream.
fn best_mode(x: &[f32], dims: &[usize], b: f64) -> (u8, Vec<u8>) {
    (0..modes(dims.len()))
        .map(|md| (md, encode_sse(x, dims, b, md).0))
        .min_by_key(|(_, s)| s.len())
        .unwrap()
}

/// Encode `x` (row-major, shape `dims`) with `|x − x̂| ≤ b` everywhere,
/// trying each prediction mode and keeping the smallest stream.
pub fn encode_f32(x: &[f32], dims: &[usize], b: f64) -> Vec<u8> {
    check(x, dims);
    assert!(b > 0.0);
    best_mode(x, dims, b).1
}

/// Encode for an RMSE target (`rmse` absolute): the mode is picked at the
/// bound `rmse·√3`, then that mode's bound is rate controlled as in
/// [`crate::interp::encode_rmse_f32`]. The target is checked on the exact
/// reconstruction.
pub fn encode_rmse_f32(x: &[f32], dims: &[usize], rmse: f64) -> Vec<u8> {
    check(x, dims);
    assert!(rmse > 0.0);
    let (mode, _) = best_mode(x, dims, rmse * 3f64.sqrt());
    rate_control(x.len(), rmse, |b| encode_sse(x, dims, b, mode))
}

/// [`encode_rmse_f32`] at `PSNR = 20 log₁₀(range / RMSE)` dB.
pub fn encode_psnr_f32(x: &[f32], dims: &[usize], psnr: f64) -> Vec<u8> {
    let (lo, hi) = x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v as f64), b.max(v as f64))
        });
    let range = if hi > lo { hi - lo } else { 1.0 };
    encode_rmse_f32(x, dims, range / 10f64.powf(psnr / 20.0))
}

/// Decode a Lorenzo stream.
pub fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
    let mut rd = ByteReader::new(bytes);
    if rd.take(4)? != MAGIC {
        return Err(DecodeError::Invalid("magic"));
    }
    let nd = rd.u8()? as usize;
    let dims = (0..nd)
        .map(|_| rd.varint().map(|v| v as usize))
        .collect::<Result<Vec<_>, _>>()?;
    if nd == 0
        || nd > 8
        || dims.contains(&0)
        || dims
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .is_none_or(|n| n as u64 > 1 << 34)
    {
        return Err(DecodeError::Invalid("dims"));
    }
    let b = rd.f64()?;
    let mode = rd.u8()?;
    if b.is_nan() || b <= 0.0 {
        return Err(DecodeError::Invalid("bound"));
    }
    if mode >= modes(nd) {
        return Err(DecodeError::Invalid("mode"));
    }
    let mut d = Decoder::new(&bytes[rd.position()..]);
    let mut m = Models::new();
    let n: usize = dims.iter().product();
    let mut r = vec![0f32; n];
    let mut rc = vec![0u8; n];
    let step = 2.0 * b;
    walk(&dims, mode, b, &mut r, &mut rc, |_, p, cx| {
        match get(&mut d, &mut m, cx) {
            Ok(q) => ((p + q as f64 * step) as f32, record(q, false)),
            Err(v) => (v, record(0, true)),
        }
    });
    if d.overrun() {
        return Err(DecodeError::Invalid("truncated"));
    }
    Ok(Decoded::F32 { dims, data: r })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(d: &[usize]) -> Vec<f32> {
        let n: usize = d.iter().product();
        (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                (t * 40.0).sin() * 3.0
                    + ((i * 7919) % 13) as f32 * 0.4
                    + if i % 997 == 0 { 50.0 } else { 0.0 }
            })
            .collect()
    }

    #[test]
    fn round_trips_within_bound_in_every_mode() {
        for d in [
            vec![1000],
            vec![33, 70],
            vec![17, 9, 31],
            vec![1, 1, 5],
            vec![1],
        ] {
            let x = field(&d);
            for b in [1e-1, 1e-3, 1e-6] {
                for md in 0..modes(d.len()) {
                    let (bytes, sse) = encode_sse(&x, &d, b, md);
                    let Decoded::F32 { data, dims } = decode(&bytes).unwrap() else {
                        panic!()
                    };
                    assert_eq!(dims, d);
                    let mut got = 0f64;
                    for (a, c) in x.iter().zip(&data) {
                        assert!(((a - c) as f64).abs() <= b, "{d:?} b={b} m={md} {a} {c}");
                        got += (*a as f64 - *c as f64).powi(2);
                    }
                    assert_eq!(got, sse);
                }
            }
        }
    }

    #[test]
    fn psnr_targets_are_met() {
        let d = [64, 50];
        let x = field(&d);
        let (lo, hi) = x
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v as f64), b.max(v as f64))
            });
        for p in [40.0, 70.0, 100.0] {
            let s = encode_psnr_f32(&x, &d, p);
            let Decoded::F32 { data: y, .. } = decode(&s).unwrap() else {
                panic!()
            };
            let mse = x
                .iter()
                .zip(&y)
                .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                .sum::<f64>()
                / x.len() as f64;
            assert!(20.0 * ((hi - lo) / mse.sqrt()).log10() >= p);
        }
    }

    #[test]
    fn picks_the_repeating_axis() {
        // each particle holds a value for 8 steps (axis 0, staggered per
        // particle); random across particles
        let d = [64, 64];
        let x: Vec<f32> = (0..64 * 64)
            .map(|i| {
                let (t, p) = (i / 64, i % 64);
                ((p * 2654435761usize) % 1000) as f32 * 0.01
                    + (((p + 64 * ((t + p) / 8)) * 40503) % 97) as f32 * 0.05
            })
            .collect();
        let (mode, _) = best_mode(&x, &d, 1e-3);
        assert_eq!(mode, 1);
    }
}
