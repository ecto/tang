//! SZ3-style multilevel interpolation codec for pointwise (max-abs) bounds.
//!
//! Aimed at loose bounds on smooth fields, where the wavelet codec's fixed
//! overheads (centroids, band predictors, outlier map) dominate a tiny
//! stream.
//!
//! 1. **Prediction.** The grid is refined coarse to fine: at stride `s` (a
//!    power of two, largest first) and for each axis `k` in turn, the points
//!    whose `k` coordinate is an odd multiple of `s` are predicted from
//!    already reconstructed points at `±s, ±3s` along `k` — cubic
//!    `(−1, 9, 9, −1)/16` inside, one-sided quadratics near an edge, linear
//!    or copy at the boundary. Each level picks linear or cubic by a trial
//!    cost (one header bit per level).
//! 2. **Quantization.** `q = round((x − x̃)/2b_ℓ)` with a level bound
//!    `b_ℓ = b / min(α^(ℓ−1), β) ≤ b` (finer prediction anchors, as in QoZ);
//!    the reconstruction `x̃ + 2b_ℓq` is cast to f32 and checked against `b`
//!    exactly as the decoder will compute it, and any point that fails (or
//!    overflows the code range) is escaped and stored raw. `(α, β)` is picked
//!    from a small preset list by actual stream size.
//! 3. **Coding.** Adaptive binary range coding ([`crate::abac`]): a zero flag
//!    with contexts (level, `log₂` of the local neighbour difference in
//!    quanta, previous symbol class), a sign bit per level and an adaptive
//!    Elias-gamma magnitude.
//!
//! Prediction is f64 arithmetic on f32 reconstructions in a fixed order, so
//! encoder and decoder agree bit for bit.

use crate::abac::{BitModel, Decoder, Encoder, UintModel};
use crate::bitio::{ByteReader, ByteWriter, DecodeError};
use crate::codec::Decoded;

/// Stream magic of the interpolation codec.
pub const MAGIC: &[u8; 4] = b"TCI1";

/// Magnitude that marks an escaped (raw f32) point.
const ESC: u64 = 1 << 24;
const LVL: usize = 4;
const ACT: usize = 9;
const PREV: usize = 3;
/// `(α, β)` presets for the level-wise bound.
const TUNE: [(f64, f64); 4] = [(1.0, 1.0), (1.25, 1.5), (1.5, 2.0), (2.0, 4.0)];

struct Models {
    zero: Vec<BitModel>,
    sign: Vec<BitModel>,
    mag: Vec<UintModel>,
    prev: usize,
}

impl Models {
    fn new() -> Self {
        Self {
            zero: vec![BitModel::default(); LVL * ACT * PREV],
            sign: vec![BitModel::default(); LVL],
            mag: vec![UintModel::default(); LVL * 4],
            prev: 0,
        }
    }
}

/// Coding context of one point.
#[derive(Clone, Copy)]
struct Ctx {
    lvl: usize,
    act: usize,
}

fn levels(dims: &[usize]) -> u32 {
    let m = dims.iter().copied().max().unwrap_or(1);
    usize::BITS - (m.max(1) - 1).leading_zeros()
}

fn level_bound(b: f64, lv: u32, tune: (f64, f64)) -> f64 {
    b / tune.0.powi(lv as i32 - 1).min(tune.1)
}

/// Activity bucket: `⌈log₂(1 + |b − a|/step)⌉`, clamped; `ACT − 1` = no right neighbour.
fn act_bucket(d: Option<f64>, step: f64) -> usize {
    match d {
        None => ACT - 1,
        Some(d) => {
            let u = (d / step) as u64;
            ((u64::BITS - u.leading_zeros()) as usize).min(ACT - 2)
        }
    }
}

/// Visit every point of one refinement pass (stride `s`, axis `k`) in
/// decode order, calling `f(index, prediction, neighbour difference)`; `f`
/// returns the reconstruction, which is written to `r`.
fn pass(
    dims: &[usize],
    s: usize,
    k: usize,
    cubic: bool,
    r: &mut [f32],
    mut f: impl FnMut(usize, f64, Option<f64>) -> f32,
) {
    let nd = dims.len();
    if dims[k] <= s {
        return;
    }
    let mut strides = vec![1usize; nd];
    for j in (0..nd.saturating_sub(1)).rev() {
        strides[j] = strides[j + 1] * dims[j + 1];
    }
    let (start, step): (Vec<usize>, Vec<usize>) = (0..nd)
        .map(|j| match j.cmp(&k) {
            std::cmp::Ordering::Less => (0, s),
            std::cmp::Ordering::Equal => (s, 2 * s),
            std::cmp::Ordering::Greater => (0, 2 * s),
        })
        .unzip();
    let st = strides[k];
    let n = dims[k];
    let mut c = start.clone();
    loop {
        let i: usize = c.iter().zip(&strides).map(|(a, b)| a * b).sum();
        let ck = c[k];
        let a = r[i - s * st] as f64;
        let (pred, diff) = if ck + s < n {
            let b = r[i + s * st] as f64;
            let p = if cubic {
                match (ck >= 3 * s, ck + 3 * s < n) {
                    (true, true) => {
                        (-(r[i - 3 * s * st] as f64) + 9.0 * a + 9.0 * b - r[i + 3 * s * st] as f64)
                            / 16.0
                    }
                    (false, true) => (3.0 * a + 6.0 * b - r[i + 3 * s * st] as f64) / 8.0,
                    (true, false) => (-(r[i - 3 * s * st] as f64) + 6.0 * a + 3.0 * b) / 8.0,
                    (false, false) => (a + b) / 2.0,
                }
            } else {
                (a + b) / 2.0
            };
            (p, Some((b - a).abs()))
        } else {
            (a, None)
        };
        r[i] = f(i, pred, diff);
        // odometer, last axis fastest
        let mut j = nd;
        loop {
            if j == 0 {
                return;
            }
            j -= 1;
            c[j] += step[j];
            if c[j] < dims[j] {
                break;
            }
            c[j] = start[j];
        }
    }
}

fn zctx(m: &Models, cx: Ctx) -> usize {
    (cx.lvl * ACT + cx.act) * PREV + m.prev
}

fn mctx(cx: Ctx) -> usize {
    cx.lvl * 4 + cx.act.min(3)
}

fn put(e: &mut Encoder, m: &mut Models, cx: Ctx, q: i64, raw: Option<f32>) {
    let z = zctx(m, cx);
    if q == 0 && raw.is_none() {
        e.encode(false, &mut m.zero[z]);
        m.prev = 0;
        return;
    }
    e.encode(true, &mut m.zero[z]);
    e.encode(q < 0, &mut m.sign[cx.lvl]);
    match raw {
        Some(v) => {
            m.mag[mctx(cx)].encode(e, ESC - 1);
            e.encode_raw(v.to_bits() as u64, 32);
            m.prev = 2;
        }
        None => {
            let a = q.unsigned_abs();
            m.mag[mctx(cx)].encode(e, a - 1);
            m.prev = if a == 1 { 1 } else { 2 };
        }
    }
}

/// `Ok(q)` or `Err(raw value)`.
fn get(d: &mut Decoder, m: &mut Models, cx: Ctx) -> Result<i64, f32> {
    let z = zctx(m, cx);
    if !d.decode(&mut m.zero[z]) {
        m.prev = 0;
        return Ok(0);
    }
    let neg = d.decode(&mut m.sign[cx.lvl]);
    let a = m.mag[mctx(cx)].decode(d) + 1;
    if a >= ESC {
        m.prev = 2;
        return Err(f32::from_bits(d.decode_raw(32) as u32));
    }
    m.prev = if a == 1 { 1 } else { 2 };
    Ok(if neg { -(a as i64) } else { a as i64 })
}

/// Quantize one point: `(q, reconstruction, escaped)`.
#[inline]
fn quant(x: f32, pred: f64, step: f64, b: f64) -> (i64, f32, bool) {
    let qf = ((x as f64 - pred) / step).round();
    if qf.abs() < (ESC - 1) as f64 {
        let q = qf as i64;
        let rec = (pred + q as f64 * step) as f32;
        if ((x as f64) - rec as f64).abs() <= b && rec.is_finite() {
            return (q, rec, false);
        }
    }
    (0, x, true)
}

fn encode_with(x: &[f32], dims: &[usize], b: f64, tune: (f64, f64)) -> Vec<u8> {
    let nl = levels(dims);
    let mut r = vec![0f32; x.len()];
    let mut m = Models::new();
    let mut e = Encoder::new();
    let mut kinds = 0u64;
    // origin: predicted as 0 at the top level's bound
    let step0 = 2.0 * level_bound(b, nl.max(1), tune);
    let (q, rec, esc) = quant(x[0], 0.0, step0, b);
    put(
        &mut e,
        &mut m,
        Ctx {
            lvl: LVL - 1,
            act: ACT - 1,
        },
        q,
        esc.then_some(x[0]),
    );
    r[0] = rec;
    for lv in (1..=nl).rev() {
        let s = 1usize << (lv - 1);
        let step = 2.0 * level_bound(b, lv, tune);
        let lvl = (lv as usize - 1).min(LVL - 1);
        // trial cost of linear vs cubic for this level
        let mut cost = [0f64; 2];
        for (ci, cubic) in [false, true].into_iter().enumerate() {
            for k in 0..dims.len() {
                pass(dims, s, k, cubic, &mut r, |i, p, _| {
                    let (q, rec, esc) = quant(x[i], p, step, b);
                    cost[ci] += if esc {
                        40.0
                    } else if q == 0 {
                        0.0
                    } else {
                        2.0 + 2.0 * (q.unsigned_abs() as f64).log2()
                    };
                    rec
                });
            }
        }
        let cubic = cost[1] <= cost[0];
        kinds |= (cubic as u64) << lv;
        for k in 0..dims.len() {
            pass(dims, s, k, cubic, &mut r, |i, p, d| {
                let (q, rec, esc) = quant(x[i], p, step, b);
                put(
                    &mut e,
                    &mut m,
                    Ctx {
                        lvl,
                        act: act_bucket(d, step),
                    },
                    q,
                    esc.then_some(x[i]),
                );
                rec
            });
        }
    }
    let mut w = ByteWriter::new();
    w.bytes(MAGIC);
    w.u8(dims.len() as u8);
    for &d in dims {
        w.varint(d as u64);
    }
    w.f64(b);
    w.u8(TUNE.iter().position(|&t| t == tune).unwrap() as u8);
    w.varint(kinds);
    w.bytes(&e.finish());
    w.buf
}

/// Encode `x` (row-major, shape `dims`) with `|x − x̂| ≤ b` everywhere,
/// trying each `(α, β)` preset and keeping the smallest stream.
pub fn encode_f32(x: &[f32], dims: &[usize], b: f64) -> Vec<u8> {
    assert_eq!(
        x.len(),
        dims.iter().product::<usize>(),
        "dims do not match data"
    );
    assert!(!dims.is_empty() && dims.len() <= 255 && b > 0.0);
    TUNE.iter()
        .map(|&t| encode_with(x, dims, b, t))
        .min_by_key(|v| v.len())
        .unwrap()
}

/// Fast-mode preset index into the `(α, β)` table (the one most often kept
/// by [`encode_f32`] on the seen fields).
pub const FAST_TUNE: usize = 1;

/// [`encode_f32`] with a single `(α, β)` preset ([`FAST_TUNE`]).
pub fn encode_fast_f32(x: &[f32], dims: &[usize], b: f64) -> Vec<u8> {
    assert_eq!(x.len(), dims.iter().product::<usize>(), "dims do not match data");
    assert!(!dims.is_empty() && dims.len() <= 255 && b > 0.0);
    let i = std::env::var("TC_FAST_TUNE").ok().and_then(|v| v.parse().ok()).unwrap_or(FAST_TUNE);
    encode_with(x, dims, b, TUNE[i])
}

/// Decode an interpolation stream.
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
        || dims.contains(&0)
        || dims
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .is_none_or(|n| n as u64 > 1 << 34)
    {
        return Err(DecodeError::Invalid("dims"));
    }
    let b = rd.f64()?;
    let tune = *TUNE
        .get(rd.u8()? as usize)
        .ok_or(DecodeError::Invalid("tune"))?;
    let kinds = rd.varint()?;
    if b.is_nan() || b <= 0.0 {
        return Err(DecodeError::Invalid("bound"));
    }
    let payload = &bytes[rd.position()..];
    let mut d = Decoder::new(payload);
    let mut m = Models::new();
    let n: usize = dims.iter().product();
    let mut r = vec![0f32; n];
    let nl = levels(&dims);
    let val = |q: Result<i64, f32>, p: f64, step: f64| match q {
        Ok(q) => (p + q as f64 * step) as f32,
        Err(v) => v,
    };
    let step0 = 2.0 * level_bound(b, nl.max(1), tune);
    r[0] = val(
        get(
            &mut d,
            &mut m,
            Ctx {
                lvl: LVL - 1,
                act: ACT - 1,
            },
        ),
        0.0,
        step0,
    );
    for lv in (1..=nl).rev() {
        let s = 1usize << (lv - 1);
        let step = 2.0 * level_bound(b, lv, tune);
        let lvl = (lv as usize - 1).min(LVL - 1);
        let cubic = (kinds >> lv) & 1 == 1;
        for k in 0..nd {
            pass(&dims, s, k, cubic, &mut r, |_, p, dd| {
                val(
                    get(
                        &mut d,
                        &mut m,
                        Ctx {
                            lvl,
                            act: act_bucket(dd, step),
                        },
                    ),
                    p,
                    step,
                )
            });
        }
    }
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
                    + (i % d[d.len() - 1]) as f32 * 0.01
                    + if i % 997 == 0 { 50.0 } else { 0.0 }
            })
            .collect()
    }

    #[test]
    fn round_trips_within_bound() {
        for d in [
            vec![1000],
            vec![33, 70],
            vec![17, 9, 31],
            vec![1, 1, 5],
            vec![1],
        ] {
            let x = field(&d);
            for b in [1e-1, 1e-3, 1e-6] {
                let bytes = encode_f32(&x, &d, b);
                let Decoded::F32 { data, dims } = decode(&bytes).unwrap() else {
                    panic!()
                };
                assert_eq!(dims, d);
                for (a, c) in x.iter().zip(&data) {
                    assert!(((a - c) as f64).abs() <= b, "{d:?} b={b} {a} {c}");
                }
            }
        }
    }

    #[test]
    fn smooth_is_tiny() {
        let d = [64, 64, 64];
        let x: Vec<f32> = (0..d.iter().product::<usize>())
            .map(|i| ((i / 4096) as f32 * 0.05).sin() + ((i % 64) as f32 * 0.03).cos())
            .collect();
        let bytes = encode_f32(&x, &d, 1e-2);
        assert!(bytes.len() < 2000, "{}", bytes.len());
    }
}
