//! Separable multi-level 3D wavelet transforms by lifting.
//!
//! - **CDF 9/7** (f64 lifting, the JPEG 2000 irreversible filter): the lossy
//!   path. Float, but plain IEEE `+`/`*` in a fixed order with no fused ops,
//!   so the CPU decoder is deterministic.
//! - **LeGall 5/3** (integer lifting, the JPEG 2000 reversible filter): the
//!   lossless path, exact on `i32`.
//!
//! Both use whole-sample symmetric extension and work on any length ≥ 1 (odd
//! lengths keep the extra sample in the low band). The pyramid is dyadic on the
//! LL region; at each level every axis whose current length is ≥ [`MIN_LEN`]
//! is split, so a 64×128×128 block gets 4 levels on its short axis and 5 on
//! the others (anisotropic bands where axes stop early). Arrays are always
//! handled as 3D, with leading unit axes for 1D/2D input.

/// An axis is split at a level only while its LL length is at least this.
pub const MIN_LEN: usize = 8;
/// Level cap for 2D/3D arrays.
pub const MAX_LEVELS: usize = 6;
/// Level cap for 1D signals (and the size of the basis-norm table).
pub const MAX_LEVELS_1D: usize = 13;

const ALPHA: f64 = -1.586_134_342_059_924;
const BETA: f64 = -0.052_980_118_572_961;
const GAMMA: f64 = 0.882_911_075_530_934;
const DELTA: f64 = 0.443_506_852_043_971;
const KAPPA: f64 = 1.149_604_398_860_241;

/// A subband: level `level` (0 = finest), orientation `mask` (bit `a` set =
/// highpass along axis `a`; `mask == 0` is the final LL), half-open region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Band {
    /// Decomposition level (0 = finest detail); the LL band has `level = nlev`.
    pub level: usize,
    /// Highpass axes bitmask.
    pub mask: u8,
    /// Region start per axis.
    pub start: [usize; 3],
    /// Region end per axis (exclusive).
    pub end: [usize; 3],
}

impl Band {
    /// Number of coefficients.
    pub fn len(&self) -> usize {
        (0..3).map(|a| self.end[a] - self.start[a]).product()
    }
    /// Whether the band is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The dyadic pyramid for a shape.
#[derive(Clone, Debug)]
pub struct Pyramid {
    /// Shape (3D, row-major, axis 2 fastest).
    pub dims: [usize; 3],
    /// `lens[l]` = LL extent entering level `l` (`lens[0] = dims`).
    pub lens: Vec<[usize; 3]>,
    /// `split[l][a]`: axis `a` is transformed at level `l`.
    pub split: Vec<[bool; 3]>,
    /// Bands in coding order: LL first, then detail bands coarse → fine.
    pub bands: Vec<Band>,
}

/// Pad an n-D shape (n ≤ 3) to 3D with leading unit axes.
pub fn as3(dims: &[usize]) -> [usize; 3] {
    assert!(!dims.is_empty() && dims.len() <= 3, "1D–3D arrays only");
    let mut d = [1; 3];
    d[3 - dims.len()..].copy_from_slice(dims);
    d
}

impl Pyramid {
    /// Build the pyramid for `dims` with at most `max_levels` levels.
    pub fn new(dims: [usize; 3], max_levels: usize) -> Self {
        let mut lens = vec![dims];
        let mut split = Vec::new();
        while split.len() < max_levels {
            let cur = *lens.last().unwrap();
            let ml: usize = std::env::var("TC_MINLEN").ok().and_then(|v| v.parse().ok()).unwrap_or(MIN_LEN);
            let s = [0, 1, 2].map(|a| cur[a] >= ml);
            if !s.iter().any(|&b| b) {
                break;
            }
            split.push(s);
            lens.push([0, 1, 2].map(|a| if s[a] { cur[a].div_ceil(2) } else { cur[a] }));
        }
        let nlev = split.len();
        let mut bands = vec![Band { level: nlev, mask: 0, start: [0; 3], end: lens[nlev] }];
        for l in (0..nlev).rev() {
            for mask in 1u8..8 {
                if (0..3).any(|a| mask >> a & 1 == 1 && !split[l][a]) {
                    continue;
                }
                let mut start = [0; 3];
                let mut end = [0; 3];
                for a in 0..3 {
                    if mask >> a & 1 == 1 {
                        start[a] = lens[l + 1][a];
                        end[a] = lens[l][a];
                    } else {
                        end[a] = lens[l + 1][a];
                    }
                }
                let b = Band { level: l, mask, start, end };
                if !b.is_empty() {
                    bands.push(b);
                }
            }
        }
        Self { dims, lens, split, bands }
    }

    /// Number of levels.
    pub fn nlev(&self) -> usize {
        self.split.len()
    }

    /// Total cells.
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    /// Whether the shape is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many times axis `a` is split in levels `0..l`.
    fn splits_before(&self, a: usize, l: usize) -> usize {
        self.split[..l].iter().filter(|s| s[a]).count()
    }

    /// L2 norm of the CDF 9/7 synthesis basis function of a coefficient in
    /// `band` (separable product of 1D norms). Scaling coefficients by it makes
    /// one uniform step equal-slope (MSE-matched) across bands.
    pub fn band_weight(&self, band: &Band) -> f64 {
        let (lo, hi) = norms_97();
        (0..3)
            .map(|a| {
                let l = band.level;
                if l < self.nlev() && self.split[l][a] {
                    let k = self.splits_before(a, l) + 1;
                    if band.mask >> a & 1 == 1 {
                        hi[k]
                    } else {
                        lo[k]
                    }
                } else {
                    lo[self.splits_before(a, l.min(self.nlev()))]
                }
            })
            .product()
    }

    /// The coarser-level coefficient ("parent") of position `p` in detail band
    /// `bi`: same orientation one level up, or `None` if that band does not
    /// exist (coarsest detail level, or the axis set stops splitting).
    pub fn parent_band(&self, bi: usize) -> Option<usize> {
        let b = &self.bands[bi];
        if b.mask == 0 || b.level + 1 >= self.nlev() {
            return None;
        }
        self.bands.iter().position(|c| c.level == b.level + 1 && c.mask == b.mask)
    }

    /// Map a position in band `b` to its parent position in band `pb`.
    #[inline]
    pub fn parent_pos(&self, b: &Band, pb: &Band, p: [usize; 3]) -> [usize; 3] {
        let mut q = [0; 3];
        for a in 0..3 {
            let rel = p[a] - b.start[a];
            let r = if self.split[pb.level][a] { rel / 2 } else { rel };
            q[a] = (pb.start[a] + r).min(pb.end[a] - 1);
        }
        q
    }
}

fn lines(dims: [usize; 3], ext: [usize; 3], axis: usize, mut f: impl FnMut(usize, usize)) {
    // f(offset of first element, stride) for every line along `axis` inside `ext`
    let stride = [dims[1] * dims[2], dims[2], 1];
    let others: Vec<usize> = (0..3).filter(|&a| a != axis).collect();
    for i in 0..ext[others[0]] {
        for j in 0..ext[others[1]] {
            f(i * stride[others[0]] + j * stride[others[1]], stride[axis]);
        }
    }
}

// ---------------------------------------------------------------- CDF 9/7

#[inline]
fn lift_d(s: &[f64], d: &mut [f64], c: f64) {
    // d_i += c (s_i + s_{i+1}), s_{ns} mirrors to s_{ns-1}
    let ns = s.len();
    for i in 0..d.len() {
        let r = if i + 1 < ns { s[i + 1] } else { s[ns - 1] };
        d[i] += c * (s[i] + r);
    }
}

#[inline]
fn lift_s(s: &mut [f64], d: &[f64], c: f64) {
    // s_i += c (d_{i-1} + d_i), d_{-1} = d_0, d_{nd} = d_{nd-1}
    let nd = d.len();
    for i in 0..s.len() {
        let l = if i == 0 { d[0] } else { d[i - 1] };
        let r = if i < nd { d[i] } else { d[nd - 1] };
        s[i] += c * (l + r);
    }
}

/// One level of forward CDF 9/7 on `x` (in place: low half first).
pub fn fwd97(x: &mut [f64], tmp: &mut Vec<f64>) {
    let n = x.len();
    if n < 2 {
        return;
    }
    let ns = n.div_ceil(2);
    tmp.clear();
    tmp.extend(x.iter().step_by(2));
    tmp.extend(x.iter().skip(1).step_by(2));
    let (s, d) = tmp.split_at_mut(ns);
    lift_d(s, d, ALPHA);
    lift_s(s, d, BETA);
    lift_d(s, d, GAMMA);
    lift_s(s, d, DELTA);
    for v in s.iter_mut() {
        *v *= KAPPA;
    }
    for v in d.iter_mut() {
        *v /= KAPPA;
    }
    x.copy_from_slice(tmp);
}

/// Inverse of [`fwd97`].
pub fn inv97(x: &mut [f64], tmp: &mut Vec<f64>) {
    let n = x.len();
    if n < 2 {
        return;
    }
    let ns = n.div_ceil(2);
    tmp.clear();
    tmp.extend_from_slice(x);
    let (s, d) = tmp.split_at_mut(ns);
    for v in s.iter_mut() {
        *v /= KAPPA;
    }
    for v in d.iter_mut() {
        *v *= KAPPA;
    }
    lift_s(s, d, -DELTA);
    lift_d(s, d, -GAMMA);
    lift_s(s, d, -BETA);
    lift_d(s, d, -ALPHA);
    for i in 0..ns {
        x[2 * i] = s[i];
    }
    for i in 0..n / 2 {
        x[2 * i + 1] = d[i];
    }
}

/// 1D synthesis-basis norms `(lo[k], hi[k])` after `k` splits (`lo[0] = 1`).
pub fn norms_97() -> (&'static [f64], &'static [f64]) {
    use std::sync::OnceLock;
    static N: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    let n = N.get_or_init(|| {
        let len = 1 << 16;
        let mut lo = vec![1.0];
        let mut hi = vec![0.0];
        let mut tmp = Vec::new();
        for k in 1..=MAX_LEVELS_1D {
            // lengths of the LL region per split
            let mut ls: Vec<usize> = vec![len];
            for _ in 0..k {
                ls.push(ls.last().unwrap().div_ceil(2usize));
            }
            let norm = |pos: usize, tmp: &mut Vec<f64>| {
                let mut x = vec![0.0; len];
                x[pos] = 1.0;
                for j in (0..k).rev() {
                    inv97(&mut x[..ls[j]], tmp);
                }
                x.iter().map(|v| v * v).sum::<f64>().sqrt()
            };
            lo.push(norm(ls[k] / 2, &mut tmp));
            hi.push(norm(ls[k] + (ls[k - 1] - ls[k]) / 2, &mut tmp));
        }
        (lo, hi)
    });
    (&n.0, &n.1)
}

fn apply3<T: Copy + Default>(
    p: &Pyramid,
    x: &mut [T],
    inverse: bool,
    mut f: impl FnMut(&mut [T], &mut Vec<T>, bool),
) {
    let mut line = Vec::new();
    let mut tmp = Vec::new();
    let levels: Vec<usize> = if inverse { (0..p.nlev()).rev().collect() } else { (0..p.nlev()).collect() };
    for l in levels {
        let axes: Vec<usize> = if inverse { vec![2, 1, 0] } else { vec![0, 1, 2] };
        for a in axes {
            if !p.split[l][a] {
                continue;
            }
            let ext = p.lens[l];
            let n = ext[a];
            lines(p.dims, ext, a, |off, st| {
                line.clear();
                line.extend((0..n).map(|i| x[off + i * st]));
                f(&mut line, &mut tmp, inverse);
                for i in 0..n {
                    x[off + i * st] = line[i];
                }
            });
        }
    }
}

/// Forward multi-level CDF 9/7 in place.
pub fn forward_97(p: &Pyramid, x: &mut [f64]) {
    apply3(p, x, false, |l, t, _| fwd97(l, t));
}

/// Inverse multi-level CDF 9/7 in place.
pub fn inverse_97(p: &Pyramid, x: &mut [f64]) {
    apply3(p, x, true, |l, t, _| inv97(l, t));
}

// ---------------------------------------------------------------- LeGall 5/3 (integer)

/// One level of forward integer 5/3 (reversible) on `x`.
pub fn fwd53(x: &mut [i64], tmp: &mut Vec<i64>) {
    let n = x.len();
    if n < 2 {
        return;
    }
    let ns = n.div_ceil(2);
    tmp.clear();
    tmp.extend(x.iter().step_by(2));
    tmp.extend(x.iter().skip(1).step_by(2));
    let (s, d) = tmp.split_at_mut(ns);
    for i in 0..d.len() {
        let r = if i + 1 < ns { s[i + 1] } else { s[ns - 1] };
        d[i] -= (s[i] + r).div_euclid(2);
    }
    let nd = d.len();
    for i in 0..ns {
        let l = if i == 0 { d[0] } else { d[i - 1] };
        let r = if i < nd { d[i] } else { d[nd - 1] };
        s[i] += (l + r + 2).div_euclid(4);
    }
    x.copy_from_slice(tmp);
}

/// Inverse of [`fwd53`].
pub fn inv53(x: &mut [i64], tmp: &mut Vec<i64>) {
    let n = x.len();
    if n < 2 {
        return;
    }
    let ns = n.div_ceil(2);
    tmp.clear();
    tmp.extend_from_slice(x);
    let (s, d) = tmp.split_at_mut(ns);
    let nd = d.len();
    for i in 0..ns {
        let l = if i == 0 { d[0] } else { d[i - 1] };
        let r = if i < nd { d[i] } else { d[nd - 1] };
        s[i] -= (l + r + 2).div_euclid(4);
    }
    for i in 0..nd {
        let r = if i + 1 < ns { s[i + 1] } else { s[ns - 1] };
        d[i] += (s[i] + r).div_euclid(2);
    }
    for i in 0..ns {
        x[2 * i] = s[i];
    }
    for i in 0..nd {
        x[2 * i + 1] = d[i];
    }
}

/// Forward multi-level integer 5/3 in place.
pub fn forward_53(p: &Pyramid, x: &mut [i64]) {
    apply3(p, x, false, |l, t, _| fwd53(l, t));
}

/// Inverse multi-level integer 5/3 in place.
pub fn inverse_53(p: &Pyramid, x: &mut [i64]) {
    apply3(p, x, true, |l, t, _| inv53(l, t));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(d: [usize; 3]) -> Vec<f64> {
        let mut v = Vec::new();
        for i in 0..d[0] {
            for j in 0..d[1] {
                for k in 0..d[2] {
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    v.push((0.3 * x).sin() + (0.11 * y * z).cos() + 0.01 * x * y - (k % 7) as f64 * 0.1);
                }
            }
        }
        v
    }

    #[test]
    fn perfect_reconstruction_97_any_size() {
        for d in [[1, 1, 1], [1, 1, 2], [1, 1, 37], [3, 17, 9], [13, 8, 31], [16, 16, 16], [1, 50, 23]] {
            let p = Pyramid::new(d, MAX_LEVELS);
            let x = field(d);
            let mut y = x.clone();
            forward_97(&p, &mut y);
            inverse_97(&p, &mut y);
            let e = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(e < 1e-11, "{d:?}: {e}");
            let cells: usize = p.bands.iter().map(|b| b.len()).sum();
            assert_eq!(cells, p.len(), "bands tile the array {d:?}");
        }
    }

    #[test]
    fn lossless_53_any_size() {
        for d in [[1, 1, 5], [7, 9, 11], [16, 32, 8], [2, 1, 64]] {
            let p = Pyramid::new(d, MAX_LEVELS);
            let x: Vec<i64> = field(d).iter().map(|v| (v * 1000.0) as i64 ^ 0x5a).collect();
            let mut y = x.clone();
            forward_53(&p, &mut y);
            if p.nlev() > 0 {
                assert_ne!(x, y);
            }
            inverse_53(&p, &mut y);
            assert_eq!(x, y);
        }
    }

    #[test]
    fn weights_near_orthonormal() {
        let (lo, hi) = norms_97();
        // CDF 9/7 with this scaling is close to orthonormal
        for k in 1..=MAX_LEVELS {
            assert!((lo[k] - 1.0).abs() < 0.1 && (hi[k] - 1.0).abs() < 0.15, "{k}: {} {}", lo[k], hi[k]);
        }
    }

    #[test]
    fn parent_positions_in_band() {
        let p = Pyramid::new([64, 128, 128], MAX_LEVELS);
        assert_eq!(p.nlev(), 5);
        for (bi, b) in p.bands.iter().enumerate() {
            if let Some(pi) = p.parent_band(bi) {
                let pb = &p.bands[pi];
                let q = p.parent_pos(b, pb, [b.end[0] - 1, b.end[1] - 1, b.end[2] - 1]);
                for a in 0..3 {
                    assert!(q[a] >= pb.start[a] && q[a] < pb.end[a]);
                }
            }
        }
    }
}
