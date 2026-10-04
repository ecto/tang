//! Q2_0 expert rows on the CPU: AVX-VNNI (`vpdpbusd`, 256-bit), AVX2 (`vpmaddubsw` +
//! `vpmaddwd`), and a portable "lane" path that does the same arithmetic in the same order, so
//! the SIMD paths are bitwise equal to it. The spec is [`crate::contract::q2_row_dot`]; the lane
//! path differs from it only in fp32 summation order (see [`rows`]).
//!
//! The trick that makes this cheap: a 32-byte load of a row's codes is 128 weights (4 chunks of
//! 32). `(v >> 2f) & 0x03` per byte yields, in dword lane `i`, the codes of elements
//! `16h + 4b + f` of chunk `i / 2` (`h = i % 2`, byte `b`). That is exactly the order of the
//! int8 activation words in the [`QAct`] contract, so with activations regrouped per `f`
//! ([`XPrep`]) four `vpdpbusd` give eight per-half-chunk integer dots with no shuffles: the
//! "VNNI4" interleave of arXiv 2508.06753 comes for free from the GPU's layout.

use crate::contract::{QAct, QRow};

/// Instruction set used for a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Lane,
    Avx2,
    AvxVnni,
}

impl Isa {
    /// Best available, or `TANG_MOE_ISA=lane|avx2|vnni`.
    pub fn detect() -> Isa {
        let best = Self::best();
        match std::env::var("TANG_MOE_ISA").as_deref() {
            Ok("lane") => Isa::Lane,
            Ok("avx2") if best != Isa::Lane => Isa::Avx2,
            Ok("vnni") if best == Isa::AvxVnni => Isa::AvxVnni,
            _ => best,
        }
    }

    fn best() -> Isa {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c");
            if avx2 && is_x86_feature_detected!("avxvnni") {
                return Isa::AvxVnni;
            }
            if avx2 {
                return Isa::Avx2;
            }
        }
        Isa::Lane
    }

    /// Every ISA this machine can run (for tests and benches).
    pub fn available() -> Vec<Isa> {
        match Self::best() {
            Isa::AvxVnni => vec![Isa::Lane, Isa::Avx2, Isa::AvxVnni],
            Isa::Avx2 => vec![Isa::Lane, Isa::Avx2],
            Isa::Lane => vec![Isa::Lane],
        }
    }
}

/// One token's int8 activations of width `k` (`k % 128 == 0`), regrouped for the kernels.
/// Per 128-element group `g`: `x[4g + f]` is the 8 dwords whose lane `i` holds QAct word
/// `(4g + i/2)·8 + 4(i%2) + f` (chunk `4g + i/2`, half `i % 2`, extraction `f`); `dx[g]` is each
/// lane's chunk scale; `nhx[g]` is the chunk's −Σq in even lanes and 0 in odd ones.
#[derive(Clone, Default)]
#[repr(C)]
pub struct XPrep {
    pub k: usize,
    x: Vec<[u32; 8]>,
    dx: Vec<[f32; 8]>,
    nhx: Vec<[i32; 8]>,
}

impl XPrep {
    pub fn new(k: usize) -> Self {
        assert!(k.is_multiple_of(128));
        XPrep {
            k,
            x: vec![[0; 8]; k / 32],
            dx: vec![[0.0; 8]; k / 128],
            nhx: vec![[0; 8]; k / 128],
        }
    }

    /// Fill from row `r` of `QAct { m, k }` words.
    pub fn load(&mut self, xq: &[u32], l: QAct, r: usize) {
        assert_eq!(l.k, self.k);
        let (codes, scales, sums) = (l.codes(r), l.scales(r), l.sums(r));
        for g in 0..self.k / 128 {
            for f in 0..4 {
                for i in 0..8 {
                    self.x[4 * g + f][i] = xq[codes + (4 * g + i / 2) * 8 + 4 * (i % 2) + f];
                }
            }
            for i in 0..8 {
                let c = 4 * g + i / 2;
                self.dx[g][i] = f32::from_bits(xq[scales + c]);
                self.nhx[g][i] = if i % 2 == 0 {
                    -(xq[sums + c] as i32)
                } else {
                    0
                };
            }
        }
    }

    /// Quantize `x` (`k` floats) per the [`QAct`] contract straight into this layout. Same
    /// integers and scales as `quantize_act` followed by [`load`](Self::load).
    pub fn quantize(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.k);
        for c in 0..self.k / 32 {
            let xs = &x[c * 32..c * 32 + 32];
            let amax = xs.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0;
            let mut q = [0i8; 32];
            let mut sum = 0i32;
            if d != 0.0 {
                for (o, &v) in q.iter_mut().zip(xs) {
                    let qi = (v / d).round().clamp(-127.0, 127.0) as i32;
                    sum += qi;
                    *o = qi as i8;
                }
            }
            let (g, cl) = (c / 4, c % 4);
            for h in 0..2 {
                for f in 0..4 {
                    let w = u32::from_le_bytes([
                        q[16 * h + f] as u8,
                        q[16 * h + 4 + f] as u8,
                        q[16 * h + 8 + f] as u8,
                        q[16 * h + 12 + f] as u8,
                    ]);
                    self.x[4 * g + f][2 * cl + h] = w;
                }
                self.dx[g][2 * cl + h] = d;
            }
            self.nhx[g][2 * cl] = -sum;
            self.nhx[g][2 * cl + 1] = 0;
        }
    }

    pub fn from_qact(xq: &[u32], l: QAct, r: usize) -> Self {
        let mut p = Self::new(l.k);
        p.load(xq, l, r);
        p
    }
}

/// Where a matrix's 128-weight groups live in its bytes. Group `g` (64-weight blocks `2g`,
/// `2g + 1`) of row `r` has its 32 code bytes at `cb + g·cs` and its two fp16 scales at
/// `sb + g·ss`, where `(cb, cs, sb, ss) = layout.row(n, k, r)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// [`crate::contract::q2_repack`]: `[n][k/4]` codes, then `[n][k/64]` fp16 scales. `wb` is
    /// the matrix.
    Plain,
    /// tang-compute's `flash::ExpertBlob` gate/up tiles (`wb` is the whole blob): gu row
    /// `2r + up` is gate (`up = false`) or up row `r`; tile `i` holds gu rows `16i..16i + 16` as
    /// `[20 groups][16 rows]` × 32 B, scales (2 fp16 per group) in the same order in their own
    /// plane.
    GuTiled { up: bool },
    /// tang-compute's `flash::ExpertBlob` down tiles (`wb` is the whole blob): tile `i` holds rows
    /// `16i..16i + 16` as `[5 groups][16 rows]` × 32 B, scales likewise.
    DownTiled,
}

impl Layout {
    /// tang-compute `ExpertBlob` plane offsets.
    const GU_SCALES: usize = 2 * crate::contract::FF * crate::contract::HIDDEN / 4;
    const DOWN_CODES: usize = Self::GU_SCALES + 2 * crate::contract::FF * crate::contract::HIDDEN / 32;
    const DOWN_SCALES: usize = Self::DOWN_CODES + crate::contract::HIDDEN * crate::contract::FF / 4;

    /// `(cb, cs, sb, ss)` of row `r` of an `[n, k]` matrix.
    #[inline]
    pub fn row(self, n: usize, k: usize, r: usize) -> (usize, usize, usize, usize) {
        match self {
            Layout::Plain => (r * k / 4, 32, n * k / 4 + r * (k / 64) * 2, 4),
            Layout::GuTiled { up } => {
                let g = 2 * r + up as usize;
                let i = (g / 16) * (16 * crate::contract::HIDDEN / 128) + g % 16;
                (32 * i, 512, Self::GU_SCALES + 4 * i, 64)
            }
            Layout::DownTiled => {
                let i = (r / 16) * (16 * crate::contract::FF / 128) + r % 16;
                (Self::DOWN_CODES + 32 * i, 512, Self::DOWN_SCALES + 4 * i, 64)
            }
        }
    }

    /// The 32 code bytes of group `g` of a row.
    #[inline]
    fn pair<'a>(self, wb: &'a [u8], (cb, cs, _, _): (usize, usize, usize, usize), g: usize) -> &'a [u8] {
        &wb[cb + g * cs..cb + g * cs + 32]
    }
}

/// [`rows`] for a matrix stored in `layout` (`wb` as that layout says).
#[allow(clippy::too_many_arguments)]
pub fn rows_in(
    isa: Isa,
    layout: Layout,
    wb: &[u8],
    n: usize,
    k: usize,
    r0: usize,
    r1: usize,
    xs: &[&XPrep],
    out: &mut [f32],
) {
    let t = xs.len();
    assert!((1..=8).contains(&t) && r1 <= n && r0 <= r1);
    assert!(k.is_multiple_of(128) && out.len() >= (r1 - r0) * t);
    if r1 > r0 {
        let (cb, cs, sb, ss) = layout.row(n, k, r1 - 1);
        assert!(cb + (k / 128 - 1) * cs + 32 <= wb.len() && sb + (k / 128 - 1) * ss + 4 <= wb.len());
    }
    for x in xs {
        assert_eq!(x.k, k);
    }
    macro_rules! go {
        ($f:ident) => {
            match t {
                1 => $f::<1>(layout, wb, n, k, r0, r1, xs, out),
                2 => $f::<2>(layout, wb, n, k, r0, r1, xs, out),
                3 => $f::<3>(layout, wb, n, k, r0, r1, xs, out),
                4 => $f::<4>(layout, wb, n, k, r0, r1, xs, out),
                5 => $f::<5>(layout, wb, n, k, r0, r1, xs, out),
                6 => $f::<6>(layout, wb, n, k, r0, r1, xs, out),
                7 => $f::<7>(layout, wb, n, k, r0, r1, xs, out),
                _ => $f::<8>(layout, wb, n, k, r0, r1, xs, out),
            }
        };
    }
    match isa {
        Isa::Lane => go!(rows_lane),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { go!(rows_avx2) },
        #[cfg(target_arch = "x86_64")]
        Isa::AvxVnni => unsafe { go!(rows_vnni) },
        #[allow(unreachable_patterns)]
        _ => panic!("{isa:?} not available on this target"),
    }
}

/// Rows `r0..r1` of repacked Q2_0 `[n, k]` matrix `wb` against tokens `xs` (1..=8, all width
/// `k`). Writes `out[(r - r0) · T + t]`. Each 32-byte code load is decoded once and dotted
/// against every token.
///
/// Numerics: per chunk the integer `Σ code·q − Σ q` is exact (split over two lanes), then
/// `acc_lane = fma(float(s_lane), d_w · d_x, acc_lane)` and a fixed 8-lane tree sum. Against the
/// spec's single fma chain this only reassociates fp32 adds.
#[allow(clippy::too_many_arguments)]
pub fn rows(
    isa: Isa,
    wb: &[u8],
    n: usize,
    k: usize,
    r0: usize,
    r1: usize,
    xs: &[&XPrep],
    out: &mut [f32],
) {
    assert!(wb.len() >= n * k / 4 + n * k / 32);
    rows_in(isa, Layout::Plain, wb, n, k, r0, r1, xs, out)
}

/// The fixed tree the SIMD paths use to sum 8 lanes.
fn hsum8(a: [f32; 8]) -> f32 {
    let l = [a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7]];
    (l[0] + l[2]) + (l[1] + l[3])
}

fn scale_pair(wb: &[u8], (_, _, sb, ss): (usize, usize, usize, usize), g: usize) -> (f32, f32) {
    let (a, b) = (sb + g * ss, sb + g * ss + 2);
    (
        crate::contract::f16_to_f32(u16::from_le_bytes([wb[a], wb[a + 1]])),
        crate::contract::f16_to_f32(u16::from_le_bytes([wb[b], wb[b + 1]])),
    )
}

#[allow(clippy::too_many_arguments)]
fn rows_lane<const T: usize>(
    layout: Layout,
    wb: &[u8],
    n: usize,
    k: usize,
    r0: usize,
    r1: usize,
    xs: &[&XPrep],
    out: &mut [f32],
) {
    for r in r0..r1 {
        let mut acc = [[0f32; 8]; T];
        let ra = layout.row(n, k, r);
        for g in 0..k / 128 {
            let v = layout.pair(wb, ra, g);
            let (d0, d1) = scale_pair(wb, ra, g);
            for (t, acc) in acc.iter_mut().enumerate() {
                let x = xs[t];
                for i in 0..8 {
                    let mut s = 0i32;
                    for f in 0..4 {
                        let xw = x.x[4 * g + f][i];
                        for b in 0..4 {
                            let code = ((v[4 * i + b] >> (2 * f)) & 3) as i32;
                            s += code * (xw >> (8 * b)) as u8 as i8 as i32;
                        }
                    }
                    s += x.nhx[g][i];
                    let dw = if i < 4 { d0 } else { d1 };
                    acc[i] = (s as f32).mul_add(dw * x.dx[g][i], acc[i]);
                }
            }
        }
        for t in 0..T {
            out[(r - r0) * T + t] = hsum8(acc[t]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! simd_rows {
    ($name:ident, $inner:ident, $feat:literal, $dot:ident) => {
        /// Two rows at a time when there are ≤ 4 tokens (independent chains for the
        /// out-of-order core), one row otherwise (register pressure).
        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name<const T: usize>(
            layout: Layout,
            wb: &[u8],
            n: usize,
            k: usize,
            r0: usize,
            r1: usize,
            xs: &[&XPrep],
            out: &mut [f32],
        ) {
            if T <= 4 {
                let mut r = r0;
                while r + 2 <= r1 {
                    $inner::<T, 2>(layout, wb, n, k, r, r0, xs, out);
                    r += 2;
                }
                if r < r1 {
                    $inner::<T, 1>(layout, wb, n, k, r, r0, xs, out);
                }
            } else {
                for r in r0..r1 {
                    $inner::<T, 1>(layout, wb, n, k, r, r0, xs, out);
                }
            }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $inner<const T: usize, const R: usize>(
            layout: Layout,
            wb: &[u8],
            n: usize,
            k: usize,
            r: usize,
            r0: usize,
            xs: &[&XPrep],
            out: &mut [f32],
        ) {
            use std::arch::x86_64::*;
            let mask = _mm256_set1_epi8(3);
            let perm = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
            let base = wb.as_ptr();
            let mut ra = [(0usize, 0usize, 0usize, 0usize); R];
            for (ri, a) in ra.iter_mut().enumerate() {
                *a = layout.row(n, k, r + ri);
            }
            let mut acc = [[_mm256_setzero_ps(); T]; R];
            for g in 0..k / 128 {
                let mut v = [[_mm256_setzero_si256(); 4]; R];
                let mut dw = [_mm256_setzero_ps(); R];
                for (ri, (vr, dwr)) in v.iter_mut().zip(dw.iter_mut()).enumerate() {
                    let (cb, cs, sb, ss) = ra[ri];
                    let raw = _mm256_loadu_si256(base.add(cb + g * cs) as *const __m256i);
                    vr[0] = _mm256_and_si256(raw, mask);
                    vr[1] = _mm256_and_si256(_mm256_srli_epi32(raw, 2), mask);
                    vr[2] = _mm256_and_si256(_mm256_srli_epi32(raw, 4), mask);
                    vr[3] = _mm256_and_si256(_mm256_srli_epi32(raw, 6), mask);
                    let dpair = (base.add(sb + g * ss) as *const i32).read_unaligned();
                    *dwr = _mm256_permutevar8x32_ps(_mm256_cvtph_ps(_mm_set1_epi32(dpair)), perm);
                }
                for t in 0..T {
                    let xp = xs.get_unchecked(t);
                    let x = xp.x.as_ptr().add(4 * g) as *const __m256i;
                    let x0 = _mm256_loadu_si256(x);
                    let x1 = _mm256_loadu_si256(x.add(1));
                    let x2 = _mm256_loadu_si256(x.add(2));
                    let x3 = _mm256_loadu_si256(x.add(3));
                    let nhx = _mm256_loadu_si256(xp.nhx.as_ptr().add(g) as *const __m256i);
                    let dx = _mm256_loadu_ps(xp.dx.as_ptr().add(g) as *const f32);
                    for ri in 0..R {
                        let s = $dot(nhx, v[ri][0], v[ri][1], v[ri][2], v[ri][3], x0, x1, x2, x3);
                        let sc = _mm256_mul_ps(dw[ri], dx);
                        acc[ri][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sc, acc[ri][t]);
                    }
                }
            }
            for ri in 0..R {
                for t in 0..T {
                    let a = acc[ri][t];
                    let l = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
                    let h = _mm_add_ps(l, _mm_movehl_ps(l, l));
                    let s = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
                    *out.get_unchecked_mut((r + ri - r0) * T + t) = _mm_cvtss_f32(s);
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn dot_avx2(
    init: std::arch::x86_64::__m256i,
    v0: std::arch::x86_64::__m256i,
    v1: std::arch::x86_64::__m256i,
    v2: std::arch::x86_64::__m256i,
    v3: std::arch::x86_64::__m256i,
    x0: std::arch::x86_64::__m256i,
    x1: std::arch::x86_64::__m256i,
    x2: std::arch::x86_64::__m256i,
    x3: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // Each maddubs lane is ≤ 2·3·127 = 762, so four of them fit i16.
    let p = _mm256_add_epi16(
        _mm256_add_epi16(_mm256_maddubs_epi16(v0, x0), _mm256_maddubs_epi16(v1, x1)),
        _mm256_add_epi16(_mm256_maddubs_epi16(v2, x2), _mm256_maddubs_epi16(v3, x3)),
    );
    _mm256_add_epi32(init, _mm256_madd_epi16(p, _mm256_set1_epi16(1)))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn dot_vnni(
    init: std::arch::x86_64::__m256i,
    v0: std::arch::x86_64::__m256i,
    v1: std::arch::x86_64::__m256i,
    v2: std::arch::x86_64::__m256i,
    v3: std::arch::x86_64::__m256i,
    x0: std::arch::x86_64::__m256i,
    x1: std::arch::x86_64::__m256i,
    x2: std::arch::x86_64::__m256i,
    x3: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // Two independent chains, then one add.
    let a = _mm256_dpbusd_avx_epi32(init, v0, x0);
    let b = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), v1, x1);
    let a = _mm256_dpbusd_avx_epi32(a, v2, x2);
    let b = _mm256_dpbusd_avx_epi32(b, v3, x3);
    _mm256_add_epi32(a, b)
}

#[cfg(target_arch = "x86_64")]
simd_rows!(rows_avx2, rows_avx2_r, "avx2,fma,f16c", dot_avx2);
#[cfg(target_arch = "x86_64")]
simd_rows!(rows_vnni, rows_vnni_r, "avx2,fma,f16c,avxvnni", dot_vnni);

/// The integer part `Σ code·q − Σ q` of every chunk of row `o`, as the lane path splits it
/// (summed back per chunk), for the exact-integer parity test.
pub fn chunk_ints_lane(wb: &[u8], k: usize, o: usize, x: &XPrep) -> Vec<i32> {
    let crow = &wb[o * k / 4..(o + 1) * k / 4];
    let mut out = vec![0i32; k / 32];
    for g in 0..k / 128 {
        let v = &crow[g * 32..g * 32 + 32];
        for i in 0..8 {
            let mut s = 0i32;
            for f in 0..4 {
                let xw = x.x[4 * g + f][i];
                for b in 0..4 {
                    s +=
                        ((v[4 * i + b] >> (2 * f)) & 3) as i32 * (xw >> (8 * b)) as u8 as i8 as i32;
                }
            }
            out[4 * g + i / 2] += s + x.nhx[g][i];
        }
    }
    out
}

/// Spec integer parts, for comparison with [`chunk_ints_lane`].
pub fn chunk_ints_ref(wb: &[u8], k: usize, o: usize, x: &QRow) -> Vec<i32> {
    (0..k / 32)
        .map(|c| crate::contract::q2_chunk_int(wb, k, o, c, x))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::contract::*;

    pub struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        pub fn f(&mut self) -> f32 {
            (self.next() % 20001) as f32 / 10000.0 - 1.0
        }
    }

    /// A random repacked `[n, k]` matrix with scales around `scale`.
    pub fn matrix(rng: &mut Rng, n: usize, k: usize, scale: f32) -> Vec<u8> {
        let mut raw = vec![0u8; q2_bytes(n, k)];
        for b in 0..n * k / 64 {
            let d = f32_to_f16(scale * (0.5 + (rng.next() % 1000) as f32 / 1000.0));
            raw[b * 18..b * 18 + 2].copy_from_slice(&d.to_le_bytes());
            for j in 0..16 {
                raw[b * 18 + 2 + j] = rng.next() as u8;
            }
        }
        q2_repack(&raw, n, k)
    }

    pub fn blob(rng: &mut Rng) -> Vec<u8> {
        let mut b = matrix(rng, FF, HIDDEN, 0.02);
        b.extend(matrix(rng, FF, HIDDEN, 0.02));
        b.extend(matrix(rng, HIDDEN, FF, 0.02));
        b
    }

    pub fn acts(rng: &mut Rng, m: usize, k: usize) -> Vec<u32> {
        let x: Vec<f32> = (0..m * k).map(|_| rng.f() * 3.0).collect();
        quantize_act(&x, m, k)
    }

    #[test]
    fn direct_quantize_matches_contract() {
        let mut rng = Rng(5);
        for k in [640, 2560] {
            let mut x: Vec<f32> = (0..k).map(|_| rng.f() * 4.0).collect();
            x[33] = 0.5 * 127.0 / 127.0; // exercise ties loosely
            x[64..96].fill(0.0); // a zero chunk
            let a = XPrep::from_qact(&quantize_act(&x, 1, k), QAct { m: 1, k }, 0);
            let mut b = XPrep::new(k);
            b.quantize(&x);
            assert_eq!(a.x, b.x);
            assert_eq!(a.nhx, b.nhx);
            assert!(a
                .dx
                .iter()
                .flatten()
                .zip(b.dx.iter().flatten())
                .all(|(p, q)| p.to_bits() == q.to_bits()));
        }
    }

    /// A `contract::ExpertBlob` rearranged into tang-compute's tiled `flash::ExpertBlob`.
    fn tile(plain: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; ExpertBlob::BYTES];
        let mut put = |src: &[u8], n: usize, k: usize, lay: Layout| {
            for r in 0..n {
                let (cb, cs, sb, ss) = lay.row(n, k, r);
                let (pcb, pcs, psb, pss) = Layout::Plain.row(n, k, r);
                for g in 0..k / 128 {
                    out[cb + g * cs..cb + g * cs + 32]
                        .copy_from_slice(&src[pcb + g * pcs..pcb + g * pcs + 32]);
                    out[sb + g * ss..sb + g * ss + 4]
                        .copy_from_slice(&src[psb + g * pss..psb + g * pss + 4]);
                }
            }
        };
        put(&plain[ExpertBlob::GATE..ExpertBlob::UP], FF, HIDDEN, Layout::GuTiled { up: false });
        put(&plain[ExpertBlob::UP..ExpertBlob::DOWN], FF, HIDDEN, Layout::GuTiled { up: true });
        put(&plain[ExpertBlob::DOWN..], HIDDEN, FF, Layout::DownTiled);
        out
    }

    #[test]
    fn tiled_layout_matches_plain_bitwise() {
        let mut rng = Rng(13);
        let plain = blob(&mut rng);
        let tiled = tile(&plain);
        let xq = acts(&mut rng, 3, HIDDEN);
        let hq = acts(&mut rng, 3, FF);
        for t in [1, 3] {
            let xs: Vec<XPrep> = (0..t).map(|r| XPrep::from_qact(&xq, QAct { m: 3, k: HIDDEN }, r)).collect();
            let hs: Vec<XPrep> = (0..t).map(|r| XPrep::from_qact(&hq, QAct { m: 3, k: FF }, r)).collect();
            let xr: Vec<&XPrep> = xs.iter().collect();
            let hr: Vec<&XPrep> = hs.iter().collect();
            for isa in Isa::available() {
                for (lay, pw, n, k, x) in [
                    (Layout::GuTiled { up: false }, &plain[ExpertBlob::GATE..ExpertBlob::UP], FF, HIDDEN, &xr),
                    (Layout::GuTiled { up: true }, &plain[ExpertBlob::UP..ExpertBlob::DOWN], FF, HIDDEN, &xr),
                    (Layout::DownTiled, &plain[ExpertBlob::DOWN..], HIDDEN, FF, &hr),
                ] {
                    let (r0, r1) = (5, n - 3);
                    let mut a = vec![0f32; (r1 - r0) * t];
                    let mut b = vec![1f32; (r1 - r0) * t];
                    rows(isa, pw, n, k, r0, r1, x, &mut a);
                    rows_in(isa, lay, &tiled, n, k, r0, r1, x, &mut b);
                    assert!(
                        a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
                        "{isa:?} {lay:?} t={t}"
                    );
                }
            }
        }
    }

    #[test]
    fn integer_parts_exact() {
        let mut rng = Rng(7);
        let (n, k) = (16, 2560);
        let w = matrix(&mut rng, n, k, 0.01);
        let xq = acts(&mut rng, 2, k);
        let l = QAct { m: 2, k };
        for r in 0..2 {
            let p = XPrep::from_qact(&xq, l, r);
            let q = QRow::decode(&xq, l, r);
            for o in 0..n {
                assert_eq!(chunk_ints_lane(&w, k, o, &p), chunk_ints_ref(&w, k, o, &q));
            }
        }
    }

    #[test]
    fn every_isa_matches_lane_bitwise_and_spec_closely() {
        let mut rng = Rng(11);
        for (n, k) in [(64, 2560), (96, 640)] {
            let w = matrix(&mut rng, n, k, 0.02);
            let xq = acts(&mut rng, 8, k);
            let l = QAct { m: 8, k };
            let preps: Vec<XPrep> = (0..8).map(|r| XPrep::from_qact(&xq, l, r)).collect();
            let rowsq: Vec<QRow> = (0..8).map(|r| QRow::decode(&xq, l, r)).collect();
            for t in 1..=8 {
                let xs: Vec<&XPrep> = preps[..t].iter().collect();
                let mut lane = vec![0f32; n * t];
                rows(Isa::Lane, &w, n, k, 0, n, &xs, &mut lane);
                for isa in Isa::available() {
                    let mut o = vec![0f32; n * t];
                    // Split the range to exercise r0 > 0.
                    rows(isa, &w, n, k, 0, n / 2, &xs, &mut o[..n / 2 * t]);
                    rows(isa, &w, n, k, n / 2, n, &xs, &mut o[n / 2 * t..]);
                    assert!(
                        o.iter().zip(&lane).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{isa:?} t={t} k={k} differs from lane path"
                    );
                }
                for r in 0..n {
                    for (tt, rq) in rowsq.iter().enumerate().take(t) {
                        let spec = q2_row_dot(&w, n, k, r, rq);
                        let got = lane[r * t + tt];
                        // Worst-case bound for reassociating two ≤80-term fp32 sums: ~(80 + 23)·2^-24·Σ|term| ≈ 6e-6; allow 2e-5.
                        let mag: f32 = (0..k / 32)
                            .map(|c| {
                                let b = n * k / 4 + r * (k / 64) * 2 + 2 * (c / 2);
                                let d = f16_to_f32(u16::from_le_bytes([w[b], w[b + 1]]));
                                (d * rq.d[c] * q2_chunk_int(&w, k, r, c, rq) as f32).abs()
                            })
                            .sum();
                        assert!(
                            (got - spec).abs() <= 2e-5 * mag + 1e-30,
                            "row {r} tok {tt}: {got} vs spec {spec} (Σ|terms| {mag})"
                        );
                    }
                }
            }
        }
    }
}
