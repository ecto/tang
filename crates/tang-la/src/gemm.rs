//! Row-major GEMM kernels.
//!
//! Three variants cover every product the tensor and training layers need
//! without materializing a transposed copy first:
//!
//! - [`gemm_nn`] — `C = A · B`
//! - [`gemm_nt`] — `C = A · Bᵀ`, the linear-layer forward (weights are
//!   `[out, in]`, so the contraction runs along contiguous rows of both operands)
//! - [`gemm_tn`] — `C = Aᵀ · B`, the weight gradient
//!
//! All three are blocked over `k` so the streamed operand stays resident in
//! cache, and register-blocked 4 rows at a time so each load of the inner
//! operand feeds four independent FMA chains. For `f32` and `f64` they
//! dispatch to NEON microkernels; every other `Scalar` (notably `Dual`) takes
//! the generic path, which keeps the same blocking but leaves vectorization to
//! the compiler.
//!
//! `C` is accumulated into, never zeroed here — callers pass a zeroed buffer
//! for a plain product or a live one to fuse an accumulation.
//!
//! On x86 there is currently no hand-written path: `f32`/`f64` fall through to
//! the generic kernels, which the compiler vectorizes only as well as its
//! auto-vectorizer manages. Adding AVX2/FMA microkernels means writing two more
//! `kernels!` expansions and extending the dispatch below.

// The kernels take BLAS-style `(m, n, k, a, lda, b, ldb, c, ldc)` signatures.
// Splitting that into structs would obscure a shape every reader of numerical
// code already knows.
#![allow(clippy::too_many_arguments)]

// Only the NEON dispatch needs to identify concrete float types; on other
// targets every `Scalar` takes the generic path.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use core::any::TypeId;
use tang::Scalar;

/// Depth of a `k` panel. `KC` rows of the streamed operand should sit in L2.
const KC: usize = 256;

/// Multiply-accumulate count below which spawning threads costs more than it
/// saves. Tuned to roughly a 64×64×64 product.
#[cfg(feature = "threads")]
const PAR_MIN_WORK: usize = 1 << 18;

/// Run `body` over row-blocks of `C` across the rayon pool, returning `false`
/// if the problem is too small to be worth splitting.
///
/// Splitting on rows (rather than columns) is what keeps this free of unsafe:
/// `par_chunks_mut` hands each task a genuinely disjoint `&mut [S]`, so no
/// aliasing reasoning is required. The cost is that a product with very few
/// rows — a small training batch, say — gets limited parallelism; the callers
/// that matter most (`DMat::mul_mat`, which maps its column count onto `m`,
/// and the wide output projections) all present a large `m` here.
#[cfg(feature = "threads")]
fn par_rows<S: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [S],
    ldc: usize,
    body: impl Fn(usize, usize, &mut [S]) + Send + Sync,
) -> bool {
    use rayon::prelude::*;

    let threads = rayon::current_num_threads();
    if threads < 2 || m * n * k < PAR_MIN_WORK {
        return false;
    }
    // Keep whole 4-row register tiles inside a task.
    let rows_per = m.div_ceil(threads).next_multiple_of(4);
    if rows_per >= m {
        return false;
    }
    c[..m * ldc]
        .par_chunks_mut(rows_per * ldc)
        .enumerate()
        .for_each(|(t, cc)| {
            let i0 = t * rows_per;
            body(i0, (m - i0).min(rows_per), cc);
        });
    true
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// `C[m,n] += A[m,k] · B[k,n]`, all row-major.
pub fn gemm_nn<S: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    a: &[S],
    lda: usize,
    b: &[S],
    ldb: usize,
    c: &mut [S],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    #[cfg(feature = "threads")]
    {
        // `body` re-enters the serial path on a disjoint row block.
        if par_rows(m, n, k, c, ldc, |i0, rows, cc| {
            gemm_nn_serial(rows, n, k, &a[i0 * lda..], lda, b, ldb, cc, ldc)
        }) {
            return;
        }
    }
    gemm_nn_serial(m, n, k, a, lda, b, ldb, c, ldc);
}

fn gemm_nn_serial<S: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    a: &[S],
    lda: usize,
    b: &[S],
    ldb: usize,
    c: &mut [S],
    ldc: usize,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if TypeId::of::<S>() == TypeId::of::<f32>() {
            // SAFETY: TypeId equality proves S is f32, so the reinterpret is
            // an identity cast; lengths are unchanged.
            unsafe {
                return neon::nn_f32(m, n, k, cast(a), lda, cast(b), ldb, cast_mut(c), ldc);
            }
        }
        if TypeId::of::<S>() == TypeId::of::<f64>() {
            unsafe {
                return neon::nn_f64(m, n, k, cast(a), lda, cast(b), ldb, cast_mut(c), ldc);
            }
        }
    }
    generic_nn(m, n, k, a, lda, b, ldb, c, ldc);
}

/// `C[m,n] += A[m,k] · B[n,k]ᵀ`, all row-major.
pub fn gemm_nt<S: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    a: &[S],
    lda: usize,
    b: &[S],
    ldb: usize,
    c: &mut [S],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    #[cfg(feature = "threads")]
    {
        // `body` re-enters the serial path on a disjoint row block.
        if par_rows(m, n, k, c, ldc, |i0, rows, cc| {
            gemm_nt_serial(rows, n, k, &a[i0 * lda..], lda, b, ldb, cc, ldc)
        }) {
            return;
        }
    }
    gemm_nt_serial(m, n, k, a, lda, b, ldb, c, ldc);
}

fn gemm_nt_serial<S: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    a: &[S],
    lda: usize,
    b: &[S],
    ldb: usize,
    c: &mut [S],
    ldc: usize,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if TypeId::of::<S>() == TypeId::of::<f32>() {
            unsafe {
                return neon::nt_f32(m, n, k, cast(a), lda, cast(b), ldb, cast_mut(c), ldc);
            }
        }
        if TypeId::of::<S>() == TypeId::of::<f64>() {
            unsafe {
                return neon::nt_f64(m, n, k, cast(a), lda, cast(b), ldb, cast_mut(c), ldc);
            }
        }
    }
    generic_nt(m, n, k, a, lda, b, ldb, c, ldc);
}

/// `C[m,n] += A[k,m]ᵀ · B[k,n]`, all row-major.
pub fn gemm_tn<S: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    a: &[S],
    lda: usize,
    b: &[S],
    ldb: usize,
    c: &mut [S],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    #[cfg(feature = "threads")]
    {
        // `body` re-enters the serial path on a disjoint row block.
        if par_rows(m, n, k, c, ldc, |i0, rows, cc| {
            gemm_tn_serial(rows, n, k, &a[i0..], lda, b, ldb, cc, ldc)
        }) {
            return;
        }
    }
    gemm_tn_serial(m, n, k, a, lda, b, ldb, c, ldc);
}

fn gemm_tn_serial<S: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    a: &[S],
    lda: usize,
    b: &[S],
    ldb: usize,
    c: &mut [S],
    ldc: usize,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if TypeId::of::<S>() == TypeId::of::<f32>() {
            unsafe {
                return neon::tn_f32(m, n, k, cast(a), lda, cast(b), ldb, cast_mut(c), ldc);
            }
        }
        if TypeId::of::<S>() == TypeId::of::<f64>() {
            unsafe {
                return neon::tn_f64(m, n, k, cast(a), lda, cast(b), ldb, cast_mut(c), ldc);
            }
        }
    }
    generic_tn(m, n, k, a, lda, b, ldb, c, ldc);
}

// ---------------------------------------------------------------------------
// Type-erasure helpers
// ---------------------------------------------------------------------------

/// Reinterpret `&[S]` as `&[T]`.
///
/// # Safety
/// Only call after proving `TypeId::of::<S>() == TypeId::of::<T>()`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn cast<S, T>(x: &[S]) -> &[T] {
    core::slice::from_raw_parts(x.as_ptr() as *const T, x.len())
}

/// Mutable counterpart of [`cast`].
///
/// # Safety
/// Same contract as [`cast`].
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn cast_mut<S, T>(x: &mut [S]) -> &mut [T] {
    core::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut T, x.len())
}

// ---------------------------------------------------------------------------
// Generic kernels
// ---------------------------------------------------------------------------

fn generic_nn<S: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    a: &[S],
    lda: usize,
    b: &[S],
    ldb: usize,
    c: &mut [S],
    ldc: usize,
) {
    for k0 in (0..k).step_by(KC) {
        let k1 = (k0 + KC).min(k);
        for i in 0..m {
            let arow = &a[i * lda..i * lda + k];
            let crow = &mut c[i * ldc..i * ldc + n];
            for (kk, &aik) in arow.iter().enumerate().take(k1).skip(k0) {
                let brow = &b[kk * ldb..kk * ldb + n];
                for (cj, &bj) in crow.iter_mut().zip(brow.iter()) {
                    *cj = cj.alg_add(aik.alg_mul(bj));
                }
            }
        }
    }
}

fn generic_nt<S: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    a: &[S],
    lda: usize,
    b: &[S],
    ldb: usize,
    c: &mut [S],
    ldc: usize,
) {
    for i in 0..m {
        let arow = &a[i * lda..i * lda + k];
        for j in 0..n {
            let brow = &b[j * ldb..j * ldb + k];
            // Four independent chains so the reduction is not serialized on
            // one accumulator; the tail below picks up k % 4.
            let (mut s0, mut s1, mut s2, mut s3) = (S::ZERO, S::ZERO, S::ZERO, S::ZERO);
            let mut kk = 0;
            while kk + 4 <= k {
                s0 = s0.alg_add(arow[kk].alg_mul(brow[kk]));
                s1 = s1.alg_add(arow[kk + 1].alg_mul(brow[kk + 1]));
                s2 = s2.alg_add(arow[kk + 2].alg_mul(brow[kk + 2]));
                s3 = s3.alg_add(arow[kk + 3].alg_mul(brow[kk + 3]));
                kk += 4;
            }
            let mut s = s0.alg_add(s1).alg_add(s2.alg_add(s3));
            while kk < k {
                s = s.alg_add(arow[kk].alg_mul(brow[kk]));
                kk += 1;
            }
            c[i * ldc + j] = c[i * ldc + j].alg_add(s);
        }
    }
}

fn generic_tn<S: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    a: &[S],
    lda: usize,
    b: &[S],
    ldb: usize,
    c: &mut [S],
    ldc: usize,
) {
    for k0 in (0..k).step_by(KC) {
        let k1 = (k0 + KC).min(k);
        for kk in k0..k1 {
            let brow = &b[kk * ldb..kk * ldb + n];
            for i in 0..m {
                let aki = a[kk * lda + i];
                let crow = &mut c[i * ldc..i * ldc + n];
                for (cj, &bj) in crow.iter_mut().zip(brow.iter()) {
                    *cj = cj.alg_add(aki.alg_mul(bj));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEON microkernels
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use super::KC;
    use core::arch::aarch64::*;

    /// Emits the three kernels for one float width.
    ///
    /// `nn` and `tn` share a body — both walk a scalar from `A` against a
    /// contiguous row of `B` — and differ only in how `A` is indexed, so they
    /// are one function specialized on a const `AT` flag. `nt` contracts along
    /// contiguous rows of both operands and is a separate dot-product shape.
    macro_rules! kernels {
        (
            $ty:ty, $lanes:expr,
            $dup:ident, $ld:ident, $st:ident, $fma:ident, $add:ident, $sum:ident,
            $tile:ident, $drive:ident, $nt_tile:ident, $nn:ident, $tn:ident, $nt:ident
        ) => {
            /// Accumulate `MR` rows of `C` across one k-panel.
            ///
            /// The accumulators stay in vector registers for the whole panel,
            /// so `C` is touched once at the end rather than reloaded and
            /// restored on every step of `k`. `NR` vectors wide × `MR` rows
            /// deep is chosen to fit NEON's 32 registers with room for the
            /// operand loads.
            ///
            /// # Safety
            /// Pointers must be valid for the extents implied by the leading
            /// dimensions, and `i0 + MR <= m`.
            #[inline(always)]
            unsafe fn $tile<const MR: usize, const AT: bool>(
                i0: usize,
                n: usize,
                k0: usize,
                k1: usize,
                ap: *const $ty,
                lda: usize,
                bp: *const $ty,
                ldb: usize,
                cp: *mut $ty,
                ldc: usize,
            ) {
                const NR: usize = 4;
                let vw: usize = $lanes;
                let step = NR * vw;

                // Row `r` of the tile, depth `d`. `AT` is a const, so the
                // branch folds away.
                let aat = |r: usize, d: usize| -> $ty {
                    if AT {
                        *ap.add(d * lda + i0 + r)
                    } else {
                        *ap.add((i0 + r) * lda + d)
                    }
                };

                let mut j = 0;
                // Wide path: MR × NR vector accumulators.
                while j + step <= n {
                    let mut acc = [[$dup(0.0 as $ty); NR]; MR];
                    for kk in k0..k1 {
                        let brow = bp.add(kk * ldb + j);
                        let mut bv = [$dup(0.0 as $ty); NR];
                        for (q, slot) in bv.iter_mut().enumerate() {
                            *slot = $ld(brow.add(q * vw));
                        }
                        for r in 0..MR {
                            let av = $dup(aat(r, kk));
                            for q in 0..NR {
                                acc[r][q] = $fma(acc[r][q], av, bv[q]);
                            }
                        }
                    }
                    for r in 0..MR {
                        let crow = cp.add((i0 + r) * ldc + j);
                        for q in 0..NR {
                            let p = crow.add(q * vw);
                            $st(p, $add($ld(p), acc[r][q]));
                        }
                    }
                    j += step;
                }
                // One vector at a time.
                while j + vw <= n {
                    let mut acc = [$dup(0.0 as $ty); MR];
                    for kk in k0..k1 {
                        let bv = $ld(bp.add(kk * ldb + j));
                        for r in 0..MR {
                            acc[r] = $fma(acc[r], $dup(aat(r, kk)), bv);
                        }
                    }
                    for r in 0..MR {
                        let p = cp.add((i0 + r) * ldc + j);
                        $st(p, $add($ld(p), acc[r]));
                    }
                    j += vw;
                }
                // Scalar tail.
                while j < n {
                    let mut acc = [0.0 as $ty; MR];
                    for kk in k0..k1 {
                        let bs = *bp.add(kk * ldb + j);
                        for r in 0..MR {
                            acc[r] += aat(r, kk) * bs;
                        }
                    }
                    for r in 0..MR {
                        *cp.add((i0 + r) * ldc + j) += acc[r];
                    }
                    j += 1;
                }
            }

            /// # Safety
            /// Slices must be large enough for the extents and leading dimensions.
            #[inline(always)]
            unsafe fn $drive<const AT: bool>(
                m: usize,
                n: usize,
                k: usize,
                a: &[$ty],
                lda: usize,
                b: &[$ty],
                ldb: usize,
                c: &mut [$ty],
                ldc: usize,
            ) {
                let ap = a.as_ptr();
                let bp = b.as_ptr();
                let cp = c.as_mut_ptr();
                for k0 in (0..k).step_by(KC) {
                    let k1 = (k0 + KC).min(k);
                    let mut i = 0;
                    while i + 4 <= m {
                        $tile::<4, AT>(i, n, k0, k1, ap, lda, bp, ldb, cp, ldc);
                        i += 4;
                    }
                    while i < m {
                        $tile::<1, AT>(i, n, k0, k1, ap, lda, bp, ldb, cp, ldc);
                        i += 1;
                    }
                }
            }

            /// # Safety
            /// Slices must be large enough for the extents and leading dimensions.
            pub unsafe fn $nn(
                m: usize,
                n: usize,
                k: usize,
                a: &[$ty],
                lda: usize,
                b: &[$ty],
                ldb: usize,
                c: &mut [$ty],
                ldc: usize,
            ) {
                $drive::<false>(m, n, k, a, lda, b, ldb, c, ldc)
            }

            /// # Safety
            /// Slices must be large enough for the extents and leading dimensions.
            pub unsafe fn $tn(
                m: usize,
                n: usize,
                k: usize,
                a: &[$ty],
                lda: usize,
                b: &[$ty],
                ldb: usize,
                c: &mut [$ty],
                ldc: usize,
            ) {
                $drive::<true>(m, n, k, a, lda, b, ldb, c, ldc)
            }

            /// Accumulate an `MR`×`NR` tile of `C` for the `nt` shape.
            ///
            /// Both operands are contiguous along `k`, so this is a block of
            /// dot products. Tiling both ways means one pass over `k` loads
            /// `MR + NR` vectors and issues `MR * NR` FMAs — at 4×4 that is
            /// twice the arithmetic per load of a row-at-a-time reduction,
            /// which is what the A row being re-read for every output column
            /// was costing.
            ///
            /// # Safety
            /// Pointers must be valid for the extents implied by the leading
            /// dimensions, with `i0 + MR <= m` and `j0 + NR <= n`.
            #[inline(always)]
            unsafe fn $nt_tile<const MR: usize, const NR: usize>(
                i0: usize,
                j0: usize,
                k: usize,
                ap: *const $ty,
                lda: usize,
                bp: *const $ty,
                ldb: usize,
                cp: *mut $ty,
                ldc: usize,
            ) {
                let vw: usize = $lanes;
                let mut acc = [[$dup(0.0 as $ty); NR]; MR];
                let mut kk = 0;
                while kk + vw <= k {
                    let mut bv = [$dup(0.0 as $ty); NR];
                    for (q, slot) in bv.iter_mut().enumerate() {
                        *slot = $ld(bp.add((j0 + q) * ldb + kk));
                    }
                    for r in 0..MR {
                        let av = $ld(ap.add((i0 + r) * lda + kk));
                        for q in 0..NR {
                            acc[r][q] = $fma(acc[r][q], av, bv[q]);
                        }
                    }
                    kk += vw;
                }
                let mut s = [[0.0 as $ty; NR]; MR];
                for r in 0..MR {
                    for q in 0..NR {
                        s[r][q] = $sum(acc[r][q]);
                    }
                }
                // Depth tail, below one vector.
                let mut d = kk;
                while d < k {
                    for r in 0..MR {
                        let av = *ap.add((i0 + r) * lda + d);
                        for q in 0..NR {
                            s[r][q] += av * *bp.add((j0 + q) * ldb + d);
                        }
                    }
                    d += 1;
                }
                for r in 0..MR {
                    for q in 0..NR {
                        *cp.add((i0 + r) * ldc + j0 + q) += s[r][q];
                    }
                }
            }

            /// # Safety
            /// Slices must be large enough for the extents and leading dimensions.
            pub unsafe fn $nt(
                m: usize,
                n: usize,
                k: usize,
                a: &[$ty],
                lda: usize,
                b: &[$ty],
                ldb: usize,
                c: &mut [$ty],
                ldc: usize,
            ) {
                let ap = a.as_ptr();
                let bp = b.as_ptr();
                let cp = c.as_mut_ptr();

                let mut i = 0;
                while i + 4 <= m {
                    let mut j = 0;
                    while j + 4 <= n {
                        $nt_tile::<4, 4>(i, j, k, ap, lda, bp, ldb, cp, ldc);
                        j += 4;
                    }
                    while j < n {
                        $nt_tile::<4, 1>(i, j, k, ap, lda, bp, ldb, cp, ldc);
                        j += 1;
                    }
                    i += 4;
                }
                while i < m {
                    let mut j = 0;
                    while j + 4 <= n {
                        $nt_tile::<1, 4>(i, j, k, ap, lda, bp, ldb, cp, ldc);
                        j += 4;
                    }
                    while j < n {
                        $nt_tile::<1, 1>(i, j, k, ap, lda, bp, ldb, cp, ldc);
                        j += 1;
                    }
                    i += 1;
                }
            }
        };
    }

    kernels!(
        f32,
        4,
        vdupq_n_f32,
        vld1q_f32,
        vst1q_f32,
        vfmaq_f32,
        vaddq_f32,
        vaddvq_f32,
        tile_f32,
        drive_f32,
        nt_tile_f32,
        nn_f32,
        tn_f32,
        nt_f32
    );
    kernels!(
        f64,
        2,
        vdupq_n_f64,
        vld1q_f64,
        vst1q_f64,
        vfmaq_f64,
        vaddq_f64,
        vaddvq_f64,
        tile_f64,
        drive_f64,
        nt_tile_f64,
        nn_f64,
        tn_f64,
        nt_f64
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Textbook reference, deliberately the slowest possible spelling.
    fn reference(m: usize, n: usize, k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn seq(n: usize, salt: f64) -> Vec<f64> {
        (0..n).map(|i| ((i as f64) * 0.37 + salt).sin()).collect()
    }

    fn transposed(rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
        let mut t = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = x[r * cols + c];
            }
        }
        t
    }

    /// Sizes that straddle every unroll boundary: the 4-row block, the SIMD
    /// lane count, and the KC panel.
    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (1, 5, 3),
        (3, 7, 5),
        (4, 4, 4),
        (5, 9, 11),
        (16, 576, 576),
        (7, 13, 300),
    ];

    #[test]
    fn nn_matches_reference() {
        for &(m, n, k) in SHAPES {
            let a = seq(m * k, 0.1);
            let b = seq(k * n, 0.7);
            let want = reference(m, n, k, &a, &b);
            let mut got = vec![0.0; m * n];
            gemm_nn(m, n, k, &a, k, &b, n, &mut got, n);
            for (g, w) in got.iter().zip(want.iter()) {
                assert!((g - w).abs() < 1e-9, "nn {m}x{n}x{k}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn nt_matches_reference() {
        for &(m, n, k) in SHAPES {
            let a = seq(m * k, 0.1);
            let b = seq(k * n, 0.7);
            let want = reference(m, n, k, &a, &b);
            // gemm_nt wants B as [n,k].
            let bt = transposed(k, n, &b);
            let mut got = vec![0.0; m * n];
            gemm_nt(m, n, k, &a, k, &bt, k, &mut got, n);
            for (g, w) in got.iter().zip(want.iter()) {
                assert!((g - w).abs() < 1e-9, "nt {m}x{n}x{k}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn tn_matches_reference() {
        for &(m, n, k) in SHAPES {
            let a = seq(m * k, 0.1);
            let b = seq(k * n, 0.7);
            let want = reference(m, n, k, &a, &b);
            // gemm_tn wants A as [k,m].
            let at = transposed(m, k, &a);
            let mut got = vec![0.0; m * n];
            gemm_tn(m, n, k, &at, m, &b, n, &mut got, n);
            for (g, w) in got.iter().zip(want.iter()) {
                assert!((g - w).abs() < 1e-9, "tn {m}x{n}x{k}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn f32_matches_generic_path() {
        // The f32 SIMD kernels and the generic path must agree; Dual would
        // take the generic path, so a divergence here is a real bug.
        let (m, n, k) = (9, 17, 33);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let mut simd = vec![0.0f32; m * n];
        gemm_nn(m, n, k, &a, k, &b, n, &mut simd, n);
        let mut plain = vec![0.0f32; m * n];
        generic_nn(m, n, k, &a, k, &b, n, &mut plain, n);
        for (s, p) in simd.iter().zip(plain.iter()) {
            assert!((s - p).abs() <= p.abs() * 1e-5, "{s} vs {p}");
        }
    }

    #[test]
    fn accumulates_into_c() {
        let (m, n, k) = (3, 3, 3);
        let a = seq(m * k, 0.1);
        let b = seq(k * n, 0.7);
        let want = reference(m, n, k, &a, &b);
        let mut got = vec![1.0; m * n];
        gemm_nn(m, n, k, &a, k, &b, n, &mut got, n);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - (w + 1.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn leading_dimension_larger_than_extent() {
        // Sub-matrix views: lda/ldb/ldc exceed the logical extents.
        let (m, n, k) = (2, 2, 2);
        let (lda, ldb, ldc) = (5, 6, 7);
        let mut a = vec![0.0; m * lda];
        let mut b = vec![0.0; k * ldb];
        for i in 0..m {
            for j in 0..k {
                a[i * lda + j] = (i + 2 * j) as f64;
            }
        }
        for i in 0..k {
            for j in 0..n {
                b[i * ldb + j] = (3 * i + j) as f64;
            }
        }
        let mut c = vec![0.0; m * ldc];
        gemm_nn(m, n, k, &a, lda, &b, ldb, &mut c, ldc);
        for i in 0..m {
            for j in 0..n {
                let mut want = 0.0;
                for kk in 0..k {
                    want += a[i * lda + kk] * b[kk * ldb + j];
                }
                assert!((c[i * ldc + j] - want).abs() < 1e-9);
            }
        }
    }
}
