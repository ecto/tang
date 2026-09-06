//! Fast 2-D convolution kernels: im2col + GEMM.
//!
//! [`crate::Conv2d`]'s original forward was `Tensor::from_fn` over a
//! multi-dimensional `get`, which recomputes strides for every tap. That is
//! fine for a gradient check and hopeless for a training run: a 3×3 32→32
//! convolution on a 64×64 tile is ~38 million multiply-accumulates, and
//! `from_fn` pays several index multiplies and a bounds check per one of them.
//!
//! Everything here instead flattens the problem to matrix products so the
//! blocked, NEON-accelerated kernels in [`tang_la::gemm`] do the arithmetic.
//! For one batch element the patch matrix is gathered channel-major,
//!
//! ```text
//! cm[(c·kh + ki)·kw + kj][oh·out_w + ow] = input[c][oh·s - p + ki·d][ow·s - p + kj·d]
//! ```
//!
//! `K = c_in·kh·kw` rows by `out_h·out_w` columns, and then, writing `Wᵀ` for
//! the weights as `[K, c_out]` and `Yᵀ` for the output as `[out_h·out_w,
//! c_out]`:
//!
//! | quantity | product | kernel |
//! |---|---|---|
//! | output | `Yᵀ = cmᵀ · Wᵀ` (bias pre-filled) | [`gemm_tn`] |
//! | weight gradient | `∂Wᵀ += cm · ∂Yᵀ` | [`gemm_nt`] |
//! | patch gradient | `∂cm = Wᵀ · ∂Yᵀᵀ` | [`gemm_nt`] |
//!
//! and `∂cm` is scattered back to image space by [`scatter`], the exact
//! transpose of [`gather`].
//!
//! # Why channel-major, and why the output comes out transposed
//!
//! Both choices are about which operand each GEMM streams and how many rows
//! the row-blocked kernels have to work with.
//!
//! Gathering channel-major puts one (channel, tap) pair on each row, so with
//! unit stride and dilation every output row is a `copy_from_slice` of an
//! input row rather than a per-element gather — worth ~6× on the gather. In
//! reverse the gap is far wider: scattering pixel-major means `K` dependent
//! read-modify-writes per output pixel, striding across every input plane,
//! where channel-major scatter is contiguous slice addition. On the layers
//! measured that is a 40× difference, and it was the single largest cost in
//! the backward pass before the orientation was fixed.
//!
//! Building the output transposed is the same argument applied to the GEMM.
//! The obvious arrangement, `Y[c_out, ohw] = W · cm`, makes `c_out` the row
//! count of every product: with the 32-channel layers that motivated this work
//! that is `m = 32` against `n = 4096`, so each pass over the four-row register
//! tiles streams the whole multi-megabyte patch matrix out of cache, and there
//! are only eight row blocks for the thread pool to divide. Putting the pixels
//! on the rows instead makes `m` four thousand and the streamed operand the
//! *weight* matrix — tens of kilobytes, resident in L1 for the whole product.
//! The price is one `[c_out, ohw]` transpose per batch element per direction,
//! two orders of magnitude smaller than the product it enables.
//!
//! The patch matrix is built per batch element into one reused buffer rather
//! than for the whole batch at once: the batch-wide matrix runs to `batch · K ·
//! out_h · out_w` scalars (75 MB for the second layer of a 32-channel denoiser
//! at batch 16), and re-deriving it in the backward pass is far cheaper than
//! carrying it between the two.
//!
//! Threading comes from `tang-la`'s `threads` feature, which splits each GEMM
//! over its output rows across the rayon pool — rows that the transposed
//! arrangement above makes numerous. The batch loop here stays serial, so the
//! weight-gradient accumulation needs no cross-thread reduction.

use alloc::vec::Vec;
use tang::Scalar;
use tang_la::{gemm_nt, gemm_tn};
use tang_tensor::{Shape, Tensor};

/// The shape of one 2-D convolution, resolved once per call.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvSpec {
    pub batch: usize,
    pub c_in: usize,
    pub h: usize,
    pub w: usize,
    pub c_out: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out_h: usize,
    pub out_w: usize,
}

impl ConvSpec {
    /// Columns of the patch matrix: one per (input channel, tap) pair.
    #[inline]
    pub(crate) fn k(&self) -> usize {
        self.c_in * self.kh * self.kw
    }

    /// Rows of the patch matrix: one per output pixel.
    #[inline]
    pub(crate) fn ohw(&self) -> usize {
        self.out_h * self.out_w
    }
}

/// Gather one batch element's receptive fields, channel-major: row
/// `(c·kh + ki)·kw + kj` of `cm` holds that tap's value at every output pixel.
///
/// Taps in the zero padding are written as `S::ZERO`, so `cm` need not be
/// cleared. With unit stride and dilation the innermost loop is a
/// `copy_from_slice` of the interior plus two zero fills for the padded ends,
/// which is why the patch matrix is built in this orientation and transposed
/// afterwards rather than gathered straight into pixel-major order — that is
/// worth about 6x on the gather and, in reverse, 40x on the scatter.
fn gather<S: Scalar>(input: &[S], spec: &ConvSpec, cm: &mut [S]) {
    let (h, w, out_h, out_w) = (spec.h, spec.w, spec.out_h, spec.out_w);
    let (pad, stride, dil) = (spec.padding, spec.stride, spec.dilation);
    let ohw = out_h * out_w;
    debug_assert_eq!(cm.len(), spec.k() * ohw);

    for c in 0..spec.c_in {
        let plane = &input[c * h * w..(c + 1) * h * w];
        for ki in 0..spec.kh {
            for kj in 0..spec.kw {
                let row = &mut cm[((c * spec.kh + ki) * spec.kw + kj) * ohw..][..ohw];
                let tap_j = kj * dil;
                for oh in 0..out_h {
                    let dst = &mut row[oh * out_w..(oh + 1) * out_w];
                    let ih = oh * stride + ki * dil;
                    // Unsigned throughout: `ih < pad` is the `ih - pad < 0` test.
                    if ih < pad || ih - pad >= h {
                        dst.fill(S::ZERO);
                        continue;
                    }
                    let src = &plane[(ih - pad) * w..][..w];
                    if stride == 1 && dil == 1 {
                        // `ow + kj - pad` runs over `[lo, hi)` inside the row;
                        // everything outside is padding.
                        let lo = pad.saturating_sub(kj);
                        let hi = (w + pad - kj).min(out_w);
                        dst[..lo].fill(S::ZERO);
                        dst[hi..].fill(S::ZERO);
                        dst[lo..hi].copy_from_slice(&src[lo + kj - pad..hi + kj - pad]);
                        continue;
                    }
                    for (ow, d) in dst.iter_mut().enumerate() {
                        let iw = ow * stride + tap_j;
                        *d = if iw < pad || iw - pad >= w {
                            S::ZERO
                        } else {
                            src[iw - pad]
                        };
                    }
                }
            }
        }
    }
}

/// Accumulate a channel-major patch-matrix gradient back into image space.
/// The exact transpose of [`gather`]; `grad_input` must be zeroed by the caller.
fn scatter<S: Scalar>(cm: &[S], spec: &ConvSpec, grad_input: &mut [S]) {
    let (h, w, out_h, out_w) = (spec.h, spec.w, spec.out_h, spec.out_w);
    let (pad, stride, dil) = (spec.padding, spec.stride, spec.dilation);
    let ohw = out_h * out_w;

    for c in 0..spec.c_in {
        let plane = &mut grad_input[c * h * w..(c + 1) * h * w];
        for ki in 0..spec.kh {
            for kj in 0..spec.kw {
                let row = &cm[((c * spec.kh + ki) * spec.kw + kj) * ohw..][..ohw];
                let tap_j = kj * dil;
                for oh in 0..out_h {
                    let src = &row[oh * out_w..(oh + 1) * out_w];
                    let ih = oh * stride + ki * dil;
                    if ih < pad || ih - pad >= h {
                        continue;
                    }
                    let dst = &mut plane[(ih - pad) * w..][..w];
                    if stride == 1 && dil == 1 {
                        let lo = pad.saturating_sub(kj);
                        let hi = (w + pad - kj).min(out_w);
                        for (d, s) in dst[lo + kj - pad..hi + kj - pad]
                            .iter_mut()
                            .zip(&src[lo..hi])
                        {
                            *d += *s;
                        }
                        continue;
                    }
                    for (ow, s) in src.iter().enumerate() {
                        let iw = ow * stride + tap_j;
                        if iw >= pad && iw - pad < w {
                            dst[iw - pad] += *s;
                        }
                    }
                }
            }
        }
    }
}

/// Cache-blocked transpose of a `[rows, cols]` matrix into `[cols, rows]`.
const TBLK: usize = 32;

fn transpose_into<S: Scalar>(src: &[S], rows: usize, cols: usize, dst: &mut [S]) {
    for r0 in (0..rows).step_by(TBLK) {
        let r1 = (r0 + TBLK).min(rows);
        for c0 in (0..cols).step_by(TBLK) {
            let c1 = (c0 + TBLK).min(cols);
            for r in r0..r1 {
                let s = &src[r * cols..][..cols];
                for c in c0..c1 {
                    dst[c * rows + r] = s[c];
                }
            }
        }
    }
}

/// `output[b, oc, oh, ow] = bias[oc] + Σ weight[oc, c, ki, kj] · input[…]`.
///
/// `input` must be contiguous `[batch, c_in, h, w]`, `weight` contiguous
/// `[c_out, c_in, kh, kw]`, `bias` contiguous `[c_out]`.
pub(crate) fn forward<S: Scalar>(
    input: &Tensor<S>,
    weight: &Tensor<S>,
    bias: &Tensor<S>,
    spec: &ConvSpec,
) -> Tensor<S> {
    let (k, ohw) = (spec.k(), spec.ohw());
    let in_stride = spec.c_in * spec.h * spec.w;
    let out_stride = spec.c_out * ohw;

    let a = input.data();
    let b = bias.data();
    // Wᵀ, `[K, c_out]` — tens of kilobytes, transposed once per call so the
    // GEMM streams it as the inner operand.
    let mut wt = alloc::vec![S::ZERO; k * spec.c_out];
    transpose_into(weight.data(), spec.c_out, k, &mut wt);

    let mut out = alloc::vec![S::ZERO; spec.batch * out_stride];
    let mut cm = alloc::vec![S::ZERO; k * ohw];
    // Yᵀ, `[ohw, c_out]`, pre-filled with the bias so the GEMM accumulates
    // straight onto it.
    let mut yt = alloc::vec![S::ZERO; ohw * spec.c_out];

    for n in 0..spec.batch {
        gather(&a[n * in_stride..(n + 1) * in_stride], spec, &mut cm);
        for row in yt.chunks_exact_mut(spec.c_out) {
            row.copy_from_slice(b);
        }
        // Yᵀ[ohw, c_out] += cm[K, ohw]ᵀ · Wᵀ[K, c_out]
        gemm_tn(
            ohw, spec.c_out, k, &cm, ohw, &wt, spec.c_out, &mut yt, spec.c_out,
        );
        transpose_into(
            &yt,
            ohw,
            spec.c_out,
            &mut out[n * out_stride..(n + 1) * out_stride],
        );
    }

    Tensor::new(
        out,
        Shape::from_slice(&[spec.batch, spec.c_out, spec.out_h, spec.out_w]),
    )
}

/// Gradients of [`forward`]: `(grad_input, grad_weight, grad_bias)`.
///
/// All inputs must be contiguous.
pub(crate) fn backward<S: Scalar>(
    input: &Tensor<S>,
    weight: &Tensor<S>,
    grad_output: &Tensor<S>,
    spec: &ConvSpec,
) -> (Tensor<S>, Tensor<S>, Tensor<S>) {
    let (k, ohw) = (spec.k(), spec.ohw());
    let in_stride = spec.c_in * spec.h * spec.w;
    let out_stride = spec.c_out * ohw;

    let a = input.data();
    let go = grad_output.data();
    let mut wt = alloc::vec![S::ZERO; k * spec.c_out];
    transpose_into(weight.data(), spec.c_out, k, &mut wt);

    let mut grad_in = alloc::vec![S::ZERO; spec.batch * in_stride];
    // Accumulated as ∂Wᵀ `[K, c_out]` to match the patch matrix's orientation,
    // and transposed back to `[c_out, K]` once at the end.
    let mut grad_wt = alloc::vec![S::ZERO; k * spec.c_out];
    let mut grad_b = alloc::vec![S::ZERO; spec.c_out];

    let mut cm: Vec<S> = alloc::vec![S::ZERO; k * ohw];
    let mut grad_cm: Vec<S> = alloc::vec![S::ZERO; k * ohw];
    let mut gyt: Vec<S> = alloc::vec![S::ZERO; ohw * spec.c_out];

    for n in 0..spec.batch {
        let gy = &go[n * out_stride..(n + 1) * out_stride];

        // ∂bias[oc] = Σ_{b, oh, ow} ∂Y[b, oc, oh, ow], summed on the
        // untransposed gradient where each channel is one contiguous run.
        for (oc, gb) in grad_b.iter_mut().enumerate() {
            let mut acc = S::ZERO;
            for v in &gy[oc * ohw..(oc + 1) * ohw] {
                acc += *v;
            }
            *gb += acc;
        }

        gather(&a[n * in_stride..(n + 1) * in_stride], spec, &mut cm);

        // ∂Wᵀ[K, c_out] += cm[K, ohw] · ∂Y[c_out, ohw]ᵀ
        gemm_nt(
            k,
            spec.c_out,
            ohw,
            &cm,
            ohw,
            gy,
            ohw,
            &mut grad_wt,
            spec.c_out,
        );

        // ∂cm[K, ohw] = Wᵀ[K, c_out] · ∂Yᵀ[ohw, c_out]ᵀ
        transpose_into(gy, spec.c_out, ohw, &mut gyt);
        grad_cm.fill(S::ZERO);
        gemm_nt(
            k,
            ohw,
            spec.c_out,
            &wt,
            spec.c_out,
            &gyt,
            spec.c_out,
            &mut grad_cm,
            ohw,
        );
        scatter(
            &grad_cm,
            spec,
            &mut grad_in[n * in_stride..(n + 1) * in_stride],
        );
    }

    let mut grad_w = alloc::vec![S::ZERO; spec.c_out * k];
    transpose_into(&grad_wt, k, spec.c_out, &mut grad_w);

    (
        Tensor::new(grad_in, input.shape().clone()),
        Tensor::new(grad_w, weight.shape().clone()),
        Tensor::new(grad_b, Shape::from_slice(&[spec.c_out])),
    )
}
