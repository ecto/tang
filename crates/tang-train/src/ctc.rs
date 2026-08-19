//! Connectionist Temporal Classification — loss, gradient, and decoding.
//!
//! CTC trains a network to map a long input to a shorter label sequence when
//! nobody has told you which input step produced which label. A nanopore
//! basecaller sees thousands of current samples and must emit a few hundred
//! bases; a speech model sees spectrogram frames and must emit characters.
//! Aligning those by hand is exactly the labour CTC removes.
//!
//! The trick is a `blank` symbol and a sum over every alignment that collapses
//! to the target. Collapsing merges runs of the same label, then drops blanks,
//! so `a-abb-` becomes `aab`. A blank between two identical labels is what
//! keeps `aa` from collapsing to `a`.
//!
//! Everything here works in log space. Probabilities of long alignments
//! underflow `f32` within a few dozen time steps, and a basecaller runs for
//! thousands.

use alloc::{vec, vec::Vec};
use tang::Scalar;
use tang_tensor::{Shape, Tensor};

/// What went wrong setting up a CTC computation.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CtcError {
    /// The target is longer than the input can possibly emit.
    ///
    /// Emitting `u` labels needs at least `u` steps, plus one blank between
    /// each adjacent pair of identical labels. Below that the alignment set is
    /// empty and the loss is infinite, which is a bug in the data pipeline
    /// rather than a number worth backpropagating.
    TargetTooLong {
        input_len: usize,
        target_len: usize,
        required: usize,
    },
    /// A target label was outside the class range.
    LabelOutOfRange { label: usize, classes: usize },
    /// A target contained the blank symbol, which is never a real label.
    BlankInTarget { position: usize },
    /// The blank index was outside the class range.
    BlankOutOfRange { blank: usize, classes: usize },
}

impl core::fmt::Display for CtcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CtcError::TargetTooLong { input_len, target_len, required } => write!(
                f,
                "target of {target_len} labels needs at least {required} input steps, got {input_len}"
            ),
            CtcError::LabelOutOfRange { label, classes } => {
                write!(f, "label {label} is outside 0..{classes}")
            }
            CtcError::BlankInTarget { position } => {
                write!(f, "target contains the blank symbol at position {position}")
            }
            CtcError::BlankOutOfRange { blank, classes } => {
                write!(f, "blank index {blank} is outside 0..{classes}")
            }
        }
    }
}

impl core::error::Error for CtcError {}

/// `log(exp(a) + exp(b))` without overflowing or underflowing.
///
/// Both arguments are log-probabilities, so both are at most 0 and either may
/// be `-inf` for an impossible path. Factoring out the larger term keeps the
/// surviving `exp` in a safe range.
#[inline]
fn log_add<S: Scalar>(a: S, b: S) -> S {
    if a == S::NEG_INFINITY {
        return b;
    }
    if b == S::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p_compat()
}

/// `ln(1 + x)` via the [`Scalar`] surface, without losing the small-`x` bits.
///
/// [`Scalar`] has no `ln_1p`, and the direct `(1 + x).ln()` is exactly the form
/// that fails for small `x`: forming `1 + x` rounds away the low bits of `x`,
/// and for `x` below the epsilon it rounds to `1` and returns zero instead of
/// `x`. In `log_add` this is the common case, not a corner — `x` is
/// `exp(lo - hi)`, which is tiny whenever one alignment dominates another,
/// which is most of the time once a model has learned anything.
///
/// This is Kahan's correction, and it needs nothing beyond the arithmetic
/// [`Scalar`] already has. Let `u` be the rounded `1 + x`. Then `u - 1` is the
/// perturbation that was *actually* representable, so `ln(u) / (u - 1)` is the
/// slope over the interval the hardware really used; scaling it by the true `x`
/// recovers the bits that forming `u` discarded. When `u` rounds to exactly
/// `1`, `ln(1 + x)` equals `x` to within representable precision, so return `x`.
trait Ln1p: Scalar {
    #[inline]
    fn ln_1p_compat(self) -> Self {
        let u = Self::ONE + self;
        if u == Self::ONE {
            self
        } else {
            self * u.ln() / (u - Self::ONE)
        }
    }
}
impl<S: Scalar> Ln1p for S {}

/// Interleave blanks into a target: `abc` becomes `-a-b-c-`.
///
/// The extended sequence is what the forward-backward recursion actually walks.
/// Its length is always `2 * target.len() + 1`.
fn extend(target: &[usize], blank: usize) -> Vec<usize> {
    let mut z = Vec::with_capacity(2 * target.len() + 1);
    z.push(blank);
    for &label in target {
        z.push(label);
        z.push(blank);
    }
    z
}

/// The minimum number of input steps that can emit `target`.
///
/// One step per label, plus one blank wherever two identical labels are
/// adjacent, since `aa` cannot collapse out of a single run.
pub fn min_input_len(target: &[usize]) -> usize {
    if target.is_empty() {
        return 0;
    }
    let repeats = target.windows(2).filter(|w| w[0] == w[1]).count();
    target.len() + repeats
}

fn validate(
    target: &[usize],
    input_len: usize,
    classes: usize,
    blank: usize,
) -> Result<(), CtcError> {
    if blank >= classes {
        return Err(CtcError::BlankOutOfRange { blank, classes });
    }
    for (i, &label) in target.iter().enumerate() {
        if label >= classes {
            return Err(CtcError::LabelOutOfRange { label, classes });
        }
        if label == blank {
            return Err(CtcError::BlankInTarget { position: i });
        }
    }
    let required = min_input_len(target);
    if required > input_len {
        return Err(CtcError::TargetTooLong {
            input_len,
            target_len: target.len(),
            required,
        });
    }
    Ok(())
}

/// Row-wise log-softmax of a `[time, classes]` tensor.
///
/// Subtracting the row max before exponentiating is what keeps large logits
/// from overflowing.
fn log_softmax<S: Scalar>(logits: &Tensor<S>) -> Vec<Vec<S>> {
    let time = logits.shape()[0];
    let classes = logits.shape()[1];
    let mut out = Vec::with_capacity(time);
    for t in 0..time {
        let mut row: Vec<S> = (0..classes).map(|c| logits.get(&[t, c])).collect();
        let mut max = row[0];
        for &v in &row[1..] {
            max = max.max(v);
        }
        let mut sum = S::ZERO;
        for v in &row {
            sum += (*v - max).exp();
        }
        let log_denom = max + sum.ln();
        for v in &mut row {
            *v -= log_denom;
        }
        out.push(row);
    }
    out
}

/// Forward variables: `alpha[t][s]` is the log-probability of every alignment
/// prefix that reaches extended position `s` at time `t`.
fn forward<S: Scalar>(lp: &[Vec<S>], z: &[usize]) -> Vec<Vec<S>> {
    let time = lp.len();
    let states = z.len();
    let mut alpha = vec![vec![S::NEG_INFINITY; states]; time];

    // An alignment may open on the leading blank or on the first real label.
    alpha[0][0] = lp[0][z[0]];
    if states > 1 {
        alpha[0][1] = lp[0][z[1]];
    }

    for t in 1..time {
        for s in 0..states {
            let mut acc = alpha[t - 1][s];
            if s >= 1 {
                acc = log_add(acc, alpha[t - 1][s - 1]);
            }
            // The skip-a-blank transition is only legal when it would not merge
            // two identical labels into one run.
            if s >= 2 && z[s] != z[s - 2] {
                acc = log_add(acc, alpha[t - 1][s - 2]);
            }
            alpha[t][s] = acc + lp[t][z[s]];
        }
    }
    alpha
}

/// Backward variables: `beta[t][s]` is the log-probability of every alignment
/// suffix that completes the target from position `s` at time `t`.
fn backward<S: Scalar>(lp: &[Vec<S>], z: &[usize]) -> Vec<Vec<S>> {
    let time = lp.len();
    let states = z.len();
    let mut beta = vec![vec![S::NEG_INFINITY; states]; time];

    // An alignment may close on the trailing blank or on the last real label.
    beta[time - 1][states - 1] = S::ZERO;
    if states > 1 {
        beta[time - 1][states - 2] = S::ZERO;
    }

    for t in (0..time - 1).rev() {
        for s in 0..states {
            let mut acc = beta[t + 1][s] + lp[t + 1][z[s]];
            if s + 1 < states {
                acc = log_add(acc, beta[t + 1][s + 1] + lp[t + 1][z[s + 1]]);
            }
            if s + 2 < states && z[s] != z[s + 2] {
                acc = log_add(acc, beta[t + 1][s + 2] + lp[t + 1][z[s + 2]]);
            }
            beta[t][s] = acc;
        }
    }
    beta
}

/// Total log-probability of the target, summed over every alignment.
fn log_likelihood<S: Scalar>(alpha: &[Vec<S>]) -> S {
    let last = &alpha[alpha.len() - 1];
    let states = last.len();
    if states == 1 {
        return last[0];
    }
    log_add(last[states - 1], last[states - 2])
}

/// CTC loss for one sequence: the negative log-probability of the target.
///
/// `logits` is `[time, classes]` of unnormalised scores — the same convention
/// as [`cross_entropy_loss`](crate::cross_entropy_loss), so a model's final
/// linear layer can feed this directly.
///
/// # Errors
///
/// Returns [`CtcError`] when the target cannot be emitted at all, rather than
/// silently producing an infinite loss.
pub fn ctc_loss<S: Scalar>(
    logits: &Tensor<S>,
    target: &[usize],
    blank: usize,
) -> Result<S, CtcError> {
    assert_eq!(logits.ndim(), 2, "ctc_loss expects [time, classes]");
    let time = logits.shape()[0];
    let classes = logits.shape()[1];
    validate(target, time, classes, blank)?;

    let lp = log_softmax(logits);
    let z = extend(target, blank);
    let alpha = forward(&lp, &z);
    Ok(-log_likelihood(&alpha))
}

/// Gradient of [`ctc_loss`] with respect to `logits`.
///
/// The result has the same shape as `logits`. As with cross-entropy, the
/// gradient is `softmax - posterior`: the model is pushed away from what it
/// currently predicts and toward the alignment-weighted occupancy of each
/// label at each step.
pub fn ctc_loss_grad<S: Scalar>(
    logits: &Tensor<S>,
    target: &[usize],
    blank: usize,
) -> Result<Tensor<S>, CtcError> {
    assert_eq!(logits.ndim(), 2, "ctc_loss_grad expects [time, classes]");
    let time = logits.shape()[0];
    let classes = logits.shape()[1];
    validate(target, time, classes, blank)?;

    let lp = log_softmax(logits);
    let z = extend(target, blank);
    let alpha = forward(&lp, &z);
    let beta = backward(&lp, &z);
    let total = log_likelihood(&alpha);

    // posterior[t][k] = sum over extended positions emitting k of alpha*beta,
    // normalised by the total. Accumulated in log space, then exponentiated
    // once at the end.
    let mut grad = vec![vec![S::ZERO; classes]; time];
    for t in 0..time {
        let mut occupancy = vec![S::NEG_INFINITY; classes];
        for (s, &label) in z.iter().enumerate() {
            let ab = alpha[t][s] + beta[t][s];
            if ab != S::NEG_INFINITY {
                occupancy[label] = log_add(occupancy[label], ab);
            }
        }
        for k in 0..classes {
            let posterior = if occupancy[k] == S::NEG_INFINITY {
                S::ZERO
            } else {
                (occupancy[k] - total).exp()
            };
            grad[t][k] = lp[t][k].exp() - posterior;
        }
    }

    Ok(Tensor::from_fn(
        Shape::from_slice(&[time, classes]),
        |idx| grad[idx[0]][idx[1]],
    ))
}

/// Mean CTC loss over a batch of sequences.
///
/// `logits` is `[batch, time, classes]`. Every sequence is assumed to use the
/// full time extent; ragged batches should be split or padded by the caller,
/// since padding a batch with blanks changes the loss.
pub fn ctc_loss_batch<S: Scalar>(
    logits: &Tensor<S>,
    targets: &[&[usize]],
    blank: usize,
) -> Result<S, CtcError> {
    assert_eq!(
        logits.ndim(),
        3,
        "ctc_loss_batch expects [batch, time, classes]"
    );
    let batch = logits.shape()[0];
    assert_eq!(batch, targets.len(), "one target per batch element");

    let mut total = S::ZERO;
    for (b, target) in targets.iter().enumerate() {
        total += ctc_loss(&slice_batch(logits, b), target, blank)?;
    }
    Ok(total / S::from_f64(batch as f64))
}

/// Gradient of [`ctc_loss_batch`], shaped like `logits`.
pub fn ctc_loss_batch_grad<S: Scalar>(
    logits: &Tensor<S>,
    targets: &[&[usize]],
    blank: usize,
) -> Result<Tensor<S>, CtcError> {
    assert_eq!(
        logits.ndim(),
        3,
        "ctc_loss_batch_grad expects [batch, time, classes]"
    );
    let batch = logits.shape()[0];
    let time = logits.shape()[1];
    let classes = logits.shape()[2];
    assert_eq!(batch, targets.len(), "one target per batch element");

    let scale = S::ONE / S::from_f64(batch as f64);
    let mut grads = Vec::with_capacity(batch);
    for (b, target) in targets.iter().enumerate() {
        grads.push(ctc_loss_grad(&slice_batch(logits, b), target, blank)?);
    }

    Ok(Tensor::from_fn(
        Shape::from_slice(&[batch, time, classes]),
        |idx| grads[idx[0]].get(&[idx[1], idx[2]]) * scale,
    ))
}

/// Extract batch element `b` as a `[time, classes]` tensor.
fn slice_batch<S: Scalar>(logits: &Tensor<S>, b: usize) -> Tensor<S> {
    let time = logits.shape()[1];
    let classes = logits.shape()[2];
    Tensor::from_fn(Shape::from_slice(&[time, classes]), |idx| {
        logits.get(&[b, idx[0], idx[1]])
    })
}

/// Collapse an alignment to its label sequence: merge repeats, drop blanks.
///
/// This is the CTC collapsing rule made explicit, and the inverse of what the
/// loss sums over.
pub fn collapse(alignment: &[usize], blank: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut prev = None;
    for &symbol in alignment {
        if Some(symbol) != prev && symbol != blank {
            out.push(symbol);
        }
        prev = Some(symbol);
    }
    out
}

/// Greedy (best-path) decode: take the most likely class at each step, then
/// collapse.
///
/// Fast and usually good enough to monitor training. It is not the most likely
/// *label sequence*, because many alignments can collapse to the same labels
/// and greedy decoding only ever follows one of them — [`beam_decode`] sums
/// over them instead.
pub fn greedy_decode<S: Scalar>(logits: &Tensor<S>, blank: usize) -> Vec<usize> {
    assert_eq!(logits.ndim(), 2, "greedy_decode expects [time, classes]");
    let time = logits.shape()[0];
    let classes = logits.shape()[1];

    let mut path = Vec::with_capacity(time);
    for t in 0..time {
        let mut best = 0;
        let mut best_score = logits.get(&[t, 0]);
        for c in 1..classes {
            let score = logits.get(&[t, c]);
            if score > best_score {
                best_score = score;
                best = c;
            }
        }
        path.push(best);
    }
    collapse(&path, blank)
}

/// A beam during prefix search, tracking blank- and non-blank-ending
/// probabilities separately.
///
/// The split is what lets the search know whether extending by the same label
/// continues a run or starts a new one.
#[derive(Clone)]
struct Beam {
    prefix: Vec<usize>,
    /// Log-probability of alignments for this prefix that end in a blank.
    blank: f64,
    /// Log-probability of alignments for this prefix that end in a real label.
    non_blank: f64,
}

impl Beam {
    fn total(&self) -> f64 {
        log_add_f64(self.blank, self.non_blank)
    }
}

fn log_add_f64(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// Beam search decode, summing over the alignments that share a prefix.
///
/// `width` beams are kept at each step. Larger widths cost linearly more and
/// converge on the true most-likely label sequence; `width == 1` is still not
/// identical to [`greedy_decode`], because even one beam merges alignments.
///
/// Runs in `f64` regardless of the tensor's scalar type: the search compares
/// many nearly-equal log-probabilities, and doing that in `f32` reorders beams.
pub fn beam_decode<S: Scalar>(logits: &Tensor<S>, blank: usize, width: usize) -> Vec<usize> {
    assert_eq!(logits.ndim(), 2, "beam_decode expects [time, classes]");
    let time = logits.shape()[0];
    let classes = logits.shape()[1];
    let width = width.max(1);

    let lp = log_softmax(logits);
    let lp: Vec<Vec<f64>> = lp
        .iter()
        .map(|row| row.iter().map(|v| v.to_f64()).collect())
        .collect();

    // The empty prefix starts with certainty, having emitted a blank prefix of
    // length zero.
    let mut beams = vec![Beam {
        prefix: Vec::new(),
        blank: 0.0,
        non_blank: f64::NEG_INFINITY,
    }];

    for lp_t in lp.iter().take(time) {
        let mut next: Vec<Beam> = Vec::new();

        // Find or create the beam for a prefix. Linear scan is fine: `next`
        // holds at most width * classes entries and width is small.
        fn slot(next: &mut Vec<Beam>, prefix: &[usize]) -> usize {
            if let Some(i) = next.iter().position(|b| b.prefix == prefix) {
                return i;
            }
            next.push(Beam {
                prefix: prefix.to_vec(),
                blank: f64::NEG_INFINITY,
                non_blank: f64::NEG_INFINITY,
            });
            next.len() - 1
        }

        for beam in &beams {
            let last = beam.prefix.last().copied();

            // Emit a blank: the prefix is unchanged and now ends in a blank.
            let i = slot(&mut next, &beam.prefix);
            next[i].blank = log_add_f64(next[i].blank, beam.total() + lp_t[blank]);

            for (c, &lp_c) in lp_t.iter().enumerate().take(classes) {
                if c == blank {
                    continue;
                }
                if Some(c) == last {
                    // Repeating the last label without an intervening blank
                    // extends the existing run rather than adding a symbol.
                    let i = slot(&mut next, &beam.prefix);
                    next[i].non_blank = log_add_f64(next[i].non_blank, beam.non_blank + lp_c);

                    // Reaching it through a blank does add a second symbol.
                    let mut extended = beam.prefix.clone();
                    extended.push(c);
                    let j = slot(&mut next, &extended);
                    next[j].non_blank = log_add_f64(next[j].non_blank, beam.blank + lp_c);
                } else {
                    let mut extended = beam.prefix.clone();
                    extended.push(c);
                    let j = slot(&mut next, &extended);
                    next[j].non_blank = log_add_f64(next[j].non_blank, beam.total() + lp_c);
                }
            }
        }

        next.sort_by(|a, b| {
            b.total()
                .partial_cmp(&a.total())
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        next.truncate(width);
        beams = next;
    }

    beams
        .into_iter()
        .next()
        .map(|b| b.prefix)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{log_add, Ln1p};
    use std::vec::Vec;

    /// The generic path must match the accuracy of a native `ln_1p`, not the
    /// naive `(1 + x).ln()` it replaced.
    #[test]
    fn ln_1p_matches_the_native_implementation() {
        let mut worst_ours = 0.0f64;
        let mut worst_naive = 0.0f64;
        // Sweep down to where `1 + x` stops being representable as distinct.
        let xs: Vec<f64> = (0..320).map(|i| 0.5f64.powi(i)).collect();
        for &x in &xs {
            let want = x.ln_1p();
            if want == 0.0 {
                continue;
            }
            let ours = (x.ln_1p_compat() - want).abs() / want.abs();
            let naive = ((1.0f64 + x).ln() - want).abs() / want.abs();
            worst_ours = worst_ours.max(ours);
            worst_naive = worst_naive.max(naive);
        }
        assert!(
            worst_ours < 1e-15,
            "Kahan correction drifted by {worst_ours} relative"
        );
        // The naive form loses everything once `1 + x` rounds to `1`.
        assert!(
            worst_naive > 0.5,
            "expected the naive form to fail badly, worst was {worst_naive}"
        );
    }

    /// `log_add` is the only consumer, and it must stay exact when one term
    /// dwarfs the other — the case CTC hits constantly.
    #[test]
    fn log_add_survives_lopsided_terms() {
        for exponent in [1i32, 10, 30, 60, 200, 700] {
            let hi = -1.0f64;
            let lo = hi - exponent as f64;
            let got = log_add(hi, lo);
            let want = hi + (lo - hi).exp().ln_1p();
            assert!(
                (got - want).abs() <= 1e-15 * want.abs().max(1.0),
                "log_add({hi}, {lo}) = {got}, want {want}"
            );
            // Adding a vastly smaller term must never lose the larger one.
            assert!(got >= hi, "log_add fell below its own maximum");
        }
    }

    #[test]
    fn log_add_handles_impossible_paths() {
        assert_eq!(log_add(f64::NEG_INFINITY, -3.0), -3.0);
        assert_eq!(log_add(-3.0, f64::NEG_INFINITY), -3.0);
        assert_eq!(
            log_add(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn log_add_is_commutative() {
        for (a, b) in [(-1.0f64, -2.0), (-0.5, -40.0), (-100.0, -0.25)] {
            assert_eq!(log_add(a, b), log_add(b, a));
        }
    }
}
