//! CTC is validated three ways: against closed-form values, against brute-force
//! enumeration of every alignment, and against finite-difference gradients.

use tang_tensor::{Shape, Tensor};
use tang_train::{
    beam_decode, collapse, ctc_loss, ctc_loss_batch, ctc_loss_batch_grad, ctc_loss_grad,
    greedy_decode, min_input_len, CtcError,
};

const BLANK: usize = 0;

fn tensor(rows: &[&[f64]]) -> Tensor<f64> {
    let time = rows.len();
    let classes = rows[0].len();
    let data: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::new(data, Shape::from_slice(&[time, classes]))
}

fn log_softmax_row(row: &[f64]) -> Vec<f64> {
    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let denom: f64 = row.iter().map(|v| (v - max).exp()).sum();
    row.iter().map(|v| v - max - denom.ln()).collect()
}

/// Sum the probability of every length-`time` alignment that collapses to
/// `target`, by enumerating all of them. Exponential, so only for tiny cases —
/// but it depends on nothing in the implementation under test.
fn brute_force(logits: &Tensor<f64>, target: &[usize], blank: usize) -> f64 {
    let time = logits.shape()[0];
    let classes = logits.shape()[1];
    let lp: Vec<Vec<f64>> = (0..time)
        .map(|t| {
            let row: Vec<f64> = (0..classes).map(|c| logits.get(&[t, c])).collect();
            log_softmax_row(&row)
        })
        .collect();

    let mut total = 0.0f64;
    let mut path = vec![0usize; time];
    let combos = classes.pow(time as u32);
    for n in 0..combos {
        let mut rem = n;
        for slot in path.iter_mut() {
            *slot = rem % classes;
            rem /= classes;
        }
        if collapse(&path, blank) == target {
            let logp: f64 = path.iter().enumerate().map(|(t, &c)| lp[t][c]).sum();
            total += logp.exp();
        }
    }
    total
}

#[test]
fn collapse_merges_repeats_and_drops_blanks() {
    // "a-abb-" -> "aab": the blank between the a's keeps them separate,
    // while the doubled b collapses to one.
    assert_eq!(collapse(&[1, 0, 1, 2, 2, 0], BLANK), vec![1, 1, 2]);
    assert_eq!(collapse(&[0, 0, 0], BLANK), vec![]);
    assert_eq!(collapse(&[1, 1, 1], BLANK), vec![1]);
    assert_eq!(collapse(&[], BLANK), vec![]);
}

#[test]
fn min_input_len_accounts_for_repeats() {
    assert_eq!(min_input_len(&[]), 0);
    assert_eq!(min_input_len(&[1, 2, 3]), 3);
    // "aa" needs a blank wedged between the two a's.
    assert_eq!(min_input_len(&[1, 1]), 3);
    assert_eq!(min_input_len(&[1, 1, 1]), 5);
    assert_eq!(min_input_len(&[1, 2, 2, 3]), 5);
}

#[test]
fn single_step_loss_is_just_negative_log_prob() {
    // One step, one label: the only alignment is the label itself.
    let logits = tensor(&[&[0.0, 1.0, 2.0]]);
    let loss = ctc_loss(&logits, &[1], BLANK).unwrap();
    let expected = -log_softmax_row(&[0.0, 1.0, 2.0])[1];
    assert!((loss - expected).abs() < 1e-12, "{loss} vs {expected}");
}

#[test]
fn empty_target_is_the_all_blank_path() {
    let logits = tensor(&[&[1.0, 0.5, 0.2], &[0.3, 0.8, 0.1]]);
    let loss = ctc_loss(&logits, &[], BLANK).unwrap();
    let lp0 = log_softmax_row(&[1.0, 0.5, 0.2]);
    let lp1 = log_softmax_row(&[0.3, 0.8, 0.1]);
    let expected = -(lp0[BLANK] + lp1[BLANK]);
    assert!((loss - expected).abs() < 1e-12, "{loss} vs {expected}");
}

#[test]
fn loss_matches_brute_force_enumeration() {
    // Every case small enough to enumerate exhaustively.
    let cases: &[(&[&[f64]], &[usize])] = &[
        (&[&[0.1, 0.6, 0.3], &[0.4, 0.2, 0.9]], &[1]),
        (&[&[0.1, 0.6, 0.3], &[0.4, 0.2, 0.9]], &[2]),
        (&[&[0.1, 0.6, 0.3], &[0.4, 0.2, 0.9]], &[1, 2]),
        (
            &[&[0.5, 0.1, 0.2], &[0.3, 0.7, 0.4], &[0.2, 0.1, 0.8]],
            &[1, 2],
        ),
        (
            &[&[0.5, 0.1, 0.2], &[0.3, 0.7, 0.4], &[0.2, 0.1, 0.8]],
            &[1, 1],
        ),
        (
            &[&[0.5, 0.1, 0.2], &[0.3, 0.7, 0.4], &[0.2, 0.1, 0.8]],
            &[2],
        ),
        (
            &[
                &[0.5, 0.1, 0.2],
                &[0.3, 0.7, 0.4],
                &[0.2, 0.1, 0.8],
                &[0.9, 0.2, 0.1],
            ],
            &[1, 2, 1],
        ),
    ];
    for (rows, target) in cases {
        let logits = tensor(rows);
        let loss = ctc_loss(&logits, target, BLANK).unwrap();
        let brute = brute_force(&logits, target, BLANK);
        let expected = -brute.ln();
        assert!(
            (loss - expected).abs() < 1e-10,
            "target {target:?}: forward gave {loss}, enumeration gave {expected}"
        );
    }
}

#[test]
fn all_targets_of_a_length_sum_to_at_most_one() {
    // The alignment sets for distinct targets are disjoint, so their
    // probabilities cannot sum past 1.
    let logits = tensor(&[&[0.5, 0.1, 0.2], &[0.3, 0.7, 0.4], &[0.2, 0.1, 0.8]]);
    let targets: Vec<Vec<usize>> = vec![
        vec![],
        vec![1],
        vec![2],
        vec![1, 1],
        vec![1, 2],
        vec![2, 1],
        vec![2, 2],
        vec![1, 2, 1],
        vec![2, 1, 2],
    ];
    let total: f64 = targets
        .iter()
        .filter_map(|t| ctc_loss(&logits, t, BLANK).ok())
        .map(|l| (-l).exp())
        .sum();
    assert!(total <= 1.0 + 1e-9, "probabilities sum to {total}");
    assert!(
        total > 0.5,
        "expected most of the mass to be covered, got {total}"
    );
}

#[test]
fn gradient_matches_finite_differences() {
    let cases: &[(&[&[f64]], &[usize])] = &[
        (&[&[0.1, 0.6, 0.3], &[0.4, 0.2, 0.9]], &[1]),
        (
            &[&[0.5, 0.1, 0.2], &[0.3, 0.7, 0.4], &[0.2, 0.1, 0.8]],
            &[1, 2],
        ),
        (
            &[&[0.5, 0.1, 0.2], &[0.3, 0.7, 0.4], &[0.2, 0.1, 0.8]],
            &[1, 1],
        ),
        (
            &[
                &[0.2, 0.9, 0.1],
                &[0.7, 0.3, 0.5],
                &[0.1, 0.4, 0.8],
                &[0.6, 0.2, 0.3],
            ],
            &[2, 1],
        ),
    ];

    for (rows, target) in cases {
        let logits = tensor(rows);
        let analytic = ctc_loss_grad(&logits, target, BLANK).unwrap();
        let time = logits.shape()[0];
        let classes = logits.shape()[1];
        let h = 1e-6;

        for t in 0..time {
            for c in 0..classes {
                let mut up = logits.data().to_vec();
                let mut down = logits.data().to_vec();
                up[t * classes + c] += h;
                down[t * classes + c] -= h;
                let shape = Shape::from_slice(&[time, classes]);
                let l_up = ctc_loss(&Tensor::new(up, shape.clone()), target, BLANK).unwrap();
                let l_down = ctc_loss(&Tensor::new(down, shape), target, BLANK).unwrap();
                let numeric = (l_up - l_down) / (2.0 * h);
                let got = analytic.get(&[t, c]);
                assert!(
                    (got - numeric).abs() < 1e-6,
                    "target {target:?} at [{t},{c}]: analytic {got}, numeric {numeric}"
                );
            }
        }
    }
}

#[test]
fn gradient_rows_sum_to_zero() {
    // softmax sums to 1 and the posterior sums to 1, so each time step's
    // gradient must sum to 0. A row that does not is a normalisation bug.
    let logits = tensor(&[&[0.5, 0.1, 0.2], &[0.3, 0.7, 0.4], &[0.2, 0.1, 0.8]]);
    let grad = ctc_loss_grad(&logits, &[1, 2], BLANK).unwrap();
    for t in 0..3 {
        let sum: f64 = (0..3).map(|c| grad.get(&[t, c])).sum();
        assert!(sum.abs() < 1e-10, "row {t} sums to {sum}");
    }
}

#[test]
fn impossible_targets_are_rejected() {
    let logits = tensor(&[&[0.1, 0.6, 0.3], &[0.4, 0.2, 0.9]]);
    // Three labels cannot come out of two steps.
    assert_eq!(
        ctc_loss(&logits, &[1, 2, 1], BLANK),
        Err(CtcError::TargetTooLong {
            input_len: 2,
            target_len: 3,
            required: 3
        })
    );
    // "aa" needs three steps, not two.
    assert_eq!(
        ctc_loss(&logits, &[1, 1], BLANK),
        Err(CtcError::TargetTooLong {
            input_len: 2,
            target_len: 2,
            required: 3
        })
    );
    assert_eq!(
        ctc_loss(&logits, &[0], BLANK),
        Err(CtcError::BlankInTarget { position: 0 })
    );
    assert_eq!(
        ctc_loss(&logits, &[7], BLANK),
        Err(CtcError::LabelOutOfRange {
            label: 7,
            classes: 3
        })
    );
    assert_eq!(
        ctc_loss(&logits, &[1], 9),
        Err(CtcError::BlankOutOfRange {
            blank: 9,
            classes: 3
        })
    );
}

#[test]
fn confident_alignment_has_near_zero_loss() {
    // A model that is certain of blank, a, blank should see almost no loss
    // for target "a".
    let logits = tensor(&[&[20.0, 0.0, 0.0], &[0.0, 20.0, 0.0], &[20.0, 0.0, 0.0]]);
    let loss = ctc_loss(&logits, &[1], BLANK).unwrap();
    assert!(loss < 1e-6, "expected near-zero loss, got {loss}");
}

#[test]
fn loss_is_never_negative() {
    let logits = tensor(&[&[0.5, 0.1, 0.2], &[0.3, 0.7, 0.4], &[0.2, 0.1, 0.8]]);
    for target in [vec![1], vec![2], vec![1, 2], vec![2, 1]] {
        let loss = ctc_loss(&logits, &target, BLANK).unwrap();
        assert!(loss >= 0.0, "target {target:?} gave negative loss {loss}");
    }
}

#[test]
fn long_sequences_do_not_underflow() {
    // 500 steps of near-uniform probability would underflow f64 outside log
    // space within about 40 steps.
    let time = 500;
    let classes = 5;
    let data: Vec<f64> = (0..time * classes)
        .map(|i| ((i % 7) as f64) * 0.1)
        .collect();
    let logits = Tensor::new(data, Shape::from_slice(&[time, classes]));
    let target: Vec<usize> = (0..100).map(|i| 1 + (i % 4)).collect();
    let loss = ctc_loss(&logits, &target, BLANK).unwrap();
    assert!(loss.is_finite(), "loss underflowed to {loss}");
    assert!(loss > 0.0);

    let grad = ctc_loss_grad(&logits, &target, BLANK).unwrap();
    assert!(
        grad.data().iter().all(|g| g.is_finite()),
        "gradient has non-finite entries"
    );
}

#[test]
fn greedy_decode_takes_the_best_path() {
    // Argmax per step is blank, a, a, blank, b -> collapses to "ab".
    let logits = tensor(&[
        &[5.0, 0.0, 0.0],
        &[0.0, 5.0, 0.0],
        &[0.0, 5.0, 0.0],
        &[5.0, 0.0, 0.0],
        &[0.0, 0.0, 5.0],
    ]);
    assert_eq!(greedy_decode(&logits, BLANK), vec![1, 2]);
}

#[test]
fn greedy_decode_of_all_blanks_is_empty() {
    let logits = tensor(&[&[5.0, 0.0], &[5.0, 0.0]]);
    assert_eq!(greedy_decode(&logits, BLANK), Vec::<usize>::new());
}

#[test]
fn beam_decode_agrees_with_greedy_when_the_path_is_obvious() {
    let logits = tensor(&[
        &[5.0, 0.0, 0.0],
        &[0.0, 5.0, 0.0],
        &[5.0, 0.0, 0.0],
        &[0.0, 0.0, 5.0],
    ]);
    assert_eq!(
        beam_decode(&logits, BLANK, 8),
        greedy_decode(&logits, BLANK)
    );
}

#[test]
fn beam_decode_beats_greedy_when_alignments_split() {
    // Mass for "a" is spread over several alignments while blank narrowly wins
    // each individual step. Greedy follows the per-step argmax and returns the
    // empty string; beam search sums the alignments and recovers "a".
    let logits = tensor(&[
        &[0.60, 0.40, 0.0],
        &[0.55, 0.45, 0.0],
        &[0.55, 0.45, 0.0],
        &[0.55, 0.45, 0.0],
    ]);
    let greedy = greedy_decode(&logits, BLANK);
    let beam = beam_decode(&logits, BLANK, 16);

    assert_eq!(greedy, Vec::<usize>::new(), "greedy should follow blanks");

    // Whatever beam search returns must be at least as likely as greedy's answer.
    let p_beam = (-ctc_loss(&logits, &beam, BLANK).unwrap()).exp();
    let p_greedy = (-ctc_loss(&logits, &greedy, BLANK).unwrap()).exp();
    assert!(
        p_beam >= p_greedy - 1e-12,
        "beam returned {beam:?} at p={p_beam}, worse than greedy {greedy:?} at p={p_greedy}"
    );
    assert_eq!(
        beam,
        vec![1],
        "beam should recover the split-alignment label"
    );
}

#[test]
fn wider_beams_never_get_worse() {
    let logits = tensor(&[
        &[0.5, 0.4, 0.3],
        &[0.2, 0.6, 0.5],
        &[0.4, 0.3, 0.6],
        &[0.7, 0.2, 0.4],
        &[0.3, 0.5, 0.4],
    ]);
    let mut last = f64::NEG_INFINITY;
    for width in [1, 2, 4, 8, 16, 32] {
        let decoded = beam_decode(&logits, BLANK, width);
        let p = (-ctc_loss(&logits, &decoded, BLANK).unwrap()).exp();
        assert!(
            p >= last - 1e-12,
            "width {width} gave p={p}, worse than the narrower beam's {last}"
        );
        last = p;
    }
}

#[test]
fn batch_loss_is_the_mean_of_its_parts() {
    let a = [0.5f64, 0.1, 0.2, 0.3, 0.7, 0.4, 0.2, 0.1, 0.8];
    let b = [0.2f64, 0.9, 0.1, 0.7, 0.3, 0.5, 0.1, 0.4, 0.8];
    let mut data = a.to_vec();
    data.extend_from_slice(&b);
    let batched = Tensor::new(data, Shape::from_slice(&[2, 3, 3]));

    let t0: &[usize] = &[1, 2];
    let t1: &[usize] = &[2];
    let got = ctc_loss_batch(&batched, &[t0, t1], BLANK).unwrap();

    let l0 = ctc_loss(
        &Tensor::new(a.to_vec(), Shape::from_slice(&[3, 3])),
        t0,
        BLANK,
    )
    .unwrap();
    let l1 = ctc_loss(
        &Tensor::new(b.to_vec(), Shape::from_slice(&[3, 3])),
        t1,
        BLANK,
    )
    .unwrap();
    assert!((got - (l0 + l1) / 2.0).abs() < 1e-12);
}

#[test]
fn batch_gradient_matches_finite_differences() {
    let data: Vec<f64> = vec![
        0.5, 0.1, 0.2, 0.3, 0.7, 0.4, 0.2, 0.1, 0.8, //
        0.2, 0.9, 0.1, 0.7, 0.3, 0.5, 0.1, 0.4, 0.8,
    ];
    let shape = Shape::from_slice(&[2, 3, 3]);
    let logits = Tensor::new(data.clone(), shape.clone());
    let t0: &[usize] = &[1, 2];
    let t1: &[usize] = &[2];
    let targets = [t0, t1];

    let analytic = ctc_loss_batch_grad(&logits, &targets, BLANK).unwrap();
    let h = 1e-6;
    for i in 0..data.len() {
        let mut up = data.clone();
        let mut down = data.clone();
        up[i] += h;
        down[i] -= h;
        let l_up = ctc_loss_batch(&Tensor::new(up, shape.clone()), &targets, BLANK).unwrap();
        let l_down = ctc_loss_batch(&Tensor::new(down, shape.clone()), &targets, BLANK).unwrap();
        let numeric = (l_up - l_down) / (2.0 * h);
        let got = analytic.data()[i];
        assert!(
            (got - numeric).abs() < 1e-6,
            "at {i}: analytic {got}, numeric {numeric}"
        );
    }
}

#[test]
fn gradient_descent_drives_the_loss_down() {
    // The end-to-end claim: following this gradient actually trains.
    let time = 8;
    let classes = 5;
    let target: &[usize] = &[1, 2, 2, 3];
    let mut data: Vec<f64> = (0..time * classes)
        .map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.05)
        .collect();
    let shape = Shape::from_slice(&[time, classes]);

    let first = ctc_loss(&Tensor::new(data.clone(), shape.clone()), target, BLANK).unwrap();
    // Plain gradient descent, no momentum: the tail is slow, so give it room.
    for _ in 0..2000 {
        let logits = Tensor::new(data.clone(), shape.clone());
        let grad = ctc_loss_grad(&logits, target, BLANK).unwrap();
        for (d, g) in data.iter_mut().zip(grad.data()) {
            *d -= 0.5 * g;
        }
    }
    let logits = Tensor::new(data, shape);
    let last = ctc_loss(&logits, target, BLANK).unwrap();

    assert!(last < first, "loss went {first} -> {last}");
    assert!(last < 0.01, "expected convergence, ended at {last}");
    assert!(last.is_finite());
    // And the trained model decodes to the target it was trained on.
    assert_eq!(greedy_decode(&logits, BLANK), target.to_vec());
}
