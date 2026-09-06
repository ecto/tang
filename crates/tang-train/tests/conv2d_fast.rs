//! The fast im2col + GEMM `Conv2d` against the direct `get`-per-tap reference,
//! and against a finite difference.

use tang_tensor::{Shape, Tensor};
use tang_train::{Conv2d, Module};

/// Deterministic, cheap, and decorrelated across indices — enough to catch a
/// transposed axis or an off-by-one in the padding.
fn noise(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn tensor(seed: u64, dims: &[usize]) -> Tensor<f64> {
    let shape = Shape::from_slice(dims);
    Tensor::new(noise(seed, shape.numel()), shape)
}

fn max_rel_err(a: &Tensor<f64>, b: &Tensor<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| (x - y).abs() / (x.abs().max(y.abs())).max(1.0))
        .fold(0.0, f64::max)
}

struct Case {
    batch: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

/// Odd sizes, both paddings, stride 2, dilation 2, batch > 1, and channel
/// counts that are not multiples of 8 — the shapes where a blocked kernel is
/// most likely to have an edge-case bug.
const CASES: &[Case] = &[
    Case {
        batch: 1,
        c_in: 1,
        c_out: 1,
        h: 5,
        w: 5,
        k: 3,
        stride: 1,
        padding: 0,
        dilation: 1,
    },
    Case {
        batch: 3,
        c_in: 3,
        c_out: 5,
        h: 7,
        w: 9,
        k: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
    },
    Case {
        batch: 2,
        c_in: 5,
        c_out: 7,
        h: 11,
        w: 7,
        k: 3,
        stride: 2,
        padding: 1,
        dilation: 1,
    },
    Case {
        batch: 2,
        c_in: 3,
        c_out: 3,
        h: 9,
        w: 9,
        k: 3,
        stride: 1,
        padding: 2,
        dilation: 2,
    },
    Case {
        batch: 4,
        c_in: 10,
        c_out: 13,
        h: 8,
        w: 8,
        k: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
    },
    Case {
        batch: 2,
        c_in: 6,
        c_out: 6,
        h: 13,
        w: 11,
        k: 5,
        stride: 3,
        padding: 2,
        dilation: 1,
    },
    Case {
        batch: 1,
        c_in: 4,
        c_out: 9,
        h: 15,
        w: 15,
        k: 5,
        stride: 2,
        padding: 4,
        dilation: 2,
    },
    Case {
        batch: 2,
        c_in: 7,
        c_out: 1,
        h: 6,
        w: 10,
        k: 1,
        stride: 1,
        padding: 0,
        dilation: 1,
    },
];

fn build(c: &Case, seed: u64) -> Conv2d<f64> {
    let mut conv =
        Conv2d::<f64>::with_options(c.c_in, c.c_out, c.k, c.stride, c.padding, c.dilation, seed);
    conv.weight.data = tensor(seed ^ 0x5EED_1111, &[c.c_out, c.c_in, c.k, c.k]);
    conv.bias.data = tensor(seed ^ 0x5EED_2222, &[c.c_out]);
    conv
}

#[test]
fn the_fast_forward_matches_the_reference_on_every_shape() {
    for (n, c) in CASES.iter().enumerate() {
        let seed = 0x1234 + n as u64 * 977;
        let x = tensor(seed, &[c.batch, c.c_in, c.h, c.w]);

        let mut fast = build(c, seed);
        let mut slow = build(c, seed);

        let y_fast = fast.forward(&x);
        let y_slow = slow.forward_reference(&x);

        let err = max_rel_err(&y_fast, &y_slow);
        assert!(err < 1e-12, "case {n}: forward rel err {err:e}");
    }
}

#[test]
fn the_fast_backward_matches_the_reference_on_every_shape() {
    for (n, c) in CASES.iter().enumerate() {
        let seed = 0x9ABC + n as u64 * 733;
        let x = tensor(seed, &[c.batch, c.c_in, c.h, c.w]);

        let mut fast = build(c, seed);
        let mut slow = build(c, seed);

        let y = fast.forward(&x);
        slow.forward_reference(&x);
        let gy = tensor(seed ^ 0xFACE, y.shape().dims());

        let gx_fast = fast.backward(&gy);
        let gx_slow = slow.backward_reference(&gy);

        let e_in = max_rel_err(&gx_fast, &gx_slow);
        let e_w = max_rel_err(
            fast.weight.grad.as_ref().unwrap(),
            slow.weight.grad.as_ref().unwrap(),
        );
        let e_b = max_rel_err(
            fast.bias.grad.as_ref().unwrap(),
            slow.bias.grad.as_ref().unwrap(),
        );
        assert!(e_in < 1e-12, "case {n}: grad_input rel err {e_in:e}");
        assert!(e_w < 1e-12, "case {n}: grad_weight rel err {e_w:e}");
        assert!(e_b < 1e-12, "case {n}: grad_bias rel err {e_b:e}");
    }
}

/// Gradients must accumulate across calls rather than replace, since a
/// training step may run several forward/backward pairs before the optimiser.
#[test]
fn gradients_accumulate_across_backward_calls() {
    let c = &CASES[1];
    let x = tensor(7, &[c.batch, c.c_in, c.h, c.w]);
    let mut conv = build(c, 7);
    let y = conv.forward(&x);
    let gy = tensor(8, y.shape().dims());

    conv.backward(&gy);
    let once = conv.weight.grad.clone().unwrap();
    conv.forward(&x);
    conv.backward(&gy);
    let twice = conv.weight.grad.clone().unwrap();

    for (a, b) in once.data().iter().zip(twice.data()) {
        assert!((2.0 * a - b).abs() < 1e-12);
    }
}

/// Central differences on `L = Σ g·y` for every parameter and input of a small
/// padded, strided convolution.
#[test]
fn the_backward_pass_agrees_with_a_finite_difference() {
    let c = Case {
        batch: 2,
        c_in: 3,
        c_out: 2,
        h: 6,
        w: 5,
        k: 3,
        stride: 2,
        padding: 1,
        dilation: 1,
    };
    let x = tensor(0xD1CE, &[c.batch, c.c_in, c.h, c.w]);
    let mut conv = build(&c, 0xD1CE);

    let y = conv.forward(&x);
    let g = tensor(0x5A17, y.shape().dims());
    let loss = |y: &Tensor<f64>| -> f64 { y.data().iter().zip(g.data()).map(|(a, b)| a * b).sum() };

    let grad_x = conv.backward(&g);
    let grad_w = conv.weight.grad.clone().unwrap();
    let grad_b = conv.bias.grad.clone().unwrap();

    // L(weight, bias, x) with the layer rebuilt from scratch each probe, so a
    // stale `cached_input` cannot leak between evaluations.
    let eval = |w: &Tensor<f64>, b: &Tensor<f64>, xp: &Tensor<f64>| -> f64 {
        let mut c2 =
            Conv2d::<f64>::with_options(c.c_in, c.c_out, c.k, c.stride, c.padding, c.dilation, 1);
        c2.weight.data = w.clone();
        c2.bias.data = b.clone();
        loss(&c2.forward(xp))
    };

    let step = 1e-6;
    let w0 = conv.weight.data.clone();
    let b0 = conv.bias.data.clone();
    let check = |analytic: f64, plus: f64, minus: f64, what: &str, i: usize| {
        let numeric = (plus - minus) / (2.0 * step);
        let err = (analytic - numeric).abs() / analytic.abs().max(numeric.abs()).max(1e-3);
        assert!(
            err < 1e-5,
            "{what}[{i}]: analytic {analytic} vs numeric {numeric}"
        );
    };

    for i in 0..w0.numel() {
        let mut wp = w0.clone();
        wp.data_mut()[i] += step;
        let mut wm = w0.clone();
        wm.data_mut()[i] -= step;
        check(
            grad_w.data()[i],
            eval(&wp, &b0, &x),
            eval(&wm, &b0, &x),
            "weight",
            i,
        );
    }

    for i in 0..b0.numel() {
        let mut bp = b0.clone();
        bp.data_mut()[i] += step;
        let mut bm = b0.clone();
        bm.data_mut()[i] -= step;
        check(
            grad_b.data()[i],
            eval(&w0, &bp, &x),
            eval(&w0, &bm, &x),
            "bias",
            i,
        );
    }

    for i in 0..x.numel() {
        let mut xp = x.clone();
        xp.data_mut()[i] += step;
        let mut xm = x.clone();
        xm.data_mut()[i] -= step;
        check(
            grad_x.data()[i],
            eval(&w0, &b0, &xp),
            eval(&w0, &b0, &xm),
            "input",
            i,
        );
    }
}

/// f32 takes the same path through the NEON microkernels; parity there is
/// looser only because the accumulation order differs.
#[test]
fn the_fast_path_is_correct_in_f32_too() {
    let c = &CASES[4];
    let shape = Shape::from_slice(&[c.batch, c.c_in, c.h, c.w]);
    let x64 = tensor(0x3333, shape.dims());
    let x32 = Tensor::new(
        x64.data().iter().map(|v| *v as f32).collect(),
        shape.clone(),
    );

    let mut slow = build(c, 0x3333);
    let mut fast =
        Conv2d::<f32>::with_options(c.c_in, c.c_out, c.k, c.stride, c.padding, c.dilation, 1);
    fast.weight.data = Tensor::new(
        slow.weight.data.data().iter().map(|v| *v as f32).collect(),
        slow.weight.data.shape().clone(),
    );
    fast.bias.data = Tensor::new(
        slow.bias.data.data().iter().map(|v| *v as f32).collect(),
        slow.bias.data.shape().clone(),
    );

    let y32 = fast.forward(&x32);
    let y64 = slow.forward_reference(&x64);
    let err = y32
        .data()
        .iter()
        .zip(y64.data())
        .map(|(a, b)| (*a as f64 - b).abs() / b.abs().max(1.0))
        .fold(0.0, f64::max);
    assert!(err < 1e-5, "f32 forward rel err {err:e}");
}
