use super::*;
use tang_compile::tensor::Op;

fn tiny() -> Config {
    Config {
        vocab: 11,
        d: 8,
        layers: 2,
        heads: 2,
        ff: 12,
        seq: 5,
        batch: 1,
    }
}

fn input_index(g: &TensorGraph, t: TensorId) -> usize {
    match g.nodes()[t.index()].op {
        Op::Input(k) => k,
        _ => panic!("not an input"),
    }
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Every graph input's value: tokens, targets, RoPE tables, parameters.
fn inputs(m: &Model, seed: u64) -> Vec<Vec<f64>> {
    let g = &m.graph;
    let n = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Input(_)))
        .count();
    let mut ins = vec![Vec::new(); n];
    let mut r = Rng(seed);
    let c = m.config;
    ins[input_index(g, m.tokens)] = (0..c.rows())
        .map(|i| ((i * 7 + 3) % c.vocab) as f64)
        .collect();
    ins[input_index(g, m.targets)] = (0..c.rows())
        .map(|i| ((i * 5 + 1) % c.vocab) as f64)
        .collect();
    if let Some((cos, sin)) = m.rope {
        let (cv, sv) = rope_tables(&c);
        ins[input_index(g, cos)] = cv;
        ins[input_index(g, sin)] = sv;
    }
    for p in &m.params {
        let len: usize = p.shape.iter().product();
        ins[input_index(g, p.id)] = match p.init {
            Init::Normal(s) => (0..len).map(|_| r.f() * s * 3.0).collect(),
            Init::Const(v) => (0..len).map(|_| v + 0.1 * r.f()).collect(),
        };
    }
    ins
}

fn loss_at(m: &Model, ins: &[Vec<f64>]) -> f64 {
    m.graph.eval(ins, &[m.loss])[0][0]
}

/// Central differences on a sample of every parameter's coordinates.
fn check_grads(v: Variant) {
    let m = build(tiny(), v);
    let ins = inputs(&m, 7);
    let grads = m.graph.eval(&ins, &m.grads);
    let mut r = Rng(99);
    let h = 1e-5;
    for (p, gr) in m.params.iter().zip(&grads) {
        let k = input_index(&m.graph, p.id);
        let len = ins[k].len();
        for _ in 0..len.min(6) {
            let j = ((r.f() + 1.0) * 0.5 * len as f64) as usize % len;
            let (mut a, mut b) = (ins.to_vec(), ins.to_vec());
            a[k][j] += h;
            b[k][j] -= h;
            let fd = (loss_at(&m, &a) - loss_at(&m, &b)) / (2.0 * h);
            let tol = 1e-5 + 1e-3 * fd.abs().max(gr[j].abs());
            assert!(
                (fd - gr[j]).abs() <= tol,
                "{v:?} {}[{j}]: backward {} vs finite difference {fd}",
                p.name,
                gr[j]
            );
        }
    }
}

#[test]
fn twin_gradients_match_finite_differences() {
    check_grads(Variant::TWIN);
}

#[test]
fn dyt_gradients_match_finite_differences() {
    check_grads(Variant {
        dyt: true,
        ..Variant::TWIN
    });
}

#[test]
fn relu2_gradients_match_finite_differences() {
    check_grads(Variant {
        relu2: true,
        ..Variant::TWIN
    });
}

#[test]
fn retention_gradients_match_finite_differences() {
    check_grads(Variant {
        retention: true,
        ..Variant::TWIN
    });
}

#[test]
fn all_smooth_constraints_together_match_finite_differences() {
    check_grads(Variant {
        ternary: false,
        ..Variant::NATIVE
    });
}

/// The host's absmean ternary quantizer, the same arithmetic as the graph's.
fn quantize(w: &[f64]) -> Vec<f64> {
    let s = w.iter().map(|x| x.abs()).sum::<f64>() / w.len() as f64 + 1e-8;
    w.iter()
        .map(|&x| {
            let u = x / s;
            let t = if u - 0.5 > 0.0 {
                1.0
            } else if -0.5 - u > 0.0 {
                -1.0
            } else {
                0.0
            };
            t * s
        })
        .collect()
}

fn is_linear(name: &str) -> bool {
    ["wq", "wk", "wv", "wo", "up", "gate", "down"]
        .iter()
        .any(|s| name.ends_with(&format!(".{s}")))
}

#[test]
fn ternary_is_a_straight_through_estimator() {
    // the ternary model at θ must equal the full-precision model at θ with
    // its matrices quantized — in loss and in every gradient
    for v in [Variant::parse("ternary").unwrap(), Variant::NATIVE] {
        let mt = build(tiny(), v);
        let mf = build(
            tiny(),
            Variant {
                ternary: false,
                ..v
            },
        );
        let ins_t = inputs(&mt, 3);
        let mut ins_f = ins_t.clone();
        for p in &mt.params {
            if is_linear(&p.name) {
                let k = input_index(&mt.graph, p.id);
                ins_f[k] = quantize(&ins_t[k]);
            }
        }
        let (lt, lf) = (loss_at(&mt, &ins_t), loss_at(&mf, &ins_f));
        assert!(
            (lt - lf).abs() < 1e-9,
            "{v:?}: forward uses quantized weights ({lt} vs {lf})"
        );
        let (gt, gf) = (
            mt.graph.eval(&ins_t, &mt.grads),
            mf.graph.eval(&ins_f, &mf.grads),
        );
        for (p, (a, b)) in mt.params.iter().zip(gt.iter().zip(&gf)) {
            for (x, y) in a.iter().zip(b) {
                assert!(
                    (x - y).abs() < 1e-9,
                    "{v:?} {}: STE gradient {x} vs {y}",
                    p.name
                );
            }
        }
        assert!(mt.params.iter().any(|p| is_linear(&p.name)));
    }
}

#[test]
fn every_variant_builds_and_the_ladder_parses() {
    for name in ["twin", "ternary", "dyt", "relu2", "native"] {
        let v = Variant::parse(name).unwrap();
        let m = build(tiny(), v);
        let l = loss_at(&m, &inputs(&m, 1));
        assert!(l.is_finite() && l > 0.0, "{name}: loss {l}");
    }
    assert!(Variant::parse("nope").is_none());
}

/// Every rung of the ladder learns a simple periodic sequence: the
/// training step (clipping, AdamW, the schedule input) works end to end.
#[test]
fn every_variant_learns_a_periodic_sequence() {
    use crate::train::{Hyper, Step, Trainer};
    let c = Config {
        vocab: 7,
        d: 16,
        layers: 1,
        heads: 2,
        ff: 32,
        seq: 8,
        batch: 1,
    };
    let seq: Vec<u32> = (0..c.seq + 1).map(|i| (i % 5) as u32 + 1).collect();
    let (x, y) = (&seq[..c.seq], &seq[1..]);
    for name in ["twin", "ternary", "dyt", "relu2", "native"] {
        let v = Variant::parse(name).unwrap();
        let mut tr = Trainer::new(Step::new(c, v, Hyper::default(), false), 5);
        let (first, _) = tr.step(x, y, 3e-2);
        let mut last = first;
        for _ in 0..60 {
            last = tr.step(x, y, 3e-2).0;
        }
        eprintln!("{name:8} loss {first:.3} → {last:.3}");
        assert!(
            last.is_finite() && last < 0.5 * first,
            "{name} did not learn: {first} → {last}"
        );
    }
}

/// A batch of two is the mean of the two sequences run alone: in loss and
/// in every gradient, for both attention kinds. Attention never mixes
/// sequences.
#[test]
fn a_batch_is_the_mean_of_its_sequences() {
    for v in [Variant::TWIN, Variant::NATIVE] {
        let one = build(tiny(), v);
        let two = build(Config { batch: 2, ..tiny() }, v);
        let base = inputs(&one, 4);
        let seqs: [(Vec<f64>, Vec<f64>); 2] = [
            (
                (0..5).map(|i| ((i * 3 + 1) % 11) as f64).collect(),
                (0..5).map(|i| ((i * 2 + 5) % 11) as f64).collect(),
            ),
            (
                (0..5).map(|i| ((i * 4 + 2) % 11) as f64).collect(),
                (0..5).map(|i| ((i + 7) % 11) as f64).collect(),
            ),
        ];
        let mut want_loss = 0.0;
        let mut want_grads: Vec<Vec<f64>> = Vec::new();
        for (tk, tg) in &seqs {
            let mut ins = base.clone();
            ins[input_index(&one.graph, one.tokens)] = tk.clone();
            ins[input_index(&one.graph, one.targets)] = tg.clone();
            want_loss += loss_at(&one, &ins) / 2.0;
            let gs = one.graph.eval(&ins, &one.grads);
            if want_grads.is_empty() {
                want_grads = gs
                    .iter()
                    .map(|g| g.iter().map(|x| x / 2.0).collect())
                    .collect();
            } else {
                for (w, g) in want_grads.iter_mut().zip(&gs) {
                    for (a, b) in w.iter_mut().zip(g) {
                        *a += b / 2.0;
                    }
                }
            }
        }
        // the batch-of-two graph: same parameters, sequences stacked
        let mut ins2 = inputs(&two, 4);
        for (p1, p2) in one.params.iter().zip(&two.params) {
            ins2[input_index(&two.graph, p2.id)] = base[input_index(&one.graph, p1.id)].clone();
        }
        if let (Some((c1, s1)), Some((c2, s2))) = (one.rope, two.rope) {
            ins2[input_index(&two.graph, c2)] = base[input_index(&one.graph, c1)].clone();
            ins2[input_index(&two.graph, s2)] = base[input_index(&one.graph, s1)].clone();
        }
        ins2[input_index(&two.graph, two.tokens)] = [seqs[0].0.clone(), seqs[1].0.clone()].concat();
        ins2[input_index(&two.graph, two.targets)] =
            [seqs[0].1.clone(), seqs[1].1.clone()].concat();
        let got = loss_at(&two, &ins2);
        assert!(
            (got - want_loss).abs() < 1e-9,
            "{v:?}: batch loss {got} vs {want_loss}"
        );
        let g2 = two.graph.eval(&ins2, &two.grads);
        for (p, (a, b)) in two.params.iter().zip(g2.iter().zip(&want_grads)) {
            for (x, y) in a.iter().zip(b) {
                assert!(
                    (x - y).abs() < 1e-9,
                    "{v:?} {}: batch grad {x} vs {y}",
                    p.name
                );
            }
        }
    }
}
