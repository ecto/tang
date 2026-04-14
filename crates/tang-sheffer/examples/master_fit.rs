//! Phase 3: gradient-landscape comparison.
//!
//! For each (operator, target function, depth) triple, run K random-init
//! Adam fits of a balanced binary master tree and measure the fraction
//! that converge to near-zero loss. This is the paper's proof-of-concept
//! symbolic-regression setup applied to alternative Sheffer candidates.
//!
//! Success rate here is a proxy for how benign the loss landscape is:
//! EML is known to have ~25% random-init recovery at depth 3–4; we compare
//! against PowExpSkew (constant-free + transcendental) and EDL.
//!
//! Run: `cargo run --release --example master_fit -p tang-sheffer`

use std::f64::consts::E;

use tang_sheffer::{fit, Eml, Lcg, Master, Operator, PowExpSkew, C};
use tang_sheffer::{Edl, PowSkew};

const N_DATA: usize = 16;
const N_SEEDS: usize = 16;
const N_STEPS: usize = 2000;
const LR: f64 = 0.05;
const INIT_SCALE: f64 = 0.5;
const TIGHT_LOSS: f64 = 1e-4;
const LOOSE_LOSS: f64 = 1e-2;

fn atoms_1_x(x: f64) -> Vec<C> {
    vec![C::new(1.0, 0.0), C::new(x, 0.0)]
}

fn atoms_e_x(x: f64) -> Vec<C> {
    vec![C::new(E, 0.0), C::new(x, 0.0)]
}

/// Sample points distributed on [a, b].
fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

fn random_params(rng: &mut Lcg, n: usize, scale: f64) -> Vec<f64> {
    (0..n).map(|_| rng.normal() * scale).collect()
}

/// Perturb a known-good parameter vector by Gaussian noise. Used for the
/// perturbation-recovery test: does gradient descent return the optimizer
/// to the correct basin when started near it?
fn perturb(params: &[f64], rng: &mut Lcg, noise: f64) -> Vec<f64> {
    params.iter().map(|p| p + rng.normal() * noise).collect()
}

/// Run a fit, return final loss.
fn one_run(
    master: &Master,
    op: &dyn Operator,
    xs: &[f64],
    targets: &[C],
    atoms: &dyn Fn(f64) -> Vec<C>,
    init: Vec<f64>,
) -> f64 {
    let result = fit(master, op, xs, targets, atoms, init, N_STEPS, LR);
    result.final_loss
}

struct TargetSpec<'a> {
    name: &'a str,
    f: fn(f64) -> C,
    domain: (f64, f64),
}

fn target_exp_x() -> TargetSpec<'static> {
    TargetSpec {
        name: "exp(x)",
        f: |x| C::new(x.exp(), 0.0),
        domain: (0.5, 2.0),
    }
}
fn target_ln_x() -> TargetSpec<'static> {
    TargetSpec {
        name: "ln(x)",
        f: |x| C::new(x.ln(), 0.0),
        domain: (0.5, 2.5),
    }
}
fn target_square() -> TargetSpec<'static> {
    TargetSpec {
        name: "x^2",
        f: |x| C::new(x * x, 0.0),
        domain: (0.5, 2.5),
    }
}

/// A "known good" one-hot parameter vector is constructed per target using
/// the verifier's discovered expression. Since we don't have a symbolic
/// tree-to-params converter, the perturbation-recovery test uses a
/// hand-selected one-hot as the "true" point for synthetic targets.
fn one_hot_leaf(n_atoms: usize, index: usize) -> Vec<f64> {
    let mut v = vec![-8.0; n_atoms];
    v[index] = 8.0;
    v
}

/// Build a parameter vector where each leaf is a strong one-hot picking a
/// particular atom, per `leaf_atoms[i]` for leaf i.
fn one_hot_params(n_leaves: usize, n_atoms: usize, leaf_atoms: &[usize]) -> Vec<f64> {
    (0..n_leaves)
        .flat_map(|i| one_hot_leaf(n_atoms, leaf_atoms[i]))
        .collect()
}

fn sweep(
    label: &str,
    master: &Master,
    op: &dyn Operator,
    atoms: &dyn Fn(f64) -> Vec<C>,
    spec: &TargetSpec,
) {
    let xs = linspace(spec.domain.0, spec.domain.1, N_DATA);
    let targets: Vec<C> = xs.iter().map(|x| (spec.f)(*x)).collect();

    let mut tight = 0;
    let mut loose = 0;
    let mut random_losses = Vec::with_capacity(N_SEEDS);
    for seed in 0..N_SEEDS {
        let mut rng = Lcg::new(seed as u64 * 1000 + 42);
        let init = random_params(&mut rng, master.n_params, INIT_SCALE);
        let loss = one_run(master, op, &xs, &targets, atoms, init);
        random_losses.push(loss);
        if loss < TIGHT_LOSS {
            tight += 1;
        }
        if loss < LOOSE_LOSS {
            loose += 1;
        }
    }
    random_losses.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = random_losses[N_SEEDS / 2];
    let best = random_losses[0];
    println!(
        "  {:<12} d={} | tight {:>2}/{:<2} | loose {:>2}/{:<2} | best {:.2e} | median {:.2e}",
        label,
        master.depth,
        tight,
        N_SEEDS,
        loose,
        N_SEEDS,
        best,
        median,
    );
}

/// Perturbation-recovery: start at a hand-provided "correct" one-hot
/// configuration, perturb by Gaussian noise, run the fit. Reports how many
/// seeds return to near-zero loss. This is the paper's strongest claim —
/// EML recovers 100% from perturbed starts even at depth 5–6.
fn perturb_test(
    label: &str,
    master: &Master,
    op: &dyn Operator,
    atoms: &dyn Fn(f64) -> Vec<C>,
    spec: &TargetSpec,
    leaf_atoms: &[usize],
    noise: f64,
) {
    let xs = linspace(spec.domain.0, spec.domain.1, N_DATA);
    let targets: Vec<C> = xs.iter().map(|x| (spec.f)(*x)).collect();

    let truth = one_hot_params(master.n_leaves, master.atoms, leaf_atoms);
    let truth_loss = master.loss(&truth, op, &xs, &targets, atoms);

    let mut ok = 0;
    let mut final_losses = Vec::with_capacity(N_SEEDS);
    for seed in 0..N_SEEDS {
        let mut rng = Lcg::new(seed as u64 * 100 + 7);
        let init = perturb(&truth, &mut rng, noise);
        let loss = one_run(master, op, &xs, &targets, atoms, init);
        final_losses.push(loss);
        if loss < TIGHT_LOSS {
            ok += 1;
        }
    }
    final_losses.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  {:<12} d={} | perturb σ={:.1} | recover {:>2}/{:<2} | truth loss {:.2e} | median final {:.2e}",
        label,
        master.depth,
        noise,
        ok,
        N_SEEDS,
        truth_loss,
        final_losses[N_SEEDS / 2],
    );
}

fn main() {
    println!("=== Phase 3: master-formula gradient fit ===");
    println!(
        "budget: {} seeds × {} steps × LR={} × init σ={}\n",
        N_SEEDS, N_STEPS, LR, INIT_SCALE
    );

    let targets = [target_exp_x(), target_ln_x(), target_square()];

    // -- Part A: random-init sweep at depths 1..=3 --
    println!("Part A — random-init convergence rate");
    println!("{}", "-".repeat(78));

    for spec in &targets {
        println!(
            "target: {}   (domain {:.1}..{:.1})",
            spec.name, spec.domain.0, spec.domain.1
        );

        for depth in [1usize, 2, 3] {
            let master = Master::new(depth, 2);
            sweep("EML", &master, &Eml, &atoms_1_x, spec);
            sweep("EDL", &master, &Edl, &atoms_e_x, spec);
            sweep("PowSkew", &master, &PowSkew, &atoms_1_x, spec);
            sweep("PowExpSkew", &master, &PowExpSkew, &atoms_1_x, spec);
        }
        println!();
    }

    // -- Part B: perturbation recovery from known solution --
    //
    // For exp(x) with EML at depth 1: the true expression is eml(x, 1). In
    // our master tree with n_leaves=2 and atoms=[1, x], leaf 0 = x means
    // one-hot at atom index 1, and leaf 1 = 1 means one-hot at atom index 0.
    // So leaf_atoms = [1, 0].
    println!("Part B — perturbation recovery (paper's primary positive result)");
    println!("{}", "-".repeat(78));

    let exp_x = target_exp_x();
    let eml_d1 = Master::new(1, 2);
    println!(
        "target: {} — one-hot truth = eml(leaf[1]=x, leaf[0]=1)",
        exp_x.name
    );
    for noise in [0.5, 1.0, 2.0, 3.0] {
        perturb_test("EML", &eml_d1, &Eml, &atoms_1_x, &exp_x, &[1, 0], noise);
    }
    println!();

    // For PowExpSkew: exp(x) = pow-exp-skew(x, 0) where 0 is itself a
    // deep expression. At atoms={1, x} alone, 0 isn't directly a leaf. So
    // test a simpler target where we DO know a one-hot: e.g., target
    // `pow-exp-skew(1, 1) = 0`, target 0 everywhere. Trivial but shows
    // basin recovery.
    let target_zero = TargetSpec {
        name: "0 (constant)",
        f: |_| C::new(0.0, 0.0),
        domain: (0.5, 2.0),
    };
    let pes_d1 = Master::new(1, 2);
    println!(
        "target: {} — one-hot truth = pow-exp-skew(leaf[0]=1, leaf[0]=1)",
        target_zero.name
    );
    for noise in [0.5, 1.0, 2.0, 3.0] {
        perturb_test(
            "PowExpSkew",
            &pes_d1,
            &PowExpSkew,
            &atoms_1_x,
            &target_zero,
            &[0, 0],
            noise,
        );
    }
    println!();
}
