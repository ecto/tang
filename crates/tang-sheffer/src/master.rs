//! Master formula: a fixed-shape balanced binary tree over a chosen
//! operator, with softmax-weighted leaves over a small atom set. The point
//! is to cast symbolic regression as a smooth fit: every leaf is a convex
//! combination of atoms, so parameters are continuous and gradients flow.
//!
//! At convergence the softmax typically collapses to a one-hot vector and
//! the tree becomes a pure Expr. Phase 3 uses this to compare the loss
//! landscape of different Sheffer-family operators: which ones have basins
//! that random-init gradient descent can reach?
//!
//! The gradient is computed by forward-mode finite differences. This is
//! simpler than threading `Dual<Complex<f64>>` through the operator trait
//! and adequate for the parameter counts we use here (≤ 64 params).

use crate::operator::Operator;
use crate::C;

/// Fixed balanced binary tree of operator applications. `depth = d` means
/// `2^d` leaves at the bottom and `2^d − 1` internal nodes.
#[derive(Debug, Clone, Copy)]
pub struct Master {
    pub depth: usize,
    pub atoms: usize,
    pub n_leaves: usize,
    pub n_params: usize,
}

impl Master {
    pub fn new(depth: usize, atoms: usize) -> Self {
        let n_leaves = 1 << depth;
        Self {
            depth,
            atoms,
            n_leaves,
            n_params: n_leaves * atoms,
        }
    }

    /// Evaluate the master tree at one data point. The caller supplies the
    /// atom values at that data point (typically `{1, x}` for a
    /// single-variable fit) and the operator; this routine computes
    /// softmax-weighted leaf values then reduces bottom-up.
    pub fn eval(&self, params: &[f64], atom_values: &[C], op: &dyn Operator) -> C {
        debug_assert_eq!(params.len(), self.n_params);
        debug_assert_eq!(atom_values.len(), self.atoms);

        let mut layer: Vec<C> = (0..self.n_leaves)
            .map(|leaf_idx| {
                let start = leaf_idx * self.atoms;
                let logits = &params[start..start + self.atoms];
                softmax_weighted_sum(logits, atom_values)
            })
            .collect();

        while layer.len() > 1 {
            let mut next = Vec::with_capacity(layer.len() / 2);
            for pair in layer.chunks_exact(2) {
                next.push(op.eval(pair[0], pair[1]));
            }
            layer = next;
        }
        layer[0]
    }

    /// Mean-squared-error loss across all data points. Complex residuals
    /// use their squared norm; if any evaluation produces NaN the point
    /// contributes a large penalty instead of polluting the gradient.
    pub fn loss(
        &self,
        params: &[f64],
        op: &dyn Operator,
        xs: &[f64],
        targets: &[C],
        atoms: &dyn Fn(f64) -> Vec<C>,
    ) -> f64 {
        let mut total = 0.0;
        for (x, t) in xs.iter().zip(targets) {
            let atom_values = atoms(*x);
            let pred = self.eval(params, &atom_values, op);
            if pred.re.is_nan() || pred.im.is_nan() || !pred.re.is_finite() || !pred.im.is_finite()
            {
                total += 1e6;
                continue;
            }
            total += (pred - t).norm_sqr();
        }
        total / xs.len() as f64
    }

    /// Central-difference gradient of `loss` at `params`. Returns a vector
    /// parallel to `params`. O(n_params) loss evaluations per call.
    pub fn grad_fd(
        &self,
        params: &[f64],
        op: &dyn Operator,
        xs: &[f64],
        targets: &[C],
        atoms: &dyn Fn(f64) -> Vec<C>,
        eps: f64,
    ) -> Vec<f64> {
        let mut g = vec![0.0; params.len()];
        let mut pp = params.to_vec();
        for i in 0..params.len() {
            let orig = params[i];
            pp[i] = orig + eps;
            let up = self.loss(&pp, op, xs, targets, atoms);
            pp[i] = orig - eps;
            let dn = self.loss(&pp, op, xs, targets, atoms);
            pp[i] = orig;
            g[i] = (up - dn) / (2.0 * eps);
        }
        g
    }
}

fn softmax_weighted_sum(logits: &[f64], atom_values: &[C]) -> C {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    let mut acc = C::new(0.0, 0.0);
    for (e, a) in exps.iter().zip(atom_values) {
        acc += C::new(e / z, 0.0) * a;
    }
    acc
}

/// Adam optimizer state. `m` and `v` are first and second moment estimates
/// parallel to the parameter vector.
#[derive(Debug, Clone)]
pub struct Adam {
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub t: usize,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl Adam {
    pub fn new(n_params: usize) -> Self {
        Self {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    pub fn step(&mut self, params: &mut [f64], grad: &[f64], lr: f64) {
        self.t += 1;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let bias1 = 1.0 - b1.powi(self.t as i32);
        let bias2 = 1.0 - b2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bias1;
            let v_hat = self.v[i] / bias2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

/// Run a fit: starting from `params`, do `n_steps` of Adam with the given
/// learning rate. Returns the final loss and the final parameter vector.
pub struct FitResult {
    pub final_loss: f64,
    pub losses: Vec<f64>,
    pub params: Vec<f64>,
}

pub fn fit(
    master: &Master,
    op: &dyn Operator,
    xs: &[f64],
    targets: &[C],
    atoms: &dyn Fn(f64) -> Vec<C>,
    init_params: Vec<f64>,
    n_steps: usize,
    lr: f64,
) -> FitResult {
    let mut params = init_params;
    let mut adam = Adam::new(params.len());
    let mut losses = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let loss = master.loss(&params, op, xs, targets, atoms);
        losses.push(loss);
        if !loss.is_finite() || loss < 1e-20 {
            break;
        }
        let g = master.grad_fd(&params, op, xs, targets, atoms, 1e-5);
        adam.step(&mut params, &g, lr);
    }
    let final_loss = master.loss(&params, op, xs, targets, atoms);
    FitResult {
        final_loss,
        losses,
        params,
    }
}

/// Simple deterministic LCG for seeding random initialization without pulling
/// in a `rand` dependency. Matches the style of other tang examples.
pub struct Lcg(pub u64);
impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard-normal-ish via Box-Muller on two uniforms.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Eml;

    #[test]
    fn master_depth_0_is_single_leaf() {
        let m = Master::new(0, 2);
        assert_eq!(m.n_leaves, 1);
        assert_eq!(m.n_params, 2);
    }

    #[test]
    fn master_eval_picks_pure_x() {
        // With logits strongly preferring atom index 1, every leaf ≈ x.
        // Then eml(x, x) = exp(x) - ln(x), etc. Just sanity-check no-panic.
        let m = Master::new(2, 2);
        let _params: Vec<f64> = (0..m.n_params)
            .flat_map(|i| if i % 2 == 0 { vec![-10.0] } else { vec![10.0] })
            .collect();
        // Wait — the above uses flat_map wrong; fix.
        let params: Vec<f64> = (0..m.n_params)
            .map(|i| if i % 2 == 0 { -10.0 } else { 10.0 })
            .collect();
        let atoms = vec![C::new(1.0, 0.0), C::new(0.5, 0.0)];
        let v = m.eval(&params, &atoms, &Eml);
        assert!(v.re.is_finite());
    }

    #[test]
    fn adam_reduces_quadratic_loss() {
        // Minimize ||p||² via FD through a trivial "master" of depth 0.
        // Not using Master directly; just spot-check Adam.
        let mut params = vec![1.0, -0.5, 2.0];
        let mut adam = Adam::new(params.len());
        for _ in 0..500 {
            let grad: Vec<f64> = params.iter().map(|p| 2.0 * p).collect();
            adam.step(&mut params, &grad, 0.05);
        }
        let norm_sq: f64 = params.iter().map(|p| p * p).sum();
        assert!(norm_sq < 1e-4, "Adam failed to minimize: {}", norm_sq);
    }
}
