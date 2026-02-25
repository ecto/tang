//! Reverse-mode automatic differentiation.
//!
//! Tape-based AD for efficient gradient computation when outputs are
//! scalar and inputs are many (the ML/physics optimization case).

#![no_std]

extern crate alloc;

mod tape;
mod var;

pub use tape::Tape;
pub use var::Var;

use tang_la::{DVec, DMat};
use alloc::vec::Vec;
/// Compute gradient of scalar-valued function via reverse-mode AD.
///
/// Returns gradient vector where `grad[i]` = ∂f/∂x_i.
pub fn grad<F>(f: F, x: &[f64]) -> DVec<f64>
where
    F: Fn(&[Var]) -> Var,
{
    let tape = Tape::new();
    let vars: Vec<Var> = x.iter().map(|&v| tape.var(v)).collect();
    let indices: Vec<usize> = vars.iter().map(|v| v.index).collect();
    let result = f(&vars);
    let all_grads = result.backward();
    DVec::from_fn(x.len(), |i| all_grads[indices[i]])
}

/// Compute Jacobian via forward-mode (Dual numbers).
///
/// Efficient when n (input dim) is small relative to m (output dim).
pub fn jacobian_fwd<F>(f: F, x: &[f64]) -> DMat<f64>
where
    F: Fn(&[tang::Dual<f64>]) -> Vec<tang::Dual<f64>>,
{
    let n = x.len();
    let mut columns = Vec::new();
    for i in 0..n {
        let inputs: Vec<tang::Dual<f64>> = x
            .iter()
            .enumerate()
            .map(|(j, &v)| {
                if i == j {
                    tang::Dual::var(v)
                } else {
                    tang::Dual::constant(v)
                }
            })
            .collect();
        let outputs = f(&inputs);
        columns.push(outputs.iter().map(|d| d.dual).collect::<Vec<_>>());
    }
    let m = columns.first().map_or(0, |c| c.len());
    DMat::from_fn(m, n, |i, j| columns[j][i])
}

/// Compute Hessian of scalar-valued function via forward-over-forward (Dual<Dual<f64>>).
pub fn hessian<F>(f: F, x: &[f64]) -> DMat<f64>
where
    F: Fn(&[tang::Dual<tang::Dual<f64>>]) -> tang::Dual<tang::Dual<f64>>,
{
    let n = x.len();
    DMat::from_fn(n, n, |i, j| {
        // Seed: x_k = Dual(Dual(x_k, δ_kj), Dual(δ_ki, 0))
        let inputs: Vec<tang::Dual<tang::Dual<f64>>> = (0..n)
            .map(|k| {
                let real = tang::Dual::new(x[k], if k == j { 1.0 } else { 0.0 });
                let dual = tang::Dual::new(if k == i { 1.0 } else { 0.0 }, 0.0);
                tang::Dual::new(real, dual)
            })
            .collect();
        f(&inputs).dual.dual
    })
}

/// Vector-Jacobian product via reverse-mode: v^T J
pub fn vjp<F>(f: F, x: &[f64], v: &[f64]) -> DVec<f64>
where
    F: Fn(&[Var]) -> Vec<Var>,
{
    let tape = Tape::new();
    let vars: Vec<Var> = x.iter().map(|&val| tape.var(val)).collect();
    let indices: Vec<usize> = vars.iter().map(|var| var.index).collect();
    let outputs = f(&vars);

    // Accumulate v^T J by taking weighted sum of output gradients
    let n = x.len();
    let mut result = DVec::zeros(n);
    for (k, out) in outputs.iter().enumerate() {
        let grads = out.backward();
        let vk = v[k];
        for i in 0..n {
            result[i] = result[i] + vk * grads[indices[i]];
        }
    }
    result
}

/// Jacobian-vector product via forward-mode: J v
pub fn jvp<F>(f: F, x: &[f64], v: &[f64]) -> DVec<f64>
where
    F: Fn(&[tang::Dual<f64>]) -> Vec<tang::Dual<f64>>,
{
    let inputs: Vec<tang::Dual<f64>> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| tang::Dual::new(xi, vi))
        .collect();
    let outputs = f(&inputs);
    DVec::from_fn(outputs.len(), |i| outputs[i].dual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang::Scalar;

    #[test]
    fn grad_simple() {
        // f(x, y) = x*y, grad = (y, x)
        let g = grad(|x| &x[0] * &x[1], &[3.0, 5.0]);
        assert!((g[0] - 5.0).abs() < 1e-10);
        assert!((g[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn grad_quadratic() {
        // f(x) = x^2, grad = 2x at x=4 -> 8
        let g = grad(|x| &x[0] * &x[0], &[4.0]);
        assert!((g[0] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn jacobian_fwd_linear() {
        // f(x, y) = (x + y, x - y)
        let j = jacobian_fwd(|x| {
            alloc::vec![x[0] + x[1], x[0] - x[1]]
        }, &[1.0, 2.0]);
        assert_eq!(j.get(0, 0), 1.0);
        assert_eq!(j.get(0, 1), 1.0);
        assert_eq!(j.get(1, 0), 1.0);
        assert_eq!(j.get(1, 1), -1.0);
    }

    #[test]
    fn hessian_quadratic() {
        // f(x, y) = x^2 + 2*x*y + y^2
        // H = [[2, 2], [2, 2]]
        let h = hessian(|x| {
            x[0] * x[0] + x[0] * x[1] * tang::Dual::from_f64(2.0) + x[1] * x[1]
        }, &[1.0, 1.0]);
        assert!((h.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((h.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((h.get(1, 0) - 2.0).abs() < 1e-10);
        assert!((h.get(1, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn jvp_simple() {
        // f(x, y) = (x*y, x+y), J = [[y, x], [1, 1]]
        // Jv at (3,5), v=(1,0): (5, 1)
        let result = jvp(|x| {
            alloc::vec![x[0] * x[1], x[0] + x[1]]
        }, &[3.0, 5.0], &[1.0, 0.0]);
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
    }
}
