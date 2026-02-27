//! Physics-Informed Neural Network (PINN) helpers.
//!
//! Provides utilities for computing PDE residuals using `Dual<Dual<f64>>`
//! through generic neural network layers, enabling automatic computation
//! of first and second derivatives needed for physics constraints.

use alloc::vec::Vec;
use tang::Dual;
use tang_tensor::{Shape, Tensor};

/// Compute the gradient of a scalar output w.r.t. each input component.
///
/// Given a function `f: R^n -> R`, computes `[df/dx_1, ..., df/dx_n]`
/// using forward-mode AD via `Dual<f64>`.
///
/// # Arguments
/// * `f` - Function mapping input tensor to scalar output
/// * `x` - Point at which to evaluate the gradient (1-D, length n)
pub fn grad<F>(f: F, x: &Tensor<f64>) -> Tensor<f64>
where
    F: Fn(&Tensor<Dual<f64>>) -> Dual<f64>,
{
    assert_eq!(x.ndim(), 1);
    let n = x.numel();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let dual_x = Tensor::from_fn(x.shape().clone(), |idx| {
            let val = x.get(idx);
            if idx[0] == i {
                Dual::var(val)
            } else {
                Dual::constant(val)
            }
        });

        let out = f(&dual_x);
        result.push(out.dual); // df/dx_i
    }

    Tensor::from_slice(&result)
}

/// Compute the Laplacian (sum of second derivatives) of a scalar function.
///
/// Given `f: R^n -> R`, computes `sum_i d²f/dx_i²` using nested dual numbers.
///
/// # Arguments
/// * `f` - Function mapping input tensor to scalar output
/// * `x` - Point at which to evaluate (1-D, length n)
pub fn laplacian<F>(f: F, x: &Tensor<f64>) -> f64
where
    F: Fn(&Tensor<Dual<Dual<f64>>>) -> Dual<Dual<f64>>,
{
    assert_eq!(x.ndim(), 1);
    let n = x.numel();
    let mut sum = 0.0;

    for i in 0..n {
        let dd_x = Tensor::from_fn(x.shape().clone(), |idx| {
            let val = x.get(idx);
            if idx[0] == i {
                Dual::new(Dual::var(val), Dual::constant(1.0))
            } else {
                Dual::constant(Dual::constant(val))
            }
        });

        let out = f(&dd_x);
        sum += out.dual.dual;
    }

    sum
}

/// Compute the full Hessian matrix of a scalar function.
///
/// Given `f: R^n -> R`, computes `H[i,j] = d²f/(dx_i dx_j)`.
///
/// # Arguments
/// * `f` - Function mapping input tensor to scalar output
/// * `x` - Point at which to evaluate (1-D, length n)
pub fn hessian<F>(f: &F, x: &Tensor<f64>) -> Tensor<f64>
where
    F: Fn(&Tensor<Dual<Dual<f64>>>) -> Dual<Dual<f64>>,
{
    assert_eq!(x.ndim(), 1);
    let n = x.numel();

    Tensor::from_fn(Shape::from_slice(&[n, n]), |idx| {
        let i = idx[0];
        let j = idx[1];

        let dd_x = Tensor::from_fn(x.shape().clone(), |kidx| {
            let val = x.get(kidx);
            let k = kidx[0];
            let inner = if k == i {
                Dual::var(val)
            } else {
                Dual::constant(val)
            };
            if k == j {
                Dual::new(inner, Dual::constant(1.0))
            } else {
                Dual::new(inner, Dual::constant(0.0))
            }
        });

        let out = f(&dd_x);
        out.dual.dual
    })
}

/// Generate collocation points on a uniform grid over a hypercube [lo, hi]^dim.
///
/// # Arguments
/// * `dim` - Number of spatial dimensions
/// * `n_per_dim` - Number of points per dimension
/// * `lo` - Lower bound for each dimension
/// * `hi` - Upper bound for each dimension
pub fn collocation_grid(dim: usize, n_per_dim: usize, lo: f64, hi: f64) -> Tensor<f64> {
    let total = n_per_dim.pow(dim as u32);
    Tensor::from_fn(Shape::from_slice(&[total, dim]), |idx| {
        let point_idx = idx[0];
        let d = idx[1];
        let stride: usize = n_per_dim.pow(d as u32);
        let i = (point_idx / stride) % n_per_dim;
        if n_per_dim == 1 {
            (lo + hi) * 0.5
        } else {
            lo + (hi - lo) * i as f64 / (n_per_dim - 1) as f64
        }
    })
}

/// Generate random collocation points in a hypercube [lo, hi]^dim using LCG PRNG.
///
/// # Arguments
/// * `n_points` - Number of points
/// * `dim` - Number of spatial dimensions
/// * `lo` - Lower bound
/// * `hi` - Upper bound
/// * `seed` - PRNG seed
pub fn collocation_random(
    n_points: usize,
    dim: usize,
    lo: f64,
    hi: f64,
    seed: u64,
) -> Tensor<f64> {
    let total = n_points * dim;
    let mut data = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        data.push(lo + (hi - lo) * u);
    }
    Tensor::new(data, Shape::from_slice(&[n_points, dim]))
}

/// Compute PDE residual for a given function and PDE operator.
///
/// Uses `Dual<Dual<f64>>` to compute u, grad_u, and laplacian_u in one set of passes.
///
/// # Arguments
/// * `net` - Network function: takes `Dual<Dual<f64>>` input, returns scalar
/// * `pde_op` - PDE operator: `(u, grad_u, laplacian_u, x) -> residual`
/// * `points` - Collocation points [N, dim]
///
/// Returns residual at each collocation point [N].
pub fn pde_residual<Net, Op>(net: &Net, pde_op: &Op, points: &Tensor<f64>) -> Tensor<f64>
where
    Net: Fn(&Tensor<Dual<Dual<f64>>>) -> Dual<Dual<f64>>,
    Op: Fn(f64, &Tensor<f64>, f64, &Tensor<f64>) -> f64,
{
    let n = points.shape()[0];
    let dim = points.shape()[1];

    let mut residuals = Vec::with_capacity(n);

    for i in 0..n {
        let x = Tensor::from_fn(Shape::from_slice(&[dim]), |jdx| points.get(&[i, jdx[0]]));

        // Compute u(x) with constant inputs
        let u_val = {
            let cx: Tensor<Dual<Dual<f64>>> = Tensor::from_fn(x.shape().clone(), |kidx| {
                Dual::constant(Dual::constant(x.get(kidx)))
            });
            let out = net(&cx);
            out.real.real
        };

        // Compute gradient and diagonal Hessian entries using Dual<Dual<f64>>
        let mut grad_data = Vec::with_capacity(dim);
        let mut lap = 0.0;

        for d in 0..dim {
            // Inner dual tracks d/dx_d, outer dual also tracks d/dx_d
            let dd_x = Tensor::from_fn(x.shape().clone(), |kidx| {
                let val = x.get(kidx);
                if kidx[0] == d {
                    // Both inner and outer dual = 1 for this direction
                    Dual::new(Dual::var(val), Dual::constant(1.0))
                } else {
                    Dual::constant(Dual::constant(val))
                }
            });

            let out = net(&dd_x);
            grad_data.push(out.real.dual); // du/dx_d from inner dual
            lap += out.dual.dual; // d²u/dx_d² from combined duals
        }

        let grad_u = Tensor::from_slice(&grad_data);
        residuals.push(pde_op(u_val, &grad_u, lap, &x));
    }

    Tensor::from_slice(&residuals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_quadratic() {
        // f(x, y) = x^2 + 3*y^2
        // grad = [2x, 6y]
        let x = Tensor::from_slice(&[2.0, 3.0]);
        let g = grad(
            |input| {
                let x = input.get(&[0]);
                let y = input.get(&[1]);
                let three = Dual::constant(3.0);
                x * x + three * y * y
            },
            &x,
        );
        assert!((g.get(&[0]) - 4.0).abs() < 1e-10); // 2*2 = 4
        assert!((g.get(&[1]) - 18.0).abs() < 1e-10); // 6*3 = 18
    }

    #[test]
    fn test_laplacian_quadratic() {
        // f(x, y) = x^2 + 3*y^2
        // Laplacian = 2 + 6 = 8
        let x = Tensor::from_slice(&[2.0, 3.0]);
        let lap = laplacian(
            |input| {
                let x = input.get(&[0]);
                let y = input.get(&[1]);
                let three = Dual::constant(Dual::constant(3.0));
                x * x + three * y * y
            },
            &x,
        );
        assert!((lap - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_hessian_quadratic() {
        // f(x, y) = x^2 + x*y + 3*y^2
        // H = [[2, 1], [1, 6]]
        let x = Tensor::from_slice(&[1.0, 1.0]);
        let h = hessian(
            &|input: &Tensor<Dual<Dual<f64>>>| {
                let x = input.get(&[0]);
                let y = input.get(&[1]);
                let three = Dual::constant(Dual::constant(3.0));
                x * x + x * y + three * y * y
            },
            &x,
        );
        assert!((h.get(&[0, 0]) - 2.0).abs() < 1e-10);
        assert!((h.get(&[0, 1]) - 1.0).abs() < 1e-10);
        assert!((h.get(&[1, 0]) - 1.0).abs() < 1e-10);
        assert!((h.get(&[1, 1]) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_collocation_grid() {
        let points = collocation_grid(2, 3, 0.0, 1.0);
        assert_eq!(points.shape().dims(), &[9, 2]);
        assert!((points.get(&[0, 0]) - 0.0).abs() < 1e-10);
        assert!((points.get(&[0, 1]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_collocation_random_bounds() {
        let points = collocation_random(100, 3, -1.0, 1.0, 42);
        assert_eq!(points.shape().dims(), &[100, 3]);
        for &v in points.data() {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_pde_residual_laplace() {
        // Laplace equation: nabla^2 u = 0
        // Solution: u(x, y) = x^2 - y^2 (harmonic function)
        // Residual should be 0 everywhere

        let net = |input: &Tensor<Dual<Dual<f64>>>| {
            let x = input.get(&[0]);
            let y = input.get(&[1]);
            x * x - y * y
        };

        let pde_op = |_u: f64, _grad_u: &Tensor<f64>, lap_u: f64, _x: &Tensor<f64>| lap_u;

        let points = collocation_grid(2, 5, -1.0, 1.0);
        let residual = pde_residual(&net, &pde_op, &points);

        for &r in residual.data() {
            assert!(
                r.abs() < 1e-10,
                "Laplace residual should be ~0, got {}",
                r
            );
        }
    }
}
