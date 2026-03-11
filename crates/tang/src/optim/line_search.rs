/// Armijo backtracking line search.
///
/// Returns step size α such that f(x + α*d) <= f(x) + c * α * ∇f·d
pub fn armijo<F>(
    f: &F,
    x: &[f64],
    direction: &[f64],
    grad_dot_dir: f64,
    f_x: f64,
    c: f64,
    rho: f64,
) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let mut alpha = 1.0;
    let n = x.len();
    let mut x_new = alloc::vec![0.0; n];

    for _ in 0..50 {
        for i in 0..n {
            x_new[i] = x[i] + alpha * direction[i];
        }
        let f_new = f(&x_new);
        if f_new <= f_x + c * alpha * grad_dot_dir {
            return alpha;
        }
        alpha *= rho;
    }
    alpha
}
