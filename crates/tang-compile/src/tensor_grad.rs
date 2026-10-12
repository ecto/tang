//! Reverse-mode autodiff over comprehensions.
//!
//! The gradient of a comprehension is a comprehension. For
//! `y[i] = Σ_k body(x₀[v₀(i,k)], …)` and an arg `a = x_j` read through a
//! pick view `v_j`, the contribution to `∂L/∂a[c]` sums `∂L/∂y[i] ·
//! ∂body/∂x_j` over every `(i,k)` with `v_j(i,k) = c`. Inverting a pick
//! view is renaming coordinates: the iteration coords `v_j` picks become
//! `a`'s out coords, the rest become reduce coords. Every other arg's view
//! is pushed through the same renaming, and the scalar body is
//! differentiated by `tang_expr`'s symbolic `diff`.
//!
//! Supported: `Sum` reductions and maps, width 1, pick views for the
//! differentiated arg (any affine view for the others). Not yet: `Max`
//! (needs select), tuple monoids, non-injective views such as diagonals.
//! Differentiate the *algorithm* graph, then optimize forward and backward
//! together.

use std::collections::HashMap;

use tang_expr::{ExprGraph, ExprId};

use super::{
    err, replay, Arg, Comp, Form, Monoid, MonoidKind, ShapeError, TensorGraph, TensorId, View,
};

impl TensorGraph {
    /// Gradients of `sum(y)` with respect to each of `wrt`, appended to
    /// this graph. A `wrt` that `y` does not depend on gets a zero map.
    pub fn backward(&mut self, y: TensorId, wrt: &[TensorId]) -> Result<Vec<TensorId>, ShapeError> {
        let live = self.live(&[y]);
        let mut grads: HashMap<usize, TensorId> = HashMap::new();
        let ones = {
            let shape = self.shape(y).to_vec();
            self.comp(&shape, &[], vec![], None, |_| vec![ExprId::ONE])?
        };
        grads.insert(y.n(), ones);

        // Reverse order over the original nodes; everything appended by
        // this pass sits after `y` and is never a dependency of it.
        for j in (0..=y.n()).rev() {
            if !live[j] {
                continue;
            }
            let Some(&dy) = grads.get(&j) else { continue };
            let Some(c) = self.comp_at(j) else { continue };
            let c = c.clone();
            for a in 0..c.args.len() {
                let g = self.arg_grad(&c, a, dy)?;
                let src = c.args[a].src.n();
                let acc = match grads.get(&src) {
                    Some(&prev) => self.map(&[prev, g], |v| v[0] + v[1])?,
                    None => g,
                };
                grads.insert(src, acc);
            }
        }

        wrt.iter()
            .map(|t| match grads.get(&t.n()) {
                Some(&g) => Ok(g),
                None => {
                    let shape = self.shape(*t).to_vec();
                    self.comp(&shape, &[], vec![], None, |_| vec![ExprId::ZERO])
                }
            })
            .collect()
    }

    /// `∂L/∂(arg a of c)` given `dy = ∂L/∂c`.
    fn arg_grad(&mut self, c: &Comp, a: usize, dy: TensorId) -> Result<TensorId, ShapeError> {
        if c.tuple != 0 || c.width() != 1 {
            return err("backward: tuple-valued comps are not differentiable yet");
        }
        match c.monoid.as_ref().map(Monoid::kind) {
            None | Some(MonoidKind::Sum) => {}
            Some(k) => {
                return err(format!(
                    "backward: {k:?} reductions are not differentiable yet"
                ))
            }
        }
        let old_rank = c.rank();
        let space = c.space();
        let arg = &c.args[a];
        let a_shape = self.shape(arg.src).to_vec();

        // Which old iteration coord each dim of `a` picks (None: broadcast).
        let picks: Vec<Option<usize>> = arg
            .view
            .forms
            .iter()
            .map(|f| {
                if f.coefs.iter().all(|&x| x == 0) && f.offset == 0 {
                    Ok(None)
                } else {
                    f.as_unit().map(Some).ok_or_else(|| {
                        ShapeError("backward: non-pick view on differentiated arg".into())
                    })
                }
            })
            .collect::<Result<_, _>>()?;
        let mut rename: Vec<Option<usize>> = vec![None; old_rank];
        for (d, p) in picks.iter().enumerate() {
            if let Some(coord) = p {
                if rename[*coord].replace(d).is_some() {
                    return err("backward: view reads one coord twice (diagonal)");
                }
            }
        }
        let free: Vec<usize> = (0..old_rank).filter(|&x| rename[x].is_none()).collect();
        for (i, &x) in free.iter().enumerate() {
            rename[x] = Some(a_shape.len() + i);
        }
        let new_rank = a_shape.len() + free.len();
        let red: Vec<usize> = free.iter().map(|&x| space[x]).collect();

        let rename_view = |v: &View| View {
            forms: v
                .forms
                .iter()
                .map(|f| {
                    let mut nf = Form::zero(new_rank);
                    nf.offset = f.offset;
                    for (x, &coef) in f.coefs.iter().enumerate() {
                        nf.coefs[rename[x].unwrap()] += coef;
                    }
                    nf
                })
                .collect(),
        };
        // dy is indexed by the old out coords
        let dy_view = View {
            forms: (0..c.points().len())
                .map(|x| Form::unit(new_rank, rename[x].unwrap()))
                .collect(),
        };
        let mut args = vec![Arg {
            src: dy,
            view: dy_view,
        }];
        args.extend(c.args.iter().map(|x| Arg {
            src: x.src,
            view: rename_view(&x.view),
        }));

        // body' = dy · ∂body/∂x_a
        let mut src = c.body.clone();
        let d = src.diff(c.body_outs[0], a as u16);
        let d = src.simplify(d);
        let mut body = ExprGraph::new();
        let vars: Vec<ExprId> = (0..=c.args.len()).map(|i| body.var(i as u16)).collect();
        let dexpr = replay(&src, &[d], &vars[1..], &mut body)[0];
        let out = body.mul(vars[0], dexpr);

        self.push_comp(Comp {
            out: a_shape,
            tuple: 0,
            red,
            args,
            body,
            body_outs: vec![out],
            monoid: (!free.is_empty()).then(Monoid::sum),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang::Scalar;

    /// Finite-difference check of `∂sum(y)/∂x` for input `x_idx`.
    fn check_grad(g: &TensorGraph, y: TensorId, inputs: &[Vec<f64>], x_idx: usize, grad: TensorId) {
        let got = g.eval(inputs, &[grad]).remove(0);
        let h = 1e-6;
        for i in 0..inputs[x_idx].len() {
            let mut plus = inputs.to_vec();
            plus[x_idx][i] += h;
            let mut minus = inputs.to_vec();
            minus[x_idx][i] -= h;
            let f = |inp: &[Vec<f64>]| g.eval(inp, &[y]).remove(0).iter().sum::<f64>();
            let fd = (f(&plus) - f(&minus)) / (2.0 * h);
            assert!(
                (got[i] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "input {x_idx}[{i}]: got {} vs fd {fd}",
                got[i]
            );
        }
    }

    fn ramp(n: usize, scale: f64, off: f64) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 7919) % 97) as f64 * scale + off)
            .collect()
    }

    #[test]
    fn matmul_grads() {
        let (m, k, n) = (3, 4, 2);
        let mut g = TensorGraph::new();
        let a = g.input(&[m, k]);
        let b = g.input(&[k, n]);
        let c = g.matmul(a, b).unwrap();
        let grads = g.backward(c, &[a, b]).unwrap();
        let inputs = [ramp(m * k, 0.1, -0.5), ramp(k * n, 0.2, 0.1)];
        check_grad(&g, c, &inputs, 0, grads[0]);
        check_grad(&g, c, &inputs, 1, grads[1]);
    }

    #[test]
    fn rmsnorm_grads_through_reduce_and_broadcast() {
        let (rows, dim) = (2, 5);
        let mut g = TensorGraph::new();
        let x = g.input(&[rows, dim]);
        let w = g.input(&[1, dim]);
        let sq = g.map(&[x], |v| v[0] * v[0]).unwrap();
        let sum = g.reduce(sq, 1, Monoid::sum()).unwrap();
        let inv = g
            .map(&[sum], |v| {
                ExprId::from_f64(1.0)
                    / (v[0] / ExprId::from_f64(dim as f64) + ExprId::from_f64(1e-6)).sqrt()
            })
            .unwrap();
        let inv = g.broadcast(inv, &[rows, dim]).unwrap();
        let wb = g.broadcast(w, &[rows, dim]).unwrap();
        let y = g.map(&[x, inv, wb], |v| v[0] * v[1] * v[2]).unwrap();
        let grads = g.backward(y, &[x, w]).unwrap();
        let inputs = [ramp(rows * dim, 0.3, -1.0), ramp(dim, 0.1, 0.5)];
        check_grad(&g, y, &inputs, 0, grads[0]);
        check_grad(&g, y, &inputs, 1, grads[1]);
    }

    #[test]
    fn transpose_and_shared_input() {
        // y = x · xᵀ : both args are the same input through different views
        let n = 3;
        let mut g = TensorGraph::new();
        let x = g.input(&[n, n]);
        let xt = g.transpose(x, &[1, 0]).unwrap();
        let y = g.matmul(x, xt).unwrap();
        let grads = g.backward(y, &[x]).unwrap();
        let inputs = [ramp(n * n, 0.2, -0.3)];
        check_grad(&g, y, &inputs, 0, grads[0]);
    }

    #[test]
    fn backward_graph_optimizes_like_any_other() {
        let (m, k, n) = (4, 3, 5);
        let mut g = TensorGraph::new();
        let a = g.input(&[m, k]);
        let b = g.input(&[k, n]);
        let c = g.matmul(a, b).unwrap();
        let act = g.map(&[c], |v| v[0] * v[0]).unwrap();
        let grads = g.backward(act, &[a, b]).unwrap();
        let inputs = [ramp(m * k, 0.1, -0.5), ramp(k * n, 0.2, 0.1)];
        let want: Vec<Vec<f64>> = g.eval(&inputs, &grads);
        let (fg, outs) = g.substitute_maps(&grads);
        let got = fg.eval(&inputs, &outs);
        for (w, g) in want.iter().zip(&got) {
            for (x, y) in w.iter().zip(g) {
                assert!((x - y).abs() < 1e-9, "{x} vs {y}");
            }
        }
        // ones · 2c (map) folds into the gradient reductions: only comps left
        // are the forward matmul and the two gradient reductions.
        assert_eq!(fg.count_comps(&outs), 3);
    }

    #[test]
    fn unsupported_monoid_is_an_error() {
        let mut g = TensorGraph::new();
        let x = g.input(&[2, 3]);
        let m = g.reduce(x, 1, Monoid::max()).unwrap();
        assert!(g.backward(m, &[x]).is_err());
    }
}
