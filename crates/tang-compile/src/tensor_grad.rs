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
//! Supported: `Sum`/`Max` reductions and maps, width 1, pick views for
//! the differentiated arg (any affine view for the others). `Max` routes
//! the gradient to every element equal to the max. Not yet: tuple
//! monoids, non-injective views such as diagonals.
//! Differentiate the *algorithm* graph, then optimize forward and backward
//! together.

use std::collections::HashMap;

use tang_expr::{ExprGraph, ExprId};

use super::{
    err, replay, Arg, Comp, Dim, Form, Monoid, MonoidKind, Op, ShapeError, TensorGraph, TensorId,
    View,
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
            if let Op::Reshape(src) = self.nodes[j].op {
                let shape = self.shape(src).to_vec();
                let g = self.reshape(dy, &shape)?;
                let acc = match grads.get(&src.n()) {
                    Some(&prev) => self.map(&[prev, g], |v| v[0] + v[1])?,
                    None => g,
                };
                grads.insert(src.n(), acc);
                continue;
            }
            let Some(c) = self.comp_at(j) else { continue };
            let c = c.clone();
            for a in 0..c.args.len() {
                let g = self.arg_grad(TensorId(j as u32), &c, a, dy)?;
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
    fn arg_grad(
        &mut self,
        self_id: TensorId,
        c: &Comp,
        a: usize,
        dy: TensorId,
    ) -> Result<TensorId, ShapeError> {
        if c.tuple != 0 || c.width() != 1 {
            return err("backward: tuple-valued comps are not differentiable yet");
        }
        let kind = c.monoid.as_ref().map(Monoid::kind);
        match kind {
            None | Some(MonoidKind::Sum) | Some(MonoidKind::Max) => {}
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

        // A dim of `a` that reads one iteration coord outright is inverted
        // by renaming that coord. Any other form — broadcast, offset,
        // stride, diagonal, sum of coords — becomes a *scatter dim*: its
        // coordinate stays an out dim and the body is gated by
        // `[coord == form(...)]`, with the form's coords reduced over.
        let mut rename: Vec<Option<usize>> = vec![None; old_rank];
        let mut scatter: Vec<usize> = Vec::new();
        for (d, f) in arg.view.forms.iter().enumerate() {
            let zero = f.coefs.iter().all(|&x| x == 0) && f.offset == 0;
            match f.as_unit() {
                Some(coord) if rename[coord].is_none() => rename[coord] = Some(d),
                // size-1 dim read at 0: nothing to gate
                _ if zero && a_shape[d] == Dim::Fixed(1) => {}
                _ => scatter.push(d),
            }
        }
        let free: Vec<usize> = (0..old_rank).filter(|&x| rename[x].is_none()).collect();
        for (i, &x) in free.iter().enumerate() {
            rename[x] = Some(a_shape.len() + i);
        }
        let new_rank = a_shape.len() + free.len();
        let red: Vec<Dim> = free.iter().map(|&x| space[x]).collect();

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
        let mut out = body.mul(vars[0], dexpr);

        // scatter dims: gate by [iota_d == renamed form]
        if !scatter.is_empty() {
            let mut full: Vec<Dim> = a_shape.clone();
            full.extend(red.iter().copied());
            let mut coord_var = |k: usize, body: &mut ExprGraph, args: &mut Vec<Arg>| {
                let io = self.iota(&full, k);
                args.push(Arg {
                    src: io,
                    view: View::identity(new_rank, new_rank),
                });
                body.var((args.len() - 1) as u16)
            };
            for &d in &scatter {
                let f = rename_view(&View {
                    forms: vec![arg.view.forms[d].clone()],
                })
                .forms
                .remove(0);
                let mut target = body.lit(f.offset as f64);
                for (k, &coef) in f.coefs.iter().enumerate() {
                    if coef != 0 {
                        let v = coord_var(k, &mut body, &mut args);
                        let cl = body.lit(coef as f64);
                        let t = body.mul(cl, v);
                        target = body.add(target, t);
                    }
                }
                let here = coord_var(d, &mut body, &mut args);
                let (nt, nh) = (body.neg(target), body.neg(here));
                let above = body.add(here, nt); // here - target
                let below = body.add(target, nh); // target - here
                let inner = body.select(below, ExprId::ZERO, ExprId::ONE);
                let hit = body.select(above, ExprId::ZERO, inner);
                out = body.mul(out, hit);
            }
        }
        if kind == Some(MonoidKind::Max) {
            // Gate to the argmax: read the forward max like dy and keep
            // the term only where body == max.
            let m = body.var(args.len() as u16);
            args.push(Arg {
                src: self_id,
                view: args[0].view.clone(),
            });
            let val = replay(&src, &[c.body_outs[0]], &vars[1..], &mut body)[0];
            let (nm, nv) = (body.neg(m), body.neg(val));
            let below = body.add(val, nm); // val - m
            let above = body.add(m, nv); // m - val
            let inner = body.select(above, ExprId::ZERO, ExprId::ONE);
            let hit = body.select(below, ExprId::ZERO, inner);
            out = body.mul(out, hit);
        }

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
    fn grads_through_reshape_and_strided_views() {
        // y[a, b] = sum_k r[a, 2k+1] * w[k, b] with r = reshape(x, [3, 4])
        let mut g = TensorGraph::new();
        let x = g.input(&[2, 6]);
        let w = g.input(&[2, 5]);
        let r = g.reshape(x, &[3, 4]).unwrap();
        let odd = g
            .comp(
                &[3, 2],
                &[],
                vec![Arg {
                    src: r,
                    view: View {
                        forms: vec![
                            Form::unit(2, 0),
                            Form {
                                coefs: vec![0, 2],
                                offset: 1,
                            },
                        ],
                    },
                }],
                None,
                |v| vec![v[0]],
            )
            .unwrap();
        let y = g.matmul(odd, w).unwrap();
        let grads = g.backward(y, &[x, w]).unwrap();
        let inputs = [ramp(12, 0.3, -1.0), ramp(10, 0.2, 0.1)];
        check_grad(&g, y, &inputs, 0, grads[0]);
        check_grad(&g, y, &inputs, 1, grads[1]);
    }

    #[test]
    fn max_routes_gradient_to_argmax() {
        let mut g = TensorGraph::new();
        let x = g.input(&[2, 3]);
        let m = g.reduce(x, 1, Monoid::max()).unwrap();
        let grads = g.backward(m, &[x]).unwrap();
        let got = g
            .eval(&[vec![1.0, 5.0, 2.0, 7.0, 0.0, 3.0]], &grads)
            .remove(0);
        assert_eq!(got, vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn attention_grads() {
        let (n, d) = (4, 3);
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, d]);
        let kt = g.transpose(k, &[1, 0]).unwrap();
        let s = g.matmul(q, kt).unwrap();
        let m = g.reduce(s, 1, Monoid::max()).unwrap();
        let mb = g.broadcast(m, &[n, n]).unwrap();
        let p = g.map(&[s, mb], |x| (x[0] - x[1]).exp()).unwrap();
        let l = g.reduce(p, 1, Monoid::sum()).unwrap();
        let o = g.matmul(p, v).unwrap();
        let lb = g.broadcast(l, &[n, d]).unwrap();
        let y = g.map(&[o, lb], |x| x[0] / x[1]).unwrap();
        let grads = g.backward(y, &[q, k, v]).unwrap();
        let inputs = [
            ramp(n * d, 0.3, -0.5),
            ramp(n * d, 0.2, 0.1),
            ramp(n * d, 0.4, -1.0),
        ];
        for (i, &gr) in grads.iter().enumerate() {
            check_grad(&g, y, &inputs, i, gr);
        }
    }

    #[test]
    fn tuple_monoid_is_an_error() {
        let mut g = TensorGraph::new();
        let x = g.input(&[2, 3]);
        let m = g.reduce(x, 1, Monoid::max()).unwrap();
        let mb = g.broadcast(m, &[2, 3]).unwrap();
        let p = g.map(&[x, mb], |v| (v[0] - v[1]).exp()).unwrap();
        let l = g.reduce(p, 1, Monoid::sum()).unwrap();
        let y = g.map(&[l], |v| v[0] * v[0]).unwrap();
        let (g, outs) = g.substitute_maps(&[y]);
        let (mut g, outs) = g.stabilize_softmax(&outs);
        assert!(g
            .reduction_kinds(&outs)
            .contains(&MonoidKind::OnlineSoftmax));
        assert!(g.backward(outs[0], &[]).is_err());
    }
}
