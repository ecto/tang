//! Enumerative verification: given an operator and a leaf pool, generate all
//! expression trees up to a bound, dedup by numeric value, then match against
//! targets.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::expr::{Expr, Leaf, LeafSource};
use crate::operator::Operator;
use crate::targets::Target;
use crate::C;

/// Bit-quantized fingerprint of a complex value. Two values with the same
/// ValueKey are considered "the same" for dedup purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueKey(u64, u64);

/// Quantize an f64 for hashing:
///   - NaN → sentinel u64::MAX
///   - ±inf → its own bit pattern (preserves sign of inf)
///   - finite → clear low 10 mantissa bits (≈3 decimal digits of slack)
///
/// The slack absorbs accumulated roundoff on short evaluation chains but is
/// tight enough to keep genuinely distinct values distinct. Two calls with
/// the same `x` always produce the same key.
fn quantize_f64(x: f64) -> u64 {
    if x.is_nan() {
        return u64::MAX;
    }
    // Normalize -0.0 to +0.0 so they hash the same.
    let x = if x == 0.0 { 0.0 } else { x };
    let bits = x.to_bits();
    if x.is_infinite() {
        return bits;
    }
    bits & !0x3FF
}

/// Fingerprint a complex value.
pub fn quantize(c: C) -> ValueKey {
    ValueKey(quantize_f64(c.re), quantize_f64(c.im))
}

/// A concrete (Expr, value) pair recorded during enumeration. `gen` tags
/// which bootstrap iteration this entry was first added in, so incremental
/// enumeration can skip pairs where both sides already existed in a prior
/// iteration (those pairs were explored last time already).
#[derive(Debug, Clone)]
pub struct Found {
    pub expr: Rc<Expr>,
    pub value: C,
    pub gen: u32,
}

/// Per-iteration progress snapshot from `bootstrap_with_progress`.
#[derive(Debug)]
pub struct BootstrapProgress<'a> {
    pub iteration: usize,
    /// Current leaf pool size (includes leaves just added this iteration).
    pub leaf_count: usize,
    /// Number of dedup'd distinct values found in this iteration's catalogue.
    pub catalogue_size: usize,
    /// Target names first discovered this iteration.
    pub new_targets: &'a [String],
    /// Total targets found so far across all iterations.
    pub total_found: usize,
    pub iter_elapsed: std::time::Duration,
    pub total_elapsed: std::time::Duration,
}

/// A successful target match: a target value has been hit by some discovered
/// expression in the catalogue.
#[derive(Debug, Clone)]
pub struct Discovery {
    pub target_name: String,
    pub expression: Rc<Expr>,
    pub value: C,
    pub size: usize,
    /// Size in the base alphabet after expanding any Derived leaves.
    pub expanded_size: usize,
    pub residual: f64,
    /// Which bootstrap iteration this was first found at (0 for single-pass).
    pub iteration: usize,
}

/// Verifier owns the leaf pool and tolerance; methods do the enumeration and
/// target matching without mutating state, so a single verifier can drive
/// multiple searches.
pub struct Verifier {
    pub leaves: Vec<Leaf>,
    pub tol: f64,
}

impl Verifier {
    pub fn new(leaves: Vec<Leaf>) -> Self {
        Self { leaves, tol: 1e-10 }
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Enumerate all expressions up to `max_ops` internal operator
    /// applications. Returns a per-level catalogue: result[k] is the set of
    /// dedup'd (expr, value) pairs with exactly k ops.
    ///
    /// Dedup is global across all sizes: a value first discovered at k=2 is
    /// not re-added at k=3, so each distinct value appears exactly once in
    /// the smallest level that reaches it.
    pub fn enumerate(&self, op: &dyn Operator, max_ops: usize) -> Vec<Vec<Found>> {
        let mut by_ops: Vec<Vec<Found>> = vec![Vec::new(); max_ops + 1];
        let mut seen: HashSet<ValueKey> = HashSet::new();

        // Base level: leaves at ops=0.
        for (i, leaf) in self.leaves.iter().enumerate() {
            if leaf.value.re.is_nan() || leaf.value.im.is_nan() {
                continue;
            }
            let key = quantize(leaf.value);
            if seen.insert(key) {
                by_ops[0].push(Found {
                    expr: Rc::new(Expr::Leaf(i)),
                    value: leaf.value,
                    gen: 0,
                });
            }
        }

        // Build up by op count. A tree with n ops decomposes as
        // op(left with k ops, right with n-1-k ops) for k in 0..n.
        for n in 1..=max_ops {
            for k in 0..n {
                let rhs_ops = n - 1 - k;
                // Clone index slices to dodge borrow conflicts; Rc<Expr>
                // makes the clones cheap.
                let l_entries = by_ops[k].clone();
                let r_entries = by_ops[rhs_ops].clone();
                for l in &l_entries {
                    for r in &r_entries {
                        let value = op.eval(l.value, r.value);
                        if value.re.is_nan() || value.im.is_nan() {
                            continue;
                        }
                        let key = quantize(value);
                        if seen.insert(key) {
                            let expr = Rc::new(Expr::Op(l.expr.clone(), r.expr.clone()));
                            by_ops[n].push(Found {
                                expr,
                                value,
                                gen: 0,
                            });
                        }
                    }
                }
            }
        }

        by_ops
    }

    /// Linear-scan match of targets against a flat catalogue. For each target,
    /// returns the smallest-size Found whose value is within tolerance.
    pub fn match_targets(&self, catalogue: &[Found], targets: &[Target]) -> Vec<Discovery> {
        let mut out = Vec::new();
        for t in targets {
            let mut best: Option<(f64, usize, &Found)> = None;
            for f in catalogue {
                // Skip non-finite values: tolerance comparison isn't meaningful.
                if !(f.value.re.is_finite() && f.value.im.is_finite()) {
                    continue;
                }
                let diff = (f.value - t.value).norm();
                // NaN compares false both ways, so an explicit `< tol` rejects
                // non-finite diffs without a separate is_nan check.
                if !(diff < self.tol) {
                    continue;
                }
                let size = f.expr.size();
                let better = match best {
                    None => true,
                    Some((_, s, _)) if size < s => true,
                    Some((d, s, _)) if size == s && diff < d => true,
                    _ => false,
                };
                if better {
                    best = Some((diff, size, f));
                }
            }
            if let Some((residual, size, f)) = best {
                out.push(Discovery {
                    target_name: t.name.to_string(),
                    expression: f.expr.clone(),
                    value: f.value,
                    size,
                    expanded_size: f.expr.expanded_size(&self.leaves),
                    residual,
                    iteration: 0,
                });
            }
        }
        out
    }

    /// Iterative bootstrap: run enumerate+match, promote every newly-found
    /// target to a leaf, re-run with the expanded leaf pool. Stops when no
    /// new targets appear or after `max_iterations`.
    ///
    /// The catalogue is rebuilt from scratch each iteration, but because
    /// promoted leaves shortcut long expressions to size 1, downstream
    /// searches reach further into tree space at the same budget.
    pub fn bootstrap(
        &mut self,
        op: &dyn Operator,
        targets: &[Target],
        max_ops_per_iter: usize,
        max_iterations: usize,
    ) -> Vec<Discovery> {
        self.bootstrap_with_progress(op, targets, max_ops_per_iter, max_iterations, |_| {})
    }

    /// Same as `bootstrap` but calls `progress` after every iteration with a
    /// snapshot of that iteration's state. Useful for slow searches where
    /// you want live feedback.
    ///
    /// Uses incremental enumeration: the catalogue is built once at iter 0
    /// and extended at each subsequent iteration by considering only pairs
    /// where at least one side was added since the previous iteration. At
    /// iter N+1, pairs that were both present in iter N are skipped because
    /// those compositions were already tried.
    pub fn bootstrap_with_progress(
        &mut self,
        op: &dyn Operator,
        targets: &[Target],
        max_ops_per_iter: usize,
        max_iterations: usize,
        mut progress: impl FnMut(&BootstrapProgress),
    ) -> Vec<Discovery> {
        let mut best: HashMap<String, Discovery> = HashMap::new();

        // Persistent catalogue across iterations. `gen` on each Found records
        // which iteration it was added in; pair filtering uses this to skip
        // pairs already explored in prior iterations.
        let mut by_ops: Vec<Vec<Found>> = vec![Vec::new(); max_ops_per_iter + 1];
        let mut processed_leaves: usize = 0;

        let start = std::time::Instant::now();
        for iter in 0..max_iterations {
            let iter_start = std::time::Instant::now();
            let current_gen: u32 = iter as u32;

            // `seen` is per-iteration: we want new shortcut expressions from
            // freshly-added leaves to coexist with older deep expressions for
            // the same value, even though the old form is already in by_ops.
            // The gen-filter on pairs keeps this from blowing up — pairs that
            // were both already-present in a prior iter are skipped.
            let mut seen: HashSet<ValueKey> = HashSet::new();

            // Ingest any leaves appended since the previous iteration. If the
            // new leaf's value matches an existing catalogue entry, we still
            // want the size-1 leaf form in by_ops so downstream compositions
            // can chain from it, so we bypass the global-dedup check here.
            while processed_leaves < self.leaves.len() {
                let leaf = &self.leaves[processed_leaves];
                if !(leaf.value.re.is_nan() || leaf.value.im.is_nan()) {
                    let key = quantize(leaf.value);
                    // Per-iter dedup only: multiple new leaves with the same
                    // value collapse, but an old entry with the same value
                    // does not block the new leaf.
                    if seen.insert(key) {
                        by_ops[0].push(Found {
                            expr: Rc::new(Expr::Leaf(processed_leaves)),
                            value: leaf.value,
                            gen: current_gen,
                        });
                    }
                }
                processed_leaves += 1;
            }

            // Extend the catalogue. For each level n in 1..=max_ops_per_iter,
            // consider all (l at k, r at n-1-k) pairs; skip pairs where both
            // sides were present in a prior iter (their composition was tried
            // then). The per-iter seen set prevents multiple compositions at
            // the same iter from producing duplicate entries.
            for n in 1..=max_ops_per_iter {
                for k in 0..n {
                    let rhs_ops = n - 1 - k;
                    let l_snapshot = by_ops[k].clone();
                    let r_snapshot = by_ops[rhs_ops].clone();
                    for l in &l_snapshot {
                        for r in &r_snapshot {
                            if l.gen < current_gen && r.gen < current_gen {
                                continue;
                            }
                            let value = op.eval(l.value, r.value);
                            if value.re.is_nan() || value.im.is_nan() {
                                continue;
                            }
                            let key = quantize(value);
                            if seen.insert(key) {
                                let expr = Rc::new(Expr::Op(l.expr.clone(), r.expr.clone()));
                                by_ops[n].push(Found {
                                    expr,
                                    value,
                                    gen: current_gen,
                                });
                            }
                        }
                    }
                }
            }

            let catalogue_size: usize = by_ops.iter().map(|l| l.len()).sum();
            let flat: Vec<Found> = by_ops.iter().flatten().cloned().collect();
            let hits = self.match_targets(&flat, targets);

            let mut new_leaves: Vec<Leaf> = Vec::new();
            let mut first_time: Vec<String> = Vec::new();

            for mut d in hits {
                d.iteration = iter;
                let better = match best.get(&d.target_name) {
                    None => true,
                    Some(prev) => d.expanded_size < prev.expanded_size,
                };
                if better {
                    let is_new = !best.contains_key(&d.target_name);
                    best.insert(d.target_name.clone(), d.clone());
                    if is_new {
                        first_time.push(d.target_name.clone());
                        // Skip promoting size-1 discoveries: those are
                        // existing leaves (a target that literally names a
                        // base leaf), re-adding them duplicates the pool.
                        if d.size > 1 {
                            new_leaves.push(Leaf {
                                name: d.target_name.clone(),
                                value: d.value,
                                source: LeafSource::Derived {
                                    original: d.expression.clone(),
                                },
                            });
                        }
                    }
                }
            }

            let iter_elapsed = iter_start.elapsed();
            progress(&BootstrapProgress {
                iteration: iter,
                leaf_count: self.leaves.len() + new_leaves.len(),
                catalogue_size,
                new_targets: &first_time,
                total_found: best.len(),
                iter_elapsed,
                total_elapsed: start.elapsed(),
            });

            if new_leaves.is_empty() {
                break;
            }
            self.leaves.extend(new_leaves);
        }

        let mut out: Vec<Discovery> = best.into_values().collect();
        out.sort_by_key(|d| (d.iteration, d.expanded_size, d.target_name.clone()));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Leaf;
    use crate::operator::Eml;
    use std::f64::consts::E;

    #[test]
    fn eml_generates_e_from_one() {
        let v = Verifier::new(vec![Leaf::constant("1", C::new(1.0, 0.0))]);
        let cat: Vec<Found> = v.enumerate(&Eml, 1).into_iter().flatten().collect();
        let e = C::new(E, 0.0);
        assert!(
            cat.iter().any(|f| (f.value - e).norm() < 1e-10),
            "e should be discoverable via eml(1, 1)"
        );
    }

    #[test]
    fn eml_generates_exp_x_from_one_and_x() {
        let leaves = vec![
            Leaf::constant("1", C::new(1.0, 0.0)),
            Leaf::variable("x", C::new(0.5772156649015329, 0.0)),
        ];
        let v = Verifier::new(leaves);
        let cat: Vec<Found> = v.enumerate(&Eml, 1).into_iter().flatten().collect();
        let target = C::new(0.5772156649015329_f64.exp(), 0.0);
        assert!(
            cat.iter().any(|f| (f.value - target).norm() < 1e-10),
            "exp(x) should be discoverable via eml(x, 1)"
        );
    }

    #[test]
    fn enumerate_dedups_by_value() {
        // eml(1, 1) and eml(eml(1,1), 1) produce different values, and the
        // catalogue should contain both with distinct sizes.
        let v = Verifier::new(vec![Leaf::constant("1", C::new(1.0, 0.0))]);
        let cat: Vec<Found> = v.enumerate(&Eml, 3).into_iter().flatten().collect();
        let mut keys: HashSet<ValueKey> = HashSet::new();
        for f in &cat {
            assert!(
                keys.insert(quantize(f.value)),
                "duplicate value in catalogue"
            );
        }
        assert!(cat.len() >= 2);
    }

    #[test]
    fn quantize_handles_infinities() {
        assert_ne!(quantize_f64(f64::INFINITY), quantize_f64(f64::NEG_INFINITY));
        assert_eq!(quantize_f64(0.0), quantize_f64(-0.0));
        assert_eq!(quantize_f64(f64::NAN), u64::MAX);
    }
}
