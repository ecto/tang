//! Guard: `Scalar::alg_add`/`alg_sub`/`alg_mul` must never appear in code whose
//! floating-point evaluation *order* is load-bearing.
//!
//! The `algebraic` feature lowers these to Rust's `algebraic_add`/`_sub`/`_mul`,
//! which let the compiler reassociate and vectorize float chains. That is a win
//! in bulk reductions (dot, norm, sum) where a ~1 ulp drift per element is
//! acceptable. It is *silently wrong* in the files listed below, and the failure
//! mode is a subtly incorrect answer — no compile error, no panic. Hence a lint.
//!
//! To add a file: put it in `FORBIDDEN` with a comment saying **why** the order
//! matters there. The reasoning is the point; the list is just its shadow.

use std::path::{Path, PathBuf};

/// Files where reassociation breaks correctness, not just accuracy.
const FORBIDDEN: &[(&str, &str)] = &[
    (
        "crates/tang/src/predicates.rs",
        // Shewchuk adaptive-precision predicates (via the `robust` crate) work by
        // computing exact error terms of each operation and summing them in a
        // strictly defined order. Reassociation discards those terms, so
        // orientation/incircle/insphere return the WRONG SIGN near degeneracy —
        // which downstream shows up as broken boolean geometry in vcad.
        "exact predicates: sign correctness depends on exact evaluation order",
    ),
    (
        "crates/tang-la/src/svd.rs",
        // One-sided Jacobi. The Gram accumulations (app/aqq/apq) feed the
        // convergence test `|apq| < tol * sqrt(app*aqq)`; drift in those sums
        // perturbs a near-tolerance comparison and can cost convergence.
        // This is also where compensated (Kahan) summation would live if added:
        // reassociation cancels the compensation term, making the result
        // strictly worse than a plain strict sum while looking like a speedup.
        "SVD: Jacobi convergence test + any compensated summation",
    ),
    (
        "crates/tang/src/la/svd.rs",
        // Mirror of crates/tang-la/src/svd.rs — same reasoning.
        "SVD: Jacobi convergence test + any compensated summation",
    ),
];

const NEEDLES: &[&str] = &["alg_add", "alg_sub", "alg_mul"];

fn repo_root() -> PathBuf {
    // crates/tang -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("manifest dir has a repo root")
        .to_path_buf()
}

#[test]
fn no_algebraic_ops_in_order_sensitive_files() {
    let root = repo_root();
    let mut violations = Vec::new();

    for (rel, why) in FORBIDDEN {
        let path = root.join(rel);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("forbidden-list entry {rel} is unreadable ({e}) — the file moved or was deleted; update FORBIDDEN"));

        // Comment lines are skipped so the "don't use alg_*" notices in these
        // files don't trip their own guard — except inside a doc-test fence,
        // where the "comment" is real compiled-and-run code. `/* */` blocks are
        // not skipped: nothing in them executes, so scanning them only risks a
        // false positive, which is the safe direction for a lint like this.
        let mut in_doctest = false;

        for (i, line) in src.lines().enumerate() {
            let trimmed = line.trim_start();
            let is_doc = trimmed.starts_with("///") || trimmed.starts_with("//!");

            if is_doc
                && trimmed
                    .trim_start_matches(['/', '!'])
                    .trim_start()
                    .starts_with("```")
            {
                in_doctest = !in_doctest;
                continue;
            }
            if trimmed.starts_with("//") && !in_doctest {
                continue;
            }
            for needle in NEEDLES {
                if line.contains(needle) {
                    violations.push(format!("{rel}:{} uses `{needle}` — {why}", i + 1));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "reassociable float ops used where evaluation order is load-bearing:\n  {}\n\n\
         Use strict `+`/`-`/`*` here. See the comment at the top of this test.",
        violations.join("\n  ")
    );
}
