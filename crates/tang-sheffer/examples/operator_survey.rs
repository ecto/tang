//! Compare candidate Sheffer operators on two axes:
//!   1. Growth rate when iterated from a seed: bounded / poly / exp / double-exp
//!   2. Number of standard-target discoveries reachable by bootstrap at a
//!      small budget
//!
//! Operators with double-exponential growth (EML, EDL) overflow quickly and
//! need special handling; polynomial operators stay numerically stable but
//! may be incomplete. This example surfaces that trade-off in a single run.

use std::f64::consts::E;

use tang_sheffer::{
    growth, standard_constants, standard_functions, CoshAcosh, Edl, Eml, ExpMinusSqrt, Leaf,
    Operator, PowRatio, PowSkew, SinhAsinh, SinhLn, SqrDivSqrt, SqrSqrt, TanAtan, TanhAtanh,
    Verifier, C, TEST_POINT,
};

const BUDGET: usize = 4;
const ITERATIONS: usize = 4;

fn main() {
    println!("=== Operator survey ===");
    println!("budget: {} ops × {} iterations\n", BUDGET, ITERATIONS);

    let candidates: Vec<Box<dyn Operator>> = vec![
        Box::new(Eml),
        Box::new(Edl),
        Box::new(SinhLn),
        Box::new(SinhAsinh),
        Box::new(CoshAcosh),
        Box::new(TanhAtanh),
        Box::new(TanAtan),
        Box::new(SqrSqrt),
        Box::new(SqrDivSqrt),
        Box::new(PowSkew),
        Box::new(PowRatio),
        Box::new(ExpMinusSqrt),
    ];

    // -- Growth profiling --
    //
    // Seed with a generic complex value (not 1+0i, which is a fixed point for
    // pow-ratio, sqr/sqrt, and several others) and run 6 iterations. The
    // classification considers the longest finite prefix of the sequence.
    let seed = C::new(1.3, 0.7);
    println!(
        "{:<12}  {:<18}  {}",
        "operator", "growth class", "|f^k(1.3+0.7i)| sequence"
    );
    println!("{}", "-".repeat(100));
    for op in &candidates {
        let prof = growth::profile(op.as_ref(), seed, 6);
        let magseq = prof
            .magnitudes
            .iter()
            .map(|m| {
                if m.is_nan() {
                    "nan".to_string()
                } else if !m.is_finite() {
                    "inf".to_string()
                } else if *m < 1e-3 || *m > 1e6 {
                    format!("{:.1e}", m)
                } else {
                    format!("{:.2}", m)
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "{:<12}  {:<18?}  {}",
            op.name(),
            prof.classification,
            magseq
        );
    }

    // -- Bootstrap coverage --
    //
    // For each operator, run bootstrap with its "natural" distinguished
    // constant. EML/SinhLn/etc. use 1; EDL uses e; algebraic candidates use
    // 2 (so x² is non-trivial). PowSkew uses e. This isn't a perfect
    // head-to-head but it's the "fair try" per operator.
    let mut targets = standard_constants();
    targets.extend(standard_functions());

    println!(
        "\n{:<12}  {:<7}  {:>5}  {:>6}  {}",
        "operator", "const", "found", "/total", "first-seen depth histogram"
    );
    println!("{}", "-".repeat(80));

    let runs: Vec<(&dyn Operator, &str, C)> = vec![
        (&Eml, "1", C::new(1.0, 0.0)),
        (&Edl, "e", C::new(E, 0.0)),
        (&SinhLn, "1", C::new(1.0, 0.0)),
        (&SinhAsinh, "1", C::new(1.0, 0.0)),
        (&CoshAcosh, "1", C::new(1.0, 0.0)),
        (&TanhAtanh, "1", C::new(1.0, 0.0)),
        (&TanAtan, "1", C::new(1.0, 0.0)),
        (&SqrSqrt, "2", C::new(2.0, 0.0)),
        (&SqrDivSqrt, "2", C::new(2.0, 0.0)),
        (&PowSkew, "2", C::new(2.0, 0.0)),
        (&PowRatio, "2", C::new(2.0, 0.0)),
        (&ExpMinusSqrt, "1", C::new(1.0, 0.0)),
    ];

    for (op, const_name, const_value) in runs {
        let leaves = vec![
            Leaf::constant(const_name, const_value),
            Leaf::variable("x", C::new(TEST_POINT, 0.0)),
        ];
        let mut v = Verifier::new(leaves);
        let discoveries = v.bootstrap(op, &targets, BUDGET, ITERATIONS);
        let found = discoveries.len();
        let total = targets.len();

        // Depth histogram: bucket by iteration of first discovery.
        let mut hist = [0usize; 8];
        for d in &discoveries {
            let idx = d.iteration.min(7);
            hist[idx] += 1;
        }
        let hist_str = hist
            .iter()
            .enumerate()
            .take(ITERATIONS)
            .map(|(i, n)| format!("i{}:{}", i, n))
            .collect::<Vec<_>>()
            .join(" ");

        println!(
            "{:<12}  {:<7}  {:>5}  {:>6}  {}",
            op.name(),
            const_name,
            found,
            total,
            hist_str
        );
    }
}
