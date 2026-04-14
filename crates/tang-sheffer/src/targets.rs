//! Target sets the verifier tries to discover. Targets are named complex
//! values; the verifier matches them against the dedup'd catalogue by
//! tolerance. Standard targets cover the core constants and elementary
//! functions evaluated at a fixed transcendental test point.

use std::f64::consts::{E, PI};

use crate::C;

/// A named target value.
#[derive(Debug, Clone)]
pub struct Target {
    pub name: &'static str,
    pub value: C,
}

/// Transcendental test point for single-variable function searches.
/// Euler-Mascheroni γ ≈ 0.5772... is (conjecturally, via Schanuel)
/// algebraically independent of {e, π, ln(2), ...}, so a numerical match
/// at this point is strong evidence of a true formula match rather than
/// coincidence.
pub const TEST_POINT: f64 = 0.5772156649015329;

/// Second transcendental test point for two-variable searches. The
/// Glaisher-Kinkelin constant A ≈ 1.2824... is conjecturally algebraically
/// independent of γ and the other standard transcendentals.
pub const TEST_POINT_2: f64 = 1.2824271291006226;

/// Extended constant target list including stepping-stone values that are
/// essential for EML's IEEE-infinity-based chains: `±∞` let you reach 0
/// and −1, `e−1` and `exp(e)` are common intermediate values. Having
/// these as explicit targets lets the bootstrap loop promote them to
/// leaves and shortcut downstream searches.
pub fn stepping_stone_constants() -> Vec<Target> {
    let mut v = standard_constants();
    v.extend([
        Target { name: "+inf", value: C::new(f64::INFINITY, 0.0) },
        Target { name: "-inf", value: C::new(f64::NEG_INFINITY, 0.0) },
        Target { name: "exp(e)", value: C::new(E.exp(), 0.0) },
        Target { name: "e-1", value: C::new(E - 1.0, 0.0) },
        Target { name: "e+1", value: C::new(E + 1.0, 0.0) },
        Target { name: "2e", value: C::new(2.0 * E, 0.0) },
        Target { name: "e/2", value: C::new(E / 2.0, 0.0) },
        Target { name: "ln(pi)", value: C::new(PI.ln(), 0.0) },
        Target { name: "i*pi/2", value: C::new(0.0, PI / 2.0) },
        Target { name: "-i*pi", value: C::new(0.0, -PI) },
        Target { name: "2i", value: C::new(0.0, 2.0) },
        Target { name: "i/2", value: C::new(0.0, 0.5) },
    ]);
    v
}

/// Core constant targets every universal operator should generate.
pub fn standard_constants() -> Vec<Target> {
    vec![
        Target {
            name: "0",
            value: C::new(0.0, 0.0),
        },
        Target {
            name: "1",
            value: C::new(1.0, 0.0),
        },
        Target {
            name: "-1",
            value: C::new(-1.0, 0.0),
        },
        Target {
            name: "2",
            value: C::new(2.0, 0.0),
        },
        Target {
            name: "-2",
            value: C::new(-2.0, 0.0),
        },
        Target {
            name: "1/2",
            value: C::new(0.5, 0.0),
        },
        Target {
            name: "e",
            value: C::new(E, 0.0),
        },
        Target {
            name: "-e",
            value: C::new(-E, 0.0),
        },
        Target {
            name: "1/e",
            value: C::new(1.0 / E, 0.0),
        },
        Target {
            name: "e^2",
            value: C::new(E * E, 0.0),
        },
        Target {
            name: "pi",
            value: C::new(PI, 0.0),
        },
        Target {
            name: "pi/2",
            value: C::new(PI / 2.0, 0.0),
        },
        Target {
            name: "2pi",
            value: C::new(2.0 * PI, 0.0),
        },
        Target {
            name: "i",
            value: C::new(0.0, 1.0),
        },
        Target {
            name: "-i",
            value: C::new(0.0, -1.0),
        },
        Target {
            name: "i*pi",
            value: C::new(0.0, PI),
        },
    ]
}

/// Standard elementary functions evaluated at TEST_POINT. If the verifier's
/// leaf pool contains `x = TEST_POINT`, a hit here means the operator builds
/// that function from its inputs. Values that would be NaN in real arithmetic
/// (e.g. `ln(ln(γ))` because ln(γ) < 0) are computed in complex arithmetic so
/// the test point still matches the principal-branch result.
pub fn standard_functions() -> Vec<Target> {
    let xr = TEST_POINT;
    let x = C::new(xr, 0.0);
    vec![
        Target {
            name: "x",
            value: x,
        },
        Target {
            name: "exp(x)",
            value: x.exp(),
        },
        Target {
            name: "ln(x)",
            value: x.ln(),
        },
        Target {
            name: "-x",
            value: -x,
        },
        Target {
            name: "1/x",
            value: C::new(1.0, 0.0) / x,
        },
        Target {
            name: "x^2",
            value: x * x,
        },
        Target {
            name: "sqrt(x)",
            value: x.sqrt(),
        },
        Target {
            name: "x+1",
            value: x + C::new(1.0, 0.0),
        },
        Target {
            name: "x-1",
            value: x - C::new(1.0, 0.0),
        },
        Target {
            name: "2x",
            value: x * C::new(2.0, 0.0),
        },
        Target {
            name: "e*x",
            value: x * C::new(E, 0.0),
        },
        Target {
            name: "exp(exp(x))",
            value: x.exp().exp(),
        },
        Target {
            name: "ln(ln(x))",
            value: x.ln().ln(),
        },
        Target {
            name: "sin(x)",
            value: x.sin(),
        },
        Target {
            name: "cos(x)",
            value: x.cos(),
        },
    ]
}
