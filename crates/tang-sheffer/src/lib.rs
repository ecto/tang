//! tang-sheffer — search for universal binary operators over elementary functions.
//!
//! Background: Odrzywołek (arXiv:2603.21852, 2026) proved that the single
//! binary operator `eml(x, y) = exp(x) − ln(y)` paired with the constant `1`
//! generates every elementary function. This is the continuous analog of NAND
//! universality for Boolean logic. The paper also conjectures that *better*
//! operators exist — polynomial growth, no complex intermediates, no
//! distinguished constant, lower tree depths for common operations.
//!
//! This crate builds the infrastructure to verify the EML result and search
//! the operator space for improved candidates. Phase 1 is enumerative
//! verification: given a candidate operator and a pool of base leaves,
//! enumerate all expression trees up to a size bound, dedup by numeric value,
//! and match against target values (constants, standard functions).

pub mod crosscheck;
pub mod expr;
#[cfg(feature = "gpu")]
pub mod gpu_search;
pub mod growth;
pub mod hp_verify;
pub mod master;
pub mod op_arena;
pub mod op_enum;
pub mod op_score;
pub mod operator;
#[cfg(test)]
mod powskew;
pub mod shape_bytecode;
pub mod targets;
pub mod verify;

pub use crosscheck::{cross_check, rebind_leaves, CrossCheckReport, TEST_POINTS};

pub use master::{fit, Adam, FitResult, Lcg, Master};

pub use expr::{Expr, Leaf, LeafSource};
pub use growth::{profile, GrowthClass, GrowthProfile};
pub use operator::{
    CoshAcosh, Edl, Eml, ExpDiff, ExpMinusSqrt, LnDiff, Operator, PowExpSkew, PowLnSkew, PowMinus,
    PowRatio, PowSkew, SinAsin, SinhAsinh, SinhDiff, SinhLn, SqrDivSqrt, SqrSqrt, TanAtan, TanDiff,
    TanhAtanh,
};
pub use targets::{
    standard_constants, standard_functions, stepping_stone_constants, Target, TEST_POINT,
};
pub use verify::{BootstrapProgress, Discovery, Found, Verifier};

pub use num_complex::Complex;
pub type C = Complex<f64>;
