//! Tucker codec (placeholder).
use crate::bitio::DecodeError;
use crate::codec::{Decoded, ErrorBound};

/// Stream magic.
pub const MAGIC: &[u8; 4] = b"TCT1";

/// Stats.
#[derive(Clone, Debug, Default)]
pub struct TuckerStats {
    /// Core bytes.
    pub core_bytes: usize,
    /// Factor bytes.
    pub factor_bytes: usize,
    /// Ranks.
    pub ranks: Vec<usize>,
    /// Step.
    pub delta: f64,
    /// RMSE.
    pub rmse: f64,
    /// Max abs.
    pub max_abs: f64,
}

/// Encode.
pub fn encode_f32(_x: &[f32], _dims: &[usize], _b: ErrorBound) -> (Vec<u8>, TuckerStats) {
    unimplemented!()
}

/// Decode.
pub fn decode(_b: &[u8]) -> Result<Decoded, DecodeError> {
    Err(DecodeError::Invalid("tucker"))
}
