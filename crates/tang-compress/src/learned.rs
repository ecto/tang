//! Learned coefficient models (M1b). Placeholder.

use crate::bitio::{ByteReader, ByteWriter, DecodeError};
use crate::coefcode::{Bits, CoefModel, Feat, HandModel};
use crate::wavelet::Pyramid;

/// How to obtain the learned model.
#[derive(Clone, Debug)]
pub struct LearnedSpec;

impl LearnedSpec {
    /// Write the model section and return the model.
    pub fn prepare(&self, _p: &Pyramid, _q: &[i32], _preds: &[crate::coefcode::Pred], w: &mut ByteWriter) -> NetModel {
        w.u8(0);
        NetModel(HandModel::default())
    }
}

/// Network-driven coefficient model.
pub struct NetModel(HandModel);

impl CoefModel for NetModel {
    fn code<B: Bits>(&mut self, b: &mut B, f: &Feat, v: i32) -> i32 {
        self.0.code(b, f, v)
    }
}

/// Read the model section for `tag`.
pub fn read_model(_tag: u8, _r: &mut ByteReader) -> Result<NetModel, DecodeError> {
    Err(DecodeError::Invalid("model"))
}

/// Parse a CLI model argument (`hand`, `overfit`, `pretrained:PATH`).
pub fn parse_model_arg(s: &str) -> crate::wavcodec::ModelChoice {
    match s {
        "hand" => crate::wavcodec::ModelChoice::Hand,
        _ => panic!("unknown model {s}"),
    }
}
