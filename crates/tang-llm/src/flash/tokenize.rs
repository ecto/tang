//! Text → ids for Flash-Next. (Not built yet: pass ids.)

use anyhow::{bail, Result};
use std::path::Path;

/// The chat template applied to one user message, tokenized.
pub fn encode_chat(_gguf: &Path, _text: &str) -> Result<Vec<u32>> {
    bail!("--prompt isn't supported yet; pass --prompt-ids or --ids-file")
}
