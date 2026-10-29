//! A long prompt through the Coder's wide prefill (`prefill_wide`) against the narrow one
//! (`prefill_slot`, windows of `MAX_T`): same engine, same tokens, greedy decode after each,
//! tokens must match. The prompt is several wide windows long, the shape that crashed dart
//! in serve (`fe_q8: CUDA_ERROR_ILLEGAL_ADDRESS` on the first 3,318-token prompt) when stage
//! 0's view kept the narrow kernels while it pointed at the wide scratch.
//!
//! Needs a CUDA GPU and `TANG_LLM_TEST_GGUF`: the first shard of a `qwen3next` GGUF (e.g.
//! Qwen3-Coder-Next UD-Q4_K_XL). Skipped without the env var:
//!
//! ```text
//! TANG_LLM_TEST_GGUF=/root/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
//! cargo test --release -p tang-llm \
//!     --no-default-features --features cuda --test coder_long_prefill -- --nocapture
//! ```

#![cfg(feature = "cuda")]

use anyhow::{ensure, Result};
use tang_llm::flash::coder::{Coder, CoderOpts};

/// Prompt length: the crashing serve prompt's.
const LEN: usize = 3318;
/// Wide window rows (serve's default).
const ROWS: usize = 256;
/// Greedy decode steps compared after each prefill.
const STEPS: usize = 8;

fn gguf() -> Option<std::path::PathBuf> {
    match std::env::var("TANG_LLM_TEST_GGUF") {
        Ok(p) => Some(p.into()),
        Err(_) => {
            eprintln!("TANG_LLM_TEST_GGUF not set: skipping");
            None
        }
    }
}

fn decode(e: &mut Coder, slot: usize, mut next: u32) -> Result<Vec<u32>> {
    let mut toks = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        next = e.decode_batch(&[(slot, next)])?[0];
        toks.push(next);
    }
    Ok(toks)
}

#[test]
fn wide_prefill_of_thousands_of_tokens_matches_narrow() -> Result<()> {
    let Some(path) = gguf() else {
        return Ok(());
    };
    ensure!(tang_moe::gpu::Gpu::for_test().is_some(), "no CUDA device");

    let opts = CoderOpts {
        max_ctx: (LEN + STEPS + 8).next_multiple_of(1024),
        ..CoderOpts::default()
    };
    let mut e = Coder::load(&path, opts)?;
    ensure!(e.temperature == 0.0, "greedy only");
    e.enable_wide(ROWS)?;
    // Arbitrary nonzero ids, not periodic in the window size.
    let ids: Vec<u32> = (0..LEN as u32).map(|i| 1000 + (i * 7919) % 30000).collect();

    let s1 = e.alloc_slot()?;
    let n1 = e.prefill_slot(s1, &ids)?;
    let narrow = decode(&mut e, s1, n1)?;
    e.free_slot(s1);

    let s2 = e.alloc_slot()?;
    let n2 = e.prefill_wide(s2, &ids)?;
    let wide = decode(&mut e, s2, n2)?;
    e.free_slot(s2);

    eprintln!("narrow {n1} {narrow:?}\nwide   {n2} {wide:?}");
    ensure!(n1 == n2, "first token: narrow {n1} vs wide {n2}");
    ensure!(narrow == wide, "decode diverged");
    Ok(())
}
