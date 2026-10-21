//! The OpenCL prefill over packed 4-bit weights, in f32 and in fp16,
//! against the f32-map prefill on a fixed prompt: last-token logits within
//! a bound and the same top-1. Needs `--features opencl`,
//! `TANG_BACKEND=opencl` and a Qwen3 MLX 4-bit checkpoint in `QWEN_DIR`
//! (skipped otherwise).
use tang_bench::qwen_session::prefill_bench;

#[test]
fn packed_and_fp16_prefill_match_f32_maps() {
    let Ok(dir) = std::env::var("QWEN_DIR") else {
        eprintln!("QWEN_DIR unset: skipped");
        return;
    };
    if std::env::var("TANG_BACKEND").as_deref() != Ok("opencl") {
        eprintln!("TANG_BACKEND is not opencl: skipped");
        return;
    }
    let checks = prefill_bench(&dir, &[20, 49], 0, true);
    assert_eq!(checks.len(), 4, "no OpenCL runtime?");
    for c in &checks {
        // f32 over packed weights: reassociated sums only (measured
        // 4.5e-3 on Qwen3-1.7B); fp16 operands: 3.3e-2
        let bound = if c.variant == "q4" { 2e-2 } else { 1e-1 };
        assert!(c.max_abs < bound, "{c:?}");
        assert!(c.top1_agree, "{c:?}");
    }
}
