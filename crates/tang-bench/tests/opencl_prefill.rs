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

/// A short turn after a cached prompt runs through the tail chunks
/// (OpenCL default `64,32,16,8,4,2`: a voice turn's ~20 new tokens are
/// 16 + 4 or so, plus at most one decode step) and leaves the logits that decode steps alone leave.
#[test]
fn tail_chunks_match_decode_steps() {
    use tang_bench::qwen_session::{Sampling, Session};
    let Ok(dir) = std::env::var("QWEN_DIR") else {
        eprintln!("QWEN_DIR unset: skipped");
        return;
    };
    if std::env::var("TANG_BACKEND").as_deref() != Ok("opencl") {
        eprintln!("TANG_BACKEND is not opencl: skipped");
        return;
    }
    let greedy = Sampling::default();
    let mut s = Session::open(&dir, 1024).unwrap();
    let sys = "<|im_start|>system\nYou are a small robot. Answer in one short sentence. \
               Be kind, be brief, never invent facts, and say when you do not know.<|im_end|>\n";
    let p0 = s.encode(sys);
    let mut p = p0.clone();
    p.extend(s.encode(
        "<|im_start|>user\nWhat is the weather like today?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ));
    let new = p.len() - p0.len();
    assert!((17..32).contains(&new), "{new} new tokens");
    s.prompt(&p0, &greedy);
    let (first, reused) = s.prompt(&p, &greedy);
    assert_eq!(reused, p0.len());
    let warm = s.last_logits();
    // the same prompt through decode steps only
    std::env::set_var("QWEN_PREFILL_CHUNKS", "100000");
    s.reset();
    let (first0, _) = s.prompt(&p, &greedy);
    let cold = s.last_logits();
    std::env::remove_var("QWEN_PREFILL_CHUNKS");
    let d = warm
        .iter()
        .zip(&cold)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{new} new tokens: max |dlogit| {d:.2e} vs decode steps");
    assert_eq!(first, first0);
    assert!(d < 0.1, "{d}");
}
