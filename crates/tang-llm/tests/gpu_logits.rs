//! End to end: a GPU backend's logits against the CPU backend's for the same checkpoint and
//! tokens, through one prefill and through token-by-token decode.
//!
//! Needs `TANG_LLM_TEST_MODEL`: a model directory, or a Hugging Face repo id already in the
//! local cache, small enough to also run on the CPU (e.g. `mlx-community/Qwen3-0.6B-4bit`).
//! Skipped without it. `TANG_LLM_TEST_TOKENS` sets the prompt length (default 48). Run with
//! `--release`: the CPU reference is slow.

use std::path::{Path, PathBuf};
use tang_compute::{ComputeDevice, CpuDevice};
use tang_llm::{Dtype, Model};
use tokenizers::Tokenizer;

/// Decode steps checked at the end of the prompt.
const STEPS: usize = 4;

fn model_dir() -> Option<PathBuf> {
    let Ok(spec) = std::env::var("TANG_LLM_TEST_MODEL") else {
        eprintln!("TANG_LLM_TEST_MODEL not set: skipping");
        return None;
    };
    Some(tang_llm::resolve_model(&spec).expect("TANG_LLM_TEST_MODEL"))
}

/// Log-softmax in f64.
fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z = x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
    x.iter().map(|&v| v as f64 - m - z).collect()
}

fn argmax(x: &[f32]) -> usize {
    (0..x.len()).fold(0, |b, i| if x[i] > x[b] { i } else { b })
}

/// Rows of `got` against `want` (`[rows, vocab]`): same top-1 everywhere, KL below 1e-3 (the
/// bar `scripts/check_logits*.py` use).
fn check(what: &str, got: &[f32], want: &[f32], vocab: usize) {
    assert_eq!(got.len(), want.len(), "{what}: shape");
    let (mut max_kl, mut max_diff) = (0.0f64, 0.0f32);
    for (r, (g, w)) in got.chunks(vocab).zip(want.chunks(vocab)).enumerate() {
        assert_eq!(argmax(g), argmax(w), "{what}: top-1 differs at row {r}");
        let (lg, lw) = (log_softmax(g), log_softmax(w));
        let kl: f64 = lw.iter().zip(&lg).map(|(w, g)| w.exp() * (w - g)).sum();
        max_kl = max_kl.max(kl);
        max_diff = g
            .iter()
            .zip(w)
            .map(|(a, b)| (a - b).abs())
            .fold(max_diff, f32::max);
    }
    eprintln!("{what}: max |Δlogit| {max_diff:.4} · max KL {max_kl:.2e}");
    assert!(max_kl < 1e-3, "{what}: KL {max_kl:.2e}");
}

fn compare<D: ComputeDevice>(dev: D, dir: &Path, name: &str) {
    let n: usize = std::env::var("TANG_LLM_TEST_TOKENS")
        .ok()
        .map_or(48, |v| v.parse().expect("TANG_LLM_TEST_TOKENS"));
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let text = "The quick brown fox jumps over the lazy dog. def fib(n): return n if n < 2 \
                else fib(n - 1) + fib(n - 2)\n";
    let mut ids = Vec::new();
    while ids.len() < n {
        ids.extend(tok.encode(text, false).expect("encode").get_ids());
    }
    ids.truncate(n.max(STEPS + 1));
    let n = ids.len();

    let cpu = Model::load(CpuDevice::new(), dir, n + 8, Dtype::Bf16).expect("load (cpu)");
    let gpu = Model::load(dev, dir, n + 8, Dtype::Bf16).expect("load");
    let vocab = cpu.cfg.vocab_size;
    let want = cpu.forward(&ids, &mut cpu.new_cache(), true).unwrap();

    let got = gpu.forward(&ids, &mut gpu.new_cache(), true).unwrap();
    check(&format!("{name} prefill, {n} tokens"), &got, &want, vocab);

    // Prefill all but the last STEPS tokens, then decode them one at a time.
    let mut cache = gpu.new_cache();
    let mut got = gpu.forward(&ids[..n - STEPS], &mut cache, false).unwrap();
    for &id in &ids[n - STEPS..] {
        got.extend(gpu.forward(&[id], &mut cache, false).unwrap());
    }
    let tail = &want[(n - STEPS - 1) * vocab..];
    check(&format!("{name} decode, last {STEPS}"), &got, tail, vocab);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_logits_match_cpu() {
    let Some(dir) = model_dir() else { return };
    let dev = tang_compute::CudaComputeDevice::new().expect("no CUDA device");
    compare(dev, &dir, "cuda");
}

#[cfg(feature = "metal")]
#[test]
fn metal_logits_match_cpu() {
    let Some(dir) = model_dir() else { return };
    let dev = tang_compute::MetalDevice::new().expect("no Metal device");
    compare(dev, &dir, "metal");
}
