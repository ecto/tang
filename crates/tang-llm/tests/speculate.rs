//! Speculative decoding must not change outputs: the same requests with drafting off and on,
//! greedy and seeded sampling, with thinking budgets, stop strings and length limits.
//!
//! Needs `TANG_LLM_TEST_MODEL` (e.g. `Qwen/Qwen3-0.6B`); skipped without it. Run with
//! `--release`.
//!
//! Exactness has two layers. The algorithm is exact: every token is sampled from the target's
//! logits in order with the same sampler state, and drafts only skip forwards. The arithmetic
//! is exact only where a k-token forward's rows are bit-identical to one-token forwards: on
//! Metal with bf16 weights that holds for k <= 4, so `bitwise_*` caps drafts at 3 there and
//! demands identical bytes. With longer drafts (and on other kernels), rows differ by ~1e-5 in
//! the logits, which can flip a near-tie; `default_drafts_match` checks the full-length
//! drafts anyway (they matched on Metal and CUDA when written).
#![cfg(any(feature = "metal", feature = "cuda"))]

use serde_json::json;
use tang_compute::ComputeDevice;
use tang_llm::chat::Piece;
use tang_llm::draft::{DraftConfig, COST_METAL};
use tang_llm::engine::{Engine, Request, Usage};
use tang_llm::sample::Sampling;
use tang_llm::Dtype;

fn model() -> Option<std::path::PathBuf> {
    let Ok(spec) = std::env::var("TANG_LLM_TEST_MODEL") else {
        eprintln!("TANG_LLM_TEST_MODEL not set: skipping");
        return None;
    };
    Some(tang_llm::resolve_model(&spec).expect("TANG_LLM_TEST_MODEL"))
}

const CODE: &str = r#"use std::collections::HashMap;

/// Count how often each word appears, ignoring case and punctuation.
pub fn word_counts(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        let word: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if word.is_empty() {
            continue;
        }
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}

/// The `n` most common words, most common first (ties broken alphabetically).
pub fn top_words(text: &str, n: usize) -> Vec<(String, usize)> {
    let mut all: Vec<(String, usize)> = word_counts(text).into_iter().collect();
    all.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    all.truncate(n);
    all
}
"#;

fn requests() -> Vec<(&'static str, Request)> {
    let greedy = Sampling {
        temperature: 0.0,
        ..Default::default()
    };
    let seeded = Sampling {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.95,
        seed: 42,
    };
    let req = |msg: String,
               think: bool,
               budget: Option<usize>,
               s: &Sampling,
               max: usize,
               stop: &[&str]| Request {
        messages: json!([{ "role": "user", "content": msg }]),
        images: vec![],
        tools: None,
        think: Some(think),
        thinking_budget: budget,
        sampling: s.clone(),
        max_tokens: Some(max),
        stop: stop.iter().map(|s| s.to_string()).collect(),
    };
    let edit = format!(
        "Here is src/words.rs:\n\n```rust\n{CODE}```\n\nRename `word_counts` to `count_words` everywhere and output the whole updated file in a rust code block, nothing else."
    );
    vec![
        (
            "edit/greedy",
            req(edit.clone(), false, None, &greedy, 400, &[]),
        ),
        (
            "edit/seeded",
            req(edit.clone(), false, None, &seeded, 400, &[]),
        ),
        (
            "chat/greedy",
            req(
                "Write a short poem about a lighthouse keeper.".into(),
                false,
                None,
                &greedy,
                120,
                &[],
            ),
        ),
        (
            "chat/seeded",
            req(
                "Explain what a suffix automaton is in three sentences.".into(),
                false,
                None,
                &seeded,
                120,
                &[],
            ),
        ),
        (
            "think+budget/greedy",
            req(edit.clone(), true, Some(40), &greedy, 200, &[]),
        ),
        (
            "think+budget/seeded",
            req(
                "What is 17 * 23? Answer briefly.".into(),
                true,
                Some(24),
                &seeded,
                120,
                &[],
            ),
        ),
        (
            "stop/greedy",
            req(edit.clone(), false, None, &greedy, 300, &["top_words"]),
        ),
        (
            "short-limit/greedy",
            req(edit, false, None, &greedy, 37, &[]),
        ),
    ]
}

fn run<D: ComputeDevice>(e: &mut Engine<D>, r: &Request) -> (String, Usage, Vec<u32>) {
    e.reset();
    let mut out = String::new();
    let (finish, usage) = e
        .complete(r, |p| {
            match p {
                Piece::Reasoning(t) => out.push_str(&format!("[r]{t}")),
                Piece::Text(t) => out.push_str(&t),
                Piece::ToolCall { name, arguments } => {
                    out.push_str(&format!("[call {name} {arguments}]"))
                }
            }
            true
        })
        .unwrap();
    out.push_str(&format!("\n[finish {finish:?}]"));
    (out, usage, e.cached_tokens().to_vec())
}

fn device() -> impl ComputeDevice {
    #[cfg(feature = "metal")]
    return tang_compute::MetalDevice::new().expect("Metal");
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    return tang_compute::CudaComputeDevice::new().expect("CUDA");
}

/// Off vs on for every request: (name, identical, off, on, usage on, caches identical).
fn compare(cfg: DraftConfig) -> Vec<(String, bool, String, String, Usage, bool)> {
    let dir = model().unwrap();
    let mut e = Engine::load(device(), &dir, 4096, Dtype::Bf16).unwrap();
    let mut rows = Vec::new();
    for (name, r) in requests() {
        e.set_speculation(None);
        let (off, _, cache_off) = run(&mut e, &r);
        e.set_speculation(Some(cfg.clone()));
        let (on, u, cache_on) = run(&mut e, &r);
        eprintln!(
            "{name}: identical={} completion={} drafts {}/{} accepted",
            off == on,
            u.completion_tokens,
            u.accepted_tokens,
            u.draft_tokens
        );
        // Rolled back correctly: both caches hold the prompt and the generated tokens (the
        // last token may or may not have been fed).
        let (a, b) = (&cache_off, &cache_on);
        let cache_ok = (a.starts_with(b) || b.starts_with(a)) && a.len().abs_diff(b.len()) <= 1;
        rows.push((name.to_string(), off == on, off, on, u, cache_ok));
    }
    rows
}

fn first_diff(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

#[test]
fn bitwise_exact_with_short_drafts() {
    if model().is_none() {
        return;
    }
    let mut cfg = DraftConfig::new(COST_METAL);
    cfg.max_draft = 3;
    cfg.global_tokens = 0;
    let rows = compare(cfg);
    for (name, same, off, on, _, cache_same) in &rows {
        assert!(
            same,
            "{name}: outputs differ at byte {}\noff: {off}\non:  {on}",
            first_diff(off, on)
        );
        assert!(cache_same, "{name}: the KV cache holds different tokens");
    }
    let edit = &rows[0].4;
    assert!(
        edit.accepted_tokens > 50,
        "the edit task should accept many drafts: {edit:?}"
    );
}

#[test]
fn default_drafts_match() {
    if model().is_none() {
        return;
    }
    let rows = compare(DraftConfig::new(COST_METAL));
    let mut bad = Vec::new();
    for (name, same, off, on, _, cache_same) in &rows {
        if !same || !cache_same {
            let i = first_diff(off, on);
            eprintln!(
                "{name}: differs at byte {i} of {}: off {:?} / on {:?}",
                off.len(),
                &off[i.saturating_sub(40)..(i + 40).min(off.len())],
                &on[i.saturating_sub(40)..(i + 40).min(on.len())]
            );
            bad.push(name.clone());
        }
    }
    assert!(bad.is_empty(), "outputs differ: {bad:?}");
}
