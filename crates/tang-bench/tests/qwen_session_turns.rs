//! Session prefix reuse across turns, with abandoned (speculative) turns in
//! between: a reused prompt must leave the same logits as the same prompt
//! prefilled from position 0. Needs a Qwen3 MLX 4-bit checkpoint in
//! `QWEN_DIR` (skipped otherwise), e.g. QWEN_DIR=~/models/qwen3-1.7b-4bit.
use tang_bench::qwen_session::{Sampling, Session};

fn chat(sys: &str, turns: &[(&str, &str)], user: &str) -> String {
    let mut s = format!("<|im_start|>system\n{sys}<|im_end|>\n");
    for (u, a) in turns {
        s += &format!("<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{a}<|im_end|>\n");
    }
    s += &format!(
        "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    s
}

fn reply(s: &mut Session, prompt: &[u32], n: usize, sm: &Sampling) -> Vec<u32> {
    let end = s.token_id("<|im_end|>").unwrap();
    let (mut t, _) = s.prompt(prompt, sm);
    let mut out = Vec::new();
    while t != end && out.len() < n {
        out.push(t);
        t = s.step(sm);
    }
    out
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn reuse_after_abandoned_turns_matches_fresh_prefill() {
    let Ok(dir) = std::env::var("QWEN_DIR") else {
        eprintln!("QWEN_DIR unset: skipped");
        return;
    };
    // broca's cache, and a system prompt long enough that turns run in the
    // 1024-position prefill window (broca's prompts are ~750 tokens)
    let mut s = Session::open(&dir, 4096).unwrap();
    let greedy = Sampling::default();
    let rules =
        "Rule: be kind, be brief, never invent facts, and say when you do not know. ".repeat(40);
    let sys = format!("You are a small robot. Answer in one short sentence.\n{rules}");
    let sys = sys.as_str();
    let users = [
        "What color is the sky?",
        "Name a fruit.",
        "How many legs does a spider have?",
        "Say a greeting in French.",
    ];
    let mut hist: Vec<(String, String)> = Vec::new();
    let mut seen = Vec::new();
    for (k, u) in users.iter().enumerate() {
        let h: Vec<(&str, &str)> = hist.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        // speculative turns on other texts, abandoned at different depths
        for (j, spec) in ["Tell me about East Palestine.", "Can you search the web?"]
            .iter()
            .enumerate()
        {
            let p = s.encode(&chat(sys, &h, spec));
            reply(&mut s, &p, 3 + 7 * j + k, &greedy);
        }
        let p = s.encode(&chat(sys, &h, u));
        let (first, reused) = s.prompt(&p, &greedy);
        assert!(
            reused > 0 && reused < p.len(),
            "turn {k}: reused {reused} of {}",
            p.len()
        );
        let warm = s.last_logits();
        let r = reply(&mut s, &p, 40, &greedy);
        // the same prompt from position 0
        s.reset();
        let (first0, reused0) = s.prompt(&p, &greedy);
        assert_eq!(reused0, 0);
        let cold = s.last_logits();
        let d = max_diff(&warm, &cold);
        eprintln!(
            "turn {k}: {u:?} -> {:?} (reused {reused}/{}, logit diff {d:.4})",
            s.decode(&r),
            p.len()
        );
        assert_eq!(
            first, first0,
            "turn {k}: first token differs from a fresh prefill"
        );
        assert!(
            d < 0.5,
            "turn {k}: logits differ from a fresh prefill by {d}"
        );
        let r0 = reply(&mut s, &p, 40, &greedy);
        assert_eq!(r, r0, "turn {k}: reply differs from a fresh prefill");
        assert!(
            !seen.contains(&r),
            "turn {k}: same reply as an earlier turn"
        );
        seen.push(r.clone());
        hist.push((u.to_string(), s.decode(&r)));
    }
}

/// The logits path (sampling, penalties) must see the logits of the last
/// prompt position whichever program ran it: a prefill chunk or a decode step.
#[test]
fn logits_path_matches_device_argmax() {
    let Ok(dir) = std::env::var("QWEN_DIR") else {
        eprintln!("QWEN_DIR unset: skipped");
        return;
    };
    let mut s = Session::open(&dir, 4096).unwrap();
    let greedy = Sampling::default();
    // greedy, but through the logits readback
    let host = Sampling {
        presence_penalty: 1e-9,
        ..Sampling::default()
    };
    let sys = "You are a small robot. Answer in one short sentence.";
    let base = s.encode(&chat(sys, &[], "Hi."));
    s.prompt(&base, &greedy);
    for (k, u) in [
        "Name a fruit",
        "Name a fruit, please",
        "What is two plus two?",
        "Count to three",
        "Say hello to Cam",
        "What day comes after Monday?",
        "Name a color",
        "Is fire hot?",
        "Who wrote Hamlet?",
    ]
    .iter()
    .enumerate()
    {
        let p = s.encode(&chat(sys, &[], u));
        let a = reply(&mut s, &p, 12, &greedy);
        let b = reply(&mut s, &p, 12, &host);
        eprintln!("{k}: {u:?} -> {:?} / {:?}", s.decode(&a), s.decode(&b));
        assert_eq!(a, b, "{u:?}: logits path diverged from the device argmax");
    }
}

/// A prompt whose new part ends exactly on a prefill chunk (no decode step
/// runs it): the reply must continue from the prompt's end.
#[test]
fn prompt_ending_on_a_chunk_continues_from_its_end() {
    let Ok(dir) = std::env::var("QWEN_DIR") else {
        eprintln!("QWEN_DIR unset: skipped");
        return;
    };
    let mut s = Session::open(&dir, 4096).unwrap();
    let greedy = Sampling::default();
    let sys = "You are a small robot. Answer in one short sentence.";
    let p0 = s.encode(&chat(sys, &[], "Tell me about East Palestine."));
    reply(&mut s, &p0, 20, &greedy);
    for u in [
        "Name a fruit.",
        "What is two plus two?",
        "Say hello to Cam.",
        "Who wrote Hamlet?",
    ] {
        let mut p = s.encode(&chat(sys, &[], u));
        let lcp = p.iter().zip(s.cached()).take_while(|(a, b)| a == b).count();
        for tail in [8usize, 16, 64] {
            // pad the user text with spaces until the new part is `tail` tokens
            let mut pad = String::new();
            while p.len() - lcp < tail {
                pad.push_str(" ok");
                p = s.encode(&chat(sys, &[], &format!("{u}{pad}")));
            }
            if p.len() - lcp != tail {
                continue;
            }
            let a = reply(&mut s, &p, 12, &greedy);
            s.reset();
            let b = reply(&mut s, &p, 12, &greedy);
            eprintln!(
                "{u:?} tail {tail}: {:?} / cold {:?}",
                s.decode(&a),
                s.decode(&b)
            );
            assert_eq!(
                a, b,
                "{u:?}: a prompt ending on a {tail}-token chunk replied off its end"
            );
            // back to a long cache for the next case
            reply(&mut s, &p0, 20, &greedy);
        }
    }
}

/// `ban_next` forbids a token for one pick only.
#[test]
fn ban_next_blocks_one_pick() {
    let Ok(dir) = std::env::var("QWEN_DIR") else {
        eprintln!("QWEN_DIR unset: skipped");
        return;
    };
    let mut s = Session::open(&dir, 1024).unwrap();
    let greedy = Sampling::default();
    let p = s.encode(&chat("Answer in one word.", &[], "Name a fruit."));
    let (a, _) = s.prompt(&p, &greedy);
    s.ban_next(&[a]);
    let (b, _) = s.prompt(&p, &greedy);
    assert_ne!(a, b, "the banned token was picked");
    let (c, _) = s.prompt(&p, &greedy);
    assert_eq!(a, c, "the ban outlived its pick");
}
