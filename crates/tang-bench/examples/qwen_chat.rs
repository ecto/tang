//! Chat turns through `tang_bench::qwen_session::Session`: the KV cache is
//! kept between turns, so each turn prefills only what is new (the system
//! prompt once). Prints, per turn: positions reused, prefilled, time to the
//! first token, decode tok/s and the reply.
//!
//! ```text
//! qwen_chat <model dir> [--ctx 2048] [--gen 48] [--system FILE] [--temp 0] [--cold]
//!     --cold: forget the cache every turn (check: replies must not change)
//! ```
use std::time::Instant;

use tang_bench::qwen_session::{Sampling, Session};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
    };
    let ctx: usize = opt("--ctx").map_or(2048, |v| v.parse().unwrap());
    let gen: usize = opt("--gen").map_or(48, |v| v.parse().unwrap());
    let system = opt("--system").map(|f| std::fs::read_to_string(f).unwrap()).unwrap_or_else(|| {
        "You are K1, a small friendly humanoid robot built by Cam in his garage. You speak out loud, \
         so answer in one or two short spoken sentences, with no lists, no markdown and no emoji. \
         You can walk, wave, look around with your head camera, and talk. If you do not know \
         something, say so plainly. Be warm, a little playful, and curious about the people you meet."
            .into()
    });
    let sampling = Sampling {
        temperature: opt("--temp").map_or(0.0, |v| v.parse().unwrap()),
        ..Default::default()
    };
    let t = Instant::now();
    let mut s = Session::open(&args[0], ctx).expect("open");
    eprintln!(
        "loaded in {:.1}s on {}",
        t.elapsed().as_secs_f64(),
        s.rt.name()
    );
    let end = s.token_id("<|im_end|>").expect("im_end");
    let t = Instant::now();
    s.warm(1024);
    eprintln!(
        "prefill programs built in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    let turns = [
        "Hey robot, what is a good name for a dog?",
        "Why that one?",
        "Can you remember what we were just talking about?",
    ];
    let mut convo = format!("<|im_start|>system\n{system}<|im_end|>\n");
    for user in turns {
        convo.push_str(&format!(
            "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        let ids = s.encode(&convo);
        if args.iter().any(|a| a == "--cold") {
            s.reset();
        }
        let t = Instant::now();
        let (mut tok, reused) = s.prompt(&ids, &sampling);
        let ttft = t.elapsed().as_secs_f64() * 1e3;
        let mut reply = vec![tok];
        let t = Instant::now();
        while tok != end && reply.len() < gen {
            tok = s.step(&sampling);
            reply.push(tok);
        }
        let dt = t.elapsed().as_secs_f64();
        let text = s.decode(&reply);
        println!(
            "turn: {} prompt tokens, {reused} reused, {} prefilled, first token {ttft:.0} ms, {} tokens at {:.1} tok/s\n  user: {user}\n  k1:   {}",
            ids.len(),
            ids.len() - reused,
            reply.len(),
            (reply.len() - 1) as f64 / dt.max(1e-9),
            text.trim_end_matches("<|im_end|>").trim()
        );
        convo.push_str(&text);
        if !text.ends_with("<|im_end|>") {
            convo.push_str("<|im_end|>");
        }
        convo.push('\n');
    }
}
