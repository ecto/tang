//! Per-stream and total decode tok/s at several batch sizes (see `docs/batching.md`).
//!
//! `cargo run --release --example bench_batch --no-default-features --features cuda -- <model>
//! [--q4] [--batches 1,2,4,8] [--tokens 256] [--speculate]`
//!
//! Each batch gets that many different prompts, prefilled first; then decode is timed until
//! every stream has `--tokens` tokens (greedy; essays rarely stop that early, and a stream
//! that does just counts fewer). Per-stream tok/s is
//! a stream's tokens over the decode time; total is the batch's tokens over it. It also
//! checks each stream's tokens against the same prompt decoded alone.
use anyhow::Result;
use serde_json::json;
use std::time::Instant;
use tang_compute::ComputeDevice;
use tang_llm::draft::{DraftConfig, COST_CUDA};
use tang_llm::engine::{Control, Engine, Lane, Request, Stream};
use tang_llm::sample::Sampling;
use tang_llm::Dtype;

const TOPICS: [&str; 8] = [
    "the history of lighthouses",
    "how a compiler's register allocator works",
    "growing tomatoes on a balcony",
    "the rules of cricket",
    "why the sky is blue",
    "the design of a bicycle gearbox",
    "how vaccines train the immune system",
    "writing a good bug report",
];

fn device() -> impl ComputeDevice {
    #[cfg(feature = "metal")]
    return tang_compute::MetalDevice::new().expect("Metal");
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    return tang_compute::CudaComputeDevice::new().expect("CUDA");
}

fn request(i: usize, tokens: usize) -> Request {
    Request {
        messages: json!([{ "role": "user", "content": format!("Write a long essay about {}.", TOPICS[i % 8]) }]),
        images: vec![],
        tools: None,
        response_schema: None,
        think: Some(false),
        thinking_budget: None,
        sampling: Sampling {
            temperature: 0.0,
            ..Default::default()
        },
        temperature_set: true,
        top_k_set: false,
        top_p_set: false,
        presence_penalty: None,
        max_tokens: Some(tokens),
        stop: vec![],
        cache_key: None,
        prefill_only: false,
    }
}

/// Generated tokens, and the decode-phase seconds and steps.
fn run<D: ComputeDevice>(
    e: &mut Engine<D>,
    reqs: &[Request],
) -> Result<(Vec<Vec<u32>>, f64, usize)> {
    e.reset();
    let mut streams: Vec<Stream<D::Buffer>> = reqs
        .iter()
        .map(|r| e.admit(r, None))
        .collect::<Result<_>>()?;
    let mut decode: Option<Instant> = None;
    let mut steps = 0;
    while streams.iter().any(|s| !s.done()) {
        if decode.is_none() && !streams.iter().any(|s| s.prefilling()) {
            e.model.dev.sync();
            decode = Some(Instant::now());
        }
        let mut ons: Vec<Box<dyn FnMut(tang_llm::chat::Piece) -> bool>> = (0..streams.len())
            .map(|_| Box::new(|_| true) as Box<_>)
            .collect();
        let mut ctls = vec![(); streams.len()];
        let mut lanes: Vec<Lane<_>> = streams
            .iter_mut()
            .filter(|s| !s.done())
            .zip(ons.iter_mut())
            .zip(ctls.iter_mut())
            .map(|((s, on), c)| Lane {
                stream: s,
                on: &mut **on,
                ctl: c as &mut dyn Control,
            })
            .collect();
        e.step(&mut lanes, 512)?;
        if decode.is_some() {
            steps += 1;
        }
    }
    let secs = decode.map_or(0.0, |t| t.elapsed().as_secs_f64());
    let mut outs = Vec::new();
    for s in streams {
        e.retire(s)?;
        outs.push(e.last_tokens().1.to_vec());
    }
    Ok((outs, secs, steps))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: bench_batch <model> [--q4] [--batches 1,2,4,8] [--tokens N] [--speculate]");
    let (mut dtype, mut batches, mut tokens, mut speculate) =
        (Dtype::Bf16, vec![1, 2, 4, 8], 256, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--q4" => dtype = Dtype::Q4,
            "--batches" => {
                batches = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|b| b.parse().unwrap())
                    .collect()
            }
            "--tokens" => tokens = args.next().unwrap().parse()?,
            "--speculate" => speculate = true,
            o => anyhow::bail!("unknown option {o}"),
        }
    }
    let dir = tang_llm::resolve_model(&model)?;
    let mut e = Engine::load(device(), &dir, 4096, dtype)?;
    let max = *batches.iter().max().unwrap();
    e.set_slots(max);
    if speculate {
        e.set_speculation(Some(DraftConfig::new(COST_CUDA)));
    }
    // Never stop early, so every stream decodes exactly `tokens`.
    let reqs: Vec<Request> = (0..max).map(|i| request(i, tokens)).collect();
    // Warm up kernels, then each prompt alone for the agreement check.
    run(&mut e, &reqs[..1])?;
    let alone: Vec<Vec<u32>> = (0..max)
        .map(|i| run(&mut e, &reqs[i..i + 1]).map(|r| r.0.into_iter().next().unwrap()))
        .collect::<Result<_>>()?;
    println!("batch | per-stream tok/s | total tok/s | ms/step | same as alone");
    for &b in &batches {
        let (outs, secs, steps) = run(&mut e, &reqs[..b])?;
        let n: usize = outs.iter().map(|o| o.len()).sum();
        let same = outs.iter().zip(&alone).filter(|(a, b)| a == b).count();
        println!(
            "{b:5} | {:16.1} | {:11.1} | {:7.2} | {same}/{b}",
            n as f64 / b as f64 / secs,
            n as f64 / secs,
            secs * 1e3 / steps.max(1) as f64,
        );
    }
    Ok(())
}
