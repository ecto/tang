//! `tang-llm logits <model-dir> <token ids...>` — dump logits for every position as JSON (for
//! checking against a reference implementation).
//! `tang-llm generate <model-dir> <token ids...> [-n N]` — greedy decode, print ids and speed.
//! `tang-llm serve <model-dir | hf-repo-id> [--port P] [--ctx N]` — OpenAI-compatible server.
//! `tang-llm image-features <model-dir> <pixels.f32>` — the projector's output for an image
//! (`[896, 896, 3]` f32, normalized), as JSON.
//! `tang-llm logits-image <model-dir> <pixels.f32> <token ids...> [--last N]` — logits with the
//! image standing in for its placeholder tokens.

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::time::Instant;
use tang_compute::{ComputeDevice, MetalDevice};
use tang_llm::{Dtype, Engine, Model};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, dir) = match args.as_slice() {
        [c, d, ..] => (c.as_str(), PathBuf::from(d)),
        _ => bail!("usage: tang-llm <logits|generate> <model-dir> <ids...> [-n N] [--f32]"),
    };
    if args.first().map(String::as_str) == Some("serve") {
        return serve(&args[1..]);
    }
    // Multimodal checks take a raw f32 pixel file before the token ids.
    let mut pixels: Option<Vec<f32>> = None;
    let mut n = 32;
    // `logits --last N`: only the last N positions (long prompts).
    let mut last: Option<usize> = None;
    let mut dtype = Dtype::Bf16;
    let mut ids = Vec::new();
    let mut rest = args[2..].iter();
    if matches!(cmd, "image-features" | "logits-image") {
        let path = rest.next().context("a pixels file")?;
        let bytes = std::fs::read(path)?;
        pixels = Some(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
    }
    while let Some(a) = rest.next() {
        if a == "--f32" {
            dtype = Dtype::F32;
        } else if a == "--q4" {
            dtype = Dtype::Q4;
        } else if a == "--last" {
            last = Some(rest.next().context("--last N")?.parse()?);
        } else if a == "-n" {
            n = rest.next().context("-n N")?.parse()?;
        } else {
            ids.push(a.parse::<u32>()?);
        }
    }
    let dev = MetalDevice::new().context("no Metal device")?;
    let t = Instant::now();
    let model = Model::load(dev, &dir, 4096, dtype)?;
    eprintln!("loaded in {:.2}s", t.elapsed().as_secs_f32());
    let mut cache = model.new_cache();
    match cmd {
        "image-features" => {
            let v = model.vision.as_ref().context("no vision tower")?;
            let t = Instant::now();
            let f = v.project(&model.dev, pixels.as_ref().unwrap());
            let out = model.dev.download(&f);
            eprintln!("encoded in {:.0}ms", t.elapsed().as_secs_f64() * 1e3);
            let rows: Vec<&[f32]> = out.chunks(model.cfg.hidden_size).collect();
            println!("{}", serde_json::to_string(&rows)?);
        }
        "logits-image" => {
            let img = model.encode_image(pixels.as_ref().unwrap())?;
            let logits = model.forward_images(&ids, &[img], &mut cache, true)?;
            let v = model.cfg.vocab_size;
            let mut rows: Vec<&[f32]> = logits.chunks(v).collect();
            if let Some(n) = last {
                rows = rows.split_off(rows.len().saturating_sub(n));
            }
            println!("{}", serde_json::to_string(&rows)?);
        }
        "logits" | "logits-step" => {
            let logits = if cmd == "logits" {
                model.forward(&ids, &mut cache, true)?
            } else {
                let mut all = Vec::new();
                for &id in &ids {
                    all.extend(model.forward(&[id], &mut cache, true)?);
                }
                all
            };
            let v = model.cfg.vocab_size;
            let mut rows: Vec<&[f32]> = logits.chunks(v).collect();
            if let Some(n) = last {
                rows = rows.split_off(rows.len().saturating_sub(n));
            }
            println!("{}", serde_json::to_string(&rows)?);
        }
        "generate" => {
            let t = Instant::now();
            let mut logits = model.forward(&ids, &mut cache, false)?;
            let prefill = t.elapsed();
            let eos = model.cfg.eos();
            let t = Instant::now();
            let mut out = Vec::new();
            for _ in 0..n {
                let next = argmax(&logits);
                out.push(next);
                if eos.contains(&next) {
                    break;
                }
                logits = model.forward(&[next], &mut cache, false)?;
            }
            let dt = t.elapsed().as_secs_f64();
            println!("{}", serde_json::to_string(&out)?);
            eprintln!(
                "prefill {} tok in {:.0}ms ({:.0} tok/s) · decode {} tok at {:.1} tok/s",
                ids.len(),
                prefill.as_secs_f64() * 1e3,
                ids.len() as f64 / prefill.as_secs_f64(),
                out.len(),
                out.len() as f64 / dt
            );
        }
        c => bail!("unknown command {c}"),
    }
    Ok(())
}

fn serve(args: &[String]) -> Result<()> {
    let spec = args
        .first()
        .context("usage: tang-llm serve <model> [--port P] [--ctx N] [--f32 | --q4]")?
        .clone();
    let (mut port, mut ctx, mut dtype) = (8911u16, 32_768usize, Dtype::Bf16);
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => port = it.next().context("--port P")?.parse()?,
            "--ctx" => ctx = it.next().context("--ctx N")?.parse()?,
            "--f32" => dtype = Dtype::F32,
            "--q4" => dtype = Dtype::Q4,
            other => bail!("unknown option {other}"),
        }
    }
    let dir = tang_llm::resolve_model(&spec)?;
    let name = spec.clone();
    tang_llm::server::serve(&format!("127.0.0.1:{port}"), name, move || {
        let t = Instant::now();
        let dev = MetalDevice::new().context("no Metal device")?;
        let e = Engine::load(dev, &dir, ctx, dtype)?;
        eprintln!(
            "tang-llm: loaded {} in {:.1}s ({} ctx)",
            dir.display(),
            t.elapsed().as_secs_f32(),
            e.context_window()
        );
        Ok(e)
    })
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}
