//! `tang-llm logits <model-dir> <token ids...>` — dump logits for every position as JSON (for
//! checking against a reference implementation).
//! `tang-llm generate <model-dir> <token ids...> [-n N]` — greedy decode, print ids and speed.

use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use std::time::Instant;
use tang_compute::MetalDevice;
use tang_llm::Model;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, dir) = match args.as_slice() {
        [c, d, ..] => (c.as_str(), PathBuf::from(d)),
        _ => bail!("usage: tang-llm <logits|generate> <model-dir> <ids...> [-n N]"),
    };
    let mut n = 32;
    let mut ids = Vec::new();
    let mut rest = args[2..].iter();
    while let Some(a) = rest.next() {
        if a == "-n" {
            n = rest.next().context("-n N")?.parse()?;
        } else {
            ids.push(a.parse::<u32>()?);
        }
    }
    let dev = MetalDevice::new().context("no Metal device")?;
    let t = Instant::now();
    let model = Model::load(dev, &dir, 4096)?;
    eprintln!("loaded in {:.2}s", t.elapsed().as_secs_f32());
    let mut cache = model.new_cache();
    match cmd {
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
            let rows: Vec<&[f32]> = logits.chunks(v).collect();
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

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}
