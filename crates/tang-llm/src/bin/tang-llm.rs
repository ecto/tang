//! `tang-llm logits <model-dir> <token ids...>` — dump logits for every position as JSON (for
//! checking against a reference implementation).
//! `tang-llm generate <model-dir> <token ids...> [-n N]` — greedy decode, print ids and speed.
//! `tang-llm serve <model-dir | hf-repo-id> [--host H] [--port P] [--ctx N] [--api-key-file F]`
//! — OpenAI-compatible server (on 127.0.0.1 unless `--host` says otherwise). With a key (from
//! the file, or `TANG_API_KEY`), requests need `Authorization: Bearer <key>`.
//! `tang-llm image-features <model-dir> <pixels.f32>` — the projector's output for an image
//! (`[896, 896, 3]` f32, normalized), as JSON.
//! `tang-llm logits-image <model-dir> <pixels.f32> <token ids...> [--last N]` — logits with the
//! image standing in for its placeholder tokens.
//!
//! `tang-llm bench-prefill <model> [--n 1024,4096,9216] [--reps R]` — cold prefill (empty
//! cache, 512-token chunks like the server) of N synthetic tokens: ms and tok/s.
//! `tang-llm bench-verify <model> [--ctx 4096,16384] [--k 1,2,4,8,16,32] [--reps R]` — one
//! forward of k tokens on top of a ctx-token cache (a speculative-decoding verify step), with
//! logits for all k rows: ms (median and min over interleaved rounds) and cost relative to the
//! first k.
//!
//! Every command takes `--device auto|metal|cuda|cpu` (auto: Metal, then CUDA, then the CPU,
//! whichever is built in and present).

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::time::Instant;
use tang_compute::{ComputeDevice, CpuDevice};
use tang_llm::{Dtype, Engine, Model};

/// Where the model runs.
#[derive(Clone, Copy, Debug)]
enum Backend {
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "cuda")]
    Cuda,
    Cpu,
}

impl Backend {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "auto" => Self::detect(),
            #[cfg(feature = "metal")]
            "metal" => Self::Metal,
            #[cfg(feature = "cuda")]
            "cuda" => Self::Cuda,
            "cpu" => Self::Cpu,
            other if ["metal", "cuda"].contains(&other) => {
                bail!("this build has no {other} support (rebuild with --features {other})")
            }
            other => bail!("unknown device {other} (auto, metal, cuda or cpu)"),
        })
    }

    /// The first backend that's built in and has a device.
    fn detect() -> Self {
        #[cfg(feature = "metal")]
        if tang_compute::MetalDevice::new().is_some() {
            return Self::Metal;
        }
        #[cfg(feature = "cuda")]
        if new_cuda().is_ok() {
            return Self::Cuda;
        }
        eprintln!("tang-llm: no GPU found, running on the CPU (slow)");
        Self::Cpu
    }
}

#[cfg(feature = "metal")]
fn new_metal() -> Result<tang_compute::MetalDevice> {
    tang_compute::MetalDevice::new().context("no Metal device")
}

/// A CUDA device, or an error when there's no driver (cudarc panics when it can't load
/// libcuda) or no GPU.
#[cfg(feature = "cuda")]
fn new_cuda() -> Result<tang_compute::CudaComputeDevice> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let dev = std::panic::catch_unwind(tang_compute::CudaComputeDevice::new);
    std::panic::set_hook(hook);
    match dev {
        Ok(Ok(dev)) => {
            // TF32 tensor cores for the prefill GEMMs: ~1.1-1.3x prefill on an RTX 3090 at
            // KL ~1e-5 against FP32 (decode's GEMVs don't use cuBLAS). `GAIA_TF32=0` opts out.
            dev.set_tf32(std::env::var("GAIA_TF32").as_deref() != Ok("0"));
            Ok(dev)
        }
        Ok(Err(e)) => bail!("no CUDA device: {e}"),
        Err(_) => bail!("no CUDA device (couldn't load the driver or cuBLAS)"),
    }
}

fn new_cpu() -> Result<CpuDevice> {
    Ok(CpuDevice::new())
}

/// Call `$f(make_device, args...)` with the constructor for `$backend`'s device type.
macro_rules! on_backend {
    ($backend:expr, $f:ident($($arg:expr),*)) => {
        match $backend {
            #[cfg(feature = "metal")]
            Backend::Metal => $f(new_metal, $($arg),*),
            #[cfg(feature = "cuda")]
            Backend::Cuda => $f(new_cuda, $($arg),*),
            Backend::Cpu => $f(new_cpu, $($arg),*),
        }
    };
}

/// Pull `--device X` out of the arguments.
fn take_device(args: &mut Vec<String>) -> Result<Backend> {
    match args.iter().position(|a| a == "--device") {
        Some(i) => {
            let spec = args
                .get(i + 1)
                .context("--device auto|metal|cuda|cpu")?
                .clone();
            args.drain(i..i + 2);
            Backend::parse(&spec)
        }
        None => Ok(Backend::detect()),
    }
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let backend = take_device(&mut args)?;
    let (cmd, dir) = match args.as_slice() {
        [c, d, ..] => (c.as_str(), PathBuf::from(d)),
        _ => bail!("usage: tang-llm <logits|generate> <model-dir> <ids...> [-n N] [--f32]"),
    };
    if args.first().map(String::as_str) == Some("serve") {
        return serve(backend, &args[1..]);
    }
    if matches!(cmd, "bench-prefill" | "bench-verify") {
        let cmd = cmd.to_string();
        return bench(backend, &cmd, &args[1..]);
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
        pixels = Some(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        );
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
    on_backend!(backend, run(cmd, &dir, pixels, n, last, dtype, ids))
}

#[allow(clippy::too_many_arguments)]
fn run<D: ComputeDevice>(
    make: fn() -> Result<D>,
    cmd: &str,
    dir: &std::path::Path,
    pixels: Option<Vec<f32>>,
    n: usize,
    last: Option<usize>,
    dtype: Dtype,
    ids: Vec<u32>,
) -> Result<()> {
    let t = Instant::now();
    let model = Model::load(make()?, dir, 4096, dtype)?;
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

fn serve(backend: Backend, args: &[String]) -> Result<()> {
    let spec = args
        .first()
        .context("usage: tang-llm serve <model> [--host H] [--port P] [--ctx N] [--api-key-file F] [--f32 | --q4]")?
        .clone();
    let (mut port, mut ctx, mut dtype) = (8911u16, 32_768usize, Dtype::Bf16);
    let mut host = "127.0.0.1".to_string();
    // Never on the command line itself, where `ps` would show it.
    let mut key = std::env::var("TANG_API_KEY").ok();
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--host" => host = it.next().context("--host H")?.clone(),
            "--port" => port = it.next().context("--port P")?.parse()?,
            "--ctx" => ctx = it.next().context("--ctx N")?.parse()?,
            "--api-key-file" => {
                let path = it.next().context("--api-key-file F")?;
                key =
                    Some(std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?);
            }
            "--f32" => dtype = Dtype::F32,
            "--q4" => dtype = Dtype::Q4,
            other => bail!("unknown option {other}"),
        }
    }
    let key = key.map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
    let dir = tang_llm::resolve_model(&spec)?;
    let addr = format!("{host}:{port}");
    if key.is_none() && !host.starts_with("127.") && host != "localhost" {
        eprintln!("tang-llm: warning: listening on {host} without an API key; anyone who can reach it can use it");
    }
    on_backend!(backend, serve_on(&addr, spec, dir, ctx, dtype, key))
}

fn serve_on<D: ComputeDevice + 'static>(
    make: fn() -> Result<D>,
    addr: &str,
    name: String,
    dir: PathBuf,
    ctx: usize,
    dtype: Dtype,
    key: Option<String>,
) -> Result<()> {
    tang_llm::server::serve(addr, name, key, move || {
        let t = Instant::now();
        let e = Engine::load(make()?, &dir, ctx, dtype)?;
        eprintln!(
            "tang-llm: loaded {} in {:.1}s ({} ctx)",
            dir.display(),
            t.elapsed().as_secs_f32(),
            e.context_window()
        );
        Ok(e)
    })
}

/// Options for the bench commands.
struct BenchOpts {
    ns: Vec<usize>,
    ks: Vec<usize>,
    reps: usize,
}

fn bench(backend: Backend, cmd: &str, args: &[String]) -> Result<()> {
    let list = |s: &String| -> Result<Vec<usize>> {
        s.split(',').map(|v| Ok(v.trim().parse()?)).collect()
    };
    let spec = args
        .first()
        .context("a model directory or Hugging Face repo id")?;
    let prefill = cmd == "bench-prefill";
    let mut o = BenchOpts {
        ns: if prefill {
            vec![1024, 4096, 9216]
        } else {
            vec![4096, 16384]
        },
        ks: vec![1, 2, 4, 8, 16, 32],
        reps: if prefill { 2 } else { 10 },
    };
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--n" | "--ctx" => o.ns = list(it.next().context("--ctx N,N,...")?)?,
            "--k" => o.ks = list(it.next().context("--k K,K,...")?)?,
            "--reps" => o.reps = it.next().context("--reps R")?.parse()?,
            other => bail!("unknown option {other}"),
        }
    }
    let dir = tang_llm::resolve_model(spec)?;
    eprintln!("{cmd} {} on {backend:?}", dir.display());
    on_backend!(backend, bench_on(&dir, prefill, o))
}

/// Deterministic, tokenizer-free token ids (timing doesn't depend on the text).
fn synthetic_ids(n: usize, vocab: usize) -> Vec<u32> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (1000 + x % (vocab.min(150_000) as u64 - 1000)) as u32
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn bench_on<D: ComputeDevice>(
    make: fn() -> Result<D>,
    dir: &std::path::Path,
    prefill: bool,
    o: BenchOpts,
) -> Result<()> {
    let max_n = *o.ns.iter().max().context("no sizes")?;
    let max_k = if prefill {
        0
    } else {
        *o.ks.iter().max().context("no k")?
    };
    let model = Model::load(make()?, dir, max_n + max_k + 1, Dtype::Bf16)?;
    let vocab = model.cfg.vocab_size;
    let mut cache = model.new_cache();
    // Warm up: kernel compilation, pools, cuBLAS/Metal pipeline setup.
    for m in [1, 4, 16, 512] {
        cache.truncate(0);
        model.forward(&synthetic_ids(m, vocab), &mut cache, false)?;
    }
    let fill = |cache: &mut tang_llm::Cache<D::Buffer>, ids: &[u32]| -> Result<()> {
        cache.truncate(0);
        for chunk in ids.chunks(512) {
            model.forward(chunk, cache, false)?;
        }
        Ok(())
    };
    if prefill {
        println!("| prompt tokens | ms | tok/s |");
        println!("|---:|---:|---:|");
        for &n in &o.ns {
            let ids = synthetic_ids(n, vocab);
            let mut times = Vec::new();
            for _ in 0..o.reps.max(1) {
                let t = Instant::now();
                fill(&mut cache, &ids)?;
                times.push(t.elapsed().as_secs_f64());
            }
            let t = median(times);
            println!("| {n} | {:.0} | {:.0} |", t * 1e3, n as f64 / t);
        }
        return Ok(());
    }
    // Rounds interleave the k values so that load from elsewhere on the machine (a shared
    // GPU) hits every k alike; the median and the minimum are both reported.
    println!("| ctx | k | ms/forward (median) | ms (min) | vs k=1 (median) | vs k=1 (min) |");
    println!("|---:|---:|---:|---:|---:|---:|");
    for &ctx in &o.ns {
        let ids = synthetic_ids(ctx + max_k, vocab);
        fill(&mut cache, &ids[..ctx])?;
        let mut times = vec![Vec::new(); o.ks.len()];
        for r in 0..o.reps.max(1) + 2 {
            for (i, &k) in o.ks.iter().enumerate() {
                cache.truncate(ctx);
                let t = Instant::now();
                let logits = model.forward(&ids[ctx..ctx + k], &mut cache, true)?;
                let dt = t.elapsed().as_secs_f64() * 1e3;
                anyhow::ensure!(logits.len() == k * vocab);
                if r >= 2 {
                    times[i].push(dt);
                }
            }
        }
        let stats: Vec<(f64, f64)> = times
            .into_iter()
            .map(|t| {
                (
                    median(t.clone()),
                    t.into_iter().fold(f64::INFINITY, f64::min),
                )
            })
            .collect();
        let (b_med, b_min) = stats[0];
        for (&k, &(med, min)) in o.ks.iter().zip(&stats) {
            println!(
                "| {ctx} | {k} | {med:.2} | {min:.2} | {:.2} | {:.2} |",
                med / b_med,
                min / b_min
            );
        }
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
