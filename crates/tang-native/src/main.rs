//! `tang-native`: tokenize a corpus into shards, train an FHE-native model
//! or its twin with checkpoints, and evaluate validation perplexity.
//!
//! ```text
//! tang-native tokenize --tokenizer tokenizer.json --out shards/ [--shard-tokens N] [--max-tokens N] FILES…
//! tang-native train --variant native --size pilot --data shards/ --out runs/native \
//!     [--steps N] [--lr X] [--warmup N] [--every N] [--eval-every N] [--eval-batches N]
//!     [--vocab N] [--batch B] [--seq T] [--seed S] [--resume runs/native/ckpt-01000] [--interp]
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tang_native::ckpt::{load_meta, load_safetensors, save_meta, save_safetensors};
use tang_native::data::Shards;
use tang_native::train::{Hyper, Rng, Step, Trainer};
use tang_native::{Config, Variant};

fn usage() -> ! {
    eprintln!("usage: tang-native tokenize|train … (see the crate docs)");
    std::process::exit(2)
}

struct Args {
    flags: BTreeMap<String, String>,
    rest: Vec<String>,
}

impl Args {
    fn parse(v: &[String]) -> Args {
        let (mut flags, mut rest) = (BTreeMap::new(), Vec::new());
        let mut i = 0;
        while i < v.len() {
            if let Some(k) = v[i].strip_prefix("--") {
                let bool_flag = matches!(k, "interp");
                if bool_flag {
                    flags.insert(k.to_string(), "1".to_string());
                } else {
                    let val = v.get(i + 1).cloned().unwrap_or_else(|| usage());
                    flags.insert(k.to_string(), val);
                    i += 1;
                }
            } else {
                rest.push(v[i].clone());
            }
            i += 1;
        }
        Args { flags, rest }
    }
    fn get<T: std::str::FromStr>(&self, k: &str) -> Option<T> {
        self.flags.get(k).map(|v| {
            v.parse().unwrap_or_else(|_| {
                eprintln!("--{k}: can't parse {v}");
                std::process::exit(2)
            })
        })
    }
    fn req(&self, k: &str) -> String {
        self.flags.get(k).cloned().unwrap_or_else(|| {
            eprintln!("--{k} is required");
            std::process::exit(2)
        })
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first() else { usage() };
    let a = Args::parse(&argv[1..]);
    match cmd.as_str() {
        "tokenize" => tokenize(&a),
        "train" => train(&a),
        _ => usage(),
    }
}

#[cfg(feature = "tokenize")]
fn tokenize(a: &Args) {
    let inputs: Vec<PathBuf> = a.rest.iter().map(PathBuf::from).collect();
    let t0 = Instant::now();
    match tang_native::data::tokenize(
        Path::new(&a.req("tokenizer")),
        &inputs,
        Path::new(&a.req("out")),
        a.get("shard-tokens").unwrap_or(100_000_000),
        a.get("max-tokens"),
    ) {
        Ok((train, val)) => eprintln!(
            "tokenized: {train} train + {val} val tokens in {:.1?}",
            t0.elapsed()
        ),
        Err(e) => {
            eprintln!("tokenize: {e}");
            std::process::exit(1)
        }
    }
}

#[cfg(not(feature = "tokenize"))]
fn tokenize(_: &Args) {
    eprintln!("built without the `tokenize` feature");
    std::process::exit(2)
}

/// The two step engines behind one interface (one lives per run, so the
/// size difference between them doesn't matter).
#[allow(clippy::large_enum_variant)]
enum Runner {
    Cpu(Trainer),
    #[cfg(feature = "gpu")]
    Gpu(tang_native::gpu::GpuTrainer),
}

impl Runner {
    fn step_ref(&self) -> &Step {
        match self {
            Runner::Cpu(t) => &t.step,
            #[cfg(feature = "gpu")]
            Runner::Gpu(t) => &t.step,
        }
    }
    fn run(&mut self, x: &[u32], y: &[u32], lr: f64) -> (f64, f64) {
        match self {
            Runner::Cpu(t) => t.step(x, y, lr),
            #[cfg(feature = "gpu")]
            Runner::Gpu(t) => {
                let (l, g) = t.step(x, y, lr);
                (l as f64, g as f64)
            }
        }
    }
    fn t(&self) -> u64 {
        match self {
            Runner::Cpu(t) => t.t,
            #[cfg(feature = "gpu")]
            Runner::Gpu(t) => t.t,
        }
    }
    fn set_t(&mut self, v: u64) {
        match self {
            Runner::Cpu(t) => t.t = v,
            #[cfg(feature = "gpu")]
            Runner::Gpu(t) => t.t = v,
        }
    }
    /// Every parameter and Adam moment by name: `name`, `name.m`, `name.v`.
    fn state(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let s = self.step_ref();
        let read = |id| -> Vec<f32> {
            match self {
                Runner::Cpu(t) => t.values[t.step.input_index(id)]
                    .iter()
                    .map(|&x| x as f32)
                    .collect(),
                #[cfg(feature = "gpu")]
                Runner::Gpu(t) => t.read_input(id),
            }
        };
        let mut out = Vec::new();
        for (p, sl) in s.model.params.iter().zip(&s.slots) {
            out.push((p.name.clone(), p.shape.clone(), read(p.id)));
            out.push((format!("{}.m", p.name), p.shape.clone(), read(sl.m)));
            out.push((format!("{}.v", p.name), p.shape.clone(), read(sl.v)));
        }
        out
    }
    fn load(&mut self, state: &BTreeMap<String, Vec<f32>>) {
        let s = self.step_ref();
        let mut writes = Vec::new();
        for (p, sl) in s.model.params.iter().zip(&s.slots) {
            for (key, id) in [
                (p.name.clone(), p.id),
                (format!("{}.m", p.name), sl.m),
                (format!("{}.v", p.name), sl.v),
            ] {
                let v = state
                    .get(&key)
                    .unwrap_or_else(|| panic!("checkpoint lacks {key}"));
                assert_eq!(v.len(), p.shape.iter().product::<usize>(), "{key}: shape");
                writes.push((id, v.clone()));
            }
        }
        for (id, v) in writes {
            match self {
                Runner::Cpu(t) => {
                    let k = t.step.input_index(id);
                    t.values[k] = v.iter().map(|&x| x as f64).collect();
                }
                #[cfg(feature = "gpu")]
                Runner::Gpu(t) => t.write_input(id, &v),
            }
        }
    }
}

fn make_runner(c: Config, v: Variant, stage2: bool, seed: u64, interp: bool) -> Runner {
    let step = Step::new(c, v, Hyper::default(), stage2);
    let cpu = Trainer::new(step, seed);
    if interp {
        return Runner::Cpu(cpu);
    }
    #[cfg(feature = "gpu")]
    {
        let init: Vec<Vec<f32>> = cpu
            .values
            .iter()
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .collect();
        let step = Step::new(c, v, Hyper::default(), stage2);
        match tang_native::gpu::GpuTrainer::new(step, &init) {
            Ok(g) => {
                eprintln!(
                    "step on {} ({} of {} updates in place)",
                    g.backend(),
                    g.donated(),
                    3 * g.step.slots.len()
                );
                return Runner::Gpu(g);
            }
            Err(e) => eprintln!("{e}; falling back to the interpreter"),
        }
    }
    eprintln!("step on the f64 interpreter (slow: for checks, not training)");
    Runner::Cpu(cpu)
}

/// Linear warmup, then BitNet's two stages for ternary variants (cosine
/// from peak to half, then from 0.4·peak to zero, weight decay off) or one
/// cosine to a tenth for the twin. Returns (lr, stage2).
fn schedule(step: u64, steps: u64, warmup: u64, peak: f64, two_stage: bool) -> (f64, bool) {
    if step < warmup {
        return (peak * (step + 1) as f64 / warmup as f64, false);
    }
    let cos = |frac: f64, hi: f64, lo: f64| {
        lo + (hi - lo) * 0.5 * (1.0 + (std::f64::consts::PI * frac.clamp(0.0, 1.0)).cos())
    };
    if two_stage {
        let mid = warmup + (steps - warmup) / 2;
        if step < mid {
            (
                cos(
                    (step - warmup) as f64 / (mid - warmup).max(1) as f64,
                    peak,
                    0.5 * peak,
                ),
                false,
            )
        } else {
            (
                cos(
                    (step - mid) as f64 / (steps - mid).max(1) as f64,
                    0.4 * peak,
                    0.0,
                ),
                true,
            )
        }
    } else {
        (
            cos(
                (step - warmup) as f64 / (steps - warmup).max(1) as f64,
                peak,
                0.1 * peak,
            ),
            false,
        )
    }
}

fn train(a: &Args) {
    let vname = a.req("variant");
    let v = Variant::parse(&vname).unwrap_or_else(|| {
        eprintln!("--variant: twin, ternary, dyt, relu2 or native");
        std::process::exit(2)
    });
    let vocab: usize = a.get("vocab").unwrap_or(128_256);
    let size = a
        .flags
        .get("size")
        .cloned()
        .unwrap_or_else(|| "pilot".into());
    let mut c = match size.as_str() {
        "pilot" => Config::pilot(vocab),
        "main" => Config::main(vocab),
        "tiny" => Config {
            vocab,
            d: 64,
            layers: 2,
            heads: 4,
            ff: 256,
            seq: 64,
            batch: 4,
        },
        _ => {
            eprintln!("--size: tiny, pilot or main");
            std::process::exit(2)
        }
    };
    if let Some(b) = a.get("batch") {
        c.batch = b;
    }
    if let Some(t) = a.get("seq") {
        c.seq = t;
    }
    let data = PathBuf::from(a.req("data"));
    let out = PathBuf::from(a.req("out"));
    std::fs::create_dir_all(&out).expect("out dir");
    let steps: u64 = a.get("steps").unwrap_or(10_000);
    let warmup: u64 = a.get("warmup").unwrap_or(375);
    let peak: f64 = a.get("lr").unwrap_or(if v.ternary { 1.5e-3 } else { 6e-4 });
    let every: u64 = a.get("every").unwrap_or(1000);
    let eval_every: u64 = a.get("eval-every").unwrap_or(every);
    let eval_batches: u64 = a.get("eval-batches").unwrap_or(20);
    let seed: u64 = a.get("seed").unwrap_or(1);
    let interp = a.flags.contains_key("interp");
    let shards = Shards::open(&data, "train").unwrap_or_else(|e| panic!("{e}"));
    let val = Shards::open(&data, "val").ok();
    eprintln!(
        "{vname} {size}: d {} × {} layers, batch {}×{}, {} train tokens, {steps} steps, peak lr {peak}",
        c.d,
        c.layers,
        c.batch,
        c.seq,
        shards.tokens()
    );

    // start, or resume
    let (mut start, mut stage2, mut rng) = (0u64, false, Rng(seed));
    let mut runner;
    if let Some(r) = a.flags.get("resume") {
        let dir = Path::new(r);
        let meta = load_meta(&dir.join("meta.txt")).expect("meta.txt");
        start = meta["step"].parse().unwrap();
        stage2 = meta.get("stage2").is_some_and(|s| s == "1");
        rng = Rng(meta["rng"].parse().unwrap());
        runner = make_runner(c, v, stage2, seed, interp);
        let mut state = load_safetensors(&dir.join("params.safetensors")).expect("params");
        state.extend(load_safetensors(&dir.join("adam.safetensors")).expect("adam"));
        runner.load(&state);
        runner.set_t(meta["t"].parse().unwrap());
        eprintln!("resumed from {} at step {start}", dir.display());
    } else {
        runner = make_runner(c, v, false, seed, interp);
    }

    let save = |runner: &Runner, step: u64, stage2: bool, rng: &Rng| {
        let dir = out.join(format!("ckpt-{step:06}"));
        std::fs::create_dir_all(&dir).expect("ckpt dir");
        let st = runner.state();
        let (params, adam): (Vec<_>, Vec<_>) = st
            .into_iter()
            .partition(|(n, _, _)| !n.ends_with(".m") && !n.ends_with(".v"));
        save_safetensors(&dir.join("params.safetensors"), &params).expect("save params");
        save_safetensors(&dir.join("adam.safetensors"), &adam).expect("save adam");
        save_meta(
            &dir.join("meta.txt"),
            &[
                ("step", step.to_string()),
                ("t", runner.t().to_string()),
                ("variant", vname.clone()),
                ("size", size.clone()),
                ("stage2", (stage2 as u8).to_string()),
                ("rng", rng.0.to_string()),
                ("config", format!("{c:?}")),
            ],
        )
        .expect("save meta");
        eprintln!("checkpoint {}", dir.display());
    };

    let log_path = out.join("log.csv");
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .expect("log");
    use std::io::Write;
    if start == 0 {
        writeln!(log, "step,loss,grad_norm,lr,tokens_per_s,val_loss").ok();
    }
    let tokens_per_step = c.rows() as f64;
    for step in start..steps {
        let (lr, s2) = schedule(step, steps, warmup, peak, v.ternary);
        if s2 && !stage2 {
            // BitNet's second stage: same values, weight decay off
            let st = runner.state();
            let t = runner.t();
            runner = make_runner(c, v, true, seed, interp);
            runner.load(&st.into_iter().map(|(n, _, x)| (n, x)).collect());
            runner.set_t(t);
            stage2 = true;
            eprintln!("stage 2 at step {step}");
        }
        let (x, y) = shards.sample(&mut rng, c.batch, c.seq).expect("data");
        let t0 = Instant::now();
        let (loss, gn) = runner.run(&x, &y, lr);
        let tps = tokens_per_step / t0.elapsed().as_secs_f64();
        let mut val_loss = String::new();
        if let (Some(vs), true) = (&val, (step + 1) % eval_every == 0) {
            val_loss = format!("{:.5}", evaluate(&runner, c, v, vs, eval_batches));
        }
        if step % 10 == 0 || !val_loss.is_empty() {
            eprintln!("step {step:6}  loss {loss:.4}  |g| {gn:.3}  lr {lr:.2e}  {tps:.0} tok/s {val_loss}");
        }
        writeln!(log, "{step},{loss},{gn},{lr},{tps:.0},{val_loss}").ok();
        if !loss.is_finite() {
            eprintln!("loss is not finite at step {step}; stopping");
            std::process::exit(1);
        }
        if (step + 1) % every == 0 || step + 1 == steps {
            save(&runner, step + 1, stage2, &rng);
        }
    }
}

/// Mean validation loss over the first `batches` deterministic batches.
fn evaluate(runner: &Runner, c: Config, v: Variant, val: &Shards, batches: u64) -> f64 {
    let state = runner.state();
    let model = tang_native::build(c, v);
    let mut sum = 0.0;
    let mut n = 0;
    let batch_of = |i: u64| -> Option<(Vec<u32>, Vec<u32>)> {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for b in 0..c.batch as u64 {
            let w = val.window_at(i * c.batch as u64 + b, c.seq + 1).ok()??;
            x.extend_from_slice(&w[..c.seq]);
            y.extend_from_slice(&w[1..]);
        }
        Some((x, y))
    };
    match runner {
        #[cfg(feature = "gpu")]
        Runner::Gpu(_) => {
            let cpu = Trainer::new(Step::new(c, v, Hyper::default(), false), 0);
            let init: Vec<Vec<f32>> = cpu
                .values
                .iter()
                .map(|v| v.iter().map(|&x| x as f32).collect())
                .collect();
            let model_ev = tang_native::build(c, v);
            // the eval graph's inputs are a prefix of the step's (same build order)
            let n_in = model_ev
                .graph
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, tang_compile::tensor::Op::Input(_)))
                .count();
            let ev = tang_native::gpu::GpuEval::new(model_ev, &init[..n_in]).expect("eval program");
            for (name, _, x) in &state {
                ev.set(name, x);
            }
            for i in 0..batches {
                let Some((x, y)) = batch_of(i) else { break };
                sum += ev.loss(&x, &y) as f64;
                n += 1;
            }
        }
        _ => {
            let n_in = model
                .graph
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, tang_compile::tensor::Op::Input(_)))
                .count();
            let cpu = match runner {
                Runner::Cpu(t) => t,
                #[allow(unreachable_patterns)]
                _ => unreachable!(),
            };
            let mut ins: Vec<Vec<f64>> = cpu.values[..n_in].to_vec();
            let idx = |id: tang_compile::tensor::TensorId| match model.graph.nodes()[id.index()].op
            {
                tang_compile::tensor::Op::Input(k) => k,
                _ => unreachable!(),
            };
            for i in 0..batches {
                let Some((x, y)) = batch_of(i) else { break };
                ins[idx(model.tokens)] = x.iter().map(|&t| t as f64).collect();
                ins[idx(model.targets)] = y.iter().map(|&t| t as f64).collect();
                sum += model.graph.eval(&ins, &[model.loss])[0][0];
                n += 1;
            }
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}
