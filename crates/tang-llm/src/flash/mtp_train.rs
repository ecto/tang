//! Offline three-step recursive MTP fine-tuning on `flash-generate --dump-mtp-train` dumps.
//! Dense block weights (including HC, router and shared expert) train with tang-train Adam;
//! routed experts, main embedding and main head stay frozen. No target model occupies VRAM.
//! Tensor contractions use tang-compute, nonlinear Jacobians use tang-ad. The export preserves
//! every GGUF directory entry and untouched byte; Q8/BF16/F32 updates keep their original type.
mod data;
mod model;
mod tape;
mod weights;
use crate::flash::reference::Hparams;
use anyhow::{bail, ensure, Context, Result};
use std::{
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use tang_compute::ComputeDevice;
use weights::Weights;

struct Args {
    main: PathBuf,
    mtp: PathBuf,
    data: PathBuf,
    out: PathBuf,
    steps: usize,
    seq: usize,
    burn: usize,
    lr: f64,
    beta: f64,
    clip: f64,
    vocab: usize,
    seed: u64,
    cpu: bool,
    eval_every: usize,
    resume: Option<PathBuf>,
    check_forward: bool,
}
fn parse(args: &[String]) -> Result<Args> {
    let usage="flash-mtp-train <main.gguf> <mtp.gguf> --data DIR --out DIR [--steps 100 --seq 32 --burn 8 --lr 1e-5 --beta .8 --clip 1 --vocab 0 --seed 42 --eval-every 20 --resume DIR --cpu]";
    let mut it = args.iter();
    let main = it.next().context(usage)?.into();
    let mtp = it.next().context(usage)?.into();
    let mut a = Args {
        main,
        mtp,
        data: PathBuf::new(),
        out: PathBuf::new(),
        steps: 100,
        seq: 32,
        burn: 8,
        lr: 1e-5,
        beta: 0.8,
        clip: 1.0,
        vocab: 0,
        seed: 42,
        cpu: false,
        eval_every: 20,
        resume: None,
        check_forward: false,
    };
    while let Some(k) = it.next() {
        if k == "--cpu" {
            a.cpu = true;
            continue;
        }
        if k == "--check-forward" {
            a.check_forward = true;
            continue;
        }
        let v = it.next().with_context(|| format!("{k} needs a value"))?;
        match k.as_str() {
            "--data" => a.data = v.into(),
            "--out" => a.out = v.into(),
            "--steps" => a.steps = v.parse()?,
            "--seq" => a.seq = v.parse()?,
            "--burn" => a.burn = v.parse()?,
            "--lr" => a.lr = v.parse()?,
            "--beta" => a.beta = v.parse()?,
            "--clip" => a.clip = v.parse()?,
            "--vocab" => a.vocab = v.parse()?,
            "--seed" => a.seed = v.parse()?,
            "--eval-every" => a.eval_every = v.parse()?,
            "--resume" => a.resume = Some(v.into()),
            _ => bail!("unknown option {k}: {usage}"),
        }
    }
    ensure!(
        !a.data.as_os_str().is_empty() && !a.out.as_os_str().is_empty(),
        "{usage}"
    );
    ensure!(
        a.seq >= 4 && a.burn < a.seq - 3,
        "invalid sequence/burn length"
    );
    ensure!(
        a.eval_every > 0
            && a.lr > 0.0
            && a.lr.is_finite()
            && a.beta > 0.0
            && a.beta <= 1.0
            && a.clip > 0.0
            && a.clip.is_finite(),
        "invalid training hyperparameters"
    );
    Ok(a)
}
pub fn cli(args: &[String]) -> Result<()> {
    let a = parse(args)?;
    if a.cpu {
        return run(&tang_compute::CpuDevice::new(), a);
    }
    #[cfg(feature = "cuda")]
    {
        let dev = tang_compute::CudaComputeDevice::new().map_err(|e| anyhow::anyhow!("{e}"))?;
        dev.set_tf32(false);
        run(&dev, a)
    }
    #[cfg(not(feature = "cuda"))]
    {
        bail!("build with cuda,mtp-train or pass --cpu")
    }
}
fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn run<D: ComputeDevice>(dev: &D, a: Args) -> Result<()> {
    ensure!(
        !a.out.exists(),
        "output directory already exists: {}",
        a.out.display()
    );
    let mut w = Weights::open(&a.main, &a.mtp, a.lr, a.vocab)?;
    let hp = Hparams::from_gguf(&w.mtp)?;
    let seqs = data::discover(&a.data, hp.hc * hp.n_embd)?;
    let eligible: Vec<_> = seqs
        .iter()
        .enumerate()
        .filter(|(_, s)| s.ids.len() >= a.seq)
        .map(|(i, _)| i)
        .collect();
    let valid: Vec<_> = eligible
        .iter()
        .copied()
        .filter(|&i| prompt_number(&seqs[i].path) % 10 == 0)
        .collect();
    let train: Vec<_> = eligible
        .iter()
        .copied()
        .filter(|&i| prompt_number(&seqs[i].path) % 10 != 0)
        .collect();
    ensure!(
        !train.is_empty() && !valid.is_empty(),
        "need completed training and held-out prompts (pNNN % 10 == 0 held out)"
    );
    if a.check_forward {
        return check_forward(dev, &mut w, &seqs[valid[0]], &hp, &a);
    }
    let mut rng = a.seed;
    let mut offset = 0;
    if let Some(p) = &a.resume {
        let restored = restore(p, &mut w)?;
        offset = restored.0;
        rng = restored.1;
    }
    std::fs::create_dir_all(&a.out)?;
    let manifest = serde_json::json!({"main":a.main,"mtp":a.mtp,"data":a.data,"steps":a.steps,"seq":a.seq,"burn":a.burn,"lr":a.lr,"beta":a.beta,"clip":a.clip,"vocab":a.vocab,"seed":a.seed,"resume":a.resume,"start_step":offset,"train":train.iter().map(|&i|&seqs[i].path).collect::<Vec<_>>(),"valid":valid.iter().map(|&i|&seqs[i].path).collect::<Vec<_>>(),"trainable":w.params.keys().collect::<Vec<_>>()});
    std::fs::write(
        a.out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!("MTP train: {} train / {} held-out prompts; {} parameters; frozen Q4 routed experts, main embedding/head; {}-row windows, {} burn-in; CPU={} start={offset}",train.len(),valid.len(),w.params.values().map(|p|p.data.numel()).sum::<usize>(),a.seq,a.burn,a.cpu);
    let mut log = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(a.out.join("metrics.jsonl"))?;
    evaluate(dev, &mut w, &seqs, &valid, &hp, &a, offset, &mut log)?;
    for local in 1..=a.steps {
        let step = offset + local;
        let s = &seqs[train[random(&mut rng) as usize % train.len()]];
        let start = random(&mut rng) as usize % (s.ids.len() - a.seq + 1);
        let b = s.batch(start, a.seq)?;
        let now = Instant::now();
        let mut t = tape::Tape::new(dev);
        let leaves = w.leafs(&mut t);
        let logits = model::forward(&mut t, &mut w, &leaves, &b, &hp)?;
        let (loss, seeds, hits, totals) = model::objective(&t, &w, &logits, &b, a.beta, a.burn)?;
        ensure!(loss.iter().all(|v| v.is_finite()), "nonfinite loss");
        let grads = t.backward(&seeds);
        drop(t);
        let norm = w.update(&leaves, grads, a.clip)?;
        dev.pool_clear();
        let metric = serde_json::json!({"kind":"train","step":step,"prompt":s.path,"start":start,"ce":loss,"teacher_hits":hits,"teacher_total":totals,"grad_norm":norm,"seconds":now.elapsed().as_secs_f64()});
        writeln!(log, "{metric}")?;
        log.flush()?;
        eprintln!(
            "step {step}: CE {:.4}/{:.4}/{:.4}, grad {norm:.3}, {:.2}s",
            loss[0],
            loss[1],
            loss[2],
            now.elapsed().as_secs_f64()
        );
        if local % a.eval_every == 0 || local == a.steps {
            evaluate(dev, &mut w, &seqs, &valid, &hp, &a, step, &mut log)?;
            let dir = a.out.join(format!("step-{step:06}"));
            std::fs::create_dir(&dir)?;
            checkpoint(&dir, &w, step, rng)?;
            w.export(&dir.join("mtp.gguf"))?;
        }
    }
    Ok(())
}
fn prompt_number(p: &Path) -> usize {
    p.file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_prefix('p'))
        .and_then(|n| n.parse().ok())
        .unwrap_or(usize::MAX)
}
fn evaluate<D: ComputeDevice>(
    dev: &D,
    w: &mut Weights,
    seqs: &[data::Sequence],
    valid: &[usize],
    hp: &Hparams,
    a: &Args,
    step: usize,
    log: &mut FileLike,
) -> Result<()> {
    let mut loss = [0.0; 3];
    let mut hits = [0; 3];
    let mut total = [0; 3];
    // Fixed slices per prompt, never random windows from training prompts. Report teacher-token
    // recursive accuracy as a diagnostic; actual greedy draft acceptance is measured by the engine.
    for &i in valid {
        let s = &seqs[i];
        let start = (s.ids.len() - a.seq) / 2;
        let b = s.batch(start, a.seq)?;
        let mut t = tape::Tape::new(dev);
        let leaves = w.leafs(&mut t);
        let lg = model::forward(&mut t, w, &leaves, &b, hp)?;
        let (l, _, h, n) = model::objective(&t, w, &lg, &b, a.beta, a.burn)?;
        for d in 0..3 {
            loss[d] += l[d] * n[d] as f64;
            hits[d] += h[d];
            total[d] += n[d];
        }
        drop(t);
        dev.pool_clear();
    }
    for d in 0..3 {
        loss[d] /= total[d] as f64;
    }
    let metric = serde_json::json!({"kind":"valid","step":step,"ce":loss,"teacher_hits":hits,"teacher_total":total});
    writeln!(log, "{metric}")?;
    log.flush()?;
    eprintln!("valid {step}: {metric}");
    Ok(())
}
type FileLike = std::fs::File;
fn checkpoint(dir: &Path, w: &Weights, step: usize, rng: u64) -> Result<()> {
    let (mut f, mut meta) = (
        BufWriter::with_capacity(1 << 20, std::fs::File::create(dir.join("state.bin"))?),
        Vec::new(),
    );
    for (n, p) in &w.params {
        meta.push((n, p.data.numel()));
        for v in p.data.data() {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    let (m, v, t) = w.optimizer.state_vecs();
    for vectors in [m, v] {
        for values in vectors {
            for x in values {
                f.write_all(&x.to_le_bytes())?;
            }
        }
    }
    f.flush()?;
    f.get_ref().sync_all()?;
    std::fs::write(
        dir.join("state.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"version":1,"step":step,"rng":rng,"adam_step":t,"params":meta}),
        )?,
    )?;
    Ok(())
}
fn restore(dir: &Path, w: &mut Weights) -> Result<(usize, u64)> {
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("state.json"))?)?;
    ensure!(meta["version"] == 1, "checkpoint version");
    let shape: Vec<(String, usize)> = serde_json::from_value(meta["params"].clone())?;
    ensure!(
        shape
            == w.params
                .iter()
                .map(|(n, p)| (n.clone(), p.data.numel()))
                .collect::<Vec<_>>(),
        "checkpoint parameter geometry"
    );
    let mut f = BufReader::with_capacity(1 << 20, std::fs::File::open(dir.join("state.bin"))?);
    for (n, p) in &mut w.params {
        for x in p.data.data_mut() {
            let mut b = [0; 4];
            f.read_exact(&mut b)?;
            *x = f32::from_le_bytes(b);
        }
        ensure!(
            p.data.data().iter().all(|v| v.is_finite()),
            "nonfinite restored weight"
        );
        w.dense.get_mut(n).unwrap().value = std::sync::Arc::new(p.data.data().to_vec());
    }
    let mut states = Vec::new();
    for _ in 0..2 {
        let mut s = Vec::new();
        for (_, n) in &shape {
            let mut v = Vec::with_capacity(*n);
            for _ in 0..*n {
                let mut b = [0; 8];
                f.read_exact(&mut b)?;
                v.push(f64::from_le_bytes(b));
            }
            ensure!(v.iter().all(|v| v.is_finite()), "nonfinite Adam state");
            s.push(v);
        }
        states.push(s);
    }
    ensure!(f.read(&mut [0u8; 1])? == 0, "trailing checkpoint bytes");
    let v = states.pop().unwrap();
    let m = states.pop().unwrap();
    w.optimizer.load_state_vecs(
        m,
        v,
        meta["adam_step"].as_u64().context("Adam step")? as usize,
    );
    Ok((
        meta["step"].as_u64().context("step")? as usize,
        meta["rng"].as_u64().context("rng")?,
    ))
}

fn check_forward<D: ComputeDevice>(
    dev: &D,
    w: &mut Weights,
    s: &data::Sequence,
    hp: &Hparams,
    a: &Args,
) -> Result<()> {
    ensure!(a.vocab == 0, "oracle check needs full vocabulary");
    w.expert_q4 = false;
    let b = s.batch(0, 4)?;
    let mut t = tape::Tape::new(dev);
    let leaves = w.leafs(&mut t);
    let out = model::forward_full(&mut t, w, &leaves, &b, hp)?;
    let main = crate::flash::reference::FlashRef::open(&a.main)?;
    let mut mtp = crate::flash::mtp::Mtp::open(&a.mtp)?;
    mtp.main_head = true;
    let oracle = crate::flash::mtp::training_oracle(&main, &mtp, &b.h, &b.tokens, &b.pos)?;
    for (d, (o, (r, top))) in out.iter().zip(oracle).enumerate() {
        let got = t.data(o.r);
        let max = got
            .iter()
            .zip(&r)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let rms = (got
            .iter()
            .zip(&r)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / r.len() as f64)
            .sqrt();
        let scale = (r.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / r.len() as f64).sqrt();
        eprintln!(
            "oracle depth {}: residual max_abs {max:.6}, relative RMS {:.6}",
            d + 1,
            rms / scale.max(1e-9)
        );
        ensure!(
            rms / scale.max(1e-9) < 0.002,
            "recursive residual parity failed"
        );
        for (row, expected) in t.data(o.logits).chunks(w.vocab.len()).zip(top) {
            let j = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                .unwrap()
                .0;
            ensure!(
                w.vocab[j] == expected.0,
                "oracle top token differs at depth {}",
                d + 1
            );
        }
    }
    eprintln!("recursive CPU oracle parity: PASS");
    Ok(())
}
