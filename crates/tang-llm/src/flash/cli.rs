//! `tang-llm flash-generate`, `flash-parity` and `flash-bench`: the CUDA engine from the
//! command line.

use super::engine::{Engine, ExpertMode, Opts, Probe, WinStats};
use super::reference::top_logprobs;
use anyhow::{bail, ensure, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tang_compute::flash::shape::*;

struct Args {
    path: PathBuf,
    ids: Vec<u32>,
    prompt: Option<String>,
    n: usize,
    opts: Opts,
    eager: bool,
    chunk: usize,
    dump_routing: Option<PathBuf>,
    reference: Option<PathBuf>,
    dump: Option<PathBuf>,
    ctx: Option<usize>,
    quiet: bool,
    save: Option<PathBuf>,
}

fn read_ids(f: &str) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(f).with_context(|| format!("reading {f}"))?;
    s.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']')
        .filter(|w| !w.is_empty())
        .map(|w| w.parse().with_context(|| format!("{f}: bad id {w:?}")))
        .collect()
}

fn parse(args: &[String]) -> Result<Args> {
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context("first argument: the first GGUF shard")?);
    let mut a = Args {
        path,
        ids: Vec::new(),
        prompt: None,
        n: 32,
        opts: Opts::default(),
        eager: false,
        chunk: 8,
        dump_routing: None,
        reference: None,
        dump: None,
        ctx: None,
        quiet: false,
        save: None,
    };
    while let Some(x) = it.next() {
        let mut val = || it.next().with_context(|| format!("{x} needs a value"));
        match x.as_str() {
            "--prompt-ids" => {
                for w in val()?.split(|c: char| c.is_whitespace() || c == ',') {
                    if !w.is_empty() {
                        a.ids.push(w.parse()?);
                    }
                }
            }
            "--ids-file" => a.ids = read_ids(val()?)?,
            "--prompt" => a.prompt = Some(val()?.clone()),
            "-n" => a.n = val()?.parse()?,
            "--max-ctx" => a.opts.max_ctx = val()?.parse()?,
            "--ctx" => a.ctx = Some(val()?.parse()?),
            "--slots" => a.opts.slots = Some(val()?.parse()?),
            "--reserve-mb" => a.opts.reserve_mb = val()?.parse()?,
            "--mtp-mb" => a.opts.mtp_mb = val()?.parse()?,
            "--profile" => a.opts.profile = Some(PathBuf::from(val()?)),
            "--dump-routing" => a.dump_routing = Some(PathBuf::from(val()?)),
            "--all-cpu" => a.opts.experts = ExpertMode::AllCpu,
            "--no-adapt" => a.opts.adapt = false,
            "--split" => a.opts.split = true,
            "--eager" => a.eager = true,
            "--chunk" => a.chunk = val()?.parse()?,
            "--ref" => a.reference = Some(PathBuf::from(val()?)),
            "--dump" => a.dump = Some(PathBuf::from(val()?)),
            "-q" => a.quiet = true,
            "--save" => a.save = Some(PathBuf::from(val()?)),
            s => bail!("unknown argument {s}"),
        }
    }
    Ok(a)
}

fn load(a: &Args) -> Result<Engine> {
    let mut e = Engine::load(&a.path, a.opts.clone())?;
    e.use_graphs = !a.eager;
    eprintln!("{}", e.load_report);
    Ok(e)
}

/// Byte-level BPE piece → text (GPT-2 byte map), for printing.
pub fn detok(vocab: &[String], ids: &[u32]) -> String {
    let mut inv = [0u8; 512];
    let mut n = 0u32;
    let mut map = std::collections::HashMap::new();
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable {
            b
        } else {
            n += 1;
            255 + n
        };
        map.insert(char::from_u32(c).unwrap(), b as u8);
        if (c as usize) < inv.len() {
            inv[c as usize] = b as u8;
        }
    }
    let mut bytes = Vec::new();
    for &i in ids {
        let piece = vocab.get(i as usize).map(String::as_str).unwrap_or("");
        if piece.starts_with("<|") && piece.ends_with("|>") {
            bytes.extend_from_slice(piece.as_bytes());
            continue;
        }
        for ch in piece.chars() {
            match map.get(&ch) {
                Some(&b) => bytes.push(b),
                None => bytes.extend_from_slice(ch.to_string().as_bytes()),
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn prompt_ids(a: &Args) -> Result<Vec<u32>> {
    if let Some(p) = &a.prompt {
        return super::tokenize::encode_chat(&a.path, p);
    }
    ensure!(!a.ids.is_empty(), "no prompt (--prompt-ids, --ids-file or --prompt)");
    Ok(a.ids.clone())
}

fn split_line(s: &WinStats) -> String {
    format!(
        "T={} wall {:.2} ms (host prep {:.2}) | GPU {:.2} ms, waiting for plan {:.2}, for CPU rows {:.2} | host plan {:.2}, CPU experts {:.2} | routed {} distinct {} missed {} ({:.1}% hit) swaps {}",
        s.t,
        s.wall_ms,
        s.host_prep_ms,
        s.gpu_ms,
        s.gpu_wait_a_ms,
        s.gpu_wait_b_ms,
        s.plan_ms,
        s.cpu_ms,
        s.routed,
        s.distinct,
        s.missed,
        100.0 * (1.0 - s.missed as f64 / s.distinct.max(1) as f64),
        s.swaps
    )
}

/// Decode `n` greedy tokens after `ids`; returns (generated, per-step stats, decode seconds).
fn decode(e: &mut Engine, ids: &[u32], n: usize, chunk: usize) -> Result<(Vec<u32>, Vec<WinStats>, f64, f64)> {
    e.reset();
    let t0 = Instant::now();
    let mut next = e.prefill(ids, chunk, None)?;
    let prefill_s = t0.elapsed().as_secs_f64();
    let mut out = vec![next];
    let mut stats = Vec::with_capacity(n);
    // The first T=1 window captures its graph; time from the second.
    next = e.step(next)?;
    out.push(next);
    stats.push(e.last);
    let t1 = Instant::now();
    for _ in 2..n {
        next = e.step(next)?;
        out.push(next);
        stats.push(e.last);
    }
    let secs = t1.elapsed().as_secs_f64();
    out.truncate(n);
    Ok((out, stats, prefill_s, secs))
}

fn mean(stats: &[WinStats]) -> WinStats {
    let n = stats.len().max(1) as f64;
    let mut m = WinStats {
        t: stats.first().map_or(1, |s| s.t),
        ..Default::default()
    };
    for s in stats {
        m.wall_ms += s.wall_ms / n;
        m.host_prep_ms += s.host_prep_ms / n;
        m.plan_ms += s.plan_ms / n;
        m.cpu_ms += s.cpu_ms / n;
        m.gpu_wait_a_ms += s.gpu_wait_a_ms / n;
        m.gpu_wait_b_ms += s.gpu_wait_b_ms / n;
        m.gpu_ms += s.gpu_ms / n;
        m.routed += s.routed;
        m.distinct += s.distinct;
        m.missed += s.missed;
        m.swaps += s.swaps;
    }
    m
}

pub fn generate(args: &[String]) -> Result<()> {
    let a = parse(args)?;
    let ids = prompt_ids(&a)?;
    let mut e = load(&a)?;
    let (out, stats, prefill_s, secs) = decode(&mut e, &ids, a.n, a.chunk)?;
    let vocab = e.vocab()?;
    println!("prompt: {} tokens, prefill {:.2} s ({:.0} tok/s)", ids.len(), prefill_s, ids.len() as f64 / prefill_s);
    println!("ids: {}", out.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
    println!("text: {}", detok(&vocab, &out));
    let timed = stats.len().saturating_sub(1);
    println!(
        "decode: {} tokens in {:.3} s = {:.1} tok/s (T=1, graphs {})",
        timed,
        secs,
        timed as f64 / secs,
        e.use_graphs
    );
    println!("mean window: {}", split_line(&mean(&stats[1.min(stats.len())..])));
    if let Some(cs) = e.cache_stats() {
        println!("cache: hit rate {:.3} over the run ({} hits, {} misses, {} swaps)", cs.hit_rate(), cs.hits, cs.misses, cs.swaps);
    }
    let (h, r) = e.ngram_stats();
    println!("n-gram rows: {r} reads, {h} cache hits");
    if let Some(p) = &a.dump_routing {
        e.save_routing(p)?;
        println!("routing counts -> {}", p.display());
    }
    Ok(())
}

/// Compare the engine's logits with a `flash-ref` JSONL (top-K logprobs per position) over a
/// prompt, and, with `--dump DIR` (a `flash-ref --dump` of the same ids), the last position's
/// per-layer intermediates.
pub fn parity(args: &[String]) -> Result<()> {
    let a = parse(args)?;
    let ids = prompt_ids(&a)?;
    let mut e = load(&a)?;
    e.reset();
    let refs: Vec<(usize, Vec<(u32, f64)>)> = match &a.reference {
        Some(p) => std::fs::read_to_string(p)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| -> Result<(usize, Vec<(u32, f64)>)> {
                let v: serde_json::Value = serde_json::from_str(l)?;
                let pos = v["pos"].as_u64().context("pos")? as usize;
                let top = v["top"]
                    .as_array()
                    .context("top")?
                    .iter()
                    .map(|p| (p[0].as_u64().unwrap() as u32, p[1].as_f64().unwrap()))
                    .collect();
                Ok((pos, top))
            })
            .collect::<Result<_>>()?,
        None => Vec::new(),
    };
    let want: std::collections::HashMap<usize, &Vec<(u32, f64)>> =
        refs.iter().map(|(p, t)| (*p, t)).collect();
    let (mut n, mut agree, mut kl_sum) = (0usize, 0usize, 0f64);
    let mut kls = Vec::new();
    let mut misses = Vec::new();
    let mut saved = String::new();
    let save = a.save.is_some();
    let mut cb = |pos0: usize, t: usize, logits: &[f32]| {
        let v = logits.len() / t;
        for i in 0..t {
            let pos = pos0 + i;
            if save {
                let top = top_logprobs(&logits[i * v..(i + 1) * v], 40);
                let s: Vec<String> = top.iter().map(|(i, lp)| format!("[{i},{lp:.6}]")).collect();
                saved += &format!("{{\"pos\":{pos},\"top\":[{}]}}\n", s.join(","));
            }
            let Some(r) = want.get(&pos) else { continue };
            let lg = &logits[i * v..(i + 1) * v];
            let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = mx + lg.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln();
            let mine = top_logprobs(lg, 1)[0].0;
            let (mut kl, mut pr, mut qr) = (0f64, 0f64, 0f64);
            for &(id, lp) in r.iter() {
                let p = lp.exp();
                let lq = lg[id as usize] as f64 - lse;
                kl += p * (lp - lq);
                pr += p;
                qr += lq.exp();
            }
            let (pr, qr) = ((1.0 - pr).max(1e-12), (1.0 - qr).max(1e-12));
            kl += pr * (pr.ln() - qr.ln());
            n += 1;
            if mine == r[0].0 {
                agree += 1;
            } else {
                misses.push((pos, r[0].0, mine, r[0].1, r.get(1).map_or(0.0, |x| x.1)));
            }
            kl_sum += kl;
            kls.push(kl);
        }
    };
    let t0 = Instant::now();
    let probe_last = a.dump.is_some();
    let total = ids.len();
    let body = if probe_last { total - 1 } else { total };
    if body > 0 {
        e.prefill(&ids[..body], a.chunk, Some(&mut cb))?;
    }
    let mut probe = Probe::default();
    if probe_last {
        e.tokens.push(ids[total - 1]);
        e.window(total - 1, 1, Some(&mut probe))?;
        cb(total - 1, 1, &e.logits(1));
    }
    let secs = t0.elapsed().as_secs_f64();
    if let Some(p) = &a.save {
        std::fs::write(p, &saved)?;
    }
    kls.sort_by(|a, b| a.total_cmp(b));
    if n > 0 {
        println!(
            "logits vs reference: {n} positions, top-1 agree {agree}/{n} ({:.1}%), KL mean {:.4} median {:.5} p99 {:.4} max {:.4}  ({:.1} s, chunk {})",
            100.0 * agree as f64 / n as f64,
            kl_sum / n as f64,
            kls[n / 2],
            kls[(n * 99 / 100).min(n - 1)],
            kls[n - 1],
            secs,
            a.chunk
        );
        for (pos, want, got, lp0, lp1) in misses.iter().take(12) {
            println!("  top-1 differs at {pos}: ref {want} ({lp0:.3}, runner-up {lp1:.3}), engine {got}");
        }
    }
    if let Some(cs) = e.cache_stats() {
        println!("cache: hit rate {:.3} ({} hits, {} misses)", cs.hit_rate(), cs.hits, cs.misses);
    }
    if let Some(dir) = &a.dump {
        compare_dump(&e, dir, &probe)?;
    }
    if let Some(p) = &a.dump_routing {
        e.save_routing(p)?;
    }
    Ok(())
}

fn read_f32(dir: &Path, name: &str) -> Result<Vec<f32>> {
    let b = std::fs::read(dir.join(format!("{name}.f32")))?;
    Ok(b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn read_u32(dir: &Path, name: &str) -> Result<Vec<u32>> {
    let b = std::fs::read(dir.join(format!("{name}.u32")))?;
    Ok(b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((x - y) as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(1e-30)).sqrt()
}

fn compare_dump(e: &Engine, dir: &Path, p: &Probe) -> Result<()> {
    println!("per layer, last position (relative L2 error vs flash-ref; router overlap of 10; QSA selection):");
    for l in 0..e.hp.n_layer {
        let pm = read_f32(dir, &format!("L{l:02}.post_mixer"))?;
        let po = read_f32(dir, &format!("L{l:02}.post_moe"))?;
        let mo = read_f32(dir, &format!("L{l:02}.mixer_out"))?;
        let eo = read_f32(dir, &format!("L{l:02}.moe_out"))?;
        let rid = read_u32(dir, &format!("L{l:02}.router_ids"))?;
        let overlap = p.router_ids[l].iter().filter(|x| rid.contains(x)).count();
        let order = p.router_ids[l] == rid;
        let sel = match &p.qsa_sel[l] {
            Some(s) => {
                let want = read_u32(dir, &format!("L{l:02}.qsa_selected")).unwrap_or_default();
                let same = *s == want;
                let common = s.iter().filter(|c| want.binary_search(c).is_ok()).count();
                format!(" qsa sel {} ({} of {} common)", if same { "identical" } else { "DIFFERS" }, common, want.len())
            }
            None => String::new(),
        };
        println!(
            "  L{l:02}: mixer_out {:.4} post_mixer {:.4} moe_out {:.4} post_moe {:.4} router {overlap}/10{}{sel}",
            rel(&p.mixer_out[l], &mo),
            rel(&p.post_mixer[l], &pm),
            rel(&p.moe_out[l], &eo),
            rel(&p.post_moe[l], &po),
            if order { " (same order)" } else { "" },
        );
    }
    let fx = read_f32(dir, "final_x")?;
    println!("  final_x {:.4}", rel(&p.final_x, &fx));
    let lg = read_f32(dir, "logits")?;
    println!("  logits {:.4}", rel(&e.logits(1), &lg));
    Ok(())
}

/// Decode benchmark: prefill `--ctx` tokens of the prompt (cycled if short), then `-n` greedy
/// T=1 tokens; prints tok/s and the per-window split.
pub fn bench(args: &[String]) -> Result<()> {
    let mut a = parse(args)?;
    a.opts.split = true;
    let mut ids = prompt_ids(&a)?;
    if let Some(c) = a.ctx {
        let base = ids.clone();
        while ids.len() < c {
            ids.extend_from_slice(&base);
        }
        ids.truncate(c);
    }
    if a.opts.max_ctx < ids.len() + a.n + 8 {
        a.opts.max_ctx = (ids.len() + a.n + 8).next_multiple_of(1024);
    }
    let mut e = load(&a)?;
    let (out, stats, prefill_s, secs) = decode(&mut e, &ids, a.n, a.chunk)?;
    let timed = &stats[1.min(stats.len())..];
    let m = mean(timed);
    let tps = timed.len() as f64 / secs;
    let walls: Vec<f64> = {
        let mut w: Vec<f64> = timed.iter().map(|s| s.wall_ms).collect();
        w.sort_by(|a, b| a.total_cmp(b));
        w
    };
    println!("flash-bench: context {} + {} new tokens, greedy, T=1, graphs {}", ids.len(), a.n, e.use_graphs);
    println!("  prefill            {:8.2} s ({:.0} tok/s, chunk {})", prefill_s, ids.len() as f64 / prefill_s, a.chunk);
    println!("  decode             {:8.1} tok/s ({} tokens in {:.3} s)", tps, timed.len(), secs);
    println!("  window wall        {:8.3} ms mean, {:.3} median, {:.3} p90", m.wall_ms, walls[walls.len() / 2], walls[walls.len() * 9 / 10]);
    println!("  GPU (graph)        {:8.3} ms", m.gpu_ms);
    println!("    waiting for plan {:8.3} ms  (handoff A)", m.gpu_wait_a_ms);
    println!("    waiting for CPU  {:8.3} ms  (CPU-miss time exposed)", m.gpu_wait_b_ms);
    println!("  host prep          {:8.3} ms  (embedding + n-gram rows)", m.host_prep_ms);
    println!("  host plan          {:8.3} ms", m.plan_ms);
    println!("  host CPU experts   {:8.3} ms", m.cpu_ms);
    println!(
        "  experts            {:.1} distinct/token, {:.2} missed/token, hit rate {:.3}, swaps {}",
        m.distinct as f64 / timed.len() as f64,
        m.missed as f64 / timed.len() as f64,
        1.0 - m.missed as f64 / m.distinct.max(1) as f64,
        m.swaps
    );
    let (h, r) = e.ngram_stats();
    println!("  n-gram rows        {r} reads, {h} cache hits");
    println!("  first tokens: {}", out.iter().take(16).map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
    if let Some(p) = &a.dump_routing {
        e.save_routing(p)?;
    }
    let _ = HIDDEN;
    Ok(())
}
