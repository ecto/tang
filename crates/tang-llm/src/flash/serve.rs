//! `tang-llm serve <first shard.gguf> --mtp <mtp.gguf>`: the Flash-Next engine behind the
//! OpenAI-compatible server (`crate::server`), for agents like frog.
//!
//! - **Prompt.** The GGUF's own chat template. An assistant message this server generated is
//!   spliced back in as the exact tokens it generated (reasoning included, as the template's
//!   `preserve_thinking` wants), found by its visible text and tool calls; so a follow-up's
//!   prompt extends the live sequence token for token.
//! - **Prefix reuse.** The engine keeps the live sequence. A prompt that extends it prefills
//!   only the new tokens; otherwise the longest saved running state (`RunningState`, ~118 MB
//!   in host RAM) that the prompt extends is restored: taken at the end of the system prompt
//!   (pinned, and saved to disk so a restart doesn't pay for it again), at each prompt's last
//!   turn boundary, and at the end of each request.
//! - **Decode.** MTP drafts, priced per width like `flash-generate --draft mtp`, verified with
//!   exact match, so the output is the engine's sampling at `Philox(seed, position)` whatever
//!   the drafts. Temperature only (no top-k/top-p); greedy unless the request names one.
//! - **Tool calls.** Qwen XML (`chat::parse_xml_call`), typed by the request's schemas.

use super::engine::{Engine, Opts, Positional, RunningState};
use super::tokenize::FlashTokenizer;
use crate::chat::{Parser, Piece};
use crate::engine::{Finish, Request, ThinkBudget, Usage, WRAP_UP};
use crate::server::Backend;
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tang_compute::flash::MAX_T;

/// What sampling a request gets for the parameters it doesn't name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SamplingMode {
    /// Temperature 0 (greedy); top-k/top-p/penalty only when asked for.
    Greedy,
    /// The model card's: thinking temperature 1.0, top-p 0.95, top-k 20; not thinking 0.7,
    /// 0.8, 20 and a presence penalty of 1.5.
    ModelCard,
}

pub struct Settings {
    pub sampling: SamplingMode,
    pub gguf: PathBuf,
    pub mtp: Option<PathBuf>,
    pub ctx: usize,
    /// Temperature for requests that don't name one (default 0: greedy).
    pub temp: f32,
    /// Saved states kept in RAM (besides the pinned system-prompt ones), and their total size.
    pub snapshots: usize,
    pub snapshot_bytes: usize,
    /// Where pinned (system prompt) states are saved; `None` keeps them in RAM only.
    pub disk: Option<PathBuf>,
    /// Write each request's prompt and output ids here (`req-N.ids`, `req-N.out`).
    pub dump: Option<PathBuf>,
    /// Chat completion bodies (JSON files) to prefill at start, so their tools block and
    /// system prompt states are saved before the first real request.
    pub warm: Vec<PathBuf>,
}

/// A saved running state and the tokens it follows.
struct Snap {
    tokens: Vec<u32>,
    state: RunningState,
    /// Positions `0..tokens.len()` of the positional state (another sequence may have
    /// overwritten them on the device since), in segments shared with the states it extends.
    pos: Vec<Arc<Positional>>,
    pinned: bool,
    used: u64,
}

/// A completion this server generated, to splice back into later prompts.
struct Memo {
    content: String,
    calls: Vec<(String, Value)>,
    /// The generation prompt's tail after `<|im_start|>assistant\n` (`<think>\n`, or the
    /// closed empty block with thinking off).
    tail: String,
    ids: Vec<u32>,
}

/// Draft pricing (the `flash-generate --draft mtp` policy): expected tokens per ms of window
/// (EMA per width) and drafting, with acceptance calibrated by draft-probability bucket.
struct Policy {
    wcost: Vec<f64>,
    mtp_cost: f64,
    calib: Vec<(f64, f64)>,
    think_room: usize,
    gate: Option<f32>,
}

impl Policy {
    fn new() -> Self {
        Policy {
            wcost: vec![10.6, 10.6, 12.9, 15.0, 17.1, 19.2, 21.3, 23.4, 25.5],
            mtp_cost: 2.5,
            calib: (0..10)
                .map(|b| {
                    let p = (b as f64 + 0.5) / 10.0;
                    let a = if p >= 0.9 { 0.985 } else if p >= 0.5 { 0.6 + (p - 0.5) * 0.8 } else { 0.3 + p * 0.6 };
                    (4.0 * a, 4.0)
                })
                .collect(),
            think_room: std::env::var("TANG_FLASH_THINK_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(MAX_T),
            gate: std::env::var("TANG_FLASH_GATE").ok().and_then(|v| v.parse().ok()),
        }
    }

    /// Drafts (and their probabilities) from the MTP chain, at most `room`.
    fn choose(&self, chain: &[(u32, f32)], room: usize) -> (Vec<u32>, Vec<f32>) {
        let n = match self.gate {
            Some(g) => chain.iter().take(room).take_while(|x| x.1 >= g).count(),
            None => {
                let mut best = (0usize, 1.0 / (self.wcost[1] + self.mtp_cost));
                let (mut cum, mut exp) = (1.0f64, 1.0f64);
                for (k, &(_, p)) in chain.iter().enumerate().take(room) {
                    let b = ((p * 10.0) as usize).min(9);
                    cum *= self.calib[b].0 / self.calib[b].1;
                    exp += cum;
                    let rate = exp / (self.wcost[k + 2] + self.mtp_cost);
                    if rate > best.1 {
                        best = (k + 1, rate);
                    }
                }
                best.0
            }
        };
        (chain.iter().take(n).map(|x| x.0).collect(), chain.iter().take(n).map(|x| x.1).collect())
    }

    fn observe(&mut self, probs: &[f32], accepted: usize, width: usize, wall_ms: f64, mtp_ms: f64) {
        for (j, &p) in probs.iter().enumerate().take(accepted + 1) {
            let b = ((p * 10.0) as usize).min(9);
            self.calib[b].1 += 1.0;
            if j < accepted {
                self.calib[b].0 += 1.0;
            }
        }
        self.wcost[width] = 0.9 * self.wcost[width] + 0.1 * wall_ms;
        self.mtp_cost = 0.9 * self.mtp_cost + 0.1 * mtp_ms;
    }
}

/// Incremental detokenization: decode a small window, emit what's new, hold back broken UTF-8.
#[derive(Default)]
struct Detok {
    prefix: usize,
    read: usize,
}

impl Detok {
    fn push(&mut self, tok: &FlashTokenizer, out: &[u32]) -> Result<Option<String>> {
        let before = tok.decode(&out[self.prefix..self.read])?;
        let after = tok.decode(&out[self.prefix..])?;
        if after.len() > before.len() && !after.ends_with('\u{fffd}') {
            let new = after[before.len()..].to_string();
            self.prefix = self.read;
            self.read = out.len();
            return Ok(Some(new));
        }
        Ok(None)
    }
}

/// What one request did, for the server log.
#[derive(Default)]
struct Line {
    prompt: usize,
    reused: usize,
    prefill_ms: f64,
    restore_ms: f64,
    snap_ms: f64,
    ttft_ms: f64,
    decode: usize,
    decode_s: f64,
    windows: usize,
    drafted: usize,
    accepted: usize,
    calls: usize,
    finish: &'static str,
    spliced: usize,
    sampling: String,
}

pub struct FlashServe {
    e: Engine,
    tok: FlashTokenizer,
    ctx: usize,
    eos: Vec<u32>,
    think: (u32, u32),
    im_start: u32,
    wrap_up: Vec<u32>,
    snaps: Vec<Snap>,
    max_snaps: usize,
    max_snap_bytes: usize,
    memo: VecDeque<Memo>,
    policy: Policy,
    clock: u64,
    temp: f32,
    sampling: SamplingMode,
    /// Widest prefill window (`TANG_FLASH_WIDE`, default 64; 0: decode-width windows only).
    wide: usize,
    disk: Option<PathBuf>,
    dump: Option<PathBuf>,
    /// Lengths of a running state's parts, to check states read from disk.
    shape: Vec<usize>,
    gdn_layers: usize,
    requests: usize,
    line: Line,
    drafts_mode: &'static str,
}

/// Tool-call arguments as objects (the template wants mappings; OpenAI clients send strings).
fn args_as_objects(msgs: &mut Value) {
    for m in msgs.as_array_mut().into_iter().flatten() {
        for c in m["tool_calls"].as_array_mut().into_iter().flatten() {
            let f = if c["function"].is_object() { &mut c["function"] } else { &mut *c };
            if let Some(a) = f["arguments"].as_str() {
                f["arguments"] = serde_json::from_str(a).ok().filter(Value::is_object).unwrap_or_else(|| json!({}));
            }
        }
    }
}

const MARK: &str = "\u{F8FF}tang-splice\u{F8FF}";

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn hash(ids: &[u32]) -> u64 {
    // FNV-1a: stable across runs (the disk cache is keyed by it).
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &i in ids {
        for b in i.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

impl FlashServe {
    pub fn load(s: &Settings) -> Result<Self> {
        let t = Instant::now();
        let opts = Opts {
            max_ctx: s.ctx,
            mtp: s.mtp.clone(),
            ..Opts::default()
        };
        let mut e = Engine::load(&s.gguf, opts)?;
        eprintln!("{}", e.load_report);
        e.use_graphs = true;
        e.use_mtp = e.has_mtp();
        e.enable_sampler()?;
        e.warm()?;
        let tok = FlashTokenizer::from_gguf(e.gguf())?;
        let id = |p: &str| tok.token_id(p).with_context(|| format!("no {p} token"));
        let mut eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|p| tok.token_id(p)).collect();
        if let Ok(n) = e.gguf().meta_u64("tokenizer.ggml.eos_token_id") {
            eos.push(n as u32);
        }
        eos.sort_unstable();
        eos.dedup();
        let think = (id("<think>")?, id("</think>")?);
        let mut wrap_up = tok.encode(WRAP_UP)?;
        wrap_up.push(think.1);
        wrap_up.extend(tok.encode("\n\n")?);
        let shape0 = e.save_running()?;
        let shape = shape0.parts().iter().map(|p| p.len()).collect();
        let gdn_layers = shape0.gdn_layers();
        let state_mb = shape0.bytes() as f64 / 1e6;
        drop(shape0);
        if let Some(d) = &s.disk {
            std::fs::create_dir_all(d)?;
        }
        if let Some(d) = &s.dump {
            std::fs::create_dir_all(d)?;
        }
        eprintln!(
            "tang-llm: flash engine ready in {:.1} s: ctx {}, drafts {}, running state {:.0} MB, up to {} saved states{}",
            t.elapsed().as_secs_f64(),
            s.ctx,
            if e.use_mtp { "mtp" } else { "none" },
            state_mb,
            s.snapshots,
            s.disk.as_ref().map_or(String::new(), |d| format!(", system prompts saved in {}", d.display()))
        );
        let mut me = FlashServe {
            drafts_mode: if e.use_mtp { "mtp" } else { "none" },
            e,
            im_start: id("<|im_start|>")?,
            tok,
            ctx: s.ctx,
            eos,
            think,
            wrap_up,
            snaps: Vec::new(),
            max_snaps: s.snapshots,
            max_snap_bytes: s.snapshot_bytes,
            memo: VecDeque::new(),
            policy: Policy::new(),
            clock: 0,
            temp: s.temp,
            sampling: s.sampling,
            wide: std::env::var("TANG_FLASH_WIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(64),
            disk: s.disk.clone(),
            dump: s.dump.clone(),
            shape,
            gdn_layers,
            requests: 0,
            line: Line::default(),
        };
        // Capture the wide prefill graphs (64, 32, 16) now, not in the first request.
        if me.wide > 0 {
            let t = Instant::now();
            let ids: Vec<u32> = (0..64 + 32 + 16 + 1).map(|i| 1000 + i as u32).collect();
            for _ in 0..2 {
                me.e.reset();
                prefill_span(&mut me.e, &ids, MAX_T, usize::MAX, me.wide, true)?;
            }
            me.e.reset();
            eprintln!("tang-llm: wide prefill graphs (up to T={}) captured in {:.1} s", me.wide, t.elapsed().as_secs_f64());
        }
        for f in &s.warm {
            let t = Instant::now();
            let body: Value = serde_json::from_str(&std::fs::read_to_string(f)?).with_context(|| format!("reading {}", f.display()))?;
            let mut req = crate::server::parse(&body).map_err(|e| anyhow::anyhow!(e))?;
            req.prefill_only = true;
            me.generate(&req, &mut |_| true)?;
            eprintln!("tang-llm: warmed {} ({} tokens, {} reused) in {:.1} s", f.display(), me.line.prompt, me.line.reused, t.elapsed().as_secs_f64());
        }
        Ok(me)
    }

    /// The template's messages: tool-call arguments as objects (the template wants mappings),
    /// and assistant messages this server generated replaced by a marker (returned with the
    /// memo they match, in order).
    fn messages(&self, req: &Request) -> (Value, Vec<usize>) {
        let mut msgs = req.messages.clone();
        args_as_objects(&mut msgs);
        let mut spliced = Vec::new();
        let splice = std::env::var("TANG_FLASH_SPLICE").map_or(true, |v| v != "0");
        for m in msgs.as_array_mut().into_iter().flatten() {
            if m["role"] != "assistant" {
                continue;
            }
            let mut calls: Vec<(String, Value)> = Vec::new();
            for c in m["tool_calls"].as_array().into_iter().flatten() {
                let f = if c["function"].is_object() { &c["function"] } else { c };
                calls.push((f["name"].as_str().unwrap_or_default().to_string(), f["arguments"].clone()));
            }
            if !splice || m["reasoning_content"].as_str().is_some_and(|r| !r.trim().is_empty()) {
                continue;
            }
            let content = m["content"].as_str().unwrap_or_default().trim().to_string();
            // The newest match: the same visible reply can come from more than one generation.
            let hit = self.memo.iter().rposition(|x| x.content.trim() == content && x.calls == calls);
            if let Some(i) = hit {
                m["content"] = json!(MARK);
                if let Some(o) = m.as_object_mut() {
                    o.remove("tool_calls");
                    o.remove("reasoning_content");
                }
                spliced.push(i);
            }
        }
        (msgs, spliced)
    }

    /// The template, with the system block always rendered as with thinking on: this template
    /// puts its reasoning instructions at the very top of the system prompt, so a client that
    /// switches thinking per turn (frog does) would otherwise invalidate every cached token.
    /// Thinking off then only closes the generation prompt's `<think>` block, as the template
    /// does (`TANG_FLASH_STABLE_SYSTEM=0`: the template exactly).
    fn render(&self, msgs: &Value, req: &Request) -> Result<String> {
        let stable = std::env::var("TANG_FLASH_STABLE_SYSTEM").map_or(true, |v| v != "0");
        if !stable || req.think != Some(false) {
            return self.tok.render(msgs, req.tools.as_ref(), req.think);
        }
        let text = self.tok.render(msgs, req.tools.as_ref(), Some(true))?;
        match text.strip_suffix("<think>\n") {
            Some(head) => Ok(format!("{head}<think>\n\n</think>\n\n")),
            None => self.tok.render(msgs, req.tools.as_ref(), req.think),
        }
    }

    /// The prompt's ids, the rendered text, and how many assistant turns were spliced.
    fn prompt(&self, req: &Request) -> Result<(Vec<u32>, String, usize)> {
        let (msgs, spliced) = self.messages(req);
        let text = self.render(&msgs, req)?;
        let mut ids = Vec::new();
        let mut at = 0;
        let mut n = 0;
        let mut ok = true;
        let mut pieces = Vec::new();
        for (k, (i, _)) in text.match_indices(MARK).enumerate() {
            let Some(&mi) = spliced.get(k) else {
                ok = false;
                break;
            };
            let head = &text[at..i];
            let open = "<|im_start|>assistant\n";
            let head = if let Some(h) = head.strip_suffix("<think>\n\n</think>\n\n").filter(|h| h.ends_with(open)) {
                h
            } else if head.ends_with(open) {
                head
            } else {
                ok = false;
                break;
            };
            let m = &self.memo[mi];
            pieces.push((format!("{head}{}", m.tail), mi));
            at = i + MARK.len();
            n += 1;
        }
        if !ok || n != spliced.len() {
            // The template didn't render the markers where expected: no splicing.
            let mut msgs = req.messages.clone();
            args_as_objects(&mut msgs);
            let text = self.render(&msgs, req)?;
            eprintln!("tang-llm: splicing skipped (template rendered generated turns unexpectedly)");
            return Ok((self.tok.encode(&text)?, text, 0));
        }
        for (piece, mi) in pieces {
            ids.extend(self.tok.encode(&piece)?);
            ids.extend_from_slice(&self.memo[mi].ids);
        }
        ids.extend(self.tok.encode(&text[at..])?);
        Ok((ids, text, n))
    }

    /// Keep a copy of the running state after `tokens` (the engine's sequence now).
    fn snapshot(&mut self, pinned: bool) -> Result<()> {
        self.clock += 1;
        let clock = self.clock;
        if let Some(s) = self.snaps.iter_mut().find(|s| s.tokens == self.e.tokens) {
            s.used = clock;
            s.pinned |= pinned;
            return Ok(());
        }
        let t = Instant::now();
        let state = self.e.save_running()?;
        let n = self.e.tokens.len();
        // Only the positions past the longest saved state this one extends are copied.
        let base = (0..self.snaps.len())
            .filter(|&i| self.e.tokens.starts_with(&self.snaps[i].tokens))
            .max_by_key(|&i| self.snaps[i].tokens.len());
        let (mut pos, from) = match base {
            Some(i) => (self.snaps[i].pos.clone(), self.snaps[i].tokens.len()),
            None => (Vec::new(), 0),
        };
        if from < n {
            pos.push(Arc::new(self.e.save_positional(from, n)?));
        }
        let tokens = self.e.tokens.clone();
        if pinned && self.disk_path(&tokens).is_some_and(|p| !p.exists()) {
            let full = self.e.save_positional(0, n)?;
            self.save_disk(&tokens, &state, &full);
        }
        self.line.snap_ms += t.elapsed().as_secs_f64() * 1e3;
        self.snaps.push(Snap { tokens, state, pos, pinned, used: clock });
        let size = |snaps: &[Snap]| {
            let mut seen = std::collections::HashSet::new();
            snaps
                .iter()
                .map(|s| s.state.bytes() + s.pos.iter().filter(|p| seen.insert(Arc::as_ptr(p))).map(|p| p.bytes()).sum::<usize>())
                .sum::<usize>()
        };
        while self.snaps.len() > 1 && size(&self.snaps) > self.max_snap_bytes {
            let lru = (0..self.snaps.len()).min_by_key(|&i| (self.snaps[i].pinned, self.snaps[i].used)).unwrap();
            self.snaps.swap_remove(lru);
        }
        // Least recently used first; pinned ones have their own (small) quota.
        for (want_pinned, cap) in [(false, self.max_snaps), (true, 4)] {
            while self.snaps.iter().filter(|s| s.pinned == want_pinned).count() > cap {
                let lru = (0..self.snaps.len())
                    .filter(|&i| self.snaps[i].pinned == want_pinned)
                    .min_by_key(|&i| self.snaps[i].used)
                    .unwrap();
                self.snaps.swap_remove(lru);
            }
        }
        Ok(())
    }

    fn disk_path(&self, ids: &[u32]) -> Option<PathBuf> {
        Some(self.disk.as_ref()?.join(format!("{}-{:016x}.state", ids.len(), hash(ids))))
    }

    /// Write a pinned state to disk (on a thread; the file appears complete or not at all).
    fn save_disk(&self, ids: &[u32], s: &RunningState, pos: &Positional) {
        let Some(path) = self.disk_path(ids) else { return };
        if path.exists() {
            return;
        }
        let mut bytes: Vec<u8> = Vec::with_capacity(s.bytes() + 4 * ids.len() + 64);
        bytes.extend_from_slice(b"TANGRS02");
        bytes.extend_from_slice(&(ids.len() as u64).to_le_bytes());
        for &i in ids {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        let parts = s.parts();
        bytes.extend_from_slice(&(parts.len() as u64).to_le_bytes());
        for p in parts {
            bytes.extend_from_slice(&(p.len() as u64).to_le_bytes());
            for &x in p {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&(pos.parts().len() as u64).to_le_bytes());
        for p in pos.parts() {
            bytes.extend_from_slice(&(p.len() as u64).to_le_bytes());
            bytes.extend_from_slice(p);
        }
        let budget = std::env::var("TANG_FLASH_DISK_GB").ok().and_then(|v| v.parse::<f64>().ok()).map_or(8e9, |g| g * 1e9) as u64;
        std::thread::spawn(move || {
            let tmp = path.with_extension("tmp");
            let r = std::fs::File::create(&tmp)
                .and_then(|mut f| f.write_all(&bytes))
                .and_then(|_| std::fs::rename(&tmp, &path));
            if let Err(e) = r {
                eprintln!("tang-llm: saving {}: {e}", path.display());
            }
            // Keep the directory under its budget, oldest first.
            let Some(dir) = path.parent() else { return };
            let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = std::fs::read_dir(dir)
                .into_iter()
                .flatten()
                .flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "state"))
                .filter_map(|e| {
                    let m = e.metadata().ok()?;
                    Some((m.modified().ok()?, m.len(), e.path()))
                })
                .collect();
            files.sort();
            let mut total: u64 = files.iter().map(|f| f.1).sum();
            for (_, len, p) in files {
                if total <= budget {
                    break;
                }
                let _ = std::fs::remove_file(&p);
                total -= len;
            }
        });
    }

    /// The longest state on disk that `ids[..=limit]` extends.
    fn load_disk(&self, ids: &[u32], limit: usize) -> Option<Snap> {
        let dir = self.disk.as_ref()?;
        let mut best: Option<(usize, PathBuf)> = None;
        for ent in std::fs::read_dir(dir).ok()?.flatten() {
            let name = ent.file_name().to_string_lossy().to_string();
            let Some(stem) = name.strip_suffix(".state") else { continue };
            let Some((n, h)) = stem.split_once('-') else { continue };
            let (Ok(n), Ok(h)) = (n.parse::<usize>(), u64::from_str_radix(h, 16)) else { continue };
            if n <= limit && n > best.as_ref().map_or(0, |b| b.0) && hash(&ids[..n]) == h {
                best = Some((n, ent.path()));
            }
        }
        let (n, path) = best?;
        let read = || -> Result<Snap> {
            let mut b = Vec::new();
            std::fs::File::open(&path)?.read_to_end(&mut b)?;
            let mut at = 8usize;
            fn take<'a>(b: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8]> {
                let s = b.get(*at..*at + n).context("state file is short")?;
                *at += n;
                Ok(s)
            }
            let word = |b: &[u8], at: &mut usize| -> Result<usize> { Ok(u64::from_le_bytes(take(b, at, 8)?.try_into()?) as usize) };
            ensure!(b.get(..8) == Some(&b"TANGRS02"[..]), "not a state file");
            let len = word(&b, &mut at)?;
            ensure!(len == n, "length");
            let tokens: Vec<u32> = take(&b, &mut at, 4 * len)?.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            ensure!(tokens == ids[..n], "tokens differ");
            ensure!(word(&b, &mut at)? == self.shape.len(), "parts");
            let mut parts = Vec::with_capacity(self.shape.len());
            for &want in &self.shape {
                let k = word(&b, &mut at)?;
                ensure!(k == want, "part size");
                parts.push(take(&b, &mut at, 4 * k)?.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
            }
            let np = word(&b, &mut at)?;
            let mut pos = Vec::with_capacity(np);
            for _ in 0..np {
                let k = word(&b, &mut at)?;
                pos.push(take(&b, &mut at, k)?.to_vec());
            }
            Ok(Snap {
                tokens,
                state: RunningState::from_parts(n, self.gdn_layers, parts)?,
                pos: vec![Arc::new(Positional::from_parts(0, n, pos))],
                pinned: true,
                used: 0,
            })
        };
        match read() {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("tang-llm: reading {}: {e:#}", path.display());
                None
            }
        }
    }

    /// Bring the engine to a prefix of `ids` (live sequence, a saved state, or empty), prefill
    /// the rest (saving states at the system prompt's end and the last turn boundary), and
    /// return (tokens reused, the sampled next token).
    fn prefill(&mut self, ids: &[u32], text: &str) -> Result<(usize, u32)> {
        let limit = ids.len() - 1;
        let live = self.e.tokens.len();
        let live_ok = live <= limit && lcp(&self.e.tokens, ids) == live;
        let best = (0..self.snaps.len())
            .filter(|&i| self.snaps[i].tokens.len() <= limit && ids.starts_with(&self.snaps[i].tokens))
            .max_by_key(|&i| self.snaps[i].tokens.len());
        let snap_len = best.map_or(0, |i| self.snaps[i].tokens.len());
        let mut start = if live_ok { live } else { 0 };
        let t = Instant::now();
        if snap_len > start {
            let i = best.unwrap();
            self.clock += 1;
            self.snaps[i].used = self.clock;
            let valid = lcp(&self.e.tokens, &self.snaps[i].tokens);
            for p in &self.snaps[i].pos {
                self.e.restore_positional(p, valid)?;
            }
            let (tokens, state) = (&self.snaps[i].tokens, &self.snaps[i].state);
            self.e.restore_running(state, tokens)?;
            start = snap_len;
        } else if !live_ok {
            match self.load_disk(ids, limit).filter(|s| s.tokens.len() > 0) {
                Some(s) => {
                    let valid = lcp(&self.e.tokens, &s.tokens);
                    for p in &s.pos {
                        self.e.restore_positional(p, valid)?;
                    }
                    self.e.restore_running(&s.state, &s.tokens)?;
                    start = s.tokens.len();
                    eprintln!("tang-llm: read the running state at {start} from disk");
                    self.snaps.push(s);
                }
                None => self.e.reset(),
            }
        }
        self.line.restore_ms = t.elapsed().as_secs_f64() * 1e3;
        // Where to save states: the end of the system prompt (the second `<|im_start|>`) and
        // the last turn boundary (the last one, which opens the generation prompt).
        let starts: Vec<usize> = ids.iter().enumerate().filter(|(_, &t)| t == self.im_start).map(|(i, _)| i).collect();
        let mut marks: Vec<(usize, bool)> = Vec::new();
        // The end of the tools block, which the template puts before the system text (that
        // has the working directory in it): the same for every session of a client.
        if let Some(i) = text.find("</IMPORTANT>").filter(|&i| i < 200_000) {
            if let Ok(head) = self.tok.encode(&text[..i + "</IMPORTANT>".len()]) {
                if head.len() > 1 && ids.starts_with(&head[..head.len() - 1]) {
                    // The last token may merge with what follows; stop one short.
                    marks.push((head.len() - 1, true));
                }
            }
        }
        if let Some(&p) = starts.get(1) {
            marks.push((p, true));
        }
        if let Some(&p) = starts.last() {
            marks.push((p, false));
        }
        marks.sort_by_key(|m| m.0);
        marks.retain(|&(p, _)| p > start && p < ids.len());
        marks.dedup_by_key(|m| m.0);
        // The MTP only over the prompt's last positions (`TANG_FLASH_MTP_PREFILL`, default 512).
        let mtp_from = ids.len().saturating_sub(std::env::var("TANG_FLASH_MTP_PREFILL").ok().and_then(|v| v.parse().ok()).unwrap_or(512));
        let mut a = start;
        for (p, pinned) in marks {
            if p > a {
                prefill_span(&mut self.e, &ids[a..p], MAX_T, mtp_from, self.wide, false)?;
                a = p;
            }
            self.snapshot(pinned)?;
        }
        let next = prefill_span(&mut self.e, &ids[a..], MAX_T, mtp_from, self.wide, true)?.0;
        Ok((start, next))
    }

    fn generate(&mut self, req: &Request, on: &mut dyn FnMut(Piece) -> bool) -> Result<(Finish, Usage)> {
        let t0 = Instant::now();
        self.line = Line::default();
        let (ids, text, spliced) = self.prompt(req)?;
        self.line.spliced = spliced;
        ensure!(ids.len() + 1 < self.ctx, "prompt is {} tokens; the context window is {}", ids.len(), self.ctx);
        let limit = req.max_tokens.unwrap_or(usize::MAX).min(self.ctx - ids.len() - 1).max(1);
        let thinking = req.think != Some(false);
        let card = match (self.sampling, thinking) {
            (SamplingMode::Greedy, _) => (self.temp, 0usize, 1.0f32, 0.0f32),
            (SamplingMode::ModelCard, true) => (1.0, 20, 0.95, 0.0),
            (SamplingMode::ModelCard, false) => (0.7, 20, 0.8, 1.5),
        };
        let temp = if req.temperature_set { req.sampling.temperature.max(0.0) } else { card.0 };
        let top_k = if req.top_k_set { req.sampling.top_k } else { card.1 };
        let top_p = if req.top_p_set { req.sampling.top_p } else { card.2 };
        let presence = req.presence_penalty.unwrap_or(card.3);
        self.e.temperature = temp;
        self.e.seed = req.sampling.seed as u32;
        if let Some(sp) = self.e.sampler_mut() {
            // top-k 0 (off) keeps the 64 best: the sampler's bound.
            sp.top_k = if top_k == 0 { super::sampler::MAX_K } else { top_k };
            sp.top_p = if top_p > 0.0 && top_p <= 1.0 { top_p } else { 1.0 };
            sp.presence = presence;
        }
        self.e.sampler_begin(ids.len())?;
        self.line.sampling = format!("T {temp} k {top_k} p {top_p} pres {presence}");
        self.requests += 1;
        let tag = self.requests;
        if let Some(d) = &self.dump {
            std::fs::write(d.join(format!("req-{tag}.ids")), ids.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "))?;
        }
        let (reused, cur) = self.prefill(&ids, &text)?;
        let prefill_s = t0.elapsed().as_secs_f64();
        self.line.prompt = ids.len();
        self.line.reused = reused;
        self.line.prefill_ms = prefill_s * 1e3;
        let usage = |out: usize, reasoning: usize, decode_s: f64, drafted: usize, accepted: usize| Usage {
            prompt_tokens: ids.len(),
            cached_tokens: reused,
            completion_tokens: out,
            reasoning_tokens: reasoning,
            prefill_tok_s: (ids.len() - reused) as f64 / prefill_s.max(1e-9),
            decode_tok_s: out as f64 / decode_s.max(1e-9),
            draft_tokens: drafted,
            accepted_tokens: accepted,
        };
        if req.prefill_only {
            self.line.finish = "prefill";
            return Ok((Finish::Stop, usage(0, 0, 0.0, 0, 0)));
        }

        let gen_at = text.rfind("<|im_start|>assistant\n").map_or(text.len(), |i| i + "<|im_start|>assistant\n".len());
        let tail = text[gen_at..].to_string();
        let opened = text.trim_end().ends_with("<think>");
        let mut parser = Parser::new(opened).with_tools(req.tools.as_ref());
        let mut think = ThinkBudget::new(Some(self.think), req.thinking_budget, opened);
        let mut in_think = opened;
        let mut spent = 0usize;
        let mut out: Vec<u32> = Vec::new();
        let mut detok = Detok::default();
        let mut text_out = String::new();
        let (mut content, mut calls) = (String::new(), Vec::new());
        let mut finish = Finish::Length;
        let (mut windows, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
        let mut first = None;
        let t1 = Instant::now();
        let mut pending = vec![cur];
        // Deliver pieces; false when the client went away.
        let mut deliver = |pieces: Vec<Piece>, content: &mut String, calls: &mut Vec<(String, Value)>, first: &mut Option<f64>| -> bool {
            for p in pieces {
                if first.is_none() {
                    *first = Some(t0.elapsed().as_secs_f64() * 1e3);
                }
                match &p {
                    Piece::Text(t) => content.push_str(t),
                    Piece::ToolCall { name, arguments } => calls.push((name.clone(), arguments.clone())),
                    Piece::Reasoning(_) => {}
                }
                if !on(p) {
                    return false;
                }
            }
            true
        };
        'gen: loop {
            let mut wrap = false;
            for &tok in &pending {
                if self.eos.contains(&tok) {
                    finish = Finish::Stop;
                    break 'gen;
                }
                out.push(tok);
                if tok == self.think.0 {
                    in_think = true;
                } else if tok == self.think.1 {
                    in_think = false;
                } else if in_think {
                    spent += 1;
                }
                wrap |= think.step(tok);
                if let Some(new) = detok.push(&self.tok, &out)? {
                    text_out.push_str(&new);
                    let stop_at = req.stop.iter().filter_map(|s| text_out.find(s.as_str())).min();
                    let new = match stop_at {
                        Some(i) => new[..new.len().saturating_sub(text_out.len() - i)].to_string(),
                        None => new,
                    };
                    if !deliver(parser.push(&new), &mut content, &mut calls, &mut first) {
                        finish = Finish::Cancelled;
                        break 'gen;
                    }
                    if stop_at.is_some() {
                        finish = Finish::Stop;
                        break 'gen;
                    }
                }
                if out.len() >= limit {
                    break 'gen;
                }
            }
            let last = *pending.last().unwrap();
            if wrap && out.len() + self.wrap_up.len() < limit {
                // Out of thinking budget: the wrap-up and `</think>` as if the model wrote them.
                think.closed(self.wrap_up.len());
                for &w in &self.wrap_up.clone() {
                    out.push(w);
                    if let Some(new) = detok.push(&self.tok, &out)? {
                        text_out.push_str(&new);
                        if !deliver(parser.push(&new), &mut content, &mut calls, &mut first) {
                            finish = Finish::Cancelled;
                            break 'gen;
                        }
                    }
                }
                in_think = false;
                let mut feed = vec![last];
                feed.extend_from_slice(&self.wrap_up);
                let next = self.e.prefill(&feed, MAX_T, None)?;
                windows += feed.len().div_ceil(MAX_T);
                pending = vec![next];
                continue;
            }
            // Drafts: none past the token limit, the thinking budget, or an end of turn (so the
            // sequence never runs past where the reply ends).
            let mut room = (MAX_T - 1).min(limit - out.len() - 1);
            if in_think {
                room = room.min(self.policy.think_room);
                if let (Some(b), false) = (req.thinking_budget, think.hit) {
                    room = room.min(b.saturating_sub(spent + 1));
                }
            }
            let (mut drafts, mut probs) = self.policy.choose(&self.e.mtp_last, room);
            if let Some(k) = drafts.iter().position(|d| self.eos.contains(d)) {
                drafts.truncate(k);
                probs.truncate(k);
            }
            let kept = self.e.verify(last, &drafts)?;
            if self.e.use_mtp {
                let wall = self.e.last.wall_ms;
                let mtp = self.e.last_mtp_ms;
                self.policy.observe(&probs, kept.len() - 1, drafts.len() + 1, wall, mtp);
            }
            windows += 1;
            drafted += drafts.len();
            accepted += kept.len() - 1;
            pending = kept;
        }
        if finish != Finish::Cancelled {
            deliver(parser.finish(), &mut content, &mut calls, &mut first);
            if !calls.is_empty() && finish == Finish::Stop {
                finish = Finish::ToolCalls;
            }
        }
        let decode_s = t1.elapsed().as_secs_f64();
        // A reply that ended on its own can be spliced into the next prompt.
        if matches!(finish, Finish::Stop | Finish::ToolCalls) && req.stop.iter().all(|s| !text_out.contains(s.as_str())) {
            self.memo.push_back(Memo { content: content.clone(), calls: calls.clone(), tail, ids: out.clone() });
            while self.memo.len() > 256 {
                self.memo.pop_front();
            }
        }
        if let Some(d) = &self.dump {
            std::fs::write(d.join(format!("req-{tag}.out")), out.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "))?;
        }
        self.line.ttft_ms = first.unwrap_or(t0.elapsed().as_secs_f64() * 1e3);
        self.line.decode = out.len();
        self.line.decode_s = decode_s;
        self.line.windows = windows;
        self.line.drafted = drafted;
        self.line.accepted = accepted;
        self.line.calls = calls.len();
        self.line.finish = match finish {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::ToolCalls => "tool_calls",
            Finish::Cancelled => "cancelled",
        };
        Ok((finish, usage(out.len(), think.reasoning, decode_s, drafted, accepted)))
    }
}

impl Backend for FlashServe {
    fn complete(&mut self, req: &Request, on: &mut dyn FnMut(Piece) -> bool) -> Result<(Finish, Usage)> {
        let r = self.generate(req, on);
        if r.is_err() {
            // Whatever the engine holds now is unknown: start over next time.
            self.e.reset();
        }
        r
    }

    fn context_window(&self) -> usize {
        self.ctx
    }

    fn after(&mut self, _req: &Request, result: &Result<(Finish, Usage)>) {
        let l = &self.line;
        match result {
            Ok(_) => eprintln!(
                "tang-llm: req {}: prompt {} (reused {}, prefilled {} in {:.0} ms = {:.0} tok/s; restore {:.0} ms, state saves {:.0} ms; {} turns spliced) | ttft {:.0} ms | decode {} tokens in {:.2} s = {:.1} tok/s, {} windows, {:.2} tokens/window, drafts {} {}/{} accepted | {} tool calls, {} | {}",
                self.requests,
                l.prompt,
                l.reused,
                l.prompt - l.reused,
                l.prefill_ms,
                (l.prompt - l.reused) as f64 / (l.prefill_ms / 1e3).max(1e-9),
                l.restore_ms,
                l.snap_ms,
                l.spliced,
                l.ttft_ms,
                l.decode,
                l.decode_s,
                l.decode as f64 / l.decode_s.max(1e-9),
                l.windows,
                l.decode as f64 / l.windows.max(1) as f64,
                self.drafts_mode,
                l.accepted,
                l.drafted,
                l.calls,
                l.finish,
                l.sampling
            ),
            Err(e) => eprintln!("tang-llm: req {}: error: {e:#}", self.requests),
        }
        // Keep the end of this conversation, in case another one runs next.
        if result.is_ok() && !self.e.tokens.is_empty() {
            let t = Instant::now();
            if let Err(e) = self.snapshot(false) {
                eprintln!("tang-llm: saving the running state: {e:#}");
            }
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if ms > 50.0 {
                eprintln!("tang-llm: saved the running state at {} in {ms:.0} ms", self.e.tokens.len());
            }
        }
    }
}

/// `tang-llm flash-resume-test <gguf> --ids-file F [--at P] [--mtp M]`: is resuming from a saved
/// running state bitwise the same as never having left? Prefills `ids[..P]`, saves the state,
/// runs unrelated tokens, restores, prefills `ids[P..]`, and compares every logit of the suffix
/// with a run that prefilled `ids[..P]` then `ids[P..]` straight through, and with one that
/// prefilled all of `ids` in one go (window boundaries differ there).
pub fn resume_test(args: &[String]) -> Result<()> {
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context("first argument: the first GGUF shard")?);
    let (mut ids, mut at, mut mtp) = (Vec::new(), None, None);
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--ids-file" => {
                ids = std::fs::read_to_string(val()?)?
                    .split(|c: char| c.is_whitespace() || c == ',')
                    .filter(|w| !w.is_empty())
                    .map(|w| w.parse::<u32>())
                    .collect::<Result<_, _>>()?
            }
            "--at" => at = Some(val()?.parse::<usize>()?),
            "--mtp" => mtp = Some(PathBuf::from(val()?)),
            s => bail!("unknown argument {s}"),
        }
    }
    ensure!(ids.len() > 16, "need a prompt of more than 16 ids");
    let p = at.unwrap_or(ids.len() * 2 / 3 + 3);
    let opts = Opts { max_ctx: (ids.len() + 256).next_multiple_of(1024), mtp, ..Opts::default() };
    let mut e = Engine::load(&path, opts)?;
    e.use_graphs = true;
    e.warm()?;
    e.use_mtp = e.has_mtp();
    let v = e.hp.n_vocab;
    let run = |e: &mut Engine, segs: &[&[u32]], from: usize| -> Result<(Vec<f32>, u32)> {
        let mut rows = vec![0f32; (ids.len() - from) * v];
        let mut next = 0;
        for seg in segs {
            let mut cb = |pos0: usize, t: usize, lg: &[f32]| {
                for i in 0..t {
                    if pos0 + i >= from {
                        rows[(pos0 + i - from) * v..(pos0 + i - from + 1) * v].copy_from_slice(&lg[i * v..(i + 1) * v]);
                    }
                }
            };
            next = e.prefill(seg, MAX_T, Some(&mut cb))?;
        }
        Ok((rows, next))
    };
    let bits = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    // Straight through, split at p.
    e.reset();
    let (split, n_split) = run(&mut e, &[&ids[..p], &ids[p..]], p)?;
    // All at once.
    e.reset();
    let (whole, n_whole) = run(&mut e, &[&ids], p)?;
    // Saved at p, disturbed, restored.
    e.reset();
    e.prefill(&ids[..p], MAX_T, None)?;
    let t = Instant::now();
    let st = e.save_running()?;
    let pos = e.save_positional(0, p)?;
    let save_ms = t.elapsed().as_secs_f64() * 1e3;
    let junk: Vec<u32> = (0..53).map(|i| 1000 + 37 * i).collect();
    e.prefill(&junk, MAX_T, None)?;
    let t = Instant::now();
    e.restore_running(&st, &ids[..p])?;
    let restore_ms = t.elapsed().as_secs_f64() * 1e3;
    let (resumed, n_resumed) = run(&mut e, &[&ids[p..]], p)?;
    // And from a fresh engine state (reset, then restore), as after another conversation.
    e.reset();
    e.prefill(&junk, MAX_T, None)?;
    let valid = lcp(&e.tokens, &ids[..p]);
    e.restore_positional(&pos, valid)?;
    e.restore_running(&st, &ids[..p])?;
    let (resumed2, _) = run(&mut e, &[&ids[p..]], p)?;
    println!(
        "resume test: {} ids, state saved at {p} ({:.0} MB + {:.0} MB positional, save {save_ms:.0} ms, restore {restore_ms:.0} ms)",
        ids.len(),
        st.bytes() as f64 / 1e6,
        pos.bytes() as f64 / 1e6
    );
    println!("  resumed vs straight (same windows): {} of {} logits differ; next token {} vs {}", bits(&resumed, &split), split.len(), n_resumed, n_split);
    println!("  resumed after other prefills vs straight: {} of {} differ", bits(&resumed2, &split), split.len());
    println!("  split at {p} vs one prefill (different window boundaries): {} of {} differ; next {} vs {}", bits(&split, &whole), split.len(), n_split, n_whole);
    let ok = bits(&resumed, &split) == 0 && bits(&resumed2, &split) == 0;
    println!("resume test: {}", if ok { "PASS (bitwise)" } else { "FAIL" });
    Ok(())
}

/// `TANG_FLASH_STATE_DIR` or `~/.cache/tang/flash-serve/<model>`.
pub fn default_disk(gguf: &Path) -> Option<PathBuf> {
    match std::env::var("TANG_FLASH_STATE_DIR").as_deref() {
        Ok("0") | Ok("") => None,
        Ok(d) => Some(PathBuf::from(d)),
        Err(_) => {
            let name = gguf.file_stem()?.to_string_lossy().to_string();
            Some(PathBuf::from(std::env::var_os("HOME")?).join(".cache/tang/flash-serve").join(name))
        }
    }
}

/// Prefill `ids` after the engine's sequence in windows of `chunk`, the MTP run only over
/// windows that reach position `mtp_from` (0: all). Returns the sampled next token, each
/// window's stats and the MTP time. The MTP's K/V for earlier positions stays unwritten: that
/// only changes its drafts' acceptance, never the output (measured: no change in tokens/window
/// at 2K and 10K with the last 512 positions).
pub fn prefill_windows(e: &mut Engine, ids: &[u32], chunk: usize, mtp_from: usize) -> Result<(u32, Vec<super::engine::WinStats>, f64)> {
    prefill_span(e, ids, chunk, mtp_from, 0, true)
}

/// [`prefill_windows`] with wide prefill windows (`Engine::prefill_wide`, widths 16/32/64 up
/// to `wide`; 0: none) for everything but the last `MAX_T` tokens when `last` (that window
/// runs the head), or for all of it otherwise (the result's next token is then meaningless).
pub fn prefill_span(e: &mut Engine, ids: &[u32], chunk: usize, mtp_from: usize, wide: usize, last: bool) -> Result<(u32, Vec<super::engine::WinStats>, f64)> {
    ensure!(!ids.is_empty(), "empty prompt");
    let pos_start = e.tokens.len();
    e.tokens.extend_from_slice(ids);
    let end = e.tokens.len();
    let (mut pos, mut next, mut stats, mut mtp_ms) = (pos_start, 0, Vec::new(), 0.0);
    while pos < end {
        let left = end - pos;
        let room = if last { left.saturating_sub(1) } else { left };
        if let Some(&w) = [64usize, 32, 16].iter().find(|&&w| w <= wide && w <= room) {
            stats.push(e.prefill_wide(pos, w)?);
            pos += w;
            continue;
        }
        let t = chunk.min(left).clamp(1, MAX_T);
        let out = e.window(pos, t, None)?;
        stats.push(e.last);
        if e.use_mtp && e.has_mtp() && pos + t > mtp_from {
            let nexts: Vec<u32> = (0..t).map(|i| e.tokens.get(pos + i + 1).copied().unwrap_or(out[t - 1])).collect();
            let t0 = Instant::now();
            e.mtp_last = e.mtp_draft(pos, &nexts)?;
            mtp_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
        next = out[t - 1];
        pos += t;
    }
    Ok((next, stats, mtp_ms))
}

/// `tang-llm flash-prefill-bench <gguf> --ids-file F [--n 2048,10240] [--mtp M] [--caps 0,16,88]
/// [--tails 0,512]`: prefill tok/s and the per-window split for the served prefill and its
/// variants (MTP over the last N positions only, GPU-streamed misses via `pcie_cap`), each
/// checked bitwise against the served path's last logits, with draft acceptance over 128
/// decoded tokens after it.
pub fn prefill_bench(args: &[String]) -> Result<()> {
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context("first argument: the first GGUF shard")?);
    let (mut ids, mut ns, mut mtp) = (Vec::<u32>::new(), vec![2048usize, 10240], None);
    let (mut caps, mut tails) = (vec![16usize, 88], vec![0usize, 512]);
    let mut quick = false;
    let mut chunks: Vec<usize> = Vec::new();
    let mut wides: Vec<usize> = Vec::new();
    let list = |s: &str| -> Result<Vec<usize>> { s.split(',').map(|x| Ok(x.parse::<usize>()?)).collect() };
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--ids-file" => {
                for f in val()?.split(',') {
                    ids.extend(std::fs::read_to_string(f)?.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()).map(|w| w.parse::<u32>()).collect::<Result<Vec<_>, _>>()?);
                }
            }
            "--n" => ns = list(val()?)?,
            "--caps" => caps = list(val()?)?,
            "--tails" => tails = list(val()?)?,
            "--mtp" => mtp = Some(PathBuf::from(val()?)),
            "--quick" => quick = true,
            "--chunks" => chunks = list(val()?)?,
            "--wide" => wides = list(val()?)?,
            s => bail!("unknown argument {s}"),
        }
    }
    if quick {
        caps.clear();
        tails = vec![512];
    }
    ensure!(!ids.is_empty(), "--ids-file");
    let max_n = *ns.iter().max().unwrap();
    let base = ids.clone();
    while ids.len() < max_n {
        ids.extend_from_slice(&base);
    }
    let opts = Opts { max_ctx: (max_n + 1024).next_multiple_of(1024), mtp, split: true, ..Opts::default() };
    let mut e = Engine::load(&path, opts)?;
    eprintln!("{}", e.load_report);
    e.use_graphs = true;
    e.warm()?;
    let has_mtp = e.has_mtp();
    let v = e.hp.n_vocab;
    let default_cap = e.pcie_cap;
    for &n in &ns {
        let p = &ids[..n];
        let mut configs: Vec<(String, usize, usize, usize)> = vec![("served (mtp all, cap default)".into(), usize::MAX, default_cap, MAX_T)];
        for &w in &wides {
            configs.push((format!("wide {w}, mtp last 512"), 512, default_cap, 1000 + w));
        }
        for &c in &chunks {
            configs.push((format!("chunk {c}, mtp last 512"), 512, default_cap, c));
        }
        for &t in &tails {
            configs.push((format!("mtp last {t}"), t, default_cap, MAX_T));
        }
        for &c in &caps {
            configs.push((format!("mtp last 512, pcie_cap {c}"), 512, c, MAX_T));
        }
        if !quick {
            configs.push(("served again".into(), usize::MAX, default_cap, MAX_T));
        }
        let mut want: Option<(u32, Vec<f32>)> = None;
        println!("prefill {n} tokens:");
        for (name, tail, cap, chunk) in configs {
            e.reset();
            e.use_mtp = has_mtp;
            if e.pcie_cap != cap {
                e.pcie_cap = cap;
                for t in 1..=MAX_T {
                    e.drop_window_graph(t, false);
                    e.drop_window_graph(t, true);
                }
            }
            let t0 = Instant::now();
            let (next, st, mtp_ms) = if chunk > 1000 {
                prefill_span(&mut e, p, MAX_T, n.saturating_sub(tail), chunk - 1000, true)?
            } else {
                prefill_windows(&mut e, p, chunk, n.saturating_sub(tail))?
            };
            let secs = t0.elapsed().as_secs_f64();
            let last_t = st.last().map_or(1, |s| s.t);
            let lg = e.logits(last_t)[(last_t - 1) * v..].to_vec();
            let same = match &want {
                None => {
                    want = Some((next, lg));
                    "reference".to_string()
                }
                Some((wn, wl)) => {
                    let d = wl.iter().zip(&lg).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    format!("{} (next {} vs {}, {d} logits differ)", if d == 0 && *wn == next { "SAME" } else { "DIFFERS" }, next, wn)
                }
            };
            // Decode 128 tokens with greedy MTP chains, for the drafts' acceptance.
            let (mut cur, mut toks, mut wins) = (next, 0usize, 0usize);
            let t1 = Instant::now();
            while toks < 128 && has_mtp {
                let d: Vec<u32> = e.mtp_last.iter().take(MAX_T - 1).map(|x| x.0).collect();
                let kept = e.verify(cur, &d)?;
                toks += kept.len();
                wins += 1;
                cur = *kept.last().unwrap();
            }
            let dsecs = t1.elapsed().as_secs_f64();
            let wide: Vec<&super::engine::WinStats> = st.iter().filter(|s| s.t > MAX_T).collect();
            if !wide.is_empty() {
                let n = wide.len() as f64;
                let ms = wide.iter().map(|s| s.wall_ms).sum::<f64>() / n;
                let pcie = wide.iter().map(|s| s.pcie).sum::<usize>() as f64 / n;
                println!(
                    "    wide windows: {} of T={}, {ms:.1} ms each, {:.0} distinct experts, {pcie:.0} over PCIe ({:.2} GB) per window",
                    wide.len(),
                    wide[0].t,
                    wide.iter().map(|s| s.distinct).sum::<usize>() as f64 / n,
                    pcie * tang_compute::flash::ExpertBlob::BYTES as f64 / 1e9
                );
            }
            let w = st.len().max(1) as f64;
            let m = |f: fn(&super::engine::WinStats) -> f64| st.iter().map(f).sum::<f64>() / w;
            let c = |f: fn(&super::engine::WinStats) -> usize| st.iter().map(f).sum::<usize>() as f64 / w;
            println!(
                "  {name:<34} {:7.0} tok/s ({:.2} s) | window {:.2} ms: GPU {:.2}, wait plan {:.2}, wait CPU rows {:.2}, host CPU experts {:.2}, plan {:.2}, MTP {:.2} | experts/window {:.0} distinct, {:.1} CPU, {:.1} PCIe, swaps {:.1} | {} | decode 128: {:.2} tok/window, {:.0} tok/s",
                n as f64 / secs,
                secs,
                m(|s| s.wall_ms),
                m(|s| s.gpu_ms),
                m(|s| s.gpu_wait_a_ms),
                m(|s| s.gpu_wait_b_ms),
                m(|s| s.cpu_ms),
                m(|s| s.plan_ms),
                mtp_ms / w,
                c(|s| s.distinct),
                c(|s| s.missed),
                c(|s| s.pcie),
                c(|s| s.swaps),
                same,
                toks as f64 / wins.max(1) as f64,
                toks as f64 / dsecs.max(1e-9),
            );
        }
    }
    Ok(())
}

/// Decode `n` tokens after `ids` from a fresh sequence, with MTP chains as drafts or none.
fn decode_n(e: &mut Engine, ids: &[u32], n: usize, drafts: bool) -> Result<Vec<u32>> {
    e.reset();
    e.sampler_begin(ids.len())?;
    let mut cur = e.prefill(ids, MAX_T, None)?;
    let mut out = vec![cur];
    while out.len() < n {
        let d: Vec<u32> = if drafts { e.mtp_last.iter().take((MAX_T - 1).min(n - out.len())).map(|x| x.0).collect() } else { Vec::new() };
        let kept = e.verify(cur, &d)?;
        out.extend_from_slice(&kept);
        cur = *kept.last().unwrap();
    }
    out.truncate(n);
    Ok(out)
}

/// `tang-llm flash-sampler-test <gguf> --ids-file F --mtp M [-n N]`: the head sampler against
/// the plain argmax (greedy, top-k 1: identical), and with and without drafts at the model
/// card's settings (identical: drafts never change a sample).
pub fn sampler_test(args: &[String]) -> Result<()> {
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context("first argument: the first GGUF shard")?);
    let (mut ids, mut mtp, mut n) = (Vec::<u32>::new(), None, 96usize);
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--ids-file" => ids = std::fs::read_to_string(val()?)?.split_whitespace().map(|w| w.parse::<u32>()).collect::<Result<_, _>>()?,
            "--mtp" => mtp = Some(PathBuf::from(val()?)),
            "-n" => n = val()?.parse()?,
            s => bail!("unknown argument {s}"),
        }
    }
    let opts = Opts { max_ctx: (ids.len() + n + 64).next_multiple_of(1024), mtp, ..Opts::default() };
    let mut e = Engine::load(&path, opts)?;
    e.use_graphs = true;
    e.use_mtp = e.has_mtp();
    e.warm()?;
    let base = decode_n(&mut e, &ids, n, true)?;
    e.enable_sampler()?;
    e.warm()?;
    let mut ok = true;
    let mut check = |name: &str, a: &[u32], b: &[u32]| {
        let same = a == b;
        ok &= same;
        println!("  {name}: {}{}", if same { "SAME" } else { "DIFFERS" }, a.iter().zip(b).position(|(x, y)| x != y).map_or(String::new(), |i| format!(" at {i}")));
    };
    let set = |e: &mut Engine, temp: f32, k: usize, p: f32, pres: f32| {
        e.temperature = temp;
        e.seed = 1234;
        let s = e.sampler_mut().unwrap();
        s.top_k = k;
        s.top_p = p;
        s.presence = pres;
    };
    println!("sampler test, {} prompt ids, {n} tokens:", ids.len());
    set(&mut e, 0.0, 1, 1.0, 0.0);
    let g = decode_n(&mut e, &ids, n, true)?;
    check("greedy, sampler top-k 1 vs argmax", &g, &base);
    for (name, t, k, p, pres) in [("thinking card (T 1.0, k 20, p 0.95)", 1.0, 20, 0.95, 0.0), ("answer card (T 0.7, k 20, p 0.8, presence 1.5)", 0.7, 20, 0.8, 1.5), ("greedy + presence 1.5", 0.0, 20, 1.0, 1.5)] {
        set(&mut e, t, k, p, pres);
        let a = decode_n(&mut e, &ids, n, false)?;
        let b = decode_n(&mut e, &ids, n, true)?;
        check(&format!("{name}: drafts vs none"), &b, &a);
        let c = decode_n(&mut e, &ids, n, true)?;
        check(&format!("{name}: repeat"), &c, &a);
        let differs = a.iter().zip(&base).filter(|(x, y)| x != y).count();
        println!("    differs from greedy in {differs} of {n}; first {:?}", &a[..12.min(a.len())]);
    }
    println!("sampler test: {}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
