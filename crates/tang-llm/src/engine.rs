//! A loaded model plus tokenizer and chat template: turns chat requests into streamed pieces,
//! reusing the KV cache for whatever prefix the new prompt shares with the previous one.

use crate::chat::{Parser, Piece, Template};
use crate::draft::{Calibration, DraftConfig, Global, Session};
use crate::model::{Cache, Dtype, Model};
use crate::sample::{Sampler, Sampling};
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::path::Path;
use std::time::Instant;
use tang_compute::ComputeDevice;
use tokenizers::Tokenizer;

/// Prefill in chunks to bound scratch memory.
const PREFILL_CHUNK: usize = 512;

/// Where an image goes in a flattened message; expanded to the model's image tokens.
pub const IMAGE_MARKER: &str = "<start_of_image>";

pub struct Request {
    pub messages: Value,
    /// Encoded images (PNG, JPEG, ...), in the order their markers appear.
    pub images: Vec<Vec<u8>>,
    pub tools: Option<Value>,
    pub think: Option<bool>,
    /// Most tokens to spend inside `<think>` before the reasoning is wrapped up (see
    /// [`ThinkBudget`]); `None` is unlimited.
    pub thinking_budget: Option<usize>,
    pub sampling: Sampling,
    pub max_tokens: Option<usize>,
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Finish {
    Stop,
    Length,
    ToolCalls,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    /// Of the completion, tokens inside `<think>` (including any wrap-up the budget injected).
    pub reasoning_tokens: usize,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    /// Speculative decoding: draft tokens verified, and how many of them were kept.
    pub draft_tokens: usize,
    pub accepted_tokens: usize,
}

/// Speculative decoding state that outlives a request.
pub struct Speculation {
    pub cfg: DraftConfig,
    pub global: Global,
    pub calib: Calibration,
}

pub struct Engine<D: ComputeDevice> {
    pub model: Model<D>,
    tok: Tokenizer,
    template: Template,
    cache: Cache<D::Buffer>,
    /// Images in the cache: where each image's tokens start, and a hash of the image, so a
    /// shared prefix is only reused when the images in it are the same.
    cache_images: Vec<(usize, u64)>,
    eos: Vec<u32>,
    /// `<think>` and `</think>`, when the model has them.
    think_tags: Option<(u32, u32)>,
    /// What a spent thinking budget appends: a wrap-up sentence, `</think>`, a blank line.
    wrap_up: Vec<u32>,
    /// Suffix drafting with verification, when on.
    spec: Option<Speculation>,
}

/// Qwen3's suggested way to end reasoning early: say so, then close the block.
const WRAP_UP: &str =
    "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n";

/// Caps reasoning at a token budget. Fed each generated token, it tracks whether decoding is
/// inside `<think>` and how much it has spent there, and says when the budget runs out so the
/// engine can append the wrap-up (through `</think>`) as if the model had written it.
#[derive(Debug)]
pub struct ThinkBudget {
    tags: Option<(u32, u32)>,
    budget: Option<usize>,
    inside: bool,
    spent: usize,
    /// Tokens inside `<think>` so far, the tags and any wrap-up included.
    pub reasoning: usize,
    /// The budget ran out and the reasoning was closed.
    pub hit: bool,
}

impl ThinkBudget {
    /// `inside`: the prompt already opened `<think>`. Without tags nothing is ever cut.
    pub fn new(tags: Option<(u32, u32)>, budget: Option<usize>, inside: bool) -> Self {
        Self {
            tags,
            budget,
            inside: inside && tags.is_some(),
            spent: 0,
            reasoning: 0,
            hit: false,
        }
    }

    /// Note a generated token. True when the reasoning must be closed now.
    pub fn step(&mut self, tok: u32) -> bool {
        let Some((open, close)) = self.tags else {
            return false;
        };
        if tok == close {
            self.reasoning += self.inside as usize;
            self.inside = false;
            return false;
        }
        if tok == open {
            self.inside = true;
        } else if self.inside {
            self.spent += 1;
        }
        if !self.inside {
            return false;
        }
        self.reasoning += 1;
        self.budget.is_some_and(|b| !self.hit && self.spent >= b)
    }

    /// The engine appended `n` tokens of wrap-up, ending with `</think>`.
    pub fn closed(&mut self, n: usize) {
        self.reasoning += n;
        self.inside = false;
        self.hit = true;
    }
}

impl<D: ComputeDevice> Engine<D> {
    pub fn load(dev: D, dir: &Path, max_ctx: usize, dtype: Dtype) -> Result<Self> {
        let model = Model::load(dev, dir, max_ctx, dtype)?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let template = Template::load(dir)?;
        let mut eos = model.cfg.eos();
        // generation_config.json often lists more stop ids (e.g. <|im_end|> and <|endoftext|>).
        let gen: Option<Value> = std::fs::read(dir.join("generation_config.json"))
            .ok()
            .and_then(|g| serde_json::from_slice(&g).ok());
        if let Some(g) = gen {
            match &g["eos_token_id"] {
                Value::Number(n) => eos.extend(n.as_u64().map(|n| n as u32)),
                Value::Array(a) => {
                    eos.extend(a.iter().filter_map(|n| n.as_u64().map(|n| n as u32)))
                }
                _ => {}
            }
        }
        for t in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"] {
            eos.extend(tok.token_to_id(t));
        }
        eos.sort_unstable();
        eos.dedup();
        let cache = model.new_cache();
        let think_tags = tok.token_to_id("<think>").zip(tok.token_to_id("</think>"));
        let mut wrap_up = Vec::new();
        if let Some((_, close)) = think_tags {
            let enc = |s: &str| -> Result<Vec<u32>> {
                Ok(tok
                    .encode(s, false)
                    .map_err(|e| anyhow!("tokenize: {e}"))?
                    .get_ids()
                    .to_vec())
            };
            wrap_up = enc(WRAP_UP)?;
            wrap_up.push(close);
            wrap_up.extend(enc("\n\n")?);
        }
        Ok(Self {
            model,
            tok,
            template,
            cache,
            cache_images: Vec::new(),
            eos,
            think_tags,
            wrap_up,
            spec: None,
        })
    }

    /// Turn speculative decoding on (with these settings) or off. Outputs don't change.
    pub fn set_speculation(&mut self, cfg: Option<DraftConfig>) {
        self.spec = cfg.map(|cfg| Speculation {
            global: Global::new(cfg.global_tokens, cfg.store.clone()),
            calib: Calibration::default(),
            cfg,
        });
    }

    pub fn speculation(&self) -> Option<&Speculation> {
        self.spec.as_ref()
    }

    /// Forget the KV cache (the next request prefills from scratch).
    pub fn reset(&mut self) {
        self.cache.truncate(0);
        self.cache_images.clear();
    }

    /// Tokens whose keys and values are in the cache.
    pub fn cached_tokens(&self) -> &[u32] {
        &self.cache.tokens
    }

    pub fn context_window(&self) -> usize {
        self.model.max_ctx()
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.encode(text)?.len())
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Generate a reply. `on` gets each piece as it's decoded and returns false to stop.
    pub fn complete(
        &mut self,
        req: &Request,
        mut on: impl FnMut(Piece) -> bool,
    ) -> Result<(Finish, Usage)> {
        let mut prompt = self
            .template
            .render(&req.messages, req.tools.as_ref(), req.think)?;
        if !req.images.is_empty() {
            let per = self
                .model
                .vision
                .as_ref()
                .map(|v| v.tokens)
                .ok_or_else(|| anyhow!("this model can't see images"))?;
            anyhow::ensure!(
                prompt.matches(IMAGE_MARKER).count() == req.images.len(),
                "the chat template dropped some images"
            );
            // As the processor does: each marker becomes a block of image tokens.
            let block = format!(
                "\n\n<start_of_image>{}<end_of_image>\n\n",
                "<image_soft_token>".repeat(per)
            );
            prompt = prompt.replace(IMAGE_MARKER, &block);
        }
        let ids = self.encode(&prompt)?;
        // Where each image's tokens start, with a hash of the image.
        let img_tok = self.model.cfg.image_token;
        let mut runs: Vec<(usize, u64)> = Vec::new();
        for (i, &t) in ids.iter().enumerate() {
            if !req.images.is_empty() && Some(t) == img_tok && (i == 0 || ids[i - 1] != t) {
                let bytes = &req.images[runs.len().min(req.images.len().saturating_sub(1))];
                runs.push((i, hash(bytes)));
            }
        }
        let ctx = self.model.max_ctx();
        anyhow::ensure!(
            ids.len() < ctx,
            "prompt is {} tokens; the context window is {ctx}",
            ids.len()
        );
        let limit = req.max_tokens.unwrap_or(usize::MAX).min(ctx - ids.len());

        // Reuse the shared prefix (at least one token must be fed to get logits).
        let shared = self
            .cache
            .tokens
            .iter()
            .zip(&ids)
            .take_while(|(a, b)| a == b)
            .count();
        // Only as far as the images match, and never into the middle of an image's tokens.
        let differ = self
            .cache_images
            .iter()
            .zip(&runs)
            .find(|(a, b)| a != b)
            .map(|(a, b)| a.0.min(b.0))
            .or_else(|| runs.get(self.cache_images.len()).map(|r| r.0))
            .unwrap_or(usize::MAX);
        let mut reuse = shared.min(ids.len() - 1).min(differ);
        let per = self.model.vision.as_ref().map_or(0, |v| v.tokens);
        if let Some(&(start, _)) = runs.iter().find(|(s, _)| *s < reuse && reuse < s + per) {
            reuse = start;
        }
        self.cache.truncate(reuse);

        let t = Instant::now();
        let mut logits = Vec::new();
        if runs.is_empty() {
            for chunk in ids[reuse..].chunks(PREFILL_CHUNK) {
                logits = self.model.forward(chunk, &mut self.cache, false)?;
            }
        } else {
            let size = self.model.vision.as_ref().map_or(896, |v| v.cfg.image_size);
            let mut feats = Vec::new();
            for (i, &(start, _)) in runs.iter().enumerate() {
                if start >= reuse {
                    let px = crate::vision::preprocess(&req.images[i], size)?;
                    feats.push(self.model.encode_image(&px)?);
                }
            }
            logits = self
                .model
                .forward_images(&ids[reuse..], &feats, &mut self.cache, false)?;
        }
        self.cache_images = runs;
        let prefill_s = t.elapsed().as_secs_f64();

        let mut sampler = Sampler::new(req.sampling.clone());
        let opened = prompt.trim_end().ends_with("<think>");
        let mut parser = Parser::new(opened);
        let mut think = ThinkBudget::new(self.think_tags, req.thinking_budget, opened);
        let mut out: Vec<u32> = Vec::new();
        let (mut prefix, mut read) = (0usize, 0usize);
        let mut text = String::new();
        let mut finish = Finish::Length;
        let mut tool_calls = 0;
        let t = Instant::now();

        let mut deliver = |pieces: Vec<Piece>, tool_calls: &mut usize| -> bool {
            for p in pieces {
                if matches!(p, Piece::ToolCall { .. }) {
                    *tool_calls += 1;
                }
                if !on(p) {
                    return false;
                }
            }
            true
        };

        // Speculation: each forward feeds the new token(s) plus a draft, and returns logits for
        // every draft position. Every token is still sampled from the target's logits in order,
        // with the same sampler state, so the output is what plain decoding would produce; a
        // draft token only saves a forward when it equals the sampled token.
        let mut sess = self
            .spec
            .as_ref()
            .map(|s| Session::new(&s.cfg, &s.global, s.calib.clone(), &ids));
        let vocab = self.model.cfg.vocab_size;
        let max_ctx = self.model.max_ctx();
        // Logits rows from the last forward: row r follows the fed tokens and draft[..r].
        let mut rows = logits;
        let mut draft = crate::draft::Draft::default();
        let mut row = 0usize;
        // Draft tokens in the cache that aren't accepted yet.
        let mut pending = 0usize;
        let (mut drafted, mut accepted) = (0usize, 0usize);

        while out.len() < limit {
            let next = sampler.sample(&rows[row * vocab..(row + 1) * vocab]);
            let hit = draft.tokens.get(row) == Some(&next);
            if row < draft.tokens.len() {
                if let Some(s) = sess.as_mut() {
                    s.calib.observe(draft.match_len, draft.shares[row], hit);
                }
            }
            if hit {
                accepted += 1;
                pending -= 1;
            }
            if self.eos.contains(&next) {
                finish = Finish::Stop;
                break;
            }
            out.push(next);
            // What the next forward must feed (a hit is already in the cache).
            let mut fed = if hit { vec![] } else { vec![next] };
            // Out of thinking budget: write the wrap-up and `</think>` for the model (into the
            // KV cache like its own tokens, and streamed as reasoning), then let it answer.
            let wrap = think.step(next) && out.len() + self.wrap_up.len() < limit;
            if wrap {
                out.extend(&self.wrap_up);
                fed.extend(&self.wrap_up);
                think.closed(self.wrap_up.len());
            }
            if let Some(s) = sess.as_mut() {
                let n = out.len() - if wrap { self.wrap_up.len() } else { 0 };
                for &tok in &out[n - 1..] {
                    s.push(tok);
                }
            }
            // Incremental detokenization: decode a small window and emit what's new, holding
            // back incomplete UTF-8.
            let before = self.decode(&out[prefix..read])?;
            let after = self.decode(&out[prefix..])?;
            if after.len() > before.len() && !after.ends_with('\u{fffd}') {
                let new = after[before.len()..].to_string();
                prefix = read;
                read = out.len();
                text.push_str(&new);
                let stop_at = req.stop.iter().filter_map(|s| text.find(s.as_str())).min();
                let new = match stop_at {
                    Some(i) => {
                        let keep = new.len().saturating_sub(text.len() - i);
                        new[..keep].to_string()
                    }
                    None => new,
                };
                if !deliver(parser.push(&new), &mut tool_calls) {
                    finish = Finish::Cancelled;
                    break;
                }
                if stop_at.is_some() {
                    finish = Finish::Stop;
                    break;
                }
            }
            if fed.is_empty() {
                // The draft was right: the next position's logits are already here.
                row += 1;
                continue;
            }
            if out.len() >= limit {
                break;
            }
            // Drop the rejected rest of the draft from the cache.
            self.cache.truncate(self.cache.len - pending);
            // A new draft, unless the wrap-up was just forced in (don't draft across it).
            draft = match sess.as_ref() {
                Some(s) if !wrap => {
                    let room = (limit - out.len())
                        .min(max_ctx.saturating_sub(self.cache.len + fed.len()));
                    s.propose(room)
                }
                _ => crate::draft::Draft::default(),
            };
            let k = draft.tokens.len();
            let m = fed.len();
            fed.extend(&draft.tokens);
            rows = self.model.forward(&fed, &mut self.cache, k > 0)?;
            if k > 0 {
                rows.drain(..(m - 1) * vocab);
            }
            row = 0;
            pending = k;
            drafted += k;
        }
        // Leave only accepted tokens in the cache.
        self.cache.truncate(self.cache.len - pending);
        if finish != Finish::Cancelled {
            deliver(parser.finish(), &mut tool_calls);
            if tool_calls > 0 && finish == Finish::Stop {
                finish = Finish::ToolCalls;
            }
        }
        let calib = sess.map(|s| s.calib);
        if let (Some(calib), Some(spec)) = (calib, self.spec.as_mut()) {
            spec.calib = calib;
            spec.global.push(&out);
        }
        let decode_s = t.elapsed().as_secs_f64();
        Ok((
            finish,
            Usage {
                prompt_tokens: ids.len(),
                cached_tokens: reuse,
                completion_tokens: out.len(),
                reasoning_tokens: think.reasoning,
                prefill_tok_s: (ids.len() - reuse) as f64 / prefill_s.max(1e-9),
                decode_tok_s: out.len() as f64 / decode_s.max(1e-9),
                draft_tokens: drafted,
                accepted_tokens: accepted,
            },
        ))
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok
            .decode(ids, false)
            .map_err(|e| anyhow!("detokenize: {e}"))
            .context("decode")
    }
}

fn hash(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::ThinkBudget;

    const OPEN: u32 = 1;
    const CLOSE: u32 = 2;

    /// Feed tokens; the index of the token after which the budget cut in, if it did.
    fn run(b: &mut ThinkBudget, toks: &[u32]) -> Option<usize> {
        for (i, &t) in toks.iter().enumerate() {
            if b.step(t) {
                b.closed(3);
                return Some(i);
            }
        }
        None
    }

    #[test]
    fn cuts_after_budget_tokens_of_reasoning() {
        let mut b = ThinkBudget::new(Some((OPEN, CLOSE)), Some(3), false);
        // `<think>`, then three reasoning tokens: the third spends the budget.
        assert_eq!(run(&mut b, &[OPEN, 10, 11, 12, 13]), Some(3));
        assert!(b.hit);
        assert_eq!(b.reasoning, 4 + 3);
        // Answer tokens after the close don't count, and a second block isn't cut.
        assert_eq!(run(&mut b, &[20, 21, OPEN, 30, 31, 32, 33]), None);
    }

    #[test]
    fn reasoning_that_ends_in_time_is_left_alone() {
        let mut b = ThinkBudget::new(Some((OPEN, CLOSE)), Some(3), false);
        assert_eq!(run(&mut b, &[OPEN, 10, 11, CLOSE, 20, 21, 22, 23]), None);
        assert!(!b.hit);
        assert_eq!(b.reasoning, 4);
    }

    #[test]
    fn a_prompt_that_opened_think_counts_from_the_first_token() {
        let mut b = ThinkBudget::new(Some((OPEN, CLOSE)), Some(2), true);
        assert_eq!(run(&mut b, &[10, 11, 12]), Some(1));
    }

    #[test]
    fn zero_budget_closes_right_after_the_open_tag() {
        let mut b = ThinkBudget::new(Some((OPEN, CLOSE)), Some(0), false);
        assert_eq!(run(&mut b, &[5, OPEN, 10]), Some(1));
    }

    #[test]
    fn no_budget_or_no_tags_never_cuts() {
        let mut b = ThinkBudget::new(Some((OPEN, CLOSE)), None, false);
        assert_eq!(run(&mut b, &[OPEN, 10, 11, 12, CLOSE]), None);
        assert_eq!(b.reasoning, 5);
        let mut b = ThinkBudget::new(None, Some(0), true);
        assert_eq!(run(&mut b, &[OPEN, 10, 11]), None);
        assert_eq!(b.reasoning, 0);
    }
}
