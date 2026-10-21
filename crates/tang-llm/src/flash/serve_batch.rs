//! Continuous batching for `flash serve --batch N` (qwen3next only, `docs/qwen3next-batching.md`):
//! the scheduler half. The engine half is the Coder's sequence slots (`alloc_slot`,
//! `prefill_slot`, `decode_batch`).
//!
//! - **Slots and prefix reuse.** A finished request's slot is parked with its tokens. A new
//!   request takes the parked slot whose tokens are a prefix of its prompt (an agent's next turn
//!   extends the conversation), so only the new tokens are prefilled; otherwise a fresh slot,
//!   evicting the least recently used parked one when all are taken. The recurrent state can't
//!   rewind, so a slot is only ever extended.
//! - **A step.** Prefilling streams advance by a chunk (at most `PREFILL_BUDGET` tokens a step
//!   across them, so a long prompt doesn't stall the decoding ones), then one batched decode
//!   window runs every decoding stream's next token (up to `MAX_T` rows a window).
//! - **Greedy.** Batched windows sample greedily until the engine has per-row sampling; a
//!   request that asks for a temperature is logged and runs greedy.
//! - No drafts in batched windows yet (per-slot verify is engine work still to come).

use super::{FlashServe, Memo};
use crate::chat::{Chunk, Detok, Parser, Piece};
use crate::engine::{Finish, Outcome, Request, Usage};
use crate::server::{DynLane, Inflight};
use anyhow::{bail, ensure, Result};
use serde_json::Value;
use std::time::Instant;
use tang_compute::flash::MAX_T;

/// Prompt tokens prefilled per step across all prefilling streams.
const PREFILL_BUDGET: usize = 256;

/// A parked or running sequence slot.
pub(super) struct SlotInfo {
    pub id: usize,
    pub busy: bool,
    pub key: Option<String>,
    pub used: u64,
}

/// One request in the batch.
pub(super) struct Stream {
    slot: usize,
    key: Option<String>,
    ids: Vec<u32>,
    /// Prompt tokens already in the slot (reused or prefilled).
    done_prompt: usize,
    reused: usize,
    /// The sampled token not yet fed to the slot.
    pending: Option<u32>,
    out: Vec<u32>,
    limit: usize,
    stop: Vec<String>,
    parser: Parser,
    detok: Detok,
    text_out: String,
    content: String,
    calls: Vec<(String, Value)>,
    tail: String,
    finish: Option<Finish>,
    t0: Instant,
    first: Option<f64>,
    prefill_s: f64,
    decode_from: Option<Instant>,
    prefill_only: bool,
}

impl FlashServe {
    pub(super) fn batch_can_admit(&self, key: Option<&str>) -> bool {
        let busy = self.slots.iter().filter(|s| s.busy).count();
        busy < self.batch
            && !(key.is_some() && self.slots.iter().any(|s| s.busy && s.key.as_deref() == key))
    }

    /// A slot for prompt `ids`: a parked one whose tokens it extends (the longest), else a
    /// fresh one, evicting the least recently used parked slot when none is free.
    fn take_slot(&mut self, ids: &[u32], key: Option<&str>) -> Result<(usize, usize)> {
        self.clock += 1;
        let best = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.busy)
            .filter_map(|(i, s)| {
                let t = self.e.slot_tokens(s.id);
                (t.len() < ids.len() && ids.starts_with(t)).then_some((i, t.len()))
            })
            .max_by_key(|&(_, n)| n);
        if let Some((i, n)) = best {
            let s = &mut self.slots[i];
            s.busy = true;
            s.key = key.map(String::from);
            s.used = self.clock;
            return Ok((s.id, n));
        }
        let id = match self.e.alloc_slot() {
            Ok(id) => id,
            Err(_) => {
                // All allocated: drop the least recently used parked slot.
                let (i, _) = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| !s.busy)
                    .min_by_key(|(_, s)| s.used)
                    .ok_or_else(|| anyhow::anyhow!("every slot is running"))?;
                let old = self.slots.remove(i);
                self.e.free_slot(old.id);
                self.e.alloc_slot()?
            }
        };
        self.slots.push(SlotInfo {
            id,
            busy: true,
            key: key.map(String::from),
            used: self.clock,
        });
        Ok((id, 0))
    }

    pub(super) fn batch_admit(&mut self, req: &Request) -> Result<Inflight> {
        self.expire_memory();
        let t0 = Instant::now();
        let (ids, text, _spliced) = self.prompt(req)?;
        ensure!(
            ids.len() + 1 < self.ctx,
            "prompt is {} tokens; the context window is {}",
            ids.len(),
            self.ctx
        );
        if req.temperature_set && req.sampling.temperature > 0.0 {
            eprintln!(
                "tang-llm: batched windows are greedy; ignoring temperature {}",
                req.sampling.temperature
            );
        }
        let (slot, reused) = self.take_slot(&ids, req.cache_key.as_deref())?;
        let limit = req
            .max_tokens
            .unwrap_or(usize::MAX)
            .min(self.ctx - ids.len() - 1)
            .max(1);
        let gen_at = text
            .rfind("<|im_start|>assistant\n")
            .map_or(text.len(), |i| i + "<|im_start|>assistant\n".len());
        let opened = text.trim_end().ends_with("<think>");
        self.requests += 1;
        Ok(Box::new(Stream {
            slot,
            key: req.cache_key.clone(),
            done_prompt: reused,
            reused,
            ids,
            pending: None,
            out: Vec::new(),
            limit,
            stop: req.stop.clone(),
            parser: Parser::new(opened).with_tools(req.tools.as_ref()),
            detok: Detok::default(),
            text_out: String::new(),
            content: String::new(),
            calls: Vec::new(),
            tail: text[gen_at..].to_string(),
            finish: None,
            t0,
            first: None,
            prefill_s: 0.0,
            decode_from: None,
            prefill_only: req.prefill_only,
        }))
    }

    /// Feed one sampled token to stream `s`'s output: end of turn, stop strings, the parser,
    /// delivery. Sets `s.finish` when the stream is done.
    fn batch_take(
        &self,
        s: &mut Stream,
        tok: u32,
        on: &mut dyn FnMut(Piece) -> bool,
    ) -> Result<()> {
        if self.eos.contains(&tok) {
            s.finish = Some(Finish::Stop);
            return Ok(());
        }
        s.out.push(tok);
        for chunk in s
            .detok
            .push(&s.out, &self.tags, |ids| self.tok.decode(ids))?
        {
            let (new, tag) = match chunk {
                Chunk::Text(t) => (t, None),
                Chunk::Tag(t) => (t.to_string(), Some(t)),
            };
            s.text_out.push_str(&new);
            let stop_at = s
                .stop
                .iter()
                .filter_map(|x| s.text_out.find(x.as_str()))
                .min();
            let new = match stop_at {
                Some(i) => new[..new.len().saturating_sub(s.text_out.len() - i)].to_string(),
                None => new,
            };
            let pieces = match tag {
                Some(t) if new == t => s.parser.push_tag(t),
                _ => s.parser.push_text(&new),
            };
            for p in pieces {
                if s.first.is_none() {
                    s.first = Some(s.t0.elapsed().as_secs_f64() * 1e3);
                }
                match &p {
                    Piece::Text(t) => s.content.push_str(t),
                    Piece::ToolCall { name, arguments } => {
                        s.calls.push((name.clone(), arguments.clone()))
                    }
                    Piece::Reasoning(_) => {}
                }
                if !on(p) {
                    s.finish = Some(Finish::Cancelled);
                    return Ok(());
                }
            }
            if stop_at.is_some() {
                s.finish = Some(Finish::Stop);
                return Ok(());
            }
        }
        if s.out.len() >= s.limit {
            s.finish = Some(Finish::Length);
        }
        Ok(())
    }

    pub(super) fn batch_step(&mut self, lanes: &mut [DynLane<'_>]) -> Result<()> {
        // Cancellations first.
        for l in lanes.iter_mut() {
            let s = l.stream.downcast_mut::<Stream>().expect("a flash stream");
            if s.finish.is_none() && l.ctl.cancelled() {
                s.finish = Some(Finish::Cancelled);
            }
        }
        // Prefill: up to PREFILL_BUDGET prompt tokens across the prefilling streams.
        let mut budget = PREFILL_BUDGET;
        for l in lanes.iter_mut() {
            if budget == 0 {
                break;
            }
            let s = l.stream.downcast_mut::<Stream>().expect("a flash stream");
            if s.finish.is_some() || s.done_prompt >= s.ids.len() {
                continue;
            }
            let n = (s.ids.len() - s.done_prompt).min(budget);
            let t = Instant::now();
            let next = self
                .e
                .prefill_slot(s.slot, &s.ids[s.done_prompt..s.done_prompt + n])?;
            s.prefill_s += t.elapsed().as_secs_f64();
            s.done_prompt += n;
            budget -= n;
            if s.done_prompt == s.ids.len() {
                if s.prefill_only {
                    s.finish = Some(Finish::Stop);
                } else {
                    s.decode_from = Some(Instant::now());
                    self.batch_take(s, next, l.on)?;
                    if s.finish.is_none() {
                        s.pending = Some(next);
                    }
                }
            }
        }
        // Decode: every stream with a pending token, MAX_T rows a window.
        let ready: Vec<usize> = lanes
            .iter_mut()
            .enumerate()
            .filter_map(|(i, l)| {
                let s = l.stream.downcast_ref::<Stream>().expect("a flash stream");
                (s.finish.is_none() && s.pending.is_some()).then_some(i)
            })
            .collect();
        for group in ready.chunks(MAX_T) {
            let rows: Vec<(usize, u32)> = group
                .iter()
                .map(|&i| {
                    let s = lanes[i].stream.downcast_ref::<Stream>().unwrap();
                    (s.slot, s.pending.unwrap())
                })
                .collect();
            let next = self.e.decode_batch(&rows)?;
            for (&i, &tok) in group.iter().zip(&next) {
                let l = &mut lanes[i];
                let s = l.stream.downcast_mut::<Stream>().unwrap();
                s.pending = None;
                self.batch_take(s, tok, l.on)?;
                if s.finish.is_none() {
                    s.pending = Some(tok);
                }
            }
        }
        Ok(())
    }

    pub(super) fn batch_state(s: &Inflight) -> (bool, bool) {
        let s = s.downcast_ref::<Stream>().expect("a flash stream");
        (s.finish.is_some(), s.done_prompt < s.ids.len())
    }

    pub(super) fn batch_retire(&mut self, s: Inflight) -> Result<Outcome> {
        let Ok(mut s) = s.downcast::<Stream>() else {
            bail!("not a flash stream")
        };
        let finish = s.finish.unwrap_or(Finish::Cancelled);
        let mut finish = finish;
        if finish != Finish::Cancelled {
            let _ = s.parser.finish();
            if !s.calls.is_empty() && finish == Finish::Stop {
                finish = Finish::ToolCalls;
            }
        }
        // Park the slot. The pending token (if any) was sampled but never fed: the slot's
        // tokens are the prompt plus the output before it.
        if let Some(sl) = self.slots.iter_mut().find(|x| x.id == s.slot) {
            sl.busy = false;
            sl.key = s.key.clone();
        }
        if matches!(finish, Finish::Stop | Finish::ToolCalls)
            && s.stop.iter().all(|x| !s.text_out.contains(x.as_str()))
        {
            self.memo.push_back(Memo {
                content: s.content.clone(),
                calls: s.calls.clone(),
                tail: s.tail.clone(),
                ids: s.out.clone(),
                at: crate::retention::now(),
            });
            while self.memo.len() > 256 {
                self.memo.pop_front();
            }
        }
        let decode_s = s.decode_from.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let prefilled = s.ids.len() - s.reused;
        eprintln!(
            "tang-llm: batch req: slot {} prompt {} (reused {}, prefilled {} in {:.0} ms) | ttft {:.0} ms | decode {} tokens in {:.2} s = {:.1} tok/s | {} tool calls",
            s.slot,
            s.ids.len(),
            s.reused,
            prefilled,
            s.prefill_s * 1e3,
            s.first.unwrap_or(0.0),
            s.out.len(),
            decode_s,
            s.out.len() as f64 / decode_s.max(1e-9),
            s.calls.len()
        );
        Ok(Outcome::Done(
            finish,
            Usage {
                prompt_tokens: s.ids.len(),
                cached_tokens: s.reused,
                completion_tokens: s.out.len(),
                prefill_tok_s: prefilled as f64 / s.prefill_s.max(1e-9),
                decode_tok_s: s.out.len() as f64 / decode_s.max(1e-9),
                ..Default::default()
            },
        ))
    }
}
