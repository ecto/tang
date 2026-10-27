//! DeepSeek-V4.1-Flash behind the OpenAI-compatible server ([`crate::server`]): the chat
//! template, `tokenizer.json`, the device forward, and DeepSeek's DSML tool calls.
//!
//! Deliberately minimal, like GLM's first serve glue:
//! - One sequence at a time. Each request prefills its whole prompt in one chunk at position
//!   0 and then decodes; there is no prefix reuse across requests yet.
//! - Requests decode speculatively with DSpark (`DS_SPEC=0` turns it off): each forward
//!   verifies the sampled token and its drafts as one window ([`forward::generate`]). Greedy
//!   gives the same tokens as plain greedy decoding; sampled requests use speculative
//!   (rejection) sampling, the same distribution as plain temperature/top-p sampling
//!   ([`SpecSampler`]). Sampled spec needs the engine's `decode_spec` to take a
//!   [`forward::SpecPick`] ([`SPEC_SAMPLING`]); until then sampled requests decode one token
//!   at a time.
//! - Sampling on the host from the last logits: greedy at temperature 0, else top-p.
//!
//! The reply streams as reasoning (inside `<think>`), text, and tool calls. DeepSeek writes a
//! tool call as
//! `<｜DSML｜ calls><｜DSML｜ invoke name="N"><｜DSML｜ parameter name="K" string="true">V
//! </｜DSML｜ parameter>…</｜DSML｜ invoke></｜DSML｜ calls>`, where `｜DSML｜` is one token;
//! [`Dsml`] turns that into [`Piece::ToolCall`]s.

use super::engine::GpuModel;
use super::forward::{self, Decoder, SpecPick, SpecSampler, SpecStep};
use super::Config;
use crate::chat::{Chunk, Detok, Parser, Piece, Tags, Template};
use crate::engine::{Finish, Request, Usage};
use crate::server::Backend;
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;
use std::time::Instant;

const DSML: &str = "｜DSML｜";

pub struct DsServe {
    m: GpuModel,
    tok: tokenizers::Tokenizer,
    tpl: Template,
    tags: Tags,
    eos: Vec<u32>,
    ctx: usize,
}

impl DsServe {
    /// `dir`: the checkpoint with `tokenizer.json`, `tokenizer_config.json`,
    /// `chat_template.jinja`, `inference/config.json` and `engram.json`.
    pub fn load(dir: &Path) -> Result<Self> {
        let t = Instant::now();
        let m = GpuModel::open(dir)?;
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
        let tpl = Template::load(dir)?;
        let tags = Tags::new(|s| tok.token_to_id(s));
        let eos = ["<｜end▁of▁sentence｜>"]
            .iter()
            .filter_map(|s| tok.token_to_id(s))
            .collect::<Vec<_>>();
        if eos.is_empty() {
            bail!("tokenizer has no <｜end▁of▁sentence｜>");
        }
        let ctx = m.cpu.cfg.max_seq_len;
        eprintln!(
            "tang-llm: deepseek ready in {:.1} s: ctx {ctx}, expert slots {}",
            t.elapsed().as_secs_f64(),
            if m.expert_slots == 0 {
                "all".to_string()
            } else {
                m.expert_slots.to_string()
            }
        );
        Ok(Self {
            m,
            tok,
            tpl,
            tags,
            eos,
            ctx,
        })
    }
}

/// Whether the GPU `decode_spec` takes a [`SpecPick`] (speculative sampling). The port: give
/// `GpuModel::forward_spec` a `draw: &mut dyn FnMut(&[f32]) -> usize` for the Markov loop
/// (as `Model::forward_spec_with`), and `GpuModel::decode_spec` a trailing
/// `pick: &mut dyn SpecPick` used as `Model::decode_spec` does (`forward::verify_rows`); then
/// flip this and call it below.
const SPEC_SAMPLING: bool = true;

impl Decoder for GpuModel {
    fn config(&self) -> &Config {
        &self.cpu.cfg
    }
    fn feed(&mut self, ids: &[usize], sp: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward(ids, sp, None)
    }
    fn seed_drafter(&mut self, main: &[f32]) -> Result<()> {
        self.forward_spec(0, main, 0).map(|_| ())
    }
    fn spec_ok(&self, greedy: bool) -> bool {
        greedy || SPEC_SAMPLING
    }
    fn spec_step(
        &mut self,
        token: usize,
        main: &[f32],
        sp: usize,
        pick: &mut dyn SpecPick,
    ) -> Result<SpecStep> {
        self.decode_spec(token, main, sp, &mut |_, _| {}, pick)
    }
}

/// Pulls DSML tool calls out of the text stream; everything else passes through. Holds back
/// text that could still turn into `<｜DSML｜ calls>`.
#[derive(Default)]
pub struct Dsml {
    buf: String,
    in_calls: bool,
}

impl Dsml {
    fn open() -> String {
        format!("<{DSML} calls>")
    }

    fn close() -> String {
        format!("</{DSML} calls>")
    }

    pub fn push(&mut self, s: &str) -> Vec<Piece> {
        self.buf.push_str(s);
        let mut out = Vec::new();
        loop {
            if self.in_calls {
                let Some(end) = self.buf.find(&Self::close()) else {
                    break;
                };
                out.extend(parse_calls(&self.buf[..end]));
                self.buf.drain(..end + Self::close().len());
                self.in_calls = false;
                continue;
            }
            let open = Self::open();
            if let Some(at) = self.buf.find(&open) {
                let before = self.buf[..at].trim_end().to_string();
                if !before.is_empty() {
                    out.push(Piece::Text(before));
                }
                self.buf.drain(..at + open.len());
                self.in_calls = true;
                continue;
            }
            // flush all but a tail that may be the start of the marker
            let keep = (1..open.len().min(self.buf.len()) + 1)
                .rev()
                .find(|&n| {
                    self.buf.is_char_boundary(self.buf.len() - n)
                        && open.starts_with(&self.buf[self.buf.len() - n..])
                })
                .unwrap_or(0);
            // and the whitespace before it: the template puts "\n\n" ahead of a calls block,
            // which a tool call drops
            let cut = self.buf[..self.buf.len() - keep].trim_end().len();
            if cut > 0 {
                out.push(Piece::Text(self.buf[..cut].to_string()));
                self.buf.drain(..cut);
            }
            break;
        }
        out
    }

    pub fn finish(&mut self) -> Vec<Piece> {
        let rest = std::mem::take(&mut self.buf);
        if self.in_calls {
            // an unterminated block: parse what's there
            return parse_calls(&rest);
        }
        if rest.is_empty() {
            Vec::new()
        } else {
            vec![Piece::Text(rest)]
        }
    }
}

/// The invokes inside a `calls` block.
fn parse_calls(body: &str) -> Vec<Piece> {
    let inv_open = format!("<{DSML} invoke name=\"");
    let inv_close = format!("</{DSML} invoke>");
    let par_open = format!("<{DSML} parameter name=\"");
    let par_close = format!("</{DSML} parameter>");
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(i) = rest.find(&inv_open) {
        rest = &rest[i + inv_open.len()..];
        let Some(q) = rest.find('"') else { break };
        let name = rest[..q].to_string();
        let end = rest.find(&inv_close).unwrap_or(rest.len());
        let mut inv = &rest[q..end];
        let mut args = serde_json::Map::new();
        while let Some(j) = inv.find(&par_open) {
            inv = &inv[j + par_open.len()..];
            let Some(q) = inv.find('"') else { break };
            let key = inv[..q].to_string();
            let Some(gt) = inv.find('>') else { break };
            let is_string = inv[q..gt].contains("string=\"true\"");
            let vend = inv[gt + 1..]
                .find(&par_close)
                .map_or(inv.len(), |e| gt + 1 + e);
            let raw = &inv[gt + 1..vend];
            let v = if is_string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
            };
            args.insert(key, v);
            inv = &inv[vend.min(inv.len())..];
        }
        out.push(Piece::ToolCall {
            name,
            arguments: Value::Object(args),
        });
        rest = &rest[end.min(rest.len())..];
    }
    out
}

impl Backend for DsServe {
    fn complete(
        &mut self,
        req: &Request,
        on: &mut dyn FnMut(Piece) -> bool,
    ) -> Result<(Finish, Usage)> {
        let think = req.think.unwrap_or(true);
        let prompt = self
            .tpl
            .render(&req.messages, req.tools.as_ref(), Some(think))?;
        let enc = self
            .tok
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let ids: Vec<usize> = enc.get_ids().iter().map(|&i| i as usize).collect();
        let max_new = req
            .max_tokens
            .unwrap_or(self.ctx)
            .min(self.ctx.saturating_sub(ids.len()));
        if ids.is_empty() || max_new == 0 {
            bail!(
                "prompt of {} tokens leaves no room in a {}-token context",
                ids.len(),
                self.ctx
            );
        }
        let mut usage = Usage::default();
        self.m.reset()?;
        let mut pick = SpecSampler::new(
            req.sampling.temperature,
            req.sampling.top_p,
            req.sampling.seed ^ 0x5eed,
        );
        let (mut detok, mut parser, mut dsml) =
            (Detok::default(), Parser::new(think), Dsml::default());
        let mut out: Vec<u32> = Vec::new();
        let mut called = false;
        let mut emit = |pieces: Vec<Piece>, dsml: &mut Dsml, called: &mut bool| -> bool {
            for p in pieces {
                let ps = match p {
                    Piece::Text(s) => dsml.push(&s),
                    other => vec![other],
                };
                for p in ps {
                    *called |= matches!(p, Piece::ToolCall { .. });
                    if !on(p) {
                        return false;
                    }
                }
            }
            true
        };
        let spec = std::env::var("DS_SPEC").map_or(true, |v| v != "0");
        let eos: Vec<usize> = self.eos.iter().map(|&e| e as usize).collect();
        let (tok, tags, stop) = (&self.tok, &self.tags, &req.stop);
        let decode = |ids: &[u32]| -> Result<String> {
            tok.decode(ids, false)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))
        };
        let mut err = None;
        let mut finish = forward::generate(
            &mut self.m,
            &ids,
            max_new,
            &mut pick,
            spec,
            &eos,
            &mut usage,
            &mut |next| {
                out.push(next as u32);
                let chunks = match detok.push(&out, tags, decode) {
                    Ok(c) => c,
                    Err(e) => {
                        err = Some(e);
                        return Some(Finish::Cancelled);
                    }
                };
                for ch in chunks {
                    let pieces = match ch {
                        Chunk::Text(s) => parser.push_text(&s),
                        Chunk::Tag(t) => parser.push_tag(t),
                    };
                    if !emit(pieces, &mut dsml, &mut called) {
                        return Some(Finish::Cancelled);
                    }
                }
                if !stop.is_empty() {
                    match decode(&out) {
                        Ok(text) if stop.iter().any(|s| text.ends_with(s.as_str())) => {
                            return Some(Finish::Stop)
                        }
                        Ok(_) => {}
                        Err(e) => {
                            err = Some(e);
                            return Some(Finish::Cancelled);
                        }
                    }
                }
                None
            },
        )
        .context("decode")?;
        if let Some(e) = err {
            return Err(e);
        }
        let tail = parser.finish();
        emit(tail, &mut dsml, &mut called);
        for p in dsml.finish() {
            called |= matches!(p, Piece::ToolCall { .. });
            on(p);
        }
        if called && finish == Finish::Stop {
            finish = Finish::ToolCalls;
        }
        Ok((finish, usage))
    }

    fn context_window(&self) -> usize {
        self.ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsml_tool_calls_parse_across_chunks() {
        let text = format!(
            "Let me look.\n\n<{DSML} calls>\n<{DSML} invoke name=\"read\">\n<{DSML} parameter name=\"path\" string=\"true\">src/main.rs</{DSML} parameter>\n<{DSML} parameter name=\"limit\" string=\"false\">40</{DSML} parameter>\n</{DSML} invoke>\n<{DSML} invoke name=\"ls\">\n</{DSML} invoke>\n</{DSML} calls>"
        );
        // feed it a few characters at a time, as a token stream would
        let mut d = Dsml::default();
        let mut got = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        for c in chars.chunks(3) {
            got.extend(d.push(&c.iter().collect::<String>()));
        }
        got.extend(d.finish());
        let text_out: String = got
            .iter()
            .filter_map(|p| {
                if let Piece::Text(s) = p {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text_out, "Let me look.");
        let calls: Vec<_> = got
            .iter()
            .filter(|p| matches!(p, Piece::ToolCall { .. }))
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[0],
            &Piece::ToolCall {
                name: "read".into(),
                arguments: serde_json::json!({"path": "src/main.rs", "limit": 40})
            }
        );
        assert_eq!(
            calls[1],
            &Piece::ToolCall {
                name: "ls".into(),
                arguments: serde_json::json!({})
            }
        );
    }

    #[test]
    fn plain_text_passes_through() {
        let mut d = Dsml::default();
        let mut got = d.push("a < b and <｜");
        got.extend(d.push("x"));
        got.extend(d.finish());
        let s: String = got
            .iter()
            .filter_map(|p| {
                if let Piece::Text(s) = p {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(s, "a < b and <｜x");
    }

    #[test]
    fn sampling_is_greedy_at_zero_and_seeded_otherwise() {
        let l = vec![0.1, 2.0, 1.9, -1.0];
        assert_eq!(SpecSampler::new(0.0, 1.0, 1).sample(&l), 1);
        let (mut a, mut b) = (
            SpecSampler::new(1.0, 0.95, 7),
            SpecSampler::new(1.0, 0.95, 7),
        );
        let xs: Vec<usize> = (0..20).map(|_| a.sample(&l)).collect();
        let ys: Vec<usize> = (0..20).map(|_| b.sample(&l)).collect();
        assert_eq!(xs, ys);
        assert!(xs.iter().all(|&i| i < 3), "top-p drops the tail");
    }
}
