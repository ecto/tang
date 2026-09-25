//! Chat templates (the checkpoint's own Jinja template, rendered with minijinja) and a streaming
//! parser that splits model output into reasoning, text and tool calls.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;

pub struct Template {
    env: minijinja::Environment<'static>,
    bos: String,
    eos: String,
}

impl Template {
    /// Load `chat_template` (and bos/eos strings) from `tokenizer_config.json`.
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: Value = serde_json::from_slice(
            &std::fs::read(dir.join("tokenizer_config.json")).context("tokenizer_config.json")?,
        )?;
        let source = match &cfg["chat_template"] {
            Value::String(s) => s.clone(),
            // A list of named templates: use "default".
            Value::Array(list) => list
                .iter()
                .find(|t| t["name"] == "default")
                .or(list.first())
                .and_then(|t| t["template"].as_str())
                .context("chat_template list")?
                .to_string(),
            _ => std::fs::read_to_string(dir.join("chat_template.jinja"))
                .context("no chat_template in tokenizer_config.json")?,
        };
        let special = |k: &str| match &cfg[k] {
            Value::String(s) => s.clone(),
            v => v["content"].as_str().unwrap_or_default().to_string(),
        };
        let mut env = minijinja::Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_template_owned("chat", source)?;
        Ok(Self {
            env,
            bos: special("bos_token"),
            eos: special("eos_token"),
        })
    }

    pub fn render(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        think: Option<bool>,
    ) -> Result<String> {
        let t = self.env.get_template("chat")?;
        let tools = tools.filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
        Ok(t.render(minijinja::context! {
            messages => messages,
            tools => tools,
            add_generation_prompt => true,
            enable_thinking => think.unwrap_or(true),
            bos_token => self.bos,
            eos_token => self.eos,
        })?)
    }
}

/// Python's `json.dumps` spacing (`", "` and `": "`), which is what templates were trained with.
fn tojson(v: minijinja::Value) -> Result<minijinja::Value, minijinja::Error> {
    struct Py;
    impl serde_json::ser::Formatter for Py {
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
        ) -> std::io::Result<()> {
            w.write_all(b": ")
        }
    }
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, Py);
    v.serialize(&mut ser).map_err(|e| {
        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
    })?;
    Ok(minijinja::Value::from_safe_string(
        String::from_utf8(out).unwrap_or_default(),
    ))
}

/// What the model is producing, as it streams.
#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Reasoning(String),
    Text(String),
    /// A complete `<tool_call>` body: `{"name": ..., "arguments": ...}`.
    ToolCall {
        name: String,
        arguments: Value,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Text,
    Think,
    Tool,
}

/// Splits streamed text on `<think>`/`</think>` and `<tool_call>`/`</tool_call>`. Holds back
/// anything that could be the start of a tag until it can tell.
pub struct Parser {
    mode: Mode,
    buf: String,
    tool: String,
    /// Text seen so far (to trim the blank lines the template puts around tags).
    text_started: bool,
}

const TAGS: [&str; 4] = ["<think>", "</think>", "<tool_call>", "</tool_call>"];

impl Parser {
    /// `thinking`: the prompt already opened a `<think>` block (some templates do).
    pub fn new(thinking: bool) -> Self {
        Self {
            mode: if thinking { Mode::Think } else { Mode::Text },
            buf: String::new(),
            tool: String::new(),
            text_started: false,
        }
    }

    pub fn push(&mut self, s: &str) -> Vec<Piece> {
        self.buf.push_str(s);
        let mut out = Vec::new();
        loop {
            match self.buf.find('<') {
                None => {
                    let all = std::mem::take(&mut self.buf);
                    self.emit(&all, &mut out);
                    break;
                }
                Some(i) => {
                    let (before, rest) = self.buf.split_at(i);
                    let before = before.to_string();
                    let rest = rest.to_string();
                    if let Some(tag) = TAGS.iter().find(|t| rest.starts_with(**t)) {
                        self.emit(&before, &mut out);
                        self.buf = rest[tag.len()..].to_string();
                        self.tag(tag, &mut out);
                    } else if TAGS.iter().any(|t| t.starts_with(rest.as_str())) {
                        // Could still become a tag: wait for more.
                        self.emit(&before, &mut out);
                        self.buf = rest;
                        break;
                    } else {
                        self.emit(&before, &mut out);
                        self.emit("<", &mut out);
                        self.buf = rest[1..].to_string();
                    }
                }
            }
        }
        out
    }

    /// Flush at end of generation (an unterminated tool call is returned as text).
    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.buf);
        self.emit(&rest, &mut out);
        if self.mode == Mode::Tool && !self.tool.trim().is_empty() {
            out.push(Piece::Text(format!(
                "<tool_call>{}",
                std::mem::take(&mut self.tool)
            )));
        }
        out
    }

    fn tag(&mut self, tag: &str, out: &mut Vec<Piece>) {
        match tag {
            "<think>" => self.mode = Mode::Think,
            "</think>" => self.mode = Mode::Text,
            "<tool_call>" => {
                self.mode = Mode::Tool;
                self.tool.clear();
            }
            _ => {
                self.mode = Mode::Text;
                let body = std::mem::take(&mut self.tool);
                match serde_json::from_str::<Value>(body.trim()) {
                    Ok(v) if v["name"].is_string() => out.push(Piece::ToolCall {
                        name: v["name"].as_str().unwrap_or_default().to_string(),
                        arguments: match &v["arguments"] {
                            Value::String(s) => {
                                serde_json::from_str(s).unwrap_or(Value::String(s.clone()))
                            }
                            Value::Null => Value::Object(Default::default()),
                            a => a.clone(),
                        },
                    }),
                    _ => out.push(Piece::Text(format!("<tool_call>{body}</tool_call>"))),
                }
            }
        }
    }

    fn emit(&mut self, s: &str, out: &mut Vec<Piece>) {
        if s.is_empty() {
            return;
        }
        match self.mode {
            Mode::Tool => self.tool.push_str(s),
            Mode::Think => push(out, Piece::Reasoning(s.to_string())),
            Mode::Text => {
                let s = if self.text_started { s } else { s.trim_start() };
                if !s.is_empty() {
                    self.text_started = true;
                    push(out, Piece::Text(s.to_string()));
                }
            }
        }
    }
}

/// Append, merging with the previous piece of the same kind.
fn push(out: &mut Vec<Piece>, p: Piece) {
    match (out.last_mut(), p) {
        (Some(Piece::Text(a)), Piece::Text(b)) => a.push_str(&b),
        (Some(Piece::Reasoning(a)), Piece::Reasoning(b)) => a.push_str(&b),
        (_, p) => out.push(p),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str]) -> Vec<Piece> {
        let mut p = Parser::new(false);
        let mut out = Vec::new();
        for c in chunks {
            for x in p.push(c) {
                push(&mut out, x);
            }
        }
        for x in p.finish() {
            push(&mut out, x);
        }
        out
    }

    #[test]
    fn splits_reasoning_text_and_tool_calls_across_chunk_boundaries() {
        let out = run(&[
            "<thi",
            "nk>\nplan it\n</th",
            "ink>\n\nOk.",
            "\n<tool_",
            "call>\n{\"name\": \"Read\", ",
            "\"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>",
        ]);
        assert_eq!(
            out,
            vec![
                Piece::Reasoning("\nplan it\n".into()),
                Piece::Text("Ok.\n".into()),
                Piece::ToolCall {
                    name: "Read".into(),
                    arguments: serde_json::json!({"path": "a.rs"})
                },
            ]
        );
    }

    #[test]
    fn stray_angle_brackets_are_text() {
        assert_eq!(
            run(&["a < b and <div>"]),
            vec![Piece::Text("a < b and <div>".into())]
        );
    }

    #[test]
    fn bad_tool_json_falls_back_to_text() {
        assert_eq!(
            run(&["<tool_call>nope</tool_call>"]),
            vec![Piece::Text("<tool_call>nope</tool_call>".into())]
        );
    }
}
