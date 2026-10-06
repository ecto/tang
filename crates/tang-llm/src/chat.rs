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

    /// A template from its Jinja source and bos/eos strings (e.g. a GGUF's metadata).
    pub fn from_source(source: String, bos: String, eos: String) -> Result<Self> {
        let mut env = minijinja::Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_template_owned("chat", source)?;
        Ok(Self { env, bos, eos })
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
///
/// A tool call's body is Hermes JSON (`{"name": ..., "arguments": {...}}`) or Qwen XML
/// (`<function=NAME><parameter=KEY>\nVALUE\n</parameter>...</function>`, see
/// [`parse_xml_call`]). Inside a call only `</tool_call>` is a tag, and an XML call's
/// `</tool_call>` only closes it once the body parses, so values may contain any of the tags.
/// Inside `<think>`, tool-call tags are reasoning text.
pub struct Parser {
    mode: Mode,
    buf: String,
    tool: String,
    /// Text seen so far (to trim the blank lines the template puts around tags).
    text_started: bool,
    /// Tool name → its JSON schema `parameters`, for typing XML parameter values.
    schemas: std::collections::HashMap<String, Value>,
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
            schemas: Default::default(),
        }
    }

    /// The request's `tools` (OpenAI format), so XML parameter values get their schema's types.
    pub fn with_tools(mut self, tools: Option<&Value>) -> Self {
        for t in tools.and_then(Value::as_array).into_iter().flatten() {
            let f = if t["function"].is_object() {
                &t["function"]
            } else {
                t
            };
            if let Some(name) = f["name"].as_str() {
                self.schemas
                    .insert(name.to_string(), f["parameters"].clone());
            }
        }
        self
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
                        if self.is_text(tag) {
                            self.emit(tag, &mut out);
                        } else {
                            self.tag(tag, &mut out);
                        }
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

    /// Flush at end of generation (an unterminated tool call is returned as text, unless it is
    /// a complete XML call that only lacks its `</tool_call>`).
    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.buf);
        self.emit(&rest, &mut out);
        if self.mode == Mode::Tool && !self.tool.trim().is_empty() {
            let body = std::mem::take(&mut self.tool);
            match parse_xml_call(&body, &self.schemas) {
                Some((name, arguments)) => out.push(Piece::ToolCall { name, arguments }),
                None => out.push(Piece::Text(format!("<tool_call>{body}"))),
            }
        }
        out
    }

    /// A tag that is plain text where it appears: tool-call tags inside reasoning, and anything
    /// but a closing `</tool_call>` that completes the call inside a call.
    fn is_text(&self, tag: &str) -> bool {
        match self.mode {
            Mode::Think => tag == "<tool_call>" || tag == "</tool_call>",
            Mode::Tool if tag != "</tool_call>" => true,
            Mode::Tool => {
                let b = self.tool.trim_start();
                b.starts_with("<function=") && parse_xml_call(&self.tool, &self.schemas).is_none()
            }
            Mode::Text => false,
        }
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
                if let Some((name, arguments)) = parse_xml_call(&body, &self.schemas) {
                    out.push(Piece::ToolCall { name, arguments });
                    return;
                }
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

/// A Qwen XML tool call body (between `<tool_call>` and `</tool_call>`):
///
/// ```text
/// <function=NAME>
/// <parameter=KEY>
/// VALUE
/// </parameter>
/// ...
/// </function>
/// ```
///
/// A value ends at the first `</parameter>` followed by another `<parameter=` or by the end of
/// the function, so values may contain `</parameter>` themselves. The newline the template puts
/// on each side of a value is dropped. Values are typed by the tool's schema (`schemas`: name →
/// `parameters`): strings stay strings, numbers, booleans, objects and arrays are parsed from
/// their JSON text (falling back to the string); parameters without a schema are strings.
/// `None` unless the whole body has that shape.
pub fn parse_xml_call(
    body: &str,
    schemas: &std::collections::HashMap<String, Value>,
) -> Option<(String, Value)> {
    const CLOSE: &str = "</parameter>";
    let rest = body.trim().strip_prefix("<function=")?;
    let (name, rest) = rest.split_once('>')?;
    let name = name.trim();
    if name.is_empty() || name.contains(['\n', '<']) {
        return None;
    }
    let mut inner = rest.trim_end().strip_suffix("</function>")?;
    let props = &schemas.get(name).map_or(&Value::Null, |s| &s["properties"]);
    let mut args = serde_json::Map::new();
    loop {
        let s = inner.trim_start();
        if s.is_empty() {
            break;
        }
        let (key, s) = s.strip_prefix("<parameter=")?.split_once('>')?;
        let mut end = None;
        let mut from = 0;
        while let Some(i) = s[from..].find(CLOSE) {
            let at = from + i;
            let after = s[at + CLOSE.len()..].trim_start();
            if after.is_empty() || after.starts_with("<parameter=") {
                end = Some(at);
                break;
            }
            from = at + CLOSE.len();
        }
        let at = end?;
        let raw = &s[..at];
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        args.insert(key.trim().to_string(), typed(raw, &props[key.trim()]));
        inner = &s[at + CLOSE.len()..];
    }
    Some((name.to_string(), Value::Object(args)))
}

/// A parameter value typed by its JSON schema (see [`parse_xml_call`]).
fn typed(raw: &str, schema: &Value) -> Value {
    let types: Vec<&str> = match &schema["type"] {
        Value::String(t) => vec![t.as_str()],
        Value::Array(a) => a
            .iter()
            .filter_map(Value::as_str)
            .filter(|t| *t != "null")
            .collect(),
        _ if schema["anyOf"].is_array() || schema["oneOf"].is_array() => vec!["any"],
        _ => vec![],
    };
    if types.is_empty() || types.contains(&"string") {
        return Value::String(raw.to_string());
    }
    match serde_json::from_str::<Value>(raw.trim()) {
        Ok(v) if !v.is_string() => v,
        _ => match raw.trim() {
            "True" => Value::Bool(true),
            "False" => Value::Bool(false),
            _ => Value::String(raw.to_string()),
        },
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

    fn run_with(tools: serde_json::Value, chunks: &[&str]) -> Vec<Piece> {
        let mut p = Parser::new(false).with_tools(Some(&tools));
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

    fn tools() -> serde_json::Value {
        serde_json::json!([
            {"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {
                "file_path": {"type": "string"}, "limit": {"type": "integer"}, "all": {"type": "boolean"}}}}},
            {"type": "function", "function": {"name": "Edit", "parameters": {"type": "object", "properties": {
                "old_string": {"type": "string"}, "new_string": {"type": "string"},
                "opts": {"type": "object"}, "lines": {"type": ["array", "null"]}}}}},
        ])
    }

    #[test]
    fn qwen_xml_calls_are_typed_by_schema_across_any_chunking() {
        let text = "<think>\nread it, then <tool_call> maybe\n</think>\n\nReading.\n\n<tool_call>\n<function=Read>\n<parameter=file_path>\nsrc/main.rs\n</parameter>\n<parameter=limit>\n40\n</parameter>\n<parameter=all>\ntrue\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=Edit>\n<parameter=old_string>\na </parameter> b\n</tool_call> c\n</parameter>\n<parameter=new_string>\n\n  x\n\n</parameter>\n<parameter=opts>\n{\"n\": 2}\n</parameter>\n<parameter=lines>\n[1, 2]\n</parameter>\n</function>\n</tool_call>";
        let want = vec![
            Piece::Reasoning("\nread it, then <tool_call> maybe\n".into()),
            Piece::Text("Reading.\n\n".into()),
            Piece::ToolCall {
                name: "Read".into(),
                arguments: serde_json::json!({"file_path": "src/main.rs", "limit": 40, "all": true}),
            },
            Piece::Text("\n".into()),
            Piece::ToolCall {
                name: "Edit".into(),
                arguments: serde_json::json!({"old_string": "a </parameter> b\n</tool_call> c",
                    "new_string": "\n  x\n", "opts": {"n": 2}, "lines": [1, 2]}),
            },
        ];
        // Whole, one byte at a time, and in 7-byte chunks (on char boundaries).
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let bytes: Vec<&str> = chars.iter().map(String::as_str).collect();
        let sevens: Vec<String> = chars.chunks(7).map(|c| c.concat()).collect();
        let sevens: Vec<&str> = sevens.iter().map(String::as_str).collect();
        for chunks in [vec![text], bytes, sevens] {
            assert_eq!(run_with(tools(), &chunks), want);
        }
    }

    #[test]
    fn xml_parameters_without_a_schema_are_strings_and_bad_numbers_fall_back() {
        let out = run_with(tools(), &["<tool_call>\n<function=Other>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call><tool_call>\n<function=Read>\n<parameter=limit>\nmany\n</parameter>\n</function>\n</tool_call>"]);
        assert_eq!(
            out,
            vec![
                Piece::ToolCall {
                    name: "Other".into(),
                    arguments: serde_json::json!({"n": "5"})
                },
                Piece::ToolCall {
                    name: "Read".into(),
                    arguments: serde_json::json!({"limit": "many"})
                },
            ]
        );
    }

    #[test]
    fn malformed_xml_falls_back_to_text_and_a_missing_close_is_forgiven() {
        assert_eq!(
            run_with(tools(), &["<tool_call>\n<function=Read>\n<parameter=file_path>\nx\n</function>\n</tool_call>"]),
            vec![Piece::Text("<tool_call>\n<function=Read>\n<parameter=file_path>\nx\n</function>\n</tool_call>".into())]
        );
        assert_eq!(
            run_with(tools(), &["<tool_call>\n<function=Read>\n<parameter=file_path>\nx\n</parameter>\n</function>\n"]),
            vec![Piece::ToolCall { name: "Read".into(), arguments: serde_json::json!({"file_path": "x"}) }]
        );
        // No parameters at all.
        assert_eq!(
            run_with(
                tools(),
                &["<tool_call>\n<function=Read>\n</function>\n</tool_call>"]
            ),
            vec![Piece::ToolCall {
                name: "Read".into(),
                arguments: serde_json::json!({})
            }]
        );
    }
}
