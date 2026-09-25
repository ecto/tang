//! OpenAI-compatible HTTP server: `/v1/chat/completions` (streaming and not), `/v1/models`,
//! and llama.cpp-style `/props` so clients can discover the context window.
//!
//! The model lives on one worker thread (GPU state isn't shareable); requests queue for it.
//!
//! With an API key, every route but `/health` wants `Authorization: Bearer <key>`.

use crate::chat::Piece;
use crate::engine::{Engine, Finish, Request, Usage};
use crate::sample::Sampling;
use axum::extract::{Request as HttpRequest, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::mpsc as std_mpsc;
use std::time::{SystemTime, UNIX_EPOCH};
use tang_compute::ComputeDevice;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

enum Out {
    Piece(Piece),
    Done(Finish, Usage),
    Error(String),
}

struct Job {
    req: Request,
    tx: mpsc::UnboundedSender<Out>,
}

#[derive(Clone)]
struct App {
    jobs: std_mpsc::Sender<Job>,
    model: String,
    ctx: usize,
    /// The model takes images.
    vision: bool,
}

/// Serve `engine` on `addr` until the process exits. `load` runs on the worker thread (so the
/// GPU device is created where it's used). With `key`, requests must present it.
pub fn serve<D, F>(addr: &str, model: String, key: Option<String>, load: F) -> anyhow::Result<()>
where
    D: ComputeDevice + 'static,
    F: FnOnce() -> anyhow::Result<Engine<D>> + Send + 'static,
{
    let (jobs, rx) = std_mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = std_mpsc::channel();
    std::thread::spawn(move || {
        let mut engine = match load() {
            Ok(e) => {
                let _ = ready_tx.send(Ok((e.context_window(), e.model.vision.is_some())));
                e
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
                return;
            }
        };
        for job in rx {
            let tx = job.tx.clone();
            let result = engine.complete(&job.req, |p| tx.send(Out::Piece(p)).is_ok());
            let _ = match result {
                Ok((finish, usage)) => job.tx.send(Out::Done(finish, usage)),
                Err(e) => job.tx.send(Out::Error(format!("{e:#}"))),
            };
        }
    });
    let (ctx, vision) = ready_rx.recv()??;

    let app = App {
        jobs,
        model,
        ctx,
        vision,
    };
    let api = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .route("/props", get(props))
        .with_state(app);
    let api = match key {
        Some(key) => api.layer(middleware::from_fn_with_state(
            std::sync::Arc::new(key),
            require_key,
        )),
        None => api,
    };
    let router = Router::new()
        .route("/health", get(|| async { "ok" }))
        .merge(api);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        eprintln!("tang-llm: listening on http://{}", listener.local_addr()?);
        axum::serve(listener, router).await?;
        anyhow::Ok(())
    })
}

/// Turn away requests without the key (compared in constant time).
async fn require_key(
    State(key): State<std::sync::Arc<String>>,
    req: HttpRequest,
    next: Next,
) -> Response {
    let given = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    if same(given.as_bytes(), key.as_bytes()) {
        return next.run(req).await;
    }
    let body =
        json!({"error": {"message": "missing or wrong API key", "type": "invalid_request_error"}});
    (StatusCode::UNAUTHORIZED, Json(body)).into_response()
}

fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn models(State(app): State<App>) -> Json<Value> {
    let caps: Vec<&str> = if app.vision {
        vec!["completion", "vision"]
    } else {
        vec!["completion"]
    };
    Json(json!({
        "object": "list",
        "data": [{ "id": app.model, "object": "model", "owned_by": "tang", "max_model_len": app.ctx, "capabilities": caps }],
    }))
}

async fn props(State(app): State<App>) -> Json<Value> {
    Json(json!({ "n_ctx": app.ctx, "default_generation_settings": { "n_ctx": app.ctx } }))
}

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    (
        status,
        Json(json!({ "error": { "message": msg.into(), "type": "invalid_request_error" } })),
    )
        .into_response()
}

/// Make content a string, which templates assume: flatten OpenAI content arrays
/// (`[{type: text, text}]`), and turn a null/missing content (tool-call-only assistant turns)
/// into "" (templates do things like `'</think>' in message.content`). Images (`image_url`
/// parts with data URLs) are returned in order, each leaving an image marker in the text.
fn normalize_messages(messages: &Value) -> Result<(Value, Vec<Vec<u8>>), String> {
    use base64::Engine as _;
    let mut out = messages.clone();
    let mut images = Vec::new();
    for m in out.as_array_mut().into_iter().flatten() {
        if let Some(parts) = m["content"].as_array() {
            let mut text: Vec<String> = Vec::new();
            for p in parts {
                if let Some(t) = p["text"].as_str() {
                    text.push(t.to_string());
                } else if let Some(url) = p["image_url"]["url"].as_str().or(p["image_url"].as_str())
                {
                    let data = url
                        .split_once(";base64,")
                        .map(|(_, d)| d)
                        .ok_or("images must be data URLs (data:image/...;base64,...)")?;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(data.trim())
                        .map_err(|e| format!("image data: {e}"))?;
                    images.push(bytes);
                    text.push(crate::engine::IMAGE_MARKER.to_string());
                }
            }
            m["content"] = json!(text.join("\n"));
        } else if !m["content"].is_string() && m.is_object() {
            m["content"] = json!("");
        }
    }
    Ok((out, images))
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    #[test]
    fn content_is_always_a_string() {
        let m = serde_json::json!([
            {"role": "assistant", "content": null, "tool_calls": []},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant"},
        ]);
        let (n, _) = super::normalize_messages(&m).unwrap();
        assert_eq!(n[0]["content"], "");
        assert_eq!(n[1]["content"], "a\nb");
        assert_eq!(n[2]["content"], "");
    }

    use super::{collect_response, stream_response, Out};
    use crate::chat::Piece;
    use crate::engine::{Finish, Usage};
    use axum::response::IntoResponse;
    use serde_json::{json, Value};
    use tokio::sync::mpsc;

    fn two_calls() -> mpsc::UnboundedReceiver<Out> {
        let (tx, rx) = mpsc::unbounded_channel();
        for name in ["a", "b"] {
            let call = Piece::ToolCall {
                name: name.into(),
                arguments: json!({}),
            };
            tx.send(Out::Piece(call)).unwrap();
        }
        let usage = Usage {
            prompt_tokens: 1,
            cached_tokens: 0,
            completion_tokens: 1,
            prefill_tok_s: 0.0,
            decode_tok_s: 0.0,
        };
        tx.send(Out::Done(Finish::ToolCalls, usage)).unwrap();
        rx
    }

    async fn body(r: axum::response::Response) -> String {
        let b = axum::body::to_bytes(r.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(b.to_vec()).unwrap()
    }

    async fn ids_non_streaming() -> Vec<String> {
        let r = collect_response("m".into(), "id".into(), 0, two_calls()).await;
        let v: Value = serde_json::from_str(&body(r).await).unwrap();
        v["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["id"].as_str().unwrap().to_string())
            .collect()
    }

    async fn ids_streaming() -> Vec<String> {
        let r = stream_response("m".into(), "id".into(), 0, false, two_calls()).into_response();
        body(r)
            .await
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter_map(|d| serde_json::from_str::<Value>(d).ok())
            .filter_map(|v| {
                v["choices"][0]["delta"]["tool_calls"][0]["id"]
                    .as_str()
                    .map(String::from)
            })
            .collect()
    }

    #[tokio::test]
    async fn tool_call_ids_are_unique_within_and_across_responses() {
        let mut ids = Vec::new();
        for _ in 0..2 {
            ids.extend(ids_non_streaming().await);
            ids.extend(ids_streaming().await);
        }
        assert_eq!(ids.len(), 8);
        for id in &ids {
            assert!(id.starts_with("call_") && id.len() == 29, "{id}");
        }
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "{ids:?}");
    }
}

fn parse(body: &Value) -> Result<Request, String> {
    let messages = body
        .get("messages")
        .filter(|m| m.is_array())
        .ok_or("messages must be an array")?;
    let d = Sampling::default();
    let f = |k: &str| body[k].as_f64();
    let (messages, images) = normalize_messages(messages)?;
    Ok(Request {
        messages,
        images,
        tools: body.get("tools").cloned().filter(|t| !t.is_null()),
        think: body["chat_template_kwargs"]["enable_thinking"]
            .as_bool()
            .or(body["think"].as_bool()),
        sampling: Sampling {
            temperature: f("temperature").map(|v| v as f32).unwrap_or(d.temperature),
            top_p: f("top_p").map(|v| v as f32).unwrap_or(d.top_p),
            top_k: body["top_k"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(d.top_k),
            seed: body["seed"].as_u64().unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|t| t.as_nanos() as u64)
                    .unwrap_or(1)
            }),
        },
        max_tokens: body["max_completion_tokens"]
            .as_u64()
            .or(body["max_tokens"].as_u64())
            .map(|n| n as usize),
        stop: match &body["stop"] {
            Value::String(s) => vec![s.clone()],
            Value::Array(a) => a
                .iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect(),
            _ => Vec::new(),
        },
    })
}

fn finish_reason(f: Finish) -> &'static str {
    match f {
        Finish::Stop | Finish::Cancelled => "stop",
        Finish::Length => "length",
        Finish::ToolCalls => "tool_calls",
    }
}

fn usage_json(u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "prompt_tokens_details": { "cached_tokens": u.cached_tokens },
        "timings": { "prompt_per_second": u.prefill_tok_s, "predicted_per_second": u.decode_tok_s },
    })
}

async fn chat(State(app): State<App>, Json(body): Json<Value>) -> Response {
    let req = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let stream = body["stream"].as_bool().unwrap_or(false);
    let include_usage = body["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    let (tx, rx) = mpsc::unbounded_channel();
    if app.jobs.send(Job { req, tx }).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "model worker stopped");
    }
    let id = format!(
        "chatcmpl-{:x}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|t| t.as_nanos())
            .unwrap_or(0)
    );
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|t| t.as_secs())
        .unwrap_or(0);
    if stream {
        stream_response(app.model, id, created, include_usage, rx).into_response()
    } else {
        collect_response(app.model, id, created, rx).await
    }
}

fn stream_response(
    model: String,
    id: String,
    created: u64,
    include_usage: bool,
    mut rx: mpsc::UnboundedReceiver<Out>,
) -> Sse<UnboundedReceiverStream<Result<Event, Infallible>>> {
    let chunk = move |delta: Value, finish: Option<&str>| {
        json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        })
    };
    let mut calls = 0usize;
    let mut first = true;
    let (sse_tx, sse_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(out) = rx.recv().await {
            let mut evs: Vec<Value> = Vec::new();
            let role = if first {
                json!("assistant")
            } else {
                Value::Null
            };
            first = false;
            match out {
                Out::Piece(Piece::Text(t)) => {
                    evs.push(chunk(json!({ "role": role, "content": t }), None))
                }
                Out::Piece(Piece::Reasoning(t)) => {
                    evs.push(chunk(json!({ "role": role, "reasoning_content": t }), None))
                }
                Out::Piece(Piece::ToolCall { name, arguments }) => {
                    evs.push(chunk(
                        json!({ "role": role, "tool_calls": [{
                            "index": calls, "id": tool_call_id(), "type": "function",
                            "function": { "name": name, "arguments": arguments.to_string() },
                        }]}),
                        None,
                    ));
                    calls += 1;
                }
                Out::Done(finish, usage) => {
                    let mut last = chunk(json!({}), Some(finish_reason(finish)));
                    if include_usage {
                        evs.push(last);
                        last = json!({ "id": last_id(&evs), "object": "chat.completion.chunk", "choices": [], "usage": usage_json(&usage) });
                    } else {
                        last["usage"] = usage_json(&usage);
                    }
                    evs.push(last);
                }
                Out::Error(e) => evs.push(json!({ "error": { "message": e } })),
            }
            let done = evs.iter().any(|e| {
                e["choices"][0]["finish_reason"].is_string()
                    || e["usage"].is_object()
                    || e["error"].is_object()
            });
            for v in evs {
                if sse_tx
                    .send(Ok(Event::default().data(v.to_string())))
                    .is_err()
                {
                    return; // client went away; dropping rx cancels generation
                }
            }
            if done {
                let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                return;
            }
        }
    });
    Sse::new(UnboundedReceiverStream::new(sse_rx))
}

/// A tool-call id unique within the response, across requests and across restarts: clients
/// (kiln among them) key tool results by id, so `call_0` on every call collides. A per-process
/// random prefix plus a process-wide counter, as `call_<24 hex>`.
fn tool_call_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    static PREFIX: OnceLock<u64> = OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let prefix = *PREFIX.get_or_init(|| {
        // RandomState is seeded from the OS RNG; mix in time and pid for good measure.
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|t| t.as_nanos())
                .unwrap_or(0),
        );
        h.write_u32(std::process::id());
        h.finish()
    });
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("call_{prefix:016x}{n:08x}")
}

fn last_id(evs: &[Value]) -> Value {
    evs.last().map(|e| e["id"].clone()).unwrap_or(Value::Null)
}

async fn collect_response(
    model: String,
    id: String,
    created: u64,
    mut rx: mpsc::UnboundedReceiver<Out>,
) -> Response {
    let (mut text, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
    while let Some(out) = rx.recv().await {
        match out {
            Out::Piece(Piece::Text(t)) => text.push_str(&t),
            Out::Piece(Piece::Reasoning(t)) => reasoning.push_str(&t),
            Out::Piece(Piece::ToolCall { name, arguments }) => calls.push(json!({
                "id": tool_call_id(), "type": "function",
                "function": { "name": name, "arguments": arguments.to_string() },
            })),
            Out::Error(e) => return error(StatusCode::BAD_REQUEST, e),
            Out::Done(finish, usage) => {
                let mut message = json!({ "role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) } });
                if !reasoning.is_empty() {
                    message["reasoning_content"] = json!(reasoning);
                }
                if !calls.is_empty() {
                    message["tool_calls"] = json!(calls);
                }
                return Json(json!({
                    "id": id, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{ "index": 0, "message": message, "finish_reason": finish_reason(finish) }],
                    "usage": usage_json(&usage),
                }))
                .into_response();
            }
        }
    }
    error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "generation ended unexpectedly",
    )
}
