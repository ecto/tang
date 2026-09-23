//! OpenAI-compatible HTTP server: `/v1/chat/completions` (streaming and not), `/v1/models`,
//! and llama.cpp-style `/props` so clients can discover the context window.
//!
//! The model lives on one worker thread (GPU state isn't shareable); requests queue for it.

use crate::chat::Piece;
use crate::engine::{Engine, Finish, Request, Usage};
use crate::sample::Sampling;
use axum::extract::State;
use axum::http::StatusCode;
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
}

/// Serve `engine` on `addr` until the process exits. `load` runs on the worker thread (so the
/// GPU device is created where it's used).
pub fn serve<D, F>(addr: &str, model: String, load: F) -> anyhow::Result<()>
where
    D: ComputeDevice + 'static,
    F: FnOnce() -> anyhow::Result<Engine<D>> + Send + 'static,
{
    let (jobs, rx) = std_mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = std_mpsc::channel();
    std::thread::spawn(move || {
        let mut engine = match load() {
            Ok(e) => {
                let _ = ready_tx.send(Ok(e.context_window()));
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
    let ctx = ready_rx.recv()??;

    let app = App { jobs, model, ctx };
    let router = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .route("/props", get(props))
        .route("/health", get(|| async { "ok" }))
        .with_state(app);
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

async fn models(State(app): State<App>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": app.model, "object": "model", "owned_by": "tang", "max_model_len": app.ctx }],
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

/// Flatten OpenAI content arrays (`[{type: text, text}]`) to strings, which templates expect.
fn normalize_messages(messages: &Value) -> Value {
    let mut out = messages.clone();
    for m in out.as_array_mut().into_iter().flatten() {
        if let Some(parts) = m["content"].as_array() {
            let text: Vec<&str> = parts.iter().filter_map(|p| p["text"].as_str()).collect();
            m["content"] = json!(text.join("\n"));
        }
    }
    out
}

fn parse(body: &Value) -> Result<Request, String> {
    let messages = body
        .get("messages")
        .filter(|m| m.is_array())
        .ok_or("messages must be an array")?;
    let d = Sampling::default();
    let f = |k: &str| body[k].as_f64();
    Ok(Request {
        messages: normalize_messages(messages),
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
                            "index": calls, "id": format!("call_{calls}"), "type": "function",
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
                "id": format!("call_{}", calls.len()), "type": "function",
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
