//! OpenAI-compatible HTTP server: `/v1/chat/completions` (streaming and not), `/v1/models`,
//! and llama.cpp-style `/props` so clients can discover the context window. `/v1/prefill`
//! takes a chat completion body and only prefills it, so the request that follows (the same
//! conversation plus a new message) starts from a warm cache.
//!
//! For frog's scheduler (see `docs/node.md`): `GET /node` describes the machine, its model,
//! measured rates, queue and KV blocks; `POST /models/load` and `/models/unload` swap the model
//! (one per process), loading only into free memory. Requests carry `x-frog-priority:
//! interactive | background`: interactive ones run first, and a background prefill gives way
//! between chunks when an interactive request is waiting (see [`crate::queue`]).
//!
//! The model lives on one worker thread (GPU state isn't shareable); requests queue for it.
//!
//! With an API key, every route but `/health` wants `Authorization: Bearer <key>`.

use crate::chat::Piece;
use crate::engine::{Control, Engine, Finish, Outcome, Progress, Prompter, Request, Usage};
use crate::model::Dtype;
use crate::node::{Need, Probe, Rates};
use crate::queue::{Priority, Queue, Ticket};
use crate::sample::Sampling;
use axum::extract::{Request as HttpRequest, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tang_compute::ComputeDevice;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

enum Out {
    Piece(Piece),
    Done(Finish, Usage),
    Error(String),
}

/// A load's or unload's answer: a JSON body, or a status and message.
type Reply = Result<Value, (StatusCode, String)>;

enum Job {
    Complete {
        req: Box<Request>,
        /// Its prompt and token ids, made by `model`'s prompter when it was queued.
        prompt: Option<(String, Vec<u32>)>,
        tx: mpsc::UnboundedSender<Out>,
        /// The model it was accepted for: it fails rather than run on another.
        model: String,
        /// What earlier runs did before giving way.
        before: Option<Before>,
    },
    Load {
        spec: String,
        reply: oneshot::Sender<Reply>,
    },
    Unload {
        reply: oneshot::Sender<Reply>,
    },
}

/// A request's first run, when it gave way part way through its prefill.
#[derive(Debug, Clone, Copy)]
struct Before {
    cached_tokens: usize,
    prefilled: usize,
    secs: f64,
}

/// What the server reports and is shared between the worker and the handlers.
struct Node {
    id: String,
    queue: Queue<Job>,
    state: Mutex<NodeState>,
    hardware: Probe,
    dtype: Dtype,
    /// Serialize model loads and heavy inference across the judge and image workers.
    resources: Arc<Mutex<()>>,
    image: Option<crate::image_server::ImageModel>,
}

#[derive(Default)]
struct NodeState {
    model: Option<Loaded>,
    /// A model being loaded, and since when.
    loading: Option<(String, Instant)>,
    running: Option<Running>,
}

struct Loaded {
    name: String,
    path: PathBuf,
    ctx: usize,
    /// The model takes images.
    vision: bool,
    /// Unix seconds.
    loaded_at: u64,
    load_secs: f64,
    /// Its weights' estimated device memory.
    need: Option<Need>,
    rates: Rates,
    prompter: Prompter,
    kv: Kv,
}

/// The loaded model's KV, as of the last request.
#[derive(Default)]
struct Kv {
    /// Sealed blocks in memory.
    blocks: Vec<u64>,
    /// Each `prompt_cache_key` with a cache, and the positions it holds.
    keys: Vec<(String, usize)>,
    /// The disk tier's directory (its `blocks/` are listed per `/node`).
    store: Option<PathBuf>,
    /// Device memory the block pool holds.
    pool_bytes: usize,
}

struct Running {
    ticket: Ticket,
    progress: Progress,
    started: Instant,
}

impl Node {
    fn state(&self) -> MutexGuard<'_, NodeState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Requests running or waiting.
    fn busy(&self) -> usize {
        self.queue.len() + self.state().running.is_some() as usize
    }
}

#[derive(Clone)]
struct App {
    node: Arc<Node>,
}

/// How to serve.
pub struct Options {
    pub addr: String,
    /// With a key, requests must present it.
    pub key: Option<String>,
    /// The model to load at start (a directory or Hugging Face repo id); `None` starts with
    /// none (load one with `POST /models/load`).
    pub model: Option<String>,
    /// How the loader keeps weights (for the fit check before a load).
    pub dtype: Dtype,
    /// Reads the hardware and its free memory.
    pub hardware: Probe,
}

/// Serve until the process exits. `load` makes an engine for a model (a directory or Hugging
/// Face repo id); it runs on the worker thread, so the GPU device is created where it's used.
pub fn serve<D, F>(opts: Options, load: F) -> anyhow::Result<()>
where
    D: ComputeDevice + 'static,
    F: Fn(&str) -> anyhow::Result<Engine<D>> + Send + 'static,
{
    serve_with_images(opts, load, None)
}

/// Serve chat/vision and a lazy resident image pipeline in one process. Heavy inference
/// is serialized so both workers share scratch headroom; neither evicts the other's weights.
pub fn serve_with_images<D, F>(
    opts: Options,
    load: F,
    image: Option<(PathBuf, fn() -> anyhow::Result<D>)>,
) -> anyhow::Result<()>
where
    D: ComputeDevice + 'static,
    F: Fn(&str) -> anyhow::Result<Engine<D>> + Send + 'static,
{
    let resources = Arc::new(Mutex::new(()));
    let image = image.map(|(root, make)| crate::image_server::mount(root, make, resources.clone()));
    let node = Arc::new(Node {
        id: crate::node::node_id(),
        queue: Queue::new(),
        state: Mutex::new(NodeState::default()),
        hardware: opts.hardware,
        dtype: opts.dtype,
        resources,
        image: image.as_ref().map(|service| service.model.clone()),
    });
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let worker = node.clone();
    let first = opts.model.clone();
    std::thread::spawn(move || {
        let mut w = Worker {
            node: worker,
            engine: None,
            load,
        };
        if let Some(spec) = first {
            let gate = w.node.resources.clone();
            let _guard = gate.lock().unwrap_or_else(|e| e.into_inner());
            if let Err(e) = w.load(&spec) {
                let _ = ready_tx.send(Err(e));
                return;
            }
        }
        let _ = ready_tx.send(Ok(()));
        w.run();
    });
    ready_rx.recv()??;

    let app = App { node };
    let mut api = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/prefill", post(prefill))
        .route("/v1/models", get(models))
        .route("/props", get(props))
        .route("/node", get(node_info))
        .route("/models/load", post(load_model))
        .route("/models/unload", post(unload_model))
        .with_state(app);
    if let Some(image) = image {
        api = api.merge(image.router);
    }
    let api = match opts.key {
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
        let listener = tokio::net::TcpListener::bind(&opts.addr).await?;
        eprintln!("tang-llm: listening on http://{}", listener.local_addr()?);
        axum::serve(listener, router).await?;
        anyhow::Ok(())
    })
}

/// The model's thread: runs jobs from the queue, interactive first.
struct Worker<D: ComputeDevice, F> {
    node: Arc<Node>,
    engine: Option<(String, Engine<D>)>,
    load: F,
}

/// Lets a background request's prefill give way to waiting interactive work, and keeps
/// `/node`'s view of the running request current.
struct Ctl<'a> {
    node: &'a Node,
    background: bool,
}

impl Control for Ctl<'_> {
    fn keep_prefilling(&mut self) -> bool {
        !(self.background && self.node.queue.interactive_waiting())
    }

    fn progress(&mut self, p: Progress) {
        if let Some(r) = self.node.state().running.as_mut() {
            r.progress = p;
        }
    }
}

impl<D, F> Worker<D, F>
where
    D: ComputeDevice,
    F: Fn(&str) -> anyhow::Result<Engine<D>>,
{
    fn run(&mut self) {
        loop {
            let (ticket, job) = self.node.queue.pop();
            let gate = self.node.resources.clone();
            let _guard = gate.lock().unwrap_or_else(|e| e.into_inner());
            match job {
                Job::Complete {
                    req,
                    prompt,
                    tx,
                    model,
                    before,
                } => self.complete(ticket, req, prompt, tx, model, before),
                Job::Load { spec, reply } => {
                    let _ = reply.send(self.swap(&spec));
                }
                Job::Unload { reply } => {
                    let _ = reply.send(Ok(self.unload()));
                }
            }
        }
    }

    fn complete(
        &mut self,
        mut ticket: Ticket,
        req: Box<Request>,
        prompt: Option<(String, Vec<u32>)>,
        tx: mpsc::UnboundedSender<Out>,
        model: String,
        before: Option<Before>,
    ) {
        if tx.is_closed() {
            return; // the client went away while it waited
        }
        let engine = match &mut self.engine {
            Some((name, e)) if *name == model => e,
            Some((name, _)) => {
                let _ = tx.send(Out::Error(format!(
                    "model {model} was unloaded ({name} is loaded now)"
                )));
                return;
            }
            None => {
                let _ = tx.send(Out::Error(format!("model {model} was unloaded")));
                return;
            }
        };
        self.node.state().running = Some(Running {
            ticket: ticket.clone(),
            progress: Progress {
                prompt_tokens: ticket.prompt_tokens.unwrap_or(0),
                ..Progress::default()
            },
            started: Instant::now(),
        });
        let mut ctl = Ctl {
            node: &self.node,
            background: ticket.priority == Priority::Background,
        };
        let piece_tx = tx.clone();
        // A background request may give way; it keeps its prompt for when it goes on.
        let kept = match ticket.priority {
            Priority::Background => prompt.clone(),
            Priority::Interactive => None,
        };
        let result = engine.complete_with(
            &req,
            prompt,
            |p| piece_tx.send(Out::Piece(p)).is_ok(),
            &mut ctl,
        );
        self.node.state().running = None;
        // Before the reply goes out, so a client that asks `/node` next sees what it left.
        self.refresh_kv();
        match result {
            Ok(Outcome::Yielded {
                prompt_tokens,
                cached_tokens,
                prefilled,
                secs,
            }) => {
                self.with_loaded(|l| l.rates.prefilled(prefilled, secs));
                let before = match before {
                    Some(b) => Before {
                        prefilled: b.prefilled + prefilled,
                        secs: b.secs + secs,
                        ..b
                    },
                    None => Before {
                        cached_tokens,
                        prefilled,
                        secs,
                    },
                };
                ticket.yields += 1;
                ticket.prompt_tokens = Some(prompt_tokens);
                let job = Job::Complete {
                    req,
                    prompt: kept,
                    tx,
                    model,
                    before: Some(before),
                };
                self.node.queue.push_front(ticket, job);
            }
            Ok(Outcome::Done(finish, mut usage)) => {
                let fresh = usage.prompt_tokens - usage.cached_tokens;
                let secs = if usage.prefill_tok_s > 0.0 {
                    fresh as f64 / usage.prefill_tok_s
                } else {
                    0.0
                };
                self.with_loaded(|l| {
                    l.rates.prefilled(fresh, secs);
                    if !req.prefill_only {
                        l.rates.decoded(usage.completion_tokens, usage.decode_tok_s);
                    }
                });
                // As the client sees it: one request, its prefill spread over its runs.
                if let Some(b) = before {
                    usage.cached_tokens = b.cached_tokens;
                    usage.prefill_tok_s =
                        (usage.prompt_tokens - b.cached_tokens) as f64 / (b.secs + secs).max(1e-9);
                }
                let engine = &mut self.engine.as_mut().expect("the engine that ran it").1;
                if engine.speculation().is_some() {
                    eprintln!(
                        "tang-llm: {} tokens at {:.1} tok/s, drafts {}/{} accepted",
                        usage.completion_tokens,
                        usage.decode_tok_s,
                        usage.accepted_tokens,
                        usage.draft_tokens
                    );
                }
                let _ = tx.send(Out::Done(finish, usage));
                // After the reply, so saving never delays it.
                engine.save(req.cache_key.as_deref());
            }
            Err(e) => {
                let _ = tx.send(Out::Error(format!("{e:#}")));
            }
        }
    }

    fn with_loaded(&self, f: impl FnOnce(&mut Loaded)) {
        if let Some(l) = self.node.state().model.as_mut() {
            f(l);
        }
    }

    /// Copy what the engine's KV holds into `/node`'s view.
    fn refresh_kv(&self) {
        let Some((_, e)) = &self.engine else { return };
        let kv = Kv {
            blocks: e.block_hashes(),
            keys: e.cached_keys(),
            store: e.kv_store_dir().map(PathBuf::from),
            pool_bytes: e.kv_bytes(),
        };
        self.with_loaded(|l| l.kv = kv);
    }

    /// Load `spec` (nothing else is loaded).
    fn load(&mut self, spec: &str) -> anyhow::Result<()> {
        let path = crate::resolve_model(spec)?;
        let need = crate::node::need(&path, self.node.dtype).ok();
        self.node.state().loading = Some((spec.to_string(), Instant::now()));
        let t = Instant::now();
        let loaded = (self.load)(spec);
        self.node.state().loading = None;
        let engine = loaded?;
        let load_secs = t.elapsed().as_secs_f64();
        self.node.state().model = Some(Loaded {
            name: spec.to_string(),
            path,
            ctx: engine.context_window(),
            vision: engine.model.vision.is_some(),
            loaded_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|t| t.as_secs())
                .unwrap_or(0),
            load_secs,
            need,
            rates: Rates::default(),
            prompter: engine.prompter(),
            kv: Kv::default(),
        });
        self.engine = Some((spec.to_string(), engine));
        self.refresh_kv();
        Ok(())
    }

    /// Drop the model, if one is loaded; says what was freed.
    fn unload(&mut self) -> Value {
        let Some((name, e)) = self.engine.take() else {
            return json!({ "unloaded": null });
        };
        let freed = self.footprint(&e);
        e.flush_kv_store();
        drop(e);
        self.node.state().model = None;
        eprintln!("tang-llm: unloaded {name}");
        json!({ "unloaded": name, "freed_bytes": freed })
    }

    /// Device memory the loaded engine holds: its weights (estimated) and KV pool.
    fn footprint(&self, e: &Engine<D>) -> u64 {
        let weights = self
            .node
            .state()
            .model
            .as_ref()
            .and_then(|l| l.need)
            .map_or(0, |n| n.weights_bytes);
        weights + e.kv_bytes() as u64
    }

    /// Load `spec` in place of the current model, if it fits in free memory plus what the
    /// current one would give back. Never takes memory anything else holds.
    fn swap(&mut self, spec: &str) -> Reply {
        if self.engine.as_ref().is_some_and(|(n, _)| n == spec) {
            return Ok(json!({ "loaded": spec, "already": true }));
        }
        let path =
            crate::resolve_model(spec).map_err(|e| (StatusCode::NOT_FOUND, format!("{e:#}")))?;
        let need = crate::node::need(&path, self.node.dtype)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("{spec}: {e:#}")))?;
        let hw = (self.node.hardware)();
        let ours = self.engine.as_ref().map_or(0, |(_, e)| self.footprint(e));
        let room = hw.free_bytes + ours;
        if need.total() > room {
            let gb = |b: u64| b as f64 / 1e9;
            return Err((
                StatusCode::INSUFFICIENT_STORAGE,
                format!(
                    "{spec} needs {:.1} GB ({:.1} GB of weights, {:.1} GB headroom) but only \
                     {:.1} GB is free ({:.1} GB free on the {}{}); loads never take memory \
                     other processes hold",
                    gb(need.total()),
                    gb(need.weights_bytes),
                    gb(need.headroom_bytes),
                    gb(room),
                    gb(hw.free_bytes),
                    hw.kind,
                    if ours > 0 {
                        format!(", {:.1} GB from unloading the current model", gb(ours))
                    } else {
                        String::new()
                    },
                ),
            ));
        }
        let previous = self.engine.as_ref().map(|(n, _)| n.clone());
        self.unload();
        match self.load(spec) {
            Ok(()) => {
                let load_secs = self
                    .node
                    .state()
                    .model
                    .as_ref()
                    .map_or(0.0, |l| l.load_secs);
                eprintln!("tang-llm: loaded {spec} in {load_secs:.1}s");
                Ok(json!({
                    "loaded": spec,
                    "replaced": previous,
                    "load_secs": load_secs,
                    "weights_bytes": need.weights_bytes,
                }))
            }
            Err(e) => {
                let mut msg = format!("loading {spec}: {e:#}");
                if let Some(p) = previous {
                    match self.load(&p) {
                        Ok(()) => msg.push_str(&format!(" ({p} is loaded again)")),
                        Err(e2) => msg.push_str(&format!(" (reloading {p} failed too: {e2:#})")),
                    }
                }
                Err((StatusCode::INTERNAL_SERVER_ERROR, msg))
            }
        }
    }
}

/// Turn away requests without the key (compared in constant time).
pub(crate) async fn require_key(
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
    let st = app.node.state();
    let mut data: Vec<Value> = st
        .model
        .iter()
        .map(|m| {
            // `thinking_budget`: requests may cap reasoning tokens (see `engine::ThinkBudget`).
            // `prompt_cache_key`: requests may name their conversation's KV slot (see
            // `crate::slots`). `prefill`: `/v1/prefill` warms a conversation's cache.
            // `priority`: `x-frog-priority` orders requests.
            let mut caps = vec![
                "completion",
                "json_object",
                "json_schema",
                "thinking_budget",
                "prompt_cache_key",
                "prefill",
                "priority",
            ];
            if m.vision {
                caps.push("vision");
            }
            json!({ "id": m.name, "object": "model", "owned_by": "tang", "max_model_len": m.ctx, "capabilities": caps })
        })
        .collect();
    if let Some(image) = &app.node.image {
        data.push(image.json());
    }
    Json(json!({ "object": "list", "data": data }))
}

async fn props(State(app): State<App>) -> Response {
    let ctx = app.node.state().model.as_ref().map(|m| m.ctx);
    match ctx {
        Some(ctx) => Json(json!({ "n_ctx": ctx, "default_generation_settings": { "n_ctx": ctx } }))
            .into_response(),
        None => error(StatusCode::SERVICE_UNAVAILABLE, "no model loaded"),
    }
}

/// `GET /node`: see `docs/node.md` for the shape.
async fn node_info(State(app): State<App>) -> Json<Value> {
    let node = app.node.clone();
    let v = tokio::task::spawn_blocking(move || describe(&node))
        .await
        .unwrap_or_else(|e| json!({ "error": e.to_string() }));
    Json(v)
}

fn describe(node: &Node) -> Value {
    let hw = (node.hardware)();
    let hex = |h: &u64| format!("{h:016x}");
    let ms = |t: Instant| t.elapsed().as_millis() as u64;
    let st = node.state();
    let loaded: Vec<Value> = st
        .model
        .iter()
        .map(|m| {
            let mut blocks = m.kv.blocks.clone();
            blocks.sort_unstable();
            let mut disk = m
                .kv
                .store
                .as_deref()
                .map(crate::kvstore::Store::block_hashes)
                .unwrap_or_default();
            disk.sort_unstable();
            json!({
                "id": m.name,
                "path": m.path,
                "context_window": m.ctx,
                "vision": m.vision,
                "loaded_at": m.loaded_at,
                "load_secs": m.load_secs,
                "memory": {
                    "weights_bytes": m.need.map(|n| n.weights_bytes),
                    "kv_bytes": m.kv.pool_bytes,
                },
                "rates": m.rates.json(),
                "kv": {
                    "block_positions": crate::blocks::BLOCK,
                    "memory_blocks": blocks.iter().map(hex).collect::<Vec<_>>(),
                    "disk_blocks": disk.iter().map(hex).collect::<Vec<_>>(),
                    "prompt_cache_keys": m.kv.keys.iter().map(|(k, n)| json!({ "key": k, "cached_tokens": n })).collect::<Vec<_>>(),
                },
            })
        })
        .collect();
    let loaded_ids: Vec<String> = st.model.iter().map(|m| m.name.clone()).collect();
    let loading = st
        .loading
        .as_ref()
        .map(|(m, t)| json!({ "id": m, "elapsed_ms": ms(*t) }));
    let running: Vec<Value> = st
        .running
        .iter()
        .map(|r| {
            json!({
                "id": r.ticket.id,
                "priority": r.ticket.priority.as_str(),
                "prompt_tokens": r.progress.prompt_tokens,
                "cached_tokens": r.progress.cached_tokens,
                "prefilled_tokens": r.progress.prefilled,
                "generated_tokens": r.progress.generated,
                "yields": r.ticket.yields,
                "queued_ms": ms(r.ticket.since),
                "running_ms": ms(r.started),
            })
        })
        .collect();
    drop(st);
    let waiting: Vec<Value> = node
        .queue
        .waiting()
        .iter()
        .map(|t| {
            json!({
                "id": t.id,
                "priority": t.priority.as_str(),
                "prompt_tokens": t.prompt_tokens,
                "yields": t.yields,
                "queued_ms": ms(t.since),
            })
        })
        .collect();
    let on_disk: Vec<Value> = crate::node::models_on_disk()
        .into_iter()
        .map(|m| {
            let need = crate::node::need(&m.path, node.dtype).ok();
            json!({
                "id": m.id,
                "path": m.path,
                "size_bytes": m.size_bytes,
                "load_bytes": need.map(|n| n.total()),
                "loaded": loaded_ids.contains(&m.id),
            })
        })
        .collect();
    json!({
        "schema": crate::node::SCHEMA,
        "node_id": node.id,
        "version": env!("CARGO_PKG_VERSION"),
        "hardware": hw.json(),
        "models": { "loaded": loaded, "loading": loading, "on_disk": on_disk },
        "queue": { "running": running, "waiting": waiting },
        "image_model": node.image.as_ref().map(|image|json!({
            "id":"z-image-turbo", "path":image.path, "resident":image.resident.load(std::sync::atomic::Ordering::Relaxed),
        })),
    })
}

/// `POST /models/load {"model": "<repo id or directory>"}`: swap the model in.
async fn load_model(State(app): State<App>, Json(body): Json<Value>) -> Response {
    let Some(spec) = body["model"].as_str().map(String::from) else {
        return error(StatusCode::BAD_REQUEST, "model must be a string");
    };
    if app
        .node
        .state()
        .model
        .as_ref()
        .is_some_and(|m| m.name == spec)
    {
        return Json(json!({ "loaded": spec, "already": true })).into_response();
    }
    control(&app, |reply| Job::Load { spec, reply }).await
}

/// `POST /models/unload`: free the model's memory.
async fn unload_model(State(app): State<App>) -> Response {
    control(&app, |reply| Job::Unload { reply }).await
}

/// Run a load or unload on the worker, when no request is running or waiting.
async fn control(app: &App, job: impl FnOnce(oneshot::Sender<Reply>) -> Job) -> Response {
    let busy = app.node.busy();
    if busy > 0 {
        return error(
            StatusCode::CONFLICT,
            format!("{busy} requests running or waiting; retry when the node is idle"),
        );
    }
    let (tx, rx) = oneshot::channel();
    app.node.queue.push(Priority::Interactive, None, job(tx));
    match rx.await {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err((status, msg))) => error(status, msg),
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "model worker stopped"),
    }
}

/// Queue a request for the loaded model, at its `x-frog-priority`, with its prompt counted.
#[allow(clippy::result_large_err)]
async fn submit(
    app: &App,
    headers: &HeaderMap,
    req: Request,
) -> Result<(String, mpsc::UnboundedReceiver<Out>), Response> {
    let priority = Priority::parse(headers.get("x-frog-priority").and_then(|v| v.to_str().ok()))
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let (model, prompter) = {
        let st = app.node.state();
        let m = st
            .model
            .as_ref()
            .ok_or_else(|| error(StatusCode::SERVICE_UNAVAILABLE, "no model loaded"))?;
        (m.name.clone(), m.prompter.clone())
    };
    // Off the model's thread: the queue knows the prompt's length, and the worker needn't
    // tokenize it again.
    let (req, prompt) = tokio::task::spawn_blocking(move || {
        let p = prompter.prompt(&req);
        (req, p)
    })
    .await
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let prompt = prompt.map_err(|e| error(StatusCode::BAD_REQUEST, format!("{e:#}")))?;
    let (tx, rx) = mpsc::unbounded_channel();
    let n = prompt.1.len();
    let job = Job::Complete {
        req: Box::new(req),
        prompt: Some(prompt),
        tx,
        model: model.clone(),
        before: None,
    };
    app.node.queue.push(priority, Some(n), job);
    Ok((model, rx))
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

    #[test]
    fn thinking_budget_from_the_field_the_kwargs_or_the_effort() {
        use super::thinking_budget as tb;
        assert_eq!(tb(&serde_json::json!({})), None);
        assert_eq!(
            tb(&serde_json::json!({ "thinking_budget": 512 })),
            Some(512)
        );
        assert_eq!(tb(&serde_json::json!({ "thinking_budget": -1 })), None);
        let kw = serde_json::json!({ "chat_template_kwargs": { "thinking_budget": 256 } });
        assert_eq!(tb(&kw), Some(256));
        assert_eq!(
            tb(&serde_json::json!({ "reasoning_effort": "low" })),
            Some(1024)
        );
        assert_eq!(tb(&serde_json::json!({ "reasoning_effort": "high" })), None);
        let both = serde_json::json!({ "thinking_budget": 100, "reasoning_effort": "low" });
        assert_eq!(tb(&both), Some(100));
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
            reasoning_tokens: 0,
            prefill_tok_s: 0.0,
            decode_tok_s: 0.0,
            draft_tokens: 0,
            accepted_tokens: 0,
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

/// A chat completion body as an engine request.
pub fn parse(body: &Value) -> Result<Request, String> {
    let messages = body
        .get("messages")
        .filter(|m| m.is_array())
        .ok_or("messages must be an array")?;
    let response_schema = crate::structured::response_schema(body).map_err(|e| e.to_string())?;
    let d = Sampling::default();
    let f = |k: &str| body[k].as_f64();
    let (messages, images) = normalize_messages(messages)?;
    Ok(Request {
        messages,
        images,
        tools: body.get("tools").cloned().filter(|t| !t.is_null()),
        response_schema,
        think: body["chat_template_kwargs"]["enable_thinking"]
            .as_bool()
            .or(body["think"].as_bool())
            .or((body["reasoning_effort"] == "none").then_some(false)),
        thinking_budget: thinking_budget(body),
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
        cache_key: body["prompt_cache_key"].as_str().map(String::from),
        prefill_only: false,
    })
}

/// Reasoning token caps for OpenAI's `reasoning_effort` (`high` is unlimited).
const EFFORT: [(&str, usize); 3] = [("minimal", 0), ("low", 1024), ("medium", 4096)];

/// The cap on reasoning tokens: `thinking_budget` (top level, or in `chat_template_kwargs` as
/// some servers take it), else from `reasoning_effort`. Negative or absent is unlimited.
fn thinking_budget(body: &Value) -> Option<usize> {
    let n = |v: &Value| v.as_u64().map(|n| n as usize);
    n(&body["thinking_budget"])
        .or_else(|| n(&body["chat_template_kwargs"]["thinking_budget"]))
        .or_else(|| {
            let effort = body["reasoning_effort"].as_str()?;
            EFFORT.iter().find(|(e, _)| *e == effort).map(|(_, n)| *n)
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
        "completion_tokens_details": { "reasoning_tokens": u.reasoning_tokens },
        // llama.cpp's names for speculative decoding stats.
        "timings": {
            "prompt_per_second": u.prefill_tok_s,
            "predicted_per_second": u.decode_tok_s,
            "draft_n": u.draft_tokens,
            "draft_n_accepted": u.accepted_tokens,
        },
    })
}

/// Prefill a chat completion body without generating: `{usage}` once the cache holds it.
async fn prefill(State(app): State<App>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    let mut req = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    req.prefill_only = true;
    let mut rx = match submit(&app, &headers, req).await {
        Ok((_, rx)) => rx,
        Err(r) => return r,
    };
    while let Some(out) = rx.recv().await {
        match out {
            Out::Piece(_) => {}
            Out::Done(_, u) => {
                return Json(json!({ "object": "prefill", "usage": usage_json(&u) }))
                    .into_response()
            }
            Out::Error(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }
    error(StatusCode::SERVICE_UNAVAILABLE, "model worker stopped")
}

async fn chat(State(app): State<App>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    let req = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let stream = body["stream"].as_bool().unwrap_or(false);
    let include_usage = body["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    let (model, rx) = match submit(&app, &headers, req).await {
        Ok(x) => x,
        Err(r) => return r,
    };
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
        stream_response(model, id, created, include_usage, rx).into_response()
    } else {
        collect_response(model, id, created, rx).await
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
