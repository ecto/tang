//! OpenAI-compatible lazy image worker. One resident pipeline, serialized jobs;
//! disconnecting a streaming client cancels at the next step boundary.
use crate::image_pipeline::{Generated, Pipeline, Request};
use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    middleware,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc as std_mpsc, Arc, Mutex, MutexGuard,
    },
};
use tang_compute::ComputeDevice;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
enum Out {
    Progress(crate::image_pipeline::Progress),
    Images(Vec<Generated>),
    Error(String),
}
struct Job {
    request: Request,
    tx: mpsc::Sender<Out>,
}
#[derive(Clone)]
struct App {
    jobs: std_mpsc::SyncSender<Job>,
    resident: Arc<AtomicBool>,
    previews: bool,
}

#[derive(Clone)]
pub(crate) struct ImageModel {
    pub path: PathBuf,
    pub resident: Arc<AtomicBool>,
    previews: bool,
}
fn capabilities(previews: bool) -> Vec<&'static str> {
    let mut caps = vec!["image_generation", "seed", "steps", "stream_progress"];
    if previews {
        caps.push("latent_previews");
    }
    caps
}
impl ImageModel {
    pub fn json(&self) -> Value {
        json!({"id":MODEL_ID,"object":"model","type":"image",
            "capabilities":capabilities(self.previews),
            "resident":self.resident.load(Ordering::Relaxed)})
    }
}
pub(crate) struct Service {
    pub router: Router,
    pub model: ImageModel,
    app: App,
}
fn images_value(images: Vec<Generated>) -> Value {
    let model_hash = images.first().map(|image| image.model_hash.clone());
    json!({"model":MODEL_ID,"model_hash":model_hash,"data":images.into_iter().map(|image|json!({"b64_json":STANDARD.encode(image.png),"seed":image.seed,"steps":image.steps,"width":image.width,"height":image.height})).collect::<Vec<_>>()})
}
async fn models(State(app): State<App>) -> Json<Value> {
    Json(
        json!({"data":[{"id":MODEL_ID,"object":"model","type":"image","capabilities":capabilities(app.previews),"resident":app.resident.load(Ordering::Relaxed)}]}),
    )
}
async fn generate(
    State(app): State<App>,
    headers: axum::http::HeaderMap,
    Json(request): Json<Request>,
) -> Response {
    if let Err(e) = request.dimensions() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":{"message":e.to_string()}})),
        )
            .into_response();
    }
    let stream = request.stream || headers.get("x-frog-progress").is_some_and(|v| v == "sse");
    if request.preview && !stream {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":{"message":"previews require SSE streaming"}})),
        )
            .into_response();
    }
    if request.preview && !app.previews {
        return (StatusCode::BAD_REQUEST, Json(json!({"error":{"message":"latent previews unavailable; calibrate the installed VAE explicitly"}}))).into_response();
    }
    let (tx, mut rx) = mpsc::channel(PROGRESS_BUFFER);
    if let Err(error) = app.jobs.try_send(Job { request, tx }) {
        let (status, message) = match error {
            std_mpsc::TrySendError::Full(_) => (StatusCode::TOO_MANY_REQUESTS, "image queue full"),
            std_mpsc::TrySendError::Disconnected(_) => {
                (StatusCode::SERVICE_UNAVAILABLE, "image worker unavailable")
            }
        };
        return (status, Json(json!({"error":{"message":message}}))).into_response();
    }
    if stream {
        let events = ReceiverStream::new(rx);
        use tokio_stream::StreamExt;
        return Sse::new(events.map(|out| {
            Ok::<_, Infallible>(match out {
                Out::Progress(progress) => {
                    let mut value = json!({"step":progress.step,"of":progress.of,"image_index":progress.image_index});
                    if let Some(png) = progress.preview_png {
                        value["preview_b64"] = json!(STANDARD.encode(png));
                        value["preview_kind"] = json!("approximate_latent");
                    }
                    Event::default().event("progress").data(value.to_string())
                },
                Out::Images(images) => Event::default()
                    .event("result")
                    .data(images_value(images).to_string()),
                Out::Error(message) => Event::default()
                    .event("error")
                    .data(json!({"error":{"message":message}}).to_string()),
            })
        }))
        .into_response();
    }
    while let Some(out) = rx.recv().await {
        match out {
            Out::Images(images) => return Json(images_value(images)).into_response(),
            Out::Error(message) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({"error":{"message":message}})),
                )
                    .into_response()
            }
            Out::Progress(_) => {}
        }
    }
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({"error":{"message":"image worker stopped"}})),
    )
        .into_response()
}
/// The GPU shared by chat and image work. Chat counts itself as waiting while it blocks,
/// and image generation hands the GPU over between denoising steps when it does.
#[derive(Default)]
pub(crate) struct Gate {
    lock: Mutex<()>,
    waiting: AtomicUsize,
}

impl Gate {
    /// Take the GPU for a chat job, signalling image work to yield.
    pub(crate) fn lock(&self) -> MutexGuard<'_, ()> {
        self.waiting.fetch_add(1, Ordering::SeqCst);
        let guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        guard
    }

    /// Take the GPU for image work, without counting as a waiter.
    fn hold(&self) -> MutexGuard<'_, ()> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Between image steps: if chat is waiting, release until every waiter has the GPU.
    fn yield_to_waiters<'a>(&'a self, guard: &mut Option<MutexGuard<'a, ()>>) {
        if self.waiting.load(Ordering::SeqCst) == 0 {
            return;
        }
        guard.take();
        while self.waiting.load(Ordering::SeqCst) > 0 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        *guard = Some(self.hold());
    }
}

/// The image model's id in `/v1/models` and responses.
pub(crate) const MODEL_ID: &str = "z-image-turbo";

/// Progress events buffered per request; a slow reader drops stale ones, never the result.
const PROGRESS_BUFFER: usize = 16;

pub(crate) fn mount<D, F>(root: PathBuf, make_device: F, gate: Arc<Gate>) -> Service
where
    D: ComputeDevice + 'static,
    F: Fn() -> Result<D> + Send + 'static,
{
    let previews = root.join("preview.json").is_file();
    let (jobs, rx) = std_mpsc::sync_channel::<Job>(8);
    let resident = Arc::new(AtomicBool::new(false));
    let worker_resident = resident.clone();
    let model = ImageModel {
        previews,
        path: root.clone(),
        resident: resident.clone(),
    };
    std::thread::spawn(move || {
        let mut pipeline = None;
        for job in rx {
            if job.tx.is_closed() {
                continue;
            }
            let mut guard = Some(gate.hold());
            if job.tx.is_closed() {
                continue;
            }
            if pipeline.is_none() {
                let result = make_device().and_then(|dev| Pipeline::load(dev, &root));
                match result {
                    Ok(p) => {
                        pipeline = Some(p);
                        worker_resident.store(true, Ordering::Relaxed);
                    }
                    Err(e) => {
                        let _ = job
                            .tx
                            .blocking_send(Out::Error(format!("loading image model: {e:#}")));
                        continue;
                    }
                }
            }
            let tx = job.tx.clone();
            let result = pipeline
                .as_ref()
                .unwrap()
                .generate(&job.request, &mut |progress| {
                    gate.yield_to_waiters(&mut guard);
                    // A full buffer only drops this update; a closed one cancels.
                    !matches!(
                        tx.try_send(Out::Progress(progress)),
                        Err(mpsc::error::TrySendError::Closed(_))
                    )
                });
            drop(guard);
            let out = match result {
                Ok(images) => Out::Images(images),
                Err(e) => Out::Error(format!("{e:#}")),
            };
            let _ = job.tx.blocking_send(out);
        }
    });
    let app = App {
        jobs,
        resident,
        previews,
    };
    let router = Router::new()
        .route("/v1/images/generations", post(generate))
        .with_state(app.clone());
    Service { router, model, app }
}

pub fn serve<D, F>(addr: &str, root: PathBuf, key: Option<String>, make_device: F) -> Result<()>
where
    D: ComputeDevice + 'static,
    F: Fn() -> Result<D> + Send + 'static,
{
    let service = mount(root, make_device, Arc::default());
    let api = service.router.merge(
        Router::new()
            .route("/v1/models", get(models))
            .with_state(service.app),
    );
    let api = match key {
        Some(key) => api.layer(middleware::from_fn_with_state(
            Arc::new(key),
            crate::server::require_key,
        )),
        None => api,
    };
    let router = Router::new()
        .route("/health", get(|| async { "ok" }))
        .merge(api);
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        eprintln!("tang-llm images: {addr}; weights load on first generation");
        axum::serve(listener, router).await
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_capability_requires_installed_calibration() {
        assert!(!capabilities(false).contains(&"latent_previews"));
        assert!(capabilities(true).contains(&"latent_previews"));
    }

    #[test]
    fn preview_without_stream_is_rejected_before_queueing() {
        let (jobs, receiver) = std_mpsc::sync_channel(8);
        let app = App {
            jobs,
            resident: Arc::new(AtomicBool::new(false)),
            previews: false,
        };
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let request = Request {
                model: "z-image-turbo".into(),
                prompt: "frog".into(),
                size: "64x64".into(),
                n: 1,
                seed: Some(42),
                steps: 8,
                stream: false,
                preview: true,
                response_format: None,
            };
            let response = generate(
                State(app.clone()),
                axum::http::HeaderMap::new(),
                Json(request.clone()),
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let mut streamed = request;
            streamed.stream = true;
            let response = generate(State(app), axum::http::HeaderMap::new(), Json(streamed)).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert!(receiver.try_recv().is_err());
        });
    }

    #[test]
    fn full_image_queue_returns_429_instead_of_growing_without_bound() {
        let (jobs, _receiver) = std_mpsc::sync_channel(8);
        let app = App {
            jobs,
            resident: Arc::new(AtomicBool::new(false)),
            previews: false,
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let request = || Request {
                model: "z-image-turbo".into(),
                prompt: "frog".into(),
                size: "64x64".into(),
                n: 1,
                seed: Some(42),
                steps: 8,
                stream: true,
                preview: false,
                response_format: None,
            };
            let mut accepted = Vec::new();
            for _ in 0..8 {
                let response = generate(
                    State(app.clone()),
                    axum::http::HeaderMap::new(),
                    Json(request()),
                )
                .await;
                assert_eq!(response.status(), StatusCode::OK);
                accepted.push(response);
            }
            let rejected =
                generate(State(app), axum::http::HeaderMap::new(), Json(request())).await;
            assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
            drop(accepted);
        });
    }
}
