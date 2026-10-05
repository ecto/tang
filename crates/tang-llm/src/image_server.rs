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
        atomic::{AtomicBool, Ordering},
        mpsc as std_mpsc, Arc, Mutex,
    },
};
use tang_compute::ComputeDevice;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
enum Out {
    Progress(usize, usize),
    Images(Vec<Generated>),
    Error(String),
}
struct Job {
    request: Request,
    tx: mpsc::UnboundedSender<Out>,
}
#[derive(Clone)]
struct App {
    jobs: std_mpsc::SyncSender<Job>,
    resident: Arc<AtomicBool>,
}

#[derive(Clone)]
pub(crate) struct ImageModel {
    pub path: PathBuf,
    pub resident: Arc<AtomicBool>,
}
impl ImageModel {
    pub fn json(&self) -> Value {
        json!({"id":"z-image-turbo","object":"model","type":"image",
            "capabilities":["image_generation","seed","steps","stream_progress"],
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
    json!({"model":"z-image-turbo","model_hash":model_hash,"data":images.into_iter().map(|image|json!({"b64_json":STANDARD.encode(image.png),"seed":image.seed,"steps":image.steps,"width":image.width,"height":image.height})).collect::<Vec<_>>()})
}
async fn models(State(app): State<App>) -> Json<Value> {
    Json(
        json!({"data":[{"id":"z-image-turbo","object":"model","type":"image","resident":app.resident.load(Ordering::Relaxed)}]}),
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
    let (tx, mut rx) = mpsc::unbounded_channel();
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
        let events = UnboundedReceiverStream::new(rx);
        use tokio_stream::StreamExt;
        return Sse::new(events.map(|out| {
            Ok::<_, Infallible>(match out {
                Out::Progress(step, of) => Event::default()
                    .event("progress")
                    .data(json!({"step":step,"of":of}).to_string()),
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
            Out::Progress(_, _) => {}
        }
    }
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({"error":{"message":"image worker stopped"}})),
    )
        .into_response()
}
pub(crate) fn mount<D, F>(root: PathBuf, make_device: F, load_gate: Arc<Mutex<()>>) -> Service
where
    D: ComputeDevice + 'static,
    F: Fn() -> Result<D> + Send + 'static,
{
    let (jobs, rx) = std_mpsc::sync_channel::<Job>(8);
    let resident = Arc::new(AtomicBool::new(false));
    let worker_resident = resident.clone();
    let model = ImageModel {
        path: root.clone(),
        resident: resident.clone(),
    };
    std::thread::spawn(move || {
        let mut pipeline = None;
        for job in rx {
            if job.tx.is_closed() {
                continue;
            }
            let _guard = load_gate.lock().unwrap_or_else(|e| e.into_inner());
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
                            .send(Out::Error(format!("loading image model: {e:#}")));
                        continue;
                    }
                }
            }
            let tx = job.tx.clone();
            let result = pipeline
                .as_ref()
                .unwrap()
                .generate(&job.request, &mut |step, of| {
                    tx.send(Out::Progress(step, of)).is_ok()
                });
            let out = match result {
                Ok(images) => Out::Images(images),
                Err(e) => Out::Error(format!("{e:#}")),
            };
            let _ = job.tx.send(out);
        }
    });
    let app = App { jobs, resident };
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
    let service = mount(root, make_device, Arc::new(Mutex::new(())));
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
    fn full_image_queue_returns_429_instead_of_growing_without_bound() {
        let (jobs, _receiver) = std_mpsc::sync_channel(8);
        let app = App {
            jobs,
            resident: Arc::new(AtomicBool::new(false)),
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
