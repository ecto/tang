//! OpenAI-compatible lazy image worker. One resident pipeline, serialized jobs;
//! disconnecting a streaming client cancels at the next step boundary.
use crate::image_pipeline::{Generated, Pipeline, Request};
use anyhow::Result;
use axum::{
    extract::{Request as HttpRequest, State},
    http::{header, StatusCode},
    middleware::{self, Next},
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
        mpsc as std_mpsc, Arc,
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
    jobs: std_mpsc::Sender<Job>,
    resident: Arc<AtomicBool>,
    key: Option<String>,
}
fn images_value(images: Vec<Generated>) -> Value {
    json!({"model":"z-image-turbo","data":images.into_iter().map(|image|json!({"b64_json":STANDARD.encode(image.png),"seed":image.seed,"steps":image.steps,"width":image.width,"height":image.height})).collect::<Vec<_>>()})
}
async fn models(State(app): State<App>) -> Json<Value> {
    Json(
        json!({"data":[{"id":"z-image-turbo","object":"model","type":"image","resident":app.resident.load(Ordering::Relaxed)}]}),
    )
}
async fn authenticate(State(app): State<App>, req: HttpRequest, next: Next) -> Response {
    if req.uri().path() != "/health" {
        if let Some(key) = &app.key {
            let provided = req
                .headers()
                .get(header::AUTHORIZATION)
                .and_then(|s| s.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "));
            if provided != Some(key) {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({"error":{"message":"invalid API key"}})),
                )
                    .into_response();
            }
        }
    }
    next.run(req).await
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
    if app.jobs.send(Job { request, tx }).is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error":{"message":"image worker unavailable"}})),
        )
            .into_response();
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
pub fn serve<D, F>(addr: &str, root: PathBuf, key: Option<String>, make_device: F) -> Result<()>
where
    D: ComputeDevice + 'static,
    F: Fn() -> Result<D> + Send + 'static,
{
    let (jobs, rx) = std_mpsc::channel::<Job>();
    let resident = Arc::new(AtomicBool::new(false));
    let worker_resident = resident.clone();
    std::thread::spawn(move || {
        let mut pipeline = None;
        for job in rx {
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
    let app = App {
        jobs,
        resident,
        key,
    };
    let router = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/v1/images/generations", post(generate))
        .layer(middleware::from_fn_with_state(app.clone(), authenticate))
        .with_state(app);
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        eprintln!("tang-llm images: {addr}; weights load on first generation");
        axum::serve(listener, router).await
    })?;
    Ok(())
}
