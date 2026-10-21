//! OpenTelemetry for tang: one process-wide tracer and meter, exported as
//! **OTLP/JSON lines to a file** by default and over OTLP/HTTP only when asked.
//!
//! Every tang crate's `otel` feature routes through here, so the K1 runtime
//! (tang-gpu's OpenCL, tang-bench's `qwen_session`) and `tang-llm serve` write
//! the same format with the same conventions as ipse's trainer
//! (`ipse-sim/src/otel.rs`): one line = one complete
//! `ExportTraceServiceRequest` or `ExportMetricsServiceRequest` in the proto3
//! JSON mapping (hex ids, nanosecond timestamps as strings), which is exactly
//! what the collector's `otlpjsonfile` receiver reads.
//!
//! # Configuration (read once, on first use)
//!
//! | Variable | Meaning |
//! |---|---|
//! | `OTEL_FILE` / `TANG_OTEL_FILE` | Where the lines go. Unset: `~/.cache/tang/otel/<service>.otel.jsonl`. `off` / `0` / empty: no file. |
//! | `TANG_OTEL_LIVE=1` | Also export over OTLP/HTTP to `OTEL_EXPORTER_OTLP_ENDPOINT`. Without it the endpoint is ignored: **no network by default**. |
//! | `OTEL_SERVICE_NAME` | Overrides the service name given to [`init`]. |
//! | `OTEL_RESOURCE_ATTRIBUTES` | Extra resource attributes. |
//!
//! Neither a file nor live export: everything is off and [`enabled`] is one
//! load of an initialised `OnceLock`.
//!
//! Spans go through batch processors (their own thread, 5 s / 512 spans), so
//! the caller's thread never writes the file. Call [`init`] early and keep the
//! [`Guard`] alive to flush at exit; without it, the first [`enabled`] call
//! initialises lazily (service `tang`) and [`flush`] pushes what is pending.

use opentelemetry::metrics::MeterProvider as _;
use opentelemetry::propagation::TextMapPropagator;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider, Temporality};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider, SpanData};
use opentelemetry_sdk::Resource;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};

pub use opentelemetry;
pub use opentelemetry::metrics::Meter;
pub use opentelemetry::trace::{Span, SpanKind, TraceContextExt, Tracer};
pub use opentelemetry::{Context, KeyValue};

struct State {
    traces: SdkTracerProvider,
    metrics: SdkMeterProvider,
    tracer: SdkTracer,
    meter: Meter,
}

static STATE: OnceLock<Option<State>> = OnceLock::new();

/// Flushes and shuts the exporters down on drop.
pub struct Guard(());

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(Some(s)) = STATE.get() {
            let _ = s.traces.force_flush();
            let _ = s.metrics.force_flush();
            let _ = s.traces.shutdown();
            let _ = s.metrics.shutdown();
        }
    }
}

/// Set up exporters for `service` (unless something already did). Returns a
/// guard to hold for the life of the process, or `None` when telemetry is off
/// or was initialised earlier. Call off any async runtime: the live
/// exporter's blocking HTTP client builds its own.
pub fn init(service: &str) -> Option<Guard> {
    let mut first = false;
    STATE.get_or_init(|| {
        first = true;
        build(service)
    });
    (first && enabled()).then_some(Guard(()))
}

/// Telemetry is on (initialising it with service `tang` if nobody has).
#[inline]
pub fn enabled() -> bool {
    STATE.get_or_init(|| build("tang")).is_some()
}

fn state() -> Option<&'static State> {
    STATE.get_or_init(|| build("tang")).as_ref()
}

/// The process tracer (`None` when off).
pub fn tracer() -> Option<&'static SdkTracer> {
    state().map(|s| &s.tracer)
}

/// The SDK providers behind [`tracer`] and [`meter`], for crates that build
/// their own instruments from them (tang-llm's `telemetry::install`).
pub fn providers() -> Option<(&'static SdkTracerProvider, &'static SdkMeterProvider)> {
    state().map(|s| (&s.traces, &s.metrics))
}

/// The process meter (`None` when off).
pub fn meter() -> Option<&'static Meter> {
    state().map(|s| &s.meter)
}

/// Push pending spans and metrics now (e.g. at the end of a turn).
pub fn flush() {
    if let Some(s) = STATE.get().and_then(Option::as_ref) {
        let _ = s.traces.force_flush();
        let _ = s.metrics.force_flush();
    }
}

/// The file in use, as resolved from the environment (`None`: no file).
pub fn file_path(service: &str) -> Option<PathBuf> {
    let v = std::env::var_os("OTEL_FILE").or_else(|| std::env::var_os("TANG_OTEL_FILE"));
    match v {
        Some(v) => {
            let s = v.to_string_lossy();
            if s.is_empty() || s == "off" || s == "0" {
                None
            } else {
                Some(PathBuf::from(v))
            }
        }
        None => {
            let home = std::env::var_os("HOME").map(PathBuf::from)?;
            Some(home.join(format!(".cache/tang/otel/{service}.otel.jsonl")))
        }
    }
}

/// Live OTLP/HTTP export was asked for (`TANG_OTEL_LIVE=1`) and has an endpoint.
pub fn live_endpoint() -> Option<String> {
    let on = matches!(
        std::env::var("TANG_OTEL_LIVE").as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    );
    let ep = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").unwrap_or_default();
    (on && !ep.trim().is_empty()).then_some(ep)
}

fn hostname() -> Option<String> {
    if let Ok(h) = std::env::var("HOSTNAME") {
        if !h.is_empty() {
            return Some(h);
        }
    }
    if let Ok(h) = std::fs::read_to_string("/etc/hostname") {
        let h = h.trim();
        if !h.is_empty() {
            return Some(h.to_string());
        }
    }
    let out = std::process::Command::new("hostname").output().ok()?;
    let h = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!h.is_empty()).then_some(h)
}

fn resource(service: &str) -> Resource {
    let mut b = Resource::builder();
    if std::env::var("OTEL_SERVICE_NAME").map_or(true, |s| s.is_empty()) {
        b = b.with_service_name(service.to_string());
    }
    let mut attrs = vec![
        KeyValue::new("process.pid", std::process::id() as i64),
        KeyValue::new("tang.version", env!("CARGO_PKG_VERSION")),
    ];
    if let Some(h) = hostname() {
        attrs.push(KeyValue::new("host.name", h));
    }
    b.with_attributes(attrs).build()
}

fn build(service: &str) -> Option<State> {
    let file = file_path(service);
    let live = live_endpoint();
    if std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_ok_and(|e| !e.trim().is_empty())
        && live.is_none()
    {
        eprintln!("tang-otel: OTEL_EXPORTER_OTLP_ENDPOINT ignored (set TANG_OTEL_LIVE=1 to export over the network)");
    }
    if file.is_none() && live.is_none() {
        eprintln!("tang-otel: OFF");
        return None;
    }
    let resource = resource(service);
    let mut tb = SdkTracerProvider::builder().with_resource(resource.clone());
    let mut mb = SdkMeterProvider::builder().with_resource(resource);
    if let Some(path) = &file {
        match FileSink::open(path) {
            Ok(sink) => {
                tb = tb.with_batch_exporter(FileSpans {
                    sink: sink.clone(),
                    resource: Resource::builder_empty().build(),
                });
                mb = mb.with_reader(
                    PeriodicReader::builder(FileMetrics { sink })
                        .with_interval(Duration::from_secs(10))
                        .build(),
                );
                eprintln!("tang-otel: writing OTLP/JSON lines to {}", path.display());
            }
            Err(e) => eprintln!("tang-otel: can't open {}: {e}", path.display()),
        }
    }
    if let Some(ep) = &live {
        use opentelemetry_otlp::{MetricExporter, Protocol, SpanExporter, WithExportConfig};
        let spans = SpanExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .build();
        let metrics = MetricExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .build();
        match (spans, metrics) {
            (Ok(s), Ok(m)) => {
                tb = tb.with_batch_exporter(s);
                mb = mb.with_reader(
                    PeriodicReader::builder(m)
                        .with_interval(Duration::from_secs(10))
                        .build(),
                );
                eprintln!("tang-otel: also exporting live to {ep}");
            }
            (Err(e), _) => eprintln!("tang-otel: live export off: {e}"),
            (_, Err(e)) => eprintln!("tang-otel: live export off: {e}"),
        }
    }
    let traces = tb.build();
    let metrics = mb.build();
    let tracer = traces.tracer("tang");
    let meter = metrics.meter("tang");
    Some(State {
        traces,
        metrics,
        tracer,
        meter,
    })
}

/// The appended file, shared by the span and metric exporters; one mutex so
/// their lines never interleave.
#[derive(Debug, Clone)]
struct FileSink(Arc<Mutex<std::fs::File>>);

impl FileSink {
    fn open(path: &std::path::Path) -> std::io::Result<Self> {
        if let Some(d) = path.parent() {
            if !d.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(d);
            }
        }
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self(Arc::new(Mutex::new(f))))
    }

    /// One object, one line, one `write_all`. Errors are swallowed: a full disk
    /// must not take the model down.
    fn line(&self, v: &impl serde::Serialize) {
        let Ok(mut s) = serde_json::to_string(v) else {
            return;
        };
        s.push('\n');
        if let Ok(mut f) = self.0.lock() {
            let _ = f.write_all(s.as_bytes());
        }
    }
}

use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
#[derive(Debug)]
struct FileSpans {
    sink: FileSink,
    resource: Resource,
}

impl opentelemetry_sdk::trace::SpanExporter for FileSpans {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        if batch.is_empty() {
            return Ok(());
        }
        let rs = opentelemetry_proto::transform::trace::tonic::group_spans_by_resource_and_scope(
            batch,
            &(&self.resource).into(),
        );
        self.sink
            .line(&ExportTraceServiceRequest { resource_spans: rs });
        Ok(())
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.resource = resource.clone();
    }
}

struct FileMetrics {
    sink: FileSink,
}

impl opentelemetry_sdk::metrics::exporter::PushMetricExporter for FileMetrics {
    async fn export(
        &self,
        metrics: &opentelemetry_sdk::metrics::data::ResourceMetrics,
    ) -> OTelSdkResult {
        let req = ExportMetricsServiceRequest::from(metrics);
        if req
            .resource_metrics
            .iter()
            .all(|r| r.scope_metrics.is_empty())
        {
            return Ok(());
        }
        self.sink.line(&req);
        Ok(())
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn shutdown_with_timeout(&self, _: Duration) -> OTelSdkResult {
        Ok(())
    }

    fn temporality(&self) -> Temporality {
        Temporality::Cumulative
    }
}

/// A remote parent from a W3C `traceparent` (`00-<32 hex>-<16 hex>-<2 hex>`),
/// e.g. broca's per-turn span; spans started under it join that trace.
pub fn parent_from_traceparent(traceparent: &str) -> Option<Context> {
    let mut h = std::collections::HashMap::new();
    h.insert("traceparent".to_string(), traceparent.trim().to_string());
    let cx = TraceContextPropagator::new().extract(&h);
    cx.span().span_context().is_valid().then_some(cx)
}

/// A remote parent from a trace id (32 hex) and span id (16 hex), sampled.
pub fn parent_from_ids(trace_id: &str, span_id: &str) -> Option<Context> {
    parent_from_traceparent(&format!("00-{trace_id}-{span_id}-01"))
}

/// The W3C `traceparent` of the span current in `cx` (to hand to a child process or service).
pub fn traceparent(cx: &Context) -> Option<String> {
    let sc = cx.span().span_context().clone();
    sc.is_valid().then(|| {
        format!(
            "00-{}-{}-{:02x}",
            sc.trace_id(),
            sc.span_id(),
            sc.trace_flags().to_u8()
        )
    })
}

/// Every `n`th call returns true (n = 0: never; n = 1: always).
#[inline]
pub fn every(counter: &AtomicU64, n: u64) -> bool {
    n != 0 && counter.fetch_add(1, Ordering::Relaxed) % n == 0
}

/// A `u64` from the environment, or `default`.
pub fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Record a finished span from explicit times, under `parent`.
pub fn span_at(
    name: &'static str,
    parent: &Context,
    start: SystemTime,
    end: SystemTime,
    attrs: Vec<KeyValue>,
) {
    let Some(t) = tracer() else { return };
    let mut s = t
        .span_builder(name)
        .with_start_time(start)
        .with_attributes(attrs)
        .start_with_context(t, parent);
    s.end_with_timestamp(end);
}
