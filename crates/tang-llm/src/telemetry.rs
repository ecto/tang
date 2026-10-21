//! Optional OpenTelemetry tracing and metrics (see `docs/observability.md`).
//!
//! With the `otel` feature, each chat request becomes
//! a server span (continuing an incoming W3C `traceparent`) with `queued`, `prefill`,
//! `decode` and `finish` children, and every batched step feeds a few metrics. Spans are
//! built when the request concludes, from timestamps it carried, so nothing runs per token.
//! Export goes through `crates/tang-otel`: OTLP/JSON lines to a file by default, the
//! network only with `TANG_OTEL_LIVE=1`. Without the feature every type here is
//! zero-sized and every call is empty; with it and `OTEL_FILE=off`, each call is one
//! check of an unset `OnceLock`.

use crate::engine::{StepStats, Usage};
use axum::http::HeaderMap;

/// How a request ended, for its `finish` span and the usage log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum End {
    Ok,
    Cancelled,
    Error,
}

impl End {
    pub fn as_str(self) -> &'static str {
        match self {
            End::Ok => "ok",
            End::Cancelled => "cancelled",
            End::Error => "error",
        }
    }
}

#[cfg(not(feature = "otel"))]
mod imp {
    use super::*;

    /// Exporters to flush on drop (none without the feature).
    pub struct Guard;

    pub fn init() -> Option<Guard> {
        None
    }

    /// A request's trace (nothing without the feature).
    #[derive(Debug, Default)]
    pub struct Trace;

    impl Trace {
        /// No trace (a request that wasn't received over HTTP).
        #[inline(always)]
        pub fn none() -> Self {
            Trace
        }
        #[inline(always)]
        pub fn incoming(_: &HeaderMap, _: &str, _: &str, _: Option<&str>) -> Self {
            Trace
        }
        #[inline(always)]
        pub fn dequeued(&mut self, _queue_depth: impl FnOnce() -> usize) {}
        #[inline(always)]
        pub fn finish(self, _: &Usage, _: End, _: u32, _: Option<&str>) {}
    }

    #[inline(always)]
    pub fn step(_: &StepStats) {}

    #[inline(always)]
    pub fn queue_depth(_: impl FnOnce() -> usize) {}
}

#[cfg(feature = "otel")]
mod imp {
    use super::*;
    use opentelemetry::metrics::{Gauge, Histogram, Meter, MeterProvider as _};
    use opentelemetry::propagation::{Extractor, TextMapPropagator};
    use opentelemetry::trace::{
        Span as _, SpanKind, Status, TraceContextExt, Tracer as _, TracerProvider as _,
    };
    use opentelemetry::{Context, KeyValue};
    use opentelemetry_sdk::metrics::SdkMeterProvider;
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider};
    use std::sync::OnceLock;
    use std::time::{Duration, SystemTime};

    struct Inst {
        tracer: SdkTracer,
        step_secs: Histogram<f64>,
        step_rows: Histogram<u64>,
        chunk_tokens: Histogram<u64>,
        hit_ratio: Histogram<f64>,
        queue_depth: Gauge<u64>,
        tok_s: Histogram<f64>,
        /// `kind` = decode | mixed | prefill, built once.
        kinds: [[KeyValue; 1]; 3],
    }

    static INST: OnceLock<Inst> = OnceLock::new();

    /// The exporters (crates/tang-otel), flushed and shut down on drop.
    pub struct Guard(#[allow(dead_code)] tang_otel::Guard);

    /// Starts telemetry through `tang-otel`: OTLP/JSON lines to `OTEL_FILE`
    /// (default `~/.cache/tang/otel/tang-llm.otel.jsonl`; `OTEL_FILE=off` for none),
    /// and over OTLP/HTTP to `OTEL_EXPORTER_OTLP_ENDPOINT` only with
    /// `TANG_OTEL_LIVE=1` (`serve --otel-live`). Call off any async runtime.
    pub fn init() -> Option<Guard> {
        let g = tang_otel::init("tang-llm")?;
        let (traces, metrics) = tang_otel::providers()?;
        install(traces, metrics);
        Some(Guard(g))
    }

    /// Record to these providers from now on (once per process; later calls are ignored).
    pub fn install(traces: &SdkTracerProvider, metrics: &SdkMeterProvider) {
        let meter: Meter = metrics.meter("tang-llm");
        let _ = INST.set(Inst {
            tracer: traces.tracer("tang-llm"),
            step_secs: meter
                .f64_histogram("tang_llm.step.duration")
                .with_unit("s")
                .with_description(
                    "One batched forward, by kind: decode, mixed (decode + prefill) or prefill",
                )
                .with_boundaries(vec![
                    0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0,
                ])
                .build(),
            step_rows: meter
                .u64_histogram("tang_llm.step.rows")
                .with_description("Streams in one batched forward")
                .with_boundaries(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
                .build(),
            chunk_tokens: meter
                .u64_histogram("tang_llm.prefill.chunk_tokens")
                .with_unit("{token}")
                .with_description("Tokens in one prefill chunk")
                .with_boundaries(vec![
                    32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0,
                ])
                .build(),
            hit_ratio: meter
                .f64_histogram("tang_llm.prefix.hit_ratio")
                .with_description("Of a request's prompt, the share reused from KV caches")
                .with_boundaries(vec![0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
                .build(),
            queue_depth: meter
                .u64_gauge("tang_llm.queue.depth")
                .with_description("Requests waiting to run")
                .build(),
            tok_s: meter
                .f64_histogram("tang_llm.stream.tokens_per_second")
                .with_unit("{token}/s")
                .with_description("A request's decode tokens per second")
                .with_boundaries(vec![
                    5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0,
                ])
                .build(),
            kinds: [
                [KeyValue::new("kind", "decode")],
                [KeyValue::new("kind", "mixed")],
                [KeyValue::new("kind", "prefill")],
            ],
        });
    }

    struct Headers<'a>(&'a HeaderMap);

    impl Extractor for Headers<'_> {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).and_then(|v| v.to_str().ok())
        }
        fn keys(&self) -> Vec<&str> {
            self.0.keys().map(|k| k.as_str()).collect()
        }
    }

    struct Live {
        parent: Context,
        received: SystemTime,
        dequeued: Option<SystemTime>,
        priority: &'static str,
        /// `request.id` (if any) and `gen_ai.request.model`, on every span.
        common: Vec<KeyValue>,
    }

    /// A request's trace: its parent context and when it reached each phase.
    #[derive(Default)]
    pub struct Trace(Option<Box<Live>>);

    impl std::fmt::Debug for Trace {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(if self.0.is_some() {
                "Trace(live)"
            } else {
                "Trace(off)"
            })
        }
    }

    impl Trace {
        /// No trace (a request that wasn't received over HTTP).
        pub fn none() -> Self {
            Trace(None)
        }

        pub fn incoming(
            headers: &HeaderMap,
            priority: &str,
            model: &str,
            request_id: Option<&str>,
        ) -> Self {
            if INST.get().is_none() {
                return Trace(None);
            }
            let parent = TraceContextPropagator::new().extract(&Headers(headers));
            let mut common = vec![KeyValue::new("gen_ai.request.model", model.to_string())];
            if let Some(id) = request_id {
                common.push(KeyValue::new("request.id", id.to_string()));
            }
            Trace(Some(Box::new(Live {
                parent,
                received: SystemTime::now(),
                dequeued: None,
                priority: if priority == "background" {
                    "background"
                } else {
                    "interactive"
                },
                common,
            })))
        }

        /// It left the queue (its first run; a yielded request's later runs keep this).
        pub fn dequeued(&mut self, queue_depth: impl FnOnce() -> usize) {
            let Some(inst) = INST.get() else { return };
            inst.queue_depth.record(queue_depth() as u64, &[]);
            if let Some(l) = self.0.as_mut() {
                l.dequeued.get_or_insert_with(SystemTime::now);
            }
        }

        /// Emit the request's spans, ending now. `usage` is as the client sees it.
        pub fn finish(self, usage: &Usage, end: End, yields: u32, error: Option<&str>) {
            let (Some(inst), Some(l)) = (INST.get(), self.0) else {
                return;
            };
            let now = SystemTime::now();
            let dequeued = l.dequeued.unwrap_or(now).max(l.received);
            let decode_secs = if usage.decode_tok_s > 0.0 && usage.completion_tokens > 0 {
                usage.completion_tokens as f64 / usage.decode_tok_s
            } else {
                0.0
            };
            let decode_start = now
                .checked_sub(Duration::from_secs_f64(decode_secs))
                .unwrap_or(now)
                .max(dequeued);
            if usage.prompt_tokens > 0 {
                inst.hit_ratio
                    .record(usage.cached_tokens as f64 / usage.prompt_tokens as f64, &[]);
            }
            if usage.decode_tok_s > 0.0 {
                inst.tok_s.record(usage.decode_tok_s, &[]);
            }

            let t = &inst.tracer;
            let mut root = t
                .span_builder("chat.completions")
                .with_kind(SpanKind::Server)
                .with_start_time(l.received)
                .with_attributes(l.common.clone())
                .start_with_context(t, &l.parent);
            root.set_attribute(KeyValue::new("outcome", end.as_str()));
            if end == End::Error {
                root.set_status(Status::error(error.unwrap_or("error").to_string()));
            }
            let cx = l
                .parent
                .with_remote_span_context(root.span_context().clone());
            let child =
                |name: &'static str, from: SystemTime, to: SystemTime, mut attrs: Vec<KeyValue>| {
                    attrs.extend(l.common.iter().cloned());
                    let mut s = t
                        .span_builder(name)
                        .with_start_time(from)
                        .with_attributes(attrs)
                        .start_with_context(t, &cx);
                    s.end_with_timestamp(to);
                };
            let wait = dequeued.duration_since(l.received).unwrap_or_default();
            child(
                "queued",
                l.received,
                dequeued,
                vec![
                    KeyValue::new("priority", l.priority),
                    KeyValue::new("queue.wait_ms", wait.as_secs_f64() * 1e3),
                    KeyValue::new("yields", yields as i64),
                ],
            );
            if l.dequeued.is_some() {
                child(
                    "prefill",
                    dequeued,
                    decode_start,
                    vec![
                        KeyValue::new("prompt_tokens", usage.prompt_tokens as i64),
                        KeyValue::new("cached_tokens", usage.cached_tokens as i64),
                        KeyValue::new(
                            "new_tokens",
                            usage.prompt_tokens.saturating_sub(usage.cached_tokens) as i64,
                        ),
                        KeyValue::new("chunks", usage.prefill_chunks as i64),
                    ],
                );
            }
            if decode_secs > 0.0 {
                let steps = usage.decode_steps.max(1) as f64;
                child(
                    "decode",
                    decode_start,
                    now,
                    vec![
                        KeyValue::new("completion_tokens", usage.completion_tokens as i64),
                        KeyValue::new("draft_tokens", usage.draft_tokens as i64),
                        KeyValue::new("accepted_draft_tokens", usage.accepted_tokens as i64),
                        KeyValue::new("verify_windows", usage.decode_steps as i64),
                        KeyValue::new(
                            "tokens_per_verify_window",
                            usage.completion_tokens as f64 / steps,
                        ),
                        KeyValue::new("shared_streams_avg", usage.shared_streams as f64 / steps),
                        KeyValue::new("tokens_per_second", usage.decode_tok_s),
                    ],
                );
            }
            child(
                "finish",
                now,
                now,
                vec![KeyValue::new("outcome", end.as_str())],
            );
            root.end_with_timestamp(now);
        }
    }

    /// One batched step's metrics (nothing for a step that ran no forward).
    pub fn step(s: &StepStats) {
        let Some(inst) = INST.get() else { return };
        if s.secs == 0.0 {
            return;
        }
        let kind = match (s.decode_streams > 0, s.prefill_chunks > 0) {
            (true, false) => 0,
            (true, true) => 1,
            _ => 2,
        };
        inst.step_secs.record(s.secs, &inst.kinds[kind]);
        inst.step_rows.record(
            (s.decode_streams + s.prefill_chunks) as u64,
            &inst.kinds[kind],
        );
        if let Some(mean) = s.prefill_tokens.checked_div(s.prefill_chunks) {
            inst.chunk_tokens.record(mean as u64, &[]);
        }
    }

    /// Requests waiting, sampled between steps.
    pub fn queue_depth(n: impl FnOnce() -> usize) {
        if let Some(inst) = INST.get() {
            inst.queue_depth.record(n() as u64, &[]);
        }
    }
}

pub use imp::*;
