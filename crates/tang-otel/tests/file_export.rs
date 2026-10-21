//! The default exporter writes OTLP/JSON lines, spans join a remote parent,
//! and nothing goes to the network without TANG_OTEL_LIVE.

use tang_otel::{KeyValue, Span as _, TraceContextExt as _, Tracer as _};

#[test]
fn spans_and_metrics_land_as_otlp_json_lines() {
    let dir = std::env::temp_dir().join(format!("tang-otel-test-{}", std::process::id()));
    let path = dir.join("t.otel.jsonl");
    std::env::set_var("OTEL_FILE", &path);
    std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9");
    std::env::remove_var("TANG_OTEL_LIVE");
    assert!(
        tang_otel::live_endpoint().is_none(),
        "endpoint alone must not enable network export"
    );
    let guard = tang_otel::init("tang-otel-test").expect("on");

    let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let parent = tang_otel::parent_from_traceparent(tp).expect("valid traceparent");
    let t = tang_otel::tracer().unwrap();
    let mut s = t
        .span_builder("unit")
        .with_attributes(vec![KeyValue::new("tang.n", 3i64)])
        .start_with_context(t, &parent);
    let child_cx = parent.with_remote_span_context(s.span_context().clone());
    assert_eq!(
        tang_otel::traceparent(&child_cx).unwrap()[3..35],
        tp[3..35],
        "same trace id"
    );
    s.end();
    tang_otel::meter()
        .unwrap()
        .u64_counter("tang.test.count")
        .build()
        .add(2, &[]);
    drop(guard);

    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let span = lines
        .iter()
        .find_map(|l| l.pointer("/resourceSpans/0/scopeSpans/0/spans/0").cloned())
        .expect("a trace line");
    assert_eq!(span["name"], "unit");
    assert_eq!(span["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(span["parentSpanId"], "00f067aa0ba902b7");
    assert!(
        span["startTimeUnixNano"].is_string(),
        "u64 as string: {span}"
    );
    let svc = lines
        .iter()
        .find_map(|l| l.pointer("/resourceSpans/0/resource/attributes").cloned())
        .unwrap();
    assert!(svc.to_string().contains("tang-otel-test"));
    assert!(
        lines.iter().any(|l| l.get("resourceMetrics").is_some()),
        "a metrics line: {text}"
    );
    let _ = std::fs::remove_dir_all(dir);
}
