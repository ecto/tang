# Observability: OpenTelemetry traces and metrics

`tang-llm serve` can export per-request traces and per-step metrics over OTLP/HTTP
(protobuf). It's behind the cargo feature `otel`, off by default: the default build has no
OpenTelemetry dependencies and every hook in `src/telemetry.rs` compiles to nothing.

```bash
cargo build --release --features cuda,otel   # or metal,otel
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io \
OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=<key>" \
OTEL_SERVICE_NAME=tang-llm \
  tang-llm serve ...
```

| Variable | Meaning |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Base URL; `/v1/traces` and `/v1/metrics` are appended. Unset or empty: nothing is exported and the hooks return after one `OnceLock` check. |
| `OTEL_EXPORTER_OTLP_HEADERS` | `k=v,k2=v2`, e.g. the backend's API key. |
| `OTEL_SERVICE_NAME` | Defaults to `tang-llm`. |
| `OTEL_RESOURCE_ATTRIBUTES` | Extra resource attributes (`host.name=…,deployment.environment=…`). |

The other standard `OTEL_EXPORTER_OTLP_*` variables (per-signal endpoints, timeouts) are read by
the exporter too. Spans go through a batch processor and metrics through a 10 s periodic
reader, each on its own thread; neither touches the model thread.

## Traces

`POST /v1/chat/completions` (and `/v1/prefill`) continue an incoming W3C `traceparent`; without
one, a new trace starts. Each request is one server span with four children:

```
chat.completions  (server; request.id, gen_ai.request.model, outcome)
├─ queued    priority, queue.wait_ms, yields
├─ prefill   prompt_tokens, cached_tokens, new_tokens, chunks
├─ decode    completion_tokens, draft_tokens, accepted_draft_tokens, verify_windows,
│            tokens_per_verify_window, shared_streams_avg, tokens_per_second
└─ finish    outcome = ok | cancelled | error
```

- `request.id` is the client's `x-request-id` (the same id as the usage log line);
  `gen_ai.request.model` is the loaded model's name. Both are on every span.
- `queued` runs from arrival to the first time the worker picks the request up. A background
  request that gives way to interactive work and goes back in the queue keeps that span;
  `yields` counts how often it gave way.
- `prefill` ends where decode starts: decode is the last `completion_tokens / decode_tok_s`
  seconds of the run, as the engine measures it.
- `verify_windows` is the decode forwards the stream took; with speculative decoding
  `tokens_per_verify_window` is above 1. `shared_streams_avg` is how many other streams shared
  its forwards, on average (0 alone).
- An error sets the root span's status to error with the message.

Spans are built when the request concludes, from timestamps it carried; there's no span per
decode step.

## Metrics

| Instrument | Kind | Attributes | What |
|---|---|---|---|
| `tang_llm.step.duration` (s) | histogram | `kind` = decode / mixed / prefill | One forward. `mixed` is decode rows plus prefill chunks in one step. |
| `tang_llm.step.rows` | histogram | `kind` | Streams in that forward. |
| `tang_llm.prefill.chunk_tokens` | histogram | | Mean prefill chunk of a step that had any. |
| `tang_llm.prefix.hit_ratio` | histogram | | Per request: cached / prompt tokens. |
| `tang_llm.queue.depth` | gauge | | Requests waiting, sampled between batched steps and when one is picked up. |
| `tang_llm.stream.tokens_per_second` | histogram | | Per request: decode tok/s. |

Step metrics come from `Engine::step`, so they cover batched and single-stream serving alike.

## Overhead

Measured with `examples/bench_batch` (it calls `telemetry::init()`), Qwen3-0.6B-Base bf16 on an
M4 Max (Metal), 256 tokens per stream, five interleaved runs of each build, the `otel` build
exporting to a local OTLP sink. A 0.6B model has the shortest steps, so this is the worst case
for relative overhead.

The hook itself, `bench_batch x --hook-cost` (2M calls of `telemetry::step`):

| build | ns per step |
|---|---|
| default | 0.4 (an empty call) |
| `otel`, exporting | 163–186 |

Against the shortest step measured (5 ms, one 0.6B stream) that is 0.003%. Per request,
`finish` builds five spans once (a few µs, off the per-step path).

End to end, ms/step (median of 5 interleaved runs; min–max):

| batch | default | `otel` |
|---|---|---|
| 1 | 5.54 (4.95–7.68) | 6.05 (5.14–7.45) |
| 4 | 11.71 (10.31–16.15) | 12.02 (10.74–18.18) |
| 8 | 16.31 (15.64–29.36) | 16.82 (15.64–30.26) |

The M4's run-to-run spread (up to 2x within this session) is far wider than 1%, so the
end-to-end runs bound nothing finer than that; the hook timing is the evidence for "under
1%". Rerun on a quiet CUDA box to confirm end to end:

```bash
cargo build --release --example bench_batch --no-default-features --features cuda,otel
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 target/release/examples/bench_batch <model> --batches 1,4,8
```

## Tests

`cargo test -p tang-llm --features otel --lib otel` runs a request through the worker with
in-memory span and metric exporters, checking the traceparent continuation, the span tree,
each phase's attributes and that every metric is exported. The default `cargo test` runs
without it.

---

## Appendix: zest (edge → router → tang-llm) follow-up spec

zest is a separate private repo; this is what it needs so one trace spans the hops. Nothing
here is built yet. The user picks the backend (Honeycomb or Grafana Cloud) and creates its
account and keys.

### Shape of a trace

```
edge  zest.request            (Worker, root unless the client sent traceparent)
├─ edge  auth                 KV key lookup
├─ edge  admit                Account DO admit
└─ edge  router.call          fetch to the router; traceparent → router
   └─ router  chat            (axum, continues edge's traceparent)
      ├─ router  admit        tier limits
      ├─ router  upstream.connect   until tang-llm's response headers
      ├─ router  upstream.ttfb      until the first body chunk
      ├─ router  stream             first chunk → end of relay
      └─ tang-llm chat.completions  (traceparent forwarded; spans above)
```

Every span carries `zest.request_id` (the edge's `requestId`, already sent to the router and
on to tang-llm as `x-request-id`, so tang-llm's `request.id` matches), plus `zest.tier`, and
on the edge `zest.key_id` (never the key). Don't put prompts or completions in spans.

### Edge (Cloudflare Worker, `edge/`)

Workers' native tracing (open beta, Oct 2026) exports OTLP to an account-level destination
and auto-instruments `fetch`, KV, D1 and Durable Object calls; custom spans use
`tracing.enterSpan()` (changelog 2026-06-16). Check the current docs for these before
building: propagation behaviour was still changing.

1. `wrangler.jsonc`:
   ```jsonc
   "observability": {
     "traces": { "enabled": true, "head_sampling_rate": 1, "destinations": ["zest-traces"] }
   }
   ```
   The destination `zest-traces` is created in the dashboard (Workers & Pages → Observability →
   Destinations): the OTLP traces endpoint plus the auth header. Its secret lives in
   Cloudflare's destination config, not in the Worker.
2. In `proxy()`: wrap the KV lookup in `enterSpan("auth")`, the DO admit in
   `enterSpan("admit")`, and the router fetch in `enterSpan("router.call")`, setting the
   attributes above and `http.response.status_code`.
3. Propagation to the router: if native tracing injects `traceparent` on the outbound
   fetch, done. If not, build it from the active span's ids (`00-<trace>-<span>-01`) and set
   it on the fetch's headers. Since the router authenticates the edge by HMAC, add
   `traceparent` to what `routerHeaders` signs (or at least never trust a client's
   `traceparent` past the edge: drop any incoming one unless a decision is made to accept
   client traces).
4. Fallback, if native export can't carry custom spans or ids: `@microlabs/otel-cf-workers`
   with `exporter: { url: env.OTEL_EXPORTER_OTLP_ENDPOINT + "/v1/traces", headers }`.
   Then the Worker needs a var `OTEL_EXPORTER_OTLP_ENDPOINT`, a secret
   `OTEL_EXPORTER_OTLP_HEADERS` (`wrangler secret put`), and `OTEL_SERVICE_NAME=zest-edge`.

Env / secrets (edge): native path — destination `zest-traces` (dashboard), nothing in the
Worker. Fallback — var `OTEL_EXPORTER_OTLP_ENDPOINT`, var `OTEL_SERVICE_NAME=zest-edge`,
secret `OTEL_EXPORTER_OTLP_HEADERS`.

### Router (Rust, axum, `router/`)

1. Deps behind a feature `otel` (default on for deploys is fine there):
   `opentelemetry 0.30`, `opentelemetry_sdk 0.30`, `opentelemetry-otlp 0.30` with
   `http-proto` + `reqwest-client` (async client: the router is already on tokio),
   `tracing-opentelemetry 0.31`. Add the OTel layer to the existing `tracing_subscriber`
   registry in `main.rs`, so `tracing` spans become OTel spans.
2. `proxy::chat`: extract the parent with `TraceContextPropagator` from the incoming
   headers (only after `identify` has verified the HMAC). Root span `chat` (kind server).
   Child spans: `admit` around `state.limits.admit`; `upstream.connect` from `send()` until
   it resolves (headers in); `upstream.ttfb` from then to the first body chunk;
   `stream` from the first chunk until `Tracker::end`, with `outcome`, `status`,
   `bytes`, and the usage tokens the tracker already parses.
3. Forward `traceparent` (and `tracestate`) to tang-llm: inject the `upstream.connect`
   span's context into the reqwest request headers.
4. Metrics (optional, same exporter): `zest_router.ttfb` histogram, `zest_router.rejected`
   counter by reason (the existing `/metrics` counters can move here).

Env (router, in `~/.config/zest/zest.env` / the systemd unit): `OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_EXPORTER_OTLP_HEADERS` (secret — keep it in `secrets.env`, chmod 600),
`OTEL_SERVICE_NAME=zest-router`, optionally
`OTEL_RESOURCE_ATTRIBUTES=deployment.environment=phase-a,host.name=mew`.

### tang-llm (on the node)

Build with `--features cuda,otel`; set `OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_EXPORTER_OTLP_HEADERS` (same secret as the router), `OTEL_SERVICE_NAME=tang-llm`.

### Backend values

| | Honeycomb | Grafana Cloud |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `https://api.honeycomb.io` (EU: `https://api.eu1.honeycomb.io`) | the stack's OTLP gateway, `https://otlp-gateway-<zone>.grafana.net/otlp` |
| `OTEL_EXPORTER_OTLP_HEADERS` | `x-honeycomb-team=<ingest key>` | `Authorization=Basic <base64(instance_id:token)>` |
| Edge destination | same endpoint + `/v1/traces`, same header | same |

Sampling: keep everything at phase-A volumes; later, head-sample at the edge
(`head_sampling_rate`) and let the router and tang-llm follow the parent's sampled flag
(the SDK default `ParentBased(AlwaysOn)` does).

### Done when

One request through the public edge shows a single trace with edge, router and tang-llm
spans, `zest.request_id` = tang-llm's `request.id`, and edge `router.call` ≈ router `chat` +
network.
