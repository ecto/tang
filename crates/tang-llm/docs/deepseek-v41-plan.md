# DeepSeek-V4.1-Flash on tang: the port, milestone by milestone

Status (2026-10-27): M0–M2 landed and **the real model serves**: `tang-llm deepseek-serve` on
1× H200 (RunPod, experts streamed through an 18000-tensor LRU) answers chat and DSML tool calls
correctly through the OpenAI API. Speed is bring-up speed (decode ~1.2 tok/s, prefill
~0.7 tok/s warm): naive kernels, host-side top-k / routing, one GPU. M3–M6 are the speed work. The goal is frog cloud's "bull" model:
`cloud/bull` in frog, served by tang.

Shapes below are from the real `config.json` (`deepseek_v41_text`), not the estimates in
[models.md](models.md), which predates the GLM-5.3 work and is stale on two counts: tang now has
a Coder-Next GPU engine (`flash/coder.rs`) and most of GLM-5.3-Flash on the CPU reference
(`flash/glm5/`). Both cut the DeepSeek bill.

## Speed work landed (2026-10-27, waves 1–2)

Eight parallel lanes, merged; all 30 `flash::deepseek` tests pass on mew (3090) with every
switch at its default. Parity bar unchanged (teacher-forced KL ~1e-15 with `DS_EXACT=1`,
~5e-15 with the fast kernels).

- **Selection on the device** (`kernels_select.cu`): indexer masks, candidate blocks, radix
  top-k (ties to the lowest index), per-expert routing CSR, window index rows. A decode step
  reads nothing back but the logits.
- **Fast kernels** (`kernels.cu`, `DS_EXACT=1` restores the bit-exact ones): tensor-core
  K-split fp8/fp4 GEMVs (61–75% of 3090 bandwidth on the big dense shapes, 26–49% on the
  small ones: launch-floor bound), fused w1/w3 + SwiGLU, tiled 64-head sparse attention
  (10× the exact kernel). Relaxed bar: ≥99.5% bf16-identical, the rest 1 ulp.
- **CUDA graphs** (`graph.rs`, `kernels_graph.cu`, `DS_GRAPH=0` disables): the whole decode
  step and the 6-token verify window captured once per shape; positions come from a device
  params buffer; persistent arena + pinned staging, no lifetime syncs. Prefill, DSpark's
  Markov sampling, the expert LRU and the EP reduce stay outside.
- **Speculative decoding with DSpark** (`decode_spec`, `forward_all`, `snapshot`/`rewind`):
  verify windows at any position, exact rewind (incl. the ratio-2 compressors' partial
  groups, via a journal), greedy and rejection sampling (`SpecPick`: same temperature/top-p
  transform on target and draft; output distribution = plain sampling, checked by TV).
  `serve` uses it at every temperature (`DS_SPEC=0` disables); `Usage` reports drafts,
  accepted, decode steps.
- **Expert parallelism** (`ep.rs`, `tang-moe/src/nccl.rs`, feature `ep`): experts sharded
  by id across GPUs, NCCL all-reduce of the routed partial sum, one thread per device.
  world=1 bit-exact; 2 ranks on one device KL 0.00. **NCCL itself has never run** (needs a
  ≥2-GPU box).

Measured ceiling (kernel lane, real shapes, 10.5 GB/token): at the measured ~52% bandwidth
efficiency ≈ 240 forwards/s on H200, ≈ 400 on B300; with DSpark's 3–4 tokens a forward that
is the 1000 tok/s class, once the launch floor and the all-reduce are hidden.

Not yet: a real-weights run of any of this (only the naive engine has run on the H200);
Engram rows still decode on the host; the max-width indexer launches at long context; the
EP reduce inside the graph.

## Real weights, 4× H200 (2026-10-27, RunPod CA-MTL-3, NVLink, NCCL 2.25)

`tang-llm deepseek-serve /workspace/ds41 --devices 0,1,2,3`, experts sharded across the four
GPUs (89 GB each resident), 35-token code prompt, thinking off:

| mode | tok/s | forwards | tokens/forward | drafts accepted |
|---|---|---|---|---|
| greedy, DSpark spec (200 tokens) | 3.8 | 46 | 4.3 | 154 / 225 (68%) |
| sampled T=0.6 top-p 0.95 (120 tokens) | 3.4 | 31 | 3.9 | 88 / 150 (59%) |
| greedy, no drafts (`DS_SPEC=0`, 100 tokens) | 1.0 | 99 | 1.0 | — |

Same prompt after the wave-3 merge (graphs under EP, grouped MoE) plus the two host-side fixes
below, same 46 forwards and 154/225 drafts, so the math is unchanged:

| build | greedy 200 | sampled T=0.6 200 |
|---|---|---|
| wave 3 merged, as measured first | 1.1 | 0.9 |
| + Engram tables pre-faulted at open | 17.0 | 12.3 |
| + DSpark Markov head on the device | 24.1 | 18.1 |

The 1.1 was not the graphs or the grouped MoE (the A/B sweep of `DS_GRAPH=0` × `DS_MOE_LOOP=1`
disagreed with itself run to run). `deepseek-profile --direct` showed verify steps of 1050 ms
decaying to 133 ms over a dozen steps with 70% of each step outside every op span, and the
decode step's gap sat before `embed`: the step's Engram rows. They are read per step from the
two 100 GB memory maps, ~300 rows of 256 bytes for a 6-token window, and on the pod's MooseFS
mount a cold row is a ~2 ms round trip (measured: 300 random rows 650 ms cold, 5 ms warm). The
first steps of every fresh prompt were the slow ones, which is also why the bench numbers
drifted with page-cache warmth. `Model::open` now touches every page of both tables (1.5 s
warm, ~5 min cold once per boot; `DS_ENGRAM_PREFAULT=0` skips it). The draft step was then
host-bound (120 ms host, 12 ms GPU): the Markov head added each position's bias with a
256 × 129k GEMV on the CPU; it runs on the device now with one row read back per draw.

Wave 4 (three lanes, merged): grouped MoE on the tensor-core K-split path (6.3 → 0.97 ms per
6-token layer on a 3090, 74% of peak; the loop's `ds_linear_csr` is exact so the grouped-vs-loop
test is now a KL/argmax bar, 0.00 on the fixture), `ds_router` block-parallel (656 → 8.7 µs per
layer; it was one serial thread per token recomputing sqrt-softplus six times) with a one-warp-
per-row GEMV for t ≤ 8 raw linears, the NCCL reduce forked onto a side stream so the shared
expert overlaps it, and one launch fewer per mHC mix. Same prompt: 25.1 greedy / 22.8 sampled
(drafts 240/151, the K-split path rounds differently from the exact loop). Less than the lane
benches add up to, so the next step is the per-request table (`DS_SERVE_PROFILE=1`) on the
real model rather than more guessing.

Also in this round: the Engram tables are copied into anonymous host memory at open
(`St::pin`, one shared copy per process, ~200 GB, 7–10 min cold on the pod's network mount;
the page cache that pre-faulting relied on was evicted within the hour), `DS41_CTX`/`--ctx`
now reach the server's limit (it reported 4096 whatever was asked), prefill runs in
`DS_PREFILL_CHUNK`-token chunks (a 10k-token prompt in one forward ran the device out of
memory), and an EP rank that fails logs and exits instead of leaving its peers spinning in NCCL.

**frog end to end** (`FROG_BASE_URL` at the pod's server through an SSH tunnel, `frog -m
cloud/bull -p "Create hello.py that prints the first 10 Fibonacci numbers, then run it."`,
ctx 32768): correct — set a title, tried Write (blocked under `-p`), wrote the file through a
shell heredoc, ran it, summarized. 18.5 minutes wall: frog's prompt is ~17k tokens and each of
its three turns prefilled it from position 0 (17272, 17483, 17626 prompt tokens; no prefix
reuse) at ~50–60 tok/s. For agent use the prefill side now matters more than decode: prefix
caching across turns (the KV is tiny: window ring + compressed) and a faster prefill (the
indexer scores every chunk position against the whole compressed cache, so long prompts slow
down as they grow).

Per-request tables from the wave-4 server (`DS_SERVE_PROFILE=1`, greedy 200, 4× H200): 42.9
tok/s on the graph path (verify 80 ms, draft 9.1 ms with 7 host syncs, decode-iteration host
3.4 ms), 37.7 on the direct path, whose 6-token verify step (90 ms, 3295 launches) splits as:

| op | ms/step | note |
|---|---|---|
| layer.attn | 53.5 | wo 19.1 (0.48/layer), sparse_attn 18.8 (80 launches), compress 9.8 (indexer 6.6, 48 launches), q 3.0, window_kv 1.8 |
| layer.hc | 16.9 | 123 mixes × 137 µs: launch latency, not math |
| layer.moe | 15.2 | grouped_gu 4.7, down 2.3, shared 3.1, reduce 2.4, router 2.2 |
| head | 1.6 | plus 3.1 MB of logits read back and ~3 ms of host verification per step |

So after wave 4 the step is attention + glue latency, and the launch count (3.3k nodes, each
~2–4 µs of front-end time even in a graph) is a 6–12 ms floor of its own: 1000 tok/s (≈ 4 ms
per iteration) needs the per-layer work fused into a handful of launches, not faster versions
of 80 kernels. Wave 5 lanes: attention block (sparse_attn onto the tiled path — it was running
the serial debug kernel —, wo_a one warp per row, the indexer batched over the window's rows:
1378 → 159, 923 → 381, 254 → 76 µs/layer on a 3090, bit-exact), mHC glue fusion and launch
count, the draft as a captured graph with device-side draws, head on the K-split path with
device-side verification.

Wave 5 merged (attention: sparse_attn onto the tiled path, wo_a one warp per row then K-split
per group, the indexer batched over the window; mHC glue fused to 4–5 launches/layer; the greedy
DSpark draft on the device with one read-back; bf16 head at ~88% of bandwidth and the verify
reading back 24 bytes of argmaxes instead of 3.1 MB of logits; the quadratic stop check fixed):
**57.0 tok/s greedy / 42.4 sampled** on the graph path (from 42.9 / 22.8), same prompt. Direct
verify 90 → 66 ms: attention 53.5 → 23.8 (wo 8.1, compress 5.1, sparse_attn 4.0, q 3.2,
window_kv 2.2), head 1.6 → 0.5, host 3.2 → 0.03 ms. Not improved: `layer.hc` 18.5 ms (the
bit-exact post-mix is load-latency bound: 150 µs per 5120×24 GEMV), `layer.moe` 20.6 (reduce
9.2 of it, which is rank 0 waiting: ranks 1–3 report 43 ms per step against rank 0's 66 and
2323 launches against 3354), and on the graph path the draft takes 20.9 ms with 6.9 syncs
(7.8 ms / 2 syncs on the direct path — the device draft isn't taken after a graph verify).
Prefill is 28–53 tok/s (a 17k-token frog turn is minutes). Wave 6 lanes: draft as a graph +
device sampling, mHC with a relaxed reduction order (DS_EXACT=1 keeps the exact kernels),
MoE grouped kernels tuned for 132 SMs + router/route in one launch + reduce overlap, attention
fusion (one wo launch, q in 2, sparse_attn in 1), and prefill (grouped MoE and GEMM tiles for
t > 8).

## Wave 6 measured, 8× H100 80 GB (2026-10-28, RunPod EUR-NO-2, NVLink NV18, NCCL 2.25)

The CA-MTL-3 H200 host never came back (its GPUs went to another tenant and H200/B200 stock
ran out everywhere), so wave 6 was measured on a fresh 8× H100 pod ($31.92/h, 2 TB host RAM:
Engram pin 160 s, ready in 226 s). Same 35-token prompt, 200 tokens out, `bab5bcfc`:

| mode | tok/s | steps | drafts accepted | graph verify | graph draft |
|---|---|---|---|---|---|
| greedy | **130.7** (×3 runs: 130.4–133.5) | 48 | 152/235 | 26.2 ms | 4.0 ms |
| sampled T=0.6 | 68–70 | 49–50 | 149–153/240–245 | 27.0 ms | 13.2 ms |
| greedy, direct path | 98.9 | 48 | 152/235 | 31.8 ms (1386 launches) | 4.4 ms (151) |
| sampled, direct path | 55.8 | 54 | 147/265 | 32.7 ms | 13.6 ms (152) |

Prefill 455–465 tok/s (was 28–53). The 4× H200 wave-5 figure was 57 greedy / 42 sampled, so
wave 6 plus the extra ranks is 2.3×; the ranks are now balanced (every rank 31.7–32.7 ms per
verify step; the old rank-0 skew was the CA-MTL-3 box). Direct-path verify op table (gpu
32.7 ms): `layer.attn` 12.8 (q 2.8 over 80 launches, sparse_attn 3.5 / 80, wo 3.4, compress
2.7), `layer.moe` 11.8 (router 2.0, route 0.8, grouped_gu 3.5, grouped_down 1.7, reduce 2.3,
shared 2.9), `layer.hc` 4.3 over 123 launches, head 0.56. Draft (greedy) 4.4 ms: named ops
cover 15%, the rest is the 3 MTP layers' glue and the device pick.

Why sampled is half of greedy: (1) `ds_spark_sample` is one 1024-thread block over the 129k
vocab with f64 `exp` per element, ~1.9 ms per draw × 5 draws = +9 ms per draft step; (2) the
sampled verify reads a 129k-f32 logits row back per reached position (one sync each) and runs
`SpecSampler::dist` on the host (f64 softmax + select_nth over the vocab), ~18 ms per step
outside any profiled span. Both are wave-7 lanes (device-side rejection verify; a multi-block
f32 sampler). The greedy path's remaining gap to 1000 tok/s (~4 ms per token-step) is launch
count and per-layer glue: 1386 launches for 26 ms of graph replay on 8 H100s, where the
bytes per rank (experts/8 + dense) put the bandwidth floor near 3 ms.

Wave 7 attn3 lane, first step (cc255a81, 8× H100): q's rope and o's inverse rope fused into the
tiled attention's q load / o store, and the window + top-k index rows in one `dg_win_idx`
launch. Verify step 1386 → 1268 launches, direct gpu 31.58 → 31.51 ms (`layer.attn` 12.75 →
12.59: q 2.76 → 2.59, wo 3.43 → 3.29, sparse_attn 3.51 → 3.68), greedy 125.2/133.5 → 125.6/133.3
tok/s (noise), acceptance 152/235 unchanged. Launch count alone buys little here: the attention
kernels are latency-bound in themselves (~45 µs a sparse_attn span for ~8 MB of keys).
`DS_ATTN_FUSE=1` (soft+pv in one launch) measured no better (3.84 ms) and stays opt-in.

Wave 7 attn3, kernel speed (5e153ea2, df2b224a, 8× H100): split-K decode attention
(`ds_attn_fd` → `ds_attn_fdm` on bf16 mma, one block per row × 64 heads × key split, online
softmax, `ds_attn_fd_comb` merge + o rope; `DS_ATTN_FD=0` / `DS_ATTN_FDM=0` fall back) and
`sinkhorn_warp` unrolled for hc = 4 (bit for bit). Bench t 6 k 646: attention 85 → 34 µs; hc
glue 57.6 → 48.0 µs/layer. In the model: greedy 125.2/133.5 → 138.9/148.5 tok/s, verify direct
31.6 → 29.3 ms, `sparse_attn` 3.51 → 1.99 ms, `layer.hc` 4.33 → 3.76 ms, acceptance 158/225
(text differs from the 152/235 run). Still 25 µs per sparse_attn span: the per-block Q load
(128 KB) and the split partials (t × S × h × d) now dominate, not the math.

Wave 6 as merged (895f89b0 → bab5bcfc): attention fusion (one
grouped K-split launch for wo, 19 → 2 launches, 143 → 54 µs on a 3090; q + window_kv fused,
the compressor/indexer key side batched over the window, ~150 → ~30 attention launches per
compressing layer; every fast linear drops its copy + qdq launches), MoE (redundant quant
passes gone, 7 fewer launches per layer; `DS_MOE_KS_ITEMS`/`DS_MOE_KS_S` grid knobs and the
opt-in fused router `DS_MOE_GATE=1` to sweep on H200), the draft as one captured graph with a
single read-back and device-side sampling that reproduces the host's draws bit for bit (66
replay nodes vs 248 launches; an obs counter bug fixed), and prefill: tiled grouped MoE and a
K-split path for raw linears at t > 8, the tiled attention chunked over rows (tiny fixture
120-token prefill 23 → 13 ms, 3696 → 592 launches; bench at real shapes: wq_b 3.7 → 1.4 ms,
wo_b 4.7 → 2.3 ms at t = 512). The mHC lane landed last (852e2eb6): the mixes' GEMV on the K-split path
with the post-mix folded in and the whole stage in one launch behind a grid barrier — 2 glue
launches per layer at t ≤ 8, 155 → ~50 µs/layer on the 3090; `DS_HC_EXACT=1` (or `DS_EXACT=1`)
keeps the bit-exact kernels, which `gpu_tiny_free_running` now pins (the tiny random fixture
amplifies ulp drift over free-running steps: 6/10 top-1 relaxed, 10/10 exact; the relaxed path
is held by `gpu_tiny_hc_ks_matches_exact`, KL 0 and argmax unchanged teacher-forced). The real
model's draft acceptance rate is the check that matters for that choice. Full GPU suite on
mew at b178b8a2: 12 engine, 2 EP, 14 kernel tests. Still open: the per-rank table (rank 0 was
66 ms vs 43 on the others); a real multi-token GEMM tile for prefill.

Steady state after the host fixes: verify ≈ 130 ms and draft ≈ 15 ms for 4.3 tokens, against a
~4 ms floor. The verify step's remaining cost is the grouped MoE on the non-tensor-core `lin_fast`
path (`moe.router` 28 ms and `moe.reduce` 10 ms per step are the next two line items), then
the launch count (3.3k per verify step).

Output is correct (a working CSV parser with tests). NCCL expert parallelism matched the
single-GPU engine at KL 0.00 on the tiny fixture at 2 and 4 ranks before this run.

So tokens/forward is where it needs to be (DSpark gives 3.8× over plain decoding, and a
6-token verify window costs about the same as a single step); the forward is ~1.0 s against a
~4 ms bandwidth floor. Two causes, both being fixed: (1) `graphable()` is false under EP (the NCCL reduce is
a host callback), so EP steps run the direct path with its syncs; (2) the MoE loop launches
~5 kernels for each of the 384 experts per layer (~80k launches a forward) instead of one
grouped launch over the routing CSR.

## How to profile a step

The step timeline (`flash/deepseek/obs.rs`, [observability.md](observability.md#deepseek-v41-flash-the-step-timeline-flashdeepseekobsrs))
says where a forward's time goes; no guessing.

```bash
# the table: per-forward wall/gpu/host, launches, syncs, bytes; per-op means with % of step
tang-llm deepseek-profile /workspace/ds41 --devices 0,1,2,3 --steps 20 --graph-debug
# the same with decode/verify interiors timed (graphs off for the run)
tang-llm deepseek-profile /workspace/ds41 --devices 0,1,2,3 --steps 20 --direct
# while serving: a line per forward, JSON lines for later
DS_OBS=1 DS_OBS_FILE=/tmp/steps.jsonl tang-llm deepseek-serve /workspace/ds41 --devices 0,1,2,3
```

Read it as: `gpu_ms` vs `wall_ms` (host-bound if they differ), `launches` per forward (a
graph replay is 0 on top of its inputs), `path` (did the decode step actually replay a graph,
see `--graph-debug`), `syncs` (1 on the token path), `expup` (expert uploads after warm-up
mean the LRU or a table rebuild), and per-op `%step` plus `reduce_wait` per rank for EP.

Overhead, RTX 3090, tiny fixture, decode µs/step (best of 10 rounds of ~100 steps):

| | `DS_OBS` unset | `DS_OBS=1` | `DS_OBS=2` (every step) |
|---|---|---|---|
| graph path | 2151 | 2157 (+0.3%) | 2157 (+0.3%; graph interiors aren't recorded) |
| direct path (`DS_OBS_DIRECT=1`) | 2527 | — | 3304 (+31%: ~1100 events on a 554-launch step) |

So `DS_OBS=1` is free; `DS_OBS=2` costs only on the sampled (`DS_OBS_EVERY`, default 50)
direct steps it records, and is for profiling runs. On the tiny decode step the interior's ops
cover 94–99% of the step's device time.

## M0 findings (2026-10-27)

Read from the checkpoint's `config.json`, `model.safetensors.index.json`, shard headers (range
requests) and DeepSeek's reference `inference/model.py`, `engram.py`, `kernel.py`, `convert.py`.

- **Experts are MXFP4 in all but name**: e2m1 values packed two to an `I8` byte along K, low
  nibble first, one `F8_E8M0` scale per row × 32. tang's MXFP4 GEMV (`tang-compute` moe) is the
  starting point, not a new format. 40 × 384 × 3 × 2304 × 5120 / 2 bytes = **272 GB**.
- **Dense weights** are `F8_E4M3` with `F8_E8M0` (power-of-two) scales per 32 × 32 block.
  `wo_a` is dequantized to bf16 at load (convert.py does the same). Norms, `embed`, `head`,
  the router, compressor and indexer projections are bf16; mHC and sinks are f32.
- **No dense layers.** All 40 layers route; 6 of 384 plus one shared expert.
- **Every GEMM quantizes its activation** to fp8 per 32 with power-of-two scales first. The
  window KV and the attention output are fp8 quant-dequantized; compressed KV and index keys
  fp4 (block 16 with e4m3 scales, and block 32 with e8m0, respectively).
- **Engram tables are `F8_E4M3` [384M, 256] + `F8_E8M0` [384M, 8]: ~101 GB each, ~202 GB
  for both.** Each token reads 3 × 8 = 24 rows a layer (256 B each). Host RAM or NVMe with a
  row cache, as PLE.
- **Attention** (model.py `Attention`): every layer attends over a 128-token ring of raw KV
  plus, when `compress_ratio > 0`, `index_topk` = 512 compressed positions. Only
  `kv_source_layers` compress (ratio 2: softmax-gated pooling of 2 tokens; ratio 1: a plain
  projection); the layers after a source read its cache. Only `index_source_layers` run the
  indexer; the others reuse the last source's top-k. Layer 20 also picks 2048 candidate blocks
  of 8 that later indexers are confined to. Ratio > 0 layers use YaRN at θ 160000 for all
  their RoPE; window-only layers plain θ 10000. A learned per-head sink logit joins the
  softmax denominator, and the output has the query's RoPE removed again.
- **mHC** carries 4 copies; each sublayer's Sinkhorn mixes are applied by the *next* sublayer.
- **DSpark** is the 3 `mtp.*` blocks: a block of 5 (the sampled token + 4 noise tokens)
  attends over its own window KV, seeded from target layers 37–39's inputs, plus the whole
  block (unmasked within it). A rank-256 Markov head adds a bigram bias to each draft
  position as it samples left to right; a confidence head scores each draft.
- **safetensors 0.4.5 can't read the headers** (no `F8_E8M0`). The port reads them itself.

Reference and fixture (no DeepSeek code is run): `scripts/deepseek_ref.py` is a torch
transcription of model.py with its kernels' numerics and stable (lowest-index) top-k ties;
`scripts/deepseek_tiny.py` writes a tiny model in the real checkpoint format plus dumps to
`/private/tmp/ds41-tiny`. Prefill-then-decode agrees with one long prefill exactly for 1–3
decode steps; later steps drift because batched and single-row bf16 rounding flips indexer
top-k near-ties (layer 0, which has no indexer, already differs by 3e-4). So parity replays
the fixture's own chunking and is judged by KL and top-1, as for GLM.

## The model, in tang's terms

| Piece | config | tang has | Gap |
|---|---|---|---|
| Residual | 4-stream HC, Sinkhorn, 20 iters, eps 1e-6 | `glm5/mhc.rs` (same shape) | none on CPU; GPU kernel is shared work with GLM |
| Attention | 40 layers, 64 q heads × 512, **1 kv head**, rope 64 of 512, q_lora 1280, o_lora 1024 in 8 groups, sliding window 128, YaRN ×16 over 64K | QSA indexer (Flash-Next), MLA latents (`glm5/mla.rs`) | the 512-wide single-head kernel, the window, o_lora groups, YaRN |
| Compressed attention | `compress_ratios` per layer: 0 / 2 / 1 (layers 0–1 none, 2–19 ratio 2, 20–39 ratio 1), own rope θ 160000 | nothing | **new**: a compressed KV per layer class |
| Cross-layer KV | KV written only at layers 2, 8, 14, 20 (`kv_source_layer_ids`); the rest read a source layer's cache | nothing | **new**: cache owned by a source layer, read by its clients |
| Indexer | 32 heads × 128, top 512 (`index_topk`), index written at layers 2, 8, 14, 20, 24, 28, 32, 36 | QSA's indexer and top-k | sharing across layers |
| Candidate blocks | from layer 20, block 8, top 2048 blocks (`candidate_*`) | glm5 pooled indexer (kpool 4, top 2048) is the closest | **new**: a block-level pre-filter in front of the token indexer |
| Router | 384 experts top-6, `sqrtsoftplus`, `noaux_tc`, scale 1.5, `norm_topk_prob` | `glm5/moe.rs` (sigmoid + noaux_tc, scale 2.5) | the scoring function; one more variant |
| Experts | FF 2304, SwiGLU limit 10, 1 shared, **FP4** weights | Q2_0, Q4X, MXFP4 GEMVs; SwiGLU limit (glm5) | an FP4 expert blob + grouped GEMV; confirm the FP4 format (block scales?) from `quantization_config` |
| Dense weights | FP8, 32×32 block scales, dynamic activations | FP8 safetensors loading (glm5) | block-scaled FP8 GEMV |
| Engram | 2 layers (1, 14), n-grams ≤ 4, 16M hashed vocab, 8 heads × 256, tables of 384,006,168 and 384,016,682 rows | PLE n-gram table with `O_DIRECT` reads and a row cache (Flash-Next) | the hashing, the two-table layout, and **sizing**: the rows alone are ~100 GB a table at bf16, so they live on host/NVMe like PLE does |
| MTP | 3 next-token layers | `flash/mtp.rs` chain (one layer, Qwen) | a 3-layer chain |
| DSpark | drafter: 128 experts top-3, markov rank 256, block 5, noise token 128799, reads target layers 37–39 | verify windows in graphs, `DraftPolicy` | **new**: a second small MoE model as the drafter |
| Vocab | 129,280 | — | tokenizer only |
| Vision | 32-layer encoder, ≤ 1024 image tokens | `qwen3vl.rs` pattern | not in scope for bull |

No `first_k_dense_replace` in the config, so assume all 40 layers route to experts until the
modelling code says otherwise. `compress_ratios` has 43 entries: 40 layers plus the 3 MTP layers.

### Bytes

- Experts: 40 × 384 × 3 × (5120 × 2304) ≈ 544B params. At FP4 ≈ **272 GB**; at 2-bit ≈ 136 GB.
- Active per token: 6 + 1 experts × 40 layers ≈ 9.9B expert params + dense ≈ **~13B** (so the
  "16B" in press copy counts something else; measure it in M0).
- Engram: two tables of ~384M rows. Row width is 8 × 256 at whatever dtype the shards use; read
  the shard index in M0 before believing any number.

The expert bytes decide the hardware. 272 GB does not fit one H200 (141 GB) or one B200
(192 GB) with KV and Engram cache. Either experts stream from host (tang-moe) or they shard
across GPUs. Streaming bounds decode near 6 × 3 × 11.8M × 0.5 B ≈ 35 MB a layer, ~1.4 GB a
token, so ≈ 25–40 tok/s on PCIe 5 before cache hits. The 800 tok/s class needs every expert in
VRAM: **4× B200/B300 or 8× H200, with expert parallelism tang does not have.** That is the
largest item in this plan and the reason the old 6–10 week figure was low.

## Milestones

Each one is a branch with a measured result in the commit body, as tang does now. Weeks are for
one engineer who knows the Flash and glm5 code; M1 and the kernels in M2 can run in parallel.

**M0 — truth (days)**
- Pull `config.json`, the modelling file, `quantization_config` and the safetensors index.
  Write down the FP4 format, the Engram row dtype, and whether any layer is dense.
- Tiny random-init `deepseek_v41` checkpoint via transformers, like `scripts/glm5next_tiny.py`.
  HF fp32 golden logits for a fixed prompt set, per-block dumps for `check_logits.py`.
- Pick the hardware target and price it: 4× B200 vs 8× H200 on RunPod.
- Exit: a sizing table with measured bytes; a tiny fixture under `tests/fixtures/deepseek/`.

**M1 — CPU f32 reference forward (2 weeks)**
- `flash/deepseek/` on the `glm5/` layout: `gguf`/`safetensors` geometry, `forward.rs`, one
  module per owner.
- Reuse: `mhc.rs` as is; `moe.rs` with a `sqrtsoftplus` scorer; glm5's FP8 loader; SwiGLU limit.
- New: the single-kv-head attention with window and o_lora groups; compressed KV with its own
  rope; cross-layer KV and index sharing; candidate blocks; Engram hashing and lookup against
  the real tables (mmap, no cache yet); YaRN.
- MTP layers and DSpark off. Vision off.
- Exit: tiny-model logits match HF fp32 at the existing KL bar, block by block; on the real
  model, the first two layers (no compression, no sharing) match HF on a 512-token prompt in RAM.

**M2 — one-GPU decode with experts offloaded (3 weeks)**
- Engine on the `coder.rs`/`glm5/engine.rs` pattern, CUDA first.
- Kernels: FP4 expert grouped GEMV (sibling of the Q2_0 and MXFP4 ones), block-scaled FP8
  GEMV for dense, the 512-wide attention with window + indexed sparse + compressed reads,
  Sinkhorn mHC on GPU (shared with GLM's speed plan), `sqrtsoftplus` + top-k.
- Engram through the PLE path: `O_DIRECT` row reads, row cache, prefetch off the n-gram of the
  tokens already known.
- Experts in a tang-moe pinned arena with the VRAM cache; proving ground is a forced small
  cache so every number has an all-VRAM twin, as hybrid-moe M5 did.
- Exit: parity with M1 (KL ≤ 0.03 on the prompt set, top-1 flips ≤ 2%); a first tok/s on one
  H200, expected 10–40; `tang-verify` shadow checks on every new kernel.

**M3 — expert parallelism (3–4 weeks; the new infrastructure)**
- Fill the `AllReduce` slot in tang-compute (`docs/extensions.md`) with NCCL through cudarc.
- EP: experts sharded by id across ranks; dispatch and combine as all-to-all; dense weights,
  attention and Engram replicated (they are small). One process, one thread per GPU, one
  CUDA graph per rank per T.
- Rank-0 owns sampling and the KV; the others mirror block ids (the design `ext.rs` already
  assumes).
- Exit: every expert in VRAM on the M0 box; T=1 decode ≥ 100 tok/s batch 1; parity unchanged.

**M4 — drafting (2 weeks)**
- 3-layer MTP chain in `mtp.rs`; `DraftPolicy` picks MTP, suffix or DSpark.
- DSpark: load its 128-expert MoE and Markov table, feed it target layers 37–39, block of 5,
  verify 6 a window. It is a second small engine; reuse the EP dispatch so its experts shard too.
- Exit: `speculate.rs` passes with drafts forced wrong (identical output); ≥ 3.5 tokens a
  window on the code prompt; decode ≥ 2.5× T=1.

**M5 — low-latency serve and bull (2 weeks)**
- CPU between steps: one graph launch per step, DP sync off the hot path, KV admission capped,
  as vLLM's DeepSeek post lists. Measure % of speed of light per step as tang does.
- `serve_glue.rs` for the OpenAI endpoint; `bull` in frog's `STATIC_CLOUD`
  (`crates/frog-engine/src/catalog.rs`), fallback as today.
- Exit: single-stream P90 tok/s published in `speed-log.md` with the SOL fraction;
  `frog -m cloud/bull` runs the frog-bench suite end to end.

**M6 — prefill and long context (ongoing)**
- Prefill kernels for compressed attention and the candidate pre-filter; bounded replay for
  cached prefixes; YaRN past 64K to the 1M the config declares; int8 KV above 8K.
- Exit: 128K prefill tok/s and decode at 128K within 10% of decode at 4K.

Sum: **~13–15 weeks** for one engineer, ~10 with two (one on the reference and formats, one on
kernels and EP from M0). Expert parallelism is a third of it and is the part that also unlocks
GLM-5.3-Flash at speed, so it is not DeepSeek-only work.

## Risks

- **The FP4 format.** If it is a DeepSeek-specific block layout rather than NVFP4/MXFP4, the
  GEMV is new and the requant path (`flash/requant.rs`) needs a target. Settle it in M0.
- **Compressed attention semantics** are only in the modelling code. Ratio 2 vs 1 vs 0 and how
  compressed and raw KV combine decide the kernel; read before designing.
- **Engram size.** Two ~100 GB tables on host means the row cache's hit rate sets prefill speed.
  PLE's numbers are for a far smaller table.
- **EP is new.** NCCL through cudarc, all-to-all in a graph, and multi-rank determinism are all
  first-time code in tang. Budget the slip here, not in the model work.
- **Hopper vs Blackwell.** FP4 tensor cores exist only on Blackwell; batch-1 GEMV dequant is
  fine on H200, but prefill and any batching want B200.
- **Press numbers.** 1500 tok/s is a vendor demo on a Tetris prompt with thinking off. The
  reproducible bar is vLLM's 826 P90 on 4× B300 TP4 with DSpark. Target that.
