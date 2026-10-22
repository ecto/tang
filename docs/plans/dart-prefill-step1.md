# dart prefill, step 1: state of the candidates and the path to launch (Oct 30)

Scope: cold prefill for Qwen3-Coder-Next (UD-Q4_K_XL) on tang Flash, served on RunPod H200s by
frog cloud. Problem (zest PR #41, `docs/plans/launch-capacity.md` in the zest launch-capacity
worktree): cold prefill ~1.85k tok/s, serialized across requests — 4 concurrent cold 58k prompts
give TTFT 32/63/93/123 s; the 4th crosses Cloudflare's 100 s edge timeout. A lone 117k prompt
takes 64 s. Cached follow-ups (0.3–0.7 s) are fine. The launch hazard is several new long
sessions arriving together. All commands run read-only in
`/Users/cam/Developer/tang/.frog/worktrees/dart-prefill` (branch `frog/dart-prefill`) unless
noted; zest paths are read-only.

## a. PR #128 — "qwen3next: tiled wide-prefill attention" (draft)

- State: OPEN, draft, author ecto, branch `claude/qwen3next-attn-tile`, base main, created
  2026-10-21 (`gh pr view 128 --json ...`). Head commit `caba8a35` ("tiled wide-prefill
  attention (cx_attn_tile; TANG_CODER_ATTN_TILE=0 for split-K)"), plus `b65f45cc` (check gate)
  and `9f74ff9b`/`5c13ba1d` (bench script) on local branch `frog/tiled-prefill`
  (`git log --all --oneline --grep=attn-tile`, `git log --oneline frog/tiled-prefill`).
- GPU-verified on H200: **no.** The PR body says gates ran on a 3090 with a tiny real-width GGUF
  (`TANG_CODER_ATTN_CHECK=1`, mean |diff| ~1e-7) and "Pending: H200 8k/30k/54k timing + KL gate".
  No H200 numbers exist anywhere for the tiled kernel.
- Worse: it lost. `docs/bench/fa2-prefill.md:5-7` (on `claude/prefill-combined`): the tiled kernel
  "came out 1-4% slower" than split-K. It stays off tensor cores; FA2 (candidate b) supersedes it.
- Tests: `scripts/bench_prefill_tile.sh` (on `frog/tiled-prefill`) is a ready H200 A/B + gate
  harness, but it benches a kernel we don't want.
- Merge conflicts vs main: none textual (`git merge-tree --write-tree main frog/tiled-prefill`
  exit 0), but it conflicts with `claude/prefill-combined` (exit 1) — same files, divergent edits.
- What's left if pursued anyway: the H200 run the PR itself lists as pending. Recommendation:
  **close #128 as superseded** by FA2 rather than spend a bench slot on it.

## b. `claude/prefill-combined` — FA2 wide-prefill attention + chunked GDN (f112b094)

Merge commit `f112b094` (2026-10-21, `git log --oneline main..claude/prefill-combined`) combining:
- `3054b910` "FA2-style tensor-core wide-prefill attention (TANG_CODER_ATTN=fa2)" — new
  `cx_attn_fa2` kernel (mma.sync bf16, GQA-packed Br=64/Bc=32, cp.async double-buffering,
  flash-decoding key splits), `crates/tang-llm/src/flash/fa2.rs`, ~230 lines + kernel source in
  `coder_kernels.rs`.
- `1d095d92` "chunked GDN prefill behind TANG_CODER_GDN=chunked" — `gdn_chunked.rs` CPU
  reference + `fl_gdn_chunk_step_seg` CUDA kernel (UT/WY form, chunk 16), drop-in for the
  per-token GDN kernel on wide windows.

FA2 is **GPU-verified on H200** — but on this worktree's HEAD, not on main:
`45a36a28` ("FA2 wide-prefill attention on by default", `git log --oneline -3` here) says:
"H200, real UD-Q4_K_XL: cold wide prefill 36.9 → 22.3 s at 30k, 87.5 → 40.7 s at 54k; greedy
identical over 64 steps, KL ≤ 3.5e-5 vs narrow (zest bench/2026-10-22)". That is 1.65× at 30k,
**2.15× at 54k**, gates passed. FA2 is not in main
(`git merge-base --is-ancestor 3054b910 main` → false).

Chunked GDN is **not GPU-verified**: `docs/bench/gdn-chunked-prefill.md:3-4` (on the branch)
says "CUDA kernel not yet compiled or run (written on a Mac with no nvcc)". I verified the CPU
side compiles and its unit tests pass on the branch in a throwaway worktree
(`git worktree add --detach /private/tmp/prefill-combined-wt f112b094` then
`CARGO_TARGET_DIR=/private/tmp/dart-prefill-target2 cargo test -p tang-llm --lib gdn_chunked` →
4 passed: chunked matches recurrent at d=16/128, strong decay, and decode continues from the
chunked state). On this HEAD, all 122 lib tests pass (`cargo test -p tang-llm --lib`).

Merge conflicts vs main: none — `git merge-tree --write-tree main claude/prefill-combined`
exits 0 with a clean tree. Note this branch's FA2 is not the "default on" variant; HEAD
(`45a36a28`) is the newer, H200-verified default-on state, so the merge should go FA2-from-HEAD
+ GDN-chunked-from-combined, not the combined branch verbatim.

What's left: (1) compile the GDN chunked kernel on a GPU at all (first nvcc/NVRTC compile —
register/smem pressure unmeasured); (2) H200 A/B + KL gate for GDN; (3) the caveats below about
bench-config vs serving-config.

**Config caveat that sizes the win.** The verified 87.5→40.7 s is from `flash-wide-check`
(256-row wide windows, `TANG_CODER_WIDE=256` per `docs/bench/fa2-prefill.md:117`). The serving
node runs 2048-row windows (`TANG_SERVE_WIDE` default 2048, `flash/serve.rs:712-714`) and
measures 58k/32 s ≈ 1.85k tok/s — much faster than the check's 54k/87.5 s ≈ 617 tok/s baseline.
So the *relative* gain at the serving config is unmeasured; attention's share of a 2048-row
window's cost may be smaller than the ~65% measured at 256 rows. Expect 1.8–2.4× on the serving
config (attention stays per-row-K/V-reread in split-K, so it should still dominate), i.e.
1.85k → **3.3–4.5k tok/s from FA2 alone**. The bench in §5 must measure the serving config.

## c. Where prefill serialization happens today

Frog cloud serves via the Flash path: `entrypoint.sh:50` runs `tang-llm serve <gguf> --batch 8`
→ `flash/serve_batch.rs`. Three layers:

1. **One worker thread owns the GPU** — `server.rs:12` ("The model lives on one worker thread"),
   loop at `server.rs:537-568`; everything is serial by construction.
2. **Shortest-remaining-first lets one stream monopolize prefill** —
   `flash/serve_batch.rs:43-73` (`prefill_plan_with`): waiting streams are sorted by remaining
   tokens and served from a single `WIDE_BUDGET` (2048 tok) pool. A stream that has already been
   served stays shortest, so it keeps getting the whole budget until it finishes; the next only
   starts after. This is exactly the observed 32/63/93/123 s staircase. Called at
   `serve_batch.rs:399-409`.
3. **Chunks run back-to-back inside the step** — `serve_batch.rs:410-453`: `for (i, n) in plan`
   calls `prefill_slot_wide`/`prefill_slot` serially; with nothing decoding there is no
   `prefill_debt` throttle (`serve_batch.rs:380-388`), so a cold prompt occupies the GPU in
   2048-token wide windows (`TANG_SERVE_WIDE`, `serve.rs:712-777`) until done.

(The generic engine path has proper chunked prefill with a lane order and per-step time budget —
`engine.rs:1372-1454`, `PREFILL_CHUNK` 512 / `STEP_BUDGET` 50 ms at `engine.rs:18-25`,
rules in `batch.rs:26-44` — but the Flash serving path frog cloud uses does not go through it.)

Fair-share (round-robin or equal budget split) in `prefill_plan_with` would make the 4 TTFTs
equal instead of staggered — but equal at ~4×31 s ≈ 125 s each, i.e. all four time out. So
**fairness alone is not the fix; kernel speed is the primary lever**, fairness second (it bounds
the worst case and stops one 117k prompt from adding 64 s to everyone behind it).

## d. Target

**≥5k tok/s cold prefill** (per-stream, and near-that aggregate under the bench). Justification:
a lone 58k prompt then costs ~12 s; 4 concurrent 58k prompts cost ≤ 4×58k/5k ≈ **46 s worst-case
TTFT** — under the 100 s Cloudflare edge timeout with >2× headroom, even fully serialized. The
alternative (≤30 s TTFT for 4×58k) implies ~7.7k tok/s aggregate, which needs the FA3-class work
(≥1 wk more, per `docs/research/prefill-survey-2026-10-21.md`) — not reachable before Oct 30.
5k is reachable: FA2 alone should give 3.3–4.5k (§b), chunked GDN adds its stage (1.3–2× on
`gdn_conv_step`, likely the next-biggest post-FA2 share since GDN is 36 of 48 layers running a
per-token serial kernel), landing ~4–5.5k. The §5 bench decides; if GDN stalls, FA2 + the
fairness tweak alone still brings the 4×58k worst case from 123 s to ~52–70 s, under the timeout.

## Fastest path to a measurable win before Oct 30 (ordered)

1. **Land FA2 in main** — it is already H200-verified with gates passed (`45a36a28`, on this
   branch's HEAD). Open PR from `frog/dart-prefill` (or cherry-pick `3054b910`+`45a36a28`),
   review, merge. Zero new GPU time strictly required; the serving-config re-measure in step 3
   doubles as its confirmation. Closes the gap between "verified on a bench" and "deployed".
2. **One H200 bench session** (§5): A/B FA2 at the *serving* config, first compile + A/B chunked
   GDN, then the 4×58k concurrency run on the best build. ~2 pod-hours, ~$7-15.
3. **If GDN passes its gates, default it on** (`TANG_CODER_GDN=chunked` → default) on the same
   evidence standard as FA2 (greedy identical 64 steps, teacher-forced KL ≤ 1e-3, clear win at
   30k/54k, no loss at 8k). If it fails to compile or loses, ship FA2 alone.
4. **Fair-share prefill in `prefill_plan_with`** (`serve_batch.rs:49`): split `WIDE_BUDGET`
   across waiting streams round-robin instead of shortest-remaining-takes-all. Small, pure,
   testable change (the file's existing unit tests at `serve_batch.rs:660-675` show the pattern).
   Bounds the worst-case TTFT and stops a 117k prompt from blocking the queue for 64 s.
5. **Deploy** via the zest node image (`TANG_REF` → new main; §6) and have the frog cloud
   manager re-run their live TTFT check (4 concurrent 58k cold prompts through the router).

## Expected gains, with reasoning

| step | cold prefill (58k) | reasoning |
|---|---|---|
| today | 31 s ≈ 1.85k tok/s | live node, zest PR #41 |
| + FA2 (verified kernel, unverified at serving config) | ~13–17 s ≈ **3.3–4.5k tok/s** | 2.15× measured at 54k/256-row windows (`45a36a28`); discounted to 1.8–2.4× because the 2048-row serving windows may already amortize some K/V traffic |
| + chunked GDN (if gates pass) | ~10–14 s ≈ **4–5.5k tok/s** | 1.3–2× on the GDN stage (`gdn-chunked-prefill.md` "honest" estimate); GDN is 36 layers of per-token serial kernels, the dominant non-attention cost after FA2 |
| + fair-share scheduling | same tok/s, worst 4×58k TTFT = total/1 | TTFT staircase 32/63/93/123 → all ≈ total time; with the kernel wins that's ≤ ~46-56 s each |

## 5. The exact GPU bench needed

- **Hardware:** 1× H200 (RunPod, the muni account, `H200_SXM`), ~2 hours, **~$7.20**
  (2 × $3.59/hr). Use a bench pod, never a serving node (zest runbook:114-116).
- **Build:** from the merged branch (FA2 default) plus `claude/prefill-combined` for GDN:
  `cargo build --release -p tang-llm --no-default-features --features cuda`
  (this compile is itself the GDN kernel's first-ever nvcc/NVRTC test — budget 15-20 min).
- **Run A (FA2 at serving config):** `tang-llm flash-wide-check` with `TANG_SERVE_WIDE=2048`-class
  windows, `TANG_WIDE_CTXS=8000,30000,54000`, `TANG_CODER_ATTN=fa2` vs `splitk`,
  `TANG_CODER_PROF=1`, gates = greedy identical 64 steps + teacher-forced KL ≤ 1e-3 (the exact
  commands are in `docs/bench/fa2-prefill.md:117-133`; `scripts/bench_prefill_tile.sh` on
  `frog/tiled-prefill` is a template for the A/B wrapper).
- **Run B (GDN chunked):** same check, `TANG_CODER_GDN=chunked` vs default, same gates; record
  the `gdn_conv_step` share from `TANG_CODER_PROF=1` before/after.
- **Run C (the launch hazard, end-to-end):** on the best build, start
  `tang-llm serve <UD-Q4_K_XL> --ctx 131072 --kv-slots 8 --batch 8` (entrypoint config,
  `TANG_SERVE_WIDE` default) and fire **4 concurrent cold 58k-token prompts** (plus one 117k
  alone) via zest-harness or curl through the local port; record per-request TTFT. Pass bar:
  worst TTFT < 50 s. This is the same shape as the frog cloud manager's PR #41 bench, so the
  before/after is directly comparable.

## 6. Deploy path (zest node, read-only) and what frog cloud needs from us

The node image builds tang from git at image-build time:
`~/Developer/zest/deploy/node/Dockerfile` clones `github.com/ecto/tang` and checks out
`--build-arg TANG_REF` (default `main`), builds `tang-llm --features cuda` (CUDA 12.8.1, Rust
1.94.0), and ships `tang-llm` + `zest-router` + `zest-harness` + `cloudflared`. One image covers
sm_80–sm_100 because kernels are NVRTC-at-load. The pod runs
`tang-llm serve <gguf> --ctx $CTX --batch 8 --kv-ttl 3600 ...` (`deploy/node/entrypoint.sh:50-56`).
So deployment is: **merge to main (or give them a ref) → they rebuild the image with
`TANG_REF=<ref>` → new pod**. No env changes needed for FA2 (default-on); `TANG_CODER_GDN=chunked`
stays opt-in until its gates pass, then we flip the default and they rebuild again.

What the frog cloud manager needs from us: (1) a merged main (or a branch ref) with FA2 + GDN
verdict; (2) the §5 bench numbers, especially the 4×58k TTFT before/after so their capacity doc
(`launch-capacity.md`:16-24, the 1,050–1,530 tok/s figures and the per-task cost formula) can be
updated; (3) confirmation of which env flags the new default changes; (4) the KL gate evidence for
their records. Their doc already tracks the gap ("neither is deployed", launch-capacity.md:43-44).

## Risks

- **FA2's serving-config gain is extrapolated, not measured** — the 2.15× is from 256-row check
  windows; 2048-row serving windows may dilute it (§b caveat). Mitigated by bench Run A being
  step 2 of the plan, before any default flips beyond FA2's already-verified one.
- **GDN chunked has never been compiled.** NVRTC register/smem pressure could spill or fail; the
  doc's own fallback (`gdn-chunked-prefill.md`) is a correctness-first C=16 kernel with modest
  expected gain. If it loses, FA2 alone still clears the timeout for 4×58k (52–70 s worst case).
- **Fair-share prefill changes tail latency for the lone-long-prompt case** — a 117k prompt no
  longer finishes first if shorter prompts keep arriving; acceptable at launch scale, and
  `prefill_debt` still protects decoding streams.
- **Merge hygiene:** #128 (tiled) conflicts with `claude/prefill-combined`; close #128 first so
  nobody re benches a superseded kernel. `prefill-combined` vs main is conflict-free today
  (`git merge-tree` exit 0) but both touch `flash/coder.rs` — merge FA2 from HEAD (default-on,
  H200-verified) rather than the combined branch's older FA2.
- **Time:** FA2 merge + one bench session + GDN verdict + fairness tweak fits before Oct 30;
  the FA3/wgmma follow-up (another ~1.5×) does not — don't let it pull focus.
- **Cloudflare's 20 s edge-router probe and 60 s queue** (`launch-capacity.md`) mean TTFT must be
  measured end-to-end through the router, not just engine prefill time; bench Run C covers this.

## Facts and the commands that showed them

- PR #128 state/body: `gh pr view 128 --json state,title,isDraft,body`; head commit
  `caba8a35` via `git log --all --oneline --grep=attn-tile`; tiled "1-4% slower":
  `git show claude/prefill-combined:docs/bench/fa2-prefill.md` (line 5-7).
- FA2 H200 verification: `git log --oneline -3` on this worktree → `45a36a28` message; not in
  main: `git merge-base --is-ancestor 3054b910 main` → false.
- GDN chunked unverified on GPU: `git show claude/prefill-combined:docs/bench/gdn-chunked-prefill.md`;
  CPU tests pass: `git worktree add --detach /private/tmp/prefill-combined-wt f112b094` +
  `cargo test -p tang-llm --lib gdn_chunked` (4 passed). HEAD tests: `cargo test -p tang-llm --lib`
  (122 passed). Merge cleanliness: `git merge-tree --write-tree main claude/prefill-combined`
  (exit 0), `... main frog/tiled-prefill` (exit 0), `... claude/prefill-combined frog/tiled-prefill`
  (exit 1).
- Serialization: `crates/tang-llm/src/flash/serve_batch.rs:380-453` (plan + serial chunks),
  `serve_batch.rs:43-73` (`prefill_plan_with`), `crates/tang-llm/src/server.rs:12,537-568`
  (single worker); generic-engine chunking exists at `engine.rs:1372-1454` but is not the Flash
  serving path (`entrypoint.sh:50` → `flash/serve_batch.rs` via `--batch`).
- Live numbers: zest PR #41 (`gh pr view 41` in zest) and
  `~/Developer/zest/.frog/worktrees/launch-capacity/docs/plans/launch-capacity.md` (prefill
  figures lines 16-24, gaps §6); prefill technique survey:
  `~/Developer/zest/docs/research/prefill-survey-2026-10-21.md`.
- Deploy: `~/Developer/zest/deploy/node/Dockerfile` (TANG_REF build), `entrypoint.sh:50-56`
  (serve flags), `:712-777` of `flash/serve.rs` (TANG_SERVE_WIDE default 2048).
