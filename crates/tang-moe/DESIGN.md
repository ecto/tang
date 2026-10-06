# tang-moe: does "the GPU reads its own misses" hold on mew?

Status: measured 2026-10-04 (numbers and commands in [BENCH.md](BENCH.md)). The question is
whether Flash-Next's expert misses can be served with zero host in the loop: hot experts in
VRAM slots, misses read by the expert kernel directly from pinned, device-mapped RAM, the whole
window one CUDA graph.

## Short answer

**No, not on mew today, and probably not at 200 tok/s even after the fix that needs root.**
Serve misses on the CPU, as Strata does. Keep the pointer table, so the GPU can take a "PCIe
share" of the misses later if passthrough makes PCIe fast.

## What the measurements say

The target load: ~7 % misses, ≈ 34 per token, ≈ 110 per 3–4-token verify window, or about 2–3
per layer. That is ≈ 152 MB of 1.38 MB blobs per window.

| way to serve 110 misses per window | measured or projected | per-window cost |
|---|---|---|
| GPU reads them in place, IOMMU on (mew today) | measured, (a′): ~340 µs per miss | **~37 ms** |
| Copy engine to VRAM staging, then compute, IOMMU on | measured, (b): 3.3 GB/s scattered | **~46 ms** |
| GPU reads in place, IOMMU passthrough | *projected*: 20–25 GB/s, ~2.5 misses per layer read serially | ~6–8 ms, and it stalls SMs |
| CPU experts at RAM speed (8 P-cores) | measured DRAM 66–70 GB/s; doorbell 4 µs round trip | ~2.3 ms of reads, mostly overlapped with GPU hits; ~0.2 ms of handoffs |

Why GPU-only loses even in the projection:

1. **Misses can't be batched.** Each layer depends on the previous one, so a window is 48
   small miss reads of ~3 MB each. Each is bound by PCIe latency and bandwidth, and nothing
   else in the layer is long enough to hide it. The all-hit expert pass for a whole window is
   only 2.4 ms (1,152 blobs at 660 GB/s).
2. **In-place mapped reads stall SMs.** A kernel waiting on PCIe loads slowed a concurrent VRAM
   expert kernel 4× (b′). Copy-engine traffic cost the same kernel only 14 %. So host reads
   from inside kernels serialise with the window instead of overlapping it.
3. **Budget.** The plan's window is ~13 ms: ~10 ms of dense GPU work plus experts, for ~3.2
   tokens. Adding 6–8 ms of serial miss reads gives ~19–21 ms, about 150–170 tok/s. With the
   IOMMU on it is ~50 ms, ~65 tok/s.

The CPU path wins because DRAM is 66–70 GB/s and the CPU can work while the GPU computes the
layer's hits and shared expert. The handoff is cheap: 3.96 µs median round trip through a
mapped flag (p99 4.3 µs), ~0.2 ms over 48 layers.

## The IOMMU finding (needs root to fix)

Every PCIe path on mew runs at 3–7.5 GB/s on a Gen4 x16 link that should give ~25. The link
trains at 16 GT/s x16 at both ends. The GPU's IOMMU group is `DMA-FQ` (Ubuntu turns VT-d on by
default). The rate depends on host page size (4 KiB 2.8 GB/s, 2 MiB 7.4 GB/s) and on how much
memory is touched (16 GiB scattered 3.3 GB/s, 1 GiB 7.3 GB/s). That is IOTLB behaviour, not
link behaviour. It is a strong diagnosis, but it is **not verified**: confirming it needs a
reboot with passthrough. It also hurts Strata's own "PCIe share" and the cache's adaptive swaps
on this box.

## Recommended v1

- **Slots.** Measured on mew with 1.67 GB already held by other clients: free is 23.26 GB.
  After 3.5 GB dense, 1.0 GB MTP and 1.0 GB KV/state, sizing leaves **12,341 slots (17.06 GB,
  50.2 % of the 24,576 experts)** with a 0.7 GB reserve, re-measured after touch.
  `ExpertCache::new` does this sizing. Strata's curve puts the hit rate near 0.93 at this
  size.
- **Host arena.** Resident mode, holding the complement of VRAM: 12,235 blobs ≈ 16.9 GB,
  pinned with one `cuMemHostRegister` of THP memory. 20 GiB registered fine without root, and
  memlock doesn't apply. Register while RAM is free: 1.8 s and 100 % huge pages then, against
  12 s and 34 % while the kernel is reclaiming page cache.
- **Misses.** CPU experts on the P-cores; per-layer doorbell in mapped memory. The GPU computes
  the hits through the pointer table, and the table's host entries are unused at first.
- **Cache adaptation.** Decayed LFU as built (every 4 windows, ≤96 swaps). Copies run on the
  copy engine on a side stream, and all table writes happen on the main stream between windows.
  96 swaps is ~40 ms of copy at today's 3.3 GB/s, overlapped with decode at ~14 % cost to
  concurrent kernels. After passthrough, expect about 4× less.
- **Graphs.** One graph per window size. A dependent kernel inside a graph costs ~1 µs, against
  1.5–1.8 µs eager and 0.15 µs fused, so 15–40 kernels per layer cost 0.7–2 ms per window. Fuse
  toward ≤10 kernels per layer. Per-step scalars live in device memory and must be written
  stream-ordered (`cuMemcpyHtoDAsync` on the graph's stream, or into mapped memory). A sync
  `cuMemcpyHtoD` races a non-blocking stream; measured wrong.
- **PCIe share, later.** Once passthrough is in, measure `tang-moe-bench all` again. If in-place
  reads reach ≥20 GB/s, let the GPU take the last few misses of a layer (Strata's 20–55 % tail)
  when the CPU is the critical path. The pointer table already points misses at mapped memory.

## What would change the answer

- **Passthrough measures ≥20 GB/s in-place *and* misses fall to ≲3 %** (≈50 per window, about 1
  per layer, ~3 ms): then GPU-only is competitive and simpler. More VRAM, smaller cold experts
  (lower-bit blobs) or a better routing profile all push misses down.
- **One-layer-ahead miss prediction good enough to prefetch with the copy engine.** Copies
  overlap compute at 14 % cost, so predicted misses could land before they are needed. This
  needs PCIe at full speed: 3.3 GB/s can't move 152 MB in a 13 ms window.
- **CPU experts slower than RAM speed in practice** (compute-bound kernels, E-core noise, host
  hiccups during device spins) would narrow the CPU path's lead. M5 measures this.

## What needs root on mew

1. **IOMMU passthrough (the important one).** Edit `/etc/default/grub` to set
   `GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt"`, or `intel_iommu=off` to disable it entirely. Then
   `sudo update-grub && sudo reboot`. Check `cat /sys/kernel/iommu_groups/17/type` reads
   `identity`, and re-run `tang-moe-bench pcie` and `all`. The reboot takes frigate and home
   services down briefly.
2. **Hugetlb pages (optional).** THP already reaches 99–100 % when memory is free. To make it
   guaranteed: `sudo sysctl -w vm.nr_hugepages=9000` (17.6 GiB of 2 MiB pages), persisted in
   `/etc/sysctl.d/90-hugepages.conf`. `HostArena` tries `MAP_HUGETLB` first.
3. **Memlock: not needed** for `cuMemHostRegister`, verified to 20 GiB with an 8 GiB limit.
   Only an `mlock` fallback would need `LimitMEMLOCK=infinity` in the service unit.

# The CPU miss path, as built

Measured in [BENCH.md](BENCH.md), "CPU miss path".

## Per-window cost at the realistic load

At 93 % hits there are ~110 misses per 3–4-token window, ≈ 2.3 distinct missed experts per
layer, mostly one token each.

| | measured |
|---|---|
| CPU time for one layer's misses (8 P-cores, AVX-VNNI) | 31 / 55 / 79 µs for 1 / 2 / 3 misses (45–53 GB/s) |
| CPU time per window (synthetic window, M = 2–3) | 2.75–4.0 ms |
| ...of which exposed (not hidden under the GPU's hits) | 0.9–2.2 ms |
| Doorbell handoff (publish, two waits, row copy) | 0.82 ms per window, 17 µs per layer |
| **Added to the GPU's window** | **≈ 1.7–3.0 ms** |

Only the GPU work that runs after routing (hits and shared expert, ~35–50 µs per layer) can
hide CPU time, so the exposed share climbs fast above 2 misses per layer. Ways to shrink it
(none built):

- Let the GPU build the plan itself from `ResidentCache::device_addrs()`, using the kernel
  track's `moe_plan_into`, and publish only `MISSING`. That drops the `FLAG_A` round trip
  (~6–8 µs per layer, ~0.35 ms per window).
- Raise the hit rate.
- Start phase A on the first missed expert before the plan is published.

## The engine interface

One `doorbell::Mailbox`: 233,472 u32 words (912 KiB) of registered, device-mapped host
memory, reused for every layer of every window. Word offsets are `doorbell::Mb`; byte offset
= 4 × word.

| words | name | written by | contents |
|---|---|---|---|
| 0 | `SEQ` | GPU, last | `win · 64 + layer + 1` |
| 64 | `FLAG_A` | host | `SEQ` value once `PLAN` is written |
| 128 | `FLAG_B` | host | `SEQ` value once `ROWS` and `CPU_ROWS` are written |
| 192 | `HDR_T` | GPU | tokens in this window, `t` (1..8) |
| 193 | `HDR_LAYER` | GPU | layer (informational) |
| 256.. | `IDS` | GPU | router ids `[t][TOPK]`, u32, token major, rank minor |
| 512.. | `XQ` | GPU | the layer's int8 activations, `QAct { m: t, k: 2560 }` words, the exact buffer `moe_grouped_into` reads (contract below) |
| 6912.. | `PLAN` | host | a `MoePlan` (`MoePlan::WORDS` = 533 words) |
| 7456.. | `CPU_ROWS` | host | `[n, dst_0 .. dst_{n-1}]`: the `parts` rows the host filled |
| 8192.. | `ROWS` | host | `[88][2560]` f32; row `dst` is `parts` row `dst` |

The flag regions are 256 bytes apart from everything the GPU writes.

Per layer, inside the window's graph:

1. After `router_topk_into`, the GPU writes `HDR_T`, `IDS` and `XQ`, then
   `__threadfence_system()`, then `SEQ` (`db_publish` does this; the real router or quantize
   kernel can write the mailbox directly). The window counter `win` is a device word written
   stream-ordered before each replay. Proposal: a third word in the kernel track's `Win`
   record.
2. `db_wait(mb, FLAG_A, win, layer, PLAN, 533, plan_vram)`: one thread spins
   (`ld.acquire.sys`), then the block copies the plan into the VRAM buffer `moe_grouped_into`
   reads.
   - **Never have many warps read mapped control words:** each warp-read is a serialised
     PCIe round trip, measured ~1 µs.
   - The plan is `contract::MoePlan`, word for word the kernel track's layout. Groups are the
     *resident* distinct experts (address = VRAM slot or scratch) plus the shared expert.
     Missed experts get no group and are listed in `MISSING`.
3. `moe_grouped_into(xq, plan_vram, parts)` runs hits and the shared expert while the host
   computes the misses.
4. `db_wait(mb, FLAG_B, win, layer, CPU_ROWS, 89, list_vram)`, then
   `db_copy_rows(mb, list_vram, parts)`: one block per listed row copies `ROWS[dst]` →
   `parts[dst]`.
5. `moe_combine_into` as usual. **Router weights are applied only on the GPU.** CPU rows are
   raw expert outputs, like VRAM-computed `parts` rows.

What the host computes for each missed expert e and each routed occurrence `j = t·TOPK + rank`
with `ids[j] = e`: `ROWS[j] = down · q(silu(gate · x̂_t) ⊙ (up · x̂_t))`, where:

- `x̂_t` is token t's row of `XQ`;
- `q()` is the same int8 contract applied to the 640-wide intermediate;
- weights are the expert's repacked Q2_0 blob `gate | up | down` (`contract::ExpertBlob`,
  1,382,400 B), read from the resident-mode host arena.

**Activation contract** (the kernel track's, followed here unchanged): per 32-element chunk,
`d = amax/127` as f32, `q = clamp(round_half_away(x/d), −127, 127)` with IEEE division, `Σq`
stored alongside. A chunk's dot is `d_w · d_x · (Σ code·q − Σq)`.

**The coordinator's sketch matches the kernel track.** Its "dot = d·s·(Σcode·x̂ − Σx̂)" is
`d_w · d_x · (Σ code·q − Σq)`, with an fp16 weight scale per 64-weight block and an f32
activation scale per 32-element chunk.

**Not yet verified against the kernel track's own code:**
- **The contract is copied, not imported** into `contract.rs`, because their
  `flash.rs`/`cpu/flash.rs` is uncommitted on `claude/flash-kernels`. When it lands,
  `contract.rs` should re-export it, and a test should run `expert_ref` against their
  `moe_grouped`.
- **The GPU may compute `silu` differently.** The reference uses `x / (1 + exp(−x))` in fp32
  with libm `exp`; the GPU may use `__expf`.

**Host side:** `doorbell::serve_layer(mb, seq, stream, exec, addr, host_blob, shared)` once
per layer, in order, on the thread that launched the graph:

- `addr(e)` is `ResidentCache::addr(layer·512 + e)`;
- `host_blob(e)` is `ResidentCache::host_blob(...)`;
- `exec` is a `MissExec` over a `Pool` of the P-cores.

Between windows, call `ResidentCache::record(routed keys)` and `ResidentCache::boundary(main)`.

**Watchdog.** `Mailbox::wait_seq` polls `cuStreamQuery` every 2 ms (`TANG_MOE_WATCHDOG_MS`).
If the stream has finished or failed without publishing, it calls `Mailbox::abort()`, which
raises both flags to `u32::MAX` so no device wait spins forever, and returns an error. No host
callbacks are used anywhere.

## Resident mode

The host arena holds only experts not in VRAM: 12,235 blobs, 16.9 GB. A swap is an exchange
through a VRAM scratch slot, one copy per window boundary (`resident.rs` has the stage table):

1. save `S → X`;
2. fill `H → S`, after which the incoming expert is admitted;
3. write back `X → H`, after which the victim is served from the host.

The victim leaves its slot as soon as its bytes are safe in scratch, and stays GPU-served from
scratch until its host copy exists. This is the one place this differs from "victim
non-resident immediately": in resident mode a victim has no host copy to serve from yet.

**Tests:**
- CPU tests simulate memory with random copy delays. They check that every window reads the
  right bytes and no in-flight copy writes a served location.
- A GPU test does the same with real copies and checksums through the device table.

**Budget:** 96 swaps in flight need 96 scratch slots (133 MB of VRAM, taken from the slot
budget). With fewer, adaptation is throttled to the free scratch count.
