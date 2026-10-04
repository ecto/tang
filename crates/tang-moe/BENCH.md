# tang-moe benchmarks on mew

Measured 2026-10-04 on mew: RTX 3090 24 GB on PCIe 4.0 x16 (CPU root port 00:01.0, link
16 GT/s x16 both ends), driver 580, CUDA 12.8; i9-12900K (8P+8E), 64 GB DDR5-4800; kernel
6.8.0-137-generic. **The Intel IOMMU is on** (Ubuntu's `CONFIG_INTEL_IOMMU_DEFAULT_ON=y`, empty
GRUB cmdline): the GPU's group (17) is type `DMA-FQ`, so every GPU DMA is translated. That
turns out to dominate every PCIe number below; see "PCIe diagnosis".

Box state during the runs: other GPU clients held 1.67 GB VRAM (sunshine, two frigate ffmpeg
decoders, several small robot-sim processes); GPU utilisation read 0–1 % before each run;
host load average 1.6–2.7 (other agents compiling); 43–45 GB RAM available. No other tang
process was on the GPU while these ran. Numbers were stable across repeated runs (the
(a) and (a') tables were each run twice, medians within 5 %).

## Build and run

```sh
# on mew, in ~/Developer/tang-runtime
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
cargo build --release -p tang-moe --features cuda --bins
TANG_REQUIRE_CUDA=1 cargo test --release -p tang-moe --features cuda

B=./target/release/tang-moe-bench
$B register --max-gb 20                 # pinning limits
$B all --arena-gb 16                    # (a) (a') (b) (e) PCIe diagnosis (c) (d); ~1 min
$B size                                 # slot arena sizing for Flash-Next
$B window --arena-gb 16 --iters 10 --no-thp   # (a') with 4 KiB pages
```

`all` registers one 16 GiB arena (pinned, 2 MiB THP), fills 12,427 blobs of 1,382,400 B with
random 2-bit codes and bf16 scales, and copies the first 1,024 of them to a 1.4 GB VRAM pool.
Every run checks the expert kernel's output against a CPU reference on one host-mapped blob
and one VRAM blob before timing anything.

The kernel, `q2_expert` (src/kernels.rs), is a T=1 matvec over one Flash-Next-shaped blob per
`blockIdx.y`: gate+up 1280 × 2560 and down 2560 × 640, 64 weights per 16-byte group plus one
16-bit scale per group, `d·(Σc·x − Σx)` with 16 dp4a per group, x int8 in shared memory. It
reads each blob through a `u64` pointer table, as the real expert kernel will.
`stream_sum` is a plain 16-byte-load streaming read, for the ceiling.

## Pinning (`register`)

`cuMemHostRegister(PORTABLE | DEVICEMAP)` on untouched anonymous memory with
`MADV_HUGEPAGE` (`MAP_HUGETLB` is tried first and fails: `HugePages_Total` is 0).

| size | result | seconds | huge pages |
|---|---|---|---|
| 1 GiB | ok, one region | 1.27 | 9 % |
| 2 GiB | ok | 2.47 | 52 % |
| 4 GiB | ok | 4.51 | 76 % |
| 8 GiB | ok | 4.04 | 68 % |
| 12 GiB | ok | 7.29 | 46 % |
| 16 GiB | ok | 12.66 | 34 % |
| 20 GiB | ok | 16.82 | 41 % |

- **RLIMIT_MEMLOCK is 8 GiB (`ulimit -l` 8193132 KiB) and a single 20 GiB registration
  succeeds without root.** The NVIDIA driver pins through its own path and does not charge
  memlock. No fallback to smaller chunks was needed at any size. (Not tried above 20 GiB, by the
  ≤ 20 GB rule.)
- Time and huge-page coverage depend on free memory, not size. The sweep above ran right after
  the page cache held 41 GB, so the kernel was reclaiming while faulting (≈1 GB/s, 34–76 %
  huge). Later 16 GiB registrations with memory already free took **1.75–1.77 s, 99–100 %
  huge pages**; with `--no-thp`, 2.51 s, 0 %.
- Unregister + munmap of 20 GiB: 0.48 s.

## (a) The kernel reads blobs in place: mapped host memory vs VRAM

Blobs at random places in the 16 GiB arena (host) or the 1.4 GB pool (VRAM). Cold = fresh
random blobs every launch; warm = the same blobs as the previous launch. GB/s = n × 1.3824 MB
/ kernel time (CUDA events), median [p10–p90] of 30 after 3 warm-ups.

| kernel | where | n | cold GB/s | cold µs | warm GB/s | warm µs |
|---|---|---|---|---|---|---|
| q2_expert | host | 1 | 4.5 [4.4–4.6] | 307 | 4.3 | 321 |
| q2_expert | host | 4 | 4.1 [3.4–5.0] | 1348 | 4.1 | 1356 |
| q2_expert | host | 16 | 3.9 [3.4–4.3] | 5724 | 3.5 | 6267 |
| q2_expert | host | 64 | 3.6 [3.5–3.8] | 24406 | 3.6 | 24788 |
| q2_expert | host | 128 | 3.6 [3.5–3.8] | 48781 | 3.6 | 49261 |
| q2_expert | VRAM | 1 | 169 [150–193] | 8 | 193 | 7 |
| q2_expert | VRAM | 4 | 415 | 13 | 450 | 12 |
| q2_expert | VRAM | 16 | 655 | 34 | 655 | 34 |
| q2_expert | VRAM | 64 | 745 | 119 | 745 | 119 |
| q2_expert | VRAM | 128 | 765 | 231 | 765 | 231 |
| stream_sum | host | 1 | 4.8 [4.7–7.1] | 290 | 4.9 | 280 |
| stream_sum | host | 16 | 5.1 | 4321 | 5.3 | 4179 |
| stream_sum | host | 128 | 5.1 [4.9–5.2] | 34921 | 5.2 | 34316 |
| stream_sum | VRAM | 128 | 873 | 203 | 873 | 203 |

**Reading mapped host memory runs at 3.6–5.1 GB/s, not the ~25 GB/s the plan assumed.** It
doesn't improve with more blobs per launch (so it isn't launch latency) or when warm (nothing
caches host memory for the GPU but the 6 MB L2). One cold miss costs ~300 µs.

## (a′) A decode window's expert pass

48 launches of `q2_expert` (one per layer) in one captured graph; each launch covers `hits`
VRAM blobs and `misses` host blobs, fresh random ids every replay (written into device memory,
stream-ordered, before the replay). Median [p10–p90] of 30 replays, host wall time.

| hits/layer | misses/layer | misses/window | window ms | Δ vs no misses | µs per miss |
|---|---|---|---|---|---|
| 24 | 0 | 0 | 2.40 [2.16–2.42] | — | — |
| 24 | 1 | 48 | 16.58 [16.08–17.10] | 14.2 | 295 |
| 24 | 2 | 96 | 35.79 [34.38–37.57] | 33.4 | 348 |
| 24 | 3 | 144 | 51.42 [50.22–52.48] | 49.0 | 340 |
| 24 | 4 | 192 | 69.77 | 67.4 | 351 |
| 24 | 6 | 288 | 104.90 | 102.5 | 356 |
| 0 | 2 | 96 | 32.06 | — | 334 |
| 0 | 3 | 144 | 47.33 | — | 329 |

- The all-hit pass (1,152 blobs, 1.59 GB) takes 2.4 ms: 660 GB/s, 70 % of the 936 GB/s peak.
- **Each miss adds ~340 µs and nothing hides it.** Hits and misses in the same launch don't
  overlap usefully: the miss-only rows are only ~10 % faster.
- With 4 KiB pages (`--no-thp`): 337–417 µs per miss, ~15 % worse.
- At the target's ~110 misses per window that is **~37 ms per window of miss reads alone.**

## (b) Copy misses to VRAM staging, then compute there

`cuMemcpyHtoDAsync` per blob from random places in the 16 GiB arena into VRAM staging, then
`q2_expert` on the staging copies (same stream). Last column: the same count of other random
blobs read in place, as in (a).

| n | copy µs | copy GB/s | copy+kernel µs | effective GB/s | read in place µs |
|---|---|---|---|---|---|
| 1 | 506 [199–509] | 2.7 | 515 | 2.7 | 305 |
| 4 | 1686 | 3.3 | 1704 | 3.2 | 1336 |
| 16 | 7013 | 3.2 | 7051 | 3.1 | 5607 |
| 64 | 27075 | 3.3 | 27278 | 3.2 | 23860 |
| 128 | 54034 | 3.3 | 54352 | 3.3 | 48955 |

Overlap. Busy = `q2_expert` over 256 VRAM blobs on stream A (354 MB); side = 64 random host
blobs (88 MB) on stream B.

| side work on B | busy alone µs | side alone µs | both, wall µs | busy kernel took µs |
|---|---|---|---|---|
| copy engine | 460 | 26804 | 26833 | 524 (+14 %) |
| mapped-read kernel | 459 | 24016 | 24595 | 1833 (**4×**) |

- **Copies overlap compute almost for free; in-place mapped reads don't.** A kernel stalled
  on PCIe loads holds SMs, which slows everything else on the GPU 4×. The copy engine costs the
  VRAM kernel 14 %.
- Scattered 1.38 MB copies from a 16 GiB arena run at 3.3 GB/s, less than half the 7.4 GB/s of
  a sequential copy (next section).

## (c) CUDA graph overhead

K dependent kernels on one stream; host wall time from enqueue/launch to sync, median of 20.
tiny = 1 block × 32 threads, one read-modify-write; medium = 256 × 256 threads, 64K floats.
Fused = one kernel looping K times.

| kernel | K | eager ms | eager µs/kernel | graph ms | graph µs/kernel | fused ms |
|---|---|---|---|---|---|---|
| tiny | 500 | 0.757 | 1.51 | 0.441 | 0.88 | 0.080 |
| tiny | 1000 | 1.510 | 1.51 | 0.923 | 0.92 | 0.154 |
| tiny | 2000 | 3.025 | 1.51 | 2.018 | 1.01 | 0.302 |
| medium | 500 | 0.886 | 1.77 | 0.540 | 1.08 | — |
| medium | 1000 | 1.779 | 1.78 | 1.068 | 1.07 | — |
| medium | 2000 | 3.567 | 1.78 | 2.124 | 1.06 | — |

- A graph costs **~1 µs per dependent kernel**; eager launch 1.5–1.8 µs; fused 0.15 µs per step.
  A 48-layer window at 15–40 kernels per layer (720–1,920 kernels) spends **0.7–2 ms** on
  kernel boundaries even inside a graph.
- **Per-step scalars in device memory replay correctly** when rewritten with
  `cuMemcpyHtoDAsync` on the graph's stream (20 replays × 1,000 kernels, exact). A synchronous
  `cuMemcpyHtoD` is *not* ordered with a non-blocking stream and gave wrong results: write
  per-step inputs stream-ordered, or into mapped memory read by the graph.

## (d) Device spin-wait on a mapped flag

A one-thread kernel spins on a `volatile` flag in the registered arena; the host thread is
pinned to P-core 2.

| measurement | median µs | p90 | p99 | max |
|---|---|---|---|---|
| round trip: host writes flag → kernel sees it, writes ack → host sees ack (20,000 rounds) | 3.96 | 4.13 | 4.30 | 260 |
| host writes flag → waiting kernel exits → `cuStreamQuery` idle (300 trials) | 3.54 | 4.02 | 4.22 | 11 |

So one host↔GPU handoff through mapped memory is ~2 µs each way; 48 per window is ~0.2 ms.
The rare 260 µs outlier is the host thread losing its core.

## (e) Host DRAM and copy rates

AVX2 64-bit sum over the whole 16 GiB pinned arena, threads pinned; median [min–max] of 5.

| threads | GB/s |
|---|---|
| 1 P-core | 27.9 [27.8–27.9] |
| 8 P-cores, one thread each | 66.5 [66.0–68.2] (68.6 in a second run) |
| 16 P threads (HT) | 70.1 [69.1–70.2] |
| 8 P + 8 E | 69.1 [68.7–69.4] |

| copy, 1 GiB | GB/s |
|---|---|
| H2D from the pinned arena | 7.4 |
| D2H to the pinned arena | 8.5 |
| H2D from a pageable `Vec` | 2.8 |

## PCIe diagnosis (`pcie`)

Why 7.4 GB/s on a Gen4 x16 link (expected ~24–26 GB/s):

| source | copy size | span touched | H2D GB/s |
|---|---|---|---|
| registered mmap, 2 MiB THP | 64 KiB | 2 MiB | 5.8 |
| registered mmap, 2 MiB THP | 2 MiB | 2 MiB | 7.4 |
| registered mmap, 2 MiB THP | 1.35 MB | 1 GiB | 7.3 |
| registered mmap, 2 MiB THP | 1 GiB | 1 GiB | 7.4 |
| `cuMemHostAlloc` (driver, 4 KiB pages) | any of the above | | 2.5–2.8 |

| source | kernel reading in place | GB/s |
|---|---|---|
| registered mmap, THP | 1 blob, same one each launch | 7.2 |
| registered mmap, THP | 64 contiguous blobs | 7.5 |
| `cuMemHostAlloc` | 1 blob / 64 blobs | 4.8 / 4.6 |

What this shows, and what it doesn't:

- The link is not the limit: sysfs reports 16 GT/s x16 on both the GPU and its CPU root port,
  and the GPU sat in P2 at 1980 MHz with no throttle reasons during the copies.
- **Throughput depends on the host page size** (4 KiB → 2.8 GB/s, 2 MiB → 7.4 GB/s) **and on
  the span touched** (scattered blobs over 16 GiB → 3.3 GB/s vs 7.3 GB/s over 1 GiB). Neither
  should matter to a bare PCIe link; both are what DMA remapping through the IOMMU's IOTLB
  looks like. The VT-d unit supports 2 MiB and 1 GiB superpages and pass-through
  (`cap`/`ecap` from sysfs).
- *Not verified:* that passthrough restores ~25 GB/s. That needs root and a reboot (see
  DESIGN.md). Until then every PCIe number here is an IOMMU-on number.

## Slot arena sizing (`size`)

Stand-ins for dense weights (3.5 GB), MTP (1.0 GB) and KV/state (1.0 GB) are allocated first;
`ExpertCache::new` takes the rest minus a 0.7 GB reserve, touches it, re-measures, and shrinks.

```
free at start 23.26 GB of 25.30 GB; after stand-ins 17.76 GB; reserve 0.7 GB
slots: first estimate 12342 → after touch free 0.70 GB → final 12341 slots (17.06 GB, 50.2 % of 24,576)
seeding 12341 slots from pinned host: 2.33 s (7.3 GB/s; host copies alias a 2 GiB arena)
one adaptation: 96 swaps; plan + publish 0.19 ms on the host; copies landed after 18.2 ms
```

From a real 17 GB scattered arena, expect the 3.3 GB/s of (b): ~5 s to seed, ~40 ms per
96-swap adaptation (on the copy engine, overlapping decode at ~14 % cost to concurrent kernels).
