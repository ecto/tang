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
