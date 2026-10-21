# tang-moe

The offloaded mixture-of-experts runtime under tang-llm's Flash engine: a pinned host arena for
expert weights, a VRAM expert cache with its eviction policy, a pointer table the GPU reads,
and the GPU↔CPU doorbell that runs cache misses on the CPU (AVX2 / AVX-VNNI).

Most users want [`tang-llm`](https://crates.io/crates/tang-llm), which uses this crate behind
its `cuda` feature. See [DESIGN.md](DESIGN.md) for the layout and the cache policy.

Features: `cuda` (NVIDIA, through cudarc and NVRTC).
