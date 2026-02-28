# tang-compute Performance Plan

Optimization roadmap for the Metal compute backend, ordered by impact.

**Hardware target:** Apple M4 Max (48 GB unified memory, ~400 GB/s bandwidth)
**Baseline:** See `/BENCHMARKS.md` for current numbers. Key reference points:
- Decode attention (q1, kv=1024): **2.97 ms** per layer
- Prefill attention (s128): **990 µs** per layer
- Full decode layer: **2.68 ms** (attention-dominated)
- Matmul (576x576): **40 µs** (naive kernel)
- Every dispatch calls `wait_until_completed()` — full GPU sync per op

---

## Phase 1: Infrastructure

These are prerequisite changes that benefit everything downstream.

### 1.1 Command buffer batching

**Problem:** Every `MetalDevice` method creates a new command buffer and calls `wait_until_completed()`. A single transformer layer does ~15 dispatches (embed, 2x rms_norm, attention with Q/K/V projections, 3x ffn projections, adds) — that's 15 GPU round-trips with idle time between each.

**Fix:** Accumulate compute commands into a shared command buffer. Add `flush()` / auto-flush. Two options:

- **Lazy flush:** Encode all dispatches into one command buffer. Only `commit + waitUntilCompleted` on `sync()` or when reading results (`download()`).
- **Event-based:** Use `MTLSharedEvent` to express dependencies between operations without serializing them.

Start with lazy flush — it's simpler and eliminates the per-op sync overhead entirely. Expect 2-5x improvement on full-layer latency from removing idle gaps alone.

```
trait change: none (sync() already exists)
metal.rs: ~80 lines — RefCell<CommandBuffer>, encode-only dispatch, flush on download/sync
```

### 1.2 Simdgroup matmul activation

**Problem:** `MATMUL_MSL` (simdgroup_matrix 8x8 tiles, 32x32 threadgroup blocks) is written but not wired up. All matmuls use `MATMUL_NAIVE_MSL` (one thread per output element).

**Fix:** Wire up the simdgroup kernel with proper dispatch geometry:
- Threadgroups: `ceil(M/32) x ceil(N/32)`
- Threads per threadgroup: 128 (4 simdgroups of 32)
- Add alignment padding for non-multiple-of-32 dimensions

The simdgroup kernel should be ~10x faster than naive for LLM-sized matmuls (576x576 and larger). Fall back to naive for small matrices (M or N < 32).

```
metal.rs: ~20 lines — dispatch logic for simdgroup kernel, size threshold
```

---

## Phase 2: Flash attention (training + prefill)

The core algorithmic improvement. Removes the 256 sequence length limit and eliminates O(N^2) memory traffic.

### 2.1 Tiled causal attention with online softmax

Replace `CAUSAL_ATTENTION_MSL` with a flash attention kernel. Process K/V in tiles of size `B_kv` (64 or 128), maintaining running softmax statistics per query row:

```
for each K/V tile [j..j+B_kv]:
    load Q_tile, K_tile, V_tile into threadgroup memory
    S_tile = Q_tile @ K_tile^T * scale       // simdgroup matmul
    apply causal mask (set future positions to -inf)
    m_new = max(m_old, rowmax(S_tile))
    P_tile = exp(S_tile - m_new)
    l = exp(m_old - m_new) * l + rowsum(P_tile)
    O = exp(m_old - m_new) * O + P_tile @ V_tile  // simdgroup matmul
O /= l
```

**Key design decisions for Metal:**
- Use `simdgroup_matrix<float, 8, 8>` for both tile matmuls (Q@K^T and P@V)
- Tile sizes: B_q=32, B_kv=64 (tuned for M-series threadgroup memory — 32 KB)
- One threadgroup per (query_tile, head) pair
- Store running (m, l, O) in registers, not threadgroup memory
- Use `exp2` instead of `exp` for faster hardware path (rescale by `log2(e)`)

**Memory:** Only O(N) for the output + O(1) statistics per query row. Never materializes the N×N attention matrix.

**Sequence length:** Unlimited (tiles handle any length).

**Reference:** Philip Turner's [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) for Metal-specific blocking strategy.

```
kernels/flash_attention_msl.rs: ~200 lines
metal.rs: ~30 lines (dispatch with tiled geometry)
cpu.rs: no change (existing impl is the reference)
```

### 2.2 Flash attention backward (training)

Two-kernel approach (following Metal FlashAttention's design):

**Kernel 1 — dQ:** For each query tile, iterate over all K/V tiles. Recompute S and P from saved (m, l) statistics. Accumulate dQ.

**Kernel 2 — dK/dV:** For each K/V tile, iterate over all query tiles. Accumulate dK and dV.

Split into two kernels because dQ parallelizes over query positions while dK/dV parallelizes over key positions — different parallelization axes.

Save from forward: output O, row-wise (m, l) statistics. Recompute everything else.

**New trait method:**
```rust
fn flash_attention_backward(
    &self,
    grad_output: &Self::Buffer,
    q: &Self::Buffer, k: &Self::Buffer, v: &Self::Buffer,
    output: &Self::Buffer, logsumexp: &Self::Buffer, // saved from forward
    seq_len: usize, n_heads: usize, head_dim: usize,
) -> (Self::Buffer, Self::Buffer, Self::Buffer); // (dQ, dK, dV)
```

```
kernels/flash_attention_msl.rs: +300 lines (two backward kernels)
device.rs: +15 lines (trait method)
metal.rs: +40 lines (dispatch)
cpu.rs: +60 lines (reference implementation)
```

---

## Phase 3: Flash-Decoding (inference)

### 3.1 Parallel KV-cache attention

**Problem:** Current `kv_attention` uses one threadgroup per head. For 32 heads, that's 32 threadgroups — far below the GPU's capacity. At cache_len=1024, each threadgroup does 1024 sequential dot products.

**Fix:** Flash-Decoding splits the KV cache into C chunks along the sequence dimension. Each (head, chunk) pair runs independently, computing partial `(O_partial, m_partial, l_partial)`. A reduction kernel combines partials via log-sum-exp:

```
// Phase 1: C threadgroups per head, each handles cache_len/C positions
for each chunk c in [0..C]:
    partial_O[h,c], partial_m[h,c], partial_l[h,c] = attend(q[h], K[chunk_c], V[chunk_c])

// Phase 2: reduce across chunks (one threadgroup per head)
for each head h:
    O[h] = logsumexp_combine(partial_O[h,:], partial_m[h,:], partial_l[h,:])
```

With C=16 chunks and 32 heads, that's 512 threadgroups in phase 1 — much better GPU utilization. The reduction in phase 2 is tiny (16 partials per head).

**Expected impact:** For decode at kv=1024, expect 4-8x speedup (currently 2.97 ms, target ~400-700 µs).

```
kernels/flash_decode_msl.rs: ~150 lines (partial attention + reduction)
device.rs: no change (uses existing kv_attention signature)
metal.rs: ~40 lines (two-phase dispatch)
```

---

## Phase 4: Kernel fusion

### 4.1 Fused RoPE + attention

**Problem:** RoPE is currently a separate kernel pass. For each attention layer, Q and K are:
1. Written to global memory by the projection matmul
2. Read by RoPE kernel, rotated, written back
3. Read again by the attention kernel

That's 4 extra global memory passes on Q and K.

**Fix:** Apply rotary embeddings inside the flash attention tiling loop. As each Q/K tile is loaded into threadgroup memory, apply the rotation in-place before computing S_tile = Q_rope @ K_rope^T. The cos/sin tables (one per position) are tiny and fit in threadgroup memory.

```
// Inside flash attention tile loop:
load Q_tile[i..i+B_q, :] into threadgroup
apply_rope_inplace(Q_tile, positions[i..i+B_q], cos_table, sin_table)
for each K/V tile:
    load K_tile[j..j+B_kv, :] into threadgroup
    apply_rope_inplace(K_tile, positions[j..j+B_kv], cos_table, sin_table)
    S_tile = Q_rope_tile @ K_rope_tile^T * scale
    ...
```

For interleaved RoPE (as used by gaia): pairs `(x[2i], x[2i+1])` are rotated together. The in-tile rotation is ~10 lines of MSL.

**Savings:** Eliminates 2 full tensor read/writes per layer. For d_model=576, seq_len=128: saves ~590 KB of memory traffic per layer.

```
kernels/flash_attention_msl.rs: +30 lines (rope fusion in tile loop)
device.rs: +rope parameters on attention methods (or separate fused_attention method)
```

### 4.2 GQA-aware cross-head tiling

**Problem:** In GQA, multiple Q heads share the same K/V head group (e.g., 8 Q heads per KV head in Llama 3). Standard flash attention processes each Q head independently, loading the same K/V tiles repeatedly.

**Fix:** Tile across Q heads within the same KV group. Load K/V tiles once into threadgroup memory, then apply them to all Q heads in the group:

```
for each KV head group g:
    for each K/V tile:
        load K_tile, V_tile once into threadgroup memory
        for each Q head h in group g:
            S_tile = Q_h_tile @ K_tile^T
            update (m_h, l_h, O_h)
```

With GQA ratio 8:1, this loads K/V 8x less. The Q data per head is small (B_q × head_dim), so all heads' accumulators fit in registers.

**Impact:** Memory traffic for K/V drops by `heads_per_kv` factor. For 8:1 GQA, that's ~4x effective bandwidth improvement on the memory-bound attention operation.

```
kernels/flash_attention_msl.rs: modification to tiling loop (~40 lines)
```

---

## Phase 5: Precision

### 5.1 Half-precision tile arithmetic

Metal has 2:1 half:float throughput. The tile matmuls (Q@K^T and P@V) are the dominant compute — doing them in `half` with `float` accumulators doubles arithmetic throughput.

```
// In tile computation:
simdgroup_half8x8 q_tile_h = simdgroup_half8x8(q_tile);  // cast
simdgroup_half8x8 k_tile_h = simdgroup_half8x8(k_tile);
simdgroup_float8x8 s_tile;
simdgroup_multiply_accumulate(s_tile, q_tile_h, k_tile_h, s_tile);  // mixed precision
```

Softmax statistics (m, l) and final output accumulation stay in float32 for numerical stability.

**Risk:** Precision loss in attention scores. Mitigated by:
- float32 accumulation in simdgroup_multiply_accumulate
- float32 softmax (only the matmul inputs are half)
- For training backward, may need to keep float32 throughout

```
kernels/flash_attention_msl.rs: ~20 lines of casts
```

### 5.2 Mixed-precision model weights

Store model weights in float16, compute in float32. Halves memory footprint and bandwidth for weight-loading (which dominates decode latency for large vocab projections like lm_head).

Requires adding `upload_f16()` and `half` buffer support to `ComputeDevice`.

```
device.rs: +2 methods (upload_f16, buffer type flag)
metal.rs: +30 lines
```

---

## Phase 6: Speculative / advanced

### 6.1 Persistent kernel pattern

Instead of dispatching N separate command encoders for N layers, use a single persistent kernel that loops over layers internally. The GPU stays resident, avoiding launch overhead between layers.

Metal supports this via indirect command buffers or simply long-running compute kernels with device-side control flow. Main challenge: all layer weights must be bound upfront (Metal has a 31-buffer limit per encoder, so use argument buffers).

### 6.2 Tang-expr attention generation

Extend tang-expr tracing to generate flash attention kernel variants automatically. Define the attention computation as a traced graph — tang-expr generates the tiled MSL with appropriate blocking for the target dimensions. This would allow:
- Auto-fused RoPE (traced as part of the attention expression)
- Auto-fused bias/masking patterns
- Dimension-specialized kernels (compile-time head_dim)

Speculative — requires significant tang-expr extension for tiled/reduction patterns.

### 6.3 Approximate long-context attention

For very long KV caches (>4K tokens), use a two-pass strategy:
1. **Window attention:** Full precision over the last W tokens (e.g., 512)
2. **Compressed attention:** Downsample older tokens via strided access or pre-computed summary vectors

Trades exactness for O(W + N/stride) complexity instead of O(N). Only useful for very long contexts where exact attention is genuinely bottlenecked.

---

## Summary

| Phase | Change | Expected impact | Effort |
|-------|--------|----------------|--------|
| 1.1 Command batching | Remove per-op sync | 2-5x layer latency | Small |
| 1.2 Simdgroup matmul | Wire up existing kernel | ~10x matmul | Small |
| 2.1 Flash attention fwd | Tiled online softmax | Removes 256 limit, O(N) memory | Medium |
| 2.2 Flash attention bwd | Two-kernel backward | Enables long-seq training | Medium |
| 3.1 Flash-Decoding | Parallel KV chunks | 4-8x decode attention | Medium |
| 4.1 Fused RoPE | In-tile rotation | ~15% layer bandwidth | Small |
| 4.2 GQA cross-head | Shared K/V tile loads | Up to `heads_per_kv` x on K/V | Small |
| 5.1 Half precision | fp16 tile matmuls | ~2x arithmetic throughput | Small |
| 5.2 Mixed-precision weights | fp16 storage | 2x weight bandwidth | Small |
| 6.x Speculative | Persistent kernels, codegen | Varies | Large |
