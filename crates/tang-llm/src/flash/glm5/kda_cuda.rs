//! The CUDA kernels of the GLM-5.3-Flash KDA layers ([`super::kda`]), naive-but-correct
//! first (GATES.md): the GDN kernels' shape ([`super::super::coder_kernels`] pattern) with
//! KDA's differences — a **per-channel** gate (a vector over each head's key dims, not a
//! scalar per head), a β gate per head, and no fused output gate (KDA gates with a gated
//! RMSNorm over the whole head output, done by the CPU reference for now).
//!
//! Two kernels, mirroring the CPU reference exactly:
//! - `kda_conv`: depthwise causal conv + SiLU over the stacked qkv of a prefill window,
//!   reading `d_conv − 1` inputs of history per channel (naive: one thread per channel,
//!   loops the taps; commit of the new history is a separate kernel, `kda_conv_commit`).
//! - `kda_step`: one decode token per launch (conv fused into the step), grid (heads, b).
//! - `kda_prefill`: the chunked prefill, grid (heads, b), block `head_dim`; each block loops
//!   its chunks serially.
//!
//! All state layouts match the CPU reference 1:1 so the parity test is a byte compare:
//! recurrent state `[slot][head][key][value]` f32, conv history `[slot][chan][d_conv-1]`
//! oldest tap first.
//!
//! # Pod commands (batched by the manager into a 1×H200 session)
//! ```text
//! # build + run the KDA parity tests (cuda feature, no default features on Linux)
//! CARGO_TARGET_DIR=/tmp/glm5-kda-target cargo test -p tang-llm --lib \
//!     --no-default-features --features cuda glm5_kda_ -- --nocapture
//! ```
#![allow(clippy::too_many_arguments)]
use anyhow::{ensure, Result};

/// The `#define` prelude for the KDA kernels.
pub fn prelude(cfg: &super::kda::KdaConfig) -> String {
    format!(
        "#define KDA_H {}\n#define KDA_D {}\n#define KDA_KC {}\n#define KDA_CONV {}\n",
        cfg.heads,
        cfg.head_dim,
        cfg.d_conv,
        cfg.conv_dim(),
    )
}

/// `kda_conv`: depthwise causal conv + SiLU over `[t][KDA_CONV]` (one layer's qkv rows).
/// `X` is the pre-conv input (post-projections), `H0` the `d_conv − 1` rows of history per
/// channel (oldest first), `W` `[chan][d_conv]` tap 0 = newest. `Y` is the conv output
/// (pre-SiLU in `H`, post-SiLU in `Y`). One thread per channel; grid `ceil(KDA_CONV/256)`.
pub const CONV_SRC: &str = r#"
typedef unsigned long long u64;

extern "C" __global__ void kda_conv(const float* __restrict__ X, unsigned int stride,
                                    const float* __restrict__ H0, const float* __restrict__ W,
                                    float* __restrict__ Y, unsigned int T) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int bi = blockIdx.y;
    if (ch >= KDA_CONV) return;
    X += (u64)bi * T * stride; Y += (u64)bi * T * KDA_CONV; H0 += (u64)bi * KDA_CONV;
    float w[KDA_KC];
    for (unsigned int i = 0; i < KDA_KC; i++) w[i] = W[ch * KDA_KC + i];
    for (unsigned int t = 0; t < T; t++) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            long long p = (long long)t - (KDA_KC - 1) + i;
            float e = p < 0 ? H0[ch * (KDA_KC - 1) + (p + (KDA_KC - 1))]
                            : X[(u64)(p) * stride + ch];
            acc = __fmaf_rn(w[i], e, acc);
        }
        Y[t * KDA_CONV + ch] = acc / (1.0f + expf(-acc));
    }
}

// commit: H0 <- the last KDA_KC-1 inputs of [hist | X]. Grid ceil(KDA_CONV/256), block 256.
extern "C" __global__ void kda_conv_commit(float* __restrict__ H0, const float* __restrict__ X,
                                           unsigned int stride, unsigned int T) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int bi = blockIdx.y;
    if (ch >= KDA_CONV) return;
    X += (u64)bi * T * stride; H0 += (u64)bi * KDA_CONV;
    float old[KDA_KC - 1];
    for (unsigned int s = 0; s < KDA_KC - 1; s++) old[s] = H0[ch * (KDA_KC - 1) + s];
    for (unsigned int s = 0; s < KDA_KC - 1; s++) {
        unsigned long long p = T + s;
        H0[ch * (KDA_KC - 1) + s] = p < KDA_KC - 1 ? old[p] : X[(p - (KDA_KC - 1)) * stride + ch];
    }
}
"#;

/// `kda_step`: the fused conv + delta-rule decode step for one token (all heads in one
/// launch). `QKV` is the raw (pre-conv) stacked projection `[KDA_CONV]`, `H0`/`W` the conv
/// history/weights, `G` the pre-exp per-channel gate `[KDA_H][KDA_D]`, `B` the post-sigmoid
/// beta `[KDA_H]`, `S` the recurrent state `[KDA_H][KDA_D][KDA_D]`, `O` the pre-norm output.
/// Grid `KDA_H`, block `KDA_D` (128) — naive: each thread owns one value column; the key
/// dim loops in registers.
pub const STEP_SRC: &str = r#"
typedef unsigned long long u64;

// One thread per value column x of one head: q/k in shared (normed), the state in GLOBAL
// (naive: read and rewritten per token; registers can't hold a row the column reduction
// needs). Decay per key row a uses g[a] = exp(G[h*D + a]) — the gate channel is the KEY dim.
extern "C" __global__ void kda_step(const float* __restrict__ QKV,
                                    const float* __restrict__ H0, const float* __restrict__ W,
                                    const float* __restrict__ G, const float* __restrict__ B,
                                    float* __restrict__ S, float* __restrict__ O) {
    unsigned int h = blockIdx.x, x = threadIdx.x, D = KDA_D;
    unsigned int bi = blockIdx.y;
    QKV += (u64)bi * KDA_CONV;
    H0 += (u64)bi * KDA_CONV * (KDA_KC - 1);
    S += (u64)bi * KDA_H * D * D;
    O += (u64)bi * KDA_H * D;
    G += (u64)bi * KDA_H * D;
    B += (u64)bi * KDA_H;
    // ---- fused conv (this head's q/k/v channels) + l2 norm + scale, in shared ----
    __shared__ float q[KDA_D], k[KDA_D], sk[KDA_D];
    float qv = 0.0f, kv = 0.0f, vv = 0.0f;
    {
        unsigned int qc = h * D + x;
        float acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[qc * (KDA_KC - 1) + i] : QKV[qc];
            acc = __fmaf_rn(W[qc * KDA_KC + i], e, acc);
        }
        qv = acc / (1.0f + expf(-acc));
        unsigned int kch = KDA_H * D + h * D + x;
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[kch * (KDA_KC - 1) + i] : QKV[kch];
            acc = __fmaf_rn(W[kch * KDA_KC + i], e, acc);
        }
        kv = acc / (1.0f + expf(-acc));
        unsigned int vch = 2 * KDA_H * D + h * D + x;
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[vch * (KDA_KC - 1) + i] : QKV[vch];
            acc = __fmaf_rn(W[vch * KDA_KC + i], e, acc);
        }
        vv = acc / (1.0f + expf(-acc));
    }
    q[x] = qv; k[x] = kv;
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
    for (unsigned int a = 0; a < D; a++) { qs += q[a] * q[a]; ks += k[a] * k[a]; }
    qs = rsqrtf(qs + 1e-6f); ks = rsqrtf(ks + 1e-6f);
    float scale = rsqrtf((float)D);
    q[x] *= qs * scale; k[x] *= ks;      // now q[]/k[] hold the normed values for ALL threads
    __syncthreads();
    // ---- delta rule: one thread owns value column x ----
    // pass 1: sk[x] = Σ_a k[a]·S[a][x] with S[a][x] *= exp(G[a]) first
    float p = 0.0f;
    for (unsigned int a = 0; a < D; a++) {
        float* sp = S + ((u64)h * D + a) * D + x;
        float s = (*sp) * expf(G[h * D + a]);
        *sp = s;
        p = __fmaf_rn(k[a], s, p);
    }
    sk[x] = p;
    __syncthreads();
    float delta = B[h] * (vv - sk[x]);
    // pass 2: S[a][x] += k[a]·δ[x]; o[x] = Σ_a q[a]·S[a][x]
    float o = 0.0f;
    for (unsigned int a = 0; a < D; a++) {
        float* sp = S + ((u64)h * D + a) * D + x;
        float s = __fmaf_rn(k[a], delta, *sp);
        *sp = s;
        o = __fmaf_rn(q[a], s, o);
    }
    O[h * D + x] = o;
}
"#;

/// `kda_prefill`: the chunked prefill (chunk 64), one block per head, the block loops the
/// chunks serially. Block = KDA_D threads, thread `a` = one **key dim** (the gate channel).
/// Scratch is `[nblk][6][C][D]` f32 (GC cumsum gates, QN/KN normed q/k, VP/KP, VN v-new),
/// `nblk` = gridDim.x, indexed `((blk*6 + s)*C + i)*D + a` — host-allocated, zeroed once.
/// `A` (C×C) and the reduction staging live in shared memory.
///
/// Matches `kda_chunk` (HF's `chunk_kimi_delta_attention`) step for step, in f32 (the CPU
/// reference's f64 accumulation is not reproduced; parity tolerance 1e-4).
pub const PREFILL_SRC: &str = r#"
typedef unsigned long long u64;

// One float per thread → the block sum (D is a power of two; naive tree over shared red).
// Each phase reads only values written before the previous __syncthreads().
__device__ __forceinline__ float block_red(float* red, unsigned int a, float p, unsigned int D) {
    red[a] = p;
    __syncthreads();
    for (unsigned int s2 = D >> 1; s2 > 0; s2 >>= 1) {
        float other = 0.0f;
        if (a < s2) other = red[a + s2];
        __syncthreads();
        if (a < s2) red[a] += other;
        __syncthreads();
    }
    float v = red[0];
    __syncthreads();
    return v;
}

extern "C" __global__ void __launch_bounds__(1024) kda_prefill(
        const float* __restrict__ QKV, unsigned int T, unsigned int stride, // [T][KDA_CONV]
        const float* __restrict__ G, unsigned int gstride,    // [T][KDA_H*KDA_D] pre-exp
        const float* __restrict__ BETA, unsigned int bstride, // [T][KDA_H]
        float* __restrict__ S,                                // [H][D][D]
        float* __restrict__ O,                                // [T][H*D]
        float* __restrict__ SCR, unsigned int C) {            // scratch [blk][6][C][D]
    unsigned int h = blockIdx.x, a = threadIdx.x, D = KDA_D;
    unsigned int bi = blockIdx.y;
    unsigned int nc = (T + C - 1) / C;
    QKV += (u64)bi * T * stride;
    G += (u64)bi * T * gstride;
    BETA += (u64)bi * T * bstride;
    O += (u64)bi * T * (KDA_H * D);
    S += (u64)bi * KDA_H * D * D;
    SCR += (u64)bi * gridDim.x * 6 * C * D;
    extern __shared__ float sh[];                // A [C][C] then red [D]
    float* A = sh;
    float* red = sh + C * C;
    float* gc = SCR + ((u64)blockIdx.x * 6 + 0) * C * D;
    float* qn = SCR + ((u64)blockIdx.x * 6 + 1) * C * D;
    float* kn = SCR + ((u64)blockIdx.x * 6 + 2) * C * D;
    float* vp = SCR + ((u64)blockIdx.x * 6 + 3) * C * D;
    float* kp = SCR + ((u64)blockIdx.x * 6 + 4) * C * D;
    float* vn = SCR + ((u64)blockIdx.x * 6 + 5) * C * D;
    for (unsigned int ci = 0; ci < nc; ci++) {
        unsigned int o0 = ci * C;
        unsigned int n = min(C, T - o0);
        // normed q/k (+scale on q) and within-chunk gate cumsum
        for (unsigned int i = 0; i < n; i++) {
            unsigned int qb = h * D + a, kb = KDA_H * D + h * D + a;
            float qv = QKV[(o0 + i) * stride + qb];
            float kv = QKV[(o0 + i) * stride + kb];
            // l2 norm over the head's dims: reduction over a
            float qs = block_red(red, a, qv * qv, D);
            float ks = block_red(red, a, kv * kv, D);
            qn[i * D + a] = qv * rsqrtf(qs + 1e-6f) * rsqrtf((float)D);
            kn[i * D + a] = kv * rsqrtf(ks + 1e-6f);
            float gv = G[(o0 + i) * gstride + h * D + a];
            gc[i * D + a] = i > 0 ? gv + gc[(i - 1) * D + a] : gv;
        }
        __syncthreads();
        // A0: strictly lower, beta on row i only. A is uninitialized shared memory — zero
        // it first (the CPU's `a` starts as zeros; the fixpoint reads row m's diagonal,
        // which must be 0 until the final +I).
        for (unsigned int i = 0; i < n; i++)
            for (unsigned int j = a; j < C; j += D) A[i * C + j] = 0.0f;
        __syncthreads();
        for (unsigned int i = 1; i < n; i++) {
            for (unsigned int j = 0; j < i; j++) {
                float p = kn[i * D + a] * kn[j * D + a]
                    * expf(gc[i * D + a] - gc[j * D + a]);
                float sum = block_red(red, a, p, D);
                if (a == 0) {
                    A[i * C + j] = -sum * BETA[(o0 + i) * bstride + h];
                }
            }
        }
        __syncthreads();
        // HF's triangular fixpoint (serial in one thread; naive): row i finalises from a
        // SNAPSHOT of A0's row i taken BEFORE this i-iteration, against the already-final
        // rows m < i — entries m < j are still A0 here (the CPU ref snapshots `row` too).
        if (a == 0) {
            for (unsigned int i = 1; i < n; i++) {
                float row[64];
                for (unsigned int m = 0; m < i; m++) row[m] = A[i * C + m];
                for (unsigned int j = 0; j < i; j++) {
                    float s = 0.0f;
                    for (unsigned int m = 0; m < i; m++) s += row[m] * A[m * C + j];
                    A[i * C + j] = row[j] + s;
                }
            }
            for (unsigned int i = 0; i < n; i++) A[i * C + i] += 1.0f;
        }
        __syncthreads();
        // VP = A·(v⊙β), KP = A·(k⊙β·exp(gc))
        float beta_i = a < n ? BETA[(o0 + a) * bstride + h] : 0.0f;
        for (unsigned int i = 0; i < n; i++) {
            float p = 0.0f;
            for (unsigned int j = 0; j <= i; j++) {
                float av = A[i * C + j];
                p = __fmaf_rn(av, QKV[(o0 + j) * stride + 2 * KDA_H * D + h * D + a]
                    * BETA[(o0 + j) * bstride + h], p);
            }
            vp[i * D + a] = p;
            p = 0.0f;
            for (unsigned int j = 0; j <= i; j++) {
                float av = A[i * C + j];
                p = __fmaf_rn(av, kn[j * D + a] * BETA[(o0 + j) * bstride + h]
                    * expf(gc[j * D + a]), p);
            }
            kp[i * D + a] = p;
        }
        __syncthreads();
        // VN = VP − KP·S. KP is over the KEY dims, S maps key→value: (K'·S)[x] =
        // Σ_a KP[a]·S[a][x]. Thread a owns the VALUE dim x = a here (the state update and
        // the output's VN reads below index it by value).
        for (unsigned int i = 0; i < n; i++) {
            float p = 0.0f;
            for (unsigned int a2 = 0; a2 < D; a2++)
                p = __fmaf_rn(kp[i * D + a2], S[((u64)h * D + a2) * D + a], p);
            vn[i * D + a] = vp[i * D + a] - p;
        }
        __syncthreads();
        // output: inter + intra (thread a owns value column a → wait, value dim ≠ key dim;
        // for the output we need Σ over key rows a of q·S[a][x] — thread a holds key a, so
        // stage per-(i) reductions over a into red, one x at a time)
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int x = 0; x < D; x++) {
                float p = qn[i * D + a] * expf(gc[i * D + a]) * S[((u64)h * D + a) * D + x];
                float o = block_red(red, a, p, D);
                for (unsigned int c2 = 0; c2 <= i; c2++) {
                    p = qn[i * D + a] * kn[c2 * D + a]
                        * expf(gc[i * D + a] - gc[c2 * D + a]);
                    float sum = block_red(red, a, p, D);
                    o += sum * vn[c2 * D + x];
                }
                if (a == 0) O[(o0 + i) * (KDA_H * D) + h * D + x] = o;
            }
        }
        // state update: S[a][x] = S[a][x]·exp(gc_last[a]) + Σ_c k_c[a]·exp(gl[a]-gc_c[a])·VN_c[x]
        float gl = gc[(n - 1) * D + a];
        for (unsigned int x = 0; x < D; x++) {
            float* sp = S + ((u64)h * D + a) * D + x;
            float s = *sp * expf(gl);
            for (unsigned int c2 = 0; c2 < n; c2++)
                s += kn[c2 * D + a] * expf(gl - gc[c2 * D + a]) * vn[c2 * D + x];
            *sp = s;
        }
        __syncthreads();
    }
}
"#;


// ---------------------------------------------------------------------------
// cuda-gated launcher + parity tests

#[cfg(feature = "cuda")]
pub mod cuda {
    use super::super::kda::{ConvState, KdaConfig, KdaState};
    use anyhow::{ensure, Result};
    use tang_moe::gpu::{DevBuf, Gpu, Stream};
    use tang_moe::{args, gpu::launch};
    use cudarc::driver::sys;

    /// All KDA kernels, NVRTC-compiled once.
    pub struct Kernels {
        pub conv: sys::CUfunction,
        pub conv_commit: sys::CUfunction,
        pub step: sys::CUfunction,
        pub prefill: sys::CUfunction,
        pub _mod: tang_moe::gpu::Module,
    }

    pub fn load(g: &Gpu, cfg: &KdaConfig) -> Result<Kernels> {
        let src = format!(
            "{}\n{}\n{}\n{}",
            super::prelude(cfg),
            super::CONV_SRC,
            super::STEP_SRC,
            super::PREFILL_SRC
        );
        g.bind()?;
        let m = g.module(&src)?;
        Ok(Kernels {
            conv: m.func("kda_conv")?,
            conv_commit: m.func("kda_conv_commit")?,
            step: m.func("kda_step")?,
            prefill: m.func("kda_prefill")?,
            _mod: m,
        })
    }

    fn sincos(g: &Gpu) -> Result<Stream> {
        let s = Stream::new()?;
        g.bind()?;
        Ok(s)
    }

    /// One KDA layer's resident device side: the conv history, recurrent state and conv
    /// weights stay on the GPU for the layer's whole life; the per-call uploads are the
    /// token's qkv row(s) and gates only, and the only read-back is the pre-norm output
    /// (the gated norm + o_proj run on the host). No per-call allocations, one persistent
    /// stream — the 117 ms/layer decode profile was 2× 4 MiB state uploads + 3 sync
    /// read-backs + a fresh stream per call, not the kernel.
    pub struct DevKda {
        pub wconv: DevBuf,
        pub h0: DevBuf,
        pub st: DevBuf,
        /// preallocated: chunk-64 x/g/beta/o/scratch for prefill, reused for decode
        pub xb: DevBuf,
        pub gb: DevBuf,
        pub bb: DevBuf,
        pub ob: DevBuf,
        pub yb: DevBuf,
        pub scr: DevBuf,
        pub s: Stream,
        ch_dim: usize,
        nh: usize,
        hd: usize,
        chunk: usize,
    }

    impl DevKda {
        /// Allocate everything once. `conv` / `state` are the CPU-side values to seed with
        /// (from the tiny-model / GGUF load or a CPU fallback window); they stay the source
        /// of truth on the host ONLY for read-back via [`Self::read_conv`] /
        /// [`Self::read_state`].
        pub fn new(
            g: &Gpu,
            k: &Kernels,
            cfg: &KdaConfig,
            wconv: &[f32],
            conv: &ConvState,
            state: &KdaState,
            slots: usize,
        ) -> Result<Self> {
            let ch_dim = cfg.conv_dim();
            let (nh, hd) = (cfg.heads, cfg.head_dim);
            let chunk = 64usize;
            g.bind()?;
            // fail fast if the weights are the wrong shape
            anyhow::ensure!(wconv.len() == ch_dim * cfg.d_conv, "conv weight shape");
            let s = Stream::new()?;
            g.bind()?;
            Ok(DevKda {
                wconv: DevBuf::from_slice(wconv)?,
                h0: DevBuf::from_slice(&conv.0)?,
                st: DevBuf::from_slice(&state.0)?,
                xb: DevBuf::zeroed(slots * chunk * ch_dim * 4)?,
                gb: DevBuf::zeroed(slots * chunk * nh * hd * 4)?,
                bb: DevBuf::zeroed(slots * chunk * nh * 4)?,
                ob: DevBuf::zeroed(slots * chunk * nh * hd * 4)?,
                yb: DevBuf::zeroed(slots * chunk * ch_dim * 4)?,
                scr: DevBuf::zeroed(slots * nh * 6 * chunk * hd * 4)?,
                s,
                ch_dim,
                nh,
                hd,
                chunk,
            })
        }

        /// The one-token decode step: upload the qkv row + gates, launch, read `o` back.
        /// `conv`/`state` host copies are updated on the host too (cheap, keeps them in
        /// sync for CPU fallback), but nothing is read back from the device.
        pub fn decode(
            &mut self,
            k: &Kernels,
            cfg: &KdaConfig,
            x: &[f32],
            gpre: &[f32],
            beta: &[f32],
            conv: &mut ConvState,
            state: &mut KdaState,
            b: usize,
        ) -> Result<Vec<f32>> {
            let (nh, hd, ch_dim) = (self.nh, self.hd, self.ch_dim);
            anyhow::ensure!(x.len() == b * ch_dim, "decode x shape");
            self.s.sync()?;
            self.xb.write(0, x)?;
            self.gb.write(0, gpre)?;
            self.bb.write(0, beta)?;
            unsafe {
                launch(
                    k.step,
                    (nh as u32, b as u32, 1),
                    (hd as u32, 1, 1),
                    (3 * hd * 4) as u32,
                    &self.s,
                    args![
                        self.xb.ptr, self.h0.ptr, self.wconv.ptr, self.gb.ptr, self.bb.ptr,
                        self.st.ptr, self.ob.ptr
                    ],
                )?;
            }
            self.s.sync()?;
            // conv history shift on the host (the device copy is not read back; the host's
            // is authoritative for CPU fallback — note the device's H0 is now stale until
            // the next write, so the shift must ALSO run on the device; done via
            // conv_commit with T=1 below to keep the two in sync).
            let m = cfg.d_conv - 1;
            let mut next = conv.0.clone();
            for bi in 0..b {
                for c in 0..ch_dim {
                    for s2 in 0..m {
                        next[(bi * ch_dim + c) * m + s2] = if s2 + 1 < m {
                            conv.0[(bi * ch_dim + c) * m + s2 + 1]
                        } else {
                            x[bi * ch_dim + c]
                        };
                    }
                }
            }
            conv.0 = next;
            // device H0: same shift via the commit kernel (T=1)
            unsafe {
                launch(
                    k.conv_commit,
                    (ch_dim.div_ceil(256) as u32, b as u32, 1),
                    (256, 1, 1),
                    0,
                    &self.s,
                    args![self.h0.ptr, self.xb.ptr, ch_dim as u32, 1u32],
                )?;
            }
            self.s.sync()?;
            Ok(self.ob.read::<f32>(0, b * nh * hd)?)
        }

        /// Prefill `t ≤ chunk` tokens (chunk ≤ 64): conv + chunked core, device-resident.
        /// `t` may be < the buffer's chunk; only `t` rows are uploaded/read.
        #[allow(clippy::too_many_arguments)]
        pub fn prefill(
            &mut self,
            k: &Kernels,
            cfg: &KdaConfig,
            x: &[f32],
            gpre: &[f32],
            beta: &[f32],
            conv: &mut ConvState,
            state: &mut KdaState,
            b: usize,
            t: usize,
        ) -> Result<Vec<f32>> {
            let (nh, hd, ch_dim) = (self.nh, self.hd, self.ch_dim);
            let chunk = self.chunk;
            anyhow::ensure!(t <= chunk && chunk == 64, "chunk window");
            self.s.sync()?;
            self.xb.write(0, x)?;
            self.gb.write(0, gpre)?;
            self.bb.write(0, beta)?;
            unsafe {
                launch(
                    k.conv,
                    (ch_dim.div_ceil(256) as u32, b as u32, 1),
                    (256, 1, 1),
                    0,
                    &self.s,
                    args![self.xb.ptr, ch_dim as u32, self.h0.ptr, self.wconv.ptr, self.yb.ptr, t as u32],
                )?;
                launch(
                    k.conv_commit,
                    (ch_dim.div_ceil(256) as u32, b as u32, 1),
                    (256, 1, 1),
                    0,
                    &self.s,
                    args![self.h0.ptr, self.xb.ptr, ch_dim as u32, t as u32],
                )?;
                launch(
                    k.prefill,
                    (nh as u32, b as u32, 1),
                    (hd as u32, 1, 1),
                    ((chunk * chunk + hd) * 4) as u32,
                    &self.s,
                    args![
                        self.yb.ptr, t as u32, ch_dim as u32, self.gb.ptr, (nh * hd) as u32,
                        self.bb.ptr, nh as u32, self.st.ptr, self.ob.ptr, self.scr.ptr, chunk as u32
                    ],
                )?;
            }
            self.s.sync()?;
            Ok(self.ob.read::<f32>(0, b * t * nh * hd)?)
        }

        /// Read the conv history back (for a CPU fallback window or a checkpoint).
        pub fn read_conv(&self, conv: &mut ConvState) -> Result<()> {
            conv.0 = self.h0.read(0, conv.0.len())?;
            Ok(())
        }

        /// Read the recurrent state back (for a CPU fallback window or a checkpoint).
        pub fn read_state(&self, state: &mut KdaState) -> Result<()> {
            state.0 = self.st.read(0, state.0.len())?;
            Ok(())
        }
    }

    /// Prefill: conv + chunked core, both on GPU, against the CPU reference path.
    /// `x` = pre-conv stacked projections `[b][t][conv_dim]`; returns (o, state, conv).
    pub fn prefill(
        g: &Gpu,
        k: &Kernels,
        cfg: &KdaConfig,
        x: &[f32],
        wconv: &[f32],
        gpre: &[f32],
        beta: &[f32],
        conv: &mut ConvState,
        state: &mut KdaState,
        b: usize,
        t: usize,
        chunk: usize,
    ) -> Result<Vec<f32>> {
        let (nh, hd, ch_dim) = (cfg.heads, cfg.head_dim, cfg.conv_dim());
        ensure!(chunk == 64, "the CUDA prefill kernel is fixed at chunk 64");
        let s = sincos(g)?;
        let xb = DevBuf::from_slice(x)?;
        let wb = DevBuf::from_slice(wconv)?;
        let gb = DevBuf::from_slice(gpre)?;
        let bb = DevBuf::from_slice(beta)?;
        let h0 = DevBuf::from_slice(&conv.0)?;
        let st = DevBuf::from_slice(&state.0)?;
        let mut y = DevBuf::zeroed(b * t * ch_dim * 4)?;
        let mut o = DevBuf::zeroed(b * t * nh * hd * 4)?;
        // scratch: per (block, head) 6 planes of C*D — grid is (nh, b)
        let mut scr = DevBuf::zeroed(b * nh * 6 * chunk * hd * 4)?;
        unsafe {
            launch(
                k.conv,
                (ch_dim.div_ceil(256) as u32, b as u32, 1),
                (256, 1, 1),
                0,
                &s,
                args![xb.ptr, ch_dim as u32, h0.ptr, wb.ptr, y.ptr, t as u32],
            )?;
            launch(
                k.conv_commit,
                (ch_dim.div_ceil(256) as u32, b as u32, 1),
                (256, 1, 1),
                0,
                &s,
                args![h0.ptr, xb.ptr, ch_dim as u32, t as u32],
            )?;
            launch(
                k.prefill,
                (nh as u32, b as u32, 1),
                (hd as u32, 1, 1),
                ((chunk * chunk + hd) * 4) as u32,
                &s,
                args![
                    y.ptr, t as u32, ch_dim as u32, gb.ptr, (nh * hd) as u32, bb.ptr,
                    nh as u32, st.ptr, o.ptr, scr.ptr, chunk as u32
                ],
            )?;
        }
        s.sync()?;
        conv.0 = h0.read(0, conv.0.len())?;
        state.0 = st.read(0, state.0.len())?;
        Ok(o.read::<f32>(0, b * t * nh * hd)?)
    }

    /// One decode step: fused conv + delta rule on GPU, vs the CPU path.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        g: &Gpu,
        k: &Kernels,
        cfg: &KdaConfig,
        x: &[f32],
        wconv: &[f32],
        gpre: &[f32],
        beta: &[f32],
        conv: &mut ConvState,
        state: &mut KdaState,
        b: usize,
    ) -> Result<Vec<f32>> {
        let (nh, hd, ch_dim) = (cfg.heads, cfg.head_dim, cfg.conv_dim());
        let s = sincos(g)?;
        let xb = DevBuf::from_slice(x)?;
        let wb = DevBuf::from_slice(wconv)?;
        let gb = DevBuf::from_slice(gpre)?;
        let bb = DevBuf::from_slice(beta)?;
        let h0 = DevBuf::from_slice(&conv.0)?;
        let st = DevBuf::from_slice(&state.0)?;
        let mut o = DevBuf::zeroed(b * nh * hd * 4)?;
        unsafe {
            launch(
                k.step,
                (nh as u32, b as u32, 1),
                (hd as u32, 1, 1),
                (3 * hd * 4) as u32,
                &s,
                args![xb.ptr, h0.ptr, wb.ptr, gb.ptr, bb.ptr, st.ptr, o.ptr],
            )?;
        }
        s.sync()?;
        // the step kernel does not update the conv history — shift it on the host (the
        // same semantics as `kda::conv1d_decode`: oldest drops, the new input is newest)
        let m = cfg.d_conv - 1;
        let mut next = conv.0.clone();
        for bi in 0..b {
            for c in 0..ch_dim {
                for s2 in 0..m {
                    next[(bi * ch_dim + c) * m + s2] = if s2 + 1 < m {
                        conv.0[(bi * ch_dim + c) * m + s2 + 1]
                    } else {
                        x[bi * ch_dim + c]
                    };
                }
            }
        }
        conv.0 = next;
        state.0 = st.read(0, state.0.len())?;
        Ok(o.read::<f32>(0, b * nh * hd)?)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::super::kda::{self, ConvState, KdaConfig, KdaState};
    use super::cuda::{decode, load, prefill, DevKda};
    use tang_moe::gpu::Gpu;

    fn loadf(name: &str) -> Vec<f32> {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glm5/kda/",);
        let b = std::fs::read(format!("{p}{name}.bin")).unwrap();
        b.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn cfg() -> KdaConfig {
        KdaConfig {
            heads: 4,
            head_dim: 16,
            d_conv: 4,
            gate_lower_bound: Some(-5.0),
            eps: 1e-5,
        }
    }

    /// GPU prefill (conv + chunk 64) vs the CPU reference on the HF fixture, at several
    /// window lengths so the state handoff across chunk boundaries is exercised.
    #[test]
    fn glm5_kda_prefill_gpu_vs_cpu() {
        let Some(g) = Gpu::for_test() else { return };
        let cfg = cfg();
        let (nh, hd) = (cfg.heads, cfg.head_dim);
        let x = loadf("conv_x");
        let wconv = loadf("conv_w");
        let gpre = loadf("g");
        let beta = loadf("beta");
        let kers = load(&g, &cfg).unwrap();
        for t in [1usize, 63, 64, 65, 100] {
            let b = 1usize;
            let mut conv = ConvState::zeros(&cfg, b);
            let mut state = KdaState::zeros(&cfg, b);
            let og = prefill(&g, &kers, &cfg, &x[..t * cfg.conv_dim()], &wconv, &gpre[..t * nh * hd],
                &beta[..t * nh], &mut conv, &mut state, b, t, 64).unwrap();
            let mut conv2 = ConvState::zeros(&cfg, b);
            let mut state2 = KdaState::zeros(&cfg, b);
            let y = kda::conv1d_prefill(&x[..t * cfg.conv_dim()], &wconv, &mut conv2, &cfg, b, t).unwrap();
            let oc = kda::kda_chunk(&y, &gpre[..t * nh * hd], &beta[..t * nh], &mut state2, &cfg, b, t, 64).unwrap();
            let e: f32 = og.iter().zip(&oc).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
            let es: f32 = state.0.iter().zip(&state2.0).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
            let ec: f32 = conv.0.iter().zip(&conv2.0).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
            println!("T={t}: out {e:.3e} state {es:.3e} conv {ec:.3e}");
            assert!(e < 1e-4, "T={t} out {e}");
            assert!(es < 1e-4, "T={t} state {es}");
            assert!(ec < 1e-4, "T={t} conv {ec}");
        }
    }

    /// Cross-chunk isolation: seed both sides with the CPU's state after 64 tokens, then
    /// run tokens 64..65 through the FULL T=65 window on GPU (two chunks in one launch).
    #[test]
    fn glm5_kda_chunk1_gpu_vs_cpu() {
        let Some(g) = Gpu::for_test() else { return };
        let cfg = cfg();
        let (nh, hd) = (cfg.heads, cfg.head_dim);
        let x = loadf("conv_x");
        let wconv = loadf("conv_w");
        let gpre = loadf("g");
        let beta = loadf("beta");
        let b = 1usize;
        // CPU state after the first 64 tokens
        let mut conv2 = ConvState::zeros(&cfg, b);
        let mut state2 = KdaState::zeros(&cfg, b);
        let y0 = kda::conv1d_prefill(&x[..64 * cfg.conv_dim()], &wconv, &mut conv2, &cfg, b, 64).unwrap();
        let _ = kda::kda_chunk(&y0, &gpre[..64 * nh * hd], &beta[..64 * nh], &mut state2, &cfg, b, 64, 64).unwrap();
        let kers = load(&g, &cfg).unwrap();
        let t = 65usize;
        // GPU: full T=65 (2 chunks), state/conv seeded with the CPU's after-64 values
        let mut conv = conv2.clone();
        let mut state = KdaState(state2.0.clone());
        let og = prefill(&g, &kers, &cfg, &x[..t * cfg.conv_dim()], &wconv,
            &gpre[..t * nh * hd], &beta[..t * nh], &mut conv, &mut state, b, t, 64).unwrap();
        let mut conv3 = conv2.clone();
        let mut state3 = KdaState(state2.0.clone());
        let y = kda::conv1d_prefill(&x[..t * cfg.conv_dim()], &wconv, &mut conv3, &cfg, b, t).unwrap();
        let oc = kda::kda_chunk(&y, &gpre[..t * nh * hd], &beta[..t * nh], &mut state3, &cfg, b, t, 64).unwrap();
        for ti in 63..t {
            let mut et = 0f32;
            for j in 0..nh * hd {
                et = et.max((og[ti * nh * hd + j] - oc[ti * nh * hd + j]).abs());
            }
            println!("  t={ti} err={et:.3e}");
        }
        let e: f32 = og.iter().zip(&oc).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
        let es: f32 = state.0.iter().zip(&state3.0).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
        println!("chunk-cross: out {e:.3e} state {es:.3e}");
        assert!(es < 1e-4, "state {es}");
        assert!(e < 1e-4, "out {e}");
    }


    /// The resident `DevKda` path: 3 chained decodes + a 64-token prefill, all with no
    /// per-call allocations, vs the CPU reference. States must match to 1e-4.
    #[test]
    fn glm5_kda_resident_gpu_vs_cpu() {
        let Some(g) = Gpu::for_test() else { return };
        let cfg = cfg();
        let (b, nh, hd, ch_dim) = (1usize, cfg.heads, cfg.head_dim, cfg.conv_dim());
        let x = loadf("conv_x");
        let wconv = loadf("conv_w");
        let gpre = loadf("g");
        let beta = loadf("beta");
        let kers = load(&g, &cfg).unwrap();
        let mut convg = ConvState::zeros(&cfg, b);
        let mut stateg = KdaState::zeros(&cfg, b);
        let mut dev = DevKda::new(&g, &kers, &cfg, &wconv, &convg, &stateg, b).unwrap();
        let mut convc = ConvState::zeros(&cfg, b);
        let mut statec = KdaState::zeros(&cfg, b);
        // 3 chained decodes
        for ti in 0..3 {
            let row = &x[ti * ch_dim..(ti + 1) * ch_dim];
            let og = dev
                .decode(&kers, &cfg, row, &gpre[ti * nh * hd..(ti + 1) * nh * hd],
                    &beta[ti * nh..(ti + 1) * nh], &mut convg, &mut stateg, b)
                .unwrap();
            let yc = kda::conv1d_decode(row, &wconv, &mut convc, &cfg, b).unwrap();
            let oc = kda::kda_decode(&yc, &gpre[ti * nh * hd..(ti + 1) * nh * hd],
                &beta[ti * nh..(ti + 1) * nh], &mut statec, &cfg, b).unwrap();
            let e: f32 = og.iter().zip(&oc).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
            assert!(e < 1e-4, "decode token {ti}: {e}");
        }
        // then a 64-token prefill on the same resident state
        let t = 64usize;
        let og = dev
            .prefill(&kers, &cfg, &x[3 * ch_dim..(3 + t) * ch_dim],
                &gpre[3 * nh * hd..(3 + t) * nh * hd], &beta[3 * nh..(3 + t) * nh],
                &mut convg, &mut stateg, b, t)
            .unwrap();
        let yc = kda::conv1d_prefill(&x[3 * ch_dim..(3 + t) * ch_dim], &wconv, &mut convc, &cfg, b, t).unwrap();
        let oc = kda::kda_chunk(&yc, &gpre[3 * nh * hd..(3 + t) * nh * hd],
            &beta[3 * nh..(3 + t) * nh], &mut statec, &cfg, b, t, 64).unwrap();
        let e: f32 = og.iter().zip(&oc).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
        println!("resident prefill err {e:.3e}");
        assert!(e < 1e-4, "prefill {e}");
        // read-back must agree with the CPU's states
        dev.read_state(&mut stateg).unwrap();
        dev.read_conv(&mut convg).unwrap();
        let es: f32 = stateg.0.iter().zip(&statec.0).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
        let ec: f32 = convg.0.iter().zip(&convc.0).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
        println!("resident states: state {es:.3e} conv {ec:.3e}");
        assert!(es < 1e-4, "state {es}");
        assert!(ec < 1e-4, "conv {ec}");
    }

    /// GPU decode step (fused conv + delta rule) vs the CPU reference, 3 tokens chained.
    #[test]
    fn glm5_kda_decode_gpu_vs_cpu() {
        let Some(g) = Gpu::for_test() else { return };
        let cfg = cfg();
        let (b, nh, hd, ch) = (1usize, cfg.heads, cfg.head_dim, 3 * cfg.heads * cfg.head_dim);
        let x = loadf("conv_x"); // the conv input rows (qkv), [t][conv_dim]
        let wconv = loadf("conv_w");
        let gpre = loadf("g");
        let beta = loadf("beta");
        let kers = load(&g, &cfg).unwrap();
        let mut convg = ConvState::zeros(&cfg, b);
        let mut stateg = KdaState::zeros(&cfg, b);
        let mut convc = ConvState::zeros(&cfg, b);
        let mut statec = KdaState::zeros(&cfg, b);
        // run 3 tokens from the conv_x fixture (its conv state is zeroed, matching both)
        for ti in 0..3 {
            let row = &x[ti * ch..(ti + 1) * ch];
            let og = decode(
                &g, &kers, &cfg, row, &wconv, &gpre[ti * nh * hd..(ti + 1) * nh * hd],
                &beta[ti * nh..(ti + 1) * nh], &mut convg, &mut stateg, b,
            )
            .unwrap();
            let yc = kda::conv1d_decode(row, &wconv, &mut convc, &cfg, b).unwrap();
            let oc = kda::kda_decode(
                &yc, &gpre[ti * nh * hd..(ti + 1) * nh * hd], &beta[ti * nh..(ti + 1) * nh],
                &mut statec, &cfg, b,
            )
            .unwrap();
            let e: f32 = og
                .iter()
                .zip(&oc)
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max);
            println!("token {ti}: GPU vs CPU out {e:.3e}");
            assert!(e < 1e-4, "token {ti}: {e}");
        }
        let es: f32 = stateg
            .0
            .iter()
            .zip(&statec.0)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        assert!(es < 1e-4, "state {es}");
    }
}
