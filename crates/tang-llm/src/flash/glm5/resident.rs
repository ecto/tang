//! The device-resident decode step (`TANG_GLM5_RESIDENT`, default on when every stage is
//! resident; `=0` forces the host-staged path).
//!
//! One decode token, slot 0, one device, every stage on ONE stream (the fused mHC's):
//!
//! - embed: the host gathers the token's row and `MhcFused::begin` uploads the 4 streams
//!   (the step's only H2D);
//! - every half block: `pre_dev` writes the normed `x` straight into the consumer's input
//!   buffer (KDA `xb`, the MLA scratch `x`, or [`Resident::x`] for the FFN); the consumer
//!   runs async on the same stream (KDA: its captured decode graph; DSA:
//!   `MlaProj::window_dev`, `kv_len`/`pos0` passed as per-call kernel args; FFN:
//!   `ffn_async`); `post_dev` hands the producer's output pointer to the next pre;
//! - head: `end_dev` (last post + stream mean) → `rs_norm` (output_norm, f64 sum like the
//!   host) → k5_q8 + k5_gemv8 (the LM head) → `rs_argmax` (first max, like `argmax`) →
//!   ONE sync and a 4-byte read: the next token.
//!
//! So per token: 1 H2D, 1 D2H, 1 host sync (was ~182 / 226 / ~226). The DSA host mirrors
//! are only resized; [`DsaCache::stale_from`] marks them and the host paths refresh them
//! from the device before any host use.

use super::*;
use tang_moe::gpu::{DevBuf, Module};

const SRC: &str = r#"
// output_norm: one block (1024 threads), x[n] -> y[n] = x * inv * w; the sum of squares
// in f64 (the host's inv_rms), mean in f64, cast to f32, + eps, 1/sqrt in f32.
extern "C" __global__ void rs_norm(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ double red[1024];
    double s = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += (double)x[i] * (double)x[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf((float)(red[0] / (double)n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

// argmax over v[n]: the FIRST maximum (strict >, like the host's argmax). One block.
// out[0] = the token; `inc` (the whole-step graph) also advances the position out[1].
extern "C" __global__ void rs_argmax(const float* v, int n, unsigned* out, int inc) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float best = v[0];
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = v[i];
        if (x > best || (x == best && i < idx)) { best = x; idx = i; }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            const float a = bv[threadIdx.x], b = bv[threadIdx.x + o];
            const int ia = bi[threadIdx.x], ib = bi[threadIdx.x + o];
            if (b > a || (b == a && ib < ia)) { bv[threadIdx.x] = b; bi[threadIdx.x] = ib; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[0] = (unsigned)bi[0];
        if (inc) out[1] += 1u;
    }
}

// ---- the device embed gather (TANG_GLM5_STEP_GRAPH=2) ----
// Row tok[0] of the GGUF token_embd table (its own quant type), dequantised exactly as the
// host's gguf::dequantize (explicit _rn ops: no FMA contraction), broadcast to the hc
// streams: r[c*n + j]. Types: 0 F32, 1 F16, 2 BF16, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K.
__device__ __forceinline__ unsigned rs_u16(const unsigned char* p) { return (unsigned)p[0] | ((unsigned)p[1] << 8); }
__device__ __forceinline__ float rs_h2f(unsigned h) {
    const unsigned s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u;
    if (e == 0) {
        const float v = __fmul_rn((float)m, 5.9604644775390625e-8f);  // m * 2^-24, exact
        return s ? -v : v;
    }
    const unsigned bits = (s << 31) | (e == 31 ? (0xffu << 23) : ((e + 112u) << 23)) | (m << 13);
    return __uint_as_float(bits);
}
__device__ __forceinline__ void rs_smk4(int j, const unsigned char* q, int& sc, int& mn) {
    if (j < 4) { sc = q[j] & 63; mn = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
__device__ float rs_deq(unsigned ty, const unsigned char* row, int j) {
    if (ty == 0) return __uint_as_float((unsigned)row[4*j] | ((unsigned)row[4*j+1] << 8) | ((unsigned)row[4*j+2] << 16) | ((unsigned)row[4*j+3] << 24));
    if (ty == 1) return rs_h2f(rs_u16(row + 2 * j));
    if (ty == 2) return __uint_as_float(rs_u16(row + 2 * j) << 16);
    if (ty == 8) {
        const unsigned char* b = row + (j / 32) * 34;
        return __fmul_rn(rs_h2f(rs_u16(b)), (float)(signed char)b[2 + j % 32]);
    }
    if (ty == 12 || ty == 13) {
        const unsigned char* b = row + (j / 256) * (ty == 12 ? 144 : 176);
        const int e = j % 256, g = e / 64, w = e % 64, hi = w / 32, l = w % 32;
        int sc, mn;
        rs_smk4(2 * g + hi, b + 4, sc, mn);
        const unsigned char* qs = b + (ty == 12 ? 16 : 48);
        int q = (qs[g * 32 + l] >> (4 * hi)) & 0xF;
        if (ty == 13 && (b[16 + l] & (1u << (2 * g + hi)))) q += 16;
        const float d1 = __fmul_rn(rs_h2f(rs_u16(b)), (float)sc);
        const float m1 = __fmul_rn(rs_h2f(rs_u16(b + 2)), (float)mn);
        return __fsub_rn(__fmul_rn(d1, (float)q), m1);
    }
    if (ty == 14) {
        const unsigned char* b = row + (j / 256) * 210;
        const int e = j % 256, n = e / 128, r = e % 128, qd = r / 32, l = r % 32;
        const unsigned char* ql = b + n * 64;
        const unsigned char* qh = b + 128 + n * 32;
        const int lo = (ql[l + (qd & 1) * 32] >> ((qd & 2) ? 4 : 0)) & 0xF;
        const int h2 = (qh[l] >> (2 * qd)) & 3;
        const int sc = (signed char)b[192 + n * 8 + (l / 16) + 2 * qd];
        return __fmul_rn(__fmul_rn(rs_h2f(rs_u16(b + 208)), (float)sc), (float)((lo | (h2 << 4)) - 32));
    }
    return __uint_as_float(0x7fc00000u);
}
extern "C" __global__ void rs_embed(const unsigned char* tab, unsigned long long row_bytes, unsigned ty,
    const unsigned* tok, float* r, int n, int hc) {
    const unsigned char* row = tab + (unsigned long long)tok[0] * row_bytes;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        const float v = rs_deq(ty, row, j);
        for (int c = 0; c < hc; c++) r[(size_t)c * n + j] = v;
    }
}
"#;

/// `rs_embed`'s type id for a GGUF type it decodes (`None`: the host gathers).
fn embed_ty(t: crate::gguf::GgmlType) -> Option<u32> {
    use crate::gguf::GgmlType::*;
    Some(match t {
        F32 => 0,
        F16 => 1,
        Bf16 => 2,
        Q8_0 => 8,
        Q4K => 12,
        Q5K => 13,
        Q6K => 14,
        _ => return None,
    })
}

/// The resident step's own device state: the FFN's x / y, output_norm's weight, the token,
/// and (when there is no Q8_0 [`HeadDev`], e.g. the f32 tiny fixture) its own LM head on
/// the generic `k5_gemv`: weights, the normed row, the logits.
pub(super) struct Resident {
    m: Module,
    x: DevBuf,
    y: DevBuf,
    norm_w: DevBuf,
    /// `[token, position]` u32: `rs_argmax` writes the token; on the whole-step graph it
    /// also advances the position, which every DSA kernel of the next replay reads.
    tok: DevBuf,
    head: Option<(super::super::moe::gpu::DevW, super::super::moe::gpu::Kernels5, DevBuf, DevBuf)>,
    /// One eager resident token has run since the last host window (every buffer the
    /// step touches is allocated): the step graph may be captured.
    warm: bool,
    /// `TANG_GLM5_STEP_GRAPH=1`: the captured segments (see [`Glm5Engine::step_resident`]).
    pub(super) graphs: Option<Vec<tang_moe::gpu::Graph>>,
    /// The whole-step graphs (default; `TANG_GLM5_STEP_GRAPH=2`), `[dense, sparse]` (the indexer
    /// branch; picked on the host from `pos0`, no transfer).
    pub(super) whole: [Option<tang_moe::gpu::Graph>; 2],
    /// The DSA cache rows the whole graphs may reach (their baked cache bases stay valid
    /// while `pos0 + 1 <= whole_cap`).
    whole_cap: usize,
    /// Replays per whole graph `[dense, sparse]` (tests).
    pub(super) whole_replays: [usize; 2],
    /// What `tok` holds on the device (`(token, position)`), when known: a replay uploads
    /// the pair only when it is out of step (after an eager token or a host window).
    dev_tp: Option<(u32, usize)>,
    /// The GGUF `token_embd` table on the device and its row bytes / `rs_embed` type, for
    /// the whole graph's embed gather (`None` until first use, or the type is unsupported).
    embd: Option<Option<(DevBuf, u64, u32)>>,
}

impl Resident {
    fn new(gpu: &tang_moe::gpu::Gpu, g: &crate::gguf::Gguf, norm_w: &[f32], own_head: bool) -> Result<Self> {
        let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 resident: {e}");
        let n = norm_w.len();
        let head = if own_head {
            let info = match g.get("output.weight") {
                Some(t) => t,
                None => g.info("token_embd.weight")?,
            };
            let (rows, cols) = (info.dims[1] as usize, info.dims[0] as usize);
            ensure!(cols == n, "lm head: {cols} cols, hidden {n}");
            let dw = super::super::moe::gpu::upload(&super::super::moe::W::Gguf(g, info), 1, rows, cols)?;
            let k5 = super::super::moe::gpu::Kernels5::new(gpu)?;
            Some((dw, k5, DevBuf::zeroed_elems::<f32>(n).map_err(ge)?, DevBuf::zeroed_elems::<f32>(rows).map_err(ge)?))
        } else {
            None
        };
        Ok(Self {
            m: gpu.module(SRC).map_err(ge)?,
            x: DevBuf::zeroed_elems::<f32>(n).map_err(ge)?,
            y: DevBuf::zeroed_elems::<f32>(n).map_err(ge)?,
            norm_w: DevBuf::from_slice(norm_w).map_err(ge)?,
            tok: DevBuf::zeroed_elems::<u32>(2).map_err(ge)?,
            head,
            warm: false,
            graphs: None,
            whole: [None, None],
            whole_cap: 0,
            whole_replays: [0, 0],
            dev_tp: None,
            embd: None,
        })
    }
}

/// `TANG_GLM5_RESIDENT=0`: force the host-staged decode.
fn resident_env() -> bool {
    !std::env::var("TANG_GLM5_RESIDENT").is_ok_and(|v| v == "0")
}

impl Glm5Engine {
    /// Can slot `slot`'s next decode token take the resident step? Every stage must be
    /// resident on one device, no shadow / dump, slot 0 of a single-slot engine, and every
    /// KDA layer's device state live.
    fn resident_ready(&self, slot: usize) -> bool {
        if !resident_env() || slot != 0 || self.n_slots != 1 || self.dump.is_some() {
            return false;
        }
        if self.kda_dbg.is_some() || self.moe_dbg.is_some() || self.mla_dbg.is_some() {
            return false;
        }
        let Some(g) = self.gpu.as_ref() else { return false };
        if g.devs.len() != 1 || !g.devs[0].1.fast || g.mhc_f.is_none() {
            return false;
        }
        let hp = &self.m.hp;
        (0..hp.n_layer).all(|l| {
            g.mhc_g[l].is_some()
                && g.has_ffn(l)
                && if hp.is_kda[l] {
                    g.kda_l[l].as_ref().is_some_and(|k| k.resident_ok()) && g.kda_dev_fresh[slot][l]
                } else {
                    g.mla_k[l].is_some() && g.mla_p[l].is_some()
                }
        })
    }

    /// Before a host path touches DSA layer `l`'s mirrors: pull the rows the resident step
    /// left on the device only.
    pub(super) fn refresh_stale_dsa(&mut self, slot: usize, l: usize) -> Result<()> {
        let Some(from) = self.dsa[slot][l].stale_from else { return Ok(()) };
        let Some(g) = self.gpu.as_ref() else { return Ok(()) };
        let Some(cu) = g.mla_k[l].as_ref() else { return Ok(()) };
        g.devs[g.dev_of[l]].0.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        cu.set_seq(slot as u64);
        let c = &mut self.dsa[slot][l];
        let rows = c.valid.len();
        if rows > from {
            cu.refresh_mirrors(from, rows - from, &mut c.lat16, &mut c.idx_k, &mut c.idx_g, &mut c.valid)
                .map_err(|e| anyhow::anyhow!("glm5 dsa mirror refresh: {e}"))?;
        }
        c.stale_from = None;
        Ok(())
    }

    /// One greedy decode token on the resident step, or `None` when it doesn't apply
    /// (the caller runs the host path).
    ///
    /// The step graph (`TANG_GLM5_STEP_GRAPH=2` whole step, the default; `1` segments; `0` = eager):
    /// after one eager token (every buffer allocated), with `=1` the step
    /// is captured as `n_dsa + 1` graph segments split at the DSA attention cores, which
    /// stay eager between the segment launches (their `pos0`/`kv_len`/cache-row pointers
    /// move every token). Any host window drops the segments ([`Self::drop_step_graph`]):
    /// a prefill can reallocate the scratch they point at.
    pub(super) fn step_resident(&mut self, slot: usize, tok: u32) -> Result<Option<u32>> {
        if !self.resident_ready(slot) {
            return Ok(None);
        }
        let hp = self.m.hp.clone();
        let (n, hc) = (hp.n_embd, hp.hc);
        ensure!((tok as usize) < hp.n_vocab, "token {tok} out of range");
        let pos0 = self.slot_tokens(slot).len();
        use super::super::trace::span;
        let tr_root = super::super::trace::window_begin(1, pos0);
        let tr = span("embed");
        // the DSA weights (Arc'd; the window_dev path reads only scalars + i_ape)
        let mut mla_w: Vec<Option<std::sync::Arc<MlaWeights>>> = vec![None; hp.n_layer];
        for l in 0..hp.n_layer {
            if !hp.is_kda[l] {
                mla_w[l] = Some(self.mla_weights(l)?);
            }
        }
        let norm_w = if self.gpu.as_ref().is_some_and(|g| g.resident.is_none()) {
            Some(self.m.vec("output_norm.weight")?)
        } else {
            None
        };
        let embd = self.m.g.info("token_embd.weight")?;
        let gguf = &self.m.g;
        let host_embed = || -> Result<Vec<f32>> {
            let e = gguf.rows(embd, tok as usize, 1)?;
            let mut r = vec![0f32; hc * n];
            for c in 0..hc {
                r[c * n..(c + 1) * n].copy_from_slice(&e);
            }
            Ok(r)
        };
        // `TANG_GLM5_STEP_GRAPH`: unset / `2` = whole step (default), `1` = segments, `0` = eager
        let mode = std::env::var("TANG_GLM5_STEP_GRAPH").unwrap_or_default();
        let whole_on = mode.is_empty() || mode == "2";
        let graph_on = mode == "1";
        // the indexer branch of this token (every DSA layer shares topk / kpool)
        let dsa_l: Vec<usize> = (0..hp.n_layer).filter(|&l| !hp.is_kda[l]).collect();
        let (topk, kpool) = dsa_l.first().map_or((usize::MAX, 1), |&l| {
            let w = mla_w[l].as_ref().unwrap();
            (w.topk, w.kpool)
        });
        ensure!(
            dsa_l.iter().all(|&l| mla_w[l].as_ref().is_some_and(|w| w.topk == topk && w.kpool == kpool)),
            "resident step: DSA layers disagree on topk / kpool"
        );
        let sparse = !super::super::mla::indexer_is_dense(pos0 + 1, topk);
        let g = self.gpu.as_mut().context("resident step without a GPU")?;
        g.devs[0].0.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        if let Some(w) = norm_w {
            let own = g.head_dev.is_none();
            g.resident = Some(Resident::new(&g.devs[0].0, &self.m.g, &w, own)?);
        }
        let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 resident: {e}");
        let mut f = g.mhc_f.take().context("fused mHC: no instance")?;
        // a non-owning alias of the step stream (the fused mHC's): the launches below
        // borrow `f` mutably while they enqueue on its stream
        let s = std::mem::ManuallyDrop::new(tang_moe::gpu::Stream(f.stream().0));
        let mut res = g.resident.take().unwrap();
        let items = plan(&hp);
        // debug: TANG_GLM5_STEP_GRAPH_EAGER=kda,ffn also runs those items eagerly between
        // the segments (bisecting a replay mismatch)
        let eager_kinds = std::env::var("TANG_GLM5_STEP_GRAPH_EAGER").unwrap_or_default();
        let boundary = |it: &Item| match it {
            Item::Dsa(_) => true,
            Item::Kda(_) => eager_kinds.contains("kda"),
            Item::Ffn(_) => eager_kinds.contains("ffn"),
            _ => false,
        };
        let dsa = &mut self.dsa[slot];
        let out = (|| -> Result<u32> {
            if whole_on && res.warm {
                if let Some(t) = step_whole(&mut f, g, &mut res, dsa, &mla_w, &hp, &items, &dsa_l, pos0, tok, sparse, &s, gguf, embd, &host_embed)? {
                    return Ok(t);
                }
                // past the graphs' cache capacity: this token runs eagerly (and regrows)
            }
            res.dev_tp = None;
            if graph_on && res.warm && res.graphs.is_none() {
                // capture pass: records only, so the mHC state machine is walked once with
                // the DSA cores skipped; the replay below runs everything
                let _tr = span("capture");
                f.set_window(1)?;
                let mut graphs = Vec::new();
                let mut rest: &[Item] = &items;
                loop {
                    let cut = rest.iter().position(&boundary).unwrap_or(rest.len());
                    let seg = &rest[..cut];
                    let gr = tang_moe::gpu::Graph::capture(&s, |cs| {
                        for it in seg {
                            exec(*it, &mut f, g, &res, dsa, &mla_w, &hp, pos0, cs, Pass::Capture)
                                .map_err(|e| tang_moe::gpu::Error(format!("{e:#}")))?;
                        }
                        Ok(())
                    })
                    .map_err(|e| anyhow::anyhow!("step graph capture: {e}"))?;
                    graphs.push(gr);
                    if cut == rest.len() {
                        break;
                    }
                    // the boundary item's host side (its mHC post) in capture order
                    exec(rest[cut], &mut f, g, &res, dsa, &mla_w, &hp, pos0, &s, Pass::HostOnly)?;
                    rest = &rest[cut + 1..];
                }
                res.graphs = Some(graphs);
                eprintln!("glm5: resident step graph captured ({} segments)", res.graphs.as_ref().unwrap().len());
            }
            f.begin_async(&host_embed()?)?;
            drop(tr);
            if let Some(graphs) = res.graphs.as_ref().filter(|_| graph_on) {
                let mut gi = graphs.iter();
                gi.next().unwrap().launch(&s).map_err(ge)?;
                for it in items.iter().filter(|it| boundary(it)) {
                    exec(*it, &mut f, g, &res, dsa, &mla_w, &hp, pos0, &s, Pass::Replay)?;
                    gi.next().context("step graph: segment count")?.launch(&s).map_err(ge)?;
                }
                // the replay skipped the host side of the mHC state machine
                f.set_window(0)?;
            } else {
                for it in &items {
                    exec(*it, &mut f, g, &res, dsa, &mla_w, &hp, pos0, &s, Pass::Eager)?;
                }
                res.warm = true;
            }
            if graph_on && std::env::var("TANG_GLM5_STEP_GRAPH_RECAPTURE").is_ok_and(|v| v == "1") {
                res.graphs = None; // debug: a fresh capture every token
            }
            let tr = span("head");
            let _tr = span("sync+read");
            s.sync().map_err(ge)?;
            let t = res.tok.read::<u32>(0, 1).map_err(ge)?[0];
            drop(tr);
            Ok(t)
        })();
        g.mhc_f = Some(f);
        g.resident = Some(res);
        let next = out?;
        self.slot_tokens_mut(slot).push(tok);
        super::super::trace::window_end(tr_root, 1, pos0);
        Ok(Some(next))
    }

    /// Drop the captured step segments (any host window: it may reallocate their scratch).
    pub(super) fn drop_step_graph(&mut self) {
        if let Some(r) = self.gpu.as_mut().and_then(|g| g.resident.as_mut()) {
            r.graphs = None;
            r.whole = [None, None];
            r.whole_cap = 0;
            r.dev_tp = None;
            r.warm = false;
        }
    }
}

/// `TANG_GLM5_STEP_GRAPH=2`: the whole decode token as ONE graph replay. The device embed
/// gather reads the token `rs_argmax` left in `res.tok[0]`; every DSA kernel reads the
/// position from `res.tok[1]` (advanced by the graph's last kernel); the host only keeps
/// its watermarks / mirror sizes in step. Two graphs, dense and sparse (the indexer
/// branch), captured lazily the first time each is needed. `None`: the token is past the
/// graphs' DSA cache capacity (the caller runs it eagerly, which regrows the caches; the
/// next token recaptures).
#[allow(clippy::too_many_arguments)]
fn step_whole(
    f: &mut super::super::mhc_fused::MhcFused,
    g: &mut GpuFfn,
    res: &mut Resident,
    dsa: &mut [DsaCache],
    mla_w: &[Option<std::sync::Arc<MlaWeights>>],
    hp: &super::super::Hparams,
    items: &[Item],
    dsa_l: &[usize],
    pos0: usize,
    tok: u32,
    sparse: bool,
    s: &tang_moe::gpu::Stream,
    gguf: &crate::gguf::Gguf,
    embd: &crate::gguf::TensorInfo,
    host_embed: &dyn Fn() -> Result<Vec<f32>>,
) -> Result<Option<u32>> {
    use super::super::trace::span;
    let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 step graph 2: {e}");
    let kv_len = pos0 + 1;
    if res.whole_cap < kv_len {
        res.whole = [None, None];
    }
    if res.embd.is_none() {
        res.embd = Some(match embed_ty(embd.ty) {
            Some(ty) => {
                let rb = embd.row_bytes()? as u64;
                Some((DevBuf::from_slice(gguf.bytes(embd)).map_err(ge)?, rb, ty))
            }
            None => {
                eprintln!("glm5: step graph 2: token_embd is {}, embed stays a host gather", embd.ty.name());
                None
            }
        });
    }
    let sp = usize::from(sparse);
    if res.whole[sp].is_none() {
        let mut cap = usize::MAX;
        for &l in dsa_l {
            let cu = g.mla_k[l].as_ref().context("mla not resident")?;
            cu.set_seq(0);
            cap = cap.min(cu.row_cap());
        }
        if cap < kv_len {
            return Ok(None);
        }
        let mut grew = false;
        for &l in dsa_l {
            let w = mla_w[l].as_ref().unwrap();
            let cu = g.mla_k[l].as_ref().unwrap();
            grew |= cu.reserve_pool(cap / w.kpool + 1).map_err(ge)?;
            cu.stage_ape(&w.i_ape).map_err(ge)?;
        }
        if grew || cap != res.whole_cap {
            res.whole[1 - sp] = None;
        }
        res.whole_cap = cap;
        let _tr = span("capture");
        f.set_window(1)?;
        let dev_embed = res.embd.as_ref().unwrap().as_ref().map(|(b, rb, ty)| (b.ptr, *rb, *ty));
        let rp = f.r_ptr();
        let res_ro: &Resident = res;
        let gr = tang_moe::gpu::Graph::capture(s, |cs| {
            let mut run = || -> Result<()> {
                if let Some((tab, rb, ty)) = dev_embed {
                    let ef = res_ro.m.func("rs_embed").map_err(ge)?;
                    let (tp, ni, hci) = (res_ro.tok.ptr, hp.n_embd as i32, hp.hc as i32);
                    let grid = hp.n_embd.div_ceil(256) as u32;
                    unsafe { launch(ef, (grid, 1, 1), (256, 1, 1), 0, cs, args![tab, rb, ty, tp, rp, ni, hci])? };
                }
                for it in items {
                    exec(*it, f, g, res_ro, dsa, mla_w, hp, pos0, cs, Pass::Whole(sparse))?;
                }
                Ok(())
            };
            run().map_err(|e| tang_moe::gpu::Error(format!("{e:#}")))
        })
        .map_err(|e| anyhow::anyhow!("whole step graph capture: {e}"))?;
        f.set_window(0)?;
        res.whole[sp] = Some(gr);
        eprintln!("glm5: whole step graph captured ({}, DSA cache cap {cap} rows)", if sparse { "sparse" } else { "dense" });
    }
    let tr = span("embed");
    if res.dev_tp != Some((tok, pos0)) {
        // out of step (first replay, or after an eager token / host window): 8 bytes up
        res.tok.write_async(0, &[tok, pos0 as u32], s).map_err(ge)?;
    }
    if res.embd.as_ref().unwrap().is_none() {
        f.begin_async(&host_embed()?)?;
        f.set_window(0)?;
    }
    drop(tr);
    if sparse {
        // pools completed before this token that no replay computed (the dense graph
        // skips the indexer): the eager incremental pool, a no-op once caught up
        let _tr = span("dsa/pool_catchup");
        for &l in dsa_l {
            let (cu, w) = (g.mla_k[l].as_ref().unwrap(), mla_w[l].as_ref().unwrap());
            cu.set_seq(0);
            let np = pos0 / w.kpool;
            if cu.pool_upto() < np {
                let c = &dsa[l];
                cu.idx_pool(&c.idx_k, &c.idx_g, &w.i_ape, pos0, 0, np, &mut [], s).map_err(ge)?;
            }
        }
    }
    {
        let _tr = span("graph");
        res.whole[sp].as_ref().unwrap().launch(s).map_err(ge)?;
        res.whole_replays[sp] += 1;
    }
    // the host's view of what the replay did to the DSA caches
    for &l in dsa_l {
        let (cu, w) = (g.mla_k[l].as_ref().unwrap(), mla_w[l].as_ref().unwrap());
        let c = &mut dsa[l];
        let (dl, idd) = (w.kv_lora, w.idx_dim);
        ensure!(c.lat16.len() == pos0 * dl, "latent cache out of step");
        c.lat16.resize(kv_len * dl, 0);
        c.idx_k.resize(kv_len * idd, 0.0);
        c.idx_g.resize(kv_len * idd, 0.0);
        c.valid.resize(kv_len, 0);
        c.stale_from = Some(c.stale_from.map_or(pos0, |s| s.min(pos0)));
        cu.set_seq(0);
        cu.append_done(pos0, 1);
        if sparse {
            cu.set_pool_upto(kv_len / w.kpool);
        }
    }
    let _tr = span("sync+read");
    s.sync().map_err(ge)?;
    let t = res.tok.read::<u32>(0, 1).map_err(ge)?[0];
    res.dev_tp = Some((t, kv_len));
    Ok(Some(t))
}

/// One step of the resident decode, in order. `Dsa` is the eager part of a DSA layer (the
/// graph segment boundary); everything else is position-independent.
#[derive(Clone, Copy)]
enum Item {
    PreAttn(usize),
    Kda(usize),
    Dsa(usize),
    PreFfn(usize),
    Ffn(usize),
    Head,
}

fn plan(hp: &super::super::Hparams) -> Vec<Item> {
    let mut v = Vec::new();
    for l in 0..hp.n_layer {
        v.push(Item::PreAttn(l));
        v.push(if hp.is_kda[l] { Item::Kda(l) } else { Item::Dsa(l) });
        v.push(Item::PreFfn(l));
        v.push(Item::Ffn(l));
    }
    v.push(Item::Head);
    v
}

/// How [`exec`] runs an item.
#[derive(Clone, Copy, PartialEq)]
enum Pass {
    /// Everything, eagerly (the no-graph step and the warm-up token).
    Eager,
    /// Inside a segment capture: KDA enqueues its fused kernels instead of launching its
    /// own graph; a DSA item only does its mHC post (host state).
    Capture,
    /// Between segment launches: a boundary item runs its device work only.
    Replay,
    /// The capture pass at a boundary item (outside any capture): its mHC post only, so
    /// the next segment's pre applies it (the replay runs its device work).
    HostOnly,
    /// Inside the whole-step capture (`TANG_GLM5_STEP_GRAPH=2`; `true` = the sparse
    /// indexer branch): as `Capture`, and a DSA item enqueues its position-counter window.
    Whole(bool),
}

/// Enqueue one [`Item`] on `s`.
#[allow(clippy::too_many_arguments)]
fn exec(
    it: Item,
    f: &mut super::super::mhc_fused::MhcFused,
    g: &mut GpuFfn,
    res: &Resident,
    dsa: &mut [DsaCache],
    mla_w: &[Option<std::sync::Arc<MlaWeights>>],
    hp: &super::super::Hparams,
    pos0: usize,
    s: &tang_moe::gpu::Stream,
    pass: Pass,
) -> Result<()> {
    let capture = matches!(pass, Pass::Capture | Pass::Whole(_));
    if pass == Pass::HostOnly {
        let y = match it {
            Item::Kda(l) => g.kda_l[l].as_ref().context("kda not resident")?.y_ptr(),
            Item::Dsa(l) => g.mla_p[l].as_ref().context("mla not resident")?.y_ptr(1)?,
            Item::Ffn(_) => res.y.ptr,
            _ => anyhow::bail!("step graph: only kda / dsa / ffn items can be boundaries"),
        };
        return f.post_dev(y);
    }
    use super::super::trace::span;
    let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 resident: {e}");
    let mh = |l: usize| g.mhc_g[l].as_ref().context("mhc not resident");
    match it {
        Item::PreAttn(l) => {
            let _tr = span("mhc_pre.attn");
            let dst = if hp.is_kda[l] {
                g.kda_l[l].as_ref().context("kda not resident")?.x_ptr()
            } else {
                g.mla_p[l].as_ref().context("mla not resident")?.x_ptr(1)?
            };
            f.pre_dev(mh(l)?, l, 0, dst)?;
        }
        Item::Kda(l) => {
            let _tr = span("kda");
            let k = g.kda_l[l].as_mut().context("kda not resident")?;
            if capture {
                k.decode_enqueue(s)?;
            } else {
                k.decode_dev(s)?;
            }
            if pass != Pass::Replay {
                f.post_dev(k.y_ptr())?;
            }
        }
        Item::Dsa(l) => {
            let (cu, p) = (g.mla_k[l].as_ref().unwrap(), g.mla_p[l].as_ref().unwrap());
            if let Pass::Whole(sparse) = pass {
                let _tr = span("dsa");
                cu.set_seq(0);
                let y = p.window_dev_pos(cu, mla_w[l].as_ref().unwrap(), res.tok.ptr + 4, sparse, s)?;
                debug_assert_eq!(y, p.y_ptr(1)?);
                return f.post_dev(y);
            }
            ensure!(!capture, "step graph: a DSA item inside a capture");
            {
                let _tr = span("dsa");
                let w = mla_w[l].as_ref().unwrap();
                cu.set_seq(0);
                let c = &mut dsa[l];
                let (dl, idd) = (w.kv_lora, w.idx_dim);
                ensure!(c.lat16.len() == pos0 * dl, "latent cache out of step");
                let kv_len = pos0 + 1;
                if !cu.rows_fit(kv_len) {
                    // the append regrows the device caches and re-uploads [0, pos0) from the
                    // host mirrors: bring the stale rows over first (rare: capacity doubles)
                    if let Some(from) = c.stale_from.take().filter(|&f| f < pos0) {
                        cu.refresh_mirrors(from, pos0 - from, &mut c.lat16, &mut c.idx_k, &mut c.idx_g, &mut c.valid)
                            .map_err(|e| anyhow::anyhow!("glm5 dsa mirror refresh: {e}"))?;
                    }
                }
                c.lat16.resize(kv_len * dl, 0);
                c.idx_k.resize(kv_len * idd, 0.0);
                c.idx_g.resize(kv_len * idd, 0.0);
                c.valid.resize(kv_len, 0);
                c.stale_from = Some(c.stale_from.map_or(pos0, |s| s.min(pos0)));
                let y = p.window_dev(cu, w, 1, pos0, &c.lat16, &c.idx_k, &c.idx_g, &c.valid, s)?;
                debug_assert_eq!(y, p.y_ptr(1)?);
            }
            // replay: the host mHC state is not walked (set_window resets it after)
            if pass != Pass::Replay {
                f.post_dev(p.y_ptr(1)?)?;
            }
        }
        Item::PreFfn(l) => {
            let _tr = span("mhc_pre.ffn");
            f.pre_dev(mh(l)?, l, 1, res.x.ptr)?;
        }
        Item::Ffn(l) => {
            let _tr = span("ffn");
            let m = g.layers[l].as_ref().context("FFN not resident")?;
            super::super::moe::gpu::ffn_async(&g.devs[0].1, s, m, res.x.ptr, res.y.ptr, 1)?;
            if pass != Pass::Replay {
                f.post_dev(res.y.ptr)?;
            }
        }
        Item::Head => {
            let _tr = span("head");
            let h = f.end_dev()?;
            let n = hp.n_embd;
            let nv = hp.n_vocab;
            let nf = res.m.func("rs_norm").map_err(ge)?;
            let af = res.m.func("rs_argmax").map_err(ge)?;
            let (nw, ni, eps) = (res.norm_w.ptr, n as i32, hp.eps);
            let logits = unsafe {
                if let Some(hd) = g.head_dev.as_ref() {
                    let xf = hd.xf.ptr;
                    launch(nf, (1, 1, 1), (1024, 1, 1), 0, s, args![h, nw, xf, ni, eps])?;
                    launch(hd.k5.q8, ((n / 32) as u32, 1, 1), (32, 1, 1), 0, s, args![hd.xf.ptr, hd.xq8.ptr, n as u32, 1u32])?;
                    launch(
                        hd.k5.gemv8,
                        (nv.div_ceil(K5_WARPS) as u32, 1, 1),
                        (128, 1, 1),
                        (K5_WARPS * ((hd.dw.w.row_bytes as usize).div_ceil(16) + 1) * 16) as u32,
                        s,
                        args![hd.dw.w, 0u32, hd.xq8.ptr, hd.xq8.ptr + n as u64, hd.out.ptr,
                            nv as u32, 1u32, nv as u32, n as u32],
                    )?;
                    hd.out.ptr
                } else {
                    let (dw, k5, xb, ob) = res.head.as_ref().context("resident: no LM head")?;
                    let xf = xb.ptr;
                    launch(nf, (1, 1, 1), (1024, 1, 1), 0, s, args![h, nw, xf, ni, eps])?;
                    launch(
                        k5.gemv,
                        (nv.div_ceil(128) as u32, 1, 1),
                        (128, 1, 1),
                        0,
                        s,
                        args![dw.w, 0u32, xb.ptr, ob.ptr, nv as u32, 1u32, nv as u32, n as u32],
                    )?;
                    ob.ptr
                }
            };
            let (nvi, tp, inc) = (nv as i32, res.tok.ptr, i32::from(matches!(pass, Pass::Whole(_))));
            unsafe { launch(af, (1, 1, 1), (1024, 1, 1), 0, s, args![logits, nvi, tp, inc])? };
        }
    }
    Ok(())
}
