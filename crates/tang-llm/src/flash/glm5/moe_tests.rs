//! Tests for [`super`] (GLM-5 MoE). Fixtures: `scripts/glm5_moe_fixtures.py` (HF modules, fp32).
//! The tiny-GGUF tests read W0's tiny model (`$TANG_GLM5_TINY`, default /private/tmp/glm5-tiny)
//! and skip with a note when it is absent.

use super::*;
use crate::gguf::{dequantize, GgmlType};
use std::path::{Path, PathBuf};

const HID: usize = 64;
const NE: usize = 16;
const K: usize = 4;
const FF: usize = 32;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5/moe")
}

fn entry(name: &str) -> serde_json::Value {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir().join("fixtures.json")).unwrap())
            .unwrap();
    meta["tensors"]
        .get(name)
        .unwrap_or_else(|| panic!("{name}: no fixture"))
        .clone()
}

fn raw(name: &str) -> Vec<u8> {
    std::fs::read(dir().join(entry(name)["file"].as_str().unwrap())).unwrap()
}

/// An f32 fixture, leaked so it can back a `W::F32` for the whole test.
fn f(name: &str) -> &'static [f32] {
    let v: Vec<f32> = raw(name)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let n: u64 = entry(name)["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap())
        .product();
    assert_eq!(v.len() as u64, n, "{name}: size");
    Box::leak(v.into_boxed_slice())
}

fn ids_of(name: &str) -> Vec<u32> {
    f(name).iter().map(|v| *v as u32).collect()
}

/// HF's fused `gate_up_proj [E, 2ff, hid]` (gate rows first) split into gate / up `[E][ff][hid]`.
fn split_gu(gu: &[f32], ff: usize, hid: usize) -> (W<'static>, W<'static>) {
    let (mut g, mut u) = (Vec::new(), Vec::new());
    for m in gu.chunks_exact(2 * ff * hid) {
        g.extend_from_slice(&m[..ff * hid]);
        u.extend_from_slice(&m[ff * hid..]);
    }
    (
        W::F32(Box::leak(g.into_boxed_slice())),
        W::F32(Box::leak(u.into_boxed_slice())),
    )
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn max_abs(a: &[f32]) -> f32 {
    a.iter().map(|v| v.abs()).fold(0.0, f32::max)
}

fn base(limit: f32) -> MoeParams<'static> {
    MoeParams {
        hidden: HID,
        n_expert: NE,
        top_k: K,
        n_group: 1,
        topk_group: 1,
        ff_exp: FF,
        scale: 2.5,
        norm_topk_prob: true,
        limit_exp: limit,
        limit_shexp: limit,
        router: Some(W::F32(f("router.weight"))),
        bias: Some(W::F32(f("router.bias"))),
        ..Default::default()
    }
}

fn experts_params(gu: &str, down: &str, limit: f32) -> MoeParams<'static> {
    let (g, u) = split_gu(f(gu), FF, HID);
    MoeParams {
        gate_exps: Some(g),
        up_exps: Some(u),
        down_exps: Some(W::F32(f(down))),
        ..base(limit)
    }
}

fn full_moe_params() -> MoeParams<'static> {
    let (g, u) = split_gu(f("moe.experts.gate_up_proj"), FF, HID);
    MoeParams {
        ff_shexp: FF, // moe_intermediate_size × n_shared_experts
        router: Some(W::F32(f("moe.gate.weight"))),
        bias: Some(W::F32(f("moe.gate.e_score_correction_bias"))),
        gate_exps: Some(g),
        up_exps: Some(u),
        down_exps: Some(W::F32(f("moe.experts.down_proj"))),
        sh_gate: Some(W::F32(f("moe.shared_experts.gate_proj.weight"))),
        sh_up: Some(W::F32(f("moe.shared_experts.up_proj.weight"))),
        sh_down: Some(W::F32(f("moe.shared_experts.down_proj.weight"))),
        ..base(10.0)
    }
}

fn dense_params(tag: &str, limit: f32) -> MoeParams<'static> {
    MoeParams {
        hidden: HID,
        ff_dense: 256,
        limit_shexp: limit,
        dense_gate: Some(W::F32(f(&format!("{tag}.gate_proj.weight")))),
        dense_up: Some(W::F32(f(&format!("{tag}.up_proj.weight")))),
        dense_down: Some(W::F32(f(&format!("{tag}.down_proj.weight")))),
        ..Default::default()
    }
}

/// Per token, (id, weight) sorted by id: torch.topk(sorted=False) has no defined order.
fn by_id(ids: &[u32], wts: &[f32], k: usize) -> Vec<Vec<(u32, f32)>> {
    ids.chunks(k)
        .zip(wts.chunks(k))
        .map(|(i, w)| {
            let mut v: Vec<(u32, f32)> = i.iter().copied().zip(w.iter().copied()).collect();
            v.sort_by_key(|p| p.0);
            v
        })
        .collect()
}

/// Same expert sets per token (panics otherwise); returns the max weight error.
fn route_err(ids: &[u32], wts: &[f32], want_ids: &[u32], want_wts: &[f32], k: usize) -> f32 {
    let mut err = 0f32;
    for (t, (g, w)) in by_id(ids, wts, k)
        .iter()
        .zip(&by_id(want_ids, want_wts, k))
        .enumerate()
    {
        let gi: Vec<u32> = g.iter().map(|p| p.0).collect();
        let wi: Vec<u32> = w.iter().map(|p| p.0).collect();
        assert_eq!(gi, wi, "token {t}: expert sets differ");
        err = g
            .iter()
            .zip(w)
            .fold(err, |e, (a, b)| e.max((a.1 - b.1).abs()));
    }
    err
}

// ---- component parity vs HF (gate R1) --------------------------------------------------------

#[test]
fn router_matches_hf() {
    let x = f("x");
    let (ids, wts) = route(&base(10.0), x, x.len() / HID).unwrap();
    let err = route_err(
        &ids,
        &wts,
        &ids_of("router_out.indices"),
        f("router_out.weights"),
        K,
    );
    eprintln!("router: max abs weight err {err:e}");
    assert!(err <= 1e-5, "router weights max abs err {err}");
}

#[test]
fn group_limited_router_matches_hf() {
    let p = MoeParams {
        n_group: 4,
        topk_group: 2,
        router: Some(W::F32(f("router_g4.weight"))),
        bias: Some(W::F32(f("router_g4.bias"))),
        ..base(10.0)
    };
    let x = f("x");
    let t_len = x.len() / HID;
    let (ids, wts) = route(&p, x, t_len).unwrap();
    let err = route_err(
        &ids,
        &wts,
        &ids_of("router_g4_out.indices"),
        f("router_g4_out.weights"),
        K,
    );
    eprintln!("group router: max abs weight err {err:e}");
    assert!(err <= 1e-5, "group router weights max abs err {err}");
    // the limit must change some choice on this fixture, or the test proves nothing
    let (plain, _) = route(
        &MoeParams {
            n_group: 1,
            topk_group: 1,
            ..p
        },
        x,
        t_len,
    )
    .unwrap();
    assert_ne!(plain, ids, "group limit is a no-op on this fixture");
}

/// The experts alone, on HF's own routing.
fn experts_err(gu: &str, down: &str, out: &str, limit: f32) -> f32 {
    let p = experts_params(gu, down, limit);
    let x = f("x");
    let y = experts(
        &p,
        x,
        x.len() / HID,
        &ids_of("router_out.indices"),
        f("router_out.weights"),
    )
    .unwrap();
    max_err(&y, f(out))
}

#[test]
fn experts_match_hf() {
    let err = experts_err(
        "experts.gate_up_proj",
        "experts.down_proj",
        "experts_out",
        10.0,
    );
    eprintln!(
        "experts: max abs err {err:e} (output max {:.2})",
        max_abs(f("experts_out"))
    );
    assert!(err <= 1e-4, "experts max abs err {err}");
}

#[test]
fn clamped_experts_match_hf() {
    let err = experts_err(
        "experts_l1.gate_up_proj",
        "experts_l1.down_proj",
        "experts_l1_out",
        1.0,
    );
    eprintln!("experts, limit 1: max abs err {err:e}");
    assert!(err <= 1e-4, "limit-1 experts max abs err {err}");
    let off = experts_err(
        "experts_l1.gate_up_proj",
        "experts_l1.down_proj",
        "experts_l1_out",
        f32::INFINITY,
    );
    assert!(
        off > 1e-2,
        "the clamp does not bite on this fixture ({off})"
    );
}

#[test]
fn shared_expert_matches_hf() {
    let p = MoeParams {
        ff_shexp: 64,
        sh_gate: Some(W::F32(f("shared.gate_proj"))),
        sh_up: Some(W::F32(f("shared.up_proj"))),
        sh_down: Some(W::F32(f("shared.down_proj"))),
        ..base(10.0)
    };
    let x = f("x");
    let err = max_err(
        &shared_expert(&p, x, x.len() / HID).unwrap(),
        f("shared_out"),
    );
    eprintln!("shared expert: max abs err {err:e}");
    assert!(err <= 1e-4, "shared expert max abs err {err}");
}

#[test]
fn full_moe_matches_hf() {
    let x = f("x");
    let err = max_err(
        &ffn(&full_moe_params(), x, x.len() / HID).unwrap(),
        f("moe_out"),
    );
    eprintln!(
        "full moe layer: max abs err {err:e} (output max {:.2})",
        max_abs(f("moe_out"))
    );
    assert!(err <= 1e-4, "full moe max abs err {err}");
}

#[test]
fn dense_ffn_matches_hf() {
    let x = f("x");
    for (tag, limit) in [("dense", 10.0), ("dense_l1", 1.0)] {
        let err = max_err(
            &ffn(&dense_params(tag, limit), x, x.len() / HID).unwrap(),
            f(&format!("{tag}_out")),
        );
        eprintln!("dense ffn {tag}: max abs err {err:e}");
        assert!(err <= 1e-4, "{tag} max abs err {err}");
    }
}

#[test]
fn sentinel_expert_is_skipped() {
    let p = experts_params("experts.gate_up_proj", "experts.down_proj", 10.0);
    let x = f("x");
    let t_len = x.len() / HID;
    let mut ids = ids_of("router_out.indices");
    let wts = f("router_out.weights");
    let full = experts(&p, x, t_len, &ids, wts).unwrap();
    // token 0's first pick -> the sentinel: exactly that term disappears
    let e0 = ids[0] as usize;
    ids[0] = NE as u32;
    let y = experts(&p, x, t_len, &ids, wts).unwrap();
    let (g, u, d) = (
        p.gate_exps.unwrap().mat(e0, FF, HID).unwrap(),
        p.up_exps.unwrap().mat(e0, FF, HID).unwrap(),
        p.down_exps.unwrap().mat(e0, HID, FF).unwrap(),
    );
    let term = mlp(&g, &u, &d, FF, HID, &x[..HID], 10.0);
    for c in 0..HID {
        let d = (full[c] - wts[0] * term[c] - y[c]).abs();
        assert!(d <= 1e-5 * (1.0 + full[c].abs()), "col {c}: {d}");
    }
    assert_eq!(&full[HID..], &y[HID..], "other tokens changed");
}

// ---- weight formats ----------------------------------------------------------------------------

/// tang's GGUF decoders against the fixture expected values on random blocks of every type the
/// MoE meets. For all but Q5_0 the expected values are llama.cpp gguf-py's dequantisation; Q5_0's
/// come from the ggml formula, because gguf-py's Q5_0 emits a different element order than ggml
/// (see glm5_moe_fixtures.py — verified against the real tiny GGUF's Q5_0 experts vs HF f32).
/// The CUDA decoders are checked on the same bytes by scripts/glm5_moe_cuda_host.sh.
#[test]
fn decoders_match_expected() {
    for name in ["f16", "q8_0", "q5_0", "q4_k", "q5_k", "q6_k"] {
        let id = entry(&format!("quant_{name}_raw"))["ggml_type"]
            .as_u64()
            .unwrap() as u32;
        let want = f(&format!("quant_{name}_want"));
        let mut got = vec![0f32; want.len()];
        dequantize(
            GgmlType::from_id(id),
            &raw(&format!("quant_{name}_raw")),
            &mut got,
        )
        .unwrap();
        let err = max_err(&got, want);
        assert!(
            err <= 1e-6 * max_abs(want).max(1.0),
            "{name}: tang decoder max abs err {err}"
        );
    }
}

/// The UD-Q4_K_XL expert sizes in the module docs (measured from the shard headers) agree with
/// tang's K-quant block geometry at the real shapes.
#[test]
fn ud_q4_k_xl_expert_bytes() {
    let (e, ff, hid) = (288u64, 2048usize, 4096usize);
    let mat =
        |ty: GgmlType, row: usize, rows: usize| ty.bytes_for(row).unwrap() as u64 * rows as u64 * e;
    let layer = |gu: GgmlType, d: GgmlType| 2 * mat(gu, hid, ff) + mat(d, ff, hid);
    let typical = layer(GgmlType::Q4K, GgmlType::Q5K);
    let l11 = layer(GgmlType::Q5K, GgmlType::Q6K);
    let l12_44 = layer(GgmlType::Q4K, GgmlType::Q6K);
    assert_eq!(typical, 4_378_853_376);
    assert_eq!(l11, 5_303_697_408);
    assert_eq!(l12_44, 4_699_717_632);
    assert_eq!(40 * typical + l11 + 2 * l12_44, 189_857_267_712);
}

// ---- the tiny GLM-5 GGUF (W0) through MoeParams::from_gguf -------------------------------------

fn tiny(file: &str) -> Option<&'static Gguf> {
    let d = std::env::var("TANG_GLM5_TINY").unwrap_or_else(|_| "/private/tmp/glm5-tiny".into());
    let p = Path::new(&d).join(file);
    if !p.exists() {
        eprintln!("{}: absent, skipping", p.display());
        return None;
    }
    Some(Box::leak(Box::new(Gguf::open(&p).unwrap())))
}

#[test]
fn tiny_gguf_layers_match_hf() {
    let Some(g) = tiny("glm5-tiny-f32.gguf") else {
        return;
    };
    let x = f("tiny_x");
    let dense = MoeParams::from_gguf(g, 0).unwrap();
    assert!(dense.is_dense() && dense.ff_dense == 768 && dense.limit_shexp == 10.0);
    let hid = dense.hidden;
    let t_len = x.len() / hid;
    let err = max_err(&ffn(&dense, x, t_len).unwrap(), f("tiny_l0_dense_out"));
    eprintln!("tiny gguf layer 0 dense: max abs err {err:e}");
    assert!(err <= 1e-4, "tiny layer 0 dense max abs err {err}");

    let moe = MoeParams::from_gguf(g, 2).unwrap();
    assert!(!moe.is_dense());
    assert_eq!(
        (moe.n_expert, moe.top_k, moe.ff_exp, moe.ff_shexp),
        (8, 2, 128, 128)
    );
    assert_eq!(
        (
            moe.scale,
            moe.norm_topk_prob,
            moe.limit_exp,
            moe.limit_shexp
        ),
        (2.5, true, 10.0, 10.0)
    );
    let (ids, wts) = route(&moe, x, t_len).unwrap();
    let rerr = route_err(
        &ids,
        &wts,
        &ids_of("tiny_l2_router_out.indices"),
        f("tiny_l2_router_out.weights"),
        2,
    );
    let err = max_err(&ffn(&moe, x, t_len).unwrap(), f("tiny_l2_moe_out"));
    eprintln!("tiny gguf layer 2 moe: router weight err {rerr:e}, output max abs err {err:e}");
    assert!(
        rerr <= 1e-5 && err <= 1e-4,
        "tiny layer 2: router {rerr}, output {err}"
    );
    assert_eq!(expert_bytes(g, 2).unwrap(), 3 * 8 * 128 * 256 * 4);
}

/// The quantised tiny GGUFs through the same CPU path: measures the quantisation error vs HF
/// (not a parity gate; a sanity bound that the decoders feed the experts correctly).
#[test]
fn tiny_quantised_gguf_layers_close_to_hf() {
    let x = f("tiny_x");
    for (file, tol) in [
        ("glm5-tiny-q8_0.gguf", 0.02),
        ("glm5-tiny-q4_k_m.gguf", 0.2),
    ] {
        let Some(g) = tiny(file) else { continue };
        for (l, want) in [(0, "tiny_l0_dense_out"), (2, "tiny_l2_moe_out")] {
            let p = MoeParams::from_gguf(g, l).unwrap();
            let want = f(want);
            let rel = max_err(&ffn(&p, x, x.len() / p.hidden).unwrap(), want) / max_abs(want);
            eprintln!("{file} layer {l}: max abs err / max |y| = {rel:.4}");
            assert!(rel <= tol, "{file} layer {l}: relative err {rel}");
        }
    }
}

// ---- GPU (gate R3 prep): naive kernels vs this CPU reference ------------------------------------

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use tang_verify::stat;
    use crate::flash::glm5::moe::gpu::{ffn as gpu_ffn, ffn_host, DevMoe, Kernels};
    use std::time::Instant;
    use tang_moe::gpu::{DevBuf, Gpu, Stream};

    /// Every GPU path vs the CPU reference: naive and fast with two-level int8 activations
    /// (q8x2, the default) within `tol`, fast with one-level int8 (q8) within `tol_fast` (all
    /// relative to max(1, max |y|)). F32 / F16 weights take f32 activations on the fast path
    /// too; quantized weights take int8 (q8 ≈1e-2 relative, q8x2 ≈1e-5).
    fn check2(name: &str, p: &MoeParams, x: &[f32], tol: f32, tol_fast: f32) {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let mut k = Kernels::new(&g).unwrap();
        let m = DevMoe::new(p).unwrap();
        let t_len = x.len() / p.hidden;
        let want = ffn(p, x, t_len).unwrap();
        for (mode, fast, act2, tl) in [
            ("naive", false, true, tol),
            ("fast q8x2", true, true, tol),
            ("fast q8", true, false, tol_fast),
        ] {
            k.fast = fast;
            k.act2 = act2;
            k.poison = fast;
            let got = ffn_host(&k, &m, x, t_len).unwrap();
            let err = max_err(&got, &want);
            let s = stat(&got, &want, 1e-3);
            eprintln!(
                "gpu {mode} {name}: max abs err vs cpu {err:e}, rel l2 {:.3e}, nonfinite {}",
                s.rel_l2, s.nonfinite_gpu
            );
            assert!(s.nonfinite_gpu == 0, "gpu {mode} {name}: {} non-finite", s.nonfinite_gpu);
            assert!(
                err <= tl * max_abs(&want).max(1.0),
                "gpu {mode} {name}: max abs err {err}"
            );
            if fast && !p.is_dense() {
                // the fast router/select picks the CPU's experts
                let (ci, cw) = route(p, x, t_len).unwrap();
                let (gi, gw) = k.last_routing(t_len, p.top_k).unwrap().unwrap();
                let werr = route_err(&gi, &gw, &ci, &cw, p.top_k);
                assert!(werr <= 1e-5, "gpu fast {name}: routing weights err {werr}");
            }
        }
    }

    fn check(name: &str, p: &MoeParams, x: &[f32], tol: f32) {
        check2(name, p, x, tol, tol)
    }

    #[test]
    fn gpu_full_moe_matches_cpu() {
        check("full moe", &full_moe_params(), f("x"), 1e-5);
    }

    #[test]
    fn gpu_clamped_experts_match_cpu() {
        // router + limit-1 experts, no shared expert
        check(
            "limit-1 experts",
            &experts_params("experts_l1.gate_up_proj", "experts_l1.down_proj", 1.0),
            f("x"),
            1e-5,
        );
    }

    #[test]
    fn gpu_group_router_moe_matches_cpu() {
        let p = MoeParams {
            n_group: 4,
            topk_group: 2,
            router: Some(W::F32(f("router_g4.weight"))),
            bias: Some(W::F32(f("router_g4.bias"))),
            ..experts_params("experts.gate_up_proj", "experts.down_proj", 10.0)
        };
        check("group-limited moe", &p, f("x"), 1e-5);
    }

    #[test]
    fn gpu_dense_matches_cpu() {
        check("dense", &dense_params("dense", 10.0), f("x"), 1e-5);
        check(
            "dense limit 1",
            &dense_params("dense_l1", 1.0),
            f("x"),
            1e-5,
        );
    }

    /// Q8_0 / Q4_K / Q5_K / Q6_K weights decoded on the GPU vs tang's CPU decoders (same GGUF).
    #[test]
    fn gpu_tiny_quantised_gguf_matches_cpu() {
        for (file, tol_fast) in [
            ("glm5-tiny-f32.gguf", 1e-4),
            ("glm5-tiny-q8_0.gguf", 3e-2),
            ("glm5-tiny-q4_k_m.gguf", 3e-2),
        ] {
            let Some(g) = tiny(file) else { continue };
            for l in [0, 2] {
                let p = MoeParams::from_gguf(g, l).unwrap();
                check2(&format!("{file} layer {l}"), &p, f("tiny_x"), 2e-4, tol_fast);
            }
        }
    }

    /// Deterministic pseudo-normal activations (Irwin–Hall of 4 uniforms), a few outliers.
    fn synth_x(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        (0..n)
            .map(|i| {
                let v = (u() + u() + u() + u() - 2.0) * 1.7;
                if i % 997 == 0 { v * 8.0 } else { v }
            })
            .collect()
    }

    /// Mean µs of `f` over `n` runs (after one warm-up).
    fn time_us(n: usize, mut f: impl FnMut()) -> f64 {
        f();
        let t0 = Instant::now();
        for _ in 0..n {
            f();
        }
        t0.elapsed().as_secs_f64() * 1e6 / n as f64
    }

    /// REAL GLM-5.3 layers, one layer's FFN resident at a time (a 24 GB card can't hold all
    /// 45): naive and fast GPU vs the CPU reference on the same x (full [t][hidden] tensors,
    /// routing sets), and per-layer µs of both (device-resident x / y, sync per call — what a
    /// decode step pays per layer). `TANG_GLM5_REAL_GGUF=<shard 1>`;
    /// `TANG_GLM5_MOE_BENCH_LAYERS` (default 0,4,11,12: dense Q8_0; Q4_K/Q5_K; Q5_K/Q6_K;
    /// Q4_K/Q6_K); `TANG_GLM5_MOE_BENCH_T` (default 1,8).
    #[test]
    fn real_layers_fast_matches_cpu() {
        let Some(path) = std::env::var("TANG_GLM5_REAL_GGUF").ok() else {
            eprintln!("skip: TANG_GLM5_REAL_GGUF not set");
            return;
        };
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let gg = Gguf::open(Path::new(&path)).unwrap();
        let mut k = Kernels::new(&g).unwrap();
        let list = |key: &str, d: &str| -> Vec<usize> {
            std::env::var(key)
                .unwrap_or_else(|_| d.into())
                .split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect()
        };
        let mut worst_fast = 0f64;
        for l in list("TANG_GLM5_MOE_BENCH_LAYERS", "0,4,11,12") {
            let p = MoeParams::from_gguf(&gg, l).unwrap();
            let t_up = Instant::now();
            let m = DevMoe::new(&p).unwrap();
            let tys = |w: &Option<W>| match w {
                Some(W::Gguf(_, t)) => t.ty.name().to_string(),
                _ => "-".into(),
            };
            eprintln!(
                "moebench l{l} {} gate {} up {} down {} sh {}/{}/{} (upload {:.1} s)",
                if p.is_dense() { "dense" } else { "moe" },
                tys(if p.is_dense() { &p.dense_gate } else { &p.gate_exps }),
                tys(if p.is_dense() { &p.dense_up } else { &p.up_exps }),
                tys(if p.is_dense() { &p.dense_down } else { &p.down_exps }),
                tys(&p.sh_gate),
                tys(&p.sh_up),
                tys(&p.sh_down),
                t_up.elapsed().as_secs_f64()
            );
            for t_len in list("TANG_GLM5_MOE_BENCH_T", "1,8") {
                let hid = p.hidden;
                let x = synth_x(t_len * hid, (l * 1000 + t_len) as u64);
                let t_c = Instant::now();
                let want = ffn(&p, &x, t_len).unwrap();
                let cpu_ms = t_c.elapsed().as_secs_f64() * 1e3;
                let xd = DevBuf::from_slice(&x).unwrap();
                let yd = DevBuf::zeroed(x.len() * 4).unwrap();
                let s = Stream::blocking().unwrap();
                let mut outs = Vec::new();
                for (mode, fast, act2) in
                    [("naive", false, true), ("fast-q8x2", true, true), ("fast-q8", true, false)]
                {
                    k.fast = fast;
                    k.act2 = act2;
                    k.poison = fast;
                    let got = ffn_host(&k, &m, &x, t_len).unwrap();
                    let st = stat(&got, &want, 2e-2);
                    k.poison = false;
                    let reps = if fast { 50 } else { 3 };
                    let us = time_us(reps, || gpu_ffn(&k, &s, &m, xd.ptr, yd.ptr, t_len).unwrap());
                    let us_host = time_us(reps, || {
                        ffn_host(&k, &m, &x, t_len).unwrap();
                    });
                    eprintln!(
                        "moebench l{l} T{t_len} {mode:<9} rel {:.3e} cos {:.7} max {:.3e} nonfinite {} | {:.0} us/call (device x/y), {:.0} us/call (host x/y) | cpu {:.0} ms",
                        st.rel_l2, st.cos, st.max_abs, st.nonfinite_gpu, us, us_host, cpu_ms
                    );
                    assert_eq!(st.nonfinite_gpu, 0, "l{l} {mode}: non-finite output");
                    let bar = if !fast { 1e-4 } else if act2 { 1e-3 } else { 1e-1 };
                    assert!(st.rel_l2 <= bar, "l{l} T{t_len} {mode}: rel l2 {} > {bar}", st.rel_l2);
                    if fast && act2 {
                        worst_fast = worst_fast.max(st.rel_l2);
                        if !p.is_dense() {
                            let (ci, _) = route(&p, &x, t_len).unwrap();
                            let (gi, _) = k.last_routing(t_len, p.top_k).unwrap().unwrap();
                            let sets = |v: &[u32]| {
                                v.chunks(p.top_k)
                                    .map(|c| {
                                        let mut c = c.to_vec();
                                        c.sort();
                                        c
                                    })
                                    .collect::<Vec<_>>()
                            };
                            assert_eq!(sets(&gi), sets(&ci), "l{l} T{t_len}: fast routing differs");
                        }
                    }
                    outs.push(got);
                }
                for (i, name) in [(1, "fast-q8x2"), (2, "fast-q8")] {
                    let ab = stat(&outs[i], &outs[0], 2e-2);
                    eprintln!("moebench l{l} T{t_len} {name}-vs-naive rel {:.3e}", ab.rel_l2);
                }
            }
        }
        eprintln!("moebench worst fast-q8x2 rel l2 {worst_fast:.3e}");
    }
}
