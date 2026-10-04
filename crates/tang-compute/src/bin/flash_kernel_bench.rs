//! `flash-kernel-bench`: GPU timings for the Qwen3.8-Flash-Next decode kernels on random
//! weights at the real shapes.
//!
//! ```text
//! cargo run --release -p tang-compute --features cuda --bin flash-kernel-bench -- gemv
//! cargo run --release -p tang-compute --features cuda --bin flash-kernel-bench -- window [T..]
//! ```
//!
//! `gemv`: GB/s of weight bytes for bf16, tang-Q4 (group 64) and Q2_0 GEMVs at T = 1 and 4,
//! timed inside CUDA graphs over enough weight copies to defeat the L2.
//!
//! `window`: one synthetic 48-layer verify window (36 GDN + 12 QSA layers, two
//! hyper-connection reads per layer with the writes fused in, router, 10 routed experts per
//! token from a VRAM pool plus a Q2 shared expert, final read and head), eager and as one
//! captured graph, with GPU time per op class (each class captured alone) and the GDN commit.
//! Dense mixer weights and the head are Q4X (tang-Q4 values, int8 activations),
//! hyper-connections and router bf16, routed and shared experts Q2_0.

use std::time::Instant;

use tang_compute::cuda::CudaBuffer;
use tang_compute::flash::shape::*;
use tang_compute::flash::{
    self, ExpertBlob, GdnMode, GdnParams, HcPending, HcWeights, MoePlan, QAct, QsaCache, QsaNorms,
};
use tang_compute::{ComputeDevice, CudaComputeDevice};

type B = CudaBuffer;

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self, s: f32) -> f32 {
        ((self.u() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * s
    }
    fn vec(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.f(s)).collect()
    }
}

fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Random host data reused (sliced) for every weight, so building 3.5 GB is quick.
struct Src {
    words: Vec<u32>,
    bf: Vec<u16>,
}

impl Src {
    fn new(rng: &mut Rng, n: usize) -> Self {
        let words = (0..n).map(|_| rng.u() as u32).collect();
        let bf = (0..n).map(|_| bf16(rng.f(0.05))).collect();
        Src { words, bf }
    }
    fn bf16(&self, g: &CudaComputeDevice, n: usize) -> B {
        g.upload_bf16(&self.bf[..n])
    }
    fn q4(&self, g: &CudaComputeDevice, n: usize, k: usize) -> B {
        let gs = n * k / 64;
        let scales = vec![bf16(0.002); gs];
        let biases = vec![bf16(-0.015); gs];
        g.upload_q4(&self.words[..n * k / 8], &scales, &biases, 64)
    }
    fn q4x(&self, g: &CudaComputeDevice, n: usize, k: usize) -> B {
        let gs = n * k / 64;
        let scales = vec![bf16(0.002); gs];
        let biases = vec![bf16(-0.015); gs];
        g.upload_q4x(&self.words[..n * k / 8], &scales, &biases, n, k)
    }
    fn q8_raw(&self, n: usize, k: usize) -> Vec<u8> {
        let mut raw = Vec::with_capacity(n * k / 32 * 34);
        let mut i = 0;
        for _ in 0..n * k / 32 {
            raw.extend(flash::f32_to_f16(0.002).to_le_bytes());
            for _ in 0..8 {
                raw.extend(self.words[i % self.words.len()].to_le_bytes());
                i += 1;
            }
        }
        raw
    }
    fn q2_raw(&self, n: usize, k: usize) -> Vec<u8> {
        let mut raw = Vec::with_capacity(flash::q2_bytes(n, k));
        let mut i = 0;
        for _ in 0..n * k / 64 {
            raw.extend(flash::f32_to_f16(0.004).to_le_bytes());
            for _ in 0..4 {
                raw.extend(self.words[i % self.words.len()].to_le_bytes());
                i += 1;
            }
        }
        raw
    }
}

fn q4_bytes(n: usize, k: usize) -> usize {
    n * k / 2 + n * k / 64 * 4
}

// ---- GEMV bandwidth ----

fn gemv(g: &CudaComputeDevice) {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let src = Src::new(&mut rng, 248_320 * 2560 / 8 + 16);
    println!(
        "GEMV GB/s of weight bytes (graph of launches over weight copies > L2; RTX 3090 peak 936)"
    );
    println!(
        "{:<18} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "shape", "fmt", "MB", "T=1 us", "T=1 GB/s", "T=4 GB/s"
    );
    for &(k, n) in &[
        (2560, 10240),
        (10240, 320),
        (2560, 2560),
        (2560, 248_320),
        (2560, GDN_PROJ),
        (6144, 2560),
        (2560, 513),
    ] {
        for fmt in ["bf16", "q4", "q4x", "q8x", "q2"] {
            let bytes = match fmt {
                "bf16" => n * k * 2,
                "q4" | "q4x" => q4_bytes(n, k),
                "q8x" => n * k / 32 * 34,
                _ => flash::q2_bytes(n, k),
            };
            if (fmt == "bf16" || fmt == "q8x" || fmt == "q4") && n * k > 200_000_000 {
                // 1.3 GB of bf16 head: skip, the 2560-wide shapes show the kernel's rate.
                continue;
            }
            let copies = (96usize << 20).div_ceil(bytes).clamp(1, 32);
            let ws: Vec<B> = (0..copies)
                .map(|_| match fmt {
                    "bf16" => src.bf16(g, n * k),
                    "q4" => src.q4(g, n, k),
                    "q4x" => src.q4x(g, n, k),
                    "q8x" => g.upload_q8x(&src.q8_raw(n, k), n, k),
                    _ => g.upload_q2(&src.q2_raw(n, k), n, k),
                })
                .collect();
            let mut line = format!(
                "{:<18} {:>6} {:>10.2}",
                format!("{k}->{n}"),
                fmt,
                bytes as f64 / 1e6
            );
            for t in [1usize, 4] {
                let x = g.upload_f32(&rng.vec(t * k, 1.0));
                let mut xq = g.alloc_f32(QAct { m: t, k }.words());
                g.quantize_act_into(&x, &mut xq, t, k);
                let mut y = g.alloc_f32(t * n);
                let reps = (copies * 4).max(8);
                let graph = g.capture(&mut || {
                    for i in 0..reps {
                        let w = &ws[i % copies];
                        if fmt == "q2" {
                            g.q2_linear_into(&xq, w, &mut y, t, k, n);
                        } else if fmt == "q4x" {
                            g.q4x_linear_into(&xq, w, &mut y, t, k, n);
                        } else if fmt == "q8x" {
                            g.q8x_linear_into(&xq, w, &mut y, t, k, n);
                        } else {
                            g.linear_into(&x, w, &mut y, t, k, n);
                        }
                    }
                });
                graph.launch().unwrap();
                let ms = (0..3)
                    .map(|_| g.event_ms(&mut || graph.launch().unwrap()))
                    .fold(f32::INFINITY, f32::min);
                let us = ms as f64 * 1e3 / reps as f64;
                let gbs = bytes as f64 / (us * 1e3);
                if t == 1 {
                    line += &format!(" {:>10.1} {:>10.0}", us, gbs);
                } else {
                    line += &format!(" {:>10.0}", gbs);
                }
            }
            println!("{line}");
        }
    }
}

// ---- 48-layer window ----

struct Hc {
    norm: B,
    down: B,
    up: B,
    inject: B,
}

impl Hc {
    fn new(g: &CudaComputeDevice, src: &Src) -> Self {
        Hc {
            norm: g.upload_f32(&vec![1.0; HC * HIDDEN]),
            down: src.bf16(g, HC_LR * HC * HIDDEN),
            up: src.bf16(g, HC * HIDDEN * HC_LR),
            inject: src.bf16(g, HC * HC * HIDDEN),
        }
    }
    fn w(&self, inject: bool) -> HcWeights<'_, B> {
        HcWeights {
            norm: &self.norm,
            down: &self.down,
            up: &self.up,
            inject: inject.then_some(&self.inject),
        }
    }
    const BYTES: usize = 2 * (HC_LR * HC * HIDDEN * 2) + HC * HC * HIDDEN * 2;
}

struct Gdn {
    conv: B,
    dt: B,
    a: B,
    norm: B,
    state: B,
    hist: B,
}

struct Qsa {
    norms: [B; 4],
    k: B,
    v: B,
    ring: B,
    pooled: B,
}

struct Layer {
    w_in: B,
    in_rows: usize,
    w_out: B,
    hc_a: Hc,
    hc_f: Hc,
    router: B,
    table: B,
    shared: u64,
    ids: Vec<B>, // per T index
    gdn: Option<Gdn>,
    qsa: Option<Qsa>,
}

const MAX_CTX: usize = 8192;
const CTX: usize = 4096;
const ROUTER_ROWS: usize = EXPERTS + 1;
const VOCAB: usize = 248_320;

#[derive(Clone, Copy, PartialEq)]
enum Class {
    Hc,
    Dense,
    Gdn,
    Qsa,
    Moe,
}
const CLASSES: [(Class, &str); 5] = [
    (Class::Hc, "hyper-connections"),
    (Class::Dense, "dense GEMVs (mixers, router, head)"),
    (Class::Gdn, "GDN conv + step"),
    (Class::Qsa, "QSA prep + select + attend"),
    (Class::Moe, "MoE route + plan + experts + combine"),
];

struct Scratch {
    win: B,
    r: B,
    x: B,
    x2: B,
    inj_a: B,
    inj_f: B,
    hc: B,
    proj: B,
    h: B,
    y: B,
    mix: B,
    logits: B,
    ids_r: B,
    w: B,
    xq: B,
    yq: B,
    plan: B,
    moe: B,
    parts: B,
    q: B,
    scores: B,
    sel: B,
    attn_s: B,
    attn: B,
    head: B,
}

struct Model {
    layers: Vec<Layer>,
    hc_out: Hc,
    head: B,
    rope: (B, B),
    _pool: Vec<B>,
}

/// Routed experts for a window: token 0 picks 10 fresh experts, later tokens 6-7 fresh and the
/// rest from the window so far, so the union is 10 / 17 / 31 at T = 1 / 2 / 4 (Strata measures
/// a window's misses at 1.75x and 3.05x one token's at T = 2 and 4).
fn synthetic_ids(rng: &mut Rng, t: usize) -> (Vec<u32>, usize) {
    let mut union: Vec<u32> = vec![];
    let mut ids = vec![];
    for tt in 0..t {
        let fresh = if tt == 0 {
            TOPK
        } else {
            [7, 7, 7, 6, 6, 6, 6][tt - 1]
        };
        let mut row: Vec<u32> = vec![];
        while row.len() < fresh {
            let e = (rng.u() % EXPERTS as u64) as u32;
            if !union.contains(&e) && !row.contains(&e) {
                row.push(e);
            }
        }
        while row.len() < TOPK {
            let e = union[(rng.u() as usize) % union.len()];
            if !row.contains(&e) {
                row.push(e);
            }
        }
        union.extend(
            row.iter()
                .filter(|e| !union.contains(e))
                .collect::<Vec<_>>(),
        );
        ids.extend(row);
    }
    (ids, union.len())
}

fn build(g: &CudaComputeDevice, ts: &[usize]) -> (Model, Vec<usize>) {
    let mut rng = Rng(0xfeed_beef_0bad_cafe);
    let t0 = Instant::now();
    let src = Src::new(&mut rng, VOCAB * HIDDEN / 8 + 16);
    // 512 resident expert blobs, shared by every layer's table (distinct per layer by a shift).
    let blob_src: Vec<u8> = {
        let mut b = src.q2_raw(FF, HIDDEN);
        b.extend(src.q2_raw(FF, HIDDEN));
        b.extend(src.q2_raw(HIDDEN, FF));
        // Already repacked-size; content is random either way.
        b
    };
    assert_eq!(blob_src.len(), ExpertBlob::BYTES);
    let pool: Vec<B> = (0..EXPERTS).map(|_| g.upload_bytes(&blob_src)).collect();
    let addrs: Vec<u64> = pool.iter().map(|b| g.buffer_addr(b)).collect();
    let mut unions = vec![0usize; ts.len()];
    let mut layers = vec![];
    for l in 0..48 {
        let is_qsa = l % 4 == 3;
        let in_rows = if is_qsa { QSA_PROJ } else { GDN_PROJ };
        let table: Vec<u32> = (0..EXPERTS)
            .flat_map(|e| {
                let p = addrs[(e + 37 * l) % EXPERTS];
                [p as u32, (p >> 32) as u32]
            })
            .collect();
        let shared_blob = g.upload_bytes(&blob_src);
        let shared = g.buffer_addr(&shared_blob);
        let ids = ts
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let (ids, u) = synthetic_ids(&mut rng, t);
                unions[i] += u;
                g.upload_u32(&ids)
            })
            .collect();
        let gdn = (!is_qsa).then(|| Gdn {
            conv: g.upload_f32(&rng.vec(GDN_CONV * GDN_TAPS, 0.5)),
            dt: g.upload_f32(&rng.vec(GDN_HV, 1.0)),
            a: g.upload_f32(&[-0.5; GDN_HV]),
            norm: g.upload_f32(&vec![1.0; GDN_D]),
            state: g.upload_f32(&rng.vec(flash::GDN_STATE, 0.1)),
            hist: g.upload_f32(&rng.vec(flash::GDN_HIST, 0.5)),
        });
        let qsa = is_qsa.then(|| Qsa {
            norms: [QSA_D, QSA_D, IDX_D, IDX_D].map(|n| g.upload_f32(&vec![1.0; n])),
            k: g.upload_bf16(&src.bf[..MAX_CTX * QSA_KV * QSA_D]),
            v: g.upload_bf16(&src.bf[..MAX_CTX * QSA_KV * QSA_D]),
            ring: g.upload_f32(&rng.vec(16 * IDX_D, 1.0)),
            pooled: g.upload_f32(&rng.vec(MAX_CTX / 4 * IDX_D, 1.0)),
        });
        layers.push(Layer {
            w_in: src.q4x(g, in_rows, HIDDEN),
            in_rows,
            w_out: src.q4x(g, HIDDEN, if is_qsa { QSA_OUT } else { GDN_V }),
            hc_a: Hc::new(g, &src),
            hc_f: Hc::new(g, &src),
            router: src.bf16(g, ROUTER_ROWS * HIDDEN),
            table: g.upload_u32(&table),
            shared,
            ids,
            gdn,
            qsa,
        });
        std::mem::forget(shared_blob); // lives as long as the process
    }
    let model = Model {
        layers,
        hc_out: Hc::new(g, &src),
        head: src.q4x(g, VOCAB, HIDDEN),
        rope: {
            let (c, s) = flash::rope_table(MAX_CTX, 1e7);
            (g.upload_f32(&c), g.upload_f32(&s))
        },
        _pool: pool,
    };
    g.sync();
    eprintln!("built 48 layers in {:.1} s", t0.elapsed().as_secs_f64());
    (model, unions.iter().map(|u| u / 48).collect())
}

fn scratch(g: &CudaComputeDevice, t: usize) -> Scratch {
    let z = |n: usize| g.alloc_f32(n);
    let win = g.upload_u32(&[(CTX - t) as u32, t as u32, 0]);
    Scratch {
        win,
        r: g.upload_f32(&Rng(5).vec(t * HC * HIDDEN, 1.0)),
        x: z(t * HIDDEN),
        x2: z(t * HIDDEN),
        inj_a: z(t * HC),
        inj_f: z(t * HC),
        hc: z(flash::hc_scratch_words(t)),
        proj: z(t * GDN_PROJ.max(QSA_PROJ)),
        h: z(t * GDN_CONV),
        y: z(t * GDN_V),
        mix: z(t * HIDDEN),
        logits: z(t * ROUTER_ROWS),
        ids_r: z(t * TOPK),
        w: z(t * TOPK),
        xq: z(QAct { m: t, k: HIDDEN }.words()),
        yq: z(QAct { m: t, k: GDN_V }.words()),
        plan: z(MoePlan::WORDS),
        moe: z(MoePlan::scratch_words()),
        parts: z(MoePlan::PARTS_ROWS * HIDDEN),
        q: z(flash::qsa_q_words(t)),
        scores: z(t * MAX_CTX / 4),
        sel: z(t * QSA_WIDTH),
        attn_s: z(flash::qsa_attend_scratch_words(t)),
        attn: z(t * QSA_OUT),
        head: z(t * VOCAB),
    }
}

/// Enqueue one verify window (the classes in `only`, or all). Returns kernel launches.
fn window(
    g: &CudaComputeDevice,
    m: &mut Model,
    s: &mut Scratch,
    t: usize,
    ti: usize,
    only: Option<Class>,
) -> usize {
    let on = |c: Class| only.is_none_or(|o| o == c);
    let mut launches = 0;
    let eps = 1e-6;
    let n_layers = m.layers.len();
    for li in 0..n_layers {
        let layer = &mut m.layers[li];
        if on(Class::Hc) {
            let pending = (li > 0).then_some(HcPending::Moe {
                parts: &s.parts,
                w: &s.w,
                logits: &s.logits,
                stride: ROUTER_ROWS,
                sg: Some(EXPERTS),
                inj: &s.inj_f,
            });
            let xq = Some(&mut s.xq);
            g.hc_read_into(
                &mut s.r,
                pending,
                &layer.hc_a.w(true),
                &mut s.x,
                xq,
                Some(&mut s.inj_a),
                &mut s.hc,
                t,
                eps,
            );
            launches += 3;
        }
        if on(Class::Dense) {
            g.q4x_linear_into(&s.xq, &layer.w_in, &mut s.proj, t, HIDDEN, layer.in_rows);
            launches += 1;
        }
        if let Some(gd) = layer.gdn.as_mut() {
            if on(Class::Gdn) {
                let p = GdnParams {
                    conv: &gd.conv,
                    dt_bias: &gd.dt,
                    ssm_a: &gd.a,
                    norm: &gd.norm,
                };
                if unfused() {
                    g.gdn_conv_into(&s.proj, GDN_PROJ, &gd.hist, &gd.conv, &mut s.h, t, eps);
                    let yq = Some(&mut s.yq);
                    g.gdn_step(
                        &mut gd.state,
                        &s.h,
                        &s.proj,
                        GDN_PROJ,
                        &p,
                        &mut s.y,
                        yq,
                        t,
                        GdnMode::ReadOnly,
                        eps,
                    );
                    launches += 2;
                } else {
                    let yq = Some(&mut s.yq);
                    let ro = GdnMode::ReadOnly;
                    g.gdn_conv_step(
                        &mut gd.state,
                        &s.proj,
                        GDN_PROJ,
                        &gd.hist,
                        &p,
                        &mut s.y,
                        yq,
                        t,
                        ro,
                        eps,
                    );
                    launches += 1;
                }
            }
            if on(Class::Dense) {
                g.q4x_linear_into(&s.yq, &layer.w_out, &mut s.mix, t, GDN_V, HIDDEN);
                launches += 1;
            }
        }
        if let Some(qs) = layer.qsa.as_mut() {
            if on(Class::Qsa) {
                let norms = QsaNorms {
                    q: &qs.norms[0],
                    k: &qs.norms[1],
                    iq: &qs.norms[2],
                    ik: &qs.norms[3],
                };
                let cache = QsaCache {
                    k: &mut qs.k,
                    v: &mut qs.v,
                    ring: &mut qs.ring,
                    pooled: &mut qs.pooled,
                };
                g.qsa_prep(
                    &s.proj,
                    QSA_PROJ,
                    &s.win,
                    &norms,
                    (&m.rope.0, &m.rope.1),
                    &mut s.q,
                    cache,
                    t,
                    eps,
                );
                g.qsa_select_into(
                    &qs.pooled,
                    &s.q,
                    &s.win,
                    &mut s.scores,
                    &mut s.sel,
                    MAX_CTX / 4,
                    t,
                );
                g.qsa_attend_into(
                    &s.q,
                    &qs.k,
                    &qs.v,
                    &s.sel,
                    &s.proj,
                    QSA_PROJ,
                    &s.win,
                    &mut s.attn_s,
                    &mut s.attn,
                    Some(&mut s.yq),
                    t,
                );
                launches += 5;
            }
            if on(Class::Dense) {
                g.q4x_linear_into(&s.yq, &layer.w_out, &mut s.mix, t, QSA_OUT, HIDDEN);
                launches += 1;
            }
        }
        if on(Class::Hc) {
            let pending = Some(HcPending::Write {
                y: &s.mix,
                inj: &s.inj_a,
            });
            let xq = Some(&mut s.xq);
            g.hc_read_into(
                &mut s.r,
                pending,
                &layer.hc_f.w(true),
                &mut s.x2,
                xq,
                Some(&mut s.inj_f),
                &mut s.hc,
                t,
                eps,
            );
            launches += 3;
        }
        if on(Class::Dense) {
            g.linear_into(&s.x2, &layer.router, &mut s.logits, t, HIDDEN, ROUTER_ROWS);
            launches += 1;
        }
        if on(Class::Moe) {
            // Router weights from the logits; plan from the synthetic routing (forced ids).
            let forced = Some(&layer.ids[ti]);
            g.moe_route_into(
                &s.logits,
                ROUTER_ROWS,
                EXPERTS,
                forced,
                &layer.table,
                layer.shared,
                &mut s.ids_r,
                &mut s.w,
                &mut s.plan,
                t,
            );
            // SAFETY: every table address and the shared address are live pool blobs.
            unsafe { g.moe_grouped_into(&s.xq, &s.plan, &mut s.moe, &mut s.parts, t) };
            // The combine runs inside the next hyper-connection read (HcPending::Moe).
            launches += 3;
        }
    }
    if on(Class::Hc) {
        let pending = Some(HcPending::Moe {
            parts: &s.parts,
            w: &s.w,
            logits: &s.logits,
            stride: ROUTER_ROWS,
            sg: Some(EXPERTS),
            inj: &s.inj_f,
        });
        let xq = Some(&mut s.xq);
        g.hc_read_into(
            &mut s.r,
            pending,
            &m.hc_out.w(false),
            &mut s.x,
            xq,
            None,
            &mut s.hc,
            t,
            eps,
        );
        launches += 3;
    }
    if on(Class::Dense) {
        g.q4x_linear_into(&s.xq, &m.head, &mut s.head, t, HIDDEN, VOCAB);
        launches += 1;
    }
    launches
}

/// The commit half: every GDN layer replays `n_keep` = T tokens into its state and history.
fn commit(g: &CudaComputeDevice, m: &mut Model, s: &mut Scratch, t: usize) {
    for layer in m.layers.iter_mut() {
        if let Some(gd) = layer.gdn.as_mut() {
            let p = GdnParams {
                conv: &gd.conv,
                dt_bias: &gd.dt,
                ssm_a: &gd.a,
                norm: &gd.norm,
            };
            let commit = GdnMode::Commit { win: &s.win };
            g.gdn_conv_step(
                &mut gd.state,
                &s.proj,
                GDN_PROJ,
                &gd.hist,
                &p,
                &mut s.y,
                None,
                t,
                commit,
                1e-6,
            );
            g.gdn_conv_commit(&mut gd.hist, &s.proj, GDN_PROJ, &s.win, t);
        }
    }
}

fn dense_bytes() -> (usize, [usize; 4]) {
    let gdn = q4_bytes(GDN_PROJ, HIDDEN) + q4_bytes(HIDDEN, GDN_V);
    let qsa = q4_bytes(QSA_PROJ, HIDDEN) + q4_bytes(HIDDEN, QSA_OUT);
    let hc = 96 * Hc::BYTES + (2 * HC_LR * HC * HIDDEN * 2);
    let router = 48 * ROUTER_ROWS * HIDDEN * 2;
    let shared = 48 * ExpertBlob::BYTES;
    let head = q4_bytes(VOCAB, HIDDEN);
    let mixers = 36 * gdn + 12 * qsa;
    (
        mixers + hc + router + shared + head,
        [mixers, hc, router + shared, head],
    )
}

/// The fastest of the repeats: other processes on a shared GPU only ever add time, in bursts,
/// so the minimum is the uncontended figure (the median is reported where it matters).
fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    if std::env::var("FKB_MEDIAN").is_ok_and(|x| x == "1") {
        v[v.len() / 2]
    } else {
        v[0]
    }
}

fn run_window(g: &CudaComputeDevice, ts: &[usize]) {
    let (mut m, unions) = build(g, ts);
    let (dense, parts) = dense_bytes();
    println!(
        "dense weight bytes per window: {:.3} GB (mixers {:.3}, hyper-connections {:.3}, router + shared expert {:.3}, head {:.3}); SOL at 936 GB/s: {:.2} ms",
        dense as f64 / 1e9,
        parts[0] as f64 / 1e9,
        parts[1] as f64 / 1e9,
        parts[2] as f64 / 1e9,
        parts[3] as f64 / 1e9,
        dense as f64 / 936e6
    );
    for (ti, &t) in ts.iter().enumerate() {
        let mut s = scratch(g, t);
        let expert = unions[ti] * 48 * ExpertBlob::BYTES;
        let launches = window(g, &mut m, &mut s, t, ti, None);
        g.sync();
        // Eager: host launches into the stream.
        let mut eager = vec![];
        let mut wall = vec![];
        for _ in 0..5 {
            let t0 = Instant::now();
            eager.push(g.event_ms(&mut || {
                window(g, &mut m, &mut s, t, ti, None);
            }));
            g.sync();
            wall.push(t0.elapsed().as_secs_f32() * 1e3);
        }
        let graph = g.capture(&mut || {
            window(g, &mut m, &mut s, t, ti, None);
        });
        graph.launch().unwrap();
        let graph_ms = median(
            (0..20)
                .map(|_| g.event_ms(&mut || graph.launch().unwrap()))
                .collect(),
        );
        let total = dense + expert;
        println!();
        println!(
            "T={t}: {launches} launches; routed experts {} distinct/layer = {:.3} GB; dense + experts {:.3} GB, SOL {:.2} ms",
            unions[ti],
            expert as f64 / 1e9,
            total as f64 / 1e9,
            total as f64 / 936e6
        );
        println!(
            "  eager  {:7.2} ms GPU ({:.2} ms wall)   graph {:7.2} ms   -> dense-only {:.0} GB/s, dense+experts {:.0} GB/s ({:.0}% of 936)",
            median(eager),
            median(wall),
            graph_ms,
            dense as f64 / (graph_ms as f64 * 1e6),
            total as f64 / (graph_ms as f64 * 1e6),
            total as f64 / (graph_ms as f64 * 1e6) / 9.36
        );
        let mut sum = 0.0;
        for (c, name) in CLASSES {
            let graph = g.capture(&mut || {
                window(g, &mut m, &mut s, t, ti, Some(c));
            });
            graph.launch().unwrap();
            let ms = median(
                (0..10)
                    .map(|_| g.event_ms(&mut || graph.launch().unwrap()))
                    .collect(),
            );
            let n = window_launches(c);
            sum += ms;
            println!("  {name:<38} {ms:7.3} ms  ({n} launches)");
        }
        println!("  {:<38} {sum:7.3} ms", "sum of classes");
        let graph = g.capture(&mut || commit(g, &mut m, &mut s, t));
        let ms = median(
            (0..10)
                .map(|_| g.event_ms(&mut || graph.launch().unwrap()))
                .collect(),
        );
        println!(
            "  {:<38} {ms:7.3} ms  (72 launches, separate graph)",
            "GDN commit, n_keep = T"
        );
    }
}

fn window_launches(c: Class) -> usize {
    match c {
        Class::Hc => 96 * 3 + 3,
        Class::Dense => 48 * 3 + 1,
        Class::Gdn => 36,
        Class::Qsa => 12 * 5,
        Class::Moe => 48 * 3,
    }
}

/// Routed experts alone: `moe_grouped_into` for 48 layers of synthetic routing (all experts
/// resident, plus the shared expert), in one graph, for T = 1, 2, 4, 8.
fn moe(g: &CudaComputeDevice) {
    let mut rng = Rng(0x5eed);
    let src = Src::new(&mut rng, 4 << 20);
    let blob: Vec<u8> = {
        let mut b = src.q2_raw(FF, HIDDEN);
        b.extend(src.q2_raw(FF, HIDDEN));
        b.extend(src.q2_raw(HIDDEN, FF));
        b
    };
    let pool: Vec<B> = (0..EXPERTS).map(|_| g.upload_bytes(&blob)).collect();
    let addrs: Vec<u64> = pool.iter().map(|b| g.buffer_addr(b)).collect();
    let shared = g.upload_bytes(&blob);
    println!("moe_grouped_into, 48 layers in one graph (GB/s of expert bytes read)");
    for t in [1usize, 2, 4, 8] {
        let mut plans = vec![];
        let mut groups = 0;
        for l in 0..48 {
            let (ids, u) = synthetic_ids(&mut rng, t);
            groups += u + 1;
            let table: Vec<u32> = (0..EXPERTS)
                .flat_map(|e| {
                    let p = addrs[(e + 37 * l) % EXPERTS];
                    [p as u32, (p >> 32) as u32]
                })
                .collect();
            let mut plan = g.alloc_f32(MoePlan::WORDS);
            g.moe_plan_into(
                &g.upload_u32(&ids),
                &g.upload_u32(&table),
                g.buffer_addr(&shared),
                &mut plan,
                t,
            );
            plans.push(plan);
        }
        let mut xq = g.alloc_f32(QAct { m: t, k: HIDDEN }.words());
        g.quantize_act_into(&g.upload_f32(&rng.vec(t * HIDDEN, 1.0)), &mut xq, t, HIDDEN);
        let (mut sc, mut parts) = (
            g.alloc_f32(MoePlan::scratch_words()),
            g.alloc_f32(MoePlan::PARTS_ROWS * HIDDEN),
        );
        let graph = g.capture(&mut || {
            for p in &plans {
                // SAFETY: plans hold live pool addresses.
                unsafe { g.moe_grouped_into(&xq, p, &mut sc, &mut parts, t) };
            }
        });
        graph.launch().unwrap();
        let ms = median(
            (0..10)
                .map(|_| g.event_ms(&mut || graph.launch().unwrap()))
                .collect(),
        );
        let bytes = groups * ExpertBlob::BYTES;
        println!(
            "  T={t}: {:.1} groups/layer, {:6.1} us/layer, {:4.0} GB/s",
            groups as f64 / 48.0,
            ms as f64 * 1e3 / 48.0,
            bytes as f64 / (ms as f64 * 1e6)
        );
    }
}

/// `TANG_FLASH_UNFUSED=1`: the bench's own A/B switch, matching the library's.
fn unfused() -> bool {
    std::env::var("TANG_FLASH_UNFUSED").is_ok_and(|v| v == "1")
}

fn main() {
    let g = CudaComputeDevice::new().expect("CUDA device");
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gemv") => gemv(&g),
        Some("moe") => moe(&g),
        Some("window") => {
            let ts: Vec<usize> = args[1..].iter().map(|a| a.parse().expect("T")).collect();
            run_window(&g, if ts.is_empty() { &[1, 2, 4] } else { &ts });
        }
        _ => eprintln!("usage: flash-kernel-bench gemv | window [T ..]"),
    }
}
