//! Kokoro/TTS-shaped ops derived from the tensor IR, kernel-only on the
//! GPU (5 warmups, 50 launches, one sync — same method as `kernel_only`
//! and `scripts/torch_kokoro_ops.py`). Every op is checked against a
//! straightforward f64 CPU reference first.
//!
//! The ops need zero padding, which the IR expresses as guarded views
//! (`View::guard`): a read outside the source returns 0.
//!
//!   cargo run --release -p tang-bench --example kokoro_ops
use std::time::Instant;

use tang::Scalar;
use tang_bench::ir_gpu::{autotune_kernels, Executable};
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;
use tang_gpu::{GpuDevice, KernelCache};

const ITERS: usize = 50;

fn time_exe(device: &GpuDevice, cache: &mut KernelCache, exe: &Executable, out: TensorId) -> f64 {
    for _ in 0..5 {
        exe.run(device, cache);
    }
    cache.flush(device);
    let _ = exe.read(device, out);
    let t = Instant::now();
    for _ in 0..ITERS {
        exe.run(device, cache);
    }
    cache.flush(device);
    let _ = exe.read(device, out);
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn rand(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = tang_bench::Rng::new(seed);
    (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * scale).collect()
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// max |got − want| / max |want|
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .fold(0.0f64, |m, (&g, &w)| m.max((g as f64 - w).abs()))
        / scale
}

/// `offset + Σ coef·coord` over a `rank`-dim space.
fn form(rank: usize, terms: &[(usize, i64)], offset: i64) -> Form {
    let mut f = Form::zero(rank);
    for &(d, c) in terms {
        f.coefs[d] += c;
    }
    f.offset = offset;
    f
}

fn view(forms: Vec<Form>, guard: bool) -> View {
    View { forms, guard }
}

fn arg(src: TensorId, view: View) -> Arg {
    Arg { src, view }
}

fn lit(v: f64) -> ExprId {
    ExprId::from_f64(v)
}

/// tanh-approximate GELU (matches `F.gelu(approximate="tanh")`)
fn gelu(x: ExprId) -> ExprId {
    let u = lit((2.0 / std::f64::consts::PI).sqrt()) * (x + lit(0.044715) * x * x * x);
    let th = lit(1.0) - lit(2.0) / ((lit(2.0) * u).exp() + lit(1.0));
    lit(0.5) * x * (lit(1.0) + th)
}

// ---------------------------------------------------------------------------
// graphs
// ---------------------------------------------------------------------------

/// `y[o,t] = Σ_{c,k} w[o,c,k] · x[c, t + d·k − p]`, zero padded ("same").
fn conv1d(g: &mut TensorGraph, x: TensorId, w: TensorId, dil: i64) -> TensorId {
    let (ci, t) = (g.shape(x)[0].n(), g.shape(x)[1].n());
    let (co, k) = (g.shape(w)[0].n(), g.shape(w)[2].n());
    let pad = dil * (k as i64 - 1) / 2;
    // space (o, t, c, k)
    let xv = view(
        vec![form(4, &[(2, 1)], 0), form(4, &[(1, 1), (3, dil)], -pad)],
        true,
    );
    let wv = View::pick(4, &[Some(0), Some(2), Some(3)]);
    g.comp(
        &[co, t],
        &[ci, k],
        vec![arg(x, xv), arg(w, wv)],
        Some(Monoid::sum()),
        |v| vec![v[0] * v[1]],
    )
    .unwrap()
}

/// The same conv as unfold (guarded map) + reshape + matmul: im2col.
fn conv1d_im2col(g: &mut TensorGraph, x: TensorId, w: TensorId, dil: i64) -> TensorId {
    let (ci, t) = (g.shape(x)[0].n(), g.shape(x)[1].n());
    let (co, k) = (g.shape(w)[0].n(), g.shape(w)[2].n());
    let pad = dil * (k as i64 - 1) / 2;
    // space (c, k, t)
    let xv = view(
        vec![form(3, &[(0, 1)], 0), form(3, &[(2, 1), (1, dil)], -pad)],
        true,
    );
    let unfold = g
        .comp(&[ci, k, t], &[], vec![arg(x, xv)], None, |v| vec![v[0]])
        .unwrap();
    let u = g.reshape(unfold, &[ci * k, t]).unwrap();
    let wr = g.reshape(w, &[co, ci * k]).unwrap();
    g.matmul(wr, u).unwrap()
}

fn conv_graph(
    c: usize,
    t: usize,
    k: usize,
    dil: i64,
    im2col: bool,
) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[c, t]);
    let w = g.input(&[c, c, k]);
    let y = if im2col {
        conv1d_im2col(&mut g, x, w, dil)
    } else {
        conv1d(&mut g, x, w, dil)
    };
    (g, vec![y])
}

/// conv + bias, then snake `v + sin²(αv)/α` as its epilogue.
fn conv_snake_graph(
    c: usize,
    t: usize,
    k: usize,
    dil: i64,
    im2col: bool,
) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[c, t]);
    let w = g.input(&[c, c, k]);
    let b = g.input(&[c, 1]);
    let alpha = g.input(&[c, 1]);
    let y = if im2col {
        conv1d_im2col(&mut g, x, w, dil)
    } else {
        conv1d(&mut g, x, w, dil)
    };
    let bb = g.broadcast(b, &[c, t]).unwrap();
    let ab = g.broadcast(alpha, &[c, t]).unwrap();
    let out = g
        .map(&[y, bb, ab], |v| {
            let z = v[0] + v[1];
            let s = (v[2] * z).sin();
            z + s * s / v[2]
        })
        .unwrap();
    (g, vec![out])
}

/// Polyphase upsample: `out[o, q, r] = Σ_{j,c} x[c, q−j] · w[o, c, r + s·j]`,
/// read as `[co, t·s]`. Equals `conv_transpose1d(k = 2s, stride s, p = 0)`
/// cropped to `t·s` samples.
fn upsample_graph(ci: usize, co: usize, t: usize, s: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[ci, t]);
    let w = g.input(&[co, ci, 2 * s]);
    // space (o, q, r, j, c)
    let xv = view(
        vec![form(5, &[(4, 1)], 0), form(5, &[(1, 1), (3, -1)], 0)],
        true,
    );
    let wv = view(
        vec![
            form(5, &[(0, 1)], 0),
            form(5, &[(4, 1)], 0),
            form(5, &[(2, 1), (3, s as i64)], 0),
        ],
        false,
    );
    let y = g
        .comp(
            &[co, t, s],
            &[2, ci],
            vec![arg(x, xv), arg(w, wv)],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let flat = g.reshape(y, &[co, t * s]).unwrap();
    (g, vec![flat])
}

/// Same upsample as unfold + one matmul per phase (`r`), weights
/// pre-arranged on the host as `wp[r, o, (c, j)]`, then a transpose to
/// `[o, q, r]`.
fn upsample_bmm_graph(ci: usize, co: usize, t: usize, s: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[ci, t]);
    let wp = g.input(&[s, co, ci * 2]);
    // unfold u[(c, j), q] = x[c, q − j]
    let uv = view(
        vec![form(3, &[(0, 1)], 0), form(3, &[(2, 1), (1, -1)], 0)],
        true,
    );
    let u = g
        .comp(&[ci, 2, t], &[], vec![arg(x, uv)], None, |v| vec![v[0]])
        .unwrap();
    let u = g.reshape(u, &[ci * 2, t]).unwrap();
    // bmm: z[r, o, q] = Σ_m wp[r, o, m] · u[m, q]
    let z = g
        .comp(
            &[s, co, t],
            &[ci * 2],
            vec![
                arg(wp, View::pick(4, &[Some(0), Some(1), Some(3)])),
                arg(u, View::pick(4, &[Some(3), Some(2)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let y = g.transpose(z, &[1, 2, 0]).unwrap();
    let flat = g.reshape(y, &[co, t * s]).unwrap();
    (g, vec![flat])
}

/// iSTFT head (Kokoro): `spec[22, T]` = 11 log-magnitudes + 11 phase
/// logits; `mag = exp(a)`, `φ = sin(b)`; inverse real DFT (n_fft 20) as a
/// reduction over bins with a cos basis, Hann-windowed, then overlap-add
/// (hop 5, 4 taps) as a polyphase reduction with guarded frame reads,
/// divided by the window envelope and center-trimmed like `torch.istft`.
fn istft_graph(frames: usize) -> (TensorGraph, Vec<TensorId>) {
    let (n_fft, hop, bins) = (20usize, 5usize, 11usize);
    let taps = n_fft / hop;
    let mut g = TensorGraph::new();
    let spec = g.input(&[2 * bins, frames]);
    let win = g.input(&[n_fft]);
    let out_len = hop * (frames - 1);
    let env = g.input(&[out_len / hop, hop]); // 1 / Σ win² at each output sample
                                              // frame[t, n] = win[n]/N · Σ_k c_k · exp(a_k) · cos(sin(b_k) + 2πkn/N)
    let space = [frames, n_fft, bins];
    let ik = g.iota(&space, 2);
    let inn = g.iota(&space, 1);
    let id3 = View::identity(3, 3);
    let frame = g
        .comp(
            &[frames, n_fft],
            &[bins],
            vec![
                arg(spec, View::pick(3, &[Some(2), Some(0)])),
                arg(
                    spec,
                    view(
                        vec![form(3, &[(2, 1)], bins as i64), form(3, &[(0, 1)], 0)],
                        false,
                    ),
                ),
                arg(win, View::pick(3, &[Some(1)])),
                arg(ik, id3.clone()),
                arg(inn, id3),
            ],
            Some(Monoid::sum()),
            |v| {
                let (a, b, w, k, n) = (v[0], v[1], v[2], v[3], v[4]);
                // c_k = 1 at DC and Nyquist, 2 elsewhere
                let edge = k * (lit((bins - 1) as f64) - k); // 0 at both ends
                let ck = ExprId::select(edge, lit(2.0), lit(1.0));
                let theta = b.sin() + lit(2.0 * std::f64::consts::PI / n_fft as f64) * k * n;
                let cos = (theta + lit(std::f64::consts::FRAC_PI_2)).sin();
                vec![ck * a.exp() * cos * w * lit(1.0 / n_fft as f64)]
            },
        )
        .unwrap();
    // y[q, r] (output sample m = 5q + r + 10 after the center trim):
    //   Σ_j frame[q + 2 − j, r + 5j] · env[q, r]
    let q = out_len / hop;
    let fv = view(
        vec![
            form(3, &[(0, 1), (2, -1)], 2),
            form(3, &[(1, 1), (2, hop as i64)], 0),
        ],
        true,
    );
    let y = g
        .comp(
            &[q, hop],
            &[taps],
            vec![arg(frame, fv), arg(env, View::pick(3, &[Some(0), Some(1)]))],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let flat = g.reshape(y, &[out_len]).unwrap();
    (g, vec![flat])
}

/// AdaIN: instance norm over T, then `(1 + γ)·x̂ + β` with `[γ; β] = W s + b`.
fn adain_graph(c: usize, t: usize, sd: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[c, t]);
    let s = g.input(&[sd, 1]);
    let w = g.input(&[2 * c, sd]);
    let b = g.input(&[2 * c, 1]);
    let h = g.matmul(w, s).unwrap();
    let h = g.map(&[h, b], |v| v[0] + v[1]).unwrap();
    let mean = g.reduce(x, 1, Monoid::sum()).unwrap();
    let mean = g.map(&[mean], |v| v[0] * lit(1.0 / t as f64)).unwrap();
    let mb = g.broadcast(mean, &[c, t]).unwrap();
    let dev = g.map(&[x, mb], |v| (v[0] - v[1]) * (v[0] - v[1])).unwrap();
    let var = g.reduce(dev, 1, Monoid::sum()).unwrap();
    let inv = g
        .map(&[var], |v| {
            lit(1.0) / (v[0] * lit(1.0 / t as f64) + lit(1e-5)).sqrt()
        })
        .unwrap();
    let ib = g.broadcast(inv, &[c, t]).unwrap();
    // γ = h[o], β = h[c + o], read through views
    let gamma = g
        .comp(
            &[c, t],
            &[],
            vec![arg(h, View::pick(2, &[Some(0), None]))],
            None,
            |v| vec![v[0]],
        )
        .unwrap();
    let beta = g
        .comp(
            &[c, t],
            &[],
            vec![arg(
                h,
                view(vec![form(2, &[(0, 1)], c as i64), Form::zero(2)], false),
            )],
            None,
            |v| vec![v[0]],
        )
        .unwrap();
    let y = g
        .map(&[x, mb, ib, gamma, beta], |v| {
            (lit(1.0) + v[3]) * (v[0] - v[1]) * v[2] + v[4]
        })
        .unwrap();
    (g, vec![y])
}

struct ConvNext {
    g: TensorGraph,
    y: TensorId,
    params: Vec<TensorId>,
}

/// Vocos ConvNeXt block on `[C, T]`: depthwise conv k7 → LayerNorm over C →
/// pointwise C→F, GELU, F→C → + residual.
fn convnext_graph(c: usize, t: usize, f: usize) -> ConvNext {
    let k = 7;
    let mut g = TensorGraph::new();
    let x = g.input(&[c, t]);
    let dw = g.input(&[c, k]);
    let db = g.input(&[c, 1]);
    let lg = g.input(&[c, 1]);
    let lb = g.input(&[c, 1]);
    let w1 = g.input(&[f, c]);
    let b1 = g.input(&[f, 1]);
    let w2 = g.input(&[c, f]);
    let b2 = g.input(&[c, 1]);
    // space (c, t, k)
    let xv = view(
        vec![form(3, &[(0, 1)], 0), form(3, &[(1, 1), (2, 1)], -3)],
        true,
    );
    let d = g
        .comp(
            &[c, t],
            &[k],
            vec![arg(x, xv), arg(dw, View::pick(3, &[Some(0), Some(2)]))],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let dbb = g.broadcast(db, &[c, t]).unwrap();
    let d = g.map(&[d, dbb], |v| v[0] + v[1]).unwrap();
    let mean = g.reduce(d, 0, Monoid::sum()).unwrap();
    let mean = g.map(&[mean], |v| v[0] * lit(1.0 / c as f64)).unwrap();
    let mb = g.broadcast(mean, &[c, t]).unwrap();
    let sq = g.map(&[d, mb], |v| (v[0] - v[1]) * (v[0] - v[1])).unwrap();
    let var = g.reduce(sq, 0, Monoid::sum()).unwrap();
    let inv = g
        .map(&[var], |v| {
            lit(1.0) / (v[0] * lit(1.0 / c as f64) + lit(1e-6)).sqrt()
        })
        .unwrap();
    let ib = g.broadcast(inv, &[c, t]).unwrap();
    let lgb = g.broadcast(lg, &[c, t]).unwrap();
    let lbb = g.broadcast(lb, &[c, t]).unwrap();
    let n = g
        .map(&[d, mb, ib, lgb, lbb], |v| {
            (v[0] - v[1]) * v[2] * v[3] + v[4]
        })
        .unwrap();
    let h = g.matmul(w1, n).unwrap();
    let b1b = g.broadcast(b1, &[f, t]).unwrap();
    let h = g.map(&[h, b1b], |v| gelu(v[0] + v[1])).unwrap();
    let o = g.matmul(w2, h).unwrap();
    let b2b = g.broadcast(b2, &[c, t]).unwrap();
    let y = g.map(&[o, b2b, x], |v| v[0] + v[1] + v[2]).unwrap();
    ConvNext {
        g,
        y,
        params: vec![x, dw, db, lg, lb, w1, b1, w2, b2],
    }
}

// ---------------------------------------------------------------------------
// CPU references (f64)
// ---------------------------------------------------------------------------

fn conv_ref(x: &[f64], w: &[f64], c: usize, t: usize, k: usize, dil: i64) -> Vec<f64> {
    let pad = dil * (k as i64 - 1) / 2;
    let mut y = vec![0.0; c * t];
    for o in 0..c {
        for ci in 0..c {
            for kk in 0..k {
                let wv = w[(o * c + ci) * k + kk];
                let sh = dil * kk as i64 - pad;
                let row = &x[ci * t..(ci + 1) * t];
                let yr = &mut y[o * t..(o + 1) * t];
                for (tt, yv) in yr.iter_mut().enumerate() {
                    let s = tt as i64 + sh;
                    if (0..t as i64).contains(&s) {
                        *yv += wv * row[s as usize];
                    }
                }
            }
        }
    }
    y
}

fn upsample_ref(x: &[f64], w: &[f64], ci: usize, co: usize, t: usize, s: usize) -> Vec<f64> {
    // conv_transpose1d(k = 2s, stride s, p = 0), cropped to t·s
    let k = 2 * s;
    let n = t * s;
    let mut y = vec![0.0; co * n];
    for o in 0..co {
        for c in 0..ci {
            for tt in 0..t {
                let xv = x[c * t + tt];
                for kk in 0..k {
                    let m = tt * s + kk;
                    if m < n {
                        y[o * n + m] += xv * w[(o * ci + c) * k + kk];
                    }
                }
            }
        }
    }
    y
}

fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

/// 1/Σ_t win²[m − hop·t] at each kept sample (after the n_fft/2 trim).
fn istft_env(frames: usize) -> Vec<f64> {
    let (n_fft, hop) = (20, 5);
    let w = hann(n_fft);
    let full = hop * (frames - 1) + n_fft;
    let mut env = vec![0.0; full];
    for t in 0..frames {
        for n in 0..n_fft {
            env[t * hop + n] += w[n] * w[n];
        }
    }
    let out_len = hop * (frames - 1);
    (0..out_len).map(|m| 1.0 / env[m + n_fft / 2]).collect()
}

fn istft_ref(spec: &[f64], frames: usize) -> Vec<f64> {
    let (n_fft, hop, bins) = (20usize, 5usize, 11usize);
    let w = hann(n_fft);
    let full = hop * (frames - 1) + n_fft;
    let mut y = vec![0.0; full];
    let mut env = vec![0.0; full];
    for t in 0..frames {
        for n in 0..n_fft {
            // irfft of X_k = mag·e^{iφ}
            let mut v = 0.0;
            for k in 0..bins {
                let mag = spec[k * frames + t].exp();
                let ph = spec[(bins + k) * frames + t].sin();
                let ck = if k == 0 || k == bins - 1 { 1.0 } else { 2.0 };
                let th = 2.0 * std::f64::consts::PI * (k * n) as f64 / n_fft as f64;
                v += ck * mag * (ph + th).cos();
            }
            y[t * hop + n] += v / n_fft as f64 * w[n];
            env[t * hop + n] += w[n] * w[n];
        }
    }
    let out_len = hop * (frames - 1);
    (0..out_len)
        .map(|m| y[m + n_fft / 2] / env[m + n_fft / 2])
        .collect()
}

fn adain_ref(
    x: &[f64],
    s: &[f64],
    w: &[f64],
    b: &[f64],
    c: usize,
    t: usize,
    sd: usize,
) -> Vec<f64> {
    let h: Vec<f64> = (0..2 * c)
        .map(|o| b[o] + (0..sd).map(|i| w[o * sd + i] * s[i]).sum::<f64>())
        .collect();
    let mut y = vec![0.0; c * t];
    for ch in 0..c {
        let row = &x[ch * t..(ch + 1) * t];
        let mean = row.iter().sum::<f64>() / t as f64;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / t as f64;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for tt in 0..t {
            y[ch * t + tt] = (1.0 + h[ch]) * (row[tt] - mean) * inv + h[c + ch];
        }
    }
    y
}

fn gelu_ref(x: f64) -> f64 {
    let u = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + u.tanh())
}

fn convnext_ref(p: &[Vec<f64>], c: usize, t: usize, f: usize) -> Vec<f64> {
    let (x, dw, db, lg, lb, w1, b1, w2, b2) = (
        &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6], &p[7], &p[8],
    );
    let mut d = vec![0.0; c * t];
    for ch in 0..c {
        for tt in 0..t {
            let mut v = db[ch];
            for k in 0..7 {
                let s = tt as i64 + k as i64 - 3;
                if (0..t as i64).contains(&s) {
                    v += x[ch * t + s as usize] * dw[ch * 7 + k];
                }
            }
            d[ch * t + tt] = v;
        }
    }
    let mut n = vec![0.0; c * t];
    for tt in 0..t {
        let mean = (0..c).map(|ch| d[ch * t + tt]).sum::<f64>() / c as f64;
        let var = (0..c)
            .map(|ch| (d[ch * t + tt] - mean).powi(2))
            .sum::<f64>()
            / c as f64;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for ch in 0..c {
            n[ch * t + tt] = (d[ch * t + tt] - mean) * inv * lg[ch] + lb[ch];
        }
    }
    let mut h = vec![0.0; f * t];
    for o in 0..f {
        for ch in 0..c {
            let wv = w1[o * c + ch];
            for tt in 0..t {
                h[o * t + tt] += wv * n[ch * t + tt];
            }
        }
        for tt in 0..t {
            h[o * t + tt] = gelu_ref(h[o * t + tt] + b1[o]);
        }
    }
    let mut y = vec![0.0; c * t];
    for o in 0..c {
        for i in 0..f {
            let wv = w2[o * f + i];
            for tt in 0..t {
                y[o * t + tt] += wv * h[i * t + tt];
            }
        }
        for tt in 0..t {
            y[o * t + tt] += b2[o] + x[o * t + tt];
        }
    }
    y
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

/// The graph passes before scheduling. `merge_reduce_dims` runs before
/// `schedule_auto`'s multi-dim tiler: mergeable reduce dims become one,
/// the tiler handles what stays split (a conv's `(c, k)`).
/// `TANG_NO_MERGE=1` skips merge_siblings / relayout / merge_reduce_dims.
fn optimized(g: &TensorGraph, outs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let (g, outs) = g.substitute_maps(outs);
    let (g, outs) = g.eliminate_common(&outs);
    if std::env::var_os("TANG_NO_MERGE").is_some() {
        return (g, outs);
    }
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    g.merge_reduce_dims(&outs)
}

struct Row {
    name: &'static str,
    best: f64,
    schedule: String,
}

/// Time `naive`, `auto` and per-kernel `autotune` for every algebra in
/// `algebras`, checking each against `want` (output 0).
fn bench(
    device: &GpuDevice,
    cache: &mut KernelCache,
    name: &'static str,
    algebras: Vec<(&'static str, TensorGraph, Vec<TensorId>)>,
    inputs: &[&[f32]],
    want: &[f64],
    tol: f64,
) -> Row {
    let mut best = (f64::INFINITY, String::new());
    let mut line = format!("{name}:");
    for (alg, g, outs) in algebras {
        let (g, outs) = optimized(&g, &outs);
        let mut auto = g.clone();
        auto.schedule_auto(&outs);
        let tuned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            autotune_kernels(device, cache, auto.clone(), &outs, inputs, 10).0
        }));
        let mut variants = vec![("naive", g), ("auto", auto)];
        match tuned {
            Ok(t) => variants.push(("autotune", t)),
            Err(_) => line.push_str(&format!("  [{alg}/autotune FAILED]")),
        }
        for (sched, g) in variants {
            let exe = Executable::new(device, g.to_wgsl(&outs), inputs);
            exe.run(device, cache);
            let got = exe.read(device, outs[0]);
            let err = rel_err(&got, want);
            let ms = time_exe(device, cache, &exe, outs[0]);
            let ok = if err <= tol { "" } else { " WRONG" };
            line.push_str(&format!(
                "\n    {alg:>8}/{sched:<8} {ms:8.3} ms  {} kernels  err {err:.1e}{ok}",
                exe.program.kernels.len()
            ));
            if err <= tol && ms < best.0 {
                best = (ms, format!("{alg}/{sched}"));
            }
        }
    }
    println!("{line}");
    Row {
        name,
        best: best.0,
        schedule: best.1,
    }
}

/// Forward + backward of `sum(y)` w.r.t. `wrt`, all schedules; checked on
/// a small instance against the f64 interpreter (`small`).
fn bench_train(
    device: &GpuDevice,
    cache: &mut KernelCache,
    name: &'static str,
    algebras: Vec<(&'static str, TensorGraph, TensorId, Vec<TensorId>)>,
    inputs: &[&[f32]],
) -> Row {
    let mut best = (f64::INFINITY, String::new());
    let mut line = format!("{name}:");
    for (alg, mut g, y, wrt) in algebras {
        let grads = g.backward(y, &wrt).unwrap();
        let mut outs = vec![y];
        outs.extend(grads);
        let (g, outs) = optimized(&g, &outs);
        let mut auto = g.clone();
        auto.schedule_auto(&outs);
        let mut variants = vec![("naive", g), ("auto", auto.clone())];
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            autotune_kernels(device, cache, auto.clone(), &outs, inputs, 10).0
        })) {
            Ok(t) => variants.push(("autotune", t)),
            Err(_) => line.push_str(&format!("  [{alg}/autotune FAILED]")),
        }
        for (sched, g) in variants {
            let exe = Executable::new(device, g.to_wgsl(&outs), inputs);
            let ms = time_exe(device, cache, &exe, outs[0]);
            line.push_str(&format!(
                "\n    {alg:>8}/{sched:<8} {ms:8.3} ms  {} kernels",
                exe.program.kernels.len()
            ));
            if ms < best.0 {
                best = (ms, format!("{alg}/{sched}"));
            }
        }
    }
    println!("{line}");
    Row {
        name,
        best: best.0,
        schedule: best.1,
    }
}

/// GPU forward+backward on a small instance vs the f64 interpreter.
fn check_train(
    device: &GpuDevice,
    cache: &mut KernelCache,
    name: &str,
    (g, y, wrt): (TensorGraph, TensorId, Vec<TensorId>),
    inputs: &[Vec<f32>],
) {
    let mut g = g;
    let grads = g.backward(y, &wrt).unwrap();
    let mut outs = vec![y];
    outs.extend(grads);
    let want = g.eval(&inputs.iter().map(|v| f64s(v)).collect::<Vec<_>>(), &outs);
    let (mut og, oouts) = optimized(&g, &outs);
    og.schedule_auto(&oouts);
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let exe = Executable::new(device, og.to_wgsl(&oouts), &refs);
    exe.run(device, cache);
    let worst = oouts
        .iter()
        .zip(&want)
        .map(|(o, w)| rel_err(&exe.read(device, *o), w))
        .fold(0.0, f64::max);
    println!("  {name} fwd+bwd check (small, vs f64 interpreter): max err {worst:.1e}");
    assert!(worst < 1e-4, "{name} backward wrong");
}

fn main() {
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
    let only: Option<String> = std::env::args().nth(1);
    let want_op = |n: &str| only.as_deref().is_none_or(|o| n.starts_with(o));
    println!(
        "{} — kernel-only, {ITERS} launches per sample\n",
        device.name
    );
    let mut rows: Vec<Row> = Vec::new();
    let tol = 1e-5;

    // (a) conv1d 512→512, k7, dilation 3, same padding, T=2000
    let (c, t, k, dil) = (512, 2000, 7, 3);
    let x = rand(c * t, 1, 1.0);
    let w = rand(c * c * k, 2, 0.05);
    if want_op("a") {
        let want = conv_ref(&f64s(&x), &f64s(&w), c, t, k, dil);
        let (gd, od) = conv_graph(c, t, k, dil, false);
        let (gi, oi) = conv_graph(c, t, k, dil, true);
        rows.push(bench(
            &device,
            &mut cache,
            "a conv1d 512→512 k7 d3 T2000",
            vec![("direct", gd, od), ("im2col", gi, oi)],
            &[&x, &w],
            &want,
            tol,
        ));
    }

    // (b) polyphase upsample 512→256, k20, s10, T=2000 → [256, 20000]
    let (ci, co, s) = (512, 256, 10);
    if want_op("b") {
        let xu = rand(ci * t, 3, 1.0);
        let wu = rand(co * ci * 2 * s, 4, 0.05);
        let want = upsample_ref(&f64s(&xu), &f64s(&wu), ci, co, t, s);
        // host-side weight arrangement for the bmm algebra: wp[r, o, (c, j)]
        let mut wp = vec![0.0f32; s * co * ci * 2];
        for r in 0..s {
            for o in 0..co {
                for cc in 0..ci {
                    for j in 0..2 {
                        wp[((r * co + o) * ci + cc) * 2 + j] =
                            wu[(o * ci + cc) * 2 * s + r + s * j];
                    }
                }
            }
        }
        let (gd, od) = upsample_graph(ci, co, t, s);
        let line = bench(
            &device,
            &mut cache,
            "b polyphase upsample 512→256 k20 s10",
            vec![("direct", gd, od)],
            &[&xu, &wu],
            &want,
            tol,
        );
        let (gb, ob) = upsample_bmm_graph(ci, co, t, s);
        let bmm = bench(
            &device,
            &mut cache,
            "b polyphase upsample (bmm algebra)",
            vec![("bmm", gb, ob)],
            &[&xu, &wp],
            &want,
            tol,
        );
        rows.push(if bmm.best < line.best {
            Row {
                name: line.name,
                ..bmm
            }
        } else {
            line
        });
    }

    // (c) iSTFT head: [22, T] → audio, n_fft 20, hop 5
    if want_op("c") {
        let frames = 4000;
        let spec = rand(22 * frames, 5, 1.0);
        let win: Vec<f32> = hann(20).iter().map(|&v| v as f32).collect();
        let env: Vec<f32> = istft_env(frames).iter().map(|&v| v as f32).collect();
        let want = istft_ref(&f64s(&spec), frames);
        let (g, o) = istft_graph(frames);
        rows.push(bench(
            &device,
            &mut cache,
            "c iSTFT n_fft20 hop5 T4000",
            vec![("dft+ola", g, o)],
            &[&spec, &win, &env],
            &want,
            tol,
        ));
    }

    // (d) conv (a) + bias + snake epilogue
    if want_op("d") {
        let b = rand(c, 6, 0.1);
        let alpha: Vec<f32> = rand(c, 7, 0.5).iter().map(|v| v + 1.0).collect();
        let mut want = conv_ref(&f64s(&x), &f64s(&w), c, t, k, dil);
        for ch in 0..c {
            for tt in 0..t {
                let z = want[ch * t + tt] + b[ch] as f64;
                let a = alpha[ch] as f64;
                want[ch * t + tt] = z + (a * z).sin().powi(2) / a;
            }
        }
        let (gd, od) = conv_snake_graph(c, t, k, dil, false);
        let (gi, oi) = conv_snake_graph(c, t, k, dil, true);
        // one kernel: the conv reduction recomputed under the snake map
        let (gf, of) = optimized(&gd, &od);
        let mut fused = gf.clone();
        let conv_node = fused.nodes().iter().position(
            |n| matches!(&n.op, tang_compile::tensor::Op::Comp(cc) if cc.monoid.is_some()),
        );
        fused.set_inline(TensorId::from_index(conv_node.unwrap()), true);
        let exe = Executable::new(&device, fused.to_wgsl(&of), &[&x, &w, &b, &alpha]);
        exe.run(&device, &mut cache);
        let err = rel_err(&exe.read(&device, of[0]), &want);
        let ms = time_exe(&device, &mut cache, &exe, of[0]);
        println!(
            "d fused into one kernel (conv inlined under snake): {ms:.3} ms  {} kernels  err {err:.1e}",
            exe.program.kernels.len()
        );
        let mut row = bench(
            &device,
            &mut cache,
            "d conv + bias + snake",
            vec![("direct", gd, od), ("im2col", gi, oi)],
            &[&x, &w, &b, &alpha],
            &want,
            tol,
        );
        if err <= tol && ms < row.best {
            row.best = ms;
            row.schedule = "direct/fused-1-kernel".into();
        }
        rows.push(row);
    }

    // (e) AdaIN, C=512, T=2000, style 128
    if want_op("e") {
        let sd = 128;
        let xa = rand(c * t, 8, 1.0);
        let st = rand(sd, 9, 1.0);
        let wa = rand(2 * c * sd, 10, 0.1);
        let ba = rand(2 * c, 11, 0.1);
        let want = adain_ref(&f64s(&xa), &f64s(&st), &f64s(&wa), &f64s(&ba), c, t, sd);
        let (g, o) = adain_graph(c, t, sd);
        rows.push(bench(
            &device,
            &mut cache,
            "e AdaIN C512 T2000 style128",
            vec![("adain", g, o)],
            &[&xa, &st, &wa, &ba],
            &want,
            tol,
        ));
    }

    // (f) ConvNeXt block, C=512, T=200, F=1536
    let (cf, tf, ff) = (512, 200, 1536);
    let cn_inputs: Vec<Vec<f32>> = vec![
        rand(cf * tf, 12, 1.0),
        rand(cf * 7, 13, 0.3),
        rand(cf, 14, 0.1),
        rand(cf, 15, 0.2).iter().map(|v| v + 1.0).collect(),
        rand(cf, 16, 0.1),
        rand(ff * cf, 17, 0.04),
        rand(ff, 18, 0.1),
        rand(cf * ff, 19, 0.02),
        rand(cf, 20, 0.1),
    ];
    let cn_refs: Vec<&[f32]> = cn_inputs.iter().map(|v| v.as_slice()).collect();
    if want_op("f") {
        let want = convnext_ref(
            &cn_inputs.iter().map(|v| f64s(v)).collect::<Vec<_>>(),
            cf,
            tf,
            ff,
        );
        let cn = convnext_graph(cf, tf, ff);
        rows.push(bench(
            &device,
            &mut cache,
            "f ConvNeXt block C512 T200 F1536",
            vec![("convnext", cn.g, vec![cn.y])],
            &cn_refs,
            &want,
            tol,
        ));
    }

    // forward + backward for (a) and (f)
    if want_op("a") || want_op("train") {
        let small = conv_graph(6, 20, 7, 3, false);
        let (sg, so) = small;
        let (ci_s, t_s) = (6, 20);
        let inputs = vec![rand(ci_s * t_s, 21, 1.0), rand(ci_s * ci_s * 7, 22, 0.3)];
        check_train(
            &device,
            &mut cache,
            "a",
            (
                sg,
                so[0],
                vec![TensorId::from_index(0), TensorId::from_index(1)],
            ),
            &inputs,
        );
        let (si, soi) = conv_graph(ci_s, t_s, 7, 3, true);
        check_train(
            &device,
            &mut cache,
            "a im2col",
            (
                si,
                soi[0],
                vec![TensorId::from_index(0), TensorId::from_index(1)],
            ),
            &inputs,
        );
        let xw = vec![TensorId::from_index(0), TensorId::from_index(1)];
        let (g, o) = conv_graph(c, t, k, dil, false);
        let (gi, oi) = conv_graph(c, t, k, dil, true);
        rows.push(bench_train(
            &device,
            &mut cache,
            "a fwd+bwd (dx, dw)",
            vec![("direct", g, o[0], xw.clone()), ("im2col", gi, oi[0], xw)],
            &[&x, &w],
        ));
    }
    if want_op("f") || want_op("train") {
        let (cs, ts, fs) = (8, 12, 16);
        let small = convnext_graph(cs, ts, fs);
        let sizes = [cs * ts, cs * 7, cs, cs, cs, fs * cs, fs, cs * fs, cs];
        let inputs: Vec<Vec<f32>> = sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| rand(n, 30 + i as u64, 0.5))
            .collect();
        check_train(
            &device,
            &mut cache,
            "f",
            (small.g, small.y, small.params),
            &inputs,
        );
        let cn = convnext_graph(cf, tf, ff);
        rows.push(bench_train(
            &device,
            &mut cache,
            "f fwd+bwd (all params)",
            vec![("convnext", cn.g, cn.y, cn.params)],
            &cn_refs,
        ));
    }

    println!("\n| op | IR best (ms) | schedule |");
    println!("|---|---:|---|");
    for r in rows {
        println!("| {} | {:.3} | {} |", r.name, r.best, r.schedule);
    }
}
