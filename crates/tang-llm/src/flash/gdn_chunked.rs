//! Chunked (chunkwise-parallel) Gated DeltaNet recurrence: the CPU reference for the CUDA
//! kernel `fl_gdn_chunk_step_seg` (`TANG_CODER_GDN=chunked`).
//!
//! The algorithm follows flash-linear-attention's `chunk_gated_delta_rule` (MIT, Songlin Yang
//! et al.; Yang, Kautz, Hatamizadeh, "Gated Delta Networks", arXiv 2412.06464; Yang et al.,
//! "Parallelizing Linear Transformers with the Delta Rule", arXiv 2406.06484). One head, state
//! `S[a][b]` (a: key dim, b: value dim), per token `t` (the order the engine's step kernel uses):
//!
//! ```text
//! S <- g_t S;   d_t = beta_t (v_t - S^T k_t);   S <- S + k_t d_t^T;   o_t = S^T q_t
//! ```
//!
//! Over a chunk of `C` tokens starting from `S0`, with `lg_i = sum_{s<=i} ln g_s` (in-chunk
//! cumulative log decay) and `G_is = exp(lg_i - lg_s)` (i >= s, so <= 1):
//!
//! ```text
//! d_i  = beta_i (v_i - e^{lg_i} S0^T k_i) - beta_i sum_{s<i} G_is (k_i.k_s) d_s   (unit lower-
//!        triangular solve: the UT / WY form, D = (I + A)^{-1} beta (V - Gamma K S0))
//! o_i  = e^{lg_i} S0^T q_i + sum_{s<=i} G_is (q_i.k_s) d_s
//! S'   = e^{lg_last} S0 + sum_s G_{last,s} k_s d_s^T
//! ```
//!
//! Everything except the `C`-step triangular solve is dense matmul work, and the sequential
//! dependence is one step a chunk instead of one a token.

/// One head's inputs for `t` tokens, row-major `[t][d]` (q, k, v) and `[t]` (log decay, beta).
pub struct Head<'a> {
    pub d: usize,
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    /// `ln g_t` (= softplus(a + dt) * A, <= 0).
    pub lg: &'a [f32],
    /// `sigmoid(b)`.
    pub beta: &'a [f32],
}

impl Head<'_> {
    fn t(&self) -> usize {
        self.lg.len()
    }
}

/// The token-by-token recurrence (the engine's current path). `st`: `[d][d]`, updated in place.
/// Returns `o`, `[t][d]` (unscaled: the caller applies 1/sqrt(d)).
pub fn recurrent(h: &Head, st: &mut [f32]) -> Vec<f32> {
    let (d, t) = (h.d, h.t());
    let mut o = vec![0f32; t * d];
    let mut sk = vec![0f32; d];
    for ti in 0..t {
        let (q, k, v) = (
            &h.q[ti * d..][..d],
            &h.k[ti * d..][..d],
            &h.v[ti * d..][..d],
        );
        let g = h.lg[ti].exp();
        st.iter_mut().for_each(|e| *e *= g);
        sk.fill(0.0);
        for a in 0..d {
            for b in 0..d {
                sk[b] += st[a * d + b] * k[a];
            }
        }
        for a in 0..d {
            for b in 0..d {
                st[a * d + b] += k[a] * (v[b] - sk[b]) * h.beta[ti];
            }
        }
        for a in 0..d {
            for b in 0..d {
                o[ti * d + b] += st[a * d + b] * q[a];
            }
        }
    }
    o
}

fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// The chunkwise form, chunks of `c` tokens (the last may be short). Same contract as
/// [`recurrent`]; equal up to float reassociation.
pub fn chunked(h: &Head, st: &mut [f32], c: usize) -> Vec<f32> {
    let (d, t) = (h.d, h.t());
    let mut o = vec![0f32; t * d];
    let mut t0 = 0;
    while t0 < t {
        let n = c.min(t - t0);
        let (q, k, v) = (
            |i: usize| &h.q[(t0 + i) * d..][..d],
            |i: usize| &h.k[(t0 + i) * d..][..d],
            |i: usize| &h.v[(t0 + i) * d..][..d],
        );
        let mut lg = vec![0f32; n];
        let mut acc = 0f32;
        for (i, l) in lg.iter_mut().enumerate() {
            acc += h.lg[t0 + i];
            *l = acc;
        }
        // S0^T k_i and S0^T q_i (one [n][d] x [d][d] GEMM each).
        let mut ks = vec![0f32; n * d];
        let mut qs = vec![0f32; n * d];
        for i in 0..n {
            for a in 0..d {
                let (ka, qa) = (k(i)[a], q(i)[a]);
                for b in 0..d {
                    ks[i * d + b] += st[a * d + b] * ka;
                    qs[i * d + b] += st[a * d + b] * qa;
                }
            }
        }
        // d_i: right-hand side, then forward substitution.
        let mut dd = vec![0f32; n * d];
        for i in 0..n {
            let (gi, bi) = (lg[i].exp(), h.beta[t0 + i]);
            for b in 0..d {
                dd[i * d + b] = bi * (v(i)[b] - gi * ks[i * d + b]);
            }
            for s in 0..i {
                let w = bi * (lg[i] - lg[s]).exp() * dot(k(i), k(s));
                for b in 0..d {
                    dd[i * d + b] -= w * dd[s * d + b];
                }
            }
        }
        for i in 0..n {
            let gi = lg[i].exp();
            for b in 0..d {
                o[(t0 + i) * d + b] = gi * qs[i * d + b];
            }
            for s in 0..=i {
                let w = (lg[i] - lg[s]).exp() * dot(q(i), k(s));
                for b in 0..d {
                    o[(t0 + i) * d + b] += w * dd[s * d + b];
                }
            }
        }
        let last = lg[n - 1];
        let gl = last.exp();
        st.iter_mut().for_each(|e| *e *= gl);
        for s in 0..n {
            let w = (last - lg[s]).exp();
            for a in 0..d {
                let ka = w * k(s)[a];
                for b in 0..d {
                    st[a * d + b] += ka * dd[s * d + b];
                }
            }
        }
        t0 += n;
    }
    o
}

/// Phase A of [`three_phase`] for one chunk (no state needed): the in-chunk cumulative log decay
/// `lg`, the gated, masked `Aqk[i][s] = e^{lg_i - lg_s} q_i.k_s` (s <= i), and the WY factors
/// `W = T diag(beta e^lg) K`, `U = T diag(beta) V` with `T = (I + Akk)^{-1}`,
/// `Akk[i][s] = beta_i e^{lg_i - lg_s} k_i.k_s` (s < i). `D = U - W S0` is then the chunk's
/// `d` rows for any start state `S0`.
pub struct ChunkPrep {
    pub n: usize,
    pub lg: Vec<f32>,
    pub aqk: Vec<f32>,
    pub w: Vec<f32>,
    pub u: Vec<f32>,
}

/// [`ChunkPrep`] for tokens `t0 .. t0 + n` of `h`.
pub fn chunk_prep(h: &Head, t0: usize, n: usize) -> ChunkPrep {
    let d = h.d;
    let row = |x: &[f32], i: usize| -> Vec<f32> { x[(t0 + i) * d..][..d].to_vec() };
    let (q, k, v): (Vec<_>, Vec<_>, Vec<_>) = (
        (0..n).map(|i| row(h.q, i)).collect(),
        (0..n).map(|i| row(h.k, i)).collect(),
        (0..n).map(|i| row(h.v, i)).collect(),
    );
    let beta = &h.beta[t0..t0 + n];
    let mut lg = vec![0f32; n];
    let mut acc = 0f32;
    for (i, l) in lg.iter_mut().enumerate() {
        acc += h.lg[t0 + i];
        *l = acc;
    }
    let mut akk = vec![0f32; n * n];
    let mut aqk = vec![0f32; n * n];
    for i in 0..n {
        for s in 0..=i {
            let g = (lg[i] - lg[s]).exp();
            aqk[i * n + s] = g * dot(&q[i], &k[s]);
            if s < i {
                akk[i * n + s] = beta[i] * g * dot(&k[i], &k[s]);
            }
        }
    }
    // T = (I + Akk)^{-1}: T_i = e_i - sum_{s<i} Akk_is T_s.
    let mut tm = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut x = if i == j { 1.0 } else { 0.0 };
            for s in j..i {
                x -= akk[i * n + s] * tm[s * n + j];
            }
            tm[i * n + j] = if j <= i { x } else { 0.0 };
        }
    }
    let (mut w, mut u) = (vec![0f32; n * d], vec![0f32; n * d]);
    for i in 0..n {
        for s in 0..=i {
            let (tw, tu) = (
                tm[i * n + s] * beta[s] * lg[s].exp(),
                tm[i * n + s] * beta[s],
            );
            for c in 0..d {
                w[i * d + c] += tw * k[s][c];
                u[i * d + c] += tu * v[s][c];
            }
        }
    }
    ChunkPrep { n, lg, aqk, w, u }
}

/// fla's three-phase layout, the structure of the `chunked2` CUDA kernels: (A) [`chunk_prep`]
/// for every chunk independently; (B) a sequential pass carrying the state,
/// `D_c = U - W S_c`, `S_{c+1} = e^{lg_last} S_c + (e^{lg_last - lg} K)^T D_c`; (C) every
/// chunk's outputs from its start state, `O = diag(e^lg) Q S_c + Aqk D_c`. Same contract as
/// [`recurrent`].
pub fn three_phase(h: &Head, st: &mut [f32], c: usize) -> Vec<f32> {
    let (d, t) = (h.d, h.t());
    let starts: Vec<usize> = (0..t).step_by(c).collect();
    // A: parallel over chunks.
    let preps: Vec<ChunkPrep> = starts
        .iter()
        .map(|&t0| chunk_prep(h, t0, c.min(t - t0)))
        .collect();
    // B: sequential, one small matmul pair a chunk.
    let mut states = Vec::with_capacity(preps.len());
    let mut ds = Vec::with_capacity(preps.len());
    for (p, &t0) in preps.iter().zip(&starts) {
        states.push(st.to_vec());
        let n = p.n;
        let mut dd = p.u.clone();
        for i in 0..n {
            for a in 0..d {
                let wa = p.w[i * d + a];
                for b in 0..d {
                    dd[i * d + b] -= wa * st[a * d + b];
                }
            }
        }
        let last = p.lg[n - 1];
        let gl = last.exp();
        st.iter_mut().for_each(|e| *e *= gl);
        for s in 0..n {
            let g = (last - p.lg[s]).exp();
            for a in 0..d {
                let ka = g * h.k[(t0 + s) * d + a];
                for b in 0..d {
                    st[a * d + b] += ka * dd[s * d + b];
                }
            }
        }
        ds.push(dd);
    }
    // C: parallel over chunks.
    let mut o = vec![0f32; t * d];
    for (ci, (p, &t0)) in preps.iter().zip(&starts).enumerate() {
        let (s0, dd, n) = (&states[ci], &ds[ci], p.n);
        for i in 0..n {
            let gi = p.lg[i].exp();
            let orow = &mut o[(t0 + i) * d..][..d];
            for a in 0..d {
                let qa = gi * h.q[(t0 + i) * d + a];
                for b in 0..d {
                    orow[b] += qa * s0[a * d + b];
                }
            }
            for s in 0..=i {
                let w = p.aqk[i * n + s];
                for b in 0..d {
                    orow[b] += w * dd[s * d + b];
                }
            }
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift: deterministic, no dependency.
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f()).collect()
        }
    }

    fn l2_rows(x: &mut [f32], d: usize) {
        for r in x.chunks_mut(d) {
            let n = r.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
            r.iter_mut().for_each(|v| *v /= n);
        }
    }

    fn check(d: usize, t: usize, c: usize, seed: u64, strong_decay: bool) {
        let mut r = Rng(seed);
        let (mut q, mut k, v) = (r.vec(t * d), r.vec(t * d), r.vec(t * d));
        l2_rows(&mut q, d);
        l2_rows(&mut k, d);
        // ln g = softplus(a + dt) * A, A < 0; beta = sigmoid(b).
        let sp = |x: f32| if x > 20.0 { x } else { x.exp().ln_1p() };
        let a_coef = if strong_decay { -8.0 } else { -0.3 };
        let lg: Vec<f32> = (0..t).map(|_| sp(r.f() * 3.0) * a_coef).collect();
        let beta: Vec<f32> = (0..t)
            .map(|_| 1.0 / (1.0 + (-(r.f() * 4.0)).exp()))
            .collect();
        let s0 = r.vec(d * d);
        let h = Head {
            d,
            q: &q,
            k: &k,
            v: &v,
            lg: &lg,
            beta: &beta,
        };
        let (mut sa, mut sb) = (s0.clone(), s0.clone());
        let oa = recurrent(&h, &mut sa);
        let ob = chunked(&h, &mut sb, c);
        let err = |x: &[f32], y: &[f32]| {
            x.iter()
                .zip(y)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        };
        let (eo, es) = (err(&oa, &ob), err(&sa, &sb));
        assert!(
            eo < 2e-4 && es < 2e-4,
            "d {d} t {t} c {c}: o err {eo}, state err {es}"
        );
        let mut sc = s0.clone();
        let oc = three_phase(&h, &mut sc, c);
        let (eo, es) = (err(&oa, &oc), err(&sa, &sc));
        assert!(
            eo < 2e-4 && es < 2e-4,
            "three-phase d {d} t {t} c {c}: o err {eo}, state err {es}"
        );
    }

    #[test]
    fn chunked_matches_recurrent_small() {
        for (t, c) in [(1, 16), (5, 4), (16, 16), (37, 16), (64, 64), (100, 32)] {
            check(16, t, c, 7 + t as u64, false);
        }
    }

    #[test]
    fn chunked_matches_recurrent_strong_decay() {
        check(16, 77, 16, 99, true);
    }

    /// The model's head size and the kernel's chunk.
    #[test]
    fn chunked_matches_recurrent_d128() {
        check(128, 45, 16, 3, false);
    }

    /// The `chunked2` kernels' shape: d 128, chunk 64, a window of several chunks plus a short
    /// tail, weak and strong decay.
    #[test]
    fn three_phase_matches_recurrent_d128_c64() {
        check(128, 64 * 3 + 21, 64, 11, false);
        check(128, 150, 64, 12, true);
    }

    /// Decode continues from the three-phase final state as from the recurrent one.
    #[test]
    fn decode_continues_from_three_phase_state() {
        let (d, t, split) = (32, 200, 150);
        let mut r = Rng(5);
        let (mut q, mut k, v) = (r.vec(t * d), r.vec(t * d), r.vec(t * d));
        l2_rows(&mut q, d);
        l2_rows(&mut k, d);
        let lg: Vec<f32> = (0..t).map(|_| -0.1 * (r.f() + 1.0)).collect();
        let beta: Vec<f32> = (0..t).map(|_| 0.5 + 0.4 * r.f()).collect();
        let part = |a: usize, b: usize| Head {
            d,
            q: &q[a * d..b * d],
            k: &k[a * d..b * d],
            v: &v[a * d..b * d],
            lg: &lg[a..b],
            beta: &beta[a..b],
        };
        let (mut sa, mut sb) = (vec![0f32; d * d], vec![0f32; d * d]);
        recurrent(&part(0, split), &mut sa);
        three_phase(&part(0, split), &mut sb, 64);
        let oa = recurrent(&part(split, t), &mut sa);
        let ob = recurrent(&part(split, t), &mut sb);
        let e = oa
            .iter()
            .zip(&ob)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(e < 1e-4, "decode after three-phase prefill: {e}");
    }

    /// The final state handed to decode: continuing token-by-token from the chunked state
    /// gives the same outputs as continuing from the recurrent one.
    #[test]
    fn decode_continues_from_chunked_state() {
        let (d, t) = (32, 50);
        let mut r = Rng(42);
        let (mut q, mut k, v) = (r.vec(t * d), r.vec(t * d), r.vec(t * d));
        l2_rows(&mut q, d);
        l2_rows(&mut k, d);
        let lg: Vec<f32> = (0..t).map(|_| -0.2 * (r.f() + 1.0)).collect();
        let beta: Vec<f32> = (0..t).map(|_| 0.5 + 0.4 * r.f()).collect();
        let split = 40;
        let pre = Head {
            d,
            q: &q[..split * d],
            k: &k[..split * d],
            v: &v[..split * d],
            lg: &lg[..split],
            beta: &beta[..split],
        };
        let post = Head {
            d,
            q: &q[split * d..],
            k: &k[split * d..],
            v: &v[split * d..],
            lg: &lg[split..],
            beta: &beta[split..],
        };
        let (mut sa, mut sb) = (vec![0f32; d * d], vec![0f32; d * d]);
        recurrent(&pre, &mut sa);
        chunked(&pre, &mut sb, 16);
        let oa = recurrent(&post, &mut sa);
        let ob = recurrent(&post, &mut sb);
        let e = oa
            .iter()
            .zip(&ob)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(e < 1e-4, "decode after chunked prefill: {e}");
    }
}
