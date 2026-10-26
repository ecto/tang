//! Shadow-check debugging for the GPU-resident KDA layers (`TANG_GLM5_KDA_DEBUG=1`).
//!
//! The staging bisect (Oct 2026) chased a GPU-vs-CPU divergence through four pod runs
//! with `TANG_GLM5_KDA_DUMP`'s first-8-value prints, paired ACROSS runs and machines — and
//! got stuck on token 0, whose core output is ~1e-7 noise on both paths. This replaces that
//! with an in-process check that answers "which kernel is wrong" in one run:
//!
//! - every resident window of a checked layer is re-run on the CPU (`kda::kda_layer_taps`)
//!   from the SAME `x` the GPU got, so upstream divergence can't leak in;
//! - the CPU shadow starts from the device's own state (resync after every window, default
//!   on), so each window tests ONE step, not accumulated drift;
//! - every intermediate is compared in full (all tokens, all channels): q/k/v
//!   projections, gates, conv output, core output, g2, gated norm, o_proj, plus the
//!   device's recurrent + conv state against the shadow's and the host conv copy;
//! - device scratch is poisoned with NaN before each window (`..._POISON`, default on),
//!   so an output region a kernel never wrote reads NaN instead of a plausible 0.0;
//! - the FIRST stage (pipeline order) that fails while its inputs pass is printed as
//!   `kdadbg FIRST DIVERGENCE`, and full tensors of failing windows go to `..._DIR`.
//!
//! Env (all optional):
//! - `TANG_GLM5_KDA_DEBUG=1`           enable
//! - `TANG_GLM5_KDA_DEBUG_LAYERS=0,1`  only these (global) layer indices (default: all KDA)
//! - `TANG_GLM5_KDA_DEBUG_WINDOWS=4`   only the first N windows per layer (default: all)
//! - `TANG_GLM5_KDA_DEBUG_TOL=1e-3`    relative-L2 bar per tensor
//! - `TANG_GLM5_KDA_DEBUG_RESYNC=0`    let the shadow carry its own state (tests drift)
//! - `TANG_GLM5_KDA_DEBUG_POISON=0`    don't NaN-fill device scratch
//! - `TANG_GLM5_KDA_DEBUG_DIR=/path`   write `w{W}_l{L}_{tap}.{gpu,cpu}.f32` (raw LE f32)
//!   for the first window checked and the first `..._DUMPS` (default 3) failing windows
//!
//! Output lines all start with `kdadbg` (grep them out of the run log).
use super::kda::{ConvState, KdaConfig, KdaLayerW, KdaState};
use anyhow::Result;
use std::fmt::Write as _;
use std::path::PathBuf;

/// Every intermediate of one KDA layer window, host-side, in the layouts both paths share:
/// `qkv_pre`/`conv_out` `[t][3·di]` (q|k|v stacked per token), `g`/`o`/`g2`/`y_norm`
/// `[t][di]`, `beta` `[t][nh]`, `out` `[t][hidden]`. A tap a path can't observe (the
/// decode step fuses the conv) is left empty and skipped.
#[derive(Clone, Debug, Default)]
pub struct KdaTaps {
    pub qkv_pre: Vec<f32>,
    pub conv_out: Vec<f32>,
    pub g: Vec<f32>,
    pub beta: Vec<f32>,
    pub o: Vec<f32>,
    pub g2: Vec<f32>,
    pub y_norm: Vec<f32>,
    pub out: Vec<f32>,
}

/// Parsed env config.
#[derive(Clone, Debug)]
pub struct Config {
    pub layers: Option<Vec<usize>>,
    pub windows: Option<usize>,
    pub tol: f64,
    pub resync: bool,
    pub poison: bool,
    pub dir: Option<PathBuf>,
    pub dumps: usize,
}

fn env_flag(k: &str, default: bool) -> bool {
    match std::env::var(k) {
        Ok(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
        Err(_) => default,
    }
}

impl Config {
    /// `None` unless `TANG_GLM5_KDA_DEBUG=1`.
    pub fn from_env() -> Option<Self> {
        if !std::env::var("TANG_GLM5_KDA_DEBUG").is_ok_and(|v| v == "1") {
            return None;
        }
        let layers = std::env::var("TANG_GLM5_KDA_DEBUG_LAYERS").ok().map(|s| {
            s.split(',')
                .filter_map(|p| p.trim().parse().ok())
                .collect::<Vec<usize>>()
        });
        Some(Self {
            layers,
            windows: std::env::var("TANG_GLM5_KDA_DEBUG_WINDOWS")
                .ok()
                .and_then(|s| s.parse().ok()),
            tol: std::env::var("TANG_GLM5_KDA_DEBUG_TOL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1e-3),
            resync: env_flag("TANG_GLM5_KDA_DEBUG_RESYNC", true),
            poison: env_flag("TANG_GLM5_KDA_DEBUG_POISON", true),
            dir: std::env::var("TANG_GLM5_KDA_DEBUG_DIR").ok().map(PathBuf::from),
            dumps: std::env::var("TANG_GLM5_KDA_DEBUG_DUMPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),
        })
    }
}

/// Whether device code should poison/tap (read once per call site; cheap).
pub fn poison_on() -> bool {
    std::env::var("TANG_GLM5_KDA_DEBUG").is_ok_and(|v| v == "1")
        && env_flag("TANG_GLM5_KDA_DEBUG_POISON", true)
}

// ---------------------------------------------------------------------------
// comparison

/// GPU-vs-CPU statistics of one tensor.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stat {
    pub n: usize,
    /// ‖gpu − cpu‖₂ / ‖cpu‖₂ (0 when both are ~0; ∞ when only the CPU is ~0)
    pub rel_l2: f64,
    pub max_abs: f64,
    pub max_idx: usize,
    pub norm_gpu: f64,
    pub norm_cpu: f64,
    pub cos: f64,
    /// exact 0.0 counts — the "kernel wrote nothing" / "kernel wrote zeros" signal
    pub zeros_gpu: usize,
    pub zeros_cpu: usize,
    /// NaN/Inf counts — with poisoning, NaN in gpu = a region no kernel wrote
    pub nonfinite_gpu: usize,
    pub nonfinite_cpu: usize,
    /// elements with |d| > tol·(|cpu| + rms(cpu))
    pub bad_elems: usize,
    pub len_mismatch: bool,
}

pub fn stat(gpu: &[f32], cpu: &[f32], tol: f64) -> Stat {
    let mut s = Stat {
        n: gpu.len().min(cpu.len()),
        len_mismatch: gpu.len() != cpu.len(),
        ..Default::default()
    };
    let (mut d2, mut g2, mut c2, mut gc) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &c) in gpu.iter().zip(cpu) {
        if g == 0.0 {
            s.zeros_gpu += 1;
        }
        if c == 0.0 {
            s.zeros_cpu += 1;
        }
        if !g.is_finite() {
            s.nonfinite_gpu += 1;
        }
        if !c.is_finite() {
            s.nonfinite_cpu += 1;
        }
        if !(g.is_finite() && c.is_finite()) {
            continue;
        }
        let (g, c) = (g as f64, c as f64);
        let d = (g - c).abs();
        if d > s.max_abs {
            s.max_abs = d;
        }
        d2 += d * d;
        g2 += g * g;
        c2 += c * c;
        gc += g * c;
    }
    // the max's index: a second pass so NaN entries can win it (they are the news)
    let mut best = -1f64;
    for (i, (&g, &c)) in gpu.iter().zip(cpu).enumerate() {
        let d = if g.is_finite() && c.is_finite() {
            ((g as f64) - (c as f64)).abs()
        } else {
            f64::INFINITY
        };
        if d > best {
            best = d;
            s.max_idx = i;
        }
    }
    s.norm_gpu = g2.sqrt();
    s.norm_cpu = c2.sqrt();
    s.rel_l2 = if s.norm_cpu > 1e-30 {
        d2.sqrt() / s.norm_cpu
    } else if s.norm_gpu > 1e-30 {
        f64::INFINITY
    } else {
        0.0
    };
    s.cos = if s.norm_gpu > 0.0 && s.norm_cpu > 0.0 {
        gc / (s.norm_gpu * s.norm_cpu)
    } else {
        0.0
    };
    let rms = if s.n > 0 { (c2 / s.n as f64).sqrt() } else { 0.0 };
    for (&g, &c) in gpu.iter().zip(cpu) {
        let d = ((g as f64) - (c as f64)).abs();
        if !d.is_finite() || d > tol * ((c as f64).abs() + rms) {
            s.bad_elems += 1;
        }
    }
    s
}

impl Stat {
    /// A tensor fails when it's off by more than `tol` (relative L2), has NaN/Inf the CPU
    /// doesn't, has notably more exact zeros (>1% of elements), or the lengths differ.
    pub fn bad(&self, tol: f64) -> bool {
        self.len_mismatch
            || self.rel_l2 > tol
            || self.nonfinite_gpu > self.nonfinite_cpu
            || self.zeros_gpu > self.zeros_cpu + self.n / 100
    }
}

/// How to name element `i` of a tap.
#[derive(Clone, Copy, Debug)]
pub enum Layout {
    /// `[t][nh][hd]` (one section of qkv, g, o, g2, y_norm)
    Inner { nh: usize, hd: usize },
    /// `[t][nh]`
    Heads { nh: usize },
    /// `[t][hidden]`
    Hidden { n: usize },
    /// recurrent state `[nh][hd(key)][hd(value)]`
    State { hd: usize },
    /// conv history `[3][nh][hd][taps]` (q|k|v channels, oldest tap first)
    Conv { nh: usize, hd: usize, taps: usize },
}

pub fn locate(l: Layout, i: usize) -> String {
    match l {
        Layout::Inner { nh, hd } => {
            let di = nh * hd;
            format!("tok{} h{} d{}", i / di, (i % di) / hd, i % hd)
        }
        Layout::Heads { nh } => format!("tok{} h{}", i / nh, i % nh),
        Layout::Hidden { n } => format!("tok{} c{}", i / n, i % n),
        Layout::State { hd } => {
            format!("h{} key{} val{}", i / (hd * hd), (i / hd) % hd, i % hd)
        }
        Layout::Conv { nh, hd, taps } => {
            let ch = i / taps;
            let di = nh * hd;
            let sec = ["q", "k", "v"].get(ch / di).copied().unwrap_or("?");
            format!("{sec} h{} d{} tap{}", (ch % di) / hd, ch % hd, i % taps)
        }
    }
}

/// Section `sec` (0=q, 1=k, 2=v) of a stacked `[t][3·di]` tensor, as `[t][di]`.
pub fn section(v: &[f32], t: usize, di: usize, sec: usize) -> Vec<f32> {
    if v.len() < t * 3 * di {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(t * di);
    for ti in 0..t {
        let b = ti * 3 * di + sec * di;
        out.extend_from_slice(&v[b..b + di]);
    }
    out
}

fn fmt_stat(name: &str, s: &Stat, l: Layout, tol: f64) -> String {
    let mut line = String::new();
    let _ = write!(
        line,
        "{:<11} {} rel {:.3e} max {:.3e} @{} |gpu| {:.4e} |cpu| {:.4e} cos {:.6} zeros {}/{} nonfinite {}/{} bad {}/{}",
        name,
        if s.bad(tol) { "BAD" } else { "ok " },
        s.rel_l2,
        s.max_abs,
        locate(l, s.max_idx),
        s.norm_gpu,
        s.norm_cpu,
        s.cos,
        s.zeros_gpu,
        s.zeros_cpu,
        s.nonfinite_gpu,
        s.nonfinite_cpu,
        s.bad_elems,
        s.n,
    );
    if s.len_mismatch {
        line.push_str(" LEN-MISMATCH");
    }
    line
}

/// What each stage reads, so a failing stage with passing inputs is "the" bug.
/// (name, inputs that must pass for this one to be the first cause, what runs here)
const PIPELINE: &[(&str, &[&str], &str)] = &[
    ("qkv.q", &[], "k5_gemv attn_q (device)"),
    ("qkv.k", &[], "k5_gemv attn_k (device)"),
    ("qkv.v", &[], "k5_gemv attn_v (device)"),
    ("g", &[], "k5_gemv f_a, f_b + k5_gate_finish (device)"),
    ("beta", &[], "k5_beta (device)"),
    ("conv.q", &["qkv.q"], "conv1d (device kda_conv / CPU tail conv1d_prefill)"),
    ("conv.k", &["qkv.k"], "conv1d"),
    ("conv.v", &["qkv.v"], "conv1d"),
    (
        "o",
        &["qkv.q", "qkv.k", "qkv.v", "g", "beta", "state.in"],
        "core: kda_step (decode) / kda_prefill (64-chunks) / CPU kda_chunk (tail)",
    ),
    ("g2", &[], "k5_gemv g_a, g_b (device)"),
    ("y_norm", &["o", "g2"], "k5_gated_norm (device)"),
    ("out", &["y_norm"], "k5_gemv attn_output (device)"),
    ("state", &["o"], "core state update (device st / tail seed)"),
    ("conv_state", &["qkv.q", "qkv.k", "qkv.v"], "conv commit (device h0) vs host shift"),
];

// ---------------------------------------------------------------------------
// the engine-side shadow

/// One shadow per KDA layer: the CPU's copy of its conv + recurrent state.
pub struct Shadow {
    pub cfg: Config,
    states: Vec<Option<(ConvState, KdaState)>>,
    windows: Vec<usize>,
    first: Option<String>,
    checked: usize,
    failed: usize,
    dumped: usize,
    worst: Vec<(f64, String)>,
}

impl Shadow {
    pub fn from_env(n_layer: usize) -> Option<Self> {
        let cfg = Config::from_env()?;
        eprintln!(
            "kdadbg ON: layers {:?} windows {:?} tol {:.1e} resync {} poison {} dir {:?}",
            cfg.layers, cfg.windows, cfg.tol, cfg.resync, cfg.poison, cfg.dir
        );
        if let Some(d) = &cfg.dir {
            let _ = std::fs::create_dir_all(d);
        }
        Some(Self {
            cfg,
            states: (0..n_layer).map(|_| None).collect(),
            windows: vec![0; n_layer],
            first: None,
            checked: 0,
            failed: 0,
            dumped: 0,
            worst: Vec::new(),
        })
    }

    /// Whether layer `l`'s next window is checked.
    pub fn wants(&self, l: usize) -> bool {
        self.cfg.layers.as_ref().is_none_or(|v| v.contains(&l))
            && self.cfg.windows.is_none_or(|n| self.windows[l] < n)
    }

    /// Seed the shadow from the engine's host state before the layer's first checked
    /// window (later windows resync from the device, or carry their own state).
    pub fn prime(&mut self, l: usize, conv: &ConvState, state: &KdaState) {
        if self.states[l].is_none() {
            self.states[l] = Some((conv.clone(), state.clone()));
        }
    }

    /// Compare one window. `gpu` are the device taps, `host_conv` the engine's host conv
    /// copy after the window, `dev_conv`/`dev_state` the device's state read back after.
    #[allow(clippy::too_many_arguments)]
    pub fn check(
        &mut self,
        l: usize,
        x: &[f32],
        t: usize,
        w: &KdaLayerW,
        kcfg: &KdaConfig,
        gpu: &KdaTaps,
        host_conv: &ConvState,
        dev_conv: &ConvState,
        dev_state: &KdaState,
        state_in_dev: Option<&KdaState>,
    ) -> Result<()> {
        let win = self.windows[l];
        self.windows[l] += 1;
        self.checked += 1;
        let tol = self.cfg.tol;
        let (nh, hd) = (kcfg.heads, kcfg.head_dim);
        let di = nh * hd;
        let hidden = w.hidden;
        let Some((sc, ss)) = self.states[l].as_mut() else {
            eprintln!("kdadbg w{win} l{l}: no shadow state (prime not called) — skipped");
            return Ok(());
        };
        // the shadow's state going IN (vs the device's, when the caller read it before)
        let mut rows: Vec<(String, Stat, Layout)> = Vec::new();
        if let Some(din) = state_in_dev {
            rows.push(("state.in".into(), stat(&din.0, &ss.0, tol), Layout::State { hd }));
        }
        let mut cpu = KdaTaps::default();
        let _ = super::kda::kda_layer_taps(x, w, kcfg, sc, ss, 1, t, Some(&mut cpu))?;
        let inner = Layout::Inner { nh, hd };
        for (sec, name) in [(0, "qkv.q"), (1, "qkv.k"), (2, "qkv.v")] {
            rows.push((
                name.into(),
                stat(&section(&gpu.qkv_pre, t, di, sec), &section(&cpu.qkv_pre, t, di, sec), tol),
                inner,
            ));
        }
        rows.push(("g".into(), stat(&gpu.g, &cpu.g, tol), inner));
        rows.push(("beta".into(), stat(&gpu.beta, &cpu.beta, tol), Layout::Heads { nh }));
        if !gpu.conv_out.is_empty() {
            for (sec, name) in [(0, "conv.q"), (1, "conv.k"), (2, "conv.v")] {
                rows.push((
                    name.into(),
                    stat(&section(&gpu.conv_out, t, di, sec), &section(&cpu.conv_out, t, di, sec), tol),
                    inner,
                ));
            }
        }
        rows.push(("o".into(), stat(&gpu.o, &cpu.o, tol), inner));
        rows.push(("g2".into(), stat(&gpu.g2, &cpu.g2, tol), inner));
        rows.push(("y_norm".into(), stat(&gpu.y_norm, &cpu.y_norm, tol), inner));
        rows.push(("out".into(), stat(&gpu.out, &cpu.out, tol), Layout::Hidden { n: hidden }));
        // state after: device vs shadow; conv: device h0 vs shadow vs the host copy
        rows.push(("state".into(), stat(&dev_state.0, &ss.0, tol), Layout::State { hd }));
        let conv_l = Layout::Conv { nh, hd, taps: kcfg.d_conv - 1 };
        rows.push(("conv_state".into(), stat(&dev_conv.0, &sc.0, tol), conv_l));
        rows.push(("conv_host".into(), stat(&host_conv.0, &sc.0, tol), conv_l));

        let xs = stat(x, x, tol);
        eprintln!(
            "kdadbg w{win} l{l} t{t} x |x| {:.4e} nonfinite {} (same input both paths)",
            xs.norm_cpu, xs.nonfinite_cpu
        );
        let mut any_bad = false;
        for (name, s, lay) in &rows {
            eprintln!("kdadbg w{win} l{l} t{t} {}", fmt_stat(name, s, *lay, tol));
            if s.bad(tol) {
                any_bad = true;
            }
            if s.rel_l2.is_finite() {
                self.worst.push((s.rel_l2, format!("w{win} l{l} {name}")));
            }
        }
        if any_bad {
            self.failed += 1;
            if self.first.is_none() {
                let bad = |n: &str| rows.iter().any(|(m, s, _)| m == n && s.bad(tol));
                let present = |n: &str| rows.iter().any(|(m, _, _)| m == n);
                // the earliest failing stage whose inputs (that we observed) all pass
                let cause = PIPELINE.iter().find(|(name, inputs, _)| {
                    present(name) && bad(name) && inputs.iter().all(|i| !bad(i))
                });
                let msg = match cause {
                    Some((name, inputs, what)) => format!(
                        "w{win} l{l} t{t}: stage `{name}` fails with its inputs {inputs:?} passing → suspect: {what}"
                    ),
                    None => format!(
                        "w{win} l{l} t{t}: failures with no clean-input stage (see the BAD rows)"
                    ),
                };
                eprintln!("kdadbg FIRST DIVERGENCE {msg}");
                self.first = Some(msg);
            }
        }
        let dump_this = (self.checked == 1) || (any_bad && self.dumped < self.cfg.dumps);
        if dump_this {
            if let Some(dir) = self.cfg.dir.clone() {
                if any_bad {
                    self.dumped += 1;
                }
                for (name, g, c) in [
                    ("qkv_pre", &gpu.qkv_pre, &cpu.qkv_pre),
                    ("conv_out", &gpu.conv_out, &cpu.conv_out),
                    ("g", &gpu.g, &cpu.g),
                    ("beta", &gpu.beta, &cpu.beta),
                    ("o", &gpu.o, &cpu.o),
                    ("g2", &gpu.g2, &cpu.g2),
                    ("y_norm", &gpu.y_norm, &cpu.y_norm),
                    ("out", &gpu.out, &cpu.out),
                ] {
                    write_f32(&dir.join(format!("w{win}_l{l}_{name}.gpu.f32")), g);
                    write_f32(&dir.join(format!("w{win}_l{l}_{name}.cpu.f32")), c);
                }
                write_f32(&dir.join(format!("w{win}_l{l}_state.gpu.f32")), &dev_state.0);
                write_f32(&dir.join(format!("w{win}_l{l}_state.cpu.f32")), &ss.0);
                write_f32(&dir.join(format!("w{win}_l{l}_x.f32")), x);
                eprintln!("kdadbg w{win} l{l}: tensors written to {}", dir.display());
            }
        }
        if self.cfg.resync {
            // the next window tests one step from the device's own state
            *sc = dev_conv.clone();
            *ss = dev_state.clone();
        }
        Ok(())
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        self.worst
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "kdadbg SUMMARY: {} windows checked, {} failed; first: {}",
            self.checked,
            self.failed,
            self.first.as_deref().unwrap_or("none (all within tol)")
        );
        for (r, what) in self.worst.iter().take(10) {
            eprintln!("kdadbg SUMMARY worst rel {r:.3e} {what}");
        }
    }
}

fn write_f32(p: &std::path::Path, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    if let Err(e) = std::fs::write(p, b) {
        eprintln!("kdadbg write {}: {e}", p.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_identical_is_clean() {
        let a = [1.0f32, -2.0, 3.0, 0.0];
        let s = stat(&a, &a, 1e-3);
        assert_eq!(s.rel_l2, 0.0);
        assert_eq!(s.zeros_gpu, 1);
        assert!(!s.bad(1e-3));
        assert!((s.cos - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stat_flags_dead_output() {
        // the "kernel wrote zeros" shape: gpu all zero where cpu is not
        let cpu: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
        let gpu = vec![0f32; 1000];
        let s = stat(&gpu, &cpu, 1e-3);
        assert!(s.bad(1e-3));
        assert!((s.rel_l2 - 1.0).abs() < 1e-9);
        assert!(s.zeros_gpu > s.zeros_cpu + 10);
    }

    #[test]
    fn stat_flags_unwritten_poison() {
        let cpu = vec![1f32; 8];
        let mut gpu = vec![1f32; 8];
        gpu[5] = f32::NAN;
        let s = stat(&gpu, &cpu, 1e-3);
        assert!(s.bad(1e-3));
        assert_eq!(s.nonfinite_gpu, 1);
        assert_eq!(s.max_idx, 5, "the NaN is the headline");
    }

    #[test]
    fn stat_near_zero_cpu_is_not_infinite_noise() {
        let s = stat(&[0.0, 0.0], &[0.0, 0.0], 1e-3);
        assert_eq!(s.rel_l2, 0.0);
        let s = stat(&[1e-3, 0.0], &[0.0, 0.0], 1e-3);
        assert!(s.rel_l2.is_infinite() && s.bad(1e-3));
    }

    #[test]
    fn section_and_locate() {
        // t=2, di=2: [q0 q1 k0 k1 v0 v1] per token
        let v: Vec<f32> = (0..12).map(|i| i as f32).collect();
        assert_eq!(section(&v, 2, 2, 1), vec![2.0, 3.0, 8.0, 9.0]);
        assert_eq!(locate(Layout::Inner { nh: 2, hd: 4 }, 13), "tok1 h1 d1");
        assert_eq!(locate(Layout::State { hd: 4 }, 16 + 4 * 2 + 3), "h1 key2 val3");
        assert_eq!(locate(Layout::Conv { nh: 1, hd: 2, taps: 3 }, 3 * 2 + 1), "k h0 d0 tap1");
    }
}
