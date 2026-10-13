//! Precision search through [`tang_gpu::program::Program`], on whichever
//! backend the [`Runtime`] is: the CUDA counterpart of
//! `ir_msl::tune_precision`, with the same flow.
//!
//! 1. **Tile moves.** For every tensor-core kernel
//!    ([`TensorGraph::cuda_tile_operands`]), its plain operands go to f16
//!    (or bf16) together. That is what puts the kernel on m16n8k16 half
//!    fragments instead of TF32; one operand alone leaves it on the
//!    transpiled fallback.
//! 2. **Per buffer.** Every remaining f32 buffer, largest first.
//! 3. Each candidate is run against the f64 reference. It is kept only if
//!    the program stays within the [`ErrorBudget`] and gets at least 3%
//!    faster than the schedule so far. At the end, the result is re-timed
//!    against the f32 baseline, and the change is dropped if the gain is
//!    under 1%.
//!
//! Accumulators stay f32: the tensor-core kernels accumulate in fp32, and
//! an f16 accumulator would put a kernel on the transpiled path.
//!
//! Candidates lower with the default tile shape (`Runtime::lower`). Tune
//! the result's tile shapes afterwards with `Program::tuned`.
use tang_compile::tensor::{ErrorBudget, ErrorStats, Format, Precision, TensorGraph, TensorId};
use tang_gpu::program::{Program, Runtime};

/// One candidate the search tried.
pub struct PrecisionTrial {
    pub what: String,
    pub error: ErrorStats,
    /// Whole-program time (ms); `None` when over budget.
    pub ms: Option<f64>,
    pub kept: bool,
}

pub struct PrecisionTune {
    pub graph: TensorGraph,
    /// Nodes whose precision changed from f32.
    pub assignment: Vec<(TensorId, Precision)>,
    pub error: ErrorStats,
    pub base_error: ErrorStats,
    pub base_ms: f64,
    pub ms: f64,
    pub trials: Vec<PrecisionTrial>,
}

/// One move: buffers set to one storage format.
#[derive(Clone)]
pub struct Move {
    pub what: String,
    pub buffers: Vec<TensorId>,
}

/// The search's moves on `g`, in order: tile operand groups, then single
/// buffers, largest first. Buffers in `exclude` stay f32 (a training
/// step's weights and optimizer state: fp32 master copies; a half tile
/// converts an f32 operand while staging).
pub fn moves(g: &TensorGraph, outputs: &[TensorId], exclude: &[TensorId]) -> Vec<Move> {
    let mut out: Vec<Move> = g
        .cuda_tile_operands(outputs)
        .into_iter()
        .map(|(k, bufs)| {
            let bufs: Vec<TensorId> = bufs.into_iter().filter(|b| !exclude.contains(b)).collect();
            (k, bufs)
        })
        .filter(|(_, bufs)| !bufs.is_empty())
        .map(|(k, bufs)| Move {
            what: format!(
                "tile {} operands {:?}",
                k.index(),
                bufs.iter().map(|b| b.index()).collect::<Vec<_>>()
            ),
            buffers: bufs,
        })
        .collect();
    let mut bufs: Vec<(TensorId, usize)> = g
        .to_cuda(outputs)
        .buffers
        .iter()
        .filter(|b| !exclude.contains(&b.node))
        .map(|b| (b.node, g.buffer_bytes(b.node)))
        .collect();
    bufs.sort_by_key(|b| std::cmp::Reverse(b.1));
    out.extend(bufs.into_iter().map(|(b, _)| Move {
        what: format!("buffer {}", b.index()),
        buffers: vec![b],
    }));
    out
}

/// `g` with `m`'s buffers stored in `f`.
pub fn apply(g: &TensorGraph, m: &Move, f: Format) -> TensorGraph {
    let mut v = g.clone();
    for &b in &m.buffers {
        let mut p = v.precision(b);
        p.storage = f;
        v.set_precision(b, p);
    }
    v
}

/// GPU error of `p`'s `outputs` against `want`.
pub fn program_error(
    rt: &Runtime,
    p: &Program,
    outputs: &[TensorId],
    want: &[Vec<f64>],
) -> ErrorStats {
    p.run(rt);
    let got: Vec<Vec<f64>> = outputs
        .iter()
        .map(|&o| p.read(rt, o).iter().map(|&v| v as f64).collect())
        .collect();
    ErrorStats::compare(&got, want)
}

/// Every kernel source the search could compile: the baseline, each
/// move on its own and every prefix of the moves taken together, in each
/// of `formats`. Hand them to `tang_gpu::cuda::precompile` before the
/// search (no GPU needed), so that nvrtc does not run while the GPU is
/// held. Sources for other combinations of kept moves still compile
/// during the search.
pub fn candidate_sources(
    g: &TensorGraph,
    outputs: &[TensorId],
    exclude: &[TensorId],
    formats: &[Format],
) -> Vec<String> {
    let ms = moves(g, outputs, exclude);
    let mut graphs = vec![g.clone()];
    for &f in formats {
        let mut acc = g.clone();
        for m in &ms {
            graphs.push(apply(g, m, f));
            acc = apply(&acc, m, f);
            graphs.push(acc.clone());
        }
    }
    let mut out: Vec<String> = graphs
        .iter()
        .flat_map(|v| v.to_cuda(outputs).kernels.into_iter().map(|k| k.source))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Greedy precision search on `rt` (see the module docs). `formats` are
/// the storage formats tried per move, in order (e.g. `[F16, BF16]`).
/// The fastest admitted format is taken if it beats the current schedule
/// by 3%.
#[allow(clippy::too_many_arguments)]
pub fn tune_precision(
    rt: &Runtime,
    g: TensorGraph,
    outputs: &[TensorId],
    inputs: &[&[f32]],
    want: &[Vec<f64>],
    budget: ErrorBudget,
    exclude: &[TensorId],
    formats: &[Format],
    iters: usize,
) -> PrecisionTune {
    let measure = |g: &TensorGraph| {
        let p = Program::compile(rt, g, outputs, inputs);
        let err = program_error(rt, &p, outputs, want);
        let ms = budget.admits(&err).then(|| p.time(rt, iters));
        (err, ms)
    };
    let _ = measure(&g);
    let (base_error, base_ms) = measure(&g);
    let base_ms =
        base_ms.unwrap_or_else(|| Program::compile(rt, &g, outputs, inputs).time(rt, iters));
    let base = g.clone();
    let mut cur = g;
    let (mut cur_err, mut cur_ms) = (base_error, base_ms);
    let mut trials = Vec::new();
    for m in moves(&cur, outputs, exclude) {
        if m.buffers
            .iter()
            .all(|&b| cur.precision(b).storage != Format::F32)
        {
            continue;
        }
        let mut best: Option<(TensorGraph, ErrorStats, f64)> = None;
        for &f in formats {
            let v = apply(&cur, &m, f);
            let (err, ms) = measure(&v);
            trials.push(PrecisionTrial {
                what: format!("{} {f:?}", m.what),
                error: err,
                ms,
                kept: false,
            });
            if let Some(t) = ms {
                if best.as_ref().is_none_or(|b| t < b.2) {
                    best = Some((v, err, t));
                }
            }
        }
        if let Some((v, err, t)) = best {
            if t < cur_ms * 0.97 {
                let tag = format!("{:?}", v.precision(m.buffers[0]).storage);
                if let Some(tr) = trials
                    .iter_mut()
                    .rev()
                    .find(|tr| tr.what == format!("{} {tag}", m.what))
                {
                    tr.kept = true;
                }
                cur = v;
                cur_err = err;
                cur_ms = t;
            }
        }
    }
    let assignment: Vec<(TensorId, Precision)> = (0..cur.nodes().len())
        .map(TensorId::from_index)
        .filter(|&t| cur.precision(t) != Precision::F32)
        .map(|t| (t, cur.precision(t)))
        .collect();
    let (mut base_ms, mut cur_ms) = (base_ms, cur_ms);
    if !assignment.is_empty() {
        let b = Program::compile(rt, &base, outputs, inputs);
        let c = Program::compile(rt, &cur, outputs, inputs);
        base_ms = f64::INFINITY;
        cur_ms = f64::INFINITY;
        for _ in 0..3 {
            base_ms = base_ms.min(b.time(rt, iters));
            cur_ms = cur_ms.min(c.time(rt, iters));
        }
        if cur_ms >= base_ms * 0.99 {
            return PrecisionTune {
                graph: base,
                assignment: Vec::new(),
                error: base_error,
                base_error,
                base_ms,
                ms: base_ms,
                trials,
            };
        }
    }
    PrecisionTune {
        graph: cur,
        assignment,
        error: cur_err,
        base_error,
        base_ms,
        ms: cur_ms,
        trials,
    }
}

/// A precision assignment as text, one `node storage` per line (for
/// handing a search's result to a later, separately locked run).
pub fn write_assignment(assignment: &[(TensorId, Precision)]) -> String {
    assignment
        .iter()
        .map(|(t, p)| format!("{} {:?}\n", t.index(), p.storage))
        .collect()
}

/// Inverse of [`write_assignment`], applied to `g`.
pub fn read_assignment(g: &mut TensorGraph, text: &str) {
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(n), Some(f)) = (it.next(), it.next()) else {
            continue;
        };
        let t = TensorId::from_index(n.parse().expect("node"));
        let f = match f {
            "F16" => Format::F16,
            "BF16" => Format::BF16,
            _ => Format::F32,
        };
        let mut p = g.precision(t);
        p.storage = f;
        g.set_precision(t, p);
    }
}

/// f64 outputs to a file (a reference computed outside the GPU lock).
pub fn write_f64s(path: &str, v: &[Vec<f64>]) {
    let mut bytes = Vec::new();
    bytes.extend((v.len() as u64).to_le_bytes());
    for o in v {
        bytes.extend((o.len() as u64).to_le_bytes());
        for x in o {
            bytes.extend(x.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}

/// Inverse of [`write_f64s`].
pub fn read_f64s(path: &str) -> Vec<Vec<f64>> {
    let b = std::fs::read(path).unwrap();
    let word = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
    let mut at = 8;
    (0..word(0))
        .map(|_| {
            let n = word(at) as usize;
            at += 8;
            let o = (0..n).map(|i| f64::from_bits(word(at + 8 * i))).collect();
            at += 8 * n;
            o
        })
        .collect()
}
