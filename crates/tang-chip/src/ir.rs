//! The circuit IR: qwenputer's `chipformer/q35/ir.py` (MIT, Copyright (c) 2026 Divyansh
//! Agrawal), the two node types of a Qwen3.5 `linear_attention` layer, scheduled.
//!
//! Every value is a scalar dim of the residual stream; [`Expr`] is a linear form over dims (dim
//! [`ONE`] is the constant, carried by the ballast). An [`Ffn`] node is a sum of units
//! `coef · a · relu(b)` (ir.py:55 `FFN`); a [`Head`] is one Gated DeltaNet key head with its value
//! heads (ir.py:84 `Head`, ir.py:70 `VHead`), channel by channel. Nodes carry the sublayer
//! (`level`) the python scheduler placed them at: heads odd (`2L + 1`), FFNs even (`2L + 2`),
//! level 0 the embedding.
//!
//! The graph is built by python (`rv/core.py` + `rv/store_rv.py`, ~3.5k lines of circuit) and
//! exported as JSON by `scripts/chip/export_ir.py`; [`Graph::from_json`] reads that.

use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

/// Dim id of the constant (`dsl/graph.py` `ONE`).
pub const ONE: u32 = 0;
/// SiLU gate scale: `silu(K n) = K relu(n)` on the circuit's values (ir.py:38).
pub const K: f64 = 128.0;
/// Key/value head dim the IR is written for (ir.py:43 `HD`).
pub const HD: usize = 128;

/// A linear form `Σ coef · dim`, in python's dict order (coefficients are dyadic rationals,
/// exact in f64).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Expr(pub Vec<(u32, f64)>);

impl Expr {
    pub fn dim(d: u32, c: f64) -> Self {
        Expr(vec![(d, c)])
    }
    /// `dsl/graph.py` `Expr.__mul__`.
    pub fn scale(&self, c: f64) -> Self {
        Expr(
            self.0
                .iter()
                .map(|&(d, v)| (d, v * c))
                .filter(|t| t.1 != 0.0)
                .collect(),
        )
    }
    /// `dsl/graph.py` `Expr.__add__`: python dict order (self's keys, then new ones), zeros
    /// dropped.
    pub fn add(&self, o: &Expr) -> Self {
        let mut t = self.0.clone();
        for &(d, v) in &o.0 {
            match t.iter_mut().find(|e| e.0 == d) {
                Some(e) => e.1 += v,
                None => t.push((d, v)),
            }
        }
        Expr(t.into_iter().filter(|e| e.1 != 0.0).collect())
    }
    pub fn dims(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.iter().map(|e| e.0)
    }
}

/// A q/k/v channel: a linear form, then conv taps over `x_t, x_{t−1}, …` (ir.py:64 `Chan`).
#[derive(Clone, Debug)]
pub struct Chan {
    pub expr: Expr,
    pub taps: [f64; 4],
}

/// One value head of a key head (ir.py:70 `VHead`; `Head.vheads()` lists the first one too).
#[derive(Clone, Debug)]
pub struct VHead {
    /// Value column → value channel (columns 0, 1 carry the row ballast `A_REF`).
    pub v: Vec<(usize, Chan)>,
    /// Decay selector: `a ∈ {0, 110, 238}` gives `exp(g) ∈ {1, ½, 0}` (ir.py:42 `DECAY`).
    pub a: Expr,
    /// Value column → output gate `silu(z)`.
    pub z: Vec<(usize, Expr)>,
    /// `(value column, residual dim, weight)`.
    pub out: Vec<(usize, u32, f64)>,
    /// Write gate (0/1); `None`: always write.
    pub beta: Option<Expr>,
    /// Outputs add onto existing dims.
    pub into: bool,
}

#[derive(Clone, Debug)]
pub struct Head {
    pub name: String,
    pub level: u32,
    /// Outputs below this may round (a halving counter's tail).
    pub tiny: f64,
    /// Key-dim channel → query / key.
    pub q: Vec<(usize, Chan)>,
    pub k: Vec<(usize, Chan)>,
    pub vheads: Vec<VHead>,
}

#[derive(Clone, Debug)]
pub struct Ffn {
    pub name: String,
    pub level: u32,
    pub out: u32,
    /// `out` is an existing dim this node adds to.
    pub into: bool,
    /// `(coef, a, b)`: `coef · a · relu(b)`.
    pub terms: Vec<(f64, Expr, Expr)>,
}

#[derive(Clone, Debug)]
pub enum Node {
    Ffn(Ffn),
    Head(Head),
}

impl Node {
    pub fn level(&self) -> u32 {
        match self {
            Node::Ffn(n) => n.level,
            Node::Head(n) => n.level,
        }
    }
    pub fn name(&self) -> &str {
        match self {
            Node::Ffn(n) => &n.name,
            Node::Head(n) => &n.name,
        }
    }
    /// Every dim the node reads (emit.py:70 `reads`).
    pub fn reads(&self) -> Vec<u32> {
        match self {
            Node::Ffn(n) => n
                .terms
                .iter()
                .flat_map(|(_, a, b)| a.dims().chain(b.dims()))
                .collect(),
            Node::Head(n) => {
                let mut r: Vec<u32> =
                    n.q.iter()
                        .chain(&n.k)
                        .flat_map(|(_, c)| c.expr.dims())
                        .collect();
                for vh in &n.vheads {
                    r.extend(vh.v.iter().flat_map(|(_, c)| c.expr.dims()));
                    r.extend(vh.a.dims());
                    r.extend(vh.z.iter().flat_map(|(_, e)| e.dims()));
                    if let Some(b) = &vh.beta {
                        r.extend(b.dims());
                    }
                }
                r
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dim {
    pub name: String,
    /// `one` | `token` | `ffn` | `head`.
    pub kind: String,
    pub level: u32,
}

/// Per-layer capacity the graph was scheduled for (ir.py `Target`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub struct Target {
    pub key_heads: usize,
    pub value_heads: usize,
    pub intermediate: usize,
}

/// A scheduled circuit (ir.py `Graph`) plus the dims the output side reads.
#[derive(Clone, Debug)]
pub struct Graph {
    pub target: Target,
    pub dims: Vec<Dim>,
    /// Dims set by the embedding (token features), sorted.
    pub token_dims: Vec<u32>,
    /// FFN units the scheduler placed per sublayer (`Graph.units`).
    pub units: BTreeMap<u32, usize>,
    /// Dims that can hold large values (never handed over by a subtraction in their MLP).
    pub big: Vec<u32>,
    /// Dims read after the circuit (the lm_head's output code, tau, mu, the store's id code).
    pub keep: Vec<u32>,
    pub nodes: Vec<Node>,
    pub code: Vec<u32>,
    pub tau: u32,
    pub mu: u32,
    pub id_code: Vec<u32>,
    /// Python's allocation of the same graph, when exported (a parity reference).
    pub python_alloc: Option<PyAlloc>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PyAlloc {
    pub slots: BTreeMap<String, usize>,
    pub n_slots: usize,
    pub subtract: BTreeMap<String, u32>,
    pub clears: BTreeMap<String, Vec<u32>>,
    pub final_: Option<Vec<u32>>,
}

// --- JSON (scripts/chip/export_ir.py) ------------------------------------------------------

type JExpr = Vec<(u32, f64)>;
type JChans = Vec<(usize, JExpr, Vec<f64>)>;

#[derive(Deserialize)]
struct JVHead {
    v: JChans,
    a: JExpr,
    z: Vec<(usize, JExpr)>,
    out: Vec<(usize, u32, f64)>,
    beta: Option<JExpr>,
    into: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum JNode {
    Ffn {
        ffn: String,
        level: u32,
        out: u32,
        into: bool,
        terms: Vec<(f64, JExpr, JExpr)>,
    },
    Head {
        head: String,
        level: u32,
        tiny: f64,
        q: JChans,
        k: JChans,
        vheads: Vec<JVHead>,
    },
}

#[derive(Deserialize)]
struct JAlloc {
    slots: BTreeMap<String, usize>,
    n_slots: usize,
    subtract: BTreeMap<String, u32>,
    clears: BTreeMap<String, Vec<u32>>,
    #[serde(rename = "final")]
    final_: Vec<u32>,
}

#[derive(Deserialize)]
struct JGraph {
    target: Target,
    dims: Vec<(String, String, u32)>,
    token_dims: Vec<u32>,
    units: BTreeMap<String, usize>,
    big: Vec<u32>,
    keep: Vec<u32>,
    nodes: Vec<JNode>,
    code: Vec<u32>,
    tau: u32,
    mu: u32,
    id_code: Vec<u32>,
    python_alloc: Option<JAlloc>,
}

fn chans(c: JChans) -> Result<Vec<(usize, Chan)>> {
    c.into_iter()
        .map(|(i, e, t)| {
            ensure!(i < HD && t.len() == 4, "channel {i}: {} taps", t.len());
            Ok((
                i,
                Chan {
                    expr: Expr(e),
                    taps: [t[0], t[1], t[2], t[3]],
                },
            ))
        })
        .collect()
}

impl Graph {
    pub fn from_json(s: &str) -> Result<Self> {
        let j: JGraph = serde_json::from_str(s).context("circuit IR json")?;
        let nodes = j
            .nodes
            .into_iter()
            .map(|n| -> Result<Node> {
                Ok(match n {
                    JNode::Ffn {
                        ffn,
                        level,
                        out,
                        into,
                        terms,
                    } => Node::Ffn(Ffn {
                        name: ffn,
                        level,
                        out,
                        into,
                        terms: terms
                            .into_iter()
                            .map(|(c, a, b)| (c, Expr(a), Expr(b)))
                            .collect(),
                    }),
                    JNode::Head {
                        head,
                        level,
                        tiny,
                        q,
                        k,
                        vheads,
                    } => Node::Head(Head {
                        name: head,
                        level,
                        tiny,
                        q: chans(q)?,
                        k: chans(k)?,
                        vheads: vheads
                            .into_iter()
                            .map(|v| -> Result<VHead> {
                                Ok(VHead {
                                    v: chans(v.v)?,
                                    a: Expr(v.a),
                                    z: v.z.into_iter().map(|(i, e)| (i, Expr(e))).collect(),
                                    out: v.out,
                                    beta: v.beta.map(Expr),
                                    into: v.into,
                                })
                            })
                            .collect::<Result<_>>()?,
                    }),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let units = j
            .units
            .into_iter()
            .map(|(k, v)| Ok((k.parse()?, v)))
            .collect::<Result<_>>()?;
        Ok(Graph {
            target: j.target,
            dims: j
                .dims
                .into_iter()
                .map(|(name, kind, level)| Dim { name, kind, level })
                .collect(),
            token_dims: j.token_dims,
            units,
            big: j.big,
            keep: j.keep,
            nodes,
            code: j.code,
            tau: j.tau,
            mu: j.mu,
            id_code: j.id_code,
            python_alloc: j.python_alloc.map(|a| PyAlloc {
                slots: a.slots,
                n_slots: a.n_slots,
                subtract: a.subtract,
                clears: a.clears,
                final_: Some(a.final_),
            }),
        })
    }

    /// From gzip'd JSON.
    pub fn from_json_gz(b: &[u8]) -> Result<Self> {
        let mut s = String::new();
        std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(b), &mut s)?;
        Self::from_json(&s)
    }

    /// Decoder layers the graph occupies (ir.py:235 `n_layers`).
    pub fn n_layers(&self) -> usize {
        let top = self.nodes.iter().map(|n| n.level()).max().unwrap_or(0);
        (top as usize).div_ceil(2)
    }

    pub fn heads(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, Node::Head(_)))
            .count()
    }
    pub fn ffn_units(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| match n {
                Node::Ffn(f) => f.terms.len(),
                _ => 0,
            })
            .sum()
    }
}
