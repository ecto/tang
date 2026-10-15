//! One graph as several kernels, for graphs too large for one shader
//! (a traced contact step's adjoint is hundreds of thousands of nodes, which
//! shader compilers take minutes and tens of GB to optimize, or reject).

use std::collections::HashMap;
use std::fmt::Write;

use crate::codegen::{expr_str, ref_str};
use crate::graph::ExprGraph;
use crate::node::{ExprId, Node};

/// WGSL kernels that together compute a graph's outputs, run in order over
/// the same environments: each reads `inputs` (`n_inputs` per environment)
/// and a per-environment `scratch` (`n_scratch` floats), and the last writes
/// `outputs`. Bindings: 0 `inputs` (read), 1 `scratch` (read_write),
/// 2 `outputs` (read_write), 3 the uniform `count`; entry point `main`.
pub struct SplitKernels {
    /// Kernel sources, in launch order.
    pub parts: Vec<String>,
    /// Scratch floats per environment.
    pub n_scratch: usize,
    /// Workgroup size of every part.
    pub workgroup_size: u32,
}

fn children(node: Node) -> [Option<ExprId>; 3] {
    match node {
        Node::Var(_) | Node::Lit(_) => [None; 3],
        Node::Add(a, b) | Node::Mul(a, b) | Node::Atan2(a, b) => [Some(a), Some(b), None],
        Node::Neg(a)
        | Node::Recip(a)
        | Node::Sqrt(a)
        | Node::Sin(a)
        | Node::Exp2(a)
        | Node::Log2(a) => [Some(a), None, None],
        Node::Select(c, a, b) => [Some(c), Some(a), Some(b)],
    }
}

impl ExprGraph {
    /// [`to_kernel`](Self::to_kernel) for WGSL, cut into kernels of at most
    /// `max_nodes` evaluated nodes each (topological order); a value used
    /// outside the kernel that computes it goes through `scratch`.
    pub fn to_wgsl_split(
        &self,
        outputs: &[ExprId],
        n_inputs: usize,
        max_nodes: usize,
    ) -> SplitKernels {
        let wg = 256u32;
        let live = self.live_set(outputs);
        let mut order: Vec<usize> = live
            .iter()
            .copied()
            .filter(|&i| !matches!(self.node(ExprId(i as u32)), Node::Var(_) | Node::Lit(_)))
            .collect();
        order.sort_unstable();
        let chunks: Vec<&[usize]> = order.chunks(max_nodes.max(1)).collect();
        let nparts = chunks.len().max(1);
        let mut part_of: HashMap<usize, usize> = HashMap::new();
        for (p, c) in chunks.iter().enumerate() {
            for &i in *c {
                part_of.insert(i, p);
            }
        }
        // scratch slots: values read by a later part (or an output, read by
        // the last part) outside the part that computes them
        let mut slot: HashMap<usize, usize> = HashMap::new();
        let need = |i: usize, reader: usize, slot: &mut HashMap<usize, usize>| {
            if let Some(&p) = part_of.get(&i) {
                if p != reader && !slot.contains_key(&i) {
                    let n = slot.len();
                    slot.insert(i, n);
                }
            }
        };
        for (p, c) in chunks.iter().enumerate() {
            for &i in *c {
                for ch in children(self.node(ExprId(i as u32))).into_iter().flatten() {
                    need(ch.0 as usize, p, &mut slot);
                }
            }
        }
        for o in outputs {
            need(o.0 as usize, nparts - 1, &mut slot);
        }
        let ns = slot.len().max(1);
        let header = format!(
            "struct Params {{ count: u32, _p1: u32, _p2: u32, _p3: u32 }}\n\
             @group(0) @binding(0) var<storage, read> inputs: array<f32>;\n\
             @group(0) @binding(1) var<storage, read_write> scratch: array<f32>;\n\
             @group(0) @binding(2) var<storage, read_write> outputs: array<f32>;\n\
             @group(0) @binding(3) var<uniform> params: Params;\n\
             @compute @workgroup_size({wg})\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let idx = gid.x;\n    if (idx >= params.count) {{ return; }}\n"
        );
        let mut parts = Vec::with_capacity(nparts);
        for p in 0..nparts {
            let chunk: &[usize] = chunks.get(p).copied().unwrap_or(&[]);
            let mut src = header.clone();
            let mut loaded: std::collections::HashSet<usize> = std::collections::HashSet::new();
            let mut body = String::new();
            let load = |id: ExprId,
                        pre: &mut String,
                        loaded: &mut std::collections::HashSet<usize>| {
                let i = id.0 as usize;
                if !loaded.insert(i) {
                    return;
                }
                match self.node(id) {
                    Node::Var(n) => {
                        writeln!(pre, "    let x{n} = inputs[idx * {n_inputs}u + {n}u];").unwrap()
                    }
                    Node::Lit(_) => {}
                    _ => {
                        if part_of.get(&i) != Some(&p) {
                            writeln!(pre, "    let t{i} = scratch[idx * {ns}u + {}u];", slot[&i])
                                .unwrap();
                        }
                    }
                }
            };
            let mut pre = String::new();
            for &i in chunk {
                let node = self.node(ExprId(i as u32));
                for ch in children(node).into_iter().flatten() {
                    load(ch, &mut pre, &mut loaded);
                }
                loaded.insert(i);
                writeln!(
                    body,
                    "    let t{i} = {};",
                    expr_str(self, node, "", crate::codegen::Dialect::Wgsl)
                )
                .unwrap();
                if let Some(s) = slot.get(&i) {
                    writeln!(body, "    scratch[idx * {ns}u + {s}u] = t{i};").unwrap();
                }
            }
            if p == nparts - 1 {
                for o in outputs {
                    load(*o, &mut pre, &mut loaded);
                }
                let no = outputs.len();
                for (k, o) in outputs.iter().enumerate() {
                    writeln!(
                        body,
                        "    outputs[idx * {no}u + {k}u] = {};",
                        ref_str(self, *o, "")
                    )
                    .unwrap();
                }
            }
            src += &pre;
            src += &body;
            src += "}\n";
            parts.push(src);
        }
        SplitKernels {
            parts,
            n_scratch: ns,
            workgroup_size: wg,
        }
    }
}
