//! Native Metal lowering of matmul-shaped tiles: `simdgroup_matrix`.
//!
//! WGSL cannot name a tensor core. When a tiled reduction is a plain
//! bilinear sum — body `a·b` over one reduce axis, each operand read
//! along one of the two tile axes and the reduce axis — the same
//! comprehension lowers to Metal Shading Language with 8×8 simdgroup
//! matrix fragments: a 64×64 block per threadgroup, four simdgroups of
//! 32×32, operands staged through threadgroup memory from any affine
//! layout (transposed, strided, batched), and the comp's epilogue map
//! (bias, activation) run on the stored block before the write-out.
//!
//! Everything else — the staging, the bounds, the epilogue — is the
//! same comprehension semantics as the WGSL tile; only the inner
//! product is an intrinsic. A kernel that does not fit the pattern
//! falls back to WGSL, so a program is a mix.
use std::fmt::Write;

use tang_expr::Node;

use super::tile::emit_scalar;
use super::wgsl::WgslKernel;
use super::{fixed, numel, Arg, TensorGraph, TensorId};

/// Rows × cols of output per threadgroup.
const BLOCK: usize = 64;
/// Reduce elements staged per step.
const CHUNK: usize = 32;
const THREADS: usize = 128;
/// Row strides of the staged tiles, padded past the bank period so the
/// 8 rows of a fragment load land in different banks.
/// Padding of the staged tiles' row strides past the bank period, so
/// the 8 rows of a fragment load land in different banks.
const PAD: usize = 4;

/// An operand's flat address as an affine form over the comp's space.
pub(super) struct Addr {
    pub offset: i64,
    pub coefs: Vec<i64>,
}

/// A matmul-shaped tile (see [`TensorGraph::bilinear_tile`]).
pub(super) struct Bilinear {
    pub red_node: usize,
    pub epilogue: bool,
    /// buffers (storage nodes) of the two operands
    pub buf_a: usize,
    pub buf_b: usize,
    /// their addresses over the reduction's space; A has no coefficient
    /// on `dx`, B none on `dy`
    pub addr_a: Addr,
    pub addr_b: Addr,
    pub pts: Vec<usize>,
    /// tile axes: columns, rows; the remaining point dims are `others`
    pub dx: usize,
    pub dy: usize,
    pub others: Vec<usize>,
    pub m: usize,
    pub n: usize,
    pub k_total: usize,
    /// index of the reduce axis in the space
    pub kdim: usize,
}

impl TensorGraph {
    /// Flat address of `a` (row-major over its source's shape) as an
    /// affine form over the iteration space.
    pub(super) fn flat_addr(&self, a: &Arg, rank: usize) -> Addr {
        let shape = fixed(&self.nodes[a.src.n()].shape);
        let mut stride = 1i64;
        let mut addr = Addr {
            offset: 0,
            coefs: vec![0; rank],
        };
        for (f, &n) in a.view.forms.iter().zip(&shape).rev() {
            addr.offset += f.offset * stride;
            for (c, &x) in addr.coefs.iter_mut().zip(&f.coefs) {
                *c += x * stride;
            }
            stride *= n as i64;
        }
        addr
    }

    /// The bilinear tile behind `root`, if `root` is (or is the epilogue
    /// of) a reduction over one axis whose body is a product of two leaf
    /// reads, each varying along one of the two tile axes and the reduce
    /// axis — the shape a matrix unit computes. Shared by the native
    /// lowerings (Metal simdgroup matrices, CUDA wmma).
    pub(super) fn bilinear_tile(&self, root: usize, reads: &[usize]) -> Option<Bilinear> {
        let c = self.comp_at(root)?;
        let red_node = self.tiled_reduction(root)?;
        let rc = self.comp_at(red_node)?;
        let epilogue = red_node != root;
        if rc.red.len() != 1 || rc.width() != 1 || rc.tuple != 0 {
            return None;
        }
        // the epilogue must read the reduction at its own coordinates
        if epilogue
            && self.epilogue_read(c, red_node)?
                != super::View::identity(rc.points().len(), rc.points().len()).forms
        {
            return None;
        }
        if c.points().len() != rc.points().len() {
            return None;
        }
        // body: exactly a product of two leaf reads
        let nodes = rc.body.nodes_slice();
        let (va, vb) = match nodes[rc.body_outs[0].index() as usize] {
            Node::Mul(x, y) => match (nodes[x.index() as usize], nodes[y.index() as usize]) {
                (Node::Var(p), Node::Var(q)) => (p as usize, q as usize),
                _ => return None,
            },
            _ => return None,
        };
        let pts = fixed(rc.points());
        let k_total = rc.red[0].n();
        let rank = rc.rank();
        let kdim = rank - 1;
        let mut non_unit = (0..pts.len()).rev().filter(|&d| pts[d] > 1);
        let dx = non_unit.next()?;
        let dy = non_unit.next()?;
        let others: Vec<usize> = (0..pts.len()).filter(|&d| d != dx && d != dy).collect();
        let (n, m) = (pts[dx], pts[dy]);
        let addr_a = self.flat_addr(&rc.args[va], rank);
        let addr_b = self.flat_addr(&rc.args[vb], rank);
        // A varies along rows (dy) and k, B along cols (dx) and k
        let (a_arg, b_arg, addr_a, addr_b) = if addr_a.coefs[dx] == 0 && addr_b.coefs[dy] == 0 {
            (va, vb, addr_a, addr_b)
        } else if addr_a.coefs[dy] == 0 && addr_b.coefs[dx] == 0 {
            (vb, va, addr_b, addr_a)
        } else {
            return None;
        };
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let is_buffer = |j: usize| reads.contains(&storage(j));
        if !is_buffer(rc.args[a_arg].src.n()) || !is_buffer(rc.args[b_arg].src.n()) {
            return None;
        }
        // epilogue args: the reduction, or buffers read at affine addresses
        if epilogue {
            for a in &c.args {
                if a.src.n() != red_node && !is_buffer(a.src.n()) {
                    return None;
                }
            }
        }
        Some(Bilinear {
            red_node,
            epilogue,
            buf_a: storage(rc.args[a_arg].src.n()),
            buf_b: storage(rc.args[b_arg].src.n()),
            addr_a,
            addr_b,
            pts,
            dx,
            dy,
            others,
            m,
            n,
            k_total,
            kdim,
        })
    }

    /// The Metal simdgroup-matrix kernel for `root`, if it is (or is the
    /// epilogue of) a bilinear reduction with a 2-D tile; `None` means
    /// use the WGSL tile.
    pub(super) fn msl_mma_kernel(&self, root: usize, reads: &[usize]) -> Option<WgslKernel> {
        #[allow(non_snake_case)]
        let (LDA, LDB, LDC) = (CHUNK + PAD, BLOCK + PAD, BLOCK + PAD);
        let c = self.comp_at(root)?;
        let t = self.bilinear_tile(root, reads)?;
        let Bilinear {
            red_node,
            epilogue,
            buf_a,
            buf_b,
            addr_a,
            addr_b,
            pts,
            dx,
            dy,
            others,
            m,
            n,
            k_total,
            kdim,
        } = t;
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let tiles_x = n.div_ceil(BLOCK);
        let tiles_y = m.div_ceil(BLOCK);
        let n_tiles = tiles_x * tiles_y * others.iter().map(|&d| pts[d]).product::<usize>();
        let w_out = c.width();

        let mut s = String::new();
        writeln!(s, "#include <metal_stdlib>").unwrap();
        writeln!(s, "#include <metal_simdgroup_matrix>").unwrap();
        writeln!(s, "using namespace metal;").unwrap();
        writeln!(
            s,
            "// Auto-generated by tang-compile: simdgroup-matrix tile"
        )
        .unwrap();
        writeln!(
            s,
            "struct Params {{ uint count; uint p1; uint p2; uint p3; }};"
        )
        .unwrap();
        let mut params: Vec<String> = reads
            .iter()
            .enumerate()
            .map(|(b, &j)| format!("device const float* b{j} [[buffer({b})]]"))
            .collect();
        params.push(format!("device float* out [[buffer({})]]", reads.len()));
        params.push(format!(
            "constant Params& params [[buffer({})]]",
            reads.len() + 1
        ));
        params.push("uint3 wid [[threadgroup_position_in_grid]]".into());
        params.push("uint tid [[thread_index_in_threadgroup]]".into());
        params.push("uint sg [[simdgroup_index_in_threadgroup]]".into());
        writeln!(s, "kernel void main0({}) {{", params.join(", ")).unwrap();
        // one threadgroup arena: the A and B tiles during the reduce,
        // the stored block (C) afterwards
        let arena = (BLOCK * LDA + CHUNK * LDB).max(BLOCK * LDC);
        writeln!(s, "    threadgroup float smem[{arena}];").unwrap();
        writeln!(s, "    threadgroup float* As = smem;").unwrap();
        writeln!(s, "    threadgroup float* Bs = smem + {}u;", BLOCK * LDA).unwrap();
        writeln!(s, "    threadgroup float* Cs = smem;").unwrap();
        writeln!(s, "    uint tile = wid.x + wid.y * 65535u;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) {{ return; }}").unwrap();
        writeln!(s, "    uint tile_x = tile % {tiles_x}u;").unwrap();
        writeln!(s, "    uint tile_y = (tile / {tiles_x}u) % {tiles_y}u;").unwrap();
        writeln!(s, "    uint lead = tile / {}u;", tiles_x * tiles_y).unwrap();
        for &d in others.iter().rev() {
            writeln!(s, "    uint p{d} = lead % {}u;", pts[d]).unwrap();
            writeln!(s, "    lead = lead / {}u;", pts[d]).unwrap();
        }
        writeln!(s, "    uint row0 = tile_y * {BLOCK}u;").unwrap();
        writeln!(s, "    uint col0 = tile_x * {BLOCK}u;").unwrap();
        // the leading-dim part of each operand's address, uniform per tile
        let lead_term = |addr: &Addr| -> String {
            let mut t = format!("{}", addr.offset);
            for &d in &others {
                if addr.coefs[d] != 0 {
                    t.push_str(&format!(" + {} * int(p{d})", addr.coefs[d]));
                }
            }
            t
        };
        writeln!(s, "    int base_a = {};", lead_term(&addr_a)).unwrap();
        writeln!(s, "    int base_b = {};", lead_term(&addr_b)).unwrap();
        writeln!(s, "    simdgroup_float8x8 acc[4][4];").unwrap();
        writeln!(s, "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.0f);").unwrap();
        writeln!(s, "    uint sr = (sg / 2u) * 32u;").unwrap();
        writeln!(s, "    uint sc = (sg % 2u) * 32u;").unwrap();
        // Direct fragments: when every tile is full and each operand is
        // contiguous along its row or its reduce axis, the fragments are
        // loaded straight from device memory (the cache does the sharing
        // the threadgroup stage would) — no staging, no barriers.
        let (cpa, cka) = (addr_a.coefs[dy], addr_a.coefs[kdim]);
        let (cpb, ckb) = (addr_b.coefs[dx], addr_b.coefs[kdim]);
        let direct = m % BLOCK == 0
            && n % BLOCK == 0
            && k_total % 8 == 0
            && (cpa == 1 || cka == 1)
            && (cpb == 1 || ckb == 1);
        if direct {
            let buf_a = buf_a;
            let buf_b = buf_b;
            // A fragment (rows sr+8i.., k kk..): row-major when k is the
            // unit stride, else stored transposed
            let (lda, ta) = if cka == 1 {
                (cpa, "false")
            } else {
                (cka, "true")
            };
            let (ldb, tb) = if cpb == 1 {
                (ckb, "false")
            } else {
                (cpb, "true")
            };
            writeln!(s, "    for (uint kk = 0u; kk < {k_total}u; kk += 8u) {{").unwrap();
            writeln!(
                s,
                "        simdgroup_float8x8 a[4]; simdgroup_float8x8 b[4];"
            )
            .unwrap();
            writeln!(s, "        for (uint i = 0; i < 4; i++) simdgroup_load(a[i], b{buf_a} + base_a + {cpa} * int(row0 + sr + 8u * i) + {cka} * int(kk), {lda}u, ulong2(0, 0), {ta});").unwrap();
            writeln!(s, "        for (uint j = 0; j < 4; j++) simdgroup_load(b[j], b{buf_b} + base_b + {ckb} * int(kk) + {cpb} * int(col0 + sc + 8u * j), {ldb}u, ulong2(0, 0), {tb});").unwrap();
            writeln!(s, "        for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);").unwrap();
            writeln!(s, "    }}").unwrap();
        } else {
            writeln!(
                s,
                "    for (uint k0 = 0u; k0 < {k_total}u; k0 += {CHUNK}u) {{"
            )
            .unwrap();
            // staging, walking the buffer in address order
            let stage = |s: &mut String,
                         name: &str,
                         addr: &Addr,
                         pdim: usize,
                         base: &str,
                         extent: usize,
                         buf: usize,
                         row_major: bool| {
                // row_major: stage[point*CHUNK + k]; else stage[k*BLOCK + point]
                let (cp, ck) = (addr.coefs[pdim], addr.coefs[kdim]);
                let k_inner = ck.unsigned_abs() <= cp.unsigned_abs();
                writeln!(
                    s,
                    "        for (uint idx = tid; idx < {}u; idx += {THREADS}u) {{",
                    BLOCK * CHUNK
                )
                .unwrap();
                if k_inner {
                    writeln!(
                        s,
                        "            uint kk = idx % {CHUNK}u; uint pt = idx / {CHUNK}u;"
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "            uint pt = idx % {BLOCK}u; uint kk = idx / {BLOCK}u;"
                    )
                    .unwrap();
                }
                writeln!(s, "            uint gp = {base} + pt; uint gk = k0 + kk;").unwrap();
                let slot = if row_major {
                    format!("pt * {LDA}u + kk")
                } else {
                    format!("kk * {LDB}u + pt")
                };
                writeln!(
                s,
                "            {name}[{slot}] = (gp < {extent}u && gk < {k_total}u) ? b{buf}[uint(base_{} + {cp} * int(gp) + {ck} * int(gk))] : 0.0f;",
                if row_major { "a" } else { "b" }
            )
            .unwrap();
                writeln!(s, "        }}").unwrap();
            };
            stage(&mut s, "As", &addr_a, dy, "row0", m, buf_a, true);
            stage(&mut s, "Bs", &addr_b, dx, "col0", n, buf_b, false);
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "        for (uint kk = 0u; kk < {CHUNK}u; kk += 8u) {{").unwrap();
            writeln!(
                s,
                "            simdgroup_float8x8 a[4]; simdgroup_float8x8 b[4];"
            )
            .unwrap();
            writeln!(s, "            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], As + (sr + 8u * i) * {LDA}u + kk, {LDA}u);").unwrap();
            writeln!(s, "            for (uint j = 0; j < 4; j++) simdgroup_load(b[j], Bs + kk * {LDB}u + sc + 8u * j, {LDB}u);").unwrap();
            writeln!(s, "            for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);").unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        }
        writeln!(s, "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], Cs + (sr + 8u * i) * {LDC}u + sc + 8u * j, {LDC}u);").unwrap();
        writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        // epilogue + write-out, one element per thread per step
        writeln!(
            s,
            "    for (uint idx = tid; idx < {}u; idx += {THREADS}u) {{",
            BLOCK * BLOCK
        )
        .unwrap();
        writeln!(s, "        uint p{dy} = row0 + idx / {BLOCK}u;").unwrap();
        writeln!(s, "        uint p{dx} = col0 + idx % {BLOCK}u;").unwrap();
        writeln!(s, "        if (p{dy} < {m}u && p{dx} < {n}u) {{").unwrap();
        writeln!(
            s,
            "            float red = Cs[(idx / {BLOCK}u) * {LDC}u + idx % {BLOCK}u];"
        )
        .unwrap();
        let vals: Vec<String> = if epilogue {
            let prank = c.points().len();
            let mut arg_vals = Vec::new();
            for a in &c.args {
                if a.src.n() == red_node {
                    arg_vals.push("red".to_string());
                    continue;
                }
                let addr = self.flat_addr(a, prank);
                let mut e = format!("{}", addr.offset);
                for (d, &coef) in addr.coefs.iter().enumerate() {
                    if coef != 0 {
                        e.push_str(&format!(" + {coef} * int(p{d})"));
                    }
                }
                arg_vals.push(format!("b{}[uint({e})]", storage(a.src.n())));
            }
            let mut body = String::new();
            let outs = emit_scalar(&mut body, &c.body, &c.body_outs, "e0_", &|v| {
                arg_vals[v as usize].clone()
            });
            for line in body.lines() {
                writeln!(s, "            {}", line.replacen("let ", "float ", 1)).unwrap();
            }
            outs
        } else {
            vec!["red".to_string()]
        };
        let mut flat = String::from("0u");
        for (d, &np) in pts.iter().enumerate() {
            flat = format!("({flat}) * {np}u + p{d}");
        }
        writeln!(s, "            uint g = {flat};").unwrap();
        for (slot, v) in vals.iter().enumerate() {
            writeln!(s, "            out[g * {w_out}u + {slot}u] = {v};").unwrap();
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();

        Some(WgslKernel {
            source: s,
            node: TensorId(root as u32),
            reads: reads.iter().map(|&j| TensorId(j as u32)).collect(),
            points: numel(c.points()) as u32,
            count: (n_tiles * THREADS) as u32,
            workgroup_size: THREADS as u32,
        })
    }
}
