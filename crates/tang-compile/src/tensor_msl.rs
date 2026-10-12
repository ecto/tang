//! Lower a [`TensorGraph`] to Metal Shading Language.
//!
//! The scalar, split and tiled schedules share their emitters with WGSL:
//! the index algebra, staging layouts and epilogue fusion are produced
//! once (under [`Dialect::Msl`], which switches buffer access to native
//! `half` / `float` elements) and the WGSL surface syntax is rewritten to
//! MSL here. MSL has no module-scope buffers, so every helper function
//! takes the kernel's buffers as a `Bufs` struct.
//!
//! On top of that, MSL gets a schedule WGSL cannot express: **fragment
//! lowering**. A contraction `C[m, n] = Σ_r A(m, r)·B(r, n)` whose
//! operands are stored in f16 and which accumulates in f32 runs on
//! `simdgroup_matrix` 8×8 fragments ([`TensorGraph::fragment_plan`]).
//! Operands are gathered through their views into threadgroup memory
//! (zero-padding guarded views and ragged edges at staging), so matmul,
//! batched matmul and conv-as-implicit-GEMM are one lowering. The result
//! tile goes through threadgroup memory, where an epilogue map is applied
//! before the write-out. The fragment geometry lives in
//! [`FragmentTarget`] so a CUDA `wmma` lowering can reuse the plan.

use std::fmt::Write;

use tang_expr::Node;

use super::wgsl::{
    arg_read, emit_decompose, emit_scalar_mapped, store, with_dialect, BufferSpec, Dialect,
};
use super::{fixed, tile_axes, Format, MonoidKind, Op, TensorGraph, TensorId};

/// Matrix-unit geometry of a target, for fragment lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FragmentTarget {
    /// Fragment shape `(m, n, k)` of one multiply-accumulate.
    pub frag: (usize, usize, usize),
    /// Threads per SIMD group / warp.
    pub simd_width: usize,
    /// Threadgroup (shared) memory per threadgroup, bytes.
    pub threadgroup_mem: usize,
    /// Output tile per threadgroup `(m, n)` and reduce chunk staged per step.
    pub tile: (usize, usize, usize),
    /// SIMD groups per threadgroup along `(m, n)`.
    pub groups: (usize, usize),
}

impl FragmentTarget {
    /// Apple GPUs (M1+): `simdgroup_half8x8` / `simdgroup_float8x8`, 32 KiB
    /// threadgroup memory. 64×64 tiles over a 2×2 grid of SIMD groups, each
    /// owning 32×32 = 4×4 fragments (the spike's shape: 11 TFLOP/s at 1024³).
    pub const APPLE: FragmentTarget = FragmentTarget {
        frag: (8, 8, 8),
        simd_width: 32,
        threadgroup_mem: 32 * 1024,
        tile: (64, 64, 32),
        groups: (2, 2),
    };

    pub fn threads(&self) -> usize {
        self.groups.0 * self.groups.1 * self.simd_width
    }

    /// Threadgroup bytes of one fragment kernel: f16 operand tiles plus the
    /// f32 result tile used by the epilogue.
    pub fn staging_bytes(&self) -> usize {
        let (bm, bn, bk) = self.tile;
        2 * (bm * bk + bk * bn) + 4 * bm * bn
    }
}

/// How a comp maps onto fragments: which point dims are the GEMM's `m`
/// and `n`, which arg is `A(m, r)` and which is `B(r, n)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentPlan {
    /// The reduction (the root itself, or the reduction under an epilogue map).
    pub core: usize,
    /// Arg index of `A` (independent of `n`) and `B` (independent of `m`).
    pub a: usize,
    pub b: usize,
    /// Point dims mapped to `m` and `n`; every other point dim is a batch dim.
    pub dm: usize,
    pub dn: usize,
}

/// One Metal dispatch.
pub struct MslKernel {
    pub source: String,
    /// Kernel function name in `source`.
    pub entry: String,
    /// Node this kernel materializes.
    pub node: TensorId,
    /// Buffer-backed nodes read, in `[[buffer(i)]]` order; the output is
    /// bound next, then a `Params { points, 0, 0, 0 }` constant.
    pub reads: Vec<TensorId>,
    pub points: u32,
    /// Threadgroups along x and y.
    pub groups: (u32, u32),
    pub threads: u32,
    /// Lowered onto matrix fragments.
    pub fragment: bool,
    /// Threadgroup memory the kernel declares, bytes. A schedule whose
    /// staging does not fit the device (32 KiB on Apple GPUs) is not a
    /// valid candidate there.
    pub threadgroup_bytes: usize,
}

pub struct MslProgram {
    pub kernels: Vec<MslKernel>,
    pub buffers: Vec<BufferSpec>,
    pub aliases: Vec<(TensorId, TensorId)>,
}

impl MslProgram {
    /// The buffer-owning node for `t` (itself unless `t` is a reshape).
    pub fn storage(&self, t: TensorId) -> TensorId {
        self.aliases
            .iter()
            .find(|(r, _)| *r == t)
            .map_or(t, |(_, s)| *s)
    }
}

fn grid(groups: usize) -> (u32, u32) {
    let x = groups.clamp(1, 65535);
    (x as u32, groups.div_ceil(65535).max(1) as u32)
}

impl TensorGraph {
    /// Lower to MSL with the Apple fragment geometry.
    pub fn to_msl(&self, outputs: &[TensorId]) -> MslProgram {
        self.to_msl_for(outputs, &FragmentTarget::APPLE)
    }

    /// Lower to MSL. Same kernels and buffers as [`to_wgsl`](Self::to_wgsl);
    /// nodes scheduled `fragment` (and eligible, see
    /// [`fragment_plan`](Self::fragment_plan)) run on matrix fragments.
    pub fn to_msl_for(&self, outputs: &[TensorId], target: &FragmentTarget) -> MslProgram {
        with_dialect(Dialect::Msl, || {
            let w = self.to_wgsl(outputs);
            let kernels = w
                .kernels
                .into_iter()
                .map(|k| {
                    let root = k.node.n();
                    let frag = self.nodes[root].fragment;
                    if let Some(plan) = frag.then(|| self.fragment_plan(k.node)).flatten() {
                        return self.fragment_kernel(root, &plan, target);
                    }
                    let reads: Vec<usize> = k.reads.iter().map(|t| t.n()).collect();
                    let groups = (k.count as usize).div_ceil(k.workgroup_size as usize);
                    let (source, threadgroup_bytes) = translate(self, &k.source, &reads, root);
                    MslKernel {
                        source,
                        threadgroup_bytes,
                        entry: "k_main".into(),
                        node: k.node,
                        reads: k.reads,
                        points: k.points,
                        groups: grid(groups),
                        threads: k.workgroup_size,
                        fragment: false,
                    }
                })
                .collect();
            MslProgram {
                kernels,
                buffers: w.buffers,
                aliases: w.aliases,
            }
        })
    }

    /// Schedule: lower `t` onto matrix fragments where a backend has them.
    pub fn set_fragment(&mut self, t: TensorId, on: bool) {
        self.nodes[t.n()].fragment = on;
    }

    /// Whether `t` can run on f16 matrix fragments with an f32
    /// accumulator: a materialized sum-of-products `Σ_r A·B` (or an
    /// elementwise map over one, as epilogue) with two wide point dims,
    /// `A` independent of one and `B` of the other, both operands read
    /// straight from f16 buffers, and an f32 accumulator.
    pub fn fragment_plan(&self, t: TensorId) -> Option<FragmentPlan> {
        let root = t.n();
        let rc = self.comp_at(root)?;
        let core = if rc.monoid.is_none() {
            self.epilogue_core(root)?
        } else {
            root
        };
        let c = self.comp_at(core)?;
        let m = c.monoid.as_ref()?;
        if m.kind() != MonoidKind::Sum || c.width() != 1 || c.tuple != 0 {
            return None;
        }
        if self.nodes[core].precision.accum != Format::F32 {
            return None;
        }
        let (u, v) = Self::product_operands(c)?;
        c.red.iter().try_for_each(|d| d.fixed().map(|_| ()))?;
        let pts: Vec<usize> = c
            .points()
            .iter()
            .map(|d| d.fixed())
            .collect::<Option<_>>()?;
        let (Some(dn), Some(dm)) = tile_axes(&pts) else {
            return None;
        };
        // operands: buffer-backed f16, or an inlined elementwise map
        // (materialized into threadgroup `half` while staging: the map runs
        // in f32 and its value is rounded to f16 there)
        for a in [&c.args[u], &c.args[v]] {
            if !(self.f16_buffer(a.src) || self.inline_map(a.src)) {
                return None;
            }
        }
        let dep = |i: usize, d: usize| c.args[i].view.depends_on(d);
        let (a, b) = if !dep(u, dn) && !dep(v, dm) {
            (u, v)
        } else if !dep(v, dn) && !dep(u, dm) {
            (v, u)
        } else {
            return None;
        };
        Some(FragmentPlan { core, a, b, dm, dn })
    }

    fn f16_buffer(&self, t: TensorId) -> bool {
        let s = self.storage(t).n();
        let backed = match &self.nodes[s].op {
            Op::Input(_) => true,
            Op::Comp(_) => !self.nodes[s].inline,
            _ => false,
        };
        backed && self.nodes[s].precision.storage == Format::F16
    }

    /// `t` is an inlined single-value map (no reduction).
    fn inline_map(&self, t: TensorId) -> bool {
        let n = &self.nodes[t.n()];
        n.inline
            && matches!(&n.op, Op::Comp(c) if c.monoid.is_none() && c.tuple == 0 && c.width() == 1)
    }

    /// Arg indices `(i, j)` of a body that is exactly `v_i · v_j`, `i ≠ j`.
    /// Other args may be present and unread (a gradient keeps the factor
    /// the product rule dropped as an argument).
    pub(crate) fn product_operands(c: &super::Comp) -> Option<(usize, usize)> {
        let nodes = c.body.nodes_slice();
        let (x, y) = match nodes[c.body_outs[0].index() as usize] {
            Node::Mul(x, y) => (x, y),
            _ => return None,
        };
        match (&nodes[x.index() as usize], &nodes[y.index() as usize]) {
            (Node::Var(i), Node::Var(j)) if i != j => Some((*i as usize, *j as usize)),
            _ => None,
        }
    }

    /// The contraction operands (storage buffers) `t` would read on
    /// fragments: the two factors of its core's `v_i · v_j` body.
    pub fn fragment_operands(&self, t: TensorId) -> Option<[TensorId; 2]> {
        let root = t.n();
        let rc = self.comp_at(root)?;
        let core = if rc.monoid.is_none() {
            self.epilogue_core(root)?
        } else {
            root
        };
        let c = self.comp_at(core)?;
        let (i, j) = Self::product_operands(c)?;
        Some([self.storage(c.args[i].src), self.storage(c.args[j].src)])
    }

    /// Whether every coordinate `arg` reads over the iteration box lies in
    /// its source: its guard never fires.
    fn view_in_bounds(&self, c: &super::Comp, which: usize) -> bool {
        let a = &c.args[which];
        let ext: Vec<i64> = c.space().iter().map(|d| d.n() as i64).collect();
        let shape = fixed(&self.nodes[a.src.n()].shape);
        a.view.forms.iter().zip(&shape).all(|(f, &n)| {
            let (mut lo, mut hi) = (f.offset, f.offset);
            for (k, e) in f.coefs.iter().zip(&ext) {
                let span = k * (e - 1);
                lo += span.min(0);
                hi += span.max(0);
            }
            lo >= 0 && hi < n as i64
        })
    }

    fn fragment_kernel(&self, root: usize, plan: &FragmentPlan, t: &FragmentTarget) -> MslKernel {
        let (pre, reads) = self.kernel_preamble(root);
        let c = self.comp_at(plan.core).unwrap();
        let pts = fixed(c.points());
        let red = fixed(&c.red);
        let kt: usize = red.iter().product();
        let (dm, dn) = (plan.dm, plan.dn);
        let (mm, nn) = (pts[dm], pts[dn]);
        let others: Vec<usize> = (0..pts.len()).filter(|&d| d != dm && d != dn).collect();
        let win = self.window_operand(c, plan, &pts, t);
        // Gather staging walks the operand's contiguous device axis: k for
        // A (and m when A is stored transposed), n for B (and k when B is
        // stored transposed, e.g. a weight gradient's `x[c, t + d·k]`).
        // Walking the threadgroup tile's slow axis pads its rows by one
        // word so those stores do not conflict.
        let unit = |which: usize, d: usize| {
            self.affine_address(c, which)
                .is_some_and(|(_, co)| co[d].abs() == 1 && co.last().copied() != Some(1))
        };
        let a_mfast = unit(plan.a, dm);
        let b_kfast = self
            .affine_address(c, plan.b)
            .is_some_and(|(_, co)| co.last().copied() == Some(1) && co[dn].abs() != 1);
        let a_ld = if a_mfast { t.tile.2 + 2 } else { t.tile.2 };
        let b_ld = if b_kfast { t.tile.1 + 2 } else { t.tile.1 };
        let (bm, bn, bk) = t.tile;
        let (fm, fn_, fk) = t.frag;
        let (gm, gn) = t.groups;
        let threads = t.threads();
        let (sm, sn) = (bm / gm, bn / gn); // per-SIMD-group sub-tile
        let (am, an) = (sm / fm, sn / fn_); // fragments per SIMD group
        assert!(win.is_some() || t.staging_bytes() <= t.threadgroup_mem);
        let (tx, ty) = (nn.div_ceil(bn), mm.div_ceil(bm));
        let n_tiles = tx * ty * others.iter().map(|&d| pts[d]).product::<usize>();

        let mut s = pre;
        writeln!(s, "// fragment lowering: m = p{dm}, n = p{dn}, K = {kt}").unwrap();
        writeln!(s, "kernel void k_main(KERNEL_PARAMS uint3 wid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {{").unwrap();
        match &win {
            Some(w) => {
                // Cs aliases the operand staging: it is written after the
                // reduce loop's last barrier
                writeln!(s, "    threadgroup half smem[{}];", w.bytes(t) / 2).unwrap();
                writeln!(s, "    threadgroup half* As = smem;").unwrap();
                writeln!(
                    s,
                    "    threadgroup half* Xs = smem + {};",
                    w.nbuf * w.a_len(bm)
                )
                .unwrap();
                writeln!(s, "    threadgroup float* Cs = (threadgroup float*)smem;").unwrap();
            }
            None => {
                writeln!(s, "    threadgroup half As[{}];", bm * a_ld).unwrap();
                writeln!(s, "    threadgroup half Bs[{}];", bk * b_ld).unwrap();
            }
        }
        if win.is_none() {
            writeln!(s, "    threadgroup float Cs[{}];", bm * bn).unwrap();
        }
        writeln!(s, "    BUFS_INIT").unwrap();
        writeln!(s, "    const uint tile = wid.x + wid.y * 65535u;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) {{ return; }}").unwrap();
        let nlead: usize = others.iter().map(|&d| pts[d]).product();
        if win.is_some() {
            // batch dims fastest: neighbouring threadgroups (e.g. the
            // phases of a polyphase conv) read neighbouring weights
            writeln!(s, "    const uint tile_x = (tile / {nlead}u) % {tx}u;").unwrap();
            writeln!(s, "    const uint tile_y = tile / {}u;", nlead * tx).unwrap();
            writeln!(s, "    uint lead = tile % {nlead}u;").unwrap();
        } else {
            writeln!(s, "    const uint tile_x = tile % {tx}u;").unwrap();
            writeln!(s, "    const uint tile_y = (tile / {tx}u) % {ty}u;").unwrap();
            writeln!(s, "    uint lead = tile / {}u;", tx * ty).unwrap();
        }
        for &d in others.iter().rev() {
            writeln!(s, "    const uint p{d} = lead % {}u;", pts[d]).unwrap();
            writeln!(s, "    lead = lead / {}u;", pts[d]).unwrap();
        }
        writeln!(s, "    const uint m0 = tile_y * {bm}u;").unwrap();
        writeln!(s, "    const uint n0 = tile_x * {bn}u;").unwrap();
        writeln!(s, "    const uint sr = (sg / {gn}u) * {sm}u;").unwrap();
        writeln!(s, "    const uint sc = (sg % {gn}u) * {sn}u;").unwrap();
        writeln!(s, "    simdgroup_float8x8 acc[{am}][{an}];").unwrap();
        writeln!(s, "    for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);").unwrap();
        // Direct path: an operand whose address is affine with unit stride
        // along k or along its free axis is loaded straight from device
        // memory into fragments (no staging) on interior tiles.
        let direct = |which: usize, free: usize| -> Option<(usize, i64, i64, String)> {
            let a = &c.args[which];
            if !kt.is_multiple_of(fk) {
                return None;
            }
            let src = self.storage(a.src).n();
            let coef = self.affine_address(c, which)?;
            let k0 = pts.len();
            let sk = *coef.1.last().unwrap_or(&0);
            // the reduce dims must flatten row-major onto one stride
            let mut want = sk;
            for i in (0..red.len()).rev() {
                if coef.1[k0 + i] != want {
                    return None;
                }
                want *= red[i] as i64;
            }
            let sf = coef.1[free];
            if sk <= 0 || sf < 0 || !(sk == 1 || sf == 1) {
                return None;
            }
            let mut base = format!("{}", coef.0);
            for &d in &others {
                if coef.1[d] != 0 {
                    base.push_str(&format!(" + {} * int(p{d})", coef.1[d]));
                }
            }
            Some((src, sk, sf, base))
        };
        // a guarded operand loads directly only on chunks whose reads are
        // all in range (checked per chunk below)
        let da = direct(plan.a, dm);
        let db = direct(plan.b, dn);
        let (ga, gb) = (c.args[plan.a].view.guard, c.args[plan.b].view.guard);
        // fragment loads of a direct operand at reduce index `kexpr`
        let load_a = |k: &str| {
            let (ja, ska, sma, _) = da.as_ref().unwrap();
            let _ = ja;
            if *ska == 1 {
                format!("simdgroup_load(fa[i], pa + (sr + i * {fm}u) * {sma} + {k}, {sma})")
            } else {
                format!("simdgroup_load(fa[i], pa + (sr + i * {fm}u) + ({k}) * {ska}, {ska}, ulong2(0, 0), true)")
            }
        };
        let load_b = |k: &str| {
            let (_, skb, snb, _) = db.as_ref().unwrap();
            if *snb == 1 {
                format!("simdgroup_load(fb[j], pb + ({k}) * {skb} + sc + j * {fn_}u, {skb})")
            } else {
                format!("simdgroup_load(fb[j], pb + {k} + (sc + j * {fn_}u) * {snb}, {snb}, ulong2(0, 0), true)")
            }
        };
        // staging: A as [m][k], B as [k][n]; consecutive threads walk the
        // operand's fastest-moving tile axis
        let stage =
            |s: &mut String, which: usize, buf: &str, rows: usize, cols: usize, a_side: bool| {
                writeln!(
                    s,
                    "        for (uint idx = tid; idx < {}u; idx += {threads}u) {{",
                    rows * cols
                )
                .unwrap();
                let (row_fast, ld) = if a_side {
                    (a_mfast, a_ld)
                } else {
                    (b_kfast, b_ld)
                };
                if row_fast {
                    writeln!(s, "            const uint ri = idx % {rows}u;").unwrap();
                    writeln!(s, "            const uint ci = idx / {rows}u;").unwrap();
                } else {
                    writeln!(s, "            const uint ri = idx / {cols}u;").unwrap();
                    writeln!(s, "            const uint ci = idx % {cols}u;").unwrap();
                }
                let (gpt, gk, ext) = if a_side {
                    ("m0 + ri", "k0 + ci", mm)
                } else {
                    ("n0 + ci", "k0 + ri", nn)
                };
                writeln!(s, "            const uint gp = {gpt};").unwrap();
                writeln!(s, "            const uint gk = {gk};").unwrap();
                writeln!(s, "            float v = 0.0f;").unwrap();
                writeln!(s, "            if (gp < {ext}u && gk < {kt}u) {{").unwrap();
                let (dp, dz) = if a_side { (dm, dn) } else { (dn, dm) };
                writeln!(s, "                const uint p{dp} = gp;").unwrap();
                writeln!(s, "                const uint p{dz} = 0u;").unwrap();
                let mut dec = String::new();
                emit_decompose(&mut dec, "gk", &red, "r");
                for line in dec.lines() {
                    writeln!(s, "            {line}").unwrap();
                }
                writeln!(
                    s,
                    "                v = {};",
                    arg_read(self, c, &c.args[which])
                )
                .unwrap();
                writeln!(s, "            }}").unwrap();
                writeln!(s, "            {buf}[ri * {ld}u + ci] = half(v);").unwrap();
                writeln!(s, "        }}").unwrap();
            };
        // One reduce loop, each operand either loaded straight from device
        // (`a_dev` / `b_dev`) or gathered into threadgroup memory.
        let emit_loop = |s: &mut String, a_dev: bool, b_dev: bool| {
            if a_dev {
                let (ja, _, sma, basea) = da.as_ref().unwrap();
                writeln!(
                    s,
                    "    device const half* pa = b{ja} + ({basea}) + int(m0) * {sma};"
                )
                .unwrap();
            }
            if b_dev {
                let (jb, _, snb, baseb) = db.as_ref().unwrap();
                writeln!(
                    s,
                    "    device const half* pb = b{jb} + ({baseb}) + int(n0) * {snb};"
                )
                .unwrap();
            }
            if a_dev && b_dev {
                writeln!(s, "    for (uint k = 0u; k < {kt}u; k += {fk}u) {{").unwrap();
                writeln!(s, "        simdgroup_half8x8 fa[{am}];").unwrap();
                writeln!(s, "        simdgroup_half8x8 fb[{an}];").unwrap();
                writeln!(
                    s,
                    "        for (uint i = 0u; i < {am}u; i++) {};",
                    load_a("k")
                )
                .unwrap();
                writeln!(
                    s,
                    "        for (uint j = 0u; j < {an}u; j++) {};",
                    load_b("k")
                )
                .unwrap();
                writeln!(s, "        for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);").unwrap();
                writeln!(s, "    }}").unwrap();
                return;
            }
            writeln!(s, "    for (uint k0 = 0u; k0 < {kt}u; k0 += {bk}u) {{").unwrap();
            if !a_dev {
                stage(s, plan.a, "As", bm, bk, true);
            }
            if !b_dev {
                stage(s, plan.b, "Bs", bk, bn, false);
            }
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "        for (uint kk = 0u; kk < {bk}u; kk += {fk}u) {{").unwrap();
            writeln!(s, "            simdgroup_half8x8 fa[{am}];").unwrap();
            writeln!(s, "            simdgroup_half8x8 fb[{an}];").unwrap();
            if a_dev {
                writeln!(
                    s,
                    "            for (uint i = 0u; i < {am}u; i++) {};",
                    load_a("k0 + kk")
                )
                .unwrap();
            } else {
                writeln!(s, "            for (uint i = 0u; i < {am}u; i++) simdgroup_load(fa[i], &As[(sr + i * {fm}u) * {a_ld}u + kk], {a_ld});").unwrap();
            }
            if b_dev {
                writeln!(
                    s,
                    "            for (uint j = 0u; j < {an}u; j++) {};",
                    load_b("k0 + kk")
                )
                .unwrap();
            } else {
                writeln!(s, "            for (uint j = 0u; j < {an}u; j++) simdgroup_load(fb[j], &Bs[kk * {b_ld}u + sc + j * {fn_}u], {b_ld});").unwrap();
            }
            writeln!(s, "            for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);").unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        };
        // A direct operand alongside a staged one reads whole chunks, so
        // the reduce extent must be a multiple of the chunk.
        let hybrid = kt.is_multiple_of(bk);
        let m_int = format!("m0 + {bm}u <= {mm}u");
        let n_int = format!("n0 + {bn}u <= {nn}u");
        let fast = match (da.is_some() && !ga, db.is_some() && !gb) {
            (true, true) => Some((format!("{m_int} && {n_int}"), true, true)),
            (true, false) if hybrid => Some((m_int.clone(), true, false)),
            (false, true) if hybrid => Some((n_int.clone(), false, true)),
            _ => None,
        };
        // per-chunk direct: both operands affine, one or both guarded, one
        // reduce dim (so a chunk is one interval of it)
        let chunked = win.is_none() && red.len() == 1 && da.is_some() && db.is_some() && (ga || gb);
        match (&win, fast) {
            (Some(w), _) => self.emit_window_loop(&mut s, c, plan, &pts, w, t),
            _ if chunked => {
                let k0 = pts.len();
                // in-range test of a guarded operand over one chunk
                let inb = |which: usize, free: usize, origin: &str, ext: usize| -> Vec<String> {
                    let a = &c.args[which];
                    if !a.view.guard {
                        return Vec::new();
                    }
                    let shape = fixed(&self.nodes[a.src.n()].shape);
                    a.view
                        .forms
                        .iter()
                        .zip(&shape)
                        .map(|(f, &n)| {
                            let mut e = format!("{}", f.offset);
                            for &d in &others {
                                if f.coefs[d] != 0 {
                                    e.push_str(&format!(" + {} * int(p{d})", f.coefs[d]));
                                }
                            }
                            let (cf, ck) = (f.coefs[free], f.coefs[k0]);
                            e.push_str(&format!(" + {cf} * int({origin}) + {ck} * int(k0)"));
                            let lo = cf.min(0) * (ext as i64 - 1) + ck.min(0) * (bk as i64 - 1);
                            let hi = cf.max(0) * (ext as i64 - 1) + ck.max(0) * (bk as i64 - 1);
                            format!("({e}) + {lo} >= 0 && ({e}) + {hi} < {n}")
                        })
                        .collect()
                };
                let mut conds = vec![
                    format!("{m_int} && {n_int}"),
                    format!("k0 + {bk}u <= {kt}u"),
                ];
                conds.extend(inb(plan.a, dm, "m0", bm));
                conds.extend(inb(plan.b, dn, "n0", bn));
                let (ja, _, sma, basea) = da.as_ref().unwrap();
                let (jb, _, snb, baseb) = db.as_ref().unwrap();
                writeln!(
                    s,
                    "    device const half* pa = b{ja} + ({basea}) + int(m0) * {sma};"
                )
                .unwrap();
                writeln!(
                    s,
                    "    device const half* pb = b{jb} + ({baseb}) + int(n0) * {snb};"
                )
                .unwrap();
                writeln!(s, "    for (uint k0 = 0u; k0 < {kt}u; k0 += {bk}u) {{").unwrap();
                writeln!(s, "        if ({}) {{", conds.join(" && ")).unwrap();
                writeln!(s, "        for (uint kk = 0u; kk < {bk}u; kk += {fk}u) {{").unwrap();
                writeln!(s, "            simdgroup_half8x8 fa[{am}];").unwrap();
                writeln!(s, "            simdgroup_half8x8 fb[{an}];").unwrap();
                writeln!(
                    s,
                    "            for (uint i = 0u; i < {am}u; i++) {};",
                    load_a("k0 + kk")
                )
                .unwrap();
                writeln!(
                    s,
                    "            for (uint j = 0u; j < {an}u; j++) {};",
                    load_b("k0 + kk")
                )
                .unwrap();
                writeln!(s, "            for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);").unwrap();
                writeln!(s, "        }}").unwrap();
                writeln!(s, "        continue;").unwrap();
                writeln!(s, "        }}").unwrap();
                stage(&mut s, plan.a, "As", bm, bk, true);
                stage(&mut s, plan.b, "Bs", bk, bn, false);
                writeln!(
                    s,
                    "        threadgroup_barrier(mem_flags::mem_threadgroup);"
                )
                .unwrap();
                writeln!(s, "        for (uint kk = 0u; kk < {bk}u; kk += {fk}u) {{").unwrap();
                writeln!(s, "            simdgroup_half8x8 fa[{am}];").unwrap();
                writeln!(s, "            simdgroup_half8x8 fb[{an}];").unwrap();
                writeln!(s, "            for (uint i = 0u; i < {am}u; i++) simdgroup_load(fa[i], &As[(sr + i * {fm}u) * {a_ld}u + kk], {a_ld});").unwrap();
                writeln!(s, "            for (uint j = 0u; j < {an}u; j++) simdgroup_load(fb[j], &Bs[kk * {b_ld}u + sc + j * {fn_}u], {b_ld});").unwrap();
                writeln!(s, "            for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);").unwrap();
                writeln!(s, "        }}").unwrap();
                writeln!(
                    s,
                    "        threadgroup_barrier(mem_flags::mem_threadgroup);"
                )
                .unwrap();
                writeln!(s, "    }}").unwrap();
            }
            (None, Some((cond, a_dev, b_dev))) => {
                writeln!(s, "    if ({cond}) {{").unwrap();
                emit_loop(&mut s, a_dev, b_dev);
                writeln!(s, "    }} else {{").unwrap();
                emit_loop(&mut s, false, false);
                writeln!(s, "    }}").unwrap();
            }
            (None, None) => emit_loop(&mut s, false, false),
        }
        writeln!(s, "    for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_store(acc[i][j], &Cs[(sr + i * {fm}u) * {bn}u + sc + j * {fn_}u], {bn});").unwrap();
        writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        // write-out (and epilogue) from the result tile
        writeln!(
            s,
            "    for (uint idx = tid; idx < {}u; idx += {threads}u) {{",
            bm * bn
        )
        .unwrap();
        writeln!(s, "        const uint gm = m0 + idx / {bn}u;").unwrap();
        writeln!(s, "        const uint gn = n0 + idx % {bn}u;").unwrap();
        writeln!(s, "        if (gm < {mm}u && gn < {nn}u) {{").unwrap();
        writeln!(s, "        const uint p{dm} = gm;").unwrap();
        writeln!(s, "        const uint p{dn} = gn;").unwrap();
        writeln!(s, "        const float cval = Cs[idx];").unwrap();
        let mut point_flat = String::from("0u");
        for (d, n) in pts.iter().enumerate() {
            point_flat = format!("({point_flat}) * {n}u + p{d}");
        }
        writeln!(s, "        const uint g = {point_flat};").unwrap();
        let out_f = self.nodes[root].precision.storage;
        if root == plan.core {
            writeln!(s, "        {}", store(out_f, "g", "cval")).unwrap();
        } else {
            let mc = self.comp_at(root).unwrap();
            let args: Vec<String> = mc
                .args
                .iter()
                .map(|a| {
                    if a.src.n() == plan.core {
                        "cval".to_string()
                    } else {
                        arg_read(self, mc, a)
                    }
                })
                .collect();
            let vals =
                emit_scalar_mapped(&mut s, "        ", &mc.body, &mc.body_outs, "e_", &|v| {
                    args[v as usize].clone()
                });
            let mw = mc.width();
            for (slot, v) in vals.iter().enumerate() {
                writeln!(
                    s,
                    "        {}",
                    store(out_f, &format!("g * {mw}u + {slot}u"), v)
                )
                .unwrap();
            }
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();

        MslKernel {
            source: translate(self, &s, &reads, root).0,
            threadgroup_bytes: win
                .as_ref()
                .map_or(2 * (bm * a_ld + bk * b_ld) + 4 * bm * bn, |w| w.bytes(t)),
            entry: "k_main".into(),
            node: TensorId(root as u32),
            reads: reads.iter().map(|&j| TensorId(j as u32)).collect(),
            points: pts.iter().product::<usize>() as u32,
            groups: grid(n_tiles),
            threads: threads as u32,
            fragment: true,
        }
    }
}

/// A sliding-window operand of a fragment contraction (conv-shaped `B`):
/// its view reads one "row" reduce dim (channels) through forms of its
/// own, and the free dim `n` plus every other reduce dim ("taps") through
/// one "time" form `st·n + Σ coef·tap + …`. One zero-padded window per
/// (row chunk, n tile) is staged in threadgroup memory, de-interleaved by
/// phase `u mod st`, and each tap's `B` fragments load from it at a
/// shifted offset: the taps reuse one staged window instead of each
/// re-gathering its own copy (implicit-GEMM gather).
///
/// `A` is staged once per chunk for every tap, as tap planes
/// `[tap][m][c]`; with an affine, unguarded `A` (a weight) each thread
/// copies all taps of one `(m, c)` with constant offsets. Staging is
/// double-buffered when two copies fit: one barrier per chunk.
#[derive(Clone, Debug)]
pub(crate) struct Window {
    src: usize,
    shape: Vec<usize>,
    forms: Vec<super::Form>,
    tform: usize,
    row: usize,
    taps: Vec<usize>,
    st: i64,
    minoff: i64,
    span: i64,
    /// Row-dim chunk staged per step.
    pub cb: usize,
    /// Tap combinations (product of tap extents).
    ntaps: usize,
    /// Staging buffers (2: double-buffered).
    nbuf: usize,
}

impl Window {
    fn a_len(&self, bm: usize) -> usize {
        bm * self.cb * self.ntaps
    }
    fn wp(&self, bn: usize) -> usize {
        bn + (self.span / self.st) as usize + 1
    }
    fn row_stride(&self, bn: usize) -> usize {
        self.st as usize * self.wp(bn)
    }
    fn x_len(&self, bn: usize) -> usize {
        self.cb * self.row_stride(bn)
    }
    /// Threadgroup bytes: the staging buffers, which the f32 result tile
    /// aliases after the reduce loop.
    fn bytes(&self, t: &FragmentTarget) -> usize {
        let (bm, bn, _) = t.tile;
        (2 * self.nbuf * (self.a_len(bm) + self.x_len(bn))).max(4 * bm * bn)
    }
}

impl TensorGraph {
    /// The windowed reading of `B`, if its view is conv-shaped (see
    /// [`Window`]) and its staging fits the target. `TANG_MSL_NO_WINDOW`
    /// disables it (A/B against the implicit-GEMM gather).
    pub(crate) fn window_operand(
        &self,
        c: &super::Comp,
        plan: &FragmentPlan,
        pts: &[usize],
        t: &FragmentTarget,
    ) -> Option<Window> {
        if std::env::var_os("TANG_MSL_NO_WINDOW").is_some() {
            return None;
        }
        let arg = &c.args[plan.b];
        if !self.f16_buffer(arg.src) {
            return None;
        }
        let k0 = pts.len();
        let red = fixed(&c.red);
        let (dm, dn) = (plan.dm, plan.dn);
        let forms = &arg.view.forms;
        let tform = forms.iter().position(|f| f.coefs[dn] != 0)?;
        let tf = &forms[tform];
        let st = tf.coefs[dn];
        if st <= 0 || tf.coefs[dm] != 0 {
            return None;
        }
        let mut row = None;
        let mut taps = Vec::new();
        for r in 0..red.len() {
            let in_t = tf.coefs[k0 + r] != 0;
            let elsewhere = forms
                .iter()
                .enumerate()
                .any(|(i, f)| i != tform && f.coefs[k0 + r] != 0);
            match (in_t, elsewhere) {
                (true, true) => return None,
                (false, true) => {
                    if row.replace(r).is_some() {
                        return None;
                    }
                }
                _ => taps.push(r),
            }
        }
        let row = row?;
        if !taps.iter().any(|&r| tf.coefs[k0 + r] != 0) {
            return None;
        }
        for (i, f) in forms.iter().enumerate() {
            if i != tform && (f.coefs[dn] != 0 || f.coefs[dm] != 0) {
                return None;
            }
        }
        let (mut minoff, mut maxoff) = (0i64, 0i64);
        for &r in &taps {
            let e = tf.coefs[k0 + r] * (red[r] as i64 - 1);
            minoff += e.min(0);
            maxoff += e.max(0);
        }
        let mut w = Window {
            src: self.storage(arg.src).n(),
            shape: fixed(&self.nodes[arg.src.n()].shape),
            forms: forms.clone(),
            tform,
            row,
            ntaps: taps.iter().map(|&r| red[r]).product(),
            taps,
            st,
            minoff,
            span: maxoff - minoff,
            cb: 0,
            nbuf: 0,
        };
        // 16- and 8-channel chunks measured equal on op (a) (0.69 / 0.70
        // ms); double buffering (one barrier per chunk) measured slower
        // (0.74), so it is opt-in: `TANG_MSL_WINDOW_BUFFERS=2`
        let rows = red[row].next_multiple_of(8);
        let nbuf: usize = std::env::var("TANG_MSL_WINDOW_BUFFERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
            .clamp(1, 2);
        [16usize, 8].into_iter().find_map(|cb| {
            w.cb = cb.min(rows);
            w.nbuf = nbuf;
            (w.bytes(t) <= t.threadgroup_mem).then(|| w.clone())
        })
    }

    fn emit_window_loop(
        &self,
        s: &mut String,
        c: &super::Comp,
        plan: &FragmentPlan,
        pts: &[usize],
        w: &Window,
        t: &FragmentTarget,
    ) {
        let red = fixed(&c.red);
        let k0 = pts.len();
        let (dm, dn) = (plan.dm, plan.dn);
        let mm = pts[dm];
        let (bm, bn, _) = t.tile;
        let (fm, fn_, _) = t.frag;
        let threads = t.threads();
        let (am, an) = (bm / t.groups.0 / fm, bn / t.groups.1 / fn_);
        let (cb, st, ntaps) = (w.cb, w.st, w.ntaps);
        let wp = w.wp(bn);
        let rs = w.row_stride(bn);
        let (al, xl) = (w.a_len(bm), w.x_len(bn));
        let uw = st as usize * (bn - 1) + w.span as usize + 1;
        let crow = red[w.row];
        let others: Vec<usize> = (0..pts.len()).filter(|&d| d != dm && d != dn).collect();
        let tf = &w.forms[w.tform];
        // tap ti's values of the tap dims
        let tap_vals = |ti: usize| {
            let mut rem = ti;
            let mut vals = vec![0i64; red.len()];
            for &r in w.taps.iter().rev() {
                vals[r] = (rem % red[r]) as i64;
                rem /= red[r];
            }
            vals
        };

        // ---- staging of chunk `c0` into `Ad` / `Xd`
        let mut stage = String::new();
        let st_ = &mut stage;
        // the window
        writeln!(
            st_,
            "        for (uint idx = tid; idx < {}u; idx += {threads}u) {{",
            cb * uw
        )
        .unwrap();
        writeln!(st_, "            const uint cl = idx / {uw}u;").unwrap();
        writeln!(st_, "            const uint u = idx % {uw}u;").unwrap();
        writeln!(st_, "            const uint gc = c0 + cl;").unwrap();
        writeln!(st_, "            half v = half(0.0f);").unwrap();
        writeln!(st_, "            if (gc < {crow}u) {{").unwrap();
        let mut conds = Vec::new();
        for (i, f) in w.forms.iter().enumerate() {
            let mut e = format!("{}", f.offset);
            for &d in &others {
                if f.coefs[d] != 0 {
                    e.push_str(&format!(" + {} * int(p{d})", f.coefs[d]));
                }
            }
            if i == w.tform {
                e.push_str(&format!(" + {st} * int(n0) + {} + int(u)", w.minoff));
            } else if f.coefs[k0 + w.row] != 0 {
                e.push_str(&format!(" + {} * int(gc)", f.coefs[k0 + w.row]));
            }
            writeln!(st_, "                const int x{i} = {e};").unwrap();
            conds.push(format!("x{i} >= 0 && x{i} < {}", w.shape[i]));
        }
        let mut terms = Vec::new();
        let mut stride = 1i64;
        for i in (0..w.forms.len()).rev() {
            terms.push(format!("x{i} * {stride}"));
            stride *= w.shape[i] as i64;
        }
        let addr = if terms.is_empty() {
            "0".into()
        } else {
            terms.join(" + ")
        };
        // every coordinate is bounds-checked, guarded view or not: the
        // window overhangs the tile
        writeln!(
            st_,
            "                if ({}) {{ v = b{}[{addr}]; }}",
            conds.join(" && "),
            w.src
        )
        .unwrap();
        writeln!(st_, "            }}").unwrap();
        writeln!(
            st_,
            "            Xd[cl * {rs}u + (u % {st}u) * {wp}u + u / {st}u] = v;"
        )
        .unwrap();
        writeln!(st_, "        }}").unwrap();
        // A, every tap, as planes Ad[tap][m][c]
        let affine = (!c.args[plan.a].view.guard || self.view_in_bounds(c, plan.a))
            .then(|| self.affine_address(c, plan.a))
            .flatten()
            .filter(|(_, co)| co[dn] == 0);
        writeln!(
            st_,
            "        for (uint idx = tid; idx < {}u; idx += {threads}u) {{",
            bm * cb
        )
        .unwrap();
        writeln!(st_, "            const uint mi = idx / {cb}u;").unwrap();
        writeln!(st_, "            const uint cl = idx % {cb}u;").unwrap();
        writeln!(
            st_,
            "            const bool ok = m0 + mi < {mm}u && c0 + cl < {crow}u;"
        )
        .unwrap();
        match affine {
            Some((off, co)) => {
                // one thread per (m, c), every tap at a constant offset;
                // consecutive threads walk c, so the plane stores never
                // conflict
                let ja = self.storage(c.args[plan.a].src).n();
                let mut base = format!("{off}");
                for &d in &others {
                    if co[d] != 0 {
                        base.push_str(&format!(" + {} * int(p{d})", co[d]));
                    }
                }
                writeln!(
                    st_,
                    "            device const half* q = b{ja} + ({base}) + int(m0 + mi) * {} + int(c0 + cl) * {};",
                    co[dm],
                    co[k0 + w.row]
                )
                .unwrap();
                for ti in 0..ntaps {
                    let vals = tap_vals(ti);
                    let o: i64 = w.taps.iter().map(|&r| co[k0 + r] * vals[r]).sum();
                    writeln!(
                        st_,
                        "            Ad[{}u + idx] = ok ? q[{o}] : half(0.0f);",
                        ti * bm * cb
                    )
                    .unwrap();
                }
            }
            None => {
                writeln!(st_, "            if (ok) {{").unwrap();
                writeln!(st_, "                const uint p{dm} = m0 + mi;").unwrap();
                writeln!(st_, "                const uint p{dn} = 0u;").unwrap();
                writeln!(st_, "                const uint r{} = c0 + cl;", w.row).unwrap();
                for ti in 0..ntaps {
                    let vals = tap_vals(ti);
                    writeln!(st_, "                {{").unwrap();
                    for &r in &w.taps {
                        writeln!(st_, "                const uint r{r} = {}u;", vals[r]).unwrap();
                    }
                    writeln!(
                        st_,
                        "                Ad[{}u + idx] = half({});",
                        ti * bm * cb,
                        arg_read(self, c, &c.args[plan.a])
                    )
                    .unwrap();
                    writeln!(st_, "                }}").unwrap();
                }
                writeln!(st_, "            }} else {{").unwrap();
                for ti in 0..ntaps {
                    writeln!(
                        st_,
                        "                Ad[{}u + idx] = half(0.0f);",
                        ti * bm * cb
                    )
                    .unwrap();
                }
                writeln!(st_, "            }}").unwrap();
            }
        }
        writeln!(st_, "        }}").unwrap();

        // ---- fragments of the staged chunk in `Ac` / `Xc`
        let mut comp = String::new();
        let cm = &mut comp;
        writeln!(cm, "        for (uint kk = 0u; kk < {cb}u; kk += 8u) {{").unwrap();
        writeln!(cm, "            simdgroup_half8x8 fa[{am}];").unwrap();
        writeln!(cm, "            simdgroup_half8x8 fb[{an}];").unwrap();
        for ti in 0..ntaps {
            let vals = tap_vals(ti);
            let off: i64 = w
                .taps
                .iter()
                .map(|&r| tf.coefs[k0 + r] * vals[r])
                .sum::<i64>()
                - w.minoff;
            let boff = (off % st) as usize * wp + (off / st) as usize;
            writeln!(cm, "            for (uint i = 0u; i < {am}u; i++) simdgroup_load(fa[i], &Ac[{}u + (sr + i * {fm}u) * {cb}u + kk], {cb});", ti * bm * cb).unwrap();
            writeln!(cm, "            for (uint j = 0u; j < {an}u; j++) simdgroup_load(fb[j], &Xc[kk * {rs}u + {boff}u + sc + j * {fn_}u], {rs});").unwrap();
            writeln!(cm, "            for (uint i = 0u; i < {am}u; i++) for (uint j = 0u; j < {an}u; j++) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);").unwrap();
        }
        writeln!(cm, "        }}").unwrap();

        writeln!(s, "    // windowed operand: rows r{}, taps {:?}, stride {st}, window {uw}, chunk {cb}, {} buffer(s)", w.row, w.taps, w.nbuf).unwrap();
        if w.nbuf == 2 {
            writeln!(s, "    {{").unwrap();
            writeln!(s, "        const uint c0 = 0u;").unwrap();
            writeln!(s, "        threadgroup half* Ad = As;").unwrap();
            writeln!(s, "        threadgroup half* Xd = Xs;").unwrap();
            s.push_str(&stage);
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
            writeln!(
                s,
                "    for (uint cc = 0u, buf = 0u; cc < {crow}u; cc += {cb}u, buf ^= 1u) {{"
            )
            .unwrap();
            writeln!(s, "        if (cc + {cb}u < {crow}u) {{").unwrap();
            writeln!(s, "        const uint c0 = cc + {cb}u;").unwrap();
            writeln!(s, "        threadgroup half* Ad = As + (buf ^ 1u) * {al}u;").unwrap();
            writeln!(s, "        threadgroup half* Xd = Xs + (buf ^ 1u) * {xl}u;").unwrap();
            s.push_str(&stage);
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        threadgroup const half* Ac = As + buf * {al}u;").unwrap();
            writeln!(s, "        threadgroup const half* Xc = Xs + buf * {xl}u;").unwrap();
            s.push_str(&comp);
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        } else {
            writeln!(s, "    for (uint c0 = 0u; c0 < {crow}u; c0 += {cb}u) {{").unwrap();
            writeln!(s, "        threadgroup half* Ad = As;").unwrap();
            writeln!(s, "        threadgroup half* Xd = Xs;").unwrap();
            s.push_str(&stage);
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "        threadgroup const half* Ac = As;").unwrap();
            writeln!(s, "        threadgroup const half* Xc = Xs;").unwrap();
            s.push_str(&comp);
            writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        }
    }
}

impl TensorGraph {
    /// Element address of arg `which` of `c` in its storage buffer as
    /// `(offset, coefficient per iteration dim)` — `None` unless the
    /// source is a plain buffer (input, reshape of one, or a materialized
    /// single-value comp).
    fn affine_address(&self, c: &super::Comp, which: usize) -> Option<(i64, Vec<i64>)> {
        let a = &c.args[which];
        let src = a.src.n();
        let storage = self.storage(a.src).n();
        match &self.nodes[storage].op {
            Op::Input(_) => {}
            Op::Comp(sc) if sc.tuple == 0 && !self.nodes[storage].inline => {}
            _ => return None,
        }
        // reshape keeps the flat index, so the reading node's own shape
        // gives the strides
        let shape = fixed(&self.nodes[src].shape);
        let mut stride = 1i64;
        let mut off = 0i64;
        let mut coefs = vec![0i64; c.rank()];
        for (f, &n) in a.view.forms.iter().zip(&shape).rev() {
            off += f.offset * stride;
            for (d, &k) in f.coefs.iter().enumerate() {
                coefs[d] += k * stride;
            }
            stride *= n as i64;
        }
        Some((off, coefs))
    }
}

// ---------------------------------------------------------------------------
// WGSL surface syntax → MSL
// ---------------------------------------------------------------------------

fn msl_ty(f: Format) -> &'static str {
    match f {
        Format::F32 => "float",
        Format::F16 => "half",
        Format::BF16 => "bfloat",
    }
}

const PRELUDE: &str = "#include <metal_stdlib>
using namespace metal;
typedef uint u32;
typedef int i32;
typedef float f32;
struct Params { uint count; uint _p1; uint _p2; uint _p3; };
inline void workgroupBarrier() { threadgroup_barrier(mem_flags::mem_threadgroup); }
";

/// Rewrite an emitter's WGSL-syntax kernel (or a fragment kernel, already
/// MSL in its `main`) to MSL.
fn translate(g: &TensorGraph, src: &str, reads: &[usize], root: usize) -> (String, usize) {
    let mut out = String::from(PRELUDE);
    // buffers: helper functions see them through `B`
    writeln!(out, "struct Bufs {{").unwrap();
    for &j in reads {
        let f = g.nodes[j].precision.storage;
        writeln!(out, "    device const {}* b{j};", msl_ty(f)).unwrap();
    }
    let out_f = g.nodes[root].precision.storage;
    writeln!(out, "    device {}* out;", msl_ty(out_f)).unwrap();
    writeln!(out, "}};").unwrap();
    let mut params = String::new();
    for (i, &j) in reads.iter().enumerate() {
        let f = g.nodes[j].precision.storage;
        write!(params, "device const {}* b{j} [[buffer({i})]], ", msl_ty(f)).unwrap();
    }
    write!(
        params,
        "device {}* out [[buffer({})]], constant Params& params [[buffer({})]], ",
        msl_ty(out_f),
        reads.len(),
        reads.len() + 1
    )
    .unwrap();
    let names: Vec<String> = reads
        .iter()
        .map(|j| format!("b{j}"))
        .chain(std::iter::once("out".to_string()))
        .collect();
    let bufs_init = format!("Bufs B{{{}}};", names.join(", "));

    let mut shared = Vec::new();
    let mut tg_bytes = 0;
    let mut body = String::new();
    for line in src.lines() {
        let t = line.trim_start();
        if t.starts_with("struct Params") || t.starts_with("@group") || t.starts_with("@compute") {
            continue;
        }
        if let Some(rest) = t.strip_prefix("var<workgroup> ") {
            // `name: array<T, N>;` → `threadgroup T name[N];`
            let (name, ty) = rest.trim_end_matches(';').split_once(": ").unwrap();
            let inner = ty
                .strip_prefix("array<")
                .unwrap()
                .strip_suffix('>')
                .unwrap();
            let (elem, n) = inner.rsplit_once(", ").unwrap();
            let width = match elem.strip_prefix("array<f32, ") {
                Some(w) => w.trim_end_matches('>').parse::<usize>().unwrap(),
                None => 1,
            };
            tg_bytes += 4 * width * n.parse::<usize>().unwrap();
            shared.push(format!("    threadgroup {elem} {name}[{n}];"));
            continue;
        }
        if t.starts_with("fn main(") {
            writeln!(
                body,
                "kernel void k_main({params}uint3 gid [[thread_position_in_grid]], uint3 wid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{"
            )
            .unwrap();
            for d in &shared {
                writeln!(body, "{d}").unwrap();
            }
            writeln!(body, "    {bufs_init}").unwrap();
            continue;
        }
        if t.starts_with("fn ") {
            writeln!(body, "{}", signature(t)).unwrap();
            continue;
        }
        if t.starts_with("//") {
            writeln!(body, "{line}").unwrap();
            continue;
        }
        let line = line
            .replace("KERNEL_PARAMS ", &params)
            .replace("BUFS_INIT", &bufs_init);
        writeln!(body, "{}", statement(&line)).unwrap();
    }
    out.push_str(&body);
    (out, tg_bytes)
}

/// `fn name(a: T, …) -> R {` → `R name(thread const Bufs& B, T a, …) {`.
fn signature(t: &str) -> String {
    let rest = &t[3..];
    let open = rest.find('(').unwrap();
    let name = &rest[..open];
    let close = matching(rest, open);
    let args = &rest[open + 1..close];
    let ret = rest[close + 1..]
        .trim()
        .trim_start_matches("->")
        .trim()
        .trim_end_matches('{')
        .trim();
    let mut ps = vec!["thread const Bufs& B".to_string()];
    for a in split_top(args) {
        let a = a.trim();
        if a.is_empty() {
            continue;
        }
        let (n, ty) = a.split_once(':').unwrap();
        ps.push(format!("{} {}", ty.trim(), n.trim()));
    }
    format!("{} {name}({}) {{", fix_floats(ret), ps.join(", "))
}

/// Index of the bracket closing the one at `open`.
fn matching(s: &str, open: usize) -> usize {
    let b = s.as_bytes();
    let (o, c) = (b[open], if b[open] == b'(' { b')' } else { b'>' });
    let mut depth = 0;
    for (i, &ch) in b.iter().enumerate().skip(open) {
        if ch == o {
            depth += 1;
        } else if ch == c {
            depth -= 1;
            if depth == 0 {
                return i;
            }
        }
    }
    panic!("unbalanced: {s}");
}

/// Split on commas outside any `()`/`<>` nesting.
fn split_top(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '<' | '[' => depth += 1,
            ')' | '>' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// One statement line: declarations, constructors, selects, buffer and
/// helper-call rewrites, float literal suffixes.
fn statement(line: &str) -> String {
    let s = decls(line);
    let s = selects(&s);
    let s = array_ctors(&s);
    let s = calls_and_buffers(&s);
    fix_floats(&s)
}

/// `let x = e` → `const auto x = e`, `let x: T = e` → `const T x = e`,
/// `var x = e` → `auto x = e`, `var x: T[= e]` → `T x[= e]`.
fn decls(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        let kw = if s[i..].starts_with("let ") {
            Some(true)
        } else if s[i..].starts_with("var ") {
            Some(false)
        } else {
            None
        };
        let boundary = i == 0 || !is_ident(b[i - 1]);
        if let (Some(is_let), true) = (kw, boundary) {
            let mut j = i + 4;
            let name_start = j;
            while j < b.len() && is_ident(b[j]) {
                j += 1;
            }
            let name = &s[name_start..j];
            let ty = if s[j..].starts_with(": ") {
                // type runs to ` =` or `;` at angle depth 0
                let ts = j + 2;
                let mut k = ts;
                let mut depth = 0;
                while k < b.len() {
                    match b[k] {
                        b'<' => depth += 1,
                        b'>' => depth -= 1,
                        b'=' | b';' if depth == 0 => break,
                        _ => {}
                    }
                    k += 1;
                }
                let t = s[ts..k].trim().to_string();
                j = k;
                Some(t)
            } else {
                None
            };
            let c = if is_let { "const " } else { "" };
            match ty {
                Some(t) => write!(out, "{c}{t} {name}").unwrap(),
                None => write!(out, "{c}auto {name}").unwrap(),
            }
            if !s[j..].starts_with(' ') && !s[j..].starts_with(';') {
                out.push(' ');
            }
            i = j;
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// `select(f, t, c)` → `((c) ? (t) : (f))`: lazy, so a guarded read that
/// would leave its buffer is never performed (MSL has no robust access).
fn selects(s: &str) -> String {
    let Some(pos) = find_word(s, "select(") else {
        return s.to_string();
    };
    let open = pos + "select".len();
    let close = matching(s, open);
    let parts = split_top(&s[open + 1..close]);
    assert_eq!(parts.len(), 3, "select arity: {s}");
    let (f, t, c) = (
        selects(parts[0].trim()),
        selects(parts[1].trim()),
        selects(parts[2].trim()),
    );
    format!(
        "{}(({c}) ? ({t}) : ({f})){}",
        &s[..pos],
        selects(&s[close + 1..])
    )
}

fn find_word(s: &str, w: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut from = 0;
    while let Some(p) = s[from..].find(w) {
        let p = from + p;
        if p == 0 || !is_ident(b[p - 1]) {
            return Some(p);
        }
        from = p + 1;
    }
    None
}

/// `array<f32, N>(a, b)` → `array<f32, N>{a, b}`.
fn array_ctors(s: &str) -> String {
    let mut out = s.to_string();
    let mut from = 0;
    while let Some(p) = out[from..].find("array<") {
        let p = from + p;
        let lt = p + "array".len();
        let gt = matching(&out, lt);
        if out.as_bytes().get(gt + 1) == Some(&b'(') {
            let close = matching(&out, gt + 1);
            out.replace_range(close..close + 1, "}");
            out.replace_range(gt + 1..gt + 2, "{");
        }
        from = gt;
    }
    out
}

/// Insert `B` into helper calls; route buffer reads and the output
/// through it.
fn calls_and_buffers(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 16);
    let mut i = 0;
    while i < b.len() {
        let boundary = i == 0 || !is_ident(b[i - 1]);
        if boundary {
            // helper call: (read|point|body|combine)_<n>(
            let mut matched = false;
            for p in ["read_", "point_", "body_", "combine_"] {
                if s[i..].starts_with(p) {
                    let mut j = i + p.len();
                    let ds = j;
                    while j < b.len() && b[j].is_ascii_digit() {
                        j += 1;
                    }
                    if j > ds && b.get(j) == Some(&b'(') {
                        out.push_str(&s[i..=j]);
                        let empty = b.get(j + 1) == Some(&b')');
                        out.push_str(if empty { "B" } else { "B, " });
                        i = j + 1;
                        matched = true;
                    }
                    break;
                }
            }
            if matched {
                continue;
            }
            // buffer: b<n>[ or out[
            if b[i] == b'b' {
                let mut j = i + 1;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
                if j > i + 1 && b.get(j) == Some(&b'[') {
                    out.push_str("B.");
                    out.push_str(&s[i..j]);
                    i = j;
                    continue;
                }
            }
            if s[i..].starts_with("out[") {
                out.push_str("B.out");
                i += 3;
                continue;
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// Give every float literal an `f` suffix (an unsuffixed literal is a
/// `double` in C++, which `auto` would then propagate).
fn fix_floats(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 8);
    let mut i = 0;
    while i < b.len() {
        let starts = b[i].is_ascii_digit() && (i == 0 || !(is_ident(b[i - 1]) || b[i - 1] == b'.'));
        if !starts {
            out.push(b[i] as char);
            i += 1;
            continue;
        }
        let st = i;
        let mut float = false;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i < b.len() && b[i] == b'.' {
            float = true;
            i += 1;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
            let mut j = i + 1;
            if j < b.len() && (b[j] == b'-' || b[j] == b'+') {
                j += 1;
            }
            if j < b.len() && b[j].is_ascii_digit() {
                float = true;
                i = j;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
        }
        out.push_str(&s[st..i]);
        if float && !(i < b.len() && (b[i] == b'f' || b[i] == b'F')) {
            out.push('f');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statement_rewrites() {
        assert_eq!(
            statement("    let x = select(0.0, read_3(u32(p0)), (u32(p0) < 4u));"),
            "    const auto x = (((u32(p0) < 4u)) ? (read_3(B, u32(p0))) : (0.0f));"
        );
        assert_eq!(
            statement("for (var r0: u32 = 0u; r0 < 4u; r0 = r0 + 1u) {"),
            "for (u32 r0 = 0u; r0 < 4u; r0 = r0 + 1u) {"
        );
        assert_eq!(
            statement("var acc = array<f32, 2>(1e-5, b2[i]);"),
            "auto acc = array<f32, 2>{1e-5f, B.b2[i]};"
        );
        assert_eq!(
            statement("let v = point_4();"),
            "const auto v = point_4(B);"
        );
        assert_eq!(
            signature("fn combine_3(a: array<f32, 2>, e: array<f32, 2>) -> array<f32, 2> {"),
            "array<f32, 2> combine_3(thread const Bufs& B, array<f32, 2> a, array<f32, 2> e) {"
        );
    }
}
