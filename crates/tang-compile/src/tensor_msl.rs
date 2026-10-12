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
            // fragments write the reduction's points: only an epilogue
            // that reads it point-for-point (identity view) fuses
            let r = self.tiled_reduction(root).filter(|&r| r != root)?;
            let k = rc.points().len();
            let read = self.epilogue_read(rc, r)?;
            if read != super::View::identity(k, k).forms || self.comp_at(r)?.out != rc.out {
                return None;
            }
            r
        } else {
            root
        };
        let c = self.comp_at(core)?;
        let m = c.monoid.as_ref()?;
        if m.kind() != MonoidKind::Sum || c.width() != 1 || c.tuple != 0 || c.args.len() != 2 {
            return None;
        }
        if self.nodes[core].precision.accum != Format::F32 {
            return None;
        }
        // body is exactly v0·v1
        let nodes = c.body.nodes_slice();
        let (x, y) = match nodes[c.body_outs[0].index() as usize] {
            Node::Mul(x, y) => (x, y),
            _ => return None,
        };
        let vars = (&nodes[x.index() as usize], &nodes[y.index() as usize]);
        if !matches!(
            vars,
            (Node::Var(0), Node::Var(1)) | (Node::Var(1), Node::Var(0))
        ) {
            return None;
        }
        c.red.iter().try_for_each(|d| d.fixed().map(|_| ()))?;
        let pts: Vec<usize> = c
            .points()
            .iter()
            .map(|d| d.fixed())
            .collect::<Option<_>>()?;
        let (Some(dn), Some(dm)) = tile_axes(&pts) else {
            return None;
        };
        // operands: buffer-backed f16
        for a in &c.args {
            let s = self.storage(a.src).n();
            let backed = match &self.nodes[s].op {
                Op::Input(_) => true,
                Op::Comp(_) => !self.nodes[s].inline,
                _ => false,
            };
            if !backed || self.nodes[s].precision.storage != Format::F16 {
                return None;
            }
        }
        let dep = |i: usize, d: usize| c.args[i].view.depends_on(d);
        let (a, b) = if !dep(0, dn) && !dep(1, dm) {
            (0, 1)
        } else if !dep(1, dn) && !dep(0, dm) {
            (1, 0)
        } else {
            return None;
        };
        Some(FragmentPlan { core, a, b, dm, dn })
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
        let (bm, bn, bk) = t.tile;
        let (fm, fn_, fk) = t.frag;
        let (gm, gn) = t.groups;
        let threads = t.threads();
        let (sm, sn) = (bm / gm, bn / gn); // per-SIMD-group sub-tile
        let (am, an) = (sm / fm, sn / fn_); // fragments per SIMD group
        assert!(t.staging_bytes() <= t.threadgroup_mem);
        let (tx, ty) = (nn.div_ceil(bn), mm.div_ceil(bm));
        let n_tiles = tx * ty * others.iter().map(|&d| pts[d]).product::<usize>();

        let mut s = pre;
        writeln!(s, "// fragment lowering: m = p{dm}, n = p{dn}, K = {kt}").unwrap();
        writeln!(s, "kernel void k_main(KERNEL_PARAMS uint3 wid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {{").unwrap();
        writeln!(s, "    threadgroup half As[{}];", bm * bk).unwrap();
        writeln!(s, "    threadgroup half Bs[{}];", bk * bn).unwrap();
        writeln!(s, "    threadgroup float Cs[{}];", bm * bn).unwrap();
        writeln!(s, "    BUFS_INIT").unwrap();
        writeln!(s, "    const uint tile = wid.x + wid.y * 65535u;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) {{ return; }}").unwrap();
        writeln!(s, "    const uint tile_x = tile % {tx}u;").unwrap();
        writeln!(s, "    const uint tile_y = (tile / {tx}u) % {ty}u;").unwrap();
        writeln!(s, "    uint lead = tile / {}u;", tx * ty).unwrap();
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
            if a.view.guard || !kt.is_multiple_of(fk) {
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
        let da = direct(plan.a, dm);
        let db = direct(plan.b, dn);
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
                writeln!(s, "            const uint ri = idx / {cols}u;").unwrap();
                writeln!(s, "            const uint ci = idx % {cols}u;").unwrap();
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
                writeln!(s, "            {buf}[idx] = half(v);").unwrap();
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
                writeln!(s, "            for (uint i = 0u; i < {am}u; i++) simdgroup_load(fa[i], &As[(sr + i * {fm}u) * {bk}u + kk], {bk});").unwrap();
            }
            if b_dev {
                writeln!(
                    s,
                    "            for (uint j = 0u; j < {an}u; j++) {};",
                    load_b("k0 + kk")
                )
                .unwrap();
            } else {
                writeln!(s, "            for (uint j = 0u; j < {an}u; j++) simdgroup_load(fb[j], &Bs[kk * {bn}u + sc + j * {fn_}u], {bn});").unwrap();
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
        let fast = match (da.is_some(), db.is_some()) {
            (true, true) => Some((format!("{m_int} && {n_int}"), true, true)),
            (true, false) if hybrid => Some((m_int, true, false)),
            (false, true) if hybrid => Some((n_int, false, true)),
            _ => None,
        };
        match fast {
            Some((cond, a_dev, b_dev)) => {
                writeln!(s, "    if ({cond}) {{").unwrap();
                emit_loop(&mut s, a_dev, b_dev);
                writeln!(s, "    }} else {{").unwrap();
                emit_loop(&mut s, false, false);
                writeln!(s, "    }}").unwrap();
            }
            None => emit_loop(&mut s, false, false),
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
            threadgroup_bytes: t.staging_bytes(),
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
template <typename T> using vec4 = vec<T, 4>;
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
                // vec4 point-minor stages (`tensor_tile` sites)
                None if elem.starts_with("vec4<") => 4,
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
