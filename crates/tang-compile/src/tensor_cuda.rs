//! Native CUDA lowering of matmul-shaped tiles: tensor cores via `mma`.
//!
//! The same bilinear tile the Metal lowering recognizes
//! (`bilinear_tile`) emitted as CUDA C++: an [`MmaShape`] block (64×64 on
//! four warps up to 128×128 on eight), operands staged through
//! double-buffered shared memory from any affine layout (`cp.async`,
//! 16-byte where the layout allows), `mma.sync` m16n8k8 TF32 fragments
//! with fp32 accumulation, and the comp's epilogue run on the
//! accumulator registers. Flash attention uses the same fragments
//! ([`FlashShape`]).
//!
//! tang-gpu runs on wgpu, which cannot dispatch CUDA; this emitter
//! produces source for an external harness (`scripts/cuda_bench.cu`,
//! compiled with nvcc) that times the kernel against cuBLAS on the same
//! device. The kernel takes the read buffers in `reads` order, then
//! `out`; `TANG_TILES` blocks of `TANG_THREADS` threads.
use std::fmt::Write;

use super::msl_mma::Bilinear;
use super::tile::emit_scalar;
use super::wgsl::{Target, WgslProgram};
use super::{TensorGraph, TensorId};

/// A tensor-core tile's block shape: a `bm × bn` output block per thread
/// block of `warps_m × warps_n` warps (each a `bm/warps_m × bn/warps_n`
/// warp tile of 16×8 `mma` fragments), the reduction staged `chunk` at a
/// time through a ring of `stages` shared-memory buffers (`cp.async`
/// where the operand is a plain read, so the next chunks' copies overlap
/// this chunk's math), and split-K up to about `split_target` blocks
/// (1: never split).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmaShape {
    pub bm: usize,
    pub bn: usize,
    pub warps_m: usize,
    pub warps_n: usize,
    pub chunk: usize,
    pub stages: usize,
    pub split_target: usize,
}

impl MmaShape {
    pub const fn new(bm: usize, bn: usize, warps_m: usize, warps_n: usize) -> Self {
        Self {
            bm,
            bn,
            warps_m,
            warps_n,
            chunk: 32,
            stages: 3,
            split_target: 100,
        }
    }
    const fn with(mut self, chunk: usize, stages: usize, split_target: usize) -> Self {
        self.chunk = chunk;
        self.stages = stages;
        self.split_target = split_target;
        self
    }
    fn threads(&self) -> usize {
        32 * self.warps_m * self.warps_n
    }
    /// Row strides of the staged operands. Fragments are read as `float2`
    /// (see the k and column pairing in the kernel): A `[bm][chunk]` as 8
    /// rows × 4 pairs per half-warp-pair (stride ≡ 8 mod 32 is
    /// conflict-free), B `[chunk][bn]` as 4 rows × 4 pairs per half warp
    /// (≡ 4 mod 32). Both stay multiples of 4 floats for 16-byte
    /// `cp.async`.
    fn lda(&self) -> usize {
        self.chunk + 8
    }
    fn ldb(&self) -> usize {
        self.bn + 4
    }
}

/// The block shapes a tuner chooses between, per kernel; the first is the
/// default. 64×64 on four warps fills the device at the model's sizes;
/// the larger blocks halve the staging per flop where there are enough
/// output tiles; split-K only pays when tiles are very few.
pub const MMA_SHAPES: [MmaShape; 6] = [
    MmaShape::new(64, 64, 2, 2),
    MmaShape::new(64, 64, 2, 2).with(32, 3, 1),
    MmaShape::new(64, 64, 2, 2).with(16, 4, 1),
    MmaShape::new(64, 64, 2, 2).with(32, 2, 246),
    MmaShape::new(128, 64, 2, 2),
    MmaShape::new(128, 128, 2, 4).with(32, 2, 100),
];

/// A flash-attention tile's shape: `bi` queries per folded head per
/// block (0: chosen from the problem so the device fills), `bj` keys per
/// warp group per pipeline step, a `cp.async` ring of `stages` key/value
/// buffers, and `kw` warp groups splitting each step's keys (their
/// softmax states merged at the end).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashShape {
    pub bi: usize,
    pub bj: usize,
    pub stages: usize,
    pub kw: usize,
}

/// The flash shapes a tuner chooses between, one per [`MMA_SHAPES`]
/// variant; the first is the default.
pub const FLASH_SHAPES: [FlashShape; 6] = [
    FlashShape {
        bi: 0,
        bj: 32,
        stages: 2,
        kw: 2,
    },
    FlashShape {
        bi: 0,
        bj: 64,
        stages: 2,
        kw: 1,
    },
    FlashShape {
        bi: 0,
        bj: 16,
        stages: 3,
        kw: 2,
    },
    FlashShape {
        bi: 0,
        bj: 16,
        stages: 2,
        kw: 4,
    },
    FlashShape {
        bi: 32,
        bj: 32,
        stages: 2,
        kw: 2,
    },
    FlashShape {
        bi: 0,
        bj: 32,
        stages: 3,
        kw: 1,
    },
];

/// The PTX helpers the tensor-core kernels share: TF32 rounding, `mma`
/// m16n8k8, and `cp.async` (zero-filling when the source is out of range).
const MMA_PRELUDE: &str = r#"static __device__ __forceinline__ unsigned tang_tf32(float x) {
    // round to nearest, ties away (cvt.rna): the mma reads only the top
    // 19 bits, so adding half a TF32 ulp is the whole conversion
    return __float_as_uint(x) + 0x1000u;
}
static __device__ __forceinline__ void tang_mma(float* c, const unsigned* a, const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
static __device__ __forceinline__ void tang_cp4(float* dst, const float* src, bool ok) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(d), "l"(src), "r"(ok ? 4 : 0));
}
static __device__ __forceinline__ void tang_cp16(float* dst, const float* src, bool ok) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(ok ? 16 : 0));
}
static __device__ __forceinline__ void tang_cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> static __device__ __forceinline__ void tang_cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
"#;

impl TensorGraph {
    /// CUDA source for `root` as a tensor-core tile in the default block
    /// shape, or `None` when it is not matmul-shaped.
    pub fn cuda_mma_kernel(&self, root: usize, reads: &[usize]) -> Option<String> {
        self.cuda_mma_kernel_shaped(root, reads, MMA_SHAPES[0])
    }

    /// CUDA source for `root` as a tensor-core tile of block shape `sh`,
    /// or `None` when it is not matmul-shaped (or does not fit). Several
    /// terms (a sum of products) and several outputs (a tuple-valued
    /// reduction such as merged gate/up projections) share one kernel:
    /// distinct operands are staged once, each output accumulates its own
    /// fragments, and the epilogue runs on the accumulator registers.
    pub fn cuda_mma_kernel_shaped(
        &self,
        root: usize,
        reads: &[usize],
        sh: MmaShape,
    ) -> Option<String> {
        let c = self.comp_at(root)?;
        let Bilinear {
            red_node,
            epilogue,
            outs,
            pts,
            dx,
            dy,
            others,
            m,
            n,
            k_total,
            kdim,
            ..
        } = self.bilinear_tile(root, reads)?;
        let (bm, bn, chunk) = (sh.bm, sh.bn, sh.chunk);
        let threads = sh.threads();
        let (lda, ldb) = (sh.lda(), sh.ldb());
        // per warp: fm × fn fragments of 16×8
        let (wtm, wtn) = (bm / sh.warps_m, bn / sh.warps_n);
        let (fm, fnn) = (wtm / 16, wtn / 8);
        if fm == 0 || wtm % 16 != 0 || wtn % 16 != 0 || chunk % 8 != 0 {
            return None;
        }
        let n_outs = outs.len();
        // distinct operand stages; each (output, term) is an (A, B) pair
        let mut a_stages: Vec<(usize, super::msl_mma::Addr, Option<usize>)> = Vec::new();
        let mut b_stages: Vec<(usize, super::msl_mma::Addr, Option<usize>)> = Vec::new();
        let mut uses: Vec<Vec<(usize, usize)>> = Vec::new();
        for terms in &outs {
            let mut u = Vec::new();
            for t in terms {
                let ai = a_stages
                    .iter()
                    .position(|(b, a, _)| *b == t.buf_a && *a == t.addr_a)
                    .unwrap_or_else(|| {
                        a_stages.push((t.buf_a, t.addr_a.clone(), t.map_a));
                        a_stages.len() - 1
                    });
                let bi = b_stages
                    .iter()
                    .position(|(b, a, _)| *b == t.buf_b && *a == t.addr_b)
                    .unwrap_or_else(|| {
                        b_stages.push((t.buf_b, t.addr_b.clone(), t.map_b));
                        b_stages.len() - 1
                    });
                u.push((ai, bi));
            }
            uses.push(u);
        }
        // accumulators: n_outs × fm × fnn × 4 floats per thread
        if n_outs * fm * fnn * 4 > 128 {
            return None;
        }
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let tiles_x = n.div_ceil(bn);
        let tiles_y = m.div_ceil(bm);
        let n_base = tiles_x * tiles_y * others.iter().map(|&d| pts[d]).product::<usize>();
        let splits = split_k(n_base, k_total, chunk, sh.split_target);
        let n_tiles = n_base * splits;
        let w_out = c.width();
        let a_floats = bm * lda;
        let b_floats = chunk * ldb;
        let stage_floats = a_stages.len() * a_floats + b_stages.len() * b_floats;
        let smem_floats = sh.stages * stage_floats;
        // dynamic shared memory, within the 3090's 99 KB per block
        if smem_floats * 4 > 99 * 1024 {
            // Many distinct operands (a multi-output, multi-term tile such
            // as the gate/up backward): shrink the ring, then the chunk,
            // rather than leave the shape without a lowering.
            let mut fit = sh;
            if sh.stages > 2 {
                fit.stages -= 1;
            } else if sh.chunk > 16 {
                fit.chunk /= 2;
            } else {
                return None;
            }
            return self.cuda_mma_kernel_shaped(root, reads, fit);
        }

        let mut s = String::new();
        writeln!(s, "// Auto-generated by tang-compile: mma (TF32) tile").unwrap();
        writeln!(
            s,
            "// block {bm}x{bn}, {}x{} warps, chunk {chunk}, {} stages",
            sh.warps_m, sh.warps_n, sh.stages
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {n_tiles}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        writeln!(s, "#define TANG_SMEM {}", smem_floats * 4).unwrap();
        writeln!(s, "#define TANG_OUT_LEN {}", numel_pts(&pts) * w_out).unwrap();
        writeln!(s, "#define select(f, t, c) ((c) ? (t) : (f))").unwrap();
        s.push_str(MMA_PRELUDE);
        if splits > 1 {
            // per-split partial blocks, and an arrival counter per output
            // tile (the last split to arrive reduces and runs the
            // epilogue, then resets its counter for the next launch)
            writeln!(
                s,
                "__device__ float tang_partial[{}];",
                n_tiles * n_outs * bm * bn
            )
            .unwrap();
            writeln!(s, "__device__ unsigned tang_arrived[{n_base}];").unwrap();
        }
        let mut params: Vec<String> = reads
            .iter()
            .map(|&j| format!("const float* __restrict__ b{j}"))
            .collect();
        params.push("float* __restrict__ out".into());
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void __launch_bounds__({threads}) main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(s, "    (void)count;").unwrap();
        writeln!(s, "    extern __shared__ __align__(16) float smem[];").unwrap();
        writeln!(s, "    unsigned tile = blockIdx.x;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) return;").unwrap();
        let per_split = k_total.div_ceil(chunk).div_ceil(splits) * chunk;
        writeln!(s, "    unsigned split = tile / {n_base}u;").unwrap();
        writeln!(s, "    tile = tile % {n_base}u;").unwrap();
        writeln!(s, "    unsigned k_begin = split * {per_split}u;").unwrap();
        writeln!(
            s,
            "    unsigned k_end = min(k_begin + {per_split}u, {k_total}u);"
        )
        .unwrap();
        writeln!(
            s,
            "    unsigned n_chunks = k_end > k_begin ? (k_end - k_begin + {}u) / {chunk}u : 0u;",
            chunk - 1
        )
        .unwrap();
        writeln!(s, "    unsigned tile_x = tile % {tiles_x}u;").unwrap();
        writeln!(s, "    unsigned tile_y = (tile / {tiles_x}u) % {tiles_y}u;").unwrap();
        writeln!(s, "    unsigned lead = tile / {}u;", tiles_x * tiles_y).unwrap();
        for &d in others.iter().rev() {
            writeln!(s, "    unsigned p{d} = lead % {}u;", pts[d]).unwrap();
            writeln!(s, "    lead = lead / {}u;", pts[d]).unwrap();
        }
        writeln!(s, "    unsigned row0 = tile_y * {bm}u;").unwrap();
        writeln!(s, "    unsigned col0 = tile_x * {bn}u;").unwrap();
        let lead_term = |addr: &super::msl_mma::Addr| -> String {
            let mut t = format!("{}", addr.offset);
            for &d in &others {
                if addr.coefs[d] != 0 {
                    t.push_str(&format!(" + {} * (int)p{d}", addr.coefs[d]));
                }
            }
            t
        };
        for (t, (_, addr, _)) in a_stages.iter().enumerate() {
            writeln!(s, "    int base_a{t} = {};", lead_term(addr)).unwrap();
        }
        for (t, (_, addr, _)) in b_stages.iter().enumerate() {
            writeln!(s, "    int base_b{t} = {};", lead_term(addr)).unwrap();
        }
        let rc = self.comp_at(red_node)?;
        let rank = rc.rank();
        writeln!(s, "    unsigned tid = threadIdx.x;").unwrap();
        writeln!(s, "    unsigned warp = tid / 32u;").unwrap();
        writeln!(s, "    unsigned lane = tid % 32u;").unwrap();
        writeln!(s, "    unsigned gq = lane / 4u, tq = lane % 4u;").unwrap();
        writeln!(s, "    unsigned wr = (warp / {}u) * {wtm}u;", sh.warps_n).unwrap();
        writeln!(s, "    unsigned wc = (warp % {}u) * {wtn}u;", sh.warps_n).unwrap();
        writeln!(s, "    float acc[{n_outs}][{fm}][{fnn}][4];").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int w = 0; w < {n_outs}; w++) for (int i = 0; i < {fm}; i++) for (int j = 0; j < {fnn}; j++) for (int r = 0; r < 4; r++) acc[w][i][j][r] = 0.0f;").unwrap();

        // staging of one chunk into buffer `buf`, walking the global
        // buffer in address order
        writeln!(s, "    auto load = [&](unsigned buf, unsigned k0) {{").unwrap();
        writeln!(s, "        float* st = smem + buf * {stage_floats}u;").unwrap();
        let stage = |s: &mut String,
                     off: usize,
                     base: &str,
                     addr: &super::msl_mma::Addr,
                     pdim: usize,
                     origin: &str,
                     extent: usize,
                     block: usize,
                     buf: usize,
                     map: Option<usize>,
                     row_major: bool| {
            let (cp, ck) = (addr.coefs[pdim], addr.coefs[kdim]);
            let slot = |pt: &str, kk: &str| {
                if row_major {
                    format!("st + {off}u + {pt} * {lda}u + {kk}")
                } else {
                    format!("st + {off}u + {kk} * {ldb}u + {pt}")
                }
            };
            // 16-byte copies along the shared row: the global stride
            // there is 1 and every vector start is 4-aligned and either
            // wholly in range or wholly out
            let lead_aligned =
                addr.offset % 4 == 0 && others.iter().all(|&d| addr.coefs[d] % 4 == 0);
            let vec4 = map.is_none()
                && lead_aligned
                && if row_major {
                    ck == 1 && cp % 4 == 0 && k_total % 4 == 0 && chunk % 4 == 0
                } else {
                    cp == 1 && ck % 4 == 0 && extent.is_multiple_of(4) && block.is_multiple_of(4)
                };
            if vec4 {
                let (inner, outer) = if row_major {
                    (chunk, block)
                } else {
                    (block, chunk)
                };
                let nv = inner / 4;
                let total = inner * outer / 4;
                // a fixed trip count where the threads tile the block
                // evenly: every thread keeps its column, rows step by a
                // constant, and the index math hoists out of the k loop
                let (o, i) = if row_major {
                    ("pt", "kk")
                } else {
                    ("kk", "pt")
                };
                if total % threads == 0 && threads.is_multiple_of(nv) {
                    writeln!(s, "        #pragma unroll").unwrap();
                    writeln!(
                        s,
                        "        for (unsigned q = 0u; q < {}u; q++) {{",
                        total / threads
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "            unsigned {i} = (tid % {nv}u) * 4u; unsigned {o} = tid / {nv}u + q * {}u;",
                        threads / nv
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "        for (unsigned idx = tid; idx < {total}u; idx += {threads}u) {{"
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "            unsigned {i} = (idx % {nv}u) * 4u; unsigned {o} = idx / {nv}u;"
                    )
                    .unwrap();
                }
                writeln!(
                    s,
                    "            unsigned gp = {origin} + pt; unsigned gk = k0 + kk;"
                )
                .unwrap();
                writeln!(
                    s,
                    "            bool ok = gp < {extent}u && gk < {k_total}u;"
                )
                .unwrap();
                writeln!(
                    s,
                    "            tang_cp16({}, b{buf} + (ok ? (unsigned)({base} + {cp} * (int)gp + {ck} * (int)gk) : 0u), ok);",
                    slot("pt", "kk")
                )
                .unwrap();
                writeln!(s, "        }}").unwrap();
                return;
            }
            let k_inner = ck.unsigned_abs() <= cp.unsigned_abs();
            writeln!(
                s,
                "        for (unsigned idx = tid; idx < {}u; idx += {threads}u) {{",
                block * chunk
            )
            .unwrap();
            if k_inner {
                writeln!(
                    s,
                    "            unsigned kk = idx % {chunk}u; unsigned pt = idx / {chunk}u;"
                )
                .unwrap();
            } else {
                writeln!(
                    s,
                    "            unsigned pt = idx % {block}u; unsigned kk = idx / {block}u;"
                )
                .unwrap();
            }
            writeln!(
                s,
                "            unsigned gp = {origin} + pt; unsigned gk = k0 + kk;"
            )
            .unwrap();
            let Some(v) = map else {
                writeln!(
                    s,
                    "            bool ok = gp < {extent}u && gk < {k_total}u;"
                )
                .unwrap();
                writeln!(
                    s,
                    "            tang_cp4({}, b{buf} + (ok ? (unsigned)({base} + {cp} * (int)gp + {ck} * (int)gk) : 0u), ok);",
                    slot("pt", "kk")
                )
                .unwrap();
                writeln!(s, "        }}").unwrap();
                return;
            };
            // a map operand: its body over the args it reads, each at its
            // address composed through the map's view
            let outer = &rc.args[v];
            let mc = self.comp_at(outer.src.n()).unwrap();
            let coord = |d: usize| -> String {
                if d == pdim {
                    "(int)gp".into()
                } else if d == kdim {
                    "(int)gk".into()
                } else {
                    format!("(int)p{d}")
                }
            };
            let affine = |offset: i64, coefs: &[i64]| -> String {
                let mut e = format!("{offset}");
                for (d, &cf) in coefs.iter().enumerate() {
                    if cf != 0 {
                        e.push_str(&format!(" + {cf} * {}", coord(d)));
                    }
                }
                e
            };
            let vals: Vec<String> = mc
                .args
                .iter()
                .map(|sa| {
                    let view = sa.view.compose(&outer.view, rank);
                    match &self.nodes[sa.src.n()].op {
                        super::Op::Iota(dd) => {
                            let f = &view.forms[*dd];
                            format!("(float)({})", affine(f.offset, &f.coefs))
                        }
                        _ => {
                            let composed = super::Arg { src: sa.src, view };
                            let a = self.flat_addr(&composed, rank);
                            format!(
                                "b{}[(unsigned)({})]",
                                storage(sa.src.n()),
                                affine(a.offset, &a.coefs)
                            )
                        }
                    }
                })
                .collect();
            writeln!(s, "            float v = 0.0f;").unwrap();
            writeln!(s, "            if (gp < {extent}u && gk < {k_total}u) {{").unwrap();
            let mut body = String::new();
            let outs = emit_scalar(
                &mut body,
                &mc.body,
                &mc.body_outs,
                &format!("m{off}_"),
                &|x| vals[x as usize].clone(),
            );
            for line in body.lines() {
                writeln!(
                    s,
                    "                {}",
                    float_literals(&line.replacen("let ", "float ", 1))
                )
                .unwrap();
            }
            writeln!(s, "                v = {};", outs[0]).unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(s, "            *({}) = v;", slot("pt", "kk")).unwrap();
            writeln!(s, "        }}").unwrap();
        };
        for (t, (buf, addr, map)) in a_stages.iter().enumerate() {
            stage(
                &mut s,
                t * a_floats,
                &format!("base_a{t}"),
                addr,
                dy,
                "row0",
                m,
                bm,
                *buf,
                *map,
                true,
            );
        }
        for (t, (buf, addr, map)) in b_stages.iter().enumerate() {
            stage(
                &mut s,
                a_stages.len() * a_floats + t * b_floats,
                &format!("base_b{t}"),
                addr,
                dx,
                "col0",
                n,
                bn,
                *buf,
                *map,
                false,
            );
        }
        writeln!(s, "        tang_cp_commit();").unwrap();
        writeln!(s, "    }};").unwrap();

        // fragments for k step `kk` of buffer `buf` into register set
        // `par`. A reduction is order-free, so within each 8-wide k step
        // the mma k slots (tq, tq + 4) read staged columns (2tq, 2tq + 1)
        // of A and rows (2tq, 2tq + 1) of B: one float2 per A row pair.
        // Columns are ours to place too: fragments (2jp, 2jp + 1) take
        // staged columns 2gq and 2gq + 1 of their 16, one float2 per B row.
        let n_a = a_stages.len();
        let n_b = b_stages.len();
        writeln!(s, "    unsigned af[2][{n_a}][{fm}][4];").unwrap();
        writeln!(s, "    unsigned bf[2][{n_b}][{fnn}][2];").unwrap();
        let frags = |s: &mut String, par: usize, buf: &str, kk: &str| {
            writeln!(s, "        {{").unwrap();
            writeln!(
                s,
                "        const float* st = smem + ({buf}) * {stage_floats}u;"
            )
            .unwrap();
            for t in 0..n_a {
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int i = 0; i < {fm}; i++) {{").unwrap();
                writeln!(s, "            const float* p = st + {}u + (wr + 16u * i + gq) * {lda}u + ({kk}) + 2u * tq;", t * a_floats).unwrap();
                writeln!(
                    s,
                    "            float2 lo = *(const float2*)p, hi = *(const float2*)(p + {}u);",
                    8 * lda
                )
                .unwrap();
                writeln!(s, "            af[{par}][{t}][i][0] = tang_tf32(lo.x); af[{par}][{t}][i][1] = tang_tf32(hi.x);").unwrap();
                writeln!(s, "            af[{par}][{t}][i][2] = tang_tf32(lo.y); af[{par}][{t}][i][3] = tang_tf32(hi.y);").unwrap();
                writeln!(s, "        }}").unwrap();
            }
            for t in 0..n_b {
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int jp = 0; jp < {}; jp++) {{", fnn / 2).unwrap();
                writeln!(s, "            const float* p = st + {}u + (({kk}) + 2u * tq) * {ldb}u + wc + 16u * jp + 2u * gq;", n_a * a_floats + t * b_floats).unwrap();
                writeln!(
                    s,
                    "            float2 r0 = *(const float2*)p, r1 = *(const float2*)(p + {ldb}u);"
                )
                .unwrap();
                writeln!(s, "            bf[{par}][{t}][2 * jp][0] = tang_tf32(r0.x); bf[{par}][{t}][2 * jp][1] = tang_tf32(r1.x);").unwrap();
                writeln!(s, "            bf[{par}][{t}][2 * jp + 1][0] = tang_tf32(r0.y); bf[{par}][{t}][2 * jp + 1][1] = tang_tf32(r1.y);").unwrap();
                writeln!(s, "        }}").unwrap();
            }
            writeln!(s, "        }}").unwrap();
        };
        let mmas = |s: &mut String, par: usize| {
            for (w, u) in uses.iter().enumerate() {
                for (ai, bi) in u {
                    writeln!(s, "        #pragma unroll").unwrap();
                    writeln!(s, "        for (int i = 0; i < {fm}; i++) for (int j = 0; j < {fnn}; j++) tang_mma(acc[{w}][i][j], af[{par}][{ai}][i], bf[{par}][{bi}][j]);").unwrap();
                }
            }
        };
        let steps = chunk / 8;
        if sh.stages >= 2 {
            // a ring of `stages` buffers, chunk c + stages - 1 in flight
            // while chunk c is multiplied; the fragments of the next k step
            // load while this one's mma issue (across the chunk boundary
            // too, after the one barrier per chunk)
            let st = sh.stages;
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (unsigned c = 0u; c < {}u; c++) {{ if (c < n_chunks) load(c, k_begin + c * {chunk}u); else tang_cp_commit(); }}", st - 1).unwrap();
            writeln!(s, "    tang_cp_wait<{}>();", st - 2).unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    if (n_chunks > 0u) {{").unwrap();
            frags(&mut s, 0, "0u", "0u");
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    for (unsigned c = 0u; c < n_chunks; c++) {{").unwrap();
            writeln!(s, "        unsigned nc = c + {}u;", st - 1).unwrap();
            writeln!(s, "        if (nc < n_chunks) load(nc % {st}u, k_begin + nc * {chunk}u); else tang_cp_commit();").unwrap();
            for ks in 0..steps {
                let par = ks % 2;
                if ks + 1 < steps {
                    frags(
                        &mut s,
                        1 - par,
                        &format!("c % {st}u"),
                        &format!("{}u", (ks + 1) * 8),
                    );
                } else {
                    writeln!(s, "        tang_cp_wait<{}>();", st - 2).unwrap();
                    writeln!(s, "        __syncthreads();").unwrap();
                    writeln!(s, "        if (c + 1u < n_chunks) {{").unwrap();
                    frags(&mut s, 1 - par, &format!("(c + 1u) % {st}u"), "0u");
                    writeln!(s, "        }}").unwrap();
                }
                mmas(&mut s, par);
            }
            if steps % 2 == 1 {
                // odd step count: the next chunk's first fragments landed
                // in set 1; move them to set 0
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int t = 0; t < {n_a}; t++) for (int i = 0; i < {fm}; i++) for (int e = 0; e < 4; e++) af[0][t][i][e] = af[1][t][i][e];").unwrap();
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int t = 0; t < {n_b}; t++) for (int j = 0; j < {fnn}; j++) for (int e = 0; e < 2; e++) bf[0][t][j][e] = bf[1][t][j][e];").unwrap();
            }
            writeln!(s, "    }}").unwrap();
        } else {
            writeln!(s, "    for (unsigned c = 0u; c < n_chunks; c++) {{").unwrap();
            writeln!(s, "        load(0u, k_begin + c * {chunk}u);").unwrap();
            writeln!(s, "        tang_cp_wait<0>();").unwrap();
            writeln!(s, "        __syncthreads();").unwrap();
            for ks in 0..steps {
                frags(&mut s, 0, "0u", &format!("{}u", ks * 8));
                mmas(&mut s, 0);
            }
            writeln!(s, "        __syncthreads();").unwrap();
            writeln!(s, "    }}").unwrap();
        }

        // the epilogue straight from the accumulators: element r of
        // fragment (i, j) is row gq + 8·(r/2) and mma column 2·tq + r%2,
        // staged column 2·(that) + j%2 of the fragment pair's 16
        let bb = bm * bn;
        let frag_loop = |s: &mut String| {
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int i = 0; i < {fm}; i++)").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int j = 0; j < {fnn}; j++)").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int r = 0; r < 4; r++) {{").unwrap();
            writeln!(s, "        unsigned lr = wr + 16u * i + gq + 8u * (r / 2);").unwrap();
            writeln!(
                s,
                "        unsigned lc = wc + 16u * (j / 2) + 4u * tq + 2u * (r % 2) + (j % 2);"
            )
            .unwrap();
        };
        if splits > 1 {
            writeln!(
                s,
                "    float* part = tang_partial + (split * {n_base}u + tile) * {}u;",
                n_outs * bb
            )
            .unwrap();
            frag_loop(&mut s);
            writeln!(s, "        for (int w = 0; w < {n_outs}; w++) part[w * {bb}u + lr * {bn}u + lc] = acc[w][i][j][r];").unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    __threadfence();").unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    __shared__ unsigned last;").unwrap();
            writeln!(s, "    if (tid == 0u) {{").unwrap();
            writeln!(
                s,
                "        last = atomicAdd(&tang_arrived[tile], 1u) == {}u;",
                splits - 1
            )
            .unwrap();
            writeln!(s, "        if (last) tang_arrived[tile] = 0u;").unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    if (!last) return;").unwrap();
            writeln!(s, "    __threadfence();").unwrap();
        }
        frag_loop(&mut s);
        writeln!(s, "        unsigned p{dy} = row0 + lr;").unwrap();
        writeln!(s, "        unsigned p{dx} = col0 + lc;").unwrap();
        writeln!(s, "        if (p{dy} < {m}u && p{dx} < {n}u) {{").unwrap();
        for w in 0..n_outs {
            if splits > 1 {
                // the splits in a fixed order: deterministic
                writeln!(s, "            float red{w} = 0.0f;").unwrap();
                writeln!(s, "            for (unsigned sp = 0u; sp < {splits}u; sp++) red{w} += __ldcg(tang_partial + (sp * {n_base}u + tile) * {}u + {}u + lr * {bn}u + lc);", n_outs * bb, w * bb).unwrap();
            } else {
                writeln!(s, "            float red{w} = acc[{w}][i][j][r];").unwrap();
            }
        }
        let vals: Vec<String> = if epilogue {
            let prank = c.points().len();
            let mut arg_vals = Vec::new();
            for a in &c.args {
                if a.src.n() == red_node {
                    arg_vals.push(format!("red{}", self.epilogue_slot(a)));
                    continue;
                }
                let addr = self.flat_addr(a, prank);
                let mut e = format!("{}", addr.offset);
                for (d, &coef) in addr.coefs.iter().enumerate() {
                    if coef != 0 {
                        e.push_str(&format!(" + {coef} * (int)p{d}"));
                    }
                }
                arg_vals.push(format!("b{}[(unsigned)({e})]", storage(a.src.n())));
            }
            let mut body = String::new();
            let outs = emit_scalar(&mut body, &c.body, &c.body_outs, "e0_", &|v| {
                arg_vals[v as usize].clone()
            });
            for line in body.lines() {
                writeln!(
                    s,
                    "            {}",
                    float_literals(&line.replacen("let ", "float ", 1))
                )
                .unwrap();
            }
            outs
        } else {
            (0..n_outs).map(|w| format!("red{w}")).collect()
        };
        let mut flat = String::from("0u");
        for (d, &np) in pts.iter().enumerate() {
            flat = format!("({flat}) * {np}u + p{d}");
        }
        writeln!(s, "            unsigned g = {flat};").unwrap();
        for (slot, v) in vals.iter().enumerate() {
            writeln!(s, "            out[g * {w_out}u + {slot}u] = {v};").unwrap();
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        Some(s)
    }

    /// The CUDA tensor-core kernel for the one tiled matmul in `outputs`'
    /// program, with its read buffers in binding order. For the harness.
    pub fn cuda_mma_program(&self, outputs: &[TensorId]) -> Option<(String, Vec<TensorId>)> {
        self.cuda_mma_program_shaped(outputs, MMA_SHAPES[0])
    }

    /// [`cuda_mma_program`](Self::cuda_mma_program) in block shape `sh`.
    pub fn cuda_mma_program_shaped(
        &self,
        outputs: &[TensorId],
        sh: MmaShape,
    ) -> Option<(String, Vec<TensorId>)> {
        let live = self.live(outputs);
        for root in (0..self.nodes.len()).filter(|&i| live[i] && !self.nodes[i].inline) {
            if self.nodes[root].tile.0 == 0 {
                continue;
            }
            let order = self.inline_closure(root);
            let reads: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&j| {
                    j != root
                        && !self.nodes[j].inline
                        && !matches!(self.nodes[j].op, super::Op::Iota(_) | super::Op::Reshape(_))
                })
                .collect();
            if let Some(src) = self.cuda_mma_kernel_shaped(root, &reads, sh) {
                return Some((src, reads.iter().map(|&j| TensorId(j as u32)).collect()));
            }
        }
        None
    }
}

/// How many ways to split a tile's reduction (split-K): up to `target`
/// blocks (the shape's, or `TANG_SPLITK_TARGET`) while every split keeps
/// at least four chunks of the reduction.
fn split_k(n_tiles: usize, k_total: usize, chunk: usize, target: usize) -> usize {
    let target: usize = std::env::var("TANG_SPLITK_TARGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(target);
    let mut s = 1;
    while n_tiles * s * 2 <= target && k_total.div_ceil(chunk) >= s * 2 * 4 && s < 16 {
        s *= 2;
    }
    s
}

fn numel_pts(pts: &[usize]) -> usize {
    pts.iter().product()
}

/// The CUDA calling convention every kernel of a [`to_cuda`] program
/// follows: `extern "C" __global__ void main0(const float* b{j}…, float* out,
/// unsigned count)` with the reads in `WgslKernel::reads` order, launched
/// as `count / workgroup_size` blocks of `workgroup_size` threads.
pub const CUDA_ENTRY: &str = "main0";

impl TensorGraph {
    /// The program as CUDA C++ (for nvrtc): matmul-shaped tiles as mma
    /// tensor-core kernels, everything else the WGSL kernels transpiled.
    /// Same `WgslProgram` shape — sources, reads, counts — so a runtime
    /// drives both the same way.
    pub fn to_cuda(&self, outputs: &[TensorId]) -> WgslProgram {
        self.to_cuda_with(outputs, true)
    }

    /// [`to_cuda`](Self::to_cuda), with the native (tensor-core)
    /// lowerings on or off. Both programs have the same kernels in the
    /// same order with the same bindings, so a tuner can pick per kernel.
    pub fn to_cuda_with(&self, outputs: &[TensorId], native: bool) -> WgslProgram {
        self.to_cuda_variant(outputs, native.then_some(0))
    }

    /// [`to_cuda`](Self::to_cuda) with the matmul tiles in block shape
    /// `MMA_SHAPES[v]`, or every kernel transpiled for `None`. All
    /// variants have the same kernels in the same order with the same
    /// bindings, so a tuner can pick a variant per kernel.
    pub fn to_cuda_variant(&self, outputs: &[TensorId], variant: Option<usize>) -> WgslProgram {
        let mut program = self.to_program_numbered(outputs, Target::Wgsl);
        for k in &mut program.kernels {
            let root = k.node.index();
            let reads: Vec<usize> = k.reads.iter().map(|t| t.index()).collect();
            let native = match variant {
                Some(v) if self.nodes[root].tile.0 > 0 => self
                    .cuda_mma_kernel_shaped(root, &reads, MMA_SHAPES[v])
                    .or_else(|| {
                        self.cuda_flash_kernel_shaped(
                            root,
                            &reads,
                            FLASH_SHAPES[v % FLASH_SHAPES.len()],
                        )
                    }),
                _ => None,
            };
            match native {
                Some(src) => {
                    k.source = src;
                    let threads = src_define(&k.source, "TANG_THREADS");
                    k.workgroup_size = threads as u32;
                    k.count = {
                        let n_tiles: usize = src_define(&k.source, "TANG_TILES");
                        (n_tiles * threads) as u32
                    };
                }
                None => k.source = wgsl_to_cuda(&k.source, k.workgroup_size),
            }
        }
        program.canonical()
    }
}

fn src_define(src: &str, name: &str) -> usize {
    src.lines()
        .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
        .and_then(|v| v.trim().parse().ok())
        .expect("define")
}

/// Generated WGSL is a small, regular language: typed `fn`s over `f32`,
/// `u32` and `array<f32, N>`, `let`/`var` bindings, `for` loops over
/// `u32`, workgroup arrays, and a `main` over builtin ids. That maps to
/// CUDA C++ line by line with a short prelude (`Arr<N>` for arrays,
/// `Vec4` for vec4, `select` as a macro) — no second emitter.
pub fn wgsl_to_cuda(wgsl: &str, workgroup_size: u32) -> String {
    let mut out = String::new();
    out.push_str(CUDA_PRELUDE);
    let mut reads: Vec<String> = Vec::new();
    let mut body = String::new();
    for line in wgsl.lines() {
        let t = line.trim_start();
        let indent = &line[..line.len() - t.len()];
        if t.starts_with("struct Params") {
            continue;
        }
        if let Some(rest) = t.strip_prefix("@group(0) @binding(") {
            // `N) var<storage, read> bJ: array<f32>;`
            if rest.contains("var<uniform>") {
                continue;
            }
            let name = rest
                .split("var<")
                .nth(1)
                .and_then(|r| r.split('>').nth(1))
                .map(|r| {
                    r.trim()
                        .trim_end_matches(':')
                        .split(':')
                        .next()
                        .unwrap()
                        .trim()
                        .to_string()
                })
                .expect("binding name");
            if name == "out" {
                body.push_str("__shared__ float* out;\n");
            } else {
                body.push_str(&format!("__shared__ const float* {name};\n"));
                reads.push(name);
            }
            continue;
        }
        if t.starts_with("@compute") {
            continue;
        }
        if t.starts_with("fn main(") {
            let mut params: Vec<String> = reads
                .iter()
                .map(|n| format!("const float* p_{n}"))
                .collect();
            params.push("float* p_out".into());
            params.push("unsigned count".into());
            body.push_str(&format!(
                "extern \"C\" __global__ void __launch_bounds__({workgroup_size}) main0({}) {{\n",
                params.join(", ")
            ));
            body.push_str("    if (threadIdx.x == 0u) {\n");
            for n in &reads {
                body.push_str(&format!("        {n} = p_{n};\n"));
            }
            body.push_str("        out = p_out;\n        params.count = count;\n    }\n    __syncthreads();\n");
            continue;
        }
        if let Some(rest) = t.strip_prefix("var<workgroup> ") {
            // `name: array<T, N>;`
            let (name, ty) = rest
                .trim_end_matches(';')
                .split_once(": ")
                .expect("workgroup var");
            body.push_str(&format!(
                "{}__shared__ {};\n",
                indent,
                cuda_array_decl(name, ty)
            ));
            continue;
        }
        if let Some(rest) = t.strip_prefix("fn ") {
            // `name(params) -> T {`
            let (sig, ret) = rest.split_once(" -> ").expect("fn return");
            let ret = ret.trim_end_matches(" {").trim_end_matches('{').trim();
            let (name, params) = sig.split_once('(').expect("fn params");
            let params = params.trim_end_matches(')');
            let params: Vec<String> = split_top(params)
                .into_iter()
                .filter(|p| !p.is_empty())
                .map(|p| {
                    let (n, ty) = p.split_once(": ").expect("param");
                    format!("{} {n}", cuda_type(ty))
                })
                .collect();
            body.push_str(&format!(
                "{indent}__device__ __forceinline__ {} {name}({}) {{\n",
                cuda_type(ret),
                params.join(", ")
            ));
            continue;
        }
        body.push_str(indent);
        body.push_str(&cuda_stmt(t));
        body.push('\n');
    }
    out.push_str("__shared__ Params params;\n");
    out.push_str(&body);
    out
}

const CUDA_PRELUDE: &str = r#"// Auto-generated by tang-compile: WGSL kernel as CUDA C++
struct Params { unsigned count; };
template <int N> struct Arr {
    float v[N];
    __device__ __forceinline__ float& operator[](unsigned i) { return v[i]; }
    __device__ __forceinline__ float operator[](unsigned i) const { return v[i]; }
};
struct Vec4 {
    float v[4];
    __device__ __forceinline__ float& operator[](unsigned i) { return v[i]; }
    __device__ __forceinline__ float operator[](unsigned i) const { return v[i]; }
};
#define select(f, t, c) ((c) ? (t) : (f))
__device__ __forceinline__ unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
"#;

/// Split on `, ` outside angle brackets (`a: array<f32, 2>, b: f32`).
fn split_top(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0usize, 0usize);
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'<' => depth += 1,
            b'>' => depth -= 1,
            b',' if depth == 0 && b.get(i + 1) == Some(&b' ') => {
                out.push(&s[start..i]);
                start = i + 2;
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    out.push(&s[start..]);
    out
}

fn cuda_type(ty: &str) -> String {
    let ty = ty.trim();
    if ty == "f32" {
        return "float".into();
    }
    if ty == "u32" {
        return "unsigned".into();
    }
    if ty == "i32" {
        return "int".into();
    }
    if ty == "bool" {
        return "bool".into();
    }
    if let Some(n) = ty
        .strip_prefix("array<f32, ")
        .and_then(|r| r.strip_suffix('>'))
    {
        return format!("Arr<{n}>");
    }
    if ty == "vec4<f32>" {
        return "Vec4".into();
    }
    panic!("wgsl type {ty}")
}

/// `name: array<T, N>` → `T name[N]`, with `T` itself an `Arr<M>` for a
/// nested array (workgroup accumulators) and `Vec4` for vec4 stages.
fn cuda_array_decl(name: &str, ty: &str) -> String {
    let ty = ty.trim();
    if let Some(inner) = ty.strip_prefix("array<").and_then(|r| r.strip_suffix('>')) {
        if let Some((elem, n)) = inner.rsplit_once(", ") {
            let elem = elem.trim();
            // outer dimension is a C array; the element type is a value type
            if elem.starts_with("array<") || elem == "vec4<f32>" {
                return format!("{} {name}[{n}]", cuda_type(elem));
            }
        }
    }
    format!("{} {name}", cuda_type(ty))
}

/// Float literals get an `f` suffix so the math stays single precision.
fn float_literals(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 8);
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let prev_ident = i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        if c.is_ascii_digit() && !prev_ident {
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            let mut is_float = false;
            if i < b.len() && b[i] == b'.' {
                is_float = true;
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            // an exponent makes it a float too (`1e30`)
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    is_float = true;
                    i = j;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            out.push_str(&s[start..i]);
            if is_float {
                out.push('f');
            }
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// One statement line. Bindings become `auto`, typed `var`s their type,
/// loops over `u32`, casts and constructors their C++ spellings.
fn cuda_stmt(t: &str) -> String {
    let mut s = t.to_string();
    if let Some(rest) = s.strip_prefix("let ") {
        s = format!("auto {rest}");
    } else if let Some(rest) = s.strip_prefix("var ") {
        // `var x: T = e;` or `var x: T;` or `var x = e;`
        let (lhs, rhs) = match rest.split_once(" = ") {
            Some((l, r)) => (l, Some(r)),
            None => (rest.trim_end_matches(';'), None),
        };
        let decl = match lhs.split_once(": ") {
            Some((n, ty)) => cuda_array_decl(n, ty),
            None => format!("auto {lhs}"),
        };
        s = match rhs {
            Some(r) => format!("{decl} = {r}"),
            None => format!("{decl};"),
        };
    } else if let Some(rest) = s.strip_prefix("for (var ") {
        // `for (var i: u32 = a; …`
        let (n, rest) = rest.split_once(": u32 = ").expect("for var");
        s = format!("for (unsigned {n} = {rest}");
    }
    s = s
        .replace("array<f32, ", "Arr<")
        .replace("vec4<f32>(", "Vec4{")
        .replace("u32(", "(unsigned)(")
        .replace("i32(", "(int)(")
        .replace("f32(", "(float)(")
        .replace("workgroupBarrier()", "__syncthreads()")
        .replace("gid.x", "(blockIdx.x * blockDim.x + threadIdx.x)")
        .replace("gid.y", "blockIdx.y")
        .replace("wid.x", "blockIdx.x")
        .replace("wid.y", "blockIdx.y")
        .replace("lid.x", "threadIdx.x");
    // `Arr<N>(a, b)` → `Arr<N>{{a, b}}`; `Vec4{a, …)` → `Vec4{{a, …}}`
    s = rewrite_ctor(&s, "Arr<");
    s = rewrite_ctor(&s, "Vec4{");
    float_literals(&s)
}

/// Turn `PREFIX…(args)` constructor calls into brace initialization.
fn rewrite_ctor(s: &str, prefix: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find(prefix) {
        out.push_str(&rest[..i]);
        let after = &rest[i..];
        // find the opening paren/brace of the argument list
        let open = if prefix.ends_with('{') {
            prefix.len() - 1
        } else {
            match after.find('(') {
                Some(p) if after[..p].contains('>') => p,
                _ => {
                    out.push_str(&after[..prefix.len()]);
                    rest = &after[prefix.len()..];
                    continue;
                }
            }
        };
        // Arr<N> followed by a space (a declaration), not a constructor
        if !prefix.ends_with('{') && after[..open].contains(' ') {
            out.push_str(&after[..prefix.len()]);
            rest = &after[prefix.len()..];
            continue;
        }
        let mut depth = 0usize;
        let mut close = None;
        for (k, ch) in after[open..].char_indices() {
            match ch {
                '(' | '{' => depth += 1,
                ')' | '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + k);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close.expect("ctor close");
        out.push_str(&after[..open]);
        out.push_str("{{");
        out.push_str(&after[open + 1..close]);
        out.push_str("}}");
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    out
}

// ---------------------------------------------------------------------------
// Attention on tensor cores
// ---------------------------------------------------------------------------

/// The shape of a fused attention: an online-softmax reduction over keys
/// whose score is an inlined bilinear comp (`Σ_d q[i,d]·k[j,d]`, possibly
/// several such terms), plus a mask over (i, j); whose value is a read
/// along (j, lane) — and a map reading it point-for-point (the
/// normalizer). Recognized by [`TensorGraph::flash_tile`].
struct Flash {
    red_node: usize,
    /// index of the arg carrying the score comp, and that comp's node
    score_arg: usize,
    score_node: usize,
    /// (q buffer, q address over the score space, k buffer, k address)
    terms: Vec<(usize, Addr, usize, Addr)>,
    /// the score read from a buffer (storage, address over the reduction
    /// space) instead of an inlined bilinear comp: the state-materializing
    /// forward of a training step, whose scores are kept for the backward
    score_buf: Option<(usize, Addr)>,
    /// whether `root` is a normalizer over the reduction (else the root is
    /// the reduction, writing its `(m, l, acc)` state)
    epilogue: bool,
    /// score space dim ↔ reduction space dim, for the score's point dims
    score_to_red: Vec<usize>,
    /// reduction space: query dim, lane dim, others (incl. unit dims)
    dy: usize,
    dx: usize,
    others: Vec<usize>,
    pts: Vec<usize>,
    m: usize,
    n_keys: usize,
    d_lanes: usize,
    kd: usize,
}

use super::msl_mma::Addr;
use super::vars_reachable;
use super::{Arg, Op};
use tang_expr::Node as ENode;

impl TensorGraph {
    fn flash_tile(&self, root: usize, reads: &[usize]) -> Option<Flash> {
        let c = self.comp_at(root)?;
        let red_node = self.tiled_reduction(root)?;
        let epilogue = red_node != root;
        let rc = self.comp_at(red_node)?;
        let m = rc.monoid.as_ref()?;
        if m.kind() != super::MonoidKind::OnlineSoftmax || m.width != 3 || rc.red.len() != 1 {
            return None;
        }
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let is_buffer = |j: usize| reads.contains(&storage(j));
        let pts = super::fixed(rc.points());
        let rank = rc.rank();
        let jdim = rank - 1;
        let dx = (0..pts.len()).rev().find(|&d| pts[d] > 1)?;
        let dy = super::tile::largest_other(&pts, dx)?;
        let others: Vec<usize> = (0..pts.len()).filter(|&d| d != dx && d != dy).collect();
        let d_lanes = pts[dx];
        if !d_lanes.is_multiple_of(16) || d_lanes > 64 {
            return None;
        }
        // body outs: (score + mask, 1, value)
        let nodes = rc.body.nodes_slice();
        let [e_s, e_one, e_v] = rc.body_outs[..] else {
            return None;
        };
        match nodes[e_one.index() as usize] {
            ENode::Lit(b) if f64::from_bits(b) == 1.0 => {}
            _ => return None,
        }
        // the one inlined comp the score reads; everything else it touches
        // is an iota or a buffer not varying along the lanes
        let s_vars = vars_reachable(&rc.body, e_s);
        let mut score_arg = None;
        // buffers the score reads along both queries and keys: candidates
        // for a materialized score
        let mut score_bufs = Vec::new();
        for &v in &s_vars {
            let a = &rc.args[v as usize];
            let p = a.src.n();
            match &self.nodes[p].op {
                Op::Comp(_) if self.nodes[p].inline => {
                    if score_arg.is_some() {
                        return None;
                    }
                    score_arg = Some(v as usize);
                }
                Op::Iota(_) => {}
                _ if is_buffer(p) => {
                    let addr = self.flat_addr(a, rank);
                    if addr.coefs[dx] != 0 {
                        return None;
                    }
                    if addr.coefs[dy] != 0 && addr.coefs[jdim] != 0 {
                        score_bufs.push(v as usize);
                    }
                }
                _ => return None,
            }
        }
        let mut score_buf = None;
        if score_arg.is_none() {
            let [v] = score_bufs[..] else {
                return None;
            };
            let a = &rc.args[v];
            score_buf = Some((storage(a.src.n()), self.flat_addr(a, rank)));
            score_arg = Some(v);
        } else if !score_bufs.is_empty() {
            return None;
        }
        let score_arg = score_arg?;
        // the value: buffers/iotas not varying along the query dim
        for &v in &vars_reachable(&rc.body, e_v) {
            let a = &rc.args[v as usize];
            let p = a.src.n();
            let ok = match &self.nodes[p].op {
                Op::Iota(_) => true,
                _ => is_buffer(p),
            };
            if !ok || self.flat_addr(a, rank).coefs[dy] != 0 {
                return None;
            }
        }
        if score_buf.is_some() {
            if epilogue {
                self.epilogue_read(c, red_node)?;
                for a in &c.args {
                    if a.src.n() != red_node && !is_buffer(a.src.n()) {
                        return None;
                    }
                }
            }
            return Some(Flash {
                red_node,
                score_arg,
                score_node: rc.args[score_arg].src.n(),
                terms: Vec::new(),
                score_buf,
                epilogue,
                score_to_red: Vec::new(),
                dy,
                dx,
                others,
                m: pts[dy],
                n_keys: rc.red[0].n(),
                d_lanes,
                kd: 8,
                pts,
            });
        }
        // the score comp: Σ_d of products, each q-like × k-like
        let sa = &rc.args[score_arg];
        let score_node = sa.src.n();
        let sc = self.comp_at(score_node)?;
        if sc.red.len() != 1
            || sc.monoid.as_ref()?.kind() != super::MonoidKind::Sum
            || sc.width() != 1
        {
            return None;
        }
        // score dims ↔ reduction dims: the view is a pick
        let score_to_red: Vec<usize> = sa
            .view
            .forms
            .iter()
            .map(|f| {
                let d = (0..rank).find(|&d| f.coefs[d] != 0)?;
                (f.coefs[d] == 1 && f.offset == 0).then_some(d)
            })
            .collect::<Option<_>>()?;
        if score_to_red.len() != sc.points().len() {
            return None;
        }
        let s_i = score_to_red.iter().position(|&d| d == dy)?;
        let s_j = score_to_red.iter().position(|&d| d == jdim)?;
        let srank = sc.rank();
        let sd = srank - 1;
        let mut pairs = Vec::new();
        let snodes = sc.body.nodes_slice();
        fn products(nodes: &[ENode], e: tang_expr::ExprId, out: &mut Vec<(usize, usize)>) -> bool {
            match nodes[e.index() as usize] {
                ENode::Add(a, b) => products(nodes, a, out) && products(nodes, b, out),
                ENode::Mul(a, b) => match (nodes[a.index() as usize], nodes[b.index() as usize]) {
                    (ENode::Var(p), ENode::Var(q)) => {
                        out.push((p as usize, q as usize));
                        true
                    }
                    _ => false,
                },
                _ => false,
            }
        }
        if !products(snodes, sc.body_outs[0], &mut pairs) {
            return None;
        }
        let mut terms = Vec::new();
        for (p, q) in pairs {
            let (ap, aq) = (&sc.args[p], &sc.args[q]);
            if !is_buffer(ap.src.n()) || !is_buffer(aq.src.n()) {
                return None;
            }
            let (addr_p, addr_q) = (self.flat_addr(ap, srank), self.flat_addr(aq, srank));
            let q_like = |a: &Addr| a.coefs[s_j] == 0 && a.coefs[sd] != 0;
            let k_like = |a: &Addr| a.coefs[s_i] == 0 && a.coefs[sd] != 0;
            if q_like(&addr_p) && k_like(&addr_q) {
                terms.push((storage(ap.src.n()), addr_p, storage(aq.src.n()), addr_q));
            } else if q_like(&addr_q) && k_like(&addr_p) {
                terms.push((storage(aq.src.n()), addr_q, storage(ap.src.n()), addr_p));
            } else {
                return None;
            }
        }
        let kd = sc.red[0].n();
        if kd % 8 != 0 {
            return None;
        }
        // the normalizer reads the reduction point-for-point and nothing
        // but buffers otherwise
        if epilogue {
            self.epilogue_read(c, red_node)?;
            for a in &c.args {
                if a.src.n() != red_node && !is_buffer(a.src.n()) {
                    return None;
                }
            }
        }
        Some(Flash {
            red_node,
            score_arg,
            score_node,
            terms,
            score_buf: None,
            epilogue,
            score_to_red,
            dy,
            dx,
            others,
            m: pts[dy],
            n_keys: rc.red[0].n(),
            d_lanes,
            kd,
            pts,
        })
    }

    /// A lead dim of the flash tile that neither the keys nor anything
    /// else of the reduction but the queries depends on, for
    /// `cuda_flash_kernel` to fold into its rows (grouped-query heads).
    fn flash_fold(&self, f: &Flash, rc: &super::Comp) -> Option<usize> {
        if f.score_buf.is_some() || std::env::var_os("TANG_FLASH_NOFOLD").is_some() {
            return None;
        }
        let rank = rc.rank();
        f.others.iter().copied().rev().find(|&fd| {
            f.pts[fd] > 1
                && f.terms.iter().all(|(_, _, _, ka)| {
                    (0..f.score_to_red.len())
                        .all(|sd| f.score_to_red[sd] != fd || ka.coefs[sd] == 0)
                })
                && rc.args.iter().enumerate().all(|(k, a)| {
                    k == f.score_arg
                        || match &self.nodes[a.src.n()].op {
                            Op::Iota(dd) => a.view.forms[*dd].coefs[fd] == 0,
                            _ => self.flat_addr(a, rank).coefs[fd] == 0,
                        }
                })
        })
    }

    /// CUDA source for `root` as a tensor-core flash-attention tile in
    /// the default shape ([`FLASH_SHAPES`]`[0]`).
    pub fn cuda_flash_kernel(&self, root: usize, reads: &[usize]) -> Option<String> {
        self.cuda_flash_kernel_shaped(root, reads, FLASH_SHAPES[0])
    }

    /// CUDA source for `root` as a flash-attention tile on `mma.sync`
    /// m16n8k8 TF32, or `None` when it is not attention-shaped. Each warp
    /// owns 16 rows (queries × folded heads); per key chunk it computes
    /// `S = Σ_t Q_t·K_tᵀ` into accumulator registers, runs the comp's
    /// mask and the online softmax on them (row statistics reduced
    /// across the fragment's quad), and feeds the same registers as the
    /// A operand of `O += P·V`: the accumulator's column pair (2tq,
    /// 2tq + 1) is the A fragment's k pair (tq, tq + 4) once the keys of
    /// each 8-wide step are permuted, which a reduction allows. O and the
    /// row statistics stay in registers across chunks; K and V come
    /// through a `cp.async` ring of `sh.stages` buffers; `sh.kw` warp
    /// groups split each chunk's keys and merge their states at the end.
    pub fn cuda_flash_kernel_shaped(
        &self,
        root: usize,
        reads: &[usize],
        sh: FlashShape,
    ) -> Option<String> {
        let f = self.flash_tile(root, reads)?;
        let c = self.comp_at(root)?;
        let rc = self.comp_at(f.red_node)?;
        let rank = rc.rank();
        let jdim = rank - 1;
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let sh = match std::env::var("TANG_FLASH_SHAPE") {
            Ok(v) => {
                let p: Vec<usize> = v.split(',').filter_map(|x| x.parse().ok()).collect();
                match p[..] {
                    [bi, bj, stages, kw] => FlashShape { bi, bj, stages, kw },
                    _ => sh,
                }
            }
            Err(_) => sh,
        };
        let (bj, kw) = (sh.bj, sh.kw);
        // keys per pipeline step, over all warp groups
        let bjs = bj * kw;
        if !bj.is_multiple_of(8) || sh.stages < 2 || kw == 0 {
            return None;
        }
        let lead: usize = f.others.iter().map(|&d| f.pts[d]).product();
        // a lead dim the keys and values do not depend on (the query
        // heads sharing a grouped K/V head) folds into the block's rows:
        // K and V are staged once for all of them
        let fold = self.flash_fold(&f, rc);
        let nf = fold.map_or(1, |fd| f.pts[fd]);
        let bi = match std::env::var("TANG_FLASH_BI")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(b) => b,
            None if sh.bi > 0 => sh.bi,
            None if fold.is_some() => {
                // the largest query slice that still fills the device
                let blocks = |b: usize| f.m.div_ceil(b) * lead / nf;
                [64, 32, 16]
                    .into_iter()
                    .find(|&b| blocks(b) >= 82 && nf * b <= 128)
                    .unwrap_or(16)
            }
            None if f.m.div_ceil(64) * lead >= 164 => 64,
            None => 32,
        };
        // BI queries of each of the nf folded heads: R rows, 16 per warp
        #[allow(non_snake_case)]
        let (BI, R) = (bi, nf * bi);
        if !R.is_multiple_of(16) {
            return None;
        }
        let row_warps = R / 16;
        #[allow(non_snake_case)]
        let T = 32 * row_warps * kw;
        if T > 1024 {
            return None;
        }
        let d = f.d_lanes;
        let kd = f.kd;
        let nq = kd / 8;
        let n_terms = f.terms.len();
        let has_k = f.score_buf.is_none();
        // fragments in registers: Q (n_terms × nq × 4), S (bj/8 × 4), O (d/8 × 4)
        if n_terms * nq * 4 + bj / 2 + d / 2 > 160 {
            return None;
        }
        // row strides: K rows are read 8 keys × 4 float2 per half warp
        // (≡ 8 mod 32 is conflict-free), V rows 4 keys × 4 float2 (≡ 4)
        let (ldk, ldv) = (kd + 8, d + 4);
        let k_floats = if has_k { n_terms * bjs * ldk } else { 0 };
        let stage_floats = k_floats + bjs * ldv;
        let q_floats = if has_k { n_terms * R * ldk } else { 0 };
        let ldo = d + 4;
        let merge_floats = if kw > 1 {
            2 * kw * R + (kw - 1) * R * ldo
        } else {
            0
        };
        let smem_floats = (sh.stages * stage_floats).max(q_floats).max(merge_floats);
        if smem_floats * 4 > 99 * 1024 {
            return None;
        }
        let tiles_i = f.m.div_ceil(BI);
        let n_tiles = tiles_i * lead / nf;
        let n_chunks = f.n_keys.div_ceil(bjs);
        let w_out = c.width();

        // coordinate names in the reduction's space
        let coord = |dim: usize| -> String {
            if dim == jdim {
                "(int)jj".into()
            } else if dim == f.dy {
                "(int)ii".into()
            } else if dim == f.dx {
                "(int)nn".into()
            } else {
                format!("(int)p{dim}")
            }
        };
        let addr_expr = |a: &Addr, coord: &dyn Fn(usize) -> String| -> String {
            let mut e = format!("{}", a.offset);
            for (dim, &cf) in a.coefs.iter().enumerate() {
                if cf != 0 {
                    e.push_str(&format!(" + {cf} * {}", coord(dim)));
                }
            }
            e
        };
        // an arg of the reduction as a CUDA expression at the current coords
        let arg_value = |a: &Arg| -> String {
            match &self.nodes[a.src.n()].op {
                Op::Iota(dd) => {
                    let fm = &a.view.forms[*dd];
                    let mut e = format!("{}", fm.offset);
                    for (dim, &cf) in fm.coefs.iter().enumerate() {
                        if cf != 0 {
                            e.push_str(&format!(" + {cf} * {}", coord(dim)));
                        }
                    }
                    format!("(float)({e})")
                }
                _ => format!(
                    "b{}[(unsigned)({})]",
                    storage(a.src.n()),
                    addr_expr(&self.flat_addr(a, rank), &coord)
                ),
            }
        };
        // score-space coordinate names: its point dims map to reduction
        // dims; its reduce dim is the head-dim index `dd`
        let score_coord = |dim: usize| -> String {
            if dim == f.score_to_red.len() {
                "(int)dd".into()
            } else {
                coord(f.score_to_red[dim])
            }
        };
        // a read that is contiguous along `inner` and 16-byte aligned at
        // every vector start (the other coefficients and the offset are
        // multiples of 4), so it stages with 16-byte `cp.async`
        let vec4 = |a: &Addr, inner: usize, extent: usize| {
            a.coefs[inner] == 1
                && a.offset % 4 == 0
                && extent.is_multiple_of(4)
                && a.coefs
                    .iter()
                    .enumerate()
                    .all(|(dim, &cf)| dim == inner || cf % 4 == 0)
        };
        // the row's query and folded head, where per-row code needs them
        let row_coords = |s: &mut String, indent: &str, r: &str| match fold {
            Some(fd) => writeln!(
                s,
                "{indent}unsigned p{fd} = ({r}) / {BI}u; unsigned ii = i0 + ({r}) % {BI}u;"
            )
            .unwrap(),
            None => writeln!(s, "{indent}unsigned ii = i0 + ({r});").unwrap(),
        };
        let emit_body = |s: &mut String,
                         indent: &str,
                         body: &tang_expr::ExprGraph,
                         outs: &[tang_expr::ExprId],
                         prefix: &str,
                         vals: &[String]|
         -> Vec<String> {
            let mut b = String::new();
            let o = emit_scalar(&mut b, body, outs, prefix, &|v| vals[v as usize].clone());
            for line in b.lines() {
                writeln!(
                    s,
                    "{indent}{}",
                    float_literals(&line.replacen("let ", "float ", 1))
                )
                .unwrap();
            }
            o
        };

        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile: flash attention on mma (TF32)"
        )
        .unwrap();
        writeln!(
            s,
            "// {R} rows ({nf} x {BI}), keys {bj} x {kw} groups, {} stages",
            sh.stages
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {n_tiles}").unwrap();
        writeln!(s, "#define TANG_THREADS {T}").unwrap();
        writeln!(s, "#define TANG_SMEM {}", smem_floats * 4).unwrap();
        writeln!(s, "#define select(f, t, c) ((c) ? (t) : (f))").unwrap();
        s.push_str(MMA_PRELUDE);
        let mut params: Vec<String> = reads
            .iter()
            .map(|&j| format!("const float* __restrict__ b{j}"))
            .collect();
        params.push("float* __restrict__ out".into());
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void __launch_bounds__({T}) main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(s, "    (void)count;").unwrap();
        writeln!(s, "    extern __shared__ __align__(16) float smem[];").unwrap();
        // the longest (least masked) row blocks first
        writeln!(s, "    unsigned tile = {}u - blockIdx.x;", n_tiles - 1).unwrap();
        writeln!(s, "    if (blockIdx.x >= {n_tiles}u) return;").unwrap();
        writeln!(s, "    unsigned tile_i = tile % {tiles_i}u;").unwrap();
        writeln!(s, "    unsigned lead = tile / {tiles_i}u;").unwrap();
        for &dim in f.others.iter().rev().filter(|&&dim| Some(dim) != fold) {
            writeln!(s, "    unsigned p{dim} = lead % {}u;", f.pts[dim]).unwrap();
            writeln!(s, "    lead = lead / {}u;", f.pts[dim]).unwrap();
        }
        writeln!(s, "    unsigned i0 = tile_i * {BI}u;").unwrap();
        writeln!(s, "    unsigned tid = threadIdx.x;").unwrap();
        writeln!(s, "    unsigned warp = tid / 32u;").unwrap();
        writeln!(s, "    unsigned lane = tid % 32u;").unwrap();
        writeln!(s, "    unsigned gq = lane / 4u, tq = lane % 4u;").unwrap();
        // warp group (which keys of a chunk) and the warp's first row
        writeln!(s, "    unsigned wg = warp / {row_warps}u;").unwrap();
        writeln!(s, "    unsigned wrow = (warp % {row_warps}u) * 16u;").unwrap();
        writeln!(s, "    unsigned kbase = wg * {bj}u;").unwrap();

        // the key-chunk range: chunks the affine mask covers entirely
        // (block-uniform) are left out at either end
        writeln!(s, "    unsigned c_lo = 0u, c_hi = {n_chunks}u;").unwrap();
        if let Some(fm) = self.score_mask(rc, f.score_arg) {
            let mut e = format!("{}", fm.offset);
            for (dim, &cf) in fm.coefs.iter().enumerate() {
                if cf == 0 {
                    continue;
                }
                let (lo, hi) = if dim == jdim {
                    (
                        "(int)j0".to_string(),
                        format!("(int)min(j0 + {bjs}u, {}u) - 1", f.n_keys),
                    )
                } else if dim == f.dy {
                    (
                        "(int)i0".to_string(),
                        format!("(int)min(i0 + {BI}u, {}u) - 1", f.m),
                    )
                } else if dim == f.dx {
                    ("0".to_string(), format!("{}", d - 1))
                } else if Some(dim) == fold {
                    ("0".to_string(), format!("{}", nf - 1))
                } else {
                    (format!("(int)p{dim}"), format!("(int)p{dim}"))
                };
                let at = if cf > 0 { lo } else { hi };
                e.push_str(&format!(" + {cf} * ({at})"));
            }
            writeln!(s, "    c_lo = {n_chunks}u; c_hi = 0u;").unwrap();
            writeln!(s, "    for (unsigned c = 0u; c < {n_chunks}u; c++) {{").unwrap();
            writeln!(s, "        unsigned j0 = c * {bjs}u;").unwrap();
            writeln!(
                s,
                "        if (!({e} > 0)) {{ c_lo = min(c_lo, c); c_hi = c + 1u; }}"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        }

        // Q fragments, staged once through shared memory and kept as TF32
        // (the head-dim pair 2tq, 2tq + 1 in mma k slots tq, tq + 4)
        if has_k {
            writeln!(s, "    unsigned qf[{n_terms}][{nq}][4];").unwrap();
            for (t, (qb, qa, _, _)) in f.terms.iter().enumerate() {
                writeln!(
                    s,
                    "    for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                    R * kd
                )
                .unwrap();
                writeln!(
                    s,
                    "        unsigned r = idx / {kd}u; unsigned dd = idx % {kd}u;"
                )
                .unwrap();
                row_coords(&mut s, "        ", "r");
                writeln!(
                    s,
                    "        smem[{}u + r * {ldk}u + dd] = ii < {}u ? b{qb}[(unsigned)({})] : 0.0f;",
                    t * R * ldk,
                    f.m,
                    addr_expr(qa, &score_coord)
                )
                .unwrap();
                writeln!(s, "    }}").unwrap();
            }
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int t = 0; t < {n_terms}; t++)").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int q = 0; q < {nq}; q++) {{").unwrap();
            writeln!(
                s,
                "        const float* p = smem + t * {}u + (wrow + gq) * {ldk}u + q * 8u + 2u * tq;",
                R * ldk
            )
            .unwrap();
            writeln!(
                s,
                "        float2 lo = *(const float2*)p, hi = *(const float2*)(p + {}u);",
                8 * ldk
            )
            .unwrap();
            writeln!(
                s,
                "        qf[t][q][0] = tang_tf32(lo.x); qf[t][q][1] = tang_tf32(hi.x);"
            )
            .unwrap();
            writeln!(
                s,
                "        qf[t][q][2] = tang_tf32(lo.y); qf[t][q][3] = tang_tf32(hi.y);"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
        }

        // staging of one key chunk (K terms, then V) into buffer `buf`
        writeln!(s, "    auto load = [&](unsigned buf, unsigned j0) {{").unwrap();
        writeln!(s, "        float* st = smem + buf * {stage_floats}u;").unwrap();
        if has_k {
            for (t, (_, _, kb, ka)) in f.terms.iter().enumerate() {
                let off = t * bjs * ldk;
                if vec4(ka, f.score_to_red.len(), kd) {
                    writeln!(
                        s,
                        "        for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                        bjs * kd / 4
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "            unsigned r = idx / {}u; unsigned dd = (idx % {}u) * 4u; unsigned jj = j0 + r;",
                        kd / 4,
                        kd / 4
                    )
                    .unwrap();
                    writeln!(s, "            bool ok = jj < {}u;", f.n_keys).unwrap();
                    writeln!(
                        s,
                        "            tang_cp16(st + {off}u + r * {ldk}u + dd, b{kb} + (ok ? (unsigned)({}) : 0u), ok);",
                        addr_expr(ka, &score_coord)
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "        for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                        bjs * kd
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "            unsigned r = idx / {kd}u; unsigned dd = idx % {kd}u; unsigned jj = j0 + r;"
                    )
                    .unwrap();
                    writeln!(s, "            bool ok = jj < {}u;", f.n_keys).unwrap();
                    writeln!(
                        s,
                        "            tang_cp4(st + {off}u + r * {ldk}u + dd, b{kb} + (ok ? (unsigned)({}) : 0u), ok);",
                        addr_expr(ka, &score_coord)
                    )
                    .unwrap();
                }
                writeln!(s, "        }}").unwrap();
            }
        }
        {
            // the value: a plain read stages asynchronously; any other
            // expression is evaluated while staging
            let e_v = rc.body_outs[2];
            let plain = match rc.body.nodes_slice()[e_v.index() as usize] {
                ENode::Var(v) => {
                    let a = &rc.args[v as usize];
                    (!matches!(self.nodes[a.src.n()].op, Op::Iota(_)))
                        .then(|| (storage(a.src.n()), self.flat_addr(a, rank)))
                }
                _ => None,
            };
            match plain {
                Some((vb, va)) if vec4(&va, f.dx, d) => {
                    writeln!(
                        s,
                        "        for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                        bjs * d / 4
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "            unsigned r = idx / {}u; unsigned nn = (idx % {}u) * 4u; unsigned jj = j0 + r;",
                        d / 4,
                        d / 4
                    )
                    .unwrap();
                    writeln!(s, "            bool ok = jj < {}u;", f.n_keys).unwrap();
                    writeln!(
                        s,
                        "            tang_cp16(st + {k_floats}u + r * {ldv}u + nn, b{vb} + (ok ? (unsigned)({}) : 0u), ok);",
                        addr_expr(&va, &coord)
                    )
                    .unwrap();
                }
                Some((vb, va)) => {
                    writeln!(
                        s,
                        "        for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                        bjs * d
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "            unsigned r = idx / {d}u; unsigned nn = idx % {d}u; unsigned jj = j0 + r;"
                    )
                    .unwrap();
                    writeln!(s, "            bool ok = jj < {}u;", f.n_keys).unwrap();
                    writeln!(
                        s,
                        "            tang_cp4(st + {k_floats}u + r * {ldv}u + nn, b{vb} + (ok ? (unsigned)({}) : 0u), ok);",
                        addr_expr(&va, &coord)
                    )
                    .unwrap();
                }
                None => {
                    writeln!(
                        s,
                        "        for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                        bjs * d
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "            unsigned r = idx / {d}u; unsigned nn = idx % {d}u; unsigned jj = j0 + r;"
                    )
                    .unwrap();
                    writeln!(s, "            float v = 0.0f;").unwrap();
                    writeln!(s, "            if (jj < {}u) {{", f.n_keys).unwrap();
                    let vals: Vec<String> = rc.args.iter().map(arg_value).collect();
                    let o = emit_body(&mut s, "                ", &rc.body, &[e_v], "v0_", &vals);
                    writeln!(s, "                v = {};", o[0]).unwrap();
                    writeln!(s, "            }}").unwrap();
                    writeln!(s, "            st[{k_floats}u + r * {ldv}u + nn] = v;").unwrap();
                }
            }
            writeln!(s, "        }}").unwrap();
        }
        writeln!(s, "        tang_cp_commit();").unwrap();
        writeln!(s, "    }};").unwrap();

        // per thread: rows wrow + gq and wrow + gq + 8 (h = 0, 1)
        let nsf = bj / 8;
        let nof = d / 8;
        writeln!(s, "    float o[{nof}][4];").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(
            s,
            "    for (int j = 0; j < {nof}; j++) for (int r = 0; r < 4; r++) o[j][r] = 0.0f;"
        )
        .unwrap();
        writeln!(s, "    float m_row[2] = {{-3.0e38f, -3.0e38f}};").unwrap();
        // per-thread partial row sums (the quad's are summed at the end)
        writeln!(s, "    float l_row[2] = {{0.0f, 0.0f}};").unwrap();
        let st = sh.stages;
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(
            s,
            "    for (unsigned c = 0u; c < {}u; c++) {{ if (c_lo + c < c_hi) load((c_lo + c) % {st}u, (c_lo + c) * {bjs}u); else tang_cp_commit(); }}",
            st - 1
        )
        .unwrap();
        writeln!(s, "    for (unsigned c = c_lo; c < c_hi; c++) {{").unwrap();
        writeln!(s, "        tang_cp_wait<{}>();", st - 2).unwrap();
        writeln!(s, "        __syncthreads();").unwrap();
        writeln!(s, "        {{").unwrap();
        writeln!(s, "            unsigned nc = c + {}u;", st - 1).unwrap();
        writeln!(
            s,
            "            if (nc < c_hi) load(nc % {st}u, nc * {bjs}u); else tang_cp_commit();"
        )
        .unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(
            s,
            "        const float* st = smem + (c % {st}u) * {stage_floats}u;"
        )
        .unwrap();
        writeln!(s, "        unsigned j0 = c * {bjs}u + kbase;").unwrap();
        writeln!(s, "        float sc[{nsf}][4];").unwrap();
        if has_k {
            // S = Σ_t Q_t · K_tᵀ: key fragment j holds keys 8j .. 8j + 7
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int j = 0; j < {nsf}; j++) for (int r = 0; r < 4; r++) sc[j][r] = 0.0f;").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int t = 0; t < {n_terms}; t++)").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int q = 0; q < {nq}; q++)").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int j = 0; j < {nsf}; j++) {{").unwrap();
            writeln!(
                s,
                "            float2 kv = *(const float2*)(st + t * {}u + (kbase + 8u * j + gq) * {ldk}u + 8u * q + 2u * tq);",
                bjs * ldk
            )
            .unwrap();
            writeln!(
                s,
                "            unsigned kf[2] = {{tang_tf32(kv.x), tang_tf32(kv.y)}};"
            )
            .unwrap();
            writeln!(s, "            tang_mma(sc[j], qf[t][q], kf);").unwrap();
            writeln!(s, "        }}").unwrap();
        }
        // the comp's score expression (mask) on each element, then the
        // online softmax per row half
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int h = 0; h < 2; h++) {{").unwrap();
        row_coords(&mut s, "            ", "wrow + gq + 8u * h");
        writeln!(s, "            float mloc = -3.0e38f;").unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int j = 0; j < {nsf}; j++)").unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int e = 0; e < 2; e++) {{").unwrap();
        writeln!(
            s,
            "                unsigned jj = j0 + 8u * j + 2u * tq + e;"
        )
        .unwrap();
        if let Some((sb, sa)) = &f.score_buf {
            writeln!(
                s,
                "                float sval = (ii < {}u && jj < {}u) ? b{sb}[(unsigned)({})] : 0.0f;",
                f.m,
                f.n_keys,
                addr_expr(sa, &coord)
            )
            .unwrap();
        } else {
            writeln!(s, "                float sval = sc[j][2 * h + e];").unwrap();
        }
        {
            let vals: Vec<String> = rc
                .args
                .iter()
                .enumerate()
                .map(|(k, a)| {
                    if k == f.score_arg {
                        "sval".to_string()
                    } else {
                        arg_value(a)
                    }
                })
                .collect();
            let o = emit_body(
                &mut s,
                "                ",
                &rc.body,
                &[rc.body_outs[0]],
                "s0_",
                &vals,
            );
            writeln!(
                s,
                "                float v = (jj < {}u) ? ({}) : -3.0e38f;",
                f.n_keys, o[0]
            )
            .unwrap();
        }
        writeln!(s, "                sc[j][2 * h + e] = v;").unwrap();
        writeln!(s, "                mloc = fmaxf(mloc, v);").unwrap();
        writeln!(s, "            }}").unwrap();
        writeln!(
            s,
            "            mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, 1));"
        )
        .unwrap();
        writeln!(
            s,
            "            mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, 2));"
        )
        .unwrap();
        writeln!(s, "            float m_new = fmaxf(m_row[h], mloc);").unwrap();
        writeln!(s, "            float alpha = expf(m_row[h] - m_new);").unwrap();
        writeln!(s, "            float lsum = 0.0f;").unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int j = 0; j < {nsf}; j++)").unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int e = 0; e < 2; e++) {{").unwrap();
        writeln!(
            s,
            "                float p = expf(sc[j][2 * h + e] - m_new); sc[j][2 * h + e] = p; lsum += p;"
        )
        .unwrap();
        writeln!(s, "            }}").unwrap();
        writeln!(
            s,
            "            l_row[h] = l_row[h] * alpha + lsum; m_row[h] = m_new;"
        )
        .unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(
            s,
            "            for (int j = 0; j < {nof}; j++) {{ o[j][2 * h] *= alpha; o[j][2 * h + 1] *= alpha; }}"
        )
        .unwrap();
        writeln!(s, "        }}").unwrap();
        // O += P · V, P straight from the score registers
        writeln!(s, "        #pragma unroll").unwrap();
        writeln!(s, "        for (int j = 0; j < {nsf}; j++) {{").unwrap();
        writeln!(
            s,
            "            unsigned pf[4] = {{tang_tf32(sc[j][0]), tang_tf32(sc[j][2]), tang_tf32(sc[j][1]), tang_tf32(sc[j][3])}};"
        )
        .unwrap();
        writeln!(s, "            #pragma unroll").unwrap();
        writeln!(s, "            for (int jp = 0; jp < {}; jp++) {{", d / 16).unwrap();
        writeln!(
            s,
            "                const float* vp = st + {k_floats}u + (kbase + 8u * j + 2u * tq) * {ldv}u + 16u * jp + 2u * gq;"
        )
        .unwrap();
        writeln!(
            s,
            "                float2 r0 = *(const float2*)vp, r1 = *(const float2*)(vp + {ldv}u);"
        )
        .unwrap();
        writeln!(
            s,
            "                unsigned v0[2] = {{tang_tf32(r0.x), tang_tf32(r1.x)}}, v1[2] = {{tang_tf32(r0.y), tang_tf32(r1.y)}};"
        )
        .unwrap();
        writeln!(
            s,
            "                tang_mma(o[2 * jp], pf, v0); tang_mma(o[2 * jp + 1], pf, v1);"
        )
        .unwrap();
        writeln!(s, "            }}").unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    tang_cp_wait<0>();").unwrap();
        // the quad's partial row sums
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int h = 0; h < 2; h++) {{").unwrap();
        writeln!(
            s,
            "        l_row[h] += __shfl_xor_sync(0xffffffffu, l_row[h], 1);"
        )
        .unwrap();
        writeln!(
            s,
            "        l_row[h] += __shfl_xor_sync(0xffffffffu, l_row[h], 2);"
        )
        .unwrap();
        writeln!(s, "    }}").unwrap();
        // element r of O fragment j: row gq + 8·(r/2), lane (the V column
        // pairing above) 16·(j/2) + 4·tq + 2·(r%2) + j%2
        let frag_loop = |s: &mut String| {
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int j = 0; j < {nof}; j++)").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int r = 0; r < 4; r++) {{").unwrap();
            writeln!(s, "        unsigned lr = wrow + gq + 8u * (r / 2);").unwrap();
            writeln!(
                s,
                "        unsigned lc = 16u * (j / 2) + 4u * tq + 2u * (r % 2) + (j % 2);"
            )
            .unwrap();
        };
        if kw > 1 {
            // groups 1.. hand their (m, l, O) to group 0 through shared memory
            let (mo, lo, oo) = (0, kw * R, 2 * kw * R);
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    if (gq < 8u && tq == 0u) {{").unwrap();
            writeln!(s, "        for (int h = 0; h < 2; h++) {{").unwrap();
            writeln!(s, "            unsigned lr = wrow + gq + 8u * h;").unwrap();
            writeln!(s, "            smem[{mo}u + wg * {R}u + lr] = m_row[h];").unwrap();
            writeln!(s, "            smem[{lo}u + wg * {R}u + lr] = l_row[h];").unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    if (wg > 0u) {{").unwrap();
            frag_loop(&mut s);
            writeln!(
                s,
                "        smem[{oo}u + ((wg - 1u) * {R}u + lr) * {ldo}u + lc] = o[j][r];"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    if (wg > 0u) return;").unwrap();
            writeln!(s, "    float sc_g[{kw}][2];").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int h = 0; h < 2; h++) {{").unwrap();
            writeln!(s, "        unsigned lr = wrow + gq + 8u * h;").unwrap();
            writeln!(s, "        float mx = m_row[h];").unwrap();
            writeln!(
                s,
                "        for (int g = 1; g < {kw}; g++) mx = fmaxf(mx, smem[{mo}u + g * {R}u + lr]);"
            )
            .unwrap();
            writeln!(s, "        float l = 0.0f;").unwrap();
            writeln!(s, "        for (int g = 0; g < {kw}; g++) {{").unwrap();
            writeln!(
                s,
                "            float mg = g == 0 ? m_row[h] : smem[{mo}u + g * {R}u + lr];"
            )
            .unwrap();
            writeln!(
                s,
                "            float lg = g == 0 ? l_row[h] : smem[{lo}u + g * {R}u + lr];"
            )
            .unwrap();
            writeln!(
                s,
                "            sc_g[g][h] = expf(mg - mx); l += lg * sc_g[g][h];"
            )
            .unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        m_row[h] = mx; l_row[h] = l;").unwrap();
            writeln!(s, "    }}").unwrap();
            frag_loop(&mut s);
            writeln!(s, "        float acc = o[j][r] * sc_g[0][r / 2];").unwrap();
            writeln!(
                s,
                "        for (int g = 1; g < {kw}; g++) acc += smem[{oo}u + ((g - 1) * {R}u + lr) * {ldo}u + lc] * sc_g[g][r / 2];"
            )
            .unwrap();
            writeln!(s, "        o[j][r] = acc;").unwrap();
            writeln!(s, "    }}").unwrap();
        }
        // the output per (row, lane): the state, or the normalizer
        frag_loop(&mut s);
        writeln!(s, "        unsigned nn = lc;").unwrap();
        row_coords(&mut s, "        ", "lr");
        writeln!(
            s,
            "        float m_v = m_row[r / 2], l_v = l_row[r / 2], acc_v = o[j][r];"
        )
        .unwrap();
        writeln!(s, "        if (ii < {}u) {{", f.m).unwrap();
        if !f.epilogue {
            // the root is the reduction: its (m, l, acc) state per point
            let mut flat = String::from("0u");
            for (dd, &np) in f.pts.iter().enumerate() {
                flat = format!("({flat}) * {np}u + (unsigned)({})", coord(dd));
            }
            writeln!(s, "            unsigned g = {flat};").unwrap();
            writeln!(s, "            out[g * {w_out}u + 0u] = m_v;").unwrap();
            writeln!(s, "            out[g * {w_out}u + 1u] = l_v;").unwrap();
            writeln!(s, "            out[g * {w_out}u + 2u] = acc_v;").unwrap();
        } else {
            // the normalizer at (ii, nn), written in root's layout
            let ep = self.epilogue_read(c, f.red_node)?;
            let root_rank = c.points().len();
            let mut root_coord: Vec<Option<String>> = vec![None; root_rank];
            for (rdim, fm) in ep.iter().enumerate() {
                if let Some(dd) = (0..root_rank).find(|&dd| fm.coefs[dd] != 0) {
                    root_coord[dd] = Some(coord(rdim));
                }
            }
            let root_coord: Vec<String> = root_coord.into_iter().collect::<Option<_>>()?;
            let rcoord = |dim: usize| root_coord[dim].clone();
            let vals: Vec<String> = c
                .args
                .iter()
                .map(|a| {
                    if a.src.n() == f.red_node {
                        match self.epilogue_slot(a) {
                            0 => "m_v".to_string(),
                            1 => "l_v".to_string(),
                            _ => "acc_v".to_string(),
                        }
                    } else {
                        let addr = self.flat_addr(a, root_rank);
                        format!(
                            "b{}[(unsigned)({})]",
                            storage(a.src.n()),
                            addr_expr(&addr, &rcoord)
                        )
                    }
                })
                .collect();
            let outs = emit_body(&mut s, "            ", &c.body, &c.body_outs, "e0_", &vals);
            let pts_root = super::fixed(c.points());
            let mut flat = String::from("0u");
            for (dd, &np) in pts_root.iter().enumerate() {
                flat = format!("({flat}) * {np}u + (unsigned)({})", root_coord[dd]);
            }
            writeln!(s, "            unsigned g = {flat};").unwrap();
            for (slot, v) in outs.iter().enumerate() {
                writeln!(s, "            out[g * {w_out}u + {slot}u] = {v};").unwrap();
            }
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        let _ = f.score_node;
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::float_literals;

    #[test]
    fn literals_stay_single_precision() {
        assert_eq!(float_literals("auto x = -1e30;"), "auto x = -1e30f;");
        assert_eq!(
            float_literals("let y = 1.5 * e4 + 2u;"),
            "let y = 1.5f * e4 + 2u;"
        );
        assert_eq!(float_literals("c0_14 = 1.442695;"), "c0_14 = 1.442695f;");
        assert_eq!(
            float_literals("v[3] = 3.4028235e38"),
            "v[3] = 3.4028235e38f"
        );
        assert_eq!(
            float_literals("stage_0[idx * 64u + 1u]"),
            "stage_0[idx * 64u + 1u]"
        );
    }
}
