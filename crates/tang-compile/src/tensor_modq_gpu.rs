//! A [`ModGraph`] as batched GPU kernels, in Metal and CUDA from one source.
//!
//! Every materialized node is one kernel over `batch × size` threads: thread
//! `g` computes point `g % size` of instance `g / size`. Buffers are laid out
//! `[batch][size]`; constants have no batch dimension. Map bodies are the
//! same branch-free code the C lowering emits ([`super::c`]); only the field
//! helpers change, using the GPU's 64×64 multiply-high instead of
//! `__int128`. All kernels of a run go into one command buffer (Metal) or
//! one stream (CUDA), in graph order.
//!
//! Constant time on a GPU is a weaker statement than on a CPU core — other
//! work shares the device — but the bodies keep the same property: no
//! value-dependent branch, address or divide. Filter, the one sequential
//! node, runs one thread per instance and only on public data.
//!
//! Metal is built and run here by [`ModGraph::compile_metal`]: a generated
//! Objective-C host compiles the shader source at run time, so no Metal
//! toolchain is needed. [`ModGraph::to_cuda`] emits a `.cu` translation unit
//! with the same host protocol; [`ModGraph::compile_cuda`] builds it with
//! `nvcc` where one exists.

use super::c::{le_bytes, op_c};
use super::{MOp, ModGraph, ModId};
use std::fmt::Write as _;
use std::io;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gpu {
    Metal,
    Cuda,
}

const HELPERS: &str = r#"
DEV uint32_t ct_csub(uint32_t x) { uint32_t y = x - Q; uint32_t m = 0u - (y >> 31); return y + (Q & m); }
DEV uint32_t ct_add(uint32_t a, uint32_t b) { return ct_csub(a + b); }
DEV uint32_t ct_sub(uint32_t a, uint32_t b) { return ct_csub(a + Q - b); }
DEV uint32_t ct_mont(uint32_t a, uint32_t b) {
    uint64_t t = (uint64_t)a * b; uint32_t m = (uint32_t)t * QNEG_INV;
    return ct_csub((uint32_t)((t + (uint64_t)m * Q) >> 32));
}
DEV uint32_t ct_mul(uint32_t a, uint32_t b) { return ct_mont(ct_mont(a, b), R2); }
DEV uint64_t ct_div_q(uint64_t v) {
    uint64_t est = MULHI64(v, (uint64_t)BARRETT);
    uint64_t r = v - est * Q;
    return est + ((((uint64_t)Q - 1 - r) >> 63) & 1);
}
DEV uint32_t ct_compress(uint32_t x, uint32_t d) {
    return (uint32_t)ct_div_q(((uint64_t)x << d) + (Q >> 1)) & ((1u << d) - 1);
}
DEV uint32_t ct_decompress(uint32_t y, uint32_t d) {
    return (uint32_t)(((uint64_t)y * Q + (((uint64_t)1 << d) >> 1)) >> d);
}
DEV uint32_t ct_p2r(uint32_t x, uint32_t d) { return (x + ((1u << d) >> 1) - 1) >> d; }
DEV uint32_t ct_sel(uint32_t c, uint32_t a, uint32_t b) { uint32_t m = 0u - (c & 1); return (a & m) | (b & ~m); }
DEV uint32_t ct_eq(uint32_t a, uint32_t b) { uint32_t d = a ^ b; return ((d | (0u - d)) >> 31) ^ 1; }
DEV uint32_t ct_high(uint32_t r, uint64_t g2, uint64_t m) {
    uint64_t alpha = 2 * g2;
    uint64_t r1 = ((uint64_t)r * m) >> 40;
    uint64_t rem = (uint64_t)r - r1 * alpha;
    r1 += ((alpha - 1 - rem) >> 63) & 1;
    uint64_t rp = (uint64_t)r - r1 * alpha;
    r1 += ((g2 - rp) >> 63) & 1;
    uint32_t d = (uint32_t)(r1 * alpha) ^ (Q - 1);
    uint32_t edge = ((d | (0u - d)) >> 31) ^ 1;
    return (uint32_t)r1 & ~(0u - edge);
}
DEV uint32_t ct_low(uint32_t r, uint64_t g2, uint64_t m) {
    uint64_t alpha = 2 * g2;
    uint64_t r1 = ((uint64_t)r * m) >> 40;
    uint64_t rem = (uint64_t)r - r1 * alpha;
    r1 += ((alpha - 1 - rem) >> 63) & 1;
    uint64_t rp = (uint64_t)r - r1 * alpha;
    uint64_t up = ((g2 - rp) >> 63) & 1;
    r1 += up;
    int64_t r0 = (int64_t)rp - (int64_t)(up * alpha);
    uint32_t d = (uint32_t)(r1 * alpha) ^ (Q - 1);
    uint32_t edge = ((d | (0u - d)) >> 31) ^ 1;
    return ct_csub((uint32_t)(r0 - (int64_t)edge + (int64_t)Q));
}
"#;

/// How one kernel binds: its node, its argument buffers, and its kind.
struct Launch {
    kernel: usize,
    /// dependency level: launches on one level run in one dispatch
    level: usize,
    /// offsets the kernel reads: its own data, its sources, its views
    params: Vec<u64>,
    /// threads per instance (the node's size), or 1 for a filter
    per: usize,
}

impl ModGraph {
    /// The buffer a node reads from: aliases (reshape, declassify) resolved.
    fn storage(&self, mut i: usize) -> usize {
        loop {
            match &self.nodes[i].op {
                MOp::Reshape(s) | MOp::Declassify { src: s, .. } => i = s.0 as usize,
                _ => return i,
            }
        }
    }

    fn is_const(&self, i: usize) -> bool {
        matches!(self.nodes[self.storage(i)].op, MOp::Const(_))
    }

    /// Each node's level: 0 for inputs and constants, else one past the
    /// deepest node it reads. Nodes on one level can run in one dispatch.
    pub fn levels(&self) -> Vec<usize> {
        let mut lv = vec![0usize; self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            let srcs: Vec<usize> = match &n.op {
                MOp::Map { args, .. } => args.iter().map(|(a, _)| a.0 as usize).collect(),
                MOp::Gather { table, index } => vec![table.0 as usize, index.0 as usize],
                MOp::Filter { src, .. } => vec![src.0 as usize],
                MOp::Reshape(s) | MOp::Declassify { src: s, .. } => {
                    lv[i] = lv[s.0 as usize];
                    continue;
                }
                _ => continue,
            };
            lv[i] = 1 + srcs.iter().map(|&s| lv[s]).max().unwrap_or(0);
        }
        lv
    }

    /// Arena offsets: a data node's region starts at `batch · data[i]`, a
    /// constant's at `consts[i]` in the constants arena. Aliases get none.
    fn arena(&self) -> (Vec<u64>, Vec<u64>, u64, u64) {
        let (mut data, mut consts) = (vec![0u64; self.nodes.len()], vec![0u64; self.nodes.len()]);
        let (mut d, mut c) = (0u64, 0u64);
        for (i, n) in self.nodes.iter().enumerate() {
            let size = n.shape.iter().product::<usize>() as u64;
            match n.op {
                MOp::Reshape(_) | MOp::Declassify { .. } => {}
                MOp::Const(_) => {
                    consts[i] = c;
                    c += size;
                }
                _ => {
                    data[i] = d;
                    d += size;
                }
            }
        }
        (data, consts, d, c)
    }

    /// Device code: helpers, then one kernel per *distinct* node shape. Every
    /// offset — where a node's data, a constant table, or a view starts — is a
    /// runtime parameter, so the 24 rounds of a Keccak, or two NTTs, share
    /// one kernel; launches differ only in their parameter slices.
    fn gpu_kernels(&self, dialect: Gpu) -> (String, Vec<Launch>, usize) {
        let f = self.f;
        let mut s = String::new();
        match dialect {
            Gpu::Metal => s.push_str(
                "#include <metal_stdlib>\nusing namespace metal;\n#define DEV static inline\n#define MULHI64(a, b) mulhi((uint64_t)(a), (uint64_t)(b))\n",
            ),
            Gpu::Cuda => s.push_str(
                "#include <stdint.h>\n#define DEV __device__ __forceinline__\n#define MULHI64(a, b) __umul64hi((unsigned long long)(a), (unsigned long long)(b))\n",
            ),
        }
        let _ = writeln!(
            s,
            "#define Q {}u\n#define QNEG_INV {}u\n#define R2 {}u\n#define BARRETT {}ull{HELPERS}",
            f.q, f.qneg_inv, f.r2, f.barrett
        );
        // each distinct node shape is a device function; one `uber` kernel
        // runs every node of a level, routing each thread to its node
        let (dev_c, dev_d) = match dialect {
            Gpu::Metal => ("device const", "device"),
            Gpu::Cuda => ("const", ""),
        };
        let head = format!(
            "DEV void kf@K@(uint32_t g, {dev_c} uint32_t *cst, {dev_d} uint32_t *arena, uint32_t batch, {dev_c} uint64_t *P)"
        );
        let gid = "";
        let (data_off, const_off, _, _) = self.arena();
        let levels = self.levels();
        let mut kernels: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut launches = Vec::new();
        for (i, n) in self.nodes.iter().enumerate() {
            let size: usize = n.shape.iter().product();
            let srcs: Vec<usize> = match &n.op {
                MOp::Map { args, .. } => args
                    .iter()
                    .map(|(a, _)| self.storage(a.0 as usize))
                    .collect(),
                MOp::Gather { table, index } => {
                    vec![
                        self.storage(table.0 as usize),
                        self.storage(index.0 as usize),
                    ]
                }
                MOp::Filter { src, .. } => vec![self.storage(src.0 as usize)],
                _ => continue,
            };
            let per = if matches!(n.op, MOp::Filter { .. }) {
                1
            } else {
                size
            };
            let mut params: Vec<u64> = vec![data_off[i]];
            let mut param = |v: u64| {
                params.push(v);
                format!("P[{}]", params.len() - 1)
            };
            let mut k = String::new();
            // where source `src`'s element `flat` of instance b lives
            let at = |src: usize, flat: &str, param: &mut dyn FnMut(u64) -> String| -> String {
                if self.is_const(src) {
                    format!("cst[{} + (uint64_t)({flat})]", param(const_off[src]))
                } else {
                    let sz: usize = self.nodes[src].shape.iter().product();
                    format!(
                        "arena[(uint64_t)batch * {} + (uint64_t)b * {sz}ull + (uint64_t)({flat})]",
                        param(data_off[src])
                    )
                }
            };
            let dst = "arena[(uint64_t)batch * P[0] + (uint64_t)g]";
            match &n.op {
                MOp::Map { args, body, out } => {
                    let bparams: Vec<String> = (0..args.len())
                        .map(|a| format!("uint32_t a{a}"))
                        .chain((0..n.shape.len()).map(|d| format!("uint32_t c{d}")))
                        .collect();
                    let _ = writeln!(k, "DEV uint32_t body_@K@({}) {{", bparams.join(", "));
                    for (j, op) in body.ops.iter().enumerate() {
                        op_c(op, &mut k, j);
                    }
                    let _ = writeln!(k, "    return r{out};\n}}");
                    let _ = writeln!(k, "{head} {{\n{gid}  if (g >= batch * {size}u) return;\n  uint32_t b = g / {size}u, p = g % {size}u, rest = p;");
                    for d in (0..n.shape.len()).rev() {
                        let _ = writeln!(
                            k,
                            "  int64_t i{d} = rest % {len}u; rest /= {len}u;",
                            len = n.shape[d]
                        );
                    }
                    for (a, (src, view)) in args.iter().enumerate() {
                        let shape = self.shape(*src);
                        let coords: Vec<String> = view
                            .forms
                            .iter()
                            .map(|form| {
                                let mut e = if form.offset == 0 {
                                    "0".to_string()
                                } else {
                                    format!("(int64_t){}", param(form.offset as u64))
                                };
                                for (d, c) in form.coefs.iter().enumerate() {
                                    if *c != 0 {
                                        let _ = write!(e, " + {c} * i{d}");
                                    }
                                }
                                format!("((int64_t)({e}))")
                            })
                            .collect();
                        let mut flat = "(int64_t)0".to_string();
                        for (c, len) in coords.iter().zip(shape) {
                            flat = format!("({flat}) * {len} + {c}");
                        }
                        let read = at(srcs[a], &flat, &mut param);
                        if view.guard {
                            let inside: Vec<String> = coords
                                .iter()
                                .zip(shape)
                                .map(|(c, len)| format!("{c} >= 0 && {c} < {len}"))
                                .collect();
                            let _ = writeln!(
                                k,
                                "  uint32_t a{a} = ({}) ? {read} : 0u;",
                                inside.join(" && ")
                            );
                        } else {
                            let _ = writeln!(k, "  uint32_t a{a} = {read};");
                        }
                    }
                    let call: Vec<String> = (0..args.len())
                        .map(|a| format!("a{a}"))
                        .chain((0..n.shape.len()).map(|d| format!("(uint32_t)i{d}")))
                        .collect();
                    let _ = writeln!(k, "  {dst} = body_@K@({});\n}}", call.join(", "));
                }
                MOp::Gather { table, .. } => {
                    let tl: usize = self.shape(*table).iter().product();
                    let idx = at(srcs[1], "p", &mut param);
                    let tab = at(srcs[0], &format!("{idx} % {tl}u"), &mut param);
                    let _ = writeln!(
                        k,
                        "{head} {{\n{gid}  if (g >= batch * {size}u) return;\n  uint32_t b = g / {size}u, p = g % {size}u;\n  {dst} = {tab};\n}}"
                    );
                }
                MOp::Filter { bound, .. } => {
                    let x = at(srcs[0], "j", &mut param);
                    let _ = writeln!(
                        k,
                        "{head} {{\n{gid}  if (g >= batch) return;\n  uint32_t b = g, w = 0;\n  uint64_t o0 = (uint64_t)batch * P[0] + (uint64_t)b * {size}ull;\n  for (uint32_t j = 0; j < {size}u; j++) {{ uint32_t x = {x}; if (x < {bound}u) arena[o0 + w++] = x; }}\n  while (w < {size}u) arena[o0 + w++] = 0;\n}}"
                    );
                }
                _ => unreachable!(),
            }
            let next = kernels.len();
            let id = *kernels.entry(k.clone()).or_insert_with(|| {
                s.push_str(&k.replace("@K@", &next.to_string()));
                next
            });
            launches.push(Launch {
                kernel: id,
                level: levels[i],
                params,
                per,
            });
        }
        let count = kernels.len();
        // one entry point per distinct kernel: it runs every launch of that
        // kernel on a level, finding its own launch by binary search over the
        // launches' first threads (public), so no switch, no shared registers
        for k in 0..count {
            let route = format!(
                "  uint32_t lo = 0, hi = n;\n  while (hi - lo > 1) {{ uint32_t mid = (lo + hi) / 2; if (LV[3 * mid + 1] <= g) lo = mid; else hi = mid; }}\n  kf{k}(g - LV[3 * lo + 1], cst, arena, batch, P + LV[3 * lo + 2]);\n}}\n"
            );
            match dialect {
                Gpu::Metal => {
                    let _ = write!(
                        s,
                        "kernel void k{k}(device const uint32_t *cst [[buffer(0)]], device uint32_t *arena [[buffer(1)]], constant uint32_t &batch [[buffer(2)]], device const uint64_t *P [[buffer(3)]], device const uint32_t *LV [[buffer(4)]], constant uint32_t &n [[buffer(5)]], uint g [[thread_position_in_grid]]) {{\n{route}"
                    );
                }
                Gpu::Cuda => {
                    let _ = write!(
                        s,
                        "extern \"C\" __global__ void k{k}(const uint32_t *cst, uint32_t *arena, uint32_t batch, const uint64_t *P, const uint32_t *LV, uint32_t n, uint32_t total) {{\n  uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;\n  if (g >= total) return;\n{route}"
                    );
                }
            }
        }
        (s, launches, count)
    }

    /// The tables a host needs: node sizes, which nodes are inputs or
    /// constants, and the launch list.
    fn host_tables(&self, outputs: &[ModId], launches: &[Launch]) -> String {
        let mut s = String::new();
        let sizes: Vec<String> = self
            .nodes
            .iter()
            .map(|n| n.shape.iter().product::<usize>().to_string())
            .collect();
        let kind: Vec<String> = self
            .nodes
            .iter()
            .map(|n| match n.op {
                MOp::Input(k, _) => format!("{}", 1 + k), // 1 + input index
                MOp::Const(_) => "-1".into(),
                MOp::Reshape(_) | MOp::Declassify { .. } => "-2".into(),
                _ => "0".into(),
            })
            .collect();
        let mut consts = String::new();
        for (i, n) in self.nodes.iter().enumerate() {
            if let MOp::Const(c) = &n.op {
                let v: Vec<String> = c.iter().map(|x| format!("{x}u")).collect();
                let _ = writeln!(
                    consts,
                    "static const uint32_t K{i}[] = {{{}}};",
                    v.join(",")
                );
            }
        }
        let const_ptrs: Vec<String> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                if matches!(n.op, MOp::Const(_)) {
                    format!("K{i}")
                } else {
                    "0".into()
                }
            })
            .collect();
        let _ = writeln!(s, "{consts}");
        let _ = writeln!(
            s,
            "#define NODES {}\nstatic const size_t SIZE[] = {{{}}};",
            self.nodes.len(),
            sizes.join(",")
        );
        let _ = writeln!(s, "static const int KIND[] = {{{}}};", kind.join(","));
        let _ = writeln!(
            s,
            "static const uint32_t *CONSTS[] = {{{}}};",
            const_ptrs.join(",")
        );
        let storage: Vec<String> = (0..self.nodes.len())
            .map(|i| self.storage(i).to_string())
            .collect();
        let _ = writeln!(s, "static const int STORAGE[] = {{{}}};", storage.join(","));
        let (data, consts_off, dt, ct) = self.arena();
        let join = |v: &[u64]| {
            v.iter()
                .map(|x| format!("{x}ull"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let _ = writeln!(
            s,
            "static const uint64_t DATA_OFF[] = {{{}}};\nstatic const uint64_t CONST_OFF[] = {{{}}};\n#define DATA_TOTAL {dt}ull\n#define CONST_TOTAL {ct}ull",
            join(&data),
            join(&consts_off)
        );
        let mut params = Vec::new();
        let mut off = Vec::new();
        for l in launches {
            off.push(params.len().to_string());
            params.extend(l.params.iter().map(|x| format!("{x}ull")));
        }
        let _ = writeln!(
            s,
            "#define LEVELS {}\nstatic const int L_LEVEL[] = {{{}}};\n#define LAUNCHES {}\nstatic const int L_KERNEL[] = {{{}}};\nstatic const size_t L_PER[] = {{{}}};\nstatic const size_t L_POFF[] = {{{}}};\n#define N_PARAMS {}\nstatic const uint64_t PARAMS[] = {{{}}};",
            launches.iter().map(|l| l.level).max().unwrap_or(0) + 1,
            launches.iter().map(|l| l.level.to_string()).collect::<Vec<_>>().join(","),
            launches.len(),
            launches.iter().map(|l| l.kernel.to_string()).collect::<Vec<_>>().join(","),
            launches.iter().map(|l| l.per.to_string()).collect::<Vec<_>>().join(","),
            off.join(","),
            params.len(),
            if params.is_empty() { "0".into() } else { params.join(",") }
        );
        let outs: Vec<String> = outputs
            .iter()
            .map(|o| self.storage(o.0 as usize).to_string())
            .collect();
        let _ = writeln!(
            s,
            "#define N_IN {}\n#define N_OUT {}\nstatic const int OUTS[] = {{{}}};",
            self.inputs,
            outputs.len(),
            outs.join(",")
        );
        s
    }

    /// The Metal shader source and its Objective-C host.
    pub fn to_metal(&self, outputs: &[ModId]) -> (String, String) {
        let (msl, launches, kernels) = self.gpu_kernels(Gpu::Metal);
        let tables = self.host_tables(outputs, &launches);
        let host = format!(
            "#include <stdint.h>\n#include <stddef.h>\n#include <string.h>\n{tables}#define KERNELS_N {kernels}\n{METAL_HOST}"
        );
        (msl, host)
    }

    /// One CUDA translation unit: kernels and a host `main` with the same
    /// stdin/stdout protocol as the Metal and C runners.
    pub fn to_cuda(&self, outputs: &[ModId]) -> String {
        let (cu, launches, kernels) = self.gpu_kernels(Gpu::Cuda);
        let mut table = String::from("static void *KERNELS[] = {");
        for k in 0..kernels {
            let _ = write!(table, "(void *)k{k},");
        }
        table.push_str("};\n");
        format!(
            "{cu}\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n{}{table}{CUDA_HOST}",
            self.host_tables(outputs, &launches)
        )
    }

    /// Build the Metal runner (clang, Foundation + Metal frameworks).
    pub fn compile_metal(&self, outputs: &[ModId]) -> io::Result<GpuCompiled> {
        let dir = scratch("metal")?;
        let (msl, host) = self.to_metal(outputs);
        std::fs::write(dir.join("graph.metal"), msl)?;
        std::fs::write(dir.join("host.m"), host)?;
        let exe = dir.join("graph");
        let st = Command::new("clang")
            .args([
                "-O2",
                "-fobjc-arc",
                "-w",
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
                "-o",
            ])
            .arg(&exe)
            .arg(dir.join("host.m"))
            .status()?;
        if !st.success() {
            return Err(io::Error::other("clang failed on the Metal host"));
        }
        self.gpu_compiled(dir, exe, vec![dir_arg("graph.metal")], outputs)
    }

    /// Build the CUDA runner with `nvcc` (`$NVCC`, else `nvcc`).
    pub fn compile_cuda(&self, outputs: &[ModId]) -> io::Result<GpuCompiled> {
        let dir = scratch("cuda")?;
        std::fs::write(dir.join("graph.cu"), self.to_cuda(outputs))?;
        let exe = dir.join("graph");
        let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
        let st = Command::new(nvcc)
            .args(["-O2", "-w", "-o"])
            .arg(&exe)
            .arg(dir.join("graph.cu"))
            .status()?;
        if !st.success() {
            return Err(io::Error::other("nvcc failed"));
        }
        self.gpu_compiled(dir, exe, vec![], outputs)
    }

    fn gpu_compiled(
        &self,
        dir: PathBuf,
        exe: PathBuf,
        extra: Vec<String>,
        outputs: &[ModId],
    ) -> io::Result<GpuCompiled> {
        let mut in_sizes = vec![0; self.inputs];
        for n in &self.nodes {
            if let MOp::Input(k, _) = n.op {
                in_sizes[k] = n.shape.iter().product();
            }
        }
        let out_sizes = outputs
            .iter()
            .map(|o| self.shape(*o).iter().product())
            .collect();
        Ok(GpuCompiled {
            dir,
            exe,
            extra,
            in_sizes,
            out_sizes,
        })
    }
}

fn dir_arg(name: &str) -> String {
    name.to_string()
}

fn scratch(kind: &str) -> io::Result<PathBuf> {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "tang-modq-{kind}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// A built GPU runner.
pub struct GpuCompiled {
    pub dir: PathBuf,
    exe: PathBuf,
    /// files the runner reads, relative to `dir` (the Metal source)
    extra: Vec<String>,
    in_sizes: Vec<usize>,
    out_sizes: Vec<usize>,
}

impl Drop for GpuCompiled {
    fn drop(&mut self) {
        if std::env::var_os("TANG_MODQ_KEEP").is_none() {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

impl GpuCompiled {
    /// Run a batch: `sets[b]` is instance b's inputs. Returns each
    /// instance's outputs and the device time of the whole batch in ns.
    pub fn run(&self, sets: &[Vec<Vec<u32>>]) -> io::Result<(Vec<Vec<Vec<u32>>>, u64)> {
        use std::io::Write;
        for set in sets {
            assert_eq!(
                set.iter().map(Vec::len).collect::<Vec<_>>(),
                self.in_sizes,
                "input shapes"
            );
        }
        let mut child = Command::new(&self.exe)
            .arg(sets.len().to_string())
            .args(self.extra.iter().map(|e| self.dir.join(e)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let bytes: Vec<u8> = sets.iter().flat_map(|s| le_bytes(s)).collect();
        // a runner that fails early closes stdin; its stderr says why
        let _ = child.stdin.take().expect("stdin").write_all(&bytes);
        let out = child.wait_with_output()?;
        if !out.status.success() {
            return Err(io::Error::other(
                String::from_utf8_lossy(&out.stderr).into_owned(),
            ));
        }
        if std::env::var_os("TANG_MODQ_VERBOSE").is_some() {
            eprint!("{}", String::from_utf8_lossy(&out.stderr));
        }
        let (time, rest) = out.stdout.split_at(8);
        let ns = u64::from_le_bytes(time.try_into().expect("8 bytes"));
        let words: Vec<u32> = rest
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // the runner writes outputs as [output][batch][size]
        let b = sets.len();
        let mut res = vec![Vec::new(); b];
        let mut at = 0;
        for &n in &self.out_sizes {
            for r in res.iter_mut() {
                r.push(words[at..at + n].to_vec());
                at += n;
            }
        }
        Ok((res, ns))
    }
}

const METAL_HOST: &str = r#"
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 3) return 2;
    uint32_t B = (uint32_t)atoi(argv[1]);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[2]] encoding:NSUTF8StringEncoding error:&err];
    if (!src) { fprintf(stderr, "read: %s\n", [[err description] UTF8String]); return 3; }
    MTLCompileOptions *opt = [MTLCompileOptions new];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", [[err description] UTF8String]); return 4; }
    // inputs arrive per instance: [batch][input][size]; buffers want [input][batch][size]
    size_t per_set = 0; for (int i = 0; i < NODES; i++) if (KIND[i] > 0) per_set += SIZE[i];
    uint32_t *raw = malloc(per_set * B * 4 + 4);
    if (fread(raw, 4, per_set * B, stdin) != per_set * B) return 5;
    id<MTLBuffer> cst = [dev newBufferWithLength:(CONST_TOTAL + 1) * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> arena = [dev newBufferWithLength:(DATA_TOTAL * B + 1) * 4 options:MTLResourceStorageModeShared];
    if (!arena) { fprintf(stderr, "arena of %llu MB: allocation failed\n", DATA_TOTAL * B * 4 / 1000000); return 8; }
    uint32_t *cp = [cst contents], *ap = [arena contents];
    for (int i = 0; i < NODES; i++) if (KIND[i] == -1) memcpy(cp + CONST_OFF[i], CONSTS[i], SIZE[i] * 4);
    for (int k = 1; k <= N_IN; k++) {
      size_t off = 0;
      for (int i = 0; i < NODES; i++) if (KIND[i] > 0 && KIND[i] < k) off += SIZE[i];
      for (int i = 0; i < NODES; i++) if (KIND[i] == k)
        for (uint32_t b = 0; b < B; b++) memcpy(ap + B * DATA_OFF[i] + (uint64_t)b * SIZE[i], raw + b * per_set + off, SIZE[i] * 4);
    }
    static id<MTLComputePipelineState> pipes[KERNELS_N];
    for (int k = 0; k < KERNELS_N; k++) {
      id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithFormat:@"k%d", k]];
      pipes[k] = [dev newComputePipelineStateWithFunction:fn error:&err];
      if (!pipes[k]) { fprintf(stderr, "pipeline %d: %s\n", k, [[err description] UTF8String]); return 6; }
    }
    id<MTLBuffer> pbuf = [dev newBufferWithBytes:PARAMS length:(N_PARAMS ? N_PARAMS : 1) * 8 options:MTLResourceStorageModeShared];
    // groups: for each level, for each kernel, its launches as (kernel, first thread, params offset)
    id<MTLBuffer> lvb = [dev newBufferWithLength:(LAUNCHES * 3 + 3) * 4 options:MTLResourceStorageModeShared];
    uint32_t *lv = [lvb contents];
    int *by_level = malloc(sizeof(int) * (LAUNCHES + 1));
    size_t nb = 0;
    for (int L = 0; L < LEVELS; L++) for (int l = 0; l < LAUNCHES; l++) if (L_LEVEL[l] == L) by_level[nb++] = l;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setBuffer:cst offset:0 atIndex:0];
    [enc setBuffer:arena offset:0 atIndex:1];
    [enc setBytes:&B length:4 atIndex:2];
    [enc setBuffer:pbuf offset:0 atIndex:3];
    int dispatches = 0;
    size_t w = 0, at = 0;
    char *used = calloc(LAUNCHES, 1);
    int ungroup = getenv("TANG_MODQ_UNGROUPED") != NULL; // one launch per dispatch, for comparison
    while (at < nb) {
      int L = L_LEVEL[by_level[at]];
      size_t end = at; while (end < nb && L_LEVEL[by_level[end]] == L) end++;
      for (size_t a = at; a < end; a++) {
        int l0 = by_level[a]; if (used[l0]) continue;
        int k = L_KERNEL[l0];
        size_t first = w; uint32_t start = 0, n = 0;
        for (size_t c = a; c < end; c++) {
          int l = by_level[c]; if (used[l] || L_KERNEL[l] != k || (ungroup && l != l0)) continue;
          used[l] = 1;
          lv[3 * w] = k; lv[3 * w + 1] = start; lv[3 * w + 2] = (uint32_t)L_POFF[l];
          start += (uint32_t)(L_PER[l] * B); w++; n++;
        }
        id<MTLComputePipelineState> pl = pipes[k];
        [enc setComputePipelineState:pl];
        [enc setBuffer:lvb offset:first * 12 atIndex:4];
        [enc setBytes:&n length:4 atIndex:5];
        NSUInteger tg = pl.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(start, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        dispatches++;
      }
      at = end;
    }
    fprintf(stderr, "dispatches: %d\n", dispatches);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "gpu: %s\n", [[cb.error description] UTF8String]); return 7; }
    uint64_t ns = (uint64_t)((cb.GPUEndTime - cb.GPUStartTime) * 1e9);
    fwrite(&ns, 8, 1, stdout);
    for (int o = 0; o < N_OUT; o++) {
      int i = STORAGE[OUTS[o]];
      size_t n = SIZE[i];
      if (KIND[i] == -1) { for (uint32_t b = 0; b < B; b++) fwrite(cp + CONST_OFF[i], 4, n, stdout); }
      else fwrite(ap + B * DATA_OFF[i], 4, n * B, stdout);
    }
  }
  return 0;
}
"#;

const CUDA_HOST: &str = r#"
int main(int argc, char **argv) {
  if (argc < 2) return 2;
  uint32_t B = (uint32_t)atoi(argv[1]);
  size_t per_set = 0; for (int i = 0; i < NODES; i++) if (KIND[i] > 0) per_set += SIZE[i];
  uint32_t *raw = (uint32_t *)malloc(per_set * B * 4 + 4);
  if (fread(raw, 4, per_set * B, stdin) != per_set * B) return 5;
  uint32_t *cst = 0, *arena = 0;
  cudaMalloc((void **)&cst, (CONST_TOTAL + 1) * 4);
  cudaMalloc((void **)&arena, (DATA_TOTAL * B + 1) * 4);
  for (int i = 0; i < NODES; i++) if (KIND[i] == -1) cudaMemcpy(cst + CONST_OFF[i], CONSTS[i], SIZE[i] * 4, cudaMemcpyHostToDevice);
  for (int k = 1; k <= N_IN; k++) {
    size_t off = 0;
    for (int i = 0; i < NODES; i++) if (KIND[i] > 0 && KIND[i] < k) off += SIZE[i];
    for (int i = 0; i < NODES; i++) if (KIND[i] == k)
      for (uint32_t b = 0; b < B; b++) cudaMemcpy(arena + B * DATA_OFF[i] + (uint64_t)b * SIZE[i], raw + b * per_set + off, SIZE[i] * 4, cudaMemcpyHostToDevice);
  }
  uint64_t *dparams = 0;
  cudaMalloc((void **)&dparams, (N_PARAMS ? N_PARAMS : 1) * 8);
  cudaMemcpy(dparams, PARAMS, N_PARAMS * 8, cudaMemcpyHostToDevice);
  uint32_t *hlv = (uint32_t *)malloc((LAUNCHES * 3 + 3) * 4), *dlv = 0;
  int *by_level = (int *)malloc(sizeof(int) * (LAUNCHES + 1));
  size_t nb = 0;
  for (int L = 0; L < LEVELS; L++) for (int l = 0; l < LAUNCHES; l++) if (L_LEVEL[l] == L) by_level[nb++] = l;
  // plan the groups first so the table uploads once
  int *g_kernel = (int *)malloc(sizeof(int) * (LAUNCHES + 1));
  uint32_t *g_first = (uint32_t *)malloc(4 * (LAUNCHES + 1)), *g_n = (uint32_t *)malloc(4 * (LAUNCHES + 1)), *g_total = (uint32_t *)malloc(4 * (LAUNCHES + 1));
  char *used = (char *)calloc(LAUNCHES, 1);
  size_t w = 0, at = 0, ng = 0;
  while (at < nb) {
    int L = L_LEVEL[by_level[at]];
    size_t end = at; while (end < nb && L_LEVEL[by_level[end]] == L) end++;
    for (size_t a = at; a < end; a++) {
      int l0 = by_level[a]; if (used[l0]) continue;
      int k = L_KERNEL[l0];
      g_kernel[ng] = k; g_first[ng] = (uint32_t)w; uint32_t start = 0, n = 0;
      for (size_t c = a; c < end; c++) {
        int l = by_level[c]; if (used[l] || L_KERNEL[l] != k) continue;
        used[l] = 1;
        hlv[3 * w] = k; hlv[3 * w + 1] = start; hlv[3 * w + 2] = (uint32_t)L_POFF[l];
        start += (uint32_t)(L_PER[l] * B); w++; n++;
      }
      g_n[ng] = n; g_total[ng] = start; ng++;
    }
    at = end;
  }
  cudaMalloc((void **)&dlv, (LAUNCHES * 3 + 3) * 4);
  cudaMemcpy(dlv, hlv, (LAUNCHES * 3 + 3) * 4, cudaMemcpyHostToDevice);
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (size_t gi = 0; gi < ng; gi++) {
    const uint32_t *lvp = dlv + 3 * g_first[gi];
    uint32_t n = g_n[gi], total = g_total[gi];
    const uint64_t *P = dparams;
    void *args[7] = {&cst, &arena, &B, &P, &lvp, &n, &total};
    unsigned tb = 256, blocks = (total + tb - 1) / tb;
    cudaLaunchKernel(KERNELS[g_kernel[gi]], dim3(blocks), dim3(tb), args, 0, 0);
  }
  cudaEventRecord(t1); cudaEventSynchronize(t1);
  float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
  uint64_t ns = (uint64_t)(ms * 1e6);
  fwrite(&ns, 8, 1, stdout);
  for (int o = 0; o < N_OUT; o++) {
    int i = STORAGE[OUTS[o]];
    size_t n = SIZE[i];
    size_t total = KIND[i] == -1 ? n : n * B;
    uint32_t *h = (uint32_t *)malloc(total * 4 + 4);
    cudaMemcpy(h, KIND[i] == -1 ? cst + CONST_OFF[i] : arena + B * DATA_OFF[i], total * 4, cudaMemcpyDeviceToHost);
    if (KIND[i] == -1) { for (uint32_t b = 0; b < B; b++) fwrite(h, 4, n, stdout); } else fwrite(h, 4, total, stdout);
  }
  return 0;
}
"#;

#[cfg(test)]
mod tests {
    use super::super::{intt, ntt, Label, ModGraph};
    use super::*;

    const Q: u32 = 8380417;

    fn lcg(s: &mut u64) -> u32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as u32) % Q
    }

    fn ntt_graph() -> (ModGraph, ModId) {
        let mut g = ModGraph::new(Q);
        let (a, b) = (
            g.input(&[256], Label::Secret),
            g.input(&[256], Label::Public),
        );
        let (fa, fb) = (ntt(&mut g, a, 1753), ntt(&mut g, b, 1753));
        let p = g.map(
            &[256],
            &[(fa, ModGraph::id(1)), (fb, ModGraph::id(1))],
            |b| {
                let (x, y) = (b.arg(0), b.arg(1));
                b.mul(x, y)
            },
        );
        let c = intt(&mut g, p, 1753);
        (g, c)
    }

    fn sets(n: usize) -> Vec<Vec<Vec<u32>>> {
        let mut s = 9;
        (0..n)
            .map(|_| {
                vec![
                    (0..256).map(|_| lcg(&mut s)).collect(),
                    (0..256).map(|_| lcg(&mut s)).collect(),
                ]
            })
            .collect()
    }

    #[test]
    fn cuda_source_is_emitted() {
        let (g, c) = ntt_graph();
        let cu = g.to_cuda(&[c]);
        assert!(
            cu.contains("__global__ void k0")
                && cu.contains("cudaLaunchKernel")
                && cu.contains("__device__")
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_matches_the_interpreter_across_a_batch() {
        let (g, c) = ntt_graph();
        let built = g.compile_metal(&[c]).expect("clang with Metal");
        let batch = sets(64);
        let (got, _ns) = built.run(&batch).expect("a Metal device");
        for (b, set) in batch.iter().enumerate() {
            assert_eq!(got[b], g.eval(set, &[c]), "instance {b}");
        }
    }

    #[test]
    fn cuda_matches_the_interpreter_where_nvcc_exists() {
        if Command::new(std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into()))
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("no nvcc: CUDA emitted but not run");
            return;
        }
        let (g, c) = ntt_graph();
        let built = g.compile_cuda(&[c]).expect("nvcc");
        let batch = sets(64);
        let (got, _) = built.run(&batch).expect("a CUDA device");
        for (b, set) in batch.iter().enumerate() {
            assert_eq!(got[b], g.eval(set, &[c]), "instance {b}");
        }
    }
}
