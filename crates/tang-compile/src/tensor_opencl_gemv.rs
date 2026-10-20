//! OpenCL lowering for decode-shaped programs: the OpenCL C twin of
//! `tensor_cuda_gemv.rs`, for GPUs with OpenCL and no CUDA/Metal (the
//! Booster K1's Adreno 740, where OpenCL dispatch is several times cheaper
//! than Vulkan's).
//!
//! [`TensorGraph::to_opencl_decode`] generates the kernels once in CUDA
//! spelling, as Metal and Vulkan do, and [`transliterate`]s them. OpenCL C
//! has pointers, so the shape is CUDA's:
//!
//! - **The calling convention is `to_opencl`'s**: `main0(b{j}…, out, uint
//!   count)` with the reads in `WgslKernel::reads` order, `count / threads`
//!   groups of `threads` items (one dimension). As for Metal, an in-place
//!   update does not bind its donor: its old value is read through `out`.
//! - **Warp butterflies** (`__shfl_xor_sync` over 32 lanes, XOR offsets
//!   ≤ 16) are `TANG_SHFL_XOR(x, off)`, which the runtime defines for the
//!   device: a subgroup shuffle where the driver has one (Adreno
//!   subgroups are 64 or 128 wide; an offset ≤ 16 stays inside an aligned
//!   run of 32 lanes, so the sums are a 32-lane warp's), else through
//!   local memory ([`OPENCL_SHFL_LOCAL`], for drivers without subgroups,
//!   e.g. Apple's).
//! - **16-bit conversions are core OpenCL** (`vload_half`,
//!   `vstore_half_rte`: round to nearest even, as CUDA's `cvt.rn`), so no
//!   `cl_khr_fp16` is needed.
//! - `(&v.x)[i]`, a vector's element by index, is `((T*)&v)[i]`.
//!
//! Unlowered kernels are `to_opencl`'s WGSL transpile; they never read
//! packed inputs (`to_decode` checks).
use std::fmt::Write;

use super::cuda_gemv::{DecodeOpts, DecodeProgram, DecodeTarget, PRELUDE};
use super::{TensorGraph, TensorId};

/// [`PRELUDE`] in OpenCL C.
const OPENCL_PRELUDE: &str = r#"#define select(f, t, c) ((c) ? (t) : (f))
#define TANG_EL4(v, i) (((i) & 3) == 0 ? (v).s0 : ((i) & 3) == 1 ? (v).s1 : ((i) & 3) == 2 ? (v).s2 : (v).s3)
#define TANG_EL2(v, i) (((i) & 1) == 0 ? (v).s0 : (v).s1)
inline float tang_u4(__global const float* b, int a) {
    uint q = (((__global const uint*)b)[a >> 3] >> ((a & 7) * 4)) & 15u;
    return as_float(0x4b000000u | q) - 8388608.0f;
}
inline float tang_bf16(__global const float* b, int a) {
    uint w = ((__global const uint*)b)[a >> 1];
    return as_float((a & 1) ? (w & 0xffff0000u) : (w << 16));
}
inline float tang_h2f(uint h) {
    ushort s = (ushort)h;
    return vload_half(0, (const __private half*)&s);
}
inline uint tang_f2h(float f) {
    ushort s;
    vstore_half_rte(f, 0, (__private half*)&s);
    return (uint)s;
}
inline uint tang_f2bf(float f) {
    uint b = as_uint(f);
    return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}
inline float tang_f16(__global const float* b, int a) {
    uint w = ((__global const uint*)b)[a >> 1];
    return tang_h2f((a & 1) ? (w >> 16) : (w & 0xffffu));
}
"#;

/// `TANG_SHFL_XOR` through local memory, for drivers without subgroup
/// shuffles. Its barriers need every work-item of the group to reach
/// them, which a warp kernel's grid-stride loop over groups does not
/// guarantee (warps of one group can run different trip counts): it was
/// right on Apple's driver and wrong on the Adreno, so it is a fallback
/// for checking on hosts, not a production path.
pub const OPENCL_SHFL_LOCAL: &str = r#"#define TANG_SHFL_LOCAL 1
#define TANG_SHFL_XOR(x, off) tang_shfl_xor(tang_xch, tang_tid, (x), (off))
inline float tang_shfl_xor(__local float* xch, int tid, float x, int off) {
    barrier(CLK_LOCAL_MEM_FENCE);
    xch[tid] = x;
    barrier(CLK_LOCAL_MEM_FENCE);
    return xch[tid ^ off];
}
"#;

/// `TANG_SHFL_XOR` as `cl_khr_subgroup_shuffle`'s `sub_group_shuffle_xor`.
pub const OPENCL_SHFL_KHR: &str = r#"#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define TANG_SHFL_XOR(x, off) sub_group_shuffle_xor((x), (uint)(off))
"#;

/// `TANG_SHFL_XOR` as `cl_qcom_subgroup_shuffle` (Adreno).
pub const OPENCL_SHFL_QCOM: &str = r#"#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define TANG_SHFL_XOR(x, off) qcom_sub_group_shuffle_xor((x), (uint)(off), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.0f)
"#;

impl TensorGraph {
    /// [`to_cuda_decode`](Self::to_cuda_decode) for OpenCL: the same
    /// kernels in OpenCL C (see the module docs). Each native kernel starts
    /// with `// TANG_SHFL` where the runtime puts its `TANG_SHFL_XOR`
    /// definition ([`OPENCL_SHFL_LOCAL`], [`OPENCL_SHFL_KHR`],
    /// [`OPENCL_SHFL_QCOM`]). Buffers of packed inputs are sized in words.
    pub fn to_opencl_decode(&self, outputs: &[TensorId], opts: &DecodeOpts) -> DecodeProgram {
        self.to_decode(outputs, opts, DecodeTarget::OpenCl)
    }
}

/// Marker line (after the kernel's first line) that the runtime replaces
/// with its shuffle definition.
pub const SHFL_MARKER: &str = "// TANG_SHFL\n";

/// Index of the `)` closing the `(` at `open`.
fn close_paren(s: &str, open: usize) -> usize {
    let mut depth = 0;
    for (i, c) in s[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return open + i;
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced parentheses in a decode kernel");
}

/// Replace every `head(ARGS)` by `f(ARGS)` (ARGS balanced).
fn rewrite_calls(s: &str, head: &str, f: impl Fn(&str) -> String) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(head) {
        out.push_str(&rest[..i]);
        let open = i + head.len() - 1;
        let close = close_paren(rest, open);
        out.push_str(&f(&rest[open + 1..close]));
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    out
}

/// Replace the identifier `from` (whole tokens only) by `to`.
fn rename_ident(s: &str, from: &str, to: &str) -> String {
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if ident(b[i]) {
            let start = i;
            while i < b.len() && ident(b[i]) {
                i += 1;
            }
            let tok = &s[start..i];
            out.push_str(if tok == from { to } else { tok });
        } else {
            let start = i;
            while i < b.len() && !ident(b[i]) {
                i += 1;
            }
            out.push_str(&s[start..i]);
        }
    }
    out
}

/// A decode kernel in CUDA spelling (from `tensor_cuda_gemv.rs`) as
/// OpenCL C. `reads` are the buffers it binds (graph node numbers, the
/// `b{j}` of the source) before `out`; `donor`'s loads (an in-place
/// update's old value) read `out`.
pub(super) fn transliterate(src: &str, reads: &[usize], donor: Option<usize>) -> String {
    assert!(
        src.contains(PRELUDE),
        "decode kernel without the CUDA prelude"
    );
    let threads: usize = src
        .lines()
        .find_map(|l| l.strip_prefix("#define TANG_THREADS "))
        .and_then(|v| v.trim().parse().ok())
        .expect("TANG_THREADS");
    let mut body = String::new();
    for line in src.replace(PRELUDE, OPENCL_PRELUDE).lines() {
        if line.starts_with("extern \"C\" __global__") {
            let mut params: Vec<String> = reads
                .iter()
                .map(|j| format!("__global const float* restrict b{j}"))
                .collect();
            params.push("__global float* restrict out".into());
            params.push("uint count".into());
            writeln!(
                body,
                "__kernel __attribute__((reqd_work_group_size({threads}, 1, 1))) void main0({}) {{",
                params.join(", ")
            )
            .unwrap();
            writeln!(body, "    const int tang_bid = (int)get_group_id(0);").unwrap();
            writeln!(body, "    const int tang_tid = (int)get_local_id(0);").unwrap();
            writeln!(body, "#ifdef TANG_SHFL_LOCAL").unwrap();
            writeln!(body, "    __local float tang_xch[{threads}];").unwrap();
            writeln!(body, "#endif").unwrap();
            writeln!(body, "    if (tang_bid >= TANG_TILES) return;").unwrap();
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    // __ldg(p) is *p
    let mut s = rewrite_calls(&body, "__ldg(", |a| format!("(*({a}))"));
    // (&v.x)[i] — a vector's element by index. The index is a constant once
    // the loops unroll; a select chain folds to a register read, where
    // taking the vector's address (`((T*)&v)[i]`) would put the whole
    // register array in private memory (0.95 ms per GEMV on the Adreno,
    // ~5x Vulkan). The width comes from the array's declaration.
    let width = |name: &str| -> usize {
        for (ty, w) in [("float4 ", 4), ("uint4 ", 4), ("uint2 ", 2)] {
            if s.contains(&format!("{ty}{name}[")) {
                return w;
            }
        }
        panic!("vector element of an undeclared array `{name}` in a decode kernel")
    };
    let mut out = String::with_capacity(s.len());
    let mut rest: &str = &s;
    while let Some(i) = rest.find("(&") {
        out.push_str(&rest[..i]);
        let close = close_paren(rest, i);
        let inner = &rest[i + 2..close];
        let Some(v) = inner.strip_suffix(".x") else {
            panic!("unexpected address-of in a decode kernel: {inner}")
        };
        let name: String = v
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let after = &rest[close + 1..];
        assert!(after.starts_with('['), "vector element without an index");
        let iclose = after.find(']').expect("index ]");
        let idx = &after[1..iclose];
        let m = if width(&name) == 4 {
            "TANG_EL4"
        } else {
            "TANG_EL2"
        };
        out.push_str(&format!("{m}({v}, {idx})"));
        rest = &after[iclose + 1..];
    }
    out.push_str(rest);
    let mut s = out;
    let subs: [(&str, &str); 15] = [
        ("__shfl_xor_sync(0xffffffffu, ", "TANG_SHFL_XOR("),
        ("__syncthreads()", "barrier(CLK_LOCAL_MEM_FENCE)"),
        ("__shared__ ", "__local "),
        ("__uint_as_float(", "as_float("),
        ("__float_as_uint(", "as_uint("),
        ("ceilf(", "ceil("),
        ("floorf(", "floor("),
        ("threadIdx.x", "tang_tid"),
        ("blockIdx.x", "tang_bid"),
        ("gridDim.x", "TANG_TILES"),
        ("(const uint4*)", "(__global const uint4*)"),
        ("(const uint2*)", "(__global const uint2*)"),
        ("(const float4*)", "(__global const float4*)"),
        ("(const unsigned*)", "(__global const unsigned*)"),
        ("(unsigned*)", "(__global unsigned*)"),
    ];
    for (a, b) in subs {
        s = s.replace(a, b);
    }
    s = s.replace("unsigned* o ", "__global unsigned* o ");
    if let Some(d) = donor {
        s = rename_ident(&s, &format!("b{d}"), "out");
    }
    for bad in [
        "__ldg",
        "__shfl",
        "__device__",
        "asm(",
        "extern",
        "__restrict__",
    ] {
        assert!(
            !s.contains(bad),
            "untransliterated `{bad}` in a decode kernel:\n{s}"
        );
    }
    // after the kernel's first line (its `// … groups x … points` summary)
    match s.split_once('\n') {
        Some((first, rest)) if first.starts_with("//") => format!("{first}\n{SHFL_MARKER}{rest}"),
        _ => format!("{SHFL_MARKER}{s}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transliterates_loads_and_shuffles() {
        let src = format!(
            "// k\n#define TANG_TILES 4\n#define TANG_THREADS 32\n{PRELUDE}extern \"C\" __global__ void __launch_bounds__(32) main0(const float* __restrict__ b3, float* __restrict__ out, unsigned count) {{\n    float x = __ldg((const float4*)(b3 + (2)) + 1).x;\n    float4 v0[1][2];\n    float y = (&v0[0][e >> 2].x)[e & 3];\n    y = __shfl_xor_sync(0xffffffffu, y, 4);\n    out[0] = b3[0] + x + y;\n}}\n"
        );
        let m = transliterate(&src, &[], Some(3));
        assert!(m.starts_with(&format!("// k\n{SHFL_MARKER}")));
        assert!(m.contains(
            "reqd_work_group_size(32, 1, 1))) void main0(__global float* restrict out, uint count)"
        ));
        assert!(m.contains("(*((__global const float4*)(out + (2)) + 1)).x"));
        assert!(m.contains("TANG_EL4(v0[0][e >> 2], e & 3)"));
        assert!(m.contains("TANG_SHFL_XOR(y, 4)"));
        assert!(m.contains("out[0] = out[0]"));
        assert!(!m.contains("b3"));
    }
}
