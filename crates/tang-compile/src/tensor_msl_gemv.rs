//! Metal lowering for decode-shaped programs: the Apple twin of
//! `tensor_cuda_gemv.rs`.
//!
//! [`TensorGraph::to_msl_decode`] is `to_cuda_decode` for Metal (run
//! through wgpu's MSL passthrough, `tang_gpu::program`). The kernels are
//! the same: warp reductions are simdgroup reductions (32 lanes on every
//! Apple GPU, `simd_shuffle_xor`), packed 4-bit, bf16 and f16 inputs are
//! decoded where they are loaded, runtime loop bounds come from affine
//! `select` predicates, and in-place updates write their donor. They are
//! generated once, in CUDA spelling, and [`transliterate`]d: the
//! analysis (classification of loads, register dims, bounds) is shared,
//! and only the surface syntax differs.
//!
//! Two things are Metal's own:
//!
//! - **In-place bindings.** wgpu cannot bind one buffer read-only and
//!   read-write in one dispatch, so an in-place update does not list its
//!   donor among its reads: the old value is read through `out`.
//! - **Unlowered kernels stay WGSL** (wgpu runs both), where CUDA
//!   transpiles them.
//!
//! Matmul-shaped kernels of a prefill (`DecodeOpts::mma`) take the
//! simdgroup-matrix tile of `tensor_msl_mma.rs` when their operands are
//! f32.
use std::fmt::Write;

use super::cuda_gemv::{DecodeOpts, DecodeProgram, DecodeTarget, Kernel, PRELUDE};
use super::{Format, TensorGraph, TensorId};

/// [`PRELUDE`] in Metal: the packed loads and half conversions.
const MSL_PRELUDE: &str = r#"static inline float tang_u4(device const float* b, int a) {
    uint q = (((device const uint*)b)[a >> 3] >> ((a & 7) * 4)) & 15u;
    return as_type<float>(0x4b000000u | q) - 8388608.0f;
}
static inline float tang_bf16(device const float* b, int a) {
    uint w = ((device const uint*)b)[a >> 1];
    return as_type<float>((a & 1) ? (w & 0xffff0000u) : (w << 16));
}
static inline float tang_h2f(uint h) {
    return float(as_type<half>(ushort(h)));
}
static inline uint tang_f2h(float f) {
    return uint(as_type<ushort>(half(f)));
}
static inline uint tang_f2bf(float f) {
    uint b = as_type<uint>(f);
    return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}
static inline float tang_f16(device const float* b, int a) {
    uint w = ((device const uint*)b)[a >> 1];
    return tang_h2f((a & 1) ? (w >> 16) : (w & 0xffffu));
}
"#;

impl TensorGraph {
    /// [`to_cuda_decode`](Self::to_cuda_decode) for Metal: the same
    /// kernels in MSL (see the module docs). Buffers of packed inputs are
    /// sized in words; an in-place output's kernel does not bind its donor.
    pub fn to_msl_decode(&self, outputs: &[TensorId], opts: &DecodeOpts) -> DecodeProgram {
        self.to_decode(outputs, opts, DecodeTarget::Msl)
    }
}

/// A prefill matmul as a simdgroup-matrix tile, when its operands are
/// f32 (the tile stages `float`).
pub(super) fn mma_kernel(g: &TensorGraph, root: usize, reads: &[usize]) -> Option<Kernel> {
    let f32_only = reads
        .iter()
        .chain([&root])
        .all(|&j| g.nodes[j].precision.storage == Format::F32);
    if !f32_only {
        return None;
    }
    let k = g.msl_mma_kernel(root, reads)?;
    let threads = k.workgroup_size as usize;
    Some(Kernel {
        source: k.source,
        threads,
        blocks: (k.count as usize).div_ceil(threads),
    })
}

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

/// A decode kernel in CUDA spelling (from `tensor_cuda_gemv.rs`) as MSL.
/// `reads` are the buffers it binds (graph node numbers, the `b{j}` of
/// the source) before `out`; `donor`'s loads (an in-place update's old
/// value) read `out`. MSL sources (`#include`) pass through.
pub(super) fn transliterate(src: &str, reads: &[usize], donor: Option<usize>) -> String {
    if src.starts_with("#include") {
        return src.to_string();
    }
    assert!(
        src.contains(PRELUDE),
        "decode kernel without the CUDA prelude"
    );
    let mut body = String::new();
    for line in src.replace(PRELUDE, MSL_PRELUDE).lines() {
        if line.starts_with("extern \"C\" __global__") {
            let mut params: Vec<String> = reads
                .iter()
                .enumerate()
                .map(|(i, j)| format!("device const float* b{j} [[buffer({i})]]"))
                .collect();
            params.push(format!("device float* out [[buffer({})]]", reads.len()));
            params.push(format!(
                "constant Params& params [[buffer({})]]",
                reads.len() + 1
            ));
            params.push("uint3 tang_wid [[threadgroup_position_in_grid]]".into());
            params.push("uint3 tang_lid [[thread_position_in_threadgroup]]".into());
            writeln!(body, "kernel void main0({}) {{", params.join(", ")).unwrap();
            // grids wider than 65535 threadgroups wrap into y
            writeln!(
                body,
                "    const int tang_bid = int(tang_wid.x + tang_wid.y * 65535u);"
            )
            .unwrap();
            writeln!(body, "    if (tang_bid >= TANG_TILES) return;").unwrap();
            writeln!(body, "    const int tang_tid = int(tang_lid.x);").unwrap();
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    // __ldg(p) is *p
    let mut s = rewrite_calls(&body, "__ldg(", |a| format!("(*({a}))"));
    // (&v.x)[i] — a vector's element by index — is v[i]
    while let Some(i) = s.find("(&") {
        let close = close_paren(&s, i);
        let inner = &s[i + 2..close];
        let Some(v) = inner.strip_suffix(".x") else {
            panic!("unexpected address-of in a decode kernel: {inner}")
        };
        s = format!("{}{v}{}", &s[..i], &s[close + 1..]);
    }
    let subs: [(&str, &str); 16] = [
        ("__shfl_xor_sync(0xffffffffu, ", "simd_shuffle_xor("),
        (
            "__syncthreads()",
            "threadgroup_barrier(mem_flags::mem_threadgroup)",
        ),
        ("__shared__ ", "threadgroup "),
        ("__uint_as_float(", "as_type<float>("),
        ("__float_as_uint(", "as_type<uint>("),
        ("ceilf(", "ceil("),
        ("floorf(", "floor("),
        ("threadIdx.x", "tang_tid"),
        ("blockIdx.x", "tang_bid"),
        ("gridDim.x", "TANG_TILES"),
        ("(const uint4*)", "(device const uint4*)"),
        ("(const uint2*)", "(device const uint2*)"),
        ("(const float4*)", "(device const float4*)"),
        ("(const unsigned*)", "(device const unsigned*)"),
        ("(unsigned*)", "(device unsigned*)"),
        ("unsigned* o ", "device unsigned* o "),
    ];
    for (a, b) in subs {
        s = s.replace(a, b);
    }
    if let Some(d) = donor {
        s = rename_ident(&s, &format!("b{d}"), "out");
    }
    for bad in ["__", "asm(", "extern"] {
        assert!(
            !s.contains(bad),
            "untransliterated `{bad}` in a decode kernel"
        );
    }
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         struct Params {{ uint count; uint p1; uint p2; uint p3; }};\n{s}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transliterates_loads_and_shuffles() {
        let src = format!(
            "// k\n#define TANG_TILES 4\n{PRELUDE}extern \"C\" __global__ void main0(const float* __restrict__ b3, float* out, unsigned count) {{\n    float x = __ldg((const float4*)(b3 + (2)) + 1).x;\n    float y = (&v0[0][e >> 2].x)[e & 3];\n    y = __shfl_xor_sync(0xffffffffu, y, 4);\n    out[0] = b3[0] + x + y;\n}}\n"
        );
        let m = transliterate(&src, &[], Some(3));
        assert!(m.starts_with("#include <metal_stdlib>"));
        assert!(m.contains("(*((device const float4*)(out + (2)) + 1)).x"));
        assert!(m.contains("v0[0][e >> 2][e & 3]"));
        assert!(m.contains("simd_shuffle_xor(y, 4)"));
        assert!(m.contains("out[0] = out[0]"));
        assert!(!m.contains("b3"));
    }
}
