//! Vulkan lowering for decode-shaped programs: GLSL compute, compiled to
//! SPIR-V by the runtime and dispatched through wgpu's SPIR-V passthrough.
//!
//! [`TensorGraph::to_glsl_decode`] is `to_cuda_decode` for Vulkan GPUs
//! that have no CUDA and no Metal (the Adreno on the Booster K1, Mali, …).
//! As for Metal, the kernels are generated once in CUDA spelling and
//! `transliterate`d. GLSL has no pointers, which decides the shape:
//!
//! - **Every buffer is a `uint[]`.** Packed nibbles, bf16/f16 halves and
//!   f32 words are all read as raw words, so a bit pattern is never
//!   loaded as a float first (which a GPU may canonicalize if it is a
//!   NaN). Float reads are `uintBitsToFloat(b[i])`, float stores
//!   `floatBitsToUint`.
//! - **Pointer arithmetic becomes indexing.** `(const float4*)(b + e) + q`
//!   is four words from `e + 4q`; `(const uint4*)b + e` is four from `4e`;
//!   `((unsigned*)out)[e]` is `out[e]`; the packed loads of the CUDA
//!   prelude become macros over the buffer name, since GLSL cannot pass a
//!   buffer to a function.
//! - **Warps are 32 lanes inside a subgroup.** Vulkan subgroups are 64 or
//!   128 wide on Adreno; the kernels' butterflies shuffle by XOR offsets
//!   of at most 16, which never leave an aligned run of 32 lanes, so
//!   `subgroupShuffleXor` computes the same sums as on a 32-lane warp
//!   (subgroups being contiguous runs of invocations).
//! - **`shared` arrays move to global scope**, and the 16-bit conversions
//!   are integer ops with explicit round-to-nearest-even (`packHalf2x16`'s
//!   rounding is the driver's choice) and `unpackHalf2x16` (exact); no
//!   `shaderFloat16` needed.
//!
//! Unlowered kernels stay WGSL, which wgpu compiles itself; they never
//! read packed inputs (`to_decode` checks).
use std::fmt::Write;

use super::cuda_gemv::{DecodeOpts, DecodeProgram, DecodeTarget, PRELUDE};
use super::{TensorGraph, TensorId};

/// First line of every kernel `transliterate` emits; the runtime keys
/// SPIR-V compilation on it.
pub const GLSL_VERSION: &str = "#version 450";

/// [`PRELUDE`] in GLSL: the packed loads (macros over a `uint[]` buffer)
/// and the 16-bit conversions.
const GLSL_PRELUDE: &str = r#"#define tang_u4(b, a) (float((b[(a) >> 3] >> (((a) & 7) * 4)) & 15u))
#define tang_bf16(b, a) uintBitsToFloat((((a) & 1) != 0) ? (b[(a) >> 1] & 0xffff0000u) : (b[(a) >> 1] << 16))
float tang_h2f(uint h) { return unpackHalf2x16(h & 0xffffu).x; }
uint tang_f2h(float f) {
    // round-to-nearest-even in integer ops (Giesen's float_to_half_fast3_rtne):
    // GLSL leaves packHalf2x16's rounding to the driver, and the Adreno's
    // differs from CUDA's cvt.rn, which biases every cached value alike
    uint x = floatBitsToUint(f);
    uint sign = (x >> 16) & 0x8000u;
    x &= 0x7fffffffu;
    uint h;
    if (x >= 0x47800000u) {
        h = (x > 0x7f800000u) ? 0x7e00u : 0x7c00u;
    } else if (x < 0x38800000u) {
        h = floatBitsToUint(uintBitsToFloat(x) + 0.5) - 0x3f000000u;
    } else {
        h = (x + 0xc8000fffu + ((x >> 13) & 1u)) >> 13;
    }
    return sign | h;
}
uint tang_f2bf(float f) { uint b = floatBitsToUint(f); return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16; }
#define tang_f16(b, a) tang_h2f((((a) & 1) != 0) ? (b[(a) >> 1] >> 16) : (b[(a) >> 1] & 0xffffu))
"#;

impl TensorGraph {
    /// [`to_cuda_decode`](Self::to_cuda_decode) for Vulkan: the same
    /// kernels in GLSL (see the module docs). Buffers of packed inputs are
    /// sized in words; an in-place output's kernel does not bind its donor.
    pub fn to_glsl_decode(&self, outputs: &[TensorId], opts: &DecodeOpts) -> DecodeProgram {
        self.to_decode(outputs, opts, DecodeTarget::Glsl)
    }
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

/// Replace every `head(ARGS)` by `f(ARGS)` (ARGS balanced), innermost
/// calls first so `f` sees arguments already rewritten.
fn rewrite_calls(s: &str, head: &str, f: &dyn Fn(&str) -> String) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(head) {
        out.push_str(&rest[..i]);
        let open = i + head.len() - 1;
        let close = close_paren(rest, open);
        let inner = rewrite_calls(&rest[open + 1..close], head, f);
        out.push_str(&f(&inner));
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    out
}

/// `s` without one pair of parentheses around the whole of it, if it has one.
fn strip_outer(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('(') && close_paren(s, 0) == s.len() - 1 {
        s[1..s.len() - 1].trim()
    } else {
        s
    }
}

/// Split `BUF + EXPR` (optionally parenthesized) into the buffer name and
/// the index expression; a bare `BUF` indexes 0.
fn split_base(a: &str) -> (String, String) {
    let a = strip_outer(a);
    match a.split_once('+') {
        Some((b, e)) => (b.trim().to_string(), e.trim().to_string()),
        None => (a.to_string(), "0".to_string()),
    }
}

/// One `__ldg(ARG)` as GLSL indexing of a `uint[]` buffer.
fn ldg(a: &str) -> String {
    let a = a.trim();
    let vec = |kind: &str, words: usize, rest: &str| -> String {
        // `(const T*)(BUF + E) + q` (elements) or `(const T*)BUF + E` (vectors)
        let rest = rest.trim();
        let (buf, idx) = if rest.starts_with('(') {
            let close = close_paren(rest, 0);
            let (buf, e) = split_base(&rest[1..close]);
            let q = rest[close + 1..].trim().trim_start_matches('+').trim();
            let q = if q.is_empty() { "0" } else { q };
            (buf, format!("(({e}) + {words} * ({q}))"))
        } else {
            let (buf, e) = split_base(rest);
            (buf, format!("({words} * ({e}))"))
        };
        let lanes: Vec<String> = (0..words).map(|w| format!("{buf}[{idx} + {w}]")).collect();
        match kind {
            "float" => format!(
                "vec{words}({})",
                lanes
                    .iter()
                    .map(|l| format!("uintBitsToFloat({l})"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            _ => format!("uvec{words}({})", lanes.join(", ")),
        }
    };
    for (cast, kind, words) in [
        ("(const float4*)", "float", 4),
        ("(const uint4*)", "uint", 4),
        ("(const uint2*)", "uint", 2),
    ] {
        if let Some(rest) = a.strip_prefix(cast) {
            return vec(kind, words, rest);
        }
    }
    if let Some(rest) = a.strip_prefix("(const unsigned*)") {
        let (buf, e) = split_base(rest);
        return format!("{buf}[{e}]");
    }
    let (buf, e) = split_base(a);
    format!("uintBitsToFloat({buf}[{e}])")
}

/// The CUDA decode kernel `src` as a GLSL 450 compute shader; `reads` are
/// the bound inputs in order, `donor` the in-place input (bound as `out`).
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
    let mut head = String::new();
    let mut shared = String::new();
    let mut body = String::new();
    let mut in_body = false;
    for line in src.replace(PRELUDE, "").lines() {
        if line.starts_with("extern \"C\" __global__") {
            in_body = true;
            writeln!(body, "void main() {{").unwrap();
            // grids wider than 65535 workgroups wrap into y
            writeln!(
                body,
                "    const int tang_bid = int(gl_WorkGroupID.x + gl_WorkGroupID.y * 65535u);"
            )
            .unwrap();
            writeln!(body, "    if (tang_bid >= TANG_TILES) return;").unwrap();
            writeln!(
                body,
                "    const int tang_tid = int(gl_LocalInvocationID.x);"
            )
            .unwrap();
            continue;
        }
        if let Some(decl) = line.trim_start().strip_prefix("__shared__ ") {
            writeln!(shared, "shared {decl}").unwrap();
            continue;
        }
        if line.trim_start().starts_with("#pragma unroll") {
            continue;
        }
        let dst = if in_body { &mut body } else { &mut head };
        dst.push_str(line);
        dst.push('\n');
    }
    let mut s = body;
    // a read-modify-write alias of one word: `unsigned* o = (unsigned*)out + (E);`
    while let Some(i) = s.find("unsigned* o = (unsigned*)out + ") {
        let start = i + "unsigned* o = (unsigned*)out + ".len();
        let end = start + s[start..].find(';').expect("alias");
        let e = s[start..end].to_string();
        s = format!("{}const int o_ix = int({e});{}", &s[..i], &s[end + 1..]);
        s = s.replace("*o", "out[o_ix]");
    }
    s = rewrite_calls(&s, "__ldg(", &ldg);
    // (&v.x)[i] — a vector's element by index — is v[i]
    while let Some(i) = s.find("(&") {
        let close = close_paren(&s, i);
        let inner = &s[i + 2..close];
        let Some(v) = inner.strip_suffix(".x") else {
            panic!("unexpected address-of in a decode kernel: {inner}")
        };
        s = format!("{}{v}{}", &s[..i], &s[close + 1..]);
    }
    // word stores through a cast: ((unsigned*)out)[E] = V
    s = s.replace("((unsigned*)out)[", "out[");
    // float stores: out[E] = V;  ->  out[E] = floatBitsToUint(V);
    let mut stored = String::with_capacity(s.len());
    for line in s.lines() {
        let t = line.trim_start();
        let is_float_store = t.starts_with("out[")
            && !line.contains("tang_f2h")
            && !line.contains("tang_f2bf")
            && !line.contains("o_ix");
        if is_float_store {
            if let Some(eq) = line.find("] = ") {
                let lhs = &line[..eq + 1];
                let rhs = line[eq + 4..].trim_end().trim_end_matches(';');
                writeln!(stored, "{lhs} = floatBitsToUint({rhs});").unwrap();
                continue;
            }
        }
        stored.push_str(line);
        stored.push('\n');
    }
    s = stored;
    let subs: [(&str, &str); 14] = [
        ("__shfl_xor_sync(0xffffffffu, ", "subgroupShuffleXor("),
        ("__syncthreads()", "memoryBarrierShared(); barrier()"),
        ("__uint_as_float(", "uintBitsToFloat("),
        ("__float_as_uint(", "floatBitsToUint("),
        ("ceilf(", "ceil("),
        ("floorf(", "floor("),
        ("threadIdx.x", "tang_tid"),
        ("blockIdx.x", "tang_bid"),
        ("gridDim.x", "TANG_TILES"),
        ("(int)(", "int("),
        ("(float)(", "float("),
        ("(unsigned)(", "uint("),
        ("float4 ", "vec4 "),
        ("unsigned ", "uint "),
    ];
    for (a, b) in subs {
        s = s.replace(a, b);
    }
    s = s.replace("uint4 ", "uvec4 ").replace("uint2 ", "uvec2 ");
    s = call_casts(&s);
    // C's integer truth in a ternary: `(f & 1) ? a : b`
    s = int_conditions(&s);
    if let Some(d) = donor {
        s = rename_ident(&s, &format!("b{d}"), "out");
    }
    // `out` is a GLSL keyword
    s = rename_ident(&s, "out", "tang_out");
    let head = head
        .lines()
        .filter(|l| !l.starts_with("#define select"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut bindings = String::new();
    for (i, j) in reads.iter().enumerate() {
        writeln!(
            bindings,
            "layout(std430, set = 0, binding = {i}) readonly buffer B{j} {{ uint b{j}[]; }};"
        )
        .unwrap();
    }
    writeln!(
        bindings,
        "layout(std430, set = 0, binding = {}) buffer Out {{ uint tang_out[]; }};",
        reads.len()
    )
    .unwrap();
    writeln!(
        bindings,
        "layout(std140, set = 0, binding = {}) uniform Params {{ uint count; uint p1; uint p2; uint p3; }} params;",
        reads.len() + 1
    )
    .unwrap();
    let glsl = format!(
        "{GLSL_VERSION}\n#extension GL_KHR_shader_subgroup_shuffle : require\n\
         layout(local_size_x = {threads}) in;\n\
         #define select(f, t, c) ((c) ? (t) : (f))\n{head}\n{bindings}{GLSL_PRELUDE}{shared}{s}"
    );
    for bad in ["__", "asm(", "extern", "(const ", "*)"] {
        assert!(
            !glsl.contains(bad),
            "untransliterated `{bad}` in a decode kernel:\n{glsl}"
        );
    }
    glsl
}

/// C casts `(T)name(args)` (a cast applied to a call) as `T(name(args))`.
fn call_casts(s: &str) -> String {
    let mut s = s.to_string();
    for (c, t) in [("(int)", "int"), ("(float)", "float"), ("(uint)", "uint")] {
        while let Some(i) = s.find(c) {
            let start = i + c.len();
            let name_len = s[start..]
                .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
                .unwrap_or(0);
            let open = start + name_len;
            assert!(
                name_len > 0 && s[open..].starts_with('('),
                "unexpected cast in a decode kernel: {}",
                &s[i..(i + 40).min(s.len())]
            );
            let close = close_paren(&s, open);
            s = format!("{}{t}({}){}", &s[..i], &s[start..=close], &s[close + 1..]);
        }
    }
    s
}

/// `(IDENT & 1) ?` (an integer used as a condition, which GLSL refuses)
/// as `((IDENT & 1) != 0) ?`.
fn int_conditions(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(" & 1) ?") {
        let open = rest[..i].rfind('(').expect("condition");
        let ident = &rest[open + 1..i];
        if ident.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !ident.is_empty() {
            out.push_str(&rest[..open]);
            out.push_str(&format!("(({ident} & 1) != 0) ?"));
        } else {
            out.push_str(&rest[..i + " & 1) ?".len()]);
        }
        rest = &rest[i + " & 1) ?".len()..];
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `tang_f2h` of [`GLSL_PRELUDE`], step for step (the float add in the
    /// subnormal branch is an f32 add, which GPUs round to nearest even).
    fn f2h(f: f32) -> u16 {
        let mut x = f.to_bits();
        let sign = (x >> 16) & 0x8000;
        x &= 0x7fff_ffff;
        let h = if x >= 0x4780_0000 {
            if x > 0x7f80_0000 {
                0x7e00
            } else {
                0x7c00
            }
        } else if x < 0x3880_0000 {
            (f32::from_bits(x) + 0.5)
                .to_bits()
                .wrapping_sub(0x3f00_0000)
        } else {
            x.wrapping_add(0xc800_0fff).wrapping_add((x >> 13) & 1) >> 13
        };
        (sign | h) as u16
    }

    #[test]
    fn f16_conversion_rounds_to_nearest_even() {
        let check = |f: f32| {
            let want = half::f16::from_f32(f);
            let got = f2h(f);
            if want.is_nan() {
                assert!(
                    half::f16::from_bits(got).is_nan(),
                    "{f:e}: {got:#06x} not NaN"
                );
            } else {
                assert_eq!(got, want.to_bits(), "{f:e} ({:#010x})", f.to_bits());
            }
        };
        // every f16 halfway point (and its neighbours), both signs
        for m in 0u32..(1 << 16) {
            let tie = (m << 13) | 0x1000;
            for b in [tie - 1, tie, tie + 1] {
                check(f32::from_bits(b));
                check(f32::from_bits(b | 0x8000_0000));
            }
        }
        // a pseudo-random sweep of all f32 bit patterns
        let mut s = 0x9e37_79b9u32;
        for _ in 0..2_000_000 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            check(f32::from_bits(s));
        }
        for f in [
            0.0,
            -0.0,
            65504.0,
            65519.99,
            65520.0,
            6.1e-5,
            5.96e-8,
            2.98e-8,
            f32::INFINITY,
        ] {
            check(f);
        }
    }

    #[test]
    fn transliterates_loads_shuffles_and_stores() {
        let src = format!(
            "// k\n#define TANG_TILES 4\n#define TANG_THREADS 64\n{PRELUDE}extern \"C\" __global__ void __launch_bounds__(64) main0(const float* __restrict__ b3, const float* __restrict__ b5, float* __restrict__ out, unsigned count) {{\n    __shared__ float part[2][1];\n    float4 v0[1][1];\n    v0[0][0] = __ldg((const float4*)(b3 + (2 + (int)(x))) + 1);\n    uint4 u = __ldg((const uint4*)b5 + ((int)(k) >> 5));\n    float y = (&v0[0][0].x)[e & 3] + __ldg(b3 + (7));\n    y = __shfl_xor_sync(0xffffffffu, y, 4);\n    out[0] = y + tang_u4(b5, 9);\n}}\n"
        );
        let g = transliterate(&src, &[3, 5], None);
        assert!(g.starts_with(GLSL_VERSION), "{g}");
        assert!(g.contains("layout(local_size_x = 64) in;"));
        assert!(g.contains("readonly buffer B3 { uint b3[]; }"));
        assert!(g.contains("shared float part[2][1];"));
        assert!(
            g.contains("vec4(uintBitsToFloat(b3[(((2 + int(x))) + 4 * (1)) + 0]"),
            "{g}"
        );
        assert!(g.contains("uvec4(b5[(4 * ((int(k) >> 5))) + 0]"), "{g}");
        assert!(
            g.contains("v0[0][0][e & 3] + uintBitsToFloat(b3[(7)])"),
            "{g}"
        );
        assert!(g.contains("subgroupShuffleXor(y, 4)"));
        assert!(
            g.contains("tang_out[0] = floatBitsToUint(y + tang_u4(b5, 9));"),
            "{g}"
        );
    }
}
