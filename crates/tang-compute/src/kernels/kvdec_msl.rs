//! Compressed-KV decode kernels (MSL), built on [`super::llm_msl::kv_source`] so they write
//! straight into f32 or bf16 KV caches.
//!
//! - `tkv_decode`: the GPU layout of `tang_compress::kvcodec` (TKV1, flag `gpu`). One thread
//!   per lane = one (stream, block, head) adaptive-binary-arithmetic substream; tables from
//!   `tang_compress::kvgpu::Batch`. Integer only up to the final `float(q) * unit`, so with RoPE
//!   off and an f32 destination the output is bit-identical to the CPU decoder.
//! - `kivi_dequant`: KIVI-style asymmetric integer groups (2/4/8-bit codes packed in `u32`,
//!   half scale and zero per group; K grouped per channel along tokens, V per token along
//!   channels). One thread per value.

/// See the module docs. Lane descriptor layout: `tang_compress::kvgpu::LANE_WORDS`.
pub const KVDEC_MSL: &str = r#"
inline uint tkv_rb(device const uint* s, uint a, uint end) {
    return a < end ? (s[a >> 2] >> ((a & 3u) * 8u)) & 0xffu : 0u;
}
struct TkvRC { uint code; uint range; uint pos; uint end; };
inline bool tkv_dp(thread TkvRC& d, device const uint* s, uint p1) {
    p1 = clamp(p1, 32u, 65504u);
    uint bound = (d.range >> 16) * (65536u - p1);
    bool bit = d.code >= bound;
    if (bit) { d.code -= bound; d.range -= bound; } else { d.range = bound; }
    while (d.range < (1u << 24)) {
        d.range <<= 8;
        d.code = (d.code << 8) | tkv_rb(s, d.pos, d.end);
        d.pos++;
    }
    return bit;
}
inline bool tkv_dm(thread TkvRC& d, device const uint* s, device uint* m) {
    uint w = *m;
    uint f = w & 0xffffu, sl = w >> 16;
    bool b = tkv_dp(d, s, (f + sl) >> 1);
    if (b) { f += (65535u - f) >> 4; sl += (65535u - sl) >> 7; }
    else { f -= f >> 4; sl -= sl >> 7; }
    *m = f | (sl << 16);
    return b;
}
inline int tkv_get(thread TkvRC& d, device const uint* s, device uint* m) {
    if (tkv_dm(d, s, m)) return 0;
    bool neg = tkv_dp(d, s, 32768u);
    uint k = 0;
    while (k < 62u && tkv_dm(d, s, m + 1u + min(k, 23u))) k++;
    uint w = 1u;
    for (int i = int(k) - 1; i >= 0; i--) {
        bool bit = (uint(i) + 1u == k) ? tkv_dm(d, s, m + 25u + min(k, 23u)) : tkv_dp(d, s, 32768u);
        w = (w << 1) | uint(bit);
    }
    return neg ? -int(w) : int(w);
}

kernel void tkv_decode(
    device const uint* S [[buffer(0)]],    // stream words
    device const uint* L [[buffer(1)]],    // lanes, 14 words each
    device const int4* C [[buffer(2)]],    // per channel: mu, rho, class, dense class
    device const uint* P [[buffer(3)]],    // prior table [class][act][49]
    device uint* M [[buffer(4)]],          // model scratch
    device kv_t* O [[buffer(5)]],
    device const float* COS [[buffer(6)]], // [pos][hd/2] when a lane has RoPE on
    device const float* SIN [[buffer(7)]],
    constant uint& nl [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= nl) return;
    device const uint* l = L + gid * 14u;
    uint pos0 = l[2], n = l[3], ch = l[4], tstride = l[6], hd = l[11];
    ulong out = l[5];
    device uint* m = M + l[7];
    float u = as_type<float>(l[8]);
    uint sink = l[9];
    int sdiv = int(l[10]);
    bool rope = l[12] != 0u;
    for (uint c = 0; c < hd; c++) {
        int4 q = C[ch + c];
        uint dc = uint(q.w) * 147u, cl = uint(q.z) * 147u;
        for (uint k = 0; k < 147u; k++) m[dc + k] = P[cl + k];
    }
    TkvRC d;
    d.pos = l[0];
    d.end = l[0] + l[1];
    d.range = 0xffffffffu;
    d.code = 0u;
    for (int i = 0; i < 4; i++) { d.code = (d.code << 8) | tkv_rb(S, d.pos, d.end); d.pos++; }
    int prev[256];
    uint act[256];
    float row[256];
    for (uint c = 0; c < hd; c++) act[c] = 0u;
    uint half_dim = hd / 2u;
    for (uint t = 0; t < n; t++) {
        int ts = (pos0 + t < sink) ? 1 : sdiv;
        for (uint c = 0; c < hd; c++) {
            int4 q = C[ch + c];
            int mu = q.x;
            int pred = t == 0u ? mu : mu + ((q.y * (prev[c] - mu)) >> 4);
            int qq = tkv_get(d, S, m + (uint(q.w) * 3u + act[c]) * 49u);
            int r = pred + qq * ts;
            prev[c] = r;
            act[c] = min(uint(abs(qq)), 2u);
            row[c] = float(r) * u;
        }
        device kv_t* dst = O + out + (ulong)t * tstride;
        if (rope) {
            uint tc = (pos0 + t) * half_dim;
            for (uint i = 0; i < half_dim; i++) {
                float cs = COS[tc + i], sn = SIN[tc + i];
                float x0 = row[i], x1 = row[i + half_dim];
                kv_st(dst + i, x0 * cs - x1 * sn);
                kv_st(dst + i + half_dim, x1 * cs + x0 * sn);
            }
        } else {
            for (uint c = 0; c < hd; c++) kv_st(dst + c, row[c]);
        }
    }
}

// KIVI-style dequant. params: [bits, g, nkv, hd, tokens, is_key, page shift].
// Codes in value order [t][h][c]; groups: K (t/g, h, c), V (t, h, c/g); SZ = (scale, zero).
kernel void kivi_dequant(
    device const uint* Q [[buffer(0)]],
    device const half2* SZ [[buffer(1)]],
    device kv_t* O [[buffer(2)]],
    device const uint* p [[buffer(3)]],
    device const uint* T [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint bits = p[0], g = p[1], nkv = p[2], hd = p[3], ntok = p[4];
    uint kvd = nkv * hd;
    if (gid >= ntok * kvd) return;
    uint t = gid / kvd, hc = gid % kvd;
    uint b = gid * bits;
    uint code = (Q[b >> 5] >> (b & 31u)) & ((1u << bits) - 1u);
    uint gi = p[5] != 0u ? (t / g) * kvd + hc : t * (kvd / g) + hc / g;
    float2 sz = float2(SZ[gi]);
    kv_st(O + kv_row(T, p[6], t) * kvd + hc, float(code) * sz.x + sz.y);
}
"#;
