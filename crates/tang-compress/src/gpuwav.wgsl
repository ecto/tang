// tc-gpu decode kernels (hand-written WGSL, integer only).
//
// Placeholders replaced by the host before compilation: LANES, NMODEL,
// SIG0, MAG0, LOW0, SGN0, ESC0, KB_, NE_, NA_, NPB_, MAXL, K0..K5, SHS, BS,
// BRS, NKEY_.
//
// Every operation here mirrors `gpuwav.rs` exactly: u32/i32 wrapping
// arithmetic, arithmetic right shifts on i32, and an emulated 32x32->64
// product for the fixed-point multiplies.

struct Params {
    off_shapes: u32, off_bands: u32, off_preds: u32, off_init: u32,
    off_dl: u32, off_bricks: u32, off_esc: u32, nesc: u32,
    nbricks: u32, brick_base: u32, batch: u32, level: u32,
    axis: u32, mean_fix: u32, scale_bits: u32, corr: u32,
    d0: u32, d1: u32, d2: u32, bd0: u32,
    bd1: u32, bd2: u32, nb0: u32, nb1: u32,
    nb2: u32, ncells: u32, pad0: u32, pad1: u32,
    pad2: u32, pad3: u32, pad4: u32, pad5: u32,
}

@group(0) @binding(0) var<storage, read> mt: array<u32>;
@group(0) @binding(1) var<storage, read> stream: array<u32>;
@group(0) @binding(2) var<storage, read_write> models: array<u32>;
@group(0) @binding(3) var<storage, read_write> coef: array<i32>;
@group(0) @binding(4) var<storage, read_write> outb: array<u32>;
@group(0) @binding(5) var<uniform> P: Params;

var<workgroup> wg_u: u32;

// ---------------------------------------------------------------- rANS lane

var<private> rx: u32;
var<private> rpos: u32;
var<private> rend: u32;

fn word(i: u32) -> u32 {
    if (i >= rend) {
        return 0u;
    }
    return (stream[i >> 1u] >> ((i & 1u) * 16u)) & 0xFFFFu;
}

fn rstep(cum: u32, f: u32) {
    rx = f * (rx >> 16u) + ((rx & 0xFFFFu) - cum);
    if (rx < 65536u) {
        rx = (rx << 16u) | word(rpos);
        rpos = rpos + 1u;
    }
}

fn lane_start(lo: u32, hi: u32) {
    rend = hi;
    if (hi - lo >= 2u) {
        rx = word(lo) | (word(lo + 1u) << 16u);
    } else {
        rx = 65536u;
    }
    rpos = lo + 2u;
}

fn mp1(m: u32) -> u32 {
    return clamp(((m & 0xFFFFu) + (m >> 16u)) >> 1u, 32u, 65504u);
}

fn mupd(m: u32, b: bool) -> u32 {
    var f = m & 0xFFFFu;
    var s = m >> 16u;
    if (b) {
        f = f + ((65535u - f) >> 4u);
        s = s + ((65535u - s) >> 7u);
    } else {
        f = f - (f >> 4u);
        s = s - (s >> 7u);
    }
    return f | (s << 16u);
}

fn dbit_p(p1: u32) -> bool {
    let f0 = 65536u - p1;
    let b = (rx & 0xFFFFu) >= f0;
    if (b) {
        rstep(f0, p1);
    } else {
        rstep(0u, f0);
    }
    return b;
}


fn draw(n: u32) -> u32 {
    let v = (rx & 0xFFFFu) >> (16u - n);
    rstep(v << (16u - n), 1u << (16u - n));
    return v;
}

fn draw_n(n0: u32) -> u32 {
    var n = n0;
    var o = 0u;
    while (n > 16u) {
        o = (o << 16u) | draw(16u);
        n = n - 16u;
    }
    if (n > 0u) {
        o = (o << n) | draw(n);
    }
    return o;
}

// static models: P(1) of model `idx` is the transmitted prior
fn sbit(idx: u32) -> bool {
    return dbit_p(mp1(mt[P.off_init + idx]));
}

fn dgamma_s(base: u32) -> u32 {
    var nb = 0u;
    loop {
        if (nb >= 31u) { break; }
        if (!sbit(base + nb)) { break; }
        nb = nb + 1u;
    }
    let lo = draw_n(nb);
    return ((1u << nb) | lo) - 1u;
}

// ---------------------------------------------------------------- 64-bit

fn mul64(a: i32, b: i32) -> vec2<u32> {
    let neg = (a < 0) != (b < 0);
    let ua = u32(abs(a));
    let ub = u32(abs(b));
    let a0 = ua & 0xFFFFu;
    let a1 = ua >> 16u;
    let b0 = ub & 0xFFFFu;
    let b1 = ub >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    var lo = (p00 & 0xFFFFu) | (mid << 16u);
    var hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    if (neg) {
        lo = ~lo + 1u;
        hi = ~hi + select(0u, 1u, lo == 0u);
    }
    return vec2<u32>(lo, hi);
}

// floor((a*b + 2^(s-1)) / 2^s), low 32 bits; 1 <= s <= 62
fn mulsh(a: i32, b: i32, s: u32) -> i32 {
    let v = mul64(a, b);
    var lo = v.x;
    var hi = v.y;
    let s1 = s - 1u;
    if (s1 < 32u) {
        let r = 1u << s1;
        let nl = lo + r;
        hi = hi + select(0u, 1u, nl < lo);
        lo = nl;
    } else {
        hi = hi + (1u << (s1 - 32u));
    }
    if (s < 32u) {
        return bitcast<i32>((lo >> s) | (hi << (32u - s)));
    }
    return bitcast<i32>(hi) >> (s - 32u);
}

// ---------------------------------------------------------------- tables

const A_TH = array<u32, 11>(0u, 1u, 2u, 3u, 5u, 7u, 10u, 14u, 20u, 30u, 48u);
const P_TH = array<u32, 3>(0u, 2u, 6u);
const ROWNB_DZ = array<i32, 20>(0, 0, -1, 1, -1, -1, 1, 1, 0, 0, -2, 2, -2, -2, 2, 2, 0, 0, -4, 4);
const ROWNB_DY = array<i32, 20>(-1, 1, 0, 0, -1, 1, -1, 1, -2, 2, 0, 0, -2, 2, -2, 2, -4, 4, 0, 0);
const ROWNB_CAT = array<u32, 20>(0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u);

fn bucket_a(a: u32) -> u32 {
    for (var i = 0u; i < 11u; i = i + 1u) {
        if (a <= A_TH[i]) { return i; }
    }
    return 11u;
}

fn bucket_p(a: u32) -> u32 {
    for (var i = 0u; i < 3u; i = i + 1u) {
        if (a <= P_TH[i]) { return i; }
    }
    return 3u;
}

fn pass_of(npass: u32, zy_is_y: bool, zr: u32, yr: u32) -> u32 {
    if (npass == 7u) {
        if (((yr | zr) & 1u) == 0u) {
            let t = ((yr >> 1u) & 1u) | (((zr >> 1u) & 1u) << 1u);
            return select(select(select(1u, 3u, t == 2u), 2u, t == 1u), 0u, t == 0u);
        }
        let t = (yr & 1u) | ((zr & 1u) << 1u);
        return select(select(4u, 6u, t == 2u), 5u, t == 1u);
    }
    if (npass == 4u) {
        let t = select(zr, yr, zy_is_y);
        if ((t & 1u) == 1u) { return 3u; }
        if ((t & 3u) == 2u) { return 2u; }
        if ((t & 7u) == 4u) { return 1u; }
        return 0u;
    }
    return 0u;
}

fn cap(v: i32) -> u32 {
    return min(u32(abs(v)), 65535u);
}

fn sgn3(v: i32) -> u32 {
    return u32(sign(v) + 1);
}

// band record field
fn bf(b: u32, f: u32) -> u32 {
    return mt[P.off_bands + b * BSu + f];
}


// ---------------------------------------------------------------- decode

var<private> rb_base: array<u32, 20>;
var<private> rb_cat: array<u32, 20>;

// lattice of pass p: (rz, sz, ry, sy)
fn lattice(npass: u32, zy_is_y: bool, p: u32) -> vec4<u32> {
    if (npass == 7u) {
        let t = array<vec4<u32>, 7>(vec4<u32>(0u, 4u, 0u, 4u), vec4<u32>(2u, 4u, 2u, 4u), vec4<u32>(0u, 4u, 2u, 4u),
                                    vec4<u32>(2u, 4u, 0u, 4u), vec4<u32>(1u, 2u, 1u, 2u), vec4<u32>(0u, 2u, 1u, 2u),
                                    vec4<u32>(1u, 2u, 0u, 2u));
        return t[p];
    }
    if (npass == 4u) {
        let r = array<u32, 4>(0u, 4u, 2u, 1u);
        let st = array<u32, 4>(8u, 8u, 4u, 2u);
        if (zy_is_y) { return vec4<u32>(0u, 1u, r[p], st[p]); }
        return vec4<u32>(r[p], st[p], 0u, 1u);
    }
    return vec4<u32>(0u, 1u, 0u, 1u);
}

@compute @workgroup_size(LANES_WG)
fn decode(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let bk = P.brick_base + wid.x;
    let lane = lid.x;
    let rec = P.off_bricks + bk * BRSu;
    let shape = mt[rec + 3u];
    let cbase = mt[rec + 4u];
    let ltab = mt[rec + 5u];
    let srec = P.off_shapes + shape * SHSu;
    let ny = mt[srec + 1u];
    let nx = mt[srec + 2u];
    let boff = mt[srec + 4u];
    lane_start(mt[ltab + lane], mt[ltab + lane + 1u]);
    if (lane == 0u) { wg_u = mt[srec + 3u]; }
    let nbands = workgroupUniformLoad(&wg_u);
    for (var bi = 0u; bi < nbands; bi = bi + 1u) {
        let b = boff + bi;
        if (lane == 0u) { wg_u = bf(b, 9u); }
        let npass = workgroupUniformLoad(&wg_u);
        let z0 = bf(b, 0u); let y0 = bf(b, 1u); let x0 = bf(b, 2u);
        let z1 = bf(b, 3u); let y1 = bf(b, 4u); let x1 = bf(b, 5u);
        let cls = bf(b, 6u);
        let par = bf(b, 7u);
        let psh = bf(b, 8u);
        let ez = z1 - z0;
        let ey = y1 - y0;
        let zy_is_y = ey > 1u;
        var pst = array<u32, 3>(0u, 0u, 0u);
        var pen = array<u32, 3>(1u, 1u, 1u);
        if (par != 0xFFFFFFFFu) {
            let pb = boff + par;
            pst = array<u32, 3>(bf(pb, 0u), bf(pb, 1u), bf(pb, 2u));
            pen = array<u32, 3>(bf(pb, 3u), bf(pb, 4u), bf(pb, 5u));
        }
        let psx = (psh >> 2u) & 1u;
        let sz_ = ny * nx;
        for (var p = 0u; p < npass; p = p + 1u) {
            let pk = P.off_preds + bf(b, 10u + p) * 6u;
            let w0 = bitcast<i32>(mt[pk]); let w1 = bitcast<i32>(mt[pk + 1u]); let w2 = bitcast<i32>(mt[pk + 2u]);
            let w3 = bitcast<i32>(mt[pk + 3u]); let w4 = bitcast<i32>(mt[pk + 4u]); let w5 = bitcast<i32>(mt[pk + 5u]);
            let lt = lattice(npass, zy_is_y, p);
            var nrows = 0u;
            var cy = 1u;
            if (lt.x < ez && lt.z < ey) {
                cy = (ey - lt.z + lt.w - 1u) / lt.w;
                nrows = ((ez - lt.x + lt.y - 1u) / lt.y) * cy;
            }
            for (var j = lane; j < nrows; j = j + LANESu) {
                let z = z0 + lt.x + (j / cy) * lt.y;
                let y = y0 + lt.z + (j % cy) * lt.w;
                // visible lines
                var nr = 0u;
                for (var t = 0u; t < 20u; t = t + 1u) {
                    let z2 = i32(z) + ROWNB_DZ[t];
                    let y2 = i32(y) + ROWNB_DY[t];
                    if (z2 >= i32(z0) && z2 < i32(z1) && y2 >= i32(y0) && y2 < i32(y1)) {
                        if (pass_of(npass, zy_is_y, u32(z2) - z0, u32(y2) - y0) < p) {
                            rb_base[nr] = cbase + (u32(z2) * ny + u32(y2)) * nx;
                            rb_cat[nr] = ROWNB_CAT[t];
                            nr = nr + 1u;
                        }
                    }
                }
                var prow = 0u;
                var pz = 0u;
                var py = 0u;
                if (par != 0xFFFFFFFFu) {
                    pz = min(pst[0] + ((z - z0) >> (psh & 1u)), pen[0] - 1u);
                    py = min(pst[1] + ((y - y0) >> ((psh >> 1u) & 1u)), pen[1] - 1u);
                    prow = cbase + (pz * ny + py) * nx;
                }
                let rowb = cbase + (z * ny + y) * nx;
                var wv = 0;
                var wwv = 0;
                for (var x = x0; x < x1; x = x + 1u) {
                    var sf = 0; var sd = 0; var sr = 0;
                    var af = 0u; var ad = 0u; var ar = 0u;
                    var nf = 0u; var nd = 0u; var nrr = 0u;
                    for (var t = 0u; t < nr; t = t + 1u) {
                        let v = coef[rb_base[t] + x];
                        let c = rb_cat[t];
                        if (c == 0u) { sf = sf + v; af = af + cap(v); nf = nf + 1u; }
                        else if (c == 1u) { sd = sd + v; ad = ad + cap(v); nd = nd + 1u; }
                        else { sr = sr + v; ar = ar + cap(v); nrr = nrr + 1u; }
                    }
                    var pv = 0;
                    var pnb = 0u;
                    var has_p = 0u;
                    if (par != 0xFFFFFFFFu) {
                        let px = min(pst[2] + ((x - x0) >> psx), pen[2] - 1u);
                        let pi = prow + px;
                        pv = coef[pi];
                        has_p = 1u;
                        if (pz > pst[0]) { pnb = pnb + cap(coef[pi - sz_]); }
                        if (pz + 1u < pen[0]) { pnb = pnb + cap(coef[pi + sz_]); }
                        if (py > pst[1]) { pnb = pnb + cap(coef[pi - nx]); }
                        if (py + 1u < pen[1]) { pnb = pnb + cap(coef[pi + nx]); }
                        if (px > pst[2]) { pnb = pnb + cap(coef[pi - 1u]); }
                        if (px + 1u < pen[2]) { pnb = pnb + cap(coef[pi + 1u]); }
                    }
                    let acc = w0 * clamp(wv, -8192, 8192) + w1 * clamp(wwv, -8192, 8192) + w2 * clamp(sf, -8192, 8192)
                            + w3 * clamp(sd, -8192, 8192) + w4 * clamp(sr, -8192, 8192) + w5 * clamp(pv, -8192, 8192);
                    let pred = (acc + 2048) >> 12u;
                    let s = 2u * cap(wv) + cap(wwv) + 2u * af + ad + ar + 2u * cap(pv) + 2u * cap(pred);
                    let ws = 3u + 2u * nf + nd + nrr + 2u * has_p + 2u;
                    let ab = bucket_a(12u * s / ws);
                    var pbk = 0u;
                    if (has_p == 1u) { pbk = 1u + bucket_p(4u * min(cap(pv), 15u) + min(pnb, 60u)); }
                    let g = select(0u, 1u, nf + nd + nrr > 0u);
                    let e = min(firstLeadingBit(1u + 2u * s / ws), NE_u - 1u);
                    let sig = ((cls * 2u + g) * NA_u + ab) * NPB_u + pbk;
                    var r = 0;
                    if (sbit(SIG0u + sig)) {
                        let k = max(e, 2u) - 2u;
                        var h = 0u;
                        let mbase = MAG0u + (cls * NE_u + e) * KB_u;
                        loop {
                            if (h >= KB_u) { break; }
                            if (!sbit(mbase + h)) { break; }
                            h = h + 1u;
                        }
                        var hv = h;
                        if (h == KB_u) { hv = hv + dgamma_s(ESC0u + cls * 32u); }
                        var lo = 0u;
                        if (k >= 1u) {
                            lo = select(0u, 1u, sbit(LOW0u + e * 4u));
                        }
                        if (k >= 2u) {
                            lo = (lo << 1u) | select(0u, 1u, sbit(LOW0u + e * 4u + 2u + (lo & 1u)));
                        }
                        if (k >= 3u) {
                            lo = (lo << (k - 2u)) | draw_n(k - 2u);
                        }
                        let mv = ((hv << k) | lo) + 1u;
                        let sctx = cls * 9u + sgn3(wv) * 3u + sgn3(sf + sd);
                        if (sbit(SGN0u + sctx)) { r = -bitcast<i32>(mv); } else { r = bitcast<i32>(mv); }
                    }
                    let qv = r + pred;
                    coef[rowb + x] = qv;
                    wwv = wv;
                    wv = qv;
                }
            }
            storageBarrier();
            workgroupBarrier();
        }
    }
    // dequantize in place
    for (var bi = 0u; bi < nbands; bi = bi + 1u) {
        let b = boff + bi;
        let z0 = bf(b, 0u); let y0 = bf(b, 1u); let x0 = bf(b, 2u);
        let ez = bf(b, 3u) - z0; let ey = bf(b, 4u) - y0; let ex = bf(b, 5u) - x0;
        let cls = bf(b, 6u);
        let m = bitcast<i32>(bf(b, 18u));
        let sh = bf(b, 19u);
        let n = ez * ey * ex;
        for (var i = lane; i < n; i = i + LANESu) {
            let x = x0 + i % ex;
            let y = y0 + (i / ex) % ey;
            let z = z0 + i / (ex * ey);
            let idx = cbase + (z * ny + y) * nx + x;
            let v = coef[idx];
            if (v != 0) {
                let av = u32(abs(v));
                let a = bitcast<i32>(min(av, 0x7FFFFFFFu));
                let d = bitcast<i32>(mt[P.off_dl + cls * 3u + min(av, 3u) - 1u]);
                var c = mulsh(a, m, sh) - mulsh(d, m, sh + 9u);
                if (v < 0) { c = -c; }
                coef[idx] = c;
            }
        }
    }
}

// ---------------------------------------------------------------- inverse lifting

const KK = array<i32, 6>(KK_LIST);

@compute @workgroup_size(64)
fn lift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bk = P.brick_base + gid.y;
    if (bk >= P.nbricks) { return; }
    let rec = P.off_bricks + bk * BRSu;
    let shape = mt[rec + 3u];
    let cbase = mt[rec + 4u];
    let srec = P.off_shapes + shape * SHSu;
    let l = P.level;
    let ax = P.axis;
    if (l >= mt[srec + 5u]) { return; }
    let smask = mt[srec + 6u + 45u + l];
    if (((smask >> ax) & 1u) == 0u) { return; }
    let dims = array<u32, 3>(mt[srec], mt[srec + 1u], mt[srec + 2u]);
    let ext = array<u32, 3>(mt[srec + 6u + l * 3u], mt[srec + 7u + l * 3u], mt[srec + 8u + l * 3u]);
    let st = array<u32, 3>(dims[1] * dims[2], dims[2], 1u);
    var o0 = 0u;
    var o1 = 1u;
    if (ax == 0u) { o0 = 1u; o1 = 2u; } else if (ax == 1u) { o0 = 0u; o1 = 2u; }
    let nl = ext[o0] * ext[o1];
    let line = gid.x;
    if (line >= nl) { return; }
    let off = cbase + (line / ext[o1]) * st[o0] + (line % ext[o1]) * st[o1];
    let sa = st[ax];
    let n = ext[ax];
    if (n < 2u) { return; }
    let ns = (n + 1u) / 2u;
    let nd = n / 2u;
    var t: array<i32, MAXL_N>;
    for (var i = 0u; i < n; i = i + 1u) { t[i] = coef[off + i * sa]; }
    for (var i = 0u; i < ns; i = i + 1u) { t[i] = mulsh(t[i], KK[0], 24u); }
    for (var i = 0u; i < nd; i = i + 1u) { t[ns + i] = mulsh(t[ns + i], KK[1], 24u); }
    for (var step = 0u; step < 4u; step = step + 1u) {
        let c = KK[2u + step];
        if ((step & 1u) == 0u) {
            // s_i += c (d_{i-1} + d_i)
            for (var i = 0u; i < ns; i = i + 1u) {
                var lv = t[ns];
                if (i > 0u) { lv = t[ns + i - 1u]; }
                var rv = t[ns + nd - 1u];
                if (i < nd) { rv = t[ns + i]; }
                t[i] = t[i] + mulsh(lv + rv, c, 24u);
            }
        } else {
            // d_i += c (s_i + s_{i+1})
            for (var i = 0u; i < nd; i = i + 1u) {
                var rv = t[ns - 1u];
                if (i + 1u < ns) { rv = t[i + 1u]; }
                t[ns + i] = t[ns + i] + mulsh(t[i] + rv, c, 24u);
            }
        }
    }
    for (var i = 0u; i < ns; i = i + 1u) { coef[off + 2u * i * sa] = t[i]; }
    for (var i = 0u; i < nd; i = i + 1u) { coef[off + (2u * i + 1u) * sa] = t[ns + i]; }
}

// ---------------------------------------------------------------- outliers

var<private> om: array<u32, 96>;

fn obit(idx: u32) -> bool {
    let m = om[idx];
    let b = dbit_p(mp1(m));
    om[idx] = mupd(m, b);
    return b;
}

fn ogamma(base: u32) -> u32 {
    var nb = 0u;
    loop {
        if (nb >= 31u) { break; }
        if (!obit(base + nb)) { break; }
        nb = nb + 1u;
    }
    let lo = draw_n(nb);
    return ((1u << nb) | lo) - 1u;
}

@compute @workgroup_size(64)
fn outliers(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bk = gid.x;
    if (bk >= P.nbricks) { return; }
    let rec = P.off_bricks + bk * BRSu;
    let cbase = mt[rec + 4u];
    let ltab = mt[rec + 5u];
    let shape = mt[rec + 3u];
    let srec = P.off_shapes + shape * SHSu;
    let ncell = mt[srec] * mt[srec + 1u] * mt[srec + 2u];
    for (var i = 0u; i < 96u; i = i + 1u) { om[i] = 0x80008000u; }
    lane_start(mt[ltab + LANESu], mt[ltab + LANESu + 1u]);
    let n = ogamma(0u);
    var last = -1;
    let corr = bitcast<i32>(P.corr);
    for (var i = 0u; i < n; i = i + 1u) {
        let g = ogamma(32u);
        let idx = u32(last + 1 + bitcast<i32>(g));
        last = bitcast<i32>(idx);
        let neg = draw(1u) == 1u;
        var k = bitcast<i32>(ogamma(64u) + 1u);
        if (neg) { k = -k; }
        if (idx < ncell) {
            coef[cbase + idx] = coef[cbase + idx] + k * corr;
        }
    }
}

// ---------------------------------------------------------------- output

@compute @workgroup_size(256)
fn finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x + gid.y * 65535u * 256u;
    if (g >= P.ncells) { return; }
    let x = g % P.d2;
    let y = (g / P.d2) % P.d1;
    let z = g / (P.d2 * P.d1);
    let bz = z / P.bd0;
    let by = y / P.bd1;
    let bx = x / P.bd2;
    let bk = (bz * P.nb1 + by) * P.nb2 + bx;
    let rec = P.off_bricks + bk * BRSu;
    let shape = mt[rec + 3u];
    let cbase = mt[rec + 4u];
    let srec = P.off_shapes + shape * SHSu;
    let lz = z - mt[rec];
    let ly = y - mt[rec + 1u];
    let lx = x - mt[rec + 2u];
    let li = (lz * mt[srec + 1u] + ly) * mt[srec + 2u] + lx;
    let v = clamp(coef[cbase + li] + bitcast<i32>(P.mean_fix), -16777215, 16777215);
    outb[g] = bitcast<u32>(f32(v) * bitcast<f32>(P.scale_bits));
}

@compute @workgroup_size(64)
fn escapes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.nesc) { return; }
    outb[mt[P.off_esc + 2u * i]] = mt[P.off_esc + 2u * i + 1u];
}

// ================================================================ v3 (code-blocks)
//
// One global pyramid (single brick record / shape record), code-blocks
// decoded one per invocation, dispatched level by level (coarse -> fine).
// The `models` binding holds the residual buffer (i32 bits).
// P.pad0 = off_blocks, P.pad1 = resctx, P.pad2 = off_tiles, P.pad3 = ntiles.

fn mmag(i: u32, q: i32) -> i32 {
    if (P.pad1 != 0u) { return bitcast<i32>(models[i]); }
    return q;
}

@compute @workgroup_size(64)
fn decode3(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.batch) { return; }
    let rec = P.pad0 + (P.brick_base + gid.x) * 10u;
    let b = mt[rec];
    let z0 = mt[rec + 1u]; let y0 = mt[rec + 2u]; let x0 = mt[rec + 3u];
    let z1 = mt[rec + 4u]; let y1 = mt[rec + 5u]; let x1 = mt[rec + 6u];
    lane_start(mt[rec + 7u], mt[rec + 8u]);
    let ny = P.d1;
    let nx = P.d2;
    let org = array<u32, 3>(bf(b, 0u), bf(b, 1u), bf(b, 2u));
    let cls = bf(b, 6u);
    let par = bf(b, 7u);
    let psh = bf(b, 8u);
    let pk = P.off_preds + bf(b, 10u) * 6u;
    let w0 = bitcast<i32>(mt[pk]); let w1 = bitcast<i32>(mt[pk + 1u]); let w2 = bitcast<i32>(mt[pk + 2u]);
    let w3 = bitcast<i32>(mt[pk + 3u]); let w4 = bitcast<i32>(mt[pk + 4u]); let w5 = bitcast<i32>(mt[pk + 5u]);
    var pst = array<u32, 3>(0u, 0u, 0u);
    var pen = array<u32, 3>(1u, 1u, 1u);
    if (par != 0xFFFFFFFFu) {
        pst = array<u32, 3>(bf(par, 0u), bf(par, 1u), bf(par, 2u));
        pen = array<u32, 3>(bf(par, 3u), bf(par, 4u), bf(par, 5u));
    }
    let psx = (psh >> 2u) & 1u;
    let sz_ = ny * nx;
    for (var z = z0; z < z1; z = z + 1u) {
        for (var y = y0; y < y1; y = y + 1u) {
            var nr = 0u;
            for (var t = 0u; t < 20u; t = t + 1u) {
                let z2 = i32(z) + ROWNB_DZ[t];
                let y2 = i32(y) + ROWNB_DY[t];
                if (z2 >= i32(z0) && z2 < i32(z1) && y2 >= i32(y0) && y2 < i32(y1)) {
                    if (z2 < i32(z) || (z2 == i32(z) && y2 < i32(y))) {
                        rb_base[nr] = (u32(z2) * ny + u32(y2)) * nx;
                        rb_cat[nr] = ROWNB_CAT[t];
                        nr = nr + 1u;
                    }
                }
            }
            var prow = 0u;
            var pz = 0u;
            var py = 0u;
            if (par != 0xFFFFFFFFu) {
                pz = min(pst[0] + ((z - org[0]) >> (psh & 1u)), pen[0] - 1u);
                py = min(pst[1] + ((y - org[1]) >> ((psh >> 1u) & 1u)), pen[1] - 1u);
                prow = (pz * ny + py) * nx;
            }
            let rowb = (z * ny + y) * nx;
            var wv = 0;
            var wwv = 0;
            var wm = 0;
            var wwm = 0;
            for (var x = x0; x < x1; x = x + 1u) {
                var sf = 0; var sd = 0; var sr = 0;
                var af = 0u; var ad = 0u; var ar = 0u;
                var nf = 0u; var nd = 0u; var nrr = 0u;
                for (var t = 0u; t < nr; t = t + 1u) {
                    let i = rb_base[t] + x;
                    let v = coef[i];
                    let vm = mmag(i, v);
                    let c = rb_cat[t];
                    if (c == 0u) {
                        sf = sf + v; af = af + cap(vm); nf = nf + 1u;
                        if (x > x0) { ad = ad + cap(mmag(i - 1u, coef[i - 1u])); nd = nd + 1u; }
                        if (x + 1u < x1) { ad = ad + cap(mmag(i + 1u, coef[i + 1u])); nd = nd + 1u; }
                    }
                    else if (c == 1u) { sd = sd + v; ad = ad + cap(vm); nd = nd + 1u; }
                    else { sr = sr + v; ar = ar + cap(vm); nrr = nrr + 1u; }
                }
                var pv = 0;
                var pm = 0;
                var pnb = 0u;
                var has_p = 0u;
                if (par != 0xFFFFFFFFu) {
                    let px = min(pst[2] + ((x - org[2]) >> psx), pen[2] - 1u);
                    let pi = prow + px;
                    pv = coef[pi];
                    pm = mmag(pi, pv);
                    has_p = 1u;
                    if (pz > pst[0]) { pnb = pnb + cap(mmag(pi - sz_, coef[pi - sz_])); }
                    if (pz + 1u < pen[0]) { pnb = pnb + cap(mmag(pi + sz_, coef[pi + sz_])); }
                    if (py > pst[1]) { pnb = pnb + cap(mmag(pi - nx, coef[pi - nx])); }
                    if (py + 1u < pen[1]) { pnb = pnb + cap(mmag(pi + nx, coef[pi + nx])); }
                    if (px > pst[2]) { pnb = pnb + cap(mmag(pi - 1u, coef[pi - 1u])); }
                    if (px + 1u < pen[2]) { pnb = pnb + cap(mmag(pi + 1u, coef[pi + 1u])); }
                }
                let acc = w0 * clamp(wv, -8192, 8192) + w1 * clamp(wwv, -8192, 8192) + w2 * clamp(sf, -8192, 8192)
                        + w3 * clamp(sd, -8192, 8192) + w4 * clamp(sr, -8192, 8192) + w5 * clamp(pv, -8192, 8192);
                let pred = (acc + 2048) >> 12u;
                let s = 2u * cap(wm) + cap(wwm) + 2u * af + ad + ar + 2u * cap(pm) + 2u * cap(pred);
                let ws = 3u + 2u * nf + nd + nrr + 2u * has_p + 2u;
                let ab = bucket_a(12u * s / ws);
                var pbk = 0u;
                if (has_p == 1u) { pbk = 1u + bucket_p(4u * min(cap(pm), 15u) + min(pnb, 60u)); }
                let g = select(0u, 1u, nf + nd + nrr > 0u);
                let e = min(firstLeadingBit(1u + 2u * s / ws), NE_u - 1u);
                let sig = ((cls * 2u + g) * NA_u + ab) * NPB_u + pbk;
                var r = 0;
                if (P.pad4 != 0u) {
                    let rawv = array<i32, 18>(wv, wwv, sf, sd, sr, pv, pred, i32(cap(wm)), i32(cap(wwm)), i32(af), i32(ad), i32(ar), i32(cap(pm)), i32(pnb), i32(nf), i32(nd), i32(nrr), i32(has_p));
                    r = learned_res(rawv, cls);
                } else {
                    if (sbit(SIG0u + sig)) {
                        let k = max(e, 2u) - 2u;
                        var h = 0u;
                        let mbase = MAG0u + (cls * NE_u + e) * KB_u;
                        loop {
                            if (h >= KB_u) { break; }
                            if (!sbit(mbase + h)) { break; }
                            h = h + 1u;
                        }
                        var hv = h;
                        if (h == KB_u) { hv = hv + dgamma_s(ESC0u + cls * 32u); }
                        var lo = 0u;
                        if (k >= 1u) { lo = select(0u, 1u, sbit(LOW0u + e * 4u)); }
                        if (k >= 2u) { lo = (lo << 1u) | select(0u, 1u, sbit(LOW0u + e * 4u + 2u + (lo & 1u))); }
                        if (k >= 3u) { lo = (lo << (k - 2u)) | draw_n(k - 2u); }
                        let mv = ((hv << k) | lo) + 1u;
                        let sctx = cls * 9u + sgn3(wm) * 3u + sgn3(sf + sd);
                        if (sbit(SGN0u + sctx)) { r = -bitcast<i32>(mv); } else { r = bitcast<i32>(mv); }
                    }
                }
                let qv = r + pred;
                coef[rowb + x] = qv;
                models[rowb + x] = bitcast<u32>(r);
                wwv = wv;
                wv = qv;
                wwm = wm;
                wm = mmag(rowb + x, qv);
            }
        }
    }
}

@compute @workgroup_size(64)
fn dequant3(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.batch) { return; }
    let rec = P.pad0 + gid.x * 10u;
    let b = mt[rec];
    let cls = bf(b, 6u);
    let m = bitcast<i32>(bf(b, 18u));
    let sh = bf(b, 19u);
    for (var z = mt[rec + 1u]; z < mt[rec + 4u]; z = z + 1u) {
        for (var y = mt[rec + 2u]; y < mt[rec + 5u]; y = y + 1u) {
            for (var x = mt[rec + 3u]; x < mt[rec + 6u]; x = x + 1u) {
                let idx = (z * P.d1 + y) * P.d2 + x;
                let v = coef[idx];
                if (v != 0) {
                    let av = u32(abs(v));
                    let a = bitcast<i32>(min(av, 0x7FFFFFFFu));
                    let d = bitcast<i32>(mt[P.off_dl + cls * 3u + min(av, 3u) - 1u]);
                    var c = mulsh(a, m, sh) - mulsh(d, m, sh + 9u);
                    if (v < 0) { c = -c; }
                    coef[idx] = c;
                }
            }
        }
    }
}

@compute @workgroup_size(64)
fn outliers3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ti = gid.x;
    if (ti >= P.pad3) { return; }
    let rec = P.pad2 + ti * 8u;
    let o = array<u32, 3>(mt[rec], mt[rec + 1u], mt[rec + 2u]);
    let td = array<u32, 3>(mt[rec + 3u], mt[rec + 4u], mt[rec + 5u]);
    let ncell = td[0] * td[1] * td[2];
    for (var i = 0u; i < 96u; i = i + 1u) { om[i] = 0x80008000u; }
    lane_start(mt[rec + 6u], mt[rec + 7u]);
    let n = ogamma(0u);
    var last = -1;
    let corr = bitcast<i32>(P.corr);
    for (var i = 0u; i < n; i = i + 1u) {
        let g = ogamma(32u);
        let li = u32(last + 1 + bitcast<i32>(g));
        last = bitcast<i32>(li);
        let neg = draw(1u) == 1u;
        var k = bitcast<i32>(ogamma(64u) + 1u);
        if (neg) { k = -k; }
        if (li < ncell) {
            let z = li / (td[1] * td[2]);
            let y = (li / td[2]) % td[1];
            let x = li % td[2];
            let gi = ((o[0] + z) * P.d1 + o[1] + y) * P.d2 + o[2] + x;
            coef[gi] = coef[gi] + k * corr;
        }
    }
}

// v3 with LANES lanes per code-block: one workgroup per block, parity passes
// inside the block (TCG1's rule), barrier between passes.

fn npass_blk(ez: u32, ey: u32) -> u32 {
    if (ez > 1u && ey > 1u) { return 7u; }
    if (ez > 1u || ey > 1u) { return 4u; }
    return 1u;
}

@compute @workgroup_size(LANES_WG)
fn decode3l(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let rec = P.pad0 + (P.brick_base + wid.x) * 10u;
    let b = mt[rec];
    let x0 = mt[rec + 3u];
    let x1 = mt[rec + 6u];
    let ltab = mt[rec + 9u];
    lane_start(mt[ltab + lane], mt[ltab + lane + 1u]);
    if (lane == 0u) { wg_u = npass_blk(mt[rec + 4u] - mt[rec + 1u], mt[rec + 5u] - mt[rec + 2u]); }
    let npass = workgroupUniformLoad(&wg_u);
    let ny = P.d1;
    let nx = P.d2;
    let z0 = mt[rec + 1u]; let y0 = mt[rec + 2u];
    let z1 = mt[rec + 4u]; let y1 = mt[rec + 5u];
    let ez = z1 - z0;
    let ey = y1 - y0;
    let zy_is_y = ey > 1u;
    let org = array<u32, 3>(bf(b, 0u), bf(b, 1u), bf(b, 2u));
    let cls = bf(b, 6u);
    let par = bf(b, 7u);
    let psh = bf(b, 8u);
    var pst = array<u32, 3>(0u, 0u, 0u);
    var pen = array<u32, 3>(1u, 1u, 1u);
    if (par != 0xFFFFFFFFu) {
        pst = array<u32, 3>(bf(par, 0u), bf(par, 1u), bf(par, 2u));
        pen = array<u32, 3>(bf(par, 3u), bf(par, 4u), bf(par, 5u));
    }
    let psx = (psh >> 2u) & 1u;
    let sz_ = ny * nx;
    for (var p = 0u; p < npass; p = p + 1u) {
        let pk = P.off_preds + bf(b, 10u + p) * 6u;
        let w0 = bitcast<i32>(mt[pk]); let w1 = bitcast<i32>(mt[pk + 1u]); let w2 = bitcast<i32>(mt[pk + 2u]);
        let w3 = bitcast<i32>(mt[pk + 3u]); let w4 = bitcast<i32>(mt[pk + 4u]); let w5 = bitcast<i32>(mt[pk + 5u]);
        let lt = lattice(npass, zy_is_y, p);
        var nrows = 0u;
        var cy = 1u;
        if (lt.x < ez && lt.z < ey) {
            cy = (ey - lt.z + lt.w - 1u) / lt.w;
            nrows = ((ez - lt.x + lt.y - 1u) / lt.y) * cy;
        }
        for (var j = lane; j < nrows; j = j + LANESu) {
            let z = z0 + lt.x + (j / cy) * lt.y;
            let y = y0 + lt.z + (j % cy) * lt.w;
            var nr = 0u;
            for (var t = 0u; t < 20u; t = t + 1u) {
                let z2 = i32(z) + ROWNB_DZ[t];
                let y2 = i32(y) + ROWNB_DY[t];
                if (z2 >= i32(z0) && z2 < i32(z1) && y2 >= i32(y0) && y2 < i32(y1)) {
                    if (pass_of(npass, zy_is_y, u32(z2) - z0, u32(y2) - y0) < p) {
                        rb_base[nr] = (u32(z2) * ny + u32(y2)) * nx;
                        rb_cat[nr] = ROWNB_CAT[t];
                        nr = nr + 1u;
                    }
                }
            }
            var prow = 0u;
            var pz = 0u;
            var py = 0u;
            if (par != 0xFFFFFFFFu) {
                pz = min(pst[0] + ((z - org[0]) >> (psh & 1u)), pen[0] - 1u);
                py = min(pst[1] + ((y - org[1]) >> ((psh >> 1u) & 1u)), pen[1] - 1u);
                prow = (pz * ny + py) * nx;
            }
            let rowb = (z * ny + y) * nx;
            var wv = 0; var wwv = 0; var wm = 0; var wwm = 0;
            for (var x = x0; x < x1; x = x + 1u) {
                var sf = 0; var sd = 0; var sr = 0;
                var af = 0u; var ad = 0u; var ar = 0u;
                var nf = 0u; var nd = 0u; var nrr = 0u;
                for (var t = 0u; t < nr; t = t + 1u) {
                    let i = rb_base[t] + x;
                    let v = coef[i];
                    let vm = mmag(i, v);
                    let c = rb_cat[t];
                    if (c == 0u) { sf = sf + v; af = af + cap(vm); nf = nf + 1u; }
                    else if (c == 1u) { sd = sd + v; ad = ad + cap(vm); nd = nd + 1u; }
                    else { sr = sr + v; ar = ar + cap(vm); nrr = nrr + 1u; }
                }
                var pv = 0; var pm = 0; var pnb = 0u; var has_p = 0u;
                if (par != 0xFFFFFFFFu) {
                    let px = min(pst[2] + ((x - org[2]) >> psx), pen[2] - 1u);
                    let pi = prow + px;
                    pv = coef[pi];
                    pm = mmag(pi, pv);
                    has_p = 1u;
                    if (pz > pst[0]) { pnb = pnb + cap(mmag(pi - sz_, coef[pi - sz_])); }
                    if (pz + 1u < pen[0]) { pnb = pnb + cap(mmag(pi + sz_, coef[pi + sz_])); }
                    if (py > pst[1]) { pnb = pnb + cap(mmag(pi - nx, coef[pi - nx])); }
                    if (py + 1u < pen[1]) { pnb = pnb + cap(mmag(pi + nx, coef[pi + nx])); }
                    if (px > pst[2]) { pnb = pnb + cap(mmag(pi - 1u, coef[pi - 1u])); }
                    if (px + 1u < pen[2]) { pnb = pnb + cap(mmag(pi + 1u, coef[pi + 1u])); }
                }
                let acc = w0 * clamp(wv, -8192, 8192) + w1 * clamp(wwv, -8192, 8192) + w2 * clamp(sf, -8192, 8192)
                        + w3 * clamp(sd, -8192, 8192) + w4 * clamp(sr, -8192, 8192) + w5 * clamp(pv, -8192, 8192);
                let pred = (acc + 2048) >> 12u;
                let s = 2u * cap(wm) + cap(wwm) + 2u * af + ad + ar + 2u * cap(pm) + 2u * cap(pred);
                let ws = 3u + 2u * nf + nd + nrr + 2u * has_p + 2u;
                let ab = bucket_a(12u * s / ws);
                var pbk = 0u;
                if (has_p == 1u) { pbk = 1u + bucket_p(4u * min(cap(pm), 15u) + min(pnb, 60u)); }
                let g = select(0u, 1u, nf + nd + nrr > 0u);
                let e = min(firstLeadingBit(1u + 2u * s / ws), NE_u - 1u);
                let sig = ((cls * 2u + g) * NA_u + ab) * NPB_u + pbk;
                var r = 0;
                if (P.pad4 != 0u) {
                    let rawv = array<i32, 18>(wv, wwv, sf, sd, sr, pv, pred, i32(cap(wm)), i32(cap(wwm)), i32(af), i32(ad), i32(ar), i32(cap(pm)), i32(pnb), i32(nf), i32(nd), i32(nrr), i32(has_p));
                    r = learned_res(rawv, cls);
                } else {
                    if (sbit(SIG0u + sig)) {
                        let k = max(e, 2u) - 2u;
                        var h = 0u;
                        let mbase = MAG0u + (cls * NE_u + e) * KB_u;
                        loop {
                            if (h >= KB_u) { break; }
                            if (!sbit(mbase + h)) { break; }
                            h = h + 1u;
                        }
                        var hv = h;
                        if (h == KB_u) { hv = hv + dgamma_s(ESC0u + cls * 32u); }
                        var lo = 0u;
                        if (k >= 1u) { lo = select(0u, 1u, sbit(LOW0u + e * 4u)); }
                        if (k >= 2u) { lo = (lo << 1u) | select(0u, 1u, sbit(LOW0u + e * 4u + 2u + (lo & 1u))); }
                        if (k >= 3u) { lo = (lo << (k - 2u)) | draw_n(k - 2u); }
                        let mv = ((hv << k) | lo) + 1u;
                        let sctx = cls * 9u + sgn3(wm) * 3u + sgn3(sf + sd);
                        if (sbit(SGN0u + sctx)) { r = -bitcast<i32>(mv); } else { r = bitcast<i32>(mv); }
                    }
                }
                let qv = r + pred;
                coef[rowb + x] = qv;
                models[rowb + x] = bitcast<u32>(r);
                wwv = wv; wv = qv;
                wwm = wm; wm = mmag(rowb + x, qv);
            }
        }
        storageBarrier();
        workgroupBarrier();
    }
}

// ---------------------------------------------------------------- learned model (v3)
// Net at P.pad4: [off_sig, off_pow, nl, sizes(nl+1), per layer W (out x in), b].

fn lg8(v: u32) -> i32 {
    let w = v + 1u;
    let e = firstLeadingBit(w);
    var m = 0u;
    if (e >= 8u) { m = (w >> (e - 8u)) & 0xFFu; } else { m = (w << (8u - e)) & 0xFFu; }
    return i32(e) * 256 + i32(m);
}

fn slg8(v: i32) -> i32 {
    if (v < 0) { return -lg8(u32(-v)); }
    return lg8(u32(v));
}

fn sig16(z: i32) -> u32 {
    let a = min(u32(abs(z)), 4096u);
    let s = mt[mt[P.pad4] + a];
    if (z >= 0) { return s; }
    return 65536u - s;
}

fn zval(d0: i32, ls0: i32) -> i32 {
    let ls = clamp(ls0, -4096, 6144);
    let i = (ls - ((ls % 256) + 256) % 256) / 256;
    let f = ls - i * 256;
    let m = bitcast<i32>(mt[mt[P.pad4 + 1u] + u32(256 - f)]);
    // exact 32-bit form of mulsh(d, m, 16) for |d| <= 2^25, m in [2^16, 2^17]:
    // d·m + 2^15 = dh·m·2^16 + dl·2^16 + dl·(m − 2^16) + 2^15
    let d = clamp(d0, -33554432, 33554432);
    let dh = d >> 16u;
    let dl = u32(d) & 0xFFFFu;
    let t = dh * m + i32(dl) + i32((dl * u32(m - 65536) + 32768u) >> 16u);
    let sh = i + 1;
    if (sh >= 0) { return t >> u32(min(sh, 31)); }
    let k = u32(-sh);
    let lim = 4097 >> k;
    if (t > lim) { return 4097; }
    if (t < -lim) { return -4097; }
    return t << k;
}

var<private> mx_pi: u32;
var<private> mx_mu0: i32;
var<private> mx_mu1: i32;
var<private> mx_ls0: i32;
var<private> mx_ls1: i32;

fn lcdf(m: i32) -> u32 {
    let t = clamp(m, -65535, 65535) * 256 + 128;
    let c0 = sig16(zval(t - mx_mu0, mx_ls0));
    let c1 = sig16(zval(t - mx_mu1, mx_ls1));
    return (mx_pi * c0 + (32768u - mx_pi) * c1) >> 15u;
}

fn ratio16(num: u32, den: u32) -> u32 {
    var p = 32768u;
    if (den != 0u) { p = (min(num, den) * 65535u) / den; }
    return clamp(p, 32u, 65504u);
}

var<private> l_neg: bool;

fn ltail(m: u32) -> u32 {
    if (m > (1u << 30u)) { return 0u; }
    if (l_neg) { return lcdf(-i32(m)); }
    return 65536u - lcdf(i32(m) - 1);
}

var<private> na: array<i32, 64>;
var<private> nb: array<i32, 64>;

fn learned_res(raw: array<i32, 18>, cls: u32) -> i32 {
    // features (Q8)
    for (var i = 0u; i < 64u; i = i + 1u) { na[i] = 0; }
    for (var i = 0u; i < 7u; i = i + 1u) { na[i] = slg8(raw[i]); }
    for (var i = 7u; i < 14u; i = i + 1u) { na[i] = lg8(u32(raw[i])); }
    na[14] = raw[14] * 64;
    na[15] = raw[15] * 32;
    na[16] = raw[16] * 32;
    na[17] = raw[17] * 256;
    na[18u + min(cls, 9u)] = 256;
    for (var i = 0u; i < 28u; i = i + 1u) { na[i] = clamp(na[i], -16384, 16384); }
    mlp_gen();
    mx_pi = sig16(na[0]) >> 1u;
    mx_mu0 = na[1]; mx_mu1 = na[2]; mx_ls0 = na[3]; mx_ls1 = na[4];
    // binarization
    let c0 = lcdf(0);
    let cm1 = lcdf(-1);
    let pz = select(0u, c0 - cm1, c0 > cm1);
    if (!dbit_p(ratio16(65536u - pz, 65536u))) { return 0; }
    let posm = 65536u - c0;
    l_neg = dbit_p(ratio16(cm1, cm1 + posm));
    // tails are carried between steps: same values, fewer CDF evaluations
    var k = 0u;
    var ta = ltail(1u);
    var tb = 0u;
    loop {
        if (k >= 30u) { break; }
        tb = ltail(1u << (k + 1u));
        if (!dbit_p(ratio16(tb, ta))) { break; }
        ta = tb;
        k = k + 1u;
    }
    var lo = 1u << k;
    var hi = (1u << k) + ((1u << k) - 1u);
    // tail(lo) = ta; tail(hi + 1) = tb (k < 30) or tail(2^31) = 0 (k = 30)
    var tl = ta;
    var end = select(0u, tb, k < 30u);
    loop {
        if (lo >= hi) { break; }
        let mid = lo + (hi - lo + 1u) / 2u;
        let tm = ltail(mid);
        let up = select(0u, tm - end, tm > end);
        let all = select(0u, tl - end, tl > end);
        if (dbit_p(ratio16(up, all))) { lo = mid; tl = tm; } else { hi = mid - 1u; end = tm; }
    }
    if (l_neg) { return -bitcast<i32>(lo); }
    return bitcast<i32>(lo);
}

// MLP_BEGIN (generic version; the host substitutes a straight-line version
// with the stream's weights as literals)
fn mlp_gen() {
    let base = P.pad4 + 2u;
    let nl = mt[base];
    var wo = base + 2u + nl;
    for (var l = 0u; l < nl; l = l + 1u) {
        let ni = mt[base + 1u + l];
        let no = mt[base + 2u + l];
        let bo = wo + ni * no;
        for (var o = 0u; o < no; o = o + 1u) {
            var acc = bitcast<i32>(mt[bo + o]);
            for (var i = 0u; i < ni; i = i + 1u) { acc = acc + bitcast<i32>(mt[wo + o * ni + i]) * na[i]; }
            let v = (acc + 128) >> 8u;
            if (l + 1u < nl) { nb[o] = clamp(v, 0, 16384); } else { nb[o] = clamp(v, -16384, 16384); }
        }
        for (var o = 0u; o < no; o = o + 1u) { na[o] = nb[o]; }
        wo = bo + no;
    }
}
// MLP_END
