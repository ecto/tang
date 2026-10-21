//! GPU decoder for `TCI1` image streams ([`crate::img`]): one WGSL module
//! **generated per stream**, every network compiled to straight-line code
//! with its weights as literals (the ARM, IFCE and synthesis bodies are the
//! `tang-expr` integer-dialect lowering of [`Mlp::graph`] /
//! [`segment_graph`]). All arithmetic is wrapping `i32` / `u32`; the encoder
//! guarantees (and the CPU decoder re-checks) that no accumulation leaves
//! `±2^30`, so the GPU output is byte-identical to [`img::decode`].
//!
//! Kernels, in order:
//! 1. per grid (coarse → fine): `ifce{g}` (one invocation per pixel of grid
//!    `g+1`) then `lat{g}` (one workgroup per tile; each wavefront step runs
//!    the ARM for every active row in parallel, then each rANS lane decodes
//!    its rows; lane states persist across grids in a buffer);
//! 2. `uinit`, then per pyramid level `uh`/`uv` (2× transposed separable
//!    filter) and `ph`/`pv` (pre-concatenation filter);
//! 3. `seg{i}` synthesis segments; the last one writes RGBA8.

use crate::img::{
    self, segment_graph, segments, tables, Layout, Mlp, Model, Parsed, NLB, NSYM, QA, QA_ARM, R,
};
use std::fmt::Write;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use tang_expr::int::{IId, IntGraph};
#[cfg(target_arch = "wasm32")]
use web_time::Instant;
use wgpu::util::DeviceExt;

const WG: usize = 256;
const NB: u32 = 8;

/// GPU device for image decoding.
pub struct ImgGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pl: wgpu::PipelineLayout,
    /// Adapter description.
    pub name: String,
}

/// A stream compiled and uploaded, ready to decode.
pub struct Prepared {
    pipes: Vec<wgpu::ComputePipeline>,
    names: Vec<String>,
    /// (pipeline, workgroups x, workgroups y)
    plan: Vec<(usize, u32, u32)>,
    bind: wgpu::BindGroup,
    lanes0: Vec<u32>,
    lanes_buf: wgpu::Buffer,
    out: wgpu::Buffer,
    /// Pixels.
    pub npx: usize,
    /// Image height, width.
    pub hw: (usize, usize),
    /// Generated WGSL.
    pub src: String,
    /// Shader + pipeline build seconds.
    pub compile_s: f64,
}

fn body(g: &IntGraph, outs: &[IId], np: usize, depth: usize) -> String {
    let p: Vec<String> = (0..np).map(|i| format!("q{i}")).collect();
    let d: Vec<String> = (0..outs.len()).map(|o| format!("o{o}")).collect();
    let pr: Vec<&str> = p.iter().map(|s| s.as_str()).collect();
    let dr: Vec<&str> = d.iter().map(|s| s.as_str()).collect();
    let mut s = String::new();
    for o in 0..outs.len() {
        let _ = writeln!(s, "{}var o{o}: i32;", "    ".repeat(depth));
    }
    s + &g.to_wgsl_body(outs, &pr, &dr, depth)
}

fn mlp_body(m: &Mlp, depth: usize) -> String {
    let (g, o) = m.graph();
    body(&g, &o, m.nin(), depth)
}

/// Pixel-grid dispatch (≤ 65535 groups per dimension) of `n` invocations.
fn groups(n: usize) -> (u32, u32) {
    let g = n.div_ceil(WG);
    if g <= 65535 {
        (g.max(1) as u32, 1)
    } else {
        (65535, g.div_ceil(65535) as u32)
    }
}

const GID: &str = "let i = wid.y * 65535u * 256u + wid.x * 256u + lid.x;";
const ARGS: &str =
    "@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>";

/// Generate the per-stream WGSL module and the dispatch plan.
pub fn shader(m: &Model) -> (String, Vec<(String, u32, u32)>, Sizes) {
    let n = m.sizes.len();
    let mut s = String::new();
    let mut plan: Vec<(String, u32, u32)> = vec![];
    let tab = tables();
    let woff = tab.len() + slots().len();
    let _ = writeln!(
        s,
        "@group(0) @binding(0) var<storage, read> ro: array<u32>;\n\
         @group(0) @binding(1) var<storage, read_write> lanes: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> lat: array<i32>;\n\
         @group(0) @binding(3) var<storage, read_write> ifc: array<i32>;\n\
         @group(0) @binding(4) var<storage, read_write> fa: array<i32>;\n\
         @group(0) @binding(5) var<storage, read_write> fb: array<i32>;\n\
         @group(0) @binding(6) var<storage, read_write> fc: array<i32>;\n\
         @group(0) @binding(7) var<storage, read_write> outb: array<u32>;\n\
         fn rd(pos: u32) -> u32 {{ return (ro[{woff}u + (pos >> 1u)] >> ((pos & 1u) * 16u)) & 0xffffu; }}\n"
    );
    // grid offsets in `lat`
    let mut loff = vec![0usize; n];
    let mut acc = 0;
    for g in 0..n {
        loff[g] = acc;
        acc += m.sizes[g].0 * m.sizes[g].1;
    }
    let lat_len = acc;
    let mut maxa = 1;
    let mut ifc_len = 1;
    for g in (0..n).rev() {
        let (h, w) = m.sizes[g];
        let lay = Layout::new(h, w, m);
        assert!(lay.lanes <= WG, "lanes per tile > {WG}");
        // ---- IFCE
        if let Some(a) = m.ifce_of[g] {
            let mlp = &m.ifces[a];
            let (h1, w1) = m.sizes[g + 1];
            ifc_len = ifc_len.max(h1 * w1 * m.nifce);
            let mut k = format!("@compute @workgroup_size(256) fn ifce{g}({ARGS}) {{\n    {GID}\n    if (i >= {}u) {{ return; }}\n    let y = i / {w1}u; let x = i % {w1}u;\n", h1 * w1);
            for (c, kk) in (g + 1..n).enumerate() {
                let _ = writeln!(k, "    var y{c} = y; var x{c} = x;");
                for j in g + 1..kk {
                    if m.sizes[j] != m.sizes[j + 1] {
                        let (hh, ww) = m.sizes[j + 1];
                        let _ = writeln!(
                            k,
                            "    y{c} = min(y{c} / 2u, {}u); x{c} = min(x{c} / 2u, {}u);",
                            hh - 1,
                            ww - 1
                        );
                    }
                }
                let _ = writeln!(
                    k,
                    "    let q{c} = lat[{}u + y{c} * {}u + x{c}] << {QA_ARM}u;",
                    loff[kk], m.sizes[kk].1
                );
            }
            assert_eq!(mlp.nin(), n - g - 1);
            k += &mlp_body(mlp, 1);
            for j in 0..m.nifce {
                let _ = writeln!(k, "    ifc[i * {}u + {j}u] = o{j};", m.nifce);
            }
            k += "}\n";
            s += &k;
            let (gx, gy) = groups(h1 * w1);
            plan.push((format!("ifce{g}"), gx, gy));
        }
        // ---- latents
        let na = lay.th.min(lay.tw / lay.s + 2);
        maxa = maxa.max(na);
        let nc = m.ctx.len();
        let mut k = format!(
            "@compute @workgroup_size(256) fn lat{g}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {{\n\
             \x20   let lid = lid3.x; let tile = wid.x;\n\
             \x20   let y0 = (tile / {ntx}u) * {th}u; let x0 = (tile % {ntx}u) * {tw}u;\n\
             \x20   let tht = min({th}u, {h}u - y0); let twt = min({tw}u, {w}u - x0);\n\
             \x20   let gl = tile * {L}u + lid;\n\
             \x20   var st = 0u; var pos = 0u;\n\
             \x20   if (lid < {L}u) {{ st = lanes[2u * gl]; pos = lanes[2u * gl + 1u]; }}\n\
             \x20   for (var t = 0u; t < {steps}u; t++) {{\n\
             \x20       var ylo = 0u; if (t + 1u > twt) {{ ylo = (t - twt) / {S}u + 1u; }}\n\
             \x20       let yhi = min(tht - 1u, t / {S}u);\n\
             \x20       for (var y = ylo + lid; y <= yhi; y += 256u) {{\n\
             \x20           let gx = x0 + t - {S}u * y; let gy = y0 + y;\n",
            tw = lay.tw,
            L = lay.lanes,
            steps = lay.steps,
            S = lay.s,
            th = lay.th,
            ntx = lay.ntx,
        );
        for (i, &(dy, dx)) in m.ctx.iter().enumerate() {
            let _ = writeln!(
                k,
                "            var q{i} = 0; {{ let yy = i32(gy) + ({dy}); let xx = i32(gx) + ({dx}); if (yy >= i32(y0) && yy < i32(y0 + tht) && xx >= i32(x0) && xx < i32(x0 + twt)) {{ q{i} = lat[{}u + u32(yy) * {w}u + u32(xx)] << {QA_ARM}u; }} }}",
                loff[g]
            );
        }
        if m.ifce_of[g].is_some() {
            let (h1, w1) = m.sizes[g + 1];
            let _ = writeln!(
                k,
                "            let ib = (min(gy / 2u, {}u) * {w1}u + min(gx / 2u, {}u)) * {}u;",
                h1 - 1,
                w1 - 1,
                m.nifce
            );
            for j in 0..m.nifce {
                let _ = writeln!(k, "            let q{} = ifc[ib + {j}u];", nc + j);
            }
        } else {
            for j in 0..m.nifce {
                let _ = writeln!(k, "            let q{} = 0;", nc + j);
            }
        }
        k += &mlp_body(&m.arm, 3);
        let _ = write!(
            k,
            "            let mu = clamp(o0, -16384, 16383);\n\
             \x20           let base = (mu + 128) >> 8u;\n\
             \x20           let fbk = u32((mu - (base << 8u) + 128) >> {fbs}u);\n\
             \x20           let lb = u32(clamp((o1 + {lbo}) >> {lbs}u, 0, {nlb1}));\n\
             \x20           let tof = (fbk * {NLB}u + lb) * {ns1}u;\n\
             \x20           sbase[y - ylo] = base; soff[y - ylo] = tof;\n\
             \x20           sca[y - ylo] = vec4<u32>(ro[tof + {Rm2}u], ro[tof + {Rm1}u], ro[tof + {R}u], ro[tof + {R1}u]); scb[y - ylo] = vec4<u32>(ro[tof + {R2}u], ro[tof + {R3}u], ro[tof + {R4}u], ro[tof + {R5}u]);\n\
             \x20       }}\n\
             \x20       storageBarrier(); workgroupBarrier();\n\
             \x20       if (lid < {L}u) {{\n\
             \x20           for (var y = ylo + (lid + {L}u - ylo % {L}u) % {L}u; y <= yhi; y += {L}u) {{\n\
             \x20               let off = soff[y - ylo]; let base = sbase[y - ylo];\n\
             \x20               let v = st & 32767u;\n\
             \x20               let ca = sca[y - ylo]; let cb = scb[y - ylo];\n\
             \x20               var lo: u32; var c0: u32; var c1: u32;\n\
             \x20               if (v >= ca.x && v < cb.w) {{\n\
             \x20                   let cs = array<u32, 8>(ca.x, ca.y, ca.z, ca.w, cb.x, cb.y, cb.z, cb.w);\n\
             \x20                   let k = u32(v >= ca.y) + u32(v >= ca.z) + u32(v >= ca.w) + u32(v >= cb.x) + u32(v >= cb.y) + u32(v >= cb.z);\n\
             \x20                   lo = {Rm2}u + k; c0 = cs[k]; c1 = cs[k + 1u];\n\
             \x20               }} else {{\n\
             \x20                   lo = ro[{soff}u + off / {ns1}u * 128u + (v >> 8u)];\n\
             \x20                   while (ro[off + lo + 1u] <= v) {{ lo += 1u; }}\n\
             \x20                   c0 = ro[off + lo]; c1 = ro[off + lo + 1u];\n\
             \x20               }}\n\
             \x20               st = (c1 - c0) * (st >> 15u) + v - c0;\n\
             \x20               if (st < 65536u) {{ st = (st << 16u) | rd(pos); pos += 1u; }}\n\
             \x20               var val = base + i32(lo) - {r1};\n\
             \x20               if (lo == 0u || lo == {nsm1}u) {{\n\
             \x20                   var e = 0u;\n\
             \x20                   for (var k = 0u; k < 2u; k++) {{\n\
             \x20                       let pk = st & 32767u; let b = pk >> 7u;\n\
             \x20                       st = 128u * (st >> 15u) + pk - (b << 7u);\n\
             \x20                       if (st < 65536u) {{ st = (st << 16u) | rd(pos); pos += 1u; }}\n\
             \x20                       e = (e << 8u) | b;\n\
             \x20                   }}\n\
             \x20                   if (lo == 0u) {{ val = base - {r1} - i32(e); }} else {{ val = base + {r1} + i32(e); }}\n\
             \x20               }}\n\
             \x20               lat[{lo_g}u + (y0 + y) * {w}u + x0 + t - {S}u * y] = val;\n\
             \x20           }}\n\
             \x20       }}\n\
             \x20       storageBarrier(); workgroupBarrier();\n\
             \x20   }}\n\
             \x20   if (lid < {L}u) {{ lanes[2u * gl] = st; lanes[2u * gl + 1u] = pos; }}\n\
             }}\n",
            nlb1 = NLB - 1,
            fbs = img::FB_SHIFT,
            lbs = img::LB_SHIFT,
            lbo = 1280 + (1 << (img::LB_SHIFT - 1)),
            Rm2 = R - 2,
            Rm1 = R - 1,
            R = R,
            R1 = R + 1,
            R2 = R + 2,
            R3 = R + 3,
            R4 = R + 4,
            R5 = R + 5,
            soff = tables().len(),
            ns1 = NSYM + 1,
            L = lay.lanes,
            r1 = R + 1,
            nsm1 = NSYM - 1,
            lo_g = loff[g],
            S = lay.s,
        );
        s += &k;
        plan.push((format!("lat{g}"), lay.nt as u32, 1));
    }
    let _ = writeln!(s, "var<workgroup> sbase: array<i32, {maxa}>;\nvar<workgroup> soff: array<u32, {maxa}>;\nvar<workgroup> sca: array<vec4<u32>, {maxa}>;\nvar<workgroup> scb: array<vec4<u32>, {maxa}>;");

    // ---- upsampling pyramid
    let idx: Vec<usize> = (0..n).filter(|&i| !m.hyper[i]).collect();
    let g0 = *idx.last().unwrap();
    let (mut ch, mut cw) = m.sizes[g0];
    let _ = writeln!(s, "@compute @workgroup_size(256) fn uinit({ARGS}) {{\n    {GID}\n    if (i >= {}u) {{ return; }}\n    fa[i] = lat[{}u + i] << {QA}u;\n}}", ch * cw, loff[g0]);
    let (gx, gy) = groups(ch * cw);
    plan.push(("uinit".into(), gx, gy));
    let mut fb_len = 1;
    let mut nchan = 1;
    for (k, &g) in idx.iter().rev().skip(1).enumerate() {
        let (th, tw) = m.sizes[g];
        let f = &m.tconv[k % m.tconv.len()];
        let kk = f.w.len();
        let p0 = kk / 2;
        let cc = 3 * p0 - 1;
        assert_eq!(kk, 8, "upsampling kernel size");
        let we: Vec<i32> = (0..4).map(|i| f.w[2 * i]).collect();
        let wo: Vec<i32> = (0..4).map(|i| f.w[2 * i + 1]).collect();
        let taps = |src: &str, idx: &dyn Fn(&str) -> String| -> String {
            let mut e = String::from("0");
            for i in 0..4 {
                let _ = write!(
                    e,
                    " + select({}, {}, par == 1u) * {src}[{}]",
                    we[i],
                    wo[i],
                    idx(&format!("u32(clamp(i32(q) - {i} - {p0}, 0, LIM))"))
                );
            }
            e
        };
        let rsh = |e: &str, fw: u32| {
            if fw == 0 {
                format!("({e})")
            } else {
                format!("((({e}) + {}) >> {fw}u)", 1i64 << (fw - 1))
            }
        };
        // horizontal: T[c][y][x2], x2 < tw
        fb_len = fb_len.max(nchan * ch * tw);
        let hx = taps("fa", &|j: &str| {
            format!(
                "c * {}u + y * {cw}u + {}",
                ch * cw,
                j.replace("LIM", &format!("{}", cw - 1))
            )
        });
        let _ = writeln!(
            s,
            "@compute @workgroup_size(256) fn uh{k}({ARGS}) {{\n    {GID}\n    if (i >= {tot}u) {{ return; }}\n    let c = i / {cs}u; let r = i % {cs}u; let y = r / {tw}u; let x2 = r % {tw}u;\n    let q = (x2 + {cc}u) >> 1u; let par = (x2 + {cc}u) & 1u;\n    fb[i] = {};\n}}",
            rsh(&hx, f.fw),
            tot = nchan * ch * tw,
            cs = ch * tw
        );
        let (gx, gy) = groups(nchan * ch * tw);
        plan.push((format!("uh{k}"), gx, gy));
        // vertical: A[c+1][y2][x2], y2 < th
        let vx = taps("fb", &|j: &str| {
            format!(
                "c * {}u + {} * {tw}u + x2",
                ch * tw,
                j.replace("LIM", &format!("{}", ch - 1))
            )
        });
        let _ = writeln!(
            s,
            "@compute @workgroup_size(256) fn uv{k}({ARGS}) {{\n    {GID}\n    if (i >= {tot}u) {{ return; }}\n    let c = i / {cs}u; let r = i % {cs}u; let y2 = r / {tw}u; let x2 = r % {tw}u;\n    let q = (y2 + {cc}u) >> 1u; let par = (y2 + {cc}u) & 1u;\n    fa[(c + 1u) * {cs}u + r] = {};\n}}",
            rsh(&vx, f.fw),
            tot = nchan * th * tw,
            cs = th * tw
        );
        let (gx, gy) = groups(nchan * th * tw);
        plan.push((format!("uv{k}"), gx, gy));
        // pre-concatenation filter on the target grid (zero padding, residual)
        let pf = &m.pconv[k % m.pconv.len()];
        let pk = pf.w.len();
        let pp = (pk / 2) as i64;
        let mut eh = String::from("0");
        let mut ev = String::from("0");
        for t in 0..pk {
            let d = t as i64 - pp;
            let _ = write!(eh, " + select(0, {} * (lat[{}u + y * {tw}u + u32(i32(x) + ({d}))] << {QA}u), i32(x) + ({d}) >= 0 && i32(x) + ({d}) < {tw})", pf.w[t], loff[g]);
            let _ = write!(ev, " + select(0, {} * fb[u32(i32(y) + ({d})) * {tw}u + x], i32(y) + ({d}) >= 0 && i32(y) + ({d}) < {th})", pf.w[t]);
        }
        let _ = writeln!(
            s,
            "@compute @workgroup_size(256) fn ph{k}({ARGS}) {{\n    {GID}\n    if (i >= {tot}u) {{ return; }}\n    let y = i / {tw}u; let x = i % {tw}u;\n    fb[i] = {};\n}}\n\
             @compute @workgroup_size(256) fn pv{k}({ARGS}) {{\n    {GID}\n    if (i >= {tot}u) {{ return; }}\n    let y = i / {tw}u; let x = i % {tw}u;\n    fa[i] = {} + (lat[{}u + i] << {QA}u);\n}}",
            rsh(&eh, pf.fw),
            rsh(&ev, pf.fw),
            loff[g],
            tot = th * tw
        );
        let (gx, gy) = groups(th * tw);
        plan.push((format!("ph{k}"), gx, gy));
        plan.push((format!("pv{k}"), gx, gy));
        nchan += 1;
        (ch, cw) = (th, tw);
    }
    assert_eq!((ch, cw), (m.h, m.w));
    // ---- synthesis
    let hw = m.h * m.w;
    let segs = segments(m);
    let mut src = "fa";
    let mut fc_len = 1;
    for (si, seg) in segs.iter().enumerate() {
        let last = si + 1 == segs.len();
        let (g, o) = segment_graph(m, seg, last);
        let c0 = seg[0];
        let cin = c0.cin();
        let kk = c0.k;
        let pad = (kk / 2) as i64;
        let mut k = format!("@compute @workgroup_size(256) fn seg{si}({ARGS}) {{\n    {GID}\n    if (i >= {hw}u) {{ return; }}\n    let y = i / {}u; let x = i % {}u;\n", m.w, m.w);
        let mut np = 0;
        for ci in 0..cin {
            for dy in 0..kk {
                for dx in 0..kk {
                    let (oy, ox) = (dy as i64 - pad, dx as i64 - pad);
                    let yy = if oy == 0 {
                        "y".to_string()
                    } else {
                        format!("u32(clamp(i32(y) + ({oy}), 0, {}))", m.h - 1)
                    };
                    let xx = if ox == 0 {
                        "x".to_string()
                    } else {
                        format!("u32(clamp(i32(x) + ({ox}), 0, {}))", m.w - 1)
                    };
                    let _ = writeln!(
                        k,
                        "    let q{np} = {src}[{}u + {yy} * {}u + {xx}];",
                        ci * hw,
                        m.w
                    );
                    np += 1;
                }
            }
        }
        if c0.residual {
            for ci in 0..cin {
                let _ = writeln!(k, "    let q{np} = {src}[{}u + i];", ci * hw);
                np += 1;
            }
        }
        if last {
            if let Some(st) = &m.stab {
                for ci in 0..st.cin() {
                    let _ = writeln!(k, "    let q{np} = fa[{}u + i];", ci * hw);
                    np += 1;
                }
            }
        }
        k += &body(&g, &o, np, 1);
        let dst = if src == "fb" { "fc" } else { "fb" };
        if last {
            assert_eq!(o.len(), 3);
            k += "    outb[i] = u32(o0) | (u32(o1) << 8u) | (u32(o2) << 16u) | 0xff000000u;\n";
        } else {
            for j in 0..o.len() {
                let _ = writeln!(k, "    {dst}[{}u + i] = o{j};", j * hw);
            }
            if dst == "fb" {
                fb_len = fb_len.max(o.len() * hw);
            } else {
                fc_len = fc_len.max(o.len() * hw);
            }
        }
        k += "}\n";
        s += &k;
        let (gx, gy) = groups(hw);
        plan.push((format!("seg{si}"), gx, gy));
        src = dst;
    }
    let fa_len = nchan * hw;
    (
        s,
        plan,
        Sizes {
            lat: lat_len,
            ifc: ifc_len,
            fa: fa_len,
            fb: fb_len,
            fc: fc_len,
            out: hw,
            woff,
        },
    )
}

/// Buffer sizes (elements) for one stream.
#[derive(Debug, Clone, Copy)]
pub struct Sizes {
    lat: usize,
    ifc: usize,
    fa: usize,
    fb: usize,
    fc: usize,
    out: usize,
    woff: usize,
}

impl ImgGpu {
    /// Open the default high-performance adapter.
    pub fn new() -> Result<Self, String> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|e| format!("adapter: {e}"))?;
            let al = adapter.limits();
            let limits = wgpu::Limits {
                max_buffer_size: al.max_buffer_size,
                max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
                max_storage_buffers_per_shader_stage: NB,
                ..Default::default()
            };
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("tc-img"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                })
                .await
                .map_err(|e| format!("device: {e}"))?;
            let info = adapter.get_info();
            let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..NB)
                .map(|b| wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: b == 0 },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            Ok(Self {
                device,
                queue,
                layout,
                pl,
                name: format!("{} ({:?})", info.name, info.backend),
            })
        })
    }

    /// Compile the stream's kernels and upload its data.
    pub fn prepare(&self, p: &Parsed) -> Prepared {
        let t0 = Instant::now();
        let (src, plan, sz) = shader(&p.m);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("tc-img"),
                source: wgpu::ShaderSource::Wgsl(src.clone().into()),
            });
        let mut names: Vec<String> = vec![];
        let mut pipes = vec![];
        let mut plan2 = vec![];
        for (name, gx, gy) in plan {
            let i = match names.iter().position(|n| *n == name) {
                Some(i) => i,
                None => {
                    pipes.push(self.device.create_compute_pipeline(
                        &wgpu::ComputePipelineDescriptor {
                            label: Some(&name),
                            layout: Some(&self.pl),
                            module: &module,
                            entry_point: Some(&name),
                            compilation_options: Default::default(),
                            cache: None,
                        },
                    ));
                    names.push(name);
                    names.len() - 1
                }
            };
            plan2.push((i, gx, gy));
        }
        let compile_s = t0.elapsed().as_secs_f64();
        // read-only data: tables then packed words (+ zero pad)
        let mut ro: Vec<u32> = tables().clone();
        ro.extend_from_slice(slots());
        assert_eq!(ro.len(), sz.woff);
        let w = &p.words;
        for c in w.chunks(2) {
            ro.push(c[0] as u32 | (*c.get(1).unwrap_or(&0) as u32) << 16);
        }
        ro.extend([0u32; 8]);
        let mut lanes0 = vec![];
        for &s in &p.starts {
            let s = s as usize;
            lanes0.push((w[s] as u32) << 16 | w[s + 1] as u32);
            lanes0.push(s as u32 + 2);
        }
        let d = &self.device;
        let buf = |data: &[u32], usage: wgpu::BufferUsages| {
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck_u32(data),
                usage,
            })
        };
        let st = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let rob = buf(&ro, st);
        let lanes_buf = buf(&lanes0, st);
        let mk = |n: usize| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n.max(1) * 4) as u64,
                usage: st,
                mapped_at_creation: false,
            })
        };
        let (lat, ifc, fa, fb, fc, out) = (
            mk(sz.lat),
            mk(sz.ifc),
            mk(sz.fa),
            mk(sz.fb),
            mk(sz.fc),
            mk(sz.out),
        );
        let bufs = [&rob, &lanes_buf, &lat, &ifc, &fa, &fb, &fc, &out];
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        Prepared {
            pipes,
            names,
            plan: plan2,
            bind,
            lanes0,
            lanes_buf,
            out,
            npx: sz.out,
            hw: (p.m.h, p.m.w),
            src,
            compile_s,
        }
    }

    /// Run the whole decode; returns GPU wall seconds (submit → done).
    pub fn run(&self, pr: &Prepared) -> f64 {
        self.run_upto(pr, pr.plan.len())
    }

    /// Kernel names in dispatch order.
    pub fn plan_names(&self, pr: &Prepared) -> Vec<String> {
        pr.plan
            .iter()
            .map(|&(i, _, _)| pr.names[i].clone())
            .collect()
    }

    /// Run the first `k` dispatches only (profiling).
    pub fn run_upto(&self, pr: &Prepared, k: usize) -> f64 {
        self.queue
            .write_buffer(&pr.lanes_buf, 0, bytemuck_u32(&pr.lanes0));
        let _ = self.device.poll(wgpu::PollType::Wait);
        let t0 = Instant::now();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &pr.bind, &[]);
            for &(i, gx, gy) in &pr.plan[..k] {
                cp.set_pipeline(&pr.pipes[i]);
                cp.dispatch_workgroups(gx, gy, 1);
            }
        }
        self.queue.submit([enc.finish()]);
        let _ = self.device.poll(wgpu::PollType::Wait);
        t0.elapsed().as_secs_f64()
    }

    /// Decoded image as interleaved RGB8.
    pub fn readback(&self, pr: &Prepared) -> Vec<u8> {
        let size = (pr.npx * 4) as u64;
        let st = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&pr.out, 0, &st, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = st.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let mut v = Vec::with_capacity(pr.npx * 3);
        for c in data.as_chunks::<4>().0 {
            v.extend_from_slice(&c[..3]);
        }
        v
    }
}

/// Per table, 128 slots of `2^{PB−7}`: the first symbol whose CDF interval
/// reaches the slot (the GPU's fallback search starts there).
fn slots() -> &'static Vec<u32> {
    static S: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let t = tables();
        let mut out = vec![];
        for cum in t.chunks(NSYM + 1) {
            let mut sym = 0;
            for j in 0..128u32 {
                while cum[sym + 1] <= j << 8 {
                    sym += 1;
                }
                out.push(sym as u32);
            }
        }
        out
    })
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding; any byte view of it is valid.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// Decode a `TCI1` stream on the GPU: (h, w, RGB8, prepared state).
pub fn decode(
    gpu: &ImgGpu,
    bytes: &[u8],
) -> Result<(usize, usize, Vec<u8>), crate::bitio::DecodeError> {
    let p = img::parse(bytes)?;
    let pr = gpu.prepare(&p);
    gpu.run(&pr);
    Ok((p.m.h, p.m.w, gpu.readback(&pr)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_matches_cpu() {
        let Ok(gpu) = ImgGpu::new() else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        let c = img::tests::synth(7);
        for (lanes, tw, th) in [(1, 0, 0), (16, 0, 0), (16, 8, 0), (32, 8, 8)] {
            let (bytes, _) = img::encode(&c, lanes, tw, th);
            let cpu = img::decode(&bytes).unwrap();
            let gpu_out = decode(&gpu, &bytes).unwrap();
            assert_eq!(cpu, gpu_out, "lanes {lanes} tile {tw}x{th}");
        }
    }
}
