//! The TCG3 rANS lane decoder as `tang-expr` integer graphs.
//!
//! One residual per evaluation: the lane's 32-bit rANS state and word
//! position go in and come out (loop-carried state), with renormalization,
//! 16-bit word fetch from the stream, and the binarization (static priors or
//! the learned mixture CDF). The same graph is lowered to the WGSL kernel
//! (replacing the hand-written `res_static_gen` / `res_learned_gen`) and run
//! by the CPU interpreter; a stream can also ship its own graphs
//! ([`crate::shipdec`]).
//!
//! The graphs are pure, so `if` becomes a predicate: a decision under a
//! false predicate still computes its step but keeps the old state (and its
//! word fetch reads in bounds or 0). Loops carry the predicate in their
//! condition, so they run exactly the hand-written trip counts.
//!
//! ABI (both roles): buffer 0 `mt` (`u32`: priors / σ and 2^x tables),
//! buffer 1 `stream` (`u32`, two little-endian 16-bit words each);
//! params `p0 = x`, `p1 = pos`, `p2 = end` (lane state and word bound), then
//! - static: `p3 = sig`, `p4 = e`, `p5 = cls`, `p6 = sctx`, `p7 = off_init`
//!   (all `u32`);
//! - learned: `p3..p7` = the net outputs `o0..o4` (`i32`), `p8 = off_sig`,
//!   `p9 = off_pow` (`u32`);
//! - adaptive learned (per-lane APM, [`crate::lmodel3::Adapt`]): the learned
//!   params, then `p10 = off_stretch` (the stretch table in `mt`), `p11 = wa`,
//!   `p12 = lim`, `p13 = mask` (`u32`), and scratch buffer 2 `adt` (136 `u32`
//!   words, `(p << 8) | count` per bin, reset per block by the host);
//!
//! outputs `x'`, `pos'` (`u32`), `r` (`i32`), plus the final effect token
//! (`u32`, always 0) for the adaptive role.

use crate::coefcode::{KB, NE};
use crate::gpuwav::{ESC0, LOW0, MAG0, SGN0, SIG0};
use crate::lmodel3::{AB, AC};
use tang_expr::int::{Elem, IId, IType, IntGraph};

/// Buffer ABI of the residual graphs.
pub const BUFS: [(&str, Elem); 2] = [("mt", Elem::U32), ("stream", Elem::U32)];
/// Param types of the static graph.
pub const STATIC_PARAMS: [IType; 8] = [IType::U32; 8];
/// Param types of the learned graph.
pub const LEARNED_PARAMS: [IType; 10] = [
    IType::U32,
    IType::U32,
    IType::U32,
    IType::I32,
    IType::I32,
    IType::I32,
    IType::I32,
    IType::I32,
    IType::U32,
    IType::U32,
];
/// Output types of both residual graphs.
pub const OUTS: [IType; 3] = [IType::U32, IType::U32, IType::I32];
/// Param types of the adaptive learned graph.
pub const LEARNED_AD_PARAMS: [IType; 14] = [
    IType::U32,
    IType::U32,
    IType::U32,
    IType::I32,
    IType::I32,
    IType::I32,
    IType::I32,
    IType::I32,
    IType::U32,
    IType::U32,
    IType::U32,
    IType::U32,
    IType::U32,
    IType::U32,
];
/// Output types of the adaptive learned graph (the last is the effect token).
pub const OUTS_AD: [IType; 4] = [IType::U32, IType::U32, IType::I32, IType::U32];
/// Scratch-buffer ABI of the adaptive learned graph: the lane's APM bins
/// (`adt[ad_b + i]` in the kernel's workgroup memory).
pub const RW_AD: [(&str, Elem, u32, &str); 1] = [("adt", Elem::U32, (AC * AB) as u32, "ad_b")];

/// rANS lane state in a graph.
#[derive(Clone, Copy)]
struct St {
    x: IId,
    pos: IId,
}

/// The adaptive layer's inputs in a graph.
#[derive(Clone, Copy)]
struct Ad {
    buf: u16,
    so: IId,
    wa: IId,
    lim: IId,
    /// Per decision group: the layer is on.
    en: [IId; 4],
}

struct B {
    g: IntGraph,
    mt: u16,
    stream: u16,
    end: IId,
    ad: Option<Ad>,
}

impl B {
    fn new() -> Self {
        let mut g = IntGraph::new();
        let mt = g.buffer(BUFS[0].0, BUFS[0].1);
        let stream = g.buffer(BUFS[1].0, BUFS[1].1);
        let end = g.param(2, IType::U32);
        Self {
            g,
            mt,
            stream,
            end,
            ad: None,
        }
    }
    fn u(&mut self, v: u32) -> IId {
        self.g.u32(v)
    }
    fn i(&mut self, v: i32) -> IId {
        self.g.i32(v)
    }
    fn word(&mut self, i: IId) -> IId {
        let g = &mut self.g;
        let inb = g.lt(i, self.end);
        let one = g.u32(1);
        let w = g.shr(i, one);
        let w = g.load(self.stream, w);
        let par = g.and(i, one);
        let sixteen = g.u32(16);
        let sh = g.mul(par, sixteen);
        let v = g.shr(w, sh);
        let m = g.u32(0xFFFF);
        let v = g.and(v, m);
        let z = g.u32(0);
        g.select(inb, v, z)
    }
    /// One rANS step (`cum`, `f`), applied only when `on`.
    fn step(&mut self, s: St, cum: IId, f: IId, on: Option<IId>) -> St {
        let sixteen = self.u(16);
        let m = self.u(0xFFFF);
        let lim = self.u(65536);
        let one = self.u(1);
        let g = &mut self.g;
        let hi = g.shr(s.x, sixteen);
        let lo = g.and(s.x, m);
        let a = g.mul(f, hi);
        let b = g.sub(lo, cum);
        let x1 = g.add(a, b);
        let ren = g.lt(x1, lim);
        let w = self.word(s.pos);
        let g = &mut self.g;
        let xs = g.shl(x1, sixteen);
        let xr = g.or(xs, w);
        let x2 = g.select(ren, xr, x1);
        let p1 = g.add(s.pos, one);
        let p2 = g.select(ren, p1, s.pos);
        match on {
            None => St { x: x2, pos: p2 },
            Some(c) => St {
                x: g.select(c, x2, s.x),
                pos: g.select(c, p2, s.pos),
            },
        }
    }
    /// A binary decision at `P(1) = p1/2^16`; the bit is false when off.
    fn bit(&mut self, s: St, p1: IId, on: Option<IId>) -> (St, IId) {
        let k = self.u(65536);
        let m = self.u(0xFFFF);
        let z = self.u(0);
        let g = &mut self.g;
        let f0 = g.sub(k, p1);
        let lo = g.and(s.x, m);
        let b = g.le(f0, lo);
        let cum = g.select(b, f0, z);
        let f = g.select(b, p1, f0);
        let s = self.step(s, cum, f, on);
        let b = match on {
            None => b,
            Some(c) => self.g.and(b, c),
        };
        (s, b)
    }
    /// A decision through the lane's adaptive probability map on context
    /// `ctx` (decision group `grp`), as [`crate::lmodel3::Adapt`]: the two
    /// stretch-domain bins around `p1` are interpolated and mixed with `p1`,
    /// then the nearer bin moves toward the bit. Returns the state, the bit
    /// and the next effect token; nothing is written when `on` is false.
    fn ad_bit(
        &mut self,
        s: St,
        tok: IId,
        ctx: IId,
        grp: usize,
        p1: IId,
        on: Option<IId>,
    ) -> (St, IId, IId) {
        let a = self.ad.expect("adaptive graph");
        let g = &mut self.g;
        let (one, four, eight) = (g.u32(1), g.u32(4), g.u32(8));
        let si = g.shr(p1, four);
        let si = g.add(a.so, si);
        let z = g.load(self.mt, si);
        let z = g.bitcast(IType::I32, z);
        let (lo, hi, off) = (g.i32(-2048), g.i32(2047), g.i32(2048));
        let z = g.clamp(z, lo, hi);
        let z = g.add(z, off);
        let st = g.bitcast(IType::U32, z);
        let j = g.shr(st, eight);
        let m255 = g.u32(255);
        let w = g.and(st, m255);
        let kab = g.u32(AB as u32);
        let b = g.mul(ctx, kab);
        let b = g.add(b, j);
        let b1 = g.add(b, one);
        let e0 = g.load_rw(a.buf, b, tok);
        let e1 = g.load_rw(a.buf, b1, tok);
        let t0 = g.shr(e0, eight);
        let t1 = g.shr(e1, eight);
        let k256 = g.u32(256);
        let wi = g.sub(k256, w);
        let x = g.mul(t0, wi);
        let y = g.mul(t1, w);
        let pr = g.add(x, y);
        let pr = g.shr(pr, eight);
        let x = g.mul(pr, a.wa);
        let k16 = g.u32(16);
        let wb = g.sub(k16, a.wa);
        let y = g.mul(p1, wb);
        let pf = g.add(x, y);
        let pf = g.shr(pf, four);
        let (c32, c65504) = (g.u32(32), g.u32(65504));
        let pf = g.clamp(pf, c32, c65504);
        let en = a.en[grp];
        let pu = g.select(en, pf, p1);
        let (s, got) = self.bit(s, pu, on);
        let g = &mut self.g;
        let k128 = g.u32(128);
        let near = g.lt(w, k128);
        let k = g.select(near, b, b1);
        let e = g.select(near, e0, e1);
        let t = g.shr(e, eight);
        let t = g.bitcast(IType::I32, t);
        let n = g.and(e, m255);
        let (zi, top) = (g.i32(0), g.i32(65535));
        let tg = g.select(got, top, zi);
        let d = g.sub(tg, t);
        // rate 2^-⌊log₂(n+2)⌋
        let two = g.u32(2);
        let n2 = g.add(n, two);
        let lz = g.clz(n2);
        let c31 = g.u32(31);
        let sh = g.sub(c31, lz);
        let dd = g.shr(d, sh);
        let t2 = g.add(t, dd);
        let t2 = g.bitcast(IType::U32, t2);
        let t2 = g.shl(t2, eight);
        let more = g.lt(n, a.lim);
        let n1 = g.add(n, one);
        let nn = g.select(more, n1, n);
        let ne = g.or(t2, nn);
        let wr = match on {
            None => en,
            Some(c) => g.and(en, c),
        };
        let v = g.select(wr, ne, e);
        let tok = g.store(a.buf, k, v, tok);
        (s, got, tok)
    }
    /// `n ≤ 16` raw bits.
    fn raw(&mut self, s: St, n: IId, on: Option<IId>) -> (St, IId) {
        let sixteen = self.u(16);
        let m = self.u(0xFFFF);
        let one = self.u(1);
        let g = &mut self.g;
        let sh = g.sub(sixteen, n);
        let lo = g.and(s.x, m);
        let v = g.shr(lo, sh);
        let cum = g.shl(v, sh);
        let f = g.shl(one, sh);
        (self.step(s, cum, f, on), v)
    }
    /// `n` raw bits in chunks of ≤ 16 (`n = 0`: none).
    fn raw_n(&mut self, s: St, n: IId) -> (St, IId) {
        let z = self.u(0);
        let mut b = std::mem::take(&mut self.g);
        let (stream, end, mt) = (self.stream, self.end, self.mt);
        let out = b.while_loop(&[n, z, s.x, s.pos], |g, st| {
            let mut me = B {
                g: std::mem::take(g),
                mt,
                stream,
                end,
                ad: None,
            };
            let zero = me.u(0);
            let sixteen = me.u(16);
            let go = me.g.lt(zero, st[0]);
            let c = me.g.min(st[0], sixteen);
            let (s2, v) = me.raw(
                St {
                    x: st[2],
                    pos: st[3],
                },
                c,
                None,
            );
            let o = me.g.shl(st[1], c);
            let o = me.g.or(o, v);
            let n2 = me.g.sub(st[0], c);
            *g = me.g;
            (go, vec![n2, o, s2.x, s2.pos])
        });
        b.cap_loop(out[0], 3);
        self.g = b;
        (
            St {
                x: out[2],
                pos: out[3],
            },
            out[1],
        )
    }
    fn prior(&mut self, idx: IId, off: IId) -> IId {
        let g = &mut self.g;
        let a = g.add(off, idx);
        let m = g.load(self.mt, a);
        let mask = g.u32(0xFFFF);
        let sixteen = g.u32(16);
        let one = g.u32(1);
        let lo = g.and(m, mask);
        let hi = g.shr(m, sixteen);
        let s = g.add(lo, hi);
        let s = g.shr(s, one);
        let (a, b) = (g.u32(32), g.u32(65504));
        g.clamp(s, a, b)
    }
    fn sbit(&mut self, s: St, idx: IId, off: IId, on: Option<IId>) -> (St, IId) {
        let p = self.prior(idx, off);
        self.bit(s, p, on)
    }
    /// A loop of decisions on consecutive models `base + j` while they come
    /// out 1, at most `max` (gated by `on`): returns the count.
    fn unary(&mut self, s: St, base: IId, off: IId, max: u32, on: IId) -> (St, IId) {
        let z = self.u(0);
        let mut b = std::mem::take(&mut self.g);
        let (stream, end, mt) = (self.stream, self.end, self.mt);
        let out = b.while_loop(&[z, s.x, s.pos, on], |g, st| {
            let mut me = B {
                g: std::mem::take(g),
                mt,
                stream,
                end,
                ad: None,
            };
            let mx = me.u(max);
            let more = me.g.lt(st[0], mx);
            let go = me.g.and(st[3], more);
            let idx = me.g.add(base, st[0]);
            let (s2, bit) = me.sbit(
                St {
                    x: st[1],
                    pos: st[2],
                },
                idx,
                off,
                None,
            );
            let one = me.u(1);
            let h1 = me.g.add(st[0], one);
            let h = me.g.select(bit, h1, st[0]);
            *g = me.g;
            (go, vec![h, s2.x, s2.pos, bit])
        });
        b.cap_loop(out[0], max + 1);
        self.g = b;
        (
            St {
                x: out[1],
                pos: out[2],
            },
            out[0],
        )
    }
}

/// The static-prior residual decoder (`res_static_gen`).
pub fn static_graph() -> (IntGraph, Vec<IId>) {
    let mut b = B::new();
    let g = &mut b.g;
    let x = g.param(0, IType::U32);
    let pos = g.param(1, IType::U32);
    let sig = g.param(3, IType::U32);
    let e = g.param(4, IType::U32);
    let cls = g.param(5, IType::U32);
    let sctx = g.param(6, IType::U32);
    let off = g.param(7, IType::U32);
    let s = St { x, pos };
    let sig0 = b.u(SIG0 as u32);
    let si = b.g.add(sig0, sig);
    let (s, en) = b.sbit(s, si, off, None);
    // k = max(e, 2) - 2
    let two = b.u(2);
    let k = b.g.max(e, two);
    let k = b.g.sub(k, two);
    // magnitude unary over MAG0 + (cls·NE + e)·KB
    let ne = b.u(NE as u32);
    let kb = b.u(KB as u32);
    let mag0 = b.u(MAG0 as u32);
    let t = b.g.mul(cls, ne);
    let t = b.g.add(t, e);
    let t = b.g.mul(t, kb);
    let mbase = b.g.add(mag0, t);
    let (s, h) = b.unary(s, mbase, off, KB as u32, en);
    // escape: Elias gamma on ESC0 + cls·32
    let full = b.g.eq(h, kb);
    let esc = b.g.and(en, full);
    let esc0 = b.u(ESC0 as u32);
    let c32 = b.u(32);
    let t = b.g.mul(cls, c32);
    let ebase = b.g.add(esc0, t);
    let (s, nb) = b.unary(s, ebase, off, 31, esc);
    let (s, glo) = b.raw_n(s, nb);
    let one = b.u(1);
    let p = b.g.shl(one, nb);
    let p = b.g.or(p, glo);
    let p = b.g.sub(p, one);
    let hp = b.g.add(h, p);
    let hv = b.g.select(esc, hp, h);
    // low bits: two modelled, then raw
    let low0 = b.u(LOW0 as u32);
    let four = b.u(4);
    let e4 = b.g.mul(e, four);
    let lb = b.g.add(low0, e4);
    let ge1 = b.g.le(one, k);
    let on1 = b.g.and(en, ge1);
    let (s, b1) = b.sbit(s, lb, off, Some(on1));
    let z = b.u(0);
    let lo1 = b.g.select(b1, one, z);
    let ge2 = b.g.le(two, k);
    let on2 = b.g.and(en, ge2);
    let lb2 = b.g.add(lb, two);
    let lb2 = b.g.add(lb2, lo1);
    let (s, b2) = b.sbit(s, lb2, off, Some(on2));
    let l2 = b.g.shl(lo1, one);
    let bv = b.g.select(b2, one, z);
    let l2 = b.g.or(l2, bv);
    let lo = b.g.select(on2, l2, lo1);
    let three = b.u(3);
    let ge3 = b.g.le(three, k);
    let on3 = b.g.and(en, ge3);
    let km2 = b.g.sub(k, two);
    let nraw = b.g.select(on3, km2, z);
    let (s, rv) = b.raw_n(s, nraw);
    let l3 = b.g.shl(lo, nraw);
    let l3 = b.g.or(l3, rv);
    let lo = b.g.select(on3, l3, lo);
    let mv = b.g.shl(hv, k);
    let mv = b.g.or(mv, lo);
    let mv = b.g.add(mv, one);
    // sign
    let sgn0 = b.u(SGN0 as u32);
    let si = b.g.add(sgn0, sctx);
    let (s, neg) = b.sbit(s, si, off, Some(en));
    let mi = b.g.bitcast(IType::I32, mv);
    let nm = b.g.neg(mi);
    let r = b.g.select(neg, nm, mi);
    let zi = b.i(0);
    let r = b.g.select(en, r, zi);
    (b.g, vec![s.x, s.pos, r])
}

impl B {
    fn sig16(&mut self, z: IId, off_sig: IId) -> IId {
        let zero = self.i(0);
        let g = &mut self.g;
        let neg = g.lt(z, zero);
        let nz = g.neg(z);
        let a = g.select(neg, nz, z);
        let a = g.bitcast(IType::U32, a);
        let cap = g.u32(4096);
        let a = g.min(a, cap);
        let i = g.add(off_sig, a);
        let s = g.load(self.mt, i);
        let k = g.u32(65536);
        let c = g.sub(k, s);
        g.select(neg, c, s)
    }
    fn zval(&mut self, d0: IId, ls0: IId, off_pow: IId) -> IId {
        let c = |b: &mut B, v: i32| b.i(v);
        let (lo, hi) = (c(self, -4096), c(self, 6144));
        let ls = self.g.clamp(ls0, lo, hi);
        let eight = self.u(8);
        let i = self.g.shr(ls, eight);
        let m255 = c(self, 255);
        let f = self.g.and(ls, m255);
        let k256 = c(self, 256);
        let fi = self.g.sub(k256, f);
        let fu = self.g.bitcast(IType::U32, fi);
        let pi = self.g.add(off_pow, fu);
        let m = self.g.load(self.mt, pi);
        let m = self.g.bitcast(IType::I32, m);
        let (dl_, dh_) = (c(self, -33554432), c(self, 33554432));
        let d = self.g.clamp(d0, dl_, dh_);
        let sixteen = self.u(16);
        let dh = self.g.shr(d, sixteen);
        let du = self.g.bitcast(IType::U32, d);
        let mask = self.u(0xFFFF);
        let dl = self.g.and(du, mask);
        let a = self.g.mul(dh, m);
        let dli = self.g.bitcast(IType::I32, dl);
        let a = self.g.add(a, dli);
        let k65536 = c(self, 65536);
        let mm = self.g.sub(m, k65536);
        let mmu = self.g.bitcast(IType::U32, mm);
        let t2 = self.g.mul(dl, mmu);
        let h = self.u(32768);
        let t2 = self.g.add(t2, h);
        let t2 = self.g.shr(t2, sixteen);
        let t2 = self.g.bitcast(IType::I32, t2);
        let t = self.g.add(a, t2);
        let one = c(self, 1);
        let sh = self.g.add(i, one);
        let zero = c(self, 0);
        let pos = self.g.le(zero, sh);
        let c31 = c(self, 31);
        let shm = self.g.min(sh, c31);
        let shu = self.g.bitcast(IType::U32, shm);
        let rp = self.g.shr(t, shu);
        let nsh = self.g.neg(sh);
        let k = self.g.bitcast(IType::U32, nsh);
        let c4097 = c(self, 4097);
        let lim = self.g.shr(c4097, k);
        let tl = self.g.shl(t, k);
        let n4097 = c(self, -4097);
        let nlim = self.g.neg(lim);
        let below = self.g.lt(t, nlim);
        let sat = self.g.select(below, n4097, tl);
        let above = self.g.lt(lim, t);
        let sat = self.g.select(above, c4097, sat);
        self.g.select(pos, rp, sat)
    }
}

struct Mix {
    pi: IId,
    mu: [IId; 2],
    ls: [IId; 2],
    off_sig: IId,
    off_pow: IId,
}

impl B {
    fn lcdf(&mut self, mx: &Mix, m: IId) -> IId {
        let (a, b) = (self.i(-65535), self.i(65535));
        let t = self.g.clamp(m, a, b);
        let k256 = self.i(256);
        let t = self.g.mul(t, k256);
        let k128 = self.i(128);
        let t = self.g.add(t, k128);
        let mut c = Vec::with_capacity(2);
        for j in 0..2 {
            let d = self.g.sub(t, mx.mu[j]);
            let z = self.zval(d, mx.ls[j], mx.off_pow);
            c.push(self.sig16(z, mx.off_sig));
        }
        let k = self.u(32768);
        let a = self.g.mul(mx.pi, c[0]);
        let w = self.g.sub(k, mx.pi);
        let b = self.g.mul(w, c[1]);
        let s = self.g.add(a, b);
        let f = self.u(15);
        self.g.shr(s, f)
    }
    fn ratio16(&mut self, num: IId, den: IId) -> IId {
        let g = &mut self.g;
        let z = g.u32(0);
        let nz = g.cmp(tang_expr::int::CmpOp::Ne, den, z);
        let mn = g.min(num, den);
        let k = g.u32(65535);
        let p = g.mul(mn, k);
        let p = g.div(p, den);
        let h = g.u32(32768);
        let p = g.select(nz, p, h);
        let (a, b) = (g.u32(32), g.u32(65504));
        g.clamp(p, a, b)
    }
    fn ltail(&mut self, mx: &Mix, m: IId, neg: IId) -> IId {
        let mi = self.g.bitcast(IType::I32, m);
        let nmi = self.g.neg(mi);
        let one = self.i(1);
        let mm1 = self.g.sub(mi, one);
        let arg = self.g.select(neg, nmi, mm1);
        let c = self.lcdf(mx, arg);
        let k = self.u(65536);
        let up = self.g.sub(k, c);
        let v = self.g.select(neg, c, up);
        let big = self.u(1 << 30);
        let over = self.g.lt(big, m);
        let z = self.u(0);
        self.g.select(over, z, v)
    }
}

/// The learned-mixture residual decoder (`res_learned_gen`).
pub fn learned_graph() -> (IntGraph, Vec<IId>) {
    learned_graph_with(false)
}

/// The learned-mixture residual decoder through the per-lane adaptive
/// probability maps (the adaptive role, see the module docs).
pub fn learned_ad_graph() -> (IntGraph, Vec<IId>) {
    learned_graph_with(true)
}

fn learned_graph_with(adaptive: bool) -> (IntGraph, Vec<IId>) {
    let mut b = B::new();
    let x = b.g.param(0, IType::U32);
    let pos = b.g.param(1, IType::U32);
    let o: Vec<IId> = (0..5).map(|k| b.g.param(3 + k, IType::I32)).collect();
    let off_sig = b.g.param(8, IType::U32);
    let off_pow = b.g.param(9, IType::U32);
    let one = b.u(1);
    let z = b.u(0);
    let mut tok = z;
    if adaptive {
        let g = &mut b.g;
        let buf = g.buffer_rw(RW_AD[0].0, RW_AD[0].1, RW_AD[0].2, RW_AD[0].3);
        let so = g.param(10, IType::U32);
        let wa = g.param(11, IType::U32);
        let lim = g.param(12, IType::U32);
        let mask = g.param(13, IType::U32);
        let on = g.cmp(tang_expr::int::CmpOp::Ne, wa, z);
        let en = std::array::from_fn(|k| {
            let kk = g.u32(k as u32);
            let m = g.shr(mask, kk);
            let m = g.and(m, one);
            let m = g.cmp(tang_expr::int::CmpOp::Ne, m, z);
            g.and(on, m)
        });
        b.ad = Some(Ad {
            buf,
            so,
            wa,
            lim,
            en,
        });
        tok = b.g.token();
    }
    let ad = b.ad;
    // one decision: adaptive (context `ctx`, group `grp`) or plain
    let dec = |b: &mut B,
               s: St,
               tok: IId,
               ctx: IId,
               grp: usize,
               p: IId,
               on: Option<IId>|
     -> (St, IId, IId) {
        if adaptive {
            b.ad_bit(s, tok, ctx, grp, p, on)
        } else {
            let (s, bit) = b.bit(s, p, on);
            (s, bit, tok)
        }
    };
    let pi = b.sig16(o[0], off_sig);
    let pi = b.g.shr(pi, one);
    let mx = Mix {
        pi,
        mu: [o[1], o[2]],
        ls: [o[3], o[4]],
        off_sig,
        off_pow,
    };
    let s = St { x, pos };
    let zi = b.i(0);
    let m1 = b.i(-1);
    let c0 = b.lcdf(&mx, zi);
    let cm1 = b.lcdf(&mx, m1);
    let gt = b.g.lt(cm1, c0);
    let d = b.g.sub(c0, cm1);
    let pz = b.g.select(gt, d, z);
    let k = b.u(65536);
    let nzm = b.g.sub(k, pz);
    let p = b.ratio16(nzm, k);
    let (s, en, tok) = dec(&mut b, s, tok, z, 0, p, None);
    let posm = b.g.sub(k, c0);
    let den = b.g.add(cm1, posm);
    let p = b.ratio16(cm1, den);
    let (s, neg, tok) = dec(&mut b, s, tok, one, 1, p, Some(en));
    // tail(1) from the CDFs already at hand
    let ta0 = b.g.select(neg, cm1, posm);
    // exp-Golomb bucket: tails carried between steps
    let mut gb = std::mem::take(&mut b.g);
    let (stream, end, mt) = (b.stream, b.end, b.mt);
    let mxr = &mx;
    let out = gb.while_loop(&[z, ta0, z, s.x, s.pos, en, tok], |g, st| {
        let mut me = B {
            g: std::mem::take(g),
            mt,
            stream,
            end,
            ad,
        };
        let c30 = me.u(30);
        let more = me.g.lt(st[0], c30);
        let go = me.g.and(st[5], more);
        let one = me.u(1);
        let k1 = me.g.add(st[0], one);
        let m = me.g.shl(one, k1);
        let tb = me.ltail(mxr, m, neg);
        let p = me.ratio16(tb, st[1]);
        let (two, three) = (me.u(2), me.u(3));
        let ctx = me.g.min(st[0], three);
        let ctx = me.g.add(ctx, two);
        let (s2, bit, tk) = dec(
            &mut me,
            St {
                x: st[3],
                pos: st[4],
            },
            st[6],
            ctx,
            2,
            p,
            None,
        );
        let ta = me.g.select(bit, tb, st[1]);
        let kk = me.g.select(bit, k1, st[0]);
        *g = me.g;
        (go, vec![kk, ta, tb, s2.x, s2.pos, bit, tk])
    });
    gb.cap_loop(out[0], 31);
    b.g = gb;
    let (kq, ta, tb, tok) = (out[0], out[1], out[2], out[6]);
    let s = St {
        x: out[3],
        pos: out[4],
    };
    let lo = b.g.shl(one, kq);
    let lm1 = b.g.sub(lo, one);
    let hi = b.g.add(lo, lm1);
    let c30 = b.u(30);
    let lt30 = b.g.lt(kq, c30);
    let endv = b.g.select(lt30, tb, z);
    // mantissa by bisection (context 6 for the first bit, then 7)
    let first = b.g.bool(true);
    let mut gb = std::mem::take(&mut b.g);
    let out = gb.while_loop(&[lo, hi, ta, endv, s.x, s.pos, tok, first], |g, st| {
        let mut me = B {
            g: std::mem::take(g),
            mt,
            stream,
            end,
            ad,
        };
        let lt = me.g.lt(st[0], st[1]);
        let go = me.g.and(en, lt);
        let one = me.u(1);
        let w = me.g.sub(st[1], st[0]);
        let w = me.g.add(w, one);
        let w = me.g.shr(w, one);
        let mid = me.g.add(st[0], w);
        let tm = me.ltail(mxr, mid, neg);
        let z = me.u(0);
        let gt = me.g.lt(st[3], tm);
        let d = me.g.sub(tm, st[3]);
        let up = me.g.select(gt, d, z);
        let gt = me.g.lt(st[3], st[2]);
        let d = me.g.sub(st[2], st[3]);
        let all = me.g.select(gt, d, z);
        let p = me.ratio16(up, all);
        let (c6, c7) = (me.u(6), me.u(7));
        let ctx = me.g.select(st[7], c6, c7);
        let (s2, bit, tk) = dec(
            &mut me,
            St {
                x: st[4],
                pos: st[5],
            },
            st[6],
            ctx,
            3,
            p,
            None,
        );
        let mm1 = me.g.sub(mid, one);
        let nlo = me.g.select(bit, mid, st[0]);
        let ntl = me.g.select(bit, tm, st[2]);
        let nhi = me.g.select(bit, st[1], mm1);
        let nend = me.g.select(bit, st[3], tm);
        let nf = me.g.bool(false);
        *g = me.g;
        (go, vec![nlo, nhi, ntl, nend, s2.x, s2.pos, tk, nf])
    });
    gb.cap_loop(out[0], 31);
    b.g = gb;
    let s = St {
        x: out[4],
        pos: out[5],
    };
    let li = b.g.bitcast(IType::I32, out[0]);
    let nl = b.g.neg(li);
    let r = b.g.select(neg, nl, li);
    let r = b.g.select(en, r, zi);
    let mut outs = vec![s.x, s.pos, r];
    if adaptive {
        outs.push(out[6]);
    }
    (b.g, outs)
}

/// WGSL for `res_static_gen` / `res_learned_gen` from the given graphs
/// (`guarded`: bounds-checked loads, for untrusted graphs). A learned graph
/// with four outputs is the adaptive role: it reads the kernel's `ad_*`
/// state and `adt` bins (set up per block by `ad_reset`).
pub fn wgsl(stat: (&IntGraph, &[IId]), learned: (&IntGraph, &[IId]), guarded: bool) -> String {
    let body = |g: &IntGraph, o: &[IId], p: &[&str], d: &[&str]| {
        if guarded {
            g.to_wgsl_body_guarded(o, p, d, 1)
        } else {
            g.to_wgsl_body(o, p, d, 1)
        }
    };
    let res = ["rx", "rpos", "r", "adk"];
    let mut s = String::from(
        "fn res_static_gen(sig: u32, e: u32, cls: u32, sctx: u32) -> i32 {\n    var r: i32;\n",
    );
    s += &body(
        stat.0,
        stat.1,
        &[
            "rx",
            "rpos",
            "rend",
            "sig",
            "e",
            "cls",
            "sctx",
            "P.off_init",
        ],
        &res[..3],
    );
    s += "    return r;\n}\n\nfn res_learned_gen(o0: i32, o1: i32, o2: i32, o3: i32, o4: i32) -> i32 {\n    var r: i32;\n";
    let mut p = vec![
        "rx",
        "rpos",
        "rend",
        "o0",
        "o1",
        "o2",
        "o3",
        "o4",
        "mt[P.pad4]",
        "mt[P.pad4 + 1u]",
    ];
    if learned.1.len() == OUTS_AD.len() {
        s += "    var adk: u32;\n";
        p.extend(["ad_so", "ad_wa", "ad_lim", "ad_mask"]);
    }
    s += &body(learned.0, learned.1, &p, &res[..learned.1.len()]);
    s += "    return r;\n}\n";
    s
}

/// The built-in residual decoders as WGSL.
pub fn builtin_wgsl() -> &'static str {
    static S: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let (a, ao) = static_graph();
        let (b, bo) = learned_graph();
        wgsl((&a, &ao), (&b, &bo), false)
    })
}

/// [`builtin_wgsl`] with the adaptive learned decoder.
pub fn builtin_wgsl_ad() -> &'static str {
    static S: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let (a, ao) = static_graph();
        let (b, bo) = learned_ad_graph();
        wgsl((&a, &ao), (&b, &bo), false)
    })
}

/// Replace the `// RES_BEGIN` … `// RES_END` region of a kernel source.
pub fn splice_res(src: &str, res: &str) -> String {
    let a = src.find("// RES_BEGIN").expect("RES_BEGIN");
    let b = src.find("// RES_END").expect("RES_END");
    let mut out = format!("{}// RES_BEGIN\n{}{}", &src[..a], res, &src[b..]);
    if out.contains("tx_mulhi_") && !out.contains("fn tx_mulhi_u32") {
        out.push_str(tang_expr::int::WGSL_PRELUDE);
    }
    out
}

/// Stream bytes as the kernels see them: `u32` words, little endian.
pub fn stream_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(b)
        })
        .collect()
}

/// One lane's rANS state (as `lane_start` in the kernel).
#[derive(Clone, Copy, Debug)]
pub struct Lane {
    /// State.
    pub x: u32,
    /// Next 16-bit word index.
    pub pos: u32,
    /// Word bound.
    pub end: u32,
    lo: u32,
}

impl Lane {
    /// Lane over 16-bit words `lo..hi` of `stream`.
    pub fn new(stream: &[u32], lo: u32, hi: u32) -> Self {
        let w = |i: u32| {
            if i < hi {
                (stream.get((i >> 1) as usize).copied().unwrap_or(0) >> ((i & 1) * 16)) & 0xFFFF
            } else {
                0
            }
        };
        let x = if hi.wrapping_sub(lo) >= 2 && hi >= lo {
            w(lo) | w(lo + 1) << 16
        } else {
            1 << 16
        };
        Self {
            x,
            pos: lo + 2,
            end: hi,
            lo,
        }
    }
    /// Whether more words were consumed than the lane holds.
    pub fn overrun(&self) -> bool {
        self.end >= self.lo + 2 && self.pos > self.end
    }
}

/// CPU evaluation of residual graphs (the interpreter; one per thread).
#[derive(Clone)]
pub struct CpuRes {
    prog: tang_expr::int::Program,
    learned: bool,
    off: [u32; 2],
    /// Adaptive role: `[off_stretch, wa, lim, mask]`.
    ad: Option<[u32; 4]>,
}

impl CpuRes {
    /// A static graph (`off_init` into `mt`) or a learned one
    /// (`off_sig`, `off_pow`).
    pub fn new(g: &IntGraph, outs: &[IId], learned: bool, off: [u32; 2]) -> Self {
        Self {
            prog: tang_expr::int::Program::new(g, outs),
            learned,
            off,
            ad: None,
        }
    }
    /// The adaptive learned role with settings `ad` (`off` as for a learned
    /// graph; the stretch table follows the 2^x table, see [`cpu_mt`]).
    pub fn new_ad(g: &IntGraph, outs: &[IId], off: [u32; 2], ad: crate::lmodel3::AdCfg) -> Self {
        let so = off[1] + crate::lmodel3::tables().1.len() as u32;
        Self {
            prog: tang_expr::int::Program::new(g, outs),
            learned: true,
            off,
            ad: Some([so, ad.wa, ad.lim, ad.mask]),
        }
    }
    /// Whether this is the learned role.
    pub fn learned(&self) -> bool {
        self.learned
    }
    /// Decode one static-prior residual.
    #[allow(clippy::too_many_arguments)]
    pub fn res_static(
        &mut self,
        l: &mut Lane,
        mt: &[u32],
        stream: &[u32],
        sig: u32,
        e: u32,
        cls: u32,
        sctx: u32,
    ) -> i32 {
        let p = [l.x, l.pos, l.end, sig, e, cls, sctx, self.off[0]];
        self.run(l, &p, mt, stream)
    }
    /// Decode one learned residual from the net outputs; `adt` is the lane's
    /// adaptive state ([`crate::lmodel3::ad_init`], unused by the plain role).
    pub fn res_learned(
        &mut self,
        l: &mut Lane,
        mt: &[u32],
        stream: &[u32],
        o: &[i32; 5],
        adt: &mut [u32],
    ) -> i32 {
        let a = self.ad.unwrap_or_default();
        let p = [
            l.x,
            l.pos,
            l.end,
            o[0] as u32,
            o[1] as u32,
            o[2] as u32,
            o[3] as u32,
            o[4] as u32,
            self.off[0],
            self.off[1],
            a[0],
            a[1],
            a[2],
            a[3],
        ];
        let mut out = [0u32; 4];
        self.prog.run_rw(&p, &[mt, stream], &mut [adt], &mut out);
        l.x = out[0];
        l.pos = out[1];
        out[2] as i32
    }
    fn run(&mut self, l: &mut Lane, p: &[u32], mt: &[u32], stream: &[u32]) -> i32 {
        let mut o = [0u32; 3];
        self.prog.run(p, &[mt, stream], &mut o);
        l.x = o[0];
        l.pos = o[1];
        o[2] as i32
    }
}

/// `mt` for CPU decoding: priors at 0, then the σ, 2^x and stretch tables
/// (the kernel's layout); returns the buffer and `(off_sig, off_pow)`.
pub fn cpu_mt(init: &[u32]) -> (Vec<u32>, [u32; 2]) {
    let (sig, pw) = crate::lmodel3::tables();
    let mut v = init.to_vec();
    let os = v.len() as u32;
    v.extend_from_slice(sig);
    let op = v.len() as u32;
    v.extend_from_slice(pw);
    v.extend(crate::lmodel3::stretch_tab().iter().map(|&z| z as u32));
    (v, [os, op])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpuwav::{code_res, rans_encode, Feats, RecSink, NMODEL};
    use crate::lmodel3::{code_learned, Mix3};

    fn rng(s: &mut u64) -> u32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 33) as u32
    }

    fn to_bytes(w: &[u16]) -> Vec<u8> {
        w.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn static_graph_decodes_native_encoding() {
        let mut s = 1u64;
        let init: Vec<u32> = (0..NMODEL)
            .map(|_| (rng(&mut s) & 0xFFFF) | (rng(&mut s) & 0xFFFF) << 16)
            .collect();
        let mut syms = RecSink {
            syms: vec![Vec::new()],
        };
        let mut items = Vec::new();
        for _ in 0..4000 {
            let e = (rng(&mut s) % NE as u32) as usize;
            let cls = (rng(&mut s) % 10) as usize;
            let f = Feats {
                taps: [0; crate::gpuwav::NTAP],
                sig: (rng(&mut s) % 240) as usize,
                e,
                sctx: cls * 9 + (rng(&mut s) % 9) as usize,
                raw: [0; 18],
            };
            let r = match rng(&mut s) % 4 {
                0 => 0,
                1 => (rng(&mut s) % 7) as i32 - 3,
                2 => (rng(&mut s) % 70000) as i32 - 35000,
                _ => (rng(&mut s) as i32) >> (rng(&mut s) % 31),
            };
            let mut m = init.clone();
            let got = code_res::<RecSink, false>(&mut syms, 0, &mut m, cls, &f, r);
            assert_eq!(got, r);
            items.push((f, cls, r));
        }
        let words = rans_encode(&syms.syms[0]);
        let mut bytes = vec![0u8, 0, 0, 0, 0, 0]; // offset by 3 words
        bytes.extend(to_bytes(&words));
        let sw = stream_words(&bytes);
        let (g, o) = static_graph();
        let mut cr = CpuRes::new(&g, &o, false, [0, 0]);
        let mut l = Lane::new(&sw, 3, 3 + words.len() as u32);
        for (f, cls, r) in items {
            assert_eq!(
                cr.res_static(
                    &mut l,
                    &init,
                    &sw,
                    f.sig as u32,
                    f.e as u32,
                    cls as u32,
                    f.sctx as u32
                ),
                r
            );
        }
        assert!(!l.overrun());
        assert_eq!(l.pos, l.end);
    }

    #[test]
    fn learned_graph_decodes_native_encoding() {
        let mut s = 2u64;
        let mut syms = RecSink {
            syms: vec![Vec::new()],
        };
        let mut items = Vec::new();
        for _ in 0..3000 {
            let o = [
                (rng(&mut s) % 4000) as i32 - 2000,
                (rng(&mut s) % 6000) as i32 - 3000,
                (rng(&mut s) % 6000) as i32 - 3000,
                (rng(&mut s) % 9000) as i32 - 4096,
                (rng(&mut s) % 9000) as i32 - 4096,
            ];
            let r = match rng(&mut s) % 3 {
                0 => 0,
                1 => (rng(&mut s) % 41) as i32 - 20,
                _ => (rng(&mut s) as i32) >> (rng(&mut s) % 31 + 1),
            };
            let got = code_learned(&mut syms, 0, &Mix3::new(&o), r);
            assert_eq!(got, r);
            items.push((o, r));
        }
        let words = rans_encode(&syms.syms[0]);
        let sw = stream_words(&to_bytes(&words));
        let (mt, off) = cpu_mt(&[]);
        let (g, go) = learned_graph();
        let mut cr = CpuRes::new(&g, &go, true, off);
        let mut l = Lane::new(&sw, 0, words.len() as u32);
        for (o, r) in items {
            assert_eq!(cr.res_learned(&mut l, &mt, &sw, &o, &mut []), r);
        }
        assert_eq!(l.pos, l.end);
    }

    #[test]
    fn learned_ad_graph_decodes_native_encoding() {
        use crate::lmodel3::{ad_init, code_learned_ad, AdCfg, Adapt};
        let (g, go) = learned_ad_graph();
        // and through the wire (the shipped form)
        let w = g.to_bytes(&go);
        let (g2, go2) = IntGraph::from_bytes_rw(
            &w,
            &LEARNED_AD_PARAMS,
            &BUFS,
            &RW_AD,
            &OUTS_AD,
            &tang_expr::int::Limits::default(),
        )
        .unwrap();
        for c in [
            AdCfg {
                wa: 10,
                lim: 14,
                mask: 15,
            },
            AdCfg {
                wa: 16,
                lim: 3,
                mask: 5,
            },
            AdCfg {
                wa: 0,
                lim: 0,
                mask: 0,
            },
        ] {
            let mut s = 3u64;
            let mut syms = RecSink {
                syms: vec![Vec::new()],
            };
            let mut ad = Adapt::new(c);
            let mut items = Vec::new();
            for i in 0..4000 {
                if i % 1000 == 0 {
                    ad = Adapt::new(c);
                }
                let o = [
                    (rng(&mut s) % 4000) as i32 - 2000,
                    (rng(&mut s) % 600) as i32 - 300,
                    (rng(&mut s) % 6000) as i32 - 3000,
                    (rng(&mut s) % 3000) as i32 - 1000,
                    (rng(&mut s) % 9000) as i32 - 4096,
                ];
                let r = match rng(&mut s) % 3 {
                    0 => 0,
                    1 => (rng(&mut s) % 41) as i32 - 20,
                    _ => (rng(&mut s) as i32) >> (rng(&mut s) % 31 + 1),
                };
                assert_eq!(code_learned_ad(&mut syms, 0, &Mix3::new(&o), r, &mut ad), r);
                items.push((o, r));
            }
            let words = rans_encode(&syms.syms[0]);
            let sw = stream_words(&to_bytes(&words));
            let (mt, off) = cpu_mt(&[]);
            for (g, go) in [(&g, &go), (&g2, &go2)] {
                let mut cr = CpuRes::new_ad(g, go, off, c);
                let mut l = Lane::new(&sw, 0, words.len() as u32);
                let mut adt = ad_init();
                for (i, (o, r)) in items.iter().enumerate() {
                    if i % 1000 == 0 {
                        adt = ad_init();
                    }
                    assert_eq!(
                        cr.res_learned(&mut l, &mt, &sw, o, &mut adt),
                        *r,
                        "{c:?} item {i}"
                    );
                }
                assert_eq!(l.pos, l.end);
            }
        }
    }
}
