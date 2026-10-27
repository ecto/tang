//! The forward on the device (M2): [`super::forward`]'s control flow over [`super::gpu`]'s
//! kernels, so every step has a CPU twin and the fixture checks the whole.
//!
//! Correctness first. On the device: every GEMM, norm, RoPE, quant-dequant, attention, mHC,
//! the router, expert MLPs and Engram's gate. Still on the host, each a later fast-path step:
//! the indexer's masks, candidate blocks and top-k (its scores are computed on the device and
//! read back), the per-expert routing lists, Engram's hashing and row gather (the tables stay
//! mapped on the host: 101 GB each for the real model), and DSpark's left-to-right Markov
//! sampling. Weights upload lazily on first use, in their checkpoint format.
//!
//! Ordering: the stream is a blocking one, so kernels run after uploads and memsets (both on
//! the legacy stream; a pageable `cuMemcpyHtoD` returns before its DMA lands). `cuMemFree`
//! still doesn't wait for queued kernels, so every helper that drops a temporary a queued
//! kernel reads syncs first.

use super::forward::{argmax, topk_stable, Model, SpecStep};
use super::gpu::{blocks, DevW, Kernels};
use super::quant::{e4m3_to_f32, e8m0_to_f32};
use anyhow::{Context, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use tang_moe::args;
use tang_moe::gpu::{DevBuf, Gpu, Stream};

const B256: (u32, u32, u32) = (256, 1, 1);

/// An uninitialized f32 buffer for a kernel to fill. Not `DevBuf::zeroed`: its memset runs on
/// the legacy NULL stream, which our non-blocking stream doesn't order against, so it can land
/// after the kernel's writes.
fn fbuf(n: usize) -> Result<DevBuf> {
    Ok(DevBuf::alloc(n.max(1) * 4)?)
}

/// A zeroed f32 buffer, by synchronous host copy (see [`fbuf`]).
fn zbuf(n: usize) -> Result<DevBuf> {
    up(&vec![0f32; n.max(1)])
}

fn up<T: Copy>(v: &[T]) -> Result<DevBuf> {
    Ok(DevBuf::from_slice(v)?)
}

#[derive(Default)]
struct ExpertCache {
    map: HashMap<String, (Rc<DevW>, u64)>,
    tick: u64,
    misses: usize,
}

/// Which attention a block runs.
enum Attn<'a> {
    Main,
    Spark(&'a DevBuf, usize),
}

/// mHC coefficients on the device: pre [T][hc], post [T][hc], comb [T][hc][hc].
struct Mix {
    pre: DevBuf,
    post: DevBuf,
    comb: DevBuf,
}

pub struct GpuModel {
    /// The CPU reference: config, the mapped checkpoint, Engram hashing, host-side selection.
    pub cpu: Model,
    k: Kernels,
    s: Stream,
    _g: Gpu,
    w: RefCell<HashMap<String, Rc<DevW>>>,
    experts: RefCell<ExpertCache>,
    /// Routed-expert tensors kept on the device (`DS_EXPERT_SLOTS`; 0 = all).
    pub expert_slots: usize,
    tabs: Vec<DevBuf>,
    win: Vec<DevBuf>,
    ckv: HashMap<usize, DevBuf>,
    idxk: HashMap<usize, DevBuf>,
    cst: HashMap<usize, (DevBuf, DevBuf)>,
    qk: RefCell<HashMap<usize, Rc<DevBuf>>>,
    sh_ckv: usize,
    sh_idxk: usize,
    topk: Vec<Vec<i32>>,
    cand: Vec<Vec<bool>>,
    /// Debugging: each backbone layer's input stream and pre-mix (as [`Model::force`]).
    pub force: Option<Vec<(Vec<f32>, Vec<f32>)>>,
    /// Intermediates read back by name when `DS_TAP` is set (as [`Model::taps`]).
    pub taps: RefCell<Vec<(String, Vec<f32>)>>,
    /// While a [`GpuSnapshot`] is open: the compressors' partial-group writes (as
    /// [`Model::snapshot`]): lid, position, kv row, score row.
    journal: Option<Vec<(usize, usize, DevBuf, DevBuf)>>,
}

/// As [`super::forward::Snapshot`], on the device.
pub struct GpuSnapshot {
    win: Vec<DevBuf>,
    cst: HashMap<usize, (DevBuf, DevBuf)>,
}

impl GpuModel {
    pub fn open(dir: &Path) -> Result<Self> {
        let g = Gpu::new()?;
        g.bind()?;
        let k = Kernels::new(&g)?;
        // blocking: weight uploads (pageable `cuMemcpyHtoD`) finish their DMA on the legacy
        // stream, which only a blocking stream orders behind
        let s = Stream::blocking()?;
        let cpu = Model::open(dir)?;
        let c = cpu.cfg.clone();
        let (hd, sl) = (c.head_dim, c.max_seq_len);
        let tabs = cpu
            .freqs
            .iter()
            .map(|t| up(&t.iter().flat_map(|&(a, b)| [a, b]).collect::<Vec<f32>>()))
            .collect::<Result<_>>()?;
        let n_all = c.n_layers + c.n_mtp_layers;
        let win = (0..n_all)
            .map(|_| zbuf(c.window_size * hd))
            .collect::<Result<_>>()?;
        let mut ckv = HashMap::new();
        let mut idxk = HashMap::new();
        let mut cst = HashMap::new();
        for &l in &c.kv_source_layers {
            let r = c.compress_ratios[l];
            ckv.insert(l, zbuf(sl / r * hd)?);
            idxk.insert(l, zbuf(sl / r * c.index_head_dim)?);
            if r > 1 {
                cst.insert(l, (zbuf(r * hd)?, up(&vec![f32::NEG_INFINITY; r * hd])?));
            }
        }
        Ok(Self {
            cpu,
            k,
            s,
            _g: g,
            w: RefCell::new(HashMap::new()),
            experts: RefCell::new(ExpertCache::default()),
            // an expert MLP holds three tensors at once
            expert_slots: match std::env::var("DS_EXPERT_SLOTS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
            {
                Some(0) | None => 0,
                Some(n) => n.max(3),
            },
            tabs,
            win,
            ckv,
            idxk,
            cst,
            qk: RefCell::new(HashMap::new()),
            sh_ckv: 0,
            sh_idxk: 0,
            topk: Vec::new(),
            cand: Vec::new(),
            force: None,
            taps: RefCell::new(Vec::new()),
            journal: None,
        })
    }

    /// Forget the sequence: the next forward must start at position 0. The caches need no
    /// clearing (nothing past a sequence's own positions is ever read); the ratio-2
    /// compressors' partial groups do.
    pub fn reset(&mut self) -> Result<()> {
        let hd = self.cpu.cfg.head_dim;
        for (&l, st) in self.cst.iter_mut() {
            let r = self.cpu.cfg.compress_ratios[l];
            *st = (zbuf(r * hd)?, up(&vec![f32::NEG_INFINITY; r * hd])?);
        }
        self.sh_ckv = 0;
        self.sh_idxk = 0;
        self.topk.clear();
        self.cand.clear();
        Ok(())
    }

    fn dw(&self, name: &str) -> Result<Rc<DevW>> {
        if name.contains(".ffn.experts.") {
            return self.expert_w(name);
        }
        if let Some(w) = self.w.borrow().get(name) {
            return Ok(w.clone());
        }
        let w = Rc::new(DevW::upload(&self.cpu.st, name)?);
        self.w.borrow_mut().insert(name.to_string(), w.clone());
        Ok(w)
    }

    /// A routed expert's weight through the LRU of `expert_slots` tensors (unbounded when 0),
    /// uploaded from the mapped checkpoint on a miss. Evicting frees device memory, which is
    /// safe here: every expert GEMM syncs before returning (`lin` on a quantized weight).
    fn expert_w(&self, name: &str) -> Result<Rc<DevW>> {
        let mut ex = self.experts.borrow_mut();
        ex.tick += 1;
        let tick = ex.tick;
        if let Some((w, t)) = ex.map.get_mut(name) {
            *t = tick;
            return Ok(w.clone());
        }
        if self.expert_slots > 0 && ex.map.len() >= self.expert_slots {
            let oldest = ex
                .map
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone())
                .unwrap();
            ex.map.remove(&oldest);
        }
        ex.misses += 1;
        let w = Rc::new(DevW::upload(&self.cpu.st, name)?);
        ex.map.insert(name.to_string(), (w.clone(), tick));
        Ok(w)
    }

    /// Routed-expert tensors uploaded so far (cache misses).
    pub fn expert_misses(&self) -> usize {
        self.experts.borrow().misses
    }

    fn tap(&self, name: &str, b: &DevBuf, n: usize) -> Result<()> {
        if std::env::var_os("DS_TAP").is_some() {
            let v = self.read(b, n)?;
            self.taps.borrow_mut().push((name.to_string(), v));
        }
        Ok(())
    }

    fn read(&self, b: &DevBuf, n: usize) -> Result<Vec<f32>> {
        self.s.sync()?;
        Ok(b.read::<f32>(0, n)?)
    }

    fn copy(&self, dst: u64, src: u64, n: usize) -> Result<()> {
        let nn = n as i64;
        unsafe {
            self.k.go(
                "ds_copy",
                blocks(n, 256),
                B256,
                0,
                &self.s,
                args![dst, src, nn],
            )
        }
    }

    /// `linear`: quantized weights get an fp8-quantized copy of the activation; bf16 out.
    fn lin(&self, x: &DevBuf, t: usize, name: &str) -> Result<DevBuf> {
        let w = self.dw(name)?;
        let (k, n) = (w.v.cols as usize, w.v.rows as usize);
        let y = fbuf(t * n)?;
        if w.v.fmt >= 2 {
            let xq = fbuf(t * k)?;
            self.copy(xq.ptr, x.ptr, t * k)?;
            self.k.fp8_qdq(&self.s, xq.ptr, t * k)?;
            self.k.linear(&self.s, &w.v, xq.ptr, y.ptr, t, true)?;
            // cuMemFree doesn't wait for this stream: drain before `xq` drops
            self.s.sync()?;
        } else {
            self.k.linear(&self.s, &w.v, x.ptr, y.ptr, t, true)?;
        }
        Ok(y)
    }

    /// An f32 GEMM with no activation quantization or rounding (router, mHC, compressor > 1,
    /// the head).
    fn raw(&self, x: u64, t: usize, name: &str) -> Result<DevBuf> {
        let w = self.dw(name)?;
        let y = fbuf(t * w.v.rows as usize)?;
        self.k.linear(&self.s, &w.v, x, y.ptr, t, false)?;
        Ok(y)
    }

    fn rms(&self, x: &DevBuf, rows: usize, name: &str) -> Result<()> {
        let w = self.dw(name)?;
        self.k
            .rms(&self.s, x.ptr, &w.v, rows, self.cpu.cfg.norm_eps)
    }

    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        x: &DevBuf,
        rows: usize,
        per_row: usize,
        d: usize,
        lid: usize,
        pos0: usize,
        pstride: usize,
        fixed: Option<usize>,
        inverse: bool,
    ) -> Result<()> {
        let rd = self.cpu.cfg.rope_head_dim;
        self.k.rope(
            &self.s,
            x.ptr,
            self.tabs[lid].ptr,
            rows,
            per_row,
            d,
            rd,
            pos0,
            pstride,
            fixed,
            inverse,
        )
    }

    // attention

    fn attention(
        &mut self,
        p: &str,
        lid: usize,
        x: &DevBuf,
        t: usize,
        sp: usize,
    ) -> Result<DevBuf> {
        let c = self.cpu.cfg.clone();
        let (h, hd) = (c.n_heads, c.head_dim);
        let qr = self.lin(x, t, &format!("{p}.attn.wq_a.weight"))?;
        self.rms(&qr, t, &format!("{p}.attn.q_norm.weight"))?;
        self.tap(&format!("{p}.qr"), &qr, t * c.q_lora_rank)?;
        let q = self.lin(&qr, t, &format!("{p}.attn.wq_b.weight"))?;
        self.tap(&format!("{p}.q0"), &q, t * h * hd)?;
        self.rope(&q, t, h, hd, lid, sp, 1, None, false)?;
        self.tap(&format!("{p}.q"), &q, t * h * hd)?;
        let (kv, nkv, mut idxs) = self.window_kv(p, lid, x, t, sp)?;
        self.tap(&format!("{p}.wkv"), &kv, nkv * hd)?;
        let all = if c.compress_ratios[lid] > 0 {
            let (src, clen, cidx) = self.compress_kv(p, lid, x, &qr, t, sp, nkv)?;
            for (a, b) in idxs.iter_mut().zip(cidx) {
                a.extend(b);
            }
            let all = fbuf((nkv + clen) * hd)?;
            self.copy(all.ptr, kv.ptr, nkv * hd)?;
            self.copy(
                all.ptr + (nkv * hd * 4) as u64,
                self.ckv[&src].ptr,
                clen * hd,
            )?;
            all
        } else {
            kv
        };
        self.attn_out(p, &q, &all, &idxs, lid, sp)
    }

    fn attn_out(
        &self,
        p: &str,
        q: &DevBuf,
        kv: &DevBuf,
        idxs: &[Vec<i32>],
        lid: usize,
        pos0: usize,
    ) -> Result<DevBuf> {
        let c = &self.cpu.cfg;
        let (h, hd, g, ol) = (c.n_heads, c.head_dim, c.o_groups, c.o_lora_rank);
        let t = idxs.len();
        let k = idxs.iter().map(|r| r.len()).max().unwrap_or(0).max(1);
        let flat: Vec<i32> = idxs
            .iter()
            .flat_map(|r| r.iter().copied().chain(std::iter::repeat(-1)).take(k))
            .collect();
        let di = up(&flat)?;
        let sink = self.dw(&format!("{p}.attn.attn_sink"))?;
        let o = fbuf(t * h * hd)?;
        self.k.sparse_attn(
            &self.s,
            q.ptr,
            kv.ptr,
            sink.v.w,
            di.ptr,
            o.ptr,
            t,
            h,
            hd,
            k,
            (hd as f32).powf(-0.5),
        )?;
        self.tap(&format!("{p}.sa"), &o, t * h * hd)?;
        self.rope(&o, t, h, hd, lid, pos0, 1, None, true)?;
        self.k.fp8_qdq(&self.s, o.ptr, t * h * hd)?;
        let wo_a = self.dw(&format!("{p}.attn.wo_a.weight"))?;
        let mid = fbuf(t * g * ol)?;
        let (wv, ti, gi, oli) = (wo_a.v, t as i32, g as i32, ol as i32);
        unsafe {
            self.k.go(
                "ds_wo_a",
                blocks(t * g * ol * 32, 256),
                B256,
                0,
                &self.s,
                args![o.ptr, wv, mid.ptr, ti, gi, oli],
            )?
        };
        // the kernels read `di` asynchronously: drain before it drops
        let out = self.lin(&mid, t, &format!("{p}.attn.wo_b.weight"))?;
        self.s.sync()?;
        Ok(out)
    }

    fn ring_seed(&self, lid: usize, kv: &DevBuf, t: usize) -> Result<()> {
        let (hd, win) = (self.cpu.cfg.head_dim, self.cpu.cfg.window_size);
        let ring = self.win[lid].ptr;
        if t <= win {
            return self.copy(ring, kv.ptr, t * hd);
        }
        let cut = t % win;
        let tail = kv.ptr + ((t - win) * hd * 4) as u64;
        self.copy(ring + (cut * hd * 4) as u64, tail, (win - cut) * hd)?;
        self.copy(ring, tail + ((win - cut) * hd * 4) as u64, cut * hd)
    }

    fn window_kv(
        &mut self,
        p: &str,
        lid: usize,
        x: &DevBuf,
        t: usize,
        sp: usize,
    ) -> Result<(DevBuf, usize, Vec<Vec<i32>>)> {
        let c = self.cpu.cfg.clone();
        let (hd, win) = (c.head_dim, c.window_size);
        let kv = self.lin(x, t, &format!("{p}.attn.wkv.weight"))?;
        self.rms(&kv, t, &format!("{p}.attn.kv_norm.weight"))?;
        self.rope(&kv, t, 1, hd, lid, sp, 1, None, false)?;
        self.k.fp8_qdq(&self.s, kv.ptr, t * hd)?;
        if sp == 0 {
            self.ring_seed(lid, &kv, t)?;
            let n = t.min(win);
            let idxs = (0..t)
                .map(|e| {
                    let lo = (e as i64 - win as i64 + 1).max(0);
                    (0..n as i64)
                        .map(|j| {
                            if lo + j > e as i64 {
                                -1
                            } else {
                                (lo + j) as i32
                            }
                        })
                        .collect()
                })
                .collect();
            return Ok((kv, t, idxs));
        }
        let ring = self.win[lid].ptr;
        if t > 1 {
            // a verify window (as forward.rs): the pre-chunk ring then the chunk's rows
            let all = fbuf((win + t) * hd)?;
            self.copy(all.ptr, ring, win * hd)?;
            self.copy(all.ptr + (win * hd * 4) as u64, kv.ptr, t * hd)?;
            let idxs = (0..t)
                .map(|e| {
                    let q = sp + e;
                    ((q + 1).saturating_sub(win)..=q)
                        .map(|j| if j < sp { j % win } else { win + j - sp } as i32)
                        .collect()
                })
                .collect();
            for e in 0..t {
                let s = ((sp + e) % win * hd * 4) as u64;
                self.copy(ring + s, kv.ptr + (e * hd * 4) as u64, hd)?;
            }
            self.s.sync()?;
            return Ok((all, win + t, idxs));
        }
        self.copy(ring + ((sp % win) * hd * 4) as u64, kv.ptr, hd)?;
        let all = fbuf(win * hd)?;
        self.copy(all.ptr, ring, win * hd)?;
        let oldest = sp % win + 1;
        let idxs = (oldest..win)
            .chain(0..oldest)
            .map(|i| if i > sp { -1 } else { i as i32 })
            .collect();
        Ok((all, win, vec![idxs]))
    }

    /// The source's new compressed latents (pre-RoPE) and how many.
    fn compressor(
        &mut self,
        p: &str,
        lid: usize,
        x: &DevBuf,
        t: usize,
        sp: usize,
    ) -> Result<Option<(DevBuf, usize)>> {
        let c = self.cpu.cfg.clone();
        let (r, hd) = (c.compress_ratios[lid], c.head_dim);
        let norm = format!("{p}.attn.compressor.norm.weight");
        if r == 1 {
            let kv = self.lin(x, t, &format!("{p}.attn.compressor.wkv.weight"))?;
            self.rms(&kv, t, &norm)?;
            return Ok(Some((kv, t)));
        }
        let kv = self.raw(x.ptr, t, &format!("{p}.attn.compressor.wkv.weight"))?;
        let sc = self.raw(x.ptr, t, &format!("{p}.attn.compressor.wgate.weight"))?;
        let (kst, sst) = {
            let (a, b) = &self.cst[&lid];
            (a.ptr, b.ptr)
        };
        let row = (hd * 4) as u64;
        let pool = |kvp: u64, scp: u64, groups: usize| -> Result<DevBuf> {
            let out = fbuf(groups * hd)?;
            let (gi, ri, hdi) = (groups as i32, r as i32, hd as i32);
            unsafe {
                self.k.go(
                    "ds_pool",
                    blocks(groups * hd, 256),
                    B256,
                    0,
                    &self.s,
                    args![kvp, scp, out.ptr, gi, ri, hdi],
                )?
            };
            let n = (groups * hd) as i64;
            unsafe {
                self.k.go(
                    "ds_round",
                    blocks(groups * hd, 256),
                    B256,
                    0,
                    &self.s,
                    args![out.ptr, n],
                )?
            };
            Ok(out)
        };
        let (out, n) = if sp == 0 {
            if t < r {
                self.copy(kst, kv.ptr, t * hd)?;
                self.copy(sst, sc.ptr, t * hd)?;
                self.s.sync()?;
                return Ok(None);
            }
            let rem = t % r;
            let cut = t - rem;
            if rem > 0 {
                self.copy(kst, kv.ptr + cut as u64 * row, rem * hd)?;
                self.copy(sst, sc.ptr + cut as u64 * row, rem * hd)?;
            }
            (pool(kv.ptr, sc.ptr, cut / r)?, cut / r)
        } else {
            let slot = (sp % r) as u64;
            self.copy(kst + slot * row, kv.ptr, hd)?;
            self.copy(sst + slot * row, sc.ptr, hd)?;
            if self.journal.is_some() {
                let (jk, js) = (fbuf(hd)?, fbuf(hd)?);
                self.copy(jk.ptr, kv.ptr, hd)?;
                self.copy(js.ptr, sc.ptr, hd)?;
                self.journal.as_mut().unwrap().push((lid, sp, jk, js));
            }
            if (sp + 1) % r != 0 {
                self.s.sync()?;
                return Ok(None);
            }
            (pool(kst, sst, 1)?, 1)
        };
        self.rms(&out, n, &norm)?;
        self.s.sync()?;
        Ok(Some((out, n)))
    }

    #[allow(clippy::too_many_arguments)]
    fn compress_kv(
        &mut self,
        p: &str,
        lid: usize,
        x: &DevBuf,
        qr: &DevBuf,
        t: usize,
        sp: usize,
        offset: usize,
    ) -> Result<(usize, usize, Vec<Vec<i32>>)> {
        let c = self.cpu.cfg.clone();
        let (r, hd) = (c.compress_ratios[lid], c.head_dim);
        if sp > 0 && t > 1 {
            // a verify window one position at a time (as forward.rs `compress_kv_steps`)
            let (d, ql) = (c.dim, c.q_lora_rank);
            let topk_in = std::mem::take(&mut self.topk);
            let cand_in = std::mem::take(&mut self.cand);
            let (mut idxs, mut topk, mut cand, mut last) =
                (Vec::new(), Vec::new(), Vec::new(), (0, 0));
            for e in 0..t {
                self.topk = topk_in.get(e).cloned().into_iter().collect();
                self.cand = cand_in.get(e).cloned().into_iter().collect();
                let (xe, qe) = (fbuf(d)?, fbuf(ql)?);
                self.copy(xe.ptr, x.ptr + (e * d * 4) as u64, d)?;
                self.copy(qe.ptr, qr.ptr + (e * ql * 4) as u64, ql)?;
                let (src, clen, mut ie) = self.compress_kv(p, lid, &xe, &qe, 1, sp + e, offset)?;
                self.s.sync()?;
                last = (src, clen);
                idxs.push(ie.pop().unwrap_or_default());
                topk.append(&mut self.topk);
                cand.append(&mut self.cand);
            }
            self.topk = topk;
            self.cand = cand;
            return Ok((last.0, last.1, idxs));
        }
        let clen = (sp + t) / r;
        let mut latent = None;
        if c.is_kv_source(lid) {
            latent = self.compressor(p, lid, x, t, sp)?;
            self.sh_ckv = lid;
        }
        let idxs = if c.is_index_source(lid) {
            let idxs = if clen == 0 {
                vec![Vec::new(); t]
            } else {
                self.indexer(p, lid, x, qr, latent.as_ref(), t, sp, offset)?
            };
            self.topk = idxs.clone();
            idxs
        } else {
            self.topk.clone()
        };
        if let Some((lat, n)) = latent {
            let (pos0, stride, fixed) = if sp == 0 {
                (0, r, None)
            } else {
                (0, 0, Some(sp + 1 - r))
            };
            self.rope(&lat, n, 1, hd, lid, pos0, stride, fixed, false)?;
            self.k.fp4_qdq(&self.s, lat.ptr, n * hd, 16, true)?;
            self.copy(
                self.ckv[&lid].ptr + ((sp / r) * hd * 4) as u64,
                lat.ptr,
                n * hd,
            )?;
            self.s.sync()?;
        }
        Ok((self.sh_ckv, clen, idxs))
    }

    #[allow(clippy::too_many_arguments)]
    fn indexer(
        &mut self,
        p: &str,
        lid: usize,
        x: &DevBuf,
        qr: &DevBuf,
        latent: Option<&(DevBuf, usize)>,
        t: usize,
        sp: usize,
        offset: usize,
    ) -> Result<Vec<Vec<i32>>> {
        let c = self.cpu.cfg.clone();
        let (r, nh, ihd) = (c.compress_ratios[lid], c.index_n_heads, c.index_head_dim);
        let end = sp + t;
        let owns = c.is_kv_source(lid);
        if let (true, Some((lat, n))) = (owns, latent) {
            let k = self.lin(lat, *n, &format!("{p}.attn.indexer.wk.weight"))?;
            self.rms(&k, *n, &format!("{p}.attn.indexer.k_norm.weight"))?;
            let (pos0, stride, fixed) = if sp == 0 {
                (0, r, None)
            } else {
                (0, 0, Some(sp + 1 - r))
            };
            self.rope(&k, *n, 1, ihd, lid, pos0, stride, fixed, false)?;
            self.k.fp4_qdq(&self.s, k.ptr, n * ihd, 32, false)?;
            self.copy(
                self.idxk[&lid].ptr + ((sp / r) * ihd * 4) as u64,
                k.ptr,
                n * ihd,
            )?;
            self.s.sync()?;
        }
        if owns {
            self.sh_idxk = lid;
        }
        let q = self.lin(qr, t, &format!("{p}.attn.indexer.wq_b.weight"))?;
        self.rope(&q, t, nh, ihd, lid, sp, 1, None, false)?;
        self.k.fp4_qdq(&self.s, q.ptr, t * nh * ihd, 32, false)?;
        let width = end / r;
        let wts = self.lin(x, t, &format!("{p}.attn.indexer.weights_proj.weight"))?;
        let ws = ((ihd as f64).powf(-0.5) * (nh as f64).powf(-0.5)) as f32;
        let nw = (t * nh) as i64;
        unsafe {
            self.k.go(
                "ds_scale_round",
                blocks(t * nh, 256),
                B256,
                0,
                &self.s,
                args![wts.ptr, ws, nw],
            )?
        };
        let score = fbuf(t * width)?;
        let (ti, wi, nhi, ihdi) = (t as i32, width as i32, nh as i32, ihd as i32);
        let ik = self.idxk[&self.sh_idxk].ptr;
        unsafe {
            self.k.go(
                "ds_index_score",
                blocks(t * width * 32, 256),
                B256,
                0,
                &self.s,
                args![q.ptr, ik, wts.ptr, score.ptr, ti, wi, nhi, ihdi],
            )?
        };
        let mut score = self.read(&score, t * width)?;
        // host: masks, candidate blocks, top-k (as forward.rs)
        let lens: Vec<usize> = (0..t)
            .map(|ti| if sp == 0 { (ti + 1) / r } else { end / r })
            .collect();
        if sp == 0 {
            for ti in 0..t {
                for j in lens[ti]..width {
                    score[ti * width + j] = f32::NEG_INFINITY;
                }
            }
        }
        let src = c.candidate_source_layer;
        if lid as i64 == src {
            self.cand = self.cpu.candidates(&score, width, &lens);
        } else if src >= 0 && (src as usize) < lid {
            for ti in 0..t {
                for j in 0..width {
                    if !self.cand[ti][j] {
                        score[ti * width + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let k = c.index_topk.min(width);
        Ok((0..t)
            .map(|ti| {
                let mut sel = topk_stable(&score[ti * width..(ti + 1) * width], k);
                sel.sort();
                sel.into_iter()
                    .map(|i| {
                        if i < lens[ti] {
                            (i + offset) as i32
                        } else {
                            -1
                        }
                    })
                    .collect()
            })
            .collect())
    }

    // moe

    fn swiglu(&self, g: &DevBuf, u: &DevBuf, w: u64, n: usize, inter: usize) -> Result<DevBuf> {
        let h = fbuf(n * inter)?;
        let (nn, ii, lim) = ((n * inter) as i64, inter as i32, self.cpu.cfg.swiglu_limit);
        unsafe {
            self.k.go(
                "ds_swiglu",
                blocks(n * inter, 256),
                B256,
                0,
                &self.s,
                args![g.ptr, u.ptr, w, h.ptr, nn, ii, lim],
            )?
        };
        Ok(h)
    }

    fn mlp(&self, p: &str, x: &DevBuf, n: usize, w: u64) -> Result<DevBuf> {
        let g = self.lin(x, n, &format!("{p}.w1.weight"))?;
        let u = self.lin(x, n, &format!("{p}.w3.weight"))?;
        let inter = self.dw(&format!("{p}.w1.weight"))?.v.rows as usize;
        let h = self.swiglu(&g, &u, w, n, inter)?;
        self.lin(&h, n, &format!("{p}.w2.weight"))
    }

    fn moe(&self, p: &str, x: &DevBuf, t: usize, lid: usize) -> Result<DevBuf> {
        let c = &self.cpu.cfg;
        let (ne, topk) = c.moe(lid);
        let d = c.dim;
        let raw = self.raw(x.ptr, t, &format!("{p}.ffn.gate.weight"))?;
        let bias = self.dw(&format!("{p}.ffn.gate.bias"))?;
        let (ids, wts) = (DevBuf::alloc(t * topk * 4)?, fbuf(t * topk)?);
        let (ti, nei, ki, norm, rs) = (
            t as i32,
            ne as i32,
            topk as i32,
            c.norm_topk_prob as i32,
            c.route_scale,
        );
        unsafe {
            self.k.go(
                "ds_router",
                blocks(t, 64),
                (64, 1, 1),
                0,
                &self.s,
                args![raw.ptr, bias.v.w, ids.ptr, wts.ptr, ti, nei, ki, norm, rs],
            )?
        };
        self.s.sync()?;
        let (ids, wts) = (ids.read::<i32>(0, t * topk)?, wts.read::<f32>(0, t * topk)?);
        let mut routes: Vec<Vec<(i32, f32)>> = vec![Vec::new(); ne];
        for ti in 0..t {
            for j in 0..topk {
                routes[ids[ti * topk + j] as usize].push((ti as i32, wts[ti * topk + j]));
            }
        }
        let y = zbuf(t * d)?;
        for (e, rs) in routes.iter().enumerate() {
            if rs.is_empty() {
                continue;
            }
            let n = rs.len();
            let rows = up(&rs.iter().map(|r| r.0).collect::<Vec<i32>>())?;
            let w = up(&rs.iter().map(|r| r.1).collect::<Vec<f32>>())?;
            let xs = fbuf(n * d)?;
            let (ni, di) = (n as i32, d as i32);
            unsafe {
                self.k.go(
                    "ds_gather_rows",
                    blocks(n * d, 256),
                    B256,
                    0,
                    &self.s,
                    args![x.ptr, rows.ptr, xs.ptr, ni, di],
                )?
            };
            let out = self.mlp(&format!("{p}.ffn.experts.{e}"), &xs, n, w.ptr)?;
            unsafe {
                self.k.go(
                    "ds_scatter_add",
                    blocks(n * d, 256),
                    B256,
                    0,
                    &self.s,
                    args![y.ptr, out.ptr, rows.ptr, ni, di],
                )?
            };
            self.s.sync()?;
        }
        let sh = self.mlp(&format!("{p}.ffn.shared_experts"), x, t, 0)?;
        let nn = (t * d) as i64;
        unsafe {
            self.k.go(
                "ds_add_round",
                blocks(t * d, 256),
                B256,
                0,
                &self.s,
                args![y.ptr, sh.ptr, nn],
            )?
        };
        self.s.sync()?;
        Ok(y)
    }

    // mhc and blocks

    fn hc_mixes(&self, p: &str, kind: &str, x: &DevBuf, t: usize) -> Result<Mix> {
        let c = &self.cpu.cfg;
        let (hc, d) = (c.hc_mult, c.dim);
        let mix = (2 + hc) * hc;
        let mixes = self.raw(x.ptr, t, &format!("{p}.hc_{kind}_fn"))?;
        let (n, mi, eps) = ((hc * d) as i32, mix as i32, c.norm_eps);
        unsafe {
            self.k.go(
                "ds_hc_scale",
                (t as u32, 1, 1),
                B256,
                0,
                &self.s,
                args![x.ptr, mixes.ptr, n, mi, eps],
            )?
        };
        let m = Mix {
            pre: fbuf(t * hc)?,
            post: fbuf(t * hc)?,
            comb: fbuf(t * hc * hc)?,
        };
        let sc = self.dw(&format!("{p}.hc_{kind}_scale"))?;
        let base = self.dw(&format!("{p}.hc_{kind}_base"))?;
        self.k.sinkhorn(
            &self.s,
            mixes.ptr,
            sc.v.w,
            base.v.w,
            m.pre.ptr,
            m.post.ptr,
            m.comb.ptr,
            t,
            hc,
            c.hc_sinkhorn_iters,
            c.hc_eps,
        )?;
        self.s.sync()?;
        Ok(m)
    }

    fn hc_pre(&self, x: &DevBuf, pre: &DevBuf, t: usize) -> Result<DevBuf> {
        let (hc, d) = (self.cpu.cfg.hc_mult, self.cpu.cfg.dim);
        let out = fbuf(t * d)?;
        let (ti, hci, di) = (t as i32, hc as i32, d as i32);
        unsafe {
            self.k.go(
                "ds_hc_pre",
                blocks(t * d, 256),
                B256,
                0,
                &self.s,
                args![x.ptr, pre.ptr, out.ptr, ti, hci, di],
            )?
        };
        Ok(out)
    }

    fn hc_post(&self, h: &DevBuf, res: &DevBuf, m: &Mix, t: usize) -> Result<DevBuf> {
        let (hc, d) = (self.cpu.cfg.hc_mult, self.cpu.cfg.dim);
        let out = fbuf(t * hc * d)?;
        let (ti, hci, di) = (t as i32, hc as i32, d as i32);
        unsafe {
            self.k.go(
                "ds_hc_post",
                blocks(t * hc * d, 256),
                B256,
                0,
                &self.s,
                args![h.ptr, res.ptr, m.post.ptr, m.comb.ptr, out.ptr, ti, hci, di],
            )?
        };
        self.s.sync()?;
        Ok(out)
    }

    fn block(
        &mut self,
        p: &str,
        lid: usize,
        x: DevBuf,
        t: usize,
        sp: usize,
        pre_mix: &DevBuf,
        attn: Attn,
    ) -> Result<(DevBuf, DevBuf)> {
        let am = self.hc_mixes(p, "attn", &x, t)?;
        let h = self.hc_pre(&x, pre_mix, t)?;
        self.tap(&format!("{p}.attn_pre"), &h, t * self.cpu.cfg.dim)?;
        self.rms(&h, t, &format!("{p}.attn_norm.weight"))?;
        self.tap(&format!("{p}.attn_in"), &h, t * self.cpu.cfg.dim)?;
        let d = self.cpu.cfg.dim;
        let h = match attn {
            Attn::Main => self.attention(p, lid, &h, t, sp)?,
            Attn::Spark(main_x, mt) => self.dspark_attention(p, lid, &h, t, sp, main_x, mt)?,
        };
        self.tap(&format!("{p}.attn"), &h, t * d)?;
        let x = self.hc_post(&h, &x, &am, t)?;
        let fm = self.hc_mixes(p, "ffn", &x, t)?;
        let h = self.hc_pre(&x, &am.pre, t)?;
        self.rms(&h, t, &format!("{p}.ffn_norm.weight"))?;
        self.tap(&format!("{p}.moe_in"), &h, t * d)?;
        let h = self.moe(p, &h, t, lid)?;
        self.tap(&format!("{p}.moe"), &h, t * d)?;
        let x = self.hc_post(&h, &x, &fm, t)?;
        Ok((x, fm.pre))
    }

    // engram

    /// Rows of an engram table, decoded on the host from the mapped checkpoint (fp8 with a
    /// per-row × 32 e8m0 scale): only the rows a step touches are ever read.
    fn engram_rows(&self, name: &str, rows: &[i64]) -> Result<Vec<f32>> {
        let info = self.cpu.st.info(name)?;
        let ehd = info.shape[1];
        let w = self.cpu.st.bytes(name)?;
        let s = self
            .cpu
            .st
            .bytes(&format!("{}scale", name.strip_suffix("weight").unwrap()))?;
        let mut out = Vec::with_capacity(rows.len() * ehd);
        for &r in rows {
            let r = r as usize;
            for i in 0..ehd {
                out.push(e4m3_to_f32(w[r * ehd + i]) * e8m0_to_f32(s[r * (ehd / 32) + i / 32]));
            }
        }
        Ok(out)
    }

    fn engram(&self, p: &str, lid: usize, x: &DevBuf, hashes: &[Vec<i64>]) -> Result<DevBuf> {
        let c = &self.cpu.cfg;
        let (hc, d, t) = (c.hc_mult, c.dim, hashes.len());
        let flat: Vec<i64> = hashes.iter().flatten().copied().collect();
        let rows = up(&self.engram_rows(&format!("{p}.engram.embed.weight"), &flat)?)?;
        self.tap(
            &format!("{p}.eg_rows"),
            &rows,
            flat.len() * self.cpu.st.info(&format!("{p}.engram.embed.weight"))?.shape[1],
        )?;
        let kv = self.lin(&rows, t, &format!("{p}.engram.wkv.weight"))?;
        self.tap(&format!("{p}.eg_kv"), &kv, t * (hc + 1) * d)?;
        let qk = {
            let cached = self.qk.borrow().get(&lid).cloned();
            match cached {
                Some(q) => q,
                None => {
                    let (qw, kw) = (
                        self.cpu.w(&format!("{p}.engram.q_weight"))?,
                        self.cpu.w(&format!("{p}.engram.k_weight"))?,
                    );
                    let v: Vec<f32> = qw.data.iter().zip(&kw.data).map(|(a, b)| a * b).collect();
                    let b = Rc::new(up(&v)?);
                    self.qk.borrow_mut().insert(lid, b.clone());
                    b
                }
            }
        };
        let out = fbuf(t * hc * d)?;
        let (hci, di, eps) = (hc as i32, d as i32, c.norm_eps);
        unsafe {
            self.k.go(
                "ds_engram_gate",
                ((t * hc) as u32, 1, 1),
                B256,
                0,
                &self.s,
                args![x.ptr, kv.ptr, qk.ptr, out.ptr, hci, di, eps],
            )?
        };
        self.s.sync()?;
        Ok(out)
    }

    // the forward

    fn embed_hc(&self, ids: &[usize], copies: usize) -> Result<DevBuf> {
        let d = self.cpu.cfg.dim;
        let e = self.dw("embed.weight")?;
        let t = ids.len();
        let di = up(&ids.iter().map(|&i| i as i32).collect::<Vec<_>>())?;
        let out = fbuf(t * copies * d)?;
        let (ti, ci, dd) = (t as i32, copies as i32, d as i32);
        unsafe {
            self.k.go(
                "ds_embed",
                blocks(t * copies * d, 256),
                B256,
                0,
                &self.s,
                args![e.v.w, di.ptr, out.ptr, ti, ci, dd],
            )?
        };
        self.s.sync()?;
        Ok(out)
    }

    fn one_hot(&self, t: usize) -> Result<DevBuf> {
        let hc = self.cpu.cfg.hc_mult;
        up(&(0..t * hc)
            .map(|i| (i % hc == 0) as u8 as f32)
            .collect::<Vec<f32>>())
    }

    /// As [`Model::forward`]: the last position's logits and the DSpark target hiddens.
    /// `hook` (when given) sees the engram hashes and each block's stream, read back.
    pub fn forward(
        &mut self,
        ids: &[usize],
        sp: usize,
        hook: Option<&mut dyn FnMut(&str, &[f32])>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward_impl(ids, sp, hook, false)
    }

    /// As [`Model::forward_all`]: every position's logits [T][V] (a verify window).
    pub fn forward_all(&mut self, ids: &[usize], sp: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward_impl(ids, sp, None, true)
    }

    fn forward_impl(
        &mut self,
        ids: &[usize],
        sp: usize,
        mut hook: Option<&mut dyn FnMut(&str, &[f32])>,
        all: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = self.cpu.cfg.clone();
        let (hc, d, t) = (c.hc_mult, c.dim, ids.len());
        let hashes = if c.engram_layer_ids.is_empty() {
            None
        } else {
            Some(self.cpu.engram_hash(ids, sp))
        };
        if let (Some(hs), Some(h)) = (&hashes, hook.as_mut()) {
            h(
                "hash",
                &hs.iter()
                    .flatten()
                    .flatten()
                    .map(|&v| v as f32)
                    .collect::<Vec<_>>(),
            );
        }
        let mut h = self.embed_hc(ids, hc)?;
        let mut pre_mix = self.one_hot(t)?;
        let mut mains: Vec<DevBuf> = Vec::new();
        for i in 0..c.n_layers {
            let p = format!("layers.{i}");
            if let Some(f) = &self.force {
                h = up(&f[i].0)?;
                pre_mix = up(&f[i].1)?;
            }
            if let Some(j) = c.engram_layer_ids.iter().position(|&l| l == i) {
                let hs: Vec<Vec<i64>> = hashes
                    .as_ref()
                    .context("hashes")?
                    .iter()
                    .map(|ht| ht[j].clone())
                    .collect();
                h = self.engram(&p, i, &h, &hs)?;
            }
            if c.dspark_target_layer_ids.contains(&i) {
                let m = fbuf(t * d)?;
                let (ti, hci, di) = (t as i32, hc as i32, d as i32);
                unsafe {
                    self.k.go(
                        "ds_mean_hc",
                        blocks(t * d, 256),
                        B256,
                        0,
                        &self.s,
                        args![h.ptr, m.ptr, ti, hci, di],
                    )?
                };
                mains.push(m);
            }
            let (nh, npm) = self.block(&p, i, h, t, sp, &pre_mix, Attn::Main)?;
            h = nh;
            pre_mix = npm;
            if let Some(hk) = hook.as_mut() {
                hk(&format!("layer{i}"), &self.read(&h, t * hc * d)?);
            }
        }
        let x = self.hc_pre(&h, &pre_mix, t)?;
        self.rms(&x, t, "norm.weight")?;
        let rows = if all { t } else { 1 };
        let logits = self.raw(x.ptr + ((t - rows) * d * 4) as u64, rows, "head.weight")?;
        let logits = self.read(&logits, rows * c.vocab_size)?;
        let n = mains.len();
        let mut main = vec![0f32; t * d * n];
        for (j, m) in mains.iter().enumerate() {
            let v = self.read(m, t * d)?;
            for ti in 0..t {
                main[(ti * n + j) * d..(ti * n + j + 1) * d]
                    .copy_from_slice(&v[ti * d..(ti + 1) * d]);
            }
        }
        Ok((logits, main))
    }

    // dspark

    #[allow(clippy::too_many_arguments)]
    fn dspark_attention(
        &mut self,
        p: &str,
        lid: usize,
        x: &DevBuf,
        b: usize,
        sp: usize,
        main_x: &DevBuf,
        t: usize,
    ) -> Result<DevBuf> {
        let c = self.cpu.cfg.clone();
        let (h, hd, win) = (c.n_heads, c.head_dim, c.window_size);
        let mkv = self.lin(main_x, t, &format!("{p}.attn.wkv.weight"))?;
        self.rms(&mkv, t, &format!("{p}.attn.kv_norm.weight"))?;
        self.rope(&mkv, t, 1, hd, lid, sp, 1, None, false)?;
        self.k.fp8_qdq(&self.s, mkv.ptr, t * hd)?;
        if sp == 0 {
            self.ring_seed(lid, &mkv, t)?;
            self.s.sync()?;
            return fbuf(1);
        }
        let pos0 = sp + t;
        let qr = self.lin(x, b, &format!("{p}.attn.wq_a.weight"))?;
        self.rms(&qr, b, &format!("{p}.attn.q_norm.weight"))?;
        let q = self.lin(&qr, b, &format!("{p}.attn.wq_b.weight"))?;
        self.rope(&q, b, h, hd, lid, pos0, 1, None, false)?;
        let kv = self.lin(x, b, &format!("{p}.attn.wkv.weight"))?;
        self.rms(&kv, b, &format!("{p}.attn.kv_norm.weight"))?;
        self.rope(&kv, b, 1, hd, lid, pos0, 1, None, false)?;
        self.k.fp8_qdq(&self.s, kv.ptr, b * hd)?;
        let row: Vec<i32> = (0..win.min(sp + t) as i32)
            .chain((0..b as i32).map(|i| win as i32 + i))
            .collect();
        let idxs = vec![row; b];
        let ring = self.win[lid].ptr;
        for i in 0..t {
            let s = ((sp + i) % win * hd * 4) as u64;
            self.copy(ring + s, mkv.ptr + (i * hd * 4) as u64, hd)?;
        }
        let all = fbuf((win + b) * hd)?;
        self.copy(all.ptr, ring, win * hd)?;
        self.copy(all.ptr + (win * hd * 4) as u64, kv.ptr, b * hd)?;
        self.attn_out(p, &q, &all, &idxs, lid, pos0)
    }

    /// As [`Model::forward_spec`].
    pub fn forward_spec(
        &mut self,
        token: usize,
        main_hidden: &[f32],
        sp: usize,
    ) -> Result<Option<super::forward::Draft>> {
        let c = self.cpu.cfg.clone();
        let (hc, d, blk, n_mtp, v) = (
            c.hc_mult,
            c.dim,
            c.dspark_block_size,
            c.n_mtp_layers,
            c.vocab_size,
        );
        let t = main_hidden.len() / (d * c.dspark_target_layer_ids.len());
        let mh = up(main_hidden)?;
        let main_x = self.lin(&mh, t, "mtp.0.main_proj.weight")?;
        self.rms(&main_x, t, "mtp.0.main_norm.weight")?;
        let mut ids = vec![c.dspark_noise_token_id; blk];
        ids[0] = token;
        let mut h = self.embed_hc(&ids, hc)?;
        let mut pre_mix = self.one_hot(blk)?;
        for s in 0..n_mtp {
            let (lid, p) = (c.n_layers + s, format!("mtp.{s}"));
            if sp == 0 {
                self.dspark_attention(&p, lid, &h, blk, 0, &main_x, t)?;
                continue;
            }
            let (nh, npm) = self.block(&p, lid, h, blk, sp, &pre_mix, Attn::Spark(&main_x, t))?;
            h = nh;
            pre_mix = npm;
        }
        if sp == 0 {
            return Ok(None);
        }
        let last = format!("mtp.{}", n_mtp - 1);
        let x = self.hc_pre(&h, &pre_mix, blk)?;
        let xh = self.read(&x, blk * d)?;
        self.rms(&x, blk, &format!("{last}.norm.weight"))?;
        let logits = self.raw(x.ptr, blk, "head.weight")?;
        let mut logits = self.read(&logits, blk * v)?;
        // host: the Markov head samples left to right, then the confidence head
        let rank = c.dspark_markov_rank;
        let memb_w = self.cpu.w(&format!("{last}.markov_head.embed.weight"))?;
        let mhead = self.cpu.w(&format!("{last}.markov_head.head.weight"))?;
        let proj = self.cpu.w(&format!("{last}.confidence_head.proj.weight"))?;
        let mut out = vec![token];
        let mut conf = Vec::with_capacity(blk);
        for i in 0..blk {
            let memb = &memb_w.data[out[i] * rank..(out[i] + 1) * rank];
            let bias = super::forward::matmul(memb, 1, &mhead);
            let li = &mut logits[i * v..(i + 1) * v];
            for (a, b) in li.iter_mut().zip(&bias) {
                *a += b;
            }
            let best = (0..v).fold(0, |b, j| if li[j] > li[b] { j } else { b });
            out.push(best);
            let feat: Vec<f32> = xh[i * d..(i + 1) * d].iter().chain(memb).copied().collect();
            conf.push(super::forward::matmul(&feat, 1, &proj)[0]);
        }
        Ok(Some(super::forward::Draft {
            ids: out,
            logits,
            conf,
        }))
    }

    // speculative decoding

    fn dup(&self, b: &DevBuf, n: usize) -> Result<DevBuf> {
        let o = fbuf(n)?;
        self.copy(o.ptr, b.ptr, n)?;
        Ok(o)
    }

    /// As [`Model::snapshot`].
    pub fn snapshot(&mut self) -> Result<GpuSnapshot> {
        let c = self.cpu.cfg.clone();
        let hd = c.head_dim;
        let win = (0..c.n_layers)
            .map(|l| self.dup(&self.win[l], c.window_size * hd))
            .collect::<Result<_>>()?;
        let mut cst = HashMap::new();
        for (&l, (a, b)) in &self.cst {
            let n = c.compress_ratios[l] * hd;
            cst.insert(l, (self.dup(a, n)?, self.dup(b, n)?));
        }
        self.s.sync()?;
        self.journal = Some(Vec::new());
        Ok(GpuSnapshot { win, cst })
    }

    /// As [`Model::rewind`]: back to only positions `< from` fed.
    pub fn rewind(&mut self, snap: &GpuSnapshot, from: usize, end: usize) -> Result<()> {
        let c = self.cpu.cfg.clone();
        let (win, hd) = (c.window_size, c.head_dim);
        for (l, old) in snap.win.iter().enumerate() {
            for pos in from..end.min(from + win) {
                let o = ((pos % win) * hd * 4) as u64;
                self.copy(self.win[l].ptr + o, old.ptr + o, hd)?;
            }
        }
        for (l, (a, b)) in &snap.cst {
            let n = c.compress_ratios[*l] * hd;
            let (ka, kb) = (self.cst[l].0.ptr, self.cst[l].1.ptr);
            self.copy(ka, a.ptr, n)?;
            self.copy(kb, b.ptr, n)?;
        }
        let journal = self.journal.take().unwrap_or_default();
        for (lid, pos, kv, sc) in &journal {
            if *pos < from {
                let o = ((pos % c.compress_ratios[*lid]) * hd * 4) as u64;
                let (ka, kb) = (self.cst[lid].0.ptr, self.cst[lid].1.ptr);
                self.copy(ka + o, kv.ptr, hd)?;
                self.copy(kb + o, sc.ptr, hd)?;
            }
        }
        self.s.sync()?;
        drop(journal);
        Ok(())
    }

    /// As [`Model::decode_spec`].
    pub fn decode_spec(
        &mut self,
        token: usize,
        main: &[f32],
        sp: usize,
        edit: &mut dyn FnMut(usize, &mut [usize]),
    ) -> Result<SpecStep> {
        let c = self.cpu.cfg.clone();
        let (v, w) = (c.vocab_size, c.dim * c.dspark_target_layer_ids.len());
        let pos0 = sp + main.len() / w;
        let draft = self
            .forward_spec(token, main, sp)?
            .context("decode_spec needs sp > 0")?;
        let mut ids = draft.ids;
        edit(pos0 + 1, &mut ids[1..]);
        let snap = self.snapshot()?;
        let (logits, mains) = self.forward_all(&ids, pos0)?;
        let n = ids.len();
        let mut k = 0;
        while k + 1 < n && argmax(&logits[k * v..(k + 1) * v]) == ids[k + 1] {
            k += 1;
        }
        let next = argmax(&logits[k * v..(k + 1) * v]);
        self.rewind(&snap, pos0 + k + 1, pos0 + n)?;
        let mut tokens = ids[1..=k].to_vec();
        tokens.push(next);
        Ok(SpecStep {
            tokens,
            drafted: n - 1,
            accepted: k,
            main: mains[..(k + 1) * w].to_vec(),
            sp: pos0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::tests::tiny;
    use super::*;

    fn load_ref(name: &str) -> Option<Vec<f32>> {
        let b = std::fs::read(format!("{}/ref/{name}.bin", tiny())).ok()?;
        Some(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }

    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let m = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-12);
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / m
    }

    fn kl(p_logits: &[f32], q_logits: &[f32]) -> f64 {
        let ls = |l: &[f32]| {
            let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
            let z = l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
            l.iter().map(|&x| x as f64 - z).collect::<Vec<f64>>()
        };
        let (p, q) = (ls(p_logits), ls(q_logits));
        p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum()
    }

    fn argmax(v: &[f32]) -> usize {
        (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
    }

    fn open() -> Option<(GpuModel, serde_json::Value)> {
        Gpu::for_test()?;
        let dir = Path::new(&tiny()).join("hf");
        if !Path::new(&tiny()).join("ref/step0_in0.bin").exists() {
            eprintln!("skip: no {} (run scripts/deepseek_tiny.py)", tiny());
            return None;
        }
        let meta = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/ref/meta.json", tiny())).unwrap(),
        )
        .unwrap();
        Some((GpuModel::open(&dir).unwrap(), meta))
    }

    /// Step 0 intermediates, device against the CPU reference on the same forced input.
    #[test]
    #[ignore]
    fn gpu_tiny_taps() {
        let Some((mut m, meta)) = open() else { return };
        std::env::set_var("DS_TAP", "1");
        let n_layers = m.cpu.cfg.n_layers;
        let ids: Vec<usize> = meta["steps"][0]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let force: Vec<(Vec<f32>, Vec<f32>)> = (0..n_layers)
            .map(|i| {
                (
                    load_ref(&format!("step0_in{i}")).unwrap(),
                    load_ref(&format!("step0_pm{i}")).unwrap(),
                )
            })
            .collect();
        let mut cpu = Model::open(&Path::new(&tiny()).join("hf")).unwrap();
        cpu.force = Some(force.clone());
        cpu.forward(&ids, 0, &mut |_, _| {}).unwrap();
        m.force = Some(force);
        m.forward(&ids, 0, None).unwrap();
        let want: HashMap<String, Vec<f32>> = cpu.taps.borrow().iter().cloned().collect();
        for (name, v) in m.taps.borrow().iter() {
            if let Some(w) = want.get(name) {
                eprintln!("{name}: rel {:.2e}", rel(v, w));
            }
        }
    }

    /// The device forward teacher-forced against the fixture, as the CPU reference's test.
    #[test]
    fn gpu_tiny_teacher_forced() {
        let Some((mut m, meta)) = open() else { return };
        let n_layers = m.cpu.cfg.n_layers;
        for (s, step) in meta["steps"].as_array().unwrap().iter().enumerate() {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            m.force = Some(
                (0..n_layers)
                    .map(|i| {
                        (
                            load_ref(&format!("step{s}_in{i}")).unwrap(),
                            load_ref(&format!("step{s}_pm{i}")).unwrap(),
                        )
                    })
                    .collect(),
            );
            let mut got: HashMap<String, Vec<f32>> = HashMap::new();
            let mut hook = |n: &str, t: &[f32]| {
                got.insert(n.to_string(), t.to_vec());
            };
            let (logits, _) = m.forward(&ids, sp, Some(&mut hook)).unwrap();
            let rels: Vec<f32> = (0..n_layers)
                .map(|i| {
                    rel(
                        &got[&format!("layer{i}")],
                        &load_ref(&format!("step{s}_layer{i}")).unwrap(),
                    )
                })
                .collect();
            let k = kl(&load_ref(&format!("step{s}_logits")).unwrap(), &logits);
            let main = load_ref(&format!("step{s}_main_hidden")).unwrap();
            let next = step["next_id"].as_u64().unwrap() as usize;
            let draft = m.forward_spec(next, &main, sp).unwrap();
            let spec = match (draft, step["spec_ids"].as_array()) {
                (Some(d), Some(want)) => {
                    let want: Vec<usize> =
                        want.iter().map(|v| v.as_u64().unwrap() as usize).collect();
                    let r = load_ref(&format!("step{s}_spec_logits")).unwrap();
                    format!(
                        "spec {} kl0 {:.0e}",
                        if d.ids == want {
                            "same".into()
                        } else {
                            format!("{:?} want {want:?}", d.ids)
                        },
                        kl(
                            &r[..m.cpu.cfg.vocab_size],
                            &d.logits[..m.cpu.cfg.vocab_size]
                        )
                    )
                }
                _ => String::new(),
            };
            eprintln!(
                "step {s}: [{}] KL {k:.1e} top1 {} {spec}",
                rels.iter()
                    .map(|r| format!("{r:.1e}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                argmax(&logits) == next
            );
            assert!(k < 1e-3, "step {s}: KL {k}");
            for (i, r) in rels.iter().enumerate() {
                assert!(*r < 5e-2, "step {s} layer {i}: rel {r}");
            }
        }
    }

    /// The expert LRU changes where weights live, never the math: a 3-slot cache gives the
    /// unbounded run's logits bit for bit.
    #[test]
    fn gpu_tiny_expert_cache_is_exact() {
        let Some((mut a, meta)) = open() else { return };
        let mut b = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
        b.expert_slots = 3;
        for step in meta["steps"].as_array().unwrap().iter().take(4) {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            let (la, _) = a.forward(&ids, sp, None).unwrap();
            let (lb, _) = b.forward(&ids, sp, None).unwrap();
            assert_eq!(la, lb, "step at {sp}");
        }
        eprintln!(
            "misses: unbounded {} vs 3 slots {}",
            a.expert_misses(),
            b.expert_misses()
        );
        assert!(b.expert_misses() > a.expert_misses());
    }

    /// Free-running on the device, prefill then greedy decode; informational past top-1.
    #[test]
    fn gpu_tiny_free_running() {
        let Some((mut m, meta)) = open() else { return };
        let steps = meta["steps"].as_array().unwrap();
        let mut top1 = 0;
        for (s, step) in steps.iter().enumerate() {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            let (logits, main) = m.forward(&ids, sp, None).unwrap();
            let next = step["next_id"].as_u64().unwrap() as usize;
            top1 += (argmax(&logits) == next) as usize;
            m.forward_spec(argmax(&logits), &main, sp).unwrap();
            eprintln!(
                "step {s}: KL {:.2e} top1 {}",
                kl(&load_ref(&format!("step{s}_logits")).unwrap(), &logits),
                argmax(&logits) == next
            );
        }
        assert!(top1 + 1 >= steps.len(), "top-1 {top1}/{}", steps.len());
    }

    fn prompt(meta: &serde_json::Value) -> Vec<usize> {
        meta["steps"][0]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    }

    /// As forward.rs `tiny_chunk_and_rewind_exact`, on the device.
    #[test]
    fn gpu_tiny_chunk_and_rewind_exact() {
        let Some((mut a, meta)) = open() else { return };
        let ids = prompt(&meta);
        let (v, n) = (a.cpu.cfg.vocab_size, ids.len());
        a.forward(&ids, 0, None).unwrap();
        let stream: Vec<usize> = (0..14).map(|i| (37 * i + 11) % v).collect();
        let mut want = Vec::new();
        for (i, &t) in stream.iter().enumerate() {
            want.push(a.forward(&[t], n + i, None).unwrap().0);
        }
        drop(a);
        for keep in 1..=6 {
            let mut b = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
            b.forward(&ids, 0, None).unwrap();
            b.forward(&stream[..1], n, None).unwrap();
            let snap = b.snapshot().unwrap();
            let (lw, _) = b.forward_all(&stream[1..7], n + 1).unwrap();
            for e in 0..6 {
                assert_eq!(lw[e * v..(e + 1) * v], want[1 + e][..], "window row {e}");
            }
            b.rewind(&snap, n + 1 + keep, n + 7).unwrap();
            for i in 1 + keep..stream.len() {
                let (l, _) = b.forward(&[stream[i]], n + i, None).unwrap();
                assert_eq!(l, want[i], "keep {keep} step {i}");
            }
        }
    }

    /// As forward.rs `tiny_spec_matches_greedy`, on the device.
    #[test]
    fn gpu_tiny_spec_matches_greedy() {
        let Some((mut m, meta)) = open() else { return };
        let ids = prompt(&meta);
        let (v, n, len) = (m.cpu.cfg.vocab_size, ids.len(), 40);
        let (mut l, _) = m.forward(&ids, 0, None).unwrap();
        let mut want = Vec::new();
        while want.len() < len {
            want.push(argmax(&l));
            l = m
                .forward(&[*want.last().unwrap()], n + want.len() - 1, None)
                .unwrap()
                .0;
        }
        drop(m);
        let (oracle, oracle2) = (want.clone(), want.clone());
        let mut rng = 1u64;
        type Edit = Box<dyn FnMut(usize, &mut [usize])>;
        let runs: Vec<(&str, Edit)> = vec![
            ("dspark", Box::new(|_, _: &mut [usize]| {})),
            (
                "oracle",
                Box::new(move |pos, d: &mut [usize]| {
                    for (i, x) in d.iter_mut().enumerate() {
                        if let Some(&w) = oracle.get(pos + i - n) {
                            *x = w;
                        }
                    }
                }),
            ),
            (
                "wrong",
                Box::new(move |pos, d: &mut [usize]| {
                    for (i, x) in d.iter_mut().enumerate() {
                        if let Some(&w) = oracle2.get(pos + i - n) {
                            *x = w;
                        }
                        rng = rng
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        if (rng >> 33) % 5 == 0 {
                            *x = (*x + 1 + (rng >> 40) as usize % (v - 1)) % v;
                        }
                    }
                }),
            ),
        ];
        for (name, mut edit) in runs {
            let mut m = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
            let (l, main) = m.forward(&ids, 0, None).unwrap();
            let t0 = argmax(&l);
            m.forward_spec(t0, &main, 0).unwrap();
            let (l, mut main) = m.forward(&[t0], n, None).unwrap();
            let mut out = vec![t0, argmax(&l)];
            let (mut sp, mut drafted, mut accepted) = (n, 0, 0);
            while out.len() < len {
                let s = m
                    .decode_spec(*out.last().unwrap(), &main, sp, &mut *edit)
                    .unwrap();
                drafted += s.drafted;
                accepted += s.accepted;
                out.extend(s.tokens);
                main = s.main;
                sp = s.sp;
            }
            out.truncate(len);
            eprintln!(
                "{name}: accepted {accepted}/{drafted} ({:.1}%)",
                100.0 * accepted as f64 / drafted.max(1) as f64
            );
            assert_eq!(out, want, "{name}");
        }
    }
}
