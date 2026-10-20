//! Streams that carry their own decoder (`TCS1`).
//!
//! A `TCS1` container is `"TCS1"`, a varint length, a decoder program, then
//! an ordinary TCG3 stream. The program replaces the receiver's built-in
//! coding path: the rANS residual decoders ([`crate::irdec`]) and optionally
//! the learned model's MLP (its weights as graph constants, so the receiver
//! needs no model registry). The receiver verifies it
//! ([`tang_expr::int::IntGraph::from_bytes`] plus the role ABIs below),
//! lowers it to WGSL (bounds-checked loads) and to the CPU interpreter, and
//! caches the result by content hash. Plain TCG3 streams decode as before.
//!
//! Program layout: `"TCDP"`, version byte (1), flags byte (bit 0 static
//! residual graph, bit 1 learned residual graph, bit 2 MLP graph), then per
//! present section a varint length and a `TIR1` graph. A missing residual
//! section falls back to the built-in graph.
//!
//! Role ABIs (anything else is rejected):
//! - static residual: params `u32 × 8`, buffers `mt`, `stream` (`u32`),
//!   outputs `u32, u32, i32` (see [`crate::irdec`]);
//! - learned residual: params `u32 × 3, i32 × 5, u32 × 2`, same buffers and
//!   outputs;
//! - MLP: params `i32 × NFL`, no buffers, outputs `i32 × NO`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::bitio::DecodeError;
use crate::irdec::{BUFS, LEARNED_PARAMS, OUTS, STATIC_PARAMS};
use crate::learned::NO;
use crate::lmodel3::{Net32, NFL};
use tang_expr::int::{IId, IType, IntGraph, Limits, Program};

/// Container magic.
pub const MAGIC: &[u8; 4] = b"TCS1";
const PMAGIC: &[u8; 4] = b"TCDP";
/// Largest program accepted (bytes).
pub const MAX_PROGRAM: usize = 1 << 20;

/// A graph with its outputs.
pub type Graph = (IntGraph, Vec<IId>);

/// A verified decoder program, ready for both backends.
pub struct Decoder {
    /// Static residual graph.
    pub stat: Graph,
    /// Learned residual graph.
    pub learned: Graph,
    /// Shipped MLP (replaces the stream's model).
    pub mlp: Option<Graph>,
    /// WGSL for the kernel's RES region (bounds-checked loads).
    pub wgsl_res: String,
    /// WGSL `fn mlp_gen()` when the MLP is shipped.
    pub wgsl_mlp: Option<String>,
    /// Content hash of the program bytes.
    pub key: u64,
    /// Program size in bytes.
    pub size: usize,
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Decoder {{ key: {:016x}, size: {}, mlp: {} }}", self.key, self.size, self.mlp.is_some())
    }
}

impl Decoder {
    /// CPU interpreter for the MLP, if shipped.
    pub fn mlp_program(&self) -> Option<Program> {
        self.mlp.as_ref().map(|(g, o)| Program::new(g, o))
    }
}

/// FNV-1a 64.
pub fn fnv1a(b: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn put(o: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            o.push(b);
            return;
        }
        o.push(b | 0x80);
    }
}

fn get(b: &[u8], p: &mut usize) -> Result<u64, DecodeError> {
    let mut v = 0u64;
    for s in 0..10 {
        let x = *b.get(*p).ok_or(DecodeError::Truncated)?;
        *p += 1;
        v |= ((x & 0x7f) as u64) << (7 * s);
        if x & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err(DecodeError::Invalid("varint"))
}

/// Serialize a decoder program from graphs (`None` sections are left to the
/// receiver's built-ins).
pub fn program_bytes(stat: Option<&Graph>, learned: Option<&Graph>, mlp: Option<&Graph>) -> Vec<u8> {
    let mut o = PMAGIC.to_vec();
    o.push(1);
    let secs = [stat, learned, mlp];
    o.push(secs.iter().enumerate().map(|(i, s)| (s.is_some() as u8) << i).sum());
    for (g, outs) in secs.into_iter().flatten() {
        let b = g.to_bytes(outs);
        put(&mut o, b.len() as u64);
        o.extend(b);
    }
    o
}

/// The built-in decoder as a program: both residual graphs, plus `net`'s
/// MLP when given.
pub fn builtin_program(net: Option<&Net32>) -> Vec<u8> {
    let mlp = net.map(|n| n.graph());
    program_bytes(Some(&crate::irdec::static_graph()), Some(&crate::irdec::learned_graph()), mlp.as_ref())
}

/// Wrap a TCG3 stream with a decoder program.
pub fn wrap(stream: &[u8], program: &[u8]) -> Vec<u8> {
    let mut o = MAGIC.to_vec();
    put(&mut o, program.len() as u64);
    o.extend_from_slice(program);
    // the stream starts on a 16-bit word boundary (lane offsets are words)
    if o.len() % 2 == 1 {
        o.push(0);
    }
    o.extend_from_slice(stream);
    o
}

fn res_limits() -> Limits {
    Limits { max_nodes: 1 << 14, max_loops: 16, max_depth: 2, max_trip: 64, max_cost: 1 << 16 }
}

fn mlp_limits() -> Limits {
    Limits { max_nodes: 1 << 16, max_loops: 0, max_depth: 0, max_trip: 0, max_cost: 1 << 16 }
}

/// Verify and lower a program (no cache).
pub fn compile(prog: &[u8]) -> Result<Decoder, DecodeError> {
    let bad = |_| DecodeError::Invalid("decoder program rejected by the verifier");
    if prog.len() > MAX_PROGRAM {
        return Err(DecodeError::Invalid("decoder program too large"));
    }
    if prog.len() < 6 || &prog[..4] != PMAGIC || prog[4] != 1 {
        return Err(DecodeError::Invalid("decoder program header"));
    }
    let flags = prog[5];
    if flags & !7 != 0 {
        return Err(DecodeError::Invalid("decoder program flags"));
    }
    let mut p = 6;
    let sec = |p: &mut usize| -> Result<&[u8], DecodeError> {
        let n = get(prog, p)? as usize;
        let s = prog.get(*p..p.checked_add(n).ok_or(DecodeError::Truncated)?).ok_or(DecodeError::Truncated)?;
        *p += n;
        Ok(s)
    };
    let stat = match flags & 1 {
        0 => crate::irdec::static_graph(),
        _ => IntGraph::from_bytes(sec(&mut p)?, &STATIC_PARAMS, &BUFS, &OUTS, &res_limits()).map_err(bad)?,
    };
    let learned = match flags & 2 {
        0 => crate::irdec::learned_graph(),
        _ => IntGraph::from_bytes(sec(&mut p)?, &LEARNED_PARAMS, &BUFS, &OUTS, &res_limits()).map_err(bad)?,
    };
    let mlp = match flags & 4 {
        0 => None,
        _ => Some(IntGraph::from_bytes(sec(&mut p)?, &[IType::I32; NFL], &[], &[IType::I32; NO], &mlp_limits()).map_err(bad)?),
    };
    if p != prog.len() {
        return Err(DecodeError::Invalid("decoder program trailing bytes"));
    }
    let wgsl_res = crate::irdec::wgsl((&stat.0, &stat.1), (&learned.0, &learned.1), true);
    let wgsl_mlp = mlp.as_ref().map(|(g, o)| {
        let params: Vec<String> = (0..NFL).map(|i| format!("na[{i}]")).collect();
        let dests: Vec<String> = (0..o.len()).map(|k| format!("na[{k}]")).collect();
        let pr: Vec<&str> = params.iter().map(|s| s.as_str()).collect();
        let dr: Vec<&str> = dests.iter().map(|s| s.as_str()).collect();
        format!("fn mlp_gen() {{\n{}}}\n", g.to_wgsl_body_guarded(o, &pr, &dr, 1))
    });
    Ok(Decoder { stat, learned, mlp, wgsl_res, wgsl_mlp, key: fnv1a(prog), size: prog.len() })
}

type Cache = Mutex<HashMap<u64, Vec<(Vec<u8>, Arc<Decoder>)>>>;

fn cache() -> &'static Cache {
    static C: OnceLock<Cache> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// [`compile`] through the process-wide cache (keyed by FNV-1a 64 of the
/// program, confirmed by a full byte comparison, so a hash collision can
/// never select another program).
pub fn compile_cached(prog: &[u8]) -> Result<Arc<Decoder>, DecodeError> {
    let k = fnv1a(prog);
    if let Some(v) = cache().lock().unwrap().get(&k) {
        if let Some((_, d)) = v.iter().find(|(b, _)| b == prog) {
            return Ok(d.clone());
        }
    }
    let d = Arc::new(compile(prog)?);
    let mut c = cache().lock().unwrap();
    let v = c.entry(k).or_default();
    if v.len() < 16 {
        v.push((prog.to_vec(), d.clone()));
    }
    Ok(d)
}

/// Split a stream: `(Some(decoder), inner TCG3)` for `TCS1`, `(None, bytes)`
/// otherwise.
pub fn open(bytes: &[u8]) -> Result<(Option<Arc<Decoder>>, &[u8]), DecodeError> {
    if bytes.len() < 4 || &bytes[..4] != MAGIC {
        return Ok((None, bytes));
    }
    let mut p = 4;
    let n = get(bytes, &mut p)? as usize;
    if n > MAX_PROGRAM {
        return Err(DecodeError::Invalid("decoder program too large"));
    }
    let prog = bytes.get(p..p + n).ok_or(DecodeError::Truncated)?;
    let start = (p + n).next_multiple_of(2);
    let inner = bytes.get(start..).ok_or(DecodeError::Truncated)?;
    Ok((Some(compile_cached(prog)?), inner))
}

/// Placeholder net for a stream whose model is shipped as a graph (its
/// weights are never used; the shipped MLP runs instead).
pub fn placeholder_net() -> Net32 {
    Net32 { sizes: vec![NFL, NO], w: vec![vec![0; NFL * NO]], b: vec![vec![0; NO]] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpuwav3::{decode3_box, encode3_f32, parse3, Gpu3Config, Model3};
    use crate::ErrorBound;

    fn field(d: [usize; 3]) -> Vec<f32> {
        let mut v = Vec::with_capacity(d.iter().product());
        for z in 0..d[0] {
            for y in 0..d[1] {
                for x in 0..d[2] {
                    let (fz, fy, fx) = (z as f32 * 0.21, y as f32 * 0.13, x as f32 * 0.17);
                    v.push((fz.sin() * fy.cos() + (fx * 1.3).sin() * 0.5 + (fx * fy * 0.05).cos() * 0.25) * 3.0);
                }
            }
        }
        v
    }

    fn streams() -> Vec<(String, Vec<u8>)> {
        let d = [20, 37, 45];
        let x = field(d);
        let mut out = Vec::new();
        for (lanes, bound) in [(1, ErrorBound::Psnr(60.0)), (8, ErrorBound::Psnr(100.0)), (8, ErrorBound::MaxAbs(1e-4))] {
            let cfg = Gpu3Config { block: 16, lanes, tile: 16, ..Gpu3Config::new(bound) };
            out.push((format!("static lanes {lanes} {bound:?}"), encode3_f32(&x, &d, &cfg, 2).0));
            let model = Model3::Overfit { hidden: vec![8], per_epoch: 20_000, epochs: 2 };
            let cfg = Gpu3Config { block: 16, lanes, tile: 16, model, ..Gpu3Config::new(bound) };
            out.push((format!("learned lanes {lanes} {bound:?}"), encode3_f32(&x, &d, &cfg, 2).0));
        }
        out
    }

    fn shipped(s: &[u8], with_mlp: bool) -> Vec<u8> {
        let ps = parse3(s).unwrap();
        let net = if with_mlp { ps.net.as_ref() } else { None };
        wrap(s, &builtin_program(net))
    }

    #[test]
    fn shipped_decoder_matches_builtin_on_cpu() {
        for (name, s) in streams() {
            let want = decode3_box(&s, [0; 3], [20, 37, 45], 2).unwrap().bits;
            for mlp in [false, true] {
                let w = shipped(&s, mlp);
                let ps = parse3(&w).unwrap();
                assert!(ps.dec.is_some());
                let got = decode3_box(&w, [0; 3], [20, 37, 45], 2).unwrap().bits;
                assert!(got == want, "{name} mlp {mlp}");
                // random access too
                let sub = decode3_box(&w, [3, 5, 7], [11, 30, 20], 1).unwrap();
                let base = decode3_box(&s, [3, 5, 7], [11, 30, 20], 1).unwrap();
                assert!(sub.bits == base.bits, "{name} box");
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn shipped_decoder_matches_builtin_on_gpu() {
        let Ok(g) = crate::gpudec::GpuDecoder::new(8, 128) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        for (name, s) in streams() {
            let want = decode3_box(&s, [0; 3], [20, 37, 45], 1).unwrap().bits;
            assert!(g.decode3(&s).unwrap().0 == want, "{name} builtin gpu");
            for mlp in [false, true] {
                let w = shipped(&s, mlp);
                assert!(g.decode3(&w).unwrap().0 == want, "{name} shipped gpu mlp {mlp}");
            }
        }
    }

    /// A pretrained-model stream (only the fingerprint inside) decodes with no
    /// registered model once the MLP ships as a graph.
    #[test]
    fn shipped_mlp_needs_no_registry() {
        let path = std::env::var("TC_TEST_TCM").unwrap_or_else(|_| "/Users/cam/Developer/tang-compress/bench/models/pre3v4_16.tcm".into());
        let Ok(tcm) = std::fs::read(&path) else {
            eprintln!("no {path}; skipping");
            return;
        };
        let net = Arc::new(crate::learned::Net::from_bytes(&tcm).unwrap());
        let d = [20, 37, 45];
        let cfg = Gpu3Config { block: 16, lanes: 8, tile: 16, model: Model3::Pretrained(net.clone()), ..Gpu3Config::new(ErrorBound::Psnr(80.0)) };
        let (s, _) = encode3_f32(&field(d), &d, &cfg, 2);
        assert!(parse3(&s).is_err(), "model must not be registered yet");
        let prog = builtin_program(Some(&Net32::new(&net)));
        let w = wrap(&s, &prog);
        let got = decode3_box(&w, [0; 3], d, 2).unwrap().bits;
        #[cfg(feature = "gpu")]
        let gpu = crate::gpudec::GpuDecoder::new(8, 64).ok().map(|g| g.decode3(&w).unwrap().0);
        crate::learned::register(net);
        let want = decode3_box(&s, [0; 3], d, 2).unwrap().bits;
        assert!(got == want);
        #[cfg(feature = "gpu")]
        if let Some(gv) = gpu {
            assert!(gv == want, "gpu, shipped MLP");
        }
    }

    #[test]
    fn corrupt_programs_are_rejected_not_run() {
        let s = &streams()[1].1;
        let prog = builtin_program(parse3(s).unwrap().net.as_ref());
        let mut seed = 5u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        let (mut ok, mut bad) = (0, 0);
        for _ in 0..300 {
            let mut p = prog.clone();
            let i = 6 + rnd() % (p.len() - 6);
            p[i] = rnd() as u8;
            match compile(&p) {
                // a mutated but well-formed program is total: decoding returns (any bits or an error)
                Ok(_) => {
                    ok += 1;
                    let _ = decode3_box(&wrap(s, &p), [0; 3], [20, 37, 45], 1);
                }
                Err(_) => bad += 1,
            }
        }
        assert!(bad > 0);
        eprintln!("mutations: {ok} accepted, {bad} rejected");
        // truncations
        for n in [0, 3, 6, 7, prog.len() / 2, prog.len() - 1] {
            assert!(compile(&prog[..n]).is_err());
        }
        // an uncapped loop cannot be expressed on the wire; a huge cap is refused
        let spin = |cap: u32| {
            let mut g = IntGraph::new();
            for (n, e) in BUFS {
                g.buffer(n, e);
            }
            let x = g.param(0, IType::U32);
            let out = g.while_loop(&[x], |g, s| {
                let t = g.bool(true);
                let one = g.u32(1);
                (t, vec![g.add(s[0], one)])
            });
            g.cap_loop(out[0], cap);
            let z = g.i32(0);
            program_bytes(Some(&(g, vec![out[0], x, z])), None, None)
        };
        assert!(compile(&spin(1 << 20)).is_err());
        // within the caps an infinite loop just stops (and garbage decodes to an error or garbage)
        let d = compile(&spin(64)).unwrap();
        assert!(d.wgsl_res.contains("if c0 >= 64u"));
        let _ = decode3_box(&wrap(s, &spin(64)), [0; 3], [20, 37, 45], 1);
    }
}
