use super::*;

fn rng(seed: &mut u64) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed >> 16) as u32
}

/// Interesting 32-bit values plus random ones.
fn samples(n: usize) -> Vec<u32> {
    let mut v = vec![
        0,
        1,
        2,
        31,
        32,
        33,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
        0xffff,
        0x1_0000,
        (-2047i32) as u32,
        2047,
    ];
    let mut s = 0x9e37_79b9_7f4a_7c15;
    while v.len() < n {
        let x = rng(&mut s);
        v.push(if x & 1 == 0 { x } else { x >> (x % 31) });
    }
    v
}

const BINS: [BinOp; 11] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::MulHi,
    BinOp::And,
    BinOp::Or,
    BinOp::Xor,
    BinOp::Shl,
    BinOp::Shr,
    BinOp::Min,
    BinOp::Max,
];

/// One graph with every op on both integer types; outputs in a fixed order.
fn all_ops() -> (IntGraph, Vec<IId>) {
    let mut g = IntGraph::new();
    let a = g.param(0, IType::I32);
    let b = g.param(1, IType::I32);
    let ua = g.bitcast(IType::U32, a);
    let ub = g.bitcast(IType::U32, b);
    let mut outs = Vec::new();
    for (x, y) in [(a, b), (ua, ub)] {
        for op in BINS {
            let r = g.bin(op, x, y);
            outs.push(g.bitcast(IType::U32, r));
        }
        for op in [CmpOp::Eq, CmpOp::Ne, CmpOp::Lt, CmpOp::Le] {
            let c = g.cmp(op, x, y);
            let one = g.u32(1);
            let zero = g.u32(0);
            outs.push(g.select(c, one, zero));
        }
        let n = g.neg(x);
        outs.push(g.bitcast(IType::U32, n));
        let n = g.not(x);
        outs.push(g.bitcast(IType::U32, n));
        let n = g.clz(x);
        outs.push(g.bitcast(IType::U32, n));
        let lo = g.i32(-16384);
        let hi = g.i32(16384);
        let (lo, hi) = if g.ty(x) == IType::I32 {
            (lo, hi)
        } else {
            (g.u32(3), g.u32(1 << 20))
        };
        let c = g.clamp(x, lo, hi);
        outs.push(g.bitcast(IType::U32, c));
    }
    (g, outs)
}

fn reference(a: u32, b: u32) -> Vec<u32> {
    let (sa, sb) = (a as i32, b as i32);
    let mut v = vec![
        sa.wrapping_add(sb) as u32,
        sa.wrapping_sub(sb) as u32,
        sa.wrapping_mul(sb) as u32,
        ((sa as i64 * sb as i64) >> 32) as u32,
        a & b,
        a | b,
        a ^ b,
        sa.wrapping_shl(b) as u32,
        sa.wrapping_shr(b) as u32,
        sa.min(sb) as u32,
        sa.max(sb) as u32,
        (sa == sb) as u32,
        (sa != sb) as u32,
        (sa < sb) as u32,
        (sa <= sb) as u32,
        sa.wrapping_neg() as u32,
        !a,
        a.leading_zeros(),
        sa.clamp(-16384, 16384) as u32,
    ];
    v.extend([
        a.wrapping_add(b),
        a.wrapping_sub(b),
        a.wrapping_mul(b),
        ((a as u64 * b as u64) >> 32) as u32,
        a & b,
        a | b,
        a ^ b,
        a.wrapping_shl(b),
        a.wrapping_shr(b),
        a.min(b),
        a.max(b),
        (a == b) as u32,
        (a != b) as u32,
        (a < b) as u32,
        (a <= b) as u32,
        a.wrapping_neg(),
        !a,
        a.leading_zeros(),
        a.clamp(3, 1 << 20),
    ]);
    v
}

#[test]
fn ops_match_reference() {
    let (g, outs) = all_ops();
    let mut p = Program::new(&g, &outs);
    let mut out = vec![0; outs.len()];
    let s = samples(200);
    for &a in &s {
        for &b in s.iter().step_by(7) {
            p.run(&[a, b], &[], &mut out);
            assert_eq!(out, reference(a, b), "a={a:#x} b={b:#x}");
        }
    }
}

#[test]
fn constants_fold_with_eval_semantics() {
    let mut g = IntGraph::new();
    let a = g.i32(-7);
    let b = g.u32(33);
    let r = g.shr(a, b); // -7 >> 1
    assert!(matches!(g.node(r), INode::Const(IType::I32, v) if v as i32 == -4));
    let m = g.i32(i32::MIN);
    let n = g.neg(m);
    assert!(matches!(g.node(n), INode::Const(_, v) if v == 0x8000_0000));
}

/// Sum over j < i < n of (i·j) ^ k, k loop invariant (hoisted), nested loops.
fn nested() -> (IntGraph, Vec<IId>) {
    let mut g = IntGraph::new();
    let n = g.param(0, IType::U32);
    let k0 = g.param(1, IType::U32);
    let zero = g.u32(0);
    let out = g.for_range(n, &[zero], |g, i, s| {
        let kk = g.mul(k0, k0); // invariant of both loops
        let inner = g.for_range(i, &[s[0]], |g, j, t| {
            let ij = g.mul(i, j);
            let x = g.xor(ij, kk);
            vec![g.add(t[0], x)]
        });
        vec![inner[0]]
    });
    (g, out)
}

fn nested_ref(n: u32, k: u32) -> u32 {
    let mut s = 0u32;
    for i in 0..n {
        for j in 0..i {
            s = s.wrapping_add(i.wrapping_mul(j) ^ k.wrapping_mul(k));
        }
    }
    s
}

/// Data-dependent trip count: Collatz steps of x, plus a two-state swap
/// (checks the parallel state update).
fn collatz() -> (IntGraph, Vec<IId>) {
    let mut g = IntGraph::new();
    let x = g.param(0, IType::U32);
    let zero = g.u32(0);
    let one = g.u32(1);
    let out = g.while_loop(&[x, zero, one, zero], |g, s| {
        let one = g.u32(1);
        let three = g.u32(3);
        let gt = g.lt(one, s[0]);
        let odd = g.and(s[0], one);
        let is_odd = g.eq(odd, one);
        let t = g.mul(s[0], three);
        let up = g.add(t, one);
        let down = g.shr(s[0], one);
        let nx = g.select(is_odd, up, down);
        let steps = g.add(s[1], one);
        // (a, b) <- (b, a + b): Fibonacci mod 2^32
        let fib = g.add(s[2], s[3]);
        (gt, vec![nx, steps, fib, s[2]])
    });
    (g, vec![out[1], out[3]])
}

fn collatz_ref(mut x: u32) -> [u32; 2] {
    let (mut steps, mut a, mut b) = (0u32, 1u32, 0u32);
    while x > 1 {
        x = if x & 1 == 1 {
            x.wrapping_mul(3).wrapping_add(1)
        } else {
            x >> 1
        };
        steps += 1;
        (a, b) = (a.wrapping_add(b), a);
    }
    [steps, b]
}

/// Loads: sum of i8 bytes and u8 bytes and i32 words over 0..n.
fn loads() -> (IntGraph, Vec<IId>) {
    let mut g = IntGraph::new();
    let bi8 = g.buffer("bi8", Elem::I8);
    let bu8 = g.buffer("bu8", Elem::U8);
    let bw = g.buffer("bw", Elem::I32);
    let n = g.param(0, IType::U32);
    let z = g.i32(0);
    let zu = g.u32(0);
    let out = g.for_range(n, &[z, zu, z], |g, i, s| {
        let a = g.load(bi8, i);
        let b = g.load(bu8, i);
        let c = g.load(bw, i);
        vec![g.add(s[0], a), g.add(s[1], b), g.add(s[2], c)]
    });
    let o: Vec<IId> = out.iter().map(|&x| g.bitcast(IType::U32, x)).collect();
    (g, o)
}

#[test]
fn loops_match_reference() {
    let (g, o) = nested();
    let mut p = Program::new(&g, &o);
    let mut out = [0u32];
    for (n, k) in [(0, 5), (1, 5), (2, 7), (17, 0xdead_beef), (40, 3)] {
        p.run(&[n, k], &[], &mut out);
        assert_eq!(out[0], nested_ref(n, k));
    }
    let (g, o) = collatz();
    let mut p = Program::new(&g, &o);
    let mut out = [0u32; 2];
    for x in [0, 1, 2, 3, 27, 97, 871, 77031] {
        p.run(&[x], &[], &mut out);
        assert_eq!(out, collatz_ref(x), "x={x}");
    }
    let (g, o) = loads();
    let bytes: Vec<i8> = (0..64).map(|i: i32| (i * 37 - 100) as i8).collect();
    let pack = |b: &[u8]| {
        b.chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<u32>>()
    };
    let raw: Vec<u8> = bytes.iter().map(|&b| b as u8).collect();
    let packed = pack(&raw);
    let words: Vec<u32> = (0..64).map(|i: i32| (i * -12345) as u32).collect();
    let mut p = Program::new(&g, &o);
    let mut out = [0u32; 3];
    p.run(&[61], &[&packed, &packed, &words], &mut out);
    let r0: i32 = bytes[..61].iter().map(|&b| b as i32).sum();
    let r1: u32 = raw[..61].iter().map(|&b| b as u32).sum();
    let r2 = words[..61]
        .iter()
        .fold(0i32, |a, &w| a.wrapping_add(w as i32));
    assert_eq!(out, [r0 as u32, r1, r2 as u32]);
}

#[test]
fn invariants_are_hoisted() {
    let (g, o) = nested();
    let src = g.to_wgsl_fn("nested", &o);
    // k·k is computed before the outer loop
    let kk = src.find("((p1) * (p1))").expect("k*k emitted");
    assert!(kk < src.find("loop").unwrap(), "{src}");
}

#[test]
#[should_panic(expected = "unrelated loops")]
fn sibling_loop_states_rejected() {
    let mut g = IntGraph::new();
    let z = g.u32(0);
    let mut leak = None;
    g.while_loop(&[z], |g, s| {
        leak = Some(s[0]);
        (g.bool(false), vec![s[0]])
    });
    g.while_loop(&[z], |g, s| {
        let x = g.add(s[0], leak.unwrap());
        (g.bool(false), vec![x])
    });
}

fn wgsl_module(fns: &str) -> String {
    format!(
        "{WGSL_PRELUDE}\n@group(0) @binding(0) var<storage, read> bi8: array<u32>;\n@group(0) @binding(1) var<storage, read> bu8: array<u32>;\n@group(0) @binding(2) var<storage, read> bw: array<i32>;\n{fns}\n@compute @workgroup_size(1) fn main() {{}}\n"
    )
}

#[test]
fn wgsl_validates() {
    let mut fns = String::new();
    for (name, (g, o)) in [
        ("all_ops", all_ops()),
        ("nested", nested()),
        ("collatz", collatz()),
        ("loads", loads()),
    ] {
        fns.push_str(&g.to_wgsl_fn(name, &o));
    }
    let src = wgsl_module(&fns);
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(&src)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
}

/// The Rust lowering, compiled with rustc and run against the interpreter.
#[test]
fn rust_lowering_matches_eval() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let dir = std::env::temp_dir().join(format!("tang_expr_int_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (g0, o0) = all_ops();
    let (g1, o1) = nested();
    let (g2, o2) = collatz();
    let (g3, o3) = loads();
    let s = samples(40);
    let bytes: Vec<u32> = (0..16u32).map(|i| i.wrapping_mul(0x9e37_79b9)).collect();
    let mut src = String::new();
    for (n, (g, o)) in [
        ("all_ops", (&g0, &o0)),
        ("nested", (&g1, &o1)),
        ("collatz", (&g2, &o2)),
        ("loads", (&g3, &o3)),
    ] {
        src.push_str(&g.to_rust(n, o));
    }
    src.push_str("fn main() {\n");
    src.push_str(&format!(
        "    let s: &[u32] = &{s:?};\n    let b: &[u32] = &{bytes:?};\n"
    ));
    src.push_str(
        "    for &a in s { for &c in s { println!(\"{:?}\", all_ops(a as i32, c as i32)); } }\n",
    );
    src.push_str("    for &a in s { println!(\"{:?}\", nested(a % 23, a)); println!(\"{:?}\", collatz(a % 100000)); }\n");
    src.push_str("    for n in 0..17 { println!(\"{:?}\", loads(n, b, b, b)); }\n}\n");
    let file = dir.join("lowered.rs");
    std::fs::write(&file, &src).unwrap();
    let exe = dir.join("lowered");
    let st = std::process::Command::new(&rustc)
        .args(["-O", "--edition", "2021", "-o"])
        .arg(&exe)
        .arg(&file)
        .status();
    let Ok(st) = st else {
        eprintln!("rustc not available; skipping");
        return;
    };
    assert!(st.success(), "generated Rust failed to compile:\n{src}");
    let got = String::from_utf8(std::process::Command::new(&exe).output().unwrap().stdout).unwrap();
    let mut want = String::new();
    let fmt = |v: &[u32], ty: &[IType]| -> String {
        let parts: Vec<String> = v
            .iter()
            .zip(ty)
            .map(|(&x, t)| {
                if *t == IType::I32 {
                    (x as i32).to_string()
                } else {
                    x.to_string()
                }
            })
            .collect();
        if ty.windows(2).all(|w| w[0] == w[1]) {
            format!("[{}]", parts.join(", "))
        } else {
            format!("({},)", parts.join(", "))
        }
    };
    let tys = |g: &IntGraph, o: &[IId]| o.iter().map(|&x| g.ty(x)).collect::<Vec<_>>();
    for &a in &s {
        for &c in &s {
            want += &fmt(&g0.eval(&o0, &[a, c], &[]), &tys(&g0, &o0));
            want.push('\n');
        }
    }
    for &a in &s {
        want += &fmt(&g1.eval(&o1, &[a % 23, a], &[]), &tys(&g1, &o1));
        want.push('\n');
        want += &fmt(&g2.eval(&o2, &[a % 100000], &[]), &tys(&g2, &o2));
        want.push('\n');
    }
    for n in 0..17 {
        want += &fmt(
            &g3.eval(&o3, &[n], &[&bytes, &bytes, &bytes]),
            &tys(&g3, &o3),
        );
        want.push('\n');
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(got, want);
}

#[test]
fn div_rem_have_wgsl_semantics() {
    let mut g = IntGraph::new();
    let a = g.param(0, IType::I32);
    let b = g.param(1, IType::I32);
    let (ua, ub) = (g.bitcast(IType::U32, a), g.bitcast(IType::U32, b));
    let o = vec![g.div(a, b), g.rem(a, b), g.div(ua, ub), g.rem(ua, ub)];
    let mut p = Program::new(&g, &o);
    for &(x, y) in &[
        (7i32, 2i32),
        (-7, 2),
        (7, -2),
        (5, 0),
        (i32::MIN, -1),
        (i32::MIN, 0),
        (-1, 3),
    ] {
        let mut out = [0u32; 4];
        p.run(&[x as u32, y as u32], &[], &mut out);
        let sd = if y == 0 { x } else { x.wrapping_div(y) };
        let sr = if y == 0 { 0 } else { x.wrapping_rem(y) };
        let (ux, uy) = (x as u32, y as u32);
        assert_eq!(
            out,
            [
                sd as u32,
                sr as u32,
                ux.checked_div(uy).unwrap_or(ux),
                ux.checked_rem(uy).unwrap_or(0)
            ]
        );
    }
}

fn capped_collatz(cap: u32) -> (IntGraph, Vec<IId>) {
    let (mut g, o) = collatz();
    g.cap_loop(o[0], cap);
    (g, o)
}

#[test]
fn loop_caps_bound_trips() {
    let (g, o) = capped_collatz(5);
    let (h, _) = collatz();
    let ho = collatz().1;
    for x in [1u32, 2, 3, 27, 97] {
        let c = g.eval(&o, &[x], &[]);
        let u = h.eval(&ho, &[x], &[]);
        assert_eq!(c[0], u[0].min(5), "x={x}");
    }
    // nested: inner loops re-arm their counter on every entry
    let (mut g, o) = nested();
    let inner = g.loops.iter().position(|l| l.parent.is_some()).unwrap();
    g.loops[inner].cap = Some(3);
    let n = 9u32;
    let mut s = 0u32;
    for i in 0..n {
        for j in 0..i.min(3) {
            s = s.wrapping_add(i.wrapping_mul(j) ^ 25);
        }
    }
    assert_eq!(g.eval(&o, &[n, 5], &[]), vec![s]);
}

fn abi() -> (Vec<IType>, Vec<(&'static str, Elem)>) {
    (
        vec![IType::U32, IType::U32],
        vec![("bi8", Elem::I8), ("bu8", Elem::U8), ("bw", Elem::I32)],
    )
}

#[test]
fn wire_round_trip_and_guarded_wgsl() {
    let (mut g, o) = nested();
    for l in 0..g.loops.len() {
        g.loops[l].cap = Some(64);
    }
    let (pt, bufs) = abi();
    let bytes = g.to_bytes(&o);
    let (h, ho) =
        IntGraph::from_bytes(&bytes, &pt, &bufs, &[IType::U32], &Limits::default()).unwrap();
    for (n, k) in [(0u32, 3u32), (5, 7), (40, 1)] {
        assert_eq!(h.eval(&ho, &[n, k], &[]), g.eval(&o, &[n, k], &[]));
    }
    let (mut lg, lo) = loads();
    for l in 0..lg.loops.len() {
        lg.loops[l].cap = Some(64);
    }
    let lb = lg.to_bytes(&lo);
    let (lh, lho) = IntGraph::from_bytes(
        &lb,
        &pt[..lg.nparams().len()],
        &bufs,
        &vec![IType::I32; lo.len()]
            .iter()
            .zip(&lo)
            .map(|(_, &x)| lg.ty(x))
            .collect::<Vec<_>>(),
        &Limits::default(),
    )
    .unwrap();
    let body = lh.to_wgsl_body_guarded(
        &lho,
        &["0u", "1u"],
        &(0..lho.len()).map(|_| "_").collect::<Vec<_>>(),
        1,
    );
    let src = wgsl_module(&format!("fn guarded() {{\n{body}}}\n"));
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(&src)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
    // out-of-bounds loads read 0
    assert!(lh
        .eval(&lho, &[1 << 30, 1 << 30], &[&[], &[], &[]])
        .iter()
        .all(|&v| v == 0));
}

#[test]
fn wire_rejects_bad_graphs() {
    let (pt, bufs) = abi();
    let lim = Limits::default();
    let rd = |b: &[u8], p: &[IType]| IntGraph::from_bytes(b, p, &bufs, &[IType::U32], &lim);
    let (g, o) = capped_collatz(100);
    let good = g.to_bytes(&o[..1]);
    assert!(rd(&good, &pt[..1]).is_ok());
    // wrong param type
    assert!(rd(&good, &[IType::I32]).is_err());
    // over the trip limit / cost limit
    let (g2, o2) = capped_collatz(1 << 13);
    assert!(rd(&g2.to_bytes(&o2[..1]), &pt[..1]).is_err());
    let tight = Limits {
        max_cost: 10,
        ..lim
    };
    assert!(IntGraph::from_bytes(&good, &pt[..1], &bufs, &[IType::U32], &tight).is_err());
    // truncations, trailing bytes and random corruption never panic
    for n in 0..good.len() {
        assert!(rd(&good[..n], &pt[..1]).is_err());
    }
    let mut t = good.clone();
    t.push(0);
    assert!(rd(&t, &pt[..1]).is_err());
    let mut seed = 99u64;
    for _ in 0..20000 {
        let mut b = good.clone();
        for _ in 0..1 + rng(&mut seed) % 4 {
            let i = 4 + rng(&mut seed) as usize % (b.len() - 4);
            b[i] = rng(&mut seed) as u8;
        }
        if let Ok((h, ho)) = rd(&b, &pt[..1]) {
            // whatever passes is total and terminates
            let _ = h.eval(&ho, &[27], &[&[], &[], &[]]);
            let _ = h.to_wgsl_body_guarded(&ho, &["1u"], &["_"], 1);
        }
    }
}

/// A loop of read-modify-write steps on a scratch buffer: `scr[(7i + 3) % 13]
/// += i` for `i < n`, returning the sum of the values read and the token.
fn scratch() -> (IntGraph, Vec<IId>) {
    let mut g = IntGraph::new();
    let sb = g.buffer_rw("scr", Elem::U32, 13, "sb");
    let n = g.param(0, IType::U32);
    let z = g.u32(0);
    let t0 = g.token();
    let out = g.for_range(n, &[z, t0], |g, i, s| {
        let (seven, three, m) = (g.u32(7), g.u32(3), g.u32(13));
        let a = g.mul(i, seven);
        let a = g.add(a, three);
        let idx = g.rem(a, m);
        let v = g.load_rw(sb, idx, s[1]);
        let nv = g.add(v, i);
        let t = g.store(sb, idx, nv, s[1]);
        vec![g.add(s[0], v), t]
    });
    g.cap_loop(out[0], 64);
    // an out-of-range store is dropped, an out-of-range load reads 0
    let big = g.u32(99);
    let one = g.u32(1);
    let t = g.store(sb, big, one, out[1]);
    let v = g.load_rw(sb, big, t);
    let s = g.add(out[0], v);
    (g, vec![s, t])
}

fn scratch_ref(n: u32, scr: &mut [u32]) -> u32 {
    let mut s = 0u32;
    for i in 0..n {
        let idx = ((i * 7 + 3) % 13) as usize;
        s = s.wrapping_add(scr[idx]);
        scr[idx] = scr[idx].wrapping_add(i);
    }
    s
}

#[test]
fn scratch_effects_are_ordered() {
    let (g, o) = scratch();
    let mut p = Program::new(&g, &o);
    let bytes = g.to_bytes(&o);
    let (g2, o2) = IntGraph::from_bytes_rw(
        &bytes,
        &[IType::U32],
        &[],
        &[("scr", Elem::U32, 13, "sb")],
        &[IType::U32, IType::U32],
        &Limits::default(),
    )
    .unwrap();
    let mut p2 = Program::new(&g2, &o2);
    for n in [0u32, 1, 5, 13, 40] {
        let mut want = [5u32; 13];
        let s = scratch_ref(n, &mut want);
        for p in [&mut p, &mut p2] {
            let mut scr = [5u32; 13];
            let mut out = [0u32; 2];
            p.run_rw(&[n], &[], &mut [&mut scr], &mut out);
            assert_eq!(out, [s, 0]);
            assert_eq!(scr, want);
        }
    }
    // the WGSL lowerings (scratch in workgroup memory) validate
    for guarded in [false, true] {
        let body = if guarded {
            g.to_wgsl_body_guarded(&o, &["n"], &["a", "b"], 1)
        } else {
            g.to_wgsl_body(&o, &["n"], &["a", "b"], 1)
        };
        assert!(body.contains("scr[(sb) + "), "{body}");
        let src = format!("var<workgroup> scr: array<u32, 64>;\nvar<private> sb: u32;\nfn f(n: u32) {{\n    var a: u32;\n    var b: u32;\n{body}}}\n@compute @workgroup_size(1) fn main() {{ f(3u); }}\n");
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(&src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
    }
    assert!(g.to_rust("f", &o).contains("scr: &mut [u32]"));
}

#[test]
fn scratch_needs_a_scratch_buffer() {
    let mut g = IntGraph::new();
    let b = g.buffer("ro", Elem::U32);
    let (i, t) = (g.u32(0), g.token());
    let n = INode::Store(b, i, i, t);
    assert!(g.try_infer(n).is_err());
    let sb = g.buffer_rw("scr", Elem::U32, 4, "0u");
    assert!(g.try_infer(INode::Load(sb, i)).is_err());
}
