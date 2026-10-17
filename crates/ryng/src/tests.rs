use super::*;

const TOY_Q: u32 = 17;

fn res(v: &[i64], q: u32) -> Vec<u32> {
    v.iter().map(|x| x.rem_euclid(q as i64) as u32).collect()
}

/// The toy's public a, moved into the NTT domain (the KEM samples A there).
fn toy_ahat(a: &[u32]) -> Vec<u32> {
    let mut g = ModGraph::new(TOY_Q);
    let x = g.input(&[4], Label::Public);
    let y = ntt(&mut g, x, Params::TOY.zeta);
    g.eval(&[a.to_vec()], &[y]).remove(0)
}

#[test]
fn toy_matches_the_hand_calculation() {
    let r = Ryng::new(Params::TOY);
    let q = TOY_Q;
    let ahat = toy_ahat(&[5, 11, 3, 14]);
    let s = res(&[1, 0, -1, 1], q);
    let e = res(&[0, 1, 0, -1], q);
    let t = r.pke_keygen(&ahat, &s, &e);
    assert_eq!(t, vec![14, 6, 1, 7]);

    let rr = res(&[1, -1, 0, 0], q);
    let e1 = res(&[0, 0, 1, 0], q);
    let e2 = res(&[-1, 0, 0, 1], q);
    let (u, v) = r.pke_encrypt(&ahat, &t, &rr, &e1, &e2, &[1, 0, 1, 1]);
    assert_eq!(u, vec![2, 6, 10, 11]);
    assert_eq!(v, vec![11, 9, 3, 15]);

    let (m, w) = r.pke_decrypt(&u, &v, &s);
    assert_eq!(w, vec![5, 2, 6, 8]);
    assert_eq!(m, vec![1, 0, 1, 1]);
}

#[test]
fn every_graph_passes_the_secrecy_check() {
    for p in [Params::TOY, Params::RYNG] {
        let r = Ryng::new(p);
        let [kg, en, de] = r.check().expect("no secret steers an address or a count");
        // the only declassifications are the MLWE assumption, one per published poly
        assert!(kg.iter().all(|&x| x == PK_PROOF) && kg.len() == p.k);
        assert!(en.iter().all(|&x| x == CT_PROOF) && en.len() == p.k + 1);
        assert!(de.is_empty());
        assert!(
            r.dec_output_is_secret(),
            "the decrypted message must stay secret"
        );
    }
}

#[test]
fn sizes_match_the_design() {
    let p = Params::RYNG;
    assert_eq!((p.pk_bytes(), p.ct_bytes()), (1312, 1440));
    let r = Ryng::new(p);
    let (pk, _) = r.keygen(&[1; 64]);
    let (ct, _) = r.seal(&pk, &[2; 32]);
    assert_eq!((pk.len(), ct.len()), (1312, 1440));
}

#[test]
fn seal_open_round_trips() {
    let r = Ryng::new(Params::RYNG);
    for i in 0..8u8 {
        let (pk, sk) = r.keygen(&[i; 64]);
        let (ct, key) = r.seal(&pk, &[i.wrapping_mul(37).wrapping_add(1); 32]);
        assert_eq!(r.open(&sk, &ct), key, "seed {i}");
    }
}

#[test]
fn tampering_gives_the_rejection_key() {
    let r = Ryng::new(Params::RYNG);
    let (pk, sk) = r.keygen(&[3; 64]);
    let (mut ct, key) = r.seal(&pk, &[4; 32]);
    ct[100] ^= 1;
    let got = r.open(&sk, &ct);
    assert_ne!(got, key);
    let z = &sk[sk.len() - 32..];
    assert_eq!(got, shake256_32(b"reject", &[z, &ct[..]].concat()));
    assert_eq!(r.open(&sk, &ct), got, "deterministic");
}

#[test]
fn decryption_noise_stays_far_inside_the_bound() {
    // |w − m·⌊q/2⌋| must stay below q/4; the design predicts < 2^−185 failure,
    // so every sampled coefficient should sit far inside it.
    let p = Params::RYNG;
    let r = Ryng::new(p);
    let mut worst = 0f64;
    for i in 0..4u8 {
        let (pk, sk) = r.keygen(&[i + 10; 64]);
        let m = [0xa5u8; 32];
        let (ct, _) = r.seal(&pk, &m);
        let s_len = p.k * p.n * bits(p.q) as usize / 8;
        let s = unpack(&sk[..s_len], bits(p.q), p.k * p.n);
        let (u, v) = r.unpack_ct(&ct);
        let (bits_out, w) = r.pke_decrypt(&u, &v, &s);
        assert_eq!(bits_out, bytes_to_bits(&m, p.n));
        for (wi, bi) in w.iter().zip(&bits_out) {
            let centre = (*bi * (p.q / 2)) as i64;
            let d = (*wi as i64 - centre).rem_euclid(p.q as i64);
            let d = d.min(p.q as i64 - d) as f64;
            worst = worst.max(d / (p.q as f64 / 4.0));
        }
    }
    eprintln!("worst decryption noise: {worst:.4} of the q/4 margin");
    assert!(
        worst < 0.25,
        "worst noise used {worst:.3} of the q/4 margin"
    );
}

/// The FO transform on the host — SHAKE-256 G and J, host samplers, the PKE
/// graphs — as an independent check on the in-graph seal and open.
fn host_seal(r: &Ryng, pk: &[u8], m: &[u8; 32]) -> (Vec<u8>, [u8; 32]) {
    let p = r.p;
    let h = hash256(b"pk", pk);
    let kc = shake(b"seal", &[&m[..], &h], 64);
    let (key, coins) = kc.split_at(32);
    let (rho, t1b) = pk.split_at(32);
    let a = sample_a(&p, rho);
    let t1 = unpack(t1b, bits(p.q) - p.dt, p.k * p.n);
    let rr = cbd2(&p, coins, 0);
    let e1 = uniform12(&p, coins, p.k as u8);
    let e2 = uniform12(&p, coins, 2 * p.k as u8);
    let (u, v) = r.pke_encrypt(&a, &t1, &rr, &e1, &e2, &bytes_to_bits(m, p.n));
    let mut ct = pack(&u, p.du);
    ct.extend(pack(&v, p.dv));
    (ct, key.try_into().unwrap())
}

fn shake(name: &[u8], parts: &[&[u8]], len: usize) -> Vec<u8> {
    let mut x = Shake256::default();
    x.update(&sep(name));
    for p in parts {
        x.update(p);
    }
    let mut out = vec![0; len];
    x.finalize_xof().read(&mut out);
    out
}

#[test]
fn in_graph_fo_matches_the_host_reference() {
    let r = Ryng::new(Params::RYNG);
    let p = r.p;
    let (pk, sk) = r.keygen(&[21; 64]);
    // keygen: ρ ‖ σ = SHAKE-256(sep ‖ d), s from σ by the host sampler
    let rs = shake(b"keygen", &[&[21u8; 32]], 64);
    assert_eq!(&pk[..32], &rs[..32]);
    let s_len = p.k * p.n * bits(p.q) as usize / 8;
    assert_eq!(
        unpack(&sk[..s_len], bits(p.q), p.k * p.n),
        cbd2(&p, &rs[32..], 0)
    );
    for i in 0..3u8 {
        let m = [i.wrapping_mul(91); 32];
        let (ct, key) = r.seal(&pk, &m);
        assert_eq!((ct.clone(), key), host_seal(&r, &pk, &m), "seal {i}");
        assert_eq!(r.open(&sk, &ct), key, "open {i}");
    }
}

#[test]
fn every_kem_graph_passes_the_secrecy_check() {
    let r = Ryng::new(Params::RYNG);
    let k = r.p.k;
    let proofs = r
        .check_kem()
        .expect("no secret steers an address or a count");
    assert_eq!(proofs[0], vec![RHO_PROOF]);
    assert_eq!(proofs[1], vec![PK_PROOF; k]);
    assert_eq!(proofs[2], vec![CT_PROOF; k + 1]);
    assert!(
        proofs[3].is_empty(),
        "open declassifies nothing: the key and the verdict stay secret"
    );
}

#[test]
fn packing_round_trips() {
    let v: Vec<u32> = (0..256).map(|i| (i * 37) % 1024).collect();
    assert_eq!(unpack(&pack(&v, 10), 10, 256), v);
    let b: Vec<u32> = (0..256).map(|i| i % 2).collect();
    assert_eq!(bytes_to_bits(&bits_to_bytes(&b), 256), b);
}

#[test]
fn ct_eq_is_a_mask() {
    assert_eq!(ct_eq(b"abc", b"abc"), 0xff);
    assert_eq!(ct_eq(b"abc", b"abd"), 0x00);
}

/// SHA3-256 over (pk ‖ sk ‖ ct ‖ key) for three seeds: any change to what the
/// KEM computes changes this.
fn kat() -> String {
    let r = Ryng::new(Params::RYNG);
    let mut h = Sha3_256::new();
    for i in 0..3u8 {
        let (pk, sk) = r.keygen(&[i; 64]);
        let (ct, key) = r.seal(&pk, &[i + 100; 32]);
        for part in [&pk[..], &sk, &ct, &key] {
            Digest::update(&mut h, part);
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn known_answer() {
    // Re-pinned when G moved from SHA3-512 to SHAKE-256 (the in-graph FO);
    // `in_graph_fo_matches_the_host_reference` ties it to an independent host computation.
    assert_eq!(
        kat(),
        "95e7754edebac9caa2e3328748d68f45594897e462315e0c997b25fb979af481"
    );
}

#[test]
fn keccak_graph_matches_sha3_shake256() {
    use keccak::*;
    let mut g = ModGraph::new(Params::RYNG.q);
    let t = Tables::new(&mut g);
    let st = g.input_words(&[50], Label::Secret);
    let out = squeeze(&mut g, &t, st, SHAKE256_RATE, 3);
    assert_eq!(
        g.check(&[(out, Label::Secret)]),
        Ok(vec![]),
        "π's gather index is public"
    );
    assert!(
        g.check(&[(out, Label::Public)]).is_err(),
        "a hash of a secret is still secret"
    );
    for len in [0usize, 1, 33, 65, 135] {
        let msg: Vec<u8> = (0..len).map(|i| (i * 31 + len) as u8).collect();
        let words = g.eval(&[absorb_one_block(&msg, 136)], &[out]).remove(0);
        let mut want = vec![0u8; 3 * 136];
        let mut x = Shake256::default();
        x.update(&msg);
        x.finalize_xof().read(&mut want);
        assert_eq!(words_to_bytes(&words), want, "len {len}");
    }
}

#[test]
fn sampler_graph_matches_host_reference() {
    let p = Params::RYNG;
    let s = sample::Sampler::new(p.q, p.n, p.k, p.k + 1);
    assert_eq!(s.check(), Ok(vec![]));
    let seed = [0x5au8; 32];
    let (c, u) = s.run(prf_states(&seed, 2 * p.k + 1));
    assert_eq!(c, cbd2(&p, &seed, 0));
    let mut want = uniform12(&p, &seed, p.k as u8);
    want.extend(uniform12(&p, &seed, 2 * p.k as u8));
    assert_eq!(u, want);
}

mod signature {
    use crate::sig::*;
    use tang_compile::tensor::modq::{Label, ModGraph};

    #[test]
    fn sign_verify_round_trips_and_rejects_forgeries() {
        let s = Sig::new();
        let (pk, sk) = s.keygen(&[7; 32]);
        assert_eq!(pk.len(), PK_BYTES);
        let msg = b"ryng: one ring to sign them all";
        let (sig, attempts) = s.sign(&sk, msg);
        eprintln!("signed in {attempts} attempt(s)");
        assert_eq!(sig.len(), SIG_BYTES);
        assert!(s.verify(&pk, msg, &sig));
        assert!(!s.verify(&pk, b"another message", &sig));
        let mut bad = sig.clone();
        bad[100] ^= 1;
        assert!(!s.verify(&pk, msg, &bad));
        let (pk2, _) = s.keygen(&[8; 32]);
        assert!(!s.verify(&pk2, msg, &sig));
        assert_eq!(s.sign(&sk, msg).0, sig, "deterministic");
    }

    #[test]
    fn attempts_average_near_the_predicted_5_09() {
        // FIPS 204 predicts exp(256·β·(ℓ/γ₁ + k/γ₂)) ≈ 5.09 for these parameters
        let s = Sig::new();
        let (_, sk) = s.keygen(&[9; 32]);
        let runs: usize = std::env::var("RYNG_RUNS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let total: usize = (0..runs).map(|i| s.sign(&sk, &[i as u8; 8]).1).sum();
        let mean = total as f64 / runs as f64;
        eprintln!("mean attempts over {runs} signatures: {mean:.2}");
        // geometric with p ≈ 0.2: the mean of 16 is within [2.5, 9] except with tiny probability
        assert!((2.5..=9.0).contains(&mean), "mean {mean}");
    }

    #[test]
    fn every_signature_graph_passes_the_secrecy_check() {
        let s = Sig::new();
        let proofs = s.check().expect("no secret steers an address or a count");
        assert_eq!(proofs[0], vec![RHO_PROOF]);
        assert!(proofs[1].iter().all(|&p| p == PK_PROOF) && proofs[1].len() == K);
        // the attempt graph: z, h, c̃ gated by acceptance, plus the attempt count
        let mut want = vec![SIG_PROOF; L + K + 1];
        want.push(ATTEMPT_PROOF);
        let mut got = proofs[2].clone();
        got.sort();
        want.sort();
        assert_eq!(got, want);
        assert!(proofs[3].is_empty(), "verification is public end to end");
    }

    #[test]
    fn sample_in_ball_graph_matches_host() {
        let mut g = ModGraph::new(Q);
        let t = crate::keccak::Tables::new(&mut g);
        let ct = g.input_words(&[12], Label::Secret);
        let c = sample_in_ball(&mut g, &t, ct);
        assert_eq!(
            g.check(&[(c, Label::Secret)]),
            Ok(vec![]),
            "secret index reads go through the mux"
        );
        for seed in 0..4u8 {
            let bytes: Vec<u8> = (0..48).map(|i| i as u8 ^ seed.wrapping_mul(77)).collect();
            let words: Vec<u32> = bytes
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let got = g.eval(&[words], &[c]).remove(0);
            assert_eq!(got, sample_in_ball_host(&bytes));
            assert_eq!(
                got.iter().filter(|&&x| x != 0).count(),
                49,
                "τ = 49 nonzero"
            );
        }
    }
}

/// Rejection rates per condition (run with RYNG_DIAG=<attempts>, --nocapture).
#[test]
fn rejection_rates() {
    use crate::sig::*;
    let Some(n) = std::env::var("RYNG_DIAG")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    else {
        return;
    };
    let s = Sig::new();
    let (_, sk) = s.keygen(&[9; 32]);
    let rows = s.diagnose(&sk, b"rates", n);
    let rate = |i: usize| rows.iter().filter(|r| r[i] == 1).count() as f64 / n as f64;
    let all = rows
        .iter()
        .filter(|r| r[..4].iter().all(|&x| x == 1))
        .count() as f64
        / n as f64;
    let mean_h = rows.iter().map(|r| r[4] as f64).sum::<f64>() / n as f64;
    let max_h = rows.iter().map(|r| r[4]).max().unwrap_or(0);
    eprintln!("attempts {n}: z ok {:.3}  r0 ok {:.3}  ct0 ok {:.3}  hints ok {:.3}  all {:.3}  hints mean {:.1} max {}",
        rate(0), rate(1), rate(2), rate(3), all, mean_h, max_h);
}

/// Compiled C: the same answers as the interpreter, and bodies that the
/// machine-code lint passes.
mod compiled {
    use super::*;
    use tang_compile::tensor::modq::c::welch_t;

    fn open_inputs(r: &Ryng, seed: u8) -> (Vec<Vec<u32>>, [u8; 32]) {
        let p = r.p;
        let (pk, sk) = r.keygen(&[seed; 64]);
        let (ct, key) = r.seal(&pk, &[seed ^ 0x55; 32]);
        let s_len = p.k * p.n * bits(p.q) as usize / 8;
        let s = unpack(&sk[..s_len], bits(p.q), p.k * p.n);
        let pk_len = p.pk_bytes();
        let h = &sk[s_len + pk_len..s_len + pk_len + 32];
        let z = &sk[s_len + pk_len + 32..];
        let a = sample_a(&p, &pk[..32]);
        let t1 = unpack(&pk[32..], bits(p.q) - p.dt, p.k * p.n);
        let (u, v) = r.unpack_ct(&ct);
        let (jt, _) = graph::tail(64 + ct.len(), &[(0, &sep(b"reject")), (64, &ct)]);
        (vec![a, t1, s, words(z), u, v, g_tail(h), jt], key)
    }

    #[test]
    fn open_in_c_matches_the_interpreter() {
        let r = Ryng::new(Params::RYNG);
        let fo = r.fo.as_ref().unwrap();
        let built = fo.open.g.compile(&[fo.open.key]).expect("a C compiler");
        let (ins, key) = open_inputs(&r, 3);
        let c = built.run(&ins).unwrap();
        assert_eq!(c, fo.open.g.eval(&ins, &[fo.open.key]));
        assert_eq!(bytes(&c[0]), key);
    }

    #[test]
    fn signing_attempt_in_c_matches_the_interpreter() {
        let s = sig::Sig::new();
        let (_, sk) = s.keygen(&[4; 32]);
        let ins = s.attempt_example(&sk, b"compiled");
        let graphs = s.graphs();
        let (g, outs) = &graphs[2];
        let built = g.compile(outs).expect("a C compiler");
        assert_eq!(built.run(&ins).unwrap(), g.eval(&ins, outs));
    }

    #[test]
    fn every_ryng_graph_lints_clean() {
        let r = Ryng::new(Params::RYNG);
        let fo = r.fo.as_ref().unwrap();
        let mut kem: Vec<(&ModGraph, Vec<ModId>)> = vec![
            (&fo.seeds.g, vec![fo.seeds.rho, fo.seeds.sigma]),
            (
                &fo.keyt.g,
                fo.keyt.t1.iter().chain(&fo.keyt.s).copied().collect(),
            ),
            (
                &fo.seal.g,
                fo.seal
                    .u
                    .iter()
                    .copied()
                    .chain([fo.seal.v, fo.seal.key])
                    .collect(),
            ),
            (&fo.open.g, vec![fo.open.key]),
        ];
        let s = sig::Sig::new();
        kem.extend(s.graphs());
        let mut total = 0;
        for (i, (g, outs)) in kem.iter().enumerate() {
            let built = g.compile(outs).expect("a C compiler");
            let n = built.lint().unwrap_or_else(|bad| {
                panic!("graph {i} leaks:\n{}", bad[..bad.len().min(20)].join("\n"))
            });
            total += n;
        }
        eprintln!("{total} body functions across 8 graphs: no conditional branch, no divide");
    }

    /// The KEM's open on the GPU: the same keys as the interpreter for every
    /// instance of a batch, and throughput against the C build.
    #[test]
    #[cfg(target_os = "macos")]
    fn open_on_metal() {
        let r = Ryng::new(Params::RYNG);
        let fo = r.fo.as_ref().unwrap();
        let (msl, _) = fo.open.g.to_metal(&[fo.open.key]);
        eprintln!(
            "open: {} nodes → {} distinct Metal kernels",
            fo.open.g.nodes_sizes().len(),
            msl.matches("kernel void").count()
        );
        let built = fo
            .open
            .g
            .compile_metal(&[fo.open.key])
            .expect("clang with Metal");
        let cases: Vec<_> = (0..4u8).map(|i| open_inputs(&r, 40 + i)).collect();
        // a tampered ciphertext too: the rejection key, on the GPU
        let mut bad = cases[0].0.clone();
        bad[4][7] ^= 1;
        let want_bad = fo.open.g.eval(&bad, &[fo.open.key]).remove(0);
        let batch: Vec<Vec<Vec<u32>>> = (0..32)
            .map(|i| {
                if i == 31 {
                    bad.clone()
                } else {
                    cases[i % 4].0.clone()
                }
            })
            .collect();
        let (got, _) = built.run(&batch).expect("a Metal device");
        for (i, out) in got.iter().enumerate() {
            let want = if i == 31 {
                want_bad.clone()
            } else {
                words(&cases[i % 4].1)
            };
            assert_eq!(out[0], want, "instance {i}");
        }
        if std::env::var_os("RYNG_BENCH").is_some() {
            let c = fo.open.g.compile(&[fo.open.key]).expect("cc");
            let one = vec![cases[0].0.clone(); 64];
            let mut t = c.time(&one).unwrap();
            t.sort();
            let cpu = t[t.len() / 2] as f64;
            eprintln!(
                "open, C -O2 (noinline bodies), one core: {:.1} µs",
                cpu / 1e3
            );
            for b in [1usize, 32, 128, 512, 2048] {
                let sets: Vec<_> = (0..b).map(|i| cases[i % 4].0.clone()).collect();
                let (_, ns) = built.run(&sets).unwrap();
                eprintln!(
                    "open, Metal batch {b:4}: {:8.2} ms device, {:7.1} µs per open, {:5.1}× one C core",
                    ns as f64 / 1e6,
                    ns as f64 / 1e3 / b as f64,
                    cpu * b as f64 / ns as f64
                );
            }
        }
    }

    /// dudect: does open's running time depend on the secret key, or on
    /// whether the ciphertext was valid? Fixed class: the key the ciphertext
    /// was made for (re-encryption matches). Random class: random s and z
    /// (re-encryption mismatches, implicit rejection). |t| < 4.5 passes.
    #[test]
    fn open_runs_in_constant_time() {
        let n: usize = std::env::var("RYNG_DUDECT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);
        let r = Ryng::new(Params::RYNG);
        let p = r.p;
        let fo = r.fo.as_ref().unwrap();
        let built = fo.open.g.compile(&[fo.open.key]).expect("a C compiler");
        let (fixed, _) = open_inputs(&r, 7);
        let mut st = 0x1234_5678_u64;
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as u32
        };
        let mut sets = Vec::with_capacity(2 * n);
        let mut class = Vec::with_capacity(2 * n);
        for _ in 0..2 * n {
            let c = next() & 1;
            let mut ins = fixed.clone();
            if c == 1 {
                ins[2] = (0..p.k * p.n).map(|_| next() % p.q).collect();
                ins[3] = (0..8).map(|_| next()).collect();
            }
            sets.push(ins);
            class.push(c);
        }
        let times = built.time(&sets).unwrap();
        let (a, b): (Vec<u64>, Vec<u64>) = {
            let mut a = Vec::new();
            let mut b = Vec::new();
            for (t, c) in times.iter().zip(&class) {
                if *c == 0 {
                    a.push(*t)
                } else {
                    b.push(*t)
                }
            }
            (a, b)
        };
        let med = |v: &[u64]| {
            let mut v = v.to_vec();
            v.sort();
            v[v.len() / 2]
        };
        let ts: Vec<f64> = [0.0, 0.1, 0.3]
            .iter()
            .map(|&c| welch_t(&a, &b, c))
            .collect();
        eprintln!(
            "open: {} vs {} runs, median {} vs {} ns, t at crop 0/0.1/0.3 = {:.2} / {:.2} / {:.2}",
            a.len(),
            b.len(),
            med(&a),
            med(&b),
            ts[0],
            ts[1],
            ts[2]
        );
        assert!(
            ts.iter().all(|t| t.abs() < 4.5),
            "timing depends on the secret: t = {ts:?}"
        );
    }
}

#[test]
fn graph_sizes() {
    if std::env::var_os("RYNG_SIZES").is_none() {
        return;
    }
    let r = Ryng::new(Params::RYNG);
    let fo = r.fo.as_ref().unwrap();
    for (name, g) in [
        ("seeds", &fo.seeds.g),
        ("keyt", &fo.keyt.g),
        ("seal", &fo.seal.g),
        ("open", &fo.open.g),
    ] {
        let words: usize = g.nodes_sizes().iter().sum();
        eprintln!(
            "{name}: {} nodes, {:.1} MB of buffers per instance",
            g.nodes_sizes().len(),
            words as f64 * 4.0 / 1e6
        );
    }
}
