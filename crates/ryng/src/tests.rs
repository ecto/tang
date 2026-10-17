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
        let h = hash256(b"pk", &pk);
        let (_, ct) = r.encrypt_fo(&pk, &m, &h);
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
    assert_eq!(
        kat(),
        "3cc1d807609d6b76fa22cbb8727fdf325c9d928463ca46af7dd1b76e4ba8c6d7"
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
