//! Needle retrieval through the compressed-KV decode path (see `tang_llm::kvcomp`).
//!
//!     cargo run --release -p tang-llm --features metal --example kvcomp_niah -- \
//!         <model-dir> <hay.u32> --ctx 2048,8192 --depths 0.1,0.3,0.5,0.7,0.9 \
//!         --modes bf16,tkv:0.07:16:2
//!
//! Same protocol as `bench/kv_hf.py niah`: a "special magic number for <city>" sentence is
//! spliced into a haystack of `ctx` tokens at each depth; the context is prefilled and
//! compressed (whole 128-token blocks), then the question and answer are teacher-forced one
//! decode step at a time, so every step attends to the decoded cache. A case is correct when
//! every answer token is the argmax. Prints one JSON line per (ctx, mode).
use tang_compute::MetalDevice;
use tang_llm::kvcomp::{KvMode, KvOpts};
use tang_llm::model::Model;
use tokenizers::Tokenizer;

const CITIES: [&str; 10] = [
    "Lisbon", "Nairobi", "Osaka", "Quito", "Tbilisi", "Perth", "Bergen", "Cusco", "Hanoi", "Dakar",
];

fn flag(a: &[String], f: &str) -> Option<String> {
    a.iter().position(|x| x == f).map(|i| a[i + 1].clone())
}

fn argmax(v: &[f32]) -> u32 {
    let mut b = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[b] {
            b = i;
        }
    }
    b as u32
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&a[1]);
    let hay: Vec<u32> = std::fs::read(&a[2])?
        .chunks(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let list = |f: &str, d: &str| -> Vec<String> {
        flag(&a, f)
            .unwrap_or(d.into())
            .split(',')
            .map(String::from)
            .collect()
    };
    let ctxs: Vec<usize> = list("--ctx", "2048")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    let depths: Vec<f64> = list("--depths", "0.1,0.3,0.5,0.7,0.9")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    let modes = list("--modes", "bf16");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let enc = |s: &str| -> Vec<u32> { tok.encode(s, false).unwrap().get_ids().to_vec() };
    let model = Model::load(
        MetalDevice::new().unwrap(),
        dir,
        ctxs.iter().max().unwrap() + 64,
        tang_llm::model::Dtype::Bf16,
    )?;
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    // fixed cases per ctx, shared by every mode
    let mut cases = Vec::new();
    for &ctx in &ctxs {
        for &dep in &depths {
            let city = CITIES[cases.len() % CITIES.len()];
            let num = 1_000_000 + next() % 9_000_000;
            let nd = enc(&format!(" The special magic number for {city} is {num}. "));
            let q = enc(&format!(
                "\nWhat is the special magic number for {city}? The special magic number for {city} is"
            ));
            let ans = enc(&format!(" {num}"));
            let off = (next() as usize) % (hay.len() - ctx);
            let h = &hay[off..off + ctx - nd.len()];
            let pos = (dep * h.len() as f64) as usize;
            let mut c = h[..pos].to_vec();
            c.extend_from_slice(&nd);
            c.extend_from_slice(&h[pos..]);
            cases.push((ctx, dep, c, q, ans));
        }
    }
    for m in &modes {
        for &ctx in &ctxs {
            let (mut ok, mut n, mut bpv) = (0, 0, 16.0);
            for (cl, dep, c, q, ans) in &cases {
                if *cl != ctx {
                    continue;
                }
                *model.kvcomp.lock().unwrap() = None;
                let mut cache = model.new_cache();
                model.forward(c, &mut cache, false)?;
                if m != "bf16" {
                    let (base, opts) = KvOpts::split(m);
                    let kc =
                        model.compress_kv_with(&cache, KvMode::parse(base).expect("mode"), opts);
                    bpv = 8.0 * kc.bytes as f64 / kc.values as f64;
                    *model.kvcomp.lock().unwrap() = Some(kc);
                }
                let mut feed = q.clone();
                feed.extend_from_slice(ans);
                let mut hit = true;
                for (i, &t) in feed.iter().enumerate() {
                    let lg = model.forward(&[t], &mut cache, false)?;
                    let j = i + 1; // position of the token this step predicts
                    if j >= q.len() && j < feed.len() && argmax(&lg) != feed[j] {
                        hit = false;
                    }
                }
                ok += hit as usize;
                n += 1;
                eprintln!("{m} ctx={ctx} depth={dep} correct={hit}");
            }
            println!(
                "{{\"ctx\":{ctx},\"mode\":\"{m}\",\"bpv\":{bpv:.3},\"correct\":{ok},\"cases\":{n}}}"
            );
        }
    }
    Ok(())
}
