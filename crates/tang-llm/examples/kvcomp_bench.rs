//! Decode speed and quality with a compressed KV cache (see `tang_llm::kvcomp`).
//!
//!     cargo run --release -p tang-llm --features metal --example kvcomp_bench -- \
//!         <model-dir> <ids.u32> --ctx 2048,8192 --n 64 --modes bf16,kivi:4:32,tkv:0.07
//!
//! Per (ctx, mode): prefill `ctx` tokens of the id file, compress the cache (whole 128-token
//! blocks), then teacher-force the next `n` tokens one decode step at a time. Reports decode
//! tok/s and ms/step (after 4 warm-up steps), NLL of the `n` continuation tokens, and the
//! compressed size of the covered rows in bits per value.
use std::time::Instant;
use tang_compute::MetalDevice;
use tang_llm::kvcomp::KvMode;
use tang_llm::model::Model;

fn flag(a: &[String], f: &str) -> Option<String> {
    a.iter().position(|x| x == f).map(|i| a[i + 1].clone())
}

fn nll(logits: &[f32], tgt: u32) -> f64 {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = logits.iter().map(|&v| (v as f64 - m).exp()).sum();
    m + z.ln() - logits[tgt as usize] as f64
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ids: Vec<u32> = std::fs::read(&a[2])?.chunks(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let ctxs: Vec<usize> = flag(&a, "--ctx").unwrap_or("2048".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let n: usize = flag(&a, "--n").map_or(64, |v| v.parse().unwrap());
    let modes: Vec<String> = flag(&a, "--modes").unwrap_or("bf16".into()).split(',').map(String::from).collect();
    let maxc = ctxs.iter().max().unwrap() + n + 8;
    let model = Model::load(MetalDevice::new().unwrap(), std::path::Path::new(&a[1]), maxc, tang_llm::model::Dtype::Bf16)?;
    for &ctx in &ctxs {
        for m in &modes {
            *model.kvcomp.lock().unwrap() = None;
            let mut cache = model.new_cache();
            let mut lg = model.forward(&ids[..ctx], &mut cache, false)?;
            let mut bpv = 16.0;
            let mut ctime = 0.0;
            if m != "bf16" {
                let mode = KvMode::parse(m).expect("mode");
                let t = Instant::now();
                let kc = model.compress_kv(&cache, mode);
                ctime = t.elapsed().as_secs_f64();
                bpv = 8.0 * kc.bytes as f64 / kc.values as f64;
                *model.kvcomp.lock().unwrap() = Some(kc);
            }
            let mut tot = 0.0;
            let mut times = Vec::new();
            for i in 0..n {
                tot += nll(&lg, ids[ctx + i]);
                let t = Instant::now();
                lg = model.forward(&ids[ctx + i..ctx + i + 1], &mut cache, false)?;
                times.push(t.elapsed().as_secs_f64());
            }
            let st = &times[4.min(n - 1)..];
            let ms = 1e3 * st.iter().sum::<f64>() / st.len() as f64;
            println!(
                "{{\"ctx\":{ctx},\"mode\":\"{m}\",\"bpv\":{bpv:.3},\"ms_per_step\":{ms:.3},\"tok_s\":{:.2},\"nll\":{:.4},\"compress_s\":{ctime:.2}}}",
                1e3 / ms,
                tot / n as f64
            );
        }
    }
    Ok(())
}
