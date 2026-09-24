//! Are a k-token verify forward's logits the same as k one-token forwards? (Probe.)
use tang_llm::{Dtype, Model};

#[test]
fn verify_rows_match_steps() {
    let Ok(spec) = std::env::var("TANG_LLM_TEST_MODEL") else { return };
    let dir = tang_llm::resolve_model(&spec).unwrap();
    if std::env::var("CPU").is_ok() {
        return go(tang_compute::CpuDevice::new(), &dir);
    }
    go(tang_compute::MetalDevice::new().unwrap(), &dir);
}

fn go<D: tang_compute::ComputeDevice>(dev: D, dir: &std::path::Path) {
    let m = Model::load(dev, dir, 4096, Dtype::Bf16).unwrap();
    let v = m.cfg.vocab_size;
    let ids: Vec<u32> = (0..60u32).map(|i| 1000 + (i * 7919) % 30000).collect();
    for k in [2usize, 3, 4, 8, 16] {
        let mut c = m.new_cache();
        m.forward(&ids[..40], &mut c, false).unwrap();
        let batch = m.forward(&ids[40..40 + k], &mut c, true).unwrap();
        c.truncate(40);
        let mut steps = Vec::new();
        for i in 0..k {
            steps.extend(m.forward(&ids[40 + i..41 + i], &mut c, true).unwrap());
        }
        let maxd = batch.iter().zip(&steps).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let same = batch == steps;
        let am = |x: &[f32]| (0..x.len()).fold(0, |b, i| if x[i] > x[b] { i } else { b });
        let top: Vec<bool> = batch.chunks(v).zip(steps.chunks(v)).map(|(a, b)| am(a) == am(b)).collect();
        eprintln!("k={k} bitwise={same} max|d|={maxd:.5} top1 same={top:?}");
    }
}
