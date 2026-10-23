//! KDA parity vs the HF reference (torch, /tmp fixtures written by /tmp/kda_hf_ref.py).
//! Skipped when the fixtures are absent.

use tang_llm::flash::glm5::kda::{kda_forward, KdaWeights};
use tang_llm::gguf::Gguf;

fn read_f32(p: &str) -> Option<Vec<f32>> {
    let d = std::fs::read(p).ok()?;
    Some(
        d.as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    )
}

#[test]
fn kda_matches_hf() {
    // prefer the real-model input (the normed collapsed streams); fall back to the random one
    let (x, want) = match (read_f32("/tmp/kda_realx.bin"), read_f32("/tmp/hf_kda_out_realx.bin")) {
        (Some(x), Some(w)) => (x, w),
        _ => (
            read_f32("/tmp/kda_x.bin").unwrap(),
            read_f32("/tmp/kda_out.bin").unwrap(),
        ),
    };

    let t_len = 52;
    let n_embd = 256;
    assert_eq!(x.len(), t_len * n_embd);
    let g = Gguf::open(std::path::Path::new(
        "/private/tmp/glm5-tiny/glm5-tiny-f32.gguf",
    ))
    .unwrap();
    let hp = tang_llm::flash::glm5::Hparams::from_gguf(&g).unwrap();
    let w = KdaWeights::from_gguf(&g, &hp, 0).unwrap();
    let out = kda_forward(&w, &x, t_len).unwrap();
    {
        let b: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write("/tmp/my_out.bin", b).unwrap();
    }
    let d = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let d_out = d(&out, &want);
    println!("out maxabs {d_out}");
    assert!(d_out < 1e-3, "kda out vs HF: {d_out}");
}
