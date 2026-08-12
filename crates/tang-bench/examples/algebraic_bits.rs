//! Prints exact bit patterns of every reduction touched by the `algebraic`
//! work, so the feature-off build can be diffed against the pre-change code.
//! Uses only public API, so it runs unmodified on either revision.

use tang_infer::{Sampler, SamplingConfig};
use tang_tensor::{Shape, Tensor};
use tang_train::{
    cross_entropy_loss, cross_entropy_loss_grad, huber_loss, mse_loss, sequence_cross_entropy,
    LayerNorm, Module, RMSNorm,
};

fn fill<S: tang::Scalar>(n: usize, seed: u64) -> Vec<S> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            S::from_f64(u * 2.0 - 1.0)
        })
        .collect()
}

fn p32(name: &str, v: f32) {
    println!("{name:<44} {:08x}", v.to_bits());
}
fn p64(name: &str, v: f64) {
    println!("{name:<44} {:016x}", v.to_bits());
}
fn pt32(name: &str, t: &Tensor<f32>) {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in t.data() {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    println!("{name:<44} {h:016x}");
}

fn main() {
    for &n in &[1usize, 2, 7, 1024, 4095, 65536] {
        let a32 = Tensor::<f32>::new(fill(n, 7), Shape::from_slice(&[n]));
        let b32 = Tensor::<f32>::new(fill(n, 11), Shape::from_slice(&[n]));
        let a64 = Tensor::<f64>::new(fill(n, 7), Shape::from_slice(&[n]));
        let b64 = Tensor::<f64>::new(fill(n, 11), Shape::from_slice(&[n]));
        p32(&format!("sum f32 n={n}"), a32.sum());
        p64(&format!("sum f64 n={n}"), a64.sum());
        p32(&format!("mean f32 n={n}"), a32.mean());
        p32(&format!("mse f32 n={n}"), mse_loss(&a32, &b32));
        p64(&format!("mse f64 n={n}"), mse_loss(&a64, &b64));
        p32(&format!("huber f32 n={n}"), huber_loss(&a32, &b32, 0.5));
        pt32(&format!("softmax1d f32 n={n}"), &a32.softmax(0));
    }

    for &(r, c) in &[(1usize, 1usize), (3, 5), (8, 576), (16, 4096)] {
        let n = r * c;
        let x32 = Tensor::<f32>::new(fill(n, 17), Shape::from_slice(&[r, c]));
        let g32 = Tensor::<f32>::new(fill(n, 19), Shape::from_slice(&[r, c]));
        pt32(&format!("softmax2d {r}x{c}"), &x32.softmax(1));
        pt32(&format!("sum_axis1 {r}x{c}"), &x32.sum_axis(1));
        pt32(&format!("sum_axis0 {r}x{c}"), &x32.sum_axis(0));

        let mut rms = RMSNorm::<f32>::new(c);
        let out = rms.forward(&x32);
        pt32(&format!("rmsnorm fwd {r}x{c}"), &out);
        pt32(&format!("rmsnorm bwd {r}x{c}"), &rms.backward(&g32));

        let mut ln = LayerNorm::<f32>::new(c);
        let out = ln.forward(&x32);
        pt32(&format!("layernorm fwd {r}x{c}"), &out);
        pt32(&format!("layernorm bwd {r}x{c}"), &ln.backward(&g32));

        let targets = Tensor::<f32>::from_fn(Shape::from_slice(&[r]), |i| (i[0] % c) as f32);
        p32(
            &format!("cross_entropy {r}x{c}"),
            cross_entropy_loss(&x32, &targets),
        );
        pt32(
            &format!("cross_entropy_grad {r}x{c}"),
            &cross_entropy_loss_grad(&x32, &targets),
        );
        p32(
            &format!("seq_cross_entropy {r}x{c}"),
            sequence_cross_entropy(&x32, &targets, 0),
        );
    }

    for &vocab in &[7usize, 1024, 32000] {
        let logits = Tensor::<f32>::new(fill(vocab, 31), Shape::from_slice(&[vocab]));
        for &(temp, top_p, top_k) in &[(1.0, 1.0, 0usize), (0.7, 0.9, 40)] {
            let mut s = Sampler::with_seed(
                SamplingConfig {
                    temperature: temp,
                    top_p,
                    top_k,
                    ..Default::default()
                },
                42,
            );
            println!(
                "{:<44} {}",
                format!("sample vocab={vocab} t={temp} p={top_p}"),
                s.sample(&logits, &[1, 2, 3])
            );
        }
    }
}
