//! The extension point end to end: renamed copies of built-in ops (and of the whole step)
//! registered, resolved at load, run on the hot path, and held to the parity harness.
//! Runs on the CPU, and on Metal or CUDA when built with those features.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tang_compute::ext::Entry;
use tang_compute::ComputeDevice;
use tang_llm::ext::{self, parity, KvAttentionPaged, Linear, Match, RmsNorm, Step};
use tang_llm::{Dtype, Model};

#[derive(Default)]
struct Calls {
    linear: AtomicUsize,
    norm: AtomicUsize,
    attention: AtomicUsize,
    step: AtomicUsize,
}

/// Copies of the built-in kernels under other names, counting their calls. The GEMV copy
/// declines more than 8 rows, as a real decode kernel would.
fn register<D: ComputeDevice>(arch: &str, calls: &Arc<Calls>) {
    let only = || Match::any().arch(arch).model("qwen3");
    let c = calls.clone();
    ext::register(Entry {
        name: "copy-gemv",
        matches: only(),
        priority: 1,
        imp: Linear::<D>(Arc::new(move |dev, x, w, m, k, n| {
            if m > 8 {
                return None;
            }
            c.linear.fetch_add(1, Ordering::Relaxed);
            Some(dev.linear(x, w, m, k, n))
        })),
    });
    let c = calls.clone();
    ext::register(Entry {
        name: "copy-rms-norm",
        matches: only(),
        priority: 1,
        imp: RmsNorm::<D>(Arc::new(move |dev, x, w, rows, dim, eps| {
            c.norm.fetch_add(1, Ordering::Relaxed);
            Some(dev.rms_norm(x, w, rows, dim, eps))
        })),
    });
    let c = calls.clone();
    ext::register(Entry {
        name: "copy-attention",
        matches: only(),
        priority: 1,
        imp: KvAttentionPaged::<D>(Arc::new(move |dev, a| {
            c.attention.fetch_add(1, Ordering::Relaxed);
            Some(dev.kv_attention_paged(
                a.q, a.k_pool, a.v_pool, a.pages, a.start, a.rows, a.heads, a.window, a.causal,
            ))
        })),
    });
    let c = calls.clone();
    ext::register(Entry {
        name: "copy-step",
        matches: only(),
        priority: 1,
        imp: Step::<D>(Arc::new(move |m: &Model<D>, segs| {
            // Decode-only, like a persistent megakernel would be.
            if segs.iter().any(|g| g.tokens.len() != 1) {
                return None;
            }
            c.step.fetch_add(1, Ordering::Relaxed);
            Some(m.forward_batch_builtin(segs))
        })),
    });
}

fn run<D: ComputeDevice>(make: impl Fn() -> D, dtype: Dtype) {
    let arch = make().arch();
    let dir = std::env::temp_dir().join(format!("tang-ext-{arch}-{}", std::process::id()));
    parity::tiny_checkpoint(&dir).unwrap();

    // Nothing registered: nothing resolves.
    let m = Model::load(make(), &dir, 256, dtype).unwrap();
    assert!(m.ext().names().is_empty());
    drop(m);

    let calls = Arc::new(Calls::default());
    register::<D>(&arch, &calls);
    let mut m = Model::load(make(), &dir, 256, dtype).unwrap();
    assert_eq!(m.ext().target.arch, arch);
    assert_eq!(
        m.ext().names(),
        ["copy-gemv", "copy-rms-norm", "copy-attention", "copy-step"]
    );

    let ops = &m.ext().ops;
    let lin = ops.linear.as_ref().unwrap().imp.clone();
    let norm = ops.rms_norm.as_ref().unwrap().imp.clone();
    let cases = [(1, 64, 128), (4, 64, 256), (32, 128, 64)];
    assert_eq!(parity::linear(&m.dev, &lin, &cases, 0.0).unwrap(), 0.0);
    assert_eq!(
        parity::rms_norm(&m.dev, &norm, &[(1, 64), (5, 128)], 0.0).unwrap(),
        0.0
    );

    // Two prompts of different lengths, then batched decode: prefill exercises the op slots
    // (with GEMV declining the long rows), decode the step override.
    let prompts: [&[u32]; 2] = [&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], &[42, 7, 99]];
    let err = parity::model(&mut m, &prompts, 6, 0.0).unwrap();
    assert_eq!(err, 0.0);
    for (name, n) in [
        ("linear", &calls.linear),
        ("rms_norm", &calls.norm),
        ("attention", &calls.attention),
        ("step", &calls.step),
    ] {
        assert!(n.load(Ordering::Relaxed) > 0, "{name} override never ran");
    }
    // The harness puts the extensions back.
    assert_eq!(m.ext().names().len(), 4);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cpu_copies_pass_parity() {
    run(tang_compute::CpuDevice::new, Dtype::F32);
}

#[cfg(feature = "metal")]
#[test]
fn metal_copies_pass_parity() {
    run(|| tang_compute::MetalDevice::new().unwrap(), Dtype::Bf16);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_copies_pass_parity() {
    run(
        || tang_compute::CudaComputeDevice::new().unwrap(),
        Dtype::Bf16,
    );
}
