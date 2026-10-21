//! An extension that loads a model's weights itself (`ext::Loader`) and runs it through its
//! step override: the shell model holds no weights, its KV pool has the width the loader asked
//! for, and its logits are those of the same checkpoint loaded by tang, through `forward_batch`
//! and `forward` alike. On the CPU (its own test binary: the loader is registered globally).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tang_compute::ext::Entry;
use tang_compute::CpuDevice;
use tang_llm::ext::{self, External, Loader, Match, Step};
use tang_llm::model::Segment;
use tang_llm::{Dtype, Model};

/// Set while the loader loads its own copy, so that load goes through tang.
static INNER: AtomicBool = AtomicBool::new(false);

fn register() {
    ext::register(Entry {
        name: "test-loader",
        matches: Match::any().model("qwen3"),
        priority: 1,
        imp: Loader::<CpuDevice>(Arc::new(|_dev, dir, cfg, max_ctx| {
            if INNER.swap(true, Ordering::SeqCst) {
                return None;
            }
            let m = Model::load(CpuDevice::new(), dir, max_ctx, Dtype::F32);
            INNER.store(false, Ordering::SeqCst);
            Some(m.map(|m| External {
                kv_dim: cfg.kv_dim(),
                state: Arc::new(m),
            }))
        })),
    });
    // The step runs the loader's copy over the shell's caches (same layers and KV width).
    ext::register(Entry {
        name: "test-external-step",
        matches: Match::any().model("qwen3"),
        priority: 1,
        imp: Step::<CpuDevice>(Arc::new(|m: &Model<CpuDevice>, segs| {
            let inner = m.external()?.state.downcast_ref::<Model<CpuDevice>>()?;
            Some(inner.forward_batch(segs))
        })),
    });
}

#[test]
fn external_weights_match_tang_load() {
    let dir = std::env::temp_dir().join(format!("tang-external-{}", std::process::id()));
    ext::parity::tiny_checkpoint(&dir).unwrap();
    // tang's own load first (nothing registered yet), as the reference.
    let reference = Model::load(CpuDevice::new(), &dir, 512, Dtype::F32).unwrap();
    register();
    let shell = Model::load(CpuDevice::new(), &dir, 512, Dtype::F32).unwrap();
    assert!(shell.external().is_some());
    assert_eq!(shell.n_layers(), 0);
    assert_eq!(shell.row_elems(), reference.row_elems());

    let prompt = [1u32, 5, 9, 2, 7];
    let run = |m: &Model<CpuDevice>| {
        let mut c = m.new_cache();
        let mut out = m.forward(&prompt, &mut c, false).unwrap();
        for t in [3u32, 4, 8] {
            let mut segs = [Segment {
                tokens: &[t],
                cache: &mut c,
                all: false,
            }];
            out.extend(m.forward_batch(&mut segs).unwrap().remove(0));
        }
        out
    };
    assert_eq!(run(&shell), run(&reference));
}
