//! The port against the python compiler: allocation, layout and output dims of the bundled
//! store program at Qwen3.5-4B shapes, and (ignored, needs the checkpoint) every emitted tensor
//! of layers 0..11 against the qwenputer Filesystem checkpoint.

use std::collections::HashMap;
use std::path::PathBuf;

use tang_chip::emit::TensorRef;
use tang_chip::{emit, pick_hidden, store_program, ChipShape};

#[test]
fn chip_alloc_matches_python() {
    let g = store_program().unwrap();
    assert_eq!(
        (g.dims.len(), g.heads(), g.ffn_units(), g.n_layers()),
        (3753, 164, 47532, 12)
    );
    let a = tang_chip::allocate(&g, &g.keep, g.target.intermediate);
    let py = g.python_alloc.as_ref().unwrap();
    assert_eq!(a.n_slots, py.n_slots);
    let slots: HashMap<u32, usize> = py
        .slots
        .iter()
        .map(|(k, &v)| (k.parse().unwrap(), v))
        .collect();
    assert_eq!(a.slots, slots);
    let sub: HashMap<u32, u32> = py
        .subtract
        .iter()
        .map(|(k, &v)| (k.parse().unwrap(), v))
        .collect();
    assert_eq!(a.subtract, sub);
    let clears: Vec<(u32, Vec<u32>)> = py
        .clears
        .iter()
        .map(|(k, v)| (k.parse().unwrap(), v.clone()))
        .collect();
    let mut ours: Vec<(u32, Vec<u32>)> = a.clears.clone().into_iter().collect();
    let mut theirs = clears;
    ours.sort();
    theirs.sort();
    assert_eq!(ours, theirs);
    assert_eq!(Some(&a.final_), py.final_.as_ref());
}

#[test]
fn chip_layout_matches_checkpoint_meta() {
    assert_eq!(pick_hidden(2560, 1807), (4608, 18));
    let g = store_program().unwrap();
    let c = emit(&g, &ChipShape::QWEN35_4B).unwrap();
    // chipformer.json of the Filesystem checkpoint
    assert_eq!(c.hidden, 4608);
    assert_eq!(c.code_dims, (2710..2731).collect::<Vec<_>>());
    assert_eq!((c.tau_dim, c.mu_dim), (2733, 2732));
    assert_eq!(c.id_code_dims, (3146..3164).collect::<Vec<_>>());
    // auto-sized for 4B lands on the same layout
    let auto = ChipShape {
        hidden_chip: None,
        ballast: None,
        ..ChipShape::QWEN35_4B
    };
    let c2 = emit(&g, &auto).unwrap();
    assert_eq!((c2.hidden, c2.layout.m), (4608, 18));
    let nnz: usize = c.layers.iter().map(|l| l.nnz()).sum();
    eprintln!("4B: {nnz} nonzero weights over {} layers", c.layers.len());
}

fn bf16_le(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
        .collect()
}

/// Milestone 2 parity: the emitter's layers 0..11 equal the checkpoint's, every element of every
/// tensor (nonzero blocks and zero padding alike), bitwise in bf16 (the checkpoint's dtype; every
/// emitted weight is a bf16 value, checked at emission).
#[test]
#[ignore = "needs the qwenputer checkpoint slice ($TANG_CHIP_DIR/chip.safetensors)"]
fn chip_parity_4b_weights() {
    let dir = PathBuf::from(
        std::env::var("TANG_CHIP_DIR").unwrap_or("/private/tmp/claude-501/qwp-ckpt".into()),
    );
    let file = std::fs::File::open(dir.join("chip.safetensors")).unwrap();
    let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let st = safetensors::SafeTensors::deserialize(&map).unwrap();
    let g = store_program().unwrap();
    let c = emit(&g, &ChipShape::QWEN35_4B).unwrap();
    let (mut n_t, mut n_el, mut n_nz) = (0, 0usize, 0usize);
    for (l, lw) in c.layers.iter().enumerate() {
        for (name, t) in lw.tensors() {
            let key = format!("model.layers.{l}.{name}");
            let view = st.tensor(&key).unwrap();
            assert_eq!(view.dtype(), safetensors::Dtype::BF16, "{key}");
            let want = bf16_le(view.data());
            let ours = match t {
                TensorRef::Sparse(s) => {
                    assert_eq!(view.shape(), [s.rows, s.cols], "{key}");
                    n_nz += s.nnz();
                    s.to_dense()
                }
                TensorRef::Dense(d) => d.to_vec(),
            };
            assert_eq!(ours.len(), want.len(), "{key}");
            let bad: Vec<usize> = (0..ours.len())
                .filter(|&i| ours[i].to_bits() != want[i].to_bits())
                .collect();
            assert!(
                bad.is_empty(),
                "{key}: {} elements differ, first at {}: ours {} checkpoint {}",
                bad.len(),
                bad[0],
                ours[bad[0]],
                want[bad[0]]
            );
            n_t += 1;
            n_el += ours.len();
        }
    }
    eprintln!("{n_t} tensors, {n_el} elements bitwise equal ({n_nz} nonzero weights)");
}
