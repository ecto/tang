//! A compiler from a circuit IR to Gated DeltaNet decoder-layer weights: a Rust port of the
//! lowering of qwenputer (<https://github.com/divagr18/qwenputer>, branch `rv-store`,
//! `experimental/riscv/chipformer/q35`; MIT, Copyright (c) 2026 Divyansh Agrawal; see NOTICE).
//!
//! - [`ir`]: the scheduled circuit graph (`q35/ir.py`), read from a JSON export.
//! - [`alloc`]: residual slot sharing (`q35/emit.py` `allocate`).
//! - [`emit`]: the lowering to Qwen3.5 `linear_attention` layer weights for any
//!   [`ChipShape`](emit::ChipShape) (`q35/emit.py` `emit_layers`, `embedding_rows`), as sparse
//!   matrices.
//! - [`store_program`]: the Filesystem checkpoint's circuit (the RV32 core plus the object store:
//!   put/get/list/delete/append), bundled.
//!
//! See `docs/design/gdn-chip.md` ("Compiler").

pub mod alloc;
pub mod emit;
pub mod ir;

pub use alloc::{allocate, Alloc};
pub use emit::{emit, pick_hidden, ChipShape, Compiled, LayerWeights, Layout, Sparse};
pub use ir::Graph;

/// The circuit of qwenputer's Filesystem checkpoint (`core.build(T4B, Text, store.Config)`),
/// exported by `scripts/chip/export_ir.py`: 12 layers, 164 heads, 47,532 FFN units.
pub fn store_program() -> anyhow::Result<Graph> {
    Graph::from_json_gz(include_bytes!("../data/store_ir.json.gz"))
}

/// The put/get/list reproduction trace as IR token features (`scripts/chip/export_fixture.py`).
pub const TRACE_PUT_GET_LIST: &[u8] = include_bytes!("../data/trace_put_get_list.json.gz");

/// gunzip a bundled file.
pub fn gunzip(b: &[u8]) -> anyhow::Result<String> {
    let mut s = String::new();
    std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(b), &mut s)?;
    Ok(s)
}
