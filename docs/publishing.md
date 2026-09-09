# Publishing tang

The crates.io idiom this workspace follows: **public crates depend on each other
by version**, git dependencies exist only for crates that are not published yet,
and a developer who wants to build against a local sibling checkout does it in an
untracked `.cargo/config.toml` `[patch.crates-io]` — never in a committed
manifest.

## State on crates.io

| crate | published | plan |
| --- | --- | --- |
| `tang` | 0.2.0 | **0.2.1** |
| `tang-la` | 0.1.0 | 0.1.1 |
| `tang-ad` | 0.1.0 | 0.1.1 |
| `tang-expr` | 0.1.0 | 0.1.1 |
| `tang-tensor` | — | 0.1.0 (first publish) |
| `tang-optim` | — | 0.1.0 (first publish) |
| `tang-gpu` | — | 0.1.0 (first publish) |
| `tang-train` | — | 0.1.0 (first publish) |
| `tang-3dgs` | — | 0.1.0 (first publish) |

`tang` goes to 0.2.1, not 0.3.0: everything between the 0.2.0 commit (`f82c629`,
"consolidate all subcrates into tang 0.2.0 with feature flags") and `c9ef210` is
additive — the opt-in `algebraic` feature, SIMD stats moments, row-major GEMM
kernels, the symmetric-eigensolver convergence fix. No public item was removed or
had its signature narrowed.

The remaining library crates (`tang-sparse`, `tang-safetensors`, `tang-hub`,
`tang-infer`, `tang-onnx`, `tang-compile`, `tang-compute`, `tang-mesh`,
`tang-holo`, `tang-sheffer`) carry full crates.io metadata and are publishable at
0.1.0 whenever they are wanted; nothing downstream needs them yet.

## Not published

| crate | why |
| --- | --- |
| `tang-bench` | benchmarks and their fixtures, not an API anyone depends on |
| `tang-site-wasm` | a cdylib for the tang.rs site, not a library crate |
| `tang-sheffer-wasm` | a cdylib for the sheffer search demo, not a library crate |

## Versions are not bumped in this branch

tang has no release tags and no dedicated release commit — 0.2.0 was bumped
inline inside a feature commit. So this branch writes the plan and leaves the
numbers alone. Bump `[workspace.package] version` (and `crates/tang/Cargo.toml`'s
own `version`) on the release commit, then publish.

## Publish order

Dependency order within the workspace. Each `publish` waits for the previous
crate to appear on the index.

```sh
cargo publish -p tang
cargo publish -p tang-la
cargo publish -p tang-ad
cargo publish -p tang-expr
cargo publish -p tang-tensor
cargo publish -p tang-optim
cargo publish -p tang-gpu
cargo publish -p tang-train
cargo publish -p tang-3dgs
```

tang is the root of the stack: publish it before phyz, kosm-render, vcad, kosm.

## Local development against a sibling checkout

tang has no sibling git dependencies, so it needs no `.cargo/config.toml`. Its
*consumers* do — see the matching doc in phyz, vcad and kosm.
