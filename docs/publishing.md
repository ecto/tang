# Publishing tang

The crates.io idiom this workspace follows: **public crates depend on each other
by version**, git dependencies exist only for crates that are not published yet,
and a developer who wants to build against a local sibling checkout does it in an
untracked `.cargo/config.toml` `[patch.crates-io]` — never in a committed
manifest.

## State on crates.io

| crate | published | this release |
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

## Versions

The numbers above are in the manifests. `tang` carries its own `version` (it
always has); `tang-la`, `tang-ad` and `tang-expr` now carry theirs too, because
they moved off the workspace number to 0.1.1. Everything else still takes
`[workspace.package] version`, which stays 0.1.0.

Every intra-workspace dependency is `{ path = "...", version = "..." }` in
`[workspace.dependencies]`: `path` for the local build, `version` for what
crates.io records. The requirements are major-minor (`tang = "0.2"`,
`tang-* = "0.1"`), so a patch bump on a dependency does not need a matching
edit in every dependent.

## Publish order

Dependency order (the topological order of the publishable crates from `cargo
metadata`). Each publish waits 30s for the previous crate to appear on the
index. Run it with `scripts/publish.sh`, which does exactly this and stops on
the first failure:

```sh
cargo publish -p tang
cargo publish -p tang-expr
cargo publish -p tang-la
cargo publish -p tang-ad
cargo publish -p tang-gpu
cargo publish -p tang-tensor
cargo publish -p tang-3dgs
cargo publish -p tang-optim
cargo publish -p tang-train
```

Only `tang` can be verified before the release: `cargo publish --dry-run -p
tang` compiles against crates.io alone. A dry run of any dependent fails on the
new versions not being on the index yet — that is expected, and the packaging
of each is checked with `cargo package -p <crate> --no-verify` instead.

`cargo package` itself resolves against the index, so before `tang` 0.2.1 is up
five of the nine still fail, all for the same reason — a dependency that is not
published yet:

- `tang-3dgs`, `tang-train` — `tang-gpu` / `tang-tensor` are first publishes.
- `tang-la`, `tang-tensor`, `tang-optim` — they enable `tang/algebraic`, which
  exists in 0.2.1 but not in the 0.2.0 on the index. The requirement `^0.2` is
  still right: cargo refuses 0.2.0 for a crate that needs the feature rather
  than resolving to it, so once 0.2.1 is up the feature pins the resolution.

Each one packages cleanly as soon as the crate ahead of it in the order lands,
which is what `scripts/publish.sh` walks.

## Next release: tang-expr 0.2.0, tang-compile 0.1.0, cram-codec 0.1.0

| crate | crates.io now | this release | depends on (tang-*) |
| --- | --- | --- | --- |
| `tang-expr` | 0.1.0, 0.1.1 | **0.2.0** | `tang` 0.2 (0.2.1 on the index) |
| `tang-compile` | — | 0.1.0 (first publish) | `tang` 0.2, `tang-expr` 0.2 |
| `cram-codec` | — | 0.1.0 (first publish) | `tang-la` 0.1, `tang-optim` 0.1, `tang-expr` 0.2 |

`tang-expr` goes to 0.2.0, not 0.1.2: besides the additive `int`, `reverse`
and `split` modules, the public alias `compile::CompiledMany` gained a
`Send + Sync` bound (a non-`Send` closure that fit it before no longer does)
and `abs`/`min`/`max`/`tanh` now trace to different graphs (`select`
instead of `sqrt` forms), which changes `diff`/sparsity output. In 0.x,
cargo treats the minor as the breaking digit.

`tang-la` 0.1.1, `tang-optim` 0.1.1 and `tang` 0.2.1 are already on the
index and unchanged in the parts these crates use, so they are not
republished. Order (`scripts/publish.sh`):

```sh
cargo publish -p tang-expr
cargo publish -p tang-compile
cargo publish -p cram-codec
```

`tang-expr` dry-runs against crates.io alone. The two dependents cannot
until `tang-expr` 0.2.0 is on the index; they were verified by packaging
`tang-expr` and patching crates.io to that unpacked tarball (its normalized
manifest takes `tang` from crates.io, so the graph is the real one):

```sh
cargo package -p tang-expr
cargo publish --dry-run -p tang-compile \
  --config 'patch.crates-io.tang-expr.path="target/package/tang-expr-0.2.0"'
```

(Patching to `crates/tang-expr` instead pulls a second, path-sourced `tang`
into the graph and fails with a spurious "multiple versions of `tang`"
error.)

Published crates still requiring `tang-expr` 0.1 (`tang-gpu`, `tang-3dgs`,
`tang-mesh`, …) keep resolving to 0.1.1; the workspace requirement is now
`0.2`, so their next publish takes 0.2.

tang is the root of the stack: publish it before phyz, kosm-render, vcad, kosm.

## Local development against a sibling checkout

tang has no sibling git dependencies, so it needs no `.cargo/config.toml`. Its
*consumers* do — see the matching doc in phyz, vcad and kosm.
