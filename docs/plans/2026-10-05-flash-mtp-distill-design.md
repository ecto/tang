# Recursive MTP fine-tuning

The current Flash MTP layer is reused recursively at inference, but its later draft positions
have poorer acceptance than the first. Train the shared cell on the existing completed thinking
text dumps, following FastMTP's three-depth teacher-token objective. Keep the target model frozen.

Three implementation choices were considered: train every MTP expert (large optimizer state),
train adapters and add an adapter inference path (additional serving/kernel work), or train the
existing dense block weights and export them in their existing GGUF types. Start with the third:
all dense MTP projections, normalization, hyper-connections, router and shared expert train;
512 routed experts and the main embedding/output head remain frozen. This is a constrained
fine-tuning experiment, not a claim to reproduce full FastMTP training.

Use tang-compute for matrix contractions, tang-ad for scalar nonlinear Jacobians, and tang-train
Adam for updates. A small reverse-mode tensor tape keeps recursive residual and K/V paths attached.
At depth one, cell i sees teacher cells through i. Subsequent depths see the same teacher prefix
and only their own earlier chain cells. Feed teacher token i+k, predict i+k+1, and weight the three
cross-entropies by normalized [1, beta, beta²]. Normalize loss independently by valid rows per depth.

Only prompt directories with `done` are read. Validate record size, residual size, monotonically
contiguous positions and token continuity before slicing. Keep EOS as a possible target; discard
positions at/after EOS. Split entire prompts by their stable pNNN identifier, every tenth prompt
held out. Window sampling cannot cross prompts. A frozen manifest records the completed prompt
snapshot so later collector additions do not silently change a run.

Absolute RoPE positions are retained. Windows lack earlier prompt/generated K/V context; an
initial burn-in region has no direct loss. This is an approximation requiring engine acceptance
measurements. Default full-vocabulary loss avoids silently losing labels. Restricted vocabulary
is optional and errors if a label is absent.

Clip finite gradients globally, checkpoint f32 parameters plus f64 Adam moments and RNG state,
then export a copy of the original MTP GGUF. Patch only same-sized F32/BF16/Q8 tensor payloads,
reload the export, and preserve routed experts, tokenizer metadata and the directory. Original
weights are never overwritten. The engine's default Q8 -> Q4 routed expert conversion is
reproduced for training; oracle comparison uses original Q8 weights on both paths.

Validation gates: merged-head exactness and four-cell throughput baseline; finite-difference
Jacobian checks including repeated gathers, GQA, norms and recursive residuals; data and export
checks; three-depth residual/top-token parity with the independent CPU reference; a small
real-data training run and exported-engine reload; held-out CE plus actual speculative acceptance,
window counts, tok/s and exact output comparison before selecting any checkpoint.
