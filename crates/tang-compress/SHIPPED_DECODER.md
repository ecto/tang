# Decoder in the stream (TCS1)

A tang-compress TCG3 stream can carry the program that decodes it. The
receiver checks the program, compiles it to WGSL and to the CPU interpreter,
and caches the result. Plain TCG1 and TCG3 streams decode exactly as before.

## What the program covers

The coding path is the part of TCG3 decoding that is specific to the
entropy coder. It is now built as `tang-expr` integer graphs in
`src/irdec.rs`:

- **Static residual decoder.**
  - Inputs: rANS state, word position, lane bound, and the context
    (`sig`, `e`, `cls`, `sctx`, `off_init`).
  - Steps: significance bit, the magnitude unary loop over the `MAG0` models,
    the Elias-gamma escape, two modelled low bits, raw bits in chunks of at
    most 16, then the sign.
  - Outputs: state, position, and the residual.
- **Learned residual decoder.**
  - Inputs: the same lane state, plus the net's 5 outputs.
  - Steps: σ-table mixture weight, the 2-logistic integer CDF (`zval` and
    `sig16` read tables from `mt`), the zero flag, the sign, the exp-Golomb
    bucket with carried tails, and bisection for the mantissa.
- **MLP (optional).** The learned model's forward pass, with the weights as
  graph constants (`Net32::graph`).

The rest of decoding stays fixed and is not part of the program:
neighbourhood contexts, prediction, dequantization, the lifting transform
and outliers.

These graphs are also the built-in decoder. By default the WGSL kernel's
`RES_BEGIN…RES_END` region is generated from them. `TC_GPU_HANDRES=1`
restores the hand-written version for A/B runs. On the CPU, native Rust
remains the default; `TC_CPU_IRRES=1` routes decoding through the
interpreter instead.

The graphs are pure, so an `if` becomes a predicate. A decision taken under
a false predicate still computes its rANS step, but keeps the old state; its
word fetch either reads in bounds or returns 0. Each loop folds its predicate
into its condition, so it runs exactly as many iterations as the
hand-written loop.

## Container format

```
TCS1  varint(len)  program[len]  [pad to even]  TCG3 stream
program = "TCDP" 0x01 flags { varint(len) TIR1-graph }*
flags: bit0 static residual, bit1 learned residual, bit2 MLP
```

- **Missing residual sections.** If a residual section is absent, the
  receiver uses its built-in graph for that role.
- **Alignment.** The pad byte keeps the inner stream on a 16-bit word
  boundary. `parse3` shifts every lane, tile and block offset by the header
  length, so the existing CPU and GPU decoders index the outer buffer
  unchanged.
- **Learned streams with only a fingerprint.** Normally the decoder must
  find a registered model matching that fingerprint. If the stream ships its
  MLP, an unregistered fingerprint gets a placeholder net instead, and the
  shipped graph runs. Such a stream needs no model registry
  (`shipped_mlp_needs_no_registry`).

### Graph wire format (`TIR1`, in tang-expr `src/int/wire.rs`)

- **Loop table first.** For each loop: its parent, its trip cap, the number
  of states, the init ids, the condition id, and the next-state ids.
- **Then the nodes in id order.** Each node is a tag byte followed by its
  fields. Operands are encoded as `self − operand`, which is always ≥ 1.
- **Then the output ids.**

Buffers are not named on the wire. The reader supplies the names.

## Verifier rules

The verifier is `IntGraph::from_bytes` plus the role ABIs in
`shipdec::compile`. It rejects:

1. Truncation, trailing bytes, malformed varints, unknown tags or ops.
2. Any operand that is not an earlier node. This makes the graph a DAG by
   construction.
3. Type errors. The checks are the builder's own: arithmetic on bool,
   mismatched operands, a non-`u32` load index, and so on.
4. Scope errors:
   - a node that mixes states of unrelated loops;
   - a loop body that uses a sibling loop's values;
   - a loop init that does not come from the enclosing scope.
5. Malformed loops:
   - a missing or duplicate state node;
   - a loop result that appears before the loop's body;
   - a loop whose parent is not an earlier loop.
6. ABI mismatches. Params must match the role's count and types, buffers
   must match exactly (`mt`, `stream`; none for the MLP), and outputs must be
   top-level values of the role's types.
7. Resource limits:

   | limit | residual roles | MLP |
   |---|---|---|
   | nodes | 2^14 | 2^16 |
   | loops | 16 | 0 |
   | nesting depth | 2 | – |
   | trip cap per loop | 64 | – |
   | worst-case work (Σ nodes × enclosing caps) | 2^16 | 2^16 |
   | program size | 1 MiB | 1 MiB |

What the graph can do is fixed by its op set. That set has integer
arithmetic only; there is no float, store, atomic, barrier, call or I/O.
Loads are read-only, and both backends bounds-check them so that an
out-of-bounds load reads 0: the interpreter does it, and
`to_wgsl_body_guarded` guards with `arrayLength`. Every loop carries a trip
cap, and both backends count trips the same way. A verified graph is
therefore a total, terminating function of its parameters and the two
buffers.

## Threat model

The attacker controls the whole stream, including the program.

- **Memory safety.** Program bytes never index host memory directly:
  - registers are bounded by the node count;
  - loads are clamped to read 0 out of bounds;
  - buffer names come from the host, so no attacker text reaches the WGSL
    source.

  Every identifier the lowering emits is generated by the host (`t{id}`,
  `s{l}_{k}`, `c{l}`), and every literal is a typed integer.
- **Denial of service.**
  - Per residual, work is capped at 2^16 node evaluations.
  - Compile cost is bounded by the node limit.
  - Inlined expressions nest at most 32 deep, because the lowerings recurse
    on them.

  A hostile stream can still make one decode slow, by up to about 2^16× per
  coefficient. A host that decodes untrusted streams should apply its own
  time budget or GPU watchdog. A budget that tracks the stream's coefficient
  count would be a good follow-up.
- **Wrong output.** A valid but different program decodes to different
  numbers. That is the point of shipping a decoder, and it is not a safety
  issue. The integrity of the field rests on the stream as a whole: sign or
  hash the container if that matters. The built-in decoder can always be
  forced by stripping the TCS1 header.
- **Cache poisoning.** The cache is keyed by FNV-1a 64 of the program bytes,
  and a hit is confirmed by comparing the full bytes. A hash collision
  therefore never selects another program's compiled code. GPU pipelines are
  keyed by (net, program hash, program size) inside one `GpuDecoder`. That
  key does not compare bytes. A crafted FNV collision with the same size
  could reuse another program's pipeline, which yields wrong output only;
  hashing with BLAKE3 would close this.
- **Cross-backend divergence.** Byte-identical output across CPU and GPU is
  guaranteed for in-bounds programs. A hostile program that reads out of
  bounds also gets 0 on both backends. Buffer lengths differ between them
  (the GPU `mt` holds the whole metadata block), however, so an
  out-of-bounds index on one backend may be in bounds on the other. Such a
  program may decode differently on CPU and GPU, but it is still safe.

## Results

The gate and the NYX numbers are in the session notes
(`shipped-decoder.md` in the shared notes folder) and in the commit message.
