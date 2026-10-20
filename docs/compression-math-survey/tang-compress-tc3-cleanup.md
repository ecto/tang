# tang-compress Phase 3 cleanup: three Phase 2 debts

Branch `claude/tc3-cleanup-126408` (from `claude/tang-compress-qtt`). Builds, tests and
timing ran on mew (RTX 3090, Vulkan) after the Mac crash, except where marked M4.

## (a) TCG3 GPU kernel speed, NYX 512³, 80 dB, 8 lanes, block 32

Kernel throughput (GB/s of f32 output), best of 7, IR-generated residual decoders
(the default). All streams byte-identical to the CPU decode.

| stream | M4, quiet (static ≥ 1.76) | 3090, shared, round 1 / 2 |
|---|---|---|
| static | 1.76 / 1.90 | 1.19 / 1.00 |
| pre3v4_16, no APM | 0.97 / 0.97 | 0.82 / 0.73 |
| pre3v4_16, APM 10,14,15 | 0.83 / 0.96 | 0.78 / 0.73 |
| pre3v4_32x16, no APM | 0.74 / 0.81 | 0.78 / 0.78 |
| pre3v4_32x16, APM 10,14,15 | 0.70 / 0.77 | 0.75 / 0.74 |

- **1 GB/s gate: NOT met.** Best adaptive 16-hidden number is 0.96 GB/s on a quiet M4
  (one window where the vcad app had let go; static ran at the Track A2 quiet level), so
  the APM layer costs ~1–14% and the 16-hidden learned path sits just under the gate
  with or without it. 32×16 is 0.77.
- **3090 load:** the GPU was never idle (100% utilization, 15.4–20.6 GB used by
  tang-llm ×2, broca, the images training and ffmpeg). Our process stayed under 2 GB on
  cores 4,5,7–11. Absolute 3090 numbers are contended; static at ~1.0–1.2 shows it.
- Note: the older `bench/out/trackA2c/*.tcg3` streams decode at ~0.2 GB/s even on a quiet
  M4; they are not the configuration above (re-encoded with `--l3 8 --bricks 32`).
- Next: the A2 follow-ups (packed i16 stretch table in workgroup memory, no APM on later
  mantissa bits) aim at the last ~4%.

Reproduce (streams in `bench/out/tc3a`):
```
gpubench encode3 nyx-temperature.f32 512x512x512 --bricks 32 --targets p80 --l3 8 --model3 pretrained:pre3v4_16.tcm --out S.tcg3   # TC_AD=0,0,0 for no APM
gpubench gpu3 S.tcg3 pretrained:pre3v4_16.tcm --reps 7      # TC_GPU_HANDRES=1: hand-written residual decoders
```

## (b) Adaptive APM layer in the integer IR

- **tang-expr:** scratch buffers (`buffer_rw`), token-ordered `load_rw` / `store`
  effects run in the one shared schedule order by every backend (interpreter, WGSL,
  Rust; never inlined or moved), out-of-range loads read 0 and stores drop; wire tags
  10–12 and `from_bytes_rw` (scratch ABI from the host); `UnOp::Clz`.
- **tang-compress:** `irdec::learned_ad_graph` is `lmodel3::Adapt` (stretch lookup,
  2-bin interpolation, mix, count-rate update of the nearer bin, packed
  `(p << 8) | n` per bin in the kernel's workgroup `adt[ad_b + i]`). Per-block reset
  stays host side. TCDP flag bit 3 ships it (`builtin_program(net, ad)`); programs
  without the section fall back to the built-in adaptive graph. The hand-path fallback
  for adaptive streams is removed from CPU (`IrBase`), `parse3` and the GPU spec
  pipelines (`TC_GPU_HANDRES=1` keeps the hand path for A/B).
- **Byte identity:** CPU interpreter vs the native coder on three APM settings, also
  through the wire (`learned_ad_graph_decodes_native_encoding`); CPU shipped vs
  built-in (`shipped_decoder_matches_builtin_on_cpu`, now with adaptive streams);
  GPU (`shipped_decoder_matches_builtin_on_gpu`, `gpu_matches_cpu`, Metal before the
  crash and Vulkan on mew); NYX 512³ above, all identical; browser gate in headless
  Chrome on the M4 before the crash: **50/50** identical on WebGPU and WASM CPU (5
  fields × p80/m1e-4 × tcg1, tcg3s, tcg3l, tcs3s, tcs3l; 20 adaptive streams, 10 of
  them shipping the adaptive graph). Not re-run on mew (no Chrome there); the decode
  code has not changed since.
- **WebGPU fix:** the APM's 17 KB workgroup array exceeded WebGPU's 16 KB default, so
  every TCG3 stream decoded to garbage in the browser since Track A2.
  `GpuDecoder` now requests the adapter's `maxComputeWorkgroupStorageSize` and fails
  cleanly when it is too small.
- **Speed (3090, interleaved, IR vs hand):** APM streams 0.78/0.73 vs 0.64/0.71
  (16) and 0.75/0.74 vs 0.63/0.65 (32×16): the IR decoder is 3–19% faster, within the
  10% bound. No-APM streams 0.82/0.73 vs 0.69/0.69 and 0.78/0.78 vs 0.68/0.66.

## (c) Interpolation codec at PSNR targets

`interp::encode_rmse_f32` / `encode_psnr_f32`: rate control of the max-abs bound per
`(α, β)` preset (secant on log SSE vs log b inside a bisection bracket, ≤ 12 encodes,
smallest passing stream wins). The stream format is unchanged (an ordinary max-abs
stream at that b), and the target is checked on the exact reconstruction. Achieved
PSNR lands within 0.05 dB of the target. It is now a portfolio candidate at every
bound (no threshold, nothing tuned).

**Seen fields** (19 = 5 originals + 14 phase-1 held-out, 30–100 dB, 146 points):
interp wins 5 points, all at 30–50 dB (cesm-FLDS 40 dB 0.80×, cesm-TS 40 dB 0.96×,
miranda-density 30 dB 0.98×, qmcpack-orb100 40/50 dB 0.99/0.98×); portfolio geomean
0.998×. Median interp/auto per target is 1.4–2.6×, so it never wins at high rate on
seen data. Cost: ~0.8–1.5 s more per PSNR encode of a 1M-value field.

**Diagnostic, not a re-gate** (fresh set; needs a new fresh set to gate):

| point | old portfolio | interp | best baseline | interp / baseline |
|---|---|---|---|---|
| hacc-vx 100 dB | 14.76 | **11.76** | SZ3 12.11 | 0.971× |
| xgc-dpot 100 dB | 12.20 | **11.42** | SPERR 11.22 | 1.018× |
| hacc-vx 90 dB | 10.53 | 10.09 | – | – |
| xgc-dpot 90 dB | 9.14 | 9.75 | – | – |

Baselines from the Phase 2 fresh table. Interp takes both 100 dB points from the
wavelet; HACC would pass and XGC stays a narrow miss.
