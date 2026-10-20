# tang-compress portfolio codec (Track E, phase 1)

Branch `claude/tc-portfolio-716321`. **Frozen code: `6b00aff`** (committed before any held-out run). Harness: `bench/run_portfolio.py`, `bench/summarize_portfolio.py`. External-codec rows are reused unchanged from the M1 held-out run (`bench/out/heldout_*.csv` from the `claude/tang-compress-qtt` worktree, copied to `bench/out/ext/`; QTT there is row-major, the same codec the portfolio calls).

## What changed

`encode_auto_f32` / `tc --codec auto` is now a portfolio (`encode_portfolio_f32`). Same 1-byte tag, new values: `0xA0` wavelet, `0xA1` Tucker, `0xA2` QTT, `0xA3` interp. `decode_any` dispatches on the tag + magic.

| candidate | when | notes |
|---|---|---|
| wavelet (configured model; overfit in these runs) | always | unchanged |
| Tucker | PSNR/RMSE, ≥2D | unchanged |
| **interp** (new, `src/interp.rs`) | max-abs | SZ3-style multilevel interpolation: per-level linear/cubic choice by trial cost, level-wise bound `b/min(α^(ℓ−1), β)` with 4 `(α, β)` presets picked by real stream size, error-bounded quantization checked on the exact f32 reconstruction (escapes stored raw), adaptive range coding with (level, log₂ neighbour-difference, previous-symbol) contexts. Median encode 0.13 s per 1M values. |
| **QTT** row-major | any bound, **only if the pre-screen passes** | `codec::encode_f32` with `DigitOrder::RowMajor` |

**Pre-screen** (no encodes): `peakiness = range / (q99 − q01)` on a ≤65k strided sample; QTT runs only when peakiness ≥ 8. Validated on the 5 original fields only (`bench/out/portfolio_dev.csv`):

- peakiness 1.0–1.7 (Miranda, Hurricane), 4.2 (NYX temperature);
- QTT never wins on any original point (it is 1.1–15× larger than the winner) and costs 0.1–58 s per encode (58 s on NYX at 1e-2), so skipping it there is free in rate;
- the threshold 8 was set at ~2× the most heavy-tailed original field.

Dev check on the originals, max-abs, interp vs SZ3: interp is smaller at every point (e.g. Hurricane TC 1e-2: 0.0112 vs 0.0195 bpv; Miranda density 1e-4: 2.87 vs 3.40). It wins outright in the portfolio only at Hurricane TC 1e-2.

## Held-out gate (14 fields, frozen `6b00aff`)

Decoder-side measured error, real file sizes (tag byte included). PSNR points are interpolated on each codec's Pareto front, as in M1 §8.

**Gate: FAIL at 1 of 97 points.** The failing point is qmcpack-orb0 at max 1e-4: 0.3346 bpv vs QTT's 0.2909 (1.150×). The pre-screen skipped QTT there (peakiness 1.4). Post-hoc diagnostic, not part of the gate: with the screen off (`portfolio-all`), that point picks QTT at 0.2909 (1.000×), so the screen is the only reason for the miss.

The peaky-data hypothesis was half right:

- EXAFEL (peakiness 48) was caught, and both of its QTT-won points went to QTT;
- QTT also wins on QMCPACK orbital 0, which is smooth and low-TT-rank rather than peaky.

Peakiness is the wrong signal for that second case.

| point | geo-mean tc-portfolio / best external | M1 tc-auto |
|---|---|---|
| 40 dB | 0.429 (n=13) | 0.429 |
| 60 dB | 0.775 | 0.775 |
| 80 dB | 0.737 | 0.737 |
| 100 dB | 0.808 | 0.808 |
| max 1e-2 | **0.677** | 0.810 |
| max 1e-3 | **0.811** | 0.843 |
| max 1e-4 | **0.871** | 0.876 |

Win/loss/tie (±1%) vs best external:

- PSNR: 54–1–0, unchanged; the loss is Hurricane-V 60 dB at 1.014×, which is inside the gate.
- max-abs: **38–1–3**, against M1's 34–8.

The 8 former max-abs losses:

| field / point | M1 tc-auto | portfolio | pick |
|---|---|---|---|
| scale-PRES max 1e-2 | 1.489× | **0.749×** | interp |
| qmcpack-orb100 max 1e-2 | 1.211× | **0.770×** | interp |
| qmcpack-orb0 max 1e-2 | 1.113× | **0.654×** | interp |
| scale-T max 1e-2 | 1.016× | **0.949×** | interp |
| exafel-ev0 max 1e-3 | 1.313× | **1.000×** | QTT |
| exafel-ev0 max 1e-4 | 1.082× | **1.000×** | QTT |
| qmcpack-orb0 max 1e-3 | 1.077× | **0.998×** | interp |
| qmcpack-orb0 max 1e-4 | 1.150× | 1.150× **FAIL** | wavelet (QTT screened out) |

Picks: max-abs: interp 14, QTT 2, wavelet 26. PSNR: Tucker 62, wavelet 48, QTT 2 (EXAFEL 30 dB).

## Encode-time cost

Per encode, held-out, loaded machine.

| | median | mean |
|---|---|---|
| M1 tc-auto, max-abs | 1.55 s | 1.58 s |
| portfolio, max-abs | 1.73 s | 3.83 s |
| M1 tc-auto, PSNR | 2.02 s | 4.29 s |
| portfolio, PSNR | 2.12 s | 5.23 s |

Per candidate: wavelet median 1.42 s, Tucker 0.59 s, interp 0.13 s (max 0.67 s), QTT 5.6 s median and 11.2 s mean (max 48 s, 11 encodes, all on EXAFEL). The interp candidate adds about 8% to a max-abs encode. The mean increase is almost entirely QTT on EXAFEL. Without the screen, QTT would run on every field, at up to 58 s per encode on NYX-like data.

## Follow-ups

- Replace peakiness with a screen that predicts QTT's rate. Options are a TT-rank probe (one Gram eigen-decomposition of the first row-major unfolding, or QTT on a 2× decimated copy), or simply "always try QTT when n·(expected rank) is small". On qmcpack-orb0 QTT costs only 0.35 s, so a cost-based rule ("run QTT if its predicted time < wavelet's") would have caught the miss. Re-validate on the originals, then re-freeze. The held-out set is now seen for this decision, so a fresh held-out set is needed for an honest re-gate.
- interp at PSNR targets. SZ3 does not win PSNR points, so this is low priority.

## Full per-point table

| field | point | best external | bpv | tc-auto (M1) | ratio | tc-portfolio | ratio | pick |
|---|---|---|---|---|---|---|---|---|
| cesm-CLDTOT | 40dB | sz3 | 0.1017 | 0.04556 | 0.448 | 0.04556 | 0.448 |  |
| cesm-CLDTOT | 60dB | sperr | 1.408 | 1.19 | 0.845 | 1.19 | 0.845 |  |
| cesm-CLDTOT | 80dB | sperr | 4.576 | 4.064 | 0.888 | 4.064 | 0.888 |  |
| cesm-CLDTOT | 100dB | sperr | 7.903 | 7.656 | 0.969 | 7.656 | 0.969 |  |
| cesm-CLDTOT | max0.01 | sz3 | 0.4818 | 0.3809 | 0.791 | 0.3782 | 0.785 | interp |
| cesm-CLDTOT | max0.001 | sz3 | 2.45 | 2.263 | 0.924 | 2.24 | 0.914 | interp |
| cesm-CLDTOT | max0.0001 | sz3 | 5.754 | 5.262 | 0.915 | 5.262 | 0.915 | wavelet |
| cesm-FLDS | 40dB | sz3 | 0.01215 | 0.008722 | 0.718 | 0.008722 | 0.718 |  |
| cesm-FLDS | 60dB | sperr | 0.09943 | 0.09055 | 0.911 | 0.09055 | 0.911 |  |
| cesm-FLDS | 80dB | sperr | 1.572 | 1.319 | 0.839 | 1.319 | 0.839 |  |
| cesm-FLDS | 100dB | sperr | 4.73 | 4.27 | 0.903 | 4.27 | 0.903 |  |
| cesm-FLDS | max0.01 | sz3 | 0.05589 | 0.04033 | 0.722 | 0.03299 | 0.590 | interp |
| cesm-FLDS | max0.001 | sz3 | 0.5677 | 0.4379 | 0.771 | 0.4379 | 0.771 | wavelet |
| cesm-FLDS | max0.0001 | sz3 | 2.662 | 2.435 | 0.915 | 2.422 | 0.910 | interp |
| cesm-LHFLX | 40dB | sz3 | 0.06862 | 0.01863 | 0.271 | 0.01863 | 0.271 |  |
| cesm-LHFLX | 60dB | sperr | 0.2076 | 0.1784 | 0.859 | 0.1784 | 0.859 |  |
| cesm-LHFLX | 80dB | sperr | 1.547 | 1.278 | 0.826 | 1.278 | 0.826 |  |
| cesm-LHFLX | 100dB | sperr | 4.499 | 4.06 | 0.903 | 4.06 | 0.903 |  |
| cesm-LHFLX | max0.01 | sz3 | 0.1329 | 0.1117 | 0.840 | 0.09692 | 0.729 | interp |
| cesm-LHFLX | max0.001 | sperr | 0.7516 | 0.5493 | 0.731 | 0.5493 | 0.731 | wavelet |
| cesm-LHFLX | max0.0001 | sperr | 2.787 | 2.329 | 0.836 | 2.329 | 0.836 | wavelet |
| cesm-TS | 40dB | sz3 | 0.01073 | 0.008514 | 0.794 | 0.008514 | 0.794 |  |
| cesm-TS | 60dB | sperr | 0.04098 | 0.03946 | 0.963 | 0.03946 | 0.963 |  |
| cesm-TS | 80dB | sperr | 0.3342 | 0.277 | 0.829 | 0.277 | 0.829 |  |
| cesm-TS | 100dB | sperr | 1.592 | 1.292 | 0.812 | 1.292 | 0.812 |  |
| cesm-TS | max0.01 | sz3 | 0.05013 | 0.04461 | 0.890 | 0.03007 | 0.600 | interp |
| cesm-TS | max0.001 | sz3 | 0.2293 | 0.2012 | 0.878 | 0.1797 | 0.784 | interp |
| cesm-TS | max0.0001 | sz3 | 0.8866 | 0.6843 | 0.772 | 0.6843 | 0.772 | wavelet |
| exafel-ev0 | 60dB | sz3 | 2.606 | 2.28 | 0.875 | 2.28 | 0.875 |  |
| exafel-ev0 | 80dB | sz3 | 5.824 | 5.627 | 0.966 | 5.627 | 0.966 |  |
| exafel-ev0 | 100dB | sz3 | 9.196 | 8.97 | 0.975 | 8.97 | 0.975 |  |
| exafel-ev0 | max0.01 | qtt | 0.4859 | 0.2083 | 0.429 | 0.2083 | 0.429 | wavelet |
| exafel-ev0 | max0.001 | qtt | 2.624 | 3.446 | 1.313 | 2.624 | 1.000 | qtt |
| exafel-ev0 | max0.0001 | qtt | 6.273 | 6.79 | 1.082 | 6.273 | 1.000 | qtt |
| qmcpack-orb0 | 40dB | qtt | 0.008399 | 0.004281 | 0.510 | 0.004281 | 0.510 |  |
| qmcpack-orb0 | 60dB | qtt | 0.03877 | 0.01553 | 0.401 | 0.01553 | 0.401 |  |
| qmcpack-orb0 | 80dB | tthresh | 0.1102 | 0.04545 | 0.413 | 0.04545 | 0.413 |  |
| qmcpack-orb0 | 100dB | tthresh | 0.2286 | 0.1218 | 0.533 | 0.1218 | 0.533 |  |
| qmcpack-orb0 | max0.01 | sz3 | 0.0556 | 0.06188 | 1.113 | 0.03634 | 0.654 | interp |
| qmcpack-orb0 | max0.001 | qtt | 0.1302 | 0.1402 | 1.077 | 0.1299 | 0.998 | interp |
| qmcpack-orb0 | max0.0001 | qtt | 0.2909 | 0.3346 | 1.150 | 0.3346 | 1.150 **FAIL** | wavelet |
| qmcpack-orb100 | 40dB | sz3 | 0.02619 | 0.0134 | 0.512 | 0.0134 | 0.512 |  |
| qmcpack-orb100 | 60dB | sperr | 0.0835 | 0.0648 | 0.776 | 0.0648 | 0.776 |  |
| qmcpack-orb100 | 80dB | tthresh | 0.304 | 0.1896 | 0.624 | 0.1896 | 0.624 |  |
| qmcpack-orb100 | 100dB | tthresh | 0.6353 | 0.4855 | 0.764 | 0.4855 | 0.764 |  |
| qmcpack-orb100 | max0.01 | sz3 | 0.05058 | 0.06128 | 1.211 | 0.03895 | 0.770 | interp |
| qmcpack-orb100 | max0.001 | sz3 | 0.2444 | 0.1954 | 0.800 | 0.1838 | 0.752 | interp |
| qmcpack-orb100 | max0.0001 | qtt | 0.8216 | 0.5178 | 0.630 | 0.5178 | 0.630 | wavelet |
| scale-PRES | 40dB | qtt | 0.001362 | 0.0005641 | 0.414 | 0.0005641 | 0.414 |  |
| scale-PRES | 60dB | qtt | 0.004391 | 0.003034 | 0.691 | 0.003034 | 0.691 |  |
| scale-PRES | 80dB | qtt | 0.06823 | 0.03979 | 0.583 | 0.03979 | 0.583 |  |
| scale-PRES | 100dB | tthresh | 0.3145 | 0.1983 | 0.630 | 0.1983 | 0.630 |  |
| scale-PRES | max0.01 | sz3 | 0.008682 | 0.01292 | 1.489 | 0.0065 | 0.749 | interp |
| scale-PRES | max0.001 | qtt | 0.05598 | 0.03733 | 0.667 | 0.03733 | 0.667 | wavelet |
| scale-PRES | max0.0001 | qtt | 0.2024 | 0.1812 | 0.895 | 0.1812 | 0.895 | wavelet |
| scale-QV | 40dB | qtt | 0.01676 | 0.006808 | 0.406 | 0.006808 | 0.406 |  |
| scale-QV | 60dB | sperr | 0.1588 | 0.1157 | 0.729 | 0.1157 | 0.729 |  |
| scale-QV | 80dB | sperr | 1.398 | 1.207 | 0.864 | 1.207 | 0.864 |  |
| scale-QV | 100dB | sperr | 3.684 | 3.317 | 0.900 | 3.317 | 0.900 |  |
| scale-QV | max0.01 | sperr | 0.1222 | 0.08893 | 0.728 | 0.08893 | 0.728 | wavelet |
| scale-QV | max0.001 | sperr | 0.7633 | 0.6475 | 0.848 | 0.6475 | 0.848 | wavelet |
| scale-QV | max0.0001 | sperr | 2.511 | 2.156 | 0.858 | 2.156 | 0.858 | wavelet |
| scale-T | 40dB | qtt | 0.001936 | 0.0006773 | 0.350 | 0.0006773 | 0.350 |  |
| scale-T | 60dB | sperr | 0.02316 | 0.02149 | 0.928 | 0.02149 | 0.928 |  |
| scale-T | 80dB | tthresh | 0.4548 | 0.3285 | 0.722 | 0.3285 | 0.722 |  |
| scale-T | 100dB | tthresh | 2.574 | 2.209 | 0.858 | 2.209 | 0.858 |  |
| scale-T | max0.01 | qtt | 0.02405 | 0.02444 | 1.016 | 0.02282 | 0.949 | interp |
| scale-T | max0.001 | sperr | 0.2468 | 0.1876 | 0.760 | 0.1876 | 0.760 | wavelet |
| scale-T | max0.0001 | sperr | 1.417 | 1.245 | 0.879 | 1.245 | 0.879 | wavelet |
| scale-U | 40dB | qtt | 0.0294 | 0.01149 | 0.391 | 0.01149 | 0.391 |  |
| scale-U | 60dB | tthresh | 0.2482 | 0.1499 | 0.604 | 0.1499 | 0.604 |  |
| scale-U | 80dB | tthresh | 1.616 | 1.292 | 0.799 | 1.292 | 0.799 |  |
| scale-U | 100dB | tthresh | 4.539 | 3.92 | 0.864 | 3.92 | 0.864 |  |
| scale-U | max0.01 | sperr | 0.1636 | 0.122 | 0.746 | 0.122 | 0.746 | wavelet |
| scale-U | max0.001 | sperr | 0.952 | 0.8242 | 0.866 | 0.8242 | 0.866 | wavelet |
| scale-U | max0.0001 | sperr | 3.158 | 2.716 | 0.860 | 2.716 | 0.860 | wavelet |
| hurricane-V | 40dB | sz3 | 0.03155 | 0.008008 | 0.254 | 0.008008 | 0.254 |  |
| hurricane-V | 60dB | sperr | 0.2593 | 0.2628 | 1.014 | 0.2628 | 1.014 |  |
| hurricane-V | 80dB | tthresh | 1.397 | 1.135 | 0.813 | 1.135 | 0.813 |  |
| hurricane-V | 100dB | tthresh | 3.133 | 2.757 | 0.880 | 2.757 | 0.880 |  |
| hurricane-V | max0.01 | sz3 | 0.143 | 0.09356 | 0.654 | 0.09356 | 0.654 | wavelet |
| hurricane-V | max0.001 | sperr | 0.8708 | 0.7919 | 0.909 | 0.7919 | 0.909 | wavelet |
| hurricane-V | max0.0001 | sperr | 2.637 | 2.423 | 0.919 | 2.423 | 0.919 | wavelet |
| miranda-pressure | 40dB | tthresh | 0.0635 | 0.02818 | 0.444 | 0.02818 | 0.444 |  |
| miranda-pressure | 60dB | sperr | 0.2322 | 0.1772 | 0.763 | 0.1772 | 0.763 |  |
| miranda-pressure | 80dB | sperr | 1.036 | 0.5731 | 0.553 | 0.5731 | 0.553 |  |
| miranda-pressure | 100dB | sperr | 2.648 | 1.553 | 0.587 | 1.553 | 0.587 |  |
| miranda-pressure | max0.01 | sperr | 0.1931 | 0.1142 | 0.592 | 0.1142 | 0.592 | wavelet |
| miranda-pressure | max0.001 | sperr | 0.6916 | 0.4852 | 0.702 | 0.4852 | 0.702 | wavelet |
| miranda-pressure | max0.0001 | sperr | 1.829 | 1.455 | 0.796 | 1.455 | 0.796 | wavelet |
| nyx-velocity_x | 40dB | sz3 | 0.01771 | 0.006378 | 0.360 | 0.006378 | 0.360 |  |
| nyx-velocity_x | 60dB | sperr | 0.2585 | 0.1934 | 0.748 | 0.1934 | 0.748 |  |
| nyx-velocity_x | 80dB | sperr | 2.603 | 2.179 | 0.837 | 2.179 | 0.837 |  |
| nyx-velocity_x | 100dB | sperr | 5.978 | 5.489 | 0.918 | 5.489 | 0.918 |  |
| nyx-velocity_x | max0.01 | sz3 | 0.1264 | 0.08398 | 0.664 | 0.08152 | 0.645 | interp |
| nyx-velocity_x | max0.001 | sz3 | 1.101 | 0.8186 | 0.743 | 0.8186 | 0.743 | wavelet |
| nyx-velocity_x | max0.0001 | sz3 | 3.878 | 3.405 | 0.878 | 3.405 | 0.878 | wavelet |

**Geo-mean tc-portfolio / best external:** 40dB 0.429 (n=13), 60dB 0.775 (n=14), 80dB 0.737 (n=14), 100dB 0.808 (n=14), max0.01 0.677 (n=14), max0.001 0.811 (n=14), max0.0001 0.871 (n=14)

**Win/loss/tie (±1%) vs best external:** PSNR 54–1–0; max-abs 38–1–3

**Gate (no point > 1.05× best external):** FAIL at 1 points
- qmcpack-orb0 max0.0001: tc-portfolio 0.3346 vs qtt 0.2909 (1.150×), pick=wavelet

**Encode time per encode (s), median / mean:** tc-auto maxabs: 1.55 / 1.58 (n=42); tc-auto psnr: 2.02 / 4.29 (n=112); tc-portfolio maxabs: 1.73 / 3.83 (n=42); tc-portfolio psnr: 2.12 / 5.23 (n=112)

**Picks:** maxabs/interp=14, maxabs/qtt=2, maxabs/wavelet=26, psnr/qtt=2, psnr/tucker=62, psnr/wavelet=48

**QTT tried (pre-screen passed):** 11 of 154 encodes; peakiness per field: cesm-CLDTOT 1.3, cesm-FLDS 1.3, cesm-LHFLX 1.3, cesm-TS 1.8, exafel-ev0 48.5, qmcpack-orb0 1.4, qmcpack-orb100 1.6, scale-PRES 1.0, scale-QV 1.1, scale-T 1.0, scale-U 1.3, hurricane-V 1.2, miranda-pressure 2.5, nyx-velocity_x 2.7
