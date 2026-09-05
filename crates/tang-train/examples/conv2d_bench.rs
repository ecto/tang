//! `Conv2d` on the kernel-predicting-denoiser workload, three ways.
//!
//! The network is the one in `kosm-spike`'s `court::denoise::kpn`: three 3×3
//! convolutions, 10→32→32→25, on 64×64 tiles at batch 16, forward and
//! backward. That is ~80 million multiply-accumulates per tile in the forward
//! pass alone, which is why that crate hand-wrote its own flat `f32` loops
//! instead of using this one.
//!
//! ```text
//! cargo run --release -p tang-train --example conv2d_bench
//! cargo run --release -p tang-train --example conv2d_bench -- --iters 20
//! ```
//!
//! Three implementations are timed on identical data:
//!
//! - `fast` — the im2col + GEMM path `Conv2d` now takes.
//! - `reference` — `Conv2d::forward_reference` / `backward_reference`, the
//!   `Tensor::from_fn` over a multi-dimensional `get` this replaced. Timed for
//!   a single step, because it is slow enough that more would be rude.
//! - `handwritten` — the flat planar loops from `kosm-spike`, transcribed here
//!   (with this layer's zero padding rather than that one's clamp-to-edge, so
//!   all three compute the same function) to check the GEMM path against what
//!   a careful hand-rolled convolution achieves.

use std::time::Instant;

use tang_tensor::{Shape, Tensor};
use tang_train::{Conv2d, Module};

const BATCH: usize = 16;
const S: usize = 64;
const KS: usize = 3;
const CHANNELS: [usize; 4] = [10, 32, 32, 25];

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
        })
        .collect()
}

fn tensor(seed: u64, dims: &[usize]) -> Tensor<f32> {
    let shape = Shape::from_slice(dims);
    Tensor::new(noise(seed, shape.numel()), shape)
}

fn layers() -> Vec<Conv2d<f32>> {
    CHANNELS
        .windows(2)
        .enumerate()
        .map(|(i, cs)| {
            let mut c = Conv2d::<f32>::with_options(cs[0], cs[1], KS, 1, 1, 1, 1 + i as u64);
            c.weight.data = tensor(0xC0FFEE + i as u64, &[cs[1], cs[0], KS, KS]);
            c.bias.data = tensor(0xBEEF + i as u64, &[cs[1]]);
            c
        })
        .collect()
}

/// Multiply-accumulates in one forward pass, for a MAC/s figure.
fn forward_macs() -> f64 {
    CHANNELS
        .windows(2)
        .map(|cs| (BATCH * cs[0] * cs[1] * KS * KS * S * S) as f64)
        .sum()
}

fn step(net: &mut [Conv2d<f32>], x: &Tensor<f32>, reference: bool) -> f32 {
    let mut a = x.clone();
    for l in net.iter_mut() {
        a = if reference {
            l.forward_reference(&a)
        } else {
            l.forward(&a)
        };
    }
    // A stand-in for dL/dy; the loss itself is not what is being timed.
    let checksum = a.data().iter().sum::<f32>();
    let mut g = Tensor::new(
        vec![1.0f32 / a.numel() as f32; a.numel()],
        a.shape().clone(),
    );
    for l in net.iter_mut().rev() {
        g = if reference {
            l.backward_reference(&g)
        } else {
            l.backward(&g)
        };
        l.weight.zero_grad();
        l.bias.zero_grad();
    }
    checksum + g.data()[0]
}

// ---------------------------------------------------------------------------
// The hand-written planar loops, after kosm-spike's `court::denoise::kpn`.
// ---------------------------------------------------------------------------

struct Flat {
    c_in: usize,
    c_out: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

/// The tap's valid output range along one axis: `o + off` must land in `[0, s)`.
#[inline]
fn span(off: isize, s: usize) -> (usize, usize) {
    let lo = (-off).max(0) as usize;
    let hi = (s as isize - off).min(s as isize).max(0) as usize;
    (lo, hi)
}

fn flat_forward(c: &Flat, a: &[f32], s: usize, out: &mut [f32]) {
    let n = s * s;
    for o in 0..c.c_out {
        let dst = &mut out[o * n..(o + 1) * n];
        dst.fill(c.b[o]);
        for i in 0..c.c_in {
            let src = &a[i * n..(i + 1) * n];
            let wk = &c.w[(o * c.c_in + i) * KS * KS..(o * c.c_in + i + 1) * KS * KS];
            for dy in 0..KS {
                let (ylo, yhi) = span(dy as isize - 1, s);
                for dx in 0..KS {
                    let k = wk[dy * KS + dx];
                    if k == 0.0 {
                        continue;
                    }
                    let (xlo, xhi) = span(dx as isize - 1, s);
                    let (oy, ox) = (dy as isize - 1, dx as isize - 1);
                    for y in ylo..yhi {
                        let sy = (y as isize + oy) as usize * s;
                        let d = &mut dst[y * s + xlo..y * s + xhi];
                        let sl = &src[(sy as isize + xlo as isize + ox) as usize..][..xhi - xlo];
                        for (dv, sv) in d.iter_mut().zip(sl) {
                            *dv += k * sv;
                        }
                    }
                }
            }
        }
    }
}

fn flat_backward(
    c: &Flat,
    a: &[f32],
    gz: &[f32],
    s: usize,
    gw: &mut [f32],
    gb: &mut [f32],
    ga: &mut [f32],
) {
    let n = s * s;
    ga.fill(0.0);
    for o in 0..c.c_out {
        let gzo = &gz[o * n..(o + 1) * n];
        gb[o] += gzo.iter().sum::<f32>();
        for i in 0..c.c_in {
            let src = &a[i * n..(i + 1) * n];
            let base = (o * c.c_in + i) * KS * KS;
            for dy in 0..KS {
                let (ylo, yhi) = span(dy as isize - 1, s);
                for dx in 0..KS {
                    let (xlo, xhi) = span(dx as isize - 1, s);
                    let (oy, ox) = (dy as isize - 1, dx as isize - 1);
                    let mut acc = 0.0f32;
                    for y in ylo..yhi {
                        let sy = (y as isize + oy) as usize * s;
                        let g = &gzo[y * s + xlo..y * s + xhi];
                        let sl = &src[(sy as isize + xlo as isize + ox) as usize..][..xhi - xlo];
                        for (gv, sv) in g.iter().zip(sl) {
                            acc += gv * sv;
                        }
                    }
                    gw[base + dy * KS + dx] += acc;
                }
            }
        }
    }
    for i in 0..c.c_in {
        let gi = &mut ga[i * n..(i + 1) * n];
        for o in 0..c.c_out {
            let gzo = &gz[o * n..(o + 1) * n];
            let wk = &c.w[(o * c.c_in + i) * KS * KS..(o * c.c_in + i + 1) * KS * KS];
            for dy in 0..KS {
                let (ylo, yhi) = span(dy as isize - 1, s);
                for dx in 0..KS {
                    let k = wk[dy * KS + dx];
                    if k == 0.0 {
                        continue;
                    }
                    let (xlo, xhi) = span(dx as isize - 1, s);
                    let (oy, ox) = (dy as isize - 1, dx as isize - 1);
                    for y in ylo..yhi {
                        let sy = (y as isize + oy) as usize * s;
                        let g = &gzo[y * s + xlo..y * s + xhi];
                        let d = &mut gi[(sy as isize + xlo as isize + ox) as usize..][..xhi - xlo];
                        for (dv, gv) in d.iter_mut().zip(g) {
                            *dv += k * gv;
                        }
                    }
                }
            }
        }
    }
}

fn flat_step(net: &[Flat], x: &[f32], batch: usize, side: usize) -> f32 {
    let n = side * side;
    let mut acts: Vec<Vec<f32>> = Vec::new();
    for b in 0..batch {
        let mut a = x[b * CHANNELS[0] * n..(b + 1) * CHANNELS[0] * n].to_vec();
        let mut saved = vec![a.clone()];
        for l in net {
            let mut out = vec![0.0f32; l.c_out * n];
            flat_forward(l, &a, side, &mut out);
            saved.push(out.clone());
            a = out;
        }
        acts.push(saved.concat());
    }

    let mut checksum = 0.0f32;
    let mut offsets = Vec::new();
    let mut off = 0;
    for c in CHANNELS {
        offsets.push(off);
        off += c * n;
    }

    for saved in &acts {
        let last = &saved[offsets[3]..];
        checksum += last.iter().sum::<f32>();
        let mut g = vec![1.0f32 / (batch * last.len()) as f32; last.len()];
        for (li, l) in net.iter().enumerate().rev() {
            let a = &saved[offsets[li]..offsets[li] + l.c_in * n];
            let mut gw = vec![0.0f32; l.w.len()];
            let mut gb = vec![0.0f32; l.b.len()];
            let mut ga = vec![0.0f32; l.c_in * n];
            flat_backward(l, a, &g, side, &mut gw, &mut gb, &mut ga);
            checksum += gw[0] + gb[0];
            g = ga;
        }
    }
    checksum
}

/// Best of `reps`, not the mean: on a shared machine the mean measures the
/// other tenants. The minimum is the closest thing to an uncontended run that
/// a wall clock can see.
fn best(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut lo = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        lo = lo.min(t.elapsed().as_secs_f64());
    }
    lo * 1e3
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters: usize = args
        .iter()
        .position(|a| a == "--iters")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let skip_reference = args.iter().any(|a| a == "--no-reference");

    let x = tensor(0xDA7A, &[BATCH, CHANNELS[0], S, S]);
    let macs = forward_macs();
    println!(
        "kpn denoiser: batch {BATCH}, {S}x{S}, 3x3 pad 1, channels {:?}",
        CHANNELS
    );
    println!(
        "{:.1} MMAC per tile forward, {:.2} GMAC per batch forward, best of {iters}\n",
        macs / BATCH as f64 / 1e6,
        macs / 1e9
    );

    let mut sink = 0.0f32;

    let mut net = layers();
    let fast_ms = best(iters, || sink += step(&mut net, &x, false));
    println!("{:<28} {fast_ms:>12.2} ms/step", "fast (im2col + GEMM)");

    let flat: Vec<Flat> = net
        .iter()
        .map(|l| Flat {
            c_in: l.weight.data.shape()[1],
            c_out: l.weight.data.shape()[0],
            w: l.weight.data.data().to_vec(),
            b: l.bias.data.data().to_vec(),
        })
        .collect();
    let flat_ms = best(iters, || sink += flat_step(&flat, x.data(), BATCH, S));
    println!("{:<28} {flat_ms:>12.2} ms/step", "handwritten (kosm-spike)");

    // One sample only: the `from_fn` path takes minutes, and the point of the
    // number is its order of magnitude.
    let slow_ms = if skip_reference {
        f64::NAN
    } else {
        let mut net = layers();
        let t = Instant::now();
        sink += step(&mut net, &x, true);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "{:<28} {ms:>12.2} ms/step  (1 sample)",
            "reference (from_fn)"
        );
        ms
    };

    println!();
    if slow_ms.is_finite() {
        println!("speedup vs reference   {:>8.1}x", slow_ms / fast_ms);
    }
    println!("vs handwritten         {:>8.2}x", flat_ms / fast_ms);
    println!(
        "throughput             {:>8.1} GMAC/s (forward + backward ~ 3x forward work)",
        macs * 3.0 / (fast_ms / 1e3) / 1e9
    );

    breakdown(&x, iters, &mut sink);
    scaled_ratio(iters, &mut sink);
    println!("\n(checksum {sink:e})");
}

/// The same step, timed one layer-direction at a time.
///
/// A whole step is a couple of seconds; on a busy machine `best` never catches
/// a quiet window that long, so the totals above read as whatever else is
/// running. Each piece here is tens of milliseconds, short enough that the
/// minimum over a handful of samples is close to an uncontended figure — so
/// this table, and not the one above, is the one to compare against a clean
/// machine.
fn breakdown(x: &Tensor<f32>, iters: usize, sink: &mut f32) {
    println!("\nper layer-direction (best of {}):", iters * 4);
    let mut net = layers();
    let mut acts = alloc_activations(&mut net, x);
    let mut total = 0.0;

    for (i, l) in net.iter_mut().enumerate() {
        let a = acts[i].clone();
        let ms = best(iters * 4, || {
            *sink += l.forward(&a).data()[0];
        });
        total += ms;
        println!("  layer {i} forward   {:>9.2} ms", ms);
    }
    for (i, l) in net.iter_mut().enumerate().rev() {
        let g = acts[i + 1].clone();
        let g = Tensor::new(
            vec![1.0f32 / g.numel() as f32; g.numel()],
            g.shape().clone(),
        );
        l.forward(&acts[i]);
        let ms = best(iters * 4, || {
            *sink += l.backward(&g).data()[0];
            l.weight.zero_grad();
            l.bias.zero_grad();
        });
        total += ms;
        println!("  layer {i} backward  {:>9.2} ms", ms);
    }
    acts.clear();
    println!("  {:<17} {:>9.2} ms/step", "sum", total);
}

/// The fast path against the `from_fn` reference on a workload small enough to
/// time both properly.
///
/// One reference step on the full workload runs into minutes, so it can only
/// ever be sampled once — and a single sample on a contended machine measures
/// the contention. At batch 2 on 16x16 tiles the same network is about 1/500th
/// of the work, which puts a reference step in the tens of milliseconds and
/// lets both paths be measured best-of-N under identical conditions. The
/// arithmetic is identical and the ratio is the honest one.
fn scaled_ratio(iters: usize, sink: &mut f32) {
    const SIDE: usize = 32;
    const SB: usize = 4;
    let x = tensor(0x5EA1, &[SB, CHANNELS[0], SIDE, SIDE]);

    let mut fast = layers();
    let mut slow = layers();
    let flat: Vec<Flat> = fast
        .iter()
        .map(|l| Flat {
            c_in: l.weight.data.shape()[1],
            c_out: l.weight.data.shape()[0],
            w: l.weight.data.data().to_vec(),
            b: l.bias.data.data().to_vec(),
        })
        .collect();

    let fast_ms = best(iters * 4, || *sink += step(&mut fast, &x, false));
    let flat_ms = best(iters * 4, || *sink += flat_step(&flat, x.data(), SB, SIDE));
    let slow_ms = best(iters * 3, || *sink += step(&mut slow, &x, true));

    println!(
        "\nsame network at batch {SB} on {SIDE}x{SIDE} — 1/32 the work, so all three fit a quiet window:"
    );
    println!("  fast                {fast_ms:>9.2} ms/step");
    println!(
        "  handwritten         {flat_ms:>9.2} ms/step  ({:.2}x fast)",
        flat_ms / fast_ms
    );
    println!(
        "  reference           {slow_ms:>9.2} ms/step  ({:.1}x fast)",
        slow_ms / fast_ms
    );
}

/// Forward once to capture each layer's input, for the per-layer timings.
fn alloc_activations(net: &mut [Conv2d<f32>], x: &Tensor<f32>) -> Vec<Tensor<f32>> {
    let mut acts = vec![x.clone()];
    for l in net.iter_mut() {
        let y = l.forward(acts.last().unwrap());
        acts.push(y);
    }
    acts
}
