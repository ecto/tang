use core::cell::RefCell;
use tang::{Dual, Scalar};
use tang_tensor::{Shape, Tensor};
use tang_train::{
    cross_entropy_loss, cross_entropy_loss_grad, softmax, DataLoader, Linear, Module, ModuleAdam,
    Optimizer, Sequential, Tanh, TensorDataset,
};
use wasm_bindgen::prelude::*;

// ── Quantum Poet ─────────────────────────────────────────────────────────────

const VOCAB: [char; 30] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '\n', ',', '.',
];
const VOCAB_SIZE: usize = 30;
const WINDOW: usize = 8;
const INPUT_DIM: usize = WINDOW * VOCAB_SIZE; // 240

const CORPUS: &str = "\
entropy rises,\n\
disorder finds its freedom,\n\
heat death whispers on.\n\
\n\
photons race through void,\n\
wavelengths paint the spectrum bright,\n\
light bends around mass.\n\
\n\
spacetime curves and warps,\n\
gravity is geometry,\n\
falling is floating.\n\
\n\
quantum fields vibrate,\n\
particles dance in the void,\n\
nothing stays at rest.\n\
\n\
the arrow of time,\n\
entropy always increases,\n\
stars burn and fade out.\n\
\n\
waves upon the shore,\n\
frequency hums in the dark,\n\
energy persists.\n\
\n\
electrons jump and spin,\n\
orbits are clouds of maybe,\n\
certainty dissolves.\n\
\n\
black holes bend the light,\n\
singularity awaits,\n\
time itself stands still.\n\
\n\
neutrinos pass through,\n\
ghostly messengers of stars,\n\
barely touching worlds.\n\
\n\
dark matter hides well,\n\
unseen scaffolding of space,\n\
holding worlds in place.\n\
\n\
strings vibrate below,\n\
dimensions curl up and hide,\n\
theory seeks the truth.\n\
\n\
the cosmos expands,\n\
redshift whispers of the bang,\n\
silence fills the void.\n\
\n\
symmetry can break,\n\
forces split from ancient one,\n\
cooling makes them new.\n\
\n\
spin entangles far,\n\
measurement collapses waves,\n\
spooky action reigns.\n\
\n\
relativity,\n\
mass and energy are one,\n\
speed of light the law.\n\
\n\
momentum conserved,\n\
collisions trade their motion,\n\
nothing can be lost.\n\
\n\
the vacuum fluctuates,\n\
virtual pairs appear and fade,\n\
emptiness is full.\n\
\n\
a muon decays,\n\
half life ticks in silent math,\n\
nature counts in chance.\n\
\n\
planck length defines,\n\
the smallest scene in the play,\n\
spacetime is granular.\n\
\n\
the double slit shows,\n\
interference paints the bands,\n\
watching changes all.";

fn char_to_idx(c: char) -> Option<usize> {
    VOCAB.iter().position(|&v| v == c)
}

fn idx_to_char(i: usize) -> char {
    VOCAB[i]
}

fn build_dataset() -> (Vec<f64>, Vec<f64>, usize) {
    let chars: Vec<char> = CORPUS.chars().collect();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut count = 0;

    for i in WINDOW..chars.len() {
        let window_ok = (0..WINDOW).all(|j| char_to_idx(chars[i - WINDOW + j]).is_some());
        let target_idx = char_to_idx(chars[i]);
        if !window_ok || target_idx.is_none() {
            continue;
        }

        let mut one_hot = vec![0.0f64; INPUT_DIM];
        for j in 0..WINDOW {
            let ci = char_to_idx(chars[i - WINDOW + j]).unwrap();
            one_hot[j * VOCAB_SIZE + ci] = 1.0;
        }

        inputs.extend_from_slice(&one_hot);
        targets.push(target_idx.unwrap() as f64);
        count += 1;
    }

    (inputs, targets, count)
}

struct PoetState {
    model: Sequential<f64>,
    optimizer: ModuleAdam,
    dataset_inputs: Tensor<f64>,
    dataset_targets: Tensor<f64>,
}

thread_local! {
    static POET: RefCell<Option<PoetState>> = RefCell::new(None);
}

/// Initialize the Quantum Poet model and dataset.
#[wasm_bindgen]
pub fn poet_init() {
    let (input_data, target_data, n_samples) = build_dataset();
    let inputs = Tensor::new(input_data, Shape::new(vec![n_samples, INPUT_DIM]));
    let targets = Tensor::new(target_data, Shape::new(vec![n_samples]));

    let model = Sequential::<f64>::new(vec![
        Box::new(Linear::new(INPUT_DIM, 64, 42)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(64, 64, 137)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(64, VOCAB_SIZE, 256)),
    ]);

    let optimizer = ModuleAdam::new(0.005);

    POET.with(|p| {
        *p.borrow_mut() = Some(PoetState {
            model,
            optimizer,
            dataset_inputs: inputs,
            dataset_targets: targets,
        });
    });
}

/// Run one training epoch. Returns average loss for the epoch.
#[wasm_bindgen]
pub fn poet_train_epoch() -> f64 {
    POET.with(|p| {
        let mut state = p.borrow_mut();
        let s = state.as_mut().expect("call poet_init first");

        let dataset = TensorDataset::new(s.dataset_inputs.clone(), s.dataset_targets.clone());
        let mut loader = DataLoader::new(&dataset, 32);

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for (inputs, targets) in loader.by_ref() {
            s.model.zero_grad();
            let pred = s.model.forward(&inputs);
            let (loss, grad) = (
                cross_entropy_loss(&pred, &targets),
                cross_entropy_loss_grad(&pred, &targets),
            );
            s.model.backward(&grad);

            let mut params = s.model.parameters_mut();
            s.optimizer.step(&mut params);

            epoch_loss += loss;
            batch_count += 1;
        }

        if batch_count > 0 {
            epoch_loss / batch_count as f64
        } else {
            0.0
        }
    })
}

/// Generate text from a seed string with temperature sampling.
#[wasm_bindgen]
pub fn poet_generate(seed: &str, temperature: f64) -> String {
    POET.with(|p| {
        let mut state = p.borrow_mut();
        let s = state.as_mut().expect("call poet_init first");

        let mut window: Vec<usize> = seed.chars().filter_map(char_to_idx).collect();

        // Pad or truncate to WINDOW size
        while window.len() < WINDOW {
            window.insert(0, char_to_idx(' ').unwrap());
        }
        window.truncate(WINDOW);

        let mut output = String::from(seed);

        for _ in 0..200 {
            let mut input_data = vec![0.0f64; INPUT_DIM];
            for (j, &ci) in window.iter().enumerate() {
                input_data[j * VOCAB_SIZE + ci] = 1.0;
            }
            let input = Tensor::new(input_data, Shape::new(vec![1, INPUT_DIM]));

            let logits = s.model.forward(&input);
            let row = Tensor::new(logits.data().to_vec(), Shape::new(vec![VOCAB_SIZE]));

            let scaled = row.scale(1.0 / temperature);
            let probs = softmax(&scaled);

            // Deterministic sampling from probability distribution
            let hash = logits.data().iter().fold(0u64, |acc, &v| {
                acc.wrapping_mul(6364136223846793005)
                    .wrapping_add(v.to_bits() ^ 0xdeadbeef)
            });
            let r = (hash % 10000) as f64 / 10000.0;
            let mut cumulative = 0.0;
            let mut next_idx = VOCAB_SIZE - 1;
            for i in 0..VOCAB_SIZE {
                cumulative += probs.data()[i];
                if r < cumulative {
                    next_idx = i;
                    break;
                }
            }

            let next_char = idx_to_char(next_idx);
            output.push(next_char);

            window.remove(0);
            window.push(next_idx);
        }

        output
    })
}

/// Evaluate f(x) = x * sin(x) using Dual<f64>.
/// Returns [value, derivative].
#[wasm_bindgen]
pub fn dual_eval(x: f64) -> Box<[f64]> {
    let d = Dual::var(x);
    let y = d * d.sin();
    Box::new([y.real, y.dual])
}

/// Generate SVG path `d` attribute for f(x) = x * sin(x).
/// Maps math coordinates to SVG pixel coordinates.
#[wasm_bindgen]
pub fn curve_svg_path(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    w: f64,
    h: f64,
    steps: u32,
) -> String {
    let mut d = String::with_capacity(steps as usize * 20);
    for i in 0..steps {
        let t = i as f64 / (steps - 1) as f64;
        let x = x_min + (x_max - x_min) * t;
        let y = x * x.sin();
        let sx = (x - x_min) / (x_max - x_min) * w;
        let sy = h - (y - y_min) / (y_max - y_min) * h;
        if i == 0 {
            d.push_str(&format!("M{:.1},{:.1}", sx, sy));
        } else {
            d.push_str(&format!(" L{:.1},{:.1}", sx, sy));
        }
    }
    d
}

/// Convert math coordinates to SVG pixel coordinates.
/// Returns [sx, sy].
#[wasm_bindgen]
pub fn to_svg(
    x: f64,
    y: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    w: f64,
    h: f64,
) -> Box<[f64]> {
    let sx = (x - x_min) / (x_max - x_min) * w;
    let sy = h - (y - y_min) / (y_max - y_min) * h;
    Box::new([sx, sy])
}

/// Compute tangent line endpoints in SVG coordinates.
/// Returns [x1, y1, x2, y2].
#[wasm_bindgen]
pub fn tangent_svg(
    x: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    w: f64,
    h: f64,
    half_len: f64,
) -> Box<[f64]> {
    let d = Dual::var(x);
    let y_d = d * d.sin();
    let y_val = y_d.real;
    let slope = y_d.dual;

    let x1 = x - half_len;
    let y1 = y_val - slope * half_len;
    let x2 = x + half_len;
    let y2 = y_val + slope * half_len;

    let sx1 = (x1 - x_min) / (x_max - x_min) * w;
    let sy1 = h - (y1 - y_min) / (y_max - y_min) * h;
    let sx2 = (x2 - x_min) / (x_max - x_min) * w;
    let sy2 = h - (y2 - y_min) / (y_max - y_min) * h;

    Box::new([sx1, sy1, sx2, sy2])
}
mod graph;
mod kernel;

/// Acceleration toward a unit-mass planet at the origin (GM = 1).
pub(crate) fn gravity<S: Scalar>(x: S, y: S) -> (S, S) {
    let r2 = x * x + y * y;
    let inv_r3 = S::from_f64(1.0) / (r2 * r2.sqrt());
    (S::from_f64(0.0) - x * inv_r3, S::from_f64(0.0) - y * inv_r3)
}

/// One orbit-transfer attempt, written once over `S: Scalar`: launch from the
/// top of a unit planet (GM = 1) with `speed` and `heading` (radians above the
/// local horizon), integrate with leapfrog, and score the closest approach to
/// the target. With `Dual<f64>` the score carries one exact partial derivative.
fn orbit_miss<S: Scalar>(speed: S, heading: S, tx: f64, ty: f64, path: &mut Vec<f64>) -> S {
    let dt = S::from_f64(0.01);
    let half = S::from_f64(0.5);
    let (s, c) = heading.sin_cos();
    let (mut x, mut y) = (S::from_f64(0.0), S::from_f64(1.0));
    let (mut vx, mut vy) = (speed * c, speed * s);
    let acc = gravity::<S>;
    let (mut ax, mut ay) = acc(x, y);
    let mut best = S::from_f64(f64::INFINITY);
    for k in 0..1600 {
        let (dx, dy) = (x - S::from_f64(tx), y - S::from_f64(ty));
        let d2 = dx * dx + dy * dy;
        if d2.to_f64() < best.to_f64() {
            best = d2;
        }
        if k % 8 == 0 {
            path.push(x.to_f64());
            path.push(y.to_f64());
        }
        vx = vx + ax * dt * half;
        vy = vy + ay * dt * half;
        x = x + vx * dt;
        y = y + vy * dt;
        (ax, ay) = acc(x, y);
        vx = vx + ax * dt * half;
        vy = vy + ay * dt * half;
        let r2 = (x * x + y * y).to_f64();
        if r2 < 1.0 || r2 > 64.0 {
            break;
        }
    }
    best
}

/// Score a launch and its gradient, one dual pass per parameter.
/// Returns [miss², ∂/∂speed, ∂/∂heading, path xy...].
#[wasm_bindgen]
pub fn orbit_eval(speed: f64, heading: f64, tx: f64, ty: f64) -> Box<[f64]> {
    let mut path = Vec::new();
    let ds = orbit_miss(Dual::var(speed), Dual::constant(heading), tx, ty, &mut path);
    let mut scratch = Vec::new();
    let dh = orbit_miss(
        Dual::constant(speed),
        Dual::var(heading),
        tx,
        ty,
        &mut scratch,
    );
    let mut out = vec![ds.real, ds.dual, dh.dual];
    out.extend(path);
    out.into_boxed_slice()
}

/// Planar forward kinematics for a rose full arm (side view): shoulder pitch
/// `q1` from hanging straight down, elbow `q2`. Links from rose_full_layout.py.
pub(crate) fn arm_tip<S: Scalar>(q1: S, q2: S) -> (S, S) {
    let (l1, l2) = (S::from_f64(0.227), S::from_f64(0.236));
    let (s1, c1) = q1.sin_cos();
    let (s12, c12) = (q1 + q2).sin_cos();
    (l1 * s1 + l2 * s12, S::from_f64(0.0) - l1 * c1 - l2 * c12)
}

/// One damped-least-squares IK step toward (tx, ty), with the Jacobian from
/// two dual passes through `arm_tip`. Returns [q1, q2, error, elbow x, elbow y, tip x, tip y].
#[wasm_bindgen]
pub fn arm_ik_step(q1: f64, q2: f64, tx: f64, ty: f64) -> Box<[f64]> {
    let (a, b) = arm_tip(Dual::var(q1), Dual::constant(q2));
    let (c, d) = arm_tip(Dual::constant(q1), Dual::var(q2));
    let (j11, j21, j12, j22) = (a.dual, b.dual, c.dual, d.dual);
    let (ex, ey) = (tx - a.real, ty - b.real);
    // (JᵀJ + λ²I)⁻¹ Jᵀ e
    let lam2 = 0.0004;
    let (m11, m12, m22) = (
        j11 * j11 + j21 * j21 + lam2,
        j11 * j12 + j21 * j22,
        j12 * j12 + j22 * j22 + lam2,
    );
    let (g1, g2) = (j11 * ex + j21 * ey, j12 * ex + j22 * ey);
    let det = m11 * m22 - m12 * m12;
    let dq1 = (m22 * g1 - m12 * g2) / det;
    let dq2 = (m11 * g2 - m12 * g1) / det;
    let nq1 = (q1 + 0.5 * dq1).clamp(-1.0, 3.0);
    let nq2 = (q2 + 0.5 * dq2).clamp(0.0, 2.6);
    let (tipx, tipy) = arm_tip(nq1, nq2);
    let err = ((tx - tipx).powi(2) + (ty - tipy).powi(2)).sqrt();
    Box::new([
        nq1,
        nq2,
        err,
        0.227 * nq1.sin(),
        -0.227 * nq1.cos(),
        tipx,
        tipy,
    ])
}
