//! The Quantum Poet (GPU) — character-level text generator trained on physics haikus.
//!
//! GPU port of the tang-train quantum_poet example using tang-gpu's
//! fused wgpu training pipeline.
//!
//! ```sh
//! cargo run --example quantum_poet -p tang-gpu
//! ```

use tang_gpu::*;

// --- Vocabulary ---------------------------------------------------------------

const VOCAB: [char; 30] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '\n', ',', '.',
];

fn char_to_idx(c: char) -> Option<usize> {
    VOCAB.iter().position(|&v| v == c)
}

fn idx_to_char(i: usize) -> char {
    VOCAB[i]
}

const VOCAB_SIZE: usize = 30;
const WINDOW: usize = 8;
const INPUT_DIM: usize = WINDOW * VOCAB_SIZE; // 240

// --- Corpus -------------------------------------------------------------------

const CORPUS: &str = "\
entropy rises,
disorder finds its freedom,
heat death whispers on.

photons race through void,
wavelengths paint the spectrum bright,
light bends around mass.

spacetime curves and warps,
gravity is geometry,
falling is floating.

quantum fields vibrate,
particles dance in the void,
nothing stays at rest.

the arrow of time,
entropy always increases,
stars burn and fade out.

waves upon the shore,
frequency hums in the dark,
energy persists.

electrons jump and spin,
orbits are clouds of maybe,
certainty dissolves.

black holes bend the light,
singularity awaits,
time itself stands still.

neutrinos pass through,
ghostly messengers of stars,
barely touching worlds.

dark matter hides well,
unseen scaffolding of space,
holding worlds in place.

strings vibrate below,
dimensions curl up and hide,
theory seeks the truth.

the cosmos expands,
redshift whispers of the bang,
silence fills the void.

symmetry can break,
forces split from ancient one,
cooling makes them new.

spin entangles far,
measurement collapses waves,
spooky action reigns.

relativity,
mass and energy are one,
speed of light the law.

momentum conserved,
collisions trade their motion,
nothing can be lost.

the vacuum fluctuates,
virtual pairs appear and fade,
emptiness is full.

a muon decays,
half life ticks in silent math,
nature counts in chance.

planck length defines,
the smallest scene in the play,
spacetime is granular.

the double slit shows,
interference paints the bands,
watching changes all.";

// --- Build training data ------------------------------------------------------

fn build_dataset() -> (Vec<f32>, Vec<f32>, usize) {
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

        // one-hot encode the window
        let mut one_hot = vec![0.0f32; INPUT_DIM];
        for j in 0..WINDOW {
            let ci = char_to_idx(chars[i - WINDOW + j]).unwrap();
            one_hot[j * VOCAB_SIZE + ci] = 1.0;
        }

        inputs.extend_from_slice(&one_hot);
        targets.push(target_idx.unwrap() as f32);
        count += 1;
    }

    (inputs, targets, count)
}

// --- Temperature sampling -----------------------------------------------------

fn sample_with_temperature(probs: &[f32], temperature: f32, hash_seed: u64) -> usize {
    // Apply temperature to log-probs, then re-normalize
    let log_probs: Vec<f32> = probs.iter().map(|&p| p.max(1e-12).ln() / temperature).collect();
    let max_lp = log_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let scaled: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

    let r = (hash_seed % 10000) as f32 / 10000.0;
    let mut cumulative = 0.0;
    for i in 0..VOCAB_SIZE {
        cumulative += scaled[i];
        if r < cumulative {
            return i;
        }
    }
    VOCAB_SIZE - 1
}

// --- Main ---------------------------------------------------------------------

fn main() {
    println!("=== The Quantum Poet (GPU) ===\n");

    let device = GpuDevice::new_sync().expect("failed to create GPU device");
    let mut cache = KernelCache::new();

    // Build training data
    let (input_data, target_data, n_samples) = build_dataset();
    println!(
        "corpus: {} chars, {} training samples, {} window",
        CORPUS.len(),
        n_samples,
        WINDOW,
    );

    // GpuDataLoader expects target_dim — for class indices it's 1
    let mut loader = GpuDataLoader::new(input_data, target_data, INPUT_DIM, 1, 32);

    // Build model: Linear(240,64) -> Tanh -> Linear(64,64) -> Tanh -> Linear(64,30)
    let mut model = GpuSequential::new(vec![
        Box::new(GpuLinear::kaiming(&device, INPUT_DIM, 64, 42)),
        Box::new(GpuTanhLayer::new()),
        Box::new(GpuLinear::kaiming(&device, 64, 64, 137)),
        Box::new(GpuTanhLayer::new()),
        Box::new(GpuLinear::kaiming(&device, 64, VOCAB_SIZE, 256)),
    ]);

    let n_params: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("model: {} parameters\n", n_params);

    // Train
    println!("training...");
    let mut trainer = GpuTrainer::new(0.005, 200)
        .with_loss_fn(gpu_cross_entropy_loss);
    let losses = trainer.fit(&device, &mut cache, &mut model, &mut loader);

    for (i, &loss) in losses.iter().enumerate() {
        if (i + 1) % 20 == 0 || i == 0 {
            println!("  epoch {:>3}: loss = {:.4}", i + 1, loss);
        }
    }

    println!(
        "\nfinal loss: {:.4}\n",
        losses.last().copied().unwrap_or(0.0)
    );

    // --- Generate text --------------------------------------------------------

    let seeds = ["entropy ", "spacetim", "quantum ", "the vacu"];

    for seed in &seeds {
        let mut window: Vec<usize> = seed.chars().map(|c| char_to_idx(c).unwrap()).collect();
        assert_eq!(window.len(), WINDOW);

        let mut output = String::from(*seed);
        let mut hash = 0u64;

        for _ in 0..200 {
            // one-hot encode current window
            let mut input_vec = vec![0.0f32; INPUT_DIM];
            for (j, &ci) in window.iter().enumerate() {
                input_vec[j * VOCAB_SIZE + ci] = 1.0;
            }
            let input = GpuTensor::from_slice(&device, &input_vec, &[1, INPUT_DIM]);

            let logits = model.forward_train(&device, &mut cache, &input);
            cache.flush(&device);
            let logits_data = logits.to_vec_sync(&device);

            // Softmax on CPU for sampling
            let max_val = logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = logits_data.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

            // Deterministic pseudo-random hash from logits
            hash = logits_data.iter().fold(hash, |acc, &v| {
                acc.wrapping_mul(6364136223846793005)
                    .wrapping_add((v.to_bits() as u64) ^ 0xdeadbeef)
            });

            let next_idx = sample_with_temperature(&probs, 0.8, hash);
            let next_char = idx_to_char(next_idx);
            output.push(next_char);

            window.remove(0);
            window.push(next_idx);
        }

        println!("--- seed: {:?} ---", seed);
        println!("{}\n", output);
    }
}
