//! The Quantum Poet — a character-level text generator trained on physics haikus.
//!
//! Demonstrates the full tang training pipeline: dataset construction,
//! model building, training with cross-entropy loss, and text generation.
//!
//! ```sh
//! cargo run --example quantum_poet -p tang-train
//! ```

use tang_tensor::{Shape, Tensor};
use tang_train::{
    cross_entropy_loss, cross_entropy_loss_grad, softmax, DataLoader, Linear, ModuleAdam, Module,
    Sequential, Tanh, TensorDataset, Trainer,
};

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

fn build_dataset() -> (Vec<f64>, Vec<f64>, usize) {
    let chars: Vec<char> = CORPUS.chars().collect();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut count = 0;

    for i in WINDOW..chars.len() {
        // check that all chars in window + target are in vocab
        let window_ok = (0..WINDOW).all(|j| char_to_idx(chars[i - WINDOW + j]).is_some());
        let target_idx = char_to_idx(chars[i]);
        if !window_ok || target_idx.is_none() {
            continue;
        }

        // one-hot encode the window
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

// --- Temperature sampling -----------------------------------------------------

fn sample_with_temperature(logits: &Tensor<f64>, temperature: f64) -> usize {
    let scaled = logits.scale(1.0 / temperature);
    let probs = softmax(&scaled);

    // simple pseudo-random sampling using the probabilities
    // we use a deterministic but varied seed based on logits
    let hash = logits.data().iter().fold(0u64, |acc, &v| {
        acc.wrapping_mul(6364136223846793005)
            .wrapping_add((v.to_bits()) ^ 0xdeadbeef)
    });

    let r = (hash % 10000) as f64 / 10000.0;
    let mut cumulative = 0.0;
    for i in 0..VOCAB_SIZE {
        cumulative += probs.data()[i];
        if r < cumulative {
            return i;
        }
    }
    VOCAB_SIZE - 1
}

// --- Main ---------------------------------------------------------------------

fn main() {
    println!("=== The Quantum Poet ===\n");

    // build training data
    let (input_data, target_data, n_samples) = build_dataset();
    println!(
        "corpus: {} chars, {} training samples, {} parameters window",
        CORPUS.len(),
        n_samples,
        WINDOW
    );

    let inputs = Tensor::new(input_data, Shape::new(vec![n_samples, INPUT_DIM]));
    let targets = Tensor::new(target_data, Shape::new(vec![n_samples]));
    let dataset = TensorDataset::new(inputs, targets);
    let mut loader = DataLoader::new(&dataset, 32);

    // build model: Linear(240,64) -> Tanh -> Linear(64,64) -> Tanh -> Linear(64,30)
    let mut model = Sequential::<f64>::new(vec![
        Box::new(Linear::new(INPUT_DIM, 64, 42)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(64, 64, 137)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(64, VOCAB_SIZE, 256)),
    ]);

    let n_params: usize = model.parameters().iter().map(|p| p.data.numel()).sum();
    println!("model: {} parameters\n", n_params);

    // train
    println!("training...");
    let losses = Trainer::new(&mut model, ModuleAdam::new(0.005))
        .loss_fn(|pred, target| {
            (
                cross_entropy_loss(pred, target),
                cross_entropy_loss_grad(pred, target),
            )
        })
        .epochs(200)
        .fit(&mut loader);

    for (i, loss) in losses.iter().enumerate() {
        if (i + 1) % 20 == 0 || i == 0 {
            println!("  epoch {:>3}: loss = {:.4}", i + 1, loss);
        }
    }

    println!(
        "\nfinal loss: {:.4}\n",
        losses.last().copied().unwrap_or(0.0)
    );

    // --- Save & reload weights ------------------------------------------------

    let save_path = std::path::Path::new("quantum_poet.safetensors");
    let state = model.state_dict();
    let map: std::collections::HashMap<String, Tensor<f64>> = state.into_iter().collect();
    tang_safetensors::save(&map, save_path).unwrap();
    println!("saved model to {}\n", save_path.display());

    // Load into a fresh model to prove it works
    let loaded = tang_safetensors::load(save_path).unwrap();
    let state: Vec<(String, Tensor<f64>)> = loaded.into_iter().collect();
    model.load_state_dict(&state);

    // Clean up
    std::fs::remove_file(save_path).ok();

    // --- Generate text --------------------------------------------------------

    let seeds = ["entropy ", "spacetim", "quantum ", "the vacu"];

    for seed in &seeds {
        let mut window: Vec<usize> = seed.chars().map(|c| char_to_idx(c).unwrap()).collect();
        assert_eq!(window.len(), WINDOW);

        let mut output = String::from(*seed);

        for _ in 0..200 {
            // one-hot encode current window
            let mut input_data = vec![0.0f64; INPUT_DIM];
            for (j, &ci) in window.iter().enumerate() {
                input_data[j * VOCAB_SIZE + ci] = 1.0;
            }
            let input = Tensor::new(input_data, Shape::new(vec![1, INPUT_DIM]));

            let logits = model.forward(&input);
            // logits shape: [1, VOCAB_SIZE] — extract the row
            let row = Tensor::new(logits.data().to_vec(), Shape::new(vec![VOCAB_SIZE]));

            let next_idx = sample_with_temperature(&row, 0.8);
            let next_char = idx_to_char(next_idx);
            output.push(next_char);

            // slide window
            window.remove(0);
            window.push(next_idx);
        }

        println!("--- seed: {:?} ---", seed);
        println!("{}\n", output);
    }
}
