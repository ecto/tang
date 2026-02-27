//! PINN — solving the Poisson equation on [0,1]² with zero Dirichlet BCs.
//!
//! Manufactured solution: `u*(x,y) = sin(πx)·sin(πy)`
//! Forcing:               `f(x,y) = 2π²·sin(πx)·sin(πy)`
//!
//! Uses raw weight matrices (not `Module`) so the forward pass is generic over
//! `S: Scalar` and works with `Dual<Dual<f64>>` for second derivatives.
//!
//! ```sh
//! cargo run --example pinn_poisson -p tang-train
//! ```

use tang::Scalar;
use tang_tensor::Tensor;
use tang_train::pinn::{collocation_random, laplacian};

// --- Inline LCG PRNG --------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn normal(&mut self) -> f64 {
        // Box-Muller
        let u1 = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.max(1e-30).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// --- Network weights ---------------------------------------------------------

/// A 2-hidden-layer network: (2) → H → tanh → H → tanh → (1)
/// Stored as flat weight/bias vectors for finite-difference gradient computation.
const H: usize = 16;

struct PinnNet {
    // Layer 1: [H, 2] weights + [H] bias
    w1: Vec<f64>,
    b1: Vec<f64>,
    // Layer 2: [H, H] weights + [H] bias
    w2: Vec<f64>,
    b2: Vec<f64>,
    // Layer 3: [1, H] weights + [1] bias
    w3: Vec<f64>,
    b3: Vec<f64>,
}

impl PinnNet {
    fn new(rng: &mut Lcg) -> Self {
        let xavier = |fan_in: usize, fan_out: usize, rng: &mut Lcg| -> Vec<f64> {
            let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
            (0..fan_in * fan_out)
                .map(|_| rng.normal() * scale)
                .collect()
        };
        Self {
            w1: xavier(2, H, rng),
            b1: vec![0.0; H],
            w2: xavier(H, H, rng),
            b2: vec![0.0; H],
            w3: xavier(H, 1, rng),
            b3: vec![0.0; 1],
        }
    }

    /// Flatten all parameters into a single vector.
    fn params(&self) -> Vec<f64> {
        let mut p = Vec::with_capacity(self.n_params());
        p.extend_from_slice(&self.w1);
        p.extend_from_slice(&self.b1);
        p.extend_from_slice(&self.w2);
        p.extend_from_slice(&self.b2);
        p.extend_from_slice(&self.w3);
        p.extend_from_slice(&self.b3);
        p
    }

    /// Load parameters from a flat vector.
    fn load_params(&mut self, p: &[f64]) {
        let mut i = 0;
        let mut take = |n: usize| -> &[f64] {
            let s = &p[i..i + n];
            i += n;
            s
        };
        self.w1 = take(2 * H).to_vec();
        self.b1 = take(H).to_vec();
        self.w2 = take(H * H).to_vec();
        self.b2 = take(H).to_vec();
        self.w3 = take(H).to_vec();
        self.b3 = take(1).to_vec();
    }

    fn n_params(&self) -> usize {
        2 * H + H + H * H + H + H + 1
    }

    /// Generic forward pass: works with f64, Dual<f64>, Dual<Dual<f64>>.
    fn forward<S: Scalar>(&self, input: &Tensor<S>) -> S {
        let x = input.get(&[0]);
        let y = input.get(&[1]);

        // Layer 1: z1 = tanh(W1 @ [x,y] + b1)
        let mut z1 = Vec::with_capacity(H);
        for j in 0..H {
            let w_x = S::from_f64(self.w1[j * 2]);
            let w_y = S::from_f64(self.w1[j * 2 + 1]);
            let b = S::from_f64(self.b1[j]);
            z1.push((w_x * x + w_y * y + b).tanh());
        }

        // Layer 2: z2 = tanh(W2 @ z1 + b2)
        let mut z2 = Vec::with_capacity(H);
        for j in 0..H {
            let mut acc = S::from_f64(self.b2[j]);
            for k in 0..H {
                acc = acc + S::from_f64(self.w2[j * H + k]) * z1[k];
            }
            z2.push(acc.tanh());
        }

        // Layer 3: out = W3 @ z2 + b3
        let mut out = S::from_f64(self.b3[0]);
        for k in 0..H {
            out = out + S::from_f64(self.w3[k]) * z2[k];
        }

        // Enforce zero Dirichlet BCs: multiply by x*(1-x)*y*(1-y)
        let one = S::from_f64(1.0);
        out * x * (one - x) * y * (one - y)
    }
}

// --- Adam state --------------------------------------------------------------

struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
}

impl Adam {
    fn new(n_params: usize, lr: f64) -> Self {
        Self {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// --- Loss computation --------------------------------------------------------

fn compute_loss(net: &PinnNet, interior: &Tensor<f64>) -> f64 {
    let pi = std::f64::consts::PI;

    // PDE residual loss: -∇²u - f = 0
    // BCs are enforced by the x*(1-x)*y*(1-y) mask in the forward pass.
    let n_int = interior.shape()[0];
    let mut pde_loss = 0.0;
    for i in 0..n_int {
        let x_val = interior.get(&[i, 0]);
        let y_val = interior.get(&[i, 1]);
        let pt = Tensor::from_slice(&[x_val, y_val]);

        let lap = laplacian(|input| net.forward(input), &pt);
        let forcing = 2.0 * pi * pi * (pi * x_val).sin() * (pi * y_val).sin();
        let residual = -lap - forcing;
        pde_loss += residual * residual;
    }
    pde_loss / n_int as f64
}

// --- Main --------------------------------------------------------------------

fn main() {
    println!("=== PINN: Poisson Equation ===\n");
    println!("PDE:      -nabla^2 u = f  on [0,1]^2");
    println!("exact:    u(x,y) = sin(pi*x)*sin(pi*y)");
    println!("forcing:  f(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y)\n");

    let mut rng = Lcg::new(42);

    // Generate collocation points (interior only; BCs enforced by mask)
    let interior = collocation_random(100, 2, 0.05, 0.95, 1337);

    // Initialize network
    let mut net = PinnNet::new(&mut rng);
    let n_params = net.n_params();
    println!("network: 2 -> {} -> tanh -> {} -> tanh -> 1", H, H);
    println!("parameters: {}", n_params);
    println!("interior points: 100 (BCs enforced by mask)\n");

    // Training with finite-difference gradients + Adam
    let mut adam = Adam::new(n_params, 0.005);
    let eps = 1e-5;
    let num_epochs = 800;

    println!("training (finite-difference gradients)...");
    for epoch in 0..num_epochs {
        let loss = compute_loss(&net, &interior);

        if (epoch + 1) % 100 == 0 || epoch == 0 {
            println!("  epoch {:>3}: loss = {:.6}", epoch + 1, loss);
        }

        // Compute gradients via forward finite differences
        let mut params = net.params();
        let mut grads = vec![0.0; n_params];
        for i in 0..n_params {
            params[i] += eps;
            net.load_params(&params);
            let loss_plus = compute_loss(&net, &interior);
            params[i] -= eps;
            net.load_params(&params);
            grads[i] = (loss_plus - loss) / eps;
        }

        // Adam update
        adam.step(&mut params, &grads);
        net.load_params(&params);
    }

    // Verification: compare on 10x10 grid
    let pi = std::f64::consts::PI;
    println!("\nverification on 10x10 grid:");
    let mut max_error = 0.0f64;
    let mut total_error = 0.0f64;
    let n_test = 10;
    for i in 1..n_test {
        for j in 1..n_test {
            let x = i as f64 / n_test as f64;
            let y = j as f64 / n_test as f64;
            let pt = Tensor::from_slice(&[x, y]);
            let pred = net.forward::<f64>(&pt);
            let exact = (pi * x).sin() * (pi * y).sin();
            let err = (pred - exact).abs();
            max_error = max_error.max(err);
            total_error += err;
        }
    }
    let avg_error = total_error / ((n_test - 1) * (n_test - 1)) as f64;
    println!("  max error: {:.6}", max_error);
    println!("  avg error: {:.6}", avg_error);

    if max_error < 0.05 {
        println!("\nSUCCESS: max error < 0.05");
    } else {
        println!("\nNOTE: max error = {:.4} (target < 0.05, may need more epochs)", max_error);
    }
}
