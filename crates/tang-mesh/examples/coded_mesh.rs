//! Coded mesh inference — 2-node QUIC test.
//!
//! Single binary, role-based: coordinator or worker.
//!
//! ```sh
//! # Machine B (remote worker):
//! cargo run -p tang-mesh --example coded_mesh -- worker --node-id 1
//!
//! # Machine A (coordinator + local worker, connects to B):
//! cargo run -p tang-mesh --example coded_mesh -- coordinator --node-id 0 \
//!     --peer <machine_b_iroh_node_id>
//!
//! # Single-machine test (both workers in-process, no QUIC):
//! cargo run -p tang-mesh --example coded_mesh -- local
//! ```

use tang_mesh::coded::{CodedModel, Generator};
use tang_mesh::coordinator::Coordinator;
use tang_mesh::inference::{
    apply_bias, attention_forward, build_coded_transformer, embed_tokens, silu,
    Activation, CodedInferenceServer, TransformerConfig, TransformerWeights,
};
use tang_mesh::mesh::NodeId;
use tang_mesh::transport::MeshTransport;
use tang_mesh::worker::Worker;

// ---------------------------------------------------------------------------
// Deterministic RNG (same across machines given same seed)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32 - 0.5
    }
    fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32() * scale).collect()
    }
}

// ---------------------------------------------------------------------------
// Test config
// ---------------------------------------------------------------------------

const D_MODEL: usize = 64;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16; // d_model / n_heads
const FF_DIM: usize = 128;
const N_LAYERS: usize = 2;
const VOCAB_SIZE: usize = 256;
const EPS: f32 = 1e-5;

fn test_config() -> TransformerConfig {
    TransformerConfig {
        d_model: D_MODEL,
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        ff_dim: FF_DIM,
        n_layers: N_LAYERS,
        vocab_size: VOCAB_SIZE,
        eps: EPS,
    }
}

fn test_weights(rng: &mut Rng) -> TransformerWeights {
    let d_q = N_HEADS * HEAD_DIM;
    let d_k = N_KV_HEADS * HEAD_DIM;
    let s = 0.3; // weight scale

    TransformerWeights {
        block_weights: (0..N_LAYERS)
            .map(|_| {
                [
                    rng.vec(d_q * D_MODEL, s),
                    rng.vec(d_k * D_MODEL, s),
                    rng.vec(d_k * D_MODEL, s),
                    rng.vec(D_MODEL * D_MODEL, s),
                    rng.vec(FF_DIM * D_MODEL, s),
                    rng.vec(FF_DIM * D_MODEL, s),
                    rng.vec(D_MODEL * FF_DIM, s),
                ]
            })
            .collect(),
        block_biases: (0..N_LAYERS)
            .map(|_| {
                [
                    rng.vec(d_q, s),
                    rng.vec(d_k, s),
                    rng.vec(d_k, s),
                    rng.vec(D_MODEL, s),
                    rng.vec(FF_DIM, s),
                    rng.vec(FF_DIM, s),
                    rng.vec(D_MODEL, s),
                ]
            })
            .collect(),
        block_norms: (0..N_LAYERS)
            .map(|_| {
                // Scale initialized near 1.0
                let ln1: Vec<f32> = rng.vec(D_MODEL, 0.1).iter().map(|x| 1.0 + x).collect();
                let ln2: Vec<f32> = rng.vec(D_MODEL, 0.1).iter().map(|x| 1.0 + x).collect();
                (ln1, ln2)
            })
            .collect(),
        ln_final_scale: rng.vec(D_MODEL, 0.1).iter().map(|x| 1.0 + x).collect(),
        lm_head_weight: rng.vec(VOCAB_SIZE * D_MODEL, s),
        lm_head_bias: rng.vec(VOCAB_SIZE, s),
        embed_table: rng.vec(VOCAB_SIZE * D_MODEL, s),
    }
}

// ---------------------------------------------------------------------------
// Uncoded reference
// ---------------------------------------------------------------------------

/// Batched matmul: W (d_out × d_in) @ x [seq_len, d_in] → [seq_len, d_out]
fn matmul_batched(w: &[f32], x: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
    let seq_len = x.len() / d_in;
    let mut out = vec![0.0f32; seq_len * d_out];
    for t in 0..seq_len {
        for r in 0..d_out {
            let mut sum = 0.0f32;
            for c in 0..d_in {
                sum += w[r * d_in + c] * x[t * d_in + c];
            }
            out[t * d_out + r] = sum;
        }
    }
    out
}

fn uncoded_forward(
    config: &TransformerConfig,
    weights: &TransformerWeights,
    tokens: &[u32],
) -> Vec<f32> {
    let d_model = config.d_model;
    let d_q = config.n_heads * config.head_dim;
    let d_k = config.n_kv_heads * config.head_dim;
    let ff_dim = config.ff_dim;
    let seq_len = tokens.len();

    let mut x = embed_tokens(tokens, &weights.embed_table, d_model);

    for i in 0..config.n_layers {
        let [ref wq, ref wk, ref wv, ref wo, ref w_gate, ref w_up, ref w_down] =
            weights.block_weights[i];
        let [ref bq, ref bk, ref bv, ref bo, ref b_gate, ref b_up, ref b_down] =
            weights.block_biases[i];
        let (ref ln1, ref ln2) = weights.block_norms[i];

        // Attn half
        let residual = x.clone();
        Activation::RMSNorm {
            eps: config.eps,
            scale: Some(ln1.clone()),
        }
        .apply(&mut x);

        let mut q = matmul_batched(wq, &x, d_q, d_model);
        let mut k = matmul_batched(wk, &x, d_k, d_model);
        let mut v = matmul_batched(wv, &x, d_k, d_model);
        apply_bias(&mut q, bq);
        apply_bias(&mut k, bk);
        apply_bias(&mut v, bv);

        let d_qkv = d_q + d_k + d_k;
        let mut qkv = vec![0.0f32; seq_len * d_qkv];
        for t in 0..seq_len {
            qkv[t * d_qkv..t * d_qkv + d_q]
                .copy_from_slice(&q[t * d_q..(t + 1) * d_q]);
            qkv[t * d_qkv + d_q..t * d_qkv + d_q + d_k]
                .copy_from_slice(&k[t * d_k..(t + 1) * d_k]);
            qkv[t * d_qkv + d_q + d_k..t * d_qkv + d_qkv]
                .copy_from_slice(&v[t * d_k..(t + 1) * d_k]);
        }
        let (attn_out, _, _, _, _) = attention_forward(
            &qkv,
            seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
        );

        x = matmul_batched(wo, &attn_out, d_model, d_model);
        apply_bias(&mut x, bo);
        for (xi, ri) in x.iter_mut().zip(&residual) {
            *xi += ri;
        }

        // FFN half
        let residual = x.clone();
        Activation::RMSNorm {
            eps: config.eps,
            scale: Some(ln2.clone()),
        }
        .apply(&mut x);

        let mut gate = matmul_batched(w_gate, &x, ff_dim, d_model);
        let mut up = matmul_batched(w_up, &x, ff_dim, d_model);
        apply_bias(&mut gate, b_gate);
        apply_bias(&mut up, b_up);
        let swiglu: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();

        x = matmul_batched(w_down, &swiglu, d_model, ff_dim);
        apply_bias(&mut x, b_down);
        for (xi, ri) in x.iter_mut().zip(&residual) {
            *xi += ri;
        }
    }

    // ln_final + lm_head
    Activation::RMSNorm {
        eps: config.eps,
        scale: Some(weights.ln_final_scale.clone()),
    }
    .apply(&mut x);

    let mut logits = matmul_batched(&weights.lm_head_weight, &x, config.vocab_size, d_model);
    apply_bias(&mut logits, &weights.lm_head_bias);
    logits
}

// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------

async fn run_local() {
    println!("=== Coded Mesh: Local (in-process) ===\n");

    let config = test_config();
    let mut rng = Rng::new(42);
    let weights = test_weights(&mut rng);
    let (layers, all_ws) = build_coded_transformer(&config, &weights);

    let n = 4;
    let k = 2;
    let g = Generator::cauchy(n, k);

    // Set up workers with coded models
    let coordinator = Coordinator::new();
    let mut workers = Vec::new();
    let weight_refs: Vec<&[f32]> = all_ws.iter().map(|w| w.as_slice()).collect();
    for i in 0..n {
        let worker = Worker::new();
        let model = CodedModel::from_layer_weights(&weight_refs, &g, i);
        worker.set_coded_model(model).await;
        coordinator
            .add_worker(NodeId(i as u32), worker.spawn_channel())
            .await;
        workers.push(worker);
    }

    let server = CodedInferenceServer::new(
        coordinator,
        g,
        vec![NodeId(0), NodeId(1)],
        layers,
    );

    let tokens = vec![1u32, 42, 7, 100];
    let input = embed_tokens(&tokens, &weights.embed_table, config.d_model);

    println!(
        "config: d_model={}, n_heads={}, ff_dim={}, n_layers={}, vocab={}",
        D_MODEL, N_HEADS, FF_DIM, N_LAYERS, VOCAB_SIZE
    );
    println!("tokens: {:?}", tokens);
    println!(
        "coded layers: {}, linear shards: {}",
        server.count_linear_layers(),
        all_ws.len()
    );
    println!();

    // Coded inference
    let coded_logits = server.infer(input).await.unwrap();

    // Uncoded reference
    let ref_logits = uncoded_forward(&config, &weights, &tokens);

    // Compare
    let max_diff: f32 = coded_logits
        .iter()
        .zip(&ref_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_diff: f32 = coded_logits
        .iter()
        .zip(&ref_logits)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / coded_logits.len() as f32;

    println!(
        "output shape: [{}, {}]",
        tokens.len(),
        config.vocab_size
    );
    println!("max |coded - uncoded|: {:.6e}", max_diff);
    println!("mean |coded - uncoded|: {:.6e}", mean_diff);

    // Argmax per token
    for t in 0..tokens.len() {
        let start = t * config.vocab_size;
        let end = start + config.vocab_size;
        let coded_argmax = coded_logits[start..end]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let ref_argmax = ref_logits[start..end]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let match_str = if coded_argmax == ref_argmax {
            "MATCH"
        } else {
            "MISMATCH"
        };
        println!(
            "  token[{}]: coded argmax={}, ref argmax={} [{}]",
            t, coded_argmax, ref_argmax, match_str
        );
    }

    let pass = max_diff < 0.1;
    println!(
        "\n{}",
        if pass { "PASS" } else { "FAIL" }
    );
}

async fn run_worker(node_id: usize) {
    println!("=== Coded Mesh: Worker (node {}) ===\n", node_id);

    let config = test_config();
    let mut rng = Rng::new(42);
    let weights = test_weights(&mut rng);
    let (_, all_ws) = build_coded_transformer(&config, &weights);

    let n = 4;
    let k = 2;
    let g = Generator::cauchy(n, k);
    let weight_refs: Vec<&[f32]> = all_ws.iter().map(|w| w.as_slice()).collect();
    let model = CodedModel::from_layer_weights(&weight_refs, &g, node_id);

    let worker = Worker::new();
    worker.set_coded_model(model).await;

    let transport = MeshTransport::new().await.unwrap();
    println!("node ID: {}", transport.node_id());
    println!("waiting for connections...\n");

    worker.serve(&transport).await.ok();
}

async fn run_coordinator(node_id: usize, peer: &str) {
    println!("=== Coded Mesh: Coordinator (node {}) ===\n", node_id);

    let config = test_config();
    let mut rng = Rng::new(42);
    let weights = test_weights(&mut rng);
    let (layers, all_ws) = build_coded_transformer(&config, &weights);

    let n = 4;
    let k = 2;
    let g = Generator::cauchy(n, k);
    let weight_refs: Vec<&[f32]> = all_ws.iter().map(|w| w.as_slice()).collect();

    // Local worker (in-process)
    let local_worker = Worker::new();
    let local_model = CodedModel::from_layer_weights(&weight_refs, &g, node_id);
    local_worker.set_coded_model(local_model).await;

    let coordinator = Coordinator::new();
    coordinator
        .add_worker(NodeId(node_id as u32), local_worker.spawn_channel())
        .await;

    // Connect to remote worker via QUIC
    let transport = MeshTransport::new().await.unwrap();
    let peer_id: iroh::EndpointId = peer.parse().expect("invalid peer node ID");
    println!("connecting to peer {}...", peer);
    let conn = transport.connect(peer_id).await.unwrap();
    let (send, recv) = conn.open_bi().await.unwrap();
    let stream = tang_mesh::transport::QuicStream::new(send, recv);
    let rpc_transport = tang_mesh::transport::tarpc_transport(stream);
    let client = tang_mesh::transport::WorkerServiceClient::new(
        tarpc::client::Config::default(),
        rpc_transport,
    )
    .spawn();

    // Remote worker gets node_id 1 (assume coordinator is 0, worker is 1)
    let remote_node_id = if node_id == 0 { 1u32 } else { 0u32 };
    coordinator
        .add_worker(NodeId(remote_node_id), client)
        .await;
    println!("connected to remote worker as node {}", remote_node_id);

    let server = CodedInferenceServer::new(
        coordinator,
        g,
        vec![NodeId(node_id as u32), NodeId(remote_node_id)],
        layers,
    );

    let tokens = vec![1u32, 42, 7, 100];
    let input = embed_tokens(&tokens, &weights.embed_table, config.d_model);

    println!("running inference...");
    let coded_logits = server.infer(input).await.unwrap();

    let ref_logits = uncoded_forward(&config, &weights, &tokens);
    let max_diff: f32 = coded_logits
        .iter()
        .zip(&ref_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("max |coded - uncoded|: {:.6e}", max_diff);
    println!(
        "{}",
        if max_diff < 0.1 { "PASS" } else { "FAIL" }
    );

    transport.close().await;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn usage() -> ! {
    eprintln!("Usage:");
    eprintln!("  coded_mesh local");
    eprintln!("  coded_mesh worker --node-id <N>");
    eprintln!("  coded_mesh coordinator --node-id <N> --peer <iroh_node_id>");
    std::process::exit(1);
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
    }

    match args[1].as_str() {
        "local" => run_local().await,
        "worker" => {
            let node_id: usize = parse_arg(&args, "--node-id")
                .expect("--node-id required")
                .parse()
                .expect("node-id must be a number");
            run_worker(node_id).await;
        }
        "coordinator" => {
            let node_id: usize = parse_arg(&args, "--node-id")
                .expect("--node-id required")
                .parse()
                .expect("node-id must be a number");
            let peer = parse_arg(&args, "--peer").expect("--peer required");
            run_coordinator(node_id, &peer).await;
        }
        _ => usage(),
    }
}
