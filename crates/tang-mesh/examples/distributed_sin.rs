//! Distributed sin(x) fitting — multi-worker function fitting over tang-mesh.
//!
//! Two in-process workers each get a shard of `y = sin(x)` data, compute
//! forward passes, gradient allreduce at coordinator.
//!
//! Model: `w1 * sin(w2 * x + w3) + w4` (4 trainable parameters)
//! Ground truth: `1 * sin(1 * x + 0) + 0`
//!
//! ```sh
//! cargo run --example distributed_sin -p tang-mesh
//! ```

use tang_expr::ExprGraph;
use tang_mesh::{
    AllReduce, Coordinator, NodeId, Worker,
    protocol::WireGraph,
};

// --- Build the model as a WireGraph ------------------------------------------

/// Build `w1 * sin(w2 * x + w3) + w4` as a WireGraph.
///
/// Inputs: x=Var(0), w1=Var(1), w2=Var(2), w3=Var(3), w4=Var(4)
/// Output: w1 * sin(w2 * x + w3) + w4
fn build_model_graph() -> WireGraph {
    let mut g = ExprGraph::new();
    let x = g.var(0);   // input
    let w1 = g.var(1);  // amplitude
    let w2 = g.var(2);  // frequency
    let w3 = g.var(3);  // phase
    let w4 = g.var(4);  // offset

    let w2x = g.mul(w2, x);
    let inner = g.add(w2x, w3);           // w2*x + w3
    let s = g.sin(inner);                  // sin(w2*x + w3)
    let w1s = g.mul(w1, s);
    let out = g.add(w1s, w4);              // w1*sin(...) + w4

    WireGraph::from_expr_graph(&g, &[out], 5)
}

// --- Generate sin(x) data ---------------------------------------------------

fn generate_data(n: usize) -> Vec<(f32, f32)> {
    (0..n)
        .map(|i| {
            let x = -std::f32::consts::PI + 2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32;
            (x, x.sin())
        })
        .collect()
}

// --- Finite-difference gradient computation ----------------------------------

fn compute_loss_and_grads(
    graph: &WireGraph,
    params: &[f32; 4],
    data: &[(f32, f32)],
) -> (f32, [f32; 4]) {
    let (eg, outputs) = graph.to_expr_graph();
    let output = outputs[0];

    // Compute MSE loss
    let eval_at = |w: &[f32; 4], x: f32| -> f32 {
        let inputs = [x as f64, w[0] as f64, w[1] as f64, w[2] as f64, w[3] as f64];
        eg.eval::<f64>(output, &inputs) as f32
    };

    let mse = |w: &[f32; 4]| -> f32 {
        let mut sum = 0.0f32;
        for &(x, y) in data {
            let pred = eval_at(w, x);
            let err = pred - y;
            sum += err * err;
        }
        sum / data.len() as f32
    };

    let loss = mse(params);

    // Finite-difference gradients
    let eps = 1e-4;
    let mut grads = [0.0f32; 4];
    for i in 0..4 {
        let mut p_plus = *params;
        p_plus[i] += eps;
        grads[i] = (mse(&p_plus) - loss) / eps;
    }

    (loss, grads)
}

// --- Main (async) ------------------------------------------------------------

#[tokio::main]
async fn main() {
    println!("=== Distributed sin(x) Fitting ===\n");
    println!("model: w1 * sin(w2 * x + w3) + w4");
    println!("target: sin(x)  =>  w1=1, w2=1, w3=0, w4=0\n");

    // Build model graph
    let graph = build_model_graph();
    println!("graph: {} nodes, {} outputs, {} inputs",
        graph.nodes.len(), graph.outputs.len(), graph.n_inputs);

    // Spawn 2 in-process workers
    let coordinator = Coordinator::new();
    let w1 = Worker::new();
    let w2 = Worker::new();
    coordinator.add_worker(NodeId(0), w1.spawn_channel()).await;
    coordinator.add_worker(NodeId(1), w2.spawn_channel()).await;
    println!("workers: 2 (in-process channel transport)");

    // Compile graph on both workers
    let task_id = coordinator.compile_all(&graph).await.unwrap();
    println!("compiled on all workers (task_id={})\n", task_id);

    // Generate data
    let data = generate_data(100);
    let shard_size = data.len() / 2;
    let shard0: Vec<(f32, f32)> = data[..shard_size].to_vec();
    let shard1: Vec<(f32, f32)> = data[shard_size..].to_vec();

    // Initialize parameters: w1=0.5, w2=0.8, w3=0.1, w4=0.1
    let mut params: [f32; 4] = [0.5, 0.8, 0.1, 0.1];
    let lr: f32 = 0.05;
    let allreduce = AllReduce::mean();
    let num_epochs = 200;

    println!("training: {} epochs, lr={}, data split={}/{}", num_epochs, lr, shard0.len(), shard1.len());
    println!("initial params: w1={:.3}, w2={:.3}, w3={:.3}, w4={:.3}\n", params[0], params[1], params[2], params[3]);

    for epoch in 0..num_epochs {
        // Each worker computes loss and gradients on its shard
        let (loss0, grads0) = compute_loss_and_grads(&graph, &params, &shard0);
        let (loss1, grads1) = compute_loss_and_grads(&graph, &params, &shard1);

        // Forward on workers (verify they can execute the compiled graph)
        if epoch == 0 {
            let test_input = vec![1.0f32, params[0], params[1], params[2], params[3]];
            let result = coordinator.execute_on(NodeId(0), task_id, test_input.clone()).await.unwrap();
            println!("worker 0 verify: f(1.0) = {:.4} (expected: {:.4})", result[0], 1.0f32.sin());
            let result = coordinator.execute_on(NodeId(1), task_id, test_input).await.unwrap();
            println!("worker 1 verify: f(1.0) = {:.4}\n", result[0]);
        }

        // AllReduce gradients
        let reduced_grads = allreduce.reduce(&[grads0.to_vec(), grads1.to_vec()]);
        let avg_loss = (loss0 + loss1) / 2.0;

        // Update parameters (SGD)
        for i in 0..4 {
            params[i] -= lr * reduced_grads[i];
        }

        // Sync updated params to workers
        coordinator
            .sync_params_all(vec![("params".into(), params.to_vec())])
            .await
            .unwrap();

        if (epoch + 1) % 20 == 0 || epoch == 0 {
            println!(
                "epoch {:>3}: loss = {:.6}  w1={:.3} w2={:.3} w3={:.3} w4={:.3}",
                epoch + 1, avg_loss, params[0], params[1], params[2], params[3]
            );
        }
    }

    // Verification
    println!("\n--- Results ---\n");
    println!(
        "fitted:  w1={:.4}, w2={:.4}, w3={:.4}, w4={:.4}",
        params[0], params[1], params[2], params[3]
    );
    println!("target:  w1=1.0000, w2=1.0000, w3=0.0000, w4=0.0000");

    let errors: Vec<f64> = [
        (params[0] - 1.0) as f64,
        (params[1] - 1.0) as f64,
        params[2] as f64,
        params[3] as f64,
    ]
    .iter()
    .map(|e| e.abs())
    .collect();
    println!(
        "errors:  {:.4}, {:.4}, {:.4}, {:.4}",
        errors[0], errors[1], errors[2], errors[3]
    );

    let loss_decreasing = {
        let (loss_first, _) = compute_loss_and_grads(&graph, &[0.5, 0.8, 0.1, 0.1], &data);
        let (loss_final, _) = compute_loss_and_grads(&graph, &params, &data);
        loss_final < loss_first
    };
    println!(
        "\nloss decreasing: {}",
        if loss_decreasing { "YES" } else { "NO" }
    );

    // Shutdown workers
    coordinator.shutdown_all().await;
    println!("workers shut down.");
}
