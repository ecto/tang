# tang-mesh

Distributed compute for tang. Ship expression graphs over the wire — not tensors — and let each worker compile them to GPU kernels locally.

## Architecture

```
                    ┌─────────────┐
                    │ Coordinator │
                    └──────┬──────┘
            partition │  allreduce │  health check
           ┌─────────┼────────────┼─────────┐
           ▼         ▼            ▼         ▼
       ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
       │Worker 0│ │Worker 1│ │Worker 2│ │Worker 3│
       │ Metal  │ │ Vulkan │ │ DX12   │ │  CPU   │
       └────────┘ └────────┘ └────────┘ └────────┘

    Each worker:  WireGraph → WGSL → local compile → dispatch
```

## Training

```rust
use tang_mesh::*;

let mesh = Mesh::builder()
    .node(peer_id_0)
    .node(peer_id_1)
    .build().await?;

let trainer = DistributedTrainer::new(0.001, 100)
    .distribute(&mesh).await?;

trainer.fit_distributed(&graph, &data, &targets).await?;
```

## Inference

```rust
let server = InferenceServer::new(&mesh).await?;
server.load_model("gpt", graph, &mesh).await?;
let output = server.infer("gpt", input).await?;
```

## Parallelism strategies

| Strategy | How it works |
|----------|-------------|
| **DataParallel** | Replicate model, shard data, allreduce gradients |
| **Pipeline** | Partition graph into stages, stream activations |
| **TensorParallel** | Shard parameters across devices |

The partitioner can cut at arbitrary graph nodes (not just layer boundaries), minimizing cross-device activation traffic.

## Networking

Built on [iroh](https://iroh.computer) for QUIC transport with NAT traversal. RPC via tarpc with postcard serialization.

```
iroh (QUIC + holepunching)
  └─ tarpc (async RPC)
       └─ postcard (compact serialization)
```

Optional `mdns` feature for LAN peer discovery.

## Fault tolerance

`HealthMonitor` tracks node liveness via heartbeats:

```
Healthy ──timeout──► Suspect ──timeout──► Dead
                         │                  │
                         └──heartbeat──► Healthy
                                            │
                              FaultHandler re-partitions
```

On node failure, the `FaultHandler` re-partitions the graph across surviving workers without moving weights.

## Wire protocol

`WireGraph` mirrors tang-expr's 11 operations in a serializable format. Protocol version 2 — version mismatches are caught at compile time.

## API

| Component | Purpose |
|-----------|---------|
| `Mesh` / `MeshBuilder` | Cluster topology |
| `Coordinator` | Orchestrate workers |
| `Worker` | Receive, compile, execute graphs |
| `DistributedTrainer` | Data-parallel training with allreduce |
| `InferenceServer` | Pipeline execution |
| `HealthMonitor` / `FaultHandler` | Liveness + recovery |
| `partition()` / `auto_partition()` | Graph splitting |

## Features

| Feature | Default | Purpose |
|---------|---------|---------|
| `mdns` | no | LAN peer discovery via mDNS |

## License

[MIT](../../LICENSE)

## Coded Mesh (upcoming)

Erasure-coded model distribution. The model isn't replicated — it's split into k blocks and encoded via a Cauchy generator matrix so that any k-of-n nodes can reconstruct the full weights. Each node stores model_size / k.

```
Node 0: G[0,:] · W     ─┐
Node 1: G[1,:] · W      │  any k decode
Node 2: G[2,:] · W      ├─────────────► full W
  ...                   │
Node n: G[n,:] · W     ─┘
```

**Inference** runs as coded tensor parallelism — k nodes form an ephemeral group, each computes shard @ x (1/k of the matmul). Linear layers stay fully coded. At non-linearities (GELU, LayerNorm), the group decodes via k×k solve, applies the activation, and re-encodes — ~2 barriers per transformer layer, each O(k² · d) vs O(k · d²) for the matmuls.

**Learning** happens on every inference. The backward pass produces block gradients δW, which are encoded the same way as the weights: δshard_i = G[i,:] · δW. Coding structure is preserved — updates are just linear combinations of linear combinations. Gradients propagate via epidemic gossip (O(log n) rounds to reach all nodes).

**Verification** is inherent. If a node's shard is wrong — didn't apply updates, applied garbage, or is lying about its version — the k-node decode produces nonsense. The other k-1 honest nodes detect this immediately. No blockchain, no tokens, no stake. The linear algebra is the trust layer.

### Prior art

| System | Inference | Training | Fault tolerant | Verification | Incentive |
|--------|-----------|----------|----------------|--------------|-----------|
| [Petals](https://petals.dev/) | pipeline parallel | fine-tune | no — one node dies, pipeline breaks | none | none (declined) |
| [Bittensor](https://bittensor.com/) | via subnets | no | n/a | stake-based ([gamed](https://arxiv.org/html/2507.02951v1)) | TAO tokens (plutocratic) |
| [Prime Intellect](https://www.primeintellect.ai/) | separate | async RL | partial | TOPLOC proofs | open-source ethos |
| [Gensyn](https://www.gensyn.ai/) | no | yes | yes (85-90% at 30% loss) | crypto proofs | payments |
| [Hivemind](https://github.com/learning-at-home/hivemind) | no | yes | no — one bad peer jeopardizes run | none | none |
| **tang-mesh** | coded TP | every inference | MDS-optimal (any k of n) | linear algebra | infer = train |

Key differences:
- **No separation of training and inference.** Every forward pass includes a backward pass. The network improves from its own usage.
- **No tokens, no stake, no blockchain.** Verification comes from erasure coding math: corrupt shards produce detectable garbage deterministically.
- **MDS-optimal fault tolerance.** At n=1M, k=3: tolerates 999,997 simultaneous failures. Any 3 nodes reconstruct the full model.

### References

- Lee et al., [Speeding Up Distributed ML Using Codes](https://arxiv.org/abs/1512.02673), IEEE Trans. IT 2018
- Tandon et al., [Gradient Coding](https://proceedings.mlr.press/v70/tandon17a.html), ICML 2017
- Jhunjhunwala et al., [COIN: Erasure Coded Neural Network Inference](https://arxiv.org/abs/2409.01420), ISIT 2024
- Sun et al., [Test-Time Training with Self-Supervision](https://arxiv.org/abs/1909.13231), ICML 2020
- Sun et al., [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620), ICML 2024
