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
