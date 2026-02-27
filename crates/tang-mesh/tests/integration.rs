//! End-to-end integration tests for tang-mesh.
//!
//! Tests the full path: QUIC transport → compile → partition → pipeline inference.

use tang_mesh::transport::{MeshTransport, ALPN};
use tang_mesh::worker::Worker;
use tang_mesh::coordinator::Coordinator;
use tang_mesh::inference::InferenceServer;
use tang_mesh::mesh::NodeId;
use tang_mesh::protocol::{WireGraph, WireNode, PROTOCOL_VERSION};

use iroh::endpoint::RelayMode;
use iroh::Endpoint;

fn simple_add_graph() -> WireGraph {
    WireGraph {
        version: PROTOCOL_VERSION,
        nodes: vec![
            WireNode::Lit(0.0_f64.to_bits()),
            WireNode::Lit(1.0_f64.to_bits()),
            WireNode::Lit(2.0_f64.to_bits()),
            WireNode::Var(0),
            WireNode::Var(1),
            WireNode::Add(3, 4),
        ],
        outputs: vec![5],
        n_inputs: 2,
    }
}

/// Graph for pipeline: (x0 + x1) * 2.0
fn pipeline_graph() -> WireGraph {
    WireGraph {
        version: PROTOCOL_VERSION,
        nodes: vec![
            WireNode::Var(0),
            WireNode::Var(1),
            WireNode::Add(0, 1),
            WireNode::Lit(2.0_f64.to_bits()),
            WireNode::Mul(2, 3),
        ],
        outputs: vec![4],
        n_inputs: 2,
    }
}

#[tokio::test]
async fn quic_compile_and_execute() {
    // Create two iroh endpoints (relay disabled for localhost)
    let ep_worker = Endpoint::empty_builder(RelayMode::Disabled)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
        .unwrap();
    let ep_coord = Endpoint::empty_builder(RelayMode::Disabled)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
        .unwrap();

    let worker_addr = ep_worker.addr();

    // Worker serves tarpc over QUIC
    let worker = Worker::new();
    let transport = MeshTransport::from_endpoint(ep_worker);
    let worker_clone = worker.clone();
    tokio::spawn(async move {
        worker_clone.serve(&transport).await.ok();
    });

    // Give the worker a moment to start accepting
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Coordinator connects to worker via QUIC
    let conn = ep_coord.connect(worker_addr, ALPN).await.unwrap();
    let (send, recv) = conn.open_bi().await.unwrap();
    let stream = tang_mesh::transport::QuicStream::new(send, recv);
    let transport = tang_mesh::transport::tarpc_transport(stream);
    let client = tang_mesh::transport::WorkerServiceClient::new(
        tarpc::client::Config::default(),
        transport,
    )
    .spawn();

    // Compile and execute
    let graph = simple_add_graph();
    let result = client
        .compile_graph(tarpc::context::current(), 1, graph)
        .await
        .unwrap();
    assert!(result.is_ok());

    let result = client
        .execute(tarpc::context::current(), 1, vec![3.0, 4.0])
        .await
        .unwrap();
    let output = result.unwrap();
    assert!((output[0] - 7.0).abs() < 1e-5);

    // Verify ping works over QUIC too
    let seq = client.ping(tarpc::context::current(), 99).await.unwrap();
    assert_eq!(seq, 99);
}

#[tokio::test]
async fn quic_two_workers_same_result() {
    // Create 3 iroh endpoints (2 workers + 1 coordinator)
    let ep_w1 = Endpoint::empty_builder(RelayMode::Disabled)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
        .unwrap();
    let ep_w2 = Endpoint::empty_builder(RelayMode::Disabled)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
        .unwrap();
    let ep_coord = Endpoint::empty_builder(RelayMode::Disabled)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
        .unwrap();

    let w1_addr = ep_w1.addr();
    let w2_addr = ep_w2.addr();

    // Start workers
    let w1 = Worker::new();
    let w2 = Worker::new();
    let t1 = MeshTransport::from_endpoint(ep_w1);
    let t2 = MeshTransport::from_endpoint(ep_w2);

    let w1c = w1.clone();
    tokio::spawn(async move { w1c.serve(&t1).await.ok(); });
    let w2c = w2.clone();
    tokio::spawn(async move { w2c.serve(&t2).await.ok(); });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Connect coordinator to both workers via QUIC
    let coordinator = Coordinator::new();

    // Worker 1
    let conn1 = ep_coord.connect(w1_addr, ALPN).await.unwrap();
    let (s1, r1) = conn1.open_bi().await.unwrap();
    let stream1 = tang_mesh::transport::QuicStream::new(s1, r1);
    let transport1 = tang_mesh::transport::tarpc_transport(stream1);
    let client1 = tang_mesh::transport::WorkerServiceClient::new(
        tarpc::client::Config::default(),
        transport1,
    )
    .spawn();
    coordinator.add_worker(NodeId(0), client1).await;

    // Worker 2
    let conn2 = ep_coord.connect(w2_addr, ALPN).await.unwrap();
    let (s2, r2) = conn2.open_bi().await.unwrap();
    let stream2 = tang_mesh::transport::QuicStream::new(s2, r2);
    let transport2 = tang_mesh::transport::tarpc_transport(stream2);
    let client2 = tang_mesh::transport::WorkerServiceClient::new(
        tarpc::client::Config::default(),
        transport2,
    )
    .spawn();
    coordinator.add_worker(NodeId(1), client2).await;

    assert_eq!(coordinator.num_workers().await, 2);

    // Compile on both
    let graph = simple_add_graph();
    let task_id = coordinator.compile_all(&graph).await.unwrap();

    // Execute on each — should get the same result
    let r1 = coordinator
        .execute_on(NodeId(0), task_id, vec![3.0, 4.0])
        .await
        .unwrap();
    let r2 = coordinator
        .execute_on(NodeId(1), task_id, vec![3.0, 4.0])
        .await
        .unwrap();

    assert!((r1[0] - 7.0).abs() < 1e-5);
    assert!((r2[0] - 7.0).abs() < 1e-5);
    assert_eq!(r1, r2);
}

#[tokio::test]
async fn quic_pipeline_inference() {
    // Create 3 endpoints
    let ep_w1 = Endpoint::empty_builder(RelayMode::Disabled)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
        .unwrap();
    let ep_w2 = Endpoint::empty_builder(RelayMode::Disabled)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
        .unwrap();
    let ep_coord = Endpoint::empty_builder(RelayMode::Disabled)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
        .unwrap();

    let w1_addr = ep_w1.addr();
    let w2_addr = ep_w2.addr();

    let w1 = Worker::new();
    let w2 = Worker::new();
    let t1 = MeshTransport::from_endpoint(ep_w1);
    let t2 = MeshTransport::from_endpoint(ep_w2);

    let w1c = w1.clone();
    tokio::spawn(async move { w1c.serve(&t1).await.ok(); });
    let w2c = w2.clone();
    tokio::spawn(async move { w2c.serve(&t2).await.ok(); });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let coordinator = Coordinator::new();

    // Connect both workers
    let conn1 = ep_coord.connect(w1_addr, ALPN).await.unwrap();
    let (s1, r1) = conn1.open_bi().await.unwrap();
    let client1 = tang_mesh::transport::WorkerServiceClient::new(
        tarpc::client::Config::default(),
        tang_mesh::transport::tarpc_transport(
            tang_mesh::transport::QuicStream::new(s1, r1),
        ),
    )
    .spawn();
    coordinator.add_worker(NodeId(0), client1).await;

    let conn2 = ep_coord.connect(w2_addr, ALPN).await.unwrap();
    let (s2, r2) = conn2.open_bi().await.unwrap();
    let client2 = tang_mesh::transport::WorkerServiceClient::new(
        tarpc::client::Config::default(),
        tang_mesh::transport::tarpc_transport(
            tang_mesh::transport::QuicStream::new(s2, r2),
        ),
    )
    .spawn();
    coordinator.add_worker(NodeId(1), client2).await;

    // Pipeline inference: (x0 + x1) * 2.0
    let mut server = InferenceServer::from_coordinator(coordinator);
    let graph = pipeline_graph();
    let mesh = tang_mesh::Mesh::mock(2);
    server.load_model("test", graph, &mesh).await.unwrap();

    let result = server.infer("test", vec![3.0, 4.0]).await.unwrap();
    assert!((result[0] - 14.0).abs() < 1e-5);
}
