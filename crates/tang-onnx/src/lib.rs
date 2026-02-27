//! ONNX model import/export for tang.
//!
//! Provides a lightweight, protobuf-free representation of ONNX computation
//! graphs that can be serialized to/from the ONNX binary format and mapped
//! to/from tang operations.
//!
//! # Supported ops
//!
//! MatMul, Gemm, Conv, Relu, Sigmoid, Tanh, Softmax, Add, Sub, Mul, Div,
//! Reshape, Transpose, Concat, Split, Gather, Squeeze, Unsqueeze, Flatten,
//! BatchNormalization, LayerNormalization, GroupNormalization, Dropout,
//! MaxPool, AveragePool, GlobalAveragePool, LSTM, GRU, and more.
//!
//! # Example
//!
//! ```ignore
//! use tang_onnx::{OnnxGraph, OnnxNode, OnnxOp, OnnxTensor};
//!
//! let mut graph = OnnxGraph::new("my_model");
//! graph.add_input("x", vec![1, 784]);
//! graph.add_node(OnnxNode::new("matmul1", OnnxOp::MatMul, vec!["x", "w1"], vec!["h1"]));
//! graph.add_node(OnnxNode::new("relu1", OnnxOp::Relu, vec!["h1"], vec!["h1_act"]));
//!
//! let bytes = graph.to_bytes();
//! let loaded = OnnxGraph::from_bytes(&bytes).unwrap();
//! ```

mod graph;
mod ops;
mod serialize;

pub use graph::{OnnxGraph, OnnxNode, OnnxTensor, OnnxValueInfo};
pub use ops::OnnxOp;
pub use serialize::{SerializeError, ONNX_MAGIC};
