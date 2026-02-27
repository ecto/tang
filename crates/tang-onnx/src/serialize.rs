//! Compact binary serialization for ONNX graphs.
//!
//! This is a custom binary format (not protobuf) for self-contained
//! round-tripping of tang ONNX graphs. For interop with real ONNX tools,
//! use the protobuf-based export (requires the `onnx-proto` feature).

use crate::graph::{ElemType, OnnxGraph, OnnxNode, OnnxTensor, OnnxValueInfo};
use crate::ops::OnnxOp;

/// Magic bytes identifying a tang-onnx binary file.
pub const ONNX_MAGIC: [u8; 4] = *b"TONX";
const VERSION: u32 = 1;

/// Serialization errors.
#[derive(Debug)]
pub enum SerializeError {
    InvalidMagic,
    UnsupportedVersion(u32),
    UnexpectedEof,
    InvalidUtf8,
    InvalidData(String),
}

impl std::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
            Self::UnexpectedEof => write!(f, "unexpected end of data"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 string"),
            Self::InvalidData(msg) => write!(f, "invalid data: {msg}"),
        }
    }
}

impl std::error::Error for SerializeError {}

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_str(&mut self, s: &str) {
        self.write_u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }

    fn write_usize_vec(&mut self, v: &[usize]) {
        self.write_u32(v.len() as u32);
        for &x in v {
            self.write_u32(x as u32);
        }
    }

    #[allow(dead_code)]
    fn write_i64_vec(&mut self, v: &[i64]) {
        self.write_u32(v.len() as u32);
        for &x in v {
            self.write_i64(x);
        }
    }

    fn write_str_vec(&mut self, v: &[String]) {
        self.write_u32(v.len() as u32);
        for s in v {
            self.write_str(s);
        }
    }

    fn write_f32_slice(&mut self, v: &[f32]) {
        self.write_u32(v.len() as u32);
        for &x in v {
            self.write_f32(x);
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], SerializeError> {
        if self.remaining() < n {
            return Err(SerializeError::UnexpectedEof);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32, SerializeError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i64(&mut self) -> Result<i64, SerializeError> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32, SerializeError> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_str(&mut self) -> Result<String, SerializeError> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| SerializeError::InvalidUtf8)
    }

    fn read_usize_vec(&mut self) -> Result<Vec<usize>, SerializeError> {
        let len = self.read_u32()? as usize;
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(self.read_u32()? as usize);
        }
        Ok(v)
    }

    #[allow(dead_code)]
    fn read_i64_vec(&mut self) -> Result<Vec<i64>, SerializeError> {
        let len = self.read_u32()? as usize;
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(self.read_i64()?);
        }
        Ok(v)
    }

    fn read_str_vec(&mut self) -> Result<Vec<String>, SerializeError> {
        let len = self.read_u32()? as usize;
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(self.read_str()?);
        }
        Ok(v)
    }

    fn read_f32_vec(&mut self) -> Result<Vec<f32>, SerializeError> {
        let len = self.read_u32()? as usize;
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(self.read_f32()?);
        }
        Ok(v)
    }
}

// Op serialization tags
const OP_ADD: u8 = 0;
const OP_SUB: u8 = 1;
const OP_MUL: u8 = 2;
const OP_DIV: u8 = 3;
const OP_MATMUL: u8 = 4;
const OP_GEMM: u8 = 5;
const OP_RELU: u8 = 6;
const OP_SIGMOID: u8 = 7;
const OP_TANH: u8 = 8;
const OP_SOFTMAX: u8 = 9;
const OP_RESHAPE: u8 = 10;
const OP_TRANSPOSE: u8 = 11;
const OP_BATCHNORM: u8 = 12;
const OP_CONV: u8 = 13;
const OP_MAXPOOL: u8 = 14;
const OP_AVGPOOL: u8 = 15;
const OP_GLOBALAVGPOOL: u8 = 16;
const OP_FLATTEN: u8 = 17;
const OP_CONCAT: u8 = 18;
const OP_DROPOUT: u8 = 19;
const OP_IDENTITY: u8 = 20;
const OP_GELU: u8 = 21;
const OP_SILU: u8 = 22;
const OP_LAYERNORM: u8 = 23;
const OP_LSTM: u8 = 24;
const OP_GRU: u8 = 25;
const OP_NEG: u8 = 26;
const OP_SQRT: u8 = 27;
const OP_EXP: u8 = 28;
const OP_LOG: u8 = 29;
const OP_CUSTOM: u8 = 255;

fn write_op(w: &mut Writer, op: &OnnxOp) {
    match op {
        OnnxOp::Add => w.buf.push(OP_ADD),
        OnnxOp::Sub => w.buf.push(OP_SUB),
        OnnxOp::Mul => w.buf.push(OP_MUL),
        OnnxOp::Div => w.buf.push(OP_DIV),
        OnnxOp::MatMul => w.buf.push(OP_MATMUL),
        OnnxOp::Gemm {
            alpha,
            beta,
            trans_a,
            trans_b,
        } => {
            w.buf.push(OP_GEMM);
            w.write_f32(*alpha);
            w.write_f32(*beta);
            w.buf.push(*trans_a as u8);
            w.buf.push(*trans_b as u8);
        }
        OnnxOp::Relu => w.buf.push(OP_RELU),
        OnnxOp::Sigmoid => w.buf.push(OP_SIGMOID),
        OnnxOp::Tanh => w.buf.push(OP_TANH),
        OnnxOp::Softmax { axis } => {
            w.buf.push(OP_SOFTMAX);
            w.write_i64(*axis);
        }
        OnnxOp::Reshape => w.buf.push(OP_RESHAPE),
        OnnxOp::Transpose { perm } => {
            w.buf.push(OP_TRANSPOSE);
            w.write_usize_vec(perm);
        }
        OnnxOp::BatchNormalization { epsilon, momentum } => {
            w.buf.push(OP_BATCHNORM);
            w.write_f32(*epsilon);
            w.write_f32(*momentum);
        }
        OnnxOp::Conv {
            kernel_shape,
            strides,
            pads,
            dilations,
            group,
        } => {
            w.buf.push(OP_CONV);
            w.write_usize_vec(kernel_shape);
            w.write_usize_vec(strides);
            w.write_usize_vec(pads);
            w.write_usize_vec(dilations);
            w.write_u32(*group as u32);
        }
        OnnxOp::MaxPool {
            kernel_shape,
            strides,
            pads,
        } => {
            w.buf.push(OP_MAXPOOL);
            w.write_usize_vec(kernel_shape);
            w.write_usize_vec(strides);
            w.write_usize_vec(pads);
        }
        OnnxOp::AveragePool {
            kernel_shape,
            strides,
            pads,
        } => {
            w.buf.push(OP_AVGPOOL);
            w.write_usize_vec(kernel_shape);
            w.write_usize_vec(strides);
            w.write_usize_vec(pads);
        }
        OnnxOp::GlobalAveragePool => w.buf.push(OP_GLOBALAVGPOOL),
        OnnxOp::Flatten { axis } => {
            w.buf.push(OP_FLATTEN);
            w.write_i64(*axis);
        }
        OnnxOp::Concat { axis } => {
            w.buf.push(OP_CONCAT);
            w.write_i64(*axis);
        }
        OnnxOp::Dropout { ratio } => {
            w.buf.push(OP_DROPOUT);
            w.write_f32(*ratio);
        }
        OnnxOp::Identity => w.buf.push(OP_IDENTITY),
        OnnxOp::Gelu => w.buf.push(OP_GELU),
        OnnxOp::Silu => w.buf.push(OP_SILU),
        OnnxOp::LayerNormalization { axis, epsilon } => {
            w.buf.push(OP_LAYERNORM);
            w.write_i64(*axis);
            w.write_f32(*epsilon);
        }
        OnnxOp::LSTM {
            hidden_size,
            direction,
        } => {
            w.buf.push(OP_LSTM);
            w.write_u32(*hidden_size as u32);
            w.write_str(direction);
        }
        OnnxOp::GRU {
            hidden_size,
            direction,
        } => {
            w.buf.push(OP_GRU);
            w.write_u32(*hidden_size as u32);
            w.write_str(direction);
        }
        OnnxOp::Neg => w.buf.push(OP_NEG),
        OnnxOp::Sqrt => w.buf.push(OP_SQRT),
        OnnxOp::Exp => w.buf.push(OP_EXP),
        OnnxOp::Log => w.buf.push(OP_LOG),
        other => {
            w.buf.push(OP_CUSTOM);
            w.write_str(other.op_type());
        }
    }
}

fn read_op(r: &mut Reader) -> Result<OnnxOp, SerializeError> {
    let tag = r.read_bytes(1)?[0];
    match tag {
        OP_ADD => Ok(OnnxOp::Add),
        OP_SUB => Ok(OnnxOp::Sub),
        OP_MUL => Ok(OnnxOp::Mul),
        OP_DIV => Ok(OnnxOp::Div),
        OP_MATMUL => Ok(OnnxOp::MatMul),
        OP_GEMM => {
            let alpha = r.read_f32()?;
            let beta = r.read_f32()?;
            let trans_a = r.read_bytes(1)?[0] != 0;
            let trans_b = r.read_bytes(1)?[0] != 0;
            Ok(OnnxOp::Gemm {
                alpha,
                beta,
                trans_a,
                trans_b,
            })
        }
        OP_RELU => Ok(OnnxOp::Relu),
        OP_SIGMOID => Ok(OnnxOp::Sigmoid),
        OP_TANH => Ok(OnnxOp::Tanh),
        OP_SOFTMAX => {
            let axis = r.read_i64()?;
            Ok(OnnxOp::Softmax { axis })
        }
        OP_RESHAPE => Ok(OnnxOp::Reshape),
        OP_TRANSPOSE => {
            let perm = r.read_usize_vec()?;
            Ok(OnnxOp::Transpose { perm })
        }
        OP_BATCHNORM => {
            let epsilon = r.read_f32()?;
            let momentum = r.read_f32()?;
            Ok(OnnxOp::BatchNormalization { epsilon, momentum })
        }
        OP_CONV => {
            let kernel_shape = r.read_usize_vec()?;
            let strides = r.read_usize_vec()?;
            let pads = r.read_usize_vec()?;
            let dilations = r.read_usize_vec()?;
            let group = r.read_u32()? as usize;
            Ok(OnnxOp::Conv {
                kernel_shape,
                strides,
                pads,
                dilations,
                group,
            })
        }
        OP_MAXPOOL => {
            let kernel_shape = r.read_usize_vec()?;
            let strides = r.read_usize_vec()?;
            let pads = r.read_usize_vec()?;
            Ok(OnnxOp::MaxPool {
                kernel_shape,
                strides,
                pads,
            })
        }
        OP_AVGPOOL => {
            let kernel_shape = r.read_usize_vec()?;
            let strides = r.read_usize_vec()?;
            let pads = r.read_usize_vec()?;
            Ok(OnnxOp::AveragePool {
                kernel_shape,
                strides,
                pads,
            })
        }
        OP_GLOBALAVGPOOL => Ok(OnnxOp::GlobalAveragePool),
        OP_FLATTEN => {
            let axis = r.read_i64()?;
            Ok(OnnxOp::Flatten { axis })
        }
        OP_CONCAT => {
            let axis = r.read_i64()?;
            Ok(OnnxOp::Concat { axis })
        }
        OP_DROPOUT => {
            let ratio = r.read_f32()?;
            Ok(OnnxOp::Dropout { ratio })
        }
        OP_IDENTITY => Ok(OnnxOp::Identity),
        OP_GELU => Ok(OnnxOp::Gelu),
        OP_SILU => Ok(OnnxOp::Silu),
        OP_LAYERNORM => {
            let axis = r.read_i64()?;
            let epsilon = r.read_f32()?;
            Ok(OnnxOp::LayerNormalization { axis, epsilon })
        }
        OP_LSTM => {
            let hidden_size = r.read_u32()? as usize;
            let direction = r.read_str()?;
            Ok(OnnxOp::LSTM {
                hidden_size,
                direction,
            })
        }
        OP_GRU => {
            let hidden_size = r.read_u32()? as usize;
            let direction = r.read_str()?;
            Ok(OnnxOp::GRU {
                hidden_size,
                direction,
            })
        }
        OP_NEG => Ok(OnnxOp::Neg),
        OP_SQRT => Ok(OnnxOp::Sqrt),
        OP_EXP => Ok(OnnxOp::Exp),
        OP_LOG => Ok(OnnxOp::Log),
        OP_CUSTOM => {
            let name = r.read_str()?;
            Ok(OnnxOp::Custom(name))
        }
        _ => Err(SerializeError::InvalidData(format!("unknown op tag: {tag}"))),
    }
}

impl OnnxGraph {
    /// Serialize to compact binary format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer::new();

        // Header
        w.buf.extend_from_slice(&ONNX_MAGIC);
        w.write_u32(VERSION);
        w.write_str(&self.name);
        w.write_i64(self.opset_version);

        // Inputs
        w.write_u32(self.inputs.len() as u32);
        for inp in &self.inputs {
            w.write_str(&inp.name);
            w.write_usize_vec(&inp.shape);
            w.write_u32(inp.elem_type.onnx_code() as u32);
        }

        // Outputs
        w.write_u32(self.outputs.len() as u32);
        for out in &self.outputs {
            w.write_str(&out.name);
            w.write_usize_vec(&out.shape);
            w.write_u32(out.elem_type.onnx_code() as u32);
        }

        // Initializers
        w.write_u32(self.initializers.len() as u32);
        for init in &self.initializers {
            w.write_str(&init.name);
            w.write_usize_vec(&init.shape);
            w.write_f32_slice(&init.data);
        }

        // Nodes
        w.write_u32(self.nodes.len() as u32);
        for node in &self.nodes {
            w.write_str(&node.name);
            write_op(&mut w, &node.op);
            w.write_str_vec(&node.inputs);
            w.write_str_vec(&node.outputs);
        }

        w.buf
    }

    /// Deserialize from compact binary format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, SerializeError> {
        let mut r = Reader::new(data);

        // Header
        let magic = r.read_bytes(4)?;
        if magic != ONNX_MAGIC {
            return Err(SerializeError::InvalidMagic);
        }
        let version = r.read_u32()?;
        if version != VERSION {
            return Err(SerializeError::UnsupportedVersion(version));
        }
        let name = r.read_str()?;
        let opset_version = r.read_i64()?;

        // Inputs
        let n_inputs = r.read_u32()? as usize;
        let mut inputs = Vec::with_capacity(n_inputs);
        for _ in 0..n_inputs {
            let name = r.read_str()?;
            let shape = r.read_usize_vec()?;
            let elem_code = r.read_u32()? as i32;
            let elem_type = ElemType::from_onnx_code(elem_code)
                .ok_or_else(|| SerializeError::InvalidData(format!("unknown elem type: {elem_code}")))?;
            inputs.push(OnnxValueInfo {
                name,
                shape,
                elem_type,
            });
        }

        // Outputs
        let n_outputs = r.read_u32()? as usize;
        let mut outputs = Vec::with_capacity(n_outputs);
        for _ in 0..n_outputs {
            let name = r.read_str()?;
            let shape = r.read_usize_vec()?;
            let elem_code = r.read_u32()? as i32;
            let elem_type = ElemType::from_onnx_code(elem_code)
                .ok_or_else(|| SerializeError::InvalidData(format!("unknown elem type: {elem_code}")))?;
            outputs.push(OnnxValueInfo {
                name,
                shape,
                elem_type,
            });
        }

        // Initializers
        let n_inits = r.read_u32()? as usize;
        let mut initializers = Vec::with_capacity(n_inits);
        for _ in 0..n_inits {
            let name = r.read_str()?;
            let shape = r.read_usize_vec()?;
            let data = r.read_f32_vec()?;
            initializers.push(OnnxTensor { name, shape, data });
        }

        // Nodes
        let n_nodes = r.read_u32()? as usize;
        let mut nodes = Vec::with_capacity(n_nodes);
        for _ in 0..n_nodes {
            let name = r.read_str()?;
            let op = read_op(&mut r)?;
            let node_inputs = r.read_str_vec()?;
            let node_outputs = r.read_str_vec()?;
            nodes.push(OnnxNode {
                name,
                op,
                inputs: node_inputs,
                outputs: node_outputs,
            });
        }

        Ok(OnnxGraph {
            name,
            opset_version,
            inputs,
            outputs,
            nodes,
            initializers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_roundtrip() {
        let mut g = OnnxGraph::new("test_model");
        g.opset_version = 17;
        g.add_input("x", vec![1, 3, 224, 224]);
        g.add_output("y", vec![1, 1000]);

        g.add_initializer(OnnxTensor::new("w", vec![64, 3, 7, 7], vec![0.1; 64 * 3 * 7 * 7]));

        g.add_node(OnnxNode::new(
            "conv1",
            OnnxOp::Conv {
                kernel_shape: vec![7, 7],
                strides: vec![2, 2],
                pads: vec![3, 3, 3, 3],
                dilations: vec![1, 1],
                group: 1,
            },
            vec!["x", "w"],
            vec!["h1"],
        ));
        g.add_node(OnnxNode::new("relu1", OnnxOp::Relu, vec!["h1"], vec!["h1_act"]));
        g.add_node(OnnxNode::new(
            "pool1",
            OnnxOp::GlobalAveragePool,
            vec!["h1_act"],
            vec!["pooled"],
        ));
        g.add_node(OnnxNode::new(
            "gemm1",
            OnnxOp::Gemm {
                alpha: 1.0,
                beta: 0.0,
                trans_a: false,
                trans_b: true,
            },
            vec!["pooled", "fc_w"],
            vec!["y"],
        ));

        let bytes = g.to_bytes();
        let loaded = OnnxGraph::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.name, "test_model");
        assert_eq!(loaded.opset_version, 17);
        assert_eq!(loaded.inputs.len(), 1);
        assert_eq!(loaded.inputs[0].shape, vec![1, 3, 224, 224]);
        assert_eq!(loaded.outputs.len(), 1);
        assert_eq!(loaded.nodes.len(), 4);
        assert_eq!(loaded.initializers.len(), 1);
        assert_eq!(loaded.initializers[0].data.len(), 64 * 3 * 7 * 7);

        // Check op roundtrip
        assert_eq!(loaded.nodes[0].op.op_type(), "Conv");
        assert_eq!(loaded.nodes[1].op.op_type(), "Relu");
        assert_eq!(loaded.nodes[2].op.op_type(), "GlobalAveragePool");
        assert_eq!(loaded.nodes[3].op.op_type(), "Gemm");
    }

    #[test]
    fn serialize_lstm_gru() {
        let mut g = OnnxGraph::new("rnn");
        g.add_node(OnnxNode::new(
            "lstm1",
            OnnxOp::LSTM {
                hidden_size: 256,
                direction: "forward".to_string(),
            },
            vec!["x", "w", "r"],
            vec!["y", "h", "c"],
        ));
        g.add_node(OnnxNode::new(
            "gru1",
            OnnxOp::GRU {
                hidden_size: 128,
                direction: "bidirectional".to_string(),
            },
            vec!["x2"],
            vec!["y2"],
        ));

        let bytes = g.to_bytes();
        let loaded = OnnxGraph::from_bytes(&bytes).unwrap();

        match &loaded.nodes[0].op {
            OnnxOp::LSTM {
                hidden_size,
                direction,
            } => {
                assert_eq!(*hidden_size, 256);
                assert_eq!(direction, "forward");
            }
            _ => panic!("expected LSTM"),
        }
        match &loaded.nodes[1].op {
            OnnxOp::GRU {
                hidden_size,
                direction,
            } => {
                assert_eq!(*hidden_size, 128);
                assert_eq!(direction, "bidirectional");
            }
            _ => panic!("expected GRU"),
        }
    }
}
