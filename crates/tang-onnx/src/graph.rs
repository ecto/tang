//! ONNX computation graph representation.

use crate::ops::OnnxOp;

/// Tensor data for initializers and constants.
#[derive(Clone, Debug)]
pub struct OnnxTensor {
    /// Tensor name.
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Raw f32 data.
    pub data: Vec<f32>,
}

impl OnnxTensor {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            shape,
            data,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Type/shape info for graph inputs and outputs.
#[derive(Clone, Debug)]
pub struct OnnxValueInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub elem_type: ElemType,
}

/// Element type for ONNX tensors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ElemType {
    Float32,
    Float64,
    Float16,
    Int32,
    Int64,
    Int8,
    Uint8,
    Bool,
}

impl ElemType {
    /// ONNX element type integer code.
    pub fn onnx_code(self) -> i32 {
        match self {
            Self::Float32 => 1,
            Self::Uint8 => 2,
            Self::Int8 => 3,
            Self::Int32 => 6,
            Self::Int64 => 7,
            Self::Float16 => 10,
            Self::Float64 => 11,
            Self::Bool => 9,
        }
    }

    pub fn from_onnx_code(code: i32) -> Option<Self> {
        match code {
            1 => Some(Self::Float32),
            2 => Some(Self::Uint8),
            3 => Some(Self::Int8),
            6 => Some(Self::Int32),
            7 => Some(Self::Int64),
            9 => Some(Self::Bool),
            10 => Some(Self::Float16),
            11 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// A node in the ONNX computation graph.
#[derive(Clone, Debug)]
pub struct OnnxNode {
    /// Node name (optional, for debugging).
    pub name: String,
    /// The operation type.
    pub op: OnnxOp,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
}

impl OnnxNode {
    pub fn new(
        name: impl Into<String>,
        op: OnnxOp,
        inputs: Vec<impl Into<String>>,
        outputs: Vec<impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            op,
            inputs: inputs.into_iter().map(Into::into).collect(),
            outputs: outputs.into_iter().map(Into::into).collect(),
        }
    }
}

/// An ONNX computation graph.
#[derive(Clone, Debug)]
pub struct OnnxGraph {
    /// Model name.
    pub name: String,
    /// ONNX opset version.
    pub opset_version: i64,
    /// Graph input descriptions.
    pub inputs: Vec<OnnxValueInfo>,
    /// Graph output descriptions.
    pub outputs: Vec<OnnxValueInfo>,
    /// Computation nodes in topological order.
    pub nodes: Vec<OnnxNode>,
    /// Weight/bias initializers.
    pub initializers: Vec<OnnxTensor>,
}

impl OnnxGraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            opset_version: 17,
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            initializers: Vec::new(),
        }
    }

    /// Add a graph input.
    pub fn add_input(&mut self, name: impl Into<String>, shape: Vec<usize>) {
        let name = name.into();
        self.inputs.push(OnnxValueInfo {
            name,
            shape,
            elem_type: ElemType::Float32,
        });
    }

    /// Add a graph output.
    pub fn add_output(&mut self, name: impl Into<String>, shape: Vec<usize>) {
        let name = name.into();
        self.outputs.push(OnnxValueInfo {
            name,
            shape,
            elem_type: ElemType::Float32,
        });
    }

    /// Add a computation node.
    pub fn add_node(&mut self, node: OnnxNode) {
        self.nodes.push(node);
    }

    /// Add a weight initializer.
    pub fn add_initializer(&mut self, tensor: OnnxTensor) {
        self.initializers.push(tensor);
    }

    /// Get a node by name.
    pub fn get_node(&self, name: &str) -> Option<&OnnxNode> {
        self.nodes.iter().find(|n| n.name == name)
    }

    /// Get an initializer by name.
    pub fn get_initializer(&self, name: &str) -> Option<&OnnxTensor> {
        self.initializers.iter().find(|t| t.name == name)
    }

    /// Total number of parameters across all initializers.
    pub fn num_parameters(&self) -> usize {
        self.initializers.iter().map(|t| t.numel()).sum()
    }

    /// Topological sort validation: checks that every node input is either
    /// a graph input, an initializer, or an output of a preceding node.
    pub fn validate_topology(&self) -> Result<(), String> {
        let mut known: std::collections::HashSet<&str> = std::collections::HashSet::new();

        // Register inputs and initializers
        for inp in &self.inputs {
            known.insert(&inp.name);
        }
        for init in &self.initializers {
            known.insert(&init.name);
        }

        // Walk nodes
        for node in &self.nodes {
            for input in &node.inputs {
                if !input.is_empty() && !known.contains(input.as_str()) {
                    return Err(format!(
                        "Node '{}' references unknown input '{}'",
                        node.name, input
                    ));
                }
            }
            for output in &node.outputs {
                known.insert(output);
            }
        }

        // Check graph outputs are produced
        for out in &self.outputs {
            if !known.contains(out.name.as_str()) {
                return Err(format!("Graph output '{}' is never produced", out.name));
            }
        }

        Ok(())
    }

    /// Convert a tang-train Linear layer to ONNX nodes.
    pub fn from_linear(
        &mut self,
        name: &str,
        input_name: &str,
        output_name: &str,
        weight: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
    ) {
        let w_name = format!("{name}.weight");
        self.add_initializer(OnnxTensor::new(
            &w_name,
            vec![out_features, in_features],
            weight.to_vec(),
        ));

        if let Some(bias_data) = bias {
            let b_name = format!("{name}.bias");
            self.add_initializer(OnnxTensor::new(
                &b_name,
                vec![out_features],
                bias_data.to_vec(),
            ));
            self.add_node(OnnxNode::new(
                name,
                OnnxOp::Gemm {
                    alpha: 1.0,
                    beta: 1.0,
                    trans_a: false,
                    trans_b: true,
                },
                vec![input_name.to_string(), w_name, b_name],
                vec![output_name.to_string()],
            ));
        } else {
            self.add_node(OnnxNode::new(
                name,
                OnnxOp::MatMul,
                vec![input_name.to_string(), w_name],
                vec![output_name.to_string()],
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_graph() {
        let mut g = OnnxGraph::new("test");
        g.add_input("x", vec![1, 784]);
        g.add_output("out", vec![1, 10]);

        g.from_linear("fc1", "x", "h1", &vec![0.0; 784 * 128], None, 784, 128);
        g.add_node(OnnxNode::new(
            "relu1",
            OnnxOp::Relu,
            vec!["h1"],
            vec!["h1_act"],
        ));
        g.from_linear(
            "fc2",
            "h1_act",
            "out",
            &vec![0.0; 128 * 10],
            Some(&vec![0.0; 10]),
            128,
            10,
        );

        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.initializers.len(), 3); // fc1.weight + fc2.weight + fc2.bias
        assert!(g.validate_topology().is_ok());
    }

    #[test]
    fn topology_validation() {
        let mut g = OnnxGraph::new("bad");
        g.add_input("x", vec![1]);
        g.add_node(OnnxNode::new(
            "n1",
            OnnxOp::Relu,
            vec!["missing"],
            vec!["y"],
        ));
        assert!(g.validate_topology().is_err());
    }

    #[test]
    fn op_type_roundtrip() {
        let ops = vec!["Add", "MatMul", "Relu", "Softmax", "BatchNormalization", "LSTM"];
        for name in ops {
            let op = OnnxOp::from_op_type(name);
            assert_eq!(op.op_type(), name);
        }
    }

    #[test]
    fn num_parameters() {
        let mut g = OnnxGraph::new("test");
        g.add_initializer(OnnxTensor::new("w1", vec![10, 5], vec![0.0; 50]));
        g.add_initializer(OnnxTensor::new("b1", vec![10], vec![0.0; 10]));
        assert_eq!(g.num_parameters(), 60);
    }
}
