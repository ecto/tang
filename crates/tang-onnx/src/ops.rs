//! ONNX operator definitions.

/// ONNX operators supported by tang.
#[derive(Clone, Debug, PartialEq)]
pub enum OnnxOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Pow,
    Reciprocal,

    // Linear algebra
    MatMul,
    Gemm {
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    },

    // Convolution
    Conv {
        kernel_shape: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
        dilations: Vec<usize>,
        group: usize,
    },
    ConvTranspose {
        kernel_shape: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
        output_padding: Vec<usize>,
        group: usize,
    },

    // Activations
    Relu,
    Sigmoid,
    Tanh,
    LeakyRelu {
        alpha: f32,
    },
    Elu {
        alpha: f32,
    },
    Selu {
        alpha: f32,
        gamma: f32,
    },
    Gelu,
    Silu,
    Mish,
    Softmax {
        axis: i64,
    },
    LogSoftmax {
        axis: i64,
    },

    // Normalization
    BatchNormalization {
        epsilon: f32,
        momentum: f32,
    },
    LayerNormalization {
        axis: i64,
        epsilon: f32,
    },
    GroupNormalization {
        num_groups: usize,
        epsilon: f32,
    },
    InstanceNormalization {
        epsilon: f32,
    },

    // Pooling
    MaxPool {
        kernel_shape: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
    },
    AveragePool {
        kernel_shape: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
    },
    GlobalAveragePool,

    // Shape manipulation
    Reshape,
    Transpose {
        perm: Vec<usize>,
    },
    Flatten {
        axis: i64,
    },
    Squeeze {
        axes: Vec<i64>,
    },
    Unsqueeze {
        axes: Vec<i64>,
    },
    Concat {
        axis: i64,
    },
    Split {
        axis: i64,
        split: Vec<usize>,
    },
    Gather {
        axis: i64,
    },
    Slice,
    Pad,

    // Reduction
    ReduceMean {
        axes: Vec<i64>,
        keepdims: bool,
    },
    ReduceSum {
        axes: Vec<i64>,
        keepdims: bool,
    },
    ReduceMax {
        axes: Vec<i64>,
        keepdims: bool,
    },

    // Recurrent
    LSTM {
        hidden_size: usize,
        direction: String,
    },
    GRU {
        hidden_size: usize,
        direction: String,
    },

    // Tensor creation
    Constant,
    ConstantOfShape,
    Shape,
    Cast {
        to: i32,
    },

    // Comparison
    Equal,
    Greater,
    Less,
    Where,
    Clip,

    // Other
    Dropout {
        ratio: f32,
    },
    Identity,
    Resize {
        mode: String,
    },

    /// Custom/unknown op — stores the op_type string.
    Custom(String),
}

impl OnnxOp {
    /// Returns the ONNX op_type string.
    pub fn op_type(&self) -> &str {
        match self {
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::Neg => "Neg",
            Self::Abs => "Abs",
            Self::Sqrt => "Sqrt",
            Self::Exp => "Exp",
            Self::Log => "Log",
            Self::Pow => "Pow",
            Self::Reciprocal => "Reciprocal",
            Self::MatMul => "MatMul",
            Self::Gemm { .. } => "Gemm",
            Self::Conv { .. } => "Conv",
            Self::ConvTranspose { .. } => "ConvTranspose",
            Self::Relu => "Relu",
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
            Self::LeakyRelu { .. } => "LeakyRelu",
            Self::Elu { .. } => "Elu",
            Self::Selu { .. } => "Selu",
            Self::Gelu => "Gelu",
            Self::Silu => "Silu",
            Self::Mish => "Mish",
            Self::Softmax { .. } => "Softmax",
            Self::LogSoftmax { .. } => "LogSoftmax",
            Self::BatchNormalization { .. } => "BatchNormalization",
            Self::LayerNormalization { .. } => "LayerNormalization",
            Self::GroupNormalization { .. } => "GroupNormalization",
            Self::InstanceNormalization { .. } => "InstanceNormalization",
            Self::MaxPool { .. } => "MaxPool",
            Self::AveragePool { .. } => "AveragePool",
            Self::GlobalAveragePool => "GlobalAveragePool",
            Self::Reshape => "Reshape",
            Self::Transpose { .. } => "Transpose",
            Self::Flatten { .. } => "Flatten",
            Self::Squeeze { .. } => "Squeeze",
            Self::Unsqueeze { .. } => "Unsqueeze",
            Self::Concat { .. } => "Concat",
            Self::Split { .. } => "Split",
            Self::Gather { .. } => "Gather",
            Self::Slice => "Slice",
            Self::Pad => "Pad",
            Self::ReduceMean { .. } => "ReduceMean",
            Self::ReduceSum { .. } => "ReduceSum",
            Self::ReduceMax { .. } => "ReduceMax",
            Self::LSTM { .. } => "LSTM",
            Self::GRU { .. } => "GRU",
            Self::Constant => "Constant",
            Self::ConstantOfShape => "ConstantOfShape",
            Self::Shape => "Shape",
            Self::Cast { .. } => "Cast",
            Self::Equal => "Equal",
            Self::Greater => "Greater",
            Self::Less => "Less",
            Self::Where => "Where",
            Self::Clip => "Clip",
            Self::Dropout { .. } => "Dropout",
            Self::Identity => "Identity",
            Self::Resize { .. } => "Resize",
            Self::Custom(s) => s,
        }
    }

    /// Parse an ONNX op_type string into a default-attributed op.
    pub fn from_op_type(s: &str) -> Self {
        match s {
            "Add" => Self::Add,
            "Sub" => Self::Sub,
            "Mul" => Self::Mul,
            "Div" => Self::Div,
            "Neg" => Self::Neg,
            "Abs" => Self::Abs,
            "Sqrt" => Self::Sqrt,
            "Exp" => Self::Exp,
            "Log" => Self::Log,
            "Pow" => Self::Pow,
            "Reciprocal" => Self::Reciprocal,
            "MatMul" => Self::MatMul,
            "Gemm" => Self::Gemm {
                alpha: 1.0,
                beta: 1.0,
                trans_a: false,
                trans_b: false,
            },
            "Relu" => Self::Relu,
            "Sigmoid" => Self::Sigmoid,
            "Tanh" => Self::Tanh,
            "Gelu" => Self::Gelu,
            "Silu" => Self::Silu,
            "Mish" => Self::Mish,
            "Softmax" => Self::Softmax { axis: -1 },
            "LogSoftmax" => Self::LogSoftmax { axis: -1 },
            "BatchNormalization" => Self::BatchNormalization {
                epsilon: 1e-5,
                momentum: 0.9,
            },
            "LayerNormalization" => Self::LayerNormalization {
                axis: -1,
                epsilon: 1e-5,
            },
            "GlobalAveragePool" => Self::GlobalAveragePool,
            "Reshape" => Self::Reshape,
            "Flatten" => Self::Flatten { axis: 1 },
            "Concat" => Self::Concat { axis: 0 },
            "Constant" => Self::Constant,
            "Shape" => Self::Shape,
            "Equal" => Self::Equal,
            "Greater" => Self::Greater,
            "Less" => Self::Less,
            "Where" => Self::Where,
            "Clip" => Self::Clip,
            "Identity" => Self::Identity,
            "Slice" => Self::Slice,
            "Pad" => Self::Pad,
            other => Self::Custom(other.to_string()),
        }
    }
}
