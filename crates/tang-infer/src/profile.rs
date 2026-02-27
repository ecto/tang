//! Model profiling and memory tracking.
//!
//! Provides tools to instrument forward passes with timing and memory
//! statistics for each layer/operation.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Timing and memory record for a single layer/operation.
#[derive(Clone, Debug)]
pub struct LayerProfile {
    /// Layer name (e.g., "linear_0", "relu_1").
    pub name: String,
    /// Forward pass time in microseconds.
    pub forward_us: u64,
    /// Number of trainable parameters.
    pub num_params: usize,
    /// Number of FLOPs (multiply-accumulate counted as 2).
    pub flops: u64,
    /// Input shape.
    pub input_shape: Vec<usize>,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// Estimated memory for activations in bytes (f32).
    pub activation_bytes: usize,
}

/// Profiler that collects per-layer statistics.
#[derive(Clone, Debug)]
pub struct Profiler {
    layers: Vec<LayerProfile>,
    peak_memory_bytes: usize,
    current_memory_bytes: usize,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            peak_memory_bytes: 0,
            current_memory_bytes: 0,
        }
    }

    /// Record a layer's profile.
    pub fn record(&mut self, profile: LayerProfile) {
        self.current_memory_bytes += profile.activation_bytes;
        if self.current_memory_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = self.current_memory_bytes;
        }
        self.layers.push(profile);
    }

    /// Record a linear layer.
    pub fn record_linear(
        &mut self,
        name: &str,
        in_features: usize,
        out_features: usize,
        batch_size: usize,
        has_bias: bool,
        forward_us: u64,
    ) {
        let num_params = in_features * out_features + if has_bias { out_features } else { 0 };
        let flops = (2 * in_features * out_features * batch_size) as u64;
        let output_numel = batch_size * out_features;
        self.record(LayerProfile {
            name: String::from(name),
            forward_us,
            num_params,
            flops,
            input_shape: alloc::vec![batch_size, in_features],
            output_shape: alloc::vec![batch_size, out_features],
            activation_bytes: output_numel * 4,
        });
    }

    /// Record a conv2d layer.
    pub fn record_conv2d(
        &mut self,
        name: &str,
        batch: usize,
        in_ch: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        out_h: usize,
        out_w: usize,
        forward_us: u64,
    ) {
        let num_params = out_ch * in_ch * kh * kw + out_ch;
        let flops = (2 * in_ch * kh * kw * out_ch * out_h * out_w * batch) as u64;
        let output_numel = batch * out_ch * out_h * out_w;
        self.record(LayerProfile {
            name: String::from(name),
            forward_us,
            num_params,
            flops,
            input_shape: alloc::vec![batch, in_ch, out_h, out_w], // approximate
            output_shape: alloc::vec![batch, out_ch, out_h, out_w],
            activation_bytes: output_numel * 4,
        });
    }

    /// Record an activation or element-wise layer (no parameters).
    pub fn record_elementwise(
        &mut self,
        name: &str,
        shape: &[usize],
        forward_us: u64,
    ) {
        let numel: usize = shape.iter().product();
        self.record(LayerProfile {
            name: String::from(name),
            forward_us,
            num_params: 0,
            flops: numel as u64,
            input_shape: shape.to_vec(),
            output_shape: shape.to_vec(),
            activation_bytes: numel * 4,
        });
    }

    /// Record an attention layer.
    pub fn record_attention(
        &mut self,
        name: &str,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        forward_us: u64,
    ) {
        let d_model = num_heads * head_dim;
        // QKV projection + output projection
        let num_params = 4 * d_model * d_model;
        // QK^T: B*H * S*S*D, softmax: B*H*S*S, AV: B*H*S*S*D
        let flops = (batch * num_heads * (2 * seq_len * seq_len * head_dim + seq_len * seq_len)) as u64;
        self.record(LayerProfile {
            name: String::from(name),
            forward_us,
            num_params,
            flops,
            input_shape: alloc::vec![batch, seq_len, d_model],
            output_shape: alloc::vec![batch, seq_len, d_model],
            activation_bytes: batch * num_heads * seq_len * seq_len * 4,
        });
    }

    /// Get all layer profiles.
    pub fn layers(&self) -> &[LayerProfile] {
        &self.layers
    }

    /// Total forward time in microseconds.
    pub fn total_time_us(&self) -> u64 {
        self.layers.iter().map(|l| l.forward_us).sum()
    }

    /// Total parameters.
    pub fn total_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_params).sum()
    }

    /// Total FLOPs.
    pub fn total_flops(&self) -> u64 {
        self.layers.iter().map(|l| l.flops).sum()
    }

    /// Peak activation memory in bytes.
    pub fn peak_memory_bytes(&self) -> usize {
        self.peak_memory_bytes
    }

    /// Estimated parameter memory in bytes (f32).
    pub fn param_memory_bytes(&self) -> usize {
        self.total_params() * 4
    }

    /// Reset profiler state.
    pub fn reset(&mut self) {
        self.layers.clear();
        self.peak_memory_bytes = 0;
        self.current_memory_bytes = 0;
    }

    /// Generate a summary table.
    pub fn summary(&self) -> String {
        use alloc::fmt::Write;
        let mut s = String::new();

        let _ = writeln!(s, "{:<30} {:>12} {:>10} {:>12} {:>10}",
            "Layer", "Output Shape", "Params", "FLOPs", "Time (μs)");
        let _ = writeln!(s, "{}", "-".repeat(78));

        for l in &self.layers {
            let shape_str = format!("{:?}", l.output_shape);
            let _ = writeln!(s, "{:<30} {:>12} {:>10} {:>12} {:>10}",
                l.name, shape_str, l.num_params, l.flops, l.forward_us);
        }

        let _ = writeln!(s, "{}", "-".repeat(78));
        let _ = writeln!(s, "Total params: {}", self.total_params());
        let _ = writeln!(s, "Total FLOPs: {}", self.total_flops());
        let _ = writeln!(s, "Total time: {} μs", self.total_time_us());
        let _ = writeln!(s, "Param memory: {:.2} MB", self.param_memory_bytes() as f64 / 1_048_576.0);
        let _ = writeln!(s, "Peak activation memory: {:.2} MB", self.peak_memory_bytes() as f64 / 1_048_576.0);

        s
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_basic() {
        let mut p = Profiler::new();
        p.record_linear("fc1", 784, 256, 32, true, 100);
        p.record_elementwise("relu1", &[32, 256], 10);
        p.record_linear("fc2", 256, 10, 32, true, 50);

        assert_eq!(p.layers().len(), 3);
        assert_eq!(p.total_time_us(), 160);
        assert_eq!(p.total_params(), 784 * 256 + 256 + 256 * 10 + 10);
    }

    #[test]
    fn profiler_peak_memory() {
        let mut p = Profiler::new();
        p.record_linear("fc1", 100, 1000, 1, false, 0);
        let after_fc1 = p.peak_memory_bytes();
        p.record_linear("fc2", 1000, 10, 1, false, 0);

        assert!(p.peak_memory_bytes() >= after_fc1);
    }

    #[test]
    fn profiler_summary() {
        let mut p = Profiler::new();
        p.record_linear("fc1", 784, 128, 1, true, 50);
        p.record_elementwise("relu", &[1, 128], 5);
        let s = p.summary();
        assert!(s.contains("fc1"));
        assert!(s.contains("relu"));
        assert!(s.contains("Total params"));
    }

    #[test]
    fn profiler_attention() {
        let mut p = Profiler::new();
        p.record_attention("attn_0", 2, 128, 8, 64, 200);
        assert!(p.total_flops() > 0);
        assert!(p.peak_memory_bytes() > 0);
    }

    #[test]
    fn profiler_reset() {
        let mut p = Profiler::new();
        p.record_linear("fc1", 10, 10, 1, false, 10);
        assert_eq!(p.layers().len(), 1);
        p.reset();
        assert_eq!(p.layers().len(), 0);
        assert_eq!(p.peak_memory_bytes(), 0);
    }
}
