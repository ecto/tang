//! GpuTensor: GPU buffer with shape metadata.

use crate::buffer::GpuBuffer;
use crate::device::GpuDevice;
use crate::kernel::KernelCache;
use crate::realize::map_elementwise;

/// A tensor stored on the GPU.
pub struct GpuTensor {
    pub(crate) buffer: GpuBuffer,
    pub(crate) shape: Vec<usize>,
}

impl GpuTensor {
    /// Create a GPU tensor from CPU data.
    pub fn from_slice(device: &GpuDevice, data: &[f32], shape: &[usize]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "data length {} != shape product {}",
            data.len(),
            expected
        );
        Self {
            buffer: GpuBuffer::from_slice(device, data),
            shape: shape.to_vec(),
        }
    }

    /// Create an uninitialized GPU tensor.
    pub(crate) fn uninit(device: &GpuDevice, shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        Self {
            buffer: GpuBuffer::uninit(device, len),
            shape: shape.to_vec(),
        }
    }

    /// Download to CPU.
    pub async fn to_vec(&self, device: &GpuDevice) -> Vec<f32> {
        self.buffer.to_vec(device).await
    }

    /// Download to CPU synchronously.
    pub fn to_vec_sync(&self, device: &GpuDevice) -> Vec<f32> {
        self.buffer.to_vec_sync(device)
    }

    /// Download to CPU synchronously, flushing any pending batched commands first.
    pub fn to_vec_flushed(&self, device: &GpuDevice, cache: &mut KernelCache) -> Vec<f32> {
        cache.flush(device);
        self.buffer.to_vec_sync(device)
    }

    /// Shape of this tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Clone this tensor entirely on the GPU (no CPU readback).
    pub fn clone_gpu(&self, device: &GpuDevice) -> GpuTensor {
        GpuTensor {
            buffer: self.buffer.clone_gpu(device),
            shape: self.shape.clone(),
        }
    }

    /// Clone this tensor on GPU, enqueuing the copy into the kernel cache batch.
    pub fn clone_gpu_batched(&self, device: &GpuDevice, cache: &mut KernelCache) -> GpuTensor {
        GpuTensor {
            buffer: self.buffer.clone_gpu_batched(device, cache),
            shape: self.shape.clone(),
        }
    }

    /// Transpose a 2D tensor on GPU: [M, N] -> [N, M].
    pub fn transpose_gpu(&self, device: &GpuDevice, cache: &mut KernelCache) -> GpuTensor {
        assert_eq!(self.ndim(), 2, "transpose requires 2D tensor");
        let m = self.shape[0] as u32;
        let n = self.shape[1] as u32;
        let numel = (m * n) as u32;
        let out = GpuTensor::uninit(device, &[n as usize, m as usize]);

        // Pack [count, M, N, 0] into params — reuse the existing params layout
        // where count = total elements, and we encode M, N in slots 1, 2
        let wgsl = r#"// Transpose: [M, N] -> [N, M]

struct Params {
    count: u32,
    M: u32,
    N: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    let i = idx / params.N;
    let j = idx % params.N;
    outputs[j * params.M + i] = inputs[idx];
}
"#;

        cache.dispatch_with_params(device, wgsl, &self.buffer, &out.buffer, &[numel, m, n, 0]);
        out
    }

    /// Transpose a 2D tensor: [M, N] -> [N, M] (CPU fallback).
    pub fn transpose(&self, device: &GpuDevice) -> GpuTensor {
        assert_eq!(self.ndim(), 2, "transpose requires 2D tensor");
        let m = self.shape[0];
        let n = self.shape[1];
        let data = self.buffer.to_vec_sync(device);
        let mut out = vec![0.0f32; data.len()];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = data[i * n + j];
            }
        }
        GpuTensor::from_slice(device, &out, &[n, m])
    }

    /// Element-wise addition of two tensors with matching shapes.
    pub fn add(&self, device: &GpuDevice, other: &GpuTensor) -> GpuTensor {
        let a_data = self.buffer.to_vec_sync(device);
        let b_data = other.buffer.to_vec_sync(device);
        assert_eq!(a_data.len(), b_data.len());
        let out: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect();
        GpuTensor::from_slice(device, &out, self.shape())
    }

    /// Scale all elements by a scalar factor.
    pub fn scale(&self, device: &GpuDevice, cache: &mut KernelCache, s: f32) -> GpuTensor {
        let scale_tensor = GpuTensor::from_slice(device, &vec![s; self.numel()], self.shape());
        map_elementwise(device, cache, &[self, &scale_tensor], |args| args[0] * args[1])
    }
}
