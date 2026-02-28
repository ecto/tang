//! Shape-tracked buffer wrapper for ComputeDevice buffers.

use crate::device::{ComputeBuffer, ComputeDevice};

/// A tensor: a device buffer paired with shape metadata.
pub struct ComputeTensor<B: ComputeBuffer> {
    pub buffer: B,
    shape: Vec<usize>,
}

impl<B: ComputeBuffer> ComputeTensor<B> {
    /// Upload f32 data to device with the given shape.
    pub fn from_data<D: ComputeDevice<Buffer = B>>(dev: &D, data: &[f32], shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "data length {} != shape product {}", data.len(), numel);
        Self { buffer: dev.upload(data), shape: shape.to_vec() }
    }

    /// Create a zero-filled tensor on device.
    pub fn zeros<D: ComputeDevice<Buffer = B>>(dev: &D, shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let data = vec![0.0f32; numel];
        Self { buffer: dev.upload(&data), shape: shape.to_vec() }
    }

    /// Wrap an existing buffer with shape metadata.
    pub fn from_buffer(buffer: B, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(buffer.len(), numel, "buffer len {} != shape product {}", buffer.len(), numel);
        Self { buffer, shape }
    }

    /// Zero-copy reshape. Panics if numel doesn't match.
    pub fn reshape(self, new_shape: &[usize]) -> Self {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(self.numel(), new_numel, "reshape: {} != {}", self.numel(), new_numel);
        Self { buffer: self.buffer, shape: new_shape.to_vec() }
    }

    /// Shape of this tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Download contents to CPU.
    pub fn to_vec(&self) -> Vec<f32> {
        self.buffer.to_vec()
    }

    /// Transpose a 2D tensor via CPU roundtrip.
    pub fn transpose_2d<D: ComputeDevice<Buffer = B>>(self, dev: &D) -> Self {
        assert_eq!(self.shape.len(), 2, "transpose_2d requires 2D tensor");
        let rows = self.shape[0];
        let cols = self.shape[1];
        let data = dev.download(&self.buffer);
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        Self::from_data(dev, &out, &[cols, rows])
    }
}
