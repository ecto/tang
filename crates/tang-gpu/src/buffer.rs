//! GPU buffer: upload, download, staging.

use crate::device::GpuDevice;

/// A GPU storage buffer holding f32 values.
pub struct GpuBuffer {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) len: usize,
}

impl GpuBuffer {
    /// Create a storage buffer initialized from a slice.
    pub fn from_slice(device: &GpuDevice, data: &[f32]) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tang-gpu storage"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        Self {
            buffer,
            len: data.len(),
        }
    }

    /// Create an uninitialized storage buffer of `len` f32 elements.
    pub fn uninit(device: &GpuDevice, len: usize) -> Self {
        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tang-gpu storage"),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer, len }
    }

    /// Download buffer contents to CPU.
    pub async fn to_vec(&self, device: &GpuDevice) -> Vec<f32> {
        let size = (self.len * std::mem::size_of::<f32>()) as u64;
        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tang-gpu staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tang-gpu download"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
        device.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        device.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Download buffer contents synchronously.
    pub fn to_vec_sync(&self, device: &GpuDevice) -> Vec<f32> {
        pollster::block_on(self.to_vec(device))
    }

    /// Number of f32 elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes.
    pub fn byte_size(&self) -> u64 {
        (self.len * std::mem::size_of::<f32>()) as u64
    }

    /// Clone this buffer entirely on the GPU (no CPU readback).
    /// Submits immediately — prefer `clone_gpu_batched` when batching commands.
    pub fn clone_gpu(&self, device: &GpuDevice) -> GpuBuffer {
        let dst = Self::uninit(device, self.len);
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tang-gpu clone"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &dst.buffer, 0, self.byte_size());
        device.queue.submit(std::iter::once(encoder.finish()));
        dst
    }

    /// Clone this buffer on GPU, using the kernel cache's batching mode.
    pub fn clone_gpu_batched(&self, device: &GpuDevice, cache: &mut crate::kernel::KernelCache) -> GpuBuffer {
        let dst = Self::uninit(device, self.len);
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tang-gpu clone"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &dst.buffer, 0, self.byte_size());
        cache.submit_or_enqueue(device, encoder.finish());
        dst
    }
}
