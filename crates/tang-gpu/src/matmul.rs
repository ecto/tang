//! Tiled matrix multiplication on GPU.

use crate::buffer::GpuBuffer;
use crate::device::GpuDevice;
use crate::kernel::KernelCache;
use crate::tensor::GpuTensor;

use std::hash::{Hash, Hasher};

const TILE_SIZE: u32 = 16;

/// GPU matrix multiply: C = A @ B.
///
/// A: [M, K], B: [K, N] -> C: [M, N]
///
/// Uses a cached tiled WGSL kernel with 16x16 shared-memory tiles.
pub fn matmul(
    device: &GpuDevice,
    cache: &mut KernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> GpuTensor {
    assert_eq!(a.ndim(), 2, "matmul: A must be 2D");
    assert_eq!(b.ndim(), 2, "matmul: B must be 2D");
    let m = a.shape()[0];
    let k = a.shape()[1];
    assert_eq!(b.shape()[0], k, "matmul: inner dimensions must match");
    let n = b.shape()[1];

    let wgsl = matmul_wgsl();
    let out_buf = GpuBuffer::uninit(device, m * n);

    // Use the KernelCache's 4-binding layout (same as dispatch_rr_w)
    // but with custom 2D workgroup dispatch
    let dims_data: [u32; 4] = [m as u32, n as u32, k as u32, 0];

    let hash = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        wgsl.hash(&mut hasher);
        hasher.finish()
    };

    let cached = cache.get_or_compile_rr_w(device, &wgsl, hash);

    use wgpu::util::DeviceExt;
    let dims_buf = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matmul dims"),
            contents: bytemuck::cast_slice(&dims_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul bind group"),
        layout: &cached.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dims_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul dispatch"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = (n as u32).div_ceil(TILE_SIZE);
        let wg_y = (m as u32).div_ceil(TILE_SIZE);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    cache.submit_or_enqueue(device, encoder.finish());

    GpuTensor {
        buffer: out_buf,
        shape: vec![m, n],
    }
}

fn matmul_wgsl() -> String {
    format!(
        r#"// Tiled matmul: C[M,N] = A[M,K] @ B[K,N]

struct Dims {{
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const TILE: u32 = {TILE_SIZE}u;

var<workgroup> tile_a: array<f32, {tile_area}>;
var<workgroup> tile_b: array<f32, {tile_area}>;

@compute @workgroup_size({TILE_SIZE}, {TILE_SIZE})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let row = gid.y;
    let col = gid.x;
    let lr = lid.y;
    let lc = lid.x;

    var acc: f32 = 0.0;
    let n_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {{
        let a_col = t * TILE + lc;
        let b_row = t * TILE + lr;

        if (row < dims.M && a_col < dims.K) {{
            tile_a[lr * TILE + lc] = A[row * dims.K + a_col];
        }} else {{
            tile_a[lr * TILE + lc] = 0.0;
        }}

        if (b_row < dims.K && col < dims.N) {{
            tile_b[lr * TILE + lc] = B[b_row * dims.N + col];
        }} else {{
            tile_b[lr * TILE + lc] = 0.0;
        }}

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE; i = i + 1u) {{
            acc = acc + tile_a[lr * TILE + i] * tile_b[i * TILE + lc];
        }}

        workgroupBarrier();
    }}

    if (row < dims.M && col < dims.N) {{
        C[row * dims.N + col] = acc;
    }}
}}
"#,
        TILE_SIZE = TILE_SIZE,
        tile_area = TILE_SIZE * TILE_SIZE,
    )
}
