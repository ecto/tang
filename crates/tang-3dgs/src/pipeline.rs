//! Full forward rendering pipeline: project → sort → rasterize.
//!
//! Orchestrates GPU buffer creation, shader dispatch, and readback.
//! All computation happens on the GPU via wgpu compute shaders.

use crate::camera::Camera;
use crate::cloud::GaussianCloud;
use crate::{ForwardContext, GaussianGradients, RasterConfig, RenderOutput, TILE_SIZE};

/// The gaussian splatting rasterizer.
///
/// Owns compiled shader pipelines and reusable GPU resources.
#[allow(dead_code)]
pub struct Rasterizer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Compiled pipelines
    project_pipeline: wgpu::ComputePipeline,
    project_bgl0: wgpu::BindGroupLayout,
    project_bgl1: wgpu::BindGroupLayout,
    rasterize_pipeline: wgpu::ComputePipeline,
    rasterize_bgl0: wgpu::BindGroupLayout,
    rasterize_bgl1: wgpu::BindGroupLayout,
    // Sort pipelines
    count_tiles_pipeline: wgpu::ComputePipeline,
    count_tiles_bgl0: wgpu::BindGroupLayout,
    count_tiles_bgl1: wgpu::BindGroupLayout,
    write_keys_pipeline: wgpu::ComputePipeline,
    write_keys_bgl0: wgpu::BindGroupLayout,
    write_keys_bgl1: wgpu::BindGroupLayout,
    prefix_sum_pipeline: wgpu::ComputePipeline,
    prefix_sum_bgl0: wgpu::BindGroupLayout,
    radix_count_pipeline: wgpu::ComputePipeline,
    radix_count_bgl0: wgpu::BindGroupLayout,
    radix_count_bgl1: wgpu::BindGroupLayout,
    radix_scatter_pipeline: wgpu::ComputePipeline,
    radix_scatter_bgl0: wgpu::BindGroupLayout,
    radix_scatter_bgl1: wgpu::BindGroupLayout,
    tile_ranges_pipeline: wgpu::ComputePipeline,
    tile_ranges_bgl0: wgpu::BindGroupLayout,
    tile_ranges_bgl1: wgpu::BindGroupLayout,
    // Backward pass
    rasterize_bw_pipeline: wgpu::ComputePipeline,
    rasterize_bw_bgl0: wgpu::BindGroupLayout,
    rasterize_bw_bgl1: wgpu::BindGroupLayout,
    config: RasterConfig,
}

impl Rasterizer {
    /// Create a new rasterizer, compiling all shaders.
    pub fn new(config: RasterConfig) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("no GPU adapter found");

        let adapter_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("tang-3dgs"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: adapter_limits.max_storage_buffers_per_shader_stage,
                    max_bind_groups: adapter_limits.max_bind_groups,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("failed to create GPU device");

        // Compile projection shader
        let (project_pipeline, project_bgl0, project_bgl1) =
            compile_project_pipeline(&device);

        // Compile rasterization shader
        let (rasterize_pipeline, rasterize_bgl0, rasterize_bgl1) =
            compile_rasterize_pipeline(&device);

        // Compile sort shaders
        let (count_tiles_pipeline, count_tiles_bgl0, count_tiles_bgl1) =
            compile_count_tiles_pipeline(&device);
        let (write_keys_pipeline, write_keys_bgl0, write_keys_bgl1) =
            compile_write_keys_pipeline(&device);
        let (prefix_sum_pipeline, prefix_sum_bgl0) =
            compile_prefix_sum_pipeline(&device);
        let (radix_count_pipeline, radix_count_bgl0, radix_count_bgl1) =
            compile_radix_count_pipeline(&device);
        let (radix_scatter_pipeline, radix_scatter_bgl0, radix_scatter_bgl1) =
            compile_radix_scatter_pipeline(&device);
        let (tile_ranges_pipeline, tile_ranges_bgl0, tile_ranges_bgl1) =
            compile_tile_ranges_pipeline(&device);
        let (rasterize_bw_pipeline, rasterize_bw_bgl0, rasterize_bw_bgl1) =
            compile_rasterize_backward_pipeline(&device);

        Self {
            device,
            queue,
            project_pipeline,
            project_bgl0,
            project_bgl1,
            rasterize_pipeline,
            rasterize_bgl0,
            rasterize_bgl1,
            count_tiles_pipeline,
            count_tiles_bgl0,
            count_tiles_bgl1,
            write_keys_pipeline,
            write_keys_bgl0,
            write_keys_bgl1,
            prefix_sum_pipeline,
            prefix_sum_bgl0,
            radix_count_pipeline,
            radix_count_bgl0,
            radix_count_bgl1,
            radix_scatter_pipeline,
            radix_scatter_bgl0,
            radix_scatter_bgl1,
            tile_ranges_pipeline,
            tile_ranges_bgl0,
            tile_ranges_bgl1,
            rasterize_bw_pipeline,
            rasterize_bw_bgl0,
            rasterize_bw_bgl1,
            config,
        }
    }

    /// Render a gaussian cloud from a camera viewpoint.
    pub fn forward(&self, cloud: &GaussianCloud, camera: &Camera) -> RenderOutput {
        let n = cloud.count as u32;
        let w = self.config.width;
        let h = self.config.height;
        let num_tiles_x = (w + TILE_SIZE - 1) / TILE_SIZE;
        let num_tiles_y = (h + TILE_SIZE - 1) / TILE_SIZE;
        let num_tiles = num_tiles_x * num_tiles_y;

        // --- Upload gaussian data ---
        let positions_buf = self.create_buffer_f32(&flatten_vec3(&cloud.positions));
        let scales_buf = self.create_buffer_f32(&flatten_vec3(&cloud.scales));
        let rotations_buf = self.create_buffer_f32(&flatten_vec4(&cloud.rotations));
        let opacities_buf = self.create_buffer_f32(
            &cloud.opacities.iter().map(|&o| sigmoid(o)).collect::<Vec<_>>(),
        );
        let colors_buf = self.create_buffer_f32(&sh_to_rgb(cloud));

        // Camera uniform
        let camera_uniform = camera.as_uniform();
        let camera_buf = self.create_buffer_bytes(bytemuck::bytes_of(&camera_uniform));

        // Config uniform for projection
        let project_config = [
            n,
            w,
            h,
            cloud.sh_degree,
            self.config.near.to_bits(),
            self.config.far.to_bits(),
            0,
            0,
        ];
        let project_config_buf = self.create_buffer_bytes(bytemuck::cast_slice(&project_config));

        // --- Output buffers for projection ---
        let means_2d_buf = self.create_buffer_zero((n as usize) * 2);
        let conics_buf = self.create_buffer_zero((n as usize) * 4);
        let radii_buf = self.create_buffer_zero(n as usize);
        let depths_buf = self.create_buffer_zero(n as usize);

        // === PASS 1: Project ===
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("project bg0"),
                layout: &self.project_bgl0,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: positions_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: scales_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: rotations_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: camera_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: project_config_buf.as_entire_binding() },
                ],
            });
            let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("project bg1"),
                layout: &self.project_bgl1,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: means_2d_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: conics_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: radii_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: depths_buf.as_entire_binding() },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.project_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.dispatch_workgroups((n + 255) / 256, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // === PASS 2: Sort ===
        // 2a: Count tiles per gaussian
        let tile_counts_buf = self.create_buffer_zero(n as usize);
        let total_pairs_buf = self.create_buffer_zero(1);

        let sort_config = [n, w, h, num_tiles_x, num_tiles_y, 0, 0, 0u32];
        let sort_config_buf = self.create_buffer_bytes(bytemuck::cast_slice(&sort_config));

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("count_tiles bg0"),
                layout: &self.count_tiles_bgl0,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: means_2d_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: radii_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: depths_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: sort_config_buf.as_entire_binding() },
                ],
            });
            let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("count_tiles bg1"),
                layout: &self.count_tiles_bgl1,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: tile_counts_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: total_pairs_buf.as_entire_binding() },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.count_tiles_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.dispatch_workgroups((n + 255) / 256, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back total pairs count to size sort buffers
        let total_pairs = self.readback_u32(&total_pairs_buf, 1)[0];

        if total_pairs == 0 {
            // No visible gaussians — return black image
            let pixels = (w * h * 3) as usize;
            let mut image = vec![0.0f32; pixels];
            for i in 0..(w * h) as usize {
                image[i * 3] = self.config.bg_color[0];
                image[i * 3 + 1] = self.config.bg_color[1];
                image[i * 3 + 2] = self.config.bg_color[2];
            }
            return RenderOutput {
                image,
                ctx: ForwardContext {
                    final_transmittance: vec![1.0; (w * h) as usize],
                    n_contrib: vec![0; (w * h) as usize],
                    sorted_indices: vec![],
                    tile_ranges: vec![[0, 0]; num_tiles as usize],
                    means_2d: vec![[0.0; 2]; n as usize],
                    conics: vec![[0.0; 3]; n as usize],
                    radii: vec![0; n as usize],
                },
            };
        }

        // 2b-2e: Sort on CPU (GPU prefix sum and radix sort have cross-block bugs).
        // Read back projection results.
        let means_2d_raw = self.readback_f32(&means_2d_buf, n as usize * 2);
        let radii_raw = self.readback_u32(&radii_buf, n as usize);
        let depths_raw = self.readback_f32(&depths_buf, n as usize);

        // Generate (tile_id, depth, gaussian_idx) tuples for all tile overlaps.
        let mut pairs: Vec<(u32, f32, u32)> = Vec::with_capacity(total_pairs as usize);
        for idx in 0..n as usize {
            let r = radii_raw[idx];
            if r == 0 { continue; }
            let mx = means_2d_raw[idx * 2];
            let my = means_2d_raw[idx * 2 + 1];
            let tile_min_x = ((mx - r as f32) / 16.0).max(0.0) as u32;
            let tile_max_x = (((mx + r as f32) / 16.0) as u32 + 1).min(num_tiles_x);
            let tile_min_y = ((my - r as f32) / 16.0).max(0.0) as u32;
            let tile_max_y = (((my + r as f32) / 16.0) as u32 + 1).min(num_tiles_y);
            for ty in tile_min_y..tile_max_y {
                for tx in tile_min_x..tile_max_x {
                    let tile_id = ty * num_tiles_x + tx;
                    pairs.push((tile_id, depths_raw[idx], idx as u32));
                }
            }
        }

        // Sort by (tile_id, depth)
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));

        let _actual_pairs = pairs.len() as u32;
        let sorted_indices_cpu: Vec<u32> = pairs.iter().map(|p| p.2).collect();

        // Compute tile ranges
        let mut tile_ranges_cpu = vec![[0u32; 2]; num_tiles as usize];
        for (i, &(tile_id, _, _)) in pairs.iter().enumerate() {
            let tid = tile_id as usize;
            if i == 0 || pairs[i - 1].0 != tile_id {
                tile_ranges_cpu[tid][0] = i as u32;
            }
            if i + 1 == pairs.len() || pairs[i + 1].0 != tile_id {
                tile_ranges_cpu[tid][1] = (i + 1) as u32;
            }
        }

        // Upload to GPU
        let values_buf = self.create_buffer_bytes(bytemuck::cast_slice(&sorted_indices_cpu));
        let tile_ranges_flat: Vec<u32> = tile_ranges_cpu.iter().flat_map(|r| [r[0], r[1]]).collect();
        let tile_ranges_buf = self.create_buffer_bytes(bytemuck::cast_slice(&tile_ranges_flat));

        // === PASS 3: Rasterize ===
        let image_buf = self.create_buffer_zero((w * h * 3) as usize);
        let final_t_buf = self.create_buffer_zero((w * h) as usize);
        let n_contrib_buf = self.create_buffer_zero((w * h) as usize);

        let raster_config = [
            w,
            h,
            num_tiles_x,
            num_tiles_y,
            self.config.bg_color[0].to_bits(),
            self.config.bg_color[1].to_bits(),
            self.config.bg_color[2].to_bits(),
            0u32,
        ];
        let raster_config_buf = self.create_buffer_bytes(bytemuck::cast_slice(&raster_config));

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rasterize bg0"),
                layout: &self.rasterize_bgl0,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: values_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: tile_ranges_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: means_2d_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: conics_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: opacities_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: colors_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: raster_config_buf.as_entire_binding() },
                ],
            });
            let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rasterize bg1"),
                layout: &self.rasterize_bgl1,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: image_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: final_t_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: n_contrib_buf.as_entire_binding() },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.rasterize_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.dispatch_workgroups(num_tiles_x, num_tiles_y, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // === Readback ===
        let image = self.readback_f32(&image_buf, (w * h * 3) as usize);
        let final_transmittance = self.readback_f32(&final_t_buf, (w * h) as usize);
        let n_contrib = self.readback_u32(&n_contrib_buf, (w * h) as usize);
        let sorted_indices = self.readback_u32(&values_buf, total_pairs as usize);
        let tile_ranges_raw = self.readback_u32(&tile_ranges_buf, (num_tiles * 2) as usize);
        let tile_ranges: Vec<[u32; 2]> = tile_ranges_raw
            .chunks(2)
            .map(|c| [c[0], c[1]])
            .collect();
        let means_2d_raw = self.readback_f32(&means_2d_buf, (n * 2) as usize);
        let means_2d: Vec<[f32; 2]> = means_2d_raw.chunks(2).map(|c| [c[0], c[1]]).collect();
        let conics_raw = self.readback_f32(&conics_buf, (n * 4) as usize);
        let conics: Vec<[f32; 3]> = conics_raw.chunks(4).map(|c| [c[0], c[1], c[2]]).collect();
        let radii = self.readback_u32(&radii_buf, n as usize);

        RenderOutput {
            image,
            ctx: ForwardContext {
                final_transmittance,
                n_contrib,
                sorted_indices,
                tile_ranges,
                means_2d,
                conics,
                radii,
            },
        }
    }

    /// Compute gradients w.r.t. gaussian parameters given image-space loss gradients.
    ///
    /// `dL_dimage` is the gradient of the loss w.r.t. the rendered image [H*W*3].
    /// `cloud` and `camera` must be the same as used in the forward pass.
    /// `ctx` is the ForwardContext from the forward pass.
    pub fn backward(
        &self,
        cloud: &GaussianCloud,
        _camera: &Camera,
        ctx: &ForwardContext,
        #[allow(non_snake_case)] dL_dimage: &[f32],
    ) -> GaussianGradients {
        let n = cloud.count as u32;
        let w = self.config.width;
        let h = self.config.height;
        let num_tiles_x = (w + TILE_SIZE - 1) / TILE_SIZE;
        let num_tiles_y = (h + TILE_SIZE - 1) / TILE_SIZE;

        // Re-upload gaussian data needed for recomputing alpha
        let means_2d_flat: Vec<f32> = ctx.means_2d.iter().flat_map(|m| [m[0], m[1]]).collect();
        let means_2d_buf = self.create_buffer_f32(&means_2d_flat);
        let conics_flat: Vec<f32> = ctx.conics.iter().flat_map(|c| [c[0], c[1], c[2], 0.0]).collect();
        let conics_buf = self.create_buffer_f32(&conics_flat);
        let opacities_buf = self.create_buffer_f32(
            &cloud.opacities.iter().map(|&o| sigmoid(o)).collect::<Vec<_>>(),
        );
        let colors_buf = self.create_buffer_f32(&sh_to_rgb(cloud));

        // Re-upload sorted indices and tile ranges
        let sorted_indices_buf = {
            use wgpu::util::DeviceExt;
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&ctx.sorted_indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let tile_ranges_flat: Vec<u32> = ctx.tile_ranges.iter().flat_map(|r| [r[0], r[1]]).collect();
        let tile_ranges_buf = {
            use wgpu::util::DeviceExt;
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&tile_ranges_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };

        // Config uniform (same struct as forward rasterize)
        let raster_config = [
            w,
            h,
            num_tiles_x,
            num_tiles_y,
            self.config.bg_color[0].to_bits(),
            self.config.bg_color[1].to_bits(),
            self.config.bg_color[2].to_bits(),
            0u32,
        ];
        let raster_config_buf = self.create_buffer_bytes(bytemuck::cast_slice(&raster_config));

        // Upload dL/dimage, final_T, n_contrib
        let dl_image_buf = self.create_buffer_f32(dL_dimage);
        let final_t_buf = self.create_buffer_f32(&ctx.final_transmittance);
        let n_contrib_buf = {
            use wgpu::util::DeviceExt;
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&ctx.n_contrib),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };

        // Gradient output buffers (zeroed, atomic<u32>)
        let grad_colors_buf = self.create_buffer_zero((n as usize) * 3);
        let grad_opacities_buf = self.create_buffer_zero(n as usize);
        let grad_conics_buf = self.create_buffer_zero((n as usize) * 3);
        let grad_means2d_buf = self.create_buffer_zero((n as usize) * 2);

        // Dispatch backward rasterization
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rasterize_bw bg0"),
                layout: &self.rasterize_bw_bgl0,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: sorted_indices_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: tile_ranges_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: means_2d_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: conics_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: opacities_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: colors_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: raster_config_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: dl_image_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: final_t_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: n_contrib_buf.as_entire_binding() },
                ],
            });
            let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rasterize_bw bg1"),
                layout: &self.rasterize_bw_bgl1,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: grad_colors_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: grad_opacities_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: grad_conics_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: grad_means2d_buf.as_entire_binding() },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.rasterize_bw_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.dispatch_workgroups(num_tiles_x, num_tiles_y, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Readback gradients (reinterpret atomic u32 as f32)
        let grad_colors_raw = self.readback_f32(&grad_colors_buf, (n as usize) * 3);
        let grad_opacities_raw = self.readback_f32(&grad_opacities_buf, n as usize);
        let grad_conics_raw = self.readback_f32(&grad_conics_buf, (n as usize) * 3);
        let grad_means2d_raw = self.readback_f32(&grad_means2d_buf, (n as usize) * 2);

        // Chain through backward projection: dL/d{conic, mean_2d} → dL/d{position, scale, rotation}
        #[allow(non_snake_case)]
        let dL_dconics: Vec<[f32; 3]> = grad_conics_raw.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        #[allow(non_snake_case)]
        let dL_dmeans2d: Vec<[f32; 2]> = grad_means2d_raw.chunks(2).map(|m| [m[0], m[1]]).collect();

        #[allow(non_snake_case)]
        let (dL_dpositions, dL_dscales, dL_drotations) =
            crate::project_backward::backward_projection(
                &cloud.positions,
                &cloud.scales,
                &cloud.rotations,
                _camera,
                &ctx.radii,
                &dL_dconics,
                &dL_dmeans2d,
            );

        GaussianGradients {
            positions: dL_dpositions,
            scales: dL_dscales,
            rotations: dL_drotations,
            opacities: grad_opacities_raw,
            sh_coeffs: grad_colors_raw, // dL/d(evaluated_color), chain through SH later
            _conics: dL_dconics,
            _means_2d: dL_dmeans2d,
        }
    }

    // --- Buffer helpers ---

    fn create_buffer_f32(&self, data: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn create_buffer_bytes(&self, data: &[u8]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn create_buffer_zero(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[allow(dead_code)]
    fn clone_buffer(&self, src: &wgpu::Buffer) -> wgpu::Buffer {
        let dst = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: src.size(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(src, 0, &dst, 0, src.size());
        self.queue.submit(std::iter::once(encoder.finish()));
        dst
    }

    fn readback_f32(&self, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let size = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    fn readback_u32(&self, buf: &wgpu::Buffer, len: usize) -> Vec<u32> {
        let size = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    // --- Sort helpers ---

    #[allow(dead_code)]
    fn run_prefix_sum(&self, buf: &wgpu::Buffer, n: usize) {
        let params = [n as u32, 0, 0, 0u32];
        let params_buf = self.create_buffer_bytes(bytemuck::cast_slice(&params));

        let num_blocks = (n + 511) / 512;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prefix_sum bg"),
                layout: &self.prefix_sum_bgl0,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.prefix_sum_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(num_blocks as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    #[allow(dead_code)]
    fn radix_sort_pass(
        &self,
        keys_in: &wgpu::Buffer,
        values_in: &wgpu::Buffer,
        keys_out: &wgpu::Buffer,
        values_out: &wgpu::Buffer,
        num_pairs: u32,
        sort_component: u32,
    ) {
        // 4 passes for 32-bit keys (8 bits at a time)
        for bit_offset in (0..32).step_by(8) {
            let histogram_buf = self.create_buffer_zero(256);
            let sort_params = [num_pairs, bit_offset as u32, sort_component, 0u32];
            let sort_params_buf = self.create_buffer_bytes(bytemuck::cast_slice(&sort_params));

            // Count pass
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.radix_count_bgl0,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: keys_in.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: values_in.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: sort_params_buf.as_entire_binding() },
                    ],
                });
                let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.radix_count_bgl1,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: histogram_buf.as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.radix_count_pipeline);
                pass.set_bind_group(0, &bg0, &[]);
                pass.set_bind_group(1, &bg1, &[]);
                pass.dispatch_workgroups((num_pairs + 255) / 256, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));

            // Prefix sum on histogram
            self.run_prefix_sum(&histogram_buf, 256);

            // Scatter pass
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.radix_scatter_bgl0,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: keys_in.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: values_in.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: sort_params_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: histogram_buf.as_entire_binding() },
                    ],
                });
                let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.radix_scatter_bgl1,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: keys_out.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: values_out.as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.radix_scatter_pipeline);
                pass.set_bind_group(0, &bg0, &[]);
                pass.set_bind_group(1, &bg1, &[]);
                pass.dispatch_workgroups((num_pairs + 255) / 256, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }
    }
}

// --- Shader compilation (one function per pipeline to keep it organized) ---

fn make_bgl_entries(entries: &[(wgpu::BufferBindingType, bool)]) -> Vec<wgpu::BindGroupLayoutEntry> {
    entries
        .iter()
        .enumerate()
        .map(|(i, &(ty, read_only))| {
            let binding_ty = match ty {
                wgpu::BufferBindingType::Uniform => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                _ => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            };
            wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: binding_ty,
                count: None,
            }
        })
        .collect()
}

fn create_pipeline(
    device: &wgpu::Device,
    wgsl: &str,
    entry: &str,
    bgl0_entries: &[(wgpu::BufferBindingType, bool)],
    bgl1_entries: &[(wgpu::BufferBindingType, bool)],
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    let s = wgpu::BufferBindingType::Storage { read_only: false }; // dummy, overridden
    let _ = s;

    let entries0 = make_bgl_entries(bgl0_entries);
    let entries1 = make_bgl_entries(bgl1_entries);

    let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries0,
    });
    let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries1,
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl0, &bgl1],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });

    (pipeline, bgl0, bgl1)
}

use wgpu::BufferBindingType as BBT;
const RO: bool = true;
const RW: bool = false;
const STORAGE_RO: (BBT, bool) = (BBT::Storage { read_only: true }, RO);
const STORAGE_RW: (BBT, bool) = (BBT::Storage { read_only: false }, RW);
const UNIFORM: (BBT, bool) = (BBT::Uniform, true);

fn compile_project_pipeline(d: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    create_pipeline(d, crate::project::PROJECT_SHADER, "main",
        &[STORAGE_RO, STORAGE_RO, STORAGE_RO, UNIFORM, UNIFORM],
        &[STORAGE_RW, STORAGE_RW, STORAGE_RW, STORAGE_RW],
    )
}

fn compile_rasterize_pipeline(d: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    create_pipeline(d, crate::rasterize::RASTERIZE_FORWARD_SHADER, "main",
        &[STORAGE_RO, STORAGE_RO, STORAGE_RO, STORAGE_RO, STORAGE_RO, STORAGE_RO, UNIFORM],
        &[STORAGE_RW, STORAGE_RW, STORAGE_RW],
    )
}

fn compile_count_tiles_pipeline(d: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    create_pipeline(d, crate::sort::GENERATE_KEYS_SHADER, "count_tiles",
        &[STORAGE_RO, STORAGE_RO, STORAGE_RO, UNIFORM],
        &[STORAGE_RW, STORAGE_RW],
    )
}

fn compile_write_keys_pipeline(d: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    create_pipeline(d, crate::sort::WRITE_KEYS_SHADER, "write_pairs",
        &[STORAGE_RO, STORAGE_RO, STORAGE_RO, UNIFORM],
        &[STORAGE_RO, STORAGE_RW, STORAGE_RW],
    )
}

fn compile_prefix_sum_pipeline(d: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(crate::sort::PREFIX_SUM_SHADER.into()),
    });
    let entries = make_bgl_entries(&[STORAGE_RW, UNIFORM]);
    let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    });
    let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("scan"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, bgl)
}

fn compile_radix_count_pipeline(d: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    create_pipeline(d, crate::sort::RADIX_SORT_SHADER, "count",
        &[STORAGE_RO, STORAGE_RO, UNIFORM],
        &[STORAGE_RW],
    )
}

fn compile_radix_scatter_pipeline(d: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    create_pipeline(d, crate::sort::RADIX_SCATTER_SHADER, "scatter",
        &[STORAGE_RO, STORAGE_RO, UNIFORM, STORAGE_RW],
        &[STORAGE_RW, STORAGE_RW],
    )
}

fn compile_rasterize_backward_pipeline(d: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    create_pipeline(d, crate::rasterize::RASTERIZE_BACKWARD_SHADER, "main",
        // group 0: sorted_indices, tile_ranges, means_2d, conics, opacities, colors, config, dL_dimage, final_T, n_contrib
        &[STORAGE_RO, STORAGE_RO, STORAGE_RO, STORAGE_RO, STORAGE_RO, STORAGE_RO, UNIFORM, STORAGE_RO, STORAGE_RO, STORAGE_RO],
        // group 1: grad_colors, grad_opacities, grad_conics, grad_means2d
        &[STORAGE_RW, STORAGE_RW, STORAGE_RW, STORAGE_RW],
    )
}

fn compile_tile_ranges_pipeline(d: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    create_pipeline(d, crate::sort::IDENTIFY_TILE_RANGES_SHADER, "main",
        &[STORAGE_RO, UNIFORM],
        &[STORAGE_RW],
    )
}

// --- Data conversion helpers ---

fn flatten_vec3(data: &[[f32; 3]]) -> Vec<f32> {
    // Pad to vec4 for GPU alignment
    data.iter().flat_map(|v| [v[0], v[1], v[2], 0.0]).collect()
}

fn flatten_vec4(data: &[[f32; 4]]) -> Vec<f32> {
    data.iter().flat_map(|v| [v[0], v[1], v[2], v[3]]).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Extract DC (degree-0) SH as RGB color, padded to vec4.
fn sh_to_rgb(cloud: &GaussianCloud) -> Vec<f32> {
    let sh_per_g = cloud.sh_coeffs_per_gaussian();
    let mut colors = Vec::with_capacity(cloud.count * 4);
    for i in 0..cloud.count {
        let base = i * sh_per_g;
        // SH DC component: color = SH_0 * C0 + 0.5 where C0 = 0.28209479...
        let c0 = 0.28209479;
        let r = (cloud.sh_coeffs[base] * c0 + 0.5).clamp(0.0, 1.0);
        let g = (cloud.sh_coeffs[base + 1] * c0 + 0.5).clamp(0.0, 1.0);
        let b = (cloud.sh_coeffs[base + 2] * c0 + 0.5).clamp(0.0, 1.0);
        colors.extend_from_slice(&[r, g, b, 1.0]);
    }
    colors
}
