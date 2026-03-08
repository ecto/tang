//! Tile-based alpha compositing rasterizer.
//!
//! Each workgroup processes one 16×16 tile. Gaussians are loaded in batches
//! into shared memory. Each thread computes one pixel's color by evaluating
//! and blending all contributing gaussians front-to-back.
//!
//! The forward pass stores per-pixel transmittance and last-contributor index
//! for the backward pass to recover intermediate values.

/// WGSL compute shader for forward rasterization.
///
/// Inputs:
///   - sorted_indices [M] — gaussian indices in tile-major, depth-minor order
///   - tile_ranges [T, 2] — start/end into sorted_indices per tile
///   - means_2d [N, 2], conics [N, 3], opacities [N], sh_coeffs [N, C]
///
/// Outputs:
///   - image [H, W, 3] — rendered RGB
///   - final_T [H, W] — per-pixel final transmittance (for backward)
///   - n_contrib [H, W] — number of contributing gaussians per pixel (for backward)
pub const RASTERIZE_FORWARD_SHADER: &str = r#"
// Tile dimensions
const TILE_W: u32 = 16u;
const TILE_H: u32 = 16u;
const BLOCK_SIZE: u32 = 256u; // TILE_W * TILE_H

struct Config {
    image_width: u32,
    image_height: u32,
    num_tiles_x: u32,
    num_tiles_y: u32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(1) var<storage, read> tile_ranges: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> means_2d: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> conics: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> opacities: array<f32>;
@group(0) @binding(5) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> config: Config;

@group(1) @binding(0) var<storage, read_write> image: array<f32>;
@group(1) @binding(1) var<storage, read_write> final_T: array<f32>;
@group(1) @binding(2) var<storage, read_write> n_contrib: array<u32>;

// Shared memory for loading gaussian batches
var<workgroup> shared_means: array<vec2<f32>, BLOCK_SIZE>;
var<workgroup> shared_conics: array<vec3<f32>, BLOCK_SIZE>;
var<workgroup> shared_opacities: array<f32, BLOCK_SIZE>;
var<workgroup> shared_colors: array<vec3<f32>, BLOCK_SIZE>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let tile_id = wg_id.y * config.num_tiles_x + wg_id.x;
    let pixel_x = wg_id.x * TILE_W + local_id.x;
    let pixel_y = wg_id.y * TILE_H + local_id.y;
    let inside = pixel_x < config.image_width && pixel_y < config.image_height;
    let pixel_f = vec2<f32>(f32(pixel_x) + 0.5, f32(pixel_y) + 0.5);

    // Per-pixel state
    var T: f32 = 1.0;
    var C: vec3<f32> = vec3<f32>(0.0);
    var contributor_count: u32 = 0u;

    let range = tile_ranges[tile_id];
    let start = range.x;
    let end = range.y;

    // Process gaussians in batches of BLOCK_SIZE
    var batch_start = start;
    loop {
        if batch_start >= end {
            break;
        }

        // Collaboratively load a batch into shared memory
        workgroupBarrier();
        let load_idx = batch_start + local_idx;
        if load_idx < end {
            let g_idx = sorted_indices[load_idx];
            shared_means[local_idx] = means_2d[g_idx];
            shared_conics[local_idx] = conics[g_idx].xyz;
            shared_opacities[local_idx] = opacities[g_idx];
            shared_colors[local_idx] = colors[g_idx].xyz;
        }
        workgroupBarrier();

        let batch_end = min(end - batch_start, BLOCK_SIZE);

        if inside {
            for (var j: u32 = 0u; j < batch_end; j++) {
                let mean = shared_means[j];
                let con = shared_conics[j];
                let opacity = shared_opacities[j];

                // Evaluate 2D gaussian
                let d = pixel_f - mean;
                let power = -0.5 * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;

                if power > 0.0 {
                    continue;
                }

                let G = exp(power);
                let alpha = min(0.99, opacity * G);

                if alpha < 1.0 / 255.0 {
                    continue;
                }

                // Alpha composite
                C += shared_colors[j] * alpha * T;
                T *= (1.0 - alpha);
                contributor_count += 1u;

                // Early exit
                if T < 0.0001 {
                    break;
                }
            }
        }

        batch_start += BLOCK_SIZE;
    }

    if inside {
        // Add background
        let bg = vec3<f32>(config.bg_r, config.bg_g, config.bg_b);
        C += bg * T;

        let pixel_idx = pixel_y * config.image_width + pixel_x;
        image[pixel_idx * 3u + 0u] = C.x;
        image[pixel_idx * 3u + 1u] = C.y;
        image[pixel_idx * 3u + 2u] = C.z;
        final_T[pixel_idx] = T;
        n_contrib[pixel_idx] = contributor_count;
    }
}
"#;

/// WGSL compute shader for backward rasterization.
///
/// Processes gaussians in REVERSE depth order (back to front).
/// Recovers transmittance from stored final_T and computes gradients
/// via atomicAdd per gaussian.
///
/// TODO: Implement. This is the most complex shader — needs:
/// - Reverse traversal of sorted gaussians per tile
/// - Transmittance recovery: T_i = final_T / prod_{j>i}(1 - alpha_j)
/// - Gradients: dL/d{color, opacity, conic, mean_2d} per gaussian
/// - atomicAdd to accumulate gradients from all pixels
pub const RASTERIZE_BACKWARD_SHADER: &str = "// TODO: backward rasterization shader";
