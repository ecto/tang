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
/// Processes gaussians in REVERSE depth order (back to front) per tile.
/// Recovers transmittance from stored final_T and computes gradients
/// dL/d{color, opacity, conic, mean_2d} per gaussian via CAS-based
/// atomic float add (since WGSL lacks native f32 atomicAdd).
///
/// Inputs (group 0):
///   - sorted_indices, tile_ranges, means_2d, conics, opacities, colors, config (same as forward)
///   - dL_dimage [H*W*3] — gradient of loss w.r.t. rendered image
///   - final_T [H*W] — per-pixel final transmittance from forward
///   - n_contrib [H*W] — per-pixel contributing gaussian count from forward
///
/// Outputs (group 1, atomic<u32> for CAS float add):
///   - grad_colors [N*3]
///   - grad_opacities [N]
///   - grad_conics [N*3]
///   - grad_means2d [N*2]
pub const RASTERIZE_BACKWARD_SHADER: &str = r#"
const TILE_W: u32 = 16u;
const TILE_H: u32 = 16u;
const BLOCK_SIZE: u32 = 256u;

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

// Group 0: read-only inputs
@group(0) @binding(0) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(1) var<storage, read> tile_ranges: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> means_2d: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> conics: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> opacities: array<f32>;
@group(0) @binding(5) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> config: Config;
@group(0) @binding(7) var<storage, read> dL_dimage: array<f32>;
@group(0) @binding(8) var<storage, read> final_T_buf: array<f32>;
@group(0) @binding(9) var<storage, read> n_contrib_buf: array<u32>;

// Group 1: gradient outputs (atomic<u32> for CAS-based float add)
@group(1) @binding(0) var<storage, read_write> grad_colors: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> grad_opacities: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> grad_conics: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> grad_means2d: array<atomic<u32>>;

// Shared memory for batched gaussian loading
var<workgroup> shared_means: array<vec2<f32>, BLOCK_SIZE>;
var<workgroup> shared_conics: array<vec3<f32>, BLOCK_SIZE>;
var<workgroup> shared_opacities: array<f32, BLOCK_SIZE>;
var<workgroup> shared_colors: array<vec3<f32>, BLOCK_SIZE>;
var<workgroup> shared_g_idx: array<u32, BLOCK_SIZE>;

// CAS-based atomic float add (WGSL has no native f32 atomicAdd).
// Separate functions per buffer because WGSL forbids storage pointer params.
fn add_grad_color(idx: u32, val: f32) {
    if abs(val) < 1e-10 { return; }
    var old_val = atomicLoad(&grad_colors[idx]);
    loop {
        let result = atomicCompareExchangeWeak(&grad_colors[idx], old_val,
            bitcast<u32>(bitcast<f32>(old_val) + val));
        if result.exchanged { break; }
        old_val = result.old_value;
    }
}
fn add_grad_opacity(idx: u32, val: f32) {
    if abs(val) < 1e-10 { return; }
    var old_val = atomicLoad(&grad_opacities[idx]);
    loop {
        let result = atomicCompareExchangeWeak(&grad_opacities[idx], old_val,
            bitcast<u32>(bitcast<f32>(old_val) + val));
        if result.exchanged { break; }
        old_val = result.old_value;
    }
}
fn add_grad_conic(idx: u32, val: f32) {
    if abs(val) < 1e-10 { return; }
    var old_val = atomicLoad(&grad_conics[idx]);
    loop {
        let result = atomicCompareExchangeWeak(&grad_conics[idx], old_val,
            bitcast<u32>(bitcast<f32>(old_val) + val));
        if result.exchanged { break; }
        old_val = result.old_value;
    }
}
fn add_grad_mean2d(idx: u32, val: f32) {
    if abs(val) < 1e-10 { return; }
    var old_val = atomicLoad(&grad_means2d[idx]);
    loop {
        let result = atomicCompareExchangeWeak(&grad_means2d[idx], old_val,
            bitcast<u32>(bitcast<f32>(old_val) + val));
        if result.exchanged { break; }
        old_val = result.old_value;
    }
}

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

    // Per-pixel backward state
    var T: f32 = 1.0;
    var contributor: u32 = 0u;
    var dL_dC = vec3<f32>(0.0);

    if inside {
        let pixel_idx = pixel_y * config.image_width + pixel_x;
        T = final_T_buf[pixel_idx];
        contributor = n_contrib_buf[pixel_idx];
        dL_dC = vec3<f32>(
            dL_dimage[pixel_idx * 3u + 0u],
            dL_dimage[pixel_idx * 3u + 1u],
            dL_dimage[pixel_idx * 3u + 2u],
        );
    }

    // Suffix color accumulator: starts at bg * T_final
    // S = sum_{j > current} c_j * a_j * T_j + bg * T_{N+1}
    var S = vec3<f32>(config.bg_r, config.bg_g, config.bg_b) * T;

    let range = tile_ranges[tile_id];
    let start = range.x;
    let end = range.y;
    let total = end - start;
    let num_batches = (total + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    // Process batches in REVERSE order (back to front)
    for (var batch_idx: u32 = num_batches; batch_idx > 0u; batch_idx--) {
        let batch_start_offset = (batch_idx - 1u) * BLOCK_SIZE;
        let global_batch_start = start + batch_start_offset;

        // Collaboratively load this batch into shared memory
        workgroupBarrier();
        let load_idx = global_batch_start + local_idx;
        if load_idx < end {
            let g_idx = sorted_indices[load_idx];
            shared_means[local_idx] = means_2d[g_idx];
            shared_conics[local_idx] = conics[g_idx].xyz;
            shared_opacities[local_idx] = opacities[g_idx];
            shared_colors[local_idx] = colors[g_idx].xyz;
            shared_g_idx[local_idx] = g_idx;
        }
        workgroupBarrier();

        let batch_count = min(end - global_batch_start, BLOCK_SIZE);

        if inside && contributor > 0u {
            // Iterate within this batch in REVERSE
            for (var j: u32 = batch_count; j > 0u; j--) {
                let jj = j - 1u;

                let mean = shared_means[jj];
                let con = shared_conics[jj];
                let opacity = shared_opacities[jj];
                let color = shared_colors[jj];
                let g_idx = shared_g_idx[jj];

                // Recompute alpha (must match forward exactly)
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

                // This gaussian contributed in the forward pass
                contributor -= 1u;

                // Recover transmittance BEFORE this gaussian was applied
                // Forward did: T_{i+1} = T_i * (1 - alpha_i)
                // So: T_i = T_{i+1} / (1 - alpha_i)
                let one_minus_alpha = 1.0 - alpha;
                T = T / one_minus_alpha;

                // --- Color gradient ---
                // C_pixel depends on c_i via: c_i * alpha_i * T_i
                // dL/dc_i = alpha_i * T_i * dL/dC  (always valid, even if alpha clamped)
                let w_color = alpha * T;
                let dL_dc = dL_dC * w_color;

                add_grad_color(g_idx * 3u + 0u, dL_dc.x);
                add_grad_color(g_idx * 3u + 1u, dL_dc.y);
                add_grad_color(g_idx * 3u + 2u, dL_dc.z);

                // --- Alpha gradient ---
                // dC/dalpha_i = c_i * T_i - S_after_i / (1 - alpha_i)
                // where S_after_i = accumulated color from gaussians behind this one + bg
                let dL_dalpha = dot(dL_dC, color * T - S / one_minus_alpha);

                // Update suffix accumulator for next iteration
                S += color * alpha * T;

                // --- Chain through alpha to gaussian parameters ---
                // alpha = min(0.99, opacity * G)
                // When clamped (opacity * G >= 0.99), d(alpha)/d(params) = 0
                let vis = opacity * G;
                if vis < 0.99 {
                    // d(alpha)/d(opacity) = G
                    let dL_dopacity_val = dL_dalpha * G;

                    // d(alpha)/d(G) = opacity, dG/d(power) = G
                    let dL_dpower = dL_dalpha * opacity * G;

                    // d(power)/d(conic)
                    let dL_dcon_x = dL_dpower * (-0.5 * d.x * d.x);
                    let dL_dcon_y = dL_dpower * (-d.x * d.y);
                    let dL_dcon_z = dL_dpower * (-0.5 * d.y * d.y);

                    // d(power)/d(mean_2d)
                    // power = f(pixel - mean), d(pixel-mean)/d(mean) = -1
                    // dp/d(dx) = -(con.x * dx + con.y * dy)
                    // dp/d(mean_x) = con.x * dx + con.y * dy
                    let dL_dmean_x = dL_dpower * (con.x * d.x + con.y * d.y);
                    let dL_dmean_y = dL_dpower * (con.z * d.y + con.y * d.x);

                    // Accumulate gradients via atomic CAS
                    add_grad_opacity(g_idx, dL_dopacity_val);
                    add_grad_conic(g_idx * 3u + 0u, dL_dcon_x);
                    add_grad_conic(g_idx * 3u + 1u, dL_dcon_y);
                    add_grad_conic(g_idx * 3u + 2u, dL_dcon_z);
                    add_grad_mean2d(g_idx * 2u + 0u, dL_dmean_x);
                    add_grad_mean2d(g_idx * 2u + 1u, dL_dmean_y);
                }

                if contributor == 0u {
                    break;
                }
            }
        }
    }
}
"#;
