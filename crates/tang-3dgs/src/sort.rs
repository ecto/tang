//! GPU radix sort for tile-based rendering.
//!
//! Sorts (key, value) pairs where:
//!   key = (tile_id << 32) | depth_as_u32
//!   value = gaussian_index
//!
//! Implements a parallel radix sort in WGSL compute shaders.
//! Based on the counting sort approach: for each radix digit,
//! count occurrences → prefix sum → scatter.
//!
//! Reference: web-splat (Fuchsia RadixSort port), Onesweep (Adinets & Merrill 2022)

/// Shader that generates sort keys from projected gaussian data.
///
/// For each visible gaussian, generates one (key, value) pair per overlapping tile.
/// key = (tile_id << 32) | float_to_sortable_uint(depth)
/// value = gaussian_index
pub const GENERATE_KEYS_SHADER: &str = r#"
struct Config {
    num_gaussians: u32,
    image_width: u32,
    image_height: u32,
    num_tiles_x: u32,
    num_tiles_y: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> means_2d: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> radii: array<u32>;
@group(0) @binding(2) var<storage, read> depths: array<f32>;
@group(0) @binding(3) var<uniform> config: Config;

// Output: number of tiles each gaussian overlaps (for prefix sum)
@group(1) @binding(0) var<storage, read_write> tile_counts: array<u32>;
// Output: total number of (key, value) pairs (atomic counter)
@group(1) @binding(1) var<storage, read_write> total_pairs: array<atomic<u32>>;

@compute @workgroup_size(256)
fn count_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= config.num_gaussians {
        return;
    }

    let r = radii[idx];
    if r == 0u {
        tile_counts[idx] = 0u;
        return;
    }

    let mean = means_2d[idx];
    let tile_min_x = u32(max(0.0, (mean.x - f32(r)) / 16.0));
    let tile_max_x = min(config.num_tiles_x, u32((mean.x + f32(r)) / 16.0) + 1u);
    let tile_min_y = u32(max(0.0, (mean.y - f32(r)) / 16.0));
    let tile_max_y = min(config.num_tiles_y, u32((mean.y + f32(r)) / 16.0) + 1u);

    let count = (tile_max_x - tile_min_x) * (tile_max_y - tile_min_y);
    tile_counts[idx] = count;
    atomicAdd(&total_pairs[0], count);
}
"#;

/// Shader that writes (key, value) pairs after prefix sum gives offsets.
pub const WRITE_KEYS_SHADER: &str = r#"
struct Config {
    num_gaussians: u32,
    image_width: u32,
    image_height: u32,
    num_tiles_x: u32,
    num_tiles_y: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> means_2d: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> radii: array<u32>;
@group(0) @binding(2) var<storage, read> depths: array<f32>;
@group(0) @binding(3) var<uniform> config: Config;

@group(1) @binding(0) var<storage, read> offsets: array<u32>;
@group(1) @binding(1) var<storage, read_write> keys: array<vec2<u32>>;
// keys[i] = (high_key, low_key) where combined = tile_id:depth
// values are stored as keys[i].y's original gaussian index is tracked via a parallel array
@group(1) @binding(2) var<storage, read_write> values: array<u32>;

// Convert float depth to sortable uint (preserves ordering)
fn float_to_sort_key(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    // Flip sign bit and conditionally flip all bits for negative floats
    let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
    return bits ^ mask;
}

@compute @workgroup_size(256)
fn write_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= config.num_gaussians {
        return;
    }

    let r = radii[idx];
    if r == 0u {
        return;
    }

    let mean = means_2d[idx];
    let depth_key = float_to_sort_key(depths[idx]);

    let tile_min_x = u32(max(0.0, (mean.x - f32(r)) / 16.0));
    let tile_max_x = min(config.num_tiles_x, u32((mean.x + f32(r)) / 16.0) + 1u);
    let tile_min_y = u32(max(0.0, (mean.y - f32(r)) / 16.0));
    let tile_max_y = min(config.num_tiles_y, u32((mean.y + f32(r)) / 16.0) + 1u);

    var offset = offsets[idx];
    for (var ty = tile_min_y; ty < tile_max_y; ty++) {
        for (var tx = tile_min_x; tx < tile_max_x; tx++) {
            let tile_id = ty * config.num_tiles_x + tx;
            keys[offset] = vec2<u32>(tile_id, depth_key);
            values[offset] = idx;
            offset++;
        }
    }
}
"#;

/// Prefix sum (scan) shader for computing offsets from tile counts.
/// Uses a simple work-efficient parallel scan (Blelloch).
pub const PREFIX_SUM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // (n, 0, 0, 0)

var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let n = params.x;
    let block_offset = wg.x * 512u;
    let ai = lid;
    let bi = lid + 256u;

    // Load into shared memory
    temp[ai] = select(0u, data[block_offset + ai], block_offset + ai < n);
    temp[bi] = select(0u, data[block_offset + bi], block_offset + bi < n);

    // Up-sweep (reduce)
    var offset = 1u;
    var d = 256u;
    loop {
        if d == 0u { break; }
        workgroupBarrier();
        if lid < d {
            let ai2 = offset * (2u * lid + 1u) - 1u;
            let bi2 = offset * (2u * lid + 2u) - 1u;
            if bi2 < 512u {
                temp[bi2] += temp[ai2];
            }
        }
        offset *= 2u;
        d /= 2u;
    }

    // Set last element to 0 (exclusive scan)
    if lid == 0u {
        temp[511u] = 0u;
    }

    // Down-sweep
    d = 1u;
    loop {
        if d > 256u { break; }
        offset /= 2u;
        workgroupBarrier();
        if lid < d {
            let ai2 = offset * (2u * lid + 1u) - 1u;
            let bi2 = offset * (2u * lid + 2u) - 1u;
            if bi2 < 512u {
                let t = temp[ai2];
                temp[ai2] = temp[bi2];
                temp[bi2] += t;
            }
        }
        d *= 2u;
    }

    workgroupBarrier();

    // Write results
    if block_offset + ai < n {
        data[block_offset + ai] = temp[ai];
    }
    if block_offset + bi < n {
        data[block_offset + bi] = temp[bi];
    }
}
"#;

/// Radix sort shader — sorts (key, value) pairs by key.
/// Two-pass counting sort per radix digit (8 bits at a time, 4 passes for 32-bit keys).
/// We sort by tile_id first (high 32 bits), then by depth (low 32 bits).
pub const RADIX_SORT_SHADER: &str = r#"
const RADIX_BITS: u32 = 8u;
const RADIX_SIZE: u32 = 256u; // 2^8
const WG_SIZE: u32 = 256u;

struct SortParams {
    num_pairs: u32,
    bit_offset: u32, // which 8-bit digit we're sorting (0, 8, 16, 24)
    sort_component: u32, // 0 = sort by keys.y (depth), 1 = sort by keys.x (tile_id)
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> keys_in: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> values_in: array<u32>;
@group(0) @binding(2) var<uniform> params: SortParams;

@group(1) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;

// Pass 1: Count occurrences of each radix digit
@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_pairs {
        return;
    }
    let key = select(keys_in[idx].y, keys_in[idx].x, params.sort_component == 1u);
    let digit = (key >> params.bit_offset) & (RADIX_SIZE - 1u);
    atomicAdd(&histogram[digit], 1u);
}

// After prefix-summing the histogram, pass 2 scatters elements to sorted positions.
// This requires a second dispatch with the prefix-summed histogram.
"#;

/// Shader to scatter sorted elements using prefix-summed histogram.
pub const RADIX_SCATTER_SHADER: &str = r#"
const RADIX_BITS: u32 = 8u;
const RADIX_SIZE: u32 = 256u;

struct SortParams {
    num_pairs: u32,
    bit_offset: u32,
    sort_component: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> keys_in: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> values_in: array<u32>;
@group(0) @binding(2) var<uniform> params: SortParams;
@group(0) @binding(3) var<storage, read_write> histogram: array<atomic<u32>>;

@group(1) @binding(0) var<storage, read_write> keys_out: array<vec2<u32>>;
@group(1) @binding(1) var<storage, read_write> values_out: array<u32>;

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_pairs {
        return;
    }
    let key = select(keys_in[idx].y, keys_in[idx].x, params.sort_component == 1u);
    let digit = (key >> params.bit_offset) & (RADIX_SIZE - 1u);
    let dst = atomicAdd(&histogram[digit], 1u);
    keys_out[dst] = keys_in[idx];
    values_out[dst] = values_in[idx];
}
"#;

/// Shader to identify per-tile ranges in the sorted array.
/// After sorting, scan for transitions in tile_id to find where each tile's
/// gaussians begin and end.
pub const IDENTIFY_TILE_RANGES_SHADER: &str = r#"
struct Config {
    num_pairs: u32,
    num_tiles: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> sorted_keys: array<vec2<u32>>;
@group(0) @binding(1) var<uniform> config: Config;

@group(1) @binding(0) var<storage, read_write> tile_ranges: array<vec2<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= config.num_pairs {
        return;
    }

    let tile_id = sorted_keys[idx].x;

    // Check if this is the start of a new tile
    if idx == 0u || sorted_keys[idx - 1u].x != tile_id {
        tile_ranges[tile_id].x = idx;
    }

    // Check if this is the end of a tile
    if idx == config.num_pairs - 1u || sorted_keys[idx + 1u].x != tile_id {
        tile_ranges[tile_id].y = idx + 1u;
    }
}
"#;
