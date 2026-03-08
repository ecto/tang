//! Projection stage: 3D gaussians → 2D screen space.
//!
//! Computes:
//! - Screen-space mean (pixel coordinates)
//! - 2D covariance via Jacobian of perspective projection
//! - Conic (inverse 2D covariance) for per-pixel evaluation
//! - Screen-space radius (3-sigma cutoff)
//! - Tile overlap count for sorting

/// WGSL compute shader for projecting 3D gaussians to 2D.
///
/// Inputs (storage buffers):
///   - positions [N, 3]
///   - scales [N, 3]
///   - rotations [N, 4]
///   - camera uniform
///
/// Outputs (storage buffers):
///   - means_2d [N, 2] — screen-space center
///   - conics [N, 3] — inverse 2D covariance (a, b, c) where Σ^-1 = [[a,b],[b,c]]
///   - radii [N] — screen-space radius in pixels
///   - depths [N] — camera-space z for sorting
///   - visible [N] — frustum culling mask
pub const PROJECT_SHADER: &str = r#"
struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
    focal: vec2<f32>,
    principal: vec2<f32>,
};

struct Config {
    num_gaussians: u32,
    image_width: u32,
    image_height: u32,
    sh_degree: u32,
    near: f32,
    far: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> scales: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rotations: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
@group(0) @binding(4) var<uniform> config: Config;

@group(1) @binding(0) var<storage, read_write> means_2d: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read_write> conics: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> radii: array<u32>;
@group(1) @binding(3) var<storage, read_write> depths: array<f32>;

// Build rotation matrix from quaternion (w, x, y, z)
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    let x2 = x + x; let y2 = y + y; let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;

    return mat3x3<f32>(
        vec3<f32>(1.0 - yy - zz, xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - xx - zz, yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - xx - yy),
    );
}

// Build 3D covariance from scale and rotation: Σ = R·S·S^T·R^T
fn compute_cov3d(scale: vec3<f32>, q: vec4<f32>) -> mat3x3<f32> {
    let R = quat_to_mat3(q);
    // S is diagonal, so R·S = column-scale of R
    let RS = mat3x3<f32>(
        R[0] * scale.x,
        R[1] * scale.y,
        R[2] * scale.z,
    );
    // Σ = RS · RS^T
    return RS * transpose(RS);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= config.num_gaussians {
        return;
    }

    let pos = positions[idx].xyz;
    let s = exp(scales[idx].xyz); // log-scale → actual scale
    let q = normalize(rotations[idx]); // ensure unit quaternion

    // Transform to camera space
    let view = camera.view_matrix;
    let pos_cam = (view * vec4<f32>(pos, 1.0)).xyz;

    // In right-handed view space, objects in front of camera have negative z.
    // We work with positive depth for sorting/projection.
    let depth = -pos_cam.z;

    // Frustum culling
    if depth < config.near || depth > config.far {
        radii[idx] = 0u;
        return;
    }

    // Project mean to screen
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let cx = camera.principal.x;
    let cy = camera.principal.y;
    let mean_2d_x = fx * pos_cam.x / depth + cx;
    let mean_2d_y = fy * (-pos_cam.y) / depth + cy;

    // Jacobian of perspective projection
    let z2 = depth * depth;
    let J = mat3x3<f32>(
        vec3<f32>(fx / depth, 0.0, 0.0),
        vec3<f32>(0.0, fy / depth, 0.0),
        vec3<f32>(-fx * pos_cam.x / z2, -fy * pos_cam.y / z2, 0.0),
    );

    // View rotation (upper-left 3×3 of view matrix)
    let W = mat3x3<f32>(
        vec3<f32>(view[0][0], view[0][1], view[0][2]),
        vec3<f32>(view[1][0], view[1][1], view[1][2]),
        vec3<f32>(view[2][0], view[2][1], view[2][2]),
    );

    // 3D covariance in world space
    let cov3d = compute_cov3d(s, q);

    // Project to 2D: Σ_2D = J · W · Σ_3D · W^T · J^T
    let T = J * W;
    let cov2d = T * cov3d * transpose(T);

    // Add low-pass filter for stability
    let cov2d_00 = cov2d[0][0] + 0.3;
    let cov2d_01 = cov2d[0][1];
    let cov2d_11 = cov2d[1][1] + 0.3;

    // Compute conic (inverse of 2D covariance)
    let det = cov2d_00 * cov2d_11 - cov2d_01 * cov2d_01;
    if det <= 0.0 {
        radii[idx] = 0u;
        return;
    }
    let inv_det = 1.0 / det;
    let conic = vec3<f32>(cov2d_11 * inv_det, -cov2d_01 * inv_det, cov2d_00 * inv_det);

    // Compute radius from eigenvalues (3-sigma cutoff)
    let mid = 0.5 * (cov2d_00 + cov2d_11);
    let disc = max(0.1, mid * mid - det);
    let lambda_max = mid + sqrt(disc);
    let radius = u32(ceil(3.0 * sqrt(lambda_max)));

    // Bounds check
    if mean_2d_x + f32(radius) < 0.0 || mean_2d_x - f32(radius) >= f32(config.image_width) ||
       mean_2d_y + f32(radius) < 0.0 || mean_2d_y - f32(radius) >= f32(config.image_height) {
        radii[idx] = 0u;
        return;
    }

    means_2d[idx] = vec2<f32>(mean_2d_x, mean_2d_y);
    conics[idx] = vec4<f32>(conic, 0.0);
    radii[idx] = radius;
    depths[idx] = depth;
}
"#;
