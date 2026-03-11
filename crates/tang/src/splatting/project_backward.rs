//! Backward projection: chain dL/d{conic, mean_2d} through projection
//! to get dL/d{position, scale, rotation}.
//!
//! Per-gaussian operation, runs on CPU. Not the bottleneck (rasterization is).
//!
//! Forward chain:
//!   position → pos_cam → mean_2d, J
//!   scale, rotation → cov3d
//!   J, W, cov3d → cov2d → conic
//!
//! Backward chain (explicit scalar derivatives, no matrix inverse formula):
//!   dL/dconic → dL/d{a,b,c} (cov2d params) via explicit Jacobian
//!   dL/d{a,b,c} → dL/dcov3d via T matrix
//!   dL/dcov3d → dL/d{RS} → dL/d{scale, rotation}
//!   dL/dmean_2d + dL/dJ → dL/dpos_cam → dL/dposition

use super::camera::Camera;

/// Compute backward projection for all gaussians.
///
/// Returns (dL_dpositions, dL_dlog_scales, dL_drotations).
#[allow(non_snake_case)]
pub fn backward_projection(
    positions: &[[f32; 3]],
    log_scales: &[[f32; 3]],
    rotations: &[[f32; 4]],
    camera: &Camera,
    radii: &[u32],
    dL_dconics: &[[f32; 3]],
    dL_dmeans2d: &[[f32; 2]],
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 4]>) {
    let n = positions.len();
    let mut dL_dpos = vec![[0.0f32; 3]; n];
    let mut dL_dscale = vec![[0.0f32; 3]; n];
    let mut dL_drot = vec![[0.0f32; 4]; n];

    let view = &camera.view_matrix;
    // Extract 3x3 rotation part of view matrix (column-major storage)
    // w[row][col] = view matrix element at (row, col)
    let w = [
        [view[0], view[4], view[8]],
        [view[1], view[5], view[9]],
        [view[2], view[6], view[10]],
    ];

    let fx = camera.intrinsics.fx;
    let fy = camera.intrinsics.fy;

    for i in 0..n {
        if radii[i] == 0 {
            continue;
        }

        let pos = positions[i];
        let s = [
            log_scales[i][0].exp(),
            log_scales[i][1].exp(),
            log_scales[i][2].exp(),
        ];
        let q = normalize4(rotations[i]);

        // === Recompute forward values ===
        let pos_cam = mat4x4_transform(view, pos);
        let depth = -pos_cam[2];
        if depth <= 0.0 {
            continue;
        }
        let z2 = depth * depth;

        let r = quat_to_mat3(q);
        let rs = [
            [r[0][0] * s[0], r[0][1] * s[1], r[0][2] * s[2]],
            [r[1][0] * s[0], r[1][1] * s[1], r[1][2] * s[2]],
            [r[2][0] * s[0], r[2][1] * s[1], r[2][2] * s[2]],
        ];
        let cov3d = mat3_mul_transpose(rs, rs);

        // J = Jacobian of perspective projection (only rows 0,1 matter for 2D cov)
        // J = [[fx/z,  0,      -fx*px/z²],
        //      [0,     fy/z,   -fy*py/z²],
        //      [0,     0,       0        ]]
        // where px=pos_cam[0], py=pos_cam[1], z=depth=-pos_cam[2]
        let j00 = fx / depth;
        let j11 = fy / depth;
        let j02 = -fx * pos_cam[0] / z2;
        let j12 = -fy * pos_cam[1] / z2;

        // T = J * W (only rows 0 and 1 matter)
        // T_row0 = J[0][0]*W_row0 + J[0][2]*W_row2
        // T_row1 = J[1][1]*W_row1 + J[1][2]*W_row2
        let t0 = [
            j00 * w[0][0] + j02 * w[2][0],
            j00 * w[0][1] + j02 * w[2][1],
            j00 * w[0][2] + j02 * w[2][2],
        ];
        let t1 = [
            j11 * w[1][0] + j12 * w[2][0],
            j11 * w[1][1] + j12 * w[2][1],
            j11 * w[1][2] + j12 * w[2][2],
        ];

        // cov2d upper-left 2x2: a = T[0]*cov3d*T[0]^T, etc.
        let v0 = mat3_vec(cov3d, t0); // cov3d * T[0]
        let v1 = mat3_vec(cov3d, t1); // cov3d * T[1]

        let a = dot3(t0, v0) + 0.3; // cov2d[0][0] + low-pass
        let b = dot3(t0, v1);        // cov2d[0][1]
        let c = dot3(t1, v1) + 0.3;  // cov2d[1][1] + low-pass

        let det = a * c - b * b;
        if det <= 0.0 {
            continue;
        }
        let det2_inv = 1.0 / (det * det);

        // === 1. dL/d{conic} → dL/d{a, b, c} via explicit scalar Jacobian ===
        // conic = (c/det, -b/det, a/det) where det = ac - b²
        let dL_cx = dL_dconics[i][0]; // dL/d(conic.x)
        let dL_cy = dL_dconics[i][1]; // dL/d(conic.y)
        let dL_cz = dL_dconics[i][2]; // dL/d(conic.z)

        let dL_da = det2_inv
            * (-c * c * dL_cx + b * c * dL_cy - b * b * dL_cz);
        let dL_db = det2_inv
            * (2.0 * b * c * dL_cx - (a * c + b * b) * dL_cy + 2.0 * a * b * dL_cz);
        let dL_dc_val = det2_inv
            * (-b * b * dL_cx + a * b * dL_cy - a * a * dL_cz);

        // === 2. dL/d{a,b,c} → dL/d(cov3d) ===
        // cov2d_00 = t0 · cov3d · t0^T, cov2d_01 = t0 · cov3d · t1^T, cov2d_11 = t1 · cov3d · t1^T
        // dL/d(cov3d[p][q]) = dL_da * t0[p]*t0[q] + dL_db * t0[p]*t1[q] + dL_dc * t1[p]*t1[q]
        let mut dL_dcov3d = [[0.0f32; 3]; 3];
        for p in 0..3 {
            for qq in 0..3 {
                dL_dcov3d[p][qq] = dL_da * t0[p] * t0[qq]
                    + dL_db * t0[p] * t1[qq]
                    + dL_dc_val * t1[p] * t1[qq];
            }
        }

        // === 3. dL/d(cov3d) → dL/d(RS) → dL/d{scale, rotation} ===
        // cov3d = RS * RS^T → dL/d(RS) = (dL_dcov3d + dL_dcov3d^T) * RS
        let mut dL_drs = [[0.0f32; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                for k in 0..3 {
                    dL_drs[row][col] +=
                        (dL_dcov3d[row][k] + dL_dcov3d[k][row]) * rs[k][col];
                }
            }
        }

        // RS[:,j] = R[:,j] * s[j]
        for j_idx in 0..3 {
            let mut ds = 0.0;
            for k in 0..3 {
                ds += r[k][j_idx] * dL_drs[k][j_idx];
            }
            // Chain: d(exp(log_s))/d(log_s) = s
            dL_dscale[i][j_idx] = ds * s[j_idx];
        }

        let mut dL_dr = [[0.0f32; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                dL_dr[row][col] = dL_drs[row][col] * s[col];
            }
        }
        dL_drot[i] = dR_dquat(q, dL_dr);

        // === 4. dL/d{a,b,c} → dL/d(T) → dL/d(J) → dL/d(pos_cam) ===
        // dL/d(T[0][m]) = 2*dL_da * v0[m] + dL_db * v1[m]
        // dL/d(T[1][m]) = dL_db * v0[m] + 2*dL_dc * v1[m]
        let mut dL_dt0 = [0.0f32; 3];
        let mut dL_dt1 = [0.0f32; 3];
        for m in 0..3 {
            dL_dt0[m] = 2.0 * dL_da * v0[m] + dL_db * v1[m];
            dL_dt1[m] = dL_db * v0[m] + 2.0 * dL_dc_val * v1[m];
        }

        // T_row0 = J[0][0]*W_row0 + J[0][2]*W_row2
        // T_row1 = J[1][1]*W_row1 + J[1][2]*W_row2
        let dL_dj00 = dot3(dL_dt0, w[0]);
        let dL_dj11 = dot3(dL_dt1, w[1]);
        let dL_dj02 = dot3(dL_dt0, w[2]);
        let dL_dj12 = dot3(dL_dt1, w[2]);

        // Chain J elements back to pos_cam:
        // J[0][0] = fx/depth       → dJ00/d(pos_cam.z) = fx/depth² (depth = -pos_cam.z)
        // J[1][1] = fy/depth       → dJ11/d(pos_cam.z) = fy/depth²
        // J[0][2] = -fx*px/z²      → dJ02/d(pos_cam.x) = -fx/z², dJ02/d(pos_cam.z) = -2*fx*px/z³ = 2*j02/depth
        // J[1][2] = -fy*py/z²      → dJ12/d(pos_cam.y) = -fy/z², dJ12/d(pos_cam.z) = -2*fy*py/z³ = 2*j12/depth
        let mut dL_dpos_cam = [0.0f32; 3];

        // dL/d(pos_cam.x) from J[0][2]
        dL_dpos_cam[0] += dL_dj02 * (-fx / z2);
        // dL/d(pos_cam.y) from J[1][2]
        dL_dpos_cam[1] += dL_dj12 * (-fy / z2);
        // dL/d(pos_cam.z) from J[0][0], J[1][1], J[0][2], J[1][2]
        dL_dpos_cam[2] += dL_dj00 * (fx / z2)
            + dL_dj11 * (fy / z2)
            + dL_dj02 * (2.0 * j02 / depth)
            + dL_dj12 * (2.0 * j12 / depth);

        // === 5. dL/d(mean_2d) → dL/d(pos_cam) ===
        // mean_2d_x = fx * pos_cam.x / depth + cx
        // mean_2d_y = fy * (-pos_cam.y) / depth + cy
        let dL_dm = dL_dmeans2d[i];
        dL_dpos_cam[0] += dL_dm[0] * (fx / depth);
        dL_dpos_cam[1] += dL_dm[1] * (-fy / depth);
        dL_dpos_cam[2] += dL_dm[0] * (fx * pos_cam[0] / z2)
            + dL_dm[1] * (-fy * pos_cam[1] / z2);

        // === 6. dL/d(pos_cam) → dL/d(position) via W^T ===
        for j_idx in 0..3 {
            for k in 0..3 {
                dL_dpos[i][j_idx] += w[k][j_idx] * dL_dpos_cam[k];
            }
        }
    }

    (dL_dpos, dL_dscale, dL_drot)
}

// --- Math helpers ---

fn normalize4(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len < 1e-10 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
}

fn mat4x4_transform(m: &[f32; 16], p: [f32; 3]) -> [f32; 3] {
    // Column-major 4x4 * [p, 1]
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ]
}

/// Quaternion (w, x, y, z) → 3x3 rotation matrix (row-major).
fn quat_to_mat3(q: [f32; 4]) -> [[f32; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;

    [
        [1.0 - yy - zz, xy + wz, xz - wy],
        [xy - wz, 1.0 - xx - zz, yz + wx],
        [xz + wy, yz - wx, 1.0 - xx - yy],
    ]
}

/// Gradient of rotation matrix w.r.t. quaternion.
/// Given dL/dR (3x3 row-major), compute dL/dq (4-vector: w, x, y, z).
#[allow(non_snake_case)]
fn dR_dquat(q: [f32; 4], dL_dR: [[f32; 3]; 3]) -> [f32; 4] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let m = dL_dR;

    // dL/dq_i = sum_{j,k} dL/dR_{jk} * dR_{jk}/dq_i
    // Derivatives of R elements w.r.t. quaternion components:
    let dL_dw = 2.0
        * (m[0][1] * z + m[0][2] * (-y) + m[1][0] * (-z) + m[1][2] * x + m[2][0] * y
            + m[2][1] * (-x));

    let dL_dx = 2.0
        * (m[0][1] * y + m[0][2] * z + m[1][0] * y + m[1][1] * (-2.0 * x) + m[1][2] * w
            + m[2][0] * z
            + m[2][1] * (-w)
            + m[2][2] * (-2.0 * x));

    let dL_dy = 2.0
        * (m[0][0] * (-2.0 * y) + m[0][1] * x + m[0][2] * (-w) + m[1][0] * x + m[1][2] * z
            + m[2][0] * w
            + m[2][1] * z
            + m[2][2] * (-2.0 * y));

    let dL_dz = 2.0
        * (m[0][0] * (-2.0 * z) + m[0][1] * w + m[0][2] * x + m[1][0] * (-w)
            + m[1][1] * (-2.0 * z)
            + m[1][2] * y
            + m[2][0] * x
            + m[2][1] * y);

    [dL_dw, dL_dx, dL_dy, dL_dz]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Matrix-vector multiply: M * v (M is 3x3 row-major).
fn mat3_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// A * B^T
fn mat3_mul_transpose(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    c
}
