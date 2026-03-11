//! Camera model for gaussian splatting.
//!
//! Pinhole camera with intrinsics (focal length, principal point) and
//! extrinsics (view matrix). Handles 3D→2D projection and provides the
//! Jacobian needed for covariance projection.

/// Camera intrinsics.
#[derive(Debug, Clone, Copy)]
pub struct Intrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

/// Pinhole camera with intrinsics and extrinsics.
#[derive(Debug, Clone)]
pub struct Camera {
    pub intrinsics: Intrinsics,
    /// View matrix (world → camera), 4×4 column-major.
    pub view_matrix: [f32; 16],
    /// Projection matrix, 4×4 column-major.
    pub proj_matrix: [f32; 16],
    /// Camera position in world space.
    pub position: [f32; 3],
}

impl Camera {
    /// Create a camera looking at a target from a given position.
    pub fn look_at(
        eye: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
        intrinsics: Intrinsics,
        width: u32,
        height: u32,
        near: f32,
        far: f32,
    ) -> Self {
        let view_matrix = look_at_rh(eye, target, up);
        let proj_matrix = perspective(
            intrinsics.fx, intrinsics.fy,
            intrinsics.cx, intrinsics.cy,
            width, height,
            near, far,
        );

        Self {
            intrinsics,
            view_matrix,
            proj_matrix,
            position: eye,
        }
    }

    /// Pack camera data into a uniform buffer.
    pub fn as_uniform(&self) -> CameraUniform {
        CameraUniform {
            view_matrix: self.view_matrix,
            proj_matrix: self.proj_matrix,
            camera_position: self.position,
            _pad: 0.0,
            focal: [self.intrinsics.fx, self.intrinsics.fy],
            principal: [self.intrinsics.cx, self.intrinsics.cy],
        }
    }
}

/// GPU-friendly camera data (uniform buffer layout).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CameraUniform {
    pub view_matrix: [f32; 16],
    pub proj_matrix: [f32; 16],
    pub camera_position: [f32; 3],
    pub _pad: f32,
    pub focal: [f32; 2],
    pub principal: [f32; 2],
}

/// Right-handed look-at view matrix (column-major).
fn look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = normalize(sub(target, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);

    [
        s[0], u[0], -f[0], 0.0,
        s[1], u[1], -f[1], 0.0,
        s[2], u[2], -f[2], 0.0,
        -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0,
    ]
}

/// Perspective projection from intrinsics (column-major).
fn perspective(
    fx: f32, fy: f32, cx: f32, cy: f32,
    width: u32, height: u32,
    near: f32, far: f32,
) -> [f32; 16] {
    let w = width as f32;
    let h = height as f32;

    [
        2.0 * fx / w, 0.0, 0.0, 0.0,
        0.0, 2.0 * fy / h, 0.0, 0.0,
        1.0 - 2.0 * cx / w, 2.0 * cy / h - 1.0, -(far + near) / (far - near), -1.0,
        0.0, 0.0, -2.0 * far * near / (far - near), 0.0,
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}
