//! tang-3dgs — Differentiable 3D Gaussian Splatting
//!
//! A fully differentiable gaussian splatting rasterizer running on Metal/Vulkan/DX12
//! via wgpu compute shaders. Supports both forward rendering and backward gradient
//! computation for training.
//!
//! # Architecture
//!
//! The pipeline has three stages, each implemented as wgpu compute shaders:
//!
//! 1. **Project**: Transform 3D gaussians to 2D screen space, compute conics
//! 2. **Sort**: GPU radix sort by [tile_id | depth] for front-to-back ordering
//! 3. **Rasterize**: Tile-based alpha compositing (16×16 tiles)
//!
//! The backward pass reverses the rasterization in back-to-front order,
//! computing gradients via atomicAdd per gaussian.
//!
//! # Usage
//!
//! ```ignore
//! use tang_3dgs::{GaussianCloud, Camera, Intrinsics, RasterConfig, Rasterizer};
//!
//! let cloud = GaussianCloud::random(1000, 0);
//! let camera = Camera::look_at(
//!     [0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
//!     Intrinsics { fx: 500.0, fy: 500.0, cx: 256.0, cy: 256.0 },
//!     512, 512, 0.01, 100.0,
//! );
//! let rasterizer = Rasterizer::new(RasterConfig::default());
//! let output = rasterizer.forward(&cloud, &camera);
//! // output.image is [H*W*3] RGB floats
//! ```

pub mod camera;
pub mod cloud;
pub mod densify;
pub mod pipeline;
pub mod project;
pub mod rasterize;
pub mod sort;

pub use camera::{Camera, Intrinsics};
pub use cloud::GaussianCloud;
pub use pipeline::Rasterizer;

/// Tile size for the tile-based rasterizer (16×16 pixels per tile).
pub const TILE_SIZE: u32 = 16;

/// Configuration for the rasterizer.
#[derive(Debug, Clone)]
pub struct RasterConfig {
    pub width: u32,
    pub height: u32,
    /// Spherical harmonics degree (0-3). Higher = more view-dependent color detail.
    pub sh_degree: u32,
    /// Background color [R, G, B] in [0, 1].
    pub bg_color: [f32; 3],
    /// Near plane for frustum culling.
    pub near: f32,
    /// Far plane for frustum culling.
    pub far: f32,
}

impl Default for RasterConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            sh_degree: 0,
            bg_color: [0.0, 0.0, 0.0],
            near: 0.01,
            far: 100.0,
        }
    }
}

/// Context saved during forward pass, needed for backward.
pub struct ForwardContext {
    pub final_transmittance: Vec<f32>,
    pub n_contrib: Vec<u32>,
    pub sorted_indices: Vec<u32>,
    pub tile_ranges: Vec<[u32; 2]>,
    pub means_2d: Vec<[f32; 2]>,
    pub conics: Vec<[f32; 3]>,
    pub radii: Vec<u32>,
}

/// Output of a forward rendering pass.
pub struct RenderOutput {
    /// Rendered image [H*W*3] in row-major order, RGB float [0, 1].
    pub image: Vec<f32>,
    /// Context for backward pass.
    pub ctx: ForwardContext,
}

/// Gradients w.r.t. all gaussian parameters.
pub struct GaussianGradients {
    pub positions: Vec<[f32; 3]>,
    pub scales: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub opacities: Vec<f32>,
    pub sh_coeffs: Vec<f32>,
}
