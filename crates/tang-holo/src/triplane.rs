//! Learnable triplane feature representation.
//!
//! A triplane is three axis-aligned feature planes (XY, XZ, YZ) that encode
//! spatial information. Points are projected onto each plane, features are
//! bilinearly interpolated, then summed.
//!
//! Used as the spatial backbone for the canonical gaussian predictor.

/// A triplane with learnable features.
pub struct Triplane {
    /// Resolution of each plane.
    pub resolution: u32,
    /// Feature dimension per plane.
    pub feature_dim: usize,
    /// XY plane features [resolution * resolution * feature_dim].
    pub xy: Vec<f32>,
    /// XZ plane features [resolution * resolution * feature_dim].
    pub xz: Vec<f32>,
    /// YZ plane features [resolution * resolution * feature_dim].
    pub yz: Vec<f32>,
}

impl Triplane {
    /// Create a triplane with small random initialization.
    pub fn new(resolution: u32, feature_dim: usize) -> Self {
        let n = (resolution * resolution) as usize * feature_dim;
        let scale = 0.01;
        let mut seed = 12345u64;
        let mut rand = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((seed >> 33) as u32) as f32 / u32::MAX as f32;
            (bits - 0.5) * 2.0 * scale
        };

        Self {
            resolution,
            feature_dim,
            xy: (0..n).map(|_| rand()).collect(),
            xz: (0..n).map(|_| rand()).collect(),
            yz: (0..n).map(|_| rand()).collect(),
        }
    }

    /// Query triplane features at a 3D point.
    ///
    /// `point`: (x, y, z) in [-1, 1] normalized coordinates.
    /// Returns: feature vector of length `feature_dim`.
    pub fn query(&self, point: [f32; 3]) -> Vec<f32> {
        let [x, y, z] = point;

        let xy_feat = self.bilinear_sample(&self.xy, x, y);
        let xz_feat = self.bilinear_sample(&self.xz, x, z);
        let yz_feat = self.bilinear_sample(&self.yz, y, z);

        // Sum features from all three planes
        let mut out = vec![0.0f32; self.feature_dim];
        for i in 0..self.feature_dim {
            out[i] = xy_feat[i] + xz_feat[i] + yz_feat[i];
        }
        out
    }

    /// Query features for multiple points, returning [N, feature_dim].
    pub fn query_batch(&self, points: &[[f32; 3]]) -> Vec<f32> {
        let mut out = Vec::with_capacity(points.len() * self.feature_dim);
        for &pt in points {
            out.extend(self.query(pt));
        }
        out
    }

    /// Bilinear interpolation on a single plane.
    fn bilinear_sample(&self, plane: &[f32], u: f32, v: f32) -> Vec<f32> {
        let res = self.resolution as f32;
        // Map [-1, 1] → [0, res-1]
        let px = (u * 0.5 + 0.5) * (res - 1.0);
        let py = (v * 0.5 + 0.5) * (res - 1.0);

        let x0 = (px.floor() as i32).clamp(0, self.resolution as i32 - 1) as usize;
        let y0 = (py.floor() as i32).clamp(0, self.resolution as i32 - 1) as usize;
        let x1 = (x0 + 1).min(self.resolution as usize - 1);
        let y1 = (y0 + 1).min(self.resolution as usize - 1);

        let fx = px - px.floor();
        let fy = py - py.floor();

        let idx = |x: usize, y: usize| (y * self.resolution as usize + x) * self.feature_dim;

        let mut out = vec![0.0f32; self.feature_dim];
        for i in 0..self.feature_dim {
            let v00 = plane[idx(x0, y0) + i];
            let v10 = plane[idx(x1, y0) + i];
            let v01 = plane[idx(x0, y1) + i];
            let v11 = plane[idx(x1, y1) + i];

            out[i] = v00 * (1.0 - fx) * (1.0 - fy)
                + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy
                + v11 * fx * fy;
        }
        out
    }

    /// Total number of learnable parameters.
    pub fn num_params(&self) -> usize {
        let n = (self.resolution * self.resolution) as usize * self.feature_dim;
        n * 3 // three planes
    }

    /// Get all parameters as a flat slice (for optimizer).
    pub fn params_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.num_params());
        out.extend_from_slice(&self.xy);
        out.extend_from_slice(&self.xz);
        out.extend_from_slice(&self.yz);
        out
    }

    /// Set parameters from a flat slice.
    pub fn set_params_flat(&mut self, params: &[f32]) {
        let n = (self.resolution * self.resolution) as usize * self.feature_dim;
        self.xy.copy_from_slice(&params[..n]);
        self.xz.copy_from_slice(&params[n..2 * n]);
        self.yz.copy_from_slice(&params[2 * n..3 * n]);
    }

    /// Get gradients as three separate slices from a flat gradient vector.
    pub fn split_grads<'a>(&self, grads: &'a [f32]) -> (&'a [f32], &'a [f32], &'a [f32]) {
        let n = (self.resolution * self.resolution) as usize * self.feature_dim;
        (&grads[..n], &grads[n..2 * n], &grads[2 * n..3 * n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triplane_query() {
        let tp = Triplane::new(8, 4);
        let feat = tp.query([0.0, 0.0, 0.0]);
        assert_eq!(feat.len(), 4);
        // Center point should be non-zero (random init)
        assert!(feat.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_triplane_params_roundtrip() {
        let tp = Triplane::new(4, 2);
        let params = tp.params_flat();
        let mut tp2 = Triplane::new(4, 2);
        tp2.set_params_flat(&params);
        assert_eq!(tp.xy, tp2.xy);
        assert_eq!(tp.xz, tp2.xz);
        assert_eq!(tp.yz, tp2.yz);
    }
}
