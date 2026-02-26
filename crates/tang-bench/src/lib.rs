//! Shared helpers for tang benchmarks: seeded RNG, input generators.

use tang::{Mat3, Mat4, Quat, Vec3};
use tang_la::{DMat, DVec};

/// Simple xoshiro256** PRNG for reproducible benchmarks (no rand dependency in lib).
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to expand seed into state
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [-1, 1]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64) * 2.0 - 1.0
    }

    /// Uniform f32 in [-1, 1]
    pub fn f32(&mut self) -> f32 {
        self.f64() as f32
    }
}

pub fn make_rng() -> Rng {
    Rng::new(0xDEAD_BEEF_CAFE_BABE)
}

// --- tang generators ---

pub fn random_tang_vec3f32(n: usize) -> Vec<Vec3<f32>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| Vec3::new(rng.f32(), rng.f32(), rng.f32()))
        .collect()
}

pub fn random_tang_vec3f64(n: usize) -> Vec<Vec3<f64>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| Vec3::new(rng.f64(), rng.f64(), rng.f64()))
        .collect()
}

pub fn random_tang_mat3f64(n: usize) -> Vec<Mat3<f64>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            Mat3::new(
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
            )
        })
        .collect()
}

pub fn random_tang_mat4f64(n: usize) -> Vec<Mat4<f64>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            Mat4::new(
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
            )
        })
        .collect()
}

pub fn random_tang_mat4f32(n: usize) -> Vec<Mat4<f32>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            Mat4::new(
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
            )
        })
        .collect()
}

pub fn random_tang_quat(n: usize) -> Vec<Quat<f64>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| Quat::new(rng.f64(), rng.f64(), rng.f64(), rng.f64()).normalize())
        .collect()
}

// --- DMat/DVec generators ---

pub fn random_dmat(size: usize) -> DMat<f64> {
    let flat = random_f64_flat(size * size);
    DMat::from_fn(size, size, |i, j| flat[j * size + i])
}

pub fn random_dvec(size: usize) -> DVec<f64> {
    let flat = random_f64_flat(size);
    DVec::from_fn(size, |i| flat[i])
}

/// Symmetric positive-definite matrix: A^T * A + εI
pub fn random_spd_dmat(size: usize) -> DMat<f64> {
    let a = random_dmat(size);
    let at = a.transpose();
    let ata = at.mul_mat(&a);
    // ata + 0.1 * I to ensure positive definiteness
    let mut result = ata;
    for i in 0..size {
        result.set(i, i, result.get(i, i) + 0.1);
    }
    result
}

// --- nalgebra generators (used only in bench files via dev-dependencies) ---
// These are generic over the nalgebra types which are only available in benches.
// We provide the data as raw Vecs that bench code converts.

pub fn random_f64_triples(n: usize) -> Vec<[f64; 3]> {
    let mut rng = make_rng();
    (0..n).map(|_| [rng.f64(), rng.f64(), rng.f64()]).collect()
}

pub fn random_f32_triples(n: usize) -> Vec<[f32; 3]> {
    let mut rng = make_rng();
    (0..n).map(|_| [rng.f32(), rng.f32(), rng.f32()]).collect()
}

pub fn random_f64_quads(n: usize) -> Vec<[f64; 4]> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| [rng.f64(), rng.f64(), rng.f64(), rng.f64()])
        .collect()
}

pub fn random_f64_mat4s(n: usize) -> Vec<[f64; 16]> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            let mut m = [0.0f64; 16];
            for v in &mut m {
                *v = rng.f64();
            }
            m
        })
        .collect()
}

pub fn random_f32_mat4s(n: usize) -> Vec<[f32; 16]> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            let mut m = [0.0f32; 16];
            for v in &mut m {
                *v = rng.f32();
            }
            m
        })
        .collect()
}

pub fn random_f64_mat3s(n: usize) -> Vec<[f64; 9]> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            let mut m = [0.0f64; 9];
            for v in &mut m {
                *v = rng.f64();
            }
            m
        })
        .collect()
}

pub fn random_f64_flat(size: usize) -> Vec<f64> {
    let mut rng = make_rng();
    (0..size).map(|_| rng.f64()).collect()
}

/// SPD matrix as flat column-major data (size x size)
pub fn random_spd_flat(size: usize) -> Vec<f64> {
    let mut rng = Rng::new(0xDEAD_BEEF_CAFE_BABE);
    // Generate random A
    let a: Vec<f64> = (0..size * size).map(|_| rng.f64()).collect();
    // Compute A^T * A + 0.1 * I
    let mut result = vec![0.0f64; size * size];
    for i in 0..size {
        for j in 0..size {
            let mut sum = 0.0;
            for k in 0..size {
                // a is column-major: a[k + i*size] = A[k][i]
                sum += a[i * size + k] * a[j * size + k]; // A^T[i][k] * A[k][j]
            }
            if i == j {
                sum += 0.1;
            }
            result[j * size + i] = sum; // column-major
        }
    }
    result
}
