//! Quantized tensorization: an `n₁×…×n_D` grid → a `d`-way tensor with tiny
//! mode sizes (mostly 2).
//!
//! Each axis of length `p` is written in mixed radix, `p = f₁·f₂·…·f_L`, and
//! every digit becomes one tensor mode. With `p = 2^L` this is the classic
//! binary QTT ("quantics") tensorization. The digit *order* across axes is a
//! choice:
//!
//! - [`DigitOrder::RowMajor`] — all digits of axis 0 (most significant
//!   first), then axis 1, …: the tensor index is the row-major linear index.
//! - [`DigitOrder::Morton`] — digits interleaved coarse-to-fine, aligned at
//!   the least significant digit (Z-order for powers of two). Neighbouring
//!   cells share long index prefixes along *every* axis.
//!
//! Lengths that are not `7`-smooth are padded. [`PadPolicy::MixedRadix`]
//! pads only to the next 7-smooth length (factors 2,3,5,7 become modes of that
//! size; odd factors are placed most significant), [`PadPolicy::Pow2`] pads to
//! the next power of two (pure binary QTT). Padding is mirror reflection, so
//! the padded field stays continuous at the seam; the padded cells are coded
//! (they cost bits) but are cropped on decode and never enter error metrics.

/// How digits of different axes are ordered in the tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DigitOrder {
    /// Row-major bit order (axis 0's digits first).
    RowMajor,
    /// Bit-interleaved (Z-order / Morton).
    Morton,
}

/// How non-power-of-two lengths are handled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadPolicy {
    /// Factor into 2,3,5,7 modes; pad only to the next 7-smooth length.
    MixedRadix,
    /// Pad every axis to the next power of two (binary modes only).
    Pow2,
}

/// The tensorization of one grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Original grid shape (row-major, last axis fastest).
    pub dims: Vec<usize>,
    /// Padded grid shape.
    pub padded: Vec<usize>,
    /// Tensor mode sizes, most significant first.
    pub modes: Vec<usize>,
    /// For each tensor mode, the stride of that digit in the padded grid.
    pub strides: Vec<usize>,
    /// Digit order used.
    pub order: DigitOrder,
    /// Padding policy used.
    pub pad: PadPolicy,
}

fn smooth_factors(mut p: usize) -> Option<Vec<usize>> {
    let mut f = Vec::new();
    for q in [7usize, 5, 3, 2] {
        while p.is_multiple_of(q) {
            f.push(q);
            p /= q;
        }
    }
    (p == 1).then_some(f)
}

/// Axis length after padding under `pad`, and its digits (most significant first).
pub fn padded_len(n: usize, pad: PadPolicy) -> (usize, Vec<usize>) {
    assert!(n >= 1);
    match pad {
        PadPolicy::Pow2 => {
            let p = n.next_power_of_two();
            (p, vec![2; p.trailing_zeros() as usize])
        }
        PadPolicy::MixedRadix => {
            let mut p = n;
            loop {
                if let Some(f) = smooth_factors(p) {
                    // smooth_factors yields 7s,5s,3s then 2s: odd radices most
                    // significant, binary digits finest.
                    return (p, f);
                }
                p += 1;
            }
        }
    }
}

impl Layout {
    /// Build the tensorization of a grid of shape `dims`.
    pub fn new(dims: &[usize], order: DigitOrder, pad: PadPolicy) -> Self {
        assert!(!dims.is_empty(), "Layout: empty shape");
        let mut padded = Vec::new();
        let mut digits: Vec<Vec<usize>> = Vec::new();
        for &n in dims {
            let (p, f) = padded_len(n, pad);
            padded.push(p);
            digits.push(f);
        }
        // Row-major strides of the padded grid.
        let mut axis_stride = vec![1usize; dims.len()];
        for a in (0..dims.len().saturating_sub(1)).rev() {
            axis_stride[a] = axis_stride[a + 1] * padded[a + 1];
        }
        // Weight of each digit within its axis (product of finer digits).
        let digit_weight = |a: usize, j: usize| -> usize { digits[a][j + 1..].iter().product() };
        let mut modes = Vec::new();
        let mut strides = Vec::new();
        match order {
            DigitOrder::RowMajor => {
                for a in 0..dims.len() {
                    for j in 0..digits[a].len() {
                        modes.push(digits[a][j]);
                        strides.push(digit_weight(a, j) * axis_stride[a]);
                    }
                }
            }
            DigitOrder::Morton => {
                let lmax = digits.iter().map(|d| d.len()).max().unwrap_or(0);
                for level in (1..=lmax).rev() {
                    for a in 0..dims.len() {
                        let l = digits[a].len();
                        if l >= level {
                            let j = l - level;
                            modes.push(digits[a][j]);
                            strides.push(digit_weight(a, j) * axis_stride[a]);
                        }
                    }
                }
            }
        }
        if modes.is_empty() {
            // A single cell: one trivial mode keeps the TT machinery uniform.
            modes.push(1);
            strides.push(1);
        }
        Self {
            dims: dims.to_vec(),
            padded,
            modes,
            strides,
            order,
            pad,
        }
    }

    /// Number of original cells.
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    /// Whether the grid is empty (never, by construction).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of tensor entries (padded cells).
    pub fn tensor_len(&self) -> usize {
        self.modes.iter().product()
    }

    /// Mirror-pad `data` (shape `dims`) to the padded grid.
    pub fn pad_grid(&self, data: &[f64]) -> Vec<f64> {
        assert_eq!(data.len(), self.len());
        if self.dims == self.padded {
            return data.to_vec();
        }
        let nd = self.dims.len();
        let total: usize = self.padded.iter().product();
        let mut src_stride = vec![1usize; nd];
        for a in (0..nd.saturating_sub(1)).rev() {
            src_stride[a] = src_stride[a + 1] * self.dims[a + 1];
        }
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; nd];
        for _ in 0..total {
            let mut off = 0;
            for a in 0..nd {
                off += reflect(idx[a], self.dims[a]) * src_stride[a];
            }
            out.push(data[off]);
            for a in (0..nd).rev() {
                idx[a] += 1;
                if idx[a] < self.padded[a] {
                    break;
                }
                idx[a] = 0;
            }
        }
        out
    }

    /// Padded grid → tensor order (gather).
    pub fn to_tensor(&self, padded: &[f64]) -> Vec<f64> {
        let n = self.tensor_len();
        assert_eq!(padded.len(), n);
        if self.is_identity() {
            return padded.to_vec();
        }
        let mut out = Vec::with_capacity(n);
        self.odometer(|src| out.push(padded[src]));
        out
    }

    /// Tensor order → original grid (scatter + crop).
    pub fn from_tensor(&self, tensor: &[f64]) -> Vec<f64> {
        let n = self.tensor_len();
        assert_eq!(tensor.len(), n);
        let grid = if self.is_identity() {
            tensor.to_vec()
        } else {
            let mut g = vec![0.0; n];
            let mut t = 0;
            self.odometer(|src| {
                g[src] = tensor[t];
                t += 1;
            });
            g
        };
        self.crop(&grid)
    }

    fn crop(&self, grid: &[f64]) -> Vec<f64> {
        if self.dims == self.padded {
            return grid.to_vec();
        }
        let nd = self.dims.len();
        let mut pstride = vec![1usize; nd];
        for a in (0..nd.saturating_sub(1)).rev() {
            pstride[a] = pstride[a + 1] * self.padded[a + 1];
        }
        let mut out = Vec::with_capacity(self.len());
        let mut idx = vec![0usize; nd];
        for _ in 0..self.len() {
            let off: usize = (0..nd).map(|a| idx[a] * pstride[a]).sum();
            out.push(grid[off]);
            for a in (0..nd).rev() {
                idx[a] += 1;
                if idx[a] < self.dims[a] {
                    break;
                }
                idx[a] = 0;
            }
        }
        out
    }

    fn is_identity(&self) -> bool {
        // Row-major digit order with strides equal to the natural ones.
        let mut s = 1usize;
        for k in (0..self.modes.len()).rev() {
            if self.strides[k] != s {
                return false;
            }
            s *= self.modes[k];
        }
        true
    }

    /// Visit the padded-grid offset of every tensor entry, in tensor order.
    fn odometer(&self, mut f: impl FnMut(usize)) {
        let d = self.modes.len();
        let n = self.tensor_len();
        let mut digit = vec![0usize; d];
        let mut off = 0usize;
        for _ in 0..n {
            f(off);
            let mut k = d;
            while k > 0 {
                k -= 1;
                digit[k] += 1;
                off += self.strides[k];
                if digit[k] < self.modes[k] {
                    break;
                }
                off -= self.strides[k] * self.modes[k];
                digit[k] = 0;
            }
        }
    }
}

/// Mirror index `i` into `0..n` (reflection without repeating the edge).
fn reflect(i: usize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let r = i % period;
    if r < n {
        r
    } else {
        period - r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rowmajor_pow2_is_identity() {
        let l = Layout::new(&[8, 4], DigitOrder::RowMajor, PadPolicy::Pow2);
        assert_eq!(l.modes, vec![2; 5]);
        let x: Vec<f64> = (0..32).map(|i| i as f64).collect();
        assert_eq!(l.to_tensor(&l.pad_grid(&x)), x);
    }

    #[test]
    fn morton_matches_bit_interleave() {
        let l = Layout::new(&[4, 4], DigitOrder::Morton, PadPolicy::Pow2);
        let x: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let t = l.to_tensor(&x);
        // Tensor index bits (MSB first) = y1 x1 y0 x0.
        for (ti, &v) in t.iter().enumerate() {
            let y = ((ti >> 3) & 1) << 1 | ((ti >> 1) & 1);
            let xx = ((ti >> 2) & 1) << 1 | (ti & 1);
            assert_eq!(v, (y * 4 + xx) as f64);
        }
    }

    #[test]
    fn roundtrip_all_policies() {
        for dims in [
            vec![1000],
            vec![12, 20],
            vec![5, 6, 7],
            vec![11, 3, 9],
            vec![1],
        ] {
            for order in [DigitOrder::RowMajor, DigitOrder::Morton] {
                for pad in [PadPolicy::MixedRadix, PadPolicy::Pow2] {
                    let l = Layout::new(&dims, order, pad);
                    let n = l.len();
                    let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
                    let t = l.to_tensor(&l.pad_grid(&x));
                    assert_eq!(t.len(), l.tensor_len());
                    assert_eq!(l.from_tensor(&t), x, "{dims:?} {order:?} {pad:?}");
                }
            }
        }
    }

    #[test]
    fn mixed_radix_lengths() {
        assert_eq!(padded_len(384, PadPolicy::MixedRadix).0, 384);
        assert_eq!(padded_len(384, PadPolicy::Pow2).0, 512);
        assert_eq!(
            padded_len(500, PadPolicy::MixedRadix).1,
            vec![5, 5, 5, 2, 2]
        );
        assert_eq!(padded_len(11, PadPolicy::MixedRadix).0, 12);
        assert_eq!(reflect(5, 4), 1);
    }
}
