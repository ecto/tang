use alloc::vec::Vec;

/// N-dimensional shape descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn from_slice(dims: &[usize]) -> Self {
        Self { dims: dims.to_vec() }
    }

    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product::<usize>().max(1)
    }

    /// Compute contiguous row-major strides.
    pub fn contiguous_strides(&self) -> Vec<usize> {
        let n = self.dims.len();
        if n == 0 {
            return Vec::new();
        }
        let mut strides = alloc::vec![0usize; n];
        strides[n - 1] = 1;
        for i in (0..n - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Broadcast two shapes according to NumPy rules.
    /// Returns the broadcast shape or None if incompatible.
    pub fn broadcast(a: &Shape, b: &Shape) -> Option<Shape> {
        let n = a.ndim().max(b.ndim());
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let da = if i < n - a.ndim() { 1 } else { a.dims[i - (n - a.ndim())] };
            let db = if i < n - b.ndim() { 1 } else { b.dims[i - (n - b.ndim())] };
            if da == db {
                result.push(da);
            } else if da == 1 {
                result.push(db);
            } else if db == 1 {
                result.push(da);
            } else {
                return None;
            }
        }
        Some(Shape::new(result))
    }
}

impl core::ops::Index<usize> for Shape {
    type Output = usize;
    fn index(&self, i: usize) -> &usize {
        &self.dims[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_basics() {
        let s = Shape::from_slice(&[2, 3, 4]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.contiguous_strides(), alloc::vec![12, 4, 1]);
    }

    #[test]
    fn broadcast_rules() {
        let a = Shape::from_slice(&[3, 1]);
        let b = Shape::from_slice(&[1, 4]);
        assert_eq!(Shape::broadcast(&a, &b), Some(Shape::from_slice(&[3, 4])));

        let a = Shape::from_slice(&[2, 3]);
        let b = Shape::from_slice(&[3]);
        assert_eq!(Shape::broadcast(&a, &b), Some(Shape::from_slice(&[2, 3])));

        let a = Shape::from_slice(&[2, 3]);
        let b = Shape::from_slice(&[4]);
        assert_eq!(Shape::broadcast(&a, &b), None);
    }
}
