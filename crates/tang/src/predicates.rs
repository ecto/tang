//! Exact geometric predicates using adaptive-precision arithmetic.
//!
//! These predicates give exact results for topological decisions
//! (point orientation, incircle/insphere) regardless of floating-point error.
//! Uses the `robust` crate (Shewchuk's algorithm).

use crate::Point2;
use crate::Point3;

/// Orientation sign
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sign {
    Negative,
    Zero,
    Positive,
}

impl Sign {
    pub fn from_f64(v: f64) -> Self {
        if v > 0.0 {
            Sign::Positive
        } else if v < 0.0 {
            Sign::Negative
        } else {
            Sign::Zero
        }
    }

    pub fn is_positive(self) -> bool {
        self == Sign::Positive
    }
    pub fn is_negative(self) -> bool {
        self == Sign::Negative
    }
    pub fn is_zero(self) -> bool {
        self == Sign::Zero
    }
}

fn c2(p: &Point2<f64>) -> robust::Coord<f64> {
    robust::Coord { x: p.x, y: p.y }
}
fn c3(p: &Point3<f64>) -> robust::Coord3D<f64> {
    robust::Coord3D {
        x: p.x,
        y: p.y,
        z: p.z,
    }
}

/// Orientation of point `c` relative to directed line `a → b`.
/// Positive = left (CCW), Negative = right (CW), Zero = collinear.
pub fn orient2d(a: &Point2<f64>, b: &Point2<f64>, c: &Point2<f64>) -> Sign {
    Sign::from_f64(robust::orient2d(c2(a), c2(b), c2(c)))
}

/// Orientation of point `d` relative to plane through `a, b, c`.
pub fn orient3d(a: &Point3<f64>, b: &Point3<f64>, c: &Point3<f64>, d: &Point3<f64>) -> Sign {
    Sign::from_f64(robust::orient3d(c3(a), c3(b), c3(c), c3(d)))
}

/// Is point `d` inside the circumcircle of triangle `abc`?
/// Triangle must be CCW (positive orient2d).
pub fn incircle(a: &Point2<f64>, b: &Point2<f64>, c: &Point2<f64>, d: &Point2<f64>) -> Sign {
    Sign::from_f64(robust::incircle(c2(a), c2(b), c2(c), c2(d)))
}

/// Is point `e` inside the circumsphere of tetrahedron `abcd`?
pub fn insphere(
    a: &Point3<f64>,
    b: &Point3<f64>,
    c: &Point3<f64>,
    d: &Point3<f64>,
    e: &Point3<f64>,
) -> Sign {
    Sign::from_f64(robust::insphere(c3(a), c3(b), c3(c), c3(d), c3(e)))
}

// Derived predicates

pub fn point_on_segment_2d(p: &Point2<f64>, a: &Point2<f64>, b: &Point2<f64>) -> bool {
    if !orient2d(a, b, p).is_zero() {
        return false;
    }
    let d = *b - *a;
    let t = *p - *a;
    let dot = d.dot(t);
    dot >= 0.0 && dot <= d.norm_sq()
}

pub fn point_on_plane(p: &Point3<f64>, a: &Point3<f64>, b: &Point3<f64>, c: &Point3<f64>) -> bool {
    orient3d(a, b, c, p).is_zero()
}

pub fn are_coplanar(a: &Point3<f64>, b: &Point3<f64>, c: &Point3<f64>, d: &Point3<f64>) -> bool {
    orient3d(a, b, c, d).is_zero()
}

pub fn are_collinear_2d(a: &Point2<f64>, b: &Point2<f64>, c: &Point2<f64>) -> bool {
    orient2d(a, b, c).is_zero()
}

/// Which side of line `a → b` is point `p` on?
/// Returns None if `p` is at an endpoint (for ray-casting use).
pub fn point_side_of_line(p: &Point2<f64>, a: &Point2<f64>, b: &Point2<f64>) -> Option<Sign> {
    let eps = 1e-10;
    let d_a = p.distance_sq(*a);
    let d_b = p.distance_sq(*b);
    if d_a < eps || d_b < eps {
        return None;
    }
    Some(orient2d(a, b, p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccw_triangle() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.0, 1.0);
        assert!(orient2d(&a, &b, &c).is_positive());
    }

    #[test]
    fn collinear() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(2.0, 0.0);
        assert!(orient2d(&a, &b, &c).is_zero());
    }

    #[test]
    fn point_inside_circumcircle() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.0, 1.0);
        let inside = Point2::new(0.25, 0.25);
        assert!(incircle(&a, &b, &c, &inside).is_positive());
    }

    #[test]
    fn orient3d_above_below() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(1.0, 0.0, 0.0);
        let c = Point3::new(0.0, 1.0, 0.0);
        let above = Point3::new(0.0, 0.0, 1.0);
        let below = Point3::new(0.0, 0.0, -1.0);
        // robust crate: orient3d > 0 when d is below the plane (abc is CCW from below)
        let s_above = orient3d(&a, &b, &c, &above);
        let s_below = orient3d(&a, &b, &c, &below);
        // Above and below should have opposite signs
        assert_ne!(s_above, s_below);
        assert!(!s_above.is_zero());
        assert!(!s_below.is_zero());
    }
}
