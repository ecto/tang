use crate::Tape;
use alloc::sync::Arc;
use alloc::vec;
use core::ops::{Add, Div, Mul, Neg, Sub};
use tang_la::DVec;

/// A variable tracked on the AD tape.
///
/// Supports standard arithmetic operations. Call `backward()` on the final
/// result to compute gradients w.r.t. all input variables.
#[derive(Clone, Debug)]
pub struct Var {
    pub(crate) index: usize,
    pub(crate) value: f64,
    pub(crate) tape: Arc<Tape>,
}

impl Var {
    /// The current value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Compute gradients via reverse-mode AD.
    ///
    /// Returns a vector of gradients indexed by variable position on the tape.
    pub fn backward(&self) -> DVec<f64> {
        let ops = self.tape.ops.borrow();
        let n = ops.len();
        let mut adjoints = vec![0.0_f64; n];
        adjoints[self.index] = 1.0;

        // Reverse sweep
        for i in (0..n).rev() {
            let adj = adjoints[i];
            if adj == 0.0 {
                continue;
            }
            let op = &ops[i];
            if op.num_inputs >= 1 {
                adjoints[op.inputs[0]] += adj * op.partials[0];
            }
            if op.num_inputs >= 2 {
                adjoints[op.inputs[1]] += adj * op.partials[1];
            }
        }

        DVec::from_vec(adjoints)
    }

    /// Sine.
    pub fn sin(&self) -> Var {
        self.tape
            .unary(self.index, self.value.sin(), self.value.cos())
    }

    /// Cosine.
    pub fn cos(&self) -> Var {
        self.tape
            .unary(self.index, self.value.cos(), -self.value.sin())
    }

    /// Exponential.
    pub fn exp(&self) -> Var {
        let e = self.value.exp();
        self.tape.unary(self.index, e, e)
    }

    /// Natural log.
    pub fn ln(&self) -> Var {
        self.tape
            .unary(self.index, self.value.ln(), 1.0 / self.value)
    }

    /// Square root.
    pub fn sqrt(&self) -> Var {
        let s = self.value.sqrt();
        self.tape.unary(self.index, s, 0.5 / s)
    }

    /// Absolute value.
    pub fn abs(&self) -> Var {
        self.tape
            .unary(self.index, self.value.abs(), self.value.signum())
    }

    /// Tanh.
    pub fn tanh(&self) -> Var {
        let t = self.value.tanh();
        self.tape.unary(self.index, t, 1.0 - t * t)
    }

    /// Power.
    pub fn powf(&self, p: f64) -> Var {
        let val = self.value.powf(p);
        self.tape
            .unary(self.index, val, p * self.value.powf(p - 1.0))
    }

    /// Create a constant (no gradient).
    pub fn constant(tape: &Arc<Tape>, value: f64) -> Var {
        let mut ops = tape.ops.borrow_mut();
        let index = ops.len();
        ops.push(crate::tape::Op {
            inputs: [0, 0],
            num_inputs: 0,
            partials: [0.0, 0.0],
        });
        Var {
            index,
            value,
            tape: Arc::clone(tape),
        }
    }
}

impl Add for &Var {
    type Output = Var;
    fn add(self, rhs: &Var) -> Var {
        self.tape
            .binary(self.index, rhs.index, self.value + rhs.value, 1.0, 1.0)
    }
}

impl Sub for &Var {
    type Output = Var;
    fn sub(self, rhs: &Var) -> Var {
        self.tape
            .binary(self.index, rhs.index, self.value - rhs.value, 1.0, -1.0)
    }
}

impl Mul for &Var {
    type Output = Var;
    fn mul(self, rhs: &Var) -> Var {
        self.tape.binary(
            self.index,
            rhs.index,
            self.value * rhs.value,
            rhs.value,
            self.value,
        )
    }
}

impl Div for &Var {
    type Output = Var;
    fn div(self, rhs: &Var) -> Var {
        let inv = 1.0 / rhs.value;
        self.tape.binary(
            self.index,
            rhs.index,
            self.value * inv,
            inv,
            -self.value * inv * inv,
        )
    }
}

impl Neg for &Var {
    type Output = Var;
    fn neg(self) -> Var {
        self.tape.unary(self.index, -self.value, -1.0)
    }
}

// Owned variants
impl Add for Var {
    type Output = Var;
    fn add(self, rhs: Var) -> Var {
        (&self) + (&rhs)
    }
}

impl Sub for Var {
    type Output = Var;
    fn sub(self, rhs: Var) -> Var {
        (&self) - (&rhs)
    }
}

impl Mul for Var {
    type Output = Var;
    fn mul(self, rhs: Var) -> Var {
        (&self) * (&rhs)
    }
}

impl Div for Var {
    type Output = Var;
    fn div(self, rhs: Var) -> Var {
        (&self) / (&rhs)
    }
}

impl Neg for Var {
    type Output = Var;
    fn neg(self) -> Var {
        -(&self)
    }
}

// Scalar * Var and Var * scalar convenience
impl Mul<f64> for &Var {
    type Output = Var;
    fn mul(self, rhs: f64) -> Var {
        self.tape.unary(self.index, self.value * rhs, rhs)
    }
}

impl Mul<f64> for Var {
    type Output = Var;
    fn mul(self, rhs: f64) -> Var {
        (&self) * rhs
    }
}

impl Add<f64> for &Var {
    type Output = Var;
    fn add(self, rhs: f64) -> Var {
        self.tape.unary(self.index, self.value + rhs, 1.0)
    }
}

impl Add<f64> for Var {
    type Output = Var;
    fn add(self, rhs: f64) -> Var {
        (&self) + rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_of_sum() {
        let tape = Tape::new();
        let x = tape.var(3.0);
        let y = tape.var(5.0);
        let z = &x + &y;
        let grads = z.backward();
        assert_eq!(grads[x.index], 1.0);
        assert_eq!(grads[y.index], 1.0);
    }

    #[test]
    fn gradient_of_product() {
        let tape = Tape::new();
        let x = tape.var(3.0);
        let y = tape.var(5.0);
        let z = &x * &y;
        let grads = z.backward();
        assert_eq!(grads[x.index], 5.0); // dz/dx = y
        assert_eq!(grads[y.index], 3.0); // dz/dy = x
    }

    #[test]
    fn gradient_of_square() {
        let tape = Tape::new();
        let x = tape.var(3.0);
        let z = &x * &x;
        let grads = z.backward();
        assert!((grads[x.index] - 6.0).abs() < 1e-10); // d/dx x² = 2x = 6
    }

    #[test]
    fn gradient_of_exp() {
        let tape = Tape::new();
        let x = tape.var(1.0);
        let z = x.exp();
        let grads = z.backward();
        let e = 1.0_f64.exp();
        assert!((grads[0] - e).abs() < 1e-10);
    }

    #[test]
    fn gradient_of_sin() {
        let tape = Tape::new();
        let x = tape.var(0.0);
        let z = x.sin();
        let grads = z.backward();
        assert!((grads[0] - 1.0).abs() < 1e-10); // cos(0) = 1
    }

    #[test]
    fn gradient_chain_rule() {
        // d/dx sin(x²) = 2x cos(x²)
        let tape = Tape::new();
        let x = tape.var(1.0);
        let x2 = &x * &x;
        let z = x2.sin();
        let grads = z.backward();
        let expected = 2.0 * 1.0_f64.cos();
        assert!((grads[x.index] - expected).abs() < 1e-10);
    }

    #[test]
    fn gradient_complex_expr() {
        // f(x, y) = x*y + sin(x)
        // df/dx = y + cos(x), df/dy = x
        let tape = Tape::new();
        let x = tape.var(2.0);
        let y = tape.var(3.0);
        let xy = &x * &y;
        let sx = x.sin();
        let z = xy + sx;
        let grads = z.backward();
        assert!((grads[x.index] - (3.0 + 2.0_f64.cos())).abs() < 1e-10);
        assert!((grads[y.index] - 2.0).abs() < 1e-10);
    }
}
