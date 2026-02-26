use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;

/// An operation recorded on the tape.
#[derive(Clone, Debug)]
pub(crate) struct Op {
    /// Indices of inputs in the tape's node list.
    pub inputs: [usize; 2],
    /// Number of actual inputs (1 or 2).
    pub num_inputs: u8,
    /// Partial derivatives w.r.t. each input.
    pub partials: [f64; 2],
}

/// The AD tape that records the computational graph.
#[derive(Debug)]
pub struct Tape {
    pub(crate) ops: RefCell<Vec<Op>>,
}

impl Tape {
    /// Create a new empty tape.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            ops: RefCell::new(Vec::new()),
        })
    }

    /// Create a new input variable.
    pub fn var(self: &Arc<Self>, value: f64) -> super::Var {
        let mut ops = self.ops.borrow_mut();
        let index = ops.len();
        ops.push(Op {
            inputs: [0, 0],
            num_inputs: 0,
            partials: [0.0, 0.0],
        });
        super::Var {
            index,
            value,
            tape: Arc::clone(self),
        }
    }

    /// Record a unary operation.
    pub(crate) fn unary(self: &Arc<Self>, input: usize, value: f64, partial: f64) -> super::Var {
        let mut ops = self.ops.borrow_mut();
        let index = ops.len();
        ops.push(Op {
            inputs: [input, 0],
            num_inputs: 1,
            partials: [partial, 0.0],
        });
        super::Var {
            index,
            value,
            tape: Arc::clone(self),
        }
    }

    /// Record a binary operation.
    pub(crate) fn binary(
        self: &Arc<Self>,
        a: usize,
        b: usize,
        value: f64,
        da: f64,
        db: f64,
    ) -> super::Var {
        let mut ops = self.ops.borrow_mut();
        let index = ops.len();
        ops.push(Op {
            inputs: [a, b],
            num_inputs: 2,
            partials: [da, db],
        });
        super::Var {
            index,
            value,
            tape: Arc::clone(self),
        }
    }
}
