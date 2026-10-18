//! The "type an expression, see the kernel" demo: a tiny parser over
//! `tang-expr`, emitting WGSL / CUDA / MSL / C for the site.

use tang_expr::codegen::Dialect;
use tang_expr::{graph::ExprGraph, node::ExprId};
use wasm_bindgen::prelude::*;

const VARS: [&str; 3] = ["x", "y", "z"];

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    g: &'a mut ExprGraph,
    used: [bool; 3],
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            return true;
        }
        false
    }

    fn expr(&mut self) -> Result<ExprId, String> {
        let mut a = self.term()?;
        loop {
            if self.eat(b'+') {
                let b = self.term()?;
                a = self.g.add(a, b);
            } else if self.eat(b'-') {
                let b = self.term()?;
                let nb = self.g.neg(b);
                a = self.g.add(a, nb);
            } else {
                return Ok(a);
            }
        }
    }

    fn term(&mut self) -> Result<ExprId, String> {
        let mut a = self.unary()?;
        loop {
            if self.eat(b'*') {
                let b = self.unary()?;
                a = self.g.mul(a, b);
            } else if self.eat(b'/') {
                let b = self.unary()?;
                let rb = self.g.recip(b);
                a = self.g.mul(a, rb);
            } else {
                return Ok(a);
            }
        }
    }

    fn unary(&mut self) -> Result<ExprId, String> {
        if self.eat(b'-') {
            let a = self.unary()?;
            return Ok(self.g.neg(a));
        }
        self.power()
    }

    fn power(&mut self) -> Result<ExprId, String> {
        let base = self.atom()?;
        if !self.eat(b'^') {
            return Ok(base);
        }
        let start = self.i;
        let exp = self.unary()?;
        // small integer powers as multiplies, everything else as exp2(e·log2(b))
        let text = core::str::from_utf8(&self.s[start..self.i])
            .unwrap_or("")
            .trim();
        if let Ok(n) = text.parse::<u32>() {
            if (1..=8).contains(&n) {
                let mut r = base;
                for _ in 1..n {
                    r = self.g.mul(r, base);
                }
                return Ok(r);
            }
        }
        let l = self.g.log2(base);
        let m = self.g.mul(exp, l);
        Ok(self.g.exp2(m))
    }

    fn atom(&mut self) -> Result<ExprId, String> {
        self.ws();
        if self.eat(b'(') {
            let e = self.expr()?;
            return if self.eat(b')') {
                Ok(e)
            } else {
                Err("expected )".into())
            };
        }
        let start = self.i;
        let c = *self.s.get(self.i).ok_or("unexpected end")?;
        if c.is_ascii_digit() || c == b'.' {
            while self.i < self.s.len()
                && (self.s[self.i].is_ascii_digit() || self.s[self.i] == b'.')
            {
                self.i += 1;
            }
            let t = core::str::from_utf8(&self.s[start..self.i]).unwrap();
            let v: f64 = t.parse().map_err(|_| format!("bad number {t}"))?;
            return Ok(self.g.lit(v));
        }
        if !c.is_ascii_alphabetic() {
            return Err(format!("unexpected '{}'", c as char));
        }
        while self.i < self.s.len() && self.s[self.i].is_ascii_alphanumeric() {
            self.i += 1;
        }
        let name = core::str::from_utf8(&self.s[start..self.i]).unwrap();
        if let Some(k) = VARS.iter().position(|v| *v == name) {
            self.used[k] = true;
            return Ok(self.g.var(k as u16));
        }
        match name {
            "pi" => return Ok(self.g.lit(core::f64::consts::PI)),
            "e" => return Ok(self.g.lit(core::f64::consts::E)),
            _ => {}
        }
        if !self.eat(b'(') {
            return Err(format!("unknown name {name}"));
        }
        let a = self.expr()?;
        let b = if self.eat(b',') {
            Some(self.expr()?)
        } else {
            None
        };
        if !self.eat(b')') {
            return Err("expected )".into());
        }
        let g = &mut *self.g;
        Ok(match (name, b) {
            ("sin", None) => g.sin(a),
            ("cos", None) => {
                let h = g.lit(core::f64::consts::FRAC_PI_2);
                let s = g.add(a, h);
                g.sin(s)
            }
            ("sqrt", None) => g.sqrt(a),
            ("exp", None) => {
                let k = g.lit(core::f64::consts::LOG2_E);
                let m = g.mul(a, k);
                g.exp2(m)
            }
            ("ln", None) | ("log", None) => {
                let l = g.log2(a);
                let k = g.lit(core::f64::consts::LN_2);
                g.mul(l, k)
            }
            ("tanh", None) => {
                // tanh(a) = 1 - 2 / (exp(2a) + 1)
                let k = g.lit(2.0 * core::f64::consts::LOG2_E);
                let m = g.mul(a, k);
                let e = g.exp2(m);
                let one = g.lit(1.0);
                let d = g.add(e, one);
                let r = g.recip(d);
                let two = g.lit(-2.0);
                let t = g.mul(two, r);
                g.add(one, t)
            }
            ("atan2", Some(b)) => g.atan2(a, b),
            _ => return Err(format!("unknown function {name}")),
        })
    }
}

/// Parse `src` in x, y, z into a graph; outputs are the simplified expression
/// and, with `grad`, its simplified partials. Also returns the input count.
pub(crate) fn parse_graph(
    src: &str,
    grad: bool,
) -> Result<(ExprGraph, Vec<ExprId>, usize), String> {
    let mut g = ExprGraph::new();
    let mut p = Parser {
        s: src.as_bytes(),
        i: 0,
        g: &mut g,
        used: [false; 3],
    };
    let root = p.expr()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("unexpected '{}'", p.s[p.i] as char));
    }
    let n_inputs = p.used.iter().rposition(|u| *u).map_or(1, |k| k + 1);
    let mut outs = vec![g.simplify(root)];
    if grad {
        for k in 0..n_inputs {
            let d = g.diff(root, k as u16);
            outs.push(g.simplify(d));
        }
    }
    Ok((g, outs, n_inputs))
}

/// Parse `src` in x, y, z, optionally append its gradient, and emit a kernel.
/// Returns the source, or a line starting with `error:`.
#[wasm_bindgen]
pub fn expr_kernel(src: &str, dialect: &str, grad: bool) -> String {
    let dialect = match dialect {
        "cuda" => Dialect::Cuda,
        "msl" => Dialect::Msl,
        "c" => Dialect::C,
        _ => Dialect::Wgsl,
    };
    let (g, outs, n_inputs) = match parse_graph(src, grad) {
        Ok(r) => r,
        Err(e) => return format!("error: {e}"),
    };
    g.to_kernel(&outs, n_inputs, dialect).source
}
