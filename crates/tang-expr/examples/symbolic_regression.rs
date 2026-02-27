//! Symbolic Regression via genetic programming over expression graphs.
//!
//! Given noisy data from `f(x) = x² · sin(x)`, evolves a population of
//! expression trees to rediscover the formula.
//!
//! ```sh
//! cargo run --example symbolic_regression -p tang-expr
//! ```

use tang_expr::{ExprGraph, ExprId};

// --- Inline LCG PRNG (matches `collocation_random` pattern) -----------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize % n
    }
}

// --- AST representation for genetic programming -----------------------------

#[derive(Clone, Debug)]
enum Expr {
    X,
    Lit(f64),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Sin(Box<Expr>),
    Neg(Box<Expr>),
}

impl Expr {
    /// Count nodes in this tree.
    fn size(&self) -> usize {
        match self {
            Expr::X | Expr::Lit(_) => 1,
            Expr::Sin(a) | Expr::Neg(a) => 1 + a.size(),
            Expr::Add(a, b) | Expr::Mul(a, b) => 1 + a.size() + b.size(),
        }
    }

    /// Convert to ExprGraph node, returning the root ExprId.
    fn to_expr(&self, g: &mut ExprGraph) -> ExprId {
        match self {
            Expr::X => g.var(0),
            Expr::Lit(v) => g.lit(*v),
            Expr::Add(a, b) => {
                let a = a.to_expr(g);
                let b = b.to_expr(g);
                g.add(a, b)
            }
            Expr::Mul(a, b) => {
                let a = a.to_expr(g);
                let b = b.to_expr(g);
                g.mul(a, b)
            }
            Expr::Sin(a) => {
                let a = a.to_expr(g);
                g.sin(a)
            }
            Expr::Neg(a) => {
                let a = a.to_expr(g);
                g.neg(a)
            }
        }
    }

    /// Format as string using ExprGraph's fmt_expr.
    fn format(&self) -> String {
        let mut g = ExprGraph::new();
        let root = self.to_expr(&mut g);
        g.fmt_expr(root)
    }

    /// Evaluate at a point using ExprGraph's compiled eval.
    fn eval_at(&self, x: f64) -> f64 {
        let mut g = ExprGraph::new();
        let root = self.to_expr(&mut g);
        g.eval(root, &[x])
    }
}

// --- Random expression generation -------------------------------------------

fn random_expr(depth: usize, rng: &mut Lcg) -> Expr {
    if depth == 0 || (depth < 3 && rng.uniform() < 0.3) {
        return match rng.range(3) {
            0 => Expr::X,
            1 => Expr::Lit((rng.uniform() * 4.0 - 2.0) * 10.0_f64.powi(-((rng.uniform() * 2.0) as i32))),
            _ => Expr::X,
        };
    }

    match rng.range(5) {
        0 => Expr::Add(
            Box::new(random_expr(depth - 1, rng)),
            Box::new(random_expr(depth - 1, rng)),
        ),
        1 | 2 => Expr::Mul(
            Box::new(random_expr(depth - 1, rng)),
            Box::new(random_expr(depth - 1, rng)),
        ),
        3 => Expr::Sin(Box::new(random_expr(depth - 1, rng))),
        _ => Expr::Neg(Box::new(random_expr(depth - 1, rng))),
    }
}

// --- Mutation operators ------------------------------------------------------

/// Grow mutation: replace a random subtree with a new random one.
fn mutate_grow(expr: &Expr, rng: &mut Lcg) -> Expr {
    let size = expr.size();
    let target = rng.range(size);
    grow_at(expr, target, &mut 0, rng)
}

fn grow_at(expr: &Expr, target: usize, counter: &mut usize, rng: &mut Lcg) -> Expr {
    if *counter == target {
        *counter += expr.size(); // skip subtree
        return random_expr(3, rng);
    }
    *counter += 1;
    match expr {
        Expr::X => Expr::X,
        Expr::Lit(v) => Expr::Lit(*v),
        Expr::Add(a, b) => Expr::Add(
            Box::new(grow_at(a, target, counter, rng)),
            Box::new(grow_at(b, target, counter, rng)),
        ),
        Expr::Mul(a, b) => Expr::Mul(
            Box::new(grow_at(a, target, counter, rng)),
            Box::new(grow_at(b, target, counter, rng)),
        ),
        Expr::Sin(a) => Expr::Sin(Box::new(grow_at(a, target, counter, rng))),
        Expr::Neg(a) => Expr::Neg(Box::new(grow_at(a, target, counter, rng))),
    }
}

/// Point mutation: change a single node's operation.
fn mutate_point(expr: &Expr, rng: &mut Lcg) -> Expr {
    let size = expr.size();
    let target = rng.range(size);
    point_at(expr, target, &mut 0, rng)
}

fn point_at(expr: &Expr, target: usize, counter: &mut usize, rng: &mut Lcg) -> Expr {
    if *counter == target {
        *counter += 1;
        return match expr {
            Expr::X => Expr::Lit(rng.uniform() * 2.0 - 1.0),
            Expr::Lit(_) => Expr::X,
            Expr::Add(a, b) => Expr::Mul(a.clone(), b.clone()),
            Expr::Mul(a, b) => Expr::Add(a.clone(), b.clone()),
            Expr::Sin(a) => Expr::Neg(a.clone()),
            Expr::Neg(a) => Expr::Sin(a.clone()),
        };
    }
    *counter += 1;
    match expr {
        Expr::X => Expr::X,
        Expr::Lit(v) => Expr::Lit(*v),
        Expr::Add(a, b) => Expr::Add(
            Box::new(point_at(a, target, counter, rng)),
            Box::new(point_at(b, target, counter, rng)),
        ),
        Expr::Mul(a, b) => Expr::Mul(
            Box::new(point_at(a, target, counter, rng)),
            Box::new(point_at(b, target, counter, rng)),
        ),
        Expr::Sin(a) => Expr::Sin(Box::new(point_at(a, target, counter, rng))),
        Expr::Neg(a) => Expr::Neg(Box::new(point_at(a, target, counter, rng))),
    }
}

/// Simplify mutation: convert to ExprGraph, simplify, convert back.
/// Falls back to identity if conversion back would be complex.
fn mutate_simplify(expr: &Expr) -> Expr {
    let mut g = ExprGraph::new();
    let root = expr.to_expr(&mut g);
    let simplified = g.simplify(root);
    let s = g.fmt_expr(simplified);

    // Quick check: if simplified form is much shorter, it's better
    let orig = expr.format();
    if s.len() < orig.len() {
        // Re-evaluate to confirm it still works
        let test = g.eval::<f64>(simplified, &[1.0]);
        let orig_test = expr.eval_at(1.0);
        if (test - orig_test).abs() < 1e-10 || (test.is_nan() && orig_test.is_nan()) {
            // Return a new tree built from the simplified graph
            return expr_from_graph(&g, simplified);
        }
    }
    expr.clone()
}

/// Reconstruct an Expr tree from an ExprGraph node.
fn expr_from_graph(g: &ExprGraph, id: ExprId) -> Expr {
    match g.node(id) {
        tang_expr::node::Node::Var(_) => Expr::X,
        tang_expr::node::Node::Lit(bits) => {
            let v = f64::from_bits(bits);
            if v == 0.0 {
                Expr::Lit(0.0)
            } else {
                Expr::Lit(v)
            }
        }
        tang_expr::node::Node::Add(a, b) => {
            Expr::Add(Box::new(expr_from_graph(g, a)), Box::new(expr_from_graph(g, b)))
        }
        tang_expr::node::Node::Mul(a, b) => {
            Expr::Mul(Box::new(expr_from_graph(g, a)), Box::new(expr_from_graph(g, b)))
        }
        tang_expr::node::Node::Neg(a) => Expr::Neg(Box::new(expr_from_graph(g, a))),
        tang_expr::node::Node::Sin(a) => Expr::Sin(Box::new(expr_from_graph(g, a))),
        // For operations we don't represent in our AST, just evaluate as literal
        _ => {
            let v = g.eval::<f64>(id, &[1.0]); // fallback
            Expr::Lit(v)
        }
    }
}

// --- Fitness evaluation ------------------------------------------------------

fn target(x: f64) -> f64 {
    x * x * x.sin()
}

fn generate_data(n: usize, rng: &mut Lcg) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let x = -3.0 + 6.0 * i as f64 / (n - 1) as f64;
            let noise = (rng.uniform() - 0.5) * 0.01;
            (x, target(x) + noise)
        })
        .collect()
}

/// Evaluate fitness: MSE + complexity penalty.
fn fitness(expr: &Expr, data: &[(f64, f64)]) -> f64 {
    let mut g = ExprGraph::new();
    let root = expr.to_expr(&mut g);
    let compiled = g.compile(root);

    let mut mse = 0.0;
    for &(x, y) in data {
        let pred = compiled(&[x]);
        if pred.is_nan() || pred.is_infinite() {
            return f64::INFINITY;
        }
        let err = pred - y;
        mse += err * err;
    }
    mse /= data.len() as f64;

    let penalty = 0.001 * expr.size() as f64;
    mse + penalty
}

// --- Selection ---------------------------------------------------------------

fn tournament<'a>(pop: &'a [(Expr, f64)], k: usize, rng: &mut Lcg) -> &'a Expr {
    let mut best_idx = rng.range(pop.len());
    for _ in 1..k {
        let idx = rng.range(pop.len());
        if pop[idx].1 < pop[best_idx].1 {
            best_idx = idx;
        }
    }
    &pop[best_idx].0
}

// --- Main --------------------------------------------------------------------

fn main() {
    println!("=== Symbolic Regression ===\n");
    println!("target: f(x) = x^2 * sin(x)\n");

    let mut rng = Lcg::new(42);
    let data = generate_data(50, &mut rng);

    const POP_SIZE: usize = 200;
    const GENERATIONS: usize = 100;
    const TOURNAMENT_K: usize = 5;

    // Initialize population
    let mut population: Vec<(Expr, f64)> = (0..POP_SIZE)
        .map(|_| {
            let expr = random_expr(4, &mut rng);
            let fit = fitness(&expr, &data);
            (expr, fit)
        })
        .collect();

    let mut best_expr: Option<Expr> = None;
    let mut best_fitness = f64::INFINITY;

    for gen in 0..GENERATIONS {
        // Track best
        for (expr, fit) in &population {
            if *fit < best_fitness {
                best_fitness = *fit;
                best_expr = Some(expr.clone());
            }
        }

        if (gen + 1) % 10 == 0 || gen == 0 {
            let b = best_expr.as_ref().unwrap();
            println!(
                "gen {:>3}: best fitness = {:.6}  nodes = {:>3}  expr = {}",
                gen + 1,
                best_fitness,
                b.size(),
                b.format(),
            );
        }

        // Build next generation
        let mut next_pop = Vec::with_capacity(POP_SIZE);

        // Elitism: keep top 5
        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_by(|&a, &b| population[a].1.partial_cmp(&population[b].1).unwrap());
        for &i in indices.iter().take(5) {
            next_pop.push(population[i].clone());
        }

        // Fill rest via mutation
        while next_pop.len() < POP_SIZE {
            let parent = tournament(&population, TOURNAMENT_K, &mut rng);
            let child = match rng.range(10) {
                0..=3 => mutate_grow(parent, &mut rng),
                4..=6 => mutate_point(parent, &mut rng),
                7..=8 => mutate_simplify(parent),
                _ => random_expr(4, &mut rng), // fresh blood
            };

            // Skip overly large expressions
            if child.size() > 50 {
                continue;
            }

            let fit = fitness(&child, &data);
            next_pop.push((child, fit));
        }

        population = next_pop;
    }

    // Final results
    println!();
    let best = best_expr.unwrap();
    let mut g = ExprGraph::new();
    let root = best.to_expr(&mut g);
    let simplified = g.simplify(root);
    let expr_str = g.fmt_expr(simplified);
    println!("best expression: {}", expr_str);
    println!("best fitness:    {:.6}", best_fitness);

    // Verify on test points
    println!("\nverification:");
    let compiled = g.compile(simplified);
    for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
        let pred = compiled(&[x]);
        let exact = target(x);
        println!(
            "  f({:>5.1}) = {:>8.4}  (predicted: {:>8.4}, error: {:>8.4})",
            x, exact, pred, (pred - exact).abs()
        );
    }

    // Symbolic derivative via ExprGraph
    let dx = g.diff(simplified, 0);
    let dx = g.simplify(dx);
    println!("\nsymbolic derivative: d/dx [{}] = {}", expr_str, g.fmt_expr(dx));
}
