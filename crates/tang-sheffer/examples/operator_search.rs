//! Strategy A — programmatic operator enumeration and scoring.
//!
//! Enumerates every expression tree up to `MAX_SIZE` nodes, dedupes by
//! semantic value at five complex test-point pairs, scores every survivor
//! via the existing `Verifier::bootstrap` against the 31 standard targets,
//! ranks the results, and prints a table plus a unique-target honour
//! roll.
//!
//! Uses the `op_arena` flat-arena representation so MAX_SIZE=8 and 9 fit
//! in reasonable RAM. The arena caches sizes 1..=MAX_SIZE−1 and streams
//! the top level: candidates that pass dedup+filter get pushed to the
//! arena; the rest are discarded.
//!
//! Run: `cargo run --release --example operator_search -p tang-sheffer`

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use tang_sheffer::op_arena::{
    build_arena, eval_node_view, uses_both_view, ArenaDedupSet, ArenaOp, Node, NodeId, OpArena,
};
use tang_sheffer::op_enum::{BinaryOp, UnaryOp};
use tang_sheffer::op_score::{
    ranking_key, score_generic, standard_all_targets, Scorecard, SCORE_BUDGET, SCORE_ITERS,
};
use tang_sheffer::{Edl, Eml, Leaf, PowSkew, Verifier, C, TEST_POINT};

/// Maximum tree node count to enumerate. At the arena representation,
/// MAX_SIZE=9 uses ~12 GB RAM and runs in tens of minutes; MAX_SIZE=8
/// uses ~1 GB and finishes in a minute or two.
const MAX_SIZE: usize = 9;

/// Baseline runs use a stronger budget than candidate scoring so the
/// "unique reach" column reflects what the hand-curated operators can
/// genuinely do, not their budget-3 handicap.
const BASELINE_BUDGET: usize = 4;
const BASELINE_ITERS: usize = 5;

/// Print the top-K operators from the ranked list.
const TOP_K: usize = 30;

fn baseline_reached() -> HashSet<String> {
    let targets = standard_all_targets();
    let mut union: HashSet<String> = HashSet::new();

    let eml_leaves = vec![
        Leaf::constant("1", C::new(1.0, 0.0)),
        Leaf::variable("x", C::new(TEST_POINT, 0.0)),
    ];
    let mut v = Verifier::new(eml_leaves);
    for d in v.bootstrap(&Eml, &targets, BASELINE_BUDGET, BASELINE_ITERS) {
        union.insert(d.target_name);
    }

    let edl_leaves = vec![
        Leaf::constant("e", C::new(std::f64::consts::E, 0.0)),
        Leaf::variable("x", C::new(TEST_POINT, 0.0)),
    ];
    let mut v = Verifier::new(edl_leaves);
    for d in v.bootstrap(&Edl, &targets, BASELINE_BUDGET, BASELINE_ITERS) {
        union.insert(d.target_name);
    }

    let pow_leaves = vec![Leaf::variable("x", C::new(TEST_POINT, 0.0))];
    let mut v = Verifier::new(pow_leaves);
    for d in v.bootstrap(&PowSkew, &targets, BASELINE_BUDGET, BASELINE_ITERS) {
        union.insert(d.target_name);
    }

    union
}

fn main() {
    println!("=== Strategy A: programmatic operator enumeration (arena) ===");
    println!(
        "max_size = {}, scoring budget = {} ops × {} iters\n",
        MAX_SIZE, SCORE_BUDGET, SCORE_ITERS
    );

    // ---- Enumerate + dedup via arena ----
    //
    // Build the cache up to MAX_SIZE − 1, then stream the top level
    // (size MAX_SIZE). Candidates that pass dedup + uses_both get
    // pushed onto the arena to get a stable NodeId for scoring; the
    // rest are discarded.
    let t0 = Instant::now();
    let cache_up_to = MAX_SIZE.saturating_sub(1).max(1);
    let (mut arena, by_size) = build_arena(cache_up_to);
    println!(
        "  cache built: sizes 1..={} → {} arena nodes  [{:.2}s]",
        cache_up_to,
        arena.len(),
        t0.elapsed().as_secs_f64(),
    );

    let t1 = Instant::now();
    let mut raw_count: usize = 0;
    let mut dedup = ArenaDedupSet::new();
    let mut candidates: Vec<NodeId> = Vec::new();
    let mut by_size = by_size;

    // Process cached sizes first (they're already in the arena).
    for n in 1..=cache_up_to {
        let len = by_size[n].len();
        for i in 0..len {
            let id = by_size[n][i];
            raw_count += 1;
            if arena.uses_both(id) && dedup.insert_id(&arena, id) {
                candidates.push(id);
            }
        }
    }

    // Stream the top level (size MAX_SIZE). Only surviving candidates
    // are pushed to the arena; the rest are dropped as the Node goes
    // out of scope. We iterate by index to avoid clones of the big
    // index vectors (`by_size[8]` at MAX_SIZE=9 is ~3.5 GB).
    if MAX_SIZE > cache_up_to {
        let n = MAX_SIZE;

        // Unary wraps of size n-1. Iterate by index to avoid borrowing
        // `by_size[n-1]` while we mutate `arena`.
        let unary_len = by_size[n - 1].len();
        for i in 0..unary_len {
            let parent = by_size[n - 1][i];
            for op in UnaryOp::ALL {
                raw_count += 1;
                let node = Node::Unary(op, parent);
                if uses_both_view(&arena, node) && dedup.insert_node(&arena, node) {
                    let new_id = arena.push(node);
                    candidates.push(new_id);
                }
            }
        }

        // Size (n-1) is not referenced by any later level at MAX_SIZE=n,
        // so we can drop its NodeId index now to reclaim RAM. At
        // MAX_SIZE=9 this frees ~3.5 GB (872M × 4 bytes for the
        // size-8 index).
        by_size[n - 1] = Vec::new();

        // Binary splits (k, n-1-k). None of them reference size n-1
        // (because k ≤ n-2 and rk = n-1-k ≥ 1), so the drop above is
        // safe.
        if n >= 3 {
            for k in 1..=(n - 2) {
                let rk = n - 1 - k;
                let l_len = by_size[k].len();
                let r_len = by_size[rk].len();
                for i in 0..l_len {
                    let l = by_size[k][i];
                    for j in 0..r_len {
                        let r = by_size[rk][j];
                        for op in BinaryOp::ALL {
                            raw_count += 1;
                            let node = Node::Binary(op, l, r);
                            if uses_both_view(&arena, node) && dedup.insert_node(&arena, node) {
                                let new_id = arena.push(node);
                                candidates.push(new_id);
                            }
                        }
                    }
                }
            }
        }

        // All of `by_size` is now unused; drop it to free the remaining
        // index vectors before scoring.
        drop(by_size);
    }

    println!(
        "  enumerate ≤{} nodes: {} raw → {} unique (after uses_both + dedup)",
        MAX_SIZE,
        raw_count,
        candidates.len(),
    );
    println!(
        "  arena grew to {} nodes ({:.1} MB)  [{:.2}s]",
        arena.len(),
        (arena.len() as f64 * 16.0) / 1_048_576.0,
        t1.elapsed().as_secs_f64(),
    );

    // Make arena immutable and shareable for parallel scoring.
    let arena: Arc<OpArena> = Arc::new(arena);

    // ---- Build baseline ----
    let t0 = Instant::now();
    let baseline = baseline_reached();
    println!(
        "baseline (EML ∪ EDL ∪ PowSkew @ budget {}×{}): {} targets  [{:.2}s]\n",
        BASELINE_BUDGET,
        BASELINE_ITERS,
        baseline.len(),
        t0.elapsed().as_secs_f64(),
    );

    // ---- Score all candidates in parallel ----
    //
    // Memory-conscious filter: most candidates are broken (NaN/inf at the
    // test point or reach 0 targets), and storing a full Scorecard for
    // all 10M+ of them eats tens of GB. Drop anything with coverage <
    // `KEEP_THRESHOLD` immediately, keep only the interesting ones.
    const KEEP_THRESHOLD: usize = 5;

    let t0 = Instant::now();
    let n = candidates.len();
    let scorecards: Vec<Scorecard> = candidates
        .par_iter()
        .filter_map(|&id| {
            let op = ArenaOp::new(arena.clone(), id);
            let size = op.size();
            let card = score_generic(&op, size, &baseline);
            if card.with_const_coverage >= KEEP_THRESHOLD
                || card.const_free_coverage >= KEEP_THRESHOLD
                || !card.unique_targets.is_empty()
            {
                Some(card)
            } else {
                None
            }
        })
        .collect();
    let elapsed = t0.elapsed().as_secs_f64();
    println!(
        "scored {} candidates in {:.1}s ({:.2} ms/candidate avg); kept {} interesting",
        n,
        elapsed,
        1000.0 * elapsed / n as f64,
        scorecards.len(),
    );

    // ---- Rank ----
    let mut ranked = scorecards.clone();
    ranked.sort_by_key(ranking_key);

    println!();
    println!("--- Top {} operators by (coverage + 2·const-free, size, growth) ---", TOP_K);
    println!(
        "  {:<4} {:<38} {:>4} {:>6} {:>6} {:>8} {:>10}",
        "rank", "operator", "size", "{1,x}", "{x}", "ln(x)", "growth"
    );
    println!("  {}", "-".repeat(78));
    for (i, s) in ranked.iter().take(TOP_K).enumerate() {
        let ln_depth = s
            .ln_x_expanded_size
            .map(|d| format!("{}", d))
            .unwrap_or_else(|| "—".to_string());
        println!(
            "  {:<4} {:<38} {:>4} {:>6} {:>6} {:>8} {:>10}",
            i + 1,
            truncate(&s.op_name, 38),
            s.op_size,
            s.with_const_coverage,
            s.const_free_coverage,
            ln_depth,
            format!("{:?}", s.growth),
        );
    }

    // ---- Unique-target honour roll ----
    let mut uniques: Vec<&Scorecard> = ranked.iter().filter(|s| !s.unique_targets.is_empty()).collect();
    uniques.sort_by_key(|s| -(s.unique_targets.len() as i64));
    println!();
    println!("--- Unique-target honour roll (reaches what EML∪EDL∪PowSkew cannot) ---");
    if uniques.is_empty() {
        println!("  (none — every candidate's reach is a subset of the hand-curated baseline)");
    } else {
        for s in uniques.iter().take(TOP_K) {
            let mut targets = s.unique_targets.clone();
            targets.sort();
            println!(
                "  {:<38} size {:>2}  +{}  {{{}}}",
                truncate(&s.op_name, 38),
                s.op_size,
                targets.len(),
                targets.join(", "),
            );
        }
    }

    // ---- Constant-free focus ----
    let mut const_free: Vec<&Scorecard> = ranked
        .iter()
        .filter(|s| s.const_free_coverage >= 5)
        .collect();
    const_free.sort_by_key(|s| (-(s.const_free_coverage as i64), s.op_size as i64));
    println!();
    println!("--- Constant-free leaderboard ({{x}} pool, coverage ≥ 5) ---");
    if const_free.is_empty() {
        println!("  (none reached 5+ targets from {{x}} alone)");
    } else {
        for s in const_free.iter().take(TOP_K) {
            println!(
                "  {:<38} size {:>2}  const-free {:>2} / {{1,x}} {:>2}  growth {:?}",
                truncate(&s.op_name, 38),
                s.op_size,
                s.const_free_coverage,
                s.with_const_coverage,
                s.growth,
            );
        }
    }

    // ---- Polynomial-growth focus ----
    let mut poly: Vec<&Scorecard> = ranked
        .iter()
        .filter(|s| {
            matches!(
                s.growth,
                tang_sheffer::GrowthClass::Polynomial | tang_sheffer::GrowthClass::Bounded
            )
        })
        .filter(|s| s.with_const_coverage >= 5)
        .collect();
    poly.sort_by_key(|s| (-(s.with_const_coverage as i64), s.op_size as i64));
    println!();
    println!("--- Polynomial-or-bounded growth candidates (coverage ≥ 5) ---");
    if poly.is_empty() {
        println!("  (none — all competitive candidates blow up)");
    } else {
        for s in poly.iter().take(TOP_K) {
            println!(
                "  {:<38} size {:>2}  {{1,x}} {:>2}  {{x}} {:>2}  growth {:?}",
                truncate(&s.op_name, 38),
                s.op_size,
                s.with_const_coverage,
                s.const_free_coverage,
                s.growth,
            );
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}
