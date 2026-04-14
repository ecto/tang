//! GPU-accelerated operator search — minimal validation at MAX_SIZE=5.
//!
//! Runs the WGSL kernel over every (shape, assignment) pair up to
//! MAX_SIZE nodes, reads back hits, decodes each back to an `OpExpr`
//! on CPU, and runs the full f64 scoring + detector on the survivors.
//!
//! This is the validation step before scaling to MAX_SIZE=10. We
//! verify that the GPU pipeline finds expected-to-be-there operators
//! like EML (exp(x) - ln(y)), compares hit counts to the CPU
//! operator_search, and surfaces any surprises.
//!
//! Run with `cargo run --release --example gpu_search -p tang-sheffer --features gpu`.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use tang_sheffer::gpu_search::{GpuSearcher, TARGET_NAMES};
use tang_sheffer::op_enum::EnumOp;
use tang_sheffer::op_score::score_generic;
use tang_sheffer::shape_bytecode::{build_shape_table, reconstruct_opexpr};

const MAX_SIZE: usize = 10;
const TARGET_THRESHOLD: u32 = 5;
const MAX_HITS: u32 = 1 << 24; // 16M hit records
const CHUNK_THREADS: u32 = 1 << 22; // 4M threads per dispatch
const LIMIT_EXP: i32 = -5; // 1e-5 threshold at f32

fn main() {
    println!("=== GPU Sheffer search at MAX_SIZE={} ===", MAX_SIZE);
    let t_total = Instant::now();

    // ---- Build shape table ----
    let t0 = Instant::now();
    let table = build_shape_table(MAX_SIZE);
    let n_shapes = table.shape_info.len();
    let total_assignments: u64 = table
        .shape_info
        .iter()
        .map(|i| (i.assignment_count_hi as u64) << 32 | (i.assignment_count_lo as u64))
        .sum();
    println!(
        "  shapes: {} | total raw candidates: {} | table bytes: {}  [{:.2}s]",
        n_shapes,
        total_assignments,
        table.all_instrs.len() * 4,
        t0.elapsed().as_secs_f64()
    );

    // ---- Init GPU ----
    let t0 = Instant::now();
    let table_arc = Arc::new(table);
    let mut gpu = match GpuSearcher::new(table_arc.clone(), MAX_HITS) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to init GPU: {}", e);
            std::process::exit(1);
        }
    };
    println!("  GPU initialized  [{:.2}s]", t0.elapsed().as_secs_f64());

    // ---- Dispatch over every shape ----
    let t0 = Instant::now();
    gpu.reset_hit_count();
    for shape_idx in 0..n_shapes as u32 {
        gpu.dispatch_shape(shape_idx, TARGET_THRESHOLD, CHUNK_THREADS, LIMIT_EXP);
    }
    println!("  all dispatches submitted  [{:.2}s]", t0.elapsed().as_secs_f64());

    // ---- Read back hits ----
    let t0 = Instant::now();
    let hits = gpu.read_hits();
    println!(
        "  {} hits returned  [{:.2}s]",
        hits.len(),
        t0.elapsed().as_secs_f64()
    );

    // ---- Decode + CPU verify top-ranked hits ----
    //
    // Group hits by the number of target bits set. The highest
    // popcount candidates are the most interesting.
    let t0 = Instant::now();
    let mut by_popcount: Vec<(u32, usize)> = hits
        .iter()
        .enumerate()
        .map(|(i, h)| (h.target_bits.count_ones(), i))
        .collect();
    by_popcount.sort_by(|a, b| b.0.cmp(&a.0));

    println!();
    println!("--- Top 20 GPU hits by target popcount ---");
    println!(
        "  {:<4} {:<4} {:<6} {:<6} {}",
        "rank", "size", "popct", "artif", "expression"
    );
    println!("  {}", "-".repeat(78));
    let mut printed = 0;
    let mut seen_exprs = HashSet::new();
    for (popcount, idx) in by_popcount.iter().take(500) {
        if printed >= 20 {
            break;
        }
        let hit = &hits[*idx];
        let assignment = (hit.assignment_hi as u64) << 32 | (hit.assignment_lo as u64);
        let shape = &table_arc.shapes[hit.shape_idx as usize];
        let expr = reconstruct_opexpr(shape, assignment);
        let pretty = expr.pretty();
        if !seen_exprs.insert(pretty.clone()) {
            continue;
        }
        let artifact_mark = if hit.artifact != 0 { "ART" } else { "—" };
        let target_names: Vec<&str> = (0..31)
            .filter(|ti| hit.target_bits & (1 << ti) != 0)
            .map(|ti| TARGET_NAMES[ti])
            .collect();
        println!(
            "  {:<4} {:<4} {:<6} {:<6} {}  →  {{{}}}",
            printed + 1,
            hit.size,
            popcount,
            artifact_mark,
            truncate(&pretty, 40),
            target_names.join(", ")
        );
        printed += 1;
    }

    // ---- CPU re-verification of the top distinct hits ----
    //
    // For each top-20 unique operator found by the GPU, reconstruct it
    // on CPU, wrap in EnumOp, run score_generic with full multi-point
    // + detector, and report the REAL genuine coverage. GPU's depth-1
    // bootstrap is a coarse filter; CPU's full bootstrap is the ground
    // truth. Anything the GPU surfaced that scores well on CPU is a
    // real new candidate worth multi-point verifying.
    println!();
    println!("--- CPU re-verification of top GPU candidates ---");
    let baseline: HashSet<String> = HashSet::new();
    let mut re_seen: HashSet<String> = HashSet::new();
    let mut checked = 0;
    for (popcount, idx) in by_popcount.iter() {
        if checked >= 15 {
            break;
        }
        if *popcount < 4 {
            break;
        }
        let hit = &hits[*idx];
        let shape = &table_arc.shapes[hit.shape_idx as usize];
        let expr = reconstruct_opexpr(
            shape,
            (hit.assignment_hi as u64) << 32 | (hit.assignment_lo as u64),
        );
        let pretty = expr.pretty();
        if !re_seen.insert(pretty.clone()) {
            continue;
        }
        let op = EnumOp::new(expr.clone());
        let card = score_generic(&op, expr.size(), &baseline);
        let artifact_note = if hit.artifact != 0 {
            " [GPU flagged artifact]"
        } else {
            ""
        };
        println!(
            "  size {} | {{1,x}} {:>2} | {{x}} {:>2} | {}{}",
            expr.size(),
            card.with_const_coverage,
            card.const_free_coverage,
            truncate(&pretty, 45),
            artifact_note,
        );
        checked += 1;
    }

    println!();
    println!(
        "total wall clock: {:.2}s (decoded/printed in {:.2}s)",
        t_total.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64(),
    );
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}
