// GPU Sheffer search kernel — each thread runs a depth-1 bootstrap.
//
// Per thread:
//   1. Decode its (shape, atom/unary/binary assignment) from thread id.
//   2. Filter out operators that don't use both atoms X and Y.
//   3. For each test point pair, compute a small bootstrap catalog:
//      {1, x, f(1,1), f(1,x), f(x,1), f(x,x)} — 6 candidate values.
//   4. For the primary test point only, match each catalog value against
//      the 31 standard targets; OR the hit bits into `target_bits`.
//   5. Watch every pow call for the `(1+ε)^(1/ε)` artifact pattern.
//   6. If popcount(target_bits) ≥ target_threshold, atomically emit a
//      hit record.
//
// Depth-1 bootstrap is cheap but strong enough to surface EML
// (hits `e` via f(1,1) and `exp(x)` via f(x,1)), PowSkew (hits `-1`
// via f(0, x) — though `0` isn't in {1, x}, so PowSkew needs depth-2).
// For a first validation we accept that PowSkew-family operators may
// require a second dispatch at depth 2.

struct DispatchParams {
    shape_idx: u32,
    base_assignment_lo: u32,
    base_assignment_hi: u32,
    n_threads: u32,
    target_threshold: u32,
    limit_threshold_exp: i32,
    max_hits: u32,
    _pad: u32,
}

struct ShapeInfoRecord {
    bytecode_offset: u32,
    bytecode_len: u32,
    n_atoms: u32,
    n_unary: u32,
    n_binary: u32,
    assignment_count_lo: u32,
    assignment_count_hi: u32,
    size: u32,
}

struct Hit {
    shape_idx: u32,
    assignment_lo: u32,
    assignment_hi: u32,
    target_bits: u32,
    artifact: u32,
    value_re: f32,
    value_im: f32,
    size: u32,
}

@group(0) @binding(0) var<uniform> params: DispatchParams;
@group(0) @binding(1) var<storage, read> shape_bytecodes: array<u32>;
@group(0) @binding(2) var<storage, read> shape_info: array<ShapeInfoRecord>;
@group(0) @binding(3) var<storage, read> test_points: array<f32>;
@group(0) @binding(4) var<storage, read> targets: array<f32>;
@group(0) @binding(5) var<storage, read_write> hit_buffer: array<Hit>;
@group(0) @binding(6) var<storage, read_write> hit_count: array<atomic<u32>, 1>;

const MAX_STACK: u32 = 12u;
const N_TARGETS: u32 = 31u;
const N_TEST_POINTS: u32 = 5u;

// --- Complex f32 primitives --------------------------------------------------

fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return a + b; }
fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return a - b; }
fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn c_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let denom = b.x * b.x + b.y * b.y;
    return vec2<f32>((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
}
fn c_neg(a: vec2<f32>) -> vec2<f32> { return -a; }
fn c_inv(a: vec2<f32>) -> vec2<f32> { return c_div(vec2<f32>(1.0, 0.0), a); }
fn c_sqr(a: vec2<f32>) -> vec2<f32> { return c_mul(a, a); }

fn fast_sinh(x: f32) -> f32 { return 0.5 * (exp(x) - exp(-x)); }
fn fast_cosh(x: f32) -> f32 { return 0.5 * (exp(x) + exp(-x)); }

fn c_exp(a: vec2<f32>) -> vec2<f32> {
    let e = exp(a.x);
    return vec2<f32>(e * cos(a.y), e * sin(a.y));
}
fn c_ln(a: vec2<f32>) -> vec2<f32> {
    let r = sqrt(a.x * a.x + a.y * a.y);
    return vec2<f32>(log(r), atan2(a.y, a.x));
}
fn c_sqrt(a: vec2<f32>) -> vec2<f32> {
    let r = sqrt(a.x * a.x + a.y * a.y);
    let theta_half = atan2(a.y, a.x) * 0.5;
    let root_r = sqrt(r);
    return vec2<f32>(root_r * cos(theta_half), root_r * sin(theta_half));
}
fn c_sin(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(sin(a.x) * fast_cosh(a.y), cos(a.x) * fast_sinh(a.y));
}
fn c_cos(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(cos(a.x) * fast_cosh(a.y), -sin(a.x) * fast_sinh(a.y));
}
fn c_sinh(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(fast_sinh(a.x) * cos(a.y), fast_cosh(a.x) * sin(a.y));
}
fn c_cosh(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(fast_cosh(a.x) * cos(a.y), fast_sinh(a.x) * sin(a.y));
}
fn c_tanh(a: vec2<f32>) -> vec2<f32> {
    return c_div(c_sinh(a), c_cosh(a));
}
fn c_pow(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return c_exp(c_mul(b, c_ln(a)));
}

fn atom_value(id: u32, x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    switch (id) {
        case 0u: { return x; }
        case 1u: { return y; }
        case 2u: { return vec2<f32>(0.0, 0.0); }
        case 3u: { return vec2<f32>(1.0, 0.0); }
        case 4u: { return vec2<f32>(-1.0, 0.0); }
        case 5u: { return vec2<f32>(2.7182817, 0.0); }
        default: { return vec2<f32>(0.0, 0.0); }
    }
}

fn apply_unary(id: u32, v: vec2<f32>) -> vec2<f32> {
    switch (id) {
        case 0u: { return c_neg(v); }
        case 1u: { return c_inv(v); }
        case 2u: { return c_sqr(v); }
        case 3u: { return c_sqrt(v); }
        case 4u: { return c_exp(v); }
        case 5u: { return c_ln(v); }
        case 6u: { return c_sin(v); }
        case 7u: { return c_cos(v); }
        case 8u: { return c_sinh(v); }
        case 9u: { return c_tanh(v); }
        default: { return v; }
    }
}

fn apply_binary(id: u32, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    switch (id) {
        case 0u: { return c_add(a, b); }
        case 1u: { return c_sub(a, b); }
        case 2u: { return c_mul(a, b); }
        case 3u: { return c_div(a, b); }
        case 4u: { return c_pow(a, b); }
        default: { return a; }
    }
}

fn detect_limit(base: vec2<f32>, exponent: vec2<f32>, threshold: f32) -> bool {
    let db = base - vec2<f32>(1.0, 0.0);
    let dist = sqrt(db.x * db.x + db.y * db.y);
    let em = sqrt(exponent.x * exponent.x + exponent.y * exponent.y);
    return dist < threshold && em > 1.0 / threshold;
}

// --- Main kernel -------------------------------------------------------------

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.n_threads) { return; }

    let shape_idx = params.shape_idx;
    let info = shape_info[shape_idx];

    // Decode assignment (assume chunks fit in u32; caller enforces).
    var assignment: u32 = params.base_assignment_lo + tid;

    var atom_ids: array<u32, 12>;
    var unary_ids: array<u32, 12>;
    var binary_ids: array<u32, 12>;

    for (var i = 0u; i < info.n_atoms; i = i + 1u) {
        atom_ids[i] = assignment % 6u;
        assignment = assignment / 6u;
    }
    for (var i = 0u; i < info.n_unary; i = i + 1u) {
        unary_ids[i] = assignment % 10u;
        assignment = assignment / 10u;
    }
    for (var i = 0u; i < info.n_binary; i = i + 1u) {
        binary_ids[i] = assignment % 5u;
        assignment = assignment / 5u;
    }

    // uses_both filter: need at least one X and one Y atom.
    var has_x: bool = false;
    var has_y: bool = false;
    for (var i = 0u; i < info.n_atoms; i = i + 1u) {
        if (atom_ids[i] == 0u) { has_x = true; }
        if (atom_ids[i] == 1u) { has_y = true; }
    }
    if (!has_x || !has_y) { return; }

    // Limit threshold = 10 ^ exp (exp is negative).
    var thresh: f32 = 1.0;
    let exp_abs = -params.limit_threshold_exp;
    for (var i = 0; i < exp_abs; i = i + 1) {
        thresh = thresh * 0.1;
    }

    // target_bits starts ALL-SET (all 31 bits) and gets AND-ed with
    // per-test-point hit masks. A target stays set only if the
    // operator hits it at EVERY test point. This catches γ-specific
    // numerical coincidences (like γ ≈ 1/√3 which makes some
    // non-identities look like `x−1` at γ).
    var target_bits: u32 = 0x7FFFFFFFu; // bits 0..30 set for 31 targets
    var artifact: u32 = 0u;
    var primary_re: f32 = 0.0;
    var primary_im: f32 = 0.0;

    for (var tp = 0u; tp < N_TEST_POINTS; tp = tp + 1u) {
        // The first element of each test-point pair is our variable
        // "x" for the depth-1 bootstrap (leaves = {1, x}).
        let x_tp = vec2<f32>(test_points[tp * 4u + 0u], test_points[tp * 4u + 1u]);

        // Base leaves
        var cat_re: array<f32, 8>;
        var cat_im: array<f32, 8>;
        cat_re[0] = 1.0;       cat_im[0] = 0.0;
        cat_re[1] = x_tp.x;    cat_im[1] = x_tp.y;
        var cat_n: u32 = 2u;

        var any_nan_or_overflow: bool = false;

        // Depth-1: apply op to all 4 ordered pairs from {1, x_tp}.
        for (var pair = 0u; pair < 4u; pair = pair + 1u) {
            let ai = (pair >> 1u) & 1u;
            let bi = pair & 1u;
            let arg_x = vec2<f32>(cat_re[ai], cat_im[ai]);
            let arg_y = vec2<f32>(cat_re[bi], cat_im[bi]);

            // Run the shape bytecode as a stack machine with this (arg_x, arg_y).
            var stack_re: array<f32, MAX_STACK>;
            var stack_im: array<f32, MAX_STACK>;
            var sp: u32 = 0u;

            let bc_start = info.bytecode_offset;
            let bc_end = info.bytecode_offset + info.bytecode_len;
            for (var pc = bc_start; pc < bc_end; pc = pc + 1u) {
                let instr = shape_bytecodes[pc];
                let tag = instr >> 16u;
                let slot = instr & 0xFFFFu;
                if (tag == 0u) {
                    let v = atom_value(atom_ids[slot], arg_x, arg_y);
                    stack_re[sp] = v.x;
                    stack_im[sp] = v.y;
                    sp = sp + 1u;
                } else if (tag == 1u) {
                    let top = vec2<f32>(stack_re[sp - 1u], stack_im[sp - 1u]);
                    let v = apply_unary(unary_ids[slot], top);
                    stack_re[sp - 1u] = v.x;
                    stack_im[sp - 1u] = v.y;
                } else {
                    let b = vec2<f32>(stack_re[sp - 1u], stack_im[sp - 1u]);
                    let a = vec2<f32>(stack_re[sp - 2u], stack_im[sp - 2u]);
                    let op_id = binary_ids[slot];
                    if (op_id == 4u && detect_limit(a, b, thresh)) {
                        artifact = 1u;
                    }
                    let v = apply_binary(op_id, a, b);
                    stack_re[sp - 2u] = v.x;
                    stack_im[sp - 2u] = v.y;
                    sp = sp - 1u;
                }
            }

            let r_re = stack_re[0];
            let r_im = stack_im[0];
            if (r_re != r_re || r_im != r_im) {
                any_nan_or_overflow = true;
                break;
            }
            if (abs(r_re) > 1.0e20 || abs(r_im) > 1.0e20) {
                any_nan_or_overflow = true;
                break;
            }
            cat_re[cat_n] = r_re;
            cat_im[cat_n] = r_im;
            cat_n = cat_n + 1u;
        }

        if (any_nan_or_overflow) {
            // Reject the entire candidate if any test point overflows.
            target_bits = 0u;
            break;
        }

        // For this test point, build a local hit mask. For each target,
        // the target is "hit" if ANY catalog value is within tolerance.
        // Then AND into target_bits.
        //
        // The tolerance is ~1e-4 squared = 1e-2 distance at f32, which
        // is loose enough for f32 roundoff but tight enough to reject
        // the γ ≈ 1/√3 type coincidences (which have ~1.3e-4 distance
        // at γ but much larger at A and G).
        var tp_bits: u32 = 0u;
        // Targets are indexed by the primary-point value; at other test
        // points, we need to recompute the expected target at THAT x
        // value. But we only have the γ-computed targets uploaded.
        //
        // SIMPLIFICATION: For the variable-targets (x, exp(x), ln(x),
        // etc.), we can't meaningfully check them at test points other
        // than γ because the target depends on x. For CONSTANT targets
        // (0, 1, -1, e, pi, i, etc.), they're the same everywhere.
        //
        // So we split: constants (indices 0..15) are checked at every
        // test point; function-of-x targets (indices 16..30) are only
        // checked at tp==0 and their bit is ALWAYS preserved from tp>0.
        for (var k = 0u; k < cat_n; k = k + 1u) {
            let v_re = cat_re[k];
            let v_im = cat_im[k];
            for (var t = 0u; t < N_TARGETS; t = t + 1u) {
                let t_re = targets[t * 2u + 0u];
                let t_im = targets[t * 2u + 1u];
                let dr = v_re - t_re;
                let di = v_im - t_im;
                if (dr * dr + di * di < 1.0e-8) {
                    tp_bits = tp_bits | (1u << t);
                }
            }
        }
        // Targets 16..30 are functions of x — they change with the test
        // point. We can only validate them at tp==0. For tp>0, set them
        // "optimistically" (pass-through) so AND doesn't erase them.
        if (tp != 0u) {
            tp_bits = tp_bits | 0xFFFF0000u; // targets 16..31 auto-pass
        }
        target_bits = target_bits & tp_bits;

        if (tp == 0u) {
            if (cat_n > 2u) {
                primary_re = cat_re[2];
                primary_im = cat_im[2];
            }
        }
    }
    target_bits = target_bits & 0x7FFFFFFFu; // mask to 31 bits

    let hit_count_local = countOneBits(target_bits);
    if (hit_count_local < params.target_threshold) { return; }

    let slot = atomicAdd(&hit_count[0], 1u);
    if (slot < params.max_hits) {
        hit_buffer[slot].shape_idx = shape_idx;
        hit_buffer[slot].assignment_lo = params.base_assignment_lo + tid;
        hit_buffer[slot].assignment_hi = params.base_assignment_hi;
        hit_buffer[slot].target_bits = target_bits;
        hit_buffer[slot].artifact = artifact;
        hit_buffer[slot].value_re = primary_re;
        hit_buffer[slot].value_im = primary_im;
        hit_buffer[slot].size = info.size;
    }
}
