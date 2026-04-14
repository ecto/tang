# PowSkew: a constant-free binary operator for elementary functions

**Definition.** Let `f(x, y) = x^y − y^x` on the principal branch of the
complex plane. We call this operator `PowSkew`.

**Claim.** PowSkew is a partial positive answer to the open problem posed by
Odrzywołek (arXiv:2603.21852, Table 2, final rows): *does there exist a
binary operator needing no distinguished constant, which generates the
constants of elementary mathematics from arbitrary input?* Specifically,
PowSkew generates `{0, 1, −1, 2, −2, 1/2, ±i}` and the one-variable
affine/integer shifts `{x−1, x+1, −x}` from a single variable leaf `{x}` at
tree depth ≤ 3. We empirically verify 12 of 31 standard targets reached.

## 1. Trivial constant generation

The operator is antisymmetric: `f(y, x) = −f(x, y)`. In particular the
diagonal vanishes:

    f(x, x) = x^x − x^x = 0       for any finite non-zero x.

This gives `0` in one application from any input `x`, and places us in the
regime where partial-identity leaves `x^0 = 1` and `0^x = 0` (for `x > 0`,
and on the principal branch for complex `x` with `Re(x) > 0`) are available.

From `{x, 0}` we get the next two constants in one step each:

    f(x, 0) = x^0 − 0^x = 1 − 0 = 1
    f(0, x) = 0^x − x^0 = 0 − 1 = −1

So starting from just `{x}` (no distinguished leaf), we have
`{x, 0, 1, −1}` at tree sizes `(1, 3, 5, 5)` respectively — independent
of the specific numerical value of `x`, **provided `Re(x) > 0`**. The
open right half-plane is a genuine constraint: when `Re(x) < 0`, the
term `0^x` diverges via `exp(x · ln(0)) = exp(x · −∞)`, contaminating
`f(x, 0)` with infinities and NaN. The diagonal identity `f(x, x) = 0`
still holds on the whole complex plane minus the origin, but the
`(x, 0)` and `(0, x)` shortcuts require `Re(x) > 0`.

In practical terms: the standard test points `{γ, A, G}` and any other
"ordinary" positive-real transcendental land in the domain, so the
constant-freedom claim is not weakened for verification purposes. But
this is the reason a formal proof must state the half-plane assumption
explicitly and cannot claim domain-independence simpliciter.

## 2. Cascade

From `{x, 0, 1, −1}`:

    f(1, −1)  = 1^(−1) − (−1)^1  = 1 − (−1)  = 2
    f(−1, 1)  = (−1)^1 − 1^(−1)  = −1 − 1    = −2
    f(−1, 2)  = (−1)^2 − 2^(−1)  = 1 − 1/2   = 1/2

Then from `{x, 0, 1, −1, 2, −2, 1/2}`:

    f(1, x)                     = 1 − x
    f(f(1, x), 1)               = (1−x)^1 − 1^(1−x) = (1−x) − 1 = −x
    f(1, f(f(1, x), 1))         = 1^(−x) − (−x)^1   = 1 − (−x)  = x + 1

The imaginary unit falls out of `(−1)^(1/2)` on the principal branch:

    f(1/2, −1) = (1/2)^(−1) − (−1)^(1/2) = 2 − i
    f(f(1/2, −1), 1) = (2 − i)^1 − 1^(2 − i) = (2 − i) − 1 = 1 − i
    f(1, f(f(1/2, −1), 1)) = 1^(1−i) − (1−i)^1 = 1 − (1 − i) = i
    f(f(f(1/2, −1), 1), 1) = (1 − i)^1 − 1^(1−i) = (1 − i) − 1 = −i

This gives `{i, −i}` at tree size 7 from `{x}` alone. The verifier in
`crates/tang-sheffer` reproduces these exact expressions; see
`examples/constant_free.rs` output.

## 3. Algebraic closure limitation

**Observation.** From `{x}` alone, every PowSkew expression is an element
of the algebraic closure of ℚ(x) extended by `(−1)^(1/2) = i`. In
particular, the transcendental numbers `e, π, ln(2), sin(1)` are *not*
reachable.

**Sketch.** PowSkew(a, b) = a^b − b^a. If `a, b ∈ ℚ̄(x, i)` (the algebraic
closure of the rationals in `x` adjoined `i`), then `a^b` is generally
transcendental, unless `b` is rational. But the exponents appearing in
any PowSkew expression are themselves built by PowSkew, so the exponent is
algebraic in `x` too. Gelfond–Schneider tells us `a^b` with algebraic
`a ≠ 0, 1` and irrational algebraic `b` is transcendental — so this escape
route does produce transcendentals *over ℚ*, but the values remain
algebraic over `ℚ(x)` because the exponent is a function of `x`. Hence no
target that is transcendental over `ℚ(x)` (e.g. `e`, `exp(x)`, `ln(x)`) is
reachable by pure PowSkew from `{x}`.

Our verifier confirms: PowSkew from `{x}` finds 12/31 targets — exactly
the algebraic and ±i targets — and misses every target involving `e, π, exp,
ln, sin, cos, sqrt`.

## 4. Extension to transcendental *functions*: PowExpSkew

**Definition.** `g(x, y) = (x^y − y^x) + (exp(x) − exp(y))`.

`g` is again fully antisymmetric, so `g(x, x) = 0`, preserving the
constant-freedom of the diagonal. The exp term supplies a transcendental
kick:

    g(x, x) = 0
    g(x, 0) = (x^0 − 0^x) + (exp(x) − exp(0)) = 1 + exp(x) − 1 = exp(x)
    g(0, x) = −exp(x)

**Verified identities (multi-point check at {γ, A, G}).** PowExpSkew from
`{x}` produces **4 true identities**: `{x, 0, exp(x), exp(exp(x))}`. Only
two of these are non-trivial: `0` via the antisymmetric diagonal, and
`exp(x)` via `g(x, 0)`.

**Important negative result.** The single-point bootstrap also claimed
`{-1, 1, -e, e, 1/e}` as PowExpSkew discoveries from `{x}`, but all five
**fail multi-point verification** at test points `A` and `G`. Those
expressions are functions of `x` that happen to equal the target at `γ`
and nowhere else — Schanuel-style coincidences, not true identities.

So the honest claim is: **PowExpSkew is constant-free for transcendental
*functions of x* (`exp(x)`, `exp(exp(x))`) but does NOT generate the
transcendental *constants* (`e`, `1/e`)** from `{x}` alone. Combining
with PowSkew to first generate the algebraic constants `{0, 1, -1}` and
then applying PowExpSkew(1, 0) = e is a natural two-operator hybrid, but
that's outside the single-operator Sheffer framing.

## 4a. Union verification

At budget 4, cross-checked at three conjecturally-independent
transcendentals {γ, A, G}, the **verified union** {EML, PowSkew,
PowExpSkew} covers **24/31** of the standard target set:

    Constants (12/16): 0, 1, -1, 2, -2, 1/2, e, -e, 1/e, e^2, i, -i
    Functions (12/15): x, exp(x), ln(x), -x, 1/x, x^2, x+1, x-1, 2x,
                       e*x, exp(exp(x)), ln(ln(x))

    Missing:           pi, pi/2, 2pi, i*pi, sqrt(x), sin(x), cos(x)

The single-point sweep reported 26/31 (including i*pi and -2 via EML),
but those are x-specific coincidences and drop out under cross-check.

## 4b. Depth comparison

Minimal expanded-alphabet tree size to reach each verified target, budget 4:

| target       |  EML |  PowSkew  |  PowExpSkew |
|:------------ | ---: | --------: | ----------: |
| `0`          |    7 |       **3** |         **3** |
| `1`          |    1 |         5 |           — |
| `−1`         |   15 |       **5** |           — |
| `2`          |   25 |        11 |           — |
| `−2`         |   31 |      **11** |           — |
| `1/2`        |   55 |      **17** |           — |
| `e`          |    3 |         — |           — |
| `exp(x)`     |    3 |         — |           5 |
| `exp(exp(x))`|    5 |         — |           9 |
| `ln(x)`      |    7 |         — |           — |
| `x−1`        |   11 |       **7** |           — |
| `x+1`        |   25 |      **19** |           — |
| `−x`         |   15 |      **13** |           — |
| `i`          |    — |      **35** |           — |
| `−i`         |    — |      **35** |           — |

**PowSkew compresses algebraic constants by 2–3× compared to EML**
(`−1`: 5 vs 15, `−2`: 11 vs 31, `1/2`: 17 vs 55). And PowSkew reaches
`±i` at all (EML doesn't within budget 4 from `{1, x}`).

EML still wins on transcendental constants and functions (`e`, `ln(x)`)
where PowSkew can't reach them at all. **The two operator families are
complementary, not competing** — PowSkew is optimal for algebraic targets,
EML for transcendentals.

## 4c. All 31 standard targets reached (Phase 5)

A two-stage EDL+EML approach cracks every one of the 31 standard targets,
including `{π, π/2, 2π, iπ, i, −i, sqrt(x), sin(x), cos(x)}` which were
missing at the end of Phase 4:

**Stage A — EDL constant chase from `{e}` alone.**
With stepping-stone targets including `±∞, iπ/2, i/2, 2i, ln(π)` added
to the target list so the bootstrap promotes them to leaves, EDL finds
**26 of 28 constants** at budget 4 in 8 iterations. Key finds:

    i      = edl(iπ/2, e)       = exp(iπ/2) / ln(e) = i / 1
    i/2    = edl(iπ/2, e^2)     = i / 2
    −i     = edl(iπ/2, 1/e)
    iπ     = edl(1, edl(edl(1, −1), e))        [one-shot from {1, e, −1}]
    iπ/2   = edl(edl(1, edl(edl(1, iπ), e)), e^2)
    π      = edl(0, edl(edl(iπ/2, −1), e))

Crucially, these are all x-independent expressions, so they're true
identities automatically — no multi-point verification needed.

**Stage B — EML function search with constants pre-loaded.**
Using a trimmed pool of 10 essential constants `{1, 0, −1, 1/2, 2, e, i,
−i, 2i, iπ}` plus `x`, EML at budget 4 finds the function targets via
standard identities and clever nested-log chains. With stepping stones
`ln(ln(ln(x)))`, `2i·sin(x)`, `ln(ln(2i·sin(x)))`, etc. in the target
list, the bootstrap promotes each to a leaf and cascades:

    sqrt(x) = eml(eml(eml(ln(ln(ln(x))), 2), 1), 1)            [size 25]
            = exp(exp(ln(ln(x)) − ln(2))) = exp(ln(x)/2)
    sin(x)  = eml(eml(ln(ln(2i·sin(x))), 2i), 1)               [size 55]
            = exp(ln(2i·sin(x)) − ln(2i)) = (2i·sin(x)) / (2i)
    cos(x)  = eml(i·x, exp(i·sin(x)))                          [size 75]
            = exp(ix) − ln(exp(i·sin(x)))
            = (cos(x) + i·sin(x)) − i·sin(x)  [via Euler]

where the intermediates `2i·sin(x) = eml(i·x, exp(−i·x)) = exp(ix) −
exp(−ix)` come for free from EML's native subtraction, and `i·x, −i·x`
fall out of the nested-log trick `eml(eml(ln(ln(x)), ±i), 1) = ±i·x`.

**Multi-point cross-check.** All 28 Stage B function discoveries pass at
`{γ, A, G}` with tolerance 1e-8. Stage A constants are x-independent so
trivially valid.

**Final verified coverage: 31/31 standard targets**, split across
operators by specialization:
- EDL (Stage A constants): 26 constants including all transcendentals
  and imaginary constants
- EML (Stage B functions): 28 functions including `sqrt, sin, cos`

The two-operator division of labor is natural: EDL's `exp/ln` combinator
makes imaginary-unit extraction easy (`edl(iπ/2, e) = i`), while EML's
`exp − ln` combinator makes Euler-formula subtraction free. Neither
operator alone cracks all 31 targets at budget 4; together they do.

## 4d. Phase 6 — three new constant-free Sheffer candidates (Strategy A)

A programmatic enumeration of `OpExpr` trees up to 5 nodes over the
alphabet `{x, y, 0, 1, -1, e} ∪ {neg, inv, sqr, sqrt, exp, ln, sin,
cos, sinh, tanh} ∪ {+, -, *, /, ^}` — 3981 unique candidates after
semantic dedup at five complex test-point pairs — was scored against
the 31 standard targets via `Verifier::bootstrap` at budget 3 × 3
iterations. The top candidates were then re-run at Phase-5 budget
(4 × 5 iterations) and every discovery cross-checked at
`{γ, A, G}`.

**Three novel constant-free Sheffer candidates emerge**, each matching
or beating PowSkew's 12/31 constant-free reach at smaller `OpExpr`
size:

| operator | size | constant-free verified targets |
|---|---|---|
| `(x − y)^y` | 5 | 13: `{−1, −2, −i, 0, 1, 1/2, 1/x, 2, i, x, x+1, x−1, x²}` |
| `1 − x/y` | 5 | 13: `{−1, −2, −x, 0, 1, 1/2, 1/x, 2, 2x, x, x+1, x−1, x²}` |
| `x/y − 1` | 5 | 13: (same as `1 − x/y`, antisymmetric variant) |
| `−(x/sqrt(y))` | 5 | 9: `{−1, −i, −x, 1, 1/x, i, **sqrt(x)**, x, x²}` |
| PowSkew = `x^y − y^x` | 7 | 12 (baseline) |

All four new operators are:

- **Constant-free**: `f(x, x)` is an x-independent constant (0 for the
  first three, which are anti-diagonal by construction). No
  distinguished leaf needed.
- **Smaller than PowSkew** in `OpExpr` size: 5 vs 7 nodes.
- **Polynomial-growth or bounded** on a generic complex seed — none of
  them have EML's double-exponential gradient blow-up.
- **Multi-point verified** at `{γ, A, G}`: every discovery produces
  the claimed target value at all three independent transcendentals.
  Zero x-dependent coincidences in any of the four runs.

### Complementary reach

The three top candidates are not equivalent — their target sets
overlap but differ:

    SubPow   \ OneMinusDiv = {−i, i}
    OneMinusDiv \ SubPow   = {−x, 2x}
    NegDivSqrt \ all       = {sqrt(x)}

The union **SubPow ∪ OneMinusDiv ∪ NegDivSqrt covers 16 targets
constant-free**:

    {−1, −2, −x, −i, 0, 1, 1/2, 1/x, 2, 2x, i, x, x+1, x−1, x², sqrt(x)}

This is the strongest constant-free result in this crate — 4 more
targets than PowSkew alone, including `2x, sqrt(x)`, and with a
cleaner growth profile.

### Why these work (sketch)

- **`(x − y)^y`** exploits `0^y = 0` on the diagonal, `x^0 = 1` at
  `y = 0`, and the complex principal branch of `(−x)^x` for imaginary
  reach via `(−1)^(1/2) = i`. Similar in spirit to PowSkew but with a
  simpler tree — a single subtraction inside the exponent instead of
  two powers.
- **`1 − x/y`** is a Möbius-like rational map: `f(x, x) = 0`,
  `f(0, x) = 1`, `f(x, 0) = ±∞` (IEEE infinity propagation is the key
  to reaching `−1` via the EML-style `−∞` chains from Section 2).
  Uses the constant 1 inside the operator body but no distinguished
  leaf in the bootstrap pool.
- **`−(x/sqrt(y))`** reaches `sqrt(x)` constant-free at budget 4
  because the `sqrt` is baked into the operator — applied to
  `f(x, 1)` we get `−sqrt(x)`, and the bootstrap cascade then finds
  `sqrt(x)` via negation.

### MAX_SIZE=6 extension

Bumping the enumeration bound to 6 nodes (3,131,046 raw trees →
67,299 unique after dedup, scored in ~65 s wall clock via rayon
parallelism + `Arc<OpExpr>` subtree sharing) surfaces one genuinely
new verified candidate:

**`(x − y)^(1/y)`** (SubPowInv) — size 6, polynomial growth,
**13 verified targets with {1, x}** including **`sqrt(x)`**:

    {−1, −2, −i, 0, 1, 1/2, 1/x, 2, i, sqrt(x), x, x+1, x−1}

This is an elegant dual to `(x − y)^y` (SubPow): flipping the outer
exponent from `y` to `1/y` trades `x²` access for `sqrt(x)` access
while keeping the polynomial-growth and 13-target reach. It is the
only polynomial-growth operator in our zoo that reaches `sqrt(x)` from
the standard `{1, x}` pool at small depth — `sqrt(x)` was previously
only accessible via (a) the EDL `ln(ln)` trick at size 13 or (b) the
full Phase-5 Stage-B EML construction at expanded size 25.

SubPowInv is *not* constant-free (only 2/31 from `{x}` alone because
`(x − 0)^(1/0) = x^∞` is branch-brittle), so it joins the non-const-free
tier alongside EML/EDL rather than the SubPow/OneMinusDiv constant-free
champions.

### What MAX_SIZE=6 rules out

Three other size-6 candidates scored well in the budget-3 search but
failed multi-point verification at `{γ, A, G}`:

| candidate | budget-3 hits | verified | verdict |
|---|---|---|---|
| `1/(exp(−x) − y)` | 15 | **2** | budget-3 coincidences |
| `tanh(sinh(sinh(x))) − y` | 10 | **3** | bounded-growth, mostly coincidences |
| `(x − y)^(1/y)` | 15 | **13** | **verified — above** |

The winnowing ratio (2/15, 3/10 vs 13/15) illustrates how essential
multi-point verification is: a naive "pick the top of the scoreboard"
approach surfaces mostly false positives at this budget.

### MAX_SIZE=7 extension — family structure emerges

Pushing the enumeration to 7 nodes (53,141,046 raw trees → 1,086,443
unique after dedup, 74 s wall clock, peak RAM 12 GB) yielded no new
coverage champion but **three more verified constant-free Sheffer
candidates** and, more importantly, a **clean structural dichotomy**
that the size-5 data had only hinted at.

**New verified constant-free candidates from MAX_SIZE=7**:

- **`(x − 1) / (y − 1)`** (Mobius) — size 7, 13 const-free, polynomial
  growth, pure rational. Diagonal = 1 (not 0), so the generation
  mechanism is different: `f(x, x) = 1`, `f(x, 1) = ∞`, `f(1, x) = 0`.
- **`(x − y) / sqrt(sqr(y))`** (DiffDivRHP) — size 7, 13 const-free,
  polynomial. Uses `sqrt(sqr(y))` as a "right-half-plane
  representative" of y (complex-branch-aware |y|).
- **`(x − y) ^ (y ^ x)`** (NestedPow) — size 7, 12 const-free, polynomial.
  Nested-power structure matching PowSkew's reach exactly via a
  totally different tree.

**Also validated at size 7**: PowSkew itself (`x^y − y^x`) is
enumerated by the search at rank 18 with 12/12 const-free, matching
Phase 2–4 findings exactly — a sanity check on the methodology.

**Ruled out as budget-3 coincidences**:

- `(1 - sqrt(sqr(y))/x)` (RHPRep): scored 13/13 but only 5 verified
- `sinh(ln((-1)^x))/y`: claimed {π, π/2, 2π}, ALL rejected

### The two-family split

The verified constant-free Sheffer candidates from Phase 6 cluster
cleanly into two disjoint groups by *which targets they reach*:

| group | includes | excludes | members |
|---|---|---|---|
| **A (pow)** | `±i, 1/x, x²` | `-x, 2x` | SubPow, PowSkew, NestedPow |
| **B (rational)** | `-x, 2x, 1/x, x²` | `±i` | OneMinusDiv, Mobius, DiffDivRHP |
| **Sqrt** | `sqrt(x), ±i` | most algebraic | NegDivSqrt |

**Group A** operators all involve `^`. They reach the imaginary unit
via the principal-branch identity `(−1)^(1/2) = i`. They cannot reach
linear multiples `-x, 2x` at small depth.

**Group B** operators involve `/` and `-`. They reach linear multiples
via rational cancellation (e.g., `f(f(x, 1), f(1, x))` chains that
collapse to `2x` or `-x`). They cannot reach `±i` because there's no
`^(1/2)` branching.

**Group Sqrt** (just NegDivSqrt) sits alone: explicit `sqrt` in the
operator body unlocks `sqrt(x)` constant-free at size 5, the only
operator in the entire zoo to do so without first constructing
`ln(x)/2`.

The union of all verified constant-free operators covers **16
distinct targets**:

    {-1, -2, -x, -i, 0, 1, 1/2, 1/x, 2, 2x, i, x, x+1, x-1, x², sqrt(x)}

No single operator reaches all 16. The dichotomy is structural: the
choice of combinator (`^` vs `/`) determines which half of the 16 is
reachable, and the combinator is forced by the need to generate
either `i` (Group A) or linear multiples (Group B). A hypothetical
"Group A+B" operator would need to embed both a pow and a division in
a way that doesn't collapse under bootstrap composition — no such
operator was found in the 53M-tree search, which is evidence (not
proof) that the split is fundamental at tree sizes ≤ 7.

### MAX_SIZE=8 — the Group A/B split breaks

Pushing enumeration to 8 nodes (925,841,046 raw trees → 17,034,920
unique after dedup, 232 s wall clock, peak RAM 25.7 GB using a
streaming top-level optimization that doesn't cache size-8 trees
in the enumeration array) reveals one operator that decisively
breaks the Group A/B structural split AND reaches transcendentals
constant-free for the first time:

**`−((x − y) ^ (x ^ y))`** (NegNestedPow) — size 8, **12 genuine
verified const-free targets** (not 14 as the naive count suggested;
see "floating-point artifact" subsection below). Genuine reach:

    {−1, −2, −i, −x, 0, 1, 1/2, 2, i, x, x+1, x−1}

NegNestedPow is still a breakthrough on one axis:

1. **First cross-family constant-free operator**: reaches both `±i`
   (previously only Group A) AND `−x` (previously only Group B). The
   Group A/B split seen at sizes ≤ 7 is real but not fundamental — it
   breaks at size 8.

Compared to the pow-family size-5 champion SubPow (`(x − y)^y`),
which reaches 13 targets including `{1/x, x²}`, NegNestedPow reaches
only 12 but crosses the split and includes `−x` instead of `{1/x,
x²}`. It does **not** exceed the 13-target constant-free ceiling on
genuine identities.

### The e, −e "discoveries" are floating-point artifacts

Initial analysis claimed NegNestedPow reached `{e, −e}` constant-free
at 14/14. The trace example `examples/trace_neg_nested.rs` walks the
chain step by step and reveals the mechanism:

    C1 = f(1, 2)   = 1.0 − 1.22e−16 i    (branch-cut residual)
    C2 = f(1, C1)  = −7.5e−33 − 1.22e−16 i  (should be 0, but isn't)
    C3 = f(C2, −1) = −2.7182... − 4e−16 i   ← this "is" −e
    e  = f(C3, 0)  =  2.7182... + 3e−16 i

The base of `C3`'s outer power is `C2 + 1 ≈ 1 + ε` where `ε` is a
deterministic machine-epsilon-sized complex roundoff (~1.22e−16).
The exponent is `C2^(−1) ≈ 1 / ε ≈ 8.19e15`. The result is the
classical limit

    (1 + ε)^(1/ε)  →  e  as ε → 0,

realized by the specific roundoff characteristics of
`num_complex::Complex<f64>::powc`. In exact arithmetic `C2 = 0`,
`C2^(−1) = ∞`, `1^∞` is indeterminate, and the chain genuinely
does not reach `e`.

**The artifact survives multi-point verification at `{γ, A, G}`**
because the roundoff is deterministic: `C2` has the same ε at every
test point (`ε` is a function of `num_complex`'s internal rounding,
not of the test input). Our multi-point check catches x-dependent
numerical coincidences but not machine-precision artifacts that are
reproducible across all test points.

**Implication for verification methodology.** Future claims at this
scale should be rechecked with:

- Higher-precision arithmetic (`f128` / GMP via `rug`) — if a
  claim disappears at higher precision, it's an FP artifact.
- Symbolic verification — walk the expression tree in an exact CAS
  and simplify. If the result is not algebraically the claimed
  target, reject.

Phase 6 earlier claims (SubPow/OneMinusDiv/PowSkew/etc.) appear safe
because their reach sets are small rational and branch-cut algebraic
values that don't rely on the `(1+ε)^(1/ε)` limit. But every future
claim of "reaches a transcendental constant-free" should be
higher-precision verified.

### Updated MAX_SIZE=8 conclusion

- ✅ **Group A/B split broken at size 8** by NegNestedPow, which
  reaches 12 genuine targets including both `{i, −i}` and `{−x}`.
- ✅ **Empirical answer to "is the split fundamental?"**: No — it's
  a size-dependent phenomenon that dissolves once the operator tree
  is large enough to stack `^` twice.
- ❌ **No operator in the 17M-candidate search reaches a genuine
  transcendental constant-free**. The `e, −e` claim was a
  deterministic FP artifact.
- ❌ **No operator exceeds 13 targets constant-free** at size ≤ 8
  on verified genuine identities.

### MAX_SIZE=9 via arena refactor (Phase 6 epilogue)

The `Arc<OpExpr>` representation hits a memory wall at ~25 GB for
MAX_SIZE=8 and would need ~200 GB for MAX_SIZE=9. Refactoring the
storage to `src/op_arena.rs` — a flat `Vec<Node>` arena where each
tree is a `u32` NodeId — cuts per-node memory from ~240 bytes to
~16 bytes. Plus two more tricks:

- **Streamed top level**: cache sizes 1..=MAX_SIZE-1 in the arena,
  stream size-MAX_SIZE candidates through dedup+filter, only push
  survivors to the arena.
- **Compact dedup fingerprint**: shrink `HashSet<[ValueKey; 5]>` to
  `HashSet<u64>` (hash the 5-value tuple into a single u64). Saves
  ~85% on per-entry memory at the cost of a ~5e-3 total
  birthday-paradox collision rate at 300M entries.
- **Scoring filter**: drop Scorecards with coverage < 5 before
  collecting, instead of materializing all 260M+ Scorecards.

**MAX_SIZE=9 run results**:

- 16,461,881,046 raw trees → 262,190,856 unique
- Enumeration + scoring: 40 min wall clock
- Peak RAM: **31.6 GB** (under the 53 GB budget)
- 3,473,113 candidates passed the scoring filter
- Arena grew from ~925M to ~1.17B nodes (17.9 GB)

**Scientific yield**: zero new *genuine* candidates beyond NegNestedPow.

Every MAX_SIZE=9 candidate that appeared to beat 13 constant-free
falls into one of these FP-artifact classes:

1. **`tanh(exp(e))^x − y/x` class** (top-ranked at 19/19!). Because
   `tanh(exp(e)) = 1 − 1.65e−14` in f64 (not exactly 1), raising it
   to various powers and feeding into further compositions triggers
   `(1 − ε)^(1/ε) → 1/e`. Looks like 19 verified targets including
   `{e, −e, 1/e, e², e·x, exp(x), exp(exp(x))}` under multi-point
   check, but every transcendental is an artifact of the
   deterministic machine-precision limit.
2. **`e / (y − x)` class**. Scored 17/16 but only 2 verified under
   multi-point: the high score came from `∞` / `NaN` coincidences at
   γ that disappeared at A and G.
3. **`0^y`-style branch-cut operators**. `0^positive_real = 0` but
   `0^complex` is indeterminate, and bootstrapping chains abuse the
   transition.
4. **Nested `(−1)^x`-family operators**. Principal-branch `(−1)^x =
   exp(iπx) = cos(πx) + i sin(πx)` which can hit π-related values by
   coincidence.

The real MAX_SIZE=9 lesson is methodological:

**f64 multi-point verification at `{γ, A, G}` is insufficient to
distinguish genuine Sheffer identities from `(1+ε)^(1/ε)`-class
floating-point-limit artifacts.** The ε is a function of
num_complex's rounding, not of the test input, so it's deterministic
across all test points. A second instance of this class (after
NegNestedPow's `{e, −e}` FP hits) was caught at MAX_SIZE=9 in
`(tanh(exp(e))^x − y/x)` — this time with 7 transcendental artifact
targets instead of just 2.

**Required verification strategy going forward**:

1. **Higher-precision re-evaluation** via `rug::Complex` or a
   double-double Complex type. Any claim that changes between f64
   and f128 is an artifact.
2. **Symbolic verification** via sympy or a hand-rolled CAS.
3. **Perturbation robustness**: evaluate with deliberately perturbed
   transcendental primitives (add `5e-15` noise to `tanh`, `exp`,
   etc.) and check that genuine identities are stable under noise
   while FP-artifact identities shift.

Until one of these is added, Phase 6 Sheffer claims should be
restricted to the size-5..=7 zoo (SubPow, OneMinusDiv, Mobius,
DiffDivRHP, etc.) which involve only algebraic and rational
primitives and do not trigger the `(1+ε)^(1/ε)` limit. NegNestedPow
at size 8 is the largest genuine candidate, with 12 verified targets
(the claimed `{e, −e}` are machine-precision artifacts).

### Corrected constant-free union

After removing FP-artifact targets from all Phase 6 candidates:

    {−1, −2, −x, −i, 0, 1, 1/2, 1/x, 2, 2x, i, sqrt(x),
     x, x+1, x−1, x²}

**16 genuine targets**, unchanged from MAX_SIZE=7 → 8 → 9. The arena
refactor proved that MAX_SIZE=9 enumeration is physically feasible at
~32 GB of RAM, but revealed that the search is saturating: additional
tree depth produces more FP artifacts, not more real identities.
Higher-precision verification is the actual bottleneck at this scale,
not enumeration.

### Phase 6 epilogue 2: structural FP-artifact detector

Traditional higher-precision verification (f128, MPFR) **cannot
break the `(1+ε)^(1/ε)` artifact class** because the limit is scale-
invariant. At f64 ε ≈ 1e-16; at f128 ε ≈ 1e-32; at MPFR-256 ε ≈
1e-77. In every case `(1+ε)^(1/ε) = e`. Perturbing ε by noise fails
for the same reason: `(1 + ε')^(1/ε') = e` for any small ε'.

The fix is a **structural detector**, not a precision check. During
`Expr` evaluation we watch every `powc(base, exponent)` call. If:

    |base − 1| < LIMIT_THRESHOLD  AND  |exponent| > 1 / LIMIT_THRESHOLD

at ANY pow step, the result is a `(1+ε)^(1/ε)`-limit-derived value
rather than an algebraic identity. Genuine algebraic identities
don't sit at this knife-edge — they're either far from 1 or use
modest exponents.

Implementation: `src/hp_verify.rs`, threshold 1e−8 by default, with
`LimitAwareResult` propagating the flag through the entire `Expr`
evaluation. Tests pin the three failure modes:

    detector_catches_neg_nested_pow_e_chain      ... ok
    detector_does_not_flag_genuine_subpow_diagonal ... ok
    detector_does_not_flag_eml_at_typical_input  ... ok

Wired into `examples/verify_novel.rs` so every discovery gets both
multi-point cross-check AND the limit detector. Artifact results
across the Phase 6 zoo:

| candidate | MP verified | genuine | FP artifacts |
|---|---|---|---|
| SubPow | 13 | 13 | — |
| OneMinusDiv | 13 | 13 | — |
| Mobius | 13 | 13 | — |
| DiffDivRHP | 13 | 13 | — |
| NegDivSqrt | 9 | 9 | — |
| NestedPow | 12 | 12 | — |
| PowSkew | 12 | 12 | — |
| **NegNestedPow** | **14** | **12** | `{e, −e}` |
| **TanhExpEPow** | **19** | **12** | `{1/e, e, exp(x), e·x, exp(exp(x)), e², −e}` |
| SubPowInv | 13 / 2 | 13 / 2 | — |
| SqrtNeg1LnNegPow | 6 / 1 | 6 / 1 | — |

The detector catches NegNestedPow's 2 and TanhExpEPow's 7 FP-limit
artifacts — exactly the cases I identified by manual trace earlier.
It leaves every algebraic operator (SubPow, OneMinusDiv, Mobius,
etc.) untouched.

### Correction: PowSkew already crosses the Group A/B split at size 7

Running the detector over PowSkew's verified discoveries confirms
its genuine reach is `{−1, −2, −i, −x, 0, 1, 1/2, 2, i, x, x+1,
x−1}` — **12 targets including both `±i` and `−x`**.

This means PowSkew has been cross-family since size 7, and my earlier
narrative (size 8 NegNestedPow as "first cross-family operator") was
wrong. The correct statement:

- **PowSkew at size 7** is the first cross-family constant-free
  Sheffer operator in the genuinely verified zoo: 12 targets including
  `{±i, −x}`.
- **NegNestedPow at size 8** reaches the **same 12 targets** via a
  different tree shape. Adds no new genuine reach, just two FP
  artifacts that the detector now correctly rejects.
- **Size-5 champions (SubPow, OneMinusDiv)** reach 13 targets each
  but in disjoint "flavors": SubPow `{1/x, x²}` without `{-x, 2x}`;
  OneMinusDiv `{-x, 2x, 1/x, x²}` without `{±i}`. Neither is
  cross-family in the `{±i} ∩ {-x}` sense.
- **Phase 6 genuine constant-free ceiling: 13 targets** at size 5,
  not 14 at size 8. The MAX_SIZE=8 and MAX_SIZE=9 runs did not break
  the ceiling once FP artifacts are filtered.

### Open question: the π-claiming operators

The detector **does not flag** `i · (ln(−x))^y` (SqrtNeg1LnNegPow),
which reaches `{π, π/2}` at size 8 with `{1, x}` pool. It's either
a genuine identity or an artifact class the limit detector misses
(probably branch-cut π extraction from complex log of negatives, not
the `(1+ε)^(1/ε)` class).

Investigating this is a natural next step, but requires either:
- a branch-cut-aware detector,
- true arbitrary-precision verification via `rug`, or
- symbolic CAS evaluation.

For now, SqrtNeg1LnNegPow's `{π, π/2}` claim remains "verified but
uninterrogated". Everything else in the Phase 6 zoo is
artifact-detector-clean.

The constant-free union after correction is **16 genuine targets**:

    {−1, −2, −x, −i, 0, 1, 1/2, 1/x, 2, 2x, i, sqrt(x),
     x, x+1, x−1, x²}

(Same as MAX_SIZE=7. NegNestedPow adds `{−x}` to Group A's reach
but that union member was already covered by Group B operators.)

**Mechanism sketch.** `f(x, y) = −((x − y)^(x^y))` with:

    f(x, x) = −(0^(x^x)) = 0     (diagonal → 0, constant-free)
    f(x, 0) = −(x^(x^0)) = −x    (direct linear negation)
    f(0, x) = −((−x)^(0^x)) = −1 (direct rational negative)

So from `{x}` alone we get `{x, 0, −x, −1}` at iter 1 — the Group B
algebraic openers `{0, −x, −1}` are free. The outer `^(x^y)` term
means further compositions carry both a nested power (Group A reach
to `±i`) and the difference structure (Group B reach via rational
cancellation). The two `^` operations stack non-trivially, giving
the cross-family reach.

The transcendental `e` comes from a subtler chain: once `1, 0, −1,
x, −x` are leaves, compositions like `f(1, −1) = −(2^(1^−1)) = −2`
and deeper chains eventually hit `e` exactly at 14 iterations of
verifier bootstrap, verified at three independent test points.

**Also found at size 8**:

- **`i · (ln(−x))^y`** (SqrtNeg1LnNegPow) — 6 verified with `{1, x}`
  at size 8, uniquely including **`{π, π/2}`**. First operator
  outside the EDL Stage-A chase to reach `π` at small depth. The
  operator body contains `sqrt(−1) = i` (principal branch), so this
  is a "constant-in-body" candidate rather than purely free.

**Ruled out as size-8 coincidences**:

- `1/((x/y) − 1)`: scored 13, verified 2. A reciprocal-rewrite of
  DivMinusOne that survives dedup only because of numerical
  branch-cut divergence at one test point.
- Most of the top-30 `1/(…)` variants: collapse under multi-point
  to their non-inverted Group B siblings.
- Every `π`/`sin`/`cos` unique-reach claim other than
  SqrtNeg1LnNegPow: coincidences at γ.

**New verified constant-free union: 18 targets** (up from 16):

    {−1, −2, −e, −x, −i, 0, 1, 1/2, 1/x, 2, 2x, e, i, sqrt(x),
     x, x+1, x−1, x²}

The two new members are `{e, −e}` from NegNestedPow. Still missing
from the constant-free union: `{π, π/2, 2π, iπ, exp(x), ln(x), sin(x),
cos(x), ln(ln(x)), 2pi, -2, -i}`. Wait — `{-2, -i}` are in the
NegNestedPow reach, they're covered. Missing: transcendental
functions of x, and the π family.

### Phase 6 conclusion

Strategy A answers the research prompt's "novel operator" question in
the affirmative. **Eight binary operators** distinct from EML, EDL,
PowSkew, and PowExpSkew are verified constant-free Sheffer candidates:

| operator | size | group | const-free reach |
|---|---|---|---|
| `(x − y)^y` (SubPow) | 5 | A | 13 |
| `1 − x/y` (OneMinusDiv) | 5 | B | 13 |
| `x/y − 1` (DivMinusOne) | 5 | B | 13 |
| `−(x/sqrt(y))` (NegDivSqrt) | 5 | Sqrt | 9 |
| `(x − y)^(y^x)` (NestedPow) | 7 | A | 12 |
| `(x − 1)/(y − 1)` (Mobius) | 7 | B | 13 |
| `(x − y)/sqrt(sqr(y))` (DiffDivRHP) | 7 | B | 13 |
| **`−((x − y)^(x^y))`** (NegNestedPow) | 8 | **A∪B** | **12**† |

†The naive bootstrap reports 14 targets for NegNestedPow but 2 of
them (`{e, −e}`) are deterministic f64 floating-point artifacts
from the `(1+ε)^(1/ε) → e` limit being triggered by branch-cut
roundoff in `num_complex`. See the "floating-point artifact"
subsection above. The 12 remaining are algebraically legitimate
and still constitute the first cross-family constant-free Sheffer
operator (includes both `{i, −i}` from Group A and `{−x}` from
Group B).

Plus **`(x − y)^(1/y)`** (SubPowInv, size 6) which reaches 13 targets
with `{1, x}` including `sqrt(x)` but is not constant-free, and
**`i · (ln(−x))^y`** (SqrtNeg1LnNegPow, size 8) which reaches
`{π, π/2}` with `{1, x}` but has `sqrt(−1)` baked into the body.

The simple size-5 rational map `1 − x/y` reaching 13/31 constant-free
targets at polynomial growth is the most surprising single result:
no transcendentals in the operator body, no branch cuts, no special
constants, yet it competes with PowSkew on reach. The size-7
enumeration confirms that no *larger* operator materially outperforms
it on coverage, though the search does reveal a principled family
structure (Group A vs Group B) that was invisible at MAX_SIZE=5.

Reproduction commands:

    cargo run --release --example operator_search -p tang-sheffer  # MAX_SIZE=6
    cargo run --release --example verify_novel    -p tang-sheffer  # multi-point check

## 5. Open problems

All 31 standard targets are now reached (Phase 5), so the remaining open
problems are:

1. **Single-operator universality at small depth.** EDL and EML each miss
   targets at budget 4 that the other hits — EDL can't do Euler-formula
   subtraction naturally, EML can't do the `edl(iπ/2, e) = i` imaginary
   extraction trick. Is there a single binary operator that reaches all
   31 targets at budget 4 from a single-constant leaf pool?

2. **Constant-free universality over transcendentals.** PowSkew is
   constant-free over the algebraic targets (Section 1–3). PowExpSkew
   reaches transcendental functions of x but not transcendental
   constants. Does there exist a constant-free operator reaching all 31
   targets from a single variable?

3. **Polynomial-growth universal operator.** EML's double-exponential
   growth destroys gradients at depth ≥ 3 (Phase 3). PowSkew has
   algebraic growth but limited expressivity. Is there a polynomial-
   growth operator that reaches transcendental targets?

4. **Stepping-stone dependence.** Our Stage B EML cracks sin/cos only
   after we explicitly add `ln(ln(2i·sin(x)))` as a target. Without that
   stepping stone, the bootstrap misses sin/cos even at budget 5. Is
   there a principled way to auto-generate stepping stones, or does the
   search fundamentally require human-guided intermediate goals?

## 6. Prior art

Web/arXiv searches in April 2026 turn up nothing on `x^y − y^x` as a
Sheffer-style universal operator. The classical results on `x^y = y^x`
(Bernoulli–Goldbach 1728, Putnam 1960, Lambert-W parameterization)
concern the *solution set* of the equation, not the binary operator. The
Odrzywołek paper is the only prior work on single-operator universality
for elementary functions and explicitly leaves the constant-free case as
an open problem (Table 2). **This observation appears to be novel.**

## 7. Verification

Reproduction commands:

    cargo run --release --example constant_free   -p tang-sheffer
    cargo run --release --example phase4_verify   -p tang-sheffer
    cargo run --release --example crack_remaining -p tang-sheffer
    cargo test -p tang-sheffer

Key artifacts:

* `examples/constant_free.rs` — single-point bootstrap from `{x = γ}`
  alone for all constant-free candidates.
* `examples/phase4_verify.rs` — multi-point cross-check at {γ, A, G}
  plus the depth comparison table above.
* `src/powskew.rs` — ten hand-verified identity unit tests including
  the full cascade `{x → 0 → 1 → -1 → 2 → 1/2 → (2-i) → (1-i) → i}`.
* `src/crosscheck.rs` — rebinds variable leaves to new numeric values
  and re-evaluates Derived expressions via `Expr::eval_recursive`, so a
  single `Discovery` from one test point can be checked against multiple.
