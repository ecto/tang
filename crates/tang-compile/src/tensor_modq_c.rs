//! A [`ModGraph`] lowered to C, built by the system compiler, and checked at
//! the machine-code level.
//!
//! The lowering keeps the IR's split visible in the binary. Each map body —
//! the only code that computes on values — becomes its own `noinline`
//! function of scalars (its args and its point's coordinates), with the
//! field helpers forced inline into it. Everything around the bodies is a
//! driver: loops over public shapes, affine reads, and gathers whose index
//! the secrecy check has proven public. So "secrets never steer a branch or
//! an address" becomes something a disassembly can confirm:
//! [`Compiled::lint`] reads the compiler's assembly and requires every body
//! to contain no conditional branch and no divide.
//!
//! Residue inputs must already be canonical (< q); the C does not reduce
//! them, since `%` on a secret would be a divide.

use super::{BOp, MOp, ModGraph, ModId};
use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

const PRELUDE: &str = r#"#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define AI static inline __attribute__((always_inline))
AI uint32_t ct_csub(uint32_t x) { uint32_t y = x - Q; uint32_t m = 0u - (y >> 31); return y + (Q & m); }
AI uint32_t ct_add(uint32_t a, uint32_t b) { return ct_csub(a + b); }
AI uint32_t ct_sub(uint32_t a, uint32_t b) { return ct_csub(a + Q - b); }
AI uint32_t ct_mont(uint32_t a, uint32_t b) {
    uint64_t t = (uint64_t)a * b; uint32_t m = (uint32_t)t * QNEG_INV;
    return ct_csub((uint32_t)((t + (uint64_t)m * Q) >> 32));
}
AI uint32_t ct_mul(uint32_t a, uint32_t b) { return ct_mont(ct_mont(a, b), R2); }
AI uint64_t ct_div_q(uint64_t v) {
    uint64_t est = (uint64_t)(((unsigned __int128)v * BARRETT) >> 64);
    uint64_t r = v - est * Q;
    return est + ((((uint64_t)Q - 1 - r) >> 63) & 1);
}
AI uint32_t ct_compress(uint32_t x, uint32_t d) {
    return (uint32_t)ct_div_q(((uint64_t)x << d) + (Q >> 1)) & ((1u << d) - 1);
}
AI uint32_t ct_decompress(uint32_t y, uint32_t d) {
    return (uint32_t)(((uint64_t)y * Q + ((1ull << d) >> 1)) >> d);
}
AI uint32_t ct_p2r(uint32_t x, uint32_t d) { return (x + ((1u << d) >> 1) - 1) >> d; }
AI uint32_t ct_sel(uint32_t c, uint32_t a, uint32_t b) { uint32_t m = 0u - (c & 1); return (a & m) | (b & ~m); }
AI uint32_t ct_eq(uint32_t a, uint32_t b) { uint32_t d = a ^ b; return ((d | (0u - d)) >> 31) ^ 1; }
AI void ct_decompose(uint32_t r, uint64_t g2, uint64_t m, uint32_t *hi, uint32_t *lo) {
    uint64_t alpha = 2 * g2;
    uint64_t r1 = ((uint64_t)r * m) >> 40;
    uint64_t rem = (uint64_t)r - r1 * alpha;
    r1 += ((alpha - 1 - rem) >> 63) & 1;
    uint64_t rp = (uint64_t)r - r1 * alpha;
    uint64_t up = ((g2 - rp) >> 63) & 1;
    r1 += up;
    int64_t r0 = (int64_t)rp - (int64_t)(up * alpha);
    uint32_t d = (uint32_t)(r1 * alpha) ^ (Q - 1);
    uint32_t edge = ((d | (0u - d)) >> 31) ^ 1;
    *hi = (uint32_t)r1 & ~(0u - edge);
    *lo = ct_csub((uint32_t)(r0 - (int64_t)edge + (int64_t)Q));
}
AI uint32_t ct_high(uint32_t r, uint64_t g2, uint64_t m) { uint32_t h, l; ct_decompose(r, g2, m, &h, &l); return h; }
AI uint32_t ct_low(uint32_t r, uint64_t g2, uint64_t m) { uint32_t h, l; ct_decompose(r, g2, m, &h, &l); return l; }
"#;

fn op_c(op: &BOp, out: &mut String, i: usize) {
    let r = |k: usize| format!("r{k}");
    let e = match *op {
        BOp::Arg(a) => format!("a{a}"),
        BOp::Lit(x) => format!("{x}u"),
        BOp::Coord(d) => format!("c{d}"),
        BOp::Add(a, b) => format!("ct_add({}, {})", r(a), r(b)),
        BOp::Sub(a, b) => format!("ct_sub({}, {})", r(a), r(b)),
        BOp::Mul(a, b) => format!("ct_mul({}, {})", r(a), r(b)),
        BOp::Sel(c, a, b) => format!("ct_sel({}, {}, {})", r(c), r(a), r(b)),
        BOp::Compress(a, d) => format!("ct_compress({}, {d}u)", r(a)),
        BOp::Decompress(a, d) => format!("ct_decompress({}, {d}u)", r(a)),
        BOp::Power2Round(a, d) => format!("ct_p2r({}, {d}u)", r(a)),
        BOp::Xor(a, b) => format!("({} ^ {})", r(a), r(b)),
        BOp::And(a, b) => format!("({} & {})", r(a), r(b)),
        BOp::Or(a, b) => format!("({} | {})", r(a), r(b)),
        BOp::Not(a) => format!("(~{})", r(a)),
        BOp::Shl(a, b) => format!("({} << ({} & 31u))", r(a), r(b)),
        BOp::Shr(a, b) => format!("({} >> ({} & 31u))", r(a), r(b)),
        BOp::Eq(a, b) => format!("ct_eq({}, {})", r(a), r(b)),
        BOp::Lt(a, b) => format!("(({} - {}) >> 31)", r(a), r(b)),
        BOp::MulHi(a, b) => format!("(uint32_t)(((uint64_t){} * {}) >> 32)", r(a), r(b)),
        BOp::HighBits(a, g2) => {
            let m = (1u64 << 40) / (2 * g2 as u64);
            format!("ct_high({}, {g2}ull, {m}ull)", r(a))
        }
        BOp::LowBits(a, g2) => {
            let m = (1u64 << 40) / (2 * g2 as u64);
            format!("ct_low({}, {g2}ull, {m}ull)", r(a))
        }
    };
    let _ = writeln!(out, "    uint32_t r{i} = {e};");
}

impl ModGraph {
    /// One C translation unit: `void graph(const uint32_t *const *in,
    /// uint32_t *const *out)`, plus a `main` that runs it on stdin (`run`) or
    /// times it over a file of input sets (`time <file>`).
    pub fn to_c(&self, outputs: &[ModId]) -> String {
        let f = self.f;
        let mut s = format!(
            "#define Q {}u\n#define QNEG_INV {}u\n#define R2 {}u\n#define BARRETT {}ull\n{PRELUDE}",
            f.q, f.qneg_inv, f.r2, f.barrett
        );
        // bodies
        for (i, n) in self.nodes.iter().enumerate() {
            let MOp::Map { args, body, out } = &n.op else {
                continue;
            };
            let params: Vec<String> = (0..args.len())
                .map(|a| format!("uint32_t a{a}"))
                .chain((0..n.shape.len()).map(|d| format!("uint32_t c{d}")))
                .collect();
            let params = if params.is_empty() {
                "void".to_string()
            } else {
                params.join(", ")
            };
            let _ = writeln!(
                s,
                "__attribute__((noinline)) uint32_t body_{i}({params}) {{"
            );
            for (k, op) in body.ops.iter().enumerate() {
                op_c(op, &mut s, k);
            }
            let _ = writeln!(s, "    return r{out};\n}}");
        }
        // buffers
        for (i, n) in self.nodes.iter().enumerate() {
            let size: usize = n.shape.iter().product();
            match &n.op {
                MOp::Const(c) => {
                    let vals: Vec<String> = c.iter().map(|x| format!("{x}u")).collect();
                    let _ = writeln!(
                        s,
                        "static const uint32_t k{i}[{}] = {{{}}};",
                        size.max(1),
                        vals.join(",")
                    );
                }
                MOp::Map { .. } | MOp::Gather { .. } | MOp::Filter { .. } => {
                    let _ = writeln!(s, "static uint32_t b{i}[{}];", size.max(1));
                }
                _ => {}
            }
        }
        // one driver function per node, so no single function grows with the graph
        for (i, n) in self.nodes.iter().enumerate() {
            let _ = writeln!(s, "static const uint32_t *p{i};");
            let _ = n;
        }
        for (i, n) in self.nodes.iter().enumerate() {
            let size: usize = n.shape.iter().product();
            match &n.op {
                MOp::Gather { table, index } => {
                    let tl: usize = self.shape(*table).iter().product();
                    let _ = writeln!(
                        s,
                        "__attribute__((noinline)) static void node_{i}(void) {{ for (size_t j = 0; j < {size}; j++) b{i}[j] = p{}[p{}[j] % {tl}u]; p{i} = b{i}; }}",
                        table.0, index.0
                    );
                }
                MOp::Filter { src, bound } => {
                    let _ = writeln!(
                        s,
                        "__attribute__((noinline)) static void node_{i}(void) {{ size_t w = 0; for (size_t j = 0; j < {size}; j++) if (p{s0}[j] < {bound}u) b{i}[w++] = p{s0}[j]; while (w < {size}) b{i}[w++] = 0; p{i} = b{i}; }}",
                        s0 = src.0
                    );
                }
                MOp::Map { args, .. } => {
                    let rank = n.shape.len();
                    let _ = writeln!(s, "__attribute__((noinline)) static void node_{i}(void) {{\n    size_t at = 0;");
                    for (d, len) in n.shape.iter().enumerate() {
                        let _ = writeln!(s, "    for (int64_t i{d} = 0; i{d} < {len}; i{d}++)");
                    }
                    s.push_str("    {\n");
                    for (a, (src, view)) in args.iter().enumerate() {
                        let shape = self.shape(*src);
                        let coords: Vec<String> = view
                            .forms
                            .iter()
                            .map(|form| {
                                let mut e = format!("{}", form.offset);
                                for (d, c) in form.coefs.iter().enumerate() {
                                    if *c != 0 {
                                        let _ = write!(e, " + {c} * i{d}");
                                    }
                                }
                                format!("({e})")
                            })
                            .collect();
                        let mut flat = "0".to_string();
                        for (c, len) in coords.iter().zip(shape) {
                            flat = format!("({flat}) * {len} + {c}");
                        }
                        if view.guard {
                            let inside: Vec<String> = coords
                                .iter()
                                .zip(shape)
                                .map(|(c, len)| format!("{c} >= 0 && {c} < {len}"))
                                .collect();
                            let _ = writeln!(
                                s,
                                "      uint32_t a{a} = ({}) ? p{}[{flat}] : 0u;",
                                inside.join(" && "),
                                src.0
                            );
                        } else {
                            let _ = writeln!(s, "      uint32_t a{a} = p{}[{flat}];", src.0);
                        }
                    }
                    let call: Vec<String> = (0..args.len())
                        .map(|a| format!("a{a}"))
                        .chain((0..rank).map(|d| format!("(uint32_t)i{d}")))
                        .collect();
                    let _ = writeln!(
                        s,
                        "      b{i}[at++] = body_{i}({});\n    }}\n    p{i} = b{i};\n}}",
                        call.join(", ")
                    );
                }
                _ => {}
            }
        }
        s.push_str("void graph(const uint32_t *const *in, uint32_t *const *out) {\n");
        for (i, n) in self.nodes.iter().enumerate() {
            match &n.op {
                MOp::Input(k, _) => {
                    let _ = writeln!(s, "  p{i} = in[{k}];");
                }
                MOp::Const(_) => {
                    let _ = writeln!(s, "  p{i} = k{i};");
                }
                MOp::Reshape(src) | MOp::Declassify { src, .. } => {
                    let _ = writeln!(s, "  p{i} = p{};", src.0);
                }
                _ => {
                    let _ = writeln!(s, "  node_{i}();");
                }
            }
        }
        for (k, o) in outputs.iter().enumerate() {
            let size: usize = self.shape(*o).iter().product();
            let _ = writeln!(
                s,
                "  memcpy(out[{k}], p{}, {size} * sizeof(uint32_t));",
                o.0
            );
        }
        s.push_str("}\n");

        // the runner
        let in_sizes: Vec<usize> = self
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                MOp::Input(k, _) => Some((k, n.shape.iter().product::<usize>())),
                _ => None,
            })
            .fold(vec![0; self.inputs], |mut v, (k, sz)| {
                v[k] = sz;
                v
            });
        let out_sizes: Vec<usize> = outputs
            .iter()
            .map(|o| self.shape(*o).iter().product())
            .collect();
        let total_in: usize = in_sizes.iter().sum();
        let _ = writeln!(
            s,
            r#"static const size_t IN_SIZES[] = {{{ins}}};
static const size_t OUT_SIZES[] = {{{outs}}};
#define N_IN {nin}
#define N_OUT {nout}
#define TOTAL_IN {total_in}
static uint64_t now_ns(void) {{ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }}
int main(int argc, char **argv) {{
    static uint32_t inbuf[TOTAL_IN + 1];
    const uint32_t *in[N_IN + 1];
    uint32_t *out[N_OUT + 1];
    size_t off = 0;
    for (size_t k = 0; k < N_IN; k++) {{ in[k] = inbuf + off; off += IN_SIZES[k]; }}
    for (size_t k = 0; k < N_OUT; k++) out[k] = malloc((OUT_SIZES[k] + 1) * sizeof(uint32_t));
    if (argc > 2 && strcmp(argv[1], "time") == 0) {{
        FILE *fp = fopen(argv[2], "rb");
        if (!fp) return 2;
        uint64_t t;
        while (fread(inbuf, sizeof(uint32_t), TOTAL_IN, fp) == TOTAL_IN) {{
            uint64_t t0 = now_ns();
            graph(in, out);
            t = now_ns() - t0;
            fwrite(&t, sizeof t, 1, stdout);
        }}
        return 0;
    }}
    if (fread(inbuf, sizeof(uint32_t), TOTAL_IN, stdin) != TOTAL_IN) return 2;
    graph(in, out);
    for (size_t k = 0; k < N_OUT; k++) fwrite(out[k], sizeof(uint32_t), OUT_SIZES[k], stdout);
    return 0;
}}"#,
            ins = in_sizes
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(","),
            outs = out_sizes
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(","),
            nin = self.inputs,
            nout = outputs.len(),
        );
        s
    }

    /// Lower to C, build a runner with the system compiler (`$CC`, else `cc`)
    /// at `-O2`, and keep its assembly for [`Compiled::lint`].
    pub fn compile(&self, outputs: &[ModId]) -> io::Result<Compiled> {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "tang-modq-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir)?;
        let src = dir.join("graph.c");
        std::fs::write(&src, self.to_c(outputs))?;
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
        let exe = dir.join("graph");
        let asm = dir.join("graph.s");
        for (flag, path) in [("-o", &exe), ("-S", &asm)] {
            let mut cmd = Command::new(&cc);
            cmd.args(["-O2", "-std=c11", "-w"]);
            if flag == "-S" {
                cmd.args(["-S", "-o"]);
            } else {
                cmd.arg("-o");
            }
            let st = cmd.arg(path).arg(&src).status()?;
            if !st.success() {
                return Err(io::Error::other(format!(
                    "{cc} failed on {}",
                    src.display()
                )));
            }
        }
        let out_sizes = outputs
            .iter()
            .map(|o| self.shape(*o).iter().product())
            .collect();
        Ok(Compiled {
            dir,
            exe,
            asm,
            out_sizes,
        })
    }
}

/// A built graph: its runner, its assembly, and its output sizes.
pub struct Compiled {
    pub dir: PathBuf,
    exe: PathBuf,
    asm: PathBuf,
    out_sizes: Vec<usize>,
}

impl Drop for Compiled {
    /// Builds are tens of megabytes; keep one with `TANG_MODQ_KEEP=1`.
    fn drop(&mut self) {
        if std::env::var_os("TANG_MODQ_KEEP").is_none() {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

fn le_bytes(inputs: &[Vec<u32>]) -> Vec<u8> {
    inputs
        .iter()
        .flatten()
        .flat_map(|x| x.to_le_bytes())
        .collect()
}

impl Compiled {
    /// Run the compiled graph once.
    pub fn run(&self, inputs: &[Vec<u32>]) -> io::Result<Vec<Vec<u32>>> {
        use std::io::Write;
        let mut child = Command::new(&self.exe)
            .arg("run")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        child
            .stdin
            .take()
            .expect("stdin")
            .write_all(&le_bytes(inputs))?;
        let out = child.wait_with_output()?;
        let words: Vec<u32> = out
            .stdout
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut res = Vec::new();
        let mut at = 0;
        for &n in &self.out_sizes {
            res.push(words[at..at + n].to_vec());
            at += n;
        }
        Ok(res)
    }

    /// Time one call per input set, in nanoseconds, in the order given.
    pub fn time(&self, sets: &[Vec<Vec<u32>>]) -> io::Result<Vec<u64>> {
        let path = self.dir.join("sets.bin");
        let bytes: Vec<u8> = sets.iter().flat_map(|s| le_bytes(s)).collect();
        std::fs::write(&path, bytes)?;
        let out = Command::new(&self.exe).arg("time").arg(&path).output()?;
        Ok(out
            .stdout
            .chunks(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect())
    }

    /// The machine-code check: in the compiler's assembly, every map body
    /// must be free of conditional branches and divides. Returns how many
    /// body functions were checked, or every offending instruction.
    pub fn lint(&self) -> Result<usize, Vec<String>> {
        lint_asm(&self.asm)
    }
}

/// Instructions that would let a value steer time: conditional branches and
/// divides (arm64 and x86-64 spellings).
fn suspicious(ins: &str) -> bool {
    let m = ins.split_whitespace().next().unwrap_or("");
    m.starts_with("b.")
        || matches!(
            m,
            "cbz" | "cbnz" | "tbz" | "tbnz" | "udiv" | "sdiv" | "br" | "blr"
        )
        || (m.starts_with('j') && m != "jmp")
        || matches!(m, "div" | "idiv" | "divl" | "divq" | "idivl" | "idivq")
}

fn lint_asm(path: &Path) -> Result<usize, Vec<String>> {
    let text = std::fs::read_to_string(path).map_err(|e| vec![e.to_string()])?;
    let mut bad = Vec::new();
    let mut bodies = 0;
    let mut current: Option<String> = None;
    for line in text.lines() {
        // drop trailing comments (`;` on Apple arm64, `//` elsewhere)
        let t = line.split(';').next().unwrap_or("");
        let t = t.split("//").next().unwrap_or("").trim();
        if let Some(label) = t.strip_suffix(':') {
            if !label.starts_with('.') && !label.starts_with('L') && !label.starts_with("ltmp") {
                let name = label.trim_start_matches('_').to_string();
                current = name.starts_with("body_").then(|| {
                    bodies += 1;
                    name
                });
            }
            continue;
        }
        if t.is_empty() || t.starts_with('.') {
            continue;
        }
        if let Some(f) = &current {
            if suspicious(t) {
                bad.push(format!("{f}: {t}"));
            }
        }
    }
    if bad.is_empty() {
        Ok(bodies)
    } else {
        Err(bad)
    }
}

/// Welch's t between two timing samples, after dropping each sample's top
/// `crop` fraction (dudect's crop against interrupt noise).
pub fn welch_t(a: &[u64], b: &[u64], crop: f64) -> f64 {
    let trim = |x: &[u64]| {
        let mut v: Vec<f64> = x.iter().map(|&t| t as f64).collect();
        v.sort_by(|p, q| p.partial_cmp(q).expect("finite"));
        v.truncate(((v.len() as f64) * (1.0 - crop)).max(2.0) as usize);
        v
    };
    let (a, b) = (trim(a), trim(b));
    let stats = |v: &[f64]| {
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
        (n, m, var)
    };
    let ((na, ma, va), (nb, mb, vb)) = (stats(&a), stats(&b));
    (ma - mb) / (va / na + vb / nb).sqrt()
}

#[cfg(test)]
mod tests {
    use super::super::{intt, ntt, Label, ModGraph};
    use super::*;

    const Q: u32 = 8380417;

    fn lcg(s: &mut u64) -> u32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as u32) % Q
    }

    fn ntt_graph() -> (ModGraph, ModId) {
        let mut g = ModGraph::new(Q);
        let (a, b) = (
            g.input(&[256], Label::Secret),
            g.input(&[256], Label::Public),
        );
        let (fa, fb) = (ntt(&mut g, a, 1753), ntt(&mut g, b, 1753));
        let p = g.map(
            &[256],
            &[(fa, ModGraph::id(1)), (fb, ModGraph::id(1))],
            |b| {
                let (x, y) = (b.arg(0), b.arg(1));
                b.mul(x, y)
            },
        );
        let c = intt(&mut g, p, 1753);
        (g, c)
    }

    #[test]
    fn c_matches_the_interpreter() {
        let (g, c) = ntt_graph();
        let built = g.compile(&[c]).expect("a C compiler");
        let mut s = 5;
        for _ in 0..3 {
            let x: Vec<u32> = (0..256).map(|_| lcg(&mut s)).collect();
            let y: Vec<u32> = (0..256).map(|_| lcg(&mut s)).collect();
            let ins = [x, y];
            assert_eq!(built.run(&ins).unwrap(), g.eval(&ins, &[c]));
        }
    }

    #[test]
    fn every_body_is_branch_and_divide_free() {
        let (g, c) = ntt_graph();
        let built = g.compile(&[c]).expect("a C compiler");
        let n = built
            .lint()
            .unwrap_or_else(|bad| panic!("leaky bodies:\n{}", bad.join("\n")));
        assert!(n > 10, "found {n} body functions");
    }

    #[test]
    fn the_lint_catches_real_compiler_branches() {
        // a body with a value-dependent call and a divide, through the real compiler
        // (clang turns a simple value-dependent loop into clz + csel, which is fine)
        let dir = std::env::temp_dir().join(format!("tang-modq-leaky-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (src, asm) = (dir.join("leaky.c"), dir.join("leaky.s"));
        std::fs::write(
            &src,
            "#include <stdint.h>\nvoid ext(void);\n__attribute__((noinline)) uint32_t body_0(uint32_t a, uint32_t b) {\n  if (a == 7) ext();\n  return a / (b | 1);\n}\n",
        )
        .unwrap();
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
        let ok = Command::new(cc)
            .args(["-O2", "-S", "-o"])
            .arg(&asm)
            .arg(&src)
            .status()
            .unwrap();
        assert!(ok.success());
        let bad = lint_asm(&asm).unwrap_err();
        assert!(bad.iter().any(|l| l.contains("div")), "{bad:?}");
        assert!(
            bad.iter().any(|l| !l.contains("div")),
            "a branch too: {bad:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_lint_catches_a_branch() {
        let dir = std::env::temp_dir().join(format!("tang-modq-lint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let asm = dir.join("x.s");
        std::fs::write(
            &asm,
            "_body_0:\n\tcmp w0, #3\n\tb.ne LBB0_2\n\tudiv w0, w0, w1\n\tret\n_graph:\n\tb.ne L1\n",
        )
        .unwrap();
        let bad = lint_asm(&asm).unwrap_err();
        assert_eq!(bad.len(), 2, "{bad:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
