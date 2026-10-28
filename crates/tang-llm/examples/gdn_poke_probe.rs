//! `gdn_poke_probe`: what poking one-hot integer rows into an existing GDN head's state does.
//!
//! SYNTHETIC. No torch / llama.cpp here, so `scripts/qwen3next_tiny.py` can't build the tiny
//! fixture; this runs the exact recurrence of `flash::reference`'s `gdn` (decay, delta update,
//! read) on random inputs instead of a model. Shapes are Qwen3.5-4B-like (S = 128, 16 k / 32 v
//! heads, hidden 2560); gates use Hugging Face's Qwen3-Next *init*: `A ~ U(0, 16)`, `dt_bias = 1`,
//! `ssm_ba` rows ~ N(0, 0.02) on a unit-RMS hidden. That is a random-init model, not a trained one:
//! the gate numbers say nothing about what training learns, only how the recurrence treats a
//! poked row given those gates. Disturbance is measured on the layer's GDN output (per-head
//! RMS-normed readout, all heads concatenated), not on logits.
//!
//!     cargo run --release -p tang-llm --example gdn_poke_probe

const S: usize = 128;
const HK: usize = 16;
const HV: usize = 32;
const HIDDEN: usize = 2560;
const T: usize = 1100;
const POKE_AT: usize = 50;
const LAYERS: usize = 3;

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn n(&mut self) -> f32 {
        let (a, b) = (self.u(), self.u());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
    fn unit(&mut self) -> Vec<f32> {
        let mut v: Vec<f32> = (0..S).map(|_| self.n()).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// One layer's per-token inputs: q, k (unit, per k head), v (per v head), decay and beta per v head.
struct Layer {
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    decay: Vec<f32>,
    beta: Vec<f32>,
}

fn layer(rng: &mut Rng) -> Layer {
    let a: Vec<f32> = (0..HV).map(|_| 16.0 * rng.u()).collect(); // exp(A_log), A ~ U(0, 16)
                                                                 // alpha, beta pre-activations: a row of N(0, 0.02) dotted with a unit-RMS hidden is
                                                                 // N(0, 0.02^2 * HIDDEN); fixed rows, fresh hidden each token
    let sd = 0.02 * (HIDDEN as f32).sqrt();
    let (mut decay, mut beta) = (vec![0f32; T * HV], vec![0f32; T * HV]);
    let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for t in 0..T {
        for h in 0..HV {
            decay[t * HV + h] = (-softplus(sd * rng.n() + 1.0) * a[h]).exp();
            beta[t * HV + h] = 1.0 / (1.0 + (-sd * rng.n()).exp());
        }
        for _ in 0..HK {
            q.push(rng.unit().iter().map(|x| x / (S as f32).sqrt()).collect());
            k.push(rng.unit());
        }
        for _ in 0..HV {
            v.push((0..S).map(|_| rng.n()).collect());
        }
    }
    Layer {
        q,
        k,
        v,
        decay,
        beta,
    }
}

/// Run head `h` over all tokens; at `POKE_AT` overwrite rows 0..n with integers. Returns the
/// RMS-normed readout per token and, per token, the norm of the poked rows' deviation from `st0`.
fn run(l: &Layer, h: usize, n: usize, keep: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let kh = h / (HV / HK);
    let mut st = vec![0f32; S * S];
    let (mut ys, mut states) = (Vec::with_capacity(T), Vec::new());
    for t in 0..T {
        if t == POKE_AT {
            for r in 0..n {
                st[r * S..(r + 1) * S]
                    .iter_mut()
                    .enumerate()
                    .for_each(|(b, e)| *e = ((r + b) % 16) as f32);
            }
        }
        let (q, k, v) = (&l.q[t * HK + kh], &l.k[t * HK + kh], &l.v[t * HV + h]);
        let (dec, b) = (l.decay[t * HV + h], l.beta[t * HV + h]);
        st.iter_mut().for_each(|e| *e *= dec);
        let mut sk = vec![0f32; S];
        for (ai, &ka) in k.iter().enumerate() {
            sk.iter_mut()
                .zip(&st[ai * S..(ai + 1) * S])
                .for_each(|(acc, &sv)| *acc += sv * ka);
        }
        let d: Vec<f32> = v.iter().zip(&sk).map(|(&vv, &kk)| (vv - kk) * b).collect();
        for (ai, &ka) in k.iter().enumerate() {
            st[ai * S..(ai + 1) * S]
                .iter_mut()
                .zip(&d)
                .for_each(|(sv, &dv)| *sv += ka * dv);
        }
        let mut o = vec![0f32; S];
        for (ai, &qa) in q.iter().enumerate() {
            o.iter_mut()
                .zip(&st[ai * S..(ai + 1) * S])
                .for_each(|(acc, &sv)| *acc += sv * qa);
        }
        let rms = (o.iter().map(|x| x * x).sum::<f32>() / S as f32 + 1e-6).sqrt();
        ys.push(o.iter().map(|x| x / rms).collect());
        states.push(st[..keep * S].to_vec());
    }
    (ys, states)
}

fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn main() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    println!("SYNTHETIC: Qwen3-Next init gates on random inputs, no trained weights\n");
    println!("(b) gates over {T} tokens, and a poked row's remaining fraction after n tokens");
    println!("layer  g mean    g min     g max     beta mean min..max   heads g>0.99  remain@1  @10  @100  @1000 (median head; best head)");
    let mut mid = None;
    for li in 0..LAYERS {
        let l = layer(&mut rng);
        let st = |x: &[f32]| {
            (
                x.iter().sum::<f32>() / x.len() as f32,
                x.iter().cloned().fold(1f32, f32::min),
                x.iter().cloned().fold(0f32, f32::max),
            )
        };
        let (gm, gmin, gmax) = st(&l.decay);
        let (bm, bmin, bmax) = st(&l.beta);
        let mean_g: Vec<f32> = (0..HV)
            .map(|h| (0..T).map(|t| l.decay[t * HV + h]).sum::<f32>() / T as f32)
            .collect();
        let near1 = mean_g.iter().filter(|&&g| g > 0.99).count();
        // measured remaining fraction of the poked row (row 0, head h), via poked minus unpoked
        let remain: Vec<[f32; 4]> = (0..HV)
            .map(|h| {
                let (_, s0) = run(&l, h, 0, 1);
                let (_, s1) = run(&l, h, 1, 1);
                let d0 = (0..S)
                    .map(|b| ((b % 16) as f32).powi(2))
                    .sum::<f32>()
                    .sqrt();
                [1, 10, 100, 1000].map(|n| dist(&s1[POKE_AT + n - 1], &s0[POKE_AT + n - 1]) / d0)
            })
            .collect();
        let pick = |i: usize| {
            let mut c: Vec<f32> = remain.iter().map(|r| r[i]).collect();
            c.sort_by(f32::total_cmp);
            (c[HV / 2], c[HV - 1])
        };
        let r: Vec<String> = (0..4)
            .map(|i| {
                let (m, b) = pick(i);
                format!("{m:.1e};{b:.1e}")
            })
            .collect();
        println!("{li:5}  {gm:.2e}  {gmin:.2e}  {gmax:.2e}  {bm:.3} {bmin:.3}..{bmax:.3}  {near1:3}/{HV}  {}", r.join("  "));
        if li == 1 {
            mid = Some(l);
        }
    }

    let l = mid.unwrap();
    println!(
        "\n(a) layer 1: relative change of the layer's GDN readout (all {HV} heads) after the poke"
    );
    println!("heads  N   +1        +2        +4        +8        +16");
    let clean: Vec<Vec<Vec<f32>>> = (0..HV).map(|h| run(&l, h, 0, 0).0).collect();
    for heads in [1, 4] {
        for n in [1, 8, 64] {
            let poked: Vec<Vec<Vec<f32>>> = (0..heads).map(|h| run(&l, h, n, 0).0).collect();
            let row: Vec<String> = [1, 2, 4, 8, 16]
                .iter()
                .map(|&dt| {
                    let t = POKE_AT + dt - 1;
                    let num: f32 = (0..heads)
                        .map(|h| dist(&poked[h][t], &clean[h][t]).powi(2))
                        .sum::<f32>();
                    let den: f32 = (0..HV)
                        .map(|h| clean[h][t].iter().map(|x| x * x).sum::<f32>())
                        .sum();
                    format!("{:.2e}", (num / den).sqrt())
                })
                .collect();
            println!("{heads:5}  {n:2}  {}", row.join("  "));
        }
    }
}
