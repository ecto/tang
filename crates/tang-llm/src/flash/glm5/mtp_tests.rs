//! MTP draft-and-verify on the tiny fixture: the output must equal plain greedy decode for
//! any drafter (real MTP, junk, oracle), and the oracle must be accepted every time.

use super::engine::Glm5Engine;
use super::mtp::{generate, Drafter, Mtp, MtpDrafter};
use crate::gguf::Gguf;
use anyhow::Result;
use std::path::Path;

const TINY: &str = "/private/tmp/glm5-tiny/glm5-tiny-f32.gguf";

fn tiny() -> Option<String> {
    let p = std::env::var("TANG_GLM5_TINY").unwrap_or_else(|_| TINY.to_string());
    if Path::new(&p).exists() {
        Some(p)
    } else {
        eprintln!("SKIP: no tiny fixture at {p}");
        None
    }
}

const PROMPT: [u32; 6] = [3, 17, 42, 7, 99, 5];

fn plain(p: &str, n: usize) -> Vec<u32> {
    let mut e = Glm5Engine::open(Path::new(p)).unwrap();
    let mut cur = e.prefill(&PROMPT, 4).unwrap();
    let mut g = vec![cur];
    while g.len() < n {
        cur = e.step(cur).unwrap();
        g.push(cur);
    }
    g
}

/// Proposes fixed junk.
struct Junk(u32);
impl Drafter for Junk {
    fn observe(&mut self, _: &Gguf, _: &[f32], _: &[u32]) -> Result<()> {
        Ok(())
    }
    fn draft(&mut self, _: &Gguf, k: usize) -> Result<Vec<u32>> {
        self.0 = self.0.wrapping_mul(31).wrapping_add(7) % 200;
        Ok(vec![self.0; k])
    }
}

/// Knows the answer.
struct Oracle {
    want: Vec<u32>,
    done: usize,
}
impl Drafter for Oracle {
    fn observe(&mut self, _: &Gguf, _: &[f32], next: &[u32]) -> Result<()> {
        if self.done == 0 {
            self.done = 1; // the prefill's token
        } else {
            self.done += next.len();
        }
        Ok(())
    }
    fn draft(&mut self, _: &Gguf, k: usize) -> Result<Vec<u32>> {
        Ok(self.want[self.done..].iter().take(k).copied().collect())
    }
}

#[test]
fn mtp_any_drafter_matches_plain_greedy() {
    let _env = super::env_read();
    let Some(p) = tiny() else { return };
    let n = 12;
    let want = plain(&p, n);
    for k in [1, 2, 3] {
        let mut e = Glm5Engine::open(Path::new(&p)).unwrap();
        let mut dr = MtpDrafter::new(Mtp::load(&e).unwrap());
        let (g, st) = generate(&mut e, &mut dr, &PROMPT, n, 4, k).unwrap();
        eprint!("k={k} mtp\n{}", st.report());
        assert_eq!(g, want, "mtp k={k}");

        let mut e = Glm5Engine::open(Path::new(&p)).unwrap();
        let (g, st) = generate(&mut e, &mut Junk(1), &PROMPT, n, 4, k).unwrap();
        eprint!("k={k} junk\n{}", st.report());
        assert_eq!(g, want, "junk k={k}");

        let mut e = Glm5Engine::open(Path::new(&p)).unwrap();
        let mut o = Oracle {
            want: want.clone(),
            done: 0,
        };
        let (g, st) = generate(&mut e, &mut o, &PROMPT, n, 4, k).unwrap();
        eprint!("k={k} oracle\n{}", st.report());
        assert_eq!(g, want, "oracle k={k}");
        assert_eq!(st.rerun, 0, "oracle drafts must all be accepted");
        assert_eq!(st.verify, (n - 1).div_ceil(k + 1), "oracle k={k}: passes");
    }
}

/// The MTP cache rewinds exactly: drafting (chain cells + truncate) leaves the cache as if it
/// never ran, so the same committed cells give the same draft twice.
#[test]
fn mtp_draft_is_side_effect_free() {
    let _env = super::env_read();
    let Some(p) = tiny() else { return };
    let mut e = Glm5Engine::open(Path::new(&p)).unwrap();
    let lg = e.window(&PROMPT).unwrap();
    let nv = e.m.hp.n_vocab;
    let cur = super::engine::argmax(&lg[(PROMPT.len() - 1) * nv..]);
    let mut nexts = PROMPT[1..].to_vec();
    nexts.push(cur);
    let mut dr = MtpDrafter::new(Mtp::load(&e).unwrap());
    dr.observe(&e.m.g, &e.last_h, &nexts).unwrap();
    let len = dr.mtp.len();
    let a = dr.draft(&e.m.g, 3).unwrap();
    assert_eq!(dr.mtp.len(), len);
    let b = dr.draft(&e.m.g, 3).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 3);
}
