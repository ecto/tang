//! Build a small integer program (a 32-bit FNV-1a-style hash over a buffer
//! of bytes) and print the CPU result, the WGSL and the Rust lowering.
//!
//! ```text
//! cargo run -p tang-expr --example int_dialect
//! ```
use tang_expr::int::{Elem, IType, IntGraph};

fn main() {
    let mut g = IntGraph::new();
    let n = g.param(0, IType::U32);
    let data = g.buffer("data", Elem::U8);
    let basis = g.u32(0x811c_9dc5);
    let prime = g.u32(0x0100_0193);
    let h = g.for_range(n, &[basis], |g, i, s| {
        let b = g.load(data, i);
        let x = g.xor(s[0], b);
        vec![g.mul(x, prime)]
    });

    let bytes = b"tang";
    // u8 buffers are passed as u32 words, little-endian packed
    let word = u32::from_le_bytes(*bytes);
    let got = g.eval(&h, &[bytes.len() as u32], &[&[word]]);

    let mut want: u32 = 0x811c_9dc5;
    for &b in bytes {
        want = (want ^ b as u32).wrapping_mul(0x0100_0193);
    }
    assert_eq!(got[0], want);
    println!("hash(\"tang\") = {:#010x}", got[0]);
    println!("--- WGSL ---\n{}", g.to_wgsl_fn("fnv", &h));
    println!("--- Rust ---\n{}", g.to_rust("fnv", &h));
}
