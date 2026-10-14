//! Qwen3 token ids ↔ text, for `qwen_decode`'s prompts and outputs.
//!
//! ```text
//! qwen_tok <tokenizer.json> encode <text file>   ids, whitespace separated
//! qwen_tok <tokenizer.json> decode <ids file>    text
//! ```
//!
//! `mlx-community/Qwen3-8B-4bit` ships no `tokenizer.json`; every Qwen3
//! size shares one tokenizer (the 4B's works).
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tok = tokenizers::Tokenizer::from_file(&args[0]).expect("tokenizer.json");
    let input = std::fs::read_to_string(&args[2]).expect("input file");
    match args[1].as_str() {
        "encode" => {
            let enc = tok.encode(input.as_str(), false).expect("encode");
            let ids: Vec<String> = enc.get_ids().iter().map(|i| i.to_string()).collect();
            println!("{}", ids.join(" "));
        }
        "decode" => {
            let ids: Vec<u32> = input
                .split_whitespace()
                .map(|t| t.parse().unwrap())
                .collect();
            println!("{}", tok.decode(&ids, false).expect("decode"));
        }
        _ => eprintln!("usage: qwen_tok <tokenizer.json> encode|decode <file>"),
    }
}
