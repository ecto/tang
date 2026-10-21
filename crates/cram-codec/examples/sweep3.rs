//! `sweep3` — CPU-only TCG3 ratio sweep over the 19 bench fields (no GPU decode).
//!
//! ```text
//! sweep3 DATA_ROOT [--model3 pretrained:M.tcm] [--l3 8] [--targets p80] [--threads 16]
//! ```
//! Prints one JSON line per (field, target) and the geomean bpv ratio vs
//! serial tc-wavelet (from `DATA_ROOT/out/gpu_ratio.jsonl`) on all / orig / held.
use cram_codec::codec::ErrorBound;
use cram_codec::gpuwav3::{self, Gpu3Config, Model3};

fn opt(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| d.to_string())
}

const ORIG: [&str; 5] = [
    "miranda-density",
    "miranda-velocityx",
    "hurricane-TC",
    "hurricane-U",
    "nyx-temperature",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = std::path::PathBuf::from(&args[0]);
    let threads: usize = opt(&args, "--threads", "16").parse().unwrap();
    let lanes: usize = opt(&args, "--l3", "8").parse().unwrap();
    let m = opt(&args, "--model3", "static");
    let model = if m == "static" {
        Model3::Static
    } else if let Some(p) = m.strip_prefix("pretrained:") {
        let net = std::sync::Arc::new(
            cram_codec::learned::Net::from_bytes(&std::fs::read(p).expect("model"))
                .expect("parse"),
        );
        cram_codec::learned::register(net.clone());
        Model3::Pretrained(net)
    } else {
        panic!("model3 {m}")
    };
    let mut fields: Vec<(String, std::path::PathBuf, Vec<usize>)> = ORIG
        .iter()
        .map(|n| {
            (
                n.to_string(),
                root.join(format!("out/{n}-64x128x128.f32")),
                vec![64, 128, 128],
            )
        })
        .collect();
    let man = std::fs::read_to_string(root.join("data/heldout/manifest.json")).unwrap();
    // minimal manifest parse: [{"name": "...", "shape": [..]}, ...]
    for obj in man.split('{').skip(1) {
        let name = obj
            .split("\"name\"")
            .nth(1)
            .unwrap()
            .split('"')
            .nth(1)
            .unwrap()
            .to_string();
        let sh = obj
            .split("\"shape\"")
            .nth(1)
            .unwrap()
            .split('[')
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap();
        let shape: Vec<usize> = sh.split(',').map(|v| v.trim().parse().unwrap()).collect();
        fields.push((
            name.clone(),
            root.join(format!("data/heldout/{name}.f32")),
            shape,
        ));
    }
    let only = opt(&args, "--only", "");
    let ser: std::collections::HashMap<(String, String), f64> =
        std::fs::read_to_string(root.join("out/gpu_ratio.jsonl"))
            .unwrap()
            .lines()
            .filter(|l| l.contains("\"tc-wavelet\""))
            .map(|l| {
                let g = |k: &str| {
                    l.split(&format!("\"{k}\":"))
                        .nth(1)
                        .unwrap()
                        .split([',', '}'])
                        .next()
                        .unwrap()
                        .trim()
                        .trim_matches('"')
                        .to_string()
                };
                (
                    (g("field").replace("-64x128x128", ""), g("target")),
                    g("bpv").parse().unwrap(),
                )
            })
            .collect();
    let mut acc: std::collections::HashMap<(String, String), Vec<f64>> = Default::default();
    for t in opt(&args, "--targets", "p80").split(',') {
        for (name, path, dims) in &fields {
            if !only.is_empty() && !only.split(',').any(|o| name.contains(o)) {
                continue;
            }
            let b = std::fs::read(path).unwrap();
            let x: Vec<f32> = b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect();
            let range = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64
                - x.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64;
            let bound = if let Some(p) = t.strip_prefix('p') {
                ErrorBound::Psnr(p.parse().unwrap())
            } else {
                ErrorBound::MaxAbs(t[1..].parse::<f64>().unwrap() * range)
            };
            let cfg = Gpu3Config {
                lanes,
                compact: !args.iter().any(|a| a == "--nocompact"),
                axis: !args.iter().any(|a| a == "--noaxis"),
                model: model.clone(),
                ..Gpu3Config::new(bound)
            };
            let (s, _) = gpuwav3::encode3_f32(&x, dims, &cfg, threads);
            let bpv = s.len() as f64 * 8.0 / x.len() as f64;
            let r = ser.get(&(name.clone(), t.to_string())).map(|sb| bpv / sb);
            println!(
                "{{\"field\": \"{name}\", \"target\": \"{t}\", \"bpv\": {bpv}, \"vs_serial\": {}}}",
                r.map_or("null".into(), |v| format!("{v:.4}"))
            );
            if let Some(r) = r {
                let g = if ORIG.contains(&name.as_str()) {
                    "orig"
                } else {
                    "held"
                };
                acc.entry((t.to_string(), "all".into()))
                    .or_default()
                    .push(r);
                acc.entry((t.to_string(), g.into())).or_default().push(r);
            }
        }
    }
    let mut keys: Vec<_> = acc.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let v = &acc[&k];
        let gm = (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp();
        println!("GEOMEAN {} {} n={} {:.4}", k.0, k.1, v.len(), gm);
    }
}
