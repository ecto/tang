//! Compare every DiT boundary to the official fp32 diffusers fixture.
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};
use tang_compute::{ComputeDevice, CpuDevice};
use tang_llm::{dit::DiT, Dtype};
#[derive(Deserialize)]
struct Case {
    height: usize,
    width: usize,
    #[serde(default)]
    caption_tokens: usize,
    #[serde(default)]
    t: f32,
    latent: Vec<f32>,
    #[serde(default)]
    caption: Vec<f32>,
    traces: BTreeMap<String, Vec<f32>>,
    output: Vec<f32>,
}
fn compare(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) -> Result<()> {
    ensure!(
        actual.len() == expected.len(),
        "{name}: shape {} != {}",
        actual.len(),
        expected.len()
    );
    let mut max = 0f32;
    let mut worst = 0usize;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        ensure!(a.is_finite(), "{name}: nonfinite value at {i}");
        let err = (a - e).abs();
        if err > max {
            max = err;
            worst = i;
        }
    }
    println!("{name}: max abs {max:.8} at {worst}");
    ensure!(
        max < tolerance,
        "{name}: actual {} expected {} at {worst}",
        actual[worst],
        expected[worst]
    );
    Ok(())
}
fn run<D: ComputeDevice>(dev: D, root: &Path, file: &Path, tolerance: f32) -> Result<()> {
    let vae = root.join("vae-fixture").exists();
    let model = if vae {
        None
    } else {
        Some(DiT::load(&dev, root, Dtype::F32)?)
    };
    let case: Case = serde_json::from_slice(&std::fs::read(file)?)?;
    let mut seen = BTreeMap::new();
    let mut trace = |name: &str, tensor: &D::Buffer| {
        seen.insert(name.to_owned(), dev.download(tensor));
    };
    let out = if let Some(model) = model {
        let cap = dev.upload(&case.caption);
        model.forward_trace(
            &dev,
            &case.latent,
            case.height,
            case.width,
            &cap,
            case.caption_tokens,
            case.t,
            &mut trace,
        )?
    } else {
        tang_llm::vae::Vae::load(&dev, root)?.decode_trace(
            &dev,
            &case.latent,
            case.height,
            case.width,
            &mut trace,
        )?
    };
    for (name, expected) in case.traces {
        compare(
            &name,
            seen.get(&name).context("missing trace")?,
            &expected,
            tolerance,
        )?;
    }
    compare("output", &out, &case.output, tolerance)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let root = Path::new(args.get(1).context("fixture directory required")?);
    let metal = args.iter().any(|a| a == "--metal");
    let files = if root.join("vae-fixture").exists() {
        ["case-4.json", "case-32.json"]
    } else {
        ["case-8.json", "case-32.json"]
    };
    for file in files {
        println!("{file}:");
        if metal {
            #[cfg(feature = "metal")]
            run(
                tang_compute::MetalDevice::new().context("Metal unavailable")?,
                root,
                &root.join(file),
                0.001,
            )?;
            #[cfg(not(feature = "metal"))]
            anyhow::bail!("rebuild with --features metal");
        } else {
            run(CpuDevice::new(), root, &root.join(file), 0.0001)?;
        }
    }
    Ok(())
}
